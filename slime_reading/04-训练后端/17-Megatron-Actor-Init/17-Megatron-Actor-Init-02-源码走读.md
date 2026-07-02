---
type: batch-doc
module: 17-Megatron-Actor-Init
batch: "17"
doc_type: walkthrough
title: "Megatron Actor 初始化 · 源码走读"
tags:
  - slime/batch/17
  - slime/module/megatron-actor-init
  - slime/doc/walkthrough
updated: 2026-07-02
---

# Megatron Actor 初始化 · 源码走读

> 按 **调用顺序** 精读 `MegatronTrainRayActor.init` → `initialize.init` → `sleep` / `wake_up`。

---

## 1. init 入口与 debug 短路

**Explain：** 计时器 `train_wait` 在 init 完成前计入等待；debug 模式直接返回，避免启动 Megatron。

**Code：**

```python
# 来源：slime/backends/megatron_utils/actor.py L47-L57
@with_defer(lambda: Timer().start("train_wait"))
def init(
    self,
    args: Namespace,
    role: str,
    with_ref: bool = False,
    with_opd_teacher: bool = False,
) -> int | None:
    if args.debug_rollout_only:
        self.args = args
        return 0
```

**Comment：**

- `debug_rollout_only` 时 `train()` / `save_model()` / `update_weights()` 也会 early return
- 返回值 `0` 使 `args.start_rollout_id` 默认为 0（若未显式指定）

---

## 2. 分布式补丁与 Megatron init

**Explain：** 正常路径先 patch torch dist，再父类 NCCL，再 Megatron 并行拓扑。

**Code：**

```python
# 来源：slime/backends/megatron_utils/actor.py L59-L65
monkey_patch_torch_dist()
super().init(args, role, with_ref, with_opd_teacher)

init(args)

if is_megatron_main_rank():
    init_tracking(args, primary=False, role=role)

self.prof = TrainProfiler(args)
```

**Comment：**

- `monkey_patch_torch_dist` 来自 `reloadable_process_group`，支持 offload 后重建 process group
- `TrainProfiler` 在 init 末调用 `on_init_end()` 关闭 init 阶段 profiling

---

## 3. HF 配置与 offload 显存边距

**Explain：** tokenizer 串行加载后，若 offload 训练则设置 `torch_memory_saver` 保留边距。

**Code：**

```python
# 来源：slime/backends/megatron_utils/actor.py L78-L85
if args.offload_train:
    if (x := args.train_memory_margin_bytes) > 0:
        logger.info(f"Set torch_memory_saver.memory_margin_bytes to {x}")
        torch_memory_saver.memory_margin_bytes = x

self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id = initialize_model_and_optimizer(
    args, role
)
```

**Comment：**

- `initialize_model_and_optimizer` 详见 [[18-Model-Init]]；此处只关心返回的 `loaded_rollout_id`
- `train_memory_margin_bytes` 在 `debug_rollout_only` 下被 arguments 强制为 0

---

## 4. 训练并行配置字典

**Explain：** 将 DP/CP/VPP/microbatch 元信息缓存到 actor，供后续 data pipeline 与调度使用。

**Code：**

```python
# 来源：slime/backends/megatron_utils/actor.py L87-L101
vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1
if vpp_size > 1:
    from megatron.core.utils import get_model_config

    microbatch_group_size_per_vp_stage = get_model_config(self.model[0]).microbatch_group_size_per_vp_stage
else:
    microbatch_group_size_per_vp_stage = 1
self.train_parallel_config = {
    "dp_size": mpu.get_data_parallel_world_size(with_context_parallel=False),
    "cp_size": mpu.get_context_parallel_world_size(),
    "vpp_size": vpp_size,
    "microbatch_group_size_per_vp_stage": microbatch_group_size_per_vp_stage,
}

start_rollout_id = loaded_rollout_id + 1
```

**Comment：**

- VPP>1 时从 model config 读取 microbatch 分组大小
- `start_rollout_id` 表示 **下一条** rollout 的 id（checkpoint 已完成的步数 +1）

---

## 5. Actor 权重备份与辅助模型

**Explain：** 仅 actor 角色创建 `TensorBackuper` 并加载 ref/teacher/old_actor；critic 在此之前 return。

**Code：**

```python
# 来源：slime/backends/megatron_utils/actor.py L108-L131
self.weights_backuper = TensorBackuper.create(
    source_getter=lambda: named_params_and_buffers(
        self.args,
        self.model,
        convert_to_global_name=args.megatron_to_hf_mode == "raw",
    ),
    single_tag=None,
)
self._active_model_tag: str | None = "actor"
self.weights_backuper.backup("actor")

if with_ref:
    self.load_other_checkpoint("ref", args.ref_load)

if with_opd_teacher:
    self.load_other_checkpoint("teacher", args.opd_teacher_load)

if self.args.keep_old_actor:
    self.load_other_checkpoint("old_actor", args.load)
    if args.update_weights_interval == 1:
        self.weights_backuper.backup("rollout_actor")
```

**Comment：**

- `load_other_checkpoint` 临时改 `args.load` 等字段，调用 Megatron `load_checkpoint` 后 `backup(tag)`
- `keep_old_actor` + `update_weights_interval==1` 维护 rollout_actor 队列（PPO 旧策略）

---

## 6. weight_updater 策略选型

**Explain：** 根据 colocate、delta、transport 三轴选择 UpdateWeight 实现类。

**Code：**

```python
# 来源：slime/backends/megatron_utils/actor.py L133-L168
if self.args.vocab_size is None:
    hf_vocab = getattr(self.hf_config, "vocab_size", None)
    self.args.vocab_size = hf_vocab if hf_vocab is not None else self.tokenizer.vocab_size

if self.args.colocate:
    assert (
        self.args.update_weight_mode == "full"
    ), "--update-weight-mode=delta is not supported with --colocate"
    update_weight_cls = UpdateWeightFromTensor
elif self.args.update_weight_mode == "delta":
    assert (
        self.args.update_weight_transport == "disk"
    ), "--update-weight-mode=delta requires --update-weight-transport=disk"
    from .update_weight.update_weight_from_disk_delta import UpdateWeightFromDiskDelta

    update_weight_cls = UpdateWeightFromDiskDelta
else:
    assert self.args.update_weight_mode == "full"
    if self.args.update_weight_transport == "disk":
        update_weight_cls = UpdateWeightFromDisk
    else:
        update_weight_cls = UpdateWeightFromDistributed

self.weight_updater = update_weight_cls(
    self.args,
    self.model,
    weights_getter=lambda: self.weights_backuper.get("actor"),
    model_name=type(self.hf_config).__name__.lower() if self.args.model_name is None else self.args.model_name,
    quantization_config=getattr(self.hf_config, "quantization_config", None),
)
```

**Comment：**

- colocate 模式训练与推理共享 GPU，必须 tensor 直传（见 [[24-WeightSync-Dist]]）
- delta 模式仅 disk transport，引擎侧 reload HF checkpoint

---

## 7. init 收尾：清显存、offload sleep、返回值

**Explain：** actor init 完成后清缓存；offload 时切回 actor tag 并 sleep；返回 `start_rollout_id`。

**Code：**

```python
# 来源：slime/backends/megatron_utils/actor.py L170-L188
clear_memory()

if self.args.offload_train:
    self._switch_model("actor")
    self.sleep()

self.rollout_engines = None

self.rollout_data_postprocess = None
if self.args.rollout_data_postprocess_path is not None:
    from slime.utils.misc import load_function

    self.rollout_data_postprocess = load_function(self.args.rollout_data_postprocess_path)

self.prof.on_init_end()

return start_rollout_id
```

**Comment：**

- init 末尾 sleep 是为 **立刻** 把 GPU 让给 Rollout（与 train 循环内 sleep 同理）
- `rollout_data_postprocess` 是可选 hook，在 [[19-Train-Step]] `train_actor` 中调用

---

## 8. sleep：释放训练栈

**Explain：** 断言 `offload_train`；断开 rollout NCCL（特定 critic 场景）；销毁 process group；暂停 memory saver。

**Code：**

```python
# 来源：slime/backends/megatron_utils/actor.py L190-L207
@timer
def sleep(self) -> None:
    assert self.args.offload_train

    clear_memory(clear_host_memory=True)
    print_memory("before offload model")
    if (
        self.role == "actor"
        and self.args.use_critic
        and not self.args.colocate
        and hasattr(self.weight_updater, "disconnect_rollout_engines")
    ):
        self.weight_updater.disconnect_rollout_engines()
    destroy_process_groups()

    torch_memory_saver.pause()

    print_memory("after offload model")
```

**Comment：**

- `clear_host_memory=True` 比 init 中的 `clear_memory()` 更激进
- critic + 非 colocate 时 actor 需断开与 SGLang 的 weight sync 连接，避免 NCCL 组冲突

---

## 9. wake_up：恢复训练栈

**Explain：** resume memory saver → 清显存 → 重建 process group → actor 恢复权重 tag。

**Code：**

```python
# 来源：slime/backends/megatron_utils/actor.py L209-L220
@timer
def wake_up(self) -> None:
    assert self.args.offload_train
    print_memory("before wake_up model")

    torch_memory_saver.resume()

    clear_memory()
    reload_process_groups()
    if self.role == "actor":
        self._switch_model("actor")
    print_memory("after wake_up model")
```

**Comment：**

- `train()` / `save_model()` / 部分 `update_weights` 路径在操作前调用 `wake_up()`
- `_switch_model("actor")` 从 `weights_backuper` 恢复 GPU 上的 actor 权重

---

## 10. initialize._initialize_distributed

**Explain：** 将 CLI 中的 TP/PP/CP/EP 等参数传入 Megatron Core 的 `initialize_model_parallel`。

**Code：**

```python
# 来源：slime/backends/megatron_utils/initialize.py L33-L53
def _initialize_distributed(args, get_embedding_ranks=None, get_position_embedding_ranks=None):
    mpu.initialize_model_parallel(
        args.tensor_model_parallel_size,
        args.pipeline_model_parallel_size,
        args.virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_comm_backend=args.pipeline_model_parallel_comm_backend,
        context_parallel_size=args.context_parallel_size,
        hierarchical_context_parallel_sizes=args.hierarchical_context_parallel_sizes,
        expert_model_parallel_size=args.expert_model_parallel_size,
        num_distributed_optimizer_instances=args.num_distributed_optimizer_instances,
        expert_tensor_parallel_size=args.expert_tensor_parallel_size,
        distributed_timeout_minutes=args.distributed_timeout_minutes,
        nccl_communicator_config_path=args.nccl_communicator_config_path,
        order="tp-cp-ep-dp-pp" if not args.use_tp_pp_dp_mapping else "tp-cp-ep-pp-dp",
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
        create_gloo_process_groups=args.enable_gloo_process_groups,
    )
```

**Comment：**

- `order` 决定 rank 到并行维度的映射；MoE 大模型常用默认 `tp-cp-ep-dp-pp`
- 与 `TrainRayActor.init` 中的 world-level NCCL 配合，形成完整通信拓扑

---

## 11. initialize._set_random_seed

**Explain：** 按 PP rank、可选 DP rank 偏移 seed，保证各 stage 可复现又不完全相同。

**Code：**

```python
# 来源：slime/backends/megatron_utils/initialize.py L14-L30
def _set_random_seed(
    seed_: int,
    data_parallel_random_init: bool = False,
    te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
):
    seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())
    if data_parallel_random_init:
        seed = seed + (10 * mpu.get_data_parallel_rank(with_context_parallel=False))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tensor_parallel.model_parallel_cuda_manual_seed(seed, te_rng_tracker, inference_rng_tracker, use_cudagraphable_rng)
```

**Comment：**

- TransformerEngine 与 inference RNG tracker 由 args 开关控制
- `deterministic_mode` 分支在 `init()` 末尾额外设置 cuDNN 确定性

---

## 12. initialize 可选扩展

**Explain：** TP comm overlap 与自定义 init hook。

**Code：**

```python
# 来源：slime/backends/megatron_utils/initialize.py L88-L104
if args.deterministic_mode:
    if args.rank == 0:
        logger.info("> running in deterministic mode")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)

if args.tp_comm_overlap:
    from megatron.training.initialize import _initialize_tp_communicators

    _initialize_tp_communicators()

if getattr(args, "custom_megatron_init_path", None):
    from slime.utils.misc import load_function

    custom_init = load_function(args.custom_megatron_init_path)
    custom_init(args)
```

**Comment：**

- 自定义 hook 在 Megatron 标准 init 之后执行，可注册额外 buffer 或 patch
- 与 [[28-Customization]] 中的 `load_function` 模式一致

---

## 13. debug_rollout_only 在 train / update_weights 的联动

**Explain：** 同一 flag 在训练与推权路径跳过 Megatron 逻辑。

**Code：**

```python
# 来源：slime/backends/megatron_utils/actor.py L380-L382, L558-L560, L583-L585
def train(self, rollout_id: int, rollout_data_ref: Box, external_data=None):
    if self.args.debug_rollout_only:
        return None

def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
    if self.args.debug_rollout_only:
        return

def update_weights(self) -> None:
    if self.args.debug_train_only or self.args.debug_rollout_only:
        return
```

**Comment：**

- 与 `debug_train_only` 互斥（arguments 解析阶段 assert）
- 纯 Rollout 调试时不应触发 Megatron NCCL 或 checkpoint IO

---

## 走读小结

| 步骤 | 函数/代码块 | 产出状态 |
|------|-------------|----------|
| 1 | debug 短路 | 仅 args |
| 2 | super().init + initialize.init | dist + mpu |
| 3 | HF config/tokenizer | self.hf_config, self.tokenizer |
| 4 | initialize_model_and_optimizer | model, optim, loaded_rollout_id |
| 5 | critic return 或 weights_backuper | 多 tag 权重 |
| 6 | weight_updater | 推权重策略对象 |
| 7 | clear + sleep | GPU 可给 Rollout |
| 8 | return start_rollout_id | 全局 rollout 起点 |
