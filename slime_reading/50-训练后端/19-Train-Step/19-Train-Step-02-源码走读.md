---
type: batch-doc
module: 19-Train-Step
batch: "19"
doc_type: walkthrough
title: "Train Step · 源码走读"
tags:
  - slime/batch/19
  - slime/module/train-step
  - slime/doc/walkthrough
updated: 2026-07-05
---

# Train Step · 源码走读

## 1. Ray 分发与 Actor 入口

### 1.1 `RayTrainGroup.async_train` 对每个 rank 发起远程 train

问题与约束：
- Megatron 每个 rank 都是一个 Ray actor，单次 rollout train 要广播到所有 rank。
- critic 训练可能需要每个 worker 接收不同的 `external_data`。
- actor 和 critic 的返回值语义不同：critic last PP stage 返回 values，actor 不需要返回训练结果。

设计选择：
- `async_train` 返回每个 actor 的 `train.remote(...)` ObjectRef。
- `external_data` 如果是 list，要求长度等于 actor handlers 数量，并按 worker zip 传入。
- 非 list 的 `external_data` 作为同一个对象广播给所有 worker。

Explain：
RayTrainGroup 只做分发，不等待训练完成，也不解释返回值。调用方可以对返回 refs 做并发等待或作为 critic→actor 的数据依赖传递。

来源：slime/ray/actor_group.py L131-L149

Code：

```python
def async_train(self, rollout_id, rollout_data_ref, external_data=None):
    """Do one rollout training. Returns a list of Ray refs (one per worker)."""
    if isinstance(external_data, list):
        assert len(external_data) == len(self._actor_handlers)
        return [
            actor.train.remote(rollout_id, rollout_data_ref, external_data=ed)
            for actor, ed in zip(self._actor_handlers, external_data, strict=False)
        ]
    return [
        actor.train.remote(rollout_id, rollout_data_ref, external_data=external_data)
        for actor in self._actor_handlers
    ]
```

代码逻辑：
- 检查 external_data 是否按 worker 列表传入。
- 列表模式下校验长度。
- 每个 actor 调用 `train.remote`，带上对应 external data。
- 广播模式下所有 actor 传同一个 external data。
- 返回 Ray refs 列表。

为什么这样写：
- 训练入口保持异步，允许上层同时调度多个 rank。
- critic 结果可按 worker 精细注入 actor，也可广播一个 dict。
- Ray ObjectRef 由外层管理，RayTrainGroup 不耦合 critic/actor 的聚合逻辑。

不变量与失败模式：
- list 形式的 external_data 长度必须等于 worker 数。
- `_actor_handlers` 中每个 actor 都必须暴露 `train` remote 方法。
- 返回 ref 的异常会在调用方 `ray.get` 时暴露。

Comment：
这是从集群控制面进入每个 Megatron rank 的边界。

### 1.2 `MegatronTrainRayActor.train` 做 offload 唤醒、数据预处理和角色分派

问题与约束：
- 同一个 Ray actor 类服务 actor 和 critic 两种训练角色。
- offload train 模式下，训练前后需要恢复/释放显存占用。
- rollout data 先在 object store/CPU 侧，进入训练前要按 DP rank 预处理。

设计选择：
- debug rollout-only 直接返回。
- `offload_train` 时先 `wake_up()`，结束后删除 rollout_data 并 `sleep()`。
- 用 timer 包住 `_get_rollout_data`。
- 按 `self.role` 分派到 `train_critic` 或 `train_actor`。

Explain：
`train` 是每个 rank 的统一入口。它不直接写训练逻辑，而是处理生命周期和角色分发：准备数据、唤醒模型、调用角色训练、必要时再 offload。

来源：slime/backends/megatron_utils/actor.py L380-L400

Code：

```python
def train(self, rollout_id: int, rollout_data_ref: Box, external_data=None):
    if self.args.debug_rollout_only:
        return None

    if self.args.offload_train:
        self.wake_up()

    with timer("data_preprocess"):
        rollout_data = self._get_rollout_data(rollout_data_ref)

    if self.role == "critic":
        result = self.train_critic(rollout_id, rollout_data)
    else:
        self.train_actor(rollout_id, rollout_data, external_data=external_data)
        result = None

    if self.args.offload_train:
        del rollout_data
        self.sleep()

    return result
```

代码逻辑：
- debug rollout-only 跳过训练。
- offload 模式先恢复模型状态。
- 将 rollout_data_ref 转成当前 rank 可用的 rollout_data。
- critic 调 `train_critic` 并保存返回值。
- actor 调 `train_actor`，返回 None。
- offload 模式释放 rollout_data 并 sleep。
- 返回角色结果。

为什么这样写：
- actor/critic 共享数据预处理和 offload 生命周期，减少重复。
- 角色分支集中在一个入口，RayTrainGroup 不需要知道内部细节。
- offload 的 wake/sleep 成对出现，支持训练和 rollout 复用 GPU。

不变量与失败模式：
- `self.role` 必须是 actor 或 critic 语义之一。
- `_get_rollout_data` 失败会阻止训练。
- offload 模式中 `wake_up/sleep` 必须和模型状态一致，否则后续训练或 rollout 会读到错误设备状态。

Comment：
这段是每个 rank 的训练事务边界。

### 1.3 `_get_rollout_data` 按 DP rank 切片并预搬 GPU

问题与约束：
- rollout data 是全局 batch，需要按 data parallel rank 切片。
- 训练 hot path 要减少首次 forward 前的数据搬运延迟。
- context parallel 场景下 logprob 需要按 total/response length 切片。
- 多模态训练输入也可能包含 tensor。

设计选择：
- 调 `process_rollout_data(args, rollout_data_ref, dp_rank, dp_world_size)` 获取本 rank 数据。
- 将 tokens、loss_masks、rollout_mask_sums、多模态 tensor 预先搬到当前 CUDA device。
- 对 `rollout_log_probs` 和 `teacher_log_probs` 调 `slice_log_prob_with_cp` 后转 float32 GPU tensor。

Explain：
这一步把 object store 中的 rollout batch 变成当前 Megatron rank 可训练的数据结构。DP 切片、device 搬运和 CP logprob 对齐都在这里完成。

来源：slime/backends/megatron_utils/actor.py L222-L276

Code：

```python
def _get_rollout_data(self, rollout_data_ref: Box) -> RolloutBatch:
    rollout_data = process_rollout_data(
        self.args,
        rollout_data_ref,
        mpu.get_data_parallel_rank(with_context_parallel=False),
        mpu.get_data_parallel_world_size(with_context_parallel=False),
    )
    device = torch.cuda.current_device()
    rollout_data["tokens"] = [
        t.to(device=device, dtype=torch.long, non_blocking=True) for t in rollout_data["tokens"]
    ]
    rollout_data["loss_masks"] = [
        t.to(device=device, dtype=torch.int, non_blocking=True) for t in rollout_data["loss_masks"]
    ]
    if "rollout_mask_sums" in rollout_data:
        rollout_data["rollout_mask_sums"] = rollout_data["rollout_mask_sums"].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
    if "multimodal_train_inputs" in rollout_data:
        rollout_data["multimodal_train_inputs"] = [
            (
                {
                    key: value.to(device=device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                    for key, value in mm_dict.items()
                }
                if mm_dict is not None
                else None
            )
            for mm_dict in rollout_data["multimodal_train_inputs"]
        ]

    for key in ["rollout_log_probs", "teacher_log_probs"]:
        if key not in rollout_data:
            continue
        rollout_data[key] = [
            slice_log_prob_with_cp(log_prob, total_length, response_length).to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            for log_prob, total_length, response_length in zip(
                rollout_data[key],
                rollout_data["total_lengths"],
                rollout_data["response_lengths"],
                strict=False,
            )
        ]
    return rollout_data
```

代码逻辑：
- 用非 CP 的 DP rank/world size 切分 rollout data。
- 取当前 CUDA device。
- tokens 转 long GPU tensor。
- loss masks 转 int GPU tensor。
- 可选 rollout mask sums 转 float32 GPU tensor。
- 多模态 dict 中的 tensor 逐项搬到 GPU。
- logprob 类字段按 CP 切片后转 float32 GPU tensor。
- 返回处理后的 rollout_data。

为什么这样写：
- DP 切片必须先发生，否则每个 rank 会训练全量数据。
- 提前搬 GPU 避免每个 microbatch 临时搬运。
- logprob 与 response token 对齐后再进 loss，避免 CP 切分下长度错配。

不变量与失败模式：
- `process_rollout_data` 必须生成 tokens/loss_masks/total_lengths/response_lengths 等字段。
- logprob list 与 total/response length list 必须按样本对齐。
- 多模态输入中的非 tensor 值会原样保留。

Comment：
训练 step 的很多正确性问题其实会在这里暴露：DP 切片、长度和 device 必须一致。

## 2. Critic 与 Actor 分支

### 2.1 `train_critic` 先 forward values，再训练 value loss

问题与约束：
- critic 训练需要当前 value 预测来计算 advantages/returns。
- value loss 训练使用同一份 rollout data。
- actor 可能需要 critic 产出的 values 作为 external data。

设计选择：
- 构造 data iterator，并读取 num_microbatches/global_batch_sizes。
- 先调用 `forward_only(get_values, ...)`，将 values 写回 rollout_data。
- 调 `compute_advantages_and_returns`。
- 将 `args.loss_type` 设置为 `value_loss` 后调用 model-level `train`。
- pipeline last stage 把 values 转 CPU 返回。

Explain：
critic 路径是“先评估当前 value，再训练 critic”。返回 CPU values 让 actor 训练可以复用旧 value 计算优势或 policy loss 相关项。

来源：slime/backends/megatron_utils/actor.py L402-L428

Code：

```python
def train_critic(self, rollout_id: int, rollout_data: RolloutBatch):
    """Train critic and return CPU values (used as old-values for the next actor train)."""
    data_iterator = get_data_iterator(rollout_data)
    num_microbatches = rollout_data["num_microbatches"]
    global_batch_sizes = rollout_data["global_batch_sizes"]

    rollout_data.update(forward_only(get_values, self.args, self.model, data_iterator, num_microbatches))

    compute_advantages_and_returns(self.args, rollout_data)

    self.args.loss_type = "value_loss"
    train(
        rollout_id,
        self.model,
        self.optimizer,
        self.opt_param_scheduler,
        data_iterator,
        num_microbatches,
        global_batch_sizes,
    )

    if mpu.is_pipeline_last_stage() and "values" in rollout_data:
        from slime.backends.megatron_utils.data import tensors_to_cpu

        return {"values": tensors_to_cpu(rollout_data["values"])}
    return {}
```

代码逻辑：
- 从 rollout_data 构造 data iterator。
- 取出 microbatch 和 global batch size 配置。
- forward-only 计算 values 并更新 rollout_data。
- 计算 advantages/returns。
- 设置 value loss。
- 调用 Megatron train loop。
- last PP stage 返回 CPU values。
- 非 last stage 返回空 dict。

为什么这样写：
- value 预测必须在 value loss 训练前得到，并参与 advantage/return 计算。
- `loss_type` 是下游 loss_function 的分支开关。
- 只有 last PP stage 拥有完整输出，其他 stage 不应返回 values。

不变量与失败模式：
- `forward_only(get_values, ...)` 必须产出 values 字段。
- `num_microbatches` 与 `global_batch_sizes` 长度要在 model.train 中一致。
- 如果 pipeline last stage 没有 values，返回空 dict，actor 侧不能假设一定有 external values。

Comment：
critic 分支的返回值是后续 actor 分支的数据依赖。

### 2.2 `train_actor` 先补齐 logprob/values/advantage，再进入 policy train

问题与约束：
- policy loss 可能需要 ref log_probs、teacher log_probs、old actor log_probs、critic values、advantages/returns 等多种字段。
- 某些字段要在整批 rollout 上计算，不能放到 microbatch loss 中临时计算。
- routing replay 需要在 forward 阶段设置环境变量。

设计选择：
- 在 `compute_advantages_and_returns` 前按需要切换到 ref、teacher、old_actor/actor 做 `compute_log_prob`。
- 当 use_critic 且 external_data 可用时，在 last PP stage 把 values 搬回 GPU 写入 rollout_data。
- 切回 actor 权重后计算 advantages/returns。
- 允许在某些条件下复用 rollout logprobs，跳过额外 actor logprob forward。

Explain：
actor 训练前半段是 rollout_data 的补全状态机。它先确保 loss 所需的统计量都在整批级别可用，再进入真正的 backward。

来源：slime/backends/megatron_utils/actor.py L430-L509

Code：

```python
def train_actor(self, rollout_id: int, rollout_data: RolloutBatch, external_data=None) -> None:
    data_iterator = get_data_iterator(rollout_data)
    num_microbatches = rollout_data["num_microbatches"]
    global_batch_sizes = rollout_data["global_batch_sizes"]

    if self.args.use_rollout_routing_replay:
        self.fill_routing_replay(data_iterator, num_microbatches, rollout_data)

    with inverse_timer("train_wait"), timer("train"):
        if self.args.compute_advantages_and_returns:
            if "ref" in self.weights_backuper.backup_tags:
                if self.args.use_routing_replay:
                    os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                self._switch_model("ref")
                rollout_data.update(
                    self.compute_log_prob(
                        data_iterator,
                        num_microbatches,
                        store_prefix="ref_",
                    )
                )

            if "teacher" in self.weights_backuper.backup_tags:
                if self.args.use_routing_replay:
                    os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                self._switch_model("teacher")
                rollout_data.update(
                    self.compute_log_prob(
                        data_iterator,
                        num_microbatches,
                        store_prefix="teacher_",
                    )
                )

            self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
            can_reuse_log_probs_in_loss = (
                len(num_microbatches) == 1
                and self.args.loss_type == "policy_loss"
                and self.args.kl_coef == 0
                and not self.args.use_rollout_logprobs
                and not self.args.get_mismatch_metrics
                and not self.args.use_critic
                and not self.args.keep_old_actor
                and not self.args.use_routing_replay
                and self.args.advantage_estimator != "gspo"
            )
            if (
                not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics
            ) and not can_reuse_log_probs_in_loss:
                rollout_data.update(
                    self.compute_log_prob(
                        data_iterator,
                        num_microbatches,
                        store_prefix="",
                    )
                )

            if self.args.use_critic:
                if external_data is not None and mpu.is_pipeline_last_stage():
                    values = external_data.get("values")
                    if values is not None:
                        from slime.backends.megatron_utils.data import tensors_to_gpu

                        rollout_data["values"] = tensors_to_gpu(values)
            if self._active_model_tag != "actor":
                self._switch_model("actor")

            compute_advantages_and_returns(self.args, rollout_data)
```

代码逻辑：
- 构造 data iterator 并读取 batch 切分信息。
- 可选填充 routing replay 数据。
- ref 权重存在时切到 ref 并计算 ref logprob。
- teacher 权重存在时切到 teacher 并计算 teacher logprob。
- 切到 old actor 或 actor。
- 根据 loss 配置判断是否可以复用 logprob。
- 必要时计算 actor logprob。
- critic values 从 external data 搬回 GPU。
- 确保最终 active model 是 actor。
- 计算 advantages/returns。

为什么这样写：
- advantage 归一化和 return 计算通常需要整批数据，必须在 microbatch 训练前完成。
- 多个模型权重共享同一个 actor 进程，切换 model tag 比启动多个 actor 更节省资源。
- logprob 复用条件严格限制，避免在 KL、critic、routing replay 等场景下使用错误统计量。

不变量与失败模式：
- `weights_backuper.backup_tags` 中的 tag 必须可切换。
- external values 只在 pipeline last stage 注入。
- 若最终没有切回 actor，后续 backward 会训练错模型；源码显式检查并切回。

Comment：
actor 分支前半段的核心是补齐 loss 需要的整批统计量。

### 2.3 `train_actor` 后半执行 backward、备份 actor 并按间隔更新 ref

问题与约束：
- rollout_data 可能有自定义后处理和调试保存。
- routing replay 的 backward 阶段需要设置环境变量。
- 训练后 CPU 侧 actor 权重备份必须更新，否则后续权重同步可能读到旧参数。
- ref model 可能按固定 rollout 间隔更新。

设计选择：
- 训练前先执行可选 rollout_data_postprocess 和 log_rollout_data。
- 在 `actor_train` timer 内调用 model-level `train`。
- profiler step 后保存 debug train data。
- routing replay 清理全局状态。
- 调 `weights_backuper.backup("actor")`，必要时再 backup `"ref"`。
- 记录 perf metrics。

Explain：
actor 后半段才是真正的优化器更新。更新完成后，它立即刷新 actor 备份和可选 ref 备份，保证后续 update_weights 或 ref logprob 使用最新状态。

来源：slime/backends/megatron_utils/actor.py L511-L555

Code：

```python
if self.rollout_data_postprocess is not None:
    self.rollout_data_postprocess(self.args, rollout_id, rollout_data)

log_rollout_data(
    rollout_id,
    self.args,
    rollout_data,
)

if self.args.use_routing_replay:
    os.environ["ROUTING_REPLAY_STAGE"] = "replay_backward"
with timer("actor_train"):
    train(
        rollout_id,
        self.model,
        self.optimizer,
        self.opt_param_scheduler,
        data_iterator,
        num_microbatches,
        global_batch_sizes,
    )

self.prof.step(rollout_id=rollout_id)

train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

if self.args.use_routing_replay:
    RoutingReplay.clear_all()

self.weights_backuper.backup("actor")

if (
    self.args.ref_update_interval is not None
    and (rollout_id + 1) % self.args.ref_update_interval == 0
    and "ref" in self.weights_backuper.backup_tags
):
    with timer("ref_model_update"):
        if is_megatron_main_rank():
            logger.info(f"Updating ref model at rollout_id {rollout_id}")
        self.weights_backuper.backup("ref")

log_perf_data(rollout_id, self.args, extra_metrics=self.weight_updater.pop_metrics())
```

代码逻辑：
- 可选执行 rollout data 后处理。
- 记录 rollout data 日志。
- routing replay 进入 backward 阶段。
- 调用 Megatron train loop。
- 更新 profiler。
- 保存 debug train data。
- 清理 routing replay。
- 备份 actor 权重。
- 满足间隔条件时更新 ref 备份。
- 记录性能指标。

为什么这样写：
- 后处理和日志要在训练前看到完整 rollout_data。
- actor 权重备份必须紧跟 optimizer 更新。
- ref 更新按 rollout id 间隔发生，和训练步数解耦。
- routing replay 状态是全局的，训练后要清理避免污染下一轮。

不变量与失败模式：
- `train` 成功返回后才能备份 actor。
- ref backup tag 不存在时不会更新 ref。
- `weight_updater.pop_metrics()` 会消费指标，后续重复读取拿不到同一批 metrics。

Comment：
这段是 actor train step 的收尾，也是权重同步正确性的关键。

## 3. Megatron train loop

### 3.1 `model.train` 准备 iterator、DDP overlap 配置并逐 step 调 `train_one_step`

问题与约束：
- dynamic batch 可能把一次 rollout 拆成多个 step，每个 step 有自己的 microbatch 数和 global batch size。
- data iterator 需要在训练前重置。
- DDP overlap grad reduce/param gather 要配置到 Megatron model config。
- 首个 step 可能需要临时关闭 forward pre-hook。

设计选择：
- 断言 `num_microbatches` 与 `global_batch_sizes` 长度一致。
- 重置所有 iterator，并将 model chunks 切到 train mode。
- 配置 `grad_scale_func/no_sync_func/grad_sync_func/param_sync_func/finalize_model_grads_func`。
- 创建 microbatch progress bar。
- 遍历 `num_steps_per_rollout`，逐 step 调 `train_one_step`。

Explain：
model-level `train` 是 rollout 粒度的训练循环。它把一次 rollout 的动态 batch 切成若干 Megatron step，每个 step 再由 `train_one_step` 执行 forward/backward/optimizer。

来源：slime/backends/megatron_utils/model.py L732-L845

Code：

```python
args = get_args()

assert len(num_microbatches) == len(global_batch_sizes), (
    f"num_microbatches and global_batch_sizes must have the same length, "
    f"got {len(num_microbatches)} vs {len(global_batch_sizes)}"
)

for iterator in data_iterator:
    iterator.reset()

for model_module in model:
    model_module.train()

config = get_model_config(model[0])
config.grad_scale_func = optimizer.scale_loss
config.timers = None
if isinstance(model[0], DDP) and args.overlap_grad_reduce:
    assert config.no_sync_func is None, (...)
    config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
    if len(model) == 1:
        config.no_sync_func = config.no_sync_func[0]
    if args.align_grad_reduce:
        config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
        if len(model) == 1:
            config.grad_sync_func = config.grad_sync_func[0]
if args.overlap_param_gather and args.align_param_gather:
    config.param_sync_func = [model_chunk.start_param_sync for model_chunk in model]
    if len(model) == 1:
        config.param_sync_func = config.param_sync_func[0]
config.finalize_model_grads_func = finalize_model_grads

num_steps_per_rollout = len(num_microbatches)
microbatch_pbar = tqdm(...)

for step_id in range(num_steps_per_rollout):
    loss_dict, grad_norm = train_one_step(
        args,
        rollout_id,
        step_id,
        data_iterator,
        model,
        optimizer,
        opt_param_scheduler,
        num_microbatches[step_id],
        global_batch_sizes[step_id],
        microbatch_pbar=microbatch_pbar,
    )
```

代码逻辑：
- 获取 Megatron 全局 args。
- 校验 step 切分数组长度一致。
- 重置 data iterators。
- 所有 model chunks 进入 train mode。
- 配置 optimizer loss scaling 和 DDP overlap hooks。
- 设置 finalize grads 函数。
- 创建 progress bar。
- 逐 step 调 `train_one_step`，传入该 step 的 microbatch 和 global batch size。

为什么这样写：
- dynamic batch 下 scheduler 增量和 loss scaling 需要 per-step global batch size。
- iterator reset 保证 forward/backward 从 rollout 数据开头读。
- overlap hooks 必须写入 Megatron config，pipeline engine 才能使用。
- forward pre-hook 的禁用/恢复围绕首 step，降低 checkpoint 初始化错误传播风险。

不变量与失败模式：
- `num_microbatches` 与 `global_batch_sizes` 长度必须一致。
- DDP overlap 模式下 `config.no_sync_func` 不能已有自定义值。
- 每个 step 的 `global_batch_sizes[step_id]` 必须对应该 step 数据。

Comment：
这一层负责把 rollout batch 切成 Megatron 可执行的训练 step。

### 3.2 `train_one_step` 执行 zero grad、pipeline forward/backward、optimizer step

问题与约束：
- 每个 step 前要清空梯度 buffer 和 optimizer grad。
- forward_step 必须从 data iterator 取出包含 policy/value loss 所需字段的 batch。
- Megatron pipeline engine 负责 forward/backward 调度。
- NaN/Inf 梯度和 MTP CI 检查必须发生在 optimizer step 前。
- LR scheduler 增量应使用该 step 的 global batch size。

设计选择：
- 先 `zero_grad_buffer` 和 `optimizer.zero_grad()`。
- 定义内部 `forward_step`，调用 `get_batch` 并把 batch 字段传给 model。
- `forward_backward_func(..., forward_only=False)` 执行训练图。
- 根据 grad 检查结果决定是否 `optimizer.step()`。
- step 成功后 `opt_param_scheduler.step(increment=step_global_batch_size)`。
- 最后释放 grad。

Explain：
`train_one_step` 是单个 Megatron step 的核心事务：构造 microbatch forward 闭包，交给 Megatron pipeline engine 跑 backward，然后执行优化器与调度器更新。

来源：slime/backends/megatron_utils/model.py L549-L680

Code：

```python
for model_chunk in model:
    model_chunk.zero_grad_buffer()
optimizer.zero_grad()

def forward_step(data_iterator: DataIterator, model: GPTModel, return_schedule_plan: bool = False):
    batch = get_batch(
        data_iterator,
        _with_rollout_top_p_token_keys(
            args,
            [
                "tokens",
                "multimodal_train_inputs",
                "packed_seq_params",
                "total_lengths",
                "response_lengths",
                "loss_masks",
                "log_probs",
                "ref_log_probs",
                "values",
                "advantages",
                "returns",
                "rollout_log_probs",
                "teacher_log_probs",
                "rollout_mask_sums",
            ],
        ),
        args.data_pad_size_multiplier,
        args.allgather_cp,
    )

    forward_kwargs = {
        "input_ids": batch["tokens"],
        "position_ids": None,
        "attention_mask": None,
        "labels": None,
        "packed_seq_params": batch["packed_seq_params"],
        "loss_mask": batch["full_loss_masks"],
    }

    if batch["multimodal_train_inputs"] is not None:
        forward_kwargs.update(batch["multimodal_train_inputs"])

    if args.enable_mtp_training:
        forward_kwargs["mtp_kwargs"] = {"mtp_labels": batch["tokens"]}

    output_tensor = model(**forward_kwargs)
    return output_tensor, partial(loss_function, args, batch, num_microbatches, step_global_batch_size)

forward_backward_func = get_forward_backward_func()
losses_reduced = forward_backward_func(
    forward_step_func=_wrap_forward_step_with_microbatch_pbar(forward_step, microbatch_pbar),
    data_iterator=data_iterator,
    model=model,
    num_microbatches=num_microbatches,
    seq_length=args.seq_length,
    micro_batch_size=args.micro_batch_size,
    decoder_seq_length=args.decoder_seq_length,
    forward_only=False,
)

if valid_step:
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    assert update_successful
    opt_param_scheduler.step(increment=step_global_batch_size)

for model_chunk in model:
    model_chunk.zero_grad_buffer()
optimizer.zero_grad()
```

代码逻辑：
- 清空 model chunk grad buffer。
- 清空 optimizer 梯度。
- `forward_step` 从 iterator 取训练 batch。
- 构造 model forward kwargs。
- 多模态和 MTP 字段按需加入。
- 返回 output tensor 和 loss function partial。
- 调 Megatron forward_backward_func 运行 backward。
- valid step 时更新 optimizer 和 scheduler。
- step 末尾再次清空梯度。

为什么这样写：
- Megatron pipeline engine 要求 forward_step 返回 output 和 loss callback。
- loss_function 持有 batch、microbatch 数和 global batch size，才能做 policy/value/SFT 分支和缩放。
- scheduler 使用 sample/global batch size 增量，动态 batch 才能正确累计。
- step 后清 grad 避免下一 step 污染。

不变量与失败模式：
- batch keys 必须覆盖当前 loss_type 所需字段。
- `optimizer.step()` 必须返回 update_successful。
- MTP 训练与 combined schedule plan 的约束由 forward_step 中 assert 保护。

Comment：
这就是训练 step 的最小闭环：batch → forward/backward → optimizer → scheduler。

### 3.3 日志与 CI 断言只在主日志 rank 上检查关键指标

问题与约束：
- 分布式训练中不是每个 rank 都应写日志。
- CI 需要捕获初始 KL、rollout logprob mismatch 等回归。
- routing replay 等特殊模式会改变初始 KL 的预期。

设计选择：
- log_dict 中写入 grad norm、MTP loss、各 param group LR、per-step global batch size 和 accumulated step。
- `args.ci_test` 时对 rollout logprob abs diff、PPO KL、KL loss 做断言。
- routing replay 场景跳过初始 KL 为零的断言。

Explain：
训练日志不只是观测指标，也承载 CI 不变量。源码把关键 PPO 初始状态检查嵌入 train loop，避免 silent regression。

来源：slime/backends/megatron_utils/model.py L880-L907

Code：

```python
log_dict[f"train/{role_tag}grad_norm"] = grad_norm
if args.enable_mtp_training:
    log_dict[f"train/{role_tag}mtp_loss"] = mtp_losses

for param_group_id, param_group in enumerate(optimizer.param_groups):
    log_dict[f"train/{role_tag}lr-pg_{param_group_id}"] = opt_param_scheduler.get_lr(param_group)

log_dict[f"train/{role_tag}global_batch_size"] = global_batch_sizes[step_id]
log_dict["train/step"] = accumulated_step_id
logging_utils.log(args, log_dict, step_key="train/step")

if args.ci_test and "train/train_rollout_logprob_abs_diff" in log_dict:
    assert log_dict["train/train_rollout_logprob_abs_diff"] <= 0.1, f"{log_dict=}"

if args.ci_test and not args.ci_disable_kl_checker:
    if step_id == 0 and "train/ppo_kl" in log_dict and "train/pg_clipfrac" in log_dict:
        assert log_dict["train/ppo_kl"] < 1e-8, f"{log_dict=}"
    if (
        accumulated_step_id == 0
        and not getattr(args, "use_rollout_routing_replay", False)
        and "train/kl_loss" in log_dict
    ):
        assert log_dict["train/kl_loss"] < 1e-8, f"{log_dict=}"
```

代码逻辑：
- 记录 grad norm。
- MTP 开启时记录 MTP loss。
- 遍历 optimizer param groups 记录 LR。
- 记录当前 step global batch size 和累计 step。
- 写日志。
- CI 模式检查 rollout logprob 差异。
- CI 模式检查 PPO 初始 KL 和 KL loss。
- routing replay 模式跳过 KL loss 零断言。

为什么这样写：
- per-step global batch size 是动态 batch 诊断的关键指标。
- 初始 KL 接近零是 PPO/ref/actor 对齐的重要不变量。
- routing replay 改变路由路径，初始 KL 不一定严格为零，因此源码显式排除。

不变量与失败模式：
- CI 阈值失败会直接 assert。
- `log_dict` 只有在对应指标存在时才检查，避免无关 loss type 报错。
- logging rank 的选择在外层控制，这里假设当前代码运行在需要记录的路径上。

Comment：
这段把训练正确性检查嵌进常规 train loop，比单独离线检查更早发现回归。
