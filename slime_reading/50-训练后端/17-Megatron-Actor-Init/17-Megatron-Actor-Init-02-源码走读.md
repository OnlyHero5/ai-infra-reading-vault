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
updated: 2026-07-05
---

# Megatron Actor 初始化 · 源码走读

> 读法：`MegatronTrainRayActor.init` 是训练侧 actor 的状态装配入口。主线是 debug 短路、分布式初始化、HF 配置读取、模型与 optimizer 初始化、权重备份、推权策略选择、offload sleep/wake，以及 `initialize.py` 中的 Megatron 并行拓扑和随机种子设置。

---

## 1. Actor init 主路径

### 1.1 `init` 入口：计时与 rollout-only debug 短路

来源：slime/backends/megatron_utils/actor.py L47-L57

**问题与约束：** 训练 actor 初始化可能很重，但 rollout-only debug 模式不应启动 Megatron、NCCL 或 checkpoint 逻辑；同时初始化等待时间要纳入 timer。

**设计选择：** 用 `with_defer` 在 init 期间启动 `train_wait` timer；函数开头若 `debug_rollout_only` 为真，仅保存 `args` 并返回 `0`。

**Explain：** 这个短路把“只调试 rollout”与完整训练 actor 初始化彻底分开。返回 `0` 仍满足上层期望的 start rollout id 形态。

**Code：**

```python
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

**代码逻辑：** decorator 在进入 init 时启动等待计时。函数签名接收角色和可选辅助模型标志；第一段判断 rollout-only debug，设置 actor 的 args 后直接返回。

**为什么这样写：** rollout-only debug 不需要训练进程组、Megatron model 或 optimizer。早退可以避免昂贵初始化，也避免无训练场景下触发分布式副作用。

**不变量与失败模式：** debug 短路后只保证 `self.args` 可用；调用方不能期待 `self.model`、`weight_updater` 或 process group 已初始化。若 debug 模式继续走正常 init，可能在没有训练资源时卡在 Megatron/NCCL 初始化。

**Comment：** 入口先定清楚 actor 是否真的要成为训练 actor；这是后续所有重初始化的前置分支。

### 1.2 分布式补丁、父类 init 与 Megatron init

来源：slime/backends/megatron_utils/actor.py L59-L67

**问题与约束：** Megatron 初始化依赖 torch distributed 与父类训练 actor 的通信上下文；Slime 还需要支持 offload 后重建 process group。

**设计选择：** 先 `monkey_patch_torch_dist()`，再调用 `super().init(...)` 建立基础 actor/通信状态，随后执行 Megatron `init(args)`，主 rank 初始化 tracking，最后创建 `TrainProfiler`。

**Explain：** 这里把 Slime 的 Ray actor 通信层和 Megatron 的模型并行层按顺序接起来。patch 必须早于父类和 Megatron 使用 torch.distributed。

**Code：**

```python
        monkey_patch_torch_dist()
        super().init(args, role, with_ref, with_opd_teacher)

        init(args)

        if is_megatron_main_rank():
            init_tracking(args, primary=False, role=role)

        self.prof = TrainProfiler(args)
```

**代码逻辑：** 函数先打分布式补丁，再进入父类初始化；之后调用本模块的 Megatron 初始化函数。主 rank 才初始化 tracking，所有 rank 都创建 profiler。

**为什么这样写：** process group 的可重载能力要在任何分布式调用前生效；tracking 只需主 rank 执行，避免重复日志或外部追踪记录。

**不变量与失败模式：** patch 必须在 `super().init` 前；Megatron `init(args)` 需要父类已设置 rank/world 等基础字段；主 rank 判断要基于 Megatron 并行状态。顺序颠倒会导致 process group 生命周期和 Megatron 拓扑不一致。

**Comment：** 这一段是 init 的“通信栈搭桥”：Slime actor 基础通信在前，Megatron 并行拓扑在后。

### 1.3 HF config/tokenizer：按本机 GPU 序列化加载

来源：slime/backends/megatron_utils/actor.py L69-L76

**问题与约束：** 多个 rank 同时从同一 HF checkpoint 读取 config/tokenizer，可能触发并发写缓存或文件锁问题；但每个 actor 都需要本地 `hf_config` 和 tokenizer。

**设计选择：** 按 `args.num_gpus_per_node` 循环，让同一节点上只有与当前槽位匹配的 rank 执行 `AutoConfig.from_pretrained` 和 `AutoTokenizer.from_pretrained`，每轮用 Gloo barrier 串行化。

**Explain：** 这是本机内的串行读取栅栏。它不改变 HF 读取结果，只降低同节点多进程并发访问 HF cache 的风险。

**Code：**

```python
        # read config and tokenizer serialized to prevent concurrent writing bug.
        for i in range(args.num_gpus_per_node):
            if i == dist.get_rank() % args.num_gpus_per_node:
                self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())

        dist.barrier(group=get_gloo_group())
```

**代码逻辑：** 循环按本机 GPU 槽位推进；每次只有一个本地 rank 满足条件并读取 config/tokenizer，随后所有 rank 在 Gloo group 上等待。循环结束后再做一次 barrier。

**为什么这样写：** HF 加载可能会写 tokenizer/config 缓存或触发 remote code 文件准备。并行读取不是训练关键路径的性能瓶颈，串行化换来更稳定的初始化。

**不变量与失败模式：** `num_gpus_per_node` 应等于本机参与 rank 数；所有 rank 必须进入相同 barrier 次数；checkpoint 路径对所有 rank 可读。若某个 rank 跳过 barrier，会导致同节点其他 rank 死等。

**Comment：** Megatron actor 初始化中，这是一段偏工程稳定性的同步，而不是模型并行逻辑。

### 1.4 offload memory margin 与模型/optimizer 初始化

来源：slime/backends/megatron_utils/actor.py L78-L85

**问题与约束：** 开启 train offload 时，训练显存释放/恢复要给 memory saver 留出边距；随后必须初始化模型、optimizer、scheduler，并知道 checkpoint 已加载到哪个 rollout id。

**设计选择：** 若 `offload_train` 且 `train_memory_margin_bytes > 0`，设置 `torch_memory_saver.memory_margin_bytes`；然后调用 `initialize_model_and_optimizer(args, role)`。

**Explain：** memory margin 是 offload 模式的运行时保护参数。模型初始化返回的 `loaded_rollout_id` 后面会被转成下一轮 rollout 起点。

**Code：**

```python
        if args.offload_train:
            if (x := args.train_memory_margin_bytes) > 0:
                logger.info(f"Set torch_memory_saver.memory_margin_bytes to {x}")
                torch_memory_saver.memory_margin_bytes = x

        self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id = initialize_model_and_optimizer(
            args, role
        )
```

**代码逻辑：** 条件分支只在 offload train 下设置 memory saver 的边距；随后模型初始化函数一次性返回模型、优化器、学习率调度器和加载点。

**为什么这样写：** offload 恢复训练栈时，如果完全占满显存，rollout/训练切换更容易 OOM。把边距设置放在模型初始化前，可以影响后续 memory saver 行为。

**不变量与失败模式：** memory margin 只应在 offload 模式下生效；`initialize_model_and_optimizer` 必须返回四元组；`loaded_rollout_id` 要和 checkpoint 语义一致。若边距过小，offload 恢复可能和 rollout 显存争抢。

**Comment：** 这里开始进入真正训练状态装配：显存策略先设好，再创建 Megatron 训练对象。

### 1.5 `train_parallel_config`：缓存 DP/CP/VPP 元信息

来源：slime/backends/megatron_utils/actor.py L87-L101

**问题与约束：** 后续 data pipeline 和调度需要知道训练并行维度；VPP 大于 1 时，microbatch group size 还要从 model config 读取。

**设计选择：** 查询 Megatron mpu 的 VPP、DP、CP world size；VPP>1 时从 `get_model_config(self.model[0])` 取 `microbatch_group_size_per_vp_stage`，最后写入 `self.train_parallel_config`。

**Explain：** 这份 dict 是 actor 对外暴露训练拓扑的轻量缓存，避免后续模块直接重复访问 Megatron mpu 或 model config。

**Code：**

```python
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

**代码逻辑：** 代码先计算 VPP size，按是否启用 VPP 决定 microbatch group size；随后把 DP、CP、VPP 和 microbatch 组大小写入 dict。最后把 loaded rollout id 加一得到下一轮起点。

**为什么这样写：** 并行维度来自 Megatron 初始化后的全局状态，必须等模型和 mpu 就绪后读取。`start_rollout_id = loaded + 1` 保证从 checkpoint 下一步继续 rollout，而不是重复已完成编号。

**不变量与失败模式：** `self.model[0]` 在 VPP>1 时必须有 config；DP size 查询显式不含 context parallel；`start_rollout_id` 表示下一条 rollout id。若 VPP group size 读取错误，microbatch 调度会和模型流水线不匹配。

**Comment：** 训练 actor 把 Megatron 并行世界压缩成一个 dict，供 Slime 自己的调度层使用。

### 1.6 actor 权重备份与辅助 checkpoint

来源：slime/backends/megatron_utils/actor.py L108-L131

**问题与约束：** actor、ref、teacher、old_actor 等权重需要在同一 Megatron actor 内切换或备份；权重名还可能需要转成 HF 全局名。

**设计选择：** 创建 `TensorBackuper`，source getter 读取当前 model 的 named params/buffers；先备份 `"actor"`，再按 flag 加载并备份 ref、teacher、old_actor，必要时额外备份 `"rollout_actor"`。

**Explain：** `TensorBackuper` 让同一个 actor 能在多个权重 tag 之间切换，而不用维护多份完整模型对象。

**Code：**

```python
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

        # Load teacher model for Megatron-based on-policy distillation
        if with_opd_teacher:
            self.load_other_checkpoint("teacher", args.opd_teacher_load)

        if self.args.keep_old_actor:
            # Load old_actor checkpoint
            self.load_other_checkpoint("old_actor", args.load)
            # Create rollout_actor as a copy of current actor
            if args.update_weights_interval == 1:
                self.weights_backuper.backup("rollout_actor")
```

**代码逻辑：** source getter 捕获 args 和 model，按模式决定是否转换全局名。actor 权重先备份；之后按 `with_ref`、`with_opd_teacher`、`keep_old_actor` 分支加载其他 checkpoint 并进入对应 tag 管理。

**为什么这样写：** RL 训练常需要 actor/ref/old policy 对比；复制完整模型对象成本太高。tag 式 tensor 备份能复用同一 Megatron 模型结构，按需要恢复权重。

**不变量与失败模式：** `"actor"` tag 必须先存在；global name 转换模式要与后续权重同步一致；辅助 checkpoint 路径必须和模型结构兼容。若 active tag 和实际权重不一致，训练或推权会拿错策略。

**Comment：** 这段建立了 actor 内部的多权重视图，是后续 `_switch_model`、old actor 和 rollout actor 的基础。

### 1.7 `weight_updater`：按 colocate/mode/transport 选择推权实现

来源：slime/backends/megatron_utils/actor.py L133-L168

**问题与约束：** 训练权重要同步到 rollout engine；不同部署形态支持不同路径：colocate 只能 tensor 直传，delta 目前要求 disk transport，full 权重可走 disk 或 NCCL distributed。

**设计选择：** 先补齐 `args.vocab_size`；然后按 `colocate`、`update_weight_mode=="delta"`、full+disk/full+nccl 分支选择 `UpdateWeight*` 类，最后实例化 `self.weight_updater`。

**Explain：** 这里把 CLI 配置转成实际推权策略对象，并用 assert 阻止不支持的组合。

**Code：**

```python
        if self.args.vocab_size is None:
            # Prefer HF config vocab_size (which may include model-native padding)
            # over tokenizer vocab_size (which may be smaller, e.g. GPT-OSS).
            hf_vocab = getattr(self.hf_config, "vocab_size", None)
            self.args.vocab_size = hf_vocab if hf_vocab is not None else self.tokenizer.vocab_size

        if self.args.colocate:
            assert (
                self.args.update_weight_mode == "full"
            ), "--update-weight-mode=delta is not supported with --colocate"
            update_weight_cls = UpdateWeightFromTensor
        elif self.args.update_weight_mode == "delta":
            # Delta sync is disk-transport only: each host applies the published deltas into
            # its local checkpoint and the engines reload via vanilla update_weights_from_disk.
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
                assert (
                    self.args.update_weight_mode == "full" and self.args.update_weight_transport == "nccl"
                ), f"unsupported weight sync mode/transport: {self.args.update_weight_mode!r}/{self.args.update_weight_transport!r}"
                update_weight_cls = UpdateWeightFromDistributed
        self.weight_updater = update_weight_cls(
            self.args,
            self.model,
            weights_getter=lambda: self.weights_backuper.get("actor"),
            model_name=type(self.hf_config).__name__.lower() if self.args.model_name is None else self.args.model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
        )
```

**代码逻辑：** 函数先用 HF config 或 tokenizer 填 vocab size。策略分支中，colocate 选 tensor；delta 断言 disk 并导入 delta disk 类；full 模式按 disk/nccl 选类。实例化时传入 actor 权重 getter、model name 和量化配置。

**为什么这样写：** 推权路径受部署模式强约束。把组合校验放在 init 阶段，可以在训练开始前发现不支持的配置；权重 getter 固定指向 `"actor"` tag，避免同步到 ref/old_actor。

**不变量与失败模式：** colocate 不支持 delta；delta 只支持 disk；NCCL transport 只用于 full 权重；vocab size 要匹配 HF 模型 padding。若推权策略选错，rollout engine 会加载错误权重形态或无法通信。

**Comment：** `weight_updater` 是训练 actor 和 rollout engine 之间的权重同步桥，init 阶段就把桥型定死。

### 1.8 init 收尾：清显存、offload sleep 与返回起点

来源：slime/backends/megatron_utils/actor.py L170-L188

**问题与约束：** 初始化结束后 GPU 上可能残留临时缓存；offload 模式希望训练 actor 初始化后立即释放训练栈，让 rollout 能占用显存；还要加载可选 rollout data postprocess hook。

**设计选择：** 先 `clear_memory()`；若 `offload_train`，切回 actor tag 并调用 `sleep()`；初始化 rollout engine/postprocess 字段，结束 profiler，并返回 `start_rollout_id`。

**Explain：** 这段把“训练 actor 已准备好”转换成“训练 actor 可以暂时让出 GPU”。返回值告诉上层下一轮 rollout 从哪个 id 开始。

**Code：**

```python
        # empty cache after initialization
        clear_memory()

        if self.args.offload_train:
            # recover to actor in the end.
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

**代码逻辑：** 函数清空缓存；offload 时恢复 actor 权重并进入 sleep；然后初始化 rollout 相关字段，按路径加载 postprocess hook，标记 profiler 初始化结束，最后返回 rollout 起点。

**为什么这样写：** offload 模式的目标是训练和 rollout 轮流占用 GPU。init 完毕立即 sleep，可以让系统在进入 rollout 前处于低训练显存状态。

**不变量与失败模式：** `sleep()` 只在 `offload_train` 下合法；sleep 前要确保 active model 是 actor；postprocess hook 必须是可加载函数。若 init 后不 sleep，rollout 可能因训练栈仍占显存而 OOM。

**Comment：** init 的最后一步不是继续训练，而是把训练 actor 放到可切换状态。

---

## 2. Offload 生命周期

### 2.1 `sleep`：释放训练栈并暂停 memory saver

来源：slime/backends/megatron_utils/actor.py L190-L207

**问题与约束：** offload 模式下，训练阶段结束或初始化结束后要释放显存和分布式组；actor+critic 且非 colocate 时，还可能存在与 rollout engine 的权重同步连接。

**设计选择：** `sleep` 断言 `offload_train`，清理 GPU/host memory；特定 actor+critic 场景下断开 rollout engine 连接；销毁 process groups，暂停 `torch_memory_saver`。

**Explain：** sleep 是训练栈的“下电”过程。它不仅释放 tensor 缓存，也把分布式通信状态拆掉，为 rollout 或其他阶段让资源。

**Code：**

```python
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

**代码逻辑：** 函数先确认 offload 模式，清理 host/GPU memory 并打印内存。满足 actor+critic+非 colocate 且 updater 支持时断开 rollout engine；随后销毁进程组，暂停 memory saver，再打印内存。

**为什么这样写：** process group 和权重同步连接都可能持有 GPU/NCCL 资源。sleep 必须同时处理内存和通信，否则后续 rollout 或 wake_up 会遇到资源冲突。

**不变量与失败模式：** 非 offload 模式不能调用 sleep；disconnect 方法是可选能力，需要 `hasattr`；destroy process groups 后必须靠 wake_up 重建。若不暂停 memory saver，后续恢复时状态可能和实际内存布局不一致。

**Comment：** sleep 是训练 actor 的显存让渡点，清理范围比普通 `clear_memory()` 更深。

### 2.2 `wake_up`：恢复 memory saver、进程组与 actor 权重

来源：slime/backends/megatron_utils/actor.py L209-L220

**问题与约束：** sleep 后训练 actor 要重新进入可训练状态，需要恢复 memory saver、重建 process group，并确保 actor 权重 tag 在 GPU 上有效。

**设计选择：** `wake_up` 断言 offload 模式，先 resume memory saver，再清理缓存、reload process groups；actor 角色额外 `_switch_model("actor")`。

**Explain：** wake_up 是 sleep 的逆过程。它重建通信上下文，并把权重视图切回 actor，确保后续 train/save/update 能使用正确模型状态。

**Code：**

```python
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

**代码逻辑：** 函数打印恢复前内存，恢复 memory saver，清理缓存并重载分布式进程组。actor 角色恢复 actor 权重，最后打印恢复后内存。

**为什么这样写：** 分布式组销毁后不能直接训练；权重备份可能驻留在 memory saver 管理的状态里。先 resume 再 reload/switch，可以让内存管理和通信状态按顺序恢复。

**不变量与失败模式：** wake_up 必须和 sleep 成对出现在 offload 模式；reload 后的 process group 要与 Megatron 初始化拓扑一致；actor tag 必须存在。若训练前忘记 wake_up，会在通信或模型权重访问处失败。

**Comment：** sleep/wake_up 把训练 actor 做成可暂停资源，而不是一次性常驻 GPU 的进程。

---

## 3. Megatron initialize.py

### 3.1 `_set_random_seed`：按 PP/DP rank 偏移随机种子

来源：slime/backends/megatron_utils/initialize.py L14-L30

**问题与约束：** 不同 pipeline stage 和可选 data parallel rank 需要可复现但不完全相同的随机序列；同时 Megatron tensor parallel CUDA RNG tracker 也要同步设置。

**设计选择：** 基础 seed 加上 `100 * pipeline_rank`；若 `data_parallel_random_init` 开启，再加 `10 * data_parallel_rank`；随后设置 Python、NumPy、Torch 和 tensor parallel CUDA seed。

**Explain：** 这段把全局 seed 映射到不同并行 rank 的本地 seed，避免所有 stage 使用完全相同随机流。

**Code：**

```python
def _set_random_seed(
    seed_: int,
    data_parallel_random_init: bool = False,
    te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
):
    """Set random seed for reproducability."""
    # Ensure that different pipeline MP stages get different seeds.
    seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())
    # Ensure different data parallel ranks get different seeds
    if data_parallel_random_init:
        seed = seed + (10 * mpu.get_data_parallel_rank(with_context_parallel=False))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tensor_parallel.model_parallel_cuda_manual_seed(seed, te_rng_tracker, inference_rng_tracker, use_cudagraphable_rng)
```

**代码逻辑：** 函数根据 PP rank 和可选 DP rank 计算本地 seed，然后依次设置 Python random、NumPy、Torch CPU/CUDA seed 和 Megatron tensor parallel RNG。

**为什么这样写：** Pipeline stage 处理不同模型分段，完全相同随机流不一定合理；但偏移规则固定，仍能保证复现实验。Megatron 自己的 CUDA RNG tracker 也必须使用同一最终 seed。

**不变量与失败模式：** 所有 rank 必须用相同基础 seed 和相同偏移规则；DP rank 查询显式不含 context parallel；TE/inference/cudagraphable tracker 参数要透传。若只设置 torch seed 而不设置 tensor parallel tracker，模型并行层随机性会不一致。

**Comment：** seed 初始化不是单点赋值，而是把同一实验 seed 分配到 Megatron 并行拓扑上。

### 3.2 `_initialize_distributed`：把 CLI 并行参数交给 Megatron Core

来源：slime/backends/megatron_utils/initialize.py L33-L53

**问题与约束：** TP、PP、CP、EP、distributed optimizer 实例数和通信 backend 都来自 Slime CLI；Megatron Core 需要一次性初始化模型并行通信拓扑。

**设计选择：** 调用 `mpu.initialize_model_parallel(...)`，显式传入各并行维度、timeout、NCCL 配置、rank order、embedding rank hook 和可选 Gloo process groups。

**Explain：** 这是 Megatron 并行世界的实际创建点。Slime 的 args 在这里被翻译成 Megatron Core 的 parallel state。

**Code：**

```python
def _initialize_distributed(args, get_embedding_ranks=None, get_position_embedding_ranks=None):
    """Initialize torch.distributed and core model parallel."""
    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
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

**代码逻辑：** 函数把各并行 size 和通信配置传入 `initialize_model_parallel`；`order` 根据 `use_tp_pp_dp_mapping` 选择两种 rank 到维度的映射；embedding/position rank hook 原样透传。

**为什么这样写：** Slime 需要支持 MoE、context parallel、virtual pipeline 等 Megatron 组合。集中调用 Megatron Core 初始化，可以让后续 actor 只查询 mpu，而不重复解释 CLI。

**不变量与失败模式：** 各并行维度乘积必须匹配 world size；rank order 必须和训练/权重同步假设一致；Gloo process group 开关要和后续 barrier 用法兼容。若 order 选错，rank 会落到错误并行维度。

**Comment：** 训练 actor 的分布式语义最终由这一调用落地。

### 3.3 deterministic、TP overlap 与 custom init hook

来源：slime/backends/megatron_utils/initialize.py L88-L104

**问题与约束：** 有些实验要求强确定性；有些模型需要 TP communication overlap；用户还可能需要在 Megatron 标准 init 后挂自定义初始化逻辑。

**设计选择：** deterministic 模式下设置 cuDNN deterministic、关闭 benchmark，并启用 deterministic algorithms；TP overlap 时调用 Megatron 的 `_initialize_tp_communicators`；有 custom path 时动态加载并执行。

**Explain：** 这些是 Megatron 初始化尾部的可选扩展点。它们都依赖标准分布式/parallel state 已经建立。

**Code：**

```python
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

**代码逻辑：** 代码按三个独立开关执行：确定性开关修改 PyTorch backend 行为；TP overlap 初始化额外 communicator；custom path 加载函数并传入 args。

**为什么这样写：** 这些行为不是所有训练都需要。放在 init 末尾，可以在 Megatron 默认初始化完成后再增强通信或应用用户 patch，降低对基础初始化的干扰。

**不变量与失败模式：** deterministic algorithms 可能拒绝非确定性算子；TP communicator 初始化要求 Megatron parallel state 已存在；custom init 必须是可调用函数。若 custom hook 放在 Megatron init 前，可能读不到 parallel state。

**Comment：** `initialize.py` 的尾部给实验控制和用户扩展留了明确入口。

---

## 4. Debug flag 在运行路径的联动

### 4.1 `train`：rollout-only debug 跳过训练

来源：slime/backends/megatron_utils/actor.py L380-L382

**问题与约束：** rollout-only debug 下没有完整训练 actor 状态，`train()` 不能访问模型、optimizer 或 rollout data。

**设计选择：** `train` 函数开头检查 `self.args.debug_rollout_only`，命中直接返回 `None`。

**Explain：** 这和 init 的 early return 配套，保证后续即使误调用 train，也不会进入 Megatron 训练逻辑。

**Code：**

```python
    def train(self, rollout_id: int, rollout_data_ref: Box, external_data=None):
        if self.args.debug_rollout_only:
            return None
```

**代码逻辑：** 函数进入后第一行判断 debug flag；命中时不解引用 rollout data，也不访问训练对象。

**为什么这样写：** debug 模式常复用上层调度流程，不能假设 train 永远不会被调到。方法级 guard 能把非法路径变成无副作用返回。

**不变量与失败模式：** debug rollout-only actor 只保证 args 存在；返回 `None` 表示没有训练结果。若缺少 guard，init 短路后的 `self.model` 访问会报错。

**Comment：** init 短路不是孤立分支，train 方法也必须承认这个轻量 actor 状态。

### 4.2 `save_model`：rollout-only debug 跳过 checkpoint IO

来源：slime/backends/megatron_utils/actor.py L558-L560

**问题与约束：** rollout-only debug 没有训练模型可保存，也不应触发 checkpoint 写入。

**设计选择：** `save_model` 开头检查 `debug_rollout_only`，命中直接返回。

**Explain：** 这个 guard 避免调试 rollout 时误写 checkpoint 或访问未初始化模型。

**Code：**

```python
    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only:
            return
```

**代码逻辑：** 方法入口先判断 debug flag；命中后直接结束，没有 checkpoint IO 副作用。

**为什么这样写：** 上层保存逻辑可能按周期触发，不一定知道当前是 rollout-only debug actor。方法内部 guard 更稳妥。

**不变量与失败模式：** debug 模式返回值为 `None`；不应创建 checkpoint 目录或同步权重。若继续保存，会因为模型/optimizer 未初始化而失败。

**Comment：** checkpoint 路径也要和 init 短路保持一致：没有训练状态，就没有可保存状态。

### 4.3 `update_weights`：train-only 与 rollout-only 都跳过推权

来源：slime/backends/megatron_utils/actor.py L583-L585

**问题与约束：** debug train only 或 debug rollout only 都不应执行完整推权：前者没有 rollout engine 需要更新，后者没有训练权重可推。

**设计选择：** `update_weights` 开头同时检查两个 debug flag，任一为真直接返回。

**Explain：** 推权是训练 actor 与 rollout engine 之间的桥。任一侧被 debug 模式裁掉时，这座桥都不应该运行。

**Code：**

```python
    def update_weights(self) -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return
```

**代码逻辑：** 方法入口用 or 条件覆盖两个 debug 模式；命中时不访问 `weight_updater`。

**为什么这样写：** 两种 debug 模式裁剪的是不同半边系统，但都让权重同步没有意义。统一 guard 可以避免 NCCL/disk 推权路径误触发。

**不变量与失败模式：** debug 模式下 `weight_updater` 可能不存在或没有目标 engine；正常模式必须继续执行后续推权逻辑。若缺少 guard，debug 运行可能卡在通信或 IO。

**Comment：** 这三个方法级 guard 修复了 init 短路后的运行时边界，保证 debug actor 不被当成完整训练 actor 使用。
