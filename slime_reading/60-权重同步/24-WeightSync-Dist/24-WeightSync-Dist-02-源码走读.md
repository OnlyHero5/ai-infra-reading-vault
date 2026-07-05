---
type: batch-doc
module: 24-WeightSync-Dist
batch: "24"
doc_type: walkthrough
title: "NCCL 权重同步 · 源码走读"
tags:
  - slime/batch/24
  - slime/module/weight-sync-dist
  - slime/doc/walkthrough
updated: 2026-07-05
---

# NCCL 权重同步 · 源码走读

> 走读主线：Actor 根据 `update_weight_mode/transport` 选择 updater；训练轮次结束后 Actor 取得可更新 rollout engines；`UpdateWeightFromDistributed` 建立训练 rank0 到 SGLang engine GPU 的 NCCL group；参数先按 Megatron 全局命名和 TP/EP gather 转成 HF chunk，再用 Ray 发送 metadata、NCCL broadcast tensor。

---

## 1. Actor 侧选型与编排

### 1.1 Actor 初始化阶段选择 UpdateWeightFromDistributed

问题与约束：
- Slime 同时支持 colocate tensor 更新、disk 更新、delta disk 更新和 NCCL 全量更新；Actor 必须在初始化时选出唯一 updater。

设计选择：
- `colocate` 走 `UpdateWeightFromTensor`，`delta` 强制 disk transport，full+disk 走 `UpdateWeightFromDisk`，full+nccl 走 `UpdateWeightFromDistributed`。

Explain：
Actor 根据参数组合选择 updater 类，并统一传入 args、model、actor 权重 getter、model_name 和 quantization_config。NCCL 路径只在 `update_weight_mode == "full"` 且 `update_weight_transport == "nccl"` 时成立。

来源：slime/backends/megatron_utils/actor.py L139-L168

Code：

```python
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

代码逻辑：
- colocate 与 delta 是互斥路径。
- delta 只允许 disk transport。
- 非 disk 的 full transport 必须是 nccl。
- updater 构造参数统一，便于不同实现共享接口。

为什么这样写：
- 权重同步方式影响后续通信拓扑，必须在 Actor 初始化时固定。
- 通过 assert 明确拒绝不支持组合，避免训练中途才暴露配置错误。

不变量与失败模式：
- `update_weight_mode/update_weight_transport/colocate` 必须组合合法。
- NCCL 路径要求 rollout engines 能加入训练 rank0 创建的 NCCL group。

Comment：
NCCL 权重同步不是单独开关，而是 full mode 下的一种 transport。

### 1.2 Actor.update_weights 取得可更新引擎并触发 updater

问题与约束：
- rollout engines 可能动态恢复、扩缩或暂时不存在；offload/critic 非 colocate 场景还可能需要重建通信组。

设计选择：
- Actor 每次更新前从 RolloutManager 获取 engines、分布式锁和 engine GPU 信息；新 engine 或 reconnect 时先 `connect_rollout_engines`，再调用 updater 的 `update_weights`。

Explain：
fault tolerance 模式先在 rank0 恢复可更新引擎，再用 gloo barrier 同步。没有 engine 且无需 reconnect 时跳过更新。offload 训练期间用 `torch_memory_saver.disable()` 包住真正的更新。

来源：slime/backends/megatron_utils/actor.py L583-L628

Code：

```python
def update_weights(self) -> None:
    if self.args.debug_train_only or self.args.debug_rollout_only:
        return

    if self.args.use_fault_tolerance:
        if dist.get_rank() == 0:
            ray.get(self.rollout_manager.recover_updatable_engines.remote())
        dist.barrier(group=get_gloo_group())

    (
        rollout_engines,
        rollout_engine_lock,
        num_new_engines,
        engine_gpu_counts,
        engine_gpu_offsets,
        all_engine_actors,
    ) = ray.get(self.rollout_manager.get_updatable_engines_and_lock.remote())

    reconnect_rollout_engines = self.args.offload_train and self.args.use_critic and not self.args.colocate
    if not rollout_engines and not reconnect_rollout_engines:
        if dist.get_rank() == 0:
            logger.info("No updatable SGLang engines are running; skip weight update.")
        return

    if reconnect_rollout_engines:
        self.wake_up()
    elif self.args.offload_train:
        reload_process_groups()

    if num_new_engines > 0 or reconnect_rollout_engines:
        self.weight_updater.connect_rollout_engines(...)
        dist.barrier(group=get_gloo_group())

    with torch_memory_saver.disable() if self.args.offload_train else nullcontext():
        self.weight_updater.update_weights()
```

代码逻辑：
- debug 模式直接跳过。
- RolloutManager 返回 engines、锁和 GPU 拓扑。
- 新 engine/reconnect 时重建 updater 与 engine 的连接。
- 最后只调用 updater 的统一接口。

为什么这样写：
- Actor 不直接处理 NCCL broadcast，避免训练编排和 transport 细节耦合。
- engine 动态变化时先 connect 再 update，可以支持 fault tolerance 和 engine 重启。

不变量与失败模式：
- 所有训练 rank 必须在 connect 后通过 gloo barrier 对齐。
- `rollout_engine_lock` 必须是所有 PP source rank 共享的 Ray lock。
- 没有 rollout engine 时更新会被跳过，rollout 侧继续使用旧权重。

Comment：
Actor 侧只负责编排“什么时候更新”和“更新给哪些 engine”。

---

## 2. Distributed updater 生命周期

### 2.1 UpdateWeightFromDistributed 保存模型与版本状态

问题与约束：
- NCCL group 不一定在 updater 初始化时可用，因为 rollout engines 会动态出现；但 updater 需要持有模型、转换配置和版本号。

设计选择：
- `__init__` 只保存 args/model/model_name/quantization_config，初始化 `weight_version=0` 和 `_model_update_groups=None`。

Explain：
类 docstring 说明每个 PP rank 使用 `slime-pp_{pp_rank}` group，只有 DP=TP=0 的 PP source rank broadcast；non-expert 和 expert 参数分两类处理。

来源：slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py L23-L50

Code：

```python
class UpdateWeightFromDistributed:
    """
    Update distributed engines via NCCL. Each PP rank: group "slime-pp_{pp_rank}",
    only DP=TP=0 broadcasts. Non-expert (TP) and expert (EP) params separate.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self._model_update_groups = None
        self.update_weight_metrics: dict[str, float] = {}
```

代码逻辑：
- group 句柄延迟到 `connect_rollout_engines` 创建。
- `weight_version` 每次 `update_weights` 自增。
- metrics 用字典累积，后续由 Actor pop。

为什么这样写：
- rollout engines 动态变化，提前创建 NCCL group 会绑定错误 world size。
- 版本号放在 updater 内，能和实际 broadcast 次数保持一致。

不变量与失败模式：
- updater 被使用前必须至少连接一次 rollout engines。
- quantization_config 要与 engine 侧加载格式一致。

Comment：
初始化只建本地状态，不碰分布式通信。

### 2.2 connect_rollout_engines 只让 PP source rank 建 NCCL 组

问题与约束：
- 每个 PP stage 都可能有自己的参数，但一个 PP stage 内不应所有 TP/DP rank 都向 engine broadcast。

设计选择：
- 只有 `data_parallel_rank==0` 且 `tensor_model_parallel_rank==0` 的 rank 设置 `_is_pp_src_rank=True`，用 `slime-pp_{pp_rank}` group 连接 rollout engines。

Explain：
若已有 `_model_update_groups`，先 disconnect 旧组再重建。非 PP source rank 仍保留 updater 对象，但不会创建 NCCL group 或 broadcast。

来源：slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py L57-L100

Code：

```python
def connect_rollout_engines(
    self,
    rollout_engines: Sequence[ActorHandle],
    rollout_engine_lock: ActorHandle,
    engine_gpu_counts: Sequence[int] | None = None,
    engine_gpu_offsets: Sequence[int] | None = None,
    all_engine_actors: Sequence[ActorHandle] | None = None,
) -> None:
    self.rollout_engines = rollout_engines
    self.rollout_engine_lock = rollout_engine_lock
    self._engine_gpu_counts = engine_gpu_counts

    self._is_pp_src_rank = (
        mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
    )
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    if self._is_pp_src_rank:
        self._group_name = f"slime-pp_{pp_rank}"

    if self._is_pp_src_rank:
        if self._model_update_groups is not None:
            disconnect_rollout_engines_from_distributed(
                self.args, self._group_name, self._model_update_groups, self.rollout_engines
            )
        self._model_update_groups = connect_rollout_engines_from_distributed(
            self.args,
            self._group_name,
            rollout_engines,
            engine_gpu_counts=engine_gpu_counts,
        )
```

代码逻辑：
- 保存 engines 与 Ray lock。
- 计算当前 rank 是否为 PP source。
- PP source 以 PP rank 生成 group name。
- 已有 group 时先销毁再重连。

为什么这样写：
- 每个 PP stage 需要独立同步自己负责的参数。
- 只让 TP rank0 broadcast，避免同一参数被多个 TP rank 重复发送。

不变量与失败模式：
- rollout engines 必须和训练 rank0 使用同一个 group name/world size。
- 重连前必须销毁旧 group，否则 NCCL 组可能悬挂。

Comment：
NCCL 同步的核心拓扑是“每个 PP source rank 对所有 engine GPU 建一个组”。

### 2.3 update_weights 用 pause/flush 包围权重 broadcast

问题与约束：
- rollout engine 正在生成时更新权重会破坏请求一致性；压缩量化模型还需要在加载前后做处理。

设计选择：
- rank0 先 pause 所有 engines 并 flush cache；所有 rank barrier 后发送权重；rank0 做量化后处理并 continue generation；最后再 barrier。

Explain：
`weight_version` 先自增，作为本次同步版本。compressed-tensors 量化在发送前调用 `post_process_weights(restore_weights_before_load=True)`，发送后再 `post_process_quantization=True`。

来源：slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py L102-L134

Code：

```python
@torch.no_grad()
def update_weights(self) -> None:
    self.weight_version += 1

    if dist.get_rank() == 0:
        ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
        ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])

        if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
            post_process_weights(
                restore_weights_before_load=True,
                post_process_quantization=False,
                rollout_engines=self.rollout_engines,
            )
    dist.barrier(group=get_gloo_group())

    pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_pp_src_rank else None
    self._send_weights(pbar)

    if dist.get_rank() == 0:
        if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
            post_process_weights(
                restore_weights_before_load=False,
                post_process_quantization=True,
                rollout_engines=self.rollout_engines,
            )
        ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
    dist.barrier(group=get_gloo_group())
```

代码逻辑：
- pause/flush 只由全局 rank0 触发。
- gloo barrier 保证所有训练 rank 在发送前对齐。
- `_send_weights` 在所有 rank 上调用，但只有 PP source 产生并 broadcast chunk。
- continue 后再 barrier，确保所有 rank完成更新周期。

为什么这样写：
- pause/flush 与 NCCL payload 分离，控制面走 Ray，数据面走 NCCL。
- KV cache flush 避免旧权重缓存参与新权重生成。

不变量与失败模式：
- 所有训练 rank 都必须进入同一 barrier 序列。
- 若 pause 后 broadcast 失败，engine 可能停留在 paused 状态，需要上层恢复。

Comment：
这段定义了权重更新期间 rollout engine 的可服务状态切换。

### 2.4 _send_weights 分非 expert 与 expert 两趟发送

问题与约束：
- 普通参数只需要 TP gather；MoE expert 参数还需要跨 EP rank 收集完整 expert 集合，二者不能混在同一个简单迭代器里。

设计选择：
- `_send_weights` 依次遍历 `_iter_non_expert_chunks()` 和 `_iter_expert_chunks()`，每个 chunk 先触发 `_on_chunk` hook，再调用 `_update_bucket_weights_from_distributed`，每趟结束后 gloo barrier。

Explain：
非 expert 和 expert 分两趟可以让各自的 gather/convert 逻辑保持独立。barrier 保证所有 rank 都完成一类参数后再进入下一类参数。

来源：slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py L136-L146

Code：

```python
def _send_weights(self, pbar: tqdm | None) -> None:
    for chunk_iter in (self._iter_non_expert_chunks(), self._iter_expert_chunks()):
        for hf_chunk in chunk_iter:
            self._on_chunk(hf_chunk)
            self._update_bucket_weights_from_distributed(hf_chunk, pbar=pbar)
        dist.barrier(group=get_gloo_group())
```

代码逻辑：
- 两个 iterator 都产出 HF named tensors chunk。
- `_on_chunk` 默认空实现，子类可插入 delta 行为。
- 每个 chunk 都经过统一 broadcast 函数。

为什么这样写：
- expert gather 需要 EP 语义，普通参数不应该为此承担额外通信。
- barrier 划分阶段，降低不同 rank 进入不同 collective 顺序的风险。

不变量与失败模式：
- 所有 rank 必须以相同顺序遍历 non-expert 和 expert 阶段。
- 子类 `_on_chunk` 不应改变 chunk 顺序或破坏 collective 对齐。

Comment：
这是 NCCL 路径的主循环，真正的数据发送都从这里进入。

### 2.5 _iter_non_expert_chunks 做 TP gather、HF convert 和分桶

问题与约束：
- 非 expert 参数可能被 TP 切分，SGLang engine 需要 HF 格式权重，单次 broadcast 又不能超过 buffer 限制。

设计选择：
- 遍历全局命名参数，跳过 `.experts.`，每个参数先 `all_gather_param`，PP source rank 转 HF 并按 `update_weight_buffer_size` 累积分桶。

Explain：
非 PP source rank 仍然执行 `all_gather_param` 参与 collective，但不生成 HF chunk。PP source 根据转换后 tensor 的真实字节数决定何时 yield bucket。

来源：slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py L153-L176

Code：

```python
def _iter_non_expert_chunks(self) -> Iterator[list[tuple[str, torch.Tensor]]]:
    buffer_size = 0
    buffer: list[tuple[str, torch.Tensor]] = []
    for name, param in named_params_and_buffers(self.args, self.model):
        if ".experts." in name:
            continue
        param = all_gather_param(name, param)
        if not self._is_pp_src_rank:
            continue
        hf_chunk = convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
        chunk_bytes = sum(t.numel() * t.element_size() for _, t in hf_chunk)
        if buffer and buffer_size + chunk_bytes > self.args.update_weight_buffer_size:
            yield buffer
            buffer = []
            buffer_size = 0
        buffer.extend(hf_chunk)
        buffer_size += chunk_bytes
    if buffer:
        yield buffer
```

代码逻辑：
- `.experts.` 参数留给 expert iterator。
- `all_gather_param` 在所有相关 rank 上执行。
- 只有 PP source 做 HF 转换和 bucket 累积。

为什么这样写：
- TP collective 需要所有 TP rank 参与，即使最终只有 TP rank0 发送。
- 按转换后 HF tensor 计量 bucket，更贴近实际 broadcast payload。

不变量与失败模式：
- `named_params_and_buffers` 输出顺序必须在各 rank 一致。
- `convert_to_hf` 可能把一个 Megatron 参数拆成多个 HF tensor，bucket 要按拆分后大小计算。

Comment：
非 expert 路径是“TP gather 后由 PP source 发送”的标准路径。

### 2.6 _iter_expert_chunks 先按 EP world size 预估 bucket

问题与约束：
- MoE expert 权重被 EP 分片，PP source 发送前必须收齐所有 EP rank 的 expert 参数。

设计选择：
- expert iterator 先对每个 expert 参数做 TP gather，再按 `param_size × expert_model_parallel_world_size` 估算 EP gather 后大小，超过阈值就触发 `_ep_gather_and_convert`。

Explain：
batch 里保存 `(name, param)`，当估算大小超过 buffer 限制时，先收集当前 batch，再清空继续。末尾剩余 batch 也会被 gather/convert。

来源：slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py L178-L203

Code：

```python
def _iter_expert_chunks(self) -> Iterator[list[tuple[str, torch.Tensor]]]:
    params = ((n, p) for n, p in named_params_and_buffers(self.args, self.model) if ".experts." in n)
    buffer_size = 0
    batch: list[tuple[str, torch.Tensor]] = []
    for name, param in params:
        param = all_gather_param(name, param)
        param_size = param.numel() * param.element_size()
        if (
            buffer_size + param_size
        ) * mpu.get_expert_model_parallel_world_size() > self.args.update_weight_buffer_size:
            hf_chunk = self._ep_gather_and_convert(batch)
            if hf_chunk:
                yield hf_chunk
            batch = []
            buffer_size = 0
        batch.append((name, param))
        buffer_size += param_size
    if batch:
        hf_chunk = self._ep_gather_and_convert(batch)
        if hf_chunk:
            yield hf_chunk
```

代码逻辑：
- expert 参数筛选基于名称包含 `.experts.`。
- TP gather 仍先执行，因为 expert tensor 也可能 TP 切分。
- EP 后总大小按 world size 放大估算。

为什么这样写：
- 只有 EP gather 后才是完整 expert 权重集合，分桶必须考虑放大后的大小。
- 将 expert 路径独立出来，可以避免普通参数为 EP 通信付成本。

不变量与失败模式：
- EP world size 必须正确反映 expert 分片数量。
- 若单个 expert 参数放大后超过 buffer size，仍会形成超限 chunk，需要上层配置足够大。

Comment：
expert 路径的关键差异是“TP gather 后还要 EP gather”。

### 2.7 _ep_gather_and_convert 对齐名称并异步 all_gather expert 参数

问题与约束：
- 不同 EP rank 持有不同 expert 分片；直接按本 rank 名称转换会丢 expert 或顺序错乱。

设计选择：
- 先 `all_gather_object` 收集各 EP rank 的 name 列表并校验长度，再对每个 tensor 发起 async `dist.all_gather`，PP source rank 汇总所有 EP rank 的 `(name, tensor)` 后转 HF。

Explain：
非 PP source rank 参与 EP all_gather，但在等待完成后返回空列表。`named_tensors.clear()` 释放 batch 引用，减少显存占用。

来源：slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py L204-L238

Code：

```python
def _ep_gather_and_convert(self, named_tensors: list[tuple[str, torch.Tensor]]) -> list[tuple[str, torch.Tensor]]:
    names = [name for name, _ in named_tensors]
    all_names = [None] * mpu.get_expert_model_parallel_world_size()
    dist.all_gather_object(all_names, names, group=mpu.get_expert_model_parallel_group())

    for names in all_names:
        assert len(named_tensors) == len(names), f"mismatch names length: {len(named_tensors)} != {len(names)}"

    all_gathered_params = [[] for _ in range(mpu.get_expert_model_parallel_world_size())]
    handles = []
    for i, (_name, param) in enumerate(named_tensors):
        params = [
            torch.empty_like(param.data, device=torch.cuda.current_device())
            for _ in range(mpu.get_expert_model_parallel_world_size())
        ]
        handle = dist.all_gather(params, param.data, group=mpu.get_expert_model_parallel_group(), async_op=True)
        handles.append(handle)
        for ep_rank, names in enumerate(all_names):
            all_gathered_params[ep_rank].append((names[i], params[ep_rank]))
    for handle in handles:
        handle.wait()

    named_tensors.clear()
    if not self._is_pp_src_rank:
        return []

    all_gathered_params = sum(all_gathered_params, [])
    converted_hf_tensors = []
    for name, param in all_gathered_params:
        converted_hf_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
    return converted_hf_tensors
```

代码逻辑：
- name list 先跨 EP rank 对齐。
- 每个 tensor 都异步 all_gather 到 `params` 列表。
- PP source 展平所有 EP rank 的结果并转 HF。

为什么这样写：
- expert id 在各 EP rank 上不同，必须携带 name 才能保持 HF 转换正确。
- 异步 all_gather 批量 wait 能让多个 expert 参数通信重叠。

不变量与失败模式：
- 所有 EP rank 的 batch 长度必须一致。
- name 与 tensor 顺序必须保持一一对应。
- 非 PP source 返回空 chunk，不应进入 broadcast。

Comment：
这段是 MoE 权重同步正确性的核心。

### 2.8 _update_bucket_weights_from_distributed 用 Ray lock 串行 bucket broadcast

问题与约束：
- 多 PP stage 或多个 bucket 同时向同一批 rollout engines 发 NCCL broadcast，collective 顺序不一致会导致死锁。

设计选择：
- 每个 bucket broadcast 前 spin-acquire `rollout_engine_lock`，broadcast 完并等待 engine refs 后清空 bucket、释放 lock、更新进度条。

Explain：
锁由 RolloutManager 提供，跨 PP source rank 共享。`update_weights_from_distributed` 返回 engine 侧 Ray refs，必须 `ray.get(refs)` 确认 engine 侧加载完成后再释放。

来源：slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py L240-L265

Code：

```python
def _update_bucket_weights_from_distributed(
    self,
    converted_named_tensors: list[tuple[str, torch.Tensor]],
    pbar: tqdm | None = None,
    load_format: str | None = None,
) -> None:
    while not ray.get(self.rollout_engine_lock.acquire.remote()):
        time.sleep(0.1)

    refs = update_weights_from_distributed(
        self._group_name,
        self._model_update_groups,
        self.weight_version,
        self.rollout_engines,
        converted_named_tensors,
        load_format=load_format,
    )

    ray.get(refs)
    converted_named_tensors.clear()
    ray.get(self.rollout_engine_lock.release.remote())
    pbar.update(1)
```

代码逻辑：
- acquire 失败时 sleep 后重试。
- metadata 和 NCCL broadcast 在锁内执行。
- engine refs 完成后才释放。
- bucket tensor 列表清空以释放引用。

为什么这样写：
- NCCL collective 对调用顺序敏感，Ray lock 是跨进程排序点。
- 等待 engine refs 能保证 engine 侧完成 metadata 接收和 tensor load。

不变量与失败模式：
- `pbar` 在 PP source 上应非空；若传 None 会在 `pbar.update` 失败。
- 发生异常时当前代码没有 finally release lock，需要上层故障恢复。

Comment：
锁不是性能优化，而是避免 NCCL 死锁的正确性约束。

### 2.9 connect_rollout_engines_from_distributed 建立训练 rank0 + engine GPUs 的 NCCL group

问题与约束：
- rollout engines 可能每个占用不同 GPU 数；训练 rank0 和所有 engine GPU 必须加入同一个 NCCL process group。

设计选择：
- 训练端选取本机 IP 和空闲端口，world size 设为 `sum(engine_gpu_counts)+1`，每个 engine 通过 Ray 远程初始化 group，训练 rank0 以 rank 0 调用 `init_process_group`。

Explain：
`rank_offset` 使用 engine GPU count 的 cumulative sum，让每个 engine 在 NCCL group 中占据连续 rank 区间。`ray.get(refs)` 等待所有 engine 完成初始化。

来源：slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py L269-L314

Code：

```python
def connect_rollout_engines_from_distributed(
    args: Namespace,
    group_name: str,
    rollout_engines: Sequence[ActorHandle],
    engine_gpu_counts: Sequence[int] | None = None,
) -> dist.ProcessGroup:
    if engine_gpu_counts is None:
        engine_gpu_counts = [args.rollout_num_gpus_per_engine] * len(rollout_engines)

    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    world_size = sum(engine_gpu_counts) + 1

    cumulative = [0]
    for c in engine_gpu_counts:
        cumulative.append(cumulative[-1] + c)

    refs = [
        engine.init_weights_update_group.remote(
            master_address=master_address,
            master_port=master_port,
            rank_offset=cumulative[i] + 1,
            world_size=world_size,
            group_name=group_name,
            backend="nccl",
        )
        for i, engine in enumerate(rollout_engines)
    ]
    model_update_groups = init_process_group(
        backend="nccl",
        init_method=f"tcp://{_wrap_ipv6(master_address)}:{master_port}",
        world_size=world_size,
        rank=0,
        group_name=group_name,
    )
    ray.get(refs)
    return model_update_groups
```

代码逻辑：
- 没传 GPU count 时使用统一 `rollout_num_gpus_per_engine`。
- 训练 rank0 占 NCCL rank 0。
- engine ranks 从 1 开始按 offset 排布。

为什么这样写：
- 异构 engine TP size 需要 per-engine GPU count，而不是固定 rank stride。
- 训练端和 engine 端必须使用相同 master 地址、端口、world size 和 group name。

不变量与失败模式：
- master port 必须能被 engine 访问。
- engine 初始化失败会让 `ray.get(refs)` 抛错，group 不可用。

Comment：
这段定义了 NCCL 数据面拓扑。

### 2.10 update_weights_from_distributed 用 Ray 发 metadata、NCCL 发 tensor

问题与约束：
- engine 侧需要先知道本 bucket 的 tensor name、dtype、shape、version 和 load_format，才能在 NCCL group 中接收 broadcast。

设计选择：
- 对每个 engine 先 Ray 调用 `update_weights_from_distributed.remote(...)` 传 metadata，再由训练 rank0 对 bucket 内每个 tensor 发起 async NCCL broadcast。

Explain：
Ray refs 表示 engine 侧接收和加载流程；NCCL handles 表示训练端 tensor broadcast。函数先 wait 所有 broadcast handle，再返回 Ray refs 给调用者等待。

来源：slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py L326-L355

Code：

```python
def update_weights_from_distributed(
    group_name: str,
    group: dist.ProcessGroup,
    weight_version: int,
    rollout_engines: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
    load_format: str | None = None,
) -> list[ObjectRef]:
    refs = [
        engine.update_weights_from_distributed.remote(
            names=[name for name, _ in converted_named_tensors],
            dtypes=[param.dtype for _, param in converted_named_tensors],
            shapes=[param.shape for _, param in converted_named_tensors],
            group_name=group_name,
            weight_version=str(weight_version),
            load_format=load_format,
        )
        for engine in rollout_engines
    ]

    handles = []
    for _, param in converted_named_tensors:
        handles.append(dist.broadcast(param.data, 0, group=group, async_op=True))
    for handle in handles:
        handle.wait()

    return refs
```

代码逻辑：
- metadata 列表按 bucket tensor 顺序生成。
- 每个 tensor 一次 broadcast。
- broadcast 完成后返回 engine Ray refs。

为什么这样写：
- Ray 适合传小 metadata 和调度 engine 方法；大 tensor 走 NCCL 避免经 Ray object store。
- dtype/shape 在 engine 侧分配接收 buffer 前必须已知。

不变量与失败模式：
- metadata 顺序必须与 broadcast 顺序完全一致。
- engine 侧必须加入同名 NCCL group，并按相同顺序 recv。

Comment：
这是控制面和数据面分离的最小实现。

---

## 3. 参数命名与 gather

### 3.1 all_gather_param 把 TP shard 拼成完整 Megatron tensor

问题与约束：
- Megatron 参数可能按普通 TP 或 expert TP 切分；HF 转换需要完整 tensor。

设计选择：
- 根据参数名选择 expert TP group 或普通 TP group；非 TP、duplicated、expert_bias 直接返回；TP 参数 all_gather 后按 partition dim concat，并处理 GLU 和 grouped MoE 的维度修正。

Explain：
`linear_fc1` 的 GLU 权重需要先按 dim0 chunk 再重新排列；`linear_fc2.weight` 在 grouped MoE bug 场景下把 partition dim 从 0 修正为 1。

来源：slime/backends/megatron_utils/update_weight/common.py L15-L50

Code：

```python
def all_gather_param(name: str, param: torch.nn.Parameter) -> torch.Tensor:
    if "expert_bias" in name:
        return param

    assert hasattr(param, "tensor_model_parallel"), f"{name} does not have tensor_model_parallel attribute"
    if not param.tensor_model_parallel or getattr(param, "parallel_mode", None) == "duplicated":
        return param.data

    if ".experts." in name:
        tp_size = mpu.get_expert_tensor_parallel_world_size()
        tp_group = mpu.get_expert_tensor_parallel_group()
    else:
        tp_size = mpu.get_tensor_model_parallel_world_size()
        tp_group = mpu.get_tensor_model_parallel_group()

    param_partitions = [torch.empty_like(param.data) for _ in range(tp_size)]
    dist.all_gather(param_partitions, param.data, group=tp_group)
    partition_dim = param.partition_dim
    if "linear_fc1.weight" in name or "linear_fc1.bias" in name:
        param_partitions = [p.chunk(2, dim=0) for p in param_partitions]
        param_partitions = [p[0] for p in param_partitions] + [p[1] for p in param_partitions]
    if "linear_fc2.weight" in name:
        if partition_dim == 0:
            partition_dim = 1
    param = torch.cat(param_partitions, dim=partition_dim)
    return param
```

代码逻辑：
- expert 参数使用 expert TP group。
- duplicated 参数不需要 gather。
- GLU 和 MoE 特例在 concat 前后修正。

为什么这样写：
- HF 权重格式通常不是 Megatron TP shard，必须还原完整张量后转换。
- expert 参数的 TP group 与普通参数不同，不能混用。

不变量与失败模式：
- TP 参数必须带 `tensor_model_parallel/partition_dim/partition_stride` 属性。
- 未支持的 partition_stride 会触发 assert。

Comment：
这是 distributed 非 expert/expert 两条路径都依赖的基础 gather 函数。

### 3.2 all_gather_params_async 批量发起 TP all_gather 再统一 wait

问题与约束：
- Direct iterator 需要一次处理 bucket 内多个参数；逐个同步 all_gather 会减少通信重叠。

设计选择：
- 函数分三阶段：先为所有 TP 参数发起 async all_gather，再统一 wait，最后 concat 和修正 GLU/MoE 特例。

Explain：
expert_bias、非 TP、duplicated 参数不发 collective，但仍作为任务加入结果列表，保证输出顺序和输入 param_infos 一致。

来源：slime/backends/megatron_utils/update_weight/common.py L53-L115

Code：

```python
def all_gather_params_async(
    param_infos_and_params: list[tuple[ParamInfo, torch.Tensor]],
) -> list[torch.Tensor]:
    gather_tasks = []
    handles = []

    for info, param in param_infos_and_params:
        if "expert_bias" in info.name:
            gather_tasks.append((info, param, None, None, None))
            handles.append(None)
        elif not param.tensor_model_parallel or getattr(param, "parallel_mode", None) == "duplicated":
            gather_tasks.append((info, param.data, None, None, None))
            handles.append(None)
        else:
            if ".experts." in info.name:
                tp_size = mpu.get_expert_tensor_parallel_world_size()
                tp_group = mpu.get_expert_tensor_parallel_group()
            else:
                tp_size = mpu.get_tensor_model_parallel_world_size()
                tp_group = mpu.get_tensor_model_parallel_group()
            param_partitions = [torch.empty_like(param.data) for _ in range(tp_size)]
            handle = dist.all_gather(param_partitions, param.data, group=tp_group, async_op=True)
            gather_tasks.append((info, None, handle, param_partitions, param.partition_dim))
            handles.append(handle)

    for handle in handles:
        if handle is not None:
            handle.wait()

    gathered_params = []
    for info, direct_param, handle, param_partitions, partition_dim in gather_tasks:
        if handle is None:
            param = direct_param
        else:
            if "linear_fc1.weight" in info.name or "linear_fc1.bias" in info.name:
                param_partitions = [p.chunk(2, dim=0) for p in param_partitions]
                param_partitions = [p[0] for p in param_partitions] + [p[1] for p in param_partitions]
            if "linear_fc2.weight" in info.name:
                if partition_dim == 0:
                    partition_dim = 1
            param = torch.cat(param_partitions, dim=partition_dim)
        gathered_params.append(param)
    return gathered_params
```

代码逻辑：
- 第一阶段只提交 collective。
- 第二阶段集中等待。
- 第三阶段保持输入顺序生成完整 tensor。

为什么这样写：
- bucket 内参数多时，异步提交能提高 NCCL overlap。
- Direct iterator 使用 ParamInfo 保存 attrs，因此不依赖原始 parameter 对象完整上下文。

不变量与失败模式：
- 输入顺序必须与后续 HF 转换的 ParamInfo 顺序一致。
- 所有 rank 必须为同一 bucket 调用相同 collective 集合。

Comment：
distributed updater 用单参数 `all_gather_param`，Direct iterator 用批量 async 版本。

### 3.3 named_params_and_buffers 生成跨 PP/EP 一致的全局名称

问题与约束：
- Megatron 本地层号和 expert id 是分片视角；HF/SGLang engine 需要全局一致的参数名。

设计选择：
- `named_params_and_buffers` 默认走 `_named_params_and_buffers_global`，根据 PP layer offset 和 EP expert offset 重写 decoder layer 与 expert 参数名；vanilla 模式保留 vp stage 前缀。

Explain：
函数还支持 `translate_gpu_to_cpu`，可用 torch_memory_saver 的 CPU backup 替换 tensor。global 路径处理 decoder layers、MTP layers 和 expert_bias buffer。

来源：slime/backends/megatron_utils/update_weight/common.py L118-L219

Code：

```python
def named_params_and_buffers(
    args: Namespace,
    model: Sequence[torch.nn.Module],
    convert_to_global_name: bool = True,
    translate_gpu_to_cpu: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    if convert_to_global_name:
        ans = _named_params_and_buffers_global(args, model)
    else:
        ans = _named_params_and_buffers_vanilla(model)

    if translate_gpu_to_cpu:
        ans = ((name, _maybe_get_cpu_backup(tensor)) for name, tensor in ans)

    return ans

def _named_params_and_buffers_global(
    args: Namespace, model: Sequence[torch.nn.Module]
) -> Iterator[tuple[str, torch.Tensor]]:
    ep_size = mpu.get_expert_model_parallel_world_size()
    ep_rank = mpu.get_expert_model_parallel_rank()
    if args.num_experts:
        expert_offset = ep_rank * args.num_experts // ep_size

    for vp_stage, model_module in enumerate(model):
        layer_offset = get_transformer_layer_offset(model_module.config, vp_stage)
        for name, param in model_module.named_parameters():
            if not name.startswith("module.module."):
                name = "module." + name
            decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
            match = re.match(decoder_layers_pattern, name)
            if not match:
                yield name, param
                continue
            layer_idx, rest = match.groups()
            layer_idx = int(layer_idx) + layer_offset
            expert_pattern = r"mlp\.experts\.(.+)\.(weight|bias)(\d+)"
            match = re.match(expert_pattern, rest)
            if match:
                rest, param_type, expert_idx = match.groups()
                expert_idx = int(expert_idx) + expert_offset
                yield f"module.module.decoder.layers.{layer_idx}.mlp.experts.{rest}.{param_type}{expert_idx}", param
            else:
                yield f"module.module.decoder.layers.{layer_idx}.{rest}", param
```

代码逻辑：
- PP layer offset 把本地 layer index 转成全局 layer index。
- EP expert offset 把本地 expert id 转成全局 expert id。
- 非 decoder layer 参数原样或按其他分支处理。

为什么这样写：
- engine 侧按 HF/global name 更新权重，本地 Megatron name 不能直接用。
- 全 rank 名称一致是后续 bucket 对齐和 NCCL 顺序一致的前提。

不变量与失败模式：
- layer offset 与 VP/PP 配置必须正确。
- `args.num_experts` 缺失时 expert_offset 未定义，依赖调用配置保证 MoE 场景提供该字段。

Comment：
权重同步的很多 silent mismatch 都会发生在命名层，先读这里很有必要。

---

## 4. Direct iterator 的分桶参考

### 4.1 HfWeightIteratorDirect 按预计算 bucket 产出 HF chunk

问题与约束：
- save_hf 或 direct 路径也需要把 Megatron local weights 转成 HF named tensors，并保持和 distributed 路径相似的 bucket 语义。

设计选择：
- 初始化时预计算 `megatron_local_param_info_buckets`；`get_hf_weight_chunks` 对每个 bucket 先还原 Megatron full params，再转 HF 并 yield。

Explain：
Direct iterator 不直接向 engine broadcast，而是复用同样的参数收集、TP gather 和 HF 转换逻辑，为其他权重导出/更新路径提供 chunk 迭代。

来源：slime/backends/megatron_utils/update_weight/hf_weight_iterator_direct.py L19-L41

Code：

```python
class HfWeightIteratorDirect(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.megatron_local_param_info_buckets = _get_megatron_local_param_info_buckets(self.args, self.model)

    def get_hf_weight_chunks(self, megatron_local_weights, progress_desc: str = "Update weights"):
        rank = dist.get_rank()

        for megatron_local_param_infos in tqdm(
            self.megatron_local_param_info_buckets, disable=rank != 0, desc=progress_desc
        ):
            megatron_full_params = _get_megatron_full_params(megatron_local_param_infos, megatron_local_weights)
            hf_named_tensors = self._convert_to_hf_named_tensors(megatron_full_params, megatron_local_param_infos)
            yield hf_named_tensors
            del megatron_full_params

    def _convert_to_hf_named_tensors(self, megatron_full_params: Sequence[torch.Tensor], param_infos: list[ParamInfo]):
        hf_named_tensors = []
        for info, param in zip(param_infos, megatron_full_params, strict=False):
            hf_named_tensors.extend(
                convert_to_hf(self.args, self.model_name, info.name, param, self.quantization_config)
            )
        return hf_named_tensors
```

代码逻辑：
- bucket 元信息在构造期确定。
- 每个 bucket 转成完整 Megatron 参数后再转 HF。
- rank0 显示进度条，其他 rank 静默参与 collective。

为什么这样写：
- 预计算 bucket 可在迭代时避免重复扫描模型。
- Direct 与 distributed 共用转换函数，减少格式差异。

不变量与失败模式：
- `megatron_local_weights` 必须包含每个 ParamInfo 的源权重。
- bucket 内所有 rank 必须调用相同 collective 顺序。

Comment：
Direct iterator 是理解分桶和转换语义的参考实现。

### 4.2 _get_megatron_full_params 广播 PP/EP 源参数后批量 TP gather

问题与约束：
- Direct 路径的某个 rank 不一定持有 bucket 内每个参数；需要先从源 rank 广播到当前并行组，再按 TP 拼完整参数。

设计选择：
- 对每个 ParamInfo 创建本地 parameter 或空 tensor；跨 PP group 和 EP group async broadcast；恢复 ParamInfo attrs 后调用 `all_gather_params_async`。

Explain：
PP broadcast 用 `info.src_rank` 作为源。EP broadcast 只处理 `.experts.` 参数，若源 rank 不在当前 expert group，则使用当前 rank 作为 src。最后把 ParamInfo 的 tensor parallel attrs 写回 param，供 async gather 使用。

来源：slime/backends/megatron_utils/update_weight/hf_weight_iterator_direct.py L44-L105

Code：

```python
def _get_megatron_full_params(
    megatron_local_param_infos: Sequence[ParamInfo],
    megatron_local_weights,
) -> Sequence[torch.Tensor]:
    monkey_patch_torch_reductions()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    ep_size = mpu.get_expert_model_parallel_world_size()
    rank = dist.get_rank()
    params = []
    for info in megatron_local_param_infos:
        if dist.get_rank() == info.src_rank:
            params.append(
                torch.nn.Parameter(
                    megatron_local_weights[info.name].to(device=torch.cuda.current_device(), non_blocking=True),
                    requires_grad=False,
                )
            )
        else:
            params.append(torch.empty(info.shape, dtype=info.dtype, device=torch.cuda.current_device()))
    torch.cuda.synchronize()

    if pp_size > 1:
        handles = []
        for info, param in zip(megatron_local_param_infos, params, strict=False):
            if info.src_rank in dist.get_process_group_ranks(mpu.get_pipeline_model_parallel_group()):
                handles.append(
                    torch.distributed.broadcast(
                        param, src=info.src_rank, group=mpu.get_pipeline_model_parallel_group(), async_op=True
                    )
                )
        for handle in handles:
            handle.wait()

    if ep_size > 1:
        handles = []
        for info, param in zip(megatron_local_param_infos, params, strict=False):
            if ".experts." in info.name:
                src_rank = (
                    info.src_rank
                    if info.src_rank in dist.get_process_group_ranks(mpu.get_expert_model_parallel_group())
                    else rank
                )
                handles.append(torch.distributed.broadcast(param, src=src_rank, group=mpu.get_expert_model_parallel_group(), async_op=True))
        for handle in handles:
            handle.wait()

    for info, param in zip(megatron_local_param_infos, params, strict=False):
        for key, value in info.attrs.items():
            setattr(param, key, value)

    gathered_params = all_gather_params_async(list(zip(megatron_local_param_infos, params, strict=False)))
    return gathered_params
```

代码逻辑：
- 源 rank 持有真实权重，其他 rank 先分配空 tensor。
- PP/EP broadcast 让每个参与 rank 获得对应 shard。
- ParamInfo attrs 恢复 TP gather 所需属性。
- 最后批量 all_gather 得到完整 tensor。

为什么这样写：
- Direct 路径从本地权重字典出发，不一定和当前 rank 参数对象一一对应。
- 先广播源 shard，再 TP gather，能统一处理 PP/EP/TP 组合。

不变量与失败模式：
- ParamInfo 的 `src_rank/shape/dtype/attrs` 必须准确。
- broadcast group 内各 rank 必须对相同参数集合调用一致 collective。

Comment：
这段把 ParamInfo 从“元数据”变成实际可转换的完整 Megatron tensor。

### 4.3 _get_megatron_local_param_info_buckets 按 full size 估算 bucket

问题与约束：
- bucket 限制应反映 all_gather 后完整参数大小，而不是本地 shard 大小。

设计选择：
- 对每个 ParamInfo，普通参数乘 regular TP size，expert 参数乘 expert TP size；累计超过 `update_weight_buffer_size` 时开启新 bucket。

Explain：
函数先从 `_get_megatron_local_param_infos` 得到所有 rank 一致的 ParamInfo 列表，再按参数 full size 分桶。空 bucket 初始存在，只有当前 bucket 已有参数时才切换。

来源：slime/backends/megatron_utils/update_weight/hf_weight_iterator_direct.py L108-L135

Code：

```python
def _get_megatron_local_param_info_buckets(args: Namespace, model: Sequence[torch.nn.Module]) -> list[list[ParamInfo]]:
    param_infos = _get_megatron_local_param_infos(args, model)
    param_info_buckets = [[]]
    buffer_size = 0

    for info in param_infos:
        if ".experts." in info.name:
            tp_size = mpu.get_expert_tensor_parallel_world_size()
        else:
            tp_size = mpu.get_tensor_model_parallel_world_size()

        param_size = info.size * tp_size

        if buffer_size + param_size > args.update_weight_buffer_size and len(param_info_buckets[-1]) > 0:
            param_info_buckets.append([])
            buffer_size = 0

        param_info_buckets[-1].append(info)
        buffer_size += param_size

    return param_info_buckets
```

代码逻辑：
- `info.size` 是本地 shard 字节数。
- 乘 TP size 得到 full parameter 估算。
- bucket 不为空时才切新 bucket，避免产生前置空 bucket。

为什么这样写：
- 后续转换和 broadcast 处理的是 full tensor，按 shard 大小分桶会低估内存和通信。
- expert TP 与普通 TP 分开计算，符合 MoE 参数布局。

不变量与失败模式：
- `ParamInfo.size` 必须是字节数而非元素数。
- 单个参数超过 buffer size 时会独占一个超限 bucket。

Comment：
Direct 分桶逻辑解释了 `update_weight_buffer_size` 应按完整权重 payload 理解。

### 4.4 _get_megatron_local_param_infos 校验全 rank 参数元数据一致

问题与约束：
- 分布式权重导出要求所有 rank 对参数顺序、shape 和 dtype 达成一致，否则后续 collective 和转换会 silent mismatch。

设计选择：
- 每个 rank 收集本地 ParamInfo，经 PP/EP `all_gather_object` 合并后排序，再通过 gloo all_gather_object 校验所有 rank 的列表一致。

Explain：
PP 合并时同名参数选择更小 `src_rank`，用于处理 MTP virtual PP 造成的重复。EP 合并时补齐其他 expert rank 的参数，并把 src_rank 改成 expert group 内 rank。

来源：slime/backends/megatron_utils/update_weight/hf_weight_iterator_direct.py L138-L211

Code：

```python
def _get_megatron_local_param_infos(args: Namespace, model: Sequence[torch.nn.Module]) -> list[ParamInfo]:
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    ep_size = mpu.get_expert_model_parallel_world_size()

    param_infos = {}
    rank = dist.get_rank()
    for name, param in named_params_and_buffers(args, model):
        param_infos[name] = ParamInfo(
            name=name,
            dtype=param.dtype,
            shape=param.shape,
            attrs={
                "tensor_model_parallel": getattr(param, "tensor_model_parallel", False),
                "partition_dim": getattr(param, "partition_dim", -1),
                "partition_stride": getattr(param, "partition_stride", 1),
                "parallel_mode": getattr(param, "parallel_mode", None),
            },
            size=param.numel() * param.element_size(),
            src_rank=rank,
        )

    if pp_size > 1:
        param_infos_list = [None] * pp_size
        dist.all_gather_object(
            obj=(rank, param_infos), object_list=param_infos_list, group=mpu.get_pipeline_model_parallel_group()
        )
        for src_rank, infos in param_infos_list:
            for name, info in infos.items():
                if name in param_infos:
                    old_info = param_infos[name]
                    if old_info.src_rank > src_rank:
                        param_infos[name] = info
                else:
                    param_infos[name] = info

    if ep_size > 1:
        param_infos_list = [None] * ep_size
        dist.all_gather_object(
            obj=(rank, param_infos), object_list=param_infos_list, group=mpu.get_expert_model_parallel_group()
        )
        for src_rank, infos in param_infos_list:
            for name, info in infos.items():
                if name not in param_infos:
                    info = dataclasses.replace(info, src_rank=src_rank)
                    param_infos[name] = info

    param_infos = sorted(param_infos.values(), key=lambda info: info.name)
    all_param_info_list = [None] * dist.get_world_size()
    dist.all_gather_object(obj=param_infos, object_list=all_param_info_list, group=get_gloo_group())
    for i, param_info in enumerate(param_infos):
        for infos in all_param_info_list:
            assert infos[i].name == param_info.name
            assert infos[i].shape == param_info.shape
            assert infos[i].dtype == param_info.dtype

    return param_infos
```

代码逻辑：
- 本地 ParamInfo 保存 name、dtype、shape、TP attrs、size 和源 rank。
- PP/EP 合并补齐跨并行维度参数。
- 排序后全 rank 校验 name/shape/dtype。

为什么这样写：
- bucket 和 collective 顺序依赖每个 rank 持有完全相同的 ParamInfo 列表。
- 提前 assert 比后续 NCCL 死锁或错误权重更容易排查。

不变量与失败模式：
- 所有 rank 的 ParamInfo 列表长度必须一致；源码逐项比较时默认长度已经一致。
- 参数名、shape 或 dtype 任一不一致都会触发 assert。

Comment：
这是 Direct iterator 的完整性闸门，也说明 distributed 路径为什么重视全局命名。
