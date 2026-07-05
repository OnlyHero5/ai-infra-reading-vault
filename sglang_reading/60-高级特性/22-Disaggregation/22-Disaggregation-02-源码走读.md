---
type: batch-doc
module: 22-Disaggregation
batch: "22"
doc_type: walkthrough
title: "PD 分离 · 源码走读"
tags:
 - sglang/batch/22
 - sglang/module/disaggregation
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-05
---
# PD 分离 · 源码走读

## 走读顺序

1. `disagg_service.py` - 启动 bootstrap 服务
2. `utils.py` - 传输状态、metadata gate 与跨 rank 同步
3. `prefill.py` - bootstrap 队列、metadata buffer 生命周期
4. `decode.py` - decode 侧 prealloc token pool
5. `decode_hicache_mixin.py` - HiCache prefix 恢复与传输 gating
6. `encode_server.py` - 多模态 encoder 独立服务

---

## 1. 传输完成不等于请求可 decode

### 1.1 `_apply_metadata_gate` 把 metadata 未落地的 Success 降级

问题与约束：
- PD 分离中 KV 传输和 metadata 写入不是同一个动作。
- decode 侧如果只看 `kv_receiver.poll()`，可能在 `bootstrap_room` 尚未写入时误判请求可执行。
- fake transfer 测试后端没有真实 metadata 交换，不能用同一个 gate 阻塞。

设计选择：
- 只处理已经是 `KVPoll.Success` 的请求。
- 对 fake transfer 直接跳过 gate。
- 读取 `metadata_buffers.bootstrap_room[metadata_buffer_index, 0]`，值为 0 时把状态改回 `Transferring`。

Explain：
这个函数把“KV bytes 到达”和“decode metadata 可见”之间的异步窗口显式建模。它不改变失败状态，也不主动推进传输，只是在状态汇总前修正过早的 Success。

来源：python/sglang/srt/disaggregation/utils.py L103-L118

Code：

```python
def _apply_metadata_gate(polls, decode_reqs, metadata_buffers, server_args) -> None:
    """Downgrade Success -> Transferring for requests whose metadata hasn't landed.

    Mutates `polls` in-place. Called before all-reduce so that MIN across TP
    ranks naturally prevents any rank from committing before all ranks are ready.
    """
    for i, poll_val in enumerate(polls):
        if poll_val == int(KVPoll.Success):
            decode_req = decode_reqs[i]
            if _is_fake_transfer(decode_req.req, server_args):
                continue
            actual_room = metadata_buffers.bootstrap_room[
                decode_req.metadata_buffer_index, 0
            ].item()
            if actual_room == 0:
                polls[i] = int(KVPoll.Transferring)
```

代码逻辑：
- 遍历每个 poll 结果。
- 只检查已经成功的传输项。
- 取出对应的 decode request。
- fake transfer 场景直接保留 Success。
- 从共享 metadata buffer 读取实际 room 标记。
- room 为 0 时把本地状态改成 Transferring。

为什么这样写：
- gate 放在 poll 结果上，比把 metadata 检查塞进各个 backend 更集中。
- fake backend 能复用同一调度路径，又不会被真实 metadata 条件卡住。
- 降级而不是报错，符合“传输未最终 ready”的可重试语义。

不变量与失败模式：
- `decode_req.metadata_buffer_index` 必须指向有效 metadata slot。
- `bootstrap_room` 的 0/非 0 语义必须由 prefill/decode 两端共同维护。
- 如果 metadata slot 泄漏或复用错误，decode 可能长期停在 Transferring 或读到其他请求的 room。

Comment：
PD decode 的 ready 条件至少包含 KV receiver 和 metadata gate 两部分。

### 1.2 `poll_and_all_reduce` 用 MIN 汇总所有 rank 的状态

问题与约束：
- Tensor parallel 或 pipeline 相关 rank 必须对同一批请求观察到一致的传输状态。
- 某个 rank 仍在传输时，其他 rank 不能先进入 decode。
- metadata gate 必须在跨 rank 汇总前生效，避免局部 Success 被错误扩散。

设计选择：
- 先调用 `_poll_with_failure_injection` 得到本 rank 状态。
- 当 decode request、metadata buffer 和 server args 都存在时应用 metadata gate。
- 将状态转成 CPU `uint8` tensor，通过 `dist.all_reduce(..., ReduceOp.MIN)` 汇总。

Explain：
`KVPoll` 状态被设计成可以用最小值表达“全局最保守状态”。因此任意 rank 仍未 ready，MIN 都会把整个请求保持在未 ready 状态。

来源：python/sglang/srt/disaggregation/utils.py L121-L140

Code：

```python
def poll_and_all_reduce(
    pollers,
    gloo_group: dist.ProcessGroup,
    decode_reqs=None,
    metadata_buffers: Optional[MetadataBuffers] = None,
    server_args: Optional[ServerArgs] = None,
):
    polls = _poll_with_failure_injection(pollers)
    if (
        decode_reqs is not None
        and metadata_buffers is not None
        and server_args is not None
    ):
        _apply_metadata_gate(polls, decode_reqs, metadata_buffers, server_args)
    tensor_to_reduce = torch.tensor(polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.MIN, group=gloo_group)
    return tensor_to_reduce.tolist()
```

代码逻辑：
- 对所有 poller 读取当前状态。
- 参数齐全时执行 metadata gate。
- 构造 CPU tensor。
- 在 gloo group 内执行 MIN all-reduce。
- 把汇总后的状态转回 Python list。

为什么这样写：
- CPU gloo 同步足够表达控制面状态，不占用 GPU 通信资源。
- MIN 操作让“任何一个 rank 未 ready”自然变成全局未 ready。
- metadata gate 前置，能让 metadata 缺失也参与同一个保守汇总。

不变量与失败模式：
- 所有 rank 的 poll 列表长度和请求顺序必须一致。
- `KVPoll` 数值顺序必须符合 MIN 语义。
- group 配置错误会导致状态不同步、阻塞或 collective mismatch。

Comment：
这是 PD 传输状态从单 rank 事实变成并行组共识的边界。

### 1.3 `poll_and_all_reduce_attn_cp_tp_group` 在 TP 和 CP 两层同步

问题与约束：
- Attention context parallel 部署下，一个 decode 请求横跨 TP 和 CP 两个维度。
- 只在 TP 组内同步不能覆盖 CP shard，只在 CP 组内同步也不能覆盖同一 CP 内的 TP shard。
- 所有 TP x CP 参与者必须对请求状态收敛到同一个值。

设计选择：
- 先复用 `poll_and_all_reduce` 在 attn-tp 组内同步。
- 再把 TP 汇总结果放入 tensor，在 attn-cp 组内做一次 MIN all-reduce。
- 返回 CP 维度汇总后的状态 list。

Explain：
这段把二维并行网格拆成两次一维 collective：先保证每个 CP shard 内 TP 一致，再保证各 CP shard 之间一致。

来源：python/sglang/srt/disaggregation/utils.py L143-L160

Code：

```python
def poll_and_all_reduce_attn_cp_tp_group(
    pollers,
    attn_cp_cpu_group: dist.ProcessGroup,
    attn_tp_cpu_group: dist.ProcessGroup,
):
    polls = poll_and_all_reduce(pollers, attn_tp_cpu_group)

    tensor_to_reduce = torch.tensor(polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(
        tensor_to_reduce,
        op=dist.ReduceOp.MIN,
        group=attn_cp_cpu_group,
    )
    return tensor_to_reduce.tolist()
```

代码逻辑：
- 在 attn TP CPU group 内 poll 并汇总。
- 将第一次汇总结果转成 CPU tensor。
- 在 attn CP CPU group 内继续做 MIN all-reduce。
- 返回 TP x CP 全局一致状态。

为什么这样写：
- 复用已有 TP 同步逻辑，避免为 CP 重新实现 poll 流程。
- 两层 MIN 保持与普通 TP 模式相同的保守 ready 语义。
- CPU group 适合传输状态这种小控制面数据。

不变量与失败模式：
- TP group 和 CP group 的 rank 拓扑必须和 attention 并行布局一致。
- 两次 collective 的调用顺序必须在所有相关 rank 上一致。
- 如果某个 CP shard 缺席，其他 shard 会在第二次 all-reduce 阻塞。

Comment：
CP 模式没有改变 PD 状态机，只是扩大了需要达成共识的 rank 集合。

## 2. 分阶段传输与 metadata 生命周期

### 2.1 `poll_and_all_reduce_with_staging` 在 poll 前推进 scatter

问题与约束：
- 某些传输 backend 不能一次性把 KV 放到最终位置，需要 staging scatter。
- 底层 receiver 可能已经返回 Success，但 staging scatter 仍未完成。
- staging 模式仍要复用 metadata gate 和跨 rank MIN 汇总。

设计选择：
- 对每个 `decode_req`，若 receiver 要求 staging 且未完成，就调用 `advance_scatter`。
- raw poll 成功后再次检查 staging 是否 done，未 done 时降级为 Transferring。
- 最后应用 metadata gate 并执行 all-reduce。

Explain：
这段把 staging handler 纳入 poll tick：每次查询状态前先推进数据搬运，查询后再用 staging 完成度修正 Success。

来源：python/sglang/srt/disaggregation/utils.py L163-L191

Code：

```python
def poll_and_all_reduce_with_staging(
    decode_reqs,
    staging_handler,
    gloo_group: dist.ProcessGroup,
    metadata_buffers: Optional[MetadataBuffers] = None,
    server_args: Optional[ServerArgs] = None,
):
    for decode_req in decode_reqs:
        if decode_req.kv_receiver.require_staging and not staging_handler.is_done(
            decode_req
        ):
            staging_handler.advance_scatter(decode_req)

    receivers = [dr.kv_receiver for dr in decode_reqs]
    raw_polls = _poll_with_failure_injection(receivers)
    for i, decode_req in enumerate(decode_reqs):
        if raw_polls[i] == int(KVPoll.Success):
            if decode_req.kv_receiver.require_staging and not staging_handler.is_done(
                decode_req
            ):
                raw_polls[i] = int(KVPoll.Transferring)
    if metadata_buffers is not None and server_args is not None:
        _apply_metadata_gate(raw_polls, decode_reqs, metadata_buffers, server_args)
    poll_tensor = torch.tensor(raw_polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(poll_tensor, op=dist.ReduceOp.MIN, group=gloo_group)
    return poll_tensor.tolist()
```

代码逻辑：
- 遍历 decode requests。
- 对需要 staging 且未完成的请求推进 scatter。
- 收集 receiver 并执行 poll。
- 对 Success 项重新检查 staging 完成度。
- staging 未完成时降级为 Transferring。
- metadata 参数存在时继续应用 metadata gate。
- 用 CPU all-reduce 汇总状态。

为什么这样写：
- staging 的推进和状态判断在同一个 tick 内完成，调度器不需要额外维护 backend 细节。
- raw Success 不直接放行，避免“底层收完但本地 scatter 未完成”的窗口。
- metadata gate 和 all-reduce 保持与非 staging 路径一致。

不变量与失败模式：
- `staging_handler.is_done` 必须准确反映最终 KV 是否可读。
- `advance_scatter` 需要可重复调用并能逐步推进。
- 如果 staging handler 卡住，状态会持续 Transferring，decode 不会消费未完成 KV。

Comment：
staging 路径把“网络传输完成”和“本地 KV 可读”拆成了两个 ready 条件。

### 2.2 `maybe_release_metadata_buffer` 释放 prefill 侧 slot

问题与约束：
- metadata buffer 是有限资源，按请求分配。
- prefill 完成、失败或 abort 后，如果 slot 不释放，后续请求会被耗尽。
- 重复释放或释放未分配 slot 都可能污染 allocator。

设计选择：
- 只在 `req.metadata_buffer_index >= 0` 时释放。
- 调 allocator 的 `free` 归还 slot。
- 释放后把请求上的 index 重置为 -1。

Explain：
这个函数把 metadata slot 的释放做成幂等风格的清理入口。调用方不用先判断请求是否真的拿过 slot，只要交给该函数即可。

来源：python/sglang/srt/disaggregation/prefill.py L87-L101

Code：

```python
def maybe_release_metadata_buffer(
    req: Req, allocator: ReqToMetadataIdxAllocator
) -> None:
    """
    Release the metadata buffer index allocated for a request in prefill disaggregation mode.
    """
    if req.metadata_buffer_index >= 0:
        allocator.free(req.metadata_buffer_index)
        req.metadata_buffer_index = -1
```

代码逻辑：
- 检查请求是否持有有效 metadata index。
- 如果有效，把 index 归还 allocator。
- 将请求字段重置为 -1，表示不再持有 slot。

为什么这样写：
- 把释放条件集中到一个函数，减少完成路径和异常路径重复判断。
- 重置 index 能防止同一个请求被后续清理逻辑二次释放。
- allocator 不需要理解 Req，只负责 slot 回收。

不变量与失败模式：
- `metadata_buffer_index` 的无效值约定是 -1。
- 调用方必须覆盖所有请求结束路径。
- 如果释放后仍有 decode 侧读取该 slot，可能读到被新请求复用的 metadata。

Comment：
metadata gate 的可靠性依赖 slot 分配和释放的生命周期正确。

### 2.3 `PrefillBootstrapQueue` 持有传输、metadata 和并行上下文

问题与约束：
- Prefill 侧要为请求创建 KV sender，并与 decode 侧的 prealloc/metadata 约定配合。
- MLA、draft KV、TP rank、pipeline rank、bootstrap 端口和 gloo group 都会影响传输行为。
- 队列需要同时访问调度器和 transfer backend。

设计选择：
- 构造函数显式接收 token KV pool、draft KV pool、metadata allocator、metadata buffers。
- 保存 TP/GPU/bootstrap/gloo/max token/scheduler/PP/transfer backend 等上下文。
- 初始化时根据 token pool 判断是否 MLA backend。

Explain：
`PrefillBootstrapQueue` 是 prefill 侧 PD bootstrap 的状态容器。它不只是一个 FIFO，还把创建 sender、分配 metadata、协调并行 rank 所需的依赖聚在一起。

来源：python/sglang/srt/disaggregation/prefill.py L104-L130

Code：

```python
class PrefillBootstrapQueue:
    """
    Store the requests in bootstrapping
    """

    def __init__(
        self,
        token_to_kv_pool: KVCache,
        draft_token_to_kv_pool: Optional[KVCache],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        tp_rank: int,
        tp_size: int,
        gpu_id: int,
        bootstrap_port: int,
        gloo_group: ProcessGroup,
        max_total_num_tokens: int,
        scheduler: Scheduler,
        pp_rank: int,
        pp_size: int,
        transfer_backend: TransferBackend,
    ):
        self.token_to_kv_pool = token_to_kv_pool
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self.is_mla_backend = is_mla_backend(token_to_kv_pool)
```

代码逻辑：
- 接收主 KV pool 和可选 draft KV pool。
- 接收 metadata allocator 与共享 metadata buffers。
- 接收 TP、GPU、bootstrap port、gloo group 等并行/通信参数。
- 接收 scheduler、PP rank/size 和 transfer backend。
- 保存 pool，并判断当前 KV pool 是否 MLA backend。

为什么这样写：
- bootstrap 队列需要同时操作内存、网络和 scheduler 状态，构造参数显式能降低隐式全局依赖。
- draft KV pool 放在同一队列里，便于投机场景复用 PD bootstrap。
- metadata allocator 和 buffers 成对注入，保证分配与写入由同一组件管理。

不变量与失败模式：
- `tp_rank/tp_size`、`pp_rank/pp_size` 必须和实际并行拓扑一致。
- `metadata_buffers` 的大小要覆盖并发 bootstrap 请求。
- `transfer_backend` 与 decode 侧 receiver backend 必须匹配，否则 handshake/poll 无法收敛。

Comment：
Prefill bootstrap 队列是 PD prefill 的控制面枢纽。

## 3. Decode 侧容量与 HiCache 恢复

### 3.1 `should_force_retry` 用确定性 hash 注入 optimistic retry

问题与约束：
- optimistic prefill retry 属于罕见路径，需要测试时可控触发。
- 随机触发必须可复现，否则很难定位跨节点失败。
- 已经 retry 过或被 retract 的请求不能反复注入 retry。

设计选择：
- 从环境变量读取 retry 概率。
- 概率小于等于 0、已有 retry 计数或请求已 retract 时直接返回 False。
- 对 `rid` 做 SHA-256，取前 8 字节与概率阈值比较。

Explain：
这个测试钩子把 retry 注入做成“按 rid 确定”的概率事件。同一个请求 id 在相同概率下结果稳定，适合复现 PD bootstrap 的回滚路径。

来源：python/sglang/srt/disaggregation/prefill.py L77-L84

Code：

```python
def should_force_retry(req: Req) -> bool:
    """Test hook to force a request into optimistic prefill retry."""
    retry_prob = envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.get()
    if retry_prob <= 0 or req.time_stats.prefill_retry_count > 0 or req.is_retracted:
        return False

    digest = hashlib.sha256(str(req.rid).encode()).digest()
    return int.from_bytes(digest[:8], "big") < retry_prob * 2**64
```

代码逻辑：
- 读取测试环境变量中的 retry 概率。
- 排除关闭、已 retry、已 retract 三类请求。
- 将请求 id 转成字符串并计算 SHA-256。
- 取 digest 前 8 字节转换为整数。
- 与概率阈值比较，决定是否强制 retry。

为什么这样写：
- 不使用运行时随机数，避免同一测试在不同 rank 或不同重跑中分歧。
- 限制每个请求只注入一次，避免 retry 风暴。
- 环境变量默认关闭，不影响生产路径。

不变量与失败模式：
- `rid` 必须在测试复现中保持稳定。
- retry 概率应在 0 到 1 之间配置。
- 如果多个 rank 对同一请求的 `rid` 或 retry count 认知不同，可能出现 retry 决策不一致。

Comment：
这是为 PD 故障路径准备的确定性混沌开关。

### 3.2 `DecodeReqToTokenPool` 为 prealloc 请求单独订阅容量

问题与约束：
- 普通 `ReqToTokenPool` 把 pre-allocated、transfer、running 一起受 `--max-running-requests` 限制。
- PD decode 希望提前为更多请求预分配位置，以便 unblock prefill。
- 预分配不能挤占真正 running 请求的上限语义。

设计选择：
- 新建 `DecodeReqToTokenPool`。
- 构造时接收 `size` 和额外的 `pre_alloc_size`。
- 实际分配大小为 `size + pre_alloc_size + 1`，其中 0 行保留 padding。

Explain：
decode 侧把“正在运行的请求容量”和“等待 KV 到达的预分配容量”拆开。这样 running 仍受 `size` 限制，而 prealloc/transfer 可以使用额外空闲内存。

来源：python/sglang/srt/disaggregation/decode.py L107-L133

Code：

```python
class DecodeReqToTokenPool:
    """
    The difference of DecodeReqToTokenPool and ReqToTokenPool is that
    DecodeReqToTokenPool subscribes memory for pre-allocated requests.

    In ReqToTokenPool, if `--max-running-requests` is 8,
    #pre-allocated + #transfer + #running <= 8, but there are in fact more memory can carry pre-allocated requests.

    In DecodeReqToTokenPool, if `--max-running-requests` is 8,
    #running <= 8, #pre-allocated + #transfer <= pre_alloc_size, so we can use the free memory to pre-allocate requests to unblock prefill.
    """

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        pre_alloc_size: int,
    ):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size
        self._alloc_size = size + pre_alloc_size + 1
```

代码逻辑：
- 定义 decode 专用 token pool。
- 构造函数接收 running 容量、上下文长度、device、memory saver 开关和 prealloc 容量。
- 创建 memory saver adapter。
- 保存 running 容量。
- 计算包含 prealloc 和 padding 的总分配行数。

为什么这样写：
- prealloc 能提前完成 decode 侧占位，prefill 不必等 running 槽位释放。
- running 上限仍然由 `size` 控制，不因为预分配扩大真实并发执行数。
- 保留 padding 行延续普通 token pool 的 index 约定。

不变量与失败模式：
- `pre_alloc_size` 需要和 KV cache 空闲容量匹配。
- 调度器必须区分 preallocated/transfer/running 三类请求。
- 如果 prealloc 过大，会增加内存压力；过小则无法充分 unblock prefill。

Comment：
PD 分离的吞吐收益部分来自 decode 侧提前占位，而不是只优化传输本身。

### 3.3 HiCache mixin 把 prefix 命中、prefetch 和 restore gate 接入 decode

问题与约束：
- decode 侧可能已有 L1 device prefix，也可能在 L2 host 或 L3 storage 中命中更长 prefix。
- L3 storage prefetch 是异步操作，不能让 KV receiver 的 Success 直接放行。
- abort 或失败时需要清理 prefetch 注册和 tree cache lock 引用。

设计选择：
- `DecodePrefixMatch` 统一记录 L1/L2/L3 命中长度与 restore token 数。
- `DecodeHiCachePreallocMixin` 在 admission 后启动 L3 prefetch，并统计 pending restore token。
- `HiCacheRestoreGatedKVReceiver` 包装原 receiver：底层 Success 但 restore 仍 pending 时返回 Transferring。
- `DecodeHiCacheTransferMixin` 负责清理 prefetch 资源和 restored node lock。

Explain：
HiCache 让 decode 不必完全依赖远端 KV 传输：已命中的 prefix 可以从本地 device、host 或 storage 恢复。关键是 restore ready 要并入 PD ready 条件，避免提前消费尚未 load back 的 KV。

来源：python/sglang/srt/disaggregation/decode_hicache_mixin.py L23-L180

Code：

```python
@dataclass
class DecodePrefixMatch:
    prefix_indices: torch.Tensor
    l2_host_hit_length: int
    l3_storage_hit_length: int
    last_device_node: Any
    last_host_node: Any = None
    prefetch_registered: bool = False

    @property
    def decode_prefix_len(self) -> int:
        return self.l1_prefix_len + self.l2_host_hit_length + self.l3_storage_hit_length

    @property
    def restore_token_count(self) -> int:
        return self.decode_prefix_len - self.l1_prefix_len

class DecodeHiCachePreallocMixin:
    def _start_hicache_prefetch(
        self, req: Req, prefix_match: Optional[DecodePrefixMatch]
    ) -> None:
        if (
            prefix_match is None
            or prefix_match.l3_storage_hit_length <= 0
            or prefix_match.last_host_node is None
        ):
            return
        try:
            self.tree_cache.prefetch_from_storage(
                req.rid, node, suffix, last_hash, prefix_keys
            )
            prefix_match.prefetch_registered = (
                req.rid in self.tree_cache.ongoing_prefetch
            )
        except Exception as e:
            logger.warning(...)
            prefix_match.l3_storage_hit_length = 0
            prefix_match.prefetch_registered = False

class HiCacheRestoreGatedKVReceiver:
    def poll(self) -> KVPoll:
        poll = self.decode_req.kv_receiver.poll()
        if (
            poll == KVPoll.Success
            and self.decode_req.hicache_restore_status == HiCacheRestoreResult.PENDING
        ):
            return KVPoll.Transferring
        return poll
```

代码逻辑：
- 用 dataclass 保存 prefix 命中和 prefetch 状态。
- 根据 L1/L2/L3 命中长度计算 decode prefix 和需 restore token 数。
- admission 成功后，对 L3 命中发起 storage prefetch。
- prefetch 失败时降级为 L2-only restore。
- poll 时先读取底层 receiver 状态。
- 若底层 Success 但 HiCache restore 仍 pending，则返回 Transferring。
- 清理路径释放 prefetch 注册和 restored node lock。

为什么这样写：
- prefix 命中结构把 radix tree、host cache、storage cache 的结果统一成 decode 可理解的 ready 条件。
- prefetch 失败选择降级而不是中断请求，能保持服务可用性。
- gated receiver 让 HiCache restore 复用已有 poll/all-reduce 状态机，不需要单独给 scheduler 新增 ready 通道。

不变量与失败模式：
- `restore_token_count` 必须等于需要从 L2/L3 load back 的 token 数。
- pending restore token 需要计入容量，否则可能预留不足。
- 如果 restore 状态没有从 PENDING 推进到 READY/FAILED，请求会持续 Transferring。
- abort 路径必须释放 prefetch 与 lock，否则 tree cache 引用会泄漏。

Comment：
HiCache 在 PD decode 中是“本地恢复路径”，但它必须服从同一个 ready gate。

## 4. 多模态 PD encoder

### 4.1 `MMEncoder` 独立初始化多模态模型与并行环境

问题与约束：
- 多模态 PD 场景中，vision/audio 编码可以独立于 text prefill/decode 部署。
- encoder 仍然要加载模型权重、初始化 TP 并绑定正确 GPU。
- 远端权重加载、fast image processor、vision config 和 audio sample rate 都会影响服务行为。

设计选择：
- `MMEncoder.__init__` 从 `ServerArgs` 构建 `ModelConfig` 和 `LoadConfig`。
- 使用 `base_gpu_id + rank` 选择设备，并初始化 distributed/model parallel。
- 加载多模态 processor、vision config、audio sample rate 和模型。
- 创建 ZMQ context 与线程池用于后续请求处理。

Explain：
`MMEncoder` 是 PD 架构中多模态编码服务的本地 worker。它把多模态预处理、模型加载和并行初始化放在 encoder 进程内完成，使 encode -> prefill -> decode 可以作为三段服务组合。

来源：python/sglang/srt/disaggregation/encode_server.py L233-L294

Code：

```python
class MMEncoder:
    def __init__(
        self,
        server_args: ServerArgs,
        schedule_path=None,
        dist_init_method=None,
        rank: int = 0,
    ):
        logger.info(f"init MMEncoder {rank}/{server_args.tp_size}")
        self.server_args = server_args
        set_global_server_args_for_scheduler(server_args)
        self.rank = rank
        self.profiler = EncoderProfiler(rank)
        self._load_mm_processor(server_args)

        self.model_config = ModelConfig.from_server_args(
            server_args,
        )
        self.load_config = LoadConfig(
            load_format=server_args.load_format,
            download_dir=server_args.download_dir,
            model_loader_extra_config=server_args.model_loader_extra_config,
            remote_instance_weight_loader_seed_instance_ip=server_args.remote_instance_weight_loader_seed_instance_ip,
            remote_instance_weight_loader_seed_instance_service_port=server_args.remote_instance_weight_loader_seed_instance_service_port,
            remote_instance_weight_loader_send_weights_group_ports=server_args.remote_instance_weight_loader_send_weights_group_ports,
        )
        self.model_type = getattr(
            self.model_config.hf_config, "model_type", "unknown"
        ).lower()

        self.device = server_args.device
        self.gpu_id = server_args.base_gpu_id + rank
        self.device_config = DeviceConfig(
            device=self.device,
            gpu_id=self.gpu_id,
        )
        torch.get_device_module(self.device).set_device(self.gpu_id)
        self._build_vision_config(server_args.mm_process_config)
        self.model_audio_sr = self._resolve_audio_sr()

        init_distributed_environment(
            backend=get_default_distributed_backend(self.device),
            world_size=server_args.tp_size,
            rank=rank,
            distributed_init_method=dist_init_method,
            local_rank=rank,
        )
        initialize_model_parallel(tensor_model_parallel_size=server_args.tp_size)
        initialize_dp_attention(server_args, self.model_config)

        self.model = get_model(
            model_config=self.model_config,
            load_config=self.load_config,
            device_config=self.device_config,
        )
```

代码逻辑：
- 保存 server args 并设置 scheduler 全局参数。
- 初始化 rank profiler 和多模态 processor。
- 从 server args 构建模型配置和加载配置。
- 解析模型类型。
- 计算当前 rank 对应的 GPU。
- 设置 device config 并切换当前设备。
- 构建 vision config 并解析 audio sample rate。
- 初始化分布式环境、TP model parallel 和 DP attention。
- 调 `get_model` 加载 encoder 模型。

为什么这样写：
- encoder 独立部署时不能依赖 prefill/decode 进程替它初始化模型环境。
- `LoadConfig` 保留远端实例权重加载字段，支持和主模型相同的加载机制。
- ZMQ 与线程池后续处理请求，但模型和并行环境必须先稳定建立。

不变量与失败模式：
- `server_args.tp_size`、`rank`、`base_gpu_id` 必须和启动拓扑一致。
- 多模态 processor 与模型类型必须匹配。
- 分布式初始化失败会阻止 encoder 服务启动。
- 远端权重加载参数错误会在 `get_model` 前后暴露为加载失败。

Comment：
多模态 PD 的 encoder 是独立服务，但仍沿用 SGLang 模型加载和并行初始化栈。
