---
type: batch-doc
module: 07-Scheduler
batch: "07"
doc_type: walkthrough
title: "Scheduler · 源码走读"
tags:
 - sglang/batch/07
 - sglang/module/scheduler
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-03
---
# Scheduler · 源码走读

> 走读顺序：`run_scheduler_process` → `Scheduler.__init__` → IPC/收请求 → 事件循环 → 组 batch → `run_batch` → 结果处理 → PP mixin

## 源码阅读依据

本文重写前已完整阅读以下 upstream 文件，再基于代码路径补充设计解释：

| upstream 文件 | 本文使用方式 |
|---------------|--------------|
| `sglang/python/sglang/srt/managers/scheduler.py` | Scheduler 初始化、事件循环、请求入队、batch 组织、forward、pause/flush/shutdown |
| `sglang/python/sglang/srt/managers/scheduler_components/request_receiver.py` | ZMQ 拉取、TP/CP/DP 广播、PP 链式接收、多模态与共享内存收尾 |
| `sglang/python/sglang/srt/managers/scheduler_pp_mixin.py` | PP microbatch 事件循环、proxy/output 通信、PD prefill/decode consensus、动态 chunk |

## 首次阅读路径（约 30 分钟）

| 顺序 | 章节锚点 | 读完应能回答的问题 | 预计分钟 |
|------|----------|-------------------|----------|
| 1 | [[#1. 进程入口与初始化链]] | Scheduler 进程如何启动、`__init__` 为何必须按固定顺序？ | 5 |
| 2 | [[#2. 请求分发（TypeBasedDispatcher）]] | Generate/Embedding 等消息如何按类型路由到 handler？ | 5 |
| 3 | [[#4. 事件循环]] | `event_loop_normal` 与 `event_loop_overlap` 各做什么、默认走哪条？ | 7 |
| 4 | [[#5. 组 Batch：`get_next_batch_to_run`]] | prefill merge、PrefillAdder 组 batch、decode update 如何衔接？ | 8 |
| 5 | [[#6. GPU 前向：`run_batch`]] | 组好的 batch 如何进入 ModelRunner、结果如何返回？ | 5 |

**跳过策略：** 二遍再读 §3 收请求细节、§7 PP mixin、§8 Overlap 基础设施；若已读过 [[08-SchedulePolicy-02-源码走读]]，§5.2 中 PrefillAdder 调用可略扫。

---

## 设计主线：Scheduler 为什么长成这样

Scheduler 的核心不是“从队列里取请求然后 forward”，而是在单个调度进程里同时维持四组约束：

1. **把控制面集中到调度进程。** TokenizerManager 负责前台异步与分词，TpWorker/ModelRunner 负责 GPU 执行；Scheduler 持有请求状态、KV cache 分配器、调度策略、metrics 和控制请求 dispatcher。这让 GPU batch 的所有资源决策在同一处发生。
2. **把 prefill 与 decode 当成两类资源问题。** Prefill 关心 TTFT、prefix/cache 命中和大块 token admission；decode 关心每步新增 KV slot。源码中 `get_next_batch_to_run()` 先合并上轮 prefill，再优先尝试新 prefill，最后才更新 decode running batch。
3. **用流水线换吞吐，用不变量守住正确性。** 默认 `event_loop_overlap()` 让“当前 batch GPU forward”和“上一轮 CPU result processing”错开一拍；代价是需要 `result_queue`、`FutureMap`、WAR barrier、batch snapshot 和 D2H copy event 共同维护跨 stream 生命周期。
4. **把资源不足当作正常分支。** KV 不足时不是直接 OOM，而是 `retract_decode()` 撤回部分请求、释放 slot、重新入队；chunked prefill、HiCache、PP/PD、LoRA 都被纳入同一个 admission 流程。

读 Scheduler 时要抓这条哲学：**它不是 GPU worker 的薄封装，而是 serving runtime 的资源仲裁器。**

## 1. 进程入口与初始化链

### 1.1 `run_scheduler_process`

**问题与约束：** Scheduler 是独立子进程，必须先完成插件加载、进程命名、日志、NUMA/CPU affinity、tracing，再向父进程确认“GPU worker 已可用”。如果初始化失败，不能让同组 NCCL rank 悬挂。

**设计选择：** 入口函数只做进程级准备和生命周期包裹；真正调度状态全部封装在 `Scheduler` 对象里。初始化成功后通过 pipe 发送 `get_init_info()`，异常时通知父进程并按环境变量选择是否杀进程组。

**Explain：** Engine 为每个 GPU worker fork/spawn 一个 scheduler 进程。初始化完成后通过 pipe 回报 `get_init_info()`，然后进入事件循环。

**Code：**

```python
## 来源：python/sglang/srt/managers/scheduler.py L4292-L4311
    # Create a scheduler and run the event loop
    scheduler = None
    try:
        scheduler = Scheduler(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            attn_cp_rank,
            moe_dp_rank,
            dp_rank,
        )

        # Send initialization info back to the parent process
        pipe_writer.send(scheduler.get_init_info())

        # Run the event loop (blocks until a ShutdownReq sets gracefully_exit)
        scheduler.run_event_loop()
```

**代码逻辑：**
- `load_plugins()` 在构造 Scheduler 前执行，给插件覆盖依赖的机会。
- `configure_scheduler_process()` 设置进程 title、日志、faulthandler、CPU/NUMA 绑定。
- `Scheduler(...)` 完成模型、KV cache、IPC、dispatcher 等状态装配。
- pipe handshake 成功后才进入阻塞式事件循环。

**为什么这样写：** Scheduler 进程是 serving 运行时的 GPU 侧控制点；把进程配置放在对象构造外，把调度状态放在对象内，可以清晰区分“进程生命周期”和“调度生命周期”。异常时主动通知父进程，是为了避免一个 rank 失败后其他 rank 继续卡在通信原语里。

**不变量与失败模式：**
- `pipe_writer.send(scheduler.get_init_info())` 必须在 `run_event_loop()` 前完成，否则父进程无法判断子进程就绪。
- 异常路径不能调用可能卡住的 GPU teardown；源码只在 graceful exit 时释放 host resources。

**Comment：** 读者应抓住：Scheduler 的入口不是普通函数调用，而是多 rank serving 中的进程级 handshake 与故障边界。

### 1.2 `Scheduler.__init__` 初始化顺序

**问题与约束：** Scheduler 初始化时既要构造 GPU worker，又要拿到 worker 暴露的 KV cache 能力、stream、memory budget，再反过来初始化调度策略和接收器。顺序错了会出现 patch 不生效、pool 未建好、dispatcher 访问未初始化组件等问题。

**设计选择：** 源码把初始化拆成大量 `init_*` 方法，但在 `__init__` 中保持单向装配顺序：配置/IPC/Tokenizer → worker/KV pool → running status/policy/disaggregation/overlap → control components → receiver/result processor。

**Explain：** 初始化严格有序：先配置与 IPC，再 model worker 与 KV cache，最后 running 状态与 dispatcher。`init_model_worker` 必须在 `maybe_revert_pr_fix` 之后，以便 patch 生效。

**Code：**

```python
## 来源：python/sglang/srt/managers/scheduler.py L391-L422（节选）
        # Init model configs
        self.init_model_config()

        # Init metrics stats
        self.init_metrics_collector(tp_rank, pp_rank, dp_rank)

        # Init inter-process communication
        self.init_ipc_channels(port_args)
        self.init_idle_sleeper()

        # Init ZBAL, switch allocator should before any torch alloc action
        self.init_zbal_on_npu()

        # Init PD-multiplexing context
        if self.enable_pdmux:
            self.init_pdmux()

        # Init tokenizer
        self.init_tokenizer()

        # Init moe config and GEMM config (FP8 GEMM, etc.)
        self.init_moe_gemm_config()

        # Init mamba backend
        self.init_mamba_backend()

        # Must precede init_model_worker: revert targets like _init_pools run during it,
        # so patching them afterwards is a no-op.
        maybe_revert_pr_fix()

        # Launch a model worker and draft model worker if using speculative decoding
        self.init_model_worker()
```

**代码逻辑：**
- 前半段先解析 `server_args`，构造 `ParallelState`，再初始化模型配置、metrics、IPC、tokenizer。
- `maybe_revert_pr_fix()` 必须在 `init_model_worker()` 前执行，因为 worker 初始化会触发被 patch 的内部函数。
- worker 初始化后才能 build KV cache，进而初始化 running status、schedule policy、disaggregation、overlap、dispatcher 和 receiver。

**为什么这样写：** Scheduler 采用“装配式主类”而不是把所有逻辑塞进一个巨型构造块：主类持有统一状态，横切能力拆到组件和 mixin。代价是初始化顺序成为重要不变量；收益是 IPC、metrics、flush、receiver、result processor 可以围绕同一组核心资源组合。

**不变量与失败模式：**
- `init_model_worker()` 之前不能访问 worker stream、memory pool 或 attention backend。
- `init_overlap()` 之前必须已经有 `device`、`forward_stream`、`req_to_token_pool`，否则 FutureMap 和 stream context 无法构造。
- `init_request_dispatcher()` 必须在各 handler 依赖的组件初始化后执行。

**Comment：** 关键分支：`enable_overlap = not disable_overlap_schedule and not use_mlx()`；MLX 走独立 overlap 路径。读者应把 `__init__` 看成组件依赖图，而不是线性样板代码。

### 1.3 IPC 通道初始化

**问题与约束：** 多 TP/CP/PP rank 不能都直接消费 TokenizerManager 的 ZMQ 消息，否则同一请求会被重复读取或乱序处理；但非 rank zero 仍必须拿到完全相同的控制/数据请求。

**设计选择：** 只有 `pp_rank == 0 && attn_tp_rank == 0 && attn_cp_rank == 0` 的入口 rank 创建外部 IPC socket；其他 rank 通过后续 broadcast / P2P 获取请求对象。

**Explain：** `init_ipc_channels` 根据并行 rank 判定当前 Scheduler 是否是外部消息入口。入口 rank 创建 Tokenizer/RPC socket 与 load snapshot writer；非入口 rank 保持内部通信角色。

**Code：**

```python
## 来源：python/sglang/srt/managers/scheduler.py L605-L621
    def init_ipc_channels(self, port_args: PortArgs):
        is_rank_zero = (
            self.ps.pp_rank == 0
            and self.ps.attn_tp_rank == 0
            and self.ps.attn_cp_rank == 0
        )
        self.ipc_channels = SchedulerIpcChannels.create(
            port_args=port_args,
            is_rank_zero=is_rank_zero,
            skip_tokenizer_init=self.server_args.skip_tokenizer_init,
            metrics_enabled=self.server_args.enable_metrics
            and (
                self.ps.attn_tp_rank == 0
                or self.server_args.enable_metrics_for_all_schedulers
            ),
            enable_scripted_runtime=envs.SGLANG_TEST_SCRIPTED_RUNTIME.get(),
        )
```

**代码逻辑：**
- `is_rank_zero` 同时要求 PP、attention TP、attention CP 都为入口 rank。
- `SchedulerIpcChannels.create(...)` 根据 `is_rank_zero` 决定哪些 socket 实际启用。
- metrics socket 也按 rank 与 `enable_metrics_for_all_schedulers` 控制，避免无意义重复上报。

**为什么这样写：** 外部 IPC 是“单入口、多 rank 同步”的模型。这样能把网络/IPC 消费控制在一个 rank，同时让 TP/CP/PP 内部通过确定性的 broadcast/P2P 保持一致视图。

**不变量与失败模式：**
- 多个 rank 直接读同一个 ZMQ 输入会导致请求丢失或重复调度。
- 非入口 rank 不能假设 socket 存在；后续接收路径必须通过 group 通信填充请求。

**Comment：** 仅 rank zero 创建 ZMQ socket；其他 rank 通过 `broadcast_pyobj` 或 PP `point_to_point_pyobj` 同步请求。

---

## 2. 请求分发（TypeBasedDispatcher）

### 2.1 `init_request_dispatcher`

**问题与约束：** Scheduler 收到的消息不只有生成请求，还包括 flush cache、LoRA、权重更新、pause、health/load 查询、RPC、外部 corpus 等控制面请求。事件循环不能在主路径里铺满 `isinstance` 分支。

**设计选择：** 用 `TypeBasedDispatcher` 把消息类型和 handler 显式注册成表；数据面请求进入调度队列，控制面请求直接执行并返回输出。

**Explain：** 所有从 TokenizerManager / RPC 进来的消息按类型路由到 handler。Generate/Embedding 走调度路径；FlushCache、LoRA、权重更新等走管理路径。

**Code：**

```python
## 来源：python/sglang/srt/managers/scheduler.py L1352-L1364
    def init_request_dispatcher(self):
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.handle_generate_request),
                (TokenizedEmbeddingReqInput, self.handle_embedding_request),
                (BatchTokenizedGenerateReqInput, self.handle_batch_generate_request),
                (BatchTokenizedEmbeddingReqInput, self.handle_batch_embedding_request),
                (FlushCacheReqInput, self.flush_wrapper.handle),
                (ClearHiCacheReqInput, self.clear_hicache_storage_wrapped),
                (AttachHiCacheStorageReqInput, self.attach_hicache_storage_wrapped),
                (DetachHiCacheStorageReqInput, self.detach_hicache_storage_wrapped),
                (AbortReq, self.abort_request),
                (OpenSessionReqInput, self.open_session),
```

**代码逻辑：**
- dispatcher 表在初始化时一次性构造，后续 `process_input_requests` 只调用 `_request_dispatcher(recv_req)`。
- handler 返回普通输出时走 `send_to_tokenizer`，返回 `RpcReqOutput` 时走 RPC socket。
- Generate/Embedding handler 本身通常不直接返回输出，而是构造 `Req` 后进入队列。

**为什么这样写：** Scheduler 是单线程事件循环，控制面和数据面共享入口。类型分发表把“消息语义”从事件循环中剥离出来，让主循环保持 `recv → dispatch → schedule` 的稳定形状，同时方便新控制请求以局部方式接入。

**不变量与失败模式：**
- 新消息类型必须注册，否则请求会在 dispatcher 处失败或被错误处理。
- 控制面 handler 不能长期阻塞，否则会拖慢 GPU batch launch 和请求接收。

**Comment：** `process_input_requests` 遍历 `recv_reqs`，调用 `_request_dispatcher(recv_req)`，非 RPC 结果经 `ipc_channels.send_to_tokenizer.send_output` 回传。

### 2.2 `handle_generate_request` — 构造 Req

**问题与约束：** TokenizerManager 传入的是外部请求的 tokenized 形态；Scheduler 内部需要的是带调度状态、cache 状态、metrics、session、多模态、PD bootstrap 信息的 `Req`。这个转换必须在进入 waiting queue 前一次性补齐。

**设计选择：** `handle_generate_request` 先处理 session / radix-native session / input_embeds / bootstrap 默认值，再构造内部 `Req`，随后处理 disaggregation 校验、多模态扩展、长度/logprob/routed experts 校验和 grammar 预处理。

**Explain：** 将 IPC 输入转为内部 `Req` 对象，处理 session、disaggregation bootstrap、多模态等，最后 `_add_request_to_queue`。

**Code：**

```python
## 来源：python/sglang/srt/managers/scheduler.py L2047-L2088（节选）
            req = Req(
                recv_req.rid,
                recv_req.input_text,
                recv_req.input_ids,
                recv_req.sampling_params,
                return_logprob=recv_req.return_logprob,
                top_logprobs_num=recv_req.top_logprobs_num,
                token_ids_logprob=recv_req.token_ids_logprob,
                stream=recv_req.stream,
                lora_id=recv_req.lora_id,
                session_id=recv_req.session_id,
                input_embeds=recv_req.input_embeds,
                positional_embed_overrides=recv_req.positional_embed_overrides,
                token_type_ids=recv_req.token_type_ids,
                custom_logit_processor=recv_req.custom_logit_processor,
                require_reasoning=recv_req.require_reasoning,
                return_hidden_states=recv_req.return_hidden_states,
                return_routed_experts=recv_req.return_routed_experts,
                routed_experts_start_len=recv_req.routed_experts_start_len,
                return_indexer_topk=recv_req.return_indexer_topk,
                eos_token_ids=self.model_config.hf_eos_token_id,
                bootstrap_host=recv_req.bootstrap_host,
                bootstrap_port=recv_req.bootstrap_port,
                bootstrap_room=recv_req.bootstrap_room,
                disagg_mode=self.disaggregation_mode,
                routed_dp_rank=recv_req.routed_dp_rank,
                disagg_prefill_dp_rank=recv_req.disagg_prefill_dp_rank,
                vocab_size=self.model_config.vocab_size,
                priority=recv_req.priority,
                metrics_collector=(
                    self.metrics_collector
                    if self.metrics_reporter.enable_metrics
                    else None
                ),
                routing_key=recv_req.routing_key,
                extra_key=recv_req.extra_key,
                http_worker_ipc=recv_req.http_worker_ipc,
                dllm_config=self.dllm_config,
                time_stats=recv_req.time_stats,
                multi_item_delimiter_indices=recv_req.multi_item_delimiter_indices,
            )
            req.tokenizer = self.tokenizer
```

**代码逻辑：**
- 普通请求直接构造 `Req`；session 请求通过 `SessionController` 从历史上下文派生。
- `input_embeds` 会生成 fake `input_ids`，让后续长度和调度逻辑仍有 token 长度可用。
- disaggregation 模式下要求 bootstrap 信息；缺失时直接 abort 并 stream 输出。
- 多模态请求会扩展 dummy token、补 M-RoPE 位置，并再次检查最大输入长度。

**为什么这样写：** Scheduler 的设计哲学是：一旦请求进入调度队列，后续 batch admission 不再回头理解 HTTP/OpenAI/session 原始语义，而只面对统一的 `Req`。这把前台协议复杂度压缩成内部调度对象，代价是 `handle_generate_request` 成为语义归一化的复杂入口。

**不变量与失败模式：**
- `init_req_max_new_tokens(req)` 必须在入队前执行，否则 PrefillAdder 可能接收永远无法被调度的请求。
- 多模态扩展后的长度必须重新校验；只校验原始 token 长度会低估 KV 需求。
- disaggregation 请求缺 bootstrap room 不能进入队列，否则后续 KV transfer 无法建立对应关系。

### 2.3 `_add_request_to_queue` — 入队策略

**问题与约束：** “入队”不是单一队列操作。普通 serving、PD prefill、PD decode 三种模式对请求生命周期的第一站不同：普通模式等 prefill；prefill 节点等 bootstrap；decode 节点先预分配并等待 KV transfer。

**设计选择：** `_add_request_to_queue` 用 `DisaggregationMode` 明确分流，并在每个分支设置对应 time stats。retracted decode 请求也复用这条入口，但标记 `is_retracted=True`。

**Explain：** `_add_request_to_queue` 是 Scheduler 的请求生命周期分流点：同一个 `Req` 会根据部署模式进入 waiting、prefill bootstrap 或 decode prealloc 队列。

**Code：**

```python
## 来源：python/sglang/srt/managers/scheduler.py L2288-L2310
    def _add_request_to_queue(self, req: Req, is_retracted: bool = False):
        if not self._set_or_validate_priority(req):
            return
        if self.disaggregation_mode == DisaggregationMode.NULL:
            if self._abort_on_queued_limit(req):
                return
            self._prefetch_kvcache(req)
            self.waiting_queue.append(req)
            req.time_stats.set_wait_queue_entry_time()
        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._prefetch_kvcache(req)
            self.disagg_prefill_bootstrap_queue.add(
                req, self.model_config.num_key_value_heads
            )
            req.time_stats.set_prefill_bootstrap_queue_entry_time()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.disagg_decode_prealloc_queue.add(req, is_retracted=is_retracted)
            if not is_retracted:
                req.time_stats.set_decode_prealloc_queue_entry_time()
            else:
                req.time_stats.set_retract_time()
        else:
            raise ValueError(f"Invalid {self.disaggregation_mode=}")
```

**代码逻辑：**
- 所有请求先过 `_set_or_validate_priority`，必要时为 priority 设默认极值或直接 abort。
- `NULL` 模式检查 waiting queue 上限、触发 HiCache prefetch，然后 append 到 `waiting_queue`。
- `PREFILL` 模式加入 `disagg_prefill_bootstrap_queue`，等待远端 decode 节点连接。
- `DECODE` 模式加入 `disagg_decode_prealloc_queue`，retracted 请求记录 retract time。

**为什么这样写：** Scheduler 把“请求进入系统”的语义绑定到部署形态。普通 serving 的瓶颈是本地 prefill/decode admission；PD prefill 的瓶颈是 bootstrap 与 KV 发送；PD decode 的瓶颈是 KV 接收与本地 slot 预分配。统一入口保证 abort、priority、metrics 的语义一致。

**不变量与失败模式：**
- priority 校验必须早于入队，否则禁用 priority 时仍可能污染队列排序。
- decode retract 重新入队必须带 `is_retracted=True`，否则时序统计和 prealloc 语义会混淆。
- 非法 disaggregation mode 直接 `ValueError`，避免请求进入未知生命周期。

**Comment：** 普通模式下进 `waiting_queue`；PD 分离模式下进 bootstrap/prealloc 专用队列（PD 分离详述）。

---

## 3. 收请求：`SchedulerRequestReceiver`

### 3.1 `recv_requests`

**问题与约束：** Scheduler 的每轮接收必须同时处理 ZMQ 非阻塞拉取、RPC、TP/CP/DP 广播、PP 上一 stage 转发、多模态 EPD、shared-memory feature materialize。它还不能让非入口 rank 重复消费外部 socket。

**设计选择：** `SchedulerRequestReceiver` 把接收链路独立成 frozen dataclass，主循环只调用 `recv_requests()`。内部按“拉取原始消息 → input blocker → rank 同步 → unwrap → 多模态处理 → shared memory 收尾”的固定顺序执行。

**Explain：** 每轮事件循环开头调用。rank zero 从 ZMQ NOBLOCK 拉取，经 input_blocker、broadcast、多模态 unwrap 后返回统一列表。

**Code：**

```python
## 来源：python/sglang/srt/managers/scheduler_components/request_receiver.py L72-L99
    @scheduler_nvtx_method("scheduler.recv_requests")
    def recv_requests(
        self,
    ) -> List[Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput, Any]]:
        """Receive results at tp_rank = 0 and broadcast it to all other TP ranks."""

        if self.scripted_scheduler_hook is not None:
            self.scripted_scheduler_hook.step()

        if self.recv_skipper is not None:
            if not self.recv_skipper.handle(self.get_last_forward_mode()):
                return []

        recv_reqs = self._pull_raw_reqs()

        if self.input_blocker is not None:
            recv_reqs = self.input_blocker.handle(recv_reqs)

        recv_reqs = self._broadcast_reqs_across_ranks(recv_reqs)

        if self.ps.pp_rank == 0:
            self.unwrap_pickle_wrapper(recv_reqs)

        recv_reqs = self._apply_mm_receiver(recv_reqs)

        self._finalize_shm_features(recv_reqs)

        return recv_reqs
```

**代码逻辑：**
- `recv_skipper` 可根据上一轮 forward mode 决定本轮是否收包。
- PP rank0 从 Tokenizer/RPC socket 拉取；非首 PP rank 从上一 stage 接收 pyobj。
- DP attention 下 work/control 请求分开广播；普通 TP 直接在 TP group 内广播。
- shared memory feature 在所有 broadcast 完成后再 unwrap，避免提前 unlink 造成其他 rank 反序列化失败。

**为什么这样写：** 接收链路的哲学是“外部输入单点消费，内部 rank 显式同步”。这样既避免 socket 竞争，又保证每个并行 rank 对本轮请求有一致视图。多模态 shared-memory 的收尾放在广播之后，是为了在性能和生命周期安全之间取平衡。

**不变量与失败模式：**
- `unwrap_pickle_wrapper` 只在 `pp_rank == 0` 执行；PP 后续 stage 已经接收跨 stage 对象。
- SHM feature 必须等 CPU group barrier 后 materialize，否则源 rank 释放共享内存会让 peer rank 打开失败。
- `max_recv_per_poll` 限制单轮拉取量，避免接收阶段饿死调度阶段。

**Comment：** `recv_skipper` 可在 decode 繁忙时跳过收包，降低调度开销；`max_recv_per_poll` 限制单次 poll 数量。

---

## 4. 事件循环

### 4.1 `event_loop_normal` — 基准循环

**问题与约束：** 需要一条最容易推理的基准路径，用于调试、非 overlap 设备、或复杂模式不可 overlap 的场景。

**设计选择：** normal loop 严格串行执行：收请求、处理输入、取下一个 batch、forward、处理结果；没有跨轮 result queue，也没有 CPU/GPU 错拍。

**Explain：** `event_loop_normal` 是 Scheduler 的最小正确循环。它牺牲 CPU/GPU overlap，换来状态时序最简单，适合调试竞态和理解主链路。

**Code：**

```python
## 来源：python/sglang/srt/managers/scheduler.py L1521-L1548
    def event_loop_normal(self):
        """A normal scheduler loop."""
        while True:
            if self.gracefully_exit:
                break

            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue

            # Get the next batch to run
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # When the server is idle, do self-check and re-init some states.
                self.on_idle()

            # Update last_batch
            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.invariant_checker.self_check_during_busy()
```

**代码逻辑：**
- 每轮先检查 `gracefully_exit`，再接收并 dispatch 请求。
- pause 状态下仍收请求和控制消息，但不组 batch。
- 有 batch 就同步 `run_batch` 后立刻 `process_batch_result`；无 batch 进入 `on_idle`。

**为什么这样写：** normal loop 是 overlap loop 的语义基准。只要 normal loop 正确，其他高性能路径都可以被理解为“同一组动作的重排与并行化”，而不是另一套调度语义。

**不变量与失败模式：**
- `last_batch = batch` 必须在轮尾更新，因为下一轮 prefill merge、overlap disable 判断都依赖它。
- pause 不能跳过 `recv_requests()`，否则 continue/pause/控制请求无法被消费。

**Comment：** 最简单路径：收 → 调度 → forward → 处理，无流水线重叠。

### 4.2 `event_loop_overlap` — 默认高性能路径

**问题与约束：** GPU forward 通常比 CPU 后处理更重，但 CPU 侧处理 result、stream token、更新 Req、释放 KV 仍会占用时间。串行处理会让 GPU 等 CPU。

**设计选择：** overlap loop 用 `result_queue` 保存上一轮 `(batch, result)`，当前 batch forward 后再处理上一轮结果；必要时通过 `is_disable_overlap_for_batch` 强制同步边界。

**Explain：** 用 `result_queue` 保存 `(batch, result)`，在当前 batch forward 的同时处理**上一轮**结果。采样（grammar 依赖）在 process 之后单独 `launch_batch_sample_if_needed`。

**Code：**

```python
## 来源：python/sglang/srt/managers/scheduler.py L1551-L1613（节选）
    def event_loop_overlap(self):
        """A scheduler loop that overlaps the CPU processing and GPU computation."""
        self.result_queue: Deque[
            Tuple[ScheduleBatch, Union[GenerationBatchResult, EmbeddingBatchResult]]
        ] = deque()

        def pop_and_process():
            # Process the results of the last batch
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        while True:
            if self.gracefully_exit:
                break

            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue

            self._apply_war_barrier()

            # Get the next batch to run
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch
            disable_overlap_for_batch = self.is_disable_overlap_for_batch(batch)

            # If we do not need to overlap the current batch with the last batch,
            # we can process the last batch immediately.
            if disable_overlap_for_batch:
                pop_and_process()
                # Opportunistic flush at the disable_overlap sync boundary:
                # forward_stream is idle (prev forward drained, next not launched),
                # so `_flush`'s non-urgent guard compacts freely. Sync-free, best-effort.
                if self.server_args.enable_unified_memory:
                    try:
                        self.token_to_kv_pool_allocator.flush_opportunistic()
                    except Exception:
                        pass

            # Launch the current batch
            if batch:
                batch_result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            # Process the last batch
            if self.last_batch:
                if not disable_overlap_for_batch:
                    pop_and_process()
            elif batch is None:
                # When the server is idle, do self-check and re-init some states
                self.on_idle()

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            if self.is_generation:
                self.launch_batch_sample_if_needed(batch_result)

            # Update last_batch
            self.last_batch = batch
```

**代码逻辑：**
- 每轮先收请求并处理控制面，然后执行 `_apply_war_barrier()`，防止 schedule stream 覆写上一轮 forward 仍在读的共享 buffer。
- 取出当前 batch 后判断是否要 disable overlap；如果要同步，先处理上一轮 result。
- 当前 batch forward 完成后 append 到 `result_queue`；若上一轮存在且本轮未禁用 overlap，就处理上一轮 result。
- generation 模式下延迟采样在上一轮 result 处理后执行，保证 grammar 看到最新状态。

**为什么这样写：** 这是一拍流水：GPU 做第 N 轮 forward 时，CPU 处理第 N-1 轮结果。它用状态复杂度换吞吐，代价是 batch 对象、GPU tensor 生命周期、D2H copy 和 grammar/speculative 的先后关系都必须被显式管理。

**不变量与失败模式：**
- `batch.copy()` 入队是为了避免后续 scheduler 修改 batch 影响 result processing。
- disable overlap 时必须先 drain `result_queue`，否则 grammar/spec decode 可能基于旧 token 状态继续采样。
- unified memory 的 opportunistic flush 只能在 forward stream 空闲边界做。

**Comment：**

- `_apply_war_barrier`：等待上一轮 forward 读完共享 buffer，避免 schedule stream 覆写。
- 连续两个 prefill batch 时可通过 `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP` 禁用 overlap，改善首 batch TTFT。

---

## 5. 组 Batch：`get_next_batch_to_run`

### 5.1 Merge prefill 进 running_batch

**问题与约束：** Prefill batch 完成后，请求还没结束，后续要进入 decode。Scheduler 必须把上一轮 EXTEND 的活请求合并进 `running_batch`，同时排除 chunked prefill 中间 chunk、DLLM staging 等不能立即 decode 的请求。

**设计选择：** `get_next_batch_to_run()` 每轮开头先处理上轮 prefill 的归并，再决定新 prefill 或 decode。这让 continuous batching 的状态转换集中在一个函数里。

**Explain：** 上一轮若是 EXTEND（prefill），完成后将未 finish 的请求 merge 进 `running_batch`，形成 continuous batching。

**Code：**

```python
## 来源：python/sglang/srt/managers/scheduler.py L2630-L2657
        if (
            not self.enable_hisparse
            and self.last_batch
            and self.last_batch.forward_mode.is_extend()
        ):
            if self.last_batch.chunked_req is not None:
                # In the context pipeline parallelism, after the last chunk, the current microbatch still track outdated chunked_req.
                # We need to discard it.
                chunked_req_to_exclude.add(self.last_batch.chunked_req)

            if self.dllm_config is not None and self.last_batch.reqs:
                chunked_req_to_exclude.update(self.last_batch.reqs)

            # Filter batch
            last_bs = self.last_batch.batch_size()
            self.last_batch.filter_batch(
                chunked_req_to_exclude=list(chunked_req_to_exclude)
            )
            if self.last_batch.batch_size() < last_bs:
                self.running_batch.batch_is_full = False

            # Merge the new batch into the running batch.
            if not self.last_batch.is_empty():
                if self.running_batch.is_empty():
                    self.running_batch = self.last_batch
                else:
                    # Merge running_batch with prefill batch
                    self.running_batch.merge_batch(self.last_batch)
```

**代码逻辑：**
- `chunked_req_to_exclude` 收集不应进入 running batch 的请求。
- `last_batch.filter_batch(...)` 删除已完成或应排除的请求；如果 batch 变小，重置 `batch_is_full`。
- running 为空则直接接管 `last_batch`，否则 merge 两个 batch。

**为什么这样写：** Prefill 和 decode 在 serving 中是同一请求的两个生命周期阶段。把 prefill 结果合并进 running batch，能让新请求的 prefill 和老请求的 decode 持续交错，避免“整批完成后再接下一批”的离线推理形态。

**不变量与失败模式：**
- chunked prefill 的中间 chunk 不能提前进入 decode，否则会用不完整 KV 继续生成。
- filter 后如果不重置 `batch_is_full`，Scheduler 可能误以为 running batch 已满而停止接纳新 prefill。

### 5.2 Prefill 组 batch（PrefillAdder）

**问题与约束：** Prefill admission 要同时满足 token budget、KV page、running batch 空间、chunked prefill、priority preemption、LoRA、HiCache prefetch、Mamba allocator 和 PD prefill 额外占用。简单按 waiting queue 顺序 append 会很快触发 OOM 或长尾。

**设计选择：** Scheduler 把 admission 交给 `PrefillAdder`，主循环负责准备上下文和遍历 waiting queue；`AddReqResult` 决定继续、停止或标记 batch full。

**Explain：** PrefillAdder 是 prefill 阶段的 admission controller，它决定 waiting queue 中哪些请求能安全进入本轮 EXTEND batch。

**Code：**

```python
## 来源：python/sglang/srt/managers/scheduler.py L2804-L2879（节选）
        # Prefill policy
        adder = PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            self.running_batch,
            self.new_token_ratio_tracker.current,
            self.max_prefill_tokens,
            chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
            self.priority_scheduling_preemption_threshold,
            max_prefill_bs=self.max_prefill_bs,
            max_running_requests=self.max_running_requests,
            prefill_max_requests=self.server_args.prefill_max_requests,
            prefill_delayer_single_pass=prefill_delayer_single_pass,
            dllm_config=self.dllm_config,
            waiting_queue_len=len(self.waiting_queue),
        )

        if self.chunked_req is not None:
            self.chunked_req.init_next_round_input()
            self.chunked_req = adder.add_chunked_req(self.chunked_req)

        if self.enable_lora:
            running_loras = {
                req.lora_id for req in self.running_batch.reqs if not req.finished()
            }
            # Account for LoRAs that are already loaded in the adder, such as chunked requests
            running_loras.update(req.lora_id for req in adder.can_run_list)

            if self.lora_drainer:
                self.lora_drainer.update_draining_state(
                    self.waiting_queue,
                    self.running_batch.reqs,
                )

        mamba_allocator = getattr(self.req_to_token_pool, "mamba_allocator", None)
        if mamba_allocator is not None:
            mamba_allocator.alloc_group_begin(len(self.waiting_queue))
        # Get requests from the waiting queue to a new prefill batch
        for req in self.waiting_queue:
            if self.enable_lora and not self._can_schedule_lora_req(req, running_loras):
                continue

            running_bs = len(self.running_batch.reqs)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                self.running_batch.batch_is_full = True
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                # In prefill mode, prealloc queue and transfer queue can also take memory,
                # so we need to check if the available size for the actual available size.
                if len(adder.can_run_list) >= self.req_to_token_pool.available_size():
                    self.running_batch.batch_is_full = True

            if self.running_batch.batch_is_full:
                if (
                    not self.enable_priority_preemption
                    or not adder.preempt_to_schedule(req, self.server_args)
                ):
                    break

            if self.enable_hicache_storage:
                prefetch_done = self.tree_cache.check_prefetch_progress(req.rid)
                if not prefetch_done:
                    # skip staging requests that are ongoing prefetch
                    continue
                # Pop the number of tokens loaded from storage (L3 hits)
                req.storage_hit_length = self.tree_cache.pop_prefetch_loaded_tokens(
                    req.rid
                )

            req.init_next_round_input(self.tree_cache)
            res = adder.add_one_req(
                req,
                has_chunked_req=(self.chunked_req is not None),
                truncation_align_size=self.truncation_align_size,
            )
```

**代码逻辑：**
- 先创建 `PrefillAdder`，把 page/KV/token/priority/chunk/LoRA 等约束注入进去。
- 如存在 `chunked_req`，它优先进入本轮 admission。
- waiting queue 逐个尝试：LoRA 不可调度则跳过，HiCache prefetch 未完成则暂缓，`add_one_req` 返回结果决定是否继续。
- 成功的请求从 waiting queue 移除，构造成新的 `ScheduleBatch` 并 `prepare_for_extend()`。

**为什么这样写：** Prefill 是 TTFT 与吞吐的核心冲突点。Scheduler 选择把复杂资源判断集中在 admission controller，而不是散落在 forward 前后；这样可以在同一个决策点处理 preemption、chunking、cache 命中和内存预算。

**不变量与失败模式：**
- `chunked_req` 必须优先处理，否则中间 chunk 可能被饿死或造成内存泄漏。
- Mamba allocator 的 group begin/end 必须成对出现，未加入 batch 的 mamba slot 要回滚。
- HiCache prefetch 未完成的请求不能被 staging，否则会把异步缓存加载和 GPU forward 顺序打乱。

**Comment：** `AddReqResult` 可能是 NO_TOKEN、STOP 等，决定 waiting_queue 是否继续填充本 batch（详见调度策略）。

### 5.3 Decode：`update_running_batch`

**问题与约束：** Decode 每步都会追加 KV，占用会随输出长度增长。即使 prefill admission 当时安全，后续 decode 也可能把 KV pool 吃满。

**设计选择：** 每轮 decode 前先 `check_decode_mem()`；不足时调用 `retract_decode()` 选择部分请求撤回，释放 KV slot，并通过 `_add_request_to_queue(..., is_retracted=True)` 重新进入调度生命周期。

**Explain：** `update_running_batch` 负责把 running batch 推进到下一轮 decode；它也是 decode KV 不足时触发 retract 的防线。

**Code：**

```python
## 来源：python/sglang/srt/managers/scheduler.py L3026-L3114（节选）
    def update_running_batch(self, batch: ScheduleBatch) -> Optional[ScheduleBatch]:
        """Update the current running decoding batch."""
        initial_bs = batch.batch_size()

        batch.filter_batch()
        if batch.is_empty():
            batch.batch_is_full = False
            return batch

        # Eagerly release lock_ref on completed write-through nodes so they
        # become evictable, improving batch scheduling headroom.
        if self.enable_hierarchical_cache:
            self.tree_cache.flush_write_through_acks()

        # Check if decode out of memory
        if (kv_full_retract_flag := not batch.check_decode_mem()) or (
            TEST_RETRACT and self.forward_ct % TEST_RETRACT_INTERVAL == 0
        ):
            old_available_tokens = self.token_to_kv_pool_allocator.available_size()
            old_ratio = self.new_token_ratio_tracker.current
            mamba_allocator = getattr(
                self.tree_cache.req_to_token_pool, "mamba_allocator", None
            )
            old_mamba_available = (
                mamba_allocator.available_size()
                if mamba_allocator is not None
                else None
            )
            retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode(
                self.server_args
            )
            new_available_tokens = self.token_to_kv_pool_allocator.available_size()
            new_token_gained = new_available_tokens - old_available_tokens
            mamba_num_gained = (
                mamba_allocator.available_size() - old_mamba_available
                if mamba_allocator is not None
                else None
            )

            self.metrics_reporter.num_retracted_reqs = len(retracted_reqs)
            if self.metrics_reporter.enable_metrics and len(retracted_reqs) > 0:
                self.metrics_reporter.metrics_collector.increment_retracted_reqs(
                    num_retracted_reqs=len(retracted_reqs),
                    num_retracted_input_tokens=sum(
                        len(r.origin_input_ids) for r in retracted_reqs
                    ),
                    num_retracted_output_tokens=sum(
                        len(r.output_ids) for r in retracted_reqs
                    ),
                )
            self.new_token_ratio_tracker.current = new_token_ratio
            for req in reqs_to_abort:
                abort_reason: FINISH_ABORT = req.to_finish
                self.ipc_channels.send_to_tokenizer.send_output(
                    AbortReq(
                        finished_reason=abort_reason.to_json(),
                        rid=req.rid,
                    ),
                    req,
                )

            msg_prefix = (
                "KV cache pool is full. Retract requests. "
                if kv_full_retract_flag
                else "Testing retraction. "
            )
            msg_details = f"#retracted_reqs: {len(retracted_reqs)}, #new_tokens_gained: {new_token_gained}"
            if mamba_num_gained is not None:
                msg_details += f", #mamba_num_gained: {mamba_num_gained}"
            if kv_full_retract_flag:
                msg_details += (
                    f", #new_token_ratio: {old_ratio:.4f} -> {new_token_ratio:.4f}"
                )
            logger.warning(msg_prefix + msg_details)

            for req in retracted_reqs:
                self._add_request_to_queue(req, is_retracted=True)
        else:
            self.new_token_ratio_tracker.decay_step()

        if batch.batch_size() < initial_bs:
            batch.batch_is_full = False

        if batch.is_empty():
            return batch

        # Update batch tensors
        batch.prepare_for_decode()
        return batch
```

**代码逻辑：**
- 先 filter 掉 finished 请求；空 batch 直接返回。
- hierarchical cache 下先 flush write-through ack，让可驱逐空间尽早释放。
- KV 不足时记录旧可用 token 和 new token ratio，执行 retract，更新 metrics，并把撤回请求重新入队。
- KV 足够时衰减 `new_token_ratio_tracker`，最后 `prepare_for_decode()` 更新 batch tensor。

**为什么这样写：** Decode OOM 在长输出 serving 中不是异常小概率，而是动态调度必须处理的常态。retract 用“重跑部分 prefill”的成本换“服务不中断和 KV pool 不爆”，体现 SGLang 的资源弹性哲学。

**不变量与失败模式：**
- retract 后必须更新 `new_token_ratio_tracker`，否则后续 PrefillAdder 会继续高估可承载 token。
- `reqs_to_abort` 要立即回传 TokenizerManager，否则前台协程会等待永不完成的 rid。
- `prepare_for_decode()` 必须在最终 batch 非空后调用，确保 input ids、seq lens、KV location 进入下一轮 forward 的正确形态。

**Comment：** KV 不足时 **retract**：部分请求退回 waiting_queue 重跑 prefill，释放 slot；这是 SGLang 应对 OOM 的核心机制。

---

## 6. GPU 前向：`run_batch`

### 6.1 Overlap 模式下的 forward

**问题与约束：** overlap 模式下，schedule stream、forward stream、copy stream 同时参与同一个 batch 生命周期；speculative decoding 还可能在 worker 内部中途 publish 下一轮输入。直接共享 batch 对象会产生竞态和 tensor 提前释放。

**设计选择：** `run_batch` 在 overlap 分支中用 `FutureMap` relay 下一轮 input ids，用 `_forward_isolation` 给 `ScheduleBatch` 做事务式快照，用 `batch_record_buf` 延长 GPU tensor 生命周期，并把 D2H copy 放到 `copy_stream`。

**Explain：** 在 `forward_stream` 上执行 `forward_batch_generation`；`future_map` 在 overlap 下 relay input_ids，避免 schedule stream 与 forward stream 竞态。

**Code：**

```python
## 来源：python/sglang/srt/managers/scheduler.py L3204-L3286（节选）
        if self.is_generation:
            if self.enable_overlap:
                # Self-gates on batch.spec_info.future_indices; non-spec_v2
                # no-ops (ForwardBatch.init_new lazily computes the sum).
                self.future_map.resolve_seq_lens_cpu(batch)

                with self.forward_stream_ctx:
                    self.forward_stream.wait_stream(self.schedule_stream)
                    # resolve consumes SB staging (prefill_input_ids_cpu /
                    # mix_running_indices). Run OUTSIDE isolation so the
                    # snapshot captures the post-consume state — restoring
                    # post-forward must not un-consume staging.
                    resolve_forward_inputs(batch, self.future_map)

                    with self._forward_isolation(batch, overlap=True):
                        future_indices = batch.req_pool_indices

                        # Spec_v2 fires on_publish mid-worker (between verify and
                        # draft_extend) so schedule prep can overlap with draft_extend.
                        # Non-spec has no later work — scheduler publishes after return.
                        fwd_kwargs = (
                            {
                                "on_publish": partial(
                                    self.future_map.publish, future_indices
                                )
                            }
                            if not batch.spec_algorithm.is_none()
                            else {}
                        )

                        # FIXME: pp is not compatible with overlap
                        batch_result = self.model_worker.forward_batch_generation(
                            batch, **fwd_kwargs
                        )
                        if batch.spec_algorithm.is_none():
                            self.future_map.publish(future_indices, batch.seq_lens + 1)
                        # Park any refs the worker wants kept alive 2 iters
                        # (cross-stream tensor lifetime; pinned in the same
                        # ring slot as the SB attr snapshot).
                        if batch_result.extra_keep_alive_refs:
                            self.batch_record_buf[self.batch_record_ct].extend(
                                batch_result.extra_keep_alive_refs
                            )
                        if self.server_args.enable_unified_memory:
                            # Record a `forward_done` event after the forward (before
                            # copy_to_cpu); lazy-compaction `_flush` gates src reuse on
                            # it. Only the unified pool's allocator exposes these hooks.
                            allocator = self.token_to_kv_pool_allocator
                            forward_done = self.device_module.Event()
                            forward_done.record(stream=self.forward_stream)
                            allocator.set_latest_forward_done_event(forward_done)
                            # Write-set classification: hand the allocator this
                            # forward's virtual out_cache_loc as a tensor ref (no GPU work).
                            allocator.set_inflight_forward(
                                forward_done,
                                batch.out_cache_loc,
                            )
                        # FIXME(lsyin): maybe move this to forward_batch_generation
                        batch_result.copy_done = self.device_module.Event()
                        if batch_result.delay_sample_func is None:
                            self._relay_forward_payload(future_indices, batch_result)
                            if _is_hip:
                                # Cross-stream sync costs more than the tiny D2H it
                                # overlaps.
                                batch_result.copy_to_cpu(
                                    return_logprob=batch.return_logprob,
                                    return_hidden_states=batch.return_hidden_states,
                                )
                            else:
                                # Result D2H on copy_stream overlaps the next forward
                                # instead of serializing on forward_stream; it's a leaf
                                # gated by copy_done, so nothing on forward_stream waits.
                                self.copy_stream.wait_stream(self.forward_stream)
                                with self.copy_stream_ctx:
                                    batch_result.copy_to_cpu(
                                        return_logprob=batch.return_logprob,
                                        return_hidden_states=batch.return_hidden_states,
                                    )
                        else:
                            batch_result.future_indices = future_indices

                # Next-iter input_ids relayed via future_map.
                batch.input_ids = None
```

**代码逻辑：**
- `future_map.resolve_seq_lens_cpu(batch)` 先补齐 spec/relay 需要的 CPU seq lens。
- `forward_stream.wait_stream(schedule_stream)` 保证 schedule 准备完成后再 forward。
- `resolve_forward_inputs` 消费 staging buffer，然后 `_forward_isolation` 快照 batch 字段。
- forward 后发布下一轮输入，记录 unified memory forward_done event，并在 copy stream 上 D2H。
- 最后把 `batch.input_ids = None`，强制下一轮通过 FutureMap relay 重建。

**为什么这样写：** overlap 的本质是让 CPU 调度提前准备下一轮，但 GPU forward 还在读上一轮 batch 的部分 buffer。SGLang 选择显式 relay 和快照，而不是让 batch 对象跨 stream 随意共享；这让吞吐提升可控，也把竞态集中在少数工具类里。

**不变量与失败模式：**
- `resolve_forward_inputs` 必须在 isolation 外执行，否则恢复快照会把已经消费的 staging 状态“倒回去”。
- `batch_record_buf` 至少保留两轮引用，否则 GPU tensor 可能被 Python GC 提前释放。
- HIP 分支不走 copy stream overlap，因为跨 stream sync 成本超过收益。

**Comment：** D2H 在 `copy_stream` 上与下一轮 forward 重叠；speculative decoding 时 sampling 可能延迟到 `launch_batch_sample_if_needed`。

---

## 7. Pipeline Parallelism：`SchedulerPPMixin`

### 7.1 `event_loop_pp` 骨架

**问题与约束：** Pipeline Parallelism 下，一个 batch 要穿过多个 stage，非末 rank 只产出 proxy tensors，末 rank 才有 logits/output。普通 overlap loop 的单进程 `result_queue` 无法表达跨 stage microbatch、proxy 通信和 last-rank output 回传。

**设计选择：** PP 使用独立 `event_loop_pp`，每个 stage 维护 `pp_loop_size = pp_size + pp_async_batch_depth` 个 microbatch 槽位；请求、proxy、output 分别用 pyobj / typed tensor dict 通信，last stage 可用 async batch depth 缓冲输出。

**Explain：** PP 下每个 stage 维护多个 microbatch（`pp_loop_size`）。Stage P 从上一 stage recv 请求/proxy，本地 `get_next_batch_to_run`，再 send 到下一 stage。Last stage 处理 output 并可能 overlap 与 GPU 计算。

**Code：**

```python
## 来源：python/sglang/srt/managers/scheduler_pp_mixin.py L92-L168（节选）
        self.init_pp_loop_state()
        while True:
            server_is_idle = True
            for mb_id in range(self.pp_loop_size):
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = self.last_mbs[mb_id]
                next_first_rank_mb_id = (mb_id + self.ps.pp_size) % self.pp_loop_size
                next_mb_id = (mb_id + 1) % self.pp_loop_size
                with torch.profiler.record_function("recv_requests"):
                    recv_reqs = self.request_receiver.recv_requests()
                    self.process_input_requests(recv_reqs)
                if not self.pp_group.is_last_rank:
                    self._pp_commit_comm_work(self.send_req_work)
                    with torch.profiler.record_function("send_reqs_to_next_stage"):
                        self.send_req_work = self._pp_send_pyobj_to_next_stage(
                            recv_reqs,
                            async_send=True,
                        )
                with torch.profiler.record_function("get_next_batch_to_run"):
                    self.mbs[mb_id] = self.get_next_batch_to_run()
                self.running_mbs[mb_id] = self.running_batch
                self.cur_batch: Optional[ScheduleBatch] = self.mbs[mb_id]
                if self.cur_batch:
                    server_is_idle = False
                    pp_proxy_tensors = self._pp_recv_proxy_tensors()
                next_pp_outputs = None
                next_batch_result = None
                d2h_event = None
                if self.server_args.pp_async_batch_depth > 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                self._pp_commit_comm_work(self.send_proxy_work)
                if self.cur_batch:
                    result, self.launch_event = self._pp_launch_batch(
                        mb_id,
                        pp_proxy_tensors,
                        self.mb_metadata,
                        self.last_rank_comm_queue,
                    )
                if self.server_args.pp_async_batch_depth == 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                if self.mbs[next_mb_id] is not None:
                    d2h_event.synchronize()
                    with torch.profiler.record_function("process_batch_result"):
                        self._pp_process_batch_result(
                            self.mbs[next_mb_id],
                            next_batch_result,
                        )
                    self.last_mbs[next_mb_id] = self.mbs[next_mb_id]
                if not self.pp_group.is_last_rank:
                    if self.cur_batch:
                        self.device_module.current_stream().wait_event(
                            self.launch_event
                        )
                        with torch.profiler.record_function(
                            "send_proxy_dict_to_next_stage"
                        ):
                            self.send_proxy_work = self._pp_send_dict_to_next_stage(
                                result.pp_hidden_states_proxy_tensors.tensors,
                                async_send=True,
                                msg_type="proxy",
                            )

                self.pp_outputs = next_pp_outputs

            # When the server is idle, self-check and re-init some states
            if server_is_idle:
                self.on_idle()
```

**代码逻辑：**
- 每个 `mb_id` 恢复对应 `running_mbs/last_mbs`，让 microbatch 有独立 running 状态。
- 非末 rank 先 commit 上一轮 request send，再异步发送本轮请求到下一 stage。
- 当前 stage 收 proxy、launch batch；同时按配置提前或延后处理下一个 microbatch 的 output。
- 非末 rank forward 完成后把 hidden-state proxy 发送给下一 stage；last rank 把 output 放入队列回传 rank0。

**为什么这样写：** PP 的性能问题不是单机 CPU/GPU overlap，而是 stage 间 bubble 和通信等待。独立循环把“同一请求跨 stage 流动”和“不同 microbatch 交错执行”显式化，牺牲代码简单性换取流水线填充度。

**不变量与失败模式：**
- request/proxy/output 三类 tensor dict 必须带 `msg_type`，否则 recv 顺序错位会把 proxy 当 output。
- async send 后必须在复用 buffer 前 `_pp_commit_comm_work`，否则通信还没完成就覆盖数据。
- `d2h_event.synchronize()` 必须在 process output 前执行，保证 CPU 读取的 result 已完成拷贝。

**Comment：** PP 使用 **async send + sync recv** 减少通信 CPU stall；`pp_async_batch_depth` 允许 last stage 缓冲 output 与计算重叠。

### 7.2 Chunked prefill 与 PP 的 output comm 优化

**问题与约束：** 纯 chunked prefill 的中间 chunk 只是在构建 KV，不会产生最终用户可见 token。如果每个中间 chunk 都从 last rank 把 output 回传 rank0，会增加 PP bubble 和通信量。

**设计选择：** `_pp_can_skip_output_comm` 只在“EXTEND、单请求、非最后 prefill chunk、无需 logprob”时允许跳过 output comm；否则仍走完整回传。

**Explain：** 这是一个严格保护正确性的通信剪枝：只有当 output 对用户不可见、对后续调度也不需要时才跳过。

**Code：**

```python
## 来源：python/sglang/srt/managers/scheduler_pp_mixin.py L49-L57
def _pp_can_skip_output_comm(batch: ScheduleBatch) -> bool:
    """Check if output send/recv can be skipped for this batch."""
    return (
        envs.SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM.get()
        and batch is not None
        and batch.forward_mode == ForwardMode.EXTEND
        and len(batch.reqs) == 1
        and not batch.contains_last_prefill_chunk
        and not batch.return_logprob
```

**代码逻辑：**
- 环境变量开启后才允许优化。
- batch 必须是 EXTEND，且只有一个请求。
- `contains_last_prefill_chunk` 为 false，说明还不是最后 chunk。
- `return_logprob` 为 false，避免跳过用户需要的中间 logprob 信息。

**为什么这样写：** PP 优化的原则是减少 bubble 可以，但不能改变请求可观测语义。中间 chunk 的 output 对用户不可见，跳过它能减少 last-rank 回传压力；最后 chunk 和 logprob 请求则必须保守。

**不变量与失败模式：**
- 最后一个 prefill chunk 不能跳过，否则 rank0 无法完成 prefill result processing。
- logprob 请求不能跳过，否则返回内容不完整。

**Comment：** 纯 chunked prefill 中间 chunk 无需把完整 output 传回 rank0，可跳过通信降低 PP bubble。

---

## 8. Overlap 基础设施：`init_overlap`

**问题与约束：** `run_batch` 的 overlap 分支依赖设备 stream、FutureMap、copy stream、batch lifetime buffer。即使关闭 overlap，部分路径仍需要 FutureMap relay decode input ids；PP 非 overlap 也复用 forward/copy stream。

**设计选择：** `init_overlap` 无条件创建 FutureMap；非 MLX 下无条件创建 `forward_stream_ctx` 和 `copy_stream`；只有真正启用 overlap 时才创建 `batch_record_buf`。

**Explain：** `init_overlap` 不是只服务默认 overlap loop，而是为 decode input relay、PP stream 复用、D2H copy overlap 和 GPU tensor 生命周期打底。

**Code：**

```python
## 来源：python/sglang/srt/managers/scheduler.py L1239-L1282
    def init_overlap(self):
        self.device_module = torch.get_device_module(self.device)

        # FutureMap is always-on: input_ids relay used in both modes.
        # Workers without the spec_v2_attn_backends override fall back to
        # target-only so the helper still produces a safe decision (no
        # accidental opt-out for unaudited shapes).
        if self.draft_worker is not None:
            attn_backends = getattr(
                self.draft_worker,
                "spec_v2_attn_backends",
                (self.tp_worker.model_runner.attn_backend,),
            )
        else:
            attn_backends = (self.tp_worker.model_runner.attn_backend,)
        needs_cpu_seq_lens = decide_needs_cpu_seq_lens(self.server_args, attn_backends)
        self.future_map = self.spec_algorithm.create_future_map(
            self.device,
            self.req_to_token_pool,
            needs_cpu_seq_lens=needs_cpu_seq_lens,
        )

        if use_mlx():
            # MLX uses its own overlap loop and does not create CUDA streams,
            # but the normal non-overlap scheduler path still relays decode
            # input IDs through FutureMap.
            self.result_queue: Deque = deque()
            return

        # forward_stream_ctx / copy_stream are also used by PP (non-overlap)
        # via scheduler_pp_mixin; init unconditionally to match main.
        self.forward_stream_ctx: CudaStreamContext = self.device_module.stream(
            self.forward_stream
        )
        self.copy_stream: CudaStream = self.device_module.Stream()
        self.copy_stream_ctx: CudaStreamContext = self.device_module.stream(
            self.copy_stream
        )

        if not self.enable_overlap:
            return

        self.batch_record_buf = [None] * 2
        self.batch_record_ct = 0
```

**代码逻辑：**
- 根据 draft worker 或 target worker 的 attention backend 决定 FutureMap 是否需要 CPU seq lens。
- MLX 走自己的 overlap loop，不创建 CUDA stream，但仍保留 `result_queue` 和 FutureMap 语义。
- PyTorch 后端创建 forward stream context 与 copy stream context，供 overlap 和 PP 共同使用。
- `enable_overlap` 为 false 时提前返回，不分配 batch record buffer。

**为什么这样写：** SGLang 把“下一轮输入 relay”提升为通用基础设施，而不是只绑在 overlap 上。这样 non-overlap spec、PP、MLX 都可以复用同一套输入交接语义；真正与跨 stream 生命周期相关的 buffer 则只在 overlap 下启用。

**不变量与失败模式：**
- FutureMap 必须早于任何 `resolve_forward_inputs` 使用。
- `batch_record_buf` 只在 overlap 下存在，非 overlap 路径调用 `_forward_isolation(..., overlap=False)` 不能依赖它。
- copy stream 用于 D2H overlap；如果错误地让 forward stream 等 copy，会抵消 overlap 收益。

**Comment：** `FutureMap` 在 overlap 与非 overlap 模式下均用于 decode input_ids relay；`batch_record_buf` 防止 GPU tensor 被 GC 提前释放。

---

## 走读小结

| 阶段 | 关键函数 | 输出 |
|------|----------|------|
| 启动 | `run_scheduler_process` | Scheduler 子进程 + handshake |
| 收包 | `recv_requests` → `process_input_requests` | Req 入 waiting/disagg 队列 |
| 调度 | `get_next_batch_to_run` | `ScheduleBatch`（prefill 或 decode） |
| 计算 | `run_batch` | `GenerationBatchResult` |
| 后处理 | `process_batch_result` | 更新 Req、stream token、释放 KV |
| PP | `event_loop_pp` | 跨 stage proxy + microbatch |
