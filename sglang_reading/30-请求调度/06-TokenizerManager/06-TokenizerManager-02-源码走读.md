---
type: batch-doc
module: 06-TokenizerManager
batch: "06"
doc_type: walkthrough
title: "TokenizerManager · 源码走读"
tags:
 - sglang/batch/06
 - sglang/module/tokenizer-manager
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-05
---
# TokenizerManager · 源码走读

> 走读顺序：初始化子系统 → ZMQ 通道 → `generate_request` 前台协程 → 分词与 tokenized object → 发送 Scheduler → 后台 `handle_loop` 接收输出 → 前台 `_wait_one_response` 返回 HTTP → score/control/multi-worker 扩展。

---

## 1. 初始化与 IPC 边界

### 1.1 构造函数按子系统初始化

**问题与约束：** TokenizerManager 同时负责模型配置、tokenizer / processor、IPC、运行状态、日志、权重更新、LoRA、disaggregation、metrics 和 request dispatcher；如果都写在一个大构造函数里，会很难被 `TokenizerWorker` 等子类复用或覆写。

**设计选择：** `__init__` 只保存基础 args，并按固定顺序调用 `init_*` 方法初始化各子系统。

**Explain：** 这是一个分层启动流程：先把全局 server args 和模型/tokenizer 能力定下来，再建立通信和运行时状态。

来源：python/sglang/srt/managers/tokenizer_manager.py L257-L297

**Code：**

```python
def __init__(
    self,
    server_args: ServerArgs,
    port_args: PortArgs,
):
    self.server_args = server_args
    self.enable_metrics = server_args.enable_metrics
    self.preferred_sampling_params = server_args.preferred_sampling_params
    self.crash_dump_folder = server_args.crash_dump_folder
    set_global_server_args_for_tokenizer(server_args)

    self.init_model_config()
    self.init_tokenizer_and_processor()
    self.init_ipc_channels(port_args)
    self.init_running_status()
    self.init_request_logging_and_dumping()
    self.init_weight_update()
    self.init_lora()
    self.init_disaggregation()
    self.init_metric_collector_watchdog()
    self.init_request_dispatcher()
```

**代码逻辑：** 基础配置先落到实例字段和 tokenizer 侧全局 args；随后依次初始化模型配置、tokenizer、多模态 processor、IPC、运行状态、日志、权重更新、LoRA、解聚合、metrics/watchdog 和 result dispatcher。

**为什么这样写：** TokenizerManager 是 HTTP/data-plane 的汇聚点，启动顺序错了会影响后续请求路径；拆成 `init_*` 也让多 worker 和测试场景可复用局部初始化。

**不变量与失败模式：** `init_ipc_channels` 之前必须已有 `server_args`；`init_request_dispatcher` 之后后台接收循环才能正确分派非 batch output。

**Comment：** 读构造函数时不要把它看成普通字段初始化，它定义了 TokenizerManager 的职责边界。

### 1.2 ZMQ 通道与 Scheduler 分发

**问题与约束：** TokenizerManager 一边要从 Detokenizer 接收结果，一边要把 tokenized request 发给 Scheduler；多 HTTP worker 模式下还要给请求打上 worker IPC，以便结果回到正确进程。

**设计选择：** 创建 `recv_from_detokenizer` PULL socket 和 `send_to_scheduler` PUSH socket；单 worker 直连 Scheduler，多 worker 发给 tokenizer worker router；分发前按需 `stamp_http_worker_ipc`。

**Explain：** IPC 通道把前台 HTTP 入口和后端 Scheduler/Detokenizer 进程解耦。

来源：python/sglang/srt/managers/tokenizer_manager.py L382-L413

**Code：**

```python
def init_ipc_channels(self, port_args: PortArgs):
    context = zmq.asyncio.Context(2)
    self.recv_from_detokenizer = get_zmq_socket(
        context, zmq.PULL, port_args.tokenizer_ipc_name, True
    )
    if self.server_args.tokenizer_worker_num == 1:
        self.send_to_scheduler = get_zmq_socket(
            context, zmq.PUSH, port_args.scheduler_input_ipc_name, True
        )
        self.tokenizer_ipc_name = None
    else:
        self.send_to_scheduler = get_zmq_socket(
            context, zmq.PUSH, port_args.tokenizer_worker_ipc_name, False
        )
        self.tokenizer_ipc_name = port_args.tokenizer_ipc_name

def _dispatch_to_scheduler(self, obj: Any) -> None:
    if self.tokenizer_ipc_name is not None:
        stamp_http_worker_ipc(obj, self.tokenizer_ipc_name)
    sock_send(self.send_to_scheduler, obj)
```

**代码逻辑：** 接收 socket 固定从 tokenizer IPC 收 Detokenizer 输出；发送 socket 根据 worker 数选择 Scheduler 输入或 worker router；`_dispatch_to_scheduler` 是同步发送入口，异步版本同样先 stamp 再 `async_sock_send`。

**为什么这样写：** 多 worker 下 Scheduler/Detokenizer 不能只凭 rid 知道结果应该回哪个 HTTP worker；`http_worker_ipc` 是结果路由的必要元数据。

**不变量与失败模式：** 多 worker 模式下发送前必须 stamp；如果缺失 `http_worker_ipc`，Detokenizer 或 Router 无法把结果分发回原请求进程。

**Comment：** 这段和 [[05-gRPC-Proto-02-源码走读]] 的 IPC struct 设计可以对照看。

---

## 2. generate_request：前台数据面主协程

### 2.1 入口归一化、pause、reader lock 和发送

**问题与约束：** HTTP generate、OpenAI 兼容层、Engine API 都会汇入同一生成入口；请求可能是单条或 batch，可能遇到 pause、权重更新、LoRA 校验、EPD encode 或输入错误。

**设计选择：** `generate_request` 先启动后台接收 loop，归一化请求并注册 req state；在 pause condition 和 model update reader lock 保护下分词、发送 Scheduler，然后等待输出。

**Explain：** 这是 TokenizerManager 的前台请求生命周期：注册状态 → 进入安全区 → tokenized request 入 Scheduler → async generator 产出响应。

来源：python/sglang/srt/managers/tokenizer_manager.py L589-L636

**Code：**

```python
async def generate_request(
    self,
    obj: Union[GenerateReqInput, EmbeddingReqInput],
    request: Optional[fastapi.Request] = None,
):
    self.auto_create_handle_loop()

    obj.normalize_batch_and_arguments()
    self._set_default_priority(obj)

    self._init_req_state(obj, request)
    try:
        if self.server_args.language_only:
            self._handle_epd_disaggregation_encode_request(obj)

        self.request_logger.log_received_request(obj, self.tokenizer, request)

        async with self.is_pause_cond:
            await self.is_pause_cond.wait_for(lambda: not self.is_pause)

        async with self.model_update_lock.reader_lock:
            await self._validate_and_resolve_lora(obj)

            if obj.is_single:
                tokenized_obj = await self._tokenize_one_request(obj)
                state = self.rid_to_state[obj.rid]
                if obj.return_prompt_token_ids:
                    state.prompt_token_ids = list(tokenized_obj.input_ids)
                self._send_one_request(tokenized_obj)
                async for response in self._wait_one_response(obj, request):
                    yield response
            else:
                async for response in self._handle_batch_request(obj, request):
                    yield response
```

**代码逻辑：** `auto_create_handle_loop` 保证后台收包协程存在；请求归一化后先建 rid state；pause condition 阻止新请求进入；reader lock 避免与权重写更新冲突；单请求和 batch 请求走不同发送/等待路径。

**为什么这样写：** TokenizerManager 必须同时保证数据面吞吐和权重更新一致性；reader lock 允许多个请求并发读当前权重版本，但会被写更新阻塞。

**不变量与失败模式：** `_init_req_state` 必须早于发送，否则后端返回时找不到 state；发送前必须完成 LoRA 校验和 tokenization。

**Comment：** 前台协程只负责“送进去”和“等出来”，真正输出接收由后台 `handle_loop` 完成。

### 2.2 异常清理 rid_to_state

**问题与约束：** `_init_req_state` 在 try 内主要逻辑之前已经创建 state；如果输入长度校验或分词阶段失败，请求从未进入 Scheduler，正常的输出回收路径不会触发。

**设计选择：** `except Exception` 中调用 `_discard_pending_req_states(obj)`，再重新抛出异常。

**Explain：** 这是防止前置失败导致 `rid_to_state` 泄漏的补偿路径。

来源：python/sglang/srt/managers/tokenizer_manager.py L637-L646

**Code：**

```python
except Exception:
    self._discard_pending_req_states(obj)
    raise
```

**代码逻辑：** 任何异常都会尝试丢弃仍处于 pending 的 req state；已经由正常完成路径移除的 state，pop 是 no-op。

**为什么这样写：** 请求 state 的正常生命周期由 Scheduler/Detokenizer 输出结束；但分词前失败没有输出事件，必须由入口自己清理。

**不变量与失败模式：** 只清理 pending state，不能误删已经交给后端且可能还会返回结果的 state；否则后端结果到达时会找不到 rid。

**Comment：** 如果线上看到 `rid_to_state` 持续增长，分词/校验异常路径是第一处要查的地方。

---

## 3. 分词与 tokenized object

### 3.1 文本分词策略选择

**问题与约束：** 输入可能是单字符串、batch 文本或 cross-encoder；tokenizer 可能是 fast tokenizer，也可能是慢速 tokenizer；还可能启用 async dynamic batch tokenizer。

**设计选择：** 单字符串且 async dynamic batch tokenizer 可用时走动态批分词；否则 fast tokenizer 走 batch encode，慢 tokenizer 退回逐条 `encode`。

**Explain：** `_tokenize_texts` 的核心是选择成本最低且语义匹配的 tokenizer 调用方式。

来源：python/sglang/srt/managers/tokenizer_manager.py L757-L786

**Code：**

```python
use_async_tokenizer = (
    self.async_dynamic_batch_tokenizer is not None
    and input_format == InputFormat.SINGLE_STRING
)

if use_async_tokenizer:
    result = await self.async_dynamic_batch_tokenizer.encode(
        tokenizer_input[0], **tokenizer_kwargs
    )
    input_ids = [result["input_ids"]]
    token_type_ids = (
        [result["token_type_ids"]]
        if is_cross_encoder and result.get("token_type_ids")
        else None
    )
else:
    if not is_cross_encoder and (not getattr(self.tokenizer, "is_fast", False)):
        input_ids = [self.tokenizer.encode(t) for t in tokenizer_input]
        token_type_ids = None
    else:
        encoded = self.tokenizer(tokenizer_input, **tokenizer_kwargs)
        input_ids = encoded["input_ids"]
        token_type_ids = (
            encoded.get("token_type_ids") if is_cross_encoder else None
        )
```

**代码逻辑：** 动态批 tokenizer 只用于单字符串；cross-encoder 需要保留 token_type_ids；慢 tokenizer 不走 batch call，直接循环 encode。

**为什么这样写：** Python tokenizer 开销在高并发短请求里很明显；动态批能合并短窗口请求，但 batch / cross-encoder 语义需要保守处理。

**不变量与失败模式：** 返回结果最终要恢复到统一 batch 形态；cross-encoder 的 token_type_ids 不能丢。

**Comment：** TokenizerManager 的性能瓶颈不只在模型后端，分词策略也会影响吞吐。

### 3.2 单请求输入优先级

**问题与约束：** 一个请求可能提供 `input_embeds`、`input_ids` 或 text；多模态音频场景可能没有文本；`skip_tokenizer_init=True` 时不能接收 text。

**设计选择：** `_tokenize_one_request` 按 `input_embeds` > `input_ids` > text 分词的优先级处理；`input_embeds` 要求禁用 radix cache；空文本多模态交给 processor 后续填充。

**Explain：** 单请求分词先决定“输入来源”，再进入多模态和校验逻辑。

来源：python/sglang/srt/managers/tokenizer_manager.py L793-L832

**Code：**

```python
async def _tokenize_one_request(
    self,
    obj: Union[GenerateReqInput, EmbeddingReqInput],
):
    input_embeds = None
    input_text = obj.text
    token_type_ids = None
    is_cross_encoder_request = (
        isinstance(obj, EmbeddingReqInput) and obj.is_cross_encoder_request
    )
    if obj.input_embeds is not None:
        if not self.server_args.disable_radix_cache:
            raise ValueError(
                "input_embeds is provided while disable_radix_cache is False. "
                "Please add `--disable-radix-cache` when you launch the server "
                "if you want to use input_embeds as inputs."
            )
        input_embeds = obj.input_embeds
        input_ids = obj.input_ids
    elif obj.input_ids is not None:
        input_ids = obj.input_ids
    else:
        if self.tokenizer is None:
            raise ValueError(...)
        if not input_text and self.mm_processor and obj.contains_mm_input():
            input_ids = []
        else:
            input_ids, token_type_ids = await self._tokenize_texts(
                input_text, is_cross_encoder_request
            )
```

**代码逻辑：** embedding 输入优先；token id 直传次之；文本路径要求 tokenizer 存在；音频等多模态空文本先用空 ids 占位。

**为什么这样写：** Radix cache 以 token id 序列为 key，任意 embedding 输入无法安全复用；空文本多模态也不能简单当作无输入。

**不变量与失败模式：** 使用 `input_embeds` 时必须禁用 radix cache；`skip_tokenizer_init=True` 的 engine 不能接收 text prompt。

**Comment：** 这段是判断一个请求究竟由谁产生 `input_ids` 的入口。

### 3.3 构造 TokenizedGenerateReqInput

**问题与约束：** Scheduler 需要的是结构化 IPC 对象，不是原始 HTTP 请求；sampling params 要合并默认值、normalize、verify，token ids 也要转成更适合 IPC 的数组。

**设计选择：** `_create_tokenized_object` 把 `input_ids` 转为 `array("q")`，合并 preferred sampling params 和请求 sampling params，构造 `TokenizedGenerateReqInput`。

**Explain：** 这是从“API 请求对象”到“Scheduler IPC 请求对象”的边界。

来源：python/sglang/srt/managers/tokenizer_manager.py L1122-L1160

**Code：**

```python
input_ids_arr: Optional[array[int]] = (
    array("q", input_ids) if input_ids is not None else None
)
if self.preferred_sampling_params:
    sampling_kwargs = {**self.preferred_sampling_params, **obj.sampling_params}
else:
    sampling_kwargs = obj.sampling_params
sampling_params = self.sampling_params_class(**sampling_kwargs)
sampling_params.normalize(self.tokenizer)
sampling_params.verify(self.model_config.vocab_size)

if isinstance(obj, GenerateReqInput):
    session_params = (
        SessionParams(**obj.session_params) if obj.session_params else None
    )

    bootstrap_room = obj.bootstrap_room
    if (
        bootstrap_room is None
        and self.server_args.disaggregation_transfer_backend == "fake"
    ):
        bootstrap_room = self.fake_bootstrap_room_counter
        self.fake_bootstrap_room_counter += 1

    tokenized_obj = TokenizedGenerateReqInput(
        input_text=input_text,
        input_ids=input_ids_arr,
        mm_inputs=mm_inputs,
        sampling_params=sampling_params,
        return_logprob=obj.return_logprob,
        stream=obj.stream,
        rid=obj.rid,
```

**代码逻辑：** token ids 转 C array；preferred params 作为默认值被显式请求参数覆盖；sampling params 做 tokenizer 相关 normalize 和 vocab 校验；fake disagg backend 自动分配 bootstrap room。

**为什么这样写：** Scheduler 侧不应再解析 HTTP 级 sampling 参数，也不应接收 Python list 形态的大量 token id。

**不变量与失败模式：** `sampling_params.verify` 必须在发送前通过；fake bootstrap room 需要单调递增避免房间号冲突。

**Comment：** 这里生成的结构和 [[09-ScheduleBatch-IO-02-源码走读]] 里的 `TokenizedGenerateReqInput` 对上。

---

## 4. 发送到 Scheduler

### 4.1 单请求和批请求发送

**问题与约束：** 发送前要记录 dispatch time、处理共享内存多模态特征、包装不可 msgpack 的字段；发送后还要恢复本地 time_stats 对象继续记录完成时间。

**设计选择：** `_send_one_request` 对单请求执行 time stats、`wrap_shm_features`、`wrap_pickle_fields`、dispatch；批请求先逐个 wrap，再封装为 batch IPC 对象。

**Explain：** 发送阶段不只是 socket send，而是 IPC 前的最后一次对象整理。

来源：python/sglang/srt/managers/tokenizer_manager.py L1331-L1363

**Code：**

```python
def _send_one_request(
    self,
    tokenized_obj: Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput],
):
    tokenized_obj.time_stats.set_api_server_dispatch_time()
    tokenized_obj = wrap_shm_features(tokenized_obj)
    time_stats = tokenized_obj.time_stats
    tokenized_obj.wrap_pickle_fields()
    self._dispatch_to_scheduler(tokenized_obj)
    tokenized_obj.time_stats = time_stats
    tokenized_obj.time_stats.set_api_server_dispatch_finish_time()

def _send_batch_request(
    self,
    tokenized_objs: List[
        Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput]
    ],
):
    set_time_batch(tokenized_objs, "set_api_server_dispatch_time")
    time_stats = [tokenized_obj.time_stats for tokenized_obj in tokenized_objs]
    for tokenized_obj in tokenized_objs:
        tokenized_obj.wrap_pickle_fields()

    if isinstance(tokenized_objs[0], TokenizedGenerateReqInput):
        batch_req = BatchTokenizedGenerateReqInput(batch=tokenized_objs)
    else:
        batch_req = BatchTokenizedEmbeddingReqInput(batch=tokenized_objs)

    self._dispatch_to_scheduler(batch_req)
```

**代码逻辑：** 单请求先可能把大特征转 shared memory；pickle 包装会改变字段形态，所以本地保存 time_stats 引用并在发送后恢复；批请求按首个对象类型选择 batch wrapper。

**为什么这样写：** Scheduler 要收到可跨进程传输的对象；TokenizerManager 本地仍要保留可继续更新的 observability 对象。

**不变量与失败模式：** batch 内对象类型应一致；`wrap_pickle_fields` 必须在 dispatch 前调用，否则 opaque 字段会破坏 IPC 编码。

**Comment：** 多模态性能问题常常和这里的 shared memory / pickle 边界有关。

---

## 5. 后台接收与前台等待

### 5.1 auto_create_handle_loop 和 handle_loop

**问题与约束：** 前台 `generate_request` 在等待 response 时不能自己阻塞接收 socket；多个请求共享同一个 Detokenizer/Scheduler 输出通道，需要后台循环统一收包并分派。

**设计选择：** 首次请求时创建 `handle_loop` 任务和 sigterm watchdog；`handle_loop` 从 `recv_from_detokenizer` 收对象，batch output 走 `_handle_batch_output`，其他控制输出走 `_result_dispatcher`。

**Explain：** TokenizerManager 使用“前台请求协程 + 后台收包协程”的双协程模型。

来源：python/sglang/srt/managers/tokenizer_manager.py L1822-L1860

**Code：**

```python
def auto_create_handle_loop(self):
    if self.event_loop is not None:
        return

    loop = get_or_create_event_loop()
    self.asyncio_tasks.add(
        loop.create_task(print_exception_wrapper(self.handle_loop))
    )
    self.event_loop = loop

    if threading.current_thread() is threading.main_thread():
        signal_handler = self.signal_handler_class(self)
        loop.add_signal_handler(signal.SIGTERM, signal_handler.sigterm_handler)
        loop.add_signal_handler(
            signal.SIGQUIT, signal_handler.running_phase_sigquit_handler
        )

    self.asyncio_tasks.add(
        loop.create_task(print_exception_wrapper(self.sigterm_watchdog))
    )

async def handle_loop(self):
    while True:
        with self.soft_watchdog.disable():
            recv_obj = await async_sock_recv(self.recv_from_detokenizer)
        if isinstance(
            recv_obj,
            (BatchStrOutput, BatchEmbeddingOutput, BatchTokenIDOutput),
        ):
            await self._handle_batch_output(recv_obj)
        else:
            self._result_dispatcher(recv_obj)
        self.last_receive_tstamp = real_time()
        self.soft_watchdog.feed()
```

**代码逻辑：** loop 只创建一次；主线程才注册 signal handler；后台循环 await ZMQ 输出；batch output 更新请求 state，控制类输出交给类型分发器。

**为什么这样写：** 多个 HTTP 请求都要等同一个输出通道，集中收包能避免多个协程竞争同一 socket。

**不变量与失败模式：** `handle_loop` 必须在请求发送前存在；如果后台 task 崩溃，前台 `_wait_one_response` 会一直等不到 event。

**Comment：** `auto_create_handle_loop()` 也会被控制面 API 调用，因为控制回复同样从这个通道回来。

### 5.2 _handle_batch_output：写 ReqState 和流式 delta

**问题与约束：** 后端返回的是 batch output；TokenizerManager 要按 rid 找到对应 state，更新文本、output_ids、meta_info，并根据 streaming / incremental 模式决定返回给前台的 chunk 形态。

**设计选择：** `_handle_batch_output` 对每个请求设置 `state.finished`，`BatchStrOutput` 下追加文本和 token ids；incremental streaming 返回 delta，非 incremental 中间步返回 `text=None` 避免重复构造全量字符串。

**Explain：** 这一步把后端批输出变成每个请求 state 的 pending output。

来源：python/sglang/srt/managers/tokenizer_manager.py L1970-L2011

**Code：**

```python
state.finished = recv_obj.finished_reasons[i] is not None
if isinstance(recv_obj, BatchStrOutput):
    is_stream = getattr(state.obj, "stream", False)
    incremental = (
        self.server_args.incremental_streaming_output and is_stream
    )
    delta_text = recv_obj.output_strs[i]
    delta_output_ids = list(recv_obj.output_ids[i])
    output_offset = state.last_output_offset
    state.append_text(delta_text)
    state.output_ids.extend(delta_output_ids)

    if is_stream:
        if incremental:
            output_token_ids = delta_output_ids
            _slice_streaming_output_meta_info(
                meta_info,
                output_offset,
                state.customized_info_accumulated.keys(),
            )
            state.last_output_offset = len(state.output_ids)
            out_dict = {
                "text": delta_text,
                "output_ids": output_token_ids,
                "meta_info": meta_info,
            }
        elif state.finished:
            out_dict = {
                "text": state.get_text(),
                "output_ids": state.output_ids.copy(),
                "meta_info": meta_info,
            }
        else:
            out_dict = {
                "text": None,
                "output_ids": state.output_ids,
                "meta_info": meta_info,
            }
```

**代码逻辑：** finish reason 决定 state 是否结束；每个 BatchStrOutput 都更新累计文本和 token ids；incremental 模式切 meta_info 并返回本次 delta；非 incremental 中间 chunk 推迟全量文本构造到结束时。

**为什么这样写：** 流式请求需要低延迟输出；非 incremental 模式如果每步都拼全量字符串，会把长输出变成 O(n²) 开销。

**不变量与失败模式：** `last_output_offset` 必须只在 incremental streaming 下推进；否则 meta_info 切片和 token delta 会错位。

**Comment：** 这里解释了为什么流式中间响应可能出现 `text=None`。

### 5.3 _wait_one_response：前台等待 event

**问题与约束：** 前台 HTTP 协程要等待后台 `handle_loop` 写入 `state.out_list`；客户端可能断连；incremental streaming 可能在前台醒来前积压多个 delta。

**设计选择：** 循环 `await state.event.wait()`，超时后检测 disconnect；醒来后原子 drain `out_list`，incremental 多 chunk 时调用 `_coalesce_streaming_chunks`。

**Explain：** 这是前台协程把 `ReqState` 中的 pending output 转成 HTTP yield 的地方。

来源：python/sglang/srt/managers/tokenizer_manager.py L1455-L1492

**Code：**

```python
while True:
    try:
        await asyncio.wait_for(
            state.event.wait(), timeout=_REQUEST_STATE_WAIT_TIMEOUT
        )
    except asyncio.TimeoutError:
        if (
            request is not None
            and not obj.background
            and await request.is_disconnected()
        ):
            self.abort_request(obj.rid)
            raise ValueError(
                f"Request is disconnected from the client side (type 1). Abort request {obj.rid=}"
            )
        continue

    out_list = state.out_list
    state.out_list = []
    finished = state.finished
    state.event.clear()

    incremental_stream = (
        is_stream and self.server_args.incremental_streaming_output
    )
    if incremental_stream and len(out_list) > 1:
        out = self._coalesce_streaming_chunks(
            out_list,
            obj.rid,
            state.customized_info_accumulated.keys(),
        )
    else:
        out = out_list[-1]
```

**代码逻辑：** 等待 event 超时不是失败，只用于周期性检查客户端断连；断连时主动 abort；正常醒来后清空 out_list 并清 event；多个 incremental chunk 合并成一个输出。

**为什么这样写：** 后台收包和前台发送速度可能不同；coalesce 可以减少 HTTP 写出次数，同时不丢 token ids。

**不变量与失败模式：** `out_list` 不应为空时访问 `out_list[-1]`；disconnect abort 只对非 background 请求生效。

**Comment：** 这段是“客户端断连如何传播到 Scheduler”的关键链路之一。

---

## 6. Mixin 复用数据面和控制面

### 6.1 score_request 复用 generate_request

**问题与约束：** 打分接口需要 generation model 的 logprob 或 embedding model 的 pooled logits；如果另建一条数据面链路，会重复分词、调度、输出处理和多模态逻辑。

**设计选择：** Score mixin 根据模型类型构造 `GenerateReqInput` 或 `EmbeddingReqInput`，generation 打分用 `max_new_tokens=0`，然后直接调用 `generate_request(...).__anext__()`。

**Explain：** score 是 generate/embedding 数据面的一种参数化调用，而不是独立后端。

来源：python/sglang/srt/managers/tokenizer_manager_score_mixin.py L691-L713

**Code：**

```python
if is_generation:
    batch_request = GenerateReqInput(
        text=text_prompts,
        input_ids=input_ids,
        token_ids_logprob=label_token_ids,
        return_logprob=True,
        logprob_start_len=0 if use_multi_item_scoring else -1,
        stream=False,
        sampling_params={"max_new_tokens": 0},
        positional_embed_overrides=positional_embed_overrides,
        multi_item_delimiter_indices=mis_delimiter_indices,
    )
else:
    batch_request = EmbeddingReqInput(
        text=text_prompts,
        input_ids=input_ids,
        positional_embed_overrides=positional_embed_overrides,
        return_pooled_hidden_states=return_pooled_hidden_states,
        multi_item_delimiter_indices=mis_delimiter_indices,
    )

results = await self.generate_request(batch_request, request).__anext__()
```

**代码逻辑：** CausalLM 走 GenerateReqInput 并请求 logprob；非 generation 模型走 EmbeddingReqInput；两者都复用 generate_request 取第一批结果。

**为什么这样写：** Scheduler 和模型 runner 已经支持 logprob/embedding forward；score 只需要构造合适请求并后处理结果。

**不变量与失败模式：** generation 打分必须 `max_new_tokens=0`；multi-item scoring 的 delimiter indices 要和输入序列位置一致。

**Comment：** 打分功能的正确性主要依赖请求构造和结果后处理，不是新的一套调度系统。

### 6.2 flush_cache 复用控制面 communicator

**问题与约束：** 控制面请求不是按 rid 返回的 generate output，但它的回复仍从 Scheduler/TokenizerManager 的结果通道回来；如果后台 loop 没启动，future 可能永远无人消费。

**设计选择：** 控制面方法先 `auto_create_handle_loop()`，然后调用对应 FanOut communicator，例如 `flush_cache_communicator`。

**Explain：** 控制面和数据面共享后台接收循环，但用 communicator 区分回复。

来源：python/sglang/srt/managers/tokenizer_control_mixin.py L256-L262

**Code：**

```python
async def flush_cache(
    self: TokenizerManager, timeout_s: Optional[float] = None
) -> FlushCacheReqOutput:
    self.auto_create_handle_loop()
    return (
        await self.flush_cache_communicator(FlushCacheReqInput(timeout_s=timeout_s))
    )[0]
```

**代码逻辑：** 确保后台 loop 存在；构造 FlushCacheReqInput；等待 communicator 返回的列表并取第一个结果。

**为什么这样写：** 控制命令可能 fan out 到多个 DP rank，但对单结果接口可取第一个或合并；后台 `handle_loop` 负责把响应送到 communicator。

**不变量与失败模式：** `init_communicators` 必须已经注册 `FlushCacheReqOutput` 的 dispatcher；否则控制回复不会唤醒 communicator。

**Comment：** 这也是其他权重更新、LoRA、profile 控制 API 的共同模式。

---

## 7. 多 HTTP Worker 路由

### 7.1 MultiTokenizerRouter 的职责

**问题与约束：** 多个 TokenizerWorker 能分担 HTTP 和分词压力，但 Scheduler/Detokenizer 侧仍需要一个统一前后向路由；pause/continue 状态还必须在所有 worker 间一致。

**设计选择：** `MultiTokenizerRouter` 位于 tokenizer workers 与 Scheduler/Detokenizer 之间：forward 聚合 worker 请求到 Scheduler，backward 把结果路由回 worker，并广播 pause/continue。

**Explain：** 多 worker 模式不是让每个 worker 都直连后端，而是增加一个 router 维护路由和广播语义。

来源：python/sglang/srt/managers/multi_tokenizer_mixin.py L379-L385

**Code：**

```python
class MultiTokenizerRouter:
    """A router between tokenizer managers and the scheduler/detokenizer manager.

    Forward: tokenizer managers → router → scheduler.
    Backward: detokenizer manager → router → tokenizer managers.
    Also broadcasts pause/continue to all tokenizer managers for consistent is_pause state.
    """
```

**代码逻辑：** 类 docstring 定义三件事：worker 到 Scheduler 的 forward path、Detokenizer 到 worker 的 backward path、pause/continue 广播。

**为什么这样写：** 多 worker 引入了额外状态一致性问题；只路由请求不够，还要统一 pause 状态。

**不变量与失败模式：** 所有 worker 必须通过 router 注册自己的 IPC；否则 backward 输出或 pause 广播无法到达。

**Comment：** 单 worker 的 `stamp_http_worker_ipc` 到这里变成多 worker 路由的核心依据。

### 7.2 router_worker_obj：注册、广播与转发

**问题与约束：** worker 会发送普通请求、注册请求、pause/continue 请求；pause/continue 既要通知所有 worker 更新本地状态，也可能要转发给 Scheduler。

**设计选择：** `router_worker_obj` 从 worker PULL socket 收对象；注册请求只更新 `all_worker_ipcs`；pause/continue 广播给所有 worker，并在非 abort pause 模式下转发 Scheduler；其他对象直接转发 Scheduler。

**Explain：** forward path 中最特殊的不是普通 generate，而是 pause/continue 的一致性处理。

来源：python/sglang/srt/managers/multi_tokenizer_mixin.py L449-L480

**Code：**

```python
async def router_worker_obj(self):
    while True:
        recv_obj = await async_sock_recv(self.receive_from_worker)

        if isinstance(recv_obj, TokenizerWorkerRegistrationReq):
            if recv_obj.worker_ipc_name not in self.all_worker_ipcs:
                self.all_worker_ipcs.add(recv_obj.worker_ipc_name)
            continue

        if isinstance(
            recv_obj, (PauseGenerationReqInput, ContinueGenerationReqInput)
        ):
            is_pause = isinstance(recv_obj, PauseGenerationReqInput)
            broadcast = PauseContinueBroadcastReq(is_pause=is_pause)
            for ipc_name in self.all_worker_ipcs:
                self.socket_mapping.send_output(ipc_name, broadcast)
            if not (
                isinstance(recv_obj, PauseGenerationReqInput)
                and recv_obj.mode == "abort"
            ):
                await async_sock_send(self.send_to_scheduler, recv_obj)
            continue

        await async_sock_send(self.send_to_scheduler, recv_obj)
```

**代码逻辑：** worker 注册只登记 IPC；pause/continue 生成广播对象并发到所有 worker；abort pause 不转发 Scheduler，其他 pause/continue 转发；普通请求直接 async send。

**为什么这样写：** pause 是 TokenizerManager 本地 gate 和 Scheduler 后端状态的组合动作；多 worker 下必须先保证所有前台入口都同步暂停/恢复。

**不变量与失败模式：** `all_worker_ipcs` 必须包含所有活跃 worker；abort 模式的特殊跳过转发依赖后续轮询/abort 逻辑完成清理。

**Comment：** 多 worker 的正确性边界主要在 pause/continue 和结果回路，而不是普通请求转发。

---

## 8. 走读小结

TokenizerManager 的核心不是“只做分词”，而是把 HTTP/API 请求变成 Scheduler 能消费的 IPC 对象，并维护前台请求 state 与后台输出事件之间的映射。它的关键不变量是：

- 每个 rid 在发送前有 `ReqState`，结束或异常时能清理。
- 发送前 opaque 字段完成 pickle/shared-memory 包装。
- 后台 `handle_loop` 是唯一收包点，负责唤醒前台 `_wait_one_response`。
- 多 worker 模式下每个请求和控制广播都必须带正确 worker 路由信息。
