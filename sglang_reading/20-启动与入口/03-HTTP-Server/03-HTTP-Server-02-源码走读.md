---
type: batch-doc
module: 03-HTTP-Server
batch: "03"
doc_type: walkthrough
title: "HTTP Server 入口 · 源码走读"
tags:
 - sglang/batch/03
 - sglang/module/http-server
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-05
---
# HTTP Server 入口 · 源码走读

> 走读主线：`launch_server` 先启动 SRT engine 子进程，再把 tokenizer/template/scheduler 信息交给 FastAPI；`lifespan` 初始化 serving handler 和 warmup；HTTP 路由最终把 `/generate`、health 和 OpenAI-compatible API 委托给 `TokenizerManager` 或 serving handler。

---

## 1. launch_server 与 Engine 启动链

### 1.1 launch_server 把 HTTP server 与 SRT engine 串起来

问题与约束：
- SGLang HTTP server 不是单进程服务：主进程运行 HTTP、Engine 和 TokenizerManager，Scheduler/Detokenizer 在子进程中，启动顺序必须先让 engine 就绪再对外监听。

设计选择：
- `launch_server` 先调用 `Engine._launch_subprocesses`，拿到 tokenizer manager、template manager、port args、scheduler init 结果和 watchdog，再进入 `_setup_and_run_http_server`。

Explain：
函数 docstring 明确了三类 engine 组件：TokenizerManager、Scheduler subprocess、DetokenizerManager subprocess。ZMQ IPC 端口由启动阶段分配，HTTP 层只拿到已初始化的全局状态。

来源：python/sglang/srt/entrypoints/http_server.py L2471-L2517

Code：

```python
def launch_server(
    server_args: ServerArgs,
    init_tokenizer_manager_func: Callable = init_tokenizer_manager,
    run_scheduler_process_func: Callable = run_scheduler_process,
    run_detokenizer_process_func: Callable = run_detokenizer_process,
    execute_warmup_func: Callable = _execute_server_warmup,
    launch_callback: Optional[Callable[[], None]] = None,
):
    """
    Launch SRT (SGLang Runtime) Server.

    The SRT server consists of an HTTP server and an SRT engine.
    """
    (
        tokenizer_manager,
        template_manager,
        port_args,
        scheduler_init_result,
        subprocess_watchdog,
    ) = Engine._launch_subprocesses(
        server_args=server_args,
        init_tokenizer_manager_func=init_tokenizer_manager_func,
        run_scheduler_process_func=run_scheduler_process_func,
        run_detokenizer_process_func=run_detokenizer_process_func,
    )

    _setup_and_run_http_server(
        server_args,
        tokenizer_manager,
        template_manager,
        port_args,
        scheduler_init_result.scheduler_infos,
        subprocess_watchdog,
        execute_warmup_func=execute_warmup_func,
        launch_callback=launch_callback,
    )
```

代码逻辑：
- Engine 启动负责子进程、端口和模型加载握手。
- HTTP setup 只接收启动产物，不再负责 fork scheduler。
- warmup 函数和 callback 以参数形式注入，便于测试或定制。

为什么这样写：
- HTTP 层与 engine 子进程生命周期分离后，CLI、Python API 和测试都可以复用 Engine 启动逻辑。
- `_setup_and_run_http_server` 是阻塞运行点，必须在子进程 ready 信息可用后进入。

不变量与失败模式：
- `scheduler_init_result.scheduler_infos` 必须至少有一个 scheduler info。
- 如果 `_launch_subprocesses` 失败，HTTP server 不应开始监听。

Comment：
HTTP Server 入口的第一跳是 `launch_server`，不是 FastAPI route。

### 1.2 _launch_subprocesses 按顺序配置环境、启动 scheduler/detokenizer/tokenizer

问题与约束：
- engine 启动要同时处理插件、端口、scheduler、detokenizer、多节点 rank、multi-tokenizer、模型加载 ready 和子进程崩溃监控。

设计选择：
- `_launch_subprocesses` 作为启动总控：先配置环境与端口，再启动 scheduler；rank 0 继续启动 detokenizer 和 tokenizer，等待 ready 后启动 watchdog。

Explain：
非零 node rank 不运行 tokenizer/detokenizer，只等待 scheduler ready 后进入 dummy health server 或直接返回。rank 0 初始化 tokenizer manager 后等待 scheduler ready，并把 `max_req_input_len` 回填给 tokenizer manager。

来源：python/sglang/srt/entrypoints/engine.py L782-L908

Code：

```python
configure_logger(server_args)
_set_envs_and_config(server_args)
load_plugins()
server_args.check_server_args()
_set_gc(server_args)

if port_args is None:
    port_args = PortArgs.init_new(server_args)

scheduler_init_result, scheduler_procs = cls._launch_scheduler_processes(
    server_args, port_args, run_scheduler_process_func
)

if server_args.node_rank >= 1:
    scheduler_init_result.wait_for_ready()
    launch_dummy_health_check_server(
        server_args.host, server_args.port, server_args.enable_metrics
    )
    scheduler_init_result.wait_for_completion()
    return (None, None, port_args, scheduler_init_result, None)

detoken_procs, detoken_names = cls._launch_detokenizer_subprocesses(...)

if server_args.tokenizer_worker_num == 1:
    tokenizer_manager, template_manager = init_tokenizer_manager_func(
        server_args, port_args
    )
else:
    tokenizer_manager = MultiTokenizerRouter(server_args, port_args)
    template_manager = None

scheduler_init_result.wait_for_ready()
tokenizer_manager.max_req_input_len = scheduler_init_result.scheduler_infos[0][
    "max_req_input_len"
]
subprocess_watchdog = SubprocessWatchdog(processes=processes, process_names=names)
subprocess_watchdog.start()
return (
    tokenizer_manager,
    template_manager,
    port_args,
    scheduler_init_result,
    subprocess_watchdog,
)
```

代码逻辑：
- `load_plugins` 在启动早期保证插件 hook 已注册。
- scheduler 先于 detokenizer/tokenizer 启动。
- multi-node 非零 rank 只保留 scheduler 生命周期。
- tokenizer manager 通过 scheduler info 获取输入长度上限。

为什么这样写：
- Scheduler 是模型加载和推理调度核心，HTTP 就绪必须依赖 scheduler ready。
- 非零 rank 不接收 HTTP 请求，避免重复 tokenizer/detokenizer 资源。
- watchdog 在所有子进程 pid 已知后启动，才能监控完整进程集合。

不变量与失败模式：
- `server_args.check_server_args()` 必须在 fork 前完成，避免子进程继承错误配置。
- scheduler ready 前不能接受生成请求。
- multi-tokenizer 模式下 `template_manager=None`，后续 worker 会在 lifespan 中重建。

Comment：
这段是 SRT server 的生命周期骨架，HTTP 只是它的最后一层入口。

### 1.3 _launch_scheduler_processes 区分 TP/PP 网格和 DP controller

问题与约束：
- 单 DP 模式需要按 PP×TP 网格启动多个 scheduler；多 DP 模式需要一个 controller 管理多个 DP rank。

设计选择：
- `dp_size == 1` 时逐个创建 scheduler process 并绑定 GPU；`dp_size > 1` 时只 fork `run_data_parallel_controller_process`。

Explain：
单 DP 分支用 `_calculate_rank_ranges` 得到当前 node 的 PP/TP rank 范围，为每个 rank 创建 Pipe 和进程，并计算 `gpu_id/attn_cp_rank/moe_dp_rank/moe_ep_rank`。DP 分支只创建一个 controller process，初始化握手仍通过 Pipe。

来源：python/sglang/srt/entrypoints/engine.py L607-L673

Code：

```python
if server_args.dp_size == 1:
    memory_saver_adapter = TorchMemorySaverAdapter.create(
        enable=server_args.enable_memory_saver
    )
    scheduler_pipe_readers = []
    pp_rank_range, tp_rank_range, pp_size_per_node, tp_size_per_node = (
        _calculate_rank_ranges(...)
    )

    for pp_rank in pp_rank_range:
        for tp_rank in tp_rank_range:
            reader, writer = mp.Pipe(duplex=False)
            gpu_id = (
                server_args.base_gpu_id
                + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
            )
            attn_cp_rank, moe_dp_rank, moe_ep_rank = _compute_parallelism_ranks(
                server_args, tp_rank
            )
            with maybe_reindex_device_id(gpu_id) as gpu_id:
                proc = mp.Process(
                    target=run_scheduler_process_func,
                    args=(..., gpu_id, tp_rank, attn_cp_rank, moe_dp_rank, moe_ep_rank, pp_rank, None, writer),
                )
                proc.start()
            scheduler_procs.append(proc)
            scheduler_pipe_readers.append(reader)
else:
    reader, writer = mp.Pipe(duplex=False)
    scheduler_pipe_readers = [reader]
    proc = mp.Process(
        target=run_data_parallel_controller_process,
        kwargs=dict(
            server_args=server_args,
            port_args=port_args,
            pipe_writer=writer,
            run_scheduler_process_func=run_scheduler_process_func,
        ),
    )
    proc.start()
    scheduler_procs.append(proc)
```

代码逻辑：
- 每个 scheduler 都有一个初始化 Pipe reader。
- GPU id 由 base、PP offset、TP offset 和 gpu step 计算。
- DP controller 模式把 scheduler 管理下沉到 controller 子进程。

为什么这样写：
- TP/PP scheduler 直接 fork 更简单；DP 模式需要额外控制层处理多 rank 调度。
- Pipe 只用于 ready 握手，运行期通信交给 ZMQ/IPC。

不变量与失败模式：
- PP/TP rank range 必须与当前 node 的 GPU 资源匹配。
- `gpu_id` 计算错误会导致多个进程争用同一设备。
- DP controller 进程失败会让整个 DP 初始化失败。

Comment：
这一段解释了为什么 SGLang 在不同并行配置下看到的 scheduler 子进程数量不同。

### 1.4 init_tokenizer_manager 构造 TokenizerManager 与模板推断

问题与约束：
- tokenizer manager 要和 server args、port args 绑定；OpenAI chat/tool/reasoning 的解析器又可能由 chat template 自动推断。

设计选择：
- `init_tokenizer_manager` 允许注入 TokenizerManagerClass，初始化 `TemplateManager` 后，对 `reasoning_parser/tool_call_parser == "auto"` 的字段做推断或禁用。

Explain：
函数先创建 tokenizer manager，再让 template manager 根据模型路径、chat template 和 completion template 初始化模板。若模板能给出建议 parser，就写回 `server_args`；否则写成 `None` 并 warning。

来源：python/sglang/srt/entrypoints/engine.py L135-L180

Code：

```python
def init_tokenizer_manager(
    server_args: ServerArgs,
    port_args: PortArgs,
    TokenizerManagerClass: Optional[TokenizerManager] = None,
) -> Tuple[TokenizerManager, TemplateManager]:
    TokenizerManagerClass = TokenizerManagerClass or TokenizerManager
    tokenizer_manager = TokenizerManagerClass(server_args, port_args)

    template_manager = TemplateManager()
    template_manager.initialize_templates(
        tokenizer_manager=tokenizer_manager,
        model_path=server_args.model_path,
        chat_template=server_args.chat_template,
        completion_template=server_args.completion_template,
    )

    for attr, suggested, label in (
        ("reasoning_parser", template_manager.suggested_reasoning_parser, "reasoning parser"),
        ("tool_call_parser", template_manager.suggested_tool_call_parser, "tool-call parser"),
    ):
        if getattr(server_args, attr) != "auto":
            continue
        if suggested is not None:
            setattr(server_args, attr, suggested)
        else:
            setattr(server_args, attr, None)

    return tokenizer_manager, template_manager
```

代码逻辑：
- 默认使用标准 `TokenizerManager`，但调用方可替换类。
- TemplateManager 初始化后保留 suggested parser。
- 只处理值为 `"auto"` 的 parser 字段。

为什么这样写：
- tokenizer manager 是 HTTP 请求进入调度系统的第一站，必须在 HTTP setup 前可用。
- parser 自动推断依赖模板，放在 tokenizer/template 初始化之后最自然。

不变量与失败模式：
- 模板识别失败时，auto parser 会被禁用而不是保留 `"auto"`。
- 注入的 TokenizerManagerClass 必须兼容标准构造参数。

Comment：
这个函数是 chat template、tool call 和 reasoning parser 自动化的连接点。

### 1.5 _wait_for_scheduler_ready 防止子进程死亡后主进程挂死

问题与约束：
- scheduler 模型加载可能因为 OOM 或初始化错误被杀死；主进程如果阻塞在 Pipe `recv()`，会永久卡住。

设计选择：
- 用 `poll(timeout=5.0)` 循环等待 ready 消息，同时检查所有 scheduler 进程是否仍 alive。

Explain：
收到 Pipe 数据后要求 `status == "ready"`，否则抛初始化失败。Pipe EOF 或轮询期间发现进程死亡，都会通过 `_scheduler_died_error` 抛出带进程信息的错误。

来源：python/sglang/srt/entrypoints/engine.py L1368-L1397

Code：

```python
def _wait_for_scheduler_ready(
    scheduler_pipe_readers: List,
    scheduler_procs: List,
) -> List[Dict]:
    scheduler_infos = []
    for i in range(len(scheduler_pipe_readers)):
        while True:
            if scheduler_pipe_readers[i].poll(timeout=5.0):
                try:
                    data = scheduler_pipe_readers[i].recv()
                except EOFError:
                    raise _scheduler_died_error(i, scheduler_procs[i])
                if data["status"] != "ready":
                    raise RuntimeError(
                        "Initialization failed. Please see the error messages above."
                    )
                scheduler_infos.append(data)
                break

            for j in range(len(scheduler_procs)):
                if not scheduler_procs[j].is_alive():
                    raise _scheduler_died_error(j, scheduler_procs[j])

    return scheduler_infos
```

代码逻辑：
- 每个 Pipe reader 必须收到一个 ready 数据。
- poll 超时不是失败，只是触发子进程存活检查。
- scheduler info 按 reader 顺序收集。

为什么这样写：
- OOM killer 不一定能通过正常异常传回主进程，必须主动检查进程状态。
- 5 秒 poll 能兼顾快速失败和模型加载长耗时。

不变量与失败模式：
- scheduler 子进程必须在 ready 时写入 dict 且包含 `status`。
- scheduler_procs 与 pipe_readers 顺序必须对应。

Comment：
这是启动稳定性关键点：防止“模型没起来但 HTTP 主进程一直等”的故障。

### 1.6 Engine.generate 复用 TokenizerManager 的 async generator

问题与约束：
- Python API 需要提供同步调用体验，但底层 tokenizer manager 是 async generator；还要支持 stream 与 non-stream 两种返回方式。

设计选择：
- 构造 `GenerateReqInput` 后调用 `tokenizer_manager.generate_request(obj, None)`，stream 返回同步 iterator，非 stream 消费第一块结果。

Explain：
`Engine.generate` 与 HTTP `/generate` 使用同一类请求对象。同步 API 通过 `self.loop.run_until_complete(generator.__anext__())` 把 async generator 结果转成同步返回。

来源：python/sglang/srt/entrypoints/engine.py L366-L415

Code：

```python
routed_dp_rank = self._resolve_routed_dp_rank(
    routed_dp_rank, data_parallel_rank
)

obj = GenerateReqInput(
    text=prompt,
    input_ids=input_ids,
    sampling_params=sampling_params,
    image_data=image_data,
    audio_data=audio_data,
    video_data=video_data,
    return_logprob=return_logprob,
    stream=stream,
    routed_dp_rank=routed_dp_rank,
    disagg_prefill_dp_rank=disagg_prefill_dp_rank,
    rid=rid,
    session_id=session_id,
    priority=priority,
)
generator = self.tokenizer_manager.generate_request(obj, None)

if stream:
    def generator_wrapper():
        while True:
            try:
                chunk = self.loop.run_until_complete(generator.__anext__())
                yield chunk
            except StopAsyncIteration:
                break

    return generator_wrapper()
else:
    ret = self.loop.run_until_complete(generator.__anext__())
    return ret
```

代码逻辑：
- DP rank routing 先归一化。
- 请求字段覆盖文本、多模态、logprob、session、disaggregation 和优先级。
- stream 包装器逐块消费 async generator。
- non-stream 只取第一块完整结果。

为什么这样写：
- Python API 与 HTTP API 共用 `GenerateReqInput`，减少两套路径语义漂移。
- 同步包装让用户不必直接管理 event loop。

不变量与失败模式：
- non-stream 模式要求第一块就是最终结果。
- stream iterator 必须正确处理 `StopAsyncIteration`。

Comment：
这段说明 HTTP `/generate` 不是唯一入口；Python API 也走同一个 tokenizer manager 核心。

---

## 2. FastAPI 生命周期与服务运行

### 2.1 init_multi_tokenizer 在 worker 进程中从共享内存重建状态

问题与约束：
- 多 tokenizer worker 模式下，uvicorn worker 会重新 import FastAPI app，不能直接共享主进程里的 tokenizer manager 对象。

设计选择：
- `init_multi_tokenizer` 从 shared memory 读取 port/server/scheduler info，为当前 worker 创建新的 tokenizer IPC name 和 `TokenizerWorker`，再设置全局状态。

Explain：
函数还断言 API key 不支持 multi-tokenizer 模式。每个 worker 用临时文件名生成独立 IPC 地址，初始化 templates，并把 scheduler 的 `max_req_input_len` 写入 tokenizer manager。

来源：python/sglang/srt/entrypoints/http_server.py L210-L258

Code：

```python
async def init_multi_tokenizer() -> ServerArgs:
    main_pid = get_main_process_id()
    port_args, server_args, scheduler_info = read_from_shared_memory(
        f"multi_tokenizer_args_{main_pid}"
    )

    assert (
        server_args.api_key is None
    ), "API key is not supported in multi-tokenizer mode"

    port_args.tokenizer_ipc_name = (
        f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
    )
    tokenizer_manager = TokenizerWorker(server_args, port_args)
    template_manager = TemplateManager()
    template_manager.initialize_templates(
        tokenizer_manager=tokenizer_manager,
        model_path=server_args.model_path,
        chat_template=server_args.chat_template,
        completion_template=server_args.completion_template,
    )

    tokenizer_manager.max_req_input_len = scheduler_info["max_req_input_len"]

    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            template_manager=template_manager,
            scheduler_info=scheduler_info,
        )
    )

    return server_args
```

代码逻辑：
- shared memory 名称由主进程 pid 派生。
- worker 本地创建 tokenizer IPC。
- templates 和 global state 在 worker 内重新初始化。

为什么这样写：
- 多进程 HTTP worker 不能复用主进程 Python 对象，只能通过共享内存传递启动参数。
- 每个 tokenizer worker 使用独立 IPC，避免请求在 worker 间争用同一 tokenizer socket。

不变量与失败模式：
- 主进程必须在启动 worker 前写入 shared memory。
- multi-tokenizer 模式不支持 API key 认证。

Comment：
这段是理解 multi-tokenizer 模式为何和 single-tokenizer setup 不同的关键。

### 2.2 lifespan 选择 single/multi tokenizer 并初始化 metrics/tracing

问题与约束：
- FastAPI worker 启动时需要知道自己是主进程单 tokenizer，还是 multi-tokenizer worker；同时 metrics 和 tracing 要在请求处理前完成初始化。

设计选择：
- `lifespan` 根据 app 属性判断 single tokenizer；否则调用 `init_multi_tokenizer`。随后按 server args 初始化 Prometheus middleware、函数计时和 tracing thread label。

Explain：
single 模式的参数由 `_setup_and_run_http_server` 写到 app 属性上；multi 模式通过 shared memory 重建。disaggregation 模式会给 thread label 加上 Prefill/Decode 前缀。

来源：python/sglang/srt/entrypoints/http_server.py L261-L292

Code：

```python
@asynccontextmanager
async def lifespan(fast_api_app: FastAPI):
    if getattr(fast_api_app, "is_single_tokenizer_mode", False):
        server_args = fast_api_app.server_args
        warmup_thread_kwargs = fast_api_app.warmup_thread_kwargs
        thread_label = "Tokenizer"
    else:
        server_args = await init_multi_tokenizer()
        warmup_thread_kwargs = dict(server_args=server_args)
        thread_label = f"MultiTokenizer-{_global_state.tokenizer_manager.worker_id}"

    if server_args.enable_metrics:
        add_prometheus_middleware(app)
        enable_func_timer()

    if server_args.enable_trace:
        process_tracing_init(
            server_args.otlp_traces_endpoint,
            "sglang",
            trace_modules=server_args.trace_modules,
        )
        if server_args.disaggregation_mode == "prefill":
            thread_label = "Prefill" + thread_label
        elif server_args.disaggregation_mode == "decode":
            thread_label = "Decode" + thread_label
        trace_set_thread_info(thread_label)
```

代码逻辑：
- single 模式读取 app 上预置参数。
- multi 模式在 lifespan 内初始化 tokenizer worker。
- metrics 与 tracing 都在 serving handler 创建前完成。

为什么这样写：
- uvicorn worker 生命周期是初始化 per-worker 对象的安全位置。
- tracing thread label 要包含 tokenizer/Prefill/Decode 语义，便于跨进程追踪。

不变量与失败模式：
- single 模式下 app 必须已经设置 `server_args/warmup_thread_kwargs`。
- multi 模式 shared memory 读取失败会导致 worker 启动失败。

Comment：
FastAPI 的 `lifespan` 是 HTTP worker 真正完成运行时绑定的地方。

### 2.3 lifespan 注册 serving handler 与 warmup 线程

问题与约束：
- HTTP route 不应在每个请求里重复创建 OpenAI/Ollama/Anthropic serving handler；warmup 又可能阻塞，需要后台执行。

设计选择：
- lifespan 把 serving handler 放到 `fast_api_app.state`，可选初始化 tool server，再启动 `_wait_and_warmup` 后台线程。

Explain：
Chat completion、completion、embedding、rerank、tokenize、responses 等 handler 都共享 global tokenizer/template manager。custom warmups 可先执行，通用 warmup 在独立线程中启动；FastAPI yield 后开始处理请求，shutdown 时关闭 tool server 并 join warmup 线程。

来源：python/sglang/srt/entrypoints/http_server.py L292-L392

Code：

```python
fast_api_app.state.openai_serving_completion = OpenAIServingCompletion(
    _global_state.tokenizer_manager, _global_state.template_manager
)
fast_api_app.state.openai_serving_chat = (
    _global_state.tokenizer_manager.serving_chat_class(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
)
fast_api_app.state.openai_serving_embedding = OpenAIServingEmbedding(
    _global_state.tokenizer_manager, _global_state.template_manager
)
fast_api_app.state.ollama_serving = OllamaServing(_global_state.tokenizer_manager)
fast_api_app.state.anthropic_serving = AnthropicServing(
    fast_api_app.state.openai_serving_chat
)

if server_args.warmups is not None:
    await execute_warmups(
        server_args.disaggregation_mode,
        server_args.warmups.split(","),
        _global_state.tokenizer_manager,
    )

warmup_thread = threading.Thread(
    target=_wait_and_warmup,
    kwargs=warmup_thread_kwargs,
)
warmup_thread.start()

try:
    yield
finally:
    if tool_server is not None and hasattr(tool_server, "aclose"):
        await tool_server.aclose()
    warmup_thread.join()
```

代码逻辑：
- serving handler 在 app state 上保存，route 只取 handler 调用。
- warmup thread 和 FastAPI serving 并行。
- shutdown 时等待 warmup 线程结束。

为什么这样写：
- Handler 初始化包含模板、tokenizer 和协议状态，放在 lifespan 可避免请求内重复构造。
- warmup 独立线程可以让 HTTP server 启动监听，同时 `/health` 用 status 控制 ready。

不变量与失败模式：
- `_global_state.tokenizer_manager` 必须在 lifespan 前可用或在 multi 模式中被初始化。
- warmup 失败会影响 server status，route 层仍需靠 health/status 判断。

Comment：
OpenAI-compatible 路由之所以很薄，是因为业务对象已经在 lifespan 中挂到 app state。

### 2.4 FastAPI app 注册 CORS、解压中间件和附加 router

问题与约束：
- HTTP server 要提供统一 FastAPI app，同时支持可选 OpenAPI、CORS、请求解压和额外 router。

设计选择：
- 模块级创建 `FastAPI(lifespan=...)`，添加 CORS middleware；环境变量打开时添加 request decompression；再 include `v1_loads_router`。

Explain：
FastAPI app 是模块级对象，multi-worker uvicorn 可以通过字符串导入同一对象。OpenAPI URL 可通过环境变量关闭。

来源：python/sglang/srt/entrypoints/http_server.py L395-L417

Code：

```python
app = FastAPI(
    lifespan=lifespan,
    openapi_url=None if get_bool_env_var("DISABLE_OPENAPI_DOC") else "/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if envs.SGLANG_ENABLE_REQUEST_DECOMPRESSION.get():
    from sglang.srt.entrypoints.http_request_decompression import (
        RequestDecompressionMiddleware,
    )

    app.add_middleware(RequestDecompressionMiddleware)

from sglang.srt.entrypoints.v1_loads import router as v1_loads_router

app.include_router(v1_loads_router)
```

代码逻辑：
- app 创建时绑定 lifespan。
- CORS 默认全放开。
- request decompression 是可选中间件。
- v1 loads router 作为子路由挂入。

为什么这样写：
- 模块级 app 便于 uvicorn/granian 用 import string 加载。
- 可选中间件通过环境变量控制，避免默认增加解压成本。

不变量与失败模式：
- CORS 全开放适合服务端推理 API，但生产环境可能需要额外网关限制。
- request decompression 依赖请求体格式与中间件实现，开启后要确认客户端压缩方式兼容。

Comment：
FastAPI app 本身很薄，核心状态在 lifespan 和 global state 中。

### 2.5 _setup_and_run_http_server 写入 global state 并配置鉴权

问题与约束：
- HTTP route 需要访问 tokenizer manager、template manager 和 scheduler info；single tokenizer 模式还要支持 API key/admin API key middleware。

设计选择：
- `_setup_and_run_http_server` 先 `set_global_state`，把 watchdog 挂到 tokenizer manager；single 模式把 server args 和 warmup kwargs 写到 app 属性，并按配置添加鉴权 middleware。

Explain：
metrics 开启时还会加 response tracking middleware。multi-tokenizer 模式不能直接传对象给 worker，因此把参数写入 shared memory，供 `init_multi_tokenizer` 读取。

来源：python/sglang/srt/entrypoints/http_server.py L2267-L2335

Code：

```python
def _setup_and_run_http_server(
    server_args: ServerArgs,
    tokenizer_manager,
    template_manager,
    port_args: PortArgs,
    scheduler_infos: List[Dict],
    subprocess_watchdog: Optional[SubprocessWatchdog],
    execute_warmup_func: Callable = _execute_server_warmup,
    launch_callback: Optional[Callable[[], None]] = None,
):
    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            template_manager=template_manager,
            scheduler_info=scheduler_infos[0],
        )
    )

    if tokenizer_manager is not None:
        tokenizer_manager._subprocess_watchdog = subprocess_watchdog

    if server_args.enable_metrics:
        add_prometheus_track_response_middleware(app)

    if server_args.tokenizer_worker_num == 1:
        app.is_single_tokenizer_mode = True
        app.server_args = server_args
        app.warmup_thread_kwargs = dict(
            server_args=server_args,
            launch_callback=launch_callback,
            execute_warmup_func=execute_warmup_func,
        )
        if (
            server_args.api_key
            or server_args.admin_api_key
            or app_has_admin_force_endpoints(app)
        ):
            from sglang.srt.utils.auth import add_api_key_middleware

            add_api_key_middleware(
                app,
                api_key=server_args.api_key,
                admin_api_key=server_args.admin_api_key,
            )
    else:
        app.is_single_tokenizer_mode = False
        multi_tokenizer_args_shm = write_data_for_multi_tokenizer(
            port_args, server_args, scheduler_infos[0]
        )
```

代码逻辑：
- global state 是 route/lifespan 访问运行时对象的入口。
- single tokenizer 通过 app 属性传 lifespan 参数。
- multi-tokenizer 通过 shared memory 传 worker 初始化参数。
- admin-force endpoint 存在时，即使没有普通 API key，也会触发鉴权 middleware。

为什么这样写：
- FastAPI route 函数不适合显式传递大量运行时对象，global state 和 app state 分担了两类状态。
- single/multi tokenizer 的对象传递机制不同，必须在这里分支。

不变量与失败模式：
- `scheduler_infos[0]` 必须存在。
- multi-tokenizer 模式下鉴权不支持，不能设置 API key。
- watchdog 只在 tokenizer manager 非空时挂载。

Comment：
这段把 engine 启动结果转换成 HTTP worker 可访问状态。

### 2.6 HTTP server 运行路径区分 Granian、SSL refresh、uvicorn 和多 worker

问题与约束：
- SGLang HTTP server 要支持 HTTP/2、SSL refresh、普通 uvicorn，以及 tokenizer_worker_num > 1 的多 HTTP worker 模式。

设计选择：
- single tokenizer 下按 `enable_http2/enable_ssl_refresh` 选择 Granian、uvicorn Server API 或普通 `uvicorn.run`；multi-tokenizer 下使用 import string 或 Granian worker 模式。

Explain：
普通 single 模式直接把 `app` 对象传给 uvicorn；multi 模式需要让 worker import `"sglang.srt.entrypoints.http_server:app"`，再在 lifespan 中重建 tokenizer worker。

来源：python/sglang/srt/entrypoints/http_server.py L2353-L2445

Code：

```python
if server_args.tokenizer_worker_num == 1:
    if server_args.enable_http2:
        _run_granian_server(
            host=server_args.host,
            port=server_args.port,
            log_level=server_args.log_level_http or server_args.log_level,
            ssl_certfile=server_args.ssl_certfile,
            ssl_keyfile=server_args.ssl_keyfile,
            ssl_ca_certs=server_args.ssl_ca_certs,
            ssl_keyfile_password=server_args.ssl_keyfile_password,
            ssl_verify=False,
        )
    elif server_args.enable_ssl_refresh:
        config = uvicorn.Config(app, host=server_args.host, port=server_args.port, ...)
        config.load()
        server = uvicorn.Server(config)
        asyncio.run(_run_with_ssl_refresh())
    else:
        uvicorn.run(
            app,
            host=server_args.host,
            port=server_args.port,
            root_path=server_args.fastapi_root_path,
            log_level=server_args.log_level_http or server_args.log_level,
            timeout_keep_alive=envs.SGLANG_TIMEOUT_KEEP_ALIVE.get(),
            loop="uvloop",
            ssl_keyfile=server_args.ssl_keyfile,
            ssl_certfile=server_args.ssl_certfile,
        )
else:
    if server_args.enable_http2:
        _run_granian_server(
            host=server_args.host,
            port=server_args.port,
            tokenizer_worker_num=server_args.tokenizer_worker_num,
            ssl_certfile=server_args.ssl_certfile,
            ssl_keyfile=server_args.ssl_keyfile,
        )
```

代码逻辑：
- HTTP/2 走 Granian。
- SSL refresh 走 uvicorn Config/Server API，以便访问 SSLContext。
- 默认走 `uvicorn.run(app, ...)`。
- multi-tokenizer 分支禁用 SSL refresh 并配置多 worker 运行。

为什么这样写：
- 不同 server backend 暴露的能力不同，必须按功能选择运行方式。
- multi-worker import string 能让每个 worker 自己执行 lifespan 初始化。

不变量与失败模式：
- SSL refresh 不支持多 tokenizer worker。
- HTTP/2 路径依赖 Granian 可用。
- 多 worker 模式必须已写入 shared memory，否则 worker import app 后无法初始化 tokenizer。

Comment：
这里是 HTTP server 最终阻塞点；走到这里才真正开始监听端口。

### 2.7 _execute_server_warmup 先等 /model_info 再构造 warmup 请求

问题与约束：
- HTTP server 监听端口后，模型信息和 serving handler 可能还没完全就绪；warmup 需要先确认 `/model_info` 可访问，再选择 generate/chat/encode 请求。

设计选择：
- warmup 循环最多等待 120 秒探测 `/model_info`；成功后根据 `is_generation`、VLM 能力和 tokenizer 初始化状态选择 warmup endpoint。

Explain：
`/model_info` 成功后，代码从返回 JSON 判断是否 generation 模型、是否 VLM。generation 默认 warm `/generate`，embedding warm `/encode`，部分 VLM 使用 `/v1/chat/completions`。

来源：python/sglang/srt/entrypoints/http_server.py L1992-L2029

Code：

```python
success = False
for _ in range(120):
    time.sleep(1)
    try:
        res = requests.get(
            url + "/model_info", timeout=5, headers=headers, verify=ssl_verify
        )
        assert res.status_code == 200, f"{res=}, {res.text=}"
        success = True
        break
    except (AssertionError, requests.exceptions.RequestException):
        last_traceback = get_exception_traceback()
        pass

if not success:
    logger.error(f"Initialization failed. warmup error: {last_traceback}")
    kill_process_tree(os.getpid())
    return success

model_info = res.json()
is_vlm = bool(model_info.get("has_image_understanding", False)) and not is_mps()
if model_info["is_generation"]:
    if is_vlm and not server_args.skip_tokenizer_init:
        request_name = "/v1/chat/completions"
    else:
        request_name = "/generate"
else:
    request_name = "/encode"
max_new_tokens = 8 if model_info["is_generation"] else 1
json_data = {
    "sampling_params": {
        "temperature": 0,
        "max_new_tokens": max_new_tokens,
    },
}
```

代码逻辑：
- `/model_info` 是 warmup 的 readiness 前置检查。
- 探测失败会 kill 当前进程树。
- endpoint 选择依赖模型类型。
- generation warmup 请求默认生成 8 个 token。

为什么这样写：
- 先读 model_info 可以避免对错误 endpoint 发送 warmup。
- warmup 失败直接 kill 进程，避免服务表面监听但内部不可用。

不变量与失败模式：
- `/model_info` 必须在 120 秒内返回 200。
- VLM warmup 和 disaggregation 模式存在特殊限制，源码中另有分支处理。

Comment：
warmup 的目标不是性能预热本身，而是把“可监听”推进到“可服务”。

### 2.8 _wait_and_warmup 控制 server status 进入 Up

问题与约束：
- HTTP server 可以先启动监听，但 health 需要知道模型权重和 warmup 是否已经完成。

设计选择：
- `_wait_and_warmup` 可选先等 checkpoint engine 权重，再执行 warmup；跳过 warmup 时直接把 tokenizer manager 状态设为 Up。

Explain：
warmup 成功后打印 ready 日志。若 `execute_warmup_func` 返回 False，函数直接返回，不会打印 ready，也不会推进为 Up。

来源：python/sglang/srt/entrypoints/http_server.py L2145-L2161

Code：

```python
def _wait_and_warmup(
    server_args: ServerArgs,
    launch_callback: Optional[Callable[[], None]] = None,
    execute_warmup_func: Callable = _execute_server_warmup,
):
    if server_args.checkpoint_engine_wait_weights_before_ready:
        _wait_weights_ready()

    if not server_args.skip_server_warmup:
        if not execute_warmup_func(server_args):
            return
    else:
        _global_state.tokenizer_manager.server_status = ServerStatus.Up

    logger.info("The server is fired up and ready to roll!")
```

代码逻辑：
- checkpoint engine 等权重优先于 warmup。
- `skip_server_warmup` 直接置 Up。
- warmup 失败不继续执行 ready 日志。

为什么这样写：
- 分布式权重加载和 warmup 都可能影响真正可服务时间，health 不应只看端口监听。
- 跳过 warmup 是显式选择，因此直接置 Up。

不变量与失败模式：
- `_global_state.tokenizer_manager` 必须存在。
- warmup 函数需要负责失败时设置合适状态或结束进程。

Comment：
`ServerStatus.Up` 的设置点在 warmup 链路，不在 uvicorn 启动时。

---

## 3. HTTP 路由

### 3.1 /generate 将 native 请求转给 TokenizerManager

问题与约束：
- native `/generate` 既要支持 streaming SSE，也要支持普通 JSON；客户端断开不应被误报为服务端错误。

设计选择：
- FastAPI 直接把 body 解析为 `GenerateReqInput`，stream 模式包装 `tokenizer_manager.generate_request` async iterator，非 stream 模式取第一块结果。

Explain：
stream 模式每个输出以 `data: ...\n\n` 发送，结束时发送 `[DONE]`。StreamingResponse 的 background 注册 abort task，保证客户端取消时能通知 tokenizer manager。

来源：python/sglang/srt/entrypoints/http_server.py L785-L835

Code：

```python
@app.api_route(
    "/generate",
    methods=["POST", "PUT"],
    response_class=SGLangORJSONResponse,
)
async def generate_request(obj: GenerateReqInput, request: Request):
    if envs.SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES.get():
        apply_header_overrides(obj, request.headers)
    if obj.stream:

        async def stream_results() -> AsyncIterator[bytes]:
            try:
                async for out in _global_state.tokenizer_manager.generate_request(
                    obj, request
                ):
                    yield b"data: " + dumps_json(out) + b"\n\n"
            except ValueError as e:
                if request is not None and await request.is_disconnected():
                    logger.info(f"[http_server] Client disconnected: {e}")
                    return
                out = {"error": {"message": str(e), "type": "invalid_request_error", "code": 400, "retryable": False}}
                yield b"data: " + dumps_json(out) + b"\n\n"
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
            background=_global_state.tokenizer_manager.create_abort_task(obj),
        )
    else:
        try:
            ret = await _global_state.tokenizer_manager.generate_request(
                obj, request
            ).__anext__()
            return orjson_response(ret)
        except ValueError as e:
            return _create_error_response(e)
```

代码逻辑：
- header overrides 是可选开关。
- stream 模式保持 SSE 格式。
- 客户端断开时停止生成错误事件。
- non-stream 模式返回第一块结果。

为什么这样写：
- TokenizerManager 是 native API 和 Python API 共同的请求入口，HTTP route 不重复调度逻辑。
- abort task 绑定到 StreamingResponse background，可以在连接结束时清理请求。

不变量与失败模式：
- `GenerateReqInput` 校验失败会在 FastAPI/Pydantic 层返回错误。
- stream 错误要保持 SSE 格式，否则客户端解析会失败。

Comment：
从 `/generate` 开始，后续链路进入 TokenizerManager、Scheduler 和 Detokenizer。

### 3.2 /health 可选择轻量探活或真实生成探活

问题与约束：
- 简单端口存活不等于 scheduler/detokenizer 健康；但每次 health 都生成 token 又会增加负载。

设计选择：
- server 正在退出或 Starting 时返回 503；默认 `/health` 只返回 200。环境变量打开生成探活或访问生成探活路径时，发送 max_new_tokens=1 请求并检查 last_receive_tstamp。

Explain：
健康探测使用唯一 rid，generation 模型构造 `GenerateReqInput`，embedding 模型构造 `EmbeddingReqInput`。只要在 timeout 内收到 detokenizer/scheduler 响应，就取消探测任务、清理 rid，并把 server_status 设为 Up。

来源：python/sglang/srt/entrypoints/http_server.py L588-L652

Code：

```python
if _global_state.tokenizer_manager.gracefully_exit:
    return Response(status_code=503)

if _global_state.tokenizer_manager.server_status == ServerStatus.Starting:
    return Response(status_code=503)

if (
    not envs.SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION.get()
    and request.url.path == "/health"
):
    return Response(status_code=200)

sampling_params = {"max_new_tokens": 1, "temperature": 0.0}
rid = f"{HEALTH_CHECK_RID_PREFIX}_{uuid.uuid4().hex}"

if _global_state.tokenizer_manager.is_generation:
    gri = GenerateReqInput(
        rid=rid,
        input_ids=[0],
        sampling_params=sampling_params,
        log_metrics=False,
    )
else:
    gri = EmbeddingReqInput(
        rid=rid, input_ids=[0], sampling_params=sampling_params, log_metrics=False
    )

async def gen():
    async for _ in _global_state.tokenizer_manager.generate_request(gri, request):
        break

task = asyncio.create_task(gen())
tic = time.time()
while time.time() < tic + HEALTH_CHECK_TIMEOUT:
    await asyncio.sleep(1)
    if _global_state.tokenizer_manager.last_receive_tstamp > tic:
        task.cancel()
        _global_state.tokenizer_manager.rid_to_state.pop(rid, None)
        _global_state.tokenizer_manager.server_status = ServerStatus.Up
        return Response(status_code=200)

task.cancel()
_global_state.tokenizer_manager.rid_to_state.pop(rid, None)
_global_state.tokenizer_manager.server_status = ServerStatus.UnHealthy
return Response(status_code=503)
```

代码逻辑：
- shutdown 和 Starting 都是明确 503。
- 轻量 `/health` 默认不生成。
- 生成探活检查 tokenizer manager 是否收到后端响应。
- 探测结束清理 rid。

为什么这样写：
- 默认 health 要低成本；深度 health 可通过开关启用。
- 用 `last_receive_tstamp` 判断后端响应，比只看 task 是否完成更贴近 scheduler/detokenizer 活性。

不变量与失败模式：
- rid 必须唯一，否则可能污染现有请求状态。
- timeout 内无响应会把 server 标为 UnHealthy。

Comment：
健康检查分两层：端口级和一次真实后端往返。

### 3.3 OpenAI-compatible 路由只委托 serving handler

问题与约束：
- OpenAI API 协议包含 chat/completions、embeddings、responses 等复杂业务，不适合把逻辑都写在 route 函数里。

设计选择：
- route 只做请求类型声明和 JSON 校验依赖，然后调用 `raw_request.app.state.openai_serving_*` handler。

Explain：
`/v1/chat/completions` 的 route 直接把 typed request 和 raw request 交给 `openai_serving_chat.handle_request`。handler 是 lifespan 中初始化的对象，持有 tokenizer manager 和 template manager。

来源：python/sglang/srt/entrypoints/http_server.py L1606-L1613

Code：

```python
@app.post("/v1/chat/completions", dependencies=[Depends(validate_json_request)])
async def openai_v1_chat_completions(
    request: ChatCompletionRequest, raw_request: Request
):
    """OpenAI-compatible chat completion endpoint."""
    return await raw_request.app.state.openai_serving_chat.handle_request(
        request, raw_request
    )
```

代码逻辑：
- FastAPI 先把 body 解析为 `ChatCompletionRequest`。
- `validate_json_request` 做 JSON 请求约束。
- route 不直接调用 tokenizer manager，而是交给 serving chat handler。

为什么这样写：
- OpenAI 协议处理包含模板、tool call、stream 格式和错误语义，独立 handler 更清晰。
- app state 让 route 保持无状态。

不变量与失败模式：
- lifespan 必须已经设置 `openai_serving_chat`。
- handler 需要兼容 raw request 中的 headers、disconnect 和 streaming 行为。

Comment：
OpenAI API 专题应深入 handler；HTTP Server 这里只看委托边界。
