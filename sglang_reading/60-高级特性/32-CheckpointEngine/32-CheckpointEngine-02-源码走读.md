---
type: batch-doc
module: 32-CheckpointEngine
batch: "32"
doc_type: walkthrough
title: "CheckpointEngine · 源码走读"
tags:
  - sglang/batch/32
  - sglang/module/checkpoint-engine
  - sglang/doc/walkthrough
aliases:
  - "02-源码走读"
updated: 2026-07-05
---

# CheckpointEngine · 源码走读

> 走读主线：CheckpointEngine 路径让外部 ParameterServer 通过 ZMQ 暴露权重，再由 SGLang 的 `/update_weights_from_ipc` 控制面触发各 TP rank 拉取并加载。SGLang 自身不实现 checkpoint 分发算法，而是在启动等待、请求契约、Scheduler 同步、ModelRunner 加载和 worker post hook 上补齐 serving 侧的安全边界。

---

## 1. 启动等待：先允许引擎启动，再等待外部灌权重

### 1.1 server_args 暴露 wait-before-ready 开关

问题与约束：
- CheckpointEngine 场景常见启动顺序是先起 SGLang engine，再由外部训练或参数服务灌入权重；普通 serving 则应保持冷启动权重可用后直接服务。

设计选择：
- 用 `checkpoint_engine_wait_weights_before_ready` 显式选择等待模式，默认值为 `False`，避免影响普通模型加载路径。

Explain：
该参数的说明写明：开启后 server 会等待 checkpoint-engine 或其他 update method 完成初始权重加载，然后才服务推理请求。

来源：python/sglang/srt/server_args.py L2505-L2508

Code：

```python
checkpoint_engine_wait_weights_before_ready: A[
    bool,
    "If set, the server will wait for initial weights to be loaded via checkpoint-engine or other update methods before serving inference requests.",
] = False
```

代码逻辑：
- 参数挂在 `ServerArgs` 上，由启动和 TokenizerManager 初始化阶段读取。
- 默认关闭，只有显式启用才进入等待权重模式。
- 说明同时覆盖 checkpoint-engine 和其他 update method。

为什么这样写：
- CheckpointEngine 是热灌权重能力，不应让普通 HTTP serving 额外等待。
- 将等待语义做成启动参数，便于 RL loop 或外部控制器按场景打开。

不变量与失败模式：
- 如果未开启该参数，`initial_weights_loaded` 初始就是 ready 状态。
- 如果开启但外部从未调用 update API，server 会停在 warmup 前的等待阶段直到超时日志出现。

Comment：
这个开关定义了 CheckpointEngine 与普通冷启动之间的第一条分界线。

### 1.2 TokenizerManager 初始化 initial_weights_loaded

问题与约束：
- `_wait_weights_ready` 需要一个跨 HTTP/update 路径共享的状态位；该状态不能在每次请求时临时推断。

设计选择：
- 在 `init_weight_update` 中初始化 `initial_weights_loaded`：普通模式为 `True`，wait-before-ready 模式改成 `False`。

Explain：
TokenizerManager 初始化 weight update 相关状态时，同时设置 `model_update_lock` 等更新控制结构；`initial_weights_loaded` 因此成为 control plane 和启动等待共同使用的状态。

来源：python/sglang/srt/managers/tokenizer_manager.py L459-L468

Code：

```python
def init_weight_update(self):
    self.initial_weights_loaded = True
    if self.server_args.checkpoint_engine_wait_weights_before_ready:
        self.initial_weights_loaded = False

    self.model_update_lock = RWLock()
    self.model_update_result: Optional[Awaitable[UpdateWeightFromDiskReqOutput]] = (
```

代码逻辑：
- 先按普通启动假设权重已加载。
- 如果 server args 要求等待 checkpoint-engine 权重，再把状态置为 `False`。
- 随后初始化模型更新锁和结果对象。

为什么这样写：
- 等待状态必须在 server warmup 之前可见。
- 复用 TokenizerManager 作为 control plane 状态中心，避免在 HTTP server 和 scheduler 间散落状态。

不变量与失败模式：
- `initial_weights_loaded=False` 只表示初始权重未确认，不表示进程未启动。
- 如果后续 update API 成功但没有把该字段置回 `True`，warmup 前等待会误判。

Comment：
CheckpointEngine 的 readiness 不是由 `/ping` 决定，而是由这个权重状态位决定。

### 1.3 _wait_and_warmup 在 warmup 前阻塞等待权重

问题与约束：
- warmup 请求应使用最终服务权重；如果 dummy 权重启动后立刻 warmup，后续外部灌权重可能绕过启动时的验证节奏。

设计选择：
- `_wait_and_warmup` 在执行 warmup 之前调用 `_wait_weights_ready`；等待函数每秒检查 `initial_weights_loaded`，超时只打 error，不直接杀进程。

Explain：
启动顺序是：如果开启等待权重，先阻塞检查；随后执行 warmup 或在 skip warmup 时设置 `ServerStatus.Up`；最后打印 ready 日志并触发可选 callback。

来源：python/sglang/srt/entrypoints/http_server.py L2145-L2191

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

def _wait_weights_ready():
    timeout = WAIT_WEIGHTS_READY_TIMEOUT
    start_time = time.time()

    for _ in range(timeout):
        if _global_state.tokenizer_manager.initial_weights_loaded:
            logger.info(
                f"Weights are ready after {time.time() - start_time:.2f} seconds"
            )
            return
        time.sleep(1)

    logger.error(
        f"Weights are not ready after waiting {timeout} seconds. "
        f"Consider increasing SGLANG_WAIT_WEIGHTS_READY_TIMEOUT environment variable. "
        f"Current status: initial_weights_loaded={_global_state.tokenizer_manager.initial_weights_loaded}"
    )
```

代码逻辑：
- wait-before-ready 只影响 warmup 前的入口。
- `_wait_weights_ready` 以秒为粒度轮询状态位。
- 成功时记录耗时；超时时记录当前状态和 env 调整建议。
- warmup 成功后才进入 ready 日志。

为什么这样写：
- HTTP 进程可以先 listen，外部 ParameterServer 才有机会发起 update。
- warmup 放在权重 ready 之后，能让启动验证覆盖真实权重。

不变量与失败模式：
- 超时日志不是硬失败；如果外部稍后灌权重，进程仍可能继续。
- 如果 `skip_server_warmup=True`，状态会直接置为 `ServerStatus.Up`，但仍受前置 wait 控制。

Comment：
这段代码把 CheckpointEngine 的冷启动体验设计成“服务先可连接，推理 ready 后置”。

---

## 2. HTTP 与 TokenizerManager：把外部 IPC handles 变成内部控制消息

### 2.1 UpdateWeightsFromIPCReqInput 定义请求契约

问题与约束：
- 外部 ParameterServer 需要告诉每个 GPU rank 去哪里接收权重，并允许热更新后清 cache、更新权重版本。

设计选择：
- 单独定义 `UpdateWeightsFromIPCReqInput`，核心字段是 `zmq_handles: Dict[str, str]`，并附带 `flush_cache`、`weight_version`、`torch_empty_cache`。

Explain：
请求对象只用于 CheckpointEngine；键是 GPU UUID，值是对应 ZMQ socket path。SGLang worker 后续会用本地 GPU UUID 从该字典取自己的 handle。

来源：python/sglang/srt/managers/io_struct.py L1600-L1615

Code：

```python
class UpdateWeightsFromIPCReqInput(BaseReq, kw_only=True):
    zmq_handles: Dict[str, str]
    flush_cache: bool = True
    weight_version: Optional[str] = None
    torch_empty_cache: bool = False


class UpdateWeightsFromIPCReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str
```

代码逻辑：
- 请求以 GPU UUID 到 socket path 的映射描述 IPC 入口。
- 默认热更新后 flush cache。
- 可选权重版本会由 TokenizerManager 写入 serving 状态。
- 输出只返回成功位和消息。

为什么这样写：
- IPC 权重更新不同于 tensor payload 更新，HTTP 请求不直接承载 tensor。
- UUID 绑定比本地 device id 更适合跨进程对齐物理 GPU。

不变量与失败模式：
- `zmq_handles` 必须覆盖所有参与 update 的 GPU UUID。
- 如果缺失当前 worker 的 UUID，worker 侧会抛错而不是使用错误 socket。

Comment：
这个 schema 是 CheckpointEngine 路径的最小控制面协议。

### 2.2 HTTP endpoint 转发请求并设置初始权重 ready

问题与约束：
- 外部系统只能通过 HTTP 管理接口触发更新；SGLang 需要把 update 成功和启动等待状态接起来。

设计选择：
- `/update_weights_from_ipc` 调用 TokenizerManager 的同名方法；成功后如果 `initial_weights_loaded` 仍为 `False`，将其置为 `True`。

Explain：
HTTP 层不直接加载权重，只负责鉴权、调用 control plane、组装响应。失败时返回 400，成功时返回默认 200。

来源：python/sglang/srt/entrypoints/http_server.py L1306-L1322

Code：

```python
@app.post("/update_weights_from_ipc")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def update_weights_from_ipc(
    obj: Annotated[UpdateWeightsFromIPCReqInput, Body()], request: Request
):
    success, message = await _global_state.tokenizer_manager.update_weights_from_ipc(
        obj, request
    )

    content = {"success": success, "message": message}
    if success:
        if _global_state.tokenizer_manager.initial_weights_loaded is False:
            _global_state.tokenizer_manager.initial_weights_loaded = True
        return ORJSONResponse(content)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)
```

代码逻辑：
- FastAPI endpoint 接收 IPC update 请求体。
- 请求交给 TokenizerManager 异步处理。
- 成功后释放启动等待状态。
- 失败时带失败消息返回 400。

为什么这样写：
- HTTP 层要尽量薄，避免在 web worker 中承担分布式加载逻辑。
- `initial_weights_loaded` 只在成功后翻转，能保证 warmup 前等待看到真实成功信号。

不变量与失败模式：
- 如果 TokenizerManager 返回失败，启动等待状态不会被误置为 ready。
- 该接口是 admin optional，部署侧仍需按安全策略控制访问面。

Comment：
这个 endpoint 是外部 ParameterServer 进入 SGLang serving 运行时的门。

### 2.3 TokenizerManager 串行化 IPC update

问题与约束：
- 权重更新不能和正常请求读模型状态互相穿插；同时 pause 状态下已经具备独占语义，不能额外死锁。

设计选择：
- TokenizerManager 先创建 handle loop，校验 DP 条件；若当前已 pause，直接 fan-out；否则进入 `model_update_lock.writer_lock` 后 fan-out。

Explain：
成功后如果请求带 `weight_version`，TokenizerManager 会更新服务端权重版本并把版本信息追加到返回消息。

来源：python/sglang/srt/managers/tokenizer_control_mixin.py L486-L519

Code：

```python
async def update_weights_from_ipc(
    self: TokenizerManager,
    obj: UpdateWeightsFromIPCReqInput,
    request: Optional[fastapi.Request] = None,
) -> Tuple[bool, str]:
    self.auto_create_handle_loop()
    try:
        assert (
            self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for update weights from IPC"
        logger.info("Starting IPC weight update")

        async with self.is_pause_cond:
            is_paused = self.is_pause
            if is_paused:
                result = (await self.update_weights_from_ipc_communicator(obj))[0]
                success, message = result.success, result.message

        if not is_paused:
            async with self.model_update_lock.writer_lock:
                result = (await self.update_weights_from_ipc_communicator(obj))[0]
                success, message = result.success, result.message
    except Exception as e:
        error_msg = f"IPC weight update failed: {str(e)}"
        logger.error(error_msg)
        success, message = False, error_msg

    if success and obj.weight_version is not None:
        self._update_weight_version_if_provided(obj.weight_version)
        message += f" Weight version updated to {obj.weight_version}."

    return success, message
```

代码逻辑：
- 创建控制面 handle loop。
- 校验 DP 支持范围。
- pause 状态下直接通过 communicator 下发。
- 非 pause 状态下用写锁保护模型更新。
- 异常被转成 `(False, message)`。
- 成功且带版本时更新 weight version。

为什么这样写：
- 权重替换是写操作，必须排斥并发读写。
- pause 状态代表请求调度已停住，可以避免重复加锁路径扩大等待窗口。

不变量与失败模式：
- 当前只支持 `dp_size == 1` 或开启 DP attention 的场景。
- communicator 只取第一个结果，意味着调用方预期 fan-out 结果已在下游同步。

Comment：
TokenizerManager 的职责是并发控制和结果归一，不参与具体 tensor 加载。

---

## 3. Scheduler 执行：暂停批次、更新 worker、清 cache、跨 rank 对齐

### 3.1 Scheduler 把 IPC 请求路由给 WeightUpdater

问题与约束：
- Scheduler 主循环收到不同控制消息时，需要稳定路由到对应组件；IPC update 不能落到 tensor 或 disk update 路径。

设计选择：
- 在 scheduler 的请求类型映射中，将 `UpdateWeightsFromIPCReqInput` 绑定到 `self.weight_updater.update_weights_from_ipc`。

Explain：
这一层只做类型到 handler 的分发，真正的 pause、worker 调用和 barrier 在 WeightUpdater 中执行。

来源：python/sglang/srt/managers/scheduler.py L1390-L1397

Code：

```python
(
    UpdateWeightsFromTensorReqInput,
    self.weight_updater.update_weights_from_tensor,
),
(
    UpdateWeightsFromIPCReqInput,
    self.weight_updater.update_weights_from_ipc,
),
```

代码逻辑：
- tensor update 和 IPC update 是两个不同请求类型。
- Scheduler 收到 IPC 类型后进入 IPC 专用 handler。
- handler 所属组件是 `weight_updater`。

为什么这样写：
- 请求类型路由集中放在 scheduler，便于控制面消息进入统一队列。
- IPC update 需要和其他 weight update 共用 pause/cache/barrier 逻辑。

不变量与失败模式：
- 如果请求类型没有注册，Scheduler 不会调用 IPC 更新路径。
- 如果误用 tensor update schema，`zmq_handles` 不会被 worker 看到。

Comment：
CheckpointEngine 在 Scheduler 侧只是新增一种 weight update 消息。

### 3.2 WeightUpdater 在观察窗口中执行 IPC 更新

问题与约束：
- 权重更新过程中 running batch 需要暂停；更新成功后 radix cache 与旧权重绑定，不能继续复用。

设计选择：
- `update_weights_from_ipc` 在 `_observe_weight_load("ipc")` 中调用 TP worker；如有 draft worker，TP 成功后再更新 draft；TP 成功后 flush cache，最后做 TP CPU group barrier。

Explain：
`tp_success` 被单独保存，说明只要主 TP worker 更新成功，就会触发 cache flush；draft worker 失败会影响最终 success，但不阻止主权重更新后的 cache 清理。

来源：python/sglang/srt/managers/scheduler_components/weight_updater.py L166-L178

Code：

```python
def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
    with self._observe_weight_load("ipc"):
        success, message = self.tp_worker.update_weights_from_ipc(recv_req)
        tp_success = success
        if success and self.draft_worker is not None:
            success, message = self.draft_worker.update_weights_from_ipc(recv_req)
        if tp_success:
            self.flush_cache_after_weight_update(recv_req)
        if not success:
            logger.error(message)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return UpdateWeightsFromIPCReqOutput(success=success, message=message)
```

代码逻辑：
- 进入 weight load 观察上下文。
- 先更新主 TP worker。
- 主 TP 成功且存在 draft worker 时更新 draft。
- 主 TP 成功后按请求参数 flush cache。
- 失败记录错误日志。
- TP CPU group barrier 后返回结构化输出。

为什么这样写：
- cache 的有效性取决于主模型权重，而不是 draft worker 是否也成功。
- barrier 确保各 rank 对 update 完成状态有一致边界。

不变量与失败模式：
- draft worker 失败会让最终 success 为 false。
- 如果 TP worker 失败，则不会 flush cache，旧权重和旧 cache 仍保持一致。

Comment：
这段是 CheckpointEngine 接入 serving 调度语义的核心。

### 3.3 TPWorker 只把 IPC 更新下沉给 ModelRunner

问题与约束：
- TPWorker 管理 rank 级执行入口，但模型加载细节属于 ModelRunner。

设计选择：
- `TPWorker.update_weights_from_ipc` 直接调用 `self.model_runner.update_weights_from_ipc(recv_req)`，不在这一层解析 ZMQ handles。

Explain：
这一层保持薄转发，让 tensor update、IPC update 都在 ModelRunner 附近接触模型对象。

来源：python/sglang/srt/managers/tp_worker.py L176-L179

Code：

```python
def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
    success, message = self.model_runner.update_weights_from_ipc(recv_req)
    return success, message
```

代码逻辑：
- 接收 Scheduler 下发的 IPC 请求对象。
- 调用 ModelRunner 同名方法。
- 原样返回成功位和消息。

为什么这样写：
- TPWorker 不应知道 checkpoint-engine worker extension 的实现细节。
- 让 ModelRunner 统一拥有模型 loader 和当前 CUDA device 上下文。

不变量与失败模式：
- 如果 ModelRunner 抛错或返回失败，TPWorker 不做恢复。
- TPWorker 层没有额外同步，跨 rank 同步由 WeightUpdater 完成。

Comment：
TPWorker 是 IPC 更新链路中的透明转发层。

### 3.4 ModelRunner 创建 CheckpointEngine worker extension

问题与约束：
- checkpoint-engine 包是可选依赖；普通安装不应因为未安装该包而影响其他功能。

设计选择：
- 在 `update_weights_from_ipc` 方法内部延迟 import `SGLangCheckpointEngineWorkerExtensionImpl`，创建 worker 后调用 `worker.update_weights_from_ipc(recv_req.zmq_handles)`。

Explain：
ImportError 被转成失败消息；其他异常写 error 日志并返回异常字符串。成功时返回固定成功消息。

来源：python/sglang/srt/model_executor/model_runner.py L3246-L3261

Code：

```python
def update_weights_from_ipc(self, recv_req):
    try:
        from sglang.srt.checkpoint_engine.checkpoint_engine_worker import (
            SGLangCheckpointEngineWorkerExtensionImpl,
        )

        worker = SGLangCheckpointEngineWorkerExtensionImpl(self)
        worker.update_weights_from_ipc(recv_req.zmq_handles)
        return True, "IPC weight update completed successfully"
    except ImportError as e:
        return False, f"IPC weight update failed: ImportError {e}"
    except Exception as e:
        logger.error(f"IPC weight update failed: {e}")
        return False, str(e)
```

代码逻辑：
- 延迟导入 checkpoint-engine integration。
- 把当前 ModelRunner 包成 worker extension。
- 将请求中的 `zmq_handles` 交给 extension。
- 区分 ImportError 和普通运行时异常。

为什么这样写：
- 可选依赖只在真正使用 IPC update 时才成为硬要求。
- extension 需要访问 `model_runner.model.load_weights` 和当前 GPU 上下文，因此由 ModelRunner 创建。

不变量与失败模式：
- 未安装 checkpoint-engine 时，HTTP 会收到失败消息而不是进程启动失败。
- worker 更新过程中的任意异常都会返回 false，并由上层转成 400。

Comment：
ModelRunner 是 SGLang 内部模型对象和外部 checkpoint-engine worker API 的适配点。

---

## 4. Worker extension：按 GPU UUID 接 ZMQ，再调用模型 loader 与 post hook

### 4.1 checkpoint-engine 依赖只在 worker 模块导入时检查

问题与约束：
- IPC 更新真正依赖第三方 `checkpoint_engine.worker.update_weights_from_ipc`；没有该包时不能继续。

设计选择：
- 在 worker 模块顶部尝试导入第三方函数；失败时抛出带安装提示的 ImportError。

Explain：
由于 ModelRunner 是延迟 import 这个模块，普通推理路径不会碰到该依赖；但一旦执行 IPC update，缺包会被明确报告。

来源：python/sglang/srt/checkpoint_engine/checkpoint_engine_worker.py L25-L31

Code：

```python
try:
    from checkpoint_engine.worker import update_weights_from_ipc
except ImportError:
    raise ImportError(
        "checkpoint-engine is not installed. "
        "Please install it with: pip install sglang[checkpoint-engine]"
    )
```

代码逻辑：
- 导入第三方 worker 函数。
- ImportError 时重新抛出带安装命令的异常。
- 上层 ModelRunner 捕获该 ImportError 并转成失败消息。

为什么这样写：
- 可选集成需要清楚告诉部署者缺少哪个 extra。
- worker 模块不能在缺少核心依赖时提供半可用行为。

不变量与失败模式：
- 只要导入该模块失败，后续 extension 类都不会创建。
- 安装提示绑定到 `sglang[checkpoint-engine]`，不是普通 `sglang`。

Comment：
这一段把可选依赖失败显式化，减少线上排障成本。

### 4.2 Worker 基类用 GPU UUID 选择本 rank 的 ZMQ handle

问题与约束：
- 外部请求携带多个 GPU 的 socket path；每个 SGLang worker 只能连接属于当前 GPU 的 handle。

设计选择：
- 基类维护 ZMQ context，调用 `get_device_uuid()` 和 `get_device_id()`，用 UUID 查 `zmq_handles`，再调用第三方 `update_weights_from_ipc`。

Explain：
如果当前 GPU UUID 不在请求字典中，代码直接抛 `ValueError`，并打印请求中可用的 key，避免误连其他 rank 的 socket。

来源：python/sglang/srt/checkpoint_engine/checkpoint_engine_worker.py L69-L89

Code：

```python
def update_weights_from_ipc(self, zmq_handles: Dict[str, str]):
    if self._zmq_ctx is None:
        self._zmq_ctx = zmq.Context()
    device_uuid = self.get_device_uuid()
    device_id = self.get_device_id()
    if device_uuid not in zmq_handles:
        raise ValueError(
            f"Device UUID {device_uuid} not found in zmq_handles: {list(zmq_handles.keys())}"
        )
    update_weights_from_ipc(
        self._zmq_ctx,
        zmq_handles[device_uuid],
        device_id=device_id,
        run=self.get_model_loader(),
        post_hook=self.get_post_hook(),
    )
```

代码逻辑：
- 首次调用时创建 ZMQ context。
- 读取当前 GPU UUID 和 device id。
- 用 UUID 从请求 handles 中取 socket path。
- 调用第三方函数，并传入模型 loader 与 post hook。

为什么这样写：
- UUID 比 rank id 更贴近 checkpoint-engine 暴露的物理设备映射。
- `run` 和 `post_hook` 作为回调注入，让第三方库只负责接收和驱动加载。

不变量与失败模式：
- `zmq_handles` 必须以 `GPU-<uuid>` 形式覆盖当前设备。
- 如果当前 GPU 属性不可读或 UUID 不匹配，更新会显式失败。

Comment：
基类中的这个方法是真正跨过 SGLang 和 checkpoint-engine 边界的调用点。

### 4.3 SGLang 实现把 worker extension 绑定到 ModelRunner

问题与约束：
- 基类只定义接口；SGLang 需要提供当前 CUDA device、GPU UUID 和模型权重 loader。

设计选择：
- `SGLangCheckpointEngineWorkerExtensionImpl` 保存 `model_runner`，用 `torch.cuda.current_device()` 获取 device id，用 CUDA device properties 取 UUID，并返回 `model_runner.model.load_weights`。

Explain：
UUID 获取失败时抛 `ValueError`，这样上层能把错误一路返回到 HTTP 调用方。

来源：python/sglang/srt/checkpoint_engine/checkpoint_engine_worker.py L98-L117

Code：

```python
def __init__(self, model_runner):
    super().__init__()
    self.model_runner = model_runner

def get_device_uuid(self) -> str:
    device_id = torch.cuda.current_device()
    try:
        return f"GPU-{torch.cuda.get_device_properties(device_id).uuid!s}"
    except AssertionError as e:
        raise ValueError(f"Failed to get GPU UUID for device {device_id}") from e

def get_device_id(self) -> int:
    return torch.cuda.current_device()

def get_model_loader(self) -> Callable:
    return self.model_runner.model.load_weights
```

代码逻辑：
- 构造时保存当前 ModelRunner。
- device id 来自当前 CUDA 上下文。
- UUID 转成 `GPU-...` 字符串。
- loader 直接返回模型的 `load_weights` 方法。

为什么这样写：
- checkpoint-engine 接收的是设备相关 IPC handle，必须在对应 CUDA device 上执行。
- 复用模型自身 `load_weights`，保证热更新和普通加载使用同一套参数装载语义。

不变量与失败模式：
- 当前 CUDA device 必须已经正确设置。
- 模型对象必须暴露 `load_weights`，否则第三方 worker 无法执行回调。

Comment：
实现类很薄，但它决定了 IPC handle 与实际模型 loader 的绑定关系。

### 4.4 post_hook 对齐冷启动后的量化与模型后处理

问题与约束：
- IPC 方式绕过普通 loader 的部分外围流程；量化模块可能需要在权重加载后执行 repack 或 device 相关处理。

设计选择：
- `get_post_hook` 返回闭包：遍历模型模块，若存在 `quant_method`，在 `device_loading_context` 中调用 `process_weights_after_loading`；最后调用模型级 `post_load_weights`。

Explain：
hook 内部捕获所有异常并记录 warning，不让 post hook 失败直接抛出到第三方 worker 调用栈。

来源：python/sglang/srt/checkpoint_engine/checkpoint_engine_worker.py L122-L141

Code：

```python
def post_hook():
    try:
        from sglang.srt.model_loader.loader import device_loading_context

        for _, module in self.model_runner.model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                target_device = torch.device(
                    "cuda", torch.cuda.current_device()
                )
                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)
        if hasattr(self.model_runner.model, "post_load_weights"):
            self.model_runner.model.post_load_weights()
    except Exception as e:
        logger.warning(f"Post-hook processing failed: {e}")
```

代码逻辑：
- 延迟导入 `device_loading_context`。
- 遍历模型所有子模块。
- 对有 `quant_method` 的模块执行加载后处理。
- 如果模型实现了 `post_load_weights`，继续调用模型级 hook。
- 捕获异常并记录 warning。

为什么这样写：
- 量化权重加载后常需要额外转换，热更新不能只覆盖 tensor 值。
- 使用同类 loader context 可以对齐冷启动时的 device 处理。

不变量与失败模式：
- post hook 失败不会让 `update_weights_from_ipc` 自动失败，可能留下性能或量化格式风险。
- 量化模型接入 CheckpointEngine 时应重点验证该 warning 是否出现。

Comment：
CheckpointEngine 热更新能否等价于冷启动加载，很大程度取决于这个 post hook 是否成功。

---

## 5. 外部 update.py：准备 handles，并回调 SGLang HTTP endpoint

### 5.1 req_inference 只让每个 inference group 的 src rank 发 HTTP 请求

问题与约束：
- ParameterServer 更新时会产生一组 socket paths；如果每个 rank 都 POST，SGLang 会收到重复控制请求。

设计选择：
- `req_inference` 根据 `RANK` 和 `inference_parallel_size` 计算 group src，只允许 src rank 发 `/update_weights_from_ipc`；请求体截取当前 inference group 的 socket paths。

Explain：
请求体把 `socket_paths[src:src+inference_parallel_size]` 转成 dict，附带 `flush_cache=True` 和可选 `weight_version`。

来源：python/sglang/srt/checkpoint_engine/update.py L108-L134

Code：

```python
def req_inference(
    endpoint: str,
    inference_parallel_size: int,
    timeout: float = 300.0,
    uds: str | None = None,
    weight_version: str | None = None,
) -> Callable[[list[tuple[str, str]]], None]:
    rank = int(os.getenv("RANK", 0))
    src = rank // inference_parallel_size * inference_parallel_size

    def req_func(socket_paths: list[tuple[str, str]]):
        if rank == src:
            with httpx.Client(transport=httpx.HTTPTransport(uds=uds)) as client:
                resp = client.post(
                    f"{endpoint}/update_weights_from_ipc",
                    json={
                        "zmq_handles": dict(
                            socket_paths[src : src + inference_parallel_size]
                        ),
                        "flush_cache": True,
                        "weight_version": weight_version,
                    },
                    timeout=timeout,
                )
                resp.raise_for_status()

    return req_func
```

代码逻辑：
- 读取环境变量 `RANK`。
- 按 inference parallel size 计算 group src。
- 只有 src rank 发送 HTTP POST。
- POST 内容包含本 group 的 ZMQ handles。
- HTTP 非成功状态会抛异常。

为什么这样写：
- 每个 inference parallel group 只需要一次控制请求。
- socket path 切片保证请求只覆盖当前 SGLang group，而不是全部训练 ranks。

不变量与失败模式：
- `socket_paths` 的顺序必须和 inference group rank 布局一致。
- HTTP 400 或连接失败会让 `resp.raise_for_status()` 抛错，外部 update 流程应感知失败。

Comment：
`req_inference` 是外部 ParameterServer 完成后回调 SGLang 的桥。

### 5.2 check_sglang_ready 轮询 /ping 后再发起更新

问题与约束：
- 外部脚本可能比 SGLang HTTP server 更早启动；直接 POST update 可能连接失败。

设计选择：
- 每个 inference group 的 src rank 轮询 `GET /ping`，支持 UDS transport；其他 rank 直接 return。

Explain：
轮询间隔是 0.1 秒，每 10 次失败打一条 warning，直到请求成功并 `raise_for_status` 不抛异常。

来源：python/sglang/srt/checkpoint_engine/update.py L49-L71

Code：

```python
def check_sglang_ready(
    endpoint: str, inference_parallel_size: int, uds: str | None = None
):
    rank = int(os.getenv("RANK", 0))
    if rank != rank // inference_parallel_size * inference_parallel_size:
        return
    retry_num = 0
    transport = None
    if uds is not None:
        transport = httpx.HTTPTransport(uds=uds)
    with httpx.Client(transport=transport) as client:
        while True:
            try:
                response = client.get(f"{endpoint}/ping", timeout=10)
                response.raise_for_status()
                break
            except (httpx.ConnectError, httpx.HTTPStatusError) as e:
                if retry_num % 10 == 0:
                    logger.warning(
                        f"fail to check sglang ready, retry {retry_num} times, error: {e}"
                    )
                retry_num += 1
                time.sleep(0.1)
```

代码逻辑：
- 非 src rank 不执行 HTTP 探活。
- 可选使用 Unix domain socket transport。
- src rank 循环请求 `/ping`。
- 连接错误和非成功状态都会进入 retry。

为什么这样写：
- CheckpointEngine 更新依赖 SGLang HTTP 入口已经 listen。
- 只让 src rank 探活可以减少无意义请求和日志噪声。

不变量与失败模式：
- `/ping` 成功只表示 HTTP 可达，不表示 `initial_weights_loaded=True`。
- 如果 endpoint 配错，循环不会自动退出。

Comment：
外部脚本的 ready 检查是“可连接”，SGLang 内部的 ready 检查才是“权重已加载”。

### 5.3 update_weights 串联注册 checkpoint、建组、收集 metadata 与触发更新

问题与约束：
- 外部 ParameterServer 需要先注册 checkpoint 和张量，再初始化通信组、聚合 metadata，最后才能暴露 handles 给 SGLang。

设计选择：
- `update_weights` 先 `register_checkpoint` 和 `init_process_group`，等待 SGLang HTTP ready 后 barrier；随后 gather metas，可选保存 metas；最后按 `update_method` 调用 `ps.update`。

Explain：
`broadcast` 模式不指定 ranks；`p2p` 模式传入 `range(inference_parallel_size)`，且在进入 p2p 前 sleep 2 秒等待销毁旧 process group。

来源：python/sglang/srt/checkpoint_engine/update.py L137-L172

Code：

```python
def update_weights(
    ps,
    checkpoint_name: str,
    checkpoint_files: list[str],
    named_tensors: dict[str, torch.Tensor],
    req_func: Callable[[list[tuple[str, str]]], None],
    inference_parallel_size: int,
    endpoint: str,
    save_metas_file: str | None = None,
    update_method: Literal["broadcast", "p2p", "all"] = "broadcast",
    uds: str | None = None,
):
    ps.register_checkpoint(
        checkpoint_name, files=checkpoint_files, named_tensors=named_tensors
    )
    ps.init_process_group()
    check_sglang_ready(endpoint, inference_parallel_size, uds)
    dist.barrier()
    with timer("Gather metas"):
        ps.gather_metas(checkpoint_name)
    if save_metas_file and int(os.getenv("RANK")) == 0:
        with open(save_metas_file, "wb") as f:
            pickle.dump(ps.get_metas(), f)

    if update_method == "broadcast" or update_method == "all":
        with timer("Update weights without setting ranks"):
            ps.update(checkpoint_name, req_func)

    if update_method == "p2p" or update_method == "all":
        if update_method:
            time.sleep(2)
        with timer("Update weights with setting ranks"):
            ps.update(
                checkpoint_name, req_func, ranks=list(range(inference_parallel_size))
            )
```

代码逻辑：
- 注册 checkpoint 文件和 named tensors。
- 初始化外部 ParameterServer 的 process group。
- 等待 SGLang HTTP 可达。
- 分布式 barrier 后收集 checkpoint metadata。
- rank 0 可选 dump metas。
- 根据 update method 触发 broadcast、p2p 或两者。

为什么这样写：
- metadata 必须在 ParameterServer update 前聚齐，SGLang worker 才能按 handles 正确接收。
- HTTP ready 检查放在 gather 前，避免外部 update 完成后回调一个还不可达的 server。

不变量与失败模式：
- `update_method` 只覆盖 `broadcast`、`p2p`、`all` 三类分支。
- `p2p` 分支依赖 ranks 和 inference parallel layout 对齐。

Comment：
SGLang 侧看到的是一个 HTTP update 请求；外部脚本侧则要完成 checkpoint 注册、元数据聚合和传输策略选择。

### 5.4 FlattenedTensorBucket 支持预扁平化数据重建

问题与约束：
- 权重同步和 IPC 传输通常更适合扁平 buffer；但模型 loader 最终需要按原始 name、shape、dtype 还原 tensor。

设计选择：
- `FlattenedTensorBucket` 既可从 `named_tensors` 构造扁平 buffer，也可从 `flattened_tensor + metadata` 反序列化。

Explain：
构造路径会把每个 tensor flatten 后按 `torch.uint8` view，记录 name、shape、dtype、start/end、numel；反序列化路径要求同时提供扁平 tensor 和 metadata。

来源：python/sglang/srt/weight_sync/tensor_bucket.py L28-L80

Code：

```python
class FlattenedTensorBucket:
    supports_multi_dtypes = True

    def __init__(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]] = None,
        flattened_tensor: torch.Tensor = None,
        metadata: List[FlattenedTensorMetadata] = None,
    ):
        if named_tensors is not None:
            self.metadata: List[FlattenedTensorMetadata] = [None] * len(named_tensors)
            self.flattened_tensor: torch.Tensor = None

            if not named_tensors:
                raise ValueError("Cannot create empty tensor bucket")

            current_idx = 0
            flattened_tensors: List[torch.Tensor] = [None] * len(named_tensors)

            for i, (name, tensor) in enumerate(named_tensors):
                flattened = tensor.flatten().view(torch.uint8)
                flattened_tensors[i] = flattened
                numel = flattened.numel()
                metadata_obj = FlattenedTensorMetadata(
                    name=name,
                    shape=tensor.shape,
                    dtype=tensor.dtype,
                    start_idx=current_idx,
                    end_idx=current_idx + numel,
                    numel=numel,
                )
                self.metadata[i] = metadata_obj
                current_idx += numel

            self.flattened_tensor = torch.cat(flattened_tensors, dim=0)
        else:
            if flattened_tensor is None or metadata is None:
                raise ValueError(
                    "Must provide either named_tensors or both flattened_tensor and metadata"
                )
            self.flattened_tensor = flattened_tensor
            self.metadata = metadata
```

代码逻辑：
- 从 named tensors 构造时逐个 flatten 成 byte view。
- 为每个 tensor 记录还原所需 metadata。
- 将所有 flattened tensor 拼接成一个大 buffer。
- 从预扁平数据构造时要求 metadata 同时存在。

为什么这样写：
- 传输层可以用连续 buffer 提高处理效率。
- metadata 保留了跨 dtype、跨 shape 的还原能力。

不变量与失败模式：
- `named_tensors` 不能为空。
- 预扁平路径缺少 tensor 或 metadata 时直接失败。
- metadata 的 start/end 必须和扁平 buffer 对齐，否则重建会错位。

Comment：
虽然 CheckpointEngine 主要通过第三方 worker 接收权重，但这个 bucket 展示了 SGLang 在权重同步路径中对扁平 buffer 与结构化 tensor 的通用契约。
