---
type: batch-doc
module: 10-Detokenizer
batch: "10"
doc_type: walkthrough
title: "Detokenizer · 源码走读"
tags:
  - sglang/batch/10
  - sglang/module/detokenizer
  - sglang/doc/walkthrough
aliases:
  - "02-源码走读"
updated: 2026-07-05
---

# Detokenizer · 源码走读

> 走读主线：Scheduler 输出 token ids 后，Detokenizer 进程负责按请求维护增量解码状态，将 token span 批量 decode 成字符串增量，再组装 `BatchStrOutput` 发回 TokenizerManager。核心不只是 `tokenizer.decode`，而是状态边界、stop 裁剪、未完成 UTF-8 字符处理和控制消息串行化。

---

## 1. 进程与事件循环

### 1.1 `run_detokenizer_process` 固定 Detokenizer 的进程生命周期

问题与约束：
- Detokenizer 独立进程要跟随父进程生命周期退出，异常时不能留下旧 socket mapping。
- 单 tokenizer worker 和多 tokenizer worker 的回传路径不同，入口需要在创建 manager 后选择事件循环。

设计选择：
- 入口函数负责进程守护、日志初始化和异常兜底；业务逻辑交给 `DetokenizerManager`，并通过 `tokenizer_worker_num` 选择单 worker loop 或 multi HTTP worker loop。

Explain：
`run_detokenizer_process` 是 Detokenizer 子进程的 OS 边界。它设置进程名、配置日志、保留父进程句柄，创建 manager 后进入对应事件循环；一旦 manager 初始化或循环抛异常，就记录 traceback、清理 socket mapping，并向父进程发送 `SIGQUIT`。

来源：python/sglang/srt/managers/detokenizer_manager.py L483-L505

Code：

```python
def run_detokenizer_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    detokenizer_manager_class=DetokenizerManager,
):
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang::detokenizer")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    manager = None
    try:
        manager = detokenizer_manager_class(server_args, port_args)
        if server_args.tokenizer_worker_num == 1:
            manager.event_loop()
        else:
            manager.multi_http_worker_event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DetokenizerManager hit an exception: {traceback}")
        if manager is not None:
            manager.maybe_clear_socket_mapping()
        parent_process.send_signal(signal.SIGQUIT)
```

代码逻辑：
- 调用 `kill_itself_when_parent_died()` 绑定父进程死亡语义。
- 设置进程名和日志。
- 构造 manager。
- `tokenizer_worker_num == 1` 时进入普通 `event_loop`。
- 多 worker 时进入 mixin 提供的 `multi_http_worker_event_loop`。
- 异常时清理 socket mapping，并通知父进程退出。

为什么这样写：
- Detokenizer 是 serving pipeline 的中间进程，异常后继续运行父进程会造成上下游 IPC 卡死。
- `detokenizer_manager_class` 可注入，便于测试替换 manager 实现。

不变量与失败模式：
- `maybe_clear_socket_mapping()` 只有 manager 已创建时才可调用。
- 父进程句柄必须可用；否则异常传播路径本身也可能失败。
- 多 worker 模式依赖 `MultiHttpWorkerDetokenizerMixin` 的事件循环实现。

Comment：
这段把 Detokenizer 的故障处理放在进程入口，而不是分散到每个 handler。

### 1.2 初始化按 IPC、tokenizer、运行态、dispatcher 四步展开

问题与约束：
- Detokenizer 同时依赖 ZMQ socket、HF tokenizer、增量解码状态和消息类型分发。
- 单 worker 模式需要把字符串结果 PUSH 回 TokenizerManager；多 worker 模式结果由 socket mapping 直接推回对应 worker。

设计选择：
- 构造函数只串联四个初始化函数；IPC 初始化根据 `tokenizer_worker_num` 决定是否创建单一 `send_to_tokenizer` socket。

Explain：
`DetokenizerManager.__init__` 明确把 manager 状态分成四层：上下游 IPC、tokenizer、运行态和 dispatcher。运行态包含有界 `decode_status`、batch decode 开关、gpt-oss tool call parser 标志、soft watchdog，以及可选 CPU monitor。

来源：python/sglang/srt/managers/detokenizer_manager.py L94-L159

Code：

```python
def __init__(
    self,
    server_args: ServerArgs,
    port_args: PortArgs,
):
    self.init_ipc_channels(port_args, server_args)
    self.init_tokenizer(server_args)
    self.init_running_status(server_args)
    self.init_request_dispatcher()

def init_ipc_channels(self, port_args: PortArgs, server_args: ServerArgs):
    context = zmq.Context(2)
    self.recv_from_scheduler = get_zmq_socket(
        context, zmq.PULL, port_args.detokenizer_ipc_name, True
    )
    if server_args.tokenizer_worker_num == 1:
        self.send_to_tokenizer = get_zmq_socket(
            context, zmq.PUSH, port_args.tokenizer_ipc_name, False
        )

def init_request_dispatcher(self):
    self._request_dispatcher = TypeBasedDispatcher(
        [
            (BatchEmbeddingOutput, self.handle_batch_embedding_out),
            (BatchTokenIDOutput, self.handle_batch_token_id_out),
            (FreezeGCReq, self.handle_freeze_gc_req),
            (ConfigureLoggingReq, self.handle_configure_logging_req),
        ]
    )
```

代码逻辑：
- `recv_from_scheduler` 是 Scheduler 到 Detokenizer 的 PULL socket。
- 单 worker 模式创建 `send_to_tokenizer` PUSH socket。
- `skip_tokenizer_init` 时 tokenizer 为空，否则按 server args 加载 tokenizer。
- `decode_status` 用有限容量字典保存请求状态。
- dispatcher 将四类输入消息映射到对应 handler。

为什么这样写：
- 初始化阶段清晰分层，方便测试和多 worker 模式复用。
- 单 worker 和多 worker 的回传通道差异只出现在 IPC 初始化和事件循环，不污染解码逻辑。

不变量与失败模式：
- 普通 `event_loop` 需要 `send_to_tokenizer` 已存在；因此只适用于单 worker 模式。
- `BatchEmbeddingOutput` 走透传 handler，不需要 tokenizer。
- `FreezeGCReq`、`ConfigureLoggingReq` 属于控制请求，handler 可能返回 None。

Comment：
这四步初始化基本定义了 Detokenizer 的职责边界：收 Scheduler 消息，按类型处理，再输出字符串结果或执行控制副作用。

### 1.3 `event_loop` 在阻塞 recv 周围关闭 soft watchdog

问题与约束：
- Detokenizer 主循环长期阻塞在 ZMQ recv 上是正常状态，不应该被 watchdog 判定为卡死。
- 一旦收到 Scheduler 消息，handler 可能返回输出，也可能只执行控制副作用。

设计选择：
- 在 `sock_recv` 周围使用 `self.soft_watchdog.disable()`，处理完消息后再 feed watchdog；只有 handler 返回非 None 时才发送给 TokenizerManager。

Explain：
主循环非常短：阻塞收消息、按类型分发、可选发送输出、喂 watchdog。它把“等待上游”排除在 stuck 检测之外，只让实际处理阶段暴露卡住风险。

来源：python/sglang/srt/managers/detokenizer_manager.py L161-L169

Code：

```python
def event_loop(self):
    """The event loop that handles requests"""
    while True:
        with self.soft_watchdog.disable():
            recv_obj = sock_recv(self.recv_from_scheduler)
        output = self._request_dispatcher(recv_obj)
        if output is not None:
            sock_send(self.send_to_tokenizer, output)
        self.soft_watchdog.feed()
```

代码逻辑：
- 循环等待 Scheduler 消息。
- recv 期间临时禁用 soft watchdog。
- 调用 `TypeBasedDispatcher` 执行 handler。
- handler 有输出时通过 `send_to_tokenizer` 发送。
- 本轮处理完成后 feed watchdog。

为什么这样写：
- IPC recv 是正常空闲等待，不代表 Detokenizer stuck。
- 控制类消息不产生下游输出，返回 None 能自然阻止发送。

不变量与失败模式：
- 该循环依赖单 worker 模式下已创建 `send_to_tokenizer`。
- handler 抛异常会逃到进程入口的异常兜底。
- 如果 handler 长时间阻塞，watchdog 不会被 disable 包住，可以检测处理阶段卡住。

Comment：
Detokenizer 的事件循环本身很薄，复杂度集中在 `BatchTokenIDOutput` 的增量解码 handler。

## 2. 增量状态与容量保护

### 2.1 `DecodeStatus` 用文本长度和 token offset 描述一个流式请求

问题与约束：
- 流式解码不能每步都从头发送完整文本，只能发送新增字符串。
- tokenizer decode 可能依赖前文 token 边界，不能简单地只 decode 最新 token。

设计选择：
- 每个请求保存 `decoded_text`、待 decode 的 `decode_ids`、`surr_offset`、`read_offset`、`sent_offset` 和延迟拼接的文本 chunks。

Explain：
`DecodeStatus` 保存一个请求的增量解码状态。`surr_offset/read_offset` 定义 token 窗口，`decoded_text_len` 记录已提交字符串长度，`sent_offset` 记录已经发给 TokenizerManager 的字符串偏移；chunks 延迟合并，避免每步都对长字符串做拼接。

来源：python/sglang/srt/managers/detokenizer_manager.py L63-L88

Code：

```python
@dataclasses.dataclass
class DecodeStatus:
    """Store the status of incremental decoding."""

    decoded_text: str
    decode_ids: List[int]
    surr_offset: int
    read_offset: int
    sent_offset: int = 0
    decoded_text_len: int = dataclasses.field(init=False)
    decoded_text_chunks: List[str] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        self.decoded_text_len = len(self.decoded_text)

    def append_decoded_text(self, text: str):
        if text:
            self.decoded_text_chunks.append(text)
            self.decoded_text_len += len(text)

    def get_decoded_text(self) -> str:
        if self.decoded_text_chunks:
            self.decoded_text += "".join(self.decoded_text_chunks)
            self.decoded_text_chunks.clear()
        return self.decoded_text
```

代码逻辑：
- 初始化后用当前 `decoded_text` 长度填充 `decoded_text_len`。
- 新增干净文本先 append 到 chunk list，并只更新长度。
- 结束时或需要完整文本时再一次性合并 chunks。
- `sent_offset` 默认 0，由流式分支维护。

为什么这样写：
- `decoded_text_len` 让流式路径不用频繁 materialize 完整字符串。
- token offset 和 string offset 分开，可以处理 tokenizer 边界和 UTF-8 字符边界不同步的情况。

不变量与失败模式：
- `decoded_text_len` 必须等于 `decoded_text + decoded_text_chunks` 的总长度。
- 流式路径要求 `sent_offset >= decoded_text_len`，差值表示已发送但未提交的可打印前缀。
- 如果请求状态被驱逐，后续 batch 会无法继续增量解码。

Comment：
理解 `DecodeStatus` 后，后面的 `surr_ids/read_ids/new_text` 差分逻辑才好读。

### 2.2 `DETOKENIZER_MAX_STATES` 给状态表设定上限

问题与约束：
- 每个活跃流式请求都会占用一条 decode 状态；高并发或异常请求堆积可能让状态表无限增长。
- 容量需要可由部署侧调整，而不是写死在代码里。

设计选择：
- 从环境变量 `SGLANG_DETOKENIZER_MAX_STATES` 读取容量，默认 `1 << 16`。

Explain：
这个常量是 Detokenizer 状态表的全局容量来源。注释说明超过容量时最旧请求状态会被驱逐，并建议使用 2 的幂次值来改善内存分配行为。

来源：python/sglang/srt/managers/detokenizer_manager.py L56-L60

Code：

```python
DETOKENIZER_MAX_STATES = int(os.environ.get("SGLANG_DETOKENIZER_MAX_STATES", 1 << 16))
```

代码逻辑：
- 读取环境变量。
- 没有设置时使用 65536。
- `init_running_status` 用它作为 `LimitedCapacityDict` 容量。

为什么这样写：
- 状态容量和部署并发直接相关，需要允许线上按负载调大。
- 默认值足够大，避免普通流式请求被频繁驱逐。

不变量与失败模式：
- 环境变量必须能解析为整数。
- 容量过小会导致请求状态被驱逐，后续增量解码抛出状态缺失错误。
- 容量过大则增加 Detokenizer 进程内存上限。

Comment：
这个值是流式高并发场景下排查状态缺失错误的第一配置项。

### 2.3 `LimitedCapacityDict` 以 FIFO 驱逐旧请求状态

问题与约束：
- 状态表超过容量时必须释放旧状态，否则 Detokenizer 内存不可控。
- 驱逐策略要简单、确定，并且与 Python dict 现有顺序语义兼容。

设计选择：
- 继承 `OrderedDict`，在 `__setitem__` 时容量已满就 `popitem(last=False)` 删除最早插入的状态。

Explain：
`LimitedCapacityDict` 不实现复杂 LRU 更新，只在插入新状态时删除最旧项。对 Detokenizer 来说，状态只需要防止无限增长；一旦被驱逐，后续同 rid 的增量会在 decode 阶段报错并提示调大容量。

来源：python/sglang/srt/managers/detokenizer_manager.py L470-L480

Code：

```python
class LimitedCapacityDict(OrderedDict):
    def __init__(self, capacity: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capacity = capacity

    def __setitem__(self, key, value):
        if len(self) >= self.capacity:
            self.popitem(last=False)
        super().__setitem__(key, value)
```

代码逻辑：
- 保存容量。
- 每次插入前检查当前长度。
- 长度达到容量时删除最早的 item。
- 再插入新状态。

为什么这样写：
- FIFO 驱逐实现简单，适合把状态表作为内存保护阀。
- 不在读取时移动 key，避免请求访问模式改变驱逐顺序。

不变量与失败模式：
- `capacity` 必须大于 0；为 0 时插入前 `popitem` 会在空 dict 上失败。
- 被驱逐请求如果仍有后续 token 输出，会在 `_decode_batch_token_id_output` 中触发 `RuntimeError`。
- 这不是严格 LRU；长时间活跃但较早插入的请求也可能在容量压力下被驱逐。

Comment：
这里选择的是可预测的容量保护，而不是尽量保活每个请求。

## 3. 解码核心

### 3.1 `trim_matched_stop` 在 token 与字符串两种层面处理 stop

问题与约束：
- `finished_reason.matched` 可能是 stop string，也可能是 stop token id。
- 部分请求设置 `no_stop_trim` 后需要保留 matched 内容；gpt-oss tool call token 又有特殊语义。

设计选择：
- 根据 matched 类型分支处理：字符串 stop 在字符串里查找位置；token stop 默认去掉最后一个 token，但 gpt-oss 的 `<|call|>` token 保留。

Explain：
这个函数会在 token decode 前和 finished 字符串输出前被调用。对 list[int] 输入，它假定最后一个 token 是 matched stop token；对字符串输入，它找到 matched stop string 的位置并按 `no_stop_trim` 决定是否保留 matched。

来源：python/sglang/srt/managers/detokenizer_manager.py L171-L201

Code：

```python
def trim_matched_stop(
    self, output: Union[str, List[int]], finished_reason: Dict, no_stop_trim: bool
):
    if not finished_reason:
        return output

    matched = finished_reason.get("matched", None)
    if not matched:
        return output

    if isinstance(matched, str) and isinstance(output, str):
        pos = output.find(matched)
        if pos == -1:
            return output
        end = pos + len(matched)
        return output[:end] if no_stop_trim else output[:pos]

    if isinstance(matched, int) and isinstance(output, list):
        if no_stop_trim:
            return output
        if output[-1] == 200012 and self.is_tool_call_parser_gpt_oss:
            return output
        assert len(output) > 0
        return output[:-1]
    return output
```

代码逻辑：
- 没有 finished reason 或没有 matched 时原样返回。
- 字符串 matched 只处理字符串输出。
- `no_stop_trim=True` 时字符串输出保留 matched 到结尾。
- token matched 只处理 list[int] 输出。
- gpt-oss tool call token `200012` 在对应 parser 下不裁掉。
- 默认 token stop 去掉最后一个 token。

为什么这样写：
- stop token 应在 decode 前裁掉，避免 tokenizer 把 EOS 展示成文本。
- stop string 只能在 decoded text 上可靠查找。
- gpt-oss 的 tool call token 同时是 EOS 之一，但仍可能是工具调用语义的一部分。

不变量与失败模式：
- token stop 分支假设最后一个 token 就是 matched stop token。
- 字符串 stop 只处理第一个 matched 位置，源码 TODO 标注了多 stop string 命中场景未完整处理。
- `output[-1]` 在空 list 上会出错；源码在访问后才断言长度，因此调用方应保证 token list 非空。

Comment：
stop 裁剪同时影响最终文本和中间 decode ids，是增量解码正确性的边界条件。

### 3.2 `_grouped_batch_decode` 按 tokenizer 能力和 decode flags 合并工作

问题与约束：
- 流式 batch 中每个请求的 `skip_special_tokens` 和 `spaces_between_special_tokens` 可能不同。
- 空 token span 应该返回空字符串，但没必要让 tokenizer 为每一行空列表付出 decode 开销。

设计选择：
- 先过滤空 ids 并保留原索引；慢 tokenizer 逐行 decode，fast tokenizer 在 flags 相同时整批 decode，不同时按 `(skip, space)` 分组 decode 后 scatter 回原位置。

Explain：
该函数把 batch decode 的性能收益限制在语义安全范围内。只有 fast tokenizer 且 flags 相同或按 flags 分组后，才调用 `batch_decode`；慢 tokenizer 走 `decode_without_hf_kwargs`，避免传入 HF kwargs 造成兼容问题。

来源：python/sglang/srt/managers/detokenizer_manager.py L207-L269

Code：

```python
n = len(ids_list)
if n == 0:
    return []

keep_idx: Optional[List[int]] = None
if not all(ids_list):
    keep_idx = [i for i, ids in enumerate(ids_list) if ids]
    if not keep_idx:
        return [""] * n
    ids_list = [ids_list[i] for i in keep_idx]
    skip_list = [skip_list[i] for i in keep_idx]
    space_list = [space_list[i] for i in keep_idx]

if not getattr(self.tokenizer, "is_fast", False):
    decoded = [
        decode_without_hf_kwargs(self.tokenizer, ids, skip)
        for ids, skip in zip(ids_list, skip_list)
    ]
else:
    first_skip, first_space = skip_list[0], space_list[0]
    if all(
        s == first_skip and sp == first_space
        for s, sp in zip(skip_list, space_list)
    ):
        decoded = self.tokenizer.batch_decode(
            ids_list,
            skip_special_tokens=first_skip,
            spaces_between_special_tokens=first_space,
        )
```

代码逻辑：
- 空 batch 返回空列表。
- 过滤空 ids，全部为空时返回同长度空字符串列表。
- 慢 tokenizer 逐行 decode。
- fast tokenizer 且 flags 全相同，直接整批 `batch_decode`。
- flags 不同则按 `(skip, space)` 分组，各组 batch decode。
- 如果过滤过空 ids，最后按 `keep_idx` scatter 回原 batch 位置。

为什么这样写：
- `batch_decode` 能降低高并发流式场景中的 per-row decode 开销。
- 不同 flags 不能混在同一次 decode 中，否则特殊 token 或空格处理会错。
- 空 span 不进入 tokenizer，减少无效调用。

不变量与失败模式：
- `ids_list`、`skip_list`、`space_list` 必须等长。
- tokenizer 为 None 时无法 decode；embedding 或 skip-tokenizer 路径不应进入 token id decode。
- 慢 tokenizer 分支不使用 `spaces_between_special_tokens`，这是兼容优先的选择。

Comment：
这是 Detokenizer 中少数纯性能优化，但它严格围绕 decode flags 做分组，避免改变输出语义。

### 3.3 `_decode_batch_token_id_output` 先构造 `read_ids` 与 `surr_ids`

问题与约束：
- 新 token 可能和上一轮末尾 token 一起决定最终字符串，例如 BPE、空格或 Unicode 边界。
- 每个请求的历史状态可能已存在，也可能是本轮第一次出现。

设计选择：
- 对每个 rid 初始化或更新 `DecodeStatus`；`read_ids` 取从 `surr_offset` 到当前全部 token 的可读窗口，`surr_ids` 取从 `surr_offset` 到旧 `read_offset` 的上轮上下文窗口。

Explain：
`read_texts - surr_texts` 是增量字符串的核心。Detokenizer 不直接 decode 新增 token，而是 decode 一个包含上下文的 read window，再减去已经可读的 surrounding window，从而处理 tokenizer 边界依赖。

来源：python/sglang/srt/managers/detokenizer_manager.py L271-L333

Code：

```python
bs = len(recv_obj.rids)

read_ids, surr_ids = [], []
for i in range(bs):
    rid = recv_obj.rids[i]
    if rid not in self.decode_status:
        s = DecodeStatus(
            decoded_text=recv_obj.decoded_texts[i],
            decode_ids=list(recv_obj.decode_ids[i]),
            surr_offset=0,
            read_offset=recv_obj.read_offsets[i],
        )
        self.decode_status[rid] = s
    else:
        s = self.decode_status[rid]
        s.decode_ids.extend(recv_obj.decode_ids[i])

    read_ids.append(
        self.trim_matched_stop(
            s.decode_ids[s.surr_offset :],
            recv_obj.finished_reasons[i],
            recv_obj.no_stop_trim[i],
        )
    )
    surr_ids.append(s.decode_ids[s.surr_offset : s.read_offset])
```

代码逻辑：
- 计算 batch size。
- 新 rid 创建 `DecodeStatus`，初始 `read_offset` 来自 Scheduler。
- 已存在 rid 将本轮新增 decode ids 追加到历史列表。
- `read_ids` 以 `surr_offset` 为起点，包含当前可尝试 decode 的全部 token，并在结束时裁 stop。
- `surr_ids` 以 `surr_offset` 到旧 `read_offset` 作为差分基线。
- 后续根据 `disable_tokenizer_batch_decode` 选择 grouped batch decode 或逐行 decode。

为什么这样写：
- 多 token 一起 decode 能处理 tokenizer 合并规则，直接 decode 本轮新增 token 可能少空格或产生错误字符。
- `surr_ids` 作为基线文本，让增量输出只发送新出现的后缀。

不变量与失败模式：
- `recv_obj.rids`、`decode_ids`、`decoded_texts`、`read_offsets` 等 per-request 列表必须同长度。
- `read_offset` 必须是相对 `decode_ids` 的有效边界。
- 若 stop token 裁剪后 `read_ids` 为空，batch decode 应返回空字符串。

Comment：
这一步体现 Detokenizer 的本质：它维护的是“可安全提交的字符串边界”，不是只把新增 token 映射成字符。

### 3.4 流式分支用 `sent_offset` 处理未完成 UTF-8 字符

问题与约束：
- tokenizer decode 可能在当前 token 边界产生以 replacement char `�` 结尾的文本，说明字符还没完整。
- 但 `find_printable_text` 可能已经能提取一段可打印前缀；这段前缀已发送但不能提交 token offset，下一轮还会重新出现在 decode 结果中。

设计选择：
- 对未 finished 请求，干净 `new_text` 才提交到 `decoded_text` 并推进 token offset；不完整文本只发送可打印前缀，更新 `sent_offset`，但不推进 `surr_offset/read_offset`。

Explain：
`pending = sent_offset - decoded_text_len` 表示此前已经发出但尚未 commit 的前缀长度。下一轮输出时用 `new_text[pending:]` 或 `printable[pending:]` 跳过这段文本，避免重复发送；只有 `new_text` 不以 `�` 结尾时，才将文本提交到 `DecodeStatus` 并推进 token 边界。

来源：python/sglang/srt/managers/detokenizer_manager.py L334-L386

Code：

```python
output_strs = []
for i in range(bs):
    rid = recv_obj.rids[i]
    try:
        s = self.decode_status[rid]
    except KeyError:
        raise RuntimeError(
            f"Decode status not found for request {rid}. "
            "It may be due to the request being evicted from the decode status due to memory pressure. "
        )
    new_text = read_texts[i][len(surr_texts[i]) :]
    if recv_obj.finished_reasons[i] is None:
        pending = s.sent_offset - s.decoded_text_len
        if new_text and not new_text.endswith("�"):
            s.append_decoded_text(new_text)
            s.surr_offset = s.read_offset
            s.read_offset = len(s.decode_ids)
            s.sent_offset = s.decoded_text_len
            output_strs.append(new_text[pending:] if pending else new_text)
        else:
            printable = find_printable_text(new_text)
            s.sent_offset = s.decoded_text_len + len(printable)
            output_strs.append(printable[pending:] if pending else printable)
        continue
```

代码逻辑：
- 逐请求取回 `DecodeStatus`；缺失时抛容量驱逐相关错误。
- `new_text` 是 read 文本减去 surrounding 文本后的后缀。
- 未 finished 时计算 pending 前缀长度。
- 干净文本提交到 chunks，并推进 `surr_offset/read_offset/sent_offset`。
- 不完整文本只取可打印前缀发送，保留 token offset 等下一轮重试。
- finished 分支删除状态，拼出完整文本后做 stop 裁剪，再从 `sent_offset` 切增量。

为什么这样写：
- UTF-8 字符完整性和 token 边界不总是一致，贸然推进 token offset 会丢失后续补全字符。
- `sent_offset` 允许“先发可打印前缀、后续再 commit”，同时避免重复发送。
- finished 时 materialize 完整文本，保证最后一次 stop string 裁剪基于全量字符串。

不变量与失败模式：
- 流式时 `sent_offset >= decoded_text_len`，否则 pending 为负会切错字符串。
- `surr_texts[i]` 必须是 `read_texts[i]` 的前缀，否则 `len(surr_texts[i])` 差分不可靠。
- 状态表驱逐会导致 KeyError 转 RuntimeError，提示调大 `SGLANG_DETOKENIZER_MAX_STATES`。

Comment：
这是 Detokenizer 最关键的边界处理：它允许输出尽可能及时，但只在字符完整时推进 token 提交边界。

### 3.5 `handle_batch_token_id_out` 组装字符串输出并迁移 tensor 编码开销

问题与约束：
- TokenizerManager 热路径主要负责 HTTP/请求侧逻辑，不宜再承担每请求 tensor 到 base64 的编码工作。
- idle batch 没有 rid，但仍需要让 handler 返回结构一致的输出对象。

设计选择：
- handler 先调用 `_decode_batch_token_id_output` 得到 `output_strs`，再把 `routed_experts` 和 `indexer_topk` 逐请求转成 base64 字符串，最后透传 Scheduler 输出字段构造 `BatchStrOutput`。

Explain：
`handle_batch_token_id_out` 是 token ids 到字符串输出的边界。除了 `output_strs` 和两个编码字段，它基本保留 Scheduler 给出的统计、logprob、hidden states、spec decode、缓存和时间统计字段。

来源：python/sglang/srt/managers/detokenizer_manager.py L387-L455

Code：

```python
@staticmethod
def _b64_encode_per_request(
    data_list: Optional[List[Optional[torch.Tensor]]],
) -> Optional[List[Optional[str]]]:
    if data_list is None:
        return None
    return [
        (
            pybase64.b64encode(item.numpy().tobytes()).decode("utf-8")
            if item is not None
            else None
        )
        for item in data_list
    ]

def handle_batch_token_id_out(self, recv_obj: BatchTokenIDOutput):
    output_strs = (
        self._decode_batch_token_id_output(recv_obj)
        if len(recv_obj.rids) > 0
        else []
    )
    routed_experts = self._b64_encode_per_request(recv_obj.routed_experts)
    indexer_topk = self._b64_encode_per_request(recv_obj.indexer_topk)
    return BatchStrOutput(
        rids=recv_obj.rids,
        http_worker_ipcs=recv_obj.http_worker_ipcs,
        finished_reasons=recv_obj.finished_reasons,
        output_strs=output_strs,
        output_ids=recv_obj.output_ids,
        routed_experts=routed_experts,
        indexer_topk=indexer_topk,
        time_stats=recv_obj.time_stats,
    )
```

代码逻辑：
- `None` 输入的 per-request tensor list 原样返回 None。
- 每个非 None tensor 转 numpy bytes，再 base64 编码成 UTF-8 字符串。
- 非空 rid batch 执行增量解码，空 batch 输出空字符串列表。
- 构造 `BatchStrOutput`，透传绝大多数 Scheduler 字段。

为什么这样写：
- Detokenizer 已经在 token ids 到字符串的链路上，顺手处理 debug/专家 tensor 编码可以减轻下游热路径。
- `BatchStrOutput` 保持字段完整，让 TokenizerManager 不需要回看 Scheduler 输出对象。

不变量与失败模式：
- `_b64_encode_per_request` 假设 tensor 在 CPU 或可直接 `.numpy()`；CUDA tensor 需先转 CPU，否则会失败。
- per-request list 中的 None 会保留为 None。
- idle batch 的 `output_strs=[]`，调用方不能假设输出长度总等于某个固定 batch size。

Comment：
这个 handler 是 Detokenizer 的出口：解码结果和所有请求元数据在这里汇合成下游消费的字符串批次。

## 4. 控制面 Fan-out 对照

### 4.1 `FanOutCommunicator` 限定同一时刻只有一个 in-flight 控制请求

问题与约束：
- TokenizerManager 对多个 Scheduler/DP rank 发送控制请求时，需要收齐所有响应。
- 多个控制请求并发发送会让响应归属混乱，除非显式定义队列或共享策略。

设计选择：
- `FanOutCommunicator` 持有 `_result_event`、`_result_values` 和 `_ready_queue`，并限制 mode 只能是 `queueing` 或 `watching`。

Explain：
这个类不是 Detokenizer 主数据链路的一部分，但它和 Detokenizer 一样体现了 manager 间 IPC 的状态管理方式。它用一个 event 收集当前 in-flight 请求的多个响应，用 mode 决定并发调用者是排队还是共享当前请求结果。

来源：python/sglang/srt/managers/communicator.py L11-L36

Code：

```python
class FanOutCommunicator(Generic[T]):
    """Fan-out request + collect response primitive over zmq.

    One send is fanned out to `fan_out` recipients; the caller awaits until
    all `fan_out` responses are collected. Supports two modes:
    - "queueing": requests are serialized; concurrent callers wait in a FIFO queue.
    - "watching": concurrent callers share a single in-flight request and all
      receive the same result when it completes.

    Only one request is in-flight at any time in either mode.
    """

    def __init__(
        self,
        send: Callable[[T], None],
        fan_out: int,
        mode: str = "queueing",
    ):
        self._send = send
        self._fan_out = fan_out
        self._mode = mode
        self._result_event: Optional[asyncio.Event] = None
        self._result_values: Optional[List[T]] = None
        self._ready_queue: Deque[asyncio.Event] = deque()

        assert mode in ["queueing", "watching"]
```

代码逻辑：
- 保存 send 函数和 fan-out 数量。
- 记录当前 mode。
- 用 `_result_event/_result_values` 表示当前 in-flight 请求。
- 用 `_ready_queue` 保存 queueing 模式下等待者。
- 断言 mode 只允许两种。

为什么这样写：
- 控制请求必须能把多个 rank 的回复聚合成一组。
- 单 in-flight 限制让回复归属简单，不需要为每个控制请求维护 request id。

不变量与失败模式：
- `fan_out` 必须与实际回复数量一致，否则等待协程不会被唤醒或会提前/延迟结束。
- `mode` 只支持 queueing/watchning；拼写错误会触发断言。
- 这个 communicator 需要外部在收到回复时调用 `handle_recv`。

Comment：
和 Detokenizer 的流式状态不同，这里维护的是控制请求的 fan-out/fan-in 状态。

### 4.2 `queueing_call` 串行化控制请求并按 FIFO 唤醒下一位

问题与约束：
- 控制操作可能改变全局状态，例如 flush、配置更新或同步检查，不能并发交错。
- 调用者可能只想等待当前 in-flight 完成，而不发送新对象。

设计选择：
- 若已有 in-flight 或等待队列非空，新调用者创建 event 入队；轮到自己后可选发送 obj，等待收齐 fan-out 响应，清理当前结果并唤醒下一个等待者。

Explain：
`queueing_call` 是严格串行模式。每个调用者只有在前一个请求完全收齐响应并清空 `_result_event/_result_values` 后才能进入发送阶段；`obj is None` 时跳过发送，只等待响应事件。

来源：python/sglang/srt/managers/communicator.py L38-L58

Code：

```python
async def queueing_call(self, obj: T):
    ready_event = asyncio.Event()
    if self._result_event is not None or len(self._ready_queue) > 0:
        self._ready_queue.append(ready_event)
        await ready_event.wait()
        assert self._result_event is None
        assert self._result_values is None

    if obj is not None:
        self._send(obj)

    self._result_event = asyncio.Event()
    self._result_values = []
    await self._result_event.wait()
    result_values = self._result_values
    self._result_event = self._result_values = None

    if len(self._ready_queue) > 0:
        self._ready_queue.popleft().set()

    return result_values
```

代码逻辑：
- 若已有请求在飞或已有等待者，当前调用者入 FIFO 队列。
- 被唤醒后断言没有 in-flight 结果状态。
- `obj` 非 None 时发送控制对象。
- 创建本轮 result event 和结果列表，等待 `handle_recv` 收齐响应。
- 取出结果并清空 in-flight 状态。
- 唤醒下一个等待者。

为什么这样写：
- 严格 FIFO 避免控制消息和响应交错。
- 不为每个请求生成 id，简化 fan-out 收集逻辑。
- `obj is None` 支持同步等待当前结果的场景。

不变量与失败模式：
- 必须由外部调用 `handle_recv` 填充 `_result_values` 并 set event。
- 如果某个 rank 不回复，调用者会一直等待。
- `assert self._result_event is None` 保护队列唤醒时没有残留状态。

Comment：
这是控制面和数据面最大的差异：控制请求宁可串行，也要保证响应归属清楚。

### 4.3 `watching_call` 让并发观察者共享同一个 in-flight 结果

问题与约束：
- 有些控制/观察请求不需要排队发送多次，只要共享当前正在进行的 fan-out 结果即可。
- 第一个醒来的协程会清空共享状态，后醒来的协程仍要拿到同一份结果。

设计选择：
- 没有 in-flight 时由第一个调用者创建 event 并可选发送对象；所有调用者在 await 前捕获本地 `values/event` 引用，event set 后返回 `deepcopy(values)`。

Explain：
`watching_call` 是共享 in-flight 模式。它避免重复发送观察类请求，同时用本地引用解决“第一个协程清空共享字段后，其他等待者仍要读取结果”的竞态。

来源：python/sglang/srt/managers/communicator.py L60-L78

Code：

```python
async def watching_call(self, obj):
    if self._result_event is None:
        assert self._result_values is None
        self._result_values = []
        self._result_event = asyncio.Event()

        if obj is not None:
            self._send(obj)

    values = self._result_values
    event = self._result_event
    await event.wait()

    result_values = copy.deepcopy(values)
    if self._result_event is event:
        self._result_event = self._result_values = None
    return result_values
```

代码逻辑：
- 没有 in-flight 时初始化结果列表和 event。
- `obj` 非 None 时发送一次。
- 每个调用者在 await 前保存当前 values 和 event。
- event 触发后返回结果副本。
- 如果共享字段仍指向本轮 event，当前协程负责清空状态。

为什么这样写：
- 多个观察者共享一次 fan-out，减少重复控制请求。
- `deepcopy` 避免不同协程拿到同一个可变结果列表后互相影响。
- 本地引用保证清空共享状态不会影响后醒来的协程读取本轮结果。

不变量与失败模式：
- 共享模式下并发调用者会得到同一批结果，不适合有副作用且必须逐个执行的控制请求。
- 仍然依赖外部 `handle_recv` 收齐 fan-out 响应。
- 如果返回对象不能 deepcopy，调用会失败。

Comment：
`queueing` 是“每个调用都排队执行”，`watching` 是“多个调用共享正在执行的一次观察”。

### 4.4 `handle_recv` 收齐 fan-out 响应，`merge_results` 合并成功标志

问题与约束：
- 一个 fan-out 请求必须等所有目标都返回，调用者才能获得完整结果。
- 多 rank 响应常见结构是 `success/message`，需要统一汇总。

设计选择：
- 每个回复追加到 `_result_values`；数量达到 `_fan_out` 时 set event；`merge_results` 用 all(success) 和 message join 汇总结果。

Explain：
`handle_recv` 是 communicator 的接收侧入口。它不区分 queueing 或 watching，二者共享同一个“收齐 N 个回复就唤醒等待者”的规则；`merge_results` 只负责把常见控制响应压成一个成功标志和合并消息。

来源：python/sglang/srt/managers/communicator.py L86-L96

Code：

```python
def handle_recv(self, recv_obj: T):
    self._result_values.append(recv_obj)
    if len(self._result_values) == self._fan_out:
        self._result_event.set()

@staticmethod
def merge_results(results):
    all_success = all([r.success for r in results])
    all_message = [r.message for r in results]
    all_message = " | ".join(all_message)
    return all_success, all_message
```

代码逻辑：
- 收到一个回复就 append 到当前结果列表。
- 结果数量达到 fan-out 后唤醒等待调用者。
- `merge_results` 对所有 `success` 做与运算。
- 将所有 message 用分隔符拼接。

为什么这样写：
- fan-out 控制请求的完成条件就是所有目标都回复。
- 合并逻辑保持简单，调用方仍可在需要时读取原始结果列表。

不变量与失败模式：
- `_result_values` 和 `_result_event` 必须已由调用侧初始化。
- 重复回复或超出 `_fan_out` 的回复没有额外防护。
- `merge_results` 假设结果对象都有 `success` 和 `message` 字段。

Comment：
这个控制面工具和 Detokenizer 主循环一样，都依赖“接收侧只负责收齐，语义处理放在调用侧”的设计。
