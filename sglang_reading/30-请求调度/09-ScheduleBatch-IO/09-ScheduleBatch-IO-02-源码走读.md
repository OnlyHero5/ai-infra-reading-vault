---
type: batch-doc
module: 09-ScheduleBatch-IO
batch: "09"
doc_type: walkthrough
title: "ScheduleBatch-IO · 源码走读"
tags:
 - sglang/batch/09
 - sglang/module/schedule-batch-io
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-05
---
# ScheduleBatch-IO · 源码走读

> 走读顺序：`embed_types.py` 的跨模块嵌入结构 → `io_struct.py` 的 IPC 数据契约 → `schedule_batch.py` 的 `Req` / `ScheduleBatch` 生命周期。

---

## 1. embed_types.py：打断循环依赖的嵌入结构

### 1.1 模块拆分

**问题与约束：** `io_struct.py` 需要在请求结构里携带位置嵌入，`schedule_batch.py` 又需要在调度对象里消费同一类数据；如果把类型放进任意一边，都会把 IPC 结构和调度运行时绑成循环 import。

**设计选择：** 把嵌入注入相关结构放进独立的 `embed_types.py`，让两端都只依赖这个小模块。

**Explain：** 这个文件的存在价值不是功能多，而是把“共享数据结构”从“调度逻辑”和“IPC 定义”里剥离出来。

来源：python/sglang/srt/managers/embed_types.py L14-L19

**Code：**

```python
"""
Structs for embedding injection.

These are placed in a separate module to avoid circular imports between
io_struct.py and schedule_batch.py.
"""
```

**代码逻辑：** 文件级 docstring 直接说明该模块只承载 embedding injection 的 struct，且拆分原因是避免 `io_struct.py` 与 `schedule_batch.py` 互相导入。

**为什么这样写：** 位置嵌入覆盖既属于请求输入契约，又会被 scheduler 构造成运行态请求；独立模块让类型边界稳定，避免后续字段扩展时牵动进程间协议和调度实现。

**不变量与失败模式：** 共享类型不能反向 import scheduler 或 io struct；一旦在这里引入重依赖，循环导入会在进程启动或类型解析阶段暴露。

**Comment：** 读这类小文件时重点看“为什么单独存在”，而不是按行数判断它不重要。

### 1.2 PositionalEmbeds 归一化

**问题与约束：** 用户可能传入多个位置嵌入，输入形态可以是单个已堆叠 tensor，也可以是 tensor 列表；IPC 传输又希望对象形态尽量规整。

**设计选择：** `PositionalEmbeds.__post_init__` 在结构创建后把列表归一成 `[N, hidden_dim]` tensor，并校验 tensor 第一维与位置数一致。

**Explain：** 这里把“宽松输入”收敛为“严格内部形态”，后续 scheduler 和序列化路径就不用反复处理多种输入分支。

来源：python/sglang/srt/managers/embed_types.py L27-L54

**Code：**

```python
class PositionalEmbeds(msgspec.Struct, array_like=True):
    embeds: torch.Tensor
    positions: List[int]

    def __post_init__(self):
        if isinstance(self.embeds, list):
            if not self.embeds:
                self.embeds = torch.cat(self.embeds, dim=0)
            elif self.embeds[0].dim() == 1:
                self.embeds = torch.stack(self.embeds, dim=0)
            else:
                self.embeds = torch.cat(self.embeds, dim=0)
        if self.embeds.shape[0] != len(self.positions):
```

**代码逻辑：** 列表为空时交给 `torch.cat` 抛出错误；一维元素用 `stack` 加 batch 维；已有 leading dim 的元素用 `cat` 拼接；最后用长度校验把 embedding 数和 positions 数绑定。

**为什么这样写：** IPC 字段越规整，msgpack / tensor 编码越简单；把检查前置到结构初始化阶段，也能避免模型执行阶段才发现位置嵌入数量错配。

**不变量与失败模式：** `embeds.shape[0] == len(positions)` 必须成立；空列表、维度混乱或位置数量错配都会在构造阶段失败。

**Comment：** 这段适合和 `TokenizedGenerateReqInput.positional_embed_overrides` 连起来看：请求层只保存结构，规范化责任由结构自己承担。

---

## 2. io_struct.py：进程间数据契约

### 2.1 BaseReq / BaseBatchReq / PickleWrapper

**问题与约束：** SGLang 的请求会在 TokenizerManager、Scheduler、DetokenizerManager 之间传输；一部分字段是强类型 struct，另一部分如多模态输入、time stats 是不适合直接 msgpack 的 opaque Python 对象。

**设计选择：** 单请求继承 `BaseReq`，批量消息继承 `BaseBatchReq`；opaque 字段用 `PickleWrapper` 包成 bytes，同时保留外层 msgspec struct。

**Explain：** 这相当于把“可结构化的协议外壳”和“不可结构化的局部载荷”分开处理。

来源：python/sglang/srt/managers/io_struct.py L74-L106

**Code：**

```python
class BaseReq(msgspec.Struct, tag=True, kw_only=True, array_like=True):
    rid: Optional[str] = None
    http_worker_ipc: Optional[str] = None

class BaseBatchReq(msgspec.Struct, tag=True, kw_only=True, array_like=True):
    rids: Optional[List[str]] = None
    http_worker_ipcs: Optional[List[Optional[str]]] = None

class PickleWrapper(msgspec.Struct, tag=True, array_like=True):
    data: bytes
```

**代码逻辑：** 基类固定请求 ID 和 HTTP worker IPC 字段；`PickleWrapper` 只保存 pickle 后的 bytes，并且仍是 msgspec struct，可被 msgpack 外层编码识别。

**为什么这样写：** 大多数 IPC 字段应保持显式类型，便于演进和审计；少数复杂对象单独兜底，避免把整个通道退回 pickle。

**不变量与失败模式：** opaque 字段必须显式 wrap / unwrap；未声明编码路径的对象直接塞进 msgpack 会在编码阶段抛错。

**Comment：** 这里的关键不是“用了 pickle”，而是只在边界字段上使用 pickle。

### 2.2 TokenizedGenerateReqInput：Tokenizer 到 Scheduler 的请求形态

**问题与约束：** Scheduler 不应该再处理原始 HTTP 输入；它需要的是已经分词、采样参数已解析、多模态数据已预处理但仍能跨进程传输的请求。

**设计选择：** `TokenizedGenerateReqInput` 把文本、token ids、embedding、多模态输入、采样参数、路由和观测字段集中成一个 IPC struct，并对 opaque 字段提供 wrap / unwrap 方法。

**Explain：** 这是请求进入 Scheduler 子进程前的“标准化入口对象”。

来源：python/sglang/srt/managers/io_struct.py L777-L879

**Code：**

```python
class TokenizedGenerateReqInput(BaseReq, kw_only=True):
    input_text: Optional[Union[str, List[Union[str, List[str]]]]]
    input_ids: Optional[array]
    input_embeds: Optional[List[List[float]]]
    mm_inputs: Optional[PickleWrapper]
    token_type_ids: Optional[List[int]]
    sampling_params: SamplingParams
    return_logprob: bool
    stream: bool
    positional_embed_overrides: Optional[PositionalEmbeds] = None
    routing_key: Optional[str] = None
    priority: Optional[int] = None
    time_stats: Optional[PickleWrapper] = None

    def wrap_pickle_fields(self):
        self.mm_inputs = wrap_as_pickle(self.mm_inputs)
        self.mm_data_mooncake = wrap_as_pickle(self.mm_data_mooncake)
        self.time_stats = wrap_as_pickle(self.time_stats)
```

**代码逻辑：** 固定字段保存 prompt 与采样控制；扩展字段覆盖 session、LoRA、custom logit processor、disagg、DP routing、priority、cache salt、观测信息；发送前把 `mm_inputs`、`mm_data_mooncake`、`time_stats` 这些 opaque 字段包装。

**为什么这样写：** TokenizerManager 可以把所有“HTTP / tokenizer 侧决策”一次性冻结成 IPC 数据，Scheduler 只面向稳定结构做调度和构造 `Req`。

**不变量与失败模式：** `input_ids`、`sampling_params`、流式和 logprob 相关字段要与上游解析结果一致；未 wrap 的多模态或观测对象会破坏 msgpack 编码。

**Comment：** 读 Scheduler 构造 `Req` 时，应把这个 struct 当成外部请求的来源，而不是再回到 API 层找字段。

### 2.3 BatchTokenIDOutput：Scheduler 的 token 级输出

**问题与约束：** Scheduler 每轮 forward 后拿到的是 token 级增量结果；Detokenizer 需要这些 token、offset、finish reason 和可选统计来生成字符串输出。

**设计选择：** 用 `BatchTokenIDOutput` 批量承载 token ids、增量解码辅助字段、logprobs、hidden states、MoE / indexer 信息和观测字段。

**Explain：** 这个结构是 Scheduler 到 Detokenizer 的主输出契约，也是 skip tokenizer 路径下给 TokenizerManager 的 token 级结果。

来源：python/sglang/srt/managers/io_struct.py L1194-L1273

**Code：**

```python
class BatchTokenIDOutput(BaseBatchReq, kw_only=True):
    finished_reasons: List[Optional[FinishReasonDict]]
    decoded_texts: List[str]
    decode_ids: List[array]
    read_offsets: List[int]
    output_ids: Optional[List[array]]
    prompt_tokens: List[int]
    reasoning_tokens: List[int]
    completion_tokens: List[int]
    cached_tokens: List[int]
    output_hidden_states: OutputHiddenStates
    routed_experts: Optional[List[Optional[torch.Tensor]]]
    customized_info: Optional[PickleWrapper] = None
    time_stats: Optional[PickleWrapper] = None
```

**代码逻辑：** 前半部分服务增量 detokenize 和 token 计数；中段携带 logprob / hidden state / routed expert 等可选调试或训练信息；后半段保存 cache 明细、DP rank、time stats 和 speculative 统计。

**为什么这样写：** Scheduler 不能只返回 next token；流式输出、OpenAI 兼容 finish reason、性能观测和训练侧采样追踪都要跟请求一一对齐。

**不变量与失败模式：** 批量字段长度必须与 `rids` / batch 内请求数一致；tensor 字段在 msgpack 路径要能被 `enc_hook` 处理，opaque 字段要用 `PickleWrapper`。

**Comment：** `decode_ids` 和 `read_offsets` 是理解 Detokenizer 增量输出的两个锚点。

### 2.4 BatchStrOutput：Detokenizer 的字符串输出

**问题与约束：** TokenizerManager 面向 API 客户端，需要的是字符串、token 计数和可选元数据；同时有些字段在 token 级输出里是 tensor，不适合直接回到 tokenizer 热路径处理。

**设计选择：** Detokenizer 使用 `BatchStrOutput` 把 token 级结果转换成字符串输出，并把 routed experts / indexer topk 等 tensor 预编码成字符串字段。

**Explain：** 这一步把“模型内部 token 结果”翻译为“API 层可消费结果”。

来源：python/sglang/srt/managers/io_struct.py L1276-L1349

**Code：**

```python
class BatchStrOutput(BaseBatchReq, kw_only=True):
    finished_reasons: List[Optional[FinishReasonDict]]
    output_strs: List[str]
    output_ids: Optional[List[array]]
    prompt_tokens: List[int]
    completion_tokens: List[int]
    reasoning_tokens: List[int]
    cached_tokens: List[int]
    output_hidden_states: OutputHiddenStates
    routed_experts: Optional[List[Optional[str]]]
    indexer_topk: Optional[List[Optional[str]]]
    time_stats: Optional[PickleWrapper] = None
```

**代码逻辑：** 它保留与 token 输出相同的 finish reason、计数、logprob 和观测信息，但把最终文本放在 `output_strs`，把 routed expert / indexer tensor 变成可传输字符串。

**为什么这样写：** Detokenizer 是离 tokenizer 最近的转换层，适合把二进制或 tensor 形态提前转成 API 响应更容易处理的形态。

**不变量与失败模式：** 字符串输出要与 token 输出的请求顺序一致；如果 routed expert 等字段未在 Detokenizer 侧转换，TokenizerManager 会重新承担较重的数据转换工作。

**Comment：** 这两个输出结构一前一后，正好对应 Scheduler 和 Detokenizer 的职责分界。

### 2.5 wrap_as_pickle / unwrap_from_pickle

**问题与约束：** 同一个 IPC 层要支持默认 msgpack，也要支持环境变量切到 pickle IPC；字段级包装不能在两种模式下产生不兼容对象。

**设计选择：** `wrap_as_pickle` / `unwrap_from_pickle` 在 pickle IPC 模式下保持 no-op，在 msgpack 模式下才把对象转成 `PickleWrapper`。

**Explain：** 这个小函数让调用方不用关心当前 IPC 后端，只要在 opaque 字段上固定调用即可。

来源：python/sglang/srt/managers/io_struct.py L2160-L2173

**Code：**

```python
def wrap_as_pickle(obj: Optional[object]) -> Optional[object]:
    if obj is None:
        return None
    if _USE_PICKLE_IPC:
        return obj
    return PickleWrapper(pickle.dumps(obj))

def unwrap_from_pickle(obj: Optional[object]) -> Optional[object]:
    if obj is None:
        return None
    if _USE_PICKLE_IPC:
        return obj
    assert isinstance(obj, PickleWrapper)
    return pickle.loads(obj.data)
```

**代码逻辑：** `None` 直接透传；pickle IPC 直接透传原对象；msgpack IPC 在发送端 pickle 成 bytes，接收端断言 wrapper 类型再反序列化。

**为什么这样写：** IPC backend 是部署配置，不应该扩散到每个请求字段处理点；统一函数能避免一部分字段漏处理。

**不变量与失败模式：** msgpack 模式下 unwrap 的对象必须是 `PickleWrapper`；如果发送端没有 wrap，接收端会在断言处失败。

**Comment：** 它和 `TokenizedGenerateReqInput.wrap_pickle_fields()` 是一组：一个定义策略，一个列出字段。

### 2.6 enc_hook / dec_hook

**问题与约束：** msgpack 不知道如何编码 Python `array`、`torch.Tensor`、`np.ndarray` 等运行时对象；这些对象又是批量 token 和统计结果里高频出现的数据形态。

**设计选择：** 用 msgspec 的 `enc_hook` / `dec_hook` 为少数已知类型提供显式二进制编码，未知类型直接抛错并要求使用 `PickleWrapper` 或新增分支。

**Explain：** 这里是“结构化快速通道”的核心：常见数值容器走确定编码，不让任意 Python 对象偷偷混入。

来源：python/sglang/srt/managers/io_struct.py L2176-L2220

**Code：**

```python
def enc_hook(obj: Any) -> Any:
    if isinstance(obj, array):
        return (obj.typecode, obj.tobytes())
    elif isinstance(obj, torch.Tensor):
        tensor_dtype = str(obj.dtype).removeprefix("torch.")
        raw_data = (
            obj.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        )
        return (obj.shape, tensor_dtype, raw_data)
    elif isinstance(obj, np.ndarray):
        raw_data = np.ascontiguousarray(obj).reshape(-1).view(np.uint8).data
        return (obj.shape, obj.dtype.str, raw_data)

def dec_hook(tp: Type, obj: Any) -> Any:
    if tp is array:
        typecode, raw_data = obj
        res = array(typecode)
        res.frombytes(raw_data)
        return res
    elif tp is torch.Tensor:
        shape, dtype, data = obj
        tensor_dtype = getattr(torch, dtype)
        return torch.frombuffer(bytearray(data), dtype=tensor_dtype).reshape(shape)
```

**代码逻辑：** `array` 保存 typecode 和 bytes；tensor 保存 shape、dtype 和连续 CPU bytes；numpy 保存 shape、dtype 和连续内存；解码时按类型恢复容器。

**为什么这样写：** token ids、seq lens、logprob 等数据量大且结构明确，走专门编码比全量 pickle 更可控，也能在错误信息里逼迫新类型显式设计传输方案。

**不变量与失败模式：** tensor 编码会触发 `.cpu().contiguous()`，因此 GPU tensor 跨进程要付 D2H 拷贝成本；未知对象必须显式包进 `PickleWrapper`，否则 TypeError。

**Comment：** 看到 tensor IPC 时要记住：这里传的是值拷贝，不是跨进程共享 GPU 指针。

### 2.7 msgpack_encode / msgpack_decode

**问题与约束：** IPC 顶层对象可能是 msgspec struct，也可能是少数 primitive；某些请求类型还含有 opaque 字段，需要整体包装兜底。

**设计选择：** 初始化 `_all_types` 给 msgpack decoder 一个明确 union；编码前通过 `_maybe_wrap_pickle` 决定是否整体包成 `PickleWrapper`。

**Explain：** 这里把“允许哪些顶层类型过 IPC”变成显式白名单。

来源：python/sglang/srt/managers/io_struct.py L2223-L2279

**Code：**

```python
_struct_types = tuple(
    cls
    for cls in BaseReq.__subclasses__()
    + BaseBatchReq.__subclasses__()
    + [PickleWrapper]
)
_primitive_types = (int, float, bool, bytes)
_msgpack_encoder = msgspec.msgpack.Encoder(enc_hook=enc_hook)
_msgpack_decoder = msgspec.msgpack.Decoder(Union[_all_types], dec_hook=dec_hook)

def _maybe_wrap_pickle(obj: Any) -> Any:
    if isinstance(obj, _REQ_TYPES_WITH_OPAQUE_FIELDS):
        return PickleWrapper(pickle.dumps(obj))
    if isinstance(obj, (msgspec.Struct, *_primitive_types)):
        return obj
    raise TypeError(...)

def msgpack_encode(obj: Any) -> bytes:
    return _msgpack_encoder.encode(_maybe_wrap_pickle(obj))
```

**代码逻辑：** struct 子类和 primitive 进入 decoder union；特殊请求类型可整体 wrap；其他顶层对象直接拒绝；decode 后如果顶层是 `PickleWrapper` 再解包。

**为什么这样写：** IPC 协议的稳定性依赖类型边界清晰；白名单能让新增消息类型必须经过显式注册或继承，而不是默默变成动态对象传输。

**不变量与失败模式：** 新增 struct 后若不继承基类或不 hook custom type，msgpack decode 不认识；顶层字符串不在 primitive 里，必须按错误提示处理。

**Comment：** 这段是排查“某个新 IPC 消息为什么发不出去”的第一站。

### 2.8 sock_send / sock_recv

**问题与约束：** 上层 manager 需要一个统一 socket API；底层既可能是 pickle IPC，也可能是 msgpack IPC，还要支持同步和 asyncio socket。

**设计选择：** `sock_send` / `sock_recv` 只在最外层分支 `_USE_PICKLE_IPC`，默认走 `msgpack_encode` / `msgpack_decode`；异步发送复用同一编码逻辑。

**Explain：** 这是 IPC 序列化落到 ZMQ 的最后一层。

来源：python/sglang/srt/managers/io_struct.py L2282-L2305

**Code：**

```python
def sock_send(socket: zmq.Socket, obj: Any, flags: int = 0) -> None:
    if _USE_PICKLE_IPC:
        socket.send_pyobj(obj, flags=flags, protocol=pickle.HIGHEST_PROTOCOL)
        return
    socket.send(msgpack_encode(obj), flags=flags)

def sock_recv(socket: zmq.Socket, flags: int = 0) -> Any:
    if _USE_PICKLE_IPC:
        return socket.recv_pyobj(flags=flags)
    data = socket.recv(flags=flags)
    return msgpack_decode(data)

async def async_sock_send(socket: zmq.asyncio.Socket, obj: Any, flags: int = 0) -> None:
```

**代码逻辑：** 发送端按模式选择 `send_pyobj` 或二进制 msgpack；接收端对称选择 `recv_pyobj` 或 `recv` 后 decode；异步版本只替换 socket 调用方式。

**为什么这样写：** manager 层不应散落序列化分支；统一入口让调试 IPC 行为时只需检查环境变量和这几个函数。

**不变量与失败模式：** 两端必须使用同一 IPC 模式；模式不一致会导致接收端用错误解码器解释 bytes。

**Comment：** 读全链路时，`sock_send(TokenizedGenerateReqInput)` 和 `sock_recv(BatchTokenIDOutput)` 都会落到这里。

---

## 3. schedule_batch.py：运行态请求与批次

### 3.1 多模态 pad value 的全局偏移

**问题与约束：** 多模态占位 token 要放进 `input_ids`，但不能与正常文本 token ID 冲突；同时 pad value 还要能按多模态内容区分，支撑 prefix cache。

**设计选择：** 定义 `MM_PAD_SHIFT_VALUE = 1_000_000` 作为高位偏移，实际 pad value 等于偏移加上 hash 的低 30 位。

**Explain：** 这让多模态占位 token 在 token ID 空间中远离模型 vocab，同时还能反映多模态内容身份。

来源：python/sglang/srt/managers/schedule_batch.py L127-L146

**Code：**

```python
MM_PAD_SHIFT_VALUE = 1_000_000

@lru_cache(maxsize=1)
def sanity_check_mm_pad_shift_value(vocab_size: int) -> None:
    if vocab_size > MM_PAD_SHIFT_VALUE:
        raise ValueError(...)

def _compute_pad_value(hash: int) -> int:
    return MM_PAD_SHIFT_VALUE + (hash % (1 << 30))
```

**代码逻辑：** 启动或模型配置检查时确认 vocab size 没超过偏移；生成 pad value 时只取 hash 的模，避免无限大整数进入 token 序列。

**为什么这样写：** prefix cache 依赖 token 序列作为 key 的一部分；多模态内容如果只用统一占位符，会把不同图片或视频错误视为相同前缀。

**不变量与失败模式：** `vocab_size <= MM_PAD_SHIFT_VALUE` 必须成立；否则正常 token ID 和多模态 pad value 可能重叠，cache key 和模型输入都会被污染。

**Comment：** 这是多模态输入能进入文本 token 管线的桥接点。

### 3.2 MultimodalDataItem.set_pad_value

**问题与约束：** 每个多模态 item 可能已经有外部 hash，也可能只有 feature 或预计算 embedding；调度前必须得到稳定 pad value。

**设计选择：** `set_pad_value()` 优先复用已有 `pad_value`，再按环境变量选择随机 UUID 或对 feature / precomputed embedding 计算 hash。

**Explain：** 这段把“内容身份”变成 scheduler 可以放进 token 序列的整数。

来源：python/sglang/srt/managers/schedule_batch.py L296-L318

**Code：**

```python
def set_pad_value(self):
    if self.pad_value is not None:
        return

    from sglang.srt.managers.mm_utils import hash_feature

    if envs.SGLANG_MM_SKIP_COMPUTE_HASH.get():
        import uuid
        self.hash = uuid.uuid4().int
        self.pad_value = _compute_pad_value(self.hash)
        return
    if self.hash is None:
        if self.feature is not None:
            hashed_feature = self.feature
        else:
            hashed_feature = self.precomputed_embeddings
        self.hash = hash_feature(hashed_feature)
    assert self.hash is not None
    self.pad_value = _compute_pad_value(self.hash)
```

**代码逻辑：** 已计算过则不重复；跳过 hash 的模式用 UUID 保持唯一性但牺牲跨请求复用；正常路径对 feature 或 embedding hash，再调用 `_compute_pad_value`。

**为什么这样写：** 多模态 hash 可能代价高，且在某些部署里希望跳过；但即便跳过，也不能让不同 item 共用同一个 pad value。

**不变量与失败模式：** `self.hash` 最终必须非空；跳过 hash 会降低 prefix cache 命中稳定性，外部 KV router 也需要理解这一路径的影响。

**Comment：** 如果多模态 prefix cache 命中异常，先看 hash 来源，再看 pad value 是否稳定。

### 3.3 Finish Reason

**问题与约束：** Scheduler 内部要区分 stop token、stop string、regex、长度截断和 abort；API 输出又要尽量兼容 OpenAI 的 finish reason 形态。

**设计选择：** 每类 finish reason 都实现 `to_json()`，把内部对象转换成统一 dict：`stop`、`length` 或 `abort`。

**Explain：** Scheduler 可以保留类型化内部状态，输出时再变成跨进程可传输的普通 dict。

来源：python/sglang/srt/managers/schedule_batch.py L154-L215

**Code：**

```python
class FINISH_MATCHED_TOKEN(BaseFinishReason):
    def to_json(self):
        return {"type": "stop", "matched": self.matched}

class FINISH_LENGTH(BaseFinishReason):
    def to_json(self):
        return {"type": "length", "length": self.length}

class FINISH_ABORT(BaseFinishReason):
    def to_json(self):
        return {
            "type": "abort",
            "message": self.message,
            "status_code": self.status_code,
            "err_type": self.err_type,
        }
```

**代码逻辑：** token / string / regex 命中都归一成 `type: stop`；长度限制归一成 `type: length`；abort 额外携带 message、status code 和错误类型。

**为什么这样写：** 进程内需要区分触发来源，进程间和 API 层需要稳定 JSON 形态；`to_json()` 把两者解耦。

**不变量与失败模式：** `finished_reasons` 里应保存可序列化 dict；如果直接把 finish reason 对象跨 IPC 传出，会增加 decoder 类型负担。

**Comment：** `BatchTokenIDOutput.finished_reasons` 的值就是从这里来的。

### 3.4 Req 的输入输出核心字段

**问题与约束：** 一个请求从 prefill 到多轮 decode 会持续增长输出 token，同时还要保留原始 prompt、未截断 fill ids 和本轮 extend 区间。

**设计选择：** `Req` 保存 `origin_input_ids`、`origin_input_ids_unpadded`、append-only 的 `output_ids`，以及由 `_refresh_fill_ids` 维护的 `full_untruncated_fill_ids`。

**Explain：** `Req` 是 Scheduler 内部的单请求运行态，不再只是 IPC struct。

来源：python/sglang/srt/managers/schedule_batch.py L713-L731

**Code：**

```python
self.rid = rid
self.origin_input_ids = origin_input_ids
self.origin_input_ids_unpadded = (
    origin_input_ids_unpadded
    if origin_input_ids_unpadded
    else self.origin_input_ids
)
self.output_ids = array("q")
self.full_untruncated_fill_ids = array("q")
self.extend_range: Optional[Range] = None
self.dllm_initialized: bool = False
```

**代码逻辑：** 多模态场景下保留 unpadded prompt；输出 token 用 `array("q")` 追加；完整未截断序列独立保存；`extend_range` 标记本轮 prefill / extend 要处理的区间。

**为什么这样写：** 调度、prefix cache、logprob 统计和 detokenization 都会读取不同粒度的序列；单一列表很难同时表达 padded / unpadded、已输出 / 未处理、截断 / 未截断状态。

**不变量与失败模式：** `output_ids` 语义上应 append-only；如果原地改写但长度不变，依赖长度推导的 fill ids 刷新会被悄悄破坏。

**Comment：** 分清 `origin_input_ids`、`output_ids`、`full_untruncated_fill_ids`，基本就能跟上 `Req` 的 token 生命周期。

### 3.5 Req 的 prefix cache 字段

**问题与约束：** Scheduler 要记录命中的 KV cache slot、host / storage 命中长度、Radix tree 节点和 SWA 锁信息；这些状态既影响本轮输入长度，也影响 cache 插入和释放。

**设计选择：** `Req` 内部集中保存 `prefix_indices`、last node、host hit length、storage hit length、SWA 锁和 `cache_protected_len`。

**Explain：** prefix cache 不是纯函数查询结果，它会在请求生命周期里持续影响内存和调度决策。

来源：python/sglang/srt/managers/schedule_batch.py L848-L871

**Code：**

```python
self.prefix_indices: torch.Tensor = torch.empty((0,), dtype=torch.int64)
self.last_node: Any = None
self.last_host_node: Any = None
self.best_match_node: Any = None
self.host_hit_length = 0
self.swa_host_hit_length = 0
self.mamba_host_hit_length = 0
self.num_matched_prefix_tokens = 0
self.storage_hit_length = 0
self.swa_uuid_for_lock: Optional[int] = None
self.swa_prefix_lock_released: bool = False
self.cache_protected_len: int = 0
```

**代码逻辑：** `prefix_indices` 指向已在 KV cache 中复用的 slot；host / storage 命中长度记录分层 cache 来源；SWA 字段控制滑窗 attention 的 radix tree 锁；`cache_protected_len` 记录插入树时需要保护的前缀长度。

**为什么这样写：** Scheduler 既要做 admission，又要维护 cache 生命周期；把这些字段放在 `Req` 上能让单请求携带自己的 cache 上下文。

**不变量与失败模式：** `len(prefix_indices)` 会直接影响 `prepare_for_extend` 截取未缓存 token 的起点；若命中长度和实际 KV slot 不一致，会导致漏算或重复算 prompt token。

**Comment：** `prefix_indices` 是连接 RadixAttention 和 ScheduleBatch 的关键字段。

### 3.6 ScheduleBatch.init_new

**问题与约束：** Scheduler 选中一组 `Req` 后，需要构造批次对象；但此时许多 GPU 张量还依赖 forward mode，不能在初始化时一次性填完。

**设计选择：** `init_new` 只聚合 req 列表、内存池、prefix cache、模型配置和若干布尔标志；具体张量准备交给 `prepare_for_extend` 或 `prepare_for_decode`。

**Explain：** `ScheduleBatch` 是批次运行态容器，初始化阶段只建立“这批请求是谁、共享哪些资源、有什么能力需求”。

来源：python/sglang/srt/managers/schedule_batch.py L1845-L1880

**Code：**

```python
@classmethod
def init_new(
    cls,
    reqs: List[Req],
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    tree_cache: BasePrefixCache,
    model_config: ModelConfig,
    enable_overlap: bool,
    spec_algorithm: SpeculativeAlgorithm,
    chunked_req: Optional[Req] = None,
    dllm_config: Optional[DllmConfig] = None,
):
    return_logprob = any(req.return_logprob for req in reqs)
    batch = cls(
        reqs=reqs,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        return_logprob=return_logprob,
        has_grammar=any(req.grammar for req in reqs),
        return_hidden_states=any(req.return_hidden_states for req in reqs),
        is_prefill_only=all(req.is_prefill_only for req in reqs),
    )
```

**代码逻辑：** 从 req 聚合 `return_logprob`、grammar、hidden states、prefill-only 等标志；绑定 KV pool、req pool、prefix cache 和 speculative / chunked prefill 配置。

**为什么这样写：** 同一批次可能走 prefill、decode、speculative decode 或 chunked prefill；先创建轻量批次，再按模式准备张量，能避免初始化阶段混入太多分支。

**不变量与失败模式：** `reqs` 内的请求应共享同一调度资源上下文；如果 batch 标志没有正确聚合，后续输出字段或采样约束会缺失。

**Comment：** 这段告诉读者：`ScheduleBatch` 初始化不是 forward 输入准备的终点，只是起点。

### 3.7 prepare_for_extend：Prefill / Extend 输入准备

**问题与约束：** Prefill 时并不总是计算完整 prompt；prefix cache 已命中的 token 应跳过，只为未缓存 token 分配 KV slot。

**设计选择：** `prepare_for_extend` 设置 forward mode，从 `fill_ids[len(prefix_indices):]` 截取输入，计算 seq / prefix / extend 长度，并调用 `alloc_for_extend` 分配内存。

**Explain：** 这是 prompt token 进入模型前的批量化和内存分配阶段。

来源：python/sglang/srt/managers/schedule_batch.py L2011-L2058

**Code：**

```python
def prepare_for_extend(self):
    self.forward_mode = ForwardMode.EXTEND
    if self.is_dllm():
        self.forward_mode = ForwardMode.DLLM_EXTEND

    reqs = self.reqs
    input_ids = [r.get_fill_ids()[len(r.prefix_indices) :] for r in reqs]
    extend_num_tokens = sum(len(ids) for ids in input_ids)
    seq_lens = [r.extend_range.end for r in reqs]
    prefix_lens = [len(r.prefix_indices) for r in reqs]
    extend_lens = [r.extend_range.length for r in reqs]
    pinned_input_ids = flatten_arrays_to_pinned_cpu(input_ids, _pin)
    self.prefix_lens = prefix_lens
    self.extend_lens = extend_lens
    self.extend_num_tokens = extend_num_tokens
    out_cache_loc, req_pool_indices_tensor, req_pool_indices_cpu = alloc_for_extend(self)
```

**代码逻辑：** forward mode 决定模型执行路径；`input_ids` 只取未缓存 suffix；长度数组同时保存在 CPU 和 GPU 侧；最后用 allocator 给每个 extend token 分配 KV 位置。

**为什么这样写：** prefix cache 命中越多，实际 prefill token 越少；把截取和分配集中在这里，后续 model runner 可以只消费已经整理好的 batch 输入。

**不变量与失败模式：** `extend_range` 必须已由调度阶段设置；`prefix_indices` 长度必须与已命中 cache 对齐，否则 input 截取会偏移。

**Comment：** 看 prefill 性能时，`extend_num_tokens` 比原始 prompt 长度更接近实际计算量。

### 3.8 prepare_for_decode：Decode 轮次准备

**问题与约束：** Decode 每轮通常只为每个请求新增一个 token 的 KV slot；但 speculative、encoder-decoder、overlap 和 penalty 逻辑都会改变准备路径。

**设计选择：** 常规 decode 先设置 `ForwardMode.DECODE`、清理 prefill-only 状态，再为每个 req 分配一个 slot 并更新 seq lens；特殊路径提前分支。

**Explain：** 这是每轮 decode 进入模型前的批次状态推进。

来源：python/sglang/srt/managers/schedule_batch.py L2618-L2665

**Code：**

```python
def prepare_for_decode(self):
    self.forward_mode = ForwardMode.DECODE
    self.input_embeds = None

    if not self.spec_algorithm.is_none():
        from sglang.srt.speculative.spec_utils import spec_prepare_for_decode
        spec_prepare_for_decode(self)
        return

    if self.sampling_info.penalizer_orchestrator.is_required:
        self.cumulate_penalty_output_tokens()

    self.out_cache_loc = alloc_for_decode(self, token_per_req=1)

    for req in self.reqs:
        req.decode_batch_idx += 1
        req.kv_committed_len += 1
        req.kv_allocated_len += 1

    if self.enable_overlap:
        self.seq_lens = self.seq_lens + 1
    else:
        self.seq_lens.add_(1)
    self.seq_lens_sum = None
```

**代码逻辑：** prefill 的 `input_embeds` 被清空；speculative decode 交给专门函数；常规路径先更新 penalty 状态，再分配 KV slot，最后推进每个请求和 batch 的长度计数。

**为什么这样写：** Decode 路径高频执行，必须让常规路径短而明确；overlap 模式下使用新 tensor 是为了避免和已排队 forward 共享旧引用。

**不变量与失败模式：** decode 前 batch 必须已有上轮输出作为下一轮输入；`seq_lens_sum` 失效后要由下游懒计算，不能继续使用旧和。

**Comment：** Prefill 的核心是“截取未缓存 prompt”，decode 的核心是“每轮推进一个新位置”。

### 3.9 filter_batch：移除已完成或被排除请求

**问题与约束：** 批次中请求会陆续 finish、retract 或被 chunked prefill 临时排除；如果只改 `reqs` 而不裁剪张量，batch 内部索引会错位。

**设计选择：** `filter_batch` 先计算 `keep_indices`，再同步裁剪 `reqs`、multimodal inputs、req pool indices、seq lens、logprob 字段和 sampling info。

**Explain：** 这是运行中批次的“压缩”操作。

来源：python/sglang/srt/managers/schedule_batch.py L2695-L2765

**Code：**

```python
def filter_batch(
    self,
    chunked_req_to_exclude: Optional[Union[Req, List[Req]]] = None,
    keep_indices: Optional[List[int]] = None,
):
    if keep_indices is None:
        keep_indices = [
            i
            for i in range(len(self.reqs))
            if not self.reqs[i].finished()
            and self.reqs[i] not in chunked_req_to_exclude
        ]

    if keep_indices is None or len(keep_indices) == 0:
        self.reqs = []
        return

    self.reqs = [self.reqs[i] for i in keep_indices]
    self.req_pool_indices = self.req_pool_indices[keep_indices_device]
    self.seq_lens = self.seq_lens[keep_indices_device]
    self.orig_seq_lens = self.orig_seq_lens[keep_indices_device]
    self.out_cache_loc = None
    self.seq_lens_sum = None
    self.sampling_info.filter_batch(keep_indices, keep_indices_device)
```

**代码逻辑：** 默认过滤掉 finished req 和指定 chunked req；空批次只清空 `reqs`；非空批次按 keep indices 裁剪所有按请求对齐的字段，并让缓存的 output loc / seq_lens_sum 失效。

**为什么这样写：** Scheduler 的 batch 是多个并行数组组成的结构，任何一个数组漏裁都会让请求和张量位置错配。

**不变量与失败模式：** 所有 per-req 字段都必须按同一 `keep_indices` 更新；`out_cache_loc` 过滤后不能复用，因为下一轮 prepare 会重新分配。

**Comment：** 如果出现某个请求拿到另一个请求的状态，先查 filter / merge 是否维护了所有并行字段。

### 3.10 merge_batch：合并动态批次

**问题与约束：** Chunked prefill、running decode 和 relay staged batch 可能需要合并；合并时不只是拼 `reqs`，还要拼张量、采样状态和可选 metadata。

**设计选择：** `merge_batch` 先合并 `sampling_info`，再 cat / extend 各类 per-req 张量和列表，最后更新 return flags 与 speculative info。

**Explain：** 这是 Scheduler 在动态 batching 下保持单一执行批次的入口。

来源：python/sglang/srt/managers/schedule_batch.py L2772-L2829

**Code：**

```python
def merge_batch(self, other: ScheduleBatch):
    self.sampling_info.merge_batch(other.sampling_info)

    if self.model_config.is_encoder_decoder:
        self.encoder_lens = torch.cat([self.encoder_lens, other.encoder_lens])
        self.encoder_lens_cpu.extend(other.encoder_lens_cpu)
    self.req_pool_indices = torch.cat(
        [self.req_pool_indices, other.req_pool_indices]
    )
    self.seq_lens = torch.cat([self.seq_lens, other.seq_lens])
    self.orig_seq_lens = torch.cat([self.orig_seq_lens, other.orig_seq_lens])
    self.out_cache_loc = None
    self.seq_lens_sum = None
    if self.input_ids is not None and other.input_ids is not None:
        self.input_ids = torch.cat([self.input_ids, other.input_ids])
    else:
        self.input_ids = None
    self.reqs.extend(other.reqs)
    self.return_logprob |= other.return_logprob
    self.has_grammar |= other.has_grammar
```

**代码逻辑：** 采样信息先合并，因为 orchestrator 依赖合并前的 reqs；随后拼 encoder、req pool、seq lens 和 input ids；不可靠的派生字段置空等待重建；最后合并请求列表和 batch flags。

**为什么这样写：** 动态 batching 的风险在于“看似只是拼列表”，实际有大量派生状态需要同步；先处理 sampling info 能避免 penalty orchestrator 在错误 req 上准备状态。

**不变量与失败模式：** 合并后所有 per-req 数组长度必须与 `self.reqs` 一致；如果一侧没有 `input_ids`，宁可置空让后续重建，也不能拼出部分真实部分陈旧的输入。

**Comment：** `filter_batch` 是缩小 batch，`merge_batch` 是扩大 batch；两者共同维护 ScheduleBatch 的并行数组不变量。

---

## 4. 串起来看

`TokenizedGenerateReqInput` 是跨进程请求契约；Scheduler 把它变成 `Req`，再把多个 `Req` 聚合成 `ScheduleBatch`。`prepare_for_extend` 处理 prompt 未缓存部分，`prepare_for_decode` 推进 decode 轮次，`filter_batch` / `merge_batch` 在运行中调整批次。输出方向则从 `BatchTokenIDOutput` 到 `BatchStrOutput`，由 token 级结果走向字符串结果。

最重要的不变量是：请求顺序、per-req 张量、KV slot、prefix 命中长度和输出字段必须始终对齐。ScheduleBatch-IO 的大部分代码都在维护这条对齐关系。
