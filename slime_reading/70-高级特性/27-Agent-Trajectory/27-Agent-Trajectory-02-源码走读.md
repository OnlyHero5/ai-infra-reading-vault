---
type: batch-doc
module: 27-Agent-Trajectory
batch: "27"
doc_type: walkthrough
title: "Agent Trajectory · 源码走读"
tags:
  - slime/batch/27
  - slime/module/agent-trajectory
  - slime/doc/walkthrough
updated: 2026-07-05
---

# Agent Trajectory · 源码走读

> 走读主线：协议 adapter 把 OpenAI/Anthropic 请求翻译成 chat-template messages 和 `TurnRecord`；`TrajectoryManager` 用 per-session message tree 记录多轮分支；`_SampleBuilder` 根据 token drift 把生成 turn 线性化为带 loss_mask/logprob 的 `Sample`。

---

## 1. Adapter：协议入口与 TurnRecord 生成

### 1.1 tool_call_dict 保持 manager_message 的工具调用规范形态

问题与约束：
- 轨迹树通过 dict equality 匹配历史消息；如果 tool call arguments 在一次请求里是 JSON 字符串、另一次是 dict，就会误判为不同历史分支。

设计选择：
- `tool_call_dict` 存储 OpenAI shape 的工具调用，但让 `arguments` 保持 dict，并丢弃 wire-only tool-call id。

Explain：
adapter 侧把协议消息转换成 manager 可比较的 canonical dict。这样客户端重放历史时，即使 JSON key 顺序不同，只要语义上是同一个 dict，trajectory manager 就能命中同一条路径。

来源：slime/agent/adapters/common.py L110-L118

Code：

```python
def tool_call_dict(name: str, arguments: dict | None) -> dict:
    """Canonical OpenAI-shape tool call stored on manager_message.

    arguments stays a dict (not a JSON string): the chat template needs a
    mapping, and the trajectory manager matches history by dict equality, so a
    sampled leaf and its replayed echo compare equal regardless of key order.
    The wire-only tool-call id is dropped for the same reason.
    """
    return {"type": "function", "function": {"name": name, "arguments": arguments or {}}}
```

代码逻辑：
- 函数只返回 manager 内部使用的 tool call dict。
- `arguments or {}` 避免 None 进入历史匹配。
- 不保留请求协议上的 tool call id。

为什么这样写：
- 轨迹树的匹配规则是 dict equality，canonical 形态比保留 wire 细节更重要。
- tool id 对训练消息语义不是必要字段，保留反而会导致 replay 不一致。

不变量与失败模式：
- arguments 必须是 dict；调用方如果传 JSON 字符串，仍会破坏匹配语义。
- 函数名必须稳定，否则工具调用历史会 fork。

Comment：
这是 agent trajectory 中一个很小但很关键的历史匹配不变量。

### 1.2 BaseAdapter 初始化共享 TrajectoryManager

问题与约束：
- 一个 adapter 进程会服务多个 session；每个 session 要有独立轨迹树，但 adapter 的协议逻辑应复用同一个 manager。

设计选择：
- `BaseAdapter.__init__` 创建一个共享 `TrajectoryManager`，`sid` 隔离具体树；同时注册 health/model route 和协议子类 route。

Explain：
`fork_threshold_tokens` 可传给 manager 控制短 assistant rewrite merge 的阈值。`store/inflight/closed/_sid_turn_count` 在 adapter 层管理 session 生命周期和并发请求。

来源：slime/agent/adapters/common.py L141-L176

Code：

```python
def __init__(
    self,
    *,
    tokenizer,
    sglang_url,
    tool_parser=None,
    reasoning_parser=None,
    max_turns_per_sid: int | None = None,
    fork_threshold_tokens: int | None = None,
    debug_callback: Callable[..., None] | None = None,
) -> None:
    self.tokenizer = tokenizer
    self.sglang_url = sglang_url.rstrip("/") if isinstance(sglang_url, str) else sglang_url
    self.tool_parser = tool_parser
    self.reasoning_parser = reasoning_parser
    self.store: dict[str, Any] = {}
    self.inflight: dict[str, set[asyncio.Task]] = {}
    self.closed: set[str] = set()
    self.app = web.Application(client_max_size=64 * 1024 * 1024)

    mgr_kwargs: dict[str, int] = {}
    if fork_threshold_tokens is not None:
        mgr_kwargs["fork_threshold_tokens"] = fork_threshold_tokens
    self.manager = TrajectoryManager(**mgr_kwargs)

    self.debug_callback = debug_callback
    self.max_turns_per_sid = max_turns_per_sid
    self._sid_turn_count: dict[str, int] = {}

    self.app.router.add_get("/healthz", _health)
    self.app.router.add_get("/v1/models", _health)
    self._register_routes(self.app)
```

代码逻辑：
- adapter 持有 tokenizer 和 upstream SGLang URL。
- manager 只创建一次，内部按 sid 维护树。
- `_register_routes` 是协议子类必须实现的 hook。

为什么这样写：
- 多协议 adapter 可以共享 session/turn 处理管线，只差 wire translation 和 response framing。
- manager 不依赖 HTTP 框架，便于测试和复用。

不变量与失败模式：
- sid 必须能稳定映射到同一 session，否则同一 agent 轨迹会被拆到多个树。
- max turn cap 只在 adapter 层执行，manager 不负责限流。

Comment：
Adapter 管协议生命周期，TrajectoryManager 管训练轨迹结构。

### 1.3 _sampling_params 与 call_sglang_generate 前置约束

问题与约束：
- Agent 生成要保留 stop token、special token 和 logprob 对齐；同时不能超过 session 的最大上下文。

设计选择：
- `_sampling_params` 固定 `skip_special_tokens=False`、`spaces_between_special_tokens=False`、`no_stop_trim=True`，再从协议 body 覆盖 max tokens、temperature/top_p/top_k 和 stop。

Explain：
`call_sglang_generate` 先计算 sampling params，再根据 `session.max_context_tokens` 缩小 `max_new_tokens`。如果 prompt 已超长，直接返回 finish_reason 为 length 的空输出 TurnRecord。

来源：slime/agent/adapters/common.py L416-L473

Code：

```python
def _sampling_params(session: Any, body: dict, *, max_token_keys: tuple[str, ...], stop_keys: tuple[str, ...]) -> dict:
    sp: dict[str, Any] = {
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "no_stop_trim": True,
        "max_new_tokens": 4096,
        **(session.sampling_defaults or {}),
    }

    for key in max_token_keys:
        if body.get(key) is not None:
            sp["max_new_tokens"] = min(int(sp.get("max_new_tokens", body[key])), int(body[key]))
            break

    for src_k, dst_k in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k")):
        if src_k in body:
            sp[dst_k] = body[src_k]

    for key in stop_keys:
        if body.get(key):
            sp["stop"] = body[key]
            break

    return sp

async def call_sglang_generate(...):
    sp = _sampling_params(session, body, max_token_keys=adapter.max_token_keys, stop_keys=adapter.stop_keys)

    if session.max_context_tokens > 0:
        remaining_context = session.max_context_tokens - len(prompt_ids)
        if remaining_context <= 0:
            return TurnRecord(prompt_ids=list(prompt_ids), output_ids=[], finish_reason="length")
        sp["max_new_tokens"] = min(int(sp.get("max_new_tokens", remaining_context)), remaining_context)

    rid = uuid.uuid4().hex
    headers = {"X-SMG-Routing-Key": session_id} if session_id and session_id != "default" else None
```

代码逻辑：
- adapter 子类提供 max token key 和 stop key 的协议差异。
- sampling defaults 先合并，再被请求体覆盖。
- session 上下文上限会进一步裁剪 max_new_tokens。
- sid 非 default 时作为 routing key header。

为什么这样写：
- Agent 训练需要完整 token/logprob，不应让上游 trim stop 或跳过 special token。
- max context 在 adapter 层提前处理，避免向 SGLang 发送必然越界的请求。

不变量与失败模式：
- `session.max_context_tokens` 为正时必须覆盖 prompt+response。
- 协议 body 中 max token 字段必须能转 int。

Comment：
采样参数这里已经被训练需求改造过，不是普通 OpenAI/Anthropic pass-through。

### 1.4 call_sglang_generate 用 /generate 返回构造 TurnRecord

问题与约束：
- TrajectoryManager 不直接调用 SGLang；它需要 adapter 提供 prompt/output token ids、finish reason 和 output logprob。

设计选择：
- Adapter POST `/generate`，强制 `return_logprob=True`，从 `meta_info.output_token_logprobs` 取 token id 和 logprob，封装成 `TurnRecord`。

Explain：
请求取消、aiohttp 错误或超时时，adapter 会主动 POST `/abort_request`，尽早释放 SGLang 中被孤儿生成占用的 KV。

来源：slime/agent/adapters/common.py L475-L518

Code：

```python
async with aiohttp.ClientSession(timeout=timeout) as sess, sess.post(
    f"{sglang_url}/generate",
    json={
        "rid": rid,
        "input_ids": prompt_ids,
        "sampling_params": sp,
        "return_logprob": True,
    },
    headers=headers,
) as r:
    if r.status >= 400:
        text = await r.text()
        raise RuntimeError(f"sglang upstream {r.status}: {text[:400]}")
    data = await r.json(content_type=None)
meta = data.get("meta_info") or {}
output_token_logprobs = meta.get("output_token_logprobs") or []
output_ids = [x[1] for x in output_token_logprobs]
output_log_probs = [float(x[0]) for x in output_token_logprobs]
finish = (meta.get("finish_reason") or {}).get("type", "stop") or "stop"
```

```python
except (asyncio.CancelledError, aiohttp.ClientError, asyncio.TimeoutError) as e:
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s2:
            await s2.post(f"{sglang_url}/abort_request", json={"rid": rid})
    except Exception:
        pass
    raise

return TurnRecord(
    prompt_ids=list(prompt_ids),
    output_ids=output_ids,
    finish_reason=finish,
    output_log_probs=output_log_probs,
)
```

代码逻辑：
- 请求体使用 `input_ids`，不是 text。
- `return_logprob=True` 是训练轨迹构造的硬要求。
- output token ids 和 logprobs 直接来自 SGLang meta。
- 异常路径主动 abort rid。

为什么这样写：
- 重新 tokenize 文本会引入 drift，直接使用 SGLang 返回 token id 才能和 logprob 对齐。
- manager 只需要 token 事实，不需要理解 HTTP upstream 细节。

不变量与失败模式：
- SGLang 必须返回 `output_token_logprobs`，否则 output_ids 为空。
- `output_log_probs` 长度必须和 output_ids 一致，后续 `record_turn` 会 assert。

Comment：
`TurnRecord` 是 adapter 与 trajectory manager 之间的核心数据契约。

### 1.5 _run_turn 先响应客户端，再记录 trajectory

问题与约束：
- 如果客户端在 response flush 前断开，就不应把这个 turn 记录为训练数据；否则会训练一个客户端从未收到的回复。

设计选择：
- `_run_turn` 完成 translate、SGLang generate、parse、reply build 后，先 `_respond`；只有 response 成功构造/flush 后才 `manager.record_turn`。

Explain：
wire-specific 部分由子类 hook 实现：`_translate/_build_reply/_respond`。共享部分负责 sid、turn cap、inflight、debug callback 和 record_turn。

来源：slime/agent/adapters/common.py L318-L391

Code：

```python
async def _run_turn(self, request: web.Request) -> web.StreamResponse:
    body = await request.json()
    self._preprocess_body(body)
    sid = self._session_id(request, body)
    if sid in self.closed:
        return web.Response(status=503, text="session closed")
    capped = self._check_turn_cap(sid)
    if capped is not None:
        return capped

    tok = self.tokenizer
    s = self.store.setdefault(sid, Session())
    task = asyncio.current_task()
    self.inflight.setdefault(sid, set()).add(task)
    try:
        translated, tools_schema = self._translate(body)
        prompt_ids = _render_token_ids(translated, tok, tools=tools_schema, add_generation_prompt=True)

        turn = await call_sglang_generate(prompt_ids, s, body, adapter=self, session_id=sid)
        raw_output = tok.decode(turn.output_ids, skip_special_tokens=False) if turn.output_ids else ""
        parsed = parse_model_output(
            raw_output,
            tools_schema=tools_schema,
            tool_parser_name=self.tool_parser,
            reasoning_parser_name=self.reasoning_parser,
        )
        reply = self._build_reply(parsed, turn.finish_reason, translated, tools_schema)
        turn = dataclasses.replace(turn, ill_formed=parsed.ill_formed)
        response = await self._respond(request, body, reply, in_tok, out_tok, stream)

        self.manager.record_turn(
            sid,
            turn=turn,
            prompt_messages=translated,
            response_message=reply.manager_message,
            metadata={"sid": sid},
        )
        return response
    finally:
        self.inflight.get(sid, set()).discard(task)
```

代码逻辑：
- sid 和 session store 在 adapter 层管理。
- prompt messages 先渲染为 token ids，再发给 SGLang。
- parse 结果写入 reply 和 `turn.ill_formed`。
- 成功响应后才写入 manager。

为什么这样写：
- 训练轨迹应该反映真实交互历史，而不是服务端尝试生成过的所有内容。
- inflight 记录让 finish/drop session 能等待或清理正在进行的请求。

不变量与失败模式：
- 子类 `_respond` 抛出连接断开时，turn 不会被 record。
- `_translate` 输出 messages 必须能被同一 tokenizer/chat template 渲染。

Comment：
这段定义了 agent turn 的共享管线。

### 1.6 finish_session 从 manager 消费轨迹并在 adapter 层 decode response

问题与约束：
- TrajectoryManager 是 tokenizer-free 的；但最终 Sample 需要 `.response` 字符串供日志、评估或调试。

设计选择：
- `finish_session` 先 `shutdown_session` 等待 inflight，再调用 `manager.get_trajectory`，最后由 adapter tokenizer decode 每个 sample 的 response tail。

Explain：
函数从 store 中取出 session，用 session 的 max context 作为 sample 截断上限。manager 消费 sid 后，第二次调用会返回空列表。

来源：slime/agent/adapters/common.py L251-L276

Code：

```python
async def finish_session(
    self,
    sid: str,
    *,
    base_sample,
    reward: float = 0.0,
    extra_metadata: dict | None = None,
    wait_timeout: float = 5.0,
) -> list:
    await self.shutdown_session(sid, wait_timeout=wait_timeout)
    session = self.store.pop(sid, None)
    max_sample_tokens = int(getattr(session, "max_context_tokens", 0) or 0) if session is not None else 0
    samples = self.manager.get_trajectory(
        sid,
        base_sample=base_sample,
        reward=reward,
        extra_metadata=extra_metadata,
        max_sample_tokens=max_sample_tokens,
    )
    for s in samples:
        rlen = int(s.response_length or 0)
        s.response = (
            self.tokenizer.decode(s.tokens[-rlen:], skip_special_tokens=False) if rlen and s.tokens else ""
        )
    return samples
```

代码逻辑：
- 先等待 inflight turn 收尾。
- manager 返回 Sample 列表。
- response 字符串由最后 `response_length` 个 token decode 得到。

为什么这样写：
- manager 只处理 token/loss/logprob，不依赖 tokenizer，有利于协议无关。
- adapter 才知道具体 tokenizer 和 special token decode 选项。

不变量与失败模式：
- `response_length` 必须与 Sample tokens 的尾部 response 区域一致。
- 过早 finish 可能导致 inflight 超时后未记录完整轨迹。

Comment：
session drain 是 agent rollout 进入训练样本的边界。

---

## 2. TrajectoryManager：树与线性化

### 2.1 TurnRecord 与 MessageNode 区分 generated 与 routing-only 节点

问题与约束：
- 多轮 agent 历史里既有模型真实生成的 assistant 消息，也有客户端 replay 的 system/user/tool/assistant 历史；训练只应覆盖真实生成的响应。

设计选择：
- `TurnRecord` 保存一次 SGLang `/generate` snapshot；`MessageNode.turn is not None` 表示 generated assistant，`turn is None` 表示 routing-only 节点。

Explain：
MessageNode 形成 per-session routing tree。`response_trained` 标记用于 sibling leaf 共享同一个 assistant 节点时只训练一次，后续路径把它作为 loss_mask=0 上下文重放。

来源：slime/agent/trajectory.py L28-L82

Code：

```python
@dataclasses.dataclass(frozen=True)
class TurnRecord:
    prompt_ids: list[int]
    output_ids: list[int]
    finish_reason: str
    output_log_probs: list[float] = dataclasses.field(default_factory=list)
    ill_formed: bool = False


class MessageNode:
    def __init__(
        self,
        *,
        role: str | None = None,
        message: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        parent: MessageNode | None = None,
    ) -> None:
        self.role = role
        self.message = message
        self.metadata = dict(metadata or {})
        self.parent: MessageNode | None = parent
        self.children: list[MessageNode] = []
        self.turn: TurnRecord | None = None
        self.turn_index: int | None = None
        self.response_trained: bool = False
```

代码逻辑：
- TurnRecord 是不可变 dataclass。
- MessageNode 保存一条 chat message 和树关系。
- generated/routing-only 由 `turn` 是否存在区分。
- `response_trained` 防止共享响应重复训练。

为什么这样写：
- 树结构适合表示 sub-agent、分支和 compaction 后的 replay。
- 把训练信号挂在 assistant leaf 上，能自然区分 prompt history 与模型输出。

不变量与失败模式：
- 只有 assistant generated node 应持有 TurnRecord。
- response_trained 必须在 sample 构造时更新，否则 sibling 分支会重复计 loss。

Comment：
这是 agent trajectory 的数据模型核心。

### 2.2 _SampleBuilder 根据 token drift 决定 CLEAN/REALIGN/FORK

问题与约束：
- 客户端 replay 历史时，chat template 或 TITO round-trip 可能让 token ids 与已保存序列不完全一致；直接拼接会导致上下文断裂。

设计选择：
- `_SampleBuilder.classify_token_drift` 比较 held tokens 与下一 turn prompt 的公共前缀；无 drift 为 CLEAN，短 drift 且落在最近 response span 内为 REALIGN，否则 FORK。

Explain：
REALIGN 只修复最近响应内部的短漂移，且要求当前输出长度小于 fork threshold。更早或更大的 divergence 会开启新 builder，形成新的 Sample。

来源：slime/agent/trajectory.py L141-L191

Code：

```python
class _SampleBuilder:
    def __init__(self, fork_threshold: int) -> None:
        self._fork_threshold = fork_threshold
        self.tokens: list[int] = []
        self.loss_mask: list[int] = []
        self.logprobs: list[float] = []
        self.last_response_start_idx: int | None = None
        self.leading_prompt_len: int = 0

    def classify_token_drift(self, turn: TurnRecord) -> DriftKind:
        realign_at = _common_prefix_len(self.tokens, turn.prompt_ids)
        drift = len(self.tokens) - realign_at

        if drift == 0:
            return DriftKind.CLEAN

        start = self.last_response_start_idx
        if start is not None and realign_at >= start and len(turn.output_ids) < self._fork_threshold:
            return DriftKind.REALIGN
        return DriftKind.FORK
```

代码逻辑：
- common prefix 长度定位第一次 drift。
- drift 为 0 表示 prompt 延续当前 builder。
- drift 落在最近 response span 内且输出短，允许 realign。
- 其他情况 fork。

为什么这样写：
- 小的 response replay 漂移可以修复，不必拆样本。
- 早期上下文 drift 可能改变语义，只能 fork 保守处理。

不变量与失败模式：
- `last_response_start_idx` 为 None 时不能 realign。
- fork threshold 设置过大可能吸收本该分叉的长响应，设置过小会产生更多 Sample。

Comment：
drift 分类是 agent trajectory 处理真实客户端 replay 的关键机制。

### 2.3 append_turn 和 to_sample 生成 loss_mask/logprob 训练区间

问题与约束：
- Sample 需要包含完整上下文 tokens，但 loss 和 rollout_log_probs 只覆盖 response 区域。

设计选择：
- `append_turn` 先追加 prompt tail 为 loss_mask=0，再追加 generated output；trained=False 时 output 也以 loss_mask=0 重放。`to_sample` 去掉第一轮 prompt 前缀的 loss/logprob。

Explain：
REALIGN 会调用 `_align_to_prompt` 覆盖最近 response span，并把覆盖部分设为 loss_mask=0/logprob=0。第一轮 prompt 的长度保存在 `leading_prompt_len`，用于输出 Sample 时切分 response 区域。

来源：slime/agent/trajectory.py L193-L261

Code：

```python
def append_turn(self, turn: TurnRecord, kind: DriftKind, *, trained: bool = True) -> None:
    assert kind is not DriftKind.FORK, "append_turn called on a builder that would fork"

    is_first_turn = self.last_response_start_idx is None

    if kind is DriftKind.REALIGN:
        self._align_to_prompt(turn.prompt_ids)
    else:
        self._append_tokens(turn.prompt_ids[len(self.tokens) :], loss_mask=0)

    self.last_response_start_idx = len(self.tokens)
    self._append_tokens(
        turn.output_ids, loss_mask=int(trained), logprobs=turn.output_log_probs if trained else None
    )

    if is_first_turn:
        self.leading_prompt_len = len(turn.prompt_ids)

def _align_to_prompt(self, prompt_ids: list[int]) -> None:
    response_start = self.last_response_start_idx
    tail = prompt_ids[response_start:]
    self.tokens[response_start:] = tail
    self.loss_mask[response_start:] = [0] * len(tail)
    self.logprobs[response_start:] = [0.0] * len(tail)
```

```python
return Sample(
    index=base_sample.index,
    group_index=base_sample.group_index,
    rollout_id=base_sample.rollout_id if base_sample.rollout_id is not None else base_sample.index,
    prompt=base_sample.prompt,
    label=base_sample.label,
    tokens=tokens,
    response_length=len(loss_mask) - start,
    loss_mask=loss_mask[start:],
    rollout_log_probs=logprobs[start:],
    reward=0.0,
    status=Sample.Status.COMPLETED,
    metadata=md,
)
```

代码逻辑：
- prompt tail 永远不可训练。
- generated output 是否训练由 `trained` 控制。
- realign 覆盖旧 tail，并清空训练信号。
- Sample 的 loss_mask/logprob 从第一轮 prompt 后开始。

为什么这样写：
- 训练时需要上下文 token，但只对模型响应求 loss。
- sibling branch 重放已有 assistant 响应时，trained=False 可以避免重复训练。

不变量与失败模式：
- `output_log_probs` 长度必须与 `output_ids` 一致。
- `response_length` 要与截取后的 loss_mask/logprob 长度保持一致。

Comment：
这里把 tree 上的 turns 变成训练后端可消费的 `Sample`。

### 2.4 record_turn 和 get_trajectory 管理 per-session 树生命周期

问题与约束：
- 同一个 session 的多轮消息可能共享前缀、产生分支或被重放；结束时还要一次性消费并清理该 session。

设计选择：
- `record_turn` 找到 prompt 在树上的挂载点，处理 assistant rewrite merge，挂载剩余 prompt messages，再 attach generated assistant leaf。`get_trajectory` 遍历 leaves 线性化后 pop sid。

Explain：
`record_turn` 会 assert logprob 数量和 output id 数量一致。`get_trajectory` 为每个 routing leaf 调 `_chain_to_samples`，把同一个 reward 赋给所有 Sample，然后删除该 sid 的 tree 和 turn count。

来源：slime/agent/trajectory.py L269-L344

Code：

```python
class TrajectoryManager:
    def __init__(self, *, fork_threshold_tokens: int | None = None) -> None:
        self._fork_threshold: int = 1024 if fork_threshold_tokens is None else fork_threshold_tokens
        self._trees: dict[str, MessageNode] = {}
        self._turn_count: dict[str, int] = {}

    def record_turn(
        self,
        sid: str,
        *,
        turn: TurnRecord,
        prompt_messages: list[dict[str, Any]],
        response_message: dict[str, Any] | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not prompt_messages:
            logger.warning("record_turn(sid=%s): empty prompt_messages; skipping", sid)
            return
        assert not turn.output_log_probs or len(turn.output_log_probs) == len(turn.output_ids)

        root = self._trees.setdefault(sid, MessageNode())
        node, depth = self._find_mount_point(root, prompt_messages)
        node, depth = self._try_merge_assistant_rewrite(sid, node, prompt_messages, depth)
        node = self._mount_prompt_messages(node, prompt_messages[depth:])
        self._attach_assistant_leaf(sid, node, turn=turn, response_message=response_message, metadata=metadata)

    def get_trajectory(...):
        root = self._trees.get(sid)
        if root is None:
            return []
        samples: list[Sample] = []
        for routing_leaf in root.leaves():
            if routing_leaf.is_root:
                continue
            chain = routing_leaf.path_from_root()
            samples.extend(self._chain_to_samples(chain, base_sample=base_sample, extra_metadata=extra_metadata, max_sample_tokens=max_sample_tokens))
        for s in samples:
            s.reward = reward
        self._trees.pop(sid, None)
        self._turn_count.pop(sid, None)
        return samples
```

代码逻辑：
- 每个 sid 有一棵 root tree。
- prompt messages 中已存在的前缀会复用树节点。
- 新生成的 response 作为 assistant leaf 挂载。
- get 之后 session 被消费。

为什么这样写：
- 树比线性列表更适合处理 agent 分支和 prompt replay。
- 消费后删除 sid，避免同一 trajectory 被重复训练。

不变量与失败模式：
- prompt_messages 不能为空。
- `get_trajectory` 是 destructive read，调用方不能期望重复取回同一 session。

Comment：
TrajectoryManager 的生命周期是 record 多次，finish 时 get 一次。

### 2.5 _find_mount_point 用 role 和 dict equality 匹配历史路径

问题与约束：
- 客户端每轮会把历史 messages 重放回来；manager 需要判断哪些历史消息已经在树上，哪些是新分支。

设计选择：
- 从 root 开始逐层查找 `child.role == msg.role` 且 `child.message == msg` 的子节点；匹配到最深处后返回 node 和 depth。

Explain：
dict equality 是严格匹配，因此 tool call canonicalization、message content 展平和协议翻译必须保持稳定。未匹配的后缀会由 `_mount_prompt_messages` 新增。

来源：slime/agent/trajectory.py L352-L368

Code：

```python
def _find_mount_point(self, root: MessageNode, messages: list[dict[str, Any]]) -> tuple[MessageNode, int]:
    """Walk down the tree matching each message by role and dict equality (==),
    returning the deepest node that still matches and where to mount the rest."""
    node = root
    depth = 0
    while depth < len(messages):
        msg = messages[depth]
        next_child = None
        for child in node.children:
            if child.role == msg.get("role") and child.message == msg:
                next_child = child
                break
        if next_child is None:
            break
        node = next_child
        depth += 1
    return node, depth
```

代码逻辑：
- 每层只沿一个匹配 child 继续。
- 一旦找不到匹配就停止，后续 messages 视为新路径。
- 返回 depth 用于切分已匹配前缀与待挂载后缀。

为什么这样写：
- role+message equality 是协议无关的历史匹配规则。
- 不做 fuzzy match 可以避免把语义不同的 history 合并。

不变量与失败模式：
- message dict 必须 canonical；同义但结构不同会 fork。
- 同层多个完全相同 child 时会选择第一个，源码没有额外消歧。

Comment：
这解释了为什么 adapter 必须规范化 tool call 和 content。

### 2.6 _try_merge_assistant_rewrite 合并短 assistant rewrite

问题与约束：
- 客户端可能把上一轮 assistant 消息以轻微不同的 whitespace 或格式重放；严格 dict equality 会把它识别成新分支，导致废弃生成也产出训练样本。

设计选择：
- 当 mount point 下只有一个 assistant child，且该 child 是短的 generated leaf 时，把它降级为 routing-only 并用重放消息覆盖。

Explain：
函数要求 fork threshold 开启、当前待挂载消息是 assistant、只有一个 assistant child、child 没有子节点、child 有 turn 且 output 长度低于阈值。合并时记录 metadata，清空 `turn/turn_index`。

来源：slime/agent/trajectory.py L370-L426

Code：

```python
def _try_merge_assistant_rewrite(
    self,
    sid: str,
    node: MessageNode,
    prompt_messages: list[dict[str, Any]],
    depth: int,
) -> tuple[MessageNode, int]:
    if self._fork_threshold <= 0:
        return node, depth
    if depth >= len(prompt_messages) or prompt_messages[depth].get("role") != "assistant":
        return node, depth

    asst_children = [c for c in node.children if c.role == "assistant"]
    if len(asst_children) != 1:
        return node, depth

    rewritten_node = asst_children[0]
    if (
        rewritten_node.children
        or rewritten_node.turn is None
        or len(rewritten_node.turn.output_ids) >= self._fork_threshold
    ):
        return node, depth

    rewritten_node.metadata["merged_rewrite"] = {
        "abandoned_turn_index": rewritten_node.turn_index,
        "abandoned_response_tokens": len(rewritten_node.turn.output_ids),
    }
    rewritten_node.turn = None
    rewritten_node.turn_index = None
    rewritten_node.message = prompt_messages[depth]
    return rewritten_node, depth + 1
```

代码逻辑：
- 只处理 assistant rewrite。
- 多个 assistant child 时不猜测，直接 fork。
- 合并会丢弃原 TurnRecord 的训练信号。
- 覆盖 message 后 depth 前进一位。

为什么这样写：
- 短 assistant rewrite 多半是格式重放，不应该留下死分支训练样本。
- 长响应包含更多训练信号，保守 fork 比删除更安全。

不变量与失败模式：
- 合并是不可逆的，原 TurnRecord 被清空。
- fork threshold 配置决定多少 token 的废弃响应仍保留训练。

Comment：
这个优化是为了清理客户端 replay 造成的短 assistant 死分支。

### 2.7 _split_chain_into_builders 防止 shared assistant 重复训练

问题与约束：
- 树的多个 leaf 可能共享同一个 generated assistant 前缀；如果每条 leaf 都训练它，会重复计同一响应的 loss。

设计选择：
- `_split_chain_into_builders` 遍历 chain 中 generated assistant nodes；第一次遇到节点时 trained=True，之后 response_trained 已置 True，重放为 loss_mask=0。

Explain：
当前 builder 如果无法吸收下一个 turn 的 prompt drift，就新建 builder，相当于 fork 出一个新 Sample。否则按 CLEAN/REALIGN 追加到当前 builder。

来源：slime/agent/trajectory.py L456-L500

Code：

```python
def _split_chain_into_builders(self, chain: list[MessageNode]) -> list[_SampleBuilder]:
    asst_nodes = [n for n in chain if n.role == "assistant" and n.turn is not None]

    builders: list[_SampleBuilder] = []
    for asst_node in asst_nodes:
        trained = not asst_node.response_trained
        asst_node.response_trained = True

        if not builders or (kind := builders[-1].classify_token_drift(asst_node.turn)) is DriftKind.FORK:
            builders.append(_SampleBuilder(self._fork_threshold))
            builders[-1].append_turn(asst_node.turn, DriftKind.CLEAN, trained=trained)
        else:
            builders[-1].append_turn(asst_node.turn, kind, trained=trained)
    return builders

def _chain_to_samples(...):
    asst_nodes = [n for n in chain if n.role == "assistant" and n.turn is not None]
    truncated = bool(asst_nodes) and asst_nodes[-1].turn.finish_reason == "length"
    use_tool = any(bool((n.message or {}).get("tool_calls")) for n in asst_nodes)
    ill_formed = any(n.turn.ill_formed for n in asst_nodes)
    md = {
        **(extra_metadata or {}),
        "truncated": truncated,
        "use_tool": use_tool,
        "ill_formed": ill_formed,
    }
    return [
        builder.to_sample(base_sample, md, max_sample_tokens)
        for builder in self._split_chain_into_builders(chain)
    ]
```

代码逻辑：
- 只处理有 TurnRecord 的 assistant nodes。
- response_trained 是节点级去重标记。
- drift fork 会开启新 builder。
- metadata 汇总 truncated/use_tool/ill_formed。

为什么这样写：
- 一个生成响应只能训练一次，但可以作为多个后续分支的上下文。
- drift fork 和 tree branch 都最终转化为一个或多个 Sample。

不变量与失败模式：
- response_trained 在一次 get_trajectory 内被修改；重复调用已被 get_trajectory 清理 sid 阻止。
- chain 中 routing-only assistant 不会进入训练 builder。

Comment：
这是树到线性 Sample 的核心转换。

---

## 3. 协议适配层

### 3.1 OpenAIAdapter 只实现 Chat Completions 路由与 wire hook

问题与约束：
- OpenAI-compatible 客户端通过 `/v1/chat/completions` 驱动 agent；adapter 需要转换 wire request，但不应重写 turn pipeline。

设计选择：
- `OpenAIAdapter` 继承 `BaseAdapter`，只注册 `/v1/chat/completions`，并实现 session id、translate、reply build、respond 等 hook。

Explain：
类 docstring 说明 Responses API 不在范围内。`_respond` 根据 stream 选择 SSE render 或 JSON render，turn 记录仍由 BaseAdapter `_run_turn` 完成。

来源：slime/agent/adapters/openai.py L38-L73

Code：

```python
class OpenAIAdapter(BaseAdapter):
    """OpenAI Chat-Completions-compatible HTTP adapter: wire translation and
    reply framing only; the turn machinery is inherited from BaseAdapter."""

    logger = logger
    log_prefix = "openai_adapter"
    max_token_keys = ("max_completion_tokens", "max_tokens", "max_output_tokens")
    stop_keys = ("stop",)

    def _register_routes(self, app: web.Application) -> None:
        app.router.add_post("/v1/chat/completions", self._run_turn)

    def _session_id(self, request: web.Request, body: dict) -> str:
        return _request_session_id(request, body)

    def _translate(self, body: dict) -> tuple[list[dict], list[dict] | None]:
        messages = body.get("messages") or []
        if not isinstance(messages, list):
            raise web.HTTPBadRequest(text="messages must be a list")
        translated = _translate_messages(messages)
        tools_schema = _tools_to_chat_tools(body.get("tools"))
        return translated, tools_schema

    async def _respond(self, request, body, reply, in_tok, out_tok, stream) -> web.StreamResponse:
        wire_message, wire_finish = reply.wire
        if stream:
            return await _render_stream(request, body, wire_message, wire_finish, in_tok, out_tok)
        return web.json_response(_render_response(body, wire_message, wire_finish, in_tok, out_tok))
```

代码逻辑：
- route 绑定到 inherited `_run_turn`。
- max token 和 stop 字段按 OpenAI 协议命名。
- translate 把 wire messages/tools 转成 chat-template messages/tools schema。

为什么这样写：
- 协议差异集中在 adapter hook，trajectory 逻辑复用 BaseAdapter。
- Chat Completions 足够支持常见 OpenAI-compatible agent 客户端。

不变量与失败模式：
- 请求体 `messages` 必须是 list。
- Responses API 不支持，客户端若调用 `/v1/responses` 不会进入此 adapter。

Comment：
OpenAIAdapter 是协议薄层，核心训练数据逻辑不在这里。

### 3.2 AnthropicAdapter 注册 /v1/messages 和 count_tokens

问题与约束：
- Anthropic Messages 协议有 system、messages、tool blocks 和 count_tokens；但 turn pipeline 仍应与 OpenAI adapter 共享。

设计选择：
- `AnthropicAdapter` 注册 `/v1/messages` 到 `_run_turn`，另注册 `/v1/messages/count_tokens`；在 `_preprocess_body` 中折叠 mid-list system。

Explain：
Anthropic adapter 设置 `max_token_keys=("max_tokens",)`、`stop_keys=("stop_sequences",)`。wire translation 将 system 和 messages 转成 chat-template messages，reply framing 返回 Anthropic block/stop_reason。

来源：slime/agent/adapters/anthropic.py L39-L75

Code：

```python
class AnthropicAdapter(BaseAdapter):
    """Anthropic Messages-compatible HTTP adapter: wire translation and reply
    framing only; the turn machinery is inherited from BaseAdapter."""

    logger = logger
    log_prefix = "anthropic_adapter"
    max_token_keys = ("max_tokens",)
    stop_keys = ("stop_sequences",)

    def _register_routes(self, app: web.Application) -> None:
        app.router.add_post("/v1/messages", self._run_turn)
        app.router.add_post("/v1/messages/count_tokens", _count_tokens)

    def _session_id(self, request: web.Request, body: dict) -> str:
        return _request_session_id(request)

    def _preprocess_body(self, body: dict) -> None:
        _fold_mid_list_system_into_user(body)

    def _translate(self, body: dict) -> tuple[list[dict], list[dict] | None]:
        translated = _translate_messages(body.get("messages") or [], body.get("system"))
        tools_schema = _tools_to_chat_tools(body.get("tools"))
        return translated, tools_schema

    async def _respond(self, request, body, reply, in_tok, out_tok, stream) -> web.StreamResponse:
        blocks, stop_reason = reply.wire
        if stream:
            return await _render_stream(request, blocks, stop_reason, in_tok, out_tok)
        return web.json_response(_render_response(body, blocks, stop_reason, in_tok, out_tok))
```

代码逻辑：
- `/v1/messages` 使用共享 turn pipeline。
- count_tokens 单独提供轻量 endpoint。
- stop/max token 字段采用 Anthropic 命名。
- `_preprocess_body` 在 shared pipeline 读取 sid/translate 前执行。

为什么这样写：
- Anthropic wire 结构和 OpenAI 不同，但训练轨迹仍然是同一类 prompt/output token 序列。
- count_tokens 不应记录 trajectory，因此单独 route。

不变量与失败模式：
- session id 只从 request 解析，不从 body 解析。
- mid-list system 折叠必须在 `_translate` 前完成。

Comment：
两个协议 adapter 的差异停留在 wire 层，TrajectoryManager 完全复用。
