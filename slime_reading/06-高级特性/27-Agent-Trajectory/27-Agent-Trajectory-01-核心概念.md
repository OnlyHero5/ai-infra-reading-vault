---
type: batch-doc
module: 27-Agent-Trajectory
batch: "27"
doc_type: concept
title: "Agent Trajectory · 核心概念"
tags:
  - slime/batch/27
  - slime/module/agent-trajectory
  - slime/doc/concept
updated: 2026-07-02
---

# Agent Trajectory · 核心概念

## 1. 为什么需要 TrajectoryManager？

单轮 RL 里一个 prompt 对应一次 `/generate` → 一个 `Sample`。Agent 场景下：

- 同一 prompt 可能 **多轮** 调用模型（tool → observation → 再生成）
- 可能出现 **subagent 分支**、context compaction 导致 prompt 前缀变化
- Chat template 重渲染会导致 **token 漂移**（TITO：token-in-token-out 不对齐）

`TrajectoryManager` 用 **消息树 + drift 分类** 把这些情况统一收成可训练的 `Sample` 列表。

---

## 2. TurnRecord：Adapter 与 Manager 的契约

**Explain：** 每轮 SGLang 返回的 prompt/output token ids、logprobs、finish_reason 打包为不可变 dataclass。

**Code：**

```python
# 来源：slime/agent/trajectory.py L28-L38
@dataclasses.dataclass(frozen=True)
class TurnRecord:
    prompt_ids: list[int]
    output_ids: list[int]
    finish_reason: str
    output_log_probs: list[float] = dataclasses.field(default_factory=list)
    ill_formed: bool = False
```

**Comment：**

- `prompt_ids` 由 adapter 用 `tokenizer.apply_chat_template` 渲染
- `output_ids` 来自 SGLang `output_token_logprobs`（**禁止** decode 再 re-tokenize）
- `ill_formed` 由 `parse_model_output` 设置（工具参数 JSON 解析失败等）

---

## 3. MessageNode：routing 树节点

**Explain：** 两类节点：

| 类型 | `turn` 字段 | 含义 |
|------|-------------|------|
| generated | 非 None | 模型本轮真实输出，参与 linearize |
| routing-only | None | system/user/tool 或 foreign assistant，仅路由 |

**Code：**

```python
# 来源：slime/agent/trajectory.py L65-L82
class MessageNode:
    def __init__(self, *, role=None, message=None, metadata=None, parent=None):
        ...
        self.turn: TurnRecord | None = None
        self.turn_index: int | None = None
        self.response_trained: bool = False
```

**Comment：**

- `response_trained` 防止 sibling leaf 重复训练同一 assistant 响应
- `_find_mount_point` 用 `role` + `message ==` 字典相等匹配历史

---

## 4. DriftKind：token 漂移处理

**Explain：** `_SampleBuilder.classify_token_drift` 比较已持有 tokens 与新 turn 的 `prompt_ids` 前缀。

| 种类 | 条件 | 行为 |
|------|------|------|
| CLEAN | 无漂移 | 追加 prompt tail |
| REALIGN | 漂移在最近 response 内且 output 短 | 覆盖 response 段为 loss_mask=0 |
| FORK | 漂移过大或过早 | 关闭当前 builder，新开 Sample |

**Code：**

```python
# 来源：slime/agent/trajectory.py L130-L134
class DriftKind(enum.Enum):
    CLEAN = "clean"
    REALIGN = "realign"
    FORK = "fork"
```

```python
# 来源：slime/agent/trajectory.py L169-L191
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

**Comment：** 默认 `fork_threshold=1024` tokens，可通过 adapter 构造参数覆盖。

---

## 5. Sample 输出形态

**Explain：** `to_sample` 去掉首轮 prompt 前缀，`loss_mask` / `rollout_log_probs` 只覆盖 response 区域。

**Code：**

```python
# 来源：slime/agent/trajectory.py L248-L261
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

**Comment：** `get_trajectory` 最后统一写 `reward`；adapter 的 `finish_session` 再 decode `response` 字符串。

---

## 6. BaseAdapter：协议无关的一轮 pipeline

**Explain：** Anthropic / OpenAI adapter 继承 `BaseAdapter`，共享 `_run_turn`：translate → SGLang → parse → respond → record_turn。

**Code：**

```python
# 来源：slime/agent/adapters/common.py L341-L344
            translated, tools_schema = self._translate(body)
            prompt_ids = _render_token_ids(translated, tok, tools=tools_schema, add_generation_prompt=True)
            turn = await call_sglang_generate(prompt_ids, s, body, adapter=self, session_id=sid)
```

**Comment：**

- `X-SMG-Routing-Key: session_id` 让 SGLang router 做 session affinity（前缀缓存）
- 响应 flush 失败则 **不** record_turn

---

## 7. Agent 文档：推荐集成模式

**Explain：** 官方 roadmap 建议大多数 agent 任务从 `--custom-generate-function-path` 起步。

**Code（文档摘录）：**

```python
# 来源：docs/en/get_started/agent.md L40-L52
from slime.agent.adapters import AnthropicAdapter

adapter = AnthropicAdapter(
    tokenizer=tokenizer,
    sglang_url=sglang_url,
    tool_parser=tool_parser,
    reasoning_parser=reasoning_parser,
)
adapter.open_session(session_id, sampling_defaults=sampling_params)
segments = await adapter.finish_session(session_id)
```

**Comment：** 多 segment 训练需共享 `rollout_id`（见批次 28 customization 文档）。

---

## 8. OpenAI vs Anthropic Adapter

| 维度 | OpenAIAdapter | AnthropicAdapter |
|------|---------------|------------------|
| 路由 | `/v1/chat/completions` | `/v1/messages` |
| session id | Bearer / metadata.session_id | Bearer |
| 特殊处理 | tool_calls arguments → dict | system fold、tool_use 块翻译 |
| max tokens key | max_completion_tokens 等 | max_tokens |

二者均 **不** 重新 tokenize 模型输出文本。
