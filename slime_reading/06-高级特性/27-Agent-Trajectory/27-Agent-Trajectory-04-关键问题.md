---
type: batch-doc
module: 27-Agent-Trajectory
batch: "27"
doc_type: faq
title: "Agent Trajectory · 关键问题"
tags:
  - slime/batch/27
  - slime/module/agent-trajectory
  - slime/doc/faq
updated: 2026-07-02
---

# Agent Trajectory · 关键问题

## Q1：为什么不能 decode 再 tokenize 模型输出？

SGLang 返回的 token ids 与训练 forward 必须 **bit 一致**。decode→re-tokenize 会因 template、空格、special token 产生漂移，loss 对不齐。

**正确做法：**

```python
## 来源：slime/agent/adapters/common.py L498-L500
        output_ids = [x[1] for x in output_token_logprobs]
        output_log_probs = [float(x[0]) for x in output_token_logprobs]
```

**易错：** 在 custom generate 里对 `output["text"]` 做字符串 postprocess 后再 tokenizer()——Search-R1 文档明确禁止（见[[29-Plugins-Examples-00-MOC]]）。

---

## Q2：TITO 漂移何时 FORK vs REALIGN？

- **REALIGN**：漂移点落在 **最后一轮 response 内部**，且新 output 长度 < `fork_threshold`
- **FORK**：漂移出现在 prompt/更早 turn，或 output 太长无法吸收

调低 `fork_threshold_tokens` → 更多 FORK → 更多 Sample、更短上下文链。

---

## Q3：同一 assistant 响应为何只训练一次？

Sibling leaf 共享前缀路径时，第一个 leaf 设 `response_trained=True`，后续 leaf 以 `trained=False` 重放（loss_mask=0）。

```python
## 来源：slime/agent/trajectory.py L469-L470
            trained = not asst_node.response_trained
            asst_node.response_trained = True
```

---

## Q4：OpenAI tool_calls 为何要转成 dict arguments？

Manager 用 `child.message == msg` 匹配历史。JSON 字符串 key 顺序不同会导致假 fork。

```python
## 来源：slime/agent/adapters/openai.py L108-L111
      * tool_calls[i].function.arguments is a dict (not a JSON string): the chat
        template needs a mapping, and the manager matches history by dict
        equality regardless of key order.
```

---

## Q5：客户端断连会怎样？

`_respond` 抛 `ConnectionResetError` → 返回 499，**不** record_turn。避免训练客户端未收到的幻觉 turn。

---

## Q6：max_turns_per_sid 做什么？

Adapter 侧硬上限，超限返回 HTTP 429 JSON error，防止 runaway agent loop。

```python
## 来源：slime/agent/adapters/common.py L294-L304
        if prior >= cap:
            return web.json_response({"error": {...}}, status=429)
```

与 TrajectoryManager 无关，是 **Serving 层保险丝**。

---

## Q7：何时用 Adapter vs 手写 multi-turn generate？

| 方式 | 适用 |
|------|------|
| Adapter | 已有 OpenAI/Anthropic SDK agent（Claude Code、Codex CLI） |
| 手写 `custom_generate` | 简单 tool loop（Search-R1）、完全自定义协议 |
| `--rollout-function-path` | 跨 sample 调度、fully-async 队列 |

---

## Q8：get_trajectory 为何 pop session？

防止内存泄漏；语义是 **一次性消费**。二次 `finish_session` 返回 `[]`——custom generate 应缓存 samples。

---

## Q9：测试在哪里？

`tests/test_agent/` 覆盖 drift、merge、multi-leaf。本地：

```bash
python -m pytest tests/test_agent/ -q
```

---

## Q10：与 PD disaggregation 的关系？

长上下文 agent 推荐 PD 分离 + `--sglang-config` 调优；Trajectory 层无 PD 感知，纯数据结构与 adapter HTTP。
