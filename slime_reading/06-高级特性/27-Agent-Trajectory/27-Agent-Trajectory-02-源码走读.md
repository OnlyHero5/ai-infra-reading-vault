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
updated: 2026-07-02
---

# Agent Trajectory · 源码走读

## 走读顺序

1. `adapters/common.py` — `_run_turn` 与 `call_sglang_generate`
2. `trajectory.py` — `record_turn` 建树
3. `trajectory.py` — `_SampleBuilder` 与 drift
4. `trajectory.py` — `get_trajectory` 线性化
5. `adapters/openai.py` / `anthropic.py` — 协议钩子

---

## 1. Adapter 构造：共享 TrajectoryManager

**Code：**

```python
# 来源：slime/agent/adapters/common.py L161-L166
        mgr_kwargs: dict[str, int] = {}
        if fork_threshold_tokens is not None:
            mgr_kwargs["fork_threshold_tokens"] = fork_threshold_tokens
        self.manager = TrajectoryManager(**mgr_kwargs)
```

**Comment：** 所有 session 共用一个 manager；sid 隔离树。

---

## 2. call_sglang_generate：TurnRecord 来源

**Code：**

```python
# 来源：slime/agent/adapters/common.py L475-L517
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
            ...
        output_ids = [x[1] for x in output_token_logprobs]
        output_log_probs = [float(x[0]) for x in output_token_logprobs]
        finish = (meta.get("finish_reason") or {}).get("type", "stop") or "stop"
    ...
    return TurnRecord(
        prompt_ids=list(prompt_ids),
        output_ids=output_ids,
        finish_reason=finish,
        output_log_probs=output_log_probs,
    )
```

**Comment：**

- `return_logprob=True` 是 agent 训练硬要求
- 取消/超时主动 `abort_request` 释放 KV

---

## 3. _run_turn：record_turn 时机

**Code：**

```python
# 来源：slime/agent/adapters/common.py L362-L391
            try:
                response = await self._respond(request, body, reply, in_tok, out_tok, stream)
            except (ConnectionResetError, asyncio.CancelledError) as e:
                ...
                return web.Response(status=499, text="client disconnected")

            self.manager.record_turn(
                sid,
                turn=turn,
                prompt_messages=translated,
                response_message=reply.manager_message,
                metadata={"sid": sid},
            )
            return response
```

**Comment：** parse 后可能 `dataclasses.replace(turn, ill_formed=...)` 更新 turn。

---

## 4. record_turn：挂载流程

**Code：**

```python
# 来源：slime/agent/trajectory.py L283-L305
    def record_turn(self, sid, *, turn, prompt_messages, response_message, metadata=None):
        if not prompt_messages:
            logger.warning("record_turn(sid=%s): empty prompt_messages; skipping", sid)
            return
        root = self._trees.setdefault(sid, MessageNode())
        node, depth = self._find_mount_point(root, prompt_messages)
        node, depth = self._try_merge_assistant_rewrite(sid, node, prompt_messages, depth)
        node = self._mount_prompt_messages(node, prompt_messages[depth:])
        self._attach_assistant_leaf(sid, node, turn=turn, response_message=response_message, metadata=metadata)
```

**Comment：** `_find_mount_point` 按 dict 相等匹配已有路径，未匹配部分由 `_mount_prompt_messages` 追加。

---

## 5. assistant rewrite merge

**Explain：** 客户端重放 assistant 消息时 whitespace 可能变；若唯一 assistant 子节点是短 response leaf，则 in-place 合并而非 fork。

**Code：**

```python
# 来源：slime/agent/trajectory.py L411-L426
        rewritten_node = asst_children[0]
        if (
            rewritten_node.children
            or rewritten_node.turn is None
            or len(rewritten_node.turn.output_ids) >= self._fork_threshold
        ):
            return node, depth
        rewritten_node.turn = None
        rewritten_node.turn_index = None
        rewritten_node.message = prompt_messages[depth]
        return rewritten_node, depth + 1
```

**Comment：** merge 会 **丢弃** 原 TurnRecord 的训练信号（变为 routing-only）。

---

## 6. _split_chain_into_builders

**Code：**

```python
# 来源：slime/agent/trajectory.py L465-L477
        for asst_node in asst_nodes:
            trained = not asst_node.response_trained
            asst_node.response_trained = True
            if not builders or (kind := builders[-1].classify_token_drift(asst_node.turn)) is DriftKind.FORK:
                builders.append(_SampleBuilder(self._fork_threshold))
                builders[-1].append_turn(asst_node.turn, DriftKind.CLEAN, trained=trained)
            else:
                builders[-1].append_turn(asst_node.turn, kind, trained=trained)
```

**Comment：** 第一个到达 shared assistant 的 leaf `trained=True`；后续 sibling 以 loss_mask=0 重放。

---

## 7. append_turn：CLEAN vs REALIGN

**Code：**

```python
# 来源：slime/agent/trajectory.py L201-L211
        if kind is DriftKind.REALIGN:
            self._align_to_prompt(turn.prompt_ids)
        else:
            self._append_tokens(turn.prompt_ids[len(self.tokens) :], loss_mask=0)
        self.last_response_start_idx = len(self.tokens)
        self._append_tokens(
            turn.output_ids, loss_mask=int(trained), logprobs=turn.output_log_probs if trained else None
        )
```

**Comment：** REALIGN 用 prompt 覆盖 drifted response span，全部标 loss_mask=0。

---

## 8. get_trajectory：消费 session

**Code：**

```python
# 来源：slime/agent/trajectory.py L324-L344
        root = self._trees.get(sid)
        if root is None:
            return []
        samples: list[Sample] = []
        for routing_leaf in root.leaves():
            if routing_leaf.is_root:
                continue
            chain = routing_leaf.path_from_root()
            samples.extend(self._chain_to_samples(chain, base_sample=base_sample, ...))
        for s in samples:
            s.reward = reward
        self._trees.pop(sid, None)
        self._turn_count.pop(sid, None)
        return samples
```

**Comment：** metadata 注入 `truncated` / `use_tool` / `ill_formed` 标志。

---

## 9. finish_session：decode response

**Code：**

```python
# 来源：slime/agent/adapters/common.py L261-L276
        samples = self.manager.get_trajectory(sid, base_sample=base_sample, reward=reward, ...)
        for s in samples:
            rlen = int(s.response_length or 0)
            s.response = (
                self.tokenizer.decode(s.tokens[-rlen:], skip_special_tokens=False) if rlen and s.tokens else ""
            )
        return samples
```

**Comment：** manager 故意 tokenizer-free；decode 在 adapter 层完成。

---

## 10. OpenAIAdapter 路由注册

**Code：**

```python
# 来源：slime/agent/adapters/openai.py L47-L48
    def _register_routes(self, app: web.Application) -> None:
        app.router.add_post("/v1/chat/completions", self._run_turn)
```

**Comment：** Responses API (`/v1/responses`)  intentionally 未实现。

---

## 11. AnthropicAdapter 路由

**Code：**

```python
# 来源：slime/agent/adapters/anthropic.py L48-L50
    def _register_routes(self, app: web.Application) -> None:
        app.router.add_post("/v1/messages", self._run_turn)
        app.router.add_post("/v1/messages/count_tokens", _count_tokens)
```

**Comment：** `_preprocess_body` 折叠 mid-list system 块到 user。

---

## 12. tool_call_dict：history 匹配不变量

**Code：**

```python
# 来源：slime/agent/adapters/common.py L110-L117
def tool_call_dict(name: str, arguments: dict | None) -> dict:
    return {"type": "function", "function": {"name": name, "arguments": arguments or {}}}
```

**Comment：** arguments 必须是 dict 而非 JSON 字符串，否则 replay 时 `==` 失败导致 fork。
