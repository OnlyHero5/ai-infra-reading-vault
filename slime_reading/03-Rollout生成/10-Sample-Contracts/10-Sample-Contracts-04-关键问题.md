---
type: batch-doc
module: 10-Sample-Contracts
batch: "10"
doc_type: faq
title: "Sample 契约 · 关键问题"
tags:
  - slime/batch/10
  - slime/module/sample-contracts
  - slime/doc/faq
updated: 2026-07-02
---

# Sample 契约 · 关键问题

---

## Q1：rollout_id 与 index 何时必须显式设置？

**Explain：** compact / subagent 路径下 **一次 rollout  execution 产生多个 training sample**； siblings 必须共享同一 `rollout_id`，否则 loss 聚合会把同一 rollout 重复计数。

**Code：**

```python
# 来源：slime/utils/types.py L99-L105
# 提交版本：22cdc6e1
# Compact / subagent paths ... should set the same rollout_id on every sibling,
# so loss aggregation averages within the rollout instead of over-counting it.
rollout_id: int | None = None
```

---

## Q2：loss_mask 全 0 的 sample 会进训练吗？

**Explain：** `effective_response_length==0` 时通常被 filter 或 dynamic batch 跳过；若进入训练，loss 项为 0。应在上游设 `remove_sample=True`。

**Code：**

```python
# 来源：slime/utils/types.py L249-L251
# 提交版本：22cdc6e1
@property
def effective_response_length(self):
    return sum(self.loss_mask) if self.loss_mask is not None else self.response_length
```

---

## Q3：trainable token 为何必须带 log_probs？

**Explain：** PPO/GRPO ratio 需要 rollout 引擎侧的 old log prob；缺失时 `append_response_tokens` 直接 raise。

**Code：**

```python
# 来源：slime/utils/types.py L276-L277
# 提交版本：22cdc6e1
if tokens and trainable and log_probs is None:
    raise ValueError("trainable response tokens require rollout log probabilities.")
```

---

## Q4：top-p replay 只有 ids 没有 offsets 会怎样？

**Explain：** `_validate_response_metadata_lengths` 与 extract 函数均要求 **成对出现**；缺一即 ValueError。

**Code：**

```python
# 来源：slime/utils/types.py L392-L395
# 提交版本：22cdc6e1
if self.rollout_top_p_token_ids is None or self.rollout_top_p_token_offsets is None:
    raise ValueError("rollout top-p replay must include both token ids and offsets.")
```

---

## Q5：legacy rollout 返回值还能用吗？

**Explain：** **可以**。`call_rollout_fn` 自动包装为 `RolloutFnTrainOutput(samples=output)`。

**Code：**

```python
# 来源：slime/rollout/base_types.py L22-L24
# 提交版本：22cdc6e1
if not isinstance(output, (RolloutFnTrainOutput, RolloutFnEvalOutput)):
    output = RolloutFnEvalOutput(data=output) if evaluation else RolloutFnTrainOutput(samples=output)
```

---

## Q6：reward 为 dict 时如何取标量？

**Explain：** 通过 `args.reward_key` 索引；未设置 reward_key 时 dict 不能直接用于 float 运算——需配置 key。

**Code：**

```python
# 来源：slime/utils/types.py L246-L247
# 提交版本：22cdc6e1
def get_reward_value(self, args) -> float:
    return self.reward if not args.reward_key else self.reward[args.reward_key]
```

---

## Q7：load_function 路径写错会怎样？

**Explain：** `importlib.import_module` 或 `getattr` 失败时 **直接异常终止**；无 silent fallback。

**Code：**

```python
# 来源：slime/utils/misc.py L43-L45
# 提交版本：22cdc6e1
module = importlib.import_module(module_path)
return getattr(module, attr)
```

**Comment：**

- 路径格式：`package.module.function_name`
- 函数需模块顶层可 import

---

## Q8：FAILED vs ABORTED 如何选用？

**Explain：** ABORTED 来自 SGLang `finish_reason.type=="abort"`（用户取消/引擎中止）；FAILED 由 rollout 逻辑显式设置，表示可恢复错误且可能有部分输出。

**Code：**

```python
# 来源：slime/utils/types.py L135-L138
# 提交版本：22cdc6e1
# FAILED samples may still contain partial valid output and can be retried
FAILED = "failed"
```
