---
type: batch-doc
module: 23-CP-RoutingReplay
batch: "23"
doc_type: concept
title: "CP · Routing Replay · 核心概念"
tags:
  - slime/batch/23
  - slime/module/cp-routing-replay
  - slime/doc/concept
updated: 2026-07-02
---

# CP · Routing Replay · 核心概念

---

## 1. Context Parallel zigzag 布局

**Explain：** 序列 pad 到 `2 * cp_size * chunk_size`，rank `r` 取 chunk `r` 与 chunk `2*cp_size-r-1`（首尾对称），使 attention 负载均衡。

**Code：**

```python
# 来源：cp_utils.py L307-L317
    chunk_size = (token_len + 2 * cp_size - 1) // (2 * cp_size)
    pad = 2 * cp_size * chunk_size - token_len
    ...
    start_1, end_1 = chunk_size * cp_rank, chunk_size * (cp_rank + 1)
    start_2, end_2 = chunk_size * (2 * cp_size - cp_rank - 1), chunk_size * (2 * cp_size - cp_rank)
    return torch.cat([tokens[start_1:end_1], tokens[start_2:end_2]])
```

---

## 2. Logits / token 偏移

**Explain：** logprob 对应 logits 位置 `[prompt_length-1, total_length-1)`；CP 下每 rank 持有一段 **不连续** logits，需 `get_logits_and_tokens_offset_with_cp` 映射。

| 量 | 含义 |
|----|------|
| `chunk_0`, `chunk_1` | token 空间两段 |
| `logits_0`, `logits_1` | 有效 logits 区间（比 token 少 1） |
| `token_0`, `token_1` | 用于 gather target token |

---

## 3. all_gather_with_cp

**Explain：** 各 rank 用 zero padding 拼成 `[response_length]` 形状，再 `all_reduce`（非 all_gather）合并——因每 rank 只填自己负责的 logits 区间。

**用途：** GAE/REINFORCE++（ppo_utils）、GSPO/OPSM（policy loss）、`get_values` 重分布。

---

## 4. get_sum_of_sample_mean

**Explain：** CP>1 时先把 `loss_mask` 按 `tokens_offset` 切成两段再 cat，与 local logprob 长度对齐；分母 `sample_denoms` 仍用 **full** rollout 预计算值（[[20-Train-Data]]）。

---

## 5. reduce_train_step_metrics

| 模式 | 分母 | cp_factor |
|------|------|-----------|
| per-token loss | all-reduced num_tokens | cp_size |
| per-rollout mean | `step_global_batch_size` 常数 | 1 |

---

## 6. Routing Replay 动机

**Explain：** MoE 训练时 rollout 与 train 若 expert routing 不一致，off-policy 偏差大。Slime 可：

- **record**：train forward 记录 `top_indices`
- **replay_forward / replay_backward**：用记录的 indices 做 `scores.gather`
- **fallthrough**：正常 top-k（ref/teacher）

Rollout 侧 `--use-rollout-routing-replay` 把引擎记录的 `rollout_routed_experts` 灌入 `RoutingReplay`（见 actor `fill_routing_replay`）。

---

## 7. 环境变量状态机

| `ROUTING_REPLAY_STAGE` | 行为 |
|------------------------|------|
| `record` | forward 时 record top_indices |
| `replay_forward` | pop_forward |
| `replay_backward` | pop_backward（policy train） |
| `fallthrough` | 原始 top-k |

全局开关：`ENABLE_ROUTING_REPLAY=1`（Ray actor group 注入）。

---

## 8. Megatron 集成

**Explain：** `docker/patch/*/megatron.patch` 在 MoE layer 注入 `get_routing_replay_compute_topk` 包装 `compute_topk`；`register_routing_replay` 在 module forward pre_hook 设置 thread-local `ROUTING_REPLAY`。

---

## 9. 与 allgather_cp（DSA）的关系

**Explain：** DSA 路径在 `get_batch` 先 cat 再 chunk；CP utils 仍用于 logprob 切片与 metric。loss 侧需 `0 * logits.sum()` 保证空 rank 参与 backward（[[22-Loss-Policy-01-核心概念]]）。
