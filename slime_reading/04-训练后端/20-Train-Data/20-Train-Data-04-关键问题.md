---
type: batch-doc
module: 20-Train-Data
batch: "20"
doc_type: faq
title: "Train Data · 关键问题"
tags:
  - slime/batch/20
  - slime/module/train-data
  - slime/doc/faq
updated: 2026-07-02
---

# Train Data · 关键问题

---

## Q1：为什么 pack-first 而不是先按 DP 切样本？

**Explain：** Pipeline Parallel 要求所有 DP rank 在同一 forward 步执行 **相同数量的 micro-batch**。若先按 rank 切样本再各自 pack，各 rank mbs 数易不一致。全局 pack 后再 stride/KK 分配 mbs，保证 `K % dp_size == 0`。

---

## Q2：static micro_batch_size 报 alignment AssertionError？

**原因：** `step_size % (dp_size * micro_batch_size * mb_group) != 0` 时 static 路径 **拒绝** split mbs（会破坏固定 batch 语义）。

**处理：**

- 调整 `global_batch_size` 或 `n_samples_per_prompt` 使每 step 样本数可整除
- 或改用 `use_dynamic_batch_size=True`

**Code：**

```python
## 来源：slime/utils/dp_schedule.py L177-L185
                raise AssertionError(
                    f"static path: num_mbs ({len(step_mbs)}) is not a multiple of "
                    f"dp_size * mb_group ({align_to}); ..."
                )
```

---

## Q3：超长单样本会 OOM 吗？

**Explain：** `first_fit_pack` 允许单个 length > `max_tokens_per_bin` 的样本独占一 mbs；该 mbs 是唯一可超 cap 的情况。若仍 OOM，需降 `max_tokens_per_gpu` 或过滤长样本。

---

## Q4：`balance_by_flops` 与 token cap 冲突？

**Explain：** FLOPs 分区 **不 enforce** `max_tokens_per_bin`（见 `_pack_step_into_mbs` 注释）。MoE/长 prompt 场景下 FLOPs 与 token 数相关性弱，可能 pack 出超大 mbs。

**建议：** 仅在 token 分布较均匀且 cap 留余量时开启；否则用默认 first-fit。

---

## Q5：`partition` pop 后为何只重排 `total_lengths`？

**Explain：** `micro_batch_indices` 已是 **rank-local** 下标（相对 `partitions[r]` 顺序）；`tokens` 等 list 在 RolloutManager 打包时已按 partition 子集拷贝或 Actor 侧按同一顺序存储。当前实现假定 **其他 list 字段与 partition 顺序一致**（见 rollout 打包逻辑）。

---

## Q6：`allgather_cp` 与默认 CP 怎么选？

| 模式 | 适用 | 特点 |
|------|------|------|
| zigzag（默认） | 常规 CP | 每样本 `slice_with_cp` |
| allgather_cp | DSA 等 | 全局 cat 再 chunk；loss 需 `0 * logits.sum()` 保梯度 |

见 [[23-CP-RoutingReplay-04-关键问题]]。

---

## Q7：rollout 指标与 train loss 数值对不上？

**检查：**

1. `log_rollout_data` 是否用 `(sum, count)` 而非简单 mean
2. `rollout_mask_sums` 是否在 Rollout 侧预计算
3. uneven DP 下 legacy mean 会偏——应用 `rollout_log_metric_contribution`

---

## Q8：VPP > 1 时 iterator 为何要复制多份？

**Explain：** Megatron VPP 每个 virtual stage 独立调用 `get_next`；共享同一 `micro_batch_indices` 表但 **独立 offset**，避免 stage 间抢同一游标。

---

## Q9：尾部 rollout 被丢弃会有什么问题？

**Explain：** `num_steps = floor(num_rollouts / global_batch_size)`，余数 rollout **不参与训练**。若 `rollout_batch_size` 不是 `global_batch_size` 整数倍，有效吞吐下降——配置时需对齐。

---

## Q10：相关测试

- `tests/test_dp_schedule.py` — 不变量断言
- `tests/test_seqlen_balancing.py` — KK / first-fit
- `tests/test_loss_cp_invariance.py` — 与 [[23-CP-RoutingReplay-00-MOC]] 交叉
