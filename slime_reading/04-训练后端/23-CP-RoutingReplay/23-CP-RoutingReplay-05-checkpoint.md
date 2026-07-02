---
type: batch-doc
module: 23-CP-RoutingReplay
batch: "23"
doc_type: checkpoint
title: "CP · Routing Replay · 验收清单"
tags:
  - slime/batch/23
  - slime/module/cp-routing-replay
  - slime/doc/checkpoint
updated: 2026-07-02
---

# CP · Routing Replay · 验收清单

## 读者自测

- [ ] 能解释 zigzag CP 双 chunk 布局
- [ ] 能说明 `all_gather_with_cp` 的 padding + all_reduce 逻辑
- [ ] 能描述 `get_sum_of_sample_mean` CP 分支如何切 loss_mask
- [ ] 能列举 routing replay 四种 stage
- [ ] 能说明 ref forward 为何用 fallthrough
- [ ] 能解释 CI 在 routing replay 下跳过 kl checker 的原因

## 源码覆盖

- [ ] 能画 CP offset 与 response logprob 索引关系

## 维护者检查

- [ ] tags `slime/batch/23`
- [ ] 前缀 `23-CP-RoutingReplay-`
- [ ] [[Slime-progress]] 批次 23 → ✅

---

**批次 23 状态：** ✅ 已完成
- [ ] 双链 [[20-Train-Data-00-MOC]]、[[22-Loss-Policy-00-MOC]]

## 建议验证

- [ ] `tests/test_loss_cp_invariance.py`
