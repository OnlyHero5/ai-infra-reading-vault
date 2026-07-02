---
type: batch-doc
module: 22-Loss-Policy
batch: "22"
doc_type: checkpoint
title: "Loss · Policy · 验收清单"
tags:
  - slime/batch/22
  - slime/module/loss-policy
  - slime/doc/checkpoint
updated: 2026-07-02
---

# Loss · Policy · 验收清单

## 读者自测

- [ ] 能对比 PPO clip 与 CISPO stop-gradient ratio 的梯度路径
- [ ] 能说明 `loss_function` 返回三元组如何对接 Megatron
- [ ] 能列举 `policy_loss_function` 中 OPSM / TIS / KL loss 的启用条件
- [ ] 能解释 GSPO 为何需要 `all_gather_with_cp` 再算序列 KL
- [ ] 读过 `test_cispo_loss.py` 两个测试在验什么

## 维护者检查

- [ ] 热点批 02 内嵌代码 ≥400 行、代码段 ≥15（已去重 icepop / OPSM / dual-clip 重复块）
- [ ] ICEPOP 块对齐 `loss.py` L855–878（commit `22cdc6e1`）
- [ ] Q8 覆盖 `--use-tis` / `--custom-tis-function-path` / ICEPOP 互斥关系
- [ ] frontmatter tags 完整
- [ ] 双链 [[21-Loss-Advantages-00-MOC]]、[[23-CP-RoutingReplay-00-MOC]]、[[04-Arguments-TrainRollout-01-核心概念]]
- [ ] [[Slime-progress]] 批次 22 → ✅

---

**批次 22 状态：** ✅ 已完成
