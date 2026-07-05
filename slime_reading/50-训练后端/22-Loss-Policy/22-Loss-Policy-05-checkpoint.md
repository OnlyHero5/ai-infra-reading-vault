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
