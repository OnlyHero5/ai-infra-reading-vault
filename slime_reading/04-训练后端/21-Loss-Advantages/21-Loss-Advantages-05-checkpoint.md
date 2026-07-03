---
type: batch-doc
module: 21-Loss-Advantages
batch: "21"
doc_type: checkpoint
title: "Loss · Advantages · 验收清单"
tags:
  - slime/batch/21
  - slime/module/loss-advantages
  - slime/doc/checkpoint
updated: 2026-07-02
---

# Loss · Advantages · 验收清单

## 读者自测（不打开 slime/）

- [ ] 仅读本专题 slime_reading，能说明 `compute_advantages_and_returns` 的职责：从 reward/logprob/value 生成 per-token `advantages` 与 `returns`
- [ ] 能画出本模块在 generate → train → update_weights 闭环中的位置（actor forward_only 之后、policy backward 之前）
- [ ] 能说出 4 个核心函数职责：`get_log_probs_and_entropy`、`get_values`、`compute_advantages_and_returns`、`apply_opd_kl_to_advantages`
- [ ] 能追踪一条 PPO 配置下：critic values → GAE → whitening → `policy_loss_function` 读取 advantages 的路径
- [ ] 能解释 GRPO 与 PPO 在 advantage 阶段的核心差异（broadcast reward vs GAE）
- [ ] 能说明 OPD reverse KL 如何进入 advantage（不减在 reward 上）

## 源码覆盖自测

- [ ] 能解释 `get_log_probs_and_entropy` 为何整段 `[T,V]` 计算再 `_extract_per_sample`
- [ ] 能说明 `allgather_cp` 时 `_allgather_cp_redistribute` 的必要性

## 建议动手验证

- [ ] 阅读 `tests/test_chunked_gae.py` 与 GAE 实现 `get_advantages_and_returns_batch` 对照
- [ ] （可选 GPU）跑通含 `--advantage-estimator ppo` 的最小训练，确认 `rollout_data` 含 `advantages`
