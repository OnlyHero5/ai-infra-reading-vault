---
type: batch-doc
module: 24-WeightSync-Dist
batch: "24"
doc_type: checkpoint
title: "NCCL 权重同步 · 验收清单"
tags:
  - slime/batch/24
  - slime/module/weight-sync-dist
  - slime/doc/checkpoint
updated: 2026-07-02
---

# NCCL 权重同步 · 验收清单

## 读者自测（不打开 slime/）

- [ ] 能口头说明 `UpdateWeightFromDistributed` 在 generate → train → update_weights 闭环中的职责
- [ ] 能解释为何 `--update-weight-transport=nccl` 与 `--colocate` 互斥（各自走哪条路径）
- [ ] 能画出「TP all_gather → convert_to_hf → NCCL broadcast」三步数据流
- [ ] 能说明 PP source rank（DP=0 且 TP=0）的职责，以及非 source rank 仍参与哪些 collective
- [ ] 能说出 `_iter_non_expert_chunks` 与 `_iter_expert_chunks` 分两趟的原因
- [ ] 能解释 `rollout_engine_lock` 存在的理由
- [ ] 能说明 `HfWeightIteratorDirect` 与 NCCL 路径的共享点与差异（非直接调用关系）
- [ ] 能列举 3 个关键 CLI：`update_weight_transport`、`update_weight_buffer_size`、`megatron_to_hf_mode`

## 源码锚点核对（基线 `22cdc6e1`）

| 文件 | 关键符号 | 已覆盖 |
|------|----------|--------|
| `actor.py` | `update_weights`, updater 选型 | ✅ |
| `common.py` | `all_gather_param`, `named_params_and_buffers` | ✅ |
| `update_weight_from_distributed.py` | `UpdateWeightFromDistributed`, connect/broadcast | ✅ |
| `hf_weight_iterator_direct.py` | `HfWeightIteratorDirect`, param buckets | ✅ |

## 建议动手验证

1. grep 日志 `[slime-pp_0] Update weights` 确认 bucket 数量合理
2. 对比 `--update-weight-transport disk` 与 nccl 的单轮 sync 耗时（同硬件）
3. MoE 模型确认 expert 第二趟 tqdm 有独立 progress

---

**专题 24 完成状态：** ✅ 已完成
