---
type: batch-doc
module: 20-Train-Data
batch: "20"
doc_type: checkpoint
title: "Train Data · 验收清单"
tags:
  - slime/batch/20
  - slime/module/train-data
  - slime/doc/checkpoint
updated: 2026-07-02
---

# Train Data · 验收清单

## 读者自测（不打开 slime/）

- [ ] 能说明 `build_dp_schedule` 四步：rollout 分步 → pack mbs → 对齐 K → 分配 DP
- [ ] 能解释 `process_rollout_data` 中 `partition` 的作用
- [ ] 能描述 `get_batch` 如何把 list tokens 变成 `PackedSeqParams` + CP 切片
- [ ] 能说明 `DataIterator` 与 VPP 多 stage 的关系
- [ ] 能解释 `rollout_mask_sums` 为何在 Rollout 侧预计算
- [ ] 能对比 dynamic batch 与 static `micro_batch_size` 的差异

## 源码覆盖自测

- [ ] 能解释 `cu_seqlens * cp_size` 的含义
- [ ] 能说明 `first_fit_pack` 与 `expand_bins_by_splitting` 的协作

## 建议动手验证

- [ ] 阅读 `tests/test_dp_schedule.py` 对照不变量
- [ ] （可选）打印一轮 `micro_batch_indices` 确认 rank-local 下标
