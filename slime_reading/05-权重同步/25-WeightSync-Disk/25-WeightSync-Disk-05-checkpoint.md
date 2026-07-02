---
type: batch-doc
module: 25-WeightSync-Disk
batch: "25"
doc_type: checkpoint
title: "磁盘权重同步 · 验收清单"
tags:
  - slime/batch/25
  - slime/module/weight-sync-disk
  - slime/doc/checkpoint
updated: 2026-07-02
---

# 磁盘权重同步 · 验收清单

## 读者自测（不打开 slime/）

- [ ] 能对比 full disk、delta disk、colocate tensor 三种 transport 的选型条件
- [ ] 能口头描述 delta 四步：baseline → publish → apply → vanilla reload
- [ ] 能解释为何 delta 要求 `--update-weight-transport=disk` 且不支持 colocate
- [ ] 能说明 `snapshot` 为何从 `hf_checkpoint` seed 而非 GPU 权重
- [ ] 能对比 xor 与 overwrite 编码的 wire/幂等 trade-off
- [ ] 能描述 `sync_local_checkpoint` 与 `update_weights_from_disk(local_dir)` 的分工
- [ ] 能说明 colocate IPC 中 `gather_object` + `FlattenedTensorBucket` 的作用
- [ ] 能列举 `perf/update_weights_density` 与 `wire_bytes` 的含义
- [ ] 能说出 full disk 同步中 rank 0 与非 0 rank 的职责差异

## 维护者检查

- [ ] frontmatter tags 含 `slime/batch/25` + `slime/doc/*`
- [ ] 六件套前缀 `25-WeightSync-Disk-`
- [ ] 全批内嵌代码 ≥ 15 段、≥ 200 行
- [ ] 02 热点走读 ≥ 400 行内嵌代码
- [ ] Mermaid 无 `\n`（用 `<br/>`）
- [ ] 双链 [[24-WeightSync-Dist-00-MOC]]、[[26-Checkpoint-M2HF-00-MOC]] 可解析
- [ ] [[Slime-progress]] 批次 25 → ✅

## 源码锚点核对（基线 `22cdc6e1`）

| 文件 | 关键符号 | 已覆盖 |
|------|----------|--------|
| `update_weight_from_disk.py` | `UpdateWeightFromDisk.update_weights` | ✅ |
| `update_weight_from_disk_delta.py` | baseline/publish/reload/metrics | ✅ |
| `disk_delta.py` | `init_local_checkpoint`, `apply_deltas`, encodings | ✅ |
| `update_weight_from_tensor.py` | IPC gather, hybrid NCCL | ✅ |
| `sglang_engine.py` | `sync_local_checkpoint`, disk HTTP | ✅ |
| `actor.py` | updater 选型 | ✅ |

## 建议动手验证

1. `pytest tests/test_full_disk_weight_update.py -q`
2. 对比同模型 full disk vs nccl 单轮 sync 耗时
3. delta 模式观察首轮 skip publish、次轮 density 日志
4. colocate 场景确认 `torch.cuda.ipc_collect` 后显存回落

---

**批次 25 状态：** ✅ 已完成
