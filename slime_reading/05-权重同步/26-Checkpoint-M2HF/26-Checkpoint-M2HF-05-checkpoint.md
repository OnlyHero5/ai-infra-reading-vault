---
type: batch-doc
module: 26-Checkpoint-M2HF
batch: "26"
doc_type: checkpoint
title: "Checkpoint M2HF · 验收清单"
tags:
  - slime/batch/26
  - slime/module/checkpoint-m2hf
  - slime/doc/checkpoint
updated: 2026-07-02
---

# Checkpoint M2HF · 验收清单

## 读者自测（不打开 slime/）

- [ ] 能说明 `load_checkpoint` 如何区分 Megatron ckpt 与 HF 目录
- [ ] 能解释 `bridge` vs `raw` 在加载/保存上的能力差异
- [ ] 能说出 `convert_to_hf` 的三步：padding 移除 → 模型路由 → 量化后处理
- [ ] 能描述 raw 保存时 rank 0 复制 config、多 node writer 写 safetensors 的流程
- [ ] 能指出本模块与 [[24-WeightSync-Dist-00-MOC]]、`[[25-WeightSync-Disk-00-MOC]]` 的复用关系

## 维护者检查

- [ ] frontmatter tags 含 `slime/batch/26` + `slime/doc/*`
- [ ] 六件套合计 ≥15 段内嵌代码、≥200 行
- [ ] 代码块首行含 `# 来源：` + 基线 `22cdc6e1`
- [ ] 已更新 [[Slime-progress]] 批次 26 为 ✅
- [ ] [[05-权重同步-00-MOC]] 批次 26 状态已同步

## 阶段 V 衔接

完成本批后，权重同步三批（24–26）闭环：**NCCL 在线同步 → disk/delta 离线同步 → checkpoint/HF 持久化**。
