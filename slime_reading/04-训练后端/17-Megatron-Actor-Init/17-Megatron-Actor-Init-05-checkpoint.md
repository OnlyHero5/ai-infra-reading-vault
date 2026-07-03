---
type: batch-doc
module: 17-Megatron-Actor-Init
batch: "17"
doc_type: checkpoint
title: "Megatron Actor 初始化 · 验收清单"
tags:
  - slime/batch/17
  - slime/module/megatron-actor-init
  - slime/doc/checkpoint
updated: 2026-07-02
---

# Megatron Actor 初始化 · 验收清单

## 读者自测（不打开 slime/）

- [ ] 能口头说明 `MegatronTrainRayActor.init` 从 `debug_rollout_only` 判断到返回 `start_rollout_id` 的 **至少 8 个步骤**
- [ ] 能画出本模块在 `generate → train → update_weights` 闭环中的位置（init 发生在主循环 **之前**）
- [ ] 能说出 3 个核心函数及其职责：
  - `TrainRayActor.init` — PyTorch distributed + gloo
  - `initialize.init` — Megatron mpu + seed + microbatch 计算器
  - `MegatronTrainRayActor.init` — 模型、权重备份、weight_updater、offload sleep
- [ ] 能解释 `sleep` / `wake_up` 在 init 末尾与 `train()` 内各触发一次的原因
- [ ] 能说明 `role=critic` 时 init 为何提前 return 且不创建 `weight_updater`
