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

## 维护者检查

- [ ] frontmatter tags 含 `slime/batch/17` + `slime/doc/*`
- [ ] 文件名前缀 `17-Megatron-Actor-Init-`，无泛化 `README` / `01-核心概念`
- [ ] Mermaid 块内使用 `<br/>` 换行
- [ ] 双链指向 `[[17-Megatron-Actor-Init-*]]` 或跨批 `[[]]`，无 `./` 相对路径
- [ ] 已更新 [[Slime-progress]] 批次 17 为 ✅

## 快速口试参考答案（维护者用）

1. **init 步骤（压缩版）：** debug 短路 → monkey_patch → super.init → initialize.init → tracking/profiler → HF config 串行读 → 可选 memory_margin → initialize_model_and_optimizer → train_parallel_config → critic 早退 **或** weights_backuper + ref/teacher/old_actor → weight_updater 选型 → clear_memory → offload 则 sleep → postprocess hook → return start_rollout_id

2. **闭环位置：** init 在 `create_training_models` 中完成；之后 `update_weights` 首次推权；每步 `train` 消费 rollout_data

3. **offload 双 sleep：** init 末 sleep 让出 GPU 给 Rollout；train 末 sleep 再次让出，形成时间复用
