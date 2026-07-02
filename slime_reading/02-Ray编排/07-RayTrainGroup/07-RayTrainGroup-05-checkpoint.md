---
type: batch-doc
module: 07-RayTrainGroup
batch: "07"
doc_type: checkpoint
title: "RayTrainGroup · 验收清单"
tags:
  - slime/batch/07
  - slime/module/ray-train-group
  - slime/doc/checkpoint
updated: 2026-07-02
---

# RayTrainGroup · 验收清单

## 读者自测（不打开 slime/）

- [ ] 能列出 RayTrainGroup 的 `async_init` / `async_train` / `update_weights` / `save_model` / `onload` / `offload` 六个 API 及是否内部 ray.get
- [ ] 能说明 rank 0 master addr/port 如何传播到其它 rank
- [ ] 能解释 `TrainRayActor.init` 中 NCCL + gloo 初始化顺序
- [ ] 能说明 `async_*` 返回 ObjectRef 列表的设计动机
- [ ] 能画出 rollout_data_ref 从 RolloutManager 到各 rank `train.remote` 的数据流
- [ ] 能说明 offload_train 时 runtime_env 中 LD_PRELOAD 的作用

## 维护者检查

- [ ] frontmatter tags 含 `slime/batch/07` + `slime/doc/*`
- [ ] 代码块含 `# 提交版本：22cdc6e1`
- [ ] 已更新 [[Slime-progress]] 批次 07 为 ✅

## 快速自测题

1. **默认 Actor 实现类？** `MegatronTrainRayActor`。
2. **update_weights 为何同步？** 权重 push 完成前不能开始下一轮 rollout。
3. **set_rollout_manager 谁上报 parallel config？** rank 0。

## 通过标准

全部读者自测项可口头回答，且能在 [[07-RayTrainGroup-02-源码走读]] 找到对应代码段。
