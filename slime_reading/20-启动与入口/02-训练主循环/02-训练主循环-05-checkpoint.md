---
type: batch-doc
module: 02-训练主循环
batch: "02"
doc_type: checkpoint
title: "训练主循环 · 验收清单"
tags:
  - slime/batch/02
  - slime/module/train-loop
  - slime/doc/checkpoint
updated: 2026-07-02
---

# 训练主循环 · 验收清单

## 读者自测（不打开 slime/）

- [ ] 能按顺序说出 bootstrap 四步及首次 `update_weights` 时机
- [ ] 能口述 sync 主循环单步：generate → train → save? → update_weights → eval?
- [ ] 能说明 async prefetch 与 sync 的 3 点差异
- [ ] 能解释 `should_run_periodic_action` 何时在最后一 rollout 返回 True
- [ ] 能说明 PPO critic-only 步 Actor 是否参与 `async_train`
- [ ] 能解释 colocate 下 `offload` / `onload_weights` / `onload_kv` 顺序

## 快速自测题

1. **`train_async` 为何 update 前要 `ray.get(rollout_data_next_future)`？** 防止权重更新打断进行中的 generate。
2. **`num_rollout_per_epoch` 从哪来？** `create_rollout_manager` 返回值，与 dataset 规模相关。
3. **eval 在 step 0 何时跑两次？** bootstrap 前 eval-only（num_rollout=0）与 `skip_eval_before_train=False` 时循环内 rollout_id=0 eval 不同场景。

## 通过标准

能对照 [[02-训练主循环-03-数据流与交互]] 时序图 blank-box 填空即通过。

## 下一批

→ [[03-Arguments-Ray-00-MOC]]
