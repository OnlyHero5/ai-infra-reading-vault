---
type: batch-doc
module: 19-Train-Step
batch: "19"
doc_type: checkpoint
title: "Train Step · Checkpoint"
tags:
  - slime/batch/19
  - slime/module/train-step
  - slime/doc/checkpoint
updated: 2026-07-02
---

# Train Step · Checkpoint

---

## 验收清单

- [ ] 不打开 `slime/` 目录，仅读本专题 slime_reading 文档，能说明 **Train Step** 模块职责（一次 rollout 内 Megatron 训练：数据反序列化 → forward/backward → 权重备份）
- [ ] 能画出本模块在 RL 闭环（generate → train → update_weights）中的位置，并标出 Critic / Actor 顺序
- [ ] 能说出 3 个核心函数及其职责：
  - `MegatronTrainRayActor.train` — 入口、role 分派、offload 生命周期
  - `train_actor` / `train_critic` — log-prob/value/advantage 编排
  - `model.train_one_step` — 单次 Megatron PP backward + optimizer step
- [ ] 能追踪 **PPO + Critic** 典型路径：`train.py` → `async_train` → `train_critic` → `train_actor(external_data=value_refs)` → `train_one_step`

---

## 深度自测（可选）

1. **external_data 为空时** Actor 的 PPO 会怎样？（见 [[19-Train-Step-04-关键问题]] Q3）
2. **`num_critic_only_steps=2`** 时前两个 rollout 谁在被更新？
3. **`can_reuse_log_probs_in_loss`** 需要满足哪些条件？为何 PPO 测试用例不满足？
4. **`train_async.py` 为何禁止 colocate？** 与 GPU 分时策略有何关系？

---

## 建议动手验证

```bash
# 阅读测试用例中的 PPO + critic + colocate 组合
# slime/tests/test_qwen3_4B_ppo.py
pytest slime/tests/test_qwen3_4B_ppo.py -k execute  # 需 8 GPU 环境
```

关注 wandb/日志中的 `train/ppo_kl`、`train/step`、`train/global_batch_size`。

---

## 下一批

→ [[20-Train-Data-00-MOC]]：`process_rollout_data`、`get_data_iterator`、`get_batch` 如何构造本专题消费的 `rollout_data` 字段。
