---
type: batch-doc
module: 19-Train-Step
batch: "19"
doc_type: faq
title: "Train Step · 关键问题"
tags:
  - slime/batch/19
  - slime/module/train-step
  - slime/doc/faq
updated: 2026-07-02
---

# Train Step · 关键问题

---

## Q1：`async_train` 是 Megatron 异步训练吗？

**不是。** 它只是 Ray 非阻塞 RPC：返回 `train.remote` 的 ObjectRef，主进程稍后 `ray.get`。Megatron 内部仍是同步 PP forward-backward。

```python
## 来源：slime/ray/actor_group.py L131-L133
    def async_train(self, rollout_id, rollout_data_ref, external_data=None):
        """Do one rollout training. Returns a list of Ray refs (one per worker).
```

真正「generate 与 train 重叠」见 `train_async.py` 与[[14-Alt-Rollout-00-MOC]] fully-async，与本模块 `train()` 实现正交。

---

## Q2：为何有 `num_critic_only_steps`？

Critic 需要稳定 value 估计后再更新 Actor。前 N 个 rollout 只跑 `critic_model.async_train`，不调用 `actor_model.async_train`。

```python
## 来源：slime/train.py L72-L79
        actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps

        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains_this_step:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
```

`test_qwen3_4B_ppo.py` 使用 `--num-critic-only-steps 1`：第 0 个 rollout 仅训 critic。

---

## Q3：Critic 和 Actor 各算一次 advantage，会不一致吗？

**会各自算一遍**——Critic 路径在 value forward 后立即 `compute_advantages_and_returns` 用于 value_loss；Actor 路径在注入 values 后再算一次用于 policy_loss。两次使用同一 reward / value 张量，但 Actor 侧重 policy 所需字段（如 normalized advantages）。

易错：若 Actor 未收到 `external_data["values"]`，last PP stage 的 `rollout_data["values"]` 为空，PPO advantage 会错。

**正确写法（框架已内置）：**

```python
## 来源：slime/train.py L75-L77
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains_this_step:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
```

**错误写法（反模式）：** 先 `ray.get(value_refs)` 再单独 `async_train` 不传 `external_data`——Actor 失去 old values。

---

## Q4：`loss_type` 何时被改写？

Critic 训练前临时设为 `value_loss`；Actor 训练依赖全局 args（通常 `policy_loss`）。不要在 custom hook 里永久改 `args.loss_type` 而不恢复。

```python
## 来源：slime/backends/megatron_utils/actor.py L413-L422
        self.args.loss_type = "value_loss"
        train(
            rollout_id,
            self.model,
            ...
        )
```

---

## Q5：动态 batch 下多个 train step 如何理解？

`num_microbatches` 是 **list**：每个元素对应一个 `train_one_step`。`global_batch_sizes[i]` 是该 step 的 rollout 计数，用于 loss 缩放与 LR scheduler。

```python
## 来源：slime/backends/megatron_utils/model.py L734-L737
    assert len(num_microbatches) == len(global_batch_sizes), (
        f"num_microbatches and global_batch_sizes must have the same length, "
        f"got {len(num_microbatches)} vs {len(global_batch_sizes)}"
    )
```

`test_qwen3_4B_ppo.py` 开启 `--use-dynamic-batch-size`，是验证多 step 路径的常用用例。

---

## Q6：offload 模式下 train 为何要先 wake 再 sleep？

Train 需要 GPU 上的 model 与 optimizer；offload 时 idle 状态通过 `torch_memory_saver.pause()` 释放显存。`train()` 配对调用保证与 rollout offload 分时复用 GPU。

```python
## 来源：slime/backends/megatron_utils/actor.py L384-L386, L396-L398
        if self.args.offload_train:
            self.wake_up()
        # ...
        if self.args.offload_train:
            del rollout_data
            self.sleep()
```

注意：`update_weights` 另有独立的 wake/reload 逻辑（见[[24-WeightSync-Dist-00-MOC]]）。

---

## Q7：CI 里 PPO 初始 KL 为何要求 ≈0？

Step 0 时 actor 与 ref（或 old policy）应一致，PPO ratio ≈1，KL ≈0。若失败，常见原因：权重未正确 load、routing replay 不一致、或 colocate 下权重 sync 顺序错误。

```python
## 来源：slime/backends/megatron_utils/model.py L895-L898
                if step_id == 0 and "train/ppo_kl" in log_dict and "train/pg_clipfrac" in log_dict:
                    assert log_dict["train/ppo_kl"] < 1e-8, f"{log_dict=}"
```

---

## Q8：与 GRPO 路径的差异（预览）

无 Critic 时走单 Actor `async_train`；`advantage_estimator=gspo` 等会禁用 log-prob 复用优化。GRPO 细节见[[21-Loss-Advantages-00-MOC]]–[[22-Loss-Policy-00-MOC]]，Train Step 层调用栈相同，差异在 `compute_advantages_and_returns` 与 `loss_function` 分支。

---

## 对比表：sync vs async 主循环

| 维度 | `train.py` | `train_async.py` |
|------|------------|------------------|
| colocate | 支持 | **不支持** |
| generate 时机 | 串行：generate → train | 重叠：prefetch 下一 rollout |
| train 调用 | 相同 `async_train` | 相同 |
| offload rollout | 支持 | 无等价 offload 块 |
