---
type: batch-doc
module: 22-Loss-Policy
batch: "22"
doc_type: concept
title: "Loss · Policy · 核心概念"
tags:
  - slime/batch/22
  - slime/module/loss-policy
  - slime/doc/concept
updated: 2026-07-02
---

# Loss · Policy · 核心概念

---

## 1. loss_type 三分法

| `loss_type` | 函数 | 典型场景 |
|-------------|------|----------|
| `policy_loss` | `policy_loss_function` | PPO / GRPO / GSPO / CISPO actor |
| `value_loss` | `value_loss_function` | critic 训练 |
| `sft_loss` | `sft_loss_function` | 纯 SFT / warmup |

Actor / critic 分步调用 `forward_only` + `train`，每步设置不同 `loss_type`（见 [[19-Train-Step-01-核心概念]]）。

---

## 2. policy 分支：advantage_estimator 与 loss 形态

| 估计器 | policy loss 差异 |
|--------|------------------|
| 默认 PPO | `ppo_kl = old_log_probs - log_probs`，`compute_policy_loss` clip |
| `gspo` | 序列级 KL expand 到 token，`compute_gspo_kl` |
| `cispo` | `compute_cispo_loss`：clip ratio **detach**，梯度走 `log_probs` |

**Explain：** advantage 在[[21-Loss-Advantages-00-MOC]] 算完；本专题只消费 `batch["advantages"]` 与 old/rollout logprob。

---

## 3. old policy 来源

**Code：**

```python
## 来源：slime/backends/megatron_utils/loss.py L911-L912
    advantages = torch.cat(batch["advantages"], dim=0)
    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch.get("log_probs")
```

- `use_rollout_logprobs=True`：ratio 相对 **SGLang rollout** 引擎 logprob
- 否则：相对 actor 在 train 前 `forward_only` 重算的 `log_probs`

---

## 4. ppo_utils 核心算子

| 函数 | 作用 |
|------|------|
| `compute_policy_loss` | dual-clip PPO surrogate |
| `compute_cispo_loss` | MiniMax-M1 CISPO |
| `compute_gspo_kl` | 序列 KL broadcast 到 token |
| `compute_approx_kl` | ref KL（k1/k2/k3/low_var_kl） |
| `calculate_log_probs_and_entropy` | TP vocab parallel + chunk |

---

## 5. TIS / mismatch 校正

**Explain：** `use_tis` 或 `get_mismatch_metrics` 时进入 TIS 路径；默认 `vanilla_tis_function`，ICEPOP 经 `--custom-tis-function-path` 注入。CLI 互斥与断言见 [[22-Loss-Policy-04-关键问题#Q8：vanilla_tis / ICEPOP 与 --use-tis 怎么配？]] 与 [[04-Arguments-TrainRollout-01-核心概念]]。

**Code：**

```python
## 来源：slime/backends/megatron_utils/loss.py L855-L878
def icepop_function(args, *, pg_loss, train_log_probs, rollout_log_probs, loss_masks, **kwargs):
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    ice_ratio = torch.exp(old_log_probs - rollout_log_probs)
    ice_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    ice_weight = torch.where(
        (ice_ratio >= args.tis_clip_low) & (ice_ratio <= args.tis_clip), ice_ratio, torch.zeros_like(ice_ratio)
    )
    ice_clipfrac = (ice_weight != ice_ratio).float()
    metrics = {"tis": ice_ratio.clone().detach(), "tis_clipfrac": ice_clipfrac.clone().detach(), "tis_abs": ice_abs.clone().detach()}
    pg_loss = pg_loss * ice_weight
    return pg_loss, loss_masks, metrics
```

**Comment：** `vanilla_tis_function` 对 ratio 做 clip 后 **乘** 到 `pg_loss`；ICEPOP 用 `ice_weight` 零化超界 token 的 loss 贡献，**不改** loss_masks。

---

## 6. Megatron loss 缩放

**Explain：** `loss_function` 返回的 scalar 需乘 `num_microbatches / step_global_batch_size * dp_size`（per-rollout mean 模式），以对接 Megatron 梯度累积与 CP 因子。

**Code：**

```python
## 来源：slime/backends/megatron_utils/loss.py L1290-L1298
    if not args.calculate_per_token_loss:
        loss = (
            loss
            * num_microbatches
            / step_global_batch_size
            * mpu.get_data_parallel_world_size(with_context_parallel=True)
        )
    else:
        loss = loss * mpu.get_context_parallel_world_size()
```

---

## 7. value / sft 要点

- **value**：PPO-style clip `values` vs `old_values`，max(surr1, surr2)
- **sft**：`-sum_of_sample_mean(log_probs)`，无 advantage

---

## 8. CISPO 与 PPO 梯度差异（读者必记）

CISPO：`pg_losses = -ratio_truncated.detach() * advantages * log_probs`  
PPO：梯度同时经过 ratio 与 clip（见 `compute_policy_loss`）。

验证见 [[22-Loss-Policy-04-关键问题]] + `test_cispo_loss.py`。
