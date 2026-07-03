---
type: batch-doc
module: 22-Loss-Policy
batch: "22"
doc_type: walkthrough
title: "Loss · Policy · 源码走读"
tags:
  - slime/batch/22
  - slime/module/loss-policy
  - slime/doc/walkthrough
updated: 2026-07-02
---

# Loss · Policy · 源码走读

> 走读顺序：`compute_policy_loss` / `compute_cispo_loss` → `policy_loss_function` → `value_loss_function` → `sft_loss_function` → `loss_function`  
> 基线 `22cdc6e1` · **本专题内嵌代码热点 ≥400 行**

---

## 1. ppo_utils — PPO clip

**Explain：** `ratio = exp(-ppo_kl)`；dual-clip 在 `advantages < 0` 且传入 `eps_clip_c>1` 时用第三分支下界（Slime 默认配置常不传该参数）。

**Code：**

```python
## 来源：ppo_utils.py L124-L148
@torch.compile(dynamic=True)
def compute_policy_loss(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: float | None = None,
):
    ratio = (-ppo_kl).exp()
    pg_losses1 = -ratio * advantages
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    clipfrac = torch.gt(pg_losses2, pg_losses1).float()

    if eps_clip_c is not None:
        pg_losses3 = -eps_clip_c * advantages
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    else:
        pg_losses = clip_pg_losses1

    return pg_losses, clipfrac
```

---

## 2. ppo_utils — CISPO

**Code：**

```python
## 来源：ppo_utils.py L151-L171
@torch.compile(dynamic=True)
def compute_cispo_loss(
    ppo_kl: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
):
    ratio = (-ppo_kl).exp()
    ratio_truncated = torch.clamp(ratio, min=1.0 - eps_clip, max=1.0 + eps_clip_high)
    pg_losses = -ratio_truncated.detach() * advantages * log_probs
    clipfrac = (ratio_truncated != ratio).float()
    return pg_losses, clipfrac
```

---

## 3. ppo_utils — GSPO KL

**Code：**

```python
## 来源：ppo_utils.py L95-L121
def compute_gspo_kl(
    full_log_probs: list[torch.Tensor],
    full_old_log_probs: list[torch.Tensor],
    local_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> torch.Tensor:
    ppo_kl = [
        ((old_logprob - log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)
        for log_prob, old_logprob, loss_mask in zip(full_log_probs, full_old_log_probs, loss_masks, strict=False)
    ]
    ppo_kl = [kl.expand_as(log_prob) for kl, log_prob in zip(ppo_kl, local_log_probs, strict=False)]
    ppo_kl = torch.cat(ppo_kl, dim=0)
    return ppo_kl
```

---

## 4. ppo_utils — compute_approx_kl

**Code：**

```python
## 来源：ppo_utils.py L11-L51
@torch.compile(dynamic=True)
def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_loss_type: str,
    importance_ratio: torch.Tensor | None = None,
) -> torch.Tensor:
    log_ratio = log_probs.float() - log_probs_base.float()

    if kl_loss_type == "k1":
        kl = log_ratio
    elif kl_loss_type == "k2":
        kl = log_ratio**2 / 2.0
    elif kl_loss_type in ["k3", "low_var_kl"]:
        log_ratio = -log_ratio
        kl = log_ratio.exp() - 1 - log_ratio
    else:
        raise ValueError(f"Unknown kl_loss_type: {kl_loss_type}")

    if importance_ratio is not None:
        kl = importance_ratio * kl

    if kl_loss_type == "low_var_kl":
        kl = torch.clamp(kl, min=-10, max=10)

    return kl
```

---

## 5. policy_loss_function — logprob 重算

**Code：**

```python
## 来源：loss.py L911-L932
    advantages = torch.cat(batch["advantages"], dim=0)
    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch.get("log_probs")

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
        **get_rollout_top_p_logprob_kwargs(args, batch),
    )

    log_probs = log_probs_and_entropy["log_probs"]
    if not args.use_rollout_logprobs and not old_log_probs:
        old_log_probs = [log_prob.detach() for log_prob in log_probs]
    train_log_probs_for_tis = batch.get("log_probs")
    if not train_log_probs_for_tis:
        train_log_probs_for_tis = [log_prob.detach() for log_prob in log_probs]
```

---

## 6. OPSM 与 GSPO 预 gather

**Code：**

```python
## 来源：loss.py L934-L961
    need_full_log_probs = args.use_opsm or args.advantage_estimator == "gspo"

    full_log_probs = None
    full_old_log_probs = None
    if need_full_log_probs:
        full_log_probs = [
            all_gather_with_cp(log_prob, total_length, response_length)
            for log_prob, total_length, response_length in zip(
                log_probs, total_lengths, response_lengths, strict=False
            )
        ]
        full_old_log_probs = [
            all_gather_with_cp(old_log_prob, total_length, response_length)
            for old_log_prob, total_length, response_length in zip(
                old_log_probs, total_lengths, response_lengths, strict=False
            )
        ]

    if args.use_opsm:
        opsm_mask, opsm_clipfrac = compute_opsm_mask(
            args=args,
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            advantages=batch["advantages"],
            loss_masks=batch["loss_masks"],
        )
```

---

## 7. KL 分支与 CISPO / PPO 选择

**Code：**

```python
## 来源：loss.py L963-L984
    if args.advantage_estimator == "gspo":
        ppo_kl = compute_gspo_kl(
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            local_log_probs=log_probs,
            loss_masks=batch["loss_masks"],
        )
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
    else:
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        ppo_kl = old_log_probs - log_probs

    if args.advantage_estimator == "cispo":
        pg_loss, pg_clipfrac = compute_cispo_loss(ppo_kl, log_probs, advantages, args.eps_clip, args.eps_clip_high)
    else:
        pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)

    if args.use_opsm:
        pg_loss = pg_loss * opsm_mask
```

---

## 8. TIS — vanilla_tis_function

**Code：**

```python
## 来源：loss.py L831-L852
def vanilla_tis_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    tis = torch.exp(old_log_probs - rollout_log_probs)
    tis_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
    tis_clipfrac = (tis_weights != tis).float()
    metrics = {
        "tis": tis.clone().detach(),
        "tis_clipfrac": tis_clipfrac.clone().detach(),
        "tis_abs": tis_abs.clone().detach(),
    }
    pg_loss = pg_loss * tis_weights
    return pg_loss, loss_masks, metrics
```

---

## 9. TIS 路径 reducer 重建

**Code：**

```python
## 来源：loss.py L1015-L1029
        pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)

        sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            modified_response_masks,
            batch["rollout_mask_sums"],
            args.calculate_per_token_loss,
        )
```

---

## 10. entropy、KL loss、metrics

**Code：**

```python
## 来源：loss.py L1042-L1067
    pg_loss = pg_loss_reducer(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

    entropy = log_probs_and_entropy["entropy"]
    entropy = torch.cat(entropy, dim=0)
    entropy_loss = sum_of_sample_mean(entropy)

    loss = pg_loss - args.entropy_coef * entropy_loss

    if args.use_kl_loss:
        ref_log_probs = batch["ref_log_probs"]
        ref_log_probs = torch.cat(ref_log_probs, dim=0)
        importance_ratio = None
        if args.use_unbiased_kl:
            importance_ratio = torch.exp(log_probs - old_log_probs)
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl_loss = sum_of_sample_mean(kl)

        loss = loss + args.kl_loss_coef * kl_loss
```

---

## 11. icepop_function — 硬拒绝式 TIS

**Explain：** 区间外 importance ratio 置零（非 clip 到边界），比 `vanilla_tis_function` 更激进；通过 `--custom-tis-function-path` 选用（无独立 CLI 开关）。CLI 互斥见 [[22-Loss-Policy-04-关键问题#Q8：vanilla_tis / ICEPOP 与 --use-tis 怎么配？]]。

**Code：**

```python
## 来源：loss.py L855-L878
def icepop_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    ice_ratio = torch.exp(old_log_probs - rollout_log_probs)
    ice_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    ice_weight = torch.where(
        (ice_ratio >= args.tis_clip_low) & (ice_ratio <= args.tis_clip), ice_ratio, torch.zeros_like(ice_ratio)
    )
    ice_clipfrac = (ice_weight != ice_ratio).float()
    metrics = {
        "tis": ice_ratio.clone().detach(),
        "tis_clipfrac": ice_clipfrac.clone().detach(),
        "tis_abs": ice_abs.clone().detach(),
    }
    pg_loss = pg_loss * ice_weight
    return pg_loss, loss_masks, metrics
```

**Comment：** 与 `vanilla_tis_function` 共用 `tis_clip` / `tis_clip_low`；ICEPOP **不改** `loss_masks`（仅把超界 token 的 `pg_loss` 乘零）。

---

## 12. value_loss_function

**Code：**

```python
## 来源：loss.py L1136-L1167
    old_values = torch.cat(batch["values"], dim=0)

    _, values = get_values(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
    )
    values = torch.cat([value.flatten() for value in values["values"]], dim=0)

    returns = torch.cat(batch["returns"], dim=0)

    values_clipfrac = torch.abs(values - old_values) > args.value_clip
    values_clipped = old_values + (values - old_values).clamp(-args.value_clip, args.value_clip)
    surr1 = (values_clipped - returns) ** 2
    surr2 = (values - returns) ** 2
    loss = torch.max(surr1, surr2)

    loss = sum_of_sample_mean(loss)
    values_clipfrac = sum_of_sample_mean(values_clipfrac.float())
```

---

## 13. sft_loss_function

**Code：**

```python
## 来源：loss.py L1192-L1216
    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
    )

    log_probs = log_probs_and_entropy["log_probs"]
    log_probs = torch.cat(log_probs, dim=0)
    loss = -sum_of_sample_mean(log_probs)

    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    return (
        loss,
        {
            "loss": loss.clone().detach(),
        },
    )
```

---

## 14. loss_function — dispatch 与 allgather-CP 零 loss

**Code：**

```python
## 来源：loss.py L1254-L1262
    num_tokens = sum([torch.clamp_min(loss_mask.sum(), 1) for loss_mask in batch["loss_masks"]])

    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        batch["rollout_mask_sums"],
        args.calculate_per_token_loss,
    )
```

```python
## 来源：loss.py L1281-L1320
    if args.allgather_cp and mpu.get_context_parallel_world_size() > 1:
        loss = loss + 0 * logits.sum()

    if not args.calculate_per_token_loss:
        loss = (
            loss
            * num_microbatches
            / step_global_batch_size
            * mpu.get_data_parallel_world_size(with_context_parallel=True)
        )
    else:
        loss = loss * mpu.get_context_parallel_world_size()

    return (
        loss,
        (num_tokens if args.calculate_per_token_loss else torch.tensor(1, device=logits.device)),
        {
            "keys": list(log.keys()),
            "values": torch.tensor(
                [
                    num_tokens if args.calculate_per_token_loss else 0,
                ]
                + list(log.values()),
                device=logits.device,
            ),
        },
    )
```

---

## 15. test_cispo_loss 对照

**Code：**

```python
## 来源：tests/test_cispo_loss.py L22-L31
def test_compute_cispo_loss_matches_closed_form_surrogate(eps_clip, eps_clip_high, ratios, clamped):
    ppo_kl = -torch.tensor([math.log(r) for r in ratios])

    pg_losses, clipfrac = compute_cispo_loss(ppo_kl, LOG_PROBS, ADVANTAGES, eps_clip, eps_clip_high)

    expected_losses = -torch.tensor(clamped) * ADVANTAGES * LOG_PROBS
    torch.testing.assert_close(pg_losses, expected_losses, rtol=1e-6, atol=1e-6)
```

```python
## 来源：tests/test_cispo_loss.py L35-L48
def test_compute_cispo_loss_gradient_flows_only_through_log_probs(...):
    log_ratios = torch.tensor([math.log(r) for r in ratios], requires_grad=True)
    ppo_kl = -log_ratios
    log_probs = LOG_PROBS.clone().requires_grad_()

    pg_losses, _ = compute_cispo_loss(ppo_kl, log_probs, ADVANTAGES, eps_clip, eps_clip_high)
    pg_losses.sum().backward()

    torch.testing.assert_close(log_probs.grad, -torch.tensor(clamped) * ADVANTAGES, rtol=1e-6, atol=1e-6)
    assert log_ratios.grad is None or torch.all(log_ratios.grad == 0)
```

---

## 16. compute_opsm_mask（Off-Policy Sequence Masking）

**Explain：** 仅当 **advantage<0 且 seq_kl>opsm_delta** 时 mask 整序列（OPSM 论文动机：坏 off-policy 负优势样本）。

**Code：**

```python
## 来源：ppo_utils.py L54-L92
def compute_opsm_mask(
    args: Namespace,
    full_log_probs: list[torch.Tensor],
    full_old_log_probs: list[torch.Tensor],
    advantages: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    for full_log_prob, full_old_log_prob, advantage, loss_mask in zip(
        full_log_probs, full_old_log_probs, advantages, loss_masks, strict=False
    ):
        seq_kl = ((full_old_log_prob - full_log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)
        mask = ((advantage < 0) & (seq_kl > args.opsm_delta)).float()
        opsm_mask_list.append(1 - mask)
    opsm_mask = torch.cat(opsm_mask_list, dim=0)
    return opsm_mask, opsm_clipfrac
```

---

## 17. _VocabParallelLogProbEntropy forward 要点

**Explain：** TP 分片上算 softmax；target 不在本 rank vocab 范围则 masked。`log_prob_keep_mask` 把 nucleus 外 logits 置 -inf，但 **target 行**仍保留原 logit。

**Code：**

```python
## 来源：ppo_utils.py L261-L275
            log_prob_logits = vocab_parallel_logits.masked_fill(~log_prob_keep_mask, float("-inf"))
            if local_target_rows.numel() > 0:
                log_prob_logits[local_target_rows, masked_target_1d[local_target_rows]] = vocab_parallel_logits[
                    local_target_rows, masked_target_1d[local_target_rows]
                ]
            predicted_logits, log_prob_sum_exp_logits, log_prob_softmax, _ = vocab_parallel_softmax(
                log_prob_logits, inplace=True
            )
```

---

## 18. get_grpo_returns — advantage 侧依赖

**Explain：** [[21-Loss-Advantages-00-MOC]] 调用；policy loss 侧 GSPO 用序列 KL。此处列出便于理解 ppo_utils 全文件边界。

**Code：**

```python
## 来源：ppo_utils.py L361-L368
def get_grpo_returns(rewards, kl):
    returns = []
    for i in range(len(rewards)):
        returns.append(torch.ones_like(kl[i]) * rewards[i])
    return returns
```

---

## 19. get_advantages_and_returns_batch + chunked_gae

**Explain：** PPO GAE 默认 `chunked=True` 用 parallel scan 降序列依赖；CP 下先 all_gather reward/value。

**Code：**

```python
## 来源：ppo_utils.py L604-L609
        else:
            full_advantages, full_returns = chunked_gae(
                rewards=full_rewards,
                values=full_values,
                gamma=gamma,
                lambd=lambd,
            )
```

---

## 20. loss_function logging_dict 分母占位

**Explain：** per-rollout mean 时 `values[0]=0`，`train_one_step` 用 `step_global_batch_size` 替换；避免 per-mb 分数路由常数。

**Code：**

```python
## 来源：loss.py L1305-L1318
            "values": torch.tensor(
                [num_tokens if args.calculate_per_token_loss else 0]
                + list(log.values()),
                device=logits.device,
            ),
```

---

## 21. reported_loss 中 OPD / mismatch 指标

**Explain：** `opd_reverse_kl` 若 batch 含则写入；TIS metrics 用 **pre-RS** reducer 聚合。

**Code：**

```python
## 来源：loss.py L1093-L1108
    if args.get_mismatch_metrics or args.use_tis:
        reported_loss["ois"] = sum_of_sample_mean_for_mismatch_metrics(ois).clone().detach()
        for metric_key, metric_value in tis_metrics.items():
            reported_loss[metric_key] = sum_of_sample_mean_for_mismatch_metrics(metric_value)
    if "opd_reverse_kl" in batch:
        opd_reverse_kl = torch.cat(batch["opd_reverse_kl"], dim=0)
        reported_loss["opd_reverse_kl"] = sum_of_sample_mean(opd_reverse_kl).clone().detach()
```

---

## 22. train_rollout_logprob_abs_diff

**Explain：** 诊断 rollout 与 train logprob 漂移；不参与 loss，仅 metric。

**Code：**

```python
## 来源：loss.py L1073-L1077
    if "rollout_log_probs" in batch and batch["rollout_log_probs"]:
        rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
        log_probs_to_compare = log_probs if args.use_rollout_logprobs else old_log_probs
        train_rollout_logprob_abs_diff = sum_of_sample_mean((log_probs_to_compare - rollout_log_probs).abs())
```

---

## 23. custom_pg_loss_reducer 注入点

**Explain：** 允许插件替换 pg_loss 归约（如自定义 per-sequence 权重），仍接收 total/response lengths 与 masks。

**Code：**

```python
## 来源：loss.py L1031-L1040
    if getattr(args, "custom_pg_loss_reducer_function_path", None) is not None:
        custom_pg_loss_reducer_func = load_function(args.custom_pg_loss_reducer_function_path)
        pg_loss_masks = modified_response_masks if (args.get_mismatch_metrics or args.use_tis) else batch["loss_masks"]
        pg_loss_reducer = custom_pg_loss_reducer_func(
            total_lengths, response_lengths, pg_loss_masks, args.calculate_per_token_loss
        )
    else:
        pg_loss_reducer = sum_of_sample_mean
```

---

## 24. value_loss 空 values 梯度

**Explain：** 与 policy 相同模式：`0 * values.sum()` 保图。

**Code：**

```python
## 来源：loss.py L1158-L1160
    if values.numel() == 0:
        loss += 0 * values.sum()
```

---

## 25. recompute_loss_function checkpoint

**Explain：** `torch.utils.checkpoint` 包 loss 计算，trade 计算换显存；与 Megatron activation checkpoint 正交。

**Code：**

```python
## 来源：loss.py L1276-L1277
    if args.recompute_loss_function:
        loss, log = checkpoint(func, args, batch, logits, sum_of_sample_mean, use_reentrant=False)
```

---

## 26. get_reinforce_plus_plus_returns CP 四步

**Explain：** all_gather full_kl → 末 token 加 reward → 反向折扣 → slice 回 local chunk。

**Code：**

```python
## 来源：ppo_utils.py L405-L434
        if cp_size > 1:
            full_kl_response = all_gather_with_cp(local_kl_chunk, total_len, response_len)
        ...
        token_level_rewards[last_idx] += rewards[i]
        ...
        if cp_size > 1:
            local_returns_chunk = slice_log_prob_with_cp(returns_for_seq, total_len, response_len)
```

---

## 27. 走读小结

**Explain：** 本专题 loss 栈与 Megatron `forward_step` 契约：`loss_function` 负责 rescale，各 `*_loss_function` 只产出未缩放标量与 detached metrics。

| 函数 | 梯度 | 关键输入 |
|------|------|----------|
| `policy_loss_function` | ✅ | advantages, old logprob, logits |
| `value_loss_function` | ✅ | returns, old values |
| `sft_loss_function` | ✅ | unconcat_tokens |
| `loss_function` | 调度 + rescale | loss_type, step_global_batch_size |
