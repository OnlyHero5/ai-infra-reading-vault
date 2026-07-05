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
updated: 2026-07-05
---

# Loss · Policy · 源码走读

> 走读顺序：`compute_policy_loss` / `compute_cispo_loss` → `policy_loss_function` → TIS / OPSM / KL 分支 → `value_loss_function` / `sft_loss_function` → `loss_function`。
> 基线 `22cdc6e1`。

## 源码阅读依据

| 上游文件 | 本文关注点 |
|----------|------------|
| `slime/slime/backends/megatron_utils/loss.py` | 训练侧 loss 分派、policy/value/SFT loss、TIS、指标与 Megatron 缩放 |
| `slime/slime/utils/ppo_utils.py` | PPO / CISPO / GSPO / OPSM 数学 helper、TP logprob、GAE helper |
| `slime/tests/test_cispo_loss.py` | CISPO 闭式公式与梯度路径测试 |

## 设计主线：Loss Policy 为什么长成这样

Slime 的 loss 层不是单纯把 PPO 公式写成 PyTorch，而是在 **RL 算法变体** 和 **Megatron 分布式训练契约** 之间做适配。它的设计哲学可以概括为三层：

1. **数学 helper 保持纯函数化。** `compute_policy_loss`、`compute_cispo_loss`、`compute_gspo_kl` 只产生逐 token/逐局部 shard 的张量，不负责 batch reduction，也不关心 Megatron 如何缩放 loss。
2. **policy_loss_function 保留样本边界到最后。** logprob、mask、response length 在大部分流程中保持 list 形态，只有在需要喂给统一公式或 reducer 时才 `torch.cat`。这样 CP、allgather、OPSM、GSPO、TIS 才能在正确的序列边界上工作。
3. **loss_function 是训练框架适配器。** 内层函数返回“未按 Megatron 缩放的 loss + detached metrics”，外层才根据 microbatch、DP/CP world size、per-token/per-rollout mean 做缩放和 logging 字段封装。

这意味着读这篇时不要只问“公式是什么”，还要问：这个张量此刻是 full response、CP-local response，还是已经 flatten？这个 reducer 的分母代表 token 数还是 rollout 数？这个 metric 是否参与梯度？

## 首次阅读路径（约 35 分钟）

| 顺序 | 章节锚点 | 读完应能回答的问题 | 预计分钟 |
|------|----------|-------------------|----------|
| 1 | [[#1. ppo_utils — PPO clip]] · [[#2. ppo_utils — CISPO]] | PPO / CISPO 的梯度路径为什么不同？ | 7 |
| 2 | [[#5. policy_loss_function — logprob 重算]] · [[#7. KL 分支与 CISPO / PPO 选择]] | 新旧 logprob 从哪里来，何时 flatten？ | 8 |
| 3 | [[#8. TIS — vanilla_tis_function]] · [[#9. TIS 路径 reducer 重建]] | off-policy 权重为什么要和 reducer 分开处理？ | 6 |
| 4 | [[#10. entropy、KL loss、metrics]] · [[#23. reported_loss 中 OPD / mismatch 指标]] | 哪些值进 loss，哪些只进 metric？ | 5 |
| 5 | [[#14. loss_function — reducer 构造]] · [[#15. loss_function — Megatron 缩放与 allgather-CP 零 loss]] | loss 如何交给 Megatron，为什么空 shard 也要保图？ | 6 |
| 6 | [[#21. get_advantages_and_returns_batch 调用 chunked_gae]] · [[#22. chunked_gae — 并行化 GAE]] | advantage 侧为什么也要照顾 CP 和长序列依赖？ | 3 |

---

## 1. ppo_utils — PPO clip

**Explain：** `compute_policy_loss` 把 `ppo_kl = old_log_probs - log_probs` 转回重要性采样比率 `ratio = exp(log_probs - old_log_probs)`，再做 PPO clip。

**问题与约束：** policy loss 要支持非对称 clip 上下界，还要保留 dual-clip PPO 的可选第三分支，但它不能在这里做全局平均；否则 CP/local mask 和 per-rollout mean 的分母会被提前固定。

**设计选择：** helper 返回逐位置 `pg_losses` 和 `clipfrac`，把 reduction 留给 `policy_loss_function` 的 reducer。这样公式层不感知 batch 组织方式。

**代码逻辑：**

```python
## 来源：slime/utils/ppo_utils.py L124-L148
ratio = (-ppo_kl).exp()
pg_losses1 = -ratio * advantages
pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
clipfrac = torch.gt(pg_losses2, pg_losses1).float()
...
return pg_losses, clipfrac
```

**为什么这样写：** Slime 把“策略更新公式”和“样本如何归约”拆开，方便同一套 PPO helper 复用在普通 PPO、GSPO 序列 KL、OPSM 后置 mask 等路径中。

**不变量与失败模式：** `ppo_kl`、`advantages` 必须 shape 对齐；`eps_clip_c` 若传入必须大于 1。若提前 reduce，会破坏后续 TIS/OPSM 对逐 token loss 的改写能力。

**Comment：** 这里的 `clipfrac` 是逐位置诊断张量，不是最终标量；最终标量在 `sum_of_sample_mean` 后才形成。

---

## 2. ppo_utils — CISPO

**Explain：** CISPO 使用裁剪后的 IS ratio 作为停止梯度的权重，梯度只流经 `log_probs`。

**问题与约束：** CISPO 想保留 off-policy ratio 的幅度控制，但不希望 ratio 本身的梯度路径改变 policy gradient 的形态。

**设计选择：** `ratio_truncated.detach()` 明确切断 ratio 梯度，再乘 `advantages * log_probs`；clip 命中率仍作为 metric 返回。

**代码逻辑：**

```python
## 来源：slime/utils/ppo_utils.py L151-L171
ratio = (-ppo_kl).exp()
ratio_truncated = torch.clamp(ratio, min=1.0 - eps_clip, max=1.0 + eps_clip_high)
pg_losses = -ratio_truncated.detach() * advantages * log_probs
clipfrac = (ratio_truncated != ratio).float()
return pg_losses, clipfrac
```

**为什么这样写：** 它把“采样分布和训练分布的偏差校正”当作权重，而不是让 ratio 参与反向传播；这和 PPO surrogate 的 `-ratio * advantage` 梯度形态不同。

**不变量与失败模式：** `log_probs` 必须是当前策略的 logprob，不能误传 old/rollout logprob；否则梯度会落到错误策略。`detach()` 丢失会让 ratio 路径出现梯度，测试会失败。

**Comment：** 这段是整篇最能体现“算法语义写进梯度路径”的地方：公式看起来只多了一个 `detach()`，实际改变的是优化目标的导数。

---

## 3. ppo_utils — GSPO KL

**Explain：** GSPO 把序列级 KL 扩展成 local token 形状，让下游仍能沿用逐 token loss/reducer。

**问题与约束：** GSPO 的 KL 是序列级概念，但训练图里的 `pg_loss`、mask、reducer 都按本 rank 的 local token 张量工作。

**设计选择：** 先用 full response logprob 和 full loss mask 算每条样本的平均 KL，再 `expand_as(local_log_probs)` 变回 local 形状。

**代码逻辑：**

```python
## 来源：slime/utils/ppo_utils.py L95-L121
ppo_kl = [
    ((old_logprob - log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)
    for log_prob, old_logprob, loss_mask in zip(full_log_probs, full_old_log_probs, loss_masks, strict=False)
]
ppo_kl = [kl.expand_as(log_prob) for kl, log_prob in zip(ppo_kl, local_log_probs, strict=False)]
ppo_kl = torch.cat(ppo_kl, dim=0)
```

**为什么这样写：** 它不让 GSPO 侵入整条 loss 管线，而是把序列级信号伪装成“每个 token 同值”的张量，复用 PPO/CISPO 后续分支。

**不变量与失败模式：** `full_log_probs` 和 `full_old_log_probs` 必须是同一条完整 response 的对齐结果；如果在 CP-local shard 上直接算序列 KL，会把被切分的 response 当成完整序列。

**Comment：** 这是典型的“边界适配”写法：算法要 sequence-level，工程接口要 token-level，于是在边界处做一次 expand。

---

## 4. ppo_utils — compute_approx_kl

**Explain：** `compute_approx_kl` 提供多种 KL 估计器，用于额外的 reference KL penalty，而不是 PPO surrogate 的 `ppo_kl` 本身。

**问题与约束：** KL penalty 既要支持常见 k1/k2/k3 估计，也要支持低方差版本和 DeepSeek-V3.2 风格的 unbiased KL importance ratio。

**设计选择：** 用 `kl_loss_type` 做显式分支，只对 `low_var_kl` 做 clamp；importance ratio 是可选后乘项。

**代码逻辑：**

```python
## 来源：slime/utils/ppo_utils.py L11-L51
log_ratio = log_probs.float() - log_probs_base.float()
...
elif kl_loss_type in ["k3", "low_var_kl"]:
    log_ratio = -log_ratio
    kl = log_ratio.exp() - 1 - log_ratio
...
if importance_ratio is not None:
    kl = importance_ratio * kl
if kl_loss_type == "low_var_kl":
    kl = torch.clamp(kl, min=-10, max=10)
```

**为什么这样写：** KL penalty 是可插拔的稳定器，不应和 policy gradient 主公式绑死；把类型集中到一个 helper 里，`policy_loss_function` 只负责选择是否加上它。

**不变量与失败模式：** `log_probs` 与 `log_probs_base` 必须逐 token 对齐；未知 `kl_loss_type` 直接抛错，避免静默退化成错误正则。

**Comment：** 这段的哲学是“估计器可变，接口不变”：返回的仍是逐 token KL 张量。

---

## 5. policy_loss_function — logprob 重算

**Explain：** policy loss 入口先把 advantage flatten，然后用当前 forward 的 `logits` 重算当前策略 logprob；old logprob 根据配置来自 rollout 或训练 batch。

**问题与约束：** RL 训练里 rollout 策略、旧训练策略、当前训练策略可能不是同一个分布；loss 必须明确每一路 logprob 的语义。

**设计选择：** `old_log_probs` 的来源由 `args.use_rollout_logprobs` 决定；如果没有旧 logprob，则用当前 logprob 的 detach 版本作为 on-policy baseline。TIS 另保留 `train_log_probs_for_tis`。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/loss.py L911-L932
advantages = torch.cat(batch["advantages"], dim=0)
old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch.get("log_probs")
...
log_probs = log_probs_and_entropy["log_probs"]
if not args.use_rollout_logprobs and not old_log_probs:
    old_log_probs = [log_prob.detach() for log_prob in log_probs]
train_log_probs_for_tis = batch.get("log_probs")
if not train_log_probs_for_tis:
    train_log_probs_for_tis = [log_prob.detach() for log_prob in log_probs]
```

**为什么这样写：** Slime 没有把“old”写死成某一种来源，而是把 rollout/train/current 三者的关系显式化，给 off-policy 训练、TIS 诊断和 on-policy 退化路径留空间。

**不变量与失败模式：** `advantages` flatten 后必须和后续 `log_probs` flatten 顺序一致；`batch["unconcat_tokens"]`、`total_lengths`、`response_lengths` 是重算 logprob 的对齐依据。

**Comment：** 这段是理解整条 policy loss 的入口：变量名里的 old 不只是“上一轮模型”，而是由配置决定的比较分布。

---

## 6. OPSM 与 GSPO 预 gather

**Explain：** OPSM 和 GSPO 都需要 full response 视角，所以只在这些功能打开时做 CP all-gather。

**问题与约束：** CP 会把 response 切到不同 rank；普通 PPO 可在 local shard 上工作，但序列 KL / 序列 mask 不能只看本地片段。

**设计选择：** `need_full_log_probs = args.use_opsm or args.advantage_estimator == "gspo"`，把昂贵的 gather 限制在真正需要完整序列的路径。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/loss.py L934-L961
need_full_log_probs = args.use_opsm or args.advantage_estimator == "gspo"
...
if need_full_log_probs:
    full_log_probs = [
        all_gather_with_cp(log_prob, total_length, response_length)
        for log_prob, total_length, response_length in zip(log_probs, total_lengths, response_lengths, strict=False)
    ]
...
if args.use_opsm:
    opsm_mask, opsm_clipfrac = compute_opsm_mask(...)
```

**为什么这样写：** 它把“完整序列语义”作为条件成本，而不是默认让所有 policy loss 都支付 all-gather 开销。

**不变量与失败模式：** gather 后的 logprob 必须按原 response 顺序恢复；如果 `old_log_probs` 来源和 `log_probs` 来源不一致但长度对不上，GSPO/OPSM 的序列判断会失真。

**Comment：** 这段体现的是分布式训练里的常见取舍：尽量 local，只有算法语义要求 full sequence 时才跨 rank。

---

## 7. KL 分支与 CISPO / PPO 选择

**Explain：** 这里决定 `ppo_kl` 是 GSPO 的 sequence-level KL，还是普通 token-level old-current 差值；随后再按 estimator 选择 CISPO 或 PPO loss。

**问题与约束：** 多个 advantage estimator 共享同一个 policy loss 入口，但 KL 的粒度和 surrogate 的梯度语义不同。

**设计选择：** `gspo` 先把 sequence KL expand 成 token 形状；`cispo` 只替换 surrogate helper，不替换周边 reducer、entropy、KL loss、metrics 流程。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/loss.py L963-L984
if args.advantage_estimator == "gspo":
    ppo_kl = compute_gspo_kl(...)
    old_log_probs = torch.cat(old_log_probs, dim=0)
    log_probs = torch.cat(log_probs, dim=0)
else:
    old_log_probs = torch.cat(old_log_probs, dim=0)
    log_probs = torch.cat(log_probs, dim=0)
    ppo_kl = old_log_probs - log_probs

if args.advantage_estimator == "cispo":
    pg_loss, pg_clipfrac = compute_cispo_loss(...)
else:
    pg_loss, pg_clipfrac = compute_policy_loss(...)
```

**为什么这样写：** Slime 把 estimator 的差异压缩到两个接口点：KL 构造和 surrogate 公式。其余训练框架逻辑保持稳定。

**不变量与失败模式：** `torch.cat` 前的 list 顺序必须和 `advantages` 顺序一致；OPSM mask 在 `pg_loss` 形成后相乘，因此它只能屏蔽 loss，不改变 logprob/KL 计算本身。

**Comment：** 这是该文件的“分叉-汇合”结构：算法分支短，公共路径长。

---

## 8. TIS — vanilla_tis_function

**Explain：** TIS 用 train logprob 与 rollout logprob 的差构造 importance weight，并把权重乘到已生成的 `pg_loss` 上。

**问题与约束：** rollout 生成和训练更新之间可能有策略漂移；直接训练会让高 mismatch token 对梯度贡献过大。

**设计选择：** `tis = exp(train - rollout)`，再 clamp 到 `[tis_clip_low, tis_clip]`；同时返回原始 `tis`、`tis_abs`、`tis_clipfrac` 做诊断。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/loss.py L831-L852
rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
old_log_probs = torch.cat(train_log_probs, dim=0)
tis = torch.exp(old_log_probs - rollout_log_probs)
tis_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
...
pg_loss = pg_loss * tis_weights
return pg_loss, loss_masks, metrics
```

**为什么这样写：** TIS 被设计成后处理权重，而不是改写 PPO/CISPO helper；这样自定义 TIS 函数也能沿用同一个 policy loss 主干。

**不变量与失败模式：** `rollout_log_probs` 必须存在，否则 TIS 没有参考分布；logprob 拼接顺序必须和 `pg_loss` 对齐。权重只改 loss，不改原始 mask。

**Comment：** 变量 `old_log_probs` 在这个函数里实际来自 `train_log_probs`，阅读时要按参数语义而不是局部变量名理解。

---

## 9. TIS 路径 reducer 重建

**Explain：** TIS 或自定义 TIS 可能返回修改后的 response mask，因此 policy loss 的 reducer 要按新 mask 重建。

**问题与约束：** 如果某些 token 被 rejection-style masking 置零，loss 分子变了；但 metric 分母有时仍应按原始有效 token 统计。

**设计选择：** `tis_func` 返回 `(pg_loss, modified_response_masks, tis_metrics)`；`sum_of_sample_mean` 用 modified mask 重建，但分母继续接收 `rollout_mask_sums`。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/loss.py L1015-L1029
pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)

sum_of_sample_mean = get_sum_of_sample_mean(
    total_lengths,
    response_lengths,
    modified_response_masks,
    batch["rollout_mask_sums"],
    args.calculate_per_token_loss,
)
```

**为什么这样写：** 它把“哪些 token 参与反传”和“指标按什么总体解释”解耦，避免 rejection 后把诊断指标的分母也缩小到看不出 mismatch。

**不变量与失败模式：** 自定义 TIS 返回的 mask 必须和 response token 形状一致；如果 reducer 仍用旧 mask，已拒绝 token 可能被错误计入平均。

**Comment：** 这里是 loss 设计里最容易漏看的细节：mask 不只是布尔过滤器，它还决定 reducer 的统计口径。

---

## 10. entropy、KL loss、metrics

**Explain：** policy loss 标量由 policy gradient、entropy bonus、可选 reference KL penalty 组成；metrics 都在这里转为 detached 标量。

**问题与约束：** entropy 要鼓励探索，reference KL 要约束偏离；两者都依赖当前 logprob，但是否参与梯度和如何聚合必须明确。

**设计选择：** `pg_loss` 走可替换 reducer，`pg_clipfrac`、`ppo_kl`、`entropy` 走当前 `sum_of_sample_mean`；KL penalty 只在 `args.use_kl_loss` 时加入总 loss。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/loss.py L1042-L1067
pg_loss = pg_loss_reducer(pg_loss)
pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
ppo_kl = sum_of_sample_mean(ppo_kl)
...
loss = pg_loss - args.entropy_coef * entropy_loss

if args.use_kl_loss:
    ...
    kl = compute_approx_kl(...)
    kl_loss = sum_of_sample_mean(kl)
    loss = loss + args.kl_loss_coef * kl_loss
```

**为什么这样写：** 主 loss 组合在 policy 函数里完成，外层 `loss_function` 只负责训练框架缩放；这让算法项和 Megatron 积累逻辑互不污染。

**不变量与失败模式：** `ref_log_probs` 必须与当前 `log_probs` 对齐；`args.use_unbiased_kl` 时 importance ratio 使用当前和 old logprob，old 来源错误会影响 KL penalty。

**Comment：** `reported_loss` 里的值都 `clone().detach()`，这是为了日志而不是训练图。

---

## 11. icepop_function — 硬拒绝式 TIS

**Explain：** ICEPOP 和 vanilla TIS 使用同样的 ratio，但越界 token 的权重直接置零，而不是 clamp 到边界。

**问题与约束：** 有些 off-policy mismatch 太大时，clip 仍会保留梯度贡献；硬拒绝路径希望完全去掉这些 token 的 policy loss。

**设计选择：** 用 `torch.where` 在 `[tis_clip_low, tis_clip]` 内保留 ratio，区间外置零；metrics 仍复用 `tis` / `tis_clipfrac` / `tis_abs` 键名。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/loss.py L855-L878
ice_ratio = torch.exp(old_log_probs - rollout_log_probs)
ice_weight = torch.where(
    (ice_ratio >= args.tis_clip_low) & (ice_ratio <= args.tis_clip), ice_ratio, torch.zeros_like(ice_ratio)
)
ice_clipfrac = (ice_weight != ice_ratio).float()
...
pg_loss = pg_loss * ice_weight
return pg_loss, loss_masks, metrics
```

**为什么这样写：** Slime 把 ICEPOP 放成 TIS 函数变体，而不是新增一整条 policy loss 分支；差异集中在权重生成策略。

**不变量与失败模式：** ICEPOP 不修改 `loss_masks`，只把 loss 乘零；如果读者把它理解成 reducer 层 rejection，会误判分母行为。

**Comment：** `--custom-tis-function-path` 可以替换默认 TIS；ICEPOP 展示了这个扩展点期望的函数签名。

---

## 12. value_loss_function

**Explain：** value loss 使用 PPO 风格的 value clipping：当前 value 和 old value 的偏移被限制后，与 unclipped loss 取最大。

**问题与约束：** value head 也可能在一次更新里变化过大；如果只用普通 MSE，critic 会出现过激更新。

**设计选择：** `values_clipped = old_values + clamp(values - old_values)`，再对 clipped/unclipped squared error 取 `max`。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/loss.py L1136-L1167
old_values = torch.cat(batch["values"], dim=0)
...
returns = torch.cat(batch["returns"], dim=0)

values_clipfrac = torch.abs(values - old_values) > args.value_clip
values_clipped = old_values + (values - old_values).clamp(-args.value_clip, args.value_clip)
surr1 = (values_clipped - returns) ** 2
surr2 = (values - returns) ** 2
loss = torch.max(surr1, surr2)
```

**为什么这样写：** 它复用 PPO 的“保守更新”思想到 critic：允许 value 改进，但用 clipped 对照防止一次 step 大幅偏移。

**不变量与失败模式：** `values`、`old_values`、`returns` 必须 flatten 后顺序一致；`get_values` 负责从 logits 中提取 response 对齐的 value。

**Comment：** value loss 和 policy loss 共用 reducer 入口，因此它也能继承 per-token/per-rollout mean 的全局设置。

---

## 13. sft_loss_function

**Explain：** SFT loss 是 response token 上的负 log-likelihood，不计算 entropy，也不走 PPO 的 old/current 比较。

**问题与约束：** 同一训练后端要支持 RL loss 和 SFT loss；SFT 不需要 rollout logprob、advantage、returns 等 RL 字段。

**设计选择：** 复用 `get_log_probs_and_entropy` 的 logprob 提取能力，但 `with_entropy=False`；最终 `loss = -sum_of_sample_mean(log_probs)`。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/loss.py L1192-L1216
_, log_probs_and_entropy = get_log_probs_and_entropy(
    logits,
    args=args,
    unconcat_tokens=batch["unconcat_tokens"],
    total_lengths=total_lengths,
    response_lengths=response_lengths,
    with_entropy=False,
)
...
loss = -sum_of_sample_mean(log_probs)
if log_probs.numel() == 0:
    loss += 0 * logits.sum()
```

**为什么这样写：** 它把“token 对齐和分布式提取”复用起来，只替换目标函数；这比为 SFT 写一套独立切片逻辑更不容易和 RL 路径漂移。

**不变量与失败模式：** `unconcat_tokens` 必须能恢复每条样本的 response token；空 logprob 分支要保留 `0 * logits.sum()`，否则某些 shard 可能没有梯度图。

**Comment：** SFT 路径是最小化版本的 policy loss：没有 old policy，没有 KL，没有 entropy bonus。

---

## 14. loss_function — reducer 构造

**Explain：** 外层 `loss_function` 先构造统一 reducer，再根据 `loss_type` dispatch 到 policy/value/SFT/custom loss。

**问题与约束：** 不同 loss 类型要共享同一套 mask、response length、per-token/per-rollout mean 规则；否则日志和梯度缩放不可比。

**设计选择：** `num_tokens` 用 `clamp_min(loss_mask.sum(), 1)` 避免空 mask 分母为 0；`get_sum_of_sample_mean` 作为闭包传入内层 loss。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/loss.py L1254-L1262
num_tokens = sum([torch.clamp_min(loss_mask.sum(), 1) for loss_mask in batch["loss_masks"]])

sum_of_sample_mean = get_sum_of_sample_mean(
    batch["total_lengths"],
    batch["response_lengths"],
    batch["loss_masks"],
    batch["rollout_mask_sums"],
    args.calculate_per_token_loss,
)
```

**为什么这样写：** reducer 是 loss 层的统计口径，必须在 dispatch 前统一；内层 loss 只管产出逐 token 或逐样本张量。

**不变量与失败模式：** `batch["loss_masks"]` 与 `response_lengths` 必须描述同一批样本；`rollout_mask_sums` 若错误，会让 per-rollout mean 的分母偏移。

**Comment：** 这里的设计重点不是 `sum`，而是“把 reduction 作为依赖注入给 loss 函数”。

---

## 15. loss_function — Megatron 缩放与 allgather-CP 零 loss

**Explain：** 内层 loss 计算完成后，外层把它缩放成 Megatron 期望的梯度积累尺度，并处理 allgather-CP 空 shard 的反向图。

**问题与约束：** CP rank 可能没有 loss-contributing token；如果这个 rank 不走过 allgather 的 backward，其他 rank 的 reduce-scatter 可能等待。

**设计选择：** allgather-CP 且 CP world size 大于 1 时加 `0 * logits.sum()`，强制 autograd 遍历完整图但不改变梯度值。随后按 per-rollout 或 per-token 模式做不同缩放。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/loss.py L1281-L1320
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
...
```

**为什么这样写：** loss 层不仅服务数学正确性，还要服务分布式 collective 的调用一致性；`0 * logits.sum()` 是不改数值但保留图连通性的工程保护。

**不变量与失败模式：** allgather-CP 路径上所有 rank 必须参与 backward collective；per-token 模式和 per-rollout 模式不能混用同一个缩放公式。

**Comment：** 这段是 Slime loss 层作为 Megatron 适配器的核心证据。

---

## 16. test_cispo_loss — 闭式公式

**Explain：** 第一组 CISPO 测试验证实现等于闭式 surrogate：`-clamped_ratio * advantage * log_prob`。

**问题与约束：** CISPO 的公式差异很小，容易被误写成 PPO surrogate 或漏掉 ratio clamp。

**设计选择：** 测试直接构造 raw ratio、手算 clamp 后 ratio，再对 `pg_losses` 和 `clipfrac` 做 close check。

**代码逻辑：**

```python
## 来源：tests/test_cispo_loss.py L22-L31
ppo_kl = -torch.tensor([math.log(r) for r in ratios])

pg_losses, clipfrac = compute_cispo_loss(ppo_kl, LOG_PROBS, ADVANTAGES, eps_clip, eps_clip_high)

expected_losses = -torch.tensor(clamped) * ADVANTAGES * LOG_PROBS
torch.testing.assert_close(pg_losses, expected_losses, rtol=1e-6, atol=1e-6)
```

**为什么这样写：** 测试绕开模型和 batch，只验证数学 helper 的最小语义；这正好对应 helper 纯函数化的设计。

**不变量与失败模式：** `ppo_kl = -log(ratio)` 这个关系不能写反；否则 ratio 会变成倒数，测试会立刻暴露。

**Comment：** 这类测试比端到端训练更适合守住公式边界。

---

## 17. test_cispo_loss — 梯度路径

**Explain：** 第二组 CISPO 测试验证梯度只流向 `log_probs`，不流向 ratio/log-ratio。

**问题与约束：** CISPO 的核心不是只“裁剪 ratio”，而是裁剪后的 ratio 作为 stop-gradient 权重。

**设计选择：** 让 `log_ratios` 和 `log_probs` 都带 `requires_grad`，backward 后断言 `log_probs.grad` 符合闭式导数，`log_ratios.grad` 为 None 或 0。

**代码逻辑：**

```python
## 来源：tests/test_cispo_loss.py L35-L48
log_ratios = torch.tensor([math.log(r) for r in ratios], requires_grad=True)
ppo_kl = -log_ratios
log_probs = LOG_PROBS.clone().requires_grad_()

pg_losses, _ = compute_cispo_loss(ppo_kl, log_probs, ADVANTAGES, eps_clip, eps_clip_high)
pg_losses.sum().backward()

torch.testing.assert_close(log_probs.grad, -torch.tensor(clamped) * ADVANTAGES, rtol=1e-6, atol=1e-6)
assert log_ratios.grad is None or torch.all(log_ratios.grad == 0)
```

**为什么这样写：** 它直接守住 `ratio_truncated.detach()` 的设计意图；如果未来有人重构时删掉 detach，测试会失败。

**不变量与失败模式：** `log_probs` 必须是唯一承载梯度的策略输出；ratio 路径出现非零梯度说明实现不再是 CISPO 语义。

**Comment：** 这比只测 loss 数值更关键，因为两个实现可能前向值相同、反向语义不同。

---

## 18. compute_opsm_mask（Off-Policy Sequence Masking）

**Explain：** OPSM 先计算整条 response 的平均 KL，再在高 KL 且负 advantage 的位置上屏蔽 policy loss。

**问题与约束：** off-policy 样本中，负优势且分布偏移大的 token 容易给训练带来有害更新；但判断偏移需要完整 response 视角。

**设计选择：** `seq_kl` 用 full logprob 和 full mask 计算；`mask = (advantage < 0) & (seq_kl > delta)`，最终返回 `1 - mask` 乘到 `pg_loss`。

**代码逻辑：**

```python
## 来源：slime/utils/ppo_utils.py L54-L92
seq_kl = ((full_old_log_prob - full_log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)
mask = ((advantage < 0) & (seq_kl > args.opsm_delta)).float()
opsm_clipfrac += mask.sum() / torch.clamp_min(loss_mask.sum(), 1)

opsm_mask_list.append(1 - mask)
...
return opsm_mask, opsm_clipfrac
```

**为什么这样写：** 它把风险判断放在 sequence-level KL 上，把屏蔽动作落到与 `pg_loss` 对齐的位置张量上，兼容后续逐 token reducer。

**不变量与失败模式：** `advantage`、`loss_mask`、full logprob 必须来自同一条样本；如果只用 local shard 计算 `seq_kl`，会低估跨 CP 切分后的序列偏移。

**Comment：** 原笔记说“mask 整序列”不够精确；源码里 `seq_kl` 是序列级，`mask` 的形状跟 `advantage` 走。

---

## 19. _VocabParallelLogProbEntropy forward 要点

**Explain：** TP vocab 分片下，每个 rank 只持有部分 vocab logits；目标 token 不在本 rank 时先 mask，最后用 all-reduce 合并目标 logit 和 softmax 分母。

**问题与约束：** logprob/entropy 要在 tensor-parallel vocab 上计算，还要支持 top-p keep mask；如果 nucleus 外 logits 全置 `-inf`，target logit 不能被误伤。

**设计选择：** 对 `log_prob_keep_mask` 做 masked fill 后，把本 rank target 行的原始 logit 写回；softmax 之后再 all-reduce 目标 logit。

**代码逻辑：**

```python
## 来源：slime/utils/ppo_utils.py L261-L275
log_prob_logits = vocab_parallel_logits.masked_fill(~log_prob_keep_mask, float("-inf"))
if local_target_rows.numel() > 0:
    log_prob_logits[local_target_rows, masked_target_1d[local_target_rows]] = vocab_parallel_logits[
        local_target_rows, masked_target_1d[local_target_rows]
    ]
...
predicted_logits = predicted_logits.masked_fill_(target_mask, 0.0).unsqueeze(-1)
_maybe_all_reduce(predicted_logits, dist.ReduceOp.SUM, process_group)
log_prob = predicted_logits - log_prob_sum_exp_logits.log()
```

**为什么这样写：** top-p mask 控制归一化分母的候选集，但监督目标 token 的 numerator 仍必须存在；否则被采样出的 token 可能得到 `-inf` logprob。

**不变量与失败模式：** `target` 的 vocab 范围判断必须与 TP rank 的 vocab 切片一致；keep mask 不能覆盖 target numerator。

**Comment：** 这是“采样约束”和“训练目标”之间的边界处理：mask 可以改变分母，但不能删除目标。

---

## 20. get_grpo_returns — advantage 侧依赖

**Explain：** GRPO returns 把每条样本的 scalar reward broadcast 到对应 KL 张量形状。

**问题与约束：** advantage/return 侧要产生和 response token 对齐的张量，才能喂给 policy loss 的逐 token surrogate。

**设计选择：** 用 `torch.ones_like(kl[i]) * rewards[i]` 复用 KL 张量形状，不在这里重新推断长度。

**代码逻辑：**

```python
## 来源：slime/utils/ppo_utils.py L361-L368
def get_grpo_returns(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
):
    returns = []
    for i in range(len(rewards)):
        returns.append(torch.ones_like(kl[i]) * rewards[i])
    return returns
```

**为什么这样写：** GRPO 的 reward 是样本级，但 policy loss 消费 token 级张量；这里用 shape broadcast 完成粒度转换。

**不变量与失败模式：** `rewards` 顺序必须和 `kl` list 顺序一致；`kl[i]` 形状决定返回张量形状。

**Comment：** 这段不在 policy loss 主函数里，但解释了 `batch["advantages"]` 为什么可以是 response-aligned list。

---

## 21. get_advantages_and_returns_batch 调用 chunked_gae

**Explain：** batched GAE 默认走 `chunked_gae`，把长序列反向递推改成更适合并行的 chunk scan。

**问题与约束：** 标准 GAE 对时间维有反向依赖；长 response 上逐 token Python loop 会拖慢训练侧 advantage 计算。

**设计选择：** 在 batch 维 pad 到最大 response length 后，默认调用 `chunked_gae`；CP 下源码先 all-gather full reward/value，再 slice 回 local chunk。

**代码逻辑：**

```python
## 来源：slime/utils/ppo_utils.py L566-L609
if cp_size > 1:
    from slime.backends.megatron_utils.cp_utils import all_gather_with_cp
    ...
    full_v = all_gather_with_cp(v, total_len, resp_len)
    full_r = all_gather_with_cp(r, total_len, resp_len)
...
full_advantages, full_returns = chunked_gae(
    rewards=full_rewards,
    values=full_values,
    gamma=gamma,
    lambd=lambd,
)
```

**为什么这样写：** advantage 侧也遵循同一条设计原则：完整序列语义先恢复，计算后再回到 local 形状。

**不变量与失败模式：** `response_lengths` 决定 pad/slice 边界；如果 CP 下不先 gather，GAE 的跨 token 递推会在 shard 边界断开。

**Comment：** policy loss 看似只用 advantage，实际上 advantage 的计算方式决定了后续 surrogate 的学习信号质量。

---

## 22. chunked_gae — 并行化 GAE

**Explain：** `chunked_gae` 把反向 GAE 翻转成正向 scan，在 chunk 内用矩阵乘法并行计算，再用 `s_prev` 串联 chunk 状态。

**问题与约束：** GAE 递推 `S[t] = delta[t] + gamma*lambda*S[t+1]` 有序列依赖；完全串行会让长 response 成本高。

**设计选择：** 先计算 `deltas` 并翻转时间维；每个 chunk 构造上三角权重矩阵 `M`，chunk 间只传播最后状态 `s_prev`。

**代码逻辑：**

```python
## 来源：slime/utils/ppo_utils.py L666-L806
deltas = rewards + gamma * next_values - values
w = gamma * lambd
deltas_rev = torch.flip(deltas, dims=[1])
...
M[mask] = w ** diff[mask].to(dtype)
S_local_flat = deltas_flat @ M
...
S_global = S_local + s_prev.unsqueeze(1) * pow_vec[:Lc]
s_prev = S_global[:, -1]
advantages = torch.flip(S_rev, dims=[1])
returns = advantages + values
```

**为什么这样写：** 它把 O(T) 的逐步依赖压缩成“chunk 内并行 + chunk 间递推”，减少 Python 层顺序循环对长序列训练的影响。

**不变量与失败模式：** `rewards` 和 `values` 必须是 `[B, T]`；padding 必须在翻回原时间顺序前去掉。`w == 0` 单独处理，避免 `0 ** 0` 语义污染矩阵。

**Comment：** 这是源码里很清楚的性能哲学：不改变 GAE 数学定义，只改变计算组织方式。

---

## 23. loss_function logging_dict 分母占位

**Explain：** `logging_dict["values"][0]` 是后续日志聚合的分母位置；per-rollout mean 模式先放 0，占位给 `train_one_step` 替换成 step 级 batch size。

**问题与约束：** per-token 模式的分母来自实际 token 数；per-rollout mean 的分母是 step 级样本数，不适合从 microbatch 的局部分数一路 all-reduce。

**设计选择：** per-token 写入 `num_tokens`，per-rollout 写入 0；注释说明由消费方替换常数。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/loss.py L1305-L1318
"values": torch.tensor(
    [
        num_tokens if args.calculate_per_token_loss else 0,
    ]
    + list(log.values()),
    device=logits.device,
),
```

**为什么这样写：** 它避免把 per-rollout 的全局常数拆成 microbatch 分数再聚合，降低日志分母和训练缩放不一致的风险。

**不变量与失败模式：** 消费方必须知道 0 是占位，不是实际样本数；per-token 模式必须把真实 `num_tokens` 传出去。

**Comment：** 这是日志契约，不是训练 loss 数值本身。

---

## 24. reported_loss 中 OPD / mismatch 指标

**Explain：** TIS/mismatch 指标使用 pre-RS reducer 聚合；OPD reverse KL 只有在 batch 提供时才进入日志。

**问题与约束：** rejection-style mask 会改变参与反传的 token，但 mismatch 指标常常要描述原始 rollout 的偏移情况。

**设计选择：** `ois` 和 `tis_metrics` 用 `sum_of_sample_mean_for_mismatch_metrics`；`opd_reverse_kl` 作为可选 batch 字段追加。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/loss.py L1093-L1108
if args.get_mismatch_metrics or args.use_tis:
    reported_loss["ois"] = sum_of_sample_mean_for_mismatch_metrics(ois).clone().detach()
    for metric_key, metric_value in tis_metrics.items():
        key_name = f"{metric_key}"
        reported_loss[key_name] = sum_of_sample_mean_for_mismatch_metrics(metric_value)

if "opd_reverse_kl" in batch:
    opd_reverse_kl = torch.cat(batch["opd_reverse_kl"], dim=0)
    reported_loss["opd_reverse_kl"] = sum_of_sample_mean(opd_reverse_kl).clone().detach()
```

**为什么这样写：** 训练分子可以被 mask 修正，但诊断指标应尽量保留“原始 mismatch 有多严重”的观测口径。

**不变量与失败模式：** `tis_metrics` 的每个张量必须和原始 valid token 对齐；OPD 字段不存在时不能假设有该 metric。

**Comment：** 这段和 §9 合起来看，才能理解为什么源码保留两个 reducer。

---

## 25. train_rollout_logprob_abs_diff

**Explain：** 该指标度量训练侧 logprob 与 rollout logprob 的绝对差，只用于诊断，不参与 loss。

**问题与约束：** off-policy 训练最需要观察的是 rollout 分布和训练分布偏离多大；但这个偏离不能反向影响当前 step。

**设计选择：** 如果 batch 有 `rollout_log_probs`，就选择当前对比口径下的 logprob 与 rollout logprob 做绝对差并 reduce。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/loss.py L1073-L1077
if "rollout_log_probs" in batch and batch["rollout_log_probs"]:
    rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
    log_probs_to_compare = log_probs if args.use_rollout_logprobs else old_log_probs
    train_rollout_logprob_abs_diff = sum_of_sample_mean((log_probs_to_compare - rollout_log_probs).abs())
```

**为什么这样写：** 它把 mismatch 可观测化，但不把诊断项混进优化目标，避免隐式改变算法。

**不变量与失败模式：** 对比张量必须与 rollout logprob 同顺序 flatten；`args.use_rollout_logprobs` 会改变用于对比的 train/current 口径。

**Comment：** 这是排查 rollout lag、策略漂移和 TIS 效果时很重要的指标。

---

## 26. custom_pg_loss_reducer 注入点

**Explain：** Slime 允许只替换 policy loss 的 reducer，而不替换整个 policy loss 公式。

**问题与约束：** 有些实验只想改变 pg_loss 的聚合方式，例如按 sequence 加权，而不是重写 logprob、KL、entropy 和 metric 流程。

**设计选择：** 如果提供 `custom_pg_loss_reducer_function_path`，动态加载函数，并把 total/response lengths、mask、per-token 开关传给它。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/loss.py L1031-L1040
if getattr(args, "custom_pg_loss_reducer_function_path", None) is not None:
    custom_pg_loss_reducer_func = load_function(args.custom_pg_loss_reducer_function_path)
    pg_loss_masks = modified_response_masks if (args.get_mismatch_metrics or args.use_tis) else batch["loss_masks"]
    pg_loss_reducer = custom_pg_loss_reducer_func(
        total_lengths, response_lengths, pg_loss_masks, args.calculate_per_token_loss
    )
else:
    pg_loss_reducer = sum_of_sample_mean
```

**为什么这样写：** 它把最常实验的“聚合口径”做成窄扩展点，比开放整个 loss function 更安全。

**不变量与失败模式：** 自定义 reducer 必须返回可调用对象，并接受与 `pg_loss` 对齐的张量；TIS 后要使用 modified mask。

**Comment：** 这是 Slime loss 层“可扩展但不失控”的设计：扩展点窄，主流程稳定。

---

## 27. value_loss 空 values 梯度

**Explain：** value loss 在空 values 场景下加 `0 * values.sum()`，保留计算图连通性。

**问题与约束：** 分布式切分后某些 rank 可能没有本地 response/value token；直接返回无关标量会让该 rank 缺少必要的反向路径。

**设计选择：** 数值上不改变 loss，图上连接到 `values`。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/loss.py L1158-L1160
if values.numel() == 0:
    loss += 0 * values.sum()
```

**为什么这样写：** 它和 policy/SFT/allgather-CP 的零 loss 保护是一套思想：空 local shard 也要参与图和 collective。

**不变量与失败模式：** `values` 必须是从当前 logits 派生的张量；如果用常数零代替，会断开和模型输出的图关系。

**Comment：** 这类 `0 * tensor.sum()` 不是多余代码，而是分布式 autograd 保护。

---

## 28. recompute_loss_function checkpoint

**Explain：** `recompute_loss_function` 允许用 activation checkpoint 包住 loss 计算，以计算换显存。

**问题与约束：** policy loss 里 logprob、entropy、KL、TIS 可能保留较多中间张量；长序列训练时显存压力明显。

**设计选择：** 在外层 dispatch 后统一包 `checkpoint(func, args, batch, logits, sum_of_sample_mean, use_reentrant=False)`。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/loss.py L1276-L1277
if args.recompute_loss_function:
    loss, log = checkpoint(func, args, batch, logits, sum_of_sample_mean, use_reentrant=False)
```

**为什么这样写：** checkpoint 放在 loss 函数边界，覆盖 policy/value/SFT/custom loss，但不侵入各个 helper 的公式实现。

**不变量与失败模式：** 被 checkpoint 的函数必须在重算时可复现；如果自定义 loss 有不可重放副作用，会破坏 checkpoint 假设。

**Comment：** 这是性能开关，不改变 loss 语义。

---

## 29. get_reinforce_plus_plus_returns CP 四步

**Explain：** REINFORCE++ returns 在 CP 下先恢复完整 response KL，再把末 token 奖励加入 token-level reward，反向折扣后 slice 回 local chunk。

**问题与约束：** 折扣回报跨 response token 递推，不能在 CP-local shard 上孤立计算；末 token reward 也必须加在 full response 的最后有效位置。

**设计选择：** all-gather local KL → full mask 找最后有效 token → 反向计算 returns → `slice_log_prob_with_cp` 回到本 rank。

**代码逻辑：**

```python
## 来源：slime/utils/ppo_utils.py L405-L434
if cp_size > 1:
    full_kl_response = all_gather_with_cp(local_kl_chunk, total_len, response_len)
...
last_idx = full_mask.nonzero(as_tuple=True)[0][-1]
token_level_rewards[last_idx] += rewards[i]
...
if cp_size > 1:
    local_returns_chunk = slice_log_prob_with_cp(returns_for_seq, total_len, response_len)
```

**为什么这样写：** return 的时间依赖是 sequence-level 语义，CP 只是计算切分；因此必须先按语义恢复，再按并行布局切回。

**不变量与失败模式：** `full_mask.sum()` 必须大于 0；否则源码 assert 报错。`total_len` 和 `response_len` 必须与 CP slicing 规则一致。

**Comment：** 这段和 GSPO/OPSM 的 gather 逻辑互相呼应：凡是 sequence-level 语义，都不能只看 local shard。

---

## 走读小结

| 层次 | 代表函数 | 设计职责 |
|------|----------|----------|
| 数学 helper | `compute_policy_loss` / `compute_cispo_loss` / `compute_gspo_kl` | 产出逐位置张量，不做最终 reduction |
| policy 主干 | `policy_loss_function` | 对齐 logprob、选择 estimator、组合 entropy/KL/TIS/metrics |
| 训练适配 | `loss_function` | dispatch、checkpoint、Megatron 缩放、logging dict |
| 支撑 helper | `_VocabParallelLogProbEntropy` / `chunked_gae` | 处理 TP/CP、长序列与性能约束 |

读这条 loss 栈时，最重要的不是背 PPO 公式，而是持续追踪三个边界：**样本边界何时保留、序列语义何时需要 full gather、统计分母由哪个 reducer 定义**。
