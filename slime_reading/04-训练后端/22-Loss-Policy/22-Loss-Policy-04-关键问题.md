---
type: batch-doc
module: 22-Loss-Policy
batch: "22"
doc_type: faq
title: "Loss · Policy · 关键问题"
tags:
  - slime/batch/22
  - slime/module/loss-policy
  - slime/doc/faq
updated: 2026-07-02
---

# Loss · Policy · 关键问题

---

## Q1：PPO 与 CISPO 选型？

| | PPO | CISPO |
|---|-----|-------|
| clip 对象 | ratio 进入 `max/min` surrogate | ratio **detach** 后乘 `log_probs` |
| 被 clip token 的梯度 | 通常无（clip 分支） | **仍有**（经 log_probs） |
| 典型配置 | `advantage_estimator=ppo` | `advantage_estimator=cispo`，常 `eps_clip>=1` 禁下界 |

运行：`pytest tests/test_cispo_loss.py -q`

---

## Q2：GRPO / gspo 与默认 PPO 在 loss 侧差在哪？

**Explain：** GRPO 的 advantage 在[[21-Loss-Advantages-00-MOC]] 用 group baseline；loss 侧若 `advantage_estimator=="gspo"`，**额外**用序列级 KL expand 到 token（`compute_gspo_kl`），而非 per-token `old - new`。

---

## Q3：`can_reuse_log_probs_in_loss` 何时为真？

**Explain：** actor 在极窄条件下可跳过 train 前第二次 logprob forward，直接复用 backward 里的 logprob（见 `actor.py` L467-L477）。启用 TIS、critic、routing replay、GSPO 等任一则为 false。

---

## Q4：TIS 为何 metrics 用 pre-RS mask？

**Explain：** rejection sampling 会把 token 从 **loss 分母** 剔除；若用 modified mask 聚合 `truncate_fraction`， rejected token 不在分母，指标可被人为压到 0。见 `policy_loss_function` 注释 L992-L995。

---

## Q5：空 batch 为何 `loss += 0 * logits.sum()`？

**Explain：** allgather-CP 下部分 rank 无有效 token，需保持 autograd 连通 CP gather，避免 backward deadlock（policy / value / sft / loss_function 均有类似 guard）。

---

## Q6：KL loss 与 advantage 里 KL 惩罚区别？

- **advantage 阶段** `kl_coef`：从 reward 减 KL（PPO/GRPO 路径，[[21-Loss-Advantages-00-MOC]]）
- **`use_kl_loss`**：在 policy loss 上加 `kl_loss_coef * KL(ref)`，可用 `use_unbiased_kl` IS 校正

二者可同时开启，需小心系数叠加。

---

## Q7：验证命令

```bash
cd slime && pytest tests/test_cispo_loss.py tests/test_ppo_logprob_entropy_gpu.py -q
```

---

## Q8：vanilla_tis / ICEPOP 与 `--use-tis` 怎么配？

**Explain：** Slime **没有** `--icepop` 开关。TIS 族实现三选一：`vanilla_tis_function`（默认）、`icepop_function`（硬拒绝）、或任意 `--custom-tis-function-path`。启用条件与互斥由 `arguments.py` 断言 + `policy_loss_function` 分支共同决定。

| 开关 / 函数 | 作用 | 与其他参数关系 |
|-------------|------|----------------|
| `--use-tis` | 对 `pg_loss` 乘 importance 权重 | 与 `--use-rollout-logprobs` **互斥**；需 batch 含 `rollout_log_probs` |
| `--get-mismatch-metrics` | 只算 mismatch 指标也走 TIS 路径 | **必须** 同时设 `--custom-tis-function-path` |
| （默认）`vanilla_tis_function` | `clamp(ratio)` 后乘到 `pg_loss` | `use_tis=True` 且未设 custom path 时选用 |
| `icepop_function` | 区间外 ratio **置零** | 经 `--custom-tis-function-path slime.backends.megatron_utils.loss:icepop_function` 选用 |
| `--tis-clip` / `--tis-clip-low` | ratio 上下界 | vanilla 与 ICEPOP **共用** |

**Code（CLI 定义 + validate 断言）：**

```python
## 来源：slime/utils/arguments.py L1048-L1070
            parser.add_argument(
                "--use-tis",
                action="store_true",
                default=False,
                help="Enable TIS from https://fengyao.notion.site/off-policy-rl for off-policy importance sampling.",
            )
            parser.add_argument(
                "--tis-clip",
                type=float,
                default=2.0,
                help="Clipping threshold C for importance sampling ratios to control variance.",
            )
            parser.add_argument(
                "--tis-clip-low",
                type=float,
                default=0,
                help="Lower bound clipping threshold C for importance sampling ratios to control variance.",
            )
            parser.add_argument(
                "--custom-tis-function-path",
                type=str,
                default=None,
                help="Path to the custom TIS/RS function (e.g., examples/train_infer_mismatch_helper/mis.py:compute_mis_weights_with_cp).",
            )
```

```python
## 来源：slime/utils/arguments.py L1804-L1810
    if args.use_rollout_logprobs:
        assert not args.use_tis, "use_rollout_logprobs and use_tis cannot be set at the same time."

    if args.get_mismatch_metrics:
        assert (
            args.custom_tis_function_path is not None
        ), "custom_tis_function_path must be set when get_mismatch_metrics is set"
```

**Code（运行时选用 vanilla vs custom / ICEPOP）：**

```python
## 来源：loss.py L987-L1014
    if args.get_mismatch_metrics or args.use_tis:
        assert "rollout_log_probs" in batch, "rollout_log_probs must be provided for TIS"
        ...
        if args.custom_tis_function_path is not None:
            tis_func = load_function(args.custom_tis_function_path)
        else:
            tis_func = vanilla_tis_function
        pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)
```

**Comment：** ICEPOP 与 vanilla **不是** 互斥 CLI，而是同一 hook 点的两种实现；`get_mismatch_metrics` 单独开启时不能落回 vanilla（必须显式 custom path）。更多 `*-path` 约定见 [[04-Arguments-TrainRollout-01-核心概念]]。

---

## Q9：与 OpenRLHF / verl  lineage

`ppo_utils.py` 头部注明改编自 OpenRLHF；Slime 扩展了 vocab parallel autograd、GSPO/CISPO、TIS、CP gather。
