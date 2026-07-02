---
type: batch-doc
module: 21-Loss-Advantages
batch: "21"
doc_type: faq
title: "Loss · Advantages · 关键问题"
tags:
  - slime/batch/21
  - slime/module/loss-advantages
  - slime/doc/faq
updated: 2026-07-02
---

# Loss · Advantages · 关键问题

---

## Q1：为什么 advantage 要在 train backward 之前整批算完？

**Explain：** `normalize_advantages` 需要 **整个 rollout batch**（跨 DP）的 masked 均值和方差。若在 micro-batch 循环内逐块算，whitening 统计不一致。PPO 的 GAE 也在 sample 级需要完整 response 上的 value/reward 序列（CP 下先 all_gather 再算）。

---

## Q2：`use_rollout_logprobs` 与训练侧 `log_probs` 怎么选？

| 场景 | 建议 | 原因 |
|------|------|------|
| 严格 on-policy、引擎与训练 logprob 一致 | 训练侧重算 | 避免引擎/训练数值差 |
| 大 batch 省一次 forward | `use_rollout_logprobs` | 跳过 `compute_log_prob` |
| TIS / mismatch metrics | 必须存 rollout + train 两套 | 见 [[22-Loss-Policy-04-关键问题]] |

**易错：** `use_rollout_logprobs` 与 `use_tis` **互斥**（arguments 断言）。

**Code：**

```python
# 来源：arguments.py L1804–1805
    if args.use_rollout_logprobs:
        assert not args.use_tis, "use_rollout_logprobs and use_tis cannot be set at the same time."
```

---

## Q3：`kl_coef==0` 为什么还要构造 `kl` 列表？

**Explain：** GRPO/REINFORCE 等分支用 `kl[i]` 的 **shape/device** 广播 reward；若 `log_probs` 也为空（非 last stage 或中间态），fallback 到 `values`。零 KL 保证列表长度与 sample 数一致。

**Code：**

```python
# 来源：loss.py L700–703
    if args.kl_coef == 0 or not log_probs:
        xs = log_probs or rollout_log_probs or values
        kl = [torch.zeros_like(x, dtype=torch.float32, device=x.device) for x in xs]
```

---

## Q4：PPO 为什么只在 `cp_rank==0` 加 reward？

**Explain：** Context Parallel 把 **同一 response** 切到多 rank；环境 reward 是 **序列级标量**，只能加一次。实现上选择 rank 0 的本地 chunk 末 token（与 `k[-1]` 对齐）。

**Code：**

```python
# 来源：loss.py L731–735
        for reward, k in zip(old_rewards, kl, strict=False):
            k *= kl_coef
            if cp_rank == 0:
                k[-1] += reward
            rewards.append(k)
```

---

## Q5：OPD 开启但报错 `teacher_log_probs missing`？

**检查清单：**

1. `--use-opd` 与 `opd_type` 是否匹配 Megatron teacher vs rollout teacher
2. Rollout：`post_process_rewards` 是否写入 sample；[[20-Train-Data-00-MOC]] 是否 tensorize 到 batch
3. Megatron：`weights_backuper` 是否含 `"teacher"` tag 且已 `store_prefix="teacher_"` forward
4. student/teacher logprob **长度** 是否与 `response_lengths` 一致

**Code（失败路径）：**

```python
# 来源：loss.py L644–646
    if teacher_log_probs is None:
        raise ValueError(f"OPD with opd_type='{args.opd_type}' requires teacher_log_probs, but it is missing.")
```

---

## Q6：`normalize_advantages` 与 OpenRLHF / veRL 差异？

**Explain：** 代码注释（L775）指出 OpenRLHF 常做 advantage norm，veRL 未必。Slime 默认由 CLI 控制；`reinforce_plus_plus*` **强制**开启。CP 下必须重建 mask chunk，否则会触发 shape assert：

```python
# 来源：loss.py L813–815
            assert (
                all_advs.size() == all_masks.size()
            ), f"Shape mismatch before whitening: advantages {all_advs.size()}, masks {all_masks.size()}"
```

---

## Q7：`rollout_top_p != 1.0` 训练报错缺字段？

**Explain：** 必须在 rollout 侧记录 nucleus token；训练 `get_rollout_top_p_logprob_kwargs` 硬校验。

**正确：** rollout 配置开启 top-p 记录 → `_convert_samples_to_train_data` 写入 ids/offsets → policy forward 传入 kwargs。

**错误：** 仅训练侧设 `rollout_top_p` 而无 rollout 数据。

---

## Q8：非 last PP stage 上 advantages 为空？

**Explain：** 预期行为。Last stage 写入 `rollout_data` 后由 Megatron 广播或下一 microbatch 仅 last stage 消费 loss。Debug 时在 **last stage rank** 打印 `rollout_data["advantages"]`。

---

## Q9：custom_advantage_function 契约是什么？

**Explain：** 在 KL 写入后、内置分支前被调用；必须 **自行填充** `rollout_data["advantages"]` 与 `rollout_data["returns"]`。之后仍走 OPD 与 `normalize_advantages`（除非 custom 函数改 args 行为）。

---

## Q10：与批次 22 的边界？

| 本批（21） | 批次 22 |
|-----------|---------|
| `compute_advantages_and_returns` | `policy_loss_function` |
| `get_log_probs`（forward_only / 无梯度或 detach 语境） | 带梯度的 policy logprob + clip |
| `get_values` 提取 | `value_loss_function` |
| OPD 改 advantage | OPD metric 上报 |

---

## 调试建议

1. 在 `compute_advantages_and_returns` 末尾临时 log：`rewards[0]`, `advantages[0].mean()`, `kl[0].mean()`
2. 对比 `use_rollout_logprobs` 与 train logprob：`train_rollout_logprob_abs_diff`（policy loss metrics）
3. 跑 `tests/test_chunked_gae.py` 验证 GAE 与 chunk 边界
