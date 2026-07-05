---
type: batch-doc
module: 23-CP-RoutingReplay
batch: "23"
doc_type: faq
title: "CP · Routing Replay · 关键问题"
tags:
  - slime/batch/23
  - slime/module/cp-routing-replay
  - slime/doc/faq
updated: 2026-07-02
---

# CP · Routing Replay · 关键问题

---

## Q1：为何 all_gather_with_cp 用 all_reduce 而非 all_gather？

**Explain：** 各 rank 构造的 full 张量 **互斥非零**（其余为零），sum reduce 等价于 gather。且 zero 区需 `requires_grad=True` 接 autograd。

---

## Q2：CP rank 无 loss token 会死锁？

**Explain：** `allgather_cp` 下可能出现；`loss_function` 加 `0 * logits.sum()` 强制全图 backward。见 [[22-Loss-Policy-04-关键问题]] Q8。

---

## Q3：`cu_seqlens * cp_size` 和 zigzag 何关系？

**Explain：** Megatron THD packed API 用 **逻辑 token 数**；物理 tensor 长度是 CP chunk 之和。乘以 cp_size 把物理长度还原为逻辑 cu_seqlens。

---

## Q4：record 与 replay_forward 能否混用？

**Explain：** 同一训练步内 actor 先 replay_forward（logprob）再 record 会 **追加** 列表；`clear_all_forward` 在 logprob 后重置 forward_index 供 backward。顺序由 actor 严格控制。

---

## Q5：ref 为何必须 fallthrough？

**Explain：** 参考策略不参与 rollout routing 对齐；对其 replay rollout experts 无意义且破坏 KL 语义。

---

## Q6：routing replay 需要 Megatron patch 吗？

**Explain：** 是。upstream Megatron 不含 hook；Slime docker patch 注入 `get_routing_replay_compute_topk`。

---

## Q7：per-rollout mean 在 CP 下分母是否 × cp_size？

**Explain：** **不**。`rollout_mask_sums` 在 full mask 上预计算；local reducer 用同一 denom，分子只累加 local masked sum。metric gather 时 `rollout_log_metric_contribution` 乘 `cp_size` 修正 rank 局部和。

---

## Q8：tests/test_loss_cp_invariance 测什么？

**Explain：** CP 宽度变化时 **标量 loss** 与无 CP 参考一致（在数值误差内），验证 `get_sum_of_sample_mean` + rescale 自洽。

---

## Q9：DSA allgather_cp 与 zigzag 能同时开吗？

**Explain：** 由 `args.allgather_cp` 切换 **get_batch** 路径；CP utils 仍用于 logprob。配置互斥由 arguments 校验（见 arguments.py）。

---

## Q10：ROUTING_REPLAY 全局变量线程安全吗？

**Explain：** 训练为单进程单线程 forward；pre_hook 在每 layer 前设置。不支持多 stream 并发 forward。

---

## Q11：routing replay 与 `forward_only` 是什么关系？

**Explain：** 二者是 **正交职责**：

| 维度 | `forward_only` | routing replay |
|------|----------------|----------------|
| 作用 | Megatron pipeline **无 backward** 的 eval 前向（logprob / values） | MoE `compute_topk` 用 record/replay 固定 expert 路径 |
| stage 谁管 | **不管**——读 actor 事先写入的 `ROUTING_REPLAY_STAGE` | actor + `train_one_step` 闭包分工（§3 阶段表） |
| 典型调用 | `compute_log_prob` → ref/teacher/actor logprob | 同上 logprob + policy `train()` |

**关键细节：**

1. `forward_only`（model.py L345–506）只有 `get_batch` + `model(**forward_kwargs)`，**没有** L602–636 的 stage 切换。
2. Policy train 走 `train_one_step`（model.py L509+）：actor 设 `replay_backward` 后，闭包 forward 段临时 `replay_forward` 再恢复，backward 段用 `replay_backward`（[[23-CP-RoutingReplay-02-源码走读]] §20）。
3. `use_rollout_routing_replay` 时 `fill_routing_replay` 在 **任何** `forward_only` 之前预填 indices；logprob 后 `clear_all_forward()` 供 train forward 再次 `pop_forward`。

**Comment：** 勿把「eval 模式」与「fallthrough stage」混为一谈——ref logprob 虽走 `forward_only` + `model.eval()`，但 stage 是 `fallthrough` 而非 replay。
