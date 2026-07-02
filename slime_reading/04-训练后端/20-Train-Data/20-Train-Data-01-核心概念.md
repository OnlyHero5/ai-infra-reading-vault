---
type: batch-doc
module: 20-Train-Data
batch: "20"
doc_type: concept
title: "Train Data · 核心概念"
tags:
  - slime/batch/20
  - slime/module/train-data
  - slime/doc/concept
updated: 2026-07-02
---

# Train Data · 核心概念

> 本批关注 **训练侧如何吃 Rollout 数据**：不是 prompt 从哪来（[[11-DataSource-00-MOC]]），而是 tensor 化之后如何分区、打包、喂给 Megatron forward。

---

## 1. RolloutBatch 与调度字段

| 字段 | 含义 | 写入方 |
|------|------|--------|
| `tokens` | 每样本 1D token id 列表 | RolloutManager |
| `total_lengths` | 每样本 token 数 | split 前全局；Actor 按 `partition` 重排 |
| `micro_batch_indices` | `list[list[list[int]]]`，DP rank → mbs → local 样本下标 | `build_dp_schedule` |
| `num_microbatches` | 每 training step 每 rank 的 mbs 数（PP 同步要求各 rank 相同） | 同上 |
| `global_batch_sizes` | 每 step 含多少 **rollout**（非 sample 数） | 同上 |
| `rollout_mask_sums` | 每 rollout 组内 loss_mask 总和（per-rollout mean 分母） | RolloutManager |
| `partition` | 本 DP rank 的全局 sample 下标（Ray 传输用，Actor pop 掉） | split 打包 |

**Explain：** `RolloutBatch` 是 Actor 上 `train()` / `forward_only` 的共享字典；`DataIterator` 只按 `micro_batch_indices` 切片，不复制大张量。

**Code：**

```python
# 来源：megatron_utils/data.py L201-L217
class DataIterator:
    """Iterator over a rollout dict following an explicit micro-batch index schedule."""

    def __init__(
        self,
        rollout_data: RolloutBatch,
        micro_batch_indices: list[list[int]],
    ) -> None:
        self.rollout_data = rollout_data
        self.micro_batch_indices = micro_batch_indices
        self.offset = 0
```

---

## 2. pack-first-distribute-second

**Explain：** `dp_schedule.py` 的哲学是 **先在 step 内把样本打成 micro-batch，再把 mbs 分给 DP rank**，而不是先按 rank 切样本再各自 pack。这样：

- 每个 DP rank 的 **mbs 数量一致**（PP 需要）
- dynamic batch 下 token 上限 `max_tokens_per_gpu * cp_size` 在 **全局 first-fit** 阶段 enforce

**Code：**

```python
# 来源：dp_schedule.py L8-L23（模块 docstring 节选）
# The scheduling philosophy is **pack first, distribute second**:
#   1. Group samples by rollout id ...
#   2. For each step, pack its samples into K micro-batches ...
#   3. Adjust K to a multiple of dp_size * (mb_group if vpp>1 else 1) ...
#   4. Distribute the K mbs across dp_size ranks ...
```

---

## 3. Context Parallel 与 `get_batch`

两种 CP 布局：

| 模式 | 触发 | token 处理 |
|------|------|------------|
| 默认 zigzag | `allgather_cp=False` | 每样本 `slice_with_cp` 后 cat |
| DSA / allgather | `allgather_cp=True` | 先 cat 全序列，再按 cp_rank chunk 一次 |

**Explain：** `unconcat_tokens` 保留 CP 切片前的原始 list，供 `get_log_probs_and_entropy` 在 full 序列上算 logprob（见 [[23-CP-RoutingReplay-00-MOC]]）。

**Code：**

```python
# 来源：megatron_utils/data.py L63-L64, L88-L90
    batch["unconcat_tokens"] = tokens
    ...
    else:
        tokens = [slice_with_cp(t, pad_token_id) for t in tokens]
```

---

## 4. loss_mask 与 token 流对齐

**Explain：** response 上的 `loss_mask` 长度等于 response token 数；packed 流上需在 **prompt_length-1 左 pad、1 右 pad**，使 mask 与 shifted logprob 位置对齐。

**Code：**

```python
# 来源：megatron_utils/data.py L122-L130
        prompt_length = total_length - response_length
        loss_mask = F.pad(loss_mask, (prompt_length - 1, 1), value=0)
```

---

## 5. `process_rollout_data`：Actor 还原 partition

**Explain：** Ray 每个 DP rank 收到一个 `Box`；`partition` 是 **全局** sample 下标列表。Actor 弹出 `partition` 后，用其重排 `total_lengths`，并把完整 batch 的 seqlen 记入 `Timer` 供 perf 统计。

**Code：**

```python
# 来源：utils/data.py L292-L303
def process_rollout_data(args, rollout_data_ref, dp_rank, dp_size):
    assert len(rollout_data_ref) == dp_size
    rollout_data = ray.get(rollout_data_ref[dp_rank].inner)

    partition = rollout_data.pop("partition")
    total_lengths = rollout_data["total_lengths"]

    Timer().seq_lens = total_lengths
    rollout_data["total_lengths"] = [total_lengths[i] for i in partition]

    return rollout_data
```

---

## 6. 序列长度平衡算法

| 函数 | 用途 |
|------|------|
| `get_seqlen_balanced_partitions` | Karmarkar-Karp 多路划分，最小化各 partition 的 seqlen 和 spread |
| `first_fit_pack` | dynamic batch：bin packing，bin 和 ≤ `max_tokens_per_bin` |
| `expand_bins_by_splitting` | mbs 数不足对齐倍数时，拆分最大多样本 bin |

**Code：**

```python
# 来源：seqlen_balancing.py L146-L161
def get_seqlen_balanced_partitions(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    """get order of seq lengths to make partitions balanced ..."""
    assert len(seqlen_list) >= k_partitions
    partitions = karmarkar_karp(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size)
    return _check_and_sort_partitions(partitions)
```

---

## 7. 与训练 loss 归一化的关系

`get_batch` 只构造 forward 输入；**per-rollout mean** 的分母 `rollout_mask_sums` 在 Rollout 侧写入，在 `loss_function` 里传给 `get_sum_of_sample_mean`（批次 21–22）。本批只需记住：**调度按 rollout id 分 step，loss 归一化按 rollout 组聚合**。

---

## 8. 选型速查

| 配置 | 行为 |
|------|------|
| `use_dynamic_batch_size=True` | first-fit + 可 split bin 对齐 mbs |
| `balance_by_flops=True` | 用 FLOPs 估计 pack（不保证 token cap） |
| `balance_data=True` | mbs 按 FLOPs 分给 DP rank（KK，`equal_size=True`） |
| 默认 | mbs 按 stride `range(r, K, dp_size)` 轮询分配 |
