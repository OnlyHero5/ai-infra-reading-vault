---
type: batch-doc
module: 20-Train-Data
batch: "20"
doc_type: walkthrough
title: "Train Data · 源码走读"
tags:
  - slime/batch/20
  - slime/module/train-data
  - slime/doc/walkthrough
updated: 2026-07-02
---

# Train Data · 源码走读

> 走读顺序：`build_dp_schedule` → `process_rollout_data` → `get_batch` → `DataIterator` → `log_rollout_data`  
> 基线 commit `22cdc6e1`

---

## 1. build_dp_schedule — 按 rollout 分步再 pack mbs

### 1.1 模块级不变量

**Explain：** `dp_schedule.py` 是纯 Python 调度器，不 import Ray/SGLang，便于 CPU CI 单测。核心不变量：各 DP rank 每 step 的 `num_microbatches` 相同；dynamic batch 下除超长单样本外，每个 mbs token 和 ≤ `max_tokens_per_gpu * cp_size`。

**Code：**

```python
# 来源：dp_schedule.py L82-L111
def build_dp_schedule(
    args: Any,
    train_parallel_config: dict,
    total_lengths: list[int],
    *,
    global_batch_size: int,
    rollout_indices: list[int],
) -> tuple[list[list[int]], list[list[list[int]]], list[int], list[int]]:
    dp_size = train_parallel_config["dp_size"]
    cp_size = train_parallel_config["cp_size"]
    vpp_size = train_parallel_config["vpp_size"]
    mb_group = train_parallel_config["microbatch_group_size_per_vp_stage"]
    ...
    return partitions, micro_batch_indices, num_microbatches, global_batch_sizes
```

**Comment：** `global_batch_size` 计数单位是 **rollout**（`Sample.index`），不是 training sample 数；compact/subagent 下一 rollout 可对应多样本，但同 id 样本必在同一步。

### 1.2 按 rollout id 分组

**Explain：** 先建 `rollout_id_to_samples`，再按 `rollout_ids` 顺序每 `global_batch_size` 个 rollout 切一步。尾部不足一整步的 rollout 被丢弃（`num_steps = len(rollout_ids) // global_batch_size`）。

**Code：**

```python
# 来源：dp_schedule.py L127-L148
    rollout_id_to_samples: dict[int, list[int]] = {}
    for sample_pos, rid in enumerate(rollout_indices):
        rollout_id_to_samples.setdefault(rid, []).append(sample_pos)
    rollout_ids = list(rollout_id_to_samples.keys())

    num_steps = len(rollout_ids) // global_batch_size
    ...
    for step_i in range(num_steps):
        step_rollouts = rollout_ids[step_i * global_batch_size : (step_i + 1) * global_batch_size]
        sample_indices = [pos for rid in step_rollouts for pos in rollout_id_to_samples[rid]]
        step_lengths = [total_lengths[i] for i in sample_indices]
```

**Comment：** 与 per-rollout loss 归一化（`rollout_mask_sums`）对齐：一步内所有 sibling 样本共享同一 rollout 组分母。

### 1.3 _pack_step_into_mbs

**Explain：** dynamic 路径用 `first_fit_pack`；`balance_by_flops=True` 时用 FLOPs 估计做 KK 划分（**不保证** token cap）。static 路径按固定 `micro_batch_size` 切块。

**Code：**

```python
# 来源：dp_schedule.py L55-L79
def _pack_step_into_mbs(
    step_lengths: list[int],
    *,
    args: Any,
    use_dynamic_batch_size: bool,
    max_per_bin: int | None,
    micro_batch_size: int | None,
    balance_by_flops: bool = False,
) -> list[list[int]]:
    if use_dynamic_batch_size:
        assert max_per_bin is not None
        if balance_by_flops:
            total_tokens = sum(step_lengths)
            num_mbs = max(1, (total_tokens + max_per_bin - 1) // max_per_bin)
            if num_mbs >= len(step_lengths):
                return [[i] for i in range(len(step_lengths))]
            workloads = _calculate_workloads(step_lengths, args)
            return get_seqlen_balanced_partitions(workloads, num_mbs, equal_size=False)
        return first_fit_pack(step_lengths, max_per_bin)
    assert micro_batch_size is not None
    n = len(step_lengths)
    return [list(range(i, min(i + micro_batch_size, n))) for i in range(0, n, micro_batch_size)]
```

### 1.4 mbs 数对齐与 DP 分配

**Explain：** `align_to = dp_size * (mb_group if vpp>1 else 1)`。dynamic 下 `expand_bins_by_splitting` 把最大多样本 bin 二分直到 `K == target_K`；static 下若不能整除则 **AssertionError**（不能 split 固定 mbs）。

**Code：**

```python
# 来源：dp_schedule.py L167-L207
        target_K = max(((len(step_mbs) + align_to - 1) // align_to) * align_to, align_to)
        if target_K != len(step_mbs):
            if args.use_dynamic_batch_size:
                expand_bins_by_splitting(step_mbs, target_K, step_lengths)
            else:
                raise AssertionError(...)
        K = len(step_mbs)
        num_mbs_per_rank = K // dp_size
        if args.balance_data:
            step_workloads = _calculate_workloads(step_lengths, args)
            mbs_weights = [sum(step_workloads[i] for i in bin_) for bin_ in step_mbs]
            rank_mbs_idx = get_seqlen_balanced_partitions(mbs_weights, dp_size, equal_size=True)
        else:
            rank_mbs_idx = [list(range(r, K, dp_size)) for r in range(dp_size)]
        for r in range(dp_size):
            for mbs_idx in rank_mbs_idx[r]:
                mbs_locals = step_mbs[mbs_idx]
                local_start = len(partitions[r])
                partitions[r].extend(sample_indices[i] for i in mbs_locals)
                micro_batch_indices[r].append(list(range(local_start, local_start + len(mbs_locals))))
```

---

## 2. seqlen_balancing — Karmarkar-Karp 与 first-fit

### 2.1 get_seqlen_balanced_partitions

**Explain：** 默认用 Karmarkar-Karp（最大差分法）最小化各 partition 的 seqlen 和 spread。`equal_size=True` 时要求 `len(seqlen_list) % k_partitions == 0`。

**Code：**

```python
# 来源：seqlen_balancing.py L146-L177
def get_seqlen_balanced_partitions(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    assert len(seqlen_list) >= k_partitions
    partitions = karmarkar_karp(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size)
    return _check_and_sort_partitions(partitions)
```

### 2.2 first_fit_pack 与 expand_bins_by_splitting

**Code：**

```python
# 来源：seqlen_balancing.py L180-L229
def first_fit_pack(total_lengths, max_tokens_per_bin):
    bins: list[list[int]] = []
    bin_sums: list[int] = []
    for idx, length in enumerate(total_lengths):
        for j in range(len(bins)):
            if bin_sums[j] + length <= max_tokens_per_bin:
                bins[j].append(idx)
                bin_sums[j] += length
                break
        else:
            bins.append([idx])
            bin_sums.append(length)
    return bins

def expand_bins_by_splitting(bins: list[list[int]], target_count: int, lengths) -> None:
    while len(bins) < target_count:
        candidates = [(sum(lengths[i] for i in b), idx) for idx, b in enumerate(bins) if len(b) > 1]
        if not candidates:
            break
        _, idx = max(candidates)
        left, right = _split_bin_by_tokens(bins[idx], lengths)
        bins[idx] = left
        bins.append(right)
```

**Comment：** 超长单样本单独占一 bin，该 bin 可超过 `max_tokens_per_bin`——这是唯一允许的例外。

---

## 3. process_rollout_data — Actor 侧还原 partition

**Explain：** RolloutManager 为每个 DP rank 打包 `Box`，内含 `partition`（全局 sample 下标）。Actor `ray.get` 后弹出 `partition`，按此重排 `total_lengths`，并把 **全局** seqlen 列表记入 `Timer` 供 perf 统计。

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

**Comment：** `args` 参数保留供扩展；当前实现未使用。`micro_batch_indices` 已是 rank-local 下标，无需再 permute。

---

## 4. get_batch — CP-ready packed micro-batch

### 4.1 入口与 unconcat_tokens

**Explain：** `DataIterator.get_next` 返回 per-sample token **list**；`get_batch` 将其变为 `[1, T_padded]` 与 Megatron `PackedSeqParams`（THD layout）。`unconcat_tokens` 保留 CP 切片前副本，供 loss 在 full 序列上算 logprob。

**Code：**

```python
# 来源：megatron_utils/data.py L28-L64
def get_batch(
    data_iterator: "DataIterator",
    keys: Sequence[str],
    pad_multiplier: int = 128,
    allgather_cp: bool = False,
) -> dict[str, torch.Tensor | PackedSeqParams | list[torch.Tensor] | None]:
    assert "tokens" in keys
    batch = data_iterator.get_next(keys)
    tokens = batch["tokens"]
    pad_token_id = 0
    pad_size = mpu.get_tensor_model_parallel_world_size() * pad_multiplier
    batch["unconcat_tokens"] = tokens
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
```

### 4.2 默认 CP zigzag 路径

**Code：**

```python
# 来源：megatron_utils/data.py L88-L104
    else:
        tokens = [slice_with_cp(t, pad_token_id) for t in tokens]
        cu_seqlens = [0]
        for t in tokens:
            cu_seqlens.append(cu_seqlens[-1] + t.size(0))
        tokens = torch.cat(tokens)
        pad = (pad_size - tokens.size(0) % pad_size) % pad_size
        if pad != 0:
            tokens = F.pad(tokens, (0, pad), value=pad_token_id)
            cu_seqlens.append(cu_seqlens[-1] + pad)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int).cuda() * cp_size
```

**Comment：** `cu_seqlens * cp_size` 是 Megatron THD CP 约定：逻辑长度 = 物理 chunk 长度 × cp_size。

### 4.3 allgather_cp（DSA）路径

**Explain：** 先 cat 全序列，全局 pad 使长度可被 `cp_size * pad_size` 整除，再 `chunk(cp_size)[cp_rank]` 一次切片。loss_mask 同样 cat → pad → chunk。

**Code：**

```python
# 来源：megatron_utils/data.py L69-L87, L137-L142
    if allgather_cp:
        cu_seqlens_list: list[int] = [0]
        for t in tokens:
            cu_seqlens_list.append(cu_seqlens_list[-1] + t.size(0))
        tokens = torch.cat(tokens, dim=0)
        global_pad_size = cp_size * pad_size
        pad = (global_pad_size - tokens.size(0) % global_pad_size) % global_pad_size
        ...
        tokens = tokens.chunk(cp_size, dim=0)[cp_rank]
    ...
    if allgather_cp:
        loss_masks = torch.cat(loss_masks, dim=0)
        if pad != 0:
            loss_masks = F.pad(loss_masks, (0, pad), value=0)
        loss_masks = loss_masks.chunk(cp_size, dim=0)[cp_rank].unsqueeze(0)
```

### 4.4 loss_mask 对齐 packed 流

**Code：**

```python
# 来源：megatron_utils/data.py L121-L147
    for loss_mask, total_length, response_length in zip(
        batch["loss_masks"], batch["total_lengths"], batch["response_lengths"], strict=True,
    ):
        prompt_length = total_length - response_length
        loss_mask = F.pad(loss_mask, (prompt_length - 1, 1), value=0)
        if allgather_cp:
            loss_masks.append(loss_mask)
            continue
        loss_mask = slice_with_cp(loss_mask, 0)
        loss_masks.append(loss_mask)
    ...
    assert loss_masks.shape == tokens.shape
    batch["full_loss_masks"] = loss_masks
```

---

## 5. DataIterator 与 VPP

**Explain：** Virtual Pipeline Parallel 需要 **每个 VPP stage 各一份** `DataIterator`，共享同一 `micro_batch_indices`  schedule；`offset` 在 `reset()` 时归零，供 `forward_only` 多遍扫描。

**Code：**

```python
# 来源：megatron_utils/data.py L201-L245
class DataIterator:
    def get_next(self, keys: Sequence[str]) -> dict[str, list[object] | None]:
        batch = {}
        indices = self.micro_batch_indices[self.offset]
        for key in keys:
            vals = self.rollout_data.get(key, None)
            if vals is None:
                batch[key] = None
            else:
                batch[key] = [vals[i] for i in indices]
        self.offset += 1
        return batch

def get_data_iterator(rollout_data: RolloutBatch) -> list[DataIterator]:
    vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1
    micro_batch_indices = rollout_data["micro_batch_indices"]
    return [DataIterator(rollout_data, micro_batch_indices) for _ in range(vpp_size)]
```

---

## 6. log_rollout_data 与 CP 正确聚合

**Explain：** 仅在 PP last + TP rank 0 执行。token 级指标（log_probs/advantages 等）通过 `get_sum_of_sample_mean` + `rollout_log_metric_contribution` 产出 `(sum, count)`，使跨 DP gather 后等于 per-rollout mean（与 train 侧一致）。

**Code：**

```python
# 来源：megatron_utils/data.py L262-L328
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        ...
        for key, val in rollout_data.items():
            if key in ["tokens", "micro_batch_indices", ...]:
                continue
            if isinstance(val, (list, tuple)) and isinstance(val[0], torch.Tensor):
                if key in ["log_probs", "advantages", "values", ...]:
                    tensor = torch.cat(val).clone().detach()
                    sum_of_sample_mean = get_sum_of_sample_mean(
                        total_lengths, response_lengths, loss_masks, rollout_mask_sums,
                    )
                    sum_value, count = rollout_log_metric_contribution(
                        sum_of_sample_mean(tensor).item(),
                        cp_size=cp_size,
                        num_rollouts_in_rollout=num_rollouts_in_rollout,
                        dp_size=dp_world,
                    )
                    log_dict[key] = (sum_value, count)
```

**Comment：** `clone().detach()` 防止 in-place 修改污染下一轮 rollout 张量。

---

## 7. gather_log_data

**Code：**

```python
# 来源：megatron_utils/data.py L166-L198
def gather_log_data(metric_name, args, rollout_id, log_dict):
    reduced = gather_and_reduce_log_dict(
        log_dict,
        dp_size=mpu.get_data_parallel_world_size(with_context_parallel=True),
        dp_src_rank=mpu.get_data_parallel_src_rank(with_context_parallel=True),
        dp_group=mpu.get_data_parallel_group_gloo(with_context_parallel=True),
    )
    if reduced is None:
        return None
    reduced_log_dict = {f"{metric_name}/{k}": v for k, v in reduced.items()}
    step = compute_rollout_step(args, rollout_id)
    reduced_log_dict["rollout/step"] = step
    logging_utils.log(args, reduced_log_dict, step_key="rollout/step")
    return reduced_log_dict
```

---

## 8. CPU/GPU 张量搬运 helper

**Code：**

```python
# 来源：megatron_utils/data.py L512-L540
def tensors_to_cpu(tensor_list):
    if tensor_list is None:
        return None
    return [t.detach().cpu() for t in tensor_list]

def tensors_to_gpu(tensor_list, device=None):
    if tensor_list is None:
        return None
    if device is None:
        device = torch.cuda.current_device()
    return [t.to(device=device, dtype=torch.float32) for t in tensor_list]
```

**Comment：** critic `values` 在 PP stage 间经 Ray/CPU 传递时使用。

---

## 9. karmarkar_karp 堆合并直觉

**Explain：** 将每个样本 seqlen 视为 State，反复 pop spread 最大的两 State merge，等价于 Largest Differencing Method；`equal_size=True` 时初始 batch 为排序后每 k 个一组。

**Code：**

```python
# 来源：seqlen_balancing.py L109-L117
    while len(states_pq) > 1:
        state0 = heapq.heappop(states_pq)
        state1 = heapq.heappop(states_pq)
        state0.merge(state1)
        heapq.heappush(states_pq, state0)
    final_state = states_pq[0]
    partitions = final_state.get_partitions()
```

**Comment：** spread 最小化 ≈ 各 DP rank 或 mbs 间 token 负载均衡。

---

## 10. greedy_partition 备选

**Explain：** verl 遗留实现；KK 不可用时作参考。Slime 生产路径默认 KK。

**Code：**

```python
# 来源：seqlen_balancing.py L126-L137
def greedy_partition(seqlen_list, k_partitions, equal_size):
    bias = sum(seqlen_list) + 1 if equal_size else 0
    sorted_seqlen = [(seqlen + bias, i) for i, seqlen in enumerate(seqlen_list)]
    ...
    for seqlen, i in sorted_seqlen:
        min_idx = ...  # 当前和最小的 partition
        partitions[min_idx].append(i)
```

---

## 11. PackedSeqParams 字段

**Explain：** THD layout 要求 `cu_seqlens_q/kv` 单调递增；`max_seqlen_*` 供 flash-attn 选 kernel。

**Code：**

```python
# 来源：megatron_utils/data.py L106-L113
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format="thd",
    )
```

---

## 12. multimodal_train_inputs 拼接

**Explain：** 若 batch 含多模态 tensor dict，按 key cat dim=0，与 token cat 顺序一致。

**Code：**

```python
# 来源：megatron_utils/data.py L150-L161
    multimodal_train_inputs = batch.get("multimodal_train_inputs", None)
    if multimodal_train_inputs is not None:
        multimodal_data = {}
        for mm_input_dict in multimodal_train_inputs:
            if mm_input_dict is not None:
                for key, mm_tensor in mm_input_dict.items():
                    multimodal_data[key] = torch.cat([multimodal_data[key], mm_tensor], dim=0) if key in multimodal_data else mm_tensor
```

---

## 13. log_passrate 与 GRPO 组

**Explain：** `raw_reward` reshape 为 `[num_groups, n_samples_per_prompt]` 算 pass@k。

**Code：**

```python
# 来源：megatron_utils/data.py L474-L491
def log_passrate(rollout_id, args, rollout_data):
    ...
            log_dict |= compute_pass_rate(
                flat_rewards=val,
                group_size=args.n_samples_per_prompt,
                num_groups=args.rollout_batch_size,
            )
```

---

## 14. log_perf_data 与 FLOPs

**Explain：** 委托 `train_metric_utils.log_perf_data_raw`；FLOPs 用 **全局** `Timer().seq_lens`（process_rollout_data 写入）。

**Code：**

```python
# 来源：megatron_utils/data.py L496-L508
def log_perf_data(rollout_id, args, extra_metrics=None):
    train_metric_utils.log_perf_data_raw(
        rollout_id=rollout_id,
        args=args,
        is_primary_rank=(... dp_rank == 0),
        compute_total_fwd_flops=lambda seq_lens: calculate_fwd_flops(seqlens=seq_lens, args=args) / dist.get_world_size() / 1e12,
        extra_metrics=extra_metrics,
    )
```

---

## 15. CI kl checker（与 routing 交叉）

**Explain：** rollout_id==0 时比较 actor/ref logprob；`use_rollout_routing_replay` 时跳过（见批次 23）。

**Code：**

```python
# 来源：megatron_utils/data.py L345-L359
        if args.ci_test and reduced_log_dict is not None:
            if rollout_id == 0 and not getattr(args, "use_rollout_routing_replay", False):
                assert abs(reduced_log_dict["rollout/log_probs"] - reduced_log_dict["rollout/ref_log_probs"]) < 1e-8
```

---

## 走读小结

| 函数 | 职责 |
|------|------|
| `build_dp_schedule` | rollout 分步 → pack mbs → 对齐 K → 分配 DP |
| `process_rollout_data` | Ray Box → rank-local `RolloutBatch` |
| `get_batch` | list tokens → THD packed + CP + loss_mask |
| `DataIterator` | 按 mbs 索引表切片 dict 字段 |
| `log_rollout_data` | CP/DP 正确的 rollout 指标 logging |
