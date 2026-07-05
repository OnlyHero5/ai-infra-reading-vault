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
updated: 2026-07-05
---

# Train Data · 源码走读

> 走读顺序：`build_dp_schedule` → `process_rollout_data` → `get_batch` → `DataIterator` → `log_rollout_data`
> 基线 commit `22cdc6e1`

Train Data 这一层的主线不是“读数据”，而是把 rollout 产物转换成训练后端可以稳定消费的形状：先按 rollout 保持 sibling 样本的语义边界，再在 DP/CP/VPP 约束下形成 micro-batch，最后在 Megatron 入口处转换为 CP-ready token stream、THD packed attention 参数和可跨 rank 还原的日志指标。

---

## 1. 调度入口：把 rollout 语义变成 DP schedule

### 1.1 build_dp_schedule 的边界

**问题与约束：** 训练侧不能只按 sample 数平均切分，因为同一个 rollout 下的多个样本共享一组奖励、mask 与日志归一化语义；同时 DP rank、CP rank、VPP stage 都要求每个训练 step 的 micro-batch 个数可对齐。

**设计选择：** `build_dp_schedule` 只依赖 Python list 与并行配置，不直接依赖 Ray 或 Megatron 对象；它返回四类纯结构数据：每个 DP rank 的样本分区、每个 rank 的本地 micro-batch 索引、每 step 的 micro-batch 数、每 step 的 rollout 数。

**Explain：** 这相当于把 rollout 数据的“语义批次”先固定下来，再把硬件并行约束映射为 schedule。源码注释强调这是 CPU CI 可测的纯 Python 调度器，设计重心是先 pack、再 distribute。

**Code：**

```python
## 来源：slime/utils/dp_schedule.py L82-L111
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
```

**代码逻辑：** 函数从并行配置取出 `dp_size`、`cp_size`、`vpp_size` 与 VPP 的 micro-batch group size，再计算 dynamic batch 的 token 上限与 schedule 对齐粒度，最后持续构造 rank-local partition 与 micro-batch indices。

**为什么这样写：** 这层不直接搬 tensor，避免把调度策略绑死在 GPU 或分布式运行时上；纯结构输出可以被 Ray 数据分发、Megatron iterator 和单测共同复用。

**不变量与失败模式：** 所有 DP rank 在同一个 step 看到的 `num_microbatches` 必须一致；dynamic batch 下正常 micro-batch 的 token 和不应超过 `max_tokens_per_gpu * cp_size`，但单个超长样本会被允许独立成 batch。

**Comment：** `global_batch_size` 这里的单位是 rollout id，不是最终训练 sample 数；这是后续 per-rollout 归一化能成立的前提。

### 1.2 rollout id 分组

**问题与约束：** rollout 后处理可能把一个 prompt 展开成多个训练样本，如果直接按 sample position 切 step，同一个 rollout 的 sibling 样本会被拆到不同 step，奖励统计与 mask 分母会变得难以解释。

**设计选择：** 源码先用 `rollout_id_to_samples` 保留每个 rollout id 对应的 sample position，再按 rollout id 的出现顺序每 `global_batch_size` 个一组切 step。

**Explain：** 这里的“保序”不是为了好看，而是为了让调度结果仍能回到 rollout 语义：一个 step 包含若干完整 rollout，step 内再处理这些 rollout 展开的所有样本。

**Code：**

```python
## 来源：slime/utils/dp_schedule.py L127-L148
rollout_id_to_samples: dict[int, list[int]] = {}
for sample_pos, rid in enumerate(rollout_indices):
    rollout_id_to_samples.setdefault(rid, []).append(sample_pos)
rollout_ids = list(rollout_id_to_samples.keys())

num_steps = len(rollout_ids) // global_batch_size
assert num_steps >= 1
...
step_rollouts = rollout_ids[step_i * global_batch_size : (step_i + 1) * global_batch_size]
sample_indices = [pos for rid in step_rollouts for pos in rollout_id_to_samples[rid]]
```

**代码逻辑：** `rollout_indices` 中相同 id 的样本被聚合；尾部不足一个完整 global batch 的 rollout 被丢弃；每个 step 再展开为该 step 的 sample position 与对应 `total_lengths`。

**为什么这样写：** RL 训练中的统计单位经常是 rollout 而不是 token 或单样本。先按 rollout 切 step，可以让梯度步、日志步和 pass rate 等指标对齐同一组问题。

**不变量与失败模式：** `num_steps` 必须至少为 1；如果上游产生的 rollout id 数少于 `global_batch_size`，这里会直接 assert，而不是生成语义不完整的训练 step。

**Comment：** 尾部丢弃的是 rollout 维度的残缺 step，不是随机丢样本；这比把残余 sibling 混入下一轮更容易保持训练统计稳定。

### 1.3 step 内 pack micro-batch

**问题与约束：** 一个 step 内的序列长度差异很大；固定 micro-batch size 容易造成 padding 浪费，dynamic batch 又必须避免单卡 token 过载。

**设计选择：** dynamic 模式默认用 `first_fit_pack` 先按 token cap 装箱；如果启用 `balance_by_flops`，则改用 FLOPs 估计做 Karmarkar-Karp 划分；static 模式按固定 `micro_batch_size` 切块。

**Explain：** 这里把“micro-batch 内如何装样本”和“micro-batch 如何分给 DP rank”拆开。前者优化单个 micro-batch 的 token/FLOPs 形状，后者再优化 rank 间负载。

**Code：**

```python
## 来源：slime/utils/dp_schedule.py L55-L79
def _pack_step_into_mbs(args, lengths, *, max_per_bin, balance_by_flops, cp_size):
    if args.dynamic_batch_size:
        if balance_by_flops:
            workloads = calculate_fwd_flops(seqlens=lengths, args=args)
            ...
            return get_seqlen_balanced_partitions(workloads, num_mbs, equal_size=False)
        return first_fit_pack(lengths, max_per_bin=max_per_bin)
    return [list(range(i, min(i + args.micro_batch_size, len(lengths)))) ...]
```

**代码逻辑：** dynamic/token 路径以 `max_per_bin` 作为容量；dynamic/FLOPs 路径先估算需要的 micro-batch 数，再按 FLOPs workload 做平衡；static 路径不看 token 长度，只按样本数切。

**为什么这样写：** token cap 对显存更直接，FLOPs 平衡对 step time 更直接；两种目标不完全等价，所以代码把它们作为显式策略，而不是混在一个启发式里。

**不变量与失败模式：** `balance_by_flops=True` 时源码注释明确不保证 token cap；如果用户同时期待严格显存上限和 FLOPs 平衡，需要理解这里优先平衡计算量。

**Comment：** `cp_size` 参与 `max_per_bin`，因为 CP 会把上下文切到多个 rank 上；调度层按全局序列长度估算每个 CP 组可承载的 token 总量。

### 1.4 对齐后再分发到 DP rank

**问题与约束：** PP/VPP 训练要求每个 DP rank 的 micro-batch 数一致；VPP 下还要满足 micro-batch group 对齐，否则 pipeline schedule 不能稳定推进。

**设计选择：** dynamic 模式通过拆分已有 bin 把 micro-batch 数扩展到 `align_to` 的倍数；static 模式如果不能对齐就直接报错。分配到 DP rank 时，可以选择 FLOPs 平衡，也可以用 round-robin。

**Explain：** 源码把“数量对齐”放在“DP 分配”之前，这能保证无论采用 KK 还是 round-robin，最后每个 rank 的 micro-batch 个数都能被训练循环接受。

**Code：**

```python
## 来源：slime/utils/dp_schedule.py L167-L207
if len(mbs) % align_to != 0:
    if args.dynamic_batch_size:
        mbs = expand_bins_by_splitting(mbs, step_lengths, target_count=target_count)
    else:
        raise AssertionError(...)

if args.balance_data:
    mb_workloads = [sum(calculate_fwd_flops(...)) for mb in mbs]
    dp_partitions = get_seqlen_balanced_partitions(mb_workloads, dp_size, equal_size=True)
else:
    dp_partitions = [list(range(r, len(mbs), dp_size)) for r in range(dp_size)]
```

**代码逻辑：** `align_to = dp_size * mb_group`（VPP 生效时）或 `dp_size`；dynamic 可以通过拆分较大的 micro-batch 补齐数量，static 无法改变 batch 边界则失败；DP 分配后再生成全局 sample partition 与 rank-local micro-batch indices。

**为什么这样写：** 先对齐数量能保护 Megatron 的训练循环；再做负载平衡能减少长序列集中在同一 DP rank 导致的 straggler。

**不变量与失败模式：** 每个 rank 的 `micro_batch_indices` 是该 rank partition 内的本地下标，不是全局 sample index；如果后续代码把二者混用，会取错样本。

**Comment：** dynamic 模式的“拆 bin”不会创造新样本，只是把已有 micro-batch 内的样本再分成更小组，从而换取 pipeline 可调度性。

---

## 2. 序列长度平衡：调度器的两个工具

### 2.1 Karmarkar-Karp 分区

**问题与约束：** 长短序列混合时，只按样本数均分会让某些 DP rank 的 FLOPs 明显更高，训练 step 被最慢 rank 拖住。

**设计选择：** `get_seqlen_balanced_partitions` 使用 Karmarkar-Karp 风格的贪心分区，把样本长度或 FLOPs workload 作为权重，返回覆盖全部索引的 k 个 partition。

**Explain：** 这个工具并不理解 rollout，也不理解 Megatron；它只解决一个小问题：给定一组权重，把索引尽量均衡地拆成 k 组。

**Code：**

```python
## 来源：slime/utils/seqlen_balancing.py L146-L177
def get_seqlen_balanced_partitions(
    seqlens: list[int],
    k_partitions: int,
    equal_size: bool,
) -> list[list[int]]:
    assert len(seqlens) >= k_partitions
    partitions = karmarkar_karp(seqlens, k_partitions, equal_size)
    return _check_and_sort_partitions(partitions, seqlens, k_partitions)
```

**代码逻辑：** 函数先要求元素数量不少于 partition 数，再调用 KK 算法，最后通过 `_check_and_sort_partitions` 校验 partition 数、非空性、索引覆盖与排序。

**为什么这样写：** 平衡算法如果返回重复或漏掉的索引，后面会变成训练样本重复或消失；把校验集中在工具层能让调度层只处理合法 partition。

**不变量与失败模式：** `len(seqlens) < k_partitions` 会 assert；这避免生成空 partition，因为 DP/VPP 侧通常不能接受某些 rank 完全没有 micro-batch。

**Comment：** 当 `equal_size=True` 时，它不仅平衡权重，也约束每组元素数一致；这正好用于“每个 DP rank micro-batch 数一致”的场景。

### 2.2 first-fit pack 与拆 bin

**问题与约束：** dynamic batch 需要把多个短序列合并到一个 micro-batch 里提高利用率，但不能为了装箱而重排出不可控的超大 micro-batch。

**设计选择：** `first_fit_pack` 采用 first-fit bin packing；`expand_bins_by_splitting` 在数量不满足 pipeline 对齐时，反复拆分当前 token 最大且可拆的 bin。

**Explain：** 这是一组偏工程的启发式：先用简单、确定性的装箱减少 padding 和碎片，再在必须对齐数量时拆大不拆小。

**Code：**

```python
## 来源：slime/utils/seqlen_balancing.py L180-L229
def first_fit_pack(lengths: list[int], max_per_bin: int) -> list[list[int]]:
    ...
def expand_bins_by_splitting(
    bins: list[list[int]],
    lengths: list[int],
    target_count: int,
) -> list[list[int]]:
    while len(bins) < target_count:
        candidates = [(sum(lengths[i] for i in b), idx) for idx, b in enumerate(bins) if len(b) > 1]
        if not candidates:
            break
        _, idx = max(candidates)
        left, right = _split_bin_by_tokens(bins.pop(idx), lengths)
```

**代码逻辑：** `first_fit_pack` 对每个样本尝试放入已有 bin，放不下则新建 bin；单个样本超过上限时会作为单独 bin 保留。扩展 bin 时只拆含多个样本的 bin，并用 LPT 风格把 token 分到两半。

**为什么这样写：** 超长单样本无法再拆，强行失败会让训练无法处理长回复；把它独立成 bin 后，显存风险被局部化，后续 schedule 仍可继续。

**不变量与失败模式：** `expand_bins_by_splitting` 可能因为所有 bin 都是单样本而达不到 `target_count`；调用方必须再检查数量对齐是否真的满足。

**Comment：** 这也解释了 dynamic batch 的边界：它能改善大多数长度分布，但不是一个严格的全局最优装箱求解器。

### 2.3 KK 合并循环

**问题与约束：** 平衡分区需要在可接受的时间内处理训练 batch，不适合在热路径里跑复杂整数规划。

**设计选择：** `karmarkar_karp` 把每个 state 放入 heap，反复取出 spread 最大的两个 state，并用相反顺序合并它们的 partition。

**Explain：** 这个策略追求“足够均衡且开销小”。它把最不均衡的两个状态相互抵消，逐步形成 k 个 workload 接近的组。

**Code：**

```python
## 来源：slime/utils/seqlen_balancing.py L109-L117
while len(heap) > 1:
    state1 = heapq.heappop(heap)
    state2 = heapq.heappop(heap)
    new_state = sorted(
        [state1[i] + state2[k - 1 - i] for i in range(k)],
        key=lambda x: x.sum,
    )
    heapq.heappush(heap, new_state)
```

**代码逻辑：** heap 每次弹出两个候选；一个按低到高、另一个按高到低组合；组合后再按 partition sum 排序并放回 heap。

**为什么这样写：** 长 workload 与短 workload 配对能降低分组间差距；与完整搜索相比，这个局部合并策略更适合每个 rollout step 频繁调用。

**不变量与失败模式：** `equal_size` 初始化路径会保证每个 partition 的元素数量一致，并在最后 assert；如果输入规模不能整除 k，调用方不应启用这个约束。

**Comment：** Slime 在 DP 分配 micro-batch 时用的是 workload 维度的平衡，而不是样本数维度的平均。

### 2.4 greedy_partition 的定位

**问题与约束：** 有些场景可能需要更直观、更低成本的 fallback，而不是 KK 的状态合并。

**设计选择：** `greedy_partition` 保留了一个贪心备选实现：每次把当前最长序列放入目前负载最低的 partition；`equal_size=True` 时用 bias 逼近等数量约束。

**Explain：** 它不是当前 `build_dp_schedule` 的主路径，但能帮助读者理解 Slime 对“长度均衡”的抽象：输入是一组权重，输出是一组索引 partition。

**Code：**

```python
## 来源：slime/utils/seqlen_balancing.py L126-L137
def greedy_partition(seqlens, k_partitions, equal_size: bool):
    sorted_seqlens = sorted(enumerate(seqlens), key=lambda x: x[1], reverse=True)
    ...
    for idx, seqlen in sorted_seqlens:
        min_idx = argmin(partition_sums)
        partitions[min_idx].append(idx)
        partition_sums[min_idx] += seqlen
```

**代码逻辑：** 函数先按长度降序处理样本，再把当前样本塞给当前累计长度最小的 partition。

**为什么这样写：** LPT 风格的贪心算法简单、可解释、足够快；当需要调试 KK 结果或做轻量 fallback 时，它提供了可对照的 baseline。

**不变量与失败模式：** 贪心不保证最优；当长度分布极端时，它可能比 KK 更偏，但不会漏掉输入索引。

**Comment：** 这类工具被放在 `utils`，说明 Slime 把“长度平衡”作为可复用策略，而不是某个训练入口的私有逻辑。

---

## 3. Ray rollout 分片进入训练 rank

### 3.1 process_rollout_data

**问题与约束：** rollout 数据通过 Ray object ref 传到训练 worker；每个 DP rank 只应拿到自己的样本分区，但全局长度统计仍要保留给 timer 和性能日志。

**设计选择：** `process_rollout_data` 要求传入的 ref 数等于 DP size，每个 rank 读取自己的 ref；从数据中弹出 `partition`，用它把 `total_lengths` 改写成本 rank 局部视图，同时把全局长度写入 `Timer().seq_lens`。

**Explain：** 这里完成 schedule 到数据的第一次落地：调度器产生的全局 sample index 被用于裁剪 rollout batch，训练 rank 后续只看本地样本。

**Code：**

```python
## 来源：slime/utils/data.py L292-L303
def process_rollout_data(args, rollout_data_ref, dp_rank, dp_size):
    assert len(rollout_data_ref) == dp_size
    rollout_data = ray.get(rollout_data_ref[dp_rank].inner)
    partition = rollout_data.pop("partition")
    total_lengths = rollout_data["total_lengths"]
    Timer().seq_lens = total_lengths
    rollout_data["total_lengths"] = [total_lengths[i] for i in partition]
    return rollout_data
```

**代码逻辑：** 函数按 `dp_rank` 取回对象；`partition` 只作为分片索引使用，不留在训练 batch 中；`total_lengths` 从全局数组变成本 rank 的长度列表。

**为什么这样写：** Ray 传输的是 rollout 结果，Megatron 消费的是 rank-local batch；在这一层完成裁剪，可以让后面的 `DataIterator` 和 `get_batch` 都不用再关心全局 partition。

**不变量与失败模式：** ref 数量必须与 DP size 一致；如果 `partition` 与 `total_lengths` 不匹配，会在索引阶段暴露，而不是默默产生错位训练样本。

**Comment：** `Timer().seq_lens` 保留全局长度，这让性能统计仍能基于完整 rollout，而不是仅看单个 DP rank 的局部样本。

---

## 4. Megatron batch bridge：把样本变成 CP-ready tensor

### 4.1 get_batch 入口与原始 token 保留

**问题与约束：** Megatron 前向需要 packed token tensor，但训练日志、logprob 对齐和 CP 场景有时仍需要原始未拼接的 token 列表。

**设计选择：** `get_batch` 从 `DataIterator` 取字段后，先把 `tokens` 原样保存到 `batch["unconcat_tokens"]`，再进入 CP slicing、concat 与 padding。

**Explain：** 这是一个桥接函数：输入仍是 list-of-samples，输出变成当前 CUDA rank 可消费的 packed tensor 与 `PackedSeqParams`。

**Code：**

```python
## 来源：slime/backends/megatron_utils/data.py L28-L64
def get_batch(data_iterator, keys, pad_multiplier=128, allgather_cp=False):
    assert "tokens" in keys
    batch = data_iterator.get_next(keys)
    tokens = batch["tokens"]
    pad_token_id = 0
    pad_size = mpu.get_tensor_model_parallel_world_size() * pad_multiplier
    batch["unconcat_tokens"] = tokens
```

**代码逻辑：** 函数强制要求 `tokens` 字段；从 iterator 取当前 micro-batch；设置 padding 粒度；保留未 concat token；读取 CP size 与 rank 后进入两种 CP 路径。

**为什么这样写：** token stream 一旦 concat 并按 CP rank 切片，就失去样本级边界；提前保存原始列表能让后续需要样本边界的逻辑不用反推。

**不变量与失败模式：** `keys` 中必须包含 `tokens`；如果 iterator 的 `micro_batch_indices` 已越界，错误会在取样本时暴露。

**Comment：** `pad_token_id=0` 是这里的工程假设；它只用于 padding token stream，不改变 loss mask 的有效位置。

### 4.2 allgather_cp 的全局 concat 路径

**问题与约束：** 某些 CP 模式需要先在全局 token stream 上保留完整序列边界，再一次性切给各 CP rank；否则各 rank 的 chunk 可能长度不一致。

**设计选择：** `allgather_cp=True` 时，代码先构造全局 `cu_seqlens_list`，把所有 token concat 成一个 stream，再 pad 到 `cp_size * pad_size` 的倍数，最后按 CP rank chunk。

**Explain：** 这条路径的设计哲学是“先全局对齐，再局部切片”，这样每个 CP rank 得到的 token chunk 长度一致。

**Code：**

```python
## 来源：slime/backends/megatron_utils/data.py L69-L87
if allgather_cp:
    cu_seqlens_list = [0]
    for t in tokens:
        cu_seqlens_list.append(cu_seqlens_list[-1] + t.size(0))
    tokens = torch.cat(tokens, dim=0)
    global_pad_size = cp_size * pad_size
    pad = (global_pad_size - tokens.size(0) % global_pad_size) % global_pad_size
    ...
    tokens = tokens.chunk(cp_size, dim=0)[cp_rank]
```

**代码逻辑：** `cu_seqlens_list` 记录原始样本边界；全局 concat 后按 `cp_size * pad_size` pad；padding 长度也追加到 `cu_seqlens_list`；随后把 token stream 均分给当前 CP rank。

**为什么这样写：** 如果先局部切每个 sample，再拼起来，跨样本边界和 padding 位置会更复杂；全局 concat 让 CP chunk 的形状约束更容易满足。

**不变量与失败模式：** 全局 token 长度 pad 后必须能被 `cp_size` 整除；`pad` 变量还会被 loss mask 路径复用，因此 token 与 mask 必须走同一套 padding 规则。

**Comment：** 这条路径适合需要“完整序列边界先于 CP 切片”的场景，代价是当前 rank 的 token 已不再对应完整样本。

### 4.3 默认 CP zigzag 切片路径

**问题与约束：** 默认 CP 训练希望每个样本先按 CP 规则切片，再把当前 rank 持有的片段 concat；同时 packed attention 仍需要能还原原始序列长度。

**设计选择：** `allgather_cp=False` 时，源码对每个样本调用 `slice_with_cp`，再 concat 当前 rank 的片段；padding 后将 `cu_seqlens` 乘以 `cp_size`，让 THD packed attention 看到原始长度尺度。

**Explain：** 这是“先样本内 CP 切片，再 rank-local concat”的路径。它更贴近普通 CP 训练：每个 rank 只处理每个序列的一部分。

**Code：**

```python
## 来源：slime/backends/megatron_utils/data.py L88-L104
else:
    tokens = [slice_with_cp(t, pad_token_id) for t in tokens]
    cu_seqlens = [0]
    for t in tokens:
        cu_seqlens.append(cu_seqlens[-1] + t.size(0))
    tokens = torch.cat(tokens)
    pad = (pad_size - tokens.size(0) % pad_size) % pad_size
    ...
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int).cuda() * cp_size
```

**代码逻辑：** 每条序列先被 `slice_with_cp` 切到当前 CP rank；concat 后按 tensor-parallel padding 粒度补齐；`cu_seqlens` 使用 rank-local 长度累加后乘回 CP size。

**为什么这样写：** THD attention 的 metadata 需要描述逻辑上的原始序列，而 rank-local token 只是一段切片；乘以 `cp_size` 是把二者重新对齐。

**不变量与失败模式：** `slice_with_cp` 必须对 token 与 loss mask 使用一致的切法；如果二者不同步，后面的 shape assert 可能通过不了，或者更糟糕地错算 loss。

**Comment：** `cu_seqlens` 乘 CP size 是这段代码里最容易漏看的设计点，它说明 metadata 表示的是逻辑序列边界，不只是本 rank 实际 token 数。

### 4.4 PackedSeqParams

**问题与约束：** Megatron packed sequence 前向需要 `cu_seqlens_q/kv`、最大序列长度和 QKV layout；如果这些 metadata 与 token stream 不一致，attention kernel 会读错边界。

**设计选择：** `get_batch` 基于前面构造的 `cu_seqlens` 计算 `max_seqlen`，同时把 q/k/v 都设为同一套 seqlens，并声明 `qkv_format="thd"`。

**Explain：** Slime 在这里把可变长样本统一转换成 Megatron packed THD 入口。训练循环不再需要知道原始样本有多少条，只消费一个 packed token tensor。

**Code：**

```python
## 来源：slime/backends/megatron_utils/data.py L106-L113
max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
packed_seq_params = PackedSeqParams(
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_kv=cu_seqlens,
    max_seqlen_q=max_seqlen,
    max_seqlen_kv=max_seqlen,
    qkv_format="thd",
)
```

**代码逻辑：** 相邻 `cu_seqlens` 的差得到每条序列长度；最大值写入 q 和 kv 的最大长度；同一批自回归训练里 q 与 kv 使用相同边界。

**为什么这样写：** 用标准 `PackedSeqParams` 把 Slime 的 rollout batch 接到 Megatron kernel，可以复用已有 packed attention 路径，而不是另写一套变长 attention 调用。

**不变量与失败模式：** `cu_seqlens` 必须单调递增且长度至少为 2；如果 micro-batch 为空，`max()` 会失败，这也反向要求调度器不能给出空 micro-batch。

**Comment：** 这里没有显式传 batch size，因为 packed THD 的 batch 边界已经编码在 `cu_seqlens` 里。

### 4.5 loss mask 与 token stream 对齐

**问题与约束：** loss mask 原本只覆盖 response 位置，但 token stream 包含 prompt、response、右移预测位置和 padding；mask 必须和最终 token tensor shape 完全一致。

**设计选择：** 每个样本根据 `prompt_length = total_length - response_length` 左侧补 `prompt_length - 1` 个 0，右侧补 1 个 0；随后按 token 相同的 CP 路径和 padding 规则处理 mask。

**Explain：** 这段把“response-level mask”对齐到“next-token prediction 的 token stream”。左侧跳过 prompt，右侧跳过没有 target 的末位。

**Code：**

```python
## 来源：slime/backends/megatron_utils/data.py L121-L147
for loss_mask, total_length, response_length in zip(..., strict=True):
    prompt_length = total_length - response_length
    loss_mask = F.pad(loss_mask, (prompt_length - 1, 1), value=0)
    if allgather_cp:
        loss_masks.append(loss_mask)
        continue
    loss_mask = slice_with_cp(loss_mask, 0)
    loss_masks.append(loss_mask)
...
assert loss_masks.shape == tokens.shape
```

**代码逻辑：** mask 先按样本补齐到 token 级；allgather 路径先 concat、global pad、chunk；默认路径先 CP slice、concat、local pad；最后用 shape assert 校验。

**为什么这样写：** 训练 loss 的每个位置必须和 token stream 的每个位置一一对应；把 mask 走和 token 同构的处理路径，能减少 CP 切片引入的错位风险。

**不变量与失败模式：** `zip(..., strict=True)` 要求 `loss_masks`、`total_lengths`、`response_lengths` 数量一致；最终 shape 不一致会直接 assert，而不是继续训练出错误梯度。

**Comment：** 这里的右侧补 1 个 0 是 next-token 训练常见的末位无 target 处理，不是 response 末尾额外丢数据。

### 4.6 multimodal_train_inputs 合并

**问题与约束：** 多模态样本可能携带额外 tensor 字段，训练模型希望按 key 得到 batch 级 tensor，而不是每个样本一个 dict。

**设计选择：** `get_batch` 检查 `multimodal_train_inputs`，跳过 `None` 样本，把相同 key 的 tensor 沿 dim 0 concat。

**Explain：** 多模态数据没有进入 CP token slicing 的主路径，而是在 token batch 形成后做 key-wise 合并，保持语言 token 和多模态 tensor 的职责分离。

**Code：**

```python
## 来源：slime/backends/megatron_utils/data.py L150-L161
multimodal_train_inputs = batch.get("multimodal_train_inputs", None)
if multimodal_train_inputs is not None:
    multimodal_data = {}
    for mm_input_dict in multimodal_train_inputs:
        if mm_input_dict is not None:
            for key, mm_tensor in mm_input_dict.items():
                multimodal_data[key] = mm_tensor if key not in multimodal_data else torch.cat(...)
    batch["multimodal_train_inputs"] = multimodal_data
```

**代码逻辑：** 输入是 list[dict | None]；输出是 dict[key, concatenated tensor]；同一 key 多次出现时按 batch 维拼接。

**为什么这样写：** 多模态字段的 shape 与 token 序列不一定一一对应；把它们按 key 聚合，可以让模型侧按各自 encoder 的输入格式读取。

**不变量与失败模式：** 同一 key 下的 tensor 必须能沿 dim 0 concat；如果单样本 dict 的 key 集不一致，缺失 key 不会报错，但模型侧需要能处理缺项语义。

**Comment：** 这段没有改变 `tokens` 与 `loss_masks`，说明多模态扩展被设计为附加输入，而不是重写语言训练 batch。

---

## 5. Iterator 与日志：让训练循环和指标都看见同一份 schedule

### 5.1 gather_log_data

**问题与约束：** 日志指标来自不同 DP/CP rank，有些值是普通 rank 均值，有些值已经是 `(sum, count)` 形式；简单 all-reduce mean 会把不等样本数的 rank 算偏。

**设计选择：** `gather_log_data` 把归约逻辑委托给 `gather_and_reduce_log_dict`，支持 tuple 指标按 `Σsum / Σcount` 聚合，普通 scalar 按 DP 均值聚合，再统一加 metric 前缀并写日志。

**Explain：** 这里把“如何跨 rank 归约”和“如何对外记录”分层。前者在 CP helper 中可单测，后者在训练数据模块里绑定 rollout step 与 logger。

**Code：**

```python
## 来源：slime/backends/megatron_utils/data.py L166-L198
def gather_log_data(metric_name, args, rollout_id, log_dict):
    reduced = gather_and_reduce_log_dict(
        log_dict,
        dp_size=mpu.get_data_parallel_world_size(with_context_parallel=True),
        dp_src_rank=mpu.get_data_parallel_src_rank(with_context_parallel=True),
        dp_group=mpu.get_data_parallel_group_gloo(with_context_parallel=True),
    )
    ...
    step = compute_rollout_step(args, rollout_id)
    logging_utils.log(args, reduced_log_dict, step_key="rollout/step")
```

**代码逻辑：** 函数在 DP+CP group 内收集并归约 log dict；非源 rank 可能得到 `None`；源 rank 给 key 加上 `metric_name/` 前缀，计算 rollout step 后写入日志系统。

**为什么这样写：** 训练数据可能被 uneven DP 分片，日志必须按真实 count 加权；否则样本少的 rank 会和样本多的 rank 拥有同等话语权。

**不变量与失败模式：** tuple 指标必须是 `(sum, count)` 语义；如果调用方把均值伪装成 sum，最终日志会再次平均而失真。

**Comment：** `with_context_parallel=True` 说明日志归约把 CP rank 也纳入数据并行语义，否则 CP 切片后的 token-level 指标会缺片。

### 5.2 DataIterator 与 VPP stage

**问题与约束：** 训练循环每次只需要当前 micro-batch 的字段子集；schedule 已经给出了 rank-local micro-batch indices，iterator 不能再自行重排。

**设计选择：** `DataIterator` 持有 `rollout_data` 和 `micro_batch_indices`，每次 `get_next(keys)` 按当前 offset 取出对应本地样本；`get_data_iterator` 为每个 VPP stage 返回一个 iterator。

**Explain：** 这是一个非常薄的 iterator，几乎没有策略。策略已经在 `build_dp_schedule` 中确定，这里只负责按 schedule 播放数据。

**Code：**

```python
## 来源：slime/backends/megatron_utils/data.py L201-L245
class DataIterator:
    def __init__(self, rollout_data, micro_batch_indices):
        self.rollout_data = rollout_data
        self.micro_batch_indices = micro_batch_indices
        self.offset = 0

    def get_next(self, keys):
        indices = self.micro_batch_indices[self.offset]
        ...
        self.offset += 1

def get_data_iterator(rollout_data):
    vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1
    return [DataIterator(rollout_data, micro_batch_indices) for _ in range(vpp_size)]
```

**代码逻辑：** iterator 根据 `offset` 选择一个 micro-batch 的本地索引；每个 requested key 如果不存在则返回 `None`，存在则按索引取 list 子集；`reset()` 只把 offset 归零。

**为什么这样写：** VPP stage 需要各自推进 micro-batch，但它们共享同一份 rank-local rollout data 和 schedule；返回多个 iterator 比复制数据更轻。

**不变量与失败模式：** `micro_batch_indices` 中的索引必须是 rank-local index；如果误传全局 sample index，会在 `vals[i]` 处越界或取错样本。

**Comment：** iterator 不做 shuffle，这是有意的：RL rollout 的分组和 DP 平衡已经由前序 schedule 决定。

### 5.3 log_rollout_data 的主统计路径

**问题与约束：** 训练日志既包含 token-level 指标，也包含 sample-level 和 rollout-level 指标；CP 切分、uneven DP 和 sibling rollout 都会影响平均值的分母。

**设计选择：** `log_rollout_data` 只在 TP rank 0 且 PP last stage 汇总；跳过训练控制字段，对 list/tensor/scalar 分别转成 `(sum, count)`；对 log_probs、returns、advantages 等 token-level 指标使用 `get_sum_of_sample_mean` 和 `rollout_log_metric_contribution`。

**Explain：** 这段代码的核心不是“打印指标”，而是把训练 loss 使用的 per-rollout 平均语义复用到日志里，避免日志和梯度信号讲两套数学。

**Code：**

```python
## 来源：slime/backends/megatron_utils/data.py L262-L328
if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
    cp_size = mpu.get_context_parallel_world_size()
    rollout_mask_sums = rollout_data.get("rollout_mask_sums", None)
    dp_world = mpu.get_data_parallel_world_size(with_context_parallel=False)
    num_rollouts_in_rollout = sum(rollout_data["global_batch_sizes"])
    ...
    sum_of_sample_mean = get_sum_of_sample_mean(...)
    sum_value, count = rollout_log_metric_contribution(...)
    log_dict[key] = (sum_value, count)
```

**代码逻辑：** 函数遍历 rollout_data；跳过 `tokens`、`loss_masks`、schedule 字段等不应直接均值的字段；token-level 训练指标先按样本/rollout mask 求均值贡献，再转成可跨 DP 加权归约的 tuple。

**为什么这样写：** 如果直接对 concat 后的 token tensor 求 mean，长回答会比短回答权重更大；如果直接对 rank 均值求 mean，样本少的 rank 权重又过大。tuple 聚合同时解决这两个偏差。

**不变量与失败模式：** 被纳入 token-level 特殊路径的 tensor list 必须与 `total_lengths`、`response_lengths`、`loss_masks` 对齐；普通 list 也假设至少有一个元素，因为代码访问 `val[0]`。

**Comment：** `rollout_mask_sums` 是让 sibling rollout 共享分母的关键字段；缺失时 helper 会退回 sample-level 分母。

### 5.4 CI KL 检查

**问题与约束：** 初始 actor/ref 的 logprob 应在普通路径下几乎一致，可作为 CI 的快速正确性检查；但 routing replay 会让 actor/ref 的路由路径有意不同。

**设计选择：** CI 下只在 `rollout_id == 0`、未禁用 checker、未启用 `use_rollout_routing_replay` 且两个 logprob 指标都存在时检查差值小于 `1e-8`。

**Explain：** 这不是训练逻辑，而是数据路径和模型路径的早期 sanity check。它特意避开 routing replay，是因为 replay 的 actor forward 与 reference forward 不追求 bit-level 相同路由。

**Code：**

```python
## 来源：slime/backends/megatron_utils/data.py L345-L359
if args.ci_test and reduced_log_dict is not None:
    if (
        rollout_id == 0
        and not getattr(args, "ci_disable_kl_checker", False)
        and not getattr(args, "use_rollout_routing_replay", False)
        and "rollout/log_probs" in reduced_log_dict
        and "rollout/ref_log_probs" in reduced_log_dict
    ):
        assert abs(reduced_log_dict["rollout/log_probs"] - reduced_log_dict["rollout/ref_log_probs"]) < 1e-8
```

**代码逻辑：** 只有首个 rollout step 触发严格 KL 检查；随后还会检查 logprob 和 entropy 的合理范围。

**为什么这样写：** 首步检查能捕捉数据对齐、mask 对齐、reference path 错位等基础问题；但在 routing replay 场景强行要求相同会制造误报。

**不变量与失败模式：** 如果 CI 开启且普通 actor/ref 初始 logprob 差异超过阈值，说明数据路径或模型路径至少有一个环节不再对齐。

**Comment：** 这段和 CP/RoutingReplay 笔记相互呼应：有些差异是 bug，有些差异是功能选择，CI 条件需要区分二者。

### 5.5 passrate 日志

**问题与约束：** pass@k 是按 prompt/problem 组计算的指标，不能把所有 reward 当成无结构的一维均值。

**设计选择：** `log_passrate` 只读取 `raw_reward`，使用 `args.n_samples_per_prompt` 作为 group size、`args.rollout_batch_size` 作为 group 数，交给 `compute_pass_rate` 生成指标。

**Explain：** 这里把 reward 的组结构显式传给 metric helper，保证 pass@k 的统计单位仍是问题组，而不是单条 sample。

**Code：**

```python
## 来源：slime/backends/megatron_utils/data.py L474-L491
def log_passrate(rollout_id, args, rollout_data):
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        log_dict = {}
        for key, val in rollout_data.items():
            if key != "raw_reward":
                continue
            log_dict |= compute_pass_rate(
                flat_rewards=val,
                group_size=args.n_samples_per_prompt,
                num_groups=args.rollout_batch_size,
            )
```

**代码逻辑：** 函数只在主日志 rank 执行；找到 `raw_reward` 后按配置 reshape/分组计算 pass rate；最终仍复用 `gather_log_data` 写出。

**为什么这样写：** pass@k 与普通 reward mean 的分母不同。把组参数显式放进调用，能避免读者误以为 `raw_reward` 只是一个可直接平均的列表。

**不变量与失败模式：** `flat_rewards` 的长度必须和 `group_size * num_groups` 语义匹配；如果 rollout batch 或采样数配置错位，pass rate 会失去解释性。

**Comment：** 这段说明 Train Data 的日志层保留了 rollout 的任务结构，不只是训练 tensor 的统计。

### 5.6 perf data

**问题与约束：** 性能日志需要把序列长度转换成 FLOPs，并且只应由一个 primary rank 对外记录，避免重复写指标。

**设计选择：** `log_perf_data` 把 primary rank 条件、FLOPs 计算 lambda 和 extra metrics 传给 `train_metric_utils.log_perf_data_raw`。

**Explain：** 性能统计在这里保持轻薄：本模块知道并行 rank 和序列长度如何解释，真正的日志格式与吞吐计算交给通用 metric utils。

**Code：**

```python
## 来源：slime/backends/megatron_utils/data.py L496-L508
def log_perf_data(rollout_id, args, extra_metrics=None):
    train_metric_utils.log_perf_data_raw(
        rollout_id=rollout_id,
        args=args,
        is_primary_rank=(... and mpu.get_data_parallel_rank(with_context_parallel=True) == 0),
        compute_total_fwd_flops=lambda seq_lens: calculate_fwd_flops(seqlens=seq_lens, args=args)
        / dist.get_world_size()
        / 1e12,
        extra_metrics=extra_metrics,
    )
```

**代码逻辑：** primary rank 条件要求 TP rank 0、PP last stage、DP+CP rank 0；FLOPs 用 `calculate_fwd_flops` 后除以 world size 并换算到 TFLOPs。

**为什么这样写：** FLOPs 是全局 workload，但每个 rank 只承担其中一部分；除以 `dist.get_world_size()` 可以让日志表达单 rank 平均承担的前向计算量。

**不变量与失败模式：** `seq_lens` 需要来自前面保留的 rollout 长度统计；如果只用 rank-local 长度，吞吐日志会随 DP 分片波动。

**Comment：** `extra_metrics` 保留扩展入口，方便新的训练后端指标接入而不改 Train Data 主路径。

### 5.7 CPU/GPU tensor 搬运

**问题与约束：** Ray object store 更适合传 CPU tensor，但训练前向需要 GPU tensor；直接把带梯度或 GPU resident 的 tensor 放进跨进程对象会增加显存和序列化压力。

**设计选择：** `tensors_to_cpu` 对 list 内 tensor 做 `detach().cpu()`；`tensors_to_gpu` 在需要时搬回当前 CUDA device，并转成 `float32`。

**Explain：** 这是 rollout 与训练 worker 之间的最小搬运适配层。它不改变 list 结构，只改变 tensor 的 device 和 dtype。

**Code：**

```python
## 来源：slime/backends/megatron_utils/data.py L512-L540
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

**代码逻辑：** 两个函数都保留 `None` 语义；CPU 路径断开 autograd 并搬到 host；GPU 路径默认使用当前 device，并统一转为 float32。

**为什么这样写：** Rollout 数据是训练输入，不应携带上游计算图；统一 float32 能减少日志/训练辅助 tensor 因 dtype 不一致造成的后续分支。

**不变量与失败模式：** 这两个 helper 假设输入是 tensor list；如果传入嵌套 dict 或非 tensor 元素，不会递归处理。

**Comment：** 它们是小函数，但体现了 Train Data 的边界意识：跨进程传输和 GPU 训练各用适合自己的 tensor 形态。

---

## 6. 串起来看

Train Data 的设计可以概括为三层分离：

1. `build_dp_schedule` 只决定样本如何按 rollout、micro-batch、DP rank 排列。
2. `process_rollout_data` 和 `DataIterator` 只按 schedule 播放 rank-local 数据。
3. `get_batch` 与日志函数只在 Megatron 边界处理 CP slicing、packed metadata、mask 对齐和跨 rank 指标语义。

这种分层让 Slime 可以在不重写训练循环的前提下接入 dynamic batch、CP、VPP、routing replay 以及 rollout-level 指标。读源码时要特别注意两条线：一条是样本索引如何从 global partition 变成 rank-local micro-batch index，另一条是 token、loss mask、`cu_seqlens` 如何始终保持同构变换。
