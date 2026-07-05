---
type: batch-doc
module: FA05-KV-Cache
batch: "FA05"
doc_type: faq
title: "KV Cache 与推理特性 · 关键问题"
tags:
  - flash-attn/batch/fa05
  - flash-attn/module/kv-cache
  - flash-attn/doc/faq
updated: 2026-07-05
---

# KV Cache 与推理特性 · 关键问题

## 1. 为什么 KV cache 路径不支持 backward？

**Explain：** KV cache API 面向 incremental decoding。它会 in-place 更新 cache，并组合 RoPE、cache indexing、paged KV、SplitKV；这些都是推理 runtime 语义，不是训练 autograd 语义。

**Comment：** 训练长上下文要看普通/varlen forward + backward；decode serving 要看 `flash_attn_with_kvcache`。

## 2. 为什么 paged KV 要求 page block size 是 256 的倍数？

**Explain：** C++ 入口显式检查 `page_block_size % 256 == 0`。这是 kernel 对 block 对齐、访存组织和 page table 解析的约束。

**Code：**

```cpp
// 来源：csrc/flash_attn/flash_api.cpp L1264-L1268
const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
const int num_blocks = !paged_KV ? 0 : kcache.size(0);
const int page_block_size = !paged_KV ? 1 : kcache.size(1);
TORCH_CHECK(!paged_KV || page_block_size % 256 == 0, "Paged KV cache block size must be divisible by 256");
const int seqlen_k = !paged_KV ? kcache.size(1) : max_num_blocks_per_seq * page_block_size;
```

**Comment：** 上层 memory manager 的 block size 不能只按自身碎片率选择，还必须满足 attention backend 的 kernel 约束。

## 3. 为什么 paged KV 不能和 `cache_batch_idx` 同时用？

**Explain：** paged KV 已经通过 `block_table` 描述 batch 到物理 block 的映射；`cache_batch_idx` 是另一套 dense cache indexing 机制。源码直接禁止两者组合，避免语义冲突。

**Code：**

```cpp
// 来源：csrc/flash_attn/flash_api.cpp L1247-L1254
const bool paged_KV = block_table_.has_value();
if (paged_KV) {
    TORCH_CHECK(!cache_batch_idx_.has_value(), "Paged KVcache does not support cache_batch_idx");
    block_table = block_table_.value();
    CHECK_DEVICE(block_table);
    TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
    TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
}
```

**Comment：** serving runtime 需要在调度层选定一种 cache addressing 方案，不应把两套机制叠加。

## 4. `num_splits=0` 是什么意思？

**Explain：** Python docstring 说明 `num_splits=0` 让 kernel 用 heuristic 自动选择 split 数。C++ 侧会根据 batch/head/query blocks、SM 数、K blocks 估计 occupancy。

**Comment：** 手动调 `num_splits` 是性能调参，不是 correctness 参数。除非你明确知道 batch、上下文长度和 GPU SM 数，否则默认 heuristic 更稳。

## 5. `seqlen_q=1` 为什么特殊？

**Explain：** decode 每步 query 很少，普通 tile 会缺并行度。源码的 GQA reshape 把 head group 维转换到 query 维，使 kernel 有更多行可处理。

**Comment：** 这类优化说明 attention backend 和 serving runtime 的 workload 形态高度耦合。

## 6. KV cache API 的入口参数为什么这么多？

**Explain：** `flash_attn_with_kvcache` 同时覆盖 dense cache、paged cache、可选新 KV 写入、RoPE、ALiBi、local window、softcap 和 SplitKV。它不是训练 forward 的变体，而是把 serving decode 中常见的 cache 操作合并进一次 backend 调用。

**Code：**

```python
# 来源：flash_attn/flash_attn_interface.py L1485-L1508
def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    return_softmax_lse=False,
):
    """
    If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
    k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
```

**Comment：** 上层 serving runtime 应该把这个 API 看成“cache 更新 + attention”的融合算子，而不是只替换普通 attention kernel。

## 7. Python 层最终怎样落到 KV cache kernel？

**Explain：** 参数整理后，Python 入口直接调用 `flash_attn_gpu.fwd_kvcache`，并把 cache 索引、分页表、RoPE、窗口、softcap 和 split 数一并传下去。`return_softmax_lse` 只影响返回内容，不改变主计算路径。

**Code：**

```python
# 来源：flash_attn/flash_attn_interface.py L1603-L1627
cache_batch_idx = maybe_contiguous(cache_batch_idx)
block_table = maybe_contiguous(block_table)
out, softmax_lse = flash_attn_gpu.fwd_kvcache(
    q,
    k_cache,
    v_cache,
    k,
    v,
    cache_seqlens,
    rotary_cos,
    rotary_sin,
    cache_batch_idx,
    cache_leftpad,
    block_table,
    alibi_slopes,
    None,
    softmax_scale,
    causal,
    window_size[0],
    window_size[1],
    softcap,
    rotary_interleaved,
    num_splits,
)
return (out, softmax_lse) if return_softmax_lse else out
```

**Comment：** 调试 KV cache 时，Python 侧最值得检查的是这些可选张量是否 contiguous、shape 是否与 cache manager 一致，以及 `block_table` 和 `cache_batch_idx` 是否误混用。

