---
type: batch-doc
module: FA05-KV-Cache
batch: "FA05"
doc_type: concept
title: "KV Cache 与推理特性 · 核心概念"
tags:
  - flash-attn/batch/fa05
  - flash-attn/module/kv-cache
  - flash-attn/doc/concept
updated: 2026-07-04
---

# KV Cache 与推理特性 · 核心概念

## 1. Prefill 与 decode 的 attention 形态不同

**Explain：** Prefill 通常是 `seqlen_q` 和 `seqlen_k` 都较长的一次 full attention；decode 通常每步只来少量 query token，但要读完整历史 KV cache。前者偏矩阵乘吞吐，后者更容易被 KV cache 读取、并行度不足和调度开销影响。

| 阶段 | `Q` | `K/V` | 主要瓶颈 |
|------|-----|-------|----------|
| prefill | prompt 全部 token | prompt 全部 token | 长序列 QK/PV 与 IO |
| decode | 当前新 token | 历史 KV cache + 新 KV | cache 读带宽、小 `seqlen_q` 并行度 |

## 2. `flash_attn_with_kvcache` 是推理 API

**Explain：** 这个 API 明确支持把新 `k/v` 写入 cache，再对更新后的 cache 做 attention。它还支持 RoPE、paged KV、batch index、leftpad、SplitKV。

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
    window_size=(-1, -1),
    softcap=0.0,
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    return_softmax_lse=False,
):
```

**Comment：** `block_table` 是 paged KV 的入口；`num_splits` 是长 K/V 或并行度不足时的重要旋钮。

## 3. KV cache 路径不支持 backward

**Explain：** decode 推理不需要训练 backward。源码 docstring 明确说明该路径不支持 backward，这让 kernel 可以围绕 cache update 和 forward latency 优化。

**Code：**

```python
# 来源：flash_attn/flash_attn_interface.py L1531-L1532
Note: Does not support backward pass.
```

**Comment：** 训练长序列和 serving decode 是两个不同算子场景；不要用训练 forward/backward 的心智模型理解 KV cache API。

## 4. Paged KV 是 serving 系统接口

**Explain：** 普通 KV cache 是 `[batch, seqlen_cache, heads_k, head_dim]`；paged KV 把 cache 分成 blocks，用 `block_table` 描述每条序列的逻辑 token 到物理 block 的映射。

**Code：**

```python
# 来源：flash_attn/flash_attn_interface.py L1556-L1561
k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
    or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
    page_block_size must be a multiple of 256.
v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
    or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
```

**Comment：** 这和 SGLang/vLLM 的 paged KV 管理思想一致：上层 runtime 管内存分页，attention kernel 按 block table 读物理 cache。

