---
type: batch-doc
module: FA05-KV-Cache
batch: "FA05"
doc_type: checkpoint
title: "KV Cache 与推理特性 · checkpoint"
tags:
  - flash-attn/batch/fa05
  - flash-attn/module/kv-cache
  - flash-attn/doc/checkpoint
updated: 2026-07-04
---

# KV Cache 与推理特性 · checkpoint

## 自测问题

- [ ] 能解释 prefill 和 decode 的 attention 形态差异。
- [ ] 能说明 `flash_attn_with_kvcache` 为什么不支持 backward。
- [ ] 能从 Python API 追到 C++ `mha_fwd_kvcache` 和 pybind `fwd_kvcache`。
- [ ] 能解释 `k_cache/v_cache` 的 dense 布局与 paged KV 布局。
- [ ] 能说明 `block_table`、`cache_seqlens`、`cache_batch_idx`、`leftpad_k` 分别解决什么问题。
- [ ] 能解释为什么有新 K/V、paged KV 或 `cache_batch_idx` 时会强制 SplitKV。
- [ ] 能说明 SplitKV 的 partial `out_accum/softmax_lse_accum` 为什么需要 combine。

## 口述练习

用五分钟讲清楚：

> 一次 decode step 中，新 token 的 K/V 如何写入 cache，query 如何读取旧 cache 做 attention，paged KV 和 SplitKV 分别解决什么系统问题。

## 下一步

进入 [[FA06-Hopper-CuTe-00-MOC]]，理解为什么新 GPU 架构和 CuTeDSL/JIT 会产生 FA3/FA4 这条新实现路径。

