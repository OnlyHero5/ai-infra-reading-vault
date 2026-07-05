---
type: module-moc
module: FA05-KV-Cache
batch: "FA05"
doc_type: moc
title: "KV Cache 与推理特性"
tags:
  - flash-attn/batch/fa05
  - flash-attn/module/kv-cache
  - flash-attn/doc/moc
updated: 2026-07-04
---

# KV Cache 与推理特性

> 从训练/prefill 的 full attention，切换到 serving decode 的 cache attention。

## 阅读顺序

| 顺序 | 文件 | 目标 |
|------|------|------|
| 01 | [[FA05-KV-Cache-01-核心概念]] | 理解 prefill、decode、KV cache、paged KV |
| 02 | [[FA05-KV-Cache-02-源码走读]] | 走读 `flash_attn_with_kvcache → fwd_kvcache → splitkv` |
| 03 | [[FA05-KV-Cache-03-数据流与交互]] | 看 cache update、RoPE、block table、SplitKV 数据流 |
| 04 | [[FA05-KV-Cache-04-关键问题]] | 解答 backward、cache batch idx、leftpad、num_splits |
| 05 | [[FA05-KV-Cache-05-checkpoint]] | 自测是否能解释 decode attention |

## 核心源码

| 文件 | 作用 |
|------|------|
| `flash_attn/flash_attn_interface.py` | `flash_attn_with_kvcache` Python API |
| `csrc/flash_attn/flash_api.cpp` | `mha_fwd_kvcache` C++ 入口 |
| `csrc/flash_attn/src/flash_fwd_launch_template.h` | SplitKV kernel 与 combine kernel |
| `tests/test_flash_attn.py` | KV cache 测试矩阵 |

## 一句话

**Explain：** KV cache 路径把“更新新 K/V、可选 RoPE、读取旧 cache、执行 attention”合并在一次 forward 中，目标是服务 incremental decode 的低延迟与高吞吐。

