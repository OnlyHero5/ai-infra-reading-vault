---
type: module-moc
module: FA04-FA2-Forward
batch: "FA04"
doc_type: moc
title: "FA2 CUDA Forward"
tags:
  - flash-attn/batch/fa04
  - flash-attn/module/fa2-forward
  - flash-attn/doc/moc
updated: 2026-07-04
---

# FA2 CUDA Forward

> 从 `mha_fwd` 到 `flash_fwd_kernel`，看 FlashAttention v2 如何把 IO-aware attention 落到 CUDA kernel。

## 阅读顺序

| 顺序 | 文件 | 目标 |
|------|------|------|
| 01 | [[FA04-FA2-Forward-01-核心概念]] | 理解 `Flash_fwd_params`、tile、kernel traits |
| 02 | [[FA04-FA2-Forward-02-源码走读]] | 走读 `mha_fwd → run_mha_fwd → kernel launch` |
| 03 | [[FA04-FA2-Forward-03-数据流与交互]] | 串起 QK、mask、online softmax、PV、O/LSE 写回 |
| 04 | [[FA04-FA2-Forward-04-关键问题]] | 解答 template 爆炸、head_dim、dropout、mask 的问题 |
| 05 | [[FA04-FA2-Forward-05-checkpoint]] | 自测是否能口述 forward 主循环 |

## 核心源码

| 文件 | 作用 |
|------|------|
| `csrc/flash_attn/flash_api.cpp` | C++ 入口、shape/dtype 检查、参数装配 |
| `csrc/flash_attn/src/flash.h` | `Flash_fwd_params` 结构体 |
| `csrc/flash_attn/src/kernel_traits.h` | block size、warp 数、shared memory layout |
| `csrc/flash_attn/src/flash_fwd_launch_template.h` | head_dim/dtype/mask/dropout dispatch |
| `csrc/flash_attn/src/flash_fwd_kernel.h` | forward kernel 主循环 |
| `csrc/flash_attn/src/softmax.h` | online softmax 工具 |

## 一句话

**Explain：** FA2 forward 的核心是：每个 thread block 负责一块 query 行，分块扫描 K/V，在 register 中维护 softmax 状态和输出累积，只把最终 `O` 和 `LSE` 写回 HBM。

