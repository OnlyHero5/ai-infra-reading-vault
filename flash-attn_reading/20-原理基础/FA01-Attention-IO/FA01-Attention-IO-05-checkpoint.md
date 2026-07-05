---
type: batch-doc
module: FA01-Attention-IO
batch: "FA01"
doc_type: checkpoint
title: "Attention IO · checkpoint"
tags:
  - flash-attn/batch/fa01
  - flash-attn/module/attention-io
  - flash-attn/doc/checkpoint
updated: 2026-07-05
---

# Attention IO · checkpoint

## 读者自测

- [ ] 能解释标准 attention 为什么会产生 `N x N` 中间矩阵
- [ ] 能说明 HBM、shared memory、register 在 FlashAttention 中的职责
- [ ] 能解释为什么 FlashAttention 不是近似算法
- [ ] 能说出 `softmax_lse` 为什么重要
- [ ] 能用自己的话解释“多算一点，少搬很多”的系统取舍

## 源码定位

- [ ] 能在 `flash.h` 找到 `softmax_lse_ptr`
- [ ] 能在 `kernel_traits.h` 找到 `kBlockM/kBlockN`
- [ ] 能在 `flash_fwd_kernel.h` 找到 `acc_s → softmax → acc_o` 主链路
- [ ] 能说明 `p_ptr` 与 `return_softmax ? p.data_ptr() : nullptr` 的关系
- [ ] 能把 `mQ/gQ/sQ/tSrQ` 分别归到 HBM view、HBM tile、shared memory、register fragment
- [ ] 能指出 epilogue 中 `O` 与 `LSE` 的写回位置
