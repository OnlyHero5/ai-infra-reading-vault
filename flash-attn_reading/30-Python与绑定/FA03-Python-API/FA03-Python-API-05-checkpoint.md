---
type: batch-doc
module: FA03-Python-API
batch: "FA03"
doc_type: checkpoint
title: "Python API 与绑定 · checkpoint"
tags:
  - flash-attn/batch/fa03
  - flash-attn/module/python-api
  - flash-attn/doc/checkpoint
updated: 2026-07-04
---

# Python API 与绑定 · checkpoint

## 自测问题

- [ ] 能区分 `flash_attn_func`、`flash_attn_qkvpacked_func`、`flash_attn_varlen_func`、`flash_attn_with_kvcache` 的使用场景。
- [ ] 能解释 `cu_seqlens` 为什么是 varlen kernel 的边界数组。
- [ ] 能从 `flash_attn_func` 追到 `_flash_attn_forward`，再追到 `flash_attn_gpu.fwd`。
- [ ] 能说明 `softmax_lse` 为什么比保存完整 attention matrix 更适合 backward。
- [ ] 能解释 `return_attn_probs` 为什么不是生产主路径。
- [ ] 能说明 `flash_attn_2_cuda`、pybind `m.def("fwd")` 和 C++ `mha_fwd` 的关系。

## 口述练习

用三分钟讲清楚：

> 一个 batch 内序列长短不同的训练样本，如何通过 `unpad_input → cu_seqlens → varlen_fwd → pad_input` 进入 FlashAttention，并在保持 correctness 的同时减少 padding token 计算。

## 下一步

进入 [[FA04-FA2-Forward-00-MOC]]，沿着 `mha_fwd → Flash_fwd_params → run_mha_fwd → flash_fwd_kernel` 看 CUDA forward 主路径。

