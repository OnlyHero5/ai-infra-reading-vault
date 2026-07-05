---
type: batch-doc
module: FA04-FA2-Forward
batch: "FA04"
doc_type: checkpoint
title: "FA2 CUDA Forward · checkpoint"
tags:
  - flash-attn/batch/fa04
  - flash-attn/module/fa2-forward
  - flash-attn/doc/checkpoint
updated: 2026-07-04
---

# FA2 CUDA Forward · checkpoint

## 自测问题

- [ ] 能从 `mha_fwd` 说出 dtype、stride、head_dim、head 数检查的原因。
- [ ] 能解释 `Flash_fwd_params` 为什么是 Python/C++ 与 CUDA kernel 的分界线。
- [ ] 能说明 `kBlockM`、`kBlockN`、`kHeadDim` 分别代表什么。
- [ ] 能把 `gemm(Q,K)`、mask、`softmax_rescale_o`、`gemm(P,V)` 对应到 forward 主循环。
- [ ] 能解释为什么主路径写回 `O` 和 `LSE`，而不是完整 `P`。
- [ ] 能解释 `BOOL_SWITCH`、`HEADDIM_SWITCH`、`DROPOUT_SWITCH` 的工程目的。

## 口述练习

用五分钟讲清楚：

> FA2 forward 如何让一个 query tile 扫描多个 K/V tile，并在 register 中维护 online softmax 状态，最终只写回 `O` 和 `softmax_lse`。

## 下一步

进入 [[FA05-KV-Cache-00-MOC]]，看当 attention 从训练/prefill 变成 decode 时，KV cache、SplitKV、paged KV 如何改变 kernel 形态。

