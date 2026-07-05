---
type: batch-doc
module: FA06-Hopper-CuTe
batch: "FA06"
doc_type: checkpoint
title: "FA3/FA4 Hopper 与 CuTe · checkpoint"
tags:
  - flash-attn/batch/fa06
  - flash-attn/module/hopper-cute
  - flash-attn/doc/checkpoint
updated: 2026-07-04
---

# FA3/FA4 Hopper 与 CuTe · checkpoint

## 自测问题

- [ ] 能说明 FA2、FA3、FA4 在源码目录和实现方式上的差异。
- [ ] 能解释 FA3 `run_mha_fwd` 中 arch、SplitKV、paged KV、PackGQA、softcap 这些 dispatch 维度。
- [ ] 能从 `flash_attn.cute.flash_attn_func` 追到 `_flash_attn_fwd`。
- [ ] 能解释 FA4 为什么需要 `_get_device_arch`、compile key 和 `compile_cache`。
- [ ] 能说明 CuTeDSL 没有改变 IO-aware attention 原理，只改变 kernel 组织方式。
- [ ] 能指出 FP8 在 FA4 中的关键限制。
- [ ] 能说明 JIT cache 对 serving warmup 和形状稳定性的影响。

## 口述练习

用五分钟讲清楚：

> 为什么 FlashAttention 在 FA2 之外还需要 FA3/FA4 路径，以及 CuTeDSL/JIT cache 对新 GPU 架构适配和生产 serving 会带来哪些收益与风险。

## 收官

回到 [[FlashAttention-90-总结复盘-00-MOC]]，把 IO-aware 原理、online softmax、Python/C++/CUDA 绑定、KV cache 和 FA3/FA4 演进串成一条完整 AI infra 知识链。

