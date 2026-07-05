---
type: stage-moc
module: FlashAttention-Summary
batch: "FA90"
doc_type: moc
title: "FlashAttention 总结复盘"
tags:
  - flash-attn/batch/fa90
  - flash-attn/module/summary
  - flash-attn/doc/moc
updated: 2026-07-05
---

# FlashAttention 总结复盘

> 用 AI infra 视角把 FlashAttention 串成一条线：代际演进 → memory wall → online softmax → kernel specialization → KV cache → 新 GPU 后端。

## 复盘路径

| 顺序 | 文档 | 复盘问题 |
|------|------|----------|
| 1 | [[FlashAttention-代际演进]] | FA1 到 FA4 的问题重心如何变化 |
| 2 | [[FA01-Attention-IO-00-MOC]] | 标准 attention 为什么被 HBM traffic 卡住 |
| 3 | [[FA02-Online-Softmax-00-MOC]] | 分块后如何保持 softmax 精确 |
| 4 | [[FA03-Python-API-00-MOC]] | 上层框架如何进入 CUDA extension |
| 5 | [[FA04-FA2-Forward-00-MOC]] | FA2 forward 如何执行 tile attention |
| 6 | [[FA05-KV-Cache-00-MOC]] | decode serving 为什么需要 KV cache 专门路径 |
| 7 | [[FA06-Hopper-CuTe-00-MOC]] | FA3/FA4 为什么适配新 GPU 与 JIT 编译 |

## 一张总图

```mermaid
flowchart TB
    IO["Attention IO<br/>避免 S/P 落 HBM"]
    OS["Online Softmax<br/>row_max row_sum acc_o"]
    API["Python API<br/>dense packed varlen kvcache"]
    CPP["C++ Params<br/>Flash_fwd_params"]
    DIS["Dispatch<br/>dtype head_dim mask split"]
    KER["CUDA Kernel<br/>QK softmax PV"]
    DEC["Decode<br/>KV cache paged SplitKV"]
    NEW["FA3/FA4<br/>Hopper CuTe JIT"]
    GEN["FA1 to FA4<br/>算法原点到 JIT 后端"]
    GEN --> IO --> OS --> API --> CPP --> DIS --> KER --> DEC --> NEW
```

## 必须掌握的六句话

1. FA1 是 IO-aware exact attention 的算法原点；当前源码主线主要从 FA2、FA3、FA4 展开。
2. FlashAttention 的核心不是少算 attention，而是避免把 `S=QK^T` 和 `P=softmax(S)` 完整写入 HBM。
3. Online softmax 让每个 query 行在分块扫描 K/V 时维护 `row_max`、`row_sum` 和 `acc_o`，从而得到与全量 softmax 等价的结果。
4. Python API 的 `causal/window/ALiBi/softcap/dropout/varlen/KV cache` 参数最终会影响 C++ 参数和 CUDA template dispatch。
5. 训练/prefill 与 decode 是不同 attention workload；decode 需要 KV cache、paged KV、SplitKV 和 `seqlen_q=1` 优化。
6. FA3/FA4 不是原理变化，而是为了新 GPU 架构、新特性组合和更可维护的 kernel 编译路径。

## AI infra 对照

| 系统层 | 你应该能联系到 |
|--------|----------------|
| Slime | RL rollout 依赖 serving 引擎，不直接实现 attention kernel |
| SGLang | serving forward 需要 attention backend，KV cache 管理与 decode kernel 强相关 |
| FlashAttention | attention backend 的底层 IO-aware kernel 案例 |
| GPU | HBM/SRAM/register/Tensor Core 决定 kernel 设计 |

## 自测总题

- [ ] 不看源码，能画出 `flash_attn_func → custom_op → flash_attn_2_cuda.fwd → mha_fwd → run_mha_fwd → flash_fwd_kernel`。
- [ ] 能说明 FA1、FA2、FA3、FA4 分别解决什么层面的问题。
- [ ] 能解释为什么 `softmax_lse` 足够支持 backward 重算。
- [ ] 能说明 varlen 的 `cu_seqlens` 和 paged KV 的 `block_table` 分别解决什么问题。
- [ ] 能解释为什么 kernel dispatch 维度会导致编译实例数量很多。
- [ ] 能说清楚 FA4 JIT cache 对生产 warmup 的影响。

## 入口回跳

[[FlashAttention源码阅读指南]] · [[FlashAttention-04-导读路径]] · [[FlashAttention-全链路Attention追踪]] · [[91_dashboard/cross-library-map]]
