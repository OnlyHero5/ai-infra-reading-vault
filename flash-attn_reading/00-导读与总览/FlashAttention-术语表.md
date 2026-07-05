---
type: index-doc
title: "FlashAttention 术语表"
doc_type: reference
tags:
  - flash-attn/index-layer
  - flash-attn/doc/reference
updated: 2026-07-04
---

# FlashAttention 术语表

| 术语 | 中文解释 |
|------|----------|
| HBM | GPU 高带宽显存，大但相对慢，attention 中间矩阵落 HBM 是主要瓶颈 |
| SRAM / shared memory | GPU SM 内共享内存，小但快，FlashAttention 将 tile 放在这里复用 |
| register | 线程私有寄存器，保存 score、softmax 状态、输出累积 |
| tile/block | Q/K/V 的分块单位，决定并行度、共享内存占用和 occupancy |
| online softmax | 流式更新 softmax 最大值与归一化分母的算法 |
| LSE | log-sum-exp，forward 保存给 backward 重算 softmax |
| causal mask | 自回归遮罩，query 只能看见当前位置及之前的 key |
| local attention | sliding window attention，只看窗口内 key |
| ALiBi | Attention with Linear Bias，对 attention score 加线性位置偏置 |
| softcap | 对 score 做 soft cap，减少极端 logits |
| dropout | 训练时对 attention probability 做随机丢弃 |
| MHA | Multi-Head Attention，Q/K/V head 数相同 |
| MQA | Multi-Query Attention，多个 Q head 共享一个 KV head |
| GQA | Grouped-Query Attention，多个 Q head 分组共享 KV head |
| varlen | 变长序列 batch，用 `cu_seqlens` 表达每条样本边界 |
| `cu_seqlens` | cumulative sequence lengths，形状通常为 `(batch + 1,)` |
| KV cache | 推理 decode 时缓存历史 K/V，避免重复计算 |
| paged KV | 将 KV cache 切成 page/block 管理，服务动态 batch 与长上下文 |
| splitKV | 将长 K/V 维度拆成多个并行分片，再 combine |
| TMA | Tensor Memory Accelerator，Hopper 上高效异步内存搬运机制 |
| GMMA | Hopper warpgroup 级矩阵乘指令族 |
| CuTe | CUTLASS 的 layout/tensor algebra 抽象 |
| CuTeDSL | 用 Python DSL 表达 CuTe kernel 并 JIT 编译 |
| kernel specialization | 针对 dtype、head_dim、mask、dropout 等生成不同 kernel |

