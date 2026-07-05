---
type: index-doc
title: "FlashAttention 导读与总览"
doc_type: moc
tags:
  - flash-attn/index-layer
  - flash-attn/doc/moc
updated: 2026-07-05
---

# FlashAttention 导读与总览

> AI infra 算子层入口：从 attention memory wall 到 CUDA/CuTe kernel。

## 本阶段目标

读完导读层后，你应能回答：

| 问题 | 要点 |
|------|------|
| FlashAttention 解决什么瓶颈 | 标准 attention 的中间矩阵读写导致 HBM 压力 |
| 为什么它不是近似 attention | 分块 + online softmax 保持精确结果 |
| FA1 到 FA4 如何演进 | FA1 是 IO-aware 算法原点，FA2 是主包重写，FA3/FA4 面向新 GPU 与新编译方式 |
| 为什么这是 AI infra 核心 | 训练吞吐、长上下文、KV cache、serving decode 都依赖 attention kernel |

## 阅读顺序

| 顺序 | 文档 | 说明 |
|------|------|------|
| 1 | [[FlashAttention-00-零基础先修]] | attention、GPU memory hierarchy、online softmax 直觉 |
| 2 | [[FlashAttention-代际演进]] | FA1、FA2、FA3、FA4 的演进与边界 |
| 3 | [[FlashAttention-01-项目总览]] | 仓库定位与 FA2/FA3/FA4 分层 |
| 4 | [[FlashAttention-02-架构分层]] | Python API、C++ binding、CUDA/CuTe kernel 边界 |
| 5 | [[FlashAttention-03-关键概念]] | 术语与核心机制 |
| 6 | [[FlashAttention-全链路Attention追踪]] | 一次 `flash_attn_func` 调用走到 kernel |
| 7 | [[FlashAttention-04-导读路径]] | 16 步原理驱动 tour |
| 8 | [[FlashAttention-05-文件地图]] | 按文件反查专题 |
| 9 | [[FlashAttention-术语表]] | 术语速查 |

## 专题入口

| 专题 | 入口 | 核心主题 |
|------|------|----------|
| Attention IO | [[FA01-Attention-IO-00-MOC]] | HBM/SRAM/register 与 memory wall |
| Online Softmax | [[FA02-Online-Softmax-00-MOC]] | 分块 softmax 的精确性 |
| Python API | [[FA03-Python-API-00-MOC]] | torch custom op、autograd、pybind |
| FA2 Forward | [[FA04-FA2-Forward-00-MOC]] | launch template 与 CUDA 主循环 |
| KV Cache | [[FA05-KV-Cache-00-MOC]] | decode、append KV、paged KV、splitKV |
| Hopper/CuTe | [[FA06-Hopper-CuTe-00-MOC]] | FA3/FA4、TMA/GMMA、JIT cache |

## 学习方式

**Explain：** FlashAttention 的源码并不是一个普通 Python 库。它的核心价值藏在“为什么这样切 tile、为什么要保存 LSE、为什么会重算、为什么不同硬件用不同 kernel”这些问题里。阅读时先建立原理模型，再用源码验证模型。

**Code：**

```python
# 来源：flash_attn/flash_attn_interface.py L1156-L1167
def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
```

**Comment：**
- 这些参数正好对应后续专题：dropout、causal/local mask、softcap、ALiBi、deterministic backward。
- 读 API 参数比直接读 generated `.cu` 更适合建立第一层心智模型。

