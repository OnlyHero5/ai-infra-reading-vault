---
type: index-doc
title: "FlashAttention 导读路径"
doc_type: walkthrough
tags:
  - flash-attn/index-layer
  - flash-attn/doc/walkthrough
updated: 2026-07-05
---

# FlashAttention 导读路径

> 16 步原理驱动 Guided Tour：先建立 AI infra 心智模型，再看源码证据。

## Step 0 · Attention 与 GPU 先修

**目标：** 理解 `QK^T → softmax → PV`、HBM/SRAM/register、warp/block、Tensor Core。

**阅读：** [[FlashAttention-00-零基础先修]]

**Code：**

```text
S = QK^T
P = softmax(S)
O = PV
```

**Comment：** 第一轮不要急着进 `.cu`，先确认自己能解释为什么 `S` 和 `P` 落 HBM 会昂贵。

## Step 1 · 代际演进与项目分层

**目标：** 区分 FA1、FA2、FA3、FA4、ROCm、模型生态，知道当前基线为什么主要走读 FA2/FA3/FA4。

**阅读：** [[FlashAttention-代际演进]] · [[FlashAttention-01-项目总览]] · [[FlashAttention-02-架构分层]]

**Code：**

```python
# 来源：flash_attn/cute/__init__.py L10-L13
from .interface import (
    flash_attn_func,
    flash_attn_varlen_func,
)
```

**Comment：** FA1 是 IO-aware 算法原点；FA4 有独立 CuTeDSL API，不等同于 FA2 的 `flash_attn_2_cuda` 路径。

## Step 2 · IO-aware 原理

**目标：** 解释为什么 FlashAttention 的核心是减少 HBM traffic。

**阅读：** [[FA01-Attention-IO-00-MOC]]

**→ 下一站：** online softmax 精确性。

## Step 3 · Online Softmax

**目标：** 能写出按 block 更新 row max、row sum、O accumulator 的直觉。

**阅读：** [[FA02-Online-Softmax-00-MOC]]

**→ 下一站：** Python API 如何表达这些需求。

## Step 4 · Python 公开 API

**目标：** 理解普通、packed、varlen、KV cache 四类 API。

**阅读：** [[FA03-Python-API-00-MOC]]

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

**Comment：** 参数就是后续 kernel specialization 的来源。

## Step 5 · Autograd 与 custom_op

**目标：** 理解 `torch.library.custom_op`、fake tensor、forward/backward context。

**阅读：** [[FA03-Python-API-02-源码走读]]

## Step 6 · C++ Binding

**目标：** 从 `flash_attn_gpu.fwd` 走到 pybind `m.def("fwd")`。

**阅读：** [[FA03-Python-API-03-数据流与交互]]

## Step 7 · 参数装配

**目标：** 理解 `Flash_fwd_params`/`Flash_bwd_params` 存哪些指针、shape、mask 与随机数状态。

**阅读：** [[FA04-FA2-Forward-01-核心概念]]

## Step 8 · Kernel Dispatch

**目标：** 理解 dtype/head_dim/causal/dropout/local/alibi/softcap 如何变成 template 参数。

**阅读：** [[FA04-FA2-Forward-02-源码走读]]

## Step 9 · Forward Kernel 主循环

**目标：** 串起 QK、mask、online softmax、dropout、PV、LSE/O 写回。

**阅读：** [[FA04-FA2-Forward-03-数据流与交互]]

## Step 10 · Backward 原理

**目标：** 理解为什么 backward 重算 score，保存 LSE 而不保存 attention matrix。

**阅读：** [[FA02-Online-Softmax-04-关键问题]]

## Step 11 · Varlen 与 Padding

**目标：** 理解 `cu_seqlens`、`unpad_input`、`pad_input` 对训练吞吐的意义。

**阅读：** [[FA03-Python-API-03-数据流与交互]]

## Step 12 · Decode 与 KV Cache

**目标：** 理解 prefill 和 decode 为什么是不同 kernel 形态。

**阅读：** [[FA05-KV-Cache-00-MOC]]

## Step 13 · SplitKV 与 Paged KV

**目标：** 理解长上下文 serving 中 K/V 如何拆分、分页、合并。

**阅读：** [[FA05-KV-Cache-02-源码走读]]

## Step 14 · FA3 Hopper

**目标：** 理解 Hopper 上 TMA/GMMA、persistent scheduling、combine kernel 的意义。

**阅读：** [[FA06-Hopper-CuTe-01-核心概念]]

## Step 15 · FA4 CuTeDSL

**目标：** 理解 JIT compile/cache 为什么成为新路径。

**阅读：** [[FA06-Hopper-CuTe-02-源码走读]]

## Step 16 · 测试与生产判断

**目标：** 用测试矩阵反推功能边界：dtype、head_dim、causal、local、ALiBi、softcap、KV cache。

**阅读：** [[FlashAttention-90-总结复盘-00-MOC]]

## 收官

完成导读后，用 [[FlashAttention-全链路Attention追踪]] 把一次调用从 Python 走到 kernel，再回看 [[91_dashboard/cross-library-map]]，理解它在 SGLang/Slime 栈中的位置。

