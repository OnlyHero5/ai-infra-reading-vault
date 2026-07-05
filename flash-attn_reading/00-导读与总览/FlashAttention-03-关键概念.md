---
type: index-doc
title: "FlashAttention 关键概念"
doc_type: concept
tags:
  - flash-attn/index-layer
  - flash-attn/doc/concept
updated: 2026-07-04
---

# FlashAttention 关键概念

| 概念 | 一句话解释 | 深入 |
|------|------------|------|
| IO-aware | 以 HBM/SRAM/register 读写成本为中心设计 kernel | [[FA01-Attention-IO-01-核心概念]] |
| Online Softmax | 分块更新每行 max 与 sum，避免保存完整 `P` | [[FA02-Online-Softmax-01-核心概念]] |
| LSE | log-sum-exp，forward 保存给 backward 重算 | [[FA02-Online-Softmax-03-数据流与交互]] |
| QKV packed | Q/K/V 合并存储，backward 避免显式 concat | [[FA03-Python-API-01-核心概念]] |
| Varlen | 用 `cu_seqlens` 表达变长 batch，减少 padding 计算 | [[FA03-Python-API-03-数据流与交互]] |
| GQA/MQA | Q heads 多于 KV heads，推理常用 | [[FA05-KV-Cache-01-核心概念]] |
| SplitKV | 把长 K/V 拆给多个 CTA，最后 combine | [[FA05-KV-Cache-03-数据流与交互]] |
| Paged KV | KV cache 以 page/block 管理，服务长上下文与动态 batch | [[FA05-KV-Cache-02-源码走读]] |
| TMA/GMMA | Hopper 时代的异步拷贝与矩阵指令 | [[FA06-Hopper-CuTe-01-核心概念]] |
| CuTeDSL | 用 Python DSL 表达 CUTLASS/CuTe kernel，并 JIT 编译 | [[FA06-Hopper-CuTe-02-源码走读]] |

## 一个核心公式

FlashAttention 的关键状态可简化为每行三个变量：

```text
m_i = 当前已处理 key block 的最大 score
l_i = 当前已处理 key block 的 exp 累积和
o_i = 当前已处理 key block 的 value 加权累积
```

每来一个新的 K/V block，就更新这三个量。最后输出 `o_i / l_i`，并保存 `log(l_i) + m_i` 作为 LSE。

## 源码锚点

**Explain：** CUDA forward 主循环中，score tile 先进入 `acc_s`，mask/softcap 后进入 online softmax，再转换为低精度概率块乘 V。

**Code：**

```cpp
// 来源：csrc/flash_attn/src/flash_fwd_kernel.h L319-L367
FLASH_NAMESPACE::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
    acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
    smem_thr_copy_Q, smem_thr_copy_K
);
if constexpr (Is_softcap){
    FLASH_NAMESPACE::apply_softcap(acc_s, params.softcap);
}

mask.template apply_mask<Is_causal, Is_even_MN>(
    acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
);

masking_step == 0
    ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
    : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
```

**Comment：**
- `acc_s` 是局部 `QK^T` score。
- `softmax_rescale_o` 同时更新 softmax 状态与 `acc_o` 的缩放。
- `gemm_rs` 将局部概率块与 V 相乘并累积到输出。

