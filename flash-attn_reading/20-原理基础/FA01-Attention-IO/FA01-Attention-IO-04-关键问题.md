---
type: batch-doc
module: FA01-Attention-IO
batch: "FA01"
doc_type: faq
title: "Attention IO · 关键问题"
tags:
  - flash-attn/batch/fa01
  - flash-attn/module/attention-io
  - flash-attn/doc/faq
updated: 2026-07-05
---

# Attention IO · 关键问题

## Q1：FlashAttention 是近似 attention 吗？

不是。FlashAttention 改变的是执行顺序和内存访问方式，不改变 softmax attention 的数学定义。分块计算通过 online softmax 保持精确结果。

## Q2：为什么少写 HBM 会这么重要？

Transformer 训练和长上下文推理中，attention 的中间矩阵规模是 `N x N`。即使 GPU matmul 很快，反复把大矩阵写入/读出 HBM 也会成为瓶颈。

## Q3：为什么会愿意重算？

Backward 如果保存完整 attention probability，会占用大量显存。FlashAttention 保存更紧凑的 LSE，在 backward 中重算局部 score，以计算换显存和 HBM traffic。

## Q4：IO-aware 和普通 CUDA 优化有什么区别？

普通 CUDA 优化可能关注 occupancy、warp divergence、bank conflict。IO-aware 的第一性问题是：哪些中间结果必须跨层级存储，哪些可以在 tile 内消费掉。FlashAttention 的设计起点是后者。

## Q5：为什么 head_dim 会影响 kernel？

head_dim 影响 vectorized load、shared memory layout、register pressure、Tensor Core tile shape。因此源码会按 head_dim 显式实例化大量 kernel。

## Q6：`p_ptr` 存在是否说明仍保存完整 attention matrix？

**Explain：** 不说明。`p_ptr` 是 optional `return_softmax` 路径。C++ 入口明确只有 `return_softmax` 时才分配 `p`，并且要求 dropout 打开；常规路径传 `nullptr`。

**Code：**

```cpp
// 来源：csrc/flash_attn/flash_api.cpp L441-L464
auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
at::Tensor p;
// Only return softmax if there's dropout to reduce compilation time
if (return_softmax) {
    TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
    p = torch::empty({ batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded }, opts);
}

set_params_fprop(params,
                 batch_size,
                 seqlen_q, seqlen_k,
                 seqlen_q_rounded, seqlen_k_rounded,
                 num_heads, num_heads_k,
                 head_size, head_size_rounded,
                 q, k, v, out,
                 /*cu_seqlens_q_d=*/nullptr,
                 /*cu_seqlens_k_d=*/nullptr,
                 /*seqused_k=*/nullptr,
                 return_softmax ? p.data_ptr() : nullptr,
                 softmax_lse.data_ptr(),
```

**Comment：** 判断 IO 模式时以常规调用路径为准，不要只看结构体里有没有某个指针字段。

## Q7：为什么要看 shared memory layout，而不是只看算法公式？

**Explain：** 算法公式只说明可以分块；性能来自每个块如何被搬运和复用。`kernel_traits.h` 里 `kBlockKSmem`、`kGmemThreadsPerRow`、`GmemTiledCopyQKV` 决定一次 global memory 访问如何落到 shared memory。

**Code：**

```cpp
// 来源：csrc/flash_attn/src/kernel_traits.h L111-L137
static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                              Stride<Int<kGmemThreadsPerRow>, _1>>;

using Gmem_copy_struct = std::conditional_t<
    Has_cp_async,
    SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
    AutoVectorizingCopyWithAssumedAlignment<128>
>;
using GmemTiledCopyQKV = decltype(
    make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                    GmemLayoutAtom{},
                    Layout<Shape<_1, _8>>{}));
```

**Comment：** FlashAttention 的“IO-aware”最终要落在这种 copy atom、layout 和 tile 参数上。
