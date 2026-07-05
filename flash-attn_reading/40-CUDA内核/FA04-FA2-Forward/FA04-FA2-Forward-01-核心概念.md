---
type: batch-doc
module: FA04-FA2-Forward
batch: "FA04"
doc_type: concept
title: "FA2 CUDA Forward · 核心概念"
tags:
  - flash-attn/batch/fa04
  - flash-attn/module/fa2-forward
  - flash-attn/doc/concept
updated: 2026-07-04
---

# FA2 CUDA Forward · 核心概念

## 1. `Flash_fwd_params` 是 kernel 的输入契约

**Explain：** Python 和 C++ 的张量对象不会直接进入 CUDA 主循环。CUDA kernel 消费的是 `Flash_fwd_params`：输入输出指针、stride、shape、mask、softmax scale、cu_seqlens、KV cache metadata 等都被压进这个结构体。

**Code：**

```cpp
// 来源：csrc/flash_attn/src/flash.h L48-L75
struct Flash_fwd_params : public Qkv_params {
    void * __restrict__ o_ptr;
    void * __restrict__ oaccum_ptr;

    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    void * __restrict__ p_ptr;

    void * __restrict__ softmax_lse_ptr;
    void * __restrict__ softmax_lseaccum_ptr;

    int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim, total_q;

    float scale_softmax;
    float scale_softmax_log2;
```

**Comment：** `p_ptr` 只是可选概率输出；主路径需要的是 `o_ptr` 和 `softmax_lse_ptr`。这与 [[FA01-Attention-IO-01-核心概念]] 的“不保存完整 attention matrix”一致。

## 2. Kernel traits 固化 tile 形状

**Explain：** `Flash_fwd_kernel_traits` 把 head_dim、query block、key block、warp 数、shared memory layout 固化成编译期常量。高性能来自“为具体形状编译专门 kernel”，不是一个通用 kernel 跑所有情况。

**Code：**

```cpp
// 来源：csrc/flash_attn/src/kernel_traits.h L51-L72
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, bool Is_Q_in_regs_=false, bool Share_Q_K_smem_=false, typename elem_type=cutlass::half_t,
         typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_traits : public Base {
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
```

**Comment：** `kBlockM` 是 query 行块，`kBlockN` 是 key/value 列块。FlashAttention 的“分块”在源码里就是这些 traits 的具体组合。

## 3. Online softmax 是 forward 主循环的数值核心

**Explain：** 每扫描一个 K/V block，kernel 需要把新 scores 融入已有输出累积。`Softmax` 维护每行 `row_max` 和 `row_sum`，并在最大值变化时重缩放历史 `acc_o`。

**Code：**

```cpp
// 来源：csrc/flash_attn/src/softmax.h L128-L142
template <int kNRows>
struct Softmax {
    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max, row_sum;

    template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1>
    __forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
        Tensor scores = make_tensor(acc_s.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_s.layout()));
        if (Is_first) {
            FLASH_NAMESPACE::template reduce_max</*zero_init=*/true>(scores, row_max);
            FLASH_NAMESPACE::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            FLASH_NAMESPACE::reduce_sum</*zero_init=*/true>(scores, row_sum);
```

**Comment：** 这里连接 [[FA02-Online-Softmax-01-核心概念]]：分块扫描仍能得到全量 softmax 等价结果。

## 4. Dispatch 分两层

**Explain：** 第一层按 dtype/head_dim/causal/SplitKV 选择大类；第二层按 even shape、local、ALiBi、softcap、dropout 等条件选择具体模板参数。

**Code：**

```cpp
// 来源：csrc/flash_attn/flash_api.cpp L243-L253
void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                if (params.num_splits <= 1 && !force_split_kernel) {
                    run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                } else {
                    run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                }
            });
        });
    });
}
```

**Comment：** `HEADDIM_SWITCH` 是理解源码文件数量的关键：不同 head_dim 会选择不同 traits 和编译单元。

