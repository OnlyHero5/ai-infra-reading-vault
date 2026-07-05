---
type: batch-doc
module: FA02-Online-Softmax
batch: "FA02"
doc_type: concept
title: "Online Softmax · 核心概念"
tags:
  - flash-attn/batch/fa02
  - flash-attn/module/online-softmax
  - flash-attn/doc/concept
updated: 2026-07-05
---

# Online Softmax · 核心概念

## 1. 为什么普通分块 softmax 会错

softmax 每一行需要看完整行的最大值和分母：

```text
softmax(x_j) = exp(x_j - max(x)) / sum_k exp(x_k - max(x))
```

如果每个 block 独立 softmax，分母只覆盖局部 block，结果就不是全局 softmax。

## 2. Online Softmax 状态

对每一行维护三个状态：

| 状态 | 含义 |
|------|------|
| `m_i` | 已处理 key block 的最大 score |
| `l_i` | 在当前 `m_i` 标尺下的 exp 累积和 |
| `o_i` | 当前归一化标尺下的 V 加权累积 |

新 block 到来时，如果新最大值变大，历史 `l_i` 和 `o_i` 都要按比例缩放。

## 3. LSE 的作用

**Explain：** Forward 最终保存 `softmax_lse`。Backward 有了 Q/K/V/O/dO 和 LSE，就可以重算局部 score 与 probability，不需要保存完整 `P`。

**Code：**

```python
# 来源：flash_attn/flash_attn_interface.py L485-L499
out_padded, softmax_lse, S_dmask, rng_state =  _wrapped_flash_attn_forward(
    q,
    k,
    v,
    dropout_p,
    softmax_scale,
    causal=causal,
    window_size_left=window_size[0],
    window_size_right=window_size[1],
    softcap=softcap,
    alibi_slopes=alibi_slopes,
    return_softmax=return_softmax and dropout_p > 0,
)
if is_grad:
    ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
```

**Comment：** `softmax_lse` 是 forward 到 backward 的压缩摘要。

## 4. 为什么这是精确算法

Online softmax 保存的是足以恢复全局归一化的信息，不是采样或稀疏近似。因此 FlashAttention 仍然是 exact attention。

## 5. `row_max` / `row_sum` 是源码里的真实状态

**Explain：** `Softmax<kNRows>` 明确持有 `row_max` 与 `row_sum`。第一次 K block 初始化这两个状态；后续 K block 会先把旧状态 rescale 到新的 max 标尺，再累加当前 block。

**Code：**

```cpp
// 来源：csrc/flash_attn/src/softmax.h L128-L166
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
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            FLASH_NAMESPACE::template reduce_max</*zero_init=*/false>(scores, row_max);
            Tensor acc_o_rowcol = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale;
                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
            }
            FLASH_NAMESPACE::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            FLASH_NAMESPACE::reduce_sum</*zero_init=*/false>(scores, row_sum);
        }
    };
```

**Comment：** 这段代码直接对应公式：新最大值改变后，旧分母和旧输出累积都必须乘上同一个 `exp(m_old - m_new)` 比例。

## 6. LSE 同时服务 forward 输出归一化和 backward 重算

**Explain：** `normalize_softmax_lse` 在最后对 `acc_o` 归一化，并返回 `lse = row_max * scale + log(row_sum)`。Backward 只需要 `q/k/v/out/dout/softmax_lse/rng_state`，不需要完整 `P`。

**Code：**

```cpp
// 来源：csrc/flash_attn/src/softmax.h L169-L185
template<bool Is_dropout=false, bool Split=false, typename Tensor0>
__forceinline__ __device__ TensorT normalize_softmax_lse(Tensor0 &acc_o, float softmax_scale, float rp_dropout=1.0) {
    SumOp<float> sum_op;
    quad_allreduce_(row_sum, row_sum, sum_op);
    TensorT lse = make_fragment_like(row_sum);
    Tensor acc_o_rowcol = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
    #pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
        float sum = row_sum(mi);
        float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
        lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : row_max(mi) * softmax_scale + __logf(sum);
        float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
        #pragma unroll
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scale; }
    }
    return lse;
};
```

**Comment：** `lse` 是“每行一个标量”的压缩摘要；它把 full `P` 的 `seqlen_q * seqlen_k` 状态压成 `seqlen_q` 状态。
