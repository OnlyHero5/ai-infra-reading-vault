---
type: batch-doc
module: FA02-Online-Softmax
batch: "FA02"
doc_type: faq
title: "Online Softmax · 关键问题"
tags:
  - flash-attn/batch/fa02
  - flash-attn/module/online-softmax
  - flash-attn/doc/faq
updated: 2026-07-05
---

# Online Softmax · 关键问题

## Q1：分块 softmax 为什么还能精确？

因为每行保存了足够的全局状态：当前最大值、归一化分母、输出累积。新 block 改变最大值时，历史状态会被缩放到新标尺。

## Q2：为什么保存 LSE 而不是 row sum？

LSE 更稳定，也更方便 backward 中结合 score 重算 probability。源码里 `softmax_lse` 使用 float 保存。

## Q3：dropout 会破坏重算吗？

不会，但需要保存随机数状态。Python autograd context 中保存了 `rng_state`，C++ 侧 forward 也会设置 Philox 随机状态。

## Q4：softcap 和 dropout 为什么有组合限制？

源码中 C++ 检查 `softcap > 0` 时不支持 dropout。这类限制来自 kernel 实现和数值路径组合，不是 attention 数学本身的限制。

**Code：**

```cpp
// 来源：csrc/flash_attn/flash_api.cpp L397-L397
if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }
```

**Comment：** 阅读限制时要区分“算法不可行”和“当前 kernel 未实现该组合”。

## Q5：为什么新 block 改变最大值时，`acc_o` 也要缩放？

**Explain：** `acc_o` 保存的是已经处理过的 `P V` 累积。如果新的 row max 变大，历史概率的标尺发生变化；只缩放 `row_sum` 不缩放 `acc_o` 会让分母和分子不一致。

**Code：**

```cpp
// 来源：csrc/flash_attn/src/softmax.h L146-L161
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
```

**Comment：** 这就是 online softmax 与 `PV` 融合时最容易漏掉的一步。

## Q6：`softmax_lse` 为什么比 `row_sum` 更适合作为 backward 保存值？

**Explain：** `row_sum` 依赖当前 row max 的标尺；LSE 把 max 和 sum 合成一个稳定标量。Backward 重算 score 后，用 LSE 可以直接恢复概率归一化。

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

**Comment：** LSE 是 `log(sum(exp(score)))` 的稳定表达，也正好是每个 query 行一个值。
