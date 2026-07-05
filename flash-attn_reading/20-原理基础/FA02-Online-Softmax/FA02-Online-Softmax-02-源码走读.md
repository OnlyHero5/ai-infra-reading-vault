---
type: batch-doc
module: FA02-Online-Softmax
batch: "FA02"
doc_type: walkthrough
title: "Online Softmax · 源码走读"
tags:
  - flash-attn/batch/fa02
  - flash-attn/module/online-softmax
  - flash-attn/doc/walkthrough
updated: 2026-07-05
---

# Online Softmax · 源码走读

## 1. Online Softmax 的状态模型

### 1.1 `Softmax` 只保存每行 max 与 sum

问题与约束：
- FlashAttention 按 K/V block 流式扫描，单个 kernel 不能把完整 attention score 矩阵留在片上。
- softmax 的数值稳定依赖按行最大值，输出 `O = softmax(S)V` 又依赖同一分母。
- 后续 K block 可能出现新的行最大值，历史分母和历史 `acc_o` 都必须能被重新标尺化。

设计选择：
- `Softmax<kNRows>` 维护 `row_max` 与 `row_sum` 两个行向量，并把状态更新集中在 `softmax_rescale_o`。
- 调用方传入当前 score tile `acc_s` 和输出累积 `acc_o`，由 softmax 工具同时更新概率 tile 与历史输出。

Explain：
`row_max` 与 `row_sum` 是 online softmax 的最小状态：前者定义指数基准，后者记录当前基准下的分母。`softmax_rescale_o` 把当前 score tile 转成未归一化概率，并在必要时把历史 `acc_o` 缩放到新的基准。

来源：csrc/flash_attn/src/softmax.h L128-L142

Code：

```cpp
template <int kNRows>
struct Softmax {

    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max, row_sum;

    __forceinline__ __device__ Softmax() {};

    template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1>
    __forceinline__ __device__ void softmax_rescale_o(
        Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2
    ) {
        Tensor scores = make_tensor(acc_s.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_s.layout()));
```

代码逻辑：
- 模板参数 `kNRows` 固定当前 MMA tile 中的 query 行数。
- `row_max`、`row_sum` 使用同一个行向量类型保存每行状态。
- `softmax_rescale_o` 接收当前 score accumulator 与输出 accumulator。
- `acc_s` 被转换成 row/col 布局，后续 reduce 和逐行缩放都基于这个视图。

为什么这样写：
- 两个行向量足以把跨 block softmax 拆成增量更新，而不保存完整 `P`。
- `acc_o` 与 softmax 状态放在同一个函数里更新，避免分母基准变了但输出分子仍在旧基准上。
- row/col 视图让后续代码按 query 行处理，贴合 softmax 的数学维度。

不变量与失败模式：
- `size<0>(scores)` 必须等于 `kNRows`，源码用 `static_assert` 固化这个约束。
- `acc_s` 在函数内会被原地改写为未归一化概率 tile，调用方不能再把它当 raw score 使用。
- `row_max` 与 `row_sum` 的生命周期覆盖一个 query tile 对所有 K/V block 的扫描。

Comment：
这里的关键不是“做 softmax”，而是把跨 block softmax 压缩成 `row_max/row_sum/acc_o` 三个片上状态。

### 1.2 Forward 主循环按 first/non-first 选择更新路径

问题与约束：
- 第一块 K/V 没有历史状态，后续块则必须和历史状态合并。
- causal/local mask 可能让某些行出现全 `-inf`，后续更新需要知道是否检查这个特殊情况。

设计选择：
- forward kernel 用 `masking_step == 0` 选择 `Is_first=true`，其余 block 使用 `Is_first=false`。
- `Check_inf` 由 `Is_causal || Is_local` 决定，把 mask 场景传给 softmax 更新函数。

Explain：
主循环在 mask 后立即调用 `softmax_rescale_o`。第一次调用初始化 `row_max/row_sum`；后续调用先重标尺历史状态，再合并当前 block 的概率贡献。

来源：csrc/flash_attn/src/flash_fwd_kernel.h L341-L344

Code：

```cpp
masking_step == 0
    ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(
        acc_s, acc_o, params.scale_softmax_log2
    )
    : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(
        acc_s, acc_o, params.scale_softmax_log2
    );
```

代码逻辑：
- 当前 block 的 score 已经完成 mask。
- `masking_step == 0` 进入初始化路径。
- 非首块进入增量路径，并把 causal/local 的 `-inf` 检查需求传下去。

为什么这样写：
- 初始化路径不需要读取旧状态，能少做一次历史缩放。
- 增量路径必须保留数学等价性，不能把每个 block 的 softmax 独立归一化后再相加。
- `Check_inf` 只在可能产生整行无效的 mask 场景启用，避免普通路径额外分支。

不变量与失败模式：
- `masking_step` 必须与 K block 扫描顺序一致，否则第一块与后续块的状态语义会颠倒。
- `acc_s` 必须先完成 mask 再进入 softmax；否则被 mask 的位置会贡献概率。
- `params.scale_softmax_log2` 必须和后续 LSE 使用的 softmax scale 保持一致。

Comment：
`Is_first` 是源码里 online softmax 从初始化切到增量状态机的入口。

### 1.3 Epilogue 统一归一化并生成 LSE

问题与约束：
- 主循环中 `acc_o` 一直是未归一化的概率加权和，只有扫完全部 K block 后分母才完整。
- backward 需要稳定的 softmax 归一化信息，但不能保存完整概率矩阵。

设计选择：
- epilogue 调用 `normalize_softmax_lse`，一次性归一化 `acc_o`，并返回每行 LSE。

Explain：
`normalize_softmax_lse` 把 online softmax 的最终状态收束为两个结果：归一化后的输出 tile 和 `softmax_lse`。`softmax_lse = row_max * scale + log(row_sum)` 是 backward 重算概率时需要的压缩证据。

来源：csrc/flash_attn/src/flash_fwd_kernel.h L433-L433

Code：

```cpp
Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(
    acc_o, params.scale_softmax, params.rp_dropout
);
```

代码逻辑：
- 主循环结束后调用 softmax epilogue。
- `acc_o` 被函数原地归一化。
- 返回值 `lse` 继续参与后续写回。

为什么这样写：
- 每个 block 都除以局部分母会破坏全局 softmax；等待最终分母完整后再除法才正确。
- LSE 保存的是每行归一化常数，比保存 `P` 小得多。
- 输出归一化与 LSE 生成共享 `row_max/row_sum`，避免重复计算。

不变量与失败模式：
- 调用 epilogue 前，所有相关 K/V block 必须已经贡献到 `row_sum` 和 `acc_o`。
- dropout 路径需要传入 `rp_dropout`，否则输出期望缩放不一致。
- 全 mask 或 NaN 行由 `normalize_softmax_lse` 内部保护，调用方仍要正确处理返回的 LSE。

Comment：
forward kernel 只在最后把 online 状态落成用户可见的 `O` 与训练可用的 `LSE`。

## 2. 行级归约与指数路径

### 2.1 行级 reduce 匹配 MMA fragment 布局

问题与约束：
- score tile 分布在线程持有的 CuTe tensor fragment 中，不是连续的全行数组。
- softmax 的 max 需要跨 lane 合并，sum 在部分路径中可以延后跨 lane reduce。

设计选择：
- `thread_reduce_` 先在每个线程本地按行归约，`quad_allreduce_` 再用 4-lane allreduce 合并。
- `reduce_max` 走完整 `reduce_`，`reduce_sum` 先做线程内累加，最终归一化时再补齐 lane 间合并。

Explain：
FlashAttention 不把 score tile 写回 shared/global memory 再归约，而是在寄存器 fragment 上按布局直接 reduce。这个设计让 online softmax 的行状态可以留在寄存器级别更新。

来源：csrc/flash_attn/src/softmax.h L23-L63

Code：

```cpp
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(
    Tensor<Engine0, Layout0> const &tensor,
    Tensor<Engine1, Layout1> &summary,
    Operator &op
) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(
    Tensor<Engine0, Layout0> &dst,
    Tensor<Engine1, Layout1> &src,
    Operator &op
) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max(
    Tensor<Engine0, Layout0> const& tensor,
    Tensor<Engine1, Layout1> &max
) {
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(
    Tensor<Engine0, Layout0> const& tensor,
    Tensor<Engine1, Layout1> &sum
) {
    SumOp<float> sum_op;
    thread_reduce_<zero_init>(tensor, sum, sum_op);
}
```

代码逻辑：
- `thread_reduce_` 要求输入是二维 tensor、输出是一维行摘要。
- 每一行先用本线程持有的列片段做归约。
- `quad_allreduce_` 把 4 个 lane 的摘要合并。
- `reduce_max` 调 `reduce_`，因此包含线程内 reduce 与 quad allreduce。
- `reduce_sum` 只调用线程内 reduce。

为什么这样写：
- fragment 布局决定了最便宜的归约粒度是线程内行片段，再做小范围 lane 合并。
- max 必须立刻跨 lane 完整化，因为指数缩放要用完整行最大值。
- sum 可以在循环中先保留局部累计，等最终需要完整分母时再统一 allreduce。

不变量与失败模式：
- 输入 tensor rank 必须是 2，summary rank 必须是 1。
- summary 的行数必须等于 tensor 的行数。
- 使用 `reduce_sum` 后不能立即假设 `row_sum` 已经跨 lane 完整，除非后续显式 allreduce。

Comment：
这段是 online softmax 的底层形状约束：状态是按行的，但行分散在 MMA fragment 与 lane 上。

### 2.2 `exp2` 把 scale、减 max 与硬件路径合并

问题与约束：
- softmax 需要先减行最大值保证数值稳定。
- attention scale 也要并入指数计算。
- GPU 上直接 `expf(score - max)` 不是源码选择的最佳计算形态。

设计选择：
- 使用 `exp2f(score * scale - max_scaled)`，其中 scale 通常是 `softmax_scale * log2(e)`。
- 对全 `-inf` 行把 `max_scaled` 置为 0，避免 `-inf - -inf` 产生 NaN。
- 默认允许编译器生成 fused multiply-add，必要时用 `UNFUSE_FMA` 禁用。

Explain：
`scale_apply_exp2` 将 raw score tile 原地变成以当前 `row_max` 为基准的未归一化概率。数学上它仍是 `exp((score - max) * softmax_scale)`，只是换成以 2 为底的指数和更适合 GPU 的乘加形式。

来源：csrc/flash_attn/src/softmax.h L65-L92

Code：

```cpp
template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(
    Tensor<Engine0, Layout0> &tensor,
    Tensor<Engine1, Layout1> const &max,
    const float scale
) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            #ifdef UNFUSE_FMA
                tensor(mi, ni) = exp2f(__fmul_rn(tensor(mi, ni), scale) - max_scaled);
            #else
                tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
            #endif
        }
    }
}
```

代码逻辑：
- 校验输入 score tile 与 max 向量的 rank 和行数。
- 每行计算缩放后的 `max_scaled`。
- 对每个 score 元素执行 `score * scale - max_scaled`，再取 `exp2f`。
- `UNFUSE_FMA` 分支控制是否拆开乘法和减法。

为什么这样写：
- 减 max 维持 softmax 稳定性。
- `exp2f` 与 FMA 形态更贴近 GPU 指令路径。
- 全 mask 行如果不处理 `-inf`，会在指数前产生 NaN，并污染后续 `row_sum/acc_o`。

不变量与失败模式：
- 传入的 `scale` 必须和调用点约定一致；若已经乘了 `log2(e)`，不能再按自然指数解释。
- `Scale_max=false` 时 max 的缩放规则不同，调用方要保证对应数学语义。
- 输入 tensor 会被原地改写，后续代码读到的是概率而不是 score。

Comment：
这不是近似 softmax，而是把同一个稳定 softmax 改写成更适合 kernel 的指数路径。

## 3. Block 级状态更新

### 3.1 第一块 K/V 初始化 `row_max` 与 `row_sum`

问题与约束：
- 第一块没有历史 `row_max/row_sum/acc_o` 可以合并。
- 初始化路径仍要完成 mask 后 score 到概率的转换。

设计选择：
- `Is_first=true` 时直接从当前 `scores` 归约 `row_max`，随后指数化并初始化 `row_sum`。

Explain：
首块路径建立 online softmax 的初始基准。当前 tile 的行最大值成为 `row_max`，当前 tile 的未归一化概率和成为 `row_sum`，`acc_s` 被原地转换为当前 block 的概率 tile。

来源：csrc/flash_attn/src/softmax.h L136-L145

Code：

```cpp
template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1>
__forceinline__ __device__ void softmax_rescale_o(
    Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2
) {
    Tensor scores = make_tensor(acc_s.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_s.layout()));
    static_assert(decltype(size<0>(scores))::value == kNRows);
    if (Is_first) {
        FLASH_NAMESPACE::template reduce_max</*zero_init=*/true>(scores, row_max);
        FLASH_NAMESPACE::scale_apply_exp2(scores, row_max, softmax_scale_log2);
        FLASH_NAMESPACE::reduce_sum</*zero_init=*/true>(scores, row_sum);
    } else {
```

代码逻辑：
- `acc_s` 转为 row/col 视图。
- `reduce_max` 初始化每行最大值。
- `scale_apply_exp2` 使用当前最大值把 scores 改成未归一化概率。
- `reduce_sum` 初始化每行分母。

为什么这样写：
- 空历史不需要 rescale，直接初始化更简单也更少指令。
- 初始化之后，后续 block 可以统一使用增量合并公式。
- `row_max` 先于指数化生成，避免大 score 直接指数溢出。

不变量与失败模式：
- `acc_s` 必须已经包含当前 K block 的 mask 后 score。
- 首块路径不触碰 `acc_o` 的历史缩放，因为此时历史为空。
- `row_sum` 此处只完成当前实现约定下的局部累积，最终完整分母在 epilogue 统一处理。

Comment：
第一块的职责是建立基准；复杂性被留给后续块的重标尺过程。

### 3.2 后续 K/V block 同时重标尺分母与输出累积

问题与约束：
- 新 block 可能让 `row_max` 变大，旧概率分母必须按新最大值重缩放。
- `acc_o` 是历史 `P @ V` 的未归一化分子累积，必须和 `row_sum` 使用同一缩放因子。

设计选择：
- 保存旧 `row_max`，把当前 scores 的最大值合并进 `row_max`。
- 计算 `scores_scale = exp2((old_max - new_max) * scale)`，同时乘到 `row_sum` 和 `acc_o`。
- 再按新 `row_max` 指数化当前 scores，并把当前 block 的 sum 累加进 `row_sum`。

Explain：
online softmax 的精确性来自同一个重标尺因子同时作用在分母与输出分子上。只修正 `row_sum` 不够，历史 `acc_o` 也必须从旧最大值基准迁移到新最大值基准。

来源：csrc/flash_attn/src/softmax.h L146-L166

Code：

```cpp
Tensor scores_max_prev = make_fragment_like(row_max);
cute::copy(row_max, scores_max_prev);
FLASH_NAMESPACE::template reduce_max</*zero_init=*/false>(scores, row_max);
Tensor acc_o_rowcol = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
#pragma unroll
for (int mi = 0; mi < size(row_max); ++mi) {
    float scores_max_cur = !Check_inf
        ? row_max(mi)
        : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
    float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
    row_sum(mi) *= scores_scale;
    #pragma unroll
    for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
        acc_o_rowcol(mi, ni) *= scores_scale;
    }
}
FLASH_NAMESPACE::scale_apply_exp2(scores, row_max, softmax_scale_log2);
FLASH_NAMESPACE::reduce_sum</*zero_init=*/false>(scores, row_sum);
```

代码逻辑：
- 复制旧最大值到 `scores_max_prev`。
- 用 `zero_init=false` 将当前 score 最大值合并到 `row_max`。
- 把 `acc_o` 转成 row/col 视图，逐行缩放。
- `row_sum` 与 `acc_o` 同乘 `scores_scale`。
- 当前 scores 按新 `row_max` 指数化，并累加到 `row_sum`。

为什么这样写：
- softmax 分母和 `P @ V` 分子必须在同一指数基准下累加。
- 先缩放历史，再加入当前 block，等价于对已扫描的所有 K/V 重新使用新 `row_max`。
- `Check_inf` 保护 causal/local mask 产生的全无效行，避免特殊行影响缩放因子。

不变量与失败模式：
- `scores_max_prev` 必须在更新 `row_max` 前复制。
- `scores_scale` 必须同时应用到 `row_sum` 和 `acc_o`。
- 如果 `acc_o` 的 row/col 视图行数与 `kNRows` 不一致，源码会在编译期断言。

Comment：
这一小段是 FlashAttention 能流式扫描 K/V 但仍保持精确 softmax 的核心公式。

### 3.3 主循环把概率 tile 立即消费为 `P @ V`

问题与约束：
- 概率 tile `P` 很大，不能写出完整矩阵再乘 V。
- dropout 和 `Return_softmax` 都需要在概率 tile 生命周期内处理。

设计选择：
- mask 后立即调用 `softmax_rescale_o`，把 `acc_s` 变成概率 tile。
- 将 `acc_s` 转为 `rP`，可选保存测试用 softmax 或应用 dropout，然后立刻用 `gemm_rs` 累加到 `acc_o`。

Explain：
forward 主循环的局部数据流是 `QK^T -> mask -> online softmax -> P @ V`。概率 tile 只短暂存在于寄存器中，生成后马上参与与 V 的矩阵乘。

来源：csrc/flash_attn/src/flash_fwd_kernel.h L341-L367

Code：

```cpp
masking_step == 0
    ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(
        acc_s, acc_o, params.scale_softmax_log2
    )
    : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(
        acc_s, acc_o, params.scale_softmax_log2
    );

Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
int block_row_idx = m_block * (kBlockM / 16) + tidx / 32;
int block_col_idx = n_block * (kBlockN / 32);
if (Return_softmax) {
    Tensor rP_drop = make_fragment_like(rP);
    cute::copy(rP, rP_drop);
    dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
        rP_drop, block_row_idx, block_col_idx, kNWarps
    );
    cute::copy(rP_drop, tSgS);
    tSgS.data() = tSgS.data() + (-kBlockN);
}
if (Is_dropout) {
    dropout.apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
}

Tensor tOrP = make_tensor(
    rP.data(),
    FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout())
);
FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
```

代码逻辑：
- online softmax 将 `acc_s` 原地转成概率。
- `rP` 把 fp32 概率转成 V GEMM 使用的 element 类型。
- `Return_softmax` 分支复制概率并写出调试/测试需要的 softmax。
- dropout 分支在 `rP` 上应用随机 mask。
- `gemm_rs` 使用概率 tile 与 V tile 更新 `acc_o`。

为什么这样写：
- `P` 生成即消费，避免 `O(N^2)` 写回。
- dropout 必须作用在概率上，放在 `gemm_rs` 前最自然。
- `Return_softmax` 是旁路功能，复制一份概率而不改变主路径的数据流。

不变量与失败模式：
- `acc_s` 在转换为 `rP` 前必须已经完成 softmax。
- dropout 的 block index 必须和概率 tile 的全局位置一致。
- `tOrP` 的布局转换必须匹配 MMA 对 A 寄存器的要求，否则后续 GEMM 会读错 fragment。

Comment：
这段展示了 FlashAttention 的关键工程取舍：不保存 `P`，只在寄存器里短暂拥有它。

## 4. Epilogue 与 Autograd

### 4.1 `normalize_softmax_lse` 补齐分母并写出稳定 LSE

问题与约束：
- 主循环中 `row_sum` 还需要跨 lane 合并。
- 输出 `acc_o` 需要除以最终分母，dropout 路径还要补偿保留概率。
- 全 mask 或异常行不能把 NaN 继续传播成不可控输出。

设计选择：
- 先对 `row_sum` 做 `quad_allreduce_`。
- 每行计算 `inv_sum` 与 `lse`，再把 `acc_o` row/col 视图逐元素乘缩放因子。
- 对 `sum == 0` 或 NaN 的行使用保护值。

Explain：
epilogue 是 online softmax 状态的收口点。它把局部累积的 `row_sum` 合并成完整分母，用这个分母归一化 `acc_o`，并把 `row_max * scale + log(row_sum)` 作为 LSE 返回给后续写回和 backward。

来源：csrc/flash_attn/src/softmax.h L169-L185

Code：

```cpp
template<bool Is_dropout=false, bool Split=false, typename Tensor0>
__forceinline__ __device__ TensorT normalize_softmax_lse(
    Tensor0 &acc_o, float softmax_scale, float rp_dropout=1.0
) {
    SumOp<float> sum_op;
    quad_allreduce_(row_sum, row_sum, sum_op);
    TensorT lse = make_fragment_like(row_sum);
    Tensor acc_o_rowcol = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
    #pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
        float sum = row_sum(mi);
        float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
        lse(mi) = (sum == 0.f || sum != sum)
            ? (Split ? -INFINITY : INFINITY)
            : row_max(mi) * softmax_scale + __logf(sum);
        float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
        #pragma unroll
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
            acc_o_rowcol(mi, ni) *= scale;
        }
    }
    return lse;
};
```

代码逻辑：
- `quad_allreduce_` 合并 lane 间 `row_sum`。
- 为每行构造 `inv_sum`。
- 正常行用 `row_max * softmax_scale + log(sum)` 计算 LSE。
- `acc_o` 每个输出元素乘以归一化因子。
- dropout 路径把 `rp_dropout` 融入输出缩放。

为什么这样写：
- 分母完整后才归一化，避免每个 K block 重复除法和精度损失。
- LSE 是 backward 重算 softmax 的稳定输入。
- 异常行保护让 mask 边界和 split 场景不会直接产生未定义行为。

不变量与失败模式：
- `row_sum` 在函数入口前包含所有 K block 的局部累积。
- `softmax_scale` 是自然对数基下用于 LSE 的 scale，不是前面 `exp2` 路径的 `scale_softmax_log2`。
- 对 split 场景，异常 LSE 使用 `-INFINITY`，非 split 使用 `INFINITY`，调用方要按约定解释。

Comment：
LSE 不是附加统计量，而是“不保存完整 P”后 backward 还能重算概率的关键状态。

### 4.2 Python autograd 保存 LSE 而不是保存概率矩阵

问题与约束：
- 训练 backward 需要 softmax 概率参与梯度计算。
- 保存完整 `P` 会让激活显存回到 `O(N^2)`。
- dropout backward 还需要复现随机状态。

设计选择：
- forward 返回并保存 `softmax_lse`、`out_padded`、`q/k/v` 与 `rng_state`。
- backward 调 `_wrapped_flash_attn_backward`，传入 LSE 和随机状态，由 CUDA backward 重算局部概率与梯度。

Explain：
Python autograd 层把 kernel 里的 LSE 设计显式暴露出来：forward 不保存完整概率矩阵，只保存足以重算 softmax 的紧凑状态。Backward 用输入、输出、LSE 和 rng state 重新构造需要的局部概率。

来源：flash_attn/flash_attn_interface.py L485-L539

Code：

```python
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

q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
_wrapped_flash_attn_backward(
    dout_padded,
    q,
    k,
    v,
    out,
    softmax_lse,
    dqkv[:, :, 0],
    dqkv[:, :, 1],
    dqkv[:, :, 2],
    ctx.dropout_p,
    ctx.softmax_scale,
    ctx.causal,
    ctx.window_size[0],
    ctx.window_size[1],
    ctx.softcap,
    ctx.alibi_slopes,
    ctx.deterministic,
    rng_state=rng_state,
)
```

代码逻辑：
- forward wrapper 返回输出、LSE、可选 softmax mask 和 rng state。
- 梯度开启时保存 `q/k/v/out_padded/softmax_lse/rng_state`。
- backward 从 context 取回这些张量。
- CUDA backward 接收 LSE、dropout 参数、mask/window/softcap/alibi 配置和 rng state。

为什么这样写：
- LSE 加输入输出足以让 backward 分块重算 softmax，而不用保存 `P`。
- `rng_state` 保证 dropout backward 能复现 forward 的随机 mask。
- Python 层只保存 autograd 所需状态，具体重算逻辑仍留在 CUDA kernel。

不变量与失败模式：
- forward 与 backward 使用的 `softmax_scale`、mask、window、softcap、alibi 配置必须一致。
- dropout 场景必须保存并传回 `rng_state`。
- head dim padding 后，backward 末尾要裁回原始维度，否则梯度形状会不匹配。

Comment：
这段把 CUDA 内核的 online softmax 状态连接到 PyTorch 训练语义：保存 LSE，反向重算概率。
