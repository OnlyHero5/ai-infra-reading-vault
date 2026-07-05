---
type: batch-doc
module: FA04-FA2-Forward
batch: "FA04"
doc_type: walkthrough
title: "FA2 CUDA Forward · 源码走读"
tags:
  - flash-attn/batch/fa04
  - flash-attn/module/fa2-forward
  - flash-attn/doc/walkthrough
updated: 2026-07-05
---

# FA2 CUDA Forward · 源码走读

## 1. C++ 入口先把 kernel traits 的前置条件锁死

### 1.1 `mha_fwd` 校验架构、dtype、stride、head_dim 和 GQA 关系

问题与约束：
- FA2 forward kernel 依赖预编译的 template traits，不支持任意架构、dtype 或 head_dim。
- Q/K/V 的内层 head_dim 必须连续，才能满足向量化 load 假设。
- MQA/GQA 要求 query heads 能整除 KV heads。
- device guard 必须跟随 q，否则 kernel 可能从错误 CUDA device 启动。

设计选择：
- 入口用 `CUDAGuard{q.device()}` 绑定当前 device。
- 要求 compute capability 至少 Ampere。
- 只允许 fp16/bf16，且 Q/K/V dtype 一致。
- 要求 Q/K/V 最后一维 stride 为 1。
- 要求 head_dim 不超过 256 且是 8 的倍数。
- 要求 `num_heads % num_heads_k == 0`。

Explain：
这些检查是 CUDA kernel traits 的前置条件，不只是普通防御代码。后续 launch template 会假设 dtype、head_dim 对齐、GQA 映射和向量化读取都已经合法。

来源：csrc/flash_attn/flash_api.cpp L350-L405

Code：

```cpp
std::vector<at::Tensor>
mha_fwd(at::Tensor &q,
        const at::Tensor &k,
        const at::Tensor &v,
        std::optional<at::Tensor> &out_,
        std::optional<at::Tensor> &alibi_slopes_,
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        std::optional<at::Generator> gen_) {
    at::cuda::CUDAGuard device_guard{q.device()};

    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    TORCH_CHECK(is_sm8x_min, "FlashAttention only supports Ampere GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(head_size % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
```

代码逻辑：
- 进入 dense forward C++ API。
- 设置当前 CUDA device。
- 读取并校验 compute capability。
- 校验 dtype 类型和 Q/K/V dtype 一致性。
- 校验 Q/K/V 位于 CUDA device。
- 校验 head_dim 是连续内层维度。
- 读取 batch、seqlen、head 数和 head_dim。
- 校验 batch、head_dim 上限/对齐和 GQA head 关系。

为什么这样写：
- template kernel 已经针对有限 head_dim/dtype/arch 组合编译。
- 在 C++ 入口 fail fast，比在 CUDA kernel 中出现越界或错误结果更容易定位。
- 对齐和 stride 约束保护向量化 load/store。

不变量与失败模式：
- 输入必须在当前 CUDA 可访问 device 上。
- Q/K/V dtype 必须完全一致。
- head_dim 不是 8 的倍数或超过 256 会直接失败。
- `num_heads` 不能被 `num_heads_k` 整除时，GQA 映射不存在。

Comment：
FA2 forward 的性能依赖 specialization，入口检查就是 specialization 的安全边界。

## 2. 常规 forward 保存 `out` 和 LSE，而不是完整 P

### 2.1 输出张量和 `softmax_lse` 在 C++ 入口创建或校验

问题与约束：
- 常规 forward 返回 attention 输出，backward 需要 LSE 做稳定重算。
- 完整 softmax matrix `P` 是二次方大小，不能成为默认持久输出。
- 用户可选传入外部 `out_`，但 shape、dtype、device 和 stride 必须匹配。
- `return_softmax` 主要用于 dropout/test 路径。

设计选择：
- 如果传入 `out_`，校验后复用。
- 否则用 `torch::empty_like(q)` 创建输出。
- 按 `[batch, heads, seqlen_q]` 创建 fp32 `softmax_lse`。
- 只有 `return_softmax` 且 `p_dropout > 0` 时分配完整 `p`。
- 不返回 softmax 时创建空 tensor。

Explain：
FA2 的内存优势来自不保存完整 attention matrix。C++ 入口固定保存 `out` 和 LSE；完整 P 只在受限路径中分配。

来源：csrc/flash_attn/flash_api.cpp L420-L450

Code：

```cpp
at::Tensor out;
if (out_.has_value()) {
    out = out_.value();
    TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
    CHECK_DEVICE(out);
    TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
    CHECK_SHAPE(out, batch_size, sizes[1], sizes[2], head_size);
    if (seqlenq_ngroups_swapped) {
        out = out.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2);
    }
} else {
    out = torch::empty_like(q);
}

auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
at::Tensor p;
if (return_softmax) {
    TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
    p = torch::empty({ batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded }, opts);
}
else {
    p = torch::empty({ 0 }, opts);
}
```

代码逻辑：
- 声明输出 tensor。
- 外部输出存在时校验 dtype、device、stride 和 shape。
- GQA swap 场景下调整外部输出视图。
- 外部输出不存在时按 q 创建输出。
- 创建 fp32 LSE。
- 根据 `return_softmax` 和 dropout 分配 P 或空 tensor。

为什么这样写：
- LSE 足够支持 backward 重算 softmax 归一化项。
- 默认不分配 P，避免破坏 FlashAttention 的 IO/内存优势。
- 外部 out 支持调用方复用内存，但必须满足 kernel 写入布局。

不变量与失败模式：
- 外部 `out_` 的最后一维必须 contiguous。
- `softmax_lse` 形状必须和 kernel 写入顺序一致。
- `return_softmax=True` 但 dropout 关闭会直接失败。
- 若 GQA swap 发生，外部输出必须能 reshape 成对应布局。

Comment：
这段是“保存 LSE，不保存完整 attention matrix”的直接源码证据。

## 3. `Flash_fwd_params` 把动态张量信息压成 kernel 参数

### 3.1 `set_params_fprop` 后再调用 `run_mha_fwd`

问题与约束：
- PyTorch tensor 的 shape、stride、data pointer 和 mask 参数需要传入 CUDA kernel。
- dense path 没有 varlen 的 cu_seqlens，也没有 `seqused_k`。
- dropout 需要 RNG state 和 philox counter。
- 空 K 序列不能启动正常 attention kernel。

设计选择：
- 创建 `Flash_fwd_params params`。
- 调 `set_params_fprop` 写入 batch、seqlen、heads、head_dim、Q/K/V/O 指针、P 指针、LSE 指针、dropout、scale、window 和 softcap。
- dense path 将 cu_seqlens 和 seqused 指针传 `nullptr`。
- 调 `set_params_splitkv` 建立 splitKV 累积区引用。
- dropout 时准备 philox RNG。
- `seqlen_k > 0` 才取 stream 并运行 kernel；否则输出置零、LSE 置 inf。

Explain：
C++ 入口把动态信息集中装入 `Flash_fwd_params`，CUDA launch 之后 kernel 只读这个结构。dense、varlen、splitKV 可以共用相近的参数组织方式。

来源：csrc/flash_attn/flash_api.cpp L452-L504

Code：

```cpp
Flash_fwd_params params;
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
                 p_dropout,
                 softmax_scale,
                 window_size_left,
                 window_size_right,
                 softcap);

at::Tensor softmax_lse_accum, out_accum;
std::tie(softmax_lse_accum, out_accum) = set_params_splitkv(
    params, batch_size, num_heads, head_size, seqlen_k, seqlen_q,
    head_size_rounded, p_dropout, /*num_splits*/ 0, get_num_sm(get_current_device()), opts);

if (p_dropout > 0.0)  {
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());
    std::lock_guard<std::mutex> lock(gen->mutex_);
    params.philox_args = gen->philox_cuda_state(counter_offset);
}

set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

if (seqlen_k > 0) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_mha_fwd(params, stream);
} else {
    out.zero_();
    softmax_lse.fill_(std::numeric_limits<float>::infinity());
}
```

代码逻辑：
- 创建 forward 参数结构。
- 写入 dense forward 所需的 shape、指针和标量。
- 将 varlen 指针置空。
- 设置 splitKV 辅助参数和累积 tensor。
- dropout 时从 CUDA generator 取得 philox 状态。
- 设置 ALiBi 参数。
- K 长度非空时启动 kernel。
- K 长度为空时直接填充输出和 LSE。

为什么这样写：
- 参数结构减少 launch 侧参数散落，kernel 内部访问统一。
- dense path 用 nullptr 标识“非 varlen”，避免维护单独结构。
- 空 K 直接处理，避免 kernel 处理无意义网格或非法访问。

不变量与失败模式：
- `params` 中所有 data pointer 对应的 tensor 生命周期必须覆盖 kernel 使用。
- dropout 使用 generator 时必须持锁获取 philox 状态。
- cu_seqlens 为 nullptr 的路径必须被 kernel 当作 dense 处理。
- `seqlen_k == 0` 时不能进入正常 attention kernel。

Comment：
读 CUDA 内核前，先理解 `Flash_fwd_params` 是所有动态信息的汇总容器。

## 4. Launch template 把运行时布尔转成模板常量

### 4.1 `BOOL_SWITCH` 系列选择 specialized kernel

问题与约束：
- Attention 主循环极热，运行时分支会影响展开、寄存器分配和指令路径。
- 不同组合包含 causal/local/ALiBi/softcap/dropout/return_softmax/even shape 等路径。
- 同时为所有组合手写函数不可维护。

设计选择：
- 先计算 grid、`is_even_MN`、`is_even_K` 和 `return_softmax`。
- 用 `BOOL_SWITCH/EVENK_SWITCH/LOCAL_SWITCH/ALIBI_SWITCH/SOFTCAP_SWITCH` 把运行时条件转成模板常量。
- 用这些模板常量实例化 `flash_fwd_kernel`。
- 对 even shape、even K、return softmax、softcap 等组合在模板参数里进一步收敛。

Explain：
这些 switch 宏不是为了可读性，而是为了 specialization。运行时条件在 launch 前被展开成编译期模板参数，kernel 内部就能走更短、更可优化的路径。

来源：csrc/flash_attn/src/flash_fwd_launch_template.h L63-L84

Code：

```cpp
const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
dim3 grid(num_m_block, params.b, params.h);
const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
const bool is_even_K = params.d == Kernel_traits::kHeadDim;
const bool return_softmax = params.p_ptr != nullptr;
BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
            BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                    SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                        auto kernel = &flash_fwd_kernel<Kernel_traits, Is_dropout && !Is_softcap, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && !Has_alibi && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst && !ReturnSoftmaxConst && !Has_alibi, Is_softcap, ReturnSoftmaxConst && Is_dropout && !Is_softcap>;
```

代码逻辑：
- 根据 sequence length 和 blockM 计算 grid x 维。
- grid y/z 对应 batch 和 head。
- 判断 M/N 和 K 是否对齐模板 tile。
- 判断是否返回 softmax。
- 多层 switch 宏把运行时 bool 转成模板 bool。
- 用模板 bool 选择具体 `flash_fwd_kernel` 实例。

为什么这样写：
- 编译期常量能让编译器消除分支并优化寄存器/共享内存路径。
- even shape fast path 可以跳过边界处理。
- 代价是实例数量增加，编译时间和二进制体积上升。

不变量与失败模式：
- switch 条件必须和 kernel 内部模板语义一致。
- even shape 判断错误会导致越界或漏算。
- 特性组合越多，编译产物越多。
- 某些组合被模板参数限制后，调用侧必须在 C++ 入口提前约束。

Comment：
FA2 的高性能来自大量编译期 specialization，launch template 是 specialization 的开关矩阵。

## 5. head_dim 决定 tile traits

### 5.1 `run_mha_fwd_hdim64` 为 dropout 和非 dropout 选择不同 tile

问题与约束：
- 不同 head_dim 对寄存器、shared memory 和 tile shape 的最优点不同。
- dropout 会增加随机数和 mask 处理，改变资源压力。
- 单一 tile 配置无法兼顾所有 head_dim 与 dropout 组合。

设计选择：
- 为 head_dim=64 定义专门的 `run_mha_fwd_hdim64`。
- 用 `DROPOUT_SWITCH` 把 dropout 条件转成模板常量。
- 无 dropout 使用 `Flash_fwd_kernel_traits<64, 128, 128, 4, ...>`。
- 有 dropout 使用 `Flash_fwd_kernel_traits<64, 128, 64, 4, ...>`。
- 源码注释保留了其他 tile/warp 配置的性能对比。

Explain：
FA2 不只按 dtype 和 causal 分发，也按 head_dim 和 dropout 选择 tile。head_dim=64 的无 dropout 路径用更大的 N block；dropout 路径缩小 N block 来控制资源压力。

来源：csrc/flash_attn/src/flash_fwd_launch_template.h L203-L220

Code：

```cpp
template<typename T, bool Is_causal>
void run_mha_fwd_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            // Using 8 warps is 18% slower for seqlen=2k, 2 warps is 5% slower
            // Using block size (64 x 256) is 27% slower for seqlen=2k
            // Using block size (256 x 64) is 85% slower for seqlen=2k, because of register spilling
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}
```

代码逻辑：
- 定义 head_dim=64 的 launch helper。
- 固定 `Headdim=64`。
- 用 dropout switch 生成 dropout/non-dropout 两条模板路径。
- non-dropout 调用 128x128 tile。
- dropout 调用 128x64 tile。
- 两条路径都传入 dtype 和 causal 模板参数。

为什么这样写：
- head_dim 固化后，kernel traits 可以固定 tile、warp 和 shared memory 布局。
- dropout 增加额外工作，缩小 N tile 有助于控制资源使用。
- 注释中的性能数据说明 tile 选择来自实际 profiling，而不是任意常量。

不变量与失败模式：
- 调用该 helper 的 head_dim 必须确实为 64。
- dropout 条件要与 `params.p_dropout` 语义一致。
- tile 选择影响寄存器溢出和 occupancy，错误配置会显著降速。
- 新架构上性能最优 tile 可能变化，需要重新 profiling。

Comment：
读 generated kernel 前，先看 launch helper 如何把 head_dim 映射到 tile traits。
