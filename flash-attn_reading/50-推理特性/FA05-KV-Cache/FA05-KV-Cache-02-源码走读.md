---
type: batch-doc
module: FA05-KV-Cache
batch: "FA05"
doc_type: walkthrough
title: "KV Cache 与推理特性 · 源码走读"
tags:
  - flash-attn/batch/fa05
  - flash-attn/module/kv-cache
  - flash-attn/doc/walkthrough
updated: 2026-07-05
---

# KV Cache 与推理特性 · 源码走读

## 1. Python API 把 decode cache 语义集中到 extension 调用

### 1.1 `flash_attn_with_kvcache` 调用 `fwd_kvcache`

问题与约束：
- Decode serving 的一次 attention 往往要同时读取历史 KV cache，并可选 append 新 K/V。
- Python 层需要整理 cache seqlens、paged block table、RoPE、ALiBi、window 和 SplitKV 参数。
- 用户只需要输出，LSE 只在显式要求时返回。

设计选择：
- 检查 K/V cache 最后一维连续。
- 将 q/k/v、cache batch index、block table 等转成 contiguous。
- `cache_seqlens` 为 int 时扩展成 int32 tensor。
- 调 `flash_attn_gpu.fwd_kvcache(...)`，用 `None` 作为可选输出张量占位。
- 根据 `return_softmax_lse` 决定返回 `out` 还是 `(out, softmax_lse)`。

Explain：
Python 入口把 KV cache decode 的参数统一塞进一个 extension 调用。这样 append 新 KV、读旧 cache、RoPE、paged block table 和 SplitKV 都能由底层 C++/CUDA 一次处理。

来源：flash_attn/flash_attn_interface.py L1593-L1627

Code：

```python
assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
if softmax_scale is None:
    softmax_scale = q.shape[-1] ** (-0.5)
if cache_seqlens is not None and isinstance(cache_seqlens, int):
    cache_seqlens = torch.full(
        (q.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
    )
    cache_seqlens = maybe_contiguous(cache_seqlens)
cache_batch_idx = maybe_contiguous(cache_batch_idx)
block_table = maybe_contiguous(block_table)
out, softmax_lse = flash_attn_gpu.fwd_kvcache(
    q,
    k_cache,
    v_cache,
    k,
    v,
    cache_seqlens,
    rotary_cos,
    rotary_sin,
    cache_batch_idx,
    cache_leftpad,
    block_table,
    alibi_slopes,
    None,
    softmax_scale,
    causal,
    window_size[0],
    window_size[1],
    softcap,
    rotary_interleaved,
    num_splits,
)
return (out, softmax_lse) if return_softmax_lse else out
```

代码逻辑：
- 校验 K/V cache 的最后一维连续。
- 规范化 q/k/v contiguous。
- 计算默认 softmax scale。
- 将整数 cache seqlens 扩展成 batch 维 int32 tensor。
- 规范化 cache batch index 和 block table。
- 调用 C++ extension 的 KV cache forward。
- 按调用者要求返回 LSE。

为什么这样写：
- Python 只做轻量形状/布局整理，减少 decode hot path 中的 Python 往返。
- 单 extension 调用可以把 append 和 attention 合并到底层 kernel 路径。
- `return_softmax_lse` 保持默认返回简单，同时保留调试/上层需求。

不变量与失败模式：
- `k_cache/v_cache` 最后一维必须 contiguous。
- `cache_seqlens` tensor 需要位于 cache device 且 dtype 为 int32。
- `block_table` 与 `cache_batch_idx` 的互斥规则在 C++ 层继续校验。
- `window_size` 必须能拆成 left/right 两个整数。

Comment：
Python API 是参数归一化层，真正的 cache addressing 和 kernel 选择在 C++ 入口。

## 2. C++ 入口先确认 cache addressing 模式

### 2.1 `mha_fwd_kvcache` 用 `block_table` 识别 paged KV

问题与约束：
- KV cache 既可以是 dense cache，也可以是 paged KV cache。
- paged KV 的 block table 与 `cache_batch_idx` 都是寻址机制，不能同时使用。
- 错误 dtype、device 或 stride 会导致 kernel 读取错误物理块。

设计选择：
- 入口创建 `CUDAGuard`，确保 kernel 在 q 所在 device 上启动。
- 要求 Ampere 或更新 GPU。
- 要求 q/kcache/vcache dtype 一致且为 fp16 或 bf16。
- 要求 q/kcache/vcache 最后一维连续。
- `block_table_.has_value()` 决定 `paged_KV`。
- paged KV 下禁止 `cache_batch_idx`，并校验 block table device、int32 dtype 和最后一维连续。

Explain：
C++ 入口把 cache addressing 模式定下来。dense cache、cache batch remap 和 paged KV 是不同寻址语义，其中 paged KV 必须通过 int32 block table 映射物理块。

来源：csrc/flash_attn/flash_api.cpp L1206-L1254

Code：

```cpp
mha_fwd_kvcache(at::Tensor &q,
                const at::Tensor &kcache,
                const at::Tensor &vcache,
                std::optional<const at::Tensor> &k_,
                std::optional<const at::Tensor> &v_,
                std::optional<const at::Tensor> &seqlens_k_,
                std::optional<const at::Tensor> &rotary_cos_,
                std::optional<const at::Tensor> &rotary_sin_,
                std::optional<const at::Tensor> &cache_batch_idx_,
                std::optional<const at::Tensor> &leftpad_k_,
                std::optional<at::Tensor> &block_table_,
                std::optional<at::Tensor> &alibi_slopes_,
                std::optional<at::Tensor> &out_,
                const float softmax_scale,
                bool is_causal,
                int window_size_left,
                int window_size_right,
                const float softcap,
                bool is_rotary_interleaved,
                int num_splits) {
    at::cuda::CUDAGuard device_guard{q.device()};
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    TORCH_CHECK(is_sm8x_min, "FlashAttention only supports Ampere GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(kcache.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(vcache.dtype() == q_dtype, "query and value must have the same dtype");

    at::Tensor block_table;
    const bool paged_KV = block_table_.has_value();
    if (paged_KV) {
        TORCH_CHECK(!cache_batch_idx_.has_value(), "Paged KVcache does not support cache_batch_idx");
        block_table = block_table_.value();
        CHECK_DEVICE(block_table);
        TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
        TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
    }
```

代码逻辑：
- 接收 q、cache、新 K/V、cache seqlens、RoPE、paged table、输出和参数。
- 用 q 的 device 设置 CUDA guard。
- 检查 GPU 架构。
- 检查 dtype 和 device。
- 检查输入最后一维连续。
- 通过 block table 是否存在判断 paged KV。
- paged KV 时校验互斥、device、dtype 和 stride。

为什么这样写：
- cache addressing 错误会变成 silent correctness bug，必须在 kernel 前拦截。
- paged KV 和 cache batch remap 同时启用会让物理地址解释冲突。
- dtype/stride 检查靠近 C++ 入口，能保护所有从 Python API 进入的调用。

不变量与失败模式：
- q、kcache、vcache 必须在同一 CUDA device 语义下使用。
- KV cache dtype 必须和 query 一致。
- paged KV 的 block table 必须是 int32，且最后一维 contiguous。
- 传入 paged KV 同时传 cache batch index 会直接失败。

Comment：
Paged KV 是强约束路径；它先确定寻址语义，再进入后续 shape 和 kernel 选择。

## 3. Decode 单步有专门的 GQA 布局优化

### 3.1 `seqlen_q == 1` 时把 GQA group 转到 sequence 维

问题与约束：
- Decode 常见 query length 为 1，单 token query 并行度低。
- GQA 场景中 Q head 数大于 KV head 数。
- local window 或 ALiBi 会改变 attention 语义，不能随意 reshape。
- head dim 需要满足 8 对齐条件。

设计选择：
- `seqlen_q == 1` 且无 ALiBi 时关闭 causal，因为单 query 下等价。
- causal 时将 `window_size_right` 设为 0。
- 满足单 token、Q heads 多于 KV heads、无 local window、head dim 8 对齐、无 ALiBi 时启用 `seqlenq_ngroups_swapped`。
- 将 q reshape/transpose，让 GQA group 变成 sequence 维。
- 更新 `seqlen_q` 为 group 数，`num_heads` 为 KV head 数。

Explain：
这段是 decode hot path 的利用率优化。它把原本藏在 head 维的 GQA group 展开到 query sequence 维，给 kernel 更多 row 级并行工作。

来源：csrc/flash_attn/flash_api.cpp L1275-L1288

Code：

```cpp
if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }
if (is_causal) { window_size_right = 0; }

const int seqlenq_ngroups_swapped =
    seqlen_q == 1 && num_heads > num_heads_k &&
    window_size_left < 0 && window_size_right < 0 &&
    head_size_og % 8 == 0 && !alibi_slopes_.has_value();
if (seqlenq_ngroups_swapped) {
    const int ngroups = num_heads / num_heads_k;
    q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og}).transpose(1, 2);
    seqlen_q = ngroups;
    num_heads = num_heads_k;
}
```

代码逻辑：
- 单 token 且无 ALiBi 时关闭 causal。
- causal 为真时限制右窗口。
- 计算是否能做 GQA group swap。
- 满足条件时求 group 数。
- 将 q reshape 为 `[batch, kv_heads, groups, dim]` 后转置。
- 更新 query length 和 head 数。

为什么这样写：
- 单 token decode 对 GPU 利用率不友好。
- GQA group 本来就是可独立处理的维度，转到 sequence 维能增加并行 row 数。
- 条件限制确保语义不被 local window、ALiBi 或非对齐 head dim 破坏。

不变量与失败模式：
- `num_heads` 必须能被 `num_heads_k` 整除。
- 只有无 local window 和无 ALiBi 时才能做该变换。
- head dim 需要 8 对齐。
- reshape 后的 q shape 必须通过后续 `CHECK_SHAPE`。

Comment：
Decode attention 不是 full attention 的小输入版本，它有专门针对单步 GQA 的布局变换。

## 4. 新 K/V append 通过 params 传入 kernel

### 4.1 新 K/V 的 data pointer 与 stride 被写入 `params`

问题与约束：
- KV cache decode 可能在同一次调用中追加新 token 的 K/V。
- 如果传入 key，就必须同时传入 value 和 cache seqlens。
- 新 K/V shape、dtype、device 和 stride 必须和 cache attention 语义一致。
- head dim 不是 8 的倍数时需要 padding 后再给 kernel。

设计选择：
- `k_.has_value()` 作为 append 路径开关。
- 强制要求 `v_` 和 `seqlens_k_` 同时存在。
- 校验新 K/V dtype、device、最后一维 contiguous 和 shape。
- 必要时 pad K/V 的最后一维到 8 对齐。
- 将 `seqlen_knew`、data pointer 和 batch/row/head stride 写入 `params`。

Explain：
这段把“本次新增 K/V”转换成 CUDA kernel 可用的指针和 stride。后续 splitKV kernel 可以在一次 launch 中完成 append 与 attention。

来源：csrc/flash_attn/flash_api.cpp L1355-L1385

Code：

```cpp
at::Tensor k, v, k_padded, v_padded;
if (k_.has_value()) {
    TORCH_CHECK(v_.has_value(), "If key is supplied, value must also be passed in");
    TORCH_CHECK(seqlens_k_.has_value(), "If key is supplied, seqlens_k must also be passed in");
    TORCH_CHECK(seqlen_q <= seqlen_k, "If key is supplied, it must have seqlen <= the seqlen of the KV cache");
    k = k_.value();
    v = v_.value();
    TORCH_CHECK(k.dtype() == q_dtype, "Key must have the same dtype as query");
    TORCH_CHECK(v.dtype() == q_dtype, "Value must have the same dtype as query");
    CHECK_DEVICE(k); CHECK_DEVICE(v);
    TORCH_CHECK(k.stride(-1) == 1, "Key tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Value tensor must have contiguous last dimension");
    int seqlen_knew = k.size(1);
    CHECK_SHAPE(k, batch_size, seqlen_knew, num_heads_k, head_size_og);
    CHECK_SHAPE(v, batch_size, seqlen_knew, num_heads_k, head_size_og);
    if (head_size_og % 8 != 0) {
        k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        k_padded = k;
        v_padded = v;
    }
    params.seqlen_knew = seqlen_knew;
    params.knew_ptr = k_padded.data_ptr();
    params.vnew_ptr = v_padded.data_ptr();
    params.knew_batch_stride = k_padded.stride(0);
    params.vnew_batch_stride = v_padded.stride(0);
    params.knew_row_stride = k_padded.stride(-3);
    params.vnew_row_stride = v_padded.stride(-3);
    params.knew_head_stride = k_padded.stride(-2);
```

代码逻辑：
- 初始化新 K/V 相关 tensor 变量。
- 检查是否传入新 K。
- 要求 V 和 seqlens_k 同时存在。
- 校验新 KV 长度不超过 cache 长度。
- 取出 K/V 并校验 dtype、device、stride 和 shape。
- 根据 head dim 是否 8 对齐决定是否 padding。
- 写入新 KV 长度、指针和 stride。

为什么这样写：
- append 和 attention 共用一个 params 结构，kernel launch 不需要额外 Python/C++ 分支。
- stride 以 element 为单位传入，kernel 可以按布局访问新 KV。
- padding 在 C++ 侧完成，底层 kernel 能保持对齐假设。

不变量与失败模式：
- 传 K 必须传 V 和 seqlens_k。
- 新 K/V dtype 必须和 Q 一致。
- 新 K/V shape 必须是 `[batch, seqlen_knew, kv_heads, head_dim]`。
- 如果 padding 后生命周期不足，kernel 会读到无效内存；因此 C++ 变量需保持到 launch 后。

Comment：
`knew_ptr` 是否为空，是后续 splitKV launch 中是否 append KV 的关键信号。

## 5. KV cache 场景最终强制进入 split kernel

### 5.1 新 KV、cache batch remap 或 paged KV 都触发 `force_split_kernel`

问题与约束：
- KV cache decode 可能涉及长上下文 split、paged block table、cache batch remap 和新 KV append。
- 普通 forward kernel 不一定覆盖这些 cache-specific 语义。
- splitKV 需要 partial LSE/O 累积区和 page metadata。

设计选择：
- 先调用 `set_params_splitkv` 设置 splitKV 参数和累积 buffer。
- paged KV 时把 block table 指针和 batch stride 写入 params。
- 写入 `page_block_size`。
- 调 `set_params_alibi`。
- 调 `run_mha_fwd`，当 `k_.has_value() || cache_batch_idx_.has_value() || paged_KV` 时强制 split kernel。

Explain：
这是 KV cache 路径和普通 forward 路径的最终分流点。只要调用涉及 append、cache remap 或 paged KV，底层就必须走支持这些语义的 split kernel。

来源：csrc/flash_attn/flash_api.cpp L1442-L1460

Code：

```cpp
at::Tensor softmax_lse_accum, out_accum;
std::tie(softmax_lse_accum, out_accum) = set_params_splitkv(
    params, batch_size, num_heads, head_size, seqlen_k, seqlen_q,
    head_size_rounded, /*dropout*/ 0.f, num_splits, get_num_sm(get_current_device()), opts);

if (paged_KV) {
    params.block_table = block_table.data_ptr<int>();
    params.block_table_batch_stride = block_table.stride(0);
}
params.page_block_size = page_block_size;

set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

auto stream = at::cuda::getCurrentCUDAStream().stream();
run_mha_fwd(params, stream, /*force_split_kernel=*/k_.has_value() || cache_batch_idx_.has_value() || paged_KV);
```

代码逻辑：
- 声明用于延长生命周期的 LSE/O 累积 tensor。
- 通过 `set_params_splitkv` 配置 splitKV 参数。
- paged KV 时传入 block table 指针和 batch stride。
- 设置 page block size。
- 设置 ALiBi 参数。
- 读取当前 CUDA stream。
- 根据 append/remap/paged KV 条件强制 split kernel。

为什么这样写：
- splitKV 提供长上下文分裂和 cache-specific addressing 的统一 launch 路径。
- block table 指针必须在 kernel 前写入 params。
- 统一强制 split kernel 能降低 cache 语义散落到普通 forward 路径的风险。

不变量与失败模式：
- `softmax_lse_accum/out_accum` 需要保持到 kernel 使用结束。
- paged KV 时 block table 指针和 stride 必须有效。
- `page_block_size` 必须和 cache/page table 的物理布局一致。
- 如果 force 条件漏掉某种 cache-specific 语义，可能错误走普通 forward kernel。

Comment：
KV cache 与普通 forward 的关键分界不在 Python 函数名，而在 C++ 最终传给 `run_mha_fwd` 的 `force_split_kernel`。
