---
type: batch-doc
module: FA04-FA2-Forward
batch: "FA04"
doc_type: faq
title: "FA2 CUDA Forward · 关键问题"
tags:
  - flash-attn/batch/fa04
  - flash-attn/module/fa2-forward
  - flash-attn/doc/faq
updated: 2026-07-05
---

# FA2 CUDA Forward · 关键问题

## 1. 为什么源码里有大量 head_dim 文件？

**Explain：** head_dim 决定 MMA tile、shared memory layout、寄存器压力和 block size。把 head_dim 做成编译期常量，kernel 才能使用静态 layout 和展开后的访存路径。

**Comment：** 不要逐个 generated `.cu` 文件阅读。正确读法是看 `HEADDIM_SWITCH` 和 `run_mha_fwd_hdim*` 如何选择 traits。

## 2. 为什么 dropout 会改变 kernel 选择？

**Explain：** dropout 需要随机数状态、mask 编码和 backward 对齐。源码中 `DROPOUT_SWITCH` 会让是否 dropout 成为模板条件，避免主循环里处处动态判断。

**Code：**

```cpp
// 来源：csrc/flash_attn/src/flash_fwd_launch_template.h L203-L216
DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
    if constexpr(!Is_dropout) {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
    } else {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
    }
});
```

**Comment：** 训练打开 dropout，评估关闭 dropout，两者不是完全相同的 kernel 形态。

## 3. 为什么 `softcap` 和 dropout 有限制？

**Explain：** C++ 入口显式禁止 `softcap > 0` 和 dropout 同时使用。原因不是 API 忘了支持，而是 softcap 会改变 score 变换路径，dropout 又需要概率/随机状态一致，组合支持需要额外 kernel 与测试。

**Code：**

```cpp
// 来源：csrc/flash_attn/flash_api.cpp L395
if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }
```

**Comment：** 读 AI infra 源码时要特别关注这种约束，它会直接影响模型特性在某个 backend 上能否启用。

## 4. 为什么 causal mask 有时会被关闭？

**Explain：** 当 `seqlen_q == 1` 且没有 ALiBi 时，causal 与非 causal 对单个 query 的效果可能等价，源码会把 `is_causal` 置 false，减少不必要的 mask 逻辑。

**Code：**

```cpp
// 来源：csrc/flash_attn/flash_api.cpp L400-L401
if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }
if (is_causal) { window_size_right = 0; }
```

**Comment：** 这类小优化在 decode 场景很常见，因为每步 `seqlen_q` 往往很小。

## 5. SplitKV 为什么要单独路径？

**Explain：** 当 `seqlen_k` 很长或并行度不足时，把 K/V 按 sequence 维切成多个 split 可以提高 SM 占用率，但需要额外 combine kernel 合并 partial output 和 LSE。

**Comment：** SplitKV 是典型的 throughput/extra-memory tradeoff：它增加一些 HBM 读写，换取更好的并行度。后续见 [[FA05-KV-Cache-02-源码走读]]。

## 6. C++ forward 返回值如何体现主路径边界？

**Explain：** C++ API 在参数构造完成后只调用一次 `run_mha_fwd`。如果 `seqlen_k == 0`，它不进入 kernel，而是直接把输出清零、把 LSE 填成无穷；正常路径最终返回 `out`、`softmax_lse`、可选 `p` 和 `rng_state`。

**Code：**

```cpp
// 来源：csrc/flash_attn/flash_api.cpp L497-L511
if (seqlen_k > 0) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_mha_fwd(params, stream);
} else {
    // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
    out.zero_();
    softmax_lse.fill_(std::numeric_limits<float>::infinity());
}

if (seqlenq_ngroups_swapped) {
    out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size});
    q = q.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size});
    softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
}
return {out, softmax_lse, p, rng_state};
```

**Comment：** 这里是从 C++ binding 进入 CUDA launch 的边界。排查 forward 结果异常时，先看是否进入 `run_mha_fwd`，再看返回 tensor 是否因为 GQA reshape 被改形。

