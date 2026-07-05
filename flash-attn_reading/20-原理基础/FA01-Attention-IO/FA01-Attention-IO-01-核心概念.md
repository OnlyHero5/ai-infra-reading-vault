---
type: batch-doc
module: FA01-Attention-IO
batch: "FA01"
doc_type: concept
title: "Attention IO · 核心概念"
tags:
  - flash-attn/batch/fa01
  - flash-attn/module/attention-io
  - flash-attn/doc/concept
updated: 2026-07-05
---

# Attention IO · 核心概念

## 1. Attention 的真正瓶颈

**Explain：** 标准 attention 看起来是两次矩阵乘：`QK^T` 和 `PV`。但对长序列，`S = QK^T` 与 `P = softmax(S)` 都是 `N x N` 中间矩阵。如果这些矩阵写入 HBM，再从 HBM 读回，内存访问会压过纯计算成本。

**Code：**

```text
S = QK^T
P = softmax(S)
O = PV
```

**Comment：**
- `Q`、`K`、`V` 的大小随 `N` 线性增长。
- `S`、`P` 的大小随 `N^2` 增长。
- FlashAttention 的重点是避免将 `S`、`P` 完整 materialize 到 HBM。

## 2. IO-aware 的含义

**Explain：** IO-aware 不是“尽量少算”，而是“尽量少搬”。GPU 上 Tensor Core 很快，HBM 虽然带宽高但仍比 shared memory/register 慢得多。高性能 kernel 往往愿意多做一点计算，换取少读写 HBM。

| 存储层 | 特点 | FlashAttention 使用方式 |
|--------|------|-------------------------|
| HBM | 容量大，访问慢 | 存 Q/K/V、最终 O、LSE |
| Shared memory | 容量小，访问快 | 暂存 Q/K/V tile |
| Register | 最快，最稀缺 | 保存 score、row max、row sum、O accumulator |

## 3. 源码中的 IO 线索

**Explain：** `Flash_fwd_params` 只保存输入输出与必要辅助量指针，不包含完整 attention matrix 指针。可选的 `p_ptr` 主要用于 dropout/debug 相关返回，不是常规 forward 必需输出。

**Code：**

```cpp
// 来源：csrc/flash_attn/src/flash.h L48-L71
struct Flash_fwd_params : public Qkv_params {
    // The O matrix.
    void * __restrict__ o_ptr;
    void * __restrict__ oaccum_ptr;

    // The pointer to the softmax sum.
    void * __restrict__ softmax_lse_ptr;
    void * __restrict__ softmax_lseaccum_ptr;

    float scale_softmax;
    float scale_softmax_log2;
```

**Comment：**
- `o_ptr` 是最终输出。
- `softmax_lse_ptr` 是 backward 所需的 compact 状态。
- 没有常规保存完整 `P` 的长期输出路径。

## 4. AI infra 视角

训练场景关心 activation memory 与吞吐；推理 prefill 关心长 prompt 的吞吐；decode 场景关心 KV cache 读取与小 `seqlen_q` 的利用率。FlashAttention 是这些场景共用的底层优化思想。

## 5. `P` 指针为什么不能误读

**Explain：** 参数结构里确实有 `p_ptr`，但 C++ 入口只在 `return_softmax` 打开时分配 `p`，而且源码注释说明这是为了减少编译时间而受限的可选路径。主路径仍然是 `out + softmax_lse`。

**Code：**

```cpp
// 来源：csrc/flash_attn/flash_api.cpp L441-L447
auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
at::Tensor p;
// Only return softmax if there's dropout to reduce compilation time
if (return_softmax) {
    TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
    p = torch::empty({ batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded }, opts);
}
```

**Comment：** `softmax_lse` 总会分配；`p` 只有可选返回时才分配。读源码时要区分“参数结构支持某路径”和“常规执行总会保存”。

## 6. IO-aware 不等于只看显存峰值

**Explain：** FlashAttention 减少的是大矩阵跨 HBM 的读写次数；显存峰值下降只是结果之一。`set_params_fprop` 把 optional `p` 与 `softmax_lse` 分别传给 kernel，也把 mask/dropout/window/softcap 这些运行时条件转为参数。

**Code：**

```cpp
// 来源：csrc/flash_attn/flash_api.cpp L452-L470
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
                 softcap
                 );
```

**Comment：** `return_softmax ? p.data_ptr() : nullptr` 是判断主路径的关键证据；`softmax_lse.data_ptr()` 则是 forward/backward 合作的固定接口。

## 7. Tile 的三层存储对应

**Explain：** 源码变量可以直接对应三层存储：`mQ/mK/mV` 是 HBM tensor view，`gQ/gK/gV` 是当前 block 的 global-memory tile，`sQ/sK/sV` 是 shared memory staging，`acc_s/acc_o` 才是 register fragment。

| 存储层 | 源码变量 | 生命周期 |
|--------|----------|----------|
| HBM view | `mQ` / `mK` / `mV` / `mO` | 整个 kernel 可寻址 |
| HBM tile | `gQ` / `gK` / `gV` / `gO` | 当前 query block 或 K/V block |
| Shared memory | `sQ` / `sK` / `sV` / `sO` | 当前 CTA 内复用 |
| Register fragment | `acc_s` / `acc_o` / `row_max` / `row_sum` | 当前线程组局部累积 |

**Comment：** 读 `flash_fwd_kernel.h` 时先标出变量所在存储层，再看计算顺序，会比直接读 CuTe layout 更稳。
