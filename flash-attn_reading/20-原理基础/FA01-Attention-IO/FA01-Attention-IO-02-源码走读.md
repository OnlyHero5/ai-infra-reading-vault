---
type: batch-doc
module: FA01-Attention-IO
batch: "FA01"
doc_type: walkthrough
title: "Attention IO · 源码走读"
tags:
  - flash-attn/batch/fa01
  - flash-attn/module/attention-io
  - flash-attn/doc/walkthrough
updated: 2026-07-05
---

# Attention IO · 源码走读

## 1. 参数结构：长期状态只保留 O 与 LSE

### 1.1 `Flash_fwd_params` 暴露 softmax 压缩状态

问题与约束：
- 标准 attention 的 `S = QK^T` 和 `P = softmax(S)` 都是 `seqlen_q x seqlen_k` 级别，不能作为常规 forward 的长期 HBM 状态。
- backward 仍需要每行 softmax 的归一化常数，否则无法稳定重算概率。
- split-KV 等路径可能需要 partial accumulation，但仍不能退回完整 `P` 常驻。

设计选择：
- forward 参数结构保留 `o_ptr`、`softmax_lse_ptr`、`softmax_lseaccum_ptr`、`oaccum_ptr` 和 softmax scale。
- 常规主路径把长期输出压缩为 `O + LSE`；partial accumulation 使用累积 buffer，而不是完整 attention matrix。

Explain：
`Flash_fwd_params` 的字段边界直接体现 IO-aware 设计：kernel 的主要落点是输出 `O` 和每行 `softmax_lse`。`softmax_lseaccum_ptr/oaccum_ptr` 支持 splitKV 等中间累积，但仍是按行或按输出维度的压缩状态。

来源：csrc/flash_attn/src/flash.h L62-L71

Code：

```cpp
// The pointer to the softmax sum.
void * __restrict__ softmax_lse_ptr;
void * __restrict__ softmax_lseaccum_ptr;
void * __restrict__ oaccum_ptr;

float scale_softmax;
float scale_softmax_log2;
```

代码逻辑：
- `softmax_lse_ptr` 指向 forward 要写出的每行 log-sum-exp。
- `softmax_lseaccum_ptr` 和 `oaccum_ptr` 为需要跨 split 合并的路径保留累积空间。
- `scale_softmax` 与 `scale_softmax_log2` 分别服务 LSE 计算和 `exp2` softmax 路径。

为什么这样写：
- 不保存完整 `P` 才能避免 HBM 占用回到 `O(N^2)`。
- 保存 LSE 足以让 backward 分块重算概率。
- 同时保存自然指数 scale 和 log2 scale，避免 kernel 内反复转换。

不变量与失败模式：
- `softmax_lse_ptr` 的布局必须和 forward/backward 约定一致，varlen 路径还会由 `unpadded_lse` 改变解释。
- `scale_softmax` 与 `scale_softmax_log2` 必须来自同一个 softmax scale，否则 forward 输出和 LSE 不一致。
- splitKV 累积 buffer 只能表达 partial `O/LSE`，不能被误解为完整概率矩阵。

Comment：
FA01 的第一条证据就是参数结构：长期 HBM 状态围绕 `O/LSE`，不是围绕 `S/P`。

### 1.2 参数结构用可选 `p_ptr` 支持测试/返回 softmax

问题与约束：
- 常规 forward 不应保存完整概率矩阵。
- 用户或测试路径可能要求返回 softmax/dropout mask，因此内核接口仍要能表达可选 `P` 输出。
- 同一个参数结构还要承载 shape、stride 和 scale，供 kernel 直接索引。

设计选择：
- `Flash_fwd_params` 同时包含 `o_ptr`、可选 `p_ptr`、`softmax_lse_ptr` 和 shape/stride/scale 字段。
- `p_ptr` 是可选能力入口，常规路径依赖 `O/LSE`。

Explain：
源码没有把 `P` 作为默认输出，而是把它放在可选指针位置。常规路径写 `O` 与 LSE；只有 `Return_softmax` 等特殊场景才会使用 `p_ptr`。

来源：csrc/flash_attn/src/flash.h L48-L71

Code：

```cpp
struct Flash_fwd_params : public Qkv_params {

    // The O matrix (output).
    void * __restrict__ o_ptr;
    void * __restrict__ oaccum_ptr;

    // The stride between rows of O.
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    // The pointer to the P matrix.
    void * __restrict__ p_ptr;

    // The pointer to the softmax sum.
    void * __restrict__ softmax_lse_ptr;
    void * __restrict__ softmax_lseaccum_ptr;

    // The dimensions.
    int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim, total_q;

    // The scaling factors for the kernel.
    float scale_softmax;
    float scale_softmax_log2;
```

代码逻辑：
- `o_ptr` 与 O strides 定义最终输出写回位置。
- `p_ptr` 提供概率矩阵的可选写出地址。
- `softmax_lse_ptr` 和 `softmax_lseaccum_ptr` 保存 softmax 行级状态。
- shape 与 scale 字段把 Python/C++ 调用层的张量语义转成 kernel 参数。

为什么这样写：
- 一套 kernel 参数可以覆盖常规 forward 与调试/测试返回 softmax 的路径。
- 可选 `p_ptr` 不迫使常规路径分配或写出 `P`。
- 把 stride/shape 放进参数结构，kernel 内可以统一处理 padded、varlen、GQA 等布局。

不变量与失败模式：
- `Return_softmax` 使用 `p_ptr` 时，调用层必须保证指针非空且布局匹配 `seqlen_q_rounded/seqlen_k_rounded`。
- 常规路径不能依赖 `p_ptr` 存在。
- O 与 LSE 的 stride/shape 不一致会导致 backward 读取错误的行级状态。

Comment：
`p_ptr` 的存在说明源码能返回概率，但它的可选性正说明常规 IO 设计不以完整 `P` 为中心。

### 1.3 同一结构承载推理 KV cache 与变体元数据

问题与约束：
- decode 路径除了 attention，还可能要追加新 KV、应用 RoPE、访问 paged KV cache。
- 如果这些操作拆成多个 kernel，会增加 launch 开销和 HBM 往返。
- GQA、splitKV、varlen LSE 都需要额外 metadata。

设计选择：
- `Flash_fwd_params` 继续包含 `knew_ptr/vnew_ptr`、rotary 指针、cache index、paged block table、dropout/local window/softcap、splitKV 和 varlen 标志。

Explain：
FlashAttention 的参数结构不是只服务训练 prefill。它把 cache append、paged KV、RoPE、ALiBi、splitKV 等推理或变体信息放进同一 forward 参数对象，让 kernel family 可以在一次 attention 调用中处理更多 IO 边界。

来源：csrc/flash_attn/src/flash.h L83-L143

Code：

```cpp
// The K_new and V_new matrices.
void * __restrict__ knew_ptr;
void * __restrict__ vnew_ptr;

// The stride between rows of the Q, K and V matrices.
index_t knew_batch_stride;
index_t vnew_batch_stride;
index_t knew_row_stride;
index_t vnew_row_stride;
index_t knew_head_stride;
index_t vnew_head_stride;

// The cos and sin matrices for rotary embedding.
void * __restrict__ rotary_cos_ptr;
void * __restrict__ rotary_sin_ptr;

// The indices to index into the KV cache.
int * __restrict__ cache_batch_idx;

// Paged KV cache
int * __restrict__ block_table;
index_t block_table_batch_stride;
int page_block_size;

int num_splits;  // For split-KV version

bool unpadded_lse;
bool seqlenq_ngroups_swapped;
```

代码逻辑：
- `knew_ptr/vnew_ptr` 和 stride 描述本轮新增 KV。
- rotary 指针为 Q/K 位置编码提供输入。
- `cache_batch_idx`、`block_table` 与 `page_block_size` 描述 KV cache 寻址。
- `num_splits` 支持 splitKV 合并。
- `unpadded_lse` 与 `seqlenq_ngroups_swapped` 调整 varlen/GQA decode 布局解释。

为什么这样写：
- 推理路径的 IO 成本常在 cache 读写和地址转换上，把 metadata 放进同一结构能让 kernel 合并处理。
- paged KV 与 dense KV 共享一套参数边界，减少调用层分叉。
- splitKV 与 varlen LSE 需要和 `O/LSE` 写回策略一致，放在 forward 参数里最直接。

不变量与失败模式：
- cache 指针和 block table 只有在对应路径启用时才有意义，调用层必须保证组合合法。
- `page_block_size`、`block_table_batch_stride` 与实际 cache layout 不一致会读错 KV 页。
- `unpadded_lse` 改变 LSE 布局，backward 或后续 consumer 必须按同一标志读取。

Comment：
这些字段把 “IO-aware attention” 从训练前向扩展到 serving decode：少搬、合并搬、按 cache layout 搬。

## 2. Traits：把 IO 决策固化为编译期形状

### 2.1 tile、warp 与 shared memory layout 都来自 traits

问题与约束：
- head_dim、blockM、blockN、warp 数会共同决定 shared memory 占用、MMA 形状和 copy 方式。
- kernel 主循环不能在运行时频繁分支选择 layout，否则会损失展开和寄存器/共享内存规划。

设计选择：
- `Flash_fwd_kernel_traits` 把模板参数固化为 `kBlockM/kBlockN/kHeadDim/kNWarps`，再派生 `kBlockKSmem/kBlockKGmem/kSwizzle` 和 shared memory layout。

Explain：
traits 是 IO-aware 设计落到 CUDA 编译期的地方。不同 head_dim 与 tile 参数对应不同 shared memory 排布、global copy 形状和 MMA tile；kernel 主体只消费 traits 展开的类型。

来源：csrc/flash_attn/src/kernel_traits.h L51-L72

Code：

```cpp
struct Flash_fwd_kernel_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;

    static constexpr bool Share_Q_K_smem = Share_Q_K_smem_;
    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_ || Share_Q_K_smem;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
```

代码逻辑：
- 从模板参数生成元素类型、accum 类型和索引类型。
- 固化是否共享 Q/K shared memory、Q 是否放寄存器。
- 计算线程数、Q/K tile 大小和 head_dim。
- 根据 head_dim 选择 shared/global memory K 维访问粒度和 swizzle。

为什么这样写：
- 编译期常量允许循环展开、静态 shared memory 分配和类型级 layout 组合。
- `kBlockKSmem` 与 `kBlockKGmem` 分开，给 shared memory bank conflict 与 global memory 向量化留出不同选择。
- `Is_Q_in_regs` 与 `Share_Q_K_smem` 影响 shared memory 压力，是核心 IO tradeoff。

不变量与失败模式：
- `kHeadDim` 必须是 32 的倍数。
- traits 实例的 tile 形状必须和 launch 选择的 kernel specialization 对齐。
- 错误的 `Share_Q_K_smem/Is_Q_in_regs` 组合会改变 shared memory 分配和同步假设。

Comment：
FA 的 IO 优化不是运行时“智能判断”，而是提前编译成一组具体 kernel traits。

### 2.2 traits 继续派生 MMA、shared memory 与 copy layout

问题与约束：
- Q/K/V/O 在 shared memory 中的 layout 要同时满足 Tensor Core MMA、bank conflict 和写回 coalescing。
- 不同 head_dim 下的 layout 不能靠手写索引散落在 kernel 主体里。

设计选择：
- traits 生成 `TiledMma`、`SmemLayoutQ`、`SmemLayoutKV`、`SmemLayoutO`、shared memory size，以及 Q/K 是否共享 smem 的容量规则。

Explain：
这一段把“tile 形状”进一步落成“内存布局”。Q tile、K/V tile 和 O tile 都由 traits 产生 CuTe layout，kernel 后续只通过 tensor partition 和 copy atom 使用这些 layout。

来源：csrc/flash_attn/src/kernel_traits.h L51-L90

Code：

```cpp
static constexpr int kNWarps = kNWarps_;
static constexpr int kNThreads = kNWarps * 32;

static constexpr int kBlockM = kBlockM_;
static constexpr int kBlockN = kBlockN_;
static constexpr int kHeadDim = kHeadDim_;
static_assert(kHeadDim % 32 == 0);
static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

using TiledMma = TiledMMA<
    typename Base::MMA_Atom_Arch,
    Layout<Shape<Int<kNWarps>,_1,_1>>,
    Tile<Int<16 * kNWarps>, _16, _16>>;

using SmemLayoutQ = decltype(tile_to_shape(
    SmemLayoutAtomQ{},
    Shape<Int<kBlockM>, Int<kHeadDim>>{}));

using SmemLayoutKV = decltype(tile_to_shape(
    SmemLayoutAtomQ{},
    Shape<Int<kBlockN>, Int<kHeadDim>>{}));
```

代码逻辑：
- `TiledMma` 把 warp 数映射成 Tensor Core tile 组织。
- `SmemLayoutQ` 生成 Q tile 的 shared memory layout。
- `SmemLayoutKV` 生成 K/V tile 的 shared memory layout。
- 后续 traits 还会根据这些 layout 计算 shared memory size。

为什么这样写：
- Q tile 与 K/V tile 的行数不同，但 head_dim 维度和 swizzle 策略相关。
- 用类型表达 layout，编译器能在 copy 和 MMA 分区中静态推导形状。
- traits 集中管理 layout，避免 kernel 主体在每个访问点重复处理 head_dim 分支。

不变量与失败模式：
- `SmemLayoutQ` 的 shape 必须对应 `kBlockM x kHeadDim`。
- `SmemLayoutKV` 的 shape 必须对应 `kBlockN x kHeadDim`。
- `TiledMma` 与 fragment partition 的假设必须匹配，否则后续 `partition_fragment_A/B/C` 会产生错误布局。

Comment：
这一层解释了为什么 FA 源码里到处是 traits：IO 形状必须和 MMA 形状一起被编译。

### 2.3 Global memory copy 显式按 128 bit 向量化

问题与约束：
- FlashAttention 减少了 `P` 的 HBM 往返，但 Q/K/V/O 仍必须从 HBM 读写。
- 这些必要访问如果不 coalesced/vectorized，仍会成为瓶颈。
- shared memory 写入还要避开 bank conflict。

设计选择：
- traits 使用 `cute::uint128_t` 定义每次 global memory copy 的元素数。
- 用 static assert 约束 head_dim 和线程布局可整除。
- Q/K/V 读优先使用 `SM80_CP_ASYNC_CACHEGLOBAL`，否则退到 128-bit 对齐 copy；O 写回使用对齐向量化 store。

Explain：
IO-aware 不只是“不写 P”，还包括把必须读写的 Q/K/V/O 做成规整搬运。`GmemLayoutAtom` 和 `GmemTiledCopyQKV/O` 定义线程如何用 128-bit transaction 覆盖连续元素。

来源：csrc/flash_attn/src/kernel_traits.h L111-L137

Code：

```cpp
static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
// Using kBlockKSmem here is 6-10% faster than kBlockKGmem for d=128 because of bank conflicts.
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
using GmemTiledCopyO = decltype(
    make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                    GmemLayoutAtom{},
                    Layout<Shape<_1, _8>>{}));
```

代码逻辑：
- `kGmemElemsPerLoad` 将 128 bit 转成当前元素类型下的元素个数。
- `kGmemThreadsPerRow` 由 shared memory K 粒度和每次 load 元素数决定。
- `GmemLayoutAtom` 描述线程在一行内的访问排布。
- `GmemTiledCopyQKV` 选择 cp.async 或自动向量化 copy。
- `GmemTiledCopyO` 定义输出写回 copy。

为什么这样写：
- 必要 HBM 访问必须尽量合并成宽访问，减少 transaction 数。
- 使用 `kBlockKSmem` 而不是 `kBlockKGmem` 是为了规避某些 d=128 场景的 shared memory bank conflict。
- Q/K/V 读可以用 cp.async 与计算流水化，O 写回则重视对齐 store。

不变量与失败模式：
- `kHeadDim` 必须能被 128-bit load 的元素数整除。
- `kNThreads` 必须能整除每行 global memory copy 的线程数。
- 指针实际对齐需要满足 `AutoVectorizingCopyWithAssumedAlignment<128>` 的假设。

Comment：
减少 HBM 次数和优化单次 HBM 访问是两层不同的 IO 优化，这里展示的是第二层。

## 3. Kernel：从 HBM tile 到寄存器状态

### 3.1 kernel 入口把全局张量切成 tile 视图

问题与约束：
- HBM 中的 Q/K/V 是带 batch、head、stride 的大张量，kernel 只处理当前 CTA 的 tile。
- GQA/MQA 下 Q head 与 KV head 数不同，需要在 tile 选择时映射。
- shared memory 需要按 traits 划分 Q/K/V 缓冲区。

设计选择：
- 用 CuTe `make_tensor` 包装 HBM 指针与 shape/stride。
- 用 `local_tile` 取当前 `m_block` 和 head 对应的 Q tile，以及所有 K/V block 视图。
- 用 traits layout 在同一块 shared memory 中构造 `sQ/sK/sV`。

Explain：
这段建立 HBM 到 shared memory 的边界。`mQ/mK/mV` 是全局视图，`gQ/gK/gV` 是当前 CTA 要搬运的 tile 视图，`sQ/sK/sV` 是片上缓冲区视图。

来源：csrc/flash_attn/src/flash_fwd_kernel.h L138-L177

Code：

```cpp
Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                      + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                        make_shape(binfo.actual_seqlen_q, params.h, params.d),
                        make_stride(params.q_row_stride, params.q_head_stride, _1{}));
Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                       make_coord(m_block, 0));
Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                                      + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
                        make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                        make_stride(params.k_row_stride, params.k_head_stride, _1{}));
Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                       make_coord(_, 0));
Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                                      + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
                        make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                        make_stride(params.v_row_stride, params.v_head_stride, _1{}));
Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                       make_coord(_, 0));

Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                        typename Kernel_traits::SmemLayoutQ{});
Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                        typename Kernel_traits::SmemLayoutKV{});
Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
```

代码逻辑：
- `binfo` 计算当前 batch 的实际 Q/K 长度和偏移。
- `gQ` 选中当前 query block。
- `gK/gV` 用 `bidh / params.h_h_k_ratio` 把 Q head 映射到 KV head。
- `sQ/sK/sV` 根据 shared memory 基址和 traits layout 构建。
- `Share_Q_K_smem` 为 true 时，Q 和 K 可复用同一段 smem。

为什么这样写：
- 用 tensor view 显式表达 shape/stride，后续 copy 和 MMA 可以共享同一套布局语义。
- GQA 映射在 tile 入口完成，主循环无需重复判断 head 关系。
- shared memory 划分由 traits 控制，便于不同 specialization 调整 IO tradeoff。

不变量与失败模式：
- `params.*_stride` 与实际张量布局必须一致。
- `h_h_k_ratio` 必须正确表达 Q head 与 KV head 比例。
- `Share_Q_K_smem` 下复用 smem 需要后续同步与生命周期严格匹配。

Comment：
从这一段开始，kernel 不再面对抽象大矩阵，而是面对可搬运、可计算的 tile。

### 3.2 主循环前预取 K 并初始化片上状态

问题与约束：
- K/V tile 从 HBM 到 shared memory 有延迟，需要与计算重叠。
- Q 如果要驻留寄存器，需要在主循环前从 shared memory retile 到 fragment。
- 每个 query tile 的 `acc_o` 与 softmax 状态要在扫描 K blocks 前初始化。

设计选择：
- 从最后一个 K block 反向扫描，先异步复制第一个 K tile 并 `cp_async_fence`。
- 必要时等待 Q copy 并把 Q 放进寄存器 fragment。
- 清零 `acc_o`，构造 `Softmax` 与 mask 对象。

Explain：
主循环前置阶段同时做三件事：启动 K tile 的异步搬运、准备 Q 的寄存器视图、初始化跨 K blocks 累积的 `acc_o/softmax` 状态。这为后续“搬运与计算交叠”建立流水线。

来源：csrc/flash_attn/src/flash_fwd_kernel.h L267-L288

Code：

```cpp
int n_block = n_block_max - 1;
FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(
    gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
    binfo.actual_seqlen_k - n_block * kBlockN
);
cute::cp_async_fence();

if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
    FLASH_NAMESPACE::cp_async_wait<1>();
    __syncthreads();
    Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
    CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
}

clear(acc_o);

FLASH_NAMESPACE::Softmax<2 * size<1>(acc_o)> softmax;

const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr
    ? 0.0f
    : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
FLASH_NAMESPACE::Mask<Is_causal, Is_local, Has_alibi> mask(
    binfo.actual_seqlen_k,
    binfo.actual_seqlen_q,
    params.window_size_left,
    params.window_size_right,
    alibi_slope
);
```

代码逻辑：
- `n_block` 从最后一个 K block 开始。
- 复制当前 K tile 到 shared memory，并用 fence 提交 async copy。
- 如果 Q 要进寄存器，等待相关 copy 后 retile。
- 清空输出 accumulator。
- 用 `acc_o` 的行数派生 softmax 行数，构造 mask。

为什么这样写：
- 反向扫描让最后一个可能不完整的 K block 先进入 masking 处理。
- `cp_async` 提前发起，后续计算可隐藏一部分 HBM 延迟。
- `acc_o` 与 `Softmax` 是跨 K blocks 的寄存器状态，必须在扫描前初始化。

不变量与失败模式：
- async copy 的 wait/fence/syncthreads 顺序必须与 smem 使用匹配，否则会读未完成数据。
- `Softmax<2 * size<1>(acc_o)>` 的行数必须对应 accumulator row layout。
- ALiBi slope 除以 `scale_softmax` 的约定要和 mask 内部加法保持一致。

Comment：
这里能看到 IO-aware 的流水线雏形：先把下一批数据送上路，再让寄存器状态跨 block 累积。

### 3.3 score tile 是短生命周期寄存器数据

问题与约束：
- `QK^T` score tile 只对当前 Q block 与当前 K block 有意义。
- 如果把 `acc_s` 或概率 tile 写回 HBM，就会重新引入 attention matrix 的 IO 瓶颈。
- mask、softmax、dropout 和 `P @ V` 都必须在 tile 生命周期内完成。

设计选择：
- 每轮循环创建 `acc_s` fragment，计算 `QK^T`，原地 mask 和 softmax。
- 将 `acc_s` 转成 `rP` 后立即用于 `gemm_rs(acc_o, P, V)`。
- 同时在计算过程中预取下一块 K。

Explain：
主循环的核心数据流是：寄存器中生成 `acc_s`，原地变成概率 tile，再立刻乘 V 累加进 `acc_o`。跨循环留下的是 `acc_o` 与 softmax 行状态，而不是 `S/P`。

来源：csrc/flash_attn/src/flash_fwd_kernel.h L303-L367

Code：

```cpp
Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
clear(acc_s);
FLASH_NAMESPACE::cp_async_wait<0>();
__syncthreads();

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

if (n_block > n_block_min) {
    FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(
        gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV
    );
    cute::cp_async_fence();
}

masking_step == 0
    ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
    : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
Tensor tOrP = make_tensor(
    rP.data(),
    FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout())
);
FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
```

代码逻辑：
- `partition_fragment_C` 分配当前 score accumulator。
- 等待 K/V copy 后执行 QK GEMM。
- softcap 与 mask 在 `acc_s` 上原地处理。
- 若还有下一块 K，则提前 copy 到 `sK`。
- online softmax 把 `acc_s` 转成概率并更新 softmax 状态。
- `rP` 转换布局后与 V 做 GEMM，累加到 `acc_o`。

为什么这样写：
- `acc_s/rP` 的生命周期被限制在一个 K block 内，避免保存 `S/P`。
- 预取下一块 K 把 HBM 延迟与当前 block 的计算重叠。
- `acc_o` 作为跨 block 累积状态留在寄存器 accumulator 中，直到 epilogue。

不变量与失败模式：
- `cp_async_wait<0>()` 和同步必须发生在读取当前 K/V shared memory 前。
- mask 必须在 softmax 前应用。
- `softmax_rescale_o` 调用后，`acc_s` 已经不是 raw score。
- `tOrP` layout 必须匹配后续 `gemm_rs` 对 A operand 的要求。

Comment：
这是 FlashAttention IO 节省的核心证据：score/probability tile 生成即消费，不落 HBM。

### 3.4 Epilogue 只把 O 与 LSE 写回 HBM

问题与约束：
- `acc_o` 在寄存器里是 fp32 accumulator，不一定适合直接 coalesced 写回。
- backward 需要 LSE，但不需要完整 `P`。
- 边界 tile 可能超出实际 seqlen/head_dim，写回要带 predicate。

设计选择：
- 先调用 `normalize_softmax_lse` 归一化 `acc_o` 并生成 `lse`。
- 将 `acc_o` 转成输出 dtype，经 shared memory `sO` retile，再用 `GmemTiledCopyO` 写回 `gO`。
- 用 `get_lse_tile` 定位 LSE 写回位置，只由对应行线程写 `gLSE`。

Explain：
epilogue 是 IO-aware 数据流的终点。此前所有 K/V blocks 的概率贡献都折叠在 `acc_o` 与 softmax 状态中；最后只把最终输出 O 和压缩 LSE 写回 HBM。

来源：csrc/flash_attn/src/flash_fwd_kernel.h L431-L493

Code：

```cpp
Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(
    acc_o, params.scale_softmax, params.rp_dropout
);

Tensor rO = FLASH_NAMESPACE::convert_type<Element>(acc_o);
Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});
auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                      + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                        make_shape(binfo.actual_seqlen_q, params.h, params.d),
                        make_stride(params.o_row_stride, params.o_head_stride, _1{}));
Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                       make_coord(m_block, 0));
Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(
    params, bidb, bidh, m_block, binfo
);

typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

Tensor tOrO = make_tensor<Element>(shape(tOgO));
cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

if (get<1>(taccOcO_row(0)) == 0) {
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = get<0>(taccOcO_row(mi));
        if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
    }
}

FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
    gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
);
```

代码逻辑：
- `normalize_softmax_lse` 用最终 softmax 状态归一化输出并生成 LSE。
- `rO` 把 fp32 accumulator 转成输出元素类型。
- `sO` 使用输出 shared memory layout 暂存 retile 后的数据。
- `gO` 和 `gLSE` 定位当前 tile 的 HBM 输出位置。
- LSE 按行写回，O 用 tiled copy 带 predicate 写回。

为什么这样写：
- shared memory staging 能把寄存器 accumulator 重排成更规整的 global store。
- 只写 O/LSE 避免保存完整概率矩阵。
- LSE 单独写出给 backward 重算概率，保留训练所需信息。

不变量与失败模式：
- `normalize_softmax_lse` 必须在 O dtype 转换前执行，保持 fp32 累积精度。
- `tOpO` 必须正确标记 head_dim 边界，避免越界写。
- LSE 写回的行索引必须和当前 `m_block` 对应。

Comment：
到 epilogue 为止，FA01 的证据链闭合：HBM 读 Q/K/V tile，片上生成并消费 `S/P`，HBM 只落 `O/LSE`。
