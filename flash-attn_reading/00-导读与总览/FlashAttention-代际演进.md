---
type: index-doc
title: "FlashAttention 代际演进"
doc_type: concept
tags:
  - flash-attn/index-layer
  - flash-attn/doc/concept
updated: 2026-07-05
---

# FlashAttention 代际演进

> 先把 FA1 到 FA4 的演进线理清，再进入 CUDA/CuTe 源码细节。

## 1. 先看清当前基线

**Explain：** 本 vault 的 upstream 基线是 `flash-attn 2.8.4`。当前可直接走读的主源码路径是 FA2、FA3 和 FA4；FA1 更适合作为算法原点来理解：它提出 IO-aware exact attention，解释为什么不把完整 `S/P` 矩阵落到 HBM。

**Code：**

```markdown
# 来源：README.md L1-L15
**FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**
**FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**
```

**Comment：** 这说明 FA1 与 FA2 是两篇论文和两代实现目标。当前仓库保留了从 1.x 升级到 2.x 的说明，但日常源码阅读不要在当前树里硬找一条独立的 FA1 kernel 主线。

## 2. 四代演进主线

| 代际 | 阅读定位 | 技术特点 | 工程设计 |
|------|----------|----------|----------|
| FA1 | 算法原点 | IO-aware exact attention、tile、online softmax、backward 重算 | 证明 `S/P` 不必完整写回 HBM；在本 vault 中由 [[FA01-Attention-IO-00-MOC]] 与 [[FA02-Online-Softmax-00-MOC]] 承接 |
| FA2 | 当前稳定主路径 | 更好的并行和 work partitioning，支持训练、varlen、KV cache 等常用路径 | `flash_attn_2_cuda` extension、C++ 参数装配、CUDA template specialization |
| FA3 | Hopper 专门路径 | 面向 H100/H800，覆盖 TMA/GMMA、FP8 forward、Hopper 调度形态 | `hopper/` 下独立安装和测试，仍偏 C++/CUDA extension 思路 |
| FA4 | CuTeDSL/JIT 路径 | 面向 Hopper/Blackwell，用 Python DSL 表达 kernel 并 JIT 编译 | `flash_attn/cute/` 独立 API、kernel object、compile key/cache、运行时能力检查 |

## 3. FA1 到 FA2：从算法证明到主包重写

**Explain：** FA1 解决的是 attention memory wall：标准 attention 会 materialize `S=QK^T` 与 `P=softmax(S)`，而 FlashAttention 用 tile + online softmax 让 `P` 在片上生成即消费。FA2 没有改变 exact attention 的数学定义，而是把实现重写成更适合并行、更多 API 形态和更多硬件组合的主包路径。

**Code：**

```markdown
# 来源：README.md L405-L420
### 2.0: Complete rewrite, 2x faster
Upgrading from FlashAttention (1.x) to FlashAttention-2
- `flash_attn_unpadded_func` -> `flash_attn_varlen_func`
```

**Comment：** `unpadded` 改名为 `varlen` 是接口语义升级：FA2 把变长序列作为一等 API，而不是只把它看成“去 padding 后的输入”。读 [[FA03-Python-API-00-MOC]] 时，应把 packed/varlen/KV cache 看作 FA2 工程化后的外部边界。

## 4. FA2 的工程设计核心

**Explain：** FA2 主包的关键工程取舍是静态专门化。dtype、head_dim、causal、dropout、local、ALiBi、softcap、SplitKV 等组合会被推到 C++/CUDA dispatch 与 template 实例中，换取 kernel 内少分支、布局固定和更好的编译期优化。

**Code：**

```python
# 来源：setup.py L304-L309
ext_modules.append(
    CUDAExtension(
        name="flash_attn_2_cuda",
        sources=[
            "csrc/flash_attn/flash_api.cpp",
            "csrc/flash_attn/src/flash_fwd_hdim32_fp16_sm80.cu",
```

**Comment：** 这里能看到 FA2 的工程形态：Python API 后面不是一个通用解释器，而是一组预编译的 C++/CUDA 实例。[[FA04-FA2-Forward-02-源码走读]] 负责解释这些实例如何由 dispatch 选中。

## 5. FA2 中后期：从训练 forward 扩展到 serving decode

**Explain：** FA2 之后的版本持续把 serving 场景补进主包：小 `seqlen_q` decode、KV cache、SplitKV、paged KV、local attention、ALiBi、softcap 都进入接口和 kernel dispatch。演进重点从“只证明省显存”变成“同一 attention backend 如何服务训练、prefill 与 decode”。

**Code：**

```markdown
# 来源：README.md L450-L458
### 2.2: Optimize for inference
The bottleneck here is to load KV cache as fast as possible
See the function `flash_attn_with_kvcache`
```

```markdown
# 来源：README.md L475-L482
### 2.5: Paged KV cache.
Support paged KV cache
### 2.6: Softcapping.
Support attention with softcapping
```

**Comment：** 这就是 [[FA05-KV-Cache-00-MOC]] 的位置：decode 不只是把 forward 的 batch size 改小，而是引入 cache append、paged addressing、SplitKV combine 等新的 IO 边界。

## 6. FA3：Hopper 时代的专门路径

**Explain：** FA3 是面向 Hopper 的 beta 路径，目标是吃到 H100/H800 上新的 memory/copy/MMA 能力。它不推翻 FA1 的 IO-aware 原理，而是把同一个 attention 问题放到 Hopper 的硬件语义下重新组织。

**Code：**

```markdown
# 来源：README.md L30-L47
## FlashAttention-3 beta release
FlashAttention-3 is optimized for Hopper GPUs
Currently released:
- FP16 / BF16 forward and backward, FP8 forward
Requirements: H100 / H800 GPU, CUDA >= 12.3.
```

**Comment：** FA3 的工程重点是 `hopper/` 里的 arch dispatch、paged KV、scheduler metadata 与 combine kernel。详见 [[FA06-Hopper-CuTe-01-核心概念]] 与 [[FA06-Hopper-CuTe-02-源码走读]]。

## 7. FA4：从静态模板走向 CuTeDSL/JIT

**Explain：** FA4 仍然是 FlashAttention，不是新的 attention 数学。变化在工程层：用 CuTeDSL 表达 kernel，把架构、shape、dtype、mask、paged KV、block sparse 等组合转换成 kernel object 和 compile key，再用 JIT cache 避免每次重复编译。

**Code：**

```markdown
# 来源：README.md L80-L91
## FlashAttention-4 (CuTeDSL)
FlashAttention-4 is written in CuTeDSL and optimized for Hopper and Blackwell GPUs
```

```markdown
# 来源：flash_attn/cute/README.md L1-L3
# FlashAttention-4 (CuTeDSL)
FlashAttention-4 is a CuTeDSL-based implementation of FlashAttention for Hopper and Blackwell GPUs.
```

**Comment：** 对生产系统来说，FA4 的新风险不是“算不准 softmax”，而是 JIT 首次编译延迟、cache key 覆盖、shape bucketing、cache 预热和多进程复用。详见 [[FA06-Hopper-CuTe-03-数据流与交互]] 与 [[FA06-Hopper-CuTe-04-关键问题]]。

## 8. 第一代与第四代如何对照

| 维度 | FA1 | FA4 |
|------|-----|-----|
| 核心问题 | attention 的 `N x N` 中间矩阵造成 HBM traffic | 新 GPU 与复杂特性组合造成 kernel 组织复杂度 |
| 算法不变量 | exact attention、tile、online softmax、不长期保存完整 `P` | 仍保持这些不变量 |
| 工程重点 | 证明 IO-aware 分块可行 | 用 CuTeDSL/JIT 管理架构、shape、dtype 和特性组合 |
| 主要风险 | 读者容易只理解“省显存”，忽略 IO traffic | 读者容易只看 benchmark，忽略 compile cache 与 warmup |
| 推荐阅读 | [[FA01-Attention-IO-00-MOC]] → [[FA02-Online-Softmax-00-MOC]] | [[FA06-Hopper-CuTe-00-MOC]] |

## 9. 读源码时的边界

- 不要把 FA4 理解成“更近似的 attention”；它改变的是 kernel 表达和编译路径。
- 不要在当前 `flash-attn 2.8.4` 源码树里编造 FA1 目录；FA1 在这里主要通过论文标题、1.x 升级说明和算法原理章节承接。
- 如果目标是生产接入，优先分清 FA2 主包、FA3 Hopper 包、FA4 `flash-attn-4` 包的安装和 import 边界。
- 如果目标是源码学习，先用 FA1 原理解释为什么 `S/P` 不落 HBM，再用 FA2/FA4 对比静态模板与 JIT cache 的工程取舍。

