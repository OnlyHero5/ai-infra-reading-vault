---
type: batch-doc
module: FA06-Hopper-CuTe
batch: "FA06"
doc_type: walkthrough
title: "FA3/FA4 Hopper 与 CuTe · 源码走读"
tags:
  - flash-attn/batch/fa06
  - flash-attn/module/hopper-cute
  - flash-attn/doc/walkthrough
updated: 2026-07-05
---

# FA3/FA4 Hopper 与 CuTe · 源码走读

## 1. FA3 的 PyTorch dispatcher 入口

### 1.1 `flash_attn_3::fwd` schema 集中暴露 serving/training 参数

问题与约束：
- FA3 要同时服务普通 Q/K/V、新 KV 写入、paged KV cache、RoPE、FP8 descale 和 SplitKV 等路径。
- 上层框架需要通过 PyTorch dispatcher 调用稳定 op 名称。
- 可选张量和会被 mutation 的输出/KV 参数必须在 schema 中表达清楚。

设计选择：
- 使用 `TORCH_LIBRARY(flash_attn_3, m)` 注册命名空间。
- 在 `fwd` schema 中列出 Q/K/V、`k_new/v_new`、`out`、cu_seqlens、page table、RoPE、descale、scheduler metadata、num_splits 等参数。
- 用 `Tensor(k_new!)?`、`Tensor(out!)?` 表示可选且可能被写入的张量。
- 返回 `(Tensor(out!), Tensor, Tensor, Tensor)`，把输出和辅助结果一起纳入 dispatcher 契约。

Explain：
FA3 的 C++ schema 本身就是功能地图。它把 Hopper 路径中的训练、varlen、KV cache serving 和 FP8/descales 参数压到一个 dispatcher 入口，调用方不需要根据特性切换不同 Python API。

来源：hopper/flash_api.cpp L1674-L1708

Code：

```cpp
m.def("fwd("
    "Tensor q,"
    "Tensor k,"
    "Tensor v,"
    "Tensor(k_new!)? k_new = None,"
    "Tensor(v_new!)? v_new = None,"
    "Tensor? q_v = None,"
    "Tensor(out!)? out = None,"
    "Tensor? cu_seqlens_q = None,"
    "Tensor? cu_seqlens_k = None,"
    "Tensor? cu_seqlens_k_new = None,"
    "Tensor? seqused_q = None,"
    "Tensor? seqused_k = None,"
    "int? max_seqlen_q = None,"
    "int? max_seqlen_k = None,"
    "Tensor? page_table = None,"
    "Tensor? kv_batch_idx = None,"
    "Tensor? leftpad_k = None,"
    "Tensor? rotary_cos = None,"
    "Tensor? rotary_sin = None,"
    "Tensor? scheduler_metadata = None,"
    "int num_splits = 0,"
    "bool? pack_gqa = None,"
    "int sm_margin = 0) -> (Tensor(out!), Tensor, Tensor, Tensor)");
```

代码逻辑：
- 注册 `flash_attn_3` 命名空间下的 `fwd` op。
- 定义输入 Q/K/V 和可选的新 KV。
- 定义 varlen、paged KV、RoPE、descale 和 scheduler 相关参数。
- 定义 SplitKV 的 `num_splits` 和 GQA packing 开关。
- 声明输出张量和辅助张量返回值。

为什么这样写：
- dispatcher schema 是 C++/Python/框架集成的共同边界。
- 单入口减少特性组合导致的 API 爆炸。
- mutation 标注让 PyTorch dispatcher 理解输出和 KV cache 写入语义。

不变量与失败模式：
- schema 中参数顺序必须和实际实现注册保持一致。
- 可写张量的 alias/mutation 标注必须准确，否则 autograd/dispatcher 可能误判副作用。
- 上层传入 page table、cu_seqlens、RoPE 等组合时，底层实现仍需做能力校验。

Comment：
FA3 的入口从 schema 就能看出它面向 serving 场景做了 KV cache 和 scheduler metadata 扩展。

## 2. FA4 Python 包入口很薄

### 2.1 `flash_attn.cute.__init__` 只公开普通和 varlen 函数

问题与约束：
- FA4 内部包含 arch dispatch、validation、CuTe tensor 转换和 JIT compile cache。
- 对用户而言，公开 API 应尽量接近 FA2 的调用习惯。
- 内部 helper 和 kernel object 不应成为稳定公共接口。

设计选择：
- 从 `.interface` 只导入 `flash_attn_func` 和 `flash_attn_varlen_func`。
- `__all__` 只列这两个函数。
- 版本缺失时 fallback 为 `0.0.0`，但不扩大公开面。

Explain：
FA4 的 package 入口刻意保持很小。读源码时，`__init__.py` 只告诉你公共 API 名称；真正的架构选择和 JIT 行为要进入 `interface.py`。

来源：flash_attn/cute/__init__.py L10-L18

Code：

```python
from .interface import (
    flash_attn_func,
    flash_attn_varlen_func,
)

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
]
```

代码逻辑：
- 从内部 interface 模块导入两个函数。
- 在 `__all__` 中声明公开导出。
- 没有导出 kernel class、compile cache 或 helper。

为什么这样写：
- API 面保持稳定，内部 JIT 和 CuTeDSL 细节可以演化。
- 用户迁移时仍看到熟悉的函数名。
- 降低内部对象被外部依赖后难以重构的风险。

不变量与失败模式：
- `interface.py` 必须导出这两个函数。
- 如果未来新增公开 API，需要同步 `__all__`。
- 用户不能依赖未导出的内部 helper 作为稳定接口。

Comment：
FA4 的复杂度不在包入口，而在 `_flash_attn_fwd` 内部。

## 3. FA4 forward 先做能力边界校验

### 3.1 `_flash_attn_fwd` 校验架构、head 形状、FP8 和输出形状

问题与约束：
- CuTe kernel 选择依赖 GPU 架构、head_dim、GQA 比例、dtype、FP8 descale 和 backward 支持。
- 不支持的组合如果进入 JIT，会浪费编译时间并产生更难理解的错误。
- 输出张量和 LSE shape 必须在 kernel 执行前固定。

设计选择：
- `_arch` 未显式传入时读取当前 device arch。
- 限制 compute capability 在 8.x 到 12.x。
- 校验 `num_head % num_head_kv == 0`。
- 根据 dtype alignment 和 arch 校验 head dim。
- 规范化 `softmax_scale`、`softcap`、`pack_gqa`。
- FP8 且需要梯度时直接抛 `NotImplementedError`。
- 预分配或校验 `out/lse`。

Explain：
这段把 FA4 能力边界前置到 Python interface。它先把输入组合规范化成 kernel 可理解的参数，再决定是否继续进入 CuTe JIT。

来源：flash_attn/cute/interface.py L446-L516

Code：

```python
arch = _get_device_arch() if _arch is None else _arch
assert arch // 10 in [8, 9, 10, 11, 12], "Unsupported compute capability. Supported: 8.x, 9.x, 10.x, 11.x, 12.x"
assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
alignment = 16 // v.element_size()
if arch // 10 not in [8, 12]:
    _validate_head_dims(head_dim, head_dim_v, arch // 10, alignment)
if softmax_scale is None:
    softmax_scale = (
        1.0 / math.sqrt(head_dim) if qv is None or q is None
        else 1.0 / math.sqrt(head_dim + head_dim_v)
    )
if softcap == 0.0:
    softcap = None
qhead_per_kvhead = num_head // num_head_kv
if pack_gqa is None:
    pack_gqa = qhead_per_kvhead > 1

is_fp8 = v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
requires_grad = any(t is not None and t.requires_grad for t in [q, k, v, qv])
if is_fp8 and requires_grad:
    raise NotImplementedError("FA4 CuTe FP8 backward is not supported yet (forward-only).")
```

代码逻辑：
- 解析目标架构。
- 校验架构白名单。
- 校验 GQA head 可整除。
- 计算 dtype alignment。
- 按架构校验 head dim。
- 推导默认 softmax scale。
- 把 softcap 0 转成 None。
- 推导 q head per kv head 和 pack GQA。
- 判断 FP8 和 requires grad。
- FP8 backward 未支持时拒绝继续。

为什么这样写：
- JIT 前快速失败能节省编译成本。
- Python 错误信息比底层 kernel 编译错误更可读。
- 形状、dtype、arch 规范化后，后面的 compile key 和 kernel object 才稳定。

不变量与失败模式：
- `_get_device_arch()` 必须反映当前执行 device。
- `num_head_kv` 不能为 0，且必须整除 `num_head`。
- FP8 descale 张量只在 FP8 输入下合法。
- unsupported arch/head_dim 组合会在 Python 层直接失败。

Comment：
FA4 把“哪些组合能跑”写在 Python interface，而不是等 CuTe 编译器报错。

## 4. 架构分支把 kernel 形态对象化

### 4.1 SM80/SM90/SM100/SM120 选择不同 forward kernel object

问题与约束：
- Ampere、Hopper、Blackwell 等架构的 MMA、TMA、persistent scheduling 和 feature support 不同。
- SM80 不支持 paged KV 和 SplitKV；SM90 不支持 SplitKV；SM100/110 有 MLA、2CTA、persistent 等特殊路径；SM120 又有自己的限制。
- kernel 编译需要把 tile、thread、GQA、mask/score mod、paged KV 等参数固定成对象配置。

设计选择：
- 按 `arch // 10` 分支创建不同 kernel object。
- SM80 创建 `FlashAttentionForwardSm80` 并拒绝 paged KV/SplitKV。
- SM90 创建 `FlashAttentionForwardSm90`，传入 overlap、RS PV、paged non-TMA 等参数。
- SM100/110 在 MLA、hd256 2CTA 和普通 Blackwell 路径间分支。
- SM120 创建 `FlashAttentionForwardSm120` 并拒绝当前不支持的 block sparsity、paged KV 和 SplitKV。

Explain：
FA4 把“要编译什么 kernel”表示成 Python kernel object。不同架构的 feature gate 和编译参数先在 Python 层定型，然后交给 CuTe JIT。

来源：flash_attn/cute/interface.py L823-L961

Code：

```python
if arch // 10 == 8:
    assert page_table is None, "paged KV not supported on SM 8.0"
    assert not is_split_kv, "SplitKV not supported on SM 8.0"
    fa_fwd = FlashAttentionForwardSm80(
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        is_causal=causal,
        is_local=local,
        pack_gqa=pack_gqa,
        tile_m=tile_m,
        tile_n=tile_n,
        num_stages=1,
        num_threads=num_threads,
        Q_in_regs=False,
        score_mod=score_mod,
        mask_mod=mask_mod,
        has_aux_tensors=aux_tensors is not None,
    )
elif arch // 10 == 9:
    assert not is_split_kv, "SplitKV not supported on SM 9.0"
    fa_fwd = FlashAttentionForwardSm90(
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        is_causal=causal,
        is_local=local,
        pack_gqa=pack_gqa,
        tile_m=tile_m,
        tile_n=tile_n,
        num_stages=2,
        num_threads=num_threads,
        Q_in_regs=False,
        intra_wg_overlap=intra_wg_overlap,
        mma_pv_is_rs=mma_pv_is_rs,
        mask_mod=mask_mod,
        score_mod=score_mod,
        has_aux_tensors=aux_tensors is not None,
        q_subtile_factor=q_subtile_factor,
        paged_kv_non_tma=page_size not in [None, tile_n],
    )
elif arch // 10 in [10, 11]:
    if qv is not None:
        fa_fwd = FlashAttentionMLAForwardSm100(...)
    else:
        flash_fwd_obj_cls = (
            BlackwellFusedMultiHeadAttentionForward
            if use_dedicated_hd256_kernel
            else FlashAttentionForwardSm100
        )
        fa_fwd = flash_fwd_obj_cls(...)
elif arch // 10 == 12:
    assert not use_block_sparsity, "Block sparsity not supported on SM 12.0"
    assert page_table is None, "Paged KV not supported on SM 12.0 in this PR"
    assert not is_split_kv, "SplitKV not supported on SM 12.0 in this PR"
    fa_fwd = FlashAttentionForwardSm120(...)
```

代码逻辑：
- 根据 `arch // 10` 进入对应架构分支。
- 每个分支先断言该架构不支持的特性没有被启用。
- 构造对应的 forward kernel object。
- 将 dtype、head dim、GQA、causal/local、tile、线程、mask/score mod 等参数写入对象。
- Blackwell 分支根据 MLA、hd256 和 dedicated kernel 条件选择更细对象。

为什么这样写：
- CuTe JIT 需要明确的 kernel object 来表达编译形态。
- 架构差异提前显式化，避免一个巨大 kernel 同时承载所有硬件分支。
- Python 层分发比静态预编译所有组合更灵活，也能给出更清楚的 unsupported feature 报错。

不变量与失败模式：
- `arch // 10` 的分支必须覆盖前面允许的所有 compute capability。
- 每个架构的 unsupported feature 断言要和 kernel 实际能力一致。
- tile/thread 参数必须满足对应 kernel object 的实现约束。
- 新增架构或 feature 时需要同步 validation、kernel object 和 compile key。

Comment：
FA4 的 dispatch 表在 Python 中显式可见，而 FA2 更多依赖 C++ template 实例分发。

## 5. CuTe 编译结果进入 forward cache

### 5.1 compile key 未命中时转换 tensor 并调用 `cute.compile`

问题与约束：
- FA4 支持的架构、shape、dtype、mask、paged KV、block sparse、aux tensor 组合很多。
- 全部静态预编译会造成构建时间和 wheel 体积膨胀。
- JIT 编译开销不能在每次 forward 都重复支付。

设计选择：
- 以 `compile_key` 查询 `_flash_attn_fwd.compile_cache`。
- cache 未命中时，把 PyTorch tensor 转成 CuTe tensor。
- 根据是否 SplitKV、是否 FP8、是否 block sparse/aux tensor 组装 compile args。
- 调用 `cute.compile(*compile_args, options="--enable-tvm-ffi")`。
- 将返回的可执行对象写入 compile cache。

Explain：
FA4 把 kernel 编译延迟到首次遇到某个实际组合时发生，并用 compile cache 复用结果。这样工程上避免预编译组合爆炸，但运行环境需要关注 warmup 和 cache 命中。

来源：flash_attn/cute/interface.py L767-L1017

Code：

```python
if compile_key not in _flash_attn_fwd.compile_cache:
    (
        cu_seqlens_q_tensor,
        cu_seqlens_k_tensor,
        seqused_q_tensor,
        seqused_k_tensor,
        learnable_sink_tensor,
    ) = [
        to_cute_tensor(t, assumed_align=4, leading_dim=0)
        if t is not None
        else None
        for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink)
    ]
    page_table_tensor = (
        to_cute_tensor(page_table, assumed_align=4, leading_dim=1)
        if page_table is not None
        else None
    )
    q_tensor, k_tensor, v_tensor, o_tensor = [
        to_cute_tensor(t) for t in (q, k, v, out if not is_split_kv else out_partial)
    ]
    compile_args = [
        fa_fwd,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        softmax_scale,
        cu_seqlens_q_tensor,
        cu_seqlens_k_tensor,
        seqused_q_tensor,
        seqused_k_tensor,
        page_table_tensor,
        window_size_left,
        window_size_right,
        learnable_sink_tensor,
    ]
    _flash_attn_fwd.compile_cache[compile_key] = cute.compile(
        *compile_args, options="--enable-tvm-ffi"
    )
```

代码逻辑：
- 检查 compile cache 是否已有当前 key。
- 缺失时转换 cu_seqlens、seqused 和 learnable sink。
- 可选转换 page table。
- 转换 Q/K/V/O。
- 根据执行模式准备 LSE、descale、sparse、aux 等参数。
- 组装 compile args。
- 调用 CuTe compile。
- 将编译产物存入 cache。

为什么这样写：
- JIT 只为实际用到的组合付费。
- cache 命中后避免重复编译。
- tensor 转 CuTe tensor 和 kernel object 一起构成编译期形态，使 CuTeDSL 能生成专用实现。

不变量与失败模式：
- `compile_key` 必须覆盖影响生成代码的所有参数。
- CuTe tensor 转换的 alignment/leading_dim 要和 kernel 访问方式一致。
- cache 未预热时首次请求会承担编译延迟。
- 如果 compile key 漏掉某个特性，可能错误复用不匹配的 kernel。

Comment：
FA4 与 FA2 最大的工程差异之一，是把组合爆炸从静态编译转移到了 JIT cache 管理。
