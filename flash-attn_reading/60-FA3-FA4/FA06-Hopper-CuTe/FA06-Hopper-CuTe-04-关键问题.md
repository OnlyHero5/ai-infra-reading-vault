---
type: batch-doc
module: FA06-Hopper-CuTe
batch: "FA06"
doc_type: faq
title: "FA3/FA4 Hopper 与 CuTe · 关键问题"
tags:
  - flash-attn/batch/fa06
  - flash-attn/module/hopper-cute
  - flash-attn/doc/faq
updated: 2026-07-05
---

# FA3/FA4 Hopper 与 CuTe · 关键问题

## 1. FA3/FA4 是否替代 FA2？

**Explain：** 源码中多条路径并存。FA2 是稳定主路径；FA3/FA4 面向新 GPU、新特性和新编译方式。实际使用哪条路径取决于安装包、import 路径、GPU 架构和上层框架适配。

**Comment：** 阅读时不要把“新路径存在”理解为“旧路径废弃”。AI infra 的算子后端常常长期多版本共存。

## 2. CuTeDSL 是不是改变了 FlashAttention 算法？

**Explain：** 没有改变核心算法：仍然是 tile attention、online softmax、减少 HBM traffic。改变的是 kernel 描述方式、dispatch 位置和编译缓存机制。

**Comment：** 如果你已经理解 [[FA01-Attention-IO-01-核心概念]] 和 [[FA02-Online-Softmax-01-核心概念]]，FA4 应该被看成同一原理在新工具链上的实现。

## 3. 为什么 FA4 要支持 arch override？

**Explain：** `_get_device_arch` 支持 `FLASH_ATTENTION_ARCH` 环境变量。这样可以在 CPU-only 编译、cross compile 或调试时显式选择 kernel path。

**Code：**

```python
# 来源：flash_attn/cute/interface.py L77-L96
arch_override = os.environ.get("FLASH_ATTENTION_ARCH", None)
if arch_override is not None:
    return _parse_arch_str(arch_override)
major, minor = torch.cuda.get_device_capability()
return major * 10 + int(minor)
```

**Comment：** 这类环境变量是排查“为什么我的机器走了某个 kernel path”的第一入口。

## 4. JIT cache 会带来什么生产问题？

**Explain：** cache miss 时需要编译，可能带来首次请求延迟；compile key 过多会增加缓存压力；不同 shape/feature 组合可能导致频繁编译。

**Comment：** 如果 serving runtime 引入 FA4，需要考虑 warmup、shape bucketing、cache 目录和多进程复用，而不只是 benchmark 单次 kernel 时间。

## 5. FP8 为什么只支持部分路径？

**Explain：** FP8 需要硬件、dtype、descale tensors、输出 dtype 和 backward 共同支持。源码明确限制 FA4 CuTe FP8 backward 尚不支持，并把 FP8 forward 限在 SM100。

**Comment：** 这类限制会影响模型压缩、推理量化和训练试验的可落地范围。

## 6. CuTe forward 在进入 kernel 前先确认哪些边界？

**Explain：** FA4 CuTe 会先检查架构、GQA head 比例、head_dim 约束、FP8 与 autograd 是否兼容，并决定输出和 LSE 的 dtype/shape。这些检查发生在 JIT/launch 前，是阅读 FA4 API 行为的第一层边界。

**Code：**

```python
# 来源：flash_attn/cute/interface.py L446-L466
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

**Comment：** 这解释了为什么“同样是 FA4”，不同 GPU、dtype、head_dim 或是否 requires_grad 会走到不同能力边界。

## 7. SplitKV 在 CuTe 路径如何变成启发式选择？

**Explain：** CuTe forward 根据 tile 配置、有效 K 长度、M block 数和 SM 数估算 `num_splits`。当 `num_splits < 1` 时，才使用 heuristic；如果特定架构和 head_dim 组合会导致 shared memory 压力过大，源码还会回退或调整 tile。

**Code：**

```python
# 来源：flash_attn/cute/interface.py L561-L583
m_block_size_effective = q_stage * tile_m
seqlen_k_loaded = max_seqlen_k if not local else max(0, min(max_seqlen_k, (window_size_right or max_seqlen_k) + (window_size_left or max_seqlen_k) + 1 + tile_m))
num_m_blocks = (seqlen_q_packgqa + m_block_size_effective - 1) // m_block_size_effective
total_mblocks = batch_size * num_head_kv * num_m_blocks
num_n_blocks = (seqlen_k_loaded + tile_n - 1) // tile_n
num_SMs = 132 if is_fake_mode() else torch.cuda.get_device_properties(device).multi_processor_count
if num_splits < 1:
    num_splits = num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, 128)

# SplitKV uses float32 partial output, which doubles the O buffer size
# in shared memory, causing OOM for diff-headdim (192, 128)
if arch // 10 in [10, 11] and head_dim != head_dim_v and num_splits > 1:
    if num_n_blocks >= 64 and head_dim_v != 512:
        tile_n = 64
        num_n_blocks = (seqlen_k_loaded + tile_n - 1) // tile_n
        num_splits = num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, 128)
    else:
        num_splits = 1

is_split_kv = num_splits > 1
if is_split_kv:
    out_partial = torch.empty(num_splits, *q_batch_seqlen_shape, num_head, head_dim_v, dtype=torch.float32, device=device)
    lse_partial = torch.empty(num_splits, *lse_shape, dtype=torch.float32, device=device)
```

**Comment：** 这段代码把 serving 中的长上下文并行度问题显式暴露出来：SplitKV 不是固定开关，而是形状、架构和 shared memory 约束共同决定的结果。

## 8. 编译缓存调用为什么要读参数列表？

**Explain：** FA4 CuTe 的 compile cache key 决定 kernel 版本，实际调用参数决定运行时张量和辅助结构。源码把 `out`、`lse`、cu_seqlens、page table、window、descale tensors、block sparse tensors 和 aux data 都传给缓存后的 kernel。

**Code：**

```python
# 来源：flash_attn/cute/interface.py L1056-L1089
call_args = [
    q_call,
    k_call,
    v_call,
    out.detach() if not is_split_kv else out_partial,
    lse_partial if is_split_kv else lse,
    softmax_scale,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_q,
    seqused_k,
    page_table,
    window_size_left,
    window_size_right,
    learnable_sink,
]
if arch // 10 in [10, 11]:
    call_args.append(descale_tensors)
call_args.extend([
    (
        normalized_block_sparse_tensors.mask_block_cnt,
        normalized_block_sparse_tensors.mask_block_idx,
        normalized_block_sparse_tensors.full_block_cnt,
        normalized_block_sparse_tensors.full_block_idx,
        normalized_block_sparse_tensors.cu_total_m_blocks,
        normalized_block_sparse_tensors.cu_block_idx_offsets,
        normalized_block_sparse_tensors.dq_write_order,
        normalized_block_sparse_tensors.dq_write_order_full,
    )
    if normalized_block_sparse_tensors is not None
    else None,
    AuxData(aux_tensors, aux_scalars),
])
_flash_attn_fwd.compile_cache[compile_key](*call_args)
```

**Comment：** 生产环境如果观察到 FA4 首次调用慢或缓存命中差，应同时检查 compile key 维度和这里的运行时参数组合，而不是只看 Python API 名称。

