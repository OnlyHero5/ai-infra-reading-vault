---
type: batch-doc
module: FA03-Python-API
batch: "FA03"
doc_type: concept
title: "Python API 与绑定 · 核心概念"
tags:
  - flash-attn/batch/fa03
  - flash-attn/module/python-api
  - flash-attn/doc/concept
updated: 2026-07-04
---

# Python API 与绑定 · 核心概念

## 1. API 形态就是 attention 场景分类

**Explain：** FlashAttention 的公开 API 可以按输入布局和 serving 场景分组。普通 API 适合 `q/k/v` 分离；packed API 避免额外拼拆；varlen API 处理 padding 后的有效 token；KV cache API 服务 incremental decode。

**Code：**

```python
# 来源：flash_attn/__init__.py L8-L16
from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)
```

**Comment：**
- `qkvpacked` 常见于训练模块，`Q/K/V` 已经堆在同一张量中。
- `varlen` 解决 batch 内不同长度样本的 padding 浪费。
- `with_kvcache` 面向推理 decode，并明确不支持 backward。

## 2. 普通 API 的参数会下沉到 kernel specialization

**Explain：** `causal`、`window_size`、`softcap`、`alibi_slopes`、`return_attn_probs` 看起来是 Python 参数，实际会影响 C++ 校验、template branch 和 kernel 形态。

**Code：**

```python
# 来源：flash_attn/flash_attn_interface.py L1156-L1167
def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
```

**Comment：** 读 kernel dispatch 时，不要把这些参数当成“运行时 if”理解；很多条件会被转成编译期模板常量。

## 3. varlen 的核心是 `cu_seqlens`

**Explain：** 对训练 batch，padding token 参与 attention 是纯浪费。`unpad_input` 把有效 token 拉平成连续张量，再用 `cu_seqlens` 告诉 kernel 每条序列的边界。

**Code：**

```python
# 来源：flash_attn/bert_padding.py L112-L126
all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
max_seqlen_in_batch = seqlens_in_batch.max().item()
cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
return (
    index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
    indices,
    cu_seqlens,
    max_seqlen_in_batch,
    used_seqlens_in_batch,
)
```

**Comment：** `cu_seqlens` 是 FlashAttention 与上层数据管线的关键接口；它把 padding 问题转成连续 token 加边界数组的问题。

## 4. Extension 名称是 ABI 边界

**Explain：** Python 中的 `flash_attn_gpu.fwd` 来自编译出的 `flash_attn_2_cuda` 扩展。这个名字连接了 Python import、pybind module 和 C++ 实现。

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

**Comment：** 当线上报 `undefined symbol` 或 import error 时，问题往往在 Python 包版本、CUDA extension 编译产物和 PyTorch/CUDA ABI 的匹配关系。

