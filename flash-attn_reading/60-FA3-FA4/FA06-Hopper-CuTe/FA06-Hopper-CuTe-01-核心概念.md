---
type: batch-doc
module: FA06-Hopper-CuTe
batch: "FA06"
doc_type: concept
title: "FA3/FA4 Hopper 与 CuTe · 核心概念"
tags:
  - flash-attn/batch/fa06
  - flash-attn/module/hopper-cute
  - flash-attn/doc/concept
updated: 2026-07-04
---

# FA3/FA4 Hopper 与 CuTe · 核心概念

## 1. FA2、FA3、FA4 的分层

**Explain：** 在本仓库中，FA2 主路径是 `flash_attn_2_cuda` 的 C++/CUDA template；FA3 在 `hopper/` 下提供 Hopper 相关实现；FA4 在 `flash_attn/cute/` 下以 CuTeDSL/JIT 方式组织新 kernel。

| 路径 | 典型入口 | 主要关注 |
|------|----------|----------|
| FA2 | `flash_attn/flash_attn_interface.py` + `csrc/flash_attn` | Ampere 及之后的稳定 CUDA extension |
| FA3 | `hopper/flash_api.cpp` | Hopper 专门 dispatch、paged KV、combine、scheduler metadata |
| FA4 | `flash_attn/cute/interface.py` | CuTeDSL、arch dispatch、compile cache、FP8/Blackwell 方向 |

## 2. FA3 仍是 C++/CUDA extension 思路

**Explain：** FA3 的 `hopper/flash_api.cpp` 仍通过 C++ 入口、参数结构和 dispatch 启动 kernel，但 dispatch 维度围绕 arch、SplitKV、paged KV、PackGQA、softcap 等新组合展开。

**Code：**

```cpp
// 来源：hopper/flash_api.cpp L367-L383
void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    TORCH_CHECK(params.num_splits >= 1);
    ARCH_SWITCH(params.arch, Arch, [&] {
        SPLIT_SWITCH(params.num_splits > 1, Split, [&] {
            PAGEDKV_SWITCH(params.page_table && !params.pagedkv_tma, PagedKVNonTMA, [&] {
                PACKGQA_SWITCH(params.pack_gqa, PackGQA_, [&] {
                    static constexpr bool PackGQA = PackGQA_ || Arch < 90 || PagedKVNonTMA || Split;
                    SOFTCAP_SWITCH(params.softcap > 0.0, Has_softcap, [&] {
                        run_mha_fwd_constexpr<Arch, Split, PagedKVNonTMA, PackGQA, Has_softcap>(params, stream);
                    });
                });
            });
        });
    });
}
```

**Comment：** 这里能看到 FA3 的系统视角：同一个 attention 算子需要同时考虑架构、SplitKV、paged KV、GQA 打包和 softcap。

## 3. FA4 把 kernel 选择搬到 Python/CuTeDSL 层

**Explain：** FA4 的 `flash_attn/cute/interface.py` 会检查设备架构、head_dim、dtype、varlen、paged KV、sparsity、SplitKV 等条件，然后选择一个 CuTeDSL kernel object，并以 compile key 缓存编译结果。

**Code：**

```python
# 来源：flash_attn/cute/interface.py L77-L96
@lru_cache(maxsize=None)
def _get_device_arch():
    """Cached device arch check.

    Override with FLASH_ATTENTION_ARCH (e.g. 'sm_80' or '80') to select which
    kernel path to use (SM80/SM90/SM100/SM120) independently of the compilation
    target (CUTE_DSL_ARCH).
    """
    arch_override = os.environ.get("FLASH_ATTENTION_ARCH", None)
    if arch_override is not None:
        return _parse_arch_str(arch_override)
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + int(minor)
```

**Comment：** 这把“当前 GPU 该走哪条 kernel 路径”的判断显式放在 Python 层，对调试和实验新架构更友好。

## 4. FA4 公开 API 更实验性、更可组合

**Explain：** FA4 `flash_attn_func` 暴露了 `qv`、`gather_kv_indices`、`learnable_sink`、`score_mod`、`mask_mod`、block sparse 等参数，说明它不仅是 FA2 API 的平移，而是面向更多 attention 变种。

**Code：**

```python
# 来源：flash_attn/cute/interface.py L2709-L2732
def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
```

**Comment：** 对 AI infra 学习者来说，FA4 是理解“attention backend 逐渐变成可编译 DSL 生态”的好入口。

