---
type: batch-doc
module: FA03-Python-API
batch: "FA03"
doc_type: walkthrough
title: "Python API 与绑定 · 源码走读"
tags:
  - flash-attn/batch/fa03
  - flash-attn/module/python-api
  - flash-attn/doc/walkthrough
updated: 2026-07-05
---

# Python API 与绑定 · 源码走读

## 1. 公开函数先进入 autograd Function

### 1.1 `flash_attn_func` 把用户参数转给 `FlashAttnFunc.apply`

问题与约束：
- 用户 API 要保持简单，不能暴露 C++ extension 的所有内部参数。
- 训练路径需要保存 backward 所需的张量、LSE 和 RNG 状态。
- inference/no-grad 路径不应无条件保存 autograd 上下文。
- MQA/GQA、causal mask、local window、ALiBi、deterministic backward 等语义要从公开 API 传下去。

设计选择：
- `flash_attn_func` 只收集用户可见参数。
- 文档说明 MQA/GQA head 关系、causal mask 对齐、sliding window 和返回值语义。
- 最终调用 `FlashAttnFunc.apply(...)`。
- 将 `torch.is_grad_enabled()` 作为显式参数传入。

Explain：
公开 API 不直接调用 CUDA extension，而是进入 autograd Function。这样同一个函数可以同时服务训练、no-grad 推理和测试返回 attention probs 的场景。

来源：flash_attn/flash_attn_interface.py L1156-L1230

Code：

```python
def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    """
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        torch.is_grad_enabled(),
    )
```

代码逻辑：
- 定义用户侧 dense attention API。
- 接收 Q/K/V、dropout、scale、causal、window、softcap、ALiBi 和 deterministic 等参数。
- 在 docstring 中说明 MQA/GQA、mask、window 和返回值。
- 调用 `FlashAttnFunc.apply`。
- 把当前 grad enabled 状态传入 autograd Function。

为什么这样写：
- 公开 API 稳定，内部 forward/backward 保存策略由 autograd Function 管理。
- `torch.is_grad_enabled()` 显式传入，forward 可根据训练/推理状态决定保存多少上下文。
- 用户参数保持 Python 语义，底层 layout/dispatch 延后处理。

不变量与失败模式：
- Q head 数必须能被 KV head 数整除。
- evaluation 中 dropout 应为 0。
- `return_attn_probs` 是测试用途，不能被当作稳定 attention matrix 输出。
- 如果 grad 状态判断不准确，可能多保存上下文或缺失 backward 所需状态。

Comment：
Python API 的第一层分叉不是 C++，而是 autograd context 管理。

## 2. Custom op wrapper 才调用 CUDA extension

### 2.1 `_flash_attn_forward` 包装 `flash_attn_gpu.fwd`

问题与约束：
- PyTorch 2 custom op/compile 路径需要有可注册的 op 名称。
- CUDA extension 需要 Q/K/V 的最后一维满足 contiguous 访问。
- Python 层不应复制 C++ 的 kernel dispatch 和 shape 校验。
- dense path 没有 varlen cu_seqlens 参数。

设计选择：
- 用 `_torch_custom_op_wrapper("flash_attn::_flash_attn_forward", ...)` 注册 CUDA custom op。
- 进入 extension 前调用 `maybe_contiguous`。
- 调 `flash_attn_gpu.fwd(...)`。
- dense path 的 varlen 位置传 `None`。
- 返回 `out/softmax_lse/S_dmask/rng_state`。

Explain：
`_flash_attn_forward` 是 Python dispatcher 到 CUDA extension 的窄桥。它处理 PyTorch custom op 接入和基本 contiguous 归一化，然后把真正的检查、参数装配和 kernel specialization 交给 C++。

来源：flash_attn/flash_attn_interface.py L84-L113

Code：

```python
@_torch_custom_op_wrapper("flash_attn::_flash_attn_forward", mutates_args=(), device_types="cuda")
def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    return_softmax: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse, S_dmask, rng_state = flash_attn_gpu.fwd(
        q,
        k,
        v,
        None,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        None,
    )
    return out, softmax_lse, S_dmask, rng_state
```

代码逻辑：
- 注册名为 `flash_attn::_flash_attn_forward` 的 CUDA custom op。
- 声明输入参数和返回 tuple 类型。
- 将 Q/K/V 转为需要时 contiguous。
- 调用 extension 的 `fwd`。
- dense path 传入 `None` 作为 varlen/输出等占位。
- 返回输出、LSE、dropout mask/softmax 和 RNG state。

为什么这样写：
- custom op 名称给 PyTorch dispatcher/compile 路径一个稳定锚点。
- contiguous 处理留在 Python，避免 C++ 入口接收太多非标准 stride 情况。
- C++/CUDA 层负责性能关键逻辑，Python wrapper 保持轻薄。

不变量与失败模式：
- custom op 只声明 CUDA device 类型。
- `maybe_contiguous` 必须保证 C++ 入口的最后一维访问约束。
- dense path 中 cu_seqlens 位置为 None，不能误走 varlen 语义。
- `S_dmask` 只有在 return/dropout 相关路径中有实际意义。

Comment：
这里是 Python 代码第一次真正越过边界进入编译出的 CUDA extension。

## 3. pybind 把 Python 名称映射到 C++ 实现

### 3.1 `PYBIND11_MODULE` 暴露 dense、varlen、backward 和 KV cache 入口

问题与约束：
- Python wrapper 需要通过同一个 extension module 调用多个 C++ 实现。
- dense、varlen、backward 和 KV cache 的参数约束不同。
- 绑定层要给每个入口稳定的 Python 名称。

设计选择：
- 用 `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` 定义扩展模块。
- `fwd` 绑定到 `mha_fwd`。
- `varlen_fwd` 绑定到 `mha_varlen_fwd`。
- `bwd/varlen_bwd` 分别绑定 dense/varlen backward。
- `fwd_kvcache` 绑定到 KV cache forward。

Explain：
这段是 Python/C++ 边界的锚点。看到 Python 里的 `flash_attn_gpu.fwd`，可以从这里直接反查到 C++ 的 `mha_fwd`。

来源：csrc/flash_attn/flash_api.cpp L1481-L1488

Code：

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashAttention";
    m.def("fwd", &FLASH_NAMESPACE::mha_fwd, "Forward pass");
    m.def("varlen_fwd", &FLASH_NAMESPACE::mha_varlen_fwd, "Forward pass (variable length)");
    m.def("bwd", &FLASH_NAMESPACE::mha_bwd, "Backward pass");
    m.def("varlen_bwd", &FLASH_NAMESPACE::mha_varlen_bwd, "Backward pass (variable length)");
    m.def("fwd_kvcache", &FLASH_NAMESPACE::mha_fwd_kvcache, "Forward pass, with KV-cache");
}
```

代码逻辑：
- 创建 Torch extension module。
- 设置模块文档字符串。
- 注册 dense forward。
- 注册 varlen forward。
- 注册 dense backward。
- 注册 varlen backward。
- 注册 KV cache forward。

为什么这样写：
- 不同调用形态拆成不同 pybind 函数，Python wrapper 可以保持薄分发。
- C++ 侧可以为每个入口写独立参数校验。
- 绑定名与 Python 调用名一一对应，便于源码追踪。

不变量与失败模式：
- `TORCH_EXTENSION_NAME` 必须和编译出的 Python module 名匹配。
- `FLASH_NAMESPACE` 下的函数签名必须与 pybind 可绑定类型一致。
- Python wrapper 调用的名称必须在模块中注册。
- 某个入口缺失会在 import 后调用时暴露为 attribute error。

Comment：
pybind 层分入口，而不是一个万能函数承接所有 attention 模式。

## 4. 模块层按 batch layout 选择 packed API

### 4.1 `FlashSelfAttention.forward` 在 dense qkvpacked 和 varlen qkvpacked 间分发

问题与约束：
- 模型层常拿到 packed QKV，而不是拆开的 Q/K/V。
- 输入可能是 padded dense batch，也可能是 unpadded varlen layout。
- varlen layout 必须提供 int32 `cu_seqlens` 和整数 `max_seqlen`。
- dropout 概率取决于训练/评估状态。

设计选择：
- 先断言 qkv dtype 为 fp16/bf16 且在 CUDA 上。
- `causal` 参数缺省时使用模块默认值。
- 以 `cu_seqlens is not None` 判断 unpadded/varlen。
- varlen 分支校验 `cu_seqlens` 和 `max_seqlen` 后调用 `flash_attn_varlen_qkvpacked_func`。
- dense 分支调用 `flash_attn_qkvpacked_func`。
- dropout 在 eval 时传 0。

Explain：
模块层不直接处理 C++ extension，也不拆分底层 kernel 参数。它只根据 batch layout 选择 packed API，让 qkvpacked/varlen wrapper 继续下钻。

来源：flash_attn/modules/mha.py L83-L130

Code：

```python
def forward(self, qkv, causal=None, cu_seqlens=None, max_seqlen=None):
    assert qkv.dtype in [torch.float16, torch.bfloat16]
    assert qkv.is_cuda
    causal = self.causal if causal is None else causal
    unpadded = cu_seqlens is not None
    if self.alibi_slopes is not None:
        self.alibi_slopes = self.alibi_slopes.to(torch.float32)
    if unpadded:
        assert cu_seqlens.dtype == torch.int32
        assert max_seqlen is not None
        assert isinstance(max_seqlen, int)
        return flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens,
            max_seqlen,
            self.drop.p if self.training else 0.0,
            softmax_scale=self.softmax_scale,
            causal=causal,
            alibi_slopes=self.alibi_slopes,
            window_size=self.window_size,
            deterministic=self.deterministic,
        )
    else:
        return flash_attn_qkvpacked_func(
            qkv,
            self.drop.p if self.training else 0.0,
            softmax_scale=self.softmax_scale,
            causal=causal,
            alibi_slopes=self.alibi_slopes,
            window_size=self.window_size,
            deterministic=self.deterministic,
        )
```

代码逻辑：
- 检查 qkv dtype 和 CUDA device。
- 解析 causal 默认值。
- 判断是否 varlen/unpadded。
- 将 ALiBi slopes 转为 float32。
- varlen 分支检查 cu_seqlens dtype 和 max_seqlen。
- 调用 varlen qkvpacked API。
- dense 分支调用 dense qkvpacked API。
- 根据训练状态决定 dropout 概率。

为什么这样写：
- 模型层只关心张量组织方式和训练状态，底层 kernel 细节由 wrapper 负责。
- qkvpacked API 避免模型层重复拆 Q/K/V。
- eval 强制 dropout 为 0，避免推理路径产生随机状态和额外开销。

不变量与失败模式：
- qkv 必须是 fp16/bf16 CUDA tensor。
- varlen 分支必须提供 int32 `cu_seqlens` 和 int `max_seqlen`。
- ALiBi slopes dtype 要满足底层期望。
- `cu_seqlens is not None` 但 max_seqlen 缺失会直接断言失败。

Comment：
上层模块通常通过 packed/varlen API 间接进入同一套底层 C++/CUDA 能力。
