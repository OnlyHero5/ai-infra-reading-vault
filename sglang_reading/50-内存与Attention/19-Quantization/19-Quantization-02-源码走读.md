---
type: batch-doc
module: 19-Quantization
batch: "19"
doc_type: walkthrough
title: "Quantization · 源码走读"
tags:
 - sglang/batch/19
 - sglang/module/quantization
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-05
---
# Quantization · 源码走读

> 读法：SGLang 的量化层不是单一 kernel，而是一套“配置选择 Method、Method 创建 layer 权重、加载后修正权重、forward 时 dispatch 到 backend”的接口体系。本篇按抽象接口、FP8、GPTQ/AWQ、KV cache、无量化 MoE 与 Marlin layout 依次看。

---

## 1. 抽象接口

### 1.1 `QuantizeMethodBase` 与 `LinearMethodBase`：create/apply 两阶段协议

来源：python/sglang/srt/layers/quantization/base_config.py L20-L84

**问题与约束：** 不同量化方法要在模型初始化阶段注册不同参数，又要在 forward 阶段用同一 layer 调用入口执行计算；Linear 层还受 tensor parallel 分片形状影响。

**设计选择：** 抽象基类把接口拆成 `create_weights`、`apply`、`process_weights_after_loading`；`LinearMethodBase` 进一步固定 linear 权重创建需要的输入分片、输出分片、全局尺寸和参数 dtype。

**Explain：** `create_weights` 管 layer 上有哪些 Parameter/scale/zero point，`apply` 管 forward 时如何消费这些字段。二者分离后，checkpoint loading 可以插在中间，并由 `process_weights_after_loading` 做转置或 scale 修正。

**Code：**

```python
class QuantizeMethodBase(ABC):
    """Base class for different quantized methods."""

    def create_weights(
        self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs
    ):
        """Create weights for a layer.

        The weights will be set as attributes of the layer."""
        raise NotImplementedError()

    @abstractmethod
    def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        """Apply the weights in layer to the input tensor.

        Expects create_weights to have been called before on the layer."""
        raise NotImplementedError()

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Process the weight after loading.

        This can be used for example, to transpose weights for computation.
        """
        return


class LinearMethodBase(QuantizeMethodBase):
    """Base class for different (maybe quantized) linear methods."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for a linear layer.
           The weights will be set as attributes of the layer.
        """
        raise NotImplementedError()

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the weights in layer to the input tensor.
        Expects create_weights to have been called before on the layer."""
        raise NotImplementedError()
```

**代码逻辑：** 基类定义创建权重、执行 forward、加载后处理三个扩展点。Linear 基类重载 `create_weights` 签名，把 TP 分片输入输出尺寸和 dtype 作为必需参数，并把 `apply` 固定为 `x + optional bias`。

**为什么这样写：** 量化方法差异主要体现在 layer 参数布局和 forward kernel，而模型层希望只持有一个 method 对象。两阶段协议让 layer 构造、checkpoint 加载和 forward dispatch 解耦。

**不变量与失败模式：** `apply` 调用前必须已经执行 `create_weights` 并完成 checkpoint load；分片尺寸必须匹配 tensor parallel 拆分；加载后处理不能改变 layer 对外语义。若某个 method 忘记注册 scale，forward kernel 会在运行时缺字段。

**Comment：** 这是全量化体系的接口底座：method 是 layer 和 backend kernel 之间的适配对象。

### 1.2 `FusedMoEMethodBase`：MoE 权重创建与 runner 绑定

来源：python/sglang/srt/layers/quantization/base_config.py L86-L100

**问题与约束：** MoE 层的权重维度多了 expert 轴，并且 forward 不只是一个 linear，而是 dispatch、expert GEMM、combine 的 runner 流程。

**设计选择：** `FusedMoEMethodBase` 要求子类实现带 `num_experts`、`hidden_size`、`intermediate_size_per_partition` 的 `create_weights`，并提供 `create_moe_runner`。

**Explain：** MoE 量化 method 既要定义每个 expert 的量化权重如何存储，也要把这些权重交给合适的 MoeRunner 执行。

**Code：**

```python
class FusedMoEMethodBase(QuantizeMethodBase):

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        raise NotImplementedError

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        raise NotImplementedError
```

**代码逻辑：** MoE 基类给权重创建增加 expert 数和 FFN 中间维度，并把 runner 创建作为单独扩展点留给具体量化实现。

**为什么这样写：** MoE 的执行计划依赖 expert 并行、runner backend 和量化格式。单纯继承 LinearMethodBase 不足以表达 expert 维和 runner 配置。

**不变量与失败模式：** `num_experts`、hidden size 和 intermediate partition 必须和 FusedMoE layer 一致；runner config 要匹配量化权重 layout。若 runner 与权重格式不匹配，会在 expert GEMM 阶段读错 scale 或 block shape。

**Comment：** MoE method 不只是“多个 linear”，它还负责把量化信息接到 MoeRunner。

### 1.3 `get_triton_quant_info`：LoRA MoE runner 的量化状态接口

来源：python/sglang/srt/layers/quantization/base_config.py L102-L120

**问题与约束：** LoRA MoE runner 调用 fused MoE kernel 时，需要知道 base weights 的量化 flags、scale 和 block shape；这些信息和各 method 的 `apply()` 内部构造必须一致。

**设计选择：** 在 MoE method 基类中定义 `get_triton_quant_info`，默认说明每个量化方法都应覆盖它，并返回 `TritonMoeQuantInfo`。

**Explain：** 这不是普通 `get_quant_method`，而是 MoE runner 读取量化状态的 side channel。它让 LoRA 路径复用 base MoE quant 信息，而不复制每个 method 的内部逻辑。

**Code：**

```python
    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        raise NotImplementedError

    def get_triton_quant_info(self, layer: torch.nn.Module) -> TritonMoeQuantInfo:
        """Return a ``TritonMoeQuantInfo`` describing the quantisation state
        stored on *layer*.

        The LoRA MoE runner calls this so that ``invoke_fused_moe_kernel``
        receives the correct flags / scales / block-shape for the base
        weights.  Each quantisation method must override this with the
        same construction it already uses inside ``apply()``.
        """
```

**代码逻辑：** 基类把 MoE `apply` 固定为接收 dispatch 输出并返回 combine 输入；随后定义 `get_triton_quant_info` 文档，要求子类返回描述 layer 量化状态的对象。

**为什么这样写：** LoRA 与基础 MoE quant 共用 kernel 调用约束。把 quant info 抽成接口，可以避免 LoRA runner 去猜每种量化方法在 layer 上放了哪些字段。

**不变量与失败模式：** `get_triton_quant_info` 的构造必须与 `apply()` 使用的量化状态一致；LoRA MoE runner 依赖它返回正确 flags/scale/block shape。若子类未覆盖或返回不一致，LoRA 路径会和 base MoE 路径行为分裂。

**Comment：** 这段接口说明了 MoE 量化状态不仅服务自身 forward，也服务组合型 runner。

---

## 2. FP8 dispatch 与 activation quant

### 2.1 `dispatch_w8a8_block_fp8_linear`：显式配置优先，auto 次之

来源：python/sglang/srt/layers/quantization/fp8_utils.py L394-L409

**问题与约束：** W8A8 block FP8 GEMM 有多个 backend，选择受用户配置、硬件能力和依赖可用性影响；调用方需要拿到一个可调用实现，而不是在每次 forward 重复判断。

**设计选择：** 先读取 `get_fp8_gemm_runner_backend()`；若不是 auto，进入显式 backend dispatch；auto 模式才根据硬件/backend 可用性选择。

**Explain：** 这是 FP8 block linear 的入口分发器。显式 `--fp8-gemm-backend` 比自动检测优先，避免用户指定 backend 后被 silent fallback。

**Code：**

```python
def dispatch_w8a8_block_fp8_linear() -> Callable:
    """
    Dispatch to the appropriate FP8 block linear implementation.

    This function selects the backend based on:
    1. The --fp8-gemm-backend server argument (preferred)
    2. Auto-detection based on hardware capabilities
    """
    backend = get_fp8_gemm_runner_backend()

    # Handle explicit backend selection via --fp8-gemm-backend
    if not backend.is_auto():
        return _dispatch_explicit_backend(backend)

    # Auto mode: Select based purely on hardware/backend availability
    return _dispatch_auto_backend()
```

**代码逻辑：** 函数取得 backend enum，显式配置时返回 `_dispatch_explicit_backend` 的结果；否则调用 `_dispatch_auto_backend`。

**为什么这样写：** backend 选择属于初始化/配置决策，不应散落在 FP8 apply 热路径。返回 callable 后，method 可以缓存并直接调用。

**不变量与失败模式：** backend enum 必须能表达 auto 和显式状态；显式配置不能被自动分支覆盖；返回值必须是兼容 W8A8 block FP8 linear 签名的 callable。若 auto 和显式混淆，会导致性能和用户预期不一致。

**Comment：** FP8 block GEMM 的第一层设计是“配置决策提前，forward 只调函数”。

### 2.2 `_dispatch_explicit_backend`：显式 FlashInfer backend 的硬件校验

来源：python/sglang/srt/layers/quantization/fp8_utils.py L419-L428

**问题与约束：** 用户显式要求 backend 时，若硬件或依赖不支持，继续 fallback 会掩盖配置错误；FlashInfer TRTLLM FP8 GEMM 需要 SM100/SM103 和 FlashInfer。

**设计选择：** `backend.is_flashinfer_trtllm()` 时检查 `is_sm100_supported()` 与 `is_flashinfer_available()`，不满足直接抛 `RuntimeError`，满足才返回 FlashInfer callable。

**Explain：** 显式 backend 是强约束而不是建议。这里用 fail-fast 把不可用 backend 暴露为启动错误。

**Code：**

```python
def _dispatch_explicit_backend(backend: Fp8GemmRunnerBackend) -> Callable:
    """Dispatch based on explicitly selected backend."""
    if backend.is_flashinfer_trtllm():
        if not (is_sm100_supported() and is_flashinfer_available()):
            raise RuntimeError(
                "FlashInfer FP8 GEMM requested via --fp8-gemm-backend=flashinfer_trtllm, "
                "but FlashInfer is not available or not supported on this hardware. "
                "FlashInfer TRTLLM FP8 GEMM requires SM100/SM103 GPUs and FlashInfer."
            )
        return flashinfer_gemm_w8a8_block_fp8_linear_with_fallback
```

**代码逻辑：** 函数识别 FlashInfer TRTLLM backend，检查硬件与库可用性；失败抛错，成功返回对应实现。

**为什么这样写：** 性能调优常依赖指定 backend。silent fallback 可能让 benchmark 或生产配置看似成功但使用了不同 kernel。

**不变量与失败模式：** FlashInfer backend 必须同时满足硬件和库条件；错误信息要指明配置来源和要求。若忽略硬件校验，kernel 可能在运行时失败或产生不支持的指令路径。

**Comment：** 显式 backend 的语义是“要么按指定跑，要么报错”。

### 2.3 `scaled_fp8_quant`：dynamic/static activation FP8 量化

来源：python/sglang/srt/layers/quantization/fp8_kernel.py L1790-L1836

**问题与约束：** FP8 activation 量化要支持 dynamic per-tensor、dynamic per-token 和 static scale；输入必须是二维 token-hidden 矩阵，输出可能需要 token padding。

**设计选择：** 函数断言 2D，按 padding 创建 FP8 output；`scale is None` 时进入 dynamic 分支，按 `use_per_token_if_dynamic` 决定 scale shape；`scale` 存在时要求标量并走 static 分支。每个分支按 aiter、vLLM op、native fallback 顺序执行。

**Explain：** 这个函数是 activation 侧量化入口，返回量化后的 FP8 tensor 和实际使用的 scale。后续 GEMM 依赖这两个输出。

**Code：**

```python
    def scaled_fp8_quant(
        input: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        num_token_padding: Optional[int] = None,
        use_per_token_if_dynamic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert input.ndim == 2, f"Expected 2D input tensor, got {input.ndim}D"
        shape = input.shape
        if num_token_padding:
            shape = (max(num_token_padding, input.shape[0]), shape[1])
        output = torch.empty(shape, device=input.device, dtype=fp8_dtype)

        if scale is None:
            # Dynamic scaling
            if use_per_token_if_dynamic:
                scale = torch.empty(
                    (shape[0], 1), device=input.device, dtype=torch.float32
                )
                if _use_aiter:
                    dynamic_per_token_scaled_quant(output, input, scale)
                elif _has_vllm:
                    torch.ops._C.dynamic_per_token_scaled_fp8_quant(
                        output, input.contiguous(), scale, None
                    )
                else:
                    _native_dynamic_per_token_quant_fp8(output, input, scale)
            else:
                scale = torch.zeros(1, device=input.device, dtype=torch.float32)
                if _use_aiter:
                    dynamic_per_tensor_quant(output, input, scale)
                elif _has_vllm:
                    torch.ops._C.dynamic_scaled_fp8_quant(output, input, scale)
                else:
                    _native_dynamic_per_tensor_quant_fp8(output, input, scale)
        else:
            # Static scaling
            assert (
                scale.numel() == 1
            ), f"Expected scalar scale, got numel={scale.numel()}"
            if _use_aiter:
                static_per_tensor_quant(output, input, scale)
            elif _has_vllm:
                torch.ops._C.static_scaled_fp8_quant(output, input, scale)
            else:
                _native_static_quant_fp8(output, input, scale)

        return output, scale
```

**代码逻辑：** 函数先确定 output shape 和 dtype；dynamic per-token 分配 `[tokens, 1]` scale，dynamic per-tensor 分配标量 scale；static 分支校验传入 scale 是标量。各分支选择可用 backend 并写入 output。

**为什么这样写：** Activation scale 有时来自 checkpoint，有时必须按输入动态计算。把三类模式集中在一个函数里，可以让 FP8 linear 只关心 `(output, scale)`，不关心 scale 来源。

**不变量与失败模式：** 输入必须 2D；static scale 必须是单元素；per-token dynamic scale shape 与 padded token 数一致；fallback 必须和 fast op 语义一致。若 padding shape 与 scale shape 不一致，block GEMM 会读到错误 scale。

**Comment：** FP8 的关键边界是 activation 量化：它把高精度输入和 scale 变成 GEMM 能消费的 FP8 输入。

---

## 3. GPTQ、AWQ 与 Marlin

### 3.1 `GPTQLinearScheme`：kernel 初始化与分片对齐检查

来源：python/sglang/srt/layers/quantization/gptq/schemes/gptq_linear.py L25-L60

**问题与约束：** GPTQ 4bit 权重按 group 和 pack factor 压缩；tensor parallel 分片后的输入/输出尺寸必须与量化 group 和打包格式对齐。

**设计选择：** 初始化时保存 quant config、识别 v2 checkpoint format，并创建 `GPTQLinearKernel`；`create_weights` 开头检查 input partition 能被 group size 整除，output partition 能被 pack factor numerator 整除。

**Explain：** GPTQ 的错误分片不能等到 kernel 解包时才发现。这里在权重创建阶段拦截不对齐的 TP 配置。

**Code：**

```python
class GPTQLinearScheme(GPTQLinearSchemeBase):
    def __init__(self, quant_config: GPTQConfig):
        self.quant_config = quant_config
        self.use_v2_format = quant_config.checkpoint_format == "gptq_v2"
        self.kernel = self._init_kernel(quant_config)

    def _init_kernel(self, quant_config: GPTQConfig):
        from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import (
            GPTQLinearKernel,
        )

        return GPTQLinearKernel(quant_config)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        params_dtype: torch.dtype,
        weight_loader,
        **kwargs,
    ):
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )
        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.pack_factor.numerator != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )
```

**代码逻辑：** 构造函数建立 GPTQ kernel；`create_weights` 按输入分片和输出分片做两类整除校验，不满足时抛出带 TP 过大提示的 ValueError。

**为什么这样写：** 量化权重打包格式决定了合法分片边界。提前检查能把“TP size 太大导致不对齐”定位到模型初始化，而不是 forward 的低层 kernel。

**不变量与失败模式：** group size 和 pack factor 必须来自同一 quant config；输出分片总和必须代表当前 rank 的 packed 输出宽度。若校验缺失，GPTQ kernel 可能按错误边界解包权重。

**Comment：** GPTQ scheme 的第一职责是保证分片后的权重仍满足量化格式。

### 3.2 `AWQConfig`：MoE 不支持 Marlin 时回退 WNA16

来源：python/sglang/srt/layers/quantization/awq/awq.py L346-L370

**问题与约束：** AWQ 对 Linear 和 MoE 有不同 method；MoE Marlin 还受 layer 结构和 group size 支持范围限制，不支持时不能硬跑 AWQ MoE Marlin。

**设计选择：** Linear 返回 `AWQLinearMethod`；MoE 先检查 skip 规则，再检查 `check_moe_marlin_supports_layer`，不支持则 warning 并回退 `MoeWNA16Config`，支持时设置 `layer.scheme` 并返回 `AWQMoEMethod`。

**Explain：** AWQ MoE 的选择不是“配置是 AWQ 就必然 AWQ kernel”。它会按 layer 支持情况选择 Marlin 或回退方案。

**Code：**

```python
            return AWQLinearMethod(self)
        elif isinstance(layer, FusedMoE):
            if is_layer_skipped_awq(prefix, self.modules_to_not_convert):
                return None
            from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Config

            if not check_moe_marlin_supports_layer(layer, self.group_size):
                logger.warning_once(
                    f"Layer '{prefix}' is not supported by AWQMoeMarlin. "
                    "Falling back to Moe WNA16 kernels."
                )
                return MoeWNA16Config.from_config(self.full_config).get_quant_method(
                    layer, prefix
                )
            layer.scheme = self.get_moe_scheme(layer)
            return AWQMoEMethod(self)
        return None

    def get_linear_scheme(self, layer: torch.nn.Module):
        return AWQMarlinLinearScheme(self)

    def get_moe_scheme(self, layer: torch.nn.Module):
        return AWQMoEScheme(self)
```

**代码逻辑：** 分支按 layer 类型返回量化 method。MoE 分支先处理跳过列表，再导入 WNA16 fallback；Marlin 不支持时 warning 并委托 WNA16 config，支持时给 layer 绑定 AWQ MoE scheme。

**为什么这样写：** MoE kernel 支持矩阵比 Linear 更受限制。显式 fallback 可以让模型继续运行，同时用 warning 告诉用户该层没有走 AWQ MoE Marlin。

**不变量与失败模式：** skip 规则应按 prefix 生效；Marlin 支持检查必须覆盖 layer 和 group size；fallback config 要和完整 AWQ config 兼容。若不回退，不支持的 MoE layer 会在 kernel 调用时失败。

**Comment：** AWQ 的 layer 选择体现了量化系统的局部降级能力：能量化则量化，不能安全量化则返回替代 method 或 None。

### 3.3 `check_marlin_format`：识别 GPTQ checkpoint 的 Marlin layout

来源：python/sglang/srt/layers/quantization/gptq/gptq.py L43-L48

**问题与约束：** GPTQ checkpoint 可能来自不同工具，Marlin layout 的字段名存在兼容差异；加载路径需要知道权重是否已经是 Marlin 格式。

**设计选择：** 同时检查 `checkpoint_format == "marlin"` 和旧字段 `is_marlin_format`。

**Explain：** 这是 checkpoint metadata 的兼容 shim。不同 GPTQ 工具写法不同，但都映射成同一个布尔判断。

**Code：**

```python
def check_marlin_format(hf_quant_cfg: Dict[str, Any]) -> bool:
    # compat: gptqmodel and autogptq (eol) main use checkpoint_format: str
    # compat: autogptq <=0.7.1 is_marlin_format: bool
    return hf_quant_cfg.get("checkpoint_format") == "marlin" or hf_quant_cfg.get(
        "is_marlin_format", False
    )
```

**代码逻辑：** 函数从 HF quant config 读取两个可能字段，任一表示 Marlin 即返回 true。

**为什么这样写：** Marlin kernel 对权重 layout 有严格要求。加载时必须区分已转换和未转换格式，同时兼容旧工具输出。

**不变量与失败模式：** metadata 字段必须可信；旧字段默认 false；返回 true 后后续路径会按 Marlin layout 解释权重。若误判 true，kernel 会按错误 layout 读取 packed weight。

**Comment：** Marlin 是否可用不仅取决于硬件，也取决于 checkpoint 权重布局。

---

## 4. KV cache 与无量化 MoE

### 4.1 `BaseKVCacheMethod`：Attention KV scale 的加载后规范化

来源：python/sglang/srt/layers/quantization/kv_cache.py L18-L85

**问题与约束：** KV cache FP8 量化不通过普通 linear `apply`，而是在 Attention 读写 cache 时使用 k/v scale；checkpoint 可能有 k/v 独立 scale、只有单个 scale，或没有 scale。

**设计选择：** `create_weights` 在 attention layer 上注册 `k_scale/v_scale`，初始为无效 `-1.0` 并跳过 weight check；`apply` 直接报错；`process_weights_after_loading` 根据加载结果推导有效 float scale，并写回 layer。

**Explain：** KV cache method 只负责 scale 的生命周期，不负责 GEMM。scale 最终被 Attention backend 消费。

**Code：**

```python
class BaseKVCacheMethod(QuantizeMethodBase):
    """
    Quant method that adds `k_scale` and `v_scale` attributes to the
    Attention layer to support loading those scaling factors from checkpoints.
    The k/v_scale will be used to:
        - quantize k/v_cache entries before saving them to the cache
        - dequantize k/v_cache entries before fetching them from the cache
    """

    def __init__(self, quant_config: QuantizationConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module):
        """
        Create "weight" (aka k_scale and v_scale) for an attention layer.
        """
        # Initialize the KV cache scales to -1.0, which is an invalid value.
        # If the k/v_scale appears in the checkpoint, it will be
        # overwritten when loading weights.
        layer.k_scale = torch.nn.Parameter(
            torch.tensor(-1.0, dtype=torch.float32), requires_grad=False
        )
        layer.v_scale = torch.nn.Parameter(
            torch.tensor(-1.0, dtype=torch.float32), requires_grad=False
        )
        layer.k_scale._skip_weight_check = True
        layer.v_scale._skip_weight_check = True

    def apply(self, layer: torch.nn.Module) -> torch.Tensor:
        raise RuntimeError(f"{self.__class__.__name__}.apply should not be called.")

    def process_weights_after_loading(self, layer) -> None:
        if layer.k_scale > 0.0 and layer.v_scale > 0.0:
            # We prefer to use separate k_scale and v_scale if present
            k_scale = layer.k_scale.to("cpu").tolist()
            v_scale = layer.v_scale.to("cpu").tolist()
            if is_fp8_fnuz():
                k_scale *= 2
                v_scale *= 2
        elif layer.k_scale < 0.0 and layer.v_scale < 0.0:
            # If no scales were loaded (both scales are invalid negative
            # values), use the default value of 1.0
            k_scale = 1.0
            v_scale = 1.0
        else:
            # If we find a single kv_scale in the checkpoint, we remap
            # kv_scale to k_scale during weight loading, and duplicate
            # k_scale to v_scale here
            assert layer.k_scale > 0.0
            scale_to_duplicate = max(layer.k_scale, layer.v_scale)
            k_scale = scale_to_duplicate.to("cpu").tolist()
            v_scale = scale_to_duplicate.to("cpu").tolist()
            if is_fp8_fnuz():
                k_scale *= 2
                v_scale *= 2

        if not isinstance(k_scale, float) or not isinstance(v_scale, float):
            raise ValueError(
                "Only support per-tensor scaling factor " "for fp8 KV cache"
            )

        # These are used in the final Attention.forward()
        layer.k_scale.copy_(k_scale)
        layer.v_scale.copy_(v_scale)
        layer.k_scale_float = k_scale
        layer.v_scale_float = v_scale
```

**代码逻辑：** 创建阶段注册两个非训练参数并标记跳过检查。加载后处理分三种情况：独立 k/v scale、无 scale 默认 1.0、单 scale 复制给 k/v；FP8 FNUZ 下 scale 乘 2。最后要求 scale 是 float，并写回 tensor 与 float 字段。

**为什么这样写：** KV cache scale 来自 attention cache 读写，不适合塞进 linear forward。初始 `-1.0` 能区分“未加载”，加载后统一成 float 能简化 Attention.forward 的 backend 参数。

**不变量与失败模式：** `apply` 不应被调用；只支持 per-tensor scale；FNUZ scale 修正必须和 dtype 语义一致；最终 `k_scale_float/v_scale_float` 必须存在。若单 scale 不复制，K/V 其中一路会保留无效值。

**Comment：** KV cache 量化是 attention cache 的 scale 管理问题，不是 linear method 的计算问题。

### 4.2 无量化 MoE：DeepGEMM/FlashInfer runner 的 bf16 quant info

来源：python/sglang/srt/layers/quantization/unquant.py L488-L513

**问题与约束：** “无量化” MoE 仍要走统一 MoeRunner 接口；不同 runner 需要一个 quant info 对象描述权重 dtype、EP/TP 信息和 routed scaling 行为。

**设计选择：** DeepGEMM 分支创建 `DeepGemmMoeQuantInfo`，根据环境变量决定是否走 FP8 dispatch path；FlashInfer CUTLASS 分支创建 `FlashInferCutlassMoeQuantInfo(quant_type="bf16", ...)`，再调用 runner。

**Explain：** 这段说明 unquant 并不等于绕过 runner。即使权重是 bf16，也要把 runner 所需元信息封装成 quant info。

**Code：**

```python
            # otherwise use_fp8=True for FP8 dispatch path
            use_fp8 = not envs.SGLANG_DEEPEP_BF16_DISPATCH.get()
            quant_info = DeepGemmMoeQuantInfo(
                w13_weight=w13_weight,
                w2_weight=w2_weight,
                use_fp8=use_fp8,
            )
            return self.runner.run(dispatch_output, quant_info)
        elif self.use_flashinfer_cutlass:
            from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
                FlashInferCutlassMoeQuantInfo,
            )

            quant_info = FlashInferCutlassMoeQuantInfo(
                quant_type="bf16",
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                output_dtype=x.dtype,
                moe_ep_size=layer.moe_ep_size,
                moe_ep_rank=layer.moe_ep_rank,
                moe_tp_size=layer.moe_tp_size,
                moe_tp_rank=layer.moe_tp_rank,
                apply_routed_scaling_factor=not layer.should_fuse_routed_scaling_factor_in_topk,
            )
            return self.runner.run(dispatch_output, quant_info)
        elif self.use_flashinfer_trtllm_moe:
```

**代码逻辑：** DeepGEMM 分支读取环境变量决定 `use_fp8`，构造 quant info 并运行 runner；FlashInfer 分支导入对应 quant info 类，填入 bf16 类型、权重、输出 dtype、EP/TP rank/size 和 routed scaling 标志。

**为什么这样写：** MoE runner 需要统一输入，不想为无量化路径另开一套调用协议。把 bf16 也表示为 quant info，可以让 dispatch/combine 框架复用。

**不变量与失败模式：** unquant MoE 的 quant info 必须准确表达 bf16 权重；EP/TP rank 信息要和 layer 一致；routed scaling 标志不能反。若把 unquant 路径当普通 dense linear，会绕过 MoE dispatch 结构。

**Comment：** 无量化也是一种 method，只是 quant info 描述的是 bf16 权重而非压缩权重。

---

## 5. `Fp8LinearMethod.apply`

### 5.1 Marlin 分支：无原生 FP8 路径的 packed kernel

来源：python/sglang/srt/layers/quantization/fp8.py L760-L776

**问题与约束：** 某些硬件或 checkpoint layout 会选择 Marlin FP8 linear，要求传入 packed weight、weight scale、workspace 和当前分片尺寸。

**设计选择：** `apply` 入口先检查 `self.use_marlin`，命中时直接调用 `torch.ops.sglang.apply_fp8_marlin_linear`，把 layer 上的 weight、scale、workspace 和分片尺寸传入。

**Explain：** Marlin 分支是 FP8 apply 的最高优先级分支之一。它绕过普通 activation quant + GEMM 流程，使用专用 op 处理 packed weight。

**Code：**

```python
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_marlin:
            return torch.ops.sglang.apply_fp8_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                workspace=layer.workspace,
                size_n=layer.output_size_per_partition,
                size_k=layer.input_size_per_partition,
                bias=bias,
            )
```

**代码逻辑：** 函数签名接收 layer、输入和可选 bias；Marlin flag 命中时直接返回 Marlin op 输出，参数全部来自 layer 属性和当前输入。

**为什么这样写：** Marlin 对权重 layout、workspace 和尺寸有专用要求。把它放在单独分支，可以避免和 block quant/default FP8 的参数协议混杂。

**不变量与失败模式：** `layer.weight` 必须是 Marlin 期望 layout；workspace 必须已创建；分片尺寸要和 packed weight 匹配。若 checkpoint layout 不是 Marlin 却进入该分支，会得到错误输出或 kernel 失败。

**Comment：** FP8 linear 的 apply 不是单一路径，Marlin 是 layout 驱动的专用快路径。

### 5.2 block quant 与默认 `apply_fp8_linear`

来源：python/sglang/srt/layers/quantization/fp8.py L801-L840

**问题与约束：** FP8 block quant 要支持 CPU AMX、tuple 输入携带 activation scale、普通输入动态量化；非 block quant 则走默认 per-channel weight + activation scale 路径。

**设计选择：** `self.block_quant` 命中时，优先检查 Intel AMX backend；tuple 输入时用 `x[1]` 作为 input scale，否则传 `input_scale=None` 让 block FP8 linear 内部处理。未命中 block quant 时调用 `apply_fp8_linear`。

**Explain：** 这一段把 FP8 linear 的非 Marlin 路径分为 block quant 和默认 FP8。block quant 使用 dispatch 得到的 W8A8 block GEMM；默认路径使用 `apply_fp8_linear` 处理 per-channel weight scale 和 dynamic activation scale。

**Code：**

```python
        if self.block_quant:
            if use_intel_amx_backend(layer):
                return torch.ops.sgl_kernel.fp8_scaled_mm_cpu(
                    x,
                    layer.weight,
                    layer.weight_scale_inv,
                    self.quant_config.weight_block_size,
                    bias,
                    x.dtype,
                    True,  # is_vnni
                )

            if isinstance(x, tuple):
                return self.w8a8_block_fp8_linear(
                    input=x[0],
                    weight=layer.weight,
                    block_size=self.quant_config.weight_block_size,
                    weight_scale=layer.weight_scale_inv,
                    input_scale=x[1],
                    bias=bias,
                )

            return self.w8a8_block_fp8_linear(
                input=x,
                weight=layer.weight,
                block_size=self.quant_config.weight_block_size,
                weight_scale=layer.weight_scale_inv,
                input_scale=None,
                bias=bias,
            )

        return apply_fp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
            cutlass_fp8_supported=self.cutlass_fp8_supported,
            use_per_token_if_dynamic=self.use_per_token_if_dynamic,
        )
```

**代码逻辑：** block quant 下先走 AMX CPU op；否则根据输入是否 tuple 选择已有 input scale 或动态处理。非 block quant 则把 weight scale、input scale、cutlass 支持标志和 per-token dynamic flag 交给 `apply_fp8_linear`。

**为什么这样写：** block quant 和默认 FP8 对 scale 粒度和 backend dispatch 的要求不同。显式分支让每条路径只接收自己需要的参数，避免在 kernel 前再做复杂判断。

**不变量与失败模式：** tuple 输入必须是 `(input, input_scale)`；block size 要和 weight scale inverse 对齐；默认路径的 `layer.input_scale` 可以表示 static 或 dynamic 语义。若 tuple scale 与 input 不匹配，W8A8 block GEMM 会按错误 activation scale 计算。

**Comment：** `Fp8LinearMethod.apply` 是 FP8 配置落地的最终分叉点：Marlin、block quant、默认 FP8 各自有独立 ABI。
