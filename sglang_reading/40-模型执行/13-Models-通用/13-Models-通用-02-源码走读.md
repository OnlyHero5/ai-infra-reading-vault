---
type: batch-doc
module: 13-Models-通用
batch: "13"
doc_type: walkthrough
title: "Models 通用 · 源码走读"
tags:
 - sglang/batch/13
 - sglang/module/models-common
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-05
---
# Models 通用 · 源码走读

> 走读主线：`model_loader/utils.py` 根据 HF config 解析模型 architecture，`registry.py` 把 architecture 映射到 SGLang 模型类，`llama.py` 展示通用 decoder-only 执行骨架，`qwen3.py` 展示 QK-Norm、attention TP 与 LayerCommunicator 的通用特化方式。

---

## 1. 模型 architecture 解析

### 1.1 get_model_architecture 把 HF architecture 转成 SGLang 模型类

问题与约束：
- HF config 的 `architectures` 只是字符串列表；运行时需要得到 SGLang 内部 `nn.Module` 类，并且要处理量化 Mixtral、Transformers fallback、MindSpore 实现等特殊情况。

设计选择：
- `get_model_architecture` 先读取 HF architectures，再根据量化和 `model_impl` 改写 architecture，最后通过 `ModelRegistry.resolve_model_cls` 得到模型类与最终 architecture。

Explain：
这段是模型加载入口的“architecture 归一化”层。它不会直接 import 具体模型文件，而是把本地支持、Transformers backend 和 MindSpore backend 的选择收敛成一个 architecture 列表，再交给 registry 解析。

来源：python/sglang/srt/model_loader/utils.py L195-L230

Code：

```python
def get_model_architecture(model_config: ModelConfig) -> Tuple[Type[nn.Module], str]:
    from sglang.srt.models.registry import ModelRegistry

    architectures = getattr(model_config.hf_config, "architectures", [])
    mixtral_supported = [
        "fp8",
        "compressed-tensors",
        "gptq_marlin",
        "awq_marlin",
        "quark_int4fp8_moe",
    ]
    if (
        model_config.quantization is not None
        and model_config.quantization not in mixtral_supported
        and "MixtralForCausalLM" in architectures
    ):
        architectures = ["QuantMixtralForCausalLM"]

    supported_archs = ModelRegistry.get_supported_archs()
    is_native_supported = any(arch in supported_archs for arch in architectures)
    if model_config.model_impl == ModelImpl.MINDSPORE:
        architectures = ["MindSporeForCausalLM"]
    elif not is_native_supported or model_config.model_impl == ModelImpl.TRANSFORMERS:
        architectures = resolve_transformers_arch(model_config, architectures)
    model_cls, resolved_arch = ModelRegistry.resolve_model_cls(architectures)
    setattr(model_config, "_resolved_model_arch", resolved_arch)
    setattr(model_config, "_resolved_model_impl", _model_impl_from_architecture(resolved_arch))
    return model_cls, resolved_arch
```

代码逻辑：
- 从 HF config 读取候选 architecture。
- 对不在支持列表内的 Mixtral 量化组合改写为 `QuantMixtralForCausalLM`。
- 若本地不支持或显式要求 Transformers，就调用 `resolve_transformers_arch`。
- 将解析结果缓存回 `model_config`，后续可直接读取 resolved 字段。

为什么这样写：
- architecture 解析必须在构造模型前完成，否则权重加载、后端选择和多模态处理都没有稳定类型。
- 把 fallback 选择放在 loader utils 中，registry 可以保持“字符串到类”的单一职责。

不变量与失败模式：
- `model_config.hf_config.architectures` 为空时，后续 registry 会走 unsupported 路径。
- 如果 architecture 被改写成 Transformers 或 MindSpore，但对应类没有注册，最终仍会在 registry 抛错。

Comment：
读模型加载链路时先看这里：它决定最终进入 native 模型、Transformers 包装模型，还是其他实现。

### 1.2 ModelRegistry 规范化候选 architecture 并按顺序 resolve

问题与约束：
- 一个 HF config 可能给出多个 architecture，且其中部分 architecture SGLang native 不支持；系统需要保留支持项，同时提供 Transformers fallback。

设计选择：
- `_normalize_archs` 先过滤 registry 已注册模型；若存在不支持项，就把 `TransformersForCausalLM` 追加到候选列表末尾。

Explain：
`resolve_model_cls` 对规范化后的候选列表按顺序尝试 `_try_load_model_cls`。这意味着 native 支持的 architecture 优先；只有找不到 native 类时才走 Transformers fallback。

来源：python/sglang/srt/models/registry.py L61-L91

Code：

```python
def _normalize_archs(
    self,
    architectures: Union[str, List[str]],
) -> List[str]:
    if isinstance(architectures, str):
        architectures = [architectures]
    if not architectures:
        logger.warning("No model architectures are specified")
    normalized_arch = list(
        filter(lambda model: model in self.models, architectures)
    )
    if len(normalized_arch) != len(architectures):
        normalized_arch.append("TransformersForCausalLM")
    return normalized_arch

def resolve_model_cls(
    self,
    architectures: Union[str, List[str]],
) -> Tuple[Type[nn.Module], str]:
    architectures = self._normalize_archs(architectures)
    for arch in architectures:
        model_cls = self._try_load_model_cls(arch)
        if model_cls is not None:
            return (model_cls, arch)
    return self._raise_for_unsupported(architectures)
```

代码逻辑：
- 字符串 architecture 被包装成单元素列表。
- 已注册类名保留原顺序。
- 只要候选中有不支持项，就把 Transformers fallback 放到最后。

为什么这样写：
- 这样既不牺牲 native 优先级，也允许未原生适配的模型走通用 Transformers 后端。
- fallback 放在末尾可以避免 native 模型被过早包装。

不变量与失败模式：
- `self.models` 必须已完成注册，否则所有 native architecture 都会被当成不支持。
- 如果 Transformers fallback 自身未注册，unsupported 错误会列出可支持 architecture。

Comment：
Registry 的 resolve 是顺序敏感的；候选列表的排列会直接影响最终模型实现。

### 1.3 import_model_classes 扫描 EntryClass 注册模型类

问题与约束：
- `sglang.srt.models` 下每个模型文件可能导出一个或多个 architecture 类；手工维护大表容易漏掉新模型。

设计选择：
- `import_model_classes` 遍历包下非 package 模块，import 后读取模块级 `EntryClass`，支持单类和 list 两种形式。

Explain：
每个模型文件通过 `EntryClass` 暴露给 registry。list 形式允许一个文件提供多个 architecture。注册时用类名作为 key，并通过 assert 防止重复实现覆盖。

来源：python/sglang/srt/models/registry.py L95-L125

Code：

```python
@lru_cache()
def import_model_classes(package_name: str, strict: bool = False):
    model_arch_name_to_cls = {}
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if not ispkg:
            if name.split(".")[-1] in envs.SGLANG_DISABLED_MODEL_ARCHS.get():
                logger.debug(f"Skip loading {name} due to SGLANG_DISABLED_MODEL_ARCHS")
                continue
            try:
                module = importlib.import_module(name)
            except Exception as e:
                if strict:
                    raise
                logger.warning(f"Ignore import error when loading {name}: {e}")
                continue
            if hasattr(module, "EntryClass"):
                entry = module.EntryClass
                if isinstance(entry, list):
                    for tmp in entry:
                        assert tmp.__name__ not in model_arch_name_to_cls
                        model_arch_name_to_cls[tmp.__name__] = tmp
                else:
                    assert entry.__name__ not in model_arch_name_to_cls
                    model_arch_name_to_cls[entry.__name__] = entry
```

代码逻辑：
- `lru_cache` 避免重复扫描同一 package。
- 环境变量可以禁用某些模型模块。
- import 失败时默认 warning 后跳过；strict 模式才抛出。
- EntryClass list 逐个注册，单类直接注册。

为什么这样写：
- 模型文件新增后只要提供 `EntryClass`，registry 就能自动发现。
- 禁用开关和非 strict import 能降低一个模型依赖缺失对整个 registry 的影响。

不变量与失败模式：
- `EntryClass.__name__` 必须与 HF architecture 可解析名称一致。
- 两个模块导出同名类会触发 assert，防止后注册覆盖先注册。

Comment：
这段解释了 SGLang 为什么能支持大量模型而不需要集中维护静态注册表。

### 1.4 unsupported 错误区分“检查失败”和“不支持”

问题与约束：
- 加载失败时，用户需要知道是已有 architecture 检查失败，还是完全不在支持列表中。

设计选择：
- `_raise_for_unsupported` 先检查原始候选是否包含支持项；包含则提示 inspection 失败，否则列出所有支持 architecture。

Explain：
如果 architecture 在支持列表里但最终没能 resolve，错误信息指向日志中的检查失败；如果完全不支持，则返回 supported architectures 列表。

来源：python/sglang/srt/models/registry.py L41-L53

Code：

```python
def _raise_for_unsupported(self, architectures: List[str]):
    all_supported_archs = self.get_supported_archs()

    if any(arch in all_supported_archs for arch in architectures):
        raise ValueError(
            f"Model architectures {architectures} failed "
            "to be inspected. Please check the logs for more details."
        )

    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {all_supported_archs}"
    )
```

代码逻辑：
- 先计算当前 registry 支持列表。
- 候选中存在支持项时，错误聚焦 import/inspect 过程。
- 候选完全不支持时，错误附带支持列表。

为什么这样写：
- 两类错误的排查路径不同：前者看 import 日志，后者看 HF config architecture。
- 把错误分类放在 registry 内，调用方不需要理解注册细节。

不变量与失败模式：
- 支持列表来自当前已注册模型；若注册阶段漏掉模块，错误会误报为不支持。
- 支持列表很长，读错误时应先看 architectures 原值，再看是否被 fallback 改写。

Comment：
这个错误信息是模型启动失败时的第一定位点。

---

## 2. Llama：通用 decoder-only 执行骨架

### 2.1 LlamaAttention 初始化 TP/GQA、RoPE 与 RadixAttention

问题与约束：
- Llama 系列既有 MHA，也有 GQA/MQA；在 tensor parallel 下，Q head 和 KV head 的切分规则不同。

设计选择：
- Q head 必须按 `tp_size` 均分；KV head 若不少于 TP size 就切分，否则在多个 TP rank 上复制。

Explain：
初始化阶段计算本 rank 的 `num_heads/num_kv_heads/q_size/kv_size`，随后用 `QKVParallelLinear` 生成融合 QKV 投影，用 `RowParallelLinear` 做输出投影，并构造 RoPE 与 `RadixAttention`。

来源：python/sglang/srt/models/llama.py L146-L205

Code：

```python
tp_size = get_parallel().tp_size
self.total_num_heads = num_heads
assert self.total_num_heads % tp_size == 0
self.num_heads = self.total_num_heads // tp_size
self.total_num_kv_heads = num_kv_heads
if self.total_num_kv_heads >= tp_size:
    assert self.total_num_kv_heads % tp_size == 0
else:
    assert tp_size % self.total_num_kv_heads == 0
self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
self.head_dim = getattr(config, "head_dim", self.hidden_size // self.total_num_heads)
self.q_size = self.num_heads * self.head_dim
self.kv_size = self.num_kv_heads * self.head_dim
self.qkv_proj = QKVParallelLinear(...)
self.o_proj = RowParallelLinear(...)
self.rotary_emb = get_rope(...)
self.attn = RadixAttention(
    self.num_heads,
    self.head_dim,
    self.scaling,
    num_kv_heads=self.num_kv_heads,
    layer_id=layer_id,
    quant_config=quant_config,
    prefix=add_prefix("attn", prefix),
)
```

代码逻辑：
- `num_heads` 直接按 TP 均分。
- KV head 数小于 TP size 时复制，否则均分。
- `q_size/kv_size` 决定 fused QKV projection 后 split 的宽度。
- `RadixAttention` 接收本 rank head 数和 KV head 数。

为什么这样写：
- GQA/MQA 的 KV head 少于 Q head，不能用 Q head 的切分规则直接处理。
- 把 RoPE 与 RadixAttention 在构造期固定下来，forward 中只处理张量流。

不变量与失败模式：
- `total_num_heads % tp_size == 0` 必须成立。
- 若 KV head 与 TP size 既不能均分也不能复制，assert 会阻止启动。

Comment：
LlamaAttention 初始化是理解 SGLang attention 维度契约的基本样板。

### 2.2 LlamaAttention.forward 在 native 与 NPU prepare 间选择

问题与约束：
- CUDA/CPU 常规路径可以直接 QKV split + RoPE；NPU decode 可以使用融合算子，但 extend 阶段或缺少接口时必须退回 native。

设计选择：
- forward 只选择 prepare 路径，然后统一调用 `RadixAttention` 和输出投影。

Explain：
native prepare 执行 `qkv_proj`、split、RoPE。NPU prepare 会在首层准备 cos/sin，再调用 `split_qkv_rmsnorm_rope`。最终两条路径都返回 `q/k/v`，后续 attention 逻辑一致。

来源：python/sglang/srt/models/llama.py L207-L252

Code：

```python
def forward_prepare_native(self, positions, hidden_states):
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k = self.rotary_emb(positions, q, k)
    return q, k, v

def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
) -> torch.Tensor:
    if (
        not _is_npu
        or not hasattr(self.rotary_emb, "get_cos_sin_with_position")
        or forward_batch.forward_mode.is_extend()
    ):
        q, k, v = self.forward_prepare_native(...)
    else:
        q, k, v = self.forward_prepare_npu(...)

    attn_output = self.attn(q, k, v, forward_batch)
    output, _ = self.o_proj(attn_output)
    return output
```

代码逻辑：
- 非 NPU、RoPE 不支持预取、extend 模式都走 native。
- NPU decode 才使用融合 prepare。
- `RadixAttention` 是两条 prepare 路径的共同后端。

为什么这样写：
- prepare 阶段可以硬件特化，attention 阶段保持统一接口。
- extend 模式通常涉及批量 prefill，保守走 native 可以避免 NPU decode 特化误用。

不变量与失败模式：
- prepare 返回的 `q/k/v` shape 必须符合 `RadixAttention` 预期。
- NPU 路径依赖 `get_cos_sin_with_position` 和 fused split 算子；缺失时必须 fallback。

Comment：
forward 的重点不是 attention 算法，而是把不同硬件的 QKV 准备收敛到同一个 attention 调用。

### 2.3 LlamaModel.forward 处理 PP 边界与 aux hidden capture

问题与约束：
- pipeline parallel 下，不同 rank 负责不同层；首 rank 从 embedding 开始，中间 rank 从 proxy tensor 接续，末 rank 才做最终 norm。

设计选择：
- `LlamaModel.forward` 根据 PP rank 选择输入来源，并只遍历 `[start_layer, end_layer)`；非末 rank 返回 `PPProxyTensors`。

Explain：
首 rank 将 `input_ids` 转成 embedding；非首 rank 从 `pp_proxy_tensors` 读取 hidden/residual。循环层时若当前层在 `layers_to_capture` 中，会保存 `hidden_states + residual` 作为 aux hidden。末 rank 做 norm 并返回 hidden 或 `(hidden, aux_hidden_states)`。

来源：python/sglang/srt/models/llama.py L385-L431

Code：

```python
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    input_embeds: torch.Tensor = None,
    pp_proxy_tensors: Optional[PPProxyTensors] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]], PPProxyTensors]:
    if self.pp_group.is_first_rank:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        residual = None
    else:
        assert pp_proxy_tensors is not None
        hidden_states = pp_proxy_tensors["hidden_states"]
        residual = pp_proxy_tensors["residual"]

    aux_hidden_states = []
    for i in range(self.start_layer, self.end_layer):
        if i in self.layers_to_capture:
            aux_hidden_states.append(hidden_states + residual)
        layer = self.layers[i]
        hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)

    if not self.pp_group.is_last_rank:
        return PPProxyTensors({"hidden_states": hidden_states, "residual": residual})
    else:
        hidden_states, _ = self.norm(hidden_states, residual)
```

代码逻辑：
- PP 首 rank 负责 embedding，非首 rank 依赖上一个 stage 的 proxy tensor。
- 每个 rank 只执行自己的层范围。
- 非末 rank 不做 final norm，只把 hidden/residual 传给下一 stage。

为什么这样写：
- pipeline parallel 要把模型层切成 stage，但上层调用仍希望看起来像一次 forward。
- aux hidden capture 放在层循环内，便于 speculative/EAGLE 类路径复用中间层状态。

不变量与失败模式：
- 非首 rank 必须传入 `pp_proxy_tensors`。
- `layers_to_capture` 中的层必须落在当前 rank 负责范围内，否则该 rank不会捕获对应 hidden。

Comment：
这段是 decoder-only 模型在 PP 下的通用模板，Qwen、Mistral 等模型多沿用同类结构。

### 2.4 LlamaForCausalLM 初始化 lm_head 与 logits processor

问题与约束：
- 有些 Llama checkpoint 共享 input embedding 和 lm_head，有些不共享；同时 DP lm_head 会影响并行组选择。

设计选择：
- 若 `tie_word_embeddings` 为真，直接复用 `embed_tokens`；否则构造 `ParallelLMHead`，并按 `enable_dp_lm_head` 决定是否使用 attention TP group。

Explain：
`LlamaForCausalLM` 在模型本体外再挂 lm_head、logits processor 和 pooler。tie embeddings 场景避免额外 lm_head 参数，不共享场景使用并行 lm_head 适配 TP/DP 布局。

来源：python/sglang/srt/models/llama.py L497-L507

Code：

```python
if self.config.tie_word_embeddings:
    self.lm_head = self.model.embed_tokens
else:
    self.lm_head = ParallelLMHead(
        config.vocab_size,
        config.hidden_size,
        quant_config=quant_config,
        prefix=add_prefix("lm_head", prefix),
        use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
    )
self.logits_processor = LogitsProcessor(config)
```

代码逻辑：
- tied embeddings 直接让 lm_head 指向 embedding module。
- 非 tied 场景创建独立并行头。
- logits processor 统一处理最终 logits 生成。

为什么这样写：
- 是否 tie embeddings 是 checkpoint 语义，不能由运行时自行决定。
- DP lm_head 需要和 attention TP group 对齐，避免并行布局不一致。

不变量与失败模式：
- tied 场景要求 embedding 权重维度与 vocab 输出一致。
- `enable_dp_lm_head` 打开时，attention TP group 必须已正确初始化。

Comment：
lm_head 是模型执行链路和采样链路的连接点。

---

## 3. Llama 权重加载

### 3.1 load_weights 先定义 stacked mapping 并处理 scale 名称

问题与约束：
- HF checkpoint 常把 q/k/v、gate/up 分开存；SGLang 内部将它们堆叠到 fused 参数中。量化 checkpoint 还可能使用旧 scale 名称。

设计选择：
- `load_weights` 内部定义 `stacked_params_mapping`，并在逐项加载前把 `.activation_scale`、`.weight_scale_inv` 重命名到内部参数名。

Explain：
mapping 描述 checkpoint shard 名和 SGLang fused 参数名的对应关系：q/k/v 合并到 `.qkv_proj`，gate/up 合并到 `.gate_up_proj`。scale 名称 remap 先于 layer 过滤和参数查找。

来源：python/sglang/srt/models/llama.py L629-L646

Code：

```python
def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
    stacked_params_mapping = [
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    ]

    params_dict = dict(self.named_parameters())

    for name, loaded_weight in weights:
        if name.endswith(".activation_scale"):
            name = name.replace(".activation_scale", ".input_scale")
        if name.endswith(".weight_scale_inv"):
            name = name.replace(".weight_scale_inv", ".weight_scale")
```

代码逻辑：
- mapping 的第三项是 shard id，传给 fused 参数的 `weight_loader`。
- `params_dict` 是最终参数存在性判断的依据。
- scale 名称先标准化，后续流程只处理内部命名。

为什么这样写：
- fused QKV 和 fused MLP 投影减少 runtime kernel 数，但权重加载必须兼容分开的 checkpoint。
- 量化名称兼容放在入口处，避免每个 loader 重复处理旧命名。

不变量与失败模式：
- checkpoint 名称必须能被 mapping 或 params_dict 识别。
- 若 shard id 与 fused 参数期望不一致，会把权重写入错误切片。

Comment：
Llama 的权重加载展示了“外部 checkpoint 命名”和“内部 fused 参数布局”的适配层。

### 3.2 PP layer 过滤和 stacked weight_loader

问题与约束：
- pipeline parallel 中每个 rank 只拥有部分层；不能把其他 stage 的权重加载到本 rank。

设计选择：
- 先用 `get_layer_id` 过滤当前 PP stage 外的权重，再对 stacked mapping 命中的参数调用参数自带 `weight_loader`。

Explain：
`start_layer/end_layer` 决定当前 rank 的层范围。命中 q/k/v 或 gate/up 的权重先替换名称，再检查参数是否存在，最后调用 fused 参数的 loader 并传入 shard id。

来源：python/sglang/srt/models/llama.py L647-L684

Code：

```python
layer_id = get_layer_id(name)
if (
    layer_id is not None
    and hasattr(self.model, "start_layer")
    and (
        layer_id < self.model.start_layer
        or layer_id >= self.model.end_layer
    )
):
    continue

for param_name, weight_name, shard_id in stacked_params_mapping:
    if weight_name not in name:
        continue
    name = name.replace(weight_name, param_name)
    if name.endswith(".bias") and name not in params_dict:
        continue
    if name not in params_dict:
        continue
    param = params_dict[name]
    weight_loader = param.weight_loader
    weight_loader(param, loaded_weight, shard_id)
    break
```

代码逻辑：
- stage 外层权重直接跳过。
- stacked 参数名替换后仍需检查是否在本 rank 参数表中。
- fused 参数的 loader 根据 shard id 写入正确子矩阵。

为什么这样写：
- PP stage 过滤减少显存和无效加载，也避免缺失参数报错。
- 参数自带 loader 能把量化、切分和 fused 写入逻辑封装在参数对象上。

不变量与失败模式：
- `get_layer_id` 必须能从权重名解析层号。
- 若 checkpoint 命名与 mapping 不匹配，会落入 fallback 分支或被跳过。

Comment：
PP 过滤与 stacked mapping 是 Llama 权重加载中最容易误判的两层逻辑。

### 3.3 fallback weight_loader 处理普通参数与额外 bias

问题与约束：
- checkpoint 中除了 stacked 权重，还有 norm、embedding、lm_head 等普通参数；GPTQ 等量化格式还可能带 SGLang 不需要的 bias 或旧 kv scale。

设计选择：
- stacked mapping 未命中时，跳过不存在的额外 bias/kv_scale；存在于 params_dict 的参数则使用自身 loader 或默认 loader。

Explain：
fallback 分支只加载当前模型参数表中存在的名称。参数对象如果定义了 `weight_loader` 就用自定义 loader，否则使用 `default_weight_loader`。

来源：python/sglang/srt/models/llama.py L686-L698

Code：

```python
else:
    if name.endswith(".bias") and name not in params_dict:
        continue
    if name.endswith(".kv_scale") and name not in params_dict:
        continue
    if name in params_dict.keys():
        param = params_dict[name]
        weight_loader = getattr(
            param, "weight_loader", default_weight_loader
        )
        weight_loader(param, loaded_weight)
```

代码逻辑：
- stacked loop 的 `else` 表示没有任何 mapping 命中。
- extra bias 和旧 kv scale 如果模型没有对应参数就跳过。
- 普通参数按名称直接加载。

为什么这样写：
- 兼容多种 checkpoint 格式时，跳过已知冗余参数比硬失败更稳。
- 自定义 loader 和默认 loader 共存，保证普通参数路径简单。

不变量与失败模式：
- 真正缺失的必需参数不会在这里立即报错，需要依赖后续参数完整性检查。
- 如果 checkpoint 参数名误拼成未知名称，可能被静默跳过。

Comment：
fallback 分支承担的是宽容兼容，而不是权重完整性验证。

---

## 4. Qwen3：通用骨架上的特化

### 4.1 Qwen3Attention 使用 attention TP 并加入 QK-Norm

问题与约束：
- DP-Attention 场景下 attention 的 TP rank/size 可能与 MLP TP 不同；Qwen3 还需要对 q/k 做 RMSNorm。

设计选择：
- `Qwen3Attention` 使用 `attn_tp_rank/attn_tp_size` 构造 QKV 和 O projection，并在初始化中创建 `q_norm/k_norm`。

Explain：
Qwen3 的 head 切分逻辑与 Llama 类似，但使用 attention TP 组。`QKVParallelLinear` 和 `RowParallelLinear` 都显式传入 attention TP rank/size；输出投影设置 `reduce_results=False`，后续交给 layer communicator 管理通信。

来源：python/sglang/srt/models/qwen3.py L87-L141

Code：

```python
attn_tp_rank = get_parallel().attn_tp_rank
attn_tp_size = get_parallel().attn_tp_size

assert self.total_num_heads % attn_tp_size == 0
self.num_heads = self.total_num_heads // attn_tp_size
self.total_num_kv_heads = num_kv_heads
if self.total_num_kv_heads >= attn_tp_size:
    assert self.total_num_kv_heads % attn_tp_size == 0
else:
    assert attn_tp_size % self.total_num_kv_heads == 0
self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)

self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)
self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)
self.qkv_proj = QKVParallelLinear(
    hidden_size,
    self.head_dim,
    self.total_num_heads,
    self.total_num_kv_heads,
    tp_rank=attn_tp_rank,
    tp_size=attn_tp_size,
    prefix=add_prefix("qkv_proj", prefix),
)
self.o_proj = RowParallelLinear(
    self.total_num_heads * self.head_dim,
    hidden_size,
    tp_rank=attn_tp_rank,
    tp_size=attn_tp_size,
    reduce_results=False,
    prefix=add_prefix("o_proj", prefix),
)
```

代码逻辑：
- head 和 KV head 都按 attention TP size 计算。
- QK-Norm 的 dtype 行为可被 RL on-policy 参数调整。
- 输出投影不在这里 reduce，给通信层留下处理空间。

为什么这样写：
- attention TP 与 MLP TP 解耦后，Qwen3 attention 必须显式使用 attention 并行组。
- QK-Norm 是 Qwen3 模型语义，必须在 RoPE/attention 前完成。

不变量与失败模式：
- `total_num_heads` 必须能被 `attn_tp_size` 整除。
- KV head 与 attention TP 的切分/复制关系必须满足 assert。

Comment：
Qwen3Attention 是 Llama attention 模板在 attention TP 与 QK-Norm 上的特化。

### 4.2 Qwen3 native prepare 先 QK-Norm 再 RoPE

问题与约束：
- Qwen3 需要在 q/k 上应用 RMSNorm；若顺序错误，会改变 RoPE 后的向量语义。

设计选择：
- native prepare 在 QKV split 后调用 `apply_qk_norm`，再执行 `self.rotary_emb`。

Explain：
初始化阶段还会检测是否可用 fused qk norm mRoPE，并把 fused kernel 需要的 scale tensor 固定在 CPU，避免 CUDA graph capture 中出现 D2H 同步。

来源：python/sglang/srt/models/qwen3.py L160-L185

Code：

```python
self.use_fused_qk_norm_mrope = (
    _has_fused_qk_norm_mrope
    and isinstance(self.rotary_emb, MRotaryEmbedding)
    and getattr(self.rotary_emb, "mrope_section", None) is not None
)
if self.use_fused_qk_norm_mrope:
    self._fused_k_scale = torch.tensor(1.0, dtype=torch.float32, device="cpu")
    self._fused_v_scale = torch.tensor(1.0, dtype=torch.float32, device="cpu")

def forward_prepare_native(self, positions, hidden_states):
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k = apply_qk_norm(
        q=q,
        k=k,
        q_norm=self.q_norm,
        k_norm=self.k_norm,
        head_dim=self.head_dim,
        alt_stream=self.alt_stream,
    )
    q, k = self.rotary_emb(positions, q, k)
    return q, k, v
```

代码逻辑：
- fused mRoPE 能力由 kernel 存在、RoPE 类型和 mRoPE section 三个条件共同决定。
- native path 的顺序是 projection → split → QK-Norm → RoPE。
- `alt_stream` 可用于 QK-Norm 的并行执行优化。

为什么这样写：
- scale tensor 放在 CPU 是为了避免 C++ kernel `.item<float>()` 引发 CUDA graph capture 同步问题。
- QK-Norm 放在 RoPE 前保持模型定义一致。

不变量与失败模式：
- fused mRoPE 只适用于 `MRotaryEmbedding` 且存在 `mrope_section` 的配置。
- 如果 QK-Norm 和 RoPE 顺序被调换，输出数值会偏离 Qwen3 训练语义。

Comment：
Qwen3 的 attention prepare 比 Llama 多了 QK-Norm 和 mRoPE 能力检测。

### 4.3 Qwen3Attention.forward 在 fused、native、NPU 三路间选择

问题与约束：
- fused qk norm mRoPE 只适合 decode 且非 RL on-policy；NPU、native 和 RL 路径都要保持正确 dtype 与 KV cache 行为。

设计选择：
- 默认 `save_kv_cache=True`；fused decode 路径调用 `forward_prepare_aiter_fused_mrope` 后设为 False，避免重复写 cache。

Explain：
RL on-policy 会把 hidden/q/k 转成 bf16，并禁用 fused 路径。非 fused 场景按 NPU 与非 NPU 分支选择 prepare。最终调用 `self.attn(..., save_kv_cache=save_kv_cache)`。

来源：python/sglang/srt/models/qwen3.py L269-L308

Code：

```python
def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
) -> torch.Tensor:
    if get_global_server_args().rl_on_policy_target is not None:
        hidden_states = hidden_states.bfloat16()

    save_kv_cache = True
    use_aiter_fused = (
        self.use_fused_qk_norm_mrope
        and forward_batch.forward_mode.is_decode()
        and get_global_server_args().rl_on_policy_target is None
    )

    if use_aiter_fused:
        q, k, v = self.forward_prepare_aiter_fused_mrope(
            positions, hidden_states, forward_batch
        )
        save_kv_cache = False
    elif not _is_npu:
        q, k, v = self.forward_prepare_native(...)
    else:
        q, k, v = self.forward_prepare_npu(...)

    if get_global_server_args().rl_on_policy_target is not None:
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)

    attn_output = self.attn(q, k, v, forward_batch, save_kv_cache=save_kv_cache)
    output, _ = self.o_proj(attn_output)
    return output
```

代码逻辑：
- RL on-policy 先调整 hidden dtype，并在 attention 前再调整 q/k。
- fused path 只在 decode 模式启用。
- fused path 已写 KV cache，因此关闭 `save_kv_cache`。

为什么这样写：
- fused decode 追求减少 kernel 和 cache 写入开销，但 prefill/extend 和 RL 训练目标需要更保守的路径。
- `save_kv_cache` 明确表达 fused prepare 与 RadixAttention 的责任边界。

不变量与失败模式：
- fused prepare 若已经写 cache却没有关闭 `save_kv_cache`，会出现重复写 cache。
- RL on-policy 目标存在时使用 fused kernel 会破坏训练/采样所需 dtype 控制。

Comment：
Qwen3 forward 的核心是“同一 attention 接口，多条 QKV prepare 路径”。

### 4.4 Qwen3DecoderLayer 用 LayerCommunicator 管理 attention/MLP 边界

问题与约束：
- attention TP、MLP TP、NPU piecewise graph 和层间 scatter/reduce 会影响每层输入输出布局；直接在 layer forward 中手写通信会很难维护。

设计选择：
- 初始化 `LayerScatterModes` 和 `LayerCommunicator`，forward 中通过 communicator 的 `prepare_attn/prepare_mlp/postprocess_layer` 管理布局。

Explain：
Qwen3 普通 dense layer 将 `is_layer_sparse/is_previous_layer_sparse/is_next_layer_sparse` 都设为 False。forward 先让 communicator 准备 attention，再调用 self-attention；随后准备 MLP，运行 MLP，并在最后 postprocess。

来源：python/sglang/srt/models/qwen3.py L376-L433

Code：

```python
self.layer_scatter_modes = LayerScatterModes.init_new(
    layer_id=layer_id,
    num_layers=config.num_hidden_layers,
    is_layer_sparse=False,
    is_previous_layer_sparse=False,
    is_next_layer_sparse=False,
)
self.layer_communicator = LayerCommunicator(
    layer_scatter_modes=self.layer_scatter_modes,
    input_layernorm=self.input_layernorm,
    post_attention_layernorm=self.post_attention_layernorm,
)

def forward(...):
    hidden_states, residual = self.layer_communicator.prepare_attn(
        hidden_states,
        residual,
        forward_batch,
        post_residual_addition=post_residual_addition,
    )
    if hidden_states.shape[0] != 0:
        hidden_states = self.self_attn(...)
    hidden_states, residual = self.layer_communicator.prepare_mlp(...)
    hidden_states = self.mlp(hidden_states, forward_batch=forward_batch)
    hidden_states, residual = self.layer_communicator.postprocess_layer(
        hidden_states, residual, forward_batch
    )
    return hidden_states, residual
```

代码逻辑：
- scatter mode 描述当前层及邻层是否 sparse。
- communicator 在 attention 和 MLP 前处理 layernorm、residual 和布局。
- layer forward 自身只保留 attention、MLP 和后处理调用顺序。

为什么这样写：
- 通信/布局是跨层模式，不应散落在每个模型的 forward 细节里。
- dense Qwen3 与 MoE/Qwen3Moe 可以共享 communicator 抽象，只改变 scatter mode。

不变量与失败模式：
- communicator 的 scatter mode 必须与实际层类型一致。
- `hidden_states.shape[0] == 0` 时必须跳过 attention，避免空 batch kernel 问题。

Comment：
LayerCommunicator 是 Qwen3 这类模型处理 attention TP 与层间布局的关键抽象。

### 4.5 Qwen3ForCausalLM.forward 对齐 Llama 的最后一跳

问题与约束：
- PP 场景下非末 rank 不能做 logits；embedding/pooling 请求又需要绕过 logits processor。

设计选择：
- Qwen3ForCausalLM 先调用 `self.model`，再在末 rank 根据 `get_embedding` 选择 logits processor 或 pooler；非末 rank 直接返回 hidden/proxy。

Explain：
如果启用 aux hidden capture，`self.model` 返回 `(hidden_states, aux_hidden_states)`，随后 logits processor 同时接收 aux hidden。非末 rank 不访问 lm_head。

来源：python/sglang/srt/models/qwen3.py L512-L546

Code：

```python
@torch.no_grad()
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    input_embeds: torch.Tensor = None,
    get_embedding: bool = False,
    pp_proxy_tensors: Optional[PPProxyTensors] = None,
) -> torch.Tensor:
    hidden_states = self.model(
        input_ids,
        positions,
        forward_batch,
        input_embeds,
        pp_proxy_tensors=pp_proxy_tensors,
    )

    aux_hidden_states = None
    if self.capture_aux_hidden_states:
        hidden_states, aux_hidden_states = hidden_states

    if self.pp_group.is_last_rank:
        if not get_embedding:
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
                aux_hidden_states,
            )
        else:
            return self.pooler(hidden_states, forward_batch)
    else:
        return hidden_states
```

代码逻辑：
- 模型主体先处理 PP、层循环和 norm。
- aux hidden 只在配置开启时拆包。
- 末 rank 负责 logits 或 embedding pooling；非末 rank 继续传递 hidden。

为什么这样写：
- CausalLM wrapper 负责输出语义，不负责层内部执行。
- PP 下只有末 rank 拥有完整最终 hidden，提前 logits 会破坏并行边界。

不变量与失败模式：
- `capture_aux_hidden_states=True` 时，底层 model 必须返回二元组。
- `get_embedding=True` 时调用 pooler，不能进入 logits processor。

Comment：
Qwen3 和 Llama 的 CausalLM wrapper 结构基本一致，差异主要在模型内部 attention 与 layer 通信。
