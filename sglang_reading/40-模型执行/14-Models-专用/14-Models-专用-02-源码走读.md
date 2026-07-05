---
type: batch-doc
module: 14-Models-专用
batch: "14"
doc_type: walkthrough
title: "Models 专用 · 源码走读"
tags:
 - sglang/batch/14
 - sglang/module/models-specialized
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-05
---
# Models 专用 · 源码走读

> 本篇以 `deepseek_v2.py` 为样本，读 SGLang 如何把 DeepSeek 系模型的 MLA、MoE、DSA、context parallel、TBO 和 expert load balancing 接进统一模型执行框架。

---

## 1. Dense MLP：先处理极端 batch 形态

### 1.1 DeepseekV2MLP.forward 的空输入短路

**问题与约束：** 推理调度里可能出现空 batch 或某个 rank 没有 token 的情况；如果仍进入 TP collective 或 GEMM，轻则浪费，重则在通信路径上挂住。

**设计选择：** Dense MLP 在 `tp_size == 1` 且 token 数为 0 时直接返回输入；其余路径继续走融合 GEMM、量化或普通 MLP 逻辑。

**Explain：** 这是 dense 层的最小防护：没有 token 就不进入后续算子。

来源：python/sglang/srt/models/deepseek_v2.py L285-L294

**Code：**

```python
def forward(
    self,
    x,
    forward_batch=None,
    should_allreduce_fusion: bool = False,
    use_reduce_scatter: bool = False,
    gemm_output_zero_allocator: BumpAllocator = None,
):
    if (self.tp_size == 1) and x.shape[0] == 0:
        return x
```

**代码逻辑：** 函数入口先检查单 TP 且空 token；满足时直接返回 `x`，避免进入后面的 fused / quantized MLP 分支。

**为什么这样写：** DeepSeek 专用模型里很多路径会根据 batch 形态、量化格式和 overlap 策略选择不同算子；空输入越早短路，后续分支越少承担异常形态。

**不变量与失败模式：** 只在 `tp_size == 1` 短路；多 TP 场景仍要保持各 rank collective 语义一致，不能随意让某个 rank 提前返回。

**Comment：** 这段虽小，但能帮助理解后面大量“空 batch / skip collective”保护为什么存在。

---

## 2. DeepseekV2MoE：routing 与专家实现解耦

### 2.1 Hash MoE 层判定

**问题与约束：** DeepSeek 系模型有些前置层可用 hash routing，不走 learned gate；但 DeepSeek V4 的 NextN 层不应套用同一逻辑。

**设计选择：** 通过 `num_hash_layers` 和 `layer_id` 判断当前层是否 hash MoE，并排除 `is_deepseek_v4 and is_nextn`。

**Explain：** `self.is_hash` 是后续 gate/topk 构造分支的开关。

来源：python/sglang/srt/models/deepseek_v2.py L580-L581

**Code：**

```python
n_hash_layers = getattr(config, "num_hash_layers", 0)
self.is_hash = layer_id < n_hash_layers and not (is_deepseek_v4 and is_nextn)
```

**代码逻辑：** 配置缺省时 hash 层数为 0；只有层号落在 hash 前缀范围且不是 V4 NextN 时，才把该层标为 hash routing。

**为什么这样写：** routing 方式属于模型结构约束，必须在层初始化时确定；如果放到 forward 时判断，会让 MoE runner 和权重布局更难稳定。

**不变量与失败模式：** `layer_id` 必须可靠；错误标记 hash 层会导致 top-k 选择方法与权重语义不匹配。

**Comment：** 这段是读 DeepSeek MoE 前先要记住的第一层分岔。

### 2.2 Gate 和 Experts 构造

**问题与约束：** MoE 层既要支持普通 routed experts，也要支持 fused shared experts、DeepEP、量化后端和冗余 expert；这些能力不应该写死在模型层里。

**设计选择：** 模型层构造 `MoEGate` 负责 routing logits / correction bias，再通过 `get_moe_impl_class(quant_config)` 选择具体 experts 实现。

**Explain：** `DeepseekV2MoE` 负责组织配置和张量流，真正 expert 执行交给 MoE 后端。

来源：python/sglang/srt/models/deepseek_v2.py L595-L635

**Code：**

```python
self.gate = MoEGate(
    config=config,
    quant_config=quant_config,
    prefix=add_prefix("gate", prefix),
    is_nextn=is_nextn,
    is_hash_moe=self.is_hash,
    is_deepseek_v4=is_deepseek_v4,
    dsa_enable_prefill_cp=dsa_enable_prefill_cp,
    mla_enable_prefill_cp=mla_enable_prefill_cp,
)

fused_shared_experts_scaling_factor = None
if (
    self.moe_ep_size > 1
    and self.num_fused_shared_experts > 0
    and not _is_deepep_fusion
):
    fused_shared_experts_scaling_factor = 1.0 / float(self.moe_ep_size)

self.experts = get_moe_impl_class(quant_config)(
    num_experts=num_experts_for_moe
    + get_global_server_args().ep_num_redundant_experts,
    num_fused_shared_experts=self.num_fused_shared_experts,
    top_k=top_k_for_moe,
    hidden_size=config.hidden_size,
    intermediate_size=config.moe_intermediate_size,
    layer_id=self.layer_id,
    quant_config=quant_config,
    routed_scaling_factor=self.routed_scaling_factor,
)
```

**代码逻辑：** gate 接收模型配置和 CP 标志；非 DeepEP fusion 的 EP shared expert 路径设置缩放因子；experts 数量把 routed、fused shared 和 redundant experts 一起纳入。

**为什么这样写：** MoE 执行后端会随量化和硬件变化，模型结构层不应感知所有 kernel 细节；通过工厂函数选择实现能保持模型 forward 稳定。

**不变量与失败模式：** `top_k_for_moe`、expert 数和实际权重布局必须一致；EP shared expert 缩放遗漏会造成 shared expert 贡献被重复计入。

**Comment：** 这里的重点是“routing 决策”和“expert 执行”分层。

### 2.3 HashTopK 替代 learned top-k

**问题与约束：** hash routing 层不应使用 learned gate 的 grouped top-k；但它仍要产生与 experts 后端兼容的 top-k expert 索引和缩放语义。

**设计选择：** `self.is_hash` 为真时构造 `HashTopK`，把 vocab size、专家数、fused shared experts 和 routed scaling factor 都传入。

**Explain：** HashTopK 把 token 到 expert 的选择固定成 hash 规则，但输出接口仍像普通 top-k。

来源：python/sglang/srt/models/deepseek_v2.py L637-L647

**Code：**

```python
if self.is_hash and not (is_nextn and is_deepseek_v4):
    self.topk = HashTopK(
        topk=config.num_experts_per_tok + self.num_fused_shared_experts,
        num_experts=config.n_routed_experts,
        num_fused_shared_experts=self.num_fused_shared_experts,
        vocab_size=config.vocab_size,
        scoring_func=config.scoring_func,
        routed_scaling_factor=self.routed_scaling_factor,
        apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,
        layer_id=self.layer_id,
    )
```

**代码逻辑：** top-k 数量包含 routed experts 和 fused shared experts；HashTopK 还知道是否把 routed scaling factor 融进 top-k 输出。

**为什么这样写：** experts 后端希望看到统一的 top-k 输入；hash 层不改变后端接口，只替换 top-k 生成方式。

**不变量与失败模式：** hash top-k 的 expert 编号空间必须与 `self.experts` 的 expert 数一致；V4 NextN 排除条件必须和 `self.is_hash` 判定保持一致。

**Comment：** Hash routing 是“换 top-k 算法”，不是“换 MoE 层输出协议”。

### 2.4 MoE forward 的后端分流

**问题与约束：** 同一 MoE 层可能走 MegaMoE、FlashInfer dual stream、normal dual stream、normal 或 DeepEP；这些路径依赖后端、capture mode、shared expert、A2A 等状态。

**设计选择：** `forward` 只做路径选择：先尝试 MegaMoE；非 A2A 时按 dual-stream 条件选择；A2A 时走 `forward_deepep`。

**Explain：** MoE forward 的复杂性主要来自“哪条执行后端适合当前 batch”，而不是模型层自己实现 expert GEMM。

来源：python/sglang/srt/models/deepseek_v2.py L853-L916

**Code：**

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    forward_batch: Optional[ForwardBatch] = None,
    should_allreduce_fusion: bool = False,
    use_reduce_scatter: bool = False,
    gemm_output_zero_allocator: BumpAllocator = None,
    input_ids: Optional[torch.Tensor] = None,
    input_ids_global: Optional[torch.Tensor] = None,
    skip_shared_experts: bool = False,
) -> torch.Tensor:
    from sglang.srt.layers.moe.mega_moe import forward_mega_moe, should_use_mega_moe

    if should_use_mega_moe(self, hidden_states):
        return forward_mega_moe(
            self,
            hidden_states,
            forward_batch,
            input_ids_global=input_ids_global,
        )

    if not self._enable_a2a_moe:
        server_args = get_global_server_args()
        if self._can_dual_stream_graph(hidden_states, server_args):
            return dsv2_flashinfer_moe_dual_stream_graph(...)
        elif self.alt_stream is not None and self.num_fused_shared_experts == 0:
            return self.forward_normal_dual_stream(...)
        else:
            return self.forward_normal(...)
    else:
        return self.forward_deepep(
            hidden_states, forward_batch, input_ids_global=input_ids_global
        )
```

**代码逻辑：** MegaMoE 优先；A2A 未启用时根据 graph / alt stream 条件走不同 normal 分支；A2A 启用时把 dispatch/combine 交给 DeepEP 路径。

**为什么这样写：** MoE kernel 性能对 batch shape 和后端条件非常敏感；把选择集中在 forward 入口，便于后端扩展而不改 DecoderLayer。

**不变量与失败模式：** 每条分支都必须返回与 `hidden_states` 对齐的输出 tensor；如果分支条件漏掉 shared expert 或 capture mode 约束，可能走到不支持的 kernel。

**Comment：** 读 MoE 性能问题时，先定位这一步实际选了哪个 forward 分支。

---

## 3. DeepseekV2AttentionMLA：同一 attention 暴露多后端

### 3.1 多 Mixin 组合

**问题与约束：** DeepSeek MLA 在不同平台和配置下可能走 CUDA absorb、ROCm、CPU 或普通 MHA fallback；但 DecoderLayer 不应该知道这些后端细节。

**设计选择：** `DeepseekV2AttentionMLA` 继承 `nn.Module` 和多个 forward mixin，把多后端实现组合进同一个 attention 类。

**Explain：** 模型层看到的是一个 `self_attn`，内部根据状态选择 MLA / MHA / ROCm / CPU 路径。

来源：python/sglang/srt/models/deepseek_v2.py L1541-L1547

**Code：**

```python
class DeepseekV2AttentionMLA(
    nn.Module,
    DeepseekMHAForwardMixin,
    DeepseekMLAForwardMixin,
    DeepseekMLARocmForwardMixin,
    DeepseekMLACpuForwardMixin,
):
```

**代码逻辑：** 一个类同时混入 MHA fallback、MLA 主路径、ROCm 和 CPU forward 实现；实例字段在初始化中决定实际路径所需参数。

**为什么这样写：** DeepSeek attention 的专用优化很多，使用 mixin 可以把后端代码拆开，但保留统一调用对象。

**不变量与失败模式：** 多个 mixin 不能暴露互相冲突的方法解析顺序；若某个 backend 需要的字段未初始化，会在 forward_prepare 或 forward_core 才暴露。

**Comment：** 这类继承结构要结合 `forward_prepare` / `forward_core` 读，单看类声明只能知道“有哪些路径”。

### 3.2 DSA 与 context parallel 标志

**问题与约束：** DSA、DSA prefill context parallel、MLA prefill context parallel 是彼此相关但不完全相同的开关；错误组合会导致 attention cache 或通信器行为不一致。

**设计选择：** 初始化时记录 `use_dsa`、`dsa_enable_prefill_cp`、`mla_enable_prefill_cp`；DSA CP 开启时断言模型确实是 DeepSeek DSA。

**Explain：** attention 层先把模型能力和运行时 CP 配置固化成字段，后续 communicator 和 forward 路径复用这些字段。

来源：python/sglang/srt/models/deepseek_v2.py L1585-L1589

**Code：**

```python
self.use_dsa = is_deepseek_dsa(config)
self.dsa_enable_prefill_cp = dsa_enable_prefill_cp
self.mla_enable_prefill_cp = mla_enable_prefill_cp
if self.dsa_enable_prefill_cp:
    assert self.use_dsa, "CP currently only supports deepseek v3.2 model"
```

**代码逻辑：** `is_deepseek_dsa(config)` 判定模型类型；两个 CP 开关独立保存；DSA CP 只允许在 DSA 模型上启用。

**为什么这样写：** CP 会改变 QKV latent 和 attention 输入切分方式，必须在构造时就拒绝不支持的组合。

**不变量与失败模式：** `dsa_enable_prefill_cp => use_dsa`；如果非 DSA 模型误开 DSA CP，assert 会阻止继续初始化。

**Comment：** MLA CP 和 DSA CP 在后面的 communicator 中都会走专门路径。

### 3.3 forward：prepare / core 两阶段

**问题与约束：** Attention 前处理要处理 positions、hidden states、forward batch、临时 allocator、scatter mode、DSA topk 复用等；真正 kernel 执行又依赖前处理产物。

**设计选择：** `forward` 只调用 `forward_prepare(...)` 得到状态对象，再把状态交给 `forward_core(s)`。

**Explain：** 这是典型的“准备运行态状态，再执行后端核心”的结构。

来源：python/sglang/srt/models/deepseek_v2.py L1836-L1855

**Code：**

```python
def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    zero_allocator: BumpAllocator,
    layer_scatter_modes: LayerScatterModes = None,
    llama_4_scaling: Optional[torch.Tensor] = None,
    prev_topk_indices: Optional[torch.Tensor] = None,
):
    s = self.forward_prepare(
        positions=positions,
        hidden_states=hidden_states,
        forward_batch=forward_batch,
        zero_allocator=zero_allocator,
        layer_scatter_modes=layer_scatter_modes,
        llama_4_scaling=llama_4_scaling,
        prev_topk_indices=prev_topk_indices,
    )
    return self.forward_core(s)
```

**代码逻辑：** 所有输入先进入 `forward_prepare`；prepare 返回的状态 `s` 再交给 core 执行。

**为什么这样写：** 多后端 attention 共享大量输入整理和临时 buffer 逻辑；拆成两阶段能让 TBO、CP、DSA 等路径复用 prepare 结果。

**不变量与失败模式：** `forward_prepare` 返回值必须满足 `forward_core` 的状态协议；如果 prepare 早退返回 tensor，core 路径也要能识别。

**Comment：** 读 MLA 细节时先追 `forward_prepare`，再看当前 backend 的 `forward_core`。

### 3.4 MHA fallback 复用 kv_b_proj

**问题与约束：** MLA 和 MHA fallback 都可能需要 `kv_b_proj`；如果复制权重引用，容易出现加载、量化或更新时两边不同步。

**设计选择：** 在 `forward_prepare` 里懒设置 `self.attn_mha.kv_b_proj = self.kv_b_proj`。

**Explain：** fallback 路径和 MLA 主路径共享同一份投影权重。

来源：python/sglang/srt/models/deepseek_v2.py L1867-L1868

**Code：**

```python
if self.attn_mha.kv_b_proj is None:
    self.attn_mha.kv_b_proj = self.kv_b_proj
```

**代码逻辑：** 第一次 forward_prepare 时，如果 MHA fallback 尚未绑定 `kv_b_proj`，就指向 MLA 对象上的 `kv_b_proj`。

**为什么这样写：** 权重加载和量化状态由主模块维护，fallback 只借用引用，减少状态复制。

**不变量与失败模式：** `self.kv_b_proj` 必须在进入 forward_prepare 前已初始化；如果 fallback 持有旧引用，权重更新后会产生不一致结果。

**Comment：** 这是一种“延迟绑定共享权重”的写法。

---

## 4. DeepseekV2DecoderLayer：把 attention、通信和 MLP 串起来

### 4.1 CP communicator 与普通 communicator

**问题与约束：** 开启 DSA/MLA prefill context parallel 时，层内 attention/MLP 前后的 scatter、layernorm 和 reduce-scatter 逻辑不同于普通路径。

**设计选择：** 构造层时按 CP 开关选择 `DSACPLayerCommunicator` 或 `LayerCommunicator`，两者接收同样的 layernorm、scatter mode、last-layer 和 QKV latent 函数。

**Explain：** DecoderLayer 不直接写通信细节，而是把通信策略封装进 communicator。

来源：python/sglang/srt/models/deepseek_v2.py L2136-L2160

**Code：**

```python
if self.dsa_enable_prefill_cp or self.mla_enable_prefill_cp:
    self.layer_communicator = DSACPLayerCommunicator(
        layer_scatter_modes=self.layer_scatter_modes,
        input_layernorm=self.input_layernorm,
        post_attention_layernorm=self.post_attention_layernorm,
        allow_reduce_scatter=True,
        is_last_layer=(
            is_nextn or (self.layer_id == self.config.num_hidden_layers - 1)
        ),
        qkv_latent_func=self.self_attn.prepare_qkv_latent,
    )
else:
    self.layer_communicator = LayerCommunicator(
        layer_scatter_modes=self.layer_scatter_modes,
        input_layernorm=self.input_layernorm,
        post_attention_layernorm=self.post_attention_layernorm,
        allow_reduce_scatter=True,
        is_last_layer=(
            is_nextn or (self.layer_id == self.config.num_hidden_layers - 1)
        ),
        qkv_latent_func=self.self_attn.prepare_qkv_latent,
    )
```

**代码逻辑：** 两个 communicator 使用相同参数表；只有类不同，表示内部处理 CP gating 和普通通信的差异。

**为什么这样写：** 让 DecoderLayer 的 forward 主链保持稳定：prepare attention、run attention、prepare MLP、run MLP、postprocess。

**不变量与失败模式：** CP 开关和 attention 层初始化必须一致；否则 communicator 会按 CP 方式切分，而 attention 端没有相应 latent / cache 语义。

**Comment：** `LayerCommunicator` 是连接模型层和并行通信策略的关键抽象。

### 4.2 forward 主链：Attention 到 MLP

**问题与约束：** 一个 decoder layer 不只是 `attn + mlp`；它还要处理 residual、layernorm、DSA topk 传递、attention TP 上下文清理、allreduce fusion 和 MoE output buffer。

**设计选择：** forward 通过 communicator 准备 attention 和 MLP，attention 返回可选 topk，MLP 前决定 reduce / allreduce 策略，MoE 输出 buffer 用上下文管理器包住。

**Explain：** DecoderLayer 是 DeepSeek 专用优化汇合点。

来源：python/sglang/srt/models/deepseek_v2.py L2195-L2270

**Code：**

```python
hidden_states_orig = hidden_states
hidden_states, residual = (
    self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
        hidden_states,
        residual,
        forward_batch,
        captured_last_layer_outputs=captured_last_layer_outputs,
        quant_format=getattr(self, "_gfx95_quant_format", ""),
    )
)

hidden_states = self.self_attn(
    positions=positions,
    hidden_states=hidden_states,
    forward_batch=forward_batch,
    zero_allocator=zero_allocator,
    llama_4_scaling=llama_4_scaling,
    layer_scatter_modes=self.layer_scatter_modes,
    prev_topk_indices=prev_topk_indices,
)
if isinstance(hidden_states, tuple):
    hidden_states, topk_indices = hidden_states
else:
    topk_indices = None
get_attn_tp_context().clear_attn_inputs()

hidden_states, residual = self.layer_communicator.prepare_mlp(
    hidden_states, residual, forward_batch
)

should_allreduce_fusion = (
    self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
        forward_batch
    )
)
use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
    forward_batch
)

with _mlp_ctx:
    hidden_states = self.mlp(
        hidden_states,
        forward_batch,
        should_allreduce_fusion,
        use_reduce_scatter,
        gemm_output_zero_allocator,
    )

if not should_allreduce_fusion:
    hidden_states, residual = self.layer_communicator.postprocess_layer(
        hidden_states, residual, forward_batch
    )

return hidden_states, residual, topk_indices
```

**代码逻辑：** attention 前由 communicator 做 norm/scatter 和 last-layer capture；attention 后清理 TP 上下文；MLP 前决定通信融合策略；MLP 后在未融合 allreduce 时立即 postprocess；最后把 DSA topk 传给下一层。

**为什么这样写：** 多种优化都跨 attention 和 MLP 边界，单纯在 attention 或 MoE 内部处理不了；DecoderLayer 必须显式协调这些阶段。

**不变量与失败模式：** `topk_indices` 的跨层传递必须与 DSA 层跳过 top-k 的策略一致；忘记 `clear_attn_inputs()` 会污染下一层 attention TP 上下文。

**Comment：** 这段是本文件最值得逐行跟踪的主链。

---

## 5. DeepseekV2Model / ForCausalLM：模型级运行态

### 5.1 DeepseekV2Model.forward 与 TBO 包装

**问题与约束：** Two-Batch Overlap 只适合部分层段；前面的 dense replacement 层可能仍要按普通循环执行，剩余层再进入 TBO 包装。

**设计选择：** `forward` 先计算 `normal_start_layer / normal_end_layer`，普通循环跑到切分点；若还有剩余层，则调用 `model_forward_maybe_tbo` 包装剩余层。

**Explain：** TBO 不是整个模型无条件开启，而是在层循环中按可运行区间接入。

来源：python/sglang/srt/models/deepseek_v2.py L2550-L2598

**Code：**

```python
normal_start_layer = self.start_layer
normal_end_layer = self.end_layer
if forward_batch.can_run_tbo:
    if (
        self.first_k_dense_replace > normal_start_layer
        and self.first_k_dense_replace < normal_end_layer
    ):
        normal_end_layer = self.first_k_dense_replace
    elif self.first_k_dense_replace < normal_start_layer:
        normal_end_layer = normal_start_layer = 0

for i in range(normal_start_layer, normal_end_layer):
    layer = self.layers[i]
    hidden_states, residual, topk_indices = layer(
        positions,
        hidden_states,
        forward_batch,
        residual,
        zero_allocator,
        gemm_output_zero_allocator,
        llama_4_scaling,
        prev_topk_indices=topk_indices,
    )

if normal_end_layer != self.end_layer:
    hidden_states, residual = model_forward_maybe_tbo(
        layers=self.layers[normal_end_layer : self.end_layer],
        enable_tbo=True,
        positions=positions,
        forward_batch=forward_batch,
        hidden_states=hidden_states,
        residual=residual,
        input_data_scatter_mode=self.layers[
            normal_end_layer - 1
        ].layer_scatter_modes.layer_output_mode,
        zero_allocator=zero_allocator,
    )
```

**代码逻辑：** TBO 可用时先根据 `first_k_dense_replace` 切出普通执行段；普通段逐层传递 `topk_indices`；剩余层交给 TBO 包装，并传入前一层输出 scatter mode。

**为什么这样写：** TBO 需要满足 batch 和层段条件；把普通层和 TBO 层分开，能避免不适合 overlap 的 dense replacement 层误入 TBO。

**不变量与失败模式：** `normal_end_layer - 1` 必须有有效层用于取 scatter mode；如果切分边界计算错误，TBO 包装会收到不匹配的输入布局。

**Comment：** 旧笔记只看 import 不够，真正的模型级 TBO 入口在这个层循环之后。

### 5.2 attention TP context 初始化

**问题与约束：** Attention TP 上下文需要知道 Q LoRA rank 和模型是否 DSA；这些信息影响输入 scatter / all-gather 等 attention 并行行为。

**设计选择：** `DeepseekV2ForCausalLM` 初始化时取 `config.q_lora_rank`，并调用 `get_attn_tp_context().init_context(...)`。

**Explain：** 这是模型级别把配置注入 attention TP 全局上下文的地方。

来源：python/sglang/srt/models/deepseek_v2.py L2723-L2724

**Code：**

```python
q_lora_rank = config.q_lora_rank if hasattr(config, "q_lora_rank") else None
get_attn_tp_context().init_context(q_lora_rank, is_deepseek_dsa(config))
```

**代码逻辑：** 配置有 `q_lora_rank` 就传入，否则传 None；DSA 判定也传入 context 初始化。

**为什么这样写：** attention TP context 被多个层访问，初始化必须在模型构造阶段完成，而不是每层重复计算。

**不变量与失败模式：** context 中的 DSA 标志要和 attention 层的 `use_dsa` 一致；不一致会导致 scatter/collective 策略错配。

**Comment：** 这种全局 context 是专用模型优化的常见入口。

### 5.3 权重加载入口

**问题与约束：** DeepSeek 权重加载要处理 expert 名称 remap、shared expert fusion、NextN 等专用逻辑；顶层模型仍需要暴露统一 `load_weights`。

**设计选择：** `load_weights` 只转调 `do_load_weights(weights, is_nextn)`，具体逻辑放在 weight loader mixin。

**Explain：** ForCausalLM 顶层保持统一接口，专用加载细节下沉到 mixin。

来源：python/sglang/srt/models/deepseek_v2.py L2857-L2858

**Code：**

```python
def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
    self.do_load_weights(weights, is_nextn)
```

**代码逻辑：** 入口参数保持权重迭代器和 NextN 标志；不在这里展开任何 remap 逻辑。

**为什么这样写：** 顶层模型类还要服务通用 loader 调用点；把复杂加载策略封装在 mixin 里，可以复用并减少 forward 类的噪声。

**不变量与失败模式：** `do_load_weights` 必须由继承链提供；如果 mixin 未正确组合，`load_weights` 会在运行时找不到方法。

**Comment：** 想看 expert 权重如何落槽，应继续读 `DeepseekV2WeightLoaderMixin`。

### 5.4 Expert location 配置

**问题与约束：** Expert load balancing 需要知道模型有多少层、每层多少 logical experts、expert group 数；这些信息来自模型 config，而不是运行时样本。

**设计选择：** 提供 `get_model_config_for_expert_location` 类方法，返回 `ModelConfigForExpertLocation`。

**Explain：** 这是模型把 MoE expert 拓扑暴露给 expert location 系统的接口。

来源：python/sglang/srt/models/deepseek_v2.py L2871-L2877

**Code：**

```python
@classmethod
def get_model_config_for_expert_location(cls, config):
    return ModelConfigForExpertLocation(
        num_layers=config.num_hidden_layers,
        num_logical_experts=config.n_routed_experts,
        num_groups=config.n_group,
    )
```

**代码逻辑：** 从 config 中抽取 hidden layer 数、routed expert 数和 group 数，包装成 expert location 模块需要的配置对象。

**为什么这样写：** Expert placement / migration 不应直接解析模型 config 的所有字段；模型类提供一个窄接口即可。

**不变量与失败模式：** `n_routed_experts` 和 `n_group` 必须和 MoE 构造时使用的配置一致；否则 load balancing 的逻辑 expert 视图会和实际 experts 不一致。

**Comment：** 这段把模型结构和运行时 expert 调度连接起来，但不执行迁移本身。

---

## 6. 走读小结

`deepseek_v2.py` 的专用性主要体现在五层：MLP/MoE kernel 分支、MLA 多后端 attention、DecoderLayer 通信器、模型级 TBO 切分、ForCausalLM 的权重与 expert 拓扑接口。读这类专用模型文件，不要把每个分支当成孤立优化；更重要的是看它们如何通过统一的 `ForwardBatch`、communicator、MoE backend 和模型接口接回 SGLang 的通用执行框架。
