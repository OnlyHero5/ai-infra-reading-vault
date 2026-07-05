---
type: batch-doc
module: 18-Model-Init
batch: "18"
doc_type: concept
title: "Model 初始化 · 核心概念"
tags:
  - slime/batch/18
  - slime/module/model-init
  - slime/doc/concept
updated: 2026-07-02
---

# Model 初始化 · 核心概念

---

## 1. model_provider 三分支

**Explain：** `_get_model_provider_func` 按优先级选择：**自定义 path** → **Megatron Bridge** → **legacy Megatron-Core GPTModel**。

**Code：**

```python
## 来源：slime/backends/megatron_utils/model_provider.py L65-L87
# 提交版本：22cdc6e1
if getattr(args, "custom_model_provider_path", None):

    def wrapped_model_provider(...) -> GPTModel:
        custom_model_provider = load_function(args.custom_model_provider_path)
        # ...
        if post_process and role == "critic":
            model.output_layer = LinearForLastLayer(...)
        return model

    return wrapped_model_provider
```

**Comment：**

- 自定义 provider 与 `--custom-rm-path` 同样用 [[10-Sample-Contracts-00-MOC]] 的 `load_function`
- Bridge 路径见 §2

---

## 2. Megatron Bridge 模式

**Explain：** `megatron_to_hf_mode=="bridge"` 时从 HF checkpoint 构建 Megatron provider，无需先跑 torch_dist convert（权重仍 lazy load）。

**Code：**

```python
## 来源：slime/backends/megatron_utils/model_provider.py L87-L123
# 提交版本：22cdc6e1
if args.megatron_to_hf_mode == "bridge":
    from megatron.bridge import AutoBridge
    import slime_plugins.megatron_bridge  # register custom bridges

    bridge = patch_auto_bridge_hf_config(AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True))
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = args.tensor_model_parallel_size
    # ... PP/EP/SP/CP ...
    provider.finalize()
    return provider.provide
```

**Comment：**

- 与 [[05-Tools-DataPrep-00-MOC]] 的 torch_dist convert 是 **替代路线**
- critic 包装 `_critic_provide` 替换 output_layer

---

## 3. Critic 的 LinearForLastLayer

**Explain：** Critic 将 LM head 替换为 **hidden→1** 线性层，输出 per-token value；支持 sequence parallel gather。

**Code：**

```python
## 来源：slime/backends/megatron_utils/model_provider.py L25-L58
# 提交版本：22cdc6e1
class LinearForLastLayer(torch.nn.Linear):
    def forward(self, input_, weight=None, runtime_gather_output=None):
        logits = super().forward(input_)
        logits = logits.float()
        if self.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)
        return logits, None
```

**Comment：**

- actor 保留 vocab_size 维 output_layer
- checkpoint shape 不匹配时会 reinit（见 initialize）

---

## 4. wrap_model_provider_with_freeze

**Explain：** 在 provider 外包一层，按 regex 列表 freeze/unfreeze 参数（LoRA/部分训练）。

**Code：**

```python
## 来源：slime/backends/megatron_utils/model_provider.py L245-L269
# 提交版本：22cdc6e1
def get_model_provider_func(args, role="actor"):
    return wrap_model_provider_with_freeze(_get_model_provider_func(args, role), args)

def freeze_model_params(model: GPTModel, args: argparse.Namespace):
    if getattr(args, "only_train_params_name_list", None):
        for name, param in model.named_parameters():
            param.requires_grad = False
            for pattern in args.only_train_params_name_list:
                if re.search(pattern, name):
                    param.requires_grad = True
```

**Comment：**

- `only_train_params_name_list` 白名单模式
- `freeze_params_name_list` 黑名单模式

---

## 5. setup_model_and_optimizer

**Explain：** 调用 Megatron `get_model(provider)` 构建 DDP chunks，再 `get_megatron_optimizer` + LR scheduler。

**Code：**

```python
## 来源：slime/backends/megatron_utils/model.py L270-L318
# 提交版本：22cdc6e1
def setup_model_and_optimizer(args, role="actor"):
    assert not args.moe_use_upcycling
    assert args.load is not None or args.pretrained_checkpoint is not None

    model = get_model(get_model_provider_func(args, role), ModelType.encoder_or_decoder)
    # OptimizerConfig from args fields ...
    optimizer = get_megatron_optimizer(config=config, model_chunks=model, ...)
    opt_param_scheduler = get_optimizer_param_scheduler(args, optimizer)
    return model, optimizer, opt_param_scheduler
```

**Comment：**

- **必须** 提供 `load` 或 `pretrained_checkpoint` 之一
- `use_stateless_adam` 路径 patch Megatron Adam 类

---

## 6. LR scheduler 与 num_rollout

**Explain：** `train_iters` 由 rollout 总数与 batch 规模 **估算**，供 cosine/linear decay 使用；动态 sampling 可能导致实际 step 漂移。

**Code：**

```python
## 来源：slime/backends/megatron_utils/model.py L204-L206
# 提交版本：22cdc6e1
args.train_iters = args.num_rollout * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
if args.lr_decay_iters is None:
    args.lr_decay_iters = args.train_iters
```

**Comment：**

- `num_rollout` 来自[[06-PlacementGroup-00-MOC]] 的 RolloutManager 推导
- 可显式 `--lr-decay-iters` 精确控制

---

## 7. forward_only 角色

**Explain：** 训练前/中用于 **只前向** 计算 ref_log_probs、values、entropy；不走 backward。

**Code：**

```python
## 来源：slime/backends/megatron_utils/model.py L345-L377
# 提交版本：22cdc6e1
def forward_only(
    f: Callable[..., dict[str, list[torch.Tensor]]],
    args: Namespace,
    model: Sequence[DDP],
    data_iterator: Sequence[DataIterator],
    num_microbatches: Sequence[int],
    store_prefix: str = "",
    use_rollout_top_p_replay: bool = False,
) -> dict[str, list[torch.Tensor]]:
    """Run forward passes only and collect non-loss outputs (e.g., logprobs)."""
```

**Comment：**

- 回调 `f` 通常是 `get_log_probs_and_entropy` 或 `get_values`（[[21-Loss-Advantages-00-MOC]]）
- Megatron actor.init 末尾调用 forward_only 填充 ref/teacher log probs
