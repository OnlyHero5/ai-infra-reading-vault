---
type: batch-doc
module: 18-Model-Init
batch: "18"
doc_type: walkthrough
title: "Model 初始化 · 源码走读"
tags:
  - slime/batch/18
  - slime/module/model-init
  - slime/doc/walkthrough
updated: 2026-07-02
---

# Model 初始化 · 源码走读

---

## 1. legacy GPTModel 构建

**Explain：** 非 bridge 路径用 `core_transformer_config_from_args` + layer spec 构建 Megatron-Core GPTModel。

**Code：**

```python
## 来源：slime/backends/megatron_utils/model_provider.py L125-L141, L203-L235
# 提交版本：22cdc6e1
def model_provider(pre_process=True, post_process=True, vp_stage=None) -> GPTModel:
    use_te = args.transformer_impl == "transformer_engine"
    config: TransformerConfig = core_transformer_config_from_args(args)
    # ... transformer_layer_spec from spec / MoE / TE ...
    with build_model_context(**build_model_context_args):
        model = GPTModel(**kwargs)
    if post_process and role == "critic":
        model.output_layer = LinearForLastLayer(input_size=config.hidden_size, output_size=1, config=config)
    return model
```

**Comment：**

- MoE 模型走 `get_gpt_decoder_block_spec`
- `--fp8-param-gather` 时用 TransformerEngine `fp8_model_init` context

---

## 2. get_model 与 DDP 包装

**Explain：** Megatron `get_model` 按 PP/VPP 切分 calling provider 多次，返回 DDP list。

**Code：**

```python
## 来源：slime/backends/megatron_utils/model.py L291-L294
# 提交版本：22cdc6e1
model = get_model(get_model_provider_func(args, role), ModelType.encoder_or_decoder)
```

**Comment：**

- `model` 是 `list[DDP]`，长度 = virtual pipeline stages
- 各 chunk 共享同一 provider 工厂

---

## 3. Stateless Adam 路径

**Explain：** OPD/省显存场景用 StatelessAdam 替换 Megatron Adam，且不保存 optimizer moment state。

**Code：**

```python
## 来源：slime/backends/megatron_utils/model.py L304-L316
# 提交版本：22cdc6e1
if args.use_stateless_adam:
    assert config.optimizer == "adam", "Stateless Adam only supports --optimizer adam."
    assert args.no_save_optim, "Stateless Adam does not save Adam moment states."

optimizer_context = _patch_megatron_adam(StatelessAdam) if args.use_stateless_adam else nullcontext()
with optimizer_context:
    optimizer = get_megatron_optimizer(...)
if args.use_stateless_adam:
    _disable_distributed_optimizer_state_initialization(optimizer)
```

**Comment：**

- `_patch_megatron_adam` 临时替换 megatron.core.optimizer.Adam
- 必须 `--no-save-optim`

---

## 4. initialize_model_and_optimizer 完整流程

**Explain：** setup → 标记 role → 检测 critic head → load checkpoint → 可选 reinit critic head。

**Code：**

```python
## 来源：slime/backends/megatron_utils/model.py L990-L1007
# 提交版本：22cdc6e1
model, optimizer, opt_param_scheduler = setup_model_and_optimizer(args, role)
model[0].role = role
reinit_critic_output_layer = _critic_output_layer_needs_reinit(args, model, role)
clear_memory()
iteration, _ = load_checkpoint(
    model, optimizer, opt_param_scheduler,
    checkpointing_context={},
    skip_load_to_model_and_opt=False,
)
if reinit_critic_output_layer:
    _reinitialize_critic_output_layer(args, model)
    if (args.fp16 or args.bf16) and optimizer is not None:
        optimizer.reload_model_params()
clear_memory()
return model, optimizer, opt_param_scheduler, iteration
```

**Comment：**

- HIP/ROCm 路径开头 patch FileSystemWriterAsync（同函数 L982-L988）
- `clear_memory` 在 load 前后各一次，缓解 peak VRAM

---

## 5. critic output layer reinit 检测

**Explain：** 从 actor checkpoint 加载 critic 时，output_layer shape 可能不匹配（vocab→1）；读 dist ckpt metadata 比对 shape。

**Code：**

```python
## 来源：slime/backends/megatron_utils/model.py L125-L166
# 提交版本：22cdc6e1
def _critic_output_layer_needs_reinit(args, model, role) -> bool:
    if role != "critic" or args.load is None:
        return False
    checkpoint_metadata = load_tensors_metadata(str(checkpoint_path))
    for _chunk_id, output_layer in _iter_critic_output_layers(model):
        expected_shape = tuple(param.shape)
        checkpoint_shape = tuple(ckpt_tensor_metadata.global_shape) if ckpt_tensor_metadata else None
        if checkpoint_shape != expected_shape:
            return True
    return False
```

**Comment：**

- 常见于 actor ckpt 直接 init critic
- reinit 后 `optimizer.reload_model_params()` 同步 master weights

---

## 6. get_optimizer_param_scheduler

**Explain：** 构造 Megatron OptimizerParamScheduler，warmup/decay 以 **sample 数** 计（× global_batch_size）。

**Code：**

```python
## 来源：slime/backends/megatron_utils/model.py L217-L233
# 提交版本：22cdc6e1
opt_param_scheduler = OptimizerParamScheduler(
    optimizer,
    init_lr=args.lr_warmup_init,
    max_lr=args.lr,
    min_lr=args.min_lr,
    lr_warmup_steps=lr_warmup_steps,
    lr_decay_steps=lr_decay_steps,
    lr_decay_style=args.lr_decay_style,
    # ...
)
```

---

## 7. forward_only 的 forward_step

**Explain：** 与 train 类似调用 `get_batch`，但 `forward_only=True` 且 model.eval()。

**Code：**

```python
## 来源：slime/backends/megatron_utils/model.py L411-L433
# 提交版本：22cdc6e1
batch = get_batch(
    data_iterator,
    batch_keys,
    args.data_pad_size_multiplier,
    args.allgather_cp,
)
forward_kwargs = {
    "input_ids": tokens,
    "position_ids": None,
    "attention_mask": None,
    "labels": None,
    "packed_seq_params": packed_seq_params,
    "loss_mask": batch["full_loss_masks"],
}
output_tensor = model(**forward_kwargs)
return output_tensor, partial(f, **output_kwargs)
```

**Comment：**

- `batch_keys` 含 tokens/loss_masks/response_lengths 等
- top-p replay 时扩展 keys（`_with_rollout_top_p_token_keys`）

---

## 8. forward_only 聚合到 rollout_data

**Explain：** 仅 pipeline last stage 收集 forward_data_store，按 key 合并 list。

**Code：**

```python
## 来源：slime/backends/megatron_utils/model.py L487-L505
# 提交版本：22cdc6e1
rollout_data = {}
if mpu.is_pipeline_last_stage():
    keys = forward_data_store[0].keys()
    for key in keys:
        values = []
        for value in forward_data_store:
            values += value[key]
        rollout_data[f"{store_prefix}{key}"] = values
return rollout_data
```

**Comment：**

- `store_prefix` 区分 ref vs actor log_probs（如 `"ref_"`）
- dynamic batch 时按 micro_batch_indices 重排

---

## 9. custom spec 与嵌套 provider

**Explain：** `--spec` 可返回 layer spec 或 **嵌套 model provider**（如 glm-omni VL）。

**Code：**

```python
## 来源：slime/backends/megatron_utils/model_provider.py L143-L156
# 提交版本：22cdc6e1
if args.spec is not None:
    transformer_layer_spec = import_module(args.spec)
    if callable(transformer_layer_spec):
        result = transformer_layer_spec(args, config, vp_stage)
        if callable(result) and "pre_process" in inspect.signature(result).parameters:
            model = result(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
            if post_process and role == "critic":
                model.output_layer = LinearForLastLayer(...)
            return model
```

---

## 10. MTP block spec（可选）

**Explain：** `--mtp-num-layers` 时附加 MTP block 到 GPTModel kwargs。

**Code：**

```python
## 来源：slime/backends/megatron_utils/model_provider.py L222-L232
# 提交版本：22cdc6e1
if args.mtp_num_layers:
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
    mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, **mtp_kwargs)
    kwargs["mtp_block_spec"] = mtp_block_spec
```

**Comment：**

- MTP 训练 step 见 [[19-Train-Step-00-MOC]]
- CI 有 mtp-only-grad 检查
