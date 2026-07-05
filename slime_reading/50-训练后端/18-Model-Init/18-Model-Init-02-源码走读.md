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
updated: 2026-07-05
---

# Model 初始化 · 源码走读

## 1. Model Provider：从 args 到 Megatron 模型

### 1.1 provider 根据 args 构造 GPTModel 与 critic head

问题与约束：
- Megatron `get_model` 会按 pipeline/virtual pipeline 多次调用 provider，因此 provider 必须接收 `pre_process/post_process/vp_stage`。
- layer spec 可能来自 TransformerEngine、本地实现、MoE 或用户自定义 spec。
- critic 角色最后一层输出维度不是 vocab，而是标量 value。
- `--fp8-param-gather` 需要在模型构造时进入 TransformerEngine 的 fp8 init context。

设计选择：
- `model_provider` 先从 args 构造 `TransformerConfig`，再解析 transformer layer spec。
- 将模型构造参数集中到 `kwargs`，在 `build_model_context` 内创建 `GPTModel`。
- `post_process and role == "critic"` 时替换 `output_layer` 为 `LinearForLastLayer(output_size=1)`。

Explain：
model provider 是 Megatron 模型构建的唯一入口。它把 Slime/Megatron args 翻译成 Megatron-Core GPTModel 所需的 config、layer spec、embedding/output 设置和可选 MTP/fp8 初始化上下文。

来源：slime/backends/megatron_utils/model_provider.py L125-L240

Code：

```python
def model_provider(pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None) -> GPTModel:
    use_te = args.transformer_impl == "transformer_engine"
    config: TransformerConfig = core_transformer_config_from_args(args)

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
        if callable(transformer_layer_spec):
            result = transformer_layer_spec(args, config, vp_stage)
            if callable(result) and "pre_process" in inspect.signature(result).parameters:
                model = result(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
                if post_process and role == "critic":
                    model.output_layer = LinearForLastLayer(
                        input_size=config.hidden_size, output_size=1, config=config
                    )
                return model
            transformer_layer_spec = result
    else:
        if args.num_experts:
            transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)
        elif use_te:
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(...)
        else:
            transformer_layer_spec = get_gpt_layer_local_spec(...)

    kwargs = {
        "config": config,
        "transformer_layer_spec": transformer_layer_spec,
        "vocab_size": args.padded_vocab_size,
        "max_sequence_length": args.max_position_embeddings,
        "pre_process": pre_process,
        "post_process": post_process,
        "parallel_output": True,
        "share_embeddings_and_output_weights": not args.untie_embeddings_and_output_weights,
        "position_embedding_type": args.position_embedding_type,
        "rotary_percent": args.rotary_percent,
        "rotary_base": args.rotary_base,
        "rope_scaling": args.use_rope_scaling,
    }

    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage

    with build_model_context(**build_model_context_args):
        model = GPTModel(**kwargs)

    if post_process and role == "critic":
        model.output_layer = LinearForLastLayer(input_size=config.hidden_size, output_size=1, config=config)

    return model
```

代码逻辑：
- 读取 transformer 实现并构造 Megatron `TransformerConfig`。
- 优先处理用户 `args.spec`，否则按 MoE/TE/local 选择 layer spec。
- 准备 GPTModel 构造参数。
- virtual pipeline stage 非空时写入 kwargs。
- 在可选 fp8 context 中构造模型。
- critic post-process stage 替换输出层为标量 head。

为什么这样写：
- provider 必须可重复调用，才能支持 pipeline model chunks。
- layer spec 和模型构造分离，便于替换 transformer block 实现。
- critic head 在 post-process stage 才存在，避免中间 pipeline stage 误加 value head。
- fp8 参数 gather 的初始化上下文必须包住模型构造，而不是构造后再补。

不变量与失败模式：
- `args.spec` 返回嵌套 provider 时，必须接受 `pre_process/post_process/vp_stage` 参数。
- critic 只有 post-process chunk 会替换 output layer。
- `fp8_param_gather` 依赖 TransformerEngine 的 `fp8_model_init`，缺失时会抛 RuntimeError。

Comment：
这段决定“训练后端实际构造的模型长什么样”，后续 optimizer/checkpoint 都基于它。

### 1.2 custom spec 可以返回 layer spec，也可以返回嵌套 provider

问题与约束：
- 有些模型只需要自定义 transformer layer spec。
- 有些多模态或特殊模型需要完全接管模型构造，但仍要被 Megatron `get_model` 按 provider 协议调用。
- critic 角色在嵌套 provider 返回模型后仍需要 value head。

设计选择：
- `args.spec` 导入后如果可调用，先执行 `result = transformer_layer_spec(args, config, vp_stage)`。
- 如果 result 仍是可调用且签名含 `pre_process`，就把它当嵌套 provider 调用。
- 否则把 result 当作 transformer layer spec。

Explain：
custom spec 的接口是双态的：可以只提供 layer spec，也可以提供完整 model provider。Slime 用签名检查区分两者，给复杂模型保留扩展点。

来源：slime/backends/megatron_utils/model_provider.py L143-L156

Code：

```python
if args.spec is not None:
    transformer_layer_spec = import_module(args.spec)
    if callable(transformer_layer_spec):
        result = transformer_layer_spec(args, config, vp_stage)
        if callable(result) and "pre_process" in inspect.signature(result).parameters:
            model = result(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
            if post_process and role == "critic":
                model.output_layer = LinearForLastLayer(
                    input_size=config.hidden_size, output_size=1, config=config
                )
            return model
        transformer_layer_spec = result
```

代码逻辑：
- 导入 `args.spec` 指向的对象。
- 可调用时先用 args/config/vp_stage 执行。
- 若返回对象仍是 provider 形态，则直接构造模型并返回。
- 嵌套 provider 模式下，critic post-process 仍替换 output layer。
- 非 provider 返回值作为 layer spec 继续走 GPTModel 构造。

为什么这样写：
- 简单自定义只需返回 layer spec，成本低。
- 复杂模型可以复用 Megatron provider 协议，而不被 GPTModel kwargs 限制。
- critic head 逻辑保留在 Slime 侧，避免每个自定义 provider 重复实现。

不变量与失败模式：
- 签名检查依赖参数名 `pre_process`，自定义 provider 必须遵守。
- 嵌套 provider 返回的模型需要有兼容的 `config.hidden_size`。
- 自定义 spec 返回错误类型会在后续 GPTModel 构造或调用时报错。

Comment：
这是 Slime 给模型结构扩展留出的最重要入口。

### 1.3 MTP layers 作为 GPTModel kwargs 的可选 block spec

问题与约束：
- MTP 训练需要在主 transformer 外附加 MTP block。
- MTP block 的 spec 必须与主 transformer layer spec 和 vp_stage 一致。
- 普通模型不能为 MTP 支付额外构造成本。

设计选择：
- 当 `args.mtp_num_layers` 非零时，导入 `get_gpt_mtp_block_spec`。
- 用当前 config、transformer layer spec 和可选 vp_stage 构造 `mtp_block_spec`。
- 将 `mtp_block_spec` 注入 GPTModel kwargs。

Explain：
MTP 是模型构造阶段的可选扩展，而不是训练 step 中临时挂载。provider 在创建 GPTModel 前把 MTP block spec 准备好。

来源：slime/backends/megatron_utils/model_provider.py L222-L232

Code：

```python
if args.mtp_num_layers:
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

    mtp_kwargs = {
        "use_transformer_engine": use_te,
    }
    if vp_stage is not None:
        mtp_kwargs["vp_stage"] = vp_stage

    mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, **mtp_kwargs)
    kwargs["mtp_block_spec"] = mtp_block_spec
```

代码逻辑：
- 检查是否启用 MTP layers。
- 延迟导入 MTP block spec factory。
- 准备是否使用 TransformerEngine 和 vp_stage 参数。
- 生成 MTP block spec。
- 写入 GPTModel kwargs。

为什么这样写：
- 只有启用 MTP 时才导入和构造相关 spec。
- MTP block 复用主 transformer layer spec，保持结构一致。
- 在 provider 阶段注入，Megatron-Core 可以统一初始化参数和并行切分。

不变量与失败模式：
- `args.mtp_num_layers` 非零时 Megatron 版本必须提供 `get_gpt_mtp_block_spec`。
- MTP spec 必须兼容当前 transformer layer spec。
- vp_stage 存在时必须传入，避免 virtual pipeline stage 构造不一致。

Comment：
MTP 的入口在模型初始化阶段，训练 step 只是消费这个结构。

## 2. setup_model_and_optimizer：模型、优化器与调度器

### 2.1 `get_model` 用 provider 构造 DDP model chunks

问题与约束：
- Megatron 需要按 pipeline/virtual pipeline 切分模型。
- provider 必须由 role 绑定，actor/critic 可能有不同输出层和配置。
- optimizer 构造依赖 Megatron 返回的 model chunks。

设计选择：
- `setup_model_and_optimizer` 调用 `get_model(get_model_provider_func(args, role), ModelType.encoder_or_decoder)`。
- 返回的 `model` 继续传给 Megatron optimizer 构造。

Explain：
`get_model` 是 Megatron-Core 对 provider 的调用点。Slime 只提供 role-aware provider factory，具体 DDP 包装和 model chunk 切分交给 Megatron。

来源：slime/backends/megatron_utils/model.py L291-L294

Code：

```python
assert not args.moe_use_upcycling
assert args.load is not None or args.pretrained_checkpoint is not None

model = get_model(get_model_provider_func(args, role), ModelType.encoder_or_decoder)
```

代码逻辑：
- 禁止当前路径使用 MoE upcycling。
- 要求存在 load checkpoint 或 pretrained checkpoint。
- 从 args/role 得到 provider function。
- 调 Megatron `get_model` 构造 encoder-or-decoder 类型模型。

为什么这样写：
- Megatron 的 `get_model` 负责 PP/VPP/DDP 细节，Slime 不重复实现。
- role-aware provider 让同一 setup 函数服务 actor 和 critic。
- checkpoint 约束提前 assert，避免模型构造后才发现没有初始化来源。

不变量与失败模式：
- `args.load` 或 `args.pretrained_checkpoint` 至少存在一个。
- `args.moe_use_upcycling` 在该路径下不能启用。
- provider 返回的模型必须兼容 Megatron DDP 和 optimizer。

Comment：
这是一行短代码，但它是 Slime 训练后端进入 Megatron 模型栈的入口。

### 2.2 Stateless Adam 通过临时 patch 替换 Megatron Adam

问题与约束：
- 某些省显存训练场景不希望保存 Adam moment state。
- Megatron optimizer 构造内部会选择 Adam 实现，外层需要临时替换。
- stateless optimizer 与保存 optimizer state 的 checkpoint 语义冲突。

设计选择：
- 启用 `args.use_stateless_adam` 时，断言 optimizer 是 Adam 且 `args.no_save_optim=True`。
- 使用 `_patch_megatron_adam(StatelessAdam)` 包住 `get_megatron_optimizer`。
- 构造后调用 `_disable_distributed_optimizer_state_initialization`。

Explain：
Stateless Adam 不是单独分支重写 optimizer 构造，而是在 Megatron optimizer 创建期间临时替换 Adam 类。这样仍复用 Megatron optimizer wrapper 和分布式优化器路径。

来源：slime/backends/megatron_utils/model.py L304-L316

Code：

```python
if args.use_stateless_adam:
    assert config.optimizer == "adam", "Stateless Adam only supports --optimizer adam."
    assert args.no_save_optim, "Stateless Adam does not save Adam moment states. Please set --no-save-optim."

optimizer_context = _patch_megatron_adam(StatelessAdam) if args.use_stateless_adam else nullcontext()
with optimizer_context:
    optimizer = get_megatron_optimizer(
        config=config,
        model_chunks=model,
        use_gloo_process_groups=args.enable_gloo_process_groups,
    )
if args.use_stateless_adam:
    _disable_distributed_optimizer_state_initialization(optimizer)
```

代码逻辑：
- 检查 stateless Adam 只配合 Adam optimizer。
- 检查不保存 optimizer state。
- 根据 flag 选择 patch context 或空 context。
- 在 context 内调用 Megatron optimizer factory。
- stateless 模式下禁用分布式 optimizer state 初始化。

为什么这样写：
- 临时 patch 可以保留 Megatron optimizer factory 的其余行为。
- `no_save_optim` 是语义约束：没有 moment state 就不能承诺保存/恢复 optimizer state。
- 禁用 state 初始化减少不必要的内存占用。

不变量与失败模式：
- `config.optimizer` 必须等于 `"adam"`。
- 必须设置 `args.no_save_optim`。
- Megatron 内部 Adam 路径变化可能影响 patch 的有效性。

Comment：
这是一个典型的“复用上游构造流程，只替换最小组件”的实现。

### 2.3 OptimizerParamScheduler 以 sample 数换算 warmup/decay

问题与约束：
- 用户配置可能以 iteration 为单位给出 warmup/decay。
- Megatron scheduler 接收的是 step 参数，Slime 这里按 global batch size 换算到 sample 计数。
- weight decay 和 WSD decay 也要进入同一个 scheduler。

设计选择：
- 先计算 `lr_warmup_steps/lr_decay_steps/wd_incr_steps/wsd_decay_steps`。
- 用 optimizer 和 args 中的 lr/wd/scheduler flag 构造 `OptimizerParamScheduler`。

Explain：
这个 scheduler 把优化器和训练全局进度绑定起来。Slime 在构造时把按 iteration 的配置换算成按 sample 数推进的步数。

来源：slime/backends/megatron_utils/model.py L217-L233

Code：

```python
opt_param_scheduler = OptimizerParamScheduler(
    optimizer,
    init_lr=args.lr_warmup_init,
    max_lr=args.lr,
    min_lr=args.min_lr,
    lr_warmup_steps=lr_warmup_steps,
    lr_decay_steps=lr_decay_steps,
    lr_decay_style=args.lr_decay_style,
    start_wd=args.start_weight_decay,
    end_wd=args.end_weight_decay,
    wd_incr_steps=wd_incr_steps,
    wd_incr_style=args.weight_decay_incr_style,
    use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
    override_opt_param_scheduler=args.override_opt_param_scheduler,
    wsd_decay_steps=wsd_decay_steps,
    lr_wsd_decay_style=args.lr_wsd_decay_style,
)

return opt_param_scheduler
```

代码逻辑：
- 传入 optimizer。
- 设置 warmup 初始 LR、最大 LR、最小 LR。
- 设置 warmup/decay steps 和 decay style。
- 设置 weight decay 起止值与递增策略。
- 设置是否使用或覆盖 checkpoint 中的 scheduler。
- 设置 WSD decay 相关参数。

为什么这样写：
- scheduler 与 optimizer 一起初始化，checkpoint load 时才能统一恢复或覆盖。
- sample 数口径能和 global batch size 关联，避免数据并行规模变化时只按 micro step 误解训练进度。
- 把 LR 和 weight decay 放在同一个 scheduler 中，保持 Megatron 优化器参数更新路径一致。

不变量与失败模式：
- `lr_warmup_steps`、`lr_decay_steps` 等必须在调用前计算完成。
- checkpoint scheduler 是否覆盖由 args flag 控制，错误组合会导致学习率恢复不符合预期。
- `optimizer` 不能为 None。

Comment：
模型初始化不仅是参数 tensor 初始化，也包括 optimizer 及其时间轴初始化。

## 3. Checkpoint 与 critic head

### 3.1 critic output layer metadata mismatch 会触发重初始化

问题与约束：
- 从 actor checkpoint 初始化 critic 时，checkpoint 的 `output_layer` 可能是 vocab logits head，而 critic 运行时需要 scalar value head。
- 直接加载 shape 不匹配的 head 会失败，或者不小心复用错误语义。
- 需要在 load 前判断是否要在 load 后重初始化。

设计选择：
- `_critic_output_layer_needs_reinit` 只在 `role == "critic"` 且 `args.load` 存在时生效。
- 读取 distributed checkpoint metadata。
- 遍历 critic output layers，对 `weight/bias` 的 runtime shape 与 checkpoint global shape 做比较。
- mismatch 或 metadata 缺失时返回 True。

Explain：
这个函数不直接加载 tensor，而是读取 checkpoint metadata 预判 critic head 是否兼容。若不兼容，主初始化流程会在 checkpoint load 之后重初始化 critic output layer。

来源：slime/backends/megatron_utils/model.py L125-L166

Code：

```python
def _critic_output_layer_needs_reinit(args: Namespace, model: Sequence[DDP], role: str) -> bool:
    if role != "critic" or args.load is None:
        return False

    from megatron.core.dist_checkpointing.serialization import load_tensors_metadata

    checkpoint_path = Path(get_load_checkpoint_path_by_args(args))
    if not (checkpoint_path / ".metadata").is_file():
        return False

    checkpoint_metadata = load_tensors_metadata(str(checkpoint_path))
    for _chunk_id, output_layer in _iter_critic_output_layers(model):
        for name in ("weight", "bias"):
            param = getattr(output_layer, name, None)
            if param is None:
                continue

            param_name = f"output_layer.{name}"
            ckpt_tensor_metadata = next(
                (
                    tensor_metadata
                    for key, tensor_metadata in checkpoint_metadata.items()
                    if key == param_name or key.endswith(f".{param_name}")
                ),
                None,
            )
            expected_shape = tuple(param.shape)
            checkpoint_shape = tuple(ckpt_tensor_metadata.global_shape) if ckpt_tensor_metadata is not None else None
            if checkpoint_shape == expected_shape:
                continue
            return True

    return False
```

代码逻辑：
- 非 critic 或无 load 路径直接 False。
- 找到 checkpoint 路径并确认 metadata 文件存在。
- 加载 checkpoint tensor metadata。
- 遍历 critic output layers 的 weight/bias。
- 找对应 metadata key。
- 比较 runtime shape 与 checkpoint shape。
- 发现缺失或不匹配则返回 True。

为什么这样写：
- metadata 比实际加载 tensor 更轻量，适合 load 前预判。
- actor→critic 初始化是常见路径，value head 不应强行复用 actor lm head。
- 返回 bool 让主流程在 checkpoint load 完成后统一重初始化并同步 optimizer 参数。

不变量与失败模式：
- metadata 缺失时函数返回 False，后续是否能正常 load 取决于 checkpoint loader。
- `_iter_critic_output_layers` 必须能定位 critic 的 output layer。
- 只比较 shape，不验证语义；shape 一致但语义错误仍需配置层保证。

Comment：
这段是 actor checkpoint 复用到 critic 的安全阀。

### 3.2 `initialize_model_and_optimizer` 串起 setup、load、reinit 和显存清理

问题与约束：
- HIP/ROCm checkpoint async writer 需要兼容 patch。
- 模型/optimizer/scheduler 构造后还要 load checkpoint。
- critic output layer 可能需要 load 后重初始化。
- checkpoint load 前后容易产生显存峰值。

设计选择：
- HIP 环境先 patch `FileSystemWriterAsync`。
- 调 `setup_model_and_optimizer(args, role)` 构造三件套，并标记 `model[0].role`。
- load 前判断 critic head 是否需要重初始化。
- load 前后各 `clear_memory()`。
- 如重初始化 critic head，fp16/bf16 且 optimizer 存在时调用 `optimizer.reload_model_params()`。

Explain：
这是训练模型初始化的总入口。它把前面的 provider、optimizer、scheduler、checkpoint 和 critic head 修正串成一个可恢复的初始化流程。

来源：slime/backends/megatron_utils/model.py L982-L1007

Code：

```python
if torch.version.hip:
    import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module

    from slime.utils.rocm_checkpoint_writer import ROCmFileSystemWriterAsync

    filesystem_async_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync
    print("[ROCm] Applied FileSystemWriterAsync patch for HIP compatibility")

model, optimizer, opt_param_scheduler = setup_model_and_optimizer(args, role)
model[0].role = role
reinit_critic_output_layer = _critic_output_layer_needs_reinit(args, model, role)
clear_memory()
iteration, _ = load_checkpoint(
    model,
    optimizer,
    opt_param_scheduler,
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

代码逻辑：
- ROCm/HIP 下替换 Megatron filesystem async writer。
- 构造模型、优化器和 scheduler。
- 在第一个 model chunk 上记录 role。
- 判断 critic head 是否要 load 后重初始化。
- 清理显存后加载 checkpoint。
- 必要时重初始化 critic output layer。
- 混合精度且 optimizer 存在时刷新 optimizer master params。
- 再次清理显存并返回 iteration。

为什么这样写：
- checkpoint load 是状态恢复边界，model/optimizer/scheduler 必须一起传入。
- critic head 重初始化必须发生在 checkpoint load 之后，否则会被 checkpoint 覆盖。
- mixed precision optimizer 可能维护 master params，重初始化模型参数后需要 reload。
- 两次 `clear_memory` 缓解初始化峰值显存。

不变量与失败模式：
- `setup_model_and_optimizer` 必须返回非空 model list。
- `load_checkpoint` 失败会中断初始化。
- critic head reinit 后如果未 reload optimizer master params，训练会使用旧 master weights。

Comment：
这段是 “模型可训练状态” 的组装点，而不是单纯 `GPTModel()`。

## 4. Forward-only 初始化后的推理辅助路径

### 4.1 forward-only step 复用 batch 构造，但禁用训练标签

问题与约束：
- rollout 评估 logprob/reward 等路径需要跑 forward-only，而不是训练 backward。
- 输入 batch 仍要经过 Megatron 的 packing、padding、context parallel 等处理。
- 多模态输入需要透传到 model forward。

设计选择：
- `forward_step` 内调用 `get_batch`，读取 tokens、packed seq params、loss masks 等。
- 构造 `forward_kwargs`，labels/position/attention mask 置为 None。
- 如果有 `multimodal_train_inputs`，合并进 forward kwargs。
- 返回 `output_tensor` 和用于打包结果的 partial。

Explain：
forward-only 路径不重新造一套数据入口，而是复用训练 batch 构造，然后把模型输出交给后处理函数计算需要收集的 rollout data。

来源：slime/backends/megatron_utils/model.py L411-L445

Code：

```python
batch = get_batch(
    data_iterator,
    batch_keys,
    args.data_pad_size_multiplier,
    args.allgather_cp,
)
unconcat_tokens = batch["unconcat_tokens"]
tokens = batch["tokens"]
packed_seq_params = batch["packed_seq_params"]
total_lengths = batch["total_lengths"]
response_lengths = batch["response_lengths"]
forward_kwargs = {
    "input_ids": tokens,
    "position_ids": None,
    "attention_mask": None,
    "labels": None,
    "packed_seq_params": packed_seq_params,
    "loss_mask": batch["full_loss_masks"],
}
if batch["multimodal_train_inputs"] is not None:
    forward_kwargs.update(batch["multimodal_train_inputs"])
output_tensor = model(**forward_kwargs)

output_kwargs = {
    "args": args,
    "unconcat_tokens": unconcat_tokens,
    "total_lengths": total_lengths,
    "response_lengths": response_lengths,
    "with_entropy": args.use_rollout_entropy,
}
if use_rollout_top_p_replay:
    output_kwargs.update(get_rollout_top_p_logprob_kwargs(args, batch))

return output_tensor, partial(f, **output_kwargs)
```

代码逻辑：
- 从 data iterator 构造 batch。
- 提取 tokens、packed seq params、长度和 response lengths。
- 构造模型 forward kwargs。
- 合并多模态训练输入。
- 执行 model forward。
- 构造输出后处理需要的 kwargs。
- top-p replay 时补充额外 logprob kwargs。
- 返回 output tensor 和 partial 后处理函数。

为什么这样写：
- forward-only 仍需与训练路径共享 batch padding/packing 逻辑，避免 token 对齐差异。
- labels 置 None 表示不走训练 loss。
- `partial(f, **output_kwargs)` 让 Megatron forward/backward framework 在 pipeline 后拿到统一后处理入口。

不变量与失败模式：
- `batch_keys` 必须覆盖 forward-only 所需字段。
- `batch["full_loss_masks"]` 必须和 tokens 对齐。
- 多模态输入字段必须匹配 model forward 签名。

Comment：
初始化后的模型不只用于 train step，也会被 rollout logprob/ref/reward 路径以 forward-only 方式调用。

### 4.2 forward-only 聚合只在 pipeline last stage 产生 rollout data

问题与约束：
- Pipeline parallel 下不是每个 stage 都有完整输出。
- forward data store 是按 microbatch 收集的 list，需要还原成按 key 聚合的 rollout data。
- dynamic batch 会改变 microbatch 顺序，需要按原始 index 还原。

设计选择：
- forward-only 完成后先把 model 切回 train mode。
- 只有 `mpu.is_pipeline_last_stage()` 时读取 `forward_data_store`。
- 按 key 合并每个 microbatch 的 list。
- dynamic batch 时用 `micro_batch_indices` 还原原始顺序。
- 用 `store_prefix` 区分不同来源的数据 key。

Explain：
forward-only 的结果只在 pipeline last stage 汇总。其他 stage 返回空 dict，避免中间 stage 产生不完整 rollout data。

来源：slime/backends/megatron_utils/model.py L487-L505

Code：

```python
rollout_data = {}
if mpu.is_pipeline_last_stage():
    keys = forward_data_store[0].keys()
    for key in keys:
        values = []
        for value in forward_data_store:
            assert isinstance(value[key], list)
            values += value[key]

        if args.use_dynamic_batch_size:
            origin_values = [None] * len(values)
            origin_indices = sum(data_iterator[0].micro_batch_indices, [])
            for value, origin_index in zip(values, origin_indices, strict=False):
                origin_values[origin_index] = value
            values = origin_values
        rollout_data[f"{store_prefix}{key}"] = values
return rollout_data
```

代码逻辑：
- 初始化空 dict。
- 仅 pipeline last stage 读取 store。
- 获取结果 keys。
- 对每个 key 拼接所有 microbatch list。
- dynamic batch 模式下按原始 indices 还原顺序。
- 加上 store prefix 后写入 rollout data。
- 返回聚合结果。

为什么这样写：
- pipeline last stage 才拥有最终 logits/outputs。
- microbatch 输出本来是 list，逐 key 拼接保留样本级数据。
- dynamic batch 重排后必须恢复原顺序，否则后续样本和 logprob/reward 会错配。
- prefix 让 actor/ref 等不同 forward-only 输出能共存在同一个 rollout sample 中。

不变量与失败模式：
- last stage 上 `forward_data_store` 必须非空。
- 每个 value 对应 key 的内容必须是 list。
- dynamic batch 的 `origin_indices` 长度必须覆盖 values。

Comment：
这段保证 forward-only 的结果在 PP/动态 batch 下仍能回到样本顺序。
