---
type: batch-doc
module: 26-Checkpoint-M2HF
batch: "26"
doc_type: walkthrough
title: "Checkpoint M2HF · 源码走读"
tags:
  - slime/batch/26
  - slime/module/checkpoint-m2hf
  - slime/doc/walkthrough
updated: 2026-07-02
---

# Checkpoint M2HF · 源码走读

## 走读顺序

1. `checkpoint.py` — 加载路由 + ShardedTensor patch
2. `megatron_to_hf/__init__.py` — `convert_to_hf` 入口
3. `megatron_to_hf/qwen2.py` — 代表 converter
4. `hf_checkpoint_saver.py` — bridge / raw 双路径保存
5. `actor.py` — save / load 挂接点

---

## 1. load_checkpoint 入口

**Explain：** Megatron `model.py` 初始化末尾调用；根据 `--load` 路径分流。

**Code：**

```python
## 来源：slime/backends/megatron_utils/checkpoint.py L97-L120
def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, checkpointing_context, skip_load_to_model_and_opt):
    args = get_args()
    load_path = args.load
    assert Path(load_path).exists() and _is_dir_nonempty(load_path), (
        f"{args.load=} does not exist or is an empty directory. Did you specify the wrong folder?"
    )
    if _is_megatron_checkpoint(load_path):
        return _load_checkpoint_megatron(
            ddp_model=ddp_model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=skip_load_to_model_and_opt,
        )
    else:
        return _load_checkpoint_hf(
            ddp_model=ddp_model,
            optimizer=optimizer,
            args=args,
            load_path=load_path,
        )
```

**Comment：**

- Megatron 分支透传 upstream `megatron.training.checkpointing.load_checkpoint`
- HF 分支忽略 optimizer scheduler 的 ckpt 恢复（iteration 固定为 0）

---

## 2. Megatron ckpt 判据

**Code：**

```python
## 来源：slime/backends/megatron_utils/checkpoint.py L123-L126
def _is_megatron_checkpoint(path: str | Path) -> bool:
    return (Path(path) / "latest_checkpointed_iteration.txt").is_file() or bool(
        re.fullmatch(r"iter_\d{7}", Path(path).name)
    )
```

**Comment：** 与 Megatron-LM 目录约定一致；勿把 HF 权重目录误命名为 `iter_0000123` 形式。

---

## 3. HF → Megatron Bridge 加载

**Code：**

```python
## 来源：slime/backends/megatron_utils/checkpoint.py L129-L151
def _load_checkpoint_hf(ddp_model, optimizer, args, load_path: str):
    assert args.megatron_to_hf_mode == "bridge", "Only bridge mode is supported for loading HF checkpoint"
    from megatron.bridge import AutoBridge
    import slime_plugins.megatron_bridge  # noqa: F401
    logger.info(f"Load checkpoint from HuggingFace model into Megatron (path={load_path})")
    with megatron_bridge_utils.patch_megatron_model(ddp_model):
        bridge = megatron_bridge_utils.patch_auto_bridge_hf_config(
            AutoBridge.from_hf_pretrained(load_path, trust_remote_code=True)
        )
        bridge.load_hf_weights(ddp_model)
    if (args.fp16 or args.bf16) and optimizer is not None:
        assert not args.load_main_params_from_ckpt
        optimizer.reload_model_params()
    iteration = 0
    num_floating_point_operations_so_far = 0
    return iteration, num_floating_point_operations_so_far
```

**Comment：**

- `patch_megatron_model` 适配 Slime 的 module 包装结构
- plugins 侧注册 Bridge 不支持的架构扩展

---

## 4. convert_to_hf 总入口

**Code：**

```python
## 来源：slime/backends/megatron_utils/megatron_to_hf/__init__.py L25-L31
def convert_to_hf(args, model_name, name, param, quantization_config=None):
    param = remove_padding(name, param, args.vocab_size)
    converted_named_tensors = _convert_to_hf_core(args, model_name, name, param)
    return quantize_params(args, name, converted_named_tensors, quantization_config)
```

**Comment：** 每个 Megatron 参数可 yield **多个** HF 张量（如 QKV 三分、gate/up 二分）。

---

## 5. 模型族路由（节选）

**Code：**

```python
## 来源：slime/backends/megatron_utils/megatron_to_hf/__init__.py L38-L66
def _convert_to_hf_core(args, model_name, name, param):
    if "minimaxm2" in model_name or "minimax_m2" in model_name:
        converted_named_tensors = convert_minimax_m2_to_hf(args, name, param)
    elif "glm4moelite" in model_name or "deepseekv3" in model_name or "glmmoedsa" in model_name:
        converted_named_tensors = convert_deepseekv3_to_hf(args, name, param)
    elif "qwen2" in model_name or "qwen3" in model_name:
        converted_named_tensors = convert_qwen2_to_hf(args, name, param)
    elif "llama" in model_name:
        converted_named_tensors = convert_llama_to_hf(args, name, param)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    ...
    return converted_named_tensors
```

**Comment：** 新增模型需添加 converter 模块并在路由表注册；MoE / VL 有独立文件。

---

## 6. Qwen2：embedding 与 output layer

**Code：**

```python
## 来源：slime/backends/megatron_utils/megatron_to_hf/qwen2.py L5-L11
def convert_qwen2_to_hf(args, name, param):
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]
```

**Comment：** Megatron 命名带 `module.module` 前缀来自 DDP + TransformerEngine 包装。

---

## 7. Qwen2：MLP gate/up 拆分

**Code：**

```python
## 来源：slime/backends/megatron_utils/megatron_to_hf/qwen2.py L52-L59
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]
```

**Comment：** SwiGLU 结构在 Megatron 侧合并为 `linear_fc1`；HF 侧分离 gate/up。

---

## 8. save_hf_model_to_path 分流

**Code：**

```python
## 来源：slime/backends/megatron_utils/hf_checkpoint_saver.py L22-L42
def save_hf_model_to_path(args, output_dir, model, *, model_name=None, quantization_config=None, progress_desc="Save HF checkpoint"):
    if args.megatron_to_hf_mode == "bridge":
        save_hf_model_bridge_to_path(args, output_dir, model)
    else:
        save_hf_model_direct_to_path(
            args, output_dir, model,
            model_name=model_name,
            quantization_config=quantization_config,
            progress_desc=progress_desc,
        )
```

**Comment：** bridge 路径更简单但依赖 Bridge 版本；raw 路径支持多节点分片写入。

---

## 9. raw 保存：目录准备与 metadata 广播

**Code：**

```python
## 来源：slime/backends/megatron_utils/hf_checkpoint_saver.py L71-L105
    is_save_rank = _is_global_rank_zero()
    if is_save_rank:
        path.mkdir(parents=True, exist_ok=True)
        _clear_existing_hf_weights(path)
        _copy_hf_assets(args.hf_checkpoint, path)
    ...
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    model_name, quantization_config = payload[0]
    hf_weight_iterator = HfWeightIteratorDirect(
        args=args, model=model, model_name=model_name, quantization_config=quantization_config,
    )
```

**Comment：**

- rank 0 从 `--hf-checkpoint` 复制 config/tokenizer（跳过权重文件）
- 禁止输出目录与 `--hf-checkpoint` 相同（防覆盖模板）

---

## 10. raw 保存：分块迭代 + 多节点 writer

**Code：**

```python
## 来源：slime/backends/megatron_utils/hf_checkpoint_saver.py L122-L138
    writer = _SafetensorShardWriter(path, enabled=is_writer_rank)
    pending_write = None
    for chunk_idx, hf_named_tensors in enumerate(
        hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights, progress_desc=progress_desc)
    ):
        if is_writer_rank and chunk_idx % num_save_nodes == save_node_rank:
            pending_write = (chunk_idx, hf_named_tensors)
            hf_named_tensors = None
        ...
        if (chunk_idx + 1) % num_save_nodes == 0:
            pending_write = _write_pending_chunk(writer, pending_write)
    pending_write = _write_pending_chunk(writer, pending_write)
    _finalize_distributed_shards(path, writer.state())
```

**Comment：**

- 每个节点一个 writer rank，按 chunk 轮转写 shard
- 最终 rank 0 合并 `model.safetensors.index.json`

---

## 11. bridge 保存

**Code：**

```python
## 来源：slime/backends/megatron_utils/hf_checkpoint_saver.py L144-L172
def save_hf_model_bridge_to_path(args, output_dir, model):
    from megatron.bridge import AutoBridge
    from megatron.core import mpu
    path = Path(output_dir)
    should_log = (
        mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
    )
    path.mkdir(parents=True, exist_ok=True)
    bridge = patch_auto_bridge_hf_config(AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True))
    with patch_megatron_model(model):
        bridge.save_hf_pretrained(model, path=path)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
```

**Comment：** 日志仅在 DP0+TP0 打印；全员 barrier 保证写完再退出。

---

## 12. Safetensor 分片写入器

**Code：**

```python
## 来源：slime/backends/megatron_utils/hf_checkpoint_saver.py L183-L209
    def write(self, named_tensors, shard_idx: int) -> None:
        from safetensors.torch import save_file
        state_dict = {}
        for name, tensor in named_tensors:
            if name in self.weight_map or name in state_dict:
                raise ValueError(f"Duplicate HF tensor while saving: {name}")
            total_size += tensor.numel() * tensor.element_size()
            state_dict[name] = _tensor_for_safetensors(tensor)
        filename = self._next_filename(shard_idx)
        save_file(state_dict, self.path / filename, metadata={"format": "pt"})
        for name in state_dict:
            self.weight_map[name] = filename
```

**Comment：** 重复 tensor 名立即失败；张量 detach + contiguous + CPU 后写入。
