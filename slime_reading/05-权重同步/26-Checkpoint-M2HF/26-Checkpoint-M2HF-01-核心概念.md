---
type: batch-doc
module: 26-Checkpoint-M2HF
batch: "26"
doc_type: concept
title: "Checkpoint M2HF · 核心概念"
tags:
  - slime/batch/26
  - slime/module/checkpoint-m2hf
  - slime/doc/concept
updated: 2026-07-02
---

# Checkpoint M2HF · 核心概念

## 1. 为什么需要 Megatron→HF？

Slime 训练侧用 Megatron 并行（TP/PP/EP/VPP），推理侧 SGLang 消费 HuggingFace 权重命名与布局。两条通路都需要格式转换：

| 场景 | 方向 | 入口 |
|------|------|------|
| 训练启动 | HF → Megatron | `checkpoint.load_checkpoint` → Bridge |
| 权重同步 / 存盘 | Megatron → HF | `convert_to_hf` + `save_hf_model_to_path` |
| NCCL broadcast | Megatron → HF 张量流 | `HfWeightIteratorDirect`（批次 24） |

---

## 2. `megatron_to_hf_mode`：bridge vs raw

**Explain：** Slime 提供两种 HF 互操作模式，由 CLI `--megatron-to-hf-mode` 控制。

| 模式 | 加载 HF | 保存 HF | 转换实现 |
|------|---------|---------|----------|
| `bridge` | ✅ AutoBridge.load_hf_weights | ✅ bridge.save_hf_pretrained | Megatron Bridge |
| `raw` | ❌ 不支持 | ✅ 自研 safetensors 分片 | `megatron_to_hf/*.py` + quantizer |

**Code：**

```python
# 来源：slime/backends/megatron_utils/hf_checkpoint_saver.py L22-L42
def save_hf_model_to_path(args, output_dir, model, *, model_name=None, quantization_config=None, ...):
    if args.megatron_to_hf_mode == "bridge":
        save_hf_model_bridge_to_path(args, output_dir, model)
    else:
        save_hf_model_direct_to_path(args, output_dir, model, model_name=model_name, ...)
```

**Comment：**

- **bridge** 依赖 `megatron.bridge.AutoBridge`，需 `--hf-checkpoint` 提供 config
- **raw** 不经过 Bridge，逐参数 `convert_to_hf` 后写 safetensors；与 NCCL 路径共享转换逻辑
- 加载 HF 时代码 **硬断言** bridge 模式（见 `_load_checkpoint_hf`）

---

## 3. Megatron checkpoint 识别

**Explain：** `load_checkpoint` 首先判断 `--load` 指向 Megatron 迭代目录还是 HF 模型目录。

**Code：**

```python
# 来源：slime/backends/megatron_utils/checkpoint.py L123-L126
def _is_megatron_checkpoint(path: str | Path) -> bool:
    return (Path(path) / "latest_checkpointed_iteration.txt").is_file() or bool(
        re.fullmatch(r"iter_\d{7}", Path(path).name)
    )
```

**Comment：**

- HF 目录通常含 `config.json` + 权重文件，不含上述 Megatron 标记
- 空目录会直接 assert 失败，避免 silent 误加载

---

## 4. HF 加载：Bridge 路径

**Explain：** 非 Megatron 路径时，通过 `AutoBridge.from_hf_pretrained` 把 HF 权重灌入已 patch 的 DDP model。

**Code：**

```python
# 来源：slime/backends/megatron_utils/checkpoint.py L129-L151
def _load_checkpoint_hf(ddp_model, optimizer, args, load_path: str):
    assert args.megatron_to_hf_mode == "bridge", "Only bridge mode is supported for loading HF checkpoint"
    from megatron.bridge import AutoBridge
    import slime_plugins.megatron_bridge  # noqa: F401
    with megatron_bridge_utils.patch_megatron_model(ddp_model):
        bridge = megatron_bridge_utils.patch_auto_bridge_hf_config(
            AutoBridge.from_hf_pretrained(load_path, trust_remote_code=True)
        )
        bridge.load_hf_weights(ddp_model)
    ...
    return 0, 0  # iteration 重置为 0
```

**Comment：**

- `slime_plugins.megatron_bridge` 注册扩展模型族
- FP16/BF16 时调用 `optimizer.reload_model_params()`
- 返回 iteration=0，与 Megatron 加载语义对齐

---

## 5. `convert_to_hf` 模型路由

**Explain：** `megatron_to_hf/__init__.py` 按 `model_name` 字符串子串路由到具体 converter；之后可选量化后处理。

**Code：**

```python
# 来源：slime/backends/megatron_utils/megatron_to_hf/__init__.py L25-L66
def convert_to_hf(args, model_name, name, param, quantization_config=None):
    param = remove_padding(name, param, args.vocab_size)
    converted_named_tensors = _convert_to_hf_core(args, model_name, name, param)
    return quantize_params(args, name, converted_named_tensors, quantization_config)

def _convert_to_hf_core(args, model_name, name, param):
    if "qwen2" in model_name or "qwen3" in model_name:
        converted_named_tensors = convert_qwen2_to_hf(args, name, param)
    elif "llama" in model_name:
        converted_named_tensors = convert_llama_to_hf(args, name, param)
    ...
    else:
        raise ValueError(f"Unsupported model: {model_name}")
```

**Comment：**

- 路由顺序有优先级（如 `glm4moe` 先于 `glm4`）
- `remove_padding` 去掉 vocab padding 行
- `quantize_params` 支持 FP8 / compressed_tensors 等后处理

---

## 6. Qwen2 转换示例：QKV 拆分

**Explain：** Megatron 把 Q/K/V 合并为 `linear_qkv`；HF 需要三个独立 proj。converter 按 GQA 组数 reshape + split。

**Code：**

```python
# 来源：slime/backends/megatron_utils/megatron_to_hf/qwen2.py L25-L36
        elif rest == "self_attention.linear_qkv.weight":
            param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            q_param, k_param, v_param = torch.split(param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1)
            q_param = q_param.reshape(-1, args.hidden_size)
            ...
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.weight", q_param),
                (f"model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
                (f"model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
            ]
```

**Comment：**

- 同样模式适用于 bias、MLP gate/up 拆分
- 未知参数名抛 `ValueError`，便于发现新层命名

---

## 7. 与 Actor 生命周期的衔接

**Explain：** `MegatronTrainRayActor.save()` 在 `--save-hf` 非空时调用 `save_hf_model_to_path`；`init` 末尾 `load_checkpoint` 恢复训练状态。

**Code：**

```python
# 来源：slime/backends/megatron_utils/actor.py L577（save 分支，节选）
save_hf_model_to_path(self.args, Path(self.args.save_hf.format(rollout_id=rollout_id)), self.model)
```

**Comment：**

- disk 权重同步（批次 25）在写临时目录时也调用同一 saver
- `--save` 仍走 Megatron 原生 `save_checkpoint`（本批不展开）

---

## 8. ShardedTensor 加载加速 patch

**Explain：** 大模型 HF 分片加载时 PyTorch 默认 `validate_non_overlapping_shards_metadata` 极慢；checkpoint.py 顶部 monkey-patch 跳过校验。

**Code：**

```python
# 来源：slime/backends/megatron_utils/checkpoint.py L14-L16
    # Here we patch out the `validate_non_overlapping_shards_metadata` in both functions
    # because it is really slow for large models with many shards.
    # TODO: find a less hacky way to do this.
```

**Comment：**

- 仅在 `torch.distributed._shard` 可用时生效
- 属于加载性能优化，不改变权重语义
