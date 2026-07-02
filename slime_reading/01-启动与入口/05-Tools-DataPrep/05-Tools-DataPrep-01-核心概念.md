---
type: batch-doc
module: 05-Tools-DataPrep
batch: "05"
doc_type: concept
title: "Tools-DataPrep · 核心概念"
tags:
  - slime/batch/05
  - slime/module/tools-dataprep
  - slime/doc/concept
updated: 2026-07-02
---

# Tools-DataPrep · 核心概念

## 架构位置

Slime RL 闭环（generate → train → update_weights）中，**训练后端 Megatron** 与 **Rollout 引擎 SGLang** 对权重格式的要求不同：

| 组件 | 典型权重来源 | 格式 |
|------|-------------|------|
| Megatron Actor | `--ref-load` / `--load` | Megatron **torch_dist**（分布式 checkpoint） |
| SGLang Rollout | 经 `update_weights` 推送 | 运行时 tensor / HF 映射（非本批重点） |
| Tokenizer / config | `--hf-checkpoint` | Hugging Face 目录（config、tokenizer、可选 FP8 元数据） |

本批工具链解决 **离线** 问题：训练启动前把 Hugging Face 权重转为 Megatron 可加载的 `torch_dist`；训练后把 Megatron checkpoint 转回 HF safetensors。

```mermaid
flowchart LR
    HF["HuggingFace 目录"]
    TD["torch_dist / release"]
    TR["Megatron 训练"]
    HF2["HF safetensors 导出"]

    HF -->|"convert_hf_to_torch_dist"| TD
    TD --> TR
    TR -->|"convert_torch_dist_to_hf"| HF2
```

## 术语表

| 术语 | 含义 |
|------|------|
| **MODEL_ARGS** | `scripts/models/*.sh` 中定义的 bash 数组，展开为 Megatron CLI（层数、hidden size、RoPE 等） |
| **torch_dist** | PyTorch Distributed Checkpoint 布局；Megatron 默认 dist checkpoint 格式 |
| **release** | 转换脚本把 iter_0000001 重命名后的目录名；`--load` 可指向含 `release` 的路径 |
| **AutoBridge** | `mbridge` 库：按 HF 架构把权重映射灌入 Megatron 模型实例 |
| **common.pt** | torch_dist 目录内的 Megatron args 快照；反向转换时恢复 `num_layers` 等 |
| **padded_vocab_size** | Megatron 对词表做对齐 padding；反向转 HF 时需 `--vocab-size` 裁切 |

## Megatron 为何不读 HF config

**Explain：** Megatron 训练/转换 CLI 需要显式并行度、层数、RoPE base 等；HF 的 `config.json` 不能单独驱动 Megatron 构图。因此 quick_start 要求先 `source scripts/models/xxx.sh`，再跑转换脚本。

**Code：**

```bash
# 来源：docs/en/get_started/quick_start.md L76-L89
# 提交版本：22cdc6e1
cd /root/slime
source scripts/models/glm4-9B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/GLM-Z1-9B-0414 \
    --save /root/GLM-Z1-9B-0414_torch_dist
```

**Comment：**

- `${MODEL_ARGS[@]}` 来自被 source 的 shell 脚本；Qwen3-4B 见下一节
- `--save` 是 Megatron 标准参数，指定 dist checkpoint 根目录
- 训练时同样要 source 模型脚本，保证 **转换与训练架构一致**

## Qwen3-4B 的 MODEL_ARGS 示例

**Explain：** `qwen3-4B.sh` 只定义 Megatron 侧结构超参，不含路径；与 HF `Qwen/Qwen3-4B` 的 config 字段应对齐（尤其 `--rotary-base`）。

**Code：**

```bash
# 来源：scripts/models/qwen3-4B.sh L1-L16
# 提交版本：22cdc6e1
MODEL_ARGS=(
   --swiglu
   --num-layers 36
   --hidden-size 2560
   --ffn-hidden-size 9728
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 8
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base "${MODEL_ARGS_ROTARY_BASE:-1000000}"
   --vocab-size 151936
   --kv-channels 128
   --qk-layernorm
)
```

**Comment：**

- `--group-query-attention` + `--num-query-groups 8` 对应 GQA
- `--vocab-size 151936` 会在 `set_default_megatron_args` 中扩展为 `padded_vocab_size`
- 环境变量 `MODEL_ARGS_ROTARY_BASE` 可覆盖 RoPE base，避免多版本 Qwen 混用

## 训练脚本中的三路 checkpoint 语义

**Explain：** quick_start 的 `CKPT_ARGS` 区分三条路径：HF 元数据、Megatron 参考权重、训练读写目录。

**Code：**

```bash
# 来源：docs/en/get_started/quick_start.md L138-L147（摘录）
# 提交版本：22cdc6e1
CKPT_ARGS=(
   --hf-checkpoint /root/GLM-Z1-9B-0414
   --ref-load /root/GLM-Z1-9B-0414_torch_dist
   --load /root/GLM-Z1-9B-0414_slime/
   --save /root/GLM-Z1-9B-0414_slime/
   --save-interval 20
)
```

**Comment：**

- `--hf-checkpoint`：**不**用于 Megatron 加载主权重；供 tokenizer、chat template、SGLang HF 路径
- `--ref-load`：本批 `convert_hf_to_torch_dist` 产出；首次训练或 `--load` 无效时加载
- `--load` / `--save`：RL 训练过程中的 Megatron checkpoint；结构与 `ref-load` 相同
- bf16 训练 + fp8 rollout 时，Megatron 仍用 bf16 转换的 torch_dist（quick_start § bf16 Training fp8 Inference）

## Slime 注入的 Megatron 默认值

**Explain：** 转换脚本调用 `set_default_megatron_args`，统一 optimizer、bf16、dist ckpt 行为，与正式训练一致。

**Code：**

```python
# 来源：slime/backends/megatron_utils/arguments.py L147-L177
# 提交版本：22cdc6e1
def _set_default_megatron_args(args):
    args.use_distributed_optimizer = True
    args.bf16 = not args.fp16
    args.use_persistent_ckpt_worker = True
    args.ckpt_assume_constant_structure = True
    args.ckpt_fully_parallel_load = True
    if args.seq_length is None:
        args.seq_length = 4096
    args.max_position_embeddings = args.seq_length
    args.dist_ckpt_save_pre_mcore_014 = True
    if args.vocab_size and not args.padded_vocab_size:
        args.padded_vocab_size = _vocab_size_with_padding(args.vocab_size, args)
    if not args.tokenizer_model and not args.tokenizer_type:
        args.tokenizer_model = args.hf_checkpoint
        args.tokenizer_type = "HuggingFaceTokenizer"
    return args
```

**Comment：**

- `padded_vocab_size` 是 embedding 转 HF 时可能「对不齐」的根源（见 [[05-Tools-DataPrep-04-关键问题]]）
- tokenizer 默认指向 `--hf-checkpoint`，转换阶段不跑训练但仍需 HF 路径合法
- 正式训练的 `parse_args()` 也会走同一套 default（批次 03–04）

## mbridge 与 slime_plugins

**Explain：** 转换脚本 import `slime_plugins.mbridge` 注册自定义桥接，再用 `AutoBridge.from_pretrained` 按 HF 路径选择映射逻辑。

**Code：**

```python
# 来源：tools/convert_hf_to_torch_dist.py L12-L14, L124-L126
# 提交版本：22cdc6e1
import slime_plugins.mbridge  # noqa: F401
from mbridge import AutoBridge
# ...
bridge = AutoBridge.from_pretrained(hf_model_path, trust_remote_code=True)
bridge.load_weights(model, hf_model_path, memory_efficient=True)
```

**Comment：**

- `memory_efficient=True` 降低大模型转换峰值内存
- 新架构通常需在 `slime_plugins/mbridge/` 扩展（批次 29 plugins）
- `--custom-model-provider-path` 可覆盖 Megatron model provider（MoE / 特殊层）

## 与 RL 闭环的关系

数据准备**不在** generate → train → update_weights 热路径上，但是闭环的**前置条件**：

1. **Train：** Megatron 从 `--ref-load` / `--load` 读 torch_dist
2. **Rollout：** SGLang 通过 `--hf-checkpoint` 拿 tokenizer；权重来自 `update_weights`
3. **Export：** 训练 save 的 iter_xxx 可用 `convert_torch_dist_to_hf.py` 回到 HF 生态

下一篇 [[05-Tools-DataPrep-02-源码走读]] 按代码顺序展开两个转换脚本。
