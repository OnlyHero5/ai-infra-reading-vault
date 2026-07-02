---
type: batch-doc
module: 05-Tools-DataPrep
batch: "05"
doc_type: faq
title: "Tools-DataPrep · 关键问题"
tags:
  - slime/batch/05
  - slime/module/tools-dataprep
  - slime/doc/faq
updated: 2026-07-02
---

# Tools-DataPrep · 关键问题

## Q1：为什么转换后 embedding 对不上？

**Explain：** Megatron 将 `vocab_size` padding 到 `padded_vocab_size`（TP 对齐）。直接转 HF 会多出 padding 行；需显式 `--vocab-size` 裁切。

**Code（易错）：**

```bash
# ❌ 未指定 vocab-size，embedding 行数可能 > HF config.vocab_size
python tools/convert_torch_dist_to_hf.py \
  --input-dir /path/to/release \
  --output-dir /out/hf \
  --origin-hf-dir /path/to/hf
```

**Code（推荐）：**

```bash
# ✅ 与 MODEL_ARGS / config 一致
python tools/convert_torch_dist_to_hf.py \
  --input-dir /path/to/release \
  --output-dir /out/hf \
  --origin-hf-dir /path/to/hf \
  --vocab-size 151936
```

**Comment：**

- quick_start 原文：*"Megatron will do padding to embedding"*（§ Convert from Megatron Format）
- `remove_padding` 只处理 embedding / output_layer 两个张量
- 训练用 Megatron 不受影响；问题多出现在 **HF 导出验证** 或 **外部推理加载**

## Q2：多卡转换怎么启动？

**Explain：** 大模型单卡 OOM 时用 `torchrun`；脚本在 `pipeline_model_parallel_size==1` 时按 world_size 自动选 PP。

**Code：**

```bash
# 来源：docs/en/get_started/quick_start.md L92（说明）+ convert 脚本行为
# 提交版本：22cdc6e1
torchrun --nproc_per_node=8 \
  tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /root/Qwen3-4B \
  --save /root/Qwen3-4B_torch_dist
```

**Comment：**

- 约束：`world_size <= num_layers`（Qwen3-4B 为 36 层，最多 36 卡 PP）
- 若已手动设置 `--pipeline-model-parallel-size`，脚本**不会**覆盖
- 需设置 `PYTHONPATH` 含 Megatron-LM（与 quick_start 一致）

## Q3：MODEL_ARGS 与 HF config 不一致会怎样？

**Explain：** 层数/hidden/head 不匹配时，bridge 灌权重会 shape error 或 silent 错层；RoPE base 错则训练仍跑但质量崩。

**Code（正确做法）：**

```bash
source scripts/models/qwen3-4B.sh
# 若确认 HF 版本 rotary_base 不同：
export MODEL_ARGS_ROTARY_BASE=5000000
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /root/Qwen3-4B \
  --save /root/Qwen3-4B_torch_dist
```

**Comment：**

- quick_start 强调检查 `--rotary-base` 等是否与**当前 HF 版本**一致
- kimi-k2 等特殊 case 需改 `config.json` 的 `model_type`（文档 Note）
- 转换与训练必须 **source 同一套** models/*.sh

## Q4：--ref-load 与 --load 何时用哪个？

| 场景 | `--ref-load` | `--load` |
|------|-------------|----------|
| 首次 RL 训练 | convert 产出的 torch_dist | 空或新目录（fallback ref-load） |
| Resume 训练 | 不变（初始参考） | 指向上次 `--save` 的 iter |
| 仅换 prompt 数据 | 不变 | 可选 resume |

**Explain：** `--load` 无有效 checkpoint 时 Megatron 从 `--ref-load` 加载；二者目录格式相同。

## Q5：bf16 训练 + FP8 Rollout 要重新 convert 吗？

**Explain：** 不需要。Megatron 仍用 **bf16 HF** 转出的 torch_dist；FP8 仅影响 `--hf-checkpoint` 供 SGLang/tokenizer。

**Code：**

```bash
# 来源：docs/en/get_started/quick_start.md L399-L407（摘录）
--hf-checkpoint /root/Qwen3-4B-FP8
--ref-load /root/Qwen3-4B_torch_dist   # 仍是 bf16 转换结果
```

**Comment：**

- FP8 权重 cast 发生在 Rollout / update_weights 路径
- 勿用 FP8 HF 目录跑 convert_hf_to_torch_dist 除非明确支持

## Q6：AMD / ROCm 转换要注意什么？

**Explain：** 必须 `--use-cpu-initialization`；脚本自动 patch checkpoint writer。

**Code：**

```bash
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --use-cpu-initialization \
  --hf-checkpoint /path/to/hf \
  --save /path/to/torch_dist
```

**Comment：**

- 断言见 `convert_hf_to_torch_dist.py` L118–119
- 详见 [[docs/en/platform_support/amd_tutorial]]（slime 文档）

## Q7：convert_torch_dist_to_hf 输出目录已存在？

**Explain：** 默认拒绝覆盖；需 `-f` / `--force`。

**Code：**

```python
# 来源：tools/convert_torch_dist_to_hf.py L209-L210
# 提交版本：22cdc6e1
if os.path.exists(args.output_dir) and not args.force:
    raise ValueError(f"Output directory {args.output_dir} already exists. Use --force to overwrite it.")
```

## Q8：与 verl / OpenRLHF 的权重流程差异？

| 框架 | 典型做法 |
|------|----------|
| Slime | 显式 HF→torch_dist 工具 + Megatron 原生 `--ref-load` |
| OpenRLHF | 多 HuggingFace 直接训练 |
| verl | FSDP/Megatron 混合，转换脚本各异 |

Slime 选择 **Megatron dist ckpt 原生路径**，换取大模型 TP/PP/EP 与训练 checkpoint 一致；代价是多一步离线 convert（本批）。

## Q9：--add-missing-from-origin-hf 何时开？

**Explain：** 部分权重未出现在 Megatron state_dict（如某些 buffer），可从原始 HF safetensors 补全。

**Code：**

```bash
python tools/convert_torch_dist_to_hf.py \
  --input-dir /path/to/iter_0000120 \
  --output-dir /out \
  --origin-hf-dir /path/to/original_hf \
  --add-missing-from-origin-hf
```

**Comment：**

- 会扫描 origin 下所有 `.safetensors` 键，跳过已在 converted_names 中的
- 适合发布 **完整 HF  repo**，而非最小权重集

## Q10：本批与批次 26 Checkpoint-M2HF 的分工？

| 批次 | 内容 |
|------|------|
| **05 Tools-DataPrep** | 离线 CLI 工具 + 训练前 ref-load + quick_start 操作 |
| **26 Checkpoint-M2HF** | 训练运行时 `megatron_to_hf`、`hf_checkpoint_saver`、与 update_weights 集成 |

本批读者只需知道：**同一套 naming 规则**，离线/在线两条入口。
