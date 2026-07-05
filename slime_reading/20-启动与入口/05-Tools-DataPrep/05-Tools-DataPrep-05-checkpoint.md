---
type: batch-doc
module: 05-Tools-DataPrep
batch: "05"
doc_type: checkpoint
title: "Tools-DataPrep · 验收清单"
tags:
  - slime/batch/05
  - slime/module/tools-dataprep
  - slime/doc/checkpoint
updated: 2026-07-02
---

# Tools-DataPrep · 验收清单

## 读者自测（不打开 slime/）

- [ ] 能说明为何 Megatron 训练前需要 `convert_hf_to_torch_dist.py`，以及 `--hf-checkpoint` 与 `--ref-load` 的分工
- [ ] 能按顺序口述 HF→torch_dist 七步：`parse_args` → `set_default_megatron_args` → `init` → `get_model` → `AutoBridge.load_weights` → `save_checkpoint` → `release` rename
- [ ] 能说明 `scripts/models/qwen3-4B.sh` 中至少 5 个 Megatron 超参含义（如 GQA、RoPE base、vocab）
- [ ] 能解释 torch_dist→HF 时为何要 `--vocab-size`，以及 `remove_padding` 作用在哪两层
- [ ] 能画出本模块在 RL 闭环中的位置：训练前 ref-load，不在 generate/train 热路径
- [ ] **阶段 I：** 能补全 `parse_args()` → `create_placement_groups()` → `create_rollout_manager()` → `create_training_models()` → `generate → async_train → update_weights` 调用栈（见 [[05-Tools-DataPrep-00-MOC]]）

## 快速自测题

1. **`common.pt` 里存的是什么？** Megatron 训练 args 快照；反向转换用来获取 `num_layers` / MoE 参数。
2. **`release` 与 `iter_0000001` 关系？** convert 脚本 save 后 rank0 把 iter_0000001 move 为 release 并写 tracker。
3. **Qwen3-4B FP8 rollout 要不要重跑 convert？** 不要；ref-load 仍用 bf16 转的 torch_dist。

## 通过标准

全部读者自测项可口头回答，且能在 [[05-Tools-DataPrep-02-源码走读]] 中找到对应内嵌代码段，即视为[[05-Tools-DataPrep-00-MOC]] 通过。
