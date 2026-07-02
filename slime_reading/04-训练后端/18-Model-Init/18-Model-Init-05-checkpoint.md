---
type: batch-doc
module: 18-Model-Init
batch: "18"
doc_type: checkpoint
title: "Model 初始化 · 验收清单"
tags:
  - slime/batch/18
  - slime/module/model-init
  - slime/doc/checkpoint
updated: 2026-07-02
---

# Model 初始化 · 验收清单

## 读者自测（不打开 slime/）

- [ ] 能说明 `get_model_provider_func` 三条路径：custom / bridge / legacy
- [ ] 能解释 critic 的 `LinearForLastLayer` 与 actor LM head 的区别
- [ ] 能口述 `initialize_model_and_optimizer` 五步：setup → role → load → reinit critic → return iteration
- [ ] 能说明 `forward_only` 与 `train_one_step` 的分工
- [ ] 能解释 `setup_model_and_optimizer` 对 load/pretrained_checkpoint 的 assert
- [ ] 能说明 `train_iters` 估算公式及 dynamic sampling 漂移影响

## 维护者检查

- [ ] frontmatter tags 含 `slime/batch/18` + `slime/doc/*`
- [ ] 代码块含 `# 提交版本：22cdc6e1`
- [ ] 已更新 [[Slime-progress]] 批次 18 为 ✅

## 快速自测题

1. **Stateless Adam 必须配什么 flag？** `--no-save-optim`。
2. **forward_only 谁聚合结果？** pipeline last stage。
3. **MoE spec 走哪个函数？** `get_gpt_decoder_block_spec`。

## 通过标准

全部读者自测项可口头回答，且能在 [[18-Model-Init-02-源码走读]] 找到对应代码段。
