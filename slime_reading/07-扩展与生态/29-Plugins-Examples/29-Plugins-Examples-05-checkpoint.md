---
type: batch-doc
module: 29-Plugins-Examples
batch: "29"
doc_type: checkpoint
title: "Plugins Examples · 验收清单"
tags:
  - slime/batch/29
  - slime/module/plugins-examples
  - slime/doc/checkpoint
updated: 2026-07-02
---

# Plugins Examples · 验收清单

## 读者自测（不打开 slime/）

- [ ] 能对比 search-r1（custom_generate）与 multi_agent（rollout_function）接入差异
- [ ] 能说明 rollout_buffer 的 write / get_rollout_data API 语义
- [ ] 能解释 Search-R1 中 loss_mask=1 vs 0 的分界
- [ ] 能说出 `discover_generators` 的发现规则
- [ ] 能举一例 slime_plugins 非 example 的扩展（如 glm5 / megatron_bridge）

## 维护者检查

- [ ] frontmatter tags 含 `slime/batch/29`
- [ ] 六件套 ≥15 段代码、≥200 行
- [ ] 已更新 [[Slime-progress]] 批次 29 为 ✅
- [ ] [[07-扩展与生态-00-MOC]] 状态已同步

## 阶段 VII 完成

批次 26–29 完成后，Slime 阅读计划仅剩批次 30（全链路复盘 onboard）。
