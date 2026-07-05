---
type: batch-doc
module: 28-Customization
batch: "28"
doc_type: checkpoint
title: "Customization · 验收清单"
tags:
  - slime/batch/28
  - slime/module/customization
  - slime/doc/checkpoint
updated: 2026-07-02
---

# Customization · 验收清单

## 读者自测（不打开 slime/）

- [ ] 能列举 ≥5 个 `--*-path` 参数及用途
- [ ] 能说明 agent 任务为何优先 custom_generate + custom_rm
- [ ] 能解释 `load_function` 如何解析 path
- [ ] 能描述 fan-out 时 `rollout_id` 契约
- [ ] 能说明 harness + adapter 如何配合 sandbox agent

## 场景自测

1. **换 rollout 函数**：读 [[28-Customization-02-源码走读]] 中 `--rollout-function-path` 签名，写出最小函数应返回哪些键（对照 [[10-Sample-Contracts-01-核心概念]]）。
2. **挂自定义 RM**：说明 `--custom-rm-path` 在 [[13-RM-FilterHub-03-数据流与交互]] 中的调用时机——在 generate 之后还是 train 之前？
3. **Agent 组合**：若同时设置 `--custom-generate-function-path` 与 `--agent-adapter-path`，对照 [[28-Customization-01-核心概念]] 说明哪条路径优先、为何文档推荐 agent 任务走 generate 侧。

## 衔接

→ [[29-Plugins-Examples-00-MOC]]：把接口落到具体 example/plugin
