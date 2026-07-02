---
type: batch-doc
module: 27-Agent-Trajectory
batch: "27"
doc_type: checkpoint
title: "Agent Trajectory · 验收清单"
tags:
  - slime/batch/27
  - slime/module/agent-trajectory
  - slime/doc/checkpoint
updated: 2026-07-02
---

# Agent Trajectory · 验收清单

## 读者自测（不打开 slime/）

- [ ] 能解释 TurnRecord 各字段及为何必须用 SGLang output_token_logprobs
- [ ] 能说明 MessageNode 树如何 mount 多轮 prompt
- [ ] 能区分 DriftKind CLEAN / REALIGN / FORK 的触发条件
- [ ] 能描述 BaseAdapter `_run_turn` 从 translate 到 record_turn 的顺序
- [ ] 能说明 `finish_session` 如何把 trajectory 转为 `list[Sample]`

## 维护者检查

- [ ] frontmatter tags 含 `slime/batch/27` + `slime/doc/*`
- [ ] 六件套 ≥15 段代码、≥200 行
- [ ] 已更新 [[Slime-progress]] 批次 27 为 ✅

## 衔接

→ [[28-Customization-00-MOC]]：17 类 `--*-path` 如何挂 agent generate
