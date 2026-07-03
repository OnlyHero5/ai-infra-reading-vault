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

## 场景自测

1. **单轮 agent**：读 [[27-Agent-Trajectory-02-源码走读]] §2 `BaseAdapter._run_turn`，口头说出 `translate → generate → record_turn` 三步各传入什么对象。
2. **多轮 drift**：对照 [[27-Agent-Trajectory-01-核心概念]] 中 DriftKind 表，假设第 2 轮 prompt 与第 1 轮不一致，应触发 REALIGN 还是 FORK？说明依据字段。
3. **闭环出口**：追踪 [[27-Agent-Trajectory-03-数据流与交互]] 中 `finish_session` → `Sample` 字段映射，指出 `loss_mask` 与 `rollout_log_probs` 从哪来。

## 衔接

→ [[28-Customization-00-MOC]]：17 类 `--*-path` 如何挂 agent generate
