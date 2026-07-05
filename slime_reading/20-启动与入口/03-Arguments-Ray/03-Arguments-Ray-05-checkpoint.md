---
type: batch-doc
module: 03-Arguments-Ray
batch: "03"
doc_type: checkpoint
title: "Arguments-Ray · 验收清单"
tags:
  - slime/batch/03
  - slime/module/arguments
  - slime/doc/checkpoint
updated: 2026-07-02
---

# Arguments-Ray · 验收清单

## 读者自测

- [ ] 列举 cluster 段 8 个核心 CLI 及含义
- [ ] 口述 `parse_args` 三阶段与 skip_sglang 条件
- [ ] 解释 colocate 下 offload_train/offload_rollout 默认值
- [ ] 说明 rollout_num_gpus=0 与 external engines 关系
- [ ] 说明 debug_rollout_only 如何改写 actor 布局
- [ ] 能解释 delta weight 与 colocate 互斥原因

## 下一批

→ [[04-Arguments-TrainRollout-00-MOC]]
