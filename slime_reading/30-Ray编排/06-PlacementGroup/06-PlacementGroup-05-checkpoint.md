---
type: batch-doc
module: 06-PlacementGroup
batch: "06"
doc_type: checkpoint
title: "Placement Group · 验收清单"
tags:
  - slime/batch/06
  - slime/doc/checkpoint
  - slime/module/placement-group
updated: 2026-07-02
---

# Placement Group · 验收清单

## 读者自测（不打开 slime/）

- [ ] 能说明 `create_placement_groups` 返回 dict 的三键 `actor` / `rollout` / `critic` 各自含义
- [ ] 能画出 colocate 与非 colocate 两种 PG bundle 切分图
- [ ] 能解释 `rollout_offset` 在 `actor_pg_reordered_bundle_indices[rollout_offset:]` 中的作用
- [ ] 能说明 InfoActor + `sort_key` 重排 bundle 的动机
- [ ] 能口述 `create_rollout_manager` → `create_training_models` 的调用顺序及依赖
- [ ] 能列举 `_get_placement_group_layout` 在 debug / external / colocate 四种分支的 `(num_gpus, offset)` 返回值

## 快速自测题

1. **PACK 策略是什么？** 尽量把 bundle 打包到最少节点。
2. **colocate 时申请几块 GPU？** `max(actor_num_gpus, rollout_num_gpus)`，offset=0。
3. **RolloutManager 为何 num_gpus=0？** 协调者 Actor 不占 GPU；engine 通过 PG 视图间接绑定 bundle。

## 通过标准

全部读者自测项可口头回答，且能在 [[06-PlacementGroup-02-源码走读]] 找到对应内嵌代码，即视为[[06-PlacementGroup-00-MOC]] 通过。
