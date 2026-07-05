---
type: batch-doc
module: 11-DataSource
batch: "11"
doc_type: checkpoint
title: "DataSource · 验收清单"
tags:
  - slime/batch/11
  - slime/module/data-source
  - slime/doc/checkpoint
updated: 2026-07-02
---

# DataSource · 验收清单

> 基线 commit `22cdc6e1` | 产出目录 `03-Rollout生成/11-DataSource/`

## 读者自测（不打开 slime/）

- [ ] 能口头说明 **prompt 三条来源**：`prompt-data` → Dataset、空 Sample、buffer 回收
- [ ] 能画出 `get_samples` → generate → `add_samples` 闭环，并指出 buffer **优先**于 dataset
- [ ] 能解释 `get_samples` 返回 `list[list[Sample]]` 与 `n_samples_per_prompt` 的关系
- [ ] 能说出 `RolloutDataSource` 四个游标：`sample_offset`, `epoch_id`, `sample_group_index`, `sample_index`
- [ ] 能说明 `pop_first` 默认 buffer 出队策略及 `--buffer-filter-path` 扩展点
- [ ] 能区分 `buffer_filter`（取 prompt 前）与 `dynamic_sampling_filter`（生成后）
- [ ] 能追踪一条 prompt 从 jsonl 行 → `_build_messages` → `Sample.prompt` → SGLang 的路径
- [ ] 知道 `process_rollout_data` 属于 rollout→train 交接，不参与 prompt 加载

## 闭环位置

- [ ] 能指出 DataSource 在 **generate → train → update_weights** 中位于 generate **入口**（供给 prompt）
- [ ] 能说明 checkpoint 文件 `global_dataset_state_dict_{rollout_id}.pt` 存什么

## 深度检查（可选）

- [ ] 能解释 epoch 回绕时 `get_samples` 单次调用跨边界的行为
- [ ] 能说明 dynamic filter 丢弃样本 **未** 写回 buffer 的影响
- [ ] 能列举 `Dataset` 支持的文件格式与 `path@[start:end]` 切片语法

## 相关文档

- [[11-DataSource-00-MOC]]
- [[11-DataSource-01-核心概念]]
- [[11-DataSource-02-源码走读]]
- [[11-DataSource-03-数据流与交互]]
- [[11-DataSource-04-关键问题]]
