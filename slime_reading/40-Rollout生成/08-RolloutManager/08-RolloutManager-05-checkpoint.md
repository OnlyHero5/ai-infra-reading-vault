---
type: batch-doc
module: 08-RolloutManager
batch: "08"
doc_type: checkpoint
title: "RolloutManager · 验收清单"
tags:
  - slime/batch/08
  - slime/module/rollout-manager
  - slime/doc/checkpoint
updated: 2026-07-02
---

# RolloutManager · 验收清单

---

## 读者自测（不打开 slime/）

- [ ] 能说明 RolloutManager 是 Ray remote Actor，`num_gpus=0`，负责 orchestrate 而非跑 forward
- [ ] 能画出 `generate()` 四步：`_get_rollout_data` → debug/log → `_convert_samples_to_train_data` → `_split_train_data_by_dp`
- [ ] 能描述 **Sample list → dict → ray.put × dp_size** 的形态变化
- [ ] 能说出 3 个核心函数职责：
  - `_get_rollout_data` — 调用 rollout fn，展平为 `list[Sample]`
  - `_convert_samples_to_train_data` — 列式 dict + reward/mask 处理
  - `_split_train_data_by_dp` — DP partition + tensorize + ObjectRef
- [ ] 能解释 `get_updatable_engines_and_lock` 为何排除 frozen 模型
- [ ] 能说明 `Sample.rollout_id` 在 compact rollout 下的必要性
- [ ] 能指出 `build_dp_schedule` 按 **rollout 数**（非 sample 数）切 training step

---

## 追踪练习（建议手写）

1. 从 `train.py` 的 `rollout_manager.generate.remote(rollout_id)` 出发，列出经过的 RolloutManager 方法（≥5 个）
2. 假设 `dp_size=2`，`global_batch_size=4`，8 条 Sample、`rollout_ids=[0,0,1,1,2,2,3,3]`，说明 `build_dp_schedule` 如何切 step
3. 画出 `rollout_data_refs[0]` 内 dict 的 key 列表（至少 10 个 key）

---

## 下一步阅读

| 专题 | 主题 | 与本专题关系 |
|------|------|-----------|
| [[09-EngineTopology-00-MOC]] | ServerGroup / Router / PD | 补全 `start_rollout_servers` |
| [[10-Sample-Contracts-00-MOC]] | Sample 全字段 | 深化 `_convert` 输入 |
| [[11-DataSource-00-MOC]] | data_source | `_get_rollout_data` 上游 |
| [[12-SGLang-Rollout-00-MOC]] | default generate_rollout | rollout fn 实现 |
| [[20-Train-Data-00-MOC]] | 训练侧消费 rollout_data | `_split` 下游 |
