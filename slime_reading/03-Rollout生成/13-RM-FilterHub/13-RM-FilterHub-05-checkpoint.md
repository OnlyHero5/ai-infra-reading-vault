---
type: batch-doc
module: 13-RM-FilterHub
batch: "13"
doc_type: checkpoint
title: "RM-FilterHub · 验收清单"
tags:
  - slime/batch/13
  - slime/module/rm-filter-hub
  - slime/doc/checkpoint
updated: 2026-07-02
---

# RM-FilterHub · 验收清单

---

## 读者自测（不打开 slime/）

- [ ] 仅读本专题 slime_reading，能口头说明 **RM Hub** 与 **Filter Hub** 的分工
- [ ] 能画出 `generate_and_rm` → `async_rm` → `call_dynamic_filter` → `RolloutFnTrainOutput` 路径
- [ ] 能说出 3 个核心函数及其职责：
  - `async_rm` — RM 分发入口
  - `grade_answer_verl` / `compute_score` — 两类数学 scorer
  - `check_reward_nonzero_std` — 组级 dynamic sampling filter
- [ ] 能解释 `math` vs `dapo` 返回值差异及 `--reward-key` 的必要性
- [ ] 能说明 `tests/test_rm_math_dapo.py` 锁定的至少 2 处实现差异

---

## 本专题产出文件

| 文件 | 状态 |
|------|------|
| `13-RM-FilterHub-00-MOC.md` | ✅ |
| `13-RM-FilterHub-13-RM-FilterHub-01-核心概念.md` | ✅ |
| `13-RM-FilterHub-13-RM-FilterHub-02-源码走读.md` | ✅ |
| `13-RM-FilterHub-13-RM-FilterHub-03-数据流与交互.md` | ✅ |
| `13-RM-FilterHub-13-RM-FilterHub-04-关键问题.md` | ✅ |
| `13-RM-FilterHub-05-checkpoint.md` | ✅ |

---

## 衔接下一批

- 上一批：[[12-SGLang-Rollout-00-MOC]] — `generate_and_rm` 调用方
- 下一批：[[14-Alt-Rollout-00-MOC]] — fully_async / streaming / SFT rollout 变体
