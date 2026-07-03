---
type: batch-doc
module: 14-Alt-Rollout
batch: "14"
doc_type: checkpoint
title: "Alt-Rollout · 验收清单"
tags:
  - slime/batch/14
  - slime/module/alt-rollout
  - slime/doc/checkpoint
updated: 2026-07-02
---

# Alt-Rollout · 验收清单

---

## 读者自测（不打开 slime/）

- [ ] 能说明 `--rollout-function-path` 在 RolloutManager 中如何加载与调用
- [ ] 能区分 **外层 rollout**、**custom-generate**、**custom-rm** 三层 hook
- [ ] 能解释 fully-async 为何需要全局 `AsyncRolloutWorker`，以及 ABORTED 组为何回灌 buffer
- [ ] 能画出 `train_async.py` + fully-async 的双重叠时序（generate N+1 ∥ train N ∥ worker 持续 in-flight）
- [ ] 能对比 sync / train_async / fully-async 三者适用场景与 colocate 限制
- [ ] 能说明 streaming generate 对 abort/partial_rollout 的优势（chunk 级写入 sample）
- [ ] 能描述 SFT rollout 如何在不调 SGLang 的情况下产出 `tokens` + `loss_mask`
- [ ] 能说明 OPD 中 `reward_func` → `teacher_log_probs` → 标量 reward=0 的数据流
- [ ] 能对比 `forge_load` 与 `load-debug-rollout-data`（SGLang 是否 live、用途差异）
- [ ] 能说出 forge_load **禁止** overwrite `sample.rollout_id` 的原因

---

## 核心函数速记

| 函数 | 文件 | 职责 |
|------|------|------|
| `generate_rollout_fully_async` | fully_async_rollout.py | 外层 fully-async 入口 |
| `AsyncRolloutWorker._loop` | fully_async_rollout.py | 跨 step 并发池 |
| `generate_streaming` | sglang_streaming_rollout.py | SSE 内层 generate |
| `generate_rollout` | sft_rollout.py | SFT tokenize |
| `reward_func` / `post_process_rewards` | on_policy_distillation.py | OPD 教师 log-prob |
| `sleep` | sleep_rollout.py | Profiling 占位 |
| `generate_rollout` | forge_load.py | 磁盘 replay |

---

## 延伸阅读

- [[14-Alt-Rollout-00-MOC]] — 专题入口
- [[12-SGLang-Rollout-00-MOC]] — 默认 rollout 路径
- [[20-Train-Data-00-MOC]] — Sample → train tensor（下游）
- Slime 文档：`docs/en/advanced/on-policy-distillation.md`、`docs/en/developer_guide/profiling.md`
