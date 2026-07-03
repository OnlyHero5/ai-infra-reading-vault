---
type: batch-doc
module: 12-SGLang-Rollout
batch: "12"
doc_type: checkpoint
title: "SGLang Rollout · 验收清单"
tags:
  - slime/batch/12
  - slime/module/sglang-rollout
  - slime/doc/checkpoint
updated: 2026-07-02
---

# SGLang Rollout · 验收清单

## 读者自测（不打开 slime/）

- [ ] 仅读本模块 slime_reading，能口头说明 `sglang_rollout.py` 的职责：async 批量 generate + RM + filter + abort
- [ ] 能画出本模块在 generate → train → update_weights 闭环中的位置（RolloutManager 与 SGLang Router 之间）
- [ ] 能说出 3 个核心符号及其职责：
  - `generate_rollout` — 同步入口，train/eval 分派
  - `GenerateState` — Singleton 调度状态（semaphore、pendings、abort）
  - `generate_and_rm_group` — 每组 `n_samples_per_prompt` 并发 generate
- [ ] 能追踪一条训练 rollout step：DataSource.get_samples → HTTP /generate → append_response_tokens → async_rm → dynamic_filter → RolloutFnTrainOutput
- [ ] 能解释 `--custom-generate-function-path` 与 `--rollout-function-path` 的区别与优先级（sample 级 override）
- [ ] 能说明 `tests/test_rollout_metrics.py` 验证的是什么（Sample top-p / loss_mask metrics 契约，而非直接测 HTTP）

## 建议验证命令

```bash
cd F:/源码阅读/slime
pytest tests/test_rollout_metrics.py -v --tb=short
```

预期：全部 unit test PASS（无需 GPU，`NUM_GPUS = 0`）。
