---
type: batch-doc
module: 10-Sample-Contracts
batch: "10"
doc_type: checkpoint
title: "Sample 契约 · 验收清单"
tags:
  - slime/batch/10
  - slime/module/sample-contracts
  - slime/doc/checkpoint
updated: 2026-07-02
---

# Sample 契约 · 验收清单

## 读者自测（不打开 slime/）

- [ ] 能说明 Sample 中 tokens / loss_mask / rollout_log_probs / reward 四字段在 PPO 中的作用
- [ ] 能解释 rollout_id 与 index 的区别及 compact 路径要求
- [ ] 能描述 RolloutFnTrainOutput 与 call_rollout_fn 的 legacy 兼容行为
- [ ] 能说明 append_response_tokens 中 trainable vs non-trainable 分支
- [ ] 能解释 RolloutBatch 与 Sample 列表的关系
- [ ] 能写出 load_function 的路径格式并举一个 CLI 挂载示例

## 维护者检查

- [ ] frontmatter tags 含 `slime/batch/10` + `slime/doc/*`
- [ ] 代码块含 `# 提交版本：22cdc6e1`
- [ ] 已更新 [[Slime-progress]] 批次 10 为 ✅
- [ ] （图谱增量）运行 `/understand --language zh` + `/understand-domain`

## 快速自测题

1. **top-p offsets 长度？** `response_length + 1`。
2. **Status.COMPLETED 对应 finish_reason？** `stop`。
3. **ParamInfo 用途？** 权重同步元数据（非 rollout 主路径）。

## 通过标准

全部读者自测项可口头回答，且能在 [[10-Sample-Contracts-02-源码走读]] 找到对应代码段。
