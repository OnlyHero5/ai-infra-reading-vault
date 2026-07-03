---
type: batch-doc
module: 00-方法论
batch: "01"
doc_type: checkpoint
title: "方法论 · 验收清单"
tags:
  - slime/batch/01
  - slime/module/methodology
  - slime/doc/checkpoint
updated: 2026-07-02
---

# 方法论 · 验收清单

## 读者自测（不打开 slime/）

- [ ] 用一句话说明 Slime 的两大核心能力
- [ ] 画出 Training / Rollout / Data Buffer 三角，并标数据方向
- [ ] 口述 `generate → train → update_weights` 各步输入输出
- [ ] 说明 Slime「native 透传」相对 veRL 抽象层的意义
- [ ] 知道 slime_reading 六件套文件名与 ETC 读法
- [ ] 能指出 `train.py` 在闭环中的位置（不必记行号）

## 快速自测题

1. **Slime 三角第三角「Data Buffer」在代码里主要对应谁？** RolloutManager + DataSource + rollout 函数，不是独立 daemon。
2. **为何博文强调不 wrap trainer class？** 方便移动 `ray.get` 做 sync/async 与实验。
3. **SGLang 参数如何传入？** CLI 加 `--sglang-` 前缀，由 `sglang_parse_args` 解析合并。

## 通过标准

全部读者自测项可口头回答，且在 [[Slime-00-方法论-02-源码走读]] 中找到对应内嵌代码段，即视为[[Slime-00-方法论-00-MOC]] 通过。

## 下一批

→ [[02-训练主循环-00-MOC]]
