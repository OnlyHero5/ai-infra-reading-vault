---
type: index-doc
title: "08-总结与索引-05-checkpoint"
doc_type: checkpoint
tags:
  - slime/batch/30
  - slime/doc/checkpoint
aliases:
  - "08-总结与索引-checkpoint"
updated: 2026-07-02
---

# 验收清单（checkpoint）

> 索引层 · 对应 slime `22cdc6e1` · [[08-总结与索引-00-MOC]] 收官验收

---

## 读者自测

### 新手路径（完成 Step 1–6 后）

- [ ] 口头解释 Slime 三角：Training / Rollout / Data Buffer 各做什么
- [ ] 能复述 `generate → train → update_weights` 三步及其输入输出
- [ ] 能画出 [[全链路RL训练追踪]] 七 hop 中 PlacementGroup、RolloutManager、MegatronActor 的位置
- [ ] 解释 `rollout_id` 在 save/eval/trace 中的作用
- [ ] 说出 `Sample` 至少 4 个字段及训练用途（tokens、loss_masks、rewards、rollout_log_probs）

### 有 RL / 分布式基础读者

- [ ] 仅读 `slime_reading/08-总结与索引/` 可复述同步 vs 异步主循环差异
- [ ] 能解释 colocate 为何强制 offload
- [ ] 能说明 NCCL weight sync 与 disk delta 选型场景
- [ ] 能列举 3 个 `--*-path` customization 入口及用途
- [ ] 能对照 [[与SGLang阅读对照]] 说明 Slime Hop 4 对应 SGLang 哪几 hop

### 专题深潜验收

- [ ] 完成至少 1 个专题批 `05-checkpoint` 自测（建议 08-RolloutManager 或 19-Train-Step）
- [ ] 运行或阅读 `tests/test_qwen3_4B_ppo.py` 注释，理解 e2e 断言

---

## 已知局限

1. **行号漂移** — 基线 commit `22cdc6e1`；以函数名为锚在 upstream 检索。
2. **megatron_server.py** — 见 [[Slime-未独立成专题导读]] §2；暂无独立专题。
3. **知识图谱更新** — [[08-总结与索引-00-MOC]] 可选跑 `/understand --full` + `/understand-domain` 终版（见 [[08-总结与索引-04-导读路径]]）。

---

## 核心结论

1. **索引层**可独立 onboarding：项目定位、7 层架构、RL 七 hop、12 核心概念、12 步导读。
2. **业务域流程**与 **模块依赖图** 串联 01–29 专题，不重复专题深度。
3. **与 SGLang 对照**帮助双库读者在 Rollout 层快速切换上下文。
4. **专题深度**仍以 [[Slime-00-方法论-00-MOC]]–29 为主体；索引层负责导航与全链路复盘。

---

## 导航

- [[08-总结与索引-00-MOC]]
- [[08-总结与索引-04-导读路径]]
-
