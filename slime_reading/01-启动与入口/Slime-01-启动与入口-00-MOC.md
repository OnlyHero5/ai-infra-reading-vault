---
type: phase-moc
phase: "01"
title: "启动与入口 · 阶段总览"
tags:
  - slime/phase/01
  - slime/doc/moc
updated: 2026-07-03
---

# 阶段 I · 启动与入口（train.py → 主循环）

> **你只需阅读本目录，不必打开 `slime/` 源码。**
> 内嵌代码对应 slime Git commit `22cdc6e1`。

---

## 本阶段解决什么问题

阶段 0（方法论）讲清了「如何读 Slime」。阶段 I 回答：**一条 `python train.py` 命令如何从 CLI 参数出发，完成 GPU 资源预订、Rollout 子系统与 Megatron Actor 初始化，并进入 `generate → train → update_weights` 主循环？**

四个专题覆盖启动链全路径：

| 模块 | 角色 | 一句话 |
|------|------|--------|
| [[02-训练主循环-00-MOC|02 训练主循环]] | 入口与主循环 | `train()` / `train_async()` bootstrap 与 sync/async 迭代 |
| [[03-Arguments-Ray-00-MOC|03 Arguments-Ray]] | 集群参数 | `--colocate` / `--offload-*` / PG 相关 CLI 与 validate |
| [[04-Arguments-TrainRollout-00-MOC|04 Arguments-TrainRollout]] | 训练与 Rollout 参数 | Megatron/SGLang 透传、`--*-path` 扩展挂载 |
| [[05-Tools-DataPrep-00-MOC|05 Tools-DataPrep]] | 训练前工具 | HF ↔ Megatron `torch_dist` 双向转换与 MODEL_ARGS |

---

## 端到端时序（阶段 I 验收图）

满足阶段 I 验收：「`parse_args()` → `create_placement_groups()` → `create_rollout_manager()` → `create_training_models()` → 主循环」。

```mermaid
sequenceDiagram
 participant CLI as 命令行 / 脚本
 participant PA as 03/04 parse_args
 participant PG as 06 PlacementGroup<br/>(阶段 II)
 participant RM as 08 RolloutManager<br/>(阶段 III)
 participant TM as 07 RayTrainGroup<br/>(阶段 II)
 participant LOOP as 02 主循环

 CLI->>PA: sys.argv
 Note over PA: 03 · cluster/colocate<br/>04 · train/rollout/*-path
 PA->>PG: create_placement_groups()
 Note over PG: bundle 分配<br/>colocate 共用 PG
 PG->>RM: create_rollout_manager()
 Note over RM: 引擎拓扑 · DataSource
 PG->>TM: create_training_models()
 Note over TM: MegatronTrainRayActor<br/>async_init + update_weights
 LOOP->>RM: generate(rollout_id)
 LOOP->>TM: async_train(rollout_data_ref)
 LOOP->>TM: update_weights()
 Note over LOOP: 02 · sync vs async<br/>offload / save / eval
```

**Explain：** 启动链上 **PG 是第一个 GPU 决策点**；RolloutManager 与 RayTrainGroup 都依赖 PG 分配结果。主循环在 driver 进程 orchestrate，实际 generate/train 通过 Ray remote 下发到 Rollout 与 Megatron Actor。

---

## 零基础一句话

**像「开店前的筹备」：** 03/04 是装修图纸（参数），05 是进货（权重格式转换），02 是开业后的日常运营（generate → train → 换货/update_weights），06/07 是租场地与排班（阶段 II 详读）。

---

## 推荐阅读顺序

严格按专题顺序 02 → 03 → 04 → 05。若时间紧，最低闭环：**03 → 02**（先懂 colocate/offload，再读主循环）。

| 顺序 | 文档 | 必读理由 |
|------|------|----------|
| 1 | [[02-训练主循环-01-核心概念|02/01-核心概念]] | sync vs async、bootstrap 术语 |
| 2 | [[02-训练主循环-02-源码走读|02/02-源码走读]] | `train.py` / `train_async.py` 全文精读 |
| 3 | [[03-Arguments-Ray-02-源码走读|03/02-源码走读]] | colocate、offload、validate 分支 |
| 4 | [[04-Arguments-TrainRollout-03-数据流与交互|04/03-数据流与交互]] | `*-path` → `load_function` 挂载链 |
| 5 | [[05-Tools-DataPrep-02-源码走读|05/02-源码走读]] | HF ↔ torch_dist 转换主流程 |

---

## 阶段衔接

| 方向 | 模块 | 衔接点 |
|------|------|--------|
| ← 上一阶段 | 00 方法论 | 阅读方法与 Git 基线 |
| → 下一阶段 | 06–07 Ray 编排 | `create_placement_groups()` → PG + RayTrainGroup |
| → Rollout | 08–16 | `create_rollout_manager()` → RolloutManager |
| → 训练 | 17–23 | `create_training_models()` → Megatron Actor |
| → 权重 | 24–26 | 首次 `update_weights` → WeightSync |

---

## 验证建议（零基础可试）

1. **参数 dry-run：** 用 `--help` 分别查看 cluster / train / rollout 参数组，确认 `--colocate` 与 `--offload-rollout` 的默认值关系。
2. **启动链 grep：** 在笔记 [[02-训练主循环-02-源码走读]] 中对照 `train()` 前 50 行，口述 PG → RM → TM 顺序。
3. **权重转换：** 按 [[05-Tools-DataPrep-04-关键问题]] 走一遍 `convert_hf_to_torch_dist.py`，确认 `--ref-load` 指向转换产物。

---

## 模块导航

| 模块 | 目录 | 状态 |
|------|------|------|
| 02 | [[02-训练主循环-00-MOC|训练主循环]] | ✅ |
| 03 | [[03-Arguments-Ray-00-MOC|Arguments-Ray]] | ✅ |
| 04 | [[04-Arguments-TrainRollout-00-MOC|Arguments-TrainRollout]] | ✅ |
| 05 | [[05-Tools-DataPrep-00-MOC|Tools-DataPrep]] | ✅ |

← [[Slime-00-方法论-00-MOC|方法论]] · → [[02-Ray编排-00-MOC|阶段 II：Ray 编排]]
