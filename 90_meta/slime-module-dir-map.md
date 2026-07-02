---
type: doc
title: "Slime 目录与专题对照"
tags:
  - meta
  - maintenance
updated: 2026-07-03
---

# Slime 目录与专题对照

> **维护者专用** — 阶段文件夹 vs 专题模块。读者请用 [[Slime源码阅读指南]] 或阶段 MOC 导航。

| 批 | 阶段 | 专题目录 | 六件套前缀 | 核心文档 |
|:--:|------|----------|------------|----------|
| **01** | 0 地基 | `00-方法论/` | `00-方法论-` | 三角架构、ETC 阅读法 |
| **02** | I 启动 | `01-启动与入口/02-训练主循环/` | `02-训练主循环-` | train.py 主循环 |
| **03** | I | `01-启动与入口/03-Arguments-Ray/` | `03-Arguments-Ray-` | colocate/offload/PG |
| **04** | I | `01-启动与入口/04-Arguments-TrainRollout/` | `04-Arguments-TrainRollout-` | customization 参数 |
| **05** | I | `01-启动与入口/05-Tools-DataPrep/` | `05-Tools-DataPrep-` | HF↔Megatron 转换 |
| **06** | II Ray | `02-Ray编排/06-PlacementGroup/` | `06-PlacementGroup-` | PG 分配 |
| **07** | II | `02-Ray编排/07-RayTrainGroup/` | `07-RayTrainGroup-` | RayTrainGroup API |
| **08** | III Rollout | `03-Rollout生成/08-RolloutManager/` | `08-RolloutManager-` | generate() 枢纽 |
| **09** | III | `03-Rollout生成/09-EngineTopology/` | `09-EngineTopology-` | PD/多模型拓扑 |
| **10** | III | `03-Rollout生成/10-Sample-Contracts/` | `10-Sample-Contracts-` | Sample 契约 |
| **11** | III | `03-Rollout生成/11-DataSource/` | `11-DataSource-` | prompt 来源 |
| **12** | III | `03-Rollout生成/12-SGLang-Rollout/` | `12-SGLang-Rollout-` | 默认 generate 路径 |
| **13** | III | `03-Rollout生成/13-RM-FilterHub/` | `13-RM-FilterHub-` | RM + Filter |
| **14** | III | `03-Rollout生成/14-Alt-Rollout/` | `14-Alt-Rollout-` | async/streaming/SFT |
| **15** | III | `03-Rollout生成/15-SGLang-Engine/` | `15-SGLang-Engine-` | engine 生命周期 |
| **16** | III | `03-Rollout生成/16-External-Engines/` | `16-External-Engines-` | 外部 engine |
| **17** | IV 训练 | `04-训练后端/17-Megatron-Actor-Init/` | `17-Megatron-Actor-Init-` | actor init |
| **18** | IV | `04-训练后端/18-Model-Init/` | `18-Model-Init-` | model 初始化 |
| **19** | IV | `04-训练后端/19-Train-Step/` | `19-Train-Step-` | train step |
| **20** | IV | `04-训练后端/20-Train-Data/` | `20-Train-Data-` | rollout→batch |
| **21** | IV | `04-训练后端/21-Loss-Advantages/` | `21-Loss-Advantages-` | advantage |
| **22** | IV | `04-训练后端/22-Loss-Policy/` | `22-Loss-Policy-` | PPO/GRPO loss |
| **23** | IV | `04-训练后端/23-CP-RoutingReplay/` | `23-CP-RoutingReplay-` | CP + MoE replay |
| **24** | V 同步 | `05-权重同步/24-WeightSync-Dist/` | `24-WeightSync-Dist-` | NCCL 同步 |
| **25** | V | `05-权重同步/25-WeightSync-Disk/` | `25-WeightSync-Disk-` | disk/delta |
| **26** | V | `05-权重同步/26-Checkpoint-M2HF/` | `26-Checkpoint-M2HF-` | checkpoint |
| **27** | VI 高级 | `06-高级特性/27-Agent-Trajectory/` | `27-Agent-Trajectory-` | Agent 轨迹 |
| **28** | VI | `06-高级特性/28-Customization/` | `28-Customization-` | 17 hooks |
| **29** | VII 扩展 | `07-扩展与生态/29-Plugins-Examples/` | `29-Plugins-Examples-` | plugins/examples |
| **30** | VIII 收官 | `08-总结与索引/` | `08-总结与索引-` | onboarding + 索引 |

---

## 命名规则

- **批次号**（01–30）= 阅读顺序与 `Slime-progress.md` 行号
- **文件夹前缀**（`00-`/`01-`/`02-`…）= **阶段目录编号**，不是批次号
- **六件套前缀**（`02-训练主循环-`）= 模块名，与批次号对齐（批次 02 → `02-训练主循环-`）
- **例外：** 批次 01 在 `00-方法论/`（方法论先于编号体系固化）；批次 30 在 `08-总结与索引/`（阶段 VIII）

---

## 阶段文件夹 ↔ 批次范围

| 阶段文件夹 | 批次范围 | 主题 |
|-----------|----------|------|
| `00-方法论/` | 01 | 阅读方法论 |
| `01-启动与入口/` | 02–05 | train、arguments、tools |
| `02-Ray编排/` | 06–07 | PlacementGroup、RayTrainGroup |
| `03-Rollout生成/` | 08–16 | Rollout 全栈 |
| `04-训练后端/` | 17–23 | Megatron 训练 |
| `05-权重同步/` | 24–26 | 权重桥 |
| `06-高级特性/` | 27–28 | Agent、Customization |
| `07-扩展与生态/` | 29 | plugins、examples |
| `08-总结与索引/` | 30 | 收官索引 |

---

## 最小主链路（时间紧）

批次：**01, 02, 06, 08, 12, 17, 19, 21, 24, 30**

→ [[08-总结与索引-04-导读路径]] · [[AGENT-DISPATCH#六、推荐阅读顺序]]

---

## 导航

- [[08-总结与索引-00-MOC]]
- [[Slime-progress]]
- [[与SGLang阅读对照]]
