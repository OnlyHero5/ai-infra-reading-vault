---
type: progress
title: "Slime 阅读进度"
tags:
  - slime/meta
updated: 2026-07-02
---

# Slime 源码阅读进度

> 最后更新：2026-07-02  
> 总批次：**30** | 已完成：**30**  
> **总览：** [[Slime源码阅读指南]]

```
[██████████████████████████████] 30/30 (100%)
```

## 分阶段进度

| 阶段 | 批次 | 主题 | 完成数 | 状态 |
|------|------|------|--------|------|
| 0 地基 | 01 | 方法论 | 1/1 | ✅ |
| I 启动 | 02–05 | 训练入口·参数·工具 | 4/4 | ✅ |
| II Ray | 06–07 | Ray 编排 | 2/2 | ✅ |
| III Rollout | 08–16 | Rollout 全栈 | 9/9 | ✅ |
| IV 训练 | 17–23 | Megatron 训练 | 7/7 | ✅ |
| V 同步 | 24–26 | 权重同步 | 3/3 | ✅ |
| VI 高级 | 27–28 | Agent·定制 | 2/2 | ✅ |
| VII 扩展 | 29 | plugins·examples | 1/1 | ✅ |
| VIII 收官 | 30 | 全链路复盘 | 1/1 | ✅ |

## 批次明细

| 批 | 状态 | 产出目录 |
|----|------|----------|
| 01 | ✅ | `00-方法论/` |
| 02 | ✅ | `01-启动与入口/02-训练主循环/` |
| 03 | ✅ | `01-启动与入口/03-Arguments-Ray/` |
| 04 | ✅ | `01-启动与入口/04-Arguments-TrainRollout/` |
| 05 | ✅ | `01-启动与入口/05-Tools-DataPrep/` |
| 06 | ✅ | `02-Ray编排/06-PlacementGroup/` |
| 07 | ✅ | `02-Ray编排/07-RayTrainGroup/` |
| 08 | ✅ | `03-Rollout生成/08-RolloutManager/` |
| 09 | ✅ | `03-Rollout生成/09-EngineTopology/` |
| 10 | ✅ | `03-Rollout生成/10-Sample-Contracts/` |
| 11 | ✅ | `03-Rollout生成/11-DataSource/` |
| 12 | ✅ | `03-Rollout生成/12-SGLang-Rollout/` |
| 13 | ✅ | `03-Rollout生成/13-RM-FilterHub/` |
| 14 | ✅ | `03-Rollout生成/14-Alt-Rollout/` |
| 15 | ✅ | `03-Rollout生成/15-SGLang-Engine/` |
| 16 | ✅ | `03-Rollout生成/16-External-Engines/` |
| 17 | ✅ | `04-训练后端/17-Megatron-Actor-Init/` |
| 18 | ✅ | `04-训练后端/18-Model-Init/` |
| 19 | ✅ | `04-训练后端/19-Train-Step/` |
| 20 | ✅ | `04-训练后端/20-Train-Data/` |
| 21 | ✅ | `04-训练后端/21-Loss-Advantages/` |
| 22 | ✅ | `04-训练后端/22-Loss-Policy/` |
| 23 | ✅ | `04-训练后端/23-CP-RoutingReplay/` |
| 24 | ✅ | `05-权重同步/24-WeightSync-Dist/` |
| 25 | ✅ | `05-权重同步/25-WeightSync-Disk/` |
| 26 | ✅ | `05-权重同步/26-Checkpoint-M2HF/` |
| 27 | ✅ | `06-高级特性/27-Agent-Trajectory/` |
| 28 | ✅ | `06-高级特性/28-Customization/` |
| 29 | ✅ | `07-扩展与生态/29-Plugins-Examples/` |
| 30 | ✅ | `08-总结与索引/` |

## 前置任务（写作前）

- [x] 运行 `/understand --language zh F:/源码阅读/slime`（2026-07-02：结构提取管线，1606 nodes / 1029 edges，commit `22cdc6e1`）
- [ ] 运行 `/understand-domain F:/源码阅读/slime`（待办）
- [ ] Review `slime/.understand-anything/.understandignore`

## 更新日志

**2026-07-02 · 图谱 + 验收脚本**

- ✅ `/understand --language zh`：`slime/.understand-anything/knowledge-graph.json`（1606 nodes，commit `22cdc6e1`）
- ✅ 新增 `90_meta/audit_slime_moc.py`；30 批扫描：缺件 1、stub 8、代码段不足 1
- ⏳ `/understand-domain` 待跑

**2026-07-02 · 维护同步：导航/阶段 MOC 对齐 30/30**

- `index.md` Slime 进度 30/30 + 八阶段 MOC 表
- 7 个阶段 MOC 批次表全 ✅、双链到各批 `{NN-Module}-00-MOC`
- `Slime源码阅读指南.md` 当前进度同步

**2026-07-02 · 批次 20/22/23/25 六件套补全（中断恢复 · P7）**

- ✅ 批次 20 补 `02–05`（原仅 MOC+01）：dp_schedule、get_batch、DataIterator
- ✅ 批次 22/23/25 六件套从零写入（policy loss、CP/routing replay、disk/delta sync）
- 02 走读均 ≥15 ETC 段、≥200 行；基线 commit `22cdc6e1`

**2026-07-02 · 批次 30 收官（08-总结与索引）**

- ✅ onboard 七件套：`01-项目总览` … `07-可观测与CI`
- ✅ 索引六文档：`全链路RL训练追踪`、`Slime-业务域流程`、`Slime-模块依赖图`、`Slime-术语表`、`与SGLang阅读对照`、`Slime-10-批次编号对照`
- ✅ `08-总结与索引-05-checkpoint` 收官验收
- 进度 **29→30**；**30/30 全部完成**

**2026-07-02 · 批次 22/23/25 完成**

- ✅ `04-训练后端/22-Loss-Policy/` — policy/value/sft loss、ppo_utils、CISPO 测试
- ✅ `04-训练后端/23-CP-RoutingReplay/` — cp_utils、routing_replay、fill_routing_replay
- ✅ `05-权重同步/25-WeightSync-Disk/` — disk/delta/tensor 权重同步
- 进度 **26→29**；阶段 IV **7/7**、阶段 V **3/3** 收官；仅剩批次 30
