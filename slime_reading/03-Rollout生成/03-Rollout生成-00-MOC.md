---
type: phase-moc
phase: "03"
title: "Rollout 生成 · 阶段总览"
tags:
  - slime/phase/03
  - slime/doc/moc
updated: 2026-07-02
---

# Rollout 生成 · 阶段总览

> 批次 **08–16** | 状态：✅ 已完成

## 本阶段批次

| 批 | 模块 | 目录 | 状态 |
|----|------|------|------|
| 08 | RolloutManager 核心 | `08-RolloutManager/` | ✅ |
| 09 | Engine 拓扑 | `09-EngineTopology/` | ✅ |
| 10 | Sample 契约 | `10-Sample-Contracts/` | ✅ |
| 11 | DataSource | `11-DataSource/` | ✅ |
| 12 | SGLang Rollout | `12-SGLang-Rollout/` | ✅ |
| 13 | RM · Filter Hub | `13-RM-FilterHub/` | ✅ |
| 14 | Alt Rollout | `14-Alt-Rollout/` | ✅ |
| 15 | SGLang Engine | `15-SGLang-Engine/` | ✅ |
| 16 | External Engines | `16-External-Engines/` | ✅ |

## 阶段目标

能追踪 prompt → SGLangEngine.generate → Sample → rollout_data tensor 化的完整路径。

## 批次入口

- [[08-RolloutManager-00-MOC]]
- [[09-EngineTopology-00-MOC]]
- [[10-Sample-Contracts-00-MOC]]
- [[11-DataSource-00-MOC]]
- [[12-SGLang-Rollout-00-MOC]]
- [[13-RM-FilterHub-00-MOC]]
- [[14-Alt-Rollout-00-MOC]]
- [[15-SGLang-Engine-00-MOC]]
- [[16-External-Engines-00-MOC]]

## SGLang 前置

建议先读 [[03-HTTP-Server-00-MOC]]、[[07-Scheduler-00-MOC]]。

## 计划详情

见 [[PLAN#阶段 III：Rollout 生成 —— SGLang 推理 + 数据产出（批 08–16）]]
