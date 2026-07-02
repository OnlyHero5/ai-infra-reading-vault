---
type: phase-moc
phase: "01"
title: "启动与入口 · 阶段总览"
tags:
  - slime/phase/01
  - slime/doc/moc
updated: 2026-07-02
---

# 启动与入口 · 阶段总览

> 批次 **02–05** | 状态：✅ 已完成

## 本阶段批次

| 批 | 模块 | 目录 | 状态 |
|----|------|------|------|
| 02 | 训练主循环 | `02-训练主循环/` | ✅ |
| 03 | Arguments · Ray | `03-Arguments-Ray/` | ✅ |
| 04 | Arguments · Train/Rollout | `04-Arguments-TrainRollout/` | ✅ |
| 05 | Tools · DataPrep | `05-Tools-DataPrep/` | ✅ |

## 阶段目标

能画出 `parse_args()` → `create_placement_groups()` → `create_rollout_manager()` → `create_training_models()` → 主循环的完整调用栈。

## 批次入口

- [[02-训练主循环-00-MOC]]
- [[03-Arguments-Ray-00-MOC]]
- [[04-Arguments-TrainRollout-00-MOC]]
- [[05-Tools-DataPrep-00-MOC]]

## 计划详情

见 [[PLAN#阶段 I：启动与入口 —— 从 train.py 到参数体系（批 02–05）]]
