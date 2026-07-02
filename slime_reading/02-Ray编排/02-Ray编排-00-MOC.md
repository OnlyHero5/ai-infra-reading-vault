---
type: phase-moc
phase: "02"
title: "Ray 编排 · 阶段总览"
tags:
  - slime/phase/02
  - slime/doc/moc
updated: 2026-07-02
---

# Ray 编排 · 阶段总览

> 批次 **06–07** | 状态：✅ 已完成

## 本阶段批次

| 批 | 模块 | 目录 | 状态 |
|----|------|------|------|
| 06 | Placement Group 与资源分配 | `06-PlacementGroup/` | ✅ |
| 07 | RayTrainGroup 与 TrainRayActor | `07-RayTrainGroup/` | ✅ |

## 阶段目标

能解释 `--colocate` / `--offload-rollout` / `--offload-train` 下 GPU 如何通过 PG + RayTrainGroup 分时复用。

## 批次入口

- [[06-PlacementGroup-00-MOC]]
- [[07-RayTrainGroup-00-MOC]]

## 计划详情

见 [[PLAN#阶段 II：Ray 编排 —— GPU 资源与 Actor 拓扑（批 06–07）]]
