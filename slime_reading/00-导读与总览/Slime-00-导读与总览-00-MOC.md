---
type: phase-moc
phase: "00"
title: "导读与总览"
tags:
  - slime/phase/00
  - slime/doc/moc
updated: 2026-07-04
---

# 导读与总览

> Slime 阅读第一站：Ray / Megatron 先修、项目总览、RL 全链路与术语索引。代码基线 `22cdc6e1`。

---

## 本目录定位

本目录解决“从哪里开始读 Slime”的问题。读者先在这里建立三个基础：

1. Ray 如何管理多机多 GPU 资源与远程 Actor。
2. Megatron 如何把大模型切到多张 GPU 上训练。
3. Slime 如何把 `generate → train → update_weights` 串成 RL 后训练闭环。

**复杂度热点、可观测、checkpoint 和未独立专题**已经移到 [[Slime-90-总结复盘-00-MOC]]。

---

## 快速入口

| 用途 | 文档 |
|------|------|
| 零基础先修 | [[Slime-00-零基础先修]] |
| 项目总览 | [[Slime-01-项目总览]] |
| RL 全链路 | [[全链路RL训练追踪]] |
| 12 步导读 | [[Slime-04-导读路径]] |
| 文件地图 | [[Slime-05-文件地图]] |
| 术语 | [[Slime-术语表]] |
| SGLang 对照 | [[与SGLang阅读对照]] |

---

## 文档地图

### Onboarding（建议顺序）

| 顺序 | 文档 | 内容 |
|------:|------|------|
| 0 | [[Slime-00-零基础先修]] | Ray、PlacementGroup、Megatron 并行、microbatch |
| 1 | [[Slime-01-项目总览]] | Slime 三角架构与 train 入口 |
| 2 | [[Slime-02-架构分层]] | 分层架构与代表代码 |
| 3 | [[Slime-03-关键概念]] | rollout_id、Sample、update_weights |
| 4 | [[全链路RL训练追踪]] | 一轮 RL 训练的七 hop |
| 5 | [[Slime-04-导读路径]] | 12 步 guided tour |

### 查阅索引

| 文档 | 内容 |
|------|------|
| [[Slime-05-文件地图]] | 顶层文件索引 |
| [[Slime-术语表]] | RL / Ray / Megatron 术语 |
| [[Slime-模块依赖图]] | 模块依赖 |
| [[Slime-业务域流程]] | 业务域流程 |
| [[与SGLang阅读对照]] | 跳回推理栈 |

---

## 推荐阅读路径

**完全新手：** [[Slime-00-零基础先修]] → [[Slime-01-项目总览]] → [[全链路RL训练追踪]] → [[Slime-04-导读路径]]

**已有 SGLang 基础：** [[Slime-00-零基础先修]] → [[全链路RL训练追踪]] → [[03-Rollout生成-00-MOC]]

**先看训练侧：** [[Slime-00-零基础先修]] → [[04-训练后端-00-MOC]] → [[05-权重同步-00-MOC]]

---

## 下一站

| 目标 | 下一站 |
|------|--------|
| 看 Ray 编排 | [[02-Ray编排-00-MOC]] |
| 看 Rollout | [[03-Rollout生成-00-MOC]] |
| 看 Megatron 训练 | [[04-训练后端-00-MOC]] |
| 看权重同步 | [[05-权重同步-00-MOC]] |
| 看复盘材料 | [[Slime-90-总结复盘-00-MOC]] |
