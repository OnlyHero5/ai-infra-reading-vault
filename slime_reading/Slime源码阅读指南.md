---
type: index
title: "Slime 源码阅读指南"
tags:
  - slime/index
  - slime/doc/index
updated: 2026-07-03
---

# Slime 源码阅读指南

本目录是对 [Slime](https://github.com/THUDM/slime) 源码的**自包含**中文讲解。  
**读者只读 `slime_reading/` 即可。**

## Slime 是什么

| 模块 | 引擎 | 职责 |
|------|------|------|
| **Training** | Megatron-LM | RL 更新（PPO/GRPO 等） |
| **Rollout** | SGLang + Router | 生成样本 + reward |
| **Data Buffer** | data_source / Sample | 统一数据接口 |

闭环：**generate → train → update_weights**

## 快速入口

| 用途 | 文档 |
|------|------|
| 总结层总览 | [[08-总结与索引-00-MOC]] → [[08-总结与索引-01-项目总览]] |
| RL 全链路 | [[全链路RL训练追踪]] |
| 12 步导读 | [[08-总结与索引-04-导读路径]] |
| 术语 | [[Slime-术语表]] |

## 按主题进入

| 主题 | 入口 |
|------|------|
| 方法论 | [[Slime-00-方法论-00-MOC]] |
| 启动与入口 | [[Slime-01-启动与入口-00-MOC]] |
| Ray 编排 | [[02-Ray编排-00-MOC]] |
| Rollout 生成 | [[03-Rollout生成-00-MOC]] |
| 训练后端 | [[04-训练后端-00-MOC]] |
| 权重同步 | [[05-权重同步-00-MOC]] |
| 高级特性 | [[06-高级特性-00-MOC]] |
| 扩展与生态 | [[07-扩展与生态-00-MOC]] |
| 总结与索引 | [[08-总结与索引-00-MOC]] |

## SGLang 推理栈

Rollout 依赖 SGLang → [[SGLang源码阅读指南]] · [[91_dashboard/cross-library-map|跨库专题对照]] · [[91_dashboard/dual-library-path|双库联合路径]]

## 文档结构

每个专题含：MOC、核心概念、源码走读、数据流与交互、关键问题、自测清单（checkpoint）。

## 版本

内嵌代码基线：**slime `22cdc6e1`**

## 推荐阅读顺序

[[08-总结与索引-01-项目总览]] → [[全链路RL训练追踪]] → [[08-总结与索引-04-导读路径]] → 按需深入各专题 MOC。
