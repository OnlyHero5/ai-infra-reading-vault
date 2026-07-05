---
type: index
title: "源码阅读 Vault"
status: done
tags:
  - index
  - sglang
  - slime
  - flash-attn
  - source-reading
updated: 2026-07-04
---

# 源码阅读 Vault

> **SGLang** — LLM 推理 serving 自包含中文讲解  
> **Slime** — RL 后训练闭环自包含中文讲解  
> **FlashAttention** — IO-aware attention kernel 原理与源码讲解

---

## 联合路径（推理 + RL + Kernel）

完整路径见 **[[91_dashboard/dual-library-path|AI Infra 联合路径]]**。

| 步骤 | 目标 | 起点 |
|:--:|------|------|
| 0 | serving 概念 | [[00-零基础先修]] |
| 1 | 推理全链路 | [[全链路请求追踪]] |
| 2 | RL 全链路 | [[全链路RL训练追踪]] |
| 3 | Attention kernel 原理 | [[FlashAttention源码阅读指南]] |
| 4 | 跨库专题对照 | [[91_dashboard/cross-library-map|跨库专题对照]] |

**从零三层：** [[00-方法论-00-MOC]] → [[全链路请求追踪]] → [[FlashAttention源码阅读指南]] → [[Slime-00-方法论-00-MOC]] → [[全链路RL训练追踪]]

**补 Attention kernel：** [[FlashAttention-00-零基础先修]] → [[FA01-Attention-IO-00-MOC]] → [[FA02-Online-Softmax-00-MOC]] → [[FlashAttention-全链路Attention追踪]]

**已有 SGLang：** [[Slime-01-项目总览]] → [[全链路RL训练追踪]]

**已有 Slime：** [[全链路请求追踪]] → [[04-导读路径]]

---

## 快速入口

### 共用

| 用途 | 链接 |
|------|------|
| AI Infra 联合路径 | [[91_dashboard/dual-library-path]] |
| 跨库专题对照 | [[91_dashboard/cross-library-map]] |
| 可视化入口 | [[91_dashboard/home]] |
| 关系图谱 | [[90_meta/obsidian-graph-presets]] · [[91_dashboard/graph-hub]] |
| 专题统计 | [[91_dashboard/batch-stats|专题统计]] |
| 文档类型分布 | [[91_dashboard/doc-type-map]] |

### SGLang（推理）

| 用途 | 链接 |
|------|------|
| 总索引 | [[SGLang源码阅读指南]] |
| 零基础 | [[00-零基础先修]] |
| 导读（15 步） | [[04-导读路径]] |
| HTTP 全链路 | [[全链路请求追踪]] |

### Slime（RL 后训练）

| 用途 | 链接 |
|------|------|
| 总索引 | [[Slime源码阅读指南]] |
| 零基础（Ray / Megatron） | [[Slime-00-零基础先修]] |
| 项目总览 | [[Slime-01-项目总览]] |
| 导读（12 步） | [[Slime-04-导读路径]] |
| RL 全链路 | [[全链路RL训练追踪]] |

### FlashAttention（Attention Kernel）

| 用途 | 链接 |
|------|------|
| 总索引 | [[FlashAttention源码阅读指南]] |
| 零基础（Attention / GPU） | [[FlashAttention-00-零基础先修]] |
| 导读（16 步） | [[FlashAttention-04-导读路径]] |
| 全链路 Attention 追踪 | [[FlashAttention-全链路Attention追踪]] |

---

## 按主题浏览

### SGLang

| 主题 | 入口 |
|------|------|
| 导读与总览 | [[00-导读与总览-00-MOC]] |
| 方法论 | [[00-方法论-00-MOC]] |
| 启动与入口 | [[01-启动与入口-00-MOC]] |
| 请求调度 | [[02-请求调度-00-MOC]] |
| 模型执行 | [[03-模型执行-00-MOC]] |
| 内存与 Attention | [[04-内存与Attention-00-MOC]] |
| 高级特性 | [[05-高级特性-00-MOC]] |
| 扩展组件 | [[06-扩展组件-00-MOC]] |
| 总结复盘 | [[90-总结复盘-00-MOC]] |

### Slime

| 主题 | 入口 |
|------|------|
| 导读与总览 | [[Slime-00-导读与总览-00-MOC]] |
| 方法论 | [[Slime-00-方法论-00-MOC]] |
| 启动与入口 | [[Slime-01-启动与入口-00-MOC]] |
| Ray 编排 | [[02-Ray编排-00-MOC]] |
| Rollout 生成 | [[03-Rollout生成-00-MOC]] |
| 训练后端 | [[04-训练后端-00-MOC]] |
| 权重同步 | [[05-权重同步-00-MOC]] |
| 高级特性 | [[06-高级特性-00-MOC]] |
| 扩展与生态 | [[07-扩展与生态-00-MOC]] |
| 总结复盘 | [[Slime-90-总结复盘-00-MOC]] |

### FlashAttention

| 主题 | 入口 |
|------|------|
| 导读与总览 | [[FlashAttention-00-导读与总览-00-MOC]] |
| 方法论 | [[FlashAttention-00-方法论-00-MOC]] |
| Attention IO 原理 | [[FA01-Attention-IO-00-MOC]] |
| Online Softmax | [[FA02-Online-Softmax-00-MOC]] |
| Python API 与绑定 | [[FA03-Python-API-00-MOC]] |
| FA2 CUDA Forward | [[FA04-FA2-Forward-00-MOC]] |
| KV Cache 与推理特性 | [[FA05-KV-Cache-00-MOC]] |
| FA3/FA4 Hopper/CuTe | [[FA06-Hopper-CuTe-00-MOC]] |
| 总结复盘 | [[FlashAttention-90-总结复盘-00-MOC]] |

---

*维护者与 AI 代理：见 [[AGENTS]]*
