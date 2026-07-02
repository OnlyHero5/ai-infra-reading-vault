---
type: dashboard
title: "双库联合路径"
tags:
  - dashboard
  - index
updated: 2026-07-03
---

# 双库联合路径 · AI Infra（推理 + RL）

> SGLang 覆盖 **LLM 推理 serving**；Slime 覆盖 **RL 后训练闭环**。  
> 组合阅读可建立完整 AI infra 心智模型。跨库专题跳转见 [[91_dashboard/cross-library-map|跨库专题对照]]。

---

## 架构分工

| 维度 | SGLang | Slime |
|------|--------|-------|
| 主循环 | 请求 → batch → forward → 响应 | `generate → train → update_weights` |
| 运行时 | Tokenizer + Scheduler + Detokenizer | Ray + Megatron + SGLang |
| 深度专精 | KV Cache、Attention、连续批处理 | Rollout、PPO/GRPO、权重同步 |
| 交叉接口 | CheckpointEngine 热更新 | SGLang HTTP generate + NCCL 推权重 |

---

## 路径 A · 从零双库（推荐）

| 步骤 | 目标 | 文档 | 时长 |
|:--:|------|------|------|
| 0 | serving 概念 | [[00-零基础先修]] | 1–2h |
| 1 | 推理全链路 | [[00-方法论-00-MOC]] → [[全链路请求追踪]] | 2–3h |
| 2 | SGLang 导读 | [[04-导读路径]] Step 1–10 | 3–4h |
| 3 | RL 三角 | [[Slime-00-方法论-00-MOC]] → [[08-总结与索引-01-项目总览]] | 1–2h |
| 4 | RL 全链路 | [[全链路RL训练追踪]] | 2–3h |
| 5 | Slime 导读 | [[08-总结与索引-04-导读路径]] Step 1–6 | 2–3h |
| 6+ | 接口 / 权重深潜 | [[91_dashboard/cross-library-map]] | 按需 |

**概览 1–2 天 · 深读 1–2 周**

---

## 路径 B · 已有 SGLang → 补 RL

1. [[08-总结与索引-01-项目总览]]
2. [[全链路RL训练追踪]]
3. [[91_dashboard/cross-library-map]]
4. 按需：[[03-Rollout生成-00-MOC]] · [[04-训练后端-00-MOC]] · [[05-权重同步-00-MOC]]

---

## 路径 C · 已有 Slime → 补推理

1. [[全链路请求追踪]]
2. [[04-导读路径]] Step 6–10
3. [[91_dashboard/cross-library-map]]
4. 按需：[[02-请求调度-00-MOC]] · [[04-内存与Attention-00-MOC]] · [[32-CheckpointEngine-00-MOC]]

---

## 路径 D · 决策 / 排障

| 场景 | 起点 |
|------|------|
| SGLang vs vLLM | [[08-设计追问与框架对比]] |
| 生产 serving 排障 | [[09-生产排障速查]] |
| PD / 多节点 | [[22-Disaggregation-00-MOC]] · [[09-EngineTopology-00-MOC]] |
| 权重热更新 | [[32-CheckpointEngine-00-MOC]] · [[24-WeightSync-Dist-00-MOC]] |
| 自定义 reward / rollout | [[13-RM-FilterHub-00-MOC]] · [[28-Customization-00-MOC]] |

---

## 双全链路

| 文档 | 覆盖 |
|------|------|
| [[全链路请求追踪]] | SGLang 推理 Hop 1–7 |
| [[全链路RL训练追踪]] | Slime RL Hop 1–7 |

---

## 导航

[[index]] · [[SGLang源码阅读指南]] · [[Slime源码阅读指南]] · [[91_dashboard/cross-library-map]]
