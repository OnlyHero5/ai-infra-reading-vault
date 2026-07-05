---
type: dashboard
title: "AI Infra 联合路径"
tags:
  - dashboard
  - index
  - sglang/index-layer
  - slime/index-layer
  - flash-attn/index-layer
updated: 2026-07-04
---

# AI Infra 联合路径 · 推理 + RL + Kernel

> SGLang 覆盖 **LLM 推理 serving**；Slime 覆盖 **RL 后训练闭环**；FlashAttention 覆盖 **attention kernel / IO-aware 算子层**。  
> 组合阅读可建立从训练闭环、推理 runtime 到 GPU kernel 的 AI infra 心智模型。跨库专题跳转见 [[91_dashboard/cross-library-map|跨库专题对照]]。

---

## 架构分工

| 维度 | SGLang | Slime | FlashAttention |
|------|--------|-------|----------------|
| 主循环 | 请求 → batch → forward → 响应 | `generate → train → update_weights` | `QK^T → online softmax → PV` |
| 运行时 | Tokenizer + Scheduler + Detokenizer | Ray + Megatron + SGLang | PyTorch custom op + CUDA/CuTe kernel |
| 深度专精 | KV Cache、Attention、连续批处理 | Rollout、PPO/GRPO、权重同步 | IO-aware attention、KV cache kernel、FA3/FA4 |
| 交叉接口 | Attention backend / CheckpointEngine | SGLang HTTP generate + NCCL 推权重 | 上层模型与 serving 的 attention backend |

---

## 路径 A · 从零三层（推荐）

| 步骤 | 目标 | 文档 | 时长 |
|:--:|------|------|------|
| 0 | serving 概念 | [[00-零基础先修]] | 1–2h |
| 1 | 推理全链路 | [[00-导读与总览-00-MOC]] → [[全链路请求追踪]] | 2–3h |
| 2 | SGLang 导读 | [[04-导读路径]] Step 1–10 | 3–4h |
| 3 | Attention kernel 原理 | [[FlashAttention-00-零基础先修]] → [[FA01-Attention-IO-00-MOC]] → [[FA02-Online-Softmax-00-MOC]] | 3–4h |
| 4 | FA2 / KV cache 深入 | [[FA04-FA2-Forward-00-MOC]] → [[FA05-KV-Cache-00-MOC]] | 3–5h |
| 5 | RL / Ray / Megatron 先修 | [[Slime-00-零基础先修]] → [[Slime-00-方法论-00-MOC]] → [[Slime-01-项目总览]] | 2–3h |
| 6 | RL 全链路 | [[全链路RL训练追踪]] | 2–3h |
| 7 | Slime 导读 | [[Slime-04-导读路径]] Step 1–6 | 2–3h |
| 8+ | 接口 / 权重 / kernel 深潜 | [[91_dashboard/cross-library-map]] | 按需 |

**概览 2–3 天 · 深读 2–3 周**

---

## 路径 B · 已有 SGLang → 补 RL

1. [[Slime-00-零基础先修]]
2. [[Slime-01-项目总览]]
3. [[全链路RL训练追踪]]
4. [[91_dashboard/cross-library-map]]
5. 按需：[[03-Rollout生成-00-MOC]] · [[04-训练后端-00-MOC]] · [[05-权重同步-00-MOC]]

---

## 路径 C · 已有 Slime → 补推理

1. [[全链路请求追踪]]
2. [[04-导读路径]] Step 6–10
3. [[91_dashboard/cross-library-map]]
4. 按需：[[02-请求调度-00-MOC]] · [[04-内存与Attention-00-MOC]] · [[32-CheckpointEngine-00-MOC]]

---

## 路径 D · 已有 serving → 补 Attention Kernel

1. [[FlashAttention-00-零基础先修]]
2. [[FlashAttention-全链路Attention追踪]]
3. [[FA01-Attention-IO-00-MOC]]
4. [[FA02-Online-Softmax-00-MOC]]
5. [[FA04-FA2-Forward-00-MOC]]
6. [[FA05-KV-Cache-00-MOC]]
7. 按需：[[FA06-Hopper-CuTe-00-MOC]]

---

## 路径 E · 决策 / 排障

| 场景 | 起点 |
|------|------|
| SGLang vs vLLM | [[90-总结复盘-02-设计追问与框架对比]] |
| 生产 serving 排障 | [[90-总结复盘-03-生产排障速查]] |
| PD / 多节点 | [[22-Disaggregation-00-MOC]] · [[09-EngineTopology-00-MOC]] |
| 权重热更新 | [[32-CheckpointEngine-00-MOC]] · [[24-WeightSync-Dist-00-MOC]] |
| 自定义 reward / rollout | [[13-RM-FilterHub-00-MOC]] · [[28-Customization-00-MOC]] |
| Attention kernel 原理 | [[FA01-Attention-IO-01-核心概念]] · [[FA02-Online-Softmax-01-核心概念]] |
| Decode / KV cache kernel | [[FA05-KV-Cache-01-核心概念]] · [[FA05-KV-Cache-02-源码走读]] |

---

## 双全链路

| 文档 | 覆盖 |
|------|------|
| [[全链路请求追踪]] | SGLang 推理 Hop 1–7 |
| [[全链路RL训练追踪]] | Slime RL Hop 1–7 |
| [[FlashAttention-全链路Attention追踪]] | FlashAttention Python → C++ → CUDA kernel |

---

## 导航

[[index]] · [[SGLang源码阅读指南]] · [[Slime源码阅读指南]] · [[FlashAttention源码阅读指南]] · [[91_dashboard/cross-library-map]]
