---
type: index-doc
title: "与 SGLang 阅读对照"
tags:
  - slime/index-layer
  - slime/doc/concept
updated: 2026-07-03
---

# 与 SGLang 阅读对照

读 Slime Rollout 层时，底层推理走 SGLang：HTTP → TokenizerManager → Scheduler → ModelRunner。本页是**读者向**前置/对照入口；完整专题映射以 **[[91_dashboard/cross-library-map|跨库专题对照]]** 为唯一权威源。

---

## 架构对照

| 维度 | Slime（当前） | SGLang（推理栈） |
|------|---------------|------------------|
| 主循环 | `generate → train → update_weights` | 请求 → batch → forward → 响应 |
| 运行时 | Ray + Megatron + SGLang | Tokenizer + Scheduler + Detokenizer |
| 权重更新 | Train→Rollout NCCL / disk | CheckpointEngine 热更新 |
| 总入口 | [[Slime源码阅读指南]] | [[SGLang源码阅读指南]] |

---

## 专题对照（Slime → SGLang）

| Slime 专题 | SGLang 专题 | 衔接说明 |
|------------|-------------|----------|
| [[12-SGLang-Rollout-00-MOC]] | [[04-OpenAI-API-00-MOC]] · [[06-TokenizerManager-00-MOC]] · [[07-Scheduler-00-MOC]] | `sglang_rollout.py` 发 HTTP generate，进入推理三进程 |
| [[15-SGLang-Engine-00-MOC]] | [[02-启动链路-00-MOC]] · [[03-HTTP-Server-00-MOC]] | engine 子进程启动与 server 生命周期 |
| [[09-EngineTopology-00-MOC]] · [[16-External-Engines-00-MOC]] | [[22-Disaggregation-00-MOC]] · [[23-Distributed-00-MOC]] | PD 分离与多节点拓扑 |
| [[24-WeightSync-Dist-00-MOC]] · [[25-WeightSync-Disk-00-MOC]] · [[26-Checkpoint-M2HF-00-MOC]] | [[12-ModelLoader-00-MOC]] · [[32-CheckpointEngine-00-MOC]] | Megatron→HF 转换与权重热更新 |
| [[12-SGLang-Rollout-00-MOC]] | [[20-Sampling-00-MOC]] | sampling_params 透传至 `SamplingParams` |
| — | [[15-RadixAttention-00-MOC]] · [[16-KV-Cache-00-MOC]] · [[17-Attention-00-MOC]] | Slime 不实现 KV；Attention 深潜需回 SGLang |

---

## 全链路 Hop 对照

| Slime Hop | SGLang 对应 | 文档 |
|-----------|-------------|------|
| `generate_and_rm_group` | HTTP `/generate` | [[全链路RL训练追踪]] · [[04-OpenAI-API-02-源码走读]] |
| RolloutManager 调度 | TokenizerManager 收请求 | [[08-RolloutManager-02-源码走读]] · [[06-TokenizerManager-02-源码走读]] |
| SGLang 子进程推理 | Scheduler + ModelRunner | [[全链路请求追踪]] Hop 3–5 |
| `rollout_log_probs` | forward logprob | [[20-Sampling-03-数据流与交互]] |
| `train` + `update_weights` | —（训练侧 Slime 独有） | [[19-Train-Step-02-源码走读]] · [[24-WeightSync-Dist-02-源码走读]] |

对照阅读：**[[全链路RL训练追踪]]**（Hop 4 嵌入）↔ **[[全链路请求追踪]]**。

---

## 推荐阅读路径

**读 Rollout 前补 SGLang：**

1. [[00-零基础先修]] — Prefill/Decode、KV Cache、三进程
2. [[全链路请求追踪]] — HTTP 全链路 7 hop
3. [[04-导读路径]] — 15 步精简导读（按需）

**已有 Slime、补推理栈：** [[全链路请求追踪]] → [[04-导读路径]]

**从零双库：** [[91_dashboard/dual-library-path|双库联合路径]]

**专题级跳转：** [[91_dashboard/cross-library-map|跨库专题对照]]

---

## 参数与权重

Slime CLI 中 `--sglang-*` 经 `sglang_parse_args()` 映射为 SGLang `ServerArgs` → [[04-Arguments-TrainRollout-02-源码走读]] · [[03-HTTP-Server-01-核心概念]]

`update_weight_from_distributed` 触发 SGLang CheckpointEngine / weight sync API → [[24-WeightSync-Dist-01-核心概念]] · [[32-CheckpointEngine-01-核心概念]]

`--colocate` 模式下 tensor 直传，与 SGLang 同进程权重共享场景相关 → [[09-EngineTopology-01-核心概念]]
