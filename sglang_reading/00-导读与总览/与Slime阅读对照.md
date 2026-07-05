---
type: index-doc
title: "与 Slime 阅读对照"
tags:
  - sglang/index-layer
  - sglang/doc/concept
updated: 2026-07-03
---

# 与 Slime 阅读对照

读完 SGLang 推理栈后，若继续 RL 后训练闭环，Slime 会把 SGLang 当作 Rollout 推理引擎。本页是**读者向**续读入口；完整专题映射以 **[[91_dashboard/cross-library-map|跨库专题对照]]** 为唯一权威源。

---

## 架构对照

| 维度 | SGLang（你已读） | Slime（续读） |
|------|------------------|---------------|
| 主循环 | 请求 → batch → forward → 响应 | `generate → train → update_weights` |
| 运行时 | Tokenizer + Scheduler + Detokenizer | Ray + Megatron + SGLang |
| 权重更新 | CheckpointEngine 热更新 | Train→Rollout NCCL / disk |
| 总入口 | [[SGLang源码阅读指南]] | [[Slime源码阅读指南]] |

---

## 专题对照（SGLang → Slime）

| SGLang 专题 | Slime 专题 | 衔接说明 |
|-------------|------------|----------|
| [[04-OpenAI-API-00-MOC]] · [[06-TokenizerManager-00-MOC]] · [[07-Scheduler-00-MOC]] | [[12-SGLang-Rollout-00-MOC]] | HTTP `/generate` 进入推理栈；Slime 通过 `sglang_rollout.py` 调用 |
| [[02-启动链路-00-MOC]] · [[03-HTTP-Server-00-MOC]] | [[15-SGLang-Engine-00-MOC]] | engine 启动与 server 生命周期 |
| [[22-Disaggregation-00-MOC]] · [[23-Distributed-00-MOC]] | [[09-EngineTopology-00-MOC]] · [[16-External-Engines-00-MOC]] | PD 拓扑与多节点部署 |
| [[12-ModelLoader-00-MOC]] · [[32-CheckpointEngine-00-MOC]] | [[24-WeightSync-Dist-00-MOC]] · [[25-WeightSync-Disk-00-MOC]] · [[26-Checkpoint-M2HF-00-MOC]] | 权重格式转换与热更新 |
| [[20-Sampling-00-MOC]] | [[12-SGLang-Rollout-00-MOC]] | `SamplingParams` 与 sampling_params 透传 |
| [[15-RadixAttention-00-MOC]] · [[16-KV-Cache-00-MOC]] · [[17-Attention-00-MOC]] | — | Slime 不实现 KV；纯推理深潜留在 SGLang |

---

## 全链路 Hop 对照

| SGLang Hop | Slime 对应 | 文档 |
|------------|------------|------|
| HTTP `/generate` | `generate_and_rm_group` | [[全链路请求追踪]] · [[12-SGLang-Rollout-02-源码走读]] |
| TokenizerManager | rollout 请求组装 | [[06-TokenizerManager-02-源码走读]] |
| Scheduler + ModelRunner | SGLang 子进程（Slime 不介入） | [[07-Scheduler-02-源码走读]] |
| 响应 / logprob | `rollout_log_probs` | [[20-Sampling-03-数据流与交互]] |
| — | `train` + `update_weights` | [[全链路RL训练追踪]] |

Slime RL 全链路 Hop 4 嵌入 SGLang 推理栈；对照阅读：**[[全链路请求追踪]]** ↔ **[[全链路RL训练追踪]]**。

---

## 推荐阅读路径

**已有 SGLang 基础：**

1. [[Slime-01-项目总览]] — Slime 三角架构
2. [[全链路RL训练追踪]] — `parse_args` → generate → train → update_weights
3. [[Slime-04-导读路径]] — 12 步精简导读

**从零双库：** [[91_dashboard/dual-library-path|双库联合路径]]

**专题级跳转：** [[91_dashboard/cross-library-map|跨库专题对照]]

---

## 参数与权重

Slime `--sglang-*` 参数经 `sglang_parse_args()` 注入 SGLang `ServerArgs` → [[03-HTTP-Server-01-核心概念]] · [[04-Arguments-TrainRollout-02-源码走读]]

权重同步：`update_weight_from_distributed` 对接 CheckpointEngine → [[32-CheckpointEngine-01-核心概念]] · [[24-WeightSync-Dist-01-核心概念]]
