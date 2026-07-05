---
type: index-doc
title: "Slime 未独立成专题导读"
doc_type: concept
tags:
  - slime/index-layer
  - slime/doc/concept
updated: 2026-07-03
---

# Slime 未独立成专题导读

> 索引层 · 对应 slime `22cdc6e1`  
> 下列 upstream 子系统**暂无独立六件套专题**，核心逻辑已在相关专题中分散讲解。

---

## 1. 导读表

| Upstream 主题 | 为何未独立成专题 | 在本 vault 中的阅读路径 | 深度 |
|---------------|----------------|-------------------------|------|
| **megatron_server.py** | 辅助 HTTP 服务，非 train 主循环 | [[Slime-90-总结复盘-02-可观测与CI]] 摘要 · [[17-Megatron-Actor-Init-04-关键问题]] | 浅 |
| **CI / 生产可观测** | 测试 harness 非 RL 闭环 | [[Slime-90-总结复盘-02-可观测与CI]] · `tests/test_qwen3_4B_ppo.py`（导读路径 Step 12） | 浅–中 |
| **slime_plugins/rollout_buffer** | 可选插件，非默认路径 | [[29-Plugins-Examples-02-源码走读]] · [[11-DataSource-04-关键问题]] buffer 局限 | 浅 |
| **Critic-only 训练阶段** | 与 actor 共用 train loop | [[02-训练主循环-02-源码走读]] · [[19-Train-Step-01-核心概念]] | 中 |
| **train_async.py** | 异步变体 | [[14-Alt-Rollout-00-MOC]] · [[02-训练主循环-04-关键问题]] | 中 |
| **FSDP 后端** | Megatron 为主路径 | [[17-Megatron-Actor-Init-01-核心概念]] 对比表；FSDP 仅索引提及 | 浅 |

---

## 2. megatron_server — 15 分钟补课

**Explain：** 部分部署用独立 HTTP 进程暴露 Megatron 权重或 debug 接口，**不参与** `generate → train → update_weights` 主闭环。

| 顺序 | 文档 | 关注点 |
|------|------|--------|
| 1 | [[Slime-90-总结复盘-02-可观测与CI]] | 文件地图中的 `megatron_server.py` 定位 |
| 2 | [[17-Megatron-Actor-Init-01-核心概念]] | Actor 与 server 进程边界 |
| 3 | upstream `slime/backends/megatron_server.py` | 需打开 upstream 时以函数名为锚 |

---

## 3. CI 与 plugin 契约 — 20 分钟补课

**Explain：** Slime 用 CPU plugin contract 测试保证 `--*-path` 签名稳定。

| 顺序 | 文档 | 关注点 |
|------|------|--------|
| 1 | [[28-Customization-02-源码走读]] §17 | pytest 入口 |
| 2 | [[29-Plugins-Examples-04-关键问题]] | 何时 fork example vs 写 plugin |
| 3 | [[Slime-04-导读路径]] Step 12 | e2e 测试阅读 |

---

## 4. 与 SGLang 交叉主题

| 主题 | Slime 侧 | SGLang 侧 |
|------|----------|-----------|
| Rollout 推理 | [[15-SGLang-Engine-00-MOC]] | [[07-Scheduler-00-MOC]] |
| 权重热更新 | [[24-WeightSync-Dist-00-MOC]] | [[32-CheckpointEngine-00-MOC]] |
| Agent tool parse | [[28-Customization-02-源码走读]] §13 | [[04-OpenAI-API-02-源码走读]] function call |

双库对照：[[与SGLang阅读对照]] · [[91_dashboard/cross-library-map]]

---

## 导航

- [[Slime-00-导读与总览-00-MOC]]
- [[Slime-90-总结复盘-04-checkpoint]]
- [[90-总结复盘-05-未独立成专题导读]]（SGLang 侧姊妹篇）
