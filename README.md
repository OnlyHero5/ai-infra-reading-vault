---
type: readme
title: "AI Infra Reading Vault"
status: active
tags:
  - index
  - source-reading
  - sglang
  - slime
  - flash-attn
updated: 2026-07-04
---

# AI Infra Reading Vault

**中文 LLM 系统 / AI infra 源码阅读知识库** — 推理 serving（SGLang）+ RL 后训练（Slime）+ attention kernel（FlashAttention），Obsidian 自包含笔记。

> 中文自包含源码阅读 · **推理 serving + RL 后训练 + Attention Kernel**  
> [SGLang](https://github.com/sgl-project/sglang) · [Slime](https://github.com/THUDM/slime) · [FlashAttention](https://github.com/Dao-AILab/flash-attention) · Obsidian

[![AI Infra](https://img.shields.io/badge/AI_Infra-Reading_Vault-2563eb)](#这是什么)
[![SGLang](https://img.shields.io/badge/SGLang-serving-10b981)](#sglang--推理-serving)
[![Slime](https://img.shields.io/badge/Slime-RL_后训练-8b5cf6)](#slime--rl-后训练)
[![FlashAttention](https://img.shields.io/badge/FlashAttention-kernel-f97316)](#flashattention--attention-kernel)
[![Obsidian](https://img.shields.io/badge/Obsidian-Ready-7c3aed)](#快速开始)

---

## 这是什么

面向 **LLM 系统 / AI infra 工程师** 的 Obsidian 知识库：用中文把 upstream 源码「讲进笔记里」，读者日常**不必**打开 `sglang/`、`slime/`、`flash-attn/` 对照目录。

| 轴线 | 子库 | 框架 | 你学到什么 |
|------|------|------|------------|
| **推理** | [`sglang_reading/`](sglang_reading/) | SGLang | HTTP → 调度 → KV Cache → 模型执行 → 分布式 serving |
| **训练** | [`slime_reading/`](slime_reading/) | Slime | `generate → train → update_weights` RL 闭环 |
| **算子** | [`flash-attn_reading/`](flash-attn_reading/) | FlashAttention | Attention IO → online softmax → CUDA kernel → KV cache → FA3/FA4 |
| **组合** | [`91_dashboard/`](91_dashboard/) | — | 联合路径、跨库专题对照、Dataview 仪表盘 |

每篇笔记采用 **Explain → Code → Comment**，内嵌源码片段并标注 upstream 路径与行号。

| 框架 | 笔记基线 commit |
|------|-----------------|
| SGLang | `70df09b` |
| Slime | `22cdc6e1` |
| FlashAttention | `002cce0` |

---

## 快速开始

### 克隆

```bash
git clone git@github.com:OnlyHero5/ai-infra-reading-vault.git
cd ai-infra-reading-vault
```

HTTPS：

```bash
git clone https://github.com/OnlyHero5/ai-infra-reading-vault.git
```

### 用 Obsidian 打开

1. 安装 [Obsidian](https://obsidian.md/)
2. **打开文件夹作为仓库** → 选择克隆下来的根目录
3. 从 [`index.md`](index.md) 进入

### 推荐阅读路径

| 你想… | 从这里开始 |
|--------|------------|
| 总览导航 | [`index.md`](index.md) |
| **推理 + RL + Kernel 一条线读完** | [`91_dashboard/dual-library-path.md`](91_dashboard/dual-library-path.md) |
| 跨库专题跳转 | [`91_dashboard/cross-library-map.md`](91_dashboard/cross-library-map.md) |
| 只读 SGLang | [`sglang_reading/SGLang源码阅读指南.md`](sglang_reading/SGLang源码阅读指南.md) |
| 零基础（serving 概念） | [`sglang_reading/00-导读与总览/00-零基础先修.md`](sglang_reading/00-导读与总览/00-零基础先修.md) |
| HTTP 请求全链路 | [`sglang_reading/00-导读与总览/全链路请求追踪.md`](sglang_reading/00-导读与总览/全链路请求追踪.md) |
| 只读 Slime | [`slime_reading/Slime源码阅读指南.md`](slime_reading/Slime源码阅读指南.md) |
| Slime 零基础（Ray / Megatron） | [`slime_reading/00-导读与总览/Slime-00-零基础先修.md`](slime_reading/00-导读与总览/Slime-00-零基础先修.md) |
| RL 训练全链路 | [`slime_reading/00-导读与总览/全链路RL训练追踪.md`](slime_reading/00-导读与总览/全链路RL训练追踪.md) |
| 只读 FlashAttention | [`flash-attn_reading/FlashAttention源码阅读指南.md`](flash-attn_reading/FlashAttention源码阅读指南.md) |
| Attention IO 原理 | [`flash-attn_reading/20-原理基础/FA01-Attention-IO/FA01-Attention-IO-00-MOC.md`](flash-attn_reading/20-原理基础/FA01-Attention-IO/FA01-Attention-IO-00-MOC.md) |
| 全链路 Attention 追踪 | [`flash-attn_reading/00-导读与总览/FlashAttention-全链路Attention追踪.md`](flash-attn_reading/00-导读与总览/FlashAttention-全链路Attention追踪.md) |
| 生产 serving 排障 | [`sglang_reading/90-总结复盘/90-总结复盘-03-生产排障速查.md`](sglang_reading/90-总结复盘/90-总结复盘-03-生产排障速查.md) |

---

## SGLang · 推理 serving

入口：[`SGLang源码阅读指南.md`](sglang_reading/SGLang源码阅读指南.md)

按主题进入：导读与总览 → 启动与入口 → 请求调度 → 模型执行 → 内存与 Attention → 高级特性 → 扩展组件 → 总结复盘（见 [`index.md`](index.md) 阶段 MOC 表）。

---

## Slime · RL 后训练

入口：[`Slime源码阅读指南.md`](slime_reading/Slime源码阅读指南.md)

Slime 以 SGLang 为 Rollout 引擎；读 Rollout / 权重同步专题时，可配合 [`cross-library-map`](91_dashboard/cross-library-map.md) 跳回推理栈。

---

## FlashAttention · Attention Kernel

入口：[`FlashAttention源码阅读指南.md`](flash-attn_reading/FlashAttention源码阅读指南.md)

FlashAttention 以原理优先方式讲解 attention memory wall、online softmax、FA2 CUDA forward、KV cache 推理路径，以及 FA3/FA4 Hopper/CuTe 演进。

---

## 仓库布局

```
ai-infra-reading-vault/
├── README.md                 ← 本文件
├── index.md                  ← Obsidian 首页
├── AGENTS.md                 ← AI 代理 / 维护者指南
├── sglang_reading/           ← SGLang 阅读笔记
├── slime_reading/            ← Slime 阅读笔记
├── flash-attn_reading/       ← FlashAttention 阅读笔记
├── 91_dashboard/             ← 联合导航与可视化
└── 90_meta/                  ← 规范与维护脚本
```

**可选：** 在根目录 clone upstream 用于对照（已在 `.gitignore` 中排除，不会进版本库）：

```bash
git clone https://github.com/sgl-project/sglang.git sglang
git -C sglang checkout 70df09b

git clone https://github.com/THUDM/slime.git slime
git -C slime checkout 22cdc6e1

git clone https://github.com/Dao-AILab/flash-attention.git flash-attn/flash-attention
git -C flash-attn/flash-attention checkout 002cce0
```

---

## 维护者与 AI 代理

| 文档 | 用途 |
|------|------|
| [`AGENTS.md`](AGENTS.md) | Vault 边界、启动协议 |
| [`90_meta/obsidian-syntax-rules.md`](90_meta/obsidian-syntax-rules.md) | 命名、双链、frontmatter |

---

## 相关链接

- [SGLang](https://github.com/sgl-project/sglang) · [文档](https://docs.sglang.ai/)
- [Slime](https://github.com/THUDM/slime)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)

---

## License

阅读笔记为个人学习整理。SGLang、Slime 源码版权归各自 upstream 项目所有。
