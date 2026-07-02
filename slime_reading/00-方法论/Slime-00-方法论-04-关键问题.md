---
type: batch-doc
module: 00-方法论
batch: "01"
doc_type: faq
title: "方法论 · 关键问题"
tags:
  - slime/batch/01
  - slime/module/methodology
  - slime/doc/faq
updated: 2026-07-02
---

# 方法论 · 关键问题

---

## Q1：推荐阅读顺序是什么？

**Explain：** 先建立三角与闭环（本批），再跟 `train.py` 主循环，然后参数/Ray，再深入 Rollout 与 Megatron。

| 阶段 | 批次 | 主题 |
|------|------|------|
| 0 | 01 | 方法论（本批） |
| I | 02–05 | 训练入口、参数、数据准备 |
| II | 06–07 | Ray PG、RayTrainGroup |
| III | 08–16 | Rollout 全栈 |
| IV | 17–23 | Megatron 训练 |
| V | 24–26 | 权重同步 |
| VI–VIII | 27–30 | Agent、插件、收官 |

**Comment：** 完整表见 [[Slime-progress]] 与 [[AGENT-DISPATCH]]。

---

## Q2：读 Slime 前需要哪些 SGLang 知识？

**Explain：** Slime 假设读者理解 SGLang **server mode**、router、以及 RL 场景的权重热更新。

| SGLang 主题 | slime_reading 对应 | sglang_reading 入口 |
|-------------|-------------------|---------------------|
| HTTP Server 启动 | [[15-SGLang-Engine-00-MOC]] | [[03-HTTP-Server-00-MOC]] |
| Scheduler / Batch | Rollout 吞吐 | [[07-Scheduler-00-MOC]] |
| 分布式 TP/EP | `--sglang-*` + engine 拓扑 | [[23-Distributed-00-MOC]] |

**Code：**

```python
# 来源：docs/en/blogs/introducing_slime.md L53-L54
# RL workloads involve tons of online sampling during training,
# which makes the inference performance crucial.
# Therefore, slime exclusively integrates SGLang ...
```

**Comment：** 不必读完 32 批 SGLang；至少走读 Server 启动 + 一次 generate HTTP 路径即可。

---

## Q3：Slime 与 veRL 有何本质区别？

| 问题 | Slime | veRL（典型） |
|------|-------|-------------|
| 训练栈 | Megatron 原生 | 常 FSDP + HybridEngine |
| 推理栈 | 深度绑定 SGLang | 多 backend 抽象 |
| 扩展方式 | `*-path` 函数 hook | Worker/RolloutWorker 子类 |
| 参数 | 透传 Megatron/SGLang CLI | 统一 config 再映射 |

**Code：**

```python
# 来源：README_zh.md L22-L24
# **从设计开始就是 native**：slime 直接透传 Megatron 参数，
# 并通过 `--sglang-` 前缀暴露当前安装版本 SGLang 支持的参数。
# 新的上游训练和 serving 优化可以直接使用，不需要在 slime 里再加一层抽象。
```

**Code：**

```python
# 来源：README_zh.md L14-L14（致谢）
# 特别感谢 ... OpenRLHF、veRL ...
# （Slime 借鉴生态但选择不同架构取舍）
```

**Comment：**

- veRL 适合 HuggingFace 权重 + 多推理后端的快速实验
- Slime 适合 **大规模 Megatron + SGLang 生产 RL**（GLM/Qwen/DeepSeek 发布路径）

---

## Q4：为什么只选 SGLang 一个 rollout backend？

**Explain：** 多 backend 框架只能暴露「公共能力子集」，会遮住 MoE EP、PD 分离、delta sync 等 SGLang 强项。

**Code：**

```python
# 来源：README.md L24-L24
# By choosing one rollout backend, slime can use SGLang-specific capabilities directly
# instead of flattening multiple inference engines into a lowest-common-denominator abstraction.
```

**Comment：** vime 证明 rollout 后端可换，但需另建适配层；Slime 主线仍 SGLang-native。

---

## Q5：Data Buffer 是独立服务吗？

**Explain：** 不是。README 的「Data Buffer」是 **架构角色名**：prompt 管理 + rollout 产出 sample 的集合。

**Code：**

```python
# 来源：README_zh.md L91-L93
# **data buffer**：桥梁模块，管理 prompt 初始化、自定义数据与 rollout 生成方法
# （包括以同一套接口产出 sample 的 agentic workflow）。
```

**Comment：** 实现上见 `RolloutDataSourceWithBuffer`、`RolloutManager.generate`（[[11-DataSource-00-MOC]]）。

---

## Q6：slime_reading 与直接读 slime/ 有何不同？

| 维度 | slime_reading | slime/ 源码 |
|------|---------------|-------------|
| 受众 | 读者不打开 upstream | 维护者 / 调试 |
| 代码 | 内嵌 + 行号标注 | 实时变更 |
| 结构 | 六件套 ETC | 仓库目录 |

**Comment：** 本库基线 commit `22cdc6e1`； upstream 升级后需 diff 核对行号。

---

## Q7：Agent RL 要不要换框架？

**Explain：** 不需要。优先 `--custom-generate-function-path` + `--custom-rm-path`。

**Code：**

```python
# 来源：docs/en/get_started/customization.md L34-L36
# Agentic workflows ... plug into slime through the existing customization interfaces;
# slime does not require a separate agent framework.
# start with --custom-generate-function-path plus --custom-rm-path
```

**Comment：** 仅当默认 `sglang_rollout` 外循环不够时，才用 `--rollout-function-path` 全替换。

---

## Q8：如何验证「读懂了」本批？

见 [[Slime-00-方法论-05-checkpoint]] 读者自测项；能不看源码口述三角 + 闭环三步即通过。
