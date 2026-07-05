---
type: maintenance-report
title: "双库笔记覆盖度审计 · 2026-07-04"
status: done
tags:
  - maintenance
  - audit
  - sglang/meta
  - slime/meta
updated: 2026-07-04
---

# 双库笔记覆盖度审计 · 2026-07-04

> 维护者文档。用于判断 `sglang_reading/` 与 `slime_reading/` 是否全面、详细、小白可读，并记录本轮整理动作。

## 结论

| 维度 | SGLang | Slime |
|------|--------|-------|
| 核心覆盖 | 达标：serving 主链路与核心子系统完整 | 达标：RL 主闭环与 SGLang 接口完整 |
| 详细程度 | 较厚：30 个完整专题，约 180.6 万字符 | 充足：28 个完整专题，约 111.9 万字符 |
| 小白可读 | 强：有 [[00-零基础先修]]、全链路、导读、用户故事 | 已补强：有 [[Slime-00-零基础先修]]、全链路和导读 |
| 图谱/Dataview | 本轮后通过基础检查 | 本轮后通过基础检查 |

总体判断：当前 vault 已经覆盖 SGLang serving 与 Slime RL 后训练的核心路径，可以作为系统性源码阅读材料。SGLang 与 Slime 均已有零基础入口；Slime 侧先修重点补 Ray 编排、Megatron 并行与 RL 闭环的基本心智模型。

## 客观统计

| 库 | Markdown 篇数 | 完整专题数 | 六件套完整率 | 字符量 | 平均专题字符量 |
|----|---------------|------------|--------------|--------|----------------|
| SGLang | 215 | 30 | 30 / 30 | 1,806,506 | 51,808 |
| Slime | 198 | 28 | 28 / 28 | 1,119,319 | 35,209 |

六件套指：MOC、核心概念、源码走读、数据流与交互、关键问题、checkpoint。

## 核心覆盖判断

SGLang 已覆盖：

- 启动入口：CLI、HTTP Server、OpenAI API、gRPC。
- 请求调度：TokenizerManager、Scheduler、SchedulePolicy、ScheduleBatch、Detokenizer。
- 模型执行：ModelRunner、ModelLoader、通用/专用模型适配。
- 内存与 Attention：RadixAttention、KV Cache、Attention backend、MoE、Quantization。
- 高级与扩展：Sampling、Speculative、Disaggregation、Distributed、Observability、CheckpointEngine、Multimodal、LoRA、sgl-kernel、model-gateway、Frontend lang。
- 导读与复盘层：零基础先修、全链路请求追踪、导读路径、文件地图、用户故事、复杂度热点、生产排障。

Slime 已覆盖：

- 启动入口：训练主循环、Ray 参数、Train/Rollout 参数、数据准备工具。
- Ray 编排：PlacementGroup、RayTrainGroup。
- Rollout：RolloutManager、EngineTopology、Sample 契约、DataSource、SGLang Rollout、RM/FilterHub、替代 Rollout、SGLang Engine、External Engines。
- 训练后端：Megatron Actor 初始化、Model 初始化、Train Step、Train Data、Advantage、Policy Loss、CP/Routing Replay。
- 权重同步：NCCL 分布式同步、磁盘同步、Megatron 到 HF checkpoint。
- 高级与生态：Agent trajectory、Customization、插件与 examples。
- 导读与复盘层：项目总览、架构分层、关键概念、导读路径、RL 全链路追踪、与 SGLang 对照、可观测与 CI。

## 小白可读性判断

做得好的部分：

- 两库都有总入口、阶段 MOC、全链路追踪与导读路径。
- 核心专题普遍采用 Explain → Code → Comment，并嵌入源码片段。
- 多个核心模块有用户故事开场，例如 Scheduler、KV Cache、WeightSync-Dist。
- 双库路径和跨库对照能把 “SGLang 推理” 与 “Slime Rollout/权重同步” 接起来。

需要继续补强的部分：

- Slime 已新增 [[Slime-00-零基础先修]]；后续可继续把 PPO/GRPO 公式、reward shaping、critic/value 等内容拆成更细的“算法先修”。
- SGLang 的高级边角目录仍可加索引说明，如 `compilation`、`elastic_ep`、`eplb`、`function_call`、`parser`、`session`。这些不是主链路必需，但会影响“全面到高级细节”的评价。
- Slime 的 examples 生态可以继续加厚：`fully_async`、`delta_weight_sync`、`eval_multi_task`、`multi_agent`、`search-r1`、`tau-bench` 等适合做短导读。

## 本轮整理

- 去除 22 个 Slime 模块 MOC 文件开头的 BOM，使 frontmatter 可被解析。
- 修复 [[90_meta/obsidian-graph-presets]] 中裸 `-path:sglang` / `-path:slime` 过滤，避免误排除 `sglang_reading` / `slime_reading`。
- 修复 [[04-导读路径]] 中旧式裸链接 `04-关键问题`，改为唯一文件名 [[90-总结复盘-04-关键问题]]。
- 扩展 [[91_dashboard/batch-stats]] 与 [[91_dashboard/doc-type-map]] 为双库统计视图。
- 将两库入口重排为 `00-导读与总览`、专题阶段、`90-总结复盘`，避免入门材料落在收尾目录。

## 复查结果

- frontmatter：读者目录与 dashboard/meta 文档无缺失。
- tags：无缺失。
- 每篇一个 H1：读者目录通过；模板文件不计入。
- Mermaid：实际 Mermaid 块无 `\n` 字面换行风险。
- WikiLink：读者文档无真实断链；剩余命中仅为规范示例或模板占位。

## 后续优先级

1. 在 [[Slime-90-总结复盘-03-未独立成专题导读]] 中补 examples 生态索引，先覆盖 fully async、delta weight sync、多任务 eval、多 agent。
2. 在 [[90-总结复盘-05-未独立成专题导读]] 中补 SGLang 高级边角目录索引，区分主链路、生产特性和实验性组件。
3. 抽查 5-8 篇最厚源码走读，压缩过长上游注释，增加“读这一段要抓什么”的中文提示。
