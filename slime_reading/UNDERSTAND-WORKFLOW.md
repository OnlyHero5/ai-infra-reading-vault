---
type: meta-doc
title: "Slime 阅读 · Understand-Anything 工作流"
tags:
  - slime/meta
  - slime/doc/concept
updated: 2026-07-02
---

# Slime 阅读 · Understand-Anything 工作流

> 本文件是 **维护者/Agent 写作侧** 的操作手册。读者不读本文；读者只读各批次六件套正文。  
> 严格对齐 `/understand`、`/understand-explain`、`/understand-onboard`、`/understand-domain` 四个 Skill。

---

## 0. 前置条件（批次 01 写作前必做）

### 0.1 生成知识图谱

```bash
# 在 Cursor 中执行（推荐中文输出）
/understand --language zh F:/源码阅读/slime
```

**产出路径：**

| 文件 | 用途 |
|------|------|
| `slime/.understand-anything/knowledge-graph.json` | 节点/边/layers/tour 主图谱 |
| `slime/.understand-anything/meta.json` | commit hash、分析时间 |
| `slime/.understand-anything/domain-graph.json` | 业务域流程（Phase 4/5 后由 `/understand-domain` 生成） |

**已生成：** `.understand-anything/.understandignore`（写作前请 review；默认 **不** 排除 `tests/`、`examples/`，以便图谱覆盖插件契约与示例入口）。

### 0.2 生成业务域图（可选，建议批次 05 前）

```bash
/understand-domain F:/源码阅读/slime
```

从已有 `knowledge-graph.json` 推导 domain/flow/step 节点，写入 `domain-graph.json`。

### 0.3 图谱与阅读计划对照

批次 ↔ 图谱映射见 [[GRAPH-BATCH-MAP]]。每写完一批，在 `Slime-progress.md` 勾选对应 `graphNodesUpdated` 字段。

---

## 1. 每批次标准工作流（六步）

```text
Step 0  读 GRAPH-BATCH-MAP 中本批 nodeIds / tourStep / domainFlow
Step 1  /understand-explain <本批主文件>  → 草稿（角色/结构/连接/数据流）
Step 2  精读源码 → 提取 ≥15 段内嵌代码 → 写入 01–04 六件套
Step 3  对照 knowledge-graph.json layers/tour 检查遗漏节点
Step 4  （每 5 批）/understand-domain → 更新 03 或阶段 MOC 中的业务域小节
Step 5  填写 05-checkpoint → 用「只读 slime_reading」自测 → 更新 Slime-progress.md
Step 6  （每 5 批）/understand --language zh 增量更新图谱
```

---

## 2. `/understand-explain` → 六件套章节映射

| explain 输出维度 | slime_reading 文档 | 必含内容 |
|------------------|-------------------|----------|
| 架构角色（哪一层、为何存在） | `{模块}-01-核心概念.md` § 架构位置 | Mermaid 节点 + layer id |
| 内部结构（类/函数/contains） | `{模块}-02-源码走读.md` 全文 | 按调用顺序 ≥8 段代码 |
| 外部连接（imports/calls/depends_on） | `{模块}-03-数据流与交互.md` § 上下游 | 图谱 edge 类型标注 |
| 数据流（输入→处理→输出） | `{模块}-03-数据流与交互.md` § 数据流 | 逐步 + 代码 |
| 模式/idiom/复杂度 | `{模块}-04-关键问题.md` | 易错 vs 正确 + 验证建议 |

**MOC（00）额外要求：** 本批在 `knowledge-graph.json` 中的 `nodeIds` 列表 + 对应 `tour` step 标题。

---

## 3. `/understand-onboard` → 批次 30 交付映射

| onboard 章节 | 批次 30 产出文件 | 数据来源 |
|--------------|-----------------|----------|
| Project Overview | `08-总结与索引-01-项目总览.md` | `graph.project` + README |
| Architecture Layers | `08-总结与索引-02-架构分层.md` | `graph.layers[]` |
| Key Concepts | `08-总结与索引-03-关键概念.md` | `concept:` / 高 tag 节点 |
| Guided Tour | `08-总结与索引-04-导读路径.md` | `graph.tour[]` |
| File Map | `08-总结与索引-05-文件地图.md` | file-level nodes by layer |
| Complexity Hotspots | `08-总结与索引-06-复杂度热点.md` | `complexity: complex` 节点 |

**批次 30 额外：**

| 文件 | Skill | 说明 |
|------|-------|------|
| `全链路RL训练追踪.md` | — | tour 扩展版，每跳内嵌代码 |
| `Slime-业务域流程.md` | understand-domain | domain-graph 文字版 + 入口代码 |
| `Slime-模块依赖图.md` | understand layers/edges | Mermaid + import 代码 |
| `Slime-术语表.md` | — | RL 术语 + 首次出现代码片段 |
| `与SGLang阅读对照.md` | — | slime 模块 ↔ sglang_reading 批次 |
| `08-总结与索引-07-可观测与CI.md` | — | trace/profile/CI/fault-tolerance |

---

## 4. 图谱增量更新节奏

| 完成批次 | 动作 |
|----------|------|
| 05 | `/understand` 增量 + `/understand-domain` |
| 10 | 同上 |
| 15 | 同上 |
| 20 | 同上 |
| 25 | 同上 |
| 30 | `/understand --full` 全量复盘（可选）+ domain 终版 |

增量前确认 `meta.json` 中 `gitCommitHash` 与 `slime_reading/Slime-PLAN.md` 基线一致；upstream 升级时更新 Slime-PLAN 基线并标注 diff 范围。

---

## 5. 质量门禁（每批 05-checkpoint 维护者项）

- [ ] frontmatter：`slime/batch/NN` + `slime/doc/*` + `slime/module/*`
- [ ] 六件套文件名含模块前缀（禁止 `README.md` / `01-核心概念.md`）
- [ ] Mermaid 用 `<br/>`，禁止 `\n`
- [ ] 双链 `[[NN-Module-01-核心概念]]`，禁止 `./01-*.md`
- [ ] 五篇正文 ≥15 段代码、≥200 行
- [ ] MOC 列出本批 `nodeIds`（来自图谱或 GRAPH-BATCH-MAP 预规划）
- [ ] `04-关键问题` 含「验证建议」：指向 `tests/` 中对应用例（见 GRAPH-BATCH-MAP）
- [ ] 已更新 [[Slime-progress]]

---

## 6. 复杂度热点（写作时加长走读）

| 文件 | ~行数 | 建议批次 |
|------|-------|----------|
| `slime/utils/arguments.py` | 2000 | 03–04 |
| `slime/ray/rollout.py` | 1486 | 08–09 |
| `slime/backends/megatron_utils/loss.py` | 1322 | 21–22 |
| `slime/backends/megatron_utils/model.py` | 968 | 18–19 |
| `slime/backends/megatron_utils/server/megatron_server.py` | 770 | 30 |
| `slime/backends/sglang_utils/sglang_engine.py` | 691 | 15 |
| `slime/backends/megatron_utils/actor.py` | 683 | 17–19 |
| `slime/rollout/sglang_rollout.py` | 641 | 12 |
| `slime/agent/trajectory.py` | 509 | 27 |
| `slime/agent/adapters/common.py` | 524 | 27 |

热点批次 `02-源码走读` 目标 **≥400 行**内嵌代码。

---

*最后更新: 2026-07-02 — 对齐 understand-anything Phase 0–7*
