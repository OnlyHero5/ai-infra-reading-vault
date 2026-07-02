---
type: map
title: "Obsidian 图谱过滤预设"
aliases:
 - "Graph Presets"
 - "图谱过滤预设"
doc_type: concept
tags:
 - map
 - obsidian
 - graph
 - meta
updated: 2026-07-03
---

# Obsidian 图谱过滤预设

> **默认配置已写入** `.obsidian/graph.json`。颜色图例见 [[91_dashboard/graph-hub]]。

## 推荐预设（复制到 Graph 搜索框）

### 1. 模块 MOC 主干图（首选默认）

```text
tag:#sglang/doc/moc OR tag:#sglang/stage-moc -path:_TEMPLATE -path:sglang
```

用途：32 个模块的入口页（`*-MOC`），看清阅读体系骨架，节点数约 30。

### 2. 核心概念层

```text
tag:#sglang/doc/concept -path:_TEMPLATE
```

用途：各模块「是什么」——节点标签如 `07-Scheduler-核心概念`、`15-RadixAttention-核心概念`。

### 3. 数据流 / 交互层

```text
tag:#sglang/doc/dataflow -path:_TEMPLATE
```

用途：ZMQ、HTTP、GPU 边界与 IO 结构。

### 4. 源码走读层（最深）

```text
tag:#sglang/doc/walkthrough -path:_TEMPLATE
```

用途：各模块主文档；节点较密，建议配合 Local Graph。

### 5. 按阶段看调度栈

```text
path:sglang_reading/02-请求调度 tag:#sglang/doc/concept
```

可替换为 `01-启动与入口`、`04-内存与Attention` 等阶段目录。

### 6. 索引与 onboarding（总结与索引）

```text
tag:#sglang/index-layer
```

用途：全链路追踪、导读路径、术语表等总结层。

### 7. 排除 checkpoint 与噪声

```text
path:sglang_reading -tag:#sglang/doc/checkpoint -path:_TEMPLATE -path:sglang
```

用途：日常阅读关系图；去掉验收清单节点。

### 8. 全库可读图（仍不含 upstream 源码）

```text
-path:sglang -path:90_meta -path:_TEMPLATE
```

## 图谱颜色分组（已写入 `.obsidian/graph.json`）

使用 **frontmatter 属性**着色（`[doc_type:concept]`），避免 `tag:#sglang/doc/xxx` 斜杠被 Obsidian 截断。

| 查询 | 颜色 | 含义 |
|------|------|------|
| `[type:index-doc]` | 金 | 总结索引层 |
| `[type:stage-moc]` | 蓝 | 阶段 MOC |
| `[doc_type:moc]` | 绿 | 模块 MOC |
| `[doc_type:concept]` | 青 | 核心概念 |
| `[doc_type:dataflow]` | 橙 | 数据流 |
| `[doc_type:walkthrough]` | 紫 | 源码走读 |
| `[doc_type:faq]` | 珊瑚 | FAQ |
| `[doc_type:checkpoint]` | 灰 | 验收 |

**默认过滤：** `(path:sglang_reading OR path:slime_reading) -path:_TEMPLATE`（勿用 `-path:sglang`，会误排除 `sglang_reading`）

### 9. Slime 模块 MOC 主干

```text
tag:#slime/doc/moc -path:_TEMPLATE path:slime_reading
```

### 10. 双库全量（仍不含 upstream 源码）

```text
(path:sglang_reading OR path:slime_reading) -path:_TEMPLATE -path:sglang -path:slime
```

详情见 [[91_dashboard/graph-hub]]。

## Local Graph 枢纽

| 主题 | 起点笔记 | 深度 |
|------|----------|------|
| HTTP 七 hop | [[全链路请求追踪]] | 1–2 |
| gRPC 七 hop | [[全链路请求追踪-gRPC]] | 1–2 |
| 调度核心 | [[07-Scheduler-00-MOC]] | 1 |
| 前缀缓存 | [[15-RadixAttention-00-MOC]] | 1 |
| PD 分离 | [[22-Disaggregation-00-MOC]] | 1 |
| 五层架构 | [[00-方法论-01-核心概念]] | 1 |
| RL 闭环 | [[全链路RL训练追踪]] | 1–2 |
| 双库联合路径 | [[91_dashboard/dual-library-path]] | 1 |
| 跨库对照 | [[91_dashboard/cross-library-map]] | 1 |
| Slime 总入口 | [[Slime源码阅读指南]] | 1 |

## 维护规则

- 新建文档须唯一文件名 + frontmatter `tags`（见 [[90_meta/obsidian-syntax-rules]]）
- 代码块内 `[[` 不是链接
- 节点过多时用 tag 预设或 Local Graph
