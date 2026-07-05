---
type: dashboard
title: "文档类型分布"
tags:
  - dashboard
  - sglang/meta
  - slime/meta
  - flash-attn/meta
cssclasses:
  - 91_dashboard
updated: 2026-07-04
---

# 文档类型分布

> 与 Graph View 颜色分组一一对应（见 [[91_dashboard/graph-hub]]）。

## SGLang · 类型计数

```dataview
TABLE WITHOUT ID
  doc_type AS "类型",
  length(rows) AS "篇数",
  choice(doc_type = "moc", "🟢 模块入口", choice(doc_type = "concept", "🩵 核心概念", choice(doc_type = "walkthrough", "🟣 源码走读", choice(doc_type = "dataflow", "🟠 数据流", choice(doc_type = "faq", "🔴 FAQ", "⚪ checkpoint"))))) AS "图谱色"
FROM "sglang_reading"
WHERE type = "batch-doc" OR type = "module-moc"
GROUP BY doc_type
SORT length(rows) DESC
```

## Slime · 类型计数

```dataview
TABLE WITHOUT ID
  doc_type AS "类型",
  length(rows) AS "篇数",
  choice(doc_type = "moc", "🟢 模块入口", choice(doc_type = "concept", "🩵 核心概念", choice(doc_type = "walkthrough", "🟣 源码走读", choice(doc_type = "dataflow", "🟠 数据流", choice(doc_type = "faq", "🔴 FAQ", "⚪ checkpoint"))))) AS "图谱色"
FROM "slime_reading"
WHERE type = "batch-doc" OR type = "module-moc"
GROUP BY doc_type
SORT length(rows) DESC
```

## FlashAttention · 类型计数

```dataview
TABLE WITHOUT ID
  doc_type AS "类型",
  length(rows) AS "篇数",
  choice(doc_type = "moc", "🟢 模块入口", choice(doc_type = "concept", "🩵 核心概念", choice(doc_type = "walkthrough", "🟣 源码走读", choice(doc_type = "dataflow", "🟠 数据流", choice(doc_type = "faq", "🔴 FAQ", "⚪ checkpoint"))))) AS "图谱色"
FROM "flash-attn_reading"
WHERE type = "batch-doc" OR type = "module-moc" OR type = "stage-moc"
GROUP BY doc_type
SORT length(rows) DESC
```

## SGLang · 核心概念层（Graph 过滤：`tag:#sglang/doc/concept`）

```dataview
TABLE batch, module, title
FROM "sglang_reading"
WHERE doc_type = "concept"
SORT number(batch) ASC
```

## SGLang · 数据流层（Graph 过滤：`tag:#sglang/doc/dataflow`）

```dataview
TABLE batch, module, title
FROM "sglang_reading"
WHERE doc_type = "dataflow"
SORT number(batch) ASC
```

## SGLang · 源码走读层（Graph 过滤：`tag:#sglang/doc/walkthrough`）

```dataview
TABLE batch, module, title
FROM "sglang_reading"
WHERE doc_type = "walkthrough"
SORT number(batch) ASC
```

## Slime · 核心概念层（Graph 过滤：`tag:#slime/doc/concept`）

```dataview
TABLE batch, module, title
FROM "slime_reading"
WHERE doc_type = "concept"
SORT number(batch) ASC
```

## Slime · 数据流层（Graph 过滤：`tag:#slime/doc/dataflow`）

```dataview
TABLE batch, module, title
FROM "slime_reading"
WHERE doc_type = "dataflow"
SORT number(batch) ASC
```

## Slime · 源码走读层（Graph 过滤：`tag:#slime/doc/walkthrough`）

```dataview
TABLE batch, module, title
FROM "slime_reading"
WHERE doc_type = "walkthrough"
SORT number(batch) ASC
```

## FlashAttention · 核心概念层（Graph 过滤：`tag:#flash-attn/doc/concept`）

```dataview
TABLE batch, module, title
FROM "flash-attn_reading"
WHERE doc_type = "concept"
SORT batch ASC
```

## FlashAttention · 数据流层（Graph 过滤：`tag:#flash-attn/doc/dataflow`）

```dataview
TABLE batch, module, title
FROM "flash-attn_reading"
WHERE doc_type = "dataflow"
SORT batch ASC
```

## FlashAttention · 源码走读层（Graph 过滤：`tag:#flash-attn/doc/walkthrough`）

```dataview
TABLE batch, module, title
FROM "flash-attn_reading"
WHERE doc_type = "walkthrough"
SORT batch ASC
```
