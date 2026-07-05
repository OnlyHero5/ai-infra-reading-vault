---
type: dashboard
title: "专题统计"
tags:
  - dashboard
  - sglang/meta
  - slime/meta
  - flash-attn/meta
cssclasses:
  - 91_dashboard
updated: 2026-07-04
---

# 专题统计

## SGLang · 每个专题的文档数（含 MOC + 五件套）

```dataview
TABLE WITHOUT ID
  batch AS "专题序号",
  length(rows) AS "文档数",
  rows.module[0] AS "代表模块"
FROM "sglang_reading"
WHERE type = "batch-doc" OR type = "module-moc"
GROUP BY batch
SORT number(batch) ASC
```

## SGLang · 专题 × 文档类型矩阵

```dataview
TABLE WITHOUT ID
  batch AS "专题序号",
  length(filter(rows, (r) => r.doc_type = "concept")) AS "概念",
  length(filter(rows, (r) => r.doc_type = "walkthrough")) AS "走读",
  length(filter(rows, (r) => r.doc_type = "dataflow")) AS "数据流",
  length(filter(rows, (r) => r.doc_type = "faq")) AS "FAQ",
  length(filter(rows, (r) => r.doc_type = "checkpoint")) AS "验收"
FROM "sglang_reading"
WHERE type = "batch-doc"
GROUP BY batch
SORT number(batch) ASC
```

## Slime · 每个专题的文档数（含 MOC + 五件套）

```dataview
TABLE WITHOUT ID
  batch AS "专题序号",
  length(rows) AS "文档数",
  rows.module[0] AS "代表模块"
FROM "slime_reading"
WHERE type = "batch-doc" OR type = "module-moc"
GROUP BY batch
SORT number(batch) ASC
```

## Slime · 专题 × 文档类型矩阵

```dataview
TABLE WITHOUT ID
  batch AS "专题序号",
  length(filter(rows, (r) => r.doc_type = "concept")) AS "概念",
  length(filter(rows, (r) => r.doc_type = "walkthrough")) AS "走读",
  length(filter(rows, (r) => r.doc_type = "dataflow")) AS "数据流",
  length(filter(rows, (r) => r.doc_type = "faq")) AS "FAQ",
  length(filter(rows, (r) => r.doc_type = "checkpoint")) AS "验收"
FROM "slime_reading"
WHERE type = "batch-doc"
GROUP BY batch
SORT number(batch) ASC
```

## FlashAttention · 每个专题的文档数（含 MOC + 六件套）

```dataview
TABLE WITHOUT ID
  batch AS "专题序号",
  length(rows) AS "文档数",
  rows.module[0] AS "代表模块"
FROM "flash-attn_reading"
WHERE type = "batch-doc" OR type = "module-moc" OR type = "stage-moc"
GROUP BY batch
SORT batch ASC
```

## FlashAttention · 专题 × 文档类型矩阵

```dataview
TABLE WITHOUT ID
  batch AS "专题序号",
  length(filter(rows, (r) => r.doc_type = "concept")) AS "概念",
  length(filter(rows, (r) => r.doc_type = "walkthrough")) AS "走读",
  length(filter(rows, (r) => r.doc_type = "dataflow")) AS "数据流",
  length(filter(rows, (r) => r.doc_type = "faq")) AS "FAQ",
  length(filter(rows, (r) => r.doc_type = "checkpoint")) AS "验收"
FROM "flash-attn_reading"
WHERE type = "batch-doc"
GROUP BY batch
SORT batch ASC
```

## 最近更新

```dataview
TABLE batch, doc_type, updated, file.folder AS "目录"
FROM "sglang_reading" OR "slime_reading" OR "flash-attn_reading"
WHERE type = "batch-doc" OR type = "module-moc"
SORT updated DESC
LIMIT 15
```
