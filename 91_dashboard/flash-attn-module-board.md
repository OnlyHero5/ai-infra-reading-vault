---
type: dashboard
title: "FlashAttention 模块总览"
tags:
  - dashboard
  - flash-attn/meta
cssclasses:
  - 91_dashboard
updated: 2026-07-04
---

# FlashAttention 模块总览

```dataview
TABLE batch AS "专题序号", title AS "标题", file.link AS "入口"
FROM "flash-attn_reading"
WHERE type = "module-moc"
SORT batch ASC
```

## 按阶段浏览

### 导读 · 方法论

```dataview
LIST
FROM "flash-attn_reading"
WHERE type = "module-moc" AND (batch = "FA00" OR batch = "FA01" OR batch = "FA02")
SORT batch ASC
```

### API · FA2 内核

```dataview
LIST
FROM "flash-attn_reading"
WHERE type = "module-moc" AND (batch = "FA03" OR batch = "FA04")
SORT batch ASC
```

### 推理 · 新架构

```dataview
LIST
FROM "flash-attn_reading"
WHERE type = "module-moc" AND (batch = "FA05" OR batch = "FA06")
SORT batch ASC
```

## 单模块六件套检查

```dataview
TABLE doc_type AS "类型", file.link AS "文档"
FROM "flash-attn_reading"
WHERE module = "FA04-FA2-Forward" AND (type = "batch-doc" OR type = "module-moc")
SORT doc_type ASC
```

> 将 `module = "FA04-FA2-Forward"` 改为其他模块名即可抽查。

## 交叉入口

| 文档 | 说明 |
|------|------|
| [[FlashAttention源码阅读指南]] | FlashAttention 总入口 |
| [[91_dashboard/cross-library-map|跨库专题对照]] | SGLang / Slime / FlashAttention 专题映射 |
| [[91_dashboard/dual-library-path|AI Infra 联合路径]] | 推理 + RL + Kernel 推荐阅读顺序 |

