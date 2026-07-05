---
type: guide
title: "源码阅读 Vault — AI Agent Orientation"
status: active
tags:
  - maintenance
  - agent
  - source-reading
updated: 2026-07-04
---

# 源码阅读 Vault — AI Agent Orientation

> 面向 AI 代理（Cursor、Claude 等）。首次进入请先读本文件与 [[index]]。

---

## 1. Vault 概览

| 子库 | 目录 | 内容 |
|------|------|------|
| **SGLang** | `sglang_reading/` | LLM 推理 serving |
| **Slime** | `slime_reading/` | RL 后训练闭环 |
| **FlashAttention** | `flash-attn_reading/` | Attention IO 与 CUDA/CuTe kernel |
| **联合导航** | `91_dashboard/` | 联合路径、跨库对照、Dataview |
| **维护** | `90_meta/` | 规范、脚本、目录映射 |

`sglang/`、`slime/`、`flash-attn/` 为 upstream 对照目录，**读者日常不必打开**。

---

## 2. 目录边界

| 目录 | 用途 | AI 可写？ |
|------|------|-----------|
| `sglang_reading/` | SGLang 阅读笔记 | ✅ |
| `slime_reading/` | Slime 阅读笔记 | ✅ |
| `flash-attn_reading/` | FlashAttention 阅读笔记 | ✅ |
| `sglang/` · `slime/` · `flash-attn/` | upstream | ⚠️ 只读 |
| `90_meta/` | 规范与工具 | ✅ |
| `91_dashboard/` | 跨库视图 | ✅ |
| `.obsidian/` | Obsidian 配置 | ❌ 勿改（用户明确要求图谱修复时除外） |

### 读者资产 vs 维护资产

| 读者看 | 维护者看 |
|--------|----------|
| `index.md`、`*源码阅读指南.md`、各专题 MOC/01–05 | `90_meta/obsidian-syntax-rules.md` |
| `91_dashboard/dual-library-path` | `90_meta/audit_wikilinks.mjs` |
| `91_dashboard/cross-library-map` | — |

**禁止**在读者入口（index、指南、README）暴露内部编号、派工表、目录↔编号映射。

---

## 3. 启动协议

1. **[[AGENTS]]**（本文件）
2. **[[index]]**
3. **[[SGLang源码阅读指南]]**、**[[Slime源码阅读指南]]** 或 **[[FlashAttention源码阅读指南]]**
4. 联合导航：**[[91_dashboard/dual-library-path]]** · **[[91_dashboard/cross-library-map]]**
5. **[[90_meta/obsidian-syntax-rules]]**
6. **[[90_meta/obsidian-graph-presets]]**
7. 入门：[[00-导读与总览-00-MOC]]、[[Slime-00-导读与总览-00-MOC]] 或 [[FlashAttention-00-导读与总览-00-MOC]]
8. 深入 [[04-导读路径]]、[[Slime-04-导读路径]] 或 [[FlashAttention-04-导读路径]]；收尾复盘见 [[90-总结复盘-00-MOC]]、[[Slime-90-总结复盘-00-MOC]] 或 [[FlashAttention-90-总结复盘-00-MOC]]

---

## 4. 边界规则

### ❌ 禁止

- 修改 `.obsidian/`（用户明确要求除外）
- 删除已发布笔记（除非用户要求）
- 批量重命名专题目录（除非用户要求）
- 编造源码行为、行号、函数签名
- 在读者向文档写内部编号、派工、中间产物说明

### ⚠️ 谨慎

- 修改 `_TEMPLATE/` 结构
- 相对链接批量改双链

### ✅ 自由

- 编辑 `sglang_reading/`、`slime_reading/`、`flash-attn_reading/` 笔记
- 更新 index、dashboard、交叉对照

---

## 5. 写作规范

- 正文中文；英文限标识符与术语
- **Explain → Code → Comment**（见各库 `_TEMPLATE/*模板说明.md`）
- 架构图：Mermaid 或 ASCII；Mermaid 换行用 `<br/>`

---

## 6. Obsidian

| 规则 | 说明 |
|------|------|
| 唯一文件名 | `{模块名}-{文档类型}.md` |
| tags | `sglang/...`、`slime/...` 或 `flash-attn/...` + `doc/类型` |
| 双链 | `[[07-Scheduler-01-核心概念]]` |
| 图谱 | `.obsidian/graph.json`；[[91_dashboard/graph-hub]] |
| 联合导航 | [[91_dashboard/dual-library-path]] · [[91_dashboard/cross-library-map]] |

完整语法：[[90_meta/obsidian-syntax-rules]]

---

## 7. 维护检查

- [ ] 文件名全局唯一（Slime 冲突项用 `Slime-` 前缀）
- [ ] frontmatter 含 tags
- [ ] Mermaid 无 `\n`
- [ ] 每篇一个 H1
- [ ] 读者文档无内部编号树、无论文库/OPD 话术

---

*最后更新: 2026-07-04 — 导读与总览前置、总结复盘后置*
