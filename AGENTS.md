# 源码阅读 Vault — AI Agent Orientation

> 面向 AI 代理（Cursor、Claude 等）。首次进入请先读本文件与 [[index]]。

---

## 1. Vault 概览

| 子库 | 目录 | 内容 |
|------|------|------|
| **SGLang** | `sglang_reading/` | LLM 推理 serving |
| **Slime** | `slime_reading/` | RL 后训练闭环 |
| **双库导航** | `91_dashboard/` | 联合路径、跨库对照、Dataview |
| **维护** | `90_meta/` | 规范、脚本、目录映射 |

`sglang/`、`slime/` 为 upstream 对照目录，**读者日常不必打开**。

---

## 2. 目录边界

| 目录 | 用途 | AI 可写？ |
|------|------|-----------|
| `sglang_reading/` | SGLang 阅读笔记 | ✅ |
| `slime_reading/` | Slime 阅读笔记 | ✅ |
| `sglang/` · `slime/` | upstream | ⚠️ 只读 |
| `90_meta/` | 规范与工具 | ✅ |
| `91_dashboard/` | 跨库视图 | ✅ |
| `.obsidian/` | Obsidian 配置 | ❌ 勿改（用户明确要求图谱修复时除外） |

### 读者资产 vs 维护资产

| 读者看 | 维护者看 |
|--------|----------|
| `index.md`、`*源码阅读指南.md`、各专题 MOC/01–04 | `90_meta/*-module-dir-map.md` |
| `91_dashboard/dual-library-path` | `*/PLAN.md`、`*progress*` |
| `91_dashboard/cross-library-map` | `slime_reading/AGENT-DISPATCH.md` 等 |

**禁止**在读者入口（index、指南、README）暴露内部编号、派工表、目录↔编号映射。

---

## 3. 启动协议

1. **[[AGENTS]]**（本文件）
2. **[[index]]**
3. **[[SGLang源码阅读指南]]** 或 **[[Slime源码阅读指南]]**
4. 双库：**[[91_dashboard/dual-library-path]]** · **[[91_dashboard/cross-library-map]]**
5. **[[90_meta/obsidian-syntax-rules]]**
6. **[[90_meta/obsidian-graph-presets]]**
7. 深入 [[04-导读路径]] 或 [[08-总结与索引-04-导读路径]]

---

## 4. 边界规则

### ❌ 禁止

- 修改 `.obsidian/`（用户明确要求除外）
- 删除已发布笔记（除非用户要求）
- 批量重命名专题目录（除非用户要求）
- 编造源码行为、行号、函数签名
- 在读者向文档写内部编号、派工、中间产物说明

### ⚠️ 谨慎

- 修改 `_TEMPLATE/`、`PLAN.md` 结构
- 相对链接批量改双链

### ✅ 自由

- 编辑 `sglang_reading/`、`slime_reading/` 笔记
- 更新 index、dashboard、交叉对照
- 运行 `90_meta/fix_mermaid_newlines.py`

---

## 5. 写作规范

- 正文中文；英文限标识符与术语
- **Explain → Code → Comment**（见各库 `PLAN.md`）
- 架构图：Mermaid 或 ASCII；Mermaid 换行用 `<br/>`

---

## 6. Obsidian

| 规则 | 说明 |
|------|------|
| 唯一文件名 | `{模块名}-{文档类型}.md` |
| tags | `sglang/...` 或 `slime/...` + `doc/类型` |
| 双链 | `[[07-Scheduler-01-核心概念]]` |
| 图谱 | `.obsidian/graph.json`；[[91_dashboard/graph-hub]] |
| 双库 | [[91_dashboard/dual-library-path]] · [[91_dashboard/cross-library-map]] |

完整语法：[[90_meta/obsidian-syntax-rules]]

---

## 7. 维护检查

- [ ] 文件名全局唯一（Slime 冲突项用 `Slime-` 前缀）
- [ ] frontmatter 含 tags
- [ ] Mermaid 无 `\n`
- [ ] 每篇一个 H1
- [ ] 读者文档无内部编号树、无论文库/OPD 话术

---

*最后更新: 2026-07-03 — 读者/维护分层、跨库 canonical 对照*
