---
type: template
title: "Slime 批次文档模板"
tags:
  - slime/template
updated: 2026-07-02
---

# Slime 批次文档模板

> 复制本结构到各批次文件夹。**读者只读 slime_reading，不读 slime**——所有源码必须内嵌在文档中。

## 文件说明（Obsidian 唯一命名）

| 文件 | 用途 | doc_type tag |
| --------------------- | ------------------- | ------------------------ |
| `{模块名}-00-MOC.md` | 批次概述、目标、源码范围、验收标准 | `slime/doc/moc` |
| `{模块名}-01-核心概念.md` | 术语、设计动机、架构位置 | `slime/doc/concept` |
| `{模块名}-02-源码走读.md` | 按调用顺序的代码精读（**主文档**） | `slime/doc/walkthrough` |
| `{模块名}-03-数据流与交互.md` | 数据结构、消息流、模块边界 | `slime/doc/dataflow` |
| `{模块名}-04-关键问题.md` | FAQ、易错点、对比分析 | `slime/doc/faq` |
| `{模块名}-05-checkpoint.md` | 验收勾选清单 | `slime/doc/checkpoint` |

**示例（RolloutManager）：** `07-RolloutManager-00-MOC.md`、`07-RolloutManager-01-核心概念.md` …

> ⚠️ 禁止 `README.md`、`01-核心概念.md` 等泛化名。  
> ⚠️ 与 SGLang 冲突的文件须加 `Slime-` 前缀（如 `Slime-00-方法论-00-MOC.md`）。

**全批合计：≥ 15 段内嵌代码，≥ 200 行。**

### frontmatter 模板

```yaml
---
type: batch-doc
module: 07-RolloutManager
batch: "07"
doc_type: concept
title: "RolloutManager · 核心概念"
tags:
 - slime/batch/07
 - slime/module/rollout-manager
 - slime/doc/concept
updated: 2026-07-02
---
```

模块间链接用双链：`[[06-TrainActor-03-数据流与交互]]`。

---

## 写作格式：ETC 三段式（强制）

```markdown
### 3.1 训练主循环入口

**Explain：** `train.py` 的 `train()` 函数编排 Ray Placement Group、RolloutManager、
Megatron Actor，并在每个 rollout_id 上执行 generate → train → update_weights。

**Code：**

```python
# 来源：train.py L9-L27
def train(args):
    configure_logger()
    pgs = create_placement_groups(args)
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)
    actor_model.update_weights()
```

**Comment：**
- `create_rollout_manager` 必须先于 training models，用于计算 `num_rollout`
- 首次 `update_weights` 将 Megatron 初始权重推送到 SGLang 引擎
- 主循环见批次 02 源码走读 §2
```

---

## checkpoint 模板

```markdown
# {模块名} · 验收清单

## 读者自测（不打开 slime/）

- [ ] 仅读本模块 slime_reading，能口头说明本模块职责
- [ ] 能画出本模块在 generate → train → update_weights 闭环中的位置
- [ ] 能说出 3 个核心类/函数及其职责

```

---

## 禁止事项

1. ❌ 泛化文件名（`README.md`、`01-核心概念.md`）
2. ❌ Mermaid 标签内使用 `\n`（用 `<br/>`）
3. ❌ 只写「详见 `xxx.py` 第 N 行」而不贴代码
4. ❌ 代码块内伪造 Obsidian 双链
