---
type: template
title: "Slime 批次文档模板"
tags:
  - slime/template
updated: 2026-07-05
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

## 写作前置：源码阅读门槛

修改任何读者向正文前，必须先完成两件事：

1. 用 `node 90_meta/audit_source_evidence.mjs` 或等价方式确认本文所有 `来源：...` 能定位到 `slime/` upstream。
2. 完整阅读本文引用到的 upstream 源码文件，再写解释；不能只看已有笔记、不能只看摘录片段、不能凭经验补设计动机。

若新增代码摘录，必须同步标注 `来源：路径 Lx-Ly`。若解释的是未摘录的大段上下文，应在段落中写明“源码依据来自同文件的初始化顺序 / 调用链 / 分支条件”，避免无证据的设计判断。

---

## 写作格式：设计解释型 ETC（强制）

````markdown
### 3.1 训练主循环入口

**问题与约束：** RL 后训练不是单个训练循环，而是 rollout engine、Megatron actor、Ray placement group 和权重同步之间的闭环；入口必须把资源、生成、训练、同步的先后关系固定下来。

**设计选择：** `train()` 在 driver 侧集中编排 PG、RolloutManager、training models，并用 `generate → train → update_weights` 作为主循环骨架。

**Explain：** `train.py` 的 `train()` 函数编排 Ray Placement Group、RolloutManager、Megatron Actor，并在每个 rollout_id 上执行 generate → train → update_weights。

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

**代码逻辑：**
- 先创建 PG，因为 rollout 与 training actor 都依赖资源布局。
- 再创建 RolloutManager，因为训练模型初始化需要知道 rollout 侧信息。
- 首次 `update_weights` 把训练侧初始权重同步给 rollout engine。

**为什么这样写：** Slime 的设计哲学是让 driver 保持闭环语义，让 Ray actor 承担分布式执行。入口固定顺序后，后续自定义 rollout、reward 或权重传输只是在边界内替换组件，不改变主循环形状。

**不变量与失败模式：**
- RolloutManager 必须早于 training models，否则训练侧无法回填 rollout manager 引用。
- `update_weights` 是 generate 前的同步屏障；若跳过，rollout 可能使用旧权重生成样本。

**Comment：**
- 读者应抓住：Slime 的入口不是普通 training script，而是分布式 RL 闭环的 orchestrator。
- 主循环见批次 02 源码走读 §2
````

若 `Code` 摘录为长英文 README/docstring/help text，必须在代码块后补 `**中文释义：** ...`，再进入 `Comment`。源码原文不改写，读者说明保持中文。

### 最低解释要求

每段内嵌源码至少覆盖以下四类信息中的三类；源码走读主文档原则上四类都要覆盖：

1. **问题与约束**：这段代码解决的系统压力是什么，例如资源编排、actor 生命周期、训练/生成同步、权重一致性、容错或扩展点。
2. **设计选择与取舍**：作者选择了什么结构，放弃了什么替代方案，代价是什么。
3. **代码逻辑**：按执行顺序说明分支、状态读写、ObjectRef、Ray actor 调用、样本或权重如何流动。
4. **不变量与失败模式**：哪些条件必须成立；改错后会出现权重陈旧、资源错绑、样本错配、死锁还是性能退化。

`Explain` 给读者结论，`Code` 提供证据，`代码逻辑` 拆执行过程，`为什么这样写` 讲工程哲学和取舍，`Comment` 回到阅读主线。

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
