---
type: batch-doc
module: 04-Arguments-TrainRollout
batch: "04"
doc_type: moc
title: "Arguments-TrainRollout · 专题概述"
tags:
  - slime/batch/04
  - slime/module/arguments
  - slime/doc/moc
updated: 2026-07-02
---

# Arguments-TrainRollout · 专题概述

> **专题 04** | 阶段 I | **代码热点专题**（≥400 行内嵌代码）  

---

## 本专题目标

1. 走读 `add_train_arguments` / `add_rollout_arguments` / customization `*-path` 参数
2. 理解 `load_function` 如何挂载 rollout / RM / loss hook
3. 掌握 SGLang 参数透传（`sglang_utils/arguments.py`）
4. 掌握 Megatron validate 与 HF 配置对齐（`megatron_utils/arguments.py`）
5. 对照 `docs/en/get_started/customization.md` 接口表

---

## 文档导航

| 文档 | 内容 |
|------|------|
| [[04-Arguments-TrainRollout-01-核心概念]] | Train/Rollout/customization 术语 |
| [[04-Arguments-TrainRollout-02-源码走读]] | arguments 段 + 两 backend validate |
| [[04-Arguments-TrainRollout-03-数据流与交互]] | `*-path` → load_function → 运行时 |
| [[04-Arguments-TrainRollout-04-关键问题]] | 接口选型、plugin_contracts |
| [[04-Arguments-TrainRollout-05-checkpoint]] | 验收 |

---

## 源码范围

| 模块 | 文件 | 覆盖 |
|------|------|------|
| Train | `arguments.py` `add_train_arguments` | weight sync、freeze、Megatron hooks |
| Rollout | `add_rollout_arguments` + `add_data_arguments` | rollout fn、采样、data source |
| Customization | `add_reward_model_*` + buffer + megatron plugins | 全部 `*-path` |
| SGLang | `backends/sglang_utils/arguments.py` | 透传 + validate |
| Megatron | `backends/megatron_utils/arguments.py` | parse + HF validate |
| 文档 | `docs/en/get_started/customization.md` | 接口表 |

---

## 默认 rollout 入口

**Code：**

```python
## 来源：slime/utils/arguments.py L327-L339
            parser.add_argument(
                "--rollout-function-path",
                type=str,
                default="slime.rollout.sglang_rollout.generate_rollout",
                help=(
                    "Path to the rollout generation function."
                    "`def generate_rollout(args, rollout_id, data_source, evaluation=False) -> ..."
                ),
            )
```

---

## 衔接

- 上游 Ray 参数 → [[03-Arguments-Ray-00-MOC]]
- 下游 RolloutManager 加载 → [[08-RolloutManager-01-核心概念]]
- 插件测试 → `tests/plugin_contracts/`
