---
type: batch-doc
module: 04-Arguments-TrainRollout
batch: "04"
doc_type: concept
title: "Arguments-TrainRollout · 核心概念"
tags:
  - slime/batch/04
  - slime/module/arguments
  - slime/doc/concept
updated: 2026-07-02
---

# Arguments-TrainRollout · 核心概念

---

## 1. 三类参数中的「Slime 自身」段

本批覆盖 README 第三类：`arguments.py` 内 **Train / Rollout / Data / Algo / RM / Customization** 段（Ray 段见批次 03）。

---

## 2. Train 段核心参数

| 参数 | 作用 |
|------|------|
| `--update-weight-mode` | full / delta 权重同步 |
| `--update-weight-transport` | nccl / disk |
| `--megatron-to-hf-mode` | raw / bridge 权重格式 |
| `--only-train-params-name-list` | 正则冻结 |
| `--custom-megatron-*-path` | 训练 hook |

**Code：**

```python
# 来源：slime/utils/arguments.py L135-L144
            parser.add_argument(
                "--update-weight-mode",
                choices=["full", "delta"],
                default="full",
                help=(
                    "Weight sync strategy. 'full' (default) broadcasts every parameter "
                    "every sync. 'delta' diffs each sync against a pinned-CPU snapshot ..."
                ),
            )
```

---

## 3. Rollout 段核心参数

| 参数 | 作用 |
|------|------|
| `--hf-checkpoint` | SGLang 初始化 + tokenizer |
| `--rollout-function-path` | 整段 rollout 逻辑 |
| `--custom-generate-function-path` | 仅替换 generate 步 |
| `--rollout-temperature/top-p/...` | 采样 |
| `--partial-rollout` | 动态采样回收 |
| `--update-weights-interval` | async 稀疏 sync |

---

## 4. Customization：`*-path` 模式

**Explain：** 所有 customization 使用 **模块路径字符串**（`pkg.mod.func`），运行时 `load_function` 动态 import。

**Code：**

```python
# 来源：slime/utils/misc.py L37-L45
def load_function(path):
    module_path, _, attr = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)
```

**Comment：** 与 PyTorch dataloader 换函数同哲学（见 [[Slime-00-方法论-04-关键问题]]）。

---

## 5. 接口粒度选型（customization.md）

| 需求 | 首选参数 |
|------|---------|
| 多轮 tool / RAG / sandbox | `--custom-generate-function-path` |
| 自定义 reward / verifier | `--custom-rm-path` |
| 整段 rollout 编排不够 | `--rollout-function-path` |
| prompt 来源 / buffer | `--data-source-path` |
| 自定义 loss | `--loss-type custom_loss` + `--custom-loss-function-path` |

**Code：**

```python
# 来源：docs/en/get_started/customization.md L36-L36
# start with --custom-generate-function-path plus --custom-rm-path
```

---

## 6. SGLang 透传概念

**Explain：** 每个 SGLang `ServerArgs` CLI 在 Slime 中加 `--sglang-` 前缀；dest 常为 `sglang_*`。

**Code：**

```python
# 来源：slime/backends/sglang_utils/arguments.py L86-L91
            if isinstance(item_flag, str) and item_flag.startswith("-"):
                original_flag_stem = item_flag.lstrip("-")
                prefixed_item = f"--sglang-{original_flag_stem}"
                new_name_or_flags_list.append(prefixed_item)
```

---

## 7. Megatron 透传 + Slime validate

**Explain：** Megatron 参数由 `megatron_parse_args` 直接解析；Slime 在 `validate_args` / `_hf_validate_args` 加 RL 约束。

**Code：**

```python
# 来源：slime/backends/megatron_utils/arguments.py L72-L78
def validate_args(args):
    _megatron_validate_args(args)
    args.variable_seq_lengths = True
```

---

## 8. eval 与 rollout 函数默认关系

**Code：**

```python
# 来源：slime/utils/arguments.py L1908-L1909
    if args.eval_function_path is None:
        args.eval_function_path = args.rollout_function_path
```

---

## 9. Data 段与 rollout 步数

| 参数 | 含义 |
|------|------|
| `--rollout-batch-size` | 每步 prompt 数（required） |
| `--n-samples-per-prompt` | GRPO 每组采样数 |
| `--num-rollout` / `--num-epoch` | 训练步数 |
| `--prompt-data` | jsonl 路径 |

---

## 下一批

→ [[04-Arguments-TrainRollout-02-源码走读]]
