---
type: batch-doc
module: 04-Arguments-TrainRollout
batch: "04"
doc_type: faq
title: "Arguments-TrainRollout · 关键问题"
tags:
  - slime/batch/04
  - slime/module/arguments
  - slime/doc/faq
updated: 2026-07-02
---

# Arguments-TrainRollout · 关键问题

---

## Q1：rollout-function-path 与 custom-generate-function-path 如何选？

| 场景 | 选择 |
|------|------|
| 多轮 tool、RAG、sandbox | `--custom-generate-function-path` |
| 改采样/grpo 外层循环 | `--rollout-function-path` |
| multi-agent 编排 | 通常 `--rollout-function-path`（见 examples/multi_agent） |

**Code：**

```python
## 来源：docs/en/get_started/customization.md L36-L42
# start with --custom-generate-function-path plus --custom-rm-path
# Replace the entire rollout orchestration (only when per-sample customization is not enough)
# --rollout-function-path
```

---

## Q2：load_function 失败怎么 debug？

**Explain：** path 必须是 `module.submodule.function` 形式；模块需在 `PYTHONPATH` 可 import。

**Code：**

```python
## 来源：slime/utils/misc.py L43-L45
    module_path, _, attr = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)
```

**Comment：** 先用 `python -c "from mypkg import myfn"` 验证；再跑 `tests/plugin_contracts/` 对应文件。

---

## Q3：--sglang-mem-fraction-static 从哪来？

**Explain：** SGLang 原生 `--mem-fraction-static` → Slime CLI `--sglang-mem-fraction-static`。

**Code：**

```python
## 来源：slime/backends/sglang_utils/arguments.py L88-L91
                original_flag_stem = item_flag.lstrip("-")
                prefixed_item = f"--sglang-{original_flag_stem}"
```

---

## Q4：哪些 SGLang 参数被 skip 不暴露？

**Code：**

```python
## 来源：slime/backends/sglang_utils/arguments.py L45-L63
    skipped_args = [
        "model_path", "config", "trust_remote_code", "random_seed",
        "enable_memory_saver", "tp_size", "port", "nnodes", ...
    ]
```

**Comment：** `model_path` 由 `--hf-checkpoint` 注入；`tp_size` 由 `--rollout-num-gpus-per-engine` 推导。

---

## Q5：Megatron HF validate 失败常见原因？

**Explain：** `hidden_size` / `num_layers` / `rope_theta` 与 `--hf-checkpoint` config 不一致。

**Code：**

```python
## 来源：slime/backends/megatron_utils/arguments.py L143-L144
    if len(errors) > 0:
        raise AssertionError("hf_validate_args failed: " + "; ".join(errors))
```

**Comment：** debug rollout only 可 `skip_hf_validate`（`megatron_parse_args` 参数）。

---

## Q6：custom_loss 如何启用？

**Code：**

```python
## 来源：slime/utils/arguments.py L902-L919
            parser.add_argument(
                "--loss-type",
                choices=["policy_loss", "sft_loss", "custom_loss"],
                ...
            )
            parser.add_argument(
                "--custom-loss-function-path",
                type=str,
                default=None,
                ...
            )
```

**Comment：** 需同时设 `--loss-type custom_loss` 与 path。

---

## Q7：group RM 与 custom-rm-path？

**Code：**

```python
## 来源：docs/en/get_started/customization.md L134-L137
# Signature (batch mode, when --group-rm is enabled):
# async def batched_custom_rm(args, samples: list[Sample]) -> list[float]
```

**Code：**

```python
## 来源：slime/utils/arguments.py L1338-L1340
            parser.add_argument(
                "--group-rm", action="store_true", default=False, help="Whether to do rm on a whole group."
            )
```

---

## Q8：plugin_contracts 覆盖哪些 path？

**Code：**

```python
## 来源：docs/en/get_started/customization.md L462-L471
# test_plugin_rollout_contracts.py      → --rollout-function-path
# test_plugin_generate_contracts.py     → --custom-generate-function-path
# test_plugin_path_loading_contracts.py → eval, rm, filters, data-source, ...
# test_plugin_runtime_hook_contracts.py → log, reward-post-process, convert, postprocess
```

---

## Q9：eval 函数何时与 rollout 不同？

**Explain：** 设 `--eval-function-path`；否则 validate 复制 rollout path。

**Code：**

```python
## 来源：slime/utils/arguments.py L1908-L1909
    if args.eval_function_path is None:
        args.eval_function_path = args.rollout_function_path
```

**Comment：** eval 可单独 `--eval-temperature` 等覆盖采样参数（L817–823）。

---

## Q10：rollout_batch_size 与 global_batch_size 关系？

**Explain：** 默认 1 step/rollout 时 `global_batch_size ≈ rollout_batch_size * n_samples_per_prompt`；多步用 `--num-steps-per-rollout` 除。

**Code：**

```python
## 来源：slime/utils/arguments.py L690-L701
            reset_arg(parser, "--global-batch-size", type=int, default=None)
            parser.add_argument(
                "--num-steps-per-rollout",
                type=int,
                default=None,
                help=(
                    "Number of steps per rollout, e.g. It is equivalent to setting gbs as "
                    "`rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout`."
                ),
            )
```

---

## Q11：dynamic filter 示例 path？

**Code：**

```python
## 来源：slime/utils/arguments.py L448-L449
                    "You could use `slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std` as an example."
```

见 [[13-RM-FilterHub-02-源码走读]]。

---

## Q12：MoE routing replay 参数？

**Code：**

```python
## 来源：docs/en/get_started/customization.md L453-L456
# --use-routing-replay           Forward-backward routing consistency in training.
# --use-rollout-routing-replay   R3: Replay routing from rollout during training.
```

**Code：**

```python
## 来源：slime/utils/arguments.py L1950-L1951
    if args.use_rollout_routing_replay:
        args.use_routing_replay = True
```
