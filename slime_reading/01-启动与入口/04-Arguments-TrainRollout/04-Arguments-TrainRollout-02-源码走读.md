---
type: batch-doc
module: 04-Arguments-TrainRollout
batch: "04"
doc_type: walkthrough
title: "Arguments-TrainRollout · 源码走读"
tags:
  - slime/batch/04
  - slime/module/arguments
  - slime/doc/walkthrough
updated: 2026-07-02
---

# Arguments-TrainRollout · 源码走读

---

## §1 add_train_arguments（权重同步段）

**Code：**

```python
## 来源：slime/utils/arguments.py L107-L155
        def add_train_arguments(parser):
            parser.add_argument(
                "--qwen-gdn-backend",
                type=str,
                choices=["fla", "flashqla"],
                default="fla",
                help="GDN implementation backend for Qwen linear-attention layers.",
            )
            parser.add_argument(
                "--train-env-vars",
                type=json.loads,
                default="{}",
                help="Extra environment variables for training process, e.g. PyTorch memory management ones.",
            )
            parser.add_argument(
                "--train-memory-margin-bytes",
                type=int,
                default=1024**3,
                help="Add margin for train memory allocation. By default we will reserve 1GB as margin.",
            )
            parser.add_argument(
                "--megatron-to-hf-mode",
                choices=["raw", "bridge"],
                default="raw",
                help="The method to convert megatron weights to hugging face weights for SGLang.",
            )
            parser.add_argument(
                "--update-weight-mode",
                choices=["full", "delta"],
                default="full",
                help=(
                    "Weight sync strategy. 'full' (default) broadcasts every parameter "
                    "every sync. 'delta' diffs each sync against a pinned-CPU snapshot of the "
                    "previous one and ships only the changed bytes (disk transport only)."
                ),
            )
            parser.add_argument(
                "--update-weight-transport",
                choices=["nccl", "disk"],
                default="nccl",
                help=(
                    "Carrier for weight sync. In full mode, 'nccl' broadcasts chunks and "
                    "'disk' writes a complete HF checkpoint under --update-weight-disk-dir "
                    "before engines reload it. Delta mode is 'disk' only ..."
                ),
            )
```

**Comment：** train 段还含 freeze/only-train regex、`--allgather-cp` 等，见 L248–300。

---

## §2 add_rollout_arguments（函数路径 + 采样）

**Code：**

```python
## 来源：slime/utils/arguments.py L304-L340
        def add_rollout_arguments(parser):
            parser.add_argument(
                "--hf-checkpoint",
                type=str,
                default=None,
                help=(
                    "The huggingface checkpoint of the trained model. "
                    "This is used to initialize sglang and also provide the tokenizer. "
                    "Note that, we will always update the parameters in sglang with that of megatron before training, "
                    "so you only need to provide a huggingface checkpoint that has the same architecture ..."
                ),
            )
            parser.add_argument(
                "--rollout-function-path",
                type=str,
                default="slime.rollout.sglang_rollout.generate_rollout",
                help=(
                    "Path to the rollout generation function."
                    "The signature of the function should be "
                    "`def generate_rollout(args, rollout_id, data_source, evaluation=False) -> RolloutFnTrainOutput | RolloutFnEvalOutput`"
                ),
            )
```

**Code：**

```python
## 来源：slime/utils/arguments.py L473-L481
            parser.add_argument(
                "--custom-generate-function-path",
                type=str,
                default=None,
                help=(
                    "Only substitue the `def generate(args, sample, sampling_params)` function within the example rollout function. "
                    "This should be useful if you need to implement some special rollout logic, e.g. multi-turn, function calling."
                ),
            )
```

**Code：**

```python
## 来源：slime/utils/arguments.py L523-L528
            parser.add_argument(
                "--update-weights-interval",
                type=int,
                default=1,
                help="Interval for updating the weights",
            )
```

---

## §3 add_data_arguments（batch 与 data source）

**Code：**

```python
## 来源：slime/utils/arguments.py L625-L688
            parser.add_argument(
                "--data-source-path",
                type=str,
                default="slime.rollout.data_source.RolloutDataSourceWithBuffer",
                help="The data source class for rollout data.",
            )
            parser.add_argument(
                "--prompt-data",
                type=str,
                default=None,
                help=("The path to the prompt data. Currently we only support jsonl format ..."),
            )
            parser.add_argument(
                "--rollout-batch-size",
                type=int,
                required=True,
                help=(
                    "The number of prompts in each rollout step. "
                    "The total data returned should be rollout_batch_size * n_samples_per_prompt. "
                ),
            )
            parser.add_argument(
                "--n-samples-per-prompt", type=int, default=1, help="Number of responses for each prompt in generation"
            )
```

---

## §4 add_eval_arguments

**Code：**

```python
## 来源：slime/utils/arguments.py L766-L775
        def add_eval_arguments(parser):
            parser.add_argument(
                "--eval-function-path",
                type=str,
                default=None,
                help=(
                    "Path to the eval generation function."
                    "If not set, we will use rollout_function_path as the default. "
                ),
            )
            reset_arg(parser, "--eval-interval", type=int, default=None)
```

---

## §5 add_algo_arguments（loss / advantage）

**Code：**

```python
## 来源：slime/utils/arguments.py L902-L944
            parser.add_argument(
                "--loss-type",
                type=str,
                choices=["policy_loss", "sft_loss", "custom_loss"],
                default="policy_loss",
                help=(
                    "Choose loss type, currently support ppo policy_loss or sft_loss, "
                    "if custom_loss is set, we will use the function path from `--custom-loss-function-path`."
                ),
            )
            parser.add_argument(
                "--custom-loss-function-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom loss function, if the loss_type is `custom_loss`, "
                    "we will use this function to calculate the loss. "
                ),
            )
            parser.add_argument(
                "--advantage-estimator",
                type=str,
                choices=[
                    "grpo",
                    "gspo",
                    "cispo",
                    "reinforce_plus_plus",
                    "reinforce_plus_plus_baseline",
                    "ppo",
                ],
                default="grpo",
                help=(
                    "Advantage estimator to use. Note: on-policy distillation (OPD) is now orthogonal "
                    "to the advantage estimator. Use --opd-kl-coef > 0 to enable OPD on top of any estimator."
                ),
            )
```

**Code：**

```python
## 来源：slime/utils/arguments.py L955-L967
            parser.add_argument(
                "--custom-advantage-function-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom advantage/returns computation function. "
                    "When set, this function replaces the built-in compute_advantages_and_returns. "
                    "Signature: def custom_fn(args, rollout_data) -> None. "
                ),
            )
```

---

## §6 add_reward_model_arguments（RM customization）

**Code：**

```python
## 来源：slime/utils/arguments.py L1347-L1374
            parser.add_argument(
                "--custom-rm-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom reward model function. "
                    "If set, we will use this function to calculate the reward instead of the default one. "
                    "The function should have the signature `def custom_rm(args, sample) -> float`."
                ),
            )
            parser.add_argument(
                "--custom-reward-post-process-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom function that will post process reward, by default it will be the normalization for grpo. "
                ),
            )
            parser.add_argument(
                "--custom-convert-samples-to-train-data-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom function that converts samples to training data. "
                    "The function should have the signature `def convert_samples_to_train_data(args, samples) -> dict`."
                ),
            )
```

---

## §7 buffer / filter 路径

**Code：**

```python
## 来源：slime/utils/arguments.py L441-L451
            parser.add_argument(
                "--dynamic-sampling-filter-path",
                type=str,
                default=None,
                help=(
                    "This is the filter function for dynamic sampling. "
                    "You could use `slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std` as an example."
                ),
            )
```

**Code：**

```python
## 来源：slime/utils/arguments.py L503-L512
            parser.add_argument(
                "--buffer-filter-path",
                type=str,
                default=None,
                help=(
                    "Path to the buffer filter function. "
                    "The function should take list[list[Sample]] and return list[list[Sample]]."
                ),
            )
```

**Code：**

```python
## 来源：slime/utils/arguments.py L1415-L1435
            parser.add_argument(
                "--rollout-sample-filter-path",
                type=str,
                default=None,
                help=(
                    "Path to the rollout sample filter function. "
                    "Please directly modify the remove_sample attribute of Sample. "
                ),
            )
            parser.add_argument(
                "--rollout-all-samples-process-path",
                type=str,
                default=None,
                help=(
                    "Path to the rollout all samples process function that "
                    "can process all samples including filtered ones."
                ),
            )
```

---

## §8 Megatron 插件路径

**Code：**

```python
## 来源：slime/utils/arguments.py L1438-L1458
        def add_custom_megatron_plugins_arguments(parser):
            parser.add_argument(
                "--custom-megatron-init-path",
                type=str,
                default=None,
            )
            parser.add_argument(
                "--custom-megatron-before-log-prob-hook-path",
                type=str,
                default=None,
            )
            parser.add_argument(
                "--custom-megatron-before-train-step-hook-path",
                type=str,
                default=None,
            )
            return parser
```

---

## §9 sglang_utils：add_sglang_arguments

**Code：**

```python
## 来源：slime/backends/sglang_utils/arguments.py L35-L41
def add_sglang_arguments(parser):
    parser = add_sglang_router_arguments(parser)
    parser.set_defaults(router_balance_abs_threshold=10, router_balance_rel_threshold=1.2)
    parser.add_argument("--sglang-server-concurrency", type=int, default=512)
```

**Code：**

```python
## 来源：slime/backends/sglang_utils/arguments.py L45-L63
    skipped_args = [
        "model_path",
        "config",
        "trust_remote_code",
        "random_seed",
        "enable_memory_saver",
        "tp_size",
        "port",
        "nnodes",
        "node_rank",
        "dist_init_addr",
        "gpu_id_step",
        "base_gpu_id",
        "nccl_port",
        "skip_server_warmup",
        "enable_return_routed_experts",
    ]
```

**Comment：** Slime 自行管理 model_path/port/tp 等，故 skip 后由 rollout 启动逻辑注入。

**Code：**

```python
## 来源：slime/backends/sglang_utils/arguments.py L113-L115
    parser.add_argument = new_add_argument_wrapper
    ServerArgs.add_cli_args(parser)
    parser.add_argument = old_add_argument
```

---

## §10 sglang_config 与 PD

**Code：**

```python
## 来源：slime/backends/sglang_utils/arguments.py L117-L136
    parser.add_argument(
        "--prefill-num-servers",
        type=int,
        default=None,
        help="Number of prefill servers for disaggregation.",
    )
    parser.add_argument(
        "--sglang-config",
        type=str,
        default=None,
        help=(
            "Path to a YAML config for SGLang engine deployment. "
            "Defines server_groups with worker_type (regular/prefill/decode/placeholder), "
            "num_gpus per group, and optional per-group 'overrides' dict ..."
        ),
    )
```

**Code：**

```python
## 来源：slime/backends/sglang_utils/arguments.py L162-L173
    assert not (
        getattr(args, "sglang_config", None) is not None and getattr(args, "prefill_num_servers", None) is not None
    ), "sglang_config and prefill_num_servers are mutually exclusive. Use server_groups in the YAML config instead."
```

---

## §11 megatron_utils：HF 对齐 validate

**Code：**

```python
## 来源：slime/backends/megatron_utils/arguments.py L114-L133
    for hf_config_name, megatron_config_name, compare_fn in [
        ("hidden_size", "hidden_size", equal),
        ("num_attention_heads", "num_attention_heads", equal),
        ("num_hidden_layers", "num_layers", equal),
        ("intermediate_size", "ffn_hidden_size", equal),
        ("moe_intermediate_size", "moe_ffn_hidden_size", equal),
        ("shared_expert_intermediate_size", "moe_shared_expert_intermediate_size", equal),
        ("tie_word_embeddings", "untie_embeddings_and_output_weights", lambda x, y: not x == y),
        ("rms_norm_eps", "norm_epsilon", equal),
        ("rms_norm_eps", "layernorm_epsilon", equal),
    ]:
        if hasattr(hf_config, hf_config_name) and hasattr(args, megatron_config_name):
            if not compare_fn(getattr(hf_config, hf_config_name), getattr(args, megatron_config_name)):
                errors.append(...)
```

---

## §12 megatron 默认 RL 友好设置

**Code：**

```python
## 来源：slime/backends/megatron_utils/arguments.py L147-L168
def _set_default_megatron_args(args):
    args.use_distributed_optimizer = True
    args.bf16 = not args.fp16
    args.use_persistent_ckpt_worker = True
    args.ckpt_assume_constant_structure = True
    args.ckpt_fully_parallel_load = True
    if args.seq_length is None:
        args.seq_length = 4096
    args.max_position_embeddings = args.seq_length
    args.dist_ckpt_save_pre_mcore_014 = True
    ...
    if not args.tokenizer_model and not args.tokenizer_type:
        args.tokenizer_model = args.hf_checkpoint
        args.tokenizer_type = "HuggingFaceTokenizer"
    return args
```

---

## §13 customization.md 接口表（文档侧）

**Code：**

```python
## 来源：docs/en/get_started/customization.md L9-L30（表格摘要）
# --rollout-function-path          Override entire rollout
# --custom-generate-function-path  Override generation step only
# --custom-rm-path                 Custom reward
# --dynamic-sampling-filter-path   DAPO-style filter
# --buffer-filter-path             Buffer filter
# --rollout-sample-filter-path     Loss participation filter
# --custom-loss-function-path      Custom training loss
# --data-source-path               Prompt source
# --eval-function-path             Eval rollout
# --custom-megatron-init-path      Megatron hooks
```

---

## §14 RolloutManager 如何消费 path

**Code：**

```python
## 来源：slime/ray/rollout.py L437-L451
        data_source_cls = load_function(self.args.data_source_path)
        self.data_source = data_source_cls(args)
        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        if self.args.custom_reward_post_process_path is not None:
            self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
        if self.args.custom_convert_samples_to_train_data_path is not None:
            self.custom_convert_samples_to_train_data_func = load_function(
                self.args.custom_convert_samples_to_train_data_path
            )
```

---

## §15 plugin_contracts 测试入口

**Code：**

```python
## 来源：docs/en/get_started/customization.md L475-L481
# python -m pytest \
#   tests/plugin_contracts/test_plugin_rollout_contracts.py \
#   tests/plugin_contracts/test_plugin_generate_contracts.py \
#   tests/plugin_contracts/test_plugin_path_loading_contracts.py \
#   tests/plugin_contracts/test_plugin_runtime_hook_contracts.py
```

---

## 走读小结

| 层次 | 文件 | 读者应记住 |
|------|------|-----------|
| CLI 定义 | `arguments.py` 各 `add_*` | 默认值与签名 help |
| 解析 | `parse_args` | 三源合并 |
| SGLang | `sglang_utils/arguments.py` | 前缀透传 |
| Megatron | `megatron_utils/arguments.py` | HF 校验 + RL 默认 |
| 运行时 | `load_function` + RolloutManager | path → callable |

→ [[04-Arguments-TrainRollout-03-数据流与交互]]
