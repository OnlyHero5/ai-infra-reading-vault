---
type: batch-doc
module: 03-Arguments-Ray
batch: "03"
doc_type: walkthrough
title: "Arguments-Ray · 源码走读"
tags:
  - slime/batch/03
  - slime/module/arguments
  - slime/doc/walkthrough
updated: 2026-07-02
---

# Arguments-Ray · 源码走读

> 精读 `arguments.py` 集群段 + `parse_args` + validate 中 Ray 相关逻辑。

---

## §1 add_cluster_arguments 全文

**Explain：** 集群参数嵌在 `get_slime_extra_args_provider` 内，作为第一个 `add_*` 注册（在 custom 之后）。

**Code：**

```python
# 来源：slime/utils/arguments.py L38-L105
        def add_cluster_arguments(parser):
            parser.add_argument("--actor-num-nodes", type=int, default=1, help="Number of nodes for training actor")
            parser.add_argument(
                "--actor-num-gpus-per-node", type=int, default=8, help="Number of gpus per node for training actor"
            )

            parser.add_argument(
                "--rollout-num-gpus",
                type=int,
                default=None,
                help=(
                    "Number of GPUs for inference. Note that when using --colocate, "
                    "i.e. the training and the inference engines are on the same gpus, this param will be set as "
                    "actor_num_gpus_per_node * actor_num_nodes unless it is explicitly set. "
                    "Set it to 0 to launch routers without local SGLang engines."
                ),
            )
            parser.add_argument(
                "--rollout-num-gpus-per-engine",
                type=int,
                default=1,
                help="Number of GPUs per inference engine, just like the tp_size in sglang.",
            )
            parser.add_argument(
                "--num-gpus-per-node",
                type=int,
                default=8,
                help=(
                    "Number of gpus per node for rollout."
                    "Notice: If you are going to use less than 8 gpus per node under colocate mode, you should set this number."
                ),
            )
            parser.add_argument(
                "--colocate",
                action="store_true",
                default=False,
                help=(
                    "Whether to colocate the inference engines and the actor. "
                    "Turning this on will also set --offload to true."
                ),
            )
            parser.add_argument(
                "--offload",
                action="store_true",
                default=False,
                help=("Equivalent to --offload-train + --offload-rollout. "),
            )
            parser.add_argument(
                "--offload-train",
                action=argparse.BooleanOptionalAction,
                help=(
                    "Whether to offload the training actor to CPU during training. "
                    "This will always be true when --colocate is set."
                ),
            )
            parser.add_argument(
                "--offload-rollout",
                action=argparse.BooleanOptionalAction,
                help=(
                    "Whether to offload the rollout generator to CPU during training. "
                    "This will always be true when --colocate is set."
                ),
            )

            reset_arg(parser, "--distributed-backend", type=str, default="nccl")
            reset_arg(parser, "--distributed-timeout-minutes", type=int, default=10)

            return parser
```

**Comment：**

- `BooleanOptionalAction` 支持 `--no-offload-train` 显式关闭（colocate 下仍会被 validate 拉回 True）
- `reset_arg` 修改 Megatron 已有 distributed 默认值

---

## §2 注册顺序：cluster 最先（在 slime 段内）

**Code：**

```python
# 来源：slime/utils/arguments.py L1495-L1516
        if add_custom_arguments is not None:
            parser = add_custom_arguments(parser)

        parser = add_cluster_arguments(parser)
        parser = add_train_arguments(parser)
        parser = add_rollout_arguments(parser)
        parser = add_fault_tolerance_arguments(parser)
        parser = add_data_arguments(parser)
        parser = add_eval_arguments(parser)
        parser = add_algo_arguments(parser)
        ...
        parser = add_custom_megatron_plugins_arguments(parser)
```

**Comment：** 用户 `add_custom_arguments` 可抢先注册，但不应覆盖 cluster 字段名。

---

## §3 _pre_parse_mode：控制是否解析 SGLang

**Explain：** debug train / load debug rollout 时跳过 SGLang Phase 1。

**Code：**

```python
# 来源：slime/utils/arguments.py L1530-L1543
def _pre_parse_mode():
    temp_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    temp_parser.add_argument("--train-backend", type=str, choices=["megatron"], default="megatron")
    temp_parser.add_argument("--debug-rollout-only", action="store_true", default=False)
    temp_parser.add_argument("--debug-train-only", action="store_true", default=False)
    temp_parser.add_argument("--load-debug-rollout-data", type=str, default=None)
    temp_args, _ = temp_parser.parse_known_args()
    return temp_args
```

---

## §4 parse_args 三阶段

**Code：**

```python
# 来源：slime/utils/arguments.py L1546-L1589
def parse_args(add_custom_arguments=None):
    configure_logger()
    add_slime_arguments = get_slime_extra_args_provider(add_custom_arguments)
    pre = _pre_parse_mode()
    skip_sglang = pre.debug_train_only or pre.load_debug_rollout_data is not None

    sglang_ns = None
    if not skip_sglang:
        sglang_ns = sglang_parse_args()

    from slime.backends.megatron_utils.arguments import megatron_parse_args
    from slime.backends.megatron_utils.arguments import validate_args as megatron_validate_args

    args = megatron_parse_args(
        extra_args_provider=add_slime_arguments,
        skip_hf_validate=pre.debug_rollout_only,
    )

    for key, value in vars(pre).items():
        setattr(args, key, value)

    if sglang_ns is not None:
        for key, value in vars(sglang_ns).items():
            setattr(args, key, value)

    slime_validate_args(args)

    if pre.train_backend == "megatron" and not args.debug_rollout_only:
        megatron_validate_args(args)

    if not args.debug_train_only:
        sglang_validate_args(args)

    return args
```

**Comment：**

| Phase | 解析器 | 产出 |
|-------|--------|------|
| 0 | `_pre_parse_mode` | debug 标志 |
| 1 | `sglang_parse_args` | `sglang_*` 字段 |
| 2 | `megatron_parse_args` + slime | Megatron + cluster + … |
| 3 | validate | 副作用默认值 |

---

## §5 megatron_parse_args 与 actor 规模

**Code：**

```python
# 来源：slime/backends/megatron_utils/arguments.py L184-L199
def megatron_parse_args(extra_args_provider, skip_hf_validate=False):
    args = _megatron_parse_args(extra_args_provider=extra_args_provider, ignore_unknown_args=True)
    hf_config = None
    if args.hf_checkpoint and not skip_hf_validate:
        hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        _hf_validate_args(args, hf_config)
    if not skip_hf_validate:
        _validate_allgather_cp_supported(args, hf_config)
    args.rank = 0
    args.world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
    args = _set_default_megatron_args(args)
    return args
```

---

## §6 slime_validate_args：offload 与 critic

**Code：**

```python
# 来源：slime/utils/arguments.py L1856-L1906
    args.use_critic = args.advantage_estimator == "ppo"
    args.critic_num_gpus_per_node = args.actor_num_gpus_per_node
    args.critic_num_nodes = args.actor_num_nodes

    if args.offload:
        args.offload_train = True
        args.offload_rollout = True
    del args.offload

    if args.debug_rollout_only:
        if args.colocate and args.rollout_num_gpus is None:
            args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
        elif args.rollout_num_gpus == 0:
            args.actor_num_gpus_per_node = 0
            args.actor_num_nodes = 0
        else:
            args.actor_num_gpus_per_node = min(8, args.rollout_num_gpus)
            args.actor_num_nodes = args.rollout_num_gpus // args.actor_num_gpus_per_node
        args.colocate = False
        args.offload_train = args.offload_rollout = False
        if args.train_memory_margin_bytes > 0:
            logger.warning("Force train_memory_margin_bytes=0 since debug_rollout_only does not support it")
            args.train_memory_margin_bytes = 0

    assert not (args.debug_rollout_only and args.debug_train_only), (
        "debug_rollout_only and debug_train_only cannot be set at the same time, "
        "please set only one of them."
    )

    if args.colocate:
        if args.offload_train is None:
            args.offload_train = True
        if args.offload_rollout is None:
            args.offload_rollout = True
        if args.rollout_num_gpus is None:
            args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
        elif args.rollout_num_gpus == 0:
            logger.info("rollout_num_gpus is 0 under colocate; no local SGLang engines will be launched.")

    if args.offload_train is None:
        args.offload_train = False
    if args.offload_rollout is None:
        args.offload_rollout = False

    if args.use_critic:
        args.offload_train = True

    if args.offload_train:
        args.disable_grad_buffers_cpu_backup = True
        args.disable_param_buffers_cpu_backup = True
```

**Comment：**

- PPO critic 强制 `offload_train=True`（双模型显存）
- `debug_rollout_only` 重算 actor 节点布局，**关闭 colocate**

---

## §7 decoupled 默认 rollout_num_gpus

**Explain：** 非 debug、非 colocate 分支在 L1891 之前；若 `rollout_num_gpus is None` 在 decoupled 路径由后续逻辑设置——搜索 validate 后半段。

**Code：**

```python
# 来源：slime/utils/arguments.py L1891-L1892（colocate 块内已处理 None）
        if args.rollout_num_gpus is None:
            args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
```

**Comment：** decoupled 时常见配置：train 8 GPU + rollout 8 GPU 独立 PG；也可 rollout > actor 做更大 inference pool。

---

## §8 delta weight + colocate 互斥

**Code：**

```python
# 来源：slime/utils/arguments.py L1986-L1997
    if args.update_weight_mode == "delta":
        if args.update_weight_transport != "disk":
            raise ValueError(
                "--update-weight-mode=delta requires --update-weight-transport=disk, "
                f"got {args.update_weight_transport!r}."
            )
        if args.colocate:
            raise ValueError(
                "--update-weight-mode=delta is not supported with --colocate. Colocate transfers "
                "weights via CUDA IPC (only a handle crosses processes), so the delta bookkeeping "
                "(snapshot + diff + encode) is pure overhead."
            )
```

---

## §9 external engine 与 rollout_num_gpus

**Code：**

```python
# 来源：slime/utils/arguments.py L1851-L1854
    args.rollout_external = args.rollout_external_engine_addrs is not None

    if args.rollout_external and not args.debug_train_only:
        apply_external_engine_info_to_args(args, logger=logger)
```

**Comment：** external 模式可配合本地 0 GPU（仅 router），见 [[16-External-Engines-00-MOC]]。

---

## §10 reset_arg 工具

**Code：**

```python
# 来源：slime/utils/arguments.py L19-L32
def reset_arg(parser, name, **kwargs):
    for action in parser._actions:
        if name in action.option_strings:
            if "default" in kwargs:
                action.default = kwargs["default"]
            break
    else:
        parser.add_argument(name, **kwargs)
```

**Comment：** Megatron 已注册的 `--distributed-backend` 等用此改默认而不重复 add。

---

## §11 sglang validate 与 rollout GPU 拓扑

**Code：**

```python
# 来源：slime/backends/sglang_utils/arguments.py L141-L154
def validate_args(args):
    args.sglang_dp_size = args.sglang_data_parallel_size
    args.sglang_pp_size = args.sglang_pipeline_parallel_size
    args.sglang_ep_size = args.sglang_expert_parallel_size
    if args.sglang_pp_size > 1:
        assert args.rollout_num_gpus_per_engine % args.sglang_pp_size == 0, (
            f"rollout_num_gpus_per_engine ({args.rollout_num_gpus_per_engine}) must be divisible by "
            f"sglang_pipeline_parallel_size ({args.sglang_pp_size})"
        )
        args.sglang_tp_size = args.rollout_num_gpus_per_engine // args.sglang_pp_size
    else:
        args.sglang_tp_size = args.rollout_num_gpus_per_engine
```

**Comment：** `rollout_num_gpus_per_engine` 来自 cluster 段，连接 Ray GPU 与 SGLang TP。

---

## 走读小结

| 阶段 | 关键函数 | Ray 相关产出 |
|------|---------|-------------|
| 定义 | `add_cluster_arguments` | CLI 字段 |
| 解析 | `parse_args` | 合并 namespace |
| 校验 | `slime_validate_args` | colocate/offload 默认值 |
| 下游 | `create_placement_groups` | PG bundle（批次 06） |

→ [[03-Arguments-Ray-03-数据流与交互]]
