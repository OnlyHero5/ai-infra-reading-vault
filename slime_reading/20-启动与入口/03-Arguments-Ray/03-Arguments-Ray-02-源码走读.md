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
updated: 2026-07-05
---

# Arguments-Ray · 源码走读

> 走读主线：Slime 的 Ray 资源参数并不是一个独立 parser，而是被注入 Megatron parser，再与 SGLang parser 的 namespace 合并。`slime_validate_args` 随后根据 debug/external/colocate/offload 等模式改写 actor 与 rollout GPU 拓扑。理解这一页的关键是区分“CLI 字段定义”“多 parser 合并”和“validate 副作用默认值”三层。

---

## 1. 参数注册：把 Ray 拓扑挂到 Megatron parser

### 1.1 reset_arg 修改已有 Megatron 参数默认值

问题与约束：
- Slime 复用 Megatron parser；部分参数如 distributed backend 已经由 Megatron 注册，Slime 不能重复 `add_argument`，但需要覆盖默认值。

设计选择：
- `reset_arg` 遍历 parser actions，如果已有同名 option 且传入 default，就改 action.default；如果不存在才新增参数。

Explain：
这个工具被 cluster arguments 用来把 `--distributed-backend` 默认设为 `nccl`，把 `--distributed-timeout-minutes` 默认设为 10。

来源：slime/utils/arguments.py L19-L32

Code：

```python
def reset_arg(parser, name, **kwargs):
    for action in parser._actions:
        if name in action.option_strings:
            if "default" in kwargs:
                action.default = kwargs["default"]
            break
    else:
        parser.add_argument(name, **kwargs)
```

代码逻辑：
- 扫描 parser 中所有 action。
- 找到匹配 option string 后只更新 default。
- 没找到时按普通 `add_argument` 新增。

为什么这样写：
- 复用 Megatron parser 时不能重复注册同名参数。
- Slime 需要在不 fork Megatron 参数定义的前提下改默认值。

不变量与失败模式：
- 只会修改 default，不会修改 type、choices 或 help 等其他属性。
- 如果同一个 name 对应多个 action，只处理第一个命中的 action。

Comment：
`reset_arg` 是 Slime 和 Megatron 参数系统之间的兼容层。

### 1.2 add_cluster_arguments 定义 actor/rollout GPU 与 offload 参数

问题与约束：
- Ray placement group 需要 actor 节点数、每节点 GPU 数、rollout GPU 池、每个 SGLang engine 的 GPU 数，以及 colocate/offload 关系。

设计选择：
- cluster 段定义 `actor_num_nodes`、`actor_num_gpus_per_node`、`rollout_num_gpus`、`rollout_num_gpus_per_engine`、`num_gpus_per_node`、`colocate`、`offload`、`offload_train`、`offload_rollout`；offload train/rollout 用 `BooleanOptionalAction` 支持显式 true/false。

Explain：
`rollout_num_gpus=None` 表示后续 validate 逻辑按模式推导；`rollout_num_gpus=0` 被文档化为只启动 router、不启动本地 SGLang engines。

来源：slime/utils/arguments.py L38-L105

Code：

```python
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
    )
    parser.add_argument(
        "--colocate",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--offload-train",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--offload-rollout",
        action=argparse.BooleanOptionalAction,
    )

    reset_arg(parser, "--distributed-backend", type=str, default="nccl")
    reset_arg(parser, "--distributed-timeout-minutes", type=int, default=10)

    return parser
```

代码逻辑：
- actor 资源默认 1 节点、每节点 8 GPU。
- rollout 总 GPU 默认 None，等待 validate 推导或用户显式设置。
- 每个 rollout engine 默认 1 GPU。
- colocate 是布尔开关。
- offload 总开关后续展开成 train/rollout 两个开关。
- 修改 Megatron distributed 默认值。

为什么这样写：
- Ray placement group 和 SGLang engine 拓扑需要统一来自 CLI。
- offload 的总开关便于常用配置，细分开关便于高级场景覆盖。

不变量与失败模式：
- `rollout_num_gpus_per_engine` 后续会被 SGLang PP 校验整除。
- `rollout_num_gpus=0` 只在部分模式下有明确语义。
- colocate 下 offload 默认会被 validate 拉成 true。

Comment：
cluster 段是 Ray 资源拓扑的 CLI 入口，后续 placement group 都依赖这些字段。

### 1.3 Slime 参数 provider 先接收 custom，再注册 cluster

问题与约束：
- 用户可能传入自定义参数 provider；Slime 自己也要按固定顺序注册 cluster、train、rollout、data、eval 等参数段。

设计选择：
- `get_slime_extra_args_provider` 返回 `add_slime_arguments`，先执行 `add_custom_arguments`，再注册 cluster 和其他 Slime 参数段，最后 reset 一些 Megatron 参数。

Explain：
源码注释说明 custom arguments 放在前面是为了防止覆盖部分 Slime arguments；这意味着自定义 provider 不应重复定义 cluster 字段名。

来源：slime/utils/arguments.py L1495-L1525

Code：

```python
if add_custom_arguments is not None:
    parser = add_custom_arguments(parser)

parser = add_cluster_arguments(parser)
parser = add_train_arguments(parser)
parser = add_rollout_arguments(parser)
parser = add_fault_tolerance_arguments(parser)
parser = add_data_arguments(parser)
parser = add_eval_arguments(parser)
parser = add_algo_arguments(parser)
parser = add_on_policy_distillation_arguments(parser)
parser = add_wandb_arguments(parser)
parser = add_tensorboard_arguments(parser)
parser = add_debug_arguments(parser)
parser = add_network_arguments(parser)
parser = add_reward_model_arguments(parser)
parser = add_rollout_buffer_arguments(parser)
parser = add_mtp_training_arguments(parser)
parser = add_ci_arguments(parser)
parser = add_custom_megatron_plugins_arguments(parser)
reset_arg(
    parser,
    "--custom-config-path",
    type=str,
    default=None,
    help="Path to the YAML config for custom function arguments.",
)
reset_arg(parser, "--padded-vocab-size", type=int, default=None)

return parser
```

代码逻辑：
- custom provider 最先运行。
- cluster 段是 Slime 自有参数段的第一个。
- 训练、rollout、容错、数据、评估等段依次注册。
- 最后通过 reset_arg 处理与 Megatron 可能重名的参数。

为什么这样写：
- Slime 参数需要被 Megatron parser 一次性解析。
- custom provider 先运行可以让用户扩展 parser，但也要求避免与 Slime 核心字段冲突。

不变量与失败模式：
- 重复注册同名参数会由 argparse 报错。
- reset_arg 只能改 default，不能重写完整 action。

Comment：
这段决定了 Slime 参数和用户参数的覆盖边界。

---

## 2. parse_args 三阶段合并 namespace

### 2.1 _pre_parse_mode 先抽取控制解析流程的 debug 标志

问题与约束：
- 是否解析 SGLang 参数、是否跳过 HF validate，取决于少数 debug 参数；但完整 parser 还没构建。

设计选择：
- `_pre_parse_mode` 用一个轻量 parser 只解析 `train_backend`、`debug_rollout_only`、`debug_train_only` 和 `load_debug_rollout_data`，并用 `parse_known_args` 忽略其他 CLI。

Explain：
返回的 namespace 之后会被合并回最终 args，避免这些控制字段丢失。

来源：slime/utils/arguments.py L1530-L1543

Code：

```python
def _pre_parse_mode():
    temp_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    temp_parser.add_argument("--train-backend", type=str, choices=["megatron"], default="megatron")
    temp_parser.add_argument("--debug-rollout-only", action="store_true", default=False)
    temp_parser.add_argument("--debug-train-only", action="store_true", default=False)
    temp_parser.add_argument("--load-debug-rollout-data", type=str, default=None)
    temp_args, _ = temp_parser.parse_known_args()
    return temp_args
```

代码逻辑：
- 创建无 help 的临时 parser。
- 只注册解析流程需要的几个参数。
- 使用 `parse_known_args` 接受完整 CLI。
- 返回临时 namespace。

为什么这样写：
- 完整 Megatron/SGLang parser 初始化成本和副作用更大。
- debug train-only 或加载 debug rollout 数据时不需要 SGLang parser。

不变量与失败模式：
- 这里注册的参数不能在后续 provider 中重复注册。
- `train_backend` 当前只允许 `megatron`。

Comment：
这是 Slime 参数解析的 Phase 0。

### 2.2 parse_args 合并 SGLang、Megatron 和 Slime 参数

问题与约束：
- Slime 同时依赖 SGLang parser 和 Megatron parser；二者 namespace 要合并，validate 顺序也要按 debug 模式裁剪。

设计选择：
- `parse_args` 先配置 logger 和 Slime extra args provider；根据 pre-parse 决定是否跑 `sglang_parse_args`；再调用 `megatron_parse_args(extra_args_provider=add_slime_arguments)`；随后把 pre 和 sglang namespace 写入 Megatron args，依次执行 Slime、Megatron、SGLang validate。

Explain：
`skip_sglang = debug_train_only or load_debug_rollout_data is not None`；`debug_rollout_only` 会传给 Megatron parse 用于跳过 HF validate。

来源：slime/utils/arguments.py L1546-L1589

Code：

```python
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

代码逻辑：
- Slime extra args provider 注入 Megatron parser。
- Phase 0 解析 debug 控制字段。
- 条件解析 SGLang namespace。
- Megatron parser 解析 Megatron + Slime 字段。
- pre namespace 和 SGLang namespace 写回 args。
- Slime validate 总是先执行。
- debug_rollout_only 跳过 Megatron validate。
- debug_train_only 跳过 SGLang validate。

为什么这样写：
- Megatron parser 是最终 args 容器，Slime/Ray 参数自然落在同一个 namespace。
- SGLang 和 Megatron validate 都可能依赖 Slime validate 改写后的字段。

不变量与失败模式：
- 同名字段后写入会覆盖前值，SGLang namespace 合并在 pre 后。
- `load_debug_rollout_data` 会跳过 SGLang parser，但后续代码不能假设所有 `sglang_*` 字段都存在。
- debug_rollout_only 只跳过 HF validate 和 Megatron validate，不跳过 SGLang validate。

Comment：
理解 Arguments-Ray 的关键是：最终 args 是 Megatron parser 的 namespace，被 Slime 和 SGLang 依次填充。

### 2.3 megatron_parse_args 设置 actor world_size

问题与约束：
- Megatron 默认 parser 不知道 Slime Ray actor 的节点和 GPU 数；训练 actor 的 world size 需要从 cluster 参数推导。

设计选择：
- `megatron_parse_args` 调 Megatron 原始 parser，使用 `ignore_unknown_args=True` 接收 SGLang/Slime 外围参数；可选做 HF config validate；最后设置 `rank=0` 和 `world_size=actor_num_nodes * actor_num_gpus_per_node`，再补默认 Megatron 参数。

Explain：
`skip_hf_validate` 由 debug_rollout_only 控制，避免只跑 rollout 时还强行验证训练 HF 配置。

来源：slime/backends/megatron_utils/arguments.py L184-L199

Code：

```python
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

代码逻辑：
- 调 Megatron parser 并忽略未知参数。
- 可选加载 HF config 并校验。
- 可选校验 all-gather CP 支持。
- actor rank 初始化为 0。
- world size 来自 actor 节点数乘每节点 GPU 数。
- 补 Megatron 默认值。

为什么这样写：
- Ray actor 启动前先得到逻辑训练 world size。
- Slime 的 cluster 参数需要注入 Megatron namespace 后才能计算 world size。

不变量与失败模式：
- `actor_num_nodes` 和 `actor_num_gpus_per_node` 必须是有效整数。
- HF checkpoint 不可访问时，非 debug_rollout_only 会在 parse 阶段失败。

Comment：
这段把 CLI 里的 Ray actor 规模转成 Megatron 的 world size。

---

## 3. Slime validate 改写 Ray 拓扑

### 3.1 load-debug 和 external engine 会改写训练/rollout模式

问题与约束：
- 从调试 rollout 数据训练时不应该启动 SGLang；外部 engine 模式下 rollout GPU 数应来自远端 engine 发现结果，而不是本地 CLI 估计。

设计选择：
- `load_debug_rollout_data` 存在时把 `debug_train_only=True`；`rollout_external` 由 `rollout_external_engine_addrs is not None` 推导，且非 debug train-only 时调用 `apply_external_engine_info_to_args`。

Explain：
这段在 offload/colocate 逻辑之前执行，因此 external 信息会先写入 args。

来源：slime/utils/arguments.py L1845-L1854

Code：

```python
if args.load_debug_rollout_data is not None:
    logger.info(
        f"load_debug_rollout_data {args.load_debug_rollout_data} is set, "
        "will not instantiate sglang servers and will only run the training process."
    )
    args.debug_train_only = True

args.rollout_external = args.rollout_external_engine_addrs is not None

if args.rollout_external and not args.debug_train_only:
    apply_external_engine_info_to_args(args, logger=logger)
```

代码逻辑：
- 加载 debug rollout 数据时强制 train-only。
- external engine 地址非空时标记 external。
- external 且不是 train-only 时发现远端 engine 信息。

为什么这样写：
- debug train-only 不需要本地或外部 SGLang。
- external 模式下实际 engine 数和 GPU 数以远端 server info 为准。

不变量与失败模式：
- `rollout_external_engine_addrs` 非空即进入 external 模式。
- external discovery 失败会阻断参数 validate。

Comment：
这一段决定 Slime 是否会实例化 rollout engines。

### 3.2 external engine discovery 写回 rollout_num_engines 和 rollout_num_gpus

问题与约束：
- 外部 engine 不是本地 Ray placement group 创建的，Slime 需要从地址列表探测实际 topology。

设计选择：
- `apply_external_engine_info_to_args` 要求 `rollout_external_engine_addrs` 非空；调用 `discover_external_engines` 后，把 engine info dict 列表、engine 数和总 GPU 数写回 args。

Explain：
`rollout_num_gpus` 在 external 模式下是所有外部 engine 的 `num_gpus` 求和。

来源：slime/backends/sglang_utils/external.py L107-L120

Code：

```python
def apply_external_engine_info_to_args(args, logger=None) -> None:
    addrs = args.rollout_external_engine_addrs
    if not addrs:
        raise ValueError("apply_external_engine_info_to_args requires --rollout-external-engine-addrs.")

    infos = discover_external_engines(addrs)
    if not infos:
        raise ValueError("--rollout-external-engine-addrs did not contain any engines.")

    args.rollout_external_engine_infos = [info.to_dict() for info in infos]
    args.rollout_num_engines = len(infos)
    args.rollout_num_gpus = sum(info.num_gpus for info in infos)
```

代码逻辑：
- 校验外部地址存在。
- 探测每个外部 engine。
- 空结果直接失败。
- 保存序列化后的 engine infos。
- 写入 engine 数和 rollout 总 GPU 数。

为什么这样写：
- 外部 engine 拓扑不能由本地 Ray 资源参数推断。
- 后续 ServerGroup 和 router 逻辑需要知道 engine 数和总 GPU 数。

不变量与失败模式：
- 每个外部地址必须能返回可解析的 server info。
- 外部 engine info 的 `num_gpus` 是 rollout GPU 总数的唯一来源。

Comment：
这也是普通非 external 路径不要随意套用 external 默认值的原因。

### 3.3 offload、debug_rollout_only、colocate 共同决定 actor 与 rollout GPU

问题与约束：
- 同一套 CLI 支持训练+rollout、只跑 rollout、只跑训练、colocate 和 critic 等模式；这些模式会改变 actor GPU、rollout GPU 和 offload 默认值。

设计选择：
- `offload` 总开关展开为 `offload_train=True` 和 `offload_rollout=True` 后删除；debug_rollout_only 会重算 actor 节点布局并关闭 colocate/offload；colocate 下 offload_train/offload_rollout 默认为 True，rollout_num_gpus 缺省时设为 actor GPU 总数；use_critic 强制 offload_train。

Explain：
`rollout_num_gpus=0` 在 colocate 下只打日志，表示不启动本地 SGLang engines；debug_rollout_only 下如果 rollout_num_gpus 为 0，则 actor 节点/GPU 都置 0。

来源：slime/utils/arguments.py L1856-L1906

Code：

```python
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
    "debug_rollout_only and debug_train_only cannot be set at the same time, " "please set only one of them."
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

代码逻辑：
- PPO 模式启用 critic，并让 critic GPU 规模跟 actor 一致。
- `--offload` 展开为两个细分开关。
- debug rollout-only 重新推导 actor 规模，并关闭 colocate/offload。
- debug rollout-only 与 debug train-only 互斥。
- colocate 下默认开启 train/rollout offload。
- colocate 且 rollout_num_gpus 未指定时等于 actor 总 GPU。
- None offload 最终落为 False。
- critic 模式强制 offload_train。
- offload_train 时禁用 grad/param buffer CPU backup。

为什么这样写：
- colocate 下训练和推理共用 GPU，需要 offload 降低显存冲突。
- debug_rollout_only 不需要训练 actor，因此 actor 规模要由 rollout 需求重算或清零。
- critic 增加模型副本，训练侧显存压力更大。

不变量与失败模式：
- debug_rollout_only 和 debug_train_only 不能同时开启。
- debug_rollout_only 下 rollout_num_gpus 不是 0 时，actor_num_nodes 使用整除结果，配置不整除会产生非预期规模。
- 普通非 colocate 路径没有在这一段把 rollout_num_gpus=None 自动设成 actor 总 GPU。

Comment：
Ray GPU 拓扑的多数副作用默认值都集中在这一段。

### 3.4 batch、epoch、routing replay 与上下文长度在同一 validate 中收口

问题与约束：
- Ray 资源参数之外，rollout batch 与训练 batch、采样数、epoch/rollout 终止条件、routing replay 和上下文长度也会影响后续数据流。

设计选择：
- 如果给了 `num_steps_per_rollout`，由 rollout_batch_size 和 n_samples_per_prompt 推导 global_batch_size；n_samples_per_prompt 为 1 时关闭 grpo_std_normalization；over_sampling_batch_size 默认为 rollout_batch_size 且必须不小于它；num_epoch 和 num_rollout 至少给出一种；use_rollout_routing_replay 会打开 use_routing_replay；eval/rollout context len 也在这里补默认和校验。

Explain：
这些不是 Ray placement group 直接参数，但它们和 rollout 规模共同决定一次 rollout 产生多少训练样本。

来源：slime/utils/arguments.py L1908-L1975

Code：

```python
if args.eval_function_path is None:
    args.eval_function_path = args.rollout_function_path

if args.num_steps_per_rollout is not None:
    global_batch_size = args.rollout_batch_size * args.n_samples_per_prompt // args.num_steps_per_rollout
    if args.global_batch_size is not None:
        assert args.global_batch_size == global_batch_size, (
            f"global_batch_size {args.global_batch_size} is not equal to "
            f"rollout_batch_size {args.rollout_batch_size} * n_samples_per_prompt {args.n_samples_per_prompt} "
            f"// num_steps_per_rollout {args.num_steps_per_rollout}"
        )
    args.global_batch_size = global_batch_size

if args.n_samples_per_prompt == 1:
    args.grpo_std_normalization = False
    logger.info("n_samples_per_prompt is set to 1, grpo_std_normalization will be set to False.")

if args.over_sampling_batch_size is None:
    args.over_sampling_batch_size = args.rollout_batch_size

assert args.over_sampling_batch_size >= args.rollout_batch_size, (
    f"over_sampling_batch_size {args.over_sampling_batch_size} should be greater than or equal to "
    f"rollout_batch_size {args.rollout_batch_size}"
)

if args.num_epoch is not None:
    if args.num_rollout is not None:
        logger.info("Both num_epoch and num_rollout are set, num_epoch will be ignored.")
    else:
        assert args.rollout_global_dataset
else:
    assert args.num_rollout is not None, (
        "num_epoch is not set, but num_rollout is not set, " "please set --num-rollout or --num-epoch"
    )

if args.use_rollout_routing_replay:
    args.use_routing_replay = True

if args.eval_max_context_len is None:
    args.eval_max_context_len = args.rollout_max_context_len

if args.rollout_max_context_len is not None:
    if args.rollout_max_prompt_len is None:
        args.rollout_max_prompt_len = args.rollout_max_context_len - 1
    assert (
        args.rollout_max_prompt_len <= args.rollout_max_context_len - 1
    ), f"args.rollout_max_prompt_len ({args.rollout_max_prompt_len}) must be smaller than args.rollout_max_context_len ({args.rollout_max_context_len}) so that there is at least one generated token to compute loss."
```

代码逻辑：
- eval function 缺省继承 rollout function。
- 按 rollout steps 推导 global batch。
- 单样本采样关闭 GRPO 标准化。
- oversampling batch 缺省等于 rollout batch。
- epoch/rollout 终止条件做互斥和必填校验。
- rollout routing replay 打开底层 routing replay。
- eval context len 缺省等于 rollout context len。
- prompt len 缺省为 context len - 1 并校验至少留一个生成 token。

为什么这样写：
- 参数 validate 阶段统一收口派生字段，后续模块可以少处理 None。
- batch 和上下文长度错误越早暴露越好。

不变量与失败模式：
- `global_batch_size` 若显式设置，必须等于推导值。
- `over_sampling_batch_size >= rollout_batch_size`。
- `num_epoch` 和 `num_rollout` 至少要有一个有效终止条件。
- prompt len 必须小于 context len。

Comment：
这些校验解释了为什么 arguments 模块不是单纯 parser 文件。

### 3.5 disk/delta 权重更新约束在参数阶段提前失败

问题与约束：
- disk 权重同步需要共享目录；delta 模式只支持 disk transport，且 colocate 下 CUDA IPC 已经足够，delta bookkeeping 反而是额外成本。

设计选择：
- 如果 `update_weight_transport == "disk"` 但没有 `update_weight_disk_dir`，直接 ValueError；如果 `update_weight_mode == "delta"`，要求 transport 为 disk、禁止 colocate，并要求 `update_weight_local_checkpoint_dir`。

Explain：
这些错误都在参数 validate 阶段抛出，避免训练开始后才在权重同步路径失败。

来源：slime/utils/arguments.py L1980-L2002

Code：

```python
if args.update_weight_transport == "disk" and not args.update_weight_disk_dir:
    raise ValueError(
        "--update-weight-transport=disk requires --update-weight-disk-dir to point at "
        "a filesystem shared between the trainer and the rollout engines."
    )
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
    if not args.update_weight_local_checkpoint_dir:
        raise ValueError(
            "--update-weight-mode=delta requires --update-weight-local-checkpoint-dir "
            "(a rollout-host-local NVMe directory)."
        )
```

代码逻辑：
- disk transport 必须有共享 disk dir。
- delta 模式必须使用 disk transport。
- delta 模式禁止 colocate。
- delta 模式必须提供本地 checkpoint dir。

为什么这样写：
- 权重同步是跨进程/跨节点路径，缺目录会导致更晚、更难定位的失败。
- colocate 场景已经通过 CUDA IPC 传权重，delta 不划算。

不变量与失败模式：
- disk dir 必须被 trainer 和 rollout engines 同时可见。
- local checkpoint dir 应是 rollout host 本地可写目录。

Comment：
权重同步模式的合法性也属于 Ray/rollout 拓扑的一部分。

---

## 4. SGLang validate 把 rollout GPU 映射到 TP/PP/DP/EP

### 4.1 sglang_validate_args 计算 effective SGLang 并行尺寸

问题与约束：
- Slime CLI 使用 `sglang_*` 字段承接 SGLang parser，但下游 SGLangEngine 需要明确的 dp/pp/ep/tp size。

设计选择：
- 将 `sglang_data_parallel_size`、`sglang_pipeline_parallel_size`、`sglang_expert_parallel_size` 分别写成 `sglang_dp_size`、`sglang_pp_size`、`sglang_ep_size`；如果 PP 大于 1，要求 `rollout_num_gpus_per_engine` 能被 PP 整除，并把 TP 设为商，否则 TP 等于每 engine GPU 数。

Explain：
DP 大于 1 时要求开启 DP attention；router IP 如果存在会经过 IPv6 包裹处理；后面还有 PD disaggregation 与 sglang-config 互斥检查。

来源：slime/backends/sglang_utils/arguments.py L141-L165

Code：

```python
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

    if args.sglang_dp_size > 1:
        assert args.sglang_enable_dp_attention

    if getattr(args, "sglang_router_ip", None):
        args.sglang_router_ip = _wrap_ipv6(args.sglang_router_ip)

    assert not (
        getattr(args, "prefill_num_servers", None) is not None and getattr(args, "rollout_external", False)
    ), "prefill_num_servers cannot be set with --rollout-external-engine-addrs."
```

代码逻辑：
- 重命名 SGLang DP/PP/EP effective 字段。
- PP 大于 1 时按每 engine GPU 数除以 PP 得到 TP。
- PP 等于 1 时 TP 就是每 engine GPU 数。
- DP 大于 1 要求 DP attention。
- router IP 做 IPv6 规范化。
- prefill server 数和 external engine 地址互斥。

为什么这样写：
- `rollout_num_gpus_per_engine` 是 Ray 资源维度，SGLang 需要从中派生 TP。
- PP 切分后每个 pipeline stage 内的 GPU 才是 TP 规模。
- DP attention 是 SGLang 多 DP 的必要条件。

不变量与失败模式：
- `rollout_num_gpus_per_engine % sglang_pp_size == 0`。
- `sglang_dp_size > 1` 时必须开启 `sglang_enable_dp_attention`。
- external engine 模式不能同时使用 prefill_num_servers。

Comment：
这一步把 Ray 的 rollout engine GPU 数转成 SGLang 内部并行拓扑。

---

## 5. 走读小结

```text
CLI
  -> _pre_parse_mode
  -> optional sglang_parse_args
  -> megatron_parse_args(extra_args_provider=Slime)
  -> merge namespaces
  -> slime_validate_args
  -> megatron_validate_args / sglang_validate_args

Ray topology fields
  -> actor_num_nodes / actor_num_gpus_per_node
  -> rollout_num_gpus / rollout_num_gpus_per_engine
  -> colocate / offload_train / offload_rollout
  -> sglang_tp_size / dp_size / pp_size / ep_size
```

→ [[03-Arguments-Ray-03-数据流与交互]]
