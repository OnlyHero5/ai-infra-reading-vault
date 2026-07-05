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
updated: 2026-07-05
---

# Arguments-TrainRollout · 源码走读

> 走读顺序：`slime/utils/arguments.py` → SGLang/Megatron 参数适配 → RolloutManager 消费 `*-path` → customization 文档契约。

## 源码阅读依据

| 上游文件 | 本文关注点 |
|----------|------------|
| `slime/slime/utils/arguments.py` | Slime 统一 CLI provider、两阶段 parse、validate 时补默认与互斥约束 |
| `slime/slime/backends/sglang_utils/arguments.py` | SGLang 参数前缀化、skip list、PD/sglang_config 互斥 |
| `slime/slime/backends/megatron_utils/arguments.py` | Megatron 参数解析、HF config 对齐、RL 友好默认值 |
| `slime/slime/ray/rollout.py` | `*-path` 参数如何在 RolloutManager 中被 `load_function` 消费 |
| `slime/docs/en/get_started/customization.md` | 插件路径的用户侧契约与 contract tests |

## 设计主线：Arguments 为什么是运行时协议，不只是 CLI 表

Slime 的参数层同时服务三类系统：Megatron 训练进程、SGLang rollout 服务、Ray orchestration。它的设计哲学不是“把所有 flag 放进一个 argparse”，而是：

1. **按系统边界定义参数。** 训练、rollout、data、eval、algo、reward、debug、plugin 分组，读者能从分组看出参数影响哪个子系统。
2. **分阶段解析，最后合并。** SGLang 参数先用独立 parser 前缀化解析，Megatron parser 再吃 Slime extra args；debug 模式会影响是否解析 SGLang。
3. **validate 阶段把声明变成契约。** 很多默认值不是 `add_argument(default=...)` 完成的，而是在 `slime_validate_args` 中根据组合关系补齐、改写或拒绝。
4. **函数路径是扩展边界。** CLI 中的 `*-path` 最终被 RolloutManager 用 `load_function` 变成 callable；文档和 contract tests 则约束这些 callable 的签名。

读这一篇时，重点不是背参数名，而是追踪一个参数从 **定义 → parse/merge → validate → runtime consumption** 的路径。

---

## 1. add_train_arguments — 权重同步参数

**Explain：** 训练参数段集中定义 Megatron→HF、权重同步模式、磁盘 delta 传输以及训练显存相关开关。

**问题与约束：** 权重同步有 full/delta、nccl/disk、共享目录、本地 checkpoint、checksum、非 POSIX 文件系统 hook 等组合；这些必须在 CLI 层表达清楚。

**设计选择：** `--update-weight-mode` 和 `--update-weight-transport` 分开建模，delta 相关的 encoding/checksum/hook/local dir 单独作为参数，具体合法性留给 validate。

**代码逻辑：**

```python
## 来源：slime/utils/arguments.py L107-L155
parser.add_argument("--megatron-to-hf-mode", choices=["raw", "bridge"], default="raw")
parser.add_argument("--update-weight-mode", choices=["full", "delta"], default="full")
parser.add_argument(
    "--update-weight-transport",
    choices=["nccl", "disk"],
    default="nccl",
    help=(
        "Carrier for weight sync. In full mode, 'nccl' broadcasts chunks and "
        "'disk' writes a complete HF checkpoint under --update-weight-disk-dir "
        "before engines reload it. Delta mode is 'disk' only..."
    ),
)
```

**为什么这样写：** Slime 把策略和传输介质拆开，方便 full+nccl、full+disk、delta+disk 共享一套入口，同时让非法组合在 validate 阶段 fail loud。

**不变量与失败模式：** `update_weight_transport=disk` 必须有共享目录；`delta` 只能配 disk，还要求 rollout-host-local checkpoint dir。CLI help 说明语义，真正 enforcement 在 `slime_validate_args`。

**Comment：** 参数层已经埋下了 WeightSync-Disk 的设计边界：full 是完整 checkpoint，delta 是版本流。

---

## 2. add_rollout_arguments — rollout 函数入口

**Explain：** rollout 参数段定义 HF checkpoint、模型名、默认 rollout function path 和采样参数。

**问题与约束：** SGLang 初始化需要 HF checkpoint/tokenizer，但训练权重会在训练前同步到 rollout engine；因此 HF checkpoint 只要求架构对齐，不一定是最新权重。

**设计选择：** `--hf-checkpoint` 是 rollout 初始化和 tokenizer 来源；`--rollout-function-path` 默认指向内置 SGLang rollout，但允许整体替换。

**代码逻辑：**

```python
## 来源：slime/utils/arguments.py L304-L340
parser.add_argument("--hf-checkpoint", type=str, default=None, help=(...))
parser.add_argument(
    "--rollout-function-path",
    type=str,
    default="slime.rollout.sglang_rollout.generate_rollout",
    help=(
        "Path to the rollout generation function."
        "The signature of the function should be "
        "`def generate_rollout(args, rollout_id, data_source, evaluation=False) -> ...`"
    ),
)
```

**为什么这样写：** rollout orchestration 是可替换的最大边界；默认路径服务常规 SGLang 生成，复杂 agent/multi-turn 场景可以接管整个 rollout。

**不变量与失败模式：** 自定义 rollout function 必须返回符合 `RolloutFnTrainOutput` / `RolloutFnEvalOutput` 的数据；HF checkpoint 架构必须能和 Megatron 参数通过后续校验。

**Comment：** `hf_checkpoint` 的语义不是“训练初始权重一定来自这里”，而是“rollout/tokenizer/HF 对齐基准”。

---

## 3. custom_generate_function_path — 只替换生成步骤

**Explain：** `--custom-generate-function-path` 只替换默认 rollout 内部的 per-sample generate 逻辑，不替换整个 rollout 循环。

**问题与约束：** 多数用户只需要改 tool use、multi-turn、function calling 等生成细节；重写整个 rollout 会重复处理 buffer、reward、logging、train data 转换。

**设计选择：** 把整体 rollout 和内部 generate 分成两个扩展层级：`rollout-function-path` 管大循环，`custom-generate-function-path` 管单样本生成。

**代码逻辑：**

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

**为什么这样写：** 它提供窄扩展点，降低用户自定义时破坏默认 rollout 管线的概率。

**不变量与失败模式：** 该参数只有默认 rollout 实现主动读取才生效；如果用户替换整个 rollout function，是否支持这个子扩展点取决于自定义实现。

**Comment：** 这是 Slime 插件设计的基本风格：优先给小钩子，必要时再替换大流程。

---

## 4. update_weights_interval — rollout 与训练的同步节奏

**Explain：** `--update-weights-interval` 控制权重同步频率，默认每个 rollout step 更新一次。

**问题与约束：** 更新太频繁会增加同步成本，更新太慢会加大 rollout/train policy mismatch。

**设计选择：** 参数层只定义 interval；实际调度由训练主循环/rollout manager 侧使用。

**代码逻辑：**

```python
## 来源：slime/utils/arguments.py L523-L528
parser.add_argument(
    "--update-weights-interval",
    type=int,
    default=1,
    help="Interval for updating the weights",
)
```

**为什么这样写：** 参数是系统级节奏控制，不应绑定某一种传输实现；full、delta、tensor 更新都可以共享这个 interval。

**不变量与失败模式：** interval 应为正整数；若训练代码没有在间隔点触发 update，rollout 权重版本会滞后。

**Comment：** 这个 flag 是连接 Arguments 与 WeightSync 专题的最短路径。

---

## 5. add_data_arguments — data source 与 batch 语义

**Explain：** data 参数段定义 rollout 数据来源、prompt 格式、`rollout_batch_size`、`n_samples_per_prompt` 和训练 batch 的关系。

**问题与约束：** RL rollout 的 batch 单位是 prompt，但训练样本单位是 response；一个 prompt 可生成多个 sample。

**设计选择：** `rollout_batch_size` 设为 required，`n_samples_per_prompt` 默认 1，并在 help 中明确总返回量是二者乘积。

**代码逻辑：**

```python
## 来源：slime/utils/arguments.py L625-L688
parser.add_argument("--data-source-path", default="slime.rollout.data_source.RolloutDataSourceWithBuffer")
parser.add_argument("--prompt-data", type=str, default=None, help=(...))
parser.add_argument(
    "--rollout-batch-size",
    type=int,
    required=True,
    help=(
        "The number of prompts in each rollout step. "
        "The total data returned should be rollout_batch_size * n_samples_per_prompt. "
    ),
)
parser.add_argument("--n-samples-per-prompt", type=int, default=1)
```

**为什么这样写：** 参数层直接暴露 prompt/sample 两种粒度，后续 reward normalization、GRPO group、global batch 推导都依赖这个区分。

**不变量与失败模式：** `rollout_batch_size` 必须提供；`global_batch_size` 若按 sample 计，就不能误按 prompt 计。

**Comment：** 这是后续“一个 rollout 产生多条训练样本”问题的根。

---

## 6. add_eval_arguments — eval 默认继承 rollout

**Explain：** eval 参数允许单独指定 eval rollout function；不指定时在 validate 阶段继承 `rollout_function_path`。

**问题与约束：** 评估通常复用 rollout 逻辑，但可能需要不同数据集、采样参数或输出格式。

**设计选择：** `--eval-function-path` 默认 None，`--eval-interval` 重置为 None，避免 Megatron 默认 eval 行为误触发。

**代码逻辑：**

```python
## 来源：slime/utils/arguments.py L766-L775
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

**为什么这样写：** eval 是可选管线；默认不启动，启动时优先复用 rollout 逻辑，降低配置负担。

**不变量与失败模式：** 若设置 `eval_interval`，validate 要求 eval dataset 已配置；否则有评估调度但无数据源。

**Comment：** 默认值 None 在这里是有意义的“延迟决策”。

---

## 7. add_algo_arguments — loss 与 advantage

**Explain：** algo 参数段把训练目标、loss 类型、advantage estimator 和 KL/clip 参数集中定义。

**问题与约束：** 同一训练后端要支持 policy loss、SFT/custom loss，以及 GRPO/GSPO/CISPO/PPO 等 estimator。

**设计选择：** `loss_type` 控制 loss 函数入口；`advantage_estimator` 控制 advantage/returns 与 policy loss 分支；OPD 被说明为 orthogonal。

**代码逻辑：**

```python
## 来源：slime/utils/arguments.py L902-L944
parser.add_argument("--loss-type", choices=["policy_loss", "sft_loss", "custom_loss"], default="policy_loss")
parser.add_argument("--custom-loss-function-path", type=str, default=None)
...
parser.add_argument(
    "--advantage-estimator",
    choices=["grpo", "gspo", "cispo", "reinforce_plus_plus", "reinforce_plus_plus_baseline", "ppo"],
    default="grpo",
    help=(
        "Advantage estimator to use. Note: on-policy distillation (OPD) is now orthogonal "
        "to the advantage estimator..."
    ),
)
```

**为什么这样写：** loss 与 advantage 被分成两个轴，避免把每个算法组合都做成一个 monolithic mode。

**不变量与失败模式：** `custom_loss` 必须配 custom loss function；`ppo` 会在 validate 中开启 critic；某些 estimator 对 normalize/clip 有额外约束。

**Comment：** 这段和 [[22-Loss-Policy-02-源码走读]] 是直接相连的。

---

## 8. custom_advantage_function_path — 替换 advantages/returns

**Explain：** 自定义 advantage 函数替换内置 `compute_advantages_and_returns`，并要求原地写入 `rollout_data`。

**问题与约束：** 自定义算法可能不改变 rollout 或 loss，只改变 advantage/return 构造。

**设计选择：** 函数签名固定为 `def custom_fn(args, rollout_data) -> None`，通过 mutating rollout_data 接回训练管线。

**代码逻辑：**

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
        "The function should set rollout_data['advantages'] and rollout_data['returns'] in-place."
    ),
)
```

**为什么这样写：** advantage 是训练数据变换，不一定需要替换 loss；把它作为独立 hook 可以缩小实验改动面。

**不变量与失败模式：** 自定义函数必须写出 `advantages` 和 `returns`；shape/order 必须与 response token 对齐。

**Comment：** 这是 Slime 把 RL 算法拆成多段可替换协议的例子。

---

## 9. add_reward_model_arguments — RM 与 train data 转换

**Explain：** reward 参数段允许替换 reward function、reward post-process 和 samples→train_data 转换。

**问题与约束：** agentic/remote RM/多维 reward 场景下，reward 不一定是内置规则函数，样本到训练字段的映射也可能变化。

**设计选择：** reward、reward 后处理、训练数据转换分别是三个 hook，而不是一个大 hook。

**代码逻辑：**

```python
## 来源：slime/utils/arguments.py L1347-L1374
parser.add_argument("--custom-rm-path", type=str, default=None, help=(...))
parser.add_argument("--custom-reward-post-process-path", type=str, default=None, help=(...))
parser.add_argument(
    "--custom-convert-samples-to-train-data-path",
    type=str,
    default=None,
    help=(
        "Path to a custom function that converts samples to training data. "
        "If set, this function will replace the default _convert_samples_to_train_data. "
        "The function should have the signature `def convert_samples_to_train_data(args, samples) -> dict`."
    ),
)
```

**为什么这样写：** reward 计算、reward normalization、训练字段构造是三个不同阶段；分开才能让用户只替换需要的阶段。

**不变量与失败模式：** train data 转换 hook 返回的 dict 必须满足训练侧字段契约；reward post-process 的长度必须和 samples 对齐。

**Comment：** 这段参数最终会在 RolloutManager 初始化和转换函数里被消费。

---

## 10. dynamic_sampling_filter_path — 采样中筛选

**Explain：** dynamic sampling filter 在采样过程中判断一个 prompt group 是否保留，典型用途是 DAPO 风格过滤。

**问题与约束：** 某些 prompt 的多样本 reward 全对或全错，对 GRPO 类训练信号弱；过滤应发生在补齐 rollout batch 的阶段。

**设计选择：** 只暴露函数路径，签名和返回类型由 rollout/filter hub 契约定义。

**代码逻辑：**

```python
## 来源：slime/utils/arguments.py L441-L451
parser.add_argument(
    "--dynamic-sampling-filter-path",
    type=str,
    default=None,
    help=(
        "This is the filter function for dynamic sampling. "
        "It should be able to judge whether the result of a prompt should be selected or not."
        "We will do dynamic filter for sampling as in DAPO..."
    ),
)
```

**为什么这样写：** dynamic filter 改的是“是否继续采样/保留这个 group”，不是训练 loss mask；它应该位于 rollout 生成侧。

**不变量与失败模式：** filter 需要返回 rollout 代码可识别的结果；若过度过滤，可能导致采样无法及时凑够 batch。

**Comment：** 它和 buffer/sample filter 是三个不同层级，不要混用。

---

## 11. buffer_filter_path — buffer 级筛选

**Explain：** buffer filter 在 rollout buffer 里选择要进入训练的数据。

**问题与约束：** 部分算法需要在已生成样本池里做优先级、质量或分布平衡，而不是采样时立即决定。

**设计选择：** 函数接收 `list[list[Sample]]` 并返回同形态结构，让 buffer 级过滤保留 prompt/group 结构。

**代码逻辑：**

```python
## 来源：slime/utils/arguments.py L503-L512
parser.add_argument(
    "--buffer-filter-path",
    type=str,
    default=None,
    help=(
        "Path to the buffer filter function. "
        "It should be able to select the samples in the buffer. "
        "The function should take list[list[Sample]] and return list[list[Sample]]."
    ),
)
```

**为什么这样写：** buffer filter 是 data source/buffer 层的扩展点，不应直接改 loss mask。

**不变量与失败模式：** 返回结构必须仍能被后续 rollout data 转换处理；如果打散 group，reward normalization 和 rollout_id 统计可能失真。

**Comment：** 这是“数据进入训练前”的筛选。

---

## 12. rollout_sample_filter_path / all_samples_process_path

**Explain：** sample filter 通过修改 `Sample.remove_sample` 控制 loss 参与；all-samples process 可以处理包括 filtered ones 在内的所有样本。

**问题与约束：** 有些样本要从 loss 中排除，但仍需要进入日志、统计或 advantage normalization 语境。

**设计选择：** sample filter 不返回新列表，而是原地改 `remove_sample`；all-samples process 单独存在，避免过滤后信息丢失。

**代码逻辑：**

```python
## 来源：slime/utils/arguments.py L1415-L1435
parser.add_argument(
    "--rollout-sample-filter-path",
    type=str,
    default=None,
    help=(
        "Path to the rollout sample filter function. "
        "This function determines whether a sample will participate in loss calculation. "
        "Please directly modify the remove_sample attribute of Sample. "
        "Note: This attribute does not determine whether the sample participates in advantage normalization."
    ),
)
parser.add_argument("--rollout-all-samples-process-path", type=str, default=None, help=(...))
```

**为什么这样写：** loss 参与和样本存在是两回事；用 `remove_sample` 能让训练 mask 归零，同时保留样本用于其他统计。

**不变量与失败模式：** hook 必须直接修改 Sample；如果返回新对象但调用方不接收，就不会生效。

**Comment：** 这一点解释了为什么后续 train_data 转换里会把 `remove_sample` 转成全零 loss mask。

---

## 13. Megatron 插件路径

**Explain：** Megatron plugin 参数提供初始化、logprob 前、train step 前三个 hook。

**问题与约束：** 有些实验需要改 Megatron runtime 状态，而不是 rollout 或 loss 数据结构。

**设计选择：** 只暴露少数明确时机的 hook path，不开放任意 parser 注入。

**代码逻辑：**

```python
## 来源：slime/utils/arguments.py L1438-L1458
parser.add_argument("--custom-megatron-init-path", type=str, default=None)
parser.add_argument("--custom-megatron-before-log-prob-hook-path", type=str, default=None)
parser.add_argument("--custom-megatron-before-train-step-hook-path", type=str, default=None)
```

**为什么这样写：** Megatron 侧状态复杂，扩展点越窄越容易维护调用时机和参数契约。

**不变量与失败模式：** hook 必须匹配文档签名；在分布式训练中 hook 代码要能在每个相关 rank 上运行。

**Comment：** 这类 hook 不应该承担 rollout 数据转换职责。

---

## 14. add_slime_arguments — provider 组装顺序

**Explain：** Slime extra args provider 把所有子分组依次注册到 Megatron parser，并允许用户自定义参数先注册。

**问题与约束：** Slime 要扩展 Megatron parser，同时避免用户自定义参数被 Slime 默认值覆盖。

**设计选择：** `add_custom_arguments` 先执行，然后按 cluster/train/rollout/data/eval/algo/... 顺序注册，最后 reset `custom-config-path` 和 `padded-vocab-size`。

**代码逻辑：**

```python
## 来源：slime/utils/arguments.py L1495-L1525
if add_custom_arguments is not None:
    parser = add_custom_arguments(parser)

parser = add_cluster_arguments(parser)
parser = add_train_arguments(parser)
parser = add_rollout_arguments(parser)
...
parser = add_custom_megatron_plugins_arguments(parser)
reset_arg(parser, "--custom-config-path", type=str, default=None, help="Path to the YAML config for custom function arguments.")
reset_arg(parser, "--padded-vocab-size", type=int, default=None)
```

**为什么这样写：** provider 组装顺序就是参数命名空间的 ownership 顺序；先给用户扩展机会，再加 Slime 标准参数。

**不变量与失败模式：** `reset_arg` 会改已有 Megatron 参数默认值；读者不能只看 Megatron 原始默认。

**Comment：** 这是 Arguments 专题最容易漏掉的源码点：Slime 不是独立 parser，而是 Megatron parser 的 extra provider。

---

## 15. parse_args — 两阶段解析与合并

**Explain：** `parse_args` 先 pre-parse 少数影响流程的 flag，再独立解析 SGLang args，最后解析 Megatron+Slime args 并合并 namespace。

**问题与约束：** SGLang 的 CLI 来自 `ServerArgs.add_cli_args`，Megatron 也有自己的 parser；直接混在一个 parser 里容易冲突。

**设计选择：** debug-train-only 或 load-debug-rollout-data 时跳过 SGLang parse；Megatron parser 用 `ignore_unknown_args=True` 忽略 `--sglang-*`。

**代码逻辑：**

```python
## 来源：slime/utils/arguments.py L1546-L1589
pre = _pre_parse_mode()
skip_sglang = pre.debug_train_only or pre.load_debug_rollout_data is not None
...
if not skip_sglang:
    sglang_ns = sglang_parse_args()
...
args = megatron_parse_args(extra_args_provider=add_slime_arguments, skip_hf_validate=pre.debug_rollout_only)
...
for key, value in vars(pre).items():
    setattr(args, key, value)
if sglang_ns is not None:
    for key, value in vars(sglang_ns).items():
        setattr(args, key, value)
...
slime_validate_args(args)
```

**为什么这样写：** 解析本身是系统编排：SGLang、Megatron、Slime 各自有参数来源，最终训练只想拿到一个 `args`。

**不变量与失败模式：** pre-parsed flags 必须合并回最终 args；如果 debug_train_only 仍解析 SGLang，可能要求不必要的 rollout 参数。

**Comment：** 这解释了为什么一些参数在源码里看起来不在同一个 parser，却最终都能在 `args` 上访问。

---

## 16. slime_validate_args — RL 互斥与派生默认

**Explain：** validate 阶段把 RL 算法约束、动态 batch、CISPO clip、eval reward key 等运行时约束落地。

**问题与约束：** 很多参数只有组合起来才有意义，单个 `choices/default` 不能表达互斥和派生。

**设计选择：** 在 `slime_validate_args` 中 assert/改写：KL reward shaping 和 KL loss 不能同时开，TIS 与 rollout logprobs 互斥，dynamic batch 必须有 max tokens，CISPO clip 给 warning。

**代码逻辑：**

```python
## 来源：slime/utils/arguments.py L1796-L1835
assert not (args.kl_coef != 0 and args.kl_loss_coef != 0), "Only one of kl_coef and kl_loss_coef can be set"
...
if args.use_rollout_logprobs:
    assert not args.use_tis, "use_rollout_logprobs and use_tis cannot be set at the same time."
...
if args.use_dynamic_batch_size:
    assert args.max_tokens_per_gpu is not None
...
if args.advantage_estimator == "cispo" and args.eps_clip < 1.0:
    logger.warning("CISPO is canonically single-sided, but --eps-clip=%s keeps the lower clip bound %s active. ...")
```

**为什么这样写：** 参数合法性是算法语义的一部分；放在 validate 中可以在所有来源合并后统一判断。

**不变量与失败模式：** 如果自定义 config 后才覆盖参数，必须发生在相关 validate 前；互斥项不能只靠文档约束。

**Comment：** 这里的 warning/assert 是 Loss-Policy 笔记中很多分支的前置条件。

---

## 17. slime_validate_args — disk delta 合法组合

**Explain：** disk-backed weight sync 的必填路径和 delta 的非法组合也在 validate 阶段检查。

**问题与约束：** delta 只支持 disk transport，不支持 colocate；disk transport 必须有共享目录，delta 还必须有 host-local checkpoint dir。

**设计选择：** 对 `update_weight_transport == "disk"` 先检查 shared dir；对 `update_weight_mode == "delta"` 再检查 transport、colocate 和 local dir。

**代码逻辑：**

```python
## 来源：slime/utils/arguments.py L1980-L2002
if args.update_weight_transport == "disk" and not args.update_weight_disk_dir:
    raise ValueError("--update-weight-transport=disk requires --update-weight-disk-dir ...")
if args.update_weight_mode == "delta":
    if args.update_weight_transport != "disk":
        raise ValueError("--update-weight-mode=delta requires --update-weight-transport=disk ...")
    if args.colocate:
        raise ValueError("--update-weight-mode=delta is not supported with --colocate. ...")
    if not args.update_weight_local_checkpoint_dir:
        raise ValueError("--update-weight-mode=delta requires --update-weight-local-checkpoint-dir ...")
```

**为什么这样写：** 权重同步参数在 CLI 定义处保持组合灵活，在 validate 处把工程不可行组合剪掉。

**不变量与失败模式：** colocate 下权重走 CUDA IPC，delta bookkeeping 只会增加开销；没有 local checkpoint dir 时 rollout host 没地方 apply delta。

**Comment：** 这是 Arguments 和 WeightSync-Disk 两篇之间的约束闭环。

---

## 18. sglang_utils — add_sglang_arguments

**Explain：** SGLang 参数适配先加入 router 参数，再设置 Slime 需要的 router balance 默认值，并增加 Slime 自己的 concurrency 参数。

**问题与约束：** SGLang 自身参数很多，Slime 需要透传大部分，但也要覆盖少数适合 RL rollout 的默认值。

**设计选择：** `add_sglang_arguments` 包装 router + server args，而不是在 Slime 主 parser 中手写 SGLang 全量参数。

**代码逻辑：**

```python
## 来源：slime/backends/sglang_utils/arguments.py L35-L41
def add_sglang_arguments(parser):
    parser = add_sglang_router_arguments(parser)
    parser.set_defaults(router_balance_abs_threshold=10, router_balance_rel_threshold=1.2)
    parser.add_argument("--sglang-server-concurrency", type=int, default=512)
```

**为什么这样写：** SGLang 参数随上游演进，Slime 通过调用 `ServerArgs.add_cli_args` 复用上游定义，只在边界处加前缀/默认。

**不变量与失败模式：** 上游 SGLang 参数变更会影响 Slime 可用的 `--sglang-*` 参数；Slime 自己管理的字段不能重复透传。

**Comment：** 这是一种“包裹上游 CLI”的设计，而不是 fork 一份参数表。

---

## 19. sglang_utils — skipped_args

**Explain：** `skipped_args` 列出 Slime 自己管理的 SGLang server fields，例如 model_path、tp_size、port、node_rank、dist_init_addr。

**问题与约束：** 这些字段由 Ray placement、rollout topology、checkpoint 和启动逻辑计算，不能让用户通过 `--sglang-*` 直接覆盖。

**设计选择：** 在 monkey-patched `add_argument` wrapper 中遇到 canonical name 属于 skip list 就不注册。

**代码逻辑：**

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

**为什么这样写：** Slime 需要保证 rollout engine 的拓扑和训练 orchestration 一致；允许用户覆盖这些字段会破坏 placement 和通信地址。

**不变量与失败模式：** skip list 必须覆盖所有由 Slime runtime 注入的字段；漏掉字段可能导致用户配置和 Ray 启动逻辑冲突。

**Comment：** “前缀透传”不是无限透传，skip list 是 ownership 边界。

---

## 20. ServerArgs.add_cli_args wrapper

**Explain：** Slime 临时替换 `parser.add_argument`，让 SGLang server args 自动变成 `--sglang-*` 并改写 dest。

**问题与约束：** 上游 `ServerArgs.add_cli_args` 不知道 Slime 的前缀约定；Slime 又不想复制上游参数定义。

**设计选择：** monkey patch `parser.add_argument`，调用 `ServerArgs.add_cli_args(parser)` 后恢复原函数。

**代码逻辑：**

```python
## 来源：slime/backends/sglang_utils/arguments.py L113-L115
parser.add_argument = new_add_argument_wrapper
ServerArgs.add_cli_args(parser)
parser.add_argument = old_add_argument
```

**为什么这样写：** 这是低成本适配上游 CLI 的桥接层，既复用上游参数，又把 Slime namespace 保持在 `sglang_` 前缀下。

**不变量与失败模式：** wrapper 必须在调用后恢复；否则后续 Slime 自己的参数也会被错误前缀化。

**Comment：** 这段虽短，但解释了 `args.sglang_*` 字段的来源。

---

## 21. sglang_config 与 PD 参数

**Explain：** SGLang config 和 prefill server 参数用于 PD disaggregation 或多 server group 部署。

**问题与约束：** 简单 `prefill_num_servers` 和 YAML `sglang_config` 都能描述 rollout 拓扑，但两者不能同时作为来源。

**设计选择：** CLI 支持 `--prefill-num-servers` 简化路径，也支持 `--sglang-config` YAML 描述 worker_type、num_gpus、overrides、placeholder。

**代码逻辑：**

```python
## 来源：slime/backends/sglang_utils/arguments.py L117-L136
parser.add_argument("--prefill-num-servers", type=int, default=None, help="Number of prefill servers...")
parser.add_argument(
    "--sglang-config",
    type=str,
    default=None,
    help=(
        "Path to a YAML config for SGLang engine deployment. "
        "Defines server_groups with worker_type (regular/prefill/decode/placeholder), "
        "num_gpus per group, and optional per-group 'overrides' dict..."
    ),
)
```

**为什么这样写：** 简单场景给一个 flag，复杂拓扑给 YAML；两者共享后续 `SglangConfig` 解析。

**不变量与失败模式：** `sglang_config` 的总 GPU 数必须和 `rollout_num_gpus` 对齐，后续 `start_rollout_servers` 会验证。

**Comment：** 参数层预留了 regular/prefill/decode/placeholder 等多组 rollout 形态。

---

## 22. sglang_validate_args — PD / external 互斥

**Explain：** SGLang validate 把 dp/pp/tp 派生值和 PD/external/sglang_config 的互斥关系落地。

**问题与约束：** external rollout engines 已经由用户提供拓扑，不能再由 Slime 的 prefill_num_servers 或 sglang_config 同时描述。

**设计选择：** 设置 `sglang_dp_size/pp_size/ep_size` 派生字段；检查 `prefill_num_servers`、`sglang_config`、`rollout_external` 互斥。

**代码逻辑：**

```python
## 来源：slime/backends/sglang_utils/arguments.py L162-L173
assert not (
    getattr(args, "prefill_num_servers", None) is not None and getattr(args, "rollout_external", False)
), "prefill_num_servers cannot be set with --rollout-external-engine-addrs."

assert not (
    getattr(args, "sglang_config", None) is not None and getattr(args, "rollout_external", False)
), "sglang_config cannot be set with --rollout-external-engine-addrs."

assert not (
    getattr(args, "sglang_config", None) is not None and getattr(args, "prefill_num_servers", None) is not None
), "sglang_config and prefill_num_servers are mutually exclusive..."
```

**为什么这样写：** rollout topology 只能有一个权威来源；多个来源同时存在会让 Ray placement 和 external server 信息互相冲突。

**不变量与失败模式：** PP size > 1 时 rollout GPUs per engine 必须能整除 PP size；DP size > 1 要求 SGLang DP attention。

**Comment：** 这里的互斥是部署拓扑约束，不是算法约束。

---

## 23. megatron_utils — HF 对齐 validate

**Explain：** Megatron 参数解析后会拿 HF config 对齐 hidden size、head 数、layer 数、FFN/MoE、embedding tying、norm eps 等关键结构。

**问题与约束：** Slime 用 HF checkpoint 初始化 rollout/tokenizer，又用 Megatron 训练；二者结构不一致会在权重转换或 logprob 对齐时失败。

**设计选择：** `_hf_validate_args` 针对字段列表逐项比较，MoE/multimodal/rope_theta 有特殊处理。

**代码逻辑：**

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
    ...
```

**为什么这样写：** 参数层提前阻止“看似能启动、实际权重不可对齐”的训练配置。

**不变量与失败模式：** HF config 和 Megatron args 必须描述同一架构；debug_rollout_only 会跳过 HF validate，因为此时不启动训练。

**Comment：** 这是 `--hf-checkpoint` 被用作架构基准的源码证据。

---

## 24. Megatron 默认 RL 友好设置

**Explain：** Slime 在 Megatron parse 后设置一批适合 RL 训练的默认值，包括 distributed optimizer、bf16、checkpoint I/O、seq length、tokenizer。

**问题与约束：** Megatron 原生默认值面向通用预训练；Slime 需要更适合 RLHF/rollout 闭环的默认配置。

**设计选择：** `_set_default_megatron_args` 在 parse 后统一改写 args，而不是在 CLI 每个参数上重写一遍。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/arguments.py L147-L168
args.use_distributed_optimizer = True
args.bf16 = not args.fp16
args.use_persistent_ckpt_worker = True
args.ckpt_assume_constant_structure = True
args.ckpt_fully_parallel_load = True
if args.seq_length is None:
    args.seq_length = 4096
args.max_position_embeddings = args.seq_length
...
if args.vocab_size and not args.padded_vocab_size:
    args.padded_vocab_size = _vocab_size_with_padding(args.vocab_size, args)
```

**为什么这样写：** 默认值需要基于解析后的上下文和 Megatron 字段存在性来决定，集中在 post-parse 函数更清楚。

**不变量与失败模式：** 如果用户以为 Megatron 原始默认仍生效，会误判实际训练配置；tokenizer_model 默认会落到 `hf_checkpoint`。

**Comment：** 参数源码中“默认值”的真实位置有两个：`add_argument(default=...)` 和 validate/post-parse 改写。

---

## 25. customization.md — 接口总表

**Explain：** customization 文档把 CLI 中的 path 参数整理成用户可见的扩展接口。

**问题与约束：** 单看源码里的 `add_argument` 很难知道哪个 hook 应该用于哪类自定义。

**设计选择：** 文档按接口列出 rollout、generate、RM、filter、loss、data source、eval、Megatron hooks 等扩展点。

**代码逻辑：**

```markdown
## 来源：docs/en/get_started/customization.md L9-L30
| `--rollout-function-path` | Override the entire rollout generation logic. |
| `--custom-generate-function-path` | Override only the generation step ... |
| `--custom-rm-path` | Implement custom reward computation logic. |
| `--dynamic-sampling-filter-path` | Filter samples during dynamic sampling ... |
| `--buffer-filter-path` | Filter samples in the rollout buffer before training. |
| `--custom-loss-function-path` | Implement custom training loss computation. |
| `--data-source-path` | Override the data source for rollout prompts. |
| `--eval-function-path` | Override the rollout function specifically for evaluation. |
| `--custom-megatron-init-path` | Custom initialization after Megatron setup. |
```

**为什么这样写：** 文档把源码中的参数表转译成“什么时候用哪个扩展点”，避免用户一上来替换整条管线。

**不变量与失败模式：** 文档必须和 CLI 参数保持同步；否则用户会配置存在但未被运行时消费的路径，或反过来漏掉新 hook。

**Comment：** 对源码阅读者来说，这张表是 path 参数的索引，而源码是执行证据。

---

## 26. RolloutManager 如何消费 path

**Explain：** RolloutManager 初始化时把 `data_source_path`、`rollout_function_path`、`eval_function_path` 和部分 post-process path 通过 `load_function` 变成 callable。

**问题与约束：** CLI 只保存字符串；运行时必须在 Ray actor 内导入函数，保证自定义逻辑在 rollout 进程环境中执行。

**设计选择：** manager 启动 server 后加载 data source 和 rollout functions；reward post-process 与 samples-to-train-data hook 只有在不为 None 时加载。

**代码逻辑：**

```python
## 来源：slime/ray/rollout.py L437-L451
data_source_cls = load_function(self.args.data_source_path)
self.data_source = data_source_cls(args)

self.generate_rollout = load_function(self.args.rollout_function_path)
self.eval_generate_rollout = load_function(self.args.eval_function_path)
if self.args.custom_reward_post_process_path is not None:
    self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
if self.args.custom_convert_samples_to_train_data_path is not None:
    self.custom_convert_samples_to_train_data_func = load_function(...)
```

**为什么这样写：** 它把扩展点从“配置字符串”变成“Ray actor 内的可调用对象”，并把加载失败提前到 manager 初始化阶段。

**不变量与失败模式：** path 必须在 rollout actor 的 Python 环境可 import；函数签名错误通常会在调用阶段暴露。

**Comment：** 这是判断某个 CLI path 是否真实生效的关键证据。

---

## 27. plugin_contracts 测试入口

**Explain：** customization 文档给出 CPU-only contract tests，用于验证插件路径加载和签名契约。

**问题与约束：** path hook 的失败往往到分布式训练时才暴露；contract tests 希望在本地先检查导入和接口形状。

**设计选择：** 将 rollout/generate/path-loading/runtime-hook tests 分组，用户可以直接 pytest。

**代码逻辑：**

```markdown
## 来源：docs/en/get_started/customization.md L475-L481
python -m pytest \
  tests/plugin_contracts/test_plugin_rollout_contracts.py \
  tests/plugin_contracts/test_plugin_generate_contracts.py \
  tests/plugin_contracts/test_plugin_path_loading_contracts.py \
  tests/plugin_contracts/test_plugin_runtime_hook_contracts.py
```

**为什么这样写：** path-based plugin 的边界是动态导入，测试必须覆盖“字符串能否解析成 callable”而不只是训练逻辑。

**不变量与失败模式：** contract tests 只能验证接口契约，不保证业务逻辑正确；仍需要端到端 rollout/training 验证。

**Comment：** 这让自定义接口从“文档承诺”变成“可测试契约”。

---

## 走读小结

| 层次 | 文件 | 设计职责 |
|------|------|----------|
| CLI provider | `slime/utils/arguments.py` | 定义 Slime 训练、rollout、算法、hook 参数 |
| parse/merge | `parse_args` | 合并 pre-parse、SGLang、Megatron+Slime namespace |
| validate | `slime_validate_args` / backend validate | 将组合关系、默认派生和互斥约束落地 |
| runtime consumption | `slime/ray/rollout.py` | 将 path 字符串加载为 Ray actor 内 callable |
| 用户契约 | `customization.md` | 说明每个 hook 的用途、签名和 contract tests |

Arguments 这一层的核心哲学是：**CLI 不是配置清单，而是跨训练、rollout、分布式部署和插件系统的协议入口**。

→ [[04-Arguments-TrainRollout-03-数据流与交互]]
