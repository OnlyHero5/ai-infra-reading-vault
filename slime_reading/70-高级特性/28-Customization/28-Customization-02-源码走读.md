---
type: batch-doc
module: 28-Customization
batch: "28"
doc_type: walkthrough
title: "Customization · 源码走读"
tags:
  - slime/batch/28
  - slime/module/customization
  - slime/doc/walkthrough
updated: 2026-07-05
---

# Customization · 源码走读

> 走读顺序：`load_function` → `RolloutManager` → `sglang_rollout.generate_and_rm` → `rm_hub` → `MegatronTrainRayActor` → `agent.parsing` → `agent.harness`

Customization 的核心设计不是“到处允许用户传函数”，而是把扩展点放在几个稳定边界上：启动期 import path 解析、rollout 外层编排、单样本生成、reward model、sample-to-train-data 转换、训练侧 postprocess、agent harness。这样用户可以替换策略，而 Slime 仍保留调度、日志、DP split、loss reducer 和 CPU-only contract tests 的共同约束。

---

## 1. 统一的函数路径解析

### 1.1 load_function 是所有 path hook 的入口

**问题与约束：** 自定义接口通过 CLI 字符串传入，运行时必须把 `package.module.function` 解析为可调用对象；如果路径错误，应在初始化阶段暴露，而不是 silent fallback。

**设计选择：** `load_function` 只做两步：用 `rpartition(".")` 拆出 module path 和 attr，再 `importlib.import_module` 并 `getattr`。

**Explain：** 这个实现非常薄，说明 Slime 把“路径解析”当成共同机制，而把“签名、返回值、副作用”交给各调用点和 contract tests 约束。

**Code：**

```python
## 来源：slime/utils/misc.py L37-L45
def load_function(path):
    """
    Load a function from a module.
    :param path: The path to the function, e.g. "module.submodule.function".
    :return: The function object.
    """
    module_path, _, attr = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)
```

**代码逻辑：** 函数不捕获 `ImportError` 或 `AttributeError`；模块不存在、属性不存在、path 格式错误都会直接抛出。

**为什么这样写：** 自定义函数是训练语义的一部分，启动期 fail-fast 比运行一段时间后发现默认逻辑被误用更安全。

**不变量与失败模式：** path 必须包含可 import 的模块和属性名；`load_function` 不检查返回对象是否 async，也不检查签名，调用方需要按自己的接口契约调用。

**Comment：** 读后续 hook 时，先看它是在初始化时 load，还是每次请求/每步训练时 load；这决定了错误暴露和热路径开销。

---

## 2. Rollout 侧扩展点

### 2.1 RolloutManager 初始化时挂载外层 hook

**问题与约束：** RolloutManager 同时负责数据源、rollout 生成、eval 生成、reward 后处理和 sample-to-train-data 转换；这些是跨 rollout step 的策略，不应在每个 sample 的热路径里重复解析。

**设计选择：** `RolloutManager.__init__` 在启动期 load `data_source_path`、`rollout_function_path`、`eval_function_path`，并按需 load reward postprocess 和 convert hook。

**Explain：** 这是 Slime 的第一层自定义边界：替换外层编排和数据转换。它比 `custom_generate` 更强，但也承担更多完整性责任。

**Code：**

```python
## 来源：slime/ray/rollout.py L437-L449
data_source_cls = load_function(self.args.data_source_path)
self.data_source = data_source_cls(args)

self.generate_rollout = load_function(self.args.rollout_function_path)
self.eval_generate_rollout = load_function(self.args.eval_function_path)
self.custom_reward_post_process_func = None
if self.args.custom_reward_post_process_path is not None:
    self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
self.custom_convert_samples_to_train_data_func = None
if self.args.custom_convert_samples_to_train_data_path is not None:
    self.custom_convert_samples_to_train_data_func = load_function(...)
```

**代码逻辑：** 数据源类先实例化；train/eval rollout 函数保存为成员；两个可选 postprocess/convert hook 只有在 path 非空时才 load。

**为什么这样写：** 外层 hook 的粒度是 rollout step，不是单请求。初始化时固定这些函数，可以让后续 `generate()` 只调用已解析的 callable。

**不变量与失败模式：** `data_source_path`、`rollout_function_path`、`eval_function_path` 都是必需路径；可选 path 写错也会在 actor 初始化阶段失败。

**Comment：** 如果只是给每个 sample 加工具调用或 RAG，优先用 `custom_generate_function_path`；替换 `rollout_function_path` 意味着要自己维护外层数据形状。

### 2.2 custom_generate 是单样本生成替换点

**问题与约束：** agent、RAG、tool use 等常见需求只想替换单个 sample 如何生成，不想重写 oversampling、dynamic filter、group RM、abort 和 buffer 回填。

**设计选择：** 默认 `generate_and_rm` 在 semaphore 和 DP rank context 内优先读取 `sample.generate_function_path`，其次用全局 `args.custom_generate_function_path`；没有自定义函数才走内置 `generate`。

**Explain：** 这是最推荐的扩展层：保留默认 rollout 外循环，只替换“一个样本如何变成一个或多个训练样本”。

**Code：**

```python
## 来源：slime/rollout/sglang_rollout.py L249-L260
with state.dp_rank_context() as _:
    custom_func_path = getattr(sample, "generate_function_path", None) or args.custom_generate_function_path

    if custom_func_path is not None:
        custom_generate_func = load_function(custom_func_path)
        if "evaluation" in inspect.signature(custom_generate_func).parameters:
            sample = await custom_generate_func(args, sample, sampling_params, evaluation=evaluation)
        else:
            sample = await custom_generate_func(args, sample, sampling_params)
    else:
        sample = await generate(args, sample, sampling_params)
```

**代码逻辑：** per-sample path 覆盖全局 path；函数每次调用时解析；如果签名里有 `evaluation` 参数就传入 eval 标志；所有路径都 `await`。

**为什么这样写：** eval dataset 可能需要单独的 generate hook，per-sample path 提供最高优先级；`evaluation` 参数用签名探测保持向后兼容。

**不变量与失败模式：** 自定义 generate 必须是 awaitable；返回可以是 `Sample` 或 `list[Sample]`，但 fan-out 样本必须遵守 rollout_id 语义，否则后续 loss 聚合会失真。

**Comment：** 这段在 semaphore 内执行，说明自定义生成也会受 rollout 并发限制保护。

### 2.3 custom_rm 的 batch 短路

**问题与约束：** reward 既可能是单样本规则，也可能需要同组样本一起计算；尤其 group RM 不能拆成独立 sample 调用。

**设计选择：** `batched_async_rm` 如果设置 `args.custom_rm_path`，直接把整个 `samples` 列表交给自定义函数；否则才 fallback 到 per-sample `async_rm` 并发 gather。

**Explain：** 这使 `--custom-rm-path` 同时支持普通 RM 和 group RM，但调用方必须按当前模式实现正确签名。

**Code：**

```python
## 来源：slime/rollout/rm_hub/__init__.py L97-L99
if args.custom_rm_path is not None:
    rm_function = load_function(args.custom_rm_path)
    return await rm_function(args, samples, **kwargs)
```

**代码逻辑：** 只要全局 custom RM 存在，batch 入口就不再逐样本调用内置 RM；自定义函数接收完整 samples 和额外 kwargs。

**为什么这样写：** group-level reward 经常依赖多条响应之间的相对关系，强行拆成 per-sample 会丢掉上下文。

**不变量与失败模式：** batch RM 必须返回与 `samples` 等长的 reward list；返回长度错位会在后续 zip 或训练数据转换中造成 reward 对错样本。

**Comment：** 单样本 custom RM 的优先级在 `async_rm` 内处理；batch 入口体现的是 group 模式下的整体短路。

---

## 3. 训练侧扩展点

### 3.1 rollout_data_postprocess 的加载

**问题与约束：** 有些训练侧修改必须在 rollout 数据进入 Megatron actor 后进行，例如根据 logprob、mask、metadata 再改训练 batch；这不能只放在 rollout 侧 sample 阶段。

**设计选择：** `MegatronTrainRayActor.init` 在初始化末尾加载 `rollout_data_postprocess_path`，保存到 `self.rollout_data_postprocess`。

**Explain：** 这个 hook 属于 train actor，而不是 RolloutManager。它看到的是转换后的 rollout_data，而不是原始 `Sample`。

**Code：**

```python
## 来源：slime/backends/megatron_utils/actor.py L180-L184
self.rollout_data_postprocess = None
if self.args.rollout_data_postprocess_path is not None:
    from slime.utils.misc import load_function

    self.rollout_data_postprocess = load_function(self.args.rollout_data_postprocess_path)
```

**代码逻辑：** actor 初始化时默认置空 hook；path 非空时局部导入 `load_function` 并解析 callable。

**为什么这样写：** train actor 才知道 Megatron/CP/VPP 数据形态；把 hook 放在这里，可以让用户修改训练 batch 而不侵入 rollout 数据源。

**不变量与失败模式：** hook 在 actor 初始化期解析；如果 path 错误，训练 actor 不会完成初始化。

**Comment：** 这和 `custom_convert_samples_to_train_data_path` 的区别是：前者处理 train_data dict，后者处理 Sample 到 train_data 的转换。

### 3.2 rollout_data_postprocess 的调用时机

**问题与约束：** postprocess 如果太早，会看不到 actor/ref/teacher logprob 或 advantage 相关字段；如果太晚，又可能错过日志和训练输入。

**设计选择：** actor 训练路径在 `compute_advantages_and_returns` 之后、`log_rollout_data` 和 `train` 之前调用 `self.rollout_data_postprocess`。

**Explain：** 这是一个有意选择的夹层：hook 可以修改即将被日志和训练同时看到的数据。

**Code：**

```python
## 来源：slime/backends/megatron_utils/actor.py L511-L512
if self.rollout_data_postprocess is not None:
    self.rollout_data_postprocess(self.args, rollout_id, rollout_data)
```

**代码逻辑：** hook 接收 `args`、`rollout_id` 和可变的 `rollout_data`；没有返回值约束，默认按副作用修改数据。

**为什么这样写：** 对训练 batch 的后处理常常需要原地改 mask、metadata 或额外字段；让日志和 train 共享同一份修改后的数据，避免二者看到不同语义。

**不变量与失败模式：** hook 必须保持 `tokens`、`response_lengths`、`loss_masks` 等核心字段长度一致；否则后续 Megatron batch 构造或 loss reducer 会失败。

**Comment：** 这也是危险扩展点：它离训练很近，适合小范围修正，不适合重写整个 rollout 语义。

### 3.3 Megatron hooks 的接口契约

**问题与约束：** 有些用户想在 Megatron 初始化、logprob 前或 train step 前插入逻辑；这些 hook 的对象和时机不同，不能用一个模糊签名覆盖。

**设计选择：** 文档把 Megatron hooks 拆成 init、before-log-prob、before-train-step 三类，并分别给出参数。

**Explain：** 这类 hook 不改变 rollout 数据形状，而是在 Megatron 栈内部插入控制逻辑，例如额外初始化、记录模型状态、调整 optimizer 或冻结层。

**Code：**

```python
## 来源：docs/en/get_started/customization.md L421-L443
def custom_init(args) -> None

def custom_hook(args, model, store_prefix) -> None

def custom_hook(args, rollout_id, step_id, model, optimizer, opt_param_scheduler) -> None
```

**代码逻辑：** init hook 只拿 `args`；logprob hook 拿模型和 `store_prefix`；train-step hook 还能拿 rollout/step id、optimizer 和 scheduler。

**为什么这样写：** 不同阶段能安全暴露的对象不同。把签名分开，可以限制用户 hook 的作用域，减少误用。

**不变量与失败模式：** hook 不应破坏 Megatron model/optimizer 的分布式状态；尤其 before-train-step hook 如果改 optimizer，应保证所有 rank 一致。

**Comment：** 训练侧 hook 比 rollout 侧 hook 更接近底层并行状态，读者应优先查对应 train step 源码再使用。

---

## 4. 文档化的接口契约

### 4.1 rollout_function 替换整个外层循环

**问题与约束：** 完整替换 rollout 需要返回 Slime 能理解的 train/eval 输出，而不仅是样本列表；否则 RolloutManager 无法继续日志、debug dump 和 DP split。

**设计选择：** 文档要求 `generate_rollout(args, rollout_id, data_source, evaluation=False)` 返回 `RolloutFnTrainOutput | RolloutFnEvalOutput`。

**Explain：** 这是最高权限的 rollout hook，适合 multi-agent 或完全自定义采样流程，不适合只改单样本生成。

**Code：**

```python
## 来源：docs/en/get_started/customization.md L58-L59
def generate_rollout(args, rollout_id, data_source, evaluation=False) -> RolloutFnTrainOutput | RolloutFnEvalOutput
```

**代码逻辑：** 函数接收全局 args、当前 rollout id、数据源和 eval 标志；返回值必须区分 train/eval 输出类型。

**为什么这样写：** 外层函数需要自己决定如何取样本、如何处理 eval、如何返回 metrics；统一返回类型让 RolloutManager 后续仍能接上。

**不变量与失败模式：** train 路径返回的 samples 必须最终可展平为 `Sample` 序列；fan-out 结构需要遵守 rollout_id contract。

**Comment：** 一旦替换这个函数，就要自己维护 abort、partial rollout、dynamic filter 等默认外循环行为。

### 4.2 custom_rm 的单样本与 batch 签名

**问题与约束：** reward hook 同时服务普通 sample RM 和 group RM；文档必须区分两种签名，否则用户会在 group 模式下返回错误结构。

**设计选择：** 文档给出 `custom_rm(args, sample)` 和 `batched_custom_rm(args, samples)` 两种签名。

**Explain：** 单样本签名更适合 rule-based 或 remote RM；batch 签名适合需要同组比较的 verifier。

**Code：**

```python
## 来源：docs/en/get_started/customization.md L131-L136
async def custom_rm(args, sample: Sample) -> float

async def batched_custom_rm(args, samples: list[Sample]) -> list[float]
```

**代码逻辑：** 两种函数都是 async；单样本返回一个 reward，batch 返回与 samples 对齐的 reward list。

**为什么这样写：** Reward 可能是慢 IO 或外部服务，async 是默认契约；batch 模式避免重复网络调用和丢失组上下文。

**不变量与失败模式：** batch 返回长度必须等于输入样本数；单样本返回 dict 或多维 reward 时，后续 `reward_key` 选择逻辑需要能解释。

**Comment：** 如果训练算法依赖 reward normalization，hook 返回值还会影响 `_post_process_rewards` 的归一化结果。

### 4.3 dynamic sampling filter 返回 keep/reason

**问题与约束：** dynamic sampling filter 不只是丢弃样本，还需要给日志说明丢弃原因，方便判断是否过度过滤。

**设计选择：** 文档约定返回 `DynamicFilterOutput`，包含 `keep` 和 `reason`。

**Explain：** 这是一个控制流 hook：它决定 group 是否进入训练数据池，而不是修改 sample 内容本身。

**Code：**

```python
## 来源：docs/en/get_started/customization.md L168-L172
@dataclass
class DynamicFilterOutput:
    keep: bool
    reason: str | None
```

**代码逻辑：** `keep=False` 表示当前 group 被过滤；`reason` 用于 metric gatherer 统计原因。

**为什么这样写：** 过滤策略如果只返回 bool，训练曲线异常时很难知道是 reward 全同、质量阈值还是 curriculum 条件导致。

**不变量与失败模式：** filter 应以 group 为单位决策；如果在 group 内只过滤部分样本，会破坏 `n_samples_per_prompt` 相关统计。

**Comment：** 这类 hook 更适合 DAPO/GRPO 样本筛选，不适合调整单 token loss mask。

### 4.4 rollout_sample_filter 的副作用契约

**问题与约束：** 有些样本不应参与 loss，但仍要保留在 rollout 统计或数据结构里；直接删除样本会改变 group 结构。

**设计选择：** 文档要求 sample filter 返回 `None`，直接修改每个 `Sample.remove_sample`。

**Explain：** 这是“保留样本、屏蔽 loss”的设计，而不是“从 batch 删除样本”。后续转换会把 `remove_sample` 转成全 0 loss mask。

**Code：**

```python
## 来源：docs/en/get_started/customization.md L209-L211
def filter_function(args, samples: list[Sample]) -> None
```

**代码逻辑：** 函数没有返回值；注释说明需要直接修改 `Sample.remove_sample` 字段。

**为什么这样写：** 删除样本会影响 rollout_id、reward normalization 和 DP schedule；改 loss mask 能保留统计边界。

**不变量与失败模式：** filter 必须原地修改样本；如果返回新列表但不改原对象，默认转换路径不会看到过滤结果。

**Comment：** 这个契约和 `rollout_data_postprocess` 类似，都是明确依赖副作用。

### 4.5 pg_loss reducer 只替换 policy loss 归约

**问题与约束：** 用户可能只想改变 policy gradient loss 的归一化方式，例如固定分母或 per-token/per-sample 切换，不想重写整个 loss。

**设计选择：** 文档定义 `get_pg_loss_reducer(total_lengths, response_lengths, loss_masks, calculate_per_token_loss=False)`，返回一个张量 reducer。

**Explain：** 这个 hook 粒度很窄：它改变 pg_loss 如何 reduce，而不是改变 advantage、KL、entropy 等全部指标。

**Code：**

```python
## 来源：docs/en/get_started/customization.md L288-L294
def get_pg_loss_reducer(
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    calculate_per_token_loss: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]
```

**代码逻辑：** hook 接收长度与 mask 信息，返回一个可应用到 loss tensor 的 reducer callable。

**为什么这样写：** reducer 需要知道有效 token 分布，但不应拿到 optimizer 或模型；保持纯函数形态更容易测试。

**不变量与失败模式：** reducer 的输出标量语义必须与训练日志一致；否则 pg_loss 和 clip/entropy 等默认指标会不再可比。

**Comment：** 修改 reducer 前应先确认当前 loss 类型是否真的走 pg_loss 路径。

### 4.6 DataSource 必需方法

**问题与约束：** 自定义数据源不只是提供样本，还要支持被 rollout buffer 回填、保存、恢复和长度统计。

**设计选择：** 文档要求实现 `get_samples`、`add_samples`、`save`、`load`、`__len__`。

**Explain：** 这是 rollout 外层可恢复性的契约：partial rollout、checkpoint 和 epoch 计算都依赖数据源不是一次性 iterator。

**Code：**

```python
## 来源：docs/en/get_started/customization.md L387-L401
class CustomDataSource(DataSource):
    def get_samples(self, num_samples: int) -> list[list[Sample]]: ...
    def add_samples(self, samples: list[list[Sample]]): ...
    def save(self, rollout_id): ...
    def load(self, rollout_id=None): ...
    def __len__(self): ...
```

**代码逻辑：** `get_samples` 取 prompt group；`add_samples` 回填未完成或复用样本；`save/load` 支持 checkpoint；`__len__` 支持 epoch/rollout 数计算。

**为什么这样写：** RL rollout 不是静态 dataset scan，样本可能因为 abort、filter、partial rollout 被放回池中；数据源必须支持双向流动。

**不变量与失败模式：** 返回形状应是 `list[list[Sample]]`；如果直接返回扁平列表，默认 rollout 生成和 group RM 会误解 group 维度。

**Comment：** 这是替换 `data_source_path` 时最容易漏的契约。

### 4.7 contract tests 是接口回归网

**问题与约束：** customization hook 分散在 rollout、RM、runtime hook 和 path loading 多处，靠人工试跑完整 RL 任务成本太高。

**设计选择：** 文档列出 CPU-only contract tests，按 hook 形状分成 rollout、generate、path loading、runtime hook 四组。

**Explain：** 这些测试不证明训练效果，但能验证 import path、签名、返回结构和关键副作用是否符合 Slime 运行时要求。

**Code：**

```bash
## 来源：docs/en/get_started/customization.md L476-L481
python -m pytest \
  tests/plugin_contracts/test_plugin_rollout_contracts.py \
  tests/plugin_contracts/test_plugin_generate_contracts.py \
  tests/plugin_contracts/test_plugin_path_loading_contracts.py \
  tests/plugin_contracts/test_plugin_runtime_hook_contracts.py
```

**代码逻辑：** 四个测试文件分别覆盖不同 hook 家族；可以整体跑，也可以单文件直接执行。

**为什么这样写：** 扩展接口的风险主要是边界形状错，而不是 GPU kernel 错；CPU-only contract tests 能在 PR 早期捕捉这类错误。

**不变量与失败模式：** 通过 contract tests 不代表 hook 业务逻辑正确；它只说明 hook 能被 Slime 按约定加载和调用。

**Comment：** 自定义 hook 上线前，先把自己的 module path 替换进这些测试，比直接跑大规模训练更可控。

---

## 5. Agent customization：解析与 sandbox harness

### 5.1 SGLang FunctionCallParser 优先解析工具调用

**问题与约束：** Agent 生成可能混合 reasoning、visible text 和 tool call；不同模型的 tool-call 格式不同，解析失败还需要保留可诊断状态。

**设计选择：** `parse_tool_uses` 在配置了 `tool_parser_name` 且存在 tools schema 时，构造 SGLang `FunctionCallParser`；解析出的参数用 JSON loads，失败时保留 raw arguments 并标记 `ill_formed`。

**Explain：** 这把格式解析交给 SGLang 的标准 parser，Slime 只负责把结果整理成 harness-agnostic 的 `{name, input}`。

**Code：**

```python
## 来源：slime/agent/parsing.py L67-L85
if parser.has_tool_call(body_text):
    try:
        body_text, calls = parser.parse_non_stream(body_text)
    except Exception:
        logger.exception("[agent.parsing] sglang tool-call parsing failed; falling back")
for c in calls:
    try:
        args = json.loads(c.parameters or "{}")
    except json.JSONDecodeError:
        args = {"_raw_arguments": c.parameters}
        ill_formed = True
    tool_uses.append({"name": c.name or "tool", "input": args})
```

**代码逻辑：** parser 命中才 parse；parse 异常会记录并继续；每个 call 的参数尝试 JSON 解析，失败则保存原始参数。

**为什么这样写：** Agent 训练不能因为一个工具调用格式坏掉就丢失整段输出；保留 `ill_formed` 能把格式错误变成可奖励或可调试的信号。

**不变量与失败模式：** `tools_schema` 必须能转换成 SGLang `Tool`；如果 schema 名称和模型输出不一致，parser 可能无法产生 tool call。

**Comment：** 这里的 fallback 不是吞错，而是把失败降级为结构化输出，方便 RM 或 harness 再处理。

### 5.2 XML tool fallback

**问题与约束：** 一些 coding-agent 模型会输出 Anthropic 风格 XML tool call；如果只支持 JSON/function-call parser，会漏掉这些可执行动作。

**设计选择：** 当标准 parser 没有得到 tool call 且存在 tools schema 时，`parse_xml_tool_uses` 用正则扫描 `<tool_call><function=...>` 片段，并只接受 schema 中声明过的 tool name。

**Explain：** XML fallback 是兼容层，不是主协议。它让旧模型或跨框架模型仍能接入 Slime agent rollout。

**Code：**

```python
## 来源：slime/agent/parsing.py L99-L110
for m in re.finditer(
    r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
    body_text,
    flags=re.DOTALL,
):
    name, inner = m.group(1), m.group(2)
    if name in valid_tools:
        args = {p.group(1): p.group(2).strip() for p in re.finditer(...)}
        tool_uses.append({"name": name, "input": args})
```

**代码逻辑：** 正则找到 tool_call block；tool name 必须在 schema 中；参数从 `<parameter=...>` 标签抽取；匹配片段从 visible text 中移除。

**为什么这样写：** fallback 只在没有标准 tool call 时启用，避免同一段输出被双重解析。

**不变量与失败模式：** 正则解析只能覆盖简单 XML 形态；嵌套或转义复杂的参数可能无法正确还原。

**Comment：** 这类兼容逻辑应保持保守，否则容易把普通文本误识别成工具调用。

### 5.3 run_agent 的 sandbox 执行骨架

**问题与约束：** 不同 coding-agent CLI 的安装和配置不同，但运行时都需要在 sandbox 内以 agent 用户执行，并把轨迹写到统一位置。

**设计选择：** `run_agent` 创建 `.harness` 目录并 chown 给 agent 用户，再调用 `exec_and_wait` 启动命令，输出写入 `trajectory.jsonl`。

**Explain：** 这是 harness-agnostic 的执行骨架。具体 Claude/Codex 类 harness 只需要提供 start command、env 和配置文件。

**Code：**

```python
## 来源：slime/agent/harness/common.py L107-L121
async def run_agent(sb: Sandbox, *, workdir: str, start_cmd: str, env: dict[str, str], time_budget_sec: int) -> int:
    meta_dir = f"{workdir}/.harness"
    await sb.exec(f"mkdir -p {meta_dir} && chown agent:agent {meta_dir}", user="root", check=True, timeout=30)
    exit_code, _ = await exec_and_wait(
        sb,
        cmd=start_cmd,
        user="agent",
        env=env,
        workdir=workdir,
        out_file=f"{meta_dir}/trajectory.jsonl",
        time_budget_sec=time_budget_sec,
        tag="run",
        want_output=False,
    )
```

**代码逻辑：** root 先准备 metadata 目录；agent 用户执行 CLI；`exec_and_wait` 负责等待和超时；函数返回 exit code。

**为什么这样写：** 训练数据和调试轨迹需要稳定落点；以 agent 用户运行能隔离权限，避免 CLI 意外修改 sandbox 管理文件。

**不变量与失败模式：** `start_cmd` 必须在 sandbox 内可执行；超时或非零 exit code 不在这里解释，调用方根据任务语义决定 reward。

**Comment：** 这个函数不懂具体任务，说明 Slime 把 agent 生命周期和任务 scoring 分开。

### 5.4 npm CLI 安装重试

**问题与约束：** sandbox 内安装 CLI 可能遇到临时 npm/disk 失败；直接重建 sandbox 成本高，但无限重试也会隐藏真实错误。

**设计选择：** `install_npm_cli` 先安装 Node 22，再把 npm package 写到 `/tmp`，用固定次数重试全局安装和 self-check。

**Explain：** 这是工程韧性设计：对已知短暂失败给小重试预算，对持续失败保留最后日志并抛错。

**Code：**

```python
## 来源：slime/agent/harness/common.py L141-L150
for attempt in range(NPM_INSTALL_RETRIES):
    exit_code, last_log = await exec_and_wait(
        sb, cmd=install_cmd, user="root", time_budget_sec=300, tag="harness-npm-install"
    )
    if exit_code == 0:
        return
    if attempt + 1 < NPM_INSTALL_RETRIES:
        await asyncio.sleep(NPM_INSTALL_BACKOFF_SEC * (attempt + 1))
raise RuntimeError(...)
```

**代码逻辑：** 安装最多尝试 `NPM_INSTALL_RETRIES` 次；失败但还有预算时按递增 backoff sleep；最终失败抛 RuntimeError。

**为什么这样写：** CLI 安装是 agent rollout 的基础设施问题，不应把偶发安装失败直接当成模型失败；但超过预算后必须失败，避免污染训练样本。

**不变量与失败模式：** `check_cmd` 必须能验证 CLI 可用；如果 package 本身损坏，重试不会修复，只会延迟明确失败。

**Comment：** 自定义 harness 应复用这类基础设施，而不是在每个任务里重复写安装流程。

---

## 6. 串起来看

Customization 的使用顺序可以这样判断：

1. 只改单样本生成、工具调用或 RAG：优先 `--custom-generate-function-path`。
2. 需要 verifier、外部评测或 group reward：用 `--custom-rm-path`，并确认 single/batch 模式。
3. 要改样本进入训练前的结构：先看 `--custom-convert-samples-to-train-data-path`，再看 `--rollout-data-postprocess-path`。
4. 要改整个 rollout 编排：才使用 `--rollout-function-path`。
5. 要改 Megatron 内部训练行为：使用 Megatron hooks，并保证所有 rank 一致。

源码的共同思路是：扩展策略可以替换，但数据形状、rollout_id、loss mask、reward list 长度、DP schedule 和日志语义必须保持稳定。只要这些契约不破，Slime 就能把自定义 agent/workflow 接回默认训练闭环。
