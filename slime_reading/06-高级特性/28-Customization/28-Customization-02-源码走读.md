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
updated: 2026-07-03
---

# Customization · 源码走读

## 走读顺序（runtime 优先）

1. `slime/utils/misc.py` — `load_function` 统一加载
2. `slime/ray/rollout.py` — RolloutManager 挂载 rollout / data_source / convert hooks
3. `slime/rollout/sglang_rollout.py` — `custom_generate` 分支（agent 主路径）
4. `slime/rollout/rm_hub/__init__.py` — `custom_rm` / `batched_async_rm`
5. `slime/backends/megatron_utils/actor.py` — train 侧 postprocess / megatron init hooks
6. `docs/en/get_started/customization.md` — 17 类接口签名权威表
7. `slime/agent/parsing.py` · `harness/` — agent 多轮与 sandbox

> **阅读建议：** 先完成 §1–§5（runtime），再按需查 §6–§17（接口文档与 agent 实现）。§6–§7 与 [[28-Customization-01-核心概念]] 接口表互补。

---

## 1. load_function：运行时 import

**Explain：** 所有 `--*-path` 在 **Actor / RolloutManager 初始化** 时调用 `load_function`，格式 `package.module.function`。path 写错会在启动期 `ImportError` / `AttributeError`，不会 silent 回退默认（见 [[28-Customization-04-关键问题]] Q1）。

**Code：**

```python
## 来源：slime/utils/misc.py L37-L45
def load_function(path):
    module_path, _, attr = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)
```

**Comment：**

- 与 [[10-Sample-Contracts-01-核心概念]] §6 同一实现；eval 配置里的 per-dataset path 也走此函数
- 自定义函数若为 async，调用方必须 `await`（见 [[12-SGLang-Rollout-04-关键问题]]）

---

## 2. RolloutManager：Rollout 侧挂载点

**Explain：** `RolloutManager.__init__` 是 Rollout 侧 **最集中的 load_function 调用栈**：data_source、整段 rollout、eval rollout、reward post-process、sample→train_data 转换均可在此替换。

**Code：**

```python
## 来源：slime/ray/rollout.py L78-L82
        data_source_cls = load_function(self.args.data_source_path)
        self.data_source = data_source_cls(args)

        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
```

**Comment：**

- 完整 init 见 [[08-RolloutManager-02-源码走读]] §2；`generate(rollout_id)` 每步调用 `self.generate_rollout`
- `--rollout-function-path` 替换 **外层循环**；多数 agent 场景只需 `--custom-generate-function-path`（§3）

---

## 3. sglang_rollout：custom_generate 分支

**Explain：** 默认 rollout 仍走 `sglang_rollout.generate_rollout`，但 **单 sample 生成** 可在 semaphore 内替换：`sample.generate_function_path` 优先于全局 `args.custom_generate_function_path`。这是 search-r1、coding agent 等样板的主接入点。

**Code：**

```python
## 来源：slime/rollout/sglang_rollout.py L226-L234
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

**Comment：**

- 同函数后续若非 `group_rm`，会 `await async_rm` 打分（§4）
- 多 agent 需替换整段 rollout 时用 `--rollout-function-path`（见 [[29-Plugins-Examples-03-数据流与交互]]）

---

## 4. RM Hub：custom_rm 与 batch 模式

**Explain：** `--custom-rm-path` 在 `batched_async_rm` 入口短路：直接 `await rm_function(args, samples)`。未设置时 fallback 到 per-sample `async_rm`（内置 rm_type 或 remote_rm）。

**Code：**

```python
## 来源：slime/rollout/rm_hub/__init__.py L97-L99
    if args.custom_rm_path is not None:
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, samples, **kwargs)
```

**Comment：**

- per-sample 路径：`async_rm` 内同样 `load_function(sample.custom_rm_path)` 或 `args.custom_rm_path`（[[13-RM-FilterHub-01-核心概念]]）
- Agent 任务推荐 **generate + RM 双 custom**，而非仅换 rollout_function（[[28-Customization-01-核心概念]] §5）

---

## 5. Megatron Actor：Train 侧 hooks

**Explain：** Train 侧 hook 在 `MegatronTrainRayActor` init / train step 加载：`rollout_data_postprocess` 在 logprob 对齐后、`custom_megatron_init` 在 Megatron 栈构建前。

**Code：**

```python
## 来源：slime/backends/megatron_utils/actor.py L232-L236
self.rollout_data_postprocess = None
if self.args.rollout_data_postprocess_path is not None:
    from slime.utils.misc import load_function

    self.rollout_data_postprocess = load_function(self.args.rollout_data_postprocess_path)
```

**Code（Megatron init hook）：**

```python
## 来源：slime/backends/megatron_utils/actor.py L401-L403
    from slime.utils.misc import load_function

    custom_init = load_function(args.custom_megatron_init_path)
```

**Comment：**

- 调用时机见 [[19-Train-Step-02-源码走读]]；loss / advantage / pg_reducer 等同理（[[22-Loss-Policy-02-源码走读]]、[[21-Loss-Advantages-02-源码走读]]）
- 与 [[17-Megatron-Actor-Init-02-源码走读]] §7 init 链衔接

---

## 6. Rollout Function 签名（接口文档）

**Explain：** 仅当 per-sample custom_generate 不够用时，才替换整段 rollout。签名与默认 `sglang_rollout.generate_rollout` 一致。

**Code：**

```python
## 来源：docs/en/get_started/customization.md L58-L59
def generate_rollout(args, rollout_id, data_source, evaluation=False) -> RolloutFnTrainOutput | RolloutFnEvalOutput
```

**Comment：** 完全替换 [[08-RolloutManager-02-源码走读]] 调度的外层循环；multi_agent example 使用此路径。

---

## 7. Custom RM 双模式（接口文档）

**Code：**

```python
## 来源：docs/en/get_started/customization.md L131-L136
async def custom_rm(args, sample: Sample) -> float

async def batched_custom_rm(args, samples: list[Sample]) -> list[float]
```

**Comment：** `--group-rm` 启用 batch 模式；内置 `--rm-type` math/dapo/remote_rm 等可与 custom 并存策略见 [[13-RM-FilterHub-01-核心概念]]。

---

## 8. Dynamic Filter 返回类型

**Code：**

```python
## 来源：docs/en/get_started/customization.md L168-L172
@dataclass
class DynamicFilterOutput:
    keep: bool
    reason: str | None
```

**Comment：** 典型实现 `check_reward_nonzero_std` 过滤 reward 方差为 0 的 group。

---

## 9. rollout-sample-filter 副作用契约

**Code：**

```python
## 来源：docs/en/get_started/customization.md L209-L211
def filter_function(args, samples: list[Sample]) -> None
# Note: This function should directly modify the `remove_sample` attribute of each `Sample` object.
```

**Comment：** 无返回值；in-place 修改 Sample。

---

## 10. pg_loss Reducer 签名

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

**Comment：** 仅替换 pg_loss 聚合；clipfrac、entropy 仍用默认 sum_of_sample_mean。

---

## 11. DataSource 必需方法

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

**Comment：** 与 [[11-DataSource-00-MOC]] RolloutDataSourceWithBuffer 对照。

---

## 12. Megatron before-train-step hook

**Code：**

```python
## 来源：docs/en/get_started/customization.md L440-L443
def custom_hook(args, rollout_id, step_id, model, optimizer, opt_param_scheduler) -> None
```

**Comment：** 在 [[19-Train-Step-02-源码走读]] 主 step 前插入；可用于 curriculum、冻结层等。

---

## 13. parse_tool_uses：SGLang FunctionCallParser

**Code：**

```python
## 来源：slime/agent/parsing.py L67-L85
    if tool_parser_name and tools_schema:
        from sglang.srt.function_call.function_call_parser import FunctionCallParser
        sg_tools = [Tool(type="function", function=Function(**d["function"])) for d in tools_schema]
        parser = FunctionCallParser(tools=sg_tools, tool_call_parser=tool_parser_name)
        if parser.has_tool_call(body_text):
            body_text, calls = parser.parse_non_stream(body_text)
        for c in calls:
            args = json.loads(c.parameters or "{}")
            tool_uses.append({"name": c.name or "tool", "input": args})
```

**Comment：** JSON 解析失败设 `ill_formed=True`。

---

## 14. XML tool fallback

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
            args = {p.group(1): p.group(2).strip() for p in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", inner, flags=re.DOTALL)}
            tool_uses.append({"name": name, "input": args})
```

**Comment：** 仅当 SGLang parser 未命中且 schema 非空时启用。

---

## 15. run_agent：sandbox 内 detached 执行

**Code：**

```python
## 来源：slime/agent/harness/common.py L107-L121
async def run_agent(sb, *, workdir, start_cmd, env, time_budget_sec) -> int:
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
    return exit_code
```

**Comment：** trajectory.jsonl 供 debug；训练数据来自 adapter TrajectoryManager。

---

## 16. install_npm_cli 重试

**Code：**

```python
## 来源：slime/agent/harness/common.py L141-L150
    for attempt in range(NPM_INSTALL_RETRIES):
        exit_code, last_log = await exec_and_wait(sb, cmd=install_cmd, user="root", time_budget_sec=300, tag="harness-npm-install")
        if exit_code == 0:
            return
        if attempt + 1 < NPM_INSTALL_RETRIES:
            await asyncio.sleep(NPM_INSTALL_BACKOFF_SEC * (attempt + 1))
    raise RuntimeError(f"npm install failed after {NPM_INSTALL_RETRIES} attempts ...")
```

---

## 17. plugin_contracts 测试入口

**Code：**

```python
## 来源：docs/en/get_started/customization.md L476-L481
python -m pytest \
  tests/plugin_contracts/test_plugin_rollout_contracts.py \
  tests/plugin_contracts/test_plugin_generate_contracts.py \
  tests/plugin_contracts/test_plugin_path_loading_contracts.py \
  tests/plugin_contracts/test_plugin_runtime_hook_contracts.py
```

**Comment：** CPU-only；PR 可加 `run-ci-cpu-unittest` label。
