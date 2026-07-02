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
updated: 2026-07-02
---

# Customization · 源码走读

## 走读顺序

1. `customization.md` — 接口权威文档
2. `misc.load_function` — 加载机制
3. `parsing.py` — tool/reasoning 解析
4. `harness/common.py` — BaseHarness 生命周期
5. `harness/claude_code.py` — 代表 harness 实现

---

## 1. Rollout Function 签名

**Code：**

```python
# 来源：docs/en/get_started/customization.md L58-L59
def generate_rollout(args, rollout_id, data_source, evaluation=False) -> RolloutFnTrainOutput | RolloutFnEvalOutput
```

**Comment：** 完全替换 [[08-RolloutManager-02-源码走读]] 调度的外层循环；multi_agent example 使用此路径。

---

## 2. Custom RM 双模式

**Code：**

```python
# 来源：docs/en/get_started/customization.md L131-L136
async def custom_rm(args, sample: Sample) -> float

async def batched_custom_rm(args, samples: list[Sample]) -> list[float]
```

**Comment：** `--group-rm` 启用 batch 模式；内置 `--rm-type` math/dapo/remote_rm 等可与 custom 并存策略见 [[13-RM-FilterHub-01-核心概念]]。

---

## 3. Dynamic Filter 返回类型

**Code：**

```python
# 来源：docs/en/get_started/customization.md L168-L172
@dataclass
class DynamicFilterOutput:
    keep: bool
    reason: str | None
```

**Comment：** 典型实现 `check_reward_nonzero_std` 过滤 reward 方差为 0 的 group。

---

## 4. rollout-sample-filter 副作用契约

**Code：**

```python
# 来源：docs/en/get_started/customization.md L209-L211
def filter_function(args, samples: list[Sample]) -> None
# Note: This function should directly modify the `remove_sample` attribute of each `Sample` object.
```

**Comment：** 无返回值；in-place 修改 Sample。

---

## 5. pg_loss Reducer 签名

**Code：**

```python
# 来源：docs/en/get_started/customization.md L288-L294
def get_pg_loss_reducer(
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    calculate_per_token_loss: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]
```

**Comment：** 仅替换 pg_loss 聚合；clipfrac、entropy 仍用默认 sum_of_sample_mean。

---

## 6. DataSource 必需方法

**Code：**

```python
# 来源：docs/en/get_started/customization.md L387-L401
class CustomDataSource(DataSource):
    def get_samples(self, num_samples: int) -> list[list[Sample]]: ...
    def add_samples(self, samples: list[list[Sample]]): ...
    def save(self, rollout_id): ...
    def load(self, rollout_id=None): ...
    def __len__(self): ...
```

**Comment：** 与 [[11-DataSource-00-MOC]] RolloutDataSourceWithBuffer 对照。

---

## 7. Megatron before-train-step hook

**Code：**

```python
# 来源：docs/en/get_started/customization.md L440-L443
def custom_hook(args, rollout_id, step_id, model, optimizer, opt_param_scheduler) -> None
```

**Comment：** 在 [[19-Train-Step-02-源码走读]] 主 step 前插入；可用于 curriculum、冻结层等。

---

## 8. parse_tool_uses：SGLang FunctionCallParser

**Code：**

```python
# 来源：slime/agent/parsing.py L67-L85
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

## 9. XML tool fallback

**Code：**

```python
# 来源：slime/agent/parsing.py L99-L110
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

## 10. run_agent：sandbox 内 detached 执行

**Code：**

```python
# 来源：slime/agent/harness/common.py L107-L121
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

## 11. install_npm_cli 重试

**Code：**

```python
# 来源：slime/agent/harness/common.py L141-L150
    for attempt in range(NPM_INSTALL_RETRIES):
        exit_code, last_log = await exec_and_wait(sb, cmd=install_cmd, user="root", time_budget_sec=300, tag="harness-npm-install")
        if exit_code == 0:
            return
        if attempt + 1 < NPM_INSTALL_RETRIES:
            await asyncio.sleep(NPM_INSTALL_BACKOFF_SEC * (attempt + 1))
    raise RuntimeError(f"npm install failed after {NPM_INSTALL_RETRIES} attempts ...")
```

---

## 12. plugin_contracts 测试入口

**Code：**

```python
# 来源：docs/en/get_started/customization.md L476-L481
python -m pytest \
  tests/plugin_contracts/test_plugin_rollout_contracts.py \
  tests/plugin_contracts/test_plugin_generate_contracts.py \
  tests/plugin_contracts/test_plugin_path_loading_contracts.py \
  tests/plugin_contracts/test_plugin_runtime_hook_contracts.py
```

**Comment：** CPU-only；PR 可加 `run-ci-cpu-unittest` label。
