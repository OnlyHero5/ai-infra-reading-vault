---
type: batch-doc
module: 13-RM-FilterHub
batch: "13"
doc_type: walkthrough
title: "RM-FilterHub · 源码走读"
tags:
  - slime/batch/13
  - slime/module/rm-filter-hub
  - slime/doc/walkthrough
updated: 2026-07-02
---

# RM-FilterHub · 源码走读

> 阅读顺序：Rollout 调用链 → `async_rm` 分发 → 各 scorer → `generate_rollout_async` 过滤环 → metrics。

---

## 1. Rollout 侧何时调用 RM

### 1.1 单 sample 路径：`generate_and_rm`

**Explain：** 默认 `group_rm=False` 时，每个 sample 生成完成后立即打分。Fan-out generate（返回 `list[Sample]`）会对子 sample 批量 `batched_async_rm`。

**Code：**

```python
## 来源：slime/slime/rollout/sglang_rollout.py L263-L286
    if args.group_rm:
        return sample

    if isinstance(sample, list):
        samples = sample
        if any(sample.status == Sample.Status.ABORTED for sample in samples):
            return samples

        samples_need_reward = [sample for sample in samples if sample.reward is None]
        with trace_span(samples_need_reward, "reward_model"):
            rewards = await batched_async_rm(args, samples_need_reward)
        for sample, reward in zip(samples_need_reward, rewards, strict=False):
            sample.reward = reward
        return samples
    else:
        if sample.status == Sample.Status.ABORTED:
            return sample
        if sample.reward is None:
            with trace_span(sample, "reward_model"):
                sample.reward = await async_rm(args, sample)

    return sample
```

**Comment：**

- `group_rm=True` 时此处 **跳过** RM，留到 `generate_and_rm_group` 末尾
- 自定义 generate 若已填 `sample.reward`，不会重复调用 RM
- `trace_span(..., "reward_model")` 用于 rollout tracing（见 advanced observability 文档）

### 1.2 整组路径：`generate_and_rm_group` + `group_rm`

**Explain：** 需要 **组内相对比较** 的 RM（如排序、pass@k 联合打分）可实现 batch custom RM，并配合 `--group-rm`。

**Code：**

```python
## 来源：slime/slime/rollout/sglang_rollout.py L326-L331
    if not state.aborted and args.group_rm:
        with trace_span(group, "group_reward_model"):
            rewards = await batched_async_rm(args, group)
        for sample, reward in zip(group, rewards, strict=False):
            sample.reward = reward
```

**Comment：**

- `batched_async_rm` 在 `custom_rm_path` 设置时 **只** 调用 batch 插件，不会 fallback 到 per-sample `async_rm`
- eval rollout **不支持** `group_rm`（`assert not args.group_rm`）

---

## 2. `batched_async_rm` 与 `remote_rm`

### 2.1 并发 gather 模式

**Explain：** 无 global custom RM 时，对每个 sample 创建 `async_rm` task 并 `asyncio.gather`。

**Code：**

```python
## 来源：slime/slime/rollout/rm_hub/__init__.py L99-L110
async def batched_async_rm(
    args,
    samples: list[Sample],
    **kwargs,
) -> list[int | float]:
    if args.custom_rm_path is not None:
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, samples, **kwargs)
    tasks = [async_rm(args, sample, **kwargs) for sample in samples]
    rewards = await asyncio.gather(*tasks)
    return rewards
```

**Comment：**

- 内置 rm_type 路径下，N 个 sample = N 次独立打分（远程 RM 会 N 次 HTTP）
- 插件 batch 模式可合并为单次 RPC，降低 `--rm-type remote_rm` 开销

### 2.2 远程 RM：共享 Session + 退避重试

**Explain：** `remote_rm` 向 `--rm-url` POST JSON；连接池 limit=64，失败指数退避最多 10 次。

**Code：**

```python
## 来源：slime/slime/rollout/rm_hub/__init__.py L34-L52
async def remote_rm(args, sample: Sample, max_retries: int = 10):
    payload = {
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
    }
    session = _get_shared_session()
    for attempt in range(max_retries):
        try:
            async with session.post(args.rm_url, json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()
        except Exception as e:
            if attempt + 1 >= max_retries:
                logger.warning(f"remote_rm failed after {attempt + 1} attempts: {e}")
                raise
            backoff = min(2**attempt, 30) + random.random()
            await asyncio.sleep(backoff)
```

**Comment：**

- 返回值常为 dict；训练配置 `--reward-key` 提取标量
- OPD 等同理复用 HTTP RM 模式（见 [[21-Loss-Advantages-00-MOC]]）

---

## 3. `math_utils` 走读

### 3.1 提取 `\boxed{}`

**Explain：** 从右向左找最后一个 `\boxed` 或 `\fbox`，用大括号计数器匹配闭合位置。

**Code：**

```python
## 来源：slime/slime/rollout/rm_hub/math_utils.py L384-L426
def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    # ... brace counting ...
    return retval

def extract_boxed_answer(solution: str) -> str:
    solution = last_boxed_only_string(solution)
    solution = remove_boxed(solution)
    return solution
```

**Comment：**

- `async_rm` 的 `boxed_*` 前缀在 **整段 response** 上 extract，适用于 CoT 末尾 boxed
- dapo 版 `last_boxed_only_string` **无** `\fbox` fallback（测试锁定）

### 3.2 sympy 等价判定

**Explain：** 规范化后若字符串不等，尝试 `(gt)-(pred)` sympy 化简为 0；有 BAD_SUBSTRINGS 黑名单防 hang。

**Code：**

```python
## 来源：slime/slime/rollout/rm_hub/math_utils.py L351-L362
def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except Exception:
        pass
    return are_equal
```

**Comment：**

- `_sympy_parse` 使用白名单 global_dict，禁 builtins，降低代码注入风险
- 整数 GT 要求严格字符串匹配，不走 sympy（防 `2` vs `2.0` 误放）

---

## 4. `math_dapo_utils` 走读

### 4.1 Minerva 答案提取

**Explain：** 默认路径用 `(?i)Answer\s*:\s*([^\n]+)` 取 **最后一个** match，再 `normalize_final_answer`。

**Code：**

```python
## 来源：slime/slime/rollout/rm_hub/math_dapo_utils.py L199-L212
    match = re.findall(answer_pattern, solution_str)
    extracted_answer = match[-1] if match else "[INVALID]"
    pred = normalize_final_answer(extracted_answer)
    # ...
    gt = str(int(float(gt)))  # in dapo, all answers are integers
    return (pred == gt), pred
```

**Comment：**

- 无 `Answer:` 标记 → `[INVALID]` → 判错
- GT 强制 `int(float(gt))` 字符串化，与 pred 规范化后比较

### 4.2 strict box 模式

**Explain：** 只查看 pred **最后 100 字符**内的 boxed，精确字符串匹配 GT（无 sympy）。

**Code：**

```python
## 来源：slime/slime/rollout/rm_hub/math_dapo_utils.py L226-L237
    pred = pred[-100:]

    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    return 1 if (extracted_pred == gt) else -1, extracted_pred
```

**Comment：**

- `compute_score` 外层还有 **300 字符** tail 截断；二者叠加意味着长 CoT 中过早出现的 boxed 可能被忽略（测试 `test_compute_score_only_uses_last_300_chars` 验证）

### 4.3 `remove_boxed` 严格断言

**Code：**

```python
## 来源：slime/slime/rollout/rm_hub/math_dapo_utils.py L59-L62
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]
```

**Comment：** 与 `math_utils.remove_boxed`（try/except → None）不同；统一实现会导致 dapo strict 路径行为漂移。

---

## 5. `deepscaler` 完整判题

**Code：**

```python
## 来源：slime/slime/rollout/rm_hub/deepscaler.py L36-L42
    for ground_truth in processed_ground_truths:
        is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
        if is_correct:
            return 1

    return 0
```

**Comment：** 与 `math` rm_type 共用 sympy 内核，但 **response 预处理** 要求 DeepScaler 模板分隔符。

---

## 6. Filter Hub 走读

### 6.1 `generate_rollout_async` 过滤环

**Explain：** 维护 `target_data_size = rollout_batch_size` 的有效 group；每完成一组生成，先 `call_dynamic_filter`，不 keep 则 `remaining_batch_size -= 1` 触发继续过采样。

**Code：**

```python
## 来源：slime/slime/rollout/sglang_rollout.py L394-L433
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )

    metric_gatherer = MetricGatherer()
    target_data_size = args.rollout_batch_size
    # ...
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                state.remaining_batch_size -= 1
                continue
```

**Comment：**

- `over_sampling_batch_size` 控制每次从 `data_source` 拉多少 prompt **并行** 提交
- 被丢弃的 group 进入 `all_data` 但不进入 `data`；注释说明尚未全部回写 buffer（已知限制）
- 返回 `RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect())`

### 6.2 `MetricGatherer`

**Code：**

```python
## 来源：slime/slime/rollout/filter_hub/base_types.py L24-L37
class MetricGatherer:
    def on_dynamic_filter_drop(self, reason: str | None):
        if not reason:
            return
        self._dynamic_filter_drop_reason_count[reason] += 1

    def collect(self):
        return {
            f"rollout/dynamic_filter/drop_{reason}": count
            for reason, count in self._dynamic_filter_drop_reason_count.items()
        }
```

**Comment：** reason 为 `None` 时不计数；`check_reward_nonzero_std` 仅在 drop 时设置 reason。

---

## 7. Eval 注入 `custom_rm_path`

**Explain：** eval 数据集配置可覆盖 RM，写入 sample 字段。

**Code：**

```python
## 来源：slime/slime/rollout/sglang_rollout.py L568（节选上下文）
            sample.custom_rm_path = dataset_cfg.custom_rm_path
```

**Comment：** 配合 `EvalDatasetConfig.rm_type` 写入 `metadata["rm_type"]`；per-sample path 优先级最高。

---

## 8. `Sample.get_reward_value`

**Explain：** Filter 与 loss 侧统一通过该方法取标量 reward。

**Code：**

```python
## 来源：slime/slime/utils/types.py L246-L247
    def get_reward_value(self, args) -> float:
        return self.reward if not args.reward_key else self.reward[args.reward_key]
```

**Comment：**

- `rm_type=dapo` 且未设 `--reward-key` 时，filter 会对 dict 做 `torch.tensor` 可能报错——生产配置应显式设 key
- eval 可用 `--eval-reward-key` 覆盖（见 arguments）

---

## 9. CLI 参数锚点

**Code：**

```python
## 来源：slime/slime/utils/arguments.py L1316-L1356（节选）
        def add_reward_model_arguments(parser):
            parser.add_argument("--rm-type", type=str, default=None, ...)
            parser.add_argument("--reward-key", type=str, default=None, ...)
            parser.add_argument("--group-rm", action="store_true", default=False, ...)
            parser.add_argument("--rm-url", type=str, default=None, ...)
            parser.add_argument("--custom-rm-path", type=str, default=None, ...)
```

**Comment：** `--dynamic-sampling-filter-path` 在 rollout 参数组（约 L442），示例值见 quick_start 文档。

---

## 10. 走读小结

| 步骤 | 函数 | 文件 |
|------|------|------|
| 1 | `generate_and_rm` | `sglang_rollout.py` |
| 2 | `async_rm` / `batched_async_rm` | `rm_hub/__init__.py` |
| 3 | 具体 scorer | `math_utils` / `math_dapo_utils` / `deepscaler` / … |
| 4 | `call_dynamic_filter` | `filter_hub/base_types.py` |
| 5 | `check_reward_nonzero_std` | `dynamic_sampling_filters.py` |
| 6 | `metric_gatherer.collect` | 并入 rollout metrics |
