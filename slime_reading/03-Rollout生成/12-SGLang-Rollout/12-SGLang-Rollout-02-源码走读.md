---
type: batch-doc
module: 12-SGLang-Rollout
batch: "12"
doc_type: walkthrough
title: "SGLang Rollout · 源码走读"
tags:
  - slime/batch/12
  - slime/module/sglang-rollout
  - slime/doc/walkthrough
updated: 2026-07-02
---

# SGLang Rollout · 源码走读

> 走读顺序：`generate_rollout` → `generate_rollout_async` → `submit_generate_tasks` → `generate_and_rm_group` → `generate_and_rm` → `generate` → `abort`

---

## 1. 同步入口 `generate_rollout`

### 1.1 训练 vs 评估分派

**Explain：** 入口断言 `rollout_global_dataset=True`。评估不调 DataSource oversampling，直接 `eval_rollout` 遍历 `eval_datasets` 配置。

**Code：**

```python
# 来源：slime/slime/rollout/sglang_rollout.py L618-L640
def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    assert args.rollout_global_dataset
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source.get_samples))
    if aborted_samples:
        data_source.add_samples(aborted_samples)
    return output
```

**Comment：** RolloutManager 通过 `call_rollout_fn` 包装 legacy 返回值（见 `base_types.py`）。

---

## 2. 训练主循环 `generate_rollout_async`

### 2.1 初始化 filter 与 metric

**Explain：** 动态加载 `--dynamic-sampling-filter-path`；`MetricGatherer` 记录 filter drop 原因。

**Code：**

```python
# 来源：slime/slime/rollout/sglang_rollout.py L390-L407
    state = GenerateState(args)

    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )

    metric_gatherer = MetricGatherer()

    target_data_size = args.rollout_batch_size

    data = []
    all_data = []
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
```

**Comment：** `all_data` 保留全部被 filter 前的组，供 `--rollout-all-samples-process-path` 后处理。

### 2.2 提交与等待任务

**Explain：** `submit_generate_tasks` 把每组 prompt 包装为 `asyncio.create_task(generate_and_rm_group(...))`，加入 `state.pendings`。主循环 `asyncio.wait(FIRST_COMPLETED)` 增量消费。

**Code：**

```python
# 来源：slime/slime/rollout/sglang_rollout.py L137-L150
    def submit_generate_tasks(self, samples: list[list[Sample]]) -> None:
        for group in samples:
            self.pendings.add(
                asyncio.create_task(
                    generate_and_rm_group(
                        self.args,
                        group,
                        sampling_params=self.sampling_params.copy(),
                        evaluation=False,
                    )
                )
            )
        self.remaining_batch_size += len(samples)
```

**Code：**

```python
# 来源：slime/slime/rollout/sglang_rollout.py L408-L439
    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            samples = data_source(args.over_sampling_batch_size)
            state.submit_generate_tasks(samples)

        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group: list[Sample] = task.result()
            assert len(group) == args.n_samples_per_prompt
            all_data.append(group)

            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                state.remaining_batch_size -= 1
                continue

            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)
```

**Comment：** fan-out custom generate 时 `len(group)` 断言可能需插件侧保证与 `n_samples_per_prompt` 一致，或使用不同 rollout function path。

### 2.3 收尾：abort、排序、filter

**Explain：** 凑满 batch 后对剩余 pending 调用 `abort`；按 `sample.index` 排序；可选 sample filter 与 all-samples process hook。

**Code：**

```python
# 来源：slime/slime/rollout/sglang_rollout.py L447-L467
    aborted_samples = await abort(args, rollout_id)

    assert len(data) == args.rollout_batch_size
    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)

    state.reset()
    if args.rollout_sample_filter_path is not None:
        filter_func = load_function(args.rollout_sample_filter_path)
        filter_func(args, data)

    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), aborted_samples
```

---

## 3. Group 级生成 `generate_and_rm_group`

**Explain：** abort 后短路返回原 group；否则并发 spawn 组内各 sample 的 `generate_and_rm`，再处理 group RM。

**Code：**

```python
# 来源：slime/slime/rollout/sglang_rollout.py L294-L333
async def generate_and_rm_group(
    args: Namespace, group: list[Sample], sampling_params: dict[str, Any], evaluation: bool = False
) -> list[Sample] | list[list[Sample]]:
    state = GenerateState(args)

    if state.aborted:
        return group

    for sample in group:
        if sample.session_id is None:
            sample.session_id = str(uuid.uuid4())

    tasks = []
    for idx, sample in enumerate(group):
        current_sampling_params = sampling_params.copy()
        if getattr(args, "sglang_enable_deterministic_inference", False):
            current_sampling_params["sampling_seed"] = state.group_sampling_seeds[idx]
        tasks.append(
            asyncio.create_task(generate_and_rm(args, sample, current_sampling_params, evaluation=evaluation))
        )

    group = await asyncio.gather(*tasks)

    if not state.aborted and args.group_rm:
        rewards = await batched_async_rm(args, group)
        for sample, reward in zip(group, rewards, strict=False):
            sample.reward = reward

    return group
```

---

## 4. 单 sample 生成 + RM `generate_and_rm`

### 4.1 短路：已完成 / partial mask

**Explain：** 若 sample 已 COMPLETED/TRUNCATED 且 reward 已有，直接返回。partial rollout 可 mask 历史 off-policy token。

**Code：**

```python
# 来源：slime/slime/rollout/sglang_rollout.py L230-L239
    if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
        sample.loss_mask = [0] * sample.response_length

    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample
```

### 4.2 Semaphore 内 generate + per-sample RM

**Explain：** `async with state.semaphore` 限制并发；abort 时标记 ABORTED。非 group_rm 时在 generate 后立即 `async_rm`。

**Code：**

```python
# 来源：slime/slime/rollout/sglang_rollout.py L241-L286
    state = GenerateState(args)

    async with state.semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

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

    if args.group_rm:
        return sample

    if sample.status == Sample.Status.ABORTED:
        return sample
    if sample.reward is None:
        sample.reward = await async_rm(args, sample)

    return sample
```

---

## 5. 默认 HTTP 生成 `generate`

### 5.1 Prompt 编码与空响应

**Explain：** `_prepare_prompt_ids` 处理 multimodal processor 或 tokenizer；`max_new_tokens==0` 直接 TRUNCATED。

**Code：**

```python
# 来源：slime/slime/rollout/sglang_rollout.py L43-L62
def _prepare_prompt_ids(sample: Sample, tokenizer, processor: Any) -> list[int]:
    raw_multimodal_inputs = sample.multimodal_inputs or {}
    has_multimodal_inputs = any(value is not None for value in raw_multimodal_inputs.values())
    reuse_existing_input_ids = bool(sample.tokens) and (
        sample.multimodal_train_inputs is not None or not has_multimodal_inputs
    )

    if processor and has_multimodal_inputs and not reuse_existing_input_ids:
        processor_output = processor(text=sample.prompt, **build_processor_kwargs(raw_multimodal_inputs))
        prompt_ids = processor_output["input_ids"][0]
        return prompt_ids

    if reuse_existing_input_ids:
        return sample.tokens

    return tokenizer.encode(sample.prompt, add_special_tokens=False)
```

**Code：**

```python
# 来源：slime/slime/rollout/sglang_rollout.py L165-L172
    prompt_ids = _prepare_prompt_ids(sample, state.tokenizer, state.processor)

    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample
```

### 5.2 POST 与响应解析

**Code：**

```python
# 来源：slime/slime/rollout/sglang_rollout.py L158-L220
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    # ... payload 构建 ...

    with trace_span(sample, "sglang_generate", attrs={"max_new_tokens": sampling_params["max_new_tokens"]}) as span:
        output = await post(url, payload, headers=headers)
        span.update(build_sglang_meta_trace_attrs(output["meta_info"]))

    sample.append_response_tokens(
        args,
        tokens=new_response_tokens,
        log_probs=new_response_log_probs,
        trainable=True,
        meta_info=output["meta_info"],
        text=output["text"],
    )

    return sample
```

---

## 6. Abort 与 partial 回收 `abort`

**Explain：** 设置 `state.aborted=True`，向所有 SGLang worker 发 abort 直到 idle，再 drain pending tasks。`partial_rollout` 时把有 response 的 sample 写入 metadata `start_rollout_id` 并返回给 DataSource。

**Code：**

```python
# 来源：slime/slime/rollout/sglang_rollout.py L336-L372
async def abort(args: Namespace, rollout_id: int) -> list[list[Sample]]:
    aborted_samples = []

    state = GenerateState(args)
    assert not state.aborted
    state.aborted = True

    if parse(sglang_router.__version__) <= parse("0.2.1"):
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        urls = response["urls"]
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        urls = [worker["url"] for worker in response["workers"]]

    await abort_servers_until_idle(urls)

    while state.pendings:
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        if not args.partial_rollout:
            continue
        for task in done:
            group = task.result()
            for sample in group:
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples.append(group)

    return aborted_samples
```

---

## 7. 评估路径 `eval_rollout_single_dataset`（节选）

**Explain：** 评估不走 oversampling；每个 eval prompt 复制 `n_samples_per_eval_prompt` 份，可设 per-dataset `custom_generate_function_path`。

**Code：**

```python
# 来源：slime/slime/rollout/sglang_rollout.py L561-L582
    for _i, prompt_sample in enumerate(dataset.samples):
        for j in range(dataset_cfg.n_samples_per_eval_prompt):
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
            sample.custom_rm_path = dataset_cfg.custom_rm_path
            sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
            tasks.append(
                asyncio.create_task(
                    generate_and_rm(
                        args,
                        sample,
                        sampling_params=sampling_params,
                        evaluation=True,
                    )
                )
            )
```

**Comment：** eval 不支持 `group_rm`（入口 assert）。

---

## 8. 辅助：`get_model_url`

**Code：**

```python
# 来源：slime/slime/rollout/sglang_rollout.py L65-L81
def get_model_url(args: Namespace, model_name: str, endpoint: str = "/generate") -> str:
    routers = getattr(args, "sglang_model_routers", None)
    if routers and model_name in routers:
        ip, port = routers[model_name]
        return f"http://{ip}:{port}{endpoint}"
    return f"http://{args.sglang_router_ip}:{args.sglang_router_port}{endpoint}"
```

**Comment：** custom generate 多模型场景（如 ref policy）应使用此 helper 而非硬编码 router 地址。
