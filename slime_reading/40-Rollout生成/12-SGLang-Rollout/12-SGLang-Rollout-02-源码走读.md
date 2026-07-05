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
updated: 2026-07-05
---

# SGLang Rollout · 源码走读

> 读法：先从 `generate_rollout` 看同步入口如何分派训练/评估，再沿 `generate_rollout_async` 的异步任务池进入 group 与单 sample 生成，最后看 HTTP 调用、abort 回收和评估任务构造。

---

## 1. 入口与训练任务池

### 1.1 `generate_rollout`：同步入口包住异步实现

来源：slime/rollout/sglang_rollout.py L618-L640

**问题与约束：** RolloutManager 需要一个同步 callable 作为 rollout function，但实际生成路径是 async；训练与评估还要走不同数据流，训练需要把 partial abort 样本还给 DataSource。

**设计选择：** 入口强制 `rollout_global_dataset`，评估时直接 `run(eval_rollout(...))` 并返回 output；训练时 `run(generate_rollout_async(...))`，若有 `aborted_samples` 则调用 `data_source.add_samples` 回填。

**Explain：** 这层是同步框架与异步生成系统之间的适配器。它不做生成细节，只决定本轮是 eval 还是 train，并把 async 返回值恢复成 RolloutManager 期望的返回对象。

**Code：**

```python
def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to get and store samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        RolloutFnTrainOutput | RolloutFnEvalOutput: the output of the rollout
    """
    assert args.rollout_global_dataset
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source.get_samples))
    if aborted_samples:
        data_source.add_samples(aborted_samples)
    return output
```

**代码逻辑：** 函数先校验全局数据集模式；`evaluation=True` 时跳到 eval coroutine；否则把 `data_source.get_samples` 传给训练 coroutine。训练 coroutine 返回本轮有效样本与 abort 收集样本，入口负责把 abort 样本回灌到数据源。

**为什么这样写：** 上层训练循环通常按同步函数调用 rollout，而 SGLang HTTP 请求、RM 调用和 abort 都是异步操作。把 `run(...)` 集中在入口处，可以让下游逻辑保持 async，同时不改变 RolloutManager 的接口。

**不变量与失败模式：** 该实现只支持 `rollout_global_dataset`；训练路径必须返回 `(output, aborted_samples)`；评估路径不能把 eval 样本写回 DataSource。若 partial abort 样本没有回填，后续 rollout 会丢失已生成但未入 batch 的进度。

**Comment：** 入口的核心不是生成，而是把训练系统的同步协议和 rollout 内部的异步协议接起来。

### 1.2 `generate_rollout_async`：初始化 filter、metric 与目标 batch

来源：slime/rollout/sglang_rollout.py L390-L407

**问题与约束：** 训练 rollout 需要凑满 `rollout_batch_size` 个有效 prompt group；动态采样 filter 可能丢弃已生成 group，因此需要独立记录有效数据、全部数据和 drop 指标。

**设计选择：** 创建共享 `GenerateState`，按需动态加载 filter，初始化 `MetricGatherer`，将目标有效组数设为 `args.rollout_batch_size`，同时维护 `data` 与 `all_data` 两个列表。

**Explain：** 这一段建立训练生成循环的状态容器。`data` 是最终送给训练的有效样本，`all_data` 保留 filter 前的样本，用于后处理或统计，`MetricGatherer` 则把 filter 的丢弃原因转成 metrics。

**Code：**

```python
    assert args.rollout_global_dataset

    state = GenerateState(args)

    # instantiate data filters
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )

    metric_gatherer = MetricGatherer()

    # target_data_size is the total number of valid samples to get
    target_data_size = args.rollout_batch_size

    data = []
    all_data = []
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
```

**代码逻辑：** 函数再次断言 global dataset，拿到单例式 `GenerateState`；filter path 存在时通过 `load_function` 变成 callable；之后建立有效数据、全量数据、日志开关和进度条。

**为什么这样写：** oversampling 与 filter 让“提交多少组”和“最终保留多少组”不相等。把目标设为有效 batch size，并把 filter 前后的样本分开存，才能在保持 batch 大小稳定的同时保留诊断信息。

**不变量与失败模式：** `target_data_size` 表示有效 group 数，不是 sample 数；进度条总量用 `n_samples_per_prompt` 放大；filter callable 必须接受后续 `call_dynamic_filter` 的协议。若只记录 `data`，被 filter 丢弃的样本就无法进入 all-samples 后处理。

**Comment：** 训练循环从这里开始体现 oversampling 语义：生成可以多，进入训练的有效 group 必须刚好够。

### 1.3 `submit_generate_tasks`：按 group 建异步任务

来源：slime/rollout/sglang_rollout.py L137-L150

**问题与约束：** 每个 prompt group 内有 `n_samples_per_prompt` 个样本，它们需要作为一个 group 接受动态 filter 或 group reward；但组间应并发提交，以填满 SGLang router 与 RM 的吞吐。

**设计选择：** 对输入的每个 group 创建一个 `generate_and_rm_group` task，放入 `state.pendings` 集合，并按 group 数增加 `remaining_batch_size`。

**Explain：** task 粒度是 group 而不是单 sample。这样主循环等待完成时拿到的自然就是一个可过滤、可排序、可入 batch 的 group。

**Code：**

```python
    def submit_generate_tasks(self, samples: list[list[Sample]]) -> None:
        for group in samples:
            self.pendings.add(
                asyncio.create_task(
                    # submit a group of samples as a single task.
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

**代码逻辑：** 函数遍历 `samples` 中的 group，为每个 group 复制一份 sampling params 并提交 `generate_and_rm_group`。所有 task 被加入 pending set，`remaining_batch_size` 记录已经在途的 group 数。

**为什么这样写：** group 作为任务边界可以让动态 filter 和 group RM 保持输入形状稳定；复制 sampling params 避免不同任务之间修改同一个 dict；pending set 支持后续 `asyncio.wait` 增量消费。

**不变量与失败模式：** `samples` 必须是 `list[list[Sample]]`；`remaining_batch_size` 按 group 计数，不按 sample 计数；task 结果应保持与 `n_samples_per_prompt` 一致。若按 sample 提交，group filter 与 group reward 就需要额外重组。

**Comment：** 这一层把数据源的 group 结构原样转成异步调度结构，是后续 filter 能按组工作的前提。

### 1.4 主循环：oversampling、FIRST_COMPLETED 与动态过滤

来源：slime/rollout/sglang_rollout.py L408-L439

**问题与约束：** 动态 filter 会丢弃 group，因此主循环需要持续 oversampling，直到有效 group 数达到目标；同时不能等所有任务完成后才处理，否则会降低吞吐并延迟补充任务。

**设计选择：** 当 `remaining_batch_size` 低于目标时，从 DataSource 取 `over_sampling_batch_size` 并提交；随后用 `asyncio.wait(..., FIRST_COMPLETED)` 增量消费已完成 group，filter 失败则减少 remaining，filter 通过且未满目标才加入 `data`。

**Explain：** 这是一个以有效 batch 为目标的异步水位控制。`remaining_batch_size` 表示“已提交但还可能贡献有效 group 的数量”，filter drop 会降低水位，从而触发下一轮补样。

**Code：**

```python
    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            # get samples from the buffer and submit the generation requests.
            samples = data_source(args.over_sampling_batch_size)
            state.submit_generate_tasks(samples)

        # wait for the generation to finish
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group: list[Sample] = task.result()

            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                logger.info(
                    f"First rollout sample: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            all_data.append(group)

            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                state.remaining_batch_size -= 1
                continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)
```

**代码逻辑：** 外层循环以 `len(data)` 为结束条件；内层循环维持 pending 水位。每次等待至少一个 task 完成，取出 group，首次打印样本，断言 group 大小，加入 `all_data`。filter drop 时记录原因并减少水位；通过时才追加到 `data` 并推进进度条。

**为什么这样写：** FIRST_COMPLETED 让生成、RM、filter、补样形成流水线，不必批量阻塞。`remaining_batch_size` 在 drop 时减少，避免被已丢弃任务占住容量，保证最终能凑满有效 batch。

**不变量与失败模式：** `group` 长度必须等于 `n_samples_per_prompt`；`data` 不能超过 `target_data_size`；filter drop 后必须减少 `remaining_batch_size`。若 custom generate fan-out 改变 group 形状，这里的断言会暴露接口不兼容。

**Comment：** 训练 rollout 的吞吐和样本质量控制都在这一段交汇：异步水位负责吞吐，dynamic filter 负责保留有效样本。

### 1.5 收尾：abort 剩余请求、排序与 hook

来源：slime/rollout/sglang_rollout.py L447-L467

**问题与约束：** 一旦有效 batch 已满，仍可能有在途请求占用 SGLang worker；训练样本还需要按原始 index 排序，并允许用户对有效样本或全量样本做后处理。

**设计选择：** 调用 `abort(args, rollout_id)` 停掉剩余请求；断言有效 batch 大小；对 `data` 与 `all_data` 按 sample index 排序；reset 全局状态后执行 sample filter 和 all-samples process hook。

**Explain：** 收尾阶段把异步生成状态折叠回一个确定的训练 batch。abort 释放 serving 资源，排序恢复数据顺序，reset 防止本轮 pending/abort 标志影响下一轮。

**Code：**

```python
    # there are still some unfinished requests, abort them
    aborted_samples = await abort(args, rollout_id)

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)
    all_samples = sorted(
        all_data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index
    )

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()
    if args.rollout_sample_filter_path is not None:
        filter_func = load_function(args.rollout_sample_filter_path)
        filter_func(args, data)

    # There can be circumstances where users want to process all samples including filtered ones.
    if args.rollout_all_samples_process_path is not None:
        process_func = load_function(args.rollout_all_samples_process_path)
        process_func(args, all_samples, data_source)

    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), aborted_samples
```

**代码逻辑：** 函数先 abort 剩余请求并收集 partial 样本；确认有效 group 数等于配置；对普通 group 与 fan-out group 都用第一个 sample 的 index 排序。随后 reset state，执行两个可选 hook，最后返回训练输出和 abort 样本。

**为什么这样写：** 异步完成顺序不等于数据顺序，排序能让下游训练和日志更稳定。abort 发生在返回前，可以避免下一轮 rollout 与上一轮未完成请求共享 router 资源。

**不变量与失败模式：** 返回前 `len(data)` 必须等于 `rollout_batch_size`；sort key 要兼容 `list[Sample]` 与 `list[list[Sample]]`；`state.reset()` 必须发生在下一轮前。若不 abort，满 batch 后多余请求仍会继续生成并消耗 worker。

**Comment：** 这一段把“足够了”转换成“可训练了”：停止多余生成，固定顺序，清理状态，交给后续训练。

---

## 2. Group 与单样本生成

### 2.1 `generate_and_rm_group`：组内并发与 group RM

来源：slime/rollout/sglang_rollout.py L294-L333

**问题与约束：** 一个 prompt group 内的多个 sample 需要并发生成，但 group reward model 需要等全组完成后才能打分；custom generate 还可能把一个 sample fan-out 成多个 sample。

**设计选择：** abort 时直接返回原 group；为没有 `session_id` 的 sample 分配 UUID；组内每个 sample 创建 `generate_and_rm` task，可选写入 deterministic seed；`gather` 后若启用 `group_rm`，对整个 group 调 `batched_async_rm`。

**Explain：** 这一层把“同一 prompt 的多次采样”封装成一个可等待任务。组内并发提高吞吐，组后 RM 保留跨样本比较或聚合的能力。

**Code：**

```python
async def generate_and_rm_group(
    args: Namespace, group: list[Sample], sampling_params: dict[str, Any], evaluation: bool = False
) -> list[Sample] | list[list[Sample]]:
    # ``generate_and_rm`` may return either a ``Sample`` or a ``list[Sample]``
    # depending on whether the ``--custom-generate-function-path`` callable
    # emits one trainable sample or several (e.g. multi-turn agent rollouts
    # that fan out into multiple prefix-chained samples). The asyncio.gather
    # below preserves whichever shape each task produced, so the group is
    # ``list[Sample]`` for plain rollouts and ``list[list[Sample]]`` for
    # the fan-out case.
    state = GenerateState(args)

    if state.aborted:
        return group

    # Generate a unique session_id for each sample in the group
    for sample in group:
        if sample.session_id is None:
            sample.session_id = str(uuid.uuid4())

    tasks = []
    for idx, sample in enumerate(group):
        current_sampling_params = sampling_params.copy()
        if getattr(args, "sglang_enable_deterministic_inference", False):
            seed = state.group_sampling_seeds[idx]
            current_sampling_params["sampling_seed"] = seed
        tasks.append(
            asyncio.create_task(generate_and_rm(args, sample, current_sampling_params, evaluation=evaluation))
        )

    group = await asyncio.gather(*tasks)

    # for the rm that need the whole group, we will do the rm here
    if not state.aborted and args.group_rm:
        with trace_span(group, "group_reward_model"):
            rewards = await batched_async_rm(args, group)
        for sample, reward in zip(group, rewards, strict=False):
            sample.reward = reward

    return group
```

**代码逻辑：** 函数拿到共享 state 后先处理 abort。随后为每个 sample 准备 session id 与 sampling params，确定性模式下按组内位置写 seed，提交 `generate_and_rm`。所有子任务完成后，若 group RM 开启且未 abort，就批量打 reward 并写回 sample。

**为什么这样写：** session id 用于 SGLang router 的一致性路由；组内 seed 可以让多样本采样可复现；group RM 延迟到 `gather` 之后，保证 reward model 能看到完整组。

**不变量与失败模式：** group 内样本数量应与配置匹配；`group_rm` 时单 sample 路径不能提前打 reward；fan-out 返回形状需要上游 filter 和排序兼容。若 abort 后仍继续 group RM，会给不完整样本打分。

**Comment：** group 层是 SGLang rollout 的并发单元，也是 reward 粒度切换的边界。

### 2.2 `generate_and_rm`：partial rollout 的历史 mask 与已完成短路

来源：slime/rollout/sglang_rollout.py L230-L239

**问题与约束：** partial rollout 可能从已有响应继续生成，旧响应来自上一轮 off-policy 状态，不应再参与本轮 loss；已完成或截断样本也不应重复生成或重复打分。

**设计选择：** 若启用 partial 且需要 mask off-policy token，就把已有响应长度对应的 `loss_mask` 置 0；随后对 `COMPLETED` 或 `TRUNCATED` 样本做短路返回，并在非 group RM 下要求 reward 已存在。

**Explain：** 这段先处理历史 token 的训练语义，再处理样本生命周期状态。只要样本已经终态，就不再进入生成 semaphore。

**Code：**

```python
    # mask previous off-policy generation for partial rollout
    if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
        sample.loss_mask = [0] * sample.response_length

    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample
```

**代码逻辑：** 条件满足时按已有 response 长度覆盖 loss mask；之后检查状态是否已经完成或截断，确认 response 存在。若 reward 不会由 group RM 补充，则要求 sample 已带 reward，然后直接返回。

**为什么这样写：** partial rollout 的旧 token 不能被当成本轮 on-policy 训练数据；完成样本重复进入 generate 会造成响应追加或 reward 覆盖。短路把续写逻辑限定在未完成样本上。

**不变量与失败模式：** `response_length` 与已有 response token 数要一致；终态样本必须有 response；非 group RM 的终态样本必须已有 reward。若忘记 mask，旧 off-policy token 会污染本轮 loss；若终态样本继续生成，会破坏样本状态机。

**Comment：** 单样本路径先守住状态机，再谈生成和打分。

### 2.3 `generate_and_rm`：并发闸门、custom generate 与 RM

来源：slime/rollout/sglang_rollout.py L241-L286

**问题与约束：** HTTP 生成和自定义生成都要受全局并发限制；abort 可能在等待 semaphore 时发生；reward 既可能由 custom generate 填好，也可能需要 per-sample RM 或 batched RM。

**设计选择：** 在 `state.semaphore` 内检查 abort，并在 `dp_rank_context` 中选择 per-sample custom generate、全局 custom generate 或默认 `generate`。离开 semaphore 后，根据 `group_rm`、返回形状与 reward 是否为空决定是否调用 RM。

**Explain：** semaphore 只保护生成阶段，不包住后续 RM 的所有路径；这样可以限制 SGLang 请求并发，同时允许 reward 逻辑按单样本或 fan-out 批量处理。

**Code：**

```python
    state = GenerateState(args)

    # generate
    async with state.semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        with state.dp_rank_context() as _:
            # Check sample.generate_function_path for per-sample custom_generate_function_path (e.g., from eval dataset config)
            custom_func_path = getattr(sample, "generate_function_path", None) or args.custom_generate_function_path

            if custom_func_path is not None:
                custom_generate_func = load_function(custom_func_path)
                # if signature has evaluation, pass evaluation
                if "evaluation" in inspect.signature(custom_generate_func).parameters:
                    sample = await custom_generate_func(args, sample, sampling_params, evaluation=evaluation)
                else:
                    sample = await custom_generate_func(args, sample, sampling_params)
            else:
                sample = await generate(args, sample, sampling_params)

    # for the rm that need the whole group, we will not do the rm here
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
        # Some custom generate paths may have already filled the reward.
        if sample.reward is None:
            with trace_span(sample, "reward_model"):
                sample.reward = await async_rm(args, sample)

    return sample
```

**代码逻辑：** 进入 semaphore 后先检查 `state.aborted`，否则在当前 dp rank 上加载并调用 custom generate 或默认 generate。生成后如果启用 group RM 就直接返回。fan-out list 路径批量补 reward；普通 sample 路径在 reward 为空时调用 `async_rm`。

**为什么这样写：** custom generate 是扩展点，可能用于 agent、多轮或 eval dataset 特化；它可能返回多个训练样本，也可能自行填 reward。通过返回形状和 reward 是否为空判断，可以兼容这些扩展，而不把所有扩展强制成默认 HTTP 形态。

**不变量与失败模式：** abort 后 sample 状态必须改为 `ABORTED`；custom generate 签名可能带或不带 `evaluation`；group RM 模式不能在单样本处打分；fan-out 中任一样本 abort 时不再补 RM。若 semaphore 外发起 generate，会绕开并发控制压垮 router。

**Comment：** 这一段是 rollout 扩展性的核心：默认 SGLang HTTP、用户 custom generate、fan-out 与 RM 策略都在这里汇合。

---

## 3. 默认 SGLang HTTP 生成

### 3.1 `_prepare_prompt_ids`：多模态、已有 tokens 与 tokenizer 三路选择

来源：slime/rollout/sglang_rollout.py L43-L62

**问题与约束：** 样本可能是纯文本，也可能带多模态输入；partial 或预处理数据可能已经有 token ids。多模态 processor 还会产出训练侧需要复用的额外输入。

**设计选择：** 先判断是否有多模态输入与能否复用 `sample.tokens`；需要 processor 时调用 processor 并缓存非 prompt key 到 `sample.multimodal_train_inputs`；能复用则返回已有 tokens，否则走 tokenizer 编码。

**Explain：** prompt ids 的来源不是唯一的。函数优先保留已有训练输入，其次让多模态 processor 按模型规则展开输入，最后才用文本 tokenizer。

**Code：**

```python
def _prepare_prompt_ids(sample: Sample, tokenizer, processor: Any) -> list[int]:
    raw_multimodal_inputs = sample.multimodal_inputs or {}
    has_multimodal_inputs = any(value is not None for value in raw_multimodal_inputs.values())
    reuse_existing_input_ids = bool(sample.tokens) and (
        sample.multimodal_train_inputs is not None or not has_multimodal_inputs
    )

    if processor and has_multimodal_inputs and not reuse_existing_input_ids:
        processor_output = processor(text=sample.prompt, **build_processor_kwargs(raw_multimodal_inputs))
        prompt_ids = processor_output["input_ids"][0]
        if sample.multimodal_train_inputs is None:
            sample.multimodal_train_inputs = {
                k: v for k, v in processor_output.items() if k not in _PROCESSOR_PROMPT_KEYS
            } or None
        return prompt_ids

    if reuse_existing_input_ids:
        return sample.tokens

    return tokenizer.encode(sample.prompt, add_special_tokens=False)
```

**代码逻辑：** 函数从 `sample.multimodal_inputs` 判断是否有有效输入，再决定已有 `sample.tokens` 是否可复用。processor 路径返回 `input_ids[0]` 并保存额外训练输入；复用路径直接返回 tokens；纯文本路径调用 tokenizer。

**为什么这样写：** 多模态输入的 token 化和训练附加张量必须由 processor 保持一致；已有 tokens 代表上游已经做过 token 化，重复编码可能改变 prompt 或破坏 partial 续写。

**不变量与失败模式：** 多模态 processor 输出必须包含 `input_ids`；复用已有 tokens 时不能丢失 multimodal train inputs；文本编码不加 special tokens。若多模态样本重复 tokenizer 编码，图像占位与模型 processor 规则会不一致。

**Comment：** 默认生成路径的第一个关口是 prompt ids 来源选择，它决定后续 HTTP payload 用 text 还是 input ids。

### 3.2 `generate`：prompt 准备与零生成截断

来源：slime/rollout/sglang_rollout.py L165-L172

**问题与约束：** 调用 SGLang 前必须先确定 prompt ids；`max_new_tokens` 可能被配置成 0，用于只保留 prompt 或立即截断的场景，此时不应发 HTTP 请求。

**设计选择：** 调用 `_prepare_prompt_ids` 后断言 `max_new_tokens >= 0`；当 `max_new_tokens == 0` 时把 sample 标记为 `TRUNCATED` 并直接返回。

**Explain：** 零生成是合法的截断状态，不是 router 错误。提前返回可以避免向 SGLang 发送无意义请求，也让样本状态显式进入终态。

**Code：**

```python
    prompt_ids = _prepare_prompt_ids(sample, state.tokenizer, state.processor)

    assert (
        sampling_params["max_new_tokens"] >= 0
    ), f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0"
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample
```

**代码逻辑：** 函数先获取 prompt ids，再校验生成长度非负；长度为 0 时不构造 payload，直接设置状态并返回当前 sample。

**为什么这样写：** 把截断状态放在 HTTP 前，可以让后续 RM 或 group 逻辑看到一致的 sample status，并减少 router 端的边界条件处理。

**不变量与失败模式：** `max_new_tokens` 不能为负；`TRUNCATED` 样本后续不应再进入默认生成；prompt ids 仍会被准备，因为样本可能需要保留 prompt token 信息。若向 router 发送 0 token 请求，不同后端可能返回不一致的 meta_info。

**Comment：** 这里把采样参数的边界值转成 Slime 自己的样本状态，而不是交给 SGLang 后端解释。

### 3.3 `generate`：HTTP payload、路由 header 与响应写回

来源：slime/rollout/sglang_rollout.py L158-L220

**问题与约束：** 默认生成需要把 Sample 转成 SGLang `/generate` 请求，并把返回 token、logprob、meta_info 和 text 写回 Sample；多模态、路由重放、一致性路由都要在 payload/header 层体现。

**设计选择：** 构造 router URL 和 payload，纯文本走 `input_ids`，多模态走 `image_data + text`；有 session id 且 router policy 为 consistent hashing 时设置 `X-SMG-Routing-Key`；请求返回后从 `meta_info.output_token_logprobs` 提取 token/logprob 并调用 `append_response_tokens`。

**Explain：** `generate` 是 Sample 与 SGLang HTTP API 的适配层。它把 Slime 内部字段翻译成 router 请求，再把 SGLang 的 meta_info 翻译回训练需要的响应 token、logprob 与状态。

**Code：**

```python
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    prompt_ids = _prepare_prompt_ids(sample, state.tokenizer, state.processor)

    assert (
        sampling_params["max_new_tokens"] >= 0
    ), f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0"
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    # Prepare payload for sglang server
    payload = {
        "sampling_params": sampling_params,
        "return_logprob": True,
    }

    if args.use_rollout_routing_replay:
        payload["return_routed_experts"] = True

    images = sample.multimodal_inputs.get("images") if sample.multimodal_inputs else None
    if images:
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in images]
        # For single-turn multimodal requests, send text so SGLang expands the
        # image placeholders with its own processor rules.
        payload["text"] = sample.prompt
    else:
        payload["input_ids"] = prompt_ids

    if not sample.tokens:
        sample.tokens = prompt_ids

    # Use session_id for consistent hashing routing (SGLang Model Gateway)
    headers = None
    if sample.session_id:
        if getattr(args, "router_policy", None) == "consistent_hashing":
            headers = {"X-SMG-Routing-Key": sample.session_id}

    with trace_span(sample, "sglang_generate", attrs={"max_new_tokens": sampling_params["max_new_tokens"]}) as span:
        output = await post(url, payload, headers=headers)
        span.update(build_sglang_meta_trace_attrs(output["meta_info"]))

    if "output_token_logprobs" in output["meta_info"]:
        new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    else:
        new_response_tokens, new_response_log_probs = [], []

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

**代码逻辑：** 函数初始化 state 和 URL，校验 sample 状态，准备 prompt ids 与 payload。根据路由重放、多模态和 session id 设置 payload/header；在 trace span 中 POST router，解析输出 token 与 logprob，并用 `append_response_tokens` 写回 sample。

**为什么这样写：** Slime 训练需要 token、logprob、文本和 meta_info，而 SGLang router 返回的是 HTTP JSON。集中在 `generate` 里做翻译，可以让 custom generate 之外的默认路径保持一致数据结构。

**不变量与失败模式：** 默认路径只接受 `PENDING` 或 `ABORTED` 状态；多模态请求不能同时依赖 input ids 和 image placeholder 展开；一致性路由 header 依赖稳定 session id。若 `return_logprob` 关闭，训练侧无法拿到 response token 的 logprob。

**Comment：** 默认生成的边界是 HTTP API，但返回后立即回到 Sample 抽象，后续 RM 和训练不需要关心 router JSON 细节。

### 3.4 `get_model_url`：custom generate 的多模型路由 helper

来源：slime/rollout/sglang_rollout.py L65-L81

**问题与约束：** 自定义 rollout 可能同时访问 policy、ref、reward 或其他 SGLang 模型；如果硬编码默认 router，就无法利用 `--sglang-config` 中的命名 router。

**设计选择：** 从 `args.sglang_model_routers` 中按 `model_name` 查找 `(ip, port)`，命中则返回对应 endpoint；未配置或未命中时回退到默认 router。

**Explain：** 这个 helper 给 custom generate 提供稳定路由入口。调用方只需要知道逻辑模型名，不需要展开 Slime 的 router 配置结构。

**Code：**

```python
def get_model_url(args: Namespace, model_name: str, endpoint: str = "/generate") -> str:
    """Return the router URL for a named model.

    Use this in custom rollout functions to route requests to a specific
    model when multiple models are deployed via ``--sglang-config``::

        url = get_model_url(args, "ref", "/generate")
        resp = await post(url, json=payload)

    Falls back to the default router if *model_name* is not found or
    ``sglang_model_routers`` is not set.
    """
    routers = getattr(args, "sglang_model_routers", None)
    if routers and model_name in routers:
        ip, port = routers[model_name]
        return f"http://{ip}:{port}{endpoint}"
    return f"http://{args.sglang_router_ip}:{args.sglang_router_port}{endpoint}"
```

**代码逻辑：** 函数读取可选 router 映射；如果映射存在且包含目标模型名，就用该模型的 ip/port 生成 URL；否则使用默认 `sglang_router_ip/port`。

**为什么这样写：** custom generate 是用户扩展点，不能要求每个用户都复制一遍 router lookup 规则。helper 统一回退策略，也让多模型配置变更时调用方少改代码。

**不变量与失败模式：** `endpoint` 应以 `/` 开头；命名 router 的 value 必须是 `(ip, port)`；默认 router 配置必须存在。若 custom generate 硬编码默认地址，多模型 rollout 会把请求发到错误模型。

**Comment：** 这不是默认 `generate` 的内部依赖，而是给扩展代码使用的路由约定。

---

## 4. Abort 与评估

### 4.1 `abort`：停止 worker 并回收 partial 样本

来源：slime/rollout/sglang_rollout.py L336-L372

**问题与约束：** 有效 batch 满后仍有 pending 请求，必须让 SGLang worker 停止生成并等 pending task 结束；partial rollout 模式下，已经生成 response 的样本不能直接丢弃，需要带 rollout 起点回到数据缓冲。

**设计选择：** 设置共享 `state.aborted=True`，根据 SGLang router 版本选择 worker 列表 API，调用 `abort_servers_until_idle`；随后 drain `state.pendings`，partial 模式下为有 response 的样本写 `metadata.start_rollout_id` 并收集返回。

**Explain：** abort 是训练 batch 收尾的一部分，不只是发一个停止信号。它要保证 worker idle，也要把正在途中的 task 结果 drain 掉，避免下一轮复用旧 pending state。

**Code：**

```python
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

    # make sure all the pending tasks are finished
    count = 0
    while state.pendings:
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)

        if not args.partial_rollout:
            continue

        # for partial rollout, collect the partial samples into the data buffer
        for task in done:
            group = task.result()
            for sample in group:
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples.append(group)
            count += len(group)

    if args.partial_rollout:
        logger.info(f"Collected {count} partial samples into the data buffer")

    return aborted_samples
```

**代码逻辑：** 函数创建 abort 样本列表，拿到共享 state 并设置 abort 标志；按 router 版本拉 worker URL，等待所有 worker idle。之后持续等待 pending task 完成，非 partial 只 drain，不收集；partial 则给有 response 的 sample 写起始 rollout id 并返回 group。

**为什么这样写：** SGLang worker 需要显式 abort，否则多余请求继续占资源；pending task 需要 drain，否则 task 结果和异常会悬挂。partial 样本回填能复用已消耗的生成预算，减少浪费。

**不变量与失败模式：** 调用 abort 时 `state.aborted` 必须尚未设置；worker API 随 SGLang 版本变化；partial 样本的 metadata 只能在没有 `start_rollout_id` 时写入。若不 drain pending，下一轮 state reset 前后可能遗留未观察 task。

**Comment：** abort 同时是资源清理、异步任务收束和 partial 数据回收。

### 4.2 `eval_rollout_single_dataset`：为每个 eval prompt 展开采样任务

来源：slime/rollout/sglang_rollout.py L561-L582

**问题与约束：** 评估不走训练 oversampling/filter，但每个 eval prompt 仍可能需要多次采样；dataset 级配置还可以注入 metadata、custom RM 和 custom generate path。

**设计选择：** 遍历 dataset samples，对每个 prompt 深拷贝 `n_samples_per_eval_prompt` 份，递增全局 sample index，注入 dataset metadata/RM/generate 配置；确定性模式下按 `rollout_seed + j` 写 sampling seed，然后提交 `generate_and_rm` task。

**Explain：** eval 路径复用单样本生成与 RM 逻辑，但不复用训练的 group oversampling 主循环。它直接把固定 eval 数据集展开成异步任务集合，后续按完成顺序收集再排序。

**Code：**

```python
    for _i, prompt_sample in enumerate(dataset.samples):
        for j in range(dataset_cfg.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
            sample.custom_rm_path = dataset_cfg.custom_rm_path
            sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
            sampling_params = base_sampling_params
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = base_sampling_params.copy()
                sampling_params["sampling_seed"] = args.rollout_seed + j
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

**代码逻辑：** 双层循环把 eval prompt 展开为多个 sample；每个 sample 都获得唯一 index、dataset metadata、custom RM path 和 custom generate path。确定性模式复制 sampling params 并写 seed，最后创建 `generate_and_rm(..., evaluation=True)` task。

**为什么这样写：** eval 需要固定数据集覆盖率，而不是通过 filter 凑 batch；深拷贝避免多次采样共享同一个 Sample 对象；dataset 级 hook 让不同 eval 集可以使用不同生成或打分逻辑。

**不变量与失败模式：** 每个 eval sample 必须有唯一 index；确定性 seed 应按同一 prompt 的采样序号变化；evaluation 标志要传入 custom generate。若不 deepcopy，同一 prompt 的多次采样会互相覆盖 response 与 reward。

**Comment：** 评估路径和训练路径共享生成/RM 原语，但任务展开策略不同：评估追求固定覆盖，训练追求有效 batch。
