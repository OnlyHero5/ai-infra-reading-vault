---
type: batch-doc
module: 14-Alt-Rollout
batch: "14"
doc_type: walkthrough
title: "Alt-Rollout · 源码走读"
tags:
  - slime/batch/14
  - slime/module/alt-rollout
  - slime/doc/walkthrough
updated: 2026-07-05
---

# Alt-Rollout · 源码走读

> 默认 SGLang rollout 主路径见 [[12-SGLang-Rollout-02-源码走读]]。本篇只读替代路径：它们不重写 `RolloutManager`，而是通过函数路径、custom generate、reward hook 或磁盘 replay 改变 rollout 行为。

---

## 1. RolloutManager：替代函数的接入口

### 1.1 动态加载 rollout / eval / reward 后处理

**问题与约束：** RolloutManager 是 Ray actor，负责启动 rollout server、数据源和训练数据转换；替代 rollout 不能要求用户改这个核心 actor，否则扩展成本会很高。

**设计选择：** 在初始化阶段用 `load_function` 读取 CLI 路径，把 `rollout_function_path`、`eval_function_path`、reward post-process 和 sample-to-train-data 转换函数都作为可替换插件。

**Explain：** Alt-Rollout 的共同接入点是“函数路径”，而不是继承一个新的 manager。

来源：slime/ray/rollout.py L437-L450

**Code：**

```python
data_source_cls = load_function(self.args.data_source_path)
self.data_source = data_source_cls(args)

self.generate_rollout = load_function(self.args.rollout_function_path)
self.eval_generate_rollout = load_function(self.args.eval_function_path)
self.custom_reward_post_process_func = None
if self.args.custom_reward_post_process_path is not None:
    self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
self.custom_convert_samples_to_train_data_func = None
if self.args.custom_convert_samples_to_train_data_path is not None:
    self.custom_convert_samples_to_train_data_func = load_function(
        self.args.custom_convert_samples_to_train_data_path
    )
```

**代码逻辑：** 先加载数据源类并实例化，再加载训练 rollout、eval rollout、可选 reward 后处理和可选样本转换函数；这些对象都保存在 manager 实例字段上。

**为什么这样写：** 替代 rollout 之间差异很大：有的替换外层 rollout，有的只替换单样本生成，有的只处理 reward。函数路径让扩展点足够细，不必把所有变体写进 RolloutManager。

**不变量与失败模式：** 路径必须解析为可调用对象，并且签名要匹配调用点；加载失败会在 manager 初始化阶段暴露，签名不匹配会在 rollout 调用阶段暴露。

**Comment：** 读本专题时要先判断模块替换的是哪一层：外层 rollout、内层 generate、reward hook，还是磁盘 replay。

---

## 2. fully_async_rollout：跨 step 保持热队列

### 2.1 模块职责：解耦并发任务数和训练 batch

**问题与约束：** 同步 rollout 会被最慢样本拖住；`max_concurrent_tasks` 与 `rollout_batch_size` 绑定时，下一步训练要等当前 step 的全部 in-flight 样本结束。

**设计选择：** 用后台 asyncio worker 维持固定数量的 in-flight trajectories，完成的样本组跨 rollout 边界进入输出队列；被 abort 的组重新放回 data buffer。

**Explain：** fully-async 的目标不是改变单个样本的生成逻辑，而是把“采样队列”从训练 step 的边界里拿出来。

来源：slime/rollout/fully_async_rollout.py L1-L24

**Code：**

```python
"""Fully-async rollout for slime.

Decouples ``max_concurrent_tasks`` from ``rollout_batch_size``: a background
asyncio worker keeps a fixed pool of in-flight trajectories across rollout
boundaries, so the next training step doesn't have to wait for the slowest
in-flight sample to finish.

The worker is intentionally oblivious to slime's higher-level pause /
weight-update signalling (e.g. ``GenerateState.aborted``). Each in-flight
generation short-circuits on those signals on its own and surfaces
:data:`Sample.Status.ABORTED`; the only piece the worker owns is
**redirecting ABORTED groups back to ``data_buffer``** instead of shipping
them to training, so the next rollout (with refreshed weights) can pick
them up.
"""
```

**代码逻辑：** docstring 明确三个边界：并发池跨 rollout 存活；单样本逻辑仍走 `generate_and_rm_group`；worker 只负责把 abort 组回灌 buffer。

**为什么这样写：** 这样可以复用标准 SGLang rollout 的生成、奖励和 abort 语义，只改变吞吐调度方式。

**不变量与失败模式：** worker 不能把 `Sample.Status.ABORTED` 的组送去训练；如果 abort 组没有回灌，会把旧权重下的半成品样本混入训练 batch。

**Comment：** 这类替代 rollout 的风险不是接口复杂，而是异步边界和权重更新边界是否对齐。

### 2.2 全局 worker 生命周期

**问题与约束：** Ray actor 每次调用 rollout 函数时都可能进入同一进程；如果每次都新建后台线程，队列无法保持温热，还会泄漏线程。

**设计选择：** 用模块级 `_global_worker` 和 `_worker_lock` 复用 worker；进程退出时通过 `atexit` 停止 worker。

**Explain：** fully-async 的吞吐收益来自 worker 跨 rollout 调用持续运行。

来源：slime/rollout/fully_async_rollout.py L48-L73

**Code：**

```python
_global_worker: AsyncRolloutWorker | None = None
_worker_lock = threading.Lock()

def _get_global_worker(args, data_buffer) -> AsyncRolloutWorker:
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            logger.info("starting fully-async rollout worker")
            _global_worker = AsyncRolloutWorker(
                args, data_buffer, concurrency=args.sglang_server_concurrency * get_rollout_num_engines(args)
            )
            _global_worker.start()
        return _global_worker

def _stop_global_worker() -> None:
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None

atexit.register(_stop_global_worker)
```

**代码逻辑：** 获取 worker 时加锁；不存在或线程已死就按 server concurrency 和 rollout engine 数重建；退出时 stop 并清空全局引用。

**为什么这样写：** 后台 worker 必须能在同一进程内复用，但 worker_thread 死亡也要能自愈；锁避免并发调用时重复创建。

**不变量与失败模式：** `_global_worker.worker_thread` 必须存在且存活；如果 worker 死亡后不重建，主入口会一直等不到 completed groups。

**Comment：** `atexit` 是兜底清理，不是正常的 step 边界控制。

### 2.3 AsyncRolloutWorker 的状态结构

**问题与约束：** 后台 worker 既要在线程里跑 asyncio loop，又要让主 rollout 函数安全地取已完成结果。

**设计选择：** worker 内部保存独立线程、`queue.Queue` 输出队列、`GenerateState` 和 data buffer；公开 `start`、`stop`、`get_completed_groups`。

**Explain：** 这里的同步边界是标准线程安全队列，而不是让主线程直接 await worker 内部 task。

来源：slime/rollout/fully_async_rollout.py L76-L111

**Code：**

```python
class AsyncRolloutWorker:
    def __init__(self, args, data_buffer, concurrency: int = 10):
        self.args = args
        self.data_buffer = data_buffer
        self.concurrency = concurrency
        self.running = True
        self.output_queue: queue.Queue[tuple[int, list[Sample]]] = queue.Queue(maxsize=1000)
        self.worker_thread: threading.Thread | None = None
        self.state = GenerateState(args)

    def start(self) -> None:
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._thread_main, name="fully-async-rollout", daemon=True)
            self.worker_thread.start()

    def get_completed_groups(self) -> list[tuple[int, list[Sample]]]:
        completed: list[tuple[int, list[Sample]]] = []
        while True:
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return completed
```

**代码逻辑：** 后台线程运行 `_thread_main`；输出队列限制最大积压；主线程调用 `get_completed_groups` 非阻塞 drain 当前已完成结果。

**为什么这样写：** Ray actor 调用栈和 worker 的 event loop 不在同一个 async 上下文，线程安全队列是最直接的桥。

**不变量与失败模式：** `output_queue` 中的元素必须是 `(gid, list[Sample])`；如果 callback 放入异常对象或非列表结果，主入口排序和截取会失败。

**Comment：** `queue_size()` 只是观测 warm queue，不参与调度决策。

### 2.4 worker loop：top-up active tasks

**问题与约束：** 后台 worker 需要保持固定并发，但 data buffer 可能短暂为空，任务也可能异常结束。

**设计选择：** 每轮先清理已完成 task，再从 data buffer 单组取样补足到 `max_concurrent`，为空时 break，外层 sleep。

**Explain：** 这是一个持续补水的任务池，而不是一次性创建 `rollout_batch_size` 个任务。

来源：slime/rollout/fully_async_rollout.py L118-L152

**Code：**

```python
async def _loop(self) -> None:
    active_tasks: set[asyncio.Task] = set()
    max_concurrent = self.concurrency
    gid_counter = 0

    while self.running:
        try:
            if active_tasks:
                done = {t for t in active_tasks if t.done()}
                for t in done:
                    try:
                        t.result()
                    except Exception as e:
                        logger.warning("fully-async task crashed: %r", e)
                active_tasks -= done

            while len(active_tasks) < max_concurrent and self.running:
                groups = self.data_buffer.get_samples(1)
                if not groups:
                    break
                for group in groups:
                    gid = gid_counter
                    gid_counter += 1
                    task = asyncio.create_task(
                        generate_and_rm_group(
                            self.args,
                            group,
                            sampling_params=self.state.sampling_params.copy(),
                            evaluation=False,
                        )
                    )
```

**代码逻辑：** `active_tasks` 记录未完成 task；done task 的异常在 loop 中记录；补任务时每个 group 分配递增 `gid`，并复制 sampling params 传给 `generate_and_rm_group`。

**为什么这样写：** 固定并发能稳定压住 rollout server；复制 sampling params 可以避免后续 mutation 影响已提交任务。

**不变量与失败模式：** `data_buffer.get_samples(1)` 必须返回 group 形态；如果 buffer 长期为空，worker 只会 sleep 重试，不会忙等。

**Comment：** 这里没有直接处理训练 step，它只维护持续 in-flight 池。

### 2.5 done callback：abort 回灌与输出队列

**问题与约束：** task 完成后可能返回正常 sample group、异常、非列表，或包含被 abort 的样本；这些结果不能一视同仁送去训练。

**设计选择：** callback 中验证结果类型；包含 `Sample.Status.ABORTED` 的 group 重新放回 data buffer；正常结果进入 `output_queue`。

**Explain：** 这是 fully-async 与权重更新协作的关键点。

来源：slime/rollout/fully_async_rollout.py L169-L191

**Code：**

```python
def _make_done_cb(self, gid: int):
    def _cb(done_task: asyncio.Task) -> None:
        try:
            result = done_task.result()
        except Exception:
            logger.exception("fully-async: process task raised")
            return
        if not isinstance(result, list):
            logger.warning(
                "fully-async: generate_and_rm_group returned %r, expected list[Sample]; dropping",
                type(result).__name__,
            )
            return
        if any(getattr(s, "status", None) == Sample.Status.ABORTED for s in result):
            try:
                self.data_buffer.add_samples([result])
            except Exception:
                logger.exception("fully-async: failed to requeue aborted group")
            return
        self.output_queue.put((gid, result))
    return _cb
```

**代码逻辑：** 异常 task 只记录不入队；非 list 结果丢弃；abort group 通过 `add_samples([result])` 回到数据池；正常 group 带 gid 入输出队列。

**为什么这样写：** abort 往往意味着权重或 pause 状态改变，此时继续训练会混入不一致样本；回灌让下一轮用刷新后的权重重新生成。

**不变量与失败模式：** 只有完全正常的 group 才能进入 `output_queue`；如果 `add_samples` 失败，该 group 会丢失并记录异常。

**Comment：** fully-async 的正确性主要靠这段，而不是靠主入口收集逻辑。

### 2.6 主入口：收集 rollout_batch_size 个完成组

**问题与约束：** 训练端仍需要每次 rollout 返回固定数量的 sample groups；后台 worker 的完成顺序不等于数据顺序。

**设计选择：** `_generate_rollout_async` 非阻塞 drain worker 输出，直到收集到 `rollout_batch_size`，再按 group 中 `sample.index` 排序并截取。

**Explain：** 外部看起来仍是一次普通 rollout，内部结果来自持续后台队列。

来源：slime/rollout/fully_async_rollout.py L194-L248

**Code：**

```python
async def _generate_rollout_async(args, rollout_id: int, data_buffer) -> list[list[Sample]]:
    assert args.rollout_global_dataset
    worker = _get_global_worker(args, data_buffer)

    target = args.rollout_batch_size
    collected: dict[int, list[Sample]] = {}

    while len(collected) < target:
        drained = 0
        for gid, group in worker.get_completed_groups():
            collected[gid] = group
            drained += 1

        if not drained:
            await asyncio.sleep(0.05)

    def _key(group: list[Sample]) -> int:
        for s in group:
            idx = getattr(s, "index", None)
            if idx is not None:
                return int(idx)
        return 0

    out = sorted(collected.values(), key=_key)[:target]
    return out
```

**代码逻辑：** 主协程只拉取已完成 group；没有新结果时短 sleep；收满后用 sample index 恢复 Slime 约定的确定性顺序。

**为什么这样写：** 异步完成顺序会受样本长度、server 状态和网络延迟影响；排序能减少训练数据顺序的非确定性。

**不变量与失败模式：** 该路径要求 `args.rollout_global_dataset`；evaluation 模式不支持，否则 eval 的同步语义会被后台队列破坏。

**Comment：** 这个函数是 sync Ray 调用和 async worker 之间的外层桥，最终由 `run()` 包装。

---

## 3. sglang_streaming_rollout：用 SSE 保留中途状态

### 3.1 构造 streaming 请求与 base 快照

**问题与约束：** 标准非流式 generate 要等最终 JSON 返回；如果中途 abort，sample 可能还没写入已经生成的部分 token。

**设计选择：** 请求 SGLang `/generate` 时设置 `"stream": True`，并在发起请求前快照 sample 的 base tokens、response、log_probs 和 loss_mask。

**Explain：** 每个 SSE chunk 都是“本次调用累计结果”，所以代码每次用 base 状态加 chunk delta 重建 sample。

来源：slime/rollout/sglang_streaming_rollout.py L69-L101

**Code：**

```python
payload: dict[str, Any] = {
    "sampling_params": sampling_params,
    "return_logprob": True,
    "stream": True,
}
if args.use_rollout_routing_replay:
    payload["return_routed_experts"] = True

images = sample.multimodal_inputs.get("images") if sample.multimodal_inputs else None
if images:
    payload["image_data"] = [encode_image_for_rollout_engine(image) for image in images]
    payload["text"] = sample.prompt
else:
    payload["input_ids"] = prompt_ids

base_tokens = list(sample.tokens)
base_response = sample.response or ""
base_response_length = sample.response_length
base_log_probs = None if sample.rollout_log_probs is None else list(sample.rollout_log_probs)
base_loss_mask = list(sample.loss_mask) if sample.loss_mask is not None else None
```

**代码逻辑：** payload 启用 stream 和 logprob；多模态走 `image_data` + text，纯文本走 `input_ids`；请求前保存 sample 的已有状态，支持 multi-turn 或 partial rollout 继续追加。

**为什么这样写：** SGLang streaming chunk 默认是累计输出，不能简单把每个 chunk append 到当前 sample，否则会重复写 token；base + delta 能保持幂等。

**不变量与失败模式：** 服务器若改成 incremental streaming output，当前累计处理方式需要调整；否则会把增量当累计或把累计当增量。

**Comment：** streaming 替换的是单样本 HTTP 调用，不替换外层 rollout 的 semaphore、abort 和 reward 逻辑。

### 3.2 SSE 解析与 sample 立即写回

**问题与约束：** SSE 数据可能包含空行、`[DONE]`、非 JSON chunk；同时 abort 可能在任意 chunk 后发生。

**设计选择：** 只处理 `data:` 行，解析 JSON 后提取 `meta_info.output_token_logprobs`，每个有效 chunk 立即重建 sample 并调用 `append_response_tokens`。

**Explain：** 这样即使外层 abort 打断请求，sample 至少保存了最后一个已见 chunk 的一致状态。

来源：slime/rollout/sglang_streaming_rollout.py L114-L157

**Code：**

```python
async with client.stream("POST", url, json=payload, headers=headers) as response:
    response.raise_for_status()
    async for raw_line in response.aiter_lines():
        if not raw_line or not raw_line.startswith("data:"):
            continue
        data_str = raw_line[len("data:") :].strip()
        if not data_str or data_str == "[DONE]":
            continue
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            logger.warning("sglang_streaming: skipping non-JSON chunk: %r", data_str[:120])
            continue

        meta = chunk.get("meta_info") or {}
        last_meta_info = meta
        call_text = chunk.get("text", call_text)
        if "output_token_logprobs" in meta:
            call_tokens = [item[1] for item in meta["output_token_logprobs"]]
            call_log_probs = [item[0] for item in meta["output_token_logprobs"]]

        sample.tokens = list(base_tokens)
        sample.response = base_response
        sample.response_length = base_response_length
        sample.rollout_log_probs = None if base_log_probs is None else list(base_log_probs)
        sample.loss_mask = None if base_loss_mask is None else list(base_loss_mask)
        sample.append_response_tokens(
            args,
            tokens=call_tokens,
            log_probs=call_log_probs,
            trainable=True,
            meta_info=meta,
            text=call_text,
            update_terminal_info=bool(meta.get("finish_reason")),
        )
```

**代码逻辑：** 过滤非数据行和 done 标记；JSON 错误只跳过当前 chunk；有效 chunk 更新 token、logprob、text 和终止信息；检测到 `state.aborted` 后跳出循环。

**为什么这样写：** sample 写回要早于最终响应完成，才能让 partial-rollout recycling 或 weight-update abort 拿到可训练或可回收的中间状态。

**不变量与失败模式：** `append_response_tokens` 接收的 `call_tokens` 必须与 `call_log_probs` 对齐；如果没有 finish reason 且发生 abort，函数末尾会把 sample 标为 `ABORTED`。

**Comment：** 这段的核心不是“流式展示文本”，而是“流式持久化训练样本状态”。

---

## 4. sft_rollout：离线监督数据转样本

### 4.1 tokenizer / processor / loss mask 懒加载

**问题与约束：** SFT 路径不需要访问 rollout server，但仍要产出与 RL rollout 兼容的 `Sample` 字段：tokens、response_length、loss_mask、reward。

**设计选择：** 模块级缓存 tokenizer、processor、`MultiTurnLossMaskGenerator`；从 data buffer 取 batch 后逐个 sample 生成 token ids 和 loss mask。

**Explain：** 这是把多轮消息数据离线转成训练样本的 rollout 函数。

来源：slime/rollout/sft_rollout.py L32-L68

**Code：**

```python
global TOKENIZER, PROCESSOR, MASK_GENERATOR, SAMPLE_PRINTED
if TOKENIZER is None:
    TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)

if PROCESSOR is None:
    PROCESSOR = load_processor(args.hf_checkpoint, trust_remote_code=True)

if MASK_GENERATOR is None:
    MASK_GENERATOR = MultiTurnLossMaskGenerator(TOKENIZER, tokenizer_type=args.loss_mask_type)

samples = data_buffer.get_samples(args.rollout_batch_size)

for i, sample in enumerate(samples):
    (sample,) = sample
    messages = sample.prompt
    tools = sample.metadata.get("tools", None)

    token_ids, loss_mask = MASK_GENERATOR.get_loss_mask(messages, tools=tools)
    if len(token_ids) != len(loss_mask):
        raise ValueError(...)

    response_length = MASK_GENERATOR.get_response_lengths([loss_mask])[0]
    sample.tokens = token_ids
    sample.response_length = response_length
    sample.reward = 0
    sample.loss_mask = loss_mask[-response_length:]
```

**代码逻辑：** 首次调用加载 tokenizer / processor / mask generator；data buffer 返回 group 包装，因此 `(sample,) = sample` 解出单样本；生成全序列 loss mask 后只保留 response 段。

**为什么这样写：** 后续训练张量化路径期望 response 段的 loss mask；SFT 没有环境 reward，因此 reward 设为 0，同时保留统一 sample 结构。

**不变量与失败模式：** `len(token_ids) == len(loss_mask)` 必须成立；若一个 group 不是单样本，`(sample,) = sample` 会直接解包失败。

**Comment：** SFT rollout 是“数据转换器”，不是在线生成器。

---

## 5. on_policy_distillation：教师 logprob 作为训练信号

### 5.1 reward_func：向教师服务请求 logprob

**问题与约束：** 蒸馏需要教师模型对同一 token 序列的 logprob；这里不应让教师继续生成新 token，只需要 forward 评分。

**设计选择：** reward hook 向 `args.rm_url` 发送完整 `sample.tokens`，设置 `max_new_tokens=0`、`return_logprob=True`、`logprob_start_len=0`；多模态样本补充 image data。

**Explain：** 这个 reward 函数把教师服务当成“logprob scorer”。

来源：slime/rollout/on_policy_distillation.py L8-L29

**Code：**

```python
async def reward_func(args, sample, **kwargs):
    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()
```

**代码逻辑：** payload 用已有 token 序列做输入；`max_new_tokens=0` 禁止生成；教师响应中的 logprob 由后处理函数读取。

**为什么这样写：** 蒸馏训练关心学生已生成 token 在教师下的概率，而不是教师另生成一段回答。

**不变量与失败模式：** `sample.tokens` 必须已经包含 prompt + response；教师服务需要兼容 SGLang 的 logprob 响应格式，否则后处理会找不到 `meta_info.input_token_logprobs`。

**Comment：** 这类 reward hook 返回的是结构化 JSON，不是最终标量 reward。

### 5.2 post_process_rewards：裁剪 response 段 teacher_log_probs

**问题与约束：** 教师返回的是整段输入的 token logprob，训练通常只需要 response 段，并且 Slime 的 reward 后处理接口仍要返回标量 reward 列表。

**设计选择：** 从每个样本的 reward JSON 中提取 `input_token_logprobs[1:]`，按 `response_length` 取尾部，写入 `sample.teacher_log_probs`，标量 reward 统一返回 0。

**Explain：** 真正的学习信号通过样本上的 teacher logprob 进入训练损失，标量 reward 只是保持接口兼容。

来源：slime/rollout/on_policy_distillation.py L32-L67

**Code：**

```python
def post_process_rewards(args, samples: list[Sample], **kwargs):
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    teacher_log_probs = [
        torch.tensor([item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]], dtype=torch.float32)
        for reward in raw_rewards
    ]
    teacher_log_probs = [
        t_log_prob[-response_length:]
        for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
    ]

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards
```

**代码逻辑：** `get_reward_value` 取出教师 JSON；跳过第一个 token 的 logprob；按 response length 从尾部裁剪；逐样本写 teacher logprob；返回两个同形标量列表。

**为什么这样写：** prompt token 不应参与 response 蒸馏损失；标量 reward 设 0 可以让优势估计接口继续工作，同时不引入任务奖励。

**不变量与失败模式：** 教师 logprob 长度必须覆盖 response_length；若 teacher 响应格式变化或 response_length 为错值，裁剪出的 teacher logprob 会与训练 token 不对齐。

**Comment：** 读训练后端时可以顺着 `teacher_log_probs` 找它如何进入 loss。

---

## 6. sleep_rollout：保留进程但不产出数据

### 6.1 无限 sleep

**问题与约束：** 某些 profiling 场景希望 rollout actor 和相关进程保持存在，但不希望 rollout 真的产出训练数据。

**设计选择：** 提供一个 `sleep` 函数，进入无限循环，每小时记录一次日志。

**Explain：** 这是占位 rollout，不是生成逻辑。

来源：slime/rollout/sleep_rollout.py L7-L12

**Code：**

```python
def sleep(args, rollout_id, data_source, evaluation=False):
    count = 0
    while True:
        time.sleep(3600)
        count += 1
        logger.info(f"rollout sleep for {count} hours")
```

**代码逻辑：** 调用后不读取 data source，不返回 sample，只按小时 sleep 并打印累计小时数。

**为什么这样写：** profiling 时需要一个明确不会消耗 rollout 数据、也不会触发训练数据转换的函数。

**不变量与失败模式：** 该函数永不返回；误用于正常训练会让 rollout 阶段永久阻塞。

**Comment：** 只在明确的调试或 profiling 配置里使用。

---

## 7. forge_load：磁盘 replay，同时保留 serving 链路

### 7.1 _resolve_path：训练 fallback 与 eval no-op

**问题与约束：** 内存测试常希望跳过真实 generation，但仍保留 SGLang server、weight update、offload/onload 等链路；磁盘样本路径可能是固定文件，也可能按 rollout_id 模板展开。

**设计选择：** `_resolve_path` 支持 literal 和 `{rollout_id}` 模板；训练路径找不到当前 rollout 文件时 fallback 到 `0.pt`；eval 在 literal 模式下返回 None。

**Explain：** forge replay 只替换“样本从哪里来”，不切掉 serving 生命周期。

来源：slime/rollout/forge_load.py L40-L66

**Code：**

```python
def _resolve_path(args, rollout_id: int, evaluation: bool) -> str | None:
    tpl = getattr(args, "load_forge_rollout_data", None)
    if not tpl:
        raise RuntimeError(
            "--load-forge-rollout-data not set. Pass the dump path, "
            "e.g. /path/to/rollout_data/0.pt (literal) or "
            "/path/to/rollout_data/{rollout_id}.pt (template)."
        )
    if evaluation and "{rollout_id}" not in tpl:
        return None
    rid_str = ("eval_" if evaluation else "") + str(rollout_id)
    path = tpl.format(rollout_id=rid_str)
    if os.path.exists(path):
        return path
    if not evaluation:
        fallback = tpl.format(rollout_id="0")
        if os.path.exists(fallback):
            logger.info("forge_load: %s missing, falling back to %s", path, fallback)
            return fallback
    return None
```

**代码逻辑：** 没有配置直接报错；eval literal 模式 no-op；模板路径按 `eval_` 前缀或 rollout id 展开；训练路径缺失时尝试 `0.pt`。

**为什么这样写：** 内存测试通常只有一份固定 dump，但会跑多个 rollout；训练允许 fallback 能复用样本，eval 不 fallback 则避免把训练 dump 静默塞进评测。

**不变量与失败模式：** `load_forge_rollout_data` 必须存在；模板路径要和保存文件命名一致；训练和 eval 的 fallback 语义不同，不能混用。

**Comment：** 这不是 `load_debug_rollout_data` 的等价替代，关键差别是 serving 链路仍然启动。

### 7.2 generate_rollout：加载 Sample 并保持 rollout_id

**问题与约束：** replay dump 里保存的是 sample dict；训练路径需要返回 `RolloutFnTrainOutput`，eval 路径需要返回 `RolloutFnEvalOutput`；同时不能破坏样本原有的 rollout / index 语义。

**设计选择：** 用 `torch.load(..., weights_only=False)` 读取 blob，`Sample.from_dict` 还原样本；训练直接返回样本列表，不覆盖 `sample.rollout_id`。

**Explain：** forge replay 的重点是“复用样本内容”，不是把所有样本重新标成当前 rollout。

来源：slime/rollout/forge_load.py L69-L114

**Code：**

```python
def generate_rollout(args, rollout_id, data_source, evaluation: bool = False):
    path = _resolve_path(args, rollout_id, evaluation)

    if evaluation:
        if path is None:
            logger.info("forge_load: no eval dump found; returning empty eval result")
            return RolloutFnEvalOutput(data={})
        blob = torch.load(path, weights_only=False)
        samples = [Sample.from_dict(s) for s in blob["samples"]]
        reward_key = args.eval_reward_key or args.reward_key
        rewards = [s.reward if (not reward_key or s.reward is None) else s.reward[reward_key] for s in samples]
        return RolloutFnEvalOutput(
            data={
                "forge_eval": {
                    "rewards": [r if r is not None else 0.0 for r in rewards],
                    "truncated": [s.status == Sample.Status.TRUNCATED for s in samples],
                    "samples": samples,
                }
            }
        )

    if path is None:
        raise RuntimeError(...)

    blob = torch.load(path, weights_only=False)
    samples = [Sample.from_dict(s) for s in blob["samples"]]
    return RolloutFnTrainOutput(samples=samples)
```

**代码逻辑：** eval 无 dump 时返回空 eval data；有 dump 时还原 samples、提取 reward 和 truncated 状态；训练无 dump 直接报错，有 dump 则还原 samples 并包装成训练输出。

**为什么这样写：** 训练 replay 缺文件应 fail fast；eval 在内存测试里是可选项，返回空结果更方便。保留样本原始 rollout 信息可以避免打乱 dp schedule 的分组假设。

**不变量与失败模式：** dump 必须包含 `samples`；`Sample.from_dict` 要兼容保存格式；如果强行覆盖所有 `sample.rollout_id`，可能把样本折叠进错误的 rollout 分组。

**Comment：** forge_load 适合验证系统资源和链路，不适合评估生成质量。

---

## 8. 走读小结

Alt-Rollout 的共同模式是：不改 `RolloutManager` 主体，而是在既有函数路径上替换局部行为。`fully_async_rollout` 改外层调度节奏，`sglang_streaming_rollout` 改单样本 HTTP 写回时机，`sft_rollout` 把离线数据转成 sample，`on_policy_distillation` 用教师 logprob 写入样本，`sleep_rollout` 占住进程，`forge_load` 从磁盘 replay 样本但保留 serving 链路。

判断一个替代 rollout 是否安全，主要看三点：返回对象是否满足调用层签名，sample 字段是否与训练张量化路径对齐，abort / eval / replay 这类边界场景是否有明确语义。
