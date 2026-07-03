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
updated: 2026-07-02
---

# Alt-Rollout · 源码走读

> 按 **调用顺序** 精读 6 个替代 rollout 模块。默认 sglang 路径见 [[12-SGLang-Rollout-02-源码走读]]。

---

## 0. 入口：RolloutManager 动态加载

**Explain：** `RolloutManager.__init__` 通过 `load_function` 导入 `--rollout-function-path` 指向的可调用对象，与 eval、reward post-process 并列配置。

**Code：**

```python
## 来源：slime/ray/rollout.py L437-L450
data_source_cls = load_function(self.args.data_source_path)
self.data_source = data_source_cls(args)

self.generate_rollout = load_function(self.args.rollout_function_path)
self.eval_generate_rollout = load_function(self.args.eval_function_path)
self.custom_reward_post_process_func = None
if self.args.custom_reward_post_process_path is not None:
    self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
# ...
logger.info(f"import {self.args.rollout_function_path} as generate_rollout function.")
```

**Comment：**

- 本专题 6 个模块均通过此入口接入，无需改 RolloutManager。
- `custom_reward_post_process_path` 用于 OPD 的 `post_process_rewards`。

---

## 1. fully_async_rollout · 全局 Worker 生命周期

### 1.1 模块职责与 atexit

**Explain：** 模块 docstring 阐明设计：解耦 `max_concurrent_tasks` 与 `rollout_batch_size`；ABORTED 组回灌 buffer；worker 对 pause/weight-update 无感。

**Code：**

```python
## 来源：slime/rollout/fully_async_rollout.py L1-L24
"""Fully-async rollout for slime.

Decouples ``max_concurrent_tasks`` from ``rollout_batch_size``: a background
asyncio worker keeps a fixed pool of in-flight trajectories across rollout
boundaries, so the next training step doesn't have to wait for the slowest
in-flight sample to finish.
...
The worker is intentionally oblivious to slime's higher-level pause /
weight-update signalling (e.g. ``GenerateState.aborted``). Each in-flight
generation short-circuits on those signals on its own and surfaces
:data:`Sample.Status.ABORTED`; the only piece the worker owns is
**redirecting ABORTED groups back to ``data_buffer``** instead of shipping
them to training, so the next rollout (with refreshed weights) can pick
them up.
"""
```

**Code：**

```python
## 来源：slime/rollout/fully_async_rollout.py L48-L73
_global_worker: AsyncRolloutWorker | None = None
_worker_lock = threading.Lock()

def _get_global_worker(args, data_buffer) -> AsyncRolloutWorker:
    # ... 见 01-核心概念
    return _global_worker

def _stop_global_worker() -> None:
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None

atexit.register(_stop_global_worker)
```

**Comment：**

- 进程退出时 `atexit` 停止 worker，避免 daemon 线程泄漏。
- 线程死亡时 `_get_global_worker` 会重建 worker。

### 1.2 AsyncRolloutWorker 结构与启动

**Explain：** Worker = 后台线程 + 独立 asyncio loop + 线程安全 `output_queue`。`GenerateState` 在 worker 内共享一份 sampling_params。

**Code：**

```python
## 来源：slime/rollout/fully_async_rollout.py L76-L111
class AsyncRolloutWorker:
    """Background thread + asyncio loop that continuously consumes groups
    from ``data_buffer`` and runs :func:`generate_and_rm_group` on each."""

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

**Comment：**

- `output_queue` maxsize=1000：背压保护，正常吞吐下不会满。
- `get_completed_groups` 非阻塞 drain，由主协程轮询。

### 1.3 事件循环：top-up 与 done callback

**Explain：** `_loop` 每轮：(1) reap 已完成 task；(2) 若 active < concurrency，从 buffer 取组并 `create_task(generate_and_rm_group)`；(3) sleep 1s。Done callback 处理 ABORTED 重入队。

**Code：**

```python
## 来源：slime/rollout/fully_async_rollout.py L118-L152
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
                            self.args, group, sampling_params=self.state.sampling_params.copy(), evaluation=False,
                        )
                    )
                    task.add_done_callback(self._make_done_cb(gid))
                    active_tasks.add(task)

            await asyncio.sleep(1)
        except Exception as e:
            logger.exception("fully-async loop iteration error: %s", e)
            await asyncio.sleep(1)
```

**Code：**

```python
## 来源：slime/rollout/fully_async_rollout.py L169-L191
def _make_done_cb(self, gid: int):
    def _cb(done_task: asyncio.Task) -> None:
        try:
            result = done_task.result()
        except Exception:
            logger.exception("fully-async: process task raised")
            return
        if not isinstance(result, list):
            logger.warning("fully-async: generate_and_rm_group returned %r, expected list[Sample]; dropping", type(result).__name__)
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

**Comment：**

- buffer 空时 inner while break，outer sleep 1s——避免 busy spin。
- ABORTED 重入队是 fully-async **唯一**拥有的 weight-update 协作逻辑。

### 1.4 主入口：收集 target 组

**Explain：** `_generate_rollout_async` 从 worker 非阻塞取 completed，直到 `len(collected) >= rollout_batch_size`，再按 `sample.index` 排序截取。

**Code：**

```python
## 来源：slime/rollout/fully_async_rollout.py L194-L248
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

**Comment：**

- `generate_rollout_fully_async` 用 `run()` 桥接 sync Ray 调用与 async 实现。
- evaluation 模式不支持——eval 需同步语义。

---

## 2. sglang_streaming_rollout · SSE 增量写入

### 2.1 请求构造与 base 快照

**Explain：** 与标准 `generate` 类似构造 payload，额外 `"stream": True`。在 HTTP 流开始前快照 base tokens/response/log_probs，每 chunk 重建「base + call delta」视图。

**Code：**

```python
## 来源：slime/rollout/sglang_streaming_rollout.py L69-L101
payload: dict[str, Any] = {
    "sampling_params": sampling_params,
    "return_logprob": True,
    "stream": True,
}
# ... multimodal / input_ids 分支 ...

base_tokens = list(sample.tokens)
base_response = sample.response or ""
base_response_length = sample.response_length
base_log_probs = None if sample.rollout_log_probs is None else list(sample.rollout_log_probs)
base_loss_mask = list(sample.loss_mask) if sample.loss_mask is not None else None

last_meta_info: dict[str, Any] = {}
call_tokens: list[int] = []
call_log_probs: list[float] = []
call_text: str = ""
```

**Comment：**

- base 快照支持 multi-turn：已有 tokens 上 append 本轮 call 的 cumulative chunk。
- `max_new_tokens == 0` 时直接 TRUNCATED 返回，与标准路径一致。

### 2.2 SSE 解析与 append_response_tokens

**Explain：** 读 `data:` 行，解析 JSON；从 `meta_info.output_token_logprobs` 提取 cumulative tokens；每 chunk 立即写回 sample。

**Code：**

```python
## 来源：slime/rollout/sglang_streaming_rollout.py L114-L157
async with client.stream("POST", url, json=payload, headers=headers) as response:
    response.raise_for_status()
    async for raw_line in response.aiter_lines():
        if not raw_line or not raw_line.startswith("data:"):
            continue
        data_str = raw_line[len("data:") :].strip()
        if not data_str or data_str == "[DONE]":
            continue
        chunk = json.loads(data_str)

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
            args, tokens=call_tokens, log_probs=call_log_probs,
            trainable=True, meta_info=meta, text=call_text,
            update_terminal_info=bool(meta.get("finish_reason")),
        )
        if state.aborted:
            break
```

**Comment：**

- abort 时 break 循环；无 finish_reason 则标 ABORTED。
- trace span 在流结束后用 `last_meta_info` 更新 attrs。

---

## 3. sft_rollout · 离线 tokenize

**Explain：** 懒加载 tokenizer/processor/mask generator；批量 `get_samples(rollout_batch_size)`；对每个 sample 用 messages 算 loss_mask，reward 置 0。

**Code：**

```python
## 来源：slime/rollout/sft_rollout.py L32-L68
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
        raise ValueError(f"SFT rollout produced mismatched token_ids/loss_mask lengths: ...")

    response_length = MASK_GENERATOR.get_response_lengths([loss_mask])[0]
    sample.tokens = token_ids
    sample.response_length = response_length
    sample.reward = 0
    sample.loss_mask = loss_mask[-response_length:]

return samples
```

**Comment：**

- `(sample,) = sample` 解包 group 包装——data_buffer 返回 `list[list[Sample]]`。
- `loss_mask` 只保留 response 段，与 RL rollout tensor 化一致。

---

## 4. on_policy_distillation · 教师 log-prob

### 4.1 reward_func：教师 server 调用

**Explain：** 向 `args.rm_url` POST 完整 `input_ids`；`max_new_tokens=0` 只做 forward 取 logprob；支持 multimodal image_data。

**Code：**

```python
## 来源：slime/rollout/on_policy_distillation.py L8-L29
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

**Comment：**

- 教师 server 通常是独立 SGLang 实例（见 `examples/on_policy_distillation/`）。
- 返回 JSON 含 `meta_info.input_token_logprobs`。

### 4.2 post_process_rewards：裁剪与写 teacher_log_probs

**Explain：** 从 RM 响应提取 token logprob，按 `response_length` 裁剪尾部，写入 `sample.teacher_log_probs`；标量 reward 全 0。

**Code：**

```python
## 来源：slime/rollout/on_policy_distillation.py L32-L67
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

**Comment：**

- `[1:]` 跳过 prompt 首 token 的 logprob 占位。
- 若有 task reward，可在此叠加非零 scalar。

---

## 5. sleep_rollout · Profiling 占位

**Explain：** 无限 sleep，Rollout 进程不产出数据；配合 `--rollout-function-path` 让训练侧单独跑 profile。

**Code：**

```python
## 来源：slime/rollout/sleep_rollout.py L7-L12
def sleep(args, rollout_id, data_source, evaluation=False):
    count = 0
    while True:
        time.sleep(3600)
        count += 1
        logger.info(f"rollout sleep for {count} hours")
```

**Comment：**

- 见 `docs/en/developer_guide/profiling.md`。
- generate 永不返回——仅用于特定 profiling 工作流（需配合 debug 配置）。

---

## 6. forge_load · 磁盘 replay

### 6.1 路径解析

**Explain：** `_resolve_path` 支持 literal 路径或 `{rollout_id}` 模板；训练路径缺失时 fallback 到 `0.pt`；eval literal 模式返回 None（no-op）。

**Code：**

```python
## 来源：slime/rollout/forge_load.py L40-L66
def _resolve_path(args, rollout_id: int, evaluation: bool) -> str | None:
    tpl = getattr(args, "load_forge_rollout_data", None)
    if not tpl:
        raise RuntimeError("--load-forge-rollout-data not set. ...")
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

### 6.2 generate_rollout：加载与返回

**Explain：** `torch.load` → `Sample.from_dict`；训练返回 `RolloutFnTrainOutput`；**不**修改 sample.rollout_id。

**Code：**

```python
## 来源：slime/rollout/forge_load.py L69-L114
def generate_rollout(args, rollout_id, data_source, evaluation: bool = False):
    path = _resolve_path(args, rollout_id, evaluation)

    if evaluation:
        if path is None:
            return RolloutFnEvalOutput(data={})
        blob = torch.load(path, weights_only=False)
        samples = [Sample.from_dict(s) for s in blob["samples"]]
        reward_key = args.eval_reward_key or args.reward_key
        rewards = [s.reward if (not reward_key or s.reward is None) else s.reward[reward_key] for s in samples]
        return RolloutFnEvalOutput(data={"forge_eval": {"rewards": [...], "truncated": [...], "samples": samples}})

    if path is None:
        raise RuntimeError(f"forge_load: no dump found for rollout_id={rollout_id} ...")

    blob = torch.load(path, weights_only=False)
    samples = [Sample.from_dict(s) for s in blob["samples"]]
    # IMPORTANT: do NOT overwrite sample.rollout_id with the current rollout_id.
    return RolloutFnTrainOutput(samples=samples)
```

**Comment：**

- 与 `--load-debug-rollout-data` 对比：forge 保持 SGLang + weight update 全链路。
- blob 格式与 `--save-debug-rollout-data` 输出一致。

---

## 走读小结

| 文件 | 入口函数 | 替换层级 |
|------|----------|----------|
| `fully_async_rollout.py` | `generate_rollout_fully_async` | 外层 rollout |
| `sglang_streaming_rollout.py` | `generate_streaming` | 内层 generate |
| `sft_rollout.py` | `generate_rollout` | 外层 rollout |
| `on_policy_distillation.py` | `reward_func` / `post_process_rewards` | RM hook |
| `sleep_rollout.py` | `sleep` | 外层（占位） |
| `forge_load.py` | `generate_rollout` | 外层 replay |
