---
type: batch-doc
module: 11-DataSource
batch: "11"
doc_type: walkthrough
title: "DataSource · 源码走读"
tags:
  - slime/batch/11
  - slime/module/data-source
  - slime/doc/walkthrough
updated: 2026-07-02
---

# DataSource · 源码走读

> 按 **RolloutManager 启动 → get_samples → Dataset 加载 → buffer 回写 → checkpoint** 顺序精读。  
> 基线 commit：`22cdc6e1`

---

## 1. RolloutDataSource 构造：游标与 Dataset

**Explain：** `RolloutDataSource` 维护四个游标：`sample_offset`（dataset 内偏移）、`epoch_id`（第几轮遍历）、`sample_group_index` / `sample_index`（全局组号与样本号）。仅当 `rollout_global_dataset and prompt_data` 时实例化 `Dataset`。

**Code：**

```python
## 来源：slime/rollout/data_source.py L50-L59
class RolloutDataSource(DataSource):
    def __init__(self, args):
        self.args = args

        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        self.metadata = {}
```

**Comment：** `metadata` 为历史遗留字段（TODO remove），插件 buffer 有时通过 `get_metadata/update_metadata` 传递 rollout 元信息。

---

## 2. get_samples：epoch 回绕与分组

**Explain：** 从 `dataset.samples[sample_offset:]` 顺序取 prompt；若剩余不足 `num_samples`，递增 `epoch_id`、可选 shuffle、从头补齐。这是 **顺序 epoch 遍历**，不是随机有放回采样。

**Code：**

```python
## 来源：slime/rollout/data_source.py L90-L118
    def get_samples(self, num_samples):
        if self.dataset is not None:
            if self.sample_offset + num_samples <= len(self.dataset):
                prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + num_samples]
                self.sample_offset += num_samples
            else:
                prompt_samples = self.dataset.samples[self.sample_offset :]
                num_samples -= len(prompt_samples)
                self.epoch_id += 1
                if self.args.rollout_shuffle:
                    self.dataset.shuffle(self.epoch_id)
                prompt_samples += self.dataset.samples[:num_samples]
                self.sample_offset = num_samples
        else:
            prompt_samples = [Sample() for _ in range(num_samples)]

        samples = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)
        return samples
```

**Comment：**

- epoch 回绕发生在 **单个 `get_samples` 调用内部**，一次请求可跨 epoch 边界
- `num_rollout_per_epoch = len(dataset) // rollout_batch_size` 假设每步恰好消费 `rollout_batch_size` 组且不重复；over-sampling 会加速 dataset 消耗
- 只读数据源：`add_samples` 直接 `RuntimeError`

```python
## 来源：slime/rollout/data_source.py L120-L121
    def add_samples(self, samples: list[list[Sample]]):
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")
```

---

## 3. save / load：续训游标

**Explain：** checkpoint 写入 `{save}/rollout/global_dataset_state_dict_{rollout_id}.pt`，包含 offset 与 index 计数器。load 时若 `rollout_shuffle`，按恢复的 `epoch_id` 重新 shuffle。

**Code：**

```python
## 来源：slime/rollout/data_source.py L123-L160
    def save(self, rollout_id):
        if not self.args.rollout_global_dataset:
            return

        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
        }
        path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

    def load(self, rollout_id=None):
        if not self.args.rollout_global_dataset:
            return
        if self.args.load is None:
            return

        path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            logger.info(f"Checkpoint {path} does not exist.")
            return

        state_dict = torch.load(path)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})

        if self.args.rollout_global_dataset and self.args.rollout_shuffle and self.dataset is not None:
            self.dataset.shuffle(self.epoch_id)
```

**Comment：** `RolloutManager.save/load` 在训练 loop 中调用，与 Megatron checkpoint 步对齐。

---

## 4. RolloutDataSourceWithBuffer：buffer 出队

**Explain：** 子类增加 `self.buffer: list[list[Sample]]` 与 `buffer_filter`。默认 `pop_first` FIFO 弹出；自定义 filter 签名 `(args, rollout_id, buffer, num_samples) -> list[list[Sample]]`。

**Code：**

```python
## 来源：slime/rollout/data_source.py L168-L196
class RolloutDataSourceWithBuffer(RolloutDataSource):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = []
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        if len(self.buffer) == 0 or num_samples == 0:
            return []

        samples = self.buffer_filter(self.args, None, self.buffer, num_samples)
        return samples
```

**Comment：** 注意 `_get_samples_from_buffer` 传 `rollout_id=None`；若自定义 filter 依赖 rollout_id，需从别处获取（已知限制）。

**Code：**

```python
## 来源：slime/rollout/data_source.py L225-L229
def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
```

---

## 5. add_samples：入队校验

**Explain：** 入 buffer 前校验：外层 list 的每个元素必须是长度为 `n_samples_per_prompt` 的 Sample 组。

**Code：**

```python
## 来源：slime/rollout/data_source.py L198-L212
    def add_samples(self, samples: list[list[Sample]]):
        if not samples:
            return
        assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"
        assert isinstance(samples[0], list), f"the elements of samples must be list, got {type(samples[0])}"
        for i in range(0, len(samples)):
            assert (
                len(samples[i]) == self.args.n_samples_per_prompt
            ), f"the length of the elements of samples must be equal to n_samples_per_prompt, got {len(samples[i])} != {self.args.n_samples_per_prompt}"
            group = samples[i]
            self.buffer.append(group)
```

---

## 6. read_file：jsonl / parquet 与切片

**Explain：** `read_file` 统一 jsonl 行迭代与 parquet batch 迭代；`path@[start:end]` 用 `itertools.islice` 截取子集，便于 debug 小样本。

**Code：**

```python
## 来源：slime/utils/data.py L25-L68
def read_file(path):
    path, row_slice = _parse_generalized_path(path)
    reader = None

    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt dataset path '{path}' does not exist.")

    if path.endswith(".jsonl"):
        def jsonl_reader(p):
            with open(p, encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error at line {line_num}: {e}")
                        continue
        reader = jsonl_reader(path)

    elif path.endswith(".parquet"):
        if pq is None:
            raise ImportError("pyarrow is required for parquet support")
        def parquet_reader(p):
            pf = pq.ParquetFile(p)
            for batch in pf.iter_batches():
                yield from batch.to_pylist()
        reader = parquet_reader(path)
    else:
        raise ValueError(f"Unsupported file format: {path}. Supported formats are .jsonl and .parquet.")

    if row_slice is not None:
        reader = itertools.islice(reader, row_slice.start, row_slice.stop, row_slice.step)

    yield from reader
```

**Code：**

```python
## 来源：slime/utils/data.py L71-L78
def _parse_generalized_path(s: str):
    if (m := re.match(r"^(?P<real_path>.*)@\[(?P<start>-?\d*):(?P<end>-?\d*)\]$", s)) is not None:
        path = m.group("real_path")
        start = int(x) if (x := m.group("start")) != "" else None
        end = int(x) if (x := m.group("end")) != "" else None
        return path, slice(start, end)
    return s, None
```

---

## 7. _build_messages：多模态 placeholder

**Explain：** 当 `multimodal_keys` 配置时，把 JSON 中的 image 等字段按 placeholder 拆进 OpenAI 风格 message content list，供 Qwen-VL 等模型使用。

**Code：**

```python
## 来源：slime/utils/data.py L130-L174
def _build_messages(data: dict, prompt_key: str, as_conversation: bool, multimodal_keys: dict = None):
    prompt = data.get(prompt_key)

    if isinstance(prompt, str):
        if not as_conversation:
            return prompt
        else:
            prompt = [{"role": "user", "content": prompt}]

    if multimodal_keys:
        multimodals = {}
        for type_name, data_key in multimodal_keys.items():
            mt = MultimodalTypes.get(type_name)
            if mt:
                multimodal_data = data.get(data_key)
                if multimodal_data is not None:
                    multimodals[mt.placeholder] = (mt, list(multimodal_data))

        pattern = "(" + "|".join(re.escape(p) for p in multimodals.keys()) + ")"

        for message in prompt:
            if isinstance(message["content"], str):
                content_list = []
                for segment in re.split(pattern, message["content"]):
                    if not segment:
                        continue
                    if segment in multimodals:
                        mt, content = multimodals[segment]
                        item = content.pop(0)
                        if isinstance(item, dict):
                            content_list.append(item)
                        else:
                            content_list.append({"type": mt.name, mt.name: item})
                    else:
                        content_list.append({"type": "text", "text": segment})
                message["content"] = content_list
    return prompt
```

---

## 8. filter_long_prompt：加载期长度过滤

**Explain：** 纯文本 prompt 可 batch tokenize 后按 `max_length` 过滤；多模态样本逐条走 processor。list 形式 prompt 且无 chat template 时跳过检查并打 warning。

**Code：**

```python
## 来源：slime/utils/data.py L81-L127
def filter_long_prompt(origin_samples: list[Sample], tokenizer, processor, max_length: int | None) -> list[Sample]:
    if max_length is None:
        return origin_samples

    if not isinstance(origin_samples[0].prompt, str):
        logger.warning(
            "Skipping max_length check for list prompt. Set apply_chat_template=True to enable length filtering."
        )
        return origin_samples

    if processor:
        text_only = []
        multimodal = []
        for sample in origin_samples:
            if sample.multimodal_inputs and any(v is not None for v in sample.multimodal_inputs.values()):
                multimodal.append(sample)
            else:
                text_only.append(sample)
        filtered_samples = []
        if text_only:
            prompts = [s.prompt for s in text_only]
            input_ids_list = tokenizer(prompts, add_special_tokens=False)["input_ids"]
            for sample, input_ids in zip(text_only, input_ids_list, strict=True):
                if len(input_ids) <= max_length:
                    filtered_samples.append(sample)
        # ... multimodal 分支 ...
    else:
        prompts = [sample.prompt for sample in origin_samples]
        input_ids_list = tokenizer(prompts, add_special_tokens=False)["input_ids"]
        filtered_samples = [
            sample
            for sample, input_ids in zip(origin_samples, input_ids_list, strict=True)
            if len(input_ids) <= max_length
        ]

    logger.info(f"Filtered {len(origin_samples) - len(filtered_samples)} samples longer than max_length={max_length}.")
    return filtered_samples
```

---

## 9. Dataset.shuffle：确定性 permutation

**Explain：** 同一 `epoch_id` 多次调用 shuffle 为 no-op；换 epoch 时用 `random.seed(seed + epoch_id)` 生成置换，保证续训一致。

**Code：**

```python
## 来源：slime/utils/data.py L275-L283
    def shuffle(self, new_epoch_id):
        if self.epoch_id == new_epoch_id:
            return

        random.seed(self.seed + new_epoch_id)
        permutation = list(range(len(self.samples)))
        random.shuffle(permutation)
        self.samples = [self.origin_samples[i] for i in permutation]
        self.epoch_id = new_epoch_id
```

**Comment：** `origin_samples` 保留原始顺序；shuffle 只影响 `samples` 视图，过滤后的样本集不变。

---

## 10. process_rollout_data：训练侧 Ray 分片

**Explain：** Rollout 完成后 `rollout_data_ref` 按 DP rank 分发；此函数从 Ray object ref 取出本 rank 数据，并按 `partition` 重排 `total_lengths` 供 Timer / 训练使用。属于 **rollout → train 交接**，非 prompt 加载，但同属 `utils/data.py`。

**Code：**

```python
## 来源：slime/utils/data.py L292-L303
def process_rollout_data(args, rollout_data_ref, dp_rank, dp_size):
    assert len(rollout_data_ref) == dp_size
    rollout_data = ray.get(rollout_data_ref[dp_rank].inner)

    partition = rollout_data.pop("partition")
    total_lengths = rollout_data["total_lengths"]

    Timer().seq_lens = total_lengths
    rollout_data["total_lengths"] = [total_lengths[i] for i in partition]

    return rollout_data
```

---

## 11. 走读小结：调用顺序表

| 步骤 | 函数 | 触发者 |
|------|------|--------|
| 1 | `RolloutDataSourceWithBuffer.__init__` | `RolloutManager.__init__` |
| 2 | `Dataset.__init__` → `read_file` | 数据源构造 |
| 3 | `data_source.load(rollout_id)` | 续训 |
| 4 | `get_samples(over_sampling_batch_size)` | `generate_rollout_async` 循环 |
| 5 | `_get_samples_from_buffer` → `pop_first` | buffer 非空时 |
| 6 | `RolloutDataSource.get_samples` | buffer 不足时补 dataset |
| 7 | `add_samples(aborted)` | `generate_rollout` 收尾 |
| 8 | `data_source.save(rollout_id)` | 训练步 checkpoint |
