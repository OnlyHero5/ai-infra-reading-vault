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
updated: 2026-07-05
---

# DataSource · 源码走读

> 读法：DataSource 负责把 prompt dataset 转成 rollout 所需的 `list[list[Sample]]`，并在 partial/abort 场景下接收样本回写。主线分为 dataset 顺序游标、buffer 优先出队、checkpoint 续训，以及 `utils/data.py` 中的数据文件读取、长度过滤和 rollout-to-train 分片。

---

## 1. RolloutDataSource：顺序 dataset 游标

### 1.1 构造函数：四个游标与 metadata

来源：slime/rollout/data_source.py L50-L59

**问题与约束：** 全局 dataset rollout 需要跨多次 `get_samples` 保持遍历位置，并为每个生成出来的 sample/group 分配稳定索引；续训时这些位置也要可保存。

**设计选择：** `RolloutDataSource` 初始化 `epoch_id`、`sample_group_index`、`sample_index`、`sample_offset` 和 `metadata`。

**Explain：** `sample_offset` 是当前 epoch 内 dataset 偏移；`epoch_id` 记录第几轮遍历；`sample_group_index` 和 `sample_index` 是 rollout 产物的全局编号。

**Code：**

```python
class RolloutDataSource(DataSource):
    def __init__(self, args):
        self.args = args

        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        # TODO remove this
        self.metadata = {}
```

**代码逻辑：** 构造函数只保存 args 并把所有游标归零，metadata 初始化为空 dict。

**为什么这样写：** Rollout 生成按 prompt group 消费，但训练和 checkpoint 需要知道全局 sample 顺序。把 dataset 偏移和 sample 编号分开，可以同时支持顺序读取、分组复制和续训恢复。

**不变量与失败模式：** `sample_offset` 只描述 dataset prompt 偏移，不等于 sample index；`sample_index` 会随 `n_samples_per_prompt` 增长更快；metadata 需要随 checkpoint 保存。若把 group index 和 sample index 混用，排序和 checkpoint 恢复都会错位。

**Comment：** 这几个字段就是 global dataset 的状态机，后续 save/load 只是在持久化它们。

### 1.2 `get_samples`：跨 epoch 顺序取 prompt 并复制成 group

来源：slime/rollout/data_source.py L90-L118

**问题与约束：** Rollout 需要一次拿到若干 prompt group；dataset 剩余数量可能不足本次请求，且每个 prompt 要复制成 `n_samples_per_prompt` 个独立 Sample。

**设计选择：** 若 dataset 足够，直接切片并推进 offset；不足时取尾部、递增 epoch、可选 shuffle，再从新 epoch 头部补齐。随后对每个 prompt sample 做 deepcopy，分配 group/sample index。

**Explain：** 这是顺序 epoch 遍历，不是随机有放回采样。一次 `get_samples` 可以跨 epoch 边界，但仍保持确定的消费顺序。

**Code：**

```python
    def get_samples(self, num_samples):
        # TODO further improve code
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

**代码逻辑：** 函数先决定 prompt_samples 来自 dataset 切片、跨 epoch 拼接，还是空 Sample fallback。之后每个 prompt sample 复制多份，给每份写相同 group index 和递增 sample index，最后返回 group 列表。

**为什么这样写：** 同一个 prompt 的多次采样需要共享 group index，便于 group RM 和动态 filter；但每个 sample 仍需要唯一 index，便于排序、训练分片和恢复。

**不变量与失败模式：** 返回值外层长度等于请求的 prompt group 数；每个 group 长度等于 `n_samples_per_prompt`；跨 epoch 时必须更新 `epoch_id` 和 `sample_offset`。若 deepcopy 缺失，同一 prompt 的多份 sample 会互相覆盖 response/reward。

**Comment：** `get_samples` 是 DataSource 的核心：把 prompt 级 dataset 消费转换成 rollout 级 sample group。

### 1.3 只读 DataSource：禁止 `add_samples`

来源：slime/rollout/data_source.py L120-L121

**问题与约束：** 基础 `RolloutDataSource` 是只读 dataset 视图；partial rollout 或 abort 样本回写需要 buffer 子类处理，不能塞回基础数据源。

**设计选择：** 基类 `add_samples` 直接抛 `RuntimeError`。

**Explain：** 这让“是否支持回写”成为类型语义。调用方如果需要回写，应使用 `RolloutDataSourceWithBuffer`。

**Code：**

```python
    def add_samples(self, samples: list[list[Sample]]):
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")
```

**代码逻辑：** 方法无条件抛错，错误信息包含当前类名。

**为什么这样写：** 回写样本可能带 response、metadata 和 partial 状态，直接插回只读 dataset 会污染原始 prompt 顺序和 checkpoint 游标。

**不变量与失败模式：** 基类实例不能接收 abort/partial 样本；调用方必须在需要回写时使用 buffer 版本。若 silent ignore，partial 样本会丢失；若直接 append dataset，会破坏 prompt-only 数据集。

**Comment：** 只读 DataSource 和带 buffer DataSource 的边界就在这一行。

### 1.4 `save/load`：持久化续训游标

来源：slime/rollout/data_source.py L123-L160

**问题与约束：** 训练 checkpoint 需要恢复 dataset 消费位置、epoch、sample/group 编号和 metadata；非 global dataset 模式不应保存这些状态。

**设计选择：** `save` 只在 `rollout_global_dataset` 下写 `global_dataset_state_dict_{rollout_id}.pt`；`load` 从 `args.load` 读取同名文件，恢复游标和 metadata，并在 shuffle 模式下按恢复的 epoch 重新 shuffle。

**Explain：** 续训恢复后，下一次 `get_samples` 应从 checkpoint 时相同位置继续，而不是从 dataset 头部重来。

**Code：**

```python
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

        logger.info(f"load metadata from {path}")
        logger.info(f"load metadata: {self.metadata}")
        state_dict = torch.load(path)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})

        if self.args.rollout_global_dataset and self.args.rollout_shuffle and self.dataset is not None:
            self.dataset.shuffle(self.epoch_id)
```

**代码逻辑：** save 构造 state dict 并写入 save 目录。load 先检查模式、load 路径和文件存在性；存在时读取 state dict，恢复字段，并按 epoch 对 dataset 视图重放 shuffle。

**为什么这样写：** shuffle 后的 dataset 顺序由 epoch 决定。恢复 offset 之前必须让 dataset 顺序回到同一 epoch 的排列，否则 offset 会指向错误 prompt。

**不变量与失败模式：** checkpoint 文件名必须和 rollout id 对齐；恢复时 `sample_offset` 与 `epoch_id` 必须同时恢复；shuffle 恢复依赖 Dataset.shuffle 的确定性。若只恢复 offset 不恢复 shuffle，续训样本顺序会偏移。

**Comment：** DataSource checkpoint 保存的是“数据消费位置”，不是样本内容本身。

---

## 2. Buffer DataSource：partial/abort 样本回写

### 2.1 `RolloutDataSourceWithBuffer`：buffer 优先，不足再读 dataset

来源：slime/rollout/data_source.py L168-L196

**问题与约束：** partial rollout 或 abort 后的样本应优先继续使用，减少已生成 token 的浪费；buffer 为空或不足时仍要从原始 dataset 补齐。

**设计选择：** 子类初始化 `self.buffer` 和可配置 `buffer_filter`；`get_samples` 先调用 `_get_samples_from_buffer`，扣减剩余数量，不足部分再调用父类 `get_samples`。

**Explain：** buffer 是 DataSource 的前置队列。它不替代 dataset，只在有回写样本时优先出队。

**Code：**

```python
class RolloutDataSourceWithBuffer(RolloutDataSource):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = []
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)

        if num_samples == 0:
            return samples

        samples += super().get_samples(num_samples=num_samples)
        return samples

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        if len(self.buffer) == 0 or num_samples == 0:
            return []

        samples = self.buffer_filter(self.args, None, self.buffer, num_samples)
        return samples
```

**代码逻辑：** 构造时选择默认 FIFO 或用户 filter。取样时先从 buffer 拿，若数量已满足就返回；否则用父类 dataset 取样补足。内部 `_get_samples_from_buffer` 在空 buffer 或请求 0 时返回空列表。

**为什么这样写：** partial 样本已经携带 response 进度，优先消费能提升生成效率。保留父类补样保证 buffer 不足时 batch size 仍可满足。

**不变量与失败模式：** buffer 中元素必须是 sample group；`buffer_filter` 会原地修改或返回样本，需遵守同样 shape；这里传入的 rollout_id 是 `None`。自定义 filter 若依赖 rollout_id，需要从其他上下文获取。

**Comment：** buffer 子类把 DataSource 从只读 prompt stream 扩展成“优先复用 partial 样本，再读新 prompt”的队列。

### 2.2 `add_samples`：入队前校验 group 形状

来源：slime/rollout/data_source.py L198-L212

**问题与约束：** 回写样本来自 abort/partial 流程，必须保持 `list[list[Sample]]` 和 `n_samples_per_prompt` 约定，否则后续 rollout group filter 或 RM 会收到错误形状。

**设计选择：** 空输入直接返回；非空时断言外层是 list、首元素是 list，并逐组检查长度等于 `args.n_samples_per_prompt`，通过后 append 到 buffer。

**Explain：** 入队校验把 shape 问题尽早暴露在 DataSource 边界，避免后续 generate/RM 才失败。

**Code：**

```python
    def add_samples(self, samples: list[list[Sample]]):
        """
        Add a sample group to buffer.
        """
        if not samples:
            return
        assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"
        assert isinstance(samples[0], list), f"the elements of samples must be list, got {type(samples[0])}"
        for i in range(0, len(samples)):
            assert (
                len(samples[i]) == self.args.n_samples_per_prompt
            ), f"the length of the elements of samples must be equal to n_samples_per_prompt, got {len(samples[i])} != {self.args.n_samples_per_prompt}"
            group = samples[i]  # type: ignore
            self.buffer.append(group)
```

**代码逻辑：** 方法先处理空列表；随后做外层/内层类型断言，循环检查每个 group 长度，最后 append。

**为什么这样写：** DataSource buffer 是后续 `get_samples` 的输入源。如果允许错形状入队，错误会在异步 rollout 中扩散且难定位。

**不变量与失败模式：** 每个 buffer item 都必须是完整 prompt group；group 长度必须和当前配置一致；配置变化后旧 buffer 可能不再合法。若 fan-out 自定义生成改变 group 结构，不能直接塞入这个 buffer。

**Comment：** buffer 的可靠性靠入队校验，不靠出队时再猜测结构。

### 2.3 `pop_first`：默认 FIFO buffer filter

来源：slime/rollout/data_source.py L225-L229

**问题与约束：** 默认情况下，buffer 需要简单按先入先出提供最多 `num_samples` 个 group，并从 buffer 中移除。

**设计选择：** 取 `min(len(buffer), num_samples)`，切片返回前 N 个 group，再 `del buffer[:num_to_pop]` 原地删除。

**Explain：** 这是最小策略的 buffer filter：不排序、不重打分，只按回写顺序复用 partial 样本。

**Code：**

```python
def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
```

**代码逻辑：** 函数计算可弹出数量，返回对应切片，并从原 buffer 删除相同前缀。

**为什么这样写：** filter 接口允许复杂策略，但默认行为应可预测。FIFO 保留 partial 样本的产生顺序，且原地删除能保持 buffer 状态同步。

**不变量与失败模式：** `buffer` 是可变 list；返回样本数量不超过请求；删除和返回必须覆盖同一段。若只返回不删除，样本会被重复 rollout。

**Comment：** `pop_first` 定义了 buffer 的默认语义：一个简单的可变 FIFO 队列。

---

## 3. 数据文件读取与样本构造工具

### 3.1 `read_file`：jsonl/parquet 统一迭代与可选切片

来源：slime/utils/data.py L25-L68

**问题与约束：** Prompt dataset 可能是 jsonl 或 parquet；调试时常需要只读一段数据；文件不存在、格式不支持和缺少 pyarrow 都要明确报错。

**设计选择：** 先解析 generalized path，再按扩展名创建 generator；jsonl 逐行 parse，空行跳过，坏行打印后继续；parquet 用 batch 迭代；最后用 `itertools.islice` 应用行切片。

**Explain：** `read_file` 提供统一的 dict iterator。Dataset 构造无需关心底层文件格式，只消费一条条 Python dict。

**Code：**

```python
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

        logger.info("read_file path=%s applying slice row_slice=%s", path, row_slice)
        reader = itertools.islice(reader, row_slice.start, row_slice.stop, row_slice.step)

    yield from reader
```

**代码逻辑：** 函数先拆 path 和 slice；检查文件存在；按扩展名构造 reader；unsupported 直接报错；最后在 reader 上应用可选 slice 并 yield。

**为什么这样写：** jsonl 和 parquet 的读取方式差异大，但上层 Dataset 只需要统一记录流。切片放在 reader 之后，可以复用同一格式解析逻辑。

**不变量与失败模式：** 仅支持 `.jsonl` 和 `.parquet`；jsonl 坏行会跳过而不是中止；parquet 需要 pyarrow；slice 只作用在解析后的记录流。若文件格式误判，上层会拿不到任何样本。

**Comment：** `read_file` 是 dataset 的 I/O 边界：格式差异在这里被抹平。

### 3.2 `_parse_generalized_path`：支持 `path@[start:end]`

来源：slime/utils/data.py L71-L78

**问题与约束：** 用户需要在不改数据文件的情况下读取子集，例如 debug 小样本；路径字符串本身要同时携带真实路径和切片范围。

**设计选择：** 用正则匹配 `real_path@[start:end]`，将空 start/end 转成 `None`，返回 `(path, slice(start, end))`；不匹配则返回原字符串和 `None`。

**Explain：** 这是 `read_file` 的路径扩展语法解析器。它只支持 start/end，不支持 step。

**Code：**

```python
def _parse_generalized_path(s: str):
    if (m := re.match(r"^(?P<real_path>.*)@\[(?P<start>-?\d*):(?P<end>-?\d*)\]$", s)) is not None:
        path = m.group("real_path")
        start = int(x) if (x := m.group("start")) != "" else None
        end = int(x) if (x := m.group("end")) != "" else None
        return path, slice(start, end)

    return s, None
```

**代码逻辑：** 正则命中时抽出真实路径和两个边界字符串；非空边界转 int，空边界为 None；返回 slice。未命中返回原路径。

**为什么这样写：** 把切片编码在路径里，可以让 CLI 参数保持单字符串，同时不引入额外 dataset subset 参数。

**不变量与失败模式：** 语法必须包含 `@[...]`；当前实现没有解析 step；负数边界会按 Python slice 语义传给 `islice`，但 `itertools.islice` 对负数不接受，用户需避免无效范围。若真实路径中含类似模式，可能被解析成 generalized path。

**Comment：** 这是一个调试便利语法，真正执行切片仍在 `read_file`。

### 3.3 `filter_long_prompt`：加载期长度过滤

来源：slime/utils/data.py L81-L127

**问题与约束：** Prompt 太长会超过模型或 rollout 限制；纯文本可以 batch tokenize，多模态样本需要 processor 才能得到真实 token 长度；list prompt 若未套 chat template，长度检查不可靠。

**设计选择：** `max_length=None` 直接返回；非字符串 prompt 打 warning 并跳过；有 processor 时拆成 text-only 和 multimodal 两组，分别 tokenizer 或 processor；无 processor 时 batch tokenizer；最后记录过滤数量。

**Explain：** 这段在 dataset 加载期做静态过滤，减少 rollout 阶段才发现超长 prompt 的概率。

**Code：**

```python
def filter_long_prompt(origin_samples: list[Sample], tokenizer, processor, max_length: int | None) -> list[Sample]:
    if max_length is None:
        return origin_samples

    if not isinstance(origin_samples[0].prompt, str):
        logger.warning(
            "Skipping max_length check for list prompt. Set apply_chat_template=True to enable length filtering."
        )
        return origin_samples

    if processor:
        # Use processor only for samples with actual multimodal content; use batched tokenizer for text-only.
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
        if multimodal:
            from slime.utils.processing_utils import process_vision_info

            for sample in multimodal:
                multimodal_inputs = process_vision_info(sample.prompt, processor)
                processor_output = processor(text=sample.prompt, **multimodal_inputs)
                input_ids = processor_output["input_ids"][0]
                if len(input_ids) <= max_length:
                    filtered_samples.append(sample)
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

**代码逻辑：** 函数先处理关闭过滤和 list prompt；processor 路径把样本分流，text-only 用 tokenizer 批处理，多模态逐条调用 processor；无 processor 时全部 tokenizer 批处理。每条只保留 token 长度不超过 max_length 的 sample。

**为什么这样写：** 多模态 token 长度取决于 processor 对图像/占位符的展开，不能用普通 tokenizer 估算。text-only 仍走 batch tokenizer，避免所有样本都被逐条 processor 拖慢。

**不变量与失败模式：** `origin_samples` 不能为空；list prompt 未模板化时跳过检查；多模态 processor 输出必须有 `input_ids`。若多模态样本被当纯文本估长，可能把实际超长 prompt 放进 rollout。

**Comment：** 长度过滤在加载期尽量做，但对无法可靠估长的 prompt 明确跳过而不是猜测。

### 3.4 `_build_messages`：将多模态 placeholder 展开为 message content list

来源：slime/utils/data.py L130-L174

**问题与约束：** 数据集中可能用文本 placeholder 表示图片等多模态内容；OpenAI 风格 message content 需要把文本段和多模态 item 拆成列表，且 placeholder 数量不能超过实际数据。

**设计选择：** 若 prompt 是字符串且需要 conversation，先包成 user message；配置了 `multimodal_keys` 时，构建 placeholder 到内容列表的映射，用正则拆 message content，把 placeholder 替换成 dict 或 `{type, value}`。

**Explain：** 这一步把 dataset 的紧凑表示转换成模型/processor 可理解的 conversation content 结构。

**Code：**

```python
def _build_messages(data: dict, prompt_key: str, as_conversation: bool, multimodal_keys: dict = None):
    prompt = data.get(prompt_key)

    if isinstance(prompt, str):
        # If prompt is a string and we don't apply chat template, return the prompt as is.
        if not as_conversation:
            return prompt
        else:
            prompt = [{"role": "user", "content": prompt}]

    if multimodal_keys:
        # Build mapping: placeholder -> (MultimodalType, content_list)
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
                        assert len(content) > 0, (
                            f"Not enough {mt.name} data: more '{mt.placeholder}' placeholders in prompt "
                            f"than {mt.name}s provided in data"
                        )
                        item = content.pop(0)
                        # Support rich image config from https://github.com/QwenLM/Qwen3-VL/blob/main/README.md
                        # "images": [{"type": "image", "image": "path/to/img/01.jpeg", "max_pixels": 50176, "min_pixels": 50176}, {...}]
                        if isinstance(item, dict):
                            content_list.append(item)
                        # "images": ["path/to/img/01.jpeg", "url", "base64enc"]
                        else:
                            content_list.append({"type": mt.name, mt.name: item})
                    else:
                        content_list.append({"type": "text", "text": segment})
                message["content"] = content_list
```

**代码逻辑：** 函数先把字符串 prompt 按需要转成 conversation。多模态路径中，先收集 placeholder 对应的内容队列，再逐条 message 拆分文本；placeholder 命中时弹出一个多模态 item，非 placeholder 段转成 text item。

**为什么这样写：** 同一 prompt 中可以交替出现文本和图片占位符。按 placeholder 顺序 pop 内容，可以保持文本位置与图像数据一一对应。

**不变量与失败模式：** placeholder 数量不能超过数据数量；rich dict item 直接保留；普通 item 要包装成对应 type。若 multimodal_keys 为空但 prompt 仍包含占位符，后续 processor 会看到未展开的文本。

**Comment：** `_build_messages` 把数据文件里的多模态约定落实成实际 message 结构，是 processor 前的格式桥。

---

## 4. Dataset 顺序与 rollout-to-train 交接

### 4.1 `Dataset.shuffle`：按 epoch 的确定性 permutation

来源：slime/utils/data.py L275-L283

**问题与约束：** shuffle 要在每个 epoch 改变样本顺序，但同一 epoch 的重复调用必须稳定，尤其是 checkpoint 恢复后要得到同一排列。

**设计选择：** 若当前 `epoch_id` 已等于目标 epoch，直接返回；否则用 `seed + new_epoch_id` 设置随机种子，生成 permutation，从 `origin_samples` 重建 `samples` 视图。

**Explain：** `origin_samples` 保存过滤后的原始顺序，`samples` 是当前 epoch 的视图。shuffle 不在当前 `samples` 上继续打乱，而是每次从 origin 重建。

**Code：**

```python
    def shuffle(self, new_epoch_id):
        if self.epoch_id == new_epoch_id:
            return

        random.seed(self.seed + new_epoch_id)
        permutation = list(range(len(self.samples)))
        random.shuffle(permutation)
        self.samples = [self.origin_samples[i] for i in permutation]
        self.epoch_id = new_epoch_id
```

**代码逻辑：** 函数先做 epoch 去重；新 epoch 时设置随机种子，生成下标列表并打乱，再用下标从 origin samples 生成当前 samples。

**为什么这样写：** 从 origin samples 重建可以避免多次 shuffle 累积误差；seed+epoch 让续训恢复到相同 epoch 时排列可重现。

**不变量与失败模式：** 同一 epoch 多次调用必须 no-op；origin samples 不应被修改；epoch id 更新必须发生在 samples 重建后。若恢复时不调用 shuffle，offset 会指向未打乱顺序的样本。

**Comment：** Dataset 的确定性 shuffle 是 DataSource checkpoint 能准确续训的前提之一。

### 4.2 `process_rollout_data`：按 DP rank 取 Ray 数据并重排长度

来源：slime/utils/data.py L292-L303

**问题与约束：** Rollout 完成后的数据按 DP rank 分成 Ray object ref；训练侧每个 rank 只应取自己的分片，并把 `total_lengths` 按 partition 重排供训练和 Timer 使用。

**设计选择：** 断言 ref 数等于 dp size，取 `rollout_data_ref[dp_rank].inner`，弹出 `partition`，保存原始 total lengths 到 Timer，再按 partition 重排写回。

**Explain：** 这段属于 rollout 到 train 的交接，不负责 prompt 读取，但和 DataSource 同处数据流工具层。

**Code：**

```python
def process_rollout_data(args, rollout_data_ref, dp_rank, dp_size):
    assert len(rollout_data_ref) == dp_size
    rollout_data = ray.get(rollout_data_ref[dp_rank].inner)

    partition = rollout_data.pop("partition")
    total_lengths = rollout_data["total_lengths"]

    # save the seqlen of the whole rollout batch
    Timer().seq_lens = total_lengths
    rollout_data["total_lengths"] = [total_lengths[i] for i in partition]

    return rollout_data
```

**代码逻辑：** 函数校验 DP 分片数量，取本 rank 的 Ray object；从数据里移除 partition，读取总长度列表，写入全局 Timer，再按 partition 生成本 rank 对应的长度顺序。

**为什么这样写：** 训练 rank 只处理自己分到的样本，但 Timer 可能需要看到整批长度分布。先保存整批，再重排本 rank 数据，可以同时满足 profiling 和训练输入。

**不变量与失败模式：** `rollout_data_ref` 长度必须等于 DP size；每个 object 内必须有 `partition` 和 `total_lengths`；partition 下标必须合法。若 partition 和数据内容不一致，本 rank 的 sequence length 会和样本错配。

**Comment：** DataSource 生产 prompt group，`process_rollout_data` 则在 rollout 后把结果交回训练 rank；二者构成数据流的前后半段。
