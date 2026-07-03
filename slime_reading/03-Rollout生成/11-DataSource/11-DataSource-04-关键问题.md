---
type: batch-doc
module: 11-DataSource
batch: "11"
doc_type: faq
title: "DataSource · 关键问题"
tags:
  - slime/batch/11
  - slime/module/data-source
  - slime/doc/faq
updated: 2026-07-02
---

# DataSource · 关键问题

## Q1：`rollout_global_dataset=False` 时还能用默认 generate 吗？

**Explain：** 不能。默认 `sglang_rollout.generate_rollout` / `generate_rollout_async` 开头 `assert args.rollout_global_dataset`。关闭 global dataset 意味着 prompt 由自定义 rollout 函数自行管理（如外部 buffer 服务、sleep rollout）。此时应同时替换 `--rollout-function-path` 和/或 `--data-source-path`。

**Code：**

```python
## 来源：slime/rollout/sglang_rollout.py L390, L632
    assert args.rollout_global_dataset
    # generate_rollout 入口同样 assert
```

**易错 vs 正确：**

| ❌ 易错 | ✅ 正确 |
|---------|---------|
| 只关 `--no-rollout-global-dataset`，仍用默认 sglang rollout | 自定义 `generate_rollout` + 自定义 `DataSource`，或保持 global dataset |
| 不配 `prompt_data` 期望自动有 prompt | 空 `Sample()` 需自定义逻辑填充 prompt |

---

## Q2：buffer 与 dataset 的优先级？会重复消费吗？

**Explain：** `RolloutDataSourceWithBuffer.get_samples` **先 buffer 后 dataset**，单次请求总量仍为 N，不会 double-fetch。buffer 组被 pop 后从列表删除（`pop_first` 原地 `del buffer[:num_to_pop]`）。

**Code：**

```python
## 来源：slime/rollout/data_source.py L177-L189
        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)
        if num_samples == 0:
            return samples
        samples += super().get_samples(num_samples=num_samples)
```

**Comment：** 若希望「只用 buffer、不读 dataset」，需自定义 DataSource 或保证 buffer 始终有足够组（fully-async 模式）。

---

## Q3：dynamic filter 丢弃的样本去哪了？

**Explain：** 当前 `generate_rollout_async` **未** 将被 filter 拒绝的组写回 buffer（源码注释 `# NOTE: here we have not stored all the unused samples back to the data buffer`）。这些组已消耗 dataset 游标，等于丢弃。若需回收，应实现 `--rollout-all-samples-process-path` 或 fork generate 逻辑。

**Code：**

```python
## 来源：slime/rollout/sglang_rollout.py L429-L433, L435-L437
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                state.remaining_batch_size -= 1
                continue
            # NOTE: here we have not stored all the unused samples back to the data buffer.
```

---

## Q4：如何自定义 DataSource？

**Explain：** 继承 `DataSource` 或扩展 `RolloutDataSourceWithBuffer`；注册到 `--data-source-path`。须保证 `get_samples` 返回的每组长度 = `n_samples_per_prompt`，与 `add_samples` 断言一致。

**Code：**

```python
## 来源：docs/en/get_started/customization.md（模式示意，接口同 data_source.py）
class CustomDataSource(DataSource):
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        ...

    def add_samples(self, samples: list[list[Sample]]):
        ...

    def save(self, rollout_id): ...
    def load(self, rollout_id=None): ...
    def __len__(self) -> int: ...
```

**Comment：** 插件契约测试 `test_plugin_path_loading_contracts.py` 验证 `RolloutDataSourceWithBuffer` 的 load/add/get 行为，可作参考。

---

## Q5：`buffer_filter` 与 `dynamic_sampling_filter` 区别？

| 维度 | buffer_filter | dynamic_sampling_filter |
|------|---------------|-------------------------|
| 挂载参数 | `--buffer-filter-path` | `--dynamic-sampling-filter-path` |
| 作用对象 | buffer 中 **待出队** 的组 | **已生成完** 的组 |
| 调用时机 | `get_samples` → `_get_samples_from_buffer` | `generate_rollout_async` task 完成时 |
| 默认行为 | `pop_first` FIFO | 无（None 则全保留） |
| 签名 | `(args, rollout_id, buffer, num_samples)` | `(args, group) -> DynamicFilterOutput` |

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

## Q6：`over_sampling_batch_size` 与 `rollout_batch_size` 关系？

**Explain：** 每次向 data_source 请求 `over_sampling_batch_size` 组，但循环终止条件是凑满 `rollout_batch_size` 组 **通过 filter** 的有效组。`arguments.py` 校验 `over_sampling_batch_size >= rollout_batch_size`；未设时默认等于 `rollout_batch_size`。

**Comment：** 更大的 over_sampling 减少「filter 丢弃后饥饿等待」的次数，但加速 dataset 游标前进。

---

## Q7：续训时 prompt 顺序会变吗？

**Explain：** 若 `--rollout-shuffle`，load 后按 checkpoint 中的 `epoch_id` 调用 `dataset.shuffle(epoch_id)`，与保存时 permutation 一致。`sample_offset` 恢复后从正确位置继续顺序取。若不 shuffle，纯顺序遍历，`sample_offset`  alone 决定位置。

**Code：**

```python
## 来源：slime/rollout/data_source.py L159-L160
        if self.args.rollout_global_dataset and self.args.rollout_shuffle and self.dataset is not None:
            self.dataset.shuffle(self.epoch_id)
```

---

## Q8：jsonl 一行需要什么字段？

**Explain：** 至少包含 `--input-key` 字段（默认 `input`）。可选 `--label-key`、`--metadata-key`、`--tool-key`、多模态 key。开启 `--apply-chat-template` 时 `input` 应为 OpenAI messages 列表。

**示例（jsonl 一行）：**

```json
{"input": [{"role": "user", "content": "1+1=?"}], "label": "2"}
```

**Code：**

```python
## 来源：slime/utils/arguments.py L631-L641（help 摘要）
            parser.add_argument(
                "--prompt-data",
                type=str,
                default=None,
                help=(
                    "The path to the prompt data. "
                    "Currently we only support jsonl format, and each line should contains --input-key and --label-key, "
                    "which will be used as the prompt and the label respectively. "
                ),
            )
```

---

## Q9：`len(data_source)` 何时为 0？

**Explain：** `RolloutDataSource.__len__` 在 `dataset is None` 时返回 0。此时 `get_num_rollout_per_epoch` 会除零或得 0——说明未配 prompt 数据集。有 dataset 时返回 `len(dataset)`（过滤后样本数）。

**Code：**

```python
## 来源：slime/rollout/data_source.py L162-L165
    def __len__(self) -> int:
        if self.dataset is None:
            return 0
        return len(self.dataset)
```

---

## Q10：RolloutDataSource vs RolloutDataSourceWithBuffer 怎么选？

| 场景 | 推荐 |
|------|------|
| 标准 GRPO / PPO + partial_rollout | `RolloutDataSourceWithBuffer`（**默认**） |
| 只读 prompt、无回收 | `RolloutDataSource` |
| 外部轨迹服务 / 自定义 buffer 协议 | 自定义 DataSource 或 `rollout_buffer` 插件 |

**Explain：** 默认类已是 WithBuffer；选纯 `RolloutDataSource` 会失去 `add_samples`，partial rollout  aborted 样本无法回收。
