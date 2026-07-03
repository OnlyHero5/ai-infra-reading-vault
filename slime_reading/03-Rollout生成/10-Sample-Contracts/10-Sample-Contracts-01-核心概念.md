---
type: batch-doc
module: 10-Sample-Contracts
batch: "10"
doc_type: concept
title: "Sample 契约 · 核心概念"
tags:
  - slime/batch/10
  - slime/module/sample-contracts
  - slime/doc/concept
updated: 2026-07-02
---

# Sample 契约 · 核心概念

---

## 1. Sample 身份字段

**Explain：** `group_index` / `index` 标识 GRPO 等同 prompt 多采样分组；`rollout_id` 标识 **一次 rollout 执行**（compact 路径下一 execution 可拆多个 training sample）。

**Code：**

```python
## 来源：slime/utils/types.py L97-L106
# 提交版本：22cdc6e1
group_index: int | None = None
index: int | None = None
# downstream pipeline falls back to index when rollout_id is None
rollout_id: int | None = None
prompt: str | list[dict[str, str]] = ""
tokens: list[int] = field(default_factory=list)
```

**Comment：**

- `prompt` 可为 chat message list（多模态/agent 路径）
- `tokens` = prompt tokens + response tokens 拼接序列

---

## 2. 训练相关字段

**Explain：** response 侧核心：`response_length`、`loss_mask`、`rollout_log_probs`、`reward`；PPO/GRPO 依赖 log_probs 与 mask 对齐。

**Code：**

```python
## 来源：slime/utils/types.py L114-L128
# 提交版本：22cdc6e1
response: str = ""
response_length: int = 0
label: str | None = None
reward: float | dict[str, Any] | None = None
loss_mask: list[int] | None = None
rollout_log_probs: list[float] | None = None
rollout_top_p_token_ids: list[int] | torch.Tensor | None = None
rollout_top_p_token_offsets: list[int] | torch.Tensor | None = None
rollout_routed_experts: list[list[int]] | torch.Tensor | None = None
remove_sample: bool = False
teacher_log_probs: list[float] | None = None
```

**Comment：**

| 字段 | 训练用途 |
|------|----------|
| `loss_mask` | 0=不参与 loss（tool token 等） |
| `rollout_log_probs` | PPO ratio / KL 基准 |
| `rollout_top_p_*` | top-p < 1 时 logprob replay |
| `rollout_routed_experts` | MoE routing replay |
| `teacher_log_probs` | OPD 蒸馏 |

---

## 3. Sample.Status 枚举

**Explain：** 生成终态影响是否进入训练 batch；FAILED 与 ABORTED 语义不同（前者可有部分有效输出）。

**Code：**

```python
## 来源：slime/utils/types.py L130-L140
# 提交版本：22cdc6e1
class Status(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"
    FAILED = "failed"

status: Status = Status.PENDING
```

**Comment：**

- `_apply_meta_info` 根据 SGLang `finish_reason.type` 设置终态
- Filter Hub 可能标记 `remove_sample=True` 丢弃

---

## 4. RolloutBatch 类型别名

**Explain：** Megatron 路径上 Sample 列表被转为 **dict of lists**（tokens、log_probs 等），再在各 rank 上转为 GPU Tensor。

**Code：**

```python
## 来源：slime/utils/types.py L421-L424
# 提交版本：22cdc6e1
RolloutBatch = dict[str, list[torch.Tensor] | list[int] | list[float] | list[str]]
```

**Comment：**

- 非 dataclass，运行时 dict
- 转换逻辑在 `megatron_utils.actor._get_rollout_data`（[[20-Train-Data-00-MOC]]）

---

## 5. RolloutFn 输出契约

**Explain：** 自定义 `--rollout-function-path` 应返回 `RolloutFnTrainOutput` 或 legacy list；`call_rollout_fn` 统一包装。

**Code：**

```python
## 来源：slime/rollout/base_types.py L7-L26
# 提交版本：22cdc6e1
@dataclass
class RolloutFnTrainOutput:
    samples: list[list[Sample]]
    metrics: dict[str, Any] = None

def call_rollout_fn(fn, *args, evaluation: bool, **kwargs):
    output = fn(*args, **kwargs, evaluation=evaluation)
    if not isinstance(output, (RolloutFnTrainOutput, RolloutFnEvalOutput)):
        output = RolloutFnEvalOutput(data=output) if evaluation else RolloutFnTrainOutput(samples=output)
    return output
```

**Comment：**

- `samples` 外层 list = rollout batch 维度，内层 = 每 prompt 的 n_samples
- eval 路径返回 `RolloutFnEvalOutput`

---

## 6. load_function 动态加载

**Explain：** Slime 大量使用 `"pkg.module.function"` 字符串配置自定义 hook（rollout、RM、model provider 等）。

**Code：**

```python
## 来源：slime/utils/misc.py L37-L45
# 提交版本：22cdc6e1
def load_function(path):
    """
    Load a function from a module.
    :param path: The path to the function, e.g. "module.submodule.function".
    :return: The function object.
    """
    module_path, _, attr = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)
```

**Comment：**

- 路径必须含至少一个 `.`（module + attr）
- import 失败时直接抛异常，无 fallback

---

## 7. effective_response_length

**Explain：** 有 `loss_mask` 时用 mask 求和作为有效 response token 数（用于 metrics / 动态 batch）。

**Code：**

```python
## 来源：slime/utils/types.py L249-L251
# 提交版本：22cdc6e1
@property
def effective_response_length(self):
    return sum(self.loss_mask) if self.loss_mask is not None else self.response_length
```

**Comment：**

- tool call 插入的 non-trainable token 不计入 effective length
