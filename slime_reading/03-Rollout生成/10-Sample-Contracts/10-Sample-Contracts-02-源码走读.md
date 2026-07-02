---
type: batch-doc
module: 10-Sample-Contracts
batch: "10"
doc_type: walkthrough
title: "Sample 契约 · 源码走读"
tags:
  - slime/batch/10
  - slime/module/sample-contracts
  - slime/doc/walkthrough
updated: 2026-07-02
---

# Sample 契约 · 源码走读

---

## 1. append_response_tokens 入口

**Explain：** Agent / 多轮 rollout 逐 chunk 追加 response；区分 trainable（模型生成）与 non-trainable（tool 输出）。

**Code：**

```python
# 来源：slime/utils/types.py L253-L277
# 提交版本：22cdc6e1
def append_response_tokens(
    self,
    args=None,
    *,
    tokens=None,
    log_probs=None,
    trainable: bool = True,
    meta_info: dict | None = None,
    text: str | None = None,
    update_terminal_info: bool = True,
):
    tokens = _to_int_list(tokens)
    log_probs = _to_float_list(log_probs)
    if log_probs is not None and len(log_probs) != len(tokens):
        raise ValueError(...)
    if tokens and trainable and log_probs is None:
        raise ValueError("trainable response tokens require rollout log probabilities.")
```

**Comment：**

- trainable token **必须** 带 rollout log_probs（PPO 基准）
- non-trainable 自动填 0.0 log_prob 与 loss_mask=0

---

## 2. loss_mask 与 tokens 同步增长

**Explain：** 首次追加 trainable token 时，若已有 response 但无 mask，会初始化 mask；保证 `len(loss_mask)==response_length`。

**Code：**

```python
# 来源：slime/utils/types.py L286-L292
# 提交版本：22cdc6e1
previous_response_length = self.response_length
if tokens:
    self.tokens += tokens
    self.response_length += len(tokens)
    if self.loss_mask is None:
        self.loss_mask = [1] * previous_response_length
    self.loss_mask += [1 if trainable else 0] * len(tokens)
```

**Comment：**

- 结尾 `_validate_response_metadata_lengths` 强校验
- 与 [[27-Agent-Trajectory]] 的 tool token 路径紧密相关

---

## 3. top-p token replay 提取

**Explain：** 从 SGLang meta_info 解码 ragged top-p kept token ids；用于 `--rollout-top-p < 1` 时训练侧精确 replay 采样空间。

**Code：**

```python
# 来源：slime/utils/types.py L13-L36
# 提交版本：22cdc6e1
def _extract_rollout_top_p_token_data(
    meta_info: dict[str, Any],
    *,
    expected_num_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    token_ids = decode_int32_meta_array(meta_info, _TOP_P_TOKEN_ID_META_KEYS)
    offsets = decode_int32_meta_array(meta_info, _TOP_P_TOKEN_OFFSET_META_KEYS)
    if token_ids is None and offsets is None:
        return None
    if offsets.numel() == 0 or int(offsets[0]) != 0:
        raise ValueError(...)
    if int(offsets[-1]) != token_ids.numel():
        raise ValueError(...)
    return token_ids, offsets
```

**Comment：**

- meta key 支持别名：`top_p_token_ids` / `top_p_kept_token_ids`
- offsets 长度 = response_length + 1（ragged 数组 CSR 格式）

---

## 4. decode_int32_meta_array

**Explain：** SGLang 可能以 base64 bytes、tensor 或 list 传递 meta 数组；统一解码为 CPU int32 tensor。

**Code：**

```python
# 来源：slime/utils/misc.py L12-L34
# 提交版本：22cdc6e1
def decode_int32_meta_array(meta_info: dict[str, Any], keys: str | Iterable[str]) -> torch.Tensor | None:
    if isinstance(keys, str):
        keys = (keys,)
    for key in keys:
        if key in meta_info:
            value = meta_info[key]
            break
    else:
        return None
    if isinstance(value, str):
        import pybase64
        value = pybase64.b64decode(value.encode("ascii"))
    if isinstance(value, bytes | bytearray | memoryview):
        return torch.frombuffer(bytearray(value), dtype=torch.int32)
```

**Comment：**

- top-p 与 routed_experts 共用此解码器
- 缺失 key 返回 None（非错误）

---

## 5. _apply_meta_info 终态与 spec 统计

**Explain：** 收到 `finish_reason` 时更新 Status、speculative 统计、prefix cache、weight_version。

**Code：**

```python
# 来源：slime/utils/types.py L362-L381
# 提交版本：22cdc6e1
if not update_terminal_info or "finish_reason" not in meta_info:
    return

if getattr(args, "sglang_speculative_algorithm", False):
    self.spec_info.add(meta_info=meta_info)

self.prefix_cache_info.add(meta_info=meta_info)

match meta_info["finish_reason"]["type"]:
    case "length":
        self.status = Sample.Status.TRUNCATED
    case "abort":
        self.status = Sample.Status.ABORTED
    case "stop":
        self.status = Sample.Status.COMPLETED
```

**Comment：**

- partial rollout 时不直接用 SGLang spec 累计（注释说明）
- `weight_versions` 追踪 update_weights 版本号

---

## 6. MoE routed_experts reshape

**Explain：** 从 meta 解码后 reshape 为 `[seq-1, num_layers, topk]`，与 token 序列对齐。

**Code：**

```python
# 来源：slime/utils/types.py L352-L360
# 提交版本：22cdc6e1
routed_experts = decode_int32_meta_array(meta_info, "routed_experts")
if routed_experts is not None:
    if args is None:
        raise ValueError("args is required to decode routed experts metadata.")
    self.rollout_routed_experts = routed_experts.reshape(
        len(self.tokens) - 1,
        args.num_layers,
        args.moe_router_topk,
    )
```

**Comment：**

- 需要 `args.num_layers` / `moe_router_topk` 来自 Megatron 配置
- 训练 replay 见 [[23-CP-RoutingReplay]]

---

## 7. to_dict / from_dict 序列化

**Explain：** Sample 可序列化进 Ray object store 或 checkpoint；Status 与 nested dataclass 手动转换。

**Code：**

```python
# 来源：slime/utils/types.py L222-L244
# 提交版本：22cdc6e1
def to_dict(self):
    value = self.__dict__.copy()
    value["status"] = self.status.value
    value["spec_info"] = self.spec_info.to_dict()
    value["prefix_cache_info"] = self.prefix_cache_info.to_dict()
    return value

@staticmethod
def from_dict(data: dict):
    data = dict(data)
    data["status"] = Sample.Status(data["status"])
    data["spec_info"] = Sample.SpecInfo.from_dict(data.get("spec_info", {}))
    # ...
    sample = Sample(**init_data)
    for key, value in data.items():
        if key not in field_names:
            setattr(sample, key, value)
    return sample
```

**Comment：**

- 未知字段保留为动态属性（插件扩展）
- DataSource 持久化依赖此格式

---

## 8. get_reward_value

**Explain：** reward 可以是标量或 dict（多 RM key）；`args.reward_key` 选择子键。

**Code：**

```python
# 来源：slime/utils/types.py L246-L247
# 提交版本：22cdc6e1
def get_reward_value(self, args) -> float:
    return self.reward if not args.reward_key else self.reward[args.reward_key]
```

**Comment：**

- 见 [[13-RM-FilterHub]] 的多 RM 聚合

---

## 9. RolloutFnEvalOutput

**Explain：** eval rollout 返回嵌套 dict（dataset → metrics），与 train 的 Sample 列表区分。

**Code：**

```python
# 来源：slime/rollout/base_types.py L13-L16
# 提交版本：22cdc6e1
@dataclass
class RolloutFnEvalOutput:
    data: dict[str, dict[str, Any]]
    metrics: dict[str, Any] = None
```

**Comment：**

- `evaluation=True` 时 `call_rollout_fn` 包装 legacy 返回值

---

## 10. ParamInfo（权重同步辅助类型）

**Explain：** 同文件定义的 `ParamInfo` 描述单个权重 tensor 元数据，用于 NCCL/disk 权重同步（非 Sample 主路径，但属 types 契约）。

**Code：**

```python
# 来源：slime/utils/types.py L411-L418
# 提交版本：22cdc6e1
@dataclass(frozen=True)
class ParamInfo:
    name: str
    dtype: torch.dtype
    shape: torch.Size
    attrs: dict
    size: int
    src_rank: int
```

**Comment：**

- 详见 [[24-WeightSync-Dist]]

---

## 11. 自定义路径字段

**Explain：** Sample 可携带 per-sample 的 generate / RM 函数路径，覆盖全局 CLI。

**Code：**

```python
# 来源：slime/utils/types.py L142-L144
# 提交版本：22cdc6e1
metadata: dict = field(default_factory=dict)
generate_function_path: str | None = None
custom_rm_path: str | None = None
```

**Comment：**

- RolloutManager 解析时 `load_function(path)` 加载
- 见 [[28-Customization]]
