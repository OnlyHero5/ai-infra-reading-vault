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
updated: 2026-07-05
---

# Sample 契约 · 源码走读

> 走读主线：`Sample` 是 Slime 从 rollout 到训练后端的最小样本契约。它同时承载 prompt/response、训练 loss mask、rollout logprob、top-p replay、MoE routing replay、spec/prefix-cache 统计、reward 与自定义函数路径。核心风险不是字段多少，而是 response token、mask、logprob 和 ragged metadata 必须持续对齐。

---

## 1. meta 数组与 top-p replay 基础工具

### 1.1 decode_int32_meta_array 统一 SGLang meta 数组格式

问题与约束：
- SGLang meta_info 里的数组可能是 base64 字符串、bytes、tensor、numpy-like 对象或 Python list；Sample 侧需要统一得到 CPU int32 一维 tensor。

设计选择：
- `decode_int32_meta_array` 支持单 key 或多个候选 key；缺失或值为 None 返回 None；字符串先 base64 decode，bytes 走 `torch.frombuffer`，tensor detach 到 CPU，其余对象用 `torch.as_tensor`。

Explain：
这个函数被 top-p replay 和 routed experts 复用，因此它是所有 int32 meta 契约的入口。

来源：slime/utils/misc.py L12-L34

Code：

```python
def decode_int32_meta_array(meta_info: dict[str, Any], keys: str | Iterable[str]) -> torch.Tensor | None:
    if isinstance(keys, str):
        keys = (keys,)
    for key in keys:
        if key in meta_info:
            value = meta_info[key]
            break
    else:
        return None

    if value is None:
        return None
    if isinstance(value, str):
        import pybase64

        value = pybase64.b64decode(value.encode("ascii"))
    if isinstance(value, bytes | bytearray | memoryview):
        return torch.frombuffer(bytearray(value), dtype=torch.int32)
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", dtype=torch.int32).reshape(-1)
    if hasattr(value, "flags") and not value.flags.writeable:
        value = value.copy()
    return torch.as_tensor(value, dtype=torch.int32).reshape(-1)
```

代码逻辑：
- 把单 key 规范化成 tuple。
- 按候选 key 顺序查找第一个存在的值。
- 缺失或 None 不报错。
- 字符串按 base64 解码。
- bytes-like 转成 int32 buffer。
- tensor detach 到 CPU 并 reshape。
- numpy-like 只读对象先 copy。

为什么这样写：
- SGLang 与 HTTP/IPC 边界可能选择不同序列化格式，Sample 不应把格式差异扩散到训练链路。
- 缺失 meta 是合法场景，例如没有开启 top-p replay 或 routing replay。

不变量与失败模式：
- bytes 长度必须能按 int32 对齐。
- 字符串必须是 ASCII base64。
- 返回 tensor 总是一维 CPU int32。

Comment：
读 Sample meta 处理时，先看这个解码器；它决定了后续 shape 校验的输入形态。

### 1.2 _extract_rollout_top_p_token_data 校验 ragged top-p CSR

问题与约束：
- top-p replay 需要保存每个生成 token 当时保留的候选 token id；这是 ragged 数组，必须用 ids + offsets 表达。

设计选择：
- 允许两组 meta key 别名；如果 ids 和 offsets 都缺失，返回 None；如果只缺一个、offsets 不从 0 开始、末尾 offset 不等于 ids 数量，或 offsets 长度不等于 token 数 + 1，就抛 ValueError。

Explain：
返回值是 `(token_ids, offsets)`，其中第 i 个 response token 的候选集合是 `token_ids[offsets[i]:offsets[i+1]]`。

来源：slime/utils/types.py L13-L36

Code：

```python
def _extract_rollout_top_p_token_data(
    meta_info: dict[str, Any],
    *,
    expected_num_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    token_ids = decode_int32_meta_array(meta_info, _TOP_P_TOKEN_ID_META_KEYS)
    offsets = decode_int32_meta_array(meta_info, _TOP_P_TOKEN_OFFSET_META_KEYS)
    if token_ids is None and offsets is None:
        return None
    if token_ids is None or offsets is None:
        raise ValueError("SGLang top-p token replay must include both token ids and offsets.")
    if offsets.numel() == 0 or int(offsets[0]) != 0:
        raise ValueError(f"SGLang top-p token offsets must start with 0, got {offsets[:1].tolist()}.")
    if int(offsets[-1]) != token_ids.numel():
        raise ValueError(
            "SGLang top-p token ids/offsets mismatch: "
            f"offsets[-1]={int(offsets[-1])}, len(token_ids)={token_ids.numel()}."
        )
    if expected_num_tokens is not None and offsets.numel() != expected_num_tokens + 1:
        raise ValueError(
            "SGLang top-p token offsets length must equal generated token count + 1: "
            f"len(offsets)={offsets.numel()}, generated={expected_num_tokens}."
        )
    return token_ids, offsets
```

代码逻辑：
- 分别解码 ids 和 offsets。
- 两者都缺失表示没有 top-p replay。
- 两者只缺一个直接失败。
- offsets 必须非空并从 0 开始。
- offsets 最后一项必须等于 ids 总数。
- 可选校验 offsets 长度和本次 token 数对齐。

为什么这样写：
- top-p replay 用于训练侧复现采样空间，错位会造成 token 与候选集合不匹配。
- 用严格 ValueError 比静默丢弃 replay 信息更安全。

不变量与失败模式：
- offsets 长度必须是生成 token 数 + 1。
- ids 和 offsets 必须同时存在。
- offsets[-1] 必须等于 ids numel。

Comment：
这个函数把 SGLang 的 top-p meta 变成训练可验证的 CSR 契约。

### 1.3 top-p merge 与 padding 维护多段 response 对齐

问题与约束：
- Agent 或 tool 场景会多次追加 response；有的 chunk 是 trainable 生成 token，有的 chunk 是 non-trainable tool token。top-p offsets 必须跨 chunk 连续。

设计选择：
- `_merge_rollout_top_p_token_data` 把新 token ids 拼到旧 ids 后，并把新 offsets 从第二项开始加上旧末尾 offset；`_pad_rollout_top_p_offsets` 为没有 top-p 候选的 non-trainable token 追加空 span。

Explain：
padding 不增加 token ids，只把 offsets 用旧末尾值重复 num_tokens 次，因此这些 token 的候选集合为空。

来源：slime/utils/types.py L39-L70

Code：

```python
def _merge_rollout_top_p_token_data(
    base_token_ids: list[int] | torch.Tensor | None,
    base_offsets: list[int] | torch.Tensor | None,
    token_ids: torch.Tensor,
    offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    base_token_ids = torch.as_tensor([] if base_token_ids is None else base_token_ids, dtype=torch.int32).reshape(-1)
    base_offsets = torch.as_tensor([0] if base_offsets is None else base_offsets, dtype=torch.int32).reshape(-1)
    base_offset = int(base_offsets[-1])
    return (
        torch.cat([base_token_ids, token_ids]),
        torch.cat([base_offsets, offsets[1:] + base_offset]),
    )


def _pad_rollout_top_p_offsets(
    token_ids: list[int] | torch.Tensor | None,
    offsets: list[int] | torch.Tensor | None,
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if offsets is None or token_ids is None:
        raise ValueError("Cannot append empty top-p spans without existing token ids and offsets.")
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}.")
    token_ids = torch.as_tensor(token_ids, dtype=torch.int32).reshape(-1)
    offsets = torch.as_tensor(offsets, dtype=torch.int32).reshape(-1)
    if offsets.numel() == 0:
        raise ValueError("Cannot append empty top-p spans to empty offsets.")
    if num_tokens == 0:
        return token_ids, offsets
    empty_offsets = offsets.new_full((num_tokens,), int(offsets[-1]))
    return token_ids, torch.cat([offsets, empty_offsets])
```

代码逻辑：
- 旧 ids 缺失时按空数组处理。
- 旧 offsets 缺失时按 `[0]` 处理。
- 新 offsets 去掉第一个 0 后加旧末尾 offset。
- padding 要求已有 ids 和 offsets。
- padding 数为 0 时原样返回。
- padding 正数时重复末尾 offset。

为什么这样写：
- 多轮追加时 offsets 不能重新从 0 开始，否则 response token 索引会错位。
- non-trainable token 不参与训练采样 replay，但仍要占据 response token 位置。

不变量与失败模式：
- padding 只能发生在已有 top-p replay 的样本上。
- `num_tokens` 不能为负。
- offsets 不能为空。

Comment：
这两段是多段 response 场景下保持 top-p replay 连续性的关键。

---

## 2. Sample 基础字段与统计对象

### 2.1 Sample 字段覆盖 prompt、response、训练标记与 rollout meta

问题与约束：
- 一个样本要同时服务 rollout、reward、训练、路由 replay、多模态和可扩展自定义逻辑，字段必须能承载这些跨阶段信息。

设计选择：
- `Sample` dataclass 将 prompt/token、response/reward/loss_mask、weight_versions、rollout_log_probs、top-p replay、routed_experts、metadata、自定义函数路径、train_metadata 和 session_id 放在同一个对象中。

Explain：
`Status` 初始为 `PENDING`，后续由 SGLang finish_reason 转成 completed/truncated/aborted，或由外部逻辑标记 failed。

来源：slime/utils/types.py L93-L149

Code：

```python
@dataclass
class Sample:
    group_index: int | None = None
    index: int | None = None
    rollout_id: int | None = None
    prompt: str | list[dict[str, str]] = ""
    tokens: list[int] = field(default_factory=list)
    multimodal_inputs: dict[str, Any] | None = None
    multimodal_train_inputs: dict[str, Any] | None = None
    multimodal_train_input_id: str | None = None
    apply_chat_template_kwargs: dict = field(default_factory=dict)
    response: str = ""
    response_length: int = 0
    label: str | None = None
    reward: float | dict[str, Any] | None = None
    loss_mask: list[int] | None = None
    weight_versions: list[str] = field(default_factory=list)
    rollout_log_probs: list[float] | None = None
    rollout_top_p_token_ids: list[int] | torch.Tensor | None = None
    rollout_top_p_token_offsets: list[int] | torch.Tensor | None = None
    rollout_routed_experts: list[list[int]] | torch.Tensor | None = None
    remove_sample: bool = False

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        FAILED = "failed"

    status: Status = Status.PENDING

    metadata: dict = field(default_factory=dict)
    generate_function_path: str | None = None
    custom_rm_path: str | None = None
    train_metadata: dict | None = None
    session_id: str | None = None
```

代码逻辑：
- prompt 侧支持文本或 chat message list。
- response 侧记录文本、token 长度、reward 和 loss mask。
- rollout 侧记录 logprob、top-p replay、routed experts 和 weight versions。
- metadata 和路径字段支持自定义生成/reward。
- session_id 支持路由一致性。

为什么这样写：
- Rollout 到训练的边界需要一个可序列化、可扩展的统一对象。
- 自定义函数路径保存在样本上，允许 per-sample 覆盖全局逻辑。

不变量与失败模式：
- `response_length` 应与 response-side token 数一致。
- `loss_mask`、`rollout_log_probs`、top-p offsets 等由校验函数维护长度一致。
- `reward` 可以是标量或 dict，读取时需要按配置处理。

Comment：
`Sample` 不是纯数据输入，它是 rollout 生成结果和训练消费之间的合同。

### 2.2 SpecInfo 与 PrefixCacheInfo 累加推理侧统计

问题与约束：
- speculative decoding 和 prefix cache 的统计来自 SGLang meta_info，可能在 partial rollout 中分段累计。

设计选择：
- `SpecInfo` 记录 accept/draft/verify/completion 计数，并提供 accept rate 和 accept length；`PrefixCacheInfo` 记录 cached tokens 和 total prompt tokens，并提供 hit rate。

Explain：
两个嵌套 dataclass 都提供 `to_dict/from_dict`，用于 Sample 序列化。

来源：slime/utils/types.py L153-L220

Code：

```python
@dataclass
class SpecInfo:
    spec_accept_token_num: int = 0
    spec_draft_token_num: int = 0
    spec_verify_ct: int = 0
    completion_token_num: int = 0

    @property
    def spec_accept_rate(self) -> float:
        return self.spec_accept_token_num / self.spec_draft_token_num if self.spec_draft_token_num > 0 else 0.0

    @property
    def spec_accept_length(self) -> float:
        return self.completion_token_num / self.spec_verify_ct if self.spec_verify_ct > 0 else 0.0

    def add(self, meta_info: dict):
        self.spec_accept_token_num += meta_info.get("spec_accept_token_num", 0)
        self.spec_draft_token_num += meta_info.get("spec_draft_token_num", 0)
        self.spec_verify_ct += meta_info.get("spec_verify_ct", 0)
        self.completion_token_num += meta_info.get("completion_tokens", 0)

@dataclass
class PrefixCacheInfo:
    cached_tokens: int = 0
    total_prompt_tokens: int = 0

    @property
    def prefix_cache_hit_rate(self) -> float:
        return self.cached_tokens / self.total_prompt_tokens if self.total_prompt_tokens > 0 else 0.0

    def add(self, meta_info: dict):
        self.cached_tokens += meta_info.get("cached_tokens", 0)
        self.total_prompt_tokens += meta_info.get("prompt_tokens", 0)
```

代码逻辑：
- SpecInfo 以 0 初始化所有计数。
- rate/length 分母为 0 时返回 0。
- add 从 meta_info 中按 key 累加。
- PrefixCacheInfo 用 prompt tokens 作为 hit rate 分母。
- 两个对象都挂在 Sample 上作为 default_factory。

为什么这样写：
- 推理统计应随 Sample 走到训练或评估日志阶段。
- 分段 rollout 场景要累加，而不是只保留最后一个 chunk 的 meta。

不变量与失败模式：
- meta_info 缺 key 时按 0 处理。
- 如果 SGLang meta 的 key 语义变化，统计会静默偏移。
- hit rate 和 accept rate 都只在分母大于 0 时有效。

Comment：
这些字段让 Sample 同时承担训练数据和 rollout 运行统计载体。

### 2.3 to_dict/from_dict 保留未知扩展字段

问题与约束：
- Sample 需要进入 Ray object store、checkpoint 或用户自定义扩展；新增字段不能让旧序列化格式立刻失效。

设计选择：
- `to_dict` 手动把 enum 和嵌套统计对象转成 dict；`from_dict` 先按 dataclass 字段构造 Sample，再把未知 key 设置回 sample 动态属性。

Explain：
`get_reward_value` 支持 reward 为标量或多 key dict；`args.reward_key` 非空时从 dict 中取子奖励。

来源：slime/utils/types.py L222-L247

Code：

```python
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
    data["prefix_cache_info"] = Sample.PrefixCacheInfo.from_dict(data.get("prefix_cache_info", {}))

    field_names = set(Sample.__dataclass_fields__.keys())
    init_data = {k: v for k, v in data.items() if k in field_names}
    sample = Sample(**init_data)

    for key, value in data.items():
        if key not in field_names:
            setattr(sample, key, value)

    return sample

def get_reward_value(self, args) -> float:
    return self.reward if not args.reward_key else self.reward[args.reward_key]
```

代码逻辑：
- 序列化时复制对象字段。
- enum status 写成字符串值。
- 嵌套统计对象转成 dict。
- 反序列化时恢复 enum 和嵌套对象。
- dataclass 已知字段用于构造函数。
- 未知字段作为动态属性恢复。

为什么这样写：
- 枚举和嵌套 dataclass 不能完全依赖默认 dict 化。
- 保留未知字段给插件、旧 checkpoint 和增量演进留空间。

不变量与失败模式：
- `data["status"]` 必须是合法 Status value。
- 多 reward 模式下 `args.reward_key` 必须存在于 reward dict。
- 动态字段不会进入 dataclass 字段列表，但会留在对象属性上。

Comment：
这段是 Sample 格式兼容性的关键，特别适合跨版本 checkpoint。

---

## 3. response 追加与 metadata 对齐

### 3.1 append_response_tokens 校验 trainable 与 log_probs 契约

问题与约束：
- PPO/训练侧需要模型生成 token 的 rollout log_probs；tool 或环境输出 token 不参与训练 loss，但仍会进入 response 序列。

设计选择：
- `append_response_tokens` 先把 tokens/log_probs 转成 Python list；log_probs 长度必须等于 tokens；trainable token 必须有 log_probs；non-trainable token 禁止传 log_probs，并自动补 0.0。

Explain：
文本 response 可通过 `text` 追加；token 序列、response_length、loss_mask 和 rollout_log_probs 会同步增长。

来源：slime/utils/types.py L253-L303

Code：

```python
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
        raise ValueError(f"log_probs length {len(log_probs)} != tokens length {len(tokens)}")
    if tokens and trainable and log_probs is None:
        raise ValueError("trainable response tokens require rollout log probabilities.")
    if tokens and not trainable:
        if log_probs is not None:
            raise ValueError("non-trainable response tokens should not pass rollout log probabilities.")
        log_probs = [0.0] * len(tokens)

    if text is not None:
        self.response += text

    previous_response_length = self.response_length
    if tokens:
        self.tokens += tokens
        self.response_length += len(tokens)
        if self.loss_mask is None:
            self.loss_mask = [1] * previous_response_length
        self.loss_mask += [1 if trainable else 0] * len(tokens)

    if log_probs is not None:
        if self.rollout_log_probs is None:
            if trainable and previous_response_length:
                raise ValueError(
                    "Cannot append trainable rollout log probabilities to a sample with existing response "
                    "tokens but no existing rollout_log_probs."
                )
            self.rollout_log_probs = [0.0] * previous_response_length
        self.rollout_log_probs += log_probs
```

代码逻辑：
- tokens 和 log_probs 先做类型规范化。
- 校验 log_probs 长度。
- trainable token 要求 log_probs。
- non-trainable token 自动生成 0.0 log_probs 和 loss mask 0。
- response 文本独立追加。
- tokens 增加时同步 response_length 和 loss_mask。
- rollout_log_probs 缺失时按已有 response 长度补 0.0。

为什么这样写：
- trainable token 的 logprob 是 PPO ratio/reference 计算的必要输入。
- tool token 要保留在上下文中，但不能贡献训练 loss。
- loss_mask 与 response token 数必须同长，训练 batch 才能正确切片。

不变量与失败模式：
- trainable token 无 log_probs 会失败。
- non-trainable token 带 log_probs 会失败。
- 已有 response token 但缺 rollout_log_probs 时，继续追加 trainable log_probs 会失败。

Comment：
`append_response_tokens` 是多轮/agent rollout 中最容易破坏训练契约的地方。

### 3.2 append_response_tokens 将 meta_info 交给 _apply_meta_info 并最终校验

问题与约束：
- response 追加可能同时带 SGLang meta_info；non-trainable token 没有 top-p meta，但如果样本已经启用 top-p replay，也必须补空 span。

设计选择：
- 当 `meta_info` 存在或需要 padding top-p 时调用 `_apply_meta_info`，传入本次 token 数、是否 padding top-p 和是否更新终态；最后统一调用 `_validate_response_metadata_lengths`。

Explain：
`should_pad_top_p = bool(tokens and not trainable)`，说明只有 non-trainable token 才会触发空 top-p span padding。

来源：slime/utils/types.py L304-L314

Code：

```python
should_pad_top_p = bool(tokens and not trainable)
if meta_info is not None or should_pad_top_p:
    self._apply_meta_info(
        args,
        meta_info or {},
        new_token_count=len(tokens),
        pad_missing_top_p=should_pad_top_p,
        update_terminal_info=update_terminal_info,
    )

self._validate_response_metadata_lengths()
```

代码逻辑：
- 根据 tokens 和 trainable 判断是否需要 top-p padding。
- meta_info 缺失时传空 dict。
- 把本次新增 token 数传给 `_apply_meta_info`。
- 每次追加结束都执行长度校验。

为什么这样写：
- metadata 对齐不应只在 terminal chunk 才检查。
- non-trainable token 不携带采样空间，但仍占 response 位置。

不变量与失败模式：
- `_apply_meta_info` 失败会阻断追加。
- 最终校验失败表示 Sample 已经进入不一致状态。

Comment：
这段保证追加 API 的每次调用都是一个小事务：更新后立即校验。

### 3.3 _apply_meta_info 合并或补齐 top-p replay

问题与约束：
- 每次 response chunk 都可能带一段 top-p replay；多段 top-p 需要合并，non-trainable token 需要空 span 补齐。

设计选择：
- `_apply_meta_info` 在有新增 token 时尝试提取 top-p 数据；第一次直接保存，后续调用 `_merge_rollout_top_p_token_data` 合并；如果需要 padding 且本次没有 top-p 数据，则调用 `_pad_rollout_top_p_offsets`。

Explain：
`applied_top_p_data` 防止同一次追加既合并真实 top-p，又再补空 span。

来源：slime/utils/types.py L316-L350

Code：

```python
def _apply_meta_info(
    self,
    args,
    meta_info: dict,
    *,
    new_token_count: int = 0,
    pad_missing_top_p: bool = False,
    update_terminal_info: bool = True,
) -> None:
    applied_top_p_data = False
    if new_token_count:
        top_p_data = _extract_rollout_top_p_token_data(meta_info, expected_num_tokens=new_token_count)
        if top_p_data is not None:
            applied_top_p_data = True
            base_token_ids, base_offsets = self.rollout_top_p_token_ids, self.rollout_top_p_token_offsets
            if base_token_ids is None and base_offsets is None:
                self.rollout_top_p_token_ids, self.rollout_top_p_token_offsets = top_p_data
            else:
                self.rollout_top_p_token_ids, self.rollout_top_p_token_offsets = _merge_rollout_top_p_token_data(
                    base_token_ids,
                    base_offsets,
                    *top_p_data,
                )

    if (
        pad_missing_top_p
        and new_token_count
        and self.rollout_top_p_token_offsets is not None
        and not applied_top_p_data
    ):
        self.rollout_top_p_token_ids, self.rollout_top_p_token_offsets = _pad_rollout_top_p_offsets(
            self.rollout_top_p_token_ids,
            self.rollout_top_p_token_offsets,
            new_token_count,
        )
```

代码逻辑：
- 初始化 `applied_top_p_data=False`。
- 有新增 token 时按 token 数提取 top-p。
- 第一次 top-p 直接保存。
- 后续 top-p 与旧数据合并。
- 需要 padding 且本次没有真实 top-p 时追加空 span。

为什么这样写：
- response 可能由多个 SGLang chunk 和 tool chunk 组成。
- top-p replay 一旦启用，就要保持 offsets 长度和 response_length 一致。

不变量与失败模式：
- top-p 数据的 offsets 长度必须匹配本次新增 token 数。
- padding 只有在已有 offsets 时才执行。
- 同一 chunk 不会同时真实合并和空 padding。

Comment：
这段让 top-p replay 能跨多轮 append 保持完整。

### 3.4 routed_experts、terminal status 与版本统计来自 meta_info

问题与约束：
- MoE routing replay 需要按 token、layer、topk reshape；终态信息只有在 SGLang 返回 finish_reason 时才能确定。

设计选择：
- `_apply_meta_info` 从 `routed_experts` 解码 int32 数组，并用 `len(tokens)-1, num_layers, moe_router_topk` reshape；如果没有 finish_reason 或禁用 terminal update，直接返回。否则累加 spec/prefix cache，记录 weight_version，并按 finish_reason type 更新 status。

Explain：
缺少 args 时不能 reshape routed experts，因此代码要求 args 存在。

来源：slime/utils/types.py L352-L381

Code：

```python
routed_experts = decode_int32_meta_array(meta_info, "routed_experts")
if routed_experts is not None:
    if args is None:
        raise ValueError("args is required to decode routed experts metadata.")
    self.rollout_routed_experts = routed_experts.reshape(
        len(self.tokens) - 1,
        args.num_layers,
        args.moe_router_topk,
    )

if not update_terminal_info or "finish_reason" not in meta_info:
    return

if getattr(args, "sglang_speculative_algorithm", False):
    self.spec_info.add(meta_info=meta_info)

self.prefix_cache_info.add(meta_info=meta_info)

if "weight_version" in meta_info:
    self.weight_versions.append(meta_info["weight_version"])

match meta_info["finish_reason"]["type"]:
    case "length":
        self.status = Sample.Status.TRUNCATED
    case "abort":
        self.status = Sample.Status.ABORTED
    case "stop":
        self.status = Sample.Status.COMPLETED
```

代码逻辑：
- 尝试解码 routed experts。
- routed experts 存在时要求 args。
- 按 response token 数、层数、topk reshape。
- 没有 terminal 信息时退出。
- speculative 启用时累加 spec info。
- 始终累加 prefix cache info。
- 记录 weight version。
- 根据 finish reason 更新 status。

为什么这样写：
- routed expert replay 的 shape 依赖训练配置，不能仅凭 meta 自描述。
- partial rollout 中不是每个 chunk 都应该更新终态。

不变量与失败模式：
- routed experts 元素数必须等于 `(len(tokens)-1) * num_layers * topk`。
- `finish_reason["type"]` 未覆盖的新类型不会改变 status。
- args 为 None 且 meta 包含 routed experts 会失败。

Comment：
这段把 SGLang 推理侧 meta 归并到训练样本状态。

### 3.5 _validate_response_metadata_lengths 是 Sample 的最后防线

问题与约束：
- loss mask、rollout logprobs、top-p offsets 如果与 response_length 不一致，训练 batch 会在后续阶段才报错，定位成本高。

设计选择：
- 每次 append 后调用 `_validate_response_metadata_lengths`；检查 loss_mask 和 rollout_log_probs 长度等于 response_length；top-p ids/offsets 必须同时存在，offsets 长度等于 response_length + 1，且末尾 offset 等于 ids 数量。

Explain：
如果 top-p replay 完全不存在，校验直接返回；只存在 ids 或 offsets 之一则失败。

来源：slime/utils/types.py L383-L409

Code：

```python
def _validate_response_metadata_lengths(self):
    if self.loss_mask is not None and len(self.loss_mask) != self.response_length:
        raise ValueError(f"loss_mask length {len(self.loss_mask)} != response_length {self.response_length}")

    if self.rollout_log_probs is not None and len(self.rollout_log_probs) != self.response_length:
        raise ValueError(
            f"rollout_log_probs length {len(self.rollout_log_probs)} != response_length {self.response_length}"
        )

    if self.rollout_top_p_token_ids is None and self.rollout_top_p_token_offsets is None:
        return
    if self.rollout_top_p_token_ids is None or self.rollout_top_p_token_offsets is None:
        raise ValueError("rollout top-p replay must include both token ids and offsets.")

    offsets = torch.as_tensor(self.rollout_top_p_token_offsets, dtype=torch.int32).reshape(-1)
    if offsets.numel() != self.response_length + 1:
        raise ValueError(
            "rollout_top_p_token_offsets length must equal response_length + 1: "
            f"len(offsets)={offsets.numel()}, response_length={self.response_length}."
        )
    token_id_count = _numel(self.rollout_top_p_token_ids)
    if int(offsets[-1]) != token_id_count:
        raise ValueError(
            "rollout top-p token ids/offsets mismatch: "
            f"offsets[-1]={int(offsets[-1])}, len(token_ids)={token_id_count}."
        )
```

代码逻辑：
- 校验 loss mask 长度。
- 校验 rollout logprob 长度。
- top-p 完全缺失时跳过。
- top-p 半缺失时失败。
- offsets 长度必须是 response_length + 1。
- offsets 末尾必须等于 token ids 数量。

为什么这样写：
- 这些错误越早报出，越接近产生不一致的 append 调用点。
- top-p replay 是 ragged 数据，单靠 Python list 长度无法保证正确。

不变量与失败模式：
- response_length 是所有 response-side metadata 的基准长度。
- top-p offsets 和 token ids 必须形成合法 CSR。

Comment：
Sample 契约的核心就是这些长度不变量。

---

## 4. 其他跨模块契约

### 4.1 ParamInfo 和 RolloutBatch 说明 types.py 不只服务 Sample

问题与约束：
- 同一个 types 模块还承载权重同步和 rollout batch 的共享类型；这些类型会被后端工具复用。

设计选择：
- `ParamInfo` 用 frozen dataclass 描述单个权重 tensor 的 name、dtype、shape、attrs、size 和 src_rank；`RolloutBatch` 是 rollout 到训练路径上的 dict-based batch 类型别名。

Explain：
`RolloutBatch` 的注释说明在 Megatron backend 中，多个字段会在消费前转换成 GPU 上的 torch.Tensor list。

来源：slime/utils/types.py L411-L424

Code：

```python
@dataclass(frozen=True)
class ParamInfo:
    name: str
    dtype: torch.dtype
    shape: torch.Size
    attrs: dict
    size: int
    src_rank: int


RolloutBatch = dict[str, list[torch.Tensor] | list[int] | list[float] | list[str]]
```

代码逻辑：
- ParamInfo 不可变。
- dtype 和 shape 保留 torch 类型。
- attrs 存放额外参数属性。
- src_rank 标识权重来源 rank。
- RolloutBatch 允许 tensor/int/float/string list。

为什么这样写：
- 权重同步需要在多个 rank 间交换 tensor 元数据，frozen dataclass 更容易安全传递。
- rollout batch 是后端边界类型，保持 dict 便于动态字段扩展。

不变量与失败模式：
- ParamInfo 的 shape/dtype 必须与实际 tensor 一致。
- RolloutBatch 的 key 语义由下游 consumer 约定，类型别名不做运行时校验。

Comment：
Sample 契约之外，types.py 也承担训练/权重同步的共享类型定义。

### 4.2 RolloutFnTrainOutput 与 RolloutFnEvalOutput 区分 train/eval 返回

问题与约束：
- 用户自定义 rollout 函数可能返回旧格式；训练和评估返回结构不同，调用方需要统一封装。

设计选择：
- `RolloutFnTrainOutput` 包含嵌套 Sample 列表和 metrics；`RolloutFnEvalOutput` 包含 dataset 到 metrics/data 的嵌套 dict。`call_rollout_fn` 对 legacy 返回值做兼容包装。

Explain：
evaluation=True 时，非 dataclass 输出会被包装成 `RolloutFnEvalOutput(data=output)`；否则包装成 `RolloutFnTrainOutput(samples=output)`。

来源：slime/rollout/base_types.py L7-L26

Code：

```python
@dataclass
class RolloutFnTrainOutput:
    samples: list[list[Sample]]
    metrics: dict[str, Any] = None


@dataclass
class RolloutFnEvalOutput:
    data: dict[str, dict[str, Any]]
    metrics: dict[str, Any] = None


def call_rollout_fn(fn, *args, evaluation: bool, **kwargs):
    output = fn(*args, **kwargs, evaluation=evaluation)

    if not isinstance(output, (RolloutFnTrainOutput, RolloutFnEvalOutput)):
        output = RolloutFnEvalOutput(data=output) if evaluation else RolloutFnTrainOutput(samples=output)

    return output
```

代码逻辑：
- 定义训练返回结构。
- 定义评估返回结构。
- 调用用户 rollout 函数时传入 evaluation。
- 如果返回值不是标准输出类型，按 evaluation 分支包装旧格式。

为什么这样写：
- 训练路径需要 Sample，而评估路径通常只需要数据集级结果和指标。
- 兼容旧 rollout 函数降低升级成本。

不变量与失败模式：
- evaluation 参数必须正确传给用户函数。
- legacy eval 输出需要符合 `RolloutFnEvalOutput.data` 的预期结构。
- legacy train 输出需要是嵌套 Sample 列表。

Comment：
这是 Sample 契约外层的 rollout 函数返回契约。

---

## 5. 走读小结

```text
SGLang meta_info
  -> decode_int32_meta_array
  -> top-p replay / routed experts

Sample.append_response_tokens
  -> tokens / response_length / loss_mask / rollout_log_probs
  -> _apply_meta_info
  -> _validate_response_metadata_lengths

rollout function
  -> RolloutFnTrainOutput(samples)
  -> RolloutFnEvalOutput(data)
```

**下一专题关联：** Agent 多轮与 tool token 见 [[27-Agent-Trajectory-00-MOC]]；路由 replay 见 [[23-CP-RoutingReplay-00-MOC]]；权重同步辅助类型见 [[24-WeightSync-Dist-00-MOC]]。
