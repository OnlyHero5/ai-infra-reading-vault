---
type: batch-doc
module: 12-SGLang-Rollout
batch: "12"
doc_type: faq
title: "SGLang Rollout · 关键问题"
tags:
  - slime/batch/12
  - slime/module/sglang-rollout
  - slime/doc/faq
updated: 2026-07-02
---

# SGLang Rollout · 关键问题

> FAQ、易错点、测试锚点。验证用例主要来自 `tests/test_rollout_metrics.py`。

---

## Q1：`--rollout-function-path` 和 `--custom-generate-function-path` 怎么选？

**Explain：** 前者替换整段 rollout orchestration（oversampling、filter、abort）；后者只替换「单条 sample 如何变成 response」这一步，保留默认 Slime 批调度逻辑。

| 需求 | 推荐 |
|------|------|
| 多轮 agent、tool call、自定义 HTTP 目标 | `--custom-generate-function-path` |
| fully-async、SFT、从磁盘 load rollout | `--rollout-function-path` 指向专用模块 |
| eval 单数据集特殊 generate | `EvalDatasetConfig.custom_generate_function_path` |

**Code（CLI 定义）：**

```python
# 来源：slime/slime/utils/arguments.py L473-L480
            parser.add_argument(
                "--custom-generate-function-path",
                type=str,
                default=None,
                help=(
                    "Only substitue the `def generate(args, sample, sampling_params)` function within the example rollout function. "
                    "This should be useful if you need to implement some special rollout logic, e.g. multi-turn, function calling."
                ),
            )
```

**易错：** 把 agent 逻辑写进全新 `rollout-function-path` 却漏实现 oversampling / abort，训练会 hang 或 batch 不足。

---

## Q2：custom generate 函数必须满足什么契约？

**Explain：** 须为 **async** 函数；签名 `(args, sample, sampling_params)` 或带 `evaluation`。返回值可以是单个 `Sample` 或 `list[Sample]`（fan-out）。若自行填充 `reward`，`generate_and_rm` 会跳过 `async_rm`（当 `sample.reward is not None`）。

**Code（参考实现）：**

```python
# 来源：tests/plugin_contracts/test_plugin_generate_contracts.py L71-L77
async def custom_generate(args, sample: Sample, sampling_params: dict):
    sample.tokens = [11, 12, 13]
    sample.response = "generated"
    sample.response_length = len(sample.tokens)
    sample.reward = 0.25
    sample.status = Sample.Status.COMPLETED
    return sample
```

**易错 vs 正确：**

```python
# ❌ 同步函数 — load_function 后 await 会失败
def custom_generate(args, sample, sampling_params):
    ...

# ✅ async + 设置 status
async def custom_generate(args, sample, sampling_params):
    sample.status = Sample.Status.COMPLETED
    return sample
```

**验证：** `tests/plugin_contracts/test_plugin_generate_contracts.py` 中 `test_custom_generate_function_path_supports_user_override`。

---

## Q3：为何 `GenerateState` 用 Singleton？多 rollout 会串状态吗？

**Explain：** 同一 Ray rollout worker 进程串行处理多个 `rollout_id`；Singleton 共享 tokenizer/processor（昂贵）与 semaphore。每次 `generate_rollout_async` 结束调用 `state.reset()` 清空 `pendings`、`remaining_batch_size`、`aborted`。

**易错：** 在 custom generate 里缓存跨 rollout 的可变全局列表而不清理 → 内存泄漏或 index 错乱。应写入 `Sample.metadata` 或 DataSource buffer。

---

## Q4：top-p replay 与 `test_rollout_metrics.py` 的关系？

**Explain：** 当 `rollout_top_p != 1.0`，`GenerateState` 向 SGLang 请求 `return_top_p_token_ids`。响应经 base64 int32 解码后由 `Sample.append_response_tokens` 合并到 `rollout_top_p_token_ids/offsets`。RolloutManager 的 `_compute_top_p_kept_vocab_metrics` 读取这些字段计算 `top_p_kept_vocab_per_token`。

**触发条件（generate 侧）：**

```python
# 来源：slime/slime/rollout/sglang_rollout.py L107-L108
        if args.rollout_top_p != 1.0:
            self.sampling_params["custom_params"] = {"return_top_p_token_ids": True}
```

**测试：`test_append_response_tokens_merges_top_p_tensors`**

**Explain：** 验证多段 append 时 top-p ragged tensor 正确 merge——custom multi-turn generate 必须走 `append_response_tokens` 而非手写 tensor。

```python
# 来源：tests/test_rollout_metrics.py L67-L94
def test_append_response_tokens_merges_top_p_tensors():
    sample = Sample(
        tokens=[0, 1],
        response_length=1,
        loss_mask=[1],
        rollout_log_probs=[-0.3],
        rollout_top_p_token_ids=torch.tensor([1], dtype=torch.int32),
        rollout_top_p_token_offsets=torch.tensor([0, 1], dtype=torch.int32),
    )

    sample.append_response_tokens(
        _make_args(),
        tokens=[10, 20],
        log_probs=[-0.1, -0.2],
        trainable=True,
        meta_info={
            "top_p_token_ids": _b64_int32([10, 11, 20]),
            "top_p_token_offsets": _b64_int32([0, 2, 3]),
            "finish_reason": {"type": "stop"},
        },
    )

    torch.testing.assert_close(sample.rollout_top_p_token_ids, torch.tensor([1, 10, 11, 20], dtype=torch.int32))
    torch.testing.assert_close(sample.rollout_top_p_token_offsets, torch.tensor([0, 1, 3, 4], dtype=torch.int32))
```

**测试：`test_top_p_kept_vocab_metric_uses_loss_mask`**

**Explain：** metric 只统计 `loss_mask==1` 的 token 的 kept vocab 宽度；agent 场景 tool token（mask=0）不应抬高 metric。

```python
# 来源：tests/test_rollout_metrics.py L20-L36
def test_top_p_kept_vocab_metric_uses_loss_mask():
    samples = [
        Sample(
            response_length=4,
            loss_mask=torch.tensor([1, 0, 1, 0], dtype=torch.int32),
            rollout_top_p_token_offsets=torch.tensor([0, 3, 8, 10, 20], dtype=torch.int32),
        ),
        Sample(
            response_length=2,
            loss_mask=None,
            rollout_top_p_token_offsets=torch.tensor([0, 4, 9], dtype=torch.int32),
        ),
    ]

    metrics = _compute_top_p_kept_vocab_metrics(None, samples)

    assert metrics["top_p_kept_vocab_per_token"] == pytest.approx(3.5)
```

**测试：`test_top_p_kept_vocab_metric_skips_removed_samples`**

```python
# 来源：tests/test_rollout_metrics.py L40-L50
def test_top_p_kept_vocab_metric_skips_removed_samples():
    samples = [
        Sample(
            response_length=3,
            loss_mask=[1, 1, 1],
            remove_sample=True,
            rollout_top_p_token_offsets=torch.tensor([0, 2, 4, 6], dtype=torch.int32),
        )
    ]

    assert _compute_top_p_kept_vocab_metrics(None, samples) == {}
```

---

## Q5：trainable vs non-trainable token 如何 append？

**Explain：** 模型生成 token 用 `trainable=True` + log_probs + meta_info；tool/环境 token 用 `trainable=False`，loss_mask 置 0，top-p offsets  padding。

**测试：`test_append_response_tokens_pads_top_p_for_non_trainable_tokens`**

```python
# 来源：tests/test_rollout_metrics.py L150-L167
def test_append_response_tokens_pads_top_p_for_non_trainable_tokens():
    sample = Sample(
        tokens=[0, 1],
        response_length=1,
        loss_mask=[1],
        rollout_log_probs=[-0.1],
        rollout_top_p_token_ids=torch.tensor([10, 11], dtype=torch.int32),
        rollout_top_p_token_offsets=torch.tensor([0, 2], dtype=torch.int32),
    )

    sample.append_response_tokens(tokens=[200, 201, 202], trainable=False)

    assert sample.loss_mask == [1, 0, 0, 0]
    assert sample.rollout_log_probs == [-0.1, 0.0, 0.0, 0.0]
    torch.testing.assert_close(sample.rollout_top_p_token_offsets, torch.tensor([0, 2, 2, 2, 2], dtype=torch.int32))
```

**易错 vs 正确：**

```python
# ❌ trainable=True 但不传 log_probs
sample.append_response_tokens(tokens=[10], trainable=True)  # ValueError

# ✅ non-trainable 不传 log_probs
sample.append_response_tokens(tokens=[10], trainable=False)
```

**测试：** `test_append_response_tokens_requires_trainable_log_probs`、`test_append_response_tokens_rejects_non_trainable_log_probs`。

---

## Q6：MoE routing replay 如何解码？

**测试：`test_append_response_tokens_decodes_routed_experts`**

```python
# 来源：tests/test_rollout_metrics.py L129-L146
def test_append_response_tokens_decodes_routed_experts():
    sample = Sample(tokens=[101, 102, 103])

    sample.append_response_tokens(
        _make_args(),
        tokens=[],
        trainable=True,
        meta_info={
            "routed_experts": _b64_int32([0, 1, 2, 3, 4, 5, 6, 7]),
            "finish_reason": {"type": "stop"},
        },
    )

    assert sample.rollout_routed_experts.shape == (2, 2, 2)
```

**Explain：** 需在 args 开启 `use_rollout_routing_replay`，且 generate payload 带 `return_routed_experts=True`（见 [[12-SGLang-Rollout-02-源码走读]] §5.2）。

---

## Q7：streaming / multi-chunk 生成如何保持 PENDING？

**测试：`test_append_response_tokens_can_skip_terminal_status_for_streaming_chunks`**

```python
# 来源：tests/test_rollout_metrics.py L98-L121
    sample.append_response_tokens(
        _make_args(),
        tokens=[10, 20],
        log_probs=[-0.1, -0.2],
        trainable=True,
        meta_info={"finish_reason": {"type": "stop"}, ...},
        update_terminal_info=False,
    )

    assert sample.status is Sample.Status.PENDING
```

**Explain：** custom generate 多轮循环中，中间 chunk 应 `update_terminal_info=False`，最后一轮再让 status 变 COMPLETED。

---

## Q8：abort 后 partial sample 去哪了？

**Explain：** `partial_rollout=True` 时，`abort` drain pending tasks，将有 `response` 的 sample 附 `metadata["start_rollout_id"]=rollout_id`，返回 `aborted_samples`；`generate_rollout` 调用 `data_source.add_samples` 回灌。

**易错：** 未开 partial_rollout 时 abort 直接丢弃 pending，无法续写。

---

## Q9：为何 oversampling 后 `len(data)` 断言严格等于 `rollout_batch_size`？

**Explain：** 主循环直到 `len(data) == rollout_batch_size` 才退出；dynamic filter 只影响是否 append 到 `data`，不影响继续 oversample。若 DataSource 枯竭且 filter 过严，会 infinite loop——需保证 buffer 足够或调 filter。

---

## Q10：如何本地验证本模块相关测试？

```bash
cd F:/源码阅读/slime
pytest tests/test_rollout_metrics.py -v
pytest tests/plugin_contracts/test_plugin_generate_contracts.py -v -k custom_generate
```

**说明：** `test_rollout_metrics.py` 覆盖 Sample 侧 metrics 契约（与 `generate` → `append_response_tokens` 输出对齐）；plugin contracts 覆盖 `--custom-generate-function-path` 挂载行为。
