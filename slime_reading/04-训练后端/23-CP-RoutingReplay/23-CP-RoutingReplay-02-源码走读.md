---
type: batch-doc
module: 23-CP-RoutingReplay
batch: "23"
doc_type: walkthrough
title: "CP · Routing Replay · 源码走读"
tags:
  - slime/batch/23
  - slime/module/cp-routing-replay
  - slime/doc/walkthrough
updated: 2026-07-02
---

# CP · Routing Replay · 源码走读

> 文件：`cp_utils.py`（~345 行）、`routing_replay.py`（~94 行）  
> 基线 commit `22cdc6e1`

---

## 1. get_logits_and_tokens_offset_with_cp

**Explain：** 从 prompt 起点计量 offset；空 chunk 时用 `(0,0)` 切片保梯度。

**Code：**

```python
## 来源：cp_utils.py L9-L44
def get_logits_and_tokens_offset_with_cp(total_length: int, response_length: int):
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()
    assert cp_size > 1

    prompt_length = total_length - response_length
    chunk_size = (total_length + 2 * cp_size - 1) // (2 * cp_size)

    chunk_0 = (cp_rank * chunk_size, (cp_rank + 1) * chunk_size)
    chunk_1 = ((2 * cp_size - cp_rank - 1) * chunk_size, (2 * cp_size - cp_rank) * chunk_size)

    logits_0 = (max(chunk_0[0], prompt_length - 1), min(chunk_0[1], total_length - 1))
    logits_1 = (max(chunk_1[0], prompt_length - 1), min(chunk_1[1], total_length - 1))
    ...
    return chunk_size, (chunk_0, chunk_1), (logits_0, logits_1), (token_0, token_1)
```

---

## 2. slice_with_cp

**Code：**

```python
## 来源：cp_utils.py L287-L317
def slice_with_cp(tokens: torch.Tensor, pad_value):
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size == 1:
        return tokens
    token_len = len(tokens)
    chunk_size = (token_len + 2 * cp_size - 1) // (2 * cp_size)
    pad = 2 * cp_size * chunk_size - token_len
    tokens = pad_tokens(tokens, pad)
    start_1, end_1 = chunk_size * cp_rank, chunk_size * (cp_rank + 1)
    start_2, end_2 = chunk_size * (2 * cp_size - cp_rank - 1), chunk_size * (2 * cp_size - cp_rank)
    return torch.cat([tokens[start_1:end_1], tokens[start_2:end_2]])
```

**Comment：** `get_batch` 对 tokens 与 loss_mask 均调用；pad_value 对 token 为 0，对 mask 为 0。

---

## 3. slice_log_prob_with_cp

**Explain：** 输入已是 **response 长度** 的 logprob；按 logits_offset 从 response 局部坐标切片两段再拼接。

**Code：**

```python
## 来源：cp_utils.py L320-L344
def slice_log_prob_with_cp(log_prob, total_length, response_length):
    assert len(log_prob) == response_length
    if cp_size == 1:
        return log_prob
    prompt_length = total_length - response_length
    _, _, logits_offset, _ = get_logits_and_tokens_offset_with_cp(total_length, response_length)
    chunk_1 = log_prob[logits_offset[0][0] - (prompt_length - 1) : logits_offset[0][1] - (prompt_length - 1)]
    chunk_2 = log_prob[logits_offset[1][0] - (prompt_length - 1) : logits_offset[1][1] - (prompt_length - 1)]
    return torch.cat([chunk_1, chunk_2], dim=0)
```

---

## 4. all_gather_with_cp

**Explain：** 构造 full `[response_length]` 张量，非负责区间填 zero（`requires_grad=True`），再 CP group `all_reduce` 求和合并。

**Code：**

```python
## 来源：cp_utils.py L235-L284
def all_gather_with_cp(tensor: torch.Tensor, total_length: int, response_length: int) -> torch.Tensor:
    cp_group = mpu.get_context_parallel_group()
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size == 1:
        return tensor
    _, _, logits_offset, _ = get_logits_and_tokens_offset_with_cp(total_length, response_length)
    prompt_length = total_length - response_length
    chunk_0 = tensor[: logits_offset[0][1] - logits_offset[0][0]]
    chunk_1 = tensor[logits_offset[0][1] - logits_offset[0][0] :]
    ...
    full_tensor = torch.cat([left, chunk_0, mid, chunk_1, right], dim=0)
    assert full_tensor.shape[0] == response_length
    full_tensor = dist.nn.all_reduce(full_tensor, group=cp_group)
    return full_tensor
```

---

## 5. get_sum_of_sample_mean（CP 分支）

**Code：**

```python
## 来源：cp_utils.py L91-L124
    else:
        cp_chunk_lengths: list[int] = []
        chunked_loss_masks: list[torch.Tensor] = []
        for total_length, response_length, loss_mask in zip(total_lengths, response_lengths, loss_masks, strict=False):
            prompt_length = total_length - response_length
            _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(total_length, response_length)
            loss_mask_0 = loss_mask[tokens_offset[0][0] - prompt_length : tokens_offset[0][1] - prompt_length]
            loss_mask_1 = loss_mask[tokens_offset[1][0] - prompt_length : tokens_offset[1][1] - prompt_length]
            chunked_loss_mask = torch.cat([loss_mask_0, loss_mask_1], dim=0)
            chunked_loss_masks.append(chunked_loss_mask)
            cp_chunk_lengths.append(chunked_loss_mask.size(0))

        def sum_of_sample_mean(x: torch.Tensor) -> torch.Tensor:
            return sum(
                (x_i * chunked_loss_mask).sum() / torch.clamp_min(denom, 1)
                for x_i, chunked_loss_mask, denom in zip(
                    x.split(cp_chunk_lengths, dim=0), chunked_loss_masks, sample_denoms, strict=False
                )
            )
```

---

## 6. reduce_train_step_metrics

**Code：**

```python
## 来源：cp_utils.py L127-L168
def reduce_train_step_metrics(losses_reduced, *, calculate_per_token_loss, step_global_batch_size, cp_size, dp_with_cp_group):
    keys = losses_reduced[0]["keys"]
    values = None
    for x in losses_reduced:
        values = x["values"] if values is None else values + x["values"]
    dist.all_reduce(values, group=dp_with_cp_group)
    values = values.tolist()
    if calculate_per_token_loss:
        num_samples_or_tokens = values[0]
        cp_factor = cp_size
    else:
        num_samples_or_tokens = step_global_batch_size
        cp_factor = 1
    return {key: value * cp_factor / num_samples_or_tokens for key, value in zip(keys, values[1:], strict=False)}
```

---

## 7. rollout_log_metric_contribution

**Code：**

```python
## 来源：cp_utils.py L171-L194
def rollout_log_metric_contribution(per_rank_reducer_sum, *, cp_size, num_rollouts_in_rollout, dp_size):
    sum_value = cp_size * per_rank_reducer_sum
    count = num_rollouts_in_rollout / dp_size
    return sum_value, count
```

**Comment：** 与 [[20-Train-Data-02-源码走读]] `log_rollout_data` 配对，使 W&B rollout 指标与 train 同空间。

---

## 8. gather_and_reduce_log_dict

**Code：**

```python
## 来源：cp_utils.py L197-L232
def gather_and_reduce_log_dict(log_dict, *, dp_size, dp_src_rank, dp_group):
    if dist.get_rank() == dp_src_rank:
        gathered = [None] * dp_size
        dist.gather_object(log_dict, gathered, dst=dp_src_rank, group=dp_group)
        reduced = {}
        for key in log_dict:
            values = [d[key] for d in gathered]
            first = values[0]
            if isinstance(first, tuple) and len(first) == 2:
                total_sum = sum(v[0] for v in values)
                total_count = sum(v[1] for v in values)
                reduced[key] = total_sum / total_count if total_count else 0.0
            else:
                reduced[key] = sum(values) / dp_size
        return reduced
    dist.gather_object(log_dict, None, dst=dp_src_rank, group=dp_group)
    return None
```

---

## 9. RoutingReplay 类

**Explain：** 每层 MoE 一个实例；`top_indices` pin_memory 到 CPU 省 GPU；forward/backward 各维护独立 index。

**Code：**

```python
## 来源：routing_replay.py L13-L54
class RoutingReplay:
    all_routing_replays = []

    def __init__(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []
        RoutingReplay.all_routing_replays.append(self)

    def record(self, top_indices):
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        buf.copy_(top_indices)
        self.top_indices_list.append(buf)

    def pop_forward(self):
        top_indices = self.top_indices_list[self.forward_index]
        self.forward_index += 1
        return top_indices.to(torch.cuda.current_device())

    def pop_backward(self):
        top_indices = self.top_indices_list[self.backward_index]
        self.backward_index += 1
        return top_indices.to(torch.cuda.current_device())
```

---

## 10. get_routing_replay_compute_topk

**Code：**

```python
## 来源：routing_replay.py L57-L82
def get_routing_replay_compute_topk(old_compute_topk):
    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            routing_replay_stage = os.environ["ROUTING_REPLAY_STAGE"]
            if routing_replay_stage == "fallthrough":
                return old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
            if routing_replay_stage == "record":
                probs, top_indices = old_compute_topk(...)
                ROUTING_REPLAY.record(top_indices)
            elif routing_replay_stage == "replay_forward":
                top_indices = ROUTING_REPLAY.pop_forward()
                probs = scores.gather(1, top_indices)
            elif routing_replay_stage == "replay_backward":
                top_indices = ROUTING_REPLAY.pop_backward()
                probs = scores.gather(1, top_indices)
            return probs, top_indices
        else:
            return old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
    return compute_topk
```

---

## 11. register_routing_replay

**Code：**

```python
## 来源：routing_replay.py L85-L93
def register_routing_replay(module):
    if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
        module.routing_replay = RoutingReplay()

        def pre_forward_hook(*args, **kwargs):
            set_routing_replay(module.routing_replay)

        module.register_forward_pre_hook(pre_forward_hook)
```

**Comment：** `ROUTING_REPLAY` 全局指针在每次 layer forward 前指向当前 module 的 replay 实例。

---

## 走读小结

| 模块 | 训练路径作用 |
|------|-------------|
| `slice_with_cp` | `get_batch` token/mask 切分 |
| `all_gather_with_cp` | GAE/GSPO/values 全序列还原 |
| `get_sum_of_sample_mean` | loss/metrics CP 正确平均 |
| `RoutingReplay` | MoE expert 路径确定性 |

---

## 12. legacy per-sample mean（sample_denoms=None）

**Explain：** 未传 `rollout_mask_sums` 时退化为每样本 `loss_mask.sum()` 作分母；与 per-rollout mean 不混用。

**Code：**

```python
## 来源：cp_utils.py L67-L80
    if sample_denoms is None:
        sample_denoms = [m.sum() for m in loss_masks]
    if cp_size == 1:
        def sum_of_sample_mean(x: torch.Tensor) -> torch.Tensor:
            return sum(
                (x_i * loss_mask_i).sum() / torch.clamp_min(denom, 1)
                for x_i, loss_mask_i, denom in zip(x.split(response_lengths, dim=0), loss_masks, sample_denoms, strict=False)
            )
```

---

## 13. sum_of_token 模式

**Explain：** `calculate_per_token_loss=True` 时返回 token 求和函数，不做除法；配合 `reduce_train_step_metrics` 的 cp_factor。

**Code：**

```python
## 来源：cp_utils.py L114-L124
        def sum_of_token(x: torch.Tensor) -> torch.Tensor:
            return sum(
                (x_i * chunked_loss_mask).sum()
                for x_i, chunked_loss_mask in zip(x.split(cp_chunk_lengths, dim=0), chunked_loss_masks, strict=False)
            )
    return sum_of_sample_mean if not calculate_per_token_loss else sum_of_token
```

---

## 14. all_gather_with_cp 仅 chunk_0 分支

**Explain：** 当 rank 只持有一段有效 logits（chunk_1 空），仍用 left/right zero pad 拼满 response_length。

**Code：**

```python
## 来源：cp_utils.py L266-L270
    elif chunk_0.shape[0] != 0 and chunk_1.shape[0] == 0:
        left = zero(logits_offset[0][0] - (prompt_length - 1))
        right = zero(total_length - 1 - logits_offset[0][1])
        full_tensor = torch.cat([left, chunk_0, right], dim=0)
```

---

## 15. chunk_size 与 prompt 边界

**Explain：** `chunk_size = ceil(total_length / (2*cp_size))` 保证 zigzag 覆盖；logits 区间 clamp 到 `[prompt_length-1, total_length-1)`。

**Code：**

```python
## 来源：cp_utils.py L20-L29
    prompt_length = total_length - response_length
    chunk_size = (total_length + 2 * cp_size - 1) // (2 * cp_size)
    logits_0 = (max(chunk_0[0], prompt_length - 1), min(chunk_0[1], total_length - 1))
```

---

## 16. gather_and_reduce 纯标量 legacy 路径

**Explain：** 非 `(sum,count)`  tuple 时 `Σ/dp_size` — 仅当各 rank 持有相同统计才正确。

**Code：**

```python
## 来源：cp_utils.py L228-L229
            else:
                reduced[key] = sum(values) / dp_size
```

---

## 17. RoutingReplay.clear_all_forward

**Explain：** logprob forward 后重置 forward_index，backward 仍从 0 pop_backward；forward/backward 列表长度相同（每层一次 record/replay）。

**Code：**

```python
## 来源：routing_replay.py L43-L54
    def clear_forward(self):
        self.forward_index = 0

    @staticmethod
    def clear_all_forward():
        for replay in RoutingReplay.all_routing_replays:
            replay.clear_forward()
```

---

## 18. replay_forward 形状断言

**Explain：** gather 的 top_indices 必须与 scores `[batch, topk]` 对齐，否则 MoE 层 token 错位。

**Code：**

```python
## 来源：routing_replay.py L68-L71
                assert (
                    top_indices.shape[0] == scores.shape[0] and top_indices.shape[1] == topk
                ), f"top_indices shape {top_indices.shape} does not match scores shape {scores.shape} and topk {topk}"
```

---

## 19. set_routing_replay 全局指针

**Explain：** Megatron patch 在 layer forward 前 hook 设置；`compute_topk` wrapper 读 `ROUTING_REPLAY` 全局。

**Code：**

```python
## 来源：routing_replay.py L7-L10
def set_routing_replay(replay):
    global ROUTING_REPLAY
    ROUTING_REPLAY = replay
```

---

## 20. `train_one_step` 内 routing replay stage 切换

**Explain：** 与 [[18-Model-Init-02-源码走读]] §7 的 `forward_only` **不同**——`forward_only` 的 `forward_step` **不**改 `ROUTING_REPLAY_STAGE`，logprob 阶段由 [[19-Train-Step-02-源码走读]] §5 的 `train_actor` 在调用 `compute_log_prob` 前设置。Policy **train** 走 `train_one_step`：actor 在 `train()` 前设 `replay_backward`，而 `forward_step` 闭包在 **forward 段**临时切 `replay_forward`、结束后恢复，使 Megatron 1F1B backward 仍走 `pop_backward`。

**与 `fill_routing_replay` 的衔接（[[19-Train-Step-02-源码走读]] §5）：**

| 时机 | 谁设置 stage | `top_indices_list` 来源 |
|------|-------------|------------------------|
| `use_rollout_routing_replay` | actor 设 `replay_forward` → `forward_only` logprob | `fill_routing_replay` 预 `record()` rollout experts |
| logprob 后 | `RoutingReplay.clear_all_forward()` | forward_index 归零，列表保留 |
| `train()` 前 | actor 设 `replay_backward` | — |
| `train_one_step` forward | 闭包内临时 `replay_forward` | `pop_forward()` 再消费同一列表 |
| backward | 恢复为 `replay_backward` | `pop_backward()` |

**Code：**

```python
## 来源：model.py L576-L638（train_one_step.forward_step 闭包）
        batch = get_batch(
            data_iterator,
            _with_rollout_top_p_token_keys(
                args,
                [
                    "tokens",
                    "multimodal_train_inputs",
                    "packed_seq_params",
                    "total_lengths",
                    "response_lengths",
                    "loss_masks",
                    "log_probs",
                    "ref_log_probs",
                    "values",
                    "advantages",
                    "returns",
                    "rollout_log_probs",
                    "teacher_log_probs",
                    "rollout_mask_sums",
                ],
            ),
            args.data_pad_size_multiplier,
            args.allgather_cp,
        )

        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            old_stage = os.environ["ROUTING_REPLAY_STAGE"]
            os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"

        if return_schedule_plan:
            ...
            output_tensor = model.build_schedule_plan(...)
        else:
            forward_kwargs = {
                "input_ids": batch["tokens"],
                "position_ids": None,
                "attention_mask": None,
                "labels": None,
                "packed_seq_params": batch["packed_seq_params"],
                "loss_mask": batch["full_loss_masks"],
            }
            if batch["multimodal_train_inputs"] is not None:
                forward_kwargs.update(batch["multimodal_train_inputs"])
            if args.enable_mtp_training:
                forward_kwargs["mtp_kwargs"] = {"mtp_labels": batch["tokens"]}
            output_tensor = model(**forward_kwargs)

        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            os.environ["ROUTING_REPLAY_STAGE"] = old_stage

        return output_tensor, partial(loss_function, args, batch, num_microbatches, step_global_batch_size)
```

**Comment：**

- `old_stage` 典型为 `replay_backward`（actor L521–522）；forward 用 `replay_forward` 对齐 logprob 记录的 experts，backward 用 `replay_backward` 保证梯度路径与 forward 一致。
- 无 `use_routing_replay` 时此分支不执行；`ENABLE_ROUTING_REPLAY` 由 Ray 注入（见 [[23-CP-RoutingReplay-03-数据流与交互]] §2）。
- `fill_routing_replay` 与 `get_batch` 共用 `slice_with_cp` + TP/SP 切分（actor.py L325–336），保证预填 indices 与 train microbatch layout 一致。

---

## 21. loss.py _allgather_cp_redistribute

**Explain：** [[21-Loss-Advantages-00-MOC]] 文档详述；本专题仅需知 allgather-CP logits 算完 logprob 后需 redistribute 回 CP local layout。

---

## 22. 测试 mock dp_with_cp_group

**Explain：** `reduce_train_step_metrics` 设计为可注入 mock group；单进程测试 monkeypatch `dist.all_reduce` 为 no-op。

**Code：**

```python
## 来源：cp_utils.py L150-L152
    Tests pass a mock ``dp_with_cp_group`` and monkeypatch ``dist.all_reduce``
    to a no-op, then pre-aggregate virtual ranks themselves
```

---

## 23. ENABLE_ROUTING_REPLAY 与 use_routing_replay

**Explain：** 环境变量由 Ray 注入；CLI `use_routing_replay` 控制是否启用 record/replay 逻辑；二者都需满足 patch 已应用。

---

## 24. slice_with_cp pad Callable

**Explain：** `pad_value` 可为 Callable，供非标准 pad（如 multimodal）；token 默认用 0。

**Code：**

```python
## 来源：cp_utils.py L294-L301
        if isinstance(pad_value, Callable):
            pad_func = pad_value
            tokens = pad_func(tokens, pad)
        else:
            pad_tuple = (0, 0) * (tokens.dim() - 1) + (0, pad)
            tokens = F.pad(tokens, pad_tuple, value=pad_value)
```

---

## 25. 扩展阅读

- [[18-Model-Init-02-源码走读]] §7 — `forward_only` 无 stage 切换
- [[19-Train-Step-02-源码走读]] §5–§8 — `fill_routing_replay`、`train_actor` stage 表、`train_one_step`
- [[21-Loss-Advantages-02-源码走读]] — `_build_shifted_tokens` CP 路径
- [[22-Loss-Policy-02-源码走读]] — loss rescale + zero loss
- `tests/test_loss_cp_invariance.py`
