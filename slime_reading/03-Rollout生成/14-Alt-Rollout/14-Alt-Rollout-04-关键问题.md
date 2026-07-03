---
type: batch-doc
module: 14-Alt-Rollout
batch: "14"
doc_type: faq
title: "Alt-Rollout · 关键问题"
tags:
  - slime/batch/14
  - slime/module/alt-rollout
  - slime/doc/faq
updated: 2026-07-02
---

# Alt-Rollout · 关键问题

---

## Q1：sync / train_async / fully-async 怎么选？

| 模式 | 何时用 | 限制 |
|------|--------|------|
| sync `train.py` | 默认；colocate 训练+推理 | generate 与 train 串行 |
| `train_async.py` | 非 colocate；generate 与 train 重叠 | 不支持 colocate |
| + fully-async rollout | 长尾样本多、并发池需跨 step 保温 | 不支持 eval；与 train_async 搭配 |

**Explain：** fully-async **不能**单独替代 train_async——它解决的是 rollout **内部** batch 凑齐慢，train_async 解决的是 **step 间** generate/train 串行。最佳组合是 `train_async.py` + `generate_rollout_fully_async`。

---

## Q2：fully-async 为什么不支持 evaluation？

**Explain：** eval 需要确定性、同步语义和独立 metrics 聚合；fully-async 的全局 worker 与跨边界 queue 与 eval 假设冲突。eval 应继续使用默认 `eval_function_path`（通常 `sglang_rollout` eval 路径）。

**Code：**

```python
## 来源：slime/rollout/fully_async_rollout.py L254-L255
if evaluation:
    raise ValueError("fully-async rollout doesn't support evaluation mode")
```

**Comment：** 训练用 fully-async，eval interval 仍走 RolloutManager.eval → 默认 eval 函数。

---

## Q3：权重更新时 ABORTED 样本如何处理？

**Explain：** 两条路径协作：

1. **sglang_rollout 层：** `GenerateState.aborted=True` → 进行中的 `generate_and_rm` 短路 → sample 标 ABORTED。
2. **fully-async 层：** done callback 检测 ABORTED → `data_buffer.add_samples([group])`，不进入 output_queue。

**易错 vs 正确：**

```python
# ❌ 错误理解：worker 主动监听 weight-update 信号
# worker 对 pause/weight-update 无感（模块 docstring 明确说明）

# ✅ 正确：各 generate_and_rm_group 任务自行检测 aborted；
# worker 只负责 ABORTED 组回灌 buffer
if any(getattr(s, "status", None) == Sample.Status.ABORTED for s in result):
    self.data_buffer.add_samples([result])
    return
```

---

## Q4：streaming generate 和默认 generate 的核心差异？

| 维度 | 默认 generate | streaming generate |
|------|---------------|-------------------|
| HTTP | 单次 POST 等完整 JSON | SSE `stream=True` |
| abort 时 partial state | 依赖 `/abort_request` 回收 | 每 chunk 已写入 sample |
| 外层编排 | sglang_rollout | 同左（只换 inner） |
| 配置 | 默认 | `--custom-generate-function-path` |

**Explain：** partial_rollout 场景下 streaming 减少 abort round-trip 丢状态风险；正常完成路径两者等价。

---

## Q5：SFT rollout 为什么还走 RolloutManager？

**Explain：** Slime 统一 **generate → train** 接口。SFT 通过 `--rollout-function-path` 把 generate 换成 tokenize，训练侧、checkpoint、logging 不变。文档见 `docs/en/examples/qwen3-4b-base-openhermes.md`。

**易错 vs 正确：**

```bash
# ❌ 以为 SFT 不需要 rollout-batch-size / data_source
# SFT 仍从 data_buffer.get_samples(rollout_batch_size) 取数据

# ✅ 正确配置
--rollout-function-path slime.rollout.sft_rollout.generate_rollout
--rollout-global-dataset
--prompt-data /path/to/sft.jsonl
```

---

## Q6：OPD 的 reward 为什么是 0？

**Explain：** 纯 on-policy distillation 的学习信号来自 **OPD KL penalty**（训练侧 `compute_advantages_and_returns`），不是 task reward。`post_process_rewards` 返回 `[0.0] * len(samples)` 是为兼容 GRPO/PPO advantage 接口。

**Code：**

```python
## 来源：slime/rollout/on_policy_distillation.py L61-L64
# Return scalar rewards for GRPO/PPO advantage estimator
# For pure on-policy distillation, we use 0.0 as the task reward.
# The learning signal comes entirely from the OPD KL penalty.
scalar_rewards = [0.0] * len(samples)
```

**Comment：** 若有 task reward，在 `post_process_rewards` 中叠加即可；`teacher_log_probs` 仍必须正确写入。

---

## Q7：forge_load 和 load-debug-rollout-data 区别？

| 参数 | SGLang | weight update | colocate dance | 用途 |
|------|--------|---------------|----------------|------|
| `--load-debug-rollout-data` | skip | 可选 train only | 否 | 快速训练 debug |
| `--load-forge-rollout-data` | **live** | 是 | 是 | 真实显存测量 |

**Explain：** forge_load 设计目标是 **长上下文内存测试**：replay 固定 tensor 形状，同时保留完整推理栈开销。

**Code：**

```python
## 来源：slime/rollout/forge_load.py L102-L107
# IMPORTANT: do NOT overwrite sample.rollout_id with the current rollout_id.
# Forcing all samples to share one rollout_id collapses them into a single
# "rollout", which trips the num_rollouts >= global_batch_size assert in
# slime/utils/dp_schedule.py.
```

**Comment：** 这是 forge_load 最常见踩坑——overwrite rollout_id 导致 dp_schedule 断言失败。

---

## Q8：forge_load 路径 literal vs 模板？

| 模式 | 示例 | 行为 |
|------|------|------|
| Literal | `/path/to/0.pt` | 每个 rollout_id 复用同一文件 |
| 模板 | `/path/{rollout_id}.pt` | 按 id 加载；缺失 fallback 0.pt（仅 train） |
| Eval + literal | 同上 | eval 返回空（no-op） |
| Eval + 模板 | `eval_{rollout_id}.pt` | 独立 eval dump |

---

## Q9：sleep_rollout 会卡死训练吗？

**Explain：** 会——`sleep()` 永不返回，Ray `generate.remote` 永久阻塞。这是 **故意设计**，用于 profiling 文档中的「Rollout 等待、单独 profile Train」场景，不是生产配置。

**正确用法：** 仅在 `docs/en/developer_guide/profiling.md` 描述的工作流中使用，且通常配合特定 debug 参数。

---

## Q10：fully-async 并发数如何计算？

**Explain：** `concurrency = args.sglang_server_concurrency * get_rollout_num_engines(args)`，与 `sglang_rollout` 内 per-engine semaphore 对齐，避免 worker 提交超过 SGLang 承受能力的任务。

**Code：**

```python
## 来源：slime/rollout/fully_async_rollout.py L58-L59
_global_worker = AsyncRolloutWorker(
    args, data_buffer, concurrency=args.sglang_server_concurrency * get_rollout_num_engines(args)
)
```

**Comment：** 调大 `sglang_server_concurrency` 前确认 engine 数量与 GPU 拓扑。

---

## Q11：能否 colocate + fully-async？

**Explain：** 不能。`train_async.py` 断言 `not args.colocate`；fully-async 示例均基于非 colocate 异步训练。colocate 场景用 sync `train.py` + 默认 sglang rollout。

---

## Q12：custom rollout 函数如何编写？

**Explain：** 参考 `slime/.claude/skills/add-rollout-function/SKILL.md` 与 `sft_rollout.py` 最小示例。契约：

1. 签名 `(args, rollout_id, data_source, evaluation=False)`。
2. 训练返回 samples（或 `RolloutFnTrainOutput`）。
3. eval 返回 `RolloutFnEvalOutput`（若支持）。
4. 需要 SGLang 时复用 `generate_and_rm_group`，不要重复实现 semaphore/abort。

**易错 vs 正确：**

```python
# ❌ 在 custom rollout 内直接 httpx 调 SGLang，绕过 GenerateState/abort
# ✅ 复用 generate_and_rm_group 或 register custom-generate-function-path
task = asyncio.create_task(generate_and_rm_group(args, group, sampling_params, evaluation=False))
```
