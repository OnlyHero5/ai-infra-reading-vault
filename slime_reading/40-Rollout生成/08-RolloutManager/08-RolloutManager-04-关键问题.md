---
type: batch-doc
module: 08-RolloutManager
batch: "08"
doc_type: faq
title: "RolloutManager · 关键问题"
tags:
  - slime/batch/08
  - slime/module/rollout-manager
  - slime/doc/faq
updated: 2026-07-02
---

# RolloutManager · 关键问题

---

## Q1：generate() 返回什么？train.py 怎么用？

**答：** 返回 `list[Box]`，长度 = `dp_size`。每个 `Box.x` 是 `ray.ObjectRef`，指向该 DP rank 的 `rollout_data` dict。`train.py` 整包传给 `async_train`，Megatron 各 rank 按自身 dp_rank 取对应元素。

**Code：**

```python
## 来源：train.py L67, L81
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
        ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
```

---

## Q2：rollout_id（参数）与 Sample.rollout_id（字段）有何区别？

| 概念 | 含义 |
|------|------|
| `generate(rollout_id)` 参数 | 训练全局 step 计数，传给 data_source / rollout fn |
| `Sample.rollout_id` 字段 | **loss 聚合分组 id**；compact 模式下多条 sample 共享 |

默认路径下二者常相等（每 execution 一条 sample）；subagent 一条 execution 多条 sample 时必须显式设置 `Sample.rollout_id`。

**易错 vs 正确：**

```python
# ❌ compact 模式：3 条 sibling 未设 rollout_id → assert
samples = [Sample(...), Sample(...), Sample(...)]  # rollout_id 全 None

# ✅ 同一 rollout 的 sibling 共享 id
rid = sample.index
for s in siblings:
    s.rollout_id = rid
```

**Code（校验逻辑）：**

```python
## 来源：slime/ray/rollout.py L898-L924
def _validate_rollout_id_annotated(node, depth=0):
    if node and isinstance(node[0], Sample):
        if depth >= 2 and len(node) > 1:
            rids = [s.rollout_id for s in node]
            missing = [i for i, r in enumerate(rids) if r is None]
            assert not missing, (
                f"Compact rollout returned {len(node)} samples but rollout_id is unset ..."
            )
            assert len(set(rids)) == 1, f"Sibling samples must share rollout_id; got {rids}."
```

---

## Q3：为什么 raw_reward / total_lengths 不按 DP 切分？

**答：** 训练侧 advantage / reward 计算可能需要 **全局 batch 统计**（如 GRPO group 跨 rank）。`partition` + 全局 `raw_reward` 允许 rank 用全局下标查 reward，同时只持有本地 tokens。

若自定义 convert 函数改变了 reward 语义，需同步检查 Megatron 侧索引假设（见 [[20-Train-Data-00-MOC]]）。

---

## Q4：debug_rollout_only 与 debug_train_only 区别？

| 标志 | RolloutManager 行为 |
|------|-------------------|
| `debug_train_only` | 不启动 SGLang servers；`servers={}` |
| `debug_rollout_only` | 正常 generate 到 Sample，**跳过** convert + split |

用于纯推理压测或 RM 调试，不产生训练 ObjectRef。

**Code：**

```python
## 来源：slime/ray/rollout.py L555-L558
        if self.args.debug_rollout_only:
            return
        data = self._convert_samples_to_train_data(data)
```

---

## Q5：load_debug_rollout_data 如何复现训练？

**答：** 指定路径模板（含 `{rollout_id}`），从 torch.save 的文件加载 `samples` 列表，跳过 SGLang 调用。配合 `load_debug_rollout_data_subsample` 可截断数据量。

**Code：**

```python
## 来源：slime/ray/rollout.py L636-L641
        if self.args.load_debug_rollout_data:
            data = torch.load(
                self.args.load_debug_rollout_data.format(rollout_id=rollout_id),
                weights_only=False,
            )["samples"]
            data = [Sample.from_dict(sample) for sample in data]
```

保存格式由 `_save_debug_rollout_data` 写入：`dict(rollout_id=..., samples=[sample.to_dict(), ...])`。

---

## Q6：get_updatable_engines_and_lock 为何只返回第一个 updatable server？

**答：** 多模型 weight update 尚未支持同时更新多个 policy。reference / reward 模型设 `update_weights=False`，自动排除。

**Code：**

```python
## 来源：slime/ray/rollout.py L511-L516
    def _get_updatable_server(self) -> Any | None:
        """When multiple updatable servers exist, returns the first one
        (multi-model weight update is not yet supported)."""
        for srv in self.servers.values():
            if srv.update_weights:
                return srv
        return None
```

---

## Q7：rollout_data_transport object-store vs nixl？

| 模式 | 行为 |
|------|------|
| `object-store`（默认） | 标准 `ray.put` |
| `nixl` | `ray.put(..., _tensor_transport="nixl")`，需 RolloutManager 创建时 `enable_tensor_transport=True` |

nixl 面向大 tensor 跨节点 RDMA 传输；小 batch 用默认即可。

---

## Q8：offload_rollout 与 generate 时序？

典型 sync 训练（`train.py`）：

1. `create_rollout_manager` 后若 offload → engines 已 release
2. `onload_weights` → 恢复 weights 供首次 `update_weights`
3. 每 step：`generate` → `offload` → `train` → `onload_weights` → `update_weights` → `onload_kv`

generate 期间 engines 必须已 onload weights + KV；offload 在 generate **之后** 避免与 Megatron 抢 GPU。

---

## Q9：loss_mask 与 remove_sample 如何处理？

- `loss_mask is None` → 自动 `[1] * response_length`（全 response 参与 loss）
- `remove_sample=True` → mask 强制全 0（保留样本但不训练，便于 logging）

**Code：**

```python
## 来源：slime/ray/rollout.py L750-L759
            if sample.loss_mask is None:
                sample.loss_mask = [1] * sample.response_length
            if sample.remove_sample:
                sample.loss_mask = [0] * sample.response_length
```

---

## Q10：与 verl / OpenRLHF 的数据缓冲有何不同？

Slime 不在 RolloutManager 内维护 replay buffer tensor pool；每 step **即时** convert + put ObjectRef。Buffer 逻辑在 `data_source`（如 `RolloutDataSourceWithBuffer`）侧，RolloutManager 只调用 `data_source` 接口。详见 [[11-DataSource-00-MOC]]。

---

## 对比表：RolloutManager vs SGLang Scheduler

| 维度 | RolloutManager | SGLang Scheduler |
|------|---------------|------------------|
| 进程模型 | Ray CPU Actor | 每 GPU 子进程 |
| 批处理 | Python list + dp_schedule | Continuous batching |
| 输出 | train_data ObjectRef | stream tokens |
| 权重 | 接收 Megatron update | 本地 HF weights |

交叉阅读：[[07-Scheduler-00-MOC]]（SGLang 侧）。
