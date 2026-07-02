---
type: batch-doc
module: 07-RayTrainGroup
batch: "07"
doc_type: faq
title: "RayTrainGroup · 关键问题"
tags:
  - slime/batch/07
  - slime/module/ray-train-group
  - slime/doc/faq
updated: 2026-07-02
---

# RayTrainGroup · 关键问题

---

## Q1：为什么 async_init 不 ray.get，而 update_weights 要 ray.get？

**Explain：** init/train 常与 critic、metrics 等 **组合等待**；driver 需要 ObjectRef 列表做灵活编排。update_weights 是硬同步点，必须全部 rank 完成后才能 generate，故 Group 内直接 get。

**Code：**

```python
# 来源：slime/ray/actor_group.py L121-L129 vs L155-L157
# 提交版本：22cdc6e1
# async_init → return [actor.init.remote(...) ...]
# update_weights → return ray.get([actor.update_weights.remote() ...])
```

---

## Q2：rank 0 master 端口为何随机 20000–21000？

**Explain：** 避免与集群上已有服务端口冲突；rank 0 在 **TrainRayActor.__init__** 中选端口，非 Ray head 端口。

**Code：**

```python
# 来源：slime/ray/train_actor.py L37-L38
# 提交版本：22cdc6e1
self.master_addr, self.master_port = self._get_current_node_ip_and_free_port(
    start_port=random.randint(20000, 21000)
)
```

**Comment：**

- `get_free_port` 扫描连续可用端口（misc.py）
- 多 job 同节点时需确保防火墙允许该范围

---

## Q3：critic 能用 routing replay 吗？

**Explain：** **不能**。环境变量 `ENABLE_ROUTING_REPLAY` 仅在 `role=="actor"` 时注入。

**Code：**

```python
# 来源：slime/ray/actor_group.py L86-L88
# 提交版本：22cdc6e1
if self.args.use_routing_replay and self.role == "actor":
    env_vars["ENABLE_ROUTING_REPLAY"] = "1"
```

**Comment：**

- critic 前向不需要 MoE routing replay
- 见 [[23-CP-RoutingReplay]]

---

## Q4：fractional GPU 会导致多 rank 共享物理 GPU 吗？

**Explain：** **不会**。每个 rank 仍绑定 **独立 bundle**（`placement_group_bundle_index=reordered_bundle_indices[rank]`）；fractional 值仅影响 Ray 资源 accounting。

**Code：**

```python
# 来源：slime/ray/actor_group.py L109-L115
# 提交版本：22cdc6e1
actor = TrainRayActor.options(
    num_cpus=num_gpus_per_actor,
    num_gpus=num_gpus_per_actor,
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=reordered_bundle_indices[rank],
    ),
).remote(...)
```

---

## Q5：async_train 的 external_data 列表何时用？

**Explain：** 当每个 rank 需要 **不同** 外部附加数据（如 per-DP-rank 文件路径）时传 list；否则传单个 dict 广播。

**Code：**

```python
# 来源：slime/ray/actor_group.py L140-L145
# 提交版本：22cdc6e1
if isinstance(external_data, list):
    assert len(external_data) == len(self._actor_handlers)
    return [
        actor.train.remote(rollout_id, rollout_data_ref, external_data=ed)
        for actor, ed in zip(self._actor_handlers, external_data, strict=False)
    ]
```

---

## Q6：TrainRayActor 为何不 pop CUDA_VISIBLE_DEVICES？

**Explain：** Ray 已在 worker 进程设置 visible devices；代码注释表明 pop 后仍无法覆盖 `torch.cuda.device_count()` 行为，故用 `get_local_gpu_id` 映射。

**Code：**

```python
# 来源：slime/ray/train_actor.py L45-L47
# 提交版本：22cdc6e1
# TODO: currently this doesn't work as ray has already set torch.cuda.device_count().
# os.environ.pop("CUDA_VISIBLE_DEVICES", None)
```

**Comment：**

- noset 环境变量路径见 [[06-PlacementGroup-01-核心概念]] §5

---

## Q7：易错 — 对 async_train 返回值再 ray.get 两次

**Explain：** `async_train` 返回 ref 列表；driver 应 **一次** `ray.get(refs)`。重复 get 已完成的 ref 会报错。

**正确模式：**

```python
# 提交版本：22cdc6e1（train.py 典型用法）
refs = actor_model.async_train(rollout_id, rollout_data_ref)
ray.get(refs)
```

**错误模式：** 对每个 ref 单独 get 后再 get 整个列表。
