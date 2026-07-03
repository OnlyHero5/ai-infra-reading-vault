---
type: batch-doc
module: 06-PlacementGroup
batch: "06"
doc_type: faq
title: "Placement Group · 关键问题"
tags:
  - slime/batch/06
  - slime/module/placement-group
  - slime/doc/faq
updated: 2026-07-02
---

# Placement Group · 关键问题

---

## Q1：为什么需要 InfoActor 重排 bundle，而不是直接用 Ray 默认顺序？

**Explain：** 多节点集群上 Ray 分配 bundle 的顺序随调度器变化；Megatron 与 SGLang 都假设 rank/engine id 按 **节点 IP + GPU id** 单调递增。不重排会导致 NCCL rank 与物理 GPU 不一致，引发性能下降或 hang。

**Code：**

```python
## 来源：slime/ray/placement_group.py L84-L86
# 提交版本：22cdc6e1
sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
```

**Comment：**

- 单节点多卡时重排通常是恒等映射
- 跨节点时差异最明显

---

## Q2：colocate 下 actor 和 rollout 会不会同时占 GPU？

**Explain：** PG 层面 **同时预留** bundle，但 Slime 通过 offload 协议保证同一时刻只有一侧活跃：`offload_rollout` 在 init 后释放 SGLang 显存；训练时 `wake_up` actor、`sleep` 后再 `load` rollout。

**Code：**

```python
## 来源：slime/ray/placement_group.py L243-L244
# 提交版本：22cdc6e1
if args.offload_rollout:
    ray.get(rollout_manager.offload.remote())
```

**Comment：**

- colocate **必须** 配合 offload 标志，否则 OOM
- 详见 [[17-Megatron-Actor-Init-04-关键问题]] 的 offload 生命周期

---

## Q3：`rollout_external` 时 PG 如何分配？

**Explain：** 外部 rollout 集群独立部署；本地 PG 只服务 Megatron，`rollout_offset` 设为 `actor_num_gpus` 使 rollout 视图为空切片（外部引擎不消费本地 PG）。

**Code：**

```python
## 来源：slime/ray/placement_group.py L106-L109
# 提交版本：22cdc6e1
if args.rollout_external:
    if args.debug_rollout_only:
        return 0, 0
    return actor_num_gpus, actor_num_gpus
```

**Comment：**

- `debug_rollout_only + external` 时完全不申请 GPU
- 见 [[16-External-Engines-00-MOC]]

---

## Q4：为什么 `num_gpus_per_actor=0.4`？

**Explain：** Ray fractional GPU 允许同一 bundle 上叠加多个 Actor 的 CPU/GPU 份额；0.4 为 colocate 场景预留 Ray 调度余量，避免单 bundle 被标称 `num_gpus=1.0` 占满导致辅助 Actor 无法调度。

**Code：**

```python
## 来源：slime/ray/placement_group.py L140-L148
# 提交版本：22cdc6e1
return RayTrainGroup(
    # ...
    num_gpus_per_actor=0.4,
    role=role,
    actor_cls=actor_cls,
)
```

**Comment：**

- 实际 TrainRayActor options 里 `num_gpus=num_gpus_per_actor`（见 actor_group.py）
- 每 rank 仍独占一个 bundle，fractional 是 Ray 资源 accounting 技巧

---

## Q5：critic 与 actor 的 `start_rollout_id` 不一致怎么办？

**Explain：** 代码要求所有 rank（含 critic）返回的 init 结果集合大小为 1，否则 assert 失败；有 critic 时 **优先采用 critic 的 id**（TODO 注释表明未来可能改进）。

**Code：**

```python
## 来源：slime/ray/placement_group.py L199-L205
# 提交版本：22cdc6e1
# TODO how to decide rollout start id when critic is involved?
if args.use_critic:
    start_rollout_ids = critic_start_rollout_ids
else:
    start_rollout_ids = actor_start_rollout_ids

assert len(set(start_rollout_ids)) == 1
```

**Comment：**

- 生产环境应保证 actor/critic 从 **相同 checkpoint iteration** 加载
- 用户可通过 `--start-rollout-id` 显式覆盖

---

## Q6：PG 等待超时会不会失败？

**Explain：** **不会**主动超时失败；设计意图是配合 autoscaler 无限等待直到资源就绪。若集群 GPU 永远不足，进程会一直每 30s 打日志。

**Code：**

```python
## 来源：slime/ray/placement_group.py L51-L56
# 提交版本：22cdc6e1
# The wait stays unbounded, so autoscaling clusters — where a pending
# placement group is what drives scale-up — are unaffected.
```

**Comment：**

- 运维应监控 "Waiting for placement group" 日志
- 本地调试可先 `ray status` 确认 GPU 注册数

---

## Q7：易错 — 分离模式下调换 actor/rollout GPU 数

**Explain：** 非 colocate 时 rollout bundle 从 index `actor_num_gpus` 开始；若 `rollout_num_gpus` 配置大于剩余 bundle 数，PG 创建阶段就会因资源不足而 pending（不会 silent 降级）。

**正确理解：**

```python
## 来源：slime/ray/placement_group.py L117
# 提交版本：22cdc6e1
return actor_num_gpus + args.rollout_num_gpus, actor_num_gpus
```

**错误做法：** 以为 `rollout_num_gpus` 可以独立向 Ray 再申请一个 PG——Slime **始终只创建一个 PG**（除 num_gpus=0  debug 路径外）。
