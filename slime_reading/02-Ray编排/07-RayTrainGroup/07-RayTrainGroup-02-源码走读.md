---
type: batch-doc
module: 07-RayTrainGroup
batch: "07"
doc_type: walkthrough
title: "RayTrainGroup · 源码走读"
tags:
  - slime/batch/07
  - slime/module/ray-train-group
  - slime/doc/walkthrough
updated: 2026-07-02
---

# RayTrainGroup · 源码走读

> 精读 `_allocate_gpus_for_actor` 创建循环，以及全套 Group API 的 `.remote()` 语义。

---

## 1. 构造与 PG 解包

**Explain：** `__init__` 只保存 args 与 PG 引用，立即调用 `_allocate_gpus_for_actor` 创建全部 worker（**尚未** init Megatron）。

**Code：**

```python
## 来源：slime/ray/actor_group.py L29-L46
# 提交版本：22cdc6e1
def __init__(
    self,
    args,
    num_nodes,
    num_gpus_per_node,
    pg: tuple[PlacementGroup, list[int], list[int]],
    num_gpus_per_actor: float = 1,
    role: str = "actor",
    actor_cls=None,
) -> None:
    self.args = args
    self._num_nodes = num_nodes
    self._num_gpus_per_node = num_gpus_per_node
    self.role = role
    self._actor_cls = actor_cls
    self._allocate_gpus_for_actor(pg, num_gpus_per_actor)
```

**Comment：**

- world_size = `num_nodes * num_gpus_per_node`
- PG 必须非 None（由[[06-PlacementGroup-00-MOC]] 保证）

---

## 2. Actor 实现类选型

**Explain：** 默认 Megatron 后端；自定义 backend 通过 `actor_cls` 注入。

**Code：**

```python
## 来源：slime/ray/actor_group.py L90-L103
# 提交版本：22cdc6e1
if self._actor_cls is None:
    from slime.backends.megatron_utils.actor import MegatronTrainRayActor

    actor_impl = MegatronTrainRayActor
else:
    actor_impl = self._actor_cls

actor_options = {
    "num_gpus": 1,
    "runtime_env": {"env_vars": add_default_ray_env_vars(env_vars)},
}
TrainRayActor = ray.remote(**actor_options)(actor_impl)
```

**Comment：**

- `@ray.remote` 装饰 **类** 后得到 Ray Actor class
- `enable_tensor_transport` 在 nixl 模式下追加到 options

---

## 3. rank 0 master 地址广播

**Explain：** 循环创建 worker；rank 0 创建后阻塞获取 master addr/port，供后续 rank 构造参数传入。

**Code：**

```python
## 来源：slime/ray/actor_group.py L105-L119
# 提交版本：22cdc6e1
self._actor_handlers = []
master_addr, master_port = None, None
for rank in range(world_size):
    actor = TrainRayActor.options(
        num_cpus=num_gpus_per_actor,
        num_gpus=num_gpus_per_actor,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=reordered_bundle_indices[rank],
        ),
    ).remote(world_size, rank, master_addr, master_port)
    if rank == 0:
        master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
    self._actor_handlers.append(actor)
```

**Comment：**

- rank > 0 构造时 `master_addr` 已填充，不再自选端口
- `PlacementGroupSchedulingStrategy` 保证 rank ↔ bundle 一一对应

---

## 4. async_init：Megatron 初始化

**Explain：** 对所有 handler 发起 `init.remote`，返回 ObjectRef 列表。

**Code：**

```python
## 来源：slime/ray/actor_group.py L121-L129
# 提交版本：22cdc6e1
def async_init(self, args, role, with_ref=False, with_opd_teacher=False):
    self.args = args
    return [
        actor.init.remote(args, role, with_ref=with_ref, with_opd_teacher=with_opd_teacher)
        for actor in self._actor_handlers
    ]
```

**Comment：**

- 各 rank 并行 init；Megatron 内部还有 barrier
- 返回值 per-rank 为 `loaded_rollout_id + 1` 或 debug 时 `0`

---

## 5. TrainRayActor.init：process group

**Explain：** 父类 init 建立 NCCL + gloo，设置 NUMA affinity，填充 `args.rank/world_size`。

**Code：**

```python
## 来源：slime/ray/train_actor.py L50-L70
# 提交版本：22cdc6e1
def init(self, args, role, with_ref=False, with_opd_teacher=False):
    self.args = args
    self.role = role
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(f"cuda:{local_rank}")

    backend = args.distributed_backend
    dist.init_process_group(
        backend=backend,
        timeout=timedelta(minutes=args.distributed_timeout_minutes),
    )
    init_gloo_group()

    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()
```

**Comment：**

- `init_gloo_group` 供 CPU tensor gather 等辅助路径
- MegatronTrainRayActor.init 在 super().init 之后继续加载模型（[[17-Megatron-Actor-Init-00-MOC]]）

---

## 6. async_train：训练一步

**Explain：** 每个 worker 收到相同 `rollout_data_ref`（Ray object store 引用）；可选 per-rank `external_data` 列表。

**Code：**

```python
## 来源：slime/ray/actor_group.py L131-L149
# 提交版本：22cdc6e1
def async_train(self, rollout_id, rollout_data_ref, external_data=None):
    if isinstance(external_data, list):
        assert len(external_data) == len(self._actor_handlers)
        return [
            actor.train.remote(rollout_id, rollout_data_ref, external_data=ed)
            for actor, ed in zip(self._actor_handlers, external_data, strict=False)
        ]
    return [
        actor.train.remote(rollout_id, rollout_data_ref, external_data=external_data)
        for actor in self._actor_handlers
    ]
```

**Comment：**

- critic ref 返回 `{"values": ...}`；actor 返回 `None`
- driver 侧 `ray.get` 后由 train.py 聚合 metrics

---

## 7. update_weights：同步阻塞

**Explain：** 与 async_* 不同，`update_weights` 在 Group 内 **同步** `ray.get` 全部 rank，保证权重 broadcast 完成后再进入下一轮 generate。

**Code：**

```python
## 来源：slime/ray/actor_group.py L155-L157
# 提交版本：22cdc6e1
def update_weights(self):
    """Broadcast weights from rank 0 to all other ranks."""
    return ray.get([actor.update_weights.remote() for actor in self._actor_handlers])
```

**Comment：**

- rank 0 推送到 SGLang + 内部 broadcast 逻辑在 MegatronTrainRayActor（[[24-WeightSync-Dist-00-MOC]]）
- train 主循环：`actor_model.update_weights()` 无需额外 ray.get

---

## 8. save_model / onload / offload

**Explain：** 辅助生命周期 API，均同步等待全部 rank。

**Code：**

```python
## 来源：slime/ray/actor_group.py L151-L163
# 提交版本：22cdc6e1
def save_model(self, rollout_id, force_sync=False):
    return ray.get([actor.save_model.remote(rollout_id, force_sync=force_sync) for actor in self._actor_handlers])

def onload(self):
    return ray.get([actor.wake_up.remote() for actor in self._actor_handlers])

def offload(self):
    return ray.get([actor.sleep.remote() for actor in self._actor_handlers])
```

**Comment：**

- colocate：`offload` train → `onload` rollout 或反之
- `save_model` 在 periodic checkpoint 路径调用

---

## 9. set_rollout_manager 广播

**Explain：** Group 向每个 rank 下发 RolloutManager actor handle。

**Code：**

```python
## 来源：slime/ray/actor_group.py L168-L169
# 提交版本：22cdc6e1
def set_rollout_manager(self, rollout_manager):
    return ray.get([actor.set_rollout_manager.remote(rollout_manager) for actor in self._actor_handlers])
```

**Comment：**

- 每个 rank 持有同一 RolloutManager ref
- rank 0 额外调用 `set_train_parallel_config`

---

## 10. get_local_gpu_id 与 visible devices

**Explain：** 当 Ray 设置 `CUDA_VISIBLE_DEVICES` 时，映射 Ray GPU id 到 local ordinal。

**Code：**

```python
## 来源：slime/ray/train_actor.py L20-L25
# 提交版本：22cdc6e1
def get_local_gpu_id():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cvd is None:
        return ray.get_gpu_ids()[0]
    else:
        return cvd.split(",").index(str(ray.get_gpu_ids()[0]))
```

**Comment：**

- noset visible devices 模式下 cvd 可能为空，走 `ray.get_gpu_ids()[0]`
- 与 PG 重排配合保证 LOCAL_RANK 正确

---

## 11. torch_memory_saver preload（offload_train）

**Explain：** Megatron offload 路径在 Actor 创建前注入 `LD_PRELOAD`。

**Code：**

```python
## 来源：slime/ray/actor_group.py L64-L84
# 提交版本：22cdc6e1
if self.args.offload_train and self.args.train_backend == "megatron":
    import torch_memory_saver
    for path in [
        "torch_memory_saver_hook_mode_preload_cu12.abi3.so",
        "torch_memory_saver_hook_mode_preload.abi3.so",
    ]:
        dynlib_path = os.path.join(
            os.path.dirname(os.path.dirname(torch_memory_saver.__file__)),
            path,
        )
        if os.path.exists(dynlib_path):
            break
    else:
        raise FileNotFoundError(...)
    env_vars["LD_PRELOAD"] = dynlib_path
    env_vars["TMS_INIT_ENABLE"] = "1"
```

**Comment：**

- 必须在 Actor **进程启动前** 写入 runtime_env
- 找不到 so 文件时 fail fast
