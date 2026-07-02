---
type: batch-doc
module: 07-RayTrainGroup
batch: "07"
doc_type: concept
title: "RayTrainGroup · 核心概念"
tags:
  - slime/batch/07
  - slime/module/ray-train-group
  - slime/doc/concept
updated: 2026-07-02
---

# RayTrainGroup · 核心概念

---

## 1. RayTrainGroup 职责边界

**Explain：** Group 管理 **Actor 生命周期与远程调用**，不包含 Megatron 训练逻辑；默认绑定 `MegatronTrainRayActor`，也可注入 `actor_cls` 替换后端。

**Code：**

```python
# 来源：slime/ray/actor_group.py L10-L27
# 提交版本：22cdc6e1
class RayTrainGroup:
    """
    A group of ray actors
    Functions start with 'async' should return list of object refs

    Args:
        args (Namespace): Arguments for the actor group.
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
        role (str): Logical role ("actor" / "critic").
    """
```

**Comment：**

- `async_*` vs 同步方法：`save_model` / `update_weights` / `onload` / `offload` 内部直接 `ray.get`
- `_actor_handlers` 按 rank 顺序排列，与 Megatron rank 一致

---

## 2. TrainRayActor：NCCL 环境 bootstrap

**Explain：** 每个 Ray Actor 构造时写入 `MASTER_ADDR/PORT/RANK/WORLD_SIZE/LOCAL_RANK`，再于 `init()` 调用 `dist.init_process_group`。

**Code：**

```python
# 来源：slime/ray/train_actor.py L28-L48
# 提交版本：22cdc6e1
class TrainRayActor(RayActor):
    def __init__(self, world_size, rank, master_addr, master_port):
        configure_logger()
        self._world_size = world_size
        self._rank = rank
        if master_addr:
            self.master_addr, self.master_port = master_addr, master_port
        else:
            self.master_addr, self.master_port = self._get_current_node_ip_and_free_port(
                start_port=random.randint(20000, 21000)
            )
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        os.environ["LOCAL_RANK"] = str(get_local_gpu_id())
```

**Comment：**

- rank 0 的 `master_addr` 为 None，自行选端口；rank > 0 从 rank 0 `get_master_addr_and_port` 获取
- `get_local_gpu_id()` 处理 Ray 已设置 `CUDA_VISIBLE_DEVICES` 的情况

---

## 3. RayActor 基类：地址探测

**Explain：** 最小基类提供 master 地址/port 的 getter 与 **空闲端口扫描**。

**Code：**

```python
# 来源：slime/ray/ray_actor.py L4-L10
# 提交版本：22cdc6e1
class RayActor:
    @staticmethod
    def _get_current_node_ip_and_free_port(start_port=10000, consecutive=1):
        return get_current_node_ip(), get_free_port(start_port=start_port, consecutive=consecutive)

    def get_master_addr_and_port(self):
        return self.master_addr, self.master_port
```

**Comment：**

- `get_current_node_ip` 懒加载 ray（见 misc.py），避免 CPU-only 路径硬依赖
- rank 0 Actor 创建后 driver 同步 `ray.get(actor.get_master_addr_and_port.remote())`

---

## 4. runtime_env 与 NCCL 对齐

**Explain：** 每个 TrainRayActor 携带合并后的 env：`NCCL_CUMEM_ENABLE` 与 SGLang 一致；noset visible devices；可选 torch_memory_saver preload。

**Code：**

```python
# 来源：slime/ray/actor_group.py L55-L62
# 提交版本：22cdc6e1
env_vars = {
    "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
    "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": os.environ.get("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1"),
    **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
    **self.args.train_env_vars,
}
```

**Comment：**

- `offload_train + megatron` 时追加 `LD_PRELOAD` / `TMS_INIT_ENABLE`（torch_memory_saver）
- `use_routing_replay + role=actor` 时设 `ENABLE_ROUTING_REPLAY=1`

---

## 5. 抽象 train / update_weights 接口

**Explain：** `TrainRayActor` 定义 Megatron 后端必须实现的抽象方法；Group 通过 `.remote()` 调用具体子类。

**Code：**

```python
# 来源：slime/ray/train_actor.py L101-L119
# 提交版本：22cdc6e1
@abc.abstractmethod
def sleep(self, tags):
    raise NotImplementedError

@abc.abstractmethod
def wake_up(self, tags):
    raise NotImplementedError

@abc.abstractmethod
def train(self, rollout_id, rollout_data_ref, external_data=None):
    raise NotImplementedError

@abc.abstractmethod
def update_weights(self):
    raise NotImplementedError
```

**Comment：**

- `sleep` / `wake_up` 支撑 colocate offload（批次 17/19）
- `set_rollout_manager` 在基类有默认实现，rank 0 上报 parallel config

---

## 6. set_rollout_manager 与 parallel config

**Explain：** init 完成后 Group 调用 `set_rollout_manager`，rank 0 将 `train_parallel_config` 推送给 RolloutManager。

**Code：**

```python
# 来源：slime/ray/train_actor.py L125-L128
# 提交版本：22cdc6e1
def set_rollout_manager(self, rollout_manager):
    self.rollout_manager = rollout_manager
    if not self.args.debug_rollout_only and self.args.rank == 0:
        ray.get(self.rollout_manager.set_train_parallel_config.remote(self.train_parallel_config))
```

**Comment：**

- `train_parallel_config` 在 MegatronTrainRayActor.init 填充（dp/cp/vpp）
- Rollout 侧用于 data padding 与 DP 分片对齐
