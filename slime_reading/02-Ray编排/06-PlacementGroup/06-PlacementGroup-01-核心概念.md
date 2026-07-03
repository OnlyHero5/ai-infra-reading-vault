---
type: batch-doc
module: 06-PlacementGroup
batch: "06"
doc_type: concept
title: "Placement Group · 核心概念"
tags:
  - slime/batch/06
  - slime/module/placement-group
  - slime/doc/concept
updated: 2026-07-02
---

# Placement Group · 核心概念

> 基线 commit：`22cdc6e1` | 源码：`placement_group.py`、`utils.py`

---

## 1. Ray Placement Group 与 Bundle

**Explain：** Ray PG 是一组 **预声明的资源 bundle**（此处每 bundle = 1 GPU + 1 CPU）。`strategy="PACK"` 尽量把 bundle 打包到最少节点上，减少跨节点 NCCL 通信。

**Code：**

```python
## 来源：slime/ray/placement_group.py L42-L48
# 提交版本：22cdc6e1
def _create_placement_group(num_gpus):
    """Create a placement group with the specified number of GPUs."""
    if num_gpus == 0:
        return None, [], []

    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
```

**Comment：**

- `num_gpus==0` 出现在 `debug_train_only` 或 `debug_rollout_only` 某些组合
- PACK 与 STRICT_PACK 不同：前者允许在资源不足时部分放置（配合 autoscaler）

---

## 2. Bundle 重排：逻辑 rank ↔ 物理 GPU

**Explain：** Ray 分配 bundle 的顺序不一定等于「节点 IP 升序 + GPU ID 升序」。Slime 用临时 `InfoActor` 探测每个 bundle 的 `(node_ip, gpu_id)`，再 `sort_key` 重排，使 Megatron rank `i` 与 SGLang engine `i` 对齐。

**Code：**

```python
## 来源：slime/ray/placement_group.py L21-L39
# 提交版本：22cdc6e1
def sort_key(x):
    index, node_identifier, gpu_id = x
    try:
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, gpu_id)
```

**Comment：**

- hostname 无法解析时退化为 ASCII 序，保证排序稳定
- 重排后 `pg_reordered_bundle_indices[i]` 是逻辑 rank `i` 应绑定的 **原始 bundle 下标**

---

## 3. colocate 与 rollout_offset

**Explain：** `_get_placement_group_layout` 根据 CLI 标志计算 **总 GPU 数** 与 **rollout 在 PG 中的起始偏移**。colocate 时只申请 `max(actor, rollout)` 块 GPU，rollout 与 actor 共享前 N 个 bundle。

**Code：**

```python
## 来源：slime/ray/placement_group.py L100-L117
# 提交版本：22cdc6e1
def _get_placement_group_layout(args) -> tuple[int, int]:
    actor_num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node

    if args.debug_train_only:
        return actor_num_gpus, 0

    if args.rollout_external:
        if args.debug_rollout_only:
            return 0, 0
        return actor_num_gpus, actor_num_gpus

    if args.debug_rollout_only:
        return args.rollout_num_gpus, 0

    if args.colocate:
        return max(actor_num_gpus, args.rollout_num_gpus), 0

    return actor_num_gpus + args.rollout_num_gpus, actor_num_gpus
```

**Comment：**

| 模式 | num_gpus | rollout_offset | 含义 |
|------|----------|----------------|------|
| colocate | max(actor, rollout) | 0 | 共用 PG |
| 分离 | actor + rollout | actor_num_gpus | rollout 占后半段 bundle |
| debug_train_only | actor | 0 | 无 rollout GPU |
| rollout_external | actor | actor_num_gpus | rollout 在外部集群，PG 只服务训练 |

---

## 4. PG 结果字典的三元组

**Explain：** `create_placement_groups` 返回的每个角色条目是 `(pg, reordered_bundle_indices, reordered_gpu_ids)`。`RayTrainGroup` 与 `RolloutManager` 用 **bundle index** 调度 Actor；物理 GPU ID 用于日志与 NCCL 调试。

**Code：**

```python
## 来源：slime/ray/placement_group.py L84-L88
# 提交版本：22cdc6e1
bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]
```

**Comment：**

- `reordered_gpu_ids[i]` 是逻辑 rank `i` 对应的 CUDA 设备 UUID（经 InfoActor 探测）
- rollout 视图是 actor 视图的 **切片** `[rollout_offset:]`

---

## 5. Ray 默认环境变量

**Explain：** Slime 关闭 Ray 的 uvloop 集成（避免 async actor 间歇性故障），并通过 `add_default_ray_env_vars` 合并到 RolloutManager / TrainActor 的 `runtime_env`。

**Code：**

```python
## 来源：slime/ray/utils.py L26-L33
# 提交版本：22cdc6e1
RAY_DEFAULT_ENV_VARS = {
    # Ray's uvloop integration has caused intermittent async actor issues.
    "RAY_USE_UVLOOP": "0",
}

def add_default_ray_env_vars(env_vars: dict[str, str] | None = None) -> dict[str, str]:
    return RAY_DEFAULT_ENV_VARS | (env_vars or {})
```

**Comment：**

- `NOSET_VISIBLE_DEVICES_ENV_VARS_LIST` 在 TrainGroup 侧设为 `"1"`，让 Slime 自行管理 `LOCAL_RANK`
- 见 [[07-RayTrainGroup-01-核心概念]] 的 NCCL 环境变量装配

---

## 6. 分布式 Lock（辅助原语）

**Explain：** `utils.Lock` 是基于 Ray Actor 的简易互斥锁，供权重同步等路径使用；非 PG 核心逻辑，但同属 Ray 工具层。

**Code：**

```python
## 来源：slime/ray/utils.py L46-L64
# 提交版本：22cdc6e1
@ray.remote
class Lock(RayActor):
    def __init__(self):
        self._locked = False

    def acquire(self):
        if not self._locked:
            self._locked = True
            return True
        return False

    def release(self):
        assert self._locked, "Lock is not acquired, cannot release."
        self._locked = False
```

**Comment：**

- 调用方需 spin 直到 `acquire()` 返回 True
- 继承 [[07-RayTrainGroup-00-MOC]] 文档中的 `RayActor` 基类（master addr 探测）
