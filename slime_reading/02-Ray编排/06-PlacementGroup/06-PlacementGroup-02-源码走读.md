---
type: batch-doc
module: 06-PlacementGroup
batch: "06"
doc_type: walkthrough
title: "Placement Group · 源码走读"
tags:
  - slime/batch/06
  - slime/module/placement-group
  - slime/doc/walkthrough
updated: 2026-07-02
---

# Placement Group · 源码走读

> 按 **调用顺序** 精读：`create_placement_groups` → `_create_placement_group` → `create_rollout_manager` → `create_training_models`。

---

## 1. PG 就绪等待与 autoscaler 友好

**Explain：** 不用裸 `ray.get(pg.ready())`，而是每 30s 轮询并打印集群 GPU 注册数，避免 autoscaler 扩容期间「无日志挂死」。

**Code：**

```python
## 来源：slime/ray/placement_group.py L57-L67
# 提交版本：22cdc6e1
ready_ref = pg.ready()
elapsed = 0
log_interval = 30
while not ray.wait([ready_ref], timeout=log_interval)[0]:
    elapsed += log_interval
    total = ray.cluster_resources().get("GPU", 0)
    available = ray.available_resources().get("GPU", 0)
    logger.info(
        f"Waiting for placement group of {num_gpus} GPUs (elapsed {elapsed}s): "
        f"{total:g} GPUs registered with Ray, {available:g} available."
    )
```

**Comment：**

- 等待 **无上限**，pending PG 会驱动 Ray autoscaler 加节点
- 就绪后才启动 InfoActor 探测

---

## 2. InfoActor 探测 GPU 拓扑

**Explain：** 每个 bundle 起一个 `InfoActor`，通过 `PlacementGroupSchedulingStrategy` 钉在该 bundle 上，读取 `(node_ip, ray_gpu_id)` 后立刻 `ray.kill` 释放。

**Code：**

```python
## 来源：slime/ray/placement_group.py L15-L18, L69-L82
# 提交版本：22cdc6e1
@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]

# ...
info_actors = []
for i in range(num_bundles):
    info_actors.append(
        InfoActor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=i,
            ),
        ).remote()
    )
gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
for actor in info_actors:
    ray.kill(actor)
```

**Comment：**

- `num_gpus=1` 保证 Actor 占满 bundle 的 GPU 槽位
- 探测 Actor 生命周期极短，不影响后续正式 Actor 调度

---

## 3. 日志输出重排映射

**Explain：** 重排完成后逐 bundle 打印 logical index → actual bundle index → node → gpu，便于运维对照 NCCL 日志。

**Code：**

```python
## 来源：slime/ray/placement_group.py L90-L95
# 提交版本：22cdc6e1
for i in range(num_bundles):
    actual_bundle_index = pg_reordered_bundle_indices[i]
    logger.info(
        f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
        f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
    )
```

**Comment：**

- logical `i` 即 Megatron/SGLang 的 rank/engine id（在各自 PG 视图内）
- 排查「rank 与 GPU 错位」时优先查此日志

---

## 4. allocate_train_group：PG → RayTrainGroup

**Explain：** 不直接创建 Ray Actor，而是构造 `RayTrainGroup`，由[[07-RayTrainGroup-00-MOC]] 在 `_allocate_gpus_for_actor` 里实例化各 rank 的 TrainRayActor。

**Code：**

```python
## 来源：slime/ray/placement_group.py L140-L149
# 提交版本：22cdc6e1
def allocate_train_group(args, num_nodes, num_gpus_per_node, pg, role="actor", actor_cls=None):
    return RayTrainGroup(
        args=args,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        pg=pg,
        num_gpus_per_actor=0.4,
        role=role,
        actor_cls=actor_cls,
    )
```

**Comment：**

- `num_gpus_per_actor=0.4` 允许 **同一 GPU bundle 上 Ray 调度多个 fractional GPU actor**（colocate + offload 场景预留 CPU 份额）
- `role` 区分 actor/critic 的环境变量与实现类

---

## 5. create_training_models：actor/critic 初始化编排

**Explain：** 创建 actor（及可选 critic）TrainGroup，**并行** `async_init`，统一 `start_rollout_id`，再注入 `rollout_manager` 引用。

**Code：**

```python
## 来源：slime/ray/placement_group.py L191-L212
# 提交版本：22cdc6e1
actor_start_rollout_ids = ray.get(
    actor_model.async_init(
        actor_args,
        role="actor",
        with_ref=actor_args.kl_coef != 0 or actor_args.use_kl_loss,
        with_opd_teacher=actor_args.use_opd and actor_args.opd_type == "megatron",
    )
)
# ...
assert len(set(start_rollout_ids)) == 1

if args.start_rollout_id is None:
    args.start_rollout_id = start_rollout_ids[0]

actor_model.set_rollout_manager(rollout_manager)
if args.use_critic:
    critic_model.set_rollout_manager(rollout_manager)
```

**Comment：**

- `with_ref` 由 KL 相关 flag 决定，控制 ref model 是否加载
- 所有 rank 必须返回相同 `start_rollout_id`（checkpoint 对齐）
- critic 存在时 **暂用 critic 的 start id**（代码 TODO：更精细策略）

---

## 6. Megatron 分角色 args 解析

**Explain：** 若提供 `megatron_config_path`，actor/critic 各自 `parse_megatron_role_args` 得到独立 Namespace。

**Code：**

```python
## 来源：slime/ray/placement_group.py L152-L188
# 提交版本：22cdc6e1
actor_args = args
if args.megatron_config_path is not None:
    from slime.utils.arguments import parse_megatron_role_args

    actor_args = parse_megatron_role_args(args, args.megatron_config_path, role="actor")
# ...
if args.use_critic:
    critic_args = (
        parse_megatron_role_args(args, args.megatron_config_path, role="critic")
        if args.megatron_config_path is not None
        else copy.deepcopy(args)
    )
    if args.megatron_config_path is None:
        critic_args.disable_param_buffers_cpu_backup = False
```

**Comment：**

- critic 无独立 megatron yaml 时 deepcopy 全局 args，并 **强制开启** CPU param backup
- actor/critic 共用 `pgs["actor"]` / `pgs["critic"]`（critic PG = actor PG）

---

## 7. create_rollout_manager：Rollout 侧 PG 绑定

**Explain：** RolloutManager 是 **无 GPU** 的 Ray Actor（`num_gpus=0`），但构造时传入 rollout PG 视图，供内部 SGLang engine 按 bundle 启动。

**Code：**

```python
## 来源：slime/ray/placement_group.py L220-L246
# 提交版本：22cdc6e1
def create_rollout_manager(args, pg):
    from .rollout import RolloutManager

    rollout_manager_options = {
        "num_cpus": 1,
        "num_gpus": 0,
        "runtime_env": {"env_vars": add_default_ray_env_vars()},
    }
    if getattr(args, "rollout_data_transport", "object-store") == "nixl":
        rollout_manager_options["enable_tensor_transport"] = True
    rollout_manager = RolloutManager.options(**rollout_manager_options).remote(args, pg)
    # ...
    if args.offload_rollout:
        ray.get(rollout_manager.offload.remote())

    return rollout_manager, num_rollout_per_epoch
```

**Comment：**

- `pg` 此处是 `pgs["rollout"]` 三元组，不是裸 PlacementGroup
- `offload_rollout` 时 init 末尾立刻 offload SGLang 权重，把 GPU 让给训练（colocate 关键路径）
- `nixl` transport 开启 tensor transport 加速 rollout→train 数据搬运

---

## 8. num_rollout 自动推导

**Explain：** 未指定 `--num-rollout` 时，从 global dataset 长度 × epoch 数推导。

**Code：**

```python
## 来源：slime/ray/placement_group.py L232-L237
# 提交版本：22cdc6e1
num_rollout_per_epoch = None
if args.num_rollout is None:
    num_rollout_per_epoch = ray.get(rollout_manager.get_num_rollout_per_epoch.remote())
    args.num_rollout = num_rollout_per_epoch * args.num_epoch
    assert args.num_rollout > 0
```

**Comment：**

- 依赖 RolloutManager 已连接 DataSource（[[11-DataSource-00-MOC]]）
- 必须在 `create_training_models` **之前** 调用，因 Megatron LR schedule 用 `num_rollout`

---

## 9. global dataset 预加载

**Explain：** 使用 `rollout_global_dataset` 时，在 training models 就绪后预加载 `start_rollout_id - 1` 的样本状态。

**Code：**

```python
## 来源：slime/ray/placement_group.py L214-L215
# 提交版本：22cdc6e1
if args.rollout_global_dataset:
    ray.get(rollout_manager.load.remote(args.start_rollout_id - 1))
```

**Comment：**

- resume 训练时保证 prompt 索引与 checkpoint rollout id 一致
- 与 [[11-DataSource-00-MOC]] 的持久化格式相关

---

## 10. ray_noset_visible_devices 探测

**Explain：** 检测用户是否设置 Ray experimental noset 环境变量，决定 TrainGroup 是否自行管理 CUDA visible devices。

**Code：**

```python
## 来源：slime/ray/utils.py L16-L24, L36-L37
# 提交版本：22cdc6e1
NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
    # ... NPU/HPU/TPU/Intel ...
]

def ray_noset_visible_devices(env_vars=os.environ):
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)
```

**Comment：**

- TrainGroup 默认对所有 noset 变量写 `"1"`，与 SGLang NCCL 行为对齐
- AMD/NPU 等多硬件路径共用同一列表
