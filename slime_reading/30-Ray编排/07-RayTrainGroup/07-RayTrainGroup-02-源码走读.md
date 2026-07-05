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
updated: 2026-07-05
---

# RayTrainGroup · 源码走读

> 走读主线：`RayTrainGroup` 是训练侧 Ray actor 集合的管理器。构造阶段它根据 placement group 创建每个 rank 的 actor，并把 rank 0 选出的 master addr/port 传给其他 rank；运行阶段它区分异步训练调用和同步生命周期调用。真正的 torch distributed 初始化发生在 `TrainRayActor.init` 中，而不是 group 构造时。

---

## 1. 创建训练 actor 组

### 1.1 RayTrainGroup 构造时保存拓扑并立即分配 actor

问题与约束：
- 训练 actor 的数量由节点数和每节点 GPU 数决定，且必须绑定到外部创建好的 placement group；group 对象本身不负责初始化 Megatron。

设计选择：
- `__init__` 保存 args、节点/GPU 拓扑、role 和可选 actor class，然后调用 `_allocate_gpus_for_actor(pg, num_gpus_per_actor)` 创建所有 Ray actors。

Explain：
源码注释写明这里是分配 actor GPU，而不是实例化模型；模型和 optimizer 初始化在后续 `async_init` 触发的 actor `init` 中进行。

来源：slime/ray/actor_group.py L10-L46

Code：

```python
class RayTrainGroup:
    """
    A group of ray actors
    Functions start with 'async' should return list of object refs
    """

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

代码逻辑：
- 保存参数命名空间。
- 保存训练 actor 的节点数和每节点 GPU 数。
- 保存 role 和自定义 actor class。
- 立即进入 actor 创建流程。

为什么这样写：
- placement group 资源在 group 构造阶段就要锁定，后续训练流程才能直接发 remote 调用。
- 模型初始化是重操作，延后到 `async_init` 以便调用方显式控制。

不变量与失败模式：
- `pg` 必须是外部传入的 placement group tuple。
- `num_nodes * num_gpus_per_node` 是训练 group 的 world size。
- 构造成功只表示 Ray actors 已创建，不表示 Megatron 初始化完成。

Comment：
`RayTrainGroup` 的构造是资源绑定阶段，不是模型就绪阶段。

### 1.2 _allocate_gpus_for_actor 准备 runtime env 与 offload preload

问题与约束：
- 训练 actor 进程启动前必须准备环境变量；offload_train 的 torch memory saver 依赖 `LD_PRELOAD`，必须在进程创建前注入。

设计选择：
- `_allocate_gpus_for_actor` 先计算 world size，并解包 placement group 和 bundle index；构造 env vars，包含 NCCL、TransformerEngine、Ray visible-device 相关默认值和用户 train_env_vars。若启用 Megatron offload，则查找 torch_memory_saver 动态库并写入 `LD_PRELOAD`、`TMS_INIT_ENABLE`、`TMS_INIT_ENABLE_CPU_BACKUP`。

Explain：
routing replay 只对 actor role 打开；critic 不启用 routing replay。

来源：slime/ray/actor_group.py L48-L89

Code：

```python
def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
    world_size = self._num_nodes * self._num_gpus_per_node

    assert pg is not None
    pg, reordered_bundle_indices, _reordered_gpu_ids = pg

    env_vars = {
        "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
        "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": os.environ.get("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1"),
        **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
        **self.args.train_env_vars,
    }

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
            raise FileNotFoundError(
                "Cannot find torch_memory_saver dynamic library. Please make sure torch_memory_saver is properly installed."
            )

        env_vars["LD_PRELOAD"] = dynlib_path
        env_vars["TMS_INIT_ENABLE"] = "1"
        env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"

    if self.args.use_routing_replay and self.role == "actor":
        env_vars["ENABLE_ROUTING_REPLAY"] = "1"
```

代码逻辑：
- world size 由 group 拓扑计算。
- placement group tuple 解出实际 Ray PG 和重排 bundle index。
- 构造 actor runtime env。
- offload_train 且 Megatron backend 时查找动态库。
- 找不到动态库直接失败。
- actor role 下按需启用 routing replay。

为什么这样写：
- `LD_PRELOAD` 必须在子进程启动前生效，不能等 actor init 后再设置。
- placement group 重排后的 bundle index 决定 rank 到 GPU bundle 的绑定。

不变量与失败模式：
- `pg` 不能为空。
- torch_memory_saver 动态库必须存在于预期路径之一。
- routing replay 只对 actor 生效，critic 不参与。

Comment：
这段是训练 actor 进程环境的唯一准备点。

### 1.3 actor class 与 Ray remote options 在创建前确定

问题与约束：
- 默认训练 actor 是 Megatron 后端，但测试或扩展可能注入自定义 actor class；Ray actor class 需要在 `.remote()` 创建实例前包装。

设计选择：
- 如果 `_actor_cls` 为空，导入 `MegatronTrainRayActor` 作为实现；否则使用注入类。actor options 固定 `num_gpus=1` 和 runtime env；当 rollout data transport 是 `nixl` 时打开 Ray tensor transport。

Explain：
`ray.remote(**actor_options)(actor_impl)` 返回 Ray actor class，后续循环里再 `.options(...).remote(...)` 创建每个 rank。

来源：slime/ray/actor_group.py L90-L103

Code：

```python
if self._actor_cls is None:
    from slime.backends.megatron_utils.actor import MegatronTrainRayActor

    actor_impl = MegatronTrainRayActor
else:
    actor_impl = self._actor_cls

actor_options = {
    "num_gpus": 1,
    "runtime_env": {"env_vars": add_default_ray_env_vars(env_vars)},
}
if getattr(self.args, "rollout_data_transport", "object-store") == "nixl":
    actor_options["enable_tensor_transport"] = True
TrainRayActor = ray.remote(**actor_options)(actor_impl)
```

代码逻辑：
- 选择默认或自定义 actor 实现类。
- 设置 Ray actor options。
- runtime env 通过 `add_default_ray_env_vars` 补默认值。
- nixl 传输打开 tensor transport。
- 包装成 Ray remote actor class。

为什么这样写：
- actor class 选择和 Ray runtime options 都必须在创建 actor 前完成。
- 自定义 actor class 保留了非 Megatron 后端或测试替身的入口。

不变量与失败模式：
- actor_impl 必须实现 TrainRayActor 所需接口。
- `enable_tensor_transport` 只在 Ray 支持相关能力时可用。

Comment：
这段决定了 group 里每个 rank 运行的实际 actor 类。

### 1.4 创建 actor 时 rank 0 先产生 master addr/port

问题与约束：
- torch distributed 需要所有 rank 使用同一个 master addr/port；但这个端口只有某个 actor 进程能在所在节点上挑选。

设计选择：
- actor 创建循环从 rank 0 开始；rank 0 创建后立即 `ray.get(actor.get_master_addr_and_port.remote())`，拿到 master addr/port，再传给后续 rank 的构造函数。

Explain：
每个 actor 都通过 `PlacementGroupSchedulingStrategy` 绑定到 `reordered_bundle_indices[rank]`，保证 rank 与 placement group bundle 一一对应。

来源：slime/ray/actor_group.py L105-L119

Code：

```python
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

代码逻辑：
- 初始化 actor handler 列表和 master 地址。
- 按 rank 顺序创建 actor。
- 每个 actor 使用对应 placement group bundle。
- rank 0 创建后同步取 master addr/port。
- 保存每个 actor handle。

为什么这样写：
- 后续 rank 构造时需要直接写入相同 MASTER_ADDR/PORT。
- rank 到 bundle 的稳定映射是分布式训练 rank 语义的前提。

不变量与失败模式：
- rank 0 必须成功返回 master addr/port。
- `reordered_bundle_indices` 长度必须至少等于 world size。
- 任一 actor 创建失败会中断 group 构造。

Comment：
这是 Ray actor 创建阶段最关键的同步点。

---

## 2. TrainRayActor 内部初始化

### 2.1 get_local_gpu_id 处理 Ray GPU id 与 CUDA_VISIBLE_DEVICES

问题与约束：
- Ray 可能设置 `CUDA_VISIBLE_DEVICES`，此时 `ray.get_gpu_ids()[0]` 是物理 id，而 torch 需要本进程可见设备序号。

设计选择：
- 如果没有 `CUDA_VISIBLE_DEVICES`，直接用 Ray GPU id；否则在 CVD 列表中查找 Ray GPU id 的位置，返回本地 ordinal。

Explain：
这个值会写入 `LOCAL_RANK`，后续 `TrainRayActor.init` 用它设置 `torch.cuda.set_device`。

来源：slime/ray/train_actor.py L20-L25

Code：

```python
def get_local_gpu_id():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cvd is None:
        return ray.get_gpu_ids()[0]
    else:
        return cvd.split(",").index(str(ray.get_gpu_ids()[0]))
```

代码逻辑：
- 读取 `CUDA_VISIBLE_DEVICES`。
- 未设置时返回 Ray GPU id。
- 已设置时将 Ray GPU id 转成 CVD 列表中的下标。

为什么这样写：
- torch device ordinal 是本进程可见设备序号，不一定等于物理 GPU id。
- placement group 绑定和 Ray CVD 重映射需要在 actor 内再次对齐。

不变量与失败模式：
- `ray.get_gpu_ids()` 必须非空。
- CVD 已设置时，Ray GPU id 必须出现在 CVD 列表里。

Comment：
这段是 Ray 资源调度和 torch CUDA ordinal 之间的适配。

### 2.2 TrainRayActor 构造函数写入分布式环境变量

问题与约束：
- `dist.init_process_group` 默认从环境变量读取 MASTER_ADDR、MASTER_PORT、WORLD_SIZE、RANK；这些变量要在 actor 初始化 process group 前写好。

设计选择：
- rank 0 没有 master_addr 时自己在当前节点找 IP 和空闲端口；其他 rank 使用 group 传入的 master addr/port。构造函数写入 MASTER_ADDR、MASTER_PORT、WORLD_SIZE、RANK 和 LOCAL_RANK。

Explain：
master 端口随机从 20000 到 21000 开始找；`LOCAL_RANK` 来自 `get_local_gpu_id()`。

来源：slime/ray/train_actor.py L28-L49

Code：

```python
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

代码逻辑：
- 配置 actor 进程 logger。
- 保存 world size 和 rank。
- 有 master addr 时复用传入地址。
- 没有 master addr 时本 actor 自选地址和端口。
- 写入 torch distributed 需要的环境变量。

为什么这样写：
- Ray actor 是独立进程，不能依赖 driver 进程的分布式环境变量。
- rank 0 选端口后广播给其他 rank，可以避免手工配置 master 端口。

不变量与失败模式：
- 只有 rank 0 应在 master_addr 为空时创建。
- 空闲端口选择存在竞态，但立即被 actor 环境使用。
- LOCAL_RANK 取决于 Ray GPU id 和 CVD 映射正确。

Comment：
TrainRayActor 构造完成后，进程组初始化所需 env 已经就绪。

### 2.3 TrainRayActor.init 建立 NCCL/Gloo 并设置 NUMA affinity

问题与约束：
- actor 创建后还没有初始化 torch distributed；模型加载前必须设置 CUDA device、进程组、辅助 Gloo group 和 args rank/world_size。

设计选择：
- `init` 保存 args 和 role，设置 CUDA device 到 LOCAL_RANK，按 `args.distributed_backend` 和 `distributed_timeout_minutes` 初始化 process group，再调用 `init_gloo_group`。随后从 `dist` 写回 `args.rank/world_size`，并尽量用 pynvml 设置 NUMA affinity。

Explain：
ROCm/HIP 环境跳过 NUMA affinity；pynvml 缺失或设置失败只记录 warning，不中断初始化。

来源：slime/ray/train_actor.py L50-L92

Code：

```python
def init(self, args, role, with_ref=False, ...):
    self.args = args
    self.role = role
    self.with_ref = with_ref

    torch.serialization.add_safe_globals([slime.utils.eval_config.EvalDatasetConfig])

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

    try:
        if torch.version.hip is not None:
            logger.info("Detected ROCm/HIP environment, skipping NUMA affinity setup")
        else:
            import pynvml

            pynvml.nvmlInit()
            local_rank = int(os.environ["RANK"]) % args.num_gpus_per_node
            handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
            pynvml.nvmlDeviceSetCpuAffinity(handle)
            logger.info(f"Set NUMA affinity for GPU {local_rank}")
            pynvml.nvmlShutdown()
    except ImportError:
        logger.info("Warning: pynvml not available, skipping NUMA affinity setup")
    except Exception as e:
        logger.info(f"Warning: Failed to set NUMA affinity: {e}")
```

代码逻辑：
- 保存初始化参数和角色。
- 允许 eval config 通过 torch serialization 安全加载。
- 按 LOCAL_RANK 设置 CUDA device。
- 初始化 torch distributed process group。
- 初始化辅助 Gloo group。
- 将实际 rank/world_size 写回 args。
- 尝试设置 NUMA affinity。

为什么这样写：
- Ray actor 的 CUDA device 和 distributed rank 都必须在 actor 进程内确认。
- Gloo group 支持 CPU tensor gather 等辅助路径。
- NUMA affinity 是性能优化，失败不应阻断训练。

不变量与失败模式：
- MASTER_ADDR/PORT/WORLD_SIZE/RANK 必须已在构造函数中设置。
- `args.distributed_backend` 必须被 torch 支持。
- LOCAL_RANK 必须能对应当前进程可见 CUDA device。

Comment：
真正的分布式训练初始化发生在 actor 的 `init`，不是 Ray actor 创建时。

### 2.4 TrainRayActor 定义训练后端必须实现的抽象接口

问题与约束：
- RayTrainGroup 调用的是统一 actor API，但不同训练后端可能实现不同模型加载、训练、保存和权重同步逻辑。

设计选择：
- `TrainRayActor` 将 `sleep`、`wake_up`、`train`、`save_model`、`update_weights`、`_get_parallel_config` 声明为抽象方法；`set_rollout_manager` 是基类实现，保存 rollout manager，并在非 debug rollout-only 的 rank 0 上写入 train parallel config。

Explain：
Megatron 后端 actor 继承该基类并实现这些方法；group 侧只依赖统一接口。

来源：slime/ray/train_actor.py L101-L128

Code：

```python
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
def save_model(self, rollout_id, force_sync=False):
    raise NotImplementedError

@abc.abstractmethod
def update_weights(self):
    raise NotImplementedError

@abc.abstractmethod
def _get_parallel_config(self):
    raise NotImplementedError

def set_rollout_manager(self, rollout_manager):
    self.rollout_manager = rollout_manager
    if not self.args.debug_rollout_only and self.args.rank == 0:
        ray.get(self.rollout_manager.set_train_parallel_config.remote(self.train_parallel_config))
```

代码逻辑：
- 生命周期、训练、保存、权重同步都是抽象接口。
- rollout manager handle 保存在 actor 上。
- rank 0 将 train parallel config 传给 rollout manager。
- debug rollout-only 跳过 train parallel config 写入。

为什么这样写：
- Group 层不应绑定 Megatron 细节。
- rollout manager 只需要一份训练并行配置，由 rank 0 提供即可。

不变量与失败模式：
- 自定义 actor 必须实现所有抽象方法。
- `self.train_parallel_config` 必须在 set_rollout_manager 前可用。
- rank 0 写配置失败会让 group 的同步调用失败。

Comment：
这段解释了为什么 RayTrainGroup 可以接受 `actor_cls` 注入。

---

## 3. Group API 的异步与同步边界

### 3.1 async_init 并发触发每个 actor 初始化

问题与约束：
- 所有 rank 都要初始化模型和分布式状态；driver 不应在 group 方法内部阻塞等待每个 rank 完成，而是让调用方决定何时 `ray.get`。

设计选择：
- `async_init` 更新 group 上的 args，并对所有 actor 发起 `actor.init.remote(...)`，返回 ObjectRef 列表。

Explain：
方法名以 `async` 开头，符合类 docstring 的约定：返回 Ray object refs，而不是已解析结果。

来源：slime/ray/actor_group.py L121-L129

Code：

```python
def async_init(self, args, role, with_ref=False, ...):
    self.args = args
    return [
        actor.init.remote(args, role, with_ref=with_ref, ...)
        for actor in self._actor_handlers
    ]
```

代码逻辑：
- 保存最新 args。
- 遍历所有 actor handler。
- 对每个 actor 发起远程 init。
- 返回 ObjectRef 列表。

为什么这样写：
- 初始化可能很慢，需要让调用方并发等待或和其他初始化流程组合。
- 各 rank init 内部会通过 torch distributed barrier 等机制对齐。

不变量与失败模式：
- `_actor_handlers` 必须已经由构造函数填充。
- 调用方必须对返回 refs 执行 `ray.get` 或等价等待，才能知道 init 是否成功。

Comment：
`async_init` 是 group API 中最典型的异步边界。

### 3.2 async_train 支持共享 rollout_data_ref 和 per-rank external_data

问题与约束：
- 训练一步要把同一个 rollout data Ray object ref 发给所有 rank；某些场景还需要给每个 rank 不同的外部数据。

设计选择：
- 如果 `external_data` 是 list，要求长度等于 actor 数，并按 actor zip 分发；否则把同一个 external_data 传给所有 actor。两种情况都返回 ObjectRef 列表。

Explain：
docstring 说明 critic 的 ref 可能解析成 values dict，而 actor ref 解析为 None。

来源：slime/ray/actor_group.py L131-L149

Code：

```python
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

代码逻辑：
- 判断 external_data 是否为 list。
- list 模式校验长度和 actor 数一致。
- list 模式逐 rank 传不同 external data。
- 非 list 模式广播同一个 external data。
- 返回所有 train 远程调用 refs。

为什么这样写：
- rollout data 本体通常已经在 Ray object store 中，所有 rank 共享同一个 ref。
- external_data 可能是按 rank 切分的辅助输入，需要 per-rank 分发能力。

不变量与失败模式：
- per-rank external_data 长度必须等于 actor 数。
- 调用方必须收集 refs，才能聚合训练结果或暴露异常。

Comment：
`async_train` 只负责发起训练，不在 group 内部同步等待。

### 3.3 save、update、onload/offload/clear_memory 在 group 内同步等待

问题与约束：
- 保存模型、权重同步、显存 onload/offload 和清理内存都是生命周期或一致性操作；下一步 generate/train 不能在这些操作未完成时开始。

设计选择：
- 这些方法内部直接 `ray.get([...])` 等待所有 actor 远程调用完成。`update_weights` 的 docstring 明确是从 rank 0 broadcast 到其他 rank。

Explain：
这与 `async_init/async_train` 相反：调用方拿到的是已解析结果，而不是 ObjectRef 列表。

来源：slime/ray/actor_group.py L151-L169

Code：

```python
def save_model(self, rollout_id, force_sync=False):
    return ray.get([actor.save_model.remote(rollout_id, force_sync=force_sync) for actor in self._actor_handlers])

def update_weights(self):
    return ray.get([actor.update_weights.remote() for actor in self._actor_handlers])

def onload(self):
    return ray.get([actor.wake_up.remote() for actor in self._actor_handlers])

def offload(self):
    return ray.get([actor.sleep.remote() for actor in self._actor_handlers])

def clear_memory(self):
    return ray.get([actor.clear_memory.remote() for actor in self._actor_handlers])

def set_rollout_manager(self, rollout_manager):
    return ray.get([actor.set_rollout_manager.remote(rollout_manager) for actor in self._actor_handlers])
```

代码逻辑：
- 每个方法都向所有 actor 发起对应 remote call。
- 用 `ray.get` 同步等待全部结果。
- set_rollout_manager 也同步下发 rollout manager handle。

为什么这样写：
- 权重和生命周期状态必须跨 rank 一致。
- 对调用方隐藏 ObjectRef，减少主循环遗漏同步的风险。

不变量与失败模式：
- 任一 rank 失败都会使 `ray.get` 抛异常。
- `wake_up` 和 `sleep` 由具体后端 actor 实现，group 层不处理 tags。
- update_weights 完成只表示 actor 侧同步结束，具体推送到 rollout engine 的语义由后端实现决定。

Comment：
这里是 RayTrainGroup API 的设计分界：训练是异步发起，一致性操作同步完成。

### 3.4 RayActor 提供 master 地址查询的最小父类能力

问题与约束：
- RayTrainGroup 创建 rank 0 后需要从 actor 中取 master addr/port；这个能力不应该依赖具体训练后端实现。

设计选择：
- `RayActor` 提供 `_get_current_node_ip_and_free_port` 和 `get_master_addr_and_port`；TrainRayActor 继承它，在构造函数中设置 `self.master_addr/self.master_port`。

Explain：
group 侧调用 `actor.get_master_addr_and_port.remote()`，实际执行的是父类方法。

来源：slime/ray/ray_actor.py L4-L10

Code：

```python
class RayActor:
    @staticmethod
    def _get_current_node_ip_and_free_port(start_port=10000, consecutive=1):
        return get_current_node_ip(), get_free_port(start_port=start_port, consecutive=consecutive)

    def get_master_addr_and_port(self):
        return self.master_addr, self.master_port
```

代码逻辑：
- 静态方法返回当前节点 IP 和空闲端口。
- 实例方法返回 actor 已保存的 master 地址。

为什么这样写：
- master 地址选择是 Ray actor 基础能力，不属于 Megatron 训练逻辑。
- 父类提供统一方法，group 创建逻辑可以独立于后端 actor class。

不变量与失败模式：
- 子类必须在调用前设置 `self.master_addr` 和 `self.master_port`。
- 端口空闲检测和实际占用之间仍有竞态窗口。

Comment：
这个小父类让 RayTrainGroup 的 rank 0 bootstrap 保持通用。

---

## 4. 走读小结

```text
RayTrainGroup(...)
  -> _allocate_gpus_for_actor
     -> runtime_env / LD_PRELOAD / routing replay
     -> ray.remote(actor_impl)
     -> create rank actors on PG bundles
     -> rank 0 exports master addr/port

async_init / async_train
  -> return ObjectRef list

save_model / update_weights / onload / offload / clear_memory / set_rollout_manager
  -> ray.get inside group
```

**下一专题关联：** placement group bundle 重排见 [[06-PlacementGroup-00-MOC]]；Megatron actor 初始化见 [[17-Megatron-Actor-Init-00-MOC]]；权重同步见 [[24-WeightSync-Dist-00-MOC]]。
