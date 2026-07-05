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
updated: 2026-07-05
---

# Placement Group · 源码走读

## 1. Placement Group 创建与 GPU 拓扑探测

### 1.1 PG ready 用轮询日志暴露 autoscaler 等待状态

问题与约束：
- Ray placement group 可能需要等待 GPU 注册或 autoscaler 扩容。
- 裸 `ray.get(pg.ready())` 在长时间 pending 时没有进度日志，难以判断是扩容中还是挂死。
- 等待本身不能设置短超时失败，否则会破坏 autoscaler 通过 pending PG 扩容的机制。

设计选择：
- 对 `pg.ready()` 的 object ref 使用 `ray.wait(..., timeout=30)` 轮询。
- 每轮等待失败时记录已等待时间、Ray 已注册 GPU 数和当前可用 GPU 数。
- 仍然保持无上限等待。

Explain：
Slime 创建 PG 后并不直接阻塞在 `ray.get`，而是把等待变成可观测状态。这样扩容慢时日志会持续显示 Ray 资源视图，而不会让用户误以为进程卡住。

来源：slime/ray/placement_group.py L57-L67

Code：

```python
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

代码逻辑：
- 取得 PG ready ref。
- 每 30 秒调用一次 `ray.wait`。
- 未 ready 时累计 elapsed。
- 从 Ray cluster resources 和 available resources 读取 GPU 数。
- 打印当前等待状态。

为什么这样写：
- pending PG 可以驱动 Ray autoscaler 扩容，因此等待不能轻易失败。
- 周期日志提供资源注册与可用量，便于区分“资源不足”和“调度异常”。
- ready 后再继续 GPU 拓扑探测，避免 InfoActor 被调度到尚未稳定的 bundle。

不变量与失败模式：
- `pg.ready()` 必须最终 ready，否则会持续等待。
- Ray resources 中 GPU key 缺失时按 0 记录。
- 日志只反映 Ray 资源视图，不保证底层 NCCL 或 CUDA 初始化一定成功。

Comment：
这一段的价值是可观测性：等待仍然无界，但不再无声。

### 1.2 InfoActor 被钉到每个 bundle 上读取节点与 GPU id

问题与约束：
- Ray PG bundle 的逻辑顺序不一定等同于节点/GPU 的物理顺序。
- 后续 Megatron rank 和 SGLang engine 需要稳定、可解释的 bundle→GPU 映射。
- 探测 actor 不能长期占用资源。

设计选择：
- 定义 `InfoActor`，每个 actor 要求 `num_gpus=1`。
- 对每个 bundle 用 `PlacementGroupSchedulingStrategy` 指定 bundle index 创建一个 InfoActor。
- 收集 `(node_ip, ray_gpu_id)` 后立即 `ray.kill` 探测 actor。

Explain：
InfoActor 是一次性探针。它被 Ray 调度到指定 bundle 上，通过 Ray runtime 读取该 bundle 对应的节点 IP 和 GPU id，用于后续排序和日志。

来源：slime/ray/placement_group.py L15-L82

Code：

```python
@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]

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

代码逻辑：
- `InfoActor` remote actor 占用 1 个 GPU。
- 遍历 bundle index，为每个 bundle 创建一个 actor。
- scheduling strategy 把 actor 固定到对应 bundle。
- 并行读取所有 actor 的 IP 与 GPU id。
- 探测完成后 kill actors。

为什么这样写：
- 只有真正被 Ray 调度到 bundle 上的 actor 才能读到 Ray 视角的 GPU id。
- 每 bundle 一个探针可以建立完整 PG 布局。
- 立即 kill 避免探测 actor 占住后续训练/rollout actor 的资源。

不变量与失败模式：
- 每个 bundle 必须包含可满足 `num_gpus=1` 的资源。
- `ray.get_gpu_ids()[0]` 假设 actor 至少拿到一个 GPU。
- 探测 actor 未 kill 会占用 PG 资源，影响后续 actor 调度。

Comment：
PG 不是只创建资源集合，还要把 Ray 的 bundle 顺序翻译成可用的物理拓扑信息。

### 1.3 排序后的 logical bundle 映射用于 rank 与 GPU 对照

问题与约束：
- Ray 返回的 bundle 顺序不一定按节点 IP 和 GPU id 排好。
- 分布式训练排障时需要知道 logical rank/engine index 对应哪个 actual bundle 和物理 GPU。
- 后续 actor 分配要使用同一套重排映射。

设计选择：
- 构造 `(bundle_index, node, gpu_id)` 列表并按 `sort_key` 排序。
- 生成 `pg_reordered_bundle_indices` 和 `pg_reordered_gpu_ids`。
- 逐 logical index 打印 actual bundle index、node、gpu。

Explain：
排序把 Ray 的 bundle 列表变成 Slime 内部的稳定 logical order。后续 TrainGroup 和 Rollout Server 都从这个三元组中取 bundle index 与 GPU id。

来源：slime/ray/placement_group.py L84-L97

Code：

```python
bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]

for i in range(num_bundles):
    actual_bundle_index = pg_reordered_bundle_indices[i]
    logger.info(
        f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
        f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
    )

return pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids
```

代码逻辑：
- `bundle_infos` 把 bundle index 和探测结果合并。
- `sort_key` 按节点标识和 GPU id 排序。
- 重排后的 bundle indices 用于 Ray scheduling strategy。
- 重排后的 GPU ids 用于 actor 内部 base GPU id。
- 打印 logical index 到 actual bundle/node/gpu 的映射。

为什么这样写：
- 训练 rank 和 rollout engine 使用 logical order 更容易稳定复现。
- 日志映射能和 NCCL、Ray、SGLang 日志交叉排障。
- 返回三元组让后续模块共享同一拓扑视图，避免各自重新排序。

不变量与失败模式：
- `gpu_ids` 长度必须等于 bundle 数。
- `sort_key` 必须对 IP、hostname 或其他 node identifier 产生稳定顺序。
- 后续代码必须同时使用 reordered bundle indices 和 reordered GPU ids，不能混用原始顺序。

Comment：
这一段把 PG 从“资源已分配”推进到“资源顺序可解释”。

## 2. Training 侧 actor 编排

### 2.1 `allocate_train_group` 只创建 RayTrainGroup 包装器

问题与约束：
- PG 创建后还不能直接启动训练 actor；训练 actor 的 rank、环境变量、CUDA visible devices 由 RayTrainGroup 管理。
- actor/critic 角色共享类似的资源绑定逻辑，但 role 和 actor class 可能不同。
- colocate/offload 场景下需要 Ray fractional GPU 资源声明。

设计选择：
- `allocate_train_group` 只构造 `RayTrainGroup`，把 args、节点数、每节点 GPU 数、PG、role 和 actor_cls 传入。
- 固定 `num_gpus_per_actor=0.4`，实际 CUDA 可见设备由后续 TrainGroup 逻辑处理。

Explain：
这一层不是创建每个训练 rank，而是创建训练组控制器。具体 TrainRayActor 的分配和 rank 级启动在 RayTrainGroup 内完成。

来源：slime/ray/placement_group.py L140-L149

Code：

```python
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

代码逻辑：
- 接收 role 所需的节点数、每节点 GPU 数和 PG 三元组。
- 将 role 与可选 actor class 透传给 RayTrainGroup。
- 返回 RayTrainGroup 实例。

为什么这样写：
- PlacementGroup 负责资源集合，RayTrainGroup 负责训练 rank 级 actor 编排。
- actor/critic 可以复用同一个包装器入口。
- fractional GPU 声明让 Ray 允许多个控制 actor 共享同一物理 GPU bundle 的调度资源。

不变量与失败模式：
- `pg` 应是 `_create_placement_group` 返回的三元组，而不是裸 PG。
- `num_nodes * num_gpus_per_node` 必须与 role 需要的 rank 数一致。
- `actor_cls` 若传入，需要兼容 RayTrainGroup 的 actor 构造协议。

Comment：
PG 解决“放哪里”，RayTrainGroup 解决“以什么 rank 和角色启动”。

### 2.2 actor/critic args 按角色解析或复制

问题与约束：
- actor 和 critic 可能有不同 Megatron 配置。
- 没有独立 Megatron YAML 时，critic 仍需要从全局 args 派生自己的 Namespace。
- critic 的 CPU param backup 默认策略与 actor 不同。

设计选择：
- 若提供 `megatron_config_path`，actor 用 `parse_megatron_role_args(..., role="actor")`，critic 用 `role="critic"`。
- 若未提供独立 config，critic 用 `copy.deepcopy(args)`，并设置 `disable_param_buffers_cpu_backup=False`。
- actor 和 critic 分别调用 `allocate_train_group`。

Explain：
`create_training_models` 先把训练角色的参数空间拆开，再创建对应 RayTrainGroup。这样 role-specific Megatron 配置不会互相污染。

来源：slime/ray/placement_group.py L152-L188

Code：

```python
def create_training_models(args, pgs, rollout_manager, actor_cls=None):
    actor_args = args
    if args.megatron_config_path is not None:
        from slime.utils.arguments import parse_megatron_role_args

        actor_args = parse_megatron_role_args(args, args.megatron_config_path, role="actor")

    actor_model = allocate_train_group(
        args=actor_args,
        num_nodes=args.actor_num_nodes,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        pg=pgs["actor"],
        **actor_model_kwargs,
    )

    critic_model = None
    if args.use_critic:
        critic_args = (
            parse_megatron_role_args(args, args.megatron_config_path, role="critic")
            if args.megatron_config_path is not None
            else copy.deepcopy(args)
        )
        if args.megatron_config_path is None:
            critic_args.disable_param_buffers_cpu_backup = False

        critic_model = allocate_train_group(
            args=critic_args,
            num_nodes=args.critic_num_nodes,
            num_gpus_per_node=args.critic_num_gpus_per_node,
            pg=pgs["critic"],
            role="critic",
        )
```

代码逻辑：
- actor 默认使用全局 args。
- 有 Megatron role config 时，为 actor 解析独立 args。
- 创建 actor RayTrainGroup。
- 如果启用 critic，解析或复制 critic args。
- 未提供 role config 时，为 critic 打开 CPU param backup。
- 创建 critic RayTrainGroup。

为什么这样写：
- 角色参数分离让 actor/critic 可以有不同模型并行、优化器和 checkpoint 设置。
- deepcopy 避免 critic 修改污染 actor/global args。
- critic 没有独立 YAML 时开启 CPU backup，给 offload/恢复路径保留安全默认。

不变量与失败模式：
- `pgs["actor"]` 必须存在；启用 critic 时 `pgs["critic"]` 必须可用。
- `parse_megatron_role_args` 失败会阻止训练组创建。
- actor/critic 节点数与 PG 容量不一致会在后续 RayTrainGroup 分配时报错。

Comment：
PlacementGroup 模块同时承担了“资源绑定”和“角色参数拆分”的入口职责。

### 2.3 training models 初始化后统一 rollout id 并注入 manager

问题与约束：
- 训练恢复时所有 rank 必须从同一个 rollout id 对齐。
- actor 可能需要加载 reference 相关状态。
- 训练 actor/critic 后续需要调用 rollout manager。

设计选择：
- critic 和 actor 分别 `async_init`，收集各 rank 返回的 start rollout id。
- 断言所有 start rollout id 相同。
- 如果用户未指定 `args.start_rollout_id`，用返回值写入。
- 将 `rollout_manager` 注入 actor/critic train group。

Explain：
这一步把训练组初始化和 rollout 状态对齐串起来。只有所有 rank 对 checkpoint/rollout id 达成一致，后续数据加载和训练步才不会错位。

来源：slime/ray/placement_group.py L191-L212

Code：

```python
actor_start_rollout_ids = ray.get(
    actor_model.async_init(
        actor_args,
        role="actor",
        with_ref=actor_args.kl_coef != 0 or actor_args.use_kl_loss,
        ...
    )
)
if args.use_critic:
    start_rollout_ids = critic_start_rollout_ids
else:
    start_rollout_ids = actor_start_rollout_ids

assert len(set(start_rollout_ids)) == 1

if args.start_rollout_id is None:
    args.start_rollout_id = start_rollout_ids[0]

actor_model.set_rollout_manager(rollout_manager)
if args.use_critic:
    critic_model.set_rollout_manager(rollout_manager)
```

代码逻辑：
- actor group 异步初始化，并等待所有 rank 返回。
- 根据 critic 是否存在选择 start rollout id 来源。
- 检查返回 id 是否全一致。
- 未指定时写入 `args.start_rollout_id`。
- 将 rollout manager 引用设置到 actor/critic group。

为什么这样写：
- rank 间 rollout id 不一致意味着 checkpoint 或数据状态错位，必须立即失败。
- rollout manager 注入后，训练侧才能驱动 rollout 生成和数据交换。
- `with_ref` 根据 KL 相关配置决定是否在 actor init 中准备 reference 路径。

不变量与失败模式：
- `async_init` 返回的 rollout ids 必须非空且一致。
- critic 存在时当前逻辑使用 critic start id，调用者要保证 actor/critic checkpoint 对齐。
- `rollout_manager` 必须已创建，否则无法注入训练组。

Comment：
这一段是训练资源和 rollout 资源真正接线的地方。

## 3. RolloutManager 侧 PG 绑定

### 3.1 RolloutManager 自身无 GPU，但持有 rollout PG 视图

问题与约束：
- RolloutManager 是控制 actor，不应占用 GPU。
- 内部启动的 SGLang engines 需要 rollout PG 三元组来绑定 bundle。
- rollout 数据传输方式可能要求 Ray tensor transport。
- offload rollout 时，初始化后要立刻释放 rollout 权重显存。

设计选择：
- 创建 RolloutManager Ray actor 时设置 `num_cpus=1`、`num_gpus=0`。
- 将 rollout PG 三元组作为构造参数传入。
- `rollout_data_transport == "nixl"` 时启用 tensor transport。
- 初始化后根据需要推导 `num_rollout`、检查权重、执行 offload。

Explain：
RolloutManager 不直接跑 SGLang GPU 计算，它持有 PG 视图并在内部创建 rollout engines。这个 actor 是 rollout 子系统的控制平面。

来源：slime/ray/placement_group.py L220-L246

Code：

```python
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

    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = ray.get(rollout_manager.get_num_rollout_per_epoch.remote())
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
        assert args.num_rollout > 0

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="snapshot"))
        ray.get(rollout_manager.check_weights.remote(action="reset_tensors"))

    if args.offload_rollout:
        ray.get(rollout_manager.offload.remote())

    return rollout_manager, num_rollout_per_epoch
```

代码逻辑：
- 构造 Ray actor options。
- 可选开启 tensor transport。
- 创建 RolloutManager actor，并传入 args 与 rollout PG。
- 如果未指定总 rollout 数，从 manager 查询每 epoch 数并乘 epoch。
- 可选做权重一致性检查快照。
- offload rollout 时调用 manager 的 offload。
- 返回 manager 引用和每 epoch rollout 数。

为什么这样写：
- 控制 actor 不占 GPU，GPU 资源留给内部 SGLangEngine actors。
- rollout PG 传入 manager，使 engine 启动能沿用已排序的 bundle/GPU 映射。
- `num_rollout` 需要在训练模型创建前确定，因为训练 schedule 依赖它。
- offload 在初始化后立即执行，支持训练与 rollout 共置时让出显存。

不变量与失败模式：
- `pg` 必须是 rollout PG 三元组。
- `get_num_rollout_per_epoch` 返回值必须为正，否则 assert。
- offload 失败会在 `ray.get` 处暴露。

Comment：
RolloutManager 是无 GPU 控制器，但它掌握 rollout engine 的 PG 资源视图。

### 3.2 自动推导 `num_rollout` 依赖 DataSource 已可用

问题与约束：
- 用户可能没有显式传入 `--num-rollout`。
- 训练 schedule 需要总 rollout 数，必须在训练模型创建前确定。
- 推导要依赖 RolloutManager 内部的 data source。

设计选择：
- 当 `args.num_rollout is None` 时，调用远端 `get_num_rollout_per_epoch`。
- 将返回值乘以 `args.num_epoch` 写回 `args.num_rollout`。
- assert 推导结果大于 0。

Explain：
num rollout 自动推导发生在 RolloutManager 创建之后、training models 创建之前。它把 data source 的 epoch 长度转换成训练侧需要的全局 rollout 数。

来源：slime/ray/placement_group.py L232-L237

Code：

```python
num_rollout_per_epoch = None
if args.num_rollout is None:
    num_rollout_per_epoch = ray.get(rollout_manager.get_num_rollout_per_epoch.remote())
    args.num_rollout = num_rollout_per_epoch * args.num_epoch
    assert args.num_rollout > 0
```

代码逻辑：
- 初始化每 epoch rollout 数为 None。
- 未指定总 rollout 时，远程查询 RolloutManager。
- 乘以 epoch 数得到总 rollout。
- 检查结果为正。

为什么这样写：
- DataSource 的长度属于 rollout subsystem，训练入口不重复解析。
- 写回 args 让后续 Megatron/训练 schedule 读取统一字段。
- 正数 assert 防止空数据集或错误 epoch 配置继续运行。

不变量与失败模式：
- RolloutManager 必须已经完成 DataSource 初始化。
- `args.num_epoch` 必须为正。
- 远端查询失败或返回 0 会中断启动流程。

Comment：
这看起来是一个小字段，但它决定训练循环的全局步数边界。

### 3.3 global dataset resume 在 training models 初始化后加载

问题与约束：
- 使用 global dataset 时，恢复训练需要让数据集状态与 checkpoint rollout id 对齐。
- `start_rollout_id` 在 training models init 后才确定。
- 加载过早会缺少正确的 rollout id。

设计选择：
- 在 `create_training_models` 末尾，如果启用 `rollout_global_dataset`，调用 `rollout_manager.load(args.start_rollout_id - 1)`。

Explain：
global dataset 的持久化状态按 rollout id 对齐。源码等 actor/critic 初始化确定 `start_rollout_id` 后，再让 RolloutManager 加载前一个 rollout 的数据状态。

来源：slime/ray/placement_group.py L214-L215

Code：

```python
if args.rollout_global_dataset:
    ray.get(rollout_manager.load.remote(args.start_rollout_id - 1))
```

代码逻辑：
- 检查是否使用 global dataset。
- 调用 RolloutManager 的远端 load。
- 传入 `start_rollout_id - 1`。
- `ray.get` 等待加载完成。

为什么这样写：
- resume 时当前 start id 对应下一轮要生成的数据，已持久化状态通常停在上一轮。
- 等待加载完成后再进入训练，避免 prompt index 和 checkpoint 状态错位。
- 放在训练组 init 后，能使用已经对齐的 `args.start_rollout_id`。

不变量与失败模式：
- `args.start_rollout_id` 必须已经被设置。
- `rollout_manager.load` 必须能找到对应 id 的数据状态。
- 加载失败会在 `ray.get` 处暴露。

Comment：
这是 PG 编排和数据状态恢复的交叉点：资源 ready 后还要对齐数据游标。

## 4. Ray 可见设备环境

### 4.1 `ray_noset_visible_devices` 汇总多硬件 noset 开关

问题与约束：
- Ray 默认可能设置 CUDA/ROCR/TPU 等可见设备环境变量。
- Slime/SGLang/Megatron 需要在一些场景下自行管理 visible devices。
- 不同硬件 backend 的 Ray noset 环境变量名称不同。

设计选择：
- 将 CUDA、ROCR、Ascend、Habana、Neuron、TPU、Intel GPU 的 noset 环境变量列入统一列表。
- `ray_noset_visible_devices` 只要发现任一变量存在就返回 True。
- `add_default_ray_env_vars` 统一注入默认 Ray 环境变量。

Explain：
这个工具函数给上层 actor 编排一个统一判断：当前 Ray 是否被要求不要自动设置可见设备。后续 TrainGroup 和 SGLang actor 可以据此决定是否自行管理设备可见性。

来源：slime/ray/utils.py L16-L37

Code：

```python
NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
    "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
    "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
    "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
]

RAY_DEFAULT_ENV_VARS = {
    "RAY_USE_UVLOOP": "0",
}

def add_default_ray_env_vars(env_vars: dict[str, str] | None = None) -> dict[str, str]:
    return RAY_DEFAULT_ENV_VARS | (env_vars or {})

def ray_noset_visible_devices(env_vars=os.environ):
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)
```

代码逻辑：
- 定义多硬件 noset 环境变量列表。
- 定义默认 Ray 环境变量。
- `add_default_ray_env_vars` 合并默认值和调用方传入值。
- `ray_noset_visible_devices` 遍历列表，只要某个变量存在即返回 True。

为什么这样写：
- 多硬件 backend 共享同一个调度/actor 编排框架。
- 用列表集中维护变量名，避免各处硬编码。
- 默认禁用 uvloop 能规避 Ray async actor 的间歇性问题。

不变量与失败模式：
- 环境变量只要非空就视为启用，不解析具体布尔字符串。
- 新硬件 backend 需要把对应 Ray noset 变量加入列表。
- 调用方要把合并后的 env vars 传给 Ray runtime env 才会生效。

Comment：
PlacementGroup 给出资源位置，visible-devices 逻辑决定 actor 内部如何看见这些资源。
