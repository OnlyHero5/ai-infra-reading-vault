---
type: batch-doc
module: 09-EngineTopology
batch: "09"
doc_type: walkthrough
title: "EngineTopology · 源码走读"
tags:
  - slime/batch/09
  - slime/module/engine-topology
  - slime/doc/walkthrough
updated: 2026-07-05
---

# EngineTopology · 源码走读

## 1. RolloutManager 入口与配置解析

### 1.1 RolloutManager 构造阶段启动并等待 rollout engines

问题与约束：
- rollout engine 是 Ray actor 和 SGLang server 组合，初始化是异步的。
- data source、rollout function 等 Python 组件也要在 manager 构造阶段加载。
- 训练生成前必须保证 engine init 完成，否则后续 HTTP 请求会打到未就绪服务。

设计选择：
- 非 `debug_train_only` 时先初始化 HTTP client，再调用 `start_rollout_servers(args, pg)`。
- `start_rollout_servers` 返回 `servers` 和 `rollout_init_handles`；构造末尾统一 `ray.get` 等待。

Explain：
`RolloutManager.__init__` 把拓扑启动与本地函数加载放在同一构造流程里，但最后用 `ray.get(rollout_init_handles)` 设置同步边界：server 可以异步创建，进入 generate 前必须 ready。

来源：slime/ray/rollout.py L430-L454

Code：

```python
rollout_init_handles: list[Any] = []
if self.args.debug_train_only:
    self.servers: dict[str, Any] = {}
else:
    init_http_client(args)
    self.servers, rollout_init_handles = start_rollout_servers(args, pg)

data_source_cls = load_function(self.args.data_source_path)
self.data_source = data_source_cls(args)

self.generate_rollout = load_function(self.args.rollout_function_path)
self.eval_generate_rollout = load_function(self.args.eval_function_path)

if rollout_init_handles:
    ray.get(rollout_init_handles)
```

代码逻辑：
- debug train-only 模式跳过 rollout server。
- 正常模式初始化 HTTP client 并启动 rollout servers。
- 加载 data source、generate/eval rollout function 和可选自定义函数。
- 如果有 engine init handle，阻塞等待全部完成。

为什么这样写：
- Router/engine 启动较慢，先发起异步 init 可以和 Python 组件加载重叠。
- 等待边界放在构造末尾，保证 manager 对外可用时 topology 已经 ready。
- `self.servers` 由 model name 映射到 RolloutServer，给后续多模型 rollout 路由使用。

不变量与失败模式：
- `rollout_init_handles` 中任一 Ray task 失败会在 `ray.get` 处暴露。
- `debug_train_only` 下 `self.servers` 为空，后续不能走真实 rollout server 请求路径。
- `init_http_client(args)` 必须在 server 请求前完成。

Comment：
这里的关键是“异步启动，同步进入使用阶段”，不是简单顺序构造。

### 1.2 `SglangConfig.from_yaml` 把 YAML 拓扑变成 model/group 对象

问题与约束：
- 高级拓扑需要用 YAML 描述多个 model、多个 server group 和 group 级 overrides。
- 旧配置可能使用 `engine_groups` 字段名。
- 解析阶段不能直接依赖 args 默认值，因为 model/group 级默认值还需要后续 resolve。

设计选择：
- 要求顶层存在 `sglang` 列表。
- 每个 model entry 构造 `ModelConfig`，其中 `server_groups` 来自 `server_groups` 或兼容字段 `engine_groups`。
- 只保留 YAML 显式字段，默认值交给 `ModelConfig.resolve(args)` 处理。

Explain：
`from_yaml` 只负责结构化解析：把外部 YAML 转成 `SglangConfig(models=[...])`。它不计算 router、port、GPU offset，也不启动 actor。

来源：slime/backends/sglang_utils/sglang_config.py L157-L180

Code：

```python
@staticmethod
def from_yaml(path: str) -> "SglangConfig":
    with open(path) as f:
        data = yaml.safe_load(f)

    assert "sglang" in data, (
        f"sglang config must have a 'sglang' key, got {list(data.keys())}. "
        f"Wrap your server_groups inside a model entry under 'sglang'."
    )
    models = []
    for m in data["sglang"]:
        raw_groups = m.get("server_groups") or m.get("engine_groups") or []
        groups = [ServerGroupConfig(**g) for g in raw_groups]
        models.append(
            ModelConfig(
                name=m["name"],
                model_path=m.get("model_path"),
                num_gpus_per_engine=m.get("num_gpus_per_engine"),
                server_groups=groups,
                update_weights=m.get("update_weights"),
            )
        )
    return SglangConfig(models=models)
```

代码逻辑：
- 读取 YAML 并检查顶层 key。
- 遍历 `data["sglang"]` 中的 model entry。
- 每个 group dict 转成 `ServerGroupConfig`。
- 每个 model dict 转成 `ModelConfig`。
- 返回包含全部 model 的 `SglangConfig`。

为什么这样写：
- 顶层 `sglang` 强制把多模型拓扑包成统一列表，便于后续逐 model 启动 router 和 groups。
- `engine_groups` 兼容旧配置，降低迁移成本。
- 解析和默认值补齐分离，避免 YAML loader 依赖完整训练 args。

不变量与失败模式：
- YAML 缺少顶层 `sglang` 会 assert。
- 每个 model 必须有 `name`。
- group dict 字段必须匹配 `ServerGroupConfig`，否则 dataclass 构造会失败。

Comment：
这一步只建立“想要什么拓扑”的内存表示，真正的 GPU/port/Ray actor 分配还在后面。

### 1.3 `ModelConfig.resolve` 补齐 TP、模型路径和权重更新语义

问题与约束：
- model 级配置和 group 级配置都可能省略 `num_gpus_per_engine` 或 `model_path`。
- 同一个 model 内多个 server group 必须服务同一模型权重，否则 update weights 语义不清。
- actor/ref/reward 等多模型场景下，有些模型不应接收训练权重更新。

设计选择：
- 按 group 值、model 值、args 值的优先级补齐 `num_gpus_per_engine` 和 `model_path`。
- 断言同一 model 的所有 groups 共享同一 `model_path`。
- 未显式设置 `update_weights` 时，根据 effective model path 是否等于 `args.hf_checkpoint` 推断。

Explain：
`resolve` 把 YAML 中的局部配置变成启动 SGLang engine 所需的完整 group 配置，并决定该 model 是否参与训练权重同步。

来源：slime/backends/sglang_utils/sglang_config.py L68-L100

Code：

```python
def resolve(self, args) -> None:
    default_gpus_per_engine = self.num_gpus_per_engine or args.rollout_num_gpus_per_engine
    default_model_path = self.model_path or args.hf_checkpoint
    for g in self.server_groups:
        if g.num_gpus_per_engine is None:
            g.num_gpus_per_engine = default_gpus_per_engine
        if "model_path" not in g.overrides:
            g.overrides["model_path"] = default_model_path

    if self.server_groups:
        model_paths = {g.overrides["model_path"] for g in self.server_groups}
        assert len(model_paths) == 1, (
            f"Model '{self.name}' has server groups with different model_path values: "
            f"{model_paths}. All server groups within a model must use the same model_path."
        )
        effective_model_path = model_paths.pop()
    else:
        effective_model_path = default_model_path

    if self.update_weights is None:
        if effective_model_path != args.hf_checkpoint:
            logger.warning(...)
            self.update_weights = False
        else:
            self.update_weights = True
```

代码逻辑：
- 计算默认 TP 和默认 model path。
- 逐 group 补齐缺省 TP。
- 若 overrides 没有 `model_path`，注入默认 model path。
- 检查同 model 内所有 group 的 model path 是否一致。
- 未设置 `update_weights` 时按 effective model path 推断。

为什么这样写：
- group 级 override 支持 prefill/decode 使用不同 TP。
- 同一 model 的 groups 要共享权重更新来源，否则 RolloutServer 无法用一个 `update_weights` 语义覆盖所有 group。
- ref/reward 等冻结模型通常使用不同 checkpoint，默认不接收 actor 权重更新更安全。

不变量与失败模式：
- 同一 model 内多个 model path 会 assert。
- `args.rollout_num_gpus_per_engine` 和 `args.hf_checkpoint` 必须存在。
- 显式 `update_weights` 会覆盖推断，调用者需要自行保证语义正确。

Comment：
这一步把“配置拓扑”转成“可启动拓扑”，尤其把权重同步边界定下来。

## 2. Router 与 ServerGroup 构造

### 2.1 `_start_router` 按 model 启动 PD-aware router

问题与约束：
- 每个 model 需要一个 router 入口，multi-model 时不能复用同一端口。
- PD 拓扑下 prefill/decode worker 会有不同角色，router 需要启用 PD disaggregation。
- RDMA 或高负载下 decode worker 的瞬时超时不应被 router circuit breaker 误判为 dead。

设计选择：
- 如果已有 `args.sglang_router_ip` 且不是强制新 router，则复用。
- 否则选择 host IP 和可用端口，构造 `RouterArgs`。
- PD 模式下设置 `pd_disaggregation=True` 和 `disable_circuit_breaker=True`。
- 关闭 router 自带 health check，由 Slime 自己管理健康。

Explain：
router 是 model 级入口。`force_new` 让第二个及之后的 model 拥有独立 router；`has_pd_disaggregation` 则把 router 切到 prefill/decode-aware 模式。

来源：slime/ray/rollout.py L1019-L1070

Code：

```python
def _start_router(args, *, has_pd_disaggregation: bool = False, force_new: bool = False) -> tuple[str, int]:
    if not force_new and args.sglang_router_ip is not None:
        return args.sglang_router_ip, args.sglang_router_port

    router_ip = _wrap_ipv6(get_host_info()[1])
    if force_new:
        router_port = find_available_port(random.randint(3000, 4000))
    else:
        router_port = args.sglang_router_port
        if router_port is None:
            router_port = find_available_port(random.randint(3000, 4000))

    router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)
    router_args.host = router_ip
    router_args.port = router_port
    router_args.prometheus_port = find_available_port(random.randint(4000, 5000))
    router_args.log_level = "warn"
    router_args.request_timeout_secs = args.sglang_router_request_timeout_secs

    if has_pd_disaggregation:
        router_args.pd_disaggregation = True
        router_args.disable_circuit_breaker = True

    router_args.disable_health_check = True

    process = multiprocessing.Process(target=run_router, args=(router_args,))
    process.daemon = True
    process.start()
    time.sleep(3)
    assert process.is_alive()
    return router_ip, router_port
```

代码逻辑：
- 可复用已有 router 时直接返回。
- 否则确定 router host 和 port。
- 从 CLI args 派生 router args，并填入 prometheus port、日志级别和超时。
- PD 模式下打开 PD router 并关闭 circuit breaker。
- 用 daemon process 启动 router，等待 3 秒并检查进程存活。

为什么这样写：
- router 生命周期独立于 Ray engine actor，用独立进程更贴近 SGLang router 使用方式。
- multi-model 强制新端口，避免多个 model 混到同一个 router。
- Slime 有自己的 engine health/recover 逻辑，关闭 router health check 避免双重判断冲突。

不变量与失败模式：
- 启动 3 秒后 router process 必须仍然存活。
- 用户传入的 router IP/port 只有在 `force_new=False` 时会复用。
- PD 模式关闭 circuit breaker 会降低 router 自动隔离故障 worker 的能力，依赖 Slime monitor/recover 补位。

Comment：
Router 是拓扑的 model 级入口，ServerGroup 是 worker 级注册单元。

### 2.2 `_make_group` 计算 engine 数、GPU 偏移和 offload 语义

问题与约束：
- YAML 中 group 只描述总 GPU 数、worker type 和 overrides；启动时还要映射到 placement group 的具体 GPU slice。
- 多节点 TP 下一个 engine 可能跨多个节点，engine 个数不能简单等于 GPU 数。
- rollout GPU 可能与 Megatron 训练 GPU 共置，需要判断该 group 是否要 offload。

设计选择：
- 用 `num_gpus // min(gpus_per_engine, args.num_gpus_per_node)` 计算 engine 数。
- 用闭包变量 `engine_offset/gpu_offset` 为 group 分配全局 rank 和 PG GPU 起点。
- 根据 rollout PG offset 与 Megatron GPU 范围计算 `needs_offload`。
- 构造 `ServerGroup` dataclass，并推进 offsets。

Explain：
`_make_group` 是配置对象到运行时 ServerGroup 的转换点。它把 group 的相对资源需求落到 placement group 中的连续 GPU 区间，并把 router 地址、model path、worker type 和 overrides 绑定到 group。

来源：slime/ray/rollout.py L1133-L1169

Code：

```python
def _make_group(group_cfg, router_ip, router_port, overrides_extra=None):
    nonlocal engine_offset, gpu_offset
    gpus_per_engine = group_cfg.num_gpus_per_engine
    num_gpu_per_engine_local = min(gpus_per_engine, args.num_gpus_per_node)
    num_engines = group_cfg.num_gpus // num_gpu_per_engine_local

    group_abs_start = rollout_pg_offset + gpu_offset
    needs_offload = args.offload_rollout and group_abs_start < megatron_num_gpus
    overrides = dict(group_cfg.overrides)
    if overrides_extra:
        for k, v in overrides_extra.items():
            overrides.setdefault(k, v)
    if args.offload_rollout and not needs_offload:
        overrides.setdefault("enable_memory_saver", False)

    group = ServerGroup(
        args=args,
        pg=pg,
        all_engines=[None] * num_engines if group_cfg.worker_type != "placeholder" else [],
        num_gpus_per_engine=gpus_per_engine,
        num_new_engines=0,
        worker_type=group_cfg.worker_type,
        rank_offset=engine_offset,
        gpu_offset=gpu_offset,
        sglang_overrides=overrides,
        needs_offload=needs_offload,
        model_path=overrides.get("model_path", args.hf_checkpoint),
        router_ip=router_ip,
        router_port=router_port,
    )
    engine_offset += num_engines
    gpu_offset += group_cfg.num_gpus
    return group
```

代码逻辑：
- 取 group 的 TP size。
- 计算本地每 engine GPU 数和 engine 数。
- 根据 group 在 PG 中的绝对起点判断是否与 Megatron GPU 重叠。
- 合并 group overrides 和额外 overrides。
- placeholder group 不创建 engine slots。
- 构造 ServerGroup，并推进 engine/gpu offset。

为什么这样写：
- engine rank 和 GPU offset 需要跨 group 累加，才能保证多个 group 不抢同一 PG bundle。
- 多节点 TP 用 `min(gpus_per_engine, num_gpus_per_node)` 算本地 GPU 步幅，适配跨节点 engine。
- offload 只对共置训练 GPU 的 rollout group 开启，非重叠 group 可以关闭 memory saver。

不变量与失败模式：
- `group_cfg.num_gpus` 必须能被本地 GPU 步幅整除，否则 engine 数会截断。
- `gpu_offset` 推进必须和 group GPU 总数一致。
- `overrides_extra` 使用 `setdefault`，不会覆盖 YAML 中显式设置。

Comment：
这一步决定了 ServerGroup 的资源边界：多少 engine、从哪个 GPU slice 开始、是否参与 offload。

### 2.3 `ServerGroup.start_engines` 创建 Ray actor 并分配端口

问题与约束：
- 一个 ServerGroup 可能包含多个 SGLangEngine Ray actor。
- 不同 group 在同一节点上启动时端口不能冲突。
- placement group 已经给出 bundle 和 GPU 重排信息，actor 必须落到对应 bundle。

设计选择：
- 对 `all_engines` 中的空槽创建 `ray.remote(SGLangEngine)` actor。
- 使用 `PlacementGroupSchedulingStrategy` 锁定 bundle index。
- 通过 `port_cursors` 给新 engine 分配地址、server port、NCCL port 等。
- 异步调用 `engine.init.remote(..., router_ip, router_port)`，把 init handles 返回给调用方等待。

Explain：
ServerGroup 负责把逻辑 group 实例化为 Ray actors。它不等待 engine 健康，而是返回 init handles；这样多个 group 可以并发启动，并共享端口游标避免冲突。

来源：slime/ray/rollout.py L137-L246

Code：

```python
def start_engines(self, port_cursors: dict[int, int] | None = None) -> tuple[list, dict[int, int]]:
    if port_cursors is None:
        port_cursors = {}
    if self.args.debug_train_only or self.worker_type == "placeholder":
        self.num_new_engines = 0
        return [], port_cursors

    num_gpu_per_engine = min(self.num_gpus_per_engine, self.args.num_gpus_per_node)
    pg, reordered_bundle_indices, reordered_gpu_ids = self.pg
    validate_server_group_gpu_indices(...)

    RolloutRayActor = ray.remote(SGLangEngine)

    rollout_engines = []
    for i in range(len(self.all_engines)):
        if self.all_engines[i] is not None:
            continue

        global_rank = self.rank_offset + i
        gpu_index = self.gpu_offset + i * num_gpu_per_engine
        base_gpu_id = int(reordered_gpu_ids[gpu_index])

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[gpu_index],
        )

        rollout_engine = RolloutRayActor.options(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            scheduling_strategy=scheduling_strategy,
            runtime_env={"env_vars": add_default_ray_env_vars(env_vars)},
        ).remote(
            self.args,
            rank=global_rank,
            worker_type=self.worker_type,
            base_gpu_id=base_gpu_id,
            sglang_overrides=self.sglang_overrides,
            num_gpus_per_engine=self.num_gpus_per_engine,
        )
        rollout_engines.append((global_rank, rollout_engine))
        self.all_engines[i] = rollout_engine

    base_port = max(port_cursors.values()) if port_cursors else 15000
    addr_and_ports, port_cursors = _allocate_rollout_engine_addr_and_ports_normal(...)

    init_handles = [
        engine.init.remote(
            **(addr_and_ports[rank]),
            router_ip=self.router_ip,
            router_port=self.router_port,
        )
        for rank, engine in rollout_engines
    ]
    return init_handles, port_cursors
```

代码逻辑：
- 初始化端口游标。
- debug 或 placeholder group 直接返回空 handles。
- 校验 group 的 GPU index 合法性。
- 遍历空 engine slot，计算 global rank、GPU index 和 base GPU id。
- 按 PG bundle 创建 Ray actor，并写回 `self.all_engines`。
- 为新 actor 分配端口，发起异步 init。

为什么这样写：
- actor 创建和 init 分离，便于 recover 时只补死掉的空槽。
- `port_cursors` 在 group 间传递，避免同节点端口重用。
- `worker_type` 和 overrides 在 actor 构造时传入，SGLang server args 能按 regular/prefill/decode/encoder 分化。

不变量与失败模式：
- `validate_server_group_gpu_indices` 必须通过，否则 group 资源切片不合法。
- `addr_and_ports` 必须为每个新 rank 提供 port、nccl port 和 dist init address。
- `self.all_engines[i]` 非空的 slot 不会重复创建 actor。

Comment：
这段是拓扑真正落地的地方：配置中的 group 变成具体 Ray actor 和端口集合。

## 3. Group 启动路径

### 3.1 非 EPD 路径一次遍历所有 server groups

问题与约束：
- regular 或 PD 拓扑没有 encoder 依赖，不需要先拿 encoder URL。
- 多个 group 仍要共享 `port_cursors`，避免端口冲突。
- `start_rollout_servers` 本身不应阻塞等待健康，由 RolloutManager 统一等待。

设计选择：
- 遍历 `model_cfg.server_groups`，每个 group 通过 `_make_group` 构造并调用 `start_engines`。
- 把所有 init handles 合并进 `pending_init_handles`。
- 创建 `RolloutServer` 并注册到 `servers[model_cfg.name]`，最后把 model router 表写到 args。

Explain：
无 encoder disaggregation 时，group 启动是一趟线性流程：配置顺序决定 group 创建顺序，端口游标沿 group 传递，init handles 汇总后返回给 RolloutManager。

来源：slime/ray/rollout.py L1206-L1228

Code：

```python
else:
    all_init_handles: list = []
    for group_cfg in model_cfg.server_groups:
        group = _make_group(group_cfg, router_ip, router_port)
        handles, port_cursors = group.start_engines(port_cursors)
        all_init_handles.extend(handles)
        server_groups.append(group)

    pending_init_handles.extend(all_init_handles)

servers[model_cfg.name] = RolloutServer(
    server_groups=server_groups,
    router_ip=router_ip,
    router_port=router_port,
    model_name=model_cfg.name,
    update_weights=model_cfg.update_weights,
)

args.sglang_model_routers = {name: (srv.router_ip, srv.router_port) for name, srv in servers.items()}

return servers, pending_init_handles
```

代码逻辑：
- 为当前 model 初始化 `all_init_handles`。
- 每个 group 构造 ServerGroup 并启动 engines。
- 收集 handles 和 server group。
- 用这些 groups 构造 RolloutServer。
- 将 model name 到 router 地址的映射写回 args。
- 返回 servers 和 pending init handles。

为什么这样写：
- 启动逻辑保持 group 顺序，便于 PD YAML 中 prefill/decode 拓扑直观映射。
- 不在这里 `ray.get`，让上层可以统一等待全部 model/group init。
- `args.sglang_model_routers` 给 custom rollout function 做多模型路由选择。

不变量与失败模式：
- `model_cfg.update_weights` 必须在此前 resolve 完成。
- `port_cursors` 必须沿 group 传递。
- 返回前 `servers` 已注册，但其中 engine 可能仍在异步 init。

Comment：
非 EPD 是最常见路径：group 之间没有数据依赖，只有资源和端口游标依赖。

### 3.2 EPD 路径先同步 encoder，再启动语言 worker

问题与约束：
- EPD 拓扑中 LLM worker 需要 encoder URLs 才能启动完整 server args。
- encoder URL 只有 encoder engines init 完成后才能获取。
- decode group 不需要注入 encoder URLs，视觉特征在 prefill/regular 阶段完成。

设计选择：
- Phase 1 只启动 `worker_type == "encoder"` 的 groups，并 `ray.get(handles)` 等待 ready。
- 从 encoder engines 读取 URL。
- Phase 2 启动非 encoder groups；对 prefill/regular 注入 `language_only=True` 和 `encoder_urls`。
- 非 encoder init handles 仍延迟到 RolloutManager 统一等待。

Explain：
EPD 有真实的数据依赖：语言 worker 的 server args 包含 encoder URLs。源码因此把 encoder group 的 init 放成同步阶段，其余 groups 仍保持异步 init。

来源：slime/ray/rollout.py L1171-L1205

Code：

```python
if has_epd:
    encoder_urls: list[str] = []
    for group_cfg in model_cfg.server_groups:
        if group_cfg.worker_type != "encoder":
            continue
        group = _make_group(group_cfg, router_ip, router_port)
        handles, port_cursors = group.start_engines(port_cursors)
        if handles:
            ray.get(handles)
        urls = ray.get([e.get_url.remote() for e in group.engines])
        encoder_urls.extend(u for u in urls if u is not None)
        server_groups.append(group)

    non_encoder_handles: list = []
    for group_cfg in model_cfg.server_groups:
        if group_cfg.worker_type == "encoder":
            continue
        overrides_extra = {}
        if encoder_urls and group_cfg.worker_type in ("prefill", "regular"):
            overrides_extra["language_only"] = True
            overrides_extra["encoder_urls"] = encoder_urls
        group = _make_group(group_cfg, router_ip, router_port, overrides_extra=overrides_extra)
        handles, port_cursors = group.start_engines(port_cursors)
        non_encoder_handles.extend(handles)
        server_groups.append(group)

    pending_init_handles.extend(non_encoder_handles)
```

代码逻辑：
- 第一轮只处理 encoder groups。
- encoder group 启动后立即等待 handles。
- 从 encoder engines 获取 URL，并过滤 None。
- 第二轮跳过 encoder groups。
- 对 prefill/regular 注入 language-only 和 encoder URLs。
- 启动非 encoder groups，并把 handles 加到 pending 列表。

为什么这样写：
- encoder URL 是非 encoder worker 启动参数，不能异步缺省。
- `setdefault` 语义在 `_make_group` 中保留 YAML 显式 overrides 优先级。
- decode 不注入 encoder URLs，避免 decode worker 承担不需要的多模态依赖。

不变量与失败模式：
- encoder init 失败会在 Phase 1 的 `ray.get` 暴露。
- 如果 encoder URLs 为空，非 encoder group 不会注入相关 overrides。
- EPD 配置中 worker type 必须和后续 SGLang server args 兼容。

Comment：
EPD 是唯一需要先同步部分 group 的拓扑，因为 URL 是后续 worker 的启动参数。

## 4. 恢复与文档约束

### 4.1 `RolloutServer.recover` 按 group 补齐 dead engines

问题与约束：
- 健康监控可能把某些 engine slot 置为 dead，需要只恢复缺失 actor。
- PD/EPD 中不同 group 有不同 worker type、GPU offset 和 offload 策略，恢复时必须保留这些 group 属性。
- 共置 offload 的新 engine 需要重新处理显存占用和权重状态。

设计选择：
- recover 前记录每个 group 的 dead indices。
- 对所有 group 调 `start_engines(port_cursors)`，并发启动缺失 actor。
- 等待 init 后，对需要 offload 的新 engine release/resume memory；可更新模型等待后续权重同步，冻结模型按 model path 从磁盘恢复。

Explain：
恢复逻辑复用 ServerGroup 的启动入口，而不是单独写一套 actor 创建逻辑。这样 worker type、router、overrides、rank/gpu offset 都沿用原 group 状态。

来源：slime/ray/rollout.py L340-L381

Code：

```python
def recover(self):
    dead_per_group = [[i for i, engine in enumerate(g.all_engines) if engine is None] for g in self.server_groups]

    all_handles = []
    port_cursors: dict[int, int] = {}
    for g in self.server_groups:
        handles, port_cursors = g.start_engines(port_cursors)
        all_handles.extend(handles)
    if all_handles:
        ray.get(all_handles)

    release_handles = []
    updatable_new_engines = []
    non_updatable_groups_engines: list[tuple[str, list]] = []
    for g, dead_indices in zip(self.server_groups, dead_per_group, strict=True):
        assert g.num_new_engines == len(dead_indices), "num_new_engines does not match dead_indices length"
        if g.needs_offload and dead_indices:
            new_engines = [g.all_engines[i] for i in dead_indices]
            release_handles.extend(engine.release_memory_occupation.remote() for engine in new_engines)
            if self.update_weights:
                updatable_new_engines.extend(new_engines)
            elif g.model_path:
                non_updatable_groups_engines.append((g.model_path, new_engines))

    if release_handles:
        ray.get(release_handles)
        all_resume_engines = updatable_new_engines[:]
        for _model_path, engines in non_updatable_groups_engines:
            all_resume_engines.extend(engines)
        if all_resume_engines:
            ray.get(
                [
                    engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS])
                    for engine in all_resume_engines
                ]
            )
```

代码逻辑：
- 记录每个 group 中为 None 的 engine index。
- 遍历 groups，调用 `start_engines` 补缺失 actor。
- 等待所有新 engine init 完成。
- 对每个 group 校验新建 engine 数与 dead index 数一致。
- 需要 offload 的新 engine 先 release memory occupation。
- 根据 update weights 语义收集可更新或非可更新 engine，再 resume weights memory。

为什么这样写：
- 使用 group 自身的 `start_engines` 保持拓扑一致。
- group 间共享 `port_cursors`，恢复时仍避免端口冲突。
- offload 状态和权重恢复只作用于新建 engine，减少对健康 engine 的扰动。

不变量与失败模式：
- `g.num_new_engines` 必须等于之前记录的 dead slot 数。
- `dead_per_group` 与 `server_groups` zip 使用 `strict=True`，group 数量不能变化。
- release/resume Ray task 失败会在 `ray.get` 处暴露。

Comment：
recover 说明拓扑不是一次性启动产物，而是 ServerGroup 状态可以被复用来重建缺失 worker。

### 4.2 PD 文档约束与代码路径对应

问题与约束：
- 用户有两条配置路径：简单 `--prefill-num-servers` 和高级 `--sglang-config`。
- 复杂 rollout 需要分别配置 prefill/decode TP、memory 和 per-group overrides。
- PD 模式下不能在同一 model entry 混用 regular 与 prefill/decode workers。

设计选择：
- 文档建议简单场景使用 `--prefill-num-servers`，生产拓扑使用 `--sglang-config`。
- 高级 YAML 用 `server_groups` 显式列出 prefill/decode groups。
- 操作约束要求 `--rollout-num-gpus` 等于配置总 GPU，multi-turn agent 使用 router session affinity。

Explain：
PD 文档描述的是用户可见配置面；源码中的 `_resolve_sglang_config`、`SglangConfig.from_yaml`、`_start_router(has_pd_disaggregation=True)` 和 ServerGroup 启动链路就是这些配置的实现路径。

来源：slime/docs/en/advanced/pd-disaggregation.md L17-L80

Code：

```text
Configuration Paths

Simple Path: --prefill-num-servers
--prefill-num-servers 1

Advanced Path: --sglang-config
sglang:
  - name: actor
    update_weights: true
    server_groups:
      - worker_type: prefill
        num_gpus: 4
        num_gpus_per_engine: 2
        overrides:
          chunked_prefill_size: 8192
      - worker_type: decode
        num_gpus: 12
        num_gpus_per_engine: 4
        overrides:
          mem_fraction_static: 0.88

Operational Notes:
- Keep rollout-num-gpus equal to the total GPUs described by the SGLang config.
- Do not mix regular workers with prefill/decode workers inside the same model entry.
```

代码逻辑：
- 简单路径只给出 prefill server 数。
- 高级路径用 `sglang` 顶层列表描述 actor model。
- `server_groups` 分别声明 prefill 和 decode worker。
- 每个 group 可以设置 GPU 总数、每 engine GPU 数和 overrides。
- 操作说明约束总 GPU 和 worker type 组合。

为什么这样写：
- 简单参数覆盖常见 PD split，减少配置负担。
- YAML 支持异构 TP、内存比例和多模型拓扑，适合生产 rollout。
- 禁止混合 regular 与 prefill/decode，能让 router 的 worker 注册和路由语义保持清晰。

不变量与失败模式：
- `--rollout-num-gpus` 与 YAML 总 GPU 不一致会在源码配置解析路径 assert。
- 同一 model 混合 regular 与 PD worker 会让 router/worker type 语义冲突。
- multi-turn 场景如果不使用 session affinity，prefix cache locality 可能下降。

Comment：
文档给的是外部约束；前面九个源码块说明这些约束如何落成 router、group、actor 和恢复状态。
