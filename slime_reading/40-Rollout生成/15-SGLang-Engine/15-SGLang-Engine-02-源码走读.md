---
type: batch-doc
module: 15-SGLang-Engine
batch: "15"
doc_type: walkthrough
title: "SGLang Engine · 源码走读"
tags:
  - slime/batch/15
  - slime/module/sglang-engine
  - slime/doc/walkthrough
updated: 2026-07-05
---

# SGLang Engine · 源码走读

> 走读主线：`SGLangEngine` 是 Slime rollout 侧对 SGLang HTTP server 的 Ray actor 适配层。它负责计算每个 engine 的 `ServerArgs`、启动或接入外部 SGLang、向 router 注册、把权重同步和运行控制封装成 HTTP 请求，并在本地 engine 模式下负责 shutdown。

---

## 1. 启动前：GPU 映射与 SGLang 子进程

### 1.1 get_base_gpu_id 计算每个 engine 的起始 GPU

问题与约束：
- rollout engine 可能和 actor colocate，也可能放在 actor 之后的 GPU 段；同一个节点上多个 engine 还要按 rank 切分 GPU。

设计选择：
- `get_base_gpu_id` 根据 `colocate` 分支计算物理起始 GPU；`_to_local_gpu_id` 再把物理 GPU id 映射到当前 `CUDA_VISIBLE_DEVICES` 下的本地 id。

Explain：
colocate 时从 `rank * num_gpus` 对节点 GPU 数取模；非 colocate 时先跳过 actor 占用的 GPU 数，再为 rollout engine 取模。

来源：slime/backends/sglang_utils/sglang_engine.py L24-L49

Code：

```python
def get_base_gpu_id(args, rank):
    num_gpus = min(args.num_gpus_per_node, args.rollout_num_gpus_per_engine)
    if args.colocate:
        start_index = (rank * num_gpus) % args.num_gpus_per_node
    else:
        num_actor_gpus = 0 if args.debug_rollout_only else args.actor_num_gpus_per_node * args.actor_num_nodes
        start_index = (num_actor_gpus + rank * num_gpus) % args.num_gpus_per_node
    return start_index


def _to_local_gpu_id(physical_gpu_id: int) -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return physical_gpu_id
    visible = [int(x) for x in cvd.split(",") if x.strip() != ""]
    if physical_gpu_id in visible:
        return visible.index(physical_gpu_id)
    if 0 <= physical_gpu_id < len(visible):
        return physical_gpu_id
    raise RuntimeError(
        f"GPU id {physical_gpu_id} is not valid under CUDA_VISIBLE_DEVICES={cvd}. "
        f"Expected one of {visible} (physical) or 0..{len(visible)-1} (local)."
    )
```

代码逻辑：
- 先取单 engine 实际需要的 GPU 数。
- colocate 模式按 rollout rank 直接切片。
- 非 colocate 模式先预留 actor GPU。
- `CUDA_VISIBLE_DEVICES` 存在时做物理 id 到本地 id 映射。
- 映射失败直接抛 RuntimeError。

为什么这样写：
- Ray actor 进程内的可见 GPU 可能已经被重映射，传给 SGLang 的 `base_gpu_id` 必须是本地 id。
- colocate 与非 colocate 的资源布局不同，不能共用一个偏移公式。

不变量与失败模式：
- `rollout_num_gpus_per_engine` 和 `num_gpus_per_node` 必须能表达真实布局。
- 如果物理 GPU 不在 `CUDA_VISIBLE_DEVICES` 中，初始化会失败。
- debug rollout only 下 actor GPU 预留为 0。

Comment：
这段决定了 SGLang engine 会绑定哪一段 GPU，是 rollout 资源隔离的入口。

### 1.2 launch_server_process 用 spawn 启动本地 SGLang

问题与约束：
- SGLang server 是独立进程；普通 generate server 和 encoder-only server 的启动入口不同；多节点 engine 只有 node 0 需要等待 HTTP 健康检查。

设计选择：
- encoder-only 时委托 SGLang encode server 的 `launch_server_process`；普通模式使用 `multiprocessing.Process(target=launch_server)`，强制 spawn，并在 node 0 轮询 `/health_generate`。

Explain：
`_wait_server_healthy` 每 2 秒请求一次 `/health_generate`，如果子进程提前退出则抛异常，避免 RolloutManager 继续等待一个已死进程。

来源：slime/backends/sglang_utils/sglang_engine.py L52-L99

Code：

```python
def launch_server_process(server_args: ServerArgs) -> multiprocessing.Process:
    if getattr(server_args, "encoder_only", False):
        from sglang.srt.disaggregation.encode_server import launch_server_process as sglang_launch_server_process

        return sglang_launch_server_process(
            server_args,
            start_method="spawn",
            wait_for_server=True,
        )

    from sglang.srt.entrypoints.http_server import launch_server

    multiprocessing.set_start_method("spawn", force=True)
    server_args.host = server_args.host.strip("[]")
    p = multiprocessing.Process(target=launch_server, args=(server_args,))
    p.start()

    if getattr(server_args, "node_rank", 0) != 0:
        return p

    _wait_server_healthy(
        base_url=server_args.url(),
        api_key=server_args.api_key,
        is_process_alive=lambda: p.is_alive(),
    )

    return p


def _wait_server_healthy(base_url, api_key, is_process_alive):
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {api_key}",
    }

    with requests.Session() as session:
        while True:
            try:
                response = session.get(f"{base_url}/health_generate", headers=headers)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass

            if not is_process_alive():
                raise Exception("Server process terminated unexpectedly.")

            time.sleep(2)
```

代码逻辑：
- encoder-only 分支复用 SGLang disaggregation encode server 启动器。
- 普通分支强制使用 spawn。
- 去掉 host 外层 IPv6 方括号后启动 HTTP server。
- 非 node 0 直接返回进程对象。
- node 0 阻塞等待 `/health_generate`。

为什么这样写：
- spawn 避免 fork 已初始化的 CUDA/Ray 运行时状态。
- 多节点 SGLang 的非 node 0 不暴露同一个 HTTP 健康入口。
- 启动失败要尽早冒泡给 Ray 调用方。

不变量与失败模式：
- node 0 的 `/health_generate` 必须最终返回 200。
- 子进程提前退出会抛异常。
- encoder-only 路径依赖 SGLang 自己的 encode server 入口可用。

Comment：
本地 engine 模式下，Slime 只托管 SGLang 进程生命周期，不直接嵌入 SGLang runtime。

### 1.3 init 统一 IPv6、router 和 dist_init_addr

问题与约束：
- engine 可能运行在 IPv4、IPv6 或多节点环境中；host、router 和分布式初始化地址都要在传给 SGLang 前归一化。

设计选择：
- `init` 先保存 router 地址；host 缺省取本机 host info；内部 `_format_v6_uri` 给 IPv6 地址补方括号，并对 `dist_init_addr` 的 IP 部分做同样处理。

Explain：
`dist_init_addr` 通过 `rsplit(":", 1)` 拆出端口，说明这里只处理末尾端口，前面的 IP 允许是 IPv6 文本。

来源：slime/backends/sglang_utils/sglang_engine.py L119-L147

Code：

```python
def init(
    self,
    dist_init_addr,
    port,
    nccl_port,
    host=None,
    disaggregation_bootstrap_port=None,
    router_ip=None,
    router_port=None,
):
    self.router_ip = router_ip if router_ip is not None else self.args.sglang_router_ip
    self.router_port = router_port if router_port is not None else self.args.sglang_router_port

    host = host or get_host_info()[1]

    def _format_v6_uri(addr):
        if not addr or addr.startswith("["):
            return addr
        try:
            if ipaddress.ip_address(addr).version == 6:
                return f"[{addr}]"
        except ValueError:
            pass
        return addr

    host = _format_v6_uri(host)
    ip_part, port_part = dist_init_addr.rsplit(":", 1)
    dist_init_addr = f"{_format_v6_uri(ip_part)}:{port_part}"
```

代码逻辑：
- 显式传入 router 地址优先，否则使用全局 args。
- host 缺省来自 `get_host_info`。
- IPv6 地址未加括号时补括号。
- 分布式初始化地址只格式化 IP 部分。

为什么这样写：
- HTTP URL 和 SGLang dist init 对 IPv6 地址格式有明确要求。
- router 地址可以由 ServerGroup 为不同 engine 覆盖。

不变量与失败模式：
- `dist_init_addr` 必须包含末尾端口。
- 非 IP 字符串会被原样返回。
- 已带方括号的地址不会重复包裹。

Comment：
这段是多节点和 IPv6 部署下很容易忽略的启动前处理。

---

## 2. ServerArgs 组装与 engine 初始化模式

### 2.1 init 计算 ServerArgs 并选择 external 或 normal

问题与约束：
- Slime 既支持自己启动 SGLang，也支持连接外部已启动的 SGLang；两种模式需要共用同一套期望配置。

设计选择：
- `init` 调 `_compute_server_args` 得到 `server_args_dict` 和外部 engine 需要校验的字段；保存 node rank、host、port 后按 `args.rollout_external` 选择 `_init_external` 或 `_init_normal`。

Explain：
如果权重更新模式是 `delta + disk`，初始化末尾启动 daemon thread 预热本地 checkpoint 基座。

来源：slime/backends/sglang_utils/sglang_engine.py L148-L182

Code：

```python
server_args_dict, external_engine_need_check_fields = _compute_server_args(
    self.args,
    self.rank,
    dist_init_addr,
    nccl_port,
    host,
    port,
    self.worker_type,
    disaggregation_bootstrap_port,
    base_gpu_id=self.base_gpu_id,
    sglang_overrides=self.sglang_overrides,
    num_gpus_per_engine=self.num_gpus_per_engine,
)

self.node_rank = server_args_dict["node_rank"]
self.server_host = server_args_dict["host"]
self.server_port = server_args_dict["port"]

if self.args.rollout_external:
    self._init_external(server_args_dict, external_engine_need_check_fields=external_engine_need_check_fields)
else:
    self._init_normal(server_args_dict)

if self.args.update_weight_mode == "delta" and self.args.update_weight_transport == "disk":
    from slime.utils.disk_delta import init_local_checkpoint

    threading.Thread(
        target=init_local_checkpoint,
        args=(self.args.update_weight_local_checkpoint_dir, self.args.hf_checkpoint),
        daemon=True,
    ).start()
```

代码逻辑：
- 根据 rank、端口、worker type 和 overrides 计算 SGLang 参数。
- 保存当前 actor 的 node rank 和 HTTP 地址。
- external 模式只校验并注册。
- normal 模式启动本地 SGLang 进程并注册。
- delta disk 模式异步初始化本地 checkpoint。

为什么这样写：
- external 和 normal 模式需要共享配置计算，避免字段漂移。
- delta checkpoint 初始化可能很重，放在 daemon thread 中不阻塞 engine init。

不变量与失败模式：
- `_compute_server_args` 必须返回 `node_rank`、`host`、`port`。
- external 模式要求远端 server 信息可访问且关键字段匹配。
- delta disk 线程失败不会在 init 调用栈中直接体现。

Comment：
`init` 是 SGLangEngine 的控制中枢：它不执行 rollout，而是把后续 HTTP 控制面搭起来。

### 2.2 external 模式校验字段，normal 模式启动进程

问题与约束：
- 外部 SGLang 可能由用户或其他系统启动；Slime 必须确认它和当前 rollout 配置兼容。本地模式则需要保存进程句柄供 shutdown。

设计选择：
- `_init_external` 通过 `get_server_info` 拉取实际 server args，并按 `external_engine_need_check_fields` 逐字段 assert；`_init_normal` 创建 `ServerArgs` 并调用 `launch_server_process`。

Explain：
两种模式最后都会调用 `_register_to_router`，让 router 可感知当前 engine。

来源：slime/backends/sglang_utils/sglang_engine.py L184-L202

Code：

```python
def _init_external(self, expect_server_args, external_engine_need_check_fields):
    logger.info(f"Use external SGLang engine (rank={self.rank}, expect_server_args={expect_server_args})")

    def _sanity_check_server_args(actual_server_args, expect_server_args):
        for name in external_engine_need_check_fields:
            expect_value = expect_server_args.get(name)
            actual_value = actual_server_args.get(name)
            assert (
                actual_value == expect_value
            ), f"{name=} {expect_value=} {actual_value=} {expect_server_args=} {actual_server_args=}"

    actual_server_args = get_server_info(f"http://{self.server_host}:{self.server_port}")
    _sanity_check_server_args(actual_server_args, expect_server_args)
    self._register_to_router(expect_server_args)

def _init_normal(self, server_args_dict):
    logger.info(f"Launch HttpServerEngineAdapter at: {self.server_host}:{self.server_port}")
    self.process = launch_server_process(ServerArgs(**server_args_dict))
    self._register_to_router(server_args_dict)
```

代码逻辑：
- external 模式记录期望参数。
- 从外部 server 拉实际参数。
- 对需要检查的字段逐项 assert。
- normal 模式创建 `ServerArgs` 并启动本地进程。
- 两种模式都注册 router。

为什么这样写：
- 外部 engine 的 host、port、并行规模等可能与 Slime 计算值不同，必须按白名单跳过或校验。
- 本地 engine 的进程句柄是后续 shutdown 的依据。

不变量与失败模式：
- external server 必须实现 `get_server_info` 依赖的接口。
- 任何关键字段不匹配都会 assert。
- normal 模式下 `self.process` 必须成功赋值。

Comment：
外部模式不是盲连；Slime 会验证它是否满足当前 rollout group 的约束。

### 2.3 _compute_server_args 写入 rollout 所需的 SGLang 参数

问题与约束：
- 一个 Slime rollout engine 需要把训练参数翻译成 SGLang 的 `ServerArgs`，包括模型路径、端口、多节点 rank、并行尺寸、offload、metrics 和 disaggregation mode。

设计选择：
- `_compute_server_args` 先计算单 engine GPU 数、节点数、node rank 和 base GPU，再组装基础 kwargs；prefill、decode、encoder 三类 worker 追加不同字段。

Explain：
Slime 默认 `skip_server_warmup=True` 避免 warmup timeout，并默认开启 `enable_draft_weights_cpu_backup` 和 Prometheus metrics。

来源：slime/backends/sglang_utils/sglang_engine.py L578-L623

Code：

```python
_gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
nnodes = max(1, _gpus_per_engine // args.num_gpus_per_node)
node_rank = rank % nnodes
base = base_gpu_id if base_gpu_id is not None else get_base_gpu_id(args, rank)
base = _to_local_gpu_id(base)
kwargs = {
    "model_path": args.hf_checkpoint,
    "trust_remote_code": True,
    "random_seed": args.seed + rank,
    "enable_memory_saver": args.offload_rollout,
    "host": host,
    "port": port,
    "nccl_port": nccl_port,
    "nnodes": nnodes,
    "node_rank": node_rank,
    "dist_init_addr": dist_init_addr,
    "gpu_id_step": 1,
    "base_gpu_id": base,
    "tp_size": _gpus_per_engine // args.sglang_pp_size,
    "dp_size": args.sglang_dp_size,
    "pp_size": args.sglang_pp_size,
    "ep_size": args.sglang_ep_size,
    "skip_server_warmup": True,
    "enable_draft_weights_cpu_backup": True,
    "enable_metrics": True,
}

if worker_type == "prefill":
    kwargs["disaggregation_mode"] = "prefill"
    kwargs["load_balance_method"] = "follow_bootstrap_room"
    assert (
        disaggregation_bootstrap_port is not None
    ), "disaggregation_bootstrap_port must be set for prefill worker"
    kwargs["disaggregation_bootstrap_port"] = disaggregation_bootstrap_port
elif worker_type == "decode":
    kwargs["disaggregation_mode"] = "decode"
    kwargs["prefill_round_robin_balance"] = True
elif worker_type == "encoder":
    kwargs["encoder_only"] = True
```

代码逻辑：
- 单 engine GPU 数可由参数覆盖。
- `nnodes` 由 engine GPU 数和每节点 GPU 数推导。
- `node_rank = rank % nnodes`。
- base GPU 可由调用方覆盖，否则按 rank 计算。
- 组装模型路径、随机种子、端口、并行尺寸和默认开关。
- 按 worker type 添加 PD 或 encoder 参数。

为什么这样写：
- Slime 的 rollout group 以 Ray actor rank 为坐标，需要翻译成 SGLang 的多节点/多并行配置。
- prefill worker 必须提供 bootstrap port，router 才能维护 PD 连接。

不变量与失败模式：
- `sglang_pp_size` 必须能整除单 engine GPU 数。
- prefill worker 缺少 `disaggregation_bootstrap_port` 会 assert。
- `node_rank` 公式假设同一个 engine 的多节点 actor rank 连续分布。

Comment：
这段是 Slime 参数体系到 SGLang `ServerArgs` 的主要映射表。

### 2.4 ServerArgs 字段过滤、YAML overrides 和 external 校验白名单

问题与约束：
- Slime 参数可能包含当前 SGLang 版本不支持的字段；同时用户还需要用 YAML 覆盖单个 server group 参数。外部 engine 校验又不能要求 host/port/tp_size 等由外部决定的字段完全相同。

设计选择：
- 先用 `dataclasses.fields(ServerArgs)` 得到合法字段；把 `args.sglang_{field}` 填进 kwargs；再应用 `sglang_overrides`，支持把 key 中的 `-` 规范化为 `_`；最终删除不被当前 SGLang 支持的 keys，并返回外部 engine 校验字段列表。

Explain：
`_EXTERNAL_ENGINE_SKIP_CHECK_FIELDS` 明确列出 external 模式不校验的字段，例如模型路径、host/port、node rank、并行尺寸、warmup、metrics 和 `mem_fraction_static`。

来源：slime/backends/sglang_utils/sglang_engine.py L625-L690

Code：

```python
if args.use_rollout_routing_replay:
    kwargs["enable_return_routed_experts"] = True
if args.fp16:
    kwargs["dtype"] = "float16"
external_engine_need_check_fields = [k for k in kwargs.keys() if k not in _EXTERNAL_ENGINE_SKIP_CHECK_FIELDS]

server_arg_fields = dataclasses.fields(ServerArgs)
server_arg_field_names = {attr.name for attr in server_arg_fields}
unused_keys = set(kwargs.keys())
for attr in server_arg_fields:
    if worker_type == "decode" and attr.name == "enable_hierarchical_cache":
        continue
    if hasattr(args, f"sglang_{attr.name}") and attr.name not in kwargs:
        kwargs[attr.name] = getattr(args, f"sglang_{attr.name}")
    unused_keys.discard(attr.name)

if sglang_overrides:
    for key, value in sglang_overrides.items():
        normalized_key = key.replace("-", "_")
        if normalized_key != key:
            logger.warning(
                f"sglang_overrides key '{key}' normalized to '{normalized_key}' (rank={rank}). "
                "Please use underscore style in YAML overrides."
            )
        if normalized_key in kwargs:
            logger.info(
                f"sglang_overrides: overriding {normalized_key}={kwargs[normalized_key]} -> {value} (rank={rank})"
            )
        kwargs[normalized_key] = value
        if normalized_key in server_arg_field_names:
            unused_keys.discard(normalized_key)
        else:
            unused_keys.add(normalized_key)

if len(unused_keys) > 0:
    logger.info(f"Warning: The following arguments is not supported in the current sglang: {unused_keys}.")
    for key in unused_keys:
        kwargs.pop(key)

return kwargs, external_engine_need_check_fields


_EXTERNAL_ENGINE_SKIP_CHECK_FIELDS = [
    "model_path",
    "trust_remote_code",
    "random_seed",
    "host",
    "port",
    "nccl_port",
    "nnodes",
    "node_rank",
    "dist_init_addr",
    "gpu_id_step",
    "base_gpu_id",
    "tp_size",
    "dp_size",
    "pp_size",
    "ep_size",
    "skip_server_warmup",
    "enable_draft_weights_cpu_backup",
    "enable_metrics",
    "mem_fraction_static",
]
```

代码逻辑：
- 根据 routing replay 和 fp16 追加参数。
- 在删除 unsupported keys 前先计算 external 需要校验的字段。
- 遍历 `ServerArgs` 字段，把 `args.sglang_*` 补进 kwargs。
- YAML overrides 最后应用，优先级最高。
- 不支持字段记录日志后删除。
- 返回 kwargs 和 external 校验字段。

为什么这样写：
- SGLang 版本更新时字段集合会变化，Slime 需要兼容不同版本。
- overrides 应该覆盖基础参数和 CLI 派生参数。
- external 模式只能校验真正由 Slime 期望控制的字段。

不变量与失败模式：
- override key 如果当前 SGLang 不支持，会被记录后删除。
- decode worker 跳过 `enable_hierarchical_cache` 的自动继承。
- external 校验字段是在 overrides 前按基础 kwargs 算出的，新增 override 字段不会自动进入该列表。

Comment：
这段是适配层里最关键的版本兼容逻辑。

---

## 3. Router 与 HTTP 请求封装

### 3.1 _register_to_router 只让 node 0 注册可生成 worker

问题与约束：
- 多节点 SGLang engine 只有 node 0 暴露 HTTP 服务入口；encoder-only worker 不参与 generate 路由；旧版和新版 router API 不一致。

设计选择：
- encoder worker 直接 return；`node_rank == 0` 且配置了 router 地址时才注册。router 版本小于等于 0.2.1 走 `/add_worker?url=...`，新版走 `/workers` JSON API；prefill worker 需要带 bootstrap port。

Explain：
新版 payload 包含 `url` 和 `worker_type`；如果 worker type 是 prefill，必须从 `server_args_dict` 取到 `disaggregation_bootstrap_port`。

来源：slime/backends/sglang_utils/sglang_engine.py L204-L232

Code：

```python
def _register_to_router(self, server_args_dict):
    if self.worker_type == "encoder":
        return

    if self.node_rank == 0 and self.router_ip and self.router_port:
        worker_url = f"http://{self.server_host}:{self.server_port}"
        if parse(sglang_router.__version__) <= parse("0.2.1"):
            assert self.worker_type == "regular", "pd disaggregation is not supported in old router."
            response = requests.post(
                f"http://{self.router_ip}:{self.router_port}/add_worker?url={worker_url}",
            )
        else:
            payload = {
                "url": worker_url,
                "worker_type": self.worker_type,
            }
            if self.worker_type == "prefill":
                bootstrap_port = server_args_dict.get("disaggregation_bootstrap_port")
                if bootstrap_port is None:
                    raise RuntimeError(
                        f"Prefill worker {worker_url} does not have disaggregation_bootstrap_port; "
                        "cannot register it to the PD router."
                    )
                payload["bootstrap_port"] = bootstrap_port
            response = requests.post(
                f"http://{self.router_ip}:{self.router_port}/workers",
                json=payload,
            )
        response.raise_for_status()
```

代码逻辑：
- encoder worker 不注册。
- 非 node 0 不注册。
- 旧 router 只支持 regular worker。
- 新 router 支持 worker type 和 prefill bootstrap port。
- HTTP 非成功状态直接抛异常。

为什么这样写：
- Router 只需要可接请求的 HTTP 入口，非 node 0 注册会产生不可用地址。
- PD disaggregation 依赖新版 router 能区分 prefill/decode。

不变量与失败模式：
- 旧版 router 下不能注册 prefill/decode worker。
- prefill worker 缺 bootstrap port 会 RuntimeError。
- router 地址缺失时不会注册，也不会报错。

Comment：
Router 注册是 Slime 把多个 SGLang engine 纳入 rollout 路由面的步骤。

### 3.2 _make_request 统一 node 0 HTTP POST 与错误上下文

问题与约束：
- 权重更新和运行控制 API 都是发给 SGLang HTTP server；多节点 engine 的非 node 0 不应重复发请求。

设计选择：
- `_make_request` 在 `node_rank != 0` 时直接返回；node 0 构造 endpoint URL，用 `requests.post` 发 JSON，`raise_for_status` 失败时把 response text 追加到异常 note。

Explain：
返回值统一是 `response.json()`，上层 API 不需要重复处理 HTTP 细节。

来源：slime/backends/sglang_utils/sglang_engine.py L234-L254

Code：

```python
def _make_request(self, endpoint: str, payload: dict | None = None):
    if self.node_rank != 0:
        return

    url = f"http://{self.server_host}:{self.server_port}/{endpoint}"
    response = requests.post(url, json=payload or {})
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        e.add_note(f"{response.text=}")
        raise
    return response.json()
```

代码逻辑：
- 非 node 0 直接 no-op。
- endpoint 拼到本 engine HTTP 地址后面。
- payload 为空时发送空 dict。
- HTTP 错误附加响应正文后重新抛出。
- 成功时返回 JSON。

为什么这样写：
- 多节点 engine 的分布式参与由 SGLang 内部处理，Slime 侧只需要 node 0 发控制请求。
- 统一错误上下文能让 Ray 远程异常更可读。

不变量与失败模式：
- 被调用 endpoint 必须返回 JSON。
- 非 node 0 调用会返回 `None`，调用方不能假设每个 actor 都有响应体。

Comment：
理解后续所有权重 API 时，先记住它们大多只是 `_make_request` 的薄包装。

---

## 4. 权重更新与运行控制 API

### 4.1 update_weights_from_tensor 只通过 HTTP 传 metadata

问题与约束：
- colocate 或 GPU 侧 IPC 权重更新不应把完整 tensor 通过 HTTP 传输；HTTP 只适合控制和 metadata。

设计选择：
- `update_weights_from_tensor` payload 包含 `serialized_named_tensors`、`load_format`、`flush_cache` 和可选 `weight_version`，再调用 SGLang 的 `update_weights_from_tensor` endpoint。

Explain：
函数 docstring 强调真实 weights 会直接从 GPU 复制，模型应在 GPU 上而不是 CPU 上。

来源：slime/backends/sglang_utils/sglang_engine.py L278-L301

Code：

```python
def update_weights_from_tensor(
    self,
    serialized_named_tensors: list[str],
    load_format: str | None = None,
    flush_cache: bool = False,
    weight_version: str | None = None,
):
    payload = {
        "serialized_named_tensors": serialized_named_tensors,
        "load_format": load_format,
        "flush_cache": flush_cache,
    }
    if weight_version is not None:
        payload["weight_version"] = weight_version
    return self._make_request(
        "update_weights_from_tensor",
        payload,
    )
```

代码逻辑：
- 接收序列化后的 named tensor 描述。
- 传递 load format 和 flush cache 选项。
- 可选写入 weight version。
- 通过 `_make_request` 发给 SGLang。

为什么这样写：
- HTTP 请求不承担大 tensor 数据面。
- weight version 让 rollout/训练侧能追踪当前 serving 权重版本。

不变量与失败模式：
- SGLang 侧必须能解析 `serialized_named_tensors`。
- 若模型不在 GPU，docstring 指出的 GPU 复制路径可能失败。
- flush cache 默认为 false，调用方要按更新时机显式决定。

Comment：
tensor 模式适合 colocate 场景，控制面和数据面是分离的。

### 4.2 flush_cache 轮询 GET /flush_cache 直到成功

问题与约束：
- SGLang 有 pending requests 时 flush cache 不一定立即返回 200；权重更新前后需要一个可重试的 cache 清理入口。

设计选择：
- `flush_cache` 只在 node 0 执行；最多 60 次请求 `/flush_cache`，非 200 或普通异常记录日志后 1 秒重试；超过次数抛 TimeoutError。

Explain：
`NewConnectionError` 被单独重新抛出，避免连接层错误被吞成普通重试日志。

来源：slime/backends/sglang_utils/sglang_engine.py L303-L322

Code：

```python
def flush_cache(self):
    if self.node_rank != 0:
        return
    for _ in range(60):
        try:
            response = requests.get(f"http://{self.server_host}:{self.server_port}/flush_cache")
            if response.status_code == 200:
                break
            logger.info(f"Error flushing cache: HTTP {response.status_code} {response.text!r}")
            time.sleep(1)
        except NewConnectionError as e:
            raise e
        except Exception as e:
            logger.info(f"Error flushing cache: {e}")
            time.sleep(1)
            continue
    else:
        raise TimeoutError("Timeout while flushing cache.")
```

代码逻辑：
- 非 node 0 no-op。
- 循环发 GET 请求。
- 200 退出循环。
- HTTP 非 200 记录响应体并等待。
- 普通异常记录后重试。
- 超过 60 次失败抛 TimeoutError。

为什么这样写：
- cache flush 受请求队列状态影响，立即失败不代表不可恢复。
- 权重切换时旧 KV/cache 不能继续复用。

不变量与失败模式：
- 60 秒内仍无法 flush 会失败。
- 连接错误可能直接抛出。
- 调用方若未先 pause generation，flush 更容易长期非 200。

Comment：
这段是权重更新流程里最直接的“等待请求排空”保护。

### 4.3 release/resume memory 与本地 delta checkpoint 同步

问题与约束：
- offload rollout 场景需要让 SGLang 释放或恢复显存；delta disk 模式还需要把训练侧发布的 delta 应用到本地 checkpoint。

设计选择：
- `release_memory_occupation` 先 flush cache，再调用 release endpoint；`resume_memory_occupation` 传可选 tags；`sync_local_checkpoint` 先确保本地基座存在，可选执行预读 hook，再应用 deltas。

Explain：
`sync_local_checkpoint` 的注释说明它假设 Slime actor 与其驱动的 SGLang 共享 checkpoint 文件系统。

来源：slime/backends/sglang_utils/sglang_engine.py L380-L414

Code：

```python
def release_memory_occupation(self):
    self.flush_cache()
    return self._make_request("release_memory_occupation")

def resume_memory_occupation(self, tags: list[str] = None):
    return self._make_request(
        "resume_memory_occupation",
        {"tags": tags},
    )

def check_weights(self, action: str):
    return self._make_request("weights_checker", {"action": action})

def sync_local_checkpoint(self, target_version: int):
    from slime.utils.disk_delta import apply_deltas, init_local_checkpoint

    init_local_checkpoint(self.args.update_weight_local_checkpoint_dir, self.args.hf_checkpoint)
    if self.args.custom_delta_pre_read_path:
        from slime.utils.misc import load_function

        load_function(self.args.custom_delta_pre_read_path)(self.args.update_weight_disk_dir, target_version)
    apply_deltas(
        self.args.update_weight_local_checkpoint_dir,
        self.args.update_weight_disk_dir,
        target_version,
    )
```

代码逻辑：
- release 前先清 cache。
- resume endpoint 可按 tags 恢复部分内存占用。
- check_weights 透传 action。
- delta 同步先初始化本地 checkpoint。
- 可选执行用户自定义 pre-read hook。
- 应用指定版本之前的 deltas。

为什么这样写：
- 释放显存前必须先清掉依赖旧内存状态的 cache。
- delta disk 模式把大文件 reload 转成增量应用，减少同步成本。

不变量与失败模式：
- release 依赖 flush cache 成功。
- delta 路径要求本地和 SGLang 进程共享文件系统视图。
- custom pre-read hook 失败会阻断 delta 应用。

Comment：
这组 API 是 rollout offload 和 disk delta 两条优化路径的 Slime 侧入口。

### 4.4 update_weights_from_disk 把 reload 请求交给 SGLang 文件系统路径

问题与约束：
- disk 模式下权重已经落到文件系统；HTTP 只需要告诉 SGLang 路径、格式、版本和可选文件列表。

设计选择：
- payload 至少包含 `model_path`；可选加入 `load_format`、`weight_version`、`files`，然后调用 `update_weights_from_disk` endpoint。

Explain：
函数 docstring 区分标准 HF reload 和 delta reload：delta 模式下 `model_path` 是版本子目录的父目录，`files` 是本次需要读取并应用的 basename 列表。

来源：slime/backends/sglang_utils/sglang_engine.py L415-L437

Code：

```python
def update_weights_from_disk(
    self,
    model_path: str,
    load_format: str | None = None,
    weight_version: str | None = None,
    files: list[str] | None = None,
):
    payload: dict = {"model_path": model_path}
    if load_format is not None:
        payload["load_format"] = load_format
    if weight_version is not None:
        payload["weight_version"] = weight_version
    if files is not None:
        payload["files"] = files
    return self._make_request("update_weights_from_disk", payload)
```

代码逻辑：
- 构造基础 payload。
- 按需加入 load format。
- 按需加入权重版本。
- 按需加入文件列表。
- 通过 `_make_request` 发给 SGLang。

为什么这样写：
- disk reload 的数据面在文件系统，HTTP 请求保持轻量。
- delta reload 需要显式 files，避免 SGLang 扫描或误读其他版本文件。

不变量与失败模式：
- `model_path` 必须对 SGLang 进程可见。
- delta 模式下 files 必须和本地已应用的 delta 文件一致。
- SGLang endpoint 必须支持给定 `load_format`。

Comment：
disk 模式牺牲部分延迟，换来更简单的数据面和跨进程边界。

### 4.5 distributed 权重更新先建立通信组，再传 names/dtypes/shapes

问题与约束：
- distributed update 的 tensor 数据通过 torch distributed 通信组传输；SGLang HTTP 请求只需要建组参数和 tensor metadata。

设计选择：
- `init_weights_update_group` 调 SGLang 的建组 endpoint；`destroy_weights_update_group` 尝试销毁，若请求异常则忽略；`update_weights_from_distributed` 传 names、去掉 `torch.` 前缀的 dtypes、shapes、group name、flush cache 和可选版本/format。

Explain：
`destroy_weights_update_group` 捕获 `requests.exceptions.RequestException`，兼容 engine 刚创建时还没有对应 group 的情况。

来源：slime/backends/sglang_utils/sglang_engine.py L439-L488

Code：

```python
def init_weights_update_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
    return self._make_request(
        "init_weights_update_group",
        {
            "master_address": master_address,
            "master_port": master_port,
            "rank_offset": rank_offset,
            "world_size": world_size,
            "group_name": group_name,
            "backend": backend,
        },
    )

def destroy_weights_update_group(self, group_name):
    try:
        return self._make_request(
            "destroy_weights_update_group",
            {
                "group_name": group_name,
            },
        )
    except requests.exceptions.RequestException:
        pass

def update_weights_from_distributed(
    self,
    names,
    dtypes,
    shapes,
    group_name,
    flush_cache=False,
    weight_version: str | None = None,
    load_format: str | None = None,
):
    payload = {
        "names": names,
        "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
        "shapes": shapes,
        "group_name": group_name,
        "flush_cache": flush_cache,
    }
    if weight_version is not None:
        payload["weight_version"] = weight_version
    if load_format is not None:
        payload["load_format"] = load_format
    return self._make_request(
        "update_weights_from_distributed",
        payload,
    )
```

代码逻辑：
- 建组请求包含 master 地址、端口、rank offset、world size、group name 和 backend。
- 销毁请求只传 group name。
- distributed update 请求只传 metadata 和 group name。
- dtype 字符串去掉 `torch.` 前缀。
- 可选传 weight version 和 load format。

为什么这样写：
- 分布式传输的数据面由训练进程和 SGLang 进程的通信组完成。
- HTTP 只负责让 SGLang 准备接收哪些 tensor。

不变量与失败模式：
- group name 必须和训练侧使用的通信组一致。
- names、dtypes、shapes 三个列表必须一一对应。
- destroy 异常被吞掉，可能留下服务端已有 group，需要由后续建组逻辑处理冲突。

Comment：
distributed 模式是 Slime 与 SGLang 权重同步最接近训练后端的路径。

### 4.6 pause/continue generation 与 post_process_weights 是权重切换控制点

问题与约束：
- 权重切换期间不应继续生成；某些加载路径还需要在权重写入后执行恢复或量化后处理。

设计选择：
- `pause_generation` 和 `continue_generation` 直接 POST 对应 endpoint；`post_process_weights` 通过 `_make_request` 调用 SGLang 的 `post_process_weights`，并传 `restore_weights_before_load`、`post_process_quantization` 两个控制位。

Explain：
pause/continue 没有 node rank guard，而是直接向当前 engine HTTP 地址发请求；post_process 继承 `_make_request` 的 node 0 guard。

来源：slime/backends/sglang_utils/sglang_engine.py L490-L517

Code：

```python
def pause_generation(self):
    response = requests.post(f"http://{self.server_host}:{self.server_port}/pause_generation", json={})
    response.raise_for_status()
    return response

def continue_generation(self):
    response = requests.post(f"http://{self.server_host}:{self.server_port}/continue_generation", json={})
    response.raise_for_status()
    return response

def post_process_weights(
    self,
    restore_weights_before_load: bool = False,
    post_process_quantization: bool = False,
):
    return self._make_request(
        "post_process_weights",
        {
            "restore_weights_before_load": restore_weights_before_load,
            "post_process_quantization": post_process_quantization,
        },
    )
```

代码逻辑：
- pause 发送空 JSON 到 `/pause_generation`。
- continue 发送空 JSON 到 `/continue_generation`。
- post process 传两个布尔配置。
- 所有 HTTP 错误都会通过 `raise_for_status` 或 `_make_request` 抛出。

为什么这样写：
- pause/continue 是权重更新前后最直接的请求调度阀门。
- post process 让加载流程与量化/恢复逻辑分离，避免每个 update endpoint 重复参数。

不变量与失败模式：
- pause/continue 在非 node 0 上调用会尝试访问该 actor 保存的 server 地址，调用方应确保只让有效入口执行。
- post process endpoint 必须返回 JSON。

Comment：
权重更新流程通常不是单个 update API，而是 pause、flush/update、post process、continue 的组合。

---

## 5. shutdown 与运行期辅助

### 5.1 shutdown 区分外部 engine 与本地进程

问题与约束：
- Slime 不应该杀掉外部托管的 SGLang；本地启动的 engine 则需要从 router 移除并终止进程树。router 删除 API 也有版本差异。

设计选择：
- `shutdown` 在 `rollout_external` 时直接返回；本地模式下，非 encoder 且 node 0 先按 router 版本删除 worker，然后 `kill_process_tree(self.process.pid)`。

Explain：
router 0.2.1 及以下使用 `remove_worker?url=...`；0.3.0 前使用 URL encode 后的 `/workers/{url}`；0.3.0 及以上先拉 workers 列表，根据 worker URL 找 id 再删除。

来源：slime/backends/sglang_utils/sglang_engine.py L329-L361

Code：

```python
def shutdown(self):
    if self.args.rollout_external:
        return

    logger.info(f"Shutdown engine {self.server_host}:{self.server_port}...")
    if self.worker_type != "encoder" and self.node_rank == 0:
        worker_url = f"http://{self.server_host}:{self.server_port}"
        response = None
        if parse(sglang_router.__version__) <= parse("0.2.1"):
            response = requests.post(
                f"http://{self.router_ip}:{self.router_port}/remove_worker?url=http://{self.server_host}:{self.server_port}"
            )
        elif parse(sglang_router.__version__) < parse("0.3.0"):
            worker_url = quote(worker_url, safe="")
            response = requests.delete(f"http://{self.router_ip}:{self.router_port}/workers/{worker_url}")
        else:
            try:
                all_workers = requests.get(f"http://{self.router_ip}:{self.router_port}/workers").json()["workers"]
                for worker in all_workers:
                    if worker["url"] == worker_url:
                        worker_id = worker["id"]
                        response = requests.delete(
                            f"http://{self.router_ip}:{self.router_port}/workers/{worker_id}"
                        )
                        break
                else:
                    logger.warning(f"Worker {worker_url} not found in router during shutdown.")
            except Exception as e:
                logger.warning(f"Failed to fetch workers list or remove worker: {e}")

        if response is not None:
            response.raise_for_status()
    kill_process_tree(self.process.pid)
```

代码逻辑：
- 外部 engine 不处理。
- 本地 engine 记录 shutdown 日志。
- node 0 的非 encoder worker 从 router 删除。
- 根据 router 版本选择删除 API。
- 删除响应非成功时抛异常。
- 最后杀进程树。

为什么这样写：
- 外部 engine 生命周期不属于 Slime。
- Router API 版本迁移期需要兼容多种删除方式。
- kill process tree 确保 SGLang 子进程及其子进程都退出。

不变量与失败模式：
- 本地模式必须有 `self.process`。
- router 删除失败会在 kill 前抛异常，可能影响本地进程清理。
- 0.3.0 及以上如果 workers 列表拉取失败，只打 warning，仍会继续 kill 本地进程。

Comment：
`shutdown` 的核心边界是所有权：只清理 Slime 自己启动的 engine。

### 5.2 profiling 与 crash simulation 是本地调试控制面

问题与约束：
- rollout 过程中需要能远程启动/停止 SGLang profiling，也需要模拟本地 engine 崩溃；外部 engine 模式不能被 Slime 杀掉。

设计选择：
- `start_profile` 和 `stop_profile` 直接 POST SGLang profiling endpoints；`simulate_crash` 在 external 或没有本地进程时只记录日志并返回，否则调用 `shutdown`。

Explain：
profiling payload 支持输出目录、起始 step、步数、activities、按 stage profiling、stack 和 shape 记录等选项。

来源：slime/backends/sglang_utils/sglang_engine.py L519-L562

Code：

```python
def start_profile(
    self,
    output_dir: str | None = None,
    start_step: int | None = None,
    num_steps: int | None = None,
    activities: list[str] | None = None,
    profile_by_stage: bool = False,
    with_stack: bool | None = None,
    record_shapes: bool | None = None,
):
    response = requests.post(
        f"http://{self.server_host}:{self.server_port}/start_profile",
        json={
            "output_dir": output_dir,
            "start_step": start_step,
            "num_steps": num_steps,
            "activities": activities,
            "profile_by_stage": profile_by_stage,
            "with_stack": with_stack,
            "record_shapes": record_shapes,
        },
    )
    response.raise_for_status()
    return response

def stop_profile(self):
    response = requests.post(f"http://{self.server_host}:{self.server_port}/stop_profile", json={})
    response.raise_for_status()
    return response

def simulate_crash(self):
    if self.args.rollout_external or not getattr(self, "process", None):
        logger.info(
            "simulate_crash called but no local engine process exists (rollout_external=%s); skip kill",
            self.args.rollout_external,
        )
        return

    logger.info(f"Simulating crash on engine {self.server_host}:{self.server_port}...")
    self.shutdown()
```

代码逻辑：
- start profile 组装 profiling 参数并 POST。
- stop profile POST 空 JSON。
- simulate crash 先判断是否有本地进程所有权。
- 有本地进程时复用 shutdown。

为什么这样写：
- profiling 是 HTTP 控制面能力，不需要走 `_make_request` 的 JSON 返回约束。
- crash simulation 应复用正常 shutdown，避免保留 router 注册或子进程。

不变量与失败模式：
- profiling endpoint 必须存在且返回成功状态。
- external engine 不会被 simulate crash 关闭。
- shutdown 的 router 删除失败仍可能影响 simulate crash。

Comment：
这些方法不是 rollout 主链路，但对压测、profiling 和容错演练很有用。

---

## 6. 调用链小结

```text
RolloutManager / ServerGroup
  -> Ray SGLangEngine.init
     -> _compute_server_args
     -> _init_normal / _init_external
     -> _register_to_router

weight sync
  -> pause_generation
  -> flush_cache / update_weights_*
  -> post_process_weights
  -> continue_generation

shutdown
  -> router unregister
  -> kill local SGLang process tree
```
