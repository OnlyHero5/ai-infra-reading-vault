---
type: batch-doc
module: 16-External-Engines
batch: "16"
doc_type: walkthrough
title: "External Engines · 源码走读"
tags:
  - slime/batch/16
  - slime/module/external-engines
  - slime/doc/walkthrough
updated: 2026-07-05
---

# External Engines · 源码走读

> 读法：外部 rollout engine 的主线不是“Slime 启动 SGLang”，而是“Slime 发现并接管已经运行的 SGLang”。因此顺序应从 CLI 参数进入 `args` 开始，沿 discovery、Ray 代理 actor、Router 注册、HTTP 客户端和运维边界往下看。

---

## 1. 参数入口与早期 discovery

### 1.1 `--rollout-external-engine-addrs`：外部 engine 地址列表

来源：slime/utils/arguments.py L555-L561

**问题与约束：** 外部 engine 模式下，SGLang server 已经由用户或外部系统启动，Slime 只能通过地址列表接入；一个训练任务可能对应多个 engine。

**设计选择：** CLI 参数使用 `nargs="+"` 接收一个或多个地址，默认 `None`，帮助文本明确这是 external engine 的 address/port。

**Explain：** 这个参数是 external 模式的显式开关来源。没有地址列表时，Slime 走内置 engine 启动路径；有地址列表时，后续会进入外部拓扑发现。

**Code：**

```python
            parser.add_argument(
                "--rollout-external-engine-addrs",
                type=str,
                default=None,
                nargs="+",
                help="Address and ports of the external engines.",
            )
```

**代码逻辑：** parser 将命令行中的多个 `host:port` 字符串收集到 `args.rollout_external_engine_addrs`；用户不传时字段为 `None`。

**为什么这样写：** 外部 engines 是已有服务集合，不适合由 Slime 推导端口或数量。把地址作为多值参数传入，可以让资源发现和 Router 注册都以真实 server 为准。

**不变量与失败模式：** 每个值必须能被后续规范化为 HTTP base URL；列表为空或 `None` 表示不启用 external 模式。若只支持单地址，多 engine rollout 无法表达。

**Comment：** 参数本身只收集地址；真正的模式切换和探测在参数收尾阶段发生。

### 1.2 参数收尾：设置 `rollout_external` 并触发 discovery

来源：slime/utils/arguments.py L1849-L1854

**问题与约束：** Placement Group 和 rollout 资源规划依赖 engine 数和 GPU 数；这些信息必须在正式启动 rollout manager 前写回 `args`。但 debug train only 不需要 rollout，不应探测外部服务。

**设计选择：** 用 `args.rollout_external_engine_addrs is not None` 设置 `args.rollout_external`；当 external 且不是 `debug_train_only` 时，立即调用 `apply_external_engine_info_to_args`。

**Explain：** 这就是 external discovery 的早期入口。CLI 解析阶段不只是保存字符串，还会主动访问外部 server，把拓扑信息补进 args。

**Code：**

```python
        args.debug_train_only = True

    args.rollout_external = args.rollout_external_engine_addrs is not None

    if args.rollout_external and not args.debug_train_only:
        apply_external_engine_info_to_args(args, logger=logger)
```

**代码逻辑：** 若前面逻辑判定只 debug train，则先设置 `debug_train_only`；随后根据地址参数生成 external flag。最后在需要 rollout 的 external 模式下调用 discovery 应用函数。

**为什么这样写：** 外部 engine 数量不是 Slime 本地 launch 出来的，必须通过 server_info 获取。把 discovery 放在 args 阶段，可以让后续 PG、Router、HTTP 客户端都读取同一份派生字段。

**不变量与失败模式：** `debug_train_only` 时不能强行访问外部 engine；地址非空时必须使 `rollout_external=True`；discovery 失败应尽早暴露。若等到生成阶段才探测，资源规划和 Router 初始化会缺少外部拓扑。

**Comment：** 这段把 external 模式从“用户传了地址”提升为“args 已带完整外部拓扑”。

---

## 2. 地址规范化与拓扑发现

### 2.1 `normalize_external_engine_addr`：统一 HTTP base URL

来源：slime/backends/sglang_utils/external.py L32-L44

**问题与约束：** 用户可能传 `host:port`，也可能传 `http://host:port/`；后续 HTTP 请求需要稳定的 base URL，并且必须拒绝缺端口或非 HTTP scheme 的地址。

**设计选择：** 没有 scheme 时补 `http://`，去掉尾部 `/`，用 `urlparse` 校验 scheme、hostname 和 port；非法时抛出带用法提示的 `ValueError`。

**Explain：** 规范化让后续 `get_server_info(url)` 可以直接拼 endpoint，不再关心用户输入形态。

**Code：**

```python
def normalize_external_engine_addr(addr: str) -> str:
    """Normalize ``host:port`` or ``http://host:port`` to an HTTP base URL."""
    if "://" not in addr:
        addr = f"http://{addr}"
    addr = addr.rstrip("/")
    parsed = urlparse(addr)
    if parsed.scheme != "http" or parsed.hostname is None or parsed.port is None:
        raise ValueError(
            f"Invalid external SGLang engine address {addr!r}. "
            "Use host:port or http://host:port (IPv6 must be bracketed)."
        )
    return addr
```

**代码逻辑：** 函数先补 scheme，再去尾斜杠；解析后检查必须是 HTTP、必须有 host 和 port；通过后返回规范化字符串。

**为什么这样写：** Router 注册、server_info 探测和 init kwargs 都依赖 host/port。早期校验比在后续 requests 报错更清晰，也能正确处理 IPv6 bracket 要求。

**不变量与失败模式：** 返回值不带尾部 `/`；只接受 HTTP；IPv6 必须带方括号以保留端口解析。若允许缺 port，Slime 无法构造 `dist_init_addr` 和 worker URL。

**Comment：** 地址规范化是 external 模式的第一道输入边界，后续代码都假设它已经成立。

### 2.2 `get_server_info`：兼容两个 SGLang server_info endpoint

来源：slime/backends/sglang_utils/external.py L58-L67

**问题与约束：** 不同 SGLang 版本暴露的 server info endpoint 名称可能不同；Slime 需要拿到拓扑信息，同时给失败路径保留可诊断的 endpoint 错误。

**设计选择：** 依次请求 `/server_info` 和 `/get_server_info`，任一成功就返回 JSON；全部失败时把两个 endpoint 的异常拼进 `RuntimeError`。

**Explain：** 这个函数是 external discovery 的网络探针。它屏蔽了 SGLang endpoint 命名差异，让上层只依赖一个 `dict`。

**Code：**

```python
def get_server_info(url: str, timeout: float = 30.0) -> dict:
    errors = []
    for endpoint in ("/server_info", "/get_server_info"):
        try:
            response = requests.get(f"{url}{endpoint}", timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            errors.append(f"{endpoint}: {exc}")
    raise RuntimeError(f"Failed to fetch SGLang server info from {url}: {'; '.join(errors)}")
```

**代码逻辑：** 函数维护错误列表，循环尝试两个 endpoint；每次请求都检查 HTTP status，成功则解析 JSON 返回；失败记录异常并继续。

**为什么这样写：** 外部 engine 通常由不同版本或不同部署脚本启动。兼容两个 endpoint 可以降低版本耦合；保留全部错误能快速判断是 404、连接失败还是 JSON 问题。

**不变量与失败模式：** `url` 必须是无尾斜杠 base URL；返回值必须是 JSON object；两个 endpoint 都失败时必须 fail-fast。若吞掉异常并返回空 dict，后续会用错误默认值规划 GPU。

**Comment：** discovery 的可靠性来自两个动作：兼容旧/新 endpoint，同时对完全失败不降级。

### 2.3 `discover_external_engines`：从 server_info 推断 worker 拓扑

来源：slime/backends/sglang_utils/external.py L79-L104

**问题与约束：** Slime 需要知道每个外部 engine 的 URL、host、port、worker type、GPU 数和 PD bootstrap port；不同 server_info 字段名可能不同或缺失。

**设计选择：** 遍历地址，先 normalize 再拉 server_info；TP/PP 支持短字段和长字段两套名称，GPU 数优先显式字段，否则退化为 `tp_size * pp_size`；最后构造 `ExternalEngineInfo`。

**Explain：** 这一步把用户提供的字符串地址转成结构化拓扑对象。`server_info` 原始 dict 也被保留，方便后续调试和 sanity check。

**Code：**

```python
def discover_external_engines(addrs: list[str], timeout: float = 30.0) -> list[ExternalEngineInfo]:
    infos = []
    for addr in addrs:
        url = normalize_external_engine_addr(addr)
        parsed = urlparse(url)
        assert parsed.hostname is not None and parsed.port is not None
        server_info = get_server_info(url, timeout=timeout)

        pp_size = int(server_info.get("pp_size") or server_info.get("pipeline_parallel_size") or 1)
        tp_size = int(server_info.get("tp_size") or server_info.get("tensor_parallel_size") or 1)
        num_gpus = int(server_info.get("num_gpus") or server_info.get("num_gpus_per_engine") or tp_size * pp_size)
        bootstrap_port = server_info.get("disaggregation_bootstrap_port")
        bootstrap_port = int(bootstrap_port) if bootstrap_port is not None else None

        infos.append(
            ExternalEngineInfo(
                url=url,
                host=parsed.hostname,
                port=parsed.port,
                worker_type=_infer_worker_type(server_info),
                num_gpus=num_gpus,
                disaggregation_bootstrap_port=bootstrap_port,
                server_info=server_info,
            )
        )
    return infos
```

**代码逻辑：** 对每个地址，函数得到规范化 URL 和 parsed host/port，读取 server_info，提取 PP、TP、GPU 数和 bootstrap port，并把这些值连同 worker type 与原始 server_info 放入列表。

**为什么这样写：** 外部 engine 不由 Slime launch，本地没有 Ray placement 信息可用。只能通过 server_info 建立逻辑资源视图，并用字段兼容处理不同 SGLang 版本。

**不变量与失败模式：** host/port 必须存在；`num_gpus` 必须能转为 int；PD prefill worker 需要 bootstrap port。若 server_info 缺字段且 TP/PP 也不可信，Slime 的 GPU 计数会低估或高估外部容量。

**Comment：** `discover_external_engines` 是 external 模式的拓扑来源，后面所有 engine count 和 GPU offset 都由它派生。

### 2.4 `apply_external_engine_info_to_args`：把拓扑写回 args

来源：slime/backends/sglang_utils/external.py L107-L131

**问题与约束：** 发现结果需要跨 Ray 进程和多个模块共享；后续代码不能一直持有 dataclass 对象，也不能每次都重新探测 HTTP。

**设计选择：** 调用 discovery 后，把 `ExternalEngineInfo` 序列化成 dict 列表写入 `args.rollout_external_engine_infos`，同时写入 engine 数和 GPU 总数。

**Explain：** 这一步把一次性 discovery 的结果固化到 `args`。Ray actor 创建、HTTP client 并发数和 Router 管理都从这些字段读取。

**Code：**

```python
def apply_external_engine_info_to_args(args, logger=None) -> None:
    """Detect external engines and store the derived topology on ``args``."""
    addrs = args.rollout_external_engine_addrs
    if not addrs:
        raise ValueError("apply_external_engine_info_to_args requires --rollout-external-engine-addrs.")

    infos = discover_external_engines(addrs)
    if not infos:
        raise ValueError("--rollout-external-engine-addrs did not contain any engines.")

    args.rollout_external_engine_infos = [info.to_dict() for info in infos]
    args.rollout_num_engines = len(infos)
    args.rollout_num_gpus = sum(info.num_gpus for info in infos)

    if logger is not None:
        summary = [
            {
                "url": info.url,
                "worker_type": info.worker_type,
                "num_gpus": info.num_gpus,
                "disaggregation_bootstrap_port": info.disaggregation_bootstrap_port,
            }
            for info in infos
        ]
        logger.info(f"Detected external SGLang engines: {summary}")
```

**代码逻辑：** 函数先要求地址存在，再执行 discovery；空结果直接报错。成功后把 infos 转成 dict 写回 args，计算 engine 数和 GPU 总数，并可选打印摘要。

**为什么这样写：** args 是 Slime 各组件的共享配置载体。把 dataclass 转 dict 可以安全跨进程序列化，也避免 Ray actor 反序列化自定义对象版本问题。

**不变量与失败模式：** 地址列表不能为空；discover 结果不能为空；`rollout_num_gpus` 是外部逻辑 GPU 总数，不代表 Slime 本地占用。若不写回 args，后续模块会按内置 engine 模式推断资源。

**Comment：** external discovery 的输出被压缩成三类字段：原始 infos、engine count、GPU count。

### 2.5 `external_engine_init_kwargs`：为代理 actor 构造 init 参数

来源：slime/backends/sglang_utils/external.py L46-L55

**问题与约束：** 外部 engine 已经启动，Slime 不能再给它分配 NCCL 端口或重写 host/port；但 `SGLangEngine.init()` 仍需要一组参数来完成 sanity check 和 Router 注册。

**设计选择：** 以外部 engine 的 host/port 构造 `dist_init_addr`，把 `nccl_port` 设为 `None`，并在 worker type 为 `prefill` 时附带 `disaggregation_bootstrap_port`。

**Explain：** 这个函数把 `ExternalEngineInfo` 转成 `SGLangEngine.init()` 能接受的 kwargs。它表达的是“连接已有 engine”，不是“启动新 engine”。

**Code：**

```python
def external_engine_init_kwargs(info: ExternalEngineInfo) -> dict:
    init_kwargs = {
        "dist_init_addr": f"{info.host}:{info.port}",
        "nccl_port": None,
        "host": info.host,
        "port": info.port,
    }
    if info.worker_type == "prefill":
        init_kwargs["disaggregation_bootstrap_port"] = info.disaggregation_bootstrap_port
    return init_kwargs
```

**代码逻辑：** 函数构造基础 dict，包括 dist init address、空 NCCL port 和 server host/port；prefill worker 额外写入 bootstrap port，最后返回 kwargs。

**为什么这样写：** 内置模式由 Slime 分配端口并 launch 进程；external 模式的端口来自已经运行的服务。把 init kwargs 分开构造，可以避免外部路径误用内置启动参数。

**不变量与失败模式：** `dist_init_addr` 必须指向外部 engine 自身；`nccl_port=None` 表示不由 Slime 新分配；prefill worker 缺 bootstrap port 时后续注册应失败。若把内置模式端口分配逻辑套到 external，会和已有服务冲突。

**Comment：** 这是 external 与 normal engine 的参数分界线：external 只描述已有服务，不创建服务。

---

## 3. 外部 engine 代理与 Router 注册

### 3.1 `start_external_rollout_servers`：创建零 GPU Ray 代理 actor

来源：slime/backends/sglang_utils/external.py L178-L232

**问题与约束：** Slime 仍需要一个 Ray actor 表示每个 rollout engine，用于统一 weight update、health、init handle 等管理接口；但实际 GPU 已在外部 SGLang 进程中占用，Ray actor 不能再申请 GPU。

**设计选择：** 先用外部 infos 启动 Router，并把 router ip/port 写入 args；随后为每个 external engine 创建 `num_gpus=0` 的 `SGLangEngine` actor，记录逻辑 GPU count/offset，并异步调用 `init.remote(**external_engine_init_kwargs(...))` 注册到 Router。

**Explain：** 这些 Ray actor 是控制面代理，不是推理进程。它们让 RolloutManager 可以复用内置 engine 管理接口，同时不占用训练 PG 的 GPU。

**Code：**

```python
def start_external_rollout_servers(args, *, start_router) -> tuple[dict[str, ExternalRolloutServer], list]:
    import ray

    from slime.backends.sglang_utils.sglang_engine import SGLangEngine
    from slime.ray.utils import add_default_ray_env_vars

    infos = external_engine_infos_from_args(args)
    router_ip, router_port = start_router(args, has_pd_disaggregation=any(info.is_pd_worker for info in infos))
    args.sglang_router_ip = router_ip
    args.sglang_router_port = router_port

    engines = []
    engine_gpu_counts = []
    engine_gpu_offsets = []
    init_handles = []
    RolloutRayActor = ray.remote(SGLangEngine)
    gpu_offset = 0
    for rank, info in enumerate(infos):
        rollout_engine = RolloutRayActor.options(
            num_cpus=0.2,
            num_gpus=0,
            runtime_env={"env_vars": add_default_ray_env_vars()},
        ).remote(
            args=args,
            rank=rank,
            worker_type=info.worker_type,
            base_gpu_id=0,
            num_gpus_per_engine=info.num_gpus,
        )
        engines.append(rollout_engine)
        engine_gpu_counts.append(info.num_gpus)
        engine_gpu_offsets.append(gpu_offset)
        gpu_offset += info.num_gpus
        init_handles.append(
            rollout_engine.init.remote(
                **external_engine_init_kwargs(info),
                router_ip=router_ip,
                router_port=router_port,
            )
        )

    args.sglang_model_routers = {"default": (router_ip, router_port)}
    servers = {
        "default": ExternalRolloutServer(
            engines=engines,
            engine_gpu_counts=engine_gpu_counts,
            engine_gpu_offsets=engine_gpu_offsets,
            router_ip=router_ip,
            router_port=router_port,
            model_name="default",
            update_weights=True,
            num_new_engines=len(engines),
        )
    }
    return servers, init_handles
```

**代码逻辑：** 函数读取 args 中的 external infos，启动 Router，创建每个 engine 的零 GPU actor，累计逻辑 GPU offset，提交 init 远程调用，并组装 `ExternalRolloutServer` 返回给上层。

**为什么这样写：** RolloutManager 需要统一管理对象，而不是一堆裸 HTTP 地址。零 GPU actor 让 Slime 保持同一控制面抽象，同时把数据面计算留在外部服务。

**不变量与失败模式：** Ray actor 必须 `num_gpus=0`；`engine_gpu_offsets` 按外部逻辑 GPU 累加；Router 的 PD 模式由是否存在 PD worker 决定。若 actor 申请 GPU，会错误占用训练集群资源；若 init handle 不等待，Router 可能尚未注册 worker 就开始 generate。

**Comment：** external rollout server 的本质是代理层：Slime 管控制面，外部 SGLang 管推理面。

### 3.2 `SGLangEngine._init_external`：不 launch，只 sanity check 并注册

来源：slime/backends/sglang_utils/sglang_engine.py L166-L197

**问题与约束：** 内置模式会启动 SGLang server；external 模式必须避免重复启动，只能校验外部 server 参数是否与 Slime 期望一致，然后把它挂到 Router。

**设计选择：** `init` 根据 `args.rollout_external` 分派到 `_init_external`；external 分支拉取 server_info，对指定字段做 assert，比对通过后调用 `_register_to_router`。

**Explain：** 这里是 fail-fast 保护。外部 server 可能用不同 TP、PP、模型或参数启动，Slime 必须在生成前发现不一致，而不是生成时出现 silent mismatch。

**Code：**

```python
        if self.args.rollout_external:
            self._init_external(server_args_dict, external_engine_need_check_fields=external_engine_need_check_fields)
        else:
            self._init_normal(server_args_dict)

        # Warm the host-local base off the actor's main thread: sglang serves the first rollout from
        # its init-loaded weights, so the materialize (a full base copy) only has to finish before
        # the first delta reload. init_local_checkpoint is idempotent and flock-guarded, so the first
        # sync_local_checkpoint either finds it done or blocks on the same lock — no join needed.
        if self.args.update_weight_mode == "delta" and self.args.update_weight_transport == "disk":
            from slime.utils.disk_delta import init_local_checkpoint

            threading.Thread(
                target=init_local_checkpoint,
                args=(self.args.update_weight_local_checkpoint_dir, self.args.hf_checkpoint),
                daemon=True,
            ).start()

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
```

**代码逻辑：** 初始化入口按 external flag 选择 `_init_external` 或 `_init_normal`。external helper 内部定义 sanity check 函数，遍历待检查字段，把实际 server_info 与期望 server args 对比；通过后注册 Router。

**为什么这样写：** 外部服务的生命周期在 Slime 之外，唯一安全做法是把它视为已启动黑盒并做参数握手。assert 能在启动阶段暴露配置漂移，避免训练和 rollout 使用不同模型拓扑。

**不变量与失败模式：** external 分支不得调用 `_init_normal`；检查字段集合必须排除外部服务天然不同的端口类字段；server_info endpoint 必须可达。若 sanity check 太宽，会漏掉 TP/PP 等关键不一致；太严则会拒绝合法外部端口差异。

**Comment：** `_init_external` 是控制面接管点：验证外部 server，再把它纳入 Slime Router。

### 3.3 `_register_to_router`：普通 worker 与 PD prefill 注册

来源：slime/backends/sglang_utils/sglang_engine.py L204-L232

**问题与约束：** Router 需要知道每个 worker URL 与 worker type；旧 router 不支持 PD disaggregation，新 router 的 prefill worker 还必须提供 bootstrap port。

**设计选择：** encoder worker 直接跳过；`node_rank==0` 且 router 可用时，旧版本走 `/add_worker?url=...` 并断言 regular；新版本 POST `/workers` JSON，prefill 时强制检查 `disaggregation_bootstrap_port`。

**Explain：** 注册逻辑同时兼容 router API 版本和 PD worker 类型。Prefill 缺 bootstrap port 时立即抛错，因为 PD Router 无法完成 KV transfer 路由。

**Code：**

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

**代码逻辑：** 函数先排除 encoder；只有 rank 0 负责注册。旧 router 使用 query 参数添加 worker；新 router 构造 JSON payload，按 worker type 写入 bootstrap port，最后检查 HTTP status。

**为什么这样写：** 多 rank engine 只需一个对外 worker URL 注册；Router API 在版本间变化，external 模式必须兼容。PD prefill 的 bootstrap port 是功能性必需字段，不应由 Router 猜测。

**不变量与失败模式：** 旧 router 只能 regular worker；prefill worker 必须带 bootstrap port；HTTP 注册失败必须抛错。若所有 rank 都注册，会重复添加同一 engine；若 prefill 未带 bootstrap port，PD 请求无法正确路由。

**Comment：** Router 注册把外部 SGLang 从“可访问 HTTP 服务”变成“Slime rollout router 的 worker”。

---

## 4. HTTP 通信与地址绑定

### 4.1 `init_http_client`：按 engine 数设置 async client 并发

来源：slime/utils/http_utils.py L213-L231

**问题与约束：** rollout generate 通过 HTTP 发往 Router；外部 engine 数越多，客户端连接池也要按 engine 数扩展。内部通信不应走系统代理。

**设计选择：** 用 `get_rollout_num_engines(args)` 计算 engine 数，设置 `_client_concurrency = sglang_server_concurrency * num_engines`；创建全局 `httpx.AsyncClient`，设置 `trust_env=False`，可选启用 distributed POST。

**Explain：** HTTP client 是 rollout 侧的共享资源。连接池大小按 engine 数放大，避免多 engine 模式下客户端自身成为瓶颈。

**Code：**

```python
def init_http_client(args):
    """Initialize HTTP client and optionally enable distributed POST via Ray."""
    global _http_client, _client_concurrency, _distributed_post_enabled
    num_engines = get_rollout_num_engines(args)
    if num_engines <= 0:
        return

    _client_concurrency = args.sglang_server_concurrency * num_engines
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=_client_concurrency),
            timeout=httpx.Timeout(None),
            trust_env=False,  # internal SGLang comm only — never route through system proxy
        )

    # Optionally initialize distributed POST via Ray without changing interfaces
    if args.use_distributed_post:
        _init_ray_distributed_post(args)
        _distributed_post_enabled = True
```

**代码逻辑：** 函数读取 rollout engine 数，0 或负数直接返回；否则更新全局并发数，懒创建 AsyncClient，并在配置打开时初始化 Ray distributed POST。

**为什么这样写：** 外部 engine 模式下计算资源在外部，但 HTTP 请求仍从 Slime 发出。按 engine 数扩展连接池，可以让客户端并发跟上后端容量；禁用环境代理避免内部地址被错误代理。

**不变量与失败模式：** `get_rollout_num_engines` 必须已能读到 external discovery 写入的数量；全局 client 只初始化一次；`trust_env=False` 防止代理污染。若 discovery 未提前写 engine 数，这里会错误跳过 client 初始化或并发过小。

**Comment：** external 模式不是只改启动路径，HTTP 客户端容量也要跟着外部 engine 数变化。

### 4.2 `_post`：带重试的 JSON POST

来源：slime/utils/http_utils.py L165-L198

**问题与约束：** rollout HTTP 调用可能遇到 transient 网络错误、worker 重启或 Router 短暂不可用；响应可能是 JSON，也可能是非 JSON 文本或 bytes。

**设计选择：** `_post` 最多重试 60 次，每次用 `client.post(..., json=payload)` 发送；成功后读取内容并尝试 JSON decode，失败则退化为文本；异常时记录并 sleep 1 秒，最终超过重试上限再抛出。

**Explain：** 这是 generate/RM 等路径的基础容错层。它不改变接口，只把临时失败转成有限次数重试。

**Code：**

```python
async def _post(client, url, payload, max_retries=60, headers=None):
    retry_count = 0
    while retry_count < max_retries:
        response = None
        try:
            response = await client.post(url, json=payload or {}, headers=headers)
            response.raise_for_status()
            content = await response.aread()
            try:
                output = json.loads(content)
            except json.JSONDecodeError:
                output = content.decode() if isinstance(content, bytes) else content
        except Exception as e:
            retry_count += 1

            if isinstance(e, httpx.HTTPStatusError):
                response_text = e.response.text
            else:
                response_text = None

            logger.info(
                f"Error: {e}, retrying... (attempt {retry_count}/{max_retries}, url={url}, response={response_text})"
            )
            if retry_count >= max_retries:
                logger.info(f"Max retries ({max_retries}) reached, failing... (url={url})")
                raise e
            await asyncio.sleep(1)
            continue
        finally:
            if response is not None:
                await response.aclose()
        break

    return output
```

**代码逻辑：** 函数循环发送 POST，成功时检查 status、读取 body、解析 JSON 或文本；异常时增加计数、记录响应文本、按上限决定重试或抛出；finally 确保 response 关闭。

**为什么这样写：** 外部 engine 通常跨节点或跨集群部署，网络抖动概率高于同进程调用。有限重试让短暂不可用不直接打断训练，同时保留最大失败时间边界。

**不变量与失败模式：** 成功响应必须在 break 前设置 `output`；每个 response 都要关闭；重试上限不能无限。若不关闭 response，长时间 rollout 会耗尽连接；若吞掉最终异常，训练会继续使用无效结果。

**Comment：** `_post` 是外部 engine 稳定性的下限保障：容忍短抖动，但不掩盖持续不可用。

### 4.3 `get_host_info`：允许环境变量覆盖 Router 绑定 IP

来源：slime/utils/http_utils.py L42-L46

**问题与约束：** Slime 启动 Router 时需要选择其他节点能访问的 host IP；容器、K8s 或多网卡环境中，自动探测的 IP 可能不符合用户网络拓扑。

**设计选择：** `get_host_info` 先获取 hostname，再检查 `SLIME_HOST_IP` 环境变量；若存在，直接返回 hostname 与覆盖 IP。

**Explain：** 这给部署系统一个显式控制点。用户可以通过环境变量指定 Router 对外地址，避免默认探测选错网卡。

**Code：**

```python
def get_host_info():
    hostname = socket.gethostname()

    if env_overwrite_local_ip := os.getenv(SLIME_HOST_IP_ENV, None):
        return hostname, env_overwrite_local_ip
```

**代码逻辑：** 函数先读取本机 hostname；随后检查环境变量，命中时立即返回覆盖 IP，不继续后续自动探测。

**为什么这样写：** 外部 engine 模式下，训练进程、Router 和 SGLang server 可能不在同一网络命名空间。显式覆盖比依赖自动 hostname 解析更可控。

**不变量与失败模式：** 覆盖 IP 必须能被外部 engine 或训练节点访问；环境变量命中时优先级最高；hostname 与 IP 成对返回。若覆盖成不可达地址，Router 虽能启动但 worker 注册或 generate 会失败。

**Comment：** 这是部署层 escape hatch：当自动网络探测不可信时，让用户指定 Router 地址。

---

## 5. 运维边界与官方示例

### 5.1 `RolloutHealthMonitor._check_engine_health`：Ray actor 健康检查模板

来源：slime/utils/health_monitor.py L145-L158

**问题与约束：** Rollout engine actor 可能卡死或 generate 失败；Slime 需要后台检测并在异常时清理 actor，避免后续 rollout 一直等待不可用 engine。

**设计选择：** 对每个 engine 调 `ray.get(engine.health_generate.remote(timeout=...))`；异常时记录错误并 kill actor，成功时只打 debug 日志；engine 为 `None` 则跳过。

**Explain：** 这段展示 Slime 对 rollout engine actor 的健康检查协议。external 模式下 actor 是控制面代理，health 语义仍通过 actor 方法统一表达。

**Code：**

```python
    def _check_engine_health(self, rollout_engine_id, engine) -> None:
        if engine is None:
            logger.info(f"Skipping health check for engine {rollout_engine_id} (None)")
            return

        try:
            ray.get(engine.health_generate.remote(timeout=self._check_timeout))
        except Exception as e:
            logger.error(
                f"Health check failed for rollout engine {rollout_engine_id} (ray timeout or error). Killing actor. Exception: {e}"
            )
            self._kill_engine(rollout_engine_id=rollout_engine_id)
        else:
            logger.debug(f"Health check passed for rollout engine {rollout_engine_id}")
```

**代码逻辑：** 函数先跳过空 engine；否则同步等待远程健康检查。异常路径记录并调用 `_kill_engine`，成功路径记录通过。

**为什么这样写：** Ray actor 是 Slime 控制面的最小管理单元。即使推理服务在外部，Slime 仍需要知道代理 actor 是否能完成健康 probe。

**不变量与失败模式：** `health_generate` 必须接受 timeout；异常必须导致 actor 被清理；空 engine 不能强查。若 health check 只记录不 kill，后续调度可能持续把请求交给坏 actor。

**Comment：** health monitor 不替代外部平台监控，但它给 Slime 的 rollout 控制面提供了故障隔离动作。

### 5.2 文档示例：先启动 SGLang，再把地址交给训练

来源：docs/en/advanced/external-rollout-engines.md L24-L35

**问题与约束：** external engine 的部署顺序与内置模式相反：必须先有可访问的 SGLang servers，再启动 Slime 训练并传入地址。

**设计选择：** 文档示例先用 `python -m sglang.launch_server` 启两个端口，再在 `train.py` 中传 `--rollout-external-engine-addrs host1:10090 host2:10091`。

**Explain：** 这份示例对应源码里的设计：Slime 不负责 launch server，只负责 discovery、Router 接入和后续 HTTP generate。

**Code：**

```bash
python -m sglang.launch_server --model-path /path/to/model --port 10090 ...
python -m sglang.launch_server --model-path /path/to/model --port 10091 ...

python train.py \
  --rollout-external-engine-addrs host1:10090 host2:10091 \
  ...
```

**代码逻辑：** 用户先在外部环境启动两个 SGLang server；训练命令只传地址列表，不再要求 Slime 分配 rollout engine GPU。

**为什么这样写：** 这种模式适合推理资源与训练资源分离的场景，例如独立推理集群或已由平台托管的 SGLang 服务。源码中的零 GPU Ray actor 和 server_info discovery 正是为这个部署顺序服务。

**不变量与失败模式：** 训练进程必须能 HTTP 访问这些 host/port；外部 server 参数要与 Slime 期望一致；地址列表要覆盖需要接入的所有 engines。若先启动训练再启动 server，args 阶段 discovery 会失败。

**Comment：** 文档示例是运行层闭环：先有外部 SGLang，再让 Slime 发现并接管。
