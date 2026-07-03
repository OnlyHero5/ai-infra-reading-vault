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
updated: 2026-07-02
---

# External Engines · 源码走读

> 按 **启动时序** 精读：CLI 解析 → 引擎发现 → 启动 servers → SGLangEngine 对接 → HTTP 通信。

---

## 1. CLI 参数与 early discovery

**Explain：** `--rollout-external-engine-addrs` 接受一个或多个 `host:port`；在 `parse_args` 末尾、Placement Group 创建 **之前** 就会 HTTP 探测各 engine，以便正确设置 `rollout_num_gpus` 供资源规划使用。

**Code：**

```python
## 来源：slime/utils/arguments.py L555-L561
            parser.add_argument(
                "--rollout-external-engine-addrs",
                type=str,
                default=None,
                nargs="+",
                help="Address and ports of the external engines.",
            )
```

**Comment：**

- 与 `--sglang-config` 互斥（arguments 校验逻辑在其他处 assert）。
- `debug_train_only` 时跳过 `apply_external_engine_info_to_args`，避免无 rollout 时误探测。

---

## 2. 地址规范化与 server_info 拉取

**Explain：** `normalize_external_engine_addr` 统一为 `http://host:port`；`get_server_info` 兼容 SGLang 两个 endpoint 名称。

**Code：**

```python
## 来源：slime/backends/sglang_utils/external.py L32-L44
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

```python
## 来源：slime/backends/sglang_utils/external.py L58-L67
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

**Comment：**

- IPv6 地址须带方括号，如 `http://[2001:db8::1]:10090`。
- 此函数也被 `SGLangEngine._init_external` 用于 sanity check。

---

## 3. discover_external_engines：拓扑推断

**Explain：** 遍历地址列表，从 server_info 提取 TP/PP/GPU 数，构造 `ExternalEngineInfo` 列表。

**Code：**

```python
## 来源：slime/backends/sglang_utils/external.py L79-L104
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

**Comment：**

- `num_gpus` 优先读 server_info 显式字段，否则 `tp * pp` 估算。
- 完整 `server_info` dict 保存在 `ExternalEngineInfo.server_info` 供调试。

---

## 4. apply_external_engine_info_to_args

**Explain：** 将发现结果写回 `args`，供 PG 布局、HTTP 客户端、RolloutManager 统一读取。

**Code：**

```python
## 来源：slime/backends/sglang_utils/external.py L107-L131
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

**Comment：**

- `rollout_num_gpus` 此处为 **逻辑 GPU 总数**（外部集群容量），非 Slime PG 占用数。
- `rollout_external_engine_infos` 序列化 dict 列表，Ray 跨进程传递安全。

---

## 5. external_engine_init_kwargs

**Explain：** 为 `SGLangEngine.init()` 构造 kwargs；external 模式下 `dist_init_addr` 指向外部 engine 自身地址，`nccl_port=None`。

**Code：**

```python
## 来源：slime/backends/sglang_utils/external.py L46-L55
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

**Comment：**

- 与内置 engine 不同：内置模式 Slime 分配 `nccl_port` 和 `dist_init_addr` 给新 launch 的进程。
- prefill worker 必须携带 bootstrap port 供 PD Router 路由 KV transfer。

---

## 6. start_external_rollout_servers

**Explain：** 核心启动函数——启动 Router、创建零 GPU Ray actor、异步 `init()` 注册到 Router。

**Code：**

```python
## 来源：slime/backends/sglang_utils/external.py L178-L232
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

**Comment：**

- `num_gpus=0`：Ray 不在训练 PG 上绑定 GPU；actor 纯 CPU 轻量代理。
- `has_pd_disaggregation=any(info.is_pd_worker)` 使 Router 开启 `pd_disaggregation=True`。
- 返回 `init_handles` 供 `RolloutManager` 统一 `ray.get` 等待注册完成。

---

## 7. SGLangEngine._init_external

**Explain：** external 分支不调用 `launch_server_process`；改为拉取 server_info 做字段 sanity check，然后注册 Router。

**Code：**

```python
## 来源：slime/backends/sglang_utils/sglang_engine.py L166-L197
        if self.args.rollout_external:
            self._init_external(server_args_dict, external_engine_need_check_fields=external_engine_need_check_fields)
        else:
            self._init_normal(server_args_dict)

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

**Comment：**

- `_EXTERNAL_ENGINE_SKIP_CHECK_FIELDS` 中的字段（如 port）不参与 assert——外部 engine 已自行启动。
- 若 Slime CLI 与外部 server 启动参数不一致（如 TP 不同），此处 assert 失败，**fail-fast** 避免 silent mismatch。

---

## 8. Router 注册（PD 与普通）

**Explain：** `node_rank==0` 的 engine 向 Router POST worker 信息；prefill 需 `bootstrap_port`。

**Code：**

```python
## 来源：slime/backends/sglang_utils/sglang_engine.py L204-L232
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

**Comment：**

- external PD 测试见 `test_qwen3_4B_external_pd.py`：先 launch prefill/decode，再传 addrs。
- Router 由 Slime `_start_router` 在训练节点 spawn 子进程启动。

---

## 9. init_http_client 与 post 重试

**Explain：** RolloutManager 构造时调用；建立全局 async HTTP 客户端，generate 路径通过 `post()` 发 JSON。

**Code：**

```python
## 来源：slime/utils/http_utils.py L213-L231
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

```python
## 来源：slime/utils/http_utils.py L165-L198
async def _post(client, url, payload, max_retries=60, headers=None):
    retry_count = 0
    while retry_count < max_retries:
        response = None
        try:
            response = await client.post(url, payload or {}, headers=headers)
            response.raise_for_status()
            content = await response.aread()
            try:
                output = json.loads(content)
            except json.JSONDecodeError:
                output = content.decode() if isinstance(content, bytes) else content
        except Exception as e:
            retry_count += 1
            # ... logging ...
            if retry_count >= max_retries:
                raise e
            await asyncio.sleep(1)
            continue
        finally:
            if response is not None:
                await response.aclose()
        break

    return output
```

**Comment：**

- 默认最多 60 次重试、间隔 1s——容忍 transient 网络抖动。
- distributed POST 用 `await obj_ref` 直接 await Ray ObjectRef，避免线程池瓶颈（见 http_utils 注释）。

---

## 10. RolloutHealthMonitor（内置 engine 专用）

**Explain：** 后台 daemon 线程周期性 `ray.get(engine.health_generate.remote())`；失败则 kill actor。external 模式因 `server_groups` 为空 **不会实例化**。

**Code：**

```python
## 来源：slime/utils/health_monitor.py L145-L158
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

**Comment：**

- `pause`/`resume` 与 `RolloutManager.offload`/`generate` 联动——offload 期间无法 health check。
- `_check_first_wait` 给大 MoE 模型编译留 grace period。

---

## 11. get_host_info：Router 绑定地址

**Explain：** Slime 启动 Router 时需绑定可达 IP；支持 `SLIME_HOST_IP` 覆盖与 IPv4/IPv6 严格模式。

**Code：**

```python
## 来源：slime/utils/http_utils.py L42-L46
def get_host_info():
    hostname = socket.gethostname()

    if env_overwrite_local_ip := os.getenv(SLIME_HOST_IP_ENV, None):
        return hostname, env_overwrite_local_ip
```

**Comment：**

- UDP probe → hostname resolution → loopback fallback 三级策略。
- `_wrap_ipv6` 在 Router URL 中包裹 IPv6 地址。

---

## 12. 典型部署命令（文档摘录）

**Explain：** 官方 roadmap 给出的最小 external 启动示例。

**Code：**

```bash
## 来源：docs/en/advanced/external-rollout-engines.md L24-L35
python -m sglang.launch_server --model-path /path/to/model --port 10090 ...
python -m sglang.launch_server --model-path /path/to/model --port 10091 ...

python train.py \
  --rollout-external-engine-addrs host1:10090 host2:10091 \
  ...
```

**Comment：**

- disk 模式额外需要 `--update-weight-mode full --update-weight-transport disk --update-weight-disk-dir /shared/fs/...`。
- 训练 job 须能 **HTTP 访问** 所列 host:port（防火墙 / no_proxy 配置）。
