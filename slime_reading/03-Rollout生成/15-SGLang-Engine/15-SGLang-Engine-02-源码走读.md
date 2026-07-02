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
updated: 2026-07-02
---

# SGLang Engine · 源码走读

> **阅读顺序：** 启动子进程 → `init` 分支 → Router 注册 → HTTP 请求封装 → 权重 API → shutdown

---

## 1. `launch_server_process` — 启动 SGLang 子进程

**Explain：** 根据 `encoder_only` 选择 encode server 或标准 HTTP server；node 0 阻塞等待 `/health_generate` 200，其他 node 立即返回（由 SGLang 内部 join）。

**Code：**

```python
# 来源：slime/backends/sglang_utils/sglang_engine.py L52-L99
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
```

**Comment：**

- `spawn` 避免 fork CUDA 上下文；与 Ray worker 默认一致。
- health 轮询 2s 间隔；子进程提前退出则抛异常，RolloutManager `ray.get(init_handles)` 失败。
- encoder-only 路径用于 VLM encoder 分离部署（PD / 多模态场景）。

---

## 2. `SGLangEngine.init` 主流程

**Explain：** 计算 `ServerArgs` 字典、启动或对接外部 engine、可选预热 delta checkpoint。

**Code：**

```python
# 来源：slime/backends/sglang_utils/sglang_engine.py L148-L182
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

**Comment：**

- `dist_init_addr` / `nccl_port` 由 RolloutManager 端口分配器传入，供 SGLang 多节点 TP 使用。
- delta+disk 模式下后台线程预热本地 checkpoint，与首次 `sync_local_checkpoint` 竞态由 flock 保证。

---

## 3. `_init_normal` 与 `_init_external`

**Code：**

```python
# 来源：slime/backends/sglang_utils/sglang_engine.py L184-L202
    def _init_external(self, expect_server_args, external_engine_need_check_fields):
        actual_server_args = get_server_info(f"http://{self.server_host}:{self.server_port}")
        # ... assert fields in external_engine_need_check_fields ...
        self._register_to_router(expect_server_args)

    def _init_normal(self, server_args_dict):
        logger.info(f"Launch HttpServerEngineAdapter at: {self.server_host}:{self.server_port}")
        self.process = launch_server_process(ServerArgs(**server_args_dict))
        self._register_to_router(server_args_dict)
```

**Comment：**

- external 跳过字段见 `_EXTERNAL_ENGINE_SKIP_CHECK_FIELDS`（host/port/tp_size 等由外部决定）。
- normal 路径 `self.process` 供 `shutdown` 时 `kill_process_tree`。

---

## 4. `_compute_server_args` — ServerArgs 组装

**Code：**

```python
# 来源：slime/backends/sglang_utils/sglang_engine.py L578-L623
    _gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
    nnodes = max(1, _gpus_per_engine // args.num_gpus_per_node)
    node_rank = rank % nnodes
    base = base_gpu_id if base_gpu_id is not None else get_base_gpu_id(args, rank)
    base = _to_local_gpu_id(base)
    kwargs = {
        "model_path": args.hf_checkpoint,
        "enable_memory_saver": args.offload_rollout,
        "host": host,
        "port": port,
        "nccl_port": nccl_port,
        "nnodes": nnodes,
        "node_rank": node_rank,
        "dist_init_addr": dist_init_addr,
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
        kwargs["disaggregation_bootstrap_port"] = disaggregation_bootstrap_port
    elif worker_type == "decode":
        kwargs["disaggregation_mode"] = "decode"
    elif worker_type == "encoder":
        kwargs["encoder_only"] = True
```

**Comment：**

- `node_rank = rank % nnodes`：同一 engine 的多节点 actor 共享 logical rank 空间。
- 后续循环将 `args.sglang_{field}` 与 YAML `sglang_overrides` 合并进 kwargs。

---

## 5. Router 注册

**Code：**

```python
# 来源：slime/backends/sglang_utils/sglang_engine.py L204-L232
    def _register_to_router(self, server_args_dict):
        if self.worker_type == "encoder":
            return
        if self.node_rank == 0 and self.router_ip and self.router_port:
            worker_url = f"http://{self.server_host}:{self.server_port}"
            payload = {"url": worker_url, "worker_type": self.worker_type}
            if self.worker_type == "prefill":
                payload["bootstrap_port"] = server_args_dict.get("disaggregation_bootstrap_port")
            response = requests.post(
                f"http://{self.router_ip}:{self.router_port}/workers",
                json=payload,
            )
            response.raise_for_status()
```

**Comment：**

- 仅 **node_rank==0** 注册；encoder 不注册（无 generate 路由）。
- 旧版 router ≤0.2.1 使用 `/add_worker?url=`（源码中有版本分支）。

---

## 6. `_make_request` — HTTP POST 封装

**Code：**

```python
# 来源：slime/backends/sglang_utils/sglang_engine.py L234-L254
    def _make_request(self, endpoint: str, payload: dict | None = None):
        if self.node_rank != 0:
            return
        url = f"http://{self.server_host}:{self.server_port}/{endpoint}"
        response = requests.post(url, json=payload or {})
        response.raise_for_status()
        return response.json()
```

**Comment：**

- 所有权重更新 API 均经此路径；非 node 0 直接 no-op（SGLang 子进程内 NCCL rank 仍参与 broadcast）。

---

## 7. 权重更新 API

### 7.1 `init_weights_update_group`

```python
# 来源：slime/backends/sglang_utils/sglang_engine.py L439-L450
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
```

### 7.2 `update_weights_from_distributed`

```python
# 来源：slime/backends/sglang_utils/sglang_engine.py L464-L488
    def update_weights_from_distributed(
        self, names, dtypes, shapes, group_name, flush_cache=False, weight_version=None, load_format=None,
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
        return self._make_request("update_weights_from_distributed", payload)
```

### 7.3 `update_weights_from_tensor` / `update_weights_from_disk`

```python
# 来源：slime/backends/sglang_utils/sglang_engine.py L278-L301, L415-L437
    def update_weights_from_tensor(self, serialized_named_tensors, load_format=None, flush_cache=False, weight_version=None):
        payload = {"serialized_named_tensors": serialized_named_tensors, "load_format": load_format, "flush_cache": flush_cache}
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return self._make_request("update_weights_from_tensor", payload)

    def update_weights_from_disk(self, model_path, load_format=None, weight_version=None, files=None):
        payload = {"model_path": model_path}
        if load_format is not None:
            payload["load_format"] = load_format
        if files is not None:
            payload["files"] = files
        return self._make_request("update_weights_from_disk", payload)
```

**Comment：**

- distributed：HTTP 只传 metadata；tensor 由训练 rank 0 `dist.broadcast` 推送。
- tensor：serialized meta + GPU 侧 IPC copy（colocate 典型路径）。
- disk/delta：路径 + files 列表，SGLang 从 FS reload。

---

## 8. 运行时控制

**Code：**

```python
# 来源：slime/backends/sglang_utils/sglang_engine.py L303-L322, L380-L398, L490-L498
    def flush_cache(self):
        if self.node_rank != 0:
            return
        for _ in range(60):
            response = requests.get(f"http://{self.server_host}:{self.server_port}/flush_cache")
            if response.status_code == 200:
                break
            time.sleep(1)
        else:
            raise TimeoutError("Timeout while flushing cache.")

    def release_memory_occupation(self):
        self.flush_cache()
        return self._make_request("release_memory_occupation")

    def pause_generation(self):
        response = requests.post(f"http://{self.server_host}:{self.server_port}/pause_generation", json={})
        response.raise_for_status()
```

**Comment：**

- update_weights 前需 `pause_generation` + `flush_cache`，否则 flush 可能长期非 200。
- `release_memory_occupation` 配合 `--offload-rollout` 释放 weights/kv_cache 显存。

---

## 9. `shutdown`

**Code：**

```python
# 来源：slime/backends/sglang_utils/sglang_engine.py L329-L361
    def shutdown(self):
        if self.args.rollout_external:
            return
        if self.worker_type != "encoder" and self.node_rank == 0:
            # ... remove worker from router (version-dependent API) ...
        kill_process_tree(self.process.pid)
```

---

## 10. 调用链小结

```
RolloutManager.ServerGroup.start_engines
  → ray.remote(SGLangEngine).remote(...)
  → engine.init.remote(...)
       → _compute_server_args → launch_server_process → _register_to_router
MegatronTrainRayActor.init
  → connect_rollout_engines_from_distributed
       → engine.init_weights_update_group.remote(...)  [本模块]
       → init_process_group(rank=0)                     [训练侧]
每 rollout step:
  → pause_generation / flush_cache
  → update_weights_from_distributed.remote + dist.broadcast
  → continue_generation
```
