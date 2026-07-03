---
type: batch-doc
module: 16-External-Engines
batch: "16"
doc_type: concept
title: "External Engines · 核心概念"
tags:
  - slime/batch/16
  - slime/module/external-engines
  - slime/doc/concept
updated: 2026-07-02
---

# External Engines · 核心概念

---

## 1. 架构位置

| 层级 | 组件 | 职责 |
|------|------|------|
| CLI | `--rollout-external-engine-addrs` | 声明外部 engine HTTP 地址列表 |
| 发现 | `external.py` | 探测 `/server_info`，推断 TP/PP/GPU 数与 worker 类型 |
| Ray 编排 | `start_external_rollout_servers` | 创建 **零 GPU** 的 `SGLangEngine` actor，对接外部进程 |
| HTTP 层 | `http_utils.py` | 异步 POST generate、Router 子进程、分布式 POST actor |
| 健康检查 | `health_monitor.py` | 内置 engine 的 `/health_generate` 轮询（**external 默认不启用**） |
| 训练闭环 | `update_weights` | NCCL 或 disk/delta 将 actor 权重推到外部 engine |

External 模式的核心边界：**Slime 不拥有 SGLang 进程生命周期**，只依赖 HTTP 端点与选定的权重传输通道。

---

## 2. 与 `--sglang-config` 的互斥

| 维度 | `--sglang-config` | `--rollout-external-engine-addrs` |
|------|-------------------|-----------------------------------|
| 谁 launch SGLang | Slime（`launch_server_process`） | 外部系统 |
| 多模型 / 冻结 reference | YAML 配置 `update_weights: false` | 仅 default 单模型 |
| PD 分离 | YAML `server_groups` | 探测 `disaggregation_mode` |
| GPU Placement Group | 预留 rollout bundle | **不占用** rollout GPU |
| Fault tolerance | `RolloutHealthMonitor` + recover | **不支持** recover |
| 典型场景 | 单集群 on-policy RL | 跨集群/跨 DC、异构 GPU |

---

## 3. 核心术语

| 术语 | 含义 |
|------|------|
| `ExternalEngineInfo` | 单个外部 engine 的发现结果：url、host、port、worker_type、num_gpus |
| `worker_type` | `regular` / `prefill` / `decode` / `encoder`；由 server_info 推断 |
| `ExternalRolloutServer` | 替代 `RolloutServer` 的轻量容器，持有 engines 列表与 Router 地址 |
| `rollout_external` | `args` 布尔标志，`rollout_external_engine_addrs is not None` |
| `dist_init_addr` | external 模式下指向 **外部 engine 自身** 的 host:port（非 Slime 分配） |
| `update_weight_transport` | `nccl`（同网段低延迟）或 `disk`（跨集群/异构 GPU） |

---

## 4. Worker 类型推断

**Explain：** Slime 不读 YAML，而是从 SGLang `/server_info` JSON 推断 worker 角色，决定 Router 注册 payload 是否带 `bootstrap_port`。

**Code：**

```python
## 来源：slime/backends/sglang_utils/external.py L70-L76
def _infer_worker_type(server_info: dict) -> str:
    if server_info.get("encoder_only"):
        return "encoder"
    mode = server_info.get("disaggregation_mode")
    if mode in ("prefill", "decode"):
        return mode
    return "regular"
```

**Comment：**

- PD 分离时 prefill worker 必须提供 `disaggregation_bootstrap_port`，否则 Router 注册失败。
- `encoder` worker 跳过 Router 注册（`SGLangEngine._register_to_router` 早退）。

---

## 5. Placement Group 布局（external 特例）

**Explain：** external 模式下 rollout GPU 不在 Slime PG 内，但 PG 仍按 actor GPU 数创建；`rollout_offset == actor_num_gpus` 使 rollout 相关索引为空 slice。

**Code：**

```python
## 来源：slime/ray/placement_group.py L106-L109
    if args.rollout_external:
        if args.debug_rollout_only:
            return 0, 0
        return actor_num_gpus, actor_num_gpus
```

**Comment：**

- `debug_rollout_only + external`：PG 大小为 0——纯探测外部 engine，不占 GPU。
- 这与 `--colocate` 完全不同：colocate 共享物理 GPU，external 是 **逻辑解耦**。

---

## 6. 权重同步选型（文档路线图）

| 目标 | 推荐参数 |
|------|----------|
| Trainer 与 engine 可建 NCCL group | `--update-weight-mode full --update-weight-transport nccl` |
| 不能 NCCL，共享文件系统 | `--update-weight-mode full --update-weight-transport disk` |
| 大模型跨 DC，减小传输量 | `--update-weight-mode delta --update-weight-transport disk` |
| 训练与 serving 异构 GPU/厂商 | external + **disk**（SGLang 热加载 HF/safetensors） |
| 同 DC 验证 delta 逻辑 | `--update-weight-mode delta --update-weight-transport nccl` |

**重要约束：** delta 模式 **不支持 `--colocate`**（colocate 用 CUDA IPC，delta 编码无法减少实际搬运）。

---

## 7. ExternalRolloutServer 与 Fault Tolerance

**Explain：** `ExternalRolloutServer` 故意将 `recover`、`offload`、`onload` 实现为空或 no-op，并 **不填充 `server_groups`**，因此 `RolloutManager` 不会为 external 创建 `RolloutHealthMonitor`。

**Code：**

```python
## 来源：slime/backends/sglang_utils/external.py L152-L165
    def recover(self):
        logger.warning("Fault tolerance is not supported for external rollout engines; skip recover.")

    def offload(self):
        return []

    def onload(self, tags: list[str] | None = None):
        return []
```

**Comment：**

- 外部 engine 崩溃由 **外部编排系统** 重启；Slime 侧 Ray actor 仅代理 HTTP，kill 后需人工或外部流程重建。
- `health_monitor.py` 仍服务内置 engine 路径（[[08-RolloutManager-00-MOC]] RolloutManager + `--use-fault-tolerance`）。

---

## 8. HTTP 客户端与并发

**Explain：** `init_http_client` 根据 engine 数量设置 `httpx.AsyncClient` 连接池上限；可选 `--use-distributed-post` 在每台 Ray 节点 spawn `_HttpPosterActor` 避免默认线程池瓶颈。

**Code：**

```python
## 来源：slime/utils/http_utils.py L201-L210
def get_rollout_num_engines(args) -> int:
    """Return the number of rollout HTTP engines behind the router."""
    if (num_engines := getattr(args, "rollout_num_engines", None)) is not None:
        return int(num_engines)

    rollout_num_gpus = getattr(args, "rollout_num_gpus", None) or 0
    rollout_num_gpus_per_engine = getattr(args, "rollout_num_gpus_per_engine", None) or 1
    if rollout_num_gpus <= 0:
        return 0
    return max(1, rollout_num_gpus // rollout_num_gpus_per_engine)
```

**Comment：**

- external 模式下 `apply_external_engine_info_to_args` 已设置 `rollout_num_engines`，走第一分支。
- `trust_env=False` 防止系统 HTTP 代理劫持内部 SGLang 通信。
