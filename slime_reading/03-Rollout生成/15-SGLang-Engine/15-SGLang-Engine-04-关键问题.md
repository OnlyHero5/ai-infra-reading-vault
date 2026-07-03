---
type: batch-doc
module: 15-SGLang-Engine
batch: "15"
doc_type: faq
title: "SGLang Engine · 关键问题"
tags:
  - slime/batch/15
  - slime/module/sglang-engine
  - slime/doc/faq
updated: 2026-07-02
---

# SGLang Engine · 关键问题

---

## Q1：为什么 `node_rank != 0` 的 actor 不发 HTTP？

**Explain：** 多节点 SGLang engine 中，每个 node 一个 Ray actor，但只有 node 0 绑定 HTTP 端口对外服务；权重 NCCL join 在 **SGLang 子进程内**按 TP rank 完成，不要求每个 node 的 actor 都 POST。

**Code：**

```python
## 来源：slime/backends/sglang_utils/sglang_engine.py L244-L245
        if self.node_rank != 0:
            return
```

**易错点：** 误以为 non-zero node actor 未参与权重 sync——实际上 SGLang 进程内 GPU 仍占 NCCL rank `rank_offset + local_rank`。

---

## Q2：`init_weights_update_group` 与 SGLang `nccl_port` 有什么区别？

| | 权重 update group | SGLang 推理 NCCL |
|--|-------------------|------------------|
| 建立时机 | Megatron init 后 `connect_rollout_engines` | `engine.init` → `_compute_server_args` |
| 端口 | 随机 `master_port` | RolloutManager 分配的 `nccl_port` |
| 参与者 | 训练 rank 0 + 所有 rollout GPU | 同一 engine 的 TP/PP rank |
| Slime 入口 | `init_weights_update_group` HTTP | `ServerArgs.nccl_port` 字段 |

混用两种 port 调试 NCCL 会完全误导。

---

## Q3：为什么 update_weights 前要 pause + flush？

**Explain：** 有 in-flight decode 时 `/flush_cache` 常返回非 200；未 pause 的 generate 可能与权重 reload 竞态，导致错误 logits 或 NCCL hang。

**正确顺序：**

```python
## 来源：update_weight_from_distributed.py L109-L111
ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
```

**错误做法：** 直接 `update_weights_from_distributed` 而不 pause——可能触发 `TimeoutError: Timeout while flushing cache`（60 次重试后）。

---

## Q4：external engine 模式下权重 API 还有效吗？

**Explain：** `--rollout-external` 时 Slime 不 spawn 子进程，但 **HTTP 端点仍可用**（外部 SGLang 必须暴露相同 API）。`shutdown` 不 kill 进程；sanity check 确保 tp_size 等与 Slime 预期一致。

**Code：**

```python
## 来源：slime/backends/sglang_utils/sglang_engine.py L166-L167, L329-L331
        if self.args.rollout_external:
            self._init_external(...)
    def shutdown(self):
        if self.args.rollout_external:
            return
```

---

## Q5：`destroy_weights_update_group` 为什么吞异常？

**Explain：** engine 刚创建尚未建组、或 connect 失败后的 cleanup 路径会调用 destroy；此时 SGLang 返回连接错误属正常。

**Code：**

```python
## 来源：slime/backends/sglang_utils/sglang_engine.py L452-L462
        try:
            return self._make_request("destroy_weights_update_group", {"group_name": group_name})
        except requests.exceptions.RequestException:
            pass
```

---

## Q6：colocate 下 GPU 映射错误会怎样？

**Explain：** `get_base_gpu_id` 与 Placement Group 的 `base_gpu_id` 必须一致；否则 SGLang TP 可能占用训练 GPU，导致 OOM 或 silent 性能下降。

**Code：**

```python
## 来源：slime/backends/sglang_utils/sglang_engine.py L26-L30
    if args.colocate:
        start_index = (rank * num_gpus) % args.num_gpus_per_node
    else:
        num_actor_gpus = ... 
        start_index = (num_actor_gpus + rank * num_gpus) % args.num_gpus_per_node
```

**对比：** RolloutManager 用 PG 的 `reordered_gpu_ids[gpu_index]` 覆盖 `_compute_server_args` 默认值——以 PG 为准。

---

## Q7：`server_control` 与 `SGLangEngine.flush_cache` 何时用哪个？

| 场景 | 推荐 |
|------|------|
| sync 训练 update_weights 前 | `pause_generation` + `flush_cache`（engine 方法，Megatron 已集成） |
| async rollout 需强制 abort | `abort_servers_until_idle`（async，读 `/v1/loads`） |
| offload rollout 显存 | `release_memory_occupation`（内部先 flush） |

**Code — async abort：**

```python
## 来源：server_control.py L66-L67
async def abort_servers_until_idle(urls: list[str]) -> None:
    await asyncio.gather(*(abort_server_until_idle(url) for url in urls))
```

---

## Q8：PD disaggregation 下 prefill worker 注册失败？

**Explain：** prefill 必须提供 `disaggregation_bootstrap_port`，否则 Router 无法 bootstrap room。

**Code：**

```python
## 来源：slime/backends/sglang_utils/sglang_engine.py L220-L226
                    if bootstrap_port is None:
                        raise RuntimeError(
                            f"Prefill worker {worker_url} does not have disaggregation_bootstrap_port; "
                            "cannot register it to the PD router."
                        )
```

---

## Q9：distributed vs tensor vs disk 怎么选？

| 路径 | 延迟 | 带宽 | 典型 |
|------|------|------|------|
| distributed | 低 | NCCL | 默认 on-policy |
| tensor | 最低（同机） | GPU IPC | colocate + 小模型 |
| disk | 高 | FS | 跨节点、超大模型 |
| delta | 中 | FS 增量 | `--update-weight-mode delta` |

Engine 侧 API 一一对应；训练侧在 `update_weight/*.py` 选择策略（[[24-WeightSync-Dist-00-MOC]]–[[25-WeightSync-Disk-00-MOC]]）。

---

## Q10：权重 version 不一致如何排查？

**Explain：** `get_weight_version` / `set_weight_version` 用于 delta 全零 diff 等边界；CI 可用 `--check-weight-update-equal`。

**Code：**

```python
## 来源：slime/backends/sglang_utils/sglang_engine.py L363-L378
    def get_weight_version(self):
        response = requests.get(f"http://{self.server_host}:{self.server_port}/get_weight_version")
        return response.json()["weight_version"]

    def set_weight_version(self, new_version: str):
        return self._make_request("update_weight_version", {"new_version": str(new_version)})
```
