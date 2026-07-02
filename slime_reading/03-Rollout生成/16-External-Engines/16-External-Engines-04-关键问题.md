---
type: batch-doc
module: 16-External-Engines
batch: "16"
doc_type: faq
title: "External Engines · 关键问题"
tags:
  - slime/batch/16
  - slime/module/external-engines
  - slime/doc/faq
updated: 2026-07-02
---

# External Engines · 关键问题

---

## Q1：什么时候用 external engine，什么时候用 `--sglang-config`？

| 场景 | 选择 |
|------|------|
| SGLang 已由 K8s/独立集群部署，训练 job 只连接 | **external** |
| 需要多模型（reference/reward 冻结、`update_weights: false`） | **sglang-config** |
| 训练与 serving 不同 GPU 型号/厂商 | **external + disk transport** |
| 单集群 on-policy，Slime 全权管理 engine | **sglang-config** 或默认 launch |
| PD 分离但 engine 仍由 Slime 启动 | **sglang-config** |
| PD 分离且 prefill/decode 已外部部署 | **external**（传 prefill+decode addrs） |

两者 **CLI 互斥**——不能同时传 `--rollout-external-engine-addrs` 与 `--sglang-config`。

---

## Q2：external 模式为何不支持 Slime fault tolerance？

**Explain：** `ExternalRolloutServer.recover()` 直接 warn 并 skip；引擎进程不在 Slime Ray PG 内，Slime 无法 `launch_server_process` 重启。

**Code：**

```python
# 来源：slime/backends/sglang_utils/external.py L152-L153
    def recover(self):
        logger.warning("Fault tolerance is not supported for external rollout engines; skip recover.")
```

**易错 vs 正确：**

```python
# ❌ 易错：external 模式仍开 --use-fault-tolerance  expecting auto-recover
# RolloutHealthMonitor 不会创建（server_groups 为空），recover 也是 no-op

# ✅ 正确：external 引擎生命周期交给外部编排（K8s restartPolicy / 独立 supervisor）
# Slime 侧重试在 http_utils.post 层（HTTP 60 次重试）
```

---

## Q3：`rollout_num_gpus` 在 external 模式下含义是什么？

- **内置模式：** Slime PG 预留的物理 GPU 数。
- **External 模式：** 从 `/server_info` 汇总的 **逻辑 GPU 容量**（`sum(info.num_gpus)`），用于 HTTP 连接池 sizing 与日志，**不占用训练 PG**。

若 server_info 未正确报告 `num_gpus`，会导致 `init_http_client` 连接池过小或权重 sync 分组错误——启动时注意日志 `Detected external SGLang engines:`。

---

## Q4：sanity check 失败怎么排查？

**Explain：** `_init_external` 对比 Slime 根据 CLI 计算的 `expect_server_args` 与外部 engine 实际 `/server_info`。

**常见 mismatch：**

| 字段 | 原因 |
|------|------|
| `tp_size` / `pp_size` | 外部 launch 参数与 Slime `--rollout-num-gpus-per-engine` 不一致 |
| `model_path` | 外部加载的 checkpoint 与 `--hf-checkpoint` 不同 |
| `dtype` | 量化/精度配置不一致 |

**Code（失败形态）：**

```python
# 来源：slime/backends/sglang_utils/sglang_engine.py L191-L193
                assert (
                    actual_value == expect_value
                ), f"{name=} {expect_value=} {actual_value=} ..."
```

**修复：** 对齐外部 `launch_server` 参数与 Slime 训练 CLI，或调整 Slime sglang 相关 override。

---

## Q5：disk 权重同步的路径要求？

**Explain：** 文档 Deployment Checklist 强调双向可见性。

**Code：**

```markdown
# 来源：docs/en/advanced/external-rollout-engines.md L98
- Disk transport requires trainer and SGLang engines to see the same `--update-weight-disk-dir` path
```

**易错 vs 正确：**

```bash
# ❌ 易错：trainer 写 /mnt/train-only/weights，engine 挂载不同 path
--update-weight-disk-dir /mnt/train-only/weights

# ✅ 正确：NFS/Lustre/S3-FUSE 等同一路径，trainer write + engine read
--update-weight-disk-dir /shared/fs/full-updates
--update-weight-local-checkpoint-dir /local/nvme/rollout-ckpt   # delta 模式 engine 本地
```

---

## Q6：delta 模式为何不能与 colocate 共用？

**Explain：** colocate 权重同步走 CUDA IPC / tensor 路径；delta 编码目的是 **减少跨网络传输字节**，与 colocate 的 IPC 语义冲突。文档明确禁止。

**Code：**

```markdown
# 来源：docs/en/advanced/external-rollout-engines.md L101
- Delta mode does not support `--colocate`
```

External + delta + disk 是跨 DC 大模型的典型组合（参考 Cursor Composer 2 技术报告描述的 S3 delta chain 模式）。

---

## Q7：PD external 部署要注意什么？

1. **分别 launch** prefill 与 decode server，各自 `--disaggregation-mode`。
2. prefill server 的 `/server_info` 须含 `disaggregation_bootstrap_port`。
3. `--rollout-external-engine-addrs` 传 **所有** worker 地址（prefill + decode）。
4. Slime 自动 `has_pd_disaggregation=True` 启动 PD Router。
5. 设置 `no_proxy` 包含 external host。

**Code：**

```python
# 来源：slime/backends/sglang_utils/external.py L53-L54
    if info.worker_type == "prefill":
        init_kwargs["disaggregation_bootstrap_port"] = info.disaggregation_bootstrap_port
```

---

## Q8：Health monitor 对外部 engine 完全无效吗？

**部分正确：**

- Slime **不会**为 `ExternalRolloutServer` 创建 `RolloutHealthMonitor`（无 `server_groups`）。
- 但 Slime 仍创建 `SGLangEngine` Ray actor；若 actor 自身异常，需 Ray 层监控。
- 外部 SGLang 进程健康由 **外部系统 + SGLang 自身** 负责；Router 的 health check 被 Slime 禁用（`disable_health_check=True`）。

---

## Q9：`debug_rollout_only + external` 的 PG 行为？

**Explain：** 纯 rollout 调试且 engine 外部时，PG 大小为 0——不占任何 GPU。

**Code：**

```python
# 来源：slime/ray/placement_group.py L106-L109
    if args.rollout_external:
        if args.debug_rollout_only:
            return 0, 0
        return actor_num_gpus, actor_num_gpus
```

适用于 CI 探测 external fleet 连通性而不启动训练 actor。

---

## Q10：与批次 15 SGLangEngine 的分工？

| 批次 15 | 批次 16 |
|---------|---------|
| `SGLangEngine` 全 API（launch、update_weights*） | external 专用路径 + HTTP 基础设施 |
| `launch_server_process` | **不 launch**，只对接 |
| 内置 fault tolerance | 明确不支持 recover |
| NCCL group 细节 | 部署选型 + disk/delta 路线图 |

读 external 模式时，批次 15 的 `_init_external` 与 `_register_to_router` 是必要前置。

---

## Q11：HTTP POST 失败为何重试 60 次？

**Explain：** rollout 高并发下 transient 5xx/连接 reset 常见；1s 间隔 × 60 ≈ 1 分钟窗口。若 external engine 长时间不可用，generate 会阻塞至超时——需外部监控告警。

**Code：**

```python
# 来源：slime/utils/http_utils.py L165-L166
async def _post(client, url, payload, max_retries=60, headers=None):
    retry_count = 0
```

---

## Q12：能否混用 external engine 与 Slime 自 launch engine？

**不能。** 一次训练 job 仅一种 rollout 模式：`rollout_external` 或 sglang-config 内置 topology。多模型 frozen reference 需求请用 `--sglang-config`。
