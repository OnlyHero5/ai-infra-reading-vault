---
type: batch-doc
module: 24-WeightSync-Dist
batch: "24"
doc_type: faq
title: "NCCL 权重同步 · 关键问题"
tags:
  - slime/batch/24
  - slime/module/weight-sync-dist
  - slime/doc/faq
updated: 2026-07-02
---

# NCCL 权重同步 · 关键问题

## Q1：何时选 NCCL，何时选 disk / colocate？

| 场景 | 推荐 | 原因 |
|------|------|------|
| 训练与推理 **分离**，节点间 NVLink/IB 良好 | `transport=nccl` | 低延迟、无共享盘依赖 |
| 跨机房 / 无 RDMA / 防火墙限制 NCCL | `transport=disk` | 共享 FS + 引擎 reload |
| 训练推理 **同进程 colocate** | 自动 `UpdateWeightFromTensor` | CUDA IPC，零拷贝 |
| 超大模型增量同步 | `mode=delta` + disk | NCCL 路径不支持 delta |

**Explain：** NCCL 路径要求 `--update-weight-mode=full` 且 `--update-weight-transport=nccl`，且 **不能** `--colocate`。

**Code：**

```python
## 来源：slime/backends/megatron_utils/actor.py L139-L161
        if self.args.colocate:
            update_weight_cls = UpdateWeightFromTensor
        elif self.args.update_weight_mode == "delta":
            ...
        else:
            if self.args.update_weight_transport == "disk":
                update_weight_cls = UpdateWeightFromDisk
            else:
                update_weight_cls = UpdateWeightFromDistributed
```

---

## Q2：为什么只有 PP source rank 发起 broadcast？

**Explain：** 每个 PP stage 只存储部分 decoder layers。该 stage 内 DP=0、TP=0 的 rank 持有「可代表本 stage 向引擎推送」的权限；其他 rank 仅参与 TP/EP collective 拼完整张量。若多 rank 同时 broadcast 同一 group，引擎 recv 顺序不可定义。

**排查：** 确认 `mpu.get_data_parallel_rank(with_context_parallel=True)==0` 且 `get_tensor_model_parallel_rank()==0` 的 rank 上 tqdm 进度条可见。

---

## Q3：`update_weight_buffer_size` 设太大/太小会怎样？

| 设置 | 影响 |
|------|------|
| **过小** | bucket 数增多，Ray lock acquire/release 与 NCCL launch 次数上升，同步变慢 |
| **过大** | 单次 broadcast 峰值 GPU 内存升高，可能 OOM |
| **与 expert 批量** | expert chunk 用 `param_size × EP_size` 估算，MoE 模型需留更大余量 |

**Explain：** 非 expert 与 expert 分两趟；expert 路径在 `_iter_expert_chunks` 用 EP world size 放大阈值判断。

**Code：**

```python
## 来源：update_weight/update_weight_from_distributed.py L189-L191
            if (
                buffer_size + param_size
            ) * mpu.get_expert_model_parallel_world_size() > self.args.update_weight_buffer_size:
```

---

## Q4：NCCL 同步 hang / deadlock 常见原因

| 现象 | 可能原因 | 排查 |
|------|----------|------|
| broadcast hang | metadata RPC 未到达即 broadcast | 查 engine 日志、HTTP 超时 |
| 多 PP 死锁 | 未持 lock 并发 broadcast | 确认 `rollout_engine_lock` 正常 |
| group 不一致 | 引擎重启未 reconnect | 看 `num_new_engines > 0` 是否触发 connect |
| rank 不匹配 | `engine_gpu_counts` 与真实 TP 不符 | PD 分离异构 TP 配置 |

**Explain：** `_update_bucket_weights_from_distributed` 注释写明 lock 用于防止 NCCL deadlock。

---

## Q5：MoE 模型 expert 权重为何单独一趟？

**Explain：** Expert 参数按 EP 分片存储，需额外 EP all_gather 才能凑齐全部 experts 再 convert_to_hf。与非 expert 分开可避免：(1) 单 bucket 混入 EP 放大后的巨型 tensor；(2) 非 expert 迭代被 MoE batch 逻辑拖慢。

**Code：**

```python
## 来源：update_weight/update_weight_from_distributed.py L142-L146
        for chunk_iter in (self._iter_non_expert_chunks(), self._iter_expert_chunks()):
            for hf_chunk in chunk_iter:
                ...
            dist.barrier(group=get_gloo_group())
```

---

## Q6：`HfWeightIteratorDirect` 和 NCCL 路径是什么关系？

**Explain：** **不是** `UpdateWeightFromDistributed` 的直接依赖。两者共享 `common.py` 的 gather/命名逻辑与 `convert_to_hf`。Direct 迭代器在 `megatron_to_hf_mode=raw` 时由 `HfWeightIteratorBase.create` 选用，主要服务 `save_hf_model_to_path` 等同构分桶导出。读 Direct 是为理解 buffer 划分与 async gather，而非 NCCL 必经路径。

**Code：**

```python
## 来源：slime/backends/megatron_utils/hf_checkpoint_saver.py（调用点）
    hf_weight_iterator = HfWeightIteratorDirect(
        args, model, model_name=model_name, quantization_config=quantization_config
    )
```

---

## Q7：compressed-tensors 量化模型的额外步骤

**Explain：** int4/fp4 等 compressed-tensors 在 load 前后需引擎侧 restore / post_process，否则 quant param layout 与 Megatron 直传 tensor 不一致。

**顺序：** pause → **pre post_process** → barrier → send weights → **post post_process** → continue

---

## Q8：`weight_version` 不一致说明什么？

**Explain：** `--ci-test` 下 actor 随机抽引擎比对 version。不一致意味着某次 broadcast 未完成、引擎未执行 recv、或 connect 后未同步即开始 rollout。

**Code：**

```python
## 来源：slime/backends/megatron_utils/actor.py L630-L636
            if self.args.ci_test and len(rollout_engines) > 0 and self.weight_updater.weight_version > 0:
                engine = random.choice(rollout_engines)
                engine_version = ray.get(engine.get_weight_version.remote())
                if str(engine_version) != str(self.weight_updater.weight_version):
                    raise RuntimeError(
                        f"Weight version mismatch! Engine: {engine_version}, Updater: {self.weight_updater.weight_version}"
                    )
```

---

## Q9：offload_train + critic 为何特殊处理 reconnect？

**Explain：** critic 训练完成后 actor sleep 会 destroy process groups；下次 update_weights 前必须 wake_up 并 reconnect NCCL，否则 `_model_update_groups` 无效。这与纯 actor offload（仅 reload_process_groups）不同。

---

## Q10：与 SGLang ModelLoader 的衔接

**Explain：** 引擎 recv 到的 HF 命名 tensor 由 SGLang `update_weights_from_distributed` API 写入 runtime model。权重 layout / quantization 兼容性取决于 SGLang 版本与 slime docker patch。交叉阅读 [[12-ModelLoader-01-核心概念]]、[[32-CheckpointEngine-01-核心概念]]。

---

## 易错点速查

1. **忘记 pause/flush**：推理中更新权重可能读到 half-updated 参数 → updater 在 rank 0 强制 pause + flush_cache
2. **PP 多 stage 只连一个 group**：每个 PP rank 有独立 `slime-pp_{i}`，引擎须支持多 group 或按部署只连 subset
3. **GLU linear_fc1**：all_gather 后需 rechunk，否则 HF convert shape 错误
4. **expert_bias**：作为普通 param 同步，不参与 TP gather
