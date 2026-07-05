---
type: batch-doc
module: 25-WeightSync-Disk
batch: "25"
doc_type: faq
title: "磁盘权重同步 · 关键问题"
tags:
  - slime/batch/25
  - slime/module/weight-sync-disk
  - slime/doc/faq
updated: 2026-07-02
---

# 磁盘权重同步 · 关键问题

---

## Q1：full disk vs delta vs colocate tensor 怎么选？

| 场景 | 推荐 | 原因 |
|------|------|------|
| 跨机房、无 NCCL、每轮变动 <30% 参数 | `mode=delta` + disk | wire 小、无引擎 delta 支持需求 |
| 共享 FS 可靠、变动大或调试 | `transport=disk` full | 实现简单、与 HF 工具链兼容 |
| 训练推理同机、低延迟 | `--colocate` → tensor IPC | 无磁盘往返、无 NCCL 跨节点 |
| 同机 + 部分远端 engine | colocate tensor **混合** NCCL | 见 `UpdateWeightFromTensor.use_distribute` |
| 低延迟 LAN、无共享盘 | `transport=nccl` | 见 [[24-WeightSync-Dist-04-关键问题]] |

**Code：**

```python
## 来源：slime/backends/megatron_utils/actor.py L139-L161
        if self.args.colocate:
            update_weight_cls = UpdateWeightFromTensor
        elif self.args.update_weight_mode == "delta":
            update_weight_cls = UpdateWeightFromDiskDelta
        else:
            update_weight_cls = UpdateWeightFromDisk if self.args.update_weight_transport == "disk" else UpdateWeightFromDistributed
```

---

## Q2：delta 首轮 update_weights 为何「什么都不做」？

**Explain：** 第一次调用只 `_capture_baseline()`：从 `--hf-checkpoint` 读 CPU 快照并清空 stale `delta_dir`。第二次起才 diff publish。训练日志里第一轮 sync 后 engine 权重不变是 **预期行为**。

---

## Q3：为何 baseline 从 hf_checkpoint 而非 GPU 权重？

**Explain：** 各 host 的 `local_checkpoint_dir` 也从同一 `hf_checkpoint` materialize。若 baseline 来自 Megatron gather，而 engine base 来自 HF 文件，Megatron→HF 非 byte-exact（embed/lm_head padding trim）会导致 diff apply 后 checksum 失败。

**Code：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py L98-L103
        """Seeds from hf_checkpoint — what each host materializes its base from — so the invariant
        ``snapshot == engine base`` holds even where the megatron->HF round-trip trims vocab-padding rows."""
```

---

## Q4：xor 误 apply 两次会怎样？

**Explain：** xor 是 involution：`new ^ old ^ old = new` 的 **错误 base** 上 apply 两次会 revert。因此必须保证版本链顺序 + 正确 base_version；overwrite 编码幂等，适合需要 re-apply 的场景。

---

## Q5：delta 能否与 `--colocate` 同开？

**Explain：** **不能**。colocate 断言 `update_weight_mode == full` 且走 `UpdateWeightFromTensor`。delta 设计面向 **分离部署** + 共享 FS + 各 host 本地 NVMe。

**Code：**

```python
## 来源：slime/backends/megatron_utils/actor.py L139-L142
        if self.args.colocate:
            assert self.args.update_weight_mode == "full"
            update_weight_cls = UpdateWeightFromTensor
```

---

## Q6：`density` 和 `wire_bytes` 如何解读？

| 指标 | 含义 | 典型观察 |
|------|------|----------|
| `perf/update_weights_density` | 变更字节 / 总参数字节 | RL 后期常 5–30% |
| `perf/update_weights_wire_bytes` | zstd 压缩后 safetensors 总大小 | 应显著小于 full checkpoint |

**Explain：** 二者 all_reduce 跨 rank 后写入 `update_weight_metrics`；rank 0 打 `[disk delta v=...]` 日志。

---

## Q7：共享 FS 上 rank 间写文件如何协调？

**Explain：** `_write_delta_files` 用 `dist.all_gather_object` 分配 `model-NNNNN-of-MMMMM.safetensors` 编号，**不依赖** FS 跨 rank 立即可见 listing。index.json 仅 rank 0 写。对象存储需 `custom_delta_pre_push/pre_read` hook。

---

## Q8：full disk 同步慢怎么优化？

| 手段 | 说明 |
|------|------|
| 换 delta | 变动稀疏时 wire 大幅下降 |
| NVMe 本地盘 + 高带宽 FS | 减少 save_hf 阻塞 |
| `--update-weight-disk-keep-files=false` | 省 inode/空间（不影响 reload） |
| 与 NCCL 对比 profiling | 小模型可能 NCCL 更快 |

---

## Q9：IPC colocate 内存泄漏排查

**Explain：** 每 HF chunk 后必须 `del long_lived_tensors` + `torch.cuda.ipc_collect()`；否则 IPC handle 在 consumer 关闭前占用显存。barrier 后再 collect 一次清理末 chunk handle。

**Code：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py L174-L180
            del long_lived_tensors, hf_named_tensors
            torch.cuda.ipc_collect()
        dist.barrier(group=get_gloo_group())
        torch.cuda.ipc_collect()
```

---

## Q10：checksum mismatch 如何排障？

| 可能原因 | 检查 |
|----------|------|
| 版本乱序 | `.delta_sync/state.json` version vs index `base_version` |
| 错误 encoding | index metadata `delta_encoding` 与 trainer 一致 |
| 本地 base 未 materialize | `init_local_checkpoint` 是否完成 |
| 共享 FS 读到 stale delta | 启用 `custom_delta_pre_read_path` |

**Explain：** apply 失败 **raise RuntimeError**，引擎不会 silent 加载坏权重。

---

## Q11：量化模型（compressed-tensors）注意点

**Explain：** colocate tensor 路径在 load 前后调 `post_process_weights`（与 NCCL 相同）。disk full/delta 走 HF safetensors，量化格式由 `save_hf_model_to_path` / engine reload 处理。

---

## Q12：验证命令

```bash
cd slime && pytest tests/test_full_disk_weight_update.py -q
# delta 示例见 examples/delta_weight_sync/run-*.sh
```

---

## Q13：与 SGLang delta load_format 的区别

**Explain：** Slime **trainer-side** delta（本专题）在 host apply 成 full HF 再 reload。SGLang 原生 `load_format=delta`（external PD 测试）是 **引擎侧** 读 delta 文件——路径不同，不要混淆 `UpdateWeightFromDiskDelta` 与 engine payload `load_format=delta`。
