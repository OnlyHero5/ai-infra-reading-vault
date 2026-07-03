---
type: batch-doc
module: 25-WeightSync-Disk
batch: "25"
doc_type: concept
title: "磁盘权重同步 · 核心概念"
tags:
  - slime/batch/25
  - slime/module/weight-sync-disk
  - slime/doc/concept
updated: 2026-07-02
---

# 磁盘权重同步 · 核心概念

## 用户故事：跨机房训练，NCCL 不通怎么办

### Persona

**小李**，RL 基础设施工程师。训练集群在 A 机房、SGLang 推理在 B 机房，防火墙禁止训练 rank 与推理 GPU 建 NCCL 组。他需要把 Megatron 每轮更新后的权重送到推理侧，且大模型全量 HF 落盘太慢。

### 时间线

| 时刻 | 事件 |
|------|------|
| T0 | Actor init 按 `colocate` / `update_weight_mode` / `transport` 选型 updater |
| T1 | `update_weights()`：pause → 写盘或 diff → engine reload → continue |
| T2（delta） | 首轮仅 capture baseline；次轮起 publish delta + 各 host apply |
| T3（colocate） | `UpdateWeightFromTensor`：CUDA IPC + 可选 NCCL 混合 |

---

## 1. 四种路径与三轴选型

**Explain：** 与 [[24-WeightSync-Dist-01-核心概念]] 共享同一 `actor.py` 分支；本专题覆盖 **disk full**、**disk delta**、**colocate tensor** 三条。

**Code：**

```python
## 来源：slime/backends/megatron_utils/actor.py L139-L161
        if self.args.colocate:
            assert self.args.update_weight_mode == "full"
            update_weight_cls = UpdateWeightFromTensor
        elif self.args.update_weight_mode == "delta":
            assert self.args.update_weight_transport == "disk"
            from .update_weight.update_weight_from_disk_delta import UpdateWeightFromDiskDelta
            update_weight_cls = UpdateWeightFromDiskDelta
        else:
            assert self.args.update_weight_mode == "full"
            if self.args.update_weight_transport == "disk":
                update_weight_cls = UpdateWeightFromDisk
            else:
                update_weight_cls = UpdateWeightFromDistributed
```

**Comment：**

| 类 | 场景 | 传输介质 |
|----|------|----------|
| `UpdateWeightFromDisk` | 分离 + full + disk | 共享 FS 写完整 HF 目录 |
| `UpdateWeightFromDiskDelta` | 分离 + delta + disk | 共享 FS 写压缩 diff |
| `UpdateWeightFromTensor` | colocate + full | CUDA IPC（+ 可选 NCCL 远端） |
| `UpdateWeightFromDistributed` | 分离 + full + nccl | 见[[24-WeightSync-Dist-00-MOC]] |

---

## 2. Full Disk：版本目录 + 引擎 reload

**Explain：** 每轮 `weight_version++`，rank 0 在 `update_weight_disk_dir/weight_v{NNNNNN}/` 写完整 HF checkpoint（复用 `save_hf_model_to_path`），引擎 HTTP `update_weights_from_disk` 热加载。可选 `--update-weight-disk-keep-files=false` 同步后删目录省空间。

**Code：**

```python
## 来源：update_weight/update_weight_from_disk.py L61-L98
    @torch.no_grad()
    def update_weights(self) -> None:
        self.weight_version += 1
        version_dir = Path(self.args.update_weight_disk_dir) / f"weight_v{self.weight_version:06d}"
        if dist.get_rank() == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
        save_hf_model_to_path(self.args, version_dir, self.model, ...)
        if dist.get_rank() == 0:
            refs = [
                engine.update_weights_from_disk.remote(
                    model_path=str(version_dir),
                    weight_version=str(self.weight_version),
                )
                for engine in self.rollout_engines
            ]
            ray.get(refs)
            if not self.args.update_weight_disk_keep_files:
                shutil.rmtree(version_dir, ignore_errors=True)
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
```

**Comment：** 与 NCCL 路径不同，**所有 PP rank** 都参与 `save_hf_model_to_path` collective；仅 rank 0 发 Ray RPC。

---

## 3. Delta Disk：快照 → diff → apply → vanilla reload

**Explain：** 继承 `UpdateWeightFromDistributed` 的 HF gather 逻辑，但 **不建 NCCL**。首轮 `_capture_baseline` 从 `--hf-checkpoint` 读 CPU 快照；之后每轮 diff、zstd 压缩、写 canonical safetensors + index.json。各 host 用 `disk_delta.apply_deltas` 就地 patch 本地 checkpoint，再走普通 `update_weights_from_disk`。

**Code：**

```python
## 来源：update_weight/update_weight_from_disk_delta.py L80-L96
    @torch.no_grad()
    def update_weights(self) -> None:
        if not self._baseline_captured:
            self._capture_baseline()
            self._baseline_captured = True
            return
        self.weight_version += 1
        if dist.get_rank() == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
        self._publish()
        self._reload_engines()
        self._record_metrics()
```

**Comment：** SGLang **无需** delta 格式支持——引擎只看到 patch 后的完整 HF 目录。

---

## 4. Delta 编码：xor vs overwrite

**Explain：** 二者均为 **字节级**、dtype 无关，量化 checkpoint 同路径可用。

| 编码 | 线上内容 | 应用特性 | 适用 |
|------|----------|----------|------|
| `xor`（默认） | `new ^ old` | 对正确 base **恰应用一次** | 线小、apply 快 |
| `overwrite` | 变更位置 + 新值 | **幂等**，可重复 apply | 需容错/部分 apply |

**Code：**

```python
## 来源：disk_delta.py L29-L33
def overwrite_encode(new: np.ndarray, changed_mask: np.ndarray) -> np.ndarray:
    pos = np.flatnonzero(changed_mask).astype("<u4")
    return np.concatenate([np.array([pos.size], "<u4").view(np.uint8), pos.view(np.uint8), new[changed_mask]])
```

```python
## 来源：update_weight_from_disk_delta.py L227-L235
            if self.delta_encoding == "xor":
                diff = new ^ old
                changed = int(np.count_nonzero(diff))
            elif self.delta_encoding == "overwrite":
                mask = new != old
                changed = int(np.count_nonzero(mask))
                diff = overwrite_encode(new, mask)
```

---

## 5. 本地 checkpoint 与版本链

**Explain：** 每个 rollout **host** 在 `update_weight_local_checkpoint_dir` 维护一份 HF 副本（启动时 `init_local_checkpoint` 从 `--hf-checkpoint` 拷贝）。`.delta_sync/state.json` 记录已 apply 版本；`apply_deltas` 严格顺序递增，checksum 不匹配则 **fail loud**。

**Code：**

```python
## 来源：disk_delta.py L111-L124
def init_local_checkpoint(local_ckpt_dir: str, base_dir: str) -> None:
    with _apply_lock(local_ckpt_dir):
        if _read_applied_version(local_ckpt_dir) is not None:
            return
        for entry in os.scandir(base_dir):
            if entry.is_file():
                shutil.copy2(entry.path, os.path.join(local_ckpt_dir, entry.name))
        _write_applied_version(local_ckpt_dir, "000000")
```

**Comment：** baseline 从 `hf_checkpoint` 而非当前 GPU 权重 seed，保证 Megatron→HF 非 byte-exact（如 vocab padding trim）时仍正确。

---

## 6. Colocate：UpdateWeightFromTensor + IPC

**Explain：** 训练与 SGLang 同机时，HF 张量经 `FlattenedTensorBucket` + `MultiprocessingSerializer` 序列化，`dist.gather_object` 到 engine 对应 GPU 组 source rank，Ray IPC 调 `update_weights_from_tensor`。若还有 **远端** 引擎，colocate 部分走 IPC、远端部分仍 NCCL（混合模式）。

**Code：**

```python
## 来源：update_weight/update_weight_from_tensor.py L147-L175
    @torch.no_grad()
    def update_weights(self) -> None:
        self.weight_version += 1
        megatron_local_weights = self.weights_getter()
        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
            ray.get(refs)
            del long_lived_tensors, hf_named_tensors
            torch.cuda.ipc_collect()
```

**Comment：** 每 chunk 后 `ipc_collect()` 释放 CUDA IPC 句柄，避免泄漏。

---

## 7. 关键 CLI

| 参数 | 作用 |
|------|------|
| `--update-weight-transport disk` | 启用 full disk 或 delta（delta 强制 disk） |
| `--update-weight-mode delta` | 增量同步（仅 disk） |
| `--update-weight-disk-dir` | 共享 FS：full 版本目录或 delta 发布根 |
| `--update-weight-local-checkpoint-dir` | 各 host 本地 NVMe checkpoint（delta apply 目标） |
| `--update-weight-delta-encoding xor\|overwrite` | diff 编码 |
| `--update-weight-delta-checksum xxh3-128\|blake3\|adler32` | 每 tensor 完整性 |
| `--update-weight-disk-keep-files` | full disk 同步后是否保留版本目录 |
| `--custom-delta-pre-push-path` / `--custom-delta-pre-read-path` | 对象存储 commit/refresh hook |

---

## 8. 与 Checkpoint 专题的关系

**Explain：** `save_hf_model_to_path`（full disk）与 [[26-Checkpoint-M2HF-00-MOC]] 共用 HF 转换栈；delta 的 gather 复用 distributed 路径的 `_iter_hf_tensors` / TP-EP all-gather。

---

## 概念速查

| 术语 | 含义 |
|------|------|
| `weight_version` | 每次成功 sync 递增；引擎 HTTP payload 携带 |
| `density` metric | `changed_bytes / total_bytes`；W&B `perf/update_weights_density` |
| `wire_bytes` | 压缩后 safetensors 总字节 |
| `all_engine_actors` | delta 路径：每 **host** 一个 actor，负责 `sync_local_checkpoint` |
| `SYNC_DIR` (`.delta_sync`) | 本地 checkpoint 内 apply 锁 + state.json |
