---
type: batch-doc
module: 25-WeightSync-Disk
batch: "25"
doc_type: walkthrough
title: "磁盘权重同步 · 源码走读"
tags:
  - slime/batch/25
  - slime/module/weight-sync-disk
  - slime/doc/walkthrough
updated: 2026-07-02
---

# 磁盘权重同步 · 源码走读

> 走读顺序：`UpdateWeightFromDisk` → `UpdateWeightFromDiskDelta` → `disk_delta` → `UpdateWeightFromTensor`  
> 基线 commit `22cdc6e1` · **本专题内嵌代码热点 ≥400 行**

---

## 1. UpdateWeightFromDisk — 类骨架

**Explain：** 无 NCCL 状态；`connect_rollout_engines` 仅缓存 engine handles。metrics 通过 `pop_metrics()` 与 distributed 路径对称。

**Code：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_disk.py L21-L59
class UpdateWeightFromDisk:
    """Full-weight sync through a shared filesystem and SGLang disk reload."""

    def __init__(self, args, model, weights_getter, *, model_name, quantization_config):
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self.update_weight_metrics: dict[str, float] = {}
        self.rollout_engines: Sequence[ActorHandle] = []
        self.rollout_engine_lock: ActorHandle | None = None

    def connect_rollout_engines(self, rollout_engines, rollout_engine_lock, ...):
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock

    def disconnect_rollout_engines(self) -> None:
        return

    def pop_metrics(self) -> dict[str, float]:
        out, self.update_weight_metrics = self.update_weight_metrics, {}
        return out
```

---

## 2. UpdateWeightFromDisk — update_weights 全流程

**Explain：** Gloo barrier 分隔 pause / save / reload / continue 四阶段；`save_hf_model_to_path` 为 collective（所有 rank 参与 Megatron→HF）。

**Code：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_disk.py L61-L98
    @torch.no_grad()
    def update_weights(self) -> None:
        self.weight_version += 1
        version_dir = Path(self.args.update_weight_disk_dir) / f"weight_v{self.weight_version:06d}"

        if dist.get_rank() == 0:
            shutil.rmtree(version_dir, ignore_errors=True)
        dist.barrier(group=get_gloo_group())

        if dist.get_rank() == 0:
            logger.info("Updating rollout weights from disk checkpoint %s", version_dir)
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

        save_hf_model_to_path(
            self.args,
            version_dir,
            self.model,
            model_name=self.model_name,
            quantization_config=self.quantization_config,
            progress_desc="Save HF  weights for update from disk",
        )
        dist.barrier(group=get_gloo_group())

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
        dist.barrier(group=get_gloo_group())
```

**Comment：** rank 0 独占 Ray 控制面；非 0 rank 在 barrier 处等待，避免 save 未完成即 reload。

---

## 3. UpdateWeightFromDiskDelta — 继承与 init

**Explain：** 继承 `UpdateWeightFromDistributed` 复用 `_iter_hf_tensors`、TP/EP gather；override connect/update 去掉 NCCL 组。

**Code：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py L30-L58
class UpdateWeightFromDiskDelta(UpdateWeightFromDistributed):
    """
    Delta weight sync over a shared filesystem. PP-src ranks diff each gathered HF tensor against
    a CPU snapshot of the previous sync and publish the changes as a canonical HF checkpoint dir;
    every rollout host applies the delta into its local checkpoint and reloads via the ordinary
    update_weights_from_disk path, so sglang needs no delta support.
    """

    def __init__(self, args, model, weights_getter, *, model_name, quantization_config):
        super().__init__(args, model, weights_getter, model_name=model_name, quantization_config=quantization_config)
        self.delta_dir = args.update_weight_disk_dir
        os.makedirs(self.delta_dir, exist_ok=True)
        self.delta_encoding = args.update_weight_delta_encoding
        self.checksum_algorithm = args.update_weight_delta_checksum
        self._snapshot: dict[str, np.ndarray] = {}
        self._baseline_captured = False
        self._commit_hook: Callable | None = None
        if args.custom_delta_pre_push_path:
            from slime.utils.misc import load_function
            self._commit_hook = load_function(args.custom_delta_pre_push_path)
```

---

## 4. connect_rollout_engines — all_engine_actors

**Explain：** delta apply 是 **host-local**；`all_engine_actors` 每物理机一个 Ray actor（colocate 多 engine 共享同一本地 checkpoint）。

**Code：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py L60-L75
    def connect_rollout_engines(
        self,
        rollout_engines,
        rollout_engine_lock,
        engine_gpu_counts=None,
        engine_gpu_offsets=None,
        all_engine_actors=None,
    ) -> None:
        self.rollout_engines = rollout_engines
        self.all_engine_actors = list(all_engine_actors or rollout_engines)
        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
        )

    def disconnect_rollout_engines(self) -> None:
        pass  # no NCCL groups to tear down
```

---

## 5. _capture_baseline — 首轮无 publish

**Explain：** 清空 `delta_dir`；对每个 HF tensor 名从 `hf_checkpoint` 读 uint8 flat；缺失时 fallback 当前 gathered 值并 warn。

**Code：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py L98-L124
    def _capture_baseline(self) -> None:
        if dist.get_rank() == 0:
            shutil.rmtree(self.delta_dir, ignore_errors=True)
            os.makedirs(self.delta_dir, exist_ok=True)
            if self._commit_hook is not None:
                self._commit_hook(self.args, self.delta_dir, list(self.rollout_engines))
        dist.barrier(group=get_gloo_group())

        read_hf = make_tensor_reader(self.args.hf_checkpoint)
        for name, tensor in self._iter_hf_tensors():
            try:
                self._snapshot[name] = read_hf(name)
            except KeyError:
                self._snapshot[name] = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().reshape(-1)
                logger.warning("seed: %s absent from hf_checkpoint; seeding from current weights", name)
        if dist.get_rank() == 0:
            logger.info(
                "[disk delta] captured baseline snapshot of %d tensors from %s",
                len(self._snapshot),
                self.args.hf_checkpoint,
            )
```

---

## 6. _encode_delta — 流水线 diff + 压缩

**Explain：** PP-src rank 建版本目录；pinned buffer pool 加速 GPU→CPU；ThreadPoolExecutor 并行 xor/overwrite + zstd level 1。

**Code：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py L195-L239
    def _encode_delta(self) -> None:
        self._version_dir = os.path.join(self.delta_dir, f"weight_v{self.weight_version:06d}")
        if self._is_pp_src_rank:
            os.makedirs(self._version_dir, exist_ok=True)
        snapshot = self._snapshot
        self._delta: dict[str, np.ndarray] = {}
        self._checksums: dict[str, str] = {}
        self.changed_bytes = self.total_bytes = 0

        def diff_and_compress(name, buf, nbytes, pinned):
            if pinned:
                new = np.empty(nbytes, dtype=np.uint8)
                np.copyto(new, buf.numpy()[:nbytes])
                free_q.put(buf)
            else:
                new = buf
            old = snapshot[name]
            if self.delta_encoding == "xor":
                diff = new ^ old
                changed = int(np.count_nonzero(diff))
            elif self.delta_encoding == "overwrite":
                mask = new != old
                changed = int(np.count_nonzero(mask))
                diff = overwrite_encode(new, mask)
            else:
                raise ValueError(f"unknown delta encoding {self.delta_encoding!r}")
            if not changed:
                return name, new, None, None, 0
            compressed = np.frombuffer(zstandard.ZstdCompressor(level=1).compress(diff), dtype=np.uint8)
            return name, new, compressed, checksum(self.checksum_algorithm, new), changed
```

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py L241-L269
        def collect(fut):
            name, new, compressed, digest, changed = fut.result()
            snapshot[name] = new
            if changed:
                self.changed_bytes += changed
                self._delta[name] = compressed
                self._checksums[name] = digest

        pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)
        inflight: deque = deque()
        try:
            for name, tensor in self._iter_hf_tensors():
                flat = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
                nbytes = int(flat.numel())
                ...
                inflight.append(pool.submit(diff_and_compress, name, payload, nbytes, pinned))
                if len(inflight) >= 2 * NUM_WORKERS:
                    collect(inflight.popleft())
            while inflight:
                collect(inflight.popleft())
        finally:
            pool.shutdown()
```

---

## 7. _write_delta_files — 跨 rank 协调 index

**Explain：** `all_gather_object` 统计哪些 rank 有 delta shard；rank 0 写 `model.safetensors.index.json` 含 version/base_version/encoding/checksum metadata。

**Code：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py L132-L167
    def _write_delta_files(self) -> None:
        group = get_gloo_group()
        world, rank = dist.get_world_size(), dist.get_rank()

        counts: list = [None] * world
        dist.all_gather_object(counts, int(bool(self._delta)), group=group)
        offset, total = sum(counts[:rank]), sum(counts)

        fname = None
        self.wire_bytes = 0
        if self._delta:
            fname = f"model-{offset:05d}-of-{total:05d}.safetensors"
            blob = safetensors.numpy.save(self._delta, metadata=self._checksums)
            self.wire_bytes = len(blob)
            _atomic_write(os.path.join(self._version_dir, fname), blob)

        maps: list = [None] * world
        dist.all_gather_object(maps, {name: fname for name in self._delta}, group=group)
        if rank == 0:
            index = {
                "metadata": {
                    "version": f"{self.weight_version:06d}",
                    "base_version": f"{self.weight_version - 1:06d}",
                    "delta_encoding": self.delta_encoding,
                    "compression_format": "zstd",
                    "checksum_format": self.checksum_algorithm,
                },
                "weight_map": {name: f for m in maps for name, f in m.items()},
            }
            _atomic_write(os.path.join(self._version_dir, "model.safetensors.index.json"), json.dumps(index).encode())
        dist.barrier(group=group)
```

---

## 8. _reload_engines — sync_local_checkpoint + disk reload

**Code：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py L169-L186
    def _reload_engines(self) -> None:
        if self._commit_hook is not None:
            self._commit_hook(self.args, self._version_dir, list(self.rollout_engines))
        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            ray.get([actor.sync_local_checkpoint.remote(self.weight_version) for actor in self.all_engine_actors])
            ray.get(
                [
                    engine.update_weights_from_disk.remote(
                        model_path=self.args.update_weight_local_checkpoint_dir,
                        weight_version=str(self.weight_version),
                    )
                    for engine in self.rollout_engines
                ]
            )
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())
```

**Comment：** `model_path` 指向 **本地** patch 后目录，非共享 delta 目录。

---

## 9. _record_metrics — density / wire

**Code：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py L271-L290
    def _record_metrics(self) -> None:
        counts = torch.tensor(
            [self.changed_bytes, self.total_bytes, self.wire_bytes],
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        dist.all_reduce(counts)
        changed, total, wire = counts.tolist()
        m = self.update_weight_metrics
        m["perf/update_weights_density"] = changed / max(total, 1)
        m["perf/update_weights_wire_bytes"] = wire
        if dist.get_rank() == 0:
            logger.info(
                "[disk delta v=%s] density=%.2f%% wire=%.2f GB",
                self.weight_version,
                100.0 * changed / max(total, 1),
                wire / 1e9,
            )
```

---

## 10. _atomic_write

**Code：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py L293-L299
def _atomic_write(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
```

---

## 11. disk_delta — make_tensor_reader

**Explain：** 启动时扫描所有 safetensors header，建立 name→(file, offset, nbytes) 索引；baseline seed 与 apply 校验共用。

**Code：**

```python
## 来源：slime/utils/disk_delta.py L126-L152
def _tensor_locations(ckpt_dir: str) -> dict[str, tuple[str, int, int]]:
    locations: dict[str, tuple[str, int, int]] = {}
    for path in glob.glob(os.path.join(ckpt_dir, "*.safetensors")):
        with open(path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            begin, end = info["data_offsets"]
            locations[name] = (path, 8 + header_len + begin, end - begin)
    return locations

def make_tensor_reader(ckpt_dir: str):
    locations = _tensor_locations(ckpt_dir)

    def read(name: str) -> np.ndarray:
        path, offset, nbytes = locations[name]
        with open(path, "rb") as f:
            f.seek(offset)
            return np.frombuffer(f.read(nbytes), dtype=np.uint8)

    return read
```

---

## 12. disk_delta — apply_xor 分块

**Explain：** zstd stream decompress + 2MB 块 XOR 进 mmap region；边 apply 边增量 checksum。

**Code：**

```python
## 来源：slime/utils/disk_delta.py L204-L220
        def apply_xor(item) -> None:
            name, compressed, path, offset, nbytes, want = item
            region = np.ndarray((nbytes,), dtype=np.uint8, buffer=open_mmaps[path][1], offset=offset)
            hasher = _new_hasher(algorithm)
            reader = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(bytes(compressed)))
            pos = 0
            while pos < nbytes:
                block = reader.read(min(2 << 20, nbytes - pos))
                if not block:
                    break
                chunk = np.frombuffer(block, dtype=np.uint8)
                region[pos : pos + chunk.size] ^= chunk
                hasher.update(region[pos : pos + chunk.size])
                pos += chunk.size
            if hasher.hexdigest() != want:
                with lock:
                    mismatches.append(name)
```

---

## 13. disk_delta — apply_overwrite

**Code：**

```python
## 来源：slime/utils/disk_delta.py L222-L231
        def apply_overwrite(item) -> None:
            name, compressed, path, offset, nbytes, want = item
            delta = np.frombuffer(zstandard.ZstdDecompressor().decompress(bytes(compressed)), dtype=np.uint8)
            region = np.ndarray((nbytes,), dtype=np.uint8, buffer=open_mmaps[path][1], offset=offset)
            count = int.from_bytes(delta[:4].tobytes(), "little")
            positions = np.frombuffer(delta[4 : 4 + 4 * count].tobytes(), dtype="<u4")
            region[positions] = delta[4 + 4 * count :]
            if checksum(algorithm, region) != want:
                with lock:
                    mismatches.append(name)
```

---

## 14. disk_delta — apply_deltas 版本链

**Code：**

```python
## 来源：slime/utils/disk_delta.py L155-L164, L255-L264
    applied = _read_applied_version(local_ckpt_dir)
    if applied == meta["version"]:
        return
    if applied != meta["base_version"]:
        raise RuntimeError(f"out-of-order delta: local at {applied}, delta builds on {meta['base_version']}")

def apply_deltas(local_ckpt_dir: str, delta_root: str, target_version: int) -> None:
    with _apply_lock(local_ckpt_dir):
        applied = _read_applied_version(local_ckpt_dir)
        if applied is None:
            raise RuntimeError("local checkpoint not materialized")
        for version in range(int(applied) + 1, target_version + 1):
            _apply_version(local_ckpt_dir, os.path.join(delta_root, f"weight_v{version:06d}"))
```

---

## 15. SGLangEngine — sync_local_checkpoint

**Code：**

```python
## 来源：slime/backends/sglang_utils/sglang_engine.py L396-L413
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

**Comment：** 引擎启动时 daemon thread 预热 `init_local_checkpoint`，与首次 delta reload 竞态由 flock 串行化。

---

## 16. SGLangEngine — update_weights_from_disk HTTP

**Code：**

```python
## 来源：slime/backends/sglang_utils/sglang_engine.py L415-L437
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

---

## 17. UpdateWeightFromTensor — connect 混合 colocate + 远端

**Explain：** 按 `engine_gpu_offsets` 判断哪些 engine GPU 落在 actor 节点范围内；超出部分走 NCCL distributed 子集。

**Code：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py L86-L117
        total_actor_gpus = self.args.actor_num_nodes * self.args.actor_num_gpus_per_node
        colocate_engine_nums = 0
        for gpu_offset, gpu_count in zip(engine_gpu_offsets, engine_gpu_counts, strict=True):
            if gpu_offset + gpu_count > total_actor_gpus:
                break
            colocate_engine_nums += 1

        self.use_distribute = len(rollout_engines) > colocate_engine_nums

        if self.use_distribute:
            self.rollout_engines = rollout_engines[:colocate_engine_nums]
            self.distributed_rollout_engines = rollout_engines[colocate_engine_nums:]
            ...
            self._model_update_groups = connect_rollout_engines_from_distributed(
                self.args,
                self._group_name,
                self.distributed_rollout_engines,
                engine_gpu_counts=distributed_gpu_counts,
            )
```

---

## 18. IPC Gloo gather group

**Code：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py L124-L137
        if self._ipc_gather_group is None:
            for i in range(colocate_engine_nums):
                group_ranks = list(range(colocate_gpu_offsets[i], colocate_gpu_offsets[i] + colocate_gpu_counts[i]))
                new_group = dist.new_group(ranks=group_ranks, backend="gloo")
                if dist.get_rank() in group_ranks:
                    self._ipc_gather_group = new_group
                    self._ipc_gather_src = colocate_gpu_offsets[i]

        for i, engine in enumerate(self.rollout_engines):
            start = colocate_gpu_offsets[i]
            end = start + colocate_gpu_counts[i]
            if start <= dist.get_rank() < end:
                self._ipc_engine = engine
```

---

## 19. _send_to_colocated_engine — FlattenedTensorBucket

**Code：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py L234-L287
    if getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
        converted_named_tensors_by_dtypes = {"dtype": hf_named_tensors} if hf_named_tensors else {}
    else:
        converted_named_tensors_by_dtypes = {}
        for name, tensor in hf_named_tensors:
            dtype = tensor.dtype
            if dtype not in converted_named_tensors_by_dtypes:
                converted_named_tensors_by_dtypes[dtype] = []
            converted_named_tensors_by_dtypes[dtype].append((name, tensor))

    serialized_tensors = []
    for _dtype, named_tensors in converted_named_tensors_by_dtypes.items():
        flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        metadata = flattened_tensor_bucket.get_metadata()
        flattened_tensor_data = {
            "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
            "metadata": metadata,
        }
        long_live_tensors.append(flattened_tensor_data)
        serialized_tensors.append(MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True))

    dist.gather_object(serialized_tensors, object_gather_list=serialized_named_tensors, dst=ipc_gather_src, group=ipc_gather_group)

    if dist.get_rank() == ipc_gather_src:
        ...
        refs.append(ipc_engine.update_weights_from_tensor.remote(**kwargs))
```

---

## 20. UpdateWeightFromTensor — update_weights 主循环

**Code：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py L147-L191
    @torch.no_grad()
    def update_weights(self) -> None:
        self.weight_version += 1
        rank = dist.get_rank()
        if rank == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(restore_weights_before_load=True, post_process_quantization=False, rollout_engines=self.rollout_engines)
        dist.barrier(group=get_gloo_group())

        megatron_local_weights = self.weights_getter()

        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
            ray.get(refs)
            del long_lived_tensors, hf_named_tensors
            torch.cuda.ipc_collect()

        dist.barrier(group=get_gloo_group())
        torch.cuda.ipc_collect()

        if rank == 0:
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(restore_weights_before_load=False, post_process_quantization=True, rollout_engines=self.rollout_engines)
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())
```

---

## 21. _send_hf_params — 混合 IPC + NCCL

**Code：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py L193-L216
    def _send_hf_params(self, hf_named_tensors) -> tuple[list[ObjectRef], Any]:
        all_refs = []
        refs_colocated, long_lived_tensors = _send_to_colocated_engine(
            hf_named_tensors,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            weight_version=self.weight_version,
        )
        all_refs.extend(refs_colocated)

        if self.use_distribute and self._is_distributed_src_rank:
            refs_distributed = update_weights_from_distributed(
                self._group_name,
                self._model_update_groups,
                self.weight_version,
                self.distributed_rollout_engines,
                hf_named_tensors,
            )
            if refs_distributed:
                all_refs.extend(refs_distributed)

        return all_refs, long_lived_tensors
```

---

## 22. disk_delta — _apply_lock (fcntl)

**Code：**

```python
## 来源：slime/utils/disk_delta.py L69-L78
@contextmanager
def _apply_lock(local_ckpt_dir: str):
    sync = os.path.join(local_ckpt_dir, SYNC_DIR)
    os.makedirs(sync, exist_ok=True)
    with open(os.path.join(sync, "lock"), "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
```

**Comment：** 同 host 多 colocate engine actor 共用一个 lock，避免并发 apply 损坏 mmap。

---

## 23. checksum 算法选型

**Code：**

```python
## 来源：slime/utils/disk_delta.py L49-L66
def _new_hasher(algorithm: str):
    if algorithm == "xxh3-128":
        import xxhash
        return xxhash.xxh3_128()
    if algorithm == "blake3":
        import blake3
        return blake3.blake3()
    if algorithm == "adler32":
        return _Adler32()
    raise KeyError(f"unknown checksum algorithm {algorithm!r}")

def checksum(algorithm: str, buf) -> str:
    hasher = _new_hasher(algorithm)
    hasher.update(buf)
    return hasher.hexdigest()
```

---

## 24. 引擎启动预热 init_local_checkpoint

**Code：**

```python
## 来源：slime/backends/sglang_utils/sglang_engine.py L171-L182
        if self.args.update_weight_mode == "delta" and self.args.update_weight_transport == "disk":
            from slime.utils.disk_delta import init_local_checkpoint

            threading.Thread(
                target=init_local_checkpoint,
                args=(self.args.update_weight_local_checkpoint_dir, self.args.hf_checkpoint),
                daemon=True,
            ).start()
```

---

## 走读小结

| 模块 | 职责 |
|------|------|
| `UpdateWeightFromDisk` | 全量 HF 写共享盘 → engine reload |
| `UpdateWeightFromDiskDelta` | diff + 压缩发布 → host apply → reload 本地盘 |
| `disk_delta` | baseline materialize、版本链 apply、checksum |
| `UpdateWeightFromTensor` | colocate IPC；可选混合 NCCL 远端 |

---

## 25. 测试锚点 test_full_disk_weight_update

**Code：**

```python
## 来源：tests/test_full_disk_weight_update.py L1-L5
"""E2E smoke test for full checkpoint weight updates through disk.

Runs a tiny Qwen3.5-0.8B job where each weight sync writes a complete HF
checkpoint and rollout engines reload it through ``update_weights_from_disk``.
"""
```

**Comment：** CI 验证 full disk 闭环；delta 见 `examples/delta_weight_sync/` 与 docs `delta-weight-sync.md`。
