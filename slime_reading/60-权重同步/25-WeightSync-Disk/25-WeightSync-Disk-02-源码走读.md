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
updated: 2026-07-05
---

# 磁盘权重同步 · 源码走读

> 走读顺序：`UpdateWeightFromDisk` → `UpdateWeightFromDiskDelta` → `disk_delta` → `SGLangEngine` → `UpdateWeightFromTensor` 对照。
> 基线 commit `22cdc6e1`。

## 源码阅读依据

| 上游文件 | 本文关注点 |
|----------|------------|
| `slime/slime/backends/megatron_utils/update_weight/update_weight_from_disk.py` | 全量 HF checkpoint 写盘与 SGLang disk reload |
| `slime/slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py` | delta 编码、发布、engine reload 编排 |
| `slime/slime/utils/disk_delta.py` | host-local checkpoint 初始化、mmap apply、版本链与 checksum |
| `slime/slime/backends/sglang_utils/sglang_engine.py` | engine 侧本地 checkpoint apply 与 HTTP reload |
| `slime/slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py` | tensor IPC/NCCL 直传路径，用作磁盘路径的设计对照 |
| `slime/tests/test_full_disk_weight_update.py` | full disk 同步的 E2E smoke 测试 |

## 设计主线：为什么磁盘同步不是简单 save/load

Slime 的磁盘权重同步同时解决三个问题：训练侧 Megatron 权重要转成 HF 形态，rollout 侧 SGLang 只暴露 reload/update 接口，多机文件系统还可能存在可见性和并发 apply 问题。因此源码把同步拆成两种模式：

1. **full disk：** 所有训练 rank 参与 Megatron→HF 保存，rank 0 暂停 rollout、触发 SGLang 从共享目录 reload，再按配置清理目录。它追求实现简单和兼容性。
2. **disk delta：** 训练侧只发布每轮变化的 safetensors delta；每台 rollout host 把 delta apply 到自己的 local checkpoint，再让 SGLang 走普通 `update_weights_from_disk`。它追求少写盘、少跨网传输，并刻意让 SGLang 不需要理解 delta。
3. **tensor path 对照：** colocated engine 走 GPU→CPU IPC + Ray，远端 engine 走 NCCL distributed。磁盘路径的价值正是在不建立 rollout NCCL 组时仍能同步权重。

阅读这篇时要持续追踪三个边界：**谁控制 Ray/HTTP，谁参与 torch distributed collective，哪份 checkpoint 是共享目录、哪份是 host-local 目录**。

---

## 1. UpdateWeightFromDisk — 类骨架

**Explain：** full disk updater 自身不持有 NCCL 更新组，只缓存 rollout engine handles 和 metrics。

**问题与约束：** full disk 路径依赖共享文件系统与 SGLang reload，不需要像 tensor/distributed 路径那样建立额外通信组。

**设计选择：** `connect_rollout_engines` 只记录 actor；`disconnect_rollout_engines` 为空；`pop_metrics` 保持和其他 updater 一样的接口。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_disk.py L21-L59
class UpdateWeightFromDisk:
    """Full-weight sync through a shared filesystem and SGLang disk reload."""
    ...
    def connect_rollout_engines(...):
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock

    def disconnect_rollout_engines(self) -> None:
        return

    def pop_metrics(self) -> dict[str, float]:
        out, self.update_weight_metrics = self.update_weight_metrics, {}
        return out
```

**为什么这样写：** 它把 full disk 实现成最薄的适配层：训练侧负责写出 HF checkpoint，rollout 侧负责 reload，通信复杂度交给文件系统和 Ray 控制面。

**不变量与失败模式：** `rollout_engines` 必须在首次 `update_weights` 前连接；metrics 被 `pop_metrics` 读后清空，调用方不能假设可重复读取。

**Comment：** full disk 的“简单”不是没有同步，而是没有额外权重传输组。

---

## 2. UpdateWeightFromDisk — update_weights 全流程

**Explain：** full disk 同步被分成清理版本目录、暂停/flush rollout、集体保存 HF、rank 0 触发 reload、恢复 generation 五个阶段。

**问题与约束：** 保存 HF checkpoint 是训练 rank 的 collective；SGLang reload 是 Ray/HTTP 控制面动作，只应由 rank 0 触发。

**设计选择：** 每个阶段之间用 Gloo barrier 隔开；rank 0 控 rollout engine，所有 rank 参与 `save_hf_model_to_path`。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_disk.py L61-L98
self.weight_version += 1
version_dir = Path(self.args.update_weight_disk_dir) / f"weight_v{self.weight_version:06d}"
...
ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
...
save_hf_model_to_path(...)
...
engine.update_weights_from_disk.remote(model_path=str(version_dir), weight_version=str(self.weight_version))
...
ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
```

**为什么这样写：** rank 0 只做控制面，所有 rank 只在应该参与 collective 的地方参与；barrier 把文件系统可见性和训练同步点显式化。

**不变量与失败模式：** reload 必须发生在 `save_hf_model_to_path` 全部 rank 完成后；若没有 pause/flush，rollout 可能在权重切换中继续用旧 KV/cache 生成。

**Comment：** full disk 的核心不是保存函数，而是阶段顺序。

---

## 3. UpdateWeightFromDiskDelta — 继承与 init

**Explain：** delta updater 继承 distributed updater，是为了复用 Megatron→HF tensor gather 逻辑，但最终传输介质改成共享磁盘 delta。

**问题与约束：** delta 需要先得到 canonical HF tensor；这部分和 NCCL distributed 权重更新已有实现重复。

**设计选择：** 继承 `UpdateWeightFromDistributed`，复用 `_iter_non_expert_chunks`、`_iter_expert_chunks` 等 HF tensor iterator；自身维护 CPU snapshot、encoding、checksum 和 commit hook。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py L30-L58
class UpdateWeightFromDiskDelta(UpdateWeightFromDistributed):
    """Delta weight sync over a shared filesystem..."""
    ...
    self.delta_encoding = args.update_weight_delta_encoding
    self.checksum_algorithm = args.update_weight_delta_checksum
    self._snapshot: dict[str, np.ndarray] = {}
    self._baseline_captured = False
    ...
    self._commit_hook = load_function(args.custom_delta_pre_push_path)
```

**为什么这样写：** delta 的创新点不在 HF tensor 生成，而在“如何把变化写成可 apply 的版本流”；继承避免重写分布式 gather。

**不变量与失败模式：** snapshot 的 tensor 名称必须与 HF iterator 产出的名称一致；自定义 commit hook 有副作用时必须保持幂等。

**Comment：** 这是一种务实复用：继承的是数据抽取能力，不继承 NCCL 发送语义。

---

## 4. connect_rollout_engines — all_engine_actors

**Explain：** delta apply 是 host-local 操作，所以需要每台 host 一个 actor 来 apply 本地 checkpoint，而不只是 node 0 的 rollout engine。

**问题与约束：** 多个 engine 可能共享同一台机器上的 local checkpoint；如果只通知 rollout_engines，某些 host 的本地 checkpoint 可能没被更新。

**设计选择：** `all_engine_actors` 记录每台 host 的 actor；`_is_pp_src_rank` 限定哪些训练 rank 负责发布 HF tensor diff。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py L60-L75
self.rollout_engines = rollout_engines
self.all_engine_actors = list(all_engine_actors or rollout_engines)
self._is_pp_src_rank = (
    mpu.get_data_parallel_rank(with_context_parallel=True) == 0
    and mpu.get_tensor_model_parallel_rank() == 0
)
```

**为什么这样写：** reload 是 engine 级动作，apply 是 host 级动作；源码把这两个 actor 集合分开，避免把控制粒度混在一起。

**不变量与失败模式：** `all_engine_actors` 应覆盖所有需要本地 apply 的 host；`disconnect_rollout_engines` 为空，因为 disk delta 不维护 NCCL rollout 组。

**Comment：** 这是 delta 路径和 full disk 路径的第一个关键差异。

---

## 5. _capture_baseline — 首轮只建基线

**Explain：** delta updater 首次调用只捕获 baseline，不发布 delta；baseline 从 `hf_checkpoint` 读取，缺失 tensor 才 fallback 到当前 gathered tensor。

**问题与约束：** rollout host 的 local checkpoint 也是从 `hf_checkpoint` materialize 出来的；snapshot 必须和 engine base 一致，否则第一轮 delta 会建立在错误基线之上。

**设计选择：** rank 0 清理旧 delta stream；所有 rank 用 `make_tensor_reader(args.hf_checkpoint)` seed snapshot；缺失项 warning 后用当前权重。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py L98-L124
if dist.get_rank() == 0:
    shutil.rmtree(self.delta_dir, ignore_errors=True)
    os.makedirs(self.delta_dir, exist_ok=True)
...
read_hf = make_tensor_reader(self.args.hf_checkpoint)
for name, tensor in self._iter_hf_tensors():
    try:
        self._snapshot[name] = read_hf(name)
    except KeyError:
        self._snapshot[name] = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().reshape(-1)
```

**为什么这样写：** delta 必须描述“从 engine 当前 base 到训练新权重”的变化，而不是“从训练初始权重到训练新权重”的变化；HF checkpoint 是两侧共同 base。

**不变量与失败模式：** 旧 delta 目录必须清空，否则新 run 可能把版本链接到旧 base；snapshot 与 local checkpoint 不一致会在 apply checksum 或模型语义上出错。

**Comment：** “首轮无 publish”是 delta 版本链正确性的前提。

---

## 6. _encode_delta — 流水线 diff + 压缩

**Explain：** `_encode_delta` 对每个 gathered HF tensor 做 uint8 diff，变化的 tensor 才压缩写入 `_delta`。

**问题与约束：** 大模型权重 diff 是内存带宽和 GPU→CPU 拷贝敏感路径；不能简单 `.cpu()` 后串行 xor/compress。

**设计选择：** 先尝试分配 pinned host buffer 池，主线程负责拷贝和提交任务，线程池负责 diff、zstd level 1 压缩和 checksum；处理完就把 `snapshot[name]` 更新成新 base。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py L195-L269
max_bytes = max((int(v.nbytes) for v in snapshot.values()), default=0)
...
def diff_and_compress(name, buf, nbytes, pinned):
    ...
    if self.delta_encoding == "xor":
        diff = new ^ old
        changed = int(np.count_nonzero(diff))
    elif self.delta_encoding == "overwrite":
        mask = new != old
        changed = int(np.count_nonzero(mask))
        diff = overwrite_encode(new, mask)
    ...
    compressed = np.frombuffer(zstandard.ZstdCompressor(level=1).compress(diff), dtype=np.uint8)
    return name, new, compressed, checksum(self.checksum_algorithm, new), changed
...
snapshot[name] = new
```

**为什么这样写：** delta 同步的收益取决于少写少传；但 diff 本身不能成为训练瓶颈，所以源码把拷贝、diff、压缩做成有背压的流水线。

**不变量与失败模式：** `snapshot[name]` 必须在 collect 时更新，否则下一轮会重复计算旧 base 到新权重的 diff；未知 encoding 直接抛错。

**Comment：** checksum 是对新 tensor 全量状态算的，不是对 compressed delta 算的，这样 apply 后能验证最终权重。

---

## 7. _write_delta_files — 跨 rank 协调 index

**Explain：** 每个有变化的 rank 写一个 safetensors shard，rank 0 写 canonical HF index，metadata 描述版本链和编码方式。

**问题与约束：** 共享文件系统不适合用“看到哪些文件”来推断全局写入状态；rank 间应通过 distributed object gather 协调 index。

**设计选择：** `all_gather_object` 先收集哪些 rank 有 delta，生成无空洞文件编号；再收集 tensor name → filename map，由 rank 0 写 `model.safetensors.index.json`。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py L132-L167
dist.all_gather_object(counts, int(bool(self._delta)), group=group)
offset, total = sum(counts[:rank]), sum(counts)
...
fname = f"model-{offset:05d}-of-{total:05d}.safetensors"
blob = safetensors.numpy.save(self._delta, metadata=self._checksums)
_atomic_write(os.path.join(self._version_dir, fname), blob)
...
"metadata": {
    "version": f"{self.weight_version:06d}",
    "base_version": f"{self.weight_version - 1:06d}",
    "delta_encoding": self.delta_encoding,
    "compression_format": "zstd",
    "checksum_format": self.checksum_algorithm,
}
```

**为什么这样写：** 它让 delta 目录看起来像一个 HF checkpoint 目录，但 metadata 语义是“从 base_version 到 version 的 delta”。

**不变量与失败模式：** index 必须在所有 shard 文件 atomic write 后写出；`weight_map` 缺 tensor 会导致 apply 时找不到对应 delta。

**Comment：** 文件编号用 rank 协调而不是文件系统扫描，是为了适配弱一致共享盘。

---

## 8. _reload_engines — sync_local_checkpoint + disk reload

**Explain：** 训练侧发布 delta 后，每台 host 先 apply 到 local checkpoint，再让 rollout engine 从 local path reload。

**问题与约束：** delta 目录只是版本流，不是 SGLang 能直接加载的完整模型目录；SGLang 需要看到已经 patch 好的 HF checkpoint。

**设计选择：** rank 0 调 `actor.sync_local_checkpoint.remote(weight_version)` 覆盖所有 host，再对 `rollout_engines` 调普通 `update_weights_from_disk`，`model_path` 指向 local checkpoint dir。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py L169-L186
if self._commit_hook is not None:
    self._commit_hook(self.args, self._version_dir, list(self.rollout_engines))
...
ray.get([actor.sync_local_checkpoint.remote(self.weight_version) for actor in self.all_engine_actors])
ray.get([
    engine.update_weights_from_disk.remote(
        model_path=self.args.update_weight_local_checkpoint_dir,
        weight_version=str(self.weight_version),
    )
    for engine in self.rollout_engines
])
ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
```

**为什么这样写：** 它把 delta 支持封装在 Slime 的 engine actor 层，SGLang 服务端继续使用已有 disk reload 能力。

**不变量与失败模式：** `sync_local_checkpoint` 必须先于 reload 完成；`model_path` 必须是 host-local 完整 checkpoint，而不是共享 delta 目录。

**Comment：** 这是 disk delta 的核心哲学：让推理引擎不感知 delta。

---

## 9. _record_metrics — density / wire

**Explain：** delta updater 记录变化密度和实际 wire bytes，用于判断 delta 是否真的比 full checkpoint 划算。

**问题与约束：** 单 rank 的 changed/wire 只代表本 rank；训练日志需要全局视角。

**设计选择：** 把 `[changed_bytes, total_bytes, wire_bytes]` 做 CUDA tensor 后 `all_reduce`，再写入 `update_weight_metrics`。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py L271-L290
counts = torch.tensor(
    [self.changed_bytes, self.total_bytes, self.wire_bytes],
    dtype=torch.int64,
    device=torch.cuda.current_device(),
)
dist.all_reduce(counts)
changed, total, wire = counts.tolist()
m["perf/update_weights_density"] = changed / max(total, 1)
m["perf/update_weights_wire_bytes"] = wire
```

**为什么这样写：** density 是 delta 路径的健康指标；如果变化密度接近 1 或 wire bytes 很大，full disk/tensor path 可能更合适。

**不变量与失败模式：** `self.wire_bytes` 只在写文件时设置；没有变化的 rank 应贡献 0。`max(total, 1)` 避免空统计分母为 0。

**Comment：** metrics 是设计反馈回路，不只是日志装饰。

---

## 10. _atomic_write

**Explain：** delta shard 和 index 都通过 tmp 文件 + fsync + replace 原子落盘。

**问题与约束：** rollout host 可能在 trainer 写文件后立刻读取；半写入文件会导致 safetensors header 或 payload 损坏。

**设计选择：** 先写 `path.tmp`，flush/fsync 后 `os.replace` 到目标路径。

**代码逻辑：**

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

**为什么这样写：** 文件系统同步的正确性依赖读者只能看到旧完整文件或新完整文件，不能看到中间态。

**不变量与失败模式：** tmp 与目标文件应在同一 filesystem 上；跨 filesystem replace 不能保证同样语义。

**Comment：** 这是 shared filesystem 协议的一部分，不是普通 IO 小优化。

---

## 11. disk_delta — make_tensor_reader

**Explain：** `make_tensor_reader` 先扫描 safetensors header 建 name→文件偏移索引，再按 name 直接 seek 读取 tensor bytes。

**问题与约束：** baseline capture 会读大量 tensor；每次读都重扫所有 header 会浪费 IO。

**设计选择：** `_tensor_locations` 解析所有 safetensors 的 `data_offsets`，闭包 `read(name)` 只做 seek/read，返回 uint8 view。

**代码逻辑：**

```python
## 来源：slime/utils/disk_delta.py L126-L152
def _tensor_locations(ckpt_dir: str) -> dict[str, tuple[str, int, int]]:
    ...
    begin, end = info["data_offsets"]
    locations[name] = (path, 8 + header_len + begin, end - begin)

def make_tensor_reader(ckpt_dir: str):
    locations = _tensor_locations(ckpt_dir)

    def read(name: str) -> np.ndarray:
        path, offset, nbytes = locations[name]
        with open(path, "rb") as f:
            f.seek(offset)
            return np.frombuffer(f.read(nbytes), dtype=np.uint8)
```

**为什么这样写：** delta 以 byte-level diff 工作，读取接口也直接返回 uint8 bytes，避免 dtype/shape 层的额外解释。

**不变量与失败模式：** checkpoint 必须是 safetensors；tensor name 不存在时让 `KeyError` 冒泡给 caller 决定 fallback。

**Comment：** 这也是为什么 delta snapshot 可以不关心 tensor dtype。

---

## 12. disk_delta — apply_xor 分块

**Explain：** xor delta apply 通过 zstd streaming 解压，每次读 2 MB 块，对 mmap region 原地 XOR，并增量更新 checksum。

**问题与约束：** full tensor 可能很大；一次性解压完整 diff 会增加内存峰值，apply 还要尽量利用 page cache。

**设计选择：** 使用 `stream_reader` 分块解压；每个 chunk 写入 mmap region 后立刻喂给 hasher。

**代码逻辑：**

```python
## 来源：slime/utils/disk_delta.py L204-L220
region = np.ndarray((nbytes,), dtype=np.uint8, buffer=open_mmaps[path][1], offset=offset)
hasher = _new_hasher(algorithm)
reader = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(bytes(compressed)))
pos = 0
while pos < nbytes:
    block = reader.read(min(2 << 20, nbytes - pos))
    ...
    region[pos : pos + chunk.size] ^= chunk
    hasher.update(region[pos : pos + chunk.size])
...
if hasher.hexdigest() != want:
    mismatches.append(name)
```

**为什么这样写：** xor diff 可逆且适合稀疏/小变化压缩，但 apply 后必须验证最终 tensor，而不是只相信 delta 文件。

**不变量与失败模式：** 本地 checkpoint 当前版本必须等于 delta 的 base_version；否则 XOR 会把 bytes 翻到错误状态，checksum 会失败或模型错误。

**Comment：** 2 MB chunk 是内存层面的工程选择：控制峰值，同时保持足够大的顺序 IO。

---

## 13. disk_delta — apply_overwrite

**Explain：** overwrite delta 存 changed positions 和新 bytes，apply 时只把这些位置写进 mmap region。

**问题与约束：** XOR 对同一 delta 重复 apply 会翻回去，不是幂等；overwrite 编码更适合需要幂等语义的场景。

**设计选择：** delta 前 4 bytes 存 changed count，随后是 uint32 positions，再后面是新值 payload。

**代码逻辑：**

```python
## 来源：slime/utils/disk_delta.py L222-L231
delta = np.frombuffer(zstandard.ZstdDecompressor().decompress(bytes(compressed)), dtype=np.uint8)
region = np.ndarray((nbytes,), dtype=np.uint8, buffer=open_mmaps[path][1], offset=offset)
count = int.from_bytes(delta[:4].tobytes(), "little")
positions = np.frombuffer(delta[4 : 4 + 4 * count].tobytes(), dtype="<u4")
region[positions] = delta[4 + 4 * count :]
if checksum(algorithm, region) != want:
    mismatches.append(name)
```

**为什么这样写：** 它用更大的索引开销换幂等 apply 属性；编码选择因此被做成参数。

**不变量与失败模式：** positions 必须在 tensor byte range 内；checksum mismatch 必须 fail loud，不能继续 reload。

**Comment：** overwrite 和 xor 是同一版本协议下的两种 byte-level 语义。

---

## 14. disk_delta — _apply_version 版本检查

**Explain：** `_apply_version` 先检查本地 applied version 是否等于 delta 的 base_version，保证版本链顺序。

**问题与约束：** delta 不是完整 checkpoint；跳版本或乱序 apply 都会把本地 bytes 改坏。

**设计选择：** 读取 delta index metadata；如果本地已是目标版本直接返回，如果不是 base_version 就抛 `RuntimeError`。

**代码逻辑：**

```python
## 来源：slime/utils/disk_delta.py L155-L164
with open(os.path.join(version_dir, "model.safetensors.index.json")) as f:
    meta = json.load(f)["metadata"]
applied = _read_applied_version(local_ckpt_dir)
if applied == meta["version"]:
    return
if applied != meta["base_version"]:
    raise RuntimeError(f"out-of-order delta: local at {applied}, delta builds on {meta['base_version']}")
```

**为什么这样写：** 它把 delta 目录从“若干文件”提升成严格版本流，避免弱一致文件系统或重复调用造成 silent corruption。

**不变量与失败模式：** `state.json` 必须准确记录本地 checkpoint 版本；缺失或落后都会阻止错误 apply。

**Comment：** 原笔记把两个源码范围写在一条 `来源` 里，审计无法识别；这里拆成独立证据点。

---

## 15. disk_delta — apply_deltas 版本链

**Explain：** `apply_deltas` 在 host-local lock 下从当前 applied version 逐个版本 apply 到 target version。

**问题与约束：** rollout host 可能跳过中间训练 step 才 reload；它需要顺序补齐 delta 链。

**设计选择：** 先检查 local checkpoint 已 materialize，再对 `range(applied + 1, target + 1)` 逐个调用 `_apply_version`。

**代码逻辑：**

```python
## 来源：slime/utils/disk_delta.py L255-L264
def apply_deltas(local_ckpt_dir: str, delta_root: str, target_version: int) -> None:
    with _apply_lock(local_ckpt_dir):
        applied = _read_applied_version(local_ckpt_dir)
        if applied is None:
            raise RuntimeError("local checkpoint not materialized")
        for version in range(int(applied) + 1, target_version + 1):
            _apply_version(local_ckpt_dir, os.path.join(delta_root, f"weight_v{version:06d}"))
```

**为什么这样写：** 它让 engine 侧只声明目标版本，不需要知道中间版本列表；版本链逻辑集中在 disk_delta。

**不变量与失败模式：** 中间任一 `weight_vXXXXXX` 缺失都会失败；这意味着 delta 目录的保留策略必须覆盖最慢 host 的 apply 窗口。

**Comment：** 这个函数是 host-local checkpoint 的状态机入口。

---

## 16. SGLangEngine — sync_local_checkpoint

**Explain：** engine actor 在 reload 前确保本地 checkpoint 初始化，并把共享 delta 目录 apply 到目标版本。

**问题与约束：** 某些非 POSIX 文件系统没有跨 host 的 read-after-write 一致性；刚发布的 delta 可能需要自定义 hook 刷新到本机可见。

**设计选择：** `init_local_checkpoint` 幂等调用；如配置 `custom_delta_pre_read_path`，先执行 hook，再 `apply_deltas`。

**代码逻辑：**

```python
## 来源：slime/backends/sglang_utils/sglang_engine.py L396-L413
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

**为什么这样写：** Slime 把“共享 delta 如何到达本 host”做成 hook，把“如何把 delta 应用到本地 checkpoint”留在通用 helper。

**不变量与失败模式：** actor 必须和它驱动的 SGLang 共享 local checkpoint filesystem；否则 apply 成功也不是 SGLang reload 的目录。

**Comment：** 这段解释了为什么 delta reload 的 `model_path` 是 local checkpoint，而不是 delta root。

---

## 17. SGLangEngine — update_weights_from_disk HTTP

**Explain：** engine 侧最终调用 SGLang HTTP endpoint `update_weights_from_disk`，payload 只描述 model path、load format、version 和可选 files。

**问题与约束：** Slime 需要支持 full disk、local patched checkpoint、以及 SGLang 原生 delta load_format 的可能性，但调用入口应统一。

**设计选择：** 构造 payload 后交给 `_make_request("update_weights_from_disk", payload)`；没有参数的字段不写入 payload。

**代码逻辑：**

```python
## 来源：slime/backends/sglang_utils/sglang_engine.py L415-L437
payload: dict = {"model_path": model_path}
if load_format is not None:
    payload["load_format"] = load_format
if weight_version is not None:
    payload["weight_version"] = weight_version
if files is not None:
    payload["files"] = files
return self._make_request("update_weights_from_disk", payload)
```

**为什么这样写：** HTTP endpoint 是 rollout 引擎边界；Slime 侧把 full/delta 的准备工作做完后，用同一个边界触发 reload。

**不变量与失败模式：** `node_rank != 0` 的 engine actor 不发 HTTP 请求；真正处理 reload 的是 SGLang node 0 server。

**Comment：** 统一 endpoint 是 disk delta 能“隐藏在 Slime 层”的基础。

---

## 18. UpdateWeightFromTensor — connect 混合 colocate + 远端

**Explain：** tensor updater 根据 engine GPU offset 判断哪些 engine 与 actor colocate，剩余 engine 才走 distributed NCCL 更新。

**问题与约束：** 同一作业可能同时有本机 rollout engine 和远端 rollout engine；最优传输路径不同。

**设计选择：** 先计算 actor 覆盖的 GPU 范围；前缀 colocated engines 走 IPC/Ray，后缀 distributed engines 建权重更新组。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py L86-L117
total_actor_gpus = self.args.actor_num_nodes * self.args.actor_num_gpus_per_node
colocate_engine_nums = 0
for gpu_offset, gpu_count in zip(engine_gpu_offsets, engine_gpu_counts, strict=True):
    if gpu_offset + gpu_count > total_actor_gpus:
        break
    colocate_engine_nums += 1

self.use_distribute = len(rollout_engines) > colocate_engine_nums
...
self._model_update_groups = connect_rollout_engines_from_distributed(...)
```

**为什么这样写：** tensor 直传追求低延迟，但不能假设所有 engine 都和 actor 在同一 GPU 域；源码按拓扑分流。

**不变量与失败模式：** engine offsets 必须按布局有序；一旦出现 placeholder/gap，fallback dense 假设可能不够准确。

**Comment：** 这段放在磁盘专题里，是为了对比 disk path 为什么可以避开 rollout NCCL 组。

---

## 19. IPC Gloo gather group

**Explain：** colocated tensor path 为每个 engine 的 GPU rank 范围创建 Gloo gather group，并把该组 src rank 绑定到 engine actor。

**问题与约束：** 多 GPU engine 的 HF tensor bucket 由多个训练 rank 持有，发送给一个 engine 前需要在 CPU/Gloo 侧 gather。

**设计选择：** group ranks 由 engine GPU offset/count 得到；组内最低 rank 是 gather src，也是发 Ray IPC 请求的 rank。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py L124-L137
group_ranks = list(range(colocate_gpu_offsets[i], colocate_gpu_offsets[i] + colocate_gpu_counts[i]))
new_group = dist.new_group(ranks=group_ranks, backend="gloo")
if dist.get_rank() in group_ranks:
    self._ipc_gather_group = new_group
    self._ipc_gather_src = colocate_gpu_offsets[i]
...
if start <= dist.get_rank() < end:
    self._ipc_engine = engine
```

**为什么这样写：** GPU tensor 更新要跨训练 rank 聚合，但 Ray actor 调用应只由一个 src rank 发起，避免重复更新同一 engine。

**不变量与失败模式：** placeholder rank 没有 gather group，应在发送函数里跳过；group 划分在 reconnect 后被假定固定。

**Comment：** 这和 disk updater 的“rank 0 控制面”形成对照：tensor colocate 是每个 engine group 一个 src rank。

---

## 20. _send_to_colocated_engine — FlattenedTensorBucket

**Explain：** colocated tensor path 把 HF named tensors 打平成 bucket，经 Gloo gather 到 src rank，再用 Ray IPC 发给 engine。

**问题与约束：** Ray 传输需要序列化元数据和持有底层 CUDA/CPU tensor；不同 dtype 的 bucket 可能要分组。

**设计选择：** 如果 `FlattenedTensorBucket` 不支持多 dtype，就按 dtype 分 bucket；src rank 对齐各 rank bucket 数，不足处补 empty bucket。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py L234-L287
for name, tensor in hf_named_tensors:
    dtype = tensor.dtype
    converted_named_tensors_by_dtypes.setdefault(dtype, []).append((name, tensor))
...
flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
metadata = flattened_tensor_bucket.get_metadata()
...
dist.gather_object(serialized_tensors, object_gather_list=serialized_named_tensors, dst=ipc_gather_src, group=ipc_gather_group)
...
refs.append(ipc_engine.update_weights_from_tensor.remote(**kwargs))
```

**为什么这样写：** 它减少 Ray 调用次数和对象数量，同时保留 engine 侧还原 tensor name/shape/dtype 的 metadata。

**不变量与失败模式：** `long_live_tensors` 必须在 engine 消费前存活；否则 IPC handle 指向的底层存储可能过早释放。

**Comment：** 磁盘 delta 用 safetensors 做持久协议，tensor path 用 flattened bucket 做进程间协议。

---

## 21. UpdateWeightFromTensor — update_weights 主循环

**Explain：** tensor updater 每轮暂停 rollout，按 HF weight chunks 发送，等待 refs 完成后清理 CUDA IPC，再恢复 generation。

**问题与约束：** 权重更新期间不能让 rollout 继续生成；分块发送后要释放 IPC 资源，避免长时间占用缓存。

**设计选择：** rank 0 控制 pause/flush/continue；所有 rank 获取本地 Megatron weights，按 HF iterator chunk 循环发送。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py L147-L191
self.weight_version += 1
if rank == 0:
    ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
    ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
...
megatron_local_weights = self.weights_getter()
for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
    refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
    ray.get(refs)
    del long_lived_tensors, hf_named_tensors
    torch.cuda.ipc_collect()
...
ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
```

**为什么这样写：** tensor path 是在线传输协议，不像 disk path 有完整目录作为版本边界，所以每个 chunk 都要等 engine 确认再释放。

**不变量与失败模式：** `ray.get(refs)` 必须在释放 long-lived tensors 前完成；压缩 tensor 量化路径需要 update 前后 post process。

**Comment：** 这段帮助理解 disk path 的取舍：disk path 慢但版本边界更明确，tensor path 快但状态更多。

---

## 22. _send_hf_params — 混合 IPC + NCCL

**Explain：** `_send_hf_params` 对同一批 HF tensors 同时处理 colocated IPC 和远端 distributed engines。

**问题与约束：** 一批权重 chunk 可能需要发给两类 rollout engine；调用方只想等待一个 refs 列表。

**设计选择：** 先调用 `_send_to_colocated_engine`，再在 distributed src rank 上调用 `update_weights_from_distributed`，最后合并 refs。

**代码逻辑：**

```python
## 来源：slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py L193-L216
refs_colocated, long_lived_tensors = _send_to_colocated_engine(...)
all_refs.extend(refs_colocated)

if self.use_distribute and self._is_distributed_src_rank:
    refs_distributed = update_weights_from_distributed(...)
    if refs_distributed:
        all_refs.extend(refs_distributed)

return all_refs, long_lived_tensors
```

**为什么这样写：** 它把拓扑差异隐藏在发送函数内部，主循环只关心“这批 chunk 是否全部完成”。

**不变量与失败模式：** 只有 distributed src rank 触发远端更新；非 src rank 返回空 distributed refs 是预期行为。

**Comment：** 磁盘同步没有这层混合发送，因为它把传输协议统一成文件版本。

---

## 23. disk_delta — _apply_lock

**Explain：** host-local checkpoint apply 用 `fcntl.flock` 串行化，防止同 host 多 actor 同时改 mmap 文件。

**问题与约束：** 多个 colocated engine actor 可能共享一个 local checkpoint 目录；并发 apply 会写坏同一批 safetensors。

**设计选择：** 在 checkpoint 的 `.delta_sync/lock` 上加排他锁，`apply_deltas` 和 `init_local_checkpoint` 都经由该锁。

**代码逻辑：**

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

**为什么这样写：** delta apply 是 host-local 临界区；用文件锁比跨进程 Python lock 更适合 Ray actor 多进程模型。

**不变量与失败模式：** 该锁只保护同一 host/同一 filesystem 的进程；跨 host 一致性仍由版本目录和 hook 处理。

**Comment：** 这也是为什么 `all_engine_actors` 可以每 host 一个：多 engine 会被 lock 合并成串行 apply。

---

## 24. checksum 算法选型

**Explain：** checksum helper 支持 `xxh3-128`、`blake3`、`adler32`，统一暴露 `.hexdigest()`。

**问题与约束：** delta apply 后必须校验最终 tensor bytes；不同环境可能在速度和依赖上有不同偏好。

**设计选择：** `_new_hasher` 按配置动态 import；`adler32` 用本地 wrapper 适配同一接口。

**代码逻辑：**

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

**为什么这样写：** 校验是安全边界，算法是性能/依赖选择；源码把这两者分开。

**不变量与失败模式：** publish 端和 apply 端必须使用同一 `checksum_format`；未知算法直接报错，不能默认跳过校验。

**Comment：** 对权重同步而言，checksum 失败应当停止服务新权重，而不是降级继续。

---

## 25. 引擎启动预热 init_local_checkpoint

**Explain：** SGLang engine 初始化时，如果使用 disk delta，会后台启动线程 materialize base checkpoint。

**问题与约束：** 首次 delta reload 前必须有本地完整 HF checkpoint；但完整复制可能很慢，不应阻塞 engine 启动和首轮 rollout。

**设计选择：** daemon thread 调 `init_local_checkpoint(local_dir, hf_checkpoint)`；真正 reload 时 `sync_local_checkpoint` 再幂等确认。

**代码逻辑：**

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

**为什么这样写：** 预热把 full base copy 的 IO 成本提前到后台；flock 和幂等检查保证 reload 时不会和后台复制互相踩踏。

**不变量与失败模式：** 首次 `sync_local_checkpoint` 可能阻塞等待 lock；这比 reload 一个未完成 checkpoint 更安全。

**Comment：** 这是 delta 路径对用户体验的优化：首轮生成不等 base copy，首次更新才要求它完成。

---

## 26. 测试锚点 test_full_disk_weight_update

**Explain：** E2E smoke test 验证 full disk 路径会实际写出版本目录、index 和 safetensors 文件。

**问题与约束：** 单元测试很难覆盖 Megatron 保存、SGLang reload、Ray 控制面和训练参数组合的闭环。

**设计选择：** 测试启动一个小 Qwen3.5-0.8B job，打开 `--update-weight-mode full --update-weight-transport disk`，最后断言 disk checkpoint 产物存在。

**代码逻辑：**

```python
## 来源：tests/test_full_disk_weight_update.py L1-L5
"""E2E smoke test for full checkpoint weight updates through disk.

Runs a tiny Qwen3.5-0.8B job where each weight sync writes a complete HF
checkpoint and rollout engines reload it through ``update_weights_from_disk``.
"""
```

**为什么这样写：** full disk 的风险在跨系统边界，E2E smoke 比 isolated helper test 更能覆盖真实失败模式。

**不变量与失败模式：** 测试依赖模型、数据集和多 GPU 环境；它证明 full disk 闭环，不证明 delta apply 的所有版本链场景。

**Comment：** delta 相关行为还需要结合 `disk_delta` helper 和 examples/docs 理解。

---

## 走读小结

| 路径 | 核心协议 | 适合场景 | 主要风险 |
|------|----------|----------|----------|
| `UpdateWeightFromDisk` | 写完整 HF checkpoint，SGLang reload | 简单、兼容、调试友好 | 写盘量大，依赖共享盘 |
| `UpdateWeightFromDiskDelta` | 训练侧发布 delta，host-local apply 后 reload | 权重变化稀疏、多 host rollout | 版本链、checksum、local checkpoint 一致性 |
| `UpdateWeightFromTensor` | colocated IPC + 可选 NCCL distributed | 低延迟直传 | 通信组、IPC 生命周期、拓扑复杂 |

这条源码的设计哲学是：**让每种同步方式只承担自己擅长的部分**。full disk 追求最小机制，disk delta 把差分复杂度包在 Slime 侧，tensor path 面向拓扑优化；SGLang 侧尽量只暴露稳定的 reload/update HTTP 边界。
