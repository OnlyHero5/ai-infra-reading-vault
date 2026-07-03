---
type: batch-doc
module: 24-WeightSync-Dist
batch: "24"
doc_type: walkthrough
title: "NCCL 权重同步 · 源码走读"
tags:
  - slime/batch/24
  - slime/module/weight-sync-dist
  - slime/doc/walkthrough
updated: 2026-07-02
---

# NCCL 权重同步 · 源码走读

## 走读顺序

1. `actor.py` — `update_weights` 编排
2. `update_weight/common.py` — 参数枚举与 TP gather
3. `update_weight_from_distributed.py` — NCCL 同步主逻辑
4. `hf_weight_iterator_direct.py` — 分桶 HF 迭代（raw 模式 / checkpoint）

---

## 1. update_weights：Actor 侧入口

**Explain：** 每轮 train 后由 `train.py` / `train_async.py` 调用。先 fault-tolerance 恢复引擎，再向 RolloutManager 取可更新引擎与分布式锁；必要时 `connect_rollout_engines`；最后委托 `weight_updater.update_weights()`。

**Code：**

```python
## 来源：slime/backends/megatron_utils/actor.py L583-L628
    @timer
    def update_weights(self) -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return
        if self.args.use_fault_tolerance:
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.recover_updatable_engines.remote())
            dist.barrier(group=get_gloo_group())
        (
            rollout_engines,
            rollout_engine_lock,
            num_new_engines,
            engine_gpu_counts,
            engine_gpu_offsets,
            all_engine_actors,
        ) = ray.get(self.rollout_manager.get_updatable_engines_and_lock.remote())
        reconnect_rollout_engines = self.args.offload_train and self.args.use_critic and not self.args.colocate
        if not rollout_engines and not reconnect_rollout_engines:
            if dist.get_rank() == 0:
                logger.info("No updatable SGLang engines are running; skip weight update.")
            return
        if num_new_engines > 0 or reconnect_rollout_engines:
            self.weight_updater.connect_rollout_engines(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
                all_engine_actors=all_engine_actors,
            )
            dist.barrier(group=get_gloo_group())
        with torch_memory_saver.disable() if self.args.offload_train else nullcontext():
            self.weight_updater.update_weights()
```

**Comment：**

- `offload_train + use_critic + 非 colocate` 时先 `wake_up()` 重建 process group 再连引擎
- `keep_old_actor` 队列更新在 `update_weights` 末尾操作 `weights_backuper`（与 rollout 策略相关）
- `--ci-test` 会抽查 engine `weight_version`（L630-L636）

---

## 2. named_params_and_buffers：跨 PP/EP 一致命名

**Explain：** 遍历 VPP 各 stage 的 `named_parameters` / expert_bias buffer，把 layer index 加上 PP offset、expert index 加上 EP offset，yield 全局唯一 `(name, tensor)`。

**Code：**

```python
## 来源：update_weight/common.py L208-L219
            layer_idx, rest = match.groups()
            layer_idx = int(layer_idx) + layer_offset
            expert_pattern = r"mlp\.experts\.(.+)\.(weight|bias)(\d+)"
            match = re.match(expert_pattern, rest)
            if match:
                rest, param_type, expert_idx = match.groups()
                expert_idx = int(expert_idx) + expert_offset
                yield f"module.module.decoder.layers.{layer_idx}.mlp.experts.{rest}.{param_type}{expert_idx}", param
            else:
                yield f"module.module.decoder.layers.{layer_idx}.{rest}", param
```

**Comment：** `convert_to_global_name=False`（vanilla 模式）时 yield `vp_stages.{i}.{name}`，colocate tensor 路径可能使用。

---

## 3. connect_rollout_engines：按需重建 NCCL 组

**Explain：** 新引擎上线或 reconnect 时调用。PP source 先 disconnect 旧组，再 `connect_rollout_engines_from_distributed` 阻塞直到训练 rank 0 与所有 engine GPU join 同一 NCCL group。

**Code：**

```python
## 来源：update_weight/update_weight_from_distributed.py L57-L92
    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        ...
    ) -> None:
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock
        self._engine_gpu_counts = engine_gpu_counts
        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        )
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        if self._is_pp_src_rank:
            self._group_name = f"slime-pp_{pp_rank}"
            if self._model_update_groups is not None:
                disconnect_rollout_engines_from_distributed(...)
            self._model_update_groups = connect_rollout_engines_from_distributed(
                self.args, self._group_name, rollout_engines, engine_gpu_counts=engine_gpu_counts,
            )
```

**Comment：** `sleep()` offload 路径会 `disconnect_rollout_engines` 销毁 NCCL，避免悬挂 group。

---

## 4. update_weights：pause → send → continue

**Explain：** rank 0 通过 HTTP pause 所有引擎并 flush KV；全体 gloo barrier；PP source 上 `_send_weights`；量化模型做 compressed-tensors 前后处理；rank 0 continue generation。

**Code：**

```python
## 来源：update_weight/update_weight_from_distributed.py L102-L134
    @torch.no_grad()
    def update_weights(self) -> None:
        self.weight_version += 1
        if dist.get_rank() == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())
        pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_pp_src_rank else None
        self._send_weights(pbar)
        if dist.get_rank() == 0:
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())
```

**Comment：** pause/continue 走 SGLang HTTP API（`sglang_engine.py`），与 NCCL payload 分离。

---

## 5. _send_weights：非 expert → expert 两趟

**Explain：** 先 `_iter_non_expert_chunks` 按 buffer 大小 yield HF bucket，再 barrier；再 `_iter_expert_chunks` 对 MoE 做 EP gather。每 chunk 经 `_update_bucket_weights_from_distributed` 加锁 broadcast。

**Code：**

```python
## 来源：update_weight/update_weight_from_distributed.py L136-L146
    def _send_weights(self, pbar: tqdm | None) -> None:
        for chunk_iter in (self._iter_non_expert_chunks(), self._iter_expert_chunks()):
            for hf_chunk in chunk_iter:
                self._on_chunk(hf_chunk)
                self._update_bucket_weights_from_distributed(hf_chunk, pbar=pbar)
            dist.barrier(group=get_gloo_group())
```

**Comment：** 子类（如 `UpdateWeightFromDiskDelta`）可 override `_on_chunk` 注入 delta 行为。

---

## 6. _iter_non_expert_chunks：TP gather + HF convert + 分桶

**Explain：** 跳过 `.experts.` 参数；每个 param `all_gather_param`；仅 PP source 调用 `convert_to_hf`；累计字节超 `update_weight_buffer_size` 则 yield 当前 buffer。

**Code：**

```python
## 来源：update_weight/update_weight_from_distributed.py L153-L176
    def _iter_non_expert_chunks(self) -> Iterator[list[tuple[str, torch.Tensor]]]:
        buffer_size = 0
        buffer: list[tuple[str, torch.Tensor]] = []
        for name, param in named_params_and_buffers(self.args, self.model):
            if ".experts." in name:
                continue
            param = all_gather_param(name, param)
            if not self._is_pp_src_rank:
                continue
            hf_chunk = convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
            chunk_bytes = sum(t.numel() * t.element_size() for _, t in hf_chunk)
            if buffer and buffer_size + chunk_bytes > self.args.update_weight_buffer_size:
                yield buffer
                buffer = []
                buffer_size = 0
            buffer.extend(hf_chunk)
            buffer_size += chunk_bytes
        if buffer:
            yield buffer
```

**Comment：** 非 PP source rank 仍执行 all_gather（collective 必须全员参与），但不填充 buffer。

---

## 7. _ep_gather_and_convert：MoE Expert 批量 EP all_gather

**Explain：** expert 参数先 TP gather 进 batch；batch 预估字节 × EP world_size 超阈值则触发 EP 向全 expert rank async all_gather，再在 PP source 上逐个 `convert_to_hf`。

**Code：**

```python
## 来源：update_weight/update_weight_from_distributed.py L216-L238
        all_gathered_params = [[] for _ in range(mpu.get_expert_model_parallel_world_size())]
        handles = []
        for i, (_name, param) in enumerate(named_tensors):
            params = [
                torch.empty_like(param.data, device=torch.cuda.current_device())
                for _ in range(mpu.get_expert_model_parallel_world_size())
            ]
            handle = dist.all_gather(params, param.data, group=mpu.get_expert_model_parallel_group(), async_op=True)
            handles.append(handle)
            for ep_rank, names in enumerate(all_names):
                all_gathered_params[ep_rank].append((names[i], params[ep_rank]))
        for handle in handles:
            handle.wait()
        if not self._is_pp_src_rank:
            return []
        all_gathered_params = sum(all_gathered_params, [])
        converted_hf_tensors = []
        for name, param in all_gathered_params:
            converted_hf_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
        return converted_hf_tensors
```

**Comment：** 先用 `all_gather_object` 对齐各 EP rank 上的 param name 列表，防止 expert 顺序不一致。

---

## 8. update_weights_from_distributed：Ray metadata + NCCL broadcast

**Explain：** 对每个引擎 Ray 调用 `update_weights_from_distributed.remote` 传入 names/dtypes/shapes；训练 rank 0 对 bucket 内每个 tensor async NCCL broadcast；wait 后返回 refs。

**Code：**

```python
## 来源：update_weight/update_weight_from_distributed.py L326-L355
def update_weights_from_distributed(
    group_name: str,
    group: dist.ProcessGroup,
    weight_version: int,
    rollout_engines: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
    load_format: str | None = None,
) -> list[ObjectRef]:
    refs = [
        engine.update_weights_from_distributed.remote(
            names=[name for name, _ in converted_named_tensors],
            dtypes=[param.dtype for _, param in converted_named_tensors],
            shapes=[param.shape for _, param in converted_named_tensors],
            group_name=group_name,
            weight_version=str(weight_version),
            load_format=load_format,
        )
        for engine in rollout_engines
    ]
    handles = []
    for _, param in converted_named_tensors:
        handles.append(dist.broadcast(param.data, 0, group=group, async_op=True))
    for handle in handles:
        handle.wait()
    return refs
```

**Comment：** 引擎侧收到 metadata 后在同 group 上 recv broadcast；`rollout_engine_lock` 保证同一时刻只有一个 bucket 在飞。

---

## 9. _update_bucket_weights_from_distributed：分布式锁

**Explain：** spin-acquire Ray lock → broadcast → ray.get(refs) → release lock → pbar++。注释明确：无锁时多 PP / 多 bucket 并发可能导致 NCCL 死锁。

**Code：**

```python
## 来源：update_weight/update_weight_from_distributed.py L240-L265
    def _update_bucket_weights_from_distributed(
        self,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        pbar: tqdm | None = None,
        load_format: str | None = None,
    ) -> None:
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)
        refs = update_weights_from_distributed(
            self._group_name,
            self._model_update_groups,
            self.weight_version,
            self.rollout_engines,
            converted_named_tensors,
            load_format=load_format,
        )
        ray.get(refs)
        converted_named_tensors.clear()
        ray.get(self.rollout_engine_lock.release.remote())
        pbar.update(1)
```

**Comment：** lock 是 RolloutManager 创建的 Ray actor，跨 PP stage 共享。

---

## 10. HfWeightIteratorDirect.get_hf_weight_chunks

**Explain：** 按预计算 bucket 迭代：`_get_megatron_full_params` 做 PP/EP broadcast + async TP all_gather，再 `_convert_to_hf_named_tensors` yield 一个 chunk。用于 save_hf 等同构分桶逻辑。

**Code：**

```python
## 来源：update_weight/hf_weight_iterator_direct.py L24-L33
    def get_hf_weight_chunks(self, megatron_local_weights, progress_desc: str = "Update weights"):
        rank = dist.get_rank()
        for megatron_local_param_infos in tqdm(
            self.megatron_local_param_info_buckets, disable=rank != 0, desc=progress_desc
        ):
            megatron_full_params = _get_megatron_full_params(megatron_local_param_infos, megatron_local_weights)
            hf_named_tensors = self._convert_to_hf_named_tensors(megatron_full_params, megatron_local_param_infos)
            yield hf_named_tensors
            del megatron_full_params
```

**Comment：** 与 distributed 路径差异：Direct 用 `ParamInfo.src_rank` 跨 PP broadcast 本地 shard，再 batch async TP gather。

---

## 11. _get_megatron_local_param_info_buckets

**Explain：** 扫描全部 `ParamInfo`，按 `size × tp_size` 累积分桶，保证单 bucket 不超过 `update_weight_buffer_size`（与 distributed 的 byte 累计语义对齐）。

**Code：**

```python
## 来源：update_weight/hf_weight_iterator_direct.py L116-L133
    for info in param_infos:
        if ".experts." in info.name:
            tp_size = mpu.get_expert_tensor_parallel_world_size()
        else:
            tp_size = mpu.get_tensor_model_parallel_world_size()
        param_size = info.size * tp_size
        if buffer_size + param_size > args.update_weight_buffer_size and len(param_info_buckets[-1]) > 0:
            param_info_buckets.append([])
            buffer_size = 0
        param_info_buckets[-1].append(info)
        buffer_size += param_size
```

**Comment：** `_get_megatron_local_param_infos` 还会 gloo all_gather_object 校验全 rank ParamInfo 一致，防止 silent mismatch。

---

## 12. all_gather_params_async：三阶段 overlap

**Explain：** Phase1 对所有 TP 参数发起 async all_gather；Phase2 统一 wait；Phase3 concat + GLU/MoE 修正。Direct 路径在 `_get_megatron_full_params` 末尾调用。

**Code：**

```python
## 来源：update_weight/common.py L87-L91
    for handle in handles:
        if handle is not None:
            handle.wait()
    gathered_params = []
    for info, direct_param, handle, param_partitions, partition_dim in gather_tasks:
        if handle is None:
            param = direct_param
        else:
            # ... concat partitions ...
            param = torch.cat(param_partitions, dim=partition_dim)
        gathered_params.append(param)
    return gathered_params
```

**Comment：** expert_bias / duplicated / non-TP 参数跳过 gather，直接 append。

---

## 走读小结

| 步骤 | 位置 | 产出 |
|------|------|------|
| 选型 | `actor.init` | `UpdateWeightFromDistributed` 实例 |
| 编排 | `actor.update_weights` | 连引擎、调 updater |
| 命名 | `common.named_params_and_buffers` | global param names |
| 拼片 | `common.all_gather_param` | 完整 Megatron tensor |
| 转换 | `convert_to_hf` | HF names + tensors |
| 传输 | `update_weights_from_distributed` | NCCL broadcast |
| 分桶参考 | `HfWeightIteratorDirect` | buffer_size 语义 |
