---
type: batch-doc
module: 24-WeightSync-Dist
batch: "24"
doc_type: concept
title: "NCCL 权重同步 · 核心概念"
tags:
  - slime/batch/24
  - slime/module/weight-sync-dist
  - slime/doc/concept
updated: 2026-07-02
---

# NCCL 权重同步 · 核心概念

## 用户故事：训练完一轮，推理引擎还是旧权重

### Persona

**小陈**，Post-training 工程师。8 卡 Megatron 训练 + 4 卡 SGLang 分离部署，第一轮 rollout 正常，第二轮 loss 突然爆炸——排查发现 SGLang 仍在用 step 0 权重。他需要理解 `generate → train → update_weights` 闭环里 NCCL 同步何时触发、谁负责 broadcast。

### 时间线

| 时刻 | 事件 |
|------|------|
| T0 | `train.py` 主循环：`actor_model.train()` 更新 Megatron 权重 |
| T1 | `actor_model.update_weights()` 获取可更新引擎列表 |
| T2 | `UpdateWeightFromDistributed.connect_rollout_engines` 建 NCCL 组 |
| T3 | pause → TP/EP gather → HF convert → NCCL broadcast → continue |
| T4 | SGLang `weight_version` 递增，下一轮 rollout 用新权重 |

---

## 1. 四种权重同步路径（本批聚焦 NCCL）

**Explain：** Slime 按 `colocate`、`update_weight_mode`、`update_weight_transport` 三轴选型。本批次覆盖 **分离部署 + full + nccl** 组合。

**Code：**

```python
# 来源：slime/backends/megatron_utils/actor.py L139-L161
        if self.args.colocate:
            assert (
                self.args.update_weight_mode == "full"
            ), "--update-weight-mode=delta is not supported with --colocate"
            update_weight_cls = UpdateWeightFromTensor
        elif self.args.update_weight_mode == "delta":
            assert (
                self.args.update_weight_transport == "disk"
            ), "--update-weight-mode=delta requires --update-weight-transport=disk"
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

| 类 | 场景 | 传输 |
|----|------|------|
| `UpdateWeightFromDistributed` | 分离 + nccl | NCCL broadcast |
| `UpdateWeightFromDisk` | 分离 + disk | 共享文件系统 |
| `UpdateWeightFromDiskDelta` | delta 模式 | disk + diff |
| `UpdateWeightFromTensor` | colocate | CUDA IPC |

---

## 2. PP Source Rank：谁有权广播

**Explain：** 流水线并行下每个 PP stage 只持有部分层。只有 **DP=0 且 TP=0** 的 rank 作为该 PP stage 的 source，创建 `slime-pp_{pp_rank}` NCCL 组并向引擎 broadcast。

**Code：**

```python
# 来源：update_weight/update_weight_from_distributed.py L75-L80
        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        )
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        if self._is_pp_src_rank:
            self._group_name = f"slime-pp_{pp_rank}"
```

**Comment：**

- 非 PP source rank 仍参与 `all_gather_param`（拼完整 TP 分片），但不 yield HF chunk、不发起 broadcast
- 多 PP stage 时每个 stage 有独立 NCCL 组，引擎按 PP 分片接收对应层权重

---

## 3. TP All-Gather：Megatron 分片 → 完整张量

**Explain：** Megatron 线性层按 TP 切分。同步前 `all_gather_param` 在 expert-TP 或 regular-TP 组内 all_gather，并处理 GLU（`linear_fc1`）与 MoE `linear_fc2` 的维度修正。

**Code：**

```python
# 来源：update_weight/common.py L15-L50
def all_gather_param(name: str, param: torch.nn.Parameter) -> torch.Tensor:
    if "expert_bias" in name:
        return param
    if not param.tensor_model_parallel or getattr(param, "parallel_mode", None) == "duplicated":
        return param.data
    if ".experts." in name:
        tp_size = mpu.get_expert_tensor_parallel_world_size()
        tp_group = mpu.get_expert_tensor_parallel_group()
    else:
        tp_size = mpu.get_tensor_model_parallel_world_size()
        tp_group = mpu.get_tensor_model_parallel_group()
    param_partitions = [torch.empty_like(param.data) for _ in range(tp_size)]
    dist.all_gather(param_partitions, param.data, group=tp_group)
    # ... GLU rechunk + linear_fc2 dim fix ...
    param = torch.cat(param_partitions, dim=partition_dim)
    return param
```

**Comment：** `all_gather_params_async` 对多参数批量 async all_gather，供 `HfWeightIteratorDirect` 重叠通信（见 §5）。

---

## 4. Megatron → HF 命名与 convert_to_hf

**Explain：** `named_params_and_buffers` 把 PP/EP/VPP 下的本地名映射为跨 rank 一致的 global name，再交给 `convert_to_hf` 产出 HuggingFace 侧 `(name, tensor)` 列表（可能一对多，如 fused QKV 拆分）。

**Code：**

```python
# 来源：update_weight/common.py L160-L167
def _named_params_and_buffers_global(
    args: Namespace, model: Sequence[torch.nn.Module]
) -> Iterator[tuple[str, torch.Tensor]]:
    """
    Yield (global_name, param/buffer) with consistent names across PP/EP. Adjusts indices for
    virtual PP + EP offsets. Handles decoder.layers, mtp.layers (Multi-Token Prediction), expert_bias.
    """
```

**Comment：** EP rank 对 expert 索引加 `expert_offset`；MTP speculative 层走独立 regex 分支。converter 路由见批次 26 [[26-Checkpoint-M2HF-00-MOC]]。

---

## 5. HfWeightIteratorDirect：分桶迭代器（共享基础设施）

**Explain：** 当 `--megatron-to-hf-mode=raw` 时，`HfWeightIteratorBase.create` 返回 `HfWeightIteratorDirect`。NCCL 路径在 `UpdateWeightFromDistributed` 内联了类似分桶逻辑；Direct 迭代器主要用于 **HF checkpoint 保存** 及统一的分桶/TP gather 实现。

**Code：**

```python
# 来源：update_weight/hf_weight_iterator_base.py L6-L15
    @staticmethod
    def create(args, model, **kwargs):
        from .hf_weight_iterator_bridge import HfWeightIteratorBridge
        from .hf_weight_iterator_direct import HfWeightIteratorDirect
        c = {
            "raw": HfWeightIteratorDirect,
            "bridge": HfWeightIteratorBridge,
        }[args.megatron_to_hf_mode]
        return c(args, model, **kwargs)
```

**Comment：** 理解 Direct 的 `_get_megatron_local_param_info_buckets` 有助于解读 distributed 路径的 `--update-weight-buffer-size` 行为。

---

## 6. NCCL 组拓扑：训练 rank 0 + 所有引擎 GPU

**Explain：** `connect_rollout_engines_from_distributed` 在训练节点 rank 0 与每个 SGLang 引擎占用的 GPU 间建立 world_size = 1 + Σ(engine_gpu_counts) 的 NCCL 组。异构 TP（如 PD 分离 prefill TP≠decode TP）通过 `engine_gpu_counts` 区分各引擎 rank 跨度。

**Code：**

```python
# 来源：update_weight/update_weight_from_distributed.py L288-L311
    world_size = sum(engine_gpu_counts) + 1  # +1 for training rank 0
    cumulative = [0]
    for c in engine_gpu_counts:
        cumulative.append(cumulative[-1] + c)
    refs = [
        engine.init_weights_update_group.remote(
            master_address=master_address,
            master_port=master_port,
            rank_offset=cumulative[i] + 1,
            world_size=world_size,
            group_name=group_name,
            backend="nccl",
        )
        for i, engine in enumerate(rollout_engines)
    ]
    model_update_groups = init_process_group(
        backend="nccl",
        init_method=f"tcp://{_wrap_ipv6(master_address)}:{master_port}",
        world_size=world_size,
        rank=0,
        group_name=group_name,
    )
```

**Comment：** metadata（names/dtypes/shapes）走 Ray RPC；payload 走 NCCL `dist.broadcast` from rank 0。

---

## 概念速查

| 术语 | 含义 |
|------|------|
| `weight_version` | 每次 `update_weights` 递增，引擎侧校验一致性 |
| `rollout_engine_lock` | Ray actor 锁，防止并发 broadcast 死锁 |
| `_iter_non_expert_chunks` / `_iter_expert_chunks` | 非 MoE 与 MoE expert 分两趟同步 |
| `post_process_weights` | compressed-tensors int4/fp4 量化加载前后处理 |
