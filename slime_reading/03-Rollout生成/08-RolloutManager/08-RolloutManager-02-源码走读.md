---
type: batch-doc
module: 08-RolloutManager
batch: "08"
doc_type: walkthrough
title: "RolloutManager · 源码走读"
tags:
  - slime/batch/08
  - slime/module/rollout-manager
  - slime/doc/walkthrough
updated: 2026-07-02
---

# RolloutManager · 源码走读

> 走读顺序：`__init__` → `generate` → `_get_rollout_data` → `_convert_samples_to_train_data` → `_split_train_data_by_dp` → `_tensorize_rollout_data_for_training` → `get_updatable_engines_and_lock`

---

## 1. 类定义与模块级辅助

### 1.1 tensor 化 dtype 表

**Explain：** `_ROLLOUT_DATA_TENSOR_DTYPES` 定义哪些 train_data 字段在 DP 分片后转为 CPU tensor；`rollout_routed_experts` 保持原 dtype（MoE replay）。

**Code：**

```python
# 来源：slime/ray/rollout.py L39-L47, L73-L78
_ROLLOUT_DATA_TENSOR_DTYPES = {
    "tokens": torch.long,
    "loss_masks": torch.int,
    "rollout_log_probs": torch.float32,
    "rollout_top_p_token_ids": torch.int32,
    "rollout_top_p_token_offsets": torch.int32,
    "teacher_log_probs": torch.float32,
    "rollout_routed_experts": None,
}

def _cpu_tensor(value, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(value, np.ndarray) and not value.flags.writeable:
        value = value.copy()
    tensor = torch.as_tensor(value, dtype=dtype) if dtype is not None else torch.as_tensor(value)
    return tensor.detach().cpu().contiguous()
```

**Comment：** 在 `ray.put` 前统一 CPU contiguous，避免 GPU tensor 跨节点 Object Store 传输问题；nixl transport 仍走同一 tensor 化路径。

---

## 2. RolloutManager.__init__

### 2.1 引擎启动与 data_source 加载

**Explain：** 非 `debug_train_only` 时调用 `start_rollout_servers` 创建多模型 SGLang 集群，并 `ray.get` 等待 init 完成；随后加载 data_source 与 rollout/eval 函数。

**Code：**

```python
# 来源：slime/ray/rollout.py L420-L471
@ray.remote
class RolloutManager:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, args, pg):
        configure_logger()

        self.pg = pg
        self.args = args

        rollout_init_handles: list[Any] = []
        if self.args.debug_train_only:
            self.servers: dict[str, Any] = {}
        else:
            init_http_client(args)
            self.servers, rollout_init_handles = start_rollout_servers(args, pg)

        data_source_cls = load_function(self.args.data_source_path)
        self.data_source = data_source_cls(args)

        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        # ... custom_reward / custom_convert hooks ...

        if rollout_init_handles:
            ray.get(rollout_init_handles)

        init_tracking(args, primary=False)
        self.rollout_engine_lock = Lock.options(
            num_cpus=1,
            num_gpus=0,
            runtime_env={"env_vars": add_default_ray_env_vars()},
        ).remote()
        self.rollout_id = -1

        self._health_monitors = []
        if not self.args.debug_train_only and self.args.use_fault_tolerance:
            for srv in self.servers.values():
                for group in srv.server_groups:
                    monitor = RolloutHealthMonitor(group, args)
                    monitor.start()
                    self._health_monitors.append(monitor)
```

**Comment：**

- `self.servers` 是 `dict[model_name, RolloutServer]`，支持多模型（policy + reference）
- `rollout_engine_lock` 是独立 Ray Actor，供 `update_weights` 互斥
- fault tolerance 时 `RolloutHealthMonitor` 后台检测死引擎并置 `None`

---

## 3. generate(rollout_id)

**Explain：** 训练主循环唯一入口。顺序：恢复 health monitor → 取 rollout 数据 → 可选 debug 落盘 → 日志 → 转 train_data → DP 分片 + ObjectRef。

**Code：**

```python
# 来源：slime/ray/rollout.py L546-L559
    def generate(self, rollout_id):
        start_time = time.time()
        self.rollout_id = rollout_id
        self.health_monitoring_resume()
        if self.args.ci_test and self.args.use_fault_tolerance and rollout_id >= 2:
            self._try_ci_fault_injection()
        data, metrics = self._get_rollout_data(rollout_id=rollout_id)
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=False)
        _log_rollout_data(rollout_id, self.args, data, metrics, time.time() - start_time)
        if self.args.debug_rollout_only:
            return
        data = self._convert_samples_to_train_data(data)
        return self._split_train_data_by_dp(data)
```

**Comment：**

- `debug_rollout_only` 用于只测推理/RM，不进入训练管线
- 返回值 `list[Box]` 长度 = `train_parallel_config["dp_size"]`；`train.py` 整包传给 `async_train`
- `set_train_parallel_config` 在 Actor init 时由 Megatron 侧调用（本批不展开）

---

## 4. _get_rollout_data

### 4.1 debug 加载 vs 正常 rollout

**Explain：** 支持从磁盘加载预采样的 Sample（`load_debug_rollout_data`）；正常路径通过 `call_rollout_fn` 调用用户 rollout 函数，并校验/展平嵌套 list。

**Code：**

```python
# 来源：slime/ray/rollout.py L635-L665
    def _get_rollout_data(self, rollout_id):
        if self.args.load_debug_rollout_data:
            data = torch.load(
                self.args.load_debug_rollout_data.format(rollout_id=rollout_id),
                weights_only=False,
            )["samples"]
            data = [Sample.from_dict(sample) for sample in data]
            if (ratio := self.args.load_debug_rollout_data_subsample) is not None:
                original_num_rows = len(data)
                rough_subsample_num_rows = int(original_num_rows * ratio)
                data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
            metrics = None
        else:
            data = call_rollout_fn(self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False)
            metrics = data.metrics
            data = data.samples
            _validate_rollout_id_annotated(data)
            while isinstance(data[0], list):
                data = list(itertools.chain.from_iterable(data))

        return data, metrics
```

**Comment：**

- `call_rollout_fn` 兼容 legacy 返回裸 list（见 `base_types.py`）
- 默认 rollout 输出形状 `list[list[Sample]]`（prompt × n_samples），展平后 `list[Sample]`
- `_validate_rollout_id_annotated` 仅在 compact 模式（depth≥2 的 `list[Sample]`）强制 `rollout_id` 一致

### 4.2 call_rollout_fn 包装

**Code：**

```python
# 来源：slime/rollout/base_types.py L7-L26
@dataclass
class RolloutFnTrainOutput:
    samples: list[list[Sample]]
    metrics: dict[str, Any] = None

def call_rollout_fn(fn, *args, evaluation: bool, **kwargs):
    output = fn(*args, **kwargs, evaluation=evaluation)
    if not isinstance(output, (RolloutFnTrainOutput, RolloutFnEvalOutput)):
        output = RolloutFnEvalOutput(data=output) if evaluation else RolloutFnTrainOutput(samples=output)
    return output
```

---

## 5. _convert_samples_to_train_data

### 5.1 主 dict 构建

**Explain：** 将 Sample 列表转为 **列式 dict**（每个 key 对应等长 list）。核心字段：tokens、rewards、loss_masks、rollout_ids；可选字段按首 Sample 探测。

**Code：**

```python
# 来源：slime/ray/rollout.py L713-L761
    def _convert_samples_to_train_data(self, samples: list[Sample] | list[list[Sample]]):
        if self.custom_convert_samples_to_train_data_func is not None:
            return self.custom_convert_samples_to_train_data_func(self.args, samples)

        raw_rewards, rewards = self._post_process_rewards(samples)

        rollout_ids = [sample.rollout_id for sample in samples]
        existed_rollout_id_values = set(rid for rid in rollout_ids if rid is not None)
        tmp_id = 0
        for i in range(len(rollout_ids)):
            if rollout_ids[i] is None:
                while tmp_id in existed_rollout_id_values:
                    tmp_id += 1
                rollout_ids[i] = tmp_id
                existed_rollout_id_values.add(tmp_id)

        train_data = {
            "tokens": [sample.tokens for sample in samples],
            "response_lengths": [sample.response_length for sample in samples],
            "rewards": rewards,
            "raw_reward": raw_rewards,
            "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
            "sample_indices": [sample.index for sample in samples],
            "rollout_ids": rollout_ids,
        }

        loss_masks = []
        for sample in samples:
            if sample.loss_mask is None:
                sample.loss_mask = [1] * sample.response_length
            assert len(sample.loss_mask) == sample.response_length
            if sample.remove_sample:
                sample.loss_mask = [0] * sample.response_length
            loss_masks.append(sample.loss_mask)
        train_data["loss_masks"] = loss_masks
```

**Comment：**

- `rollout_id is None` 时自动分配唯一 id，保证 DP schedule 可按 rollout 分组
- `remove_sample=True` 时 loss_mask 全 0，样本仍保留但不参与 loss
- `rewards` 可能经过 GRPO group normalization（见 §5.2）

### 5.2 rollout_mask_sums 与 GRPO reward 后处理

**Code：**

```python
# 来源：slime/ray/rollout.py L686-L711, L763-L778
    def _post_process_rewards(self, samples: list[Sample] | list[list[Sample]]):
        if self.custom_reward_post_process_func is not None:
            return self.custom_reward_post_process_func(self.args, samples)

        raw_rewards = [sample.get_reward_value(self.args) for sample in samples]
        if (
            self.args.advantage_estimator in ["grpo", "gspo", "cispo", "reinforce_plus_plus_baseline"]
            and self.args.rewards_normalization
        ):
            rewards = torch.tensor(raw_rewards, dtype=torch.float)
            if rewards.shape[-1] == self.args.n_samples_per_prompt * self.args.rollout_batch_size:
                rewards = rewards.reshape(-1, self.args.n_samples_per_prompt)
            else:
                rewards = rewards.view(-1, rewards.shape[-1])
            mean = rewards.mean(dim=-1, keepdim=True)
            rewards = rewards - mean
            if self.args.advantage_estimator in ["grpo", "gspo", "cispo"] and self.args.grpo_std_normalization:
                std = rewards.std(dim=-1, keepdim=True)
                rewards = rewards / (std + 1e-6)
            return raw_rewards, rewards.flatten().tolist()
        return raw_rewards, raw_rewards

        # rollout_mask_sums（在 train_data 构建中）:
        rollout_id_list = train_data["rollout_ids"]
        mask_sums_per_sample = [sum(m) for m in loss_masks]
        rollout_total_mask: dict[int, int] = {}
        for rid, ms in zip(rollout_id_list, mask_sums_per_sample, strict=True):
            rollout_total_mask[rid] = rollout_total_mask.get(rid, 0) + ms
        train_data["rollout_mask_sums"] = [rollout_total_mask[rid] for rid in rollout_id_list]
```

**Comment：** `rollout_mask_sums` 解决 **同一 rollout 的 sample 被 first-fit 分到不同 micro-batch** 时，loss reducer 仍用整 rollout 的 mask 总和作分母。

### 5.3 可选字段探测

**Code：**

```python
# 来源：slime/ray/rollout.py L792-L823
        if samples[0].rollout_log_probs is not None:
            train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

        if getattr(self.args, "rollout_top_p", 1.0) != 1.0:
            for sample in samples:
                assert sample.rollout_top_p_token_ids is not None
                assert sample.rollout_top_p_token_offsets is not None
                assert len(sample.rollout_top_p_token_offsets) == sample.response_length + 1
            train_data["rollout_top_p_token_ids"] = [sample.rollout_top_p_token_ids for sample in samples]
            train_data["rollout_top_p_token_offsets"] = [sample.rollout_top_p_token_offsets for sample in samples]

        if samples[0].rollout_routed_experts is not None:
            train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]

        if samples[0].teacher_log_probs is not None:
            train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]
```

---

## 6. _split_train_data_by_dp

**Explain：** 调用纯 Python 的 `build_dp_schedule` 计算 DP partition 与 micro-batch 索引，再为每个 rank 组装子 dict、tensor 化、`ray.put`。

**Code：**

```python
# 来源：slime/ray/rollout.py L829-L895
    def _split_train_data_by_dp(self, data):
        dp_size = self.train_parallel_config["dp_size"]
        total_lengths = [len(t) for t in data["tokens"]]
        data["total_lengths"] = total_lengths

        partitions, micro_batch_indices, num_microbatches, global_batch_sizes = build_dp_schedule(
            self.args,
            self.train_parallel_config,
            total_lengths,
            global_batch_size=self.args.global_batch_size,
            rollout_indices=data["rollout_ids"],
        )

        rollout_data_refs = []
        for r in range(dp_size):
            partition = partitions[r]
            rollout_data = {"partition": partition}
            for key in [
                "tokens", "multimodal_train_inputs", "response_lengths", "rewards",
                "truncated", "loss_masks", "round_number", "sample_indices",
                "rollout_ids", "rollout_mask_sums", "rollout_log_probs",
                "rollout_top_p_token_ids", "rollout_top_p_token_offsets",
                "rollout_routed_experts", "prompt", "teacher_log_probs",
            ]:
                if key not in data:
                    continue
                rollout_data[key] = [data[key][j] for j in partition]
            for key in ["raw_reward", "total_lengths"]:
                if key not in data:
                    continue
                rollout_data[key] = data[key]
            rollout_data["global_batch_sizes"] = global_batch_sizes
            rollout_data["num_microbatches"] = num_microbatches
            rollout_data["micro_batch_indices"] = micro_batch_indices[r]
            _tensorize_rollout_data_for_training(rollout_data)
            transport = getattr(self.args, "rollout_data_transport", "object-store")
            if transport == "nixl":
                rollout_data_refs.append(Box(ray.put(rollout_data, _tensor_transport="nixl")))
            elif transport == "object-store":
                rollout_data_refs.append(Box(ray.put(rollout_data)))
            else:
                raise ValueError(f"Unsupported rollout data transport: {transport!r}")
        return rollout_data_refs
```

**Comment：**

- `partition` 存 **全局 sample 下标**；rank 内 `micro_batch_indices` 再切 micro-batch
- `raw_reward` / `total_lengths` **不按 rank 切分**，训练侧自行索引
- `global_batch_sizes[s]` = 第 s 个 training step 的 rollout 数（通常恒为 `args.global_batch_size`）

### 6.1 build_dp_schedule 入口签名

**Code：**

```python
# 来源：slime/utils/dp_schedule.py L82-L111
def build_dp_schedule(
    args: Any,
    train_parallel_config: dict,
    total_lengths: list[int],
    *,
    global_batch_size: int,
    rollout_indices: list[int],
) -> tuple[list[list[int]], list[list[list[int]]], list[int], list[int]]:
    """Compute the per-rank DP partition and micro-batch schedule.

    Args:
        global_batch_size: number of rollouts (NOT training samples) per training step.
        rollout_indices: rollout id for each sample. Samples sharing the same id
            are kept together in one step.
    Returns:
        (partitions, micro_batch_indices, num_microbatches, global_batch_sizes)
    """
```

---

## 7. _tensorize_rollout_data_for_training

**Explain：** 就地修改 `rollout_data`，把 list 字段变为 `list[torch.Tensor]`（每条 sample 一个 tensor）。

**Code：**

```python
# 来源：slime/ray/rollout.py L80-L102
def _tensorize_rollout_data_for_training(rollout_data: dict[str, Any]) -> None:
    for key, dtype in _ROLLOUT_DATA_TENSOR_DTYPES.items():
        if key in rollout_data:
            rollout_data[key] = [_cpu_tensor(value, dtype=dtype) for value in rollout_data[key]]

    if "multimodal_train_inputs" in rollout_data:
        rollout_data["multimodal_train_inputs"] = [
            (
                {
                    key: _cpu_tensor(value) if isinstance(value, (np.ndarray, torch.Tensor)) else value
                    for key, value in mm_dict.items()
                }
                if mm_dict is not None
                else None
            )
            for mm_dict in rollout_data["multimodal_train_inputs"]
        ]

    if "rollout_mask_sums" in rollout_data:
        rollout_data["rollout_mask_sums"] = _cpu_tensor(
            rollout_data["rollout_mask_sums"],
            dtype=torch.float32,
        )
```

**Comment：** `rewards` / `response_lengths` 等保持 Python list，由 Megatron 训练侧按需转 tensor（减少 Object Store 序列化开销）。

---

## 8. get_updatable_engines_and_lock

**Explain：** 权重更新专用 API；Megatron `actor.py` 在 `update_weights` 中 `ray.get` 此 remote 方法。

**Code：**

```python
# 来源：slime/ray/rollout.py L527-L540
    def get_updatable_engines_and_lock(self):
        srv = self._get_updatable_server()
        engines = srv.engines if srv else []
        gpu_counts = srv.engine_gpu_counts if srv else []
        gpu_offsets = srv.engine_gpu_offsets if srv else []
        num_new = srv.num_new_engines if srv else 0
        all_engine_actors = srv.all_engines if srv else []
        return engines, self.rollout_engine_lock, num_new, gpu_counts, gpu_offsets, all_engine_actors
```

**Comment：**

- `engines` 仅 node-0 引擎（多节点 serving 时其他 node 由 NCCL 组内同步）
- `gpu_counts` / `gpu_offsets` 供权重分片映射到正确 GPU bundle
- fault tolerance 路径用 `recover_updatable_engines` 替代（L595–616）

---

## 9. 走读小结

| 函数 | 输入 | 输出 | 关键副作用 |
|------|------|------|-----------|
| `__init__` | args, pg | — | 启动 servers、lock、health monitor |
| `generate` | rollout_id | `list[Box]` | 写 rollout_id、日志 |
| `_get_rollout_data` | rollout_id | Sample list | 调用 rollout fn |
| `_convert_samples_to_train_data` | Sample list | dict | reward norm、mask |
| `_split_train_data_by_dp` | dict | ObjectRef list | ray.put |
| `_tensorize_rollout_data_for_training` | dict | — | 就地 CPU tensor |
| `get_updatable_engines_and_lock` | — | 6-tuple | 只读查询 |

下一步：[[08-RolloutManager-03-数据流与交互]] 用一张总图串起 Sample → ObjectRef。
