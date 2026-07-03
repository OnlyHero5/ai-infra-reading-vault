---
type: batch-doc
module: 19-Train-Step
batch: "19"
doc_type: walkthrough
title: "Train Step · 源码走读"
tags:
  - slime/batch/19
  - slime/module/train-step
  - slime/doc/walkthrough
updated: 2026-07-02
---

# Train Step · 源码走读

> 走读顺序：`async_train` → `train` → `_get_rollout_data` → `train_critic` / `train_actor` → `model.train` → `train_one_step`

---

## 1. Ray 层：`RayTrainGroup.async_train`

**Explain：** 每个 Megatron rank 对应一个 Ray actor；`async_train` 对所有 worker 发起 `train.remote`，返回 ref 列表。Critic ref resolve 为 `{"values": ...}` 或 `{}`；Actor ref 为 `None`。

**Code：**

```python
## 来源：slime/ray/actor_group.py L131-L149
    def async_train(self, rollout_id, rollout_data_ref, external_data=None):
        """Do one rollout training. Returns a list of Ray refs (one per worker).

        For critics, each ref resolves to ``{"values": [cpu tensors...]}`` (or ``{}``
        for non-last-PP-stage workers). Actor refs resolve to ``None``.

        ``external_data`` may be a list (one item per worker) or a single dict
        broadcast to all workers.
        """
        if isinstance(external_data, list):
            assert len(external_data) == len(self._actor_handlers)
            return [
                actor.train.remote(rollout_id, rollout_data_ref, external_data=ed)
                for actor, ed in zip(self._actor_handlers, strict=False)
            ]
        return [
            actor.train.remote(rollout_id, rollout_data_ref, external_data=external_data)
            for actor in self._actor_handlers
        ]
```

**Comment：** `external_data` 传 Critic 的 **未 resolve 的 value_refs** 时，Ray 会在 Actor worker 内 lazy fetch；last PP stage 才写入 `rollout_data["values"]`。

---

## 2. 入口：`MegatronTrainRayActor.train`

**Explain：** 统一入口：offload 时 wake → 反序列化 rollout 数据 → 按 role 分派 → offload 时 sleep。`debug_rollout_only` 直接跳过。

**Code：**

```python
## 来源：slime/backends/megatron_utils/actor.py L380-L400
    def train(self, rollout_id: int, rollout_data_ref: Box, external_data=None):
        if self.args.debug_rollout_only:
            return None

        if self.args.offload_train:
            self.wake_up()

        with timer("data_preprocess"):
            rollout_data = self._get_rollout_data(rollout_data_ref)

        if self.role == "critic":
            result = self.train_critic(rollout_id, rollout_data)
        else:
            self.train_actor(rollout_id, rollout_data, external_data=external_data)
            result = None

        if self.args.offload_train:
            del rollout_data
            self.sleep()

        return result
```

**Comment：** Critic 返回 dict 供 Ray 传回；Actor 返回 `None`。offload 路径在 train 前后配对 `wake_up`/`sleep`，与 rollout GPU 分时复用。

---

## 3. 数据预处理：`_get_rollout_data`

**Explain：** 通过 `process_rollout_data` 按 DP rank 切片；tokens / loss_masks / log_probs 提前 `.to(cuda)`；CP 场景对 log_prob 做 `slice_log_prob_with_cp`。

**Code：**

```python
## 来源：slime/backends/megatron_utils/actor.py L222-L276（节选）
    def _get_rollout_data(self, rollout_data_ref: Box) -> RolloutBatch:
        rollout_data = process_rollout_data(
            self.args,
            rollout_data_ref,
            mpu.get_data_parallel_rank(with_context_parallel=False),
            mpu.get_data_parallel_world_size(with_context_parallel=False),
        )
        device = torch.cuda.current_device()
        rollout_data["tokens"] = [
            t.to(device=device, dtype=torch.long, non_blocking=True) for t in rollout_data["tokens"]
        ]
        rollout_data["loss_masks"] = [
            t.to(device=device, dtype=torch.int, non_blocking=True) for t in rollout_data["loss_masks"]
        ]
        for key in ["rollout_log_probs", "teacher_log_probs"]:
            if key not in rollout_data:
                continue
            rollout_data[key] = [
                slice_log_prob_with_cp(log_prob, total_length, response_length).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                for log_prob, total_length, response_length in zip(
                    rollout_data[key],
                    rollout_data["total_lengths"],
                    rollout_data["response_lengths"],
                    strict=False,
                )
            ]
        return rollout_data
```

**Comment：** [[20-Train-Data-00-MOC]] 详述 `process_rollout_data` 如何构造 `num_microbatches` / `global_batch_sizes`。

---

## 4. Critic 路径：`train_critic`

**Explain：** 顺序固定：value forward → advantage → 临时改 `loss_type=value_loss` → `model.train` → last stage 把 values 转 CPU 返回。

**Code：**

```python
## 来源：slime/backends/megatron_utils/actor.py L402-L428
    def train_critic(self, rollout_id: int, rollout_data: RolloutBatch):
        data_iterator = get_data_iterator(rollout_data)
        num_microbatches = rollout_data["num_microbatches"]
        global_batch_sizes = rollout_data["global_batch_sizes"]

        rollout_data.update(forward_only(get_values, self.args, self.model, data_iterator, num_microbatches))

        compute_advantages_and_returns(self.args, rollout_data)

        self.args.loss_type = "value_loss"
        train(
            rollout_id,
            self.model,
            self.optimizer,
            self.opt_param_scheduler,
            data_iterator,
            num_microbatches,
            global_batch_sizes,
        )

        if mpu.is_pipeline_last_stage() and "values" in rollout_data:
            from slime.backends.megatron_utils.data import tensors_to_cpu

            return {"values": tensors_to_cpu(rollout_data["values"])}
        return {}
```

**Comment：** Critic 在 **同一 rollout_data** 上先算 values 再算 advantage，Actor 训练时复用 Critic 产出的 values 作为 old values（经 external_data 注入）。

---

## 5. Actor 路径：`train_actor`（前半：多模型 forward）

**Explain：** Actor 训练是状态机：ref → teacher → old_actor/actor log-prob → 注入 critic values → advantage → policy train。`weights_backuper` 切换权重 tag，routing replay 通过环境变量控制阶段。

**Code：**

```python
## 来源：slime/backends/megatron_utils/actor.py L430-L509（节选）
    def train_actor(self, rollout_id: int, rollout_data: RolloutBatch, external_data=None) -> None:
        data_iterator = get_data_iterator(rollout_data)
        num_microbatches = rollout_data["num_microbatches"]
        global_batch_sizes = rollout_data["global_batch_sizes"]

        if self.args.use_rollout_routing_replay:
            self.fill_routing_replay(data_iterator, num_microbatches, rollout_data)

        with inverse_timer("train_wait"), timer("train"):
            if self.args.compute_advantages_and_returns:
                if "ref" in self.weights_backuper.backup_tags:
                    self._switch_model("ref")
                    rollout_data.update(
                        self.compute_log_prob(data_iterator, num_microbatches, store_prefix="ref_")
                    )

                if "teacher" in self.weights_backuper.backup_tags:
                    self._switch_model("teacher")
                    rollout_data.update(
                        self.compute_log_prob(data_iterator, num_microbatches, store_prefix="teacher_")
                    )

                self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
                # ... log_prob forward（见 01-核心概念）...

                if self.args.use_critic:
                    if external_data is not None and mpu.is_pipeline_last_stage():
                        values = external_data.get("values")
                        if values is not None:
                            from slime.backends.megatron_utils.data import tensors_to_gpu

                            rollout_data["values"] = tensors_to_gpu(values)

                if self._active_model_tag != "actor":
                    self._switch_model("actor")

                compute_advantages_and_returns(self.args, rollout_data)
```

**Comment：** `compute_advantages_and_returns` 在 **整批 rollout** 上归一化 advantage（如 `--normalize-advantages`），必须在 microbatch 训练前完成。

---

## 6. Actor 路径：`train_actor`（后半：backward + 收尾）

**Explain：** 调用 `model.train` 做 Megatron backward；备份 actor 权重；按 interval 更新 ref；记录 perf 与 debug dump。

**Code：**

```python
## 来源：slime/backends/megatron_utils/actor.py L511-L555
            if self.rollout_data_postprocess is not None:
                self.rollout_data_postprocess(self.args, rollout_id, rollout_data)

            log_rollout_data(rollout_id, self.args, rollout_data)

            if self.args.use_routing_replay:
                os.environ["ROUTING_REPLAY_STAGE"] = "replay_backward"
            with timer("actor_train"):
                train(
                    rollout_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    data_iterator,
                    num_microbatches,
                    global_batch_sizes,
                )

            self.prof.step(rollout_id=rollout_id)

        train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        if self.args.use_routing_replay:
            RoutingReplay.clear_all()

        self.weights_backuper.backup("actor")

        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and "ref" in self.weights_backuper.backup_tags
        ):
            self.weights_backuper.backup("ref")

        log_perf_data(rollout_id, self.args, extra_metrics=self.weight_updater.pop_metrics())
```

**Comment：** train 结束后 **必须** `weights_backuper.backup("actor")`，保证后续 `update_weights` 读到最新 actor 参数。

---

## 7. `model.train`：多 step 循环

**Explain：** 重置 iterator、设 train 模式、配置 Megatron overlap grad/ param gather，然后对 `num_microbatches` 每个元素调用一次 `train_one_step`。

**Code：**

```python
## 来源：slime/backends/megatron_utils/model.py L732-L744, L810-L835
    args = get_args()

    assert len(num_microbatches) == len(global_batch_sizes), (
        f"num_microbatches and global_batch_sizes must have the same length, "
        f"got {len(num_microbatches)} vs {len(global_batch_sizes)}"
    )

    for iterator in data_iterator:
        iterator.reset()

    for model_module in model:
        model_module.train()

    num_steps_per_rollout = len(num_microbatches)
    # ... overlap / pre-hook 配置 ...

    for step_id in range(num_steps_per_rollout):
        loss_dict, grad_norm = train_one_step(
            args,
            rollout_id,
            step_id,
            data_iterator,
            model,
            optimizer,
            opt_param_scheduler,
            num_microbatches[step_id],
            global_batch_sizes[step_id],
            microbatch_pbar=microbatch_pbar,
        )
```

**Comment：** 动态 batch 时 `num_steps_per_rollout > 1`，每 step 的 `global_batch_sizes[step_id]` 可能不同。

---

## 8. `train_one_step`：Megatron forward-backward 核心

**Explain：** 单 step 流程：zero grad →（可选 hook）→ 定义 `forward_step` 闭包（内部 `get_batch` + `loss_function`）→ `forward_backward_func` → NaN 检查 → `optimizer.step` → `opt_param_scheduler.step(increment=step_global_batch_size)`。

**Code：**

```python
## 来源：slime/backends/megatron_utils/model.py L549-L552, L576-L638, L640-L680
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    def forward_step(data_iterator: DataIterator, model: GPTModel, return_schedule_plan: bool = False):
        batch = get_batch(
            data_iterator,
            _with_rollout_top_p_token_keys(
                args,
                [
                    "tokens",
                    "multimodal_train_inputs",
                    "packed_seq_params",
                    "total_lengths",
                    "response_lengths",
                    "loss_masks",
                    "log_probs",
                    "ref_log_probs",
                    "values",
                    "advantages",
                    "returns",
                    "rollout_log_probs",
                    "teacher_log_probs",
                    "rollout_mask_sums",
                ],
            ),
            args.data_pad_size_multiplier,
            args.allgather_cp,
        )
        output_tensor = model(
            input_ids=batch["tokens"],
            position_ids=None,
            attention_mask=None,
            labels=None,
            packed_seq_params=batch["packed_seq_params"],
            loss_mask=batch["full_loss_masks"],
            **(batch["multimodal_train_inputs"] or {}),
        )
        return output_tensor, partial(loss_function, args, batch, num_microbatches, step_global_batch_size)

    forward_backward_func = get_forward_backward_func()
    losses_reduced = forward_backward_func(
        forward_step_func=_wrap_forward_step_with_microbatch_pbar(forward_step, microbatch_pbar),
        data_iterator=data_iterator,
        model=model,
        num_microbatches=num_microbatches,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=False,
    )

    if valid_step:
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
        assert update_successful
        opt_param_scheduler.step(increment=step_global_batch_size)
```

**Comment：** `loss_function` 根据 `args.loss_type`（policy / value / sft）分支；PPO clip、KL 在[[22-Loss-Policy-00-MOC]]。PP last stage 通过 `reduce_train_step_metrics` 聚合 loss 写 wandb。

---

## 9. 日志与 CI 断言

**Explain：** 仅 DP0 + TP0 + last PP rank 打日志；CI 检查 PPO 初始 KL、grad norm 快照等。

**Code：**

```python
## 来源：slime/backends/megatron_utils/model.py L892-L907
            if args.ci_test and "train/train_rollout_logprob_abs_diff" in log_dict:
                assert log_dict["train/train_rollout_logprob_abs_diff"] <= 0.1, f"{log_dict=}"

            if args.ci_test and not args.ci_disable_kl_checker:
                if step_id == 0 and "train/ppo_kl" in log_dict and "train/pg_clipfrac" in log_dict:
                    assert log_dict["train/ppo_kl"] < 1e-8, f"{log_dict=}"
                if (
                    accumulated_step_id == 0
                    and not getattr(args, "use_rollout_routing_replay", False)
                    and "train/kl_loss" in log_dict
                ):
                    assert log_dict["train/kl_loss"] < 1e-8, f"{log_dict=}"
```

**Comment：** `test_qwen3_4B_ppo.py` 传 `--ci-test`，跑 2 个 rollout 验证此路径可通。

---

## 10. 走读小结

| 层级 | 关键函数 | 输出 |
|------|----------|------|
| Ray | `async_train` | ObjectRef 列表 |
| Actor | `train` | Critic: `{values}` / Actor: `None` |
| Actor | `train_actor` | 更新 GPU 上 actor 权重 + wandb |
| Model | `train` | 多 step Megatron 训练 |
| Model | `train_one_step` | loss_dict, grad_norm |
