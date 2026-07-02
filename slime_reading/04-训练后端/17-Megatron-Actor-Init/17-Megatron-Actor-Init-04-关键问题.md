---
type: batch-doc
module: 17-Megatron-Actor-Init
batch: "17"
doc_type: faq
title: "Megatron Actor 初始化 · 关键问题"
tags:
  - slime/batch/17
  - slime/module/megatron-actor-init
  - slime/doc/faq
updated: 2026-07-02
---

# Megatron Actor 初始化 · 关键问题

---

## Q1：`debug_rollout_only` 与 `debug_train_only` 有何区别？

| 模式 | Megatron init | Rollout / SGLang | 典型用途 |
|------|---------------|------------------|----------|
| `debug_rollout_only` | 跳过（返回 0） | 正常运行 | 只调试 generate / reward |
| `debug_train_only` | 正常 | 跳过 SGLang | 只调试 train / loss |

**Explain：** arguments 解析阶段禁止二者同时为 true。

**Code：**

```python
# 来源：slime/utils/arguments.py L1881-L1882
assert not (args.debug_rollout_only and args.debug_train_only), (
    "debug_rollout_only and debug_train_only cannot be set at the same time, "
    "please set only one of them."
)
```

**易错写法：** 以为 `debug_rollout_only` 仍会加载 Megatron 权重 —— **不会**，`init` 第一行即 return。

**正确理解：** 需要验证 Megatron checkpoint 加载应使用正常模式或 `debug_train_only`。

---

## Q2：为何 colocate 不能用 delta weight sync？

**Explain：** colocate 下训练与推理共享 GPU，权重通过 **GPU tensor 直传**（`UpdateWeightFromTensor`）；delta 模式依赖 disk 发布增量，与 colocate 内存布局假设冲突。

**Code：**

```python
# 来源：slime/backends/megatron_utils/actor.py L139-L143
if self.args.colocate:
    assert (
        self.args.update_weight_mode == "full"
    ), "--update-weight-mode=delta is not supported with --colocate"
    update_weight_cls = UpdateWeightFromTensor
```

**Comment：** delta 另需 `update_weight_transport == "disk"`，见 [[25-WeightSync-Disk]]。

---

## Q3：critic 为何没有 `weight_updater`？

Critic 只参与 value 训练，不向 SGLang 推 actor 权重；init 在 `role=="critic"` 时提前 return，不实例化 `TensorBackuper` / `weight_updater`（actor 专属）。

若 `use_critic` + `offload_train` + 非 colocate，**actor** 在 sleep 时会 `disconnect_rollout_engines`，避免 critic 占用期间 NCCL 连接悬挂。

---

## Q4：`start_rollout_id` 从哪来、为何要一致？

- 来源：`load_checkpoint` 返回的 `loaded_rollout_id + 1`
- 所有 Megatron rank 必须返回相同值，否则 `create_training_models` assert 失败
- 有 critic 时以 **critic** 的 init 返回值为准设置 `args.start_rollout_id`

**Explain：** 若手动指定 `--start-rollout-id`，需与 checkpoint 语义一致，避免 rollout 计数与 ckpt 步数错位。

---

## Q5：init 里为何调用两次 dist 初始化？

| 调用 | 位置 | 作用 |
|------|------|------|
| `dist.init_process_group` | `TrainRayActor.init` | PyTorch 全局 process group |
| `mpu.initialize_model_parallel` | `initialize._initialize_distributed` | TP/PP/DP/CP/EP 子组 |

二者 **不是重复**：Megatron 在 world group 之上切分 model-parallel 通信器。offload 时 `destroy_process_groups` / `reload_process_groups` 针对的是 patched 后的组集合。

---

## Q6：`vocab_size` 为何优先 HF config？

部分模型（如 GPT-OSS）tokenizer vocab 小于 model native padding vocab；用错会导致 embedding 越界或 loss mask 异常。

**Code：**

```python
# 来源：slime/backends/megatron_utils/actor.py L133-L137
if self.args.vocab_size is None:
    hf_vocab = getattr(self.hf_config, "vocab_size", None)
    self.args.vocab_size = hf_vocab if hf_vocab is not None else self.tokenizer.vocab_size
```

---

## Q7：offload 失败常见原因

1. **未安装 torch_memory_saver** 或 `LD_PRELOAD` 的 `.so` 找不到 → Actor 创建时即报错（见 actor_group）
2. **debug_rollout_only** 下设置 `train_memory_margin_bytes` → arguments 强制归零并 warning
3. sleep 后仍持有 CUDA tensor 引用 → `clear_memory(clear_host_memory=True)` 仍不足，需检查用户 hook

**对比：**

```python
# debug 模式 — arguments.py
if args.debug_rollout_only:
    args.offload_train = args.offload_rollout = False
    logger.warning("Force train_memory_margin_bytes=0 since debug_rollout_only does not support it")
```

---

## Q8：`custom_megatron_init_path` 何时用？

在标准 Megatron init（并行组、种子、microbatch 计算器）**之后**执行用户函数，适合：

- 注册额外 metric buffer
- 对特定模型做 one-time patch
- 与 Megatron server 模式共享初始化逻辑

**Code：** 见 [[17-Megatron-Actor-Init-02-源码走读]] §12。

---

## Q9：init 与第一次 `update_weights` 的关系

`train.py` 在 enter 主循环 **前** 调用 `actor_model.update_weights()`，将 init 加载的权重推到 SGLang。init 本身 **不** 连接 rollout engines；`rollout_engines` 在 `update_weights` 内通过 `rollout_manager.get_updatable_engines_and_lock` 获取。

若 `debug_rollout_only`，`update_weights` 直接 return，无需引擎。

---

## Q10：与 Megatron server（TeacherLogp）的差异

`TeacherLogpRayActor` 继承 `MegatronTrainRayActor`，但 megatron server 入口强制 `debug_train_only=True`、`offload_train=False`（见 `megatron_utils/server/arguments.py`）。即 **无 sleep/wake 环**，init 路径相同但运行时假设不同。

---

## 对比表：三种 GPU 模式下的 init 行为

| 配置 | init 末尾 | weight_updater |
|------|-----------|----------------|
| 默认（无 offload） | `clear_memory` only | disk 或 nccl |
| `offload_train` | `sleep()` | 同左；train 前 wake |
| `colocate` | 同 offload 或默认 | **仅** UpdateWeightFromTensor |
| `debug_rollout_only` | 仅保存 args | 无（未创建） |
