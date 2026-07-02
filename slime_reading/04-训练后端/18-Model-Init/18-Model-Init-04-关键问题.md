---
type: batch-doc
module: 18-Model-Init
batch: "18"
doc_type: faq
title: "Model 初始化 · 关键问题"
tags:
  - slime/batch/18
  - slime/module/model-init
  - slime/doc/faq
updated: 2026-07-02
---

# Model 初始化 · 关键问题

---

## Q1：load 与 pretrained_checkpoint 必须二选一吗？

**Explain：** `setup_model_and_optimizer` assert 至少有一个非 None；实际 load 路径由 Megatron `load_checkpoint` 解析。

**Code：**

```python
# 来源：slime/backends/megatron_utils/model.py L291-L292
# 提交版本：22cdc6e1
assert not args.moe_use_upcycling
assert args.load is not None or args.pretrained_checkpoint is not None
```

**Comment：**

- torch_dist ref-load 来自 [[05-Tools-DataPrep]]
- Bridge 模式可能 `--load` 指向 HF 同步后的 ckpt

---

## Q2：Bridge 与 torch_dist convert 如何选？

**Explain：** Bridge：`megatron_to_hf_mode==bridge"`，运行时从 HF 构建 Megatron 结构。Legacy：先 convert 脚本产 torch_dist，再 `--load`。

**Code：**

```python
# 来源：slime/backends/megatron_utils/model_provider.py L87-L93
# 提交版本：22cdc6e1
if args.megatron_to_hf_mode == "bridge":
    bridge = patch_auto_bridge_hf_config(AutoBridge.from_hf_pretrained(args.hf_checkpoint, ...))
    provider = bridge.to_megatron_provider(load_weights=False)
```

---

## Q3：critic 从 actor checkpoint 启动 value 头不对怎么办？

**Explain：** 自动检测 output_layer shape mismatch，reinit 并 warning；不 silent 使用错误权重。

**Code：**

```python
# 来源：slime/backends/megatron_utils/model.py L161-L165
# 提交版本：22cdc6e1
logger.warning(
    "Will reinitialize critic %s after checkpoint load because it is %s",
    param_name,
    reason,
)
return True
```

---

## Q4：use_stateless_adam 的限制？

**Explain：** 仅支持 adam optimizer；必须 `--no-save-optim`；DistributedOptimizer 的 init_state_fn 被 noop。

**Code：**

```python
# 来源：slime/backends/megatron_utils/model.py L304-L306
# 提交版本：22cdc6e1
assert config.optimizer == "adam", "Stateless Adam only supports --optimizer adam."
assert args.no_save_optim, "Stateless Adam does not save Adam moment states."
```

---

## Q5：forward_only 为何 pipeline last stage 才聚合？

**Explain：** PP 下只有 last stage 持有完整 logits/output；其它 stage 的 forward_data_store 不含最终 log_probs。

**Code：**

```python
# 来源：slime/backends/megatron_utils/model.py L488-L489
# 提交版本：22cdc6e1
if mpu.is_pipeline_last_stage():
    keys = forward_data_store[0].keys()
```

---

## Q6：train_iters 估算不准有何影响？

**Explain：** LR cosine/linear decay 可能略早或略晚达到 min_lr；**实际 step 仍由** `opt_param_scheduler.step(increment=...)` 跟踪 samples seen。

**Code：**

```python
# 来源：slime/backends/megatron_utils/model.py L195-L203
# 提交版本：22cdc6e1
# train_iters is an estimate ... actual total can drift ...
# Pass --lr-decay-iters explicitly if you need exact decay control.
args.train_iters = args.num_rollout * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
```

---

## Q7：custom_model_provider_path 如何支持 critic？

**Explain：** wrapped provider 在 `post_process and role=="critic"` 时强制替换 `LinearForLastLayer`；自定义模型必须暴露 `config.hidden_size`。

**Code：**

```python
# 来源：slime/backends/megatron_utils/model_provider.py L79-L82
# 提交版本：22cdc6e1
if post_process and role == "critic":
    model.output_layer = LinearForLastLayer(
        input_size=model.config.hidden_size, output_size=1, config=model.config
    )
```

---

## Q8：易错 — debug_rollout_only 仍调用 initialize？

**Explain：** **不会**。`debug_rollout_only` 在 MegatronTrainRayActor.init 入口短路，根本不进入 `initialize_model_and_optimizer`。

**正确理解：** debug 模式无 model/optimizer；仅 Rollout 调试。

**Code：**

```python
# 来源：slime/backends/megatron_utils/actor.py L56-L58
# 提交版本：22cdc6e1
if args.debug_rollout_only:
    self.args = args
    return 0
```
