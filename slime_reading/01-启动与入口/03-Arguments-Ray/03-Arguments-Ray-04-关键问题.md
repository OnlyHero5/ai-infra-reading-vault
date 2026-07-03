---
type: batch-doc
module: 03-Arguments-Ray
batch: "03"
doc_type: faq
title: "Arguments-Ray · 关键问题"
tags:
  - slime/batch/03
  - slime/module/arguments
  - slime/doc/faq
updated: 2026-07-02
---

# Arguments-Ray · 关键问题

---

## Q1：colocate 为何强制 offload？

**Explain：** 同一 GPU 无法同时容纳 Megatron 训练态与 SGLang KV cache；必须时间复用：训时 rollout offload，生成时 train offload。

**Code：**

```python
## 来源：slime/utils/arguments.py L70-L77
                help=(
                    "Whether to colocate the inference engines and the actor. "
                    "Turning this on will also set --offload to true."
                ),
```

**Code：**

```python
## 来源：slime/utils/arguments.py L1886-L1890
    if args.colocate:
        if args.offload_train is None:
            args.offload_train = True
        if args.offload_rollout is None:
            args.offload_rollout = True
```

**Comment：** 可用 `--no-offload-train` 尝试，但 colocate 下 validate 会把 None 设为 True；PPO critic 还会强制 offload_train。

---

## Q2：rollout_num_gpus=0 是什么意思？

**Explain：** 不在 Ray 作业内启动本地 SGLang GPU worker；RolloutManager 仍可起 router，推理走 external addrs 或纯评测 pipeline。

**Code：**

```python
## 来源：slime/utils/arguments.py L48-L52
                    "Set it to 0 to launch routers without local SGLang engines."
```

**Code：**

```python
## 来源：slime/utils/arguments.py L1893-L1894
        elif args.rollout_num_gpus == 0:
            logger.info("rollout_num_gpus is 0 under colocate; no local SGLang engines will be launched.")
```

**Comment：** `debug_rollout_only` + `rollout_num_gpus=0` 会把 **actor 训练 GPU 也置 0**（L1869–1871），仅测 rollout 路径。

---

## Q3：colocate 时 rollout_num_gpus 默认多少？

**Code：**

```python
## 来源：slime/utils/arguments.py L1891-L1892
        if args.rollout_num_gpus is None:
            args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
```

**Comment：** 与 actor 总 GPU 相同，表示 **同一组物理卡** 上交替跑 train/infer。

---

## Q4：decoupled 时 rollout 默认等于 train GPU 吗？

**Explain：** validate 中 colocate 块外的 `rollout_num_gpus is None` 也在 L1891 处理；decoupled 未设时同样 fallback 到 actor 总 GPU——常见做法是 **显式加大** `--rollout-num-gpus` 扩 inference。

---

## Q5：为何 delta weight 不能 colocate？

**Code：**

```python
## 来源：slime/utils/arguments.py L1992-L1997
        if args.colocate:
            raise ValueError(
                "--update-weight-mode=delta is not supported with --colocate. Colocate transfers "
                "weights via CUDA IPC ..."
            )
```

**Comment：** colocate 走 IPC handle；delta 设计给 disk + 跨机 disaggregation（[[25-WeightSync-Disk-00-MOC]]）。

---

## Q6：PPO 为何强制 offload_train？

**Code：**

```python
## 来源：slime/utils/arguments.py L1901-L1902
    if args.use_critic:
        args.offload_train = True
```

**Comment：** Actor + Critic 双模型；与 colocate rollout offload 正交。

---

## Q7：debug_rollout_only 如何改 cluster 布局？

**Code：**

```python
## 来源：slime/utils/arguments.py L1872-L1876
            args.actor_num_gpus_per_node = min(8, args.rollout_num_gpus)
            args.actor_num_nodes = args.rollout_num_gpus // args.actor_num_gpus_per_node
        args.colocate = False
        args.offload_train = args.offload_rollout = False
```

---

## Q8：少于 8 卡 colocate 要注意什么？

**Code：**

```python
## 来源：slime/utils/arguments.py L65-L67
                    "Notice: If you are going to use less than 8 gpus per node under colocate mode, you should set this number."
```

**Comment：** 同步设置 `--num-gpus-per-node` 与 actor 卡数。

---

## Q9：parse_args 跳过 SGLang 的条件？

**Code：**

```python
## 来源：slime/utils/arguments.py L1552-L1559
    skip_sglang = pre.debug_train_only or pre.load_debug_rollout_data is not None
    sglang_ns = None
    if not skip_sglang:
        sglang_ns = sglang_parse_args()
```

---

## Q10：rollout_num_gpus_per_engine 设错会怎样？

**Code：**

```python
## 来源：slime/backends/sglang_utils/arguments.py L147-L151
        assert args.rollout_num_gpus_per_engine % args.sglang_pp_size == 0, (
            f"rollout_num_gpus_per_engine ({args.rollout_num_gpus_per_engine}) must be divisible by "
            f"sglang_pipeline_parallel_size ({args.sglang_pp_size})"
        )
```

**Comment：** 启动时 assert；引擎数 × gpus_per_engine 应等于 rollout_num_gpus（ modulo placeholder groups）。
