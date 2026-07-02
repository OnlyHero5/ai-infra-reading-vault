---
type: batch-doc
module: 22-Loss-Policy
batch: "22"
doc_type: moc
title: "Loss · Policy · 批次概述"
tags:
  - slime/batch/22
  - slime/module/loss-policy
  - slime/doc/moc
updated: 2026-07-02
---

# Loss · Policy · 批次概述

> **批次 22** | 阶段 IV 训练后端 | 状态：✅ 已完成 | 基线 commit `22cdc6e1`  
> 源码：`loss.py`（policy / value / sft / dispatch）+ `utils/ppo_utils.py`

---

## 本批目标

1. 追踪 `loss_function` 如何按 `args.loss_type` 分发并 rescale 给 Megatron
2. 对比 **PPO clip**、**GSPO 序列 KL**、**CISPO stop-gradient ratio** 三条 policy 分支
3. 理解 **TIS / ICEPOP**、**OPSM**、**KL loss** 在 `policy_loss_function` 中的挂载顺序
4. 说明 `value_loss_function` / `sft_loss_function` 的输入输出契约
5. 会用 `tests/test_cispo_loss.py` 验证 CISPO 闭式与梯度路径

---

## 文档导航

| 文档 | 内容 |
|------|------|
| [[22-Loss-Policy-01-核心概念]] | loss_type、估计器与 reducer |
| [[22-Loss-Policy-02-源码走读]] | **主文档**（热点 ≥400 行内嵌代码） |
| [[22-Loss-Policy-03-数据流与交互]] | forward_step → loss → train_one_step 指标 |
| [[22-Loss-Policy-04-关键问题]] | PPO vs GRPO vs CISPO、TIS FAQ |
| [[22-Loss-Policy-05-checkpoint]] | 验收 |

---

## 前置与衔接

- 上游：[[20-Train-Data-00-MOC]]（batch 字段）、[[21-Loss-Advantages-00-MOC]]（advantages 已写入 batch）
- 下游：[[23-CP-RoutingReplay-00-MOC]]（`get_sum_of_sample_mean` CP 细节）

---

## 入口：Megatron forward 如何调用 loss

**Explain：** `model.train_one_step` 每个 micro-batch 调用 `loss_function`；其内部构造 `sum_of_sample_mean` 再 dispatch 到具体 loss。

**Code：**

```python
# 来源：loss.py L1264-L1279
    match args.loss_type:
        case "policy_loss":
            func = policy_loss_function
        case "value_loss":
            func = value_loss_function
        case "sft_loss":
            func = sft_loss_function
        case "custom_loss":
            func = load_function(args.custom_loss_function_path)
        case _:
            raise ValueError(f"Unknown loss type: {args.loss_type}")

    if args.recompute_loss_function:
        loss, log = checkpoint(func, args, batch, logits, sum_of_sample_mean, use_reentrant=False)
    else:
        loss, log = func(args, batch, logits, sum_of_sample_mean)
```

---

## 相关测试

- `tests/test_cispo_loss.py` — CISPO 闭式与 stop-gradient
- `tests/test_ppo_logprob_entropy_gpu.py` — vocab parallel logprob
- `tests/test_loss_cp_invariance.py` — CP 下 loss（与批次 23 共用）
