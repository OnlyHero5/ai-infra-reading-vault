---
type: batch-doc
module: 29-Plugins-Examples
batch: "29"
doc_type: faq
title: "Plugins Examples · 关键问题"
tags:
  - slime/batch/29
  - slime/module/plugins-examples
  - slime/doc/faq
updated: 2026-07-02
---

# Plugins Examples · 关键问题

## Q1：search-r1 为何用 custom_generate 而非 rollout_function？

它只需改 **单 sample 内** 多轮逻辑，仍依赖 RolloutManager 的 batching、filter、RM 调度。整段替换无收益且失去默认 buffer/filter 集成。

---

## Q2：rollout_buffer 何时值得上？

- 数据生成跑在 **另一集群**，与 GPU 训练进程分离
- 需要 **按 group 攒够** 再训练（GRPO group size）
- 多种 `TASK_TYPE` generator 共用同一队列服务

默认 Slime 内置 `RolloutDataSourceWithBuffer` 对多数单机实验足够。

---

## Q3：generator 模块必须实现什么？

| 符号 | 必需 |
|------|------|
| `TASK_TYPE` | ✅ 字符串常量 |
| `run_rollout(data)` | ✅ |
| `transform_group` | 可选 |
| `is_valid_group` | 可选（默认 len ≥ group_size） |
| `get_group_data_meta_info` | 可选 |

---

## Q4：Search-R1 partial_rollout 为何不支持？

```python
## 来源：examples/search-r1/generate_with_search.py L146
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."
```

多轮 state 与 partial 续写语义冲突；需自管 checkpoint 才能支持。

---

## Q5：observation token 为何 loss_mask=0？

Search 结果、tool 输出不是 policy 采样，不应算 policy gradient；仅 model 生成段为 1。

---

## Q6：multi_agent shuffle 会影响训练吗？

```python
random.shuffle(samples)
```

仅改变 batch 内顺序；同 group 样本仍共享 group_index。若 RM 依赖顺序需去掉 shuffle。

---

## Q7：slime_plugins 要 pip install 吗？

通常 `PYTHONPATH` 含 repo 根或 `pip install -e .`；`import slime_plugins.megatron_bridge` 在 checkpoint 加载时 side-effect 注册。

---

## Q8：glm5 plugin 与 examples 关系？

glm5 是 **模型实现**，不是 runnable example；需在 Megatron args 选对应 model provider，并确保 [[26-Checkpoint-M2HF-00-MOC]] converter 路由含该架构。

---

## Q9：如何从 example 拷贝到自己的项目？

1. 复制 generate/rollout/rm 函数到你的 package
2. CLI 改为你的 module path
3. 跑 `tests/plugin_contracts/test_plugin_generate_contracts.py` 或 rollout 对应测试
4. 小模型 smoke train

---

## Q10：examples 与 docs `_examples_synced` 区别？

文档站 build 时 sync 到 `docs/en/_examples_synced/`；源码以 `examples/` 为准。
