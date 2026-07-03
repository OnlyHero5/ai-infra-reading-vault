---
type: batch-doc
module: 28-Customization
batch: "28"
doc_type: faq
title: "Customization · 关键问题"
tags:
  - slime/batch/28
  - slime/module/customization
  - slime/doc/faq
updated: 2026-07-02
---

# Customization · 关键问题

## Q1：custom_generate vs rollout_function 怎么选？

| 问题 | 选 custom_generate | 选 rollout_function |
|------|-------------------|---------------------|
| 仍用 RolloutManager 批调度？ | ✅ | 可能完全自管 |
| 只改 per-sample 逻辑？ | ✅ | overkill |
| 需要 cross-rollout 队列 / fully-async？ | ❌ | ✅ |
| multi_agent 并行子 agent？ | 可行 | example 用 rollout 路径 |

---

## Q2：忘记设 rollout_id 会怎样？

Sibling samples 被当成独立 rollout，GRPO group、train step 切分、metrics 全错。

**正确：**

```python
s.rollout_id = sample.rollout_id if sample.rollout_id is not None else sample.index
```

---

## Q3：custom_rm 何时用 batched？

`--group-rm` 开启时框架传 `list[Sample]` 期望 `list[float]` 返回；适合 batch 调外部 RM API。

---

## Q4：dynamic filter 和 buffer filter 区别？

- **dynamic**：generate 过程中按 group 决定 keep（DAPO）
- **buffer**：数据进训练 buffer 前再筛一道

二者可叠加，顺序见 RolloutManager 源码。

---

## Q5：custom_loss 需要什么前置？

必须 `--loss-type custom_loss`，否则 custom path 不被调用。

---

## Q6：harness 的 model_label 有用吗？

`HarnessContext.model_label` 只写入 CLI 环境变量；adapter **忽略**，实际模型由 SGLang 加载权重决定。

```python
## 来源：slime/agent/harness/common.py L46-L48
    model_label is the model name the harness advertises to its CLI. The slime
    adapter ignores it and serves whatever upstream sglang has loaded
```

---

## Q7：如何调试 long-running custom generate？

文档指向 `slime.utils.trace_utils`（见 developer_guide/trace.md）；对 tool/sandbox 步骤加 span。

---

## Q8：plugin_contracts 覆盖哪些 path？

| 测试文件 | 覆盖参数 |
|----------|----------|
| test_plugin_rollout_contracts | rollout-function-path |
| test_plugin_generate_contracts | custom-generate-function-path |
| test_plugin_path_loading_contracts | eval, rm, filters, data-source 等 |
| test_plugin_runtime_hook_contracts | log, convert, postprocess |

未覆盖：Megatron hooks、TIS、pg_loss_reducer——需集成测试补验证。

---

## Q9：MoE §18 算第 18 类接口吗？

文档单独成章，走 **CLI flags** 而非 `--*-path`；与 customization 哲学一致但加载机制不同。

---

## Q10：错误 path 何时暴露？

`load_function` 在 Actor/RolloutManager **初始化** 时 import；typo 通常启动即报错，不会 silent 回退默认。
