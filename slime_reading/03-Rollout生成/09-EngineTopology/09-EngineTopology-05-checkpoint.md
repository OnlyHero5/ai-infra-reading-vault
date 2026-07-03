---
type: batch-doc
module: 09-EngineTopology
batch: "09"
doc_type: checkpoint
title: "EngineTopology · 验收清单"
tags:
  - slime/batch/09
  - slime/module/engine-topology
  - slime/doc/checkpoint
updated: 2026-07-02
---

# EngineTopology · 验收清单

> 完成本专题五篇正文后逐项自测。状态：**已完成** ✅

---

## 读者自测（不打开 slime/）

- [ ] 能说明 **SglangConfig → ModelConfig → ServerGroupConfig** 三层配置各自职责
- [ ] 能画出 **regular 单 Router 单组** 与 **PD prefill+decode 双组** 的拓扑差异
- [ ] 能对比 **`--prefill-num-servers`** 与 **`--sglang-config`** 的适用场景
- [ ] 能说出 **`ServerGroup` / `RolloutServer` / `_start_router`** 三者分工
- [ ] 能解释 **为何 PD 时 Router 设置 `disable_circuit_breaker=True`**
- [ ] 能追踪 **`RolloutManager.__init__` → `start_rollout_servers` → `ray.get(init_handles)`** 路径
- [ ] 能说明 **多模型场景下 `args.sglang_model_routers` 的用途**
- [ ] 能列举 **worker_type** 五种取值及 placeholder 的特殊行为
- [ ] 能说明 **EPD 两阶段启动** 为何 encoder 必须先 `ray.get` ready
- [ ] 能解释 **`--rollout-num-gpus` 必须与 YAML GPU 总和一致** 的原因

---

## RL 闭环位置

- [ ] 能指出 EngineTopology 在 **generate → train → update_weights** 中位于 **Rollout 引擎启动** 阶段
- [ ] 能说明拓扑 **不影响** loss 计算，但 **影响** rollout 吞吐与 PD metrics
- [ ] 能说明 **update_weights** 如何跳过 `update_weights=False` 的 RolloutServer

---

## Obsidian / 维护检查

- [ ] 文件名前缀 `09-EngineTopology-`，无泛化 `README` / `01-核心概念`
- [ ] frontmatter 含 `slime/batch/09` + `slime/doc/*`
- [ ] Mermaid 使用 `<br/>` 换行，无 `\n`
- [ ] 双链指向相邻专题（[[08-RolloutManager-00-MOC]]、[[22-Disaggregation-00-MOC]] 等）

---

## 建议动手验证（可选，需 GPU 环境）

```bash
## 来源：slime/tests/utils/test_sglang_config.py
pytest slime/tests/utils/test_sglang_config.py -k resolve -v
```

```bash
# external PD 集成测试（需多 GPU）
pytest slime/tests/test_qwen3_4B_external_pd.py -v
```

---

## 下一批预告

[[10-Sample-Contracts-00-MOC]]：`Sample` / `RolloutFnTrainOutput` 契约——拓扑就绪后，generate 产物的数据结构。
