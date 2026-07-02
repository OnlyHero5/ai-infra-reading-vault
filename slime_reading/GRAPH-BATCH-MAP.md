---
type: meta-doc
title: "Slime 图谱-批次映射表"
tags:
  - slime/meta
  - slime/doc/concept
updated: 2026-07-02
---

# Slime 图谱-批次映射表

> **用途：** `/understand` 产出 `knowledge-graph.json` 后，用本表校验每批是否覆盖 planned `nodeIds`；写作前亦可按本表预规划走读范围。  
> **基线 commit：** `22cdc6e1`  
> **图谱路径：** `F:/源码阅读/slime/.understand-anything/knowledge-graph.json`

---

## 1. 预规划架构分层（Phase 4 目标）

`/understand` Phase 4 应识别以下 7 层；批次 30 写入 `08-总结与索引-02-架构分层.md`。

| layer id | 名称 | 职责 | 代表 nodeIds |
|----------|------|------|--------------|
| `layer:entry-orchestration` | 入口与编排 | train 主循环、Ray 资源、CLI | `file:train.py`, `file:slime/ray/placement_group.py`, `file:slime/utils/arguments.py` |
| `layer:rollout-generation` | Rollout 生成 | 样本生成、RM、过滤、DataSource | `file:slime/ray/rollout.py`, `file:slime/rollout/sglang_rollout.py`, `file:slime/rollout/data_source.py` |
| `layer:sglang-backend` | SGLang 后端 | 推理引擎、PD 拓扑、外部引擎 | `file:slime/backends/sglang_utils/sglang_engine.py`, `file:slime/backends/sglang_utils/sglang_config.py` |
| `layer:megatron-training` | Megatron 训练 | Actor、Model、Loss、Data | `file:slime/backends/megatron_utils/actor.py`, `file:slime/backends/megatron_utils/loss.py`, `file:slime/backends/megatron_utils/model.py` |
| `layer:weight-sync` | 权重同步 | Train→Rollout 权重桥 | `file:slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py`, `file:slime/backends/megatron_utils/megatron_to_hf/__init__.py` |
| `layer:customization-agent` | 定制与 Agent | 插件 hook、多轮 Agent | `file:slime/agent/trajectory.py`, `document:docs/en/get_started/customization.md` |
| `layer:extensions-ops` | 扩展与运维 | plugins、examples、tools、CI | `file:slime_plugins/`, `document:docs/en/developer_guide/ci.md` |

---

## 2. 预规划 Guided Tour（Phase 5 目标）

| order | title | 核心 nodeIds | 对应批次 |
|-------|-------|--------------|----------|
| 1 | 项目愿景与三角架构 | `document:README.md`, `document:docs/en/blogs/introducing_slime.md` | 01 |
| 2 | 训练入口 train.py | `file:train.py`, `file:train_async.py` | 02 |
| 3 | 参数中枢 arguments | `file:slime/utils/arguments.py` | 03–04 |
| 4 | Ray GPU 编排 | `file:slime/ray/placement_group.py`, `file:slime/ray/actor_group.py` | 06–07 |
| 5 | RolloutManager.generate | `class:slime/ray/rollout.py:RolloutManager` | 08 |
| 6 | 默认 Rollout 路径 | `file:slime/rollout/sglang_rollout.py`, `file:slime/rollout/data_source.py` | 11–12 |
| 7 | SGLang 引擎与权重推送 | `file:slime/backends/sglang_utils/sglang_engine.py` | 15 |
| 8 | Megatron 训练一步 | `class:slime/backends/megatron_utils/actor.py:MegatronTrainRayActor` | 17–19 |
| 9 | RL Loss 与 Advantage | `file:slime/backends/megatron_utils/loss.py` | 21–22 |
| 10 | update_weights 闭环 | `function:slime/backends/megatron_utils/actor.py:update_weights` | 24–25 |
| 11 | 定制接口与 Agent | `document:docs/en/get_started/customization.md`, `file:slime/agent/trajectory.py` | 27–28 |
| 12 | 示例与插件生态 | `document:examples/README.md`, `file:slime_plugins/rollout_buffer/buffer.py` | 29 |

批次 30 将 tour 扩展为 `08-总结与索引-04-导读路径.md`。

---

## 3. 业务域流程（understand-domain 目标）

| flow id | 名称 | 步骤 | 覆盖批次 |
|---------|------|------|----------|
| `flow:rl-sync-loop` | 同步 RL 主循环 | parse_args → placement → generate → async_train → update_weights → save | 02, 06–08, 19, 24 |
| `flow:rl-async-loop` | 异步 RL 流水线 | prefetch generate(N+1) ∥ train(N) | 02, 14, 20 |
| `flow:rollout-sample` | 单样本 Rollout | data_source → generate → rm_hub → Sample → tensorize | 10–13 |
| `flow:agentic-rl` | Agentic 数据生成 | custom_generate → trajectory → convert → train | 27–28, 29 |
| `flow:weight-sync-nccl` | NCCL 权重同步 | megatron_to_hf → broadcast → sglang reload | 24–26 |
| `flow:weight-sync-delta` | Delta 磁盘同步 | delta write → engine patch | 25, 26 |

每 5 批更新 `Slime-业务域流程.md` 草稿（批次 05/10/15/20/25）；批次 30 定稿。

---

## 4. 批次 ↔ 文件 ↔ 图谱节点 ↔ 测试

| 批 | 模块目录 | 主文件（走读顺序） | 计划 nodeIds | 验证测试 |
|----|----------|-------------------|--------------|----------|
| **01** | `00-方法论/` | README, setup.py, docs/en/blogs/introducing_slime.md | `document:README.md`, `file:setup.py` | — |
| **02** | `02-训练主循环/` | train.py, train_async.py | `file:train.py`, `file:train_async.py` | `tests/test_qwen2.5_0.5B_async_short.py` |
| **03** | `03-Arguments-Ray/` | arguments.py §Ray/Cluster/Colocate | `file:slime/utils/arguments.py` | `tests/utils/test_sglang_config.py` |
| **04** | `04-Arguments-TrainRollout/` | arguments.py §Train/Rollout; megatron/sglang args | 同上 + `file:slime/backends/sglang_utils/arguments.py` | `tests/plugin_contracts/` |
| **05** | `05-Tools-DataPrep/` | tools/convert_*.py, scripts/models/*.sh | `file:tools/convert_hf_to_torch_dist.py` | — |
| **06** | `06-PlacementGroup/` | placement_group.py, ray/utils.py | `file:slime/ray/placement_group.py` | `tests/test_placement_group.py` |
| **07** | `07-RayTrainGroup/` | actor_group.py, train_actor.py, ray_actor.py | `file:slime/ray/actor_group.py`, `file:slime/ray/train_actor.py` | — |
| **08** | `08-RolloutManager/` | rollout.py RolloutManager.generate, convert, split | `class:slime/ray/rollout.py:RolloutManager` | `tests/test_rollout_validation.py` |
| **09** | `09-EngineTopology/` | rollout.py ServerGroup/RolloutServer; sglang_config.py | `file:slime/backends/sglang_utils/sglang_config.py` | `tests/test_qwen3_4B_external_pd.py` |
| **10** | `10-Sample-Contracts/` | types.py Sample; base_types.py; misc.load_function | `file:slime/utils/types.py`, `file:slime/rollout/base_types.py` | — |
| **11** | `11-DataSource/` | data_source.py; utils/data.py | `file:slime/rollout/data_source.py` | — |
| **12** | `12-SGLang-Rollout/` | sglang_rollout.py | `file:slime/rollout/sglang_rollout.py` | `tests/test_rollout_metrics.py` |
| **13** | `13-RM-FilterHub/` | rm_hub/*; filter_hub/* | `file:slime/rollout/rm_hub/__init__.py` | `tests/test_rm_math_dapo.py` |
| **14** | `14-Alt-Rollout/` | fully_async, streaming, sft, opd, forge_load | `file:slime/rollout/fully_async_rollout.py` | `tests/test_qwen2.5_0.5B_async_short.py` |
| **15** | `15-SGLang-Engine/` | sglang_engine.py, server_control.py | `file:slime/backends/sglang_utils/sglang_engine.py` | — |
| **16** | `16-External-Engines/` | external.py, health_monitor.py, http_utils.py | `file:slime/backends/sglang_utils/external.py` | — |
| **17** | `17-Megatron-Actor-Init/` | actor.py init/sleep/wake | `class:.../actor.py:MegatronTrainRayActor` | — |
| **18** | `18-Model-Init/` | initialize.py, model_provider.py, model.py init | `file:slime/backends/megatron_utils/model.py` | — |
| **19** | `19-Train-Step/` | actor.py train; model.py train | `function:.../actor.py:train` | `tests/test_qwen3_4B_ppo.py` |
| **20** | `20-Train-Data/` | megatron data.py; utils/data process_rollout_data | `file:slime/backends/megatron_utils/data.py` | — |
| **21** | `21-Loss-Advantages/` | loss.py compute_advantages, get_log_probs | `file:slime/backends/megatron_utils/loss.py` | `tests/test_chunked_gae.py` |
| **22** | `22-Loss-Policy/` | loss.py policy_loss, value_loss, custom hooks | 同上 | `tests/test_cispo_loss.py`, `tests/test_ppo_logprob_entropy_gpu.py` |
| **23** | `23-CP-RoutingReplay/` | cp_utils.py, routing_replay.py | `file:slime/backends/megatron_utils/cp_utils.py` | `tests/test_loss_cp_invariance.py` |
| **24** | `24-WeightSync-Dist/` | update_weight distributed; actor.update_weights | `file:.../update_weight_from_distributed.py` | `tests/test_full_disk_weight_update.py` |
| **25** | `25-WeightSync-Disk/` | disk, delta, tensor paths; disk_delta.py | `file:.../update_weight_from_disk_delta.py` | `examples/delta_weight_sync/` |
| **26** | `26-Checkpoint-M2HF/` | checkpoint.py, megatron_to_hf, hf_checkpoint_saver | `file:slime/backends/megatron_utils/checkpoint.py` | `tests/utils/test_hf_checkpoint_saver.py` |
| **27** | `27-Agent-Trajectory/` | trajectory.py, adapters/* | `file:slime/agent/trajectory.py` | `tests/test_agent/` |
| **28** | `28-Customization/` | customization.md; harness/*; agent parsing | `document:docs/en/get_started/customization.md` | `tests/plugin_contracts/` |
| **29** | `29-Plugins-Examples/` | slime_plugins/*; examples/search-r1, multi_agent | `file:slime_plugins/rollout_buffer/buffer.py` | `tests/gemma4/` |
| **30** | `08-总结与索引/` | 全库复盘 + megatron_server + trace/ci | 全部 layers + tour | `tests/ci/`, `docs/en/developer_guide/` |

---

## 5. 与 SGLang 阅读交叉索引

| Slime 批 | 建议 SGLang 批 | 原因 |
|----------|---------------|------|
| 15–16 | 03 HTTP Server, 07 Scheduler | SGLangEngine 启动与请求路径 |
| 09, 16 | 22 Disaggregation, 23 Distributed | PD 分离与多节点 |
| 24–26 | 12 ModelLoader, 32 CheckpointEngine | 权重格式与热更新 |
| 12 | 20 Sampling | rollout sampling_params |

---

## 6. 图谱验收（批次 30 前）

运行 `/understand` 后核对：

- [ ] `layers` 7 层 nodeIds 并集覆盖 `slime/slime/` 全部 `.py` 主文件
- [ ] `tour` ≥12 步，order 连续
- [ ] 热点文件 `complexity: complex` 标记存在（arguments, rollout, loss）
- [ ] `flow:rl-sync-loop` 各 step 能映射到 function/class 节点
- [ ] 每批 MOC 中 nodeIds 在图谱中存在（或标注「待增量更新」）

---

*图谱生成命令：`/understand --language zh F:/源码阅读/slime`*
