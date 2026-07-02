---
type: meta-doc
title: "Slime 阅读 Agent 派工任务书"
tags:
  - slime/meta
  - slime/doc/concept
updated: 2026-07-02
---

# Slime 阅读 Agent 派工任务书

> **用途：** 派 Agent 阅读 `slime/` 源码并落盘到 `slime_reading/` 的**唯一执行依据**。  
> **读者不读本文**；读者只读各批次六件套正文。  
> **维护者先读：** [[Slime-PLAN]] → [[UNDERSTAND-WORKFLOW]] → [[GRAPH-BATCH-MAP]] → 本文。

---

## 一、全局 Agent 指令（每个批次任务开头粘贴）

```markdown
## 角色
你是 Slime 源码阅读笔记作者。产出写入 `F:\源码阅读\slime_reading/`，读者**只读 slime_reading，不读 slime**。

## 硬性规则
1. 每批产出 **6 篇**：`{模块}-00-MOC` + `01–04` + `05-checkpoint`
2. 文件名格式：`{NN-ModuleName}-{00–05}-{类型}.md`（禁止 README / 01-核心概念 泛化名）
3. 每篇采用 **ETC**：Explain → Code → Comment；禁止只写路径不贴代码
4. 每批合计 **≥15 段代码块、≥200 行**；热点批（03–04, 08–09, 21–22）**≥400 行**
5. 代码块首行注释：`# 来源：相对路径 L起始-L结束`；基线 commit `22cdc6e1`
6. frontmatter tags：`slime/batch/NN` + `slime/doc/{moc|concept|walkthrough|dataflow|faq|checkpoint}` + `slime/module/xxx`
7. 模块间链接用 Obsidian 双链 `[[08-RolloutManager-01-核心概念]]`
8. Mermaid 换行用 `<br/>`，禁止 `\n`
9. 禁止编造函数签名、行号、行为；不确定时读源码核对
10. 完成后更新 `slime_reading/Slime-progress.md` 本批状态为 ✅

## 写作侧 Skills（按顺序）
1. 读 GRAPH-BATCH-MAP 本批 nodeIds / 测试文件
2. `/understand-explain <主文件>` 生成结构草稿（若图谱不存在则直接读源码）
3. 写六件套
4. 填 checkpoint 自测
5. 更新 Slime-progress.md

## 模板
复制 `_TEMPLATE/README.md` 结构与 frontmatter 规范。
```

---

## 二、30 批次总览

| 批 | 模块名 | 产出目录 | 前置批 | 工时 | 阶段验收点 |
|----|--------|----------|--------|------|-----------|
| 01 | 00-方法论 | `00-方法论/` | — | 2h | 能说明 Slime 三角架构 |
| 02 | 02-训练主循环 | `01-启动与入口/02-训练主循环/` | 01 | 4h | 能口述 sync 主循环 |
| 03 | 03-Arguments-Ray | `01-启动与入口/03-Arguments-Ray/` | 02 | 4h | 能解释 colocate/offload |
| 04 | 04-Arguments-TrainRollout | `01-启动与入口/04-Arguments-TrainRollout/` | 03 | 4h | 能列举 customization 入口参数 |
| 05 | 05-Tools-DataPrep | `01-启动与入口/05-Tools-DataPrep/` | 04 | 3h | 能说明 HF↔Megatron 转换 |
| 06 | 06-PlacementGroup | `02-Ray编排/06-PlacementGroup/` | 02 | 4h | 能画 PG 分配图 |
| 07 | 07-RayTrainGroup | `02-Ray编排/07-RayTrainGroup/` | 06 | 4h | 能说明 RayTrainGroup API |
| 08 | 08-RolloutManager | `03-Rollout生成/08-RolloutManager/` | 07 | 6h | 能追踪 generate() |
| 09 | 09-EngineTopology | `03-Rollout生成/09-EngineTopology/` | 08 | 5h | 能说明 PD/多模型拓扑 |
| 10 | 10-Sample-Contracts | `03-Rollout生成/10-Sample-Contracts/` | 08 | 3h | 能解释 Sample 字段 |
| 11 | 11-DataSource | `03-Rollout生成/11-DataSource/` | 10 | 4h | 能说明 prompt 从哪来 |
| 12 | 12-SGLang-Rollout | `03-Rollout生成/12-SGLang-Rollout/` | 11 | 5h | 能走 default generate 路径 |
| 13 | 13-RM-FilterHub | `03-Rollout生成/13-RM-FilterHub/` | 12 | 3h | 能挂 custom-rm |
| 14 | 14-Alt-Rollout | `03-Rollout生成/14-Alt-Rollout/` | 12 | 4h | 能对比 sync/async rollout |
| 15 | 15-SGLang-Engine | `03-Rollout生成/15-SGLang-Engine/` | 09 | 5h | 能说明 engine 生命周期 |
| 16 | 16-External-Engines | `03-Rollout生成/16-External-Engines/` | 15 | 3h | 能说明外部引擎模式 |
| 17 | 17-Megatron-Actor-Init | `04-训练后端/17-Megatron-Actor-Init/` | 07 | 5h | 能说明 actor init 步骤 |
| 18 | 18-Model-Init | `04-训练后端/18-Model-Init/` | 17 | 5h | 能说明 model 初始化 |
| 19 | 19-Train-Step | `04-训练后端/19-Train-Step/` | 18 | 5h | 能追踪一次 train step |
| 20 | 20-Train-Data | `04-训练后端/20-Train-Data/` | 08,19 | 4h | 能说明 rollout→batch |
| 21 | 21-Loss-Advantages | `04-训练后端/21-Loss-Advantages/` | 20 | 5h | 能解释 advantage 计算 |
| 22 | 22-Loss-Policy | `04-训练后端/22-Loss-Policy/` | 21 | 5h | 能对比 PPO/GRPO 分支 |
| 23 | 23-CP-RoutingReplay | `04-训练后端/23-CP-RoutingReplay/` | 22 | 3h | 能说明 CP 与 MoE replay |
| 24 | 24-WeightSync-Dist | `05-权重同步/24-WeightSync-Dist/` | 19,15 | 5h | 能说明 NCCL 同步 |
| 25 | 25-WeightSync-Disk | `05-权重同步/25-WeightSync-Disk/` | 24 | 4h | 能对比 disk/delta/tensor |
| 26 | 26-Checkpoint-M2HF | `05-权重同步/26-Checkpoint-M2HF/` | 25 | 4h | 能说明 checkpoint 路径 |
| 27 | 27-Agent-Trajectory | `06-高级特性/27-Agent-Trajectory/` | 12 | 5h | 能说明 trajectory→Sample |
| 28 | 28-Customization | `06-高级特性/28-Customization/` | 04,27 | 4h | 能选 customization 接口 |
| 29 | 29-Plugins-Examples | `07-扩展与生态/29-Plugins-Examples/` | 28 | 5h | 能举一个 example 接入点 |
| 30 | 08-总结与索引 | `08-总结与索引/` | 01–29 | 8h | onboard 全套 + 全链路 |

**图谱增量：** 完成 05 / 10 / 15 / 20 / 25 / 30 时运行 `/understand --language zh` + `/understand-domain`。

---

## 三、逐批 Agent 任务（复制即用）

### 批次 01 · 00-方法论

**产出目录：** `slime_reading/00-方法论/`  
**六件套前缀：** `00-方法论-`

**源码范围（按序读）：**
1. `slime/README.md` + `README_zh.md`
2. `slime/docs/en/blogs/introducing_slime.md`
3. `slime/setup.py`, `requirements.txt`
4. `slime/imgs/arch.png`（用文字复述架构图）

**走读重点：**
- Slime 两大能力：Megatron 训练 + SGLang Rollout
- Training / Rollout / Data Buffer 三角
- 与 verl / OpenRLHF 差异（原生透传 Megatron/SGLang 参数）

**01-核心概念必答：**
1. Slime 解决什么问题？
2. generate → train → update_weights 是什么？
3. slime_reading 六件套与 ETC 怎么读？

**04-关键问题：** 阅读顺序建议、SGLang 前置知识指向 [[SGLang源码阅读指南]]

**衔接：** → [[02-训练主循环-00-MOC]]

---

### 批次 02 · 02-训练主循环

**产出目录：** `slime_reading/01-启动与入口/02-训练主循环/`  
**六件套前缀：** `02-训练主循环-`  
**前置：** 01

**源码范围：**
1. `slime/train.py` — 全文（103 行）
2. `slime/train_async.py` — 全文（81 行）
3. `slime/slime/utils/misc.py` — `should_run_periodic_action`

**走读顺序（02-源码走读）：**
1. `train()` bootstrap：`create_placement_groups` → `create_rollout_manager` → `create_training_models`
2. 首次 `update_weights` 与 offload 分支
3. `for rollout_id` 主循环：`generate` → `offload` → `async_train` → `save` → `update_weights`
4. critic-only steps：`num_critic_only_steps`
5. `train_async.py` 与 sync 差异：prefetch generate、无 colocate
6. eval-only：`num_rollout == 0`

**03-数据流：** Mermaid 时序图 sync vs async  
**验证建议：** 读 `tests/test_qwen2.5_0.5B_async_short.py` 注释理解异步断言

---

### 批次 03 · 03-Arguments-Ray

**产出目录：** `slime_reading/01-启动与入口/03-Arguments-Ray/`  
**六件套前缀：** `03-Arguments-Ray-`  
**前置：** 02  
**热点：** ≥400 行内嵌代码（arguments.py 很大，本批只覆盖 Ray 段）

**源码范围：**
- `slime/slime/utils/arguments.py`：
  - `get_slime_extra_args_provider` → `add_cluster_arguments`
  - 参数：`--actor-num-nodes`, `--rollout-num-gpus`, `--colocate`, `--offload`, `--offload-rollout`, `--offload-train`
  - `parse_args()` 入口与 Ray 相关 validate

**01-核心概念术语：** colocate, offload, placement, rollout_num_gpus

**04-关键问题：** colocate 为何强制 offload；rollout_num_gpus=0 含义

---

### 批次 04 · 04-Arguments-TrainRollout

**产出目录：** `slime_reading/01-启动与入口/04-Arguments-TrainRollout/`  
**六件套前缀：** `04-Arguments-TrainRollout-`  
**前置：** 03

**源码范围：**
1. `arguments.py` — Train/Rollout/Customization 段（`*-path` 参数全集）
2. `slime/slime/backends/sglang_utils/arguments.py` — `--sglang-*` 透传
3. `slime/slime/backends/megatron_utils/arguments.py` — Megatron validate
4. `slime/docs/en/get_started/customization.md` — 接口表（内嵌表格+对应源码）

**走读重点：** `load_function` 如何解析 `--rollout-function-path` 等  
**验证建议：** `tests/plugin_contracts/` 目录列举

---

### 批次 05 · 05-Tools-DataPrep

**产出目录：** `slime_reading/01-启动与入口/05-Tools-DataPrep/`  
**六件套前缀：** `05-Tools-DataPrep-`  
**前置：** 04

**源码范围：**
1. `slime/tools/convert_hf_to_torch_dist.py`
2. `slime/tools/convert_torch_dist_to_hf.py`
3. `slime/scripts/models/qwen3-4B.sh`（代表脚本）
4. `slime/docs/en/get_started/quick_start.md` § 数据准备

**阶段 I 验收段（写入 MOC）：** 从 `parse_args` 到主循环的调用栈图

**维护者额外：** 完成后触发 `/understand` 增量 + `/understand-domain`

---

### 批次 06 · 06-PlacementGroup

**产出目录：** `slime_reading/02-Ray编排/06-PlacementGroup/`  
**六件套前缀：** `06-PlacementGroup-`

**源码范围：**
1. `slime/slime/ray/placement_group.py` — `create_placement_groups`, `_create_placement_group`, `create_rollout_manager`, `create_training_models`
2. `slime/slime/ray/utils.py` — `add_default_ray_env_vars`, `Lock`
3. `InfoActor`, `sort_key` GPU 排序逻辑

**03-数据流：** train / rollout / critic PG 如何拆分  
**验证建议：** `tests/test_placement_group.py`

---

### 批次 07 · 07-RayTrainGroup

**产出目录：** `slime_reading/02-Ray编排/07-RayTrainGroup/`  
**六件套前缀：** `07-RayTrainGroup-`

**源码范围：**
1. `slime/slime/ray/actor_group.py` — `RayTrainGroup` 全 API
2. `slime/slime/ray/train_actor.py` — `TrainRayActor` 基类
3. `slime/slime/ray/ray_actor.py` — master addr/port

**走读：** `async_init`, `async_train`, `update_weights`, `save_model` 如何 `.remote()`

---

### 批次 08 · 08-RolloutManager

**产出目录：** `slime_reading/03-Rollout生成/08-RolloutManager/`  
**六件套前缀：** `08-RolloutManager-`  
**热点：** ≥400 行

**源码范围：** `slime/slime/ray/rollout.py`
- `RolloutManager.__init__` — data_source / rollout fn 加载
- `generate(rollout_id)` ~L546
- `_get_rollout_data`, `_convert_samples_to_train_data`, `_split_train_data_by_dp`
- `_tensorize_rollout_data_for_training`
- `get_updatable_engines_and_lock`

**01-核心概念：** RolloutManager 在三角中的位置  
**03-数据流：** Sample list → tensor dict → Ray ObjectRef per DP

---

### 批次 09 · 09-EngineTopology

**产出目录：** `slime_reading/03-Rollout生成/09-EngineTopology/`  
**六件套前缀：** `09-EngineTopology-`

**源码范围：**
1. `rollout.py` — `ServerGroup`, `RolloutServer`, `start_rollout_servers`, `_start_router`
2. `slime/slime/backends/sglang_utils/sglang_config.py` — `SglangConfig`, `ServerGroupConfig`
3. `slime/docs/en/advanced/pd-disaggregation.md`（摘录+代码对应点）

**04-关键问题：** PD 分离 vs 普通拓扑选型  
**SGLang 交叉：** [[22-Disaggregation-00-MOC]]

---

### 批次 10 · 10-Sample-Contracts

**产出目录：** `slime_reading/03-Rollout生成/10-Sample-Contracts/`  
**六件套前缀：** `10-Sample-Contracts-`

**源码范围：**
1. `slime/slime/utils/types.py` — `Sample`, `RolloutBatch`, top-p replay 字段
2. `slime/slime/rollout/base_types.py` — `RolloutFnTrainOutput`, `call_rollout_fn`
3. `slime/slime/utils/misc.py` — `load_function`, `Box`

**01-核心概念：** Sample 各字段在训练中的用途（tokens, loss_masks, rewards, rollout_log_probs）

---

### 批次 11 · 11-DataSource

**产出目录：** `slime_reading/03-Rollout生成/11-DataSource/`  
**六件套前缀：** `11-DataSource-`

**源码范围：**
1. `slime/slime/rollout/data_source.py` — `RolloutDataSource`, buffer 变体
2. `slime/slime/utils/data.py` — 数据加载相关（与 rollout 交接部分）

---

### 批次 12 · 12-SGLang-Rollout

**产出目录：** `slime_reading/03-Rollout生成/12-SGLang-Rollout/`  
**六件套前缀：** `12-SGLang-Rollout-`

**源码范围：** `slime/slime/rollout/sglang_rollout.py`
- `generate_rollout` 入口
- `GenerateState`
- `generate_and_rm_group` / HTTP 调用 SGLang router
- `--custom-generate-function-path` 挂载点

**验证建议：** `tests/test_rollout_metrics.py`

---

### 批次 13 · 13-RM-FilterHub

**产出目录：** `slime_reading/03-Rollout生成/13-RM-FilterHub/`  
**六件套前缀：** `13-RM-FilterHub-`

**源码范围：**
1. `slime/slime/rollout/rm_hub/__init__.py` — `async_rm`
2. `rm_hub/math_utils.py`, `math_dapo_utils.py`, `deepscaler.py`
3. `slime/slime/rollout/filter_hub/` — dynamic sampling filters

---

### 批次 14 · 14-Alt-Rollout

**产出目录：** `slime_reading/03-Rollout生成/14-Alt-Rollout/`  
**六件套前缀：** `14-Alt-Rollout-`

**源码范围：**
1. `fully_async_rollout.py`
2. `sglang_streaming_rollout.py`
3. `sft_rollout.py`
4. `on_policy_distillation.py`
5. `sleep_rollout.py`, `forge_load.py`

**03-数据流：** fully-async 与 train_async 如何配合

**维护者：** 完成批 10 后触发图谱/domain 增量

---

### 批次 15 · 15-SGLang-Engine

**产出目录：** `slime_reading/03-Rollout生成/15-SGLang-Engine/`  
**六件套前缀：** `15-SGLang-Engine-`

**源码范围：**
1. `slime/slime/backends/sglang_utils/sglang_engine.py` — `SGLangEngine`, `launch_server_process`, `update_weights*`
2. `server_control.py`

**03-数据流：** 权重更新 NCCL group 建立

---

### 批次 16 · 16-External-Engines

**产出目录：** `slime_reading/03-Rollout生成/16-External-Engines/`  
**六件套前缀：** `16-External-Engines-`

**源码范围：**
1. `external.py` — `start_external_rollout_servers`
2. `slime/slime/utils/health_monitor.py`
3. `slime/slime/utils/http_utils.py`
4. `docs/en/advanced/external-rollout-engines.md`

---

### 批次 17 · 17-Megatron-Actor-Init

**产出目录：** `slime_reading/04-训练后端/17-Megatron-Actor-Init/`  
**六件套前缀：** `17-Megatron-Actor-Init-`

**源码范围：**
1. `slime/slime/backends/megatron_utils/actor.py` — `MegatronTrainRayActor.init`, sleep/wake, `debug_rollout_only`
2. `initialize.py` — `init()`

---

### 批次 18 · 18-Model-Init

**产出目录：** `slime_reading/04-训练后端/18-Model-Init/`  
**六件套前缀：** `18-Model-Init-`

**源码范围：**
1. `model_provider.py`
2. `model.py` — `initialize_model_and_optimizer`, `forward_only`

---

### 批次 19 · 19-Train-Step

**产出目录：** `slime_reading/04-训练后端/19-Train-Step/`  
**六件套前缀：** `19-Train-Step-`

**源码范围：**
1. `actor.py` — `train`, `async_train`, `train_actor`, `train_critic`
2. `model.py` — `train`, `train_one_step`

**验证建议：** `tests/test_qwen3_4B_ppo.py`

---

### 批次 20 · 20-Train-Data

**产出目录：** `slime_reading/04-训练后端/20-Train-Data/`  
**六件套前缀：** `20-Train-Data-`

**源码范围：**
1. `megatron_utils/data.py` — `get_data_iterator`, `get_batch`
2. `utils/data.py` — `process_rollout_data`
3. `seqlen_balancing.py`, `dp_schedule.py`

**维护者：** 完成批 15 后图谱/domain 增量

---

### 批次 21 · 21-Loss-Advantages

**产出目录：** `slime_reading/04-训练后端/21-Loss-Advantages/`  
**六件套前缀：** `21-Loss-Advantages-`  
**热点：** ≥400 行

**源码范围：** `loss.py`
- `compute_advantages_and_returns` ~L661
- `get_log_probs_and_entropy` ~L470
- `get_values` ~L564
- `apply_opd_kl_to_advantages`

---

### 批次 22 · 22-Loss-Policy

**产出目录：** `slime_reading/04-训练后端/22-Loss-Policy/`  
**六件套前缀：** `22-Loss-Policy-`

**源码范围：** `loss.py`
- `policy_loss_function` ~L881
- `value_loss_function`, `sft_loss_function`, `loss_function`
- `vanilla_tis_function`, `icepop_function`
- `slime/utils/ppo_utils.py`

**验证建议：** `tests/test_cispo_loss.py`, `tests/test_ppo_logprob_entropy_gpu.py`

---

### 批次 23 · 23-CP-RoutingReplay

**产出目录：** `slime_reading/04-训练后端/23-CP-RoutingReplay/`  
**六件套前缀：** `23-CP-RoutingReplay-`

**源码范围：**
1. `cp_utils.py`
2. `routing_replay.py`
3. `actor.py` 中 `fill_routing_replay` 相关

**验证建议：** `tests/test_loss_cp_invariance.py`

---

### 批次 24 · 24-WeightSync-Dist

**产出目录：** `slime_reading/05-权重同步/24-WeightSync-Dist/`  
**六件套前缀：** `24-WeightSync-Dist-`

**源码范围：**
1. `actor.py` — `update_weights` ~L583
2. `update_weight/common.py`
3. `update_weight/update_weight_from_distributed.py`
4. `update_weight/hf_weight_iterator_direct.py`

---

### 批次 25 · 25-WeightSync-Disk

**产出目录：** `slime_reading/05-权重同步/25-WeightSync-Disk/`  
**六件套前缀：** `25-WeightSync-Disk-`

**源码范围：**
1. `update_weight_from_disk.py`
2. `update_weight_from_disk_delta.py`
3. `update_weight_from_tensor.py`（colocate）
4. `slime/utils/disk_delta.py`
5. `docs/en/advanced/delta-weight-sync.md`

**维护者：** 完成批 20 后图谱/domain 增量

---

### 批次 26 · 26-Checkpoint-M2HF

**产出目录：** `slime_reading/05-权重同步/26-Checkpoint-M2HF/`  
**六件套前缀：** `26-Checkpoint-M2HF-`

**源码范围：**
1. `checkpoint.py`
2. `megatron_to_hf/__init__.py` — `convert_to_hf` 路由
3. `megatron_to_hf/qwen2.py`（代表一个 converter）
4. `hf_checkpoint_saver.py`

---

### 批次 27 · 27-Agent-Trajectory

**产出目录：** `slime_reading/06-高级特性/27-Agent-Trajectory/`  
**六件套前缀：** `27-Agent-Trajectory-`

**源码范围：**
1. `slime/agent/trajectory.py` — `TrajectoryManager`
2. `adapters/common.py`, `openai.py`, `anthropic.py`
3. `docs/en/get_started/agent.md`

**验证建议：** `tests/test_agent/`

---

### 批次 28 · 28-Customization

**产出目录：** `slime_reading/06-高级特性/28-Customization/`  
**六件套前缀：** `28-Customization-`

**源码范围：**
1. `docs/en/get_started/customization.md` — 17 类接口逐条配代码入口
2. `agent/harness/common.py`, `claude_code.py`, `codex.py`
3. `agent/parsing.py`

**03-数据流：** Agentic RL 接入 decision tree（用哪个 `--*-path`）

---

### 批次 29 · 29-Plugins-Examples

**产出目录：** `slime_reading/07-扩展与生态/29-Plugins-Examples/`  
**六件套前缀：** `29-Plugins-Examples-`

**源码范围：**
1. `slime_plugins/rollout_buffer/buffer.py`
2. `slime_plugins/models/glm5/glm5.py`（代表）
3. `examples/search-r1/generate_with_search.py`
4. `examples/multi_agent/rollout_with_multi_agents.py`
5. `examples/README.md`

**维护者：** 完成批 25 后图谱/domain 增量

---

### 批次 30 · 08-总结与索引（收官）

**产出目录：** `slime_reading/08-总结与索引/`  
**前置：** 01–29 全部完成

**任务：** 按 [[UNDERSTAND-WORKFLOW]] §3 产出 onboard 全套 + 以下文件：

| 文件 | 要求 |
|------|------|
| `08-总结与索引-00-MOC.md` | 阶段导航 |
| `08-总结与索引-01-项目总览.md` | 含 setup/train 代码片段 |
| `08-总结与索引-02-架构分层.md` | 7 层 + 每层代表代码 |
| `08-总结与索引-03-关键概念.md` | Sample/rollout_id/update_weights 等 |
| `08-总结与索引-04-导读路径.md` | 12 步 tour + 每步内嵌代码 |
| `08-总结与索引-05-文件地图.md` | slime/ 顶层文件 3–5 行代码+职责 |
| `08-总结与索引-06-复杂度热点.md` | 10 个热点函数完整内嵌 |
| `08-总结与索引-07-可观测与CI.md` | trace/profile/CI/fault-tolerance |
| `全链路RL训练追踪.md` | generate→train→update_weights 每跳代码 |
| `Slime-业务域流程.md` | 6 条 flow（见 GRAPH-BATCH-MAP §3） |
| `Slime-模块依赖图.md` | Mermaid + import 示例 |
| `Slime-术语表.md` | RL 术语 + 代码出处 |
| `与SGLang阅读对照.md` | 批次对照表 |
| `08-总结与索引-05-checkpoint.md` | 收官验收 |
| `90_meta/slime-module-dir-map.md` | 目录名 vs 专题名 |

**额外源码（07-可观测）：**
- `megatron_utils/server/megatron_server.py`（alternate 入口摘要）
- `utils/trace_utils.py`, `profile_utils.py`
- `docs/en/developer_guide/ci.md`, `trace.md`, `debug.md`

---

## 四、派工 Prompt 模板（复制后替换 `{NN}` `{Module}` `{dir}`）

```markdown
请执行 Slime 源码阅读批次 {NN}。

1. 阅读 `F:\源码阅读\slime_reading\AGENT-DISPATCH.md` 中「批次 {NN}」章节 + 文首全局指令
2. 阅读 `F:\源码阅读\slime_reading\_TEMPLATE\README.md`
3. 按 AGENT-DISPATCH 列出的源码范围精读 `F:\源码阅读\slime\` 下文件
4. 在 `F:\源码阅读\slime_reading/{dir}/` 创建六件套：
   - `{Module}-00-MOC.md`
   - `{Module}-01-核心概念.md`
   - `{Module}-02-源码走读.md`
   - `{Module}-03-数据流与交互.md`
   - `{Module}-04-关键问题.md`
   - `{Module}-05-checkpoint.md`
5. 更新 `F:\源码阅读\slime_reading\Slime-progress.md` 本批为 ✅
6. 在 MOC 中用双链衔接前置批与后续批

禁止：只写路径不贴代码、编造行号、使用泛化文件名。
```

**示例（批次 08）：**
```
{NN}=08  {Module}=08-RolloutManager  {dir}=03-Rollout生成/08-RolloutManager
```

---

## 五、质量验收（维护者抽检）

| 级别 | 检查项 |
|------|--------|
| P0 | 六件套齐全；≥15 段代码；checkpoint 读者自测可勾选 |
| P1 | 02-源码走读顺序与 AGENT-DISPATCH 一致；03 含 Mermaid |
| P2 | 04 含验证建议 + tests 路径；MOC 含 nodeIds |
| P3 | 双链无断链；frontmatter tags 完整 |

**自动化（可选）：** 参考 `90_meta/audit_moc.py` 为 slime 写同类脚本。

---

## 六、推荐阅读顺序（派工顺序）

**不可并行（强依赖）：** 01→02→06→07→08→17→19→24  
**可并行（依赖满足后）：** 03∥06；12∥15；21∥24（需 20/19 完成）

**最小主链路（时间紧）：** 01, 02, 06, 08, 12, 17, 19, 21, 24, 30

---

*基线 commit: `22cdc6e1` · 对齐 [[Slime-PLAN]] [[GRAPH-BATCH-MAP]] [[UNDERSTAND-WORKFLOW]]*
