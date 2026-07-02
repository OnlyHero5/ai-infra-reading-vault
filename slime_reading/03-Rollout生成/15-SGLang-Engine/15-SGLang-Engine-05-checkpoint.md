---
type: batch-doc
module: 15-SGLang-Engine
batch: "15"
doc_type: checkpoint
title: "SGLang Engine · 验收清单"
tags:
  - slime/batch/15
  - slime/module/sglang-engine
  - slime/doc/checkpoint
updated: 2026-07-02
---

# SGLang Engine · 验收清单

---

## 读者自测（不打开 slime/）

- [ ] 能说明 `SGLangEngine` 在 RL 闭环（generate → train → update_weights）中的职责：Ray 封装 + HTTP 桥 + 权重 sync 端点
- [ ] 能画出 engine 生命周期：`start_engines` → `init` → `launch_server_process` → Router 注册 → shutdown
- [ ] 能说出 3 个核心函数/方法及其职责：
  - `launch_server_process` — spawn SGLang 子进程并 health check
  - `init_weights_update_group` — HTTP 触发 SGLang 侧加入权重 NCCL 组
  - `update_weights_from_distributed` — 传 metadata，配合训练侧 `dist.broadcast`
- [ ] 能解释 NCCL group 的 rank 布局：rank 0 = Megatron PP-source，engine i 从 `cumulative[i]+1` 起占 `engine_gpu_counts[i]` 个 rank
- [ ] 能区分 SGLang 推理 `nccl_port` 与权重 update group 的 `master_port`
- [ ] 能说明 update_weights 前为何需要 `pause_generation` + `flush_cache`

---

## 维护者检查

- [ ] frontmatter tags 含 `slime/batch/15` + `slime/doc/*`
- [ ] 文件名前缀 `15-SGLang-Engine-`，无泛化 `README` / `01-核心概念`
- [ ] Mermaid 块内无 `\n`（使用 `<br/>`）
- [ ] 双链格式 `[[15-SGLang-Engine-01-核心概念]]`，无 `./` 相对路径
- [ ] 已更新 [[Slime-progress]] 批次 15 为 ✅
- [ ] 03-数据流含 NCCL group 建立完整时序（connect → init_weights_update_group → broadcast）

---

## 快速口试参考

1. **谁创建 SGLangEngine actor？** → `RolloutManager.ServerGroup.start_engines`，`ray.remote(SGLangEngine)`。
2. **权重 NCCL 组何时建立？** → Megatron `UpdateWeightFromDistributed.connect_rollout_engines`，非 engine init 时。
3. **HTTP 与 NCCL 如何分工？** → HTTP 传 names/shapes/group；tensor 数据走 NCCL broadcast from rank 0。

---

## 相关批次

- 上游：批次 08 RolloutManager（engine 创建）
- 下游：批次 24 WeightSync-Dist（Megatron 侧完整 sync 逻辑）
- SGLang 对照：[[03-HTTP-Server-00-MOC]]
