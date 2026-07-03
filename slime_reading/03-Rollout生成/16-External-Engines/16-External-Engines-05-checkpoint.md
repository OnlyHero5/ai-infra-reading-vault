---
type: batch-doc
module: 16-External-Engines
batch: "16"
doc_type: checkpoint
title: "External Engines · 验收清单"
tags:
  - slime/batch/16
  - slime/module/external-engines
  - slime/doc/checkpoint
updated: 2026-07-02
---

# External Engines · 验收清单

## 读者自测（不打开 slime/）

- [ ] 能说明 external engine 与 `--sglang-config` 的边界差异（谁 launch、谁 recover、多模型支持）
- [ ] 能画出启动时序：`parse_args` 探测 → `start_external_rollout_servers` → Router 注册 → `ray.get(init_handles)`
- [ ] 能说出 3 个核心符号职责：
  - `discover_external_engines` — HTTP 发现拓扑
  - `start_external_rollout_servers` — 创建零 GPU actor + Router
  - `init_http_client` — generate 异步 HTTP 通道
- [ ] 能解释 external 模式下 `rollout_num_gpus` 是逻辑容量而非 PG 占用
- [ ] 能对比 NCCL vs disk vs delta 三种权重同步在 external 部署中的选型
- [ ] 能说明为何 external 不支持 Slime fault tolerance / recover

## 闭环位置自测

在 generate → train → update_weights 三角中，本模块覆盖：

1. **generate 前**：发现 engine、启动 Router、注册 worker、初始化 HTTP 客户端
2. **generate 中**：`http_utils.post` → Router → 外部 SGLang
3. **update_weights**：经 `SGLangEngine` Ray actor 转 HTTP/NCCL 或 disk 到外部 engine（细节见[[24-WeightSync-Dist-00-MOC]]–[[25-WeightSync-Disk-00-MOC]]）

## 推荐验证命令

```bash
# E2E external PD 测试（需 GPU 环境）
pytest slime/tests/test_qwen3_4B_external_pd.py -v

# 仅验证 server_info 探测逻辑（mock HTTP）
pytest slime/tests/test_placement_group.py -k external -v
```

## 通过标准

全部读者自测项可口头回答；全部读者自测项可口头回答。
