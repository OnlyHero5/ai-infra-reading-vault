---
type: batch-doc
module: 26-Checkpoint-M2HF
batch: "26"
doc_type: faq
title: "Checkpoint M2HF · 关键问题"
tags:
  - slime/batch/26
  - slime/module/checkpoint-m2hf
  - slime/doc/faq
updated: 2026-07-02
---

# Checkpoint M2HF · 关键问题

## Q1：什么时候用 bridge，什么时候用 raw？

| 场景 | 推荐 |
|------|------|
| 从 HF 预训练权重 **启动** Megatron 训练 | **bridge**（唯一支持的加载路径） |
| 快速导出 HF、Bridge 已支持你的架构 | **bridge** |
| NCCL/disk 权重同步、需与 Slime converter 完全一致 | **raw** |
| 量化导出（FP8 / compressed_tensors） | **raw** + quantizer processors |
| 新架构 Bridge 尚未支持 | **raw**（需写 converter） |

**Code（加载限制）：**

```python
## 来源：slime/backends/megatron_utils/checkpoint.py L130
    assert args.megatron_to_hf_mode == "bridge", "Only bridge mode is supported for loading HF checkpoint"
```

---

## Q2：`--hf-checkpoint` 和 `--load` 有什么区别？

- **`--load`**：训练 **恢复/初始化** 权重来源（Megatron ckpt 或 HF 全量目录）
- **`--hf-checkpoint`**：raw 模式下的 **模板目录**——提供 config/tokenizer，权重由 Megatron 转换写入

**易错：** 把 `--save-hf` 输出目录设成与 `--hf-checkpoint` 相同会被拒绝：

```python
## 来源：slime/backends/megatron_utils/hf_checkpoint_saver.py L58-L59
    if hf_checkpoint == save_path:
        raise ValueError("HF save output path must not point to the same directory as --hf-checkpoint")
```

---

## Q3：新增模型族要改哪些地方？

1. 新增 `megatron_to_hf/your_model.py`，实现 `convert_your_model_to_hf(args, name, param) -> list[tuple[str, Tensor]]`
2. 在 `_convert_to_hf_core` 注册路由（注意子串匹配顺序）
3. 若需 Bridge 加载：在 `slime_plugins/megatron_bridge` 注册
4. 跑 `tests/utils/test_hf_checkpoint_saver.py` 做 smoke test

---

## Q4：QKV 转换为什么容易出错？

Megatron GQA 把 Q/K/V 沿 **query group** 维打包；converter 必须知道 `num_query_groups`、`num_attention_heads`、`kv_channels`。

**正确思路（节选）：**

```python
## 来源：slime/backends/megatron_utils/megatron_to_hf/qwen2.py L27-L31
            param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            q_param, k_param, v_param = torch.split(param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1)
```

**易错：** 直接 `chunk(3)` 而不按 group 维 view——MoE / MLA 架构会更复杂。

---

## Q5：大模型 HF 加载为什么 patch ShardedTensor？

PyTorch 2.x 加载多分片 HF 权重时默认校验 shard metadata，千万参数级模型会卡数分钟。Slime 替换 `_init_from_local_shards_and_global_metadata` 跳过 cross-rank validation。

**Comment：** 这是性能 workaround；需确保 shard 布局本身正确。

---

## Q6：save 时多节点如何分工？

`_get_node_save_layout` 按 `actor_num_gpus_per_node` 推断 node 数，每个 node 选一个 writer rank（每 node 首 GPU）：

```python
## 来源：slime/backends/megatron_utils/hf_checkpoint_saver.py L369-L376
    gpus_per_node = int(getattr(args, "actor_num_gpus_per_node", None) or ...)
    writer_ranks = [node * gpus_per_node for node in range(num_nodes) if node * gpus_per_node < world_size]
    return len(writer_ranks), node_rank, rank in writer_ranks, writer_ranks
```

**易错：** `actor_num_nodes` 配置大于实际 world_size 推断值时会被 clamp。

---

## Q7：与 tools 目录转换脚本的关系？

| 工具 | 方向 | 与本专题关系 |
|------|------|------------|
| `tools/convert_hf_to_torch_dist.py` | HF → torch_dist | 训练前离线准备（[[05-Tools-DataPrep-00-MOC]]） |
| `checkpoint._load_checkpoint_hf` | HF → Megatron 在线 | Bridge 运行时加载 |
| `save_hf_model_to_path` | Megatron → HF | 训练中/后导出 |

三者互补，不互相替代。

---

## Q8：CI 如何验证 saver？

```python
## 来源：tests/utils/test_hf_checkpoint_saver.py L57-L61
def test_save_hf_model_direct_to_path_rejects_origin_checkpoint(tmp_path):
    ...
    with pytest.raises(ValueError, match="must not point to the same directory"):
        save_hf_model_direct_to_path(args, tmp_path, model=None)
```

本地可跑：`python -m pytest tests/utils/test_hf_checkpoint_saver.py`
