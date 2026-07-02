---
type: batch-doc
module: 29-Plugins-Examples
batch: "29"
doc_type: concept
title: "Plugins Examples · 核心概念"
tags:
  - slime/batch/29
  - slime/module/plugins-examples
  - slime/doc/concept
updated: 2026-07-02
---

# Plugins Examples · 核心概念

## 1. slime_plugins  vs  examples

| 目录 | 定位 | 典型内容 |
|------|------|----------|
| `slime_plugins/` | 可 import 的库代码 | Megatron Bridge 注册、GLM5 模型、rollout_buffer 服务 |
| `examples/` | 端到端实验脚本 | search-r1、multi_agent、fully_async、coding_agent_rl |

plugins **被** 核心或 examples import；examples **通过** `--*-path` 被 Slime 调用。

---

## 2. rollout_buffer：外部 Rollout 队列

**Explain：** FastAPI 服务维护按 `instance_id` 分组的 sample 队列；外部 generator 写入，训练侧 HTTP 拉 batch。

**Code：**

```python
# 来源：slime_plugins/rollout_buffer/buffer.py L14-L18
app = FastAPI(title="Rollout Buffer Server", debug=True)

def default_is_valid_group(group_data, min_valid_group_size, task_type):
    instance_id, samples = group_data
    return len(samples) >= min_valid_group_size
```

**Comment：**

- 默认 group 大小由 `num_repeat_per_sample` 控制
- `discover_generators()` 扫描 `generator/*.py` 的 `TASK_TYPE` + `run_rollout`

---

## 3. BufferQueue 双缓冲

**Explain：** `data` 持久累积；`temp_data` 供 meta 统计；`get()` 取出 valid group 后清空对应 instance。

**Code：**

```python
# 来源：slime_plugins/rollout_buffer/buffer.py L145-L160
    def append(self, item):
        instance_id = item["instance_id"]
        if instance_id not in self.temp_data:
            self.temp_data[instance_id] = [copy.deepcopy(item)]
        else:
            self.temp_data[instance_id].append(copy.deepcopy(item))
        if instance_id not in self.data:
            self.data[instance_id] = [item]
        else:
            self.data[instance_id].append(item)
```

---

## 4. Search-R1：custom_generate 模式

**Explain：** 保留默认 `sglang_rollout` 外层循环；仅替换 per-sample 多轮 search + answer。

**关键配置：**

```python
# 来源：examples/search-r1/generate_with_search.py L14-L37
SEARCH_R1_CONFIGS = {
    "max_turns": 2,
    "search_backend": "local",
    "return_logprob": True,
    "format_score": 0.2,
}
```

**CLI 典型用法：**

```bash
--custom-generate-function-path examples.search_r1.generate_with_search.generate
--custom-rm-path examples.search_r1.generate_with_search.reward_func
```

---

## 5. multi_agent：rollout_function 模式

**Explain：** 整段替换 generate_rollout；内部 `load_function` 调自定义 multi-agent 系统，返回 shuffled `list[Sample]`。

**Code：**

```python
# 来源：examples/multi_agent/rollout_with_multi_agents.py L16-L33
async def generate_with_multi_agents(args, sample, sampling_params, evaluation=False) -> list[Sample]:
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    custom_multi_agent_func = load_function(args.custom_multi_agent_function_path)
    samples = await custom_multi_agent_func(args, sample)
    random.shuffle(samples)
    return samples
```

**Comment：** 需配合 `--rollout-function-path` 指向包装函数（见 example README）。

---

## 6. examples 目录地图（节选）

**Code：**

```markdown
# 来源：examples/README.md L7-L18
- eval_multi_task: 多任务 eval 配置
- fully_async: 全异步 rollout
- multi_agent: 多 agent RL
- search-r1: Search-R1 多轮 tool
- delta_weight_sync: 非 colocate delta 同步
- coding_agent_rl: sandbox + test reward 完整 agent RL
```

---

## 7. slime_plugins/models/glm5：模型扩展代表

**Explain：** 非 example，而是 **新架构 Megatron 模块**（DSA / sparse MLA）；训练时需配合 Bridge 或 megatron_to_hf converter。

**Code：**

```python
# 来源：slime_plugins/models/glm5/glm5.py L37-L44
def is_skip_topk_layer(layer_number, skip_topk_offset, topk_freq):
    """Whether the layer reuses a previous layer's top-k."""
    return (max(layer_number - skip_topk_offset, 0) % topk_freq) != 0
```

**Comment：** 说明 plugins 不仅是脚本，也可深度改模型 forward。

---

## 8. Search-R1 的 logprob 不变量

**Explain：** 开启 `return_logprob` 时 **禁止** 对 engine 返回字符串做 postprocess 再 tokenize。

**Code：**

```python
# 来源：examples/search-r1/generate_with_search.py L93-L98
# IMPORTANT: When we need to collect log probabilities (logp), we CANNOT do any postprocessing
# on the strings returned from the inference engine (sglang).
# Therefore, postprocess_responses is only used when return_logprob=False.
```

**Comment：** 用 `stop` 参数在边界截断，而非事后 trim 字符串（同文件 L164-L177）。

---

## 9. reward_func 与 customization

**Code：**

```python
# 来源：examples/search-r1/generate_with_search.py L277-L293
async def reward_func(args, sample, **kwargs):
    score = compute_score_em(
        solution_str=sample.prompt + sample.response,
        ground_truth=sample.label["ground_truth"],
        format_score=SEARCH_R1_CONFIGS["format_score"],
    )
    return score
```

**Comment：** EM + format 分；挂 `--custom-rm-path` 即可。

---

## 10. generator 自动发现

**Code：**

```python
# 来源：slime_plugins/rollout_buffer/buffer.py L54-L87
def discover_generators():
    generator_dir = pathlib.Path(__file__).parent / "generator"
    for file_path in glob.glob(str(generator_dir / "*.py")):
        ...
        if not hasattr(module, "TASK_TYPE"):
            continue
        if not hasattr(module, "run_rollout"):
            continue
        task_type = module.TASK_TYPE
        generator_map[task_type] = generator_info
```

**Comment：** 新任务类型 = 新 py 文件 + `TASK_TYPE` 常量 + `run_rollout(data)`。
