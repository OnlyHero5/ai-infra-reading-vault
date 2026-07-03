---
type: batch-doc
module: 29-Plugins-Examples
batch: "29"
doc_type: walkthrough
title: "Plugins Examples · 源码走读"
tags:
  - slime/batch/29
  - slime/module/plugins-examples
  - slime/doc/walkthrough
updated: 2026-07-02
---

# Plugins Examples · 源码走读

## 走读顺序

1. `rollout_buffer/buffer.py` — HTTP API + 队列
2. `examples/search-r1/generate_with_search.py` — 多轮 search
3. `examples/multi_agent/rollout_with_multi_agents.py` — rollout 替换
4. `slime_plugins/models/glm5/glm5.py` — 模型插件代表

---

## 1. RolloutBuffer HTTP：write

**Code：**

```python
## 来源：slime_plugins/rollout_buffer/buffer.py L259-L268
@app.post("/buffer/write", response_model=BufferResponse)
async def write_to_buffer(request: Request):
    data = await request.json()
    item = buffer.write(data)
    return BufferResponse(success=True, message="Data has been successfully written to buffer", ...)
```

**Comment：** 写入项需含 `instance_id` 等字段；generator 侧约定。

---

## 2. get_rollout_data：训练侧拉取

**Code：**

```python
## 来源：slime_plugins/rollout_buffer/buffer.py L277-L295
@app.post("/get_rollout_data", response_model=BufferResponse)
async def get_rollout_data(request: Request):
    items = buffer.read()
    if not items["data"]:
        return BufferResponse(success=False, message="No data available to read", ...)
    buffer.buffer.temp_data = {}
    return BufferResponse(success=True, message=f"Successfully read {len(items['data'])} items", data=items)
```

**Comment：** 读走后清空 `temp_data` meta 窗口；valid group 从 `data` 移除。

---

## 3. start_rollout：后台 generator

**Code：**

```python
## 来源：slime_plugins/rollout_buffer/buffer.py L298-L328
def run_rollout(data: dict):
    generator_map = discover_generators()
    task_type = data["task_type"]
    buffer = RolloutBuffer(
        group_size=int(data["num_repeat_per_sample"]),
        task_type=task_type,
        transform_group_func=generator_info.get("transform_group", None),
        ...
    )
    generator_info["run_rollout"](data)

@app.post("/start_rollout")
async def start_rollout(request: Request, background: BackgroundTasks):
    payload = await request.json()
    background.add_task(run_rollout, payload)
    return {"message": "Rollout started"}
```

**Comment：** 每次 start 重建全局 `buffer` 对象；多任务需不同 task_type。

---

## 4. BufferQueue.get：transform + extend

**Code：**

```python
## 来源：slime_plugins/rollout_buffer/buffer.py L184-L205
    def get(self):
        valid_groups, finished_groups = self._get_valid_groups_with_timeout(del_data=True)
        for instance_id, group in valid_groups:
            transformed_group = self.transform_group_func((instance_id, group), self.task_type)
            output["data"].extend(transformed_group[1])
            if instance_id in self.data:
                self.data.pop(instance_id)
        return output
```

---

## 5. Search-R1：多轮主循环

**Code：**

```python
## 来源：examples/search-r1/generate_with_search.py L179-L244
    for _turn_idx in range(SEARCH_R1_CONFIGS["max_turns"]):
        payload = {"text": prompt_text + response, "sampling_params": sampling_params}
        if SEARCH_R1_CONFIGS["return_logprob"]:
            payload["return_logprob"] = True
        output = await post(url, payload)
        cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        ...
        next_obs, done = await execute_predictions(cur_response)
        if done:
            break
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        loss_mask += [0] * len(obs_tokens_ids)
        sample.append_response_tokens(args, tokens=obs_tokens_ids, trainable=False)
```

**Comment：** 模型输出 loss_mask=1；search 结果 observation loss_mask=0。

---

## 6. Search-R1：stop tags 修复

**Code：**

```python
## 来源：examples/search-r1/generate_with_search.py L173-L177
    _stop_tags = ["</search>", "</answer>"]
    sampling_params = {**sampling_params, "stop": list(dict.fromkeys([*_existing_stop, *_stop_tags]))}
```

**Comment：** 避免 `return_logprob=True` 时字符串 trim 破坏 token/logp 对齐。

---

## 7. execute_predictions：tool 模拟

**Code：**

```python
## 来源：examples/search-r1/generate_with_search.py L124-L142
async def execute_predictions(prediction: str) -> str:
    action, content = postprocess_predictions(prediction)
    if action == "search":
        search_results = await search(search_query)
        next_obs = f"\n\n<information>{search_results.strip()}</information>\n\n"
        done = False
    elif action == "answer":
        next_obs = ""
        done = True
    else:
        next_obs = "\nMy previous action is invalid. ..."
        done = False
    return next_obs, done
```

---

## 8. multi_agent：load_function 包装

**Code：**

```python
## 来源：examples/multi_agent/rollout_with_multi_agents.py L8-L13
MULTI_AGENT_CONFIGS = {
    "custom_multi_agent_function_path": "examples.multi_agent.agent_system.run_agent_system",
    "num_parallel": 5,
    "incorrect_reward_weight": 0.8,
    "correct_reward_weight": 1.2,
}
```

**Comment：** 通过 `setattr(args, key, value)` 注入 example 级配置。

---

## 9. glm5：skip topk 层

**Code：**

```python
## 来源：slime_plugins/models/glm5/glm5.py L47-L52
def source_compute_layer(layer_number, skip_topk_offset, topk_freq):
    layer = layer_number
    while is_skip_topk_layer(layer, skip_topk_offset, topk_freq):
        layer -= 1
    return layer
```

**Comment：** cross-layer index sharing；训练 MoE RL 时需关注 routing replay（[[23-CP-RoutingReplay-00-MOC]]/28 §18）。

---

## 10. uvicorn 启动 buffer 服务

**Code：**

```python
## 来源：slime_plugins/rollout_buffer/buffer.py L332-L340
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8889, limit_concurrency=1000, timeout_keep_alive=5)
```

**Comment：** 默认 8889；middleware 允许 1GB body。

---

## 11. Search-R1 最终 Sample 组装

**Code：**

```python
## 来源：examples/search-r1/generate_with_search.py L255-L272
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_mask
    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "stop":
            sample.status = Sample.Status.COMPLETED
```

---

## 12. examples README 生态索引

**Code：**

```markdown
## 来源：examples/README.md L3-L3
These examples provide concrete examples to leverage slime in your own RL workflow.
```

**Comment：** 多数 example 附带 shell 脚本与 README 运行说明（本专题内嵌核心 Python）。
