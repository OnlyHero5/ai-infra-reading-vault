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
updated: 2026-07-05
---

# Plugins Examples · 源码走读

> 走读主线：`examples/README.md` 给出生态入口，`slime_plugins/rollout_buffer/buffer.py` 展示 HTTP rollout buffer 插件，`examples/search-r1/generate_with_search.py` 展示工具调用式 rollout，`examples/multi_agent/rollout_with_multi_agents.py` 展示 generate 函数替换，`slime_plugins/models/glm5/glm5.py` 展示模型结构插件。

---

## 1. examples 生态入口

### 1.1 README 把 examples 定位为可复用工作流模板

问题与约束：
- 插件与示例分散在多个目录中，读者需要先区分哪些是 rollout 生成、哪些是评测、权重同步、低精度或 agentic workflow。

设计选择：
- README 用目录索引列出每个 example 的主题，并说明这些示例用于把 slime 嵌入自己的 RL workflow。

Explain：
这不是运行时代码，但它定义了示例生态的入口：Search-R1、多 agent、delta weight sync、low precision、retool、Tau-bench 等都被放在 examples 下，作为可验证或可迁移的工程模板。

来源：examples/README.md L3-L20

Code：

```markdown
These examples provide concrete examples to leverage slime in your own RL workflow. Some examples are just demonstrative, but most of them are verifiable with a concrete performance score.

## Directory Structure

- **[fully_async](./fully_async)**: Demonstrates fully asynchronous rollout generation for higher efficiency.
- **[multi_agent](./multi_agent)**: Example of running multi-agent RL with `slime`.
- **[delta_weight_sync](./delta_weight_sync)**: Non-colocated weight sync that ships only the changed bytes over a shared filesystem (training/inference disaggregation), reloading via the vanilla `update_weights_from_disk` path.
- **[search-r1](./search-r1)**: A minimal reproduction of Search-R1, featuring multi-turn conversation and tool-calling.
- **[tau-bench](./tau-bench)**: Training in an agentic multi-turn tool use environment (Tau-bench).
```

代码逻辑：
- README 先说明 examples 的用途，再用目录列表做导航。
- 每个目录条目绑定一个可复用场景，而不是只列文件名。
- Search-R1 和 multi_agent 对应本篇后面的 rollout 扩展例子。

为什么这样写：
- slime 的扩展点很多，用 examples 聚合比把所有插件写进核心文档更轻。
- 目录索引能让用户先按 workflow 选择入口，再深入代码。

不变量与失败模式：
- README 只是索引，不能替代每个 example 的运行参数和依赖说明。
- 新增 example 后若不更新索引，读者会漏掉可复用模板。

Comment：
这篇源码走读重点选取 README 中最能代表扩展机制的几个例子展开。

---

## 2. RolloutBuffer 服务

### 2.1 discover_generators 自动发现 generator 模块

问题与约束：
- rollout buffer 需要支持不同 task_type 的生成逻辑；如果每新增一个任务都改服务主文件，插件边界会变重。

设计选择：
- `discover_generators` 扫描 `rollout_buffer/generator/*.py`，要求模块提供 `TASK_TYPE` 和 `run_rollout`，可选提供 transform/valid/meta 函数。

Explain：
函数动态加载 generator 文件，把 `TASK_TYPE` 映射到一组函数句柄。缺少必要字段的模块会 warning 后跳过，可选函数没有就填 `None`，后续由 `BufferQueue` 使用默认实现。

来源：slime_plugins/rollout_buffer/buffer.py L54-L109

Code：

```python
def discover_generators():
    generator_map = {}
    generator_dir = pathlib.Path(__file__).parent / "generator"

    for file_path in glob.glob(str(generator_dir / "*.py")):
        if file_path.endswith("__init__.py"):
            continue
        try:
            spec = importlib.util.spec_from_file_location("generator_module", file_path)
            if spec is None or spec.loader is None:
                print(f"Warning: Could not load spec for {file_path}")
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if not hasattr(module, "TASK_TYPE"):
                print(f"Warning: {file_path} does not define TASK_TYPE constant")
                continue
            if not hasattr(module, "run_rollout"):
                print(f"Warning: {file_path} does not define run_rollout function")
                continue
            task_type = module.TASK_TYPE
            generator_info = {
                "module": module,
                "file_path": file_path,
                "run_rollout": module.run_rollout,
            }
            for func_name in [
                "transform_group",
                "is_valid_group",
                "get_group_data_meta_info",
            ]:
                generator_info[func_name] = getattr(module, func_name, None)
            generator_map[task_type] = generator_info
        except Exception as e:
            print(f"Error loading generator from {file_path}: {str(e)}")
            continue
    return generator_map
```

代码逻辑：
- generator 文件按目录扫描，不需要集中注册表。
- `TASK_TYPE` 是路由 key，`run_rollout` 是必要入口。
- 三个可选函数决定成组转换、有效性判断和 meta 统计。

为什么这样写：
- rollout 生成逻辑变化快，动态发现可以把任务扩展限制在 generator 目录。
- 必需/可选函数分层，使简单任务只实现最小入口即可。

不变量与失败模式：
- 每个 generator 的 `TASK_TYPE` 必须唯一，否则后发现的模块会覆盖同名 key。
- import 失败或缺少 `run_rollout` 会让对应 task_type 无法启动。

Comment：
rollout buffer 插件的扩展点不在 FastAPI endpoint，而在 generator discovery。

### 2.2 middleware 与 BufferResponse 定义服务边界

问题与约束：
- rollout 数据可能包含大批 token、logprob 和 meta 信息，HTTP body 默认限制可能过小；响应也需要统一 success/message/data 形态。

设计选择：
- FastAPI middleware 把请求 body 限制设为 1GB，`BufferResponse` 用 Pydantic 定义统一响应结构。

Explain：
服务初始化后，所有请求都会经过 `set_body_size`。endpoint 返回 `BufferResponse`，使写入、读取失败和读取成功都能用同一响应 schema 表达。

来源：slime_plugins/rollout_buffer/buffer.py L112-L122

Code：

```python
@app.middleware("http")
async def set_body_size(request: Request, call_next):
    request._body_size_limit = 1_073_741_824  # 1GB
    response = await call_next(request)
    return response


class BufferResponse(BaseModel):
    success: bool
    message: str = ""
    data: dict[str, Any] | None = None
```

代码逻辑：
- middleware 在请求进入 endpoint 前设置 `_body_size_limit`。
- `BufferResponse` 的 `data` 允许为空，适合错误响应。
- success/message/data 三元组覆盖写入和读取两类 API。

为什么这样写：
- rollout 数据体积会随 batch、repeat 和 logprob 开关增长，1GB 上限避免大样本直接被 HTTP 层拒绝。
- 响应 schema 固定后，训练侧 client 不需要为每个 endpoint 写不同解析逻辑。

不变量与失败模式：
- `_body_size_limit` 是框架内部属性，FastAPI/Starlette 版本变化可能影响行为。
- 大 body 上限不等于内存安全，服务端仍要控制并发和 batch 大小。

Comment：
这段体现了插件示例的工程取舍：为了快速接入 rollout，先把 HTTP 边界做宽。

### 2.3 BufferQueue 初始化和 append 同时维护 data/temp_data

问题与约束：
- buffer 既要按 `instance_id` 聚合训练数据，又要保留本次读取窗口的 meta 统计数据。

设计选择：
- `BufferQueue` 维护 `data`、`temp_data` 和 `group_timestamps` 三个字典；append 时把原始 item 放入 `data`，把深拷贝放入 `temp_data`。

Explain：
`data` 用于实际训练侧读取与删除；`temp_data` 用于 meta 信息统计，避免读取前的转换或删除影响统计窗口。handler 函数没有传入时使用默认实现。

来源：slime_plugins/rollout_buffer/buffer.py L125-L160

Code：

```python
class BufferQueue:
    def __init__(
        self,
        group_size,
        task_type="math",
        transform_group_func=None,
        is_valid_group_func=None,
        get_group_data_meta_info_func=None,
    ):
        self.data = {}
        self.temp_data = {}
        self.group_timestamps = {}
        self.group_size = group_size
        self.task_type = task_type
        self.is_valid_group_func = is_valid_group_func or default_is_valid_group
        self.get_group_data_meta_info_func = get_group_data_meta_info_func or default_get_group_data_meta_info
        self.transform_group_func = transform_group_func or (lambda group, task_type: group)

    def append(self, item):
        instance_id = item["instance_id"]
        current_time = time.time()
        self.group_timestamps[instance_id] = current_time
        if instance_id not in self.temp_data:
            self.temp_data[instance_id] = [copy.deepcopy(item)]
        else:
            self.temp_data[instance_id].append(copy.deepcopy(item))
        if instance_id not in self.data:
            self.data[instance_id] = [item]
        else:
            self.data[instance_id].append(item)
```

代码逻辑：
- `instance_id` 是 group key。
- timestamp 每次 append 更新，为 timeout 逻辑预留。
- `temp_data` 使用深拷贝，`data` 保存原 item。

为什么这样写：
- 训练读取和 meta 统计的生命周期不同，分开存储能避免相互污染。
- 默认函数让 generator 只在需要定制时覆盖行为。

不变量与失败模式：
- 写入 item 必须包含 `instance_id`。
- `copy.deepcopy` 对大样本有额外成本，repeat 很大时会放大内存占用。

Comment：
`data/temp_data` 双缓冲是理解后面 `get_rollout_data` 清空 meta 窗口的前提。

### 2.4 BufferQueue.get 做有效组筛选、转换和删除

问题与约束：
- 训练侧需要读取“已经成组”的 rollout 数据；未达到 group_size 的样本不能提前进入训练。

设计选择：
- `get` 先计算 meta，再调用 `_get_valid_groups_with_timeout(del_data=True)`，对每个有效组执行 `transform_group_func` 并扩展到输出 `data`。

Explain：
有效组来自 `is_valid_group_func`。读取时会把 `finished_groups` 放入 meta，并在转换后从 `self.data` 中删除对应 instance，避免同一组重复训练。

来源：slime_plugins/rollout_buffer/buffer.py L162-L206

Code：

```python
def _get_valid_groups_with_timeout(self, del_data=False):
    valid_groups = {}
    timed_out_groups = {}
    finished_groups = []
    for instance_id, group_data in self.data.items():
        if self.is_valid_group_func((instance_id, group_data), self.group_size, self.task_type):
            valid_groups[instance_id] = group_data
    all_valid_groups = {**valid_groups, **timed_out_groups}
    return all_valid_groups, finished_groups

def get(self):
    output = {"data": [], "meta_info": {}}
    meta_info = self.get_group_data_meta_info_func(self.temp_data)
    output["meta_info"] = meta_info
    valid_groups, finished_groups = self._get_valid_groups_with_timeout(del_data=True)
    output["meta_info"]["finished_groups"] = finished_groups
    valid_groups = list(valid_groups.items())
    for instance_id, group in valid_groups:
        transformed_group = self.transform_group_func((instance_id, group), self.task_type)
        output["data"].extend(transformed_group[1])
        if instance_id in self.data:
            self.data.pop(instance_id)
    return output
```

代码逻辑：
- meta 信息基于 `temp_data`，先于转换与删除计算。
- valid group 以 `(instance_id, group)` 形式进入 transform。
- transform 返回的第二项被追加到训练数据列表。

为什么这样写：
- transform 放在读取时执行，generator 可以输出原始样本，训练侧按任务类型统一整理。
- 删除已读 group 避免重复消费。

不变量与失败模式：
- `transform_group_func` 必须返回二元结构，且第二项可迭代。
- 当前 timeout 分支保留变量但没有实际填充，依赖 timeout 行为时需要补实现。

Comment：
buffer 的消费语义是“按 group 一次性读取并删除”，不是流式逐条 pop。

### 2.5 RolloutBuffer 用锁保护 write/read

问题与约束：
- FastAPI background rollout 和训练侧读取可能并发访问同一个 buffer。

设计选择：
- `RolloutBuffer` 用 `RLock` 和 `Condition` 包住 `BufferQueue`，write 后通知，read 时在锁内检查长度并获取数据。

Explain：
`write` 在锁内 append 并累加 `total_written`；`read` 在 condition 锁内检查有效数据数量，为空则返回空结果，否则调用 `buffer.get` 并累加 `total_read`。

来源：slime_plugins/rollout_buffer/buffer.py L216-L253

Code：

```python
class RolloutBuffer:
    def __init__(...):
        self.buffer = BufferQueue(...)
        self.lock = threading.RLock()
        self.not_empty = threading.Condition(self.lock)
        self.total_written = 0
        self.total_read = 0
        self.task_type = task_type

    def write(self, data):
        with self.lock:
            self.buffer.append(data)
            self.total_written += 1
            self.not_empty.notify_all()
        return data

    def read(self):
        with self.not_empty:
            if len(self.buffer) == 0:
                return {"data": [], "meta_info": {}}
            result = self.buffer.get()
            self.total_read += len(result["data"])
            return result
```

代码逻辑：
- write 和 read 都在同一把锁保护下访问内部 queue。
- `len(self.buffer)` 只统计有效组中的样本数。
- read 不清空 `temp_data`，清空动作由 HTTP endpoint 控制。

为什么这样写：
- buffer 是全局对象，必须防止写入和读取同时修改字典。
- 把 `temp_data` 生命周期留给 endpoint，可以让普通 read 保留 meta 窗口。

不变量与失败模式：
- `Condition` 当前没有 wait 循环，只用于锁和 notify；读取不会阻塞等待数据。
- 多任务共享全局 buffer 时，后启动任务会替换旧 buffer。

Comment：
这是一层轻量线程安全包装，不是完整消息队列。

### 2.6 /buffer/write 接收 generator 写入

问题与约束：
- generator 需要通过 HTTP 把 rollout 样本写入 buffer；服务端要把解析、写入和错误报告包成 endpoint。

设计选择：
- `/buffer/write` 从 request JSON 读取单个 item，调用 `buffer.write`，成功后返回写入 item 和 meta 字符串。

Explain：
endpoint 捕获异常并打印 traceback，失败时返回 500。成功响应中 `data` 字段仍保持 `{"data": [...], "meta_info": ...}` 结构，和读取响应兼容。

来源：slime_plugins/rollout_buffer/buffer.py L259-L274

Code：

```python
@app.post("/buffer/write", response_model=BufferResponse)
async def write_to_buffer(request: Request):
    try:
        data = await request.json()
        item = buffer.write(data)
        return BufferResponse(
            success=True,
            message="Data has been successfully written to buffer",
            data={"data": [item], "meta_info": "write to buffer"},
        )
    except Exception as e:
        print(f"Write failed: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Write failed: {str(e)}") from e
```

代码逻辑：
- HTTP JSON 直接作为 buffer item。
- `buffer.write` 返回原 item，便于响应回显。
- 失败路径抛出 `HTTPException`。

为什么这样写：
- 写入 endpoint 保持薄封装，具体 group 逻辑放在 `RolloutBuffer/BufferQueue`。
- 回显写入 item 有助于调试 generator 输出。

不变量与失败模式：
- JSON 必须包含 `BufferQueue.append` 需要的字段，尤其是 `instance_id`。
- endpoint 没有做 schema 校验，字段错误会在写入或 transform 阶段暴露。

Comment：
这是 generator 到训练 buffer 的最小 HTTP 接口。

### 2.7 /get_rollout_data 读取后清空 temp_data 窗口

问题与约束：
- 训练侧周期性拉取 rollout 数据；如果当前没有有效组，需要返回 meta 但不报错。

设计选择：
- endpoint 调用 `buffer.read`，空数据返回 `success=False`，非空时清空 `buffer.buffer.temp_data` 并返回数据。

Explain：
`temp_data` 是 meta 统计窗口，读取成功后清空，下一轮 meta 只反映下一批写入。`data` 中包含训练样本和 meta_info。

来源：slime_plugins/rollout_buffer/buffer.py L277-L295

Code：

```python
@app.post("/get_rollout_data", response_model=BufferResponse)
async def get_rollout_data(request: Request):
    items = buffer.read()

    if not items["data"]:
        return BufferResponse(
            success=False,
            message="No data available to read",
            data={"data": [], "meta_info": items["meta_info"]},
        )

    print(f"return {len(items['data'])} items and save them to local")
    buffer.buffer.temp_data = {}

    return BufferResponse(
        success=True,
        message=f"Successfully read {len(items['data'])} items",
        data=items,
    )
```

代码逻辑：
- 空读不会清空 temp_data。
- 成功读之后清空 meta 窗口。
- 返回消息包含读取样本数。

为什么这样写：
- 空读时保留 meta 有助于训练侧观察当前积累状态。
- 成功读后清空 temp_data，避免下一次 meta 混入已消费样本。

不变量与失败模式：
- `buffer.read` 返回结构必须包含 `data/meta_info`。
- 如果训练侧长期不读取，`temp_data` 会持续增长。

Comment：
读取 endpoint 同时承担“消费数据”和“推进 meta 窗口”的职责。

### 2.8 /start_rollout 后台启动 generator 并重建全局 buffer

问题与约束：
- rollout 生成可能耗时很长，不能阻塞 HTTP 请求；不同 task_type 还需要不同 group/transform/valid 函数。

设计选择：
- `/start_rollout` 用 FastAPI `BackgroundTasks` 调用 `run_rollout`；`run_rollout` 自动发现 generator，并用 payload 中的 repeat 数重建全局 `RolloutBuffer`。

Explain：
`run_rollout` 先根据 `task_type` 找 generator，找不到时打印可用列表并返回。找到后用 generator 的可选 hook 初始化 buffer，再调用 generator 的 `run_rollout(data)`。

来源：slime_plugins/rollout_buffer/buffer.py L298-L329

Code：

```python
def run_rollout(data: dict):
    global buffer
    generator_map = discover_generators()

    task_type = data["task_type"]
    if task_type not in generator_map:
        print(f"Error: No generator found for task_type '{task_type}'")
        print(f"Available generators: {list(generator_map.keys())}")
        return

    generator_info = generator_map[task_type]
    buffer = RolloutBuffer(
        group_size=int(data["num_repeat_per_sample"]),
        task_type=task_type,
        transform_group_func=generator_info.get("transform_group", None),
        is_valid_group_func=generator_info.get("is_valid_group"),
        get_group_data_meta_info_func=generator_info.get("get_group_data_meta_info"),
    )
    generator_info["run_rollout"](data)

@app.post("/start_rollout")
async def start_rollout(request: Request, background: BackgroundTasks):
    payload = await request.json()
    background.add_task(run_rollout, payload)
    return {"message": "Rollout started"}
```

代码逻辑：
- 每次 start 都刷新 generator map。
- `num_repeat_per_sample` 决定 group size。
- generator hook 注入到新 buffer。
- background task 立即返回启动响应。

为什么这样写：
- rollout 生成与训练读取解耦，HTTP 请求不必等待完整 rollout。
- 重建 buffer 能让每个 task_type 使用自己的 grouping 语义。

不变量与失败模式：
- 该实现只有一个全局 buffer，同时启动多个任务会互相覆盖。
- payload 必须包含 `task_type` 和 `num_repeat_per_sample`。

Comment：
这个 endpoint 是示例级实现，适合单任务实验；生产多任务需要隔离 buffer。

### 2.9 uvicorn 默认启动 8889 端口

问题与约束：
- rollout buffer 需要独立 HTTP 服务供 generator 写入、训练侧读取。

设计选择：
- 文件作为脚本执行时直接 `uvicorn.run(app, host="0.0.0.0", port=8889, limit_concurrency=1000, timeout_keep_alive=5)`。

Explain：
默认监听所有网卡的 8889 端口，并设置连接并发上限和 keep-alive timeout。示例没有在代码里读取配置文件或命令行参数。

来源：slime_plugins/rollout_buffer/buffer.py L332-L340

Code：

```python
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8889,
        limit_concurrency=1000,  # Connection concurrency limit
        timeout_keep_alive=5,  # Keep-alive timeout,
    )
```

代码逻辑：
- 只有直接运行文件时启动服务。
- host/port 固定在示例代码中。
- 并发限制交给 uvicorn。

为什么这样写：
- 插件示例强调开箱即用，固定端口降低启动成本。
- 训练脚本和 generator 可以约定一个简单 HTTP 地址。

不变量与失败模式：
- 8889 端口被占用时服务启动失败。
- 多机部署时必须确保训练侧能访问该 host/port。

Comment：
这段说明 rollout buffer 是一个独立服务，而不是训练进程内对象。

---

## 3. Search-R1 rollout 示例

### 3.1 SEARCH_R1_CONFIGS 控制多轮与 logprob 采集

问题与约束：
- Search-R1 rollout 同时需要控制最大轮数、检索并发、检索后端和是否采集 logprob。

设计选择：
- 用模块级 `SEARCH_R1_CONFIGS` 集中配置 max turns、topk、backend、return_logprob 和 reward 格式分。

Explain：
`SEMAPHORE` 从 `search_concurrency` 构造，限制并发 search 请求。`return_logprob=True` 时后续 generate 会从 SGLang 返回的 token logprob 中构造 TIS 所需数据。

来源：examples/search-r1/generate_with_search.py L14-L41

Code：

```python
SEARCH_R1_CONFIGS = {
    "max_turns": 2,
    "topk": 3,
    "search_concurrency": 256,
    "search_backend": "local",
    "local": {
        "search_url": "http://127.0.0.1:8000/retrieve",
        "proxy": None,
    },
    "google": {
        "api_key": "your_api_key_here",
        "snippet_only": True,
        "proxy": None,
    },
    "return_logprob": True,
    "format_score": 0.2,
}

SEMAPHORE = asyncio.Semaphore(SEARCH_R1_CONFIGS["search_concurrency"])
```

代码逻辑：
- `max_turns` 限制生成-工具循环次数。
- `search_backend` 决定走 local 还是 Google 配置。
- `return_logprob` 影响 payload 和样本字段。

为什么这样写：
- 示例把算法开关集中在模块顶部，便于用户复制后修改。
- 并发信号量是全局的，避免每个 sample 自己无限制调用检索服务。

不变量与失败模式：
- local search 服务地址必须可达。
- Google 后端需要用户替换 API key。
- `return_logprob=True` 时推理服务必须返回 `output_token_logprobs`。

Comment：
Search-R1 示例不是只扩展 reward，它重写了 rollout 生成过程。

### 3.2 execute_predictions 把模型输出解释为 search/answer/invalid

问题与约束：
- 多轮工具调用需要把模型生成的文本转成环境动作，并给下一轮生成提供 observation。

设计选择：
- `execute_predictions` 调用 `postprocess_predictions` 解析 action/content，search 动作调用检索，answer 动作结束，其他动作返回格式纠错提示。

Explain：
search 结果被包装进 `<information>...</information>` 作为下一轮 observation；answer 返回空 observation 且 `done=True`；invalid action 返回自然语言提示并继续。

来源：examples/search-r1/generate_with_search.py L124-L142

Code：

```python
async def execute_predictions(prediction: str) -> str:
    action, content = postprocess_predictions(prediction)

    if action == "search":
        search_query = content
        async with SEMAPHORE:
            search_results = await search(search_query)
        next_obs = f"\n\n<information>{search_results.strip()}</information>\n\n"
        done = False
    elif action == "answer":
        next_obs = ""
        done = True
    else:
        next_obs = "\nMy previous action is invalid. If I want to search, I should put the query between <search> and </search>. If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n"
        done = False

    return next_obs, done
```

代码逻辑：
- search 使用全局 semaphore 限流。
- answer 不追加 observation，直接结束当前 sample。
- invalid action 把纠错说明作为下一轮输入。

为什么这样写：
- 工具调用环境必须把不可训练的外部观测显式加入 trajectory。
- invalid action 不直接终止，可以给模型一次自我修正机会。

不变量与失败模式：
- `postprocess_predictions` 必须能解析 `<search>` 和 `<answer>` 标签。
- 检索失败会阻塞当前 sample 的下一轮 observation。

Comment：
这里是工具环境的最小状态转移函数。

### 3.3 stop tags 修复 token/logprob 对齐

问题与约束：
- 当 `return_logprob=True` 时，后处理裁剪字符串会破坏 token 与 logprob 对齐；但没有 stop tag 又会让模型在 `</search>` 或 `</answer>` 后继续生成。

设计选择：
- 在 sampling params 中合并已有 stop 和 `</search>、</answer>`，让推理服务在边界处停止。

Explain：
代码保留已有 stop，字符串 stop 会转成列表，再用 `dict.fromkeys` 去重。注释说明该修复的目标是避免 trailing junk 进入训练 trajectory。

来源：examples/search-r1/generate_with_search.py L164-L177

Code：

```python
_stop_tags = ["</search>", "</answer>"]
_existing_stop = sampling_params.get("stop") or []
if isinstance(_existing_stop, str):
    _existing_stop = [_existing_stop]
sampling_params = {**sampling_params, "stop": list(dict.fromkeys([*_existing_stop, *_stop_tags]))}
```

代码逻辑：
- 读取调用方传入的 stop。
- 统一成 list。
- 追加工具边界标签并去重。

为什么这样写：
- stop 交给推理引擎处理，可以保留 token/logprob 原生对齐。
- 去重避免重复 stop 字符串污染 sampling 参数。

不变量与失败模式：
- 推理服务必须支持字符串 stop。
- 若 stop tag 被模型 tokenizer 分割，仍要依赖服务端正确处理 stop 序列。

Comment：
这个小修复直接影响可训练 token 和 logprob 的一一对应。

### 3.4 generate 主循环交替生成模型 token 与 observation token

问题与约束：
- Search-R1 trajectory 混合了模型输出和检索 observation；训练时只应对模型输出计算 loss，observation 不应训练。

设计选择：
- 每轮调用 SGLang `/generate`，模型输出 token 追加 `loss_mask=1`，工具 observation token 追加 `loss_mask=0`。

Explain：
循环最多执行 `max_turns`。开启 logprob 时直接使用 `output_token_logprobs` 中的 token id 和 logprob，关闭时才对字符串做 postprocess 和重新 tokenize。生成后调用 `execute_predictions`，若未完成则把 observation 追加到 response。

来源：examples/search-r1/generate_with_search.py L179-L244

Code：

```python
for _turn_idx in range(SEARCH_R1_CONFIGS["max_turns"]):
    payload = {
        "text": prompt_text + response,
        "sampling_params": sampling_params,
    }
    if SEARCH_R1_CONFIGS["return_logprob"]:
        payload["return_logprob"] = True

    output = await post(url, payload)
    if output["meta_info"]["finish_reason"]["type"] == "abort":
        sample.status = Sample.Status.ABORTED
        return sample

    cur_response = output["text"]
    if SEARCH_R1_CONFIGS["return_logprob"]:
        if "output_token_logprobs" not in output["meta_info"]:
            raise RuntimeError("output_token_logprobs not found in output meta_info.")
        cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        cur_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    else:
        cur_response = postprocess_responses(cur_response)
        cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]

    response += cur_response
    response_token_ids += cur_response_token_ids
    loss_mask += [1] * len(cur_response_token_ids)
    sample.append_response_tokens(args, tokens=cur_response_token_ids, trainable=True, ...)

    next_obs, done = await execute_predictions(cur_response)
    if done:
        break
    obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
    response += next_obs
    response_token_ids += obs_tokens_ids
    loss_mask += [0] * len(obs_tokens_ids)
    sample.append_response_tokens(args, tokens=obs_tokens_ids, trainable=False)
```

代码逻辑：
- payload 文本是 prompt 加当前累计 response。
- 模型输出 token 作为 trainable response 追加。
- observation token 作为非 trainable response 追加。
- abort 直接返回 aborted sample。

为什么这样写：
- 训练需要完整上下文复现 rollout，但 loss 只应覆盖模型自己生成的部分。
- logprob 模式下不重新 tokenize，避免 token id 和 logprob 长度错位。

不变量与失败模式：
- `output_token_logprobs` 格式必须是可取 `[logprob, token_id, ...]` 的列表。
- observation 必须非空；空 observation 只允许 answer done 分支。

Comment：
这段是工具调用 rollout 的核心：同一 response 序列里显式区分可训练和不可训练 token。

### 3.5 最终 Sample 组装状态和 logprob

问题与约束：
- 多轮循环结束后，需要把 token、response、loss_mask、状态和可选 logprob 回填到 `Sample`，供训练后端消费。

设计选择：
- 循环外统一设置 `sample.tokens/response_length/response/loss_mask/prompt`，并根据 finish reason 设置 status。

Explain：
开启 logprob 时，`rollout_log_probs` 写入 sample。finish reason 为 length/abort/stop 分别映射到 truncated/aborted/completed。

来源：examples/search-r1/generate_with_search.py L255-L274

Code：

```python
sample.tokens = prompt_tokens_ids + response_token_ids
sample.response_length = len(response_token_ids)
sample.response = response
sample.loss_mask = loss_mask
sample.prompt = prompt_text

if SEARCH_R1_CONFIGS["return_logprob"]:
    sample.rollout_log_probs = rollout_log_probs if rollout_log_probs else None

match output["meta_info"]["finish_reason"]["type"]:
    case "length":
        sample.status = Sample.Status.TRUNCATED
    case "abort":
        sample.status = Sample.Status.ABORTED
    case "stop":
        sample.status = Sample.Status.COMPLETED

return sample
```

代码逻辑：
- `tokens` 包含 prompt 与完整 response token。
- `response_length` 只统计 response 部分。
- status 从推理服务 finish reason 转换为 Slime Sample 状态。

为什么这样写：
- 训练侧依赖 `Sample` 的统一字段，不应理解 Search-R1 内部循环。
- status 映射让 reward/filter 逻辑可以复用普通 rollout contract。

不变量与失败模式：
- `loss_mask` 长度必须等于 response token 数。
- 如果循环没有成功拿到 `output`，最终 status 分支会缺少 finish reason。

Comment：
最终 Sample 组装是 example 与 Slime 核心训练管线的接口边界。

---

## 4. 其他扩展例子

### 4.1 multi_agent 用 load_function 替换 rollout 生成函数

问题与约束：
- 多 agent rollout 通常不是单次 `/generate`，而是外部 agent system 返回多个候选 sample。

设计选择：
- 示例把 agent system 函数路径写入 `MULTI_AGENT_CONFIGS`，在 generate 函数中用 `load_function` 动态加载并调用。

Explain：
`generate_with_multi_agents` 创建 tokenizer，设置上下文长度和 sampling params，然后把配置项写回 `args`。自定义函数返回 sample 列表后，示例随机打乱再返回。

来源：examples/multi_agent/rollout_with_multi_agents.py L8-L33

Code：

```python
MULTI_AGENT_CONFIGS = {
    "custom_multi_agent_function_path": "examples.multi_agent.agent_system.run_agent_system",
    "num_parallel": 5,
    "incorrect_reward_weight": 0.8,
    "correct_reward_weight": 1.2,
}

async def generate_with_multi_agents(args, sample: Sample, sampling_params, evaluation=False) -> list[Sample]:
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    max_context_length = args.rollout_max_context_len if not evaluation else args.eval_max_context_len
    args.sampling_params = sampling_params
    args.rollout_max_context_len = max_context_length
    args.tokenizer = tokenizer

    for key, value in MULTI_AGENT_CONFIGS.items():
        setattr(args, key, value)

    custom_multi_agent_func = load_function(args.custom_multi_agent_function_path)
    samples = await custom_multi_agent_func(args, sample)
    random.shuffle(samples)
    return samples
```

代码逻辑：
- 配置以 `setattr` 注入 args。
- 自定义函数路径通过字符串动态加载。
- 返回类型是 `list[Sample]`，允许一个输入扩展成多个 rollout。

为什么这样写：
- agent system 的依赖和控制流变化大，动态函数路径比在核心 rollout 中硬编码更灵活。
- 随机 shuffle 可以减少同一输入生成的样本顺序偏置。

不变量与失败模式：
- `custom_multi_agent_function_path` 必须可 import 且返回 async callable。
- 自定义函数必须返回符合 Slime contract 的 `Sample` 列表。

Comment：
这个例子展示的是“替换 generate 函数”，不是修改 Slime 核心训练循环。

### 4.2 GLM5 插件用 skip topk 层复用计算层索引

问题与约束：
- GLM5/DSA 类模型可能存在 cross-layer index sharing：部分层不计算自己的 top-k，而复用最近计算层的 top-k indices。

设计选择：
- `is_skip_topk_layer` 判断某层是否跳过 top-k；`source_compute_layer` 从当前层向前寻找最近的计算层。

Explain：
`is_skip_topk_layer` 使用 `max(layer_number - skip_topk_offset, 0) % topk_freq != 0` 判断是否 skip。`source_compute_layer` 在 skip 时不断递减 layer，直到找到计算 top-k 的层。

来源：slime_plugins/models/glm5/glm5.py L37-L52

Code：

```python
def is_skip_topk_layer(layer_number: int, skip_topk_offset: int, topk_freq: int) -> bool:
    """Whether the (1-indexed) Megatron ``layer_number`` reuses a previous layer's top-k.
    """
    return (max(layer_number - skip_topk_offset, 0) % topk_freq) != 0


def source_compute_layer(layer_number: int, skip_topk_offset: int, topk_freq: int) -> int:
    """The computing layer whose ``topk_indices`` a skip layer reuses."""
    layer = layer_number
    while is_skip_topk_layer(layer, skip_topk_offset, topk_freq):
        layer -= 1
    return layer
```

代码逻辑：
- `skip_topk_offset/topk_freq` 定义计算层周期。
- skip 层沿 layer number 反向查找。
- 返回值用于定位提供 top-k indices 的源层。

为什么这样写：
- cross-layer index sharing 要保持层间 top-k 引用稳定，不能每层独立计算。
- 把判断抽成函数后，后续构建子模块或加载权重时都能复用同一规则。

不变量与失败模式：
- `topk_freq` 必须为正，否则取模非法。
- 若配置导致向前找不到计算层，while 会持续递减并产生错误层号。

Comment：
GLM5 插件展示的是模型结构级扩展：它不是 rollout 逻辑，而是 Megatron 模型定义适配。
