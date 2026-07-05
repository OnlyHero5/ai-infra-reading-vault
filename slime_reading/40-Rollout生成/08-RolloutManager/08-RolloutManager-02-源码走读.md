---
type: batch-doc
module: 08-RolloutManager
batch: "08"
doc_type: walkthrough
title: "RolloutManager · 源码走读"
tags:
  - slime/batch/08
  - slime/module/rollout-manager
  - slime/doc/walkthrough
updated: 2026-07-05
---

# RolloutManager · 源码走读

> 走读主线：`RolloutManager` 是 Ray actor，负责启动/管理 SGLang rollout server，调用用户 rollout 函数取得 `Sample`，把样本转换为列式 `train_data`，再根据 Megatron 训练并行配置切成 DP rank 可消费的 Ray ObjectRef。

---

## 1. 数据契约：Sample、rollout 函数与 tensor 化

### 1.1 Sample 是 rollout 与训练之间的最小记录单元

问题与约束：
- rollout 函数可能来自不同任务、agent 或数据源，但训练侧需要稳定字段：token、response 长度、reward、loss mask、rollout id 和可选 logprob/top-p/MoE/多模态信息。

设计选择：
- `Sample` dataclass 把通用训练字段放在同一对象中；默认 `rollout_id=None`，compact/subagent 场景需要显式给 sibling 样本设置相同 rollout id。

Explain：
`Sample` 的注释说明默认路径可以把 `rollout_id` 回退到 `index`；如果一次 rollout execution 被拆成多个训练样本，则这些 sibling 必须共享同一个 `rollout_id`，否则 loss 聚合会把一个 rollout 过度计数。

来源：slime/utils/types.py L94-L146

Code：

```python
class Sample:
    group_index: int | None = None
    index: int | None = None
    rollout_id: int | None = None
    prompt: str | list[dict[str, str]] = ""
    tokens: list[int] = field(default_factory=list)
    multimodal_train_inputs: dict[str, Any] | None = None
    response: str = ""
    response_length: int = 0
    reward: float | dict[str, Any] | None = None
    loss_mask: list[int] | None = None
    rollout_log_probs: list[float] | None = None
    rollout_top_p_token_ids: list[int] | torch.Tensor | None = None
    rollout_top_p_token_offsets: list[int] | torch.Tensor | None = None
    rollout_routed_experts: list[list[int]] | torch.Tensor | None = None
    remove_sample: bool = False
    teacher_log_probs: list[float] | None = None
    train_metadata: dict | None = None
```

代码逻辑：
- `tokens/response_length/reward/loss_mask` 是训练主路径字段。
- `rollout_log_probs/top_p/teacher_log_probs` 支持 off-policy 或蒸馏类训练。
- `rollout_routed_experts` 记录 MoE replay 信息。
- `remove_sample` 表示保留样本但清空 loss。

为什么这样写：
- rollout 输出需要跨 Ray、SGLang、训练后端传递，统一 dataclass 能降低自定义 rollout 函数的适配成本。
- `rollout_id` 与后续 DP schedule 和 loss denominator 绑定，必须在数据对象层暴露。

不变量与失败模式：
- `loss_mask` 长度必须等于 `response_length`，后续转换阶段会 assert。
- top-p offsets 长度必须是 `response_length + 1`，并且最后一个 offset 等于 token ids 数。

Comment：
理解 RolloutManager 之前，先把 `Sample` 看成它唯一真正处理的业务对象。

### 1.2 call_rollout_fn 兼容旧 rollout 函数返回裸 list

问题与约束：
- 旧版 rollout 函数可能直接返回 list；新版接口又需要 metrics 和 train/eval 两类返回结构。

设计选择：
- `call_rollout_fn` 调用用户函数后，如果返回值不是 `RolloutFnTrainOutput/RolloutFnEvalOutput`，就按 `evaluation` 标志包装成新版输出。

Explain：
训练 rollout 的规范输出是 `RolloutFnTrainOutput(samples=...)`，eval 输出是 `RolloutFnEvalOutput(data=...)`。兼容包装让旧函数无需一次性迁移，也让 `RolloutManager._get_rollout_data` 可以总是读取 `.samples/.metrics`。

来源：slime/rollout/base_types.py L7-L26

Code：

```python
@dataclass
class RolloutFnTrainOutput:
    samples: list[list[Sample]]
    metrics: dict[str, Any] = None

@dataclass
class RolloutFnEvalOutput:
    data: dict[str, dict[str, Any]]
    metrics: dict[str, Any] = None

def call_rollout_fn(fn, *args, evaluation: bool, **kwargs):
    output = fn(*args, **kwargs, evaluation=evaluation)
    if not isinstance(output, (RolloutFnTrainOutput, RolloutFnEvalOutput)):
        output = RolloutFnEvalOutput(data=output) if evaluation else RolloutFnTrainOutput(samples=output)
    return output
```

代码逻辑：
- 用户函数总是收到 `evaluation` 参数。
- train/eval 输出类型分开。
- 非标准返回自动包装。

为什么这样写：
- rollout 函数是用户扩展点，兼容旧返回能减少升级成本。
- manager 后续逻辑只面对统一结构，避免每处重复判断。

不变量与失败模式：
- train 模式下包装后的对象必须有 `.samples`。
- eval 模式下包装后的对象必须有 `.data`。

Comment：
这是 rollout 插件接口的兼容层。

### 1.3 tensor 化只处理训练侧需要 tensor 的字段

问题与约束：
- Ray Object Store 不适合传 GPU tensor；但把所有字段都 tensor 化又会增加序列化和训练侧拆包成本。

设计选择：
- `_ROLLOUT_DATA_TENSOR_DTYPES` 只列出需要转成 CPU tensor 的字段；`_tensorize_rollout_data_for_training` 对这些字段逐 sample 转 contiguous CPU tensor，多模态字典内部只转 ndarray/tensor。

Explain：
`rollout_routed_experts` 的 dtype 是 `None`，表示保留原始 dtype 交给 `torch.as_tensor` 推断。`rollout_mask_sums` 是 per-rank 向量，被整体转成 float32 tensor，而不是逐 sample list。

来源：slime/ray/rollout.py L39-L102

Code：

```python
_ROLLOUT_DATA_TENSOR_DTYPES = {
    "tokens": torch.long,
    "loss_masks": torch.int,
    "rollout_log_probs": torch.float32,
    "rollout_top_p_token_ids": torch.int32,
    "rollout_top_p_token_offsets": torch.int32,
    "teacher_log_probs": torch.float32,
    "rollout_routed_experts": None,
}

def _cpu_tensor(value, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(value, np.ndarray) and not value.flags.writeable:
        value = value.copy()
    tensor = torch.as_tensor(value, dtype=dtype) if dtype is not None else torch.as_tensor(value)
    return tensor.detach().cpu().contiguous()

def _tensorize_rollout_data_for_training(rollout_data: dict[str, Any]) -> None:
    for key, dtype in _ROLLOUT_DATA_TENSOR_DTYPES.items():
        if key in rollout_data:
            rollout_data[key] = [_cpu_tensor(value, dtype=dtype) for value in rollout_data[key]]
```

代码逻辑：
- 不可写 numpy array 先 copy。
- tensor detach 后放 CPU，并保证 contiguous。
- 多模态输入字典逐 key 转换。
- `rollout_mask_sums` 单独整体 tensor 化。

为什么这样写：
- Object Store 中传 CPU contiguous tensor 更稳定，避免 GPU tensor 跨节点传输问题。
- 只 tensor 化训练热路径字段，保留 reward/length 等简单 list 的轻量序列化。

不变量与失败模式：
- tensor 化函数就地修改 `rollout_data`，调用方不能再假设原字段仍是 list of list。
- 字段 dtype 表漏掉新训练字段时，训练侧可能收到 Python list 而不是 tensor。

Comment：
这段决定了 RolloutManager 交给训练 actor 的数据物理形态。

---

## 2. RolloutManager 生命周期与 server 管理

### 2.1 __init__ 启动 rollout server、加载数据源和用户 hook

问题与约束：
- RolloutManager 是 Ray actor，需要在远端进程里初始化 SGLang server、数据源、rollout/eval 函数、reward/convert hook、tracking 和故障监控。

设计选择：
- `debug_train_only` 跳过 server 启动；正常路径先初始化 HTTP client 和 rollout servers，再加载数据源与函数，最后等待 engine init handles。

Explain：
`start_rollout_servers` 返回 servers 和 pending init handles；manager 先完成 Python 侧 hook 加载，再 `ray.get` 等待 rollout engine 初始化。`rollout_engine_lock` 是独立 Ray lock actor，权重更新时会被 Megatron 侧拿来做互斥。

来源：slime/ray/rollout.py L420-L471

Code：

```python
@ray.remote
class RolloutManager:
    def __init__(self, args, pg):
        configure_logger()
        self.pg = pg
        self.args = args

        rollout_init_handles: list[Any] = []
        if self.args.debug_train_only:
            self.servers: dict[str, Any] = {}
        else:
            init_http_client(args)
            self.servers, rollout_init_handles = start_rollout_servers(args, pg)

        data_source_cls = load_function(self.args.data_source_path)
        self.data_source = data_source_cls(args)
        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)

        if rollout_init_handles:
            ray.get(rollout_init_handles)

        self.rollout_engine_lock = Lock.options(...).remote()
```

代码逻辑：
- Ray actor 初始化时保存 args 和 placement group。
- 根据 debug 开关决定是否启动 rollout server。
- 动态加载数据源、rollout 函数和 eval 函数。
- 有 fault tolerance 时为 server group 启动 health monitor。

为什么这样写：
- server 初始化和函数加载都发生在 actor 进程内，后续 remote 方法可以复用这些状态。
- `ray.get(rollout_init_handles)` 放在 hook 加载后，可以并行利用 engine init 等待时间。

不变量与失败模式：
- `data_source_path/rollout_function_path/eval_function_path` 必须能被 `load_function` 导入。
- 非 debug 模式下 rollout servers 必须完成 init，generate 才能稳定调用。

Comment：
RolloutManager 初始化不是轻量对象构造，而是启动整条 rollout 侧服务链。

### 2.2 start_rollout_servers 支持多模型、多 server group 和延迟 engine init

问题与约束：
- rollout 侧可能有 policy/reference/reward 多模型，也可能有 PD/EPD disaggregation；不同 server group 的 GPU 数、worker type 和 router 都不同。

设计选择：
- 先解析 SGLang config；每个 model 创建 router 和若干 `ServerGroup`；函数返回 `servers` map 和 pending init handles，让调用方统一等待。

Explain：
对第一个模型，router ip/port 会写回旧字段 `args.sglang_router_ip/port` 以兼容旧 rollout 函数；所有模型的 router 信息还会写入 `args.sglang_model_routers`。EPD 模式先启动 encoder group 并收集 URL，再把 encoder URLs 注入非 encoder group。

来源：slime/ray/rollout.py L1089-L1228

Code：

```python
def start_rollout_servers(args, pg) -> tuple[dict[str, Any], list[Any]]:
    if args.rollout_external:
        return start_external_rollout_servers(args, start_router=_start_router)

    config = _resolve_sglang_config(args)
    servers: dict[str, RolloutServer] = {}
    pending_init_handles: list[Any] = []

    for model_idx, model_cfg in enumerate(config.models):
        model_cfg.resolve(args)
        has_pd = model_cfg.has_pd_disaggregation
        router_ip, router_port = _start_router(args, has_pd_disaggregation=has_pd, force_new=(model_idx > 0))

        if model_idx == 0:
            args.sglang_router_ip = router_ip
            args.sglang_router_port = router_port

        server_groups: list[ServerGroup] = []
        ...
        servers[model_cfg.name] = RolloutServer(
            server_groups=server_groups,
            router_ip=router_ip,
            router_port=router_port,
            model_name=model_cfg.name,
            update_weights=model_cfg.update_weights,
        )

    args.sglang_model_routers = {name: (srv.router_ip, srv.router_port) for name, srv in servers.items()}
    return servers, pending_init_handles
```

代码逻辑：
- external rollout 走独立启动路径。
- config 中每个 model 都有自己的 router。
- server group 负责实际 engine actor。
- pending init handles 由 manager 统一 `ray.get`。

为什么这样写：
- 多模型 rollout 需要隔离 router 和 engine group，但用户 rollout 函数仍需要能找到各模型 URL。
- 延迟等待 init handles 让 server group 启动可以并行推进。

不变量与失败模式：
- `sglang_config.total_num_gpus` 必须等于 `args.rollout_num_gpus`。
- 多模型 weight update 目前只选择第一个 `update_weights=True` 的 server。

Comment：
RolloutManager 看到的 `self.servers` 已经是多模型 rollout topology 的抽象结果。

### 2.3 get_updatable_engines_and_lock 只暴露可更新模型

问题与约束：
- 多模型 rollout 中 reference/reward 等 frozen 模型不应参与 Megatron 权重同步；但 update_weights 需要拿到 engine actor、GPU 布局和互斥锁。

设计选择：
- `_get_updatable_server` 找第一个 `update_weights=True` 的 server；`get_updatable_engines_and_lock` 返回该 server 的 engines、lock、new engine 数和 GPU 映射信息。

Explain：
返回值中的 `engines` 是可更新 server 的 node-0 engine 集合，`all_engine_actors` 保留完整 actor 列表，fault tolerance 后的新增 engine 数通过 `num_new` 告知训练侧。

来源：slime/ray/rollout.py L504-L540

Code：

```python
def _get_updatable_server(self) -> Any | None:
    for srv in self.servers.values():
        if srv.update_weights:
            return srv
    return None

def get_updatable_engines_and_lock(self):
    srv = self._get_updatable_server()
    engines = srv.engines if srv else []
    gpu_counts = srv.engine_gpu_counts if srv else []
    gpu_offsets = srv.engine_gpu_offsets if srv else []
    num_new = srv.num_new_engines if srv else 0
    all_engine_actors = srv.all_engines if srv else []
    return engines, self.rollout_engine_lock, num_new, gpu_counts, gpu_offsets, all_engine_actors
```

代码逻辑：
- 遍历 server map 查找可更新模型。
- 没有可更新 server 时返回空 engine 与空 GPU 信息。
- lock 总是 manager 级共享 lock。

为什么这样写：
- weight update 是 rollout engine 和训练 actor 的共享临界区，必须显式提供同一把锁。
- frozen 模型排除在返回值外，避免错误地同步 reference/reward 权重。

不变量与失败模式：
- 多个 update_weights server 时只取第一个，源码注释说明尚不支持多模型同时更新。
- 训练侧必须按返回的 GPU offset/count 解释 engine 分片。

Comment：
这个方法是 RolloutManager 暴露给 Megatron actor 的权重更新入口。

### 2.4 recover_updatable_engines 只恢复可更新 server 并暂停 health monitor

问题与约束：
- 故障恢复时不能让 health monitor 与恢复逻辑同时操作 engine group；同时非更新模型不应被权重同步路径恢复。

设计选择：
- `recover_updatable_engines` 先 pause health monitoring，再找 updatable server；如果已经开始 rollout，则调用 `srv.recover()` 并返回恢复后的 engine 和 GPU 信息。

Explain：
`rollout_id == -1` 表示还没进行过 rollout，此时不调用 recover，只返回当前 engine 信息。恢复完成后返回的 `num_new_engines` 让训练侧判断是否要给新 actor 补权重。

来源：slime/ray/rollout.py L595-L616

Code：

```python
def recover_updatable_engines(self):
    self.health_monitoring_pause()
    srv = self._get_updatable_server()
    if self.rollout_id == -1 or srv is None:
        engines = srv.engines if srv else []
        gpu_counts = srv.engine_gpu_counts if srv else []
        gpu_offsets = srv.engine_gpu_offsets if srv else []
        return engines, self.rollout_engine_lock, (srv.num_new_engines if srv else 0), gpu_counts, gpu_offsets

    srv.recover()
    return (
        srv.engines,
        self.rollout_engine_lock,
        srv.num_new_engines,
        srv.engine_gpu_counts,
        srv.engine_gpu_offsets,
    )
```

代码逻辑：
- 先暂停健康监控。
- 无 updatable server 或 rollout 未开始时只返回当前状态。
- 启动后故障恢复调用 server 的 recover。
- 返回值形态与 update 权重入口保持一致。

为什么这样写：
- 恢复与健康检查都可能读写 engine actor 状态，需要避免并发冲突。
- 训练侧使用同一类返回元数据处理正常更新和故障恢复后的补更新。

不变量与失败模式：
- `srv.recover()` 必须重建 server 内部 engines/gpu metadata。
- 调用方需要在合适时机恢复 health monitoring，否则故障检测会停留在 pause 状态。

Comment：
故障恢复路径仍围绕“可更新模型”收敛，不会影响 frozen rollout 模型。

---

## 3. 生成入口与 rollout 数据获取

### 3.1 generate 是训练主循环调用的 rollout 入口

问题与约束：
- 训练主循环需要一个远端入口完成 rollout、日志、debug dump、样本转换和 DP 切分；同时 debug rollout-only 模式不能进入训练数据转换。

设计选择：
- `generate(rollout_id)` 串行执行 health resume、可选 CI fault injection、`_get_rollout_data`、debug 保存、日志、样本转换和 DP split。

Explain：
`self.rollout_id` 在这里更新，用于故障恢复判断是否已经开始过 rollout。`debug_rollout_only` 直接 return，避免训练侧消费未转换的数据。

来源：slime/ray/rollout.py L546-L559

Code：

```python
def generate(self, rollout_id):
    start_time = time.time()
    self.rollout_id = rollout_id
    self.health_monitoring_resume()
    if self.args.ci_test and self.args.use_fault_tolerance and rollout_id >= 2:
        self._try_ci_fault_injection()
    data, metrics = self._get_rollout_data(rollout_id=rollout_id)
    self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=False)
    _log_rollout_data(rollout_id, self.args, data, metrics, time.time() - start_time)
    if self.args.debug_rollout_only:
        return
    data = self._convert_samples_to_train_data(data)
    return self._split_train_data_by_dp(data)
```

代码逻辑：
- 每次 generate 记录 rollout id 和耗时起点。
- 生成数据后先保存 debug，再日志。
- debug-only 分支提前退出。
- 正常返回 `list[Box]`，每个 Box 包装一个 rank 的 ObjectRef。

为什么这样写：
- debug dump 应保存原始 Sample，而不是转换后的 train_data。
- 日志需要原始样本统计和 rollout 时间。
- 训练侧只应拿到已经按 DP rank 切好的数据。

不变量与失败模式：
- `set_train_parallel_config` 必须在 `_split_train_data_by_dp` 前被调用。
- `_get_rollout_data` 返回的样本列表不能为空，否则后续访问 `samples[0]` 会失败。

Comment：
`generate` 是 RolloutManager 从推理世界进入训练数据世界的唯一主入口。

### 3.2 _get_rollout_data 支持 debug 复放与正常 rollout

问题与约束：
- 调试时希望从磁盘复放已生成样本；正常训练时又要调用用户 rollout 函数，并兼容嵌套 list 输出。

设计选择：
- `load_debug_rollout_data` 分支从 torch 文件读 `samples` 并恢复 `Sample`；正常分支通过 `call_rollout_fn` 取 `.samples/.metrics`，先校验 rollout id，再逐层 flatten。

Explain：
正常路径保留 rollout 函数返回的 metrics。`_validate_rollout_id_annotated` 在 flatten 前运行，因为只有嵌套结构还存在时，才能判断 compact sibling 是否属于同一次 rollout execution。

来源：slime/ray/rollout.py L635-L665

Code：

```python
if self.args.load_debug_rollout_data:
    data = torch.load(
        self.args.load_debug_rollout_data.format(rollout_id=rollout_id),
        weights_only=False,
    )["samples"]
    data = [Sample.from_dict(sample) for sample in data]
    if (ratio := self.args.load_debug_rollout_data_subsample) is not None:
        original_num_rows = len(data)
        rough_subsample_num_rows = int(original_num_rows * ratio)
        data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
    metrics = None
else:
    data = call_rollout_fn(self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False)
    metrics = data.metrics
    data = data.samples
    _validate_rollout_id_annotated(data)
    while isinstance(data[0], list):
        data = list(itertools.chain.from_iterable(data))

return data, metrics
```

代码逻辑：
- debug 样本用 `Sample.from_dict` 还原。
- 可选 subsample 取头尾两段。
- 正常 rollout 输出先通过 wrapper 归一化。
- 嵌套 list 被展开为 `list[Sample]`。

为什么这样写：
- debug 复放绕过 SGLang server，可复现实验数据转换和训练问题。
- 校验放在 flatten 前，才能识别 compact/subagent sibling。

不变量与失败模式：
- debug 文件必须包含 `"samples"` 字段。
- 正常 rollout 返回的嵌套结构最终叶子必须是 `Sample`。
- `data[0]` 假设数据非空。

Comment：
这一步把用户 rollout 函数的自由输出收敛成 manager 后续处理的平面 Sample 列表。

### 3.3 _validate_rollout_id_annotated 只约束 compact sibling

问题与约束：
- 默认 rollout 输出是 `list[list[Sample]]`，不应强制每个 Sample 都设置 rollout_id；但 compact/subagent 会把一次 rollout 拆成多个 sibling 样本，必须共享 id。

设计选择：
- 递归遍历 rollout 输出树，只在 depth >= 2 且同一 leaf list 中有多个 Sample 时，要求 rollout_id 非空且完全相同。

Explain：
源码 docstring 明确区分默认 depth-2 形态和 compact depth-3 形态。这样保持旧 rollout 函数兼容，同时保护 compact 训练样本的 loss reducer denominator。

来源：slime/ray/rollout.py L898-L927

Code：

```python
def _validate_rollout_id_annotated(node, depth=0):
    if isinstance(node, Sample):
        return
    assert isinstance(node, list), f"unexpected rollout output node type: {type(node).__name__}"
    if node and isinstance(node[0], Sample):
        if depth >= 2 and len(node) > 1:
            rids = [s.rollout_id for s in node]
            missing = [i for i, r in enumerate(rids) if r is None]
            assert not missing, (
                f"Compact rollout returned {len(node)} samples but rollout_id is unset on "
                f"positions {missing}. Set Sample.rollout_id on every sibling so the loss "
                "reducer can aggregate them as one rollout instead of N."
            )
            assert len(set(rids)) == 1, f"Sibling samples from one compact rollout must share rollout_id; got {rids}."
        return
    for item in node:
        _validate_rollout_id_annotated(item, depth + 1)
```

代码逻辑：
- 单个 Sample 直接返回。
- 非 list 节点直接 assert。
- leaf list 在 depth >= 2 时触发 compact 校验。
- sibling id 缺失或不一致都会失败。

为什么这样写：
- 默认历史输出不带 rollout_id 仍能运行。
- compact/subagent 场景如果不共享 id，后续按 rollout 聚合 loss mask 会错误。

不变量与失败模式：
- compact sibling 多样本必须全部设置同一个 rollout_id。
- 函数只校验结构，不会自动填充 compact id。

Comment：
这段是 agent/compact rollout 接入训练 reducer 的关键防线。

### 3.4 _log_rollout_data 把样本指标和性能指标写入 tracking

问题与约束：
- rollout 指标既可能来自用户函数返回 metrics，也需要从 Sample 中统一计算 response、reward、truncation 和 SGLang 性能指标。

设计选择：
- 先允许 custom log hook 完全接管；否则合并 rollout metrics、样本统计和 perf 统计，并用 `compute_rollout_step` 生成 tracking step。

Explain：
debug load 分支不重复记录 rollout 指标。默认日志把 `compute_metrics_from_samples` 加 `rollout/` 前缀，把 `compute_perf_metrics_from_samples` 加 `perf/` 前缀，最后用 `rollout/step` 作为 step key。

来源：slime/ray/rollout.py L1291-L1307

Code：

```python
def _log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    if args.custom_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_rollout_log_function_path)
        if custom_log_func(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
            return

    if args.load_debug_rollout_data:
        return

    log_dict = {**(rollout_extra_metrics or {})}
    log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), "rollout/")
    log_dict |= dict_add_prefix(compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/")
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    logging_utils.log(args, log_dict, step_key="rollout/step")
```

代码逻辑：
- custom hook 返回 truthy 时跳过默认日志。
- debug 复放不记录默认 rollout 日志。
- 统计指标分前缀合并。
- tracking step 由 rollout id 映射。

为什么这样写：
- 用户任务可自定义日志语义，但默认路径仍提供统一监控。
- debug 复放不代表真实 rollout 性能，跳过默认日志更安全。

不变量与失败模式：
- custom log hook 必须接受同一组参数。
- 样本统计函数假设 samples 是平面 `list[Sample]`。

Comment：
日志逻辑虽然不进入训练数据，但它定义了 rollout 侧可观测性的默认口径。

---

## 4. Sample 到 train_data 的列式转换

### 4.1 _post_process_rewards 支持 GRPO 类 group normalization

问题与约束：
- 不同算法对 reward 的处理不同；GRPO/GSPO/CISPO 等需要按 prompt group 做均值或标准差归一化。

设计选择：
- 若提供 custom reward post-process hook 则直接委托；否则从每个 Sample 取 raw reward，并按 advantage estimator 和 normalization flags 做 group norm。

Explain：
当 reward 数量等于 `n_samples_per_prompt * rollout_batch_size` 时，源码 reshape 成 `[prompt, n_samples]`；否则退化为 view。GRPO/GSPO/CISPO 且启用 std normalization 时再除以标准差加 `1e-6`。

来源：slime/ray/rollout.py L686-L711

Code：

```python
raw_rewards = [sample.get_reward_value(self.args) for sample in samples]
if (
    self.args.advantage_estimator in ["grpo", "gspo", "cispo", "reinforce_plus_plus_baseline"]
    and self.args.rewards_normalization
):
    rewards = torch.tensor(raw_rewards, dtype=torch.float)
    if rewards.shape[-1] == self.args.n_samples_per_prompt * self.args.rollout_batch_size:
        rewards = rewards.reshape(-1, self.args.n_samples_per_prompt)
    else:
        rewards = rewards.view(-1, rewards.shape[-1])
    mean = rewards.mean(dim=-1, keepdim=True)
    rewards = rewards - mean
    if self.args.advantage_estimator in ["grpo", "gspo", "cispo"] and self.args.grpo_std_normalization:
        std = rewards.std(dim=-1, keepdim=True)
        rewards = rewards / (std + 1e-6)
    return raw_rewards, rewards.flatten().tolist()
return raw_rewards, raw_rewards
```

代码逻辑：
- raw reward 始终保留。
- normalization 只在指定 estimator 和开关下执行。
- reward 输出 flatten 回 sample 顺序。
- 无 normalization 时 raw/reward 相同。

为什么这样写：
- 训练侧需要既能看到原始 reward，又能使用算法处理后的 reward。
- 归一化放在 manager 侧，可以在 DP 分片前看到完整 batch group。

不变量与失败模式：
- `samples` 顺序必须仍按 prompt group 排列，否则 group normalization 会混组。
- reward 必须能转为 float tensor。

Comment：
reward 后处理发生在 DP split 前，因为只有这里能看到完整 rollout batch。

### 4.2 _convert_samples_to_train_data 构造列式主字段和 loss_masks

问题与约束：
- 训练 actor 需要按字段批量读取数据；同时每个 Sample 的 `rollout_id/loss_mask/remove_sample` 规则要在分片前固定下来。

设计选择：
- 将 `list[Sample]` 转成列式 dict：tokens、response_lengths、rewards、raw_reward、truncated、sample_indices、rollout_ids 和 loss_masks。

Explain：
`rollout_id is None` 时会分配不与已有 id 冲突的临时 id。`loss_mask` 缺失时默认全 1；如果 `remove_sample=True`，loss mask 会被置零，但样本仍保留在训练数据中。

来源：slime/ray/rollout.py L713-L761

Code：

```python
raw_rewards, rewards = self._post_process_rewards(samples)
assert len(raw_rewards) == len(samples)
assert len(rewards) == len(samples)

rollout_ids = [sample.rollout_id for sample in samples]
existed_rollout_id_values = set(rid for rid in rollout_ids if rid is not None)
tmp_id = 0
for i in range(len(rollout_ids)):
    if rollout_ids[i] is None:
        while tmp_id in existed_rollout_id_values:
            tmp_id += 1
        rollout_ids[i] = tmp_id
        existed_rollout_id_values.add(tmp_id)

train_data = {
    "tokens": [sample.tokens for sample in samples],
    "response_lengths": [sample.response_length for sample in samples],
    "rewards": rewards,
    "raw_reward": raw_rewards,
    "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
    "sample_indices": [sample.index for sample in samples],
    "rollout_ids": rollout_ids,
}
```

```python
for sample in samples:
    if sample.loss_mask is None:
        sample.loss_mask = [1] * sample.response_length
    assert len(sample.loss_mask) == sample.response_length
    if sample.remove_sample:
        sample.loss_mask = [0] * sample.response_length
    loss_masks.append(sample.loss_mask)
train_data["loss_masks"] = loss_masks
```

代码逻辑：
- custom convert hook 可完全替换默认转换。
- 主字段都以 sample 顺序排列。
- 缺失 rollout id 自动补齐。
- loss mask 在这里被实例化和校验。

为什么这样写：
- 列式 dict 更适合后续按 DP partition 取子集。
- `remove_sample` 清零 loss 而不删除样本，可以保留 batch 结构和日志统计。

不变量与失败模式：
- raw rewards、processed rewards 与 samples 长度必须一致。
- 自动分配的 rollout id 只保证当前 batch 内不冲突。

Comment：
这一步把面向任务的 Sample 对象转换为面向训练调度的列式数据。

### 4.3 rollout_mask_sums 让拆开的 micro-batch 共享同一 rollout denominator

问题与约束：
- 一个 rollout 可能拆出多个 training samples，并且 first-fit packing 可能把这些 samples 放进不同 micro-batch；loss reducer 仍需要用整个 rollout 的 mask 总和作分母。

设计选择：
- 在 DP split 前按 rollout_id 汇总每个 sample 的 loss mask sum，再把总和广播回同 rollout 的每个 sample。

Explain：
`rollout_mask_sums[i]` 表示 sample i 所属 rollout 的总 loss token 数，而不是 sample 自己的 mask sum。这样 micro-batch 内只处理部分 sibling 时，reducer 仍可计算按 rollout 聚合的 token-weighted mean。

来源：slime/ray/rollout.py L763-L778

Code：

```python
rollout_id_list = train_data["rollout_ids"]
mask_sums_per_sample = [sum(m) for m in loss_masks]
rollout_total_mask: dict[int, int] = {}
for rid, ms in zip(rollout_id_list, mask_sums_per_sample, strict=True):
    rollout_total_mask[rid] = rollout_total_mask.get(rid, 0) + ms
train_data["rollout_mask_sums"] = [rollout_total_mask[rid] for rid in rollout_id_list]
```

代码逻辑：
- 先计算每个 sample 的 mask sum。
- 再按 rollout_id 聚合。
- 最后按 sample 顺序生成同长度列。

为什么这样写：
- 只有 DP split 前能看到所有 sample，适合预计算全 rollout denominator。
- 广播成 per-sample 列后，训练侧只需按本 rank partition 索引即可。

不变量与失败模式：
- 同一 compact rollout 的 sibling 必须共享 rollout_id，否则 denominator 会被拆散。
- `loss_masks` 必须已完成 remove_sample 处理。

Comment：
这是 RolloutManager 支持 compact/subagent 多样本输出的关键训练语义。

### 4.4 可选字段按首样本或配置探测后加入 train_data

问题与约束：
- 不同训练模式需要不同附加字段；强制所有样本都携带所有字段会增加 rollout 函数负担。

设计选择：
- 对 off-policy logprob、top-p replay、MoE routed experts、metadata、多模态输入和 teacher logprob 做可选字段探测；top-p replay 额外校验 offsets 合法性。

Explain：
`rollout_top_p != 1.0` 时，每个 sample 都必须带 top-p token ids 和 offsets，且 offsets 最后一项要等于 token ids 长度。`raw_reward` 可被 sample metadata 覆盖，支持混合数据源只对部分样本写 raw reward。

来源：slime/ray/rollout.py L780-L823

Code：

```python
if any(sample.metadata and "raw_reward" in sample.metadata for sample in samples):
    train_data["raw_reward"] = [
        sample.metadata["raw_reward"] if sample.metadata and "raw_reward" in sample.metadata else sample.reward
        for sample in samples
    ]

if samples[0].rollout_log_probs is not None:
    train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

if getattr(self.args, "rollout_top_p", 1.0) != 1.0:
    for sample in samples:
        assert sample.rollout_top_p_token_ids is not None
        assert sample.rollout_top_p_token_offsets is not None
        assert len(sample.rollout_top_p_token_offsets) == sample.response_length + 1
        offset_end = int(sample.rollout_top_p_token_offsets[-1])
        assert offset_end == len(sample.rollout_top_p_token_ids)
    train_data["rollout_top_p_token_ids"] = [sample.rollout_top_p_token_ids for sample in samples]
    train_data["rollout_top_p_token_offsets"] = [sample.rollout_top_p_token_offsets for sample in samples]
```

代码逻辑：
- raw reward metadata 覆盖是按任意样本存在触发。
- rollout logprob、routed experts、teacher logprob 主要按首样本判断。
- 多模态输入按任意样本非空判断。
- top-p replay 按配置强校验所有样本。

为什么这样写：
- 可选列保持 train_data 稀疏，只在训练模式需要时出现。
- top-p replay 是 ragged 编码，offset 错误会让训练侧无法还原每个 response token 的候选集。

不变量与失败模式：
- 如果首样本没有 rollout_log_probs，但后续样本有，默认路径不会加入该列。
- top-p 开启时任何样本缺少 ids/offsets 都会 assert。

Comment：
可选字段策略让默认 RL 数据路径保持简单，同时给 OPD、MoE、多模态和采样 replay 留接口。

---

## 5. DP 切分、ObjectRef 与训练交付

### 5.1 _split_train_data_by_dp 先调度再按 rank 打包 ObjectRef

问题与约束：
- 训练侧按 DP rank 消费数据；每个 rank 需要自己的 partition、micro-batch indices 和全局 step 信息，同时部分字段必须保持全局列表供训练侧索引。

设计选择：
- `_split_train_data_by_dp` 调用 `build_dp_schedule` 得到 partitions 和 micro-batch metadata，再为每个 rank 构造 `rollout_data`，tensor 化后用 Ray Object Store 或 NIXL transport 放入 `Box`。

Explain：
`tokens` 等样本级字段按 rank partition 切分；`raw_reward/total_lengths` 不切分，保留全局列。每个 rank 的 `micro_batch_indices[r]` 是本 rank partition 内的局部索引，不是全局 sample id。

来源：slime/ray/rollout.py L826-L895

Code：

```python
def set_train_parallel_config(self, config: dict):
    self.train_parallel_config = config

def _split_train_data_by_dp(self, data):
    dp_size = self.train_parallel_config["dp_size"]
    total_lengths = [len(t) for t in data["tokens"]]
    data["total_lengths"] = total_lengths

    partitions, micro_batch_indices, num_microbatches, global_batch_sizes = build_dp_schedule(
        self.args,
        self.train_parallel_config,
        total_lengths,
        global_batch_size=self.args.global_batch_size,
        rollout_indices=data["rollout_ids"],
    )

    for r in range(dp_size):
        partition = partitions[r]
        rollout_data = {"partition": partition}
        ...
        rollout_data["global_batch_sizes"] = global_batch_sizes
        rollout_data["num_microbatches"] = num_microbatches
        rollout_data["micro_batch_indices"] = micro_batch_indices[r]
        _tensorize_rollout_data_for_training(rollout_data)
        transport = getattr(self.args, "rollout_data_transport", "object-store")
```

代码逻辑：
- 训练并行配置由外部 actor 先设置。
- 计算每个样本总 token 长度。
- 调度函数输出 rank partition 和 micro-batch 切分。
- 每个 rank 的数据独立 `ray.put`。

为什么这样写：
- 调度逻辑保持纯 Python，便于单测；Ray 打包只留在 manager 里。
- rank 数据提前切好，训练 actor 不需要从全量 batch 中自行过滤。

不变量与失败模式：
- `train_parallel_config` 必须包含 `dp_size/cp_size/vpp_size/microbatch_group_size_per_vp_stage`。
- `rollout_data_transport` 只能是 `"object-store"` 或 `"nixl"`。

Comment：
`_split_train_data_by_dp` 是 RolloutManager 输出给训练侧的最后一道封装。

### 5.2 build_dp_schedule 按 rollout id 分 step，再 pack micro-batch

问题与约束：
- 一个 training step 应包含固定数量 rollout，而不是固定数量 training samples；同一 rollout 的样本必须在同一 step 内，动态 batch size 又要满足 DP/VPP 对 micro-batch 数的对齐要求。

设计选择：
- 先按 rollout id 聚合 sample indices，再每 `global_batch_size` 个 rollout 形成一个 step；step 内先 pack 成 micro-batches，再把 micro-batch 数对齐到 `dp_size * mb_group`。

Explain：
动态 batch size 下 `max_per_bin = max_tokens_per_gpu * cp_size`。如果 micro-batch 数不足对齐目标，动态路径尝试拆分 bin；静态路径直接报错，因为拆分会破坏固定 micro batch size 不变量。最后按 workload balance 或 stride round-robin 分配给 DP rank。

来源：slime/utils/dp_schedule.py L82-L209

Code：

```python
dp_size = train_parallel_config["dp_size"]
cp_size = train_parallel_config["cp_size"]
vpp_size = train_parallel_config["vpp_size"]
mb_group = train_parallel_config["microbatch_group_size_per_vp_stage"]

max_per_bin = None
if args.use_dynamic_batch_size:
    assert args.max_tokens_per_gpu is not None
    max_per_bin = args.max_tokens_per_gpu * cp_size

align_to = dp_size * (mb_group if vpp_size > 1 else 1)

rollout_id_to_samples: dict[int, list[int]] = {}
for sample_pos, rid in enumerate(rollout_indices):
    rollout_id_to_samples.setdefault(rid, []).append(sample_pos)
rollout_ids = list(rollout_id_to_samples.keys())
num_steps = len(rollout_ids) // global_batch_size
```

```python
step_mbs = _pack_step_into_mbs(...)
target_K = max(((len(step_mbs) + align_to - 1) // align_to) * align_to, align_to)
...
if args.balance_data:
    rank_mbs_idx = get_seqlen_balanced_partitions(mbs_weights, dp_size, equal_size=True)
else:
    rank_mbs_idx = [list(range(r, K, dp_size)) for r in range(dp_size)]

for r in range(dp_size):
    for mbs_idx in rank_mbs_idx[r]:
        local_start = len(partitions[r])
        partitions[r].extend(sample_indices[i] for i in mbs_locals)
        micro_batch_indices[r].append(list(range(local_start, local_start + len(mbs_locals))))
```

代码逻辑：
- rollout id 保留第一次出现顺序。
- trailing rollout 不足一个 global batch 时被丢弃。
- 每个 step 至少要有 `dp_size` 个 sample。
- partitions 存全局 sample indices，micro_batch_indices 存 rank-local indices。

为什么这样写：
- 按 rollout 分 step 才能让 rollout-level loss aggregation 与 global batch 定义一致。
- VPP 场景需要每 rank micro-batch 数对齐到 mb group。
- workload balance 可减少 rank 间长序列不均衡。

不变量与失败模式：
- `num_rollouts >= global_batch_size`，否则 assert。
- 静态 micro batch size 配置必须天然满足对齐，否则直接失败。
- 同一 rollout 的所有 sample 必须共享 rollout id，才能被聚合到同一 step。

Comment：
DP schedule 是 RolloutManager 正确交付训练数据的核心算法。

### 5.3 train_data 的 rank 打包保留全局字段与局部 micro-batch 索引

问题与约束：
- 训练 rank 需要自己的样本子集，但某些统计字段要以全局 batch 口径保存；同时 micro-batch index 必须对应本 rank 的局部数据列表。

设计选择：
- 样本级字段按 partition 取子集；`raw_reward/total_lengths` 作为全局字段完整保留；调度结果字段 `global_batch_sizes/num_microbatches/micro_batch_indices` 附加到每个 rank 数据。

Explain：
源码中可切分字段包括 tokens、response_lengths、rewards、loss_masks、rollout_ids、rollout_mask_sums、top-p、MoE、teacher logprob 等。`partition` 自身记录全局 sample 下标，训练侧可用它与全局字段对齐。

来源：slime/ray/rollout.py L853-L895

Code：

```python
partition = partitions[r]
rollout_data = {"partition": partition}
for key in [
    "tokens",
    "multimodal_train_inputs",
    "response_lengths",
    "rewards",
    "truncated",
    "loss_masks",
    "round_number",
    "sample_indices",
    "rollout_ids",
    "rollout_mask_sums",
    "rollout_log_probs",
    "rollout_top_p_token_ids",
    "rollout_top_p_token_offsets",
    "rollout_routed_experts",
    "prompt",
    "teacher_log_probs",
]:
    if key not in data:
        continue
    rollout_data[key] = [data[key][j] for j in partition]

for key in ["raw_reward", "total_lengths"]:
    if key not in data:
        continue
    rollout_data[key] = data[key]
```

```python
if transport == "nixl":
    rollout_data_refs.append(Box(ray.put(rollout_data, _tensor_transport="nixl")))
elif transport == "object-store":
    rollout_data_refs.append(Box(ray.put(rollout_data)))
else:
    raise ValueError(f"Unsupported rollout data transport: {transport!r}")
```

代码逻辑：
- 每个 rank 生成一个 rollout_data dict。
- partition 决定样本级列的切片。
- raw_reward 和 total_lengths 不切片。
- transport 分支控制 Ray put 的 tensor transport。

为什么这样写：
- 全局字段保留原始长度，方便训练侧按 partition 反查。
- Box 包装 ObjectRef，让上层调用保持轻量，不直接搬运大对象。

不变量与失败模式：
- 所有样本级列长度必须与 tokens 列一致。
- unsupported transport 会直接抛错，避免静默退回错误路径。

Comment：
这里的输出形态就是训练 actor 从 RolloutManager 接收的数据边界。

### 5.4 性能统计从 Sample trace 中抽取 SGLang 请求指标

问题与约束：
- rollout 性能不只看总耗时，还要拆出 SGLang request、prefill 和 decode 阶段指标；这些指标嵌在每个 Sample 的 trace 事件里。

设计选择：
- `compute_perf_metrics_from_samples` 汇总 response token 速度和 non-generation time，再调用 `_compute_sglang_request_perf_metrics` 从 trace 的 `sglang_generate` span_end 事件提取字段。

Explain：
`_iter_sglang_generate_attrs` 只接受 `event.type == "span_end"` 且 `name == "sglang_generate"` 的事件，attrs 必须是 dict。字段表在文件顶部定义，最后对每个 metric key 计算统计量。

来源：slime/ray/rollout.py L1324-L1407

Code：

```python
def compute_perf_metrics_from_samples(args, samples, rollout_time):
    non_generation_time = [sample.non_generation_time for sample in samples]
    log_dict = {}
    log_dict["rollout_time"] = rollout_time
    ...
    token_perf([sample.response_length for sample in samples], non_generation_time, key="")
    token_perf([sample.effective_response_length for sample in samples], non_generation_time, key="effective_")
    log_dict |= _compute_sglang_request_perf_metrics(samples)
    return log_dict

def _iter_sglang_generate_attrs(all_samples: list[Sample]):
    for sample in all_samples:
        trace = getattr(sample, "trace", None)
        if not isinstance(trace, dict):
            continue
        for event in trace.get("events") or []:
            if event.get("type") != "span_end" or event.get("name") != "sglang_generate":
                continue
            attrs = event.get("attrs")
            if isinstance(attrs, dict):
                yield attrs
```

代码逻辑：
- perf dict 总是包含 rollout_time。
- token throughput 按原始 response_length 和 effective_response_length 计算两套。
- trace 中没有 SGLang span 时返回空 metrics。
- 非数值或非 finite attrs 会被过滤。

为什么这样写：
- rollout 总耗时包含数据处理、工具调用等 non-generation 时间；拆出 SGLang span 更利于定位瓶颈。
- 通过 Sample trace 聚合，不要求 RolloutManager 直接侵入每次 SGLang 调用。

不变量与失败模式：
- trace schema 必须包含 `events` list 和 span attrs。
- 没有 trace 时不会报错，只是缺少 SGLang 细分指标。

Comment：
性能统计保持旁路，不影响 train_data 构造，但能解释 rollout 吞吐变化。
