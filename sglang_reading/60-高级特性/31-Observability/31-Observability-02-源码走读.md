---
type: batch-doc
module: 31-Observability
batch: "31"
doc_type: walkthrough
title: "可观测性 · 源码走读"
tags:
  - sglang/batch/31
  - sglang/module/observability
  - sglang/doc/walkthrough
aliases:
  - "02-源码走读"
updated: 2026-07-05
---

# 可观测性 · 源码走读

> 走读主线：SGLang 的可观测性分成三层：HTTP 侧把 Prometheus registry 暴露到 `/metrics`；Tokenizer/Scheduler 侧分别创建 metrics collector 并写入请求与调度指标；单请求 latency、request log、weight update 等旁路事件按各自时机写入。

---

## 1. HTTP 暴露面与 collector 注入

### 1.1 lifespan 只在 enable_metrics 开启时挂载 metrics 和函数计时

问题与约束：
- HTTP worker 可能是 single-tokenizer，也可能是 multi-tokenizer worker；Prometheus middleware 必须在 worker 生命周期内根据实际 server args 决定是否挂载。

设计选择：
- FastAPI lifespan 先取当前 worker 的 `server_args`，再在 `server_args.enable_metrics` 为真时调用 `add_prometheus_middleware(app)` 和 `enable_func_timer()`。

Explain：
single-tokenizer 模式从 app 属性取 server args；multi-tokenizer 模式先执行 `init_multi_tokenizer()` 重建 worker 状态。metrics 的挂载发生在这两种模式分支之后，因此 worker 使用的 server args 是同一入口。

来源：python/sglang/srt/entrypoints/http_server.py L261-L276

Code：

```python
@asynccontextmanager
async def lifespan(fast_api_app: FastAPI):
    if getattr(fast_api_app, "is_single_tokenizer_mode", False):
        server_args = fast_api_app.server_args
        warmup_thread_kwargs = fast_api_app.warmup_thread_kwargs
        thread_label = "Tokenizer"
    else:
        server_args = await init_multi_tokenizer()
        warmup_thread_kwargs = dict(server_args=server_args)
        thread_label = f"MultiTokenizer-{_global_state.tokenizer_manager.worker_id}"

    if server_args.enable_metrics:
        add_prometheus_middleware(app)
        enable_func_timer()
```

代码逻辑：
- worker 先确定 single/multi tokenizer 模式。
- metrics 开关来自当前 worker 的 server args。
- Prometheus middleware 和函数计时同一开关控制。

为什么这样写：
- multi-worker 下每个 worker 都需要自己的 lifespan 初始化，不能依赖主进程对象。
- `enable_metrics=False` 时不挂载 metrics，也不启用函数计时，减少热路径开销。

不变量与失败模式：
- multi-tokenizer 模式必须能从 shared state 初始化 server args。
- `/metrics` 只有 enable_metrics 为真时才挂载。

Comment：
HTTP 层的 metrics 入口是 lifespan，不是 route 装饰器。

### 1.2 add_prometheus_middleware 使用 multiprocess registry 暴露 /metrics

问题与约束：
- SGLang 可能有多个 HTTP worker 或子进程，Prometheus client 需要 multiprocess registry；同时 `/metrics` 不应因为 Starlette Mount 产生 307 redirect。

设计选择：
- 延迟 import `prometheus_client`，创建 `CollectorRegistry` 和 `multiprocess.MultiProcessCollector`，把 ASGI app Mount 到 `/metrics`，并覆盖 path_regex。

Explain：
源码注释强调必须在设置 `PROMETHEUS_MULTIPROC_DIR` 后再 import `prometheus_client`。`path_regex` 设为 `^/metrics(?P<path>.*)$`，避免访问 `/metrics` 时被重定向。

来源：python/sglang/srt/utils/common.py L1589-L1599

Code：

```python
def add_prometheus_middleware(app):
    from prometheus_client import CollectorRegistry, make_asgi_app, multiprocess

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    metrics_route = Mount("/metrics", make_asgi_app(registry=registry))

    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)
```

代码逻辑：
- 创建独立 Prometheus registry。
- 注册 multiprocess collector。
- Prometheus ASGI app 作为 Mount route 加入 FastAPI。
- path regex 手动修正。

为什么这样写：
- multiprocess collector 会从 Prometheus multiproc 目录聚合子进程指标。
- 延迟 import 避免 client 在环境变量设置前初始化错误目录。

不变量与失败模式：
- `PROMETHEUS_MULTIPROC_DIR` 必须先设置。
- 如果 multiprocess 目录不可写，Prometheus client 会在写 metric 时失败。

Comment：
这一层只负责暴露指标，不负责采集指标。

### 1.3 _setup_and_run_http_server 只给 response tracking 加 middleware

问题与约束：
- Prometheus `/metrics` 在 lifespan 中挂载；HTTP response tracking middleware 则需要在 server setup 阶段按全局 app 配置添加。

设计选择：
- `_setup_and_run_http_server` 在写入 global state 后，如果 `enable_metrics` 为真，调用 `add_prometheus_track_response_middleware(app)`。

Explain：
这与 lifespan 中的 `add_prometheus_middleware` 不是同一个函数：前者用于 response tracking middleware，后者用于 `/metrics` endpoint。两者都受 `enable_metrics` 控制。

来源：python/sglang/srt/entrypoints/http_server.py L2289-L2296

Code：

```python
if tokenizer_manager is not None:
    tokenizer_manager._subprocess_watchdog = subprocess_watchdog

if server_args.enable_metrics:
    add_prometheus_track_response_middleware(app)
```

代码逻辑：
- watchdog 挂到 tokenizer manager。
- metrics 开启时添加 response tracking middleware。
- single/multi tokenizer app state 设置在后续分支继续完成。

为什么这样写：
- response tracking middleware 是 HTTP app 层能力，适合在 setup 阶段配置。
- `/metrics` endpoint 需要 worker lifespan 初始化后再挂载。

不变量与失败模式：
- 只开启 metrics 时才有 response tracking 数据。
- 如果 app 已经添加过同类 middleware，重复添加可能造成重复观测。

Comment：
HTTP 层 metrics 有两个入口：setup 阶段的 response middleware 和 lifespan 阶段的 scrape endpoint。

### 1.4 resolve_collector_class 支持 embedded 场景替换 collector

问题与约束：
- 默认 collector 使用 prometheus_client；Ray Serve LLM 等 embedded 场景可能希望把指标写到自定义后端。

设计选择：
- `ServerArgs.stat_loggers` 按 role 注册 collector class；每个 collector 实例化点调用 `resolve_collector_class`，没有注册则回退默认 class。

Explain：
源码定义了 scheduler、tokenizer、storage、radix_cache、expert_dispatch 五个 role。`server_args` 或 `stat_loggers` 为 None 时都直接返回默认 class。

来源：python/sglang/srt/observability/metrics_collector.py L189-L210

Code：

```python
STAT_LOGGER_ROLE_SCHEDULER = "scheduler"
STAT_LOGGER_ROLE_TOKENIZER = "tokenizer"
STAT_LOGGER_ROLE_STORAGE = "storage"
STAT_LOGGER_ROLE_RADIX_CACHE = "radix_cache"
STAT_LOGGER_ROLE_EXPERT_DISPATCH = "expert_dispatch"

def resolve_collector_class(
    server_args: Optional[ServerArgs], role: str, default_cls: type
) -> type:
    if server_args is None:
        return default_cls
    stat_loggers = getattr(server_args, "stat_loggers", None)
    if not stat_loggers:
        return default_cls
    return stat_loggers.get(role, default_cls)
```

代码逻辑：
- role 常量统一定义。
- 解析函数容忍空 server args。
- stat logger map 未配置时不改变行为。
- 配置命中时返回替代 class。

为什么这样写：
- collector 替换保持在构造边界，不污染业务代码的 metrics 调用。
- 默认 prometheus 路径和 embedded 自定义路径共用同一接口。

不变量与失败模式：
- 自定义 collector class 必须兼容默认 collector 的方法和构造参数。
- role key 写错会静默回退默认 class。

Comment：
Observability 的 DI 扩展点集中在 collector class，而不是每个 metric 写入点。

---

## 2. Scheduler 侧指标采集

### 2.1 SchedulerMetricsCollector.init_new 决定 rank、label 和 collector 实例

问题与约束：
- Scheduler 有 TP/PP/CP/DP 多 rank；默认只应由统计 rank 写 scheduler metrics，但也要支持 all-scheduler metrics 和 KV cache events。

设计选择：
- `init_new` 计算 `enable_metrics`、`is_stats_logging_rank`、`current_scheduler_metrics_enabled` 和 `enable_kv_cache_events`，再在 metrics 开启时创建 collector 并附加 labels。

Explain：
`is_stats_logging_rank` 只看 `ps.attn_tp_rank == 0`。当前 scheduler 是否写 metrics 还要看 `enable_metrics_for_all_schedulers`。KV cache events 更严格，只在 `pp_rank/attn_tp_rank/attn_cp_rank` 都为 0 且配置存在时启用。

来源：python/sglang/srt/observability/metrics_collector.py L1027-L1085

Code：

```python
enable_metrics = server_args.enable_metrics
is_stats_logging_rank = ps.attn_tp_rank == 0
current_scheduler_metrics_enabled = enable_metrics and (
    is_stats_logging_rank or server_args.enable_metrics_for_all_schedulers
)
enable_kv_cache_events = bool(
    server_args.kv_events_config
    and ps.pp_rank == 0
    and ps.attn_tp_rank == 0
    and ps.attn_cp_rank == 0
)
collector: Optional[SchedulerMetricsCollector] = None
if enable_metrics:
    engine_type = DisaggregationMode.to_engine_type(
        server_args.disaggregation_mode
    )
    labels = {
        "model_name": server_args.served_model_name,
        "engine_type": engine_type,
        "tp_rank": tp_rank,
        "pp_rank": pp_rank,
        "moe_ep_rank": ps.moe_ep_rank,
    }
    if dp_rank is not None:
        labels["dp_rank"] = dp_rank
    if server_args.extra_metric_labels:
        labels.update(server_args.extra_metric_labels)
    scheduler_collector_cls = resolve_collector_class(
        server_args, STAT_LOGGER_ROLE_SCHEDULER, cls
    )
    collector = scheduler_collector_cls(...)
```

代码逻辑：
- 先计算开关和 rank 角色。
- metrics 开启时才创建 collector。
- labels 写入模型名、engine type 和并行 rank。
- priority、DP rank 和 extra labels 按需加入。

为什么这样写：
- 多 rank 全量上报会造成重复指标；默认只让统计 rank 写主指标。
- labels 需要在 collector 构造时固定，后续写 metric 只传值。

不变量与失败模式：
- `server_args.enable_metrics=False` 时 collector 为 None。
- extra labels 与已有 label 同名会覆盖原 label。
- KV events 只在严格 rank 0 路径开启。

Comment：
Scheduler metrics 的第一层过滤是 rank 与开关，不是 reporter。

### 2.2 Scheduler.init_metrics_collector 把 collector context 接入 Scheduler

问题与约束：
- Scheduler 需要在 parallel state 就绪后才知道 tp/pp/dp rank；同时 IPC channel 是否启用 metrics 也要用同一套 rank 逻辑。

设计选择：
- `init_metrics_collector` 调用 `SchedulerMetricsCollector.init_new`，保存 context 和 collector；`init_ipc_channels` 用 `enable_metrics` 与 rank 条件决定 `metrics_enabled`。

Explain：
Scheduler 把 LoRA、hierarchical cache、priority scheduling 的开关传给 collector，使 collector 只创建或写入对应指标。IPC channel 的 metrics_enabled 不直接看 collector，而是复用 rank 条件，避免非统计 rank 暴露重复 IPC metrics。

来源：python/sglang/srt/managers/scheduler.py L590-L620

Code：

```python
def init_metrics_collector(
    self, tp_rank: int, pp_rank: int, dp_rank: Optional[int]
) -> None:
    self.metrics_collector_context = SchedulerMetricsCollector.init_new(
        server_args=self.server_args,
        ps=self.ps,
        tp_rank=tp_rank,
        pp_rank=pp_rank,
        dp_rank=dp_rank,
        enable_priority_scheduling=self.enable_priority_scheduling,
        enable_lora=self.enable_lora,
        enable_hierarchical_cache=self.enable_hierarchical_cache,
    )
    self.metrics_collector = self.metrics_collector_context.collector

self.ipc_channels = SchedulerIpcChannels.create(
    ...,
    metrics_enabled=self.server_args.enable_metrics
    and (
        self.ps.attn_tp_rank == 0
        or self.server_args.enable_metrics_for_all_schedulers
    ),
    ...
)
```

代码逻辑：
- Scheduler 保存完整 metrics context。
- collector 引用单独保存给 reporter 和其他组件使用。
- IPC metrics 使用 enable_metrics 与 rank 条件。

为什么这样写：
- context 中除了 collector，还有 current rank 是否应记录指标、KV events 是否开启等状态。
- IPC channel 在创建时就要知道是否记录 metrics，不能等 reporter tick。

不变量与失败模式：
- `init_metrics_collector` 必须在 reporter 使用 collector 前完成。
- collector 为 None 时，调用方需要通过 context/开关短路。

Comment：
Scheduler 把 metrics collector 当作运行时依赖注入给 reporter、KV events 和 weight updater 等组件。

### 2.3 MetricsReporter 在 stats tick 上汇总 SchedulerStats 并 log_stats

问题与约束：
- Scheduler 的运行状态分散在 waiting queue、grammar manager、memory pool、PD queue、LoRA pool 和 HiCache 中；Prometheus 写入需要统一的 stats snapshot。

设计选择：
- MetricsReporter 在周期 tick 中计算 cache hit、queue count、memory pool、PD、utilization、LoRA、HiCache 等字段，写入 `self.stats` 后调用 `metrics_collector.log_stats(self.stats)`。

Explain：
代码段中 `pool_stats.update_scheduler_stats(self.stats)` 把 memory pool 相关字段写入 stats；PD disaggregation 根据当前模式填 prefill 或 decode queue；最后再写 utilization、LoRA、HiCache 并 emit KV metrics。

来源：python/sglang/srt/managers/scheduler_components/metrics_reporter.py L606-L661

Code：

```python
priority_enabled = self.scheduler.enable_priority_scheduling
effective_input_tokens = (
    prefill_stats.log_input_tokens
    - prefill_stats.reprocessed_log_input_tokens
)
effective_hit_tokens = (
    prefill_stats.log_hit_tokens - prefill_stats.reprocessed_log_hit_tokens
)
total_tokens = effective_input_tokens + effective_hit_tokens
cache_hit_rate = (
    effective_hit_tokens / total_tokens if total_tokens > 0 else 0.0
)

self.stats.num_running_reqs = prefill_stats.num_running_reqs
self.stats.num_queue_reqs = QueueCount.from_reqs(
    self.scheduler.waiting_queue, priority_enabled
)
self.stats.num_grammar_queue_reqs = len(self.scheduler.grammar_manager)
self.stats.cache_hit_rate = cache_hit_rate
pool_stats.update_scheduler_stats(self.stats)

self._calculate_utilization()
self.stats.fwd_occupancy = self.fwd_occupancy
self._update_lora_metrics()
self._log_hicache_stats()
self.metrics_collector.log_stats(self.stats)
self.scheduler.kv_events_publisher.emit_kv_metrics()
```

代码逻辑：
- cache hit rate 用有效 input/cache token 计算。
- QueueCount 保留总数和 priority breakdown。
- memory pool 通过 pool_stats 写入。
- 最后统一调用 collector。

为什么这样写：
- reporter 负责把调度器内部状态整理成稳定数据结构，collector 只负责写 metric。
- 先聚合 stats 再批量 log，减少 Prometheus 写入散落在 scheduler 热路径各处。

不变量与失败模式：
- `self.metrics_collector` 必须非空且当前 scheduler metrics 开启。
- stats 字段必须与 collector.log_stats 读取字段保持同步。

Comment：
MetricsReporter 是 Scheduler 内部状态到 Prometheus 指标的转换器。

### 2.4 log_stats 把 SchedulerStats 批量映射到 gauge/histogram

问题与约束：
- Prometheus 指标名在 collector 初始化时已经创建；每次 tick 只应更新当前数值，并按可选功能开关写入 LoRA/HiCache/streaming session 指标。

设计选择：
- `log_stats` 顺序写 basics、memory pool、spec、retract、PD、utilization、scheduler policy、CUDA graph、LoRA、HiCache、streaming session 和 routing key。

Explain：
priority queue count 通过 `_log_gauge_queue_count` 同时写总量和 per-priority breakdown。LoRA、HiCache、streaming session 都由 collector 构造参数控制，未启用时不写对应 gauge。

来源：python/sglang/srt/observability/metrics_collector.py L1260-L1358

Code：

```python
def log_stats(self, stats: SchedulerStats) -> None:
    self._log_gauge_queue_count(self.num_running_reqs, stats.num_running_reqs)
    self._log_gauge_queue_count(self.num_queue_reqs, stats.num_queue_reqs)
    self._log_gauge(self.num_grammar_queue_reqs, stats.num_grammar_queue_reqs)
    self._log_gauge(self.gen_throughput, stats.gen_throughput)
    self._log_gauge(self.cache_hit_rate, stats.cache_hit_rate)

    self._log_gauge(self.token_usage, stats.token_usage)
    self._log_gauge(self.full_token_usage, stats.full_token_usage)
    self._log_gauge(self.num_used_tokens, stats.num_used_tokens)
    self._log_gauge(self.kv_available_tokens, stats.kv_available_tokens)
    self._log_gauge(self.kv_evictable_tokens, stats.kv_evictable_tokens)

    self._log_gauge(self.spec_accept_length, stats.spec_accept_length)
    self._log_gauge(self.spec_accept_rate, stats.spec_accept_rate)
    self._log_gauge(self.num_retracted_reqs, stats.num_retracted_reqs)
    self._log_gauge(self.num_paused_reqs, stats.num_paused_reqs)

    if self.enable_lora:
        self._log_gauge(self.lora_pool_slots_used, stats.lora_pool_slots_used)
        self._log_gauge(self.lora_pool_slots_total, stats.lora_pool_slots_total)
        self._log_gauge(self.lora_pool_utilization, stats.lora_pool_utilization)

    self.last_log_time = time.perf_counter()
```

代码逻辑：
- 基础队列和吞吐指标每 tick 写入。
- memory pool 指标分 usage ratio 和 absolute token count。
- speculative/retract/PD/utilization 等指标按 stats 字段写入。
- 函数末尾更新 `last_log_time`。

为什么这样写：
- 大部分 scheduler 指标是瞬时状态，用 gauge 表示最直接。
- 可选功能指标由 feature flag 控制，避免没有对应功能时创建或更新无意义指标。

不变量与失败模式：
- Gauge/Histogram 对象必须在 collector 初始化时按同一 label set 创建。
- stats 中缺字段会在 log_stats 触发属性错误。

Comment：
`log_stats` 是 `/metrics` 中大多数 scheduler 指标的直接写入点。

### 2.5 increment_realtime_tokens 记录 decode/prefill token 增量

问题与约束：
- 周期 gauge 反映当前状态，但实时 token throughput 需要 counter 增量；DP cooperation 场景还要按 cooperation labels 额外拆分。

设计选择：
- `increment_realtime_tokens` 对 prefill_compute、prefill_cache、decode 三种 mode 逐项 inc；有 `dp_cooperation_info` 时同时写 DP cooperation counter。

Explain：
decode 路径中 MetricsReporter 每 iteration 计算 `decode_tokens = batch.batch_size() + num_correct_drafts` 并调用这个函数。delta 为 0 的 mode 被跳过，减少无意义 label 写入。

来源：python/sglang/srt/observability/metrics_collector.py L1203-L1239

Code：

```python
def increment_realtime_tokens(
    self,
    dp_cooperation_info: Optional[DPCooperationInfo],
    prefill_compute_tokens=0,
    prefill_cache_tokens=0,
    decode_tokens=0,
):
    for mode, delta in [
        ("prefill_compute", prefill_compute_tokens),
        ("prefill_cache", prefill_cache_tokens),
        ("decode", decode_tokens),
    ]:
        if delta == 0:
            continue
        self.realtime_tokens_total.labels(**self.labels, mode=mode).inc(delta)
        if dp_cooperation_info is not None:
            self.dp_cooperation_realtime_tokens_total.labels(
                **self.labels,
                mode=mode,
                **dp_cooperation_info.to_labels(),
            ).inc(delta)
```

代码逻辑：
- 三种 token mode 共用一组 counter。
- delta 为 0 时不写。
- DP cooperation info 存在时追加合作标签。

为什么这样写：
- counter 比 gauge 更适合 token 增量，Prometheus 可用 rate 计算吞吐。
- mode label 统一了 prefill/decode token 统计入口。

不变量与失败模式：
- delta 必须是非负增量。
- DP cooperation labels 必须与 counter labelnames 一致。

Comment：
这类 counter 是热路径实时吞吐指标，和周期 `log_stats` 的状态 gauge 互补。

### 2.6 emit_constants 在启动后写一次容量类 gauge

问题与约束：
- 某些指标是容量或启动信息，不应每个 stats tick 重复计算；但 dashboard 需要这些固定值做容量规划和冷启动诊断。

设计选择：
- `emit_constants` 接收 max token、page size、num pages、context len、startup GPU memory 等值，直接写入对应 gauge。

Explain：
如果 `max_running_requests_under_SLO` 不为 None，会额外写入该 gauge。其他字段总是写入，作为模型加载完成后的常量型可观测数据。

来源：python/sglang/srt/observability/metrics_collector.py L1385-L1408

Code：

```python
def emit_constants(
    self,
    max_total_num_tokens: int,
    max_running_requests_under_SLO: Optional[int],
    engine_startup_time: float,
    engine_load_weights_time: float,
    page_size: int,
    num_pages: int,
    context_len: int,
    startup_available_gpu_memory_gb: float,
) -> None:
    self._log_gauge(self.max_total_num_tokens, max_total_num_tokens)
    if max_running_requests_under_SLO is not None:
        self._log_gauge(
            self.max_running_requests_under_SLO, max_running_requests_under_SLO
        )
    self._log_gauge(self.engine_startup_time, engine_startup_time)
    self._log_gauge(self.engine_load_weights_time, engine_load_weights_time)
    self._log_gauge(self.page_size, page_size)
    self._log_gauge(self.num_pages, num_pages)
    self._log_gauge(self.context_len, context_len)
    self._log_gauge(
        self.startup_available_gpu_memory_gb, startup_available_gpu_memory_gb
    )
```

代码逻辑：
- 所有常量通过 `_log_gauge` 写当前 label set。
- SLO 下最大 running request 是可选项。
- 函数不更新 `last_log_time`。

为什么这样写：
- 容量类指标变化频率低，按事件写一次更清晰。
- dashboard 可以把这些 gauge 与动态 usage 指标组合计算利用率。

不变量与失败模式：
- 调用时机必须在这些容量值已知之后。
- 如果模型重载改变容量，需要再次调用更新 gauge。

Comment：
常量型 gauge 是 scheduler 动态指标的上下文。

---

## 3. Tokenizer、请求日志与请求级 latency

### 3.1 TokenizerManager 初始化 tokenizer collector 和 CPU watchdog

问题与约束：
- Tokenizer 侧指标和 Scheduler 侧指标不同：它关心 prompt/generation token histogram、TTFT、ITL、E2E latency、custom labels 等请求层指标。

设计选择：
- `init_metric_collector_watchdog` 在 `enable_metrics` 下构造 `TokenizerMetricsCollector`，labels 包含 model_name、engine_type、priority/custom/extra labels，并启动 tokenizer CPU monitor thread。

Explain：
Tokenizer collector 也走 `resolve_collector_class`，role 是 `tokenizer`。bucket 配置从 server args 传入，允许用户自定义 TTFT、E2E、ITL histogram bucket。

来源：python/sglang/srt/managers/tokenizer_manager.py L527-L558

Code：

```python
def init_metric_collector_watchdog(self):
    if self.enable_metrics:
        engine_type = DisaggregationMode.to_engine_type(
            self.server_args.disaggregation_mode
        )

        labels = {
            "model_name": self.server_args.served_model_name,
            "engine_type": engine_type,
        }
        if self.enable_priority_scheduling:
            labels["priority"] = ""
        if self.server_args.tokenizer_metrics_allowed_custom_labels:
            for label in self.server_args.tokenizer_metrics_allowed_custom_labels:
                labels[label] = ""
        if self.server_args.extra_metric_labels:
            labels.update(self.server_args.extra_metric_labels)
        tokenizer_collector_cls = resolve_collector_class(
            self.server_args,
            STAT_LOGGER_ROLE_TOKENIZER,
            TokenizerMetricsCollector,
        )
        self.metrics_collector = tokenizer_collector_cls(
            server_args=self.server_args,
            labels=labels,
            bucket_time_to_first_token=self.server_args.bucket_time_to_first_token,
            bucket_e2e_request_latency=self.server_args.bucket_e2e_request_latency,
            bucket_inter_token_latency=self.server_args.bucket_inter_token_latency,
        )

        start_cpu_monitor_thread("tokenizer")
```

代码逻辑：
- metrics 开启时才创建 tokenizer collector。
- label set 在构造时固定。
- custom label 白名单先放空值，后续请求可覆盖。
- 同时启动 tokenizer CPU monitor。

为什么这样写：
- Tokenizer metrics 是请求入口层指标，不能完全由 Scheduler collector 覆盖。
- custom labels 需要白名单，避免任意请求制造无限 label cardinality。

不变量与失败模式：
- 请求 custom labels 只能使用允许的 label key。
- bucket 参数不合法会影响 Prometheus histogram 创建。

Comment：
Tokenizer collector 是请求层 metrics 的入口，Scheduler collector 是调度层 metrics 的入口。

### 3.2 generate_request 在请求进入 tokenizer 后记录 request log

问题与约束：
- request log 要捕获原始 Generate/Embedding request 以及白名单 headers；同时必须在真正 tokenization/scheduler dispatch 前执行，才能记录失败前的输入。

设计选择：
- `generate_request` normalize 请求并初始化 req state 后，调用 `request_logger.log_received_request(obj, self.tokenizer, request)`；之后才等待 pause、解析 LoRA、tokenize 并发送 scheduler。

Explain：
这条日志路径独立于 Prometheus metrics，受 request logger 自己的开关和 level 控制。它位于 `model_update_lock.reader_lock` 外侧之前，可以在模型更新 pause 等待前记录已收到请求。

来源：python/sglang/srt/managers/tokenizer_manager.py L589-L635

Code：

```python
async def generate_request(
    self,
    obj: Union[GenerateReqInput, EmbeddingReqInput],
    request: Optional[fastapi.Request] = None,
):
    self.auto_create_handle_loop()
    obj.normalize_batch_and_arguments()
    self._set_default_priority(obj)
    ...
    self._init_req_state(obj, request)
    try:
        if self.server_args.language_only:
            self._handle_epd_disaggregation_encode_request(obj)

        self.request_logger.log_received_request(obj, self.tokenizer, request)

        async with self.is_pause_cond:
            await self.is_pause_cond.wait_for(lambda: not self.is_pause)

        async with self.model_update_lock.reader_lock:
            await self._validate_and_resolve_lora(obj)
            if obj.is_single:
                tokenized_obj = await self._tokenize_one_request(obj)
                self._send_one_request(tokenized_obj)
```

代码逻辑：
- 请求先 normalize 并设置默认 priority。
- `_init_req_state` 建立 request state。
- request logger 记录收到的对象。
- 后续才进入 pause/model update/LoRA/tokenize 流程。

为什么这样写：
- request log 的目标是审计入口请求，而不是只记录成功进入 scheduler 的请求。
- 在等待 pause 前记录，可以解释模型更新期间请求堆积。

不变量与失败模式：
- request logger 必须能处理 GenerateReqInput 和 EmbeddingReqInput。
- 如果 request logger level 需要 decode input ids，必须有 tokenizer。

Comment：
request logger 是运维审计路径，不是 Prometheus 指标路径。

### 3.3 RequestLogger 控制字段截断、headers 和 input_ids decode

问题与约束：
- 请求日志可能包含长 prompt 或敏感字段；同时线上通常只允许记录白名单 headers。

设计选择：
- `RequestLogger` 根据 metadata 中的 max length、skip names 和 log format 输出结构化 JSON 或普通文本；level >= 2 且只有 input_ids 时可用 tokenizer decode text。

Explain：
白名单 headers 来自默认 `x-smg-routing-key` 加环境变量扩展。JSON 模式写事件名 `request.received`，普通模式写字符串；如果 `log_requests=False` 立即返回。

来源：python/sglang/srt/utils/request_logger.py L30-L130

Code：

```python
_DEFAULT_WHITELISTED_HEADERS = ["x-smg-routing-key"]
WHITELISTED_HEADERS = _DEFAULT_WHITELISTED_HEADERS + [
    h.lower() for h in envs.SGLANG_LOG_REQUEST_HEADERS.get()
]

def _extract_whitelisted_headers(
    request: Optional[fastapi.Request],
) -> Optional[Dict[str, str]]:
    if request is None:
        return None
    return {h: v for h in WHITELISTED_HEADERS if (v := request.headers.get(h))}

def log_received_request(self, obj, tokenizer: Any = None, request: Optional[fastapi.Request] = None) -> None:
    if not self.log_requests:
        return

    max_length, skip_names, _ = self.metadata
    headers = _extract_whitelisted_headers(request)
    if self.log_requests_format == "json":
        log_data = {
            "rid": obj.rid,
            "obj": _transform_data_for_logging(obj, max_length, skip_names),
        }
        if headers:
            log_data["headers"] = headers
        log_json(self.targets, "request.received", log_data)
```

```python
if (
    self.log_requests_level >= 2
    and obj.text is None
    and obj.input_ids is not None
    and tokenizer is not None
):
    decoded = tokenizer.decode(obj.input_ids, skip_special_tokens=False)
    obj.text = decoded
```

代码逻辑：
- headers 只取白名单。
- JSON 与 text 两种输出格式分支。
- data transform 做字段截断和跳过。
- level >= 2 可把 input ids 解码成 text。

为什么这样写：
- 日志可读性与隐私/体积风险需要通过 level 和 skip list 控制。
- header 白名单避免把认证或其他敏感 header 全量写入日志。

不变量与失败模式：
- input_ids 可能是 batch list，源码对嵌套 list 有单独 decode 分支。
- 解码会修改 `obj.text`，调用方需接受这一副作用。

Comment：
请求日志是一条独立观测线，适合回放和审计，不适合直接当低基数 metric。

### 3.4 TokenizerMetricsCollector 定义请求层 histogram 和 counter

问题与约束：
- Tokenizer 侧要统计 prompt token、generation token、spec verify、TTFT、ITL、E2E 等请求维度指标；这些指标的 bucket 需要可配置。

设计选择：
- `TokenizerMetricsCollector.__init__` 延迟 import prometheus classes，按 labels 创建 counter/histogram，并用 server args bucket 覆盖默认 bucket。

Explain：
源码片段展示 prompt/generation token counter 和 token length histogram 的创建。后续 TTFT/ITL/E2E histogram 也在同一 collector 内创建并由 observe 方法写入。

来源：python/sglang/srt/observability/metrics_collector.py L1411-L1505

Code：

```python
class TokenizerMetricsCollector(_StatLoggerDIMixin):
    def __init__(
        self,
        server_args: Optional[ServerArgs] = None,
        labels: Dict[str, str] = None,
        bucket_time_to_first_token: Optional[List[float]] = None,
        bucket_inter_token_latency: Optional[List[float]] = None,
        bucket_e2e_request_latency: Optional[List[float]] = None,
    ) -> None:
        from prometheus_client import Counter as _PromCounter
        from prometheus_client import Histogram as _PromHistogram

        Counter = self._counter_cls or _PromCounter
        Histogram = self._histogram_cls or _PromHistogram

        self.labels = labels or {}

        self.prompt_tokens_total = Counter(
            name="sglang:prompt_tokens_total",
            documentation="Number of prefill tokens processed.",
            labelnames=labels.keys(),
        )
        self.generation_tokens_total = Counter(
            name="sglang:generation_tokens_total",
            documentation="Number of generation tokens processed.",
            labelnames=labels.keys(),
        )
        self.prompt_tokens_histogram = Histogram(
            name="sglang:prompt_tokens_histogram",
            documentation="Histogram of prompt token length.",
            labelnames=labels.keys(),
            buckets=generate_buckets(
                server_args.prompt_tokens_buckets, default_bucket_prompt_tokens
            ),
        )
```

代码逻辑：
- collector 支持 DI 替换 prometheus classes。
- labels 缓存在 `self.labels`。
- counter 与 histogram 都使用同一 label key set。
- bucket 从 server args 生成。

为什么这样写：
- 请求层 metrics 的 label set 必须固定，Prometheus 不允许同一 metric 使用不同 labelnames。
- bucket 可配置可以适配短 prompt 和超长上下文部署。

不变量与失败模式：
- `labels` 不能为 None 后还访问 `labels.keys()`；调用方实际传入 labels dict。
- bucket 配置过细会增大 Prometheus 资源消耗。

Comment：
Tokenizer collector 更关注请求分布，而 Scheduler collector 更关注运行时状态。

### 3.5 TokenizerManager 记录 TTFT 与 ITL

问题与约束：
- 首 token latency 和 inter-token latency 的观察时机不同：首 token 只能记录一次，后续每批 completion tokens 要按增量和时间间隔记录。

设计选择：
- 收到 response 时构造 labels，合并 custom labels 和 priority；未观察 TTFT 且非 prefill disaggregation 节点时记录 TTFT，否则按 completion token 增量记录 ITL。

Explain：
`state.ttft_observed` 防止重复记录 TTFT。ITL 的 observe 方法接收 interval 和 `num_new_tokens`，collector 内部按每个新 token 的平均 interval 更新 histogram。

来源：python/sglang/srt/managers/tokenizer_manager.py L2403-L2428

Code：

```python
custom_labels = getattr(state.obj, "custom_labels", None)
labels = dict(self.metrics_collector.labels)
if custom_labels:
    labels.update(custom_labels)
if self.enable_priority_scheduling:
    priority = getattr(state.obj, "priority", None)
    if priority is not None:
        labels["priority"] = str(priority)
if (
    not state.ttft_observed
    and self.disaggregation_mode != DisaggregationMode.PREFILL
):
    state.ttft_observed = True
    state.last_completion_tokens = completion_tokens
    self.metrics_collector.observe_time_to_first_token(
        labels, state.time_stats.get_first_token_latency()
    )
else:
    num_new_tokens = completion_tokens - state.last_completion_tokens
    if num_new_tokens:
        self.metrics_collector.observe_inter_token_latency(
            labels,
            state.time_stats.get_interval(),
            num_new_tokens,
        )
        state.time_stats.set_last_time()
        state.last_completion_tokens = completion_tokens
```

代码逻辑：
- labels 从 collector 默认 labels 拷贝。
- custom labels 和 priority 动态覆盖。
- 首 token路径只执行一次。
- 后续 token 按 completion token 增量记录 ITL。

为什么这样写：
- TTFT 是请求级单点指标，重复写会扭曲分布。
- ITL 更适合按 token 增量计入 histogram。

不变量与失败模式：
- custom labels 必须在 tokenizer collector 允许的 labelnames 内。
- prefill disaggregation 节点不记录 TTFT。

Comment：
TokenizerManager 是 TTFT/ITL 的天然观察点，因为它看到客户端响应流。

### 3.6 Tokenizer collector 的 observe 方法优化 histogram 写入

问题与约束：
- ITL 可能一次 response chunk 增加多个 token；逐 token 调 Histogram.observe 会增加开销。

设计选择：
- `observe_inter_token_latency` 计算平均 interval，然后直接更新 histogram 的 `_sum` 和第一个满足 bound 的 bucket，按 `num_new_tokens` 批量 inc。

Explain：
`observe_time_to_first_token` 走标准 histogram observe。`check_time_to_first_token_straggler` 会在样本数达到 100 后用内部 buckets 估算 p99 threshold，判断当前 TTFT 是否是 straggler。

来源：python/sglang/srt/observability/metrics_collector.py L1688-L1718

Code：

```python
def observe_time_to_first_token(self, labels: Dict[str, str], value: float):
    self.histogram_time_to_first_token.labels(**labels).observe(value)

def check_time_to_first_token_straggler(self, value: float) -> bool:
    his = self.histogram_time_to_first_token.labels(**self.labels)
    total_observations = sum(bucket._value for bucket in his._buckets)
    if total_observations < 100:
        return False
    p99_threshold = total_observations * 0.99
    cumulative_count = 0
    for i, bucket in enumerate(his._buckets):
        cumulative_count += bucket._value
        if cumulative_count > p99_threshold:
            return value >= his._upper_bounds[i]
    return False

def observe_inter_token_latency(
    self, labels: Dict[str, str], internval: float, num_new_tokens: int
):
    adjusted_interval = internval / num_new_tokens
    his = self.histogram_inter_token_latency.labels(**labels)
    his._sum.inc(internval)
    for i, bound in enumerate(his._upper_bounds):
        if adjusted_interval <= bound:
            his._buckets[i].inc(num_new_tokens)
            break
```

代码逻辑：
- TTFT 使用标准 observe。
- straggler 检查读取 histogram 内部 bucket。
- ITL 批量更新 sum 和 bucket。
- bucket 增量等于新增 token 数。

为什么这样写：
- ITL 是高频指标，批量写可减少 Python 调用和 Prometheus overhead。
- straggler 检查需要基于已有 TTFT 分布，样本不足时直接不判定。

不变量与失败模式：
- `num_new_tokens` 必须大于 0，否则会除零。
- 直接访问 histogram 内部字段依赖 prometheus_client 实现细节。

Comment：
这是性能敏感的 tokenizer metrics 写入路径。

### 3.7 ReqTimeStats 用 RequestStageConfig 区分 trace 与 metrics

问题与约束：
- 单请求生命周期包含 tokenize、dispatch、queue、prefill、decode、PD transfer、spec、CPU run batch 等阶段；并非每个阶段都应该写 Prometheus latency。

设计选择：
- `RequestStageConfig` 同时定义 stage name、trace hierarchy level 和 `metrics_is_observed`；`RequestStage` 枚举每个阶段的配置。

Explain：
例如 `PREFILL_FORWARD`、`PREFILL_CHUNKED_FORWARD`、PD bootstrap/transfer 和 decode waiting 等设置了 `metrics_is_observed=True`；tokenize、decode_forward 等可能只用于 trace 层级，不一定写 metrics。

来源：python/sglang/srt/observability/req_time_stats.py L82-L220

Code：

```python
@dataclass
class RequestStageConfig:
    stage_name: str
    level: int = 0
    metrics_is_observed: bool = False

class RequestStage:
    TOKENIZE = RequestStageConfig(
        "tokenize",
        level=1,
    )
    REQUEST_PROCESS = RequestStageConfig(
        "request_process",
        level=2,
        metrics_is_observed=True,
    )
    PREFILL_FORWARD = RequestStageConfig(
        "prefill_forward",
        level=1,
        metrics_is_observed=True,
    )
    PREFILL_BOOTSTRAP = RequestStageConfig(
        "prefill_bootstrap",
        level=1,
        metrics_is_observed=True,
    )
    DECODE_WAITING = RequestStageConfig(
        "decode_waiting",
        level=1,
        metrics_is_observed=True,
    )
```

代码逻辑：
- 每个阶段有稳定 stage name。
- level 用于 tracing 层级。
- metrics flag 控制是否写 per-stage latency histogram。

为什么这样写：
- tracing 需要更细层级，Prometheus latency 则要控制基数和写入量。
- 同一 stage config 同时服务 trace 和 metric，避免名字漂移。

不变量与失败模式：
- stage name 会进入 metrics label，不能随意改名。
- metrics flag 设错会导致关键阶段缺失或低价值阶段过量上报。

Comment：
ReqTimeStats 是请求级 latency 的统一阶段词表。

### 3.8 ReqTimeStats 绑定 collector 并同时支持 tracing

问题与约束：
- 单请求对象可能跨进程序列化；metrics collector 和 tracing context 都需要在请求生命周期中动态绑定和恢复。

设计选择：
- `set_metrics_collector` 在 collector 存在时启用 metrics；`observe_per_stage_req_latency` 只在 metrics 开启且 stage 允许观测时写 collector；`trace_slice` 独立检查 tracing context。

Explain：
`init_trace_ctx` 根据 request id、bootstrap room 和外部 trace header 创建 `TraceReqContext`；如果 tracing 未开启，则换成 `TraceNullContext`。metrics 与 tracing 可以独立启用。

来源：python/sglang/srt/observability/req_time_stats.py L260-L305

Code：

```python
def set_metrics_collector(
    self, collector: Union[SchedulerMetricsCollector, TokenizerMetricsCollector]
):
    if collector:
        self.enable_metrics = True
        self.metrics_collector = collector

def observe_per_stage_req_latency(self, stage: RequestStageConfig, latency: float):
    if self.enable_metrics and stage.metrics_is_observed:
        self.metrics_collector.observe_per_stage_req_latency(
            stage.stage_name, latency
        )

def init_trace_ctx(
    self,
    rid: str,
    bootstrap_room: Optional[int],
    external_trace_header: Optional[Dict[str, str]] = None,
):
    self.trace_ctx = TraceReqContext(...)
    if not self.trace_ctx.tracing_enable:
        self.trace_ctx = TraceNullContext()

def trace_slice(self, stage: RequestStageConfig, start_time: float, end_time: float, attrs: Optional[Dict] = None):
    if self.trace_ctx.tracing_enable:
        _slice = TraceSliceContext(
            slice_name=stage.stage_name,
            start_time_ns=convert_time_to_realtime_ns(start_time),
            end_time_ns=convert_time_to_realtime_ns(end_time),
            level=stage.level,
            attrs=attrs,
        )
        self.trace_ctx.trace_slice(_slice)
```

代码逻辑：
- collector 存在才启用 metrics。
- per-stage latency 由 stage flag 二次过滤。
- trace context 独立初始化。
- trace slice 用 realtime ns 写入。

为什么这样写：
- metrics 与 tracing 开关独立，运维可以只开一种。
- 请求对象跨进程时不能默认持有可用 collector，需要显式绑定。

不变量与失败模式：
- collector 必须实现 `observe_per_stage_req_latency`。
- trace 时间转换依赖已校准的 monotonic/realtime 差值。

Comment：
ReqTimeStats 把请求级 metric 和 trace 共用同一阶段事件，但输出路径分离。

### 3.9 ReqTimeStats 在状态 setter 中写 queue/prefill/decode latency

问题与约束：
- 请求生命周期的时间点分散在不同调度事件中；如果只在 finish 时回推，容易丢失 retract、chunked prefill、PD decode 等边界。

设计选择：
- 各 setter 在时间点首次出现时立即计算上一个阶段 latency，并调用 `observe_per_stage_req_latency` 与 `trace_slice`。

Explain：
`set_wait_queue_entry_time` 根据 disaggregation mode 把 scheduler receive 到 wait queue 的时间归为 request_process/prefill_bootstrap/decode_transferred。`set_forward_entry_time` 记录 queue time，并打开 prefill/decode forward trace。prefill 和 decode finish setter 分别记录 forward 或 decode loop latency。

来源：python/sglang/srt/observability/req_time_stats.py L711-L841

Code：

```python
def set_wait_queue_entry_time(self, ts=None):
    ts = ts or time.perf_counter()
    if self.wait_queue_entry_time == 0.0:
        if self.enable_metrics or self.trace_ctx.tracing_enable:
            if self.disagg_mode == DisaggregationMode.PREFILL:
                stage = RequestStage.PREFILL_BOOTSTRAP
                slice_start_time = self.prefill_bootstrap_queue_entry_time
            elif self.disagg_mode == DisaggregationMode.DECODE:
                stage = RequestStage.DECODE_TRANSFERRED
                slice_start_time = self.decode_transfer_queue_entry_time
            else:
                stage = RequestStage.REQUEST_PROCESS
                slice_start_time = self.scheduler_recv_time
            self.observe_per_stage_req_latency(stage, ts - slice_start_time)
            self.trace_slice(stage, slice_start_time, ts)
    self.wait_queue_entry_time = ts
```

```python
def set_forward_entry_time(self, ts=None):
    ts = ts or time.perf_counter()
    if self.forward_entry_time == 0.0:
        self.forward_entry_time = ts
        self.last_forward_entry_time = ts
        if self.enable_metrics:
            self.metrics_collector.observe_queue_time(self.get_queueing_time())
        if self.enable_metrics or self.trace_ctx.tracing_enable:
            if self.disagg_mode == DisaggregationMode.DECODE:
                stage = RequestStage.DECODE_WAITING
            else:
                stage = RequestStage.PREFILL_WAITING
            self.observe_per_stage_req_latency(stage, ts - self.wait_queue_entry_time)
            self.trace_slice(stage, self.wait_queue_entry_time, ts)
```

代码逻辑：
- setter 只在首次进入阶段时写主要 latency。
- disaggregation mode 决定 stage 类型。
- queue time 额外写 queue histogram。
- trace 与 metric 判断分开。

为什么这样写：
- 时间点出现时立即记录，能保留复杂调度路径中的阶段边界。
- 不同 disaggregation mode 下同一时间点语义不同，需要在 setter 内分流。

不变量与失败模式：
- 前置时间戳必须已设置，否则 latency 可能从 0 或错误时间计算。
- 重入或 retract 路径需要 setter 分支维护 `last_*` 字段。

Comment：
请求级 latency 不是单个 finish 回调，而是多个状态转换点共同写出的。

---

## 4. 事件型指标：weight update 与单次写入

### 4.1 _observe_weight_load 用 context manager 记录热更新耗时

问题与约束：
- 权重热更新期间 engine 会 pause，周期性 `log_stats` tick 可能不发生；如果只靠 reporter，就会漏掉 update duration。

设计选择：
- WeightUpdater 提供 `_observe_weight_load(source)` context manager，在 `finally` 中调用 metrics collector 的 `observe_weight_load(duration, source)`。

Explain：
source 用于区分 disk、distributed、tensor、ipc 四种 update path。即使 update 中间抛错，只要 context 退出且 collector 存在，duration 仍会被记录。

来源：python/sglang/srt/managers/scheduler_components/weight_updater.py L86-L100

Code：

```python
@contextmanager
def _observe_weight_load(self, source: str) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if self.metrics_collector is not None:
            self.metrics_collector.observe_weight_load(
                time.perf_counter() - t0, source
            )
```

代码逻辑：
- 进入 context 时记录开始时间。
- update 逻辑在 yield 中执行。
- finally 中按 source 写 duration。
- collector 为 None 时静默跳过。

为什么这样写：
- 事件型指标不能依赖周期 tick。
- context manager 能覆盖多个 update path，避免每个函数重复计时代码。

不变量与失败模式：
- WeightUpdater 需要被注入 metrics_collector。
- source 字符串应保持有限集合，避免 Prometheus label 高基数。

Comment：
这是典型的 edge-triggered metric：事件结束时立即写入。

### 4.2 observe_weight_load 写 weight_load_duration_seconds gauge

问题与约束：
- 热更新耗时需要按来源区分；但更新完成后只关心最近一次耗时，不需要每 tick 重复观察。

设计选择：
- collector 的 `observe_weight_load` 直接用 `weight_load_duration_seconds.labels(..., source=source).set(duration)` 写 gauge。

Explain：
源码注释说明 engine pause 时 `log_stats` 不会触发，所以这里必须 inline 写。source label 的预期集合是 disk、distributed、tensor、ipc。

来源：python/sglang/srt/observability/metrics_collector.py L1143-L1149

Code：

```python
def observe_weight_load(self, duration_seconds: float, source: str) -> None:
    self.weight_load_duration_seconds.labels(**self.labels, source=source).set(
        duration_seconds
    )
```

代码逻辑：
- 使用 collector 固定 labels。
- 额外添加 source label。
- 设置最近一次 duration。

为什么这样写：
- gauge 适合表达“最近一次 update duration”。
- source label 能把 checkpoint engine IPC 和普通 disk/distributed/tensor 更新区分开。

不变量与失败模式：
- `weight_load_duration_seconds` 必须以 source 作为 labelname 创建。
- source 不能使用动态路径或请求 id。

Comment：
这条 metric 解释热更新停顿时间，和 scheduler 的 paused request 指标互补。

### 4.3 update_weights_from_ipc 在 checkpoint-engine 路径记录 source=ipc

问题与约束：
- CheckpointEngine IPC 热更新需要更新 TP worker 和可能存在的 draft worker；成功后还要 flush cache 并做 TP CPU group barrier。

设计选择：
- `update_weights_from_ipc` 用 `_observe_weight_load("ipc")` 包裹整个更新流程；TP 成功后 flush cache，draft worker 可选更新，最后 barrier 并返回结构化输出。

Explain：
如果 TP worker 成功但 draft worker 失败，源码仍会因为 `tp_success` flush cache，然后记录错误并返回失败。context manager 保证无论成功失败都会记录 duration。

来源：python/sglang/srt/managers/scheduler_components/weight_updater.py L166-L178

Code：

```python
def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
    with self._observe_weight_load("ipc"):
        success, message = self.tp_worker.update_weights_from_ipc(recv_req)
        tp_success = success
        if success and self.draft_worker is not None:
            success, message = self.draft_worker.update_weights_from_ipc(recv_req)
        if tp_success:
            self.flush_cache_after_weight_update(recv_req)
        if not success:
            logger.error(message)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return UpdateWeightsFromIPCReqOutput(success=success, message=message)
```

代码逻辑：
- IPC update source label 固定为 `"ipc"`。
- TP worker 先更新。
- draft worker 存在时跟随更新。
- TP 成功触发 cache flush。
- barrier 后返回结果。

为什么这样写：
- IPC 更新是 checkpoint engine 集成路径，单独 source label 便于 dashboard 区分。
- barrier 确保 TP ranks 在返回前完成同步。

不变量与失败模式：
- `tp_cpu_group` 必须已初始化。
- flush cache 失败会 assert。
- draft worker 失败会让整体 success 为 False。

Comment：
CheckpointEngine 的热更新可观测性在这里和 scheduler metrics 接上。
