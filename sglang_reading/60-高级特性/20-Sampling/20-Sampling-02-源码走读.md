---
type: batch-doc
module: 20-Sampling
batch: "20"
doc_type: walkthrough
title: "Sampling · 源码走读"
tags:
  - sglang/batch/20
  - sglang/module/sampling
  - sglang/doc/walkthrough
aliases:
  - "02-源码走读"
updated: 2026-07-05
---

# Sampling · 源码走读

> 走读主线：API 层的 `SamplingParams` 先归一化 stop、温度、top-k/top-p、grammar 字段；Scheduler 将带约束请求送入 `GrammarManager` 的异步编译队列；`ScheduleBatch` 构造 `SamplingBatchInfo`，把 per-request 参数批量搬到设备；`ModelRunner.sample` 在 logits 上施加 penalty、grammar mask 和 logit bias；`Sampler` 负责 greedy/top-k/top-p/min-p/确定性采样；结果处理阶段推进 grammar FSM 和 penalty 状态。

---

## 1. SamplingParams：API 参数到内部约束

### 1.1 `SamplingParams` 同时保存 API 字段和内部字段

问题与约束：
- API 需要暴露 `stop`、`stop_regex`、`json_schema`、`regex`、`ebnf`、`structural_tag` 等字段。
- Scheduler IPC 不应重复携带 API alias，内部逻辑需要使用归一化后的字段。

设计选择：
- 用 `msgspec.Struct` 定义 API 参数；再用 `stop_strs`、`stop_regex_strs`、最大长度和 `is_normalized` 保存内部状态。

Explain：
`SamplingParams` 是采样路径的输入合同。它把随机采样参数、惩罚参数、结构化输出约束、stream/logprob 相关字段放在同一个对象里；内部字段用于在 `normalize()` 后稳定传给 Scheduler 和 Detokenizer。

来源：python/sglang/srt/sampling/sampling_params.py L75-L120

Code：

```python
class SamplingParams(msgspec.Struct, kw_only=True, omit_defaults=True):
    max_new_tokens: Optional[int] = 128
    stop: Optional[Union[str, List[str]]] = (
        None
    )
    stop_token_ids: Optional[Set[int]] = None
    stop_regex: Optional[Union[str, List[str]]] = (
        None
    )
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = TOP_K_ALL
    min_p: float = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    min_new_tokens: int = 0
    n: int = 1
    json_schema: Optional[str] = None
    regex: Optional[str] = None
    ebnf: Optional[str] = None
    structural_tag: Optional[str] = None
    ignore_eos: bool = False
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    no_stop_trim: bool = False
    custom_params: Optional[Dict[str, CustomParamValue]] = None
    stream_interval: Optional[int] = None
    logit_bias: Optional[Dict[str, float]] = None
    sampling_seed: Optional[int] = None

    stop_strs: Optional[Union[str, List[str]]] = None
    stop_regex_strs: Optional[Union[str, List[str]]] = None
    stop_str_max_len: int = 0
    stop_regex_max_len: int = 0
    is_normalized: bool = False
```

代码逻辑：
- API 字段保存用户传入的采样、stop、结构化输出和自定义参数。
- `TOP_K_ALL` 表示不限制 top-k。
- 内部 stop 字段由 `__post_init__` 或 `normalize()` 填充。
- `omit_defaults=True` 让默认值不出现在序列化结果里。

为什么这样写：
- 采样参数既要对 API 友好，又要对内部 IPC 稳定。
- 将 API alias 和内部字段拆开，可以在 normalize 后清空 alias，减少下游歧义。

不变量与失败模式：
- 归一化前后 `stop/stop_regex` 和 `stop_strs/stop_regex_strs` 的语义必须一致。
- `json_schema`、`regex`、`ebnf` 互斥，不能同时设置多个。
- `custom_params` 需要保持可 msgspec 序列化。

Comment：
这个结构是 sampling 的入口，后面的 grammar、penalty、sampler 都从这里取 per-request 配置。

### 1.2 `__post_init__` 把空值归一到默认值，并把温度 0 转为 greedy

问题与约束：
- HTTP/JSON 调用者可能显式传 `null`，非 optional 采样字段不能因此崩溃。
- 温度为 0 的语义是 greedy，但后续 sampler 仍期望温度为正数。

设计选择：
- `__post_init__` 对 None 字段回填默认值；`0 <= temperature < eps` 时设置 `temperature=1.0` 且 `top_k=1`。

Explain：
这段是采样参数的第一层防御。它将 `stop` alias 拷到 `stop_strs`，过滤 stop token ids，将 top-k `-1` 映射为全词表，并把零温 greedy 表达成 `top_k=1`。

来源：python/sglang/srt/sampling/sampling_params.py L122-L175

Code：

```python
def __post_init__(self):
    if self.is_normalized:
        return

    self.stop_strs = self.stop
    if self.stop_token_ids:
        filtered = {int(t) for t in self.stop_token_ids if t is not None}
        self.stop_token_ids = filtered or None
    else:
        self.stop_token_ids = None
    self.stop_regex_strs = self.stop_regex
    self.temperature = self.temperature if self.temperature is not None else 1.0
    self.top_p = self.top_p if self.top_p is not None else 1.0
    self.top_k = self.top_k if self.top_k is not None else -1
    self.min_p = self.min_p if self.min_p is not None else 0.0
    self.frequency_penalty = (
        self.frequency_penalty if self.frequency_penalty is not None else 0.0
    )
    self.presence_penalty = (
        self.presence_penalty if self.presence_penalty is not None else 0.0
    )
    self.repetition_penalty = (
        self.repetition_penalty if self.repetition_penalty is not None else 1.0
    )

    if 0 <= self.temperature < _SAMPLING_EPS:
        self.temperature = 1.0
        self.top_k = 1
    if self.top_k == -1:
        self.top_k = TOP_K_ALL
```

代码逻辑：
- 已归一化对象直接返回，避免反序列化后重置内部字段。
- stop alias 拷入内部字段。
- stop token ids 过滤 None 并转 int。
- 各采样字段将 None 转回默认值。
- 近似零温转成 top-k=1 greedy。
- `top_k=-1` 转成全词表常量。

为什么这样写：
- API 允许 `null` 更宽松，但内部采样路径应看到确定类型和值。
- greedy 用 `top_k=1` 表达，可以复用后续 batch flag 和 sampler 短路逻辑。

不变量与失败模式：
- `is_normalized=True` 的对象不能再被 `__post_init__` 覆盖。
- 温度负数不会在这里修正，会在 `verify()` 报错。
- stop token ids 为空集合时会被压成 None。

Comment：
采样参数的“语义规范化”从这里开始，避免下游处理 API 输入的各种边界形态。

### 1.3 `verify` 和 `normalize` 校验取值并计算 stop 缓冲长度

问题与约束：
- stop string 和 stop regex 需要 Detokenizer 保留足够 token/字符窗口才能正确匹配。
- 结构化输出 grammar 字段必须互斥，否则无法决定使用哪种 grammar compiler。

设计选择：
- `verify()` 检查范围和互斥关系；`normalize()` 将 stop 字段统一成列表，计算 stop string token 长度和 regex 最大长度，并清空 API alias。

Explain：
`normalize()` 把请求侧 stop 约束变成 Detokenizer 可用的最大缓冲长度：字符串 stop 用 tokenizer 编码长度，regex stop 用静态 regex 最大长度估计；最后调用 `raise_if_tokenizer_required` 确认需要 tokenizer 的功能可用。

来源：python/sglang/srt/sampling/sampling_params.py L176-L276

Code：

```python
def verify(self, vocab_size):
    if not math.isfinite(self.temperature) or self.temperature < 0.0:
        raise ValueError(
            f"temperature must be a non-negative finite number, got {self.temperature}."
        )
    if not 0.0 < self.top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
    if self.top_k < 1 or self.top_k == -1:
        raise ValueError(
            f"top_k must be -1 (disable) or at least 1, got {self.top_k}."
        )
    if self.logit_bias is not None:
        for token_id in self.logit_bias:
            if not 0 <= int(token_id) < vocab_size:
                raise ValueError(
                    f"logit_bias must has keys in [0, {vocab_size - 1}], got "
                    f"{token_id}."
                )

    grammars = [
        self.json_schema,
        self.regex,
        self.ebnf,
    ]
    if sum(x is not None for x in grammars) > 1:
        raise ValueError("Only one of regex, json_schema, or ebnf can be set.")

def normalize(self, tokenizer):
    if self.stop_strs is None:
        self.stop_strs = []
        self.stop_str_max_len = 0
    else:
        if isinstance(self.stop_strs, str):
            self.stop_strs = [self.stop_strs]
        stop_str_max_len = 0
        for stop_str in self.stop_strs:
            if tokenizer is not None:
                stop_str_ids = tokenizer.encode(stop_str, add_special_tokens=False)
                stop_str_max_len = max(stop_str_max_len, len(stop_str_ids))
            else:
                stop_str_max_len = max(stop_str_max_len, len(stop_str))
        self.stop_str_max_len = stop_str_max_len
```

代码逻辑：
- 检查 temperature、top_p、min_p、top_k、penalty、min/max token 等范围。
- 检查 logit bias token id 是否在 vocab 内。
- 检查 `json_schema/regex/ebnf` 至多设置一个。
- stop string 统一成列表并计算最大 token 长度。
- stop regex 统一成列表并用 regex 结构估计最大长度。
- 清空 `stop` 和 `stop_regex` alias，并设置 `is_normalized=True`。

为什么这样写：
- 采样 kernel 对参数范围很敏感，早失败比运行时 silent wrong 更好。
- stop 最大长度用于流式输出保留足够上下文。
- 清空 alias 能让 IPC 序列化只携带内部字段。

不变量与失败模式：
- tokenizer 缺失时，依赖 tokenizer 的 stop/min_new_tokens 功能会被 `raise_if_tokenizer_required` 拦截。
- regex 最大长度遇到无法静态处理的结构会保守返回大值。
- `structural_tag` 不在旧互斥列表内，后续 grammar manager 按优先级选择字段。

Comment：
SamplingParams 的职责不是执行采样，而是把用户意图转成内部可校验、可序列化的采样配置。

## 2. Grammar 异步编译与请求排队

### 2.1 BaseGrammarObject 定义 token filter 所需接口

问题与约束：
- 不同 grammar backend 需要统一对接 Scheduler、SamplingBatchInfo 和 Sampler。
- 采样前需要构造 vocab mask，采样后需要推进 grammar 状态，jump-forward 也需要可选接口。

设计选择：
- 基类规定 `accept_token`、`rollback`、`allocate_vocab_mask`、`fill_vocab_mask`、`move_vocab_mask`、`apply_vocab_mask`、`copy` 等方法。

Explain：
`BaseGrammarObject` 是受约束解码的对象协议。SamplingBatchInfo 只依赖这些接口生成 mask；结果处理器只依赖 `accept_token` 推进状态；缓存命中时用 `copy()` 复制 grammar 状态。

来源：python/sglang/srt/constrained/base_grammar_backend.py L42-L119

Code：

```python
class BaseGrammarObject:

    def __init__(self):
        self._finished = False
        self.grammar_stats = None
        self.current_token = None

    def maybe_init_reasoning(self, reasoning: bool):
        pass

    def accept_token(self, token: int) -> None:
        raise NotImplementedError()

    def rollback(self, k: int):
        raise NotImplementedError()

    def is_terminated(self):
        return False

    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        raise NotImplementedError()

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        raise NotImplementedError()

    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
        raise NotImplementedError()

    @staticmethod
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        raise NotImplementedError()
```

代码逻辑：
- 保存 finished 状态、grammar stats 和当前 token。
- 定义 reasoning 初始化钩子。
- 定义 token 接受、回滚、终止判断。
- 定义 vocab mask 分配、填充、移动和应用接口。
- 定义 jump-forward 相关接口。

为什么这样写：
- scheduler、sampler 和不同 grammar backend 解耦，只通过统一接口交互。
- `copy()` 支持缓存 compiled grammar，但每个请求仍要有自己的 matcher 状态。

不变量与失败模式：
- backend 子类必须实现 mask 与 token 状态相关方法。
- `is_terminated()` 默认 False，具体 backend 必须覆盖终止语义。
- `copy()` 默认返回 self，状态型 backend 若不覆盖会共享状态，通常不安全。

Comment：
这个接口是 structured output 贯穿采样前后两端的基础。

### 2.2 BaseGrammarBackend 用缓存和 Future 隔离 grammar 编译开销

问题与约束：
- JSON schema、regex、EBNF 或 structural tag 编译可能较慢，不能阻塞 Scheduler 主循环。
- 相同 grammar 在不同请求之间应复用编译结果，但每个请求要独立状态。

设计选择：
- backend 持有 `ThreadPoolExecutor` 和 cache；cache hit 返回 grammar copy，cache miss 提交 `_init_value_dispatch` 并返回 `Future`。

Explain：
`get_cached_or_future_value` 是 grammar 异步编译入口。它以 `(type, string)` 作为 key；命中缓存时复制 grammar 并按请求是否需要 reasoning 初始化；未命中时交给线程池后台编译。

来源：python/sglang/srt/constrained/base_grammar_backend.py L131-L210

Code：

```python
class BaseGrammarBackend:
    _enable_strict_thinking: bool = False

    def __init__(self):
        self.executor = ThreadPoolExecutor()
        self.cache: Dict[Tuple[str, str], BaseGrammarObject] = {}

    def _init_value_dispatch(
        self, key: Tuple[str, str], require_reasoning: bool
    ) -> BaseGrammarObject:
        s = time.perf_counter()
        key_type, key_string = key
        if key_type == "json":
            grammar = self.dispatch_json(key_string)
        elif key_type == "regex":
            grammar = self.dispatch_regex(key_string)
        elif key_type == "ebnf":
            grammar = self.dispatch_ebnf(key_string)
        elif key_type == "structural_tag":
            grammar = self.dispatch_structural_tag(key_string)
        else:
            grammar = self.dispatch_fallback(key_type, key_string)

        if grammar is not None and grammar.grammar_stats is not None:
            grammar.grammar_stats.compilation_time = time.perf_counter() - s
        return grammar

    def get_cached_or_future_value(
        self, key: Tuple[str, str], require_reasoning: bool
    ) -> Tuple[BaseGrammarObject | Future[BaseGrammarObject], bool]:
        value = self.cache.get(key)
        if value:
            copied_value = value.copy()
            copied_value.maybe_init_reasoning(require_reasoning)
            return copied_value, True
        value = self.executor.submit(self._init_value_dispatch, key, require_reasoning)
        return value, False
```

代码逻辑：
- 构造线程池和 grammar cache。
- `_init_value_dispatch` 根据 key type 调用对应 compile/dispatch 方法。
- 记录 compilation time。
- cache hit 时复制 grammar object。
- cache miss 时提交线程池任务，返回 Future 和 miss 标记。
- `set_cache` 将编译结果写回 cache。

为什么这样写：
- grammar 编译慢且可复用，异步加缓存能降低主调度路径抖动。
- copy 保证同一 compiled grammar 的不同请求拥有独立 matcher 状态。

不变量与失败模式：
- key 必须能唯一表示 grammar 类型和文本。
- backend 的 dispatch 方法要把编译失败转换为 `InvalidGrammarObject` 或抛异常由 manager 捕获。
- 线程池 Future 如果长期不完成，需要 manager 超时取消。

Comment：
这里是 structured output 进入异步队列的核心机制。

### 2.3 `create_grammar_backend` 按配置选择 backend 并处理 strict thinking

问题与约束：
- 后端可能是 xgrammar、outlines、llguidance、none 或自定义插件。
- `enable_strict_thinking` 需要 token filter 能力，不能静默降级到 none。

设计选择：
- 先查自定义 registry；默认后端按名称实例化；xgrammar tokenizer 不支持时，strict thinking 抛错，否则降级 none；有 reasoning parser 时包一层 `ReasonerGrammarBackend`。

Explain：
`create_grammar_backend` 是 Scheduler 初始化 grammar 管线的入口。它决定后续请求能否使用 token filtering，也决定是否在普通 grammar 外再加 reasoning 标签过滤。

来源：python/sglang/srt/constrained/base_grammar_backend.py L223-L313

Code：

```python
def create_grammar_backend(
    server_args: ServerArgs,
    tokenizer,
    vocab_size: int,
    eos_token_ids: Optional[set] = None,
    think_end_id: Optional[int] = None,
) -> Optional[BaseGrammarBackend]:
    name = server_args.grammar_backend

    if name in GRAMMAR_BACKEND_REGISTRY:
        return GRAMMAR_BACKEND_REGISTRY[name](
            server_args, tokenizer, vocab_size, eos_token_ids
        )

    if name == "outlines":
        from sglang.srt.constrained.outlines_backend import OutlinesGrammarBackend
        grammar_backend = OutlinesGrammarBackend(
            tokenizer,
            whitespace_pattern=server_args.constrained_json_whitespace_pattern,
        )
    elif name == "xgrammar":
        from sglang.srt.constrained.xgrammar_backend import (
            TokenizerNotSupportedError,
            XGrammarGrammarBackend,
        )
        eos_list = list(eos_token_ids) if eos_token_ids else None

        try:
            grammar_backend = XGrammarGrammarBackend(
                tokenizer,
                vocab_size=vocab_size,
                model_eos_token_ids=eos_list,
                any_whitespace=not server_args.constrained_json_disable_any_whitespace,
            )
        except TokenizerNotSupportedError as e:
            if server_args.enable_strict_thinking:
                raise ValueError(...) from e
            server_args.grammar_backend = "none"
            return None
    elif name == "none":
        if server_args.enable_strict_thinking:
            raise ValueError(...)
        return None
```

代码逻辑：
- 自定义 backend registry 具有最高优先级。
- outlines/xgrammar/llguidance 分别构造对应 backend。
- xgrammar 初始化失败时，strict thinking 抛错，否则降级为 none。
- grammar_backend none 且 strict thinking 开启时抛错。
- reasoning parser 和 think end id 存在时，用 `ReasonerGrammarBackend` 包装。

为什么这样写：
- 结构化输出功能依赖 backend，但普通 serving 不应因为 xgrammar 不支持 tokenizer 而无法启动。
- strict thinking 的安全边界更强，不能在无 token filter 后端时继续运行。

不变量与失败模式：
- 无效 backend name 会抛 `ValueError`。
- xgrammar tokenizer info 构造失败会进入降级或抛错路径。
- reasoning wrapper 依赖底层 backend 已成功创建。

Comment：
这个工厂把“功能可用性”和“启动降级策略”集中处理。

### 2.4 GrammarManager 在请求入队时决定编译、排队或 abort

问题与约束：
- 一个请求至多选择一种显式 grammar 约束，但 strict thinking 可能在无显式约束时也需要 token filter。
- grammar_backend 为 none 时，结构化输出请求不能继续进入 waiting queue。

设计选择：
- `process_req_with_grammar` 检查四类约束字段，构造 cache key；cache miss 放入 `grammar_queue`，cache hit 直接绑定 grammar；无 backend 或 cached invalid grammar 则 abort。

Explain：
这个函数发生在 Scheduler 接收请求后、加入 waiting queue 前。它把“能立即运行”和“需要等 grammar 编译”的请求分开，避免模型 forward 等待同步编译。

来源：python/sglang/srt/constrained/grammar_manager.py L25-L140

Code：

```python
class GrammarManager:
    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        self.server_args = scheduler.server_args
        self.grammar_queue: List[Req] = []
        if not self.server_args.skip_tokenizer_init:
            self.grammar_backend = create_grammar_backend(
                self.server_args,
                scheduler.tokenizer,
                scheduler.model_config.vocab_size,
                scheduler.model_config.hf_eos_token_id,
                think_end_id=scheduler.model_config.think_end_id,
            )
        else:
            self.grammar_backend = None

    def process_req_with_grammar(self, req: Req) -> bool:
        add_to_grammar_queue = False
        if (
            req.sampling_params.json_schema is not None
            or req.sampling_params.regex is not None
            or req.sampling_params.ebnf is not None
            or req.sampling_params.structural_tag is not None
        ):
            if self.grammar_backend is None:
                error_msg = "Grammar-based generation (json_schema, regex, ebnf, structural_tag) is not supported when the server is launched with --grammar-backend none"
                req.set_finish_with_abort(error_msg)
            else:
                if req.sampling_params.json_schema is not None:
                    key = ("json", req.sampling_params.json_schema)
                elif req.sampling_params.regex is not None:
                    key = ("regex", req.sampling_params.regex)
                elif req.sampling_params.ebnf is not None:
                    key = ("ebnf", req.sampling_params.ebnf)
                elif req.sampling_params.structural_tag:
                    key = ("structural_tag", req.sampling_params.structural_tag)

                value, cache_hit = self.grammar_backend.get_cached_or_future_value(
                    key, req.require_reasoning
                )
                req.grammar = value
```

代码逻辑：
- 初始化时创建 grammar backend，并记录 DP/TP grammar sync group。
- 请求带显式约束且无 backend 时 abort。
- 按 json/regex/ebnf/structural_tag 优先级构造 key。
- 从 backend 获取 grammar object 或 Future。
- cache miss 时记录 `req.grammar_key` 并加入 grammar queue。
- cache hit 为 invalid grammar 时 abort，否则应用 reasoning budget。
- strict thinking 且无显式约束时初始化纯 reasoning grammar。

为什么这样写：
- 请求在进入 waiting queue 前完成 grammar 可用性判断，可以避免后续 batch 构造碰到未编译对象。
- cache hit 直接运行，cache miss 异步等待，降低 structured output 首次请求的阻塞范围。

不变量与失败模式：
- `req.grammar` 在 queue 中可能是 Future，进入 batch 前必须被替换为 grammar object。
- grammar backend none 时，显式结构化约束必然 abort。
- cached `InvalidGrammarObject` 会复用失败结果，不重复编译同一坏 grammar。

Comment：
这是请求级 structured output 的分流点：可运行、等待编译、或直接失败。

### 2.5 `get_ready_grammar_requests` 轮询 Future，并在 DP/TP 组内同步就绪状态

问题与约束：
- 多 rank 必须在同一批请求的 grammar 都 ready 后才一起进入 waiting queue，否则 DP/TP rank 的 batch 会不一致。
- Future 可能长期不完成，需要超时并缓存失败。

设计选择：
- 在固定 poll interval 内检查 Future.done；超时计数超过上限时标记 failed；多 rank 用 `all_gather_object` 取 ready 交集和 failed 并集。

Explain：
这个函数在 Scheduler 取新 prefill batch 前调用。它把 ready 请求从 grammar queue 移出，拿到 Future result 后写入 backend cache，并为 failed 请求 cancel Future、缓存 timeout invalid object、设置 abort。

来源：python/sglang/srt/constrained/grammar_manager.py L142-L243

Code：

```python
def get_ready_grammar_requests(self) -> List[Req]:
    assert self.grammar_backend
    ready_req_idxs: set[int] = set()
    failed_req_idxs: set[int] = set()

    start_time = time.perf_counter()
    while time.perf_counter() - start_time < self.SGLANG_GRAMMAR_POLL_INTERVAL:
        for i, req in enumerate(self.grammar_queue):
            if i in ready_req_idxs:
                continue

            if req.finished() or req.grammar is None:
                ready_req_idxs.add(i)
                continue

            assert isinstance(req.grammar, futures.Future), f"{req=}"
            if req.grammar.done():
                ready_req_idxs.add(i)

        time.sleep(self.SGLANG_GRAMMAR_POLL_INTERVAL / 10)

    for i, req in enumerate(self.grammar_queue):
        if i not in ready_req_idxs:
            self.grammar_queue[i].grammar_wait_ct += 1
            if (
                self.grammar_queue[i].grammar_wait_ct
                >= self.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS
            ):
                failed_req_idxs.add(i)

    if self.grammar_sync_size == 1:
        synced_ready_req_idxs = ready_req_idxs
        synced_failed_req_idxs = failed_req_idxs
    else:
        all_gather_output = [None] * self.grammar_sync_size
        torch.distributed.all_gather_object(
            all_gather_output,
            (ready_req_idxs, failed_req_idxs),
            group=self.grammar_sync_group,
        )
        synced_ready_req_idxs = set.intersection(*[x[0] for x in all_gather_output])
        synced_failed_req_idxs = set.union(*[x[1] for x in all_gather_output])
```

代码逻辑：
- 在 poll interval 内轮询 queue。
- finished 或已 abort 的请求视为 ready，可以从 queue 移除。
- 未完成 Future 增加 wait count，超过上限标记 failed。
- 单 rank 直接使用本地 ready/failed。
- 多 rank 收集所有 rank 的 ready/failed，ready 取交集、failed 取并集。
- ready 请求取 Future result、写 cache、应用 reasoning budget。
- failed 请求 cancel Future、缓存 timeout invalid grammar、设置 abort。
- 从 grammar queue 删除 ready/failed 请求。

为什么这样写：
- ready 交集保证所有 rank 都具备同一 grammar 对象后再运行。
- failed 并集保证任何 rank 编译失败都会让全组一致 abort。
- 超时缓存 invalid object，避免后续相同 grammar 重复卡住。

不变量与失败模式：
- `self.grammar_backend` 必须存在。
- queue index 在 all-gather 同步期间必须对各 rank 表示同一请求顺序。
- Future result 抛异常会被转成 `InvalidGrammarObject`。

Comment：
grammar 编译是异步的，但进入模型 batch 前必须重新变成同步一致的状态。

### 2.6 Scheduler 只把 grammar ready 的请求放入 waiting queue

问题与约束：
- 新请求如果 grammar 未编译完成，不能立即进入 waiting queue；否则 batch 构造时会拿到 Future 而不是 grammar object。
- 每轮 prefill 选择前要把刚 ready 的 grammar 请求重新加入调度队列。

设计选择：
- 请求处理时根据 `process_req_with_grammar` 返回值决定是否 `_add_request_to_queue`；取新 prefill batch 前轮询 ready grammar 并加入 waiting queue。

Explain：
Scheduler 侧的两处调用把 grammar queue 接入主调度：新请求没有进入 grammar queue 时直接进入 waiting queue；有等待 grammar 时，每轮 prefill 前取 ready 请求再加入 waiting queue。

来源：python/sglang/srt/managers/scheduler.py L2235-L2250

Code：

```python
added_to_grammar_queue = self.grammar_manager.process_req_with_grammar(req)
if not added_to_grammar_queue:
    self._add_request_to_queue(req)
```

代码逻辑：
- 请求完成基础校验后调用 grammar manager。
- 返回 False 表示无需等待 grammar，直接加入调度队列。
- 返回 True 表示请求已放入 grammar queue，暂不进入 waiting queue。

为什么这样写：
- 让 grammar 编译不阻塞请求接收主路径。
- 避免未 ready grammar 请求进入 model execution。

不变量与失败模式：
- 如果请求被 grammar manager abort，仍会按返回值决定是否入队，后续 finish/abort 状态要被调度器处理。
- grammar queue 和 waiting queue 的请求状态必须互斥。

Comment：
这是一条很短但重要的连接线：grammar 编译队列在这里接入 Scheduler。

### 2.7 Scheduler 每轮 prefill 前回收 ready grammar 请求

问题与约束：
- grammar 编译完成时间和请求接收时间解耦，waiting queue 需要定期回收 ready 请求。
- failed grammar 请求也要进入统一的调度完成路径，不能永远停在 grammar queue。

设计选择：
- `_get_new_batch_prefill_raw` 在挑选新 prefill batch 前检查 `has_waiting_grammars()`，再把 ready grammar requests 加回 waiting queue。

Explain：
prefill 调度循环是 grammar queue 与普通 waiting queue 汇合的位置。Scheduler 不在请求接收时同步等待编译，而是在每轮构造 batch 前轻量轮询，把 ready 或 failed 的请求重新交给 `_add_request_to_queue`。

来源：python/sglang/srt/managers/scheduler.py L2741-L2749

Code：

```python
def _get_new_batch_prefill_raw(
    self, prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor]
) -> Optional[ScheduleBatch]:
    if self.grammar_manager.has_waiting_grammars():
        ready_grammar_requests = self.grammar_manager.get_ready_grammar_requests()
        for req in ready_grammar_requests:
            self._add_request_to_queue(req)
```

代码逻辑：
- prefill 选 batch 前检查 grammar queue。
- 获取 ready 或 failed grammar 请求。
- 将这些请求重新交给 waiting queue 处理。

为什么这样写：
- grammar ready 状态在 prefill 调度前刷新，能尽快把已编译请求并入下一批。
- failed 请求也要回到调度流程，让 abort 结果能被统一处理。

不变量与失败模式：
- 多 rank 下 ready 请求集合已经由 grammar manager 同步。
- grammar queue 的轮询频率受 prefill 调度循环影响。

Comment：
Scheduler 不在请求接收时等待编译，而是在 batch 构造前轻量轮询 ready 状态。

## 3. SamplingBatchInfo：把 per-request 参数变成批量张量

### 3.1 dataclass 字段把采样参数、grammar、penalty 和自定义 processor 合并

问题与约束：
- Sampler 需要 GPU 上的 batch 张量，而每个 Req 中的采样参数是 Python 对象。
- grammar mask、penalty、custom logit processor、deterministic seed 和 logit bias 都属于采样前 logits 处理。

设计选择：
- `SamplingBatchInfo` 保存 temperatures/top_p/top_k/min_p、采样 flags、grammar 列表、vocab mask、penalizer orchestrator、自定义 processor、seed 和 logit bias。

Explain：
这个 dataclass 是 forward batch 中的采样元数据。它将请求级配置批量化，并为 `ModelRunner._preprocess_logits` 和 `Sampler.forward` 提供统一输入。

来源：python/sglang/srt/sampling/sampling_batch_info.py L23-L75

Code：

```python
@dataclasses.dataclass
class SamplingBatchInfo:
    temperatures: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    min_ps: torch.Tensor

    is_all_greedy: bool
    need_top_p_sampling: bool
    need_top_k_sampling: bool
    need_min_p_sampling: bool

    vocab_size: int
    grammars: Optional[List] = None
    rids_int: Optional[torch.Tensor] = None
    bootstrap_room_ids_int: Optional[torch.Tensor] = None
    vocab_mask: Optional[torch.Tensor] = None
    apply_mask_func: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None

    penalizer_orchestrator: Optional[penaltylib.BatchedPenalizerOrchestrator] = None
    acc_additive_penalties: Optional[torch.Tensor] = None
    acc_scaling_penalties: Optional[torch.Tensor] = None

    has_custom_logit_processor: bool = False
    custom_params: Optional[List[Optional[Dict[str, Any]]]] = None
    custom_logit_processor: Optional[
        Dict[int, Tuple[CustomLogitProcessor, torch.Tensor]]
    ] = None

    sampling_seed: Optional[torch.Tensor] = None
    device: str = "cuda"
    logit_bias: Optional[torch.Tensor] = None
```

代码逻辑：
- 基础采样参数以 tensor 保存。
- flags 描述是否能走 greedy/simple sampling 快路径。
- grammar 字段保存 per-request grammar object 和当前 vocab mask。
- penalty 字段保存 orchestrator 及 overlap 模式下预累积 penalty。
- 自定义 logit processor 以 processor hash 到对象和 mask 的 dict 表示。
- deterministic sampling seed 和 logit bias 单独保存。

为什么这样写：
- Sampler 热路径需要张量化参数，不能逐请求访问 Python sampling params。
- grammar、penalty、logit bias 都是 logits 上的预处理，但来源不同，统一在 batch info 中协调。

不变量与失败模式：
- tensor 长度必须等于 batch size。
- `grammars` 若非空，其索引必须与请求顺序一致。
- `apply_mask_func` 必须匹配 `vocab_mask` 的 backend 格式。

Comment：
`SamplingBatchInfo` 是 SamplingParams 的 batch 形态，也是 logits 预处理的总线。

### 3.2 `from_schedule_batch` 批量搬运参数并构造 penalty orchestrator

问题与约束：
- 每个 decode/prefill batch 都要从请求列表收集采样参数，并尽量异步 H2D。
- 自定义 logit processor 可能重复出现，需要按 processor 字符串合并 mask。

设计选择：
- 从 `ScheduleBatch.reqs` 构造 temperatures/top_p/top_k/min_p/seed/logit_bias 等张量；按 processor 字符串合并自定义 processor；始终构造 `BatchedPenalizerOrchestrator`，由 penalizer 自己判断是否 required。

Explain：
`from_schedule_batch` 是采样元数据构造入口。它将请求级 Python 字段变成 GPU tensor，同时计算 `is_all_greedy`、`need_top_p_sampling`、`need_top_k_sampling`、`need_min_p_sampling` 等 flags，供 Sampler 选择快路径。

来源：python/sglang/srt/sampling/sampling_batch_info.py L76-L203

Code：

```python
@classmethod
def from_schedule_batch(cls, batch: ScheduleBatch, vocab_size: int):
    global_server_args = get_global_server_args()
    enable_deterministic = global_server_args.enable_deterministic_inference

    reqs = batch.reqs
    device = batch.device
    _pin = is_pin_memory_available(device)
    temperatures = (
        torch.tensor(
            [r.sampling_params.temperature for r in reqs],
            dtype=torch.float,
            pin_memory=_pin,
        )
        .to(device, non_blocking=True)
        .view(-1, 1)
    )
    top_ps = torch.tensor(
        [r.sampling_params.top_p for r in reqs],
        dtype=torch.float,
        pin_memory=_pin,
    ).to(device, non_blocking=True)
    top_ks = torch.tensor(
        [r.sampling_params.top_k for r in reqs],
        dtype=torch.int32,
        pin_memory=_pin,
    ).to(device, non_blocking=True)
    min_ps = torch.tensor(
        [r.sampling_params.min_p for r in reqs],
        dtype=torch.float,
        pin_memory=_pin,
    ).to(device, non_blocking=True)
```

代码逻辑：
- 读取全局 deterministic 开关。
- 使用 pin memory 可用性决定 host tensor 是否 pinned。
- 构造 temperature、top_p、top_k、min_p tensor 并异步搬到设备。
- deterministic 开启时构造 per-request sampling seed tensor，缺省 seed 用 42。
- 如果任一请求有 logit bias，创建 `[bs, vocab]` bias tensor。
- 自定义 processor 按字符串合并，并为每类 processor 建 mask。
- 构造 penalty orchestrator。
- 创建 `SamplingBatchInfo` 并调用 override hook。

为什么这样写：
- 采样参数每步都在 GPU 上使用，构造时完成 H2D 能减少 sampler 内部 Python 分支。
- 合并相同 custom processor 能减少重复反序列化和逐请求调用。
- penalizer 统一构造但 lazy prepare，降低 batch/filter/merge 复杂度。

不变量与失败模式：
- `logit_bias` key 必须已经在 `SamplingParams.verify` 中校验过 vocab 范围。
- deterministic seed tensor 只在 deterministic inference 下存在。
- 自定义 processor 字符串必须能被 `CustomLogitProcessor.from_str` 反序列化。

Comment：
这里是采样参数从请求对象进入 GPU 热路径的关键转换。

### 3.3 `update_regex_vocab_mask` 和 `apply_logits_bias` 定义 logits 约束顺序

问题与约束：
- Grammar mask 要按当前 grammar 状态逐步更新，不能复用上一 token 的 mask。
- Penalty、grammar mask 和 logit bias 都会修改 logits，顺序会影响结果。

设计选择：
- 采样前分配 `[bs, vocab]` vocab mask，让每个 active grammar 填入合法 token；`apply_logits_bias` 按 additive penalty、scaling penalty、non-overlap penalty、grammar mask、logit bias 顺序修改 logits。

Explain：
`update_regex_vocab_mask` 只负责构造当前 step 的合法 token mask；`apply_logits_bias` 将所有 logits-level 约束按固定顺序应用。grammar mask 通过 backend 的 `apply_vocab_mask` 把非法 token logits 置为不可选。

来源：python/sglang/srt/sampling/sampling_batch_info.py L222-L283

Code：

```python
def update_regex_vocab_mask(self):
    if not self.grammars:
        self.vocab_mask = None
        self.apply_mask_func = None
        return

    first_grammar = next(grammar for grammar in self.grammars if grammar)

    self.vocab_mask = first_grammar.allocate_vocab_mask(
        vocab_size=self.vocab_size,
        batch_size=len(self.temperatures),
        device=self.device,
    )
    self.apply_mask_func = (
        first_grammar.apply_vocab_mask
    )

    for i, grammar in enumerate(self.grammars):
        if grammar and not grammar.finished and not grammar.is_terminated():
            grammar.fill_vocab_mask(self.vocab_mask, i)

    self.vocab_mask = first_grammar.move_vocab_mask(self.vocab_mask, self.device)

def apply_logits_bias(self, logits: torch.Tensor):
    if self.acc_additive_penalties is not None:
        logits.add_(self.acc_additive_penalties)

    if self.acc_scaling_penalties is not None:
        apply_scaling_penalties(logits, self.acc_scaling_penalties)

    if self.penalizer_orchestrator and self.penalizer_orchestrator.is_required:
        self.penalizer_orchestrator.apply(logits)

    if self.vocab_mask is not None:
        self.apply_mask_func(logits=logits, vocab_mask=self.vocab_mask)

    if self.logit_bias is not None:
        logits.add_(self.logit_bias)
```

代码逻辑：
- 没有 grammar 时清空 vocab mask 和 apply function。
- 找到第一个非空 grammar，用它分配 mask 和绑定 apply 方法。
- 遍历每个 grammar，未 finished 且未 terminated 时填充当前行 mask。
- 将 mask 移到目标设备。
- logits 预处理依次施加 overlap penalty、non-overlap penalty、grammar mask 和 logit bias。

为什么这样写：
- grammar mask 每步依赖已经接受的 token，必须动态生成。
- penalty 放在 grammar mask 前，保证非法 token 最终仍被 mask 掉。
- logit bias 最后添加，保留用户显式偏置对合法 token 的影响。

不变量与失败模式：
- `self.grammars` 若非空，至少要有一个 truthy grammar，否则 `next` 会抛异常。
- 所有 grammar 的 mask 格式必须兼容第一个 grammar 的 `apply_vocab_mask`。
- mask 用完后需要释放，避免 overlap 闭包持有大 `[bs, vocab]` tensor。

Comment：
这段定义了 logits 约束的最终顺序，是 structured output 与 penalty 共同生效的位置。

### 3.4 XGrammar 以 bitmask kernel 表达合法 token 集

问题与约束：
- 大 vocab 下逐 token Python 过滤不可行，grammar mask 应该是紧凑 bitmask 并用 kernel 应用。
- 每个请求的 grammar matcher 要独立维护已接受 token 状态。

设计选择：
- `XGrammarGrammar` 用 `GrammarMatcher` 维护状态，用 `allocate_token_bitmask` 分配 mask，用 matcher 填充下一 token bitmask，并用 Triton/CUDA/NPU kernel 应用 mask。

Explain：
XGrammar 的 grammar object 既负责状态推进，也负责 mask 生成。`accept_token` 推进 FSM；`fill_vocab_mask` 写当前请求的合法 token；`copy` 重新创建 matcher，让缓存 grammar 可以被多个请求独立使用。

来源：python/sglang/srt/constrained/xgrammar_backend.py L59-L123

Code：

```python
class XGrammarGrammar(BaseGrammarObject):

    def __init__(
        self,
        matcher: GrammarMatcher,
        vocab_size: int,
        ctx: CompiledGrammar,
        override_stop_tokens: Optional[Union[List[int], int]],
        key_string: Optional[str] = None,
        grammar_stats: Optional[GrammarStats] = GrammarStats(),
    ) -> None:
        super().__init__()
        self.matcher = matcher
        self.vocab_size = vocab_size
        self.ctx = ctx
        self.override_stop_tokens = override_stop_tokens
        self.accepted_tokens = []
        self.key_string = key_string
        self.grammar_stats = grammar_stats

    def accept_token(self, token: int):
        if not self.is_terminated():
            self.current_token = token
            accepted = self.matcher.accept_token(token)
            if not accepted:
                raise ValueError(...)
            else:
                self.accepted_tokens.append(token)

    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        return allocate_token_bitmask(batch_size, vocab_size)

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        self.matcher.fill_next_token_bitmask(vocab_mask, idx)
```

代码逻辑：
- 保存 matcher、compiled grammar context、vocab size 和 stop token override。
- `accept_token` 调 matcher 接受 token，失败则抛 ValueError。
- `is_terminated` 查询 matcher 终止状态。
- 分配 bitmask，并由 matcher 填当前请求对应行。
- `move_vocab_mask` 将 bitmask 搬到设备。
- `apply_vocab_mask` 用平台 kernel 应用 bitmask。

为什么这样写：
- bitmask 对大 vocab 的存储和 kernel 应用更高效。
- matcher 状态与请求绑定，保证下一步 mask 反映已生成 token。
- accept 失败抛异常，让调度层能 abort 错误请求。

不变量与失败模式：
- `accept_token` 的 token 必须是当前 grammar 状态允许的 token。
- `copy()` 必须返回新 matcher，否则 cache hit 请求会共享状态。
- unsupported device 会在 apply mask 时抛 RuntimeError。

Comment：
XGrammar 是 sampling mask 的一个具体实现，接口仍由 `BaseGrammarObject` 统一约束。

### 3.5 XGrammar backend 绑定 tokenizer 信息与 bitmask kernel

问题与约束：
- grammar 编译必须使用与模型 tokenizer/vocab/EOS 一致的 token 信息。
- mask 应用需要平台相关 kernel，不能由每个请求对象重复选择。

设计选择：
- backend 初始化时优先读取 tokenizer 的 `init_xgrammar`，否则从 HuggingFace tokenizer 构建 `TokenizerInfo`；backend 静态方法根据 device 选择 bitmask kernel。

Explain：
`XGrammarGrammarBackend` 是 XGrammar 的编译器和平台适配层。它把 tokenizer 元信息转成 `GrammarCompiler` 可用的格式，并提供统一的 `apply_vocab_mask`，让上层只关心 grammar object 的接口。

来源：python/sglang/srt/constrained/xgrammar_backend.py L188-L245

Code：

```python
class XGrammarGrammarBackend(BaseGrammarBackend):
    def __init__(
        self,
        tokenizer,
        vocab_size: int,
        model_eos_token_ids: Optional[List[int]] = None,
        any_whitespace: bool = True,
    ):
        super().__init__()

        if hasattr(tokenizer, "init_xgrammar"):
            tokenizer_info, override_stop_tokens = tokenizer.init_xgrammar()

            if tokenizer_info is None:
                raise TokenizerNotSupportedError(
                    f"Tokenizer type {type(tokenizer).__name__} is not supported by XGrammar"
                )
        else:
            try:
                tokenizer_info = TokenizerInfo.from_huggingface(
                    tokenizer, vocab_size=vocab_size, stop_token_ids=model_eos_token_ids
                )
                override_stop_tokens = None
            except Exception as e:
                raise TokenizerNotSupportedError(
                    f"Failed to create XGrammar TokenizerInfo from tokenizer: {e}"
                )

        self.grammar_compiler = GrammarCompiler(tokenizer_info=tokenizer_info)
        self.vocab_size = vocab_size
        self.override_stop_tokens = override_stop_tokens
        self.any_whitespace = any_whitespace

    @staticmethod
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        if logits.device.type in {"cuda", "npu", "xpu", "musa"}:
            if _is_hip:
                apply_token_bitmask_inplace_cuda(logits, vocab_mask)
            else:
                apply_token_bitmask_inplace_triton(logits, vocab_mask)
        else:
            raise RuntimeError(f"Unsupported device: {logits.device.type}")
```

代码逻辑：
- 从 tokenizer 自定义 `init_xgrammar` 或 HF tokenizer 构建 `TokenizerInfo`。
- 初始化 `GrammarCompiler`。
- 保存 vocab size、stop token override 和 whitespace 策略。
- 对 CUDA/NPU/XPU/MUSA 使用 bitmask kernel 应用 mask。

为什么这样写：
- tokenizer 信息必须与模型 vocab 和 EOS 语义一致，否则 grammar mask 会允许/禁止错误 token。
- backend 级 `apply_vocab_mask` 给 SamplingBatchInfo 绑定静态方法，避免每行请求单独处理。

不变量与失败模式：
- tokenizer 不支持时抛 `TokenizerNotSupportedError`，由 backend 工厂决定降级或失败。
- ROCm 和非 ROCm 使用不同 kernel。
- device 不在支持集合中会报错。

Comment：
backend 层把 grammar 编译器、tokenizer 信息和 mask kernel 绑定在一起。

## 4. Penalty 与采样前 logits 预处理

### 4.1 Penalty orchestrator 懒准备并按普通/投机布局应用惩罚

问题与约束：
- frequency、presence、repetition、min_new_tokens 等 penalty 只有在请求需要时才应分配张量。
- speculative decoding 的 logits 行数可能是请求数乘 draft token 数，per-request penalty 要扩展到 draft token layout。

设计选择：
- 构造 orchestrator 时创建所有 penalizer，但每个 penalizer 自己 `prepare_if_required`；`apply` 普通路径逐个 in-place 修改 logits，repeat 路径先累积 additive/scaling penalty 再 repeat_interleave。

Explain：
`BatchedPenalizerOrchestrator` 统一管理多个 penalty。它用 weakref 引用 `ScheduleBatch`，避免循环引用；filter/merge/release 负责在 batch 变化时维护各 penalizer 的内部状态。

来源：python/sglang/srt/sampling/penaltylib/orchestrator.py L13-L104

Code：

```python
class BatchedPenalizerOrchestrator:
    def __init__(
        self,
        vocab_size: int,
        batch: ScheduleBatch,
        penalizers: Set[Type[_BatchedPenalizer]],
    ):
        self.vocab_size = vocab_size
        self._batch_ref = weakref.ref(batch)
        self.device = batch.device
        self.penalizers = {Penalizer: Penalizer(self) for Penalizer in penalizers}

        is_required = False
        for penalizer in self.penalizers.values():
            pen_is_required = penalizer.prepare_if_required()
            is_required |= pen_is_required
        self.is_required = is_required

    def cumulate_output_tokens(self, output_ids: torch.Tensor):
        for penalizer in self.penalizers.values():
            penalizer.cumulate_output_tokens(output_ids=output_ids)

    def apply(self, logits: torch.Tensor, repeat: Optional[int] = None):
        if repeat is None:
            for penalizer in self.penalizers.values():
                penalizer.apply(logits)
        else:
            bs = logits.shape[0] // repeat
            additive = torch.zeros(
                (bs, logits.shape[1]), dtype=torch.float32, device=logits.device
            )
            self.accumulate_additive_penalties(additive)
            logits.add_(torch.repeat_interleave(additive, repeat, dim=0))
            accumulated = self.accumulate_scaling_penalties()
            if accumulated is not None:
                expanded = torch.repeat_interleave(accumulated, repeat, dim=0)
                apply_scaling_penalties(logits, expanded)
```

代码逻辑：
- 为每类 penalizer 创建实例。
- 每个 penalizer 判断是否需要准备张量。
- `is_required` 汇总所有 penalizer 是否生效。
- 输出 token 产生后调用 `cumulate_output_tokens` 更新 penalizer 状态。
- 普通 logits layout 下逐个 apply。
- speculative repeat layout 下先构造 per-request penalty，再扩展到每个 draft token 行。

为什么这样写：
- penalty 状态随生成 token 更新，必须和 batch 生命周期绑定。
- lazy prepare 避免没有 penalty 的请求承担额外张量开销。
- repeat 分支避免把 per-request penalty 错应用到 draft token 展平后的行布局。

不变量与失败模式：
- repeat 必须整除 logits 第一维。
- weakref batch 释放后，penalizer 访问 batch 会失败。
- merge 要在 batch.reqs 更新前调用，源码注释明确说明这一点。

Comment：
Penalty 是 logits 约束的一类，但它需要跨 step 维护状态，和 grammar 的 FSM 状态类似。

### 4.2 `ScheduleBatch` 在 decode 前把最新输出 token 喂给 penalty

问题与约束：
- penalty 依赖历史输出 token；overlap 模式下 `batch.input_ids` 可能只是 placeholder，不能直接作为真实最新 token。
- 每步 token 累积不应阻塞 forward stream。

设计选择：
- 从每个 req 的 `output_ids[-1]` 或首 decode 前的 `origin_input_ids[-1]` 取最新 token，构造 pinned host tensor 后异步搬到设备，再调用 orchestrator 累积。

Explain：
`cumulate_penalty_output_tokens` 是 penalty 状态更新的调用点。它在进入下一步 decode 前将上一轮输出 token 交给各 penalizer，使 frequency/presence/repetition/min_new_tokens 在下一次 logits 处理时生效。

来源：python/sglang/srt/managers/schedule_batch.py L2597-L2616

Code：

```python
def cumulate_penalty_output_tokens(self):
    last_tokens = [
        req.output_ids[-1] if len(req.output_ids) else req.origin_input_ids[-1]
        for req in self.reqs
    ]
    latest_output_ids = torch.tensor(
        last_tokens,
        dtype=torch.int64,
        pin_memory=is_pin_memory_available(self.device),
    ).to(self.device, non_blocking=True)
    self.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
        latest_output_ids
    )
```

代码逻辑：
- 对每个请求取最新输出 token；若没有输出，取 prompt 最后 token。
- 构造 int64 tensor，并尽量使用 pin memory。
- 异步 H2D 到 batch device。
- 调用 sampling info 中的 penalty orchestrator 累积输出 token。

为什么这样写：
- penalty 应基于真实请求输出，而不是 overlap placeholder。
- 异步拷贝减少每 step 与 forward 的同步。

不变量与失败模式：
- 每个 req 必须至少有 origin input id。
- `sampling_info.penalizer_orchestrator` 必须已由 `SamplingBatchInfo.from_schedule_batch` 构造。
- batch 过滤/合并后 penalizer 状态要同步 filter/merge。

Comment：
这一步把上一轮采样结果反馈到下一轮 logits penalty 中。

### 4.3 ModelRunner 在 Sampler 前统一施加 grammar、penalty 和 logit bias

问题与约束：
- Sampler 应只负责从处理后的 logits 中抽 token，不应知道 grammar/penalty/logit bias 的具体状态管理。
- overlap 模式下 `vocab_mask` 大张量可能被延迟采样闭包持有，造成显存泄漏。

设计选择：
- `ModelRunner._preprocess_logits` 调用 `SamplingBatchInfo.update_regex_vocab_mask()` 和 `apply_logits_bias()`，随后立即清空 `vocab_mask`；`sample` 再调用 sampler。

Explain：
ModelRunner 是 forward output 和 sampler 的连接点。它先把所有 logits-level 约束应用到 `next_token_logits`，再调用 `self.sampler`，并根据 forward mode 传入 decode positions 或 prefill 最后 token 位置。

来源：python/sglang/srt/model_executor/model_runner.py L3143-L3191

Code：

```python
def _preprocess_logits(
    self, logits_output: LogitsProcessorOutput, sampling_info: SamplingBatchInfo
):
    sampling_info.update_regex_vocab_mask()
    sampling_info.apply_logits_bias(logits_output.next_token_logits)

    sampling_info.vocab_mask = None

def sample(
    self,
    logits_output: LogitsProcessorOutput,
    forward_batch: ForwardBatch,
) -> torch.Tensor:
    self._preprocess_logits(logits_output, forward_batch.sampling_info)

    next_token_ids = self.sampler(
        logits_output,
        forward_batch.sampling_info,
        forward_batch.return_logprob,
        forward_batch.top_logprobs_nums,
        forward_batch.token_ids_logprobs,
        (
            forward_batch.positions
            if forward_batch.forward_mode.is_decode()
            else forward_batch.seq_lens - 1
        ),
    )
    self.maybe_update_ngram_token_table(next_token_ids, forward_batch)
    return next_token_ids
```

代码逻辑：
- 更新当前 step 的 grammar vocab mask。
- 应用 penalty、mask 和 logit bias。
- 清空 vocab mask 引用。
- 调用 sampler，传入 logits output、sampling info、logprob 需求和 positions。
- 更新 ngram token table。

为什么这样写：
- logits 约束统一在 sampler 前完成，sampler 可保持后端选择和概率采样职责。
- `vocab_mask=None` 及时释放 `[bs, vocab]` GPU 张量，避免 overlap 延迟闭包延长生命周期。
- positions 同时服务 deterministic sampling 和 logprob 位置选择。

不变量与失败模式：
- `logits_output.next_token_logits` 必须可原地修改。
- grammar mask 应用后，TP rank 可能更容易出现 token id 不一致，Sampler 会按 grammar 情况同步 token ids。
- prefill 模式只用每个序列最后位置采样。

Comment：
ModelRunner 这层把“约束 logits”和“从 logits 采样”明确分开。

## 5. Sampler：从 logits/probs 到 next token

### 5.1 `Sampler.forward` 在 greedy、Ascend、RL deterministic 和标准 softmax 路径间分支

问题与约束：
- Greedy 请求可以直接 argmax；普通采样要温度缩放和 softmax；RL on-policy 需要 log-softmax 与 trainer 对齐；Ascend backend 有自己的 fused sampling 路径。
- 返回 logprob 时，需要在不同路径下保留合适的 logprob 张量。

设计选择：
- 先应用 custom logit processor 和 NaN sanitization；全 greedy 走 argmax；否则根据 backend/RL/deterministic/simple flags 选择 logprobs 或 softmax/probs 路径，最后可选附加 logprobs 并同步 TP token ids。

Explain：
Sampler 的主函数不再关心 grammar/penalty；它看到的是已经处理过的 logits。`simple_sampling_case` 表示没有 top-p/top-k/min-p 限制，可以直接从概率或 logprob 中采样。

来源：python/sglang/srt/layers/sampler.py L84-L212

Code：

```python
def _preprocess_logits(
    self, logits: torch.Tensor, sampling_info: SamplingBatchInfo
) -> torch.Tensor:
    """Apply custom logit processors and sanitize non-finite logits."""
    if sampling_info.has_custom_logit_processor:
        apply_custom_logit_processor(logits, sampling_info)
    sanitize_nan_logits(logits, "sampler: next_token_logits")
    return logits

def forward(
    self,
    logits_output: LogitsProcessorOutput,
    sampling_info: SamplingBatchInfo,
    return_logprob: bool,
    top_logprobs_nums: List[int],
    token_ids_logprobs: List[List[int]],
    positions: torch.Tensor,
):
    logits = logits_output.next_token_logits

    logits = self._preprocess_logits(logits, sampling_info)

    if sampling_info.is_all_greedy:
        if _use_aiter and not _disable_aiter_greedy_sample:
            batch_next_token_ids = torch.empty(
                logits.shape[0], device=logits.device, dtype=torch.int32
            )
            _aiter_greedy_sample(batch_next_token_ids, logits)
        else:
            batch_next_token_ids = torch.argmax(logits, -1)
        if return_logprob:
            original_logprobs = logprobs = torch.nn.functional.log_softmax(
                logits, dim=-1
            )
    else:
        simple_sampling_case = (
            not sampling_info.need_top_p_sampling
            and not sampling_info.need_top_k_sampling
            and not sampling_info.need_min_p_sampling
        )
```

代码逻辑：
- 取 `next_token_logits`。
- 应用 custom logit processor，并清理 NaN/非有限 logits。
- 全 greedy 时用 AITER 或 torch argmax。
- 非 greedy 时计算 simple sampling flag。
- RL on-policy 或 deterministic path 可走 log-softmax 采样。
- 标准路径对 logits 除以 temperature，原地 softmax 成 probs，再调用 `_sample_from_probs`。
- return logprob 时把 logprob 附加回 logits output。
- grammar 或 env 要求时同步 TP token ids。

为什么这样写：
- greedy 是最便宜路径，避免 softmax 和 sampling kernel。
- 标准路径原地把 logits 变成 probs，减少额外显存。
- RL on-policy 要和训练端 logprob 计算一致，因此走 log-softmax。

不变量与失败模式：
- all greedy 的判定来自 batch info，必须和 top_k=1 归一化一致。
- flashinfer backend 不支持 sampling seed。
- AITER greedy 可通过 env 回退到 torch argmax，避免极端 logits 行产生越界 token。

Comment：
Sampler 主函数是后端选择和概率计算的中心，但它假设 logits 约束已经由 ModelRunner 处理完成。

### 5.2 `_sample_from_probs` 封装 simple、FlashInfer 和 PyTorch fallback

问题与约束：
- top-k/top-p/min-p 组合需要不同 kernel；没有这些约束时可直接 multinomial。
- 不同后端对 deterministic seed 支持不同。

设计选择：
- simple case 走 `sampling_from_probs_torch`；复杂 case 按 `sampling_backend` 选择 flashinfer 或 pytorch；min-p 下先 top-k/top-p renorm，再 min-p sampling。

Explain：
`_sample_from_probs` 接收温度缩放并 softmax 后的概率分布。FlashInfer 路径用 fused kernel 做 top-k/top-p；PyTorch 路径作为较慢 fallback，同时支持 sampling seed。

来源：python/sglang/srt/layers/sampler.py L214-L283

Code：

```python
def _sample_from_probs(
    self,
    probs: torch.Tensor,
    sampling_info: SamplingBatchInfo,
    positions: torch.Tensor,
    simple_sampling_case: bool,
) -> torch.Tensor:
    if simple_sampling_case:
        batch_next_token_ids = sampling_from_probs_torch(
            probs,
            sampling_seed=sampling_info.sampling_seed,
            positions=positions,
        )
    else:
        backend = get_global_server_args().sampling_backend
        if backend == "flashinfer":
            assert (
                sampling_info.sampling_seed is None
            ), "Sampling seed is not supported for flashinfer backend"
            if sampling_info.need_min_p_sampling:
                probs = top_k_renorm_prob(probs, sampling_info.top_ks)
                probs = top_p_renorm_prob(probs, sampling_info.top_ps)
                batch_next_token_ids = min_p_sampling_from_probs(
                    probs, sampling_info.min_ps
                )
            else:
                batch_next_token_ids = top_k_top_p_sampling_from_probs(
                    probs.contiguous(),
                    sampling_info.top_ks,
                    sampling_info.top_ps,
                    filter_apply_order="joint",
                )
        elif backend == "pytorch":
            batch_next_token_ids = top_k_top_p_min_p_sampling_from_probs_torch(
                probs,
                sampling_info.top_ks,
                sampling_info.top_ps,
                sampling_info.min_ps,
                sampling_info.need_min_p_sampling,
                sampling_info.sampling_seed,
                positions,
            )
        else:
            raise ValueError(f"Invalid sampling backend: {backend}")
    return batch_next_token_ids
```

代码逻辑：
- 无 top-k/top-p/min-p 时直接从 probs 采样。
- flashinfer backend 不允许 sampling_seed。
- min-p 情况先做 top-k 和 top-p renorm，再 min-p sampling。
- 非 min-p 复杂情况调用 fused top-k/top-p sampling。
- pytorch backend 调用 fallback 实现。

为什么这样写：
- simple path 避免不必要的 top-k/top-p kernel。
- FlashInfer 对大 batch/vocab 的 top-k/top-p 更高效。
- PyTorch fallback 提供可移植性和 deterministic seed 支持。

不变量与失败模式：
- probs 必须是已归一化概率分布。
- flashinfer 复杂路径要求 seed 为 None。
- unknown sampling backend 会抛 `ValueError`。

Comment：
这里是 top-k/top-p/min-p 真正生效的采样后端分派点。

### 5.3 logprob 附加和 TP token id 同步是 Sampler 的后处理

问题与约束：
- 请求可能要求 top logprobs 或指定 token ids 的 logprobs，需要写回 `LogitsProcessorOutput`。
- TP rank 理论上应采到同一 token；structured output/xgrammar 场景下不同 rank 不同步会导致 hang 或状态分叉。

设计选择：
- `_attach_logprobs_to_output` 从 logprobs 中填充 top/token_ids/next-token logprob；`_sync_token_ids_across_tp` 在 env 开关或 grammar 存在时做 all_reduce MIN。

Explain：
Sampler 采样后仍要做两个后处理：把 logprob 请求的结果挂回 logits output，以及在必要时同步 token id。同步使用 MIN reduce，确保所有 TP rank 后续接受同一个 token。

来源：python/sglang/srt/layers/sampler.py L347-L395

Code：

```python
def _attach_logprobs_to_output(
    self,
    logits_output: LogitsProcessorOutput,
    logprobs: torch.Tensor,
    top_logprobs_nums: List[int],
    token_ids_logprobs: List[List[int]],
    sampling_info: SamplingBatchInfo,
    batch_next_token_ids: torch.Tensor,
):
    logprobs.clamp_(min=torch.finfo(logprobs.dtype).min)

    if any(x > 0 for x in top_logprobs_nums):
        (
            logits_output.next_token_top_logprobs_val,
            logits_output.next_token_top_logprobs_idx,
        ) = get_top_logprobs(logprobs, top_logprobs_nums, no_copy_to_cpu=True)

    if any(x is not None for x in token_ids_logprobs):
        (
            logits_output.next_token_token_ids_logprobs_val,
            logits_output.next_token_token_ids_logprobs_idx,
        ) = get_token_ids_logprobs(
            logprobs, token_ids_logprobs, no_copy_to_cpu=True
        )

    logits_output.next_token_logprobs = logprobs[
        torch.arange(len(batch_next_token_ids), device=sampling_info.device),
        batch_next_token_ids,
    ]

def _sync_token_ids_across_tp(
    self, batch_next_token_ids: torch.Tensor, sampling_info: SamplingBatchInfo
):
    if SYNC_TOKEN_IDS_ACROSS_TP or sampling_info.grammars:
        torch.distributed.all_reduce(
            batch_next_token_ids,
            op=dist.ReduceOp.MIN,
            group=self.tp_sync_group,
        )
```

代码逻辑：
- 将 logprobs clamp 到 dtype 最小值，避免 -inf。
- 按请求需要提取 top logprobs。
- 按请求指定 token ids 提取 logprobs。
- 记录被采样 token 的 logprob。
- grammar 存在或 env 强制时，在 TP group 内同步 token ids。

为什么这样写：
- logprob 计算和采样共享同一张 logprobs，避免重复 softmax/log-softmax。
- grammar 状态必须在 TP rank 间一致，否则下一步 mask 会分叉。

不变量与失败模式：
- `batch_next_token_ids` 长度必须等于 batch size。
- `sampling_info.device` 必须能索引 logprobs。
- all_reduce MIN 假设不同 rank 的合法 token id 差异应被强行统一，不能恢复导致差异的根因。

Comment：
采样结果不只是 next token ids，还包括 logprob 元数据和跨 rank 一致性维护。

## 6. 采样后的 grammar 与 penalty 状态推进

### 6.1 普通 prefill/decode 结果处理会推进 grammar FSM

问题与约束：
- 采样前 grammar mask 只限制下一 token；采样后必须把实际 token 喂给 grammar，否则下一步 mask 仍停留在旧状态。
- `accept_token` 可能因为 grammar 配置错误或 token 不合法而抛异常。

设计选择：
- 结果处理器在 prefill 或 decode 结果落地时调用 `req.grammar.accept_token`；失败时 abort 请求，并同步 grammar finished 状态。

Explain：
`_apply_prefill_grammar` 是非 speculative 场景下的 grammar 状态推进点之一。它接收 next token id，调用 grammar object 更新 FSM；如果 grammar 报错，调度器中止该请求。

来源：python/sglang/srt/managers/scheduler_components/batch_result_processor.py L485-L497

Code：

```python
def _apply_prefill_grammar(self, *, req: Req, next_token_id: int) -> None:
    try:
        req.grammar.accept_token(next_token_id)
    except ValueError as e:
        logger.error(
            f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
        )
        self.abort_request(AbortReq(rid=req.rid))
    req.grammar.finished = req.finished()
```

代码逻辑：
- 对当前请求的 grammar 接受刚生成 token。
- `ValueError` 说明 token 不被 grammar 接受，记录日志并 abort。
- 将 grammar finished 状态同步为请求 finished 状态。

为什么这样写：
- grammar mask 的下一步依赖 FSM 状态，必须在结果处理阶段推进。
- 接受失败不能继续生成，否则后续 mask 和 KV 状态都会失真。

不变量与失败模式：
- `req.grammar` 必须非 None。
- token id 必须来自本请求实际提交的输出。
- abort 后该请求还需通过调度器统一清理。

Comment：
grammar 是“采样前限制 + 采样后推进”的闭环，缺一端都会错误。

### 6.2 Spec decoding 会接受一段 token，并在 grammar 终止处截断后缀

问题与约束：
- Speculative decoding 一次可能接受多个 draft token 加 bonus token。
- Grammar 可能在这段 token 中间终止，终止后的 overdrafted suffix 不能提交到 KV，也不能发给用户。

设计选择：
- `_accept_grammar_tokens` 逐 token 调 `accept_token`，遇到 `is_terminated()` 即停止并返回保留前缀；调用方只提交 retained tokens。

Explain：
spec 路径不能简单把整段 accept tokens 都提交。结果处理器先用 grammar FSM 过滤 accepted run，保留直到 grammar 终止的前缀，然后再更新 `kv_committed_len` 和输出 token。

来源：python/sglang/srt/managers/scheduler_components/batch_result_processor.py L550-L615

Code：

```python
for i, req in enumerate(batch.reqs):
    accept_tokens = next_token_ids[i * stride : i * stride + accept_lens[i]]

    if req.is_retracted or req.finished():
        pass
    else:
        if req.grammar is not None:
            accept_tokens = self._accept_grammar_tokens(req, accept_tokens)

        num_accept_tokens = len(accept_tokens)
        req.kv_committed_len += num_accept_tokens
        req.spec_verify_ct += 1

    predict_tokens.append(accept_tokens)

def _accept_grammar_tokens(
    self, req: Req, tokens: Union[int, List[int]]
) -> List[int]:
    if isinstance(tokens, int):
        tokens = [tokens]
    retained = []
    try:
        for token_id in tokens:
            req.grammar.accept_token(token_id)
            retained.append(token_id)
            if req.grammar.is_terminated():
                break
    except ValueError as e:
        self.abort_request(AbortReq(rid=req.rid))
    return retained
```

代码逻辑：
- 按 stride 从展平 next token ids 中取每个请求的 accepted run。
- 已 retracted 或 finished 的请求不处理。
- 有 grammar 时逐 token 接受并截断到 grammar 终止点。
- 只按保留 token 数更新 KV committed length。
- 返回每个请求最终提交的 predict tokens。

为什么这样写：
- spec verify 产生的 token run 可能超过 grammar 允许的终止点。
- 不提交后缀可以避免 KV cache 和用户输出包含 grammar 外 token。

不变量与失败模式：
- `stride` 必须等于结果记录的 speculative draft token 数。
- `accept_lens[i]` 必须不超过对应请求的 stride。
- grammar accept 抛错会 abort，但 retained 前缀的处理依赖调用方后续状态清理。

Comment：
structured output 和 speculative decoding 的交集在这里处理：只提交 grammar 接受的前缀。
