---
type: batch-doc
module: 21-Speculative
batch: "21"
doc_type: walkthrough
title: "投机解码 · 源码走读"
tags:
 - sglang/batch/21
 - sglang/module/speculative
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-05
---
# 投机解码 · 源码走读

## 走读顺序

1. `spec_info.py` - 算法枚举、字符串解析与 per-algorithm 参数钩子
2. `spec_registry.py` - 自定义 speculative algorithm 注册与 worker 工厂
3. `base_spec_worker.py` - draft KV 页复制和 draft extend 准备
4. `eagle_worker_v2.py` - EAGLE V2 worker、draft worker 与 adaptive spec
5. `ngram_worker.py` - 无 draft 模型的 n-gram speculative worker
6. `reject_sampling.py` - Triton verify kernel

---

## 1. 算法选择与参数变更

### 1.1 `SpeculativeAlgorithm.from_string` 统一解析内置枚举和插件算法

问题与约束：
- CLI 传入的是字符串或 None，内部需要得到统一的 speculative algorithm 对象。
- 算法既可能是内置 enum，也可能是通过 registry 注册的插件。
- 未知名称必须尽早失败，不能默默退回 NONE。

设计选择：
- None 映射为 `SpeculativeAlgorithm.NONE`。
- 字符串统一转大写后先查 enum。
- enum 不存在时查 `_get_registered_spec`。
- 两者都不存在时抛 `ValueError`。

Explain：
`from_string` 是 speculative decoding 的入口解析层。它把命令行字符串变成后续 scheduler、worker factory 和参数钩子都能消费的 algorithm 对象。

来源：python/sglang/srt/speculative/spec_info.py L43-L57

Code：

```python
@classmethod
def from_string(
    cls, name: Optional[str]
) -> Union[SpeculativeAlgorithm, CustomSpecAlgo]:
    if name is None:
        return cls.NONE
    upper = name.upper()
    try:
        return cls[upper]
    except KeyError:
        pass
    spec = _get_registered_spec(upper)
    if spec is not None:
        return spec
    raise ValueError(f"Unknown speculative algorithm name: {name}")
```

代码逻辑：
- 判断输入是否为 None。
- 将算法名转成大写。
- 尝试按 enum 名称索引内置算法。
- 失败后查询插件 registry。
- 找不到插件时抛出未知算法错误。

为什么这样写：
- 内置算法走 enum，便于保留 `is_eagle/is_ngram` 等语义方法。
- 插件算法走 registry，避免修改 enum 也能扩展。
- 解析阶段 fail fast，能避免后续 worker 初始化到一半才发现算法不可用。

不变量与失败模式：
- registry 中插件名称应与大写算法名匹配。
- 调用方要接受返回值可能是 enum 或 `CustomSpecAlgo`。
- 未注册或拼写错误的名称会抛 `ValueError`。

Comment：
投机算法选择的第一层兼容点是“内置 enum + 外部 registry”。

### 1.2 `has_draft_kv` 区分会写 draft KV 的算法

问题与约束：
- EAGLE 等算法会在 draft 阶段写 KV chain。
- NGRAM 的候选树只存在于 verify mask 和 CPU/host 结构，不写 draft KV。
- Scheduler 计算每步 KV 分配长度时，是否需要按 topk 预留 draft KV 会影响内存。

设计选择：
- `has_draft_kv` 默认用 `not self.is_ngram()` 表达。
- NGRAM 返回 False。
- 其他内置算法保留 draft KV 语义。

Explain：
这个方法把算法级差异转成内存分配信号。NGRAM 不需要为 draft branch 写 KV，因此 per-decode KV sizing 可以跳过 topk 相关的 page rounding。

来源：python/sglang/srt/speculative/spec_info.py L121-L125

Code：

```python
def has_draft_kv(self) -> bool:
    """Whether the draft phase writes KV chains. NGRAM does not (its tree
    lives only in the verify mask), so per-decode KV sizing needs no
    per-topk page rounding; see get_alloc_len_per_decode."""
    return not self.is_ngram()
```

代码逻辑：
- 判断当前算法是否是 NGRAM。
- NGRAM 返回 False。
- 非 NGRAM 返回 True。

为什么这样写：
- 用 algorithm 方法封装内存语义，Scheduler 不需要硬编码每种算法细节。
- NGRAM 是明确特例，表达为 `not is_ngram()` 简洁且保守。
- 未来如果新增无 draft KV 算法，需要同步扩展这里或插件默认语义。

不变量与失败模式：
- `is_ngram()` 必须准确识别 NGRAM。
- 如果算法实际不写 draft KV 却返回 True，会浪费 KV 页。
- 如果算法会写 draft KV 却返回 False，verify 或 expand 可能读不到对应 KV。

Comment：
投机解码的算法选择会直接影响 KV cache 分配策略。

### 1.3 `handle_server_args` 把算法专用默认值集中到钩子

问题与约束：
- 不同 speculative 算法需要不同的 server args 衍生字段。
- EAGLE family、DFLASH、FROZEN_KV_MTP、NGRAM 的默认路径、步数或 topk 规则不同。
- 参数解析后需要在启动 worker 前完成 in-place 变更。

设计选择：
- 在 algorithm 对象上提供 `handle_server_args`。
- 延迟导入各算法的 handler，避免基础模块 import 过重。
- 按 `is_dflash/is_frozen_kv_mtp/is_eagle/is_standalone/is_ngram` 分派到对应 handler。

Explain：
这个钩子把“算法选择”连接到“启动参数规范化”。worker 初始化看到的 `ServerArgs` 已经被算法专用 handler 修正，而不是每个 worker 再自行补默认值。

来源：python/sglang/srt/speculative/spec_info.py L162-L181

Code：

```python
def handle_server_args(self, server_args: ServerArgs) -> None:
    """Hook for per-algorithm server args mutation.

    In-place updated.
    """
    from sglang.srt.arg_groups.speculative_hook import (
        _handle_dflash,
        _handle_eagle_family,
        _handle_frozen_kv_mtp,
        _handle_ngram,
    )

    if self.is_dflash():
        _handle_dflash(server_args)
    elif self.is_frozen_kv_mtp():
        _handle_frozen_kv_mtp(server_args)
    elif self.is_eagle() or self.is_standalone():
        _handle_eagle_family(server_args)
    elif self.is_ngram():
        _handle_ngram(server_args)
```

代码逻辑：
- 定义 per-algorithm server args mutation 钩子。
- 延迟导入 speculative 参数 handler。
- DFLASH 调 `_handle_dflash`。
- FROZEN_KV_MTP 调 `_handle_frozen_kv_mtp`。
- EAGLE 和 STANDALONE 调 `_handle_eagle_family`。
- NGRAM 调 `_handle_ngram`。

为什么这样写：
- 参数变更集中在 algorithm 层，避免散落在多个 worker 构造函数。
- 延迟导入能降低基础枚举模块的依赖复杂度。
- in-place 更新符合后续启动流程对单个 `ServerArgs` 对象的使用方式。

不变量与失败模式：
- 该钩子必须在创建 worker 前调用。
- handler 必须只变更对应算法需要的字段。
- 新增算法时如果没有 handler，可能缺少必要默认值。

Comment：
算法枚举不只负责命名，也负责把命名映射到启动参数修正。

### 1.4 `CustomSpecAlgo.create_worker` 对 overlap support 做启动期校验

问题与约束：
- 插件算法的 worker 由外部 factory 提供，核心代码无法静态保证其支持 overlap scheduling。
- overlap 开启但插件不支持时，继续运行会破坏 scheduler 的 V2 schema 假设。
- overlap 关闭时仍要允许旧插件同步运行，但需要提醒迁移。

设计选择：
- 如果 overlap 未禁用且 `supports_overlap=False`，直接抛 `ValueError`。
- 如果 overlap 已禁用但插件不支持 overlap，记录 warning。
- 最后调用插件 factory 创建 worker。

Explain：
插件注册允许扩展 speculative 算法，但 worker 调度能力必须和当前 scheduler 模式一致。这里把 overlap 兼容性检查放在 worker 创建前，避免运行中才暴露协议不匹配。

来源：python/sglang/srt/speculative/spec_registry.py L92-L111

Code：

```python
def handle_server_args(self, server_args: ServerArgs) -> None:
    pass

def create_worker(self, server_args: ServerArgs) -> Type:
    if not server_args.disable_overlap_schedule and not self.supports_overlap:
        raise ValueError(
            f"Speculative algorithm {self.name} does not support overlap scheduling."
        )
    if not self.supports_overlap:
        logger.warning(
            "Speculative algorithm %s is registered with "
            "supports_overlap=False, which is deprecated: the spec V1 "
            "worker path has been removed, and the algorithm now runs on "
            "the V2 scheduler schema with overlap disabled (synchronous). "
            "Migrate the plugin worker to support overlap scheduling.",
            self.name,
        )
    return self.factory(server_args)
```

代码逻辑：
- 插件默认不修改 server args。
- 创建 worker 前检查 overlap 是否启用。
- overlap 启用且插件不支持时抛错。
- 插件不支持 overlap 但 overlap 已关闭时输出 warning。
- 调用注册时提供的 factory。

为什么这样写：
- overlap 兼容性是调度协议问题，应在 worker 创建前判断。
- 同步模式保留兼容性，但 warning 明确 V1 worker path 已移除。
- factory 最后调用，避免不兼容插件产生半初始化 worker。

不变量与失败模式：
- 插件作者必须准确声明 `supports_overlap`。
- scheduler overlap 状态必须来自最终 `ServerArgs`。
- 错误声明支持 overlap 的插件仍可能在运行期破坏 V2 调度语义。

Comment：
自定义 speculative algorithm 的扩展点以 worker factory 为边界，但调度能力由核心代码校验。

## 2. EAGLE draft 路径

### 2.1 `duplicate_prefix_tail_to_draft_branches` 复制 prefix 尾页给 topk 分支

问题与约束：
- EAGLE topk 大于 1 时，每个 draft branch 都有自己的 draft page。
- prefix 最后一页可能是 partial page，branch b>=1 的首页 hole 需要读到真实 prefix tail。
- 如果不复制，draft-decode expand 会按 block id 读到错误内容。

设计选择：
- `topk <= 1` 时直接返回。
- 为 branch 1 到 topk-1 构造分支维度。
- 按 `prefix_base + page_off` 构造源位置，后续只复制 `[0, last_page)` 的 prefix tail。

Explain：
这个函数处理的是 paged KV 的边界条件：多分支 draft 共享 prefix，但每个分支有独立 draft page。partial tail 必须被复制到分支页的 hole 中，expand 才能按分支 block id 正确读 prefix。

来源：python/sglang/srt/speculative/base_spec_worker.py L22-L45

Code：

```python
def duplicate_prefix_tail_to_draft_branches(
    token_to_kv_pool,
    rows: torch.Tensor,
    prefix_base: torch.Tensor,
    last_page: torch.Tensor,
    num_new_pages: torch.Tensor,
    topk: int,
    page_size: int,
) -> None:
    """Copy the prefix partial-tail page into each branch's first-page holes (page>1 + topk>1).

    The draft-decode expand pass reads each branch's own draft page by block id
    (cache_loc // page_size), so branch b>=1's hole slots [0, last_page) must hold the
    real prefix tail (branch 0's first page already is it). Mirrors V1 #7725.
    """
    if topk <= 1:
        return
    bs = rows.shape[0]
    page_off = torch.arange(page_size, device=rows.device, dtype=torch.int64)
    branches = torch.arange(1, topk, device=rows.device, dtype=torch.int64).view(
        1, topk - 1, 1
    )
    src_pos = (prefix_base.view(bs, 1, 1) + page_off.view(1, 1, page_size)).expand(
```

代码逻辑：
- 接收 KV pool、请求行、prefix base、last page、new page 数、topk 和 page size。
- topk 不超过 1 时跳过。
- 读取 batch size。
- 构造页内 offset。
- 构造 branch 1 到 topk-1 的分支索引。
- 根据 prefix base 和页内 offset 构造源位置。

为什么这样写：
- 只在多分支情况下执行，避免单分支路径额外开销。
- 复制 prefix tail 而不是重新组织所有 KV，成本局限在 partial page。
- 使用 tensor broadcast 构造位置，适合批量处理多个请求。

不变量与失败模式：
- `prefix_base` 和 `last_page` 必须描述真实 prefix 最后一页。
- branch 0 已经持有正确 prefix tail，函数只补 branch 1+。
- 如果 `last_page` 或 page 对齐错误，expand 会从错误 KV 位置读分支上下文。

Comment：
这是 speculative topk 与 paged KV cache 交汇处的边界修复。

### 2.2 `prepare_for_draft_extend` 把 EAGLE draft token 写回 batch

问题与约束：
- EAGLE draft extend 要把预测 token 作为本轮 draft 输入。
- 该准备逻辑可能运行在 plan stream 下，不能在这里做 dtype cast 引入跨 stream 竞态。
- `seq_lens_cpu` 可能不存在，GPU-only 路径不能调用 `.tolist()` 或 `.cpu()`。

设计选择：
- 计算 `extend_num_tokens = bs * num_draft_tokens`。
- 通过 `batch.seq_lens_cpu is None` 判断 GPU-only 路径。
- 设置 `batch.spec_info = draft_extend_input`。
- 直接把 caller 传入的 `predict` 赋给 `batch.input_ids`，不在函数内 cast。
- 立即用 `maybe_detect_oob` 做 token 范围探测。

Explain：
这个函数把 scheduler batch 临时改造成 EAGLE draft extend 的 forward batch 输入。它最重要的约束是 stream safety：caller 必须在进入 plan stream 前准备好 dtype，这里只安装引用和 spec_info。

来源：python/sglang/srt/speculative/base_spec_worker.py L92-L130

Code：

```python
def prepare_for_draft_extend(
    self,
    draft_extend_input: EagleDraftExtendInput,
    batch: ScheduleBatch,
    predict: torch.Tensor,
    num_draft_tokens: int,
    draft_model_runner: Any,
    cuda_graph_runner: Any,
):
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardBatch,
        ForwardMode,
    )
    from sglang.srt.utils.async_probe import maybe_detect_oob
    from sglang.srt.utils.common import is_npu

    bs = len(batch.seq_lens)
    extend_num_tokens = bs * num_draft_tokens
    gpu_only = batch.seq_lens_cpu is None

    batch.spec_info = draft_extend_input
    batch.input_ids = predict
    maybe_detect_oob(
        batch.input_ids,
        0,
        batch.model_config.vocab_size,
        "v2 prepare_for_draft_extend input_ids",
    )
    if gpu_only:
        batch.prefix_lens = batch.seq_lens.to(torch.int32)
```

代码逻辑：
- 延迟导入 forward batch 相关类型和探测工具。
- 计算 batch size 和 draft extend token 总数。
- 判断是否走 GPU-only 长度路径。
- 将 EAGLE draft extend input 写入 `batch.spec_info`。
- 将预测 token 写入 `batch.input_ids`。
- 检查 token id 是否越界。
- GPU-only 时直接从 GPU `seq_lens` 派生 `prefix_lens`。

为什么这样写：
- `batch.spec_info` 是 attention/backend 识别 draft extend 语义的入口。
- 不在 plan stream 中 cast dtype，避免额外 stream dependency。
- GPU-only 分支避免 CPU 同步，和 overlap/graph 路径兼容。

不变量与失败模式：
- caller 传入的 `predict` dtype 和 device 必须已经满足后续 forward 要求。
- `num_draft_tokens` 必须和 `draft_extend_input` 的树结构一致。
- token 越界会被 `maybe_detect_oob` 捕获。
- 如果错误调用 CPU path，会引入不必要同步或在缺少 `seq_lens_cpu` 时失败。

Comment：
EAGLE draft extend 的 batch mutation 是精确控制 stream 与 metadata 的 hot path。

### 2.3 `EAGLEWorkerV2` 组合 target worker、draft worker 和 adaptive controller

问题与约束：
- EAGLE V2 需要同时访问 target model worker 和独立 draft worker。
- draft model 的 context length 要和 target model 对齐。
- adaptive speculative 是可选能力，不应强制所有启动路径创建 controller。
- spec V2 forward 会触碰 target/draft/draft-extend 多个 attention backend。

设计选择：
- `EAGLEWorkerV2` 继承 `BaseSpecWorker`。
- 构造时保存 topk、steps、draft token 数、tp rank、gpu id、target worker 和 page size。
- 用 `SpeculativeAlgorithm.from_string` 解析算法。
- 把 `server_args.context_length` 改为 target model context length。
- 创建内部 `EagleDraftWorker`。
- 仅在 `speculative_adaptive` 开启时创建 `AdaptiveController`。
- `spec_v2_attn_backends` 返回所有 spec V2 forward 会触碰的 backend。

Explain：
`EAGLEWorkerV2` 是 target 和 draft 两套执行资源的协调器。它自己不只是 draft model wrapper，还负责上下文长度对齐、adaptive 控制和 attention backend 依赖声明。

来源：python/sglang/srt/speculative/eagle_worker_v2.py L948-L1021

Code：

```python
class EAGLEWorkerV2(BaseSpecWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.tp_rank = tp_rank
        self.gpu_id = gpu_id
        self.device = server_args.device
        self._target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        server_args.context_length = target_worker.model_runner.model_config.context_len

        self._draft_worker = EagleDraftWorker(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            attn_cp_rank,
            moe_dp_rank,
            nccl_port,
            target_worker,
        )

        self.adaptive_controller: Optional[AdaptiveController] = None
        if server_args.speculative_adaptive:
            self.adaptive_controller = AdaptiveController(
                self,
                config_path=server_args.speculative_adaptive_config,
            )

    @property
    def spec_v2_attn_backends(self) -> tuple:
        return (
            self._target_worker.model_runner.attn_backend,
            self._draft_worker.draft_attn_backend,
            self._draft_worker.draft_extend_attn_backend
            or self._draft_worker.draft_runner.attn_backend,
        )
```

代码逻辑：
- 解析 speculative 参数并保存运行所需字段。
- 保存 target worker 和 page size。
- 解析当前 speculative algorithm。
- 将 draft 侧 context length 对齐 target model。
- 构造 `EagleDraftWorker`。
- 可选创建 adaptive controller。
- 提供 target、draft、draft-extend attention backend tuple。

为什么这样写：
- target 和 draft 模型共享请求上下文约束，context length 不一致会破坏 verify。
- draft worker 独立封装 draft 模型加载和 forward，V2 worker 负责调度组合。
- `spec_v2_attn_backends` 让上层能一次性判断是否需要 CPU seq lens 等 shared metadata。

不变量与失败模式：
- target worker 必须先初始化出可用的 `model_runner.model_config.context_len`。
- draft worker 参数要和 target 并行拓扑一致。
- adaptive config 路径错误会影响 controller 初始化。
- 如果漏掉某个 attention backend，seq lens 或 graph metadata 判断可能不完整。

Comment：
EAGLE V2 的 worker 边界是“target verify + draft generation + draft extend”的组合调度。

## 3. NGRAM 与 verify

### 3.1 `NGRAMWorker` 复用 target worker，不加载 draft 模型

问题与约束：
- NGRAM speculative 不需要 draft model 权重。
- 它仍要使用 target worker 的 memory pool 和 model runner。
- 外部 n-gram corpus 可选加载，且需要 tokenizer 转 token chunks。
- 离开 batch 的请求需要清理 corpus match 状态。

设计选择：
- `alloc_memory_pool` 延迟到 target memory pool 存在后调用 `target_worker.get_memory_pool()`。
- 构造函数保存 target worker、model runner、page size、draft token 数和 n-gram trie 参数。
- 创建 `NgramCorpus` 保存 CPU 侧匹配结构。
- 配置 external corpus path 时，读取 chunks、加载 corpus 并 commit。
- `draft_worker` property 返回 None。

Explain：
NGRAMWorker 是“没有 draft 模型的 speculative worker”。它从历史 token/corpus 中找候选树，但 verify 仍回到 target model，因此它必须复用 target worker 的资源边界。

来源：python/sglang/srt/speculative/ngram_worker.py L37-L116

Code：

```python
class NGRAMWorker(BaseSpecWorker):
    def alloc_memory_pool(self, **kwargs):
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            self._target_worker.get_memory_pool()
        )
        self.max_batch_size = self.model_runner.max_running_requests
        self._init_preallocated_tensors()

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.enable_overlap = not server_args.disable_overlap_schedule
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size
        self.draft_token_num: int = server_args.speculative_num_draft_tokens
        self.max_trie_depth: int = server_args.speculative_ngram_max_trie_depth
        self.ngram_corpus = NgramCorpus(
            min_bfs_breadth=server_args.speculative_ngram_min_bfs_breadth,
            max_bfs_breadth=server_args.speculative_ngram_max_bfs_breadth,
            match_type=server_args.speculative_ngram_match_type,
            capacity=server_args.speculative_ngram_capacity,
            max_trie_depth=server_args.speculative_ngram_max_trie_depth,
            draft_token_num=server_args.speculative_num_draft_tokens,
            external_sam_budget=server_args.speculative_ngram_external_sam_budget,
            external_corpus_max_tokens=server_args.speculative_ngram_external_corpus_max_tokens,
        )
        if server_args.speculative_ngram_external_corpus_path is not None:
            chunks = list(
                iter_external_corpus_chunks(
                    corpus_path,
                    target_worker.tokenizer,
                    server_args.speculative_ngram_external_corpus_max_tokens,
                )
            )
            loaded = self.add_external_corpus(corpus_path, chunks)
            self.commit_corpus_load(corpus_path, loaded)

    @property
    def draft_worker(self) -> Optional[EagleDraftWorkerBase]:
        return None
```

代码逻辑：
- memory pool 初始化时从 target worker 获取 req/token pool。
- 设置最大 batch size 并初始化预分配 tensor。
- 构造时保存 server args、overlap 状态、target worker 和 model runner。
- 读取 page size、draft token 数、trie depth 等参数。
- 创建 `NgramCorpus`。
- 如果配置外部 corpus 路径，按 tokenizer 读取 token chunks。
- 加载并提交外部 corpus。
- `draft_worker` 返回 None。

为什么这样写：
- NGRAM 的候选来自 corpus，不需要额外 draft 模型或 draft weights。
- target worker 已拥有真实 verify 需要的 model runner 和 memory pool，复用能避免重复分配。
- external corpus 在启动时加载，decode 时只做匹配和状态更新。

不变量与失败模式：
- `alloc_memory_pool` 必须在 target pool 就绪后调用。
- external corpus 文件必须能被 tokenizer 正确切分成 token chunks。
- `draft_worker=None` 的调用路径必须使用 `draft_worker or target_worker` 类 fallback。
- corpus match 状态需要在请求离批时清理，否则会影响后续请求匹配。

Comment：
NGRAM 是 speculative 框架里的“无模型 draft”特例。

### 3.2 reject sampling kernel 按 `coin * q < p` 逐步接受 draft token

问题与约束：
- Speculative sampling 要比较 target 分布 `p` 与 draft 分布 `q`，决定 draft token 是否被接受。
- 每个 batch 行要独立验证一条候选序列。
- 接受 token 后要更新当前概率行、预测输出和 accept index。
- 一旦拒绝，后续步骤停止，交给 final sampling 处理。

设计选择：
- Triton kernel 每个 `pid` 处理一个 batch 行。
- 从 candidate、target probs、draft probs 和 uniform samples 中加载当前 step 数据。
- 条件 `coin * q < p` 成立时接受 token。
- 拒绝时把 `continue_verifying` 置 0。
- 循环结束后写 `AcceptTokenNum`。

Explain：
这段是标准 speculative sampling 的 verify loop。它把每个 draft token 的接受/拒绝压到 Triton kernel 内完成，并把最终接受数量写回给 scheduler。

来源：python/sglang/srt/speculative/reject_sampling.py L48-L100

Code：

```python
 # Verification Loop
step = 1
continue_verifying = 1

while (step < NUM_SLOTS) and (continue_verifying == 1):
    draft_token = tl.load(cand_ptr_base + step * stride_cand_s)

    offset_prob = (
        (pid * stride_tp_b)
        + (cur_prob_row * stride_tp_s)
        + (draft_token * stride_tp_v)
    )
    offset_draft = (
        (pid * stride_dp_b)
        + (cur_prob_row * stride_dp_s)
        + (draft_token * stride_dp_v)
    )

    p = tl.load(TargetProbs + offset_prob)
    q = tl.load(DraftProbs + offset_draft)

    coin = tl.load(uni_ptr_base + (step - 1) * stride_uni_s)

    if coin * q < p:
        num_accept += 1
        cur_prob_row = step
        tl.store(Predicts + last_accepted_global_idx, draft_token)

        curr_global_idx = tl.load(idx_ptr_base + step * stride_idx_s)
        tl.store(
            AcceptIndex + pid * stride_idx_b + num_accept * stride_idx_s,
            curr_global_idx,
        )
        last_accepted_global_idx = curr_global_idx

        step += 1
    else:
        continue_verifying = 0

tl.store(AcceptTokenNum + pid, num_accept)
```

代码逻辑：
- 初始化 step 和 continue flag。
- 在 step 未超过槽位且仍继续验证时循环。
- 读取当前 draft token。
- 计算 target probs 和 draft probs 的偏移。
- 加载 `p`、`q` 和 uniform coin。
- 满足 `coin * q < p` 时接受 token。
- 写预测 token 和 accept index。
- 更新 last accepted global index 并推进 step。
- 拒绝时停止循环。
- 写本 batch 行接受 token 数。

为什么这样写：
- verify loop 放进 Triton，减少 Python 逐 token 控制开销。
- `coin * q < p` 避免显式计算 `p/q`，数值和性能都更直接。
- `AcceptIndex` 保留被接受 token 的全局位置，便于后续 KV/cache 状态更新。

不变量与失败模式：
- `TargetProbs` 和 `DraftProbs` 的 stride 必须和 offset 公式一致。
- `UniformSamples` 长度至少覆盖 draft steps。
- draft token 必须是合法 vocab index。
- 若 `q` 为 0 或概率未归一，接受概率语义会失真。

Comment：
Scheduler 看到的是 `AcceptTokenNum`，但接受决策已经在 kernel 中逐步完成。
