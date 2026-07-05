---
type: batch-doc
module: 11-ModelRunner
batch: "11"
doc_type: walkthrough
title: "ModelRunner · 源码走读"
tags:
  - sglang/batch/11
  - sglang/module/model-runner
  - sglang/doc/walkthrough
aliases:
  - "02-源码走读"
updated: 2026-07-05
---

# ModelRunner · 源码走读

> 走读主线：`TpModelWorker` 是 Scheduler 侧调用的模型执行入口；`ModelRunner` 持有模型、并行 rank、KV cache、attention backend、CUDA graph runner 和 sampler；`ForwardBatch` 把 `ScheduleBatch` 转成执行态张量；`_forward_raw` 根据 forward mode、graph 可用性和 prefill/decode 分支选择 graph replay 或 eager forward，最后由 Worker 在 PP 末 rank 采样并返回。

---

## 1. Worker 到 Runner 的边界

### 1.1 `TpModelWorker.__init__` 汇总并行 rank、模型配置和 tokenizer/processor

问题与约束：
- 一个 TP worker 需要同时知道 TP/PP/DP/MoE EP/attention CP rank，以及是否为 draft worker、多层 EAGLE worker。
- 多模态模型要加载 processor，纯文本模型只需要 tokenizer；跳过 tokenizer 初始化时两者都不能创建。

设计选择：
- Worker 构造函数先保存所有 rank 和共享 memory pool 入口，再初始化 `ModelConfig` 和 `ModelRunner`；随后按多模态与否选择 processor/tokenizer，最后同步随机种子和通信 group。

Explain：
`TpModelWorker` 是调度侧对模型执行的封装。它不直接执行模型层，而是把并行拓扑、内存池句柄、draft worker 标志和 tokenizer/processor 放在 worker 层，再把模型 forward 的重活交给 `ModelRunner`。

来源：python/sglang/srt/managers/tp_worker.py L228-L315

Code：

```python
def __init__(
    self,
    server_args: ServerArgs,
    gpu_id: int,
    tp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    attn_cp_rank: int,
    moe_dp_rank: int,
    dp_rank: Optional[int],
    nccl_port: int,
    is_draft_worker: bool = False,
    req_to_token_pool: Optional[ReqToTokenPool] = None,
    token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
    memory_pool_config: Optional[MemoryPoolConfig] = None,
    is_multi_layer_eagle: bool = False,
):
    self.server_args = server_args
    self.tp_size = server_args.tp_size
    self.ep_size = server_args.ep_size
    self.pp_size = server_args.pp_size
    self.tp_rank = tp_rank
    self.moe_ep_rank = moe_ep_rank
    self.pp_rank = pp_rank
    self.dp_rank = dp_rank
    self.gpu_id = gpu_id
    self.nccl_port = nccl_port
    self.is_draft_worker = is_draft_worker
    self.is_multi_layer_eagle = is_multi_layer_eagle
    self.req_to_token_pool = req_to_token_pool
    self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
    self.attn_cp_rank = attn_cp_rank
    self.moe_dp_rank = moe_dp_rank
    self.memory_pool_config = memory_pool_config

    self.model_runner_list: List[ModelRunner] = []

    self._init_model_config()
    self._init_model_runner()
```

代码逻辑：
- 保存 server args 和各类并行 rank。
- 保存可由外部传入的 request/token KV pool。
- 初始化 `model_runner_list`，供 multi-layer EAGLE 附加 runner 使用。
- 调用 `_init_model_config()` 和 `_init_model_runner()`。
- 多模态模型加载 processor 并从 processor 取 tokenizer，普通模型直接加载 tokenizer。
- 初始化 PP/world group，并广播随机种子。

为什么这样写：
- Worker 层需要和 Scheduler、Tokenizer、通信组打交道；Runner 层专注模型执行。
- tokenizer/processor 属于请求准备和输出侧契约，不应混进底层 forward 分支。
- draft worker 通过同一构造路径复用 target runner 的 memory pool 配置，降低投机解码特殊分支。

不变量与失败模式：
- `server_args` 中的并行大小必须与传入 rank 一致。
- `skip_tokenizer_init=True` 后，依赖 tokenizer/processor 的路径不能运行。
- 多模态模型的 processor 加载失败会影响 worker 初始化。

Comment：
这段明确了 `TpModelWorker` 是“执行入口和资源装配层”，不是模型 forward 的实现层。

### 1.2 Worker 分阶段初始化 memory pool、attention backend 和 CUDA graph

问题与约束：
- KV cache pool 大小依赖显存估算，attention backend 依赖 KV pool layout，CUDA graph 又必须在 backend 和 buffer 就绪后捕获。
- 多 runner 场景要让附加 runner 共享同一套 request-to-token 与 token-to-KV pool。

设计选择：
- Worker 暴露 `alloc_memory_pool`、`init_attention_backends`、`init_cuda_graphs` 三个阶段；每个阶段先处理主 runner，再遍历附加 runner。

Explain：
初始化顺序是 ModelRunner 可靠运行的前置条件：先分配 KV cache，再初始化 attention backend，最后捕获 decode/prefill graph。Worker 在每个阶段同步附加 runner 的 pool 句柄，保证多 runner 不各自分配互相不兼容的状态。

来源：python/sglang/srt/managers/tp_worker.py L316-L355

Code：

```python
def alloc_memory_pool(
    self,
    memory_pool_config: Optional[MemoryPoolConfig] = None,
    req_to_token_pool: Optional[ReqToTokenPool] = None,
    token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
):
    if req_to_token_pool is not None:
        self.req_to_token_pool = req_to_token_pool
        self.model_runner.req_to_token_pool = req_to_token_pool
    if token_to_kv_pool_allocator is not None:
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.model_runner.token_to_kv_pool_allocator = token_to_kv_pool_allocator
    self.model_runner.alloc_memory_pool(memory_pool_config)
    for mr in self.model_runner_list[1:]:
        mr.req_to_token_pool = self.req_to_token_pool
        mr.token_to_kv_pool_allocator = self.token_to_kv_pool_allocator
        mr.alloc_memory_pool(memory_pool_config)

    assert self.model_runner.max_running_requests > 0, "max_running_request is zero"
    max_req_len = min(
        self.model_config.context_len - 1,
        self.model_runner.max_token_pool_size - 1,
    )
    assert max_req_len > 0, "Memory pool size is too small"
```

代码逻辑：
- 可选替换 worker 和主 runner 的 pool 句柄。
- 调用主 runner 的 `alloc_memory_pool`。
- 附加 runner 复用同一 pool 句柄后再分配自身资源。
- 校验最大并发请求数和最大请求长度。
- attention backend 和 CUDA graph 阶段同样对所有 runner 调用对应方法。

为什么这样写：
- 分阶段让显存估算、KV pool、backend metadata 和 graph capture 的依赖顺序可控。
- 附加 runner 共享 pool，避免多层 EAGLE 或 draft 路径产生互相独立的 KV 索引空间。

不变量与失败模式：
- `max_running_requests` 必须大于 0。
- `max_req_len` 必须大于 0，否则说明 KV pool 太小或 context length 不可用。
- graph capture 必须晚于 attention backend 初始化。

Comment：
这三个阶段是启动时最重要的生命周期顺序：pool 先于 backend，backend 先于 graph。

## 2. ModelRunner 初始化与模型加载

### 2.1 `ModelRunner.__init__` 保存并行拓扑、模型特性和投机配置

问题与约束：
- 一个 Runner 同时要支持 TP、PP、DP attention、MoE EP/DP、attention CP、draft worker、multi-modal、MLA、hybrid SWA 和 speculative decoding。
- 后续内存池、attention backend 和 graph capture 都要读取这些配置。

设计选择：
- 构造阶段只保存拓扑与模型特征，初始化 draft 相关层数信息，延后到后续阶段再加载权重、分配 pool 和捕获 graph。

Explain：
`ModelRunner.__init__` 将运行时配置固化到对象上：device、rank、并行大小、model config、spec algorithm、page size、KV pool 句柄、MLA/SWA/multimodal 标志等。EAGLE/standalone spec target 会预读 draft model config，用于 KV cache sizing。

来源：python/sglang/srt/model_executor/model_runner.py L343-L460

Code：

```python
class ModelRunner(ModelRunnerKVCacheMixin):
    """ModelRunner runs the forward passes of the models."""

    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        moe_ep_rank: int,
        moe_ep_size: int,
        pp_rank: int,
        pp_size: int,
        nccl_port: int,
        server_args: ServerArgs,
        dp_rank: Optional[int] = None,
        attn_cp_rank: Optional[int] = None,
        moe_dp_rank: Optional[int] = None,
        is_draft_worker: bool = False,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
        memory_pool_config: Optional[MemoryPoolConfig] = None,
        draft_model_idx: Optional[int] = None,
    ):
        self.mem_fraction_static = mem_fraction_static
        self.memory_pool_config = memory_pool_config
        self.device = server_args.device
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dcp_size = server_args.dcp_size
        self.dcp_rank = self.tp_rank % self.dcp_size
        self.moe_ep_rank = moe_ep_rank
        self.moe_ep_size = moe_ep_size
        self.dp_rank = dp_rank
        self.dp_size = server_args.dp_size if server_args.enable_dp_attention else 1
```

代码逻辑：
- 保存静态显存比例和已解析的 memory pool config。
- 保存 device、GPU id、TP/PP/DP/MoE/CP rank 与 size。
- 保存 model config、server args、draft worker 标志。
- 解析 speculative algorithm。
- 保存多模态、MLA、hybrid SWA、attention chunk 等模型特征。
- 对 EAGLE/standalone target，读取 draft config 推断 draft layer 数。

为什么这样写：
- Runner 的后续阶段都依赖这些配置；构造阶段集中保存能避免各阶段重复解析 server args。
- 先不加载模型和分配 pool，使启动流程可以按显存估算和分布式初始化顺序推进。

不变量与失败模式：
- `dcp_rank = tp_rank % dcp_size` 要求 `dcp_size` 合理且非零。
- DP attention 未启用时 `dp_size` 固定为 1。
- speculative draft model path 缺失时，EAGLE target 无法从 draft config 推断辅助层数。

Comment：
Runner 构造阶段像一张执行配置快照，真正昂贵的权重和显存动作被留到后续生命周期。

### 2.2 `load_model` 先准备加载配置，再进入 ModelLoader

问题与约束：
- 权重加载会和 CPU 线程、CUDA 架构、load format、ModelOpt、remote instance loader、draft model index 等配置交织。
- 老 GPU 可能不支持 bfloat16，权重 dtype 需要在加载前修正。

设计选择：
- 加载前记录可用显存、限制 torch CPU 线程、按 CUDA capability 修正 dtype，再构造 `ModelOptConfig` 和 `LoadConfig` 交给 ModelLoader。

Explain：
`load_model` 前半段把所有“如何加载权重”的参数收敛到 `self.load_config`。这包括本地/远程加载格式、download dir、TP rank、remote instance 通信参数、ModelOpt 配置、RL quant profile 和 draft model index。

来源：python/sglang/srt/model_executor/model_runner.py L1388-L1437

Code：

```python
def load_model(self):
    tic_total = time.perf_counter()
    before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
    logger.info(
        f"Load weight begin. avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
    )

    if self.device != "cpu":
        torch.set_num_threads(1)
    if self.device == "cuda":
        if torch.cuda.get_device_capability()[0] < 8:
            logger.info(
                "Compute capability below sm80. Use float16 due to lack of bfloat16 support."
            )
            self.server_args.dtype = "float16"
            self.model_config.dtype = torch.float16
            if torch.cuda.get_device_capability()[1] < 5:
                raise RuntimeError("SGLang only supports sm75 and above.")

    set_cuda_arch()

    modelopt_config = ModelOptConfig(
        quant=self.server_args.modelopt_quant,
        checkpoint_restore_path=self.server_args.modelopt_checkpoint_restore_path,
        checkpoint_save_path=self.server_args.modelopt_checkpoint_save_path,
        export_path=self.server_args.modelopt_export_path,
        quantize_and_serve=self.server_args.quantize_and_serve,
    )

    self.load_config = LoadConfig(
        load_format=self.server_args.load_format,
        download_dir=self.server_args.download_dir,
        model_loader_extra_config=self.server_args.model_loader_extra_config,
        tp_rank=self.tp_rank,
        modelopt_config=modelopt_config,
        rl_quant_profile=self.server_args.rl_quant_profile,
        draft_model_idx=self.draft_model_idx,
    )
```

代码逻辑：
- 记录加载前可用显存。
- 非 CPU 设备把 torch CPU 线程数设为 1。
- CUDA sm80 以下改用 float16，sm75 以下直接报错。
- 设置 CUDA 架构。
- 构造 ModelOpt 配置。
- 构造 `LoadConfig`，把 load format、下载目录、远程加载和量化配置集中起来。

为什么这样写：
- 权重加载前修正 dtype 和线程设置，可以避免加载过程中的不兼容和线程争用。
- `LoadConfig` 作为 ModelLoader 的统一输入，使不同加载器只关心自己的字段。

不变量与失败模式：
- CUDA capability 低于 sm75 不被支持。
- remote instance loader 相关字段必须和 `load_format` 匹配。
- dtype 被降级后，model config 和 server args 都要同步更新。

Comment：
这段是 ModelRunner 和 ModelLoader 的接口准备层，真正加载由 loader 完成。

### 2.3 `load_model` 后半段绑定模型对象并完成权重后处理

问题与约束：
- 权重加载可能占用大量显存，需要和 memory saver、CPU backup、远程权重传输、KV cache scale、profiling hook 等功能协同。
- 分布式 TP rank 必须一起完成加载，否则后续 collective 会挂住。

设计选择：
- 在 memory saver region 内创建 loader 并加载模型；加载后处理 offloader、KV scale、sliding window、量化日志、debug hook、RoPE cache，并用 TP group barrier 检查各 rank 完成情况。

Explain：
`load_model` 后半段把 `self.model` 和 `self.loader` 真正绑定到 Runner 上。它还记录权重显存占用、应用调试/量化/卸载相关 hook，并在结束时进行分布式 barrier，确保所有 TP rank 都进入一致状态。

来源：python/sglang/srt/model_executor/model_runner.py L1461-L1617

Code：

```python
monkey_patch_vllm_parallel_state()

enable_cpu_backup = self.server_args.enable_weights_cpu_backup or (
    self.is_draft_worker and self.server_args.enable_draft_weights_cpu_backup
)
with self.memory_saver_adapter.region(
    GPU_MEMORY_TYPE_WEIGHTS,
    enable_cpu_backup=enable_cpu_backup,
):
    self.loader = get_model_loader(
        load_config=self.load_config,
        model_config=self.model_config,
    )
    self.model = self.loader.load_model(
        model_config=self.model_config,
        device_config=DeviceConfig(self.device, self.gpu_id),
    )

monkey_patch_vllm_parallel_state(reverse=True)

if not self.is_draft_worker:
    get_offloader().post_init()

after_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
self.weight_load_mem_usage = before_avail_memory - after_avail_memory

reserve_rope_cache_for_long_sequences(
    self.model,
    self.server_args,
    self.model_config,
    logger,
)

dist.monitored_barrier(
    group=get_tp_group().cpu_group,
    timeout=datetime.timedelta(seconds=UNBALANCED_MODEL_LOADING_TIMEOUT_S),
    wait_all_ranks=True,
)
```

代码逻辑：
- 临时 patch vLLM parallel state，加载完成后恢复。
- 根据 target/draft worker 配置决定是否启用权重 CPU backup。
- 在 memory saver region 内创建 loader 并调用 `load_model`。
- 非 draft worker 初始化 offloader。
- 记录权重加载后的显存占用。
- 为长序列预扩 RoPE cache。
- 用 TP group monitored barrier 检查所有 rank 完成加载。

为什么这样写：
- 权重加载是显存峰值最高的阶段，必须和 memory saver/offloader 协调。
- RoPE cache 预扩展要发生在 CUDA graph capture 前，否则 capture 后再扩容会破坏静态假设。
- barrier 能把“某个 rank 加载慢或 OOM”在启动阶段暴露出来。

不变量与失败模式：
- loader 必须能根据 `LoadConfig` 和 `ModelConfig` 返回可 forward 的模型对象。
- FP8 KV cache scale 文件存在时，模型必须支持 `load_kv_cache_scales`。
- monitored barrier 超时会转成包含当前 TP rank 的加载失败提示。

Comment：
权重加载完成后，Runner 才具备模型对象；但 KV pool、backend 和 graph 仍在后续阶段初始化。

### 2.4 `alloc_memory_pool` 和 `init_attention_backends` 保持 pool 与 backend 的顺序

问题与约束：
- attention backend 需要 KV pool、request pool、canary patch、HiSparse coordinator 等对象先存在。
- CUDA graph capture 期间的 warmup forward 要看到最终 patched 的 pool 方法。

设计选择：
- `alloc_memory_pool` 只分配 KV cache 和相关运行结构，并重置 backend/graph runner 字段；`init_attention_backends` 再按 device 初始化 cublas、attention backend 和平台特定 lazy init。

Explain：
`ModelRunner.alloc_memory_pool` 是模型加载后的显存阶段，调用 `init_memory_pool` 后安装 canary、初始化 ngram/HiSparse/expert capturer，并把 attention backend 与 graph runner 清空。`init_attention_backends` 随后根据 device 建立 attention backend。

来源：python/sglang/srt/model_executor/model_runner.py L820-L899

Code：

```python
def alloc_memory_pool(self, memory_pool_config: Optional[MemoryPoolConfig] = None):
    """Allocate KV cache memory pools only (no backends or cuda graphs)."""
    if memory_pool_config is not None:
        self.memory_pool_config = memory_pool_config

    self.init_memory_pool(self.pre_model_load_memory)

    self.canary_manager = install_canary(
        server_args=self.server_args,
        model_runner=self,
        token_oracle_manager=self._token_oracle_manager,
    )

    self.maybe_init_ngram_embedding()

    self.init_routed_experts_capturer()
    self.init_indexer_capturer()

    self.attn_backend = None
    self.decode_attn_backend = None
    self.decode_attn_backend_group = []
    self.decode_cuda_graph_runner = None
    self.graph_mem_usage = 0
    self.prefill_cuda_graph_runner = None

def init_attention_backends(self):
    """Initialize attention backends only (no cuda graph capture)."""
    self.init_aux_hidden_state_capture()

    if self.device == "cuda" or self.device == "musa":
        self.init_cublas()
        self.init_attention_backend()
    elif self.device == "cpu":
        self.init_attention_backend()
    elif self.device == "npu":
        self.init_attention_backend()
```

代码逻辑：
- 可选更新 memory pool config。
- 调用 KV cache mixin 的 `init_memory_pool`。
- 安装 canary，并初始化 ngram、HiSparse、expert/indexer capturer。
- 清空 attention backend 和 graph runner 字段。
- attention backend 阶段先初始化辅助 hidden state capture。
- 按 device 初始化 cublas 和 attention backend。

为什么这样写：
- backend 依赖 pool，graph 依赖 backend；清空字段能避免旧 runner 状态被误用。
- canary 必须在 graph capture 前安装，否则被 capture 的 warmup forward 看不到校验逻辑。

不变量与失败模式：
- `pre_model_load_memory` 和 memory pool config 必须能支撑 `init_memory_pool` sizing。
- HiSparse 需要额外 coordinator 和 host/device buffer 配置。
- device 分支漏掉平台初始化会导致 backend metadata 不完整。

Comment：
这里是启动生命周期的中段：模型已加载，接下来把可执行 forward 所需的 KV 和 attention 环境补齐。

## 3. CUDA Graph 配置与捕获

### 3.1 `init_cuda_graphs` 先建立 eager runner，再捕获 prefill/decode graph

问题与约束：
- 即使 CUDA graph 关闭或 batch 不命中图，系统也需要 eager fallback。
- graph runner 和 eager runner 要共享静态输入 buffer，否则 captured graph 和 eager fallback 可能使用不同物理分配。

设计选择：
- `init_cuda_graphs` 先创建 `EagerRunner`，再初始化 prefill graph，最后按配置捕获 decode graph；关闭 decode graph 时让 `decode_cuda_graph_runner` 指向 eager runner。

Explain：
Eager runner 在 graph capture 之前创建，用于预热 kernel、分配最大静态 buffer，并成为 graph runner 共享 buffer 的规范来源。graph capture 后才注册 forward hooks，避免 hooks 中的 tensor op 被捕获进 graph。

来源：python/sglang/srt/model_executor/model_runner.py L900-L945

Code：

```python
def init_cuda_graphs(self, capture_decode_cuda_graph: bool = True):
    """Capture cuda graphs. Requires init_attention_backends() to have run."""

    self.eager_runner = EagerRunner(self)

    self.init_prefill_cuda_graph()

    self.decode_cuda_graph_runner = None
    self.graph_mem_usage = 0

    if capture_decode_cuda_graph:
        if self.device in ("cuda", "musa", "cpu", "npu"):
            self.init_decode_cuda_graph()
        elif (
            current_platform.is_out_of_tree()
            and current_platform.support_cuda_graph()
        ):
            self.init_decode_cuda_graph()
    else:
        self.decode_cuda_graph_runner = self.eager_runner

    if self.server_args.forward_hooks:
        register_forward_hooks(self.model, self.server_args.forward_hooks)

    self.prealloc_symmetric_memory_pool()
```

代码逻辑：
- 创建 eager runner。
- 初始化 prefill CUDA graph runner。
- 清空 decode graph 状态。
- 根据 `capture_decode_cuda_graph` 和平台能力决定是否捕获 decode graph。
- 不捕获时把 decode graph runner 设为 eager runner。
- graph capture 后注册 forward hooks。
- 预分配 symmetric memory pool。

为什么这样写：
- eager runner 永远存在，提供 fallback 和共享 buffer。
- hook 注册必须晚于 capture，否则 Python hook 相关 tensor op 会进入 graph。
- prefill capture 先于 decode capture，二者可共享 eager 阶段建立的 buffer 池。

不变量与失败模式：
- 该方法要求 `init_attention_backends()` 已执行。
- draft runner 可能传入 `capture_decode_cuda_graph=False`，自行管理 graph 捕获。
- 平台不支持 graph 时必须能落到 eager。

Comment：
`init_cuda_graphs` 的重点是先建立稳定 fallback，再逐步增加 graph 加速。

### 3.2 `CudaGraphConfig` 用 phase/backend schema 约束 graph 配置

问题与约束：
- Decode 和 prefill 的 graph 形状特征不同，不能用完全相同的 backend 约束。
- ServerArgs 需要导入配置 schema，但不应在配置模块导入 torch 或后端类。

设计选择：
- 用纯 stdlib dataclass 定义 `PhaseConfig` 和 `CudaGraphConfig`；decode 默认 full，prefill 默认 tc_piecewise；每个 phase 有独立允许 backend 和 key。

Explain：
配置模块把 graph 配置拆成 phase 与 backend 两层：phase 是 decode/prefill，backend 是 full/breakable/tc_piecewise/disabled。prefill 明确不允许 full backend，因为 prefill 形状更可变。

来源：python/sglang/srt/model_executor/cuda_graph_config.py L30-L125

Code：

```python
class Phase:
    """The two phases of model forward."""

    DECODE = "decode"
    PREFILL = "prefill"
    ALL = (DECODE, PREFILL)

class Backend:
    """CUDA graph capture backends a phase can use."""

    FULL = "full"
    BREAKABLE = "breakable"
    TC_PIECEWISE = "tc_piecewise"
    DISABLED = "disabled"
    ALL = (FULL, BREAKABLE, TC_PIECEWISE, DISABLED)

ALLOWED_BACKENDS_PER_PHASE = {
    Phase.DECODE: (
        Backend.FULL,
        Backend.BREAKABLE,
        Backend.TC_PIECEWISE,
        Backend.DISABLED,
    ),
    Phase.PREFILL: (Backend.BREAKABLE, Backend.TC_PIECEWISE, Backend.DISABLED),
}

@dataclass
class CudaGraphConfig:
    """Top-level CUDA graph config: one PhaseConfig per phase."""

    decode: PhaseConfig = field(
        default_factory=lambda: PhaseConfig(backend=Backend.FULL)
    )
    prefill: PhaseConfig = field(
        default_factory=lambda: PhaseConfig(backend=Backend.TC_PIECEWISE)
    )
```

代码逻辑：
- 定义 forward phase 常量。
- 定义 graph backend 常量。
- 为 decode/prefill 分别列出允许 backend。
- `PhaseConfig` 保存 backend、max_bs、bs、tc_compiler。
- `CudaGraphConfig` 保存每个 phase 的配置，并提供 dict 转换。

为什么这样写：
- phase 级配置能表达 decode 和 prefill 的不同 capture 约束。
- 纯 stdlib 依赖保证 ServerArgs 解析配置时不会提前加载运行后端。
- diff-only `to_dict` 避免把默认值误认为用户显式设置。

不变量与失败模式：
- prefill full backend 会被上游 parser/validator 拒绝。
- unknown phase/key 在 CLI parser 中会报错，在 `from_dict` 中会被忽略。
- `bs` 为空时，prefill graph 会在 Runner 初始化阶段被禁用。

Comment：
这份配置是 graph runner 行为的输入，不是执行逻辑本身。

### 3.3 decode graph bucket 由配置、pool 上限和对齐约束共同决定

问题与约束：
- CUDA graph 只能捕获离散 batch size；实际 batch 需要 pad 到已捕获 bucket。
- DP attention、attention TP/CP、two-batch overlap 和 speculative target verify 都会改变 batch size 或 token 数对齐要求。

设计选择：
- `get_batch_sizes_to_capture` 从 `cuda_graph_config.decode.bs` 出发，按最大请求数、gathered buffer、attention CP 和 token-per-bs 对齐过滤，并生成可选 torch compile bucket。

Explain：
decode capture 并不是简单使用用户配置的 `bs` 列表。函数会先计算 `mul_base`，再把最大请求数对齐到 `mul_base`，必要时补上最大值，最后过滤掉不满足 `bs * num_tokens_per_bs % mul_base == 0` 的 bucket。

来源：python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py L58-L100

Code：

```python
def get_batch_sizes_to_capture(
    model_runner: ModelRunner, num_tokens_per_bs: int = 1
) -> Tuple[List[int], List[int]]:
    server_args = model_runner.server_args
    capture_bs = list(server_args.cuda_graph_config.decode.bs)
    num_max_requests = model_runner.req_to_token_pool.size

    mul_base = 1
    if server_args.enable_two_batch_overlap:
        mul_base *= 2
        num_tokens_per_bs = 1

    if require_gathered_buffer(server_args):
        mul_base *= get_parallel().attn_tp_size

    if mul_base % get_parallel().attn_cp_size != 0:
        mul_base *= get_parallel().attn_cp_size

    num_max_requests = (num_max_requests + mul_base - 1) // mul_base * mul_base
    if max(capture_bs) > num_max_requests:
        capture_bs += [num_max_requests]

    capture_bs = [bs for bs in capture_bs if bs * num_tokens_per_bs % mul_base == 0]
    capture_bs = [bs for bs in capture_bs if bs <= num_max_requests]
    capture_bs = list(sorted(set(capture_bs)))

    assert len(capture_bs) > 0 and capture_bs[0] > 0, f"{capture_bs=}"
    compile_bs = (
        [bs for bs in capture_bs if bs <= server_args.torch_compile_max_bs]
        if server_args.enable_torch_compile
        else []
    )
    return capture_bs, compile_bs
```

代码逻辑：
- 读取 decode phase 的 capture bs 配置。
- 从 request pool size 得到最大请求数。
- 根据 two-batch overlap、gathered buffer、attention CP 计算对齐基数。
- 将最大请求数向上对齐到基数。
- 保留 token 数满足对齐、且不超过最大请求数的 bucket。
- 根据 torch compile 开关生成 compile bucket。

为什么这样写：
- 捕获 bucket 必须同时满足 graph 静态形状和分布式 attention 的对齐要求。
- 补上对齐后的最大请求数，避免小显存或小 `max-running-requests` 下最大 batch 无 graph 可用。

不变量与失败模式：
- `cuda_graph_config.decode.bs` 不能为空。
- 过滤后 bucket 列表必须非空且第一个值大于 0。
- `num_tokens_per_bs` 在投机验证或 dLLM 中不一定为 1，会影响对齐过滤。

Comment：
这个函数解释了为什么配置里的 batch size 不一定等于最终捕获的 graph bucket。

### 3.4 `init_decode_cuda_graph` 按 generation/spec/dLLM 情况捕获 decode graph

问题与约束：
- decode graph 只对生成模型有意义；MindSpore、禁用 graph、CPU 未启用 torch compile 等场景要跳过。
- speculative target verify 和普通 decode 的每请求 token 数不同，capture 名称和 bucket 要随算法变化。

设计选择：
- 初始化时先做跳过条件判断，再根据 spec algorithm 决定 `num_tokens_per_bs`，生成 capture bucket 后创建平台对应的 graph runner。

Explain：
`init_decode_cuda_graph` 负责创建 `decode_cuda_graph_runner` 并统计 graph 显存占用。它根据当前平台选择 out-of-tree graph runner、默认 `DecodeCudaGraphRunner`、CPU graph runner 或 NPU graph runner。

来源：python/sglang/srt/model_executor/model_runner.py L2575-L2650

Code：

```python
def init_decode_cuda_graph(self):
    """Capture device graphs."""
    self.decode_cuda_graph_runner = None
    self.graph_mem_usage = 0

    if not self.is_generation:
        return

    if self.server_args.model_impl.lower() == ModelImpl.MINDSPORE:
        return

    if self.device != "cpu" and check_cuda_graph_backend(
        Phase.DECODE, Backend.DISABLED
    ):
        return

    if self.device == "cpu" and not self.server_args.enable_torch_compile:
        return

    if self.spec_algorithm.is_speculative():
        capture_name = f"{role} verify"
        num_tokens_per_bs = (
            self.spec_algorithm.get_num_tokens_per_bs_for_target_verify(
                self.server_args.speculative_num_draft_tokens,
                self.is_draft_worker,
            )
        )
    else:
        capture_name = f"{role} decode"
        num_tokens_per_bs = 1
    capture_bs, _ = get_batch_sizes_to_capture(self, num_tokens_per_bs)
```

代码逻辑：
- 清空 decode graph runner 和显存统计。
- 非 generation 模型直接返回。
- MindSpore backend 直接返回。
- decode backend disabled 时返回。
- CPU 设备未启用 torch compile 时返回。
- speculative 场景用 target verify 的 token-per-bs，否则普通 decode 为 1。
- 计算 capture bucket。
- 创建平台对应 graph runner。
- 记录 capture 前后可用显存差。

为什么这样写：
- 跳过条件能避免对不支持或无意义的 forward phase 捕获 graph。
- speculative verify 的 shape 与普通 decode 不同，必须用不同 `num_tokens_per_bs`。
- graph 显存占用是启动诊断的重要指标。

不变量与失败模式：
- generation 模型才会进入 decode graph 捕获。
- `get_batch_sizes_to_capture` 可能因配置/对齐过滤后为空而断言失败。
- 平台 graph runner 必须实现与 `DecodeCudaGraphRunner` 兼容的接口。

Comment：
decode graph 是执行热路径优化，但它建立在严格的 phase、shape 和平台条件之上。

### 3.5 `init_prefill_cuda_graph` 对可变 prefill 做更保守的捕获

问题与约束：
- prefill shape 比 decode 更可变，full CUDA graph 不适用；非语言模型、缺 capture size、层结构不符合要求都不能捕获。
- EAGLE target 在某些 prefill backend 下会和 FP4/MoE decode replay 互相干扰。

设计选择：
- 函数先处理 disabled、draft worker、EAGLE target、非语言模型和 capture size 缺失等跳过条件；通过后收集 attention/MoE/indexer 层并创建 `PrefillCudaGraphRunner`。

Explain：
prefill graph 初始化比 decode 更谨慎。它需要能解析语言模型层，收集每层 attention 和 MoE 组件；如果 attention layer 数小于模型层数，说明不满足标准 GQA 等要求，直接禁用 prefill graph。

来源：python/sglang/srt/model_executor/model_runner.py L2651-L2814

Code：

```python
def init_prefill_cuda_graph(self, force_for_draft_worker: bool = False):
    """Initialize prefill CUDA graph runner."""
    self.prefill_cuda_graph_runner = None

    if check_cuda_graph_backend(Phase.PREFILL, Backend.DISABLED):
        if not self.is_draft_worker:
            self.prefill_cuda_graph_runner = self.eager_runner
        return

    if self.is_draft_worker and not force_for_draft_worker:
        return

    if (
        self.spec_algorithm.is_eagle()
        and not self.is_draft_worker
        and not self.server_args.enable_return_hidden_states
        and not check_cuda_graph_backend(Phase.PREFILL, Backend.BREAKABLE)
    ):
        self.prefill_cuda_graph_runner = self.eager_runner
        return

    if not hasattr(self.model, "model"):
        return

    if not self.server_args.cuda_graph_config.prefill.bs:
        return

    self.model.model = resolve_language_model(self.model)
    language_model = getattr(self.model, "language_model", self.model)
```

代码逻辑：
- prefill backend disabled 时，非 draft runner 直接使用 eager runner。
- draft worker 默认跳过，除非显式 force。
- EAGLE target 在非 breakable prefill backend 下回退 eager。
- 非语言模型或 prefill capture size 未设置时返回。
- 解析 language model 和 layer model。
- 遍历层收集 attention layers、MoE layers、MoE fusions、DSA indexers。
- 若 attention layer 数不足，禁用 graph。
- 创建 `PrefillCudaGraphRunner` 并记录显存占用。

为什么这样写：
- prefill 形状和层路径更复杂，只有结构满足条件时才值得捕获。
- disabled 时设置 eager runner 能让 `_forward_raw` 的 extend 分支自然落到 eager。
- layer 收集为 piecewise/breakable prefill graph 提供后端需要的组件列表。

不变量与失败模式：
- `prefill.bs` 必须存在，否则不能决定 capture shape。
- 模型必须能解析到 language model layers。
- 层结构不标准时 attention layers 不足，prefill graph 会被禁用。

Comment：
prefill graph 是有条件优化，源码里大量 early return 是为了保护复杂模型路径的正确性。

## 4. ForwardBatch：从调度态到执行态

### 4.1 `ForwardBatch` 字段区分构造期、forward 派生期和运行期状态

问题与约束：
- 模型 forward 需要同时携带输入 ids、KV cache slot、position、sampling info、multimodal、LoRA、DP padding、context parallel、split prefill 等状态。
- 有些字段在构造时就来自 `ScheduleBatch`，有些必须在 forward 或 graph load 阶段写入。

设计选择：
- `ForwardBatch` dataclass 用字段分组表达生命周期：基础输入、extend 字段、DP padding、runtime-filled 字段、多模态、MRoPE、overlap、context parallel 和 metadata planning marker。

Explain：
`ForwardBatch` 是 ModelRunner 内部的统一执行对象。它不是单纯的 input ids 容器，而是把 attention planning、KV 写入位置、采样、LoRA、多模态和分布式 padding 信息都集中到一个 batch 上。

来源：python/sglang/srt/model_executor/forward_batch_info.py L430-L545

Code：

```python
return_hidden_states_before_norm: bool = False

positions: torch.Tensor = None

extend_num_tokens: Optional[int] = None
extend_seq_lens: Optional[torch.Tensor] = None
extend_prefix_lens: Optional[torch.Tensor] = None
extend_start_loc: Optional[torch.Tensor] = None
extend_prefix_lens_cpu: Optional[List[int]] = None
extend_seq_lens_cpu: Optional[List[int]] = None

original_global_num_tokens_cpu: Optional[List[int]] = None
global_num_tokens_cpu: Optional[List[int]] = None
global_num_tokens_gpu: Optional[torch.Tensor] = None

next_token_logits_buffer: torch.Tensor = None
temperature: torch.Tensor = None
top_p: torch.Tensor = None

mm_input_embeds: Optional[torch.Tensor] = None
cross_attention_custom_mask: Optional[torch.Tensor] = None

mrope_positions: torch.Tensor = None

forward_metadata_ready: bool = False
forward_metadata_planned_bs: Optional[int] = None
forward_metadata_planned_num_tokens: Optional[int] = None
forward_metadata_replan_equivalent: bool = False
```

代码逻辑：
- positions 和 extend 字段描述当前 forward 的 token 位置和 prefill span。
- global token 字段服务 DP attention/MLP sync。
- runtime-filled logits/sampling 字段由 forward 或 graph runner 填充。
- 多模态和 MRoPE 字段连接 multimodal processor。
- metadata ready/planned 字段标记 attention metadata 是否已被外部预规划。

为什么这样写：
- 把 execution state 放在一个对象里，模型层、attention backend、runner 和 sampler 能共享同一份 batch。
- 生命周期字段分组让“构造期”和“运行期”状态边界更清楚。

不变量与失败模式：
- `forward_metadata_ready` 的计划形状必须和当前 batch shape 匹配，否则需要重新规划。
- runtime-filled 字段在构造后可能仍为 None，使用方必须按 forward mode 判断。
- 多模态字段只有对应模型/请求存在多模态输入时才有效。

Comment：
`ForwardBatch` 是模型执行的主协议对象，后面的 runner 分支都围绕它做选择。

### 4.2 `ForwardBatch.init_new` 从 `ScheduleBatch` 拷贝执行核心字段

问题与约束：
- `ScheduleBatch` 属于调度态，包含请求对象、缓存状态和一次性 override；ModelRunner 需要的是设备张量和 forward-only 状态。
- hidden capture、seq_lens CPU cache、grammar 列表等字段只能消费一次，不能污染后续 forward。

设计选择：
- `init_new` 先消费并重置一次性字段，再根据 forward mode 构造 extend 相关字段，最后创建 `ForwardBatch` 并引用 `ScheduleBatch` 的核心张量。

Explain：
`init_new` 是调度态到执行态的转换点。它保留必要的 alias 引用，如 input ids、req pool indices、seq lens、out cache loc，同时把采样、spec、LoRA ids、rid、多模态输入等信息带入 Runner。

来源：python/sglang/srt/model_executor/forward_batch_info.py L612-L722

Code：

```python
@classmethod
def init_new(
    cls,
    batch: ScheduleBatch,
    model_runner: ModelRunner,
):
    capture_hidden_mode = batch.capture_hidden_mode
    batch.capture_hidden_mode = None
    seq_lens_cpu_cache = batch.seq_lens_cpu_cache
    batch.seq_lens_cpu_cache = None
    return_hidden_states_before_norm = batch.return_hidden_states_before_norm
    batch.return_hidden_states_before_norm = False

    if batch.forward_mode.is_decode_or_idle():
        extend_seq_lens = extend_prefix_lens = extend_logprob_start_lens = None
    else:
        extend_seq_lens = batch.extend_lens
        extend_prefix_lens = batch.prefix_lens
        extend_logprob_start_lens = batch.extend_logprob_start_lens

    ret = cls(
        forward_mode=batch.forward_mode,
        batch_size=len(batch.seq_lens),
        input_ids=batch.input_ids,
        req_pool_indices=batch.req_pool_indices,
        seq_lens=batch.seq_lens,
        out_cache_loc=batch.out_cache_loc,
        seq_lens_sum=batch.seq_lens_sum,
        seq_lens_cpu=seq_lens_cpu,
        orig_seq_lens=batch.orig_seq_lens,
        sampling_info=batch.sampling_info,
        spec_info=batch.spec_info,
    )
```

代码逻辑：
- 取出并清空 capture hidden、seq_lens cache、return hidden before norm 等一次性字段。
- 根据请求和 spec info 推导 capture hidden mode。
- decode/idle 不设置 extend 字段，extend/prefill 使用 batch 的 prefix/extend lens。
- 若 batch 有 grammar，则写入 sampling info。
- 校验 seq_lens CPU cache 的 shape。
- 构造 `ForwardBatch`，传入核心输入、cache、采样、spec、多模态、LoRA 和 rid 字段。

为什么这样写：
- 一次性 override 被消费后清空，避免同一个 `ScheduleBatch` 后续 forward 误用旧值。
- 设备张量尽量 alias 原 batch，减少不必要拷贝。
- ForwardBatch 只承载模型执行需要的状态，隔离调度对象细节。

不变量与失败模式：
- `seq_lens_cpu_cache` 如果存在，shape 必须匹配当前 `batch.seq_lens`。
- decode/idle 模式不应该携带 extend lens。
- sampling_info 在 overlap 模式下应已替换为 forward-only 副本。

Comment：
这一步把 Scheduler 的请求集合整理成 Runner 可以直接消费的执行 batch。

### 4.3 `init_new` 后半段补 position、DP token 计数、MRoPE 和 LoRA

问题与约束：
- Decode、extend、DLLM、spec、多模态 MRoPE 的 position 规则不同。
- DP attention/MLP sync 需要 global token count 的 GPU/CPU 镜像；LoRA 需要在 forward 前准备 adapter batch。

设计选择：
- 构造后按 mode 初始化 positions 和 extend metadata；按 DP token 信息构造 GPU tensor；必要时计算 MRoPE、准备 LoRA batch，并返回完整 ForwardBatch。

Explain：
`init_new` 后半段把执行所需派生字段填齐。decode 用 `clamp_position(seq_lens)`，extend 用 `compute_position` 计算 positions 和 start loc；spec/DLLM 会覆盖 position；MRoPE 和 LoRA 都在 forward 前完成准备。

来源：python/sglang/srt/model_executor/forward_batch_info.py L752-L876

Code：

```python
num_tokens = len(batch.input_ids) if batch.input_ids is not None else 0
if enable_num_token_non_padded():
    ret.num_token_non_padded = torch.tensor(num_tokens, dtype=torch.int32).to(
        device, non_blocking=True
    )
ret.num_token_non_padded_cpu = num_tokens

if batch.global_num_tokens is not None:
    ret.original_global_num_tokens_cpu = batch.global_num_tokens
    ret.global_num_tokens_cpu = global_num_tokens
    ret.global_num_tokens_gpu = torch.tensor(
        global_num_tokens, dtype=torch.int64
    ).to(device, non_blocking=True)

if ret.forward_mode.is_idle():
    ret.positions = torch.empty((0,), dtype=torch.int64, device=device)
    return ret

if ret.forward_mode.is_decode() or ret.forward_mode.is_target_verify():
    if ret.positions is None:
        ret.positions = clamp_position(batch.seq_lens)
else:
    ret.extend_seq_lens = torch.tensor(
        extend_seq_lens, dtype=torch.int32
    ).to(device, non_blocking=True)
    ret.extend_prefix_lens = torch.tensor(
        extend_prefix_lens, dtype=torch.int32
    ).to(device, non_blocking=True)
    positions, ret.extend_start_loc = compute_position(
        model_runner.server_args.attention_backend,
        ret.extend_prefix_lens,
        ret.extend_seq_lens,
        ret.extend_num_tokens,
    )
```

代码逻辑：
- 计算当前 batch 的非 padding token 数。
- 若存在 DP global token 信息，创建 CPU/GPU 镜像。
- idle 模式设置空 positions 后直接返回。
- decode/target verify 使用 seq_lens 推导当前位置。
- extend/prefill 创建 extend seq/prefix lens tensor，并计算 positions/start loc。
- MRoPE 模型按 spec 或普通多模态路径计算 mrope positions。
- LoRA 开启时 fetch 新 adapter 并 prepare LoRA batch。
- 特定 DCP/HIP 场景设置 KV write mask。

为什么这样写：
- position 和 attention metadata 必须在模型 forward 前确定。
- DP token 计数要同时服务 GPU kernel 和 CPU 侧调度/日志逻辑。
- LoRA adapter 需要在 forward 前加载到 memory pool 并绑定 batch metadata。

不变量与失败模式：
- extend mode 下 `extend_seq_lens` 与 `extend_prefix_lens` 类型必须一致，要么都是 list，要么是 tensor。
- MRoPE 依赖多模态输入里的 grid/position 信息。
- LoRA overlap loading 未开启时，`fetch_new_loras` 必须在 prepare batch 前完成。

Comment：
`ForwardBatch` 创建不是简单拷贝，它会完成模型 forward 之前的大部分 execution metadata 准备。

## 5. Forward 调度：graph 与 eager

### 5.1 `ModelRunner.forward` 包住调试、profiling、expert 记录与 EP 恢复

问题与约束：
- 每次 forward 既要执行模型，也要维护 profiling、canary、expert distribution、routed experts/indexer capture、elastic EP 等横切逻辑。
- 这些逻辑不能散落在 graph runner 和 eager runner 内部，否则不同执行路径难以保持一致。

设计选择：
- `forward` 作为外层 wrapper，统一增加 `forward_pass_id`、启动调试/profiling 上下文、调用 `_forward_raw`，再收集 metrics 和专家相关输出。

Explain：
`ModelRunner.forward` 不直接决定 eager/graph 分支；它先处理 deprecated 参数、profiling、canary 和 expert recorder，然后把真正的执行交给 `_forward_raw`。执行结束后再把 metrics、routed experts、indexer topk 和 EP 维护动作补到输出对象上。

来源：python/sglang/srt/model_executor/model_runner.py L2954-L3046

Code：

```python
def forward(
    self,
    forward_batch: ForwardBatch,
    skip_attn_backend_init: Optional[bool] = None,
    pp_proxy_tensors: Optional[PPProxyTensors] = None,
    reinit_attn_backend: bool = False,
    split_forward_count: int = 1,
) -> ModelRunnerOutput:
    forward_batch.apply_deprecated_skip_attn_backend_init(skip_attn_backend_init)

    self.forward_pass_id += 1

    if self.msprobe_debugger is not None:
        rank_id = (
            self.gpu_id if self.dp_size is not None and self.dp_size > 1 else None
        )
        self.msprobe_debugger.start(model=self.model, rank_id=rank_id)

    step_span_ctx = profile_range(_build_step_span_name(forward_batch))

    with (
        canary_ctx,
        step_span_ctx,
        get_global_expert_distribution_recorder().with_forward_pass(
            self.forward_pass_id,
            forward_batch,
        ) as recorder_outputs,
    ):
        output = self._forward_raw(
            forward_batch,
            pp_proxy_tensors,
            reinit_attn_backend,
            split_forward_count,
        )
```

代码逻辑：
- 将旧 `skip_attn_backend_init` 映射到 ForwardBatch metadata marker。
- 递增 forward pass id。
- 可选启动 msprobe 调试。
- 建立 profiling span 和 canary 上下文。
- 在 expert recorder 上下文中调用 `_forward_raw`。
- forward 后写入 expert distribution metrics。
- 可选捕获 routed experts/indexer topk，并做 EPLB/elastic EP 后处理。

为什么这样写：
- graph/eager 分支共享同一套横切逻辑，减少路径差异。
- expert 和 indexer capture 需要知道最终是否跑 graph、graph batch 是多少，因此放在 `_forward_raw` 之后。

不变量与失败模式：
- `forward_batch` 必须已经由 `ForwardBatch.init_new` 或等价路径构造。
- canary 只在非 draft worker 且 manager 存在时启用。
- elastic EP 相关后处理依赖输出对象包含可用的 forward 结果。

Comment：
外层 `forward` 是统一观测和维护入口，真正执行路径在 `_forward_raw`。

### 5.2 `_forward_raw` 优先 replay decode graph，否则准备 live batch 再走 prefill graph 或 eager

问题与约束：
- Decode CUDA graph 只有在 forward mode、runner 存在、batch shape 都满足时才能 replay。
- 非 graph 路径需要先做 DP/MLP-sync padding 和 attention-TP token 计数标准化。

设计选择：
- `_forward_raw` 先判断 decode/cpu graph 是否可运行，能跑就早返回；否则调用 `_prepare_eager_forward_batch`，再按 split prefill、prefill graph、eager 三条路径执行。

Explain：
这是 ModelRunner 的执行分发中心。decode graph 是最早的 fast path；没有命中 graph 时，所有 live batch 都先经过 eager forward batch 准备，然后 split prefill 留在 ModelRunner，extend/prefill graph 由 prefill graph runner 处理，其余 decode/extend/idle 交给 `EagerRunner`。

来源：python/sglang/srt/model_executor/model_runner.py L3048-L3141

Code：

```python
def _forward_raw(
    self,
    forward_batch: ForwardBatch,
    pp_proxy_tensors: Optional[PPProxyTensors],
    reinit_attn_backend: bool = False,
    split_forward_count: int = 1,
) -> ModelRunnerOutput:
    if has_forward_context():
        ctx_mgr = contextlib.nullcontext()
    else:
        ctx_mgr = forward_context(ForwardContext(attn_backend=self.attn_backend))
    with ctx_mgr:
        mode_check = (
            forward_batch.forward_mode.is_cpu_graph
            if self.device == "cpu"
            else forward_batch.forward_mode.is_cuda_graph
        )
        can_run_graph = bool(
            mode_check()
            and self.decode_cuda_graph_runner
            and self.decode_cuda_graph_runner.can_run_graph(forward_batch)
        )

        if can_run_graph:
            ret = self.decode_cuda_graph_runner.execute(
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)

        self._prepare_eager_forward_batch(forward_batch)

        if forward_batch.forward_mode.is_split_prefill():
            ret = self.forward_split_prefill(
                forward_batch,
                reinit_attn_backend=reinit_attn_backend,
                forward_count=split_forward_count,
            )
        elif (
            forward_batch.forward_mode.is_extend(include_draft_extend_v2=True)
            and not isinstance(self.prefill_cuda_graph_runner, EagerRunner)
            and self.prefill_cuda_graph_runner is not None
            and self.prefill_cuda_graph_runner.can_run_graph(forward_batch)
            and get_cp_strategy() is None
        ):
            ret = self.prefill_cuda_graph_runner.execute(forward_batch, **kwargs)
            can_run_graph = True
        else:
            ret = self.eager_runner.execute(
                forward_batch, pp_proxy_tensors=pp_proxy_tensors
            )
```

代码逻辑：
- 若当前没有 forward context，创建一个使用默认 attention backend 的 context。
- 根据 device 判断 CPU graph 或 CUDA graph mode。
- decode graph runner 存在且 `can_run_graph` 时直接 replay。
- graph 未命中时准备 eager/live batch。
- split prefill 走 `forward_split_prefill`。
- extend 且 prefill graph runner 可用时走 prefill graph。
- 其他情况交给 eager runner。
- 最后对 PP last rank 的 DP/MLP sync 结果做 post process。

为什么这样写：
- decode graph 使用静态 buffer，不需要 live batch padding 准备，因此可以早返回。
- eager/prefill graph 共用 live batch 准备逻辑，保证 DP padding 和 collectives 所需 metadata 一致。
- split prefill 是层切分执行，留在 ModelRunner 内部处理。

不变量与失败模式：
- `decode_cuda_graph_runner.can_run_graph` 必须同时检查 mode 和 shape。
- prefill graph 不能在 CP strategy 开启时运行该分支。
- eager runner 必须总是存在，作为所有 graph 未命中的 fallback。

Comment：
这段是模型执行路径的核心分岔点：先尝试最强约束的 graph，失败后逐步退到更通用的 eager。

### 5.3 `EagerRunner` 用固定 buffer 加载 live batch，再按 mode 调模型 forward

问题与约束：
- 即使不跑 CUDA graph，也希望复用固定输入 buffer，避免每步重新分配。
- decode、extend、idle 的 attention metadata 初始化和模型 forward kwargs 不同。

设计选择：
- `EagerRunner.load_batch` 把 live batch 拷入 eager registry 的固定 buffer；`execute` 按 forward mode 分派到 decode、extend、idle；各分支在需要时初始化 attention metadata 后调用模型 forward。

Explain：
Eager runner 不是“无优化直接调模型”。它同样维护最大静态 buffer registry，并在 decode/extend 前调用 attention backend 的 metadata 初始化。extend 分支还处理 DCP、CP-V2、HIP piecewise graph fallback 等路径。

来源：python/sglang/srt/model_executor/runner/eager_runner.py L167-L253

Code：

```python
def load_batch(
    self, forward_batch: ForwardBatch, pp_proxy_tensors=None, **kwargs
) -> ForwardBatch:
    if envs.SGLANG_EAGER_INPUT_NO_COPY.get():
        return replace(forward_batch)
    raw_bs = forward_batch.batch_size
    if forward_batch.input_ids is not None:
        raw_num_tokens = forward_batch.input_ids.shape[0]
    elif forward_batch.input_embeds is not None:
        raw_num_tokens = forward_batch.input_embeds.shape[0]
    else:
        raw_num_tokens = 0
    registry = self._eager_registry
    registry.fill_from(
        forward_batch,
        raw_bs=raw_bs,
        padded_bs=raw_bs,
        raw_num_tokens=raw_num_tokens,
        padded_num_tokens=raw_num_tokens,
        pp_proxy_tensors=pp_proxy_tensors,
    )
    return registry.extract_buffer(
        padded_bs=raw_bs,
        padded_num_tokens=raw_num_tokens,
        forward_batch_template=forward_batch,
    )

def execute(
    self, forward_batch: ForwardBatch, pp_proxy_tensors=None, **kwargs
) -> Any:
    mode = forward_batch.forward_mode
    if mode.is_decode():
        return self._execute_decode(forward_batch, pp_proxy_tensors)
    if mode.is_idle():
        return self._execute_idle(forward_batch, pp_proxy_tensors)
    if mode.is_extend(include_draft_extend_v2=True):
        return self._execute_extend(forward_batch, pp_proxy_tensors)
    raise ValueError(f"Invalid forward mode for eager runner: {mode}")
```

代码逻辑：
- 默认把 live batch 内容拷入 eager static registry。
- 根据 input ids 或 input embeds 计算真实 token 数。
- registry 以 raw batch/token shape 抽取 buffer view。
- `execute` 按 decode/idle/extend 分支调用内部方法。
- decode 分支解析 PDMux backend，初始化 forward metadata，然后调用 `model.forward(input_ids, positions, forward_batch, **kwargs)`。

为什么这样写：
- eager fallback 也复用静态 buffer，减少内存分配和与 graph buffer 的差异。
- mode 分派集中在 runner 内，ModelRunner 不需要知道每种 eager 细节。
- attention metadata 初始化在具体 mode 分支内完成，可以处理 decode/extend 的差异。

不变量与失败模式：
- `ForwardBatch.forward_mode` 必须是 eager runner 支持的 decode/idle/extend。
- registry buffer 大小必须覆盖当前 raw token 数。
- 若 metadata 已由 pre-planner 标记 ready，分支不能重复规划。

Comment：
EagerRunner 是通用 fallback，也是 graph 共享 buffer 的基准路径。

### 5.4 Decode graph runner 在 replay 前 pad 到 bucket 并构造 graph view

问题与约束：
- 已捕获 graph 的 batch size 是离散 bucket，live batch 必须 pad 到某个 bucket 后才能 replay。
- Attention metadata 需要基于 graph buffer view 重新构造，而不是直接使用 raw ForwardBatch。

设计选择：
- `load_batch` 根据 raw batch size 或 DP global token 信息 pad 到 capture bucket，把 batch 拷入 graph buffer registry，构造 replay ForwardBatch view，并在 graph 外初始化 attention metadata。

Explain：
decode graph runner 的 `load_batch` 是 raw batch 到 graph replay view 的转换。它选择 bucket、填充静态 buffer、构造 `fb_view`，并把 replay key 记录为 batch size、PDMux stream 和 LoRA variant 的组合。

来源：python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py L930-L1010

Code：

```python
buffers = self.buffers
self.recapture_if_needed(forward_batch)

raw_bs = forward_batch.batch_size
raw_num_token = raw_bs * self.num_tokens_per_bs

if self.require_mlp_tp_gather:
    max_num_tokens = max(forward_batch.global_num_tokens_cpu)
    max_batch_size = (
        max_num_tokens / self.num_tokens_per_bs
        if self.model_runner.spec_algorithm.is_eagle()
        or self.model_runner.spec_algorithm.is_standalone()
        or self.model_runner.spec_algorithm.is_dflash()
        else max_num_tokens
    )
    bs = self._pad_to_bucket(int(max_batch_size), self.capture_bs)
else:
    bs = self._pad_to_bucket(raw_bs, self.capture_bs)

self.buffer_registry.fill_from(
    forward_batch,
    raw_bs=raw_bs,
    padded_bs=bs,
    raw_num_tokens=raw_num_token,
    padded_num_tokens=bs * self.num_tokens_per_bs,
    pp_proxy_tensors=pp_proxy_tensors,
)

fb_view = build_replay_fb_view(
    forward_batch=forward_batch,
    buffers=buffers,
    bs=bs,
    raw_bs=raw_bs,
    num_tokens=bs * self.num_tokens_per_bs,
    seq_len_fill_value=self.seq_len_fill_value,
    capture_forward_mode=self.capture_forward_mode,
    is_encoder_decoder=self.is_encoder_decoder,
)
attn_backend.init_forward_metadata_out_graph(fb_view)
```

代码逻辑：
- 必要时根据 batch 重新捕获 graph。
- 计算 raw batch size 和 raw token 数。
- DP/MLP TP gather 场景根据 global token 选择 bucket，否则按 raw batch size 选择 bucket。
- 将 raw batch 数据填入 graph buffer registry。
- 构造 replay 用的 `ForwardBatch` view。
- 在 graph 外初始化 attention metadata。
- 保存 raw/padded batch 信息和 replay graph key。

为什么这样写：
- graph replay 只能使用捕获时的静态 buffer 和 shape，live batch 必须转换为对应 view。
- attention metadata 在 graph 外初始化，可避免每次 replay 进入 Python graph capture 路径。
- replay key 纳入 LoRA/PDMux variant，防止不同执行变体误用同一 graph。

不变量与失败模式：
- raw batch 必须能 pad 到某个 `capture_bs` bucket。
- buffer registry 必须有足够容量容纳 padded token 数。
- `global_num_tokens_cpu` 在 require MLP TP gather 时必须存在。

Comment：
decode graph runner 的关键不是 replay 一行代码，而是 replay 前把 live batch 安全投影到 captured shape。

### 5.5 Decode graph runner `execute` 负责加载 batch、发布 WAR event 并 replay

问题与约束：
- overlap 调度下，下一个 batch 可能写 shared req/token buffer；当前 graph forward 需要发布读完成事件，避免 write-after-read 竞争。
- Replay 要在 backend 的 replay session 中执行，并裁掉 graph padding 产生的多余输出。

设计选择：
- `execute` 在 timer 和 replay session 中调用 `load_batch`，必要时记录 read-done event，然后用 `_replay_graph_key` replay；输出为 logits 时再按 raw token/request 裁剪。

Explain：
`execute` 是 graph runner 的外层执行入口。它先把 live batch 加载进静态 buffer，随后在 decode 或 DFlash target verify 场景发布 WAR fastpath event，最后调用 backend replay。

来源：python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py L1012-L1045

Code：

```python
def execute(
    self,
    forward_batch: ForwardBatch,
    pp_proxy_tensors: Optional[PPProxyTensors] = None,
) -> Union[LogitsProcessorOutput, PPProxyTensors]:
    timer_ctx = (
        self.model_runner.device_timer.wrap(
            metadata={"category": forward_batch.forward_mode.name.lower()}
        )
        if self.model_runner.device_timer
        else contextlib.nullcontext()
    )
    with timer_ctx, self.backend.replay_session():
        self.load_batch(forward_batch, pp_proxy_tensors)
        if forward_batch.forward_mode.is_decode() or (
            forward_batch.forward_mode.is_target_verify()
            and self.model_runner.spec_algorithm.is_dflash()
        ):
            read_done = self.device_module.Event()
            read_done.record()
            self.model_runner.war_fastpath_read_done_event = read_done
        output = self.backend.replay(self._replay_graph_key, forward_batch)

    if isinstance(output, LogitsProcessorOutput):
        if self.is_dllm:
            next_token_logits = None
            full_logits = (
                output.full_logits[: self.raw_num_token]
                if output.full_logits is not None
                else None
            )
```

代码逻辑：
- 建立可选 device timer context。
- 进入 backend replay session。
- 调用 `load_batch` 准备 graph buffer 和 replay key。
- decode 或 DFlash target verify 时记录 read-done event。
- 调用 backend replay。
- 输出为 logits 时按 raw token 数裁剪 padding 结果。

为什么这样写：
- replay session 封装 backend 对 graph replay 的上下文要求。
- WAR event 让 overlap schedule 知道当前 graph 对共享 buffer 的读取已经完成。
- padding 裁剪保证下游只看到真实请求的 logits。

不变量与失败模式：
- `load_batch` 必须先设置 `_replay_graph_key`。
- backend 必须已捕获对应 graph key。
- padding 输出裁剪要和 `raw_num_token/raw_bs` 一致。

Comment：
graph runner 的执行路径把静态 graph 和动态 batch 之间的同步、padding、裁剪都封装在 runner 内。

### 5.6 process-wide input buffer pool 让 eager 与 graph 共享物理分配

问题与约束：
- 多个 runner 都会注册同名输入 buffer；如果每个 runner 独立分配，captured graph 的 buffer 指针和 eager fallback 可能不一致。
- 已捕获 graph 不能在后续注册时被重新指向其他 buffer。

设计选择：
- 用进程级 `_forward_input_buffer_pool` 按 `(name, numel, dtype, device)` 作为 key；第一个 buffer 成为 canonical，后续同 key 返回其 view。

Explain：
`share_input_buffer` 保证同名、同形状、同 dtype、同 device 的输入 buffer 共享一个物理 allocation。`share_input_buffers_in` 会递归处理 dataclass 或 dict 中的 tensor buffer，并在 NPU 上禁用共享以规避精度问题。

来源：python/sglang/srt/model_executor/input_buffers.py L16-L66

Code：

```python
def share_input_buffer(name: str, new_buffer: torch.Tensor) -> torch.Tensor:
    """Coalesce a buffer by ``(name, size, dtype, device)`` into the
    process-wide input-buffer pool.
    """
    key: _PoolKey = (name, new_buffer.numel(), new_buffer.dtype, new_buffer.device)
    canonical = _forward_input_buffer_pool.get(key, None)
    if canonical is None:
        _forward_input_buffer_pool[key] = new_buffer
        canonical = new_buffer
    return canonical.as_strided(new_buffer.size(), new_buffer.stride())

def share_input_buffers_in(obj) -> None:
    """Pool every tensor buffer on ``obj``."""
    if is_npu():
        return

    for name, buffer in list(vars(obj).items()):
        if buffer is None:
            continue
        if dataclasses.is_dataclass(buffer):
            buffer = vars(buffer)
        if isinstance(buffer, dict):
            for sub_name, sub_buffer in buffer.items():
                buffer[sub_name] = share_input_buffer(f"{name}.{sub_name}", sub_buffer)
        else:
            setattr(obj, name, share_input_buffer(name, buffer))
```

代码逻辑：
- 以 buffer 名、元素数、dtype、device 组成 key。
- key 不存在时登记当前 buffer 为 canonical。
- key 已存在时返回 canonical 的同 shape/stride view。
- `share_input_buffers_in` 遍历对象字段，处理 tensor、dict 或 dataclass 中的 tensor。
- NPU 平台直接跳过共享。

为什么这样写：
- graph capture 依赖稳定 `data_ptr`，共享 pool 避免后续 runner 改变 captured pointer。
- eager fallback 和 graph runner 使用同一物理 buffer，减少显存碎片和路径差异。

不变量与失败模式：
- 同 key buffer 必须语义等价，否则共享会让不同 runner 覆盖同一内存。
- buffer 会在 replay/forward 前立即填充，runner 执行必须顺序或互斥。
- NPU 上不共享，说明该优化有平台精度约束。

Comment：
这个 buffer pool 是 eager runner 与 graph runner 能共存的重要底层约定。

## 6. Worker 的 generation 输出与采样

### 6.1 `forward_batch_generation` 在 PP 末 rank 采样，非末 rank 只传 proxy

问题与约束：
- Pipeline parallel 下只有最后一个 PP rank 拿到 logits 并执行采样；非末 rank 只产生 hidden states/proxy tensors。
- overlap、grammar、spec verify、prefill-only 请求都需要不同的采样行为。

设计选择：
- Worker 调用 `model_runner.forward` 后，如果是 PP last rank 就构造 `GenerationBatchResult` 并根据 verify/overlap/prefill-only 分支处理采样；非 last rank 返回 PP proxy tensors。

Explain：
这段把模型 forward 输出变成 Scheduler 能消费的 generation batch result。普通 generation 直接调用 `model_runner.sample`；grammar overlap 可延迟采样；prefill-only 请求返回 dummy token id，并在需要 logprob 时只计算 logprobs。

来源：python/sglang/srt/managers/tp_worker.py L506-L572

Code：

```python
if self.pp_group.is_last_rank:
    out = self.model_runner.forward(
        forward_batch,
        pp_proxy_tensors=pp_proxy_tensors,
    )
    logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
    batch_result = GenerationBatchResult(
        logits_output=logits_output,
        can_run_cuda_graph=can_run_cuda_graph,
        expert_distribution_metrics=out.expert_distribution_metrics,
        routed_experts_output=out.routed_experts_output,
        indexer_topk_output=out.indexer_topk_output,
    )

    if is_verify:
        return batch_result

    if (
        self.enable_overlap
        and not self.enable_spec
        and forward_batch.sampling_info.grammars is not None
    ):
        def sample_batch_func():
            batch_result.next_token_ids = self.model_runner.sample(
                logits_output, forward_batch
            )
            return batch_result

        batch_result.delay_sample_func = sample_batch_func
        return batch_result

    if not forward_batch.is_prefill_only:
        batch_result.next_token_ids = self.model_runner.sample(
            logits_output, forward_batch
        )
```

代码逻辑：
- PP last rank 执行 forward 并读取 logits/can_run_graph。
- 构造 generation result，携带 expert/indexer 输出。
- verify 请求跳过采样，直接返回。
- overlap 且 grammar 存在时，把采样包装成延迟函数。
- 普通请求调用 `model_runner.sample` 得到 next token ids。
- prefill-only 请求生成 dummy token，并按需计算 logprobs。
- 非 last rank 返回 `pp_hidden_states_proxy_tensors`。

为什么这样写：
- 采样只能在有 logits 的 PP last rank 发生。
- verify 路径由 spec worker 后续发布结果，不在这里采样。
- grammar overlap 延迟采样可以把 structured output 更新与 pipeline overlap 配合起来。

不变量与失败模式：
- 非 last rank 的 `out.logits_output` 实际是 PP proxy tensors。
- prefill-only dummy token 数量必须等于序列数，而不是 token 总数。
- 延迟采样闭包会持有 `logits_output` 和 `forward_batch`，需要注意显存生命周期。

Comment：
`ModelRunner` 负责 forward 和 sample 能力，`TpModelWorker` 决定在 PP/overlap/spec 场景下何时调用 sample。

### 6.2 `ModelRunner.sample` 在 logits 上应用采样前处理并调用 sampler

问题与约束：
- 采样前要应用 grammar vocab mask、logits bias，并及时释放可能被 overlap 闭包持有的 GPU mask。
- Prefill 和 decode 选择 logits 位置的规则不同。

设计选择：
- `_preprocess_logits` 更新 regex vocab mask、应用 logits bias、释放 vocab mask；`sample` 再调用 sampler，并根据 forward mode 传入 positions 或 `seq_lens - 1`。

Explain：
采样不在 runner 后端里做，而是在 `ModelRunner.sample` 中统一执行。这样 graph/eager 的 logits output 可以共享同一套 sampling info、logprob 和 ngram token table 更新逻辑。

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
- 更新 structured output 的 regex vocab mask。
- 将 logits bias 应用到 next-token logits。
- 清空 `sampling_info.vocab_mask`，避免 overlap 延迟采样持有 GPU tensor。
- 调用 sampler，传入 logprob 和 top-logprob 需求。
- decode 模式使用 `forward_batch.positions`，prefill/extend 使用 `seq_lens - 1`。
- 更新 ngram token table。

为什么这样写：
- 采样逻辑和 forward 后端解耦，graph/eager 输出都进入同一采样路径。
- vocab mask 用完即释放，避免 structured output 在 overlap 模式下形成稳定显存泄漏。
- 位置选择反映 decode 逐 token 与 prefill 取最后 token 的差异。

不变量与失败模式：
- `logits_output.next_token_logits` 必须存在，除非调用方只做 prefill-only logprob 路径。
- `sampling_info` 必须与当前 batch 对齐。
- ngram table 更新依赖 next token ids 和 batch 中的请求状态一致。

Comment：
采样是 ModelRunner 的后处理能力，不属于 EagerRunner 或 GraphRunner 的职责。
