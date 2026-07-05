---
type: batch-doc
module: 23-Distributed
batch: "23"
doc_type: walkthrough
title: "分布式并行 · 源码走读"
tags:
  - sglang/batch/23
  - sglang/module/distributed
  - sglang/doc/walkthrough
aliases:
  - "02-源码走读"
updated: 2026-07-05
---

# 分布式并行 · 源码走读

> 走读主线：SGLang 的 distributed 层先用 torch.distributed 建 WORLD，再按 TP、attention、MoE、DCP 等语义切出多个 `GroupCoordinator`。模型层只调用 `communication_op.py` 的薄 API；具体使用 NCCL、custom all-reduce、PyMSCCLPP、共享内存 broadcaster 或 torch fallback，由 group coordinator 在运行时决定。DP Controller 则是另一条控制面：把 TokenizerManager 请求路由到多个 Scheduler 进程。

---

## 1. 默认进程组与模型并行组

### 1.1 init_distributed_environment 建 WORLD 并保存 subgroup timeout

问题与约束：
- SGLang 需要先初始化 torch.distributed WORLD，后续 TP、MoE、attention 等子组才能复用相同 rank/world_size 与超时配置。

设计选择：
- `init_distributed_environment` 在默认进程组未初始化时调用 `torch.distributed.init_process_group`；如果用户传入 `timeout`，转换成 `timedelta` 并保存到 `_MODEL_PARALLEL_GROUP_TIMEOUT`，供子组创建复用。

Explain：
函数还处理 Mooncake backend 的 host IP 与 process group options；初始化完成后根据 `local_rank` 创建 `_WORLD` group，若 `_WORLD` 已存在则检查 world size 一致。

来源：python/sglang/srt/distributed/parallel_state.py L1880-L1964

Code：

```python
def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str = "nccl",
    timeout: Optional[int] = None,
    moe_a2a_backend: Optional[str] = None,
    recovered_rank: bool = False,
):
    if not torch.distributed.is_initialized():
        global _MODEL_PARALLEL_GROUP_TIMEOUT
        assert distributed_init_method is not None
        if timeout is not None:
            assert isinstance(timeout, (int)), "timeout must be a number"
            assert timeout > 0, "timeout must be positive"
            timeout = timedelta(seconds=timeout)

        _MODEL_PARALLEL_GROUP_TIMEOUT = timeout

        pg_options = get_torch_distributed_pg_options()
        torch.distributed.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
            timeout=timeout,
            pg_options=pg_options,
        )

    if local_rank == -1:
        if distributed_init_method == "env://":
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        else:
            local_rank = rank
    global _WORLD
    if _WORLD is None:
        ranks = list(range(torch.distributed.get_world_size()))
        _WORLD = init_world_group(
            ranks, local_rank, backend, recovered_rank=recovered_rank
        )
    else:
        assert (
            _WORLD.world_size == torch.distributed.get_world_size()
        ), "world group already initialized with a different world size"
```

代码逻辑：
- 仅在默认进程组未初始化时执行 `init_process_group`。
- 将正整数 timeout 转成秒级 `timedelta`。
- 把 timeout 写入模块级 `_MODEL_PARALLEL_GROUP_TIMEOUT`。
- 推断或读取 local rank。
- 创建或校验 `_WORLD` group。

为什么这样写：
- 子组 collective 不应回落到 backend 默认超时，否则长时间加载或大规模通信更难排障。
- WORLD group 是后续所有模型并行组的父坐标系。

不变量与失败模式：
- `timeout` 必须是正整数。
- 重新初始化时 world size 必须与已有 `_WORLD` 一致。
- Mooncake/NIXL 等 backend 依赖的额外 store 或 options 创建失败会影响对应能力。

Comment：
`init_distributed_environment` 不是只包一层 torch API，它把后续 group 创建所需的超时和 local rank 也固定下来。

### 1.2 initialize_model_parallel 先校验 world_size，再构造 TP group

问题与约束：
- TP、PP、DCP、attention、MoE 等 group 都建立在相同 WORLD rank 空间中；如果尺寸关系不合法，后续 collective 会在运行期卡死。

设计选择：
- 入口先断言 `world_size == tp_size * pp_size`，校验 DCP 与 TP 的整除关系；随后按连续 rank 切出 TP group，并只在 TP group 上启用 message queue broadcaster。

Explain：
TP group 的 rank 列表形如 `[0..tp_size-1]`、`[tp_size..2*tp_size-1]`。`_TP` 通过 `init_model_parallel_group` 创建，group name 固定为 `"tp"`。

来源：python/sglang/srt/distributed/parallel_state.py L2030-L2079

Code：

```python
assert torch.distributed.is_initialized()
world_size: int = torch.distributed.get_world_size()
backend = backend or torch.distributed.get_backend(get_world_group().device_group)

if world_size != tensor_model_parallel_size * pipeline_model_parallel_size:
    raise RuntimeError(
        f"world_size ({world_size}) is not equal to "
        f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
        f"pipeline_model_parallel_size ({pipeline_model_parallel_size})"
    )
if decode_context_parallel_size < 1:
    raise RuntimeError(
        f"decode_context_parallel_size ({decode_context_parallel_size}) must be >= 1"
    )
if tensor_model_parallel_size % decode_context_parallel_size != 0:
    raise RuntimeError(
        f"tensor_model_parallel_size ({tensor_model_parallel_size}) must be divisible by "
        f"decode_context_parallel_size ({decode_context_parallel_size})"
    )

num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
global _TP
assert _TP is None, "tensor model parallel group is already initialized"
group_ranks = []
for tp_group_idx in range(num_tensor_model_parallel_groups):
    ranks = list(
        range(
            tp_group_idx * tensor_model_parallel_size,
            (tp_group_idx + 1) * tensor_model_parallel_size,
        )
    )
    group_ranks.append(ranks)

_TP = init_model_parallel_group(
    group_ranks,
    get_world_group().local_rank,
    backend,
    use_message_queue_broadcaster=envs.SGLANG_USE_MESSAGE_QUEUE_BROADCASTER.get(),
    group_name="tp",
    recovered_rank=recovered_rank,
)
```

代码逻辑：
- 要求默认进程组已经初始化。
- 根据 world backend 推断模型并行 backend。
- 校验 TP、PP、DCP 尺寸关系。
- 按 TP size 切连续 rank。
- 创建 `_TP` coordinator。

为什么这样写：
- 模型层的 tensor parallel collective 依赖固定 TP group，不能在 forward 时临时推断。
- 提前 fail 比 collective 运行时 hang 更可诊断。

不变量与失败模式：
- `_TP` 只能初始化一次。
- `world_size` 必须等于 TP 和 PP 尺寸乘积。
- DCP size 必须能整除 TP size。

Comment：
TP group 是后面 attention group、MoE group 和通信 API 的基本分块。

### 1.3 attention group 从 TP group 内继续切分

问题与约束：
- DP Attention 会让 attention 计算和 MoE/模型 TP 的并行维度不完全一致；同一个模型 rank 需要落在不同语义的 group 中。

设计选择：
- 在 TP group 内派生 `attn_cp_size`、`attn_tp_size`；如果 attention context 或 attention TP 与 TP 完全等价，则直接复用 `_TP`，否则创建独立 `_ATTN_CP` 或 `_ATTN_TP`。

Explain：
`_ATTN_CP` 按 attention TP stride 取 rank，`_ATTN_TP` 按连续 attention TP rank 切组；attention TP group 禁用 custom all-reduce 和 torch symmetric memory all-reduce，仅在需要同步 token id 或 symmetric memory 时启用 PyNCCL。

来源：python/sglang/srt/distributed/parallel_state.py L2126-L2196

Code：

```python
attn_dp_size = attention_data_parallel_size
attn_cp_size = attention_context_model_parallel_size
attn_tp_size = tensor_model_parallel_size // attn_cp_size // attn_dp_size

global _ATTN_CP
assert (
    _ATTN_CP is None
), "attention context model parallel group is already initialized"
if attn_cp_size == tensor_model_parallel_size:
    _ATTN_CP = _TP
else:
    group_ranks = []
    for tp_group_idx in range(num_tensor_model_parallel_groups):
        for dp_idx in range(attn_dp_size):
            for attn_tp_idx in range(attn_tp_size):
                st = (
                    tp_group_idx * tensor_model_parallel_size
                    + dp_idx * attn_tp_size * attn_cp_size
                    + attn_tp_idx
                )
                en = (
                    tp_group_idx * tensor_model_parallel_size
                    + (dp_idx + 1) * attn_tp_size * attn_cp_size
                    + attn_tp_idx
                )
                ranks = list(range(st, en, attn_tp_size))
                group_ranks.append(ranks)
    _ATTN_CP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=envs.SGLANG_USE_MESSAGE_QUEUE_BROADCASTER.get(),
        group_name="attn_cp",
        recovered_rank=recovered_rank,
    )

global _ATTN_TP
assert (
    _ATTN_TP is None
), "attention tensor model parallel group is already initialized"
if attn_tp_size == tensor_model_parallel_size:
    _ATTN_TP = _TP
else:
    group_ranks = []
    for tp_group_idx in range(num_tensor_model_parallel_groups):
        for cp_dp_combined_idx in range(attn_cp_size * attn_dp_size):
            st = (
                tp_group_idx * tensor_model_parallel_size
                + cp_dp_combined_idx * attn_tp_size
            )
            en = (
                tp_group_idx * tensor_model_parallel_size
                + (cp_dp_combined_idx + 1) * attn_tp_size
            )
            ranks = list(range(st, en))
            group_ranks.append(ranks)

    _ATTN_TP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_pynccl=SYNC_TOKEN_IDS_ACROSS_TP or enable_symm_mem,
        use_mscclpp_allreduce=False,
        use_custom_allreduce=False,
        use_torch_symm_mem_allreduce=False,
        use_message_queue_broadcaster=envs.SGLANG_USE_MESSAGE_QUEUE_BROADCASTER.get(),
        group_name="attention_tp",
        recovered_rank=recovered_rank,
    )
```

代码逻辑：
- 从 attention DP、CP 反推 attention TP size。
- attention CP 等价于完整 TP 时复用 `_TP`。
- 否则按 stride 生成 CP group。
- attention TP 等价于完整 TP 时复用 `_TP`。
- 否则按连续 rank 生成 attention TP group。

为什么这样写：
- DP Attention 需要让 attention collective 和 MoE/普通 TP collective 有不同通信边界。
- 复用 `_TP` 可以避免等价配置下创建重复 group。

不变量与失败模式：
- `_ATTN_CP` 和 `_ATTN_TP` 都只能初始化一次。
- `tensor_model_parallel_size` 必须能被 attention CP 和 DP 的组合整除。
- attention TP group 禁用部分优化，意味着性能策略和普通 TP group 不完全一致。

Comment：
这段解释了为什么 `communication_op.py` 要有 attention 专用 all-reduce API。

### 1.4 GroupCoordinator 绑定 rank、device、device group 与 CPU group

问题与约束：
- 同一组 rank 既需要 GPU/NPU/XPU/MUSA collective，也需要 CPU 侧协调通道；容器里每进程可能只可见一张 GPU。

设计选择：
- `GroupCoordinator.__init__` 先记录全局 rank、local rank、local size；按平台选择 device；为每组 rank 创建 device group 和 gloo CPU group，并在当前 rank 属于该组时记录 group 内 rank 信息。

Explain：
CUDA-like 平台在 `SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS` 开启时固定使用 `cuda:0`，否则按 local rank 选卡。子组 timeout 使用前面保存的 `_MODEL_PARALLEL_GROUP_TIMEOUT`。

来源：python/sglang/srt/distributed/parallel_state.py L270-L343

Code：

```python
group_name = group_name or "anonymous"
self.unique_name = _get_unique_name(group_name)
_register_group(self)

self.rank = torch.distributed.get_rank()
self.local_rank = local_rank
self.device_group = None
self.cpu_group = None
self.local_size = get_int_env_var("LOCAL_SIZE", 0)

if is_cuda_alike():
    device_id = (
        0 if envs.SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS.get() else local_rank
    )
    self.device = torch.device(f"cuda:{device_id}")
elif _is_npu:
    self.device = torch.device(f"npu:{local_rank}")
elif _is_xpu:
    self.device = torch.device(f"xpu:{local_rank}")
elif _is_musa:
    self.device = torch.device(f"musa:{local_rank}")
else:
    self.device = torch.device("cpu")
self.device_module = torch.get_device_module(self.device)

for ranks in group_ranks:
    active_ranks = torch.ones(len(ranks), dtype=torch.int32, device=self.device)
    active_ranks_cpu = torch.ones(len(ranks), dtype=torch.int32)
    subgroup_timeout = _MODEL_PARALLEL_GROUP_TIMEOUT
    device_group = torch.distributed.new_group(
        ranks,
        backend=torch_distributed_backend,
        pg_options=get_torch_distributed_pg_options(group_name),
        timeout=subgroup_timeout,
    )
    cpu_group = torch.distributed.new_group(
        ranks, backend="gloo", timeout=gloo_timeout
    )
    if self.rank in ranks:
        self.ranks = ranks
        self.world_size = len(ranks)
        self.rank_in_group = ranks.index(self.rank)
        self.device_group = device_group
        self.cpu_group = cpu_group
        self.active_ranks = active_ranks
        self.active_ranks_cpu = active_ranks_cpu

assert self.cpu_group is not None
assert self.device_group is not None
```

代码逻辑：
- 为 group 生成唯一名称并注册弱引用。
- 根据平台和 local rank 选择设备。
- 对每个 rank 列表创建 device group 与 CPU group。
- 当前 rank 命中的 group 会成为本 coordinator 的 active group。
- 初始化结束时要求两类 group 都存在。

为什么这样写：
- collective 走 device group，metadata/object 协调走 CPU group。
- weakref 注册让 torch custom op 能用 group name 找回 coordinator。

不变量与失败模式：
- 当前 rank 必须属于传入的某个 group，否则 assert 失败。
- local rank 与可见 device 映射必须正确，否则 device collective 会落错卡。
- CPU group 创建失败会影响对象广播和部分控制面同步。

Comment：
`GroupCoordinator` 是 SGLang 分布式层的中心对象：它把 rank 拓扑、设备和通信后端收束到一个接口。

---

## 2. GroupCoordinator 的 communicator 策略

### 2.1 初始化时按平台和开关挂载 communicator

问题与约束：
- 不同硬件和 workload 对 all-reduce 的最佳实现不同；单一 NCCL fallback 不能覆盖 CPU、HPU、XPU、NPU、ROCm、PyMSCCLPP、symmetric memory 等路径。

设计选择：
- GroupCoordinator 初始化时根据开关懒加载 PyNCCL、PyMSCCLPP、custom all-reduce、torch symmetric memory、HPU/XPU/NPU communicator；message queue broadcaster 只在 world_size 大于 1 且未 recovered rank 时创建。

Explain：
custom all-reduce 创建失败只打 warning 并建议显式禁用；message queue broadcaster 使用 CPU group 创建，参数是 chunk 大小 `1 << 22` 和 chunk 数 `6`。

来源：python/sglang/srt/distributed/parallel_state.py L345-L473

Code：

```python
self.use_pynccl = use_pynccl
self.use_pymscclpp = use_pymscclpp
self.use_custom_allreduce = use_custom_allreduce
self.use_torch_symm_mem_all_reduce = use_torch_symm_mem_all_reduce
self.use_hpu_communicator = use_hpu_communicator
self.use_xpu_communicator = use_xpu_communicator
self.use_npu_communicator = use_npu_communicator
self.use_message_queue_broadcaster = use_message_queue_broadcaster

self.pynccl_comm: Optional[PyNcclCommunicator] = None
if use_pynccl and self.world_size > 1:
    self.pynccl_comm = PyNcclCommunicator(
        group=self.cpu_group,
        device=self.device,
    )

self.pymscclpp_comm: Optional[PyMscclppCommunicator] = None
if use_pymscclpp and self.world_size > 1:
    self.pymscclpp_comm = PyMscclppCommunicator(
        group=self.cpu_group,
        device=self.device,
    )

self.ca_comm: Optional[Any] = None
self.qr_comm: Optional[QuickAllReduce] = None
if use_custom_allreduce and self.world_size > 1:
    try:
        CAClass = dispatch_custom_allreduce(
            group=self.cpu_group,
            device=self.device,
        )
        self.ca_comm = CAClass(
            group=self.cpu_group,
            device=self.device,
        )
    except Exception as e:
        logger.warning(
            f"Setup Custom allreduce failed with {e}. To silence this "
            "warning, specify --disable-custom-all-reduce explicitly."
        )

self.mq_broadcaster: Optional[MessageQueue] = None
if use_message_queue_broadcaster and self.world_size > 1 and not recovered_rank:
    self.mq_broadcaster = MessageQueue.create_from_process_group(
        self.cpu_group, 1 << 22, 6
    )
```

代码逻辑：
- 保存所有 communicator 开关。
- world size 大于 1 时才创建通信优化对象。
- custom all-reduce 失败不阻断 coordinator 初始化。
- message queue broadcaster 绑定 CPU group。

为什么这样写：
- 分布式通信优化是可选增强，失败时应回退到 torch collective。
- broadcaster 面向对象和小消息，不应依赖 device collective。

不变量与失败模式：
- 单卡时各 communicator 大多保持 `None`。
- custom all-reduce 异常会退化性能但不阻断启动。
- recovered rank 的 MQ broadcaster 由 elastic 路径单独处理。

Comment：
SGLang 把“可用性优先、优化可回退”的策略放在 communicator 初始化阶段。

### 2.2 all_reduce 运行时选择 communicator 或 custom op fallback

问题与约束：
- 模型层希望只调用一个 all-reduce；底层需要根据 tensor 位置、硬件、图捕获状态和 communicator 能力选择具体实现。

设计选择：
- `GroupCoordinator.all_reduce` 先处理 world size 1 和 CPU tensor，再依次尝试 HPU/XPU/NPU、symmetric memory PyNCCL、custom AR、QuickAllReduce、PyMSCCLPP、torch symmetric memory、piecewise graph PyNCCL；没有 outplace 方法时调用 `inplace_all_reduce` custom op。

Explain：
注释说明 Dynamo 不能直接把 coordinator 对象传给 custom op，所以 custom op 使用 group name 字符串查回 group。

来源：python/sglang/srt/distributed/parallel_state.py L579-L661

Code：

```python
def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
    if self.world_size == 1:
        return input_

    if input_.is_cpu:
        if is_shm_available(input_.dtype, self.world_size, self.local_size):
            torch.ops.sgl_kernel.shm_allreduce(input_, REDUCE_OP_SUM)
        else:
            torch.distributed.all_reduce(input_, group=self.device_group)
        return input_

    if self.hpu_communicator is not None and not self.hpu_communicator.disabled:
        return self.hpu_communicator.all_reduce(input_)

    if self.xpu_communicator is not None and not self.xpu_communicator.disabled:
        return self.xpu_communicator.all_reduce(input_)

    if self.npu_communicator is not None and not self.npu_communicator.disabled:
        return self.npu_communicator.all_reduce(input_)

    should_use_pymscclpp_allreduce = (
        self.pymscclpp_comm is not None
        and self.pymscclpp_comm.should_mscclpp_allreduce(input_)
    )
    if (
        self.pynccl_comm is not None
        and self.is_symmetric_memory_enabled()
        and not should_use_pymscclpp_allreduce
    ):
        self.debug_check_symmetric_mempool(self, {"input": input_}, "all_reduce")
        with self.pynccl_comm.change_state(enable=True):
            self.pynccl_comm.all_reduce(input_)
            return input_

    outplace_all_reduce_method = None
    if (
        self.ca_comm is not None
        and not self.ca_comm.disabled
        and not should_use_pymscclpp_allreduce
        and self.ca_comm.should_custom_ar(input_)
    ):
        outplace_all_reduce_method = "ca"
    elif (
        self.qr_comm is not None
        and not self.qr_comm.disabled
        and self.qr_comm.should_quick_allreduce(input_)
    ):
        outplace_all_reduce_method = "qr"
    elif self.pymscclpp_comm is not None and should_use_pymscclpp_allreduce:
        outplace_all_reduce_method = "pymscclpp"
    elif (
        self.torch_symm_mem_comm is not None
        and not self.torch_symm_mem_comm.disabled
        and self.torch_symm_mem_comm.should_torch_symm_mem_allreduce(input_)
    ):
        outplace_all_reduce_method = "torch_symm_mem"
    elif is_in_tc_piecewise_cuda_graph() and self.pynccl_comm is not None:
        outplace_all_reduce_method = "pynccl"
    if outplace_all_reduce_method is not None:
        return outplace_all_reduce(
            input_,
            group_name=self.unique_name,
            outplace_all_reduce_method=outplace_all_reduce_method,
        )
    else:
        inplace_all_reduce(input_, group_name=self.unique_name)
        return input_
```

代码逻辑：
- 单卡直接返回输入。
- CPU tensor 优先共享内存 all-reduce，否则 torch all_reduce。
- 专用硬件 communicator 优先。
- symmetric memory 和 PyMSCCLPP 互斥选择。
- 根据 communicator predicate 选择 outplace custom op。
- 无可用 outplace 方法时走 inplace custom op。

为什么这样写：
- all-reduce 是 decode 热点，必须让不同 backend 有机会接管。
- custom op 形式兼容 torch.compile/CUDA graph 对 Python 对象捕获的限制。

不变量与失败模式：
- communicator 的 `disabled` 和 `should_*` predicate 共同决定是否可用。
- 如果 custom op 注册或 group weakref 失效，fallback custom op 会失败。
- CPU tensor 使用 `device_group` 调 torch all_reduce 的路径依赖 backend 对 CPU tensor 的支持。

Comment：
模型层看到的是统一 `all_reduce`，但实际执行路径是高度条件化的。

### 2.3 fused_allreduce_rmsnorm 只在 custom AR 能力存在时返回结果

问题与约束：
- fused all-reduce + RMSNorm 是 decode 热点优化，但不是所有 custom all-reduce communicator 都支持 fused API。

设计选择：
- 如果 `ca_comm` 不存在或 disabled，直接返回 `None`；优先调用 communicator-native `fused_allreduce_rmsnorm`，失败后尝试 `custom_fused_ar_rms`；仍不具备能力时返回 `None`。

Explain：
函数根据环境变量或输入 tensor 字节数选择 1-stage/2-stage fused kernel，并在 piecewise CUDA graph capture 场景走 `fused_ar_rms` 分支。

来源：python/sglang/srt/distributed/parallel_state.py L677-L736

Code：

```python
def fused_allreduce_rmsnorm(
    self,
    input_: torch.Tensor,
    residual_inp_: torch.Tensor,
    weight_: torch.Tensor,
    eps: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    ca_comm = self.ca_comm
    if ca_comm is None or getattr(ca_comm, "disabled", True):
        return None

    if hasattr(ca_comm, "fused_allreduce_rmsnorm"):
        try:
            return ca_comm.fused_allreduce_rmsnorm(
                input_, residual_inp_, weight_, eps
            )
        except Exception:
            pass

    if not hasattr(ca_comm, "custom_fused_ar_rms"):
        return None

    if envs.SGLANG_USE_1STAGE_ALLREDUCE.is_set():
        use_1stage_ar = envs.SGLANG_USE_1STAGE_ALLREDUCE.get()
    else:
        total_bytes = input_.numel() * input_.element_size()
        use_1stage_ar = total_bytes <= 128 * 1024

    if (
        getattr(ca_comm, "_IS_CAPTURING", False)
        and not torch.cuda.is_current_stream_capturing()
        and is_in_tc_piecewise_cuda_graph()
    ):
        if not hasattr(ca_comm, "fused_ar_rms"):
            return None
        return ca_comm.fused_ar_rms(
            input_,
            residual_inp_,
            w=weight_,
            eps=eps,
            registered=False,
            use_1stage=use_1stage_ar,
        )
    fused_outputs = ca_comm.custom_fused_ar_rms(
        input_,
        residual_inp_,
        weight_,
        eps,
        use_1stage_ar,
    )
    return fused_outputs
```

代码逻辑：
- 检查 custom all-reduce communicator 是否存在且启用。
- 优先使用 communicator 原生 fused API。
- 原生 fused 失败后尝试通用 custom fused path。
- 根据环境变量或输入大小选择 stage。
- piecewise graph capture 下使用 graph 兼容 API。
- 成功返回 fused 输出，不能处理时返回 `None`。

为什么这样写：
- fused 优化收益高，但必须可回退，让 caller 能走分离 all-reduce + RMSNorm。
- 输入大小决定 stage，避免大 prefill batch 命中不适合的一阶段 kernel。

不变量与失败模式：
- 返回 `None` 不是异常，而是 caller fallback 信号。
- 原生 fused API 的异常被吞掉并尝试 fallback，可能隐藏性能退化。
- 只有 custom AR communicator 参与该 fused 路径。

Comment：
这段体现了 SGLang 对热点 kernel 的设计：能力检测优先，失败回退为正常控制流。

### 2.4 custom op 通过 group name 查回弱引用 coordinator

问题与约束：
- torch.compile 和自定义 op 不适合直接捕获 Python coordinator 对象，但 collective 仍需要定位具体 group。

设计选择：
- 注册 custom op `reg_all_to_all_single`，参数只带 tensor 和 `group_name`；运行时从 `_groups` 弱引用表取 coordinator，再调用 group 内部实现。

Explain：
如果 group name 不存在会 assert；如果 weakref 已失效，抛 `ValueError`，避免在销毁后的 group 上继续 collective。

来源：python/sglang/srt/distributed/parallel_state.py L205-L213

Code：

```python
@register_custom_op(mutates_args=["output"])
def reg_all_to_all_single(
    output: torch.Tensor, input: torch.Tensor, group_name: str
) -> None:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    group._all_to_all_single(output, input)
```

代码逻辑：
- custom op 声明会修改 `output`。
- 用 `group_name` 查 `_groups`。
- weakref 失效时显式报错。
- 调用 coordinator 的私有 collective 实现。

为什么这样写：
- custom op 可以进入编译和图捕获路径。
- group name 是可序列化、可追踪的轻量句柄。

不变量与失败模式：
- group 必须先注册到 `_groups`。
- group 生命周期结束后，旧 graph 或旧调用继续使用 group name 会失败。

Comment：
这也是 all-reduce 等 public API 只暴露薄函数、实际由 coordinator 分发的原因。

---

## 3. 对模型层暴露的通信 API

### 3.1 communication_op.py 是模型层的薄 facade

问题与约束：
- 模型代码不应关心当前 TP group 的具体 communicator，也不应直接引用 `GroupCoordinator`。

设计选择：
- `communication_op.py` 暴露 `tensor_model_parallel_*` 函数，内部只调用 `get_tp_group()` 再转发到 coordinator。

Explain：
普通 all-reduce、quant all-reduce、fused all-reduce + RMSNorm、all-gather、gather、tensor dict broadcast 都通过这个 facade 进入 distributed 层。

来源：python/sglang/srt/distributed/communication_op.py L18-L62

Code：

```python
def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_quant_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    return get_tp_group().quant_all_reduce(input_)


def tensor_model_parallel_fused_allreduce_rmsnorm(
    input_: torch.Tensor,
    residual_inp_: torch.Tensor,
    weight_: torch.Tensor,
    eps: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    return get_tp_group().fused_allreduce_rmsnorm(input_, residual_inp_, weight_, eps)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> Optional[torch.Tensor]:
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
```

代码逻辑：
- 每个函数先定位当前语义的 group。
- tensor 操作全部委托给 coordinator。
- `broadcast_tensor_dict` 在 torch.distributed 未初始化时直接返回输入。

为什么这样写：
- 模型层只依赖稳定 API 名称，通信策略可在 distributed 层演进。
- 未初始化分布式时让 broadcast dict 成为 no-op，便于单进程路径复用。

不变量与失败模式：
- 调用 TP collective 前必须已经初始化 `_TP`。
- fused API 可能返回 `None`，caller 必须实现 fallback。

Comment：
这层 facade 是模型执行代码和复杂 communicator 策略之间的隔离带。

### 3.2 attention 与 MoE collective 使用专用 group

问题与约束：
- DP Attention 和 MoE expert parallelism 下，同一 tensor 的同步范围不总是普通 TP group。

设计选择：
- 提供 attention TP、MoE TP、MoE EP 三类专用 all-reduce API，分别转发到 `get_attn_tp_group()`、`get_moe_tp_group()`、`get_moe_ep_group()`。

Explain：
attention quant all-reduce 也走 attention TP group；MoE expert 输出同步可走 expert parallel group。

来源：python/sglang/srt/distributed/communication_op.py L65-L84

Code：

```python
def attention_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    return get_attn_tp_group().all_reduce(input_)


def attention_tensor_model_parallel_quant_all_reduce(
    input_: torch.Tensor,
) -> torch.Tensor:
    return get_attn_tp_group().quant_all_reduce(input_)


def moe_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    return get_moe_tp_group().all_reduce(input_)


def moe_expert_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    return get_moe_ep_group().all_reduce(input_)
```

代码逻辑：
- attention 路径选择 attention TP group。
- attention quant 路径选择 attention TP group 的 quant all-reduce。
- MoE TP 路径选择 MoE TP group。
- expert 输出同步选择 MoE EP group。

为什么这样写：
- Attention 和 MoE 的并行维度可独立于普通 TP。
- 显式 API 名称让模型层调用点表达通信语义，而不是只传 group 参数。

不变量与失败模式：
- 对应 group 必须在 `initialize_model_parallel` 中初始化。
- 错用普通 TP API 会扩大或缩小同步范围，导致数值错误。

Comment：
这些函数是阅读 DP Attention 与 MoE forward 时定位通信边界的入口。

### 3.3 broadcast_tensor_dict 拆分 metadata 和 tensor

问题与约束：
- Python dict 中可能混合 tensor 与普通对象；metadata 适合 CPU object broadcast，tensor 需要按设备使用 collective。

设计选择：
- `_split_tensor_dict` 将 tensor value 替换成 `TensorMetadata(device, dtype, size)`，同时把原 tensor 放入 tensor list；非 tensor value 原样留在 metadata list。

Explain：
代码只记录 device type，而不是 `cuda:0` 这样的具体 index，因为接收端会在自己的 rank 上设置 device index。

来源：python/sglang/srt/distributed/parallel_state.py L113-L136

Code：

```python
def _split_tensor_dict(
    tensor_dict: Dict[str, Union[torch.Tensor, Any]],
) -> Tuple[List[Tuple[str, Any]], List[torch.Tensor]]:
    metadata_list: List[Tuple[str, Any]] = []
    tensor_list: List[torch.Tensor] = []
    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            device = value.device.type
            metadata_list.append(
                (key, TensorMetadata(device, value.dtype, value.size()))
            )
            tensor_list.append(value)
        else:
            metadata_list.append((key, value))
    return metadata_list, tensor_list
```

代码逻辑：
- 遍历 dict key/value。
- tensor value 转成 metadata，并追加原 tensor。
- 非 tensor value 保持在 metadata list。
- 返回 metadata list 与 tensor list 两部分。

为什么这样写：
- metadata 需要 pickle/object broadcast，tensor 需要 device-aware broadcast。
- 去掉 device index 可避免接收端误用发送端 GPU 编号。

不变量与失败模式：
- 接收端必须用 metadata 顺序重建 dict。
- 如果 value 是 tensor-like 但不是 `torch.Tensor`，会按普通对象处理。

Comment：
这段是 tensor dict 广播的序列化边界。

### 3.4 broadcast_tensor_dict 用 CPU group 广播 metadata，用 device group 广播 tensor

问题与约束：
- 对象 metadata 和 tensor payload 的传输通道不同；CPU tensor 与 GPU tensor 也要走不同 group。

设计选择：
- source rank 先 `_split_tensor_dict`，通过 `broadcast_object` 广播 metadata；随后对 tensor list 逐个异步 broadcast。接收端先拿 metadata，再按 metadata 创建空 tensor 并接收 payload。

Explain：
函数在 torch.distributed 未初始化或 world size 为 1 时直接返回输入；source rank 的 `src` 是 group 内 local rank。

来源：python/sglang/srt/distributed/parallel_state.py L1351-L1431

Code：

```python
def broadcast_tensor_dict(
    self,
    tensor_dict: Optional[Dict[str, Union[torch.Tensor, Any]]] = None,
    src: int = 0,
    group: Optional[ProcessGroup] = None,
    metadata_group: Optional[ProcessGroup] = None,
) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:
    if not torch.distributed.is_initialized() or self.world_size == 1:
        return tensor_dict

    group = self.device_group
    metadata_group = self.cpu_group
    assert src < self.world_size, f"Invalid src rank ({src})"

    rank_in_group = self.rank_in_group
    if rank_in_group == src:
        metadata_list: List[Tuple[Any, Any]] = []
        assert isinstance(
            tensor_dict, dict
        ), f"Expecting a dictionary, got {type(tensor_dict)}"
        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        self.broadcast_object(metadata_list, src=src)
        async_handles = []
        for tensor in tensor_list:
            if tensor.numel() == 0:
                continue
            if tensor.is_cpu:
                handle = torch.distributed.broadcast(
                    tensor, src=self.ranks[src], group=metadata_group, async_op=True
                )
            else:
                handle = torch.distributed.broadcast(
                    tensor, src=self.ranks[src], group=group, async_op=True
                )
            async_handles.append(handle)
        for async_handle in async_handles:
            async_handle.wait()

    else:
        metadata_list = self.broadcast_object(None, src=src)
        tensor_dict = {}
        async_handles = []
        for key, value in metadata_list:
            if isinstance(value, TensorMetadata):
                tensor = torch.empty(
                    value.size, dtype=value.dtype, device=value.device
                )
                if tensor.numel() == 0:
                    tensor_dict[key] = tensor
                    continue
                if tensor.is_cpu:
                    handle = torch.distributed.broadcast(
                        tensor,
                        src=self.ranks[src],
                        group=metadata_group,
                        async_op=True,
                    )
                else:
                    handle = torch.distributed.broadcast(
                        tensor, src=self.ranks[src], group=group, async_op=True
                    )
                async_handles.append(handle)
                tensor_dict[key] = tensor
            else:
                tensor_dict[key] = value
        for async_handle in async_handles:
            async_handle.wait()
    return tensor_dict
```

代码逻辑：
- 单进程或未初始化分布式时 no-op。
- source rank 拆分 metadata 和 tensor。
- metadata 经 `broadcast_object` 走 CPU group。
- tensor payload 按 CPU/GPU 选择 metadata group 或 device group。
- 接收端用 metadata 创建空 tensor 并等待异步 broadcast 完成。

为什么这样写：
- 让 Python 对象和 tensor payload 各走适合的通道。
- 异步广播多个 tensor 后统一 wait，可减少串行等待。

不变量与失败模式：
- `src` 是 group 内 rank，必须小于 `world_size`。
- source rank 的输入必须是 dict。
- metadata 和 tensor list 顺序必须保持一致。

Comment：
权重同步和控制信息广播经常会用到这类 mixed dict，这段是高频辅助路径。

### 3.5 shm_broadcast 对小对象走共享内存，大对象回退 socket

问题与约束：
- 小对象 metadata/control 消息用 socket 全量传输会增加延迟；但共享内存 ring buffer 有 chunk 大小限制。

设计选择：
- `enqueue` 先 pickle 对象；本地 reader 存在时，若序列化结果超过 chunk 上限，写 overflow 标记并通过 local socket 发送完整对象，否则写入共享内存 buffer；远端 reader 始终通过 remote socket 发送。

Explain：
`dequeue` 对本地 reader 读取 overflow 标记：非 overflow 从共享内存反序列化，overflow 从 local socket 收；远端 reader 从 remote socket 收。

来源：python/sglang/srt/distributed/device_communicators/shm_broadcast.py L444-L476

Code：

```python
def enqueue(self, obj):
    assert self._is_writer, "Only writers can enqueue"
    serialized_obj = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    if self.n_local_reader > 0:
        if len(serialized_obj) >= self.buffer.max_chunk_bytes:
            with self.acquire_write() as buf:
                buf[0] = 1
            self.local_socket.send(serialized_obj)
        else:
            with self.acquire_write() as buf:
                buf[0] = 0
                buf[1 : len(serialized_obj) + 1] = serialized_obj
    if self.n_remote_reader > 0:
        self.remote_socket.send(serialized_obj)

def dequeue(self):
    if self._is_local_reader:
        with self.acquire_read() as buf:
            overflow = buf[0] == 1
            if not overflow:
                obj = pickle.loads(buf[1:])
        if overflow:
            recv = self.local_socket.recv()
            obj = pickle.loads(recv)
    elif self._is_remote_reader:
        recv = self.remote_socket.recv()
        obj = pickle.loads(recv)
    else:
        raise RuntimeError("Only readers can dequeue")
    return obj
```

代码逻辑：
- writer 才能 enqueue。
- 对象先 pickle。
- 本地小对象写共享内存，大对象写 overflow 标记并走 local socket。
- 远端读者走 remote socket。
- reader 根据本地/远端角色选择读取路径。

为什么这样写：
- 本机小消息走共享内存可降低控制面延迟。
- 大对象回退 socket，避免超过 ring buffer chunk 容量。

不变量与失败模式：
- 非 writer 调 enqueue 会 assert。
- 非 reader 调 dequeue 会抛 RuntimeError。
- pickle 反序列化依赖发送端和接收端代码/类型兼容。

Comment：
MQ broadcaster 是 GroupCoordinator 针对小对象广播的补充优化，不替代 tensor collective。

---

## 4. DataParallelController 的进程与路由

### 4.1 LoadBalanceMethod 与 DPBudget 支持多种 dispatch 策略

问题与约束：
- 多 DP worker 场景下，请求既可能需要轮询，也可能需要按 bootstrap room 保持 locality，或根据实时负载选择 worker。

设计选择：
- `LoadBalanceMethod` 枚举支持 round-robin、bootstrap room、total requests、total tokens；`DPBudget.dispatch` 对负载型策略选择最小请求数或最小 token 数的 DP rank，并做启发式增量。

Explain：
`TOTAL_TOKENS` 在 token 数相同时用 total requests 作为 tie-breaker，避免多个 worker token 相同但请求堆积不同。

来源：python/sglang/srt/managers/data_parallel_controller.py L76-L125

Code：

```python
class LoadBalanceMethod(Enum):
    ROUND_ROBIN = auto()
    FOLLOW_BOOTSTRAP_ROOM = auto()
    TOTAL_REQUESTS = auto()
    TOTAL_TOKENS = auto()

    @classmethod
    def from_str(cls, method: str):
        method = method.upper()
        try:
            return cls[method]
        except KeyError as exc:
            raise ValueError(f"Invalid load balance method: {method}") from exc


class DPBudget:
    def __init__(self, dp_size: int):
        self.dp_size = dp_size
        self.total_requests = [0] * dp_size
        self.total_tokens = [0] * dp_size
        self.last_timestamp = [0.0] * dp_size

    def dispatch(self, method: LoadBalanceMethod, estimated_tokens: int = 0):
        if method == LoadBalanceMethod.TOTAL_REQUESTS:
            target_rank = self.total_requests.index(min(self.total_requests))
        elif method == LoadBalanceMethod.TOTAL_TOKENS:
            target_rank = min(
                range(self.dp_size),
                key=lambda i: (self.total_tokens[i], self.total_requests[i]),
            )
        else:
            return None

        self.total_requests[target_rank] += 1
        self.total_tokens[target_rank] += estimated_tokens
```

代码逻辑：
- 字符串配置转成 enum，非法值直接 ValueError。
- DPBudget 维护每个 DP rank 的请求数、token 数和快照时间戳。
- total requests 策略选择请求数最小 rank。
- total tokens 策略选择 token 数最小 rank，并用请求数打破平局。
- 选择后立即增加估计负载。

为什么这样写：
- 负载快照有采样延迟，dispatch 后做启发式增量能减少连续请求打到同一 worker。
- bootstrap room 和 round-robin 不需要 DPBudget 参与。

不变量与失败模式：
- method 字符串必须能匹配枚举名。
- 负载型策略依赖 load snapshot 更新，快照过旧会影响调度质量。

Comment：
DP Controller 的负载均衡不是单一轮询，而是按部署模式选择策略。

### 4.2 DataParallelController 初始化 ZMQ 与 dispatch 函数

问题与约束：
- TokenizerManager 只应连接一个 controller 入口；controller 内部再分发到多个 DP Scheduler。

设计选择：
- Controller 创建 `zmq.Context(1 + dp_size)`；node 0 绑定从 TokenizerManager 接收的 PULL socket；根据 `server_args.load_balance_method` 选择 dispatch 函数，并创建 DPBudget 与 load snapshot reader。

Explain：
`refresh_load_budget_on_dispatch` 只对 total requests 和 total tokens 两类负载感知策略开启。

来源：python/sglang/srt/managers/data_parallel_controller.py L129-L170

Code：

```python
class DataParallelController:
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        run_scheduler_process_func: Callable,
    ) -> None:
        self.server_args = server_args
        self.port_args = port_args
        self.load_balance_method = LoadBalanceMethod.from_str(
            server_args.load_balance_method
        )
        self.run_scheduler_process_func = run_scheduler_process_func

        self.context = zmq.Context(1 + server_args.dp_size)
        if server_args.node_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                self.context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )

        self.round_robin_counter = 0
        dispatch_lookup = {
            LoadBalanceMethod.ROUND_ROBIN: self.round_robin_scheduler,
            LoadBalanceMethod.FOLLOW_BOOTSTRAP_ROOM: self.follow_bootstrap_room_scheduler,
            LoadBalanceMethod.TOTAL_REQUESTS: self.total_requests_scheduler,
            LoadBalanceMethod.TOTAL_TOKENS: self.total_tokens_scheduler,
        }
        self.dispatching = dispatch_lookup[self.load_balance_method]
        self.refresh_load_budget_on_dispatch = self.load_balance_method in (
            LoadBalanceMethod.TOTAL_REQUESTS,
            LoadBalanceMethod.TOTAL_TOKENS,
        )

        self.dp_budget = DPBudget(server_args.dp_size)
        self.load_snapshot_reader = create_load_snapshot_reader(
            server_args,
```

代码逻辑：
- 保存 server/port args。
- 解析 load balance method。
- 创建 ZMQ context。
- node 0 绑定 tokenizer 到 controller 的入口 socket。
- 用 enum 查 dispatch 函数。
- 判断是否需要在 dispatch 时刷新负载预算。

为什么这样写：
- DP Controller 是多 Scheduler 前面的单点路由器。
- dispatch 函数在初始化时选好，event loop 中就不必重复分支判断。

不变量与失败模式：
- `server_args.load_balance_method` 必须合法。
- 非 node 0 不绑定 tokenizer 入口，跨节点部署依赖其他连接逻辑。

Comment：
Controller 初始化把“怎么收请求”和“怎么选 DP worker”都定下来。

### 4.3 Controller 为每个 DP/TP rank 启动 Scheduler 子进程

问题与约束：
- DP worker 不是一个线程内的对象，而是多个 Scheduler 子进程；每个进程要拿到自己的 GPU、TP rank、attention/MoE rank 和 IPC writer。

设计选择：
- Controller 计算各 rank 派生信息，在 `maybe_reindex_device_id` 和子进程配置上下文中启动 `mp.Process(target=run_scheduler_process_func, args=(...))`，然后等待每个 Scheduler 通过 pipe 返回模型加载信息。

Explain：
Controller 从所有 Scheduler pipe reader 收集 `max_total_num_tokens` 与 `max_req_input_len`，取第一个作为 controller 对外 ready 信息。

来源：python/sglang/srt/managers/data_parallel_controller.py L560-L604

Code：

```python
with self.env_lock, maybe_reindex_device_id(gpu_id) as gpu_id:
    proc = mp.Process(
        target=self.run_scheduler_process_func,
        args=(
            server_args,
            rank_port_args,
            gpu_id,
            tp_rank,
            attn_cp_rank,
            moe_dp_rank,
            moe_ep_rank,
            pp_rank,
            dp_rank,
            writer,
        ),
    )
    with (
        memory_saver_adapter.configure_subprocess(),
        numa_utils.configure_subprocess(server_args, gpu_id),
    ):
        proc.start()
self.scheduler_procs.append(proc)
scheduler_pipe_readers.append(reader)

scheduler_info = []
for i in range(len(scheduler_pipe_readers)):
    scheduler_info.append(scheduler_pipe_readers[i].recv())

self.max_total_num_tokens = scheduler_info[0]["max_total_num_tokens"]
self.max_req_input_len = scheduler_info[0]["max_req_input_len"]
```

代码逻辑：
- 为每个 Scheduler 进程准备 rank 和端口参数。
- 在 GPU reindex、memory saver、NUMA 配置上下文中启动进程。
- 保存子进程对象和 pipe reader。
- 等待所有 Scheduler 返回初始化信息。
- 将第一个 Scheduler 的容量信息保存到 controller。

为什么这样写：
- Scheduler 进程需要独立 CUDA context 和 rank 身份。
- controller 只有等模型加载完成后才能向父进程汇报 ready。

不变量与失败模式：
- 任一 Scheduler 没有通过 pipe 返回，controller 初始化会阻塞。
- rank 派生公式必须与模型并行 group 切分一致。

Comment：
DP Controller 的“worker”本质上是 Scheduler 进程，不是轻量协程。

### 4.4 Controller dispatch 支持显式路由、bootstrap room 和负载感知策略

问题与约束：
- 一些请求已经带有目标 DP rank；PD 场景需要同一 bootstrap room 命中固定 worker；普通场景则可以按负载选 worker。

设计选择：
- `maybe_external_dp_rank_routing` 优先处理 `req.routed_dp_rank`；bootstrap room 策略用 `bootstrap_room % len(workers)`；负载感知策略调用 DPBudget 并发送到目标 worker socket。

Explain：
`total_tokens_scheduler` 用输入 token 长度作为估计 token 增量，更新 DPBudget 的启发式负载。

来源：python/sglang/srt/managers/data_parallel_controller.py L605-L652

Code：

```python
def maybe_external_dp_rank_routing(self, req: Req):
    if req.routed_dp_rank is not None:
        logger.debug(f"Direct routing to DP rank {req.routed_dp_rank}")
        sock_send(self.workers[req.routed_dp_rank], req)
        return True
    return False

def follow_bootstrap_room_scheduler(self, req: Req):
    if self.maybe_external_dp_rank_routing(req):
        return

    assert req.bootstrap_room is not None, (
        "req.bootstrap_room should not be None. Do not send requests directly to "
        "prefill or decode instances; send to the router instead."
    )
    target_rank = req.bootstrap_room % len(self.workers)
    sock_send(self.workers[target_rank], req)

def total_requests_scheduler(self, req: Req):
    if self.maybe_external_dp_rank_routing(req):
        return
    target_worker = self.dp_budget.dispatch(LoadBalanceMethod.TOTAL_REQUESTS)
    sock_send(self.workers[target_worker], req)

def total_tokens_scheduler(self, req: Req):
    if self.maybe_external_dp_rank_routing(req):
        return
    estimated_tokens = len(req.input_ids)
    target_worker = self.dp_budget.dispatch(
        LoadBalanceMethod.TOTAL_TOKENS, estimated_tokens=estimated_tokens
    )
    sock_send(self.workers[target_worker], req)
```

代码逻辑：
- 显式 routed DP rank 最高优先级。
- bootstrap room 策略要求请求带 room。
- total requests 策略用 DPBudget 选 worker。
- total tokens 策略传入输入长度估计。
- 所有策略最终通过 ZMQ socket 发给 Scheduler。

为什么这样写：
- 显式路由可支持外部控制和恢复场景。
- bootstrap room 哈希能保持 PD 会话 locality。
- 负载策略适合普通 DP serving。

不变量与失败模式：
- `routed_dp_rank` 必须在 workers 范围内。
- bootstrap room 策略下请求不能缺 `bootstrap_room`。
- DPBudget 返回 `None` 会导致 workers 索引失败，因此只应在负载型策略里调用。

Comment：
读 DP Controller 时要先看 load balance method，否则同一个请求可能走完全不同的路由语义。

### 4.5 Controller event_loop 从 TokenizerManager 拉请求并分发

问题与约束：
- Controller 需要持续从 TokenizerManager 接收请求，同时避免阻塞 watchdog 喂狗和子进程管理。

设计选择：
- `event_loop` 中反复 feed watchdog，并用 non-blocking `sock_recv` 拉取请求；收到请求后交给 `_request_dispatcher`。controller process ready 后，node 0 才进入 event loop，随后 join Scheduler 子进程。

Explain：
`run_data_parallel_controller_process` 初始化 tracing/logging，创建 controller，向父进程发送 ready 信息和 Scheduler PIDs；只有 `server_args.node_rank == 0` 的 controller 负责接收并分发请求。

来源：python/sglang/srt/managers/data_parallel_controller.py L654-L708

Code：

```python
def event_loop(self):
    while True:
        while True:
            self.soft_watchdog.feed()
            try:
                recv_req = sock_recv(self.recv_from_tokenizer, flags=zmq.NOBLOCK)
            except zmq.ZMQError:
                break
            self._request_dispatcher(recv_req)


def run_data_parallel_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
    run_scheduler_process_func: Callable = run_scheduler_process,
):
    setproctitle.setproctitle("sglang::data_parallel_controller")
    faulthandler.enable()
    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    configure_logger(server_args)

    try:
        controller = DataParallelController(
            server_args, port_args, run_scheduler_process_func
        )
        scheduler_pids = [
            proc.pid for proc in controller.scheduler_procs if proc is not None
        ]
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": controller.max_total_num_tokens,
                "max_req_input_len": controller.max_req_input_len,
                SCHEDULER_PIDS_ARG: scheduler_pids,
            }
        )
        if server_args.node_rank == 0:
            controller.event_loop()
        for proc in controller.scheduler_procs:
            proc.join()
```

代码逻辑：
- event loop 持续喂 watchdog。
- 用非阻塞 ZMQ recv drain 当前可读请求。
- 每个请求交给类型分发器。
- process 入口设置进程名、faulthandler、父进程死亡保护和日志。
- controller ready 后向父进程发送容量与 PID 信息。
- node 0 进入主路由循环，最后 join 子进程。

为什么这样写：
- 非阻塞 recv 让 loop 可以周期性执行 watchdog 逻辑。
- 父进程只有收到 ready 信息后才知道 DP Controller 和 Scheduler 都完成初始化。

不变量与失败模式：
- node 0 才有 tokenizer 输入 socket，非 node 0 不应进入分发循环。
- 如果 `_request_dispatcher` 对请求类型没有 handler，请求会在分发层失败。

Comment：
DP Controller 是 SGLang 多 DP serving 的控制面入口：它自己不执行模型 forward，只负责进程管理和请求路由。

---

## 5. 走读小结

```text
init_distributed_environment
  -> _WORLD / timeout / local_rank
  -> initialize_model_parallel
     -> _TP / _ATTN_CP / _ATTN_TP / MoE groups
        -> GroupCoordinator
           -> communicator selection + custom op fallback
           -> communication_op facade for model code

DataParallelController
  -> Scheduler subprocesses
  -> load-balance dispatch
  -> ZMQ route to DP workers
```

**下一专题关联：** MoE EP 细节见 [[18-MoE-00-MOC]]；PD 路由见 [[22-Disaggregation-00-MOC]]。
