---
type: batch-doc
module: 17-Attention
batch: "17"
doc_type: walkthrough
title: "Attention · 源码走读"
tags:
 - sglang/batch/17
 - sglang/module/attention
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-05
---
# Attention · 源码走读

## 走读顺序

1. `base_attn_backend.py` - attention backend 契约与 CUDA Graph metadata 生命周期
2. `triton_backend.py` - Triton attention 的 forward metadata 布局
3. `flashinfer_backend.py` - FlashInfer wrapper、merge state 与层类型分发
4. `radix_attention.py` - 模型层到 backend 的调用边界

---

## 1. Backend 契约把 metadata 拆成 graph 外和 graph 内

### 1.1 `AttentionBackend` 定义三段 forward metadata 初始化协议

问题与约束：
- Attention backend 要同时支持 eager、CUDA Graph capture 和 CUDA Graph replay。
- metadata 准备里既有 host/dynamic-shape 逻辑，也有可录进 graph 的静态 GPU op。
- 旧的 capture/replay 专用 override 容易让 backend 契约分裂。

设计选择：
- 基类只保留三段协议：`init_forward_metadata`、`init_forward_metadata_out_graph`、`init_forward_metadata_in_graph`。
- out-graph 阶段负责 host op、动态 shape 和不可录制逻辑。
- in-graph 阶段只放可录制静态 GPU op。
- 文档明确旧的 capture/replay override 已被移除。

Explain：
`AttentionBackend` 的核心不是规定某个 kernel，而是规定 metadata 在 eager/capture/replay 三种执行模式里的生命周期。这样每个 backend 可以实现自己的 metadata 布局，但必须遵守同一条 graph-safe 边界。

来源：python/sglang/srt/layers/attention/base_attn_backend.py L18-L39

Code：

```python
class AttentionBackend(ABC):
    """The base class of attention backends.

    Forward-data init contract (3 methods):

      - ``init_forward_metadata(fb)`` - eager entry point. Default is a wrapper
        that calls ``_out_graph(fb)`` then ``_in_graph(fb)``.
      - ``init_forward_metadata_out_graph(fb, in_capture=False)`` - per-iter
        metadata prep, runs outside ``with graph.capture():``.
      - ``init_forward_metadata_in_graph(fb)`` - graph-recordable static-shape
        GPU op, runs inside ``with graph.capture():`` at capture time and
        is auto-replayed by ``graph.replay()``.

    The legacy ``init_forward_metadata_capture_cuda_graph`` and
    ``init_forward_metadata_replay_cuda_graph`` overrides are fully
    deprecated and removed from the ABC.
    """
```

代码逻辑：
- 定义 attention backend 抽象基类。
- 在类 docstring 中声明三段 metadata 初始化协议。
- 说明 out-graph 阶段的 capture/replay/eager 调用位置。
- 说明 in-graph 阶段会被 CUDA Graph 记录并自动 replay。
- 明确旧 capture/replay override 的迁移方向。

为什么这样写：
- 把 graph safety 写进基类契约，backend 作者不必从调用点反推规则。
- eager 路径和 graph 路径共享同一组方法，减少实现分叉。
- 删除 legacy override 后，ModelRunner 可以只围绕 out/in graph 两个阶段编排。

不变量与失败模式：
- backend override 必须把 host sync 和动态 shape 逻辑放在 out-graph 阶段。
- in-graph 阶段不应依赖 capture 后会变化的 Python 对象地址。
- out-of-tree backend 若仍实现旧方法，新的基类不会再调用它们。

Comment：
Attention backend 的第一层抽象是 metadata 生命周期，而不是具体 kernel 名称。

### 1.2 `init_forward_metadata` 默认按 out-graph 再 in-graph 执行

问题与约束：
- 非 CUDA Graph 的 eager 推理也需要初始化 attention metadata。
- 默认 eager 路径应复用 graph-safe 拆分，避免 backend 维护第三套逻辑。
- 少数 backend 可能需要完全独立的 eager 实现。

设计选择：
- 基类默认 `init_forward_metadata` 先调用 out-graph，再调用 in-graph。
- 允许子类 override 整个 eager 入口。
- 不在默认实现里区分 capture/replay；该差异由调用点传给 out-graph 阶段。

Explain：
默认 eager 入口把“每步动态准备”和“可录制 GPU 准备”串起来。这样不使用 CUDA Graph 时，也能走与 capture 兼容的 metadata 初始化顺序。

来源：python/sglang/srt/layers/attention/base_attn_backend.py L45-L51

Code：

```python
def init_forward_metadata(self, forward_batch: ForwardBatch):
    """Eager entry point. Default = ``_out_graph(fb) + _in_graph(fb)``.

    Backends may override to keep an independent eager body.
    """
    self.init_forward_metadata_out_graph(forward_batch)
    self.init_forward_metadata_in_graph(forward_batch)
```

代码逻辑：
- 接收当前 `ForwardBatch`。
- 调用 `init_forward_metadata_out_graph` 准备 graph 外 metadata。
- 调用 `init_forward_metadata_in_graph` 执行可录制准备。
- 默认不返回值，metadata 写到 backend 状态或 forward batch 关联对象中。

为什么这样写：
- eager 和 graph capture 的 metadata 语义保持一致。
- backend 若没有特殊需求，只实现 out/in 两个方法即可。
- 子类仍可整体 override，保留对高性能路径的控制权。

不变量与失败模式：
- out-graph 阶段必须先于 in-graph 阶段。
- in-graph 阶段不能依赖 out-graph 尚未填好的 buffer。
- 如果子类 override 整个 eager 入口，需要自行维护与 capture/replay 等价的 metadata 语义。

Comment：
这是 eager 路径的默认 glue code。

### 1.3 `init_forward_metadata_in_graph` 明确禁止 host sync

问题与约束：
- CUDA Graph capture 不能记录 `.item()`、`.cpu()`、`.tolist()` 这类 host 同步。
- 动态 shape 的 allocation 也不能安全进入 captured graph。
- backend 作者需要一个明确的 lint 契约来判断哪些逻辑应移出 graph。

设计选择：
- 在基类方法 docstring 中列出禁用操作。
- 把不可录制逻辑统一要求放到 `init_forward_metadata_out_graph`。
- 默认实现为空，只有需要 graph 内 GPU op 的 backend 才 override。

Explain：
这个方法的存在是为了给 CUDA Graph capture 一个干净边界：它可以在 capture 中被调用，但函数体必须只包含地址稳定、shape 静态、graph-recordable 的 GPU 操作。

来源：python/sglang/srt/layers/attention/base_attn_backend.py L75-L87

Code：

```python
def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
    """Graph-recordable static-shape GPU op.

    Runs inside ``with graph.capture():`` at capture time; recorded
    ops auto-execute at replay via ``graph.replay()``.

    Lint contract for overrides: body must NOT call ``.item()`` /
    ``.cpu()`` / ``.tolist()`` / dynamic-shape ``torch.empty()``.
    Such ops belong in :py:meth:`init_forward_metadata_out_graph`; they
    cannot be recorded into a cuda graph.

    Default: no-op.
    """
```

代码逻辑：
- 声明该方法用于 graph-recordable 静态 GPU op。
- 说明 capture 时执行，replay 时由 graph 自动重放。
- 列出禁止的 host sync 和动态 shape allocation。
- 默认不执行任何操作。

为什么这样写：
- 规则靠近 override 入口，减少 backend 实现时误把 host 逻辑录进 graph。
- 默认 no-op 让简单 backend 不必实现空方法。
- 将不可录制逻辑前移到 out-graph，能保护 CUDA Graph replay 的稳定性。

不变量与失败模式：
- override 不能读取需要 host 同步的 tensor 标量。
- override 不能创建依赖 runtime shape 的新 tensor。
- 违反契约通常会在 capture 阶段失败，或在 replay 时因地址/shape 不稳定产生错误。

Comment：
这段 docstring 实际上是 backend 作者的 CUDA Graph 安全边界。

### 1.4 `needs_cpu_seq_lens` 让 backend 显式声明是否需要 CPU 长度

问题与约束：
- 某些 backend 在 metadata 初始化时需要 `seq_lens_cpu` 或 `seq_lens_sum`。
- 另一些 backend 可以完全从 GPU 侧或预分配 buffer 推导长度。
- 不必要的 CPU length 准备会增加同步和数据搬运。

设计选择：
- 基类默认 `needs_cpu_seq_lens = True`，保持保守行为。
- 只有 backend 确认从不读取 CPU seq lens 时才 opt out。
- 该开关作为类属性暴露给调用方检查。

Explain：
这个字段是 attention backend 对 ModelRunner 的依赖声明。默认假设需要 CPU 长度，避免漏准备；性能更激进的 backend 必须主动证明自己不需要。

来源：python/sglang/srt/layers/attention/base_attn_backend.py L89-L90

Code：

```python
    # Opt out only when this backend never reads seq_lens_cpu / seq_lens_sum.
    needs_cpu_seq_lens: bool = True
```

代码逻辑：
- 在基类上定义类属性。
- 默认值为 True。
- 注释说明 opt out 条件是 backend 从不读取 CPU seq lens。

为什么这样写：
- 默认保守能保护老 backend 和外部 backend。
- opt out 作为显式声明，便于 backend 自己承担正确性责任。
- 调用方可以用统一字段决定是否准备 CPU 长度。

不变量与失败模式：
- 设置为 False 的 backend 不能在任何路径读取 `seq_lens_cpu/seq_lens_sum`。
- 如果错误 opt out，metadata 可能缺少长度输入。
- 如果忘记 opt out，功能正确但会保留不必要的 CPU 侧准备。

Comment：
这是 attention metadata 准备中的小型性能契约。

### 1.5 `init_cuda_graph_state` 预留 capture 稳定 buffer

问题与约束：
- CUDA Graph replay 要求被捕获的 tensor 地址稳定。
- attention metadata 中常有 indptr、indices、workspace 等随 batch 变化的 buffer。
- backend 需要在 capture 前按最大 batch/token 规模一次性建好可复用状态。

设计选择：
- 基类声明 `init_cuda_graph_state(max_bs, max_num_tokens)` 接口。
- 具体 buffer 布局交给 FlashInfer、Triton 等子类实现。
- 基类还声明 breakable CUDA Graph metadata capture/replay 相关接口。

Explain：
`init_cuda_graph_state` 是 graph capture 前的容量规划入口。它不关心 metadata 里有哪些 tensor，只要求 backend 用最大形状建立 replay 期间地址稳定的共享状态。

来源：python/sglang/srt/layers/attention/base_attn_backend.py L92-L115

Code：

```python
    # Most attention backends can rebuild and replace forward metadata before
    # every forward. BCG capture is different: some backends expose metadata
    # tensors to kernels across graph breaks, so the captured graph depends on
    # those tensor addresses.
    use_captured_forward_metadata_for_breakable_cuda_graph: bool = False

def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
    """Init the global shared states for cuda graph."""
    raise NotImplementedError()

def init_forward_metadata_for_breakable_cuda_graph_capture(
    self,
    forward_batch: ForwardBatch,
):
    """Create forward metadata whose tensor addresses will be graph-captured."""
    raise NotImplementedError()
```

代码逻辑：
- 用类属性标记是否需要 captured forward metadata。
- 声明 `init_cuda_graph_state` 接口，接收最大 batch 和 token 数。
- 默认抛 `NotImplementedError`，要求 backend 自行实现。
- 声明 breakable graph capture/replay metadata 接口。

为什么这样写：
- buffer 地址稳定是 CUDA Graph replay 的硬约束，必须在 backend 层显式处理。
- 不同 kernel 的 metadata 布局差异很大，基类不能统一分配。
- breakable graph 需要额外捕获 metadata tensor 地址，因此单独暴露 opt-in 标志和接口。

不变量与失败模式：
- capture 时使用的 buffer 不能在 replay 前被释放或替换。
- `max_bs/max_num_tokens` 必须覆盖 replay 的静态上界。
- backend 未实现该接口却进入 graph 路径，会直接抛 `NotImplementedError`。

Comment：
Graph state 初始化解决的是地址稳定性，而不只是内存提前分配。

## 2. 具体 backend 的 metadata 与 wrapper 选择

### 2.1 Triton `ForwardMetadata` 承载 paged KV 和 sliding window 字段

问题与约束：
- Triton attention kernel 需要 paged KV 的 `indptr/indices`、query/output offset 和 mask metadata。
- split-KV attention 需要每个请求的 KV split 数。
- Sliding window attention 使用不同的 window KV 索引和 offset。
- SWA 层的 `v_head_dim` 可能与普通 attention 不同。

设计选择：
- 用 dataclass `ForwardMetadata` 聚合所有 Triton forward 需要的 tensor。
- 同时保存 full attention 和 window attention 的 KV metadata。
- 为 SWA logits、SWA out cache loc、unified pool physical out cache loc 提供可选字段。

Explain：
`ForwardMetadata` 是 Python 调度层和 Triton kernel 之间的结构化协议。它把 paged KV、mask、split、SWA 和 unified pool 写入位置集中到一个对象中，便于 backend 在不同 forward 模式下复用。

来源：python/sglang/srt/layers/attention/triton_backend.py L81-L103

Code：

```python
@dataclass
class ForwardMetadata:
    attn_logits: torch.Tensor
    attn_lse: torch.Tensor
    max_extend_len: int
    num_kv_splits: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    qo_indptr: torch.Tensor
    custom_mask: torch.Tensor
    mask_indptr: torch.Tensor
    # Sliding window
    window_kv_indptr: torch.Tensor
    window_kv_indices: torch.Tensor
    window_num_kv_splits: torch.Tensor
    window_kv_offsets: torch.Tensor
    # Separate attn_logits for SWA layers when v_head_dim differs
    swa_attn_logits: Optional[torch.Tensor] = None
    # full->SWA translated out_cache_loc (SWA KV-store write target)
    swa_out_cache_loc: Optional[torch.Tensor] = None
    out_cache_loc_full_physical: Optional[torch.Tensor] = None
```

代码逻辑：
- 定义 Triton backend 的 forward metadata dataclass。
- 保存 attention logits、LSE 和最大 extend 长度。
- 保存 split-KV、KV page、QO、mask 相关 tensor。
- 保存 sliding window 专用的 KV page 和 offset tensor。
- 提供 SWA 与 unified pool 的可选写入字段。

为什么这样写：
- dataclass 让 metadata 字段在 Python 层有明确结构，减少参数列表膨胀。
- full attention 与 SWA 字段放在同一对象，backend 可按层类型选择。
- 可选字段支持特殊层和统一内存池，不强迫普通路径提供无用 tensor。

不变量与失败模式：
- `kv_indptr/kv_indices/qo_indptr` 必须与当前 batch 的 paged KV 布局一致。
- sliding window 层必须使用 window 字段，而不能误用 full KV 索引。
- CUDA Graph replay 中相关 tensor 地址需要由 graph state 保持稳定。

Comment：
Triton backend 的复杂度很大一部分体现在 metadata 布局，而不是 Python 层调用 kernel 的那一行。

### 2.2 `_safe_merge_state` 在 FlashInfer blockDim 超限时回退 Triton

问题与约束：
- FlashInfer `merge_state` 的 CUDA block 线程数受 `head_dim`、元素大小和 head 数影响。
- DP attention 等场景可能让 head 数超过 FlashInfer kernel 的安全上限。
- 上层 attention 逻辑不应关心具体 merge 实现是否需要 fallback。

设计选择：
- 先用 `_merge_state_max_safe_num_heads` 根据 `head_dim` 和 `element_size` 计算安全 head 上限。
- head 数不超限时调用 FlashInfer `merge_state`。
- 超限时透明回退到 `merge_state_triton`。

Explain：
`_safe_merge_state` 是 FlashInfer backend 的安全包装。它把 kernel launch 约束转化成运行时分支，避免在大 head 数场景中触发 blockDim 超限。

来源：python/sglang/srt/layers/attention/flashinfer_backend.py L97-L116

Code：

```python
def _merge_state_max_safe_num_heads(head_dim: int, element_size: int) -> int:
    vec_size = max(16 // element_size, head_dim // 32)
    bdx = head_dim // vec_size
    if bdx <= 0:
        return _MERGE_STATE_CUDA_MAX_THREADS_PER_BLOCK
    return _MERGE_STATE_CUDA_MAX_THREADS_PER_BLOCK // bdx

def _safe_merge_state(
    v_a: torch.Tensor,
    s_a: torch.Tensor,
    v_b: torch.Tensor,
    s_b: torch.Tensor,
):
    num_heads = v_a.shape[1]
    head_dim = v_a.shape[2]
    max_heads = _merge_state_max_safe_num_heads(head_dim, v_a.element_size())
    if num_heads <= max_heads:
        return merge_state(v_a, s_a, v_b, s_b)
    return merge_state_triton(v_a, s_a, v_b, s_b)
```

代码逻辑：
- 根据 element size 和 head dim 估算 FlashInfer 使用的 vector size。
- 计算每个 head 需要的 block x 维线程数。
- 推出不超过 1024 threads/block 的最大 head 数。
- 从 `v_a` 读取当前 head 数和 head dim。
- head 数安全时调用 FlashInfer merge。
- 超限时调用 Triton merge。

为什么这样写：
- fallback 判断放在 backend 内部，对调用方透明。
- 只在超限时回退，保留 FlashInfer 快路径。
- 按 FlashInfer 内部 vector 选择规则估算，比固定阈值更稳。

不变量与失败模式：
- `v_a` shape 必须是 merge_state 约定的 `[*, num_heads, head_dim]` 语义。
- `_merge_state_max_safe_num_heads` 需要跟 FlashInfer kernel 的 vec_size 规则保持同步。
- Triton fallback 必须与 FlashInfer merge 的数值语义兼容。

Comment：
这里的设计目标是把 kernel launch 限制封装成 backend 私有细节。

### 2.3 `WrapperDispatch` 给 FlashInfer 层类型分配 wrapper 族

问题与约束：
- FlashInfer backend 需要为普通 attention、sliding window attention 和 cross attention 选择不同 wrapper。
- 不同 wrapper 使用的 KV 索引和缓存语义不同。
- 层类型选择应使用稳定枚举，而不是散落字符串。

设计选择：
- 定义 `WrapperDispatch` 枚举。
- 当前显式枚举 `SLIDING_WINDOW` 和 `CROSS_ATTENTION`。
- backend 内部用枚举值映射 wrapper 数组下标或分发分支。

Explain：
这个枚举是 FlashInfer backend 内部的层类型标签。它让 SWA 和 cross attention 的 wrapper 选择成为类型化分发，而不是在多处分散判断。

来源：python/sglang/srt/layers/attention/flashinfer_backend.py L119-L121

Code：

```python
class WrapperDispatch(Enum):
    SLIDING_WINDOW = auto()
    CROSS_ATTENTION = auto()
```

代码逻辑：
- 定义 FlashInfer wrapper 分发枚举。
- 声明 sliding window wrapper 类型。
- 声明 cross attention wrapper 类型。

为什么这样写：
- 枚举比字符串更适合内部索引和分支。
- 新 wrapper 类型可以集中扩展。
- 层类型判断和 wrapper 选择解耦，便于 backend 初始化 wrapper 数组。

不变量与失败模式：
- backend 的 wrapper 数组/映射必须覆盖这些枚举值。
- sliding window 层不能误派发到 cross attention wrapper。
- 新增枚举时需要同步更新 `_get_wrapper_idx` 等分发逻辑。

Comment：
FlashInfer 的一个 backend 实例内部包含多个 wrapper 族。

## 3. 模型层调用 backend

### 3.1 `RadixAttention.forward` 统一处理 QKV reshape 与 backend 分发

问题与约束：
- 模型层传入的 `q/k/v` 需要 reshape 成 `[token, head, dim]` 形式。
- cross-layer sharing 场景中 `k/v` 可以为 None。
- piecewise CUDA Graph 路径要走自定义 op，普通路径要走当前 attention backend。
- backend 需要拿到 `RadixAttention` 层对象，以读取 head 数、layer id、SWA、cross attention 等配置。

设计选择：
- 先在层内根据 `tp_k_head_num/tp_v_head_num/qk_head_dim/v_head_dim` reshape K/V。
- extend 且存在 piecewise context 时调用 `unified_attention_with_output` 或 breakable 版本。
- 否则通过 `get_attn_backend().forward(...)` 分发到当前 backend。
- 把 `self` 和 `forward_batch` 一起传给 backend。

Explain：
`RadixAttention` 是模型层与 attention backend 的接缝。Scheduler/Radix cache 准备好的 batch metadata 最终通过 `forward_batch` 进入 backend，而层本身提供 head、dim、layer id 和 attention 类型。

来源：python/sglang/srt/layers/radix_attention.py L109-L153

Code：

```python
def forward(
    self,
    q,
    k,
    v,
    forward_batch: ForwardBatch,
    save_kv_cache: bool = True,
    **kwargs,
):
    if k is not None:
        assert v is not None
        if "k_rope" not in kwargs:
            k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
            v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
        else:
            k = k.view(-1, self.tp_k_head_num, self.v_head_dim)

    if (
        forward_batch.forward_mode.is_extend()
        and get_tc_piecewise_forward_context() is not None
    ):
        if self.qk_head_dim != self.v_head_dim:
            output = q.new_empty((q.shape[0], self.tp_q_head_num * self.v_head_dim))
        else:
            output = torch.empty_like(q)
        if is_in_breakable_cuda_graph():
            breakable_unified_attention_with_output(
                q, k, v, output, save_kv_cache, self.layer_id, **kwargs
            )
        else:
            unified_attention_with_output(
                q, k, v, output, save_kv_cache, self.layer_id, **kwargs
            )
        return output
    else:
        return get_attn_backend().forward(
            q,
            k,
            v,
            self,
            forward_batch,
            save_kv_cache,
            **kwargs,
        )
```

代码逻辑：
- 如果 K/V 存在，先校验 V 也存在。
- 根据是否传入 `k_rope` 选择 K 的 view 形状。
- 判断当前是否是 extend 模式且处于 TC piecewise context。
- piecewise 路径先分配输出 tensor。
- breakable CUDA Graph 中调用 breakable 自定义 op。
- 普通 piecewise 中调用 unified 自定义 op。
- 非 piecewise 路径调用当前 attention backend 的 `forward`。

为什么这样写：
- QKV reshape 留在模型层，backend 可以接收统一形状。
- piecewise CUDA Graph 需要自定义 op 来稳定 graph 切分和输出写入。
- 普通路径通过 `get_attn_backend()` 保持 backend 可插拔。
- 传入 `self` 让 backend 不需要重新查层配置。

不变量与失败模式：
- `q/k/v` shape 必须能按 TP head 数和 head dim reshape。
- `forward_batch.forward_mode` 必须准确标识 extend/decode 等模式。
- piecewise context 缺失时不能走 unified custom op。
- backend `forward` 必须理解传入的 attention layer 配置。

Comment：
RadixAttention 层不直接决定用 FlashInfer、Triton 还是其他 kernel；它把标准化后的输入交给当前 backend。
