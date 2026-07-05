---
type: batch-doc
module: 16-KV-Cache
batch: "16"
doc_type: walkthrough
title: "KV Cache · 源码走读"
tags:
 - sglang/batch/16
 - sglang/module/kv-cache
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-05
---
# KV Cache · 源码走读

## 1. 设备侧 KV 索引分配器

### 1.1 Base allocator 把 token/page 差异收敛到统一接口

问题与约束：
- Scheduler、RadixCache 和 ModelRunner 都需要申请/释放 KV cache 位置，但底层可能是 token 粒度或 page 粒度。
- 释放可能来自 Radix 树批量插入、重复 prefix 去重、请求结束等路径，不能让上层感知具体空闲队列形态。
- page allocator 还需要 extend/decode 的批量接口，而 token allocator 不支持这些接口。

设计选择：
- `BaseTokenToKVPoolAllocator` 保存 `size/page_size/dtype/device/kvcache/need_sort`，并统一暴露 `available_size`、`backup_state/restore_state`、`free_group`、`merge_and_sort_free`、`alloc/free`。
- `alloc_extend/alloc_decode` 在基类中默认抛错，只允许 paged allocator 覆盖。

Explain：
这个基类定义了设备侧 KV cache 索引管理的协议。上层只依赖“还有多少 token 容量、申请一段索引、释放一段索引”；底层可以把 `free_pages` 解释为 token slots，也可以解释为 pages。

来源：python/sglang/srt/mem_cache/allocator/base.py L27-L110

Code：

```python
class BaseTokenToKVPoolAllocator(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self._kvcache = kvcache
        self.need_sort = need_sort

        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []

    def available_size(self):
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))

    def merge_and_sort_free(self):
        if len(self.release_pages) > 0:
            self.free_pages = torch.cat((self.free_pages, self.release_pages))
            self.free_pages, _ = torch.sort(self.free_pages)
            self.release_pages = torch.empty(
                (0,), dtype=self.release_pages.dtype, device=self.device
            )

    def alloc_extend(self, *args, **kwargs):
        raise NotImplementedError("alloc_extend is only for paged allocator")

    def alloc_decode(self, *args, **kwargs):
        raise NotImplementedError("alloc_decode is only for paged allocator")

    @abc.abstractmethod
    def alloc(self, need_size: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def free(self, free_index: torch.Tensor):
        raise NotImplementedError()
```

代码逻辑：
- 初始化保存容量、page size、设备、KV cache 对象和是否需要排序。
- `available_size` 默认按 page 数乘 page size 折算 token 容量。
- `free_group_begin/end` 让一组释放先暂存，再一次性 `torch.cat` 后交给 `free`。
- `merge_and_sort_free` 把延迟释放队列并回空闲队列并排序。
- `alloc/free/clear` 是抽象方法，extend/decode 批量分配默认不支持。

为什么这样写：
- 上层调度逻辑不需要区分 token allocator 和 paged allocator。
- 延迟释放和排序是 allocator 内部策略，避免 RadixCache 批量操作时频繁改动空闲列表。
- 把 paged-only 接口放在基类中显式抛错，可以在错误配置时尽早暴露。

不变量与失败模式：
- `free_pages` 与 `release_pages` 必须使用同一 device 和 dtype。
- `available_size` 依赖 page size 语义，token allocator 要覆盖以去掉多余乘法。
- 调用未覆盖的 `alloc_extend/alloc_decode` 表示上层把非 paged allocator 用到了 paged-only 路径。

Comment：
这一层把 KV cache 当作“索引池”建模，真正的 KV tensor 读写由 `_kvcache` 承担。

### 1.2 Token allocator 用 slot 切片表达最小粒度 KV 分配

问题与约束：
- token 粒度模式下每个 KV slot 独立分配，`page_size=1`。
- slot 0 被保留给 padding dummy，不能作为真实 token 分配出去。
- 空间不足要返回 `None`，让 Scheduler 走 retract/evict，而不是在 allocator 内部阻塞。

设计选择：
- `clear` 初始化 `free_pages = arange(1, size + 1)`，跳过 slot 0。
- `alloc` 从 `free_pages` 前缀切片。
- `free` 根据 `need_sort` 决定进入 `release_pages` 还是直接拼回 `free_pages`；free group 模式下先追加到 `free_group`。

Explain：
`TokenToKVPoolAllocator` 是最直接的设备 KV slot 管理器。它把 `free_pages` 当作 token slot 列表，申请就是切掉前 N 个 slot，释放就是把索引放回空闲队列或延迟释放队列。

来源：python/sglang/srt/mem_cache/allocator/token.py L55-L76

Code：

```python
def alloc(self, need_size: int):
    if self.need_sort and need_size > len(self.free_pages):
        self.merge_and_sort_free()

    if need_size > len(self.free_pages):
        return None

    select_index = self.free_pages[:need_size]
    self.free_pages = self.free_pages[need_size:]
    return select_index

def free(self, free_index: torch.Tensor):
    if free_index.numel() == 0:
        return

    if self.is_not_in_free_group:
        if self.need_sort:
            self.release_pages = torch.cat((self.release_pages, free_index))
        else:
            self.free_pages = torch.cat((self.free_pages, free_index))
    else:
        self.free_group.append(free_index)
```

代码逻辑：
- 分配前如果需要排序且当前空闲不足，先合并延迟释放队列。
- 空闲仍不足则返回 `None`。
- 成功时取 `free_pages[:need_size]` 并从空闲队列移除。
- 释放空 tensor 直接返回。
- 普通释放按 `need_sort` 进入 `release_pages` 或 `free_pages`；group 释放先暂存。

为什么这样写：
- 前缀切片分配简单且快，适合 token 粒度。
- `need_sort` 时延迟合并可以减少每次 free 的排序成本。
- OOM 由上层统一处理，allocator 只报告“现在没有足够 slot”。

不变量与失败模式：
- 返回的 token index 不包含保留 slot 0。
- `need_size` 不能超过 `available_size`，否则返回 `None` 而不是部分分配。
- 如果调用者重复释放同一 slot，token allocator 本身不做 double-free 检测，依赖上层状态一致性。

Comment：
token allocator 的优势是简单；缺点是不能直接满足 paged attention 对 page 对齐的要求。

### 1.3 merge-and-sort 把延迟释放重新变成有序空闲队列

问题与约束：
- `need_sort=True` 时释放索引先进入 `release_pages`，如果一直不合并，allocator 会低估可用空间。
- 直接每次 free 都排序会增加释放路径开销。
- 空闲索引有序有助于后续分配的局部性和调试可读性。

设计选择：
- 只有在需要时调用 `merge_and_sort_free`，把 `free_pages` 与 `release_pages` 拼接后排序，并清空 `release_pages`。

Explain：
`merge_and_sort_free` 是延迟释放策略的收口点。它把“快速 free”留下的 release 队列重新并回主空闲队列，并恢复有序状态。

来源：python/sglang/srt/mem_cache/allocator/base.py L78-L84

Code：

```python
def merge_and_sort_free(self):
    if len(self.release_pages) > 0:
        self.free_pages = torch.cat((self.free_pages, self.release_pages))
        self.free_pages, _ = torch.sort(self.free_pages)
        self.release_pages = torch.empty(
            (0,), dtype=self.release_pages.dtype, device=self.device
        )
```

代码逻辑：
- 如果 `release_pages` 为空，不做任何事。
- 拼接当前空闲队列和延迟释放队列。
- 对合并后的 free list 排序。
- 用同 dtype/device 的空 tensor 重置 `release_pages`。

为什么这样写：
- 延迟合并把排序成本从 free 路径推迟到真正缺空间的 alloc 前。
- 排序后的 free list 能让后续分配更稳定地复用低地址索引。
- 重建空 tensor 保持 device/dtype 一致，避免后续 `torch.cat` 跨设备。

不变量与失败模式：
- `release_pages` 中不应包含仍被活跃请求持有的索引。
- 排序不会去重，重复释放仍可能导致同一索引重复进入 free list。
- 调用方必须在容量判断前适时合并，否则会误判 OOM。

Comment：
这是 allocator 的“整理抽屉”动作：不在每次释放时做，但在空间紧张前要做。

### 1.4 free group 为 Radix 批处理提供释放事务边界

问题与约束：
- Radix 树插入或去重时可能连续释放多个片段。
- 每个片段都立即进入 free list 会造成频繁 `cat/sort`，也可能暴露中间态。
- 分配器需要一个轻量事务边界来收集释放。

设计选择：
- `free_group_begin` 切换到 group 模式并清空 `free_group`。
- `free_group_end` 恢复普通模式，并把收集的 tensors 一次性拼接后调用 `free`。

Explain：
free group 不改变释放语义，只改变释放提交时机。它把一组小释放合成一次大释放，让子类的 `free` 仍然是唯一入口。

来源：python/sglang/srt/mem_cache/allocator/base.py L69-L76

Code：

```python
def free_group_begin(self):
    self.is_not_in_free_group = False
    self.free_group = []

def free_group_end(self):
    self.is_not_in_free_group = True
    if self.free_group:
        self.free(torch.cat(self.free_group))
```

代码逻辑：
- begin 时把 `is_not_in_free_group` 置为 False。
- group 期间子类 `free` 会把 index tensor append 到 `free_group`。
- end 时如果有暂存索引，`torch.cat` 后重新走 `free`。

为什么这样写：
- 批量释放减少多次拼接和排序。
- 重新调用 `free` 复用子类关于 `need_sort`、page 去重等逻辑。
- 空 group 不做额外操作，避免无意义 tensor 构造。

不变量与失败模式：
- begin/end 必须成对调用，否则释放会一直停留在 `free_group`。
- group 内暂存的 tensor dtype/device 必须一致，才能 `torch.cat`。
- end 后再调用 `free` 时 `is_not_in_free_group=True`，否则会递归追加而不是提交。

Comment：
这是一种很小的事务机制，服务于 RadixCache 对 KV 索引的批量重排。

### 1.5 Paged allocator 把 page id 展开成 token index

问题与约束：
- FlashInfer PagedAttention 需要 page 对齐的 KV 索引。
- allocator 内部管理 page id，但 ModelRunner 最终需要 token 级 index 张量。
- ROCm 上首次 `torch.unique` 可能触发 JIT 延迟，影响重复 prompt benchmark 的第二次请求。

设计选择：
- `PagedTokenToKVPoolAllocator` 记录 `num_pages` 与 `debug_mode`，ROCm 时预热 `torch.unique`。
- `alloc` 要求 `need_size` page 对齐，按 page 数从 `free_pages` 切片，再展开成连续 token indices。

Explain：
paged allocator 的空闲队列单位是 page。申请 N 个 token 时，先换算为 page 数，拿到 page ids 后用 `page_id * page_size + arange(page_size)` 展开为 token 位置。

来源：python/sglang/srt/mem_cache/allocator/paged.py L105-L170

Code：

```python
class PagedTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)
        self.num_pages = size // page_size
        self.debug_mode = get_bool_env_var("SGLANG_DEBUG_MEMORY_POOL")

        if _is_hip and torch.cuda.is_available():
            try:
                _warmup = torch.arange(1024, dtype=torch.int64, device=device)
                _ = torch.unique(_warmup // page_size)
                torch.cuda.synchronize()
            except Exception:
                pass
        self.clear()

    def alloc(self, need_size: int):
        if self.debug_mode:
            assert (
                need_size % self.page_size == 0
            ), "The allocation size should be page-aligned"

        num_pages = need_size // self.page_size
        if self.need_sort and num_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_pages > len(self.free_pages):
            return None

        out_pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]

        out_indices = (
            out_pages[:, None] * self.page_size
            + torch.arange(self.page_size, device=self.device)
        ).reshape(-1)

        return out_indices
```

代码逻辑：
- 初始化时计算 page 数并读取 debug 开关。
- ROCm 且 CUDA 可用时构造小 tensor，触发一次 `torch.unique`。
- `alloc` 在 debug 模式下检查 token 数是否 page 对齐。
- 用 `need_size // page_size` 得到需要的 page 数。
- 从 `free_pages` 中取出 page id，再展开为 token index。

为什么这样写：
- page id 管理更适合 paged attention 的内存布局。
- 返回 token index 保持和上层 KV 写入接口兼容。
- ROCm 预热把首次 unique 的 JIT 成本移动到初始化阶段，避免请求路径抖动。

不变量与失败模式：
- debug 模式下非 page 对齐申请会 assert。
- 如果 page 数不足，返回 `None`，上层需要 retract/evict。
- `size` 不是 `page_size` 整数倍时，`num_pages = size // page_size` 会截断，调用层应传入对齐容量。

Comment：
paged allocator 是 KV cache 与 paged attention backend 之间的形状适配层。

## 2. Prefill 与 Decode 的批量分配

### 2.1 `alloc_extend` 在 GPU 上为 prefill 新 token 生成索引

问题与约束：
- prefill extend batch 中每个请求有不同 prefix length 和 seq length。
- 需要为所有新增 token 生成 KV index，同时处理 page 边界。
- CPU 逐请求计算会拖慢大 batch prefill。

设计选择：
- `alloc_extend` 接收 `prefix_lens/seq_lens/last_loc/extend_num_tokens`，在 GPU 上调用 `alloc_extend_kernel` 填充 `out_indices`。
- 分配成功后按 `get_num_new_pages` 计算实际消耗的 page 数，并从 `free_pages` 前缀移除。

Explain：
extend 分配不是简单拿连续 N 个 token index，因为每个请求的 prefix 可能落在不同 page offset。Triton kernel 根据每个请求的边界并行写出 token index，Python 侧只负责容量检查和 free page 游标推进。

来源：python/sglang/srt/mem_cache/allocator/paged.py L172-L215

Code：

```python
def alloc_extend(
    self,
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
    num_new_pages: int = None,
):
    if self.debug_mode:
        assert torch.all(
            (last_loc + 1) % self.page_size == prefix_lens % self.page_size
        )

    bs = len(prefix_lens)
    if self.need_sort and extend_num_tokens // self.page_size + bs + 1 > len(
        self.free_pages
    ):
        self.merge_and_sort_free()

    out_indices = torch.empty(
        (extend_num_tokens,), dtype=torch.int64, device=self.device
    )

    alloc_extend_kernel[(bs,)](
        prefix_lens,
        seq_lens,
        last_loc,
        self.free_pages,
        out_indices,
        next_power_of_2(bs),
        self.page_size,
    )

    if num_new_pages is None:
        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            prefix_lens=prefix_lens_cpu,
        )
```

代码逻辑：
- debug 模式检查 `last_loc` 与 prefix page offset 一致。
- 根据 batch size 和可能需要的新 page 数，必要时合并延迟释放。
- 创建长度为 `extend_num_tokens` 的输出 index tensor。
- 启动 `alloc_extend_kernel[(bs,)]`，每个请求并行写自己的新 token indices。
- 未显式传入 `num_new_pages` 时，用 CPU seq/prefix lengths 计算。

为什么这样写：
- per-request page 边界计算适合 GPU 并行。
- 输出 token index tensor 直接供后续 KV 写入使用。
- free page 消耗量由 CPU lengths 决定，避免在 Python 侧扫描 GPU 输出。

不变量与失败模式：
- `last_loc + 1` 的 page offset 必须对应 prefix length，否则新 token 会接错 page。
- `extend_num_tokens` 必须等于 batch 中新增 token 总数。
- 若后续 `num_new_pages > len(free_pages)`，函数返回 `None`，上层要处理 OOM。

Comment：
extend 分配的难点不在“拿多少空间”，而在每个请求跨 page 的位置计算。

### 2.2 `alloc_decode` 为每个请求追加一个 token index

问题与约束：
- decode 阶段每个活跃请求通常只新增一个 token。
- 虽然输出 index 是 `[bs]`，但只有跨 page 边界的请求才真正消耗新 page。
- OOM 时同样要返回 `None`，让 Scheduler 回退。

设计选择：
- `alloc_decode` 以 `seq_lens/last_loc` 调用 `alloc_decode_kernel`，生成每个请求的单 token index。
- 用 `get_num_new_pages(..., decode=True)` 计算本轮需要弹出的 page 数。

Explain：
decode 分配把“每请求一个 token”的规则下沉到 Triton kernel。Python 侧不知道哪个请求是否刚好跨 page，只通过 `get_num_new_pages` 统一推进 free page 队列。

来源：python/sglang/srt/mem_cache/allocator/paged.py L222-L259

Code：

```python
def alloc_decode(
    self,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
):
    if self.debug_mode:
        assert torch.all(
            (last_loc + 2) % self.page_size == seq_lens % self.page_size
        )

    bs = len(seq_lens)
    if self.need_sort and bs > len(self.free_pages):
        self.merge_and_sort_free()

    out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)
    alloc_decode_kernel[(bs,)](
        seq_lens,
        last_loc,
        self.free_pages,
        out_indices,
        next_power_of_2(bs),
        self.page_size,
    )

    num_new_pages = get_num_new_pages(
        seq_lens=seq_lens_cpu,
        page_size=self.page_size,
        decode=True,
    )
    if num_new_pages > len(self.free_pages):
        return None

    self.free_pages = self.free_pages[num_new_pages:]
    return out_indices
```

代码逻辑：
- debug 模式检查 decode 后位置与 seq length 的 page offset。
- batch size 即输出 index 数。
- 空闲 page 可能不足时先 merge-and-sort。
- GPU kernel 为每个请求写一个 index。
- 计算实际新 page 数，容量不足返回 `None`。
- 消耗对应数量的 `free_pages`，返回 `[bs]` index。

为什么这样写：
- decode 每步频繁执行，保持 GPU 批量计算可以减少 Python 开销。
- 输出 `[bs]` 保持和 batch 中请求一一对应。
- 只消耗跨 page 边界所需的新 page，避免每 token 都浪费一个 page。

不变量与失败模式：
- `last_loc` 必须表示 decode 前最后一个 token 的 KV 位置。
- `seq_lens_cpu` 与 GPU `seq_lens` 必须描述同一批请求。
- 如果 batch 中多个请求共享或重复 index，debug unique 检查会在源码后续路径中暴露。

Comment：
extend 处理“多 token 跨 page”，decode 处理“单 token 是否跨 page”，两者共享 free page 队列。

## 3. HiCache 主机侧 KV 池

### 3.1 HostKVCache 初始化主机容量并校验内存预算

问题与约束：
- HiCache L2 主机池要与设备 KV pool 的 token 大小、layer 范围和 page size 对齐。
- 用户可能指定固定 GB 容量，也可能按 device pool ratio 推导容量。
- 主机内存不足时不能继续分配，否则会导致系统层面的内存压力。

设计选择：
- `HostKVCache.__init__` 从 device pool 获取 dtype/layer range，用 `get_size_per_token` 计算每 token 字节数。
- 根据 `host_size` 或 `host_to_device_ratio` 计算 token 容量，并向上对齐到 page size。
- 分配前用 `psutil.virtual_memory()` 检查可用主机内存，最后初始化 buffer 和锁。

Explain：
HostKVCache 是设备 KV pool 的主机扩展层。它先把“GB 或 ratio”翻译成 page-aligned token 容量，再检查主机内存预算，最后创建 host KV buffer 并清空空闲状态。

来源：python/sglang/srt/mem_cache/pool_host/base.py L79-L143

Code：

```python
class HostKVCache(abc.ABC):
    def __init__(
        self,
        device_pool: KVCache,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool,
        device: str,
        allocator_type: str = "default",
    ):
        self.device_pool = device_pool
        self.page_size = page_size
        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        self.allocator = get_allocator_from_storage(allocator_type)
        self.can_use_write_back_jit = False

        self.dtype = device_pool.store_dtype
        self.size_per_token = self.get_size_per_token()
        if host_size > 0:
            self.size = sync_fixed_hicache_size(
                int(host_size * 1e9 // self.size_per_token), host_size
            )
        else:
            self.size = int(device_pool.size * host_to_device_ratio)
        self.page_num = self.size // self.page_size + 1
        self.size = self.page_num * self.page_size
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer

        host_mem = psutil.virtual_memory()
        requested_bytes = self.size * self.size_per_token
        available_bytes = host_mem.available - HICACHE_HOST_MEMORY_RESERVE_BYTES
        if requested_bytes > available_bytes:
            raise ValueError(...)

        self.kv_buffer = self.init_kv_buffer()
        self.lock = threading.RLock()
        self.clear()
```

代码逻辑：
- 保存 device pool、page size、layout、pin memory 和 allocator 类型。
- 用 device pool dtype 计算 host 每 token KV 大小。
- 固定容量优先，否则按 ratio 推导。
- 容量向 page size 对齐，并继承 device pool 的 layer 范围。
- 检查 host 可用内存，成功后初始化 host KV buffer。
- 创建 `RLock` 并清空空闲 slot 状态。

为什么这样写：
- host pool 必须以 token/page 为单位和设备池对齐，才能做 KV 迁移。
- 内存预算在分配前失败，比运行中触发 OOM 更可控。
- `RLock` 为后续并发 prefetch/write-back/alloc/free 提供同步基础。

不变量与失败模式：
- 子类必须实现 `get_size_per_token` 和 `init_kv_buffer`，否则基类无法计算容量或分配 buffer。
- `requested_bytes` 超过可用内存会抛 `ValueError`。
- host pool 小于 device pool 时源码会 warning，说明 L2 命中收益可能下降。

Comment：
HiCache 主机池不是简单 Python list，而是按 KV tensor 字节布局和 page 粒度管理的 L2 cache。

### 3.2 HostKVCache 用锁和 slot bitmap 防止并发损坏

问题与约束：
- host KV pool 会被 prefetch、backup、evict 等路径并发访问。
- 主机 slot 释放和分配出错会导致 KV 内容错位，属于 silent corruption 风险。
- host pool 同样要求 page 对齐。

设计选择：
- `alloc/free` 用 `@synchronized` 包装，在 `self.lock` 下执行。
- `slot_used` bool tensor 记录每个 slot 是否被占用，alloc 检测 double-alloc，free 检测 double-free。

Explain：
主机侧 allocator 比设备侧 token allocator 多了显式一致性检查。每次分配从 `free_slots` 前缀切片并标记 used；每次释放先确认这些 slot 当前确实已分配。

来源：python/sglang/srt/mem_cache/pool_host/base.py L240-L268

Code：

```python
@synchronized
def alloc(self, need_size: int) -> Optional[torch.Tensor]:
    assert (
        need_size % self.page_size == 0
    ), "The requested size should be a multiple of the page size."
    if need_size > self.available_size():
        return None

    select_index = self.free_slots[:need_size]
    self.free_slots = self.free_slots[need_size:]

    assert not self.slot_used[select_index].any(), (
        f"Double-alloc detected: slots already allocated: "
        f"{select_index[self.slot_used[select_index]].tolist()}."
    )
    self.slot_used[select_index] = True

    return select_index

@synchronized
def free(self, indices: torch.Tensor) -> int:
    indices_cpu = indices.cpu()
    assert self.slot_used[indices_cpu].all(), (
        f"Double-free detected: slots not currently allocated: "
        f"{indices_cpu[~self.slot_used[indices_cpu]].tolist()}."
    )
    self.slot_used[indices_cpu] = False
    self.free_slots = torch.cat([self.free_slots, indices_cpu])
    return len(indices)
```

代码逻辑：
- `alloc` 先检查 page 对齐，再检查容量。
- 从 `free_slots` 前缀取出 slot，并移出空闲队列。
- `slot_used` 确认没有重复分配，再置 True。
- `free` 把输入 index 搬到 CPU，确认全部处于 used 状态。
- 释放后清除 used 标志，并把 slot 拼回 free list。

为什么这样写：
- 主机池并发路径多，用锁保证 free list 和 bitmap 同步更新。
- double-alloc/free 直接 assert，比让 KV 内容错位后继续运行更容易定位。
- page 对齐确保 host/device KV 迁移单位一致。

不变量与失败模式：
- `need_size` 必须是 page size 的倍数。
- `indices` 必须属于 host slot 范围且当前已分配。
- `free_slots` 拼回后不排序，调用方不能假设 host slot 单调递增。

Comment：
设备侧 allocator偏性能，host 侧 allocator更强调并发一致性和损坏检测。

## 4. 外部存储后端

### 4.1 StorageBackendFactory 先查注册表，再走 dynamic 后端

问题与约束：
- HiCache 可以接不同 storage backend，既有内置实现，也可能有外部动态实现。
- 后端类导入可能较重，注册时不应立即 import 所有 backend。
- 未知 backend 名称要给出可诊断错误。

设计选择：
- `create_backend` 先检查 `_registry`，命中则调用 lazy `loader()` 得到 backend class，再走 `_create_builtin_backend`。
- 如果 backend 名是 `dynamic` 且 `extra_config` 存在，则走 `_create_dynamic_backend`。
- 都不匹配时列出已注册后端并抛 `ValueError`。

Explain：
storage backend factory 把 “backend 名称” 解析成实际存储对象。它先走注册表的内置后端，保留 lazy import；动态后端通过配置加载，便于扩展 HiCache 存储层。

来源：python/sglang/srt/mem_cache/storage/backend_factory.py L66-L96

Code：

```python
@classmethod
def create_backend(
    cls,
    backend_name: str,
    storage_config: HiCacheStorageConfig,
    mem_pool_host: Any,
    **kwargs,
) -> HiCacheStorage:
    """Create a storage backend instance."""
    if backend_name in cls._registry:
        registry_entry = cls._registry[backend_name]
        backend_class = registry_entry["loader"]()
        logger.info(
            f"Creating storage backend '{backend_name}' "
            f"({registry_entry['module_path']}.{registry_entry['class_name']})"
        )
        return cls._create_builtin_backend(
            backend_name, backend_class, storage_config, mem_pool_host
        )
```

代码逻辑：
- 接收 backend 名称、storage 配置和 host memory pool。
- 如果名称在注册表中，取出 registry entry。
- 调 lazy loader 导入 backend class。
- 记录创建日志。
- 调 `_create_builtin_backend` 完成实例化。

为什么这样写：
- lazy loader 降低启动时不必要的 import 成本。
- 注册表路径让内置后端有明确名称和模块来源。
- dynamic 路径给外部存储扩展留接口，不需要改 factory 主逻辑。

不变量与失败模式：
- 注册表 entry 必须包含 `loader/module_path/class_name`。
- loader 导入失败会向上传播 ImportError 或初始化异常。
- 未注册且非 dynamic 配置会抛 `ValueError`，调用层必须给出合法 backend name。

Comment：
KV cache 的层级化不止到 host memory；storage backend factory 是继续接外部存储的边界。
