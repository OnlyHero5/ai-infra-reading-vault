---
type: batch-doc
module: 15-RadixAttention
batch: "15"
doc_type: walkthrough
title: "RadixAttention · 源码走读"
tags:
 - sglang/batch/15
 - sglang/module/radix-attention
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-05
---
# RadixAttention · 源码走读

> 读法：先看 `RadixKey` 的前缀匹配语义，再看 classic `RadixCache` 的写入、锁保护与驱逐；随后对照 `UnifiedRadixCache` 的多 component 版本，最后落到 `RadixAttention` 对 kernel 与图编译的接口约束。

---

## 1. RadixKey 与 classic RadixCache

### 1.1 `RadixKey.match`：长前缀的分歧定位

来源：python/sglang/srt/mem_cache/radix_cache.py L162-L196

**问题与约束：** Prefix cache 的热路径会反复比较请求 token 与树节点 key；长共享前缀若逐 token Python 循环，会把命中路径拖成解释器开销。匹配结果还必须同时服从 page 对齐与 EAGLE bigram 视图。

**设计选择：** 先用 slice 比较做指数 galloping，找到可能分歧窗口后再二分；最终把匹配长度裁剪到 `len(self)`、`len(other)` 与 `page_size`，bigram 模式额外减去一个逻辑 token。

**Explain：** 这里把“找第一个不同 token”拆成两个阶段：长相等区间由 C 层 slice 比较吞掉，短分歧窗口才进入二分。这样常见的长前缀命中不会被 Python token loop 放大。

**Code：**

```python
    def match(self, other: RadixKey, page_size: int = 1) -> int:
        """Logical-unit prefix length shared with ``other``. Result is rounded down to ``page_size``."""
        self._check_compatible(other)
        t0, t1 = self.token_ids, other.token_ids
        assert type(t0) is type(t1), (type(t0), type(t1))
        n = min(len(t0), len(t1))

        # Exponential search for the first diverging token: gallop in doubling
        # windows (one C-level slice compare each), then binary-search the window
        # holding the divergence -- no per-token Python loop on long shared prefixes.
        matched_tokens = n
        lo = 0
        step = 1
        while lo < n:
            hi = lo + step if lo + step < n else n
            if t0[lo:hi] != t1[lo:hi]:
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if t0[lo:mid] == t1[lo:mid]:
                        lo = mid
                    else:
                        hi = mid
                matched_tokens = lo
                break
            lo = hi
            step *= 2

        if self.is_bigram:
            matched = max(0, min(matched_tokens - 1, len(self), len(other)))
            return (matched // page_size) * page_size if page_size > 1 else matched

        matched_tokens = min(matched_tokens, len(self), len(other))
        if page_size == 1:
            return matched_tokens
        return (matched_tokens // page_size) * page_size
```

**代码逻辑：** `_check_compatible` 先保证额外 key 与 bigram 语义一致；`matched_tokens` 默认等于共同长度。循环中每次比较 `[lo:hi]`，若整段相等就把窗口翻倍推进；一旦不等，就在该窗口内二分，得到第一个分歧位置。最后按 bigram 与 page size 修正返回值。

**为什么这样写：** Radix 树节点 key 通常是一段连续 token，长 prompt 复用时共享前缀很长。galloping 能让“完全相等或很晚才分歧”的路径只做少量 slice 比较，而 page 对齐保证返回结果能直接映射到 KV page 边界。

**不变量与失败模式：** `token_ids` 类型必须一致；返回长度不能超过任一逻辑 key 长度；`page_size > 1` 时返回值必须是 page 的整数倍。bigram 若忘记减一，会让 EAGLE 缓存引用一个并不存在的 KV slot；若忽略 page 对齐，后续 `value[:len(key)]` 与 allocator page 语义会错位。

**Comment：** 这段是 prefix cache 的基本代价模型：把 token 级比较压到 slice 级，再把结果约束回 KV cache 能接受的逻辑长度。

### 1.2 `RadixCache.reset`：重建空树根

来源：python/sglang/srt/mem_cache/radix_cache.py L331-L336

**问题与约束：** `flush_cache` 或测试重置需要把 radix 树恢复到空状态，同时不能给根节点引入可驱逐 value 或普通 priority，避免根参与正常淘汰决策。

**设计选择：** 新建一个 `priority=-sys.maxsize` 的 root，把 key 设为空 `RadixKey`，并把 device 与 host value 都置为空列表。

**Explain：** 根节点是结构哨兵，不代表任何真实 token 段。最小 priority 让真实节点总能覆盖它，空 value 则让 allocator 不会尝试释放根。

**Code：**

```python
    def reset(self):
        # Initialize root with minimum priority so any real priority overrides it
        self.root_node = TreeNode(priority=-sys.maxsize)
        self.root_node.key = RadixKey(token_ids=array("q"), extra_key=None)
        self.root_node.value = []
        self.root_node.host_value = []
```

**代码逻辑：** `reset` 只替换 `root_node`，不在这里遍历旧树释放节点字段；释放动作由调用 reset 前后的 cache flush 流程保证。新 root 的 key/value 都是空形态，后续 insert 从这个根重新构建 children。

**为什么这样写：** root 需要稳定承担“无前缀”的匹配起点，而不是正常 cache segment。通过哨兵 priority 与空 value 区分 root，可以让 insert、match、evict 共享同一树结构而少写特殊分支。

**不变量与失败模式：** reset 之后根节点不能残留 children 或 value；若有活跃请求仍持有旧节点锁，直接 reset 会让请求的 `last_node` 与新树脱节。因此 flush 前必须确保没有不可释放的活跃路径。

**Comment：** 这段虽短，但定义了 classic radix cache 的空树形态：root 只承载拓扑，不承载 KV。

### 1.3 `RadixCache.insert`：把新算出的 KV 挂入树

来源：python/sglang/srt/mem_cache/radix_cache.py L415-L435

**问题与约束：** Prefill 或 chunked prefill 新算出的 KV indices 需要写入 prefix cache；写入前必须统一 EAGLE bigram 与 page 对齐语义，并返回树中已存在的 duplicate 前缀长度，供上层释放重复分配。

**设计选择：** `insert` 作为写路径入口，先处理 `disable`，再把 key/value 转为 bigram 视图、按 page 裁剪 key、按 key 长度裁剪 value，最后委托 `_insert_helper` 做树节点 split/merge。

**Explain：** 这不是简单 append。Radix 树可能已经有一段共享前缀，`_insert_helper` 会把新 key 分到已有节点或拆出新节点；`prefix_len` 告诉调用方哪部分 KV 已经被 cache 接管。

**Code：**

```python
    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)

        key = params.key
        value = params.value
        priority = params.priority
        chunked = params.chunked

        key, value = key.maybe_to_bigram_view(self.is_eagle, value)
        key = key.page_aligned(self.page_size)
        if value is not None:
            value = value[: len(key)]
        else:
            # Debug/test fallback: use token ids themselves as values.
            value = torch.tensor(key.token_ids[: len(key)], dtype=torch.int64)

        prefix_len, last_node = self._insert_helper(
            self.root_node, key, value, priority, chunked
        )
        return InsertResult(prefix_len=prefix_len, last_device_node=last_node)
```

**代码逻辑：** `maybe_to_bigram_view` 同步调整 key 与 value；`page_aligned` 可能把尾部未满页 token 去掉；`value[:len(key)]` 保证 value 与可缓存 key 等长。`_insert_helper` 返回已命中的前缀长度和插入后的最后 device 节点。

**为什么这样写：** cache 写入必须只写可以稳定复用的 KV 段。未对齐尾部、EAGLE 末尾 token 与调试场景如果混在 `_insert_helper` 内处理，会让树操作和内存语义纠缠；入口处归一化能把内部递归保持为纯 radix 结构操作。

**不变量与失败模式：** key/value 长度必须一致；`disable` 时不能修改树；`prefix_len` 必须只表示已经存在的前缀。若 value 没有按 aligned key 裁剪，后续 evict 会释放多余 slot；若 bigram 视图遗漏，EAGLE 的逻辑 token 与 KV index 会错一位。

**Comment：** `insert` 的核心作用是做边界归一化，让 `_insert_helper` 只关心 radix 树怎么拆，不关心 serving 模式差异。

### 1.4 `cache_unfinished_req`：chunk 中间态的 rematch 与锁迁移

来源：python/sglang/srt/mem_cache/radix_cache.py L488-L552

**问题与约束：** chunked prefill 或 streaming 请求尚未结束时，当前已填充 token 的 KV 既要进入 prefix cache，又要继续被该请求使用。insert 可能改变树节点与 canonical indices，旧的 `req_to_token_pool` 映射不能直接复用。

**设计选择：** 先基于 `fill_ids` 与当前 KV indices 构造 aligned `RadixKey`，调用 `insert`；释放 duplicate 段后重新 `match_prefix`，把 canonical indices 写回请求表，并把请求锁从旧 `last_node` 迁移到新节点。

**Explain：** 这里的关键是“写完再查一次”。Radix 树 split 之后，真正被 cache 持有的 KV indices 可能不再是请求原先分配的那批；rematch 把请求切换到 cache 的规范副本，避免同一前缀占两份物理内存。

**Code：**

```python
    def cache_unfinished_req(self, req: Req, chunked=False):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        token_ids = req.get_fill_ids()
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        values = kv_indices[: len(radix_key)].to(dtype=torch.int64, copy=True)

        # Radix Cache takes one ref in memory pool
        result = self.insert(
            InsertParams(
                key=radix_key,
                value=values,
                chunked=chunked,
                priority=getattr(req, "priority", 0) or 0,
            )
        )
        new_prefix_len = result.prefix_len

        self.token_to_kv_pool_allocator.free(
            kv_indices[req.cache_protected_len : new_prefix_len]
        )

        # The prefix indices could be updated, reuse it
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices, new_last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )
        assert len(new_indices) == len(
            radix_key
        ), f"{len(new_indices)=}, {len(radix_key)=}"

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        # The cache_protected_len is not always equal to len(req.prefix_indices)
        # since for page_size > 1, the partial part is added to req.prefix_indices, but that part of kv indices is not added to the tree.
        # It should be freed in the next cache_unfinished_req and final cache_finished_req to avoid memory leak.
        # So we introduce this `cache_protected_len` field to make sure the partial part can be freed correctly.
        req.cache_protected_len = len(new_indices)

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        # - page_size != 1: there is a partial page at the end, keep the full kv_indices
        # - eagle case: bigram keys will only cache len - 1 kv indices
        if len(new_indices) < len(kv_indices):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices

        req.last_node = new_last_node
```

**代码逻辑：** 函数读取请求当前 fill 长度对应的 KV indices，生成 aligned `radix_key` 与独立拷贝的 `values`。`insert` 后用 `new_prefix_len` 释放 duplicate 区间；随后 `match_prefix` 得到新的 device indices，并写回 `req_to_token_pool`。最后更新 `cache_protected_len`、锁引用和 `prefix_indices`。

**为什么这样写：** unfinished 请求既是 cache 的生产者，也是后续 decode/chunk 的消费者。只有 rematch 后切换到 canonical cache indices，才能避免重复 KV 常驻；只有锁迁移同步更新，evict 才不会回收仍被请求使用的路径。

**不变量与失败模式：** `len(new_indices)` 必须等于 aligned key 长度；`cache_protected_len` 表示已经由 cache 持有且不应重复 free 的长度；page 未对齐尾部可出现在 `prefix_indices`，但不能进入树。若忘记释放 duplicate 段会泄漏 KV；若忘记 rematch 会让请求继续指向已被释放或非 canonical 的 indices。

**Comment：** 这段把“缓存中间 chunk”变成一个原子语义：写树、释放重复、重绑定请求、迁移锁。

### 1.5 `RadixCache.evict`：按叶节点整段回收

来源：python/sglang/srt/mem_cache/radix_cache.py L563-L590

**问题与约束：** KV pool 不足时需要释放 prefix cache，但只能释放没有活跃请求锁定的叶子；释放单位是树节点 value 段，实际释放 token 数可能超过目标值。

**设计选择：** 从 `evictable_leaves` 取快照，用 eviction strategy 的 priority 建堆；每次弹出一个叶子，释放其 value 并删除叶子。如果父节点因此变成无锁叶子，就把父节点重新入堆。

**Explain：** Radix cache 的可释放对象天然是叶节点，因为内部节点仍可能承载其他分支前缀。删除叶子后向上收缩，可以一次驱逐释放一条已经没有分支依赖的路径。

**Code：**

```python
    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        leaves = list(self.evictable_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            self.token_to_kv_pool_allocator.free(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

            self._record_remove_event(x)

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)
```

**代码逻辑：** 函数在开始时复制可驱逐叶集合并建堆。循环中释放节点 value，累计 `num_evicted`，从树里删除叶子；若父节点没有 children 且未被锁定，父节点也成为可驱逐候选。结束时记录指标并返回释放量。

**为什么这样写：** evict 必须避免扫描整棵树，也不能破坏仍被其他 key 共享的内部节点。叶集合加 heap 让策略可插拔，父节点回堆让树在释放后自然压缩。

**不变量与失败模式：** `evictable_leaves` 中节点应满足 `lock_ref == 0`；被释放 value 不能再被请求表引用；返回释放量允许大于请求量。若内部节点被错误加入 heap，会破坏其他前缀；若锁状态更新滞后，会回收正在 decode 的 KV。

**Comment：** classic evict 的粒度是 radix leaf，不是 token；这解释了为什么调度侧必须按“至少释放多少”而不是“精确释放多少”来理解结果。

---

## 2. UnifiedRadixCache 的多 component 扩展

### 2.1 `UnifiedLRUList`：同一节点上的 device/host 指针隔离

来源：python/sglang/srt/mem_cache/unified_radix_cache.py L136-L151

**问题与约束：** Unified cache 在同一 `UnifiedTreeNode` 上同时管理 BASE、SWA、MAMBA 等 component，还可能同时维护 device LRU 与 host LRU。多个链表若共用同一指针字段，会互相覆盖。

**设计选择：** `UnifiedLRUList` 根据 `component_type` 与 `use_host_ptr` 计算 `_pt`，host 链表使用 `_NUM_COMPONENT_TYPES` 之后的 slot，device 与 host 指针在同一 node 上分区存放。

**Explain：** 这里把“一个 node 多条 LRU 链”的问题转成数组 slot 问题。每条链表只读写自己的 `lru_prev/lru_next[_pt]`，所以 device evict、host write-back 与不同 component 的 LRU 操作可以共享节点对象。

**Code：**

```python
class UnifiedLRUList:
    def __init__(
        self,
        component_type: ComponentType,
        tree_components: tuple[ComponentType, ...],
        use_host_ptr: bool = False,
    ):
        self.component_type = component_type
        # Pointer slot: host LRU uses offset slots so device/host pointers
        # never collide on the same node.
        self._pt: int = component_type + (_NUM_COMPONENT_TYPES if use_host_ptr else 0)
        self.head = UnifiedTreeNode(tree_components)
        self.tail = UnifiedTreeNode(tree_components)
        self.head.lru_next[self._pt] = self.tail
        self.tail.lru_prev[self._pt] = self.head
        self.cache: dict[int, UnifiedTreeNode] = {}
```

**代码逻辑：** 构造函数确定本链表对应的 component 与 pointer slot，创建哨兵 head/tail，并只在 `_pt` 位置连接双向链。`cache` 用节点 id 到节点对象的映射辅助 O(1) 查询。

**为什么这样写：** Unified cache 不能为每个 component 复制一棵 radix 树，否则 prefix 结构、锁和 HiCache 状态会分裂。共享树节点再隔离 LRU 指针，可以在同一拓扑上维护多个替换策略视图。

**不变量与失败模式：** 每条 LRU list 必须只操作自己的 `_pt`；head/tail 也是带相同 component 布局的 `UnifiedTreeNode`。若 host 与 device pointer slot 冲突，某一条链表的删除或移动会破坏另一条链表，表现为重复 evict、漏 evict 或循环链。

**Comment：** Unified 的核心不是“更大的 RadixCache”，而是在同一前缀树上叠加多套资源视图；LRU slot 是这种设计的底层铺垫。

### 2.2 `UnifiedRadixCache.match_prefix`：session 短路与全局树匹配

来源：python/sglang/srt/mem_cache/unified_radix_cache.py L561-L586

**问题与约束：** Streaming session 可能维护私有匹配视图，而普通请求需要走全局 unified radix 树。匹配还要兼容 EAGLE bigram、page 对齐、空 key，以及 HiCache 的 host 命中后处理。

**设计选择：** 入口先调用 `session.try_match_prefix`，命中则直接返回；否则归一化 key，处理 disable/空 key，再走 `_match_prefix_helper`，最后统一交给 `_match_post_processor`。

**Explain：** session 是比全局树更靠近请求生命周期的一层状态。把 session 短路放在最前面，可以避免 streaming 场景与全局树锁和 host/device 迁移逻辑互相干扰。

**Code：**

```python
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        result = self.session.try_match_prefix(params)
        if result is not None:
            return result

        key = params.key
        key, _ = key.maybe_to_bigram_view(self.is_eagle)
        if self.disable or len(key) == 0:
            return self._empty_match_result
        key = key.page_aligned(self.page_size)
        if len(key) == 0:
            return self._empty_match_result

        (
            value,
            best_match_node,
            best_match_device_node,
            best_match_device_value_len,
        ) = self._match_prefix_helper(key)
        return self._match_post_processor(
            params,
            value,
            best_match_node,
            best_match_device_node,
            best_match_device_value_len,
        )
```

**代码逻辑：** 函数先问 session 是否能直接产出 `MatchResult`。未命中时，把 key 转 bigram 视图并做 disable/空判断；page 对齐后再次处理空 key；最后 helper 返回 value、最佳逻辑节点、最佳 device 节点与 device value 长度，由 post processor 转成最终结果。

**为什么这样写：** Unified cache 的 match 结果不只是 device indices，还可能涉及 host 回读、prefetch 或多 component 状态。把 helper 保持为树匹配，把 post processor 负责存储层动作，可以避免匹配算法和 HiCache 行为耦合。

**不变量与失败模式：** disable 或空 key 必须返回空结果；page 对齐后长度可能变成 0；session 命中时不能继续访问全局树。若先对全局树做锁或 host 操作再处理 session，会让 streaming 私有状态被全局 cache 副作用污染。

**Comment：** 这一入口展示了 Unified 的层次划分：session 覆盖请求局部性，helper 负责 radix 匹配，post processor 负责多级存储语义。

### 2.3 `UnifiedRadixCache.evict`：每个 component 自己驱动回收

来源：python/sglang/srt/mem_cache/unified_radix_cache.py L604-L624

**问题与约束：** Unified cache 同时管理 BASE KV、SWA、MAMBA 等资源，不同 component 的容量单位和可驱逐集合不同；HiCache write-back 策略还要求 device evict 前后同步 host 状态。

**设计选择：** 为 `tree_components` 初始化 tracker，让每个 component 调用自己的 `drive_eviction`；若 cache controller 是 write-back，则执行 `writing_check(write_back=True)`；最后按 component 返回分项驱逐量。

**Explain：** classic cache 用单个 heap 驱动所有叶节点；Unified 不能这么做，因为一个节点可能在 BASE 上可驱逐、在 SWA 或 MAMBA 上却有不同资源语义。component 自驱动让各自的 LRU、锁与计量逻辑保持独立。

**Code：**

```python
    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()
        start_time = time.perf_counter()
        tracker = {ct: 0 for ct in self.tree_components}

        for component in self._components_tuple:
            component.drive_eviction(params=params, tracker=tracker)

        if (
            self.cache_controller is not None
            and self.cache_controller.write_policy == "write_back"
        ):
            self.writing_check(write_back=True)

        self.update_eviction_metrics(sum(tracker.values()), start_time)
        return EvictResult(
            num_tokens_evicted=tracker[BASE_COMPONENT_TYPE],
            swa_num_tokens_evicted=tracker.get(ComponentType.SWA, 0),
            mamba_num_evicted=tracker.get(ComponentType.MAMBA, 0),
        )
```

**代码逻辑：** `tracker` 以 component type 为 key 统计释放量。循环把同一 `EvictParams` 交给每个 component，component 内部决定释放哪些节点与数量。write-back 模式下调用 `writing_check`，最后把 BASE、SWA、MAMBA 的释放量拆到 `EvictResult`。

**为什么这样写：** Scheduler 需要知道不同资源分别释放了多少，而不是一个混合 token 数。component 驱动也让后续新增资源类型时只扩展 component，不必改 unified cache 的主流程。

**不变量与失败模式：** `tracker` 必须覆盖所有启用 component；BASE 释放量仍填入 `num_tokens_evicted` 以兼容旧接口；write-back 时不能丢失 host 副本一致性。若把所有 component 合成一个 heap，SWA/MAMBA 的容量预算会被 BASE token 数误导。

**Comment：** Unified evict 的返回值本身就是设计信号：多资源 cache 需要分项预算，不能再只看一个 KV token 计数。

### 2.4 `UnifiedRadixCache.inc_lock_ref`：多 component 锁保护

来源：python/sglang/srt/mem_cache/unified_radix_cache.py L626-L637

**问题与约束：** 活跃请求命中的节点需要从所有相关 component 的可驱逐集合中移除；Streaming session 可能有自己的锁语义，不能强制落到全局树。

**设计选择：** 入口先让 session 尝试处理；未命中且 cache 未禁用时，为每个 component 调用 `acquire_component_lock`，再刷新 node 对应的 evictable leaf 集合。

**Explain：** Classic cache 的 `lock_ref` 是单资源视角；Unified 中同一节点可能挂多个 component value。只有每个 component 都 acquire，节点才真正对 evict 不可见。

**Code：**

```python
    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        result = self.session.try_inc_lock_ref(node)
        if result is not None:
            return result
        if self.disable:
            return IncLockRefResult()
        result = IncLockRefResult()
        for component in self._components_tuple:
            result = component.acquire_component_lock(node=node, result=result)

        self._update_evictable_leaf_sets(node)
        return result
```

**代码逻辑：** session 若返回结果，函数直接结束。普通路径下先构造空 `IncLockRefResult`，依次把结果对象传给 component 累积锁变化；所有 component 处理后，根据节点最新状态刷新可驱逐叶集合。

**为什么这样写：** `inc_lock_ref` 是请求生命周期与 evict 的交界点。统一在这里刷新 evictable sets，可以把锁状态变化立即反映给驱逐器，避免 component 自己改锁后集合状态滞后。

**不变量与失败模式：** session 处理过的节点不能重复全局 acquire；disable 时不能改变集合；component acquire 需要在结果对象里记录释放/保护数量变化。若只锁 BASE component，SWA 或 MAMBA 仍可能被驱逐，导致请求读到不完整的前缀状态。

**Comment：** Unified 的锁不是一个布尔保护位，而是“所有启用资源视图都不可驱逐”的联合条件。

---

## 3. RadixAttention 的 kernel 接口

### 3.1 `AttentionType`：用字符串枚举固定 mask 语义

来源：python/sglang/srt/layers/radix_attention.py L43-L54

**问题与约束：** Attention 层要区分 decoder、图像 token 双向 decoder、encoder-only 等 mask 语义；这些 tag 还会穿过 `torch.compile`，枚举值必须可稳定序列化与比较。

**设计选择：** 用 `Enum` 承载 attention 类型，但枚举值使用字符串而不是整数。

**Explain：** 字符串枚举把语义直接写进值里，编译路径也能把它当稳定常量处理。这样后端收到的是明确的 attention 类型，而不是容易和别的整数配置混淆的数字。

**Code：**

```python
class AttentionType(Enum):
    """
    Attention type.
    Use string to be compatible with `torch.compile`.
    """

    # Decoder attention between previous layer Q/K/V
    DECODER = "decoder"
    # Decoder bidirectional attention between image tokens
    DECODER_BIDIRECTIONAL = "decoder_bidirectional"
    # Encoder attention between previous layer Q/K/V
    ENCODER_ONLY = "encoder_only"
```

**代码逻辑：** 类定义三个当前使用的 attention 类型，docstring 明确字符串选择是为了 `torch.compile` 兼容。每个成员值都是后续逻辑可直接传递的字符串 tag。

**为什么这样写：** attention mask 错误会直接变成模型行为错误。把类型做成可读字符串并集中定义，可以降低跨 backend、跨图编译路径传参时的歧义。

**不变量与失败模式：** 枚举值必须与后端识别的 attention type 对齐；新增 mask 语义时需要同步 backend。若换成普通整数或散落字符串，编译缓存与后端分支更容易出现不可见的不一致。

**Comment：** 这是一个小接口，但它固定了 attention 语义进入 kernel 前的名字空间。

### 3.2 `RadixAttention.forward`：Q/K/V 的 kernel layout 归一化

来源：python/sglang/srt/layers/radix_attention.py L118-L125

**问题与约束：** 模型层输出通常是 flat tensor，而 attention backend 需要 `[tokens, heads, head_dim]` 形状；cross-layer KV sharing 时 `k/v` 可能为空，MLA 路径又可能通过 `k_rope` 使用不同 layout。

**设计选择：** 只在 `k is not None` 时 reshape K/V；普通路径按 `qk_head_dim` 与 `v_head_dim` reshape，带 `k_rope` 时 K 按 value head dim reshape，避免破坏 MLA 分离 layout。

**Explain：** `RadixAttention` 在这里承担模型实现与 attention backend 之间的 shape adapter。它不负责生成 K/V，只保证已有 K/V 在进入 kernel 前具有 backend 约定形状。

**Code：**

```python
        if k is not None:
            # For cross-layer sharing, kv can be None
            assert v is not None
            if "k_rope" not in kwargs:
                k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
                v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
            else:
                k = k.view(-1, self.tp_k_head_num, self.v_head_dim)
```

**代码逻辑：** 函数先允许 `k` 为空；一旦 `k` 存在，就要求 `v` 也存在。普通路径把 K reshape 到 `tp_k_head_num * qk_head_dim`，V reshape 到 `tp_v_head_num * v_head_dim`；有 `k_rope` 时，K 使用 `v_head_dim`。

**为什么这样写：** cross-layer KV sharing 需要 attention 层能只消费 cache 中已有 KV；MLA 则把 rope 相关 K 与 value-like latent 维度拆开。显式分支可以让同一个 `RadixAttention.forward` 支持多种模型结构，而不把 layout 特例下沉到 kernel。

**不变量与失败模式：** `k` 存在时 `v` 必须存在；reshape 前 tensor 总元素数必须匹配 head 配置；`k_rope` 路径不能误用 `qk_head_dim`。若 layout 选错，错误不一定在 view 处暴露，可能在 kernel 中变成 head 维错读。

**Comment：** 这段不是性能优化，而是 ABI 适配：把模型侧 tensor 形态转换成 attention backend 统一可读的形态。

### 3.3 `unified_attention_with_output`：声明有副作用的自定义 op

来源：python/sglang/srt/layers/radix_attention.py L156-L158

**问题与约束：** Unified attention 会把结果写入外部传入的 `output` tensor，图编译器必须知道这个 op 会 mutate 参数；同时 piecewise 编译需要识别该 op 可作为拆分边界。

**设计选择：** 在函数上同时标记 `@register_custom_op(mutates_args=["output"])` 与 `@register_split_op()`。

**Explain：** 对图编译而言，attention kernel 不是纯函数。显式声明 `output` 被修改，可以避免编译器错误地重排或消除写入；split op 标记则让 attention 成为可独立处理的图节点。

**Code：**

```python
@register_custom_op(mutates_args=["output"])
@register_split_op()
def unified_attention_with_output(
```

**代码逻辑：** 两个 decorator 在函数定义前注册元信息：custom op 描述 side effect，split op 描述图拆分能力。函数体仍按普通 Python 定义执行，注册信息供编译与调度侧读取。

**为什么这样写：** attention 是 serving 中最重的算子之一，常需要被图编译、捕获或替换为 backend kernel。把副作用和拆分属性写在定义处，能让调用点保持简洁，同时减少编译器对函数体的猜测。

**不变量与失败模式：** `mutates_args` 必须准确列出被写参数；如果漏掉 `output`，图编译可能复用旧值或错误重排；如果误标纯输入为 mutable，会限制优化空间并引入不必要的依赖边。

**Comment：** 这三行定义了 `unified_attention_with_output` 在图系统里的身份：它是一个会写 output 的 attention 边界 op。

### 3.4 DeepSeek MLA：MHA companion 修正 metadata

来源：python/sglang/srt/layers/radix_attention.py L188-L193

**问题与约束：** DeepSeek MLA 每层有 `attn_mqa` 与 `attn_mha` 两个 `RadixAttention` 实例共享 `layer_id`，但 `attention_layers` 列表只存 `attn_mqa`。当 MHA 路径执行且不保存 KV cache 时，backend 需要 MHA 的 head/dim metadata。

**设计选择：** 在 HIP 且 `save_kv_cache=False`、attention layer 带 `_pcg_mha_companion` 时，把当前 attention layer 替换为 companion。

**Explain：** layer id 相同不代表 head metadata 相同。MHA replay 路径如果继续使用 MQA 实例，kernel 会拿到错误的头数或维度配置；companion 指针提供了共享 layer id 下的正确 metadata。

**Code：**

```python
    # DeepSeek MLA has two RadixAttention instances per layer (attn_mqa and
    # attn_mha) that share the same layer_id. The attention_layers list only
    # stores attn_mqa. When the MHA path is active (save_kv_cache=False), use
    # the companion attn_mha so the backend sees correct head/dim metadata.
    if _is_hip and not save_kv_cache and hasattr(attention_layer, "_pcg_mha_companion"):
        attention_layer = attention_layer._pcg_mha_companion
```

**代码逻辑：** 分支同时检查平台、是否保存 KV cache、是否存在 companion 属性；满足条件时用 companion 覆盖 `attention_layer` 变量，后续 backend 调用读取 companion 上的 metadata。

**为什么这样写：** 这是对 DeepSeek MLA 双实例结构的局部适配。与其改全局 `attention_layers` 存储结构，不如在唯一需要 MHA metadata 的路径做替换，降低对其他模型与平台的影响。

**不变量与失败模式：** 只有 HIP PCG MHA 路径需要替换；companion 必须与原实例共享语义 layer id 但携带 MHA 配置。若条件放宽，会让普通 MQA 路径误用 MHA metadata；若条件缺失，MHA kernel 会按 MQA 配置解释 tensor。

**Comment：** 这段体现 RadixAttention 的另一个职责：在统一 attention 接口里消化模型结构和平台 backend 的局部差异。

### 3.5 HIP padding zero：清理 PCG replay 的未写输出

来源：python/sglang/srt/layers/radix_attention.py L235-L252

**问题与约束：** AMD PCG replay 中 varlen attention kernel 只写实际 token 区间，padding 出来的静态 token 位置可能保留 `torch.empty` 的未初始化值；这些 NaN/Inf 会沿 residual、MoE routing、allreduce 扩散。

**设计选择：** 仅在 HIP 路径检查 `context.num_tokens` 与 `context.raw_num_tokens`，当静态 token 数大于实际 token 数时，把 output 中实际 token 之后的所有位置清零。

**Explain：** PCG 为了图复用会把 token 数 pad 到静态形状，但 kernel 的有效工作量仍是 raw token 数。补零把“未定义输出”变成确定的零输出，避免后续层把 padding 垃圾当真实 hidden state 消费。

**Code：**

```python
    if _is_hip:
        # During PCG replay on AMD, varlen attention kernels only fill positions
        # 0..actual_tokens-1 and leave padded positions with uninitialized
        # garbage from torch.empty.  Zero these so garbage (NaN/Inf) does not
        # propagate through residual connections, MoE routing, and allreduce.
        # Use context.raw_num_tokens (pre-padding count from PCG runner)
        # instead of forward_batch.extend_num_tokens, because
        # extend_num_tokens is None for TARGET_VERIFY (EAGLE) batches.
        pcg_static_tokens = context.num_tokens
        actual_tokens = context.raw_num_tokens
        if (
            pcg_static_tokens is not None
            and actual_tokens is not None
            and pcg_static_tokens > actual_tokens
        ):
            first_dim = output.shape[0]
            elems_per_token = output.numel() // first_dim
            output.view(first_dim, elems_per_token)[actual_tokens:].zero_()
```

**代码逻辑：** 分支只在 `_is_hip` 时进入。函数读取 PCG 静态 token 数与 raw token 数，确认存在 padding 后，将 output 展平为 `[tokens, elems_per_token]`，从 `actual_tokens` 之后整段置零。

**为什么这样写：** 使用 `context.raw_num_tokens` 而不是 batch 的 extend token 数，是因为 EAGLE `TARGET_VERIFY` 场景下后者可能为空。按 token 第一维展平可以覆盖所有 head/hidden 维度，不依赖具体 output rank。

**不变量与失败模式：** `output.shape[0]` 必须对应 token 维；只有 `pcg_static_tokens > actual_tokens` 才需要清理；清理范围不能包含真实 token。若遗漏补零，未初始化值可能在后续规约中放大；若清理范围过宽，会抹掉真实 attention 输出。

**Comment：** 这是平台相关的正确性补丁：它不改变 attention 数学，只把 PCG padding 的未定义区域变成显式零。
