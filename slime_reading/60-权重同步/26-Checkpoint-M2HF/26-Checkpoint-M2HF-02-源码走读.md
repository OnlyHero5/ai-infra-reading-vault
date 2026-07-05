---
type: batch-doc
module: 26-Checkpoint-M2HF
batch: "26"
doc_type: walkthrough
title: "Checkpoint M2HF · 源码走读"
tags:
  - slime/batch/26
  - slime/module/checkpoint-m2hf
  - slime/doc/walkthrough
updated: 2026-07-05
---

# Checkpoint M2HF · 源码走读

> 走读主线：Slime 在训练运行时同时支持从 Megatron checkpoint 或 HF checkpoint 加载模型；保存时 actor 可额外导出 HF checkpoint。Bridge 路径交给 Megatron Bridge，raw 路径自己把并行 Megatron 参数 all-gather、转换成 HF 命名并分布式写 safetensors shard。

---

## 1. 加载侧：Megatron checkpoint 与 HF checkpoint 分流

### 1.1 checkpoint.py 启动时 patch ShardedTensor 校验以降低大模型加载开销

问题与约束：
- Megatron 分布式 checkpoint 加载会创建大量 ShardedTensor metadata；大模型多 shard 场景下跨 rank non-overlap 校验很慢。

设计选择：
- 模块 import 时 patch `EnumerableShardingSpec.__post_init__` 和 `ShardedTensor._init_from_local_shards_and_global_metadata`，跳过部分重校验，依赖调用方保证 metadata 正确。

Explain：
patch 后 `_init_from_local_shards_and_global_metadata` 只从 global metadata 中筛出当前 rank 的 shard metadata，推断 sharding spec，构造 ShardedTensor 并调用 `_prepare_init/_post_init`，不再做原始慢校验。

来源：slime/backends/megatron_utils/checkpoint.py L13-L88

Code：

```python
def __post_init__(self):
    pass

EnumerableShardingSpec.__post_init__ = __post_init__

@classmethod
def _init_from_local_shards_and_global_metadata(
    cls,
    local_shards: list[Shard],
    sharded_tensor_metadata: ShardedTensorMetadata,
    process_group=None,
    init_rrefs=False,
    sharding_spec=None,
) -> ShardedTensor:
    process_group = cls._normalize_pg(process_group)
    current_rank = dist.get_rank()
    shards_metadata = sharded_tensor_metadata.shards_metadata
    local_shard_metadatas = []
    for shard_metadata in shards_metadata:
        rank, local_device = _parse_and_validate_remote_device(process_group, shard_metadata.placement)
        if current_rank == rank:
            local_shard_metadatas.append(shard_metadata)
    if sharding_spec is None:
        spec = shard_spec._infer_sharding_spec_from_shards_metadata(shards_metadata)
    else:
        spec = sharding_spec
    sharded_tensor = ShardedTensor.__new__(...)
    sharded_tensor._local_shards = local_shards
    sharded_tensor._prepare_init(process_group=process_group, init_rrefs=init_rrefs)
    sharded_tensor._post_init()
    return sharded_tensor
```

代码逻辑：
- import 成功时直接 monkey patch PyTorch sharded tensor 类。
- patch 只在 ImportError 之外执行。
- 当前 rank 只收集属于自己的 shard metadata。
- ShardedTensor 初始化保留 `_prepare_init/_post_init`。

为什么这样写：
- 大模型 checkpoint metadata 校验代价高，训练启动和加载路径需要压缩耗时。
- patch 放在 checkpoint 模块 import 时，能覆盖后续 Megatron checkpoint load。

不变量与失败模式：
- 代码假设 checkpoint metadata 本身正确；如果 shard overlap 或 placement 错误，patch 可能让问题延后暴露。
- PyTorch 内部 API 变化会让 monkey patch 失效。

Comment：
这是加载性能优化，不改变 Slime 自己的 checkpoint 格式判断逻辑。

### 1.2 load_checkpoint 按目录形态分流 Megatron 与 HF

问题与约束：
- `args.load` 既可能指向 Megatron dist checkpoint，也可能指向 HuggingFace 模型目录；训练初始化需要一个统一入口。

设计选择：
- `load_checkpoint` 先检查路径存在且非空，再用 `_is_megatron_checkpoint` 判断；Megatron 分支透传 upstream loader，HF 分支走 Slime bridge loader。

Explain：
函数签名保持 Megatron `load_checkpoint` 兼容，包括 ddp_model、optimizer、scheduler、checkpointing_context 和 skip 标志。HF 分支不恢复 optimizer/scheduler，而是只把 HF 权重灌入模型。

来源：slime/backends/megatron_utils/checkpoint.py L97-L120

Code：

```python
def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, checkpointing_context, skip_load_to_model_and_opt):
    args = get_args()
    load_path = args.load

    assert Path(load_path).exists() and _is_dir_nonempty(
        load_path
    ), f"{args.load=} does not exist or is an empty directory. Did you specify the wrong folder?"

    if _is_megatron_checkpoint(load_path):
        return _load_checkpoint_megatron(
            ddp_model=ddp_model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=skip_load_to_model_and_opt,
        )
    else:
        return _load_checkpoint_hf(
            ddp_model=ddp_model,
            optimizer=optimizer,
            args=args,
            load_path=load_path,
        )
```

代码逻辑：
- 从 Megatron global args 读取 load path。
- 路径必须存在且目录非空。
- Megatron checkpoint 使用 upstream loader。
- 非 Megatron 目录按 HF checkpoint 处理。

为什么这样写：
- 保持 Megatron 初始化代码只调用一个 `load_checkpoint`。
- 用户可以用同一个 `--load` 参数加载训练 checkpoint 或 HF 初始权重。

不变量与失败模式：
- HF 目录若被误命名成 Megatron iter 目录形式，会被当成 Megatron checkpoint。
- 空目录会直接 assert，避免后续 loader 给出更隐晦错误。

Comment：
这条分流决定训练启动时是恢复训练态，还是从 HF 权重初始化模型态。

### 1.3 Megatron checkpoint 判据只看 tracker 或 iter 目录名

问题与约束：
- Megatron checkpoint 有固定目录约定，但 HF 目录也可能包含大量文件；分流规则需要简单、稳定。

设计选择：
- `_is_megatron_checkpoint` 判断目录下是否有 `latest_checkpointed_iteration.txt`，或目录名是否匹配 `iter_\d{7}`。

Explain：
这是路径级别判据，不读取 checkpoint 内容。release 根目录一般通过 tracker 文件识别；直接传某个 iter 子目录时通过目录名识别。

来源：slime/backends/megatron_utils/checkpoint.py L123-L126

Code：

```python
def _is_megatron_checkpoint(path: str | Path) -> bool:
    return (Path(path) / "latest_checkpointed_iteration.txt").is_file() or bool(
        re.fullmatch(r"iter_\d{7}", Path(path).name)
    )
```

代码逻辑：
- tracker 文件存在即 Megatron checkpoint。
- 或者当前目录名是 iter 七位数字。
- 其他目录都走 HF 加载分支。

为什么这样写：
- 避免为每次启动读取重 checkpoint metadata。
- 兼容 Megatron 根目录和 iter 子目录两种常用传参。

不变量与失败模式：
- HF 目录不应命名为 `iter_0000001` 这类格式。
- Megatron checkpoint 根目录必须保留 tracker 文件。

Comment：
这个判据很轻量，也意味着目录命名会影响加载路径。

### 1.4 HF 加载只支持 bridge 模式，并在半精度下刷新 optimizer master params

问题与约束：
- HF checkpoint 没有 Megatron optimizer/rng/scheduler 状态；如果模型已被 optimizer 包装，半精度训练还需要让 optimizer 的 master params 同步到新权重。

设计选择：
- `_load_checkpoint_hf` 要求 `args.megatron_to_hf_mode == "bridge"`，用 Megatron Bridge 的 AutoBridge 加载 HF 权重；加载后如 fp16/bf16 且 optimizer 存在，调用 `optimizer.reload_model_params()`。

Explain：
Bridge 加载前用 `patch_megatron_model(ddp_model)` 适配 Slime 的模型包装；AutoBridge 也通过 `patch_auto_bridge_hf_config` 修补 HF config。返回 iteration 固定为 0，FLOPs 也从 0 开始。

来源：slime/backends/megatron_utils/checkpoint.py L129-L152

Code：

```python
def _load_checkpoint_hf(ddp_model, optimizer, args, load_path: str):
    assert args.megatron_to_hf_mode == "bridge", "Only bridge mode is supported for loading HF checkpoint"
    from megatron.bridge import AutoBridge

    import slime_plugins.megatron_bridge

    with megatron_bridge_utils.patch_megatron_model(ddp_model):
        bridge = megatron_bridge_utils.patch_auto_bridge_hf_config(
            AutoBridge.from_hf_pretrained(load_path, trust_remote_code=True)
        )
        bridge.load_hf_weights(ddp_model)

    if (args.fp16 or args.bf16) and optimizer is not None:
        assert not args.load_main_params_from_ckpt
        optimizer.reload_model_params()

    iteration = 0
    num_floating_point_operations_so_far = 0
    return iteration, num_floating_point_operations_so_far
```

代码逻辑：
- HF load 只允许 bridge 模式。
- 插件 import 触发 Megatron Bridge 扩展注册。
- patch 模型和 HF config 后加载权重。
- 半精度 optimizer 刷新 master params。
- 返回 iteration 0。

为什么这样写：
- HF checkpoint 只能提供模型权重，不能恢复训练步数和 optimizer 状态。
- optimizer master params 如果不刷新，会和模型参数不一致。

不变量与失败模式：
- raw mode 不支持 HF 加载，只支持导出保存。
- `load_main_params_from_ckpt` 与 HF 初始化冲突，因此半精度时 assert。

Comment：
HF load 是初始化权重路径，不是训练态恢复路径。

### 1.5 model.py 在 setup_model_and_optimizer 后统一调用 load_checkpoint

问题与约束：
- 模型、optimizer 和 scheduler 必须先构造出来，checkpoint loader 才能把权重和可选 optimizer state 写入正确对象。

设计选择：
- Megatron model setup 完成后调用 Slime 的 `load_checkpoint`，再按 critic 需要重初始化输出层，并在必要时 reload optimizer params。

Explain：
ROCm writer patch 也在这里执行。`load_checkpoint` 的返回 iteration 会作为训练恢复步数返回给调用方；HF load 分支因此返回 0，Megatron 分支则由 upstream loader 返回真实 iteration。

来源：slime/backends/megatron_utils/model.py L982-L1007

Code：

```python
if torch.version.hip:
    import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module
    from slime.utils.rocm_checkpoint_writer import ROCmFileSystemWriterAsync
    filesystem_async_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync

model, optimizer, opt_param_scheduler = setup_model_and_optimizer(args, role)
model[0].role = role
reinit_critic_output_layer = _critic_output_layer_needs_reinit(args, model, role)
clear_memory()
iteration, _ = load_checkpoint(
    model,
    optimizer,
    opt_param_scheduler,
    checkpointing_context={},
    skip_load_to_model_and_opt=False,
)
if reinit_critic_output_layer:
    _reinitialize_critic_output_layer(args, model)
    if (args.fp16 or args.bf16) and optimizer is not None:
        optimizer.reload_model_params()
clear_memory()

return model, optimizer, opt_param_scheduler, iteration
```

代码逻辑：
- 先构造模型与优化器。
- 再调用统一 checkpoint loader。
- critic 输出层可在加载后重初始化。
- 返回 model/optimizer/scheduler/iteration。

为什么这样写：
- loader 需要已有对象承接权重和状态。
- critic 输出层可能不应从 actor checkpoint 继承，所以放在 checkpoint load 后处理。

不变量与失败模式：
- `args.load` 必须已经配置。
- reinit 后半精度 optimizer 也要 reload model params。

Comment：
这说明 checkpoint 分流不是独立工具，而是 Megatron 模型初始化的一部分。

### 1.6 Actor.save 在保存 Megatron checkpoint 后可额外导出 HF checkpoint

问题与约束：
- 训练过程中常规保存仍要走 Megatron checkpoint；但用户可能希望在某些 rollout step 同时导出 HF checkpoint 供推理或发布。

设计选择：
- Actor save 先处理 offload/async save，再调用 Megatron `save`；如果 `args.save_hf` 不为空且 role 是 actor，则调用 `save_hf_model_to_path` 额外导出 HF。

Explain：
`args.save_hf` 是 format string，使用 rollout id 填充输出目录。只有 actor role 导出 HF，避免 critic/reference 等非目标模型误导出。

来源：slime/backends/megatron_utils/actor.py L561-L578

Code：

```python
if self.args.offload_train:
    self.wake_up()

if self.args.async_save:
    from megatron.training.async_utils import maybe_finalize_async_save

    maybe_finalize_async_save(blocking=True)

save(rollout_id, self.model, self.optimizer, self.opt_param_scheduler)

if force_sync and self.args.async_save:
    maybe_finalize_async_save(blocking=True)

if self.args.save_hf is not None and self.role == "actor":
    save_hf_model_to_path(self.args, Path(self.args.save_hf.format(rollout_id=rollout_id)), self.model)
```

代码逻辑：
- offload_train 时保存前 wake up。
- async save 前后可强制 finalize。
- Megatron checkpoint 始终先保存。
- actor role 才执行 HF 导出。

为什么这样写：
- Megatron checkpoint 是训练恢复的主路径，HF 导出是附加产物。
- HF 导出可能触发通信和磁盘 IO，放在常规 save 之后不影响恢复 checkpoint 语义。

不变量与失败模式：
- `args.save_hf` 必须能 format rollout_id。
- HF 导出失败不会在这段代码中被吞掉，会让 actor save 失败。

Comment：
M2HF 保存路径是训练 actor 的可选副作用，不替代 Megatron save。

### 1.7 load_other_checkpoint 用同一 loader 加载 ref/teacher 等辅助模型

问题与约束：
- 训练时可能需要把 reference 或 teacher checkpoint 加载到当前模型，再备份到指定 tag；但不应恢复其 optimizer/rng。

设计选择：
- 临时改写 `args.load/no_load_optim/no_load_rng/finetune`，可按 model_tag 调整 ckpt step；调用同一 `load_checkpoint`，再恢复旧 args 并备份权重。

Explain：
函数传入 optimizer/scheduler 为 None，因此只加载模型权重。ref/teacher 可通过专用 step 参数覆盖 `args.ckpt_step`，加载后 `weights_backuper.backup(model_tag)` 保存到内存备份体系。

来源：slime/backends/megatron_utils/actor.py L654-L681

Code：

```python
def load_other_checkpoint(self, model_tag: str, path: str) -> None:
    old_args = self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune
    self.args.load = path
    self.args.no_load_optim = True
    self.args.no_load_rng = True
    self.args.finetune = True

    old_ckpt_step = None
    if model_tag == "ref" and self.args.ref_ckpt_step is not None:
        old_ckpt_step = self.args.ckpt_step
        self.args.ckpt_step = self.args.ref_ckpt_step
    elif model_tag == "teacher" and self.args.opd_teacher_ckpt_step is not None:
        old_ckpt_step = self.args.ckpt_step
        self.args.ckpt_step = self.args.opd_teacher_ckpt_step

    _, _ = load_checkpoint(
        self.model,
        None,
        None,
        checkpointing_context={},
        skip_load_to_model_and_opt=False,
    )
    self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune = old_args
    if old_ckpt_step is not None:
        self.args.ckpt_step = old_ckpt_step
    self.weights_backuper.backup(model_tag)
```

代码逻辑：
- 临时切换 load 参数。
- 禁止 optimizer/rng 恢复。
- 调用统一 checkpoint loader。
- 恢复 args 后备份到 tag。

为什么这样写：
- ref/teacher 既可以来自 Megatron checkpoint，也可以来自 HF checkpoint，复用同一分流逻辑。
- 临时 args 修改把 Megatron loader 需要的全局参数喂进去，避免另写一套 loader。

不变量与失败模式：
- 函数异常时 args 恢复不在 finally 中，异常路径可能留下临时参数。
- 只加载模型权重，不能用于完整训练态恢复。

Comment：
这条路径把 M2HF loader 也用于辅助模型权重导入。

---

## 2. HF 保存侧：bridge 与 raw 两条路径

### 2.1 save_hf_model_to_path 以 megatron_to_hf_mode 分派保存实现

问题与约束：
- HF 导出可以交给 Megatron Bridge，也可以走 Slime 自己的 raw converter；两条路径依赖和能力不同。

设计选择：
- `args.megatron_to_hf_mode == "bridge"` 时调用 bridge saver，否则调用 direct saver，并透传 model_name、quantization_config 和 progress_desc。

Explain：
bridge 路径依赖 Megatron Bridge 对当前模型支持；raw 路径使用 Slime converter 和分布式 writer，更适合多节点分片写出和与 update_weight 逻辑复用。

来源：slime/backends/megatron_utils/hf_checkpoint_saver.py L22-L42

Code：

```python
def save_hf_model_to_path(
    args,
    output_dir: str | Path,
    model,
    *,
    model_name: str | None = None,
    quantization_config: dict[str, Any] | None = None,
    progress_desc: str = "Save HF checkpoint",
) -> None:
    if args.megatron_to_hf_mode == "bridge":
        save_hf_model_bridge_to_path(args, output_dir, model)
    else:
        save_hf_model_direct_to_path(
            args,
            output_dir,
            model,
            model_name=model_name,
            quantization_config=quantization_config,
            progress_desc=progress_desc,
        )
```

代码逻辑：
- 统一入口接收 args、输出目录和 Megatron model。
- bridge 模式忽略 raw-only 参数。
- raw 模式透传 converter 需要的 metadata。

为什么这样写：
- 保存路径选择集中在一个函数，Actor.save 不关心具体实现。
- 同一 flag 也被 load 侧用于限制 HF load 只走 bridge。

不变量与失败模式：
- 非 bridge 值都会走 raw direct path。
- bridge saver 和 direct saver 对输出目录清理策略不同，调用方不要混用同一路径。

Comment：
`megatron_to_hf_mode` 是 HF 保存路径的总开关。

### 2.2 raw direct saver 先准备目录、复制资产并广播 metadata

问题与约束：
- raw 导出需要 HF config/tokenizer 等非权重资产，也需要所有 rank 知道 model_name 和 quantization_config；但只有 rank 0 应操作输出目录的资产文件。

设计选择：
- rank 0 创建目录、清理旧 HF 权重、复制 `--hf-checkpoint` 中的非权重资产；再从显式参数或 AutoConfig 推断 model_name/quantization_config，并通过 distributed broadcast 发送给所有 rank。

Explain：
源码禁止输出目录等于 `--hf-checkpoint`，避免清理旧权重时删除输入模板。raw saver 要求 `--hf-checkpoint` 是本地目录，因为它要复制资产文件。

来源：slime/backends/megatron_utils/hf_checkpoint_saver.py L45-L114

Code：

```python
def save_hf_model_direct_to_path(...):
    path = Path(output_dir)
    hf_checkpoint = Path(args.hf_checkpoint).resolve()
    save_path = path.resolve()
    if hf_checkpoint == save_path:
        raise ValueError("HF save output path must not point to the same directory as --hf-checkpoint")
    if not hf_checkpoint.is_dir():
        raise ValueError(
            f"--hf-checkpoint must be a local directory when saving raw HuggingFace weights: {args.hf_checkpoint}"
        )

    is_save_rank = _is_global_rank_zero()
    setup_error = None
    if is_save_rank:
        try:
            path.mkdir(parents=True, exist_ok=True)
            _clear_existing_hf_weights(path)
            _copy_hf_assets(args.hf_checkpoint, path)
        except Exception as e:
            setup_error = repr(e)

    _raise_if_rank_zero_failed("prepare raw HuggingFace save directory", setup_error)
```

```python
payload: list[Any] = [None]
if model_name is not None:
    payload = [(model_name, quantization_config)]
else:
    if is_save_rank:
        hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        payload = [(type(hf_config).__name__.lower() if args.model_name is None else args.model_name, getattr(hf_config, "quantization_config", None))]
if dist.is_available() and dist.is_initialized():
    dist.broadcast_object_list(payload, src=0)
model_name, quantization_config = payload[0]
```

代码逻辑：
- 输出目录不能覆盖 HF 输入目录。
- rank 0 做文件系统准备。
- rank 0 失败通过 broadcast 传播到所有 rank。
- model metadata 由显式参数或 HF config 得到。

为什么这样写：
- 多 rank 同时复制资产会产生竞态。
- 所有 rank 后续都要执行 converter，因此必须拿到相同 model_name 和 quantization_config。

不变量与失败模式：
- `--hf-checkpoint` 必须是本地目录。
- AutoConfig 加载失败会广播错误并让所有 rank 失败。

Comment：
raw 保存先建立 HF 目录外壳，再开始处理权重。

### 2.3 HfWeightIteratorDirect 按 bucket 生成 HF named tensors

问题与约束：
- Megatron 参数被 TP/PP/EP 切分，不能一次性 all-gather 全模型；需要分块收集、转换和释放，控制内存峰值。

设计选择：
- `HfWeightIteratorDirect` 初始化时计算本地参数 metadata buckets；`get_hf_weight_chunks` 每次取一个 bucket，重建 full params，转换成 HF named tensors 后 yield。

Explain：
`_convert_to_hf_named_tensors` 对每个 `ParamInfo` 和 full param 调 `convert_to_hf`。一个 Megatron 参数可产生多个 HF tensor，因此输出是 named tensor 列表，而不是一一对应的 dict。

来源：slime/backends/megatron_utils/update_weight/hf_weight_iterator_direct.py L19-L41

Code：

```python
class HfWeightIteratorDirect(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.megatron_local_param_info_buckets = _get_megatron_local_param_info_buckets(self.args, self.model)

    def get_hf_weight_chunks(self, megatron_local_weights, progress_desc: str = "Update weights"):
        rank = dist.get_rank()

        for megatron_local_param_infos in tqdm(
            self.megatron_local_param_info_buckets, disable=rank != 0, desc=progress_desc
        ):
            megatron_full_params = _get_megatron_full_params(megatron_local_param_infos, megatron_local_weights)
            hf_named_tensors = self._convert_to_hf_named_tensors(megatron_full_params, megatron_local_param_infos)
            yield hf_named_tensors
            del megatron_full_params

    def _convert_to_hf_named_tensors(self, megatron_full_params: Sequence[torch.Tensor], param_infos: list[ParamInfo]):
        hf_named_tensors = []
        for info, param in zip(param_infos, megatron_full_params, strict=False):
            hf_named_tensors.extend(
                convert_to_hf(self.args, self.model_name, info.name, param, self.quantization_config)
            )
        return hf_named_tensors
```

代码逻辑：
- 初始化时构造参数 bucket。
- 每个 bucket 重建 full params。
- converter 将 Megatron 参数映射到 HF named tensors。
- bucket 处理后删除 full params。

为什么这样写：
- 分块 all-gather 限制导出时显存占用。
- iterator 形式能让 saver 边转换边写 shard，不必保留全模型 HF state_dict。

不变量与失败模式：
- 所有 rank 必须对 bucket 划分达成一致。
- converter 输出重复 HF name 会在 writer 阶段失败。

Comment：
raw 保存路径的核心不是直接读 state_dict，而是流式重建并转换并行参数。

### 2.4 _get_megatron_full_params 重建 TP/PP/EP 完整参数

问题与约束：
- 每个 rank 只持有部分 Megatron 参数；HF 保存需要完整张量。PP 和 EP rank 之间还要先把参数广播到需要参与 all-gather 的进程。

设计选择：
- 对当前 bucket，源 rank 创建 Parameter，其他 rank 创建空 tensor；再按 PP/EP group 广播，设置 tensor parallel attrs，最后调用 `all_gather_params_async` 重建 full params。

Explain：
expert 参数走 expert model parallel group；普通参数走 pipeline group。attrs 保留 `tensor_model_parallel/partition_dim/partition_stride/parallel_mode` 等信息，供 all-gather 和 converter 正确理解分片方式。

来源：slime/backends/megatron_utils/update_weight/hf_weight_iterator_direct.py L44-L105

Code：

```python
def _get_megatron_full_params(
    megatron_local_param_infos: Sequence[ParamInfo],
    megatron_local_weights,
) -> Sequence[torch.Tensor]:
    monkey_patch_torch_reductions()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    ep_size = mpu.get_expert_model_parallel_world_size()
    rank = dist.get_rank()
    params = []
    for info in megatron_local_param_infos:
        if dist.get_rank() == info.src_rank:
            params.append(
                torch.nn.Parameter(
                    megatron_local_weights[info.name].to(device=torch.cuda.current_device(), non_blocking=True),
                    requires_grad=False,
                )
            )
        else:
            params.append(torch.empty(info.shape, dtype=info.dtype, device=torch.cuda.current_device()))
    torch.cuda.synchronize()

    if pp_size > 1:
        ...
    if ep_size > 1:
        ...
    for info, param in zip(megatron_local_param_infos, params, strict=False):
        for key, value in info.attrs.items():
            setattr(param, key, value)

    gathered_params = all_gather_params_async(list(zip(megatron_local_param_infos, params, strict=False)))
    return gathered_params
```

代码逻辑：
- 源 rank 提供真实 local tensor。
- 非源 rank 分配同 shape 空 tensor。
- PP/EP 维度广播让需要的 rank 拿到 shard。
- 属性恢复后 all-gather 完整参数。

为什么这样写：
- HF converter 需要完整参数，而不是 TP shard。
- 广播和 all-gather 分离处理 PP/EP 与 TP 的不同并行语义。

不变量与失败模式：
- `info.src_rank` 必须在对应 process group 中可达。
- attrs 必须准确描述 TP 分片方式。

Comment：
这一步把训练时的分布式参数恢复成 HF 保存所需的全量张量。

### 2.5 参数 metadata bucket 在所有 rank 上保持一致

问题与约束：
- raw 保存需要所有 rank 以相同顺序处理相同参数 bucket；否则 collective all-gather 和 chunk 写入会错位。

设计选择：
- `_get_megatron_local_param_infos` 收集本 rank named params/buffers 的 `ParamInfo`，通过 PP/EP group 交换 metadata，按 name 排序，并用全局 gloo all_gather 校验所有 rank 的 name/shape/dtype 一致。

Explain：
bucket 构造时按 `update_weight_buffer_size` 控制每个 bucket 的参数总大小；expert 参数使用 expert TP size，普通参数使用 regular TP size 估算完整参数大小。

来源：slime/backends/megatron_utils/update_weight/hf_weight_iterator_direct.py L108-L211

Code：

```python
def _get_megatron_local_param_info_buckets(args: Namespace, model: Sequence[torch.nn.Module]) -> list[list[ParamInfo]]:
    param_infos = _get_megatron_local_param_infos(args, model)
    param_info_buckets = [[]]
    buffer_size = 0

    for info in param_infos:
        if ".experts." in info.name:
            tp_size = mpu.get_expert_tensor_parallel_world_size()
        else:
            tp_size = mpu.get_tensor_model_parallel_world_size()

        param_size = info.size * tp_size
        if buffer_size + param_size > args.update_weight_buffer_size and len(param_info_buckets[-1]) > 0:
            param_info_buckets.append([])
            buffer_size = 0
        param_info_buckets[-1].append(info)
        buffer_size += param_size
    return param_info_buckets
```

```python
param_infos = list(param_infos.values())
param_infos = sorted(param_infos, key=lambda info: info.name)

all_param_info_list = [None] * dist.get_world_size()
dist.all_gather_object(
    obj=param_infos,
    object_list=all_param_info_list,
    group=get_gloo_group(),
)
for i, param_info in enumerate(param_infos):
    for infos in all_param_info_list:
        assert infos[i].name == param_info.name
        assert infos[i].shape == param_info.shape
        assert infos[i].dtype == param_info.dtype
```

代码逻辑：
- 收集本地参数的 name、shape、dtype、attrs 和 src_rank。
- PP/EP 交换补齐其他阶段或专家的 metadata。
- 全局排序保证顺序稳定。
- 全 rank 校验 metadata 一致。
- 再按 buffer size 分 bucket。

为什么这样写：
- collective 通信要求所有 rank 对参数顺序和形状有一致视图。
- bucket size 控制显存峰值，避免一次 gather 全模型。

不变量与失败模式：
- 所有 rank 的 param_infos 数量、顺序、shape、dtype 必须一致。
- update_weight_buffer_size 太小会产生更多 bucket，增加通信和写文件次数。

Comment：
raw HF 保存复用了权重同步的参数 metadata 机制。

### 2.6 raw saver 按 chunk 轮转分配 writer rank

问题与约束：
- 多节点保存时不希望所有 rank 都写同一 shard；同时转换出的 HF chunks 要在节点 writer 间分摊。

设计选择：
- direct saver 根据 `_get_node_save_layout` 得到 writer ranks；遍历 HF chunks 时，只有命中 `chunk_idx % num_save_nodes == save_node_rank` 的 writer rank 暂存并写出当前 chunk。

Explain：
每个 chunk 处理后如果到达一个 save-node 轮转周期，就调用 `_write_pending_chunk` 写出。循环结束后再写最后一个 pending chunk，并调用 `_finalize_distributed_shards` 汇总所有 writer state。

来源：slime/backends/megatron_utils/hf_checkpoint_saver.py L122-L138

Code：

```python
writer = _SafetensorShardWriter(path, enabled=is_writer_rank)
pending_write = None

for chunk_idx, hf_named_tensors in enumerate(
    hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights, progress_desc=progress_desc)
):
    if is_writer_rank and chunk_idx % num_save_nodes == save_node_rank:
        pending_write = (chunk_idx, hf_named_tensors)
        hf_named_tensors = None
    else:
        del hf_named_tensors

    if (chunk_idx + 1) % num_save_nodes == 0:
        pending_write = _write_pending_chunk(writer, pending_write)

pending_write = _write_pending_chunk(writer, pending_write)
_finalize_distributed_shards(path, writer.state())
```

代码逻辑：
- 每个 writer rank 只负责一部分 chunk。
- 非负责 rank 删除当前 chunk tensors。
- pending chunk 在轮转边界写入。
- 全部写完后汇总 shard states。

为什么这样写：
- 多节点分担磁盘写入，减少单 rank IO 压力。
- 每个 chunk 写完后释放 tensors，控制内存峰值。

不变量与失败模式：
- 所有 rank 必须遍历相同 chunk_idx。
- writer layout 必须与 world_size/gpus_per_node 一致。

Comment：
raw 保存的分布式写入是按转换 chunk 轮转，而不是按参数名静态分片。

### 2.7 bridge saver 直接委托 Megatron Bridge 写 HF

问题与约束：
- Bridge 模式希望复用 Megatron Bridge 的 HF 保存实现，而不是走 Slime raw converter；但仍要适配 Slime 模型包装。

设计选择：
- `save_hf_model_bridge_to_path` 创建 AutoBridge，patch HF config 和 Megatron model，在 context 内调用 `bridge.save_hf_pretrained`，最后 distributed barrier。

Explain：
日志只在 data parallel rank 0 且 tensor model parallel rank 0 打印。所有 rank 参与 barrier，确保 bridge 保存完成后再继续。

来源：slime/backends/megatron_utils/hf_checkpoint_saver.py L144-L172

Code：

```python
def save_hf_model_bridge_to_path(args, output_dir: str | Path, model) -> None:
    from megatron.bridge import AutoBridge
    from megatron.core import mpu

    from slime.utils.megatron_bridge_utils import patch_auto_bridge_hf_config, patch_megatron_model

    path = Path(output_dir)
    should_log = (
        mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
    )
    path.mkdir(parents=True, exist_ok=True)
    bridge = patch_auto_bridge_hf_config(AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True))

    with patch_megatron_model(model):
        bridge.save_hf_pretrained(
            model,
            path=path,
        )

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
```

代码逻辑：
- Bridge 从原 HF checkpoint 推断配置。
- Slime patch 包装 HF config 和 Megatron model。
- Bridge 负责实际 HF 保存。
- 分布式环境下 barrier。

为什么这样写：
- Bridge 已经知道部分模型从 Megatron 到 HF 的保存规则，复用可减少 Slime 侧维护。
- patch context 隔离 Slime 模型包装差异。

不变量与失败模式：
- Megatron Bridge 必须支持当前模型架构。
- 输出目录创建后，Bridge 内部失败可能留下部分文件。

Comment：
bridge saver 是最短路径，但可控性不如 raw saver。

### 2.8 _SafetensorShardWriter 防止重复 tensor 和重复 shard 文件

问题与约束：
- 多个 Megatron 参数可能错误地转换出同名 HF tensor；分布式写入也可能产生重复 shard 文件名，任何一种都会生成不可加载 checkpoint。

设计选择：
- writer 在写 shard 前检查 tensor name 是否已在全局 weight_map 或本 shard state_dict 中出现；写文件前检查目标 shard filename 是否已存在。

Explain：
写入前 `_tensor_for_safetensors` 会 detach、contiguous，并搬到 CPU。writer.state 返回 total_size、weight_map 和 shard_files，供分布式 finalize 汇总。

来源：slime/backends/megatron_utils/hf_checkpoint_saver.py L175-L240

Code：

```python
class _SafetensorShardWriter:
    def write(self, named_tensors, shard_idx: int) -> None:
        if not self.enabled:
            return
        assert shard_idx is not None, "shard_idx must be set when writing HF shards"

        from safetensors.torch import save_file

        state_dict = {}
        total_size = 0
        for name, tensor in named_tensors:
            if name in self.weight_map or name in state_dict:
                raise ValueError(f"Duplicate HF tensor while saving: {name}")
            total_size += tensor.numel() * tensor.element_size()
            state_dict[name] = _tensor_for_safetensors(tensor)

        if not state_dict:
            return

        filename = self._next_filename(shard_idx)
        if (self.path / filename).exists():
            raise ValueError(f"Duplicate HF shard file while saving: {filename}")

        save_file(state_dict, self.path / filename, metadata={"format": "pt"})
        self.shard_files.append(filename)
        self.total_size += total_size
        for name in state_dict:
            self.weight_map[name] = filename
```

代码逻辑：
- disabled writer rank 不写文件。
- shard 内和历史 shard 都检查重复 tensor name。
- 空 shard 直接返回。
- 文件写出后更新 shard state。

为什么这样写：
- 重复 HF tensor name 会让 index 指向不确定权重，必须立即失败。
- 临时文件名按 chunk_idx 生成，重复文件也说明分布式 writer 逻辑出错。

不变量与失败模式：
- named_tensors 必须是 `(name, tensor)` 迭代。
- tensor 必须可被 safetensors 保存。

Comment：
writer 是 raw 保存路径的数据完整性检查点。

### 2.9 finalize 合并各 rank shard state 并生成最终 index

问题与约束：
- 多 writer rank 各自产生临时 shard 文件名和 weight map；最终 HF 目录需要连续编号的 shard 文件和统一 index。

设计选择：
- `_finalize_distributed_shards` all_gather 每个 rank 的 writer state；global rank 0 调 `_finalize_shard_files` 合并、排序、重命名并写 `model.safetensors.index.json`。

Explain：
合并时会检查重复 shard file 和重复 HF tensor name。重命名后，用 rename map 把 raw weight_map 中的临时文件名转换为最终 `model-xxxxx-of-yyyyy.safetensors` 文件名。

来源：slime/backends/megatron_utils/hf_checkpoint_saver.py L255-L320

Code：

```python
def _finalize_distributed_shards(path: Path, local_state: dict[str, Any]) -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        states = [None] * dist.get_world_size()
        dist.all_gather_object(states, local_state)
    else:
        states = [local_state]

    if _is_global_rank_zero():
        _finalize_shard_files(path, states)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
```

```python
shard_files = sorted(shard_files, key=_shard_filename_sort_key)
total_files = len(shard_files)
rename_map = {}
for idx, old_name in enumerate(shard_files, start=1):
    new_name = f"model-{idx:05d}-of-{total_files:05d}.safetensors"
    os.replace(path / old_name, path / new_name)
    rename_map[old_name] = new_name

final_weight_map = {}
for name, filename in raw_weight_map.items():
    if filename not in rename_map:
        raise ValueError(f"HF tensor {name} points to missing shard file {filename}")
    final_weight_map[name] = rename_map[filename]

index_data = {"metadata": {"total_size": total_size}, "weight_map": final_weight_map}
```

代码逻辑：
- 分布式环境下收集所有 local writer state。
- rank 0 做最终文件整理。
- 临时 shard 文件按编号排序并重命名。
- index json 使用最终文件名。

为什么这样写：
- 多 rank 不能各自写最终 index，否则会互相覆盖。
- 最终连续编号是 HF safetensors 常见约定，方便 transformers 加载。

不变量与失败模式：
- 至少要有一个 shard file，否则失败。
- weight_map 中每个文件名都必须能在 rename_map 中找到。

Comment：
raw 保存的最后一步是把分布式局部产物收敛成一个标准 HF checkpoint 目录。

### 2.10 资产复制只保留非权重 HF 文件

问题与约束：
- 导出的 HF 目录需要 config/tokenizer 等文件，但不能把原 HF 目录中的旧权重文件混进输出目录。

设计选择：
- `_clear_existing_hf_weights` 删除输出目录里的旧 HF 权重文件；`_copy_hf_assets` 从原 HF 目录复制普通文件，但跳过权重文件。

Explain：
权重文件判定包括 index 文件名和 `.safetensors/.bin/.pt/.pth/.ckpt/.msgpack` 等 suffix。复制使用 `shutil.copy2`，保留文件 metadata。

来源：slime/backends/megatron_utils/hf_checkpoint_saver.py L323-L352

Code：

```python
def _tensor_for_safetensors(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    return tensor

def _clear_existing_hf_weights(path: Path) -> None:
    for item in path.iterdir():
        if item.is_file() and _is_hf_weight_file(item):
            item.unlink()

def _copy_hf_assets(origin_hf_dir: str, output_dir: Path) -> None:
    origin = Path(origin_hf_dir)
    if not origin.is_dir():
        raise ValueError(f"--hf-checkpoint must be a local directory when using raw --save-hf: {origin_hf_dir}")

    for item in origin.iterdir():
        if item.is_file():
            if _is_hf_weight_file(item):
                continue
            shutil.copy2(item, output_dir / item.name)
```

代码逻辑：
- safetensors 写入前把 tensor 转 CPU contiguous。
- 输出目录已有权重文件先删除。
- 原 HF 目录只复制非权重普通文件。
- 权重文件由 raw converter 重新生成。

为什么这样写：
- 混入旧权重会让 index 和文件集合不一致。
- 非权重资产可以直接复用，不需要 converter 参与。

不变量与失败模式：
- 原 HF 目录必须存在且是本地目录。
- 嵌套目录资产不会被复制。

Comment：
HF checkpoint 由新权重文件和旧配置/Tokenizer 资产共同组成。

---

## 3. Megatron 到 HF 的命名转换

### 3.1 convert_to_hf 统一执行 padding removal、模型路由和量化后处理

问题与约束：
- 不同模型族的 Megatron 参数映射不同；同一个参数还可能需要裁 vocab padding 或输出多个 HF tensor。

设计选择：
- `convert_to_hf` 先调用 `remove_padding`，再用 `_convert_to_hf_core` 按 model_name 分发到模型专用 converter，最后调用 `quantize_params`。

Explain：
`_convert_to_hf_core` 支持 minimax、DeepSeek/GLM MoE、GLM、GPT-OSS、Qwen、Gemma、LLaMA、MIMO 等分支；如果 `q_lora_rank` 不为空，还会用 `_cached_tensors` 配对 q_a 和 kv_a projection，兼容 SGLang 实现。

来源：slime/backends/megatron_utils/megatron_to_hf/__init__.py L25-L95

Code：

```python
def convert_to_hf(args, model_name, name, param, quantization_config=None):
    param = remove_padding(name, param, args.vocab_size)

    converted_named_tensors = _convert_to_hf_core(args, model_name, name, param)

    return quantize_params(args, name, converted_named_tensors, quantization_config)

def _convert_to_hf_core(args, model_name, name, param):
    if "minimaxm2" in model_name or "minimax_m2" in model_name:
        converted_named_tensors = convert_minimax_m2_to_hf(args, name, param)
    elif "glm4moelite" in model_name or "deepseekv3" in model_name or "glmmoedsa" in model_name:
        converted_named_tensors = convert_deepseekv3_to_hf(args, name, param)
    elif "qwen2" in model_name or "qwen3" in model_name:
        converted_named_tensors = convert_qwen2_to_hf(args, name, param)
    elif "llama" in model_name:
        converted_named_tensors = convert_llama_to_hf(args, name, param)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if args.q_lora_rank is not None:
        ...
    return converted_named_tensors
```

代码逻辑：
- 先处理 vocab padding。
- 按 model name 分发 converter。
- unsupported model 直接报错。
- 最后执行量化参数后处理。

为什么这样写：
- 主保存流程不应包含模型族特定命名规则。
- padding removal 是通用前处理，应在模型分发前执行。

不变量与失败模式：
- `args.vocab_size` 和 model_name 必须与目标 HF config 一致。
- converter 返回的 HF tensor name 必须唯一。

Comment：
raw M2HF 的语义正确性主要由这个转换入口保证。

### 3.2 Qwen2 converter 处理 embedding、output、final norm 与 QKV 拆分

问题与约束：
- Qwen2 在 Megatron 侧有 `module.module` 前缀和 fused `linear_qkv`，而 HF 侧需要 embed_tokens、lm_head、norm 以及 q/k/v 三个投影。

设计选择：
- converter 先处理全局 embedding/output/final norm；进入 decoder layer 后，用正则取 layer id，再对 attention projection 和 fused QKV 按 head/group 维度拆分。

Explain：
`linear_qkv.weight` 被 reshape 成 `[num_query_groups, *, head_dim, hidden_size]`，再按 `[value_num_per_group, 1, 1]` split 出 q/k/v，最后 reshape 回二维矩阵。

来源：slime/backends/megatron_utils/megatron_to_hf/qwen2.py L5-L36

Code：

```python
def convert_qwen2_to_hf(args, name, param):
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()
        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.weight":
            param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            q_param, k_param, v_param = torch.split(param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1)
            q_param = q_param.reshape(-1, args.hidden_size)
            k_param = k_param.reshape(-1, args.hidden_size)
            v_param = v_param.reshape(-1, args.hidden_size)
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.weight", q_param),
                (f"model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
                (f"model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
            ]
```

代码逻辑：
- 非 layer 级参数先直接映射。
- layer 参数用正则提取 `layer_idx/rest`。
- output projection 映射到 `o_proj`。
- fused QKV 按 GQA group 拆成 q/k/v。

为什么这样写：
- HF Qwen2 命名和 Megatron fused 命名不一致，必须显式改名与拆分。
- GQA 下 Q 的数量和 K/V 数量不同，不能简单按三等分。

不变量与失败模式：
- `num_attention_heads` 必须能被 `num_query_groups` 整除。
- `linear_qkv.weight` shape 必须能按 group/head_dim reshape。

Comment：
Qwen2 converter 是理解 M2HF 命名和形状转换的代表样例。

### 3.3 Qwen2 converter 处理 QKV bias、SwiGLU MLP 和 norm

问题与约束：
- Qwen2 的 bias、MLP gate/up 和 layernorm 在 Megatron 与 HF 之间也有不同布局；未知参数名不能静默跳过。

设计选择：
- QKV bias 按 group/head_dim 拆分；`mlp.linear_fc1.weight` chunk 成 gate/up；`linear_fc2` 映射 down；attention/MLP layernorm 和 q/k norm 分别改名。

Explain：
函数末尾 `raise ValueError(f"Unknown parameter name: {name}")`，确保 converter 没覆盖的新参数不会被悄悄丢弃。

来源：slime/backends/megatron_utils/megatron_to_hf/qwen2.py L37-L71

Code：

```python
elif rest == "self_attention.linear_qkv.bias":
    param = param.view(args.num_query_groups, -1)
    q_bias, k_bias, v_bias = torch.split(
        param,
        split_size_or_sections=[value_num_per_group * head_dim, head_dim, head_dim],
        dim=1,
    )
    q_bias = q_bias.contiguous().flatten()
    k_bias = k_bias.contiguous().flatten()
    v_bias = v_bias.contiguous().flatten()
    return [
        (f"model.layers.{layer_idx}.self_attn.q_proj.bias", q_bias),
        (f"model.layers.{layer_idx}.self_attn.k_proj.bias", k_bias),
        (f"model.layers.{layer_idx}.self_attn.v_proj.bias", v_bias),
    ]
elif rest == "mlp.linear_fc1.weight":
    gate_weight, up_weight = param.chunk(2, dim=0)
    return [
        (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
        (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
    ]
elif rest == "mlp.linear_fc2.weight":
    return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]
elif rest == "self_attention.linear_qkv.layer_norm_weight":
    return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
elif rest == "mlp.linear_fc1.layer_norm_weight":
    return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
elif rest == "self_attention.q_layernorm.weight":
    return [(f"model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
elif rest == "self_attention.k_layernorm.weight":
    return [(f"model.layers.{layer_idx}.self_attn.k_norm.weight", param)]

raise ValueError(f"Unknown parameter name: {name}")
```

代码逻辑：
- QKV bias 和 QKV weight 使用一致的 group 拆分语义。
- SwiGLU merged fc1 沿输出维二分为 gate/up。
- norm 参数按 HF Qwen2 命名改写。
- 未识别参数直接失败。

为什么这样写：
- HF 模型加载要求参数名和 shape 完整匹配，漏掉参数比显式失败更危险。
- gate/up 拆分是 Megatron merged MLP 到 HF split MLP 的必要步骤。

不变量与失败模式：
- fc1 第一维必须能二等分。
- 新增 Qwen2 参数若未加入 converter，会触发 Unknown parameter。

Comment：
converter 的失败策略是保守的：宁可中断导出，也不生成缺权重的 HF checkpoint。
