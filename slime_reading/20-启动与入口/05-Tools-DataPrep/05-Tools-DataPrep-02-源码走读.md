---
type: batch-doc
module: 05-Tools-DataPrep
batch: "05"
doc_type: walkthrough
title: "Tools-DataPrep · 源码走读"
tags:
  - slime/batch/05
  - slime/module/tools-dataprep
  - slime/doc/walkthrough
updated: 2026-07-05
---

# Tools-DataPrep · 源码走读

> 走读主线：Slime 的权重准备工具有两个方向：`convert_hf_to_torch_dist.py` 把 HuggingFace checkpoint 灌入 Megatron 分布式 checkpoint，供训练/权重更新使用；`convert_torch_dist_to_hf.py` 把 Megatron dist checkpoint 重新展开、改名、裁 padding 并写成 HF safetensors。

---

## 1. HF 到 torch_dist：让 Megatron 能加载初始权重

### 1.1 add_convertion_args 只补转换脚本需要的参数

问题与约束：
- 脚本复用 Megatron 的 `parse_args`，不能重新实现训练参数解析；但转换又需要 HF checkpoint 路径和少量 Slime 专用开关。

设计选择：
- `add_convertion_args` 在 Megatron parser 上追加 `--hf-checkpoint`、custom model provider、Megatron-to-HF mode、context-parallel allgather 和可选 padded vocab size。

Explain：
`--hf-checkpoint` 是必填项，因为脚本的输入就是 HF 模型目录。`--megatron-to-hf-mode` 不是本脚本的直接输出模式，而是写入 args 后影响后续训练/权重更新链路对 Megatron 权重和 HF 权重映射的选择。

来源：tools/convert_hf_to_torch_dist.py L21-L41

Code：

```python
def add_convertion_args(parser):
    parser.add_argument("--hf-checkpoint", type=str, required=True, help="HuggingFace model path")
    parser.add_argument(
        "--custom-model-provider-path",
        type=str,
        default=None,
        help="Path to a custom model provider function.",
    )
    parser.add_argument(
        "--megatron-to-hf-mode",
        choices=["raw", "bridge"],
        default="raw",
        help="The method to convert megatron weights to hugging face weights for SGLang.",
    )
    parser.add_argument("--allgather-cp", action="store_true", default=False)
    try:
        parser.add_argument("--padded-vocab-size", type=int, default=None)
    except Exception:
        pass
    return parser
```

代码逻辑：
- 必填输入目录是 `--hf-checkpoint`。
- custom model provider 允许用户替换 Megatron model provider。
- `--allgather-cp` 传给后续 Megatron/Slime 参数校验。
- `--padded-vocab-size` 用 try 包裹，兼容 parser 已有同名参数的情况。

为什么这样写：
- 训练参数仍由 Megatron/Slime 统一解析，转换脚本只补充差异字段。
- try-add padded vocab 可以兼容不同 Megatron 参数集合。

不变量与失败模式：
- 缺少 `--hf-checkpoint` 会在 argparse 阶段失败。
- custom provider 路径如果不可导入，会在模型构建阶段失败。

Comment：
这段说明转换脚本不是独立 CLI，而是 Megatron 参数体系上的一个薄扩展。

### 1.2 get_args 用 world size 自动推导 pipeline parallel

问题与约束：
- 转换脚本可以用多 GPU 并行保存 Megatron checkpoint；如果 world size 大于层数或 PP 切分不合法，Megatron 初始化会失败。

设计选择：
- `get_args` 先调用 Megatron parser 和 Slime defaults，再设置转换所需的 batch/save 参数；当用户未显式设置 PP 且 world size > 1 时，自动寻找一个合法 pipeline size。

Explain：
源码要求 `world_size <= args.num_layers`，因为自动 PP 按层分配。若 `pipeline_model_parallel_size == 1`，脚本从 world size 开始尝试，计算 `decoder_last_pipeline_num_layers`，直到最后一段层数为正；偶数失败时继续二分。

来源：tools/convert_hf_to_torch_dist.py L44-L84

Code：

```python
def get_args():
    args = parse_args(add_convertion_args)
    args = set_default_megatron_args(args)
    args.save_interval = 1
    args.micro_batch_size = 1
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.global_batch_size = int(os.environ.get("WORLD_SIZE", "1"))

    assert world_size <= args.num_layers, (
        f"World size {world_size} must be less than or equal to number of layers {args.num_layers}. "
        "You are using too many GPUs for this conversion."
    )

    if args.pipeline_model_parallel_size == 1 and world_size > 1:
        pp_size = world_size
        while True:
            args.pipeline_model_parallel_size = pp_size
            args.decoder_last_pipeline_num_layers = args.num_layers - ceildiv(
                args.num_layers, args.pipeline_model_parallel_size
            ) * (args.pipeline_model_parallel_size - 1)
            if args.decoder_last_pipeline_num_layers > 0:
                break
            if pp_size % 2 == 0:
                pp_size //= 2
            else:
                raise ValueError(...)
    validate_args(args)
    return args
```

代码逻辑：
- 先设置能通过 Megatron 校验的 save/micro/global batch 参数。
- 从环境变量读取 world size。
- 自动 PP 只在用户没有显式 PP 时触发。
- 最后交给 Megatron `validate_args` 做一致性校验。

为什么这样写：
- 转换只需要构建模型和保存 checkpoint，不需要真实训练 batch 语义。
- 自动 PP 可以让大模型转换在多 GPU 上分摊层权重，而不要求用户手工计算 PP。

不变量与失败模式：
- world size 不能超过层数。
- 如果无法找到最后一段层数为正的 PP 切分，会抛出 `ValueError`。

Comment：
这里的 PP 自动推导是为了权重转换可运行，不是训练拓扑推荐器。

### 1.3 main 兼容 torchrun/Slurm 并处理 ROCm 初始化限制

问题与约束：
- 转换脚本需要自行初始化 torch distributed；运行环境可能是 torchrun，也可能是 Slurm。同时 ROCm 下异步 checkpoint writer 和 GPU 初始化路径存在兼容限制。

设计选择：
- main 从 `WORLD_SIZE/LOCAL_RANK/RANK` 或 Slurm 环境变量读取 rank 信息，设置 CUDA device 和默认 master 地址端口，再初始化 NCCL process group；HIP 环境下替换 writer 并要求 CPU initialization。

Explain：
ROCm patch 在 distributed 初始化前执行，把 Megatron filesystem async writer 替换成 Slime 的 `ROCmFileSystemWriterAsync`。初始化 args 后，如果检测到 HIP 且未启用 `use_cpu_initialization`，脚本直接 assert。

来源：tools/convert_hf_to_torch_dist.py L87-L120

Code：

```python
if torch.version.hip:
    import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module
    from slime.utils.rocm_checkpoint_writer import ROCmFileSystemWriterAsync

    filesystem_async_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync

world_size = int(os.getenv("WORLD_SIZE") or os.getenv("SLURM_NTASKS") or 1)
local_rank = int(os.getenv("LOCAL_RANK") or os.getenv("SLURM_LOCALID") or 0)
global_rank = int(os.getenv("RANK") or os.getenv("SLURM_PROCID") or 0)

torch.cuda.set_device(local_rank)
os.environ.setdefault("WORLD_SIZE", str(world_size))
os.environ.setdefault("RANK", str(global_rank))
os.environ.setdefault("LOCAL_RANK", str(local_rank))
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12355")
dist.init_process_group(
    backend="nccl",
    world_size=world_size,
    rank=global_rank,
    device_id=torch.device(f"cuda:{local_rank}"),
)
args = get_args()
init(args)

if hasattr(torch.version, "hip") and torch.version.hip is not None:
    assert args.use_cpu_initialization, "AMD GPU requires --use_cpu_initialization=True"
```

代码逻辑：
- HIP 分支先 patch writer。
- rank 信息优先读通用分布式环境变量，缺失时读 Slurm。
- 设置当前进程 CUDA device。
- 初始化 process group 后再执行 Megatron `init(args)`。

为什么这样写：
- Megatron checkpoint save 依赖 distributed group，单进程也要走一致入口。
- ROCm 对部分异步 checkpoint 和 GPU 初始化路径有限制，提前改写和 assert 可以把错误前置。

不变量与失败模式：
- local rank 必须对应可用 GPU。
- 默认 master addr/port 只适合单机或外部未设置 rendezvous 的场景。
- AMD GPU 路径必须传 `--use-cpu-initialization`。

Comment：
HF→torch_dist 转换本质上仍是一次 Megatron distributed 程序启动。

### 1.4 建模、灌 HF 权重并保存 Megatron dist checkpoint

问题与约束：
- 需要把 HF 权重灌入 Megatron 模型结构，但转换阶段不训练、不需要 DDP，也不需要 optimizer/scheduler state。

设计选择：
- 用 `get_model(..., wrap_with_ddp=False)` 构建 Megatron 模型；`AutoBridge.from_pretrained` 加载 HF 权重；最后调用 `save_checkpoint(1, model, None, None, 0)`。

Explain：
`memory_efficient=True` 表示 bridge 加载权重时尽量降低内存峰值。若使用 CPU initialization，源码把 `model[0]` 移回 CPU。保存 checkpoint 时 optimizer、scheduler 都传 `None`，只保留模型权重。

来源：tools/convert_hf_to_torch_dist.py L121-L137

Code：

```python
model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)

hf_model_path = args.hf_checkpoint
bridge = AutoBridge.from_pretrained(hf_model_path, trust_remote_code=True)
bridge.load_weights(model, hf_model_path, memory_efficient=True)

if args.use_cpu_initialization:
    model[0] = model[0].cpu()

print_memory("after loading model")
torch.cuda.synchronize()
gc.collect()
torch.cuda.empty_cache()

save_checkpoint(1, model, None, None, 0)
```

代码逻辑：
- model provider 由 Slime/Megatron args 决定。
- AutoBridge 负责 HF 权重到 Megatron 模型的映射。
- 保存前释放缓存并同步 CUDA。
- checkpoint iteration 固定为 1。

为什么这样写：
- DDP wrapper 对离线转换无意义，反而会干扰权重访问。
- 使用 Megatron 原生 `save_checkpoint` 能产出训练侧可直接加载的 dist checkpoint。

不变量与失败模式：
- HF checkpoint 必须和 model provider 构建出的 Megatron 结构匹配。
- `save_checkpoint` 需要所有分布式 rank 参与 collective。

Comment：
这一段是 HF→torch_dist 的核心：构建 Megatron 壳，用 mbridge 灌 HF 权重，再交给 Megatron 保存。

### 1.5 rank 0 把 iter checkpoint 改成 release 目录

问题与约束：
- Megatron `save_checkpoint(1, ...)` 会按 iteration 写目录；训练加载通常按 `release` tracker/目录约定读取。

设计选择：
- rank 0 写 checkpoint tracker 为 `release`，再把 iteration 1 的 base dir 移动到 release 路径；所有 rank 在最后 barrier 后销毁进程组。

Explain：
`get_checkpoint_name(args.save, 1, False, return_base_dir=True)` 取 iteration source dir；`get_checkpoint_name(args.save, -1, True, return_base_dir=True)` 取 release target dir。只有 rank 0 做文件移动，避免多 rank 同时操作目录。

来源：tools/convert_hf_to_torch_dist.py L139-L148

Code：

```python
if dist.get_rank() == 0:
    tracker_filename = get_checkpoint_tracker_filename(args.save)
    with open(tracker_filename, "w") as f:
        f.write("release")
    source_dir = get_checkpoint_name(args.save, 1, False, return_base_dir=True)
    target_dir = get_checkpoint_name(args.save, -1, True, return_base_dir=True)
    shutil.move(source_dir, target_dir)
dist.barrier()
dist.destroy_process_group()
```

代码逻辑：
- 只 rank 0 写 tracker。
- source 是 iter=1 checkpoint 目录。
- target 是 Megatron release 目录。
- barrier 确保其他 rank 等待文件整理完成。

为什么这样写：
- 训练脚本按 release 约定加载会更稳定，不需要知道转换时固定写了 iteration 1。
- 文件系统操作集中在 rank 0，避免竞态。

不变量与失败模式：
- source dir 必须已经由 `save_checkpoint` 成功写出。
- target dir 已存在时 `shutil.move` 可能失败，需要调用方清理输出目录。

Comment：
HF→torch_dist 最终产物是 Megatron release checkpoint，而不是裸 iter_0000001 目录。

---

## 2. torch_dist 到 HF：读取、展开、转换和分片

### 2.1 UnpicklerWrapper 让 metadata 脱离训练环境类

问题与约束：
- Megatron dist checkpoint 的 metadata pickle 可能引用训练环境中的 Megatron/GLM 类；离线导出环境不一定能 import 这些类。

设计选择：
- 自定义 `UnpicklerWrapper.find_class`，遇到 `megatron` 或 `glm` 模块名前缀时返回 DummyClass，并全局替换 `pickle.Unpickler`。

Explain：
转换脚本只需要 metadata 中的 state dict 结构和 tensor storage 信息，不需要实例化真实训练对象。DummyClass 避免 metadata load 因依赖类缺失而失败。

来源：tools/convert_torch_dist_to_hf.py L19-L31

Code：

```python
class UnpicklerWrapper(pickle.Unpickler):
    @override
    def find_class(self, mod_name, name):
        class DummyClass:
            def __init__(self, *args, **kwargs):
                pass

        if mod_name.startswith("megatron") or mod_name.startswith("glm"):
            return DummyClass
        return super().find_class(mod_name, name)

pickle.Unpickler = UnpicklerWrapper
```

代码逻辑：
- 只替换指定模块前缀的类。
- 其他类仍走标准 pickle 查找。
- 替换是全局的，影响后续 metadata 读取。

为什么这样写：
- checkpoint 转换需要尽量少依赖训练代码环境。
- DummyClass 足够支撑 metadata 反序列化，不参与权重转换计算。

不变量与失败模式：
- 如果 metadata 需要 DummyClass 的真实行为而不是占位对象，后续访问会失败。
- 非 megatron/glm 的缺失类仍会按标准 pickle 报错。

Comment：
这是 torch_dist→HF 转换的兼容性入口。

### 2.2 WrappedStorageReader 与 EmptyStateDictLoadPlanner 只加载权重张量

问题与约束：
- dist checkpoint 中可能包含 optimizer 或其他 state；导出 HF 只需要模型权重，且需要根据 metadata 预建空 tensor 承接 shard。

设计选择：
- `WrappedStorageReader.read_metadata` 用安全 unpickler 读取 `.metadata`；`EmptyStateDictLoadPlanner` 遍历 metadata，跳过 optimizer/_state 键，并为 tensor metadata 创建空 tensor。

Explain：
planner 把所有需要加载的权重 key 放进 `state_dict`，然后调用父类 setup。后续 `_load_state_dict` 会按这个 state_dict 和 planner 从文件系统读入 shard。

来源：tools/convert_torch_dist_to_hf.py L34-L63

Code：

```python
class WrappedStorageReader(dist_cp.FileSystemReader):
    @override
    def read_metadata(self):
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = UnpicklerWrapper(metadata_file).load()
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = dist_cp.StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        if metadata.planner_data is None:
            metadata.planner_data = {}
        return metadata

class EmptyStateDictLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    @override
    def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
        for k, v in metadata.state_dict_metadata.items():
            if "optimizer" in k or "_state" in k:
                continue
            if isinstance(v, dist_cp.metadata.TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)
            state_dict[k] = v
        super().set_up_planner(state_dict, metadata, is_coordinator)
```

代码逻辑：
- reader 修复缺失的 storage_meta/planner_data。
- planner 跳过 optimizer 和 `_state`。
- tensor metadata 转成同 shape/dtype 的空 tensor。
- state_dict 由 planner 填充。

为什么这样写：
- 不加载 optimizer 可以显著减少内存和导出时间。
- 预分配空 tensor 是 torch distributed checkpoint loader 读 shard 的需要。

不变量与失败模式：
- `.metadata` 必须存在于 input dir。
- 过滤规则依赖 key 名称包含 optimizer 或 `_state`，不匹配的非权重键可能仍进入 state_dict。

Comment：
这里把 Megatron dist checkpoint 削减成“只含模型权重”的待转换 state_dict。

### 2.3 get_named_params 展开 layer/expert 堆叠并补 Megatron 前缀

问题与约束：
- dist checkpoint 里有些 tensor 可能把 layer 或 expert 维堆在第 0 维；HF converter 期望逐层、逐 expert 的 Megatron 参数名。

设计选择：
- `get_layer_param` 和 `get_expert_param` 递归展开没有显式 layer/expert id 的堆叠 tensor；`get_named_params` 给每个 key 加 `module.module.` 前缀后再展开。

Explain：
如果参数名包含 `.experts.` 但不匹配具体 expert weight 名，函数要求 `param.shape[0] == num_experts`，再按 expert id 拆成多条参数。layer 展开同理，要求第 0 维等于 `num_layers`。

来源：tools/convert_torch_dist_to_hf.py L66-L103

Code：

```python
def get_expert_param(args, name, param):
    if ".experts." not in name:
        yield name, param
        return

    num_experts = args.num_experts
    match = re.search(r"mlp.experts\.(.+)\.weight(\d+)", name)
    if not match:
        assert param.shape[0] == num_experts
        for expert_id in range(num_experts):
            expert_name = name.replace(".experts.experts.", ".experts.") + str(expert_id)
            expert_param = param[expert_id]
            yield expert_name, expert_param
    else:
        yield name, param

def get_named_params(args, state_dict):
    for name, param in state_dict.items():
        name = f"module.module.{name}"
        yield from get_layer_param(args, name, param)
```

代码逻辑：
- 非 expert 参数直接 yield。
- 堆叠 expert tensor 按 expert id 拆开。
- layer 展开发生在 expert 展开之前。
- 所有参数名统一补 Megatron 双 module 前缀。

为什么这样写：
- 模型专用 converter 通常基于 Megatron 训练时的完整参数名做匹配。
- 在进入 converter 前展开层/专家维，可以让 converter 不必理解 dist checkpoint 的堆叠保存形态。

不变量与失败模式：
- 堆叠 layer/expert tensor 的第 0 维必须匹配 `num_layers/num_experts`。
- 参数名正则不匹配时会走展开路径，命名异常可能导致 assert 或错误展开。

Comment：
这一步把 checkpoint 存储形态恢复为 converter 能理解的 Megatron 命名形态。

### 2.4 convert_to_hf 按 model_name 分发到模型专用 converter

问题与约束：
- 不同模型家族的 Megatron 参数名、QKV/Gate-Up 布局、MoE 命名和 HF 目标名都不同，无法用一个通用映射覆盖。

设计选择：
- `convert_to_hf` 先做 vocab padding 裁剪，再调用 `_convert_to_hf_core` 根据 `model_name` 分发到 deepseek/qwen/glm/llama 等 converter。

Explain：
`_convert_to_hf_core` 用字符串匹配选择 converter，无法匹配时抛出 unsupported model。`args.q_lora_rank is not None` 时还有一段 q_a/kv_a pairing cache 逻辑，用于兼容 SGLang 的 DeepSeek MLA 相关实现。

来源：slime/backends/megatron_utils/megatron_to_hf/__init__.py L24-L68

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
    elif "glm4moe" in model_name:
        converted_named_tensors = convert_glm4moe_to_hf(args, name, param)
    elif "qwen2" in model_name or "qwen3" in model_name:
        converted_named_tensors = convert_qwen2_to_hf(args, name, param)
    elif "llama" in model_name:
        converted_named_tensors = convert_llama_to_hf(args, name, param)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
```

代码逻辑：
- 先统一调用 `remove_padding`。
- 按模型名选择 converter。
- converter 返回一个或多个 HF named tensor。
- 最后可走 quantization post-process。

为什么这样写：
- 模型族差异集中到专用 converter，主脚本只负责 IO、分片和流程控制。
- unsupported model 直接失败，比生成错误 safetensors 更安全。

不变量与失败模式：
- `model_name` 必须能匹配已支持的模型分支。
- converter 必须返回 `(name, tensor)` 迭代结果。

Comment：
torch_dist→HF 的语义正确性主要取决于这个模型专用分发层。

### 2.5 remove_padding 裁掉 Megatron vocab padding

问题与约束：
- Megatron 训练常把 vocab padded 到并行友好的大小；HF embedding/output layer 需要真实 vocab size，否则权重 shape 与 config 不一致。

设计选择：
- 只对 `embedding.word_embeddings.weight` 和 `output_layer.weight` 执行 `param[:vocab_size]`，其他参数原样返回。

Explain：
函数先用 `strip_param_name_prefix` 去掉 Megatron 参数名前缀，再判断是否是词表相关权重。这个裁剪既可由 `save_tensors` 的 `--vocab-size` 触发，也会在 `convert_to_hf` 内按 `args.vocab_size` 执行。

来源：slime/backends/megatron_utils/megatron_to_hf/processors/padding_remover.py L6-L12

Code：

```python
def remove_padding(name: str, param: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Remove vocab padding: param[:vocab_size] for embedding/output layers, else unchanged.
    """
    if strip_param_name_prefix(name) in {"embedding.word_embeddings.weight", "output_layer.weight"}:
        return param[:vocab_size]
    return param
```

代码逻辑：
- 参数名先剥离常见前缀。
- 只裁剪 embedding 和 output layer。
- 非词表权重不变。

为什么这样写：
- 只有 vocab 维度会被 Megatron padding 影响 HF shape。
- 对其他参数裁剪会破坏模型结构。

不变量与失败模式：
- `vocab_size` 必须是真实 HF vocab size。
- 如果模型的词表参数名不在这两个标准名中，函数不会裁剪。

Comment：
转换后 embedding shape 不对时，通常要回到这里检查 `--vocab-size`。

### 2.6 save_tensors 分片写 safetensors 并可补原 HF 缺失键

问题与约束：
- HF 权重文件需要按 safetensors 分片保存，并生成 index；部分 buffer 或非标准键可能不在 Megatron 转换结果中，需要从原 HF 目录补齐。

设计选择：
- `save_tensors` 遍历 `get_named_params`，对每个 Megatron tensor 调 `convert_to_hf`，按 `chunk_size` 分片累积；如果传入 origin HF dir，则把未转换过的 safetensors key 追加到输出。

Explain：
函数维护 `converted_names` 防止重复补键，维护 `metadata["weight_map"]` 生成 `model.safetensors.index.json`。默认 chunk size 由 CLI 传入，旧文档中的默认值来自 main 参数。

来源：tools/convert_torch_dist_to_hf.py L106-L164

Code：

```python
def save_tensors(args, model_name, state_dict, output_dir, chunk_size, vocab_size=None, origin_hf_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    current_size = 0
    total_size = 0
    modeltensors = [{}]
    converted_names = set()
    for name, param in get_named_params(args, state_dict):
        if vocab_size:
            param = remove_padding(name, param, vocab_size)
        converted_named_tensors = convert_to_hf(args, model_name, name, param)
        for converted_name, converted_param in converted_named_tensors:
            converted_names.add(converted_name)
            tensor_size = converted_param.numel() * converted_param.element_size()
            if tensor_size + current_size > chunk_size:
                modeltensors.append({})
                current_size = 0
            modeltensors[-1][converted_name] = converted_param
            current_size += tensor_size
            total_size += tensor_size
```

```python
metadata = {"metadata": {"total_size": total_size}, "weight_map": {}}
for i, tensors in enumerate(modeltensors):
    filename = f"model-{i:05d}-of-{num_files:05d}.safetensors"
    for key in tensors.keys():
        metadata["weight_map"][key] = filename
json.dump(metadata, open(index_filepath, "w"), indent=2)
safetensors.torch.save_file(tensors, filepath)
```

代码逻辑：
- 每个 Megatron 参数可转换成多个 HF 参数。
- 当前分片超出 chunk size 时开启新分片。
- index json 记录每个 tensor 属于哪个文件。
- safetensors 文件逐分片写出。

为什么这样写：
- 大模型 HF checkpoint 必须分片，单文件会超过文件系统或工具限制。
- index json 是 transformers 加载分片 safetensors 的必要入口。

不变量与失败模式：
- converter 输出的 HF tensor name 不能重复，否则后写会覆盖同分片 dict 中的键。
- `chunk_size` 太小可能导致大量碎片文件。

Comment：
这一步把抽象的 name/tensor stream 落成 HF 生态能直接加载的文件布局。

### 2.7 copy_assets 复制非权重 HF 附属文件

问题与约束：
- safetensors 只包含权重；HF 模型目录还需要 tokenizer、config、generation config 等非权重文件。

设计选择：
- `copy_assets` 遍历 origin HF dir，跳过 safetensors 和 index，只复制普通文件到 output dir。

Explain：
函数不递归目录，只处理 `os.path.isfile` 为真的文件；目录会被 skip。这样可避免把原始权重文件重新复制到导出目录，同时保留必要的文本/json 附属文件。

来源：tools/convert_torch_dist_to_hf.py L165-L175

Code：

```python
def copy_assets(origin_hf_dir, output_dir):
    for filename in os.listdir(origin_hf_dir):
        if filename == "model.safetensors.index.json" or filename.endswith(".safetensors"):
            continue
        origin_filename = os.path.join(origin_hf_dir, filename)
        if not os.path.isfile(origin_filename):
            print(f"Skip {filename}, not a file.")
            continue
        src, dst = origin_filename, os.path.join(output_dir, filename)
        shutil.copy(src, dst)
```

代码逻辑：
- safetensors 和 index 被过滤。
- 非文件路径被跳过。
- 其他文件复制到输出目录同名位置。

为什么这样写：
- 输出目录中的权重应来自转换结果，不应混入旧 safetensors。
- tokenizer/config 等文件无需转换，直接复用原 HF 目录即可。

不变量与失败模式：
- origin HF dir 必须存在。
- 嵌套资源目录不会被复制，若模型依赖目录资产需要额外处理。

Comment：
导出 HF checkpoint 时，权重转换和资产复制是两件事。

### 2.8 主程序串起 CLI 校验、common.pt、dist_cp 加载和导出

问题与约束：
- 导出脚本需要知道输入 dist checkpoint、输出目录、模型名、原 HF 资产目录、分片大小和 vocab 裁剪参数；如果没有 model name 或 origin config，就无法选择 converter。

设计选择：
- main 用 argparse 解析参数；输出目录已存在且未 force 时拒绝；`model_name` 缺失时从 `origin_hf_dir` 的 `AutoConfig` 推断；再读取 `common.pt` 的 Megatron args，用 dist_cp 单进程加载 state_dict。

Explain：
`_load_state_dict(..., no_dist=True)` 说明导出是单进程读取 dist checkpoint。`save_tensors` 的 `origin_hf_dir` 参数只有在 `--add-missing-from-origin-hf` 开启时传入；但 `copy_assets` 只要提供 origin dir 就会执行。

来源：tools/convert_torch_dist_to_hf.py L178-L244

Code：

```python
parser.add_argument("--model-name", type=str, default=None)
parser.add_argument("--input-dir", type=str, required=True)
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--origin-hf-dir", type=str, default=None)
parser.add_argument("-f", "--force", action="store_true")
parser.add_argument("-a", "--add-missing-from-origin-hf", action="store_true")
parser.add_argument("--chunk-size", type=int, default=5 * 1024**3)
parser.add_argument("--vocab-size", type=int, default=None)
args = parser.parse_args()

if os.path.exists(args.output_dir) and not args.force:
    raise ValueError(...)
if args.model_name is None and args.origin_hf_dir is None:
    raise ValueError(...)
if args.model_name is None:
    hf_config = AutoConfig.from_pretrained(args.origin_hf_dir, trust_remote_code=True)
    args.model_name = type(hf_config).__name__.lower()
```

```python
state_dict = {}
megatron_args = torch.load(os.path.join(args.input_dir, "common.pt"), weights_only=False)["args"]
dist_cp.state_dict_loader._load_state_dict(
    state_dict,
    storage_reader=WrappedStorageReader(args.input_dir),
    planner=EmptyStateDictLoadPlanner(),
    no_dist=True,
)
save_tensors(..., args.origin_hf_dir if args.add_missing_from_origin_hf else None)
if args.origin_hf_dir:
    copy_assets(args.origin_hf_dir, args.output_dir)
```

代码逻辑：
- CLI 必填 input/output。
- model name 可显式给出或从 origin HF config 推断。
- `common.pt` 提供 Megatron args。
- dist_cp 读取权重后交给 `save_tensors`。
- origin HF dir 可用于补权重键和复制附属文件。

为什么这样写：
- 模型名是 converter 分发的关键，不能缺失。
- 单进程 `no_dist=True` 导出更适合离线转换，不需要启动 Megatron 分布式程序。

不变量与失败模式：
- input dir 必须包含 `common.pt` 和 `.metadata`。
- output dir 已存在时必须显式 `--force`。
- 未提供 `model_name` 时，origin HF dir 必须能被 `AutoConfig` 读取。

Comment：
torch_dist→HF 主程序是离线导出流程，不复用 Megatron distributed 初始化。

---

## 3. 对照小结

### 3.1 两个方向的边界不同

问题与约束：
- HF→torch_dist 和 torch_dist→HF 都叫“转换”，但运行时假设完全不同：前者要进入 Megatron 分布式环境，后者是单进程读取 checkpoint 并写 HF 文件。

设计选择：
- 前者复用 Megatron `parse_args/init/save_checkpoint` 和 mbridge；后者复用 torch distributed checkpoint reader、Slime converter 和 safetensors writer。

Explain：
HF→torch_dist 的可信输出是 Megatron `release` checkpoint，适合训练加载；torch_dist→HF 的可信输出是 safetensors 分片和 HF 附属文件，适合推理、发布或 SGLang 侧加载。

来源：tools/convert_hf_to_torch_dist.py L121-L148

Code：

```python
bridge = AutoBridge.from_pretrained(hf_model_path, trust_remote_code=True)
bridge.load_weights(model, hf_model_path, memory_efficient=True)
save_checkpoint(1, model, None, None, 0)

if dist.get_rank() == 0:
    tracker_filename = get_checkpoint_tracker_filename(args.save)
    with open(tracker_filename, "w") as f:
        f.write("release")
    source_dir = get_checkpoint_name(args.save, 1, False, return_base_dir=True)
    target_dir = get_checkpoint_name(args.save, -1, True, return_base_dir=True)
    shutil.move(source_dir, target_dir)
```

代码逻辑：
- HF→torch_dist 先构建 Megatron 模型。
- mbridge 把 HF 权重加载进 Megatron 结构。
- Megatron 原生 checkpoint API 写出 dist checkpoint。
- rank 0 整理成 release 目录。

为什么这样写：
- 训练侧最可靠的格式是 Megatron 自己保存的 checkpoint。
- 反向导出不需要重建训练分布式拓扑，只需要还原权重名和 shape。

不变量与失败模式：
- 正向转换输出给训练，不保证直接是 HF 目录。
- 反向导出输出给 HF/SGLang，不包含 Megatron optimizer/scheduler。

Comment：
把这两个脚本放在一起看，能看清 Slime 在训练格式和推理格式之间的桥接边界。
