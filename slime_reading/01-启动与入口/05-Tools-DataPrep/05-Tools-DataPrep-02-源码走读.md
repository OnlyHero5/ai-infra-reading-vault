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
updated: 2026-07-02
---

# Tools-DataPrep · 源码走读

> 按**执行顺序**精读两个转换脚本。基线 commit `22cdc6e1`。

## 1. HF → torch_dist：CLI 扩展

**Explain：** `add_convertion_args` 在 Megatron `parse_args` 之上增加 Slime 转换专用参数；其中 `--hf-checkpoint` 为必填。

**Code：**

```python
# 来源：tools/convert_hf_to_torch_dist.py L21-L41
# 提交版本：22cdc6e1
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
    return parser
```

**Comment：**

- `--megatron-to-hf-mode` 影响训练后 **update_weights** 路径，转换脚本本身以 bridge 灌 HF 权重
- `--allgather-cp` 与 context parallel 相关；需 DSA 架构才安全（见 `arguments.py` 校验）

## 2. get_args：Megatron 校验与 PP 自动推导

**Explain：** `get_args()` 调用 Megatron `parse_args` + Slime defaults，并在大 world_size 时自动选择 pipeline parallel，使每层至少一张 GPU。

**Code：**

```python
# 来源：tools/convert_hf_to_torch_dist.py L44-L84
# 提交版本：22cdc6e1
def get_args():
    args = parse_args(add_convertion_args)
    args = set_default_megatron_args(args)
    args.save_interval = 1
    args.micro_batch_size = 1
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.global_batch_size = int(os.environ.get("WORLD_SIZE", "1"))
    assert world_size <= args.num_layers, (
        f"World size {world_size} must be less than or equal to number of layers {args.num_layers}."
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

**Comment：**

- `world_size <= num_layers`：PP 按层切分，GPU 数不能超过层数
- 仅当用户未设 PP（==1）且多卡时才自动推导；显式 PP 时尊重 CLI
- `validate_args` 是 Megatron 内置一致性检查（TP/PP/层数组合）

## 3. main：分布式初始化

**Explain：** 转换脚本自行 `init_process_group`，兼容 `torchrun` 与 Slurm 环境变量。

**Code：**

```python
# 来源：tools/convert_hf_to_torch_dist.py L97-L115
# 提交版本：22cdc6e1
world_size = int(os.getenv("WORLD_SIZE") or os.getenv("SLURM_NTASKS") or 1)
local_rank = int(os.getenv("LOCAL_RANK") or os.getenv("SLURM_LOCALID") or 0)
global_rank = int(os.getenv("RANK") or os.getenv("SLURM_PROCID") or 0)
torch.cuda.set_device(local_rank)
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
```

**Comment：**

- `init(args)` 来自 `slime.backends.megatron_utils.initialize`，与训练 Actor 同一套 Megatron 初始化
- ROCm 分支在 main 开头 patch `FileSystemWriterAsync`（下一节）

## 4. ROCm / AMD 特殊路径

**Explain：** HIP 环境下替换异步 checkpoint writer，并强制 CPU 初始化。

**Code：**

```python
# 来源：tools/convert_hf_to_torch_dist.py L87-L93, L117-L120
# 提交版本：22cdc6e1
if torch.version.hip:
    from slime.utils.rocm_checkpoint_writer import ROCmFileSystemWriterAsync
    filesystem_async_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync
# ...
if hasattr(torch.version, "hip") and torch.version.hip is not None:
    assert args.use_cpu_initialization, "AMD GPU requires --use_cpu_initialization=True"
```

**Comment：**

- AMD 转换在 CPU 上构图再 save，避免 HIP 兼容问题
- NVIDIA 路径默认 GPU 初始化；大模型可配合 `--use-cpu-initialization`

## 5. 建模、灌权重、保存

**Explain：** `get_model` 构建 Megatron 模型（无 DDP）；AutoBridge 从 HF 目录加载；`save_checkpoint` 写 dist ckpt。

**Code：**

```python
# 来源：tools/convert_hf_to_torch_dist.py L121-L137
# 提交版本：22cdc6e1
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

**Comment：**

- `wrap_with_ddp=False`：转换阶段不需要 DDP wrapper
- `save_checkpoint(1, ...)` 固定 iteration 1，随后 rank 0 rename 为 release
- 无 optimizer / scheduler 参数（全传 `None`）

## 6. release 目录重命名

**Explain：** rank 0 写 tracker 为 `release`，并把 iter 目录 move 到 Megatron 约定的 release 路径。

**Code：**

```python
# 来源：tools/convert_hf_to_torch_dist.py L139-L148
# 提交版本：22cdc6e1
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

**Comment：**

- 训练脚本 `--ref-load` 通常指向含 `release/` 的根目录
- 其他 rank 只参与 save_checkpoint 的 collective，不做文件 move

---

## 7. torch_dist → HF：Pickle 安全加载

**Explain：** Megatron checkpoint metadata 可能引用训练环境类；`UnpicklerWrapper` 把 megatron/glm 类替换为 Dummy，避免 import 失败。

**Code：**

```python
# 来源：tools/convert_torch_dist_to_hf.py L19-L31
# 提交版本：22cdc6e1
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

**Comment：**

- 全局替换 `pickle.Unpickler` 影响后续 metadata 读取
- 只读转换，不实例化真实 Megatron 训练对象

## 8. WrappedStorageReader 与 EmptyStateDictLoadPlanner

**Explain：** 自定义 reader/planner 跳过 optimizer state，按 metadata 预分配 empty tensor 再加载权重 shard。

**Code：**

```python
# 来源：tools/convert_torch_dist_to_hf.py L34-L63
# 提交版本：22cdc6e1
class WrappedStorageReader(dist_cp.FileSystemReader):
    @override
    def read_metadata(self):
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = UnpicklerWrapper(metadata_file).load()
        # ... storage_meta / planner_data 补全 ...
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

**Comment：**

- 过滤 `optimizer` / `_state` 键，减小内存与转换时间
- `no_dist=True` 加载（见 §10）适合单进程导出

## 9. 层 / Expert 参数展开

**Explain：** dist ckpt 中部分 tensor 按层或 expert 维堆叠；`get_named_params` 递归展开并加 `module.module.` 前缀以匹配 Megatron 命名。

**Code：**

```python
# 来源：tools/convert_torch_dist_to_hf.py L66-L103
# 提交版本：22cdc6e1
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
            yield expert_name, param[expert_id]
    else:
        yield name, param

def get_named_params(args, state_dict):
    for name, param in state_dict.items():
        name = f"module.module.{name}"
        yield from get_layer_param(args, name, param)
```

**Comment：**

- MoE 模型走 expert 展开分支；dense 模型 mostly 直通
- `megatron_args` 来自 `common.pt`，提供 `num_layers` / `num_experts`

## 10. 主程序：加载、转换、写 safetensors

**Explain：** CLI 解析 → 读 `common.pt` → dist_cp 加载 state_dict → `save_tensors` + 可选 `copy_assets`。

**Code：**

```python
# 来源：tools/convert_torch_dist_to_hf.py L178-L244
# 提交版本：22cdc6e1
parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", type=str, required=True)
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--origin-hf-dir", type=str, default=None)
parser.add_argument("-f", "--force", action="store_true")
parser.add_argument("-a", "--add-missing-from-origin-hf", action="store_true")
parser.add_argument("--chunk-size", type=int, default=5 * 1024**3)
parser.add_argument("--vocab-size", type=int, default=None)
args = parser.parse_args()
# ...
megatron_args = torch.load(os.path.join(args.input_dir, "common.pt"), weights_only=False)["args"]
dist_cp.state_dict_loader._load_state_dict(
    state_dict,
    storage_reader=WrappedStorageReader(args.input_dir),
    planner=EmptyStateDictLoadPlanner(),
    no_dist=True,
)
save_tensors(megatron_args, args.model_name, state_dict, args.output_dir, ...)
if args.origin_hf_dir:
    copy_assets(args.origin_hf_dir, args.output_dir)
```

**Comment：**

- `--input-dir` 指向 `iter_xxx/` 或 `release/` 子目录（含 `.metadata`）
- `model_name` 默认从 `origin_hf_dir` 的 `AutoConfig` 推断
- `copy_assets` 复制 tokenizer、config 等非权重文件

## 11. save_tensors 与 convert_to_hf

**Explain：** 逐 tensor 调 `convert_to_hf` 做 Megatron→HF 名字/形状映射，按 chunk_size 分片写 safetensors。

**Code：**

```python
# 来源：tools/convert_torch_dist_to_hf.py L106-L126
# 提交版本：22cdc6e1
def save_tensors(args, model_name, state_dict, output_dir, chunk_size, vocab_size=None, origin_hf_dir=None):
    modeltensors = [{}]
    current_size = 0
    for name, param in get_named_params(args, state_dict):
        if vocab_size:
            param = remove_padding(name, param, vocab_size)
        converted_named_tensors = convert_to_hf(args, model_name, name, param)
        for converted_name, converted_param in converted_named_tensors:
            tensor_size = converted_param.numel() * converted_param.element_size()
            if tensor_size + current_size > chunk_size:
                modeltensors.append({})
                current_size = 0
            modeltensors[-1][converted_name] = converted_param
            current_size += tensor_size
```

**Comment：**

- `convert_to_hf` 在 `megatron_to_hf/__init__.py` 按 `model_name` 分发到 qwen/glm/deepseek 等 converter
- 默认 chunk 5GB；生成 `model-00000-of-xxxxx.safetensors` + index json
- `--add-missing-from-origin-hf` 从原始 HF 补全未转换键（如部分 buffer）

## 12. remove_padding 机制

**Explain：** 对 embedding / output_layer 裁切到真实 vocab_size，解决 Megatron padding 导致的 HF 不对齐。

**Code：**

```python
# 来源：slime/backends/megatron_utils/megatron_to_hf/processors/padding_remover.py L6-L12
# 提交版本：22cdc6e1
def remove_padding(name: str, param: torch.Tensor, vocab_size: int) -> torch.Tensor:
    if strip_param_name_prefix(name) in {"embedding.word_embeddings.weight", "output_layer.weight"}:
        return param[:vocab_size]
    return param
```

**Comment：**

- quick_start 提醒：转换后 embedding 不对时需手动 `--vocab-size`
- `convert_to_hf` 内部也会用 `args.vocab_size` 调一次 remove_padding

---

**走读小结**

| 步骤 | HF→torch_dist | torch_dist→HF |
|------|---------------|---------------|
| 参数 | Megatron parse_args + MODEL_ARGS | argparse + common.pt |
| 并行 | torchrun + 自动 PP | 单进程 no_dist |
| 核心 IO | AutoBridge.load_weights | dist_cp load + convert_to_hf |
| 产出 | release/ torch_dist | safetensors + HF 附属文件 |
