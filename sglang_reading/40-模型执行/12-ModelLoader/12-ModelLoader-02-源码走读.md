---
type: batch-doc
module: 12-ModelLoader
batch: "12"
doc_type: walkthrough
title: "ModelLoader · 源码走读"
tags:
 - sglang/batch/12
 - sglang/module/model-loader
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-05
---
# ModelLoader · 源码走读

## 1. 默认权重加载路径

### 1.1 `DefaultModelLoader.Source` 封装权重来源与前缀语义

问题与约束：
- 同一个 loader 要支持本地路径、HF model id、revision 和附加权重前缀。
- 某些模型在加载时允许从 safetensors 回退到 `.pt`。
- loader extra config 只允许少量参数，避免隐藏配置拼错。

设计选择：
- `DefaultModelLoader` 内部定义 `Source` dataclass，记录 `model_or_path/revision/prefix/fall_back_to_pt/model_config`。
- `Source.init_new` 从 `ModelConfig` 和模型实例读取默认路径、revision 与 fallback 标志。
- `DefaultModelLoader.__init__` 校验 extra config key 只允许 multithread load 相关字段。

Explain：
`Source` 是权重 iterator 的输入描述：它不加载 tensor，只告诉 loader 从哪里找权重、是否加前缀、是否允许 `.pt` fallback。

来源：python/sglang/srt/model_loader/loader.py L352-L405

Code：

```python
class DefaultModelLoader(BaseModelLoader):
    """Model loader that can load different file types from disk."""

    DEFAULT_NUM_THREADS = 8
    _MTP_PATTERN = re.compile(r"model\.mtp\.layers\.(\d+)\.")

    @dataclasses.dataclass
    class Source:
        model_or_path: str
        revision: Optional[str]
        prefix: str = ""
        fall_back_to_pt: bool = True
        model_config: Optional[ModelConfig] = None

        @classmethod
        def init_new(cls, model_config: ModelConfig, model):
            return cls(
                model_config.model_path,
                model_config.revision,
                prefix="",
                fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
                model_config=model_config,
            )

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = load_config.model_loader_extra_config
        allowed_keys = {"enable_multithread_load", "num_threads"}
        unexpected_keys = set(extra_config.keys()) - allowed_keys

        if unexpected_keys:
            raise ValueError(...)
```

代码逻辑：
- 定义默认并行加载线程数和 MTP 权重名匹配 pattern。
- `Source` 保存权重来源元数据。
- `init_new` 从 model config 生成默认 source。
- loader 初始化时读取 extra config。
- 如果出现非白名单 key，直接抛错。

为什么这样写：
- 权重来源与加载执行分离，便于后续复用同一 iterator 逻辑。
- prefix 支持 MTP 等附加权重映射，不需要修改每个模型类。
- extra config 白名单能在启动阶段发现拼写错误，而不是静默忽略。

不变量与失败模式：
- `model_config.model_path` 和 `revision` 必须能定位权重。
- `fall_back_to_pt` 由模型属性控制，模型类需要明确是否允许 fallback。
- unexpected extra config 会抛 `ValueError`。

Comment：
默认 loader 的第一层抽象不是 tensor，而是“权重来源描述”。

### 1.2 `_prepare_weights` 选择格式、下载并过滤权重文件

问题与约束：
- 模型权重可能是 safetensors、bin、pt、Mistral consolidated 格式或本地目录。
- 远程模型需要按 allow/ignore patterns 下载。
- sharded safetensors 和 consolidated safetensors 混在一起时不能全部加载。
- 推理不需要的文件应过滤掉。

设计选择：
- 按 `LoadFormat` 选择 `allow_patterns`、`use_safetensors` 和 index file。
- 非本地路径调用 `download_weights_from_hf`，本地路径直接使用。
- 可选执行 checksum 校验。
- safetensors 路径用 index 文件过滤 duplicate；非 safetensors 路径过滤推理不需要的文件。

Explain：
`_prepare_weights` 把“模型路径或 id”变成可迭代的本地权重文件列表，并确定是否走 safetensors 读取路径。

来源：python/sglang/srt/model_loader/loader.py L431-L520

Code：

```python
def _prepare_weights(
    self, model_name_or_path: str, revision: Optional[str], fall_back_to_pt: bool
) -> Tuple[str, List[str], bool]:
    model_name_or_path = self._maybe_download_from_modelscope(
        model_name_or_path, revision
    )

    is_local = os.path.isdir(model_name_or_path)
    load_format = self.load_config.load_format
    use_safetensors = False
    index_file = SAFE_WEIGHTS_INDEX_NAME
    if load_format == LoadFormat.AUTO:
        allow_patterns = ["*.safetensors", "*.bin"]
    elif load_format == LoadFormat.SAFETENSORS or load_format == LoadFormat.FASTSAFETENSORS:
        use_safetensors = True
        allow_patterns = ["*.safetensors"]
    elif load_format == LoadFormat.MISTRAL:
        use_safetensors = True
        allow_patterns = ["consolidated*.safetensors"]
        index_file = "consolidated.safetensors.index.json"
    elif load_format == LoadFormat.PT:
        allow_patterns = ["*.pt"]
    elif load_format == LoadFormat.NPCACHE:
        allow_patterns = ["*.bin"]
    elif load_format == LoadFormat.DUMMY:
        raise ValueError(...)
    else:
        raise ValueError(f"Unknown load_format: {load_format}")

    if fall_back_to_pt:
        allow_patterns += ["*.pt"]

    if not is_local:
        hf_folder = download_weights_from_hf(...)
    else:
        hf_folder = model_name_or_path

    hf_weights_files: List[str] = []
    for pattern in allow_patterns:
        hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
        if len(hf_weights_files) > 0:
            if pattern == "*.safetensors":
                use_safetensors = True
            break

    if use_safetensors:
        hf_weights_files = filter_duplicate_safetensors_files(
            hf_weights_files, hf_folder, index_file
        )
    else:
        hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files)

    if len(hf_weights_files) == 0:
        raise RuntimeError(...)
```

代码逻辑：
- 先处理 ModelScope 下载开关。
- 根据路径是否本地决定后续是否下载。
- 按 load format 选择文件 pattern。
- fallback 开启时追加 `.pt`。
- 找到第一类匹配权重文件。
- safetensors 走 duplicate 过滤，其他格式过滤非推理文件。
- 没有任何权重文件时抛错。

为什么这样写：
- AUTO 模式优先 safetensors/bin，覆盖主流 HF checkpoint。
- Mistral consolidated 格式需要特定 index 名称。
- duplicate safetensors 不过滤会重复加载同一权重。
- DUMMY 格式由 DummyModelLoader 处理，不能进入真实文件准备。

不变量与失败模式：
- `load_config.load_format` 必须属于支持集合。
- 远程下载失败或本地目录无匹配文件会导致启动失败。
- fallback 到 `.pt` 可能改变加载格式，模型需允许该行为。

Comment：
这一步是文件层面的“收敛”：把各种外部 checkpoint 形态收敛为本地文件列表。

### 1.3 `load_model` 初始化模型、灌权重并做 postprocess

问题与约束：
- 模型初始化需要遵守目标 dtype 和设备。
- 量化配置可能影响参数类、weight loader 和后处理。
- 某些在线量化路径需要在加载权重时临时设置环境变量。

设计选择：
- `load_model` 先解析 quant config，在 `set_default_torch_dtype` 和目标 device 下 `_initialize_model`。
- 然后调用 `load_weights_and_postprocess`，内部执行 `model.load_weights(weights)`。
- NVFP4 online 路径临时禁用 fast math，加载后同步并清 cache。
- 权重加载完成后遍历 modules 执行 quant method postprocess。

Explain：
DefaultModelLoader 的主路径是“构造空模型 → 生成权重 iterator → 模型侧 load_weights → 量化后处理 → eval”。loader 不直接理解每个参数如何切片，而是把 `(name, tensor)` 交给模型类处理。

来源：python/sglang/srt/model_loader/loader.py L741-L812

Code：

```python
def load_model(
    self,
    *,
    model_config: ModelConfig,
    device_config: DeviceConfig,
) -> nn.Module:

    if hasattr(model_config, "modelopt_quant") and model_config.modelopt_quant:
        model = self._load_modelopt_base_model(model_config)
        return model.eval()

    target_device = torch.device(device_config.device)
    quant_config = _get_quantization_config(model_config, self.load_config)
    with set_default_torch_dtype(model_config.dtype):
        with target_device:
            model = _initialize_model(
                model_config,
                self.load_config,
                quant_config,
            )

        self.load_weights_and_postprocess(
            model, self._get_all_weights(model_config, model), target_device
        )

    self.counter_after_loading_weights = time.perf_counter()
    return model.eval()

@staticmethod
def load_weights_and_postprocess(model, weights, target_device):
    quant_config = getattr(model, "quant_config", None)
    is_nvfp4_online = getattr(quant_config, "is_nvfp4_online", False)

    if is_nvfp4_online:
        with temp_set_env(
            TRTLLM_DISABLE_FP4_QUANT_FAST_MATH="1",
            FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH="1",
        ):
            model.load_weights(weights)
        if target_device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    else:
        model.load_weights(weights)
```

代码逻辑：
- ModelOpt 特殊路径直接加载 base model。
- 解析目标设备和量化配置。
- 设置默认 dtype，并在目标 device 上初始化模型。
- 取出所有权重 iterator 并交给 postprocess 函数。
- postprocess 根据量化配置选择普通加载或 NVFP4 online 特殊环境。
- 返回 eval mode 模型。

为什么这样写：
- 模型结构和参数 dtype/device 必须在权重进入前确定。
- 权重切片由模型类实现，loader 只负责 iterator 与后处理。
- online FP4 需要精确加载期数学，临时环境变量能把影响限制在加载阶段。

不变量与失败模式：
- `_initialize_model` 必须根据 quant_config 创建兼容参数。
- `model.load_weights` 必须消费 iterator 并处理 missing/extra 参数。
- 量化 postprocess 需要模块暴露 `quant_method` 约定。

Comment：
DefaultModelLoader 是编排者；真正的参数名映射在具体模型类里发生。

### 1.4 模型类 `load_weights` 处理 fused/sharded 参数映射

问题与约束：
- checkpoint 权重名不总是和运行时参数名一一对应，QKV/MLP 等 fused 参数需要映射。
- TP 切片由参数自带的 `weight_loader` 完成。
- 一些 checkpoint 会包含推理不需要的 bias、旧 kv scale 或额外视觉塔权重。

设计选择：
- Llama `load_weights` 遍历权重名，对 `stacked_params_mapping` 做 name replace 和 shard id 加载。
- 普通参数从 `params_dict` 查找，优先用参数上的 `weight_loader`，否则用 `default_weight_loader`。
- 对已知可跳过的 extra bias、kv_scale、tie embedding 等直接 continue。

Explain：
模型侧加载负责把外部 checkpoint 命名空间转成运行时参数命名空间。loader 传入 `(name, tensor)`，模型类决定这个 tensor 应该落到哪个参数、哪个 shard。

来源：python/sglang/srt/models/llama.py L660-L700

Code：

```python
if name.startswith("model.vision_tower") and name not in params_dict:
    continue
if self.config.tie_word_embeddings and "lm_head.weight" in name:
    continue
if "scale" in name:
    name = maybe_remap_kv_scale_name(name, params_dict)
    if name is None:
        continue

for param_name, weight_name, shard_id in stacked_params_mapping:
    if weight_name not in name:
        continue
    name = name.replace(weight_name, param_name)
    if name.endswith(".bias") and name not in params_dict:
        continue
    if name not in params_dict:
        continue
    param = params_dict[name]
    weight_loader = param.weight_loader
    weight_loader(param, loaded_weight, shard_id)
    break
else:
    if name.endswith(".bias") and name not in params_dict:
        continue
    if name.endswith(".kv_scale") and name not in params_dict:
        continue
    if name in params_dict.keys():
        param = params_dict[name]
        weight_loader = getattr(
            param, "weight_loader", default_weight_loader
        )
        weight_loader(param, loaded_weight)
    else:
        logger.warning(f"Parameter {name} not found in params_dict")
```

代码逻辑：
- 跳过不属于当前模型参数字典的视觉塔权重。
- tie embeddings 时跳过 lm head 权重。
- scale 权重先尝试 remap。
- fused/stacked 参数命中 mapping 时替换名称并带 shard id 调 loader。
- 普通参数直接按 name 查找。
- 未找到参数则记录 warning。

为什么这样写：
- 运行时参数可能是 fused 形态，checkpoint 可能是拆分形态，必须在模型类里映射。
- 每个参数自己的 `weight_loader` 知道 TP/量化切片规则。
- 对常见额外权重静默跳过，避免旧 checkpoint 或量化 checkpoint 无谓失败。

不变量与失败模式：
- `params_dict` 必须覆盖当前运行时模型参数。
- `stacked_params_mapping` 顺序会影响先匹配哪个 fused 参数。
- 真正缺失的重要参数只会 warning，后续功能是否正确取决于模型完整性检查。

Comment：
理解 ModelLoader 时要记住：loader 找文件，模型类决定 tensor 怎样落参数。

## 2. 下载与量化配置

### 2.1 `download_weights_from_hf` 统一 HF 下载入口

问题与约束：
- 远程 HF 模型需要下载到 cache dir。
- 只应下载权重文件，而不是仓库全部内容。
- 本地路径不需要走 HF API。
- 多进程并发下载和校验可能产生竞争。

设计选择：
- 函数签名接收 model path、cache dir、allow patterns、revision、ignore patterns 和 retry 次数。
- 本地目录直接返回。
- 远程路径后续使用单一锁覆盖校验、清理与下载流程。

Explain：
`download_weights_from_hf` 是 `_prepare_weights` 的远程下载后端。它用 allow/ignore patterns 控制文件集合，并把本地路径作为快速路径。

来源：python/sglang/srt/model_loader/weight_utils.py L517-L550

Code：

```python
def download_weights_from_hf(
    model_name_or_path: str,
    cache_dir: Optional[str],
    allow_patterns: List[str],
    revision: Optional[str] = None,
    ignore_patterns: Optional[Union[str, List[str]]] = None,
    max_retries: int = 3,
) -> str:
    """Download model weights from Hugging Face Hub."""
    if os.path.isdir(model_name_or_path):
        return model_name_or_path

    # Use a SINGLE lock for the entire operation (validation + cleanup + download)
    # to prevent race conditions where:
```

代码逻辑：
- 接收下载目标与过滤规则。
- 如果输入已经是本地目录，直接返回。
- 远程路径进入带锁的下载/校验逻辑。
- `max_retries` 用于处理检测到损坏后的重试。

为什么这样写：
- 本地目录不应触发网络或 HF cache 行为。
- allow/ignore patterns 与 `_prepare_weights` 的 load format 决策对齐。
- 单锁保护能减少并发进程同时删除/下载同一文件的竞态。

不变量与失败模式：
- 本地路径必须是目录；单文件模型由对应 loader 单独处理。
- allow patterns 为空会导致后续无法定位权重。
- 远程下载受 HF offline、revision 和网络状态影响。

Comment：
下载函数本身不理解模型结构，只负责把符合 pattern 的权重放到本地。

### 2.2 `get_quant_config` 从 HF/text/compression 配置中解析量化

问题与约束：
- 量化配置可能存放在 HF config、text_config 或 compression_config。
- GGUF 没有同样形式的 config 文件。
- bitsandbytes/QLoRA 可能需要 adapter path。
- packed modules mapping 要传给量化 config，才能处理 fused 模块。

设计选择：
- 根据 `model_config.quantization` 获取 quantization config class。
- GGUF 直接用空 dict 构造。
- 依次查找 `hf_config.quantization_config`、`text_config.quantization_config`、`hf_config.compression_config`。
- 找到配置后转 dict，并注入 `packed_modules_mapping`。

Explain：
量化配置解析把各种 HF 生态中的配置位置收敛成 SGLang 内部 `QuantizationConfig`。这一步发生在模型初始化前，影响参数创建和后续 weight loader。

来源：python/sglang/srt/model_loader/weight_utils.py L237-L270

Code：

```python
def get_quant_config(
    model_config: ModelConfig,
    load_config: LoadConfig,
    packed_modules_mapping: Dict[str, List[str]],
    remap_prefix: Dict[str, str] | None = None,
) -> QuantizationConfig:
    quant_cls = get_quantization_config(model_config.quantization)

    if model_config.quantization == "gguf":
        return quant_cls.from_config({})

    hf_quant_config = getattr(model_config.hf_config, "quantization_config", None)
    hf_text_config = getattr(model_config.hf_config, "text_config", None)
    if hf_quant_config is None and hf_text_config is not None:
        hf_quant_config = getattr(hf_text_config, "quantization_config", None)
    if hf_quant_config is None:
        hf_quant_config = getattr(model_config.hf_config, "compression_config", None)
    if hf_quant_config is not None:
        if not isinstance(hf_quant_config, dict):
            hf_quant_config = hf_quant_config.to_dict()
        hf_quant_config["packed_modules_mapping"] = packed_modules_mapping
        return quant_cls.from_config(hf_quant_config)
```

代码逻辑：
- 取得当前量化类型对应的 config class。
- GGUF 返回空配置。
- 查 HF config 的 quantization config。
- 如果是 vision/text 组合模型，再查 text_config。
- 再查 compression_config。
- 非 dict 配置转 dict。
- 注入 packed modules mapping 并构造 QuantizationConfig。

为什么这样写：
- 不同模型仓库的量化元数据位置不统一。
- packed module 信息必须进入 quant config，否则 fused 参数加载和量化后处理无法对齐。
- GGUF 的量化信息从文件本身读取，不依赖 HF config。

不变量与失败模式：
- `model_config.quantization` 必须能映射到 quantization config class。
- HF config 中的量化字段若格式异常，会在 `from_config` 暴露。
- bitsandbytes/QLoRA 分支还需要额外 adapter 配置。

Comment：
量化配置不是附属信息，它会改变模型参数类型和权重加载方式。

## 3. Loader 变体

### 3.1 `RemoteInstanceModelLoader` 从远端实例拉取权重

问题与约束：
- PD 分离、热扩容或恢复场景可能不想从磁盘重新读 checkpoint。
- 远端实例权重传输需要限制 load format 和 backend。
- 远程 connector 可能有不同类型，loader 必须验证。

设计选择：
- `RemoteInstanceModelLoader` 禁止 extra config，并要求 load format 为 `REMOTE_INSTANCE`。
- 先在本地初始化同构模型。
- NCCL backend 下构造 `instance://ip:port` 权重地址，创建 remote connector。
- 只接受 `ConnectorType.INSTANCE`，否则抛错。

Explain：
远程实例 loader 的目标是“模型结构本地建，权重从已运行实例传”。这适合让新 worker 快速对齐已有 serving 实例。

来源：python/sglang/srt/model_loader/loader.py L2194-L2244

Code：

```python
class RemoteInstanceModelLoader(BaseModelLoader):
    """Model loader that can load Tensors from remote sglang instance."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(...)
        self.remote_instance_transfer_engine_weight_info = None

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        logger.info("Loading weights from remote instance ...")
        load_config = self.load_config

        assert load_config.load_format == LoadFormat.REMOTE_INSTANCE, (...)

        quant_config = _get_quantization_config(model_config, self.load_config)
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config, quant_config)

        if (
            load_config.remote_instance_weight_loader_backend
            == RemoteInstanceWeightLoaderBackend.NCCL
        ):
            model_weights = f"instance://{load_config.remote_instance_weight_loader_seed_instance_ip}:{load_config.remote_instance_weight_loader_send_weights_group_ports[load_config.tp_rank]}"
            with create_remote_connector(model_weights, device_config.device) as client:
                connector_type = get_connector_type(client)
                if connector_type == ConnectorType.INSTANCE:
                    self.load_model_from_remote_instance_by_nccl(
                        model, client, model_config, device_config
                    )
                else:
                    raise ValueError(...)
```

代码逻辑：
- 初始化时拒绝 extra config。
- load 时断言 load format。
- 解析量化配置并初始化模型。
- NCCL backend 下构造 instance URL。
- 创建 remote connector。
- 检查 connector 类型。
- 调 NCCL 远端加载实现。

为什么这样写：
- 模型结构仍由本地 config 决定，保证新实例模块布局匹配。
- 远程权重传输依赖严格的 backend 和 connector 类型，提前校验可避免错误协议读写 tensor。
- 使用 tp_rank 选择对应端口，匹配张量并行分片。

不变量与失败模式：
- `load_format` 必须是 `REMOTE_INSTANCE`。
- seed instance IP 和 send weight ports 必须配置完整。
- connector 类型不支持会抛 `ValueError`。

Comment：
这个 loader 把 checkpoint IO 换成实例间 tensor transfer。

### 3.2 `GGUFModelLoader` 将单文件 GGUF 作为特殊格式处理

问题与约束：
- GGUF 模型通常是单文件格式，不符合 HF sharded safetensors/bin 目录假设。
- GGUF 需要额外 Python 包解析。
- loader extra config 不适用于该格式。

设计选择：
- `GGUFModelLoader` 禁止 extra config。
- `_prepare_weights` 只接受文件路径，不接受目录或 model id。
- `_get_gguf_weights_map` 延迟导入 `gguf`，并在缺失时给出安装提示。

Explain：
GGUF loader 把“文件本身就是权重容器”作为特殊路径处理。它不走默认 HF 文件扫描，而是后续用 GGUF 解析逻辑按 tensor 解码。

来源：python/sglang/srt/model_loader/loader.py L2086-L2125

Code：

```python
class GGUFModelLoader(BaseModelLoader):
    """
    Model loader that can load GGUF files. This is useful for loading models
    that are quantized with GGUF and saved in the GGUF format. This loader
    supports loading both full models and sharded models.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(...)

    def _prepare_weights(self, model_name_or_path: str):
        if os.path.isfile(model_name_or_path):
            return model_name_or_path
        else:
            raise ValueError(f"{model_name_or_path} is not a file.")

    def _get_gguf_weights_map(self, model_config: ModelConfig):
        try:
            import gguf
        except ImportError as err:
            raise ImportError(
                "Please install gguf via `pip install gguf` to use gguf quantizer."
            ) from err
```

代码逻辑：
- 初始化时拒绝 extra config。
- 准备权重时要求输入是文件。
- 构建 GGUF weights map 时延迟导入 gguf。
- 缺包时抛带安装提示的 ImportError。

为什么这样写：
- GGUF 不适合默认 `_prepare_weights` 的目录 glob 策略。
- 延迟导入避免普通加载路径强依赖 gguf 包。
- 文件路径校验能尽早暴露用户把目录传给 GGUF loader 的错误。

不变量与失败模式：
- `model_name_or_path` 必须是文件。
- 环境必须安装 `gguf`。
- GGUF tensor name 还需要后续 map 到 HF/SGLang 参数命名。

Comment：
GGUF 是格式级 loader，不只是默认 loader 的另一个 allow pattern。

### 3.3 `LayeredModelLoader` 用 meta 初始化和逐层 materialize 降低峰值

问题与约束：
- 超大模型加载时，整模型同时 materialize 再量化会推高峰值显存。
- 逐层加载要求模型实现 `load_weights_to_module`。
- torchao 量化可以在单层加载后立即处理。

设计选择：
- Layered loader 将 load format 重置为 AUTO，并继承 DefaultModelLoader。
- 在 meta device 上初始化模型。
- 检查模型是否支持 `load_weights_to_module`。
- 递归遍历模块，逐模块 `to_empty(target_device)`、加载该模块权重，并可选应用 torchao config。

Explain：
LayeredModelLoader 将权重加载从“整模型灌入”改成“模块递归 materialize + 加载 + 量化”。这减少了同时驻留的未处理权重和参数数量。

来源：python/sglang/srt/model_loader/loader.py L824-L895

Code：

```python
class LayeredModelLoader(DefaultModelLoader):
    """Model loader that loads weights layer by layer so that one can quantize a
    layer before loading another to make the peak memory envelope smaller."""

    def __init__(self, load_config: LoadConfig):
        load_config.load_format = LoadFormat.AUTO
        super().__init__(load_config)

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        target_device = torch.device(device_config.device)
        quant_config = _get_quantization_config(model_config, self.load_config)

        with set_default_torch_dtype(model_config.dtype):
            with torch.device("meta"):
                model = _initialize_model(
                    model_config,
                    self.load_config,
                    quant_config,
                )

            if not hasattr(model, "load_weights_to_module"):
                raise ValueError(...)

            weights = self._get_all_weights(model_config, model)

            def fill_module(module, fqn: List[str], weights):
                for name, submod in module.named_children():
                    fill_module(submod, fqn + [name], weights)

                module.to_empty(device=target_device, recurse=False)
                fqn_path = ".".join(fqn)
                model.load_weights_to_module(
                    fqn_path,
                    weights,
                )
                if torchao_config and "proj" in fqn_path:
                    apply_torchao_config_to_model(module, torchao_config, None)

            fill_module(model, [], weights)

        if torchao_config:
            model.torchao_applied = True

        return model.eval()
```

代码逻辑：
- 构造时把 load format 回到 AUTO。
- 目标设备和 quant config 准备好。
- 在 meta device 初始化模型。
- 检查模型层级加载接口。
- 获取权重 iterator。
- 递归处理子模块。
- 每个模块 materialize 到目标设备并加载对应权重。
- 可选应用 torchao。
- 返回 eval 模型。

为什么这样写：
- meta 初始化避免一开始就为全模型分配真实参数存储。
- 模块级加载允许加载后立即量化或释放中间状态。
- 需要模型配合 `load_weights_to_module`，因为默认 `load_weights` 不知道当前模块边界。

不变量与失败模式：
- 模型必须实现 `load_weights_to_module`。
- 权重 iterator 需要能被逐模块消费或复用。
- torchao 只对匹配条件的模块应用，配置错误会在应用时暴露。

Comment：
Layered loader 是为加载峰值内存服务的，不是为了改变最终模型结构。

### 3.4 `DummyModelLoader` 构造随机权重模型用于压测

问题与约束：
- 调度、内存池或 serving 压测有时不需要真实 checkpoint。
- dummy 路径不能下载权重。
- 模型结构仍要和指定 config 对齐，才能覆盖真实推理框架路径。

设计选择：
- `DummyModelLoader` 禁止 extra config，`download_model` 为空操作。
- `load_model` 仍解析 quant config 并初始化模型。
- CPU quantization 环境变量启用时走专门路径。
- 普通路径用 `initialize_dummy_weights(model)` 赋随机值。

Explain：
Dummy loader 跳过 checkpoint IO，但不跳过模型构造。这样可以在没有真实权重的情况下测试 server 初始化、调度和内存路径。

来源：python/sglang/srt/model_loader/loader.py L1371-L1410

Code：

```python
class DummyModelLoader(BaseModelLoader):
    """Model loader that will set model weights to random values."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(...)

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:

        if get_bool_env_var("SGL_CPU_QUANTIZATION"):
            return load_model_with_cpu_quantization(
                self, model_config=model_config, device_config=device_config
            )

        quant_config = _get_quantization_config(model_config, self.load_config)

        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(
                    model_config,
                    self.load_config,
                    quant_config,
                )

            initialize_dummy_weights(model)
```

代码逻辑：
- 初始化时校验 extra config。
- 下载阶段不做任何事。
- load 时检查 CPU quantization 特殊路径。
- 解析量化配置。
- 按 dtype/device 初始化模型。
- 随机初始化权重。

为什么这样写：
- 保留模型结构能覆盖 kernel、memory pool、scheduler 等路径。
- 跳过真实 checkpoint IO 能快速做压测或调试。
- extra config 禁止避免用户误以为 dummy loader 会消费加载参数。

不变量与失败模式：
- dummy 权重不能用于正确性评测。
- 模型 config 仍必须可初始化。
- CPU quantization 路径需要对应环境和函数支持。

Comment：
Dummy loader 的目标是工程路径验证，而不是模型语义。

## 4. 权重同步打包

### 4.1 `FlattenedTensorBucket` 用 metadata 重建远端传输 tensor

问题与约束：
- 权重同步或远端加载需要把多个 tensor 打包传输。
- 接收端要恢复 tensor 名称、dtype 和 shape。
- 重建应尽量少分配和少拷贝。

设计选择：
- `reconstruct_tensors` 预分配结果 list。
- 对每条 metadata，从 flattened tensor 切片，按 metadata dtype view，再 reshape 成原 shape。
- 返回 `(name, tensor)` 列表。

Explain：
FlattenedTensorBucket 把跨进程传输的连续 buffer 还原回模型加载可消费的 named tensors。metadata 是恢复边界的关键。

来源：python/sglang/srt/weight_sync/tensor_bucket.py L90-L107

Code：

```python
def reconstruct_tensors(self) -> List[Tuple[str, torch.Tensor]]:
    """
    Reconstruct original tensors from flattened tensor with optimized performance.
    Uses memory-efficient operations to minimize allocations and copies.
    """
    reconstructed = [None] * len(self.metadata)

    for i, meta in enumerate(self.metadata):
        tensor = (
            self.flattened_tensor[meta.start_idx : meta.end_idx]
            .view(meta.dtype)
            .reshape(meta.shape)
        )

        reconstructed[i] = (meta.name, tensor)

    return reconstructed
```

代码逻辑：
- 按 metadata 长度预分配结果 list。
- 遍历每个 metadata。
- 用 start/end index 从 flattened buffer 切片。
- 使用 metadata dtype 解释切片。
- reshape 为原 tensor shape。
- 写入 `(name, tensor)`。

为什么这样写：
- 传输时连续 buffer 更适合通信后端。
- 重建时 view/reshape 尽量复用底层存储，减少额外拷贝。
- named tensor 列表可以直接接回模型加载或权重更新路径。

不变量与失败模式：
- metadata 的 start/end、dtype、shape 必须和 flatten 时一致。
- 切片长度必须能按 dtype 和 shape reshape。
- flattened tensor 生命周期要覆盖重建 tensor 的使用期。

Comment：
权重加载不只来自磁盘；远端同步路径也会产出同样的 `(name, tensor)` 语义。
