---
type: batch-doc
module: 00-方法论
batch: "01"
doc_type: walkthrough
title: "方法论 · 源码走读"
tags:
  - slime/batch/01
  - slime/module/methodology
  - slime/doc/walkthrough
updated: 2026-07-05
---

# 方法论 · 源码走读

> 本专题走读 README、愿景博文、参数透传代码与打包元数据。目标不是背诵项目宣传语，而是建立阅读 Slime 运行时源码时的设计坐标：为什么它把 Megatron 训练、SGLang rollout、自定义数据生成和 Data Buffer 放在同一条闭环里。

---

## 1. README 里的系统边界

### 1.1 两大能力：训练和数据生成不是两个框架

**问题与约束：** RL 后训练同时需要高吞吐训练和灵活 rollout。若把 agent workflow、reward/verifier、环境交互都拆成独立框架，训练侧和生成侧会反复做数据契约适配。

**设计选择：** README 把 Slime 的核心能力压成两项：Megatron+SGLang 的高性能训练，以及可自定义的数据生成。后续源码阅读应围绕这两条能力如何共享同一闭环展开。

**Explain：** 这段是方法论起点：Slime 不是“Trainer 外挂一个 rollout 服务”，而是把训练与生成都作为 RL loop 的组成部分。

**Code：**

```text
来源：README.md L9-L16
**slime** is an LLM post-training framework for RL scaling, providing two core capabilities:

1.  **High-Performance Training**: Supports efficient training in various modes by connecting Megatron with SGLang;
2.  **Flexible Data Generation**: Enables arbitrary training data generation workflows through custom data generation interfaces and server-based engines.

slime's design goal is to make these two capabilities reinforce each other without turning the system into a heavy stack of disconnected trainers, rollout services, and agent frameworks.
```

**代码逻辑：** README 先定义框架目标，再明确“不把系统变成割裂的 trainer、rollout service、agent framework”。

**为什么这样写：** 这决定了源码组织方式：训练、rollout、reward、environment 不会各自拥有一套主循环，而是围绕共同的数据路径和 Ray 编排协作。

**不变量与失败模式：** 如果读者把 Slime 当作“Megatron wrapper”或“agent framework”，会误读许多参数和 hook。它的核心不变量是：训练和数据生成互相强化，不拆成互不透明的栈。

**Comment：** 后续所有模块都可以回到这个问题：它是在增强训练侧、生成侧，还是在维护两者之间的闭环。

### 1.2 统一 Data Buffer 路径：扩展点不 fork training kernel

**问题与约束：** math/code/search/tool/sandbox 等任务差异很大，但训练 kernel 不应该因为每种任务都改一遍，否则 RL 系统无法长期维护。

**设计选择：** README 明确 Megatron training、SGLang rollout、custom data generation、reward/verifier、environment interaction 都流经同一条 training / rollout / Data Buffer path。

**Explain：** 这段是 Slime 对“可扩展性”的真正定义：不是给每种任务独立插件树，而是把样本生产都规约到同一条数据通路。

**Code：**

```text
来源：README.md L14-L16
slime's design goal is to make these two capabilities reinforce each other without turning the system into a heavy stack of disconnected trainers, rollout services, and agent frameworks. Megatron training, SGLang rollout, custom data generation, reward computation, verifier feedback, and environment interaction all flow through the same training / rollout / Data Buffer path.
```

**代码逻辑：** README 把多个异构活动并列到同一个 path 中，强调它们在系统层不是独立栈。

**为什么这样写：** Data Buffer 是训练和 rollout 的契约面。只要产物能进入这个契约，agentic workflow 就不必 fork 主训练逻辑。

**不变量与失败模式：** 自定义生成可以任意复杂，但最终必须产出训练可消费的 sample/tensor；如果绕过 Data Buffer，checkpoint、replay、debug 和权重同步都会被切断。

**Comment：** 读源码时应优先追踪“样本如何进入 Data Buffer”，而不是被某个 example 的 agent 细节带偏。

### 1.3 架构三模块：training、rollout、data buffer

**问题与约束：** 用户需要快速知道 Slime 的最小系统分解；如果一开始就列 Ray actor、SGLang server、Megatron arguments，会丢失主线。

**设计选择：** 中文 README 用三模块描述：training 读 Data Buffer 并同步权重；rollout 生成数据并写 Data Buffer；data buffer 管 prompt/custom data/rollout 方法。

**Explain：** 这不是只是图示说明，而是阅读顺序提示：先理解三个角色，再进入每个角色内部。

**Code：**

```text
来源：README_zh.md L89-L93
- **training (Megatron)**：负责主训练流程，从 Data Buffer 读取数据，训练完后将参数同步至 rollout 模块；
- **rollout (SGLang + router)**：生成新数据（含 reward/verifier），存储至 Data Buffer；通过 custom generate 可以在其上叠加 multi-turn loop、tool call、environment/sandbox 交互以及 verifier-based reward；
- **data buffer**：桥梁模块，管理 prompt 初始化、自定义数据与 rollout 生成方法（包括以同一套接口产出 sample 的 agentic workflow）。
```

**代码逻辑：** 三个 bullet 分别给出生产者、消费者和桥梁。training 与 rollout 通过 data buffer 和权重同步形成闭环。

**为什么这样写：** 三模块分解让读者不把 Slime 误解成单向 pipeline。训练会消费 rollout 数据，训练结果又通过权重同步反过来影响下一轮 rollout。

**不变量与失败模式：** Data Buffer 不是普通队列标签，而是训练/rollout 契约中心；如果忽略它，后续 `RolloutManager`、sample conversion、`ray.put` 都会显得零散。

**Comment：** 这也是本 vault 的 Slime 阅读主线：训练主循环、Ray 编排、Rollout 生成、权重同步都围绕这个三角关系展开。

### 1.4 Agentic 示例：复杂任务挂到 customization 接口

**问题与约束：** Agentic RL 示例包括 multi-agent、search/RAG、fully async、coding agent，形态差异很大。框架不能为每类任务硬编码一套主流程。

**设计选择：** README 把这些 example 都描述为通过 `--rollout-function-path` 或 `--custom-generate-function-path` 接入标准 rollout / Data Buffer 闭环。

**Explain：** 这段证据说明 agentic workflow 是 Slime 的扩展使用方式，不是独立子框架。

**Code：**

```text
来源：README.md L103-L108
- [`examples/multi_agent`](examples/multi_agent/README.md): Multi-agent rollout via a custom `--rollout-function-path`.
- [`examples/search-r1`](examples/search-r1/): Search/RAG-style multi-turn generation via `--custom-generate-function-path`.
- [`examples/fully_async`](examples/fully_async/README.md): Fully-async rollout, useful for long-tail agentic generation where some samples take much longer than others.
- [`examples/coding_agent_rl`](examples/coding_agent_rl/README.md): End-to-end SWE coding-agent RL with sandboxed tool use, test-based rewards, and token-correct trajectory segments via `--custom-generate-function-path`.
```

**代码逻辑：** 每个 example 都绑定到 path 型 hook，而不是绑定到新的 trainer 类型。

**为什么这样写：** path hook 把业务差异推到用户函数，框架保留共同的训练、同步、调试、数据契约。这样新 agent 类型不需要改变 Slime kernel。

**不变量与失败模式：** hook 可以复杂，但不能破坏 sample contract；`--rollout-function-path` 和 `--custom-generate-function-path` 粒度不同，混用会导致生成路径和数据转换位置不清。

**Comment：** 读 examples 时要问：这个例子是在替换整个 rollout，还是只替换 generate 函数。

---

## 2. 愿景博文里的设计哲学

### 2.1 Versatile / Performant / Maintainable：三目标同时成立

**问题与约束：** RL 框架常在灵活性、性能和可维护性之间摇摆：越通用越慢，越性能特化越难扩展，越抽象越难 debug。

**设计选择：** 博文直接把 Slime 定义为三目标：customizable rollout interface、SGLang+Megatron native performance、lightweight maintainable codebase。

**Explain：** 这是后续源码取舍的评价标准。某段代码如果看起来“少封装”，往往是为了 maintainable 和可实验；某段参数透传如果看起来“宽”，是为了 native performance。

**Code：**

```text
来源：docs/en/blogs/introducing_slime.md L15-L21
- **Versatile** – with a fully customizable rollout interface and flexible training setups (colocated or decoupled, synchronous or asynchronous, RL or SFT cold start).
- **Performant** - integrating SGLang for inference and Megatron-LM for training, natively.
- **Maintainable** - with a lightweight codebase and smooth transition from Megatron pretraining to SGLang deployment.

In short, a post-training framework for RL scaling.
```

**代码逻辑：** 博文先列能力目标，再把它们归纳成 RL scaling 的 post-training framework。

**为什么这样写：** Slime 的设计不是泛化到所有 backend，而是围绕 RL scaling 所需的最短路径：训练、推理、同步、生成自由度。

**不变量与失败模式：** 若某个扩展牺牲 native engine 能力或把主循环隐藏进复杂继承树，它就偏离了这三个目标。

**Comment：** 这三词可以作为代码审查标尺：这段实现是在提升灵活性、性能、可维护性中的哪一个。

### 2.2 暴露 `train.py` 主循环：让同步策略可移动

**问题与约束：** RL 同步/异步策略经常要实验；如果主循环被包在深层 Trainer class 中，移动 `ray.get`、调整 rollout/train 边界会很困难。

**设计选择：** 博文强调 Slime 不用 trainer class 包住代码，而是暴露入口 `train.py` 中的 training loop。

**Explain：** 这解释了为什么后续源码会偏脚本式主循环，而不是框架式继承树。

**Code：**

```text
来源：docs/en/blogs/introducing_slime.md L45-L45
And to make experimenting with different strategies easy, we didn't wrap the code with trainer classes, but simply exposed the training loop in entrypoint  `train.py`.
```

**代码逻辑：** 单句直接给出设计理由：为了方便不同同步策略实验。

**为什么这样写：** RL 系统工程的关键变化点常在编排层，而不是单个模型 forward。主循环可见，用户才能安全地改数据等待、rollout 触发和 train step 顺序。

**不变量与失败模式：** 暴露主循环不等于允许随意破坏契约；移动同步点时仍要保持样本版本、权重版本和 Data Buffer 消费关系一致。

**Comment：** 阅读 [[02-训练主循环-02-源码走读]] 时，应把“可移动同步点”作为主线，而不是只看函数调用顺序。

### 2.3 SGLang-native：server-based、参数透传、rollout-only debug

**问题与约束：** RL rollout 对 inference 性能敏感；如果框架包掉 SGLang 参数或把 server 生命周期藏起来，用户很难复现 standalone SGLang 的性能调优。

**设计选择：** 博文把 SGLang-native 拆成三项：内部 server-based mode、所有 SGLang 参数通过 `--sglang` 前缀透传、提供 rollout-only debug。

**Explain：** 这段解释了 Slime 为什么不是“抽象 inference backend 接口”：它选择深度保留 SGLang 控制面。

**Code：**

```text
来源：docs/en/blogs/introducing_slime.md L57-L59
- slime internally launches SGLang servers in a **server-based mode**.
- slime implements **seamless pass-through** for all SGLang parameters (with a `--sglang` prefix), ensuring that all optimization options can be enabled. For instance, you can pass `--sglang-enable-ep-moe`, `--sglang-enable-dp-attention` and `--sglang-enable-deepep-moe` for the powerful multi-node MoE inference capabilities.
- slime provides an **SGLang-only debug mode** (`--debug-rollout-only`) for easy performance tuning.
```

**代码逻辑：** 三个 bullet 分别对应部署形态、参数控制面和调试入口。

**为什么这样写：** SGLang 迭代速度快，RL 框架不应把上游优化延迟到自己重新封装。透传让新参数可以随 SGLang 升级立即使用。

**不变量与失败模式：** SGLang 参数透传必须有明确前缀，避免和 Slime/Megatron 参数冲突；debug-rollout-only 必须绕开训练侧，才能定位 serving 性能问题。

**Comment：** 后续看到大量 `--sglang-*` 参数时，不要把它们当作配置噪声；它们是 native 设计的直接产物。

### 2.4 参数透传实现：monkey-patch argparse

**问题与约束：** SGLang `ServerArgs.add_cli_args` 原本会注册自己的参数名；Slime 需要复用它，但必须给参数加 `--sglang-` 前缀，并跳过与 Slime 自身拓扑配置冲突的字段。

**设计选择：** `add_sglang_arguments` 临时替换 `parser.add_argument`，在 wrapper 里改写 flag 和 dest，再调用 `ServerArgs.add_cli_args(parser)`。

**Explain：** 这段代码是 “native pass-through” 的具体实现。它没有手写复制 SGLang 参数表，而是复用上游注册函数。

**Code：**

```python
来源：slime/backends/sglang_utils/arguments.py L65-L91
def new_add_argument_wrapper(*name_or_flags, **kwargs):
    canonical_name_for_skip_check = None
    if "dest" in kwargs:
        canonical_name_for_skip_check = kwargs["dest"]
    else:
        for flag_name_candidate in name_or_flags:
            if isinstance(flag_name_candidate, str) and flag_name_candidate.startswith("--"):
                stem = flag_name_candidate[2:]
                canonical_name_for_skip_check = stem.replace("-", "_")
                break

    if canonical_name_for_skip_check and canonical_name_for_skip_check in skipped_args:
        return
```

**代码逻辑：** wrapper 先推导原始参数名用于 skip；未跳过时把所有 flag 改成 `--sglang-*`，并按需要给 `dest` 加 `sglang_` 前缀。

**为什么这样写：** 这种写法牺牲一点 argparse 透明度，换来上游参数自动同步。Slime 不需要维护一份容易过期的 SGLang 参数镜像。

**不变量与失败模式：** wrapper 调用后必须恢复 `parser.add_argument`；否则后续 Slime 自身参数也会被错误加前缀。`skipped_args` 必须覆盖拓扑/端口/分布式等由 Slime 接管的字段。

**Comment：** 这是 Slime “native but controlled” 的典型例子：尽量复用上游，但在边界处拦住会破坏本框架拓扑的不安全参数。

### 2.5 RL 特化：权重更新不是普通 serving 问题

**问题与约束：** RL 训练会频繁更新模型权重；普通在线推理通常假设权重相对静态。若 serving backend 不支持高效更新，rollout 会被同步开销拖垮。

**设计选择：** 博文把 weight update optimization 列为 Slime 与 SGLang 协同优化点，包括 MoE 并行策略下参数更新和 bucketed update。

**Explain：** 这段解释了为什么 Slime 的权重同步专题是核心路径，而不是附加工具。

**Code：**

```text
来源：docs/en/blogs/introducing_slime.md L77-L80
**Optimizing weight updates**: Unlike inference tasks, RL training involves frequent updates to model weights. To address this, we’ve introduced several optimizations in SGLang:
    
  - Parameter updates for MoE models under various parallelism strategies ([#6265](https://github.com/sgl-project/sglang/pull/6265), [#6308](https://github.com/sgl-project/sglang/pull/6308), [#6311](https://github.com/sgl-project/sglang/pull/6311)).
```

**代码逻辑：** 文档先指出 RL 与普通 inference 的差异，再列出 SGLang 侧需要配合的更新优化。

**为什么这样写：** Slime 选择 SGLang-native 后，可以推动 serving backend 暴露 RL 所需能力，而不是在 Slime 外层用低效方式搬权重。

**不变量与失败模式：** 权重版本必须和 rollout 样本关联清楚；如果只追求快速更新而忽略版本记录，训练会混入无法解释的 stale samples。

**Comment：** 读 [[24-WeightSync-Dist-00-MOC]] 时，应把它看作 RL loop 的内在需求，而不是部署脚本优化。

### 2.6 Dynamic sampling：`/abort_request` 支持 partial rollout

**问题与约束：** DAPO 等算法会 oversampling；当已收集到足够样本时，剩余生成继续跑会浪费 rollout 时间，并拉长训练等待尾部。

**设计选择：** Slime 与 SGLang 侧设计 `/abort_request`，让正在生成的请求可以被立即终止，并回收 partial content。

**Explain：** 这段体现 Slime 的另一个 native 取舍：RL 生成控制需要 inference server 提供普通 chat API 之外的控制端点。

**Code：**

```text
来源：docs/en/blogs/introducing_slime.md L82-L86
**`/abort_request` for dynamic sampling**: In RL algorithms that require oversampling, such as [DAPO](https://arxiv.org/abs/2503.14476), some requests may continue running even after sufficient data has been collected. In collaboration with the [AReal](https://github.com/inclusionAI/AReaL) team, we designed an new endpoint: `/abort_request`. This endpoint enables:
    
  - Immediate termination of on-going requests.
  - Reclaiming partially generated content, which enables partial rollouts.
```

**代码逻辑：** 文档把 oversampling 的尾部浪费和 server 端 abort endpoint 直接连接起来。

**为什么这样写：** RL rollout 的瓶颈常在长尾生成；只有训练框架和 serving backend 协同，才能在算法层已经满足条件时主动停止无用推理。

**不变量与失败模式：** abort 后回收的 partial rollout 必须被标记并按训练逻辑处理；若当作完整样本训练，会污染 loss mask 或 reward 语义。

**Comment：** 这也是为什么 Slime 的 rollout 章节需要同时看算法参数和 SGLang server 控制 API。

### 2.7 轻量四件套：复杂度移到用户 pipeline 和核心库

**问题与约束：** 框架要支持多种 RL workflow，但不能把每种 workflow 的业务逻辑都吸收到框架核心；否则核心会变成难维护的巨型 trainer。

**设计选择：** 博文把 Slime 本体压成四件事：custom rollout interface、Ray 资源与异步、SGLang+Megatron 集成、训练与推理间权重更新。

**Explain：** 这是 Slime 的“少即是多”：框架只保留闭环中不可替代的系统能力。

**Code：**

```text
来源：docs/en/blogs/introducing_slime.md L92-L98
Focusing on customization and performance, slime:

1. Provides a customizable rollout interface.
2. Uses Ray for GPU management and asynchronous execution.
3. Integrates SGLang for inference and Megatron for training.
4. Provides weight updates between training and inference.

Pretty straightforward, right? slime transfers complexity from the framework to user-defined pipelines and core libraries (SGLang and Megatron), resulting in a lightweight, easily maintainable codebase.
```

**代码逻辑：** 列表先给出框架职责，随后指出复杂度被转移到 user-defined pipelines 和核心库。

**为什么这样写：** Slime 不试图定义所有 rollout 语义，而是定义可插拔边界。这样框架核心可读，业务复杂度留在外层。

**不变量与失败模式：** 如果把复杂业务逻辑塞回框架核心，会破坏 maintainable；如果用户 pipeline 不遵守 sample contract，又会破坏训练正确性。

**Comment：** 这段是阅读“未独立成专题导读”的判断标准：不进入核心的复杂性，不一定是缺失，可能是刻意外置。

### 2.8 扩展到 SFT / Rejection Sampling：同一后训练基座

**问题与约束：** RL、SFT、rejection sampling 都是后训练工作流，但训练目标和数据来源不同。框架若只能跑 PPO/GRPO，会限制复用。

**设计选择：** 博文把 SFT 和 rejection sampling 描述为模块化设计的自然延伸：SFT 用 Megatron token prediction loss；rejection sampling 用 SGLang filter 后接 Megatron SFT。

**Explain：** 这说明 Slime 的抽象边界不是“某个 RL 算法”，而是“训练 backend + 数据生成/过滤 + 权重/数据通路”。

**Code：**

```text
来源：docs/en/blogs/introducing_slime.md L105-L106
- **SFT**: Load Megatron and use token prediction loss.
- **Rejection Sampling**: Use SGLang for filter, followed by Megatron SFT.
```

**代码逻辑：** 两个 bullet 分别把训练目标和数据生成方式替换掉，但复用同一 Megatron/SGLang 基座。

**为什么这样写：** 后训练系统的价值在于可复用数据和执行基础设施；算法变化不应迫使用户换掉整个 runtime。

**不变量与失败模式：** SFT/rejection sampling 复用基础设施时，仍要重新检查数据字段、loss mask 和 reward/label 语义，不能把 RL 样本契约直接套用。

**Comment：** 读 Slime 时不要把所有代码都理解为“RL 专用”。有些边界是为更广义后训练预留的。

---

## 3. 包装与依赖暴露工程边界

### 3.1 `bdist_wheel`：wheel 不是纯 Python 假设

**问题与约束：** Slime 依赖 GPU/分布式训练生态，安装包需要明确平台标签；如果 wheel 被当成 pure Python，部署环境可能错误复用不兼容产物。

**设计选择：** `setup.py` 自定义 `bdist_wheel`，把 `root_is_pure` 设为 False，并生成 Python/ABI/platform tag。

**Explain：** 这段不是训练逻辑，但体现 Slime 的工程定位：它是 GPU/RL 基础设施，不是无平台差异的工具库。

**Code：**

```python
来源：setup.py L13-L28
class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    def get_tag(self):
        python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
        abi_tag = f"{python_version}"

        if platform.system() == "Linux":
            platform_tag = "manylinux1_x86_64"
        else:
            platform_tag = platform.system().lower()

        return python_version, abi_tag, platform_tag
```

**代码逻辑：** wheel finalize 时标记非 pure；`get_tag` 根据 Python 版本和平台返回 tag。

**为什么这样写：** 即便核心包本身以 Python 为主，实际运行依赖 CUDA、Megatron、SGLang、router 等平台敏感组件。打包层提前表达这个事实。

**不变量与失败模式：** 平台 tag 过于粗糙可能掩盖 CUDA/架构差异；但 pure wheel 更危险，会让部署工具误以为跨平台完全兼容。

**Comment：** 读依赖和打包文件时要关注它们暴露的运行假设：GPU、分布式、server-based rollout 都不是可选背景。

### 3.2 `_fetch_requirements`：依赖列表保持单一来源

**问题与约束：** requirements 和 setup install_requires 如果手动维护两份，容易出现安装路径与开发路径不一致。

**设计选择：** `setup.py` 用 `_fetch_requirements("requirements.txt")` 读取依赖，把 requirements 文件作为单一来源。

**Explain：** 这段小函数连接了开发安装和包安装的依赖面。

**Code：**

```python
来源：setup.py L8-L10
def _fetch_requirements(path):
    with open(path) as fd:
        return [r.strip() for r in fd.readlines() if r.strip() and not r.startswith("#")]
```

**代码逻辑：** 读取文件每行，过滤空行和注释行，返回 stripped dependency list。

**为什么这样写：** Slime 的依赖面已经足够复杂，保持一个列表能减少 setup 与 pip install 行为分叉。

**不变量与失败模式：** 该函数只忽略整行注释，不剥离行尾注释；requirements 中的 inline comment 要能被 pip/setuptools 正确处理，否则会影响安装。

**Comment：** 简单依赖读取也有边界：它适合当前 requirements 风格，但不是通用 parser。

### 3.3 `requirements.txt`：依赖映射到 RL 闭环组件

**问题与约束：** Slime 不是只跑训练脚本。它需要数据集、HTTP 客户端、Ray、router、tensor/weight 文件、监控和压缩校验等组件共同支撑闭环。

**设计选择：** requirements 显式列出 `ray[default]`、`sglang-router`、`httpx[http2]`、`datasets`、`safetensors`、`xxhash`、`zstandard` 等闭环相关依赖。

**Explain：** 依赖列表是系统边界的另一个索引。它告诉读者哪些能力是 Slime 运行时会直接用到的。

**Code：**

```text
来源：requirements.txt L1-L26
accelerate
anthropic
blake3
blobfile
datasets
e2b
httpx[http2]
mcp[cli]
memray
numba
omegaconf
openai
openai-agents
pillow
pylatexenc
pyyaml
qwen_vl_utils
ray[default]
ring_flash_attn
safetensors
sglang-router>=0.2.3
tensorboard
transformers
wandb
xxhash
zstandard
```

**代码逻辑：** 文件列出运行和实验常用依赖，没有按子系统分组，但能按功能映射到数据、编排、router、监控、权重同步和 agent/tool 生态。

**为什么这样写：** Slime 选择把多类 workflow 接入同一闭环，因此依赖面必然覆盖训练、serving、agent、数据和调试工具。

**不变量与失败模式：** 依赖增加不能替代清晰边界；如果某个 dependency 只服务特定 example，应考虑是否应移到 extra/dev，否则会扩大基础安装负担。

**Comment：** 这张依赖表帮助读者预判源码里会遇到的外部系统：Ray、SGLang router、OpenAI-compatible client、权重文件和观测工具。

### 3.4 README 参数三类入口：Megatron、SGLang、Slime 自身

**问题与约束：** 一个 RL 训练命令同时控制训练 backend、rollout backend 和框架自身编排；如果参数命名边界不清，用户很难判断某个 flag 应传给谁。

**设计选择：** README 把参数分成三类：Megatron 原生参数、带 `--sglang-` 前缀的 SGLang 参数、Slime 自身参数。

**Explain：** 这段是参数系统的用户可见契约，也对应后续 `parse_args` 的三阶段合并。

**Code：**

```text
来源：README.md L164-L168
Arguments in slime are divided into three categories:

1.  **Megatron arguments**: slime reads Megatron arguments directly. You can configure Megatron by passing arguments like `--tensor-model-parallel-size 2`.
2.  **SGLang arguments**: All arguments for the installed SGLang are supported through pass-through. These arguments must be prefixed with `--sglang-`. For example, `--mem-fraction-static` should be passed as `--sglang-mem-fraction-static`.
3.  **slime-specific arguments**: Please refer to: [slime/utils/arguments.py](slime/utils/arguments.py)
```

**代码逻辑：** 文档按归属划分参数，而不是按功能场景划分。

**为什么这样写：** 参数归属决定谁解释该参数、谁验证该参数、参数最终影响哪一层运行时。清晰的归属能减少跨 backend 配置冲突。

**不变量与失败模式：** SGLang 参数必须加前缀；Megatron 参数直接读取；Slime 参数负责编排。若用户把三者混用，可能出现参数被 parse 但不生效，或被错误 backend 消费。

**Comment：** 进入 [[03-Arguments-Ray-02-源码走读]] 和 [[04-Arguments-TrainRollout-02-源码走读]] 时，这三类入口就是读参数代码的地图。

---

## 走读小结

| 证据 | 设计含义 |
|------|----------|
| README 两大能力 | Slime 的边界是训练 + 数据生成闭环 |
| Data Buffer path | 自定义 workflow 不 fork training kernel |
| 愿景博文 | Versatile / Performant / Maintainable 是解释源码取舍的标尺 |
| `arguments.py` 透传 | native engine 通过前缀和 skip list 控制边界 |
| `setup.py` / requirements | Slime 是 GPU/RL 基础设施，不是纯 Python 小库 |

下一步进入 [[02-训练主循环-02-源码走读]]：把这些方法论落到可执行入口 `train.py`。
