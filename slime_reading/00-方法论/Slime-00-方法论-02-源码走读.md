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
updated: 2026-07-02
---

# 方法论 · 源码走读

> 本专题走读 **项目自述与安装元数据**，建立阅读 Slime 运行时源码的心智模型。  
> 基线 commit `22cdc6e1`。

---

## §1 README：两大能力与统一数据通路

**Explain：** 中英文 README 结构一致；核心是「两大能力 + 一条 data path」。

**Code：**

```python
## 来源：README.md L9-L16
# **slime** is an LLM post-training framework for RL scaling, providing two core capabilities:
# 1.  **High-Performance Training**: Supports efficient training in various modes
#     by connecting Megatron with SGLang;
# 2.  **Flexible Data Generation**: Enables arbitrary training data generation workflows
#     through custom data generation interfaces and server-based engines.
```

**Code：**

```python
## 来源：README.md L14-L16
# Megatron training, SGLang rollout, custom data generation, reward computation,
# verifier feedback, and environment interaction all flow through the same
# training / rollout / Data Buffer path.
```

**Comment：**

- math / code / search / tool / sandbox / multi-agent 都作为 **data generation 或 reward workflow** 接入，不 fork training kernel
- 与 [[Slime-00-方法论-01-核心概念]] §5 的 veRL 对比呼应

---

## §2 README：架构三模块

**Explain：** `## Architecture Overview` 对应 `imgs/arch.png` 的三段文字说明。

**Code：**

```python
## 来源：README_zh.md L89-L93
# - **training (Megatron)**：负责主训练流程，从 Data Buffer 读取数据，
#   训练完后将参数同步至 rollout 模块；
# - **rollout (SGLang + router)**：生成新数据（含 reward/verifier），存储至 Data Buffer；
# - **data buffer**：桥梁模块，管理 prompt 初始化、自定义数据与 rollout 生成方法。
```

**Comment：**

- 「存储至 Data Buffer」在实现上是 RolloutManager 把 sample 转成 train tensor 并 `ray.put`
- custom generate 可叠加 multi-turn / tool / environment（README Quick Start Agentic 示例）

---

## §3 README：Agentic 示例挂载方式

**Explain：** Agent 工作负载通过 customization 接口接入标准闭环，不是独立 framework。

**Code：**

```python
## 来源：README.md L103-L108
# - [`examples/multi_agent`](examples/multi_agent/README.md):
#     Multi-agent rollout via a custom `--rollout-function-path`.
# - [`examples/search-r1`](examples/search-r1/):
#     Search/RAG-style multi-turn generation via `--custom-generate-function-path`.
# - [`examples/fully_async`](examples/fully_async/README.md):
#     Fully-async rollout ...
# - [`examples/coding_agent_rl`](examples/coding_agent_rl/README.md): ...
```

**Comment：**

- `--rollout-function-path` vs `--custom-generate-function-path` 粒度不同，见 [[04-Arguments-TrainRollout-01-核心概念]]
- fully_async 对应 `train_async.py` 之外的更激进解耦（examples 层）

---

## §4 博文：愿景与 Versatile / Performant / Maintainable

**Explain：** `introducing_slime.md` 给出三大设计支柱。

**Code：**

```python
## 来源：docs/en/blogs/introducing_slime.md L15-L21
# - **Versatile** – fully customizable rollout interface;
#   colocated or decoupled, synchronous or asynchronous, RL or SFT cold start.
# - **Performant** - integrating SGLang for inference and Megatron-LM for training, natively.
# - **Maintainable** - lightweight codebase; smooth transition from Megatron pretraining
#   to SGLang deployment.
```

**Comment：**

- **Versatile** → `*-path` 插件 + Ray 编排
- **Performant** → 不抽象掉 SGLang/Megatron 能力
- **Maintainable** → 主循环在 `train.py` 而非深层 Trainer 继承树

---

## §5 博文：为何暴露 train.py 主循环

**Explain：** 刻意不用 trainer class 包装，方便移动 `ray.get` 做 sync/async 实验。

**Code：**

```python
## 来源：docs/en/blogs/introducing_slime.md L45-L45
# we didn't wrap the code with trainer classes, but simply exposed
# the training loop in entrypoint `train.py`.
```

**Comment：**

- 对比 HuggingFace Trainer：Slime 假设用户是 RL 系统工程师，需要看见完整编排
- 实际循环代码仅 ~100 行，见 [[02-训练主循环-02-源码走读]]

---

## §6 博文：SGLang-native 的三条实现

**Explain：** server-based mode、参数透传、rollout-only debug。

**Code：**

```python
## 来源：docs/en/blogs/introducing_slime.md L57-L59
# - slime internally launches SGLang servers in a **server-based mode**.
# - slime implements **seamless pass-through** for all SGLang parameters
#   (with a `--sglang` prefix) ...
# - slime provides an **SGLang-only debug mode** (`--debug-rollout-only`) ...
```

**Code：**

```python
## 来源：slime/backends/sglang_utils/arguments.py L65-L91（透传机制核心）
    def new_add_argument_wrapper(*name_or_flags, **kwargs):
        new_name_or_flags_list = []
        for item_flag in name_or_flags:
            if isinstance(item_flag, str) and item_flag.startswith("-"):
                original_flag_stem = item_flag.lstrip("-")
                prefixed_item = f"--sglang-{original_flag_stem}"
                new_name_or_flags_list.append(prefixed_item)
            else:
                new_name_or_flags_list.append(item_flag)
        old_add_argument(*new_name_or_flags_list, **final_kwargs)
```

**Comment：**

- `ServerArgs.add_cli_args` 被 monkey-patch 加前缀
- 详述见 [[04-Arguments-TrainRollout-02-源码走读]]

---

## §7 博文：RL 特化与 SGLang 协同

**Explain：** 权重频繁更新、DAPO dynamic sampling 的 `/abort_request` 等与 SGLang 上游共建。

**Code：**

```python
## 来源：docs/en/blogs/introducing_slime.md L77-L80
# **Optimizing weight updates**: RL training involves frequent updates to model weights.
#   - Parameter updates for MoE models under various parallelism strategies
#   - Bucketed parameter update support to reduce overhead
```

**Code：**

```python
## 来源：docs/en/blogs/introducing_slime.md L82-L86
# **`/abort_request` for dynamic sampling**: ... Immediate termination of on-going requests.
#   Reclaiming partially generated content, which enables partial rollouts.
```

**Comment：**

- 权重同步实现见 [[24-WeightSync-Dist-00-MOC]]
- partial rollout 参数 `--partial-rollout` 见[[04-Arguments-TrainRollout-00-MOC]]

---

## §8 博文：轻量四件套 + 扩展 SFT

**Explain：** 框架本体只做四件事；SFT / rejection sampling 是自然延伸。

**Code：**

```python
## 来源：docs/en/blogs/introducing_slime.md L92-L98
# 1. Provides a customizable rollout interface.
# 2. Uses Ray for GPU management and asynchronous execution.
# 3. Integrates SGLang for inference and Megatron for training.
# 4. Provides weight updates between training and inference.
```

**Code：**

```python
## 来源：docs/en/blogs/introducing_slime.md L105-L106
# - **SFT**: Load Megatron and use token prediction loss.
# - **Rejection Sampling**: Use SGLang for filter, followed by Megatron SFT.
```

---

## §9 setup.py：打包与 wheel 标签

**Explain：** 自定义 `bdist_wheel` 生成平台相关 wheel 名。

**Code：**

```python
## 来源：setup.py L13-L28
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

**Code：**

```python
## 来源：setup.py L8-L10
def _fetch_requirements(path):
    with open(path) as fd:
        return [r.strip() for r in fd.readlines() if r.strip() and not r.startswith("#")]
```

**Comment：**

- `slime_plugins*` 一并打包，扩展点见 [[29-Plugins-Examples-00-MOC]]
- `root_is_pure = False` 因含 native / CUDA 相关可选依赖

---

## §10 requirements.txt：闭环相关依赖

**Explain：** 列出与 RL 编排强相关的包。

**Code：**

```python
## 来源：requirements.txt L1-L26（分段摘录）
# accelerate
# datasets
# httpx[http2]
# omegaconf
# ray[default]
# sglang-router>=0.2.3
# safetensors
# transformers
# wandb
# xxhash  # disk delta weight sync
# zstandard
```

**Comment：**

| 包 | 用途 |
|----|------|
| `ray` | PG、Actor、Object Store |
| `sglang-router` | Rollout HTTP 路由 |
| `httpx` | RolloutManager → router 客户端 |
| `xxhash` / `zstandard` | disk delta 权重同步 |
| `datasets` | prompt 数据集加载 |

---

## §11 README：参数三类入口指针

**Code：**

```python
## 来源：README.md L164-L168
# 1.  **Megatron arguments**: slime reads Megatron arguments directly.
# 2.  **SGLang arguments**: ... prefixed with `--sglang-`.
# 3.  **slime-specific arguments**: slime/utils/arguments.py
```

**Comment：** `parse_args()` 三阶段合并，见 [[03-Arguments-Ray-02-源码走读]] §4。

---

## 走读小结

| 顺序 | 材料 | 收获 |
|------|------|------|
| 1 | README 架构段 | 三角角色 |
| 2 | 博文 Vision + Customizability | 设计哲学 |
| 3 | 博文 Performance | native 透传理由 |
| 4 | setup + requirements | 工程边界 |

→ 下一批 [[02-训练主循环-02-源码走读]] 进入可执行入口 `train.py`。
