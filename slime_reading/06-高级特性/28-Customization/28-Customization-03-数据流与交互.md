---
type: batch-doc
module: 28-Customization
batch: "28"
doc_type: dataflow
title: "Customization · 数据流与交互"
tags:
  - slime/batch/28
  - slime/module/customization
  - slime/doc/dataflow
updated: 2026-07-02
---

# Customization · 数据流与交互

## 1. Agentic RL 接入决策树

```mermaid
flowchart TD
    Q1{"只需改单 sample<br/>生成逻辑?"}
    Q2{"需自定义 RM?"}
    Q3{"需改 rollout 调度<br/>或 async 队列?"}

    Q1 -->|是| G["--custom-generate-function-path"]
    Q1 -->|否| Q3
    Q2 -->|是| R["--custom-rm-path"]
    Q2 -->|否| G
    G --> Q2
    Q3 -->|是| RF["--rollout-function-path"]
    Q3 -->|否| G

    G --> AD{"用现成 Agent SDK?"}
    AD -->|Anthropic/OpenAI| A["Adapter + harness"]
    AD -->|手写 loop| H["如 search-r1"]
```

---

## 2. 默认 Rollout 链路上的 hook 点

```mermaid
flowchart LR
    DS["data_source.get_samples"]
    GR["generate_rollout"]
    CG["custom_generate"]
    RM["custom_rm"]
    DF["dynamic_filter"]
    BF["buffer_filter"]
    PP["rollout_data_postprocess"]
    CV["convert_samples_to_train_data"]

    DS --> GR --> CG --> RM
    GR --> DF
    GR --> BF
    GR --> PP --> CV
```

**Explain：** `--custom-generate-function-path` 只替换 CG 节点；其余 hook 可选叠加。

---

## 3. Adapter + Harness 组合数据流

```mermaid
sequenceDiagram
    participant CF as custom_generate
    participant H as ClaudeCodeHarness
    participant SB as Sandbox
    participant AD as AnthropicAdapter
    participant TM as TrajectoryManager

    CF->>AD: open_session + start aiohttp
    CF->>H: run(sb, adapter_url, session_id)
    H->>SB: claude CLI → adapter /v1/messages
    SB->>AD: HTTP
    AD->>TM: record_turn × N
    CF->>AD: finish_session
    AD->>CF: list[Sample]
```

**Code（harness env 指向 adapter）：**

```python
## 来源：slime/agent/harness/claude_code.py L62-L66
        env = {
            "ANTHROPIC_BASE_URL": ctx.adapter_url,
            "ANTHROPIC_AUTH_TOKEN": ctx.session_id,
            ...
        }
```

---

## 4. 训练侧 hook 时序

| 时机 | 参数 | 典型用途 |
|------|------|----------|
| Megatron init 后 | `--custom-megatron-init-path` | 注册 buffer、自定义 optimizer |
| logprob 前 | `--custom-megatron-before-log-prob-hook-path` | MoE routing replay 准备 |
| train step 前 | `--custom-megatron-before-train-step-hook-path` | 冻结、日志 |
| loss 计算 | `--custom-loss-function-path` | 新 RL 目标 |
| advantage 前 | `--custom-reward-post-process-path` | reward shaping |

---

## 5. 与 [[27-Agent-Trajectory-00-MOC]] 的分工

| 组件 | 职责 |
|------|------|
| TrajectoryManager | token 线性化、drift、多 Sample |
| parsing.py | 原始 text → tool_uses |
| customization 文档 | **何时** 选哪个 CLI |
| harness | **如何** 在 sandbox 跑 CLI agent |

---

## 6. 日志 hook 返回值语义

**Code：**

```python
## 来源：docs/en/get_started/customization.md L364-L364
# Return: `True` to skip default logging, `False` to continue with default logging.
```

**Comment：** 自定义 wandb/tensorboard 时常返回 True 完全接管。

---

## 7. eval 与 train rollout 分离

**Explain：** `--eval-function-path` 默认等于 `--rollout-function-path`；eval 时可换更保守 sampling 或禁用 tool。

**Code：**

```python
## 来源：docs/en/get_started/customization.md L408-L409
**Default**: Same as `--rollout-function-path`
```

---

## 8. 环境变量覆盖契约测试

**Code：**

```python
## 来源：docs/en/get_started/customization.md L487-L491
python tests/plugin_contracts/test_plugin_rollout_contracts.py \
  --rollout-function-path my_project.custom_rollout.generate_rollout
```

**Comment：** 本地验证自定义 path 在 CI 断言下仍满足签名与返回结构。
