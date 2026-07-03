---
type: batch-doc
module: 28-Customization
batch: "28"
doc_type: concept
title: "Customization · 核心概念"
tags:
  - slime/batch/28
  - slime/module/customization
  - slime/doc/concept
updated: 2026-07-02
---

# Customization · 核心概念

## 1. 设计动机

Slime 核心保持 **Megatron 训练 + SGLang Rollout** 闭环稳定；业务差异（工具、RM、采样策略、loss）通过 **函数指针** 外置。好处：

- 不改 upstream 即可试验新算法
- 与 verl 式「全写在一个 repo」相比，Slime 原生透传 Megatron/SGLang 参数

---

## 2. load_function：统一加载器

**Explain：** 所有 `--*-path` 最终在运行时 `importlib.import_module` + `getattr`。

**Code：**

```python
## 来源：slime/utils/misc.py L37-L45
def load_function(path):
    module_path, _, attr = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)
```

**Comment：**

- path 格式：`package.module.function`（无 `.py` 后缀）
- 在 `RolloutManager` / `MegatronTrainRayActor` init 时加载并缓存
- 错误 path 在启动期 fail-fast

---

## 3. 17 类接口总览（Rollout 侧）

**Explain：** 摘自官方 customization.md Overview 表。

| # | CLI 参数 | 默认 / 用途 |
|---|----------|-------------|
| 1 | `--rollout-function-path` | `sglang_rollout.generate_rollout` — 整段 rollout |
| 2 | `--custom-generate-function-path` | None — 单 sample 生成 |
| 3 | `--custom-rm-path` | None — 奖励 |
| 4 | `--dynamic-sampling-filter-path` | DAPO 等动态过滤 |
| 5 | `--buffer-filter-path` | buffer 训练前过滤 |
| 6 | `--rollout-sample-filter-path` | 标记 `remove_sample` |
| 7 | `--rollout-all-samples-process-path` | 含 filtered 的全量后处理 |
| 8 | `--rollout-data-postprocess-path` | logprob 算完后处理 |
| 15 | `--data-source-path` | prompt 来源 |
| 16 | `--eval-function-path` | eval 专用 rollout |

**Code（generate 签名）：**

```python
## 来源：docs/en/get_started/customization.md L79-L79
async def custom_generate(args, sample: Sample, sampling_params: dict) -> Sample | list[Sample]
```

---

## 4. 17 类接口总览（Train 侧）

| # | CLI 参数 | 用途 |
|---|----------|------|
| 9 | `--custom-loss-function-path` | 需 `--loss-type custom_loss` |
| 10 | `--custom-tis-function-path` | off-policy TIS 权重 |
| 11 | `--custom-pg-loss-reducer-function-path` | Dr.GRPO 等 reducer |
| 12 | `--custom-reward-post-process-path` | advantage 前 reward 变换 |
| 13 | `--custom-convert-samples-to-train-data-path` | sample → batch dict |
| 14 | `--custom-rollout-log-function-path` / eval 版 | 日志 |
| 17 | `--custom-megatron-*-hook-path` | init / before logprob / before step |

**Code（convert 返回 dict 结构节选）：**

```python
## 来源：docs/en/get_started/customization.md L331-L346
dict: {
    "tokens": list[list[int]],
    "response_lengths": list[int],
    "rewards": list[float],
    "loss_masks": list[list[int]],
    ...
}
```

---

## 5. Agentic 工作流映射

**Explain：** 文档 § Agentic workflows 把 agent 需求映射到接口。

| 需求 | 接口 |
|------|------|
| 多轮 tool / sandbox | `--custom-generate-function-path` |
| 测试/规则奖励 | `--custom-rm-path` |
| 整段编排不够 | `--rollout-function-path` |
| 任务队列 / buffer | `--data-source-path` |
| loss_mask /metadata | `--rollout-data-postprocess-path` |

---

## 6. list[Sample] fan-out 契约

**Explain：** 一个 prompt rollout 可拆多个训练段（subagent、compaction 前后），必须共享 `rollout_id`。

**Code：**

```python
## 来源：docs/en/get_started/customization.md L99-L114
async def custom_generate(args, sample: Sample, sampling_params: dict) -> list[Sample]:
    segments = await run_agent_and_split_segments(args, sample, sampling_params)
    rollout_id = sample.rollout_id if sample.rollout_id is not None else sample.index
    for segment in segments:
        s = copy.copy(sample)
        s.tokens = segment.tokens
        s.loss_mask = segment.loss_mask
        s.rollout_id = rollout_id
        samples.append(s)
    return samples
```

**Comment：** 总 reward 常均分 `reward/K` 避免放大。

---

## 7. parsing.py：generate 的模型输出解析

**Explain：** Adapter 与部分 custom generate 共用；委托 SGLang ReasoningParser + FunctionCallParser。

**Code：**

```python
## 来源：slime/agent/parsing.py L25-L56
def parse_model_output(raw_output, *, tools_schema, tool_parser_name, reasoning_parser_name):
    if reasoning_parser_name:
        from sglang.srt.parser.reasoning_parser import ReasoningParser
        reasoning, body_text = ReasoningParser(...).parse_non_stream(raw_output)
    body_text, tool_uses, ill_formed = parse_tool_uses(body_text, tools_schema, tool_parser_name)
    return ParsedModelOutput(...)
```

---

## 8. harness：sandbox 内跑外部 Agent CLI

**Explain：** `BaseHarness` 定义 install_cli → write_config → launch_and_wait；Claude Code / Codex 为子类。

**Code：**

```python
## 来源：slime/agent/harness/common.py L81-L104
    async def run(self, sb, *, workdir, session_id, adapter_url, time_budget_sec, prompt) -> int:
        await _sandbox.ensure_agent_user(sb, workdir)
        ctx = HarnessContext(workdir=workdir, session_id=session_id, adapter_url=adapter_url)
        await self.write_config(sb, ctx)
        return await self.launch_and_wait(sb, ctx, prompt, time_budget_sec)
```

**Comment：** `adapter_url` 指向 AnthropicAdapter/OpenAIAdapter；`ANTHROPIC_AUTH_TOKEN=session_id` 做 Bearer routing。

---

## 9. ClaudeCodeHarness 环境变量

**Code：**

```python
## 来源：slime/agent/harness/claude_code.py L57-L71
        env = {
            "ANTHROPIC_BASE_URL": ctx.adapter_url,
            "ANTHROPIC_AUTH_TOKEN": ctx.session_id,
            "ANTHROPIC_MODEL": ctx.model_label,
            **self.static_env,
        }
        return await run_agent(sb, workdir=ctx.workdir, start_cmd=cmd, env=env, time_budget_sec=time_budget_sec)
```

**Comment：** npm CLI 通过 `SLIME_AGENT_*` 环境变量指定 tarball 路径。

---

## 10. MoE Routing Replay（文档 §18）

虽标为 §18，仍属 customization 文档延伸：

| 参数 | 作用 |
|------|------|
| `--use-routing-replay` | 训练 forward-backward 路由一致 |
| `--use-rollout-routing-replay` | R3：replay rollout 路由 |

见 [[23-CP-RoutingReplay-00-MOC]]。
