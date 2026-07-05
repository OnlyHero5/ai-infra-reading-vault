---
type: index-doc
title: "Slime-90-总结复盘-02-可观测与CI"
doc_type: concept
tags:
  - slime/index-layer
  - slime/batch/30
  - slime/doc/concept
updated: 2026-07-02
---

# 07 · 可观测与 CI

> trace / profile / CI / fault-tolerance · [[Slime-00-导读与总览-00-MOC]] 额外源码

---

## 1 · Trace（Sample 级 span）

**Explain：** `trace_utils` 为每个 Sample 附加 span 树；与 SGLang PD 时序字段联动，可用 `tools/trace_timeline_viewer.py` 可视化。

**Code：**

```python
## 来源：slime/utils/trace_utils.py L16-L24
TRACE_VERSION = 1
TRACE_CHILDREN_KEY = "_trace_children"
SGLANG_TRACE_META_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "queue_time",
    "e2e_latency",
    "decode_throughput",
)
```

**使用方式（官方文档摘录）：**

```bash
## 来源：docs/en/developer_guide/trace.md
python train.py \
    ... \
    --save-debug-rollout-data /path/to/debug/rollout_{rollout_id}.pt

python tools/trace_timeline_viewer.py /path/to/debug/rollout_0.pt
```

**Comment：**

- `--save-debug-rollout-data` 保存含 trace 的 rollout dump；`--load-debug-rollout-data` 可 replay 跳过 generate。
- 定制 rollout/RM 代码应使用 `trace_span` / `trace_event` / `trace_function` 包装关键段。

→ [[12-SGLang-Rollout-03-数据流与交互]]

---

## 2 · Profile（TrainProfiler）

**Explain：** PyTorch profiler 与 memory snapshot 按 `profile_target` 选择性启用。

**Code：**

```python
## 来源：slime/utils/profile_utils.py L13-L32
class TrainProfiler:
    def __init__(self, args):
        self.args = args
        self._torch_profiler_overall = None
        self._memory_profiler_overall = None

        if args.use_pytorch_profiler and ("train_overall" in args.profile_target):
            self._torch_profiler_overall = _create_torch_profiler(args, name="train_overall")

        if args.record_memory_history and ("train_overall" in args.profile_target):
            self._memory_profiler_overall = _BaseMemoryProfiler.create(args)
            self._memory_profiler_overall.start()

    def step(self, rollout_id: int):
        if self._torch_profiler_overall is not None:
            self._torch_profiler_overall.step()
```

**Comment：** `profile_target` 可细分为 `train_actor`、`train_log_probs` 等子循环；见 `docs/en/developer_guide/profiling.md`。

---

## 3 · Megatron Server（alternate 入口）

**Explain：** 独立 Megatron 推理/服务入口，与 RL 主循环分离；共享 `parse_args` 但追加 server 参数。

**Code：**

```python
## 来源：slime/backends/megatron_utils/server/megatron_server.py L762-L770
def main():
    from slime.utils.arguments import parse_args

    args = parse_args(add_custom_arguments=add_megatron_server_arguments)
    launch(args)

if __name__ == "__main__":
    main()
```

**Comment：** 770 行热点文件；用于非 SGLang rollout 场景或调试 Megatron forward-only。

---

## 4 · CI 架构

**Explain：** 两层 CI——CPU 常驻 correctness + label-gated GPU e2e。

**Code（文档摘录）：**

```markdown
## 来源：docs/en/developer_guide/ci.md L3-L8
slime CI has two layers:
1. Always-on CPU correctness tests on every PR/push to main.
2. Label-gated GPU end-to-end tests on self-hosted GPU runners.
```

| 层级 | Runner | 典型测试 |
|------|--------|----------|
| CPU | `ubuntu-latest` | arguments、plugin_contracts、agent adapter |
| GPU | self-hosted Docker | PPO、async、PD、checkpoint |

**GPU 锁：**

```python
## 来源：docs/en/developer_guide/ci.md L28-L31
# Acquire GPUs with tests/ci/gpu_lock_exec.py --count <num_gpus>
# Execute: python tests/<test_file>.py
```

**Comment：** CPU-only 测试应声明 `NUM_GPUS = 0`；缺失时 CI 默认 8 GPU。

---

## 5 · Fault Tolerance

**Explain：** `--use-fault-tolerance` 时 rollout engine 可被 kill/recover；`update_weights` 前调用 `recover_updatable_engines`。

**Code：**

```python
## 来源：slime/ray/rollout.py L550-L551
        if self.args.ci_test and self.args.use_fault_tolerance and rollout_id >= 2:
            self._try_ci_fault_injection()
```

```python
## 来源：slime/backends/megatron_utils/actor.py L587-L590
        if self.args.use_fault_tolerance:
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.recover_updatable_engines.remote())
            dist.barrier(group=get_gloo_group())
```

→ `slime/docs/en/advanced/fault-tolerance.md`（官方）· [[16-External-Engines-01-核心概念]]

---

## 6 · Debug 模式

| 参数 | 效果 |
|------|------|
| `--debug-rollout-only` | 只跑 generate，跳过 train/update |
| `--debug-train-only` | 跳过 generate，用 `--load-debug-rollout-data` |
| `--save-debug-rollout-data` | 保存 rollout dump + trace |
| `--check-weight-update-equal` | bootstrap 后比对 train/rollout 权重 |

→ [[02-训练主循环-04-关键问题]]

---

## 7 · 验证建议

| 场景 | 测试 |
|------|------|
| PG 分配 | `tests/test_placement_group.py` |
| PPO e2e | `tests/test_qwen3_4B_ppo.py` |
| 异步 | `tests/test_qwen2.5_0.5B_async_short.py` |
| plugin 契约 | `tests/plugin_contracts/` |
| checkpoint | `tests/utils/test_hf_checkpoint_saver.py` |

---

## 导航

- [[Slime-90-总结复盘-01-复杂度热点]]
- [[Slime-90-总结复盘-04-checkpoint]]
