---
type: batch-doc
module: 13-RM-FilterHub
batch: "13"
doc_type: faq
title: "RM-FilterHub · 关键问题"
tags:
  - slime/batch/13
  - slime/module/rm-filter-hub
  - slime/doc/faq
updated: 2026-07-02
---

# RM-FilterHub · 关键问题

---

## Q1：`math` 与 `dapo` 该选哪个？

| 维度 | `--rm-type math` | `--rm-type dapo` |
|------|-----------------|------------------|
| 实现文件 | `math_utils.py` | `math_dapo_utils.py` |
| 错题 reward | `0` | `-1.0`（dict 的 `score`） |
| 答案提取 | 最后 `\boxed{}` / `\fbox` | 默认 `Answer:` 行；可选 strict box |
| 长 response | 全文 extract | **仅最后 300 字符**参与 verify |
| 返回值 | 标量 int | dict，需 `--reward-key score` |

**Explain：** DAPO 训练脚本通常配对 `check_reward_nonzero_std`，因为 -1/1 混合更易产生组内方差；纯 GSM8K baseline 常用 `math` + 0/1。

**Code（dapo 错题信号）：**

```python
## 来源：slime/tests/test_rm_math_dapo.py L213-L218
def test_compute_score_incorrect_returns_minus_one():
    out = compute_score(r"\boxed{43}", "42", strict_box_verify=True)
    assert out["score"] == -1.0
    assert out["acc"] is False
```

---

## Q2：`math_utils` 与 `math_dapo_utils` 为何不能合并？

测试文件明确锁定三处差异，合并会导致 **静默行为漂移**：

**Code（测试注释摘要）：**

```python
## 来源：slime/tests/test_rm_math_dapo.py L1-L14
#  - remove_boxed: dapo raises AssertionError vs math_utils returns None
#  - normalize_final_answer: separate pipeline
#  - compute_score: only last 300 chars
```

**易错 vs 正确：**

```python
# ❌ 错误：假设 dapo 与 math 对无 box 文本行为一致
async_rm(..., rm_type="dapo")   # 可能走 Minerva Answer: 路径
async_rm(..., rm_type="math")   # 无 \boxed{} → 直接 0 分

# ✅ 正确：GSM8K boxed 输出用 math；DAPO 配方用 dapo + reward-key
--rm-type math
# 或
--rm-type dapo --reward-key score
```

---

## Q3：如何挂载 custom RM？

**单 sample 签名（默认）：**

```python
# 插件示例（新文件，非 slime 内嵌）
async def my_rm(args, sample, **kwargs) -> float:
    return 1.0 if "correct" in sample.response else 0.0
```

**Batch 签名（`--group-rm` 或 fan-out 批量路径）：**

```python
async def my_batched_rm(args, samples: list, **kwargs) -> list[float]:
    return [await my_rm(args, s) for s in samples]
```

**CLI：**

```bash
--custom-rm-path my_pkg.my_rm.my_rm
# group 模式需 batch 签名：
--custom-rm-path my_pkg.my_rm.my_batched_rm --group-rm
```

**Comment：** `async_rm` 内 custom 路径 **跳过** rm_type 分支；eval 时 `sample.custom_rm_path` 再覆盖 global path。

---

## Q4：`--group-rm` 何时需要？

**Explain：** 当 RM 需要 **同一 prompt 的多条 response 联合上下文**（批量化远程 RM、listwise 排序）时启用。默认 GRPO 独立打分 **不需要** group_rm。

**易错：**

```python
# ❌ custom_rm 只有 (args, sample) 签名却开启 --group-rm
# batched_async_rm 会以 (args, samples) 调用 → TypeError

# ✅ 实现 batch 签名，或关闭 group_rm
```

eval 路径 **禁止** group_rm（assert）。

---

## Q5：`check_reward_nonzero_std` 过滤掉所有组怎么办？

**现象：** 模型过强或过弱时，每组 n 条 response reward 完全相同，std≈0，filter 持续 drop，rollout 环可能极慢或看似「卡住」。

**缓解：**

- 提高 `--over-sampling-batch-size` 增加并行 prompt 吞吐
- 调整采样温度使 response 多样性增加
- 换用 curriculum / 自定义 filter（如保留部分 zero-std 组）
- 调试时可 **临时移除** `--dynamic-sampling-filter-path`

**Code（filter 判定）：**

```python
## 来源：slime/slime/rollout/filter_hub/dynamic_sampling_filters.py L10-L11
    rewards = [sample.get_reward_value(args) for sample in samples]
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-6
```

---

## Q6：`dapo` RM 未设 `--reward-key` 会怎样？

**Explain：** `get_reward_value` 在 `reward_key=None` 时直接返回 dict；`torch.tensor(rewards)` 在 filter 中可能失败。

**Code：**

```python
## 来源：slime/slime/utils/types.py L246-L247
    def get_reward_value(self, args) -> float:
        return self.reward if not args.reward_key else self.reward[args.reward_key]
```

**✅ 正确配置：**

```bash
--rm-type dapo --reward-key score
```

---

## Q7：`boxed_math` 等前缀怎么用？

**Explain：** `async_rm` 支持 `rm_type` 以 `boxed_` 开头：先从 response extract boxed 内容，再按后缀类型路由。

**Code：**

```python
## 来源：slime/slime/rollout/rm_hub/__init__.py L69-L71
    if rm_type.startswith("boxed_"):
        response = extract_boxed_answer(response) or ""
        rm_type = rm_type[len("boxed_") :]
```

**示例：** metadata 或 CLI 设 `rm_type=boxed_math` → 等价于先 extract 再 `grade_answer_verl(extracted, label)`。

---

## Q8：`remote_rm` 超时与重试策略？

- 共享 `aiohttp.ClientSession`，`total=120s` timeout
- 最多 10 次重试，backoff `min(2**attempt, 30) + jitter`
- 最终失败 **raise**，该 sample 的 generate task 失败（不会静默给 0 分）

**Code：**

```python
## 来源：slime/slime/rollout/rm_hub/__init__.py L29-L30
        timeout = aiohttp.ClientTimeout(total=120)
```

---

## Q9：与 Slime skill `add-dynamic-filter` 的关系

仓库内 `.claude/skills/add-dynamic-filter/SKILL.md` 描述如何新增 filter 模块并挂 CLI。核心契约与本专题一致：返回 `DynamicFilterOutput`，注册到 `filter_hub/dynamic_sampling_filters.py` 或独立模块。

---

## Q10：推荐验证命令

```bash
# DAPO scorer 单元测试（本专题测试锚点）
pytest slime/tests/test_rm_math_dapo.py -q

# 插件路径与签名契约
pytest slime/tests/plugin_contracts/test_plugin_path_loading_contracts.py -k "rm or dynamic_filter" -q

# 端到端短训（含 math RM + nonzero_std filter）
pytest slime/tests/test_qwen3.5_0.8B_gsm8k_short.py -q
```

**Comment：** `test_rm_math_dapo.py` 顶部 `NUM_GPUS = 0`，纯 CPU，适合 CI 快速回归。

---

## 对比：Slime vs 纯 SGLang serving RM

| | Slime RM Hub | SGLang HTTP server |
|--|-------------|-------------------|
| 执行位置 | Rollout 进程 inline | 独立服务 |
| 典型延迟 | rule-based μs 级 | 网络 + GPU 推理 |
| 扩展 | `--custom-rm-path` | 自建 API + `remote_rm` |

Slime **默认**面向 RL 后训练的 rule-based / 轻量 RM；大模型 RM 通过 `remote_rm` 解耦。
