---
type: module-moc
module: FA02-Online-Softmax
batch: "FA02"
doc_type: moc
title: "Online Softmax"
tags:
  - flash-attn/batch/fa02
  - flash-attn/module/online-softmax
  - flash-attn/doc/moc
updated: 2026-07-05
---

# Online Softmax

> 分块计算仍然精确的关键：每行维护最大值、归一化分母与输出累积。

## 阅读顺序

| 顺序 | 文件 | 目标 |
|------|------|------|
| 01 | [[FA02-Online-Softmax-01-核心概念]] | 理解 online softmax 状态更新 |
| 02 | [[FA02-Online-Softmax-02-源码走读]] | 阅读 `softmax.h` 与 forward 主循环 |
| 03 | [[FA02-Online-Softmax-03-数据流与交互]] | 看 `m_i/l_i/o_i/LSE` 的数据流 |
| 04 | [[FA02-Online-Softmax-04-关键问题]] | 解答精度、重算、dropout 相关问题 |
| 05 | [[FA02-Online-Softmax-05-checkpoint]] | 自测是否能推导核心状态 |

## 核心源码

| 文件 | 作用 |
|------|------|
| `csrc/flash_attn/src/softmax.h` | online softmax 的核心工具 |
| `csrc/flash_attn/src/flash_fwd_kernel.h` | 调用 `softmax_rescale_o` 并更新输出累积 |
| `flash_attn/flash_attn_interface.py` | 保存 `softmax_lse` 给 backward |

## 本专题验收标准

- 能解释 block-wise softmax 为什么必须维护跨 block 的全局状态。
- 能从 `softmax.h` 指出 `row_max`、`row_sum` 和 `acc_o` 的更新位置。
- 能说明 `scores_scale = exp2((old_max - new_max) * scale)` 为什么同时作用于 `row_sum` 与 `acc_o`。
- 能串起 forward 保存 `softmax_lse` 与 backward 重算 probability 的关系。

## 一句话

**Explain：** 如果只分块算 `softmax(QK^T)`，每个 block 的局部归一化会错。Online softmax 让每一行在看到新 block 时修正历史归一化，从而得到与全量 softmax 一致的结果。
