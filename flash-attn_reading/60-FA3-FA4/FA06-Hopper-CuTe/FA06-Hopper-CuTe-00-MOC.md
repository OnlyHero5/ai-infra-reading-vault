---
type: module-moc
module: FA06-Hopper-CuTe
batch: "FA06"
doc_type: moc
title: "FA3/FA4 Hopper 与 CuTe"
tags:
  - flash-attn/batch/fa06
  - flash-attn/module/hopper-cute
  - flash-attn/doc/moc
updated: 2026-07-04
---

# FA3/FA4 Hopper 与 CuTe

> 从 FA2 的静态 CUDA template，走向 Hopper 专门路径与 FA4 CuTeDSL/JIT。

## 阅读顺序

| 顺序 | 文件 | 目标 |
|------|------|------|
| 01 | [[FA06-Hopper-CuTe-01-核心概念]] | 理解 FA3、FA4、CuTeDSL、JIT cache 的位置 |
| 02 | [[FA06-Hopper-CuTe-02-源码走读]] | 走读 `hopper/flash_api.cpp` 与 `flash_attn/cute/interface.py` |
| 03 | [[FA06-Hopper-CuTe-03-数据流与交互]] | 看 arch dispatch、compile key、kernel object、cache 调用 |
| 04 | [[FA06-Hopper-CuTe-04-关键问题]] | 解答 FA3/FA4 与 FA2 的差异、FP8、paged KV、编译缓存 |
| 05 | [[FA06-Hopper-CuTe-05-checkpoint]] | 自测是否能解释新路径价值 |

## 核心源码

| 路径 | 作用 |
|------|------|
| `hopper/flash_api.cpp` | FA3/Hopper C++ API 与 dispatch |
| `flash_attn/cute/interface.py` | FA4 CuTeDSL Python API、arch dispatch、JIT compile cache |
| `flash_attn/cute/__init__.py` | FA4 公开导出 |
| `flash_attn/cute/*` | CuTeDSL kernel object 与配置 |

## 一句话

**Explain：** FA3/FA4 的意义不是“又写了一版 attention”，而是适配 Hopper/Blackwell 级 GPU 的新 memory/copy/MMA 能力，并降低复杂特性组合下手写静态模板的维护成本。

