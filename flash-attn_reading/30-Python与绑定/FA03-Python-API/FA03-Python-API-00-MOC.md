---
type: module-moc
module: FA03-Python-API
batch: "FA03"
doc_type: moc
title: "Python API 与绑定"
tags:
  - flash-attn/batch/fa03
  - flash-attn/module/python-api
  - flash-attn/doc/moc
updated: 2026-07-04
---

# Python API 与绑定

> 从 PyTorch 用户入口走到 `flash_attn_2_cuda`，理解上层框架真正依赖的边界。

## 阅读顺序

| 顺序 | 文件 | 目标 |
|------|------|------|
| 01 | [[FA03-Python-API-01-核心概念]] | 区分普通、packed、varlen、KV cache API |
| 02 | [[FA03-Python-API-02-源码走读]] | 从 `flash_attn_func` 走到 custom op |
| 03 | [[FA03-Python-API-03-数据流与交互]] | 串起 Python、pybind、C++ 参数与 padding |
| 04 | [[FA03-Python-API-04-关键问题]] | 解答 autograd、fake tensor、varlen、返回 attention 概率 |
| 05 | [[FA03-Python-API-05-checkpoint]] | 自测是否能解释 API 到内核边界 |

## 核心源码

| 文件 | 作用 |
|------|------|
| `flash_attn/flash_attn_interface.py` | 公开 API、custom op、autograd wrapper |
| `flash_attn/bert_padding.py` | `unpad_input`、`cu_seqlens`、padding 还原 |
| `flash_attn/modules/mha.py` | 模型模块中的使用方式 |
| `setup.py` | 编译 `flash_attn_2_cuda` 扩展 |
| `csrc/flash_attn/flash_api.cpp` | pybind 暴露 `fwd`、`varlen_fwd`、`bwd`、`fwd_kvcache` |

## 一句话

**Explain：** Python API 不只是薄封装，它把训练/推理的张量布局、mask 参数、varlen metadata 和 autograd 语义统一成 CUDA extension 能消费的调用。

