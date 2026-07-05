---
type: batch-doc
module: FA02-Online-Softmax
batch: "FA02"
doc_type: checkpoint
title: "Online Softmax · checkpoint"
tags:
  - flash-attn/batch/fa02
  - flash-attn/module/online-softmax
  - flash-attn/doc/checkpoint
updated: 2026-07-05
---

# Online Softmax · checkpoint

## 读者自测

- [ ] 能解释普通 block-wise softmax 为什么会错
- [ ] 能说出 online softmax 的三个核心状态
- [ ] 能解释 `softmax_lse` 为什么足以支持 backward 重算
- [ ] 能说明 FlashAttention 为什么仍是 exact attention
- [ ] 能在 `softmax.h` 找到 `Softmax` 结构
- [ ] 能在 `flash_fwd_kernel.h` 找到 `softmax_rescale_o`
- [ ] 能解释 `scores_max_prev`、`row_max`、`scores_scale` 三者的关系
- [ ] 能说明为什么 `row_sum` 与 `acc_o` 必须同时 rescale
- [ ] 能指出 Python autograd 保存 `softmax_lse` 的位置
- [ ] 能说明 `Return_softmax` 分支和常规 backward 保存路径的区别
