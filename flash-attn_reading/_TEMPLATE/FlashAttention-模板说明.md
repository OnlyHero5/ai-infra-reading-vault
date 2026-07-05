---
type: template
title: "FlashAttention 专题文档模板"
tags:
  - flash-attn/template
updated: 2026-07-05
---

# FlashAttention 专题文档模板

> 复制本结构到各专题文件夹。**读者只读 `flash-attn_reading/`，不读 `flash-attn/`**，所以关键源码必须内嵌在文档中。

## 文件说明

| 文件 | 用途 | doc_type tag |
|------|------|--------------|
| `{模块名}-00-MOC.md` | 专题概述、目标、源码范围、验收标准 | `flash-attn/doc/moc` |
| `{模块名}-01-核心概念.md` | 原理、术语、设计动机、AI infra 位置 | `flash-attn/doc/concept` |
| `{模块名}-02-源码走读.md` | 按调用顺序的代码精读 | `flash-attn/doc/walkthrough` |
| `{模块名}-03-数据流与交互.md` | HBM/SRAM/register、Python/C++/CUDA 边界 | `flash-attn/doc/dataflow` |
| `{模块名}-04-关键问题.md` | FAQ、易错点、性能/精度/平台差异 | `flash-attn/doc/faq` |
| `{模块名}-05-checkpoint.md` | 读者自测清单 | `flash-attn/doc/checkpoint` |

**示例：** `FA01-Attention-IO-00-MOC.md`、`FA01-Attention-IO-01-核心概念.md`。

**全专题合计：至少 15 段内嵌源码，至少 200 行源码摘录。** `源码走读` 是主文档，原则上不少于 8 段源码；`核心概念`、`数据流与交互`、`关键问题` 也要各有源码证据，避免只写原理提纲。

## frontmatter 模板

```yaml
---
type: batch-doc
module: FA01-Attention-IO
batch: "FA01"
doc_type: concept
title: "Attention IO · 核心概念"
tags:
  - flash-attn/batch/fa01
  - flash-attn/module/attention-io
  - flash-attn/doc/concept
updated: 2026-07-04
---
```

模块间链接用双链：`[[FA02-Online-Softmax-01-核心概念]]`。

## 写作前置：源码阅读门槛

修改任何读者向正文前，必须先完成两件事：

1. 用 `node 90_meta/audit_source_evidence.mjs` 或等价方式确认本文所有 `来源：...` 能定位到 `flash-attn/flash-attention` upstream。
2. 完整阅读本文引用到的 upstream 源码文件，再写解释；不能只看已有笔记、不能只看摘录片段、不能凭经验补设计动机。

若新增代码摘录，必须同步标注 `来源：路径 Lx-Ly`。若解释的是未摘录的大段上下文，应在段落中写明“源码依据来自同文件的模板分派 / kernel 参数构造 / launch 条件 / 测试约束”，避免无证据的设计判断。

---

## 写作格式：扩展 ETC

````markdown
### 1.1 Python API 进入 CUDA 扩展

**Explain：** `flash_attn_func` 只是用户入口，真正进入扩展的是 `_wrapped_flash_attn_forward`。

**Code：**

```python
# 来源：flash_attn/flash_attn_interface.py L98-L113
out, softmax_lse, S_dmask, rng_state = flash_attn_gpu.fwd(
    q,
    k,
    v,
    None,
    alibi_slopes,
    dropout_p,
    softmax_scale,
    causal,
    window_size_left,
    window_size_right,
    softcap,
    return_softmax,
    None,
)
```

**代码逻辑：**
- wrapper 先把 `q/k/v` 规整成 CUDA extension 可以假设的 layout。
- 参数没有在 Python 层拆分计算，而是原样下沉到 `flash_attn_gpu.fwd`。
- 返回的 `out/softmax_lse/S_dmask/rng_state` 分别对应结果、反向重算状态、测试概率输出和 dropout 随机状态。

**为什么这样写：** Python 层保留 API、autograd、fake tensor、编译器接入职责；高频主计算留给 C++/CUDA。这样既能接入 PyTorch 生态，又不让 Python 进入 tile attention 的性能关键路径。

**Comment：**
- `flash_attn_gpu` 在 CUDA 环境下通常是 `flash_attn_2_cuda`
- `softmax_lse` 是 backward 重算 attention scores 的关键中间量
````

### 最低解释要求

每段内嵌源码不能只配一句总结。至少回答三件事：

1. **代码逻辑：** 按执行顺序说明条件分支、状态读写、张量形状或指针如何变化。
2. **为什么这样写：** 解释工程动机，例如减少 HBM 访问、把运行时条件变成编译期模板、服务 autograd 重算、兼容 serving cache manager。
3. **不变量与失败模式：** 说明 dtype、stride、head_dim、block size、causal/window、dropout、paged KV 等约束中哪些不能破坏，破坏后是精度错、非法访存、性能退化还是 dispatch 走错。
4. **读者应抓住什么：** 在 `Comment` 中指出这段代码和专题主线的关系，避免停留在“这段代码做了什么”的表层。

`Explain` 负责先给结论，`Code` 提供证据，`代码逻辑` 负责拆执行过程，`为什么这样写` 负责讲设计取舍，`不变量与失败模式` 负责说明约束边界，`Comment` 负责回到阅读主线。

## Mermaid 规则

- 换行使用 `<br/>`，禁止在 Mermaid 标签中写 `\n`
- 节点 ID 用英文短名，中文放在引号标签内
- 复杂图拆成多图，单图节点建议少于 20

## 源码注释规则

- Python 代码块内用 `# 来源：...`
- C++ / CUDA 代码块内用 `// 来源：...`
- Markdown 正文可写 `来源：...`，但不要把 `## 来源` 放进 C++ 代码块
- 行号必须对应 `flash-attn/flash-attention` 当前基线 `002cce0`

## 禁止事项

1. 不写泛化文件名，如 `README.md`、`01-核心概念.md`
2. 不只写“详见某源码”，必须内嵌关键代码片段
3. 不在读者正文中使用维护派工、内部编号、临时 TODO 话术
4. 不把 generated kernel 文件逐个机械列成阅读任务
