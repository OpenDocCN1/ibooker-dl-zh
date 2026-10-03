
# 代码示例与评估流程

```python
import time


def eta_progress_message(  #1
    processed,
    total,
    start_time,
    show_eta=False,
    label="Progress",
):
    progress = f"{label}: {processed}/{total}"
    pad_width = len(f"{label}: {total}/{total} | ETA: 00h 00m 00s")
    if not show_eta or processed <= 0:          
        return progress.ljust(pad_width)

    elapsed = time.time() - start_time
    if elapsed <= 0:
        return progress.ljust(pad_width)

    remaining = max(total - processed, 0)

    if processed:
        avg_time = elapsed / processed
        eta_seconds = avg_time * remaining
    else:
        eta_seconds = 0

    eta_seconds = max(int(round(eta_seconds)), 0)
    minutes, rem_seconds = divmod(eta_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        eta = f"{hours}h {minutes:02d}m {rem_seconds:02d}s"
    elif minutes:
        eta = f"{minutes:02d}m {rem_seconds:02d}s"
    else:
        eta = f"{rem_seconds:02d}s"

    message = f"{progress} | ETA: {eta}"
    return message.ljust(pad_width)


def evaluate_math500_stream(
    model,
    tokenizer,
    device,
    math_data,
    out_path=None,
    max_new_tokens=512,
    verbose=False,
):

    if out_path is None:
        dev_name = str(device).replace(":", "-")      #2
        out_path = Path(f"math500-{dev_name}.jsonl")

    num_examples = len(math_data)
    num_correct = 0
    start_time = time.time()

    with open(out_path, "w", encoding="utf-8") as f:   #3
        for i, row in enumerate(math_data, start=1):
            prompt = render_prompt(row["problem"])     #4
            gen_text = generate_text_stream_concat(    #5
                model, tokenizer, prompt, device,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
            )

            extracted = extract_final_candidate(       #6
                gen_text
            )
            is_correct = grade_answer(                 #7
                extracted, row["answer"]
            )
            num_correct += int(is_correct)

            record = {                                 #8
                "index": i,
                "problem": row["problem"],
                "gtruth_answer": row["answer"],
                "generated_text": gen_text,
                "extracted": extracted,
                "correct": bool(is_correct),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            progress_msg = eta_progress_message(
                processed=i,
                total=num_examples,
                start_time=start_time,
                show_eta=True,
                label="MATH-500",
            )
            print(progress_msg, end="\r", flush=True)

            if verbose:                                #9
                print(
                    f"\n\n{'='*50}\n{progress_msg}\n"
                    f"{'='*50}\nExtracted: {extracted}\n"
                    f"Expected:  {row['answer']}\n"
                    f"Correct so far: {num_correct}\n{'-'*50}"
                )

    seconds_elapsed = time.time() - start_time
    acc = num_correct / num_examples if num_examples else 0.0
    print(f"\nAccuracy: {acc*100:.1f}% ({num_correct}/{num_examples})")
    print(f"Total time: {seconds_elapsed/60:.1f} min")
    print(f"Logs written to: {out_path}")
    return num_correct, num_examples, acc
```

**代码注释：**

- `#1` — 辅助函数，用于打印进度，并可选择显示 ETA（预计剩余时间）
- `#2` — 使文件名与 Windows 兼容。
- `#3` — 保存结果以供检查。
- `#4` — 应用提示模板。
- `#5` — 生成响应。
- `#6` — 提取并规范化答案。
- `#7` — 对答案评分。
- `#8` — 保存记录以供检查
- `#9` — 在生成过程中打印响应。

清单 3.15 中的 `evaluate_math500_stream` 函数使用了与清单 3.14 中较小的演示函数相同的主要步骤：对于每个问题，它渲染提示，流式获取模型响应，提取候选答案，并与参考答案进行评分。

除了在包含多个条目的数据集上进行迭代之外，它还增加了一些额外的附加功能。例如，它将生成的响应保存到 JSON 文件中，采用类似 Python 字典的格式，以便记录和更仔细地检查。

现在让我们在子集上运行这个函数：MATH-500 的前 10 个示例。在搭载 M4 芯片的 Mac Mini 上，这大约需要 0.7 分钟。（评估推理模型变体大约需要 7 分钟，因为它生成的响应更长。）

```python
print("Model:", WHICH_MODEL)
print("Device:", device)
num_correct, num_examples, acc = evaluate_math500_stream(
    model, tokenizer, device, 
    math_data=math_data[:10],  #1
    max_new_tokens=2048,
    verbose=False              #2
)
```

- `#1` 仅对前 10 个示例进行评估。
- `#2` 设置为 `true` 可在生成时实时查看响应。

在前面的代码示例中，我们将 `max_new_tokens` 设置为一个较大的值 2,048，因为推理模型变体在设计上倾向于生成更长的响应，我们不希望过早地截断它。这会导致评估时间变长，并且可能看起来生成过程停滞了。你可以选择设置 `verbose=True` 来逐 token 实时查看正在生成的响应。

运行 `evaluate_math500_stream` 的结果如下：

```
Model: base
Device: mps
MATH-500: 10/10 | ETA: 00s
Accuracy: 30.0% (3/10)
Total time: 0.4 min
```

（和往常一样，结果可能因你在 CPU、CUDA 或 MPS 设备上执行代码而有所不同。）

如你所见，该模型的准确率相对较低，仅为 30%。我们可以在文本编辑器中打开 `math500_base-mps.jsonl` 文件，结合生成的响应一起分析结果。例如，所有情况下答案都被成功提取，但这些答案完全是错误的，这表明该模型（目前）还不具备很强的数学问题解决能力。这在意料之中，因为它只是一个基础模型。

## 以编程方式加载 `.jsonl` 文件

`.jsonl` 文件后缀是一种惯例，用于表示每行包含一条数据条目的 JSON 文件。你可以在喜欢的文本编辑器中查看它。或者，你可以使用以下代码在 Python 中加载评估期间生成的 `.jsonl` 文件：

```python
dev_name = str(device).replace(":", "-")
local_path = f"math500_{WHICH_MODEL}-{dev_name}.jsonl"
results = []
with open(local_path, "r") as f:
    for line in f:
        if line.strip():
            results.append(json.loads(line))
```

推理模型变体，你可以通过在清单 3.1 中设置 `WHICH_MODEL = "reasoning"` 来启用，其表现要好得多，在相同的 10 样本子集上实现了 90% 的准确率，在完整的 500 样本数据集上实现了 50.8% 的准确率，如表 3.1 所示。

**表 3.1.** 不同设备上的 MATH-500 任务准确率

| 模式 | 设备 | 准确率 | MATH-500 规模 |
| --- | --- | --- | --- |
| 基础 | CPU | 30% | 10 |
| 基础 | CUDA | 30% | 10 |
| 基础 | MPS | 30% | 10 |
| 推理 | CPU | 90% | 10 |
| 推理 | CUDA | 90% | 10 |
| 推理 | MPS | 80% | 10 |
| 基础 | CUDA | 15.3% | 500 |
| 推理 | CUDA | 50.8% | 500 |

如表 3.1 所示，推理变体凭借更长的响应，准确率大幅提升，但同时也显著增加了计算强度和答案生成时间（在配备 M4 芯片的 Mac Mini 上，10 样本子集从基础模型的 0.4 分钟增加到推理模型的 7 分钟；在 NVIDIA H100 GPU 上，500 样本数据集从 13.3 分钟增加到 185.4 分钟），这凸显了使用推理模型的一个权衡。请注意，这些数据是在 PyTorch 2.8 中获得的，可能因 PyTorch 版本不同而有所差异。

**提示：** 本书的代码仓库包含一个额外脚本，该脚本以批处理模式运行本章代码（[`github.com/rasbt/reasoning-from-scratch/blob/main/ch03/02_math500-verifier-scripts/evaluate_math500_batched.py`](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch03/02_math500-verifier-scripts/evaluate_math500_batched.py)）。这意味着它每次前向传递处理多个示例，以加快评估速度，但需要更多内存。使用批大小为 128 来评估全部 500 个样本，可将基础模型在 H100 GPU 上的运行时间从 13.3 分钟减少到 3.3 分钟。类似地，推理模型的运行时间从 185.4 分钟减少到 14.6 分钟。请注意，这里以 H100 为例；该脚本也兼容其他 GPU。

## 练习 3.2：计算平均响应长度

尝试修改本章代码，让 `evaluate_math500_stream` 函数（清单 3.15）也报告平均响应长度。除了直接修改函数，你也可以从生成的 JSON 报告文件中计算响应长度。

## 练习 3.3：扩展或更改评估数据集

出于计算效率考虑，我们只选择了 10 个示例的子集。你也可以在更大或不同的数据集部分上运行代码，以了解 10 样本子集是否具有代表性。理想情况下，你还可以用自己的数据进行实验。（作为参考，在 H100 上对完整 MATH-500 数据集进行评估，基础模型大约需要 13.3 分钟，推理模型大约需要 185.4 分钟。）

## 练习 3.4：尝试不同的提示模板

模型可能对不同的提示模板很敏感。在清单 3.13 中尝试不同的提示模板，观察它们如何影响结果。此外，虽然 Qwen3 团队建议使用不带额外聊天模板的基础模型，但你可以在 tokenizer 中启用 `apply_chat_template=True` 设置（清单 3.1），并观察它是否提升了基础模型的性能。

至此，我们完成了基于验证的数学任务实现（图 3.11）。我选择数学，是因为它既易于实现，又在推理专项训练中被广泛使用，尤其是可验证奖励的强化学习，我们将在第 6 章中介绍。同样的概念也可以扩展到其他领域，例如代码；但我们没有在这里探讨，因为执行代码需要搭建一个安全的虚拟环境。

![](img/CH03_F11_Raschka2.png)

**图 3.11.** 本书所涵盖主题的模型图。本章实现了一个基于验证的评估流水线。在下一章中，我们将通过更先进的推理技术来提升 LLM 的推理能力。

在继续之前，值得注意，评估还有其他多种形式。在本章中，我们专注于数学问题的基于验证的准确率，因为它是一种流行的方法，而且我们将在第 6 章和第 7 章的强化学习流水线中复用这个验证器。为了让你有更全面的了解，附录 F 介绍了其他常见的评估策略，例如选择题基准、验证器、排行榜和 LLM 作为裁判的设置。如果你希望快速了解这些方法在实践中的工作原理，它还包含动手示例。

现在我们已经有了一个评估框架，下一章将重点介绍通过更先进的推理（文本生成）技术来提升推理能力。

## 总结

- LLM 主要有四种评估方法：选择题、验证器、排行榜和 LLM 裁判。
- 基于验证的评估方法允许自由形式的答案，并使用外部工具来检查正确性。
- 本章重点介绍基于验证的评估，通过构建一个数学验证器来提取、归一化，并使用 SymPy 检查答案。
  - 验证流水线涉及几个核心步骤，从加载 LLM 到在数据集上运行评估。
  - 作为验证流水线的一部分，答案提取使用字符串解析来定位 boxed 内容（对于缺失 box 的情况有回退机制）。
  - 另一个步骤实现归一化，通过去除 LaTeX 并转换数学符号来标准化各种答案格式。
- 最后，流水线使用数学等价性检查（通过 SymPy）来符号化地比较表达式。
- MATH-500 数据集提供了 500 个精选的数学问题用于评估。
- 提示模板对模型性能有显著影响。
- 推理模型比基础模型能实现更高的准确率，但需要更长的运行时间。

# 4 通过推理时扩展改进推理

## 本章涵盖

- 提示 LLM 解释其推理过程以提高答案准确率
- 修改文本生成函数以产生多样化的响应
- 通过对多个响应进行采样来提高推理可靠性

推理性能和答案准确率无需重新训练或修改模型本身即可得到提升。如图 4.1 的概览所示，本章涵盖两种推理时扩展方法。这些方法在推理阶段（即模型生成文本时）运行。你将在本章后文看到，这两种方法都将前几章使用的基础模型准确率提升了一倍以上。

在讨论图 4.1 所示的推理方法之前，我们先来看看什么是推理时扩展。

![](img/CH04_F01_Raschka2.png)

**图 4.1** 本书所涵盖主题的模型。本章聚焦于推理时扩展技术，这类技术无需额外训练即可改进推理（阶段 3）。具体而言，它扩展了文本生成函数，并实现了一种基于投票的方法来提高答案准确率。下一章将介绍另一种推理时扩展方法，即模型迭代式精炼自己的答案。

## 4.1 推理时扩展简介

改进推理主要有两种策略：

- 增加训练算力
