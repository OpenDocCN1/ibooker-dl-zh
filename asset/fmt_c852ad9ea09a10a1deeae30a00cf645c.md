
数据集条目以 Python 字典的形式存储，包含按键字母顺序排序的键和值。以下是相关的键：

- `problem`：供大语言模型解答的数学问题或题目
- `answer`：我们用来与大语言模型答案进行对比的正确（真实）答案
- `solution`：对问题的逐步详细解答（本章未使用，但对训练或分析很有用）

现在我们已经有了预训练的大语言模型、评估函数和可用的基准数据集，接下来就可以评估模型了。

## 3.9 评估模型

在本节中，我们将把图 3.10 中步骤 2–6 的大语言模型文本生成和评估工具付诸实践，并将它们应用于上一节加载的 MATH-500 数据集（步骤 8）。这整个流程在后续章节中将非常重要，因为当我们比较提示方法和基于训练的改进时，这将是我们衡量进展的主要方式。

![](img/CH03_F10_Raschka2.png)

### 图 3.10 MATH-500 数据集的完整评估流程。加载数据集（步骤 7）后，所有问题都会系统地应用步骤 2–6，以得到最终的模型评估结果（步骤 8）。

你可能还记得，在第 3.4 节中，我们的答案检查流程期望模型以方框形式返回答案，这是评估推理模型在数学问题上的表现时常用的约定。为了增加模型遵循此格式的可能性，我们可以按如下方式格式化提示词。

### 清单 3.13 用于渲染数学评估提示模板的函数

```
def render_prompt(prompt):
    template = (
        "You are a helpful math assistant.\n"
        "Answer the question and write the final result on a new line as:\n"
        "\\boxed{ANSWER}\n\n"
        f"Question:\n{prompt}\n\nAnswer:"
    )
    return template
```

现在让我们将这个提示模板应用于第 3.2 节中介绍并在此再次展示的示例提示词：

```
prompt = (
    r"If $a+b=3$ and $ab=\tfrac{13}{6}$, "
    r"what is the value of $a²+b²$?"
)
prompt_fmt = render_prompt(prompt)
print(prompt_fmt)
```

格式化后的提示词如下：

```
You are a helpful math assistant.
Answer the question and write the final result on a new line as:
\boxed{ANSWER}

Question:
If $a+b=3$ and $ab=\tfrac{13}{6}$, what is the value of $a²+b²$?

Answer:
```

接下来，在构造模型评估函数之前，我们将把提示词传递给我们在清单 3.3 中定义的文本生成包装函数，以回顾文本生成过程：

```
generated_text = generate_text_stream_concat(
    model, tokenizer, prompt_fmt, device,
    max_new_tokens=2048,
    verbose=True
)
```

使用这个示例提示词，模型给出了一个相对简短的答案：`"\boxed{10}"`。（请注意，生成的响应可能因你在 CPU、CUDA 或 MPS 设备上运行代码而有所不同。）

虽然简短的回答可以通过减少 token 数量来加快生成速度，但该回答是不正确的。相比之下，在第 3.3 节中，在没有提示模板的情况下，模型生成了更长的回答，并得出了正确答案 14/3。

在得出任何结论之前，提示模板是否适合给定的模型和任务，需要在更大的示例集（例如 MATH-500 数据集）上加以确定。

### 提示模板选择

这里使用清单 3.13 中的提示模板，是为了演示如何实现一个可自动检查答案正确性的模型评估流程。所选模板鼓励短输出，这能让你在第一次阅读时高效地学完本章。之后，我建议你使用其他设置重新审视本章的示例，以优化这里作为参考模型的官方 Qwen3 推理模型变体的准确性。

事实证明，不使用提示模板会使基础模型的性能提升 50%，但会降低推理模型的准确率 40%。

此外，我们可以尝试其他提示模板。例如，常用于 MATH-500 基准测试的标准提示词是清单 3.13 的一个变体，它将“Question:”换成了“Problem:”。这一看似微小的改变将基础模型的准确率提高了约 20%，这可能是因为它更好地匹配了模型记忆中的训练数据（假设 MATH-500 测试集包含在训练语料库中）。虽然基础模型受益于此更改，但推理模型变体的准确率下降了 30%。

前面我提到，答案提取是一项简单、机械的任务，可以用确定性代码解决，而不是使用另一个 LLM 来提取答案。根据不同提示模板下准确率的变化，我们的提取方法似乎可能并不可靠。但情况未必如此。

此外，LLM 生成格式错误的答案也未必是问题所在。它可能只是答错了，而换用另一个 LLM 来提取答案并不能解决这个问题。较小的基础模型往往对提示措辞非常敏感。在下一章中，你会看到，一旦我们引入更多提示变体，这一点会更加明显。

在实现最终的模型评估函数之前，让我们先用下面的演示函数，在一个较小的示例上端到端地测试模型评估流水线。

### 清单 3.14 用于运行评估流水线的演示函数

```
def mini_eval_demo(model, tokenizer, device):
    ex = {  #1
        "problem": "Compute 1/2 + 1/6.",
        "answer": "2/3"
    }
    prompt = render_prompt(ex["problem"])     #2
    gen_text = generate_text_stream_concat(   #3
        model, tokenizer, prompt, device,      #3
        max_new_tokens=64,                     #3
    )                                          #3
    pred_answer = extract_final_candidate(gen_text)  #4
    is_correct = grade_answer(                       #5
        pred_answer, ex["answer"]                     #5
    )                                                 #5

    print(f"Device: {device}")
    print(f"Prediction: {pred_answer}")
    print(f"Ground truth: {ex['answer']}")
    print(f"Correct: {is_correct}")
```

代码注释：

- `#1` — 使用 `"problem"` 和 `"answer"` 字段测试示例。
- `#2` — 1. 应用提示模板
- `#3` — 2. 生成响应
- `#4` — 3. 提取并规范化答案
- `#5` — 4. 对答案评分

`mini_eval_demo` 函数整合了本章目前涵盖的所有方面：

1. 应用提示模板
2. 将格式化后的提示提供给 LLM 以生成回答
3. 提取并规范化答案
4. 对答案评分

`mini_eval_demo` 函数本质上是将评估组件连接成一个小的函数，我们可以在为 MATH-500 数据集构建最终评估流水线之前用它来测试代码。代码从一个玩具示例（`ex`）开始，将问题渲染进提示模板（`prompt`），并从模型流式获取响应（`generate_text_stream_concat`）。然后，它将模型输出解析为最终候选答案（`pred_answer`），并使用 `grade_answer` 与真实答案进行对比评分。最后，它打印结果供我们评估。

调用 `mini_eval_demo(model, tokenizer, device)` 函数会产生以下输出：

```
Device: mps
Prediction: 1/3
Ground truth: 2/3
Correct: False
```

可以看到，生成的答案（`"1/3"`）被正确提取，但它与正确答案（`"2/3"`）不匹配，因此检查返回 `False`。（请注意，你的结果可能会因在 CPU、CUDA 或 MPS 设备上运行代码而有所不同。）

既然我们已经在一个简单示例上测试了工作流程，接下来让我们在 MATH-500 数据集上实现它。

### 清单 3.15 针对 MATH-500 数据集的端到端模型评估流水线
