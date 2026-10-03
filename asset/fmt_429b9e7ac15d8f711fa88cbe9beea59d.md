
观察到在列表 10.7 中之前展示的 LLM 输出，你可以理解为什么评估步骤可能会变得令人困惑。或许解决这个问题的一个方法就是建议 LLM 以单个陈述的形式提供最终答案。在下一节中，我们将继续探讨另一个提示推理的例子。

### 10.2.2 零样本 CoT 提示

正如我们的时间旅行所展示的，CoT 提示在特定类问题上的提示生成可能很昂贵。虽然效果不如前者，但有一些类似于 CoT 的技术不使用示例，并且可以更加通用。本节将检查用于在 LLM 中引发推理的简单短语。

在 VS Code 的提示流程视觉编辑器中打开`prompt_flow/zero-shot-cot-prompting/flow.dag.yaml`。这个流程与之前的 CoT 非常相似，如图 10.4 所示。下一个列表显示了描述该流程的 YAML 伪代码。

##### 列表 10.10 `zero-shot-CoT-prompting`流程

```py
   Inputs:
        statement  : the statement problem to be solved
        expected : the expected solution to the problem

LLM: cot (the prompt used to solve the problem)
        inputs: statement
        outputs: the predicted answer given the problem statement

LLM: evaluate_answer (the prompt used to evaluate the solution)
        inputs:
               statement: -> input.statement
               expected: -> input.expected
               predicted: -> output.cot

         outputs: a score of how well the problem was answered

    Outputs:
        statement: -> input.statement
        expected: -> input.expected
        predicted: -> output.cot
        evaluation_score: output.evaluate_answer

    ### Example Output
   {
       "evaluation_score": "1",     #1
       "expected": "After the final jump, ↪
          ↪ Max finds himself in the year 1980 and 
   he is 75 years old.",     #2
       "predicted": "Max starts in… ↪
          ↪ Therefore, after the final jump, ↪
          ↪ Max is 75 years old and in the year 1980.",     #3
       "statement": "In a complex time travel …"     #4
   }
```

#1 最终评估分数

#2 预期答案

#3 预测答案（已省略步骤以显示最终答案）

#4 初始问题陈述

在 VS Code 的视觉编辑器中按 `Shift-F5` 运行/测试流程。流程将运行，你应该会看到类似于列表 10.10 所示的输出。这个练习示例在相同问题上比之前的例子表现更好。

在 VS Code 中打开`cot.jinja2`提示，如列表 10.11 所示。这个提示比之前的例子简单得多，因为它只使用了零样本。然而，一个关键短语将这个简单的提示转换成了一个强大的推理引擎。提示中的这一行`让我们` `一步一步` `来思考`触发了 LLM 考虑内部上下文以展示推理。这反过来又指导 LLM 分步骤推理出问题。

##### 列表 10.11 `cot.jinja2`

```py
system:
You are an expert in solving time travel problems.
You are given a time travel problem and you have to solve it.
Let's think step by step.     #1
Please finalize your answer in a single statement.     #2

user:
{{statement}}     #3
```

#1 一行魔法代码，用于从 LLM 中构建推理

#2 要求 LLM 提供一个答案的最终陈述

#3 LLM 被要求解决的问题陈述

类似的短语要求 LLM 思考步骤或要求它以步骤的形式回答，也会提取推理。我们将在下一节中演示一个类似但更复杂的技术。

### 10.2.3 使用提示链分步骤

我们可以将向一个 LLM 逐步提问的行为扩展成一个链式提示，迫使 LLM 分步骤解决问题。在本节中，我们将探讨一种称为*提示链*的技术，该技术迫使 LLM 分步骤处理问题。

打开可视化编辑器中的`prompt_flow/prompt-chaining/flow.dag.yaml`文件，如图 10.5 所示。提示链将解决问题的推理方法分解成一系列提示。这项技术迫使 LLM 以步骤的形式回答问题。

![图](img/10-5.png)

##### 图 10.5 提示链流程

列表 10.12 展示了描述该流程的 YAML 伪代码，它提供了更多细节。这个流程将第一个 LLM 块的输出链式连接到第二个，然后从第二个连接到第三个。迫使 LLM 以这种方式处理问题揭示了推理模式，但它也可能过于冗长。

##### 列表 10.12 `prompt-chaining`流程

```py
   Inputs:
        statement  : the statement problem to be solved

   LLM: decompose_steps (the prompt used to decompose the problem)
        inputs: 
               statement: -> input.statement     #1

        outputs: the breakdown of steps to solve the problem

   LLM: calculate_steps (the prompt used to calculate the steps)
        inputs:
               statement: -> input.statement
               decompose_steps: -> output.decompose_steps     #2

               outputs: the calculation for each step
   LLM: calculate_solution (attempts to solve the problem)
        inputs:
               statement: -> input.statement
               decompose_steps: -> output.decompose_steps
               calculate_steps: -> output.calculate_steps     #3

         outputs: the final solution statement

   Outputs:
        statement: -> input.statement
        decompose_steps: -> output.decompose_steps
        calculate_steps: -> output.calculate_steps
        calculate_solution: -> output.calculate_solution

   ### Example Output
{
    "calculate_steps": "1\. The days spent by Alex",
    "decompose_steps": "To figure out the …",
    "solution": "Alex spends 13 days in the ↪
           ↪ past before the end of the battle.",     #4
    "statement": "In a sci-fi film, Alex …"    
}
```

#1 提示链的开始

#2 将前一步的输出注入到这一步

#3 将前两个步骤的输出注入到这一步

#4 最终的解决方案陈述，尽管是错误的，但更接近了。

通过从可视化编辑器按 `Shift-F5` 运行流程，你会看到如列表 10.12 所示的输出。对于 Alex 问题，答案仍然不正确，但我们可以看到 LLM 为了推理出问题所做的工作。

打开所有三个提示：`decompose_steps.jinja2`、`calculate_steps.jinja2`和`calculate_solution.jinja2`（分别见列表 10.13、10.14 和 10.15）。列表中展示的所有三个提示可以进行比较，以展示输出是如何链式连接的。

##### 列表 10.13 `decompose_steps.jinja2`

```py
system:
You are a problem solving AI assistant.
Your job is to break the users problem down into smaller steps and list 
the steps in the order you would solve them.
Think step by step, not in generalities.
Do not attempt to solve the problem, just list the steps. #1

user:
{{statement}}     #2
```

#1 迫使 LLM 只列出步骤，不列出其他任何内容

#2 初始问题陈述

##### 列表 10.14 `calculate_steps.jinja2`

```py
system:
You are a problem solving AI assistant.
You will be given a list of steps that solve a problem.
Your job is to calculate the output for each of the steps in order.
Do not attempt to solve the whole problem,
just list output for each of the steps.     #1
Think step by step.     #2

user:
{{statement}}

{{steps}}     #3
```

#1 要求 LLM 只解决步骤，而不是整个问题

#2 使用魔法语句提取推理

#3 将分解步骤步骤产生的步骤注入其中

##### 列表 10.15 `calculate_solution.jinja2`

```py
system:
You are a problem solving AI assistant.
You will be given a list of steps and the calculated output for each step.
Use the calculated output from each step to determine the final 
solution to the problem.
Provide only the final solution to the problem in a 
single concise sentence. Do not include any steps 
in your answer.     #1

user:
{{statement}}

{{steps}}     #2

{{calculated}}     #3
```

#1 要求 LLM 输出最终答案，而不输出任何步骤

#2 分解的步骤

#3 计算出的步骤

在这个练习示例中，我们并没有进行任何评估和评分。没有评估，我们可以看到这个提示序列仍然存在一些问题，无法解决之前在图 10.3 中展示的更具挑战性的时间旅行问题。然而，这并不意味着这项技术没有价值，这种提示格式在解决一些复杂问题方面表现良好。

然而，我们想要找到的是一种推理和规划方法，可以一致地解决如此复杂的问题。接下来的部分将从推理转向评估最佳解决方案。

## 10.3 使用评估来实现一致性的解决方案

在上一节中，我们了解到即使是经过最佳推理的计划也可能不会总是得出正确的解决方案。此外，我们可能并不总是有答案来确认该解决方案是否正确。现实情况是，我们经常想要使用某种形式的评估来确定解决方案的有效性。

图 10.6 显示了作为使 LLMs 进行推理和规划手段而设计的提示工程策略的比较。我们已经涵盖了左侧的两个：零样本直接提示和 CoT 提示。本节下面的示例练习将探讨与 CoT 和 ToT 技术结合的自洽性。

![figure](img/10-6.png)

##### 图 10.6 比较各种提示工程策略以实现从 LLMs 中进行推理和规划

我们将继续关注复杂的时间旅行问题，以比较这些更高级的方法，这些方法通过评估扩展了推理和规划。在下一节中，我们将评估自洽性。

### 10.3.1 评估自洽提示

提示的一致性不仅仅是降低我们发送给 LLM 的温度参数。通常，我们希望生成一个一致的计划或解决方案，同时使用较高的温度来更好地评估计划的全部变体。通过评估多个不同的计划，我们可以更好地了解解决方案的整体价值。

*自洽提示*是针对给定问题生成多个计划/解决方案的技术。然后，评估这些计划，并接受出现频率更高或更一致的计划。想象一下生成了三个计划，其中两个相似，但第三个不同。使用自洽性，我们将前两个计划评估为更一致的答案。

在 VS Code 提示流可视化编辑器中打开`prompt_flow/self-consistency-prompting/flow.dag.yaml`。流程图显示了图 10.7 中提示生成流程的简单性。在图旁边的是自洽评估流程。

![figure](img/10-7.png)

##### 图 10.7 自洽提示生成与评估流程并置

提示流使用直接无环图（DAG）格式来执行流程逻辑。DAGs 是展示和执行流程逻辑的绝佳方式，但由于它们是无环的，意味着它们不能重复，因此不能执行循环。然而，由于提示流提供了一个批量处理机制，我们可以使用它来模拟流程中的循环或重复。

参考图 10.6，我们可以看到自洽过程在收集结果并确定最佳计划/回复之前对输入进行了三次处理。我们可以应用相同的模式，但使用批量处理来生成输出。然后，评估流程将汇总结果并确定最佳答案。

在 VS Code 中打开`self-consistency-prompting/cot.jinja2`提示模板（参见列表 10.16）。列表已被缩短，因为我们之前已经看到了部分内容。此提示使用两个（少样本提示）CoT 示例来向 LLM 展示推理过程。

##### 列表 10.16 `self-consistency-prompting/cot.jinja2`

```py
system:

"In a time travel movie, Sarah travels back… "     #1

Chain of Thought:

    Initial Assumption: …     #2
    Conclusion: Sarah spends 729 days in the past before the day of the event.

"In a complex time travel movie plot, Max, a 25 year old…"     #3

Chain of Thought:
    Starting Point: Max starts …     #4
    Conclusion: After the final jump, 
Max finds himself in the year 1980 and he is 75 years old.
Think step by step,
 but only show the final answer to the statement.     #5

user:
{{statement}}
```

#1 Sarah 时间旅行问题

#2 样本 CoT，为了简洁已截断

#3 最大时间旅行问题

#4 样本 CoT，为了简洁已截断

#5 最终指南和声明以约束输出

在 VS Code 中打开`self-consistency-prompting/flow.dag.yaml`文件。通过从可视化编辑器中点击批量运行（水壶图标）来以批量模式运行示例。图 10.8 展示了逐步过程：

1.  点击批量运行。

1.  选择 JSON Lines (JSONL)输入。

1.  选择`statements.jsonl`。

1.  点击运行链接。

![图](img/10-8.png)

##### 图 10.8 启动批量过程的逐步过程

提示：如果您需要回顾过程，请参阅第九章，其中更详细地介绍了此过程。

列表 10.17 显示了批量模式下执行流程的 JSON 输出。`statements.jsonl`文件有五个相同的 Alex 时间旅行问题条目。使用相同的条目允许我们模拟在重复条目上执行提示五次。

##### 列表 10.17 `self-consistency-prompting` 批量执行输出
