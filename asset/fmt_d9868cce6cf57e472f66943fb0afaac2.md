
# 提示组件

```py
# Prompt components
persona = "You are an expert in Large Language models. You excel at breaking down complex papers into digestible summaries.`\n`"
instruction = "Summarize the key findings of the paper provided.`\n`"
context = "Your summary should extract the most crucial points that can help researchers quickly understand the most vital information of the paper.`\n`"
data_format = "Create a bullet-point summary that outlines the method. Follow this up with a concise paragraph that encapsulates the main results.`\n`"
audience = "The summary is designed for busy researchers that quickly need to grasp the newest trends in Large Language Models.`\n`"
tone = "The tone should be professional and clear.`\n`"
data = "Text to summarize: PUT_THE_DATA_TO_SUMMARIZE_HERE"
# The full prompt - remove and add pieces to view its impact on the generated output
query = persona + instruction + context + data_format + audience + tone + data
prompt = format_prompt(query)
```

###### 提示

几乎每周都有新的提示组件可能提高输出的准确性。我们可以添加各种各样的组件，每周都会发现使用情感刺激（例如，“这对我的职业非常重要。”）等创造性组件。

提示工程的一大乐趣在于你可以尽可能地富有创造力，以找出哪些提示组件的组合对你的用例有帮助。开发一个适合你的格式几乎没有约束。

然而，请注意，某些提示对特定模型的效果更好，因为它们的训练数据可能不同，或者它们的训练目的不同。

## 上下文学习：提供示例

在之前的章节中，我们试图准确描述大型语言模型（LLM）应该做什么。尽管准确和具体的描述有助于 LLM 理解用例，但我们可以更进一步。

我们为什么不直接展示任务，而是描述它呢？

我们可以向 LLM 提供我们想要实现的事物的确切示例。这通常被称为上下文学习，我们向模型提供正确的示例。

如图 4-14 所示，展示给 LLM 的示例数量会影响形式。零-shot 提示不利用示例，one-shot 提示使用一个示例，而 few-shot 提示使用两个或更多示例。

![复杂提示的示例，包含多个组件。](img/text_generation_with_gpt_models_479875_14.png)

###### 图 4-14. 复杂提示的示例，包含多个组件。

采用原始短语，我们认为“一个例子胜过千言万语”。这些示例提供了 LLM 应该如何实现的直接示例。

我们可以用一个简单的示例来说明这种方法，该示例取自描述此方法的原始论文。提示的目标是生成一个包含虚构单词的句子。为了提高生成句子的质量，我们可以向生成模型展示一个包含虚构单词的正确句子的示例。

为此，我们需要区分我们的提问（`user`）和模型提供的答案（`assistant`）：

```py
# Use a single example of using the made-up word in a sentence
one_shot_prompt = format_prompt(messages=[
    {"role": "user", "content": "Q: A 'Gigamuru' is a type of Japanese musical instrument. An example of a sentence that uses the word Gigamuru is:"},
    {"role": "assistant", "content": "A: I have a Gigamuru that my uncle gave me as a gift. I love to play it at home."},
    {"role": "user", "content": "Q: To 'screeg' something is to swing a sword at it. An example of a sentence that uses the word screeg is:"}
])
print(one_shot_prompt)
```

该提示说明了区分用户和助手的必要性。如果我们不这样做，就会似乎在自言自语：

```py
"""
<|user|>
Q: A 'Gigamuru' is a type of Japanese musical instrument. An example of a sentence that uses the word Gigamuru is:</s>
<|assistant|>
A: I have a Gigamuru that my uncle gave me as a gift. I love to play it at home.</s>
<|user|>
Q: To 'screeg' something is to swing a sword at it. An example of a sentence that uses the word screeg is:</s>
<|assistant|>
"""
```

我们可以使用这个提示来运行我们的模型：

```py
# Run generative model
outputs = pipe(one_shot_prompt, max_new_tokens=64, do_sample=`True`, return_full_text=`False`)
print(outputs[0]["generated_text"])
```

结果是一个使用虚构词“screeg”的正确句子：

```py
"A: I screeged the dragon's tail with my sword, but it only seemed to make it angrier."
```

与所有提示组件一样，一次或几次提示并不是提示工程的终极解决方案。我们可以将其作为增强我们给出的描述的拼图的一部分。模型仍然可以通过随机抽样“选择”忽略指令。

## 链式提示：拆分问题

在之前的示例中，我们探索了将提示拆分为模块化组件，以提高 LLM 的性能。尽管这在许多用例中效果很好，但对于高度复杂的提示或用例，这可能不可行。

我们可以在提示之间拆分问题，而不是在提示内部。基本上，我们将一个提示的输出作为下一个提示的输入，从而创建一个连续的交互链，解决我们的难题。

为了说明，假设我们希望使用 LLM 为我们根据一系列产品特征创建产品名称、口号和销售推介。尽管我们可以要求 LLM 一次性完成，但我们可以将问题拆分为多个部分。

因此，如在图 4-16 中所示，我们得到一个顺序管道，首先创建产品名称，然后将其与产品特征作为输入创建口号，最后使用特征、产品名称和口号来创建销售推介。

![使用产品特征的描述，链式提示创建合适的名称、口号和销售推介。](img/text_generation_with_gpt_models_479875_15.png)

###### 图 4-15. 使用产品特征的描述，链式提示创建合适的名称、口号和销售推介。

这种链式提示的技术使 LLM 能够在每个单独问题上花费更多时间，而不是处理整个问题。

让我们用一个小例子来说明。

```py
# Create name and slogan for a product
product_prompt = format_prompt("Create a name and slogan for a chatbot that leverages LLMs.")
outputs = pipe(product_prompt, max_new_tokens=32, do_sample=`True`, return_full_text=`False`)
product_description = outputs[0]["generated_text"]
# Use name and slogan as input for a sales pitch
sales_prompt = format_prompt(f"What would be a good sales pitch for the following product: '`{`product_description`}`'?")
outputs = pipe(sales_prompt, max_new_tokens=128, do_sample=`True`, return_full_text=`False`)
sales_pitch = outputs[0]["generated_text"]
# Results
print(product_description)
print(sales_pitch)
```

在这个例子中，我们首先要求模型创建名称和口号。然后，我们可以使用输出请求基于产品特征的良好销售推介。

这给我们带来了以下输出：

```py
"""
Name: LLM Assistant
Slogan: "Your go-to chatbot powered by cutting-edge language learning models."
Introducing LLM Assistant, the revolutionary chatbot that transforms the way you learn languages. Unlike traditional language learning methods, LLM Assistant utilizes the latest language learning models powered by Artificial Intelligence. With LLM Assistant, you can communicate with native speakers, practice real-life conversations, and receive instant feedback. Whether you're a novice or an advanced speaker, LLM Assistant caters to your unique learning needs, making language learning fun, interactive, and efficient. So, why wait? Say hello to LLM Assistant, your new language learning companion!
"""
```

虽然我们需要两次调用模型，但一个主要好处是我们可以给每次调用不同的参数。例如，名称和口号创建的令牌数量相对较少，而推介可以更长。

它可以用于多种用例，包括：

*响应验证*

请求 LLM 重新检查之前生成的输出

*并行提示*

并行创建多个提示，并进行最终合并。例如，请多个 LLM 并行生成多个食谱，并使用组合结果创建购物清单。

*写故事*

利用 LLM 撰写书籍或故事，可以将问题拆解为各个组成部分。例如，首先撰写摘要，再发展角色并构建故事情节，然后再深入创作对话。

在第六章中，我们将超越链接 LLM，连接其他技术组件，如内存、搜索等！在此之前，提示链的这一概念将在接下来的部分中进一步探讨，描述更复杂的提示链方法，如自一致性、链式思维和树状思维。

# 与生成模型的推理

在之前的部分中，我们主要集中于提示的模块化组件，通过迭代构建它们。这些先进的提示工程技术，如提示链，证明了实现生成模型复杂推理的第一步。

为了允许这种复杂的推理，现在是退后一步探讨推理内容的好时机。简单来说，我们的推理方法可以分为系统 1 和系统 2 思维过程，如图 5-X 所示。

系统 1 思维代表自动、直觉和几乎瞬时的反应。它与生成模型有相似之处，生成模型自动生成令牌而没有任何自我反思行为。相反，系统 2 思维是一个有意识、缓慢和逻辑的过程，类似于头脑风暴和自我反思。

如果我们能够赋予生成模型自我反思的能力，我们实际上是在模拟系统 2 的思维方式，这通常比系统 1 思维产生更深思熟虑的响应。

在本节中，我们将探讨几种技术，试图模仿这些人类推理者的思维过程，旨在提高模型的输出。

## 链式思维：回答前思考

实现生成模型复杂推理的第一大步骤是通过一种称为链式思维（CoT）的方法。CoT 旨在让生成模型“思考”后再回答问题，而不是直接回答而不进行任何推理。

如图 4-16 所示，它在提示中提供了示例，展示了模型在生成响应之前应进行的推理。这些推理过程被称为“思考”。这对涉及更高复杂度的任务（如数学问题）帮助巨大。增加这个推理步骤使模型能够在推理过程中分配更多计算资源。

![链式思维提示使用推理示例来说服生成模型在其回答中使用推理。](img/text_generation_with_gpt_models_479875_16.png)

###### 图 4-16. 链式思维提示使用推理示例来说服生成模型在其回答中使用推理。

我们将使用他们在论文中使用的例子来演示这一现象。首先，让我们探讨一下没有 CoT 的标准提示的输出。我们在提供示例时，不仅提供单一查询，而是区分用户和助手：

```py
# Answering without explicit reasoning
standard_prompt = format_prompt(messages=[
    {"role": "user", "content": "Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?"},
    {"role": "assistant", "content": "A: The answer is 11."},
    {"role": "user", "content": "Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?"}
])
# Run generative model
outputs = pipe(standard_prompt, max_new_tokens=64, do_sample=`True`, return_full_text=`False`)
print(outputs[0]["generated_text"])
```

这给出了错误的答案：

```py
"A: The answer is 26."
```

相反，我们将使用 CoT 让模型在给出答案之前展示其推理过程：

```py
# Answering with chain-of-thought
cot_prompt = format_prompt(messages=[
    {"role": "user", "content": "Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?"},
    {"role": "assistant", "content": "A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11."},
    {"role": "user", "content": "Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?"}
])
# Run generative model
outputs = pipe(cot_prompt, max_new_tokens=256, do_sample=`True`, return_full_text=`False`)
print(outputs[0]["generated_text"])
```

这次，我们得到了正确的响应：

```py
"A: Initially, there were 23 apples. They used 20 apples to make lunch, leaving 3 apples (23 - 20 = 3). Then they bought 6 more apples. So, in total, the cafeteria now has (3 + 6) apples, which is 9 apples in total. The answer is 9."
```

这个推理过程尤其有帮助，因为模型是在生成答案之前这样做的。通过这样做，它可以利用迄今为止生成的知识来计算正确的答案。

### 零-shot 思维链

尽管 CoT 是增强生成模型输出的好方法，但它确实需要一个或多个推理示例，而用户可能没有这些示例。
