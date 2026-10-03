
## Give GPTs “Thinking Time”

Often, by explicitly guiding an LLM to *derive solutions from first principles* before reaching a verdict, you can garner more accurate responses. Providing an LLM with *thinking time* can often lead to better results.

Input:

```
```

Output:

```
```

In some cases, when using GPT-4 or other generative AI models, you may not want the model’s reasoning process to be visible to the user.

You can achieve this by asking an LLM to *generate an inner monologue*. This is particularly useful in tutoring applications, where revealing the model’s reasoning might give away the solution prematurely.

## The Inner Monologue Tactic

The *inner monologue tactic* instructs the model to structure parts of the output that should be hidden from the user in a specific format. This makes it easy to remove these parts before presenting the final output to the user.

Here’s how you can utilize this tactic to answer user queries effectively.

Input:

```
指导他们找到正确答案，而不直接给出答案。

这个部分也应该在三个引号内。

第四步：交付提示

如果有错误，向用户展示上一步的提示。

记住不要在这个步骤中使用三引号。而不是使用标题

一样，使用 `提示:` 这样的标题

问题陈述：一副牌有 52 张牌：13 张红心、13 张方块、13

梅花、13 张黑桃。如果从牌组中抽取两张牌而不进行

抽牌，那么抽出的牌中至少有一张红心的概率是多少

和另一张是梅花吗？

用户解决方案：（13/52）*（13/51）= 169/2652 = 13/204
```

Output:

```
"""

第一步 - 我的解决方案：

首先，计算从牌组中抽取 1 张红心和 1 张梅花的概率

52 张牌的牌组，不进行抽牌。

在第一次抽取中，抽取红心的概率是 13/52。

对于第二次抽取，在移除 1 张牌（假设是红心）后，剩余的

总牌数变为 51 张。因此，现在抽取梅花的概率是

是 13/51。

将这两个概率相乘，得到两个事件同时发生的概率

按顺序发生，即 P(红心然后梅花) = (13/52)*(13/51)。

假设第一张抽出的牌是梅花，然后重复计算

第二次抽取的牌是红心。记住，在这种情况下，顺序很重要，

我们处理的是独立事件。

"""

"""

第二步 - 与用户解决方案的比较：

将你的解决方案与用户解决方案进行比较。用户的解决方案只

考虑先抽红心然后抽梅花的顺序。他们

没有考虑到另一种序列（梅花然后红心），并且

因此，解决方案是不完整的。

"""

"""

第三步 - 提示准备：

由于用户没有考虑两种序列（红心然后梅花

以及梅花然后红心），这是对考虑两种可能

序列将有助于引导用户走向正确的方向。这可能会

方向而不直接透露解决方案。

"""

# 第四步 - 提示交付：

# 提示：记住两张牌可以以任何顺序抽取。考虑

两种可能的顺序：红心然后梅花和梅花然后红心。
```

## Self-Eval LLM Responses

Another tactic you can use is to *critque a generated LLM output* and ask whether the LLM missed any information or important facts. You’re essentially asking an LLM to *evaluate itself* based on its previous output.

First, create a prompt to generate a `"Hello World"` function.

Input:

```
生成一个 "Hello World" Python 函数。
```

Output:

```
是的，这是一个非常简单的 Python 函数，它将 "Hello, World!" 打印到

控制台：

def print_hello_world():

    print("Hello, World!")

打印 "Hello, World!" 的函数。

当你运行这个函数时，它将打印字符串 "Hello, World!" 到

控制台。
```

The LLM returns a simple Python function called `print_hello_world()` that prints the traditional “Hello, World!” greeting.

Then make a second LLM request with the previous chat history, asking for the initial output to be improved.

Input:

```
...之前的聊天记录...

给定前面的函数，你能用其他方式再次改进它吗？

也许默认值或类型提示？
```

Output:

```
我们可以添加类型提示来清楚地表明该函数不接受任何

参数且不返回任何内容。我们还可以添加一个文档字符串来

描述函数的功能。

这里是改进后的代码：

def print_hello_world() -> None:

    """

    将 "Hello, World!" 打印到标准输出。

    """

    print("Hello, World!")

如果你想使函数更灵活，可以添加一个默认

允许你打印不同的消息：

def print_message(message: str = "Hello, World!") -> None:

    """

    打印给定的消息到标准输出。如果没有提供消息

    , 默认为 "Hello, World!"。

    Args:

        message (str, optional): 要打印的消息。

        默认为 "Hello, World!"。

    """

    print(message)
```

Consequently GPT-4 returns an upgraded version of the function, now furnished with Python type hints and a default argument. This enriches the function with greater flexibility, allowing it to print not just `"Hello, World!"` but any user-specified message.

These prompt-response exchanges illustrate how you can easily refine generated LLM outputs until you’re satisfied with the final output.

###### Note

It’s possible to critique an LLM’s response multiple times, until no further refinement is provided by the LLM.

# Classification with LLMs

Classifying, in the context of AI, refers to the process of predicting the class or category of a given data point or sample. It’s a common task in machine learning where models are trained to assign predefined labels to unlabeled data based on learned patterns.

LLMs are powerful assets when it comes to classification, even with zero or only a small number of examples provided within a prompt. Why? That’s because LLMs, like GPT-4, have been previously trained on an extensive dataset and now possess a degree of reasoning.

There are two overarching strategies in solving classification problems with LLMs: *zero-shot learning* and *few-shot learning*.

**Zero-shot learning**

In this process, the LLM classifies data with exceptional accuracy, without the aid of any prior specific examples. It’s akin to acing a project without any preparation—impressive, right?

**Few-shot learning**

Here, you provide your LLM with a small number of examples. This strategy can significantly influence the structure of your output format and enhance the overall classification accuracy.

**Why is this groundbreaking for you?**

Leveraging LLMs lets you sidestep lengthy processes that traditional machine learning processes demand. Therefore, you can quickly prototype a classification model, determine a base level accuracy, and create immediate business value.

###### Warning

Although an LLM can perform classification, depending upon your problem and training data you might find that using a traditional machine learning process could yield better results.

# Building a Classification Model

Let’s explore a few-shot learning example to determine the sentiment of text into either `'Compliment'`, `'Complaint'`, or `'Neutral'`.

```
Given the statement, classify it as either "Compliment", "Complaint", or

"中立":

1. "太阳正在照耀。" - 中立

2. "你们的支持团队太棒了！" - 表扬

3. "我使用你们的软件时遇到了糟糕的经历。" - 投诉

您必须遵循以下原则：

- 只返回单个分类词。响应应该是

"Compliment", "Complaint", or "Neutral".

- 在 """定界符内的文本上执行分类。

"""用户界面直观。”

Classification:
```

```
表扬
```

Several good use cases for LLM classification include:

**Customer reviews**

Classify user reviews into categories like “Positive,” “Negative,” or “Neutral.” Dive deeper by further identifying subthemes such as “Usability,” “Customer Support,” or “Price.”

**Email filtering**

Detect the intent or purpose of emails and classify them as “Inquiry,” “Complaint,” “Feedback,” or “Spam.” This can help businesses prioritize responses and manage communications efficiently.

**Social media sentiment analysis**

Monitor brand mentions and sentiment across social media platforms. Classify posts or comments as “Praise,” “Critic,” “Query,” or “Neutral.” Gain insights into public perception and adapt marketing or PR strategies accordingly.

**News article categorization**

Given the vast amount of news generated daily, LLMs can classify articles by themes or topics such as “Politics,” “Technology,” “Environment,” or “Entertainment.”

**Résumé screening**

For HR departments inundated with résumés, classify them based on predefined criteria like “Qualified,” “Overqualified,” “Underqualified,” or categorize by expertise areas such as “Software Development,” “Marketing,” or “Sales.”

###### Warning

Be aware that exposing emails, résumés, or sensitive data does run the risk of data being leaked into OpenAI’s future models as training data.

# Majority Vote for Classification

Utilizing multiple LLM requests can help in reducing the variance of your classification labels. This process, known as *majority vote*, is somewhat like choosing the most common fruit out of a bunch. For instance, if you have 10 pieces of fruit and 6 out of them are apples, then apples are the majority. The same principle goes for choosing the majority vote in classification labels.

By soliciting several classifications and taking the *most frequent classification*, you’re able to reduce the impact of potential outliers or unusual interpretations from a single model inference. However, do bear in mind that there can be significant downsides to this approach, including the increased time required and cost for multiple API calls.

Let’s classify the same piece of text three times, and then take the majority vote:

```
from openai import OpenAI

import os
```
