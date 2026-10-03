
因此，无论是使用 JS 还是 Python，你都可以通过返回你想要流式的值并使用`stream`来调用它来为你的自定义函数启用流式传输。

对于异步执行，你需要这样重写你的函数：

*Python*

```py
@chain
async def chatbot(values):
    prompt = await template.ainvoke(values)
    return await model.ainvoke(prompt)

await chatbot.ainvoke({"question": "Which model providers offer LLMs?"})
# > AIMessage(content="""Hugging Face's `transformers` library, OpenAI using
    the `openai` library, and Cohere using the `cohere` library offer LLMs.""")
```

这个只适用于 Python，因为在 JavaScript 中异步执行是唯一的选择。

## 声明式组合

LCEL 是一种用于组合 LangChain 组件的*声明式语言*。LangChain 将 LCEL 组合编译成*优化后的执行计划*，具有自动并行化、流式处理、跟踪和异步支持。

让我们用 LCEL 来查看相同的例子：

*Python*

```py
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# the building blocks

template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant.'),
    ('human', '{question}'),
])

model = ChatOpenAI()

# combine them with the | operator

chatbot = template | model

# use it

chatbot.invoke({"question": "Which model providers offer LLMs?"})
```

*JavaScript*

```py
import { ChatOpenAI } from '@langchain/openai'
import { ChatPromptTemplate } from '@langchain/core/prompts'
import { RunnableLambda } from '@langchain/core/runnables'

// the building blocks

const template = ChatPromptTemplate.fromMessages([
  ['system', 'You are a helpful assistant.'],
  ['human', '{question}'],
])

const model = new ChatOpenAI()

// combine them in a function

const chatbot = template.pipe(model)

// use it

await chatbot.invoke({
  "question": "Which model providers offer LLMs?"
})
```

*输出：*

```py
AIMessage(content="Hugging Face's `transformers` library, OpenAI using the 
    `openai` library, and Cohere using the `cohere` library offer LLMs.")
```

关键的是，两个例子中的最后一行是相同的——也就是说，你使用函数和 LCEL 序列的方式相同，使用`invoke/stream/batch`。在这个版本中，你不需要做任何事情就可以使用流式处理：

*Python*

```py
chatbot = template | model

for part in chatbot.stream({
    "question": "Which model providers offer LLMs?"
}):
    print(part)
    # > AIMessageChunk(content="Hugging")
    # > AIMessageChunk(content=" Face's")
    # > AIMessageChunk(content=" `transformers`")
    # ...
```

*JavaScript*

```py
const chatbot = template.pipe(model)

for await (const token of await chatbot.stream({
  "question": "Which model providers offer LLMs?"
})) {
  console.log(token)
}
```

并且，对于 Python 来说，使用异步方法也是一样的：

*Python*

```py
chatbot = template | model

await chatbot.ainvoke({
    "question": "Which model providers offer LLMs?"
})
```

# 摘要

在本章中，你学习了构建 LLM 应用程序所需的构建块和关键组件，使用 LangChain 构建 LLM 应用程序。LLM 应用程序本质上是一个由大型语言模型（用于做出预测）、提示指令（用于引导模型达到期望的输出）以及可选的输出解析器（用于转换模型输出的格式）组成的链。

所有 LangChain 组件都共享相同的接口，使用`invoke`、`stream`和`batch`方法来处理各种输入和输出。它们可以通过直接调用或使用 LCEL 声明式地组合和执行。

如果你想编写大量的自定义逻辑，命令式方法是有用的，而声明式方法则适用于简单地组装现有组件，且定制化有限。

在第二章中，你将学习如何提供外部数据作为*上下文*给你的 AI 聊天机器人，这样你就可以构建一个 LLM 应用程序，让你能够“聊天”你的数据。
