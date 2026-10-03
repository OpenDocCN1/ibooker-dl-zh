

# 第六章\. 代理架构

建立在第五章中描述的架构之上，本章将涵盖所有当前 LLM 架构中可能最重要的架构——代理架构。首先，我们介绍使 LLM 代理独特之处，然后展示如何构建它们以及如何扩展它们以适应常见用例。

在人工智能领域，创建（智能）代理的历史悠久，正如斯图尔特·罗素和彼得·诺维格在他们所著的《人工智能》（Pearson, 2020）一书中所说：“某种能够行动的东西”可以最简单地定义代理。实际上，“行动”这个词的含义比表面上看要丰富一些：

+   行动需要一些决定做什么的能力。
+   决定做什么意味着可以访问多个可能的行动方案。毕竟，没有选择的决定根本不是决定。
+   为了做出决定，代理还需要获取有关外部环境（代理本身之外的一切）的信息。

因此，一个具有代理性的 LLM 应用必须是一个使用 LLM 从一种或多种可能的行动方案中选择，给定关于世界当前状态或所需下一个状态的一些上下文的应用。这些属性通常通过混合我们在前言中首次遇到的两种提示技术来实现：

**工具调用**

在你的提示中包含一个外部函数列表，LLM 可以利用这些函数（即它可以决定采取的行动），并提供如何格式化其输出中选择的说明。你很快就会看到提示中的样子。

**思维链**

研究人员发现，当 LLM 被指示通过将复杂问题分解为一系列按顺序执行的粒度步骤来进行推理时，它们“做出更好的决定”。这通常是通过添加类似“逐步思考”的指令或包括问题和它们分解为几个步骤/行动的例子来完成的。

这里有一个同时使用工具调用和思维链的示例提示：

```py
Tools:
search: this tool accepts a web search query and returns the top results.
calculator: this tool accepts math expressions and returns their result.

If you want to use tools to arrive at the answer, output the list of tools and
inputs in CSV format, with the header row: tool,input.

Think step by step; if you need to make multiple tool calls to arrive at the
answer, return only the first one.

How old was the 30th president of the United States when he died?

tool,input
```

当在温度 0（以确保 LLM 遵循所需的输出格式，CSV）和换行符作为停止序列（这指示 LLM 在达到此字符时停止产生输出）的情况下运行`gpt-3.5-turbo`时，输出，这使得 LLM 产生一个单一的行动（正如预期的那样，因为提示要求这样做）：

```py
search,30th president of the United States
```

最新的 LLM 和聊天模型已经经过微调，以提高它们在工具调用和思维链应用中的性能，从而消除了在提示中添加特定指令的需要：

```py
add example prompt and output for tool-calling model
```

# 计划-执行循环

使代理架构与第五章中讨论的架构不同的概念是我们尚未涉及的概念：由 LLM 驱动的循环。

每个程序员在他们的代码中都遇到过循环。通过“循环”，我们指的是运行相同的代码多次，直到达到停止条件。代理架构的关键是让 LLM 控制停止条件——也就是说，决定何时停止循环。

在这个循环中我们将运行以下内容的某种变体：

+   规划动作或动作
+   执行所述动作（们）

继续前一小节的例子，我们将运行输入为“美国第 30 任总统”的`search`工具，它产生以下输出：

```py
Calvin Coolidge (born John Calvin Coolidge Jr.; /ˈkuːlɪdʒ/; July 4, 1872 – January 
5, 1933) was an American attorney and politician who served as the 30th president 
of the United States from 1923 to 1929\. John Calvin Coolidge Jr.
```

然后我们将重新运行提示，增加一点：

```py
Tools:
search: this tool accepts a web search query and returns the top results.
calculator: this tool accepts math expressions and returns their result.
output: this tool ends the interaction. Use it when you have the final answer.

If you want to use tools to arrive at the answer, output the list of tools and 
inputs in CSV format, with this header row: tool,input

Think step by step; if you need to make multiple tool calls to arrive at 
the answer, return only the first one.

How old was the 30th president of the United States when he died?

tool,input

search,30th president of the United States

search: Calvin Coolidge (born John Calvin Coolidge Jr.; /ˈkuːlɪdʒ/; July 4, 1872 – 
January 5, 1933) was an American attorney and politician who served as the 30th 
president of the United States from 1923 to 1929\. John Calvin Coolidge Jr.

tool,input
```

以及输出：

```py
calculator,1933 - 1872
```

注意我们添加了两件事：

+   一个“输出”工具——当 LLM 找到最终答案时应该使用，我们也会用它作为停止循环的信号。
+   上一次迭代中工具调用的结果，简单地以工具名称及其（文本）输出。这是为了允许 LLM 继续进行下一步交互。换句话说，我们在告诉 LLM，“嘿，我们得到了你要求的结果，你接下来想做什么？”

让我们继续进行第三次迭代：

```py
Tools:
search: this tool accepts a web search query and returns the top results.
calculator: this tool accepts math expressions and returns their result.

If you want to use tools to arrive at the answer, output the list of tools and 
inputs in CSV format, with this header row: tool,input.
output: this tool ends the interaction. Use it when you have the final answer.

Think step by step; if you need to make multiple tool calls to arrive at 
the answer, return only the first one.

How old was the 30th president of the United States when he died?

tool,input

search,30th president of the United States

search: Calvin Coolidge (born John Calvin Coolidge Jr.; /ˈkuːlɪdʒ/; July 4, 1872 – 
January 5, 1933) was an American attorney and politician who served as the 30th 
president of the United States from 1923 to 1929\. John Calvin Coolidge Jr.
tool,input

calculator,1933-1872

calculator: 61

tool, input
```

以及输出：

```py
output, 61
```

使用“计算器”工具的结果，LLM 现在拥有了足够的信息来提供最终答案，因此它选择了“输出”工具，并选择了“61”作为最终答案。

这就是使代理架构如此有用的原因——LLM 被赋予了决策权。下一步是得出答案并决定采取多少步骤——也就是说，何时停止。

这个架构，称为[ReAct](https://oreil.ly/M7hF-)，最初由姚顺宇等人提出。本章的其余部分将探讨如何通过第五章中电子邮件助手示例来提高代理架构的性能。

但首先，让我们看看使用聊天模型和 LangGraph 实现基本代理架构的样子。

# 构建 LangGraph 代理

对于这个例子，我们需要为所选的搜索工具 DuckDuckGo 安装额外的依赖项。对于 Python 的安装：

*Python*

```py
pip install duckduckgo-search
```

对于 JS，我们还需要为计算器工具安装一个依赖项：

*JavaScript*

```py
npm i duck-duck-scrape expr-eval
```

完成这些后，让我们来看看实现代理架构的实际代码：

*Python*

```py
import ast
from typing import Annotated, TypedDict

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

@tool
def calculator(query: str) -> str:
    """A simple calculator tool. Input should be a mathematical expression."""
    return ast.literal_eval(query)

search = DuckDuckGoSearchRun()
tools = [search, calculator]
model = ChatOpenAI(temperature=0.1).bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def model_node(state: State) -> State:
    res = model.invoke(state["messages"])
    return {"messages": res}

builder = StateGraph(State)
builder.add_node("model", model_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "model")
builder.add_conditional_edges("model", tools_condition)
builder.add_edge("tools", "model")

graph = builder.compile()
```

*JavaScript*

```py
import {
  DuckDuckGoSearch
} from "@langchain/community/tools/duckduckgo_search";
import {
  Calculator
} from "@langchain/community/tools/calculator";
import {
  StateGraph,
  Annotation,
  messagesStateReducer,
  START,
} from "@langchain/langgraph";
import {
  ToolNode,
  toolsCondition
} from "@langchain/langgraph/prebuilt";

const search = new DuckDuckGoSearch();
const calculator = new Calculator();
const tools = [search, calculator];
const model = new ChatOpenAI({
  temperature: 0.1
}).bindTools(tools);

const annotation = Annotation.Root({
  messages: Annotation({
    reducer: messagesStateReducer,
    default: () => []
  }),
});

async function modelNode(state) {
  const res = await model.invoke(state.messages);
  return { messages: res };
}

const builder = new StateGraph(annotation)
  .addNode("model", modelNode)
  .addNode("tools", new ToolNode(tools))
  .addEdge(START, "model")
  .addConditionalEdges("model", toolsCondition)
  .addEdge("tools", "model");

const graph = builder.compile();
```

视觉表示如图图 6-1 所示。

![模型图  描述自动生成](img/lelc_0601.png)

###### 图 6-1. 代理架构

这里有几个需要注意的地方：

+   在这个例子中，我们使用了两个工具：一个搜索工具和一个计算器工具，但你可以很容易地添加更多或替换我们使用的工具。在 Python 示例中，你还可以看到一个创建自定义工具的例子。
+   我们使用了 LangGraph 附带的两个便利函数。`ToolNode`作为我们图中的一个节点；它执行在状态中找到的最新 AI 消息中请求的工具调用，并返回每个工具的结果的`ToolMessage`。`ToolNode`还处理由工具引发的异常——使用错误消息构建一个`ToolMessage`，然后将其传递给 LLM，LLM 可能会决定如何处理该错误。
+   `tools_condition`作为条件边函数，查看状态中的最新 AI 消息，如果有任何工具要执行，则路由到`tools`节点。否则，结束图。
+   最后，请注意，这个图在模型和工具节点之间循环。也就是说，模型本身负责决定何时结束计算，这是代理架构的关键属性。每当我们在 LangGraph 中编写循环时，我们可能会想使用条件边，因为这允许你定义当图应该退出循环并停止执行时的*停止条件*。

现在让我们看看它在前面的例子中的表现：

*Python*

```py
input = {
    "messages": [
        HumanMessage("""How old was the 30th president of the United States 
 when he died?""")
    ]
}
for c in graph.stream(input):
    print(c)
```

*JavaScript*

```py
const input = {
  messages: [
    HumanMessage(`How old was the 30th president of the United States when he 
 died?`)
  ]
}
for await (const c of await graph.stream(input)) {
  console.log(c)
}
```

*输出：*

