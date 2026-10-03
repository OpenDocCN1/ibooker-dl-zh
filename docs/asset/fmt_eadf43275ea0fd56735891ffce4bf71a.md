
# 第五章\. 基于 LangGraph 的认知架构

到目前为止，我们已经探讨了 LLM 应用程序最常见的特点：

+   序言和第一章中的提示技术
+   第二章和第三章中的 RAG
+   第四章中的记忆

接下来的问题应该是：我们如何将这些组件组合成一个连贯的应用程序，以实现我们设定的目标？为了与砖石世界做一个类比，游泳池和一层的房子是由相同的材料建造的，但显然服务于非常不同的目的。使它们各自适合其不同目的的是这些材料组合的计划——即它们的架构。在构建 LLM 应用程序时也是如此。你必须做出的最重要的决定是如何将你拥有的不同组件（如 RAG、提示技术、记忆）组装成实现你目的的东西。

在我们查看具体架构之前，让我们通过一个例子来了解一下。你可能会构建的任何 LLM 应用程序都将从一个目的开始：应用程序被设计来做什么。假设你想构建一个电子邮件助手——一个在你看邮件之前阅读你的邮件的 LLM 应用程序，目的是减少你需要查看的邮件数量。应用程序可能通过存档一些不感兴趣的邮件，直接回复一些，并将其他标记为需要你稍后注意的邮件来实现这一点。

你可能还希望应用程序在行动上受到一些约束。列出这些约束非常有帮助，因为它们将有助于指导对正确架构的搜索。第八章更详细地介绍了这些约束以及如何处理它们。对于这个假设的电子邮件助手，假设我们希望它做以下事情：

+   最小化它打扰你的次数（毕竟，整个目的就是节省时间）。
+   避免让你的电子邮件联系人收到你永远不会自己发送的回复。

这暗示了在构建 LLM 应用程序时经常面临的关键权衡：在 *自主性*（或自主行动的能力）和 *可靠性*（或你可以信任其输出的程度）之间的权衡。直观地讲，如果电子邮件助手在没有你介入的情况下采取更多行动，它将更有用，但如果你做得太过分，它不可避免地会发送你希望它没有发送的电子邮件。

描述一个 LLM 应用程序的自主程度的一种方法是通过评估应用程序的行为中有多少是由 LLM（而不是代码）决定的：

+   让 LLM 决定一个步骤的输出（例如，为电子邮件写一个草稿回复）。
+   让 LLM 决定下一步要采取的行动（例如，对于一封新邮件，决定它可以在邮件上执行的三种操作：存档、回复或标记为审阅）。
+   让 LLM 决定可以采取哪些步骤（例如，让 LLM 编写执行你未预先编程到应用程序中的动态操作的代码）。

我们可以根据它们在这条自主性谱系中的位置来对构建 LLM 应用的许多流行**食谱**进行分类，也就是说，三个提到的任务中哪些是由 LLM 处理的，哪些仍然掌握在开发者或用户手中。这些食谱可以被称为**认知架构**。在人工智能领域，**认知架构**这个术语长期以来一直被用来表示人类推理模型（及其在计算机中的实现）。一个 LLM 认知架构（据我们所知，这个术语首次应用于 LLM，是在一篇论文中提出的）可以被定义为 LLM 应用要采取的步骤的**食谱**（参见图 5-1）。例如，一个**步骤**可以是检索相关文档（RAG），或者使用思维链提示调用 LLM。

![计算机应用程序的截图  自动生成的描述](img/lelc_0501.png)

###### 图 5-1\. LLM 应用的认知架构

现在让我们看看在构建您的应用时可以使用的主要架构或**食谱**（如图图 5-1 所示）：

0: 代码

这不是一个 LLM 认知架构（因此我们将其编号为**0**），因为它根本不使用 LLM。你可以把它想象成你习惯编写的常规软件。对于这本书来说，第一个有趣的架构实际上是下一个。

1: LLM 调用

这是我们到目前为止在书中看到的绝大多数例子，只有一个 LLM 调用。这主要在它是更大应用的一部分时有用，该应用使用 LLM 来完成特定任务，例如翻译或总结一段文本。

2: 链接

更高级别的是，使用预定义序列中的多个 LLM 调用。例如，一个文本到 SQL 应用（它从用户那里接收对数据库中某些计算的自然语言描述作为输入）可以连续使用两个 LLM 调用：

一个 LLM 调用，从用户提供的自然语言查询和开发者提供的数据库内容描述中生成 SQL 查询。

另一个 LLM 调用是编写一个适合非技术用户的查询解释，基于前一个调用中生成的查询。然后可以使用这个解释来使用户检查生成的查询是否与他的请求匹配。

3: 路由器

下一个步骤来自使用 LLM 来定义要采取的步骤序列。也就是说，链式架构总是执行由开发者确定的静态步骤序列（无论多少），而路由架构的特点是使用 LLM 在某些预定义步骤之间进行选择。一个例子是一个 RAG 应用程序，它包含来自不同领域的多个文档索引，以下步骤：

1.  根据用户提供的查询和开发者提供的索引描述，LLM 调用来选择使用哪个可用的索引。
2.  一个检索步骤，查询所选索引以获取用户查询的最相关文档。
3.  另一次 LLM 调用来生成答案，给定用户提供的查询和从索引中检索到的相关文档列表。

这就是本章我们将要讨论的内容。我们将依次讨论这些架构。下一章将讨论代理架构，这些架构会更多地使用 LLM。但首先，让我们谈谈一些更好的工具，以帮助我们在这个旅程中。

# 架构 #1：LLM 调用

作为 LLM 调用架构的例子，我们将回到我们在第四章中创建的聊天机器人。这个聊天机器人将直接响应用户的消息。

首先，创建一个 `StateGraph`，我们将向其中添加一个节点来表示 LLM 调用：

*Python*

```py
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

model = ChatOpenAI()

class State(TypedDict):
    # Messages have the type "list". The `add_messages` 
    # function in the annotation defines how this state should 
    # be updated (in this case, it appends new messages to the 
    # list, rather than replacing the previous messages)
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    answer = model.invoke(state["messages"])
    return {"messages": [answer]}

builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, 'chatbot')
builder.add_edge('chatbot', END)

graph = builder.compile()
```

*JavaScript*

```py
import {
  StateGraph,
  Annotation,
  messagesStateReducer,
  START, END
} from '@langchain/langgraph'
import {ChatOpenAI} from '@langchain/openai'

const model = new ChatOpenAI()

const State = {
  /**
 * The State defines three things:
 * 1\. The structure of the graph's state (which "channels" are available to 
 * read/write)
 * 2\. The default values for the state's channels
 * 3\. The reducers for the state's channels. Reducers are functions that 
 * determine how to apply updates to the state. Below, new messages are 
 * appended to the messages array.
 */
  messages: Annotation({
    reducer: messagesStateReducer,
    default: () => []
  }),
}

async function chatbot(state) {
  const answer = await model.invoke(state.messages)
  return {"messages": answer}
}

const builder = new StateGraph(State)
  .addNode('chatbot', chatbot)
  .addEdge(START, 'chatbot')
  .addEdge('chatbot', END)

const graph = builder.compile()
```

我们还可以绘制一个图形的视觉表示：

*Python*

```py
graph.get_graph().draw_mermaid_png()
```

*JavaScript*

```py
await graph.getGraph().drawMermaidPng()
```

我们刚才创建的图形看起来像图 5-2。

![聊天机器人的图示  自动生成的描述](img/lelc_0502.png)

###### 图 5-2\. LLM 调用架构

你可以用在前面章节中看到的熟悉的 `stream()` 方法来运行它：

*Python*

```py
input = {"messages": [HumanMessage('hi!)]}
for chunk in graph.stream(input):
    print(chunk)
```

*JavaScript*

```py
const input = {messages: [new HumanMessage('hi!)]}
for await (const chunk of await graph.stream(input)) {
  console.log(chunk)
}
```

*输出：*

```py
{ "chatbot": { "messages": [AIMessage("How can I help you?")] } }
```

注意到图形的输入与我们在前面定义的 `State` 对象的形状相同；也就是说，我们发送了一个字典中 `messages` 键的消息列表。

这是使用 LLM 的最简单架构，但这并不意味着它永远不会被使用。以下是一些例子，说明你可能在许多流行的产品中看到它在行动：

+   由 AI 驱动的功能，如摘要和翻译（例如在流行的写作软件 Notion 中可以找到的），可以通过一次 LLM 调用来实现。
+   简单的 SQL 查询生成可以通过一次 LLM 调用实现，具体取决于开发者心中所想的 UX 和目标用户。

# 架构 #2：链式

这个架构通过使用多个 LLM 调用，按照预定义的顺序（即，不同的应用程序调用执行相同的 LLM 调用序列，尽管输入和结果不同）来扩展所有这些内容。

以一个文本到 SQL 应用程序为例，该程序接收用户输入的自然语言描述，用于对数据库进行某些计算。我们之前提到，这可以通过一次 LLM 调用来实现，生成一个 SQL 查询，但我们可以通过按顺序使用多个 LLM 调用创建一个更复杂的应用程序。一些作者称这种架构为*流程工程*。2

首先让我们用文字描述一下流程：

1.  使用一次 LLM 调用，从用户提供的自然语言查询和开发者提供的数据库内容描述中生成 SQL 查询。
2.  使用另一个 LLM 调用，根据前一个调用生成的查询，为非技术用户编写查询的解释。这可以用来让用户检查生成的查询是否符合他的要求。

你还可以进一步扩展（但在这里我们不会这么做）在前面两个步骤之后采取的额外步骤：

1.  在数据库上执行查询，返回一个二维表。
2.  使用第三次 LLM 调用将查询结果总结为对原始用户问题的文本答案。

现在让我们用 LangGraph 来实现这个架构：

*Python*

```py
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# useful to generate SQL query
model_low_temp = ChatOpenAI(temperature=0.1)
# useful to generate natural language outputs
model_high_temp = ChatOpenAI(temperature=0.7)

class State(TypedDict):
    # to track conversation history
    messages: Annotated[list, add_messages]
    # input
    user_query: str
    # output
    sql_query: str
    sql_explanation: str

class Input(TypedDict):
    user_query: str

class Output(TypedDict):
    sql_query: str
    sql_explanation: str

generate_prompt = SystemMessage(
    """You are a helpful data analyst who generates SQL queries for users based 
 on their questions."""
)

def generate_sql(state: State) -> State:
    user_message = HumanMessage(state["user_query"])
    messages = [generate_prompt, *state["messages"], user_message]
    res = model_low_temp.invoke(messages)
    return {
        "sql_query": res.content,
        # update conversation history
        "messages": [user_message, res],
    }

explain_prompt = SystemMessage(
    "You are a helpful data analyst who explains SQL queries to users."
)

def explain_sql(state: State) -> State:
    messages = [
        explain_prompt,
        # contains user's query and SQL query from prev step
        *state["messages"],
    ]
    res = model_high_temp.invoke(messages)
    return {
        "sql_explanation": res.content,
        # update conversation history
        "messages": res,
    }

builder = StateGraph(State, input=Input, output=Output)
builder.add_node("generate_sql", generate_sql)
builder.add_node("explain_sql", explain_sql)
builder.add_edge(START, "generate_sql")
builder.add_edge("generate_sql", "explain_sql")
builder.add_edge("explain_sql", END)

graph = builder.compile()
```

*JavaScript*
