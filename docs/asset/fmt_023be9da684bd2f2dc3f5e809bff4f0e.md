
```py
from langgraph.checkpoint.memory import MemorySaver

graph = builder.compile(checkpointer=MemorySaver())
```

*JavaScript*

```py
import {MemorySaver} from '@langchain/langgraph'

graph = builder.compile({ checkpointer: new MemorySaver() })
```

这返回一个在每一步结束时存储状态的图实例，因此第一次调用之后的每次调用都不会从空白状态开始。每次调用图时，它都会首先使用检查点器来获取最新的保存状态（如果有的话），并将新输入与先前状态结合。然后才执行第一个节点。这对于启用人类在回路模式至关重要，所有这些模式都依赖于图记住先前状态。

第一种模式，`interrupt`，是最简单的控制形式——用户正在查看应用程序产生的流式输出，并在认为合适时手动中断它（参见图 8-3）。状态保存为用户点击中断按钮之前的最后一步完整步骤。从那里，用户可以选择：

+ 从那个点开始恢复，计算将像没有中断一样继续（参见“恢复”）。
+ 将新输入发送到应用程序中（例如，聊天机器人中的新消息），这将取消任何挂起的未来步骤，并开始处理新输入（参见“重启”）。
+ 不做任何事情，并且不会运行其他任何事情。

![中断模式](img/lelc_0803.png)

###### 图 8-3\. `interrupt`模式

让我们看看如何在 LangGraph 中实现这一点：

*Python*

```py
import asyncio

event = asyncio.Event()

input = {
    "messages": [
        HumanMessage("""How old was the 30th president of the United States 
 when he died?""")
    ]
}

config = {"configurable": {"thread_id": "1"}}

async with aclosing(graph.astream(input, config)) as stream:
    async for chunk in stream:
        if event.is_set():
            break
        else:
            ... # do something with the output

# Somewhere else in your application

event.set()
```

*JavaScript*

```py
const controller = new AbortController()

const input = {
  "messages": [
    new HumanMessage(`How old was the 30th president of the United States when 
 he died?`)
  ]
}

const config = {"configurable": {"thread_id": "1"}}

try {
  const output = await graph.stream(input, {
    ...config,
    signal: controller.signal
  });
  for await (const chunk of output) {
    console.log(chunk); // do something with the output
  }
} catch (e) {
  console.log(e);
}

// Somewhere else in your application
controller.abort()
```

这利用了一个事件或信号，因此你可以从运行的应用程序外部控制中断。注意在 Python 代码块中使用了`aclosing`；这确保在中断时流被正确关闭。注意在 JS 中使用了`try-catch`语句，因为中断运行将引发`abort`异常。最后注意，使用检查点时需要传递一个标识符来区分此交互与所有其他交互。

![授权模式](img/lelc_0804.png)

###### 图 8-4\. `authorize`模式

第二种控制模式是`authorize`，用户提前定义，每当特定节点即将被调用时，他们希望应用程序将控制权转交给他们（参见图 8-4）。这通常用于工具确认——在调用任何工具（或特定工具）之前，应用程序将暂停并请求确认，此时用户可以再次：

+ 恢复计算，接受工具调用。
+ 发送一条新消息以引导机器人走向不同的方向，在这种情况下，工具将不会被调用。
+ 不做任何事情。

这里是代码：

*Python*

```py
input = {
    "messages": [
        HumanMessage("""How old was the 30th president of the United States 
 when he died?""")
    ]
}

config = {"configurable": {"thread_id": "1"}}

output = graph.astream(input, config, interrupt_before=['tools'])

async for c in output:
    ... # do something with the output
```

*JavaScript*

```py
const input = {
  "messages": [
    new HumanMessage(`How old was the 30th president of the United States when 
 he died?`)
  ]
}

const config = {"configurable": {"thread_id": "1"}}

const output = await graph.stream(input, {
  ...config,
  interruptBefore: ['tools']
});
for await (const chunk of output) {
  console.log(chunk); // do something with the output
}
```

这将运行图形，直到即将进入名为`tools`的节点，从而给你机会检查当前状态，并决定是否继续。注意，`interrupt_before`是一个列表，其中顺序不重要；如果你传递多个节点名称，它将在进入每个节点之前中断。

### 恢复

从中断的图形（例如使用前两种模式之一时）继续，你只需要用空输入（或在 Python 中为`None`）重新调用图形。这被视为继续处理之前非空输入的信号：

*Python*

```py
config = {"configurable": {"thread_id": "1"}}

output = graph.astream(None, config, interrupt_before=['tools'])

async for c in output:
    ... # do something with the output
```

*JavaScript*

```py
const config = {"configurable": {"thread_id": "1"}}

const output = await graph.stream(null, {
  ...config,
  interruptBefore: ['tools']
});
for await (const chunk of output) {
  console.log(chunk); // do something with the output
}
```

### 重新启动

如果你想要一个中断的图形从第一个节点重新开始，并带有额外的新的输入，你只需要用新的输入调用它：

*Python*

```py
input = {
    "messages": [
        HumanMessage("""How old was the 30th president of the United States 
 when he died?""")
    ]
}

config = {"configurable": {"thread_id": "1"}}

output = graph.astream(input, config)

async for c in output:
    ... # do something with the output
```

*JavaScript*

```py
const input = {
  "messages": [
    new HumanMessage(`How old was the 30th president of the United States when 
 he died?`)
  ]
}

const config = {"configurable": {"thread_id": "1"}}

const output = await graph.stream(input, config);

for await (const chunk of output) {
  console.log(chunk); // do something with the output
}
```

这将保持当前图的状态，将其与新输入合并，并从第一个节点重新开始。

如果您想丢弃当前状态，只需更改`thread_id`，这将从一个空白状态开始新的交互。任何字符串值都是有效的`thread_id`；我们建议使用 UUID（或其他唯一标识符）作为线程 ID。

### 编辑状态

有时您可能想在恢复之前更新图的状态；这可以通过`update_state`方法实现。您通常会首先使用`get_state`检查当前状态。

这就是它的样子：

*Python*

```py
config = {"configurable": {"thread_id": "1"}}

state = graph.get_state(config)

# something you want to add or replace
update = { }

graph.update_state(config, update)
```

*JavaScript*

```py
const config = "configurable": {"thread_id": "1"}

const state = await graph.getState(config)

// something you want to add or replace
const update = { }

await graph.updateState(config, update)
```

这将创建一个包含您更新的新检查点。在此之后，您就可以从这个新点恢复图了。参见“恢复”了解如何操作。

### 分叉

您还可以浏览图经过的所有过去状态的记录，并且可以再次访问其中任何一个，例如，以获取一个替代答案。这在更具有创造性的应用中非常有用，在这些应用中，每次通过图运行都期望产生不同的输出。

让我们看看它看起来像什么：

*Python*

```py
config = {"configurable": {"thread_id": "1"}}

history = [
    state for state in
    graph.get_state_history(config)
]

# replay a past state
graph.invoke(None, history[2].config)
```

*JavaScript*

```py
const config = "configurable": {"thread_id": "1"}

const history = await Array.fromAsync(graph.getStateHistory(config))

// replay a past state
await graph.invoke(null, history[2].config)
```

注意我们在两种语言中如何将历史记录收集到一个列表/数组中；`get_state_history` 返回状态迭代器（以允许懒加载）。从历史方法返回的状态按最近的时间排序，最新的状态排在最前面，最旧的状态排在最后。

人类在回路控制的真实力量来自于以适合您应用程序的任何方式混合它们。

## 多任务 LLM

本节涵盖了处理 LLM 应用并发输入的问题。鉴于 LLM 相当慢，尤其是在生成长输出或链式多步架构（如使用 LangGraph 所能做到的）时，这是一个特别相关的问题。即使 LLM 变得更快，处理并发输入也将继续是一个挑战，因为延迟的改进也将为越来越多的复杂用例打开大门，就像即使是最有效率的人仍然需要面对优先处理时间竞争需求的需要。

让我们来看看选项。

### 拒绝并发输入

在处理先前的输入时接收到的任何输入都被拒绝。这是一种最简单的策略，但可能无法满足所有需求，因为这实际上意味着将并发管理委托给调用者。

### 独立处理

另一个简单的选项是将任何新的输入视为独立的调用，创建一个新的线程（用于记住状态的容器）并在该上下文中生成输出。这显然有一个明显的缺点，即需要将它们展示给用户作为两个独立且不可调和的调用，这并不总是可能或理想的。另一方面，它有一个优点是可扩展到任意大的规模，您几乎肯定会在您的应用中用到这一点。例如，这就是您考虑让聊天机器人“聊天”与两个不同用户并发的问题的方式。

### 排队并发输入

在处理之前的输入时接收到的任何输入都会排队，并在当前输入完成后进行处理。这种策略有一些优点：

+ 它支持接收任意数量的并发请求。
+ 由于我们等待当前输入完成处理，所以新输入在我们开始处理当前输入后几乎立即到达或在我们完成之前到达并不重要；最终结果将是一样的，因为我们将在处理完当前输入之前继续处理。

这种策略也有一些缺点：

+ 处理所有排队输入可能需要一段时间；事实上，如果输入的产生速度超过处理速度，队列可能会无限制地增长。
+ 由于输入在看到上一个响应之前就已经排队，并且在之后没有改变，因此在处理时输入可能已经过时。当新的输入依赖于之前的答案时，这种策略不合适。

### 中断

当接收到新的输入而另一个正在处理时，放弃当前的处理，并使用新的输入重新启动链。这种策略可以根据中断运行保留的内容而有所不同。以下是一些选项：

+ 保留无物。之前的输入被完全遗忘，就像它从未被发送或处理过一样。
+ 保留最后一个完成的步骤。在一个检查点应用程序（在计算过程中存储进度）中，保留最后一个完成的步骤产生的状态，丢弃当前执行步骤的任何挂起状态更新，并在该上下文中处理新的输入。
+ 保留最后一个完成的步骤，以及当前正在进行的步骤。在保存当时产生的任何不完整状态更新时，尝试中断当前步骤。这种方法可能不适用于最简单的架构。
+ 等待当前节点（但不包括任何后续节点）完成，然后保存并中断。

与排队并发输入相比，这个选项有一些优点：

+ 一旦接收到新的输入，就尽快处理，以减少延迟和产生过时输出的可能性。
+ 对于“保留无物”的变体，最终输出不依赖于新输入接收的时间。

但它也有缺点：

+ 实际上，这种策略仍然限制于一次处理一个输入；当接收到新的输入时，任何旧的输入都会被放弃。
+ 为下一次运行保留部分状态更新需要考虑状态的设计；如果不这样做，那么你的应用程序可能最终会处于无效状态。例如，OpenAI 聊天模型要求 AI 消息请求工具调用必须立即由包含工具输出的工具消息跟随。如果你的运行在中间被中断，你要么防御性地清理中间状态，要么冒着无法进一步进展的风险。
+ 最终输出的结果非常敏感于新输入何时接收；新输入将在处理先前输入所取得的（不完整）进展的背景下进行处理。如果您没有相应地设计，这可能会导致脆弱或不稳定的输出。

### 分叉和合并

另一种选择是并行处理新输入，当接收到新输入时，将线程的状态分叉，并在处理完最终状态后合并。这个选项需要设计您的状态，要么在没有冲突的情况下可合并（例如，使用无冲突复制数据类型[CRDTs]或其他冲突解决算法），要么在您能够理解输出或在此线程中发送新输入之前，让用户手动解决冲突。如果满足这两个要求中的任何一个，这很可能是最佳选择。这样，新输入可以及时处理，输出独立于接收时间，并且支持任意数量的并发运行。

其中一些策略在 LangGraph 平台中实现，将在第九章中介绍。

# 摘要

在本章中，我们回到了构建 LLM 应用程序时面临的主要权衡：代理与可靠性。我们了解到，有一些策略可以在不牺牲代理的情况下部分克服不利因素并获得更高的可靠性，反之亦然。

我们首先介绍了结构化输出，这可以提高 LLM 生成文本的可预测性。接下来，我们讨论了从您的应用程序中发出流式/中间输出，这可以使具有高延迟（目前代理的必然副作用）的应用程序更加易于使用。

我们还探讨了各种人机交互的控制方法——即，将一些监督权交还给您的 LLM 应用程序的最终用户——这通常可以在使高代理架构可靠方面起到关键作用。最后，我们讨论了处理应用程序并发输入的问题，这对于具有高延迟的 LLM 应用程序来说是一个特别突出的问题。

在下一章中，您将学习如何将您的 AI 应用程序部署到生产环境中。

^(1) 在金融中，投资组合优化的*有效前沿*；在经济学中，*生产可能性前沿*；在工程中，*帕累托前沿*。
