
FSM 让你能够将任务结构化为稳定、可恢复的步骤。HSM 在此基础上通过减少重复并在工作流变得复杂时集中策略来扩展这一能力。这两种模式都为你的下一步做好准备：将静态的 LLM 预测转化为动态的、有状态的代理。

## 基础：从静态模型到动态代理

正如你在前一节中看到的，添加状态和控制流对于自动化至关重要。同样的原理也适用于代理。要从静态的 LLM 使用转向代理系统，你需要显式的状态和控制流。状态将一次性的工具调用转变为具有记忆的过程。控制流添加守卫和转换，以决定何时规划、何时行动以及何时停止。

将此添加到 LLM 的原因很简单：LLM 是静态预测器。它们根据提示生成下一个 token，仅利用训练数据中编码的模式。这使得它们成为强大的推理者，但仅限于它们已知的内容。它们无法更新知识、验证声明或与外部环境交互。

*代理性* 在推理与行动相结合时出现。*推理* 是规划或决定下一步的内部过程。*行动* 通过调用工具、检索数据、执行代码或调用外部服务来扩展。它们共同形成一个迭代循环：推理、行动、接收反馈、调整。这个循环是代理行为的基础。

表 1-1 对比了静态 LLM 与代理系统。这些对比不仅说明了为什么需要代理，还展示了它们的设计如何将传统机器学习工作流转变为动态的、增强工具的系统。

**表 1-1. 静态 LLM 与代理系统的对比**

| 静态 LLM（传统用法） | 代理系统 |
| --- | --- |
| 单次传递的 token 生成器 | 迭代推理–行动–反馈系统 |
| 局限于训练数据 | 可以检索、验证和更新知识 |
| 无记忆，无状态 | 有状态，具备短期和长期记忆 |
| 线性提示–响应 | 迭代和自适应工作流 |
| 无工具使用 | 增强工具（检索、代码、API） |
| 对提示的固定解释 | 可以动态细化或重新解释目标 |
| 无外部验证 | 主动检查、纠正和改进输出 |

图 1-3 说明了模块如何将静态 LLM 转化为动态智能体。用户请求进入核心，规划与记忆塑造其推理，工具则使其能够采取具体行动。推理、行动和适应的循环正是由这些元素的协调而涌现。

![智能体核心](img/ch01_agent_core.png)

**图 1-3.** 基准智能体架构：单独的 LLM 只能预测文本，但借助规划、记忆和工具，它就会变成可用的智能体系统。

规划模块引导智能体的推理，从简单的思维链^(1)轨迹到更高级的方法，如思维树^(2)或自我批判。记忆模块提供连续性，使智能体能够回忆过去的步骤、复用先验知识或在会话之间持久化信息。工具将智能体与外部环境连接起来，无论是通过搜索、检索、代码执行还是自定义 API。

此外，传统 LLM 用法与智能体系统之间的一个关键区别在于工作流是无状态还是有状态。*无状态调用*将模型视为黑盒：它接收提示词，预测下一个词元，并返回结果。不会记住任何内容，不会采取任何操作，也不存在反馈循环。示例 1-1 演示了一个无状态 LLM 调用。

### 示例 1-1. 使用 LangChain 的无状态 LLM 调用

```
llm = ChatOpenAI(model="gpt-5-mini")
response = llm.invoke("What are AI agents?")
print(response.content)
```

此外，当你需要训练数据之外的事实，或者希望模型采取行动（搜索、计算）时，就需要引入工具。

下面的代码（示例 1-2 至示例 1-4）演示了一个极简的无状态工具使用运行流程：先把两个工具绑定到 LLM，然后运行一个微型循环来执行模型为完成任务所需的任何工具调用。在此之前，你需要先定义工具。示例 1-2 给出了工具的实现。

### 示例 1-2. 为无状态运行定义工具

```
@tool("internet_search")
def internet_search(query: str) -> str:
    """Search Google via SerpAPI for up to date information."""
    serp_api_key = os.environ["SERPAPI_API_KEY"]
    params = {"engine": "google", "gl": "us", "hl": "en"}
    search = SerpAPIWrapper(params=params, serpapi_api_key=serp_api_key)
    return search.run(query)

@tool("calculator")
def calculator(expression: str) -> str:
    """Evaluate a single line mathematical expression with numexpr."""
    local_dict = {"pi": math.pi, "e": math.e}
    out = numexpr.evaluate(
        expression.strip(),
        global_dict={},
        local_dict=local_dict,
    )
    return str(out)

tools = [internet_search, calculator]
tool_map: Dict[str, Any] = {t.name: t for t in tools}
```

1. LangChain 的`@tool`装饰器使用函数名、类型签名和 docstring 来构建模型所见的 schema。
2. docstring 是必需的：如果缺失，LangChain 在构造工具 schema 或将工具绑定到模型时会抛出错误。
3. 约束计算：拒绝全局变量，只暴露你希望暴露的常量。这保证了计算器的确定性和安全性。

使用 LangChain 的`@tool`装饰器时，函数签名和 docstring 用于构建模型所见的 schema。你需要让这些内容精确、一致，并尽可能具有解释性，以便高效完成任务。docstring 充当工具描述，是 LangChain 的硬性要求，如果缺失就会出现错误。此外，如果 docstring 含糊不清，模型更有可能产生无效调用。

### 示例 1-3. 将工具绑定到 LLM

```
llm = ChatOpenAI(model="gpt-4o", temperature=0,
                max_tokens=800).bind_tools(tools, tool_choice="auto")
```

1. 将工具绑定到模型。`tool_choice="auto"`让模型决定是否调用工具以及调用哪个工具。

## 现代 LLM 是为智能体任务而打造的

现代 LLM 不仅能够调用工具，而且日益专门针对智能体任务进行训练和优化。例如，[Kimi K2](https://moonshotai.github.io/Kimi-K2/)和[Llama 4](https://www.llama.com/docs/model-cards-and-prompt-formats/llama4/)等模型就明确针对编码、工具使用和驱动智能体系统进行了调优。这些模型不止于在聊天窗口中回答问题。它们经过指令调优而能够采取行动，调用 API、运行代码或检索数据已成为其核心设计的一部分。

绑定工具后，你就可以运行一个极简的无状态工具循环。示例 1-4 中的代码会执行模型请求的任何工具调用，将观察结果反馈回模型，并强制进行一次简短的收尾，以确保即使在最后一步以工具观察结束时，也能返回最终答案。

### 示例 1-4. 带工具循环的无状态单次运行

```
def run_once(prompt: str, max_steps: int = 8) -> str:
    messages: List[HumanMessage | AIMessage | ToolMessage] = [
    HumanMessage(content=prompt)]
    last_ai: AIMessage | None = None

    for _ in range(max_steps):
        ai: AIMessage = llm.invoke(messages)
        messages.append(ai)
        last_ai = ai

        calls = getattr(ai, "tool_calls", None) or []
        if not calls:
            return messages[-1].content

        for call in calls:
            name = call["name"]
            args = call.get("args", {}) or {}
            result = tool_map[name].invoke(args)
            messages.append(ToolMessage(
                content=str(result),
                name=name,
                tool_call_id=call.get("id")
            ))

    messages.append(HumanMessage(content="""
Finish now. Give a short final answer in this exact format:

Current temperature:
Square of current temperature:
""".strip()))
    final_ai: AIMessage = llm.invoke(messages)
    return final_ai.content

print(run_once("""Two step task.

Step 1: Use internet_search to get the current air temperature in New York City
today. Show the exact query you used, the top source title and snippet, and
extract a numeric temperature in Celsius. Return this temperature as feedback
for Step 2.

Step 2: Using the Celsius value from Step 1, compute its square with calculator.
Show the exact expression you used and the numeric result.

Important: Give a short final answer in this format:
Current temperature:
Square of current temperature:"""))
```

1. 仅用于本次运行的本地短期记忆。这使工作流在不同运行之间保持无状态。
2. 一个简单的步数上限可防止失控循环：LLMs 不断循环而不终止。
3. 一次模型步骤。助手可以选择调用工具。
4. 读取结构化工具调用。
5. 将观察结果与匹配的`tool_call_id`一起附加，以便模型将结果与调用对应起来。

工具循环可以以一次观察结果作为结束。这条最终指令确保以你要求的格式生成一条收尾的助手消息。

## 为 LLM 赋予状态

将 LLM 转变为智能体的第一步是让它具备状态。为你的 LLM 提供一个状态，记录它处于工作流中的哪个位置。基于输入和逻辑，它会转换到下一个状态，而每个状态都关联着一个函数或行为。在 LangGraph 中，这可以是任何 Python 类型，但通常是 `TypedDict` 或 [Pydantic](https://docs.pydantic.dev/latest/api/base_model/) `BaseModel`。下面概述了 LangGraph 用于构建有状态、可适应的 AI 智能体的关键概念。

`State`

: 一种共享数据结构，表示应用的当前快照。它可以是任何 Python 类型，但通常是 `TypedDict` 或 `Pydantic BaseModel`。

`Nodes`

: 对智能体逻辑进行编码的函数。它们接收当前状态，执行计算或副作用，并返回更新后的状态。

`Edges`

: 根据当前状态选择下一个节点的规则。它们可以是条件分支或固定转换，并且还应当检测是否满足结束条件。

`Command`

: 一个将控制流与状态更新相结合、以支持多智能体通信的对象。节点可以通过返回 `Command` 来同时更新状态并选择下一个节点。

到目前为止，你之前创建的工具仅被用于*无状态循环*中。每次运行都是独立的：模型可以调用工具、获取答案并收尾，但一旦运行结束，所有上下文都会丢失。现在，如果你想结合 LangGraph 的概念来构建有状态的智能体，就需要将工具嵌入到有状态的工作流中。不要在每一轮之后丢弃消息，而是将它们保存在结构化的状态中。这使得系统具备了智能体特性：

- LLM 节点推理下一步骤。
- 工具节点执行外部操作。
- 边根据是否请求了工具来路由控制流。
- 记忆检查点持久化对话，以便智能体可以在后续轮次中复用之前的结果。

这将工具的使用从一次性的辅助手段转变为持续的推理–行动–适应循环。在接下来的代码清单（示例 1-5 – 示例 1-13）中，你将使用 LangGraph 构建这样一个有状态的智能体，从 示例 1-3 - 示例 1-4 中引入的相同工具和 `llm` 绑定开始。

示例 1-5 展示了如何在 LangGraph 中实现一个状态对象来承载对话并协调控制流。你可以复用 示例 1-3 中的工具和已绑定的 `llm`，只需添加新的部分：一个类型化状态、一个追加助手消息的 LLM 节点、一个执行调用的工具节点，以及一个根据工具调用决定继续还是停止的路由器。

### 示例 1-5. 使用状态、节点和路由构建一个最小的 LangGraph

```python
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def llm_node(state: AgentState) -> AgentState:
    ai = llm.invoke(state["messages"])
    return {"messages": [ai]}

tool_node = ToolNode(tools=tools)

graph = StateGraph(AgentState)
graph.add_node("llm", llm_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "llm")

def route(state: AgentState):
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []
    return "tools" if calls else END

graph.add_conditional_edges("llm", route, {"tools": "tools", END: END})
graph.add_edge("tools", "llm")
```

- ![1](img/#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO8-1): 状态以消息列表的形式承载对话。`add_messages` 负责在节点更新之间安全地合并消息。
- ![2](img/#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO8-2): 核心推理节点。它接收当前状态并返回一条新的助手消息。
- ![3](img/#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO8-3): 用于执行助手发出的工具调用的预构建节点。
- ![4](img/#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO8-4): 节点和边的图容器。
- ![5](img/#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO8-5): 路由器。如果最后一轮助手请求了工具，则转到工具节点，否则结束。

定义好图之后，你需要让它可运行且可持久化。示例 1-6 使用内存检查点编译该图，并分配一个线程 ID。检查点会在每一轮恢复之前的消息，以便智能体可以复用之前的结果。线程 ID 将各个分支分隔开，这样你就可以并行运行多个对话而互不干扰。

### 示例 1-6. 使用内存检查点编译并配置线程

```python
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

cfg = {"configurable": {"thread_id": "nyc-weather-session"}}
```

- ![1](img/#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO9-1): 轻量级的按线程检查点机制，可在各轮之间持久化状态。
- ![2](img/#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO9-2): 生成一个知道如何保存和加载状态的可运行应用。
