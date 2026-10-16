# 第 2 章 架构与模式：规划、反应性与多智能体系统

你现在已经知道，AI 智能体并非魔法。它们是工程化系统，其力量来自架构的战略性配置。通过结合推理循环、工具、有状态设计和自适应控制，大语言模型可以超越静态提示，演化为能够实时规划、反应和反思的多智能体系统（MAS）。本章探讨了使这一切成为可能的架构，从思维链、思维树和 ReAct 等结构化推理方法，到监督者智能体和层次智能体等架构范式。在此过程中，你将更深入地理解推理流程，并看到包括审批门、修正和中止在内的“人在回路”机制如何让这些系统保持可靠、对齐且安全。



当你比较一个简单的单步智能体与一个多步智能体如何处理任务时，智能体架构的重要性就显而易见了。单步智能体接收请求，在内部规划一切，一次性执行，并在返回结果时从不重新审视中间步骤。相比之下，多步智能体在迭代循环中交替进行推理和行动，根据部分结果或工作中的新观察调整其计划。正是这种灵活性让智能体能够动态优化输出，并应对来自环境的不确定性或反馈。这种对比如图 2-1 所示，表 2-1 则展示了两种方法在相同任务上逐步展开的过程。

**表 2-1.** 示例：任务“总结文档并创建关键统计数据的柱状图”的单步执行与多步（ReAct）执行

| 步骤 | 单步智能体 | 多步（ReAct）智能体 |
| --- | --- | --- |
| 1 | 只读取一次请求 | 读取请求并对第一个动作形成初步想法（提取统计数据） |
| 2 | 在内部规划整个序列 | 动作：执行提取；观察：将提取的数据存储在内存中 |
| 3 | 一次生成摘要和图表 | 思考：规划下一步动作，从提取的数据生成摘要 |
| 4 | 返回输出，不重新审视中间结果 | 动作：生成摘要；观察：将输出存储在内存中 |
| 5 | 无执行中调整 | 思考：规划最终动作以创建柱状图；动作：创建图表；观察：复审输出 |
| 6 | — | 如果中间结果不完整，则重复循环直到满意，然后返回最终输出 |

虽然表格逐步展示了差异，图 2-1 则对整体工作流进行了并排说明。单步智能体遵循从提示到结果的线性流水线，而多步智能体将推理、行动和记忆嵌入到一个迭代循环中，该循环会持续运行，直到问题解决或计算预算耗尽。

![ch01 single vs multi step](img/ch01_single_vs_multi_step.png)

**图 2-1：** 单步（左）与 ReAct（右）智能体工作流。

## 结构化推理与行动

赋予智能体不同的角色人设（有时称为拟人化）可以锐化它们的推理，并使它们的贡献更加聚焦。一个被设定为有条不紊的战略家的规划智能体，与被设定为评估者的批评智能体，处理问题的方式会不同。当每个智能体都被赋予清晰的角色和思维模式时，它们的推理链往往更加连贯，它们在多智能体系统（MAS）中的交互也更有目的性。这种一致性提升了协作效率，因为每个智能体都知道自己应该贡献哪类知识。

然而，定义角色只是事情的一半。要让智能体更高效，你可能还需要塑造智能体的思考方式。这就是思维链、思维树和 ReAct 等结构化推理方法发挥作用的地方。

### 让 AI 智能体出声思考

*思维链提示*（CoT）^(1)受人类思维过程的启发，该技术先解决中间步骤，再得出最终答案。虽然 CoT 常被作为一种提示技巧来介绍，但在 AI 智能体的语境中，CoT 可以作为额外的推理步骤来引导控制流。通过强制模型生成中间步骤，你可以获得对推理过程更高的透明度。它通常能提高可靠性，并帮助较小的模型执行原本需要更大模型才能完成的任务。这既降低了成本，又能更清晰地洞察智能体如何得出其决策。表 2-2 为你提供了常见场景的提示词思路，你可根据需要进行调整。

**表 2-2.** 示例：按场景划分的思维链提示词

| 场景 | CoT 提示词示例 |
| --- | --- |
| 规划 | “逐步规划，创建一周的博客日程。解释每个选择背后的理由。” |
| 数据分析 | “首先描述数据集。接着突出模式或趋势。最后以清晰的见解总结收尾。” |
| 数学问题 | “将问题分解成小块。逐块求解并给出解释。然后将结果合并为最终答案。” |
| 研究任务 | “先列出核心问题。逐一调查。最后给出综合总结。” |
| 调试 | “先列出可能的原因。逐一测试每个原因。排除不吻合的并解释原因，直到找到修复方案。” |
| 决策制定 | “列出所有可用选项。逐步权衡利弊。选择最符合标准的选项并解释原因。” |

为了更明确说明你可能希望如何将其融入工作流程，让我们看一个用于数据分析的智能体。由于 CoT 智能体的核心在于能够通过一系列逻辑化的中间步骤进行推理，因此这是一个完美的例子。你可以指示智能体一步一步地思考，让它先描述数据集，再突出显示模式或趋势。示例 2-1 展示了一个可用于此类智能体的通用系统提示词。为简洁起见，我省略了其余代码，但你可以在[本书的代码仓库](https://github.com/Nicolepcx/ai-agents-the-definitive-guide)中找到完整的可运行实现。

#### 示例 2-1. CoT 提示词

```
SYSTEM_INSTRUCTIONS = """You are a careful data analysis assistant. Think step by
step and be explicit.
Begin by describing the dataset.
Next highlight patterns or trends.
Conclude with a clear summary of insights.
When you need to compute metrics or create a chart, call the python_repl tool with
code that:
1. Loads the CSV from the dataset path
2. Prints descriptive statistics
3. If a target column exists, creates one bar chart of mean values grouped by the
target and saves it to a file
4. Prints key results
Return FINAL ANSWER only after you have executed your analysis."""
```

现在，要将该提示词与你的智能体一起使用，只需创建一个智能体，如示例 2-2 所示。请注意，为简洁起见，在示例 2-2 中我使用了 LangChain 的 `AgentExecutor` 以及 LangGraph。但在随附的笔记本中，你还会找到更细粒度的 LangGraph 接线，其中模型与工具之间的路由是显式的。

两种方法产生相同的结果。然而，在生产环境中通常更倾向于使用 `AgentExecutor`，因为它能让代码保持简洁，而细粒度图则对调试以及理解消息、工具调用和控制流在底层如何运作很有价值。

#### 示例 2-2. CoT 智能体

```
def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str) -> AgentExecutor:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt + """Work autonomously using the available
            tools."""),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_functions_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        verbose=False,
    )
    return executor
```

当我运行这段代码时，智能体生成了对演示数据集的正确描述、对模式和趋势的准确分析，以及一张条形图和一份见解总结。此外，借助所给代码，你可以检查中间的工具步骤，例如：

```
(AgentActionMessageLog(tool='python_repl', tool_input={'code': "import pandas
as pd\n\ndf = pd.read_csv('demo.csv')\n\ndf.describe()"})
```

或

```
AgentActionMessageLog(tool='python_repl', tool_input={'code': 'df.describe()'})
```

这个示例展示了 CoT 如何让智能体的推理过程可见且可验证。如果你使用该示例中的代码，基本上就是强制智能体将其中间步骤外部化，这样你不仅对结果更有信心，还能就结论如何得出创建一条透明的审计追踪。

## 生成可能性之树

现在设想一个场景：你有一个更复杂的任务，但不想使用推理模型或完整的多人智能体架构。然而，需要一种更具探索性和战略性的方法。在这种情况下，你可以使用*思维树提示*（ToT）^(2)。ToT 将 CoT 推理扩展为分支搜索过程，在决定最佳下一步之前，会探索并评估多条推理路径。

在代码示例（示例 2-3 到 示例 2-5）中，我将展示如何实现 ToT。核心思想是让一个较小的单一 AI 智能体生成多条潜在路径（即“树”），让另一个智能体评估并选择最佳路径，然后再用另一个模型执行所选路径。简化的 ToT 方法在简单的 ReAct 循环（参见“如何让你的智能体采取行动并反思”）与更基于树搜索的算法的完整计算复杂度之间，提供了一种务实的中间方案。下一章将介绍这些算法。

### ToT 作为工程捷径

一种轻量级 ToT 设置——由较小的模型提出分支、较强的模型进行评估——有时可以取代对完整多人智能体架构的需求。这简化了设计，并且当较小的设置已经能提供可靠结果时，避免反复调用大型推理模型，从而节省金钱和时间。

你首先创建一棵“可能性之树”。示例 2-3 中的 `propose_options` 节点是第一步。它不是生成单一线性输出，而是使用一个较小的 LLM 为一篇博客文章生成三种不同的方案。这三个选项代表了智能体可以探索的分支路径，即“思维”。这就是 ToT 的核心原则：超越单链过程，走向多路径探索。

#### 示例 2-3. 思维生成

```
def propose_options(state: BlogState) -> BlogState:
    """Generator (small): propose 3 creative approaches (ToT-style branching)."""
    proposer = gen_llm.with_structured_output(OptionsPayload)
    prompt = (
        "Generate exactly 3 distinct approaches for a developer-focused blog on:\n"
        f"{state['topic']}\n\nFor each option, provide: title, audience, angle, "
        "a 5-bullet outline, and a concise rationale."
    )
    payload: OptionsPayload = proposer.invoke(prompt)
    state["options_json"] = payload.model_dump_json()
    state["messages"].append(AIMessage(content=state["options_json"]))
    return state
```

## 思维树（ToT）

然后，你可以让智能体进行反思和选择（示例 2-4），从而对树进行剪枝。在这里，一个更强大的 LLM 对三个提议的选项进行评估。可以指示模型根据清晰度和原创性等标准对每个选项进行评判，然后选择最佳的一个。这一步在 ToT 中至关重要，因为它确保智能体不会简单地采用第一个想法，而是批判性地评估备选方案，并选择最有希望推进的路径。

### 示例 2-4. 剪枝节点

```python
def reflect_and_select(state: BlogState) -> BlogState:
    """Judge (stronger): critique options and select best one."""
    chooser = judge_llm.with_structured_output(ChoicePayload)
    eval_prompt = (
        "Evaluate the 3 approaches for clarity, originality, developer relevance, "
        "and feasibility under time constraints. Pick ONE by index 0..2 and justify."
        f"Options JSON:\n{state['options_json']}"
    )
    choice: ChoicePayload = chooser.invoke(eval_prompt)
    opts = json.loads(state["options_json"])["options"]
    idx = max(0, min(choice.choice_index, len(opts)-1))
    choice.choice_index = idx
    state["choice_json"] = choice.model_dump_json()
    state["messages"].append(AIMessage(content=state["choice_json"]))
    return state
```

和之前一样，为了简洁，我省略了其他节点的其余代码，但你可以在[书籍仓库](https://github.com/Nicolepcx/ai-agents-the-definitive-guide)中找到完整的实现。这里我想展示的是如何在 LangGraph 中将所有内容组合在一起，以及我建议使用哪些类型的 LLM。这里的一个重要范式是交接。*交接*定义了控制如何在智能体之间转移：谁接下来发言，谁接下来行动，以及在什么条件下。在 LangGraph 中，节点对智能体的逻辑进行编码，而边代表交接。通过将多个节点连接在一起，你实际上可以在单个图内构建一个 MAS。共享状态是记忆和上下文，所有节点都在其上操作。这种连接在 ToT 中的实际形态见示例 2-5。

### 示例 2-5. 为 ToT 创建图

```python
graph = StateGraph(BlogState)
graph.add_node("propose", propose_options) 
graph.add_node("reflect", reflect_and_select) 
graph.add_node("research", research_with_tools) 
graph.add_node("draft", draft_outline_and_intro) 

graph.add_edge(START, "propose")
graph.add_edge("propose", "reflect")
graph.add_edge("reflect", "research")
graph.add_edge("research", "draft")
graph.add_edge("draft", END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)
```

![1](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO1-1): 小型创意生成 LLM。

![2](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO1-2): 更强的评判/反思 LLM。

![3](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO1-3): 使用工具的研究者。

![4](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO1-4): 小型写作模型。

当你运行代码时，你会看到不同的主题被提出，其中包括参考链接。为了进一步优化结果，你可以实现一个“人在回路”（“设计人在回路工作流”）交互，以审查和批准操作。

人们很容易将 CoT 和 ToT 仅仅视为巧妙的提示技巧，但你已经看到它们也可以成为强大的工程工具。图 2-2 对这两种方法进行了对比：左边是使用 CoT 推理的单一智能体，右边是不同角色协作探索、反思并完善以模仿 ToT 的多智能体设置。

![ch02 cot tot examples compared](img/ch02_cot_tot_examples_compared.png)

**图 2-2.** 对比 CoT 和 ToT 的控制流图。

它们共同展示了结构化推理如何从中间步骤的线性链延伸到探索与选择的分支过程。两者都强调，当智能体被引导分阶段思考时，无论是作为单一声音还是作为协调团队，其结果都更加深思熟虑和可靠。

## 如何让你的智能体采取行动并反思

在上一节中，你已经看到 CoT 和 ToT 对推理是有帮助的，但仅靠推理是不够的。一个只进行推理的智能体永远停留在纸面上，而一个只行动的智能体无法从自己的决定中学习。要将两者结合起来，你需要使用 ReAct，它代表推理与行动。



从数学上讲，你可以把这看作是将上下文`c[t]`映射到动作空间`Â = A ∪ L`中的动作`a[t]`。这里`A`是影响环境的特定任务动作集合，而`L`是基于语言的推理痕迹或思想集合。因此，每一步动作不仅产生任务动作`a[t]`，还产生一条推理痕迹`â[t]`。

这些痕迹被折回智能体的状态中，因此上下文演变为：

`c[t + 1] = (c[t], â[t])`

此外，你遵循以下策略：

`π(a[t]|c[t])`

这里，`c[t] = (o[1], a[1], ..., o[t − 1], a[t − 1], o[t])`。换句话说，智能体不仅仅把已经做过的事情带向未来，也把它选择该路径的原因带向未来。这种递归结构正是反思得以实现的基础：智能体未来的决策既参考其历史行动，也参考其积累的推理。图 2-3 中的流程图展示了在 LangGraph 中实现的最小智能体循环。

![CH 02 ReAct](img/CH_02_ReAct.png)

**图 2-3.** ReAct 智能体的控制流图。该循环是策略`π(a[t]|c[t])`的操作形式。

该过程从`__start__`开始，将控制权交给智能体，然后根据结果进行分支：

- 如果智能体决定已经得出结论，它会发出一个结束信号并移动到`__end__`。
- 如果需要更多推理或行动，智能体会发出继续信号，并可能调用工具来收集更多信息或执行操作。
- 与工具交互后，控制权流回智能体，使其能够在决定继续还是停止之前整合结果并进行反思。

这个循环就是 ReAct 的本质：在迭代循环中结合推理、行动与反思。ReAct 不仅仅是“模型先思考，再行动”。推理痕迹本身会成为不断演化状态的一部分。也就是说，ReAct 智能体在推理痕迹和工具输出之间交替切换，而这些痕迹会反馈回上下文。

虽然这个例子是在单一智能体的背景下，但同样的结构可以自然地扩展到更大的 MAS。每个智能体都可以维持自己的推理和行动循环，并且智能体之间共享或交换推理痕迹。这使得 ReAct 不仅是为个体决策服务的工具，也成为更复杂的智能体场景中协作与反思的基础。

让我们把数学和图表落到实处，看看它们如何真正映射到代码。下面的代码清单将 ReAct 方程转化为代码。和之前一样，注意我这里只展示重要的部分，但完整可运行的实现将在随附的笔记本中。

在进入代码清单之前，请记住其意图：我们携带不断增长的对话状态`c[t]`，同时模型在语言痕迹`L`和任务动作`A`之间交替。每一步都会将想法、动作或工具观察结果追加回状态。代码就是这个循环的镜像。因此，你要先为图创建状态，如示例 2-6 所示。

### 示例 2-6. 创建 ReAct 状态

```python
class AgentState(TypedDict): 
    messages: Annotated[Sequence[BaseMessage], add_messages] 
```

![1](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO2-1) ：代理的状态。

![2](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO2-2) ：上下文`c[t]`是消息流。

现在，你通过示例 2-7 赋予模型两种能力。它可以生成作为思想`L`的语言标记，也可以请求结构化的工具调用`A`。绑定工具使得这些动作成为一流输出，供运行时调度。

### 示例 2-7. 初始化 LLM

```
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
model = model.bind_tools(TOOLS)
```

![1](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO3-1) ：将工具绑定到模型上，将动作空间扩展为`Â = A ∪ L`。

示例 2-8 是策略步骤。它读取完整的消息历史，应用系统规则，并产生唯一结果：要么是思想，要么是工具调用请求，要么两者兼有。你不在此处分支，只需返回模型决定的结果。

### 示例 2-8. 调用模型

```
def call_model(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    response = model.invoke([SYSTEM_PROMPT] + state["messages"], config)
    return {"messages": [response]}
```

![1](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO4-1) ：策略`π(c[t]) → a[t]`是对当前上下文执行`model.invoke`。

![2](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO4-2) ：归约器追加新消息，实现`c[t + 1] = (c[t], a[t])`。

在示例 2-9 中，你将意图转化为效果。每个请求的工具调用都会确定性执行，其结果被封装为一条消息。通过将观察结果推入同一消息流，你让下一步策略能够在没有特殊处理的情况下基于新证据进行推理。

### 示例 2-9. 执行工具调用

```
def tool_node(state: AgentState) -> Dict[str, Any]:

    last = state["messages"][-1]
    outputs: list[ToolMessage] = []
    for tc in getattr(last, "tool_calls", []) or []:
        name = tc.get("name")
        args = tc.get("args") or {}
        tool_fn = TOOLS_BY_NAME.get(name)
        if not tool_fn:
            result = json.dumps({"error": f"Unknown tool {name}"})
        else:
            result = tool_fn.invoke(args)
        outputs.append(
            ToolMessage(
                content=result,
                name=name,
                tool_call_id=tc.get("id"),
            )
        )
    return {"messages": outputs}
```

![1](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO5-1) ：如果模型选择了动作`a[t] ∈ A`，则执行该工具。

![2](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO5-2) ：工具输出是通知下一步的观测。

示例 2-10 中的停止规则很简单。如果模型请求了工具，它必须先完成该效果才能再次请求。如果没有，则假设它产生了最终答案。这使循环可预测，并避免孤儿动作。

### 示例 2-10. 停止条件

```
def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "continue"
    return "end"
```

![1](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO6-1) ：停止测试读取最后一条 AI 消息来决定`continue`或`end`。

现在，你将各组件连接成一个小型状态机，如示例 2-11 所示。边编码了镜像 ReAct 图表的控制流。策略节点先运行。如果它发出动作，则路由到工具节点，然后返回策略。如果它只发出思想或最终答案，则终止。

### 示例 2-11. 构建工作流图

```
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "tools", "end": END},
)

workflow.add_edge("tools", "agent")
graph = workflow.compile()
```

在运行时，图一次推进一条边。条件边决定下一跳是`tools`还是`END`。从`tools`回到`agent`的固定边完成了思考-行动循环。

为了演示运行过程，我提了一个问题：苏黎世今天天气如何。LLM 先决定调用哪个工具来最好地回答，然后工具返回证据，模型根据证据写出最终答案。由于每一步都记录在消息流中，推理过程可检索：

````
> 进入新的 ReAct 轨迹...
问题：苏黎世今天天气如何？

最终答案：苏黎世今天天气如何？

> 轨迹结束。

推理摘要：根据查询选择工具
动作：
```

{

  "action": "internet_search",

  "action_input": {

    "query": "Zurich weather today"

  }

}

```

观察：
```

{

  "title": "Zurich, Zurich, Switzerland Weather Forecast",

  "snippet": "Hourly Weather \u00b7 1 PM 74\u00b0. rain drop 0% \u00b7 2 PM

  80\u00b0. rain drop 0% \u00b7 3 PM 82\u00b0. rain drop 0% \u00b7 4 PM

  83\u00b0. rain drop 0% \u00b7 5 PM 84\u00b0. rain drop 0% \u00b7 6 PM ...

```

最终答案：今天苏黎世天气以晴为主，最高气温 82°F（约 28°C），最低气温 62°F（约 17°C）。西南风轻微，风速 6 英里/小时。

更多详情，可查看完整预报
[此处](https://www.accuweather.com/en/ch/zurich/316622/weather-forecast/316622)。

> 轨迹结束。
````

现在你明白 ReAct 为何如此强大了。它不仅是代理在思考和行动之间交替，更在于推理本身成为状态的一部分。这种递归过程使代理能够适应、改进策略并解释其选择。在实践中，它甚至能让相对较小的模型更可靠地完成复杂任务，同时使其决策过程透明化。

总而言之，CoT、ToT 和 ReAct 展示了不同的推理策略如何塑造代理的智能。每种方法都有优势：CoT 提供透明度和可靠性，ToT 在提交前探索多种替代方案，而 ReAct 通过结合推理和工具使用来闭环。表 2-3 提供了这些推理模式的汇总，以及何时最有效的提示。

表 2-3. 模式总结：结构化推理方法（CoT、ToT、ReAct）

| 方法 | 应用场景 | 优势 | 权衡 |
| --- | --- | --- | --- |
| 思维链（CoT） | 数学、规划、分析中的逐步推理 | 提高可靠性和透明度；使较小的模型也能处理更难的任务 | 输出较慢、冗长；有时冗余 |
| 思维树（ToT） | 具有多条解决方案路径的探索性任务（创意写作、策略搜索） | 探索多种替代方案；减少“首答偏见” | 计算成本更高；需要编排（选择和剪枝） |
| ReAct | 需要推理和工具使用的任务（研究、编程、API） | 融合推理与行动；创建审计追踪；自适应反馈 | 更复杂的状态管理；过度调用工具的风险 |

这凸显了这些方法一旦整合到整体智能体工作流中，如何成为强大的构建模块。

## 你肯定是在开玩笑，AI 智能体！

有时，智能体的行为会很有趣。它可能调用错误的工具、无休止地循环，或者产生古怪而富有创意的回应。在那些时刻，你可能会笑着对 AI 智能体说：“你在开玩笑吧？”但当同样的不可预测性波及访问控制、财务操作或客户数据时，这就不再是玩笑了。在错误的时间或在错误的系统上执行操作可能会带来严重后果。这就是*人在回路*（HITL）存在的原因。HITL 允许智能体在关键节点暂停，请求你的批准，只有在确保安全后才继续执行。

HITL 不仅是为了在错误发生前将其捕获。它还提供了问责制和透明性。通过在智能体的工作流中插入检查点，你可以确保不可逆的操作经过审查、决策可追溯，并且敏感操作始终处于人类控制之下。这种监督非但不会拖慢系统，反而会让智能体更值得信赖，也更容易在生产环境中采用。

### 设计人在回路工作流

在使用 HITL 工作流时，你可以执行几种不同的中断操作。最常见的有以下几种。

- **批准或拒绝**：在关键步骤（如 API 调用）之前暂停图，以审查并批准该操作。如果操作被拒绝，你可以阻止图执行该步骤，并可能采取替代操作。这种模式通常涉及根据人工输入对图进行路由。
- **审查并编辑**：暂停图以审查和编辑图的状态。这对于纠正错误或用附加信息更新状态很有用。
- **审查工具调用**：如果工具调用可能影响外部系统，请在执行前检查其参数。例如，在预订旅行之前，你可以确认预订详情。
- **验证人工输入**：在图的特定步骤明确请求人工输入。这对于收集附加信息或上下文以支持智能体的决策过程，或支持多轮对话非常有用。

LangGraph 提供了[详尽的文档](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/add-human-in-the-loop/)，介绍如何在你的应用程序中应用这些模式。此外，[本书的代码仓库](https://github.com/Nicolepcx/ai-agents-the-definitive-guide)包含了六种不同模式的完整可运行示例。你可以通过一个`main()`函数在它们之间进行选择，如示例 2-12 所示。

#### 示例 2-12. HITL 模式演示的主函数

```
def main():
    menu = """
Pick a demo
1. Human feedback loop for writing
2. Approval gate before API call
3. Review and edit state
4. Parallel interrupts with resume map
5. Tool call review in a tiny ReAct loop
6. Static interrupts for debugging
q. Quit
> """
```

在接下来的代码示例（示例 2-13 - 示例 2-20）中，我将解释如何实现上面涉及的模式，并介绍在实现时需要注意的要点。

### LangGraph 中的 Command

`Command`结合了控制流（边）和状态更新（节点）。它促进了多参与者（或多智能体）之间的通信。例如，一个节点可以更新其状态并决定下一个要访问的节点。LangGraph 通过从节点函数返回`Command`对象来实现这一点。

当你想要对类似 HTTP 调用这样具有副作用的步骤设置门控时，可以使用批准或拒绝模式。

示例 2-13 让 LLM 生成请求计划，并将其存储在`proposed_request`中。

#### 示例 2-13. 使用 LLM 提出请求

```
def b_propose(state: BState) -> BState:
    prompt = "Return only JSON with keys url and params for GET to
              https://httpbin.org/get using q and limit."
    text = LLM.invoke(prompt).content 
    m = re.search(r"\{.*\}", text, re.S)
    data = {"url": "https://httpbin.org/get", "params":
           {"q": "fallback", "limit": 1}} 
    if m:
        try:
            data = json.loads(m.group(0)) 
        except Exception:
            pass
    return {"proposed_request": data} 
```



# 架构与模式：规划、反应性与多智能体系统

![1](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO7-1)：让 LLM 生成结构化计划。

![2](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO7-2)：如果解析失败，则回退到默认请求。

![3](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO7-3)：尝试从模型输出中解析 JSON。

![4](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO7-4)：将结果存储在状态的 `proposed_request` 下。

## 示例 2-14 调用 `interrupt(…)` 来暂停并请求人工批准或修改。

### 示例 2-14. 请求的批准门控

```python
def b_gate(state: BState) -> Command[Literal["b_call", "b_propose"]]:
    v = interrupt({ 
        "question": "Approve or revise request",
        "proposed_request": state["proposed_request"], 
        "schema": {
            "type": "object",
            "properties": {
                "action": {"enum": ["approve", "revise"]}, 
                "update": {"type": "object"}
            },
            "required": ["action"]
        }
    })
    action = v.get("action")
    if action == "approve":
        return Command(goto="b_call", update={"decision": "approved"}) 
    upd = v.get("update") or {}
    new_req = state["proposed_request"].copy()
    if "url" in upd:
        new_req["url"] = upd["url"]
    if isinstance(upd.get("params"), dict):
        new_req.setdefault("params", {}).update(upd["params"])
    return Command(goto="b_gate", update={"proposed_request": new_req,
    "decision": "revised"}) 
```

![1](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO8-1)：暂停执行并等待人工决策。

![2](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO8-2)：展示拟定的请求以供检查。

![3](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO8-3)：强制使用结构化的批准/修改模式。

![4](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO8-4)：批准后，路由到调用步骤。

![5](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO8-5)：修改后，使用更新后的请求循环回到门控。

在 `resume` 时，你可以传递 `Command(resume={"action": "approve"})` 来继续，或传递 `{"action": "revise", "update": {...}}` 来合并修改并循环回到门控。

## 示例 2-15 仅在获得批准后才执行 HTTP 调用，并将结果存储在 `api_result` 中。你可以拒绝以阻止执行，路由则取决于人工输入。

### 示例 2-15. 执行已批准的 HTTP 调用

```python
def b_call(state: BState) -> BState:
    import requests
    r = requests.get(state["proposed_request"]["url"],
    params=state["proposed_request"]["params"], timeout=10) 
    return {"api_result": {"status_code": r.status_code, "url": r.url}} 
```

![1](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO9-1)：仅在批准后执行外部请求。

![2](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO9-2)：将结果存储在状态的 `api_result` 下。

## 你可以使用“审查并编辑”模式，通过直接写入状态来纠正或改进模型输出。这里，示例 2-16 使用 LLM 生成一段简短摘要，并将其存储在 `summary` 中。

### 示例 2-16. 生成摘要文本

```python
def c_write(state: CState) -> CState:
    text = LLM.invoke("Write 2 sentences about why human
                     in the loop matters for agents").content 
    return {"summary": text} 
```

1. ![1](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO10-1): 让 LLM 生成一份候选摘要。

2. ![2](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO10-2): 将 `summary` 存储到状态中，以便稍后审查。

## 示例 2-17 通过 `interrupt(...)` 暂停执行，其中包含一个请求 `edited_text` 的 JSON 模式。

### 示例 2-17. 中断以进行人工编辑

```python
def c_edit(state: CState) -> CState:
    res = interrupt({ 
        "task": "Edit the summary text",
        "summary": state["summary"], 
        "schema": {
            "type": "object",
            "properties": {"edited_text": {"type": "string"}}, 
            "required": ["edited_text"]
        }
    })
    return {"summary": res["edited_text"]} 
```

1. ![1](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO11-1): 暂停并请求人工提供编辑输入。

2. ![2](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO11-2): 提供生成的摘要以供检查。

3. ![3](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO11-3): 将编辑限制为字符串字段。

4. ![4](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO11-4): 用人工修订版本覆盖 `summary`。

## 你可以使用 `Command(resume={"edited_text": "your revision"})` 恢复执行。节点会重新启动，并返回你的编辑内容而不再暂停，然后图会用人工版本更新 `summary`。也就是说，你实际上是暂停下来审查，然后用编辑后的内容覆盖图中的状态。

## 当工具可能影响外部系统且需要检查参数时，你会使用审查工具调用。示例 2-18 将任何工具包装在 `interrupt([...])` 中。中断的有效负载会请求接受、编辑或响应。

### 示例 2-18. 用审查门包装工具

```python
def add_hitl(tool_obj: BaseTool | Any) -> BaseTool:
    if not isinstance(tool_obj, BaseTool):
        tool_obj = tool(tool_obj)

    @tool(tool_obj.name,description=tool_obj.description,
    args_schema=tool_obj.args_schema)
    def wrapped(**tool_input):
        request = [{ 
            "action_request": {"action": tool_obj.name, "args": tool_input}, 
            "config": {"allow_accept": True, "allow_edit":
                       True, "allow_respond": True}, 
            "description": "Review this tool call"
        }]
        response = interrupt(request)[0] 
        if response["type"] == "accept":
            return tool_obj.invoke(tool_input) 
        if response["type"] == "edit":
            new_args = response["args"]["args"]
            return tool_obj.invoke(new_args) 
        if response["type"] == "response":
            return response["args"] 
        raise ValueError("Unsupported interrupt response type")
    return wrapped
```

1. ![1](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO12-1): 将原始工具调用包装在审查请求中。

2. ![2](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO12-2): 传递工具名称和参数以便检查。

3. ![3](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO12-3): 允许人工接受、编辑或响应。

4. ![4](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO12-4): 暂停，直到收到决定。

5. ![5](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO12-5): 如果被接受，则执行原始工具调用。

6. ![6](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO12-6): 使用修订后的参数执行工具。

7. ![7](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO12-7): 用人工响应替换工具输出。

## 你可以使用以下任一方式恢复：

```json
{"type": "accept"}
{"type": "edit", "args": {"args": {"query": "weather in NY"}}}
{"type": "response", "args": "Skip the tool right now"}
```

该包装器要么执行原始工具调用，要么使用编辑后的参数执行，要么返回人工消息而不调用工具。这样一来，你可以在工具调用到达外部世界之前对其进行检查和批准。

### 中断与副作用

切勿将有副作用的代码（API 调用、数据库写入）与 `interrupt(...)` 放在同一个节点中。一定要先暂停，只有在获得明确批准后才执行副作用，以免在恢复时重复执行。

有时候，你可能还想通过一次恢复获得并行输入。示例 2-19 展示了如何实现这一点。

#### 示例 2-19. 并行人工中断

```python
class DState(TypedDict, total=False):
    text_1: str
    text_2: str

def d_h1(state: DState):
    v = interrupt({"text_to_revise": state["text_1"]}) 
    return {"text_1": v} 

def d_h2(state: DState): 
    v = interrupt({"text_to_revise": state["text_2"]})
    return {"text_2": v} 

def build_graph_D():
    g = StateGraph(DState)
    g.add_node("d_h1", d_h1) 
    g.add_node("d_h2", d_h2)
    g.add_edge(START, "d_h1") 
    g.add_edge(START, "d_h2")
    g.add_edge("d_h1", END)
    g.add_edge("d_h2", END)
    return g.compile(checkpointer=CHECKPOINTER) 
```

1. ![1](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO13-1): 中断以获取第一个文本输入。

2. ![2](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO13-2): 使用修订后的 `text_1` 更新状态。

3. ![3](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO13-3): 中断以获取第二个文本输入。

4. ![4](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO13-4): 使用修订后的 `text_2` 更新状态。



5. ![5](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO13-5)：将两个节点都添加到图中。

6. ![6](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO13-6)：两个节点都直接从`START`分支。

7. ![7](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO13-7)：使用`checkpointer`编译，使两个中断都可以恢复。

这里，两个节点都从`START`开始，并且每个节点都调用`interrupt(...)`。你可以使用示例 2-20 构建一个将`interrupt_id`映射到其值的字典，然后使用`Command(resume=that_map)`恢复执行。

#### 示例 2-20. 恢复多个中断

```
def wait_for_interrupt_and_prompt(app, cfg):
    state = app.get_state(cfg)
    ints = getattr(state, "interrupts", []) or []
    if not ints:
        print("No interrupts pending")
        return None

    if len(ints) > 1:
        print("\nMultiple interrupts pending:")
        for i, it in enumerate(ints, 1):
            print(f"[{i}] id={it.interrupt_id} value={jdump(it.value)}")
        print("Enter values per interrupt. Leave blank to echo original.")
        resume_map = {}
        for it in ints:
            val = input(f"Value for {it.interrupt_id}: ").strip()
            if val:
                try:
                    resume_map[it.interrupt_id] = json.loads(val)
                except Exception:
                    resume_map[it.interrupt_id] = val
            else:
                resume_map[it.interrupt_id] = it.value
        return Command(resume=resume_map)
```

- ![1](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO14-1)：从状态中收集所有待处理的中断。

- ![2](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO14-2)：一次处理多个中断。

- ![3](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO14-3)：打印每个中断的 ID 及其当前值。

- ![4](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO14-4)：针对每个中断，向人工请求新值。

- ![5](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO14-5)：尝试将其解析为 JSON，以进行结构化编辑。

- ![6](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO14-6)：如果没有输入，则保留原始值。

- ![7](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO14-7)：使用完整的更新值映射恢复执行。

正如你所看到的，添加人工监督并没有一种唯一“正确”的方法。你可以根据任务的敏感度以及你想要保留的控制程度，从不同的模式中进行选择。我鼓励你在提供的 notebook 中尝试这些模式，因为这样可以更容易地看到人工监督如何从简单的审批门控演变为人类与智能体之间更丰富的协作。

## 高级智能体范式：监督者与层级智能体

如果你的任务开始比你目前见过的更复杂，或者你发现智能体难以胜任，那么将任务分解成更小的子任务会有所帮助。可以把它想象成组建一个团队：每个人都有不同的技能，共同解决更大的挑战。

同理，你可以引入更小、独立的智能体，构成一个多智能体系统（MAS）。一种常见的设置是*监督者架构*，其中每个智能体都向一个决定下一步该由谁执行的监督者汇报。更高级的版本是*层级架构*，你不只有一个监督者，而是有监督者的监督者。这种泛化允许更灵活的控制流。

图 2-4 中的图表展示了单智能体设置与这两种多智能体模式的对比。

![ch02 单一智能体、监督者与层级架构对比](img/ch02_single_vs_supervisor_vs_hierarchical.png)

**图 2-4.** 单一智能体、监督者与层级智能体架构的对比。

使用多智能体系统（MAS）的主要优势在于其模块化、专门化和可控性。通过在各个智能体之间划分功能，你可以让整个系统更易于开发、测试和维护。每个智能体都可以被设计为领域专家，从而提升系统的整体性能。与简单的函数调用不同，多智能体设置还为你提供了对智能体之间通信方式的显式控制。

### 构建层级研究团队

为了将监督者和子智能体的概念具体化，让我们通过一个示例来构建一个层级研究团队。我将使用 LangGraph 预构建的、可复用的`create_react_agent`组件来快速实现 ReAct 智能体。然后，每个工作智能体都会有明确的角色：有些专注于网络搜索，有些负责抓取页面，查询 [Exa](https://exa.ai/) 以获取更结构化的搜索结果，甚至可以在 Google 上查找专利。一个监督者智能体位于顶层，协调这些专家并决定下一步由谁执行。最终目标是组建一个市场研究团队。图 2-5 展示了你正在构建的最终应用。

![ch02 市场研究团队图](img/ch02_market_research_team_graph.png)

**图 2-5.** 采用层级架构的最终应用。

但在那之前，你会从一个更小的起点开始：构建你的第一个监督者设置——研究团队。你可以在图 2-6 中看到你所创建团队的概览。一旦完成这一步，第二步就是添加一个写作团队。这个团队将获取研究团队的输出，并将其作为简单文本文件保存到你的文件系统。

![ch02 研究团队图](img/ch02_research_team_graph.png)

**图 2-6.** 研究团队的流程图。

我将略过如何创建工具的细节，因为你已经在第一章（示例 1-2）中学过这部分内容。

你需要的第一样东西是一个用于创建监督者的辅助函数（示例 2-21）。该函数充当路由器（参见“将 FSM/HSM 映射到智能体框架”），决定下一步应由哪个工作智能体执行，或者流程是否应该结束。

#### 示例 2-21. 监督者的辅助函数

```
class State(MessagesState):
    next: str

def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router."""
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})

    return supervisor_node
```

- ![1](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO15-1)：创建状态以跟踪下一个工作智能体。

- ![2](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO15-2)：定义一个辅助函数，为任意智能体集合构建监督者节点。

- ![3](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO15-3)：提供系统提示，告知 LLM 如何扮演监督者角色。

- ![4](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO15-4)：使用结构化输出，可靠地决定下一步调用哪个工作智能体。

- ![5](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO15-5)：当主管发出`FINISH`指令时，让工作流干净地终止。

接下来，你需要一个辅助函数来创建你的工作节点，其实现方式见示例 2-22。这使得配置简洁且可复用，这样你每次添加新代理时就不必重复相同的样板代码。

#### 示例 2-22. 创建节点的辅助函数

```
def make_react_worker_node(
    *,
    llm: ChatOpenAI,
    name: str,
    tools: list,
    prompt: str | None = None,
    goto: str = "supervisor",
):
    agent = create_react_agent(llm, tools=tools, prompt=prompt) 

    def node(state: State) -> Command[Literal["supervisor"]]:
        result: Dict[str, Any] = agent.invoke(state) 
        msgs: Sequence[BaseMessage] = result.get("messages", [])
        content = getattr(msgs[-1], "content", "") if msgs else ""
        return Command( 
            update={"messages": [HumanMessage(content=content, name=name)]},
            goto=goto,
        )

    return node
```

- ![1](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO16-1)：创建一个带有工具和角色提示的专业 ReAct 代理。

- ![2](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO16-2)：使用当前状态调用代理。

- ![3](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO16-3)：将代理的最新输出传回状态，并返回给主管。

一旦有了这些辅助函数，就可以继续定义你的代理了。为每个代理指定具体角色会很有帮助。由于这里的目标是构建一个市场调研团队，因此角色应非常有针对性。我的角色提示见示例 2-23。

#### 示例 2-23. 为研究代理分配角色

```
SEARCH_PROMPT = """Role: Web researcher. Use the search tool and return a
concise research note with sources. No follow-up questions."""

SCRAPER_PROMPT = """Role: Web scraper. Use the scraping tool to fetch details
from given URLs and summarize key findings. No follow-up questions."""

EXA_PROMPT = """Role: Research assistant. You can search for all recent info
on Exa Search. Your response should clearly articulate the key points you found."""

PATENT_PROMPT = """Role: Market researcher with 20 years of experience.
You are very knowledgeable in patent research and in finding up-to-date info
about patents using the Google Patents API."""
```

现在，让我们创建代理。使用示例 2-22 中的辅助函数，你可以实例化每个工作节点，并为其分配适当的角色和工具。具体实现见示例 2-24。

#### 示例 2-24. 创建研究代理



```
specs = [
    dict(
        name="search",
        tools=[tavily_tool],
        prompt=SEARCH_PROMPT,
    ), 
    dict(
        name="web_scraper",
        tools=[scrape_webpages],
        prompt=SCRAPER_PROMPT,
    ),
    dict(
        name="exa_search",
        tools=[exa_search_tool],
        prompt=EXA_PROMPT,
    ),
    dict(
        name="patent_research",
        tools=[patent_search],
        prompt=PATENT_PROMPT,
    ),
]

nodes = {s["name"]: make_react_worker_node(llm=llm, **s) for s in specs} 

search_node = nodes["search"]
web_scraper_node = nodes["web_scraper"]
exa_search_node = nodes["exa_search"]
patent_research_node = nodes["patent_research"]
```

- ![1](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO17-1): 为每个工作节点定义其工具和角色。

- ![2](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO17-2): 使用辅助函数实例化所有工作节点，无需重复样板代码。

为了完善主管架构，我们需要设置一个能够协调所有四个研究代理的主管。这确保工作流不会停滞，并且每个代理都能在适当的时候发挥作用。具体实现见示例 2-25。

#### 示例 2-25. 创建研究主管节点

```
research_supervisor_node = make_supervisor_node(
    llm, ["search", "web_scraper", "exa_search", "patent_research"]
) 
```

- ![1](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO18-1): 创建一个协调所有四个研究代理的主管。

最后，我们需要通过注册节点和连接边来构建实际的图，这些边定义了节点之间的控制流。这样就得到了一个功能完整的研究团队图，如示例 2-26 所示。

#### 示例 2-26. 构建研究团队图

```
research_builder = StateGraph(State)

research_builder.add_node("supervisor", research_supervisor_node) 
research_builder.add_node("search", search_node)
research_builder.add_node("web_scraper", web_scraper_node)
research_builder.add_node("exa_search", exa_search_node)
research_builder.add_node("patent_research", patent_research_node)

research_builder.add_edge(START, "supervisor") 
research_builder.add_edge("search", "supervisor")
research_builder.add_edge("web_scraper", "supervisor")
research_builder.add_edge("exa_search", "supervisor")
research_builder.add_edge("patent_research", "supervisor")

research_graph = research_builder.compile() 
```

- ![1](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO19-1): 注册主管和工作节点。

- ![2](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO19-2): 连接边，使所有工作节点都路由回主管。

- ![3](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO19-3): 将图编译为可执行的工作流。

要测试此设置，你可以运行示例 2-27 中的代码。

#### 示例 2-27. 使用提示运行研究团队

```
for s in research_graph.stream(
    {"messages": [("user", """What are AI agents? Are there any patents out there
                              about LLM agents?""")]}, 
    {"recursion_limit": 100},
):
    print(s)
    print("---")
```

- ![1](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO20-1): 在查询上运行研究图；主管协调哪些代理执行操作。

这将返回关于 AI 代理的简短描述，然后使用专利工具检查最近的专利。步骤看起来像这样（为简洁起见，我省略了代理之间的实际消息）：

```
{'supervisor': {'next': 'search'}}
{'supervisor': {'next': 'patent_research'}}
{'supervisor': {'next': '__end__'}}
```

你可以用同样的方式设置你的写作团队。本节的笔记本包含完整可运行的代码，但图 2-7 快速展示了写作团队的样貌。

![ch02 写作团队图](img/ch02_writing_team_graph.png)

**图 2-7. 写作团队的流程图。**

这种模块化方法使逐步构建能力越来越强的多智能体系统（MAS）变得容易：首先连接专门的工作节点，然后将整个团队组合成更大的图。在实践中，这意味着你可以从小处着手——比如在研究团队中加入搜索、抓取和专利查询，然后将其接入写作团队，形成完整的端到端流水线，或者通过内部数据以及图表编写和绘图代理进一步扩展。

要从单个团队过渡到合适的层级结构，你需要构建另一个图来将所有内容紧密连接起来。可以把这个图看作是你的“超级图”，如示例 2-28 所示。

#### 示例 2-28. 构建分层架构

```
super_builder = StateGraph(State)
super_builder.add_node("supervisor", teams_supervisor_node)
super_builder.add_node("research_team", call_research_team)
super_builder.add_node("writing_team", call_paper_writing_team)

super_builder.add_edge(START, "supervisor")
super_graph = super_builder.compile()
```

一旦就绪，你就可以按照示例 2-29 所示来提示你的分层 MAS。

#### 示例 2-29. 运行分层 MAS

```
TARGET_FILE = "semiconductor_whitepaper.txt"

TASK_MSG = f"""
Write an 800-word research report white paper on semiconductor development.
Start with an executive summary of your findings.
Search for relevant recent patents and include links to them.
IMPORTANT: Provide links to all your sources.
Finally, save the full report to disk as a .txt file using the write_document tool.
Use file_name="{TARGET_FILE}".
"""

for step in super_graph.stream(
    {
        "messages": [
            ("user", TASK_MSG.strip())
        ],
    },
    {"recursion_limit": 150},
):
    print(step)
    print("---")
```

运行此设置将产生以下监督者决策序列（消息同样省略）：

```
{'supervisor': {'next': 'research_team'}}
{'supervisor': {'next': 'writing_team'}}
{'supervisor': {'next': 'writing_team'}}
{'supervisor': {'next': '__end__'}}
```

运行结束时，报告被成功保存到`semiconductor_whitepaper.txt`。在实践中，此工作流只需大约两分钟即可收集研究资料、生成文本并将所有内容写入文件。

### 开发智能体群

另一个重要范式是群架构。在这里，智能体根据各自的专长动态地相互交接控制权。与严格的层级结构不同，智能体群强调点对点协作。系统会跟踪最后活跃的智能体，以便后续交互与正确的智能体无缝衔接。这种来回交互使工作能够在智能体之间自然传递，例如当研究员收集到资料来源后将其交给写作智能体，而写作智能体在完成综合前可能会请求更多数据。图 2-8 展示了这种连续流程。

![ch02 agent swarm](img/ch02_agent_swarm.png)

#### 图 2-8. 智能体群之间的交接流程

LangGraph 提供了一个[开源库](https://github.com/langchain-ai/langgraph-swarm-py)，用于快速推进智能体群的开发。默认情况下，群中的智能体使用由预构建的`create_handoff_tool`创建的交接工具。你也可以创建自己的自定义交接工具，以更好地适应工作流程。例如，你可以：

*   更改工具名称或描述，使交接目的更清晰。

*   添加供 LLM 填写的工具调用参数，例如给下一个智能体的具体任务描述。

*   控制传递给下一个智能体的数据。默认情况下，`create_handoff_tool`会传递完整的消息历史（所有先前的智能体消息）以及一条确认交接的工具消息，但你可以选择仅发送过滤后的上下文子集。

在搭建完整的智能体群之前，你需要首先定义交接辅助函数，如示例 2-30 所示。这些工具充当信号，让一个智能体将控制权传递给另一个智能体。在下面的示例中，研究助理在收集到足够多的资料来源后可以交接给写作助理，而写作助理在需要更多证据时可以将控制权交回。

#### 示例 2-30. 创建交接工具

```
to_writer = create_handoff_tool( 
    agent_name="writer_assistant",
    description="""Transfer to the writing assistant to synthesize sources into
                an answer.""",
)
to_research = create_handoff_tool( 
    agent_name="research_assistant",
    description="""Return to the research assistant to fetch or scrape more
                sources.""",
)
```

![1](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO21-1)

: 定义一个工具，让研究助理可以交接给写作助理。

![2](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO21-2)

: 定义反向工具，使写作助理可以将控制权交还给研究助理。

一旦交接辅助函数就绪，你就可以继续定义自己的智能体。保持角色非常清晰是很有帮助的。在这种设置中，你希望一个智能体专注于研究，另一个专注于写作。交接工具确保工作流在需要更多资料来源或需要综合时能够来回切换。角色如示例 2-31 所示。

#### 示例 2-31. 为研究和写作助理分配角色

```
research_assistant = create_react_agent( 
    model=llm,
    tools=[tavily_tool, scrape_webpages, to_writer], 
    prompt=(
        """You are a research assistant. Search the web with Tavily.
        When you have 3 to 5 solid sources, scrape key pages for details,
        then hand off to the writer assistant."""
    ),
    name="research_assistant",
)

writer_assistant = create_react_agent( 
    model=llm,
    tools=[to_research], 
    prompt=(
        """You are a writing assistant. Read the provided Documents and messages.
        Synthesize a concise answer with citations by site name in brackets. If
        sources are thin or unclear, hand back to research with a short request."""
    ),
    name="writer_assistant",
)
```

![1](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO22-1)

: 定义研究助理，负责网络搜索和页面抓取。

![2](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO22-2)

: 授予研究助理访问其工具的权限，包括交接给写作助理的工具。

![3](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO22-3)

: 定义写作助理，负责综合和引用。

![4](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO22-4)

: 如果需要更多证据，允许写作助理将控制权交还给研究助理。

既然两个角色都已定义，我们就可以将它们接入一个智能体群。该智能体群提供了协调机制：默认从研究助理开始，但允许交接给写作助理，并在任务完成前不断来回切换。具体实现见示例 2-32。

#### 示例 2-32. 构建研究-写作智能体群

```
swarm = create_swarm(
    agents=[research_assistant, writer_assistant],
    default_active_agent="research_assistant",
).compile()
```

要测试该系统，你可以运行下面的示例提示。在这里，我请求了瑞士金融科技牌照的摘要。智能体群首先搜索资料来源，然后交接给写作助理进行综合，如果有需要，会返回研究助理获取更多数据。提示见示例 2-33。

#### 示例 2-33. 运行研究-写作智能体群



```
user_request = { 
    "messages": [
        {
            "role": "user",
            "content": (
                """Find the current Swiss fintech licensing options for small
                 startups. Gather authoritative sources and produce a short
                 summary with three bullet points and references."""
            ),
        }
    ]
}

for chunk in swarm.stream(user_request): 
    print(chunk)
    print()
```

![1](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO23-1)

: 定义一个用户提示，要求提供瑞士金融科技牌照的详细信息。

![2](img/#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO23-2)

流式传输来自群体（swarm）的结果，展示了实际运行中由交接驱动的工作流。

此次运行生成了三个简洁的要点，并附有引用，引用了瑞士官方监管机构页面和可信的金融来源。然而，如果撰写者发现证据过于单薄，它可以触发一次返回到研究（research）的交接。整个周期只需大约 30 秒即可收集信息并生成最终的结构化输出。

本节中的监督者（supervisor）、层级（hierarchical）和群体（swarm）架构展示了不同的协调策略如何影响多智能体系统（MAS）。监督者提供集中控制，层级允许可扩展地组织为团队的团队，群体则强调灵活的同伴间协作。表 2-4 对这些模式进行了比较，并重点说明了每种模式的适用场景。

表 2-4. 模式总结：多智能体协调策略

| 架构 | 适用场景 | 优势 | 权衡 |
| --- | --- | --- | --- |
| 监督者 | 任务边界清晰的小型智能体团队 | 设置简单；控制力强；避免死锁 | 集中式瓶颈；单点故障 |
| 层级 | 需要可扩展性的较大型、多团队工作流 | 模块化；可扩展；反映组织结构 | 调试较困难；需要精心设计 |
| 群体 | 同伴之间的动态协作（例如，研究 ↔︎ 撰写者） | 灵活；自然的来回交互流程 | 可预测性较低；需要稳健的交接设计 |

这表明每种协调模式都支持不同规模和风格的智能体协作。通过混合搭配它们，你可以在整体 MAS 设计中塑造出兼顾控制力、可扩展性和适应性的智能体工作流。

## 结论

本章的重要启示是：AI 智能体的智能不仅来自其所使用的 LLM，也来自其架构。正因如此，你探索了架构模式如何塑造 AI 智能体的智能和可靠性。通过超越单步执行，你看到了 CoT、ToT 和 ReAct 等推理循环如何将结构嵌入智能体的思维过程，使输出更加深思熟虑和透明。

同样重要的是，你看到了人在回路（human-in-the-loop）如何让智能体更加安全。中断、审批门控和审查步骤展示了护栏（guardrails）如何使智能体在错误会带来实际后果的真实场景中值得信赖。自主性与监督之间的平衡是现代智能体设计的一个核心主题。

你还学习了这些推理方法如何扩展到 MAS 中。监督者和层级架构将任务分配给专门的工作者，而群体设置则支持同伴之间灵活交接。总而言之，这些范式表明，智能体的智能不只是关于更强的模型，更在于如何将这些模型编排成能够规划、行动和反思的系统。

有了这些基础，你就可以进入更高的层次。在下一章中，你将探索高级范式，这些范式将这些架构延伸至可扩展执行、更深入的规划策略以及更复杂的推理方法。这将向你展示如何设计这样的智能体：它们不仅能反应和反思，还能进行大规模规划，并在更大的系统中协调复杂工作流。

^(1) Jason Wei 等人. “思维链提示可引发大型语言模型中的推理”, [*https://arxiv.org/abs/2201.11903*](https://arxiv.org/abs/2201.11903) (2023).

^(2) Shunyu Yao 等人. “思维树：利用大型语言模型进行深思熟虑的问题求解.”, [*https://arxiv.org/abs/2305.10601*](https://arxiv.org/abs/2305.10601) (2023).
