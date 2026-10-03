
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
