![](assets/ER_cover.png)

![](assets/ER_cover.png)

# AI Agents: The Definitive Guide

by Nicole  Koenigstein

Copyright © 2027 Nicole Koenigstein. All rights reserved.

Published by O’Reilly Media, Inc., 141 Stony Circle, Suite 195, Santa Rosa, CA 95401.

O’Reilly books may be purchased for educational, business, or sales promotional use. Online editions are also available for most titles ([https://oreilly.com](https://oreilly.com)). For more information, contact our corporate/institutional sales department: 800-998-9938 or *corporate@oreilly.com* .

*   Acquisitions Editor: Nicole Butterfield
*   Development Editor: Sarah Grey
*   Production Editor: Elizabeth Faerm
*   Copyeditor: TO COME
*   Proofreader: TO COME
*   Indexer: TO COME
*   Cover Designer: Susan Brown
*   Cover Illustrator: TO COME
*   Interior Designer: David Futato
*   Interior Illustrator: Kate Dullea

*   November 2026: First Edition

Add additional revdate spans below as needed. 

# Revision History for the Early Release

*   2025-09-12: First Release

See [https://oreilly.com/catalog/errata.csp?isbn=9798341666931](https://oreilly.com/catalog/errata.csp?isbn=9798341666931) for release details.

The O’Reilly logo is a registered trademark of O’Reilly Media, Inc. *AI Agents: The Definitive Guide*, the cover image, and related trade dress are trademarks of O’Reilly Media, Inc.

The views expressed in this work are those of the author and do not represent the publisher’s views. While the publisher and the author have used good faith efforts to ensure that the information and instructions contained in this work are accurate, the publisher and the author disclaim all responsibility for errors or omissions, including without limitation responsibility for damages resulting from the use of or reliance on this work. Use of the information and instructions contained in this work is at your own risk. If any code samples or other technology this work contains or describes is subject to open source licenses or the intellectual property rights of others, it is your responsibility to ensure that your use thereof complies with such licenses and/or rights. PROD: Uncomment the following sentence if appropriate and add it to the above para: This book is not intended as \[legal/medical/financial; use the appropriate reference\] advice. Please consult a qualified professional if you require \[legal/medical/financial\] advice.

979-8-341-66689-4

\[LSI\]

# Brief Table of Contents (*Not Yet Final*)

Chapter 1: From LLMs to Agents: The Foundational Blueprint (available)

Chapter 2: Architectures and Patterns: Planning, Reactivity, and Multi-Agent-Systems (available)

*Chapter 3: Advanced Paradigms: Advanced Planning and Reasoning and Scalable Execution in Agents* (unavailable)

*Chapter 4: The Models Behind the Agents: Capabilities and Optimization* (unavailable)

*Chapter 5: Interfaces and Tooling: System Integration and Execution* (unavailable)

*Chapter 6: Deploying Agents in Real Products* (unavailable)

*Chapter 7: Benchmarking LLMs and Agentic Systems for Production* (unavailable)

*Chapter 8: Multimodal and Domain-Specific Benchmarks* (unavailable)

*Chapter 9: Safety, Guardrails, and Risk Mitigation* (unavailable)

*Chapter 10: Memory, Autonomy, and Long-Horizon Use* (unavailable)

*Chapter 11: Cost Estimation and Efficiency Optimization* (unavailable)

*Chapter 12: Future Directions* (unavailable)

# Chapter 1. From LLMs to Agents: The Foundational Blueprint

One of the defining traits of human intelligence is the way we combine inner reasoning with concrete actions, and this maps surprisingly well to how large language model (LLM) agents operate when paired with tools. Take the process of building a coding project. As a human developer you begin with a prompt, the client’s request. First comes reasoning, where you sketch out a plan of how to approach it. Then comes action, such as searching for documentation, writing functions, or debugging errors. Feedback enters the loop when tests fail, a peer review points out gaps, or the client tests the application. Each step is not static but iterative, with reasoning adapting to new insights and actions changing in response.

The same applies to a single LLM agent. Given a task, the agent first reasons about what is missing or what step comes next, then takes actions by calling tools to retrieve data, run code, or check results. Like a developer’s feedback loop, the environment provides signals such as errors, gaps, or confirmations that guide the next iteration. External support, such as code libraries or documentation, maps to tools like retrieval systems or web searches, which refine the output beyond the model’s own raw ability.

An LLM agent is a large language model embedded in a loop of reasoning, acting, and feedback, where it can call external tools and adapt its behavior based on results. Unlike a standalone LLM, which is limited to static text generation, an agent operates as a decision-making entity within a workflow. In this book, I will use AI agents, agents, and LLM agents interchangeably. Strictly speaking, an AI agent or agent does not need to involve a language model at all. For example, in pure reinforcement learning agents are trained directly through trial and error interaction with an environment. The benefit of using an LLM is that it brings strong generalization, reasoning, and natural language capabilities, which makes agent behavior more flexible and broadly applicable beyond narrowly defined environments.

Just as individual skills eventually meet their limits, a single agent can only take a workflow so far before complexity demands coordination. If you think about it, in a small company, a single contributor might handle everything end-to-end: designing, coding, testing, and deploying. But as the project grows, the team needs to expand. One person might focus on backend, another on frontend, another on testing. To keep everything aligned, a tech lead or VP of engineering steps in, coordinating the workload, assigning tasks, and integrating results. The difference between a lone contributor and a coordinated team maps directly onto the difference between a single agent and a multi-agent system (MAS). Just as a tech lead coordinates multiple contributors, a supervisor agent can orchestrate multiple specialized agents to collaborate on solving a complex goal.

It is also important to recognize that neither humans nor agents operate with complete autonomy. Developers are guided by coding standards, project requirements, and organizational processes, and their freedom is bounded by these structures. Agents are similarly constrained by their workflows, operating only within the scope of the tools, permissions, and safeguards they are provided. Far from being a weakness, this lack of full autonomy is what makes both systems viable. For humans, structure ensures code quality, maintainability, and client satisfaction. For agents, constraints ensure safety, reliability, and alignment with the goals set by their developers and users.

This parallel highlights why extending pure LLMs with tools makes sense. A language model on its own is like a developer cut off from resources like StackOverflow, IDEs, or GitHub, capable of reasoning but unable to gather new data, verify results, or improve their task output without outside help. Just as software teams scale their capabilities by adding better tools, resources, and leadership, LLMs extend their capability when embedded in agentic workflows and even multiple specialized agents within a MAS. Without these upgrades, both remain confined to the limits of their initial training and quickly become overwhelmed when faced with complex tasks. With tools, feedback, and a divide and conquer approach, LLMs can iteratively refine, adapt, and solve complex tasks that would otherwise exceed their standalone capacity.

I assume in this book that you have at least a fundamental knowledge of LLMs. Maybe you’ve read the book [Hands-On Large Language Models](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952//), or a similar work. You don’t need to know how to deploy LLMs to make your agents work, since I’ll guide you through those steps in later chapters. However, I expect you have solid coding skills in Python, meaning you know what classes and functions are, and that you are familiar with core ML and deep learning concepts such as neural networks, training loops, and backpropagation. A working knowledge of linear algebra and calculus is also helpful, since they underpin much of modern deep learning. It is also useful if you understand how transformers process sequences and have at least a basic sense of how embeddings work or vector stores support retrieval.

If you haven’t touched these topics in a while, DON’T PANIC! Just like The Hitchhiker’s Guide to the Galaxy advises not to panic, I’ll say the same here in my guide. That said, every concept will be grounded in code, so your knowledge becomes usable again without requiring you to dust off old textbooks. And no, before you ask, the answer to everything about AI agents is not 42, but I promise you’ll get your answers throughout this book.

# Code for the book

All code examples from this book are available in the [GitHub accompanying repository](https://github.com/Nicolepcx/ai-agents-the-definitive-guide). The repository is organized by chapter, so you can easily find the code that belongs to each section.

If you would like a closer look at transformers across different domains and how embeddings are used for text and other modalities, my book [Transformers: The Definitive Guide](https://learning.oreilly.com/library/view/transformers-the-definitive/9781098167004/) can serve as a complementary source. It also explains the fundamentals of reinforcement learning, which will help you build a stronger foundation for understanding the advanced agent concepts introduced later in this book.

Therefore, this book is written for readers who want to go beyond curiosity, which is why understanding these fundamentals is important. This includes data scientists, software and ML engineers who are building real systems in production, and technical leaders who need to understand the trade-offs, costs, and guardrails involved. This book is not about explaining LLMs from scratch, nor is it about abstract promises, or thought experiments. It is about the practical realities of building and running AI agents that can perform and be useful outside a PowerPoint presentation or a lab demo.

This chapter explains how LLMs evolve from static prompts to dynamic agentic systems, why tool use is essential, and what being stateful and autonomous truly means in the design of modern AI agents.

# Finite and Hierarchical State Machines: The Base Paradigm of Agents

[LangGraph](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/), [CrewAI](https://www.crewai.com/), and similar frameworks build on the concept of state machines. At a high level, a *state machine* is a computational model that exists in one of a finite number of states at any given time. It transitions between states in response to specific events or inputs, with the next state determined by the current state and the input.

This section is meant to help you build a mental model of agents by showing how state machines form the foundation of modern AI agent frameworks. In simple terms, think of it as structured transitions, a pattern long used in compilers, GUIs, embedded systems, and networking protocols, now applied to reasoning and tool use in LLM-based agents.

My intention in grounding your understanding first in finite state machines (FSMs) and hierarchical state machines (HSMs) is more than a loose analogy. I want to give you a blueprint for building LLM agents that are robust, reliable, and adaptable. It will empower you to not only understand the concepts but to apply them effectively as you learn about more complex topics later in the book. However, if you haven’t worked with state machines recently, don’t worry. You’ll see how the patterns translate directly in the code examples.

## Finite State Machines

A *finite state machine* has a small vocabulary in which it can act upon and modes your system can be in, and the allowed moves between them. Each move is triggered by an event, optionally guarded by a predicate, and may perform actions that update a state. Start and end are distinguished states with special meaning. The core vocabulary is:

State
: 

A compact snapshot that captures what the system knows at a given moment. For an agent, this might be the message list, routing hints, or progress markers.

Event
: 

Something that happens since the last decision, such as a tool being invoked or returning a result.

Guard
: 

A check that decides which transition to take, based on the current state and latest event.

Action
: 

Work performed during a transition, like invoking a tool, appending a message, saving a checkpoint, or requesting human input.

Termination
: 

A condition that signals the process has reached the end.

[Figure 1-1](#ch01.html_fsm_toolcall) illustrates a finite state machine for tool use. Each transition is defined by an event, a guard, and an action.

![ch01 fsm toolcall vocab](assets/ch01_fsm_toolcall_vocab.png)

###### Figure 1-1. High level overview of a finite state machine for tool use.

An FSM is ideal when behavior alternates among a few stable modes and when recoverability matters, since you can checkpoint on every node and resume after a crash from the last completed state.

Thinking in terms of states lets you decompose a complex task into distinct steps. For example, instead of a single prompt for “write a blog post”, you might define states such as topic ideation, outline generation, drafting, editing, and SEO optimization. Each state can link to specific tools your agent will use.

*Transitions* make you specify the conditions under which one task hands off to the next. For instance, what output from the outline generation state triggers the drafting state. This structure counters unstructured, one-shot prompts that often lead to inconsistent results with LLMs alone.

As soon as you add more modes, such as planning, reflection, approval gates, and retries, an FSM can force duplication or create spaghetti routing. You need to keep control of the edges, since they are decision points: each one encodes the event, guard, and action that drives functionality. This is where hierarchy helps.

## Hierarchical State Machines

*Hierarchical state machines* let states contain other states. Superstates capture shared entry/exit behavior and shared guards. Substates inherit those rules and add their own. History nodes remember which substate you were in when you left, so you can resume exactly there later. These new additions to the state machine are explained below.

Superstate
: 

A state that groups a set of child states and their common policies. Entering the superstate runs shared logic once; exiting it runs shared cleanup once.

Substate
: 

A concrete mode that lives inside a superstate. It has its own edges but inherits the superstate’s guards and actions.

History
: 

A “remember where I was” marker. Shallow history resumes at the last active child; deep history resumes inside nested grandchildren. In agent terms, this is a checkpoint for a portion of the graph.

Parallel region
: 

Two or more child regions of a superstate that advance independently. Useful for fanning out tool calls or agent roles and then joining.

[Figure 1-2](#ch01.html_hsm_agent) shows a hierarchical state machine with a `WORKING` superstate. `PLAN`, `ACT`, and `REFLECT` are substates. `H` is a shallow history marker that records the last active substate so execution can resume there if the superstate is re-entered.

![ch01 hsm agent](assets/ch01_hsm_agent.png)

###### Figure 1-2. Overview of a hierarchical state machine with history marker hat records the last active substate.

HSMs are particularly relevant to more advanced MAS. An HSM allows for “states within states”, which reduces duplication and makes policies obvious. For example, you can attach rate limits, safety filters, or circuit breakers to a `WORKING` superstate, so planning, acting, and reflecting all inherit the same safeguards. You define a guard or router once and reuse it across nodes. Consider a research agent. Instead of treating research as a single monolithic step, you can break it into a sub-machine with states such as searching for keywords, gathering URLs, scraping data, and summarizing findings. Each of these states can even map to a dedicated agent with its own capabilities, contributing to the larger team of agents. This nested structure helps manage complexity, keeps policies consistent, and makes the overall system easier to reason about and maintain.

## Mapping FSM/HSM to Agent Frameworks

LangGraph already speaks the language of state machines and hierarchy. You implement it with subgraphs and shared keys. The router or decision-making agent that selects the next action (such as calling the next agent or tool) is analogous to the event handler in a classic state machine. It’s the part of the system that takes the current state and the LLM’s output (the event or payload) and determines the next logical step. The direct mapping is shown below.

State schema
: 

The parent graph’s TypedDict defines the shared “bus”. Subgraphs declare which keys they read and write. Overlapping keys are the superstate’s interface.

```
class State(TypedDict):
messages: List[BaseMessage]  
working_last: str | None 
```

[![1](assets/1.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO1-1)
: 

Shared bus across parent and subgraphs.

[![2](assets/2.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO1-2)
: 

Optional shallow history marker.

Nodes and subgraphs
: 

A subgraph is an HSM superstate. Its internal nodes are substates. Entering the subgraph runs its entry node; leaving it runs its exit edge back to the parent.

```
def plan_node(state: State) -> State:
    ai = AIMessage(content="Plan next step")
    return {"messages": [ai], "working_last": "plan"}

def act_node(state: State) -> State:
    ai = AIMessage(content="Act on plan")
    return {"messages": [ai], "working_last": "act"}

subgraph_builder = StateGraph(State)
subgraph_builder.add_node("PLAN", plan_node)
subgraph_builder.add_node("ACT", act_node)
subgraph_builder.add_edge(START, "PLAN")
subgraph = subgraph_builder.compile() 
```

[![1](assets/1.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO2-1)
: 

Subgraph is the superstate.

Guards and conditional edges
: 

Parent-level guards enforce global policy (budget, safety). Child-level guards route among PLAN/ACT/REFLECT. Because guards are plain Python, they remain deterministic and auditable.

```
def route_within_working(state: State):
    last = state.get("working_last")
    return "ACT" if last == "plan" else END

subgraph_builder.add_conditional_edges("PLAN",
                route_within_working, {"ACT": "ACT", END: END}) 


parent = StateGraph(State) 
parent.add_node("WORKING", subgraph)

def global_guard(state: State):
    tail = "".join([
            getattr(m, "content", "") for m in state["messages"][-3:]
            ]).lower() 
    return END if "stop" in tail else "WORKING"

parent.add_edge(START, "WORKING")
parent.add_conditional_edges("WORKING", global_guard,
                            {"WORKING": "WORKING", END: END})
```

[![1](assets/1.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO3-1)
: 

Add conditional edge inside subgraph.

[![2](assets/2.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO3-2)
: 

Parent graph with a global guard.

[![3](assets/3.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO3-3)
: 

Example policy: end if user said stop.

History and checkpointing
: 

MemorySaver gives you shallow and deep history for free: resuming a thread recreates the subgraph at the last completed node. If you want explicit “return to where I was inside WORKING,” persist a small marker (for example, state\["working\_last"\] = “reflect”) and branch on it when re-entering.

```
def reflect_node(state: State) -> State: 
    ai = AIMessage(content="Reflect on result")
    return {"messages": [ai], "working_last": "reflect"}

checkpointer = MemorySaver() 
app = parent.compile(checkpointer=checkpointer)

cfg = {"configurable": {"thread_id": "session-1"}} 
app.invoke({"messages": [HumanMessage(content="start")],
                        "working_last": None}, config=cfg)
```

[![1](assets/1.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO4-1)
: 

Substates set working\_last on exit.

[![2](assets/2.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO4-2)
: 

Compile with checkpointing.

[![3](assets/3.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO4-3)
: 

Run in a thread. MemorySaver restores last completed node on resume.

FSMs give you a way to structure tasks into stable, recoverable steps. HSMs extend this by reducing duplication and centralizing policy when workflows grow more complex. Both patterns prepare you for the next step: turning static LLM predictions into dynamic, stateful agents.

# Foundations: From Static Models to Dynamic Agents

As you saw in the previous section, adding states and control flows is essential for automation. The same principle applies to agents. To move from static LLM usage to agentic systems, you need explicit state and control flow. State turns one off tool calls into a memory bearing process. Control flow adds guards and transitions that govern when to plan, when to act, and when to stop.

The reason to add this to LLMs is simple: LLMs are static predictors. They generate the next token given a prompt, drawing only on patterns encoded in their training data. This makes them powerful reasoners but leaves them confined to what they already know. They cannot update knowledge, verify claims, or interact with an environment.

*Agency* emerges when reasoning is coupled with action. *Reasoning* is the internal process of planning or deciding the next step. *Action* extends this by invoking tools, retrieving data, executing code, or calling external services. Together they form an iterative loop: reason, act, receive feedback, and adjust. This loop is the foundation of agentic behavior.

[Table 1-1](#ch01.html_trad_vs_agents) compares static LLMs to agentic systems. These contrasts show not only why agents are necessary, but also how their design shifts traditional machine learning workflows into dynamic, tool augmented systems.

Table 1-1. Comparison of Static LLMs and Agentic Systems

 
| Static LLM (Traditional Use) || Agentic System |
| --- | --- |
| Single-pass token generator || Iterative reasoning–action–feedback system |
| --- | --- |
| Confined to training data || Can retrieve, verify, and update knowledge |
| No memory, stateless || Stateful with short- and long-term memory |
| Linear prompt–response || Iterative and adaptive workflow |
| No tool use || Tool-augmented (retrieval, code, APIs) |
| Fixed interpretation of prompt || Can refine or reinterpret goals dynamically |
| No external validation || Actively checks, corrects, and improves output |

[Figure 1-3](#ch01.html_agent_core_diagram) illustrates how modules turn a static LLM into a dynamic agent. The user request enters the core, planning and memory shape its reasoning, and tools enable concrete action. The cycle of reasoning, acting, and adapting emerges from the coordination of these elements.

![ch01 agent core](assets/ch01_agent_core.png)

###### Figure 1-3. Baseline agent architecture: an LLM alone can only predict text, but with planning, memory, and tools it becomes a usable agentic system.

Planning modules guide the agent’s reasoning, ranging from simple chain-of-thought ^([1](#ch01.html_id47)) traces to more advanced approaches such as trees of thought ^([2](#ch01.html_id48)) or self-critique. Memory modules provide continuity, allowing the agent to recall past steps, reuse prior knowledge, or persist information across sessions. Tools connect the agent to its environment, whether through search, retrieval, code execution, or custom APIs.

Moreover, a key difference between traditional LLM usage and agentic systems is whether the workflow is stateless or stateful. A *stateless call* treats the model like a black box: it takes a prompt, predicts the next tokens, and returns a result. Nothing is remembered, no actions are taken, and no feedback loop exists. [Example 1-1](#ch01.html_SimpleLLM) illustrates a stateless LLM call.

##### Example 1-1. Stateless LLM call with LangChain

```
llm = ChatOpenAI(model="gpt-5-mini")
response = llm.invoke("What are AI agents?")
print(response.content)
```

In addition, the moment you need facts from outside training data, or you want the model to act (search, compute), you introduce tools.

In the following code ([Example 1-2](#ch01.html_createTools) - [Example 1-4](#ch01.html_statelessRun)) is a minimal stateless tool-using run: first you bind two tools to the LLM, and then run a tiny loop that executes any tool calls the model needs to fulfill its task. First, you need to define your tools. [Example 1-2](#ch01.html_createTools) shows an implementation of tools.

##### Example 1-2. Defining tools for a stateless run

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

[![1](assets/1.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO5-1)
: 

LangChain’s `@tool` decorator uses the function name, typed signature, and docstring to build the schema the model sees.

[![2](assets/2.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO5-2)
: 

The docstring is mandatory: if it is missing, LangChain will raise an error when constructing the tool schema or binding tools to the model.

[![3](assets/3.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO5-3)
: 

Constrain evaluation: deny globals and expose only the constants you want. This keeps the calculator deterministic and safe.

With LangChain’s `@tool` decorator, the function signature and docstring are used to build the schema the model sees. You need to keep these precise, consistent and as explanatory as possible for the task-to-be-done efficiently. The docstring acts as the tool description and is a hard requirement from LangChain, if it’s missing, you get an error. In addition, if the docstring is vague the model is more likely to produce invalid calls.

Now, to give the LLM access to the tool, you just bind the tools to the LLM as shown in [Example 1-3](#ch01.html_bindTools).

##### Example 1-3. Bind tools to LLM

```
llm = ChatOpenAI(model="gpt-4o", temperature=0,
                max_tokens=800).bind_tools(tools, tool_choice="auto") 
```

[![1](assets/1.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO6-1)
: 

Bind tools to the model. tool\_choice="auto” lets the model decide whether and which tool to call.

# Modern LLMs are built for agentic tasks

Modern LLMs are not only capable of calling tools, they are increasingly trained and optimized specifically for agentic tasks. For example, models such as [Kimi K2](https://moonshotai.github.io/Kimi-K2/) and [Llama 4](https://www.llama.com/docs/model-cards-and-prompt-formats/llama4/) are explicitly tuned for coding, tool use, and powering agentic systems. These models go beyond answering questions in a chat window. They are instruction tuned to act, invoking APIs, running code, or retrieving data as part of their core design.

With tools bound, you can run a minimal stateless tool loop. The code in [Example 1-4](#ch01.html_statelessRun) executes any tool calls the model requests, feeds observations back, and forces a short wrap up to ensure the final answer is returned even if the last step ended on a tool observation.

##### Example 1-4. Stateless single run with tool loop

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

[![1](assets/1.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO7-1)
: 

Local short term memory for this run only. This makes the workflow stateless across runs.

[![2](assets/2.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO7-2)
: 

A simple step bound protects against runaway loops: LLMs cycling with no termination.

[![3](assets/3.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO7-3)
: 

One model step. The assistant may choose to call tools.

[![4](assets/4.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO7-4)
: 

Read structured tool calls.

[![5](assets/5.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO7-5)
: 

Attach the observation with the matching tool\_call\_id so the model can correlate results to calls.

[![6](assets/6.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO7-6)
: 

Tool cycles can end on an observation. This final instruction guarantees a closing assistant message in your requested format.

## Giving the LLM a State

The first step in turning an LLM into an agent is to make it stateful. Give your LLM a state that records where it’s in the workflow. Based on input and logic, it transitions to the next state, and each state has an associated function or behavior. In LangGraph, this can be any Python type, but is typically a `TypedDict` or [Pydantic](https://docs.pydantic.dev/latest/api/base_model/) `BaseModel`. The following outlines LangGraph’s key concepts to build stateful, adaptable AI agents.

State
: 

A shared data structure that represents the current snapshot of an application. It can be any Python type, but is typically a TypedDict or Pydantic BaseModel.

Nodes
: 

Functions that encode the agent’s logic. They take the current state, perform computation or side effects, and return an updated state.

Edges
: 

Rules that select the next node based on the current state. They can be conditional branches or fixed transitions and should also detect when a finish condition is met.

Command
: 

An object that combines control flow and state updates to support multi-actor communication. A node can both update the state and choose the next node by returning a `Command`.

So far, the tools you created earlier have only been used in *stateless loops*. Each run was independent: the model could call tools, get answers, and wrap up, but as soon as the run ended, all context was lost. Now, if you want to combine LangGraph’s concepts to build stateful agents, you need to embed the tools into a stateful workflow. Instead of discarding messages after each turn, you keep them in a structured state. This makes the system agentic:

*   The LLM node reasons about the next step.
    
*   The tool node executes external actions.
    
*   Edges route control flow based on whether tools were requested.
    
*   Memory checkpoints persist the conversation, so the agent can reuse earlier results in later turns.
    

This transforms tool use from being a one-off helper into a continuous reasoning–acting–adapting cycle. In the next code listings ([Example 1-5](#ch01.html_stateGraphSetup) – [Example 1-13](#ch01.html_memoryViews)), you’ll build such a stateful agent with LangGraph, starting from the same tools and LLM binding introduced in [Example 1-3](#ch01.html_bindTools) - [Example 1-4](#ch01.html_statelessRun).

[Example 1-5](#ch01.html_stateGraphSetup) shows how a state object can be implemented in LangGraph to carry the conversation and coordinate control flow. You can reuse the tools and the bound llm from [Example 1-3](#ch01.html_bindTools) and add only what is new: a typed state, an LLM node that appends assistant messages, a tool node that executes calls, and a router that decides whether to continue or stop based on tool calls.

##### Example 1-5. Build a minimal LangGraph with state, nodes, and routing

```
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

[![1](assets/1.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO8-1)
: 

The state carries the conversation as a message list. `add_messages` handles safe merging across node updates.

[![2](assets/2.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO8-2)
: 

Core reasoning node. It takes the current state and returns a new assistant message.

[![3](assets/3.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO8-3)
: 

Prebuilt node to execute tool calls emitted by the assistant.

[![4](assets/4.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO8-4)
: 

Graph container for nodes and edges.

[![5](assets/5.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO8-5)
: 

Router. If the last assistant turn requested tools, go to tools, otherwise finish.

With the graph defined, you need to make it runnable and persistent. [Example 1-6](#ch01.html_compileGraph) compiles the graph with an in-memory checkpointer and assigns a thread id. The checkpointer restores prior messages on each turn so the agent can reuse earlier results. The thread id separates branches, so you can run parallel conversations without interference.

##### Example 1-6. Compile with in-memory checkpoints and configure a thread

```
checkpointer = MemorySaver() 
app = graph.compile(checkpointer=checkpointer) 

cfg = {"configurable": {"thread_id": "nyc-weather-session"}} 
```

[![1](assets/1.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO9-1)
: 

Lightweight, per-thread checkpointing that persists state across turns.

[![2](assets/2.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO9-2)
: 

Produces a runnable app that knows how to save and load state.

[![3](assets/3.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO9-3)
: 

A thread id identifies one conversation branch. New ids create new branches.

Once you have a runnable app and checkpoints in place, the next step is to make its execution observable. You want to see what the graph is doing, which messages are flowing, and how tool outputs attach. [Example 1-7](#ch01.html_inspectState) introduces a compact helper that formats messages into a single readable line, making it easier to follow traces without overwhelming logs.

##### Example 1-7. Trace execution and inspect state and memory

```
def _short(msg: BaseMessage, max_len: int = 140) -> str:
    """Compact one-line view of a message."""
    role = type(msg).__name__.replace("Message", "").lower()
    content = getattr(msg, "content", "")
    if isinstance(content, list):
        # some tool outputs can be list payloads
        try:
            content = json.dumps(content)
        except Exception:
            content = str(content)
    text = str(content).replace("\n", " ").strip()
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    if hasattr(msg, "tool_calls") and getattr(msg, "tool_calls"): 
        tnames = [tc.get("name", "tool") for tc in msg.tool_calls]
        return f"{role}: tool_calls -> {tnames}"
    if isinstance(msg, ToolMessage):
        return f"{role}({msg.name}): {text}"
    return f"{role}: {text}"
```

[![1](assets/1.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO10-1)
: 

Include tool name or function call info when available.

[Example 1-8](#ch01.html_print_state_snapshot) prints a concise view of messages and routing hints. Together they let you verify that tool outputs are attached correctly and that state transitions behave as intended.

##### Example 1-8. Trace execution and inspect state and memory

```
def print_state_snapshot(app, config, title: str):
    """Print current graph state and memory for a given thread."""
    snap = app.get_state(config) 
    values = snap.values or {}
    msgs: List[BaseMessage] = values.get("messages", [])
    print(f"\n=== {title} | state snapshot ===")
    print(f"messages: {len(msgs)} total")
    for i, m in enumerate(msgs[-5:], start=max(0, len(msgs)-5) + 1): 
        print(f"  {i:>3}: {_short(m)}")

    nxt = getattr(snap, "next", None) 
    tasks = getattr(snap, "tasks", None)
    if nxt:
        print(f"next nodes: {list(nxt)}")
    if tasks:
        print(f"queued tasks: {tasks}")
    print("memory: in-memory checkpoint present for this thread")
```

[![1](assets/1.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO11-1)
: 

Read the latest state for a thread.

[![2](assets/2.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO11-2)
: 

Print a concise tail of messages to keep logs readable.

[![3](assets/3.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO11-3)
: 

Show routing info and queued tasks if present.

[Example 1-9](#ch01.html_run_with_tracing) streams node updates, so you can see each step the graph takes.

##### Example 1-9. Trace execution and inspect state and memory

```
def run_with_tracing(app, input_state: AgentState, config, title: str):
    """Run the graph while printing per-node updates and final memory."""
    print(f"\n=== {title} | execution trace ===")
    final = None
    for event in app.stream(input_state, config=config, stream_mode="updates"): 
        for node, upd in event.items():
            keys = list(upd.keys()) 
            print(f"[enter {node}] updated: {keys}")
            # if messages updated, print the last one briefly
            msgs = upd.get("messages") or []
            if msgs:
                print(f"  {_short(msgs[-1])}")
            print(f"[leave {node}]")
            final = upd
    print_state_snapshot(app, config, title=f"{title} | after run") 
    snap = app.get_state(config)
    msgs = snap.values.get("messages", [])
    return msgs[-1].content if msgs else ""
```

[![1](assets/1.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO12-1)
: 

Stream node updates to see the exact flow through the graph.

[![2](assets/2.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO12-2)
: 

`upd` is a dict like `{"messages": [<new msg>]}` or tool results.

[![3](assets/3.png)](#ch01.html_co_from_llms_to_agents__the_foundational_blueprint_CO12-3)
: 

Show final assistant message from `app.get_state`.

[Example 1-10](#ch01.html_firstTurn) asks for the current air temperature in New York City in Celsius. The trace shows the graph entering the LLM node, initiating a tool call if needed, and returning a final assistant message. The state snapshot confirms that the result is saved in memory for this thread.

##### Example 1-10. Turn 1: get the current NYC air temperature in Celsius

```
turn1_answer = run_with_tracing(
    app,
    {"messages": [HumanMessage(content="Get the current air temperature in New York
    City in Celsius.")]},
    config={**cfg, "recursion_limit": 20},
    title="TURN 1",
)
print("\nTURN 1 (final assistant):\n", turn1_answer)
```

This results in this model output:

```
TURN 1 (final assistant):
The current air temperature in New York City is 18°C.
```

You continue in the same thread. [Example 1-11](#ch01.html_secondTurn) asks to square the previously retrieved temperature. Because the thread id is unchanged, the app restores the earlier messages, allowing the assistant to use the prior result without you repeating it in the prompt.

##### Example 1-11. Turn 2: square that temperature in the same thread

```
turn2_answer = run_with_tracing(
    app,
    {"messages": [HumanMessage(content="Now compute the square of that
    temperature.")]},
    config={**cfg, "recursion_limit": 20},
    title="TURN 2",
)
print("\nTURN 2 (final assistant):\n", turn2_answer)
```

Now the square will be computed and the model returns:

```
TURN 2 (final assistant):
The square of the temperature, 18°C, is 324.
```

Agents often need to explore alternatives without losing progress. [Example 1-12](#ch01.html_branchThread) starts a separate branch by using a different thread id and requests a different follow-up. This creates an isolated conversation that can diverge from the main path while preserving the main thread’s memory intact.

##### Example 1-12. Branch a parallel thread for a different follow-up

```
branch_answer = run_with_tracing(
    app,
    {"messages": [HumanMessage(content="Instead of squaring, convert it to
    Fahrenheit and report both.")]},
    config=cfg_branch,
    title="BRANCH",
)
print("\nBRANCH (final assistant):\n", branch_answer)
```

Finally, you can inspect what each branch remembers. [Example 1-13](#ch01.html_memoryViews) prints compact snapshots for the main thread and the branch. Comparing them side by side confirms that state is persisted within a thread and isolated across threads, which is the foundation for reliable multi-step and multi-branch agent workflows.

##### Example 1-13. View snapshots for both threads

```
print_state_snapshot(app, cfg, title="MAIN THREAD memory view")
print_state_snapshot(app, cfg_branch, title="BRANCH THREAD memory view")
```

With this small implementation, you now see exactly what the graph remembers, how it routed, and what the latest assistant state is for each branch.

# Deus Ex Machina: What is Possibly Right Now?

Once you view agents as HSMs with states, the autonomy spectrum becomes a matter of how much choice you place inside a superstate. Routers are one-edge FSMs with a single guard. Tool-calling agents are HSMs with a WORKING superstate that loops among PLAN, ACT, and REFLECT. MAS compose multiple subgraphs under a supervisor superstate, sometimes with parallel regions. The system still runs inside explicit states and guards, perceived “autonomy” is just a richer choice inside well-bounded regions.

In ancient Greek theater, a *deus ex machina* was the sudden appearance of a god to resolve an unsolvable conflict or problem. It is tempting to imagine AI agents in the same way: a machine that suddenly acts with full autonomy and intelligence solving all your problems. But that’s not what’s possible today, agents are engineered systems. Their strength comes not from hidden magic, but from carefully designed workflows, well-defined tools, and the boundaries that keep them safe and usable. That said, agents are still constrained by the tools you give them, by safeguards you enforce, and by the workflows you allow them to operate within. These constraints are what make agents viable today: safe to deploy, predictable in behavior, and aligned with their intended goals.

What current systems demonstrate is a form of orchestrated autonomy. As you saw in the LangGraph examples earlier ([Example 1-5](#ch01.html_stateGraphSetup)), agents can branch conditionally, call tools, or decide whether to continue a workflow. This already looks very different from a static prompt–response call, but it is still bounded. The orchestration logic is fixed; within it, the LLM can make choices, but it cannot redefine its architecture. [Table 1-2](#ch01.html_autonomy_limits) gives a short overview what agents can do so far.

Table 1-2. Current autonomy in agents: what’s possible vs. what remains out of reach

 
| What agents can do || What agents can’t do yet |
| --- | --- |
| Route between predefined paths (e.g., router node) || Redesign or rewire their own control graph |
| --- | --- |
| Select which tools or sub-agents to call || Generate and validate new graph topologies dynamically |
| Decide if more steps are needed before completion || Continuously assess tool effectiveness or create new tools autonomously |
| Revise their own prompt or toolset within bounds || Guarantee alignment and coherence when shifting strategies mid-execution |
| Write and run small pieces of code to determine next steps || Operate as fully self-directed systems independent of orchestration frameworks |

The agent adapts at runtime, but only within the framework you design. It helps to see this as a spectrum. At one end are routers that make a single decision from a fixed set of options. In the middle are tool-calling agents that support multi-step reasoning, reflection, and memory. At the far end are MAS, where specialized agents collaborate under a supervisor, distributing decision-making across roles. [Figure 1-4](#ch01.html_router_vs_autonomy) shows the spectrum of autonomy, from simple routing decisions to the idea of fully self-directed execution. In practice, nearly all real-world systems fall into the middle: they are agentic systems, not truly autonomous agents. Their autonomy is orchestrated rather than self-evolving, decisions happen within a framework of predefined states, transitions, and toolsets, even if the agent can adaptively choose among them at runtime.

![ch01 autonomy graphic](assets/ch01_autonomy_graphic.png)

###### Figure 1-4. In orchestrated autonomy, decision points are predefined in the workflow. In full autonomy, the LLM could generate new tools, actions, or decision paths to reach its goal.

Agents can simulate flexibility within frameworks such as LangGraph using reflection, planning, and structured outputs. However, we do not yet have systems that can freely redefine their own architecture or reason about their goals in a robust, and general way.

As you now understand, today’s agents are not deus ex machina solutions but carefully orchestrated systems that extend LLMs with structure, tools, and feedback loops. Their autonomy is bounded by design, which is precisely what makes them practical, safe, and deployable. While still limited, they are already impressive and capable of reasoning, acting, and adapting in ways that go far beyond static LLMs.

# Conclusion

In this chapter, you learned how LLMs evolve from static predictors into dynamic agents through the addition of state, control flow, and tool use. By embedding reasoning in a loop of action and feedback, LLMs gain the ability to adapt to results and extend their capabilities far beyond what is possible with prompt–response interactions alone.

Finite and hierarchical state machines provided the conceptual foundation for agent design. FSMs let you structure workflows into recoverable steps, while HSMs reduced duplication and centralized policies for complex, multi-step tasks. These abstractions map directly to modern frameworks such as LangGraph, where state, nodes, and routing rules define how agents reason and act over time.

You now can also place autonomy in a better context. You understand that today’s agents operate within well-bounded workflows, choosing among predefined paths, invoking tools, and reflecting on results, don’t redesign their own architecture. This orchestrated autonomy is not a weakness but a strength, since constraints make agents safe, predictable, and practical to deploy. The example of the AI Scientist v2 showed both the promise and the limitations of this approach, demonstrating how agents can already generate and test hypotheses but still require oversight to ensure rigor and depth.

With this foundation in place, you are ready to move into the next chapter, where you’ll explore the architectures and patterns that bring planning, reactivity, reflection, and human–agent interaction to life.

^([1](#ch01.html_id47-marker)) Jason Wei et al. [“Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.”](https://arxiv.org/abs/2201.11903), (2023).

^([2](#ch01.html_id48-marker)) Shunyu Yao et al. [“Tree of Thoughts: Deliberate Problem Solving with Large Language Models.”](https://arxiv.org/abs/2305.10601), (2023).

^([3](#ch01.html_id50-marker)) Yutaro Yamada et al. [“The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search.”](https://arxiv.org/abs/2504.08066), (2025).

# Chapter 2. Architectures and Patterns: Planning, Reactivity, and Multi-Agent-Systems

You know by now that AI agents are not magic. They are engineered systems whose power comes from the strategic configuration of their architecture. By combining reasoning loops, tools, stateful design, and adaptive control, LLMs can move beyond static prompts and evolve into MAS that plan, react, and reflect in real time. This chapter explores the architectures that make this possible, from structured reasoning methods such as chain of thought, tree of thought, and ReAct to architectural paradigms like supervisor and hierarchical agents. Along the way you will gain a deeper understanding of reasoning flows and see how human-in-the-loop mechanisms including approval gates, corrections, and interruptions keep these systems reliable, aligned, and secure.

The importance of an agent’s architecture becomes obvious when you compare how a simple single-step agent handles a task versus how a multi-step agent does. A single-step agent receives the request, plans everything internally, executes in one go, and returns the result without ever revisiting intermediate steps. A multi-step agent, in contrast, alternates between reasoning and action in an iterative loop, adapting its plan based on partial results or new observations as it works. This flexibility is what allows agents to refine outputs dynamically and cope with uncertainty or feedback from their environment. The contrast is illustrated in [Figure 2-1](#ch02.html_single_vs_multi_step), with [Table 2-1](#ch02.html_single_vs_multi_step_table) showing how each approach unfolds step by step on the same task.

Table 2-1. Example: Single-step vs. Multi-step (ReAct) execution for the task “Summarize the document and create a bar chart of key statistics”

 
| Step || Single-step agent || Multi-step (ReAct) agent |
| --- | --- | --- |
| 1 || Reads the request once || Reads the request and forms an initial thought on the first action (extract statistics) |
| --- | --- | --- |
| 2 || Plans the entire sequence internally || Action: execute extraction; observation: store extracted data in memory |
| 3 || Generates summary and chart in a single pass || Thought: plan next action to generate the summary from extracted data |
| 4 || Returns outputs without revisiting intermediate results || Action: generate summary; Observation: store output in memory |
| 5 || No mid-execution adjustments || Thought: plan final action to create the bar chart; action: create chart; observation: review output |
| 6 || — || If intermediate results are incomplete, repeat the loop until satisfied, then return final outputs |

While the table shows the difference step by step, [Figure 2-1](#ch02.html_single_vs_multi_step) illustrates the overall workflows side by side. The single-step agent follows a linear pipeline from prompt to result, whereas the multi-step agent embeds reasoning, actions, and memory into an iterative loop that continues until the problem is solved or the computational budget is exhausted.

![ch01 single vs multi step](assets/ch01_single_vs_multi_step.png)

###### Figure 2-1. Single-step (left) versus ReAct (right) agent workflows.

# Structured Reasoning and Action

Giving agents distinct personas, sometimes called anthropomorphizing, can sharpen their reasoning and make their contributions more focused. A planner agent framed as a methodical strategist will approach problems differently than a critic agent framed as an evaluator. When each agent is given a clear role and mindset, their reasoning chains tend to be more coherent and their interactions in MAS more purposeful. This alignment improves collaboration, since each agent knows what kind of knowledge it’s expected to contribute.

Yet defining a role is only half the picture. To make agents more effective you might also need to shape how the agent thinks. This is where structured reasoning methods such as chain of thought, tree of thought and ReAct come into play.

## Letting Your AI Agents Think Out Loud

*Chain-of-thought prompting* (CoT) ^([1](#ch02.html_id52)) is inspired by the human thought process, this technique first solves intermediate steps before getting to the final answer. While CoT gets often introduced as prompting trick, in the context of AI agents CoT can serve as added reasoning steps that guides control flow. By forcing the model to generate intermediate steps, you gain better transparency over the reasoning process. It often improves reliability and can help smaller models perform tasks that would otherwise require a larger model. This keeps cost lower while providing clearer insights into how the agent arrives at its decisions. [Table 2-2](#ch02.html_cot_prompt_examples) provides you ideas for prompts for common scenarios that you can adapt to your needs.

Table 2-2. Examples: Chain of Thought prompts by scenario

 
| Scenario || CoT prompt example |
| --- | --- |
| Planning || “Plan step by step to create a one-week blog schedule. Explain the reasoning behind each choice.” |
| --- | --- |
| Data analysis || “Begin by describing the dataset. Next highlight patterns or trends. Conclude with a clear summary of insights.” |
| Math problem || “Break the problem into smaller pieces. Solve each piece with explanation. Then combine the results for the final answer.” |
| Research task || “List the central questions first. Investigate them one by one. Conclude with a synthesized summary.” |
| Debugging || “Start by listing possible causes. Test each cause in turn. Eliminate what does not fit and explain why until you reach the fix.” |
| Decision-making || “List all available options. Consider pros and cons step by step. Choose the option that best fits the criteria and explain why.” |

To make this more explicit how you might want to implement this into your workflow, let’s look at an agent for data analysis. Since the core of a CoT agent is its ability to reason in a series of logical, intermediate steps, this is the perfect example. You can instruct the agent to think step by step, and have it begin by describing first the dataset and next have it highlight patterns or trends. [Example 2-1](#ch02.html_CoTPrompt) shows a generic system prompt you could use for such an agent. I omit the rest of the code for brevity, but you can find the complete working implementation in the [book’s repository](https://github.com/Nicolepcx/ai-agents-the-definitive-guide).

##### Example 2-1. CoT prompt

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

Now to use the prompt with your agent, you just need to create a single agent as show in [Example 2-2](#ch02.html_CoTAgent). Note that I use the `AgentExecutor` from LangChain together with LangGraph for brevity in [Example 2-2](#ch02.html_CoTAgent). But in the accompanying notebook you’ll also find a more granular LangGraph wiring, where the routing between the model and tools is made explicit.

Both approaches yield the same result. However, using the `AgentExecutor` is often preferable in production, since it keeps your code concise, while the granular graph is valuable for debugging and for understanding how messages, tool calls, and control flow work under the hood.

##### Example 2-2. CoT Agent

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

When I ran the code, the agent produced a correct description of the demo dataset, an accurate analysis of the patterns and trends, as well as a bar chart and a summary of its insights. In addition, with the provided code, you can inspect the intermediate tool steps, for example:

```
(AgentActionMessageLog(tool='python_repl', tool_input={'code': "import pandas
as pd\n\ndf = pd.read_csv('demo.csv')\n\ndf.describe()"})
```

or

```
AgentActionMessageLog(tool='python_repl', tool_input={'code': 'df.describe()'})
```

This example illustrates how CoT enables the agent to make its reasoning process visible and verifiable. If you use the code from this example, you basically force the agent to externalize its intermediate steps, and with that you not only gain confidence in the results but also create a transparent audit trail of how conclusions are reached.

## Generating a Tree of Possibilities

Now think of a scenario where you have a more complex task, but you don’t want to use a reasoning model or a full multi-agent architecture. Still, a more exploratory and strategic approach is needed. In such cases, you can use *tree of thoughts prompting* (ToT) ^([2](#ch02.html_id53)). ToT expands CoT reasoning into a branching search process, where multiple reasoning paths are explored and evaluated before deciding on the best next step.

In the code examples ([Example 2-3](#ch02.html_thought_generation) - [Example 2-5](#ch02.html_ToTGraph)), I’ll show you how to implement ToT. The key idea is to let a smaller, single AI agent generate multiple potential paths (a “tree”), have another agent evaluate and select the best one, and then execute that chosen path with yet another model. The simplified ToT method offers a pragmatic middle ground between a simple ReAct loop (see [“How to Make Your Agent Take Action and Reflect”](#ch02.html_React_section)) and the full computational complexity of more tree-search-based algorithms, which I will cover in the next chapter.

# ToT as an engineering shortcut

A lightweight ToT setup where a smaller model proposes branches and a stronger model evaluates can sometimes replace the need for a full multi-agent architecture. This simplifies design and saves both money and time by avoiding repeated calls to large reasoning models when a smaller setup already provides reliable results.

You begin by creating a “tree” of possibilities. The `propose_options` node in [Example 2-3](#ch02.html_thought_generation) is the first step. Instead of producing a single, linear output, it uses a smaller LLM to generate three distinct approaches for a blog post. These three options represent branching paths, or “thoughts,” that the agent can explore. This is the core principle of ToT: moving beyond a single-chain process toward multi-path exploration.

##### Example 2-3. Thought Generation

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

You can then introduce the agent to reflect and select ([Example 2-4](#ch02.html_pruning_node)), which prunes the tree. Here, a more powerful LLM evaluates the three proposed options. The model can be instructed to critique each option based on criteria such as clarity and originality, and then select the best one. This step is crucial in ToT, as it ensures the agent does not simply proceed with the first idea but instead critically assesses the alternatives and chooses the most promising path to pursue.

##### Example 2-4. Pruning Node

```
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

As before, I omit the rest of the code for the other nodes for brevity, but you can find the complete implementation in the [book’s repository](https://github.com/Nicolepcx/ai-agents-the-definitive-guide). What I want to show here is how to tie everything together in LangGraph and which types of LLMs I suggest using. One important paradigm here is the handoff. A *handoff* defines how control shifts between agents: who speaks next, who acts next, and under what conditions. In LangGraph, nodes encode the logic of the agents, and edges represent the handoffs. By wiring multiple nodes together, you are effectively building a MAS within a single graph. The shared state is the memory and context that all nodes operate on. How this wiring looks in practice for ToT is illustrated in [Example 2-5](#ch02.html_ToTGraph).

##### Example 2-5. Creating the graph for ToT

```
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

[![1](assets/1.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO1-1)
: 

Small creative generator LLM.

[![2](assets/2.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO1-2)
: 

Stronger judge/reflector LLM.

[![3](assets/3.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO1-3)
: 

Tool-using researcher.

[![4](assets/4.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO1-4)
: 

Small writer model.

When you run the code, you’ll see different topics proposed, including reference links. To optimize the results further, you could implement a human-in-the-loop ([“Designing Human-in-the-Loop Workflows”](#ch02.html_HITL_section)) interaction to review and approve the actions.

It’s tempting to think of CoT and ToT as just clever prompting hacks, but you’ve seen they can be powerful engineering tools. [Figure 2-2](#ch02.html_cot_tot_examples_compared) contrasts both approaches: on the left a single agent reasoning with CoT, on the right a multi-agent setup where different roles collaborate to explore, reflect, and refine mimicking ToT.

![ch02 cot tot examples compared](assets/ch02_cot_tot_examples_compared.png)

###### Figure 2-2. Control-flow diagram comparing CoT and ToT.

Together, they show how structured reasoning can range from a linear chain of intermediate steps to a branching process of exploration and selection. Both highlight that when agents are guided to think in stages, whether as a single voice or as a coordinated team, the outcome is more deliberate and reliable.

## How to Make Your Agent Take Action and Reflect

You’ve seen in the previous section that CoT and ToT can be helpful for reasoning, but reasoning alone isn’t enough. An agent that only reasons never leaves the page, and an agent that only acts can’t learn from its decisions. To combine the two, you need ReAct, which stands for reasoning and action.

Mathematically, you can think of this as mapping a context *c*[*t*] to an action *a*[*t*] within an action space *Â* = *A* ∪ *L*. Here *A* is the set of task-specific actions that impact the environment, while *L* is the set of language-based reasoning traces or thoughts. Each action step therefore produces not just a task action *a*[*t*], but also a reasoning trace *â*[*t*].

These traces are folded back into the agent’s state so that the context evolves as:

*c*[*t* + 1] = (*c*[*t*], *â*[*t*]) 

In addition, you follow the policy:

*π*(*a*[*t*]|*c*[*t*]) 

Here, *c*[*t*] = (*o*[1], *a*[1], ..., *o*[*t* − 1], *a*[*t* − 1], *o*[*t*]). In other words, the agent carries forward not only what it has done, but also why it chose that path. This recursive structure is what enables reflection: the agent’s future decisions are informed by both its history of actions and the reasoning it has accumulated. The flow diagram in [Figure 2-3](#ch02.html_diagramReAct) illustrates a minimal agent loop implemented in LangGraph.

![CH 02 ReAct](assets/CH_02_ReAct.png)

###### Figure 2-3. Control-flow diagram of a ReAct agent. This loop is the operational form of the policy *π*(*a*[*t*]|*c*[*t*]).

The process begins at `__start__`, this hands control to the agent, and then branches depending on the outcome:

*   If the agent decides it has reached a conclusion, it produces an end signal and moves to `__end__`.
    
*   If more reasoning or action is required, the agent issues a continue signal, possibly calling tools to gather additional information or perform actions.
    
*   After interacting with tools, control flows back to the agent, allowing it to integrate results and reflect before deciding whether to continue or stop.
    

This loop is the essence of ReAct: combining reasoning, action, and reflection in an iterative cycle. ReAct is more than “the model thinks and then acts”. The reasoning trace itself becomes part of the evolving state. That is, the ReAct agent alternates between reasoning traces and tool outputs, and those traces feed back into context.

While this example is in the context of a single agent, the same structure extends naturally into larger MAS. Each agent can maintain its own cycle of reasoning and acting, with reasoning traces shared or exchanged across agents. This makes ReAct not only a tool for individual decision-making, but also a foundation for collaboration and reflection in more complex agentic setups.

Let’s make the math and the diagram concrete, and let me show you how this actually maps into code. The following listings turn the ReAct equations into code. Note, as before, I’ll just show the important parts here, but the full running implementation will be in the accompanying notebook.

Before we jump into the listings, remember the intent: we carry a growing conversation state *c*[*t*] while the model alternates between language traces *L* and task actions *A*. Each step appends either a thought, an action, or a tool observation back into the state. The code mirrors this loop. Therefore, you first create the state for the graph as in [Example 2-6](#ch02.html_reactstate).

##### Example 2-6. Create ReAct state

```
class AgentState(TypedDict): 
    messages: Annotated[Sequence[BaseMessage], add_messages] 
```

[![1](assets/1.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO2-1)
: 

The state of the agent.

[![2](assets/2.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO2-2)
: 

Context *c*[*t*] is the message stream.

Now you give the model two powers in [Example 2-7](#ch02.html_initReactModel). It can produce language tokens that act as thoughts *L*, and it can request structured tool calls *A*. Binding the tools makes these actions first-class outputs that the runtime can dispatch.

##### Example 2-7. Init LLM

```
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
model = model.bind_tools(TOOLS) 
```

[![1](assets/1.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO3-1)
: 

Binding tools to the model augments the action space to *Â* = *A* ∪ *L*.

[Example 2-8](#ch02.html_CallingModel) is the policy step. It reads the entire message history, applies the system rules, and produces exactly one thing: either a thought, or a tool call request, or both. You don’t branch here, you simply return what the model decided.

##### Example 2-8. Call the model

```
def call_model(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    response = model.invoke([SYSTEM_PROMPT] + state["messages"], config) 
    return {"messages": [response]} 
```

[![1](assets/1.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO4-1)
: 

Policy *π*(*c*[*t*]) → *â*[*t*] is `model.invoke` on the current context.

[![2](assets/2.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO4-2)
: 

The reducer appends new messages, implementing *c*[*t* + 1] = (*c*[*t*], *â*[*t*]).

In [Example 2-9](#ch02.html_tool_calls) you translate intent into effect. Every requested tool call is executed deterministically, and its result is wrapped as a message. By pushing the observation into the same message stream, you let the next policy step reason over fresh evidence without special cases.

##### Example 2-9. Execute tool calls

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

[![1](assets/1.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO5-1)
: 

If the model chose an action *a*[*t*] ∈ *A*, execute the tool.

[![2](assets/2.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO5-2)
: 

Tool output is the observation that informs the next step.

The stopping rule in [Example 2-10](#ch02.html_tot_halting) is simple. If the model asked for a tool, it must complete that effect before asking it again. If it didn’t, you’d assume it produced a final answer. This keeps the loop predictable and avoids orphaned actions.

##### Example 2-10. Halting condition

```
def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None): 
        return "continue"
    return "end"
```

[![1](assets/1.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO6-1)
: 

Halting test reads the last AI message to decide continue or end.

Now you connect the pieces into a small state machine, as shown in [Example 2-11](#ch02.html_Build_ToT_graph). The edges encode the control flow that mirrors the ReAct diagram. The policy node runs first. If it emits an action, we route to the tools node and then back to policy. If it emits only a thought or a final answer, we terminate.

##### Example 2-11. Build workflow graph

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

At runtime, the graph advances one edge at a time. The conditional edge decides whether the next hop is `tools` or `END`. The fixed edge from tools back to agent completes the think-act loop.

To see this in action, you prompt a question, I asked how the weather will be in Zurich. The LLM first decides which tool to call to answer best, then the tool returns evidence, and the model writes a final answer grounded in that evidence. Because every step is recorded in the message stream, the reasoning is retrievable:

````
> Entering new ReAct trace...
Question: What is the weather in Zurich today?

Final Answer: What is the weather in Zurich today?

> Finished trace.

Reasoning Summary: selecting a tool based on the query
Action:
```
{
  "action": "internet_search",
  "action_input": {
    "query": "Zurich weather today"
  }
}
```

Observation:
```
{
  "title": "Zurich, Zurich, Switzerland Weather Forecast",
  "snippet": "Hourly Weather \u00b7 1 PM 74\u00b0. rain drop 0% \u00b7 2 PM
  80\u00b0. rain drop 0% \u00b7 3 PM 82\u00b0. rain drop 0% \u00b7 4 PM
  83\u00b0. rain drop 0% \u00b7 5 PM 84\u00b0. rain drop 0% \u00b7 6 PM ...
```

Final Answer: Today in Zurich, the weather is mostly sunny with a high
of 82°F (about 28°C) and a low of 62°F (about 17°C). Winds are light
from the southwest at 6 mph.

For more details, you can check the full forecast
[here](https://www.accuweather.com/en/ch/zurich/316622/weather-forecast/316622).

> Finished trace.
````

Now you understand how powerful ReAct can be. It’s not just that the agent alternates between thinking and acting, but that the reasoning itself becomes part of the state. This recursive process enables agents to adapt, refine their strategies, and explain their choices. In practice, it allows even relatively small models to perform complex tasks more reliably, while making their decision-making process transparent.

Together, CoT, ToT, and ReAct illustrate how different reasoning strategies can shape the intelligence of your agents. Each approach has its strengths: CoT offers transparency and reliability, ToT explores multiple alternatives before committing, and ReAct closes the loop by combining reasoning with tool use. [Table 2-3](#ch02.html_pattern_summary_reasoning) gives you a consolidated overview of these reasoning patterns along with tips for when they are most useful.

Table 2-3. Pattern summary: Structured reasoning methods (CoT, ToT, ReAct)

 
| Method || Use Case || Benefits || Tradeoffs |
| --- | --- | --- | --- |
| Chain of Thought (CoT) || Step-by-step reasoning for math, planning, analysis || Improves reliability and transparency; enables smaller models on harder tasks || Slower, verbose outputs; sometimes redundant |
| --- | --- | --- | --- |
| Tree of Thought (ToT) || Exploratory tasks with multiple solution paths (creative writing, strategy search) || Explores alternatives; reduces “first-answer bias” || Higher compute cost; requires orchestration (selection and pruning) |
| ReAct || Tasks needing both reasoning and tool use (research, coding, APIs) || Integrates reasoning + action; creates audit trail; adaptive to feedback || More complex state management; risk of over-calling tools |

This highlights how these methods can become powerful building blocks once you integrate them into your overall agentic workflow.

# Surely You’re Joking, AI Agent!

Sometimes an agent’s behavior can be entertaining. It might call the wrong tool, loop endlessly, or produce an oddly creative response. In those moments you might laugh and say to yourself addressing the AI agent: “You’re joking, right?” But when the same unpredictability reaches into access control, financial operations, or customer data, it’s no longer a joke. An action taken at the wrong moment or on the wrong system can have serious consequences. This is why *human-in-the-loop* (HITL) exists. HITL allows the agent to pause at critical points, ask for your approval, and continue only once it’s safe to proceed.

HITL is not only about catching errors before they happen. It also provides accountability and transparency. By inserting checkpoints into an agent’s workflow, you make sure irreversible actions are reviewed, decisions are traceable, and sensitive operations stay under human control. Far from slowing the system down, this oversight makes agents more trustworthy and easier to adopt in production settings.

## Designing Human-in-the-Loop Workflows

There are a couple of different interruptions that you can perform with a HITL workflow. The most common ones are shown below.

Approve or reject
: 

Pause the graph before a critical step, such as an API call, to review and approve the action. If the action is rejected, you can prevent the graph from executing the step, and potentially take an alternative action. This pattern often involve routing the graph based on the human’s input.

Review and edit
: 

Pause the graph to review and edit the graph state. This is useful for correcting mistakes or updating the state with additional information.

Review tool calls
: 

If a tool call could affect external systems, inspect its arguments before execution. For example, you might confirm booking details before a travel reservation is made.

Validate human input
: 

Explicitly request human input at a particular step in the graph. This is useful for collecting additional information or context to inform the agent’s decision-making process or for supporting multi-turn conversations.

LangGraph offers an [extensive documentation](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/add-human-in-the-loop/) on how to apply these patterns in your own applications. In addition, the [book’s repository](https://github.com/Nicolepcx/ai-agents-the-definitive-guide) includes complete, runnable examples of six different patterns. You can select among them through a `main()` function, as shown in [Example 2-12](#ch02.html_main_choice).

##### Example 2-12. Main function for HITL pattern demos

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

In the following code examples ([Example 2-13](#ch02.html_b_propose) - [Example 2-20](#ch02.html_wait_for_interrupt_and_prompt)) I’ll explain how to implement the patterns covered above and also cover the main points you’d want to watch out for when you implement them.

# Command in LangGraph

Combines control flow (edges) and state updates (nodes). It facilitates multi-actor (or multi-agent) communication. For example, a node can update its state and decide the next node to visit. LangGraph enables this by returning a Command object from node functions.

Use approve or reject when you want to gate a side-effecting step like an HTTP call.

[Example 2-13](#ch02.html_b_propose) asks the LLM to produce a request plan and stores it in `proposed_request`.

##### Example 2-13. Propose request with LLM

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

[![1](assets/1.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO7-1)
: 

Ask the LLM to generate a structured plan.

[![2](assets/2.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO7-2)
: 

Fallback default request if parsing fails.

[![3](assets/3.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO7-3)
: 

Try to parse JSON out of the model’s output.

[![4](assets/4.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO7-4)
: 

Store result in state under proposed\_request.

[Example 2-14](#ch02.html_b_gate) calls interrupt(…​) to pause and request human approval or a revision.

##### Example 2-14. Approval gate for request

```
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

[![1](assets/1.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO8-1)
: 

Pause execution and wait for human decision.

[![2](assets/2.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO8-2)
: 

Present the proposed request for inspection.

[![3](assets/3.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO8-3)
: 

Enforce structured approval/revision schema.

[![4](assets/4.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO8-4)
: 

On approval, route to the call step.

[![5](assets/5.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO8-5)
: 

On revision, loop back to the gate with updated request.

On `resume` you pass `Command(resume={"action": "approve"})` to continue, or `{"action": "revise", "update": {...}}` to merge edits and loop back to the gate.

[Example 2-15](#ch02.html_b_call) runs the HTTP call only after approval and stores `api_result`. You can reject to prevent execution, and routing depends on human input.

##### Example 2-15. Execute approved HTTP call

```
def b_call(state: BState) -> BState:
    import requests
    r = requests.get(state["proposed_request"]["url"],
    params=state["proposed_request"]["params"], timeout=10) 
    return {"api_result": {"status_code": r.status_code, "url": r.url}} 
```

[![1](assets/1.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO9-1)
: 

Run the external request only after approval.

[![2](assets/2.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO9-2)
: 

Store result under `api_result` in state.

You use review and edit to correct or improve model output by writing directly to the state. Here, [Example 2-16](#ch02.html_c_write) generates a short summary with the LLM and stores it in `summary`.

##### Example 2-16. Generate summary text

```
def c_write(state: CState) -> CState:
    text = LLM.invoke("Write 2 sentences about why human
                     in the loop matters for agents").content 
    return {"summary": text} 
```

[![1](assets/1.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO10-1)
: 

Ask the LLM to produce a candidate summary.

[![2](assets/2.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO10-2)
: 

Store summary in state for later review.

[Example 2-17](#ch02.html_c_edit) pauses with `interrupt(...)` that includes a JSON schema asking for `edited_text`.

##### Example 2-17. Interrupt for human edit

```
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

[![1](assets/1.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO11-1)
: 

Pause and request edit input from human.

[![2](assets/2.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO11-2)
: 

Provide the generated summary for inspection.

[![3](assets/3.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO11-3)
: 

Restrict edits to a string field.

[![4](assets/4.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO11-4)
: 

Overwrite summary with human revision.

You resume with `Command(resume={"edited_text": "your revision"})`. The node restarts and returns your edit instead of pausing, then the graph updates summary with the human version. Meaning, you literally pause to review then overwrite the graph state with the edited content.

You use review tool calls when a tool may affect external systems and needs inspection of arguments. [Example 2-18](#ch02.html_add_hitl) wraps any tool with an `interrupt([...])`. The interrupt payload asks for accept, edit, or respond.

##### Example 2-18. Wrap tool with review gate

```
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

[![1](assets/1.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO12-1)
: 

Wrap original tool call in a review request.

[![2](assets/2.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO12-2)
: 

Pass tool name and arguments for inspection.

[![3](assets/3.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO12-3)
: 

Allow human to accept, edit, or respond.

[![4](assets/4.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO12-4)
: 

Pause until decision is received.

[![5](assets/5.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO12-5)
: 

Execute original tool call if accepted.

[![6](assets/6.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO12-6)
: 

Execute tool with revised arguments.

[![7](assets/7.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO12-7)
: 

Replace tool output with human response.

You can resume with on of:

```
{"type": "accept"}
{"type": "edit", "args": {"args": {"query": "weather in NY"}}}
{"type": "response", "args": "Skip the tool right now"}
```

The wrapper either executes the original tool call, executes with edited args, or returns a human message instead of calling the tool. With that, you inspect and approve tool calls before they hit the outside world.

# Interrupts and side effects

Never put side-effecting code (API calls, DB writes) in the same node as `interrupt(...)`. Always pause first, then execute effects only after explicit approval to avoid repeats when resuming.

There might be also the occasion, that you’d like to get parallel inputs with a single resume. In [Example 2-19](#ch02.html_parallel_interrupts) you see how you can implement this.

##### Example 2-19. Parallel human interrupts

```
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

[![1](assets/1.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO13-1)
: 

Interrupt for first text input.

[![2](assets/2.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO13-2)
: 

Update state with revised `text_1`.

[![3](assets/3.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO13-3)
: 

Interrupt for second text input.

[![4](assets/4.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO13-4)
: 

Update state with revised `text_2`.

[![5](assets/5.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO13-5)
: 

Add both nodes to the graph.

[![6](assets/6.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO13-6)
: 

Both nodes branch directly from START.

[![7](assets/7.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO13-7)
: 

Compile with checkpointer so both interrupts can resume.

Here, both nodes start from `START` and each calls `interrupt(...)`. You use [Example 2-20](#ch02.html_wait_for_interrupt_and_prompt) to build a dictionary that maps `interrupt_id` to its value, and then resume with `Command(resume=that_map)`.

##### Example 2-20. Resume multiple interrupts

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

[![1](assets/1.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO14-1)
: 

Collect any pending interrupts from state.

[![2](assets/2.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO14-2)
: 

Handle multiple interruptions at once.

[![3](assets/3.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO14-3)
: 

Print each interrupt ID and its current value.

[![4](assets/4.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO14-4)
: 

Ask the human for a new value per interrupt.

[![5](assets/5.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO14-5)
: 

Try to parse as JSON for structured edits.

[![6](assets/6.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO14-6)
: 

Keep the original value if no input is given.

[![7](assets/7.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO14-7)
: 

Resume execution with a full map of updated values.

As you’ve seen, there isn’t a single “right” way to add human oversight. You can choose from different patterns depending on the sensitivity of the task and the amount of control you want to keep. I encourage you to experiment with these patterns in the provided notebook, as this makes it easier to see how human oversight can evolve from simple approval gates into richer collaborations between human and agents.

# Advanced Agent Paradigms: Supervisor and Hierarchical Agents

If your tasks start getting more complex than the ones you’ve seen so far, or if you notice your agent struggling, it can help to break the task into smaller subtasks. Think of it like building a team: each person brings a different skill to the table, and together they solve the larger challenge.

In the same way, you can introduce smaller, independent agents that form a MAS. One common setup is the *supervisor architecture*, where every agent reports to a single supervisor that decides who should act next. A more advanced version is the *hierarchical architecture*, where you don’t just have one supervisor, but supervisors of supervisors. This generalization allows for more flexible control flows.

The diagram in [Figure 2-4](#ch02.html_single_vs_supervisor_vs_hierarchical) shows how a single-agent setup compares to these two multi-agent patterns.

![ch02 single vs supervisor vs hierarchical](assets/ch02_single_vs_supervisor_vs_hierarchical.png)

###### Figure 2-4. Comparison of single, supervisor, and hierarchical agent architectures.

The primary benefits of using a MAS are its modularity, specialization, and control. By separating functionality across agents, you make the overall system easier to develop, test, and maintain. Each agent can be designed as a domain expert, which boosts the system’s performance as a whole. And unlike simple function calling, a multi-agent setup also gives you explicit control over how agents communicate with one another.

## Building a Hierarchical Research Team

To make the idea of supervisors and sub-agents concrete, let’s walk through an example where you build a hierarchical research team. I’ll use LangGraph’s prebuilt, reusable `create_react_agent` component to fast-track the implementation of a ReAct agent. Each worker agent then gets a clear role: some focus on web search, others scrape pages, query [Exa](https://exa.ai/) for more structured search results, or even look up patents on Google. A supervisor agent sits on top, coordinating the specialists and deciding who should act next. The end goal is to assemble a market research team. [Figure 2-5](#ch02.html_hierarchical_architecture_graph) shows the final application you’re building.

![ch02 market research team graph](assets/ch02_market_research_team_graph.png)

###### Figure 2-5. Final application with a hierarchical architecture.

But before you get there, you’ll start smaller by building your first supervisor setup: the research team. You see the overview of the team you create in [Figure 2-6](#ch02.html_Flow_diagram_research_team). Once that’s in place, the second step is to add a writing team. This group will take the output from the research team and save it to your file system as a simple text file.

![ch02 research team graph](assets/ch02_research_team_graph.png)

###### Figure 2-6. Flow diagram for the research team.

I’ll leave out the details of how to create tools, since you already learned this in chapter one ([Example 1-2](#ch01.html_createTools)).

The first thing you need is a helper ([Example 2-21](#ch02.html_helper_supervisor)) to create your supervisor. This function acts as a router (see [“Mapping FSM/HSM to Agent Frameworks”](#ch01.html_router_section)) that decides which worker should take the next step, or whether the process should finish.

##### Example 2-21. Helper function for supervisor

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

[![1](assets/1.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO15-1)
: 

Create the state to keep track of the next worker.

[![2](assets/2.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO15-2)
: 

Define a helper that builds a supervisor node for any set of agents.

[![3](assets/3.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO15-3)
: 

Provide a system prompt that tells the LLM how to act as the supervisor.

[![4](assets/4.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO15-4)
: 

Use structured output to reliably decide which worker to call next.

[![5](assets/5.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO15-5)
: 

Allow the workflow to terminate cleanly when the supervisor says `FINISH`.

Next, you need a helper to create your worker nodes, how to implement this is shown in [Example 2-22](#ch02.html_helper_node_creation). This makes the setup concise and reusable, so you don’t have to repeat the same boilerplate each time you add a new agent.

##### Example 2-22. Helper function for node creation

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

[![1](assets/1.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO16-1)
: 

Create a specialized ReAct agent with its tools and role prompt.

[![2](assets/2.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO16-2)
: 

Invoke the agent with the current state.

[![3](assets/3.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO16-3)
: 

Pass the agent’s latest output back into the state and return to the supervisor.

Once you have these helpers, you can move on to defining your agents. It’s helpful to give each agent a specific role. Since the goal here is to build a market research team, the roles should be very targeted. My role prompts are shown in [Example 2-23](#ch02.html_roles_research_agents).

##### Example 2-23. Assign roles to research agents

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

Now, let’s create the agents. Using the helper from [Example 2-22](#ch02.html_helper_node_creation), you can instantiate each worker and assign it the appropriate role and tool. The implementation is shown in [Example 2-24](#ch02.html_create_research_agents).

##### Example 2-24. Create research agents

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

[![1](assets/1.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO17-1)
: 

Define each worker with its tools and role.

[![2](assets/2.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO17-2)
: 

Use the helper to instantiate all workers without repeating boilerplate.

To complete the supervisor architecture, we need to set up a supervisor that can coordinate across all four research agents. This ensures the workflow doesn’t stall and that each agent contributes at the right time. See [Example 2-25](#ch02.html_research_supervisor_node) for the implementation.

##### Example 2-25. Create research supervisor node

```
research_supervisor_node = make_supervisor_node(
    llm, ["search", "web_scraper", "exa_search", "patent_research"]
) 
```

[![1](assets/1.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO18-1)
: 

Create a supervisor that coordinates across all four research agents.

Finally, we need to build the actual graph by registering nodes and connecting the edges that define how control flows between them. This results in a fully functioning research team graph, as shown in [Example 2-26](#ch02.html_research_team_graph).

##### Example 2-26. Build research team graph

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

[![1](assets/1.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO19-1)
: 

Register supervisor and worker nodes.

[![2](assets/2.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO19-2)
: 

Connect edges so all workers route back to the supervisor.

[![3](assets/3.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO19-3)
: 

Compile the graph into an executable workflow.

To test this setup, you can run the code from [Example 2-27](#ch02.html_run_research_team_prompt).

##### Example 2-27. Run research team with prompt

```
for s in research_graph.stream(
    {"messages": [("user", """What are AI agents? Are there any patents out there
                              about LLM agents?""")]}, 
    {"recursion_limit": 100},
):
    print(s)
    print("---")
```

[![1](assets/1.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO20-1)
: 

Run the research graph on a query; the supervisor orchestrates which agents act.

This returns a short description AI agents and then uses the patent tool to check for recent patents. The steps look like this (I’ve left out the actual messages between agents for brevity):

```
{'supervisor': {'next': 'search'}}
{'supervisor': {'next': 'patent_research'}}
{'supervisor': {'next': '__end__'}}
```

You can set up your writing team in the same way. The notebook for this section includes the full working code, but [Figure 2-7](#ch02.html_writing_team_flow_diagram) gives you a quick overview of what the writing team looks like.

![ch02 writing team graph](assets/ch02_writing_team_graph.png)

###### Figure 2-7. Flow-chart for writing team.

This modular approach makes it easy to compose increasingly capable MAS step by step: first by wiring up specialized workers, then by combining entire teams into larger graphs. In practice, this means you can start small—say with search, scraping, and patent lookup in a research team, and then plug that into a writing team to form a full end-to-end pipeline, or extend it even further with internal data and chart writing and plotting agents.

To move from individual teams to a proper hierarchy, you’ll need to build another graph that ties everything together. Think of this as your “super graph,” shown in [Example 2-28](#ch02.html_hierarchical_structure).

##### Example 2-28. Build hierarchical architecture

```
super_builder = StateGraph(State)
super_builder.add_node("supervisor", teams_supervisor_node)
super_builder.add_node("research_team", call_research_team)
super_builder.add_node("writing_team", call_paper_writing_team)

super_builder.add_edge(START, "supervisor")
super_graph = super_builder.compile()
```

Once that’s in place, you can prompt your hierarchical MAS as shown in [Example 2-29](#ch02.html_hierarchical_MAS_run).

##### Example 2-29. Run hierarchical MAS

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

Running this setup will result in the following sequence of supervisor decisions (messages again omitted):

```
{'supervisor': {'next': 'research_team'}}
{'supervisor': {'next': 'writing_team'}}
{'supervisor': {'next': 'writing_team'}}
{'supervisor': {'next': '__end__'}}
```

At the end of the run, the report is successfully saved to `semiconductor_whitepaper.txt`. In practice, this workflow takes only about two minutes to gather the research, generate the text, and write everything to file.

## Developing a Swarm of Agents

Another important paradigm is the swarm architecture. Here, agents handoff control to one another dynamically based on their specializations. Unlike a strict hierarchy, swarms emphasize peer-to-peer collaboration. The system keeps track of which agent was last active so that subsequent interactions continue seamlessly with the right one. This back-and-forth exchange allows work to bounce naturally between agents, such as when a researcher gathers sources and then hands them to a writer, who may in turn request more data before finishing the synthesis. [Figure 2-8](#ch02.html_agent_swarm) illustrates this continuous flow.

![ch02 agent swarm](assets/ch02_agent_swarm.png)

###### Figure 2-8. Handoff flow between a swarm of agents.

LangGraph provides an [open-source library](https://github.com/langchain-ai/langgraph-swarm-py) to fast track the development of swarms of agents. By default, the agents in the swarm use handoff tools created with the prebuilt `create_handoff_tool`. You can also create your own custom handoff tools to better fit your workflow. For example, you might:

*   Change the tool name or description to make the handoff purpose clearer.
    
*   Add tool call arguments for the LLM to populate, such as a specific task description for the next agent.
    
*   Control what data is passed to the next agent. By default, `create_handoff_tool` passes the full message history (all prior agent messages) and a tool message confirming the handoff, but you may choose to send only a filtered subset of context.
    

Before wiring up a full swarm, you first need to define the handoff helpers as shown in [Example 2-30](#ch02.html_handoff_helper). These tools act as signals that let one agent pass control to another. In the example below, the research assistant can handoff to the writer once enough sources are collected, and the writer can hand control back if more evidence is needed.

##### Example 2-30. Create handoff tools

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

[![1](assets/1.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO21-1)
: 

Defines a tool that lets the research assistant handoff to the writing assistant.

[![2](assets/2.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO21-2)
: 

Defines the reverse tool so the writer can send control back to the research assistant.

Once you have the handoff helpers in place, you can move on to defining your agents. It’s helpful to keep the roles very distinct. In this setup you want one agent focused purely on research, and the other focused on writing. The handoff tools ensure the workflow can bounce back and forth when more sources are needed or when it’s time to synthesize. The roles are shown in [Example 2-31](#ch02.html_roles_research_writer).

##### Example 2-31. Assign roles to research and writing assistants

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

[![1](assets/1.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO22-1)
: 

Defines the research assistant, responsible for web search and scraping.

[![2](assets/2.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO22-2)
: 

Grants the research assistant access to its tools, including the handoff to the writer.

[![3](assets/3.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO22-3)
: 

Defines the writing assistant, responsible for synthesis and citation.

[![4](assets/4.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO22-4)
: 

Allows the writer to hand control back to the research assistant if more evidence is needed.

Now that both roles are defined, we can wire them into a swarm. The swarm provides the coordination mechanism: it starts with the research assistant by default, but allows handoff to the writer and back again until the task is complete. The implementation is shown in [Example 2-32](#ch02.html_research_writer_swarm).

##### Example 2-32. Build research–writer swarm

```
swarm = create_swarm(
    agents=[research_assistant, writer_assistant],
    default_active_agent="research_assistant",
).compile()
```

To test the system, you can run the following example prompt. Here, I requested a summary of fintech licensing in Switzerland. The swarm first searches for sources, then hands off to the writing assistant for synthesis, and if needed returns to research for more data. See [Example 2-33](#ch02.html_research_writer_run) for the prompt.

##### Example 2-33. Run research–writer swarm

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

[![1](assets/1.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO23-1)
: 

Defines a user prompt asking for fintech licensing details in Switzerland.

[![2](assets/2.png)](#ch02.html_co_architectures_and_patterns__planning__reactivity__and_multi_agent_systems_CO23-2)
: 

Streams the results from the swarm, showing the handoff-driven workflow in action.

The run produces three clean bullet points with references, citing official Swiss regulator pages and trusted financial sources. However, if the writer finds the evidence too thin, it can trigger a return handoff to research. The full cycle takes just about 30 seconds to gather the information and generate the final structured output.

The supervisor, hierarchical, and swarm architectures from this section show how different coordination strategies affect MAS. Supervisors provide centralized control, hierarchies allow scalable organization into teams of teams, and swarms emphasize flexible peer-to-peer collaboration. [Table 2-4](#ch02.html_pattern_summary_coordination) compares these patterns and highlights when to use each one.

Table 2-4. Pattern summary: Multi-agent coordination strategies

 
| Architecture || When to Use || Benefits || Tradeoffs |
| --- | --- | --- | --- |
| Supervisor || Small teams of agents with clear task boundaries || Simple setup; strong control; avoids deadlocks || Central bottleneck; single point of failure |
| --- | --- | --- | --- |
| Hierarchical || Larger, multi-team workflows requiring scalability || Modular; scalable; mirrors organizational structures || Harder to debug; requires careful design |
| Swarm || Dynamic collaboration between peers (e.g., research ↔︎ writer) || Flexible; natural back-and-forth flow || Less predictable; requires robust handoff design |

This shows how each coordination pattern supports different scales and styles of agent collaboration. By mixing and matching them, you can shape agent workflows that balance control, scalability, and adaptability within your overall MAS design.

# Conclusion

The big takeaway from this chapter is that an AI agent’s intelligence comes also from its architecture, not just from the LLM it uses. In light of this, you explored how architectural patterns shape the intelligence and reliability of AI agents. By moving beyond single-step execution, you saw how reasoning loops like CoT, ToT, and ReAct embed structure into the agent’s thought process, making outputs more deliberate and transparent.

Equally important, you saw how human-in-the-loop can make your agents more secure. Interrupts, approval gates, and review steps demonstrated how guardrails make agents trustworthy in real-world settings where errors carry real consequences. The balance between autonomy and oversight is a defining theme of modern agent design.

You also learned how these reasoning methods extend into MAS. Supervisor and hierarchical architectures distribute tasks across specialized workers, while swarm setups enable flexible handoffs between peers. Together, these paradigms show that agent intelligence is not just about stronger models, but about how those models are orchestrated into systems that can plan, act, and reflect.

With these foundations in place, you are ready to move to the next level. In the following chapter, you will explore advanced paradigms that extend these architectures toward scalable execution, deeper planning strategies, and more sophisticated reasoning methods. This will show you how to design agents that can not only react and reflect, but also plan at scale and coordinate complex workflows across larger systems.

^([1](#ch02.html_id52-marker)) Jason Wei et al. “Chain-of-Thought Prompting Elicits Reasoning in Large Language Models”, [*https://arxiv.org/abs/2201.11903*](https://arxiv.org/abs/2201.11903) (2023).

^([2](#ch02.html_id53-marker)) Shunyu Yao et al. “Tree of Thoughts: Deliberate Problem Solving with Large Language Models.”, [*https://arxiv.org/abs/2305.10601*](https://arxiv.org/abs/2305.10601) (2023).

# About the Author

**Nicole Koenigstein** is the Founder, CEO, and Co-Chief AI Officer of a deep-tech company specializing in large language model (LLM) systems and complex multiagent architectures. She is a globally recognized thought leader in LLMs and agentic architectures, with a particular focus on their transformative applications in quantitative finance. Nicole has served as an external expert for the European Commission on LLM research funding and for the International Organization of Securities Commissions (IOSCO), advising on GenAI adoption in regulated industries. She also serves as an advisory board member for QuantMinds International and QuantStrats, two of the world’s largest and most influential fintech, AI and quant strategy conferences. Over the past 2.5 years, Nicole has delivered more than 50 invited talks, keynotes, and technical workshops globally—including at Oxford University, Cornell University, Bank of America, J.P. Morgan, and the International Faculty of Finance—as well as at international events such as European Women in Technology, ACM Tech Talk, and ODSC Europe.

Nicole is a faculty member at the Artificial Intelligence in Finance Institute in New York, where she teaches coding sessions on Python, machine learning, deep learning, large language models and agentic AI. She is the author of *Math for Machine Learning* and *Transformers in Action* (Manning Publications), and her forthcoming title *Transformers: The Definitive Guide* will be published by O’Reilly Media.