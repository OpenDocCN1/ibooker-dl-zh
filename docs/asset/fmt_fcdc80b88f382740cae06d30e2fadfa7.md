
对于长对话来说，这种内存类型提供了对话的摘要版本，节省了宝贵的令牌空间。

## 交互总结缓冲内存

适用于你不仅想记录最近交互，还想将旧交互汇总成总结的情况，从而提供一种混合方法。

## 词汇缓冲内存

当定义特定令牌长度至关重要且需要维护最近交互的缓冲区时，这种内存类型很有用。它根据令牌长度确定何时刷新交互。

了解可用的不同内存选项可以帮助你根据具体情况选择最适合你的需求的一个。

# 指导方向

即使你在确定要使用哪种内存类型时，也要记住适当地指导 AI 模型。例如，使用`ConversationBufferWindowMemory`，你需要指定你想要保留的最近交互的数量（`K`）。为了获得最佳结果，请明确你的要求。

# 带有内存的 OpenAI 函数代理

通过 GitHub 上的一个综合示例深入了解代理，[GitHub](https://oreil.ly/jyLab)。在这个示例中，你将发现 OpenAI 如何整合几个关键组件：

+   使用聊天消息进行内存管理
+   使用能够处理多个函数参数的工具，如 API 请求和文件保存
+   集成自定义`SystemMessage`以指导和定义代理的行为

为了说明，考虑一下 Python 函数的文档字符串是如何用来提供工具描述的：

```py
from langchain.tools import StructuredTool

def save_interview(raw_interview_text: str):
    """Tool to save the interview. You must pass the entire interview and
 conversation in here. The interview will then be saved to a local file.
 Remember to include all of the previous chat messages. Include all of
 the messages with the user and the AI, here is a good response:
 AI: some text
 Human: some text
 ...
 ---
 """
    # Save to local file:
    with open("interview.txt", "w") as f:
        f.write(raw_interview_text)
    return f'''Interview saved! Content: {raw_interview_text}. File:
 interview.txt. You must tell the user that the interview is saved.'''

save_interview = StructuredTool.from_function(save_interview)
```

`StructuredTool.from_function()`将创建一个 LangChain 工具，该工具能够接受多个函数参数。

# 指导并指定格式

Python 函数中的文档字符串展示了指导 LLM 使用`raw_interview_text`参数内容的指定格式。

此外，`return`语句强调指导 LLM 通知用户访谈已被存储。这确保代理在工具执行后返回更具对话性的响应。

为了进一步展示提示工程技巧，让我们检查笔记本中的另一个 Python 代码片段：

```py
from pydantic.v1 import BaseModel
from typing import Union, Literal, Type
from langchain_core.tools import BaseTool

class ArgumentType(BaseModel):
    url: str
    file_type: Union[Literal["pdf"], Literal["txt"]]

class SummarizeFileFromURL(BaseTool):
    name = "SummarizeFileFromURL"
    description = "Summarize a file from a URL."
    args_schema: Type[ArgumentType] = ArgumentType
```

在这个示例中，`args_schema`在`SummarizeFileFromURL`类中使用。该属性利用`ArgumentType`类，确保在执行之前验证工具的参数。具体来说，它强制提供有效的 URL 字符串，并且`file_type`参数应该是`"pdf"`或`"txt"`。

通过添加验证检查，你可以确保代理正确处理功能参数，这反过来又提高了工具执行的整体可靠性和效率。

# 高级代理框架

你现在已经了解了 ReAct 和 OpenAI 函数，但还有其他几个代理框架。其他两个流行的框架包括*计划和执行代理*和*思维树*。

## 计划和执行代理

而不是让大型语言模型（LLM）进行任务规划和工具执行，你可以将这一过程分为两个独立的模块。每个模块都可以由一个可以访问目标、当前任务和已完成任务的独立 LLM 分别处理。

计划与执行框架的两个流行版本包括[BabyAGI](https://oreil.ly/xeijG)和[AutoGPT](https://oreil.ly/M4z8K)。

图 6-4 展示了 BabyAGI 的智能体设置，该设置旨在将 OpenAI LLM 与 Chroma/Weaviate 等向量数据库合并，以创建一个强大、自适应的任务管理系统。

在一个连续的循环中，智能体首先获取一个任务，并将其传递给`execution_agent`，该智能体利用 OpenAI 根据上下文数据执行任务。之后，结果被丰富并存档在 Chroma/Weaviate 中。

`task_creation_agent`随后介入，利用 OpenAI 从先前任务的目标和结果中识别新任务。这些任务以字典列表的形式呈现，为结果任务提供结构。

然后`prioritization_agent`与 OpenAI 交互，重新排列任务列表，确保与主要目标保持一致。这些智能体的协同作用确保系统始终在进化，以信息化的方式持续生成和优先处理任务。整合[Chroma](https://oreil.ly/9R3pU)或[Weaviate](https://oreil.ly/2wu-y)起着至关重要的作用，因为它提供了一个上下文数据的宝库，确保任务始终与其预定义的目标保持一致。

![BabyAGI](img/pega_0604.png)

###### 图 6-4\. BabyAGI 的智能体架构

[计划与执行智能体类型](https://oreil.ly/8vYF5)确实存在于 LangChain 中，尽管它仍然是实验性的。

## 思维树

随着语言模型在解决各种任务中的应用不断扩大，它们的推理方法仍然局限于基于标记的、线性的处理。这种方法在许多情况下都有效，但在需要高级战略预见或初始决策至关重要的任务面前就有限了。[思维树（ToT）框架](https://oreil.ly/1rYDI)是一种利用语言模型的新方法，它超越了传统的思维链提示技术(图 6-5)。

思维树（ToT）的核心前提是允许在连贯的文本块之间进行探索，这些文本块被称为*思维*。这些思维代表了问题解决的各个阶段，有助于语言模型进行更谨慎的决策过程。模型不再局限于一条推理路径，可以探索各种推理轨迹，并在每一步自我评估其决策。该框架旨在允许进行前瞻性规划、回顾过去的决策，并做出总体选择。

![思维树](img/pega_0605.png)

###### 图 6-5\. 思维树（ToT）

其成功的证据来自于对需要复杂规划或搜索能力的任务的实验结果。在一个像*24 点游戏*这样的游戏中，传统的 GPT-4 在提示使用思维链时，成功率为 4%。相比之下，ToT 方法将这一数字飙升至令人印象深刻的 74%。这种范式转变不仅限于游戏。ToT 方法在创意写作和迷你填字游戏等领域的应用也显示出其多功能性。

与理论相辅相成的是[LangChain 实现](https://oreil.ly/fub1z)，它展示了 ToT 如何实现。一个数独谜题作为说明性示例，主要目的是用数字替换通配符字符(*)，同时遵守数独规则。

ToT 不仅仅是一种新的方法；它是在我们设想语言模型推理方面的范式转变。通过为模型提供思考、回溯和策略规划的能力，ToT 正在重新定义 AI 问题解决的边界。

如果你将 ToT 视为指挥 LLMs 的策略，LangChain 回调可以被视为诊断和确保这些策略顺利运行的工具。让我们深入了解如何有效地利用这一功能。

# 回调

LangChain 的[*回调*](https://oreil.ly/8EhXl)使您能够无缝监控和定位应用程序中的问题。到目前为止，您在`AgentExecutor`链中遇到了`verbose=True`参数：

```py
AgentExecutor(.., verbose=True)
```

此参数记录了用于调试的有用输出，但如果你热衷于跟踪特定事件呢？请进入回调，这是你的首选解决方案。

`BaseCallbackHandler`类在您的生成式 AI 模型执行期间充当监控和响应各种事件的基础。该类中的每个方法都对应于特定的阶段，如模型的开始、结束，甚至在模型运行时发生的错误。例如，当 LLM 开始其操作时，会触发`on_llm_start`。同样，`on_chain_error`和`on_tool_end`等方法会对链中的错误或使用工具后的错误做出反应：

```py
class BaseCallbackHandler:
    """Base callback handler that can be used to handle callbacks from
 langchain."""

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str],
        **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""

    def on_chat_model_start(
        self, serialized: Dict[str, Any],
        messages: List[List[BaseMessage]], **kwargs: Any
    ) -> Any:
        """Run when Chat Model starts running."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any],
        **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
```

每个回调都可以作用域为类或单个请求。

## 全局（构造函数）回调

在构造函数中定义回调，例如`AgentExecutor(callbacks=[handler], tags=['a-tag'])`，它们会在对该实例的每次调用时被激活。这些回调仅限于该特定实例。为了说明，当一个处理程序在创建`LLMChain`时被传递，它不会与任何子链交互：

```py
from langchain.agents import AgentExecutor
from langchain.callbacks import StdOutCallbackHandler

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    callbacks=[StdOutCallbackHandler()],
    tags=['a-tag'])

agent_executor.invoke({"input": "How many letters in the word Software?"})
```

你包含的标签，如`'a-tag'`，在追踪和整理您的生成式 AI 设置输出方面非常有用。特别是在具有许多链的大型项目中，利用标签可以显著简化您的流程。

## 请求特定回调

另一方面，回调可以在`invoke()`方法中定义。例如，对`LLMChain`的请求可能会随后触发另一个`LLMChain`请求，并应用相同的处理程序：

```py
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

handler = StdOutCallbackHandler()
llm = OpenAI()
prompt = PromptTemplate.from_template("What is 1 + {number} = ")
chain = LLMChain(llm=llm, prompt=prompt)
chain.invoke({"number": 2}, {"callbacks": [handler]})
```

## 详细说明论点

常用实用工具`verbose`参数对大多数 API 对象都是可用的。当您使用`AgentExecutor(verbose=True)`时，它与将`ConsoleCallbackHandler`集成到对象的回调参数及其后代中相同。它通过直接将每个事件记录到您的控制台而充当有用的调试工具。

## 何时使用哪个？

### 构造函数回调

适用于跨整个链的日志记录或监控等全局任务。如果您目标是跟踪代理的所有交互，请在初始化期间附加处理程序。

### 请求回调

定制用于特定用例，如流式传输，其中单个请求的输出被转发到专用端点，例如 WebSocket。因此，对于需要将单个请求的输出流式传输到 WebSocket 的场景，处理程序应链接到`invoke()`方法。

### 冗长参数

对于调试和本地 LLM 开发很有用，但它可以生成大量的日志。

## 使用 LangChain 进行令牌计数

LangChain 在您与生成式 AI 模型交互期间提供了有效的令牌计数方法。

您需要设置必要的模块；导入`asyncio`模块和 LangChain 包中的相关函数：
