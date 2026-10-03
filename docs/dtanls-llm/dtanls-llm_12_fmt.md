
# 第十章：个软件框架

## 本章涵盖

+ 使用 LangChain 构建应用
+ 使用代理解决复杂任务
+ 使用 LlamaIndex 查询数据

到目前为止，我们主要使用 OpenAI 的 Python 库与语言模型进行交互。这个库提供了向 GPT 和其他 OpenAI 模型发送提示和检索答案的基本功能（以及调整和微调的选项）。来自其他提供商的库，如 Anthropic 和 Cohere，提供类似的功能。只要你的数据分析任务简单，这可能就是你所需要的。然而，如果你的数据分析需要复杂的、多步骤的流水线，可能需要整合许多不同的数据格式呢？

到那时，你可能想切换到一个更强大的软件框架。目前，有几个用于在语言模型之上构建复杂应用的更高层次框架正在兴起。在本章中，我们将讨论两个最受欢迎的竞争者：LangChain 和 LlamaIndex。前者是一个用于构建使用大型语言模型的应用的通用框架。更重要的是，它附带各种有用的内置组件，实现了语言模型的流行用例。另一方面，LlamaIndex 专门支持语言模型需要与大型数据集交互的用例。

为了熟悉这个工具，我们首先将使用 LangChain 编写一个简单的文本分类流水线。然后我们将探索 LangChain 的一些高级功能。更确切地说，我们将看到 LangChain 如何在语言模型之上支持*代理*。创建一个代理意味着将语言模型本身置于驾驶员的位置，给它在完成特定任务时使用用户提供的工具集的自由。我们将使用这样的代理独立解决复杂的数据分析任务，使用各种工具访问不同的数据源。接下来，我们将看到 LlamaIndex 如何轻松地摄取大量不同格式的数据，并使其对语言模型可用。内部，它使用廉价的语言模型将数据片段和分析任务映射到向量表示，然后根据这些向量的相似性将任务映射到数据。最后，我们将比较这两个框架，并讨论这些框架与 OpenAI 和其他语言模型提供商提供的库之间的权衡。

## 10.1 LangChain

如果你想要创建一个基于语言模型的复杂应用，你可能应该检查一下 LangChain。该框架于 2022 年 10 月推出，并迅速获得了人气（导致 2023 年 4 月创建了一个相应的初创公司）。在撰写本文时，LangChain 仍在快速发展。请确保使用正确的 LangChain 版本运行本节中的代码（因为未来的版本可能会更改接口）。

如其名所示，LangChain 与语言模型（Lang）和链（Chain）相关。在 LangChain 术语中，一个*chain*只是一个步骤序列。每个步骤可能对应于调用一个语言模型、一个数据处理步骤或调用一个任意工具。这里的重要点是，我们不再假设单次调用语言模型就能解决我们的问题（这在本书中讨论的大多数场景中都是如此）。相反，我们假设我们需要一个复杂的连接组件网络。这正是 LangChain 大放异彩的场景！

要使用 LangChain，你首先需要安装它。打开终端，并运行以下命令：

```py
pip install langchain==0.1.13
```

正如我们提到的，如果你想运行以下代码示例，你需要安装正确的 LangChain 版本！LangChain 目前正在快速变化，所以代码可能无法与不同版本兼容。

除了 LangChain 的核心功能之外，你可能还需要安装支持特定提供商语言模型的库。在接下来的章节中，我们将使用 OpenAI 的模型。请在终端中运行以下命令（并且，再次提醒，请确保使用指定的版本）：

```py
pip install langchain-openai==0.1.1
```

对其他提供商的支持，如 Anthropic 和 Cohere，同样可用。

好的，这就完成了！运行这些命令后，你就可以运行下一节中讨论的示例项目了。

## 10.2 使用 LangChain 对评论进行分类

我们最早的项目之一是使用语言模型进行文本分析。还记得第四章吗？我们使用语言模型根据潜在的情感（这是推荐还是警告？）对评论进行分类。在这里，我们也将做同样的事情；我们只是会在代码中使用 LangChain。将 LangChain 的代码与原始代码进行比较，应该能给你一个 LangChain 如何帮助简化使用语言模型构建应用程序的第一印象。

### 10.2.1 概述

我们将创建一个用于分类文本文档的链。一个 LangChain 链可能涉及许多步骤，每个步骤通过调用一个语言模型或一个通用的 Python 函数（例如，将语言模型调用的结果解析成标准格式）来实现。术语*chain*实际上有些误导。虽然你可能想象链是一个连续步骤的序列，但 LangChain 中的链要强大得多。例如，它们可能涉及并行步骤以及条件执行。然而，对于简单的文本分类应用，我们不需要这样的高级功能。相反，我们将限制自己使用只有几个步骤的简单链。

我们的链将集成 LangChain 提供的几个标准组件。我们链中的第一个组件是一个提示模板。正如第四章中所述，这个模板描述了分类任务和预期的输出格式。你可能想知道与之前的代码版本相比有什么变化。毕竟，我们一直在讨论提示模板。区别在于 LangChain 引入了一个专门用于表示提示模板的类。这个类为提示模板提供了各种便利函数：例如，用于创建和实例化它们。同时，LangChain 提供了一个中心，允许用户上传和下载提示模板（以及许多其他组件）。在我们的简单场景中，我们不需要这些高级功能。相反，我们只需要通过传递一个参数（要分类的文本）来实例化我们的提示模板。

我们链中的第二步是一个语言模型。同样，我们在这本书的整个过程中一直在使用语言模型，但 LangChain 在语言模型对象之上增加了几个有用的功能。例如，自动记录所有语言模型调用变得很容易，LangChain 还提供了针对不同调用场景的便利函数（例如，批处理和流处理）。再次强调，我们在这里不会使用这些高级功能。相反，我们将提示（我们链中的第一步）传递给语言模型以生成回复。

我们链中的第三步是一个解析器，它从语言模型生成的回复中提取答案字符串。你可能还记得第三章中提到的，OpenAI 的语言模型会生成详细的回复，其中包含一个或多个答案以及各种类型的元数据（例如，关于标记使用的相关信息）。解析器会自动从结果对象中提取我们所需的答案字符串（这对于 OpenAI 模型以及所有其他提供商都适用）。管道的结果是一个表示输入评论是否为推荐的单一标记。图 10.1 展示了这个管道的三个步骤。

![figure](img/CH10_F01_Trummer.png)

##### 图 10.1 LangChain 分类链中的组件

### 10.2.2 创建分类链

是时候用 Python 实现我们的链了！首先我们需要一个提示模板。我们使用与第四章相同的模板，但这次我们使用 LangChain 的`ChatPromptTemplate`类：

```py
from langchain_core.prompts.chat import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template(
    '{text}\n'                         #1
    'Is the sentiment positive or negative?\n'
    'Answer ("Positive"/"Negative")\n')
```

#1 文本占位符

你可能会注意到对聊天模型的引用（毕竟，我们实例化的类叫做`ChatPromptTemplate`）。在第三章中讨论过，聊天模型处理的是一系列先前的消息历史，而不是单个输入消息。许多最近发布的模型都是聊天模型。在 LangChain 中，聊天模型需要一个专门的提示模板（它实例化为一系列消息而不是单个文本）。这正是我们在这里创建的模板类型。该模板与第四章中使用的模板相同，它包含一个用于对输入文本进行分类的占位符（**1**）。我们通常使用花括号（`{}`）在提示模板中标记占位符；在实例化提示时，它们会被具体的值替换。

第二，我们需要一个语言模型来处理提示。以下代码实例化了 OpenAI 的 GPT-4o 模型：

```py
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model='gpt-4o', temperature=0, 
    max_tokens=1)
```

`ChatOpenAI`类涵盖了 OpenAI 的所有聊天模型。它从`langchain_openai`包中导入，具有在 LangChain 中使用 OpenAI 模型的功能。其他提供商，如 Anthropic 和 Cohere，也有它们自己的相关包，为它们的模型提供类似的功能（注意，你需要通过 pip 分别安装这些包）。`ChatOpenAI`构造函数中的参数可能看起来很熟悉：我们选择模型（`gpt-4o`），将`temperature`设置为 0（以减少输出中的随机程度），并将最大输出令牌数限制为 1（因为可能的两个类别标签，`Positive`和`Negative`，都由单个令牌组成）。

第三，我们需要从语言模型的（更详细的）回复中提取答案字符串。使用`StrOutputParser`来做这件事很简单。LangChain 输出解析器实现了一系列对模型调用输出的转换。在这种情况下，我们只需要一个非常简单的转换，提取所需的答案字符串。以下代码创建了一个相应的解析器：

```py
from langchain_core.output_parsers.string import StrOutputParser
parser = StrOutputParser()
```

最后，我们将所有组件串联在一起。为此，我们可以使用 LangChain 表达式语言（LCEL）。如果你是 Linux 用户，以下语法应该对你来说很熟悉：

```py
from langchain_core.runnables.passthrough import RunnablePassthrough
chain = ({'text':RunnablePassthrough()} | prompt | llm | parser)
```

要将操作的输出用作后续步骤的输入，我们使用管道符号（`|`）将它们连接起来。该命令创建了一个连接先前提到的组件的链。此外，它还指定了链期望的输入。在我们的例子中，提示模板有一个用于对文本进行分类的占位符。



在链的开头，我们将此参数标记为`RunnablePassthrough`。这为我们如何向链传递输入提供了很大的灵活性。例如，以下代码演示了如何使用先前创建的链处理输入列表：

```py
输入 = ['这部电影太棒了！', '这部电影太糟糕了！']
输出 = chain.batch(inputs)
```

### 10.2.3 整合

是时候完成我们的文本分类代码了！清单 10.1 中的代码将包含`text`列的`.csv`文件路径作为输入。执行代码会生成一个结果文件，其中包含一个名为`class`的附加列，用于存放分类结果。换句话说，该代码与第 4 章中的代码功能完全相同，但这次使用的是 LangChain。

##### 清单 10.1 使用 LangChain 进行情感分类

```py
从 langchain_openai 导入 ChatOpenAI
从 langchain_core.prompts.chat 导入 ChatPromptTemplate
从 langchain_core.output_parsers.string 导入 StrOutputParser
从 langchain_core.runnables.passthrough 导入 RunnablePassthrough
导入 argparse
导入 pandas as pd

def create_chain():                         #1
    """
    创建用于文本分类的链。
    
    返回值:
    用于文本分类的链。
    """
    prompt = ChatPromptTemplate.from_template(  #2
        '{text}\n'
        '情感是积极还是消极？\n'
        '回答 ("Positive"/"Negative")\n')
    llm = ChatOpenAI(              #3
        model='gpt-4o', temperature=0,
        max_tokens=1)
    parser = StrOutputParser()  #4
    #5
    chain = ({'text':RunnablePassthrough()} | prompt | llm | parser)
    return chain

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help='输入文件路径')
    args = parser.parse_args()
    
    df = pd.read_csv(args.file_path)  #6
    chain = create_chain()            #7
    results = chain.batch(list(df['text']))  #8
    df['class'] = results  #9
    df.to_csv('result.csv')
```

#1 创建链
#2 创建提示模板
#3 创建 LLM 对象
#4 创建输出解析器
#5 创建链
#6 读取数据
#7 创建链
#8 使用它
#9 存储输出

`create_chain`函数（**1**）实现了上一节讨论的步骤。它为分类生成一个提示模板（**2**），然后创建一个聊天模型（**3**），最后创建一个输出解析器（**4**）。结果是一个连接所有这些组件的链（**5**）。

在读取命令行参数后，代码读取输入数据（**6**），创建相应的链（**7**），最后将链应用于输入文本列表（**8**）。分类结果被添加到输入数据中并存储在磁盘上（**9**）。

### 10.2.4 尝试运行

是时候尝试一下了！像往常一样，您可以在本书配套网站的第十章部分找到清单 10.1 的代码。下载代码（`listing1.py`文件），以及（可选）一个包含待分类评论的文件（例如第 4 章的`reviews.csv`）。打开终端，切换到包含代码的文件夹。假设`reviews.csv`位于同一文件夹中，运行以下命令：

```py
python listing1.py reviews.csv
```

检查包含代码的文件夹。您应该会看到一个新文件`result.csv`，其中包含所需的分类结果。到目前为止，我们只验证了我们可以使用 LangChain 完成与直接使用 OpenAI 库相同的事情（尽管可以说，LangChain 代码更简洁）。在下一节中，我们将看到 LangChain 使我们能够做的远不止这些。

## 10.3 智能体：让大语言模型掌舵

到目前为止，您可能已将语言模型视为（高度复杂的）工具。基于您的输入，语言模型产生输出。如果数据处理需要的不仅仅是语言模型所能完成的，那么由您（开发人员）来添加必要的基础设施。例如，假设您正在为数学问题构建一个问答系统。意识到语言模型不擅长计算（具有讽刺意味的是，对于计算机程序而言），您可能会考虑以下方法：基于用户的问题，语言模型将输入转换为数学公式。然后，该公式由简单的计算器工具解析和评估。该工具的输出发送给用户。

到目前为止，一切顺利。当您拥有不止一个而是多个数学工具时，情况会变得更加复杂。也许一个工具解微分方程，另一个工具评估简单的算术方程。在这种情况下，您可以通过一个分类阶段来扩展您的方法，将用户输入映射到最合适的工具。然而，在回答用户问题可能需要多次调用不同工具（可能使用一个工具的输出作为下一次调用的输入）的情况下，这种方法就会失效。在这种情况下，手动覆盖所需的每种可能的工具调用序列根本不可行。

这就是*智能体*变得有用的用例类型。智能体是一种相当新颖的使用大语言模型的方式。这种方法的核心是视角的转变。我们不再将语言模型视为开发人员设计的管道中的一个步骤所使用的工具，而是将语言模型作为一个独立的智能体。我们不是试图编排语言模型和其他处理步骤的应用顺序（我们在上一节中做了这件事），而是让语言模型来决定应用哪些处理步骤以及以何种顺序应用。这种方法的优点是它更加灵活，使我们作为开发人员不必预先预见每种可能的发展并在我们的处理逻辑中创建相应的分支。

智能体对于复杂的数据分析任务非常有用，在这些任务中，预先不清楚可能需要哪些数据源或处理方法才能满足用户请求。智能体方法有两个核心术语，我们接下来将介绍它们：*智能体*及其*工具*。

让我们从讨论工具开始。工具可以封装任意功能。它是一个语言模型在认为必要时可以使用的函数。当我们使用 LangChain 或类似框架实现智能体时，工具通常实现为 Python 函数。每个工具都必须关联一个自然语言描述。此描述作为提示的一部分显示给语言模型。基于此描述，语言模型可以决定某个工具在给定上下文中是否有帮助。要使用工具，语言模型需要输入参数和输出语义的描述。与人类程序员类似，选择有意义的参数名称和编写精确的文档有助于语言模型有效地使用工具。由于智能体是通过语言模型实现的，所有可用工具的完整描述通常作为输入提示的一部分提供。

智能体在需要解决用户指定的复杂任务时使用工具。智能体通过语言模型实现。虽然微调可以提高语言模型作为智能体的性能，但通用模型原则上应该可以工作。将语言模型转变为智能体的秘诀与其说在于模型本身，不如说在于提示它的方式。至少，相应的提示集成了以下组件：

*   智能体应解决的高级任务描述。此描述由用户提供。
*   可用工具列表，以及它们的功能和输入输出参数的描述。
*   预期输出格式的描述。这使得可以将语言模型的输出映射到工具调用。

给定这样的提示，语言模型可以产生请求特定工具调用的输出。实现智能体方法的基础设施解析输出，将其映射到相应的工具和输入参数值，并获取调用结果。在下一次迭代中，工具调用的结果被添加到输入提示中。通过这种方式，语言模型基本上可以*访问*工具调用的结果。基于此，语言模型可以选择应用更多工具（可能使用先前工具调用的结果作为输入），或者在获得最终答案时终止。

图 10.2 总结了这一过程。用户指定的任务以及所有工具的详细描述构成语言模型的输入。语言模型的输出被解析并映射到一个动作。该动作要么代表工具调用（在这种情况下，调用命令包含工具所有输入参数的值），要么代表终止（在这种情况下，终止命令包含语言模型认为是输入任务答案的内容）。如果动作是工具调用，则执行相应的调用。结果被添加到下一次迭代使用的提示中。迭代持续进行，直到语言模型决定终止（或达到用户指定的迭代次数限制）。

![figure](img/CH10_F02_Trummer.png)

##### 图 10.2 使用语言模型作为智能体。给定描述任务和可用工具的提示，语言模型决定终止和工具调用。工具调用的结果被添加到下一次迭代使用的提示中。

此时，您可能想知道相应的提示是什么样子的。让我们检查一下 LangChain 中用于智能体的标准提示模板。您可以从 LangChain 的 hub 下载提示模板。如果您想这样做，请先在终端中使用以下命令安装 hub：

```py
pip install langchainhub==0.1.15
```

然后在 Python 中运行以下代码，打印出最流行的智能体类型之一的标准模板：

```py
从 langchain 导入 hub
prompt = hub.pull('hwchase17/react')
print(prompt.template)
```

您应该会看到以下输出：

```py
#1
尽可能回答以下问题。您可以使用以下工具：

{tools}
#2

使用以下格式：
#3

问题：您必须回答的输入问题
Thought：您应该始终思考要做什么
Action：要采取的操作，应该是 [{tool_names}] 之一
Action Input：操作的输入
Observation：操作的结果
... (此 Thought/Action/Action Input/Observation 可以重复 N 次)
Thought：我现在知道最终答案
Final Answer：原始输入问题的最终答案

开始！

问题：{input}    #4
Thought:{agent_scratchpad}  #5
```

#1 一般场景
#2 工具描述
#3 格式描述
#4 用户输入
#5 先前结果

此提示模板描述了总体场景（**1**）（有一个需要回答的问题）、可用工具（**2**）以及解决任务的流程（**3**）。提示模板包含多个占位符，代表工具描述（**2**）、用户输入（**4**）以及先前迭代的结果（**5**）。正如我们将在以下各节中看到的，LangChain 提供了各种便捷函数，用于基于此及类似的提示模板创建和执行智能体。

## 10.4 构建数据分析智能体

在本节中，我们将使用 LangChain 构建一个用于数据分析的智能体。该智能体将能够访问包含结构化和非结构化数据的不同数据源。更重要的是，智能体将决定访问哪些源以及以何种顺序访问。它甚至可能使用从一个源获得的信息来查询第二个源（例如，访问关于视频游戏销量的结构化数据库，以确定特定年份最畅销的游戏，然后使用游戏标题查询网络以获取更多信息）。

### 10.4.1 概述

我们的数据分析智能体实现了我们在上一节讨论的方法。它使用语言模型来决定调用哪些工具、以何种顺序以及使用什么输入参数。在我们的示例场景中，我们将为智能体提供访问关系数据库（以及获取其结构信息，例如可用表的名称）的工具。我们还为智能体提供了一个支持网络搜索的工具（在后台利用现有的搜索引擎）。总而言之，我们得到一个可以查询关系数据库并使用网络获取与数据库内容相关的信息的智能体。

让我们从更详细地描述我们将提供给智能体的工具开始讨论。总的来说，智能体将可以访问以下五个工具：

*   `sql_db_list_tables` 列出关系数据库中的所有表。
*   `sql_db_schema` 给定表名，返回表的 SQL 模式。
*   `sql_db_query_checker` 使智能体能够验证 SQL 查询。
*   `sql_db_query` 评估 SQL 查询并返回查询结果。
*   `search` 使智能体能够通过关键字搜索网络，返回网页文本。

前四个工具帮助智能体访问关系数据库。最后一个工具使智能体能够从网络检索信息。给定用户指定的任务，智能体（使用底层语言模型）决定调用这些工具中的哪些以及以何种顺序调用。图 10.3 说明了此场景。

![figure](img/CH10_F03_Trummer.png)

##### 图 10.3 数据智能体使用多个工具来探索结构并查询关系数据库。此外，智能体可以通过网络搜索工具检索网页文本。

### 10.4.2 使用 LangChain 创建智能体

使用 LangChain 创建智能体非常快！LangChain 甚至为访问结构化数据库的智能体提供了专门的构造函数。我们将在以下代码中使用这些功能。

智能体是通过语言模型实现的。要创建智能体，我们首先必须创建一个语言模型对象：

```py
从 langchain_openai 导入 ChatOpenAI
llm = ChatOpenAI(
    temperature=0, model='gpt-4o')
```

我们正在创建一个聊天类型的 OpenAI 语言模型。更准确地说，我们再次引用了 GPT-4o 模型。

接下来，我们创建一个表示关系数据库的对象。我们将查询存储在磁盘上的 SQLite 数据库。假设`dbpath`存储相应数据库文件的路径（通常，此类文件具有`.db`后缀）。我们可以使用以下代码创建数据库对象：

```py
从 langchain_community.utilities.sql_database 导入 SQLDatabase
db = SQLDatabase.from_uri(f'sqlite:///{dbpath}')
```

我们提到了四个用于访问关系数据库的工具。幸运的是，所有这些工具都将从数据库对象自动创建。但是，我们仍然需要创建一个用于网络搜索的工具。

我们将使用 LangChain 的内置组件 SerpAPI 工具。要使用此工具，您首先需要在 SerpAPI 网站上注册一个帐户。打开浏览器，转到[`serpapi.com/`](https://serpapi.com/)，单击“注册”按钮，并创建相应的帐户。要执行接下来提供的代码，您需要检索您的 API 访问密钥（可在[`serpapi.com/dashboard`](https://serpapi.com/dashboard)获取）。您还需要安装一个 LangChain 扩展以启用网络搜索工具。转到终端，运行以下命令：

```py
pip install google-search-results==2.4.2
```

之后，只需以下 Python 代码片段即可生成网络搜索工具（假设`llm`包含先前创建的语言模型对象，`serpaikey`包含 SerpAPI 访问密钥）：

```py
从 langchain.agents.load_tools 导入 load_tools
extra_tools = load_tools(
    ['serpapi'], serpapi_api_key=serpaikey, llm=llm)
```

`load_tools`函数用于标准工具，通过将所需工具的名称作为参数传递。在这种情况下，我们只需要网络搜索工具，并且我们在工具名称列表中只传递了一个条目（`serpapi`）。调用`load_tools`后，我们将结果存储在`extra_tools`中：一个包含单个条目（网络搜索工具）的工具列表。我们现在拥有了使用 LangChain 创建智能体所需的所有组件。

假设`db`包含先前创建的数据库对象，`llm`包含先前生成的语言模型。我们使用以下代码初始化一个用于基于 SQL 的数据访问的智能体：

```py
从 langchain_community.agent_toolkits.sql.base 导入 create_sql_agent
agent = create_sql_agent(
    llm=llm, db=db, verbose=True,
    agent_type='openai-tools',
    extra_tools=extra_tools)
```

`create_sql_agent`命令是 LangChain 提供的便捷函数，用于创建基于 SQL 的数据访问智能体。前面提到的四个用于关系数据库访问的工具（用于检索表名、显示表模式、验证 SQL 查询以及最终执行查询）会自动添加，无需我们显式添加。除了 SQL 相关工具之外，我们只需要一个额外的工具：网络搜索功能。此类工具通过`extra_tools`输入参数在列表中指定。将`verbose`标志设置为`True`使我们能够跟踪导致智能体调用特定工具的“思考过程”（我们稍后将看到一些示例输出）。智能体类型（本例中为`openai-tools`）决定了要使用的精确提示以及用于将语言模型输出映射到工具调用的解析器。

创建智能体后，我们使用以下代码将智能体应用于特定任务（我们假设变量`task`存储了我们想要解决的任务的自然语言描述）：

```py
agent.invoke({'input':task})
```

### 10.4.3 数据分析智能体的完整代码

清单 10.2 将所有内容整合在一起：在从命令行读取 SerpAPI API 访问密钥、数据库文件路径和问题后，它创建一个语言模型对象（**1**），然后创建一个数据库（**2**）、网络搜索工具（**3**），最后创建智能体（**4**）。它在输入问题上调用智能体（**5**）。智能体产生的输出以对该问题的回答结束（或者如果智能体无法找到答案，则以失败消息结束）。

##### 清单 10.2 具有网络搜索功能的数据分析智能体

```py
导入 argparse
from langchain.agents.load_tools 导入 load_tools
from langchain_community.utilities.sql_database 导入 SQLDatabase
from langchain_community.agent_toolkits.sql.base 导入 create_sql_agent
from langchain_openai 导入 ChatOpenAI

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('serpaikey', type=str, help='SERP API 访问密钥')
    parser.add_argument('dbpath', type=str, help='SQLite 数据库路径')
    parser.add_argument('question', type=str, help='要回答的问题')
    args = parser.parse_args()
    
    llm = ChatOpenAI(              #1
        temperature=0, model='gpt-4o')
    #2
    db = SQLDatabase.from_uri(f'sqlite:///{args.dbpath}')
    extra_tools = load_tools(                         #3
        ['serpapi'], serpapi_api_key=args.serpaikey, llm=llm)
    agent = create_sql_agent(    #4
        llm=llm, db=db, verbose=True,
        agent_type='openai-tools',
        extra_tools=extra_tools)
    agent.invoke({'input':args.question})  #5
```

#1 创建 LLM 客户端
#2 创建数据库对象
#3 添加网络搜索工具
#4 创建智能体
#5 使用输入调用智能体

### 10.4.4 尝试运行

让我们看看这在实践中是如何工作的！从本书的配套网站下载清单 10.2 的代码。除了代码之外，您还需要一个 SQLite 数据库来尝试数据智能体。我们将使用第 5 章中的 SQLite 数据库，其中存储了关于视频游戏的信息（您可以在本书配套网站的 Games SQLite 链接下找到相应文件）。

打开终端，切换到包含代码的目录。我们假设数据库文件`games.db`位于同一目录中。运行以下代码（将`[SerpAPI key]`替换为您的搜索密钥，可在[serpapi.com/dashboard](https://serpapi.com/dashboard)获取）：

```py
python listing2.py [SerpAPI 密钥] games.db ↪ '2016 年最畅销的游戏是什么，如何玩？'
```

您应该会看到类似以下的输出（由于网络内容变化、GPT-4o 模型的微小变化以及其他一些因素，您看到的输出可能略有不同）：

```py
[1m> 正在进入新的 SQL 代理执行器链...[0m
[32;1m[1;3m
  #1 调用：`sql_db_list_tables` with `{'tool_input': "}` [0m[38;5;200m[1;3mgames[0m[32;1m[1;3m
  #2 调用：`sql_db_schema` with `{'table_names': 'games'}`  [0m[33;1m[1;3m
CREATE TABLE games (
    rank INTEGER, 
    name TEXT,
    platform TEXT,
    year INTEGER,
    genre TEXT,
    publisher TEXT,
    americasales NUMERIC,
    eusales NUMERIC,
    japansales NUMERIC,
    othersales NUMERIC,
    globalsales NUMERIC
)
/* 3 rows from the games table:
rank    name    platform    year    genre    publisher    americasales    eusales    japansales    othersales    globalsales
1    Wii Sports    Wii    2006    Sports    Nintendo    41.4900000000     29.0200000000    3.7700000000    8.4600000000    82.7400000000
2    Super Mario Bros.   NES   1985   Platform   Nintendo    29.0800000000     3.5800000000    6.8100000000    0.7700000000    40.2400000000
3    Mario Kart Wii    Wii    2008    Racing    Nintendo    15.8500000000     12.8800000000    3.7900000000    3.3100000000    35.8200000000
*/[0m[32;1m[1;3m
  #3 调用：`sql_db_query_checker` with `{'query': 'SELECT name FROM games WHERE year = 2016 ORDER BY globalsales DESC LIMIT 1'}` responded: The games table contains the information we need.  I will query for the game with the highest global sales in 2016. [0m[36;1m[1;3mSELECT name FROM games WHERE year = 2016  ORDER BY globalsales DESC LIMIT 1[0m[32;1m[1;3m
  #4 调用：`sql_db_query` with `{'query': 'SELECT name FROM games WHERE year = 2016 ORDER BY globalsales DESC LIMIT 1'}` [0m[36;1m[1;3m[('FIFA 17',)][0m[32;1m[1;3m
  #5 调用：`Search` with `How to play FIFA 17` [0m[33;1m[1;3m["A Beginner's Guide To Complete FIFA 17 Domination ... The main steps you should take are to jump right in with a quick play game. ... EA Sports FIFA ...", 'Play FIFA 17 up to 5 days before launch for a full  10 hours when you join EA Access on Xbox One and Origin Access on PC.', "1\. Shield the ball in 360 degrees · 2\. Use Driven Shots and Driven Headers ·  3\. Use set piece upgrades to score with style · 4\. Make Fifa 17's ...", 'Play FIFA 17 as much as you want with EA Access or Origin Access  for only $4.99 per month. Now available in The Vault.', 'Cautiously Start An Online Match. Score Early After Some Self-Proclaimed  Beautiful Build Up Play. Concede 4 Goals In A Row And Convince ...', 'FIFA 17 TUTORIALS & ULTIMATE TEAM ➞  Twitter: https://twitter.com/KrasiFIFA ➞  Instagram: http://instagram.com/KrasiFIFA How I record my ...', "Draft mode is another way to play FIFA Ultimate Team,  giving you the ability to play with Players you don't own.  You'll have the opportunity to draft a random ..."]  #6 [0m[32;1m[1;3m2016 年最畅销的游戏是 FIFA 17。  
要玩 FIFA 17，您可以遵循以下步骤：  
1\. 快速开始游戏。 2\. 用 360 度保护球。 3\. 使用驱动射门和驱动角球。 4\. 使用定位球升级以优雅地得分。 5\. 谨慎地开始在线比赛。 6\. 在一些自称为美丽的 buildup play 之后尽早得分。 7\. 拟定模式是另一种玩 FIFA Ultimate Team 的方式，让您能够与您不拥有的球员一起玩。  
您将有机会挑选一支随机队伍。  
记住，熟能生巧！[0m
[1m> 完成链。[0m
```

#1 智能体检索表列表。
#2 智能体检索表模式。
#3 智能体验证 SQL 查询。
#4 智能体查询顶级游戏。
#5 智能体搜索有关 FIFA 17 的网络。
#6 智能体制定最终答案。

请记住，我们将智能体的输出切换到了详细模式。这意味着输出包含智能体调用的工具的完整日志，以及智能体的推理过程。让我们仔细看看输出来了解发生了什么。

首先，智能体检索关系数据库中可用表的列表（使用`sql_db_list_tables`）（**1**）。显然，当面对一个新数据库时，这是一个合理的步骤。工具调用的结果显示数据库只包含一个表（名为`games`）。智能体对表的内容变得“好奇”。它调用`sql_db_schema`工具来获取有关`games`表的更多信息（**2**）。请注意，此工具消耗输入参数，特别是要调查的表的名称。日志显示了每次工具调用的所有输入参数的值。

`sql_db_schema`工具的调用返回了用于创建`games`表的 SQL 命令，以及表内容的一小部分样本。接下来，智能体考虑一个 SQL 查询来检索与输入问题（“2016 年最畅销的游戏是什么，如何玩？”）相关的信息。在第一步中，它通过调用`sql_db_query_checker`工具（**3**）验证以下查询在语法上是否正确：

```py
SELECT name FROM games WHERE year = 2016 ORDER BY globalsales DESC LIMIT 1
```

同时，智能体利用这个机会“反思”所考虑查询的有用性，正如输出“The games table contains the information we need. I will query for the game with the highest global sales in 2016.”所证明的那样。语言模型可以从这种独白中受益，而不是直接写出工具调用，这似乎很奇怪。然而，已经证明，使智能体能够明确地推理手头的问题以及他们为解决它所采取的步骤可以提高他们的性能 [1]。这里也发生了同样的情况。

最后，智能体决定使用先前验证的查询从数据库检索信息，使用`sql_db_query`工具（**4**）。SQL 查询返回了 2016 年收入最高的游戏：FIFA 17，一款由 Electronic Arts 生产的足球模拟游戏。但输入问题要求的更多：“2016 年最畅销的游戏是什么，以及*如何玩*？”问题的第二部分无法从数据库内容中回答。值得称赞的是，智能体意识到了这一点，并尝试访问网络：它使用`Search`工具以搜索字符串“How to play FIFA 17”发出网络搜索请求（**5**）。请注意，智能体能够根据 SQL 查询的结果和输入问题自动制定合适的搜索字符串。网络搜索的结果是文本片段的集合（显示在输出中），其中包含有关如何玩 FIFA 2017 的信息。

最后，智能体使用网络搜索返回的信息（结合 SQL 数据库中的信息）来制定最终答案（**6**）。最终答案确定 FIFA 17 是 2016 年最受欢迎的游戏，并包含如何玩好它的详细说明。我们已经看到，智能体可以执行复杂的工具调用序列来找到所需的答案，而无需手动指定要遵循的过程。如果您有兴趣，请尝试向智能体询问更多（可能更复杂的）问题，看看它是否也能回答这些问题。

## 10.5 添加自定义工具

到目前为止，我们使用了 LangChain 为最常见用例提供的标准工具。如果我们有专门的要求怎么办？例如，假设您想通过自定义 API 使数据源可访问，或者您有智能体可以应用于数据的专门分析函数。在这些情况下，您可以定义自己的自定义工具，并使它们对 LangChain 智能体可用。

### 10.5.1 货币转换器

在上一节中，我们分析了关于视频游戏销售的数据集。原始数据以美元报告销售值。其他货币呢？为了使智能体能够使用多种货币推理游戏销售，我们将添加一个货币转换器工具。给定以美元为单位的金额以及目标货币名称作为输入，此工具返回目标货币中的等值金额。

清单 10.3 展示了如何将货币转换器工具添加到我们的数据智能体中。工具的核心只是一个 Python 函数。我们的货币转换器由`convert_currency`函数（**2**）实现。LangChain 如何知道我们想要将函数转换为工具？这是通过`@tool`装饰器（**1**）完成的，它需要直接放在函数名之前。通常，我们不必为 Python 函数的参数和返回值指定类型（尽管这确实提高了代码的可读性）。但是，如果您计划将函数转换为工具，则应该指定所有这些类型。原因如下：为了正确地将您的函数用作工具，智能体需要使用正确类型的参数调用它。您在函数头中指定的所有类型都将作为工具描述的一部分提供给智能体。因此，将参数与类型关联有助于您的智能体避免不必要的调用错误。

除了参数类型之外，智能体还应该了解您的工具可以完成什么。第一条重要信息是您的函数名称。默认情况下，您的工具将以您的函数命名。不要将您的函数命名为`XYZ`，因为那会使理解正在发生的事情变得非常困难！清单 10.3 中函数的名称`convert_currency`应该非常清楚地表明该函数的功能。类似地，输入参数的名称`USD_amount`（类型为`float`）和`currency`（类型为`str`）是不言自明的（这很好！）。函数输出是目标货币中的转换金额，或者如果请求的目标货币不受支持，则返回错误消息（这就是输出类型是字符串和浮点值的`Union`的原因）。作为经验法则，如果您计划将 Python 函数用作工具，请以您希望人类编码人员无需详细阅读代码即可理解您的函数的方式来编写它。

除了函数及其参数的名称之外，智能体还会“看到”函数文档（**3**）。同样，请确保您的文档结构良好，并解释您的工具和相关参数的语义。在这种情况下，文档描述了工具的功能、输入参数的语义（甚至包括第二个参数允许值的示例）以及输出语义。

`convert_currency`函数使用一个小型货币数据库及其相关的转换因子。例如，它包含欧元和日元的转换因子，但其他货币不多。如果您正在为您的智能体创建工具，请考虑智能体未正确使用工具的情况。如果工具描述不完整或语言模型出错（即使是顶尖的语言模型也会发生这种情况），就可能发生这种情况。在这种情况下，我们为不支持目标货币的情况添加了专门处理（即，缺少相应的转换因子）（**4**）。如果目标货币不受支持，该函数将返回一条有用的错误消息，其中包含所有支持的货币。这有助于智能体在后续调用中将参数限制为可接受的选项集。如果支持目标货币，该函数将返回转换后的金额（**5**）。

##### 清单 10.3 带有货币转换器工具的数据分析智能体

```py
导入 argparse
from langchain.agents.load_tools 导入 load_tools
from langchain.tools 导入 tool
from langchain_community.utilities.sql_database 导入 SQLDatabase
from langchain_community.agent_toolkits.sql.base 导入 create_sql_agent
from langchain_openai 导入 ChatOpenAI
from typing import Union

@tool  #1
#2
def convert_currency(USD_amount: float, currency: str)->Union[float, str]:
    #3
    """
    将美元金额转换为另一种货币。
    
    参数:
        USD_amount：美元金额。
        currency：目标货币名称（例如，“日元”）。
    
    返回值:
        目标货币中的输入金额。
    """
    conversion_factors = {
        'Euro':0.93, 'Yen':151.28, 'Yun':0.14,
        'Pound':1.26, 'Won':0.00074, 'Rupee':0.012}
    
    if currency not in conversion_factors:          #4
        error_message = (
            f'Unknown currency: {currency}!'
            f'Use one of {conversion_factors.keys()}')
        return error_message
    #5
    conversion_factor = conversion_factors[currency]
    converted_amount = USD_amount * conversion_factor
    return converted_amount

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('serpaikey', type=str, help='SERP API 访问密钥')
    parser.add_argument('dbpath', type=str, help='SQLite 数据库路径')
    parser.add_argument('question', type=str, help='要回答的问题')
    args = parser.parse_args()
    
    llm = ChatOpenAI(
        temperature=0, model='gpt-4o')
    db = SQLDatabase.from_uri(f'sqlite:///{args.dbpath}')
    extra_tools = load_tools(
        ['serpapi'], serpapi_api_key=args.serpaikey, llm=llm)
    #6
    extra_tools.append(convert_currency)
    agent = create_sql_agent(
        llm=llm, db=db, verbose=True,
        agent_type='openai-tools',
        extra_tools=extra_tools)
    agent.invoke({'input':args.question})
```

#1 将函数转换为工具
#2 带类型的函数签名
#3 函数文档
#4 为智能体提供有用的错误消息
#5 转换并返回结果
#6 添加货币转换器工具

在基于 Python 函数创建工具后，我们只需要让智能体可以使用该工具。清单 10.3 创建了与清单 10.2 几乎相同的智能体，唯一的区别是我们添加了货币转换器工具（**6**）。因为我们再次使用 SQL 智能体，所以转换器工具和网络搜索工具被插入到额外工具列表中（添加到自动提供给智能体的标准 SQL 访问工具之上）。默认情况下，工具名称等于其基于的函数的名称。因此，我们只需将`convert_currency`添加到额外工具列表（**6**）中，即可为智能体增强货币转换能力。

### 10.5.2 尝试运行

让我们看看我们的智能体是否能够使用我们新添加的工具！从本书的配套网站下载清单 10.3 的代码。您可以使用与之前相同的数据库文件（并假设`games.db`文件与代码位于同一文件夹中）。然后，打开终端并执行以下代码（将您的 SerpAPI 访问密钥替换为`[SerpAPI key]`）：

```py
python listing3.py [SerpAPI 密钥] games.db  '2015 年电脑游戏产生了多少收入？ 用日元表示是多少？'
```

显然，回答这个问题需要货币转换器工具。运行代码时，您将看到类似以下的输出：

```py
[1m> 正在进入新的 SQL 代理执行器链...[0m
[32;1m[1;3m
  #1 调用：`sql_db_list_tables` with `'tool_input': "` [0m[38;5;200m[1;3mgames[0m[32;1m[1;3m
  #2 调用：`sql_db_schema` with `'table_names': 'games'` [0m[33;1m[1;3m
CREATE TABLE games (
    rank INTEGER, 
    name TEXT,
    platform TEXT,
    year INTEGER,
    genre TEXT,
    publisher TEXT,
    americasales NUMERIC,
    eusales NUMERIC,
    japansales NUMERIC,
    othersales NUMERIC,
    globalsales NUMERIC
)
/* 3 rows from the games table:
rank    name    platform    year    genre    publisher    americasales eusales    japansales    othersales    globalsales
1    Wii Sports    Wii    2006    Sports    Nintendo    41.4900000000     29.0200000000    3.7700000000    8.4600000000    82.7400000000
2    Super Mario Bros.   NES   1985   Platform   Nintendo    29.0800000000     3.5800000000    6.8100000000    0.7700000000    40.2400000000
3    Mario Kart Wii    Wii    2008    Racing    Nintendo    15.8500000000     12.8800000000    3.7900000000    3.3100000000    35.8200000000
*/[0m[32;1m[1;3m
  #3 调用：`sql_db_query_checker` with `{'query': 'SELECT SUM(globalsales) as total_revenue FROM games WHERE year = 2015'}` responded: The "games" table contains the information we need.  The "globalsales" column represents the global revenue generated by  each game. We can sum this column for the games released in 2015 to  get the total revenue. Let's write and check the SQL query.  [0m[36;1m[1;3mSELECT SUM(globalsales) as total_revenue  FROM games WHERE year = 2015[0m[32;1m[1;3m
  #4 调用：`sql_db_query` with `{'query': 'SELECT SUM(globalsales) as total_revenue FROM games WHERE year = 2015'}`  [0m[36;1m[1;3m[(264.43999999999795,)]0m[32;1m[1;3m
  #5 调用：`convert_currency` with `{'USD_amount': 264.43999999999795, 'currency': 'Yen'}` [0m[38;5;200m[1;3m40004.48319999969[0m[32;1m[1;3m
  #6 2015 年电脑游戏产生的总收入约为 264440000 美元。  用日元表示，这大约是 40,004,483,200 日元。[0m
[1m> 完成链。[0m
```

#1 智能体检索数据库表。
#2 智能体查询表模式。
#3 智能体验证 SQL 查询。
#4 智能体查询 2015 年的销售额。
#5 智能体转换货币。
#6 智能体制定最终答案。

与之前类似，智能体首先通过检索表集（**1**）来探索数据库，然后在发现数据库只包含一个表后，检索该表的模式（**2**）。智能体正确地推断出数据库包含有关输入问题的有用信息，并首先验证（**3**）然后执行（**4**）相应的 SQL 查询。SQL 查询返回 2015 年电脑游戏销售的总值，以美元表示。为了回答问题的最后部分（“用日元表示是多少？”），智能体随后应用了货币转换器工具（**5**）。请注意，智能体根据函数描述和类型为两个输入参数选择了合适的值。最后，智能体制定了对输入问题的回答（**6**）。

## 10.6 使用 LlamaIndex 索引多模态数据

LangChain 绝不是唯一一个使语言模型更易于用于数据分析的框架。在本节中，我们将讨论另一个最近出现并迅速流行的框架：LlamaIndex。

### 10.6.1 概述

LlamaIndex 擅长于语言模型需要访问大量数据集合（可能集成各种数据类型）的用例。在这种情况下，通常不建议（甚至不可能）将所有数据直接馈送到语言模型中。相反，我们需要一种机制来快速识别给定任务的相关数据，只将相关数据传递给语言模型。顾名思义，LlamaIndex 对数据进行索引以快速识别相关子集。更准确地说，LlamaIndex 将数据片段（例如文本块）与嵌入向量相关联。我们在第 4 章简要讨论了嵌入向量。简而言之，嵌入向量将文本的语义表示为由语言模型计算的向量。如果两个文档具有相似的嵌入向量（向量之间的距离很小），我们假设它们讨论相似的主题。

一个典型的 LlamaIndex 数据处理管道包含以下步骤。首先，它加载数据（可能采用各种格式）并执行预处理。例如，预处理可能涉及将长文本文档划分为更易于处理的小块。接下来，LlamaIndex 对数据进行索引。如前所述，这意味着将数据块与嵌入向量相关联。默认情况下，LlamaIndex 使用相当小的语言模型（例如 OpenAI 的 ada 模型）来计算嵌入向量。这使得索引步骤成本低廉。此外，LlamaIndex 可以将生成的索引（嵌入向量）存储在磁盘上，以避免为每个新任务重新生成它们。

LlamaIndex 为基于生成索引的各种用例提供支持。例如，它可以使用索引数据来回答自然语言问题。给定一个问题作为输入，它首先为问题文本计算一个嵌入向量。然后，它将表示问题的向量与表示数据块的预计算向量进行比较。它识别出向量最相似的数据项。相关数据与输入问题一起包含在提示中。目标是利用相关数据作为上下文来生成问题的答案。虽然使用小模型进行索引，但我们通常使用更大的模型来生成最终回复。图 10.4 说明了此数据处理管道。

![figure](img/CH10_F04_Trummer.png)

##### 图 10.4 典型 LlamaIndex 数据处理管道的主要步骤。LlamaIndex 加载和索引数据以实现快速检索。给定一个问题，LlamaIndex 识别相关的数据项，并将它们与输入问题一起提交给语言模型以生成答案。

### 10.6.2 安装 LlamaIndex

让我们在 Python 中实现上一节中的管道。要使用 LlamaIndex，我们首先必须安装几个包。转到您的终端，运行以下命令：

```py
pip install llama-index==0.10.25
```

这将为您设置 LlamaIndex 的核心包。但是，您将使用 LlamaIndex 来分析各种数据格式。为了使 LlamaIndex 能够正确访问和解析所有这些格式，您需要安装一些额外的包。在您的终端中运行以下命令：

```py
pip install torch==2.1.2
pip install transformers==4.36.0
pip install python-pptx==0.6.23
pip install Pillow==10.2.0
```

这些库对于分析`.pdf`文档和 PowerPoint 演示文稿是必需的，我们将在以下项目中使用它们。

### 10.6.3 实现一个简单的问答系统

您回到了 Banana，并面临一个具有挑战性的问题：作为一家全球性公司，Banana 有许多不同的部门。您的老板希望您分析不同部门的数据，例如，比较它们的绩效。然而，不同部门在数据格式方面有非常不同的偏好。一些部门将其结果发布为简单的文本文档，而另一些部门则定期制作精美的 PowerPoint 演示文稿。您如何整合所有这些不同的数据格式？幸运的是，LlamaIndex 使这变得容易。

查看清单 10.4：只需几行 Python 代码，它就能处理该任务。代码接受以下输入参数：

*   数据存储库的链接。此存储库可能包含各种类型的文件。
*   要回答的问题。LlamaIndex 将使用存储库中的数据来回答它。

在从命令行解析这些参数（**1**）之后，我们从输入存储库加载数据（**2**）。幸运的是，LlamaIndex 使这一步非常简单：无需为不同的文件类型等添加处理。相反，传递目录路径就足够了。接下来，我们索引刚刚加载的数据（**3**）。默认情况下，LlamaIndex 使用 OpenAI 的 ada 模型来计算嵌入向量。数据转换和分块（例如，将大型文本文档分割成足够小的片段以供 OpenAI 的 ada 模型处理）都会自动处理。现在，我们在索引之上创建一个查询引擎（**4**）。该引擎将使用索引自动检索与输入问题相关的数据。最后，我们使用先前生成的引擎来回答输入问题（**5**）并打印结果。

##### 清单 10.4 使用 LlamaIndex 的简单问答系统

```py
导入 argparse
导入 openai
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

if __name__ == '__main__':
    #1
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir', type=str, help='数据目录路径')
    parser.add_argument('question', type=str, help='要回答的问题')
    args = parser.parse_args()
    #2
    documents = SimpleDirectoryReader(args.datadir).load_data()
    #3
    index = VectorStoreIndex.from_documents(documents)
    #4
    engine = index.as_query_engine()
    #5
    answer = engine.query(args.question)
    print(answer)
```

#1 解析命令行参数
#2 从目录加载数据
#3 索引数据
#4 在索引上启用查询
#5 生成答案

尽管 LlamaIndex 提供了各种方法来配置和定制此管道的每个步骤（以及创建其他管道），但在每个步骤中使用默认设置会产生特别简洁的代码。

### 10.6.4 尝试运行

让我们使用一些示例数据来尝试我们的管道。您可以从本书的配套网站下载清单 10.4。同时从网站下载`bananareports.zip`文件，并将其解压缩到与代码相同的文件夹中（使用 Banana Reports 链接）。查看文件夹内部：您会找到文本格式的（简短）业务报告、`.pdf`文档和 PowerPoint 演示文稿。是时候回答几个问题了！打开您的终端，切换到包含代码和`bananareports`文件夹（解压缩后）的目录。现在运行以下命令：

```py
python listing4.py bananareports '2023 年香蕉单位赚了多少钱？'
```

您应该会看到类似以下的输出：

```py
香蕉单位在 2023 年赚了 3000 万美元。
```
