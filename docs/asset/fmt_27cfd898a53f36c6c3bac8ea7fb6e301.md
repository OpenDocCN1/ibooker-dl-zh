
除了之前提到的工具之外，还有一大堆工具。值得注意的是 `MiniChain` 和 `AutoChain`。两者都旨在成为 `LangChain` 的轻量级替代品，这是非常需要的，因为许多人抱怨 `LangChain` 的臃肿。`Promptify` 是一个有趣的项目，它是 `LangChain` 的全功能替代品。说实话，我们可以列出更多，但可能没有太多意义。虽然许多项目在开始时吸引了充满活力的社区，但大多数项目已经沉寂数月，`GitHub` 的贡献也极为罕见。

很难确切地说为什么对这些项目的兴趣减弱了，但一个明显的理由是，这些项目中的大多数缺乏 `LangChain`、`Guidance` 和 `DSPy` 所拥有的赞助。许多这些项目是在 `ChatGPT` 成功的炒作中作为个人项目开始的，但炒作的能量永远不足以构建持久的软件。没有适当的支持，大多数开源项目都会失败。

我们可能描绘了一幅过于悲观的画面。然而，截至撰写本文时，情况仍然难以预料，这个领域仍在不断发展。我们推荐大家检查的有趣工具还有很多，但我们没有足够的空间来包括它们，比如 `Haystack`、`Langflow` 和 `Llama Index`。`Outlines` 作为与 `Guidance` 类似的项目，尤其值得关注，后者也非常出色。我们主要想指出的是，读者在选择这个领域的工具时应该小心，因为一切仍然都很新。如果你找到一个你喜欢的工具，请做出贡献。

## 7.4 高级提示工程技术

无论你的提示多么设计得巧妙，你的模型都无法访问到一些实际情境。例如，当前事件是一个挑战。模型本身只知道其训练日期之前的信息。当然，我们可以像我们迄今为止所做的那样，通过 `RAG` 将这个情境喂给模型，但这只是将负担转移到了保持我们的 `RAG` 系统更新。还有另一种方法。在本节中，我们将讨论如何让模型访问工具，以及我们一旦这样做后可以做什么。

### 7.4.1 为 LLM 提供工具

如果不是复杂的提示工程系统，而是让我们的模型访问互联网会怎样？如果它知道如何搜索互联网，它总能找到最新的信息。在此过程中，我们还可以让它访问计算器，这样我们就不必浪费 CPU 周期让 `LLM` 本身做基本的数学运算。我们可以让它访问时钟，这样它就知道当前时间，甚至可能是一个天气应用程序，这样它就可以告诉我们天气情况。天空才是极限！我们只需要训练模型如何使用工具，这就是 `Toolformers` 的作用所在。⁸

`Toolformers` 是一个奇妙简单的想法。让我们训练一个模型，使其知道它可以使用像 `<API></API>` 这样的标签来调用不同的工具的 `API` 调用。然后，在推理时，当我们看到这些标签时，我们可以告诉我们的解释器运行这些 `API` 调用。如果这听起来很熟悉，那是因为 `Toolformers` 只是训练了一个模型来使用字符串插值！字符串插值是评估包含占位符的字符串字面量的过程，这些占位符在运行时被替换为实际的值。例如，在 `Python` 中，我们可以取字符串字面量 `print(f'2+2 = {2+2}')`，一旦打印出来，我们会得到 `'2+2 = 4'`。占位符 `{2+2}` 被评估并执行为 `Python` 代码，返回 `4`。`Schick` 等人微调了一个 `GPT-J` 模型来使用五种不同的工具：问答数据库、计算器、维基百科搜索、翻译器和日历。通过访问这些工具，他们能够取得令人印象深刻的成果，在许多任务上超过了 `GPT-3`。

虽然 `Schick` 等人的工作开辟了道路，但这种方法的主要缺点是我们不希望每次创建一个新工具时都要微调一个模型。然而，正如我们在本章中讨论的，我们不必这样做。相反，我们可以使用巧妙的提示工程，通过 `LangChain` 或 `Guidance` 引入新工具。在下一个列表中，我们展示了如何使用 `Guidance` 创建简单的数学工具。`Guidance` 通过在识别到被调用的工具时停止生成、运行工具并再次开始生成来处理繁重的工作。

##### 列表 7.4 使用 Guidance 给我们的 LLM 模型提供工具

```py
import guidance
from guidance import models, gen

falcon = models.Transformers("tiiuae/falcon-rw-1b")     #1

@guidance
def add(lm, input1, input2):
    lm += f" = {int(input1) + int(input2)}"
    return lm

@guidance
def subtract(lm, input1, input2):
    lm += f" = {int(input1) - int(input2)}"
    return lm

@guidance
def multiply(lm, input1, input2):
    lm += f" = {float(input1) * float(input2)}"
    return lm

@guidance
def divide(lm, input1, input2):
    lm += f" = {float(input1) / float(input2)}"
    return lm

lm = (
    falcon
    + """\
1 + 2 = add(1, 2) = 3
4 - 5 = subtract(4, 5) = -1
5 * 6 = multiply(5, 6) = 30
7 / 8 = divide(7, 8) = 0.875
Generate more examples of add, subtract, multiply, and divide
"""
)
lm += gen(max_tokens=15, tools=[add, subtract, multiply, divide])
print(lm)
```

#1 加载 Hugging Face Transformers 模型

虽然这是一个简单的例子，但很容易想象构建更高级的工具。无论您使用 `LangChain` 还是 `Guidance`，在构建工具时都有几点需要注意。首先，您需要在提示中指导您的模型在哪里以及如何使用您提供的工具。这可能会更难或更容易，这取决于您的函数有多开放。其次，模型的可扩展性也很重要。我们合作的一些模型永远不会使用我们给它们的工具，甚至可能会产生不存在的其他工具。最后，对于您提供给 `LLM` 的工具的输入和错误处理要非常小心。我们在这章中之前使用的是糟糕的，并且可能会以多种方式崩溃。例如，一个 `LLM` 可能会轻易尝试运行 `add(one, two)` 或 `add(1, 2, 3)`，这两者都会抛出错误并使系统崩溃。使用 `Guidance`，为了使这更容易，我们可以通过构建语法来强制工具输入，确保我们的模型输入始终正确。

这场讨论使我们发现了使用工具的 `LLM`（大型语言模型）的一些问题。首先，我们必须小心我们提供给 `LLM` 的工具，因为我们从未真正知道它将生成什么输入。即使我们确保工具不会出错，它也可能执行我们未意图的恶意行为。其次，正如您可能在本章中了解到的那样，提示工程迅速增加了我们的输入，从而减少了我们实际用户的 token 限制；解释工具及其使用方法增加了这一限制。通常，这种限制减少了我们可以提供给 `LLM` 的工具数量，从而降低了其有用性。第三，`LLM` 在使用工具方面仍然是不稳定的，它们往往最终会使用错误的工具。例如，`LLM` 应该使用网络搜索工具还是天气工具来查找 10 天的天气预报？这对我们人类来说可能不是很重要，但对于机器人来说，结果可能会有很大差异。最后，构建工具可能很困难且容易出错，因为您需要构建一个干净的工具和一个有效的提示。

##### OpenAI 的插件

`Toolformers` 开启了 `OpenAI` 插件概念的门户([`mng.bz/q0rE`](https://mng.bz/q0rE)). 插件允许第三方轻松地将他们的工具集成到 `ChatGPT` 中，并为 `ChatGPT` 调用外部 `API` 提供了一种简单的方式。插件在 `ChatGPT` 的生命周期中相对较早地被引入，在 `Toolformers` 论文发布后不久.^a 第三方只需创建一个 `OpenAPI` 配置文件和一个 `ai-plugin.json` 文件，并将两者托管在 `API` 存在的地方。`OpenAPI` 是一种用于 `API` 的规范语言，它标准化并定义了您的 `API`，使其易于他人消费。(如果您还没有听说过 `OpenAPI` 并且有客户使用的 `API`，遵循这一做法是个好习惯。您可以在[`www.openapis.org/`](https://www.openapis.org/)了解更多信息。)有很多工具可以帮助您轻松生成该文件。`ai-plugin` 文件创建了插件。在这里，您可以定义插件的名称，如何进行身份验证，以及用于提示 `ChatGPT` 的描述。从这里，插件可以在 `ChatGPT` 的界面中注册给 `OpenAI`，经过审查过程后，您的插件可以被用户添加并使用，当他们与 `ChatGPT` 互动时。

尽管一开始热情很高，但插件从未离开 `Beta` 版本——除了 `OpenAI` 自己的网页浏览插件——并且看起来已经被放弃。这有很多原因，但根据后来被撤下的报告，主要原因来自 `Sam Altman` 的建议：“很多人认为他们想要他们的应用在 `ChatGPT` 内部，但他们真正想要的其实是 `ChatGPT` 在他们的应用中”([`mng.bz/75Dg`](https://mng.bz/75Dg))。因此，似乎没有适合 `OpenAI` 插件的产品市场契合度，这会使公司赚钱。但我们认为，完全放弃这个想法还为时过早。

随着越来越多的公司将 `LLM` 技术集成到他们的应用程序中，他们很可能会想要访问第三方工具。假设你是第一次去露营，你向一个 `LLM` 购物助手询问购买建议。在这种情况下，如果它能首先询问你打算去哪里以及什么时候去露营，然后利用这些信息来识别适合天气的装备，那就太好了。特定品牌或商店的 `LLM` 购物助手可能能够访问大量产品，但访问随机地理位置的天气预报？不太可能。

虽然你可以始终构建这些工具，但如果它们已经为你准备好了，你只需去某个中心，下载你想要的工具，然后接入，那岂不是更好？不幸的是，这个选项目前还不存在，至少不是我们在这里描述的程度。我们期待未来会创建某种市场或中心，就像 `OpenAI` 的插件一样，可以与任何 `LLM` 模型一起使用。`LLM` 仍然是一项新技术，围绕它们构建的生态系统还在发展中；我们相信这将是其中之一。

^a T. Schick 等人，“Toolformer：语言模型可以教会自己使用工具”，2023 年 2 月。

一旦我们给我们的 `LLM` 提供工具访问权限，它就打开了众多酷炫提示工程技术的门户。可能最著名的就是 `ReAct` 方法。

### 7.4.2 ReAct

推理与行动（`ReAct`）是一个旨在模拟人们在学习新任务时推理和做决策的几步提示框架。⁹ 它涉及 `LLM` 的多步骤过程，其中提出一个问题，模型确定一个动作，然后观察并推理该动作的结果以确定后续动作。

一个例子可能看起来像这样：

**问题：** 与同负载的欧洲燕子相比，未负载的非洲燕子的空速是多少？

**想法 1：** 我需要查找欧洲燕子的空速，以便与非洲燕子进行比较。

**动作 1：** 搜索[欧洲燕子空速]

**观察 1：** 我们需要知道斯特劳哈数来确定空速。鸟的斯特劳哈数在 0.2 到 0.4 之间收敛。

**想法 2：** 空速没有直接提及，但斯特劳哈数是明确的。所以我需要查找速度与斯特劳哈数的比较。

**动作 2：** 搜索[斯特劳哈拍打翼计算]

**观察 2：** (结果 1/1) 一只燕子的斯特劳哈数相当于每秒约 12 次拍打翅膀。

**想法 3：** 非洲燕子需要每秒 12 次拍打翅膀来维持空速，所以我需要查找这会使它飞得多快。

**行动 3**：查找[非洲燕子每秒振翅次数]

**观察 3**：每次振翅推动非洲燕子 2.5 英尺，使其速度约为 40.2 英里/小时

**思考 4**：我需要查找欧洲燕子的信息来比较。

**行动 4**：搜索[欧洲燕子每秒振翅次数]

**观察 4**：欧洲燕子大约是非洲燕子大小的一半，其速度约为 20.1

**思考 5**：非洲燕子的空速速度为 40.2，而欧洲燕子的空速速度为 20.1，这使得比较是 2 倍。

**行动 5**：完成[两倍于空速速度]

正如你所见，`ReAct` 的目的在于迫使模型在行动之前进行思考。这与我们之前讨论的其他提示方法并没有太大的不同。最大的区别在于我们允许模型采取行动。在我们的例子中，这包括一个“搜索”行动，或者说本质上是一种像人类一样在互联网上查找信息的能力。我们已经在上一节中展示了如何做到这一点。模型可以接受新信息，并观察从其行动中学到的东西以产生结果。

让我们通过一个例子进一步探讨这个问题。我们将使用 `LangChain`，这将使创建 `ReAct` 代理比实际要容易得多。列表 7.5 展示了如何在 `OpenAI` 模型和 `LangChain` 上利用 `ReAct`。对于我们的搜索引擎，我们将使用 `serper.dev`，因为它与 `LangChain` 很好地集成，并提供一个你可以注册的免费层。我们还需要使用计算器`"llm-math"`，这是 `LangChain` 工具包中的许多工具之一。

##### 列表 7.5 示例：ReAct 与 Langchain

```py
 import os
from langchain.llms import OpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")     #1
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

llm = OpenAI(model_name="text-davinci-003", temperature=0)
tools = load_tools(["google-serper", "llm-math"], llm=llm)
agent = initialize_agent(
    tools, llm, agent="zero-shot-react-description", verbose=True
)

agent.run(
    "Who is Olivia Wilde's boyfriend? \
    What is his current age raised to the 0.23 power?"
)
```

#1 加载 API 密钥；如果你还没有获取这些，你需要获取

输出结果

```py
# > Entering new AgentExecutor chain...
# I need to find out who Olivia Wilde's boyfriend is and then
# calculate his age raised to the 0.23 power.
# Action: Search
# Action Input: "Olivia Wilde boyfriend"
# Observation: Olivia Wilde started dating Harry Styles after ending
# her years-long engagement to Jason Sudeikis — see their relationship
# timeline.
# Thought: I need to find out Harry Styles' age.
# Action: Search
# Action Input: "Harry Styles age"
# Observation: 29 years
# Thought: I need to calculate 29 raised to the 0.23 power.
# Action: Calculator
# Action Input: 29⁰.23
# Observation: Answer: 2.169459462491557

# Thought: I now know the final answer.
# Final Answer: Harry Styles, Olivia Wilde's boyfriend, is 29 years old
# and his age raised to the 0.23 power is 2.169459462491557.

# > Finished chain.

# "Harry Styles, Olivia Wilde's boyfriend, is 29 years old and his age
# raised to the 0.23 power is 2.169459462491557."
```

列表 7.5 展示了如何结合特定的代理工具如`"google-serper"`和`"llm-math"`使用 `ReAct` 与 `LLM` 来帮助增强你的提示。现在提示工程看起来更像是一项全职工作，而不仅仅是“想出词语”，对吧？
