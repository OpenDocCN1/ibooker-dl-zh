
# 第三章：通过 API 工作：生成文本

## 本章涵盖了

+ 根据特定应用对生成式 AI 模型及其分类
+ 列出可用的模型、了解它们的功能以及选择合适的模型的过程
+ OpenAI 提供的完成 API 和聊天完成 API
+ 完成 API 和聊天完成 API 的高级选项，帮助我们引导模型并因此控制生成
+ 在对话中管理令牌以改善用户体验和成本效益的重要性

我们已经看到，大型语言模型（LLMs）提供了一套强大的机器学习工具，这些工具专门设计用于增强自然语言理解和生成。OpenAI 提供了两个显著的 API：完成 API 和聊天完成 API。这些 API 以其动态和有效的文本生成能力而独树一帜，其输出类似于人类。此外，它们还为开发者提供了独特的机遇，可以构建从聊天机器人到写作助手的各种应用。OpenAI 是第一个引入完成 API 和聊天完成 API 模式的公司，现在这一模式几乎包含了所有实现，尤其是在公司想要构建由生成式 AI 驱动的工具和产品时。

OpenAI 的完成 API 是一个高级工具，它生成与上下文相关且连贯的文本以完成用户提示。相反，聊天完成 API 被设计成模拟与机器学习模型的交互，保持对话在多次交流中的上下文，这使得它适合交互式应用。

第三章为企业的扩展奠定了基础。这些 API 可以显著加速智能应用的开发，从而缩短价值实现的时间。我们将主要使用 OpenAI 和 Azure OpenAI 作为示例，经常互换使用。代码模型保持一致，API 也大体相似。许多企业可能会因为 Azure OpenAI 提供的控制而倾向于它，而其他人可能会更偏好 OpenAI。重要的是要注意，我们在这里假设 Azure OpenAI 实例已经作为您 Azure 订阅的一部分部署，我们将在示例的上下文中引用它。

本章概述了完成 API 和聊天完成 API 的基本知识，包括它们之间的区别以及何时使用每个 API。我们将看到如何在应用中实现它们，以及我们如何引导模型生成及其随机性。我们还将看到如何管理令牌，这在部署到生产环境时是关键的操作考虑因素。这些都是构建关键任务应用所需的基本方面。但在开始之前，让我们先了解不同的模型类别及其优势。

## 3.1 模型类别

根据其特定的应用，生成式人工智能模型可以划分为各种类别，每个类别都包含不同类型的模型。我们首先通过了解生成式人工智能中模型的分类来开始我们的讨论。这种理解将帮助我们确定可用的模型范围，并选择在特定情况下最合适的一个。

不同类型和模型的可用性可能因所使用的 API 而异。例如，Azure OpenAI 和 OpenAI 提供了不同版本的 LLM。一些版本可能会被淘汰，一些可能会受限，而另一些可能仅限于特定组织。

不同的模型具有独特的特性和能力，这直接影响了它们的成本和计算需求。因此，为每个用例选择正确的模型至关重要。在传统的计算机科学中，更大的就是更好的这一观念常常应用于内存、存储、CPU 或带宽。然而，在大型语言模型（LLM）的情况下，这一原则并不总是适用。OpenAI 提供了一系列分类的模型，如表 3.1 所示。请注意，这些模型在 OpenAI 和 Azure OpenAI 中都是相同的，因为底层模型是相同的。

##### 表 3.1 OpenAI 模型类别

| 模型类别 | 描述 |
| --- | --- |
| GPT-4 | 最新且最强大的版本是一组多模态模型。GPT-4 在更大的数据集和更多参数上进行了训练，使其能力进一步增强。它可以执行之前模型无法完成的任务。GPT-4 系列中包含各种模型——GPT-4.0、GPT-4 Turbo 和最新的 GPT-4o（全能），这是在发布时该系列中最强大的多模态模型。 |
| GPT-3.5 | 一组在 GPT-3 的基础上进行改进的模型，能够理解和生成自然语言或代码。在不确定的情况下，这些应该是大多数企业的默认模型。 |
| DALL.E | 一种在给定提示时能够生成图像的模型 |
| Whisper | 一种用于语音转文本的模型，将音频转换为文本 |
| Embeddings | 一组将文本转换为数值形式的模型 |
| GPT-3 (Legacy) | 一组能够生成和理解自然语言的模型。这些是现在被认为是遗留的原始模型集合。在大多数情况下，我们希望从较新的模型开始，例如 3.5 或 4.0，这些模型源自 GPT-3。 |

每个模型类别都包含一些变体，这些变体通过某些特征（如令牌大小）进一步区分。如前一章所述，令牌大小决定了模型的内容窗口，这定义了它可以处理的输入和输出量。例如，原始 GPT-3 模型的最大令牌大小为 2K。GPT-3.5 Turbo，GPT-3.5 类别中的模型子集，有两个版本——一个令牌大小为 4K，另一个令牌大小为 16K。这是原始 GPT-3 模型令牌大小的两倍和四倍。表 3.2 概述了更受欢迎的模型及其功能。

##### 表 3.2 模型描述和功能

| 模型 | 功能 |
| --- | --- |
| Ada (旧版) | 简单的文本分类、解析和格式化。这个模型是 GPT-3 旧版的一部分。 |
| Babbage (旧版) | 语义搜索排名、中等复杂度的分类。这个模型是 GPT-3 旧版的一部分。 |
| Curie (旧版) | 回答问题、高度复杂的分类。这个模型是 GPT-3 旧版的一部分。 |
| Davinci (旧版) | 摘要、生成创意内容。这个模型是 GPT-3 旧版的一部分。 |
| Cushman-Codex (旧版) | GPT-3 系列的后代，在自然语言和数十亿行代码上进行训练。它在 Python 中最强大，并且精通十多种其他编程语言。 |
| Davinci-Codex | Cushman-codex 的更强大模型 |
| GPT3.5-Turbo | 专为聊天用例优化的最强大的 GPT-3.5 模型，比 GPT-3 Davinci 便宜 90% 且更有效。 |
| GPT-4, GPT-4 Turbo | 比任何 GPT-3.5 模型都更强大。它能够执行更复杂的任务，并针对聊天模型进行了优化。 |
| GPT-4o | 最新 GPT-4o 模型比 GPT-4 和 GPT-4 Turbo 更强大，但速度也快一倍，且便宜 50%。 |
| text-embedding-ada-002, text-embedding-ada-003 | 这个新的嵌入模型取代了五个单独的模型，用于文本搜索、相似度和代码搜索，在大多数任务中表现优于它们；此外，它还便宜了 99.8%。 |

注意，提到的旧版模型仍然可用，并且按预期工作。然而，新模型更好，拥有更多的市场份额和更长的支持。大多数人应该从 GPT-3.5 Turbo 作为默认模型开始，并根据具体情况使用 GPT-4。有时，甚至是一个较小、较旧的模型，如 GPT-3 Curie，也是好的。这提供了模型能力、成本和整体性能之间的正确平衡。

在生成式 AI 的早期，所有模型都只对某些人开放。这些将因公司、地区以及 Azure 的情况（例如，您的订阅类型）等因素而有所不同。我们必须列出可用的模型及其功能。然而，在列出模型之前，让我们看看要使一切正常工作所需的依赖项。

### 3.1.1 依赖项

在本节中，我们概述了运行时依赖项和所需的高级配置。为了使一切正常工作，我们至少需要以下项目：

+ *开发 IDE*—我们使用 Visual Studio Code 作为我们的示例，但您可以使用您感到舒适的任何东西。
+ *Python*—本书中使用的是 v3.11.3，但只要它是 v3.7.1 或更高版本，您可以使用任何版本。如果您需要安装 Python，安装说明可在[`www.python.org/`](https://www.python.org/)找到。
+ *OpenAI Python 库*—我们大多数代码和演示使用 Python 库。OpenAI Python 库可以通过 conda 简单安装，使用`conda install -c conda-forge openai`。如果您使用 pip，请使用`pip install --upgrade openai`。如果您更喜欢使用特定语言的软件开发工具包（SDKs）而不是 Python 包，也有相应的 SDKs。
+ *Azure 订阅或 OpenAI API 访问*—我们交替使用 OpenAI 的端点和 Azure OpenAI (AOAI)端点；在大多数情况下，任选其一即可。鉴于本书对企业的高度重视，我们倾向于使用 Azure OpenAI 服务：
  + 要使用带有 Azure 端点的库，我们需要`api_key`。
  + 我们还需要设置`api_type`、`api_base`和`api_version`属性。`api_type`必须设置为`azure`，`api_base`指向我们部署的端点，而 API 的相应版本通过`api_version`指定。
  + Azure OpenAI 使用`'engine'`作为参数来指定模型的名称。当在您的 Azure 订阅中部署模型时，此名称需要设置为您的选择名称。例如，图 3.1 是某个订阅中部署的截图。然而，OpenAI 使用参数`model`来指定模型的名称。这些模型名称是标准的，因为它们是发布的。您可以在[`mng.bz/yoYd`](https://mng.bz/yoYd)和[`platform.openai.com/docs/`](https://platform.openai.com/docs/)找到更多关于 Azure OpenAI 和 OpenAI 的详细信息。

注意：本书配套的 GitHub 代码库([`bit.ly/GenAIBook`](https://bit.ly/GenAIBook))包含了代码的详细信息，包括依赖项和说明。

将端点和密钥硬编码不是一种推荐的做法。有多种方法可以完成这项任务，其中一种包括使用环境变量。我们将在以下步骤中演示这种方法。其他替代方案可以是从密钥库或环境文件中获取它们。为了简单起见，我们将在这份指南中坚持使用环境变量。然而，我们鼓励您遵循您企业的最佳实践和建议。设置环境变量可以通过以下命令实现。

对于 Windows，这些是

```py
setx AOAI_KEY "your-openai-key"
setx AOAI_ENDPOINT "your-openai-endpoint"
```

注意：您可能需要重新启动终端以读取新变量。

在 Linux/Mac 上，我们有

```py
export AOAI_ENDPOINT=your-openai-endpoint
export AOAI_KEY=your-openai- key
```

Bash 使用

```py
echo export AOAI_KEY="YOUR_KEY" >> /etc/environment && source /etc/environment
echo export AOAI_ENDPOINT="YOUR_ENDPOINT" >> /etc/environment && 
      ↪source /etc/environment
```

注意：在本书中，我们将使用 conda，一个开源的包管理器，来管理我们的特定运行时版本和依赖项。技术上，使用像 conda 这样的包管理器不是强制性的，但它对于隔离和解决问题非常有帮助，并且强烈推荐。在此上下文中，我们不会深入介绍如何安装 conda；有关如何安装的详细、分步说明，请参阅官方文档[`docs.conda.io/`](https://docs.conda.io/)。

首先，让我们创建一个新的 conda 环境并安装所需的 OpenAI Python 库：

```py
$ conda create -n openai python=3.11.3
(base) $ conda activate openai
(openai) $ conda install -c conda-forge openai
```

现在我们已经安装了依赖项，让我们连接到 Azure OpenAI 端点并获取可用模型的详细信息。

### 3.1.2 列出模型

正如我们之前概述的，每个组织可能都有不同的模型可供使用。我们将首先了解我们有哪些模型可以使用；我们将使用 API 来帮助我们设置基本环境并使其运行。然后，我将向您展示如何使用 Azure OpenAI Python SDK 来完成这项工作，并概述使用 OpenAI API 时的差异。

如下所示，我们连接到 Azure OpenAI 端点，获取所有可用模型的列表，遍历这些模型，并将每个模型的详细信息打印到控制台。

##### 列表 3.1 列出可用的 Azure OpenAI 模型

```py
import os
import json
from openai import AzureOpenAI                  #1

client = AzureOpenAI(
    azure_endpoint=os.getenv("AOAI_ENDPOINT"),  #2
    api_version="2023-05-15",                   #3
    api_key=os.getenv("AOAI_KEY")   #4
    )

# Call the models API to retrieve a list of available models
models = client.models.list()

# save to file
with open('azure-oai-models.json', 'w') as file:
    models_dict = [model.__dict__ for model in models]
    json.dump(models_dict, file)

# Print out the names of all the available models, and their capabilities
for model in models:
    print("ID:", model.id)
    print("Current status:", model.lifecycle_status)
    print("Model capabilities:", model.capabilities)
    print("-------------------")
```

#1 必须用于 Azure OpenAI 端点
#2 这是指向通过 Azure 门户发布的端点的环境变量。
#3 从多个选项中选择我们想要使用的 API 版本。
#4 这是包含 API 密钥的环境变量。

运行此代码将向我们展示可用模型的列表。以下列表显示了可用模型的示例；您的确切列表可能会有所不同。

##### 列表 3.2 列出 Azure OpenAI 模型的输出

```py
{
    "id": "gpt-4-vision-preview",
    "created": null,
    "object": "model",
    "owned_by": null
},
{
    "id": "dall-e-3",
    "created": null,
    "object": "model",
    "owned_by": null
  },
  {
    "id": "gpt-35-turbo",
    "created": null,
    "object": "model",
    "owned_by": null
},
…
```

每个模型都有其独特的功能，这表明了它针对的使用场景——特别是用于聊天补全、补全（即常规文本补全）、嵌入和微调。例如，在需要大量对话交流的基于聊天的交互中，聊天补全模型将是理想的选择。相反，补全模型将最适合文本生成。我们可以在图 3.1 中查看 Azure AI Studio 中的 OpenAI 基础模型。

![figure](img/CH03_F01_Bahree.png)

##### 图 3.1 列出的基础模型

此功能是 Azure AI Studio 的一部分，您可以在登录 Azure 订阅并访问您的 Azure OpenAI 部署时访问它。您也可以直接通过门户[`oai.azure.com/portal`](https://oai.azure.com/portal)访问它。现在我们已经知道了要使用哪个模型，让我们生成一些文本。我们将使用补全 API 和一个支持补全的模型。

## 3.2 补全 API

完成 API 是一个复杂的工具，用于生成文本以完成用户提供的提示。它是 OpenAI API 的核心，提供了一个简单但强大且灵活的 API。它旨在生成与给定提示一致且上下文合适的文本。

许多非聊天型结构的生成示例都使用完成 API。我们必须使用完成 API 来生成非聊天风格的文本。完成 API 的一些好处包括

+ *上下文理解*—完成 API 可以理解提示的上下文并生成相关文本。
+ *多功能性*—它可以用于各种应用，从创建内容到回答问题，使其成为多种应用的宝贵工具。
+ *多语言理解*—完成 API 能够理解和生成多种语言的内容，使其成为全球资源。
+ *易于实现*—完成 API 简单直接，这使得它对各种技能水平的开发者都易于访问。

API 的结构非常简单，如下面的代码片段所示。输入（提示）和输出（完成）都是文本格式。API 响应是一个 JSON 对象，可以使用 text 键从中提取生成的文本。这个响应被称为文本完成。完成试图遵循提示和上下文中提供的内容，并且是潜在输出之一：

```py
from openai import OpenAI
client = OpenAI()

response = client.completions.create(
  model="gpt-3.5-turbo-instruct",
  prompt="Write a few bullets on why pets are so awesome ",
  max_tokens=100,
  temperature=0.8 
)
print(response.choices[0].text.strip())
```
