
如图 3-21 所示，我们可以通过专注于生成主题级输出而非文档级输出，来高效使用 `BERTopic`。这可以将 API 调用的数量从数百万（例如，数百万的摘要）减少到几百（例如，数百个主题）。这不仅显著加快了主题标签的生成速度，而且在使用外部 API（如 `Cohere` 或 `OpenAI`）时，也不需要大量的费用。

![使用文本生成 LLMs 和提示工程从与每个主题相关的关键词和文档中创建主题标签。](img/text_clustering_and_topic_modeling_664502_21.png)

###### 图 3-21\. 使用文本生成 LLMs 和提示工程从与每个主题相关的关键词和文档中创建主题标签。

### 提示

正如在图 3-21 中所示，文本生成的一个主要组成部分是提示。在 `BERTopic` 中，这同样重要，因为我们希望向模型提供足够的信息，以便它能决定主题内容。`BERTopic` 中的提示通常看起来像这样：

```py
prompt = """
I have a topic that contains the following documents: \n[DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS]

Based on the above information, give a short label of the topic.
"""
```

该提示包含三个组成部分。首先，它提到一些最能描述主题的文档。这些文档通过计算它们的 `c-TF-IDF` 表示并与主题的 `c-TF-IDF` 表示进行比较来选择。然后提取前四个最相似的文档，并使用“**[文档]**”标签进行引用。

```py
I have a topic that contains the following documents: \n[DOCUMENTS]
```

其次，构成主题的关键词也会传递给提示，并使用“**[关键词]**”标签进行引用。这些关键词也可以通过 `KeyBERTInspired`、词性或任何表示模型进行优化。

```py
The topic is described by the following keywords: [KEYWORDS]
```

第三，我们向大型语言模型提供具体指令。这与之前的步骤同样重要，因为这将决定模型如何生成标签。

```py
Based on the above information, give a short label of the topic.
```

该提示将被呈现为主题 XXX：

```py
"""
I have a topic that contains the following documents: 
- Our videos are also made possible by your support on patreon.co.
- If you want to help us make more videos, you can do so on patreon.com or get one of our posters from our shop.
- If you want to help us make more videos, you can do so there.
- And if you want to support us in our endeavor to survive in the world of online video, and make more videos, you can do so on patreon.com.

The topic is described by the following keywords: videos video you our support want this us channel patreon make on we if facebook to patreoncom can for and more watch 

Based on the above information, give a short label of the topic.
"""
```

### HuggingFace

幸运的是，与大多数大型语言模型一样，我们可以通过[HuggingFace 的 Modelhub](https://huggingface.co/models)使用大量开源模型。

最著名的开源大型语言模型之一是 `Flan-T5` 生成模型系列，它针对文本生成进行了优化。这些模型的有趣之处在于它们使用一种称为**指令调优**的方法进行训练。通过对许多以指令形式表达的任务进行微调，模型学会了遵循特定的指令和任务。

`BERTopic` 允许使用这样的模型生成主题标签。我们创建一个提示，请它根据每个主题的关键词生成主题，并标记为`[关键词]`。

```py
from transformers import pipeline
from bertopic.representation import TextGeneration

# Text2Text Generation with Flan-T5
generator = pipeline('text2text-generation', model='google/flan-t5-xl')
representation_model = TextGeneration(generator)

# Use the representation model in BERTopic on top of the default pipeline
topic_model.update_topics(abstracts, representation_model=representation_model)

# Show topic differences
topic_differences(topic_model, original_topics)
```

`主题：0 演讲 | asr | 识别 | 声学 | 端到端 --> 音频语法识别`

`主题：1 临床 | 医疗 | 生物医学 | 笔记 | 健康 --> ehr`

`主题：2 摘要 | 总结 | 总结 | 抽象 | 抽取 --> mds`

`主题：3 解析 | 解析器 | 依赖关系 | amr | 解析器 --> 解析器`

`主题：4 仇恨 | 攻击性 | 演讲 | 检测 | 有毒 --> Twitter`

`主题：5 词 | 嵌入 | 嵌入向量 | 相似性 --> word2vec`

`主题：6 性别 | 偏见 | 偏差 | 去偏见 | 公平性 --> 性别偏见`

`主题：7 命名 | 实体 | 识别 | 嵌套 --> ner`

`主题：8 提示 | 少样本 | 提示 | 上下文 | 调优 --> gpt3`

`主题：9 关系 | 提取 | re | 关系 | 远程 --> docre`

有趣的主题标签被创建，但我们也可以看到该模型并不是完美无缺的。

### OpenAI

当我们谈论生成性 AI 时，不能忘记 `ChatGPT` 及其惊人的表现。尽管不是开源的，但它是一个有趣的模型，在短短几个月内改变了 AI 领域。我们可以从 `OpenAI` 的集合中选择任何文本生成模型在 `BERTopic` 中使用。

由于该模型是基于 `RLHF` 训练的，并且优化用于聊天目的，因此使用该模型进行提示非常令人满意。

```py
from bertopic.representation import OpenAI

# OpenAI Representation Model
prompt = """
I have a topic that contains the following documents: \n[DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS]

Based on the information above, extract a short topic label in the following format:
topic: <topic label>
"""
representation_model = OpenAI(model="gpt-3.5-turbo", delay_in_seconds=10, chat=True)

# Use the representation model in BERTopic on top of the default pipeline
topic_model.update_topics(abstracts, representation_model=representation_model)

# Show topic differences
topic_differences(topic_model, original_topics)
```

`主题：0 演讲 | asr | 识别 | 声学 | 端到端 --> 音频语法识别`

`主题：1 临床 | 医疗 | 生物医学 | 记录 | 健康 --> ehr`

`主题：2 总结 | 摘要 | 总结 | 抽象 | 提取 --> mds`

`主题：3 解析 | 解析器 | 依赖 | amr | 解析器 --> parser`

`主题：4 仇恨 | 攻击性 | 言论 | 检测 | 有毒 --> Twitter`

`主题：5 词 | 嵌入 | 嵌入向量 | 相似性 --> word2vec`

`主题：6 性别 | 偏见 | 偏差 | 去偏见 | 公平性 --> 性别偏见`

`主题：7 命名 | 实体 | 识别 | 嵌套 --> ner`

`主题：8 提示 | 少样本 | 提示 | 上下文 | 调优 --> gpt3`

`主题：9 关系 | 提取 | re | 关系 | 远程 --> docre`

由于我们期望 `ChatGPT` 以特定格式返回主题，即“主题：<主题标签>”，因此在创建自定义提示时，指示模型按此格式返回非常重要。请注意，我们还添加了`delay_in_seconds`参数，以便在 API 调用之间创建恒定的延迟，以防你使用的是免费账户。

### Cohere

与 `OpenAI` 一样，我们可以在 `BERTopic` 的管道中使用 `Cohere` 的 API，进一步微调主题表示，结合生成文本模型。确保获取 API 密钥，这样你就可以开始生成主题表示。

```py
import cohere
from bertopic.representation import Cohere

# Cohere Representation Model
co = cohere.Client(my_api_key)
representation_model = Cohere(co)

# Use the representation model in BERTopic on top of the default pipeline
topic_model.update_topics(abstracts, representation_model=representation_model)

# Show topic differences
topic_differences(topic_model, original_topics)
```

`主题：0 演讲 | asr | 识别 | 声学 | 端到端 --> 音频语法识别`

`主题：1 临床 | 医疗 | 生物医学 | 记录 | 健康 --> ehr`

`主题：2 总结 | 摘要 | 总结 | 抽象 | 提取 --> mds`

`主题：3 解析 | 解析器 | 依赖 | amr | 解析器 --> parser`

`主题：4 仇恨 | 攻击性 | 言论 | 检测 | 有毒 --> Twitter`

`主题：5 词 | 嵌入 | 嵌入向量 | 相似性 --> word2vec`

`主题：6 性别 | 偏见 | 偏差 | 去偏见 | 公平性 --> 性别偏见`

`主题：7 命名 | 实体 | 识别 | 嵌套 --> ner`

`主题：8 提示 | 少样本 | 提示 | 上下文 | 调优 --> gpt3`

`主题：9 关系 | 提取 | re | 关系 | 远程 --> docre`

### LangChain

为了进一步提升大型语言模型的能力，我们可以利用 `LangChain` 框架。它允许任何先前的文本生成方法补充额外信息，甚至链式结合。特别是，`LangChain` 将语言模型连接到其他数据源，使它们能够与环境互动。

例如，我们可以使用它与 `OpenAI` 构建一个向量数据库，并在该数据库上应用 `ChatGPT`。由于我们希望尽量减少 `LangChain` 所需的信息量，因此将最具代表性的文档传递给该软件包。然后，我们可以使用任何 `LangChain` 支持的语言模型来提取主题。下面的示例演示了如何将 `OpenAI` 与 `LangChain` 结合使用。

```py
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from bertopic.representation import LangChain

# Langchain representation model
chain = load_qa_chain(OpenAI(temperature=0, openai_api_key=MY_API_KEY), chain_type="stuff")
representation_model = LangChain(chain)

# Use the representation model in BERTopic on top of the default pipeline
topic_model.update_topics(abstracts, representation_model=representation_model)

# Show topic differences
topic_differences(topic_model, original_topics)
```

`Topic: 0 speech | asr | recognition | acoustic | endtoend --> audio grammatical recognition`

`Topic: 1 clinical | medical | biomedical | notes | health --> ehr`

`Topic: 2 summarization | summaries | summary | abstractive | extractive --> mds`

`Topic: 3 parsing | parser | dependency | amr | parsers --> parser`

`Topic: 4 hate | offensive | speech | detection | toxic --> Twitter`

`Topic: 5 word | embeddings | embedding | vectors | similarity --> word2vec`

`Topic: 6 gender | bias | biases | debiasing | fairness --> gender bias`

`Topic: 7 ner | named | entity | recognition | nested --> ner`

`Topic: 8 prompt | fewshot | prompts | incontext | tuning --> gpt3`

`Topic: 9 relation | extraction | re | relations | distant --> docre`

## 主题建模变体

主题建模的领域相当广泛，涵盖了许多不同的应用和同一模型的变体。`BERTopic` 也不例外，它为不同目的实现了多种变体，例如动态、（半）监督、在线、分层和引导的主题建模。图 3-22-X 展示了一些主题建模变体以及如何在 `BERTopic` 中实现它们。

![ BERTopic 中的 X 主题建模变体](img/text_clustering_and_topic_modeling_664502_22.png)

###### 图 3-22\. -X BERTopic 中的主题建模变体

# 摘要

在本章中，我们讨论了一种基于聚类的主题建模方法，`BERTopic`。通过利用模块化结构，我们使用了多种大型语言模型来创建文档表示并微调主题表示。我们提取了 `ArXiv` 摘要中的主题，并观察了如何利用 `BERTopic` 的模块化结构来开发不同类型的主题表示。
