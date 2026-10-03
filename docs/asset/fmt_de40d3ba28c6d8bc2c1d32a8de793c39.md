
除了使用花哨的 Transformer 来找到正确的答案之外，我们在基本搜索之外又迈出一步，我们将实际上使用问题/答案阅读器的置信度作为重新排序器。所以，如果我们不确定在检索步骤中哪个文档最有可能包含答案，我们将让读者浏览一大堆文档，并从中找出最好的答案。这“一大堆文档”将是我们重新排序的窗口，我们可以将其设置为任何大小。

然而，请注意，实时分析文档并不高效。我们不应该要求阅读器查看 100 个文档——这将花费太多时间。我们将将其限制在一个更小的数字，比如 3 或 5。限制阅读器窗口迫使我们确保我们的搜索引擎非常准确。结果必须是相关的，因为如果一个检索器在顶部 5 个窗口大小内找不到相关候选者，它将不会给阅读者提供任何有用的东西来工作。

检索器-读者有两个独立的问题，因此我们可以用其他东西替换我们的检索器。我们已经展示了如何使用具有开箱即用的排名（第三章中介绍的 BM25）的词汇搜索引擎，但你也可以尝试使用第十三章中介绍的密集向量索引。

在我们可以处理实时用户问题之前，我们还需要训练一个问答模型来从上下文中预测最佳答案。我们将逐步构建我们的问答应用程序的完整步骤如下：

1. **使用我们的搜索引擎设置检索器**——我们将为我们的示例使用一个简单的具有高召回率的查询作为候选答案。
2. **适当整理和标记数据**——这包括将数据放入正确的格式，并从基础预训练模型中获得我们答案的第一遍。然后我们将进行第一遍手动修正，并标记用于训练和测试的示例。
3. **理解数据结构的细微差别**——我们将使用一个现有的数据结构，它将以正确的格式表示我们的训练和测试数据，以便进行微调任务。
4. **微调模型**——使用我们在上一步中进行的修正，我们将训练一个微调的问答模型，以获得比基线更好的准确性。
5. **在查询时使用模型作为读者**——我们将把所有这些放在一起，让我们可以请求查询，从搜索引擎获取候选答案，从模型中读取/重新排序答案，并将它们作为响应展示。

图 14.4 显示了我们的检索器、读者和重新排序器整个架构的流程：

1. 用户提出一个问题，并使用该问题在检索器（搜索引擎）中查询文档。
2. 搜索引擎匹配和排序以获取对读者最相关的最高-`k`个文档。
3. 原问题与每个检索到的最高-`k`上下文配对，并送入问答管道。
4. 读者将问题/上下文对进行分词并编码成跨度，然后预测最可能的前-`n`个答案跨度，其概率作为评分。

![figure](img/CH14_F04_Grainger.png)

##### 图 14.4 提取式问答的检索器-读者模式

5. 重新排序器按降序排列每个最高-`n`个答案跨度的得分。
6. 重新排序器排序的得分最高的答案被接受，并展示给用户。

要完成所有这些，我们需要调整检索器，整理数据以训练读者模型，使用该数据微调读者模型，并构建一个重新排序器。关于调整检索器（搜索引擎）的策略，本书已经进行了详细的介绍，所以接下来，我们将整理数据。

## 14.2 构建问答训练数据集

在本节中，我们将创建一个我们可以用来训练我们的问答模型的数据库。这涉及几个步骤：

- 收集和清理数据集以适应我们的问答问题空间的内容
- 从现有的模型和语料库自动创建银集（一个需要进一步标记的半精炼数据集）
- 手动纠正银集以生成金集（一个我们可以用于训练的可信数据集）
- 将数据集分割用于训练、测试和验证微调模型

我们将从 Stack Exchange 户外数据集开始，因为它的数据已经非常适合问答应用。我们需要用于微调基础模型的问题/答案对。

户外数据集已经格式良好，并且以小型的问答块形式呈现。借助 Transformers 的力量，我们可以快速地使用现成的工具和模型构建解决方案。这比尝试从其他内容（如《伟大的期望》这本书）中构建问答数据集要容易得多。如果你正在处理长篇文本，例如书籍或长篇文档，你首先需要将文本分割成段落，并手动为这些段落设计问题。

##### 金集和银集

在机器学习中，*金集*是一个准确标记的数据集，用于训练、测试和验证模型。我们将金集视为高度宝贵的资产，因为收集它们通常需要大量的手动工作。训练模型的准确性和可用性受限于金集的准确性和广度。因此，你花费更多时间来增长和验证你的金集，模型就会越好。

为了减少在标记数据时所需的一些努力，我们可以通过让机器尝试为我们生成标记数据集来节省时间。这个自动生成的标记数据集被称为*银集*，它可以防止我们必须从头开始。

银集不如金集可信。由于我们通过机器自动化的过程自动获得银集，而这些过程不如人类准确，因此会有错误。因此，理想情况下，应该通过手动审计和纠正来提高其准确性。使用银集来启动你的训练数据集可以在长期内节省大量时间和精神努力，并有助于你扩展你的训练数据整理。

### 14.2.1 收集和清理问答数据集

接下来是我们的第一步：让我们构建一个可以标记并用于训练模型的数据库集。对于这个数据集，我们需要包含答案的相关上下文的问题。列表 14.4 展示了如何在 pandas 数据框的行中获取问题和包含答案的上下文。我们需要构建两个查询：一个用于获取社区问题，另一个用于获取这些问题的被接受社区答案。我们只将使用有被接受答案的问题/答案对。我们将分别执行这两个查询并将它们合并在一起。

我们所使用的模型指的是从中提取答案作为上下文的内容。记住，我们不是在生成答案，我们只是在文本体中找到最合适的答案。

##### 列表 14.4 从 Solr 提取训练问题

```py
def get_questions():
  question_types = ["who", "what", "when",  #1
                    "where", "why", "how"]  #1
  questions = []
  for type in question_types:
    request = {"query": type,
               "query_fields": ["title"],
               "return_fields": ["id", "url", "owner_user_id",
                                 "title", "accepted_answer_id"],
               "filters": [("accepted_answer_id", "*")],  #2
               "limit": 10000}
    docs = outdoors_collection.search(**request)["docs"]
    questions += [document for document in docs  #3
                  if document["title"].lower().startswith(type)]  #3
  return questions
```

#1 窄化我们检索的问题类型范围

#2 只检索有已接受答案的问题

#3 只使用以问题类型开头的标题

在列表 14.4 中的问题列表之后，我们接下来需要获取与每个问题相关联的上下文。列表 14.5 返回一个包含以下列的数据框：`id`、`url`、`question`和`context`。我们将使用`question`和`context`在接下来的章节中为我们的问答模型生成训练和评估数据。

##### 列表 14.5 搜索已接受答案的上下文

```py
def get_answers_from_questions(questions, batch_size=500):
  answer_ids = list(set([str(q["accepted_answer_id"])  #1
                         for q in questions]))  #1
  batches = math.ceil(len(answer_ids) / batch_size)  #2
  answers = {}

  for n in range(0, batches):  #3
    ids = answer_ids[n * batch_size:(n + 1) * batch_size]
    request = {"query": "(" + " ".join(ids) + ")",
               "query_fields": "id",
               "limit": batch_size,
               "filters": [("post_type", "answer")],
               "order_by": [("score", "desc")]}
    docs = outdoors_collection.search(**request)["docs"]
    answers |= {int(d["id"]): d["body"] for d in docs}
  return answers

def get_context_dataframe(questions):
  answers = get_answers_from_questions(questions) #4
  contexts = {"id": [], "question": [], "context": [], "url": []}
  for question in questions:
    contexts["id"].append(question["id"])
    contexts["url"].append(question["url"])
    contexts["question"].append(question["title"]),
    if question["accepted_answer_id"] in answers:
      context = answers[question["accepted_answer_id"]]
    else:
      context = "Not found"
    contexts["context"].append(context)
  return pandas.DataFrame(contexts)

questions = get_questions()  #5
contexts = get_context_dataframe(questions)  #6
display(contexts[0:5])
```

#1 获取所有不同答案 ID 的列表

#2 计算需要进行的搜索请求数量

#3 聚合所有答案

#4 获取所有问题的答案数据

#5 加载列表 14.4 中的问题。

#6 加载每个问题的上下文。

输出：

```py
id    question                               context
4410  Who places the anchors that rock c...  There are two distinct styl...
5347  Who places the bolts on rock climb...  What you're talking about i...
20662 Who gets the bill if you activate ...  Almost always the victim ge...
11587 What sort of crane, and what sort ...  To answer the snake part of...
7623  What knot is this one? What are it...  Slip knot It's undoubtably ...
```

我们鼓励您检查问题和上下文对的完整输出，以欣赏所使用的各种输入和语言。如果您想访问 Stack Exchange 户外网站并自行在 Jupyter 笔记本中探索源数据，我们还包含了原始 URL。

### 14.2.2 创建银集：从预训练模型自动标记数据

现在我们有了我们的数据集，我们必须对其进行标记。为了使训练工作，我们需要告诉模型在给定问题的上下文（文档）中正确的答案是什么。存在一个 LLM（大型语言模型）已经在这方面做得相当不错：`deepset/roberta-base-squad2`。这个模型是由 Deepset 公司使用 SQuAD2 数据集预训练的，并在他们的 Hugging Face 页面上免费提供（[`huggingface.co/deepset`](https://huggingface.co/deepset)）。SQuAD 是*斯坦福问答数据集*，这是一个由成千上万的问答对组成的大型公共数据集。Deepset 团队从第十三章中介绍的 RoBERTa 架构开始，并基于此数据集对模型进行微调，以完成问答任务。

备注：熟悉 Hugging Face 网站（[`huggingface.co`](https://huggingface.co)）是个好主意。Hugging Face 社区非常活跃，已经提供了数千个免费预训练模型，任何人都可以使用。

我们的策略是使用可用的最佳预训练模型首先尝试回答所有问题。我们将这些答案称为“猜测”，整个自动标记的数据集称为“银集”。然后我们将遍历银集猜测并自行纠正它们，以获得“金集”。

列表 14.6 展示了我们的问答函数，该函数使用 `question-answering` 类型的 Transformers 流程和 `deepset/roberta-base-squad2` 模型。我们使用这些来构建一个包含适当的分词器和目标设备（CPU 或 GPU）的流程。这为我们提供了将原始数据传入并获取银牌集合所需的一切，如图 14.5 所示。

![figure](img/CH14_F05_Grainger.png)

##### 图 14.5 从准备好的数据框中获取银牌集合和金牌集合。

在 Python 中，我们创建了一个名为 `answer_questions` 的函数，该函数接受我们从检索器中提取的上下文列表。该函数将每个问题和上下文通过流程运行以生成答案，并将其追加到列表中。我们不会假设它们实际上是答案，因为其中许多将是错误的（正如你打开文件时将看到的）。我们只有在经过人工审核后才会将某些内容计为 *答案*。这是将银牌集合升级为金牌集合的本质。

`device`（CPU 或 GPU）将根据你的 Docker 环境中是否有 GPU 自动选择。现在是一个好时机来提到，如果你在仅使用 CPU 的家用电脑或 Docker 配置上运行或训练这些模型，你可能需要等待一段时间才能完成所有数据的推理。如果你没有使用 GPU，你可以自由地跳过运行列表 14.6–14.7，因为我们已经提供了运行此笔记本数据集中后续列表所需的输出。

列表 14.6 生成银牌集合，以提取出我们之前在列表 14.5 中加载的问题和接受答案上下文对中最可能的答案。

##### 列表 14.6 根据问题/上下文对生成答案。

```py
from transformers import pipeline  #1
import torch
import tqdm  #2

def get_processor_device():  #3
  return 0 if torch.cuda.is_available() else -1  #3

def answer_questions(contexts, k=10):
  nlp = pipeline("question-answering", model=model_name,  #4
                 tokenizer=model_name, device=device)  #4
  guesses = []
  for _, row in tqdm.tqdm(contexts[0:k].iterrows(), total=k):  #5
    result = nlp({"question": row["question"],  #6
                  "context": row["context"]})  #6
    guesses.append(result)
  return guesses

model_name = "deepset/roberta-base-squad2"
device = get_processor_device()  #7

guesses = answer_questions(contexts, k=len(contexts))
display_guesses(guesses)
```

#1 这是我们在图 14.1 中展示的流程。

#2 tqdm 以进度条的形式打印操作进度。

#3 如果可用，使用 GPU (CUDA) 处理；否则使用 CPU。

#4 这是我们在图 14.1 中展示的流程。

#5 tqdm 以进度条的形式打印操作进度。

#6 为每个问题/上下文对获取答案（和置信度分数）。

#7 如果可用，使用 GPU (CUDA) 处理；否则使用 CPU。

输出：

```py
score     start  end   answer
0.278927  474    516   a local enthusiast or group of enthusiasts
0.200848  81     117   the person who is creating the climb
0.018632  14     24    the victim
...
0.247008  227    265   the traditional longbow made from wood
0.480407  408    473   shoes intended to closely approximate barefoot run...
0.563754  192    232   a tube of lightweight, stretchy material
```

恭喜，我们现在已经获得了银牌集合！在下一节中，我们将对其进行改进。

##### 推荐使用 GPU

随意将这些列表运行在你的个人电脑上，但请注意——其中一些在 CPU 上可能需要一段时间。例如，列表 14.6 的总执行时间在我们的测试中，当在 GPU 上运行时比在中端 CPU 上运行减少了大约 20 倍。
