
在这个例子中，文本窗口有重叠。有一些重叠通常是好事；否则，一个重要的点可能会在窗口边界处被切断。然而，您有权决定这一点。您可能希望使窗口重叠更多，以确保永远不会切断任何想法。另一方面，为了节省存储成本，您可能选择减少或完全去除重叠，这样就会有更少的片段和相应更少的向量需要跟踪。

收集片段的另一种方法是按照自然边界，如段落或章节，将文档切割。这有助于确保每个片段最多包含一个主题，并且没有机会在句子中间被切断。

最后，您还可以考虑通过添加文本来增强您的片段，这些文本可能*应该*包含在片段中但并未包含。一个很好的例子是代码。考虑一个由单个函数组成的片段。如果该函数是独立的，那么函数的文本本身可能就足够作为片段。但如果该函数实际上是一个属于类的成员方法，那么就包括一些额外的上下文。将函数重新组装成一个包含类定义、任何初始化代码（以便包括实例变量）和方法的代码片段。这将给嵌入模型提供更多上下文，以构建更好的向量。

## 嵌入模型

您如何选择嵌入模型？首先需要提到的是，嵌入模型与 LLM（大型语言模型）不同。嵌入模型通常基于与 LLM 相同的 Transformer 架构，但它不是预测下一个标记，而是生成一个向量。更具体地说，嵌入模型已经通过称为[对比预训练](https://arxiv.org/pdf/2201.10005)的特殊训练过程进行了专门训练，以便相关的输入文本对应于附近的向量，而不相关的输入文本对应于彼此之间距离较远的向量。

嵌入模型和 LLM 之间的重要区别是，与 LLM 相比，嵌入模型非常小，并且便宜得多。这促进了索引大量文本的可能性。

在选择嵌入模型时，您有几个选择。一个选项是使用托管模型，例如来自 OpenAI 的那些。这些模型很容易开始使用，因为没有设置——只需获取一个 API 密钥即可。但随着您的应用程序成熟，您可能希望托管自己的嵌入模型，这将减少网络延迟并可能降低成本。

这些天，嵌入模型通常在代码和文本上训练，并且越来越有可能从同一个模型中获得两个领域的良好性能。但是，如果您有一个特定的用例，比如不寻常的语言（自然语言或代码语言），那么您可能想要寻找更适合您目的的模型。如果所有其他方法都失败了，您可能需要考虑训练自己的模型——这并不像训练一个大型语言模型（LLM）那么困难。

## 向量存储

嵌入是长向量，通常有数千个条目，在索引中搜索与给定向量最接近的片段嵌入不是一个简单任务。另一方面，至少这是一个已解决的问题。像 [FAISS](https://oreil.ly/TavsF) 这样的库使得向量查找足够快，以至于它们不会减慢您的提示创建。如果您不想维护自己的向量数据存储的操作开销，那么还有几个软件即服务（SaaS）选项可供选择。例如，[Pinecone.io](http://pinecone.io) 提供了一个完全托管的服务以及扩展到大量向量的能力。如果您想了解更多关于 FAISS 或支持快速向量搜索的数据结构的信息，Pinecone.io 有一些非常有用的博客文章（参见“[Facebook AI 相似性搜索 [FAISS] 简介](https://oreil.ly/DbnC6)”和“[分层可导航小词 [HNSW]](https://oreil.ly/Y7U5F)”）。

## 构建简单的 RAG 应用程序

让我们花点时间构建一个无装饰的 RAG 应用。我们将制作 图 5-12 中表示的应用程序。目标不是构建一个完美的 RAG 应用，而是制作一个最简单的 RAG 应用，其中包含您可能在更生产就绪的应用中期望的大部分基本组件。

![计算机的示意图  描述由系统自动生成](img/pefl_0512.png)

###### 图 5-12\. 一个 RAG 应用

注意，在图中，在离线过程中，索引管道将片段转换为向量并将它们存储起来（如图左侧所示）。然后，在请求时间，应用程序检索上下文并组装一个预测用户书籍评分的提示（如图右侧所示）。

在这个应用中，我们假设用户正在在线书店评论书籍，以寻找一本新书来阅读。当用户打开特定书籍的网页时，我们将为他们提供一个关于他们可能会喜欢这本书的估计。我们通过检索他们过去写过的任何相关评论，然后制作一个提示，将这本书的摘要与用户的评论进行比较，以预测用户是否会对这本书感兴趣。

首先，让我们导入所需的库并实例化一个 OpenAI 客户端：

```py
import numpy as np
import faiss
from openai import OpenAI
client = OpenAI()
```

接下来，我们收集该用户曾经做出的所有评论：

```py
reviews = [
   "I hate stories about backpacking. It's boring.",
   "A moving exploration of racial injustice and moral growth.",
   "Compelling dystopia, but overwhelmingly bleak.",
   "Timeless romance with sharp social commentary.",
   "Epic sea adventure with philosophical depth.",
   "Mesmerizing magic and romance with rich world-building.",
   "Beautifully descriptive, but predictable plot.",
   "A detailed and emotional journey through loss and art.",
   "Fresh take on Greek mythology, but pacing dragged.",
   "Brilliant exploration of complex relationships and personal growth.",
   "Another bland romantic utopia. This time on a tropical island.",
]
```

我们需要一种方法来检索嵌入向量。在这里，`get_embedding`使用提供的文本块从 OpenAI 模型中检索嵌入向量：

```py
def get_embedding(text):
   text = text.replace("\n", " ")
   return client.embeddings.create(
      input = [text], 
      model="text-embedding-3-small",
   ).data[0].embedding
```

接下来，我们将创建一个索引函数，为我们的每条评论检索向量，实例化一个 FAISS 索引，并将向量添加到索引中。此函数随后返回用于后续搜索的向量索引：

```py
def index_reviews(reviews):
   # get the embeddings for the reviews
   vectors = []
   for review in reviews:
      vectors.append(get_embedding(review))

   # create the index
   d = len(vectors[0]) # dimension of the vectors
   index = faiss.IndexFlatL2(d)

   # reshape vectors into 2D array and then add to the index
   vectors = np.array(vectors).reshape(len(vectors), -1)
   index.add(vectors)

   return index
```

接下来，我们构建一个检索函数。给定一个查询，这个函数获取查询文本的嵌入向量，在索引中找到最近的邻居，并使用最近邻居的索引来收集原始评论文本：

```py
def retrieve_reviews(index, query, reviews, k=2):
   # get the embedding for the query
   query_vector = get_embedding(query)

   # reshape vector into 2D array and then search the index
   query_vector = np.array(query_vector).reshape(1, -1)
   distances, indices = index.search(query_vector, k)

   return [reviews[i] for i in indices[0]]
```

让我们试一试：

```py
index = index_reviews(reviews)

book = "The Beach by Alex Garland critiques backpacker culture by exposing the 
selfishness and moral decay behind their pursuit of an untouched paradise."

related_reviews = retrieve_reviews(index, book, reviews)

print(related_reviews)
```

这返回了以下来自用户先前评论的合理片段：

+   我讨厌关于背包旅行的故事。这很无聊。

+   另一个平淡无奇的浪漫乌托邦。这次在一个热带岛屿上。

现在我们已经使检索工作正常，构建 RAG 应用的最后一部分是将结果以某种方式放入提示中，使模型知道如何使用它们。为此，我们创建了以下`predict_rating`函数，它使用静态模板来构建问题，并使用动态内容来传达用户的即时上下文：

```py
def predict_rating(book, related_reviews):
   reviews = "\n".join(related_reviews)

   prompt = (
      "Here is a book I might want to read:\n" +
      book + "\n\n" +

      "Here are relevant reviews from the past:\n" +
      reviews + "\n\n" +

      "On a scale of 1 (worst) to 5 (best), " +
      "how likely am I to enjoy this book? " +
      "Reply with no explanation, just a number."
   )

   response = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[{
         "role": "user",
         "content": prompt
      }],
      max_tokens=2000,
      temperature=0.7,
   )

   return response.choices[0].message.content
```

最后，我们调用我们的 RAG 应用来预测用户会对这本书给出怎样的评分：`predict_rating(book, related_reviews)`。完整的提示如下：

```py
Here is a book I might want to read:
The Beach by Alex Garland critiques backpacker culture by exposing the 
selfishness and moral decay behind their pursuit of an untouched paradise.

Here are relevant reviews from the past:
I hate stories about backpacking. It's boring.
Another bland romantic utopia. This time on a tropical island.

On a scale of 1 (worst) to 5 (best), how likely am I to enjoy this book? 
Reply with no explanation, just a number.

```

最终预测是，我们的用户会给《海滩》打 2 分——根据他们过去的书籍评论，他们不太可能喜欢《海滩》。

### 神经检索与词汇检索

在上述对话中，RAG 应用是通过神经检索构建的。目前大多数 RAG 应用都是这样构建的，但你完全没有理由不能使用词汇检索来构建 RAG。事实上，实际上有一些非常好的理由表明词汇检索可能更为可取。

词汇检索是一种经过验证的方法。它已经存在了几十年，今天，它*仍然*为大多数在线搜索体验提供动力。有许多用于词汇检索的软件解决方案，例如 Elasticsearch（这是一种开源软件）和 Algolia（这是一个平台即服务[PaaS]）。启动这些技术中的任何一种都很简单，可以索引大量文档，并以低延迟进行搜索。

在神经检索中，查询和文档被转换为不透明的向量，如果你没有看到你预期的匹配，你几乎无法做任何事情来理解问题并修复它。另一方面，在词汇检索中，当文档与查询不匹配时，很容易理解原因——这是因为查询中的标记与文档中的标记不匹配。你可以通过修改词干或用词的同义词增强文档来修复这类问题。

使用词汇检索，你还可以调整相关性以匹配用户的期望。你可以通过根据领域修改相关性分数的权重来实现这一点——例如，通过在标题字段上提升匹配比描述字段上的匹配更重要。在神经检索中，你所能做到的最接近这一点的是通过训练一个新模型，该模型以某种方式结合了这些相关性概念，然后重新索引你的整个文档集。

然而，我们不应该因噎废食，神经检索确实在词汇检索之上提供了一些好处。最有用的是，与词汇检索基于单词进行匹配不同，神经检索基于思想进行匹配。这意味着即使两份文档使用完全不同的单词来描述同一事物，它们仍然会被视为匹配。这甚至扩展到用不同语言描述同一事物的两份文档，或者匹配映射到同一嵌入空间的图像文档！

## 摘要

检索通过聚焦于最相关的片段来处理大量可能的上下文。摘要通过完全相反的方式处理大量上下文：缩小范围并提供相关信息的一个简短概述。LLMs 可以轻松用于摘要任务。例如，使用 gpt-3.5-turbo-instruct，我刚刚将上一节文本放入提示中，并附加了`Tersely summarize all of the above`。生成的结果包含了一个很好的摘要：

> RAG（检索增强生成）是一种强大的技术，它结合了语言模型和检索系统的优势，以生成相关且个性化的响应。它涉及使用嵌入模型将文本转换为向量，然后对这些向量进行索引以实现快速检索。RAG 应用可以使用神经检索或词汇检索方法构建，每种方法都有其自身的优势。神经检索基于思想而非单词进行匹配，而词汇检索则允许对相关性和调整有更多的控制。

实际上，这并不坏。我已将章节草案放入此提示中，以下是返回的摘要：

> 此模型的最大上下文长度为 4097 个标记，但你请求了 9491 个标记（提示中的 8491 个；完成时的 1000 个）。请减少你的提示或完成长度。

噢，是的，上下文窗口大小。尽管自从 gpt-3.5-turbo-instruct 以来上下文窗口大小已经显著增加，但你仍然不太可能将整本书放入你的上下文窗口中。文本过长的事实是你最初需要总结它的原因！

### 层次化总结

当要总结的文本对于上下文窗口来说太长时，解决办法是*层次化总结*。这是一种分而治之的方法，你首先将你的语料库分成不超过上下文窗口大小的语义实体，然后对它们进行总结。然后，你对总结列表进行总结。在图 5-13 中，我们首先总结《海滩》的各个章节，然后总结这些总结，以生成整本书的最终、总体总结。

甚至可能连总结的总结都不够用。例如，《圣经》有 1,189 章，即使每章简洁的总结只有 50 个字，也可能让大多数前沿 LLM 模型超过它们的 token 限制。解决这个问题的方法是使用*递归*，这意味着先总结章节，然后在书级别（圣经中有 66 本书）总结章节总结，最后总结书总结以获得《圣经》的最终总结。

如果你对世界宗教的话题不感兴趣，那么还有很多其他地方文本自然组织成层次结构。例如，如果你想总结一个大型的代码库，那么一个自然的方法就是层次化——总结文件，然后遍历目录结构，在每个级别进行总结。

这个总结过程有多贵？一般来说，只要总结的大小平均小于原始文本大小的十分之一，那么无论层次有多深，总结的成本都是由原始文本中的总 token 数决定的。

在深度层次化总结中，另一个需要警惕的问题是*谣言问题*：每次总结总结时，模型误解某事的可能性都存在，这种误解会对后续级别产生连锁反应。所以，在第一级，误解的机会只有一次，但在第三级，就有三次机会。不过，总的来说，这个电话游戏并不长，只要你不过分吝啬总结的长度，每个级别的总结损失并不足以造成太大影响。

![文本和箭头的图示   描述自动生成，置信度中等](img/pefl_0513.png)

###### 图 5-13\. 层次化总结（ChatGPT 生成的总结，包含剧透）

###### 小贴士

如果你的语料库有自然的分组——章节、部分、主题、作者和项目——尝试在这些自然边界上分割内容，并在每个摘要传递中使用恰好一个这样的组的内容。如果你必须在非自然边界上分割文本，那么避免不平衡的摘要，其中大部分要总结的文本来自一个部分，而只有一小部分来自另一个部分。

### 一般和特定摘要

摘要是压缩的一种形式，而压缩永远不是无损的。如果你的模型总结了一个关于用户上次度假的长篇社交媒体帖子，它可能会保留他们去了哪里以及他们喜欢什么。它可能不会保留关于哪本书让长途飞行更易忍受的随意评论，因为这不是帖子的核心……然而，这个评论正是 LLM 后来需要知道以做出更好的书籍推荐的信息！

答案很简单：只需在考虑你的最终应用任务时请求摘要。参见表 5-3 中的示例提示，并注意在实践中，要总结的文本会更长。
