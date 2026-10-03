
# 第九章：使用 TensorFlow 进行自然语言处理：情感分析

本章涵盖内容

+ 使用 Python 对文本进行预处理
+ 分析对模型重要的文本特定属性
+ 使用 TensorFlow 创建处理文本序列的数据管道
+ 使用递归深度学习模型（LSTM）进行情感分析
+ 对不平衡的产品评论进行模型训练
+ 实现词嵌入以提高模型性能

在之前的章节中，我们看了两个与计算机视觉相关的应用程序：图像分类和图像分割。图像分类侧重于识别图像中是否存在属于某个类别的对象。图像分割任务不仅关注于识别图像中存在的对象，还关注于图像中哪些像素属于某个对象。我们还围绕学习复杂卷积神经网络的基础展开了讨论，比如 Inception net（图像分类）和 DeepLab v3（图像分割）模型。如果我们超越图像，文本数据也是一种重要的数据形式。例如，全球范围内的网络充斥着文本数据。我们可以有把握地认为，它是网络上最常见的数据形式。因此，自然语言处理（NLP）一直是一个根深蒂固的主题，使我们能够利用免费的文本（例如，通过语言建模）的力量，并构建能够利用文本数据产生有意义结果的机器学习产品（例如，情感分析）。

*自然语言处理*是一个概括性概念，涵盖了与文本相关的各种任务。从简单的任务，比如改变文本的大小写（例如，将大写转换为小写），到复杂的任务，比如翻译语言和词义消歧（根据上下文推断具有相同拼写的单词的含义）都属于自然语言处理的范畴。以下是您在进入自然语言处理领域时将遇到的一些显著任务：

+ *停用词去除*——停用词是频繁出现在文本语料库中的无信息词（例如，“and”，“it”，“the”，“am”等）。通常情况下，这些词对文本的语义（或含义）几乎没有或没有任何贡献。为了减少特征空间，许多任务在将文本馈送到模型之前的早期阶段去除停用词。
+ *词形还原*——这是另一种减少模型需要处理的特征空间的技术。词形还原将给定单词转换为其基本形式（例如，将 buses 转换为 bus，walked 转换为 walk，went 转换为 go 等），从而减少词汇表的大小，进而减少模型需要学习的数据的维度。
+ *词性标注（PoS）*—PoS 标注就是标注给定文本中的每个单词的词性（例如名词、动词、形容词等）。Penn Treebank 项目提供了最流行和最全面的 PoS 标签列表之一。要查看完整列表，请访问[`mng.bz/mO1W`](http://mng.bz/mO1W)。
+ *命名实体识别（NER）*—NER 负责从文本中提取各种实体（例如人名/公司名、地理位置等）。
+ *语言建模*—语言建模的任务是预测第*n*（n>1）个单词，给定前面 1 至*w*-1 个单词。通过在相关数据上训练模型，可以使用语言建模生成歌曲、电影脚本、故事等。由于语言建模所需的数据具有高度的可访问性（即不需要任何标记数据），因此通常作为预训练方法用于为决策支持 NLP 模型注入语言理解。
+ *情感分析*—情感分析是根据一段文本的情感来进行评价的任务。例如，情感分析器可以分析产品/电影评论并自动生成一个得分来表示产品质量。
+ *机器翻译*—机器翻译是将源语言中的短语/句子翻译为目标语言中的短语/句子的任务。这些模型是使用双语平行文本语料库进行训练的。

作为数据科学家或 ML 研究人员，几乎不可能不遇到 NLP 任务。要快速而成功地解决 NLP 任务，重要的是要了解处理数据、使用标准模型等等。

在本章中，您将学习如何为视频游戏评论分类开发情感分析器。您将开始探索数据，并了解一些常见的 NLP 预处理步骤。您还将注意到数据集不平衡（即对于所有的类别数量不大致相等），并学习该怎么做。然后，我们将开发一个 TensorFlow 数据管道，通过该管道，我们将向模型输入数据进行训练。在这里，您将遇到一种在 NLP 领域中有影响的新型机器学习模型，即*长期短期记忆*（LSTM）模型。LSTM 模型可以通过迭代地处理序列（例如，句子中以特定顺序排列的一串单词）来生成最终结果。在这个任务中，模型将输出一个二进制值（0 表示负面评价，1 表示正面评价）。在遍历序列时，LSTM 模型保持迄今为止所见到的部分的记忆。这使得 LSTM 模型非常强大，能够处理长序列并学习其中的模式。训练模型后，我们将在某些测试数据上对其进行评估以确保其表现良好，然后保存它。我们将遵循以下高级步骤来开发此模型：

1. 下载数据。我们将使用 Amazon 的一个视频游戏评论语料库。
2. 探索并清理数据。加入一些文本清理步骤（例如，词形还原）来清理数据并为建模任务准备语料库。
3. 创建一个数据管道，将原始文本转换为机器学习模型理解的数值格式。
4. 在数据管道生成的数据上训练模型。
5. 在验证集和测试集上评估模型，以确保模型的泛化能力。
6. 保存训练好的模型，并写下性能结果。

## 9.1 这段文字是什么？探索和处理文字

你正在为一个流行的在线视频游戏商店构建情感分析器。他们希望比星级数量更多一些，因为星级数量可能由于星级的主观性而不能准确反映情感。高管们相信文字比星级更有价值。你被要求开发一个情感分析器，可以根据文本确定评论的积极或消极程度。

你决定使用亚马逊视频游戏评论数据集。它包含用户发布的各种评论以及星级数量。由于语言的复杂性、拼写错误等原因，文本可能会非常嘈杂。因此，某种类型的预处理将充当生成干净数据的守门人。在本节中，我们将检查数据和一些基本统计数据。然后，我们将执行几个预处理步骤：包括、转换为小写（例如，将“约翰”转换为“john”）、删除标点符号/数字、删除停用词（即无信息的词，如“to”、“the”、“a”等）和词形还原（将单词转换为其基本形式；例如，“walking”转换为“walk”）。

作为第一步，让我们在下一个清单中下载数据集。

清单 9.1 下载亚马逊评论数据集

```py
import os
import requests
import gzip
import shutil

# Retrieve the data
if not os.path.exists(os.path.join('data','Video_Games_5.json.gz')):      ❶
    url = 
➥ "http:/ /deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Video_Games_
➥ 5.json.gz"
    # Get the file from web
    r = requests.get(url)

    if not os.path.exists('data'):
        os.mkdir('data')

    # Write to a file
    with open(os.path.join('data','Video_Games_5.json.gz'), 'wb') as f:
        f.write(r.content)
else:                                                                     ❷
    print("The tar file already exists.")

if not os.path.exists(os.path.join('data', 'Video_Games_5.json')):        ❸
    with gzip.open(os.path.join('data','Video_Games_5.json.gz'), 'rb') as f_in:
        with open(os.path.join('data','Video_Games_5.json'), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
else:
    print("The extracted data already exists")
```

❶ 如果 gzip 文件尚未下载，请下载并保存到磁盘上。
❷ 如果 gzip 文件位于本地磁盘中，则无需下载。
❸ 如果 gzip 文件存在但尚未解压，请解压它。

这段代码将数据下载到本地文件夹（如果尚未存在）并提取内容。它将包含一个包含数据的 JSON 文件。JSON 是一种用于表示数据的格式，主要用于在 Web 请求中传输数据。它允许我们将数据定义为键值对。如果你查看 JSON 文件，你会看到每行有一条记录，每条记录都是一组键值对，键是列名，值是该记录的该列的值。你可以从数据中提取出几条记录：

```py
{"overall": 5.0, "verified": true, "reviewTime": "10 17, 2015", 
➥ "reviewerID": "xxx", "asin": "0700026657", "reviewerName": "xxx", 
➥ "reviewText": "This game is a bit hard to get the hang of, but when you 
➥ do it's great.", "summary": "but when you do it's great.", 
➥ "unixReviewTime": 1445040000}
{"overall": 4.0, "verified": false, "reviewTime": "07 27, 2015", 
➥ "reviewerID": "xxx", "asin": "0700026657", "reviewerName": "xxx", 
➥ "reviewText": "I played it a while but it was alright. The steam was a 
➥ bit of trouble. The more they move ... looking forward to anno 2205 I 
➥ really want to play my way to the moon.", "summary": "But in spite of 
➥ that it was fun, I liked it", "unixReviewTime": 1437955200}
{"overall": 3.0, "verified": true, "reviewTime": "02 23, 2015", 
➥ "reviewerID": "xxx", "asin": "0700026657", "reviewerName": "xxx", 
➥ "reviewText": "ok game.", "summary": "Three Stars", "unixReviewTime": 
➥ 1424649600}
```

接下来，我们将进一步探索我们拥有的数据：

```py
import pandas as pd

# Read the JSON file
review_df = pd.read_json(
    os.path.join('data', 'Video_Games_5.json'), lines=True, orient='records'
)
# Select on the columns we're interested in 
review_df = review_df[["overall", "verified", "reviewTime", "reviewText"]]
review_df.head()
```

数据是以 JSON 格式呈现的。pandas 提供了一个`pd.read_json()`函数来轻松读取 JSON 数据。在读取 JSON 数据时，你必须确保正确设置`orient`参数。这是因为`orient`参数使 pandas 能够理解 JSON 数据的结构。与具有更一致结构的 CSV 文件相比，JSON 数据是非结构化的。设置`orient='records'`将使 pandas 能够正确读取以这种方式结构化的数据（每行一个记录）到一个 pandas DataFrame 中。运行上述代码片段将产生表 9.1 中所示的输出。

表 9.1 Amazon 评论数据集的示例数据

| | **整体** | **验证** | **评论时间** | **评论文本** |
| --- | --- | --- | --- | --- |
| 0 | 5 | True | 10 17, 2015 | 这个游戏有点难以掌握，但... |
| 1 | 4 | False | 07 27, 2015 | 我玩了一会儿，还行。... |
| 2 | 3 | True | 02 23, 2015 | 好吧，游戏还可以。 |
| 3 | 2 | True | 02 20, 2015 | 觉得这个游戏有点太复杂，不是我想要的... |
| 4 | 5 | True | 12 25, 2014 | 好游戏，我喜欢它，从那时起就一直在玩... |

我们现在将删除评论文本列中的任何空值或 null 值记录：

```py
review_df = review_df[~review_df["reviewText"].isna()]
review_df = review_df[review_df["reviewText"].str.strip().str.len()>0]
```

正如你可能已经注意到的，有一列显示评论是否来自验证购买者。为了保护我们数据的完整性，让我们只考虑来自验证购买者的评论。但在此之前，我们必须确保在过滤未经验证的评论后有足够的数据。为此，让我们看看不同值（即 True 和 False）的验证列有多少记录。为此，我们将使用 pandas 的内置`value_counts()`函数，如下所示：

```py
review_df["verified"].value_counts()
```

这将会返回

```py
True     332504
False    164915
Name: verified, dtype: int64
```

这是个好消息。看起来我们从验证购买者那里得到的数据比未验证用户的数据要多。让我们创建一个名为`verified_df`的新 DataFrame，其中只包含验证过的评论：

```py
verified_df = review_df.loc[review_df["verified"], :]
```

接下来，我们将评估整体列中每个不同评分的评论数量：

```py
verified_df["overall"].value_counts()
```

这将会给出

```py
5    222335
4     54878
3     27973
1     15200
2     12118
Name: overall, dtype: int64
```

这是一个有趣的发现。通常，我们希望对每种不同的评分有相等数量的数据。但这在现实世界中从来都不是这样的。例如，在这里，我们有比 4 星评价多四倍的 5 星评价。这被称为*类别不平衡*。现实世界的数据通常是嘈杂的、不平衡的和肮脏的。当我们进一步研究数据时，我们将看到这些特征。在开发我们的模型时，我们将回到数据中类别不平衡的问题。

Sentiment analysis 被设计为一个分类问题。给定评论（例如，作为单词序列），模型预测一组离散类别中的一个类别。我们将专注于两个类别：积极或消极。我们将假设 5 或 4 星表示积极情感，而 3、2 或 1 星表示消极情感。精明的问题形式化，比如减少类别数量，可以使分类任务变得更容易。为此，我们可以使用方便的内置 pandas 函数`map()`。`map()`接受一个字典，其中键表示当前值，值表示当前值需要映射到的值：

```py
verified_df["label"]=verified_df["overall"].map({5:1, 4:1, 3:0, 2:0, 1:0})
```

现在让我们在转换后检查每个类别的实例数量

```py
verified_df["label"].value_counts()
```

将返回

```py
1    277213
0     55291
Name: label, dtype: int64
```

积极样本约占 83%，消极样本约占 17%。这在样本数量上存在明显的不一致。我们简单的数据探索的最后一步是确保数据没有顺序。为了对数据进行洗牌，我们将使用 pandas 的`sample()`函数。`sample()`技术上是用于从大数据集中抽样一小部分数据的。但通过设置`frac=1.0`和固定的随机种子，我们可以以随机的方式获取全部数据集的洗牌：

```py
verified_df = verified_df.sample(frac=1.0, random_state=random_seed)
```

最后，我们将输入和标签分别分开成两个变量，因为这将使下一步的处理更容易：

```py
inputs, labels = verified_df["reviewText"], verified_df["label"]
```

接下来，我们将专注于一项关键任务，这将最终改善进入模型的数据的质量：清理和预处理文本。在这里，我们将专注于执行以下子任务。您将在接下来的讨论中了解每个子任务的更多细节：

+ 将单词的大小写转换为小写。
+ 处理单词的缩写形式（例如，“aren’t”、“you’ll”等）。
+ 将文本标记为单词（称为 *分词*）。
+ 移除不相关的文本，如数字、标点和停用词。停用词是文本语料库中频繁出现但对于大多数自然语言处理任务来说其存在的价值不足以证明的单词（例如，“and”、“the”、“am”、“are”、“it”、“he”、“she”等）。
+ Lemmatize words. 词形还原是将单词转换为其基本形式的过程（例如，将复数名词转换为单数名词，将过去式动词转换为现在式动词）。

要执行大多数这些任务，我们将依赖于一个著名且广为人知的用于文本处理的 Python 库，称为 NLTK（自然语言工具包）。如果您已设置开发环境，应该已安装 NLTK 库。但我们的工作还没有完成。为了执行一些子任务，我们需要下载 NLTK 提供的几个外部资源：

+ `averaged_perceptron_tagger`—用于识别词性
+ `wordnet` *和* `omw-1.4` 将被用于词形还原（即，将单词转换为其基本形式）
+ `stopwords`—提供各种语言的停用词列表
