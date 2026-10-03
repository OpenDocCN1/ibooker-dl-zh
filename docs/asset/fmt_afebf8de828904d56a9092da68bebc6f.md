
然后，你可以构建词汇表，就像我们在第二章中所做的那样，如下所示：

```py
vocab = Vocabulary.from_instances(
    text8, min_count={'token_in': 5, 'token_out': 5})
```

注意，我们使用了`min_count`参数，该参数设置了每个`token`出现的最低限制。此外，让我们按照以下方式定义用于训练的数据加载器：

```py
data_loader = SimpleDataLoader(text8, batch_size=BATCH_SIZE)
data_loader.index_with(vocab)
```

然后，我们定义一个包含所有要学习的词嵌入的嵌入对象：

```py
embedding_in = Embedding(num_embeddings=vocab.get_vocab_size('token_in'),
                         embedding_dim=EMBEDDING_DIM)
```

这里，`EMBEDDING_DIM`是每个单词向量的长度（浮点数的数量）。一个典型的 NLP 应用程序使用几百维的单词向量（在本例中为 256），但该值在任务和数据集上有很大的依赖性。通常建议随着训练数据的增长使用更长的单词向量。

最后，你需要实现 Skip-gram 模型的主体，如下所示。

**清单 3.1 在 AllenNLP 中实现的 Skip-gram 模型。**

```py
class SkipGramModel(Model):                                   ❶

    def __init__(self, vocab, embedding_in):
        super().__init__(vocab)

        self.embedding_in = embedding_in                      ❷

        self.linear = torch.nn.Linear(
            in_features=EMBEDDING_DIM,
            out_features=vocab.get_vocab_size('token_out'),
            bias=False)                                       ❸

    def forward(self, token_in, token_out):                   ❹

        embedded_in = self.embedding_in(token_in)             ❺

        logits = self.linear(embedded_in)                     ❻

        loss = functional.cross_entropy(logits, token_out)    ❼

        return {'loss': loss}
```

❶ AllenNLP 要求每个模型都必须继承自`Model`。

❷ 嵌入对象从外部传入，而不是在内部定义。

❸ 这创建了一个线性层（请注意我们不需要偏置）。

❹ 神经网络计算的主体在`forward()`中实现。

❺ 将输入张量（单词 ID）转换为单词嵌入。

❻ 应用线性层。

❼ 计算损失。

注意以下几点：

+   AllenNLP 要求每个模型都必须继承自`allennlp.models.Model`。

+   模型的初始化函数（`__init__`）接受一个`Vocabulary`实例和定义在外部的任何其他参数或子模型。它还定义任何内部的参数或模型。

+   模型的主要计算是在`forward()`中定义的。它将实例中的所有字段（在这个例子中是`token_in`和`token_out`）作为张量（多维数组），并返回一个包含`'loss'`键的字典，该键将被优化器用于训练模型。

你可以使用以下代码来训练这个模型。

**列表 3.2 训练 Skip-gram 模型的代码**

```py
reader = SkipGramReader()
text8 = reader.read(' https:/./realworldnlpbook.s3.amazonaws.com/data/text8/text8')

vocab = Vocabulary.from_instances(
    text8, min_count={'token_in': 5, 'token_out': 5})

data_loader = SimpleDataLoader(text8, batch_size=BATCH_SIZE)
data_loader.index_with(vocab)

embedding_in = Embedding(num_embeddings=vocab.get_vocab_size('token_in'),
                         embedding_dim=EMBEDDING_DIM)

model = SkipGramModel(vocab=vocab,
                      embedding_in=embedding_in)
optimizer = optim.Adam(model.parameters())

trainer = GradientDescentTrainer(
    model=model,
    optimizer=optimizer,
    data_loader=data_loader,
    num_epochs=5,
    cuda_device=CUDA_DEVICE)
trainer.train()
```

训练需要一段时间，所以我建议首先截取训练数据，比如只使用前一百万个标记。你可以在`reader.read()`后插入`text8 = list(text8)[:1000000]`。训练结束后，你可以使用列表 3.3 中展示的方法获取相关词（具有相同意义的词）。这个方法首先获取给定词（标记）的词向量，然后计算它与词汇表中的每个其他词向量的相似度。相似性是使用所谓的*余弦相似度*来计算的。简单来说，余弦相似度是两个向量之间角度的反义词。如果两个向量相同，那么它们之间的角度就是零，相似度就是 1，这是可能的最大值。如果两个向量是垂直的，角度是 90 度，余弦相似度就是 0。如果向量完全相反，余弦相似度就是-1。

**列表 3.3 使用词嵌入获取相关词的方法**

```py
def get_related(token: str, embedding: Model, vocab: Vocabulary, 
                num_synonyms: int = 10):
    token_id = vocab.get_token_index(token, 'token_in')
    token_vec = embedding.weight[token_id]
    cosine = CosineSimilarity(dim=0)
    sims = Counter()

    for index, token in vocab.get_index_to_token_vocabulary('token_in').items():
        sim = cosine(token_vec, embedding.weight[index]).item()
        sims[token] = sim

    return sims.most_common(num_synonyms)
```

如果你对“one”和“december”这两个词运行这个模型，你将得到表 3.1 中展示的相关词列表。虽然你可能会看到一些与查询词无关的词，但整体上，结果看起来很不错。

**表 3.1 “one” 和 “december”的相关词**

| “一” | “十二月” |
| --- | --- |
| 一 | 十二月 |
| 九 | 一月 |
| 八 | 尼克斯 |
| 六 | 伦敦 |
| 五 | 植物 |
| 七 | 六月 |
| 三 | 斯密森 |
| 四 | 二月 |
| d | 卡努尼 |
| 女演员 | 十月 |

最后一点说明：如果你想要在实践中使用 Skip-gram 来训练高质量的词向量，你需要实现一些技术，即负采样和高频词子采样。虽然它们是重要的概念，但如果你刚开始学习，并想了解自然语言处理的基础知识，它们可能会分散你的注意力。如果你对了解更多感兴趣，请查看我在这个主题上写的这篇博客文章：[`realworldnlpbook.com/ch3.html#word2vec-blog`](http://realworldnlpbook.com/ch3.html#word2vec-blog)。

### 3.4.6 连续词袋（CBOW）模型

通常与 Skip-gram 模型一起提到的另一个词嵌入模型是*连续词袋*（CBOW）*模型*。作为 Skip-gram 模型的近亲，同时提出（[`realworldnlpbook.com/ch3.html# mikolov13`](http://realworldnlpbook.com/ch3.html#mikolov13)），CBOW 模型的结构与 Skip-gram 模型相似，但上下颠倒。该模型试图解决的“假”任务是从一组上下文词预测目标词。这与填空题类型的问题类似。例如，如果你看到一个句子“I heard a ___ barking in the distance.”，大多数人可能会立即猜到答案“dog”。图 3.8 显示了该模型的结构。

![CH03_F08_Hagiwara](img/CH03_F08_Hagiwara.png)

**图 3.8 连续词袋 (CBOW) 模型**

我不打算在这里从头实现 CBOW 模型，有几个原因。如果你理解了 Skip-gram 模型，实现 CBOW 模型应该很简单。此外，CBOW 模型在词语语义任务上的准确性通常略低于 Skip-gram 模型，并且 CBOW 在 NLP 中使用的频率也较低。这两个模型都在原始的 Word2vec ([`code .google.com/archive/p/word2vec/`](https://code.google.com/archive/p/word2vec/)) 工具包中实现，如果你想自己尝试它们，尽管如此，由于更近期、更强大的词嵌入模型的出现（例如 GloVe 和 fastText），这些基本的 Skip-gram 和 CBOW 模型现在使用得越来越少，这些模型在本章的其余部分中介绍。

## 3.5 GloVe

在前一节中，我实现了 Skip-gram，并展示了如何利用大量文本数据训练词嵌入。但是如果你想构建自己的 NLP 应用程序，利用高质量的词嵌入，同时避开所有麻烦呢？如果你无法承担训练词嵌入所需的计算和数据呢？

与训练词嵌入相反，你总是可以下载其他人发布的预训练词嵌入，这是许多 NLP 从业者做的。在本节中，我将介绍另一种流行的词嵌入模型——*GloVe*，名为 *Global Vectors*。由 GloVe 生成的预训练词嵌入可能是当今 NLP 应用中最广泛使用的嵌入。

### 3.5.1 GloVe 如何学习词嵌入

之前描述的两个模型与 GloVe 的主要区别在于前者是*局部的*。简而言之，Skip-gram 使用预测任务，其中上下文词（“bark”）从目标词（“dog”）预测。CBOW 基本上做相反的事情。这个过程重复了数据集中的每个单词标记的次数。它基本上扫描整个数据集，并询问：“这个词可以从另一个词预测吗？”对于数据集中每个单词的每次出现都会问这个问题。

让我们思考一下这个算法的效率。如果数据集中有两个或更多相同的句子呢？或者非常相似的句子呢？在这种情况下，Skip-gram 将重复多次相同的一组更新。你可能会问，“‘bark’ 可以从 ‘dog’ 预测吗？” 但很有可能你在几百个句子前已经问过了完全相同的问题。如果你知道 “dog” 和 “bark” 这两个词在整个数据集中共同出现了 *N* 次，那为什么要重复这 *N* 次呢？这就好像你在把 “1” 加 *N* 次到另一个东西上（x + 1 + 1 + 1 + ... + 1），而你其实可以直接加 *N* 到它上面（x + N）。我们能否直接利用这个 *全局* 信息呢？

GloVe 的设计受到这一见解的启发。它不是使用局部单词共现，而是使用整个数据集中的聚合单词共现统计信息。假设 “dog” 和 “bark” 在数据集中共同出现了 *N* 次。我不会深入讨论模型的细节，但粗略地说，GloVe 模型试图从两个单词的嵌入中预测这个数字 *N*。图 3.9 描绘了这个预测任务。它仍然对单词关系进行了一些预测，但请注意，它对于每个单词 *类型* 的组合进行了一次预测，而 Skip-gram 则对每个单词 *标记* 的组合进行了预测！

![CH03_F09_Hagiwara](img/CH03_F09_Hagiwara.png)

**图 3.9 GloVe**

标记和类型 如第 3.3.1 节所述，*标记* 是文本中单词的出现。一个语料库中可能会有同一个单词的多次出现。另一方面，*类型* 是一个独特的、唯一的词。例如，在句子 “A rose is a rose is a rose.” 中，有八个标记但只有三种类型（“a”，“rose”，和 “is”）。如果你熟悉面向对象编程，它们大致相当于实例和类。一个类可以有多个实例，但一个概念只有一个类。

### 3.5.2 使用预训练的 GloVe 向量

实际上，并不是很多自然语言处理（NLP）从业者自己从头开始训练 GloVe 向量。更常见的是，我们下载并使用预训练的词向量，这些词向量是使用大型文本语料库预训练的。这不仅快捷，而且通常有助于使您的 NLP 应用程序更准确，因为这些预训练的词向量（通常由词向量算法的发明者公开）通常是使用比我们大多数人负担得起的更大的数据集和更多的计算资源进行训练的。通过使用预训练的词向量，您可以“站在巨人的肩膀上”，并快速利用从大型文本语料库中提炼出的高质量语言知识。

在本节的其余部分，让我们看看如何使用预先训练的 GloVe 嵌入下载并搜索相似单词。首先，您需要下载数据文件。官方的 GloVe 网站（[`nlp.stanford.edu/projects/glove/`](https://nlp.stanford.edu/projects/glove/)）提供了使用不同数据集和向量大小训练的多个词嵌入文件。您可以选择任何一个（尽管取决于您选择的文件大小可能很大）并解压缩它。在接下来的内容中，我们假设您将其保存在相对路径`data/glove/`下。

大多数词嵌入文件都以相似的方式格式化。每一行都包含一个单词，后跟一系列与其单词向量对应的数字。数字的数量与维度一样多（在上述网站上分发的 GloVe 文件中，您可以从文件名后缀中以 xxx 维的形式了解维度）。每个字段由一个空格分隔。以下是 GloVe 词嵌入文件的摘录：

```py
...
if 0.15778 0.17928 -0.45811 -0.12817 0.367 0.18817 -4.5745 0.73647 ...
one 0.38661 0.33503 -0.25923 -0.19389 -0.037111 0.21012 -4.0948 0.68349 ...
has 0.08088 0.32472 0.12472 0.18509 0.49814 -0.27633 -3.6442 1.0011 ...
...
```

正如我们在第 3.4.5 节中所做的那样，我们想要做的是接受一个查询词（比如，“狗”）并在*N*维空间中找到它的邻居。一种方法是计算查询词与词汇表中的每个其他词之间的相似性，并按其相似性对单词进行排序，如清单 3.3 所示。根据词汇表的大小，这种方法可能非常缓慢。这就像线性扫描数组以找到元素而不是使用二进制搜索一样。

相反，我们将使用近似最近邻算法快速搜索相似单词。简而言之，这些算法使我们能够快速检索最近的邻居，而无需计算每个单词对之间的相似性。具体而言，我们将使用 Annoy（[`github.com/spotify/annoy`](https://github.com/spotify/annoy)）库，这是来自 Spotify 的用于近似最近邻搜索的库。您可以通过运行`pip install annoy`来安装它。它使用随机投影实现了一种流行的近似最近邻算法称为*局部敏感哈希*（LSH）。

要使用 Annoy 搜索相似的单词，您首先需要构建一个索引，可以按照清单 3.4 所示进行。请注意，我们还正在构建一个从单词索引到单词的字典，并将其保存到单独的文件中，以便以后方便进行单词查找（清单 3.5）。

**清单 3.4 构建 Annoy 索引**

```py
from annoy import AnnoyIndex
import pickle

EMBEDDING_DIM = 300
GLOVE_FILE_PREFIX = 'data/glove/glove.42B.300d{}'

def build_index():
    num_trees = 10

    idx = AnnoyIndex(EMBEDDING_DIM)

    index_to_word = {}
    with open(GLOVE_FILE_PREFIX.format('.txt')) as f:
        for i, line in enumerate(f):
            fields = line.rstrip().split(' ')
            vec = [float(x) for x in fields[1:]]
            idx.add_item(i, vec)
            index_to_word[i] = fields[0]

    idx.build(num_trees)
    idx.save(GLOVE_FILE_PREFIX.format('.idx'))
    pickle.dump(index_to_word,
                open(GLOVE_FILE_PREFIX.format('.i2w'), mode='wb'))
```
