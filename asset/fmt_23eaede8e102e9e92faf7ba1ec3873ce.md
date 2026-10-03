
词向量对于聊天机器人和搜索引擎也非常有用。对于这些应用，词向量可以帮助克服模式的僵硬、脆弱性或关键字匹配的一些问题。比如，你正在搜索关于得克萨斯州休斯顿的著名人物的信息，但你不知道他们搬到了达拉斯。从图 6.3 中，你可以看到，使用词向量进行语义搜索可以轻松处理包含丹佛和休斯顿等城市名称的搜索。即使基于字符的模式无法理解“给我介绍一个丹佛煎蛋”的不同于“给我介绍丹佛掘金队”，但基于词向量的模式可以。基于词向量的模式很可能能够区分食物项（煎蛋）和篮球队（掘金队），并据此恰当地回应用户的任何问题。

### 6.4.2 学习词向量

词嵌入是表示单词意义（语义）的向量。然而，单词的意义是一个难以捉摸的模糊事物。一个孤立的个体单词有一个非常模糊的意义。以下是一些可能影响单词意义的事情：

+ 被传达的思想是谁的
+ 该单词的理解对象
+ 单词被使用的上下文（何地和何时）
+ 假定的领域知识或背景知识
+ 所需表达的单词意义

你的大脑可能会以与我的大脑完全不同的方式理解一个词。随着时间的推移，你大脑中的一个词的含义也会发生变化。当你将一个词与其他概念建立新联系时，你会对这个词了解到新的东西。随着你学习新概念和新词汇，你会根据新词对你大脑的印象学习到这些新词的新联系。嵌入用于表示你大脑中由新词创建的神经元连接的演变模式。而这些新向量有数百个维度。

想象一个年轻的女孩说：“我妈妈是医生。”^([34]) 想象一下对她来说，“医生”这个词意味着什么。然后想象一下，随着她的成长，她对这个词的理解，她的自然语言理解处理算法是如何演变的。随着时间的推移，她将学会区分医生（M.D.）和哲学博士（Ph.D.）。想象一下，当她自己开始考虑申请医学院或博士项目的可能性时，这个词对她来说意味着什么。想象一下，这个词对她的父亲或母亲，即医生，意味着什么。想象一下，这个词对一个没有医疗保健资源的人来说意味着什么。

创建有用的单词数值表示是棘手的。你想要编码或嵌入向量中的含义不仅取决于你想要代表的含义，还取决于何时何地你希望你的机器处理和理解这个含义。在 `GloVe`、`Word2Vec` 和其他早期单词嵌入的情况下，目标是表示“平均”或最流行的含义。创建这些表示的研究人员关注的是类比问题和其他衡量人类和机器对单词理解的基准测试。例如，我们在本章前面的代码片段中使用了预训练的 `fastText` 单词嵌入。

##### 提示

预训练的单词向量表示可用于诸如维基百科、`DBPedia`、`Twitter` 和 `Freebase` 等语料库。^([35]) 这些预训练模型是你的单词向量应用的绝佳起点。

+ 谷歌提供了基于英文 Google 新闻文章的预训练`word2vec`模型。^([36])
+ Facebook 发布了他们的词模型，称为*fastText*，涵盖了 294 种语言。^([37])

幸运的是，一旦你决定了词嵌入的“受众”或“用户”，你只需要收集这些单词的用法示例。`Word2Vec`、`GloVe` 和 `fastText` 都是无监督学习算法。你只需要一些来自你和你的用户感兴趣领域的原始文本。如果你主要关注医生，你可以训练你的嵌入在医学期刊的一系列文本上。或者，如果你想要最普遍的单词表示的理解，机器学习工程师通常使用维基百科和在线新闻文章来捕捉单词的含义。毕竟，维基百科代表了我们对世界上一切的集体理解。

现在，当你拥有你的语料库时，你如何为你的词嵌入语言模型创建训练集呢？在早期，主要有两种方法：

1. *连续词袋模型*（`CBOW`）
2. 连续的*skip-gram*

*连续的词袋模型*（`CBOW`）方法从附近的上下文单词（输入单词）预测目标单词（输出或“目标”单词）。与第三章中学习的词袋（`BOW`）向量唯一的区别在于，`CBOW` 是针对每个文档内的连续滑动窗口的单词创建的。因此，你将有几乎与所有文档中的单词序列中的单词数量相同的 `CBOW` 向量。而对于 `BOW` 向量，你只有每个文档一个向量。这为你的词嵌入训练集提供了更多信息以使其生成更准确的嵌入向量。使用 `CBOW` 方法，你可以从原始文档中提取的每个可能短语创建大量的微小合成文档。

##### 图 6.4 CBOW 神经网络架构

![word2vec cbow whatever affects one drawio](img/word2vec-cbow-whatever-affects-one_drawio.png)

对于 skip-gram 方法，在这种方法中，你也会创建大量合成文档。你只需反转预测目标，这样就可以使用 `CBOW` 目标来预测 `CBOW` 特征。它预测与感兴趣的单词（输入单词）相关的上下文单词（“目标”单词）。尽管这些看起来像是你的词对被颠倒了，但你很快就会发现结果几乎是数学上等价的。

##### 图 6.5 Skip-gram 神经网络架构

![word2vec skip gram whatever affects one drawio](img/word2vec-skip-gram-whatever-affects-one_drawio.png)

你可以看到，两种神经网络方法产生相同数量的训练样本，并为 skip-gram 和 `CBOW` 方法创建相同数量的训练样本。

#### Skip-gram 方法

在 skip-gram 训练方法中，你预测上下文单词附近的单词。想象一下，你的语料库包含了 Bayard Rustin 和 Larry Dane Brimner 关于个人主义的明智拒绝。^([38])

> 我们都是一个整体。如果我们不知道这一点，那我们将会以艰难的方式发现它。
> 
> ——Bayard Rustin
> 
> *《We Are One: The Story of Bayard Rustin》，2007，第 46 页*，Larry Dane Brimner

##### 定义

*skip-gram* 是一个 2-gram 或两个 gram，其中每个 gram 都在彼此的附近。通常情况下，这些 grams 可以是您的分词器设计为预测的文本块 - 通常是单词。

对于连续的 skip-gram 训练方法，skip-grams 是跳过零到四个单词以创建 skip-gram 对的单词对。使用 `Word2Vec` skip-gram 方法训练单词嵌入时，skip-gram 中的第一个单词称为“上下文”单词。上下文单词是输入到 `Word2Vec` 神经网络中的。skip-gram 对中的第二个单词通常被称为“目标”单词。目标单词是语言模型和嵌入向量被训练以预测的单词 - 输出。

##### 图 6.6 跳过-gram 方法的训练输入和输出示例

![we are all one drawio](img/we-are-all-one_drawio.png)

在图 6.6 中，您可以看到 skip-gram 方法创建单词嵌入的神经网络架构是什么样子的。

#### softmax 是什么？

`softmax` 函数通常用作神经网络输出层的激活函数，当网络的目标是学习分类问题时。`softmax` 将输出结果压缩在 0 到 1 之间，所有输出节点的总和始终加起来为 1。因此，具有 `softmax` 函数的输出层的结果可以被视为概率。

对于每个*K* 输出节点，可以使用归一化指数函数计算 `softmax` 输出值：

![equation 6 3](img/equation_6_3.png)

如果您的三神经元输出层的输出向量如下所示：

**方程 6.3 例子 3D 向量**

![equation 6 4](img/equation_6_4.png)

`softmax` 激活后的“压缩”向量将如下所示：

**方程 6.4 例子 3D 向量经过 softmax 后**

![equation 6 5](img/equation_6_5.png)

请注意，这些值的总和（四舍五入到 3 个有效数字）约为 1.0，就像概率分布一样。

图 6.4 显示了前两个周围单词的数字网络输入和输出。在这种情况下，输入单词是“Monet”，网络的预期输出是“Claude”或“painted”，这取决于训练对。

##### 图 6.7 跳过-gram 训练的网络示例

![skipgram](img/skipgram.png)

##### 注意

当您查看单词嵌入的神经网络结构时，您会注意到实现看起来与您在第五章中发现的内容相似。

### 6.4.3 学习没有字典的含义

对于这个 `Word2Vec` 训练示例，您不需要使用字典，比如 `wiktionary.org` 来明确定义单词的含义。相反，您可以只让 `Word2Vec` 读取包含有意义的句子的文本。您将使用 PyTorch 中 `torchtext` 包中提供的 `WikiText2` 语料库。

```py
>>> import torchtext

>>> dsets = torchtext.datasets.WikiText2()
>>> num_texts = 10000

>>> filepath = DATA_DIR / f'WikiText2-{num_texts}.txt'
>>> with open(filepath, 'wt') as fout:
...     fout.writelines(list(dsets[0])[:num_texts])
```

为了让它更不神秘，您可以查看您刚刚从`WikiText2`数据集中创建的包含约 10,000 个段落的文本文件：

```py
>>> !tail -n 3 ~/nessvec-data/WikiText2-10000.txt

When Marge leaves Dr. Zweig 's office , she says ,
" Whenever the wind whistles through the leaves ,
I 'll think , Lowenstein , Lowenstein … " .
This is a reference to The Prince of Tides ; the <unk> is Dr. Lowenstein .

= = Reception = =
```

第 99,998 段碰巧包含缩写"Dr."。在这种情况下，缩写是为了单词"doctor"。您可以利用这个来练习您的"妈妈是一名医生"直觉泵。因此，您很快就会发现 `Word2Vec` 是否能学会什么是医生。或者它可能会因为使用"Dr."表示"drive"的街道地址而感到困惑。

方便的是，`WikiText2` 数据集已经将文本分词成单词。单词之间用单个空格(`" "`)字符分隔。因此，您的管道不必决定"Dr."是否是句子的结尾。如果文本没有被分词，您的 NLP 管道将需要删除所有句子末尾的句号。甚至标题分隔符文本`"=="`也已经被拆分为两个独立的标记`"="`和`"="`。段落由换行(`"\n"`)字符分隔。对于维基百科标题如`"== 接待 =="`，将创建许多"段落"，同时保留段落之间的所有空行。

您可以利用像 `SpaCy` 这样的句子边界检测器或句子分割器将段落分割成句子。这将防止您的 `Word2Vec` 训练对从一个句子溢出到另一个句子。尊重句子边界的 `Word2Vec` 可以提高词嵌入的准确性。但是我们将把这个决定留给您，看您是否需要额外的准确性提升。

这里的管道可以处理的一个关键基础设施是大型语料库的内存管理。如果您正在对数百万段落进行词嵌入的训练，您将需要使用一个管理磁盘上文本的数据集对象，只加载需要的部分到 RAM 或 GPU 中。Hugging Face Hub 的`datasets`包可以为您处理这个问题：

```py
>>> import datasets
>>> dset = datasets.load_dataset('text', data_files=str(filepath))
>>> dset
DatasetDict({
    train: Dataset({
        features: ['text'],
        num_rows: 10000
    })
})
```

但是，您仍然需要告诉 `Word2Vec` 什么是单词。这是您需要担心的唯一"监督"`Word2Vec` 数据集。您可以使用第二章中最简单的分词器来实现良好的结果。对于这种空格分词的文本，您只需使用`str.split()`方法。您可以使用`str.lower()`进行大小写折叠，将您的词汇表大小减半。令人惊讶的是，这已经足够让 `Word2Vec` 学会单词的含义和内涵，以至于能够解决类似 SAT 测试中可能会看到的类比问题，并且甚至能够推理现实世界的对象和人。

```py
def tokenize_row(row):
    row['all_tokens'] = row['text'].lower().split()
    return row
```

现在，您可以在包含数据行迭代序列的 `torchtext` 数据集上使用您的分词器，每行数据都有一个用于 `WikiText2` 数据的"text"键。

```py
>>> dset = dset.map(tokenize_row)
>>> dset

DatasetDict({
    train: Dataset({
        features: ['text', 'tokens'],
        num_rows: 10000
    })
})
```

您需要为数据集计算词汇表，以处理神经网络的一热编码和解码。

```py
>>> vocab = list(set(
...     [tok for row in dset['train']['tokens'] for tok in row]))
>>> vocab[:4]
['cast', 'kaifeng', 'recovered', 'doctorate']

>>> id2tok = dict(enumerate(vocab))
>>> list(id2tok.items())[:4]
[(0, 'cast'), (1, 'kaifeng'), (2, 'recovered'), (3, 'doctorate')]

>>> tok2id = {tok: i for (i, tok) in id2tok.items()}
>>> list(tok2id.items())[:4]
[('cast', 0), ('kaifeng', 1), ('recovered', 2), ('doctorate', 3)]
```

唯一剩下的特征工程步骤是通过对令牌序列进行窗口化，然后在这些窗口内配对跳字来创建跳字对。

```py
WINDOW_WIDTH = 10

>>> def windowizer(row, wsize=WINDOW_WIDTH):
    """ Compute sentence (str) to sliding-window of skip-gram pairs. """
...    doc = row['tokens']
...    out = []
...    for i, wd in enumerate(doc):
...        target = tok2id[wd]
...        window = [
...            i + j for j in range(-wsize, wsize + 1, 1)
...            if (i + j >= 0) & (i + j < len(doc)) & (j != 0)
...        ]

...        out += [(target, tok2id[doc[w]]) for w in window]
...    row['moving_window'] = out
...    return row
```

一旦你将 `windowizer` 应用于你的数据集，它将有一个 'window' 键，其中将存储标记的窗口。

```py
>>> dset = dset.map(windowizer)
>>> dset
DatasetDict({
    train: Dataset({
        features: ['text', 'tokens', 'window'],
        num_rows: 10000
    })
})
```

这是你的 `skip_gram` 生成函数：

```py
>>> def skip_grams(tokens, window_width=WINDOW_WIDTH):
...    pairs = []
...    for i, wd in enumerate(tokens):
...        target = tok2id[wd]
...        window = [
...            i + j for j in
...            range(-window_width, window_width + 1, 1)
...            if (i + j >= 0)
...            & (i + j < len(tokens))
...            & (j != 0)
...        ]

...        pairs.extend([(target, tok2id[tokens[w]]) for w in window])
    # huggingface datasets are dictionaries for every text element
...    return pairs
```

你的神经网络只需要窗口化数据中的跳字对：
