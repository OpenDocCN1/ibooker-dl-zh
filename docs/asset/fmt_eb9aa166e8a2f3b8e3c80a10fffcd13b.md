
```py
>>> word_vectors.most_similar(positive=['cooking', 'potatoes'], topn=5)
[('cook', 0.6973530650138855),
 ('oven_roasting', 0.6754530668258667),
 ('Slow_cooker', 0.6742032170295715),
 ('sweet_potatoes', 0.6600279808044434),
 ('stir_fry_vegetables', 0.6548759341239929)]
>>> word_vectors.most_similar(positive=['germany', 'france'], topn=1)
[('europe', 0.7222039699554443)]
```

单词向量模型还允许您确定不相关的术语。`gensim`库提供了一个名为`doesnt_match`的方法：

```py
>>> word_vectors.doesnt_match("potatoes milk cake computer".split())
'computer'
```

为了确定列表中最不相关的术语，该方法返回与所有其他列表术语的距离最大的术语。

如果您想执行计算（例如著名的例子*king + woman - man = queen*，这是最初引起 Mikolov 和他的顾问兴奋的例子），您可以通过向`most_similar`方法调用添加`negative`参数来实现：

```py
>>> word_vectors.most_similar(positive=['king', 'woman'],
...     negative=['man'], topn=2)
[('queen', 0.7118192315101624), ('monarch', 0.6189674139022827)]
```

`gensim`库还允许您计算两个术语之间的相似度。如果您想比较两个单词并确定它们的余弦相似度，请使用方法`.similarity()`：

```py
>>> word_vectors.similarity('princess', 'queen')
0.70705315983704509
```

如果您想要开发自己的函数并使用原始单词向量进行工作，您可以通过 Python 的方括号语法（`[]`）或`KeyedVector`实例上的`get()`方法访问它们。您可以将加载的模型对象视为字典，其中您感兴趣的单词是字典键。返回的数组中的每个浮点数代表一个向量维度。在谷歌的词模型中，您的 numpy 数组的形状将为 1x300。

```py
>>> word_vectors['phone']
array([-0.01446533, -0.12792969, -0.11572266, -0.22167969, -0.07373047,
       -0.05981445, -0.10009766, -0.06884766,  0.14941406,  0.10107422,
       -0.03076172, -0.03271484, -0.03125   , -0.10791016,  0.12158203,
        0.16015625,  0.19335938,  0.0065918 , -0.15429688,  0.03710938,
        ...
```

如果您想知道所有这些数字*意味着*什么，您可以找到答案。但这需要大量的工作。您需要检查一些同义词，并查看它们在数组中共享的 300 个数字中的哪些。或者，您可以找到这些数字的线性组合，构成像“位置”和“女性”之类的维度，就像您在本章的开头所做的那样。

### 6.4.6 生成自己的词向量表示

在某些情况下，您可能希望创建自己的特定领域的单词向量模型。这样做可以提高您的模型准确性，如果您的 NLP 管道正在处理使用词汇方式与 Google News 中 2006 年以前 Mikolov 训练的参考`word2vec`模型不同的文档，则会更加如此。请记住，您需要大量的文档来做到这一点，就像 Google 和 Mikolov 一样。但是，如果您的词在 Google News 上特别罕见，或者您的文本在受限领域内以独特的方式使用它们，比如医学文本或转录文本，则特定于领域的单词模型可能会提高您的模型准确性。在接下来的部分中，我们将向您展示如何训练您自己的`word2vec`模型。

为了训练一个特定于领域的`word2vec`模型，您将再次转向`gensim`，但在您开始训练模型之前，您需要使用第二章中发现的工具对语料库进行预处理。

#### 预处理步骤

首先，您需要将文档分成句子，然后将句子分成标记。`gensim`的`word2vec`模型期望得到一个句子列表，其中每个句子被分成标记。这可以防止单词向量学习邻近句子中的无关单词出现。您的训练输入应该类似于以下结构：

```py
>>> token_list
[
  ['to', 'provide', 'early', 'intervention/early', 'childhood', 'special',
   'education', 'services', 'to', 'eligible', 'children', 'and', 'their',
   'families'],
  ['essential', 'job', 'functions'],
  ['participate', 'as', 'a', 'transdisciplinary', 'team', 'member', 'to',
   'complete', 'educational', 'assessments', 'for']
  ...
]
```

要将句子分割成标记，然后将句子转换为标记，您可以应用第二章中学到的各种策略。让我们再增加一个：Detector Morse 是一个句子分段器，它改进了 NLTK 和`gensim`中提供的准确性分段器的某些应用场景。[43] 它已经在《华尔街日报》的多年文本中进行了预训练。因此，如果您的语料库包含与《华尔街日报》类似的语言，那么 Detector Morse 很可能会为您提供目前可能的最高准确性。如果您拥有来自您领域的大量句子集合，还可以在自己的数据集上重新训练 Detector Morse。一旦您将文档转换为标记列表（每个句子一个列表），您就可以开始进行`word2vec`训练了。

#### 训练您的领域特定的`word2vec`模型

通过加载*word2vec*模块开始：

```py
>>> from gensim.models.word2vec import Word2Vec
```

训练需要一些设置细节。

##### 列表 6.2 控制 word2vec 模型训练的参数

```py
>>> num_features = 300  # #1
>>> min_word_count = 3  # #2
>>> num_workers = 2  # #3
>>> window_size = 6  # #4
>>> subsampling = 1e-3  # #5
```

现在，您可以开始培训了。

##### 列表 6.3 实例化 word2vec 模型

```py
>>> model = Word2Vec(
...     token_list,
...     workers=num_workers,
...     size=num_features,
...     min_count=min_word_count,
...     window=window_size,
...     sample=subsampling)
```

根据您的语料库大小和 CPU 性能，训练将需要相当长的时间。对于较小的语料库，训练可以在几分钟内完成。但是对于一个全面的词模型，语料库将包含数百万句子。您需要有关语料库中所有不同单词的所有不同用法的几个示例。如果开始处理较大的语料库，例如维基百科语料库，预期训练时间会更长，并且内存消耗量会更大。

另外，Word2Vec 模型可能会消耗大量内存。但请记住，只有隐藏层的权重矩阵才感兴趣。一旦训练了您的词模型，如果您冻结模型并丢弃不必要的信息，您可以将内存占用减少约一半。以下命令将丢弃神经网络的不需要的输出权重：

```py
>>> model.init_sims(replace=True)
```

`init_sims`方法将冻结模型，存储隐藏层的权重并丢弃预测单词共现的输出权重。输出权重不是大多数 Word2Vec 应用程序所使用的向量的一部分。但是一旦丢弃了输出层的权重，模型就无法再进一步训练。

您可以使用以下命令保存已训练的模型，并将其保留以供以后使用：

```py
>>> model_name = "my_domain_specific_word2vec_model"
>>> model.save(model_name)
```

如果要测试您新训练的模型，可以使用与前一节学到的相同方法。

##### 列表 6.4 加载已保存的`word2vec`模型

```py
>>> from gensim.models.word2vec import Word2Vec
>>> model_name = "my_domain_specific_word2vec_model"
>>> model = Word2Vec.load(model_name)
>>> model.most_similar('radiology')
```

## 6.5 Word2Vec 替代方案

Word2Vec 是一项突破，但它依赖于必须使用反向传播进行训练的神经网络模型。自 Mikolov 首次推广词嵌入以来，研究人员提出了越来越准确和高效的方式来将单词的含义嵌入向量空间。

1.  Word2vec

1.  GloVE

1.  fastText

由杰弗里·彭宁顿（Jeffrey Pennington）领导的斯坦福自然语言处理研究人员^([44])开始研究 Word2Vec 为什么效果那么好，并找到正在优化的成本函数。他们首先统计单词共现次数并将其记录在一个方阵中。他们发现他们可以计算这个共现矩阵的奇异值分解（SVD）^([45])，将其拆分为 Word2Vec 生成的相同两个权重矩阵。^([46]) 关键是以相同方式对共现矩阵进行归一化。但在某些情况下，Word2Vec 模型无法收敛到斯坦福研究人员用 SVD 方法实现的相同全局最优解。正是对单词共现的全局向量（整个语料库中的共现）进行直接优化赋予了 GloVe 其名称。

GloVe 可以生成等效于 Word2Vec 输入权重矩阵和输出权重矩阵的矩阵，生成具有与 Word2Vec 相同准确性但时间大大缩短的语言模型。GloVe 通过更有效地利用文本数据来加快处理速度。GloVe 可以在较小的语料库上进行训练，仍然能够收敛。^([47]) SVD 算法已经被改进了几十年，因此 GloVe 在调试和算法优化方面有了一个优势。Word2Vec 依赖于反向传播来更新形成单词嵌入的权重。神经网络的反向传播比在 SVD 中使用的更成熟的优化算法效率低。

即使 Word2Vec 首先将通过单词向量进行语义推理的概念普及开来，你的得力助手应该是 GloVe 来训练新的单词向量模型。通过 GloVe，你更有可能找到这些向量表示的全局最优解，从而获得更准确的结果。而且 spaCy 将其作为默认的嵌入算法，因此当你运行:

```py
>>> import spacy
>>>
>>> nlp = spacy.load("en_core_web_sm")
>>> text = "This is an example sentence."
>>> doc = nlp(text)
>>>
>>> for token in doc:
...    print(token.text, token.vector)
```

这些结果是使用 GloVe 计算得出的！

GloVe 的优点：

+   训练速度更快

+   更好的 RAM/CPU 效率（可以处理更大的文档）

+   更有效地利用数据（有助于处理较小的语料库）

+   在相同的训练量下更准确

### 6.5.1 fastText

来自 Facebook 的研究人员将 Word2Vec 的概念推进了一步^([48])，通过向模型训练添加了一个新的技巧。他们称之为 fastText 的新算法预测周围的*n*-*字符* n-gram 而不仅仅是周围的单词，就像 Word2Vec 所做的那样。例如，单词"whisper"会生成以下 2 个和 3 个字符的 n-gram：

```py
['wh', 'whi', 'hi', 'his', 'is', 'isp', 'sp', 'spe', 'pe', 'per', 'er']
```

fastText 然后对每个*n*字符 n-gram（称为“子词”）进行向量表示的训练，包括单词、拼写错误的单词、部分单词，甚至单个字符。这种方法的优势在于，它比原始的 Word2Vec 方法更好地处理罕见或新的单词。

fastText 分词器将为较长的单词的两半创建向量，如果该较长的单词的使用频率远低于构成它的子词。例如，如果您的语料库只提到 "Superwoman" 一两次，但却成千上万次使用 "super" 和 "woman"，那么 fastText 可能会为 "super" 和 "woman" 创建向量。如果您的 fastText 语言模型在训练结束后在现实世界中遇到单词 "Superwoman"，它将合并 "Super" 和 "woman" 的向量以创建单词 "Superwoman" 的向量。这减少了 fastText 需要分配通用的 Out of Vocabulary（OOV）向量的单词数量。在您的 NLU 流水线的 "思维" 中，OOV 单词向量看起来像 "未知单词"。它的效果与您在完全陌生的语言中听到一个外来单词的效果相同。虽然 Word2vec 只能 "知道" 如何嵌入它以前见过的单词，但 fastText 由于其子词方法而更加灵活。它还相对轻量且运行速度更快。

作为 fastText 发布的一部分，Facebook 发布了 294 种语言的预训练 fastText 模型。在 Facebook 研究的 Github 页面^([49])上，您可以找到从 *阿布哈兹语* 到 *祖鲁语* 的模型。模型集合甚至包括罕见的语言，如 *萨特兰弗里西亚语*，该语言仅由少数德国人使用。Facebook 提供的预训练 fastText 模型仅在可用的维基百科语料库上进行了训练。因此，模型的词汇量和准确性会因语言而异。

我们在 `nessvec` 包中包含了用于为 OOV 单词创建新向量的 fastText 逻辑。我们还通过 Peter Norvig 的著名的优雅拼写纠正算法^([50])增强了 fastText 流水线，以处理拼写错误和错别字。这将为您提供最佳选择，即一个易于理解的训练算法和当您需要在现实世界中使用您训练过的向量时，一个强大的推断或预测模型。

#### 使用预训练模型增强您的 NLP

通过利用来自地球上最强大的公司的开源预训练嵌入来提升您的 NLP 流水线的能力。预训练的 fastText 向量几乎可以在任何可想象的语言中找到。如果您想查看您的词嵌入的所有可用选项，请查看 fastText 模型存储库（`github.com/facebookresearch/fastText/blob/main/docs/pretrained-vectors.md`）。而对于多语言支持，您可以在 fastText 嵌入的 Common Crawl 版本中找到许多支持的 157 种语言的组合模型（`fasttext.cc/docs/en/crawl-vectors.html`）。如果您愿意，您可以使用 fastText 页面上的 *bin+text* 链接下载您语言的所有不同版本的嵌入。但如果您想节省一些时间，只需下载 100 万...

##### 警告

*bin+text* `wiki.en.zip` 文件（`dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip`）大小为 *9.6 GB*。仅文本 `wiki.en.vec` 文件大小为 *6.1 GB*（`dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec`）。如果您使用的是 `nessvec` 包而不是 `gensim`，它将仅下载大小为 600MB 的 `wiki-news-300d-1M.vec.zip` 文件（`dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip`）。该 `wiki-news-300d-1M.vec.zip` 包含来自维基百科和新闻网页中最受欢迎的 100 万个单词（不区分大小写）的 300-D 向量。

`nessvec` 包将创建一个内存映射的 `DataFrame`，其中包含所有预训练向量。内存映射文件（`.hdf5`）通过按需惰性加载你所需的向量，可以防止你的计算机内存（RAM）不足。

```py
>>> from nessvec.files import load_fasttext
>>> df = load_fasttext()  # #1
>>> df.head().round(2)
      0     1     2    ...   297   298   299
,    0.11  0.01  0.00  ...  0.00  0.12 -0.04
the  0.09  0.02 -0.06  ...  0.16 -0.03 -0.03
.    0.00  0.00 -0.02  ...  0.21  0.07 -0.05
and -0.03  0.01 -0.02  ...  0.10  0.09  0.01
of  -0.01 -0.03 -0.03  ...  0.12  0.01  0.02
>>> df.loc['prosocial']  # #2
0      0.0004
1     -0.0328
2     -0.1185
        ...
297    0.1010
298   -0.1323
299    0.2874
Name: prosocial, Length: 300, dtype: float64
```
