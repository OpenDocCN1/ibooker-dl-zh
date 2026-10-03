

# 五、嵌入单词和类型

> 本文标题：[Natural-Language-Processing-with-PyTorch（五）](https://yifdu.github.io/2018/12/21/Natural-Language-Processing-with-PyTorch%EF%BC%88%E4%BA%94%EF%BC%89/)
> 文章作者：[Yif Du](https://yifdu.github.io/ "访问 Yif Du 的个人博客")
> 发布时间：2018 年 12 月 21 日 - 12:12
> 最后更新：2018 年 12 月 28 日 - 11:12
> 原始链接：[http://yifdu.github.io/2018/12/21/Natural-Language-Processing-with-PyTorch（五）/](https://yifdu.github.io/2018/12/21/Natural-Language-Processing-with-PyTorch%EF%BC%88%E4%BA%94%EF%BC%89/)
> 许可协议：[署名-非商业性使用-禁止演绎 4.0 国际](https://creativecommons.org/licenses/by-nc-nd/4.0/) 转载请保留原文链接及作者。

在实现自然语言处理（NLP）任务时，我们需要处理不同类型的离散类型。最明显的例子是单词。单词来自一个有限的集合（也就是词汇表）。其他离散类型的示例包括字符、部分语言标记、命名实体、命名实体类型、解析特性、产品目录中的项等等。本质上，当任何输入特征来自有限（或可数无限）集时，它都是离散类型。

将离散类型（如单词）表示为密集向量是 NLP 中深度学习成功的核心。术语“表示学习”和“嵌入”是指学习从一种离散类型到向量空间中的一点的映射。当离散类型为词时，密集向量表示称为词嵌入（word embedding）。我们在第 2 章中看到了基于计数的嵌入方法的例子，比如词频-逆文档频率（TF-IDF）。在本章中，我们主要研究基于学习或基于预测（Baroni et al.， 2014）的嵌入方法，即通过最大化特定学习任务的目标来学习表征；例如，根据上下文预测一个单词。基于学习的嵌入方法由于其广泛的适用性和性能而在法理上受到限制。事实上，单词嵌入在 NLP 任务中的普遍性为它们赢得了“NLP 的 Sriracha”的称号，因为您可以在任何 NLP 任务中使用单词嵌入，并期望任务的性能得到改进。但是我们认为这种绰号是误导的，因为与 Sriracha 不同，嵌入（embeddings）通常不是作为事后添加到模型中的，而是模型本身的基本组成部分。

在这一章中，我们讨论与嵌入词相关的向量表示：嵌入词的方法，优化嵌入词的方法，用于监督和非监督语言任务，可视化嵌入词的方法，以及组合嵌入词的句子和文件的方法。但是，您必须记住，我们在这里描述的方法适用于任何离散类型。

## 为什么要学习嵌入？

在前几章中，您看到了创建单词向量表示的传统方法。具体来说，您了解到可以使用单热表示—与词汇表大小相同的长度的向量，除了单个位置以外，其他地方都是 0，值为 1 表示特定的单词。此外，您还看到了计数表示——向量的长度与模型中唯一单词的数量相同，但是在向量中与句子中单词的频率相对应的位置上有计数。基于计数的表示也称为分布表示，因为它们的重要内容或意义是由向量中的多个维度表示的。分布表示具有悠久的历史（Firth, 1935），可以很好地用于许多机器学习和神经网络模型。这些表示，不是从数据中学习的，而是启发式构建的。

分布式表示的名称来源于这样一个事实：由于单词现在由低得多的密集向量表示（比如`d = 100`，而不是整个词汇的大小，可以是大约 105 到 106 或更高），并且一个单词的含义和其他属性分布在这个密集向量的不同维度上。

低维的学习密集表示比我们在前几章中看到的基于单热和计数的向量有几个优点。首先，降低维度是计算效率。其次，基于计数的表示导致高维向量在多个维度上冗余地编码相似的信息，并且不共享统计强度。第三，输入中维数过高会导致机器学习和优化中的实际问题，这通常被称为“维数灾难”。“传统上，为了处理这个维度问题，我们使用了像 PCA/SVD 这样的降维方法，但有点讽刺的是，当维度以百万的顺序排列时（典型的 NLP 情况），这些方法不能很好地进行缩放。”第四，从特定于任务的数据中学习（或微调）的表示形式最适合手头的任务。使用 TF-IDF 等启发式方法或 SVD 等低维方法，不清楚嵌入方法的优化目标是否与任务相关。

### 嵌入的有效性

为了理解嵌入是如何工作的，让我们看一个用单热向量乘以线性层中的权重矩阵的例子，如图 5-1 所示。在第 3 章和第 4 章中，单热向量的大小与词汇表相同。这个向量被称为“单热”，因为它在索引中有一个 1，表示存在一个特定的单词。

![onehot](img/cabb9e708a2dd06f3ebc33ec3d1e8702.jpg)

根据定义，接受这个单热向量作为输入的线性层的权值矩阵必须与这个单热向量的大小具有相同的行数。当您执行矩阵乘法时，如图 5-1 所示，结果向量实际上只是选择了由非零项指示的行。从这个观察结果中，我们可以跳过乘法步骤，直接使用整数作为检索所选行的索引。

关于嵌入效率的最后一点注意事项是，尽管图 5-1 中的示例显示的权重矩阵与传入的单热向量具有相同的维度，但情况并非总是如此。实际上，嵌入通常用于在低维空间中表示单词，而不是使用一个单热向量或基于计数的表示。在研究文献中，嵌入的典型尺寸从 25 维到 500 维不等，准确的选择可以归结为您必须节省的图形处理器单元（GPU）内存。

### 学习单词嵌入的方式

本章的目标不是教授特定的单词嵌入技术，而是理解什么是嵌入，如何以及在哪里使用它们，如何在模型中可靠地使用它们，以及理解它们的局限性。我们之所以这样做，是因为实践者很少发现自己处于需要编写嵌入训练算法的新单词的情况下。然而，在这一节中，我们将简要概述当前训练单词嵌入的方法。所有的单词嵌入方法都是用单词来训练的。但是是以一种监督的方式。这可以通过构造辅助监督任务来实现，在这些任务中，数据被隐式地标记，直观地认为，优化后用于解决辅助任务的表示形式将捕获文本语料库的许多统计和语言属性，以便普遍使用。以下是一些辅助任务的例子：

*   给出一个单词序列，预测下一个单词。这也称为语言建模任务。
*   给定单词前后的顺序，预测缺失的单词。
*   给定一个单词，预测出现在窗口中的单词，独立于位置。

当然，这个列表是不完整的，辅助任务的选择取决于算法设计者的直觉和计算费用。例如 Glove、连续词袋、Skipgrams 等等。我们参考 Goldberg, 2017，第 10 章，但我们将简要研究 CBOW 模型。但是，对于大多数目的来说，使用预先训练好的单词嵌入并对它们进行微调就足够了。

### 预训练词嵌入的实践用法

本章的大部分内容，以及本书的后面部分，都涉及到使用经过预先训练的单词嵌入。使用前面描述的许多方法中的一种，可以免费下载和使用预先训练过的单词嵌入、在大型语料（类似于所有新闻、维基百科和通用爬行器）上进行训练的单词嵌入。本章的其余部分将展示如何有效地加载和查找这些嵌入，研究词嵌入的一些属性，并展示在 NLP 任务中使用预先训练的嵌入的示例。

#### 加载嵌入

词嵌入已经变得非常流行和普及，您可以从原始的 Word2Vec、Stanford 的 GLoVE、Facebook 的 FastText 和许多其他版本下载许多不同的版本。通常，嵌入将以以下格式出现：每行以嵌入的单词/类型开始，然后是一系列数字（即，向量表示）。这个序列的长度就是表示的维数（也就是嵌入维数）。嵌入维数通常是数百。标记（token）类型的数量通常是词汇表的大小，以百万计。例如，这里是来自 GloVe 的狗和猫向量的前七个维度：`dog: -1.242 -0.360 0.573 0.367 0.600 -0.189 1.273 … cat: -0.964 -0.610 0.674 0.351 0.413 -0.212 1.380 …`

为了有效地加载和处理嵌入，我们描述了一个实用工具类，即预训练。该类在内存中构建所有单词向量的索引，以方便使用近似的最近邻包进行快速查找和最近邻查询。这个类显示在示例 5-1 中的第一个输入中。

**示例 5-1：使用预训练词嵌入**

```py
Input[0]
import numpy as np
from annoy import AnnoyIndex

class PreTrainedEmbeddings(object):
    def __init__(self, word_to_index, word_vectors):
        """
        Args:
            word_to_index (dict): mapping from word to integers
            word_vectors (list of numpy arrays)
        """
        self.word_to_index = word_to_index
        self.word_vectors = word_vectors
        self.index_to_word = \
            {v: k for k, v in self.word_to_index.items()}
        self.index = AnnoyIndex(len(word_vectors[0]),
                                metric='euclidean')
        for _, i in self.word_to_index.items():
            self.index.add_item(i, self.word_vectors[i])
        self.index.build(50)

    @classmethod
    def from_embeddings_file(cls, embedding_file):
        """Instantiate from pre-trained vector file.

        Vector file should be of the format:
            word0 x0_0 x0_1 x0_2 x0_3 ... x0_N
            word1 x1_0 x1_1 x1_2 x1_3 ... x1_N

        Args:
            embedding_file (str): location of the file
        Returns:
            instance of PretrainedEmbeddings
        """
        word_to_index = {}
        word_vectors = []
        with open(embedding_file) as fp:
            for line in fp.readlines():
                line = line.split(" ")
                word = line[0]
                vec = np.array([float(x) for x in line[1:]])

                word_to_index[word] = len(word_to_index)
                word_vectors.append(vec)
        return cls(word_to_index, word_vectors)
Input[1]
embeddings = \
    PreTrainedEmbeddings.from_embeddings_file('glove.6B.100d.txt')
```

在这些例子中，我们使用 Glove 词嵌入。下载它们之后，可以使用`PretrainedEmbeddings`类进行实例化，如示例 5-1 中的第二个输入所示。

## 词嵌入之间的关系

词嵌入的核心特征是对句法和语义关系进行编码，这些句法和语义关系表现为词的使用规律。例如，人们谈论猫和狗的方式非常相似（讨论宠物、喂食等）。因此，它们的嵌入彼此之间的距离比它们与其他动物（如鸭子和大象）的距离要近得多。我们可以从几个方面探讨嵌入词编码的语义关系。最流行的一种方法是类比任务（SAT 等考试中常见的推理任务）：`Word1 : Word2 :: Word3 : __`

在这个任务中，你被提供了前三个单词，需要确定第四个单词与前两个单词之间的关系是一致的。使用嵌入词，我们可以对其进行空间编码。首先，我们从单词 1 中减去单词 2。这个差异向量编码了`Word1`和`Word2`之间的关系。然后可以将这个差异添加到`Word3`中，从而生成一个接近第四个单词的向量，即空白符号所在的位置。使用这个结果向量对索引执行最近邻查询可以解决类比问题。计算这个的函数，如例 5-2 所示，完成了刚刚描述的工作：使用向量算法和近似的最近邻索引，完成类比。

**示例 5-2：使用词嵌入的类比任务**

```py
Input[0]
import numpy as np
from annoy import AnnoyIndex

class PreTrainedEmbeddings(object):
    """ implementation continued from previous code box"""
    def get_embedding(self, word):
        """
        Args:
            word (str)
        Returns
            an embedding (numpy.ndarray)
        """
        return self.word_vectors[self.word_to_index[word]]

    def get_closest_to_vector(self, vector, n=1):
        """Given a vector, return its n nearest neighbors

        Args:
            vector (np.ndarray): should match the size of the vectors
                in the Annoy index
            n (int): the number of neighbors to return
        Returns:
            [str, str, ...]: words nearest to the given vector.
                The words are not ordered by distance
        """
        nn_indices = self.index.get_nns_by_vector(vector, n)
        return [self.index_to_word[neighbor]
                   for neighbor in nn_indices]

    def compute_and_print_analogy(self, word1, word2, word3):
        """Prints the solutions to analogies using word embeddings

        Analogies are word1 is to word2 as word3 is to __
        This method will print: word1 : word2 :: word3 : word4

        Args:
            word1 (str)
            word2 (str)
            word3 (str)
        """
        vec1 = self.get_embedding(word1)
        vec2 = self.get_embedding(word2)
        vec3 = self.get_embedding(word3)

        # Simple hypothesis: Analogy is a spatial relationship
        spatial_relationship = vec2 - vec1
        vec4 = vec3 + spatial_relationship

        closest_words = self.get_closest_to_vector(vec4, n=4)
        existing_words = set([word1, word2, word3])
        closest_words = [word for word in closest_words
                              if word not in existing_words]

        if len(closest_words) == 0:
            print("Could not find nearest neighbors for the vector!")
            return

        for word4 in closest_words:
            print("{} : {} :: {} : {}".format(word1, word2, word3,
                                              word4))
```

有趣的是，简单的单词类比任务可以证明单词嵌入捕获了各种语义和语法关系，如示例 5-3 所示。

**示例 5-3：一组语言关系以向量类比编码**

