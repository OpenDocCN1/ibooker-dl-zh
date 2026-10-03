
```py
Epoch: 1, loss: 0.66147, Train accuracy: 0.61392, Test accuracy: 0.63648
Epoch: 2, loss: 0.64491, Train accuracy: 0.69712, Test accuracy: 0.70735
Epoch: 3, loss: 0.55865, Train accuracy: 0.73391, Test accuracy: 0.74278
Epoch: 4, loss: 0.38538, Train accuracy: 0.76558, Test accuracy: 0.77165
Epoch: 5, loss: 0.27227, Train accuracy: 0.79288, Test accuracy: 0.77690
Epoch: 6, loss: 0.29682, Train accuracy: 0.82119, Test accuracy: 0.78609
Epoch: 7, loss: 0.23429, Train accuracy: 0.82951, Test accuracy: 0.79003
```

仅仅经过 7 次通过训练数据集，您就在测试集上实现了 79% 的准确率。在现代笔记本电脑 CPU 上，这应该不到一分钟。并且通过最小化模型中的总参数，您将过拟合保持到最低。与嵌入层相比，CNN 使用的参数非常少。

如果您继续训练一段时间会发生什么？

##### 代码清单 7.21 继续训练

```py
pipeline.epochs = 13  # #1
pipeline = pipeline.train()
```

```py
Epoch: 1, loss: 0.24797, Train accuracy: 0.84528, Test accuracy: 0.78740
Epoch: 2, loss: 0.16067, Train accuracy: 0.86528, Test accuracy: 0.78871
...
Epoch: 12, loss: 0.04796, Train accuracy: 0.93578, Test accuracy: 0.77690
Epoch: 13, loss: 0.13394, Train accuracy: 0.94132, Test accuracy: 0.77690
```

哦，这看起来很可疑。过拟合太严重了——在训练集上达到了 94%，在测试集上达到了 78%。训练集准确率不断上升，最终超过了 90%。到了第 20 个 epoch，模型在训练集上的准确率达到了 94%。它甚至比专家人类还要好。自己阅读几个示例，不看标签，你能得到其中的 94% 吗？这是前四个示例，经过令牌化后，忽略了词汇表外的词汇，并添加了填充。

```py
pipeline.indexes_to_texts(pipeline.x_test[:4])
```

```py
['getting in the poor girl <PAD> <PAD> ...',
 'Spot Flood Combo Cree LED Work Light Bar Offroad Lamp Full ...',
 'ice the meltdown <PAD> <PAD> <PAD> <PAD> ...',
 'and burn for bush fires in St http t co <PAD> <PAD> ...']
```

如果你的答案是`["disaster", "not", "not", "disaster"]`，那你全部答对了。但继续努力吧。你能做到十九对二十吗？这就是你需要在训练集准确率上击败这个卷积神经网络所需要做到的。这不是什么意外，因为机器人一直在推特上发布听起来像是灾难的推文。有时甚至真实的人类也会对世界事件感到讽刺或煽动性。

是什么导致了这种过拟合？是参数太多了吗？神经网络的"容量"太大了吗？以下是一个好的函数，用于显示 PyTorch 神经网络每层的参数。

```py
>>> def describe_model(model):  # #1
...     state = model.state_dict()
...     names = state.keys()
...     weights = state.values()
...     params = model.parameters()
>>>     df = pd.DataFrame()
>>>     df['name'] = list(state.keys())
>>>     df['all'] = p.numel(),
...     df['learned'] = [
...         p.requires_grad  # #2
...         for p in params],  # #3
...     size=p.size(),
...     )
        for name, w, p in zip(names, weights, params)
    ]
    )
    df = df.set_index('name')
    return df

describe_model(pipeline.model)  # #4
```

```py
learned_params  all_params        size
name
embedding.weight             128064      128064  (2001, 64)  # #1
linear_layer.weight            1856        1856   (1, 1856)
linear_layer.bias                 1           1        (1,)
```

当你遇到过拟合问题时，你可以在管道中使用预训练模型来改善其泛化能力。

### 7.4.3 迁移学习

另一个可以帮助你的 CNN 模型的优化方法是使用预训练的词嵌入，如 GloVe。这并不是作弊，因为这些模型是以无监督的方式训练的，没有使用你的灾难推文数据集的任何标签。你可以将这些 GloVe 向量中所包含的所有学习迁移到您训练模型所使用的与灾难相关的词汇集上，通过使用单词的更一般的含义。你只需要调整嵌入层的大小，以容纳你希望用来初始化 CNN 的 GloVe 嵌入的大小。

##### 代码清单 7.22：为 GloVE 嵌入腾出空间

```py
>>> from torch import nn
>>> embedding = nn.Embedding(
...     num_embeddings=2000,  # #1
...     embedding_dim=50,  # #2
...     padding_idx=0)
```

就是这样。一旦 PyTorch 知道嵌入的数量和它们的维度，它就可以分配内存来保存嵌入矩阵，其中有 `num_embedding` 行和 `embedding_dim` 列。这将同时训练你的嵌入和其余的 CNN 部分。你的领域特定的词汇和嵌入将根据你的语料库进行自定义。但是，从头开始训练你的嵌入没有利用到单词在许多领域中共享的含义。

如果你希望你的管道能"跨域适应"，你可以使用在其他领域训练的嵌入。这种词嵌入的"跨训练"被称为*迁移学习*。通过使用在更广泛的文本语料库上训练的预训练词嵌入，这为你的嵌入层提前了解了单词的含义。为此，你需要过滤掉其他领域中使用的所有单词，以便你的 CNN 管道的词汇仅基于你的数据集中的单词。然后，你可以将这些单词的嵌入加载到你的`nn.Embedding`层中。

##### 代码清单 7.23：加载嵌入并与你的词汇对齐

```py
>>> from nessvec.files import load_vecs_df
>>> glove = load_vecs_df(HOME_DATA_DIR / 'glove.6B.50d.txt')
>>> zeroes = [0.] * 50
>>> embed = []
>>> for tok in vocab:  # #1
...     if tok in glove.index:
...         embed.append(glove.loc[tok])
...     else:
...         embed.append(zeros.copy())  # #2
>>> embed = np.array(embed)
>>> embed.shape
(4000, 50)
```

现在你已经将 4000 个标记的词汇表转换为一个 4000×5 的嵌入矩阵。`embed`数组中的每一行都表示一个具有 50 维向量的词汇表标记的含义。如果在你的词汇表中一个标记的 GloVe 嵌入不存在，那么它将有一个全为零的向量。这本质上使得那个标记对于理解你的文档毫无用处，就像一个 OOV（词汇表外）标记一样。

```py
>>> pd.Series(vocab)
0               a
1              in
2              to
          ...
3831         43rd
3832    beginners
3833        lover
Length: 3834, dtype: object
```

你已经从推文中取出了最常见的 4000 个标记。在这 4000 个词中，最小的 GloVE 词嵌入词汇表中有 3834 个可用的。因此，你用零向量填充了那 166 个缺失词的未知嵌入。当你在神经网络中训练嵌入层时，你的模型会学习这些词的意义并计算它们的嵌入。

现在你有了一种将标记转换为整数的一致方法，你可以将 GloVe 嵌入矩阵加载到你的 `nn.Embedding`层中。

##### 代码清单 7.24 初始化使用 GloVE 向量的嵌入层

```py
embed = torch.Tensor(embed)  # #1
print(f'embed.size(): {embed.size()}')
embed = nn.Embedding.from_pretrained(embed, freeze=False)  # #2
print(embed)
```

#### 检测有意义的模式

你说话的方式、单词的顺序，都很重要。你组合单词以创建对你来说具有重要意义的模式，以便将那个意义传达给其他人。

如果你希望你的机器成为一个有意义的自然语言处理器，它需要能够检测到更多不仅仅是特定标记的存在或不存在。你希望你的机器能够检测到隐藏在单词序列中的有意义的模式。^([23])

卷积是过滤器，它可以从单词中提取有意义的模式。最好的部分是，你不再需要将这些模式硬编码到卷积核中。训练过程将搜索最佳的模式匹配卷积，以解决你遇到的问题。每次将标记数据的错误通过网络向后传递时（反向传播），优化器会调整每个过滤器中的权重，使它们在检测意义和分类文本示例方面变得越来越好。

### 7.4.4 使用丢弃进行卷积神经网络的健壮性增强

大多数神经网络容易受到对抗样本的影响，这些样本会欺骗它们输出错误的分类或文本。有时，神经网络容易受到同义词替换、拼写错误或俚语插入等简单变化的影响。有时候只需要一点"语词沙拉"——无意义的随机词语——就能分散并困惑 NLP 算法。人类知道如何忽略噪音和过滤干扰，但机器有时会遇到麻烦。

*鲁棒化自然语言处理（NLP）*是研究处理来自不同来源的非常规文本的方法和技术。事实上，鲁棒化 NLP 的研究可能会揭示通往人工通用智能的路径。人类能够从极少的例子中学习新词和概念。而我们的泛化能力很好，既不过多也不过少。机器需要一点帮助。如果你能找出我们人类擅长之处的"秘密酱料"，然后将其编码到 NLP 流程中，那你就能够让机器具备类似的能力。

增强神经网络鲁棒性的一种常见技术是*随机丢弃法*。由于其简便性和有效性，*随机丢弃法*或简称*丢弃法*已经变得非常流行。你的神经网络几乎总会从丢弃层中受益。丢弃层会随机隐藏部分神经元的输出，使其不被其他神经元接收。这会导致你人造脑中的某条路径变得静音，并迫使其他神经元在丢弃期间学习当前的特定示例。

这似乎与直觉相悖，但丢弃法有助于使你的神经网络学习更广泛。如果没有丢弃层，你的网络将专注于帮助提高最大准确度的单词、模式和卷积滤波器。但你需要神经元们扩展他们的模式，以便你的网络能够对自然语言文本上的常见变化保持"健壮性"。

在神经网络中安装丢弃层的最佳位置是靠近末尾，就在运行完全连接的线性层之前。这个向量通过线性层传递的权重是来自 CNN 和池化层的输出。每个值代表一系列单词或意义和语法模式。通过隐藏一些模式，迫使你的预测层扩展其"思考"。虽然你的软件并没有真正考虑什么，但如果将其拟人化一点能够帮助你对为什么随机丢弃等技术可以提高模型准确度产生直觉。

## 7.5 使用 PyTorch CNN 处理灾难推文

现在进入有趣的部分。你要构造一个真实世界的 CNN，可以区分真实新闻和煽动性报道。你的模型可以帮助你过滤掉与文化战争有关的推文，让你专注于来自真实战区的新闻。

首先，你将看到新的卷积层在管道中的位置。然后你将组装所有组件，来训练一个基于"灾难推文"数据集的 CNN。如果负能量滚动和灾难不是你的菜的话，这个 CNN 也很容易适应任何带标签的推文数据集。你甚至可以选择一个你喜欢的话题作为目标标签，即使推文的作者不知道如何使用标签，你也可以找到与该标签话题相匹配的推文。

### 7.5.1 网络架构

这里是 CNN NLP 管道的每个阶段的处理步骤和张量的相应形状。构建新 CNN 中最棘手的事情之一是跟踪您张量的形状。您需要确保一个层的输出形状与下一层的输入形状相匹配，对于此示例与以前的示例相同。

1.  令牌化⇒`(N_, )`

1.  填充⇒`(N,)`

1.  嵌入⇒`(M, N)`

1.  卷积(s)⇒`(M, N - K)`

1.  激活⇒`(M, N - K)`

1.  池化⇒`(M, N - K)`

1.  丢弃（可选）⇒`(M, N - K)`

1.  线性组合⇒`(L, )`

1.  Argmax、softmax 或阈值化⇒`(L, )`

和

+   `N_`是您输入文本中的标记数。

+   `N`是您填充序列中的标记数。

+   `M`是您的单词嵌入中的维度数。

+   `K`是您的核大小。

+   `L`是您想要预测的类标签或值的数量。

您的 CNN 的 PyTorch 模型比第 5 和第六章中的要多一些超参数。然而，与以前一样，将超参数设置在`CNNTextClassifier`模型的`*init*`构造函数中是一个好主意。

##### 列表 7.25 CNN 超参数

```py
class CNNTextClassifier(nn.Module):

    def __init__(self, embeddings):
        super().__init__()

        self.seq_len = 40  # #1
        self.vocab_size = 10000  # #2
        self.embedding_size = 50  # #3
        self.out_channels = 5  # #4
        self.kernel_lengths = [2, 3, 4, 5, 6]  # #5
        self.stride = 1  # #6
        self.dropout = nn.Dropout(0)  # #7
        self.pool_stride = self.stride  # #8
        self.conv_out_seq_len = calc_out_seq_len(  # #9
            seq_len=self.seq_len,
            kernel_lengths=self.kernel_lengths,
            stride=self.stride,
            )
```

就像本章前面手工制作的卷积一样，每个卷积操作都会减少序列长度。缩短的量取决于内核的大小和步幅。`Conv1d`层的 PyTorch 文档提供了这个公式和对术语的详细解释。^([25])

```py
def calc_conv_out_seq_len(seq_len, kernel_len,
                          stride=1, dilation=1, padding=0):
    """
    L_out =     (L_in + 2 * padding - dilation * (kernel_size - 1) - 1)
            1 + _______________________________________________________
                                        stride
    """
    return (
        1 + (seq_len +
             2 * padding - dilation * (kernel_len - 1) - 1
            ) //
        stride
        )
```

您的第一个 CNN 层是一个`nn.Embedding`层，它将一系列单词 ID 整数转换为一系列嵌入向量。它的行数与词汇表中唯一标记的数量相同（包括新的填充标记）。它的每个嵌入向量的维度都有一列。您可以从 GloVe 或任何其他预训练的嵌入中加载这些嵌入向量。

##### 列表 7.26 初始化 CNN 嵌入

```py
self.embed = nn.Embedding(
    self.vocab_size,  # #1
    self.embedding_size,  # #2
    padding_idx=0)
state = self.embed.state_dict()
state['weight'] = embeddings  # #3
self.embed.load_state_dict(state)
```
