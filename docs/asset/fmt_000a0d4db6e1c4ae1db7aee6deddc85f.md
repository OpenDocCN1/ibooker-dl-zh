
## 数据处理与管道构建

首先，我们将每个序列拆分为输入和目标，并启用预取功能：

```py
# Split each sequence to an input and a target
text_ds = tf.data.Dataset.zip(
    text_ds.map(lambda x: (x[:,:-1], x[:, 1:]))
).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)    
```

最后，完整的代码可以封装在下一个清单中的函数中。

**图 10.3** 从自由文本序列创建的 `tf.data` 管道

```py
def get_tf_pipeline(data_seq, n_seq, batch_size=64, shift=1, shuffle=True):
    """ Define a tf.data pipeline that takes a set of sequences of text and 
    convert them to fixed length sequences for the model """

    text_ds = tf.data.Dataset.from_tensor_slices(tf.ragged.constant(data_seq))❶

    if shuffle:
        text_ds = text_ds.shuffle(buffer_size=len(data_seq)//2)               ❷

    text_ds = text_ds.flat_map(                                               ❸
        lambda x: tf.data.Dataset.from_tensor_slices(
            x
        ).window(
            n_seq+1, shift=shift
        ).flat_map(
            lambda window: window.batch(n_seq+1, drop_remainder=True)
        )
    ) 

    if shuffle:
        text_ds = text_ds.shuffle(buffer_size=10*batch_size)                  ❹

    text_ds = text_ds.batch(batch_size)                                       ❺

    text_ds = tf.data.Dataset.zip(
        text_ds.map(lambda x: (x[:,:-1], x[:, 1:]))
    ).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)                     ❻

    return text_ds   
```

❶ 从`data_seq`创建的不规则张量定义一个`tf.dataset`。

❷ 如果设置了`shuffle`，对数据进行洗牌（洗牌故事顺序）。

❸ 在这里，我们从更长的序列中创建窗口，给定窗口大小和偏移量，然后使用一系列`flat_map`操作来删除在此过程中创建的嵌套。

❹ 对数据进行洗牌（洗牌窗口生成的顺序）。

❺ 批量处理数据。

❻ 将每个序列拆分为输入和目标，并启用预取。

所有这些辛勤工作，如果不看生成的数据，就不会有太多意义。

```py
ds = get_tf_pipeline(train_data_seq, 5, batch_size=6)

for a in ds.take(1):
    print(a)
```

这将向您展示：

```py
(
<tf.Tensor: shape=(6, 5), dtype=int32, numpy=
array([[161,  12,  69, 396,  17],
       [  2,  72,  77,  84,  24],
       [ 87,   6,   2,  72,  77],
       [276, 484,  57,   5,  15],
       [ 75, 150,   3,   4,  11],
       [ 11,  73, 211,  35, 141]])>, 
<tf.Tensor: shape=(6, 5), dtype=int32, numpy=
array([[ 12,  69, 396,  17,  44],
       [ 72,  77,  84,  24,  51],
       [  6,   2,  72,  77,  84],
       [484,  57,   5,  15,  67],
       [150,   3,   4,  11,  73],
       [ 73, 211,  35, 141,  98]])>
)
```

很好，你可以看到我们得到了一个张量元组作为单个批处理：输入和目标。此外，您可以验证结果的正确性，因为我们清楚地看到目标是向右移动 1 位的输入。最后一件事：我们将将相同的超参数保存到磁盘上。

- n-gram 中的 `n`
- 词汇量大小
- 序列长度

```py
print("n_grams uses n={}".format(ngrams))
print("Vocabulary size: {}".format(n_vocab))

n_seq=100
print("Sequence length for model: {}".format(n_seq))

with open(os.path.join('models', 'text_hyperparams.pkl'), 'wb') as f:
    pickle.dump({'n_vocab': n_vocab, 'ngrams':ngrams, 'n_seq': n_seq}, f)
```

在这里，我们定义序列长度`n_seq=100`；这是单个输入/标签序列中我们将拥有的二元组数目。

在本节中，我们了解了用于语言建模的数据，并定义了一个强大的`tf.data`管道，该管道可以将文本序列转换为可直接用于训练模型的输入标签序列。接下来，我们将定义一个用于生成文本的机器学习模型。

**练习 1**

你有一个序列`x`，其值为`[1,2,3,4,5,6,7,8,9,0]`。你被要求编写一个`tf.data`管道，该管道生成一个输入和目标元组，其中目标是将输入向右移动两个元素（即，输入 1 的目标是 3）。你必须这样做，以便一个单独的输入/目标具有三个元素，并且连续输入序列之间没有重叠。对于前面的序列，它应该生成`[([1,2,3], [3,4,5]), ([6,7,8], [8,9,0])]`。

## 10.2 仙境中的 GRU：使用深度学习生成文本

现在我们来到了有奖励的部分：实现一个酷炫的机器学习模型。在上一章中，我们讨论了深度时序模型。鉴于数据的时序性，你可能已经猜到我们将使用其中之一深度时序模型，比如 LSTMs。在本章中，我们将使用一个略有不同的模型，称为*门控循环单元*（GRUs）。驱动该模型计算的原理与 LSTMs 相同。为了保持我们讨论的清晰度，值得提醒自己 LSTM 模型是如何工作的。

LSTM 是一类专门设计用于处理时序数据的深度神经网络。它们逐个输入地处理一系列输入。LSTM 单元从一个输入到下一个输入，同时在每个时间步产生输出（或状态）（图 10.2）。此外，为了产生给定时间步的输出，LSTMs 使用了它产生的先前输出（或状态）。这个属性对于 LSTMs 非常重要，使它们能够随时间记忆事物。

![10-02](img/10-02.png)

**图 10.2** LSTM 模型概述以及它如何处理随时间展开的输入序列

让我们总结一下我们在上一章中学到的关于 LSTMs 的知识，因为这将帮助我们比较 LSTMs 和 GRUs。一个 LSTM 有两个状态，称为单元状态和输出状态。单元状态负责维护长期记忆，而输出状态可以被视为短期记忆。在 LSTM 单元内部，输出和中间结果受三个门的控制：

- *输入门*——控制了在给定时间步中当前输入的多少会对最终输出产生影响
- *遗忘门*——控制了多少先前的单元状态影响当前单元状态的计算
- *输出门*——控制了当前单元状态对 LSTM 模型产生的最终输出的贡献

GRU 模型是在 Cho 等人的论文“Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation”中介绍的（[`arxiv.org/pdf/1406.1078v3.pdf`](https://arxiv.org/pdf/1406.1078v3.pdf)）。GRU 模型可以被认为是 LSTM 模型的简化，同时保持了相当的性能。GRU 单元有两个门：

- *更新门*（*z*[t]）——控制了多少先前的隐藏状态传递到当前的隐藏状态
- *重置门*（*r*[t]）——控制了多少隐藏状态与新输入一起被重置

与 LSTM 单元不同，GRU 单元只有一个状态向量。总的来说，与 LSTM 模型相比，GRU 有两个主要变化：

1. 输入门和遗忘门被合并成一个称为更新门的门（*z*[t]）。输入门被计算为（1-*z*[t]），而遗忘门保持*z*[t]。
2. 与 LSMT 单元中的两个状态（即单元状态和输出状态）相比，这里只有一个状态（*h*[t]）。

图 10.3 描述了 GRU 单元的各个组件。以下是构成 GRU 单元的方程式的完整列表：

*r*[t] = σ(*W*[rh]*h*[t-1] + *W*[rx]*x*[t] + *b*[r])

*z*[t] = σ(*W*[zh]*h*[t-1] + *W*[zx]*x*[t] + *b*[z])

*h̃*[t] = tanh(*W*[h]*(rh*[t-1]) + *W*[x]*x*[t] + *b*)

*h*[t] = (*z*[th]*h*[t-1] + (1 - *z*[t] )*h̃*[t]

![10-03](img/10-03.png)

**图 10.3** GRU 单元中发生的计算概览

这次讨论对于理解 GRU 模型不仅非常有帮助，还有助于了解它与 LSTM 单元的区别。你可以在 TensorFlow 中如下定义一个 GRU 单元：

```py
tf.keras.layers.GRU(units=1024, return_state=False, return_sequences=True)
```

参数`units`、`return_state`和`return_sequences`的含义与 LSTM 单元的上下文中相同。然而，注意 GRU 单元只有一个状态；因此，如果`return_state=true`，则会将相同的状态复制以模拟 LSTM 层的输出状态和单元状态。图 10.4 显示了这些参数对于 GRU 层的作用。

![10-04](img/10-04.png)

**图 10.4** 根据 GRU 单元的`return_state`和`return_sequences`参数的不同返回结果的变化

我们现在知道了定义最终模型所需的一切（清单 10.4）。我们的最终模型将包括：

- 一个嵌入层
- 一个 GRU 层（1,024 个单元），将所有最终状态向量作为一个形状为`[批量大小，序列长度，单元数]`的张量返回
- 一个具有 512 个单元和 ReLU 激活的 Dense 层
- 一个具有`n_vocab`个单元和 softmax 激活的 Dense 层

**清单 10.4** 实现语言模型

```py
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(
        input_dim=n_vocab+1, output_dim=512,input_shape=(None,)           ❶
    ),

    tf.keras.layers.GRU(1024, return_state=False, return_sequences=True), ❷

    tf.keras.layers.Dense(512, activation='relu'),                        ❸

    tf.keras.layers.Dense(n_vocab, name='final_out'),                     ❹
    tf.keras.layers.Activation(activation='softmax')                      ❹
])
```

❶ 定义一个嵌入层以学习 bigrams 的单词向量。

❷ 定义一个 LSTM 层。

❸ 定义一个 Dense 层。

❹ 定义一个最终的 Dense 层和 softmax 激活。

你会注意到，在 GRU 后的 Dense 层接收到一个三维张量（而不是传递给 Dense 层的典型二维张量）。Dense 层足够智能，能够处理二维和三维输入。如果输入是三维的（就像在我们的情况下），那么将一个接受`[批量大小，单元数]`张量的 Dense 层通过所有步骤传递序列以生成 Dense 层的输出。还要注意我们是如何将 softmax 激活与 Dense 层分开的。这实际上等价于：

```py
.Dense(n_vocab, activation=’softmax’, name=’final_out’)
```

我们不会通过重复在清单 10.4 中显示的内容来拖延对话，因为这已经是不言而喻的。

**练习 2**

你已经获得了以下模型，并被要求在现有的 GRU 层之上添加另一个具有 512 个单元并返回所有状态输出的 GRU 层。你将对以下代码做出什么改变？

```py
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(
        input_dim=n_vocab+1, output_dim=512,input_shape=(None,)   
    ),
    tf.keras.layers.GRU(1024, return_state=False, return_sequences=True),   
    tf.keras.layers.Dense(n_vocab, activation=’softmax’, name='final_out'), ])
```

在这一节中，我们学习了门控循环单元（GRUs）以及它们与 LSTMs 的比较。最后，我们定义了一个可以在我们之前下载并处理的数据上进行训练的语言模型。在下一节中，我们将学习用于评估生成文本质量的评估指标。

## 10.3 测量生成文本的质量

性能监控已经成为我们在每一章节模型之旅中不可或缺的一部分。在这里也不例外。性能监控是我们语言模型的一个重要方面，我们需要找到适合语言模型的度量标准。自然地，考虑到这是一个分类任务，你可能会想，“准确率不是一个很好的度量标准吗？”嗯，在这个任务中不完全是这样。

例如，如果语言模型得到了句子“I like my pet dog”，然后当要求预测给定“我喜欢我的宠物 ____”时缺失的单词，模型可能会预测“猫”，准确率为零。但这是不正确的；在这个例子中，“猫”和“狗”一样有意义。这里有更好的解决方案吗？

这就是困惑度！直观地，*困惑度* 衡量了模型看到前一个词序列后看到目标词时的“惊讶程度”。在理解困惑度之前，你需要了解“熵”是什么意思。

*熵* 是由著名的克劳德·香农创造的一个术语，他被认为是信息论之父。熵度量了事件的惊奇程度/不确定性/随机性。事件可以由概率分布生成所有可能的结果之一。例如，如果你考虑抛硬币（以概率`p`出现正面）是一个事件，如果`p = 0.5`，那么你将有最大的熵，因为这是抛硬币最不确定的情况。如果`p = 1`或`p = 0`，则熵最小，因为在抛硬币之前你知道结果是什么。

熵的原始解释是发送信号或消息通知事件所需的位数的期望值。位是内存的单位，可以是 1 或 0。例如，你是一支与 A 国和 B 国交战的军队的指挥官。现在有四种可能性：A 和 B 都投降，A 赢了 B 输了，A 输了 B 赢了，以及 A 和 B 都赢了。如果所有这些事件发生的可能性都是相等的，你需要两位来发送消息，其中每一位表示那个国家是否获胜。随机变量`X`的熵由方程式量化：

![10_04a](img/10_04a.png)

其中 *x* 是 *X* 的一个结果。信不信由你，每次我们使用分类交叉熵损失时，我们都在不知不觉中使用了这个方程式。分类交叉熵的关键就在于这个方程式。回到困惑度度量，困惑度就是简单地：

*困惑度 =* 2^(H(X))

由于困惑度是熵的一个函数，它衡量了模型看到目标词时的惊讶程度/不确定性，考虑了前一个词序列。困惑度也可以被认为是给定信号的所有可能组合的数量。例如，假设你发送一个带有两位的消息，所有事件都是等可能的；那么熵 = 2，这意味着困惑度 = 2² = 4。换句话说，两位可以有四种组合：00、01、10 和 11。

从建模角度来看，你可以将困惑度理解为在给定一系列前序词的情况下，模型认为有多少不同的目标适合作为下一个词的空白。这个数字越小越好，因为这意味着模型试图从更小的子集中找到一个词，表明语言理解的迹象。

要实现困惑度，我们将定义一个自定义指标。计算非常简单。我们计算分类交叉熵，然后对其进行指数化以获得困惑度。分类交叉熵简单地是熵的扩展，用于在具有两个以上类别的分类问题中测量熵。对于输入示例（*x*[i],*y*[i]），它通常定义为：

![10_04b](img/10_04b.png)

其中*y[i]*表示真实类别示例所属的独热编码向量，*ŷ*[i]是*C*个元素的预测类别概率向量，其中*ŷ*[i,c]表示示例属于类别*c*的概率。请注意，在实践中，计算更快的是使用指数（自然）基数，而不是使用 2 作为基数。以下清单描述了这个过程。

**清单 10.5** 实现困惑度指标
