
```py
def print_sequence_length(str_ser):

    """ Print the summary stats of the sequence length """

    seq_length_ser = str_ser.str.split(' ').str.len()             ❶

    print("\nSome summary statistics")                            ❷
    print("Median length: {}\n".format(seq_length_ser.median()))  ❷
    print(seq_length_ser.describe())                              ❷

    print(
        "\nComputing the statistics between the 1% and 99% quantiles (to 
➥ ignore outliers)"
    )
    p_01 = seq_length_ser.quantile(0.01)                          ❸
    p_99 = seq_length_ser.quantile(0.99)                          ❸

    print(
        seq_length_ser[
            (seq_length_ser >= p_01) & (seq_length_ser < p_99)
        ].describe()                                              ❹
    )
```

❶ 创建包含每个评论的序列长度的 `pd.Series`。

❷ 获取序列长度的中位数以及摘要统计信息。

❸ 获取给定标记（即 1％和 99％的百分位）的分位数。

❹ 打印定义的分位数之间的数据的摘要统计信息。

接下来，对数据调用此函数以获取统计信息：

```py
print("English corpus")
print('='*50)
print_sequence_length(train_df["EN"])

print("\nGerman corpus")
print('='*50)
print_sequence_length(train_df["DE"])
```

这产生：

```py
English corpus
==================================================
Some summary statistics
Median length: 6.0

count    40000.000000
mean         6.360650
std          2.667726
min          1.000000
25%          5.000000
50%          6.000000
75%          8.000000
max        101.000000
Name: EN, dtype: float64

Computing the statistics between the 1% and 99% quantiles (to ignore outliers)
count    39504.000000
mean         6.228002
std          2.328172
min          2.000000
25%          5.000000
50%          6.000000
75%          8.000000
max         14.000000
Name: EN, dtype: float64

German corpus
==================================================

Some summary statistics
Median length: 8.0

count    40000.000000
mean         8.397875
std          2.652027
min          3.000000
25%          7.000000
50%          8.000000
75%         10.000000
max         77.000000
Name: DE, dtype: float64

Computing the statistics between the 1% and 99% quantiles (to ignore outliers)
count    39166.000000
mean         8.299035
std          2.291474
min          5.000000
25%          7.000000
50%          8.000000
75%         10.000000
max         16.000000
Name: DE, dtype: float64
```

接下来，让我们打印出两种语言的词汇量和序列长度参数：

```py
print("EN vocabulary size: {}".format(en_vocab))
print("DE vocabulary size: {}".format(de_vocab))

# Define sequence lengths with some extra space for longer sequences
en_seq_length = 19
de_seq_length = 21

print("EN max sequence length: {}".format(en_seq_length))
print("DE max sequence length: {}".format(de_seq_length))
```

这将返回：

```py
EN vocabulary size: 359
DE vocabulary size: 336
EN max sequence length: 19
DE max sequence length: 21
```

现在我们有了定义模型所需的语言特定参数。在下一节中，我们将看看如何定义一个能够在语言之间进行翻译的模型。

## 练习 1

您已经获得了以下格式的 `pandas Series` `ser`：

```py
0       [a, b, c]
1          [d, e]
2    [f, g, h, i]
...

dtype: object
```

编写一个名为 `vocab_size(ser)` 的函数来返回词汇量。

## 11.2 编写英语-德语 seq2seq 机器翻译器

您有一个准备进入模型的干净数据集。您将使用一个序列到序列的深度学习模型作为机器翻译模型。它由两部分组成：一个编码器，用于生成英文（源）文本的隐藏表示，以及一个解码器，用于解码该表示以生成德文（目标）文本。编码器和解码器都是循环神经网络。此外，模型将接受原始文本，并使用 TensorFlow 提供的 `TextVectorization` 层将原始文本转换为令牌 ID。这些令牌 ID 将传递给一个嵌入层，该层将返回令牌 ID 的单词向量。

我们已经准备好并准备好使用的数据。现在让我们了解一下可以使用此数据的模型。序列到序列学习将任意长的序列映射到另一个任意长的序列。对于我们来说，这提出了一个独特的挑战，因为模型不仅需要能够消耗任意长度的序列，还需要能够生成任意长度的序列作为输出。例如，在机器翻译中，翻译通常比输入的单词少或多。因此，它们需要一种特殊类型的模型。这些模型被称为*编码器-解码器*或*seq2seq*（缩写为序列到序列）模型。

编码器-解码器模型实际上是两个不同的模型以某种方式相互连接起来。在概念上，编码器接受一个序列并产生一个上下文向量（或思考向量），其中嵌入了输入序列中的信息。解码器接受编码器产生的表示，并对其进行解码以生成另一个序列。由于两个部分（即编码器和解码器）分别在不同的事物上操作（即编码器消耗输入序列，而解码器生成输出序列），因此编码器-解码器模型非常适合解决序列到序列的任务。理解编码器和解码器的另一种方式是：编码器处理源语言输入（即要翻译的语言），解码器处理目标语言输入（即要翻译成的语言）。如图 11.1 所示。

![11-01](img/11-01.png)

图 11.1 编码器-解码器架构在机器翻译环境中的高级组件

特别地，编码器包含一个循环神经网络。我们将使用门控循环单元（GRU）模型。它通过输入序列并产生一个最终输出，这是在处理输入序列中的最后一个元素之后 GRU 单元的最终输出。

### 思想向量

*思想向量* 是由深度学习领域的泰斗杰弗里·亨滕（Geoffery Hinten）推广的一个术语，他从深度学习的起源就参与其中。思想向量指的是思想的向量化表示。生成准确的思想数值表示将彻底改变我们搜索文档或在网络上搜索（例如，谷歌）的方式。这类似于数值表示单词被称为 *单词向量* 的方式。在机器翻译的背景下，上下文向量可以称为思想向量，因为它在一个向量中捕捉了句子或短语的本质。

您可以在[`wiki.pathmind.com/thought-vectors`](https://wiki.pathmind.com/thought-vectors)阅读更多关于此的信息。

接下来，我们有解码器，它也由一个 GRU 模型和几个密集层组成。密集层的目的是生成最终的预测（目标词汇中的一个词）。解码器中存在的密集层的权重在时间上是共享的。这意味着，正如 GRU 层在从一个输入移动到另一个输入时更新相同的权重一样，密集层在时间步上重复使用相同的权重。这个过程在图 11.2 中有所描述。

![11-02](img/11-02.png)

图 11.2 编码器和解码器模块中的特定组件。编码器有一个 GRU 层，解码器由一个或多个密集层后跟的 GRU 层组成，其权重在时间上是共享的。

到目前为止，在解决 NLP 任务时，将字符串标记转换为数值 ID 被认为是预处理步骤。换句话说，我们会执行标记到 ID 的转换，并将 ID 输入模型。但它并不一定要这样。我们可以定义更加灵活的模型，让这种文本处理在内部完成并学会解决任务。Keras 提供了一些层，可以插入到您的模型中，以使其更加端到端。`tensorflow.keras.layers.experimental.preprocessing.TextVectorization` 层就是这样一种层。让我们来看看这个层的用法。

### 11.2.1 文本矢量化层

文本矢量化层接受一个字符串，对其进行标记化，并通过词汇表（或字典）查找将标记转换为 ID。它以字符串列表（或字符串数组）作为输入，其中每个字符串可以是单词/短语/句子（等等）。然后它从语料库中学习词汇。最后，该层可以用于将字符串列表转换为包含该列表中每个字符串的标记 ID 序列的张量。让我们看看这个层的作用。首先，导入该层：

```py
from tensorflow.keras.layers.experimental.preprocessing import 
➥ TextVectorization
```

然后，按以下方式定义该层。在这里，我们为英语定义了该层。请记住，我们的模型中需要两个 `TextVectorization` 层，一个用于英语，一个用于德语：

```py
en_vectorize_layer = TextVectorization(
    max_tokens=en_vocab,
    output_mode='int',
    output_sequence_length=None
)
```

值得停下来看看我们提供的不同参数：

+   `max_tokens`—指定词汇表中的单词数。如果词汇表中没有某个单词（即，超出词汇表的单词），则将其转换为 `[UNK]`。

+   `output_mode`—指定最终输出的类型。可以是 `"int"`、`"binary"`、`"count"` 和 `"tf-idf"` 之一。`"int"` 表示层将为每个标记输出一个标记 ID。`"binary"` 意味着输出将是一个 `[<批量大小>, <词汇大小>]` 张量，在这个例子中，如果该标记所指示的索引在这个示例中存在，则给定值为 1。`"count"` 给出了与 `"binary"` 类似的输出，但其中包含了该示例中标记出现的次数。`"tf-idf"` 给出了与 `"binary"` 类似的输出，但每个位置处的 TF-IDF 值。

+   `output_sequence_length`—指定转换为标记 ID 后的批量输入序列的长度。如果设置为 `None`，则意味着序列长度将设置为批量中最长序列的长度。较短的序列将使用特殊标记（特殊标记默认为 `""`）进行填充。

要充分利用这个层，我们必须在文本语料库上适应它，以便它可以学习词汇表。调用 `adapt()` 函数并向其传递字符串列表（或字符串数组）可以实现这一目的。换句话说，`adapt()` 产生了与 scikit-learn 模型的 `fit()` 方法相同的结果（[`mng.bz/aJmB`](http://mng.bz/aJmB)）。它接受一些数据并根据数据训练（或适应）模型。对于标记器来说，除其他外，它构建了一个词典（从单词到 ID 的映射）：

```py
en_vectorize_layer.adapt(np.array(train_df["EN"].tolist()).astype('str'))
```

安装该层后，可以获得词汇表：

```py
print(en_vectorize_layer.get_vocabulary()[:10])
```

这会打印：

```py
['', '[UNK]', 'tom', 'to', 'you', 'the', 'i', 'a', 'is', 'that']
```

换句话说，词汇表是一个标记列表，其中 ID 对应于它们在列表中的索引。你可以通过计算词汇表的大小来得到：

```py
print(len(en_vectorize_layer.get_vocabulary()))
```

这会返回：

```py
2238
```

接下来，要使用这个层将字符串转换为数字 ID，我们必须将其封装在一个模型中。为此，首先让我们定义一个 Keras 顺序模型。让我们将模型命名为 `toy_model`，因为这只用于学习文本向量化器的行为：

```py
toy_model = tf.keras.models.Sequential()
```

定义一个输入层，将其大小设置为接受单列张量（即，一个字符串列表），并将数据类型设置为 `tf.string`：

```py
toy_model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
```

然后添加我们定义的文本向量化层：

```py
toy_model.add(en_vectorize_layer)
```

你可以像使用其他 Keras 模型一样使用它，并将任意文本转换为数字 ID 序列。具体地，你可以在一些输入数据上使用 `model.predict()` 函数，该函数接受输入并根据模型中使用的层进行相应转换：

```py
input_data = [["run"], ["how are you"],["ectoplasmic residue"]]
pred = toy_model.predict(input_data)
```

最后，按以下方式打印输入和结果：

```py
print("Input data: \n{}\n".format(input_data))
print("\nToken IDs: \n{}".format(pred))
```

这会给出：

```py
Input data: 
[['run'], ['how are you'], ['ectoplasmic residue']]
Token IDs: 
[[427   0   0]
 [ 40  23   4]
 [  1   1   0]]
```

该层按照一切都进行。首先让我们看一下输出的形状。由于我们设置了 `output_sequence_length=None`，所以将所有输入示例填充到输入中最长输入的长度。在这里，`"how are you"` 是最长的，其中有三个单词。因此，所有行都用零填充，以便每个示例都有三列。通常，该层返回一个大小为 `[<batch size>, sequence_length]` 的输出。

如果单词在词汇表中找到，它将转换为某个数字（例如，`"run"` 转换为 427）。如果单词未在词汇表中找到（例如，`"ectoplasmic"`），则会用表示词汇表外单词的特殊 ID（1）替换。

### 11.2.2 为 seq2seq 模型定义 TextVectorization 层

通过对 `TextVectorization` 层有很好的理解，让我们定义一个函数，返回一个包装在 Keras `Model` 对象中的文本向量化层。此函数名为 `get_vectorizer()`，接受以下参数：

+   `corpus`—接受字符串列表（或数组）（即要构建词汇表的语料库）。

+   `n_vocab`—词汇量大小。保留最常见的 `n_vocab` 个词以构建词汇表。

+   `max_length`（可选）—结果标记序列的长度。默认为 `None`，此时序列长度将为最长文本序列的长度。

+   `return_vocabulary`（可选）—是否返回词汇表（即字符串标记列表）。

+   `name`（可选）—用于设置模型名称的字符串。

它定义了一个接受字符串批次（总共形状为 `[None, 1]`）的输入层。接下来，函数定义了一个文本向量化层。请注意，该层的词汇量为 `n_vocab + 2`。额外的 2 是为了容纳特殊标记 `""` 和 `"[UNK]"`。该层适应了传递给函数的文本语料库。最后，我们使用输入层（`inp`）和文本向量化层的输出（`vectorize_out`）定义了一个 Keras 模型。如果 `return_vocabulary` 设置为 `True`，则还会返回 `vectorize_layer` 的词汇表，如下一列表所示。

列表 11.3 为编码器-解码器模型定义文本向量化器
