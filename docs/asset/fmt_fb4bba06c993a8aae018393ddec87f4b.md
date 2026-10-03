
根据统计结果，我们将使用 `5` 和 `15` 作为桶边界。换句话说，评论按照以下逻辑进行分类：

+ 长度在 `[0, 5)` 的评论为短评论。
+ 长度在 `[5, 15)` 的评论为中等评论。
+ 长度在 `[15, inf)` 的评论为长评论。

最后两个小节总结了我们分析以找到词汇表大小和序列长度的过程。这里呈现的输出提供了所有信息，以便以有原则的态度选择我们的超参数。

### 9.2.4 使用 Keras 将文本转换为单词，然后转换为数字

我们有一段干净的、经过处理的文本语料库，以及我们稍后将使用的词汇表大小和序列长度参数。我们的下一个任务是将文本转换为数字。将文本转换为数字有两个标准步骤：

1. 将文本拆分为标记（例如，字符/单词/句子）。
2. 创建一个将每个唯一标记映射到唯一 ID 的字典。

例如，如果您有以下句子：

```py
the cat sat on the mat
```

我们将首先将其标记化为单词，得到：

```py
[the, cat, sat, on, the, mat]
```

并且有字典：

```py
{the: 1, cat: 2, sat: 3, on: 4, mat: 5}
```

然后，您可以创建以下序列来表示原始文本：

```py
[1,2,3,4,1,5]
```

`Keras Tokenizer` 对象支持这种功能。它接受一段文本语料库，使用一些用户定义的参数进行标记化，自动构建词典，并将其保存为状态。这样，您可以使用 `Tokenizer` 将任意文本转换为数字，次数不限。让我们看看如何使用 `Keras Tokenizer` 完成这个过程：

```py
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(
    num_words=n_vocab, 
    oov_token='unk', 
    lower=False, 
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    split=' ', 
    char_level=False
)
```

您可以看到 `Tokenizer` 传递了几个参数。让我们稍微详细地看看这些参数：

+ `num_words`——这定义了词汇表大小，以限制字典的大小。如果 `num_words` 设置为 `1,000`，则会考虑语料库中最常见的 `1,000` 个单词，并为它们分配唯一的 ID。
+ `oov_token`——这个参数处理落在定义的词汇表大小之外的词。出现在语料库中但未包含在最常见的 `num_words` 个词中的单词将被替换为此标记。
+ `lower`——这确定是否对文本进行大小写转换。由于我们已经做过了，我们将其设置为 `False`。
+ `filter`——这定义了在标记化文本之前要删除的任何字符。
+ `split`——这是用于标记化文本的分隔符字符。我们希望单词是标记，因此我们将使用空格，因为单词通常用空格分隔。
+ `char_level`——这指示是否执行字符级标记化（即，每个字符都是一个标记）。

在我们继续之前，让我们回顾一下我们的数据在当前状态下是什么样子。请记住，我们有：

+ 清理后的数据
+ 预处理数据
+ 将每个评论拆分为单独的单词

在这个过程结束时，我们的数据如下所示。首先，我们有输入，它是一个 `pd.Series` 对象，包含一系列干净的单词列表。文本前面的数字是该记录在 `pd.Series` 对象中的索引：

```py
122143    [work, perfectly, wii, gamecube, issue, compat...
444818    [loved, game, collectible, come, well, make, m...
79331     ['s, okay, game, honest, bad, type, game, --, ...
97250                        [excellent, product, describe]
324411        [level, detail, great, feel, love, car, game]
...
34481     [not, actually, believe, write, review, produc...
258474    [good, game, us, like, movie, franchise, hard,...
466203    [fun, first, person, shooter, nice, combinatio...
414288                       [love, amiibo, classic, color]
162670    [fan, halo, series, start, enjoy, game, overal...
Name: reviewText, dtype: object
```

接下来，我们有标签，其中每个标签都是一个二进制标签，用于指示评论是正面评论还是负面评论：

```py
122143    1
444818    1
79331     0
97250     1
324411    1
...
34481     1
258474    1
466203    1
414288    1
162670    0
Name: label, dtype: int64
```

从某种意义上说，标记文本的第一步已经完成了。如果已经完成了这一步，那么 `Keras Tokenizer` 足够智能，会跳过这一步。要构建 `Tokenizer` 的字典，可以调用 `tf.keras.preprocessing.text.Tokenizer.fit_on_texts()` 函数，如下所示：

```py
tokenizer.fit_on_texts(tr_x.tolist())
```

`fit_on_texts()` 函数接受一个字符串列表，其中每个字符串是你正在处理的单个实体（例如，一个句子、一个评论、一个段落等），或者是一个标记列表的列表，其中标记可以是一个单词、一个字符，甚至是一个句子。当你将 `Tokenizer` 拟合到某些文本时，你可以检查一些内部状态变量。你可以使用以下方式检查单词到 ID 的映射：

```py
tokenizer.word_index["game"]
```

这将返回：

```py
2
```

你还可以使用以下方式检查 ID 到单词的映射（即将单词映射到 ID 的反向操作）：

```py
tokenizer.index_word[4]
```

这将返回：

```py
"play"
```

要将文本语料库转换为索引序列，你可以使用 `texts_to_sequences()` 函数。它接受一个标记列表的列表，并返回一个 ID 列表的列表：

```py
tr_x = tokenizer.texts_to_sequences(tr_x.tolist())
v_x = tokenizer.texts_to_sequences(v_x.tolist())
ts_x = tokenizer.texts_to_sequences(ts_x.tolist())
```

让我们看一下 `text_to_sequences()` 函数转换的一些样本的结果：

```py
Text: ['work', 'perfectly', 'wii', 'gamecube', 'issue', 'compatibility', 
➥ 'loss', 'memory']
Sequence: [14, 295, 83, 572, 121, 1974, 2223, 345]
Text: ['loved', 'game', 'collectible', 'come', 'well', 'make', 'mask', 
➥ 'big', 'almost', 'fit', 'face', 'impressive']
Sequence: [1592, 2, 2031, 32, 23, 16, 2345, 153, 200, 155, 599, 1133]

Text: ["'s", 'okay', 'game', 'honest', 'bad', 'type', 'game', '--', "'s", 
➥ 'difficult', 'always', 'die', 'depresses', 'maybe', 'skill', 'would', 
➥ 'enjoy', 'game']
Sequence: [5, 574, 2, 1264, 105, 197, 2, 112, 5, 274, 150, 354, 1, 290, 
➥ 400, 19, 67, 2]

Text: ['excellent', 'product', 'describe']
Sequence: [109, 55, 501]

Text: ['level', 'detail', 'great', 'feel', 'love', 'car', 'game']
Sequence: [60, 419, 8, 42, 13, 265, 2]
```

太棒了！我们可以看到文本完美地转换为了 ID 序列。我们现在将继续使用 `Keras Tokenizer` 返回的数据定义 `TensorFlow` 管道。

**练习 2**

给定字符串 `s`，`"a_b_B_c_d_a_D_b_d_d"`，你能否定义一个 `tokenizer`，`tok`，将文本转换为小写形式，按下划线字符 `"_"` 拆分，具有 `3` 个词汇大小，并将 `Tokenizer` 拟合到 `s` 上。如果 `Tokenizer` 忽略从 `1` 开始的词汇索引词，那么如果你调用 `tok.texts_to_sequences([s])`，输出会是什么？

## 9.3 使用 TensorFlow 定义端到端的 NLP 管道

你已经定义了一个干净的数据集，它是模型期望的数字格式。在这里，我们将定义一个 `TensorFlow` 数据集管道，以从我们定义的数据中生成数据批次。在数据管道中，你将生成一批数据，其中批次由元组 `(x, y)` 组成。`x` 表示一批文本序列，其中每个文本序列都是一个任意长的标记 ID 序列。`y` 是与批次中的文本序列对应的标签批次。在生成一批示例时，首先根据序列长度将文本序列分配到桶中。每个桶都有一个预定义的允许序列长度间隔。批次中的示例仅由同一桶中的示例组成。

我们现在处于一个很好的位置。我们已经对数据进行了相当多的预处理，并将文本转换为了机器可读的数字。在下一步中，我们将构建一个 `tf.data` 管道，将 `Tokenizer` 的输出转换为适合模型的输出。

作为第一步，我们将目标标签（具有值 `0/1`）连接到输入。这样，我们可以以任何我们想要的方式对数据进行洗牌，并仍然保持输入和目标标签之间的关系：

```py
data_seq = [[b]+a for a,b in zip(text_seq, labels) ]
```

接下来，我们将创建一种特殊类型的 `tf.Tensor` 对象，称为 *ragged tensor*（即 `tf.RaggedTensor`）。在标准张量中，您具有固定的维度。例如，如果您定义了一个 `3×4` 大小的张量，则每行都需要有四列（即四个值）。Ragged tensors 是一种支持可变大小的张量的特殊类型。例如，可以将以下数据作为 ragged tensor：

```py
[
  [1,2],
  [3,2,5,9,10],
  [3,2,3]
]
```

此张量有三行，其中第一行有两个值，第二行有五个值，最后一行有三个值。换句话说，它具有可变的第二维。由于每篇评论的字数不同，因此导致与每篇评论对应的变大小的 ID 序列，这是我们问题的完美数据结构：

```py
max_length = 50    
tf_data = tf.ragged.constant(data_seq)[:,:max_length]
```

**对 `tf.RaggedTensor` 的初步了解**

`tf.RaggedTensor` 对象是一种特殊类型的张量，可以具有可变大小的维度。您可以在 [`mng.bz/5QZ8`](http://mng.bz/5QZ8) 了解更多关于 ragged tensors 的信息。有许多方法来定义一个 ragged tensor。

我们可以通过传递包含值的嵌套列表来定义 ragged tensor，以 `tf.ragged.constant()` 函数：

```py
a = tf.ragged.constant([[1, 2, 3], [1,2], [1]])
```

您还可以定义一系列值并定义分割行的位置：

```py
b = tf.RaggedTensor.from_row_splits([1,2,3,4,5,6,7], row_splits=[0, 3, 3, 6, 7])
```

在这里，`row_splits` 参数中的每个值定义了结果张量中随后的行结束在哪里。例如，第一行将包含从索引 `0` 到 `3`（即 `0, 1, 2`）的元素。这将输出：

```py
<tf.RaggedTensor [[1, 2, 3], [], [4, 5, 6], [7]]>
```

您可以使用 `b.shape` 获取张量的形状，这将返回：

```py
[4, None]
```

您甚至可以拥有多维的 ragged tensors，其中有多个可变大小的维度，如下所示：

```py
c = tf.RaggedTensor.from_nested_row_splits(
    flat_values=[1,2,3,4,5,6,7,8,9], 
    nested_row_splits=([0,2,3],[0,4,6,9]))
```

在这里，`nested_row_splits` 是 1D 张量的列表，其中第 `i` 个张量表示第 `i` 个维度的行拆分。`c` 如下所示：

```py
<tf.RaggedTensor [[[1, 2, 3, 4], [5, 6]], [[7, 8, 9]]]>
```

您可以在 ragged tensors 上执行切片和索引操作，类似于在普通张量上的操作：

```py
print(c[:1, :, :])
```

这将返回：

```py
<tf.RaggedTensor [[[1, 2, 3, 4], [5, 6]]]>
```

在这里：

```py
print(c[:,:1,:])
```

这将返回：

```py
<tf.RaggedTensor [[[1, 2, 3, 4]], [[7, 8, 9]]]>
```

最后，随着：

```py
print(c[:, :, :2])
```

您将获得：

```py
<tf.RaggedTensor [[[1, 2], [5, 6]], [[7, 8]]]>
```

我们将限制评论的最大长度为 `max_length`。这是在假设 `max_length` 个字足以捕获给定评论中的情感的情况下完成的。这样，我们可以避免由于数据中存在一两个极长的评论而导致最终数据过长。最大长度越高，就能更好地捕获评论中的信息。但是，更高的 `max_length` 值会带来极大的计算开销：

```py
text_ds = tf.data.Dataset.from_tensor_slices(tf_data)
```

我们将使用 `tf.data.Dataset.from_tensor_slices()` 函数创建一个数据集。该函数在我们刚创建的 ragged tensor 上将逐一提取一行（即一篇评论）。重要的是要记住每行的大小都不同。我们将过滤掉任何空评论。您可以使用 `tf.data.Dataset.filter()` 函数来做到这一点：

```py
text_ds = text_ds.filter(lambda x: tf.size(x)>1)
```

基本上，我们在这里说的是任何大小小于或等于 `1` 的评论将被丢弃。记住，每条记录至少会有一个元素（即标签）。这是一个重要的步骤，因为空评论会在模型后续处理中引起问题。

接下来，我们将解决一个极其重要的步骤，以及我们令人印象深刻的数据管道的重点。在序列处理过程中，你可能听过 *分桶*（或 *分箱*）这个术语。分桶是指在批处理数据时，使用相似大小的输入。换句话说，一批数据包括长度相似的评论，不会在同一批中有长度差距悬殊的评论。下面的侧栏更详细地解释了分桶的过程。

**分桶：相似长度的序列聚集在一起！**

让我们来看一个例子。假设你有一个评论列表，`[r1(5), r2(11), r3(6), r4(4), r5(15), r6(18), r7(25), r8(29), r9(30)]`，其中代码 `rx` 代表评论 ID，括号内的数字代表评论中的单词数。如果你选择批量大小为 `3`，那么以下列方式分组数据是有意义的：

```py
[r1, r3, r4]
[r2, r5, r6]
[r7, r8, r9]
```

你可以看到长度相近的评论被分组在一起。这实际上是通过一个称为分桶的过程来实现的。首先，我们创建几个预定义边界的桶。例如，在我们的示例中，可能有三个间隔如下的桶：

```py
[[0,11), [11, 21), [21, inf)) 
```

然后，根据评论的长度，每个评论被分配到一个桶中。最后，在获取数据批次时，从一个随机选择的桶中随机采样批次数据。

在确定桶之后，我们必须将数据分组，使得最终得到固定序列长度。这是通过在序列的末尾填充零直到我们在该批次中拥有长度相等的所有序列来实现的。假设评论 `r1`、`r3` 和 `r4` 有以下单词 ID 序列：

```py
[10, 12, 48, 21,  5]
[ 1, 93, 28,  8, 20, 10]
[32, 20,  1,  2]
```

为了将这些序列分组，我们将在短序列的末尾填充零，结果是：

```py
[10, 12, 48, 21,  5,  0]
[ 1, 93, 28,  8, 20, 10]
[32, 20,  1,  2,  0,  0]
```
