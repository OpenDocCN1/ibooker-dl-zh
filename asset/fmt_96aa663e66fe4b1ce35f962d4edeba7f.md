

# 十六、使用 RNN 和注意力机制进行自然语言处理

> 译者：[@SeanCheney](https://www.jianshu.com/u/130f76596b02)

当阿兰·图灵在 1950 年设计[图灵机](https://links.jianshu.com/go?to=http%3A%2F%2Fcogprints.org%2F499%2F1%2Fturing.html)时，他的目标是用人的智商来衡量机器。他本可以用其它方法来测试，比如看图识猫、下棋、作曲或逃离迷宫，但图灵选择了一个语言任务。更具体的，他设计了一个聊天机器人，试图迷惑对话者将其当做真人。这个测试有明显的缺陷：一套硬编码的规则可以愚弄粗心人（比如，机器可以针对一些关键词，做出预先定义的模糊响应；机器人可以假装开玩笑或喝醉；或者可以通过反问侥幸过关），忽略了人类的多方面的智力（比如非语言交流，比如面部表情，或是学习动手任务）。但图灵测试强调了一个事实，语言能力是智人最重要的认知能力。我们能创建一台可以读写自然语言的机器吗？

自然语言处理的常用方法是循环神经网络。所以接下来会从字符 RNN 开始（预测句子中出现的下一个字符），继续介绍 RNN，这可以让我们生成一些原生文本，在过程中，我们会学习如何在长序列上创建 TensorFlow Dataset。先使用的是无状态 RNN（每次迭代中学习文本中的随机部分），然后创建一个有状态 RNN（保留训练迭代之间的隐藏态，可以从断点继续，用这种方法学习长规律）。然后，我们会搭建一个 RNN，来做情感分析（例如，读取影评，提取评价者对电影的感情），这次是将句子当做词的序列来处理。然后会介绍用 RNN 如何搭建编码器-解码器架构，来做神经网络机器翻译（NMT）。我们会使用 TensorFlow Addons 项目中的 seq2seq API。

本章的第二部分，会介绍注意力机制。正如其名字，这是一种可以选择输入指定部分，模型在每个时间步都得聚焦的神经网络组件。首先，会介绍如何使用注意力机制提升基于 RNN 的编码器-解码器架构的性能，然后会完全摒弃 RNN，介绍只使用注意力的架构，被称为 Transformer（转换器）。最后，会介绍 2018、2019 两年 NLP 领域的进展，包括强大的语言模型，比如 GPT-2 和 Bert，两者都是基于 Transformer 的。

先从一个简单有趣的模型开始，它能写出莎士比亚风格的文字。

## 使用 Character RNN 生成莎士比亚风格的文本

在 2015 年一篇著名的、名为《The Unreasonable Effectiveness of Recurrent Neural Networks》博客中，Andrej Karpathy 展示了如何训练 RNN，来预测句子中的下一个字符。这个 Char-RNN 可以用来生成小说，每次一个字符。下面是一段简短的、由 Char-RNN 模型（在莎士比亚全部著作上训练而成）生成的文本：

```py
PANDARUS:
Alas, I think he shall be come approached and the day
When little srain would be attain'd into being never fed,
And who is but a chain and subjects of his death,
I should not sleep. 
```

虽然文笔一般，但只是通过学习来预测一句话中的下一个字符，模型在单词、语法、断句等等方面做的很好。接下来一步一步搭建 Char-RNN，从创建数据集开始。

### 创建训练数据集

首先，使用 Keras 的`get_file()`函数，从 Andrej Karpathy 的 [Char-RNN 项目](https://links.jianshu.com/go?to=https%3A%2F%2Fgithub.com%2Fkarpathy%2Fchar-rnn)，下载所有莎士比亚的作品：

```py
shakespeare_url = "https://homl.info/shakespeare" # shortcut URL
filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)
with open(filepath) as f:
    shakespeare_text = f.read() 
```

然后，将每个字符编码为一个整数。方法之一是创建一个自定义预处理层，就像之前在第 13 章做的那样。但在这里，使用 Keras 的`Tokenizer`会更加简单。首先，将一个将分词器拟合到文本：分词器能从文本中发现所有的字符，并将所有字符映射到不同的字符 ID，映射从 1 开始（注意不是从 0 开始，0 是用来做遮挡的，后面会看到）：

```py
tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts([shakespeare_text]) 
```

设置`char_level=True`，以得到字符级别的编码，而不是默认的单词级别的编码。这个分词器默认将所有文本转换成了小写（如果不想这样，可以设置`lower=False`）。现在分词器可以将一整句（或句子列表）编码为字符 ID 列表，这可以告诉我们文本中有多少个独立的字符，以及总字符数：

```py
>>> tokenizer.texts_to_sequences(["First"])
[[20, 6, 9, 8, 3]]
>>> tokenizer.sequences_to_texts([[20, 6, 9, 8, 3]])
['f i r s t']
>>> max_id = len(tokenizer.word_index) # number of distinct characters
>>> dataset_size = tokenizer.document_count # total number of characters 
```

现在对完整文本做编码，将每个字符都用 ID 来表示（减 1 使 ID 从 0 到 38，而不是 1 到 39）：

```py
[encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1 
```

继续之前，需要将数据集分成训练集、验证集和测试集。不能大论字符，该怎么处理这种序列式的数据集呢？

### 如何切分序列数据集

避免训练集、验证集、测试集发生重合非常重要。例如，可以取 90% 的文本作为训练集，5% 作为验证集，5% 作为测试集。在这三个数据之间留出空隙，以避免段落重叠也是非常好的主意。

当处理时间序列时，通常按照时间切分：例如，可以将从 2000 到 2012 的数据作为训练集，2013 年到 2015 年作为验证集，2016 年到 2018 年作为测试集。但是，在另一些任务中，可以按照其它维度来切分，可以得到更长的时间周期进行训练。例如，10000 家公司从 2000 年到 2018 年的金融健康数据，可以按照不同公司来切分。但是，很可能其中一些公司是高度关联的（比如，经济领域的公司涨落相同），如果训练集和测试集中有关联的公司，则测试集的意义就不大，泛化误差会存在偏移。

因此，在时间维度上切分更加安全 —— 但这实际是默认 RNN 可以（在训练集）从过去学到的规律也适用于将来。换句话说，我们假设时间序列是静态的（至少是在一个较宽的区间内）。对于时间序列，这个假设是合理的（比如，化学反应就是这样，化学定理不会每天发生改变），但其它的就不是（例如，金融市场就不是静态的，一旦交易员发现规律并从中牟利，规律就会改变）。要保证时间序列确实是静态的，可以在验证集上画出模型随时间的误差：如果模型在验证集的前端表现优于后段，则时间序列可能就不够静态，最好是在一个更短的时间区间内训练。

总而言之，将时间序列切分成训练集、验证集和测试集不是简单的工作，怎么做要取决于具体的任务。

回到莎士比亚！这里将前 90% 的文本作为训练集（剩下的作为验证集和测试集），创建一个`tf.data.Dataset`，可以从这个集和一个个返回每个字符：

```py
train_size = dataset_size * 90 // 100
dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size]) 
```

### 将序列数据集切分成多个窗口

现在训练集包含一个单独的长序列，超过 100 万的任务，所以不能直接在这个训练集上训练神经网络：现在的 RNN 等同于一个有 100 万层的深度网络，只有一个超长的单实例来训练。所以，得使用数据集的`window()`方法，将这个长序列转化为许多小窗口文本。每个实例都是完整文本的相对短的子字符串，RNN 只在这些子字符串上展开。这被称为截断沿时间反向传播。调用`window()`方法创建一个短文本窗口的数据集：

```py
n_steps = 100
window_length = n_steps + 1 # target = input 向前移动 1 个字符
dataset = dataset.window(window_length, shift=1, drop_remainder=True) 
```

> 提示：可以调节`n_steps`：用短输入序列训练 RNN 更为简单，但肯定的是 RNN 学不到任何长度超过`n_steps`的规律，所以`n_steps`不要太短。

默认情况下，`window()`方法创建的窗口是不重叠的，但为了获得可能的最大训练集，我们设定`shift=1`，好让第一个窗口包含字符 0 到 100，第二个窗口包含字符 1 到 101，等等。为了确保所有窗口是准确的 101 个字符长度（为了不做填充而创建批次），设置`drop_remainder=True`（否则，最后的 100 个窗口会包含 100 个字符、99 个字符，一直到 1 个字符）。

`window()`方法创建了一个包含窗口的数据集，每个窗口也是数据集。这是一个嵌套的数据集，类似于列表的列表。当调用数据集方法处理（比如、打散或做批次）每个窗口时，这样会很方便。但是，不能直接使用嵌套数据集来训练，因为模型要的输入是张量，不是数据集。因此，必须调用`flat_map()`方法：它能将嵌套数据集转换成打平的数据集。例如，假设`{1, 2, 3}`表示包含张量 1、2、3 的序列。如果将嵌套数据集`{{1, 2}, {3, 4, 5, 6}}`打平，就会得到`{1, 2, 3, 4, 5, 6}`。另外，`flat_map()`方法可以接收函数作为参数，可以处理嵌套数据集的每个数据集。例如，如果将函数 `lambda ds: ds.batch(2)` 传递给 `flat_map()` ，它能将`{{1, 2}, {3, 4, 5, 6}}`转变为`{[1, 2], [3, 4], [5, 6]}`：这是一个张量大小为 2 的数据集。

有了这些知识，就可以打平数据集了：

```py
dataset = dataset.flat_map(lambda window: window.batch(window_length)) 
```

我们在每个窗口上调用了`batch(window_length)`：因为所有窗口都是这个长度，对于每个窗口，都能得到一个独立的张量。现在的数据集包含连续的窗口，每个有 101 个字符。因为梯度下降在训练集中的实例独立同分布时的效果最好，需要打散这些窗口。然后我们可以对窗口做批次，分割输入（前 100 个字符）和目标（最后一个字符）：

```py
batch_size = 32
dataset = dataset.shuffle(10000).batch(batch_size)
dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:])) 
```

图 16-1 总结了数据集准备步骤（窗口长度是 11，不是 101，批次大小是 3，不是 32）。

![](img/847ef99a515939c6266e9a184ac7e061.png)

图 16-1 准备打散窗口的数据集

第 13 章讨论过，类型输入特征通常都要编码，一般是独热编码或嵌入。这里，使用独热编码，因为独立字符不多（只有 39）：

```py
dataset = dataset.map(
    lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch)) 
```

最后，加上预提取：

```py
dataset = dataset.prefetch(1) 
```

就是这样！准备数据集是最麻烦的部分。下面开始搭建模型。

### 搭建并训练 Char-RNN 模型

根据前面的 100 个字符预测下一个字符，可以使用一个 RNN，含有两个 GRU 层，每个 128 个单元，每个单元对输入（`dropout`）和隐藏态（`recurrent_dropout`）的丢弃率是 20%。如果需要的话，后面可以微调这些超参数。输出层是一个时间分布的紧密层，有 39 个单元（`max_id`），因为文本中有 39 个不同的字符，需要输出每个可能字符（在每个时间步）的概率。输出概率之后应为 1，所以使用 softmax 激活很熟。然后可以使用`"sparse_categorical_crossentropy"`损失和 Adam 优化器，编译模型。最后，就可以训练模型几个周期了（训练过程可能要几个小时，取决于硬件）：

```py
model = keras.models.Sequential([
    keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id],
                     dropout=0.2, recurrent_dropout=0.2),
    keras.layers.GRU(128, return_sequences=True,
                     dropout=0.2, recurrent_dropout=0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(max_id,
                                                    activation="softmax"))
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam")
history = model.fit(dataset, epochs=20) 
```

### 使用 Char-RNN 模型

现在就有了可以预测莎士比亚要写的下一个人物的模型了。输入数据之前，先要像之前那样做预处理，因此写个小函数来做预处理：

```py
def preprocess(texts):
    X = np.array(tokenizer.texts_to_sequences(texts)) - 1
    return tf.one_hot(X, max_id) 
```

现在，用这个模型预测文本中的下一个字母：

```py
>>> X_new = preprocess(["How are yo"])
>>> Y_pred = model.predict_classes(X_new)
>>> tokenizer.sequences_to_texts(Y_pred + 1)[0][-1] # 1st sentence, last char
'u' 
```

预测成功！接下来用这个模型生成文本。

### 生成假莎士比亚文本

要使用 Char-RNN 生成新文本，我们可以给模型输入一些文本，让模型预测出下一个字母，将字母添加到文本的尾部，再将延长后的文本输入给模型，预测下一个字母，以此类推。但在实际中，这会导致相同的单词不断重复。相反的，可以使用`tf.random.categorical()`函数，随机挑选下一个字符，概率等同于估计概率。这样就能生成一些多样且有趣的文本。根据类的对数概率（logits），`categorical()`函数随机从类索引采样。为了对生成文本的多样性更可控，我们可以用一个称为“温度“的可调节的数来除以对数概率：温度接近 0，会利于高概率字符，而高温度会是所有字符概率相近。下面的`next_char()`函数使用这个方法，来挑选添加进文本中的字符：

```py
def next_char(text, temperature=1):
    X_new = preprocess([text])
    y_proba = model.predict(X_new)[0, -1:, :]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
    return tokenizer.sequences_to_texts(char_id.numpy())[0] 
```

然后，可以写一个小函数，重复调用`next_char()`：

```py
def complete_text(text, n_chars=50, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text 
```

现在就可以生成一些文本了！先尝试下不同的温度数：

