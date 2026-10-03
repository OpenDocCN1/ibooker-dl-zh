
Keras 还包括几个用于数据增强的层，如`RandomCrop`、`RandomFlip`、`RandomTranslation`、`RandomRotation`、`RandomZoom`、`RandomHeight`、`RandomWidth`和`RandomContrast`。这些层仅在训练期间激活，并随机对输入图像应用一些转换（它们的名称是不言自明的）。数据增强将人为增加训练集的大小，通常会导致性能提升，只要转换后的图像看起来像真实的（非增强的）图像。我们将在下一章更详细地介绍图像处理。

## 注意

在幕后，Keras 预处理层基于 TensorFlow 的低级 API。例如，`Normalization`层使用`tf.nn.moments()`来计算均值和方差，`Discretization`层使用`tf.raw_ops.Bucketize()`，`CategoricalEncoding`使用`tf.math.bincount()`，`IntegerLookup`和`StringLookup`使用`tf.lookup`包，`Hashing`和`TextVectorization`使用`tf.strings`包中的几个操作，`Embedding`使用`tf.nn.embedding_lookup()`，图像预处理层使用`tf.image`包中的操作。如果 Keras 预处理 API 不满足您的需求，您可能偶尔需要直接使用 TensorFlow 的低级 API。

现在让我们看看在 TensorFlow 中另一种轻松高效地加载数据的方法。

# TensorFlow 数据集项目

[TensorFlow 数据集（TFDS）](https://tensorflow.org/datasets)项目使加载常见数据集变得非常容易，从小型数据集如 MNIST 或 Fashion MNIST 到像 ImageNet 这样的大型数据集（您将需要相当大的磁盘空间！）。列表包括图像数据集、文本数据集（包括翻译数据集）、音频和视频数据集、时间序列等等。您可以访问[*https://homl.info/tfds*](https://homl.info/tfds)查看完整列表，以及每个数据集的描述。您还可以查看[了解您的数据](https://knowyourdata.withgoogle.com)，这是一个用于探索和理解 TFDS 提供的许多数据集的工具。

TFDS 并未与 TensorFlow 捆绑在一起，但如果您在 Colab 上运行或者按照[*https://homl.info/install*](https://homl.info/install)的安装说明进行安装，那么它已经安装好了。然后您可以导入`tensorflow_datasets`，通常为`tfds`，然后调用`tfds.load()`函数，它将下载您想要的数据（除非之前已经下载过），并将数据作为数据集字典返回（通常一个用于训练，一个用于测试，但这取决于您选择的数据集）。例如，让我们下载 MNIST：

```py
import tensorflow_datasets as tfds

datasets = tfds.load(name="mnist")
mnist_train, mnist_test = datasets["train"], datasets["test"]
```

然后您可以应用任何您想要的转换（通常是洗牌、批处理和预取），然后准备训练您的模型。这里是一个简单的示例：

```py
for batch in mnist_train.shuffle(10_000, seed=42).batch(32).prefetch(1):
    images = batch["image"]
    labels = batch["label"]
    # [...] do something with the images and labels
```

## 提示

`load()`函数可以对其下载的文件进行洗牌：只需设置`shuffle_files=True`。但是这可能不够，最好对训练数据进行更多的洗牌。

请注意，数据集中的每个项目都是一个包含特征和标签的字典。但是 Keras 期望每个项目是一个包含两个元素的元组（再次，特征和标签）。您可以使用`map()`方法转换数据集，就像这样：

```py
mnist_train = mnist_train.shuffle(buffer_size=10_000, seed=42).batch(32)
mnist_train = mnist_train.map(lambda items: (items["image"], items["label"]))
mnist_train = mnist_train.prefetch(1)
```

但是通过设置`as_supervised=True`，让`load()`函数为您执行此操作会更简单（显然，这仅适用于带标签的数据集）。

最后，TFDS 提供了一种方便的方法来使用`split`参数拆分数据。例如，如果您想要使用训练集的前 90%进行训练，剩余的 10%进行验证，整个测试集进行测试，那么您可以设置`split=["train[:90%]", "train[90%:]", "test"]`。`load()`函数将返回所有三个集合。这里是一个完整的示例，使用 TFDS 加载和拆分 MNIST 数据集，然后使用这些集合来训练和评估一个简单的 Keras 模型：

```py
train_set, valid_set, test_set = tfds.load(
    name="mnist",
    split=["train[:90%]", "train[90%:]", "test"],
    as_supervised=True
)
train_set = train_set.shuffle(buffer_size=10_000, seed=42).batch(32).prefetch(1)
valid_set = valid_set.batch(32).cache()
test_set = test_set.batch(32).cache()
tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
history = model.fit(train_set, validation_data=valid_set, epochs=5)
test_loss, test_accuracy = model.evaluate(test_set)
```

恭喜，您已经到达了这个相当技术性的章节的结尾！您可能会觉得它与神经网络的抽象美有些远，但事实是深度学习通常涉及大量数据，知道如何高效加载、解析和预处理数据是一项至关重要的技能。在下一章中，我们将看一下卷积神经网络，这是图像处理和许多其他应用中最成功的神经网络架构之一。

# 练习

1.  为什么要使用 `tf.data` API？

1.  将大型数据集拆分为多个文件的好处是什么？

1.  在训练过程中，如何判断您的输入管道是瓶颈？您可以做些什么来解决它？

1.  您可以将任何二进制数据保存到 TFRecord 文件中吗，还是只能序列化协议缓冲区？

1.  为什么要费心将所有数据转换为`Example`协议缓冲区格式？为什么不使用自己的协议缓冲区定义？

1.  在使用 TFRecords 时，何时应该激活压缩？为什么不系统地这样做？

1.  数据可以在编写数据文件时直接进行预处理，或者在 `tf.data` 管道中进行，或者在模型内的预处理层中进行。您能列出每个选项的一些优缺点吗？

1.  列举一些常见的编码分类整数特征的方法。文本呢？

1.  加载时尚 MNIST 数据集（在第十章中介绍）；将其分为训练集、验证集和测试集；对训练集进行洗牌；并将每个数据集保存到多个 TFRecord 文件中。每个记录应该是一个序列化的`Example`协议缓冲区，具有两个特征：序列化图像（使用`tf.io.serialize_tensor()`来序列化每个图像），和标签。然后使用 `tf.data` 为每个集创建一个高效的数据集。最后，使用 Keras 模型来训练这些数据集，包括一个预处理层来标准化每个输入特征。尝试使输入管道尽可能高效，使用 TensorBoard 来可视化分析数据。

1.  在这个练习中，您将下载一个数据集，将其拆分，创建一个`tf.data.Dataset`来高效加载和预处理数据，然后构建和训练一个包含`Embedding`层的二元分类模型：

    1.  下载[大型电影评论数据集](https://homl.info/imdb)，其中包含来自[互联网电影数据库（IMDb）](https://imdb.com)的 50,000 条电影评论。数据组织在两个目录中，*train*和*test*，每个目录包含一个*pos*子目录，其中包含 12,500 条正面评论，以及一个*neg*子目录，其中包含 12,500 条负面评论。每个评论存储在单独的文本文件中。还有其他文件和文件夹（包括预处理的词袋版本），但在这个练习中我们将忽略它们。

    1.  将测试集分为验证集（15,000）和测试集（10,000）。

    1.  使用 `tf.data` 为每个集创建一个高效的数据集。

    1.  创建一个二元分类模型，使用`TextVectorization`层来预处理每个评论。

    1.  添加一个`Embedding`层，并计算每个评论的平均嵌入，乘以单词数量的平方根（参见第十六章）。然后将这个重新缩放的平均嵌入传递给您模型的其余部分。

    1.  训练模型并查看您获得的准确性。尝试优化您的管道，使训练尽可能快。

    1.  使用 TFDS 更轻松地加载相同的数据集：`tfds.load("imdb_reviews")`。

这些练习的解决方案可以在本章笔记本的末尾找到，网址为[*https://homl.info/colab3*](https://homl.info/colab3)。

¹ 想象一副排好序的扑克牌在您的左边：假设您只拿出前三张牌并洗牌，然后随机选取一张放在右边，将另外两张留在手中。再从左边拿一张牌，在手中的三张牌中洗牌，随机选取一张放在右边。当您像这样处理完所有的牌后，您的右边将有一副扑克牌：您认为它会被完美洗牌吗？

² 一般来说，只预取一个批次就可以了，但在某些情况下，您可能需要预取更多。或者，您可以通过将`tf.data.AUTOTUNE`传递给`prefetch()`，让 TensorFlow 自动决定。

³ 但是请查看实验性的`tf.data.experimental.prefetch_to_device()`函数，它可以直接将数据预取到 GPU。任何带有`experimental`的 TensorFlow 函数或类的名称可能会在未来版本中发生更改而没有警告。如果实验性函数失败，请尝试删除`experimental`一词：它可能已经移至核心 API。如果没有，请查看笔记本，我会确保其中包含最新的代码。

⁴ 由于 protobuf 对象旨在被序列化和传输，它们被称为*消息*。

⁵ 本章包含了您使用 TFRecords 所需了解的最基本知识。要了解更多关于 protobufs 的信息，请访问[*https://homl.info/protobuf*](https://homl.info/protobuf)。

⁶ Tomáš Mikolov 等人，“单词和短语的分布式表示及其组合性”，*第 26 届国际神经信息处理系统会议论文集* 2（2013）：3111–3119。

⁷ Malvina Nissim 等人，“公平比耸人听闻更好：男人对医生，女人对医生”，arXiv 预印本 arXiv:1905.09866（2019）。

⁸ TensorFlow Hub 没有与 TensorFlow 捆绑在一起，但如果您在 Colab 上运行或者按照[*https://homl.info/install*](https://homl.info/install)的安装说明进行安装，那么它已经安装好了。

⁹ 要精确，句子嵌入等于句子中单词嵌入的平均值乘以句子中单词数的平方根。这是为了弥补随着*n*增长，*n*个随机向量的平均值会变短的事实。

¹⁰ 对于大图像，您可以使用`tf.io.encode_jpeg()`。这将节省大量空间，但会损失一些图像质量。
