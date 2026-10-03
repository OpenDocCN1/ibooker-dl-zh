

# 第十四章：TensorBoard：TensorFlow 的大兄弟

本章内容包括

+ 在 TensorBoard 上运行和可视化图像数据
+ 实时监测模型性能和行为
+ 使用 TensorBoard 进行性能分析模型建模
+ 使用 `tf.summary` 在自定义模型训练过程中记录自定义指标
+ 在 TensorBoard 上可视化和分析词向量

到目前为止，我们已经重点关注了各种模型。我们讨论了全连接模型（例如自动编码器）、卷积神经网络和循环神经网络（例如 LSTM、GRU）。在第十三章，我们谈到了 Transformer，这是一类强大的深度学习模型，为语言理解领域的最新最优性能打下了基础。此外，受到自然语言处理领域的成就启发，Transformer 在计算机视觉领域也引起了轰动。我们已经完成了建模步骤，但还需要通过其他几个步骤来最终收获成果。其中一个步骤是确保向模型提供的数据/特征是正确的，并且模型按预期工作。

在本章中，我们将探索机器学习的一个新方面：利用可视化工具包来可视化高维数据（例如图像、词向量等），以及跟踪和监控模型性能。让我们了解为什么这是一个关键需求。由于机器学习在许多不同领域的成功示范，机器学习已经深深扎根于许多行业和研究领域。因此，这意味着我们需要更快速地训练新模型，并在数据科学工作流程的各个步骤中减少摩擦（例如了解数据、模型训练、模型评估等）。TensorBoard 是迈出这一步的方向。它可以轻松跟踪和可视化数据、模型性能，甚至对模型进行性能分析，了解数据在哪里花费了最多的时间。

通常，您会将数据和模型度量值以及其他要可视化的内容写入日志目录。写入日志目录的内容通常被组织成子文件夹，文件夹的命名包含了日期、时间以及对实验的简要描述。这将有助于在 TensorBoard 中快速识别一个实验。TensorBoard 会不断搜索指定的日志目录以查找更改，并在仪表板上进行可视化。您将在接下来的章节中了解到这些步骤的具体细节。

## 14.1 使用 TensorBoard 可视化数据

假设您是一家时尚公司的数据科学家。他们要求您评估构建一个可以在给定照片中识别服装的模型的可行性。为此，您选择了 Fashion-MNIST 数据集，该数据集包含黑白服装图像，属于 10 个类别之一。一些类别包括 T 恤、裤子和连衣裙。您将首先加载数据并通过 TensorBoard 进行分析，TensorBoard 是一种可视化工具，用于可视化数据和模型。在这里，您将可视化一些图像，并确保在加载到内存后正确分配了类标签。

首先，我们将下载 Fashion-MNIST 数据集。Fashion-MNIST 是一个带有各种服装图片和相应标签/类别的标记数据集。Fashion-MNIST 主要受到了 MNIST 数据集的启发。为了恢复我们的记忆，MNIST 数据集由 28×28 大小的 0-9 数字图像和相应的数字标签构成。由于任务的容易性，许多人建议不再将 MNIST 作为性能基准数据集，因此 Fashion-MNIST 应运而生。与 MNIST 相比，Fashion-MNIST 被认为是一项更具挑战性的任务。

下载 Fashion-MNIST 数据集非常容易，因为它可通过 `tensorflow_datasets` 库获取：

```py
import tensorflow_datasets as tfds

# Construct a tf.data.Dataset
fashion_ds = tfds.load('fashion_mnist')
```

现在，让我们打印来查看数据的格式：

```py
print(fashion_ds)
```

这将返回

```py
{'test': <PrefetchDataset shapes: {image: (28, 28, 1), label: ()}, types: 
➥ {image: tf.uint8, label: tf.int64}>, 'train': <PrefetchDataset shapes: 
➥ {image: (28, 28, 1), label: ()}, types: {image: tf.uint8, label: 
➥ tf.int64}>}
```

数据集包含两个部分，一个训练数据集和一个测试数据集。训练集有两个项：图像（每个图像尺寸为 28×28×1）和一个整数标签。测试集上也有同样的项。接下来，我们将创建三个 `tf.data` 数据集：训练集、验证集和测试集。我们将原始训练数据集分为两部分，一个训练集和一个验证集，然后将测试集保持不变（参见下一个代码清单）。

代码清单 14.1 生成训练、验证和测试数据集

```py
def get_train_valid_test_datasets(fashion_ds, batch_size, 
➥ flatten_images=False):

    train_ds = fashion_ds["train"].shuffle(batch_size*20).map(
        lambda xy: (xy["image"], tf.reshape(xy["label"], [-1]))         ❶
    )
    test_ds = fashion_ds["test"].map(
        lambda xy: (xy["image"], tf.reshape(xy["label"], [-1]))         ❷
    )

    if flatten_images:
        train_ds = train_ds.map(lambda x,y: (tf.reshape(x, [-1]), y))   ❸
        test_ds = test_ds.map(lambda x,y: (tf.reshape(x, [-1]), y))     ❸

    valid_ds = train_ds.take(10000).batch(batch_size)                   ❹

    train_ds = train_ds.skip(10000).batch(batch_size)                   ❺

    return train_ds, valid_ds, test_ds
```

❶ 获取训练数据集，对其进行洗牌，并输出一个(image, label)元组。

❷ 获取测试数据集，并输出一个(image, label)元组。

❸ 将图像扁平化为 1D 向量，用于全连接网络。

❹ 将验证数据集设置为前 10,000 个数据点。

❺ 将训练数据集设置为其余数据。

这是一个简单的数据流程。原始数据集中的每个记录都以字典形式提供，包含两个键：`image` 和 `label`。首先，我们使用 `tf.data.Dataset.map()` 函数将基于字典的记录转换为 `(image, label)` 的元组。然后，如果数据集要用于全连接网络，则可选择性地将 2D 图像扁平化为 1D 向量。换句话说，28 × 28 的图像将变为 784 大小的向量。最后，我们将前 10,000 个数据点（经过洗牌）作为验证集，其余数据作为训练集。

在 TensorBoard 上可视化数据的方式是通过将信息记录到预定义的日志目录中，通过 `tf.summary.SummaryWriter()`。这个写入器将以 TensorBoard 理解的特殊格式写入我们感兴趣的数据。接下来，您启动一个 TensorBoard 的实例，将其指向日志目录。使用这种方法，让我们使用 TensorBoard 可视化一些训练数据。首先，我们定义一个从标签 ID 到字符串标签的映射：

```py
id2label_map = {
    0: "T-shirt/top",
    1: "Trouser",
    2:"Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}
```

这些映射是从 [`mng.bz/DgdR`](http://mng.bz/DgdR) 中获得的。然后我们将定义日志目录。我们将使用日期时间戳来生成不同运行的唯一标识符，如下所示：

```py
log_datetimestamp_format = "%Y%m%d%H%M%S"

log_datetimestamp = datetime.strftime(datetime.now(), 
➥ log_datetimestamp_format)
image_logdir = "./logs/data_{}/train".format(log_datetimestamp)
```

如果你对 `log_datetimestamp_format` 变量中的奇怪格式感到困惑，那么它是 Python 的 `datetime` 库使用的标准格式，用于定义日期和时间的格式（[`mng.bz/lxM2`](http://mng.bz/lxM2)），如果你在你的代码中使用它们。具体来说，我们将会得到运行时间（由 `datetime.now()` 给出）作为一个没有分隔符的数字字符串。我们将得到一天中给定时间的年份（`%Y`）、月份（`%m`）、日期（`%d`）、24 小时制的小时（`%H`）、分钟（`%M`）和秒（`%S`）。然后，我们将数字字符串附加到日志目录中，以根据运行时间创建一个唯一标识符。接下来，我们通过调用以下函数定义一个 `tf.summary.SummaryWriter()`，并将日志目录作为参数传递。有了这个，我们使用这个摘要写入器所做的任何写入都将被记录在定义的目录中：

```py
image_writer = tf.summary.create_file_writer(image_logdir)
```

接下来，我们打开已定义的摘要写入器作为默认写入器，使用一个 `with` 子句。一旦摘要写入器打开，任何 `tf.summary.<数据类型>` 对象都将将该信息记录到日志目录中。在这里，我们使用了一个 `tf.summary.image` 对象。你可以使用几种不同的对象来记录（[`www.tensorflow.org/api_docs/python/tf/summary`](https://www.tensorflow.org/api_docs/python/tf/summary)）：

+ `tf.summary.audio` — 用于记录音频文件并在 TensorBoard 上听取的对象类型
+ `tf.summary.histogram` — 用于记录直方图并在 TensorBoard 上查看的对象类型
+ `tf.summary.image` — 用于记录图像并在 TensorBoard 上查看的对象类型
+ `tf.summary.scalar` — 用于记录标量值（例如，在几个周期内计算的模型损失）并在 TensorBoard 上显示的对象类型
+ `tf.summary.text` — 用于记录原始文本数据并在 TensorBoard 上显示的对象类型

在这里，我们将使用 `tf.summary.image()` 来编写并在 TensorBoard 上显示图像。`tf.summary.image()` 接受几个参数：

+ `name` — 摘要的描述。这将在在 TensorBoard 上显示图像时用作标签。
+ `data` — 大小为 `[b, h, w, c]` 的图像批次，其中 `b` 是批次大小，`h` 是图像高度，`w` 是图像宽度，`c` 是通道数（例如，RGB）。
+ `step` — 一个整数，可用于显示属于不同批次/迭代的图像（默认为 `None`）。
+ `max_outputs` — 在给定时间内显示的最大输出数量。如果数据中的图片数量超过 `max_outputs`，则将显示前 `max_outputs` 个图片，其余图片将被静默丢弃（默认为三）。
+ `description` — 摘要的详细描述（默认为 `None`）

我们将以两种方式编写两组图像数据：

+ 首先，我们将从训练数据集中逐个取出 10 张图像，并带有其类标签标记地写入它们。然后，具有相同类标签（即，类别）的图像将被嵌套在 TensorBoard 上的同一部分中。
+ 最后，我们将一次写入一批 20 张图像。

```py
with image_writer.as_default():
    for data in fashion_ds["train"].batch(1).take(10):
        tf.summary.image(
            id2label_map[int(data["label"].numpy())], 
            data["image"], 
            max_outputs=10, 
            step=0
        )

# Write a batch of 20 images at once
with image_writer.as_default():
    for data in fashion_ds["train"].batch(20).take(1):
        pass
    tf.summary.image("A training data batch", data["image"], max_outputs=20, step=0)
```

然后，我们就可以开始可视化 TensorBoard 了。在 Jupyter 笔记本代码单元格中只需运行以下命令即可简单地初始化和加载 TensorBoard。您现在已经知道，Jupyter 笔记本由单元格组成，您可以在其中输入文本/代码。单元格可以是代码单元格、Markdown 单元格等：

```py
%load_ext tensorboard
%tensorboard --logdir ./logs --port 6006
```

图 14.1 显示了代码在笔记本单元格中的外观。

![14-01](img/14-01.png)

图 14.1 在笔记本单元格中的 Jupyter 魔术命令

您可能会注意到，这不是典型的 Python 语法。以 `%` 符号开头的命令被称为 Jupyter 魔术命令。请记住，Jupyter 是生成笔记本的 Python 库的名称。您可以在 [`mng.bz/BMd1`](http://mng.bz/BMd1) 查看此类命令的列表。第一个命令加载 TensorBoard Jupyter 笔记本扩展程序。第二个命令使用提供的日志目录（`--logdir`）参数和端口（`--port`）参数实例化 TensorBoard。如果不指定端口，则 TensorBoard 默认在 6006（或大于 6006 的第一个未使用的端口）上运行。图 14.2 显示了可视化图像的 TensorBoard 的外观。

![14-02](img/14-02.png)

图 14.2 TensorBoard 可视化记录的图像，以内联方式在 Jupyter 笔记本中显示。您可以对图像执行各种操作，例如调整亮度或对比度。此外，数据被记录到的子目录显示在左侧面板上，让您可以轻松地显示/隐藏不同的子目录以便进行更容易的比较。

或者，您还可以将 TensorBoard 可视化为 Jupyter 笔记本之外的浏览器选项卡。在浏览器中运行这两个命令后，打开 http://localhost:6006，将显示 TensorBoard，如图 14.2 所示。在下一节中，我们将看到在模型训练过程中如何使用 TensorBoard 来跟踪和监视模型性能。

练习 1

您有一个名为 `step_image_batches` 的变量中包含五批图像的列表。列表中的每个项目对应于训练的前五个步骤。您希望在 TensorBoard 中显示这些批次，每个批次都具有正确的步骤值。您可以将每个批次命名为 `batch_0`、`batch_1` 等等。您该如何做？

## 14.2 使用 TensorBoard 跟踪和监视模型

通过对 Fashion-MNIST 数据集中的数据有很好的理解，您将使用神经网络在此数据上训练模型，以衡量您能够对不同类型的服装进行多么准确的分类。您计划使用密集网络和卷积神经网络。您将在相同的条件下训练这两个模型（例如，数据集），并在 TensorBoard 上可视化模型的准确性和损失。

TensorBoard 的主要作用是能够在模型训练时可视化模型的性能。深度神经网络以其长时间的训练时间而闻名。毫无疑问，尽早识别模型中的问题是非常值得的。TensorBoard 在其中发挥着至关重要的作用。您可以将模型性能（通过评估指标）导入到 TensorBoard 中实时显示。因此，您可以在花费过多时间之前快速发现模型中的任何异常行为并采取必要的行动。

在本节中，我们将比较全连接网络和卷积神经网络在 Fashion-MNIST 数据集上的性能。让我们将一个小型全连接模型定义为我们想要使用该数据集测试的第一个模型。它将有三层：

+ 一个具有 512 个神经元和 ReLU 激活的层，该层接收来自 Fashion-MNIST 数据集的平坦图像
+ 一个具有 256 个神经元和 ReLU 激活的层，该层接收前一层的输出
+ 一个具有 softmax 激活的具有 10 个输出的层（表示类别）

该模型使用稀疏分类交叉熵损失和 Adam 优化器进行编译。由于我们对模型准确性感兴趣，因此我们将其添加到要跟踪的指标列表中：

```py
from tensorflow.keras import layers, models

dense_model = models.Sequential([
    layers.Dense(512, activation='relu', input_shape=(784,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
])

dense_model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
```

模型完全定义后，我们对训练数据进行训练，并在验证数据集上进行评估。首先，让我们定义全连接模型的日志目录：

