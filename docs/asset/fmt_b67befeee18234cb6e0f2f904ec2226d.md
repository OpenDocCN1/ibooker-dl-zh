
# 第三章：Keras 和 TensorFlow 2 中的数据检索

本章涵盖的内容

+ Keras 中用于构建模型的不同 API
+ 检索和操作持久化数据

我们已经探讨了低级 TensorFlow API 的细节，比如定义`tf.Variable`对象和`tf.Tensor`对象，这些对象可以用来存储数字和字符串等。我们还查看了 TensorFlow 提供的一些常用功能，如`tf.Operation`。最后，我们详细讨论了一些复杂的操作，比如矩阵乘法和卷积。如果你分析任何标准的深度神经网络，你会发现它由矩阵乘法和卷积等标准数学操作构成。

然而，如果你要使用低级别的 TensorFlow API 实现这些网络，你会发现自己在代码中多次复制这些操作，这将耗费宝贵的时间，并使代码难以维护。但好消息是你不需要这样做。TensorFlow 提供了一个名为 Keras 的子模块，它解决了这个问题，这也是本章的重点。Keras 是 TensorFlow 中的一个子库，它隐藏了构建模块，并为开发机器学习模型提供了高级 API。在本章中，我们将看到 Keras 提供了几种不同的 API，可以根据解决方案的复杂性来选择使用。

我们将通过讨论机器学习的另一个重要方面来结束本章：向模型提供数据。通常，我们需要从磁盘（或网络）中检索数据，并在将其提供给模型之前清理和处理数据。我们将讨论 TensorFlow 中几种不同的数据检索工具，如`tf.data`和`tensorflow-datasets` API，以及它们如何简化读取和操作最终输入模型的数据。

## 3.1 Keras 模型构建 API

作为黑客马拉松的一部分，您正在开发一个花卉物种分类器。您的团队将创建几个不同的多层感知器变体，以便将它们的性能与花卉物种识别数据集进行比较。目标是训练能够在给定花卉的多个测量值的情况下输出花卉物种的模型。您需要开发的模型如下：

+ *模型 A*——仅从提供的特征中学习的模型（基线）
+ *模型 B*——除了使用特征本身外，还使用特征的主成分（详见 3.1.3 节）
+ *模型 C*——一个使用非常规隐藏层计算的模型，它使用了一个乘法偏差，除了传统的加法偏差之外，这在神经网络中通常是找不到的（详见 3.1.4 节）

您计划使用 Keras，并且知道它提供了多个模型构建 API。为了快速提供结果，您需要知道在哪个模型中使用哪个 Keras API。

Keras（[`keras.io/`](https://keras.io/)）最初是作为一个高级 API 启动的，可以使用多个低级后端（例如，TensorFlow、Theano），并允许开发人员轻松构建机器学习模型。换句话说，Keras 隐藏了低级操作的细节，并提供了一个直观的 API，您可以用几行代码构建模型。自从 TensorFlow 1.4 以来，Keras 已经集成到了 TensorFlow 中（[`www.tensorflow.org/guide/keras/overview`](https://www.tensorflow.org/guide/keras/overview)）。您可以使用`import tensorflow.keras`导入 Keras。Keras 有三个主要的 API：

+ 顺序式
+ 函数式
+ 子类化

顺序式 API 是最容易使用的。然而，它是一个非常受限制的 API，只允许您创建一个以一个输入开始，经过一系列层，以及以一个输出结束的网络。接下来，函数式 API 需要更多的工作才能使用。但它也提供了更多的灵活性，例如具有多个输入、并行层和多个输出。最后，子类化 API 可以被认为是最难驾驭的。其思想是创建一个代表您的模型或模型中的一层的 Python 对象，同时使用 TensorFlow 提供的低级功能来实现所需的功能。让我们简要地介绍一下如何使用这些 API。但我们不会止步于此；在接下来的章节中，我们将更详细地了解这些 API。图 3.1 突出显示了 API 之间的主要区别。

![03-01](img/03-01.png)

图 3.1 顺序式、函数式和子类化 API 的比较。

在这里，对于模型 A，我们将使用顺序式 API，因为它是最简单的。要实现模型 B，其中将有两个输入层，我们将使用函数式 API。最后，要实现模型 C，其中我们需要实现一个自定义层，我们将使用子类化 API。

### 3.1.1 引入数据集

假设您决定使用一个名为鸢尾花数据集（[`archive.ics.uci.edu/ml/datasets/Iris`](https://archive.ics.uci.edu/ml/datasets/Iris)）的流行机器学习数据集。这个数据集记录了几种不同种类的鸢尾花（Iris-setosa、Iris-versicolor 和 Iris-virginica）的萼片长度、萼片宽度、花瓣长度和花瓣宽度。对于每朵花，我们都有萼片长度/宽度和花瓣长度/宽度。正如您所看到的，每个输入都有四个特征，每个输入都可以属于三个类别之一。首先，让我们下载数据，对其进行快速分析，并将其格式化为我们可以方便地用于模型训练的格式。

首先，您需要确保环境设置正确，并且已安装所需的库，如附录 A 中所述。接下来，打开位于`Ch03-Keras-and-Data-Retrieval/3.1.Keras_APIs.ipynb`的 Jupyter 笔记本。现在，如笔记本中的代码所示，我们需要导入`requests`库来下载数据，导入`pandas`来操作该数据，当然，还有 TensorFlow：

```py
import requests
import pandas as pd
import tensorflow as tf
```

现在我们将下载数据并将数据保存到文件中：

```py
url = "https:/ /archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
r = requests.get(url)

# Writing data to a file
with open('iris.data', 'wb') as f:
  f.write(r.content)
```

然后，我们使用`pandas`库的`read_csv()`函数读取数据（[`mng.bz/j2Op`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.xhtml)）：

```py
iris_df = pd.read_csv('iris.data', header=None)
```

这里，`iris_df`是一个`pandas` DataFrame（[`mng.bz/Wxaw`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.xhtml)）。在最简单的形式下，数据帧可以被认为是一个按行和列组织的信息矩阵。您可以使用`iris_df.head()`命令检查数据的前几行，其结果如下：

```py
0     1       2       3       4
0     5.1     3.5     1.4     0.2     Iris-setosa
1     4.9     3.0     1.4     0.2     Iris-setosa
2     4.7     3.2     1.3     0.2     Iris-setosa
```

然后，我们将对数据进行一些装饰性修改，使其看起来更好。我们将提供适当的列名称（可从数据集的网页中获取）：

```py
iris_df.columns = ['sepal_length', 'sepal_width', 'petal_width', 'petal_length', 'label']
```

并将字符串标签映射为整数：

```py
iris_df["label"] = iris_df["label"].map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})
```

我们得到了以下改进后的`pandas` DataFrame：

```py
      sepal_length   sepal_width   petal_width   petal_length   label
0     5.1             3.5           1.4           0.2            0
1     4.9             3.0           1.4           0.2            0
2     4.7             3.2           1.3           0.2            0
```

作为最后一步，我们将通过从每列中减去均值来将数据居中，因为这通常会导致更好的性能：

```py
iris_df = iris_df.sample(frac=1.0, random_state=4321)
x = iris_df[["sepal_length", "sepal_width", "petal_width", "petal_length"]]
x = x - x.mean(axis=0)
y = tf.one_hot(iris_df["label"], depth=3)
```

在这里，`print(x)`将输出：

```py
        sepal_length  sepal_width  petal_width  petal_length
31      -0.443333        0.346    -2.258667     -0.798667
23      -0.743333        0.246    -2.058667     -0.698667
70       0.056667        0.146     1.041333      0.601333
100      0.456667        0.246     2.241333      1.301333
44      -0.743333        0.746    -1.858667     -0.798667
..            ...          ...          ...           ...
```

注意：在对数据进行洗牌后，索引不再按顺序排列。`print(y)`将输出：

```py
tf.Tensor(
    [[1\. 0\. 0.]
     [1\. 0\. 0.]
     [0\. 1\. 0.]
     ...
     [0\. 0\. 1.]
     [0\. 0\. 1.]
     [0\. 1\. 0.]], 
shape=(150, 3), dtype=float32)
```

对数据进行洗牌是一个重要的步骤：数据是按特定顺序排列的，每个类别都紧跟在另一个后面。但是当数据被洗牌时，每个被呈现给网络的批次都有所有类别的良好混合，这样可以获得最佳结果。您还可以看到我们对`y`（或标签）使用了一种称为*独热编码*的转换。独热编码将每个标签转换为唯一的零向量，其中一个元素为一。例如，标签 0、1 和 2 被转换为以下独热编码向量：

0 → [1, 0, 0]

1 → [0, 1, 0]

2 → [0, 0, 1]

### 3.1.2 顺序 API

数据准备就绪后，是时候实现模型 A 了，这是第一个神经网络。第一个模型非常简单，只需要提供的特征并预测花的种类。您可以使用 Keras 顺序 API，因为它是最简单的，我们所需要做的就是将几个层顺序堆叠在一起。图 3.2 描述了顺序 API 与其他 API 的比较。

![03-02](img/03-02.png)

图 3.2 顺序 API 与其他 API 的比较（被标记为灰色）

让我们创建一个具有以下特点的网络：

+ 一个具有 4 个节点的输入层
+ 一个具有 32 个节点的隐藏层
+ 一个具有 16 个节点的隐藏层
+ 一个 3 节点输出层

注意：每层节点数是模型的超参数。在这种情况下，我们任意选择了这些值。但为了获得最佳结果，我们应该使用一个超参数优化算法（[`mng.bz/8MJB`](http://mng.bz/8MJB)）来找到给定问题的最佳超参数。

在定义模型之前，我们需要导入 TensorFlow 中的某些层和顺序模型。然后，您可以使用一行代码即可实现该模型（见下一个清单）。

利用 Sequential API 实现的模型 A，如 3.1 清单所示。

```py
from tensorflow.keras.layers import Dense            ❶
from tensorflow.keras.models import Sequential       ❶
import tensorflow.keras.backend as K                 ❶

K.clear_session()                                    ❷
model = Sequential([                                 ❸
    Dense(32, activation='relu', input_shape=(4,)),  ❸
    Dense(16, activation='relu'),                    ❸
    Dense(3, activation='softmax')                   ❸
])
```

❶导入必要的模块和类。

❷在创建模型之前清除 TensorFlow 计算图。

❸使用 Sequential API 定义模型。

让我们分析一下我们刚刚做的。您可以使用`Sequential`对象创建一个顺序模型，然后传递一个层的序列，例如`Dense`层。一个层封装了神经网络中可以找到的典型的可重复使用的计算（例如隐藏层计算，卷积操作）。

`Dense`层提供了全连接网络中所发生的核心计算（即通过*h* *=* *activation*(*xW* *+* *b*)从输入（x）到隐藏输出（h））。`Dense`层有两个重要参数：隐藏单元的数量和非线性激活函数。通过堆叠一组`Dense`层，您可以构建一个多层的全连接网络。我们正在使用以下层构建网络：

+ `Dense(32, activation='relu', input_shape=(4,))`
+ `Dense(16, activation='relu')`
+ `Dense(3, activation='softmax')`

在第一个`Dense`层中，可以看到传递了一个额外的参数`input_shape`。`input_shape`是使用 TensorFlow 创建的任何模型的关键属性。您必须确切地知道要传递给模型的输入的形状，因为随后的所有层的输出都取决于输入的形状。实际上，某些层只能处理某些特定的输入形状。

在这个例子中，我们说输入的形状将是`[None, 4]`。虽然我们只在形状中指定了 4，但 Keras 会自动添加一个未指定的（即 None）维度到`input_shape`中，它表示输入的批次维度。正如您可能已经知道的，深度神经网络以批次的方式处理数据（即一次处理多个示例）。另一个尺寸（大小为 4）是特征维度，意味着网络可以接受具有四个特征的输入。将批次维度设为 None 将批次维度未指定，允许您在模型训练/推断时传递任意数量的示例。

一个层的另一个重要方面是层中使用的非线性激活函数。在这里，我们可以看到前两个层使用了 ReLU（修正线性单元）激活函数。它是前馈模型中非常简单但功能强大的激活函数。ReLU 具有以下功能：

*y* = max (0, *x*)

最后一层使用了 softmax 激活函数。正如之前讨论的，softmax 激活函数将最后一层（即 logits）的得分归一化为一个有效的概率分布。具体来说，

![03_02a](img/03_02a.png)

以一个示例为例，假设最后一层没有使用 softmax 激活函数产生了：

```py
[15, 30, 5]
```

应用 softmax 归一化将这些值转换为：

```py
[15/(15+30+5), 30/(15+30+5), 5/(15+30+5)]
= [0.3, 0.6, 0.1]
```

现在模型已经定义好了，我们需要执行一个关键步骤，称为*模型编译*，如果我们要成功地使用它的话。对于我们的模型，我们将使用：

```py
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
```

在这里，我们设置了模型的损失函数、优化器和度量标准。损失函数表示模型在给定数据上的表现如何（例如，分类交叉熵）。损失越低，模型就越好。除了损失函数之外，我们还使用了一个优化器，它知道如何改变模型的权重和偏差，以使损失减少。在这里，我们选择了损失函数`categorical_crossentropy`（[`mng.bz/EWej`](http://mng.bz/EWej)），这通常在多类别分类问题中效果良好，以及优化器`adam`（[`arxiv.org/pdf/1412.6980.pdf`](https://arxiv.org/pdf/1412.6980.pdf)），由于其在各种问题中的出色表现，是一个常见的选择。我们还可以选择性地定义度量标准来关注模型（例如，模型准确率）。最后，我们可以使用以下方法检查您刚刚创建的模型：

```py
model.summary()
```

输出
