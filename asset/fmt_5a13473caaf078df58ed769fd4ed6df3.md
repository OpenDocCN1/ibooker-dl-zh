
Keras 是 TensorFlow 的高级深度学习 API：它允许您构建、训练、评估和执行各种神经网络。最初，Keras 库是由 François Chollet 作为研究项目的一部分开发的⁠¹²，并于 2015 年 3 月作为一个独立的开源项目发布。由于其易用性、灵活性和美观的设计，它很快就受到了欢迎。

###### 注意

Keras 曾支持多个后端，包括 TensorFlow、PlaidML、Theano 和 Microsoft Cognitive Toolkit（CNTK）（最后两个遗憾地已弃用），但自版本 2.4 以来，Keras 仅支持 TensorFlow。同样，TensorFlow 曾包括多个高级 API，但在 TensorFlow 2 发布时，Keras 被正式选择为其首选的高级 API。安装 TensorFlow 将自动安装 Keras，并且没有安装 TensorFlow，Keras 将无法工作。简而言之，Keras 和 TensorFlow 相爱并结为夫妻。其他流行的深度学习库包括[Facebook 的 PyTorch](https://pytorch.org)和[Google 的 JAX](https://github.com/google/jax)。¹³

现在让我们使用 Keras！我们将首先构建一个用于图像分类的 MLP。

###### 注意

Colab 运行时已预装了最新版本的 TensorFlow 和 Keras。但是，如果您想在自己的机器上安装它们，请参阅[*https://homl.info/install*](https://homl.info/install)上的安装说明。

## 使用顺序 API 构建图像分类器

首先，我们需要加载一个数据集。我们将使用时尚 MNIST，它是 MNIST 的一个替代品（在第三章介绍）。它与 MNIST 具有完全相同的格式（70,000 个 28×28 像素的灰度图像，共 10 个类），但图像代表时尚物品而不是手写数字，因此每个类更加多样化，问题变得比 MNIST 更具挑战性。例如，一个简单的线性模型在 MNIST 上达到约 92%的准确率，但在时尚 MNIST 上只有约 83%。

### 使用 Keras 加载数据集

Keras 提供了一些实用函数来获取和加载常见数据集，包括 MNIST、时尚 MNIST 等。让我们加载时尚 MNIST。它已经被洗牌并分成一个训练集（60,000 张图片）和一个测试集（10,000 张图片），但我们将从训练集中保留最后的 5,000 张图片用于验证：

```py
import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]
```

###### 提示

TensorFlow 通常被导入为`tf`，Keras API 可通过`tf.keras`使用。

使用 Keras 加载 MNIST 或时尚 MNIST 时，与 Scikit-Learn 相比的一个重要区别是，每个图像都表示为一个 28×28 的数组，而不是大小为 784 的一维数组。此外，像素强度表示为整数（从 0 到 255），而不是浮点数（从 0.0 到 255.0）。让我们看看训练集的形状和数据类型：

```py
>>> X_train.shape
(55000, 28, 28)
>>> X_train.dtype
dtype('uint8')
```

为简单起见，我们将通过将它们除以 255.0 来将像素强度缩放到 0-1 范围（这也将它们转换为浮点数）：

```py
X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255.
```

对于 MNIST，当标签等于 5 时，这意味着图像代表手写数字 5。简单。然而，对于时尚 MNIST，我们需要类名列表以了解我们正在处理的内容：

```py
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
```

例如，训练集中的第一张图像代表一个踝靴：

```py
>>> class_names[y_train[0]]
'Ankle boot'
```

图 10-10 显示了时尚 MNIST 数据集的一些样本。

![mls3 1010](img/mls3_1010.png)

###### 图 10-10。时尚 MNIST 的样本

### 使用顺序 API 创建模型

现在让我们构建神经网络！这是一个具有两个隐藏层的分类 MLP：

```py
tf.random.set_seed(42)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=[28, 28]))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(300, activation="relu"))
model.add(tf.keras.layers.Dense(100, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))
```

让我们逐行查看这段代码：

+   首先，设置 TensorFlow 的随机种子以使结果可重现：每次运行笔记本时，隐藏层和输出层的随机权重将保持相同。您还可以选择使用`tf.keras.utils.set_random_seed()`函数，它方便地为 TensorFlow、Python（`random.seed()`）和 NumPy（`np.random.seed()`）设置随机种子。

+   下一行创建一个`Sequential`模型。这是 Keras 模型中最简单的一种，用于仅由一系列按顺序连接的层组成的神经网络。这被称为顺序 API。

+   接下来，我们构建第一层（一个`Input`层）并将其添加到模型中。我们指定输入的`shape`，它不包括批量大小，只包括实例的形状。Keras 需要知道输入的形状，以便确定第一个隐藏层的连接权重矩阵的形状。

+   然后我们添加一个`Flatten`层。它的作用是将每个输入图像转换为 1D 数组：例如，如果它接收到一个形状为`[32, 28, 28]`的批量，它将将其重塑为`[32, 784]`。换句话说，如果它接收到输入数据`X`，它会计算`X.reshape(-1, 784)`。这个层没有任何参数；它只是用来进行一些简单的预处理。

+   接下来我们添加一个具有 300 个神经元的`Dense`隐藏层。它将使用 ReLU 激活函数。每个`Dense`层都管理着自己的权重矩阵，其中包含神经元与它们的输入之间的所有连接权重。它还管理着一个偏置项向量（每个神经元一个）。当它接收到一些输入数据时，它会计算方程 10-2。

+   然后我们添加一个具有 100 个神经元的第二个`Dense`隐藏层，同样使用 ReLU 激活函数。

+   最后，我们添加一个具有 10 个神经元（每个类一个）的`Dense`输出层，使用 softmax 激活函数，因为类是互斥的。

###### 提示

指定`activation="relu"`等同于指定`activation=tf.keras.activations.relu`。其他激活函数可以在`tf.keras.activations`包中找到。我们将在本书中使用许多这些激活函数；请参阅[*https://keras.io/api/layers/activations*](https://keras.io/api/layers/activations)获取完整列表。我们还将在第十二章中定义我们自己的自定义激活函数。

与刚刚逐个添加层不同，通常更方便的做法是在创建`Sequential`模型时传递一个层列表。您还可以删除`Input`层，而是在第一层中指定`input_shape`：

```py
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
```

模型的`summary()`方法显示了所有模型的层，包括每个层的名称（除非在创建层时设置了名称，否则会自动生成），其输出形状（`None`表示批量大小可以是任意值），以及其参数数量。摘要以总参数数量结束，包括可训练和不可训练参数。在这里我们只有可训练参数（您将在本章后面看到一些不可训练参数）：

```py
>>> model.summary()
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 flatten (Flatten)           (None, 784)               0

 dense (Dense)               (None, 300)               235500

 dense_1 (Dense)             (None, 100)               30100

 dense_2 (Dense)             (None, 10)                1010

=================================================================
Total params: 266,610
Trainable params: 266,610
Non-trainable params: 0
_________________________________________________________________
```

注意，`Dense`层通常具有*大量*参数。例如，第一个隐藏层有 784 × 300 个连接权重，再加上 300 个偏置项，总共有 235,500 个参数！这使得模型具有相当大的灵活性来拟合训练数据，但也意味着模型有过拟合的风险，特别是当训练数据不多时。我们稍后会回到这个问题。

模型中的每个层必须具有唯一的名称（例如，`"dense_2"`）。您可以使用构造函数的`name`参数显式设置层名称，但通常最好让 Keras 自动命名层，就像我们刚刚做的那样。Keras 获取层的类名并将其转换为蛇形命名法（例如，`MyCoolLayer`类的层默认命名为`"my_cool_layer"`）。Keras 还确保名称在全局范围内是唯一的，即使跨模型也是如此，如果需要，会附加索引，例如`"dense_2"`。但是为什么要确保名称在模型之间是唯一的呢？这样可以轻松合并模型而不会出现名称冲突。

###### 提示

Keras 管理的所有全局状态都存储在*Keras 会话*中，您可以使用`tf.keras.backend.clear_session()`清除它。特别是，这将重置名称计数器。

您可以使用`layers`属性轻松获取模型的层列表，或使用`get_layer()`方法按名称访问层：

```py
>>> model.layers
[<keras.layers.core.flatten.Flatten at 0x7fa1dea02250>,
 <keras.layers.core.dense.Dense at 0x7fa1c8f42520>,
 <keras.layers.core.dense.Dense at 0x7fa188be7ac0>,
 <keras.layers.core.dense.Dense at 0x7fa188be7fa0>]
>>> hidden1 = model.layers[1]
>>> hidden1.name
'dense'
>>> model.get_layer('dense') is hidden1
True
```

可以使用其`get_weights()`和`set_weights()`方法访问层的所有参数。对于`Dense`层，这包括连接权重和偏差项：

```py
>>> weights, biases = hidden1.get_weights()
>>> weights
array([[ 0.02448617, -0.00877795, -0.02189048, ...,  0.03859074, -0.06889391],
 [ 0.00476504, -0.03105379, -0.0586676 , ..., -0.02763776, -0.04165364],
 ...,
 [ 0.07061854, -0.06960931,  0.07038955, ..., 0.00034875,  0.02878492],
 [-0.06022581,  0.01577859, -0.02585464, ..., 0.00272203, -0.06793761]],
 dtype=float32)
>>> weights.shape
(784, 300)
>>> biases
array([0., 0., 0., 0., 0., 0., 0., 0., 0., ...,  0., 0., 0.], dtype=float32)
>>> biases.shape
(300,)
```

请注意，`Dense`层随机初始化连接权重（这是为了打破对称性，如前所述），偏差初始化为零，这是可以的。如果要使用不同的初始化方法，可以在创建层时设置`kernel_initializer`（*kernel*是连接权重矩阵的另一个名称）或`bias_initializer`。我们将在第十一章进一步讨论初始化器，完整列表在[*https://keras.io/api/layers/initializers*](https://keras.io/api/layers/initializers)。

###### 注意

权重矩阵的形状取决于输入的数量，这就是为什么在创建模型时我们指定了`input_shape`。如果您没有指定输入形状，没关系：Keras 会等到知道输入形状后才真正构建模型参数。这将在您提供一些数据（例如，在训练期间）或调用其`build()`方法时发生。在模型参数构建之前，您将无法执行某些操作，例如显示模型摘要或保存模型。因此，如果在创建模型时知道输入形状，最好指定它。

### 编译模型

创建模型后，必须调用其`compile()`方法来指定要使用的损失函数和优化器。可选地，您可以指定在训练和评估过程中计算的额外指标列表：

```py
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
```

###### 注意

使用`loss="sparse_categorical_crossentropy"`等同于使用`loss=tf.keras.losses.sparse_categorical_​cross⁠entropy`。同样，使用`optimizer="sgd"`等同于使用`optimizer=tf.keras.optimizers.SGD()`，使用`metrics=["accuracy"]`等同于使用`metrics=​[tf.keras.metrics.sparse_categorical_accuracy`]（使用此损失时）。在本书中，我们将使用许多其他损失、优化器和指标；有关完整列表，请参见[*https://keras.io/api/losses*](https://keras.io/api/losses)、[*https://keras.io/api/optimizers*](https://keras.io/api/optimizers)和[*https://keras.io/api/metrics*](https://keras.io/api/metrics)。

这段代码需要解释。我们使用`"sparse_categorical_crossentropy"`损失，因为我们有稀疏标签（即，对于每个实例，只有一个目标类索引，本例中为 0 到 9），并且类是互斥的。如果相反，对于每个实例有一个目标概率类（例如，独热向量，例如，`[0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]`表示类 3），那么我们需要使用`"categorical_crossentropy"`损失。如果我们进行二元分类或多标签二元分类，则在输出层中使用`"sigmoid"`激活函数，而不是`"softmax"`激活函数，并且我们将使用`"binary_crossentropy"`损失。

###### 提示

如果你想将稀疏标签（即类别索引）转换为独热向量标签，请使用`tf.keras.utils.to_categorical()`函数。要反过来，使用带有`axis=1`的`np.argmax()`函数。
