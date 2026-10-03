
#### 使用 `@jax.jit` 使 JAX 函数快速

还有一件事。作为一个 JAX 用户，你将经常使用`@jax.jit`装饰器，它与`@tf.function(jit_compile=True)`装饰器行为相同。它将任何无状态的 JAX 函数转换为 XLA 编译的代码片段，通常能带来显著的执行速度提升：

```py
@jax.jit
def dense(inputs, W, b):
    return jax.nn.relu(jnp.matmul(inputs, W) + b) 
```

请注意，你只能装饰无状态函数——任何被函数更新的张量都应该是其返回值的一部分。

### 一个端到端示例：纯 JAX 中的线性分类器

现在你已经了解了足够的 JAX 知识，可以编写我们线性分类器的 JAX 版本。与 TensorFlow 和 PyTorch 版本相比，有两个主要区别：

+   我们将创建的所有函数都将是无状态的。这意味着状态（数组`W`和`b`）将作为函数参数提供，如果它们被函数修改，函数将返回它们的新的值。

+   梯度是通过 JAX 的`value_and_grad()`工具计算的。

让我们开始吧。模型函数和均方误差函数应该看起来很熟悉：

```py
def model(inputs, W, b):
    return jnp.matmul(inputs, W) + b

def mean_squared_error(targets, predictions):
    per_sample_losses = jnp.square(targets - predictions)
    return jnp.mean(per_sample_losses) 
```

为了计算梯度，我们需要将损失计算包装在单个`compute_loss()`函数中。它返回一个标量作为总损失，并接受`state`作为其第一个参数——我们需要计算梯度的所有张量的元组：

```py
def compute_loss(state, inputs, targets):
    W, b = state
    predictions = model(inputs, W, b)
    loss = mean_squared_error(targets, predictions)
    return loss 
```

在这个函数上调用`jax.value_and_grad()`给我们一个新的函数，具有与`compute_loss`相同的参数，它返回损失和相对于`state`元素的损失梯度：

```py
grad_fn = jax.value_and_grad(compute_loss) 
```

接下来，我们可以设置我们的训练步骤函数。它看起来很简单。请注意，与 TensorFlow 和 PyTorch 的等效函数不同，它必须是状态无关的，因此它必须返回`W`和`b`张量的更新值：

```py
learning_rate = 0.1

# We use the jax.jit decorator to take advantage of XLA compilation.
@jax.jit
def training_step(inputs, targets, W, b):
    # Computes the forward pass and backward pass in one go
    loss, grads = grad_fn((W, b), inputs, targets)
    grad_wrt_W, grad_wrt_b = grads
    # Updates W and b
    W = W - grad_wrt_W * learning_rate
    b = b - grad_wrt_b * learning_rate
    # Make sure to return the new values of W and b in addition to the
    # loss!
    return loss, W, b 
```

因为在我们的例子中我们不会改变`learning_rate`，所以我们可以将其视为函数本身的一部分，而不是我们模型的状态。如果我们想在训练过程中修改学习率，我们也需要将其传递过去。

最后，我们准备运行完整的训练循环。我们初始化`W`和`b`，然后通过无状态的`training_step()`调用反复更新它们：

```py
input_dim = 2
output_dim = 1

W = jax.numpy.array(np.random.uniform(size=(input_dim, output_dim)))
b = jax.numpy.array(np.zeros(shape=(output_dim,)))
state = (W, b)
for step in range(40):
    loss, W, b = training_step(inputs, targets, W, b)
    print(f"Loss at step {step}: {loss:.4f}") 
```

就这些！你现在可以编写自定义的 JAX 训练循环了。

### JAX 方法独特之处在于

在现代机器学习框架中，使 JAX 独特的最主要因素是其函数式、无状态的哲学。虽然一开始可能会觉得这会引起一些摩擦，但它正是解锁 JAX 力量的关键——它能够编译成极快的代码，并且能够扩展到任意大的模型和任意多的设备。

JAX 有很多优点：

+   它很快。对于大多数模型来说，它是你迄今为止见过的所有框架中最快的。

+   它的数值 API 与 NumPy 完全一致，这使得学习起来很愉快。

+   这是最适合在 TPU 上训练模型的，因为它从头开始就是为 XLA 和 TPU 开发的。

使用 JAX 也可能带来一些开发者的摩擦：

+   与纯即时执行相比，它的元编程和编译使用可能会使其调试变得显著困难。

+   与 TensorFlow 或 PyTorch 相比，低级训练循环通常更冗长，更难编写。

到目前为止，你已经了解了 TensorFlow、PyTorch 和 JAX 的基础知识，并且可以使用这些框架从头开始实现一个基本的线性分类器。这是一个坚实的基石，现在我们可以转向更有效的深度学习路径：Keras API。

## Keras 简介

Keras 是一个用于 Python 的深度学习 API，它提供了一种方便的方式来定义和训练任何类型的深度学习模型。它于 2015 年 3 月发布，2017 年发布了 v2 版本，2023 年发布了 v3 版本。

Keras 的用户包括学术研究人员、初创公司和大型公司的工程师和数据科学家、研究生和爱好者。Keras 被用于 Google、Netflix、Uber、YouTube、CERN、NASA、Yelp、Instacart、Square、Waymo、YouTube 以及成千上万的在各个行业中解决各种问题的较小组织。你的 YouTube 推荐来自 Keras 模型。Waymo 的自动驾驶汽车依赖于 Keras 模型来处理传感器数据。Keras 也是 Kaggle 机器学习竞赛网站上的一个流行框架。

由于 Keras 拥有多样化的用户群体，它不会强迫你遵循单一的“正确”的构建和训练模型的方式。相反，它允许广泛的不同工作流程，从非常高级到非常低级，对应不同的用户画像。例如，你有多种构建模型的方法和多种训练它们的方法，每种方法都代表了一定程度上的可用性和灵活性的权衡。在第七章中，我们将详细回顾这部分工作流程的很大一部分。

### Keras 的入门步骤

在我们编写 Keras 代码之前，在导入库之前有一些事情需要考虑。

#### 选择后端框架

Keras 可以与 JAX、TensorFlow 或 PyTorch 一起使用。它们是 Keras 的“后端框架”。通过这些后端框架，Keras 可以在不同的硬件类型上运行（参见图 3.4）——GPU、TPU 或普通 CPU——可以无缝扩展到数千台机器，并且可以部署到各种平台。

![图片](img/552c279c016c6f17a2d903764c737676.png)

图 3.4：Keras 及其后端。后端是一个低级张量计算平台；Keras 是一个高级深度学习 API。

后端框架是可插拔的：在你编写了一些 Keras 代码之后，你可以切换到不同的后端框架。你不会被锁定在单个框架和单个生态系统中——你可以根据当前需求将你的模型从 JAX 切换到 TensorFlow 或 PyTorch。例如，当你开发一个 Keras 模型时，你可以用 PyTorch 调试它，用 JAX 在 TPU 上进行训练以获得最大效率，最后使用 TensorFlow 生态系统中的优秀工具进行推理。

目前 Keras 的默认后端是 TensorFlow，所以如果你在一个全新的环境中运行`import keras`而没有进行任何配置，你将运行在 TensorFlow 上。有两种方法可以选择不同的后端：

+   设置环境变量`KERAS_BACKEND`。在你开始`python` repl 之前，你可以运行以下 shell 命令来使用 JAX 作为你的 Keras 后端：`export KERAS_BACKEND=jax`。或者，你可以在你的 Python 文件或笔记本的顶部添加以下代码片段（请注意，它必须强制性地放在第一个`import keras`之前）：

```py
import os

# Sets the environment variable from within the Python runtime
os.environ["KERAS_BACKEND"] = "jax"

# Only then should you import Keras.
import keras 
```

+   在`~/.keras/keras.json`位置编辑你的本地 Keras 配置文件。如果你已经导入过一次 Keras，这个文件已经根据默认设置创建好了。你可以使用任何文本编辑器打开和修改它——它是一个可读的 JSON 文件。它应该看起来像这样：

```py
{
    # Default floating-point precision. It should typically not be
    # changed.
    "floatx": "float32",
    # Default numerical fuzzing factor. It should typically not be
    # changed.
    "epsilon": 1e-07,
    # Change "tensorflow" to "jax" or "torch."
    "backend": "tensorflow",
    # This is the default image layout. We'll talk about this in
    # chapter 8.
    "image_data_format": "channels_last",
} 
```

现在，你可能想知道，我应该选择哪个后端？这完全取决于你自己的选择：本书中所有其余部分的 Keras 代码示例都将与所有三个后端兼容。如果需要特定后端的代码（例如在第七章中），我会展示三个版本——TensorFlow、PyTorch 和 JAX。如果你没有特别的后端偏好，我个人的推荐是 JAX。它通常是性能最好的后端。

一旦你的后端配置完成，你就可以开始实际构建和训练 Keras 模型了。让我们看看。

### 层：深度学习的构建块

神经网络中的基本数据结构是*层*，这在第二章中已经向你介绍了。层是一个数据处理模块，它接受一个或多个张量作为输入，并输出一个或多个张量。有些层是无状态的，但更常见的是层有状态：层的*权重*，这些是通过随机梯度下降学习的一个或多个张量，它们共同包含了网络的*知识*。

不同的层类型适用于不同的张量格式和不同类型的数据处理。例如，简单的向量数据，存储在形状为`(samples, features)`的 2D 张量中，通常由*密集连接*层处理，也称为*全连接*或*密集*层（Keras 中的`Dense`类）。序列数据，存储在形状为`(samples, timesteps, features)`的 3D 张量中，通常由*循环*层处理，例如`LSTM`层，或者 1D 卷积层（`Conv1D`）。图像数据，存储在四阶张量中，通常由 2D 卷积层（`Conv2D`）处理。

你可以将层视为深度学习的乐高积木，这是一个由 Keras 明确提出的比喻。在 Keras 中构建深度学习模型是通过将兼容的层拼接在一起来形成有用的数据转换管道。

#### Keras 中的基本`Layer`类

一个简单的 API 应该围绕一个单一的抽象来构建，Keras 中那就是`Layer`类。Keras 中的所有内容要么是`Layer`，要么是与`Layer`紧密交互的东西。

`Layer`是一个封装了一些状态（权重）和一些计算（前向传递）的对象。权重通常在`build()`中定义（尽管它们也可以在构造函数`__init__()`中创建），计算在`call()`方法中定义。

在上一章中，我们实现了一个`NaiveDense`类，它包含两个权重`W`和`b`，并应用了计算`output = activation(matmul(input, W) + b)`。以下是在 Keras 中相同层的样子。

```py
import keras

# All Keras layers inherit from the base Layer class.
class SimpleDense(keras.Layer):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation

    # Weight creation takes place in the build() method.
    def build(self, input_shape):
        batch_dim, input_dim = input_shape
        # add_weight is a shortcut method for creating weights. It's
        # also possible to create standalone variables and assign them
        # as layer attributes, like self.W = keras.Variable(shape=...,
        # initializer=...).
        self.W = self.add_weight(
            shape=(input_dim, self.units), initializer="random_normal"
        )
        self.b = self.add_weight(shape=(self.units,), initializer="zeros")

    # We define the forward pass computation in the call() method.
    def call(self, inputs):
        y = keras.ops.matmul(inputs, self.W) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y 
```

列表 3.33：Keras 中从头开始创建一个简单的密集层

在下一节中，我们将详细介绍这些`build()`和`call()`方法的目的。如果你现在还不完全理解，请不要担心！

一旦实例化，这样的层就可以像函数一样使用，接受一个张量作为输入：

```py
>>> # Instantiates our layer, defined previously
>>> my_dense = SimpleDense(units=32, activation=keras.ops.relu)
>>> # Creates some test inputs
>>> input_tensor = keras.ops.ones(shape=(2, 784))
>>> # Calls the layer on the inputs, just like a function
>>> output_tensor = my_dense(input_tensor)
>>> print(output_tensor.shape)
(2, 32)
```

现在，你可能想知道，为什么我们必须实现`call()`和`build()`，因为我们最终是通过直接调用它，即使用它的`__call__`方法来使用我们的层的？这是因为我们希望能够在需要时创建状态。让我们看看它是如何工作的。

#### 自动形状推断：动态构建层

就像乐高积木一样，你只能“拼接”兼容的层。这里的*层兼容性*概念具体指的是每个层将只接受特定形状的张量作为输入，并返回特定形状的张量作为输出。考虑以下示例：

```py
from keras import layers

# A dense layer with 32 output units
layer = layers.Dense(32, activation="relu") 
```

这个层将返回一个非批量维度为 32 的张量。它只能连接到一个期望 32 维向量作为输入的下游层。

当使用 Keras 时，你大多数时候不必担心尺寸兼容性，因为添加到你的模型中的层会动态构建以匹配输入数据的形状。例如，假设你编写了以下代码：
