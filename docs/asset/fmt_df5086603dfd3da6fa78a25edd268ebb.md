

# 十二、使用 TensorFlow 自定义模型并训练

> 译者：[@SeanCheney](https://www.jianshu.com/u/130f76596b02)

目前为止，我们只是使用了 TensorFlow 的高级 API —— `tf.keras`，它的功能很强大：搭建了各种神经网络架构，包括回归、分类网络、Wide & Deep 网络、自归一化网络，使用了各种方法，包括批归一化、丢弃和学习率调度。事实上，你在实际案例中 95% 碰到的情况只需要`tf.keras`就足够了（和`tf.data`，见第 13 章）。现在来深入学习 TensorFlow 的低级 Python API。当你需要实现自定义损失函数、自定义标准、层、模型、初始化器、正则器、权重约束时，就需要低级 API 了。甚至有时需要全面控制训练过程，例如使用特殊变换或对约束梯度时。这一章就会讨论这些问题，还会学习如何使用 TensorFlow 的自动图生成特征提升自定义模型和训练算法。首先，先来快速学习下 TensorFlow。

> 笔记：TensorFlow 2.0（beta）是 2019 年六月发布的，相比前代更易使用。本书第一版使用的是 TF 1，这一版使用的是 TF 2。

## TensorFlow 速览

TensorFlow 是一个强大的数值计算库，特别适合做和微调大规模机器学习（但也可以用来做其它的重型计算）。TensorFlow 是谷歌大脑团队开发的，支持了谷歌的许多大规模服务，包括谷歌云对话、谷歌图片和谷歌搜索。TensorFlow 是 2015 年 11 月开源的，（按文章引用、公司采用、GitHub 星数）是目前最流行的深度学习库。无数的项目是用 TensorFlow 来做各种机器学习任务，包括图片分类、自然语言处理、推荐系统和时间序列预测。TensorFlow 提供的功能如下：

*   TensorFlow 的核心与 NumPy 很像，但 TensorFlow 支持 GPU；
*   TensorFlow 支持（多设备和服务器）分布式计算；
*   TensorFlow 使用了即时 JIT 编译器对计算速度和内存使用优化。编译器的工作是从 Python 函数提取出计算图，然后对计算图优化（比如剪切无用的节点），最后高效运行（比如自动并行运行独立任务）；
*   计算图可以导出为迁移形式，因此可以在一个环境中训练一个 TensorFlow 模型（比如使用 Python 或 Linux），然后在另一个环境中运行（比如在安卓设备上用 Java 运行）；
*   TensorFlow 实现了自动微分，并提供了一些高效的优化器，比如 RMSProp 和 NAdam，因此可以容易的最小化各种损失函数。

基于上面这些特点，TensorFlow 还提供了许多其他功能：最重要的是`tf.keras`，还有数据加载和预处理操作（`tf.data`，`tf.io`等等），图片处理操作（`tf.image`），信号处理操作（`tf.signal`），等等（图 12-1 总结了 TensorFlow 的 Python API）。

![](img/3d01200878f6c6d7033359da8291d199.png)

图 12-1 TensorFlow 的 Python API

> 提示：这一章会介绍 TensorFlow API 的多个包和函数，但来不及介绍全部，所以读者最好自己花点时间好好看看 API。TensorFlow 的 API 十分丰富，且文档详实。

TensorFlow 的低级操作都是用高效的 C++ 实现的。许多操作有多个实现，称为`核`：每个核对应一个具体的设备型号，比如 CPU、GPU，甚至 TPU（张量处理单元）。GPU 通过将任务分成小块，在多个 GPU 线程中并行运行，可以极大提高提高计算的速度。TPU 更快：TPU 是自定义的 ASIC 芯片，专门用来做深度学习运算的（第 19 章会讨论适合使用 GPU 和 TPU）。

TensorFlow 的架构见图 12-2。大多数时候你的代码使用高级 API 就够了（特别是`tf.keras`和`tf.data`），但如果需要更大的灵活性，就需要使用低级 Python API，来直接处理张量。TensorFlow 也支持其它语言的 API。任何情况下，甚至是跨设备和机器的情况下，TensorFlow 的执行引擎都会负责高效运行。

![](img/ed0c3d9af331e1630b0d6051a6899d01.png)

图 12-2 TensorFlow 的架构

TensorFlow 不仅可以运行在 Windows、Linux 和 macOS 上，也可以运行在移动设备上（使用 TensorFlow Lite），包括 iOS 和安卓（见第 19 章）。如果不想使用 Python API，还可以使用 C++、Java、Go 和 Swift 的 API。甚至还有 JavaScript 的实现 TensorFlow.js，它可以直接在浏览器中运行。

TensorFlow 不只有这些库。TensorFlow 处于一套可扩展的生态系统库的核心位置。首先，TensorBoard 可以用来可视化。其次，TensorFlow Extended（TFX），是谷歌推出的用来生产化的库，包括：数据确认、预处理、模型分析和服务（使用 TF Serving，见第 19 章）。谷歌的 TensorFlow Hub 上可以方便下载和复用预训练好的神经网络。你还可以从 TensorFlow 的 [model garden](https://links.jianshu.com/go?to=https%3A%2F%2Fgithub.com%2Ftensorflow%2Fmodels%2F) 获取许多神经网络架构，其中一些是预训练好的。[TensorFlow Resources](https://links.jianshu.com/go?to=https%3A%2F%2Fwww.tensorflow.org%2Fresources) 和[这个页面](https://links.jianshu.com/go?to=https%3A%2F%2Fgithub.com%2Fjtoy%2Fawesome-tensorflow)上有更多的资源。你可以在 GitHub 上找到数百个 TensorFlow 项目，无论干什么都可以方便地找到现成的代码。

> 提示：越来越多的 ML 论文都附带了实现过程，一些甚至带有预训练模型。可以在[这里](https://links.jianshu.com/go?to=https%3A%2F%2Fpaperswithcode.com%2F)找到。

最后，TensorFlow 有一支热忱满满的开发者团队，也有庞大的社区。要是想问技术问题，可以去[这里](https://links.jianshu.com/go?to=http%3A%2F%2Fstackoverflow.com%2F)，问题上打上 tensorflow 和 python 标签。还可以在 [GitHub](https://links.jianshu.com/go?to=https%3A%2F%2Fgithub.com%2Ftensorflow%2Ftensorflow) 上提 bug 和新功能。一般的讨论可以去[谷歌群组](https://links.jianshu.com/go?to=https%3A%2F%2Fgroups.google.com%2Fa%2Ftensorflow.org%2Fforum%2F)。

下面开始写代码！

## 像 NumPy 一样使用 TensorFlow

TensorFlow 的 API 是围绕张量（tensor）展开的，从一个操作流动（flow）到另一个操作，所以名字叫做 TensorFlow。张量通常是一个多维数组（就像 NumPy 的`ndarray`），但也可以是标量（即简单值，比如 42）。张量对于自定义的损失函数、标准、层等等非常重要，接下来学习如何创建和操作张量。

### 张量和运算

使用`tf.constant()`创建张量。例如，下面的张量表示的是两行三列的浮点数矩阵：

```py
>>> tf.constant([[1., 2., 3.], [4., 5., 6.]]) # matrix
<tf.Tensor: id=0, shape=(2, 3), dtype=float32, numpy=
array([[1., 2., 3.],
       [4., 5., 6.]], dtype=float32)>
>>> tf.constant(42) # 标量
<tf.Tensor: id=1, shape=(), dtype=int32, numpy=42> 
```

就像`ndarray`一样，`tf.Tensor`也有形状和数据类型（`dtype`）：

```py
>>> t = tf.constant([[1., 2., 3.], [4., 5., 6.]])
>>> t.shape
TensorShape([2, 3])
>>> t.dtype
tf.float32 
```

索引和 NumPy 中很像：

```py
>>> t[:, 1:]
<tf.Tensor: id=5, shape=(2, 2), dtype=float32, numpy=
array([[2., 3.],
       [5., 6.]], dtype=float32)>
>>> t[..., 1, tf.newaxis]
<tf.Tensor: id=15, shape=(2, 1), dtype=float32, numpy=
array([[2.],
       [5.]], dtype=float32)> 
```

最重要的，所有张量运算都可以执行：

```py
>>> t + 10
<tf.Tensor: id=18, shape=(2, 3), dtype=float32, numpy=
array([[11., 12., 13.],
       [14., 15., 16.]], dtype=float32)>
>>> tf.square(t)
<tf.Tensor: id=20, shape=(2, 3), dtype=float32, numpy=
array([[ 1.,  4.,  9.],
       [16., 25., 36.]], dtype=float32)>
>>> t @ tf.transpose(t)
<tf.Tensor: id=24, shape=(2, 2), dtype=float32, numpy=
array([[14., 32.],
       [32., 77.]], dtype=float32)> 
```

可以看到，`t + 10`等同于调用`tf.add(t, 10)`，`-`和`*`也支持。`@`运算符是在 Python3.5 中出现的，用于矩阵乘法，等同于调用函数`tf.matmul()`。

可以在 tf 中找到所有基本的数学运算（`tf.add()`、`tf.multiply()`、`tf.square()`、`tf.exp()`、`tf.sqrt()`），以及 NumPy 中的大部分运算（比如`tf.reshape()`、`tf.squeeze()`、`tf.tile()`）。一些 tf 中的函数与 NumPy 中不同，例如，`tf.reduce_mean()`、`tf.reduce_sum()`、`tf.reduce_max()`、`tf.math.log()`等同于`np.mean()`、`np.sum()`、`np.max()`和`np.log()`。当函数名不同时，通常都是有原因的。例如，TensorFlow 中必须使用`tf.transpose(t)`，不能像 NumPy 中那样使用`t.T`。原因是函数`tf.transpose(t)`所做的和 NumPy 的属性`T`并不完全相同：在 TensorFlow 中，是使用转置数据的复制来生成张量的，而在 NumPy 中，`t.T`是数据的转置视图。相似的，`tf.reduce_sum()`操作之所以这么命名，是因为它的 GPU 核（即 GPU 实现）所采用的归约算法不能保证元素相加的顺序，因为 32 位的浮点数精度有限，每次调用的结果可能会有细微的不同。`tf.reduce_mean()`也是这样（`tf.reduce_max()`结果是确定的）。

> 笔记：许多函数和类都有假名。比如，`tf.add()`和`tf.math.add()`是相同的。这可以让 TensorFlow 对于最常用的操作有简洁的名字，同时包可以有序安置。

> Keras 的低级 API

> Keras API 有自己的低级 API，位于`keras.backend`，包括：函数`square()`、`exp()`、`sqrt()`。在`tf.keras`中，这些函数通常通常只是调用对应的 TensorFlow 操作。如果你想写一些可以迁移到其它 Keras 实现上，就应该使用这些 Keras 函数。但是这些函数不多，所以这本书里就直接使用 TensorFlow 的运算了。下面是一个简单的使用了`keras.backend`的例子，简记为`k`：

> ```py
> >>> from tensorflow import keras
> >>> K = keras.backend
> >>> K.square(K.transpose(t)) + 10
> <tf.Tensor: id=39, shape=(3, 2), dtype=float32, numpy=
> array([[11., 26.],
>        [14., 35.],
>        [19., 46.]], dtype=float32)> 
> ```

### 张量和 NumPy

张量和 NumPy 融合地非常好：使用 NumPy 数组可以创建张量，张量也可以创建 NumPy 数组。可以在 NumPy 数组上运行 TensorFlow 运算，也可以在张量上运行 NumPy 运算：

```py
>>> a = np.array([2., 4., 5.])
>>> tf.constant(a)
<tf.Tensor: id=111, shape=(3,), dtype=float64, numpy=array([2., 4., 5.])>
>>> t.numpy() # 或 np.array(t)
array([[1., 2., 3.],
       [4., 5., 6.]], dtype=float32)
>>> tf.square(a)
<tf.Tensor: id=116, shape=(3,), dtype=float64, numpy=array([4., 16., 25.])>
>>> np.square(t)
array([[ 1.,  4.,  9.],
       [16., 25., 36.]], dtype=float32) 
```

> 警告：NumPy 默认使用 64 位精度，TensorFlow 默认用 32 位精度。这是因为 32 位精度通常对于神经网络就足够了，另外运行地更快，使用的内存更少。因此当你用 NumPy 数组创建张量时，一定要设置`dtype=tf.float32`。

### 类型转换

类型转换对性能的影响非常大，并且如果类型转换是自动完成的，不容易被注意到。为了避免这样，TensorFlow 不会自动做任何类型转换：只是如果用不兼容的类型执行了张量运算，TensorFlow 就会报异常。例如，不能用浮点型张量与整数型张量相加，也不能将 32 位张量与 64 位张量相加：

```py
>>> tf.constant(2.) + tf.constant(40)
Traceback[...]InvalidArgumentError[...]expected to be a float[...]
>>> tf.constant(2.) + tf.constant(40., dtype=tf.float64)
Traceback[...]InvalidArgumentError[...]expected to be a double[...] 
```

这点可能一开始有点恼人，但是有其存在的理由。如果真的需要转换类型，可以使用`tf.cast()`：

```py
>>> t2 = tf.constant(40., dtype=tf.float64)
>>> tf.constant(2.0) + tf.cast(t2, tf.float32)
<tf.Tensor: id=136, shape=(), dtype=float32, numpy=42.0> 
```

### 变量

到目前为止看到的`tf.Tensor`值都是不能修改的。意味着不能使用常规张量实现神经网络的权重，因为权重必须要能被反向传播调整。另外，其它的参数也需要随着时间调整（比如，动量优化器要跟踪过去的梯度）。此时需要的是`tf.Variable`：

```py
>>> v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
>>> v
<tf.Variable 'Variable:0' shape=(2, 3) dtype=float32, numpy=
array([[1., 2., 3.],
       [4., 5., 6.]], dtype=float32)> 
```

`tf.Variable`和`tf.Tensor`很像：可以运行同样的运算，可以配合 NumPy 使用，也要注意类型。可以使用`assign()`方法对其就地修改（或`assign_add()`、`assign_sub()`）。使用切片的`assign()`方法可以修改独立的切片（直接赋值行不通），或使用`scatter_update()`、`scatter_nd_update()`方法。

