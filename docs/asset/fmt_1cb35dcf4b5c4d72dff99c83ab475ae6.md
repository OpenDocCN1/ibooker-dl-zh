
批量归一化也被证明可以减少过拟合，因此许多现代深度学习架构根本不使用 `dropout`，完全依赖批量归一化进行正则化。与大多数深度学习原则一样，在每种情况下都没有适用的黄金法则，唯一确定最佳方法的方式是测试不同的架构，看看哪种在保留数据集上表现最好。

## 构建 CNN

您现在已经看到了三种新的 Keras 层类型：`Conv2D`、`BatchNormalization` 和 `Dropout`。让我们将这些部分组合成一个 CNN 模型，并看看它在 CIFAR-10 数据集上的表现。

### 运行此示例的代码

您可以在书籍存储库中名为 `notebooks/02_deeplearning/02_cnn/cnn.ipynb` 的 Jupyter 笔记本中运行以下示例。

我们将测试的模型架构显示在示例 2-16 中。

#### 示例 2-16\. 使用 Keras 构建 CNN 模型的代码

```py
from tensorflow.keras import layers, models

input_layer = layers.Input((32,32,3))

x = layers.Conv2D(filters = 32, kernel_size = 3
	, strides = 1, padding = 'same')(input_layer)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = 'same')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(filters = 64, kernel_size = 3, strides = 2, padding = 'same')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Flatten()(x)

x = layers.Dense(128)(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)
x = layers.Dropout(rate = 0.5)(x)

output_layer = layers.Dense(10, activation = 'softmax')(x)

model = models.Model(input_layer, output_layer)
```

我们使用四个堆叠的 `Conv2D` 层，每个后面跟一个 `BatchNormalization` 和一个 `LeakyReLU` 层。在展平结果张量后，我们通过一个大小为 128 的 `Dense` 层，再次跟一个 `BatchNormalization` 和一个 `LeakyReLU` 层。紧接着是一个用于正则化的 `Dropout` 层，网络最后是一个大小为 10 的输出 `Dense` 层。

##### 提示

使用批量归一化和激活层的顺序是个人偏好的问题。通常情况下，批量归一化层放在激活层之前，但一些成功的架构会反过来使用这些层。如果选择在激活之前使用批量归一化，可以使用缩写 *BAD*（批量归一化，激活，然后是 dropout）来记住顺序！

模型摘要显示在表 2-3 中。

**表 2-3\. CIFAR-10 的 CNN 模型摘要**

| 层（类型） | 输出形状 | 参数 # |
| --- | --- | --- |
| InputLayer | (None, 32, 32, 3) | 0 |
| Conv2D | (None, 32, 32, 32) | 896 |
| BatchNormalization | (None, 32, 32, 32) | 128 |
| LeakyReLU | (None, 32, 32, 32) | 0 |
| Conv2D | (None, 16, 16, 32) | 9,248 |
| BatchNormalization | (None, 16, 16, 32) | 128 |
| LeakyReLU | (None, 16, 16, 32) | 0 |
| Conv2D | (None, 16, 16, 64) | 18,496 |
| BatchNormalization | (None, 16, 16, 64) | 256 |
| LeakyReLU | (None, 16, 16, 64) | 0 |
| Conv2D | (None, 8, 8, 64) | 36,928 |
| BatchNormalization | (None, 8, 8, 64) | 256 |
| LeakyReLU | (None, 8, 8, 64) | 0 |
| Flatten | (None, 4096) | 0 |
| Dense | (None, 128) | 524,416 |
| BatchNormalization | (None, 128) | 512 |
| LeakyReLU | (None, 128) | 0 |
| Dropout | (None, 128) | 0 |
| Dense | (None, 10) | 1290 |
| 总参数 | 592,554 |
| 可训练参数 | 591,914 |
| 不可训练参数 | 640 |

##### 提示

在继续之前，请确保您能够手工计算每一层的输出形状和参数数量。这是一个很好的练习，可以证明您已经完全理解了每一层是如何构建的，以及它是如何与前一层连接的！不要忘记包括作为 `Conv2D` 和 `Dense` 层的一部分包含的偏置权重。

### 训练和评估 CNN

我们编译和训练模型的方式与之前完全相同，并调用 `evaluate` 方法来确定其在留存集上的准确率（图 2-16）。

![](img/gdl2_0216.png)

#### 图 2-16。CNN 性能

正如您所看到的，这个模型现在的准确率达到了 71.5%，比之前的 49.0% 有所提高。好多了！图 2-17 展示了我们新卷积模型的一些预测。

通过简单地改变模型的架构来包括卷积、批量归一化和丢弃层，已经实现了这一改进。请注意，我们新模型中的参数数量实际上比之前的模型更少，尽管层数要多得多。这表明了对模型设计进行实验和熟悉不同层类型如何利用优势的重要性。在构建生成模型时，更加重要的是要理解模型的内部工作原理，因为您最感兴趣的是网络的中间层，这些层捕捉了高级特征。

![](img/gdl2_0217.png)

#### 图 2-17。CNN 预测

## 总结

本章介绍了构建深度生成模型所需的核心深度学习概念。我们首先使用 Keras 构建了一个多层感知器（MLP），并训练模型来预测来自 CIFAR-10 数据集的给定图像的类别。然后，我们通过引入卷积、批量归一化和丢弃层来改进这个架构，创建了一个卷积神经网络（CNN）。

从本章中需要牢记的一个非常重要的观点是，深度神经网络在设计上是完全灵活的，在模型架构方面实际上没有固定的规则。有指导方针和最佳实践，但您应该随意尝试不同层和它们出现的顺序。不要感到受限于仅使用您在本书或其他地方阅读过的架构！就像一个拥有一套积木的孩子一样，您的神经网络的设计仅受您自己想象力的限制。

在下一章中，我们将看到如何使用这些积木来设计一个可以生成图像的网络。

¹ Kaiming He 等人，“用于图像识别的深度残差学习”，2015 年 12 月 10 日，[`https://arxiv.org/abs/1512.03385`](https://arxiv.org/abs/1512.03385)。

² Alex Krizhevsky，“从微小图像中学习多层特征”，2009 年 4 月 8 日，[`https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf`](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)。

³ Diederik Kingma 和 Jimmy Ba，“Adam：一种随机优化方法”，2014 年 12 月 22 日，[`https://arxiv.org/abs/1412.6980v8`](https://arxiv.org/abs/1412.6980v8)。

⁴ Samuel L. Smith 等人，“不要降低学习率，增加批量大小”，2017 年 11 月 1 日，[`https://arxiv.org/abs/1711.00489`](https://arxiv.org/abs/1711.00489)。

⁵ Vincent Dumoulin 和 Francesco Visin，“深度学习卷积算术指南”，2018 年 1 月 12 日，[`https://arxiv.org/abs/1603.07285`](https://arxiv.org/abs/1603.07285)。

⁶ Sergey Ioffe 和 Christian Szegedy，“批量归一化：通过减少内部协变量转移加速深度网络训练”，2015 年 2 月 11 日，[`https://arxiv.org/abs/1502.03167`](https://arxiv.org/abs/1502.03167)。

⁷ Hinton 等人，“通过防止特征探测器的共适应来构建网络”，2012 年 7 月 3 日，[`https://arxiv.org/abs/1207.0580`](https://arxiv.org/abs/1207.0580)。

⁸ Nitish Srivastava 等人，“Dropout：防止神经网络过拟合的简单方法”，*机器学习研究杂志* 15 (2014): 1929–1958，[`http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf`](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)。
