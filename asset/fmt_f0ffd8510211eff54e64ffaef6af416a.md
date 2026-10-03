
| Inc_4a | 192 | 96 | 208 | 16 | 48 | 64 |
| Inc_4b | 160 | 112 | 224 | 24 | 64 | 64 |
| Inc_4c | 128 | 128 | 256 | 24 | 64 | 64 |
| Inc_4d | 112 | 144 | 288 | 32 | 64 | 64 |
| Inc_4e | 256 | 160 | 320 | 32 | 128 | 128 |
| Inc_5a | 256 | 160 | 320 | 32 | 128 | 128 |
| Inc_5b | 384 | 192 | 384 | 48 | 128 | 128 |

层的定义将与您已经看到的相当相似。然而，我们定义模型和编译模型的方式对于一些人来说可能是新的。正如我们讨论的那样，Inception 网络是一个多输出模型。您可以通过传递输出列表而不是单个输出来定义具有多个输出的 Keras 模型：

```py
model = Model(inputs=inp, outputs=[out_main, aux_out1, aux_out2])
```

在编译模型时，您可以将损失定义为一个字符串列表。如果定义一个字符串，那么该损失将用于所有输出。我们使用分类交叉熵损失（对于最终输出层和辅助输出层）和优化器 `adam` 来编译模型，`adam` 是一种广泛用于优化模型的先进优化器，它可以随着模型训练适当地调整学习率。此外，我们将检查模型的准确性：

```py
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

通过定义 `inception_v1()` 函数，您可以创建一个模型如下：

```py
model = inception_v1()
```

让我们花点时间回顾一下我们迄今为止所取得的成就。我们已经下载了数据，剖析了数据，并分析了数据以了解具体情况。然后，我们使用 `tensorflow.keras.preprocessing.image.ImageDataGenerator` 创建了一个图像数据管道。我们将数据分成了三部分：训练、验证和测试。最后，我们定义了我们的模型，这是一个被称为 Inception 网络的最先进的图像分类器。现在我们将看看多年来出现的其他 Inception 模型。

### 6.3.4 其他 Inception 模型

我们成功地实现了一个 Inception 网络模型，它涵盖了我们需要理解其他 Inception 模型的大部分基础知识。自 v1 模型以来已经有五个更多的 Inception 网络。让我们简要地了解一下 Inception 网络的演变。

#### Inception v1

我们已经深入讨论了 Inception 网络 v1。Inception 网络 v1 中引入的最大突破如下：

+ Inception 块的概念允许 CNN 在模型的相同深度拥有不同的感受野（即，卷积核大小）。这鼓励模型的稀疏性，导致参数更少，过拟合的机会也更少。
+ 在 Inception 模型的 20 层中，如果不小心，现代 GPU 的内存可能会被耗尽。Inception 网络通过使用 1 × 1 卷积层来减少输出通道深度，以缓解这个问题。
+ 网络越深，在模型训练过程中，梯度不稳定的可能性就越大。这是因为梯度必须经过很长的路径（从顶部到最底部），这可能导致梯度不稳定。在网络中间引入辅助输出层作为正则化器可以缓解这个问题，从而导致梯度稳定。

#### Inception v2

Inception net v2 出现在 Inception net v1 发布后不久（"Rethinking the Inception Architecture for Computer Vision"，[`arxiv.org/pdf/1512.00567.pdf`](https://arxiv.org/pdf/1512.00567.pdf)）。这个模型的主要贡献如下。

当层的容量（即参数）不足以学习输入的良好表示时，就会发生表示瓶颈。如果在深度递减时层的大小减小得太快，这种情况可能会发生。Inception v2 重新设计了架构，以确保模型中不存在表示瓶颈。这主要通过改变层大小而保持其他细节不变来实现。

进一步减少网络参数以减少过拟合被强化。这是通过用 3 × 3 卷积（也称为因式分解大卷积层）替换更高阶的卷积（例如，5 × 5 和 7 × 7）来实现的。这是如何可能的？让我为您说明一下（图 6.16）。

![06-16](img/06-16.png)

图 6.16 一个 5 × 5 卷积层（左）和两个 3 × 3 卷积层（右）

将 5 × 5 卷积表示为两个更小的 3 × 3 卷积操作，我们可以减少 28% 的参数。图 6.17 对比了 Inception v1 块和 Inception v2 块。

![06-17](img/06-17.png)

图 6.17 Inception net v1 中的 Inception 块（左）与 Inception net v2 中的 Inception 块（右）

TensorFlow 代码如下：

```py
# 1x1 layer
out1 = Conv2D(64, (1,1), strides=(1,1), activation='relu', padding='same')(inp)
# 1x1 followed by 3x3
out2_1 = Conv2D(
    96, (1,1), strides=(1,1), activation='relu', padding='same'
)(inp)
out2_2 = Conv2D(
    128, (3,3), strides=(1,1), activation='relu', padding='same'
)(out2_1)

# 1x1 followed by 5x5
# Here 5x5 is represented by two 3x3 convolution layers
out3_1 = Conv2D(
    16, (1,1), strides=(1,1), activation='relu', padding='same'
)(inp)
out3_2 = Conv2D(
    32, (3,3), strides=(1,1), activation='relu', padding='same'
)(out3_1)
out3_3 = Conv2D(
    32, (3,3), strides=(1,1), activation='relu', padding='same'
)(out3_2)

# 3x3 (pool) followed by 1x1
out4_1 = MaxPool2D((3,3), strides=(1,1), padding='same')(inp)
out4_2 = Conv2D(
    32, (1,1), strides=(1,1), activation='relu', padding='same'
)(out4_1)

out = Concatenate(axis=-1)([out1, out2_2, out3_3, out4_2])
```

但我们不必止步于此。我们可以将任何 n × n 卷积操作因式分解为两个 1 × n 和 n × 1 卷积层，例如，对于 3 × 3 卷积层，可以减少 33% 的参数（图 6.18）。经验上发现，将 n × n 操作分解为两个 1 × n 和 n × 1 操作仅在更高层中有用。您可以参考论文以了解这些类型的因式分解何时以及在哪里使用。

![06-18](img/06-18.png)

图 6.18 一个 3 × 3 卷积层（左）和一个 3 × 1 和一个 1 × 3 卷积层（右）

#### Inception v3

Inception v3 是在同一篇论文中引入的 Inception net v2。与 v2 不同的主要贡献是使用批量标准化层。批量标准化（"Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"，[`proceedings.mlr.press/v37/ioffe15.pdf`](http://proceedings.mlr.press/v37/ioffe15.pdf)）通过减去给定层 *x* 的平均值（*E*(*x*)）和标准差（√(*Var*(*x*))）来标准化给定层的输出：

![06_18a](img/06_18a.png)

这个过程帮助网络稳定其输出值，而不让它们变得太大或太小。接下来，它有两个可训练参数，*γ* 和 *β*，用于缩放和偏移归一化输出：

y = *γ* *x̂* + *β*

这样，网络就可以通过学习 *γ* 和 *β* 的最佳化来学习归一化的自己变化，以防 *x̂* 不是最佳的归一化配置。此时，你只需要理解批归一化归一化给定层的输出。我们将在下一章更详细地讨论批归一化在 Inception 网模型中的使用方式。

#### Inception v4

Inception-v4 是在论文"Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning"（[`mng.bz/J28P`](http://mng.bz/J28P)）中引入的，并没有引入任何新的概念，而是专注于使模型更简单，而不会牺牲性能。主要是，v4 简化了网络的干部和其他元素。由于这主要是为了更好地调整网络的超参数以获得更好的性能，而不是引入任何新的概念，所以我们不会深入研究这个模型。

#### Inception-ResNet v1 和 Inception-ResNet v2

Inception-ResNet v1 和 v2 在同一篇论文中被介绍，并且是其主要贡献。Inception-ResNet 简化了该模型中使用的 Inception 块，并删除了一些杂乱的细节。更重要的是，它引入了残差连接。*残差连接*（或 *跳过连接*）是由 Kaiming He 等人在题为"Deep Residual Learning for Image Recognition"的论文中介绍的（[`arxiv.org/pdf/1512.03385.pdf`](https://arxiv.org/pdf/1512.03385.pdf)）。这是一个简单而优雅的概念，却非常强大，它已经成为许多不同领域中表现最佳的模型的原因之一。

如图 6.19 所示，残差连接简单地将较低层（靠近输入）的输入添加到较高层（远离输入）的输入中。这样就在较低层和较高层之间创建了一条捷径，实质上是在结果输出与较低层之间创建了另一个捷径。我们不会在这里深入讨论太多细节，因为我们将在下一章详细讨论 Inception-ResNet 模型。接下来，我们将训练我们刚刚定义的模型，使用我们准备的图像数据。

![06-19](img/06-19.png)

图 6.19 残差连接是如何引入到网络中的。这是一个简单的操作，其中你将一层的较低输出（更接近输入）添加到一层的较高输出（远离输入）中。跳过连接可以设计成跳过您喜欢的任意数量的层。图还突出了梯度的流动；您可以看到跳过连接如何允许梯度绕过某些层并传播到较低层。

#### 练习 3

作为研究的一部分，您正在测试一种称为 *poolception* 的新技术。概念上类似于 Inception 块，poolception 具有三个并行的池化层，具有以下规格：

+ 一个带有步长为 2 和相同填充的 3×3 最大池化层
+ 一个带有步长为 2 和相同填充的 5×5 最大池化层
+ 一个带有步长为 2 和相同填充的 3×3 平均池化层

最后，这些层的输出在通道轴上串联起来。您能将此实现为一个名为 `poolception` 的 Python 函数，该函数以前一层的输入 `x` 作为参数吗？

## 6.4 训练模型和评估性能

很棒！您已经定义了一个在类似（更大）数据集上表现良好的最先进模型架构之一。您的下一个任务是训练此模型并分析其性能。

模型训练是一个必不可少的步骤，如果您需要一个性能良好的模型，一旦到了使用它的时候。训练模型会优化（即更改）模型的参数，使其能够在给定输入时产生正确的预测。通常，模型训练是在多个时期进行的，其中每个时期可以包含数千个迭代。这个过程可能需要数小时甚至数周，这取决于数据集的大小和模型。正如我们已经讨论过的，由于其众所周知的内存需求，深度神经网络以小批量方式消耗数据。优化模型与单个数据批次的步骤称为 *迭代*。当您以这种批量方式遍历整个数据集时，它被称为 *时期*。

最后，一旦训练完成，您需要确保模型在未见过的数据上表现良好。这些未见过的数据在训练过程中不得与模型发生任何交互。深度学习网络最常见的评估指标是准确率。因此，我们测量测试准确率以确保模型的稳健性。

为了训练模型，让我们首先定义一个函数，该函数计算每个时期的步数或迭代次数，给定数据集的大小和批量大小。对于每个时期都运行预定义数量的步骤总是一个好主意。有些情况下，Keras 无法确定步骤的数量，这种情况下，它可能会使模型运行，直到您停止它：

```py
def get_steps_per_epoch(n_data, batch_size):
    if n_data%batch_size==0:
        return int(n_data/batch_size)
    else:
        return int(n_data*1.0/batch_size)+1
```

这是一个非常简单的计算。每个时期的步数是数据点数（`n_data`）除以批量大小（`batch_size`）。如果 `n_data` 不可被 `batch_size` 整除，则需要将返回值加 1，以确保不会丢失任何数据。现在让我们在下面的列表中训练模型。

**列表 6.7 训练 Inception 网络**

```py
from tensorflow.keras.callbacks import CSVLogger
import time
import os

if not os.path.exists('eval'):
    os.mkdir('eval')                                               ❶

csv_logger = CSVLogger(os.path.join('eval','1_eval_base.log'))     ❷

history = model.fit(
    x=train_gen_aux,                                               ❸
    validation_data=valid_gen_aux,                                 ❸
    steps_per_epoch=get_steps_per_epoch(0.9*500*200,batch_size),   ❸
    validation_steps=get_steps_per_epoch(0.1*500*200,batch_size),  ❸
    epochs=50, 
    callbacks=[csv_logger]                                         ❸
)                                                                  ❸

if not os.path.exists('models'):
    os.mkdir("models")
model.save(os.path.join('models', 'inception_v1_base.h5'))         ❹
```

❶ 创建一个名为 `eval` 的目录来存储性能结果

❷ 这是一个您传递给 `fit()` 函数的 Keras 回调函数。它将指标数据写入 CSV 文件。

❸ 通过拟合模型，您可以看到我们正在将训练和验证数据生成器传递给函数。

❹ 将模型保存到磁盘上，以便在需要时重新加载

训练模型时，通常会遵循以下步骤：

+ 为一定数量的周期训练模型。
+ 每个训练周期结束时，在验证数据集上测量性能。
+ 所有训练周期结束后，对测试集的性能进行测量。

当在代码中调用 `model.fit()` 时，它会处理前两个步骤。我们将更详细地查看 `model.fit()` 函数。我们向函数传递以下参数：

+ `X` — 将训练数据生成器传递给模型，其中包含输入（`x`）和目标（`y`）。
+ `y` — 通常接收目标值。在这里，我们不指定 `y`，因为 `x` 已经包含了目标。
+ `validation_data` — 接收验证数据生成器。
+ `steps_per_epoch` — 训练中每个周期的步数（迭代次数）。
+ `validation_steps` — 验证中每个周期的步数（迭代次数）。
+ `epochs` — 周期数。
+ 回调函数 — 需要传递给模型的任何回调函数（有关回调函数的完整列表，请访问[`mng.bz/woEW`](http://mng.bz/woEW)）。

在训练模型后，您应该会得到以下结果之类的内容：
