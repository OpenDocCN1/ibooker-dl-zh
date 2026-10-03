
```py
img = Conv2D(32, (3, 3), activation='relu', padding='same')(inp_img)
img = MaxPooling2D((2, 2), padding='same')(img)
img = Conv2D(32, (3, 3), activation='relu', padding='same')(img)
img = UpSampling2D((2, 2))(img)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(img)
```

我们声明功能模型格式，传递输入和输出：

```py
autoencoder = Model(inp_img, decoded)
```

接下来，我们编译模型，定义损失函数和优化器——在这种情况下，我们使用 Adagrad 优化器（只是为了展示另一个例子！）。对于图像的去噪，我们希望我们的损失能够捕捉解码图像与原始、去噪前图像之间的差异。为此，我们使用二元交叉熵损失，将每个解码像素与其对应的原始像素进行比较（现在在[0,1]之间）：

```py
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
```

模型定义后，我们用 10 个训练周期来拟合它：

```py
tensorboard=TensorBoard(log_dir='some_path',histogram_freq=0,write_graph=True,write_images=True)
model_saver=ModelCheckpoint(filepath='some_path',verbose=0,period=2)
autoencoder.fit(x_train_n,x_train,epochs=10,batch_size=64,shuffle=True,validation_data=(x_test_n,x_test),callbacks=[tensorboard,model_saver])
```

希望模型能够捕捉一些内部结构，然后将其推广到其他嘈杂图像，并因此对其进行去噪。

我们将测试集用作每个周期结束时损失评估的验证数据（模型不会在此数据上进行训练），还用于在 TensorBoard 中进行可视化。除了 TensorBoard 回调，我们还添加了一个模型保存回调，并设置为每两个周期保存一次权重。

稍后，当我们希望加载我们的权重时，我们需要重建网络，然后使用`Model.load_weights()`方法，将我们的模型作为第一个参数，将保存的权重文件路径作为第二个参数（有关在第十章中保存模型的更多信息）：

```py
inp_img=Input(shape=(32,32,3))
img=Conv2D(32,(3,3),activation='relu',padding='same')(inp_img)
img=MaxPooling2D((2,2),padding='same')(img)
img=Conv2D(32,(3,3),activation='relu',padding='same')(img)
img=UpSampling2D((2,2))(img)
decoded=Conv2D(3,(3,3),activation='sigmoid',padding='same')(img)
autoencoder=Model(inp_img,decoded)
Model.load_weights(autoencoder,'some_path')
```

## h5py 要求

为了保存模型，需要安装`h5py`包。这个包主要用于存储大量数据并从 NumPy 中操作数据。您可以使用`pip`安装它：

```py
pip install h5py
```

图 7-7 显示了我们选择类别的去噪测试图像，不同训练周期的结果。

![](img/letf_0707.png)

###### 图 7-7。自动编码之前的嘈杂 CIFAR10 图像（上排）和自动编码之后的图像（下排）。底部 4 行显示了增加训练周期后的结果。

Keras 还有一堆预训练模型可供下载，如*inception*、*vgg*和*resnet*。在本章的下一部分和最后一部分中，我们将讨论这些模型，并展示如何下载和使用 TF-Slim 扩展来进行分类的预训练 VGG 模型的示例。

## TF-Slim 的预训练模型

在本章的这一部分，我们将介绍这里要涵盖的最后一个抽象概念，TF-Slim。TF-Slim 通过提供简化的语法来定义 TensorFlow 中的卷积神经网络而脱颖而出——它的抽象使得以一种干净、简洁的方式构建复杂网络变得容易。与 Keras 一样，它还提供了各种[预训练的 CNN 模型](http://bit.ly/2sZt5lE?)供下载和使用。

我们通过学习 TF-Slim 的一些一般特性和优点来开始本节，说明为什么它是构建 CNN 的绝佳工具。在本节的第二部分中，我们将演示如何下载和部署一个预训练模型（VGG）进行图像分类。

### TF-Slim

TF-Slim 是 TensorFlow 的一个相对较新的轻量级扩展，类似于其他抽象，它允许我们快速直观地定义和训练复杂模型。TF-Slim 不需要任何安装，因为它已经与 TensorFlow 合并。

这个扩展主要是关于卷积神经网络。CNN 以混乱的样板代码而闻名。TF-Slim 的设计目标是优化非常复杂的 CNN 模型的创建，使其能够通过使用高级层、变量抽象和参数作用域进行优雅编写、易于解释和调试。

除了让我们能够创建和训练自己的模型之外，TF-Slim 还提供了可以轻松下载、阅读和使用的预训练网络：VGG、AlexNet、Inception 等。

我们从简要描述 TF-Slim 的一些抽象特性开始这一部分。然后我们将重点放在如何下载和使用预训练模型上，以 VGG 图像分类模型为例进行演示。

### 使用 TF-Slim 创建 CNN 模型

使用 TF-Slim，我们可以通过定义初始化、正则化和设备来轻松创建一个变量。例如，在这里，我们定义了从截断正态分布初始化的权重，使用 L2 正则化，并放置在 CPU 上（我们将在第九章讨论跨设备分配模型部分）：

```py
import tensorflow as tf
from tensorflow.contrib import slim

W = slim.variable('w',shape=[7, 7, 3 , 3],
initializer=tf.truncated_normal_initializer(stddev=0.1),
regularizer=slim.l2_regularizer(0.07),
device='/CPU:0')
```

与本章中看到的其他抽象一样，TF-Slim 可以减少大量样板代码和冗余复制。与 Keras 或 TFLearn 一样，我们可以在抽象级别定义一个层操作，包括卷积操作、权重初始化、正则化、激活函数等，都可以在一个命令中完成：

```py
net = slim.conv2d(inputs, 64, [11, 11], 4, padding='SAME',
weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
weights_regularizer=slim.l2_regularizer(0.0007), scope='conv1')
```

TF-Slim 甚至将其优雅性扩展到了更远的地方，通过使用`repeat`、`stack`和`arg_scope`命令提供了一种紧凑地复制层的方法。

`repeat`可以避免我们反复复制粘贴相同的行，例如，与其有这种冗余复制：

```py
net = slim.conv2d(net, 128, [3, 3], scope='con1_1')
net = slim.conv2d(net, 128, [3, 3], scope='con1_2')
net = slim.conv2d(net, 128, [3, 3], scope='con1_3')
net = slim.conv2d(net, 128, [3, 3], scope='con1_4')
net = slim.conv2d(net, 128, [3, 3], scope='con1_5')
```

我们可以这样输入：

```py
net = slim.repeat(net, 5, slim.conv2d, 128, [3, 3], scope='con1')
```

但这仅适用于具有相同大小的层的情况。当这种情况不成立时，我们可以使用`stack`命令，允许我们连接不同形状的层。因此，与其这样：

```py
net = slim.conv2d(net, 64, [3, 3], scope='con1_1')
net = slim.conv2d(net, 64, [1, 1], scope='con1_2')
net = slim.conv2d(net, 128, [3, 3], scope='con1_3')
net = slim.conv2d(net, 128, [1, 1], scope='con1_4')
net = slim.conv2d(net, 256, [3, 3], scope='con1_5')
```

我们可以这样写：

```py
slim.stack(net, slim.conv2d, [(64, [3, 3]), (64, [1, 1]), 
                              (128, [3, 3]), (128, [1, 1]),
                              (256, [3, 3])], scope='con')
```

最后，我们还有一个称为`arg_scope`的作用域机制，允许用户将一组共享参数传递给同一作用域中定义的每个操作。例如，假设我们有四个具有相同激活函数、初始化、正则化和填充的层。我们可以简单地使用`slim.arg_scope`命令，指定共享参数，如下面的代码所示：

```py
with slim.arg_scope([slim.conv2d], 
                padding='VALID',
                activation_fn=tf.nn.relu,
weights_initializer=tf.truncated_normal_initializer(stddev=0.02)
weights_regularizer=slim.l2_regularizer(0.0007)):
net = slim.conv2d(inputs, 64, [11, 11], scope='con1')
net = slim.conv2d(net, 128, [11, 11], padding='VALID', scope='con2')
net = slim.conv2d(net, 256, [11, 11], scope='con3')
net = slim.conv2d(net, 256, [11, 11], scope='con4')
```

`arg_scope`命令中的各个参数仍然可以被覆盖，我们也可以将一个`arg_scope`嵌套在另一个中。

在这些示例中，我们使用了`conv2d()`：然而，TF-Slim 还有许多其他用于构建神经网络的标准方法。表 7-4 列出了一些可用选项。要查看完整列表，请参阅[文档](http://bit.ly/2txy6PN?)。

表 7-4。TF-Slim 中可用的层类型

| Layer | TF-Slim |
| --- | --- |
| BiasAdd | `slim.bias_add()` |
| BatchNorm | `slim.batch_norm()` |
| Conv2d | `slim.conv2d()` |
| Conv2dInPlane | `slim.conv2d_in_plane()` |
| Conv2dTranspose (Deconv) | `slim.conv2d_transpose()` |
| FullyConnected | `slim.fully_connected()` |
| AvgPool2D | `slim.avg_pool2d()` |
| Dropout | `slim.dropout()` |
| Flatten | `slim.flatten()` |
| MaxPool2D | `slim.max_pool2d()` |
| OneHotEncoding | `slim.one_hot_encoding()` |
| SeparableConv2 | `slim.separable_conv2d()` |
| UnitNorm | `slim.unit_norm` |

为了说明 TF-Slim 在创建复杂 CNN 时有多方便，我们将构建 Karen Simonyan 和 Andrew Zisserman 在 2014 年提出的 VGG 模型（有关更多信息，请参见即将发布的说明）。VGG 是一个很好的例子，说明了如何使用 TF-Slim 紧凑地创建具有许多层的模型。在这里，我们构建 16 层版本：13 个卷积层加上 3 个全连接层。

创建它时，我们利用了我们刚提到的两个特性：

1. 我们使用`arg_scope`功能，因为所有卷积层具有相同的激活函数和相同的正则化和初始化。
2. 许多层都是相同的副本，因此我们还利用了`repeat`命令。

结果非常引人注目——整个模型仅用 16 行代码定义：

```py
with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                weights_regularizer=slim.l2_regularizer(0.0005)):
    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='con1')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='con2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='con3')
    net = slim.max_pool2d(net, [2, 2], scope='pool3')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='con4')
    net = slim.max_pool2d(net, [2, 2], scope='pool4')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='con5')
    net = slim.max_pool2d(net, [2, 2], scope='pool5')
    net = slim.fully_connected(net, 4096, scope='fc6')
    net = slim.dropout(net, 0.5, scope='dropout6')
    net = slim.fully_connected(net, 4096, scope='fc7')
    net = slim.dropout(net, 0.5, scope='dropout7')
    net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')
```

# VGG 和 ImageNet 挑战

[ImageNet 项目](http://www.image-net.org)是一个大型图像数据库，旨在研究视觉对象识别。截至 2016 年，它包含超过 1000 万个手工注释的图像。

自 2010 年以来，每年都会举行一场名为 ImageNet 大规模视觉识别挑战（ILSVRC）的比赛，研究团队试图在 ImageNet 收集的子集中自动分类、检测和定位对象和场景。在 2012 年的挑战中，当时由 Alex Krizhevsky 创建的深度卷积神经网络 AlexNet 取得了令人瞩目的进展，成功将前 5 名（前 5 个选择的类别）的分类错误率降至仅 15.4%，以较大的优势赢得了比赛。

在接下来的几年里，错误率不断下降，从 2013 年的 ZFNet 的 14.8%，到 2014 年的 GoogLeNet（引入 Inception 模块）的 6.7%，再到 2015 年的 ResNet 的 3.6%。视觉几何组（VGG）是 2014 年比赛中的另一个 CNN 竞争对手，也取得了令人印象深刻的低错误率（7.3%）。许多人更喜欢 VGG 而不是 GoogLeNet，因为它具有更好、更简单的架构。

在 VGG 中，唯一使用的空间维度是非常小的 3×3 滤波器，步幅为 1，以及 2×2 最大池化，步幅再次为 1。它的优越性是通过使用的层数来实现的，介于 16 和 19 之间。

### 下载和使用预训练模型

接下来我们将演示如何下载和部署预训练的 VGG 模型。

首先，我们需要克隆存放实际模型的存储库，方法是运行：

```py
git clone https://github.com/tensorflow/models
```

现在我们在计算机上拥有建模所需的脚本，并且可以通过设置路径来使用它们：

```py
import sys
sys.path.append("some_path + models/slim")
```
