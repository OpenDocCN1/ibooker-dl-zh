
现在，让我们看看这个操作涉及的计算成本，并将其与我们之前应用降维层时得到的 1.63 亿次乘法进行比较：

计算量

= 1 × 1 卷积层中的操作 + 5 × 5 卷积层中的操作

= (32 × 32 × 200) 乘以 (1 × 1 × 16 + 32 × 32 × 16) 乘以 (5 × 5 × 32)

= 320 万 + 1310 万

这个操作中的总乘法次数为 1630 万次，是未使用降维层时计算的 1.63 亿次的十分之一。

## 1 × 1 卷积层

`1 × 1`卷积层的思想是它保留了输入体积的空间维度（高度和宽度），但改变了体积的通道数（深度）：

![](img/5-unnumb-5Key.png)

`1 × 1`卷积层保留了空间维度，但改变了深度。

`1 × 1`卷积层也被称为瓶颈层，因为瓶颈是瓶子中最小的一部分，降维层减少了网络的维度，使其看起来像瓶颈：

![](img/5-unnumb-6K.png)

`1 × 1`卷积层被称为瓶颈层。

### 降维对网络性能的影响

你可能会想知道，如此大幅度地缩小表示大小是否会损害神经网络的性能。Szegedy 等人进行了实验，发现只要适度地实现降维层，就可以显著缩小表示大小而不会损害性能——并且节省大量计算。

现在，让我们将降维层付诸实践，并构建一个新的具有降维功能的 inception 模块。为此，我们将保持从原始表示中连接四个层的相同概念。我们将在 `3 × 3` 和 `5 × 5` 卷积层之前添加一个 `1 × 1` 卷积降维层，以降低它们的计算成本。我们还将添加一个 `1 × 1` 卷积层在 `3 × 3` 最大池化层之后，因为池化层不会减少其输入的深度。因此，在我们进行连接之前，我们需要将降维层应用于它们的输出（图 5.13）。

![](img/5-13.png)

图 5.13 使用降维构建 inception 模块

我们在更大的卷积层之前添加了降维，以便在后续阶段计算复杂度不受控制地激增之前，显著增加每个阶段的单元数量。此外，设计遵循了实用的直觉，即视觉信息应该在各种尺度上处理，然后汇总，以便下一阶段可以同时从不同尺度抽象特征。

### Inception 模块的回顾

总结来说，如果您正在构建神经网络的一层，并且不想决定在卷积层中使用什么滤波器大小或何时添加池化层，Inception 模块允许您使用所有这些，并将所有输出的深度连接起来。这被称为 Inception 模块的朴素表示。

然后，我们遇到了使用大滤波器带来的计算成本问题。在这里，我们使用了一个称为 reduce 层的 `1 × 1` 卷积层，它显著降低了计算成本。我们在 `3 × 3` 和 `5 × 5` 卷积层之前以及最大池化层之后添加 reduce 层，以创建一个具有降维的 Inception 模块。

## 5.5.4 Inception 架构

现在我们已经了解了 Inception 模块的组成部分，我们准备构建 Inception 网络架构。我们使用 Inception 模块的降维表示，将 Inception 模块堆叠在一起，并在它们之间添加一个 `3 × 3` 池化层以进行下采样，如图 5.14 所示。

![](img/5-14.png)

图 5.14 我们通过将 Inception 模块堆叠在一起来构建 Inception 网络。

我们可以堆叠任意数量的 Inception 模块来构建一个非常深的卷积网络。在原始论文中，该团队构建了一个特定的 Inception 模块实例，并将其称为 GoogLeNet。他们在 2014 年 ILSVRC 竞赛的提交中使用了这个网络。GoogLeNet 架构如图 5.15 所示。

![](img/5-15.png)

图 5.15 完整的 GoogLeNet 模型由三部分组成：第一部分具有类似于 AlexNet 和 LeNet 的经典 CNN 架构，第二部分是一堆 Inception 模块和池化层，第三部分是传统的全连接分类器。

如您所见，GoogLeNet 使用了一堆总共九个 Inception 模块，并在每隔几个块中使用最大池化层来降低维度。为了简化这个实现，我们将 GoogLeNet 架构分解为三个部分：

+ A 部分——与 AlexNet 和 LeNet 架构相同；包含一系列卷积和池化层。
+ B 部分——包含九个 Inception 模块堆叠如下：两个 Inception 模块 + 池化层 + 五个 Inception 模块 + 池化层 + 五个 Inception 模块。
+ C 部分——网络的分类器部分，包括全连接和 softmax 层。

## 5.5.5 Keras 中的 GoogLeNet

现在，让我们在 Keras 中实现 GoogLeNet 架构（图 5.16）。注意，inception 模块将前一个模块的特征作为输入，通过四个路径传递，将所有四个路径的输出深度连接起来，然后将连接后的输出传递到下一个模块。这四个路径如下：

+ `1 × 1`卷积层
+ `1 × 1`卷积层 + `3 × 3`卷积层
+ `1 × 1`卷积层 + `5 × 5`卷积层
+ `3 × 3`池化层 + `1 × 1`卷积层

![](img/5-16.png)

图 5.16 GoogLeNet 的 inception 模块

首先，我们将构建`inception_module`函数。它接受每个卷积层的滤波器数量作为参数，并返回连接后的输出：

```
def inception_module(x, filters_1 × 1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce,
                     filters_5x5, filters_pool_proj, name=None):

conv_1x1 = Conv2D(filters_1x1, kernel_size=(1, 1), padding='same', activation='relu',
                  kernel_initializer=kernel_init, bias_initializer=bias_init)(x) ❶

      # 3 × 3 route = 1 × 1 CONV + 3 × 3 CONV 
pre_conv_3x3 = Conv2D(filters_3x3_reduce, kernel_size=(1, 1), padding='same',
                  activation='relu', kernel_initializer=kernel_init, 
                  bias_initializer=bias_init)(x)
conv_3x3 = Conv2D(filters_3x3, kernel_size=(3, 3), padding='same', activation='relu', 
                  kernel_initializer=kernel_init, 
                  bias_initializer=bias_init)(pre_conv_3x3)
      # 5 × 5 route = 1 × 1 CONV + 5 × 5 CONV 
pre_conv_5x5 = Conv2D(filters_5x5_reduce, kernel_size=(1, 1), padding='same',
                  activation='relu', kernel_initializer=kernel_init, 
                  bias_initializer=bias_init)(x)
conv_5x5 = Conv2D(filters_5x5, kernel_size=(5, 5), padding='same', activation='relu', 
                  kernel_initializer=kernel_init, 
                  bias_initializer=bias_init)(pre_conv_5x5)
      # pool route = POOL + 1 × 1 CONV 
pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu',
                  kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3,  name=name) ❷

return output
```

❶ 创建一个 `1 × 1` 卷积层，其输入直接来自前一个层

❷ 将三个滤波器的深度连接在一起

### GoogLeNet 架构

现在`inception_module`函数已经准备好了，让我们根据图 5.16 构建 GoogLeNet 架构。为了获取`inception_module`函数参数的值，我们将查看图 5.17，它代表了 Szegedy 等人在原始论文中实现的超参数设置。（注意，图中的“#3 × 3 reduce”和“#5 × 5 reduce”代表在 `3 × 3` 和 `5 × 5` 卷积层之前使用的 `1 × 1` 滤波器层。）

![](img/5-17.png)

图 5.17 Szegedy 等人原 Inception 论文中实现的超参数

现在，让我们逐一查看部分 A、B 和 C 的实现。

#### Part A: 构建网络的底部部分

让我们构建网络的底部部分。这部分包括一个 `7 × 7` 卷积层 ⇒ `3 × 3` 池化层 ⇒ `1 × 1` 卷积层 ⇒ `3 × 3` 卷积层 ⇒ `3 × 3` 池化层，如图 5.18 所示。

![](img/5-18.png)

图 5.18 网络的底部部分

在`LocalResponseNorm`层中，类似于 AlexNet，使用局部响应归一化来帮助加速收敛。如今，批量归一化被用来代替。

这里是部分 A 的 Keras 代码：

```
# input layer with size = 24 × 24 × 3
input_layer = Input(shape=(224, 224, 3))
kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value=0.2)
x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', 
kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu')(x)
x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu')(x)

x = BatchNormalization()(x)

x = MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)
```

#### Part B: 构建 inception 模块和最大池化层

要构建 inception 模块 3a 和 3b 以及第一个最大池化层，我们使用表 5.2 开始。代码如下：

表 5.2 Inception 模块 3a 和 3b

| 类型 | #1 × 1 | #3 × 3 reduce | #3 × 3 | #5 × 5 reduce | #5 × 5 | Pool proj |
| --- | --- | --- | --- | --- | --- | --- |
| Inception (3a) | 064 | 096 | 128 | 16 | 32 | 32 |
| Inception (3b) | 128 | 128 | 192 | 32 | 96 | 64 |

```
x = inception_module(x, filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128, 
                     filters_5x5_reduce=16, filters_5x5=32, filters_pool_proj=32,
                     name='inception_3a')

x = inception_module(x, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=192,
                     filters_5x5_reduce=32, filters_5x5=96, filters_pool_proj=64,
                     name='inception_3b')

x = MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)
```

类似地，让我们创建 inception 模块 4a、4b、4c、4d 和 4e 以及最大池化层：

```
x = inception_module(x, filters_1x1=192, filters_3x3_reduce=96, filters_3x3=208, 
                     filters_5x5_reduce=16, filters_5x5=48, filters_pool_proj=64,
                     name='inception_4a')

x = inception_module(x, filters_1x1=160, filters_3x3_reduce=112, filters_3x3=224,
                     filters_5x5_reduce=24, filters_5x5=64, filters_pool_proj=64,
                     name='inception_4b')

x = inception_module(x, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=256,
                     filters_5x5_reduce=24, filters_5x5=64, filters_pool_proj=64,
                     name='inception_4c')

x = inception_module(x, filters_1x1=112, filters_3x3_reduce=144, filters_3x3=288,
                     filters_5x5_reduce=32, filters_5x5=64, filters_pool_proj=64,
                     name='inception_4d')

x = inception_module(x, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320,
                     filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=128,
                     name='inception_4e')

x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)
```

现在，让我们创建模块 5a 和 5b：

```
x = inception_module(x, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320, 
                     filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=128, 
                     name='inception_5a')

x = inception_module(x, filters_1x1=384, filters_3x3_reduce=192, filters_3x3=384, 
                     filters_5x5_reduce=48, filters_5x5=128, filters_pool_proj=128, 
                     name='inception_5b')
```

#### Part C: 构建分类器部分

在他们的实验中，Szegedy 等人发现添加一个 `7 × 7` 平均池化层将 top-1 准确率提高了约 0.6%。然后他们添加了一个 40% 概率的 dropout 层来减少过拟合：
