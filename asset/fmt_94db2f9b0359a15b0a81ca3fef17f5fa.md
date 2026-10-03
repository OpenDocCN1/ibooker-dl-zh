
在深度卷积中，核被分割成一个单独的`H` × `W` × 1核，每个通道一个，每个核只对一个通道进行操作，而不是对所有通道进行操作。在这种安排中，跨通道关系与空间关系解耦。正如 Chollet 所建议的，完全解耦空间和通道卷积会导致更少的`matmul`操作，并且精度与没有解耦和正常卷积的模型相当，以及与部分解耦和空间可分离卷积的模型相当。

因此，在图 7.17 中展示的 RGB 示例中，使用 3 × 3 核，深度卷积将会有三个 3 × 3 × 1 的核。当核移动时，乘法操作的次数与正常卷积相同（例如，在三个通道上为 27）。然而，输出是一个`D`深度的特征图，而不是一个 2D（`depth=1`）的特征图。

![图片](img/CH07_F17_Ferlitsch.png)

图 7.17 在这个深度卷积中，核被分割成单个`H` × `W` × 1核。

### 7.2.7 点卷积

深度卷积的输出随后作为点卷积的输入，形成深度可分离卷积。点卷积执行解耦的空间卷积。点卷积结合深度卷积的输出，并将特征图的数量扩展到匹配指定的过滤器（特征图）数量。组合输出与正常或可分离卷积（89）相同数量的特征图，但矩阵乘法操作更少（减少了 83%）。

点卷积，如图 7.18 所示，具有 1 × 1 × `D`（通道数）。它将遍历每个像素，生成一个`N` × `M` × 1的特征图，这取代了`N` × `M` × `D`的特征图。

![图片](img/CH07_F18_Ferlitsch.png)

图 7.18 点卷积

在点卷积中，我们使用 1 × 1 × `D`核，每个输出一个。正如图 7.17 中的前一个例子一样，如果我们的输出是 256 个滤波器（特征图），我们将使用 256 个 1 × 1 × `D`核。

在图 7.17 中使用的 RGB 示例中，深度卷积使用了 3 × 3 × 3 核，每次核移动时都有 27 次乘法操作。这将随后是一个 1 × 1 × 3 × 256（其中 256 是输出滤波器的数量）——这是 768。总的乘法操作数将是 795，而不是正常卷积的 6912 次和空间可分离卷积的 4608 次。

在 Xception 架构中，inception 模块中的空间可分离卷积被深度可分离卷积所取代，通过 83%的计算复杂度（乘法操作数）减少。使用 Idiomatic procedure reuse 设计模式对 Xception 的完整代码实现可在 GitHub 上找到（[`mng.bz/5WzB`](http://mng.bz/5WzB)）。

## 7.3 SE-Net：挤压和激励

现在我们将转向另一种替代连接设计，即*squeeze-excitation-scale pattern*或*SE-Net*，它可以通过仅添加少量参数来添加到现有的残差网络中以提高准确率。

在“Squeeze-and-Excitation Networks”中引入了这种模式（[`arxiv.org/abs/1709.01507`](https://arxiv.org/abs/1709.01507)），Jie Hu 等人解释说，之前对模型的改进主要集中在卷积层之间的空间关系上。因此，他们决定采取不同的方法，并研究基于通道之间关系的新网络设计。他们的想法是，特征重校准可以使用全局信息来选择性地强调重要特征并降低不太重要的特征的重要性。

为了实现选择性强调特征的能力，作者提出了在残差块内添加一个*squeeze-excitation* (*SE*) *链接*的概念。这个块将位于卷积层（或多个层）的输出和与恒等链接的矩阵加法操作之间。这个概念赢得了 2017 年 ILSVRC 竞赛的 ImageNet。

他们的消融研究指出了 SE-Net 方法的一些好处，包括以下这些：

+   可以添加到现有的 SOTA 架构中，例如 ResNet、ResNeXt 和 Inception。

+   在实现更高准确率的同时，参数增加最小。例如：

    +   ResNet50 的 ImageNet top-5 错误率为 7.48%，SE-ResNet50 为 6.62%

    +   ResNeXt50 的 ImageNet top-5 错误率为 5.9%，SE-ResNeXt50 为 5.49%

    +   Inception 的 ImageNet top-5 错误率为 7.89%，SE-Inception 为 7.14%

### 7.3.1 SE-Net 的架构

SE-Net 架构，如图 7.19 所示，由一个现有的残差网络组成，然后通过在残差块中插入 SE 链接进行改造。改造后的 ResNet 和 ResNeXt 架构分别称为*SE-ResNet*和*SE-ResNeXt*。

![](img/CH07_F19_Ferlitsch.png)

图 7.19 SE-Net 宏架构展示了将 SE-链接添加到每个残差块

### 7.3.2 SE-Net 的组和块

如果我们分解宏观架构，我们会发现图 7.19 中的每个卷积组都由一个或多个残差块组成，构成一个残差组。每个残差块都有一个 SE 链接。这种对残差组的近距离观察在图 7.20 中展示。

![图片](img/CH07_F20_Ferlitsch.png)

图 7.20 在残差组中，每个残差块都插入了一个 SE 链接。

现在，让我们分解残差组。图 7.21 展示了 SE 链接是如何插入到残差块中，位于卷积层（s）输出和矩阵加法操作之间。

![图片](img/CH07_F21_Ferlitsch.png)

图 7.21 残差块展示了 SE 链接是如何插入到残差路径和矩阵加法操作（标记为 *Add*）之间的（对于身份链接）

以下代码是向 ResNet 残差块添加 SE 链接的示例实现。在块的末尾，在`B(3,3)`输出和矩阵加法操作（`Add()`）之间插入对`squeeze_excite_link()`的调用。在`squeeze_excite_link()`函数中，我们实现了 SE 链接（在下一小节中详细介绍）。

参数`ratio`是在激励操作之前对输入进行挤压操作的维度降低量（比率）。

```
def identity_block(x, n_filters, ratio=16):
    """ Create a Bottleneck Residual Block with Identity Link
        x        : input into the block
        n_filters: number of filters
        ratio    : amount of filter reduction during squeeze
    """
    shortcut = x

    x = Conv2D(n_filters, (1, 1), strides=(1, 1))(x)                    ❶
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same")(x)    ❷
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(n_filters * 4, (1, 1), strides=(1, 1))(x)                ❸
    x = BatchNormalization()(x)

    x = squeeze_excite_link(x, ratio)                                   ❹

    x = Add()([shortcut, x])                                            ❺
    x = ReLU()(x)
    return x
```

❶ 用于维度降低的 1 × 1 卷积

❷ 瓶颈层使用 3 × 3 卷积

❸ 1 × 1 卷积通过增加过滤器数量 4 倍来实现维度恢复

❹ 将输出通过挤压-激励链接

❺ 将身份链接（输入）添加到残差块的输出

### 7.3.3 SE 链接

现在，让我们详细探讨 SE 链接（图 7.22）。该链接由三个层组成。前两层执行挤压操作。使用全局平均池化将每个输入特征图（通道）减少到一个单一值，输出一个大小为`C`（通道）的 1D 向量，然后将其重塑为一个大小为`C`（通道）的 1-×-1 像素的 2D 矩阵。密集层随后通过减少比率`r`进一步减少输出，结果是一个大小为`C` / `r`（通道）的 1-×-1 像素的 2D 矩阵。

![图片](img/CH07_F22_Ferlitsch.png)

图 7.22 挤压-激励块展示了挤压、激励然后缩放操作

挤压后的输出随后传递到第三层，该层通过恢复到链接输入的通道数（`C`）来进行激励。请注意，这与使用 1 × 1 线性投影卷积相当，但这里使用的是密集层。

最后一步是一个缩放操作，它由一个从输入的身份链接组成，其中来自挤压-激励操作的 1 × 1 × `C`向量与输入（`H` × `W` × `C`）进行矩阵乘法。缩放操作之后，输出维度（特征图的数量和大小）恢复到输入的原始维度（缩放）。

现在让我们看看 SE 链接的一个示例实现，包括挤压、激励和缩放操作。注意在`GlobalAveragePooling2D`之后的`Reshape`操作，将池化的 1D 向量转换为 1-×-1 像素的 2D 向量，以便后续的两个`Dense`层执行挤压和激励操作。激励产生的 1 × 1 × `C`矩阵随后与输入（`shortcut`）进行矩阵乘法，以进行缩放操作：

```
def squeeze_excite_link(x, ratio=16):
    """ Create a Squeeze and Excite link
        x    : input to the link
        ratio : amount of filter reduction during squeeze
    """
    shortcut = x
    n_filters = x.shape[-1]                                  ❶
    x = GlobalAveragePooling2D()(x)                          ❷
    x = Reshape((1, 1, n_filters))(x)                        ❸
    x = Dense(n_filters // ratio, activation='relu')(x)      ❹
    x = Dense(n_filters, activation='sigmoid')(x)            ❺
    x = Multiply()([shortcut, x])                            ❻
    return x
```

❶ 获取 SE 链接输入中的特征图（滤波器）数量

❷ 使用全局平均池化进行维度减少的挤压操作，将输出一个 1D 向量

❸ 将输出重塑为 1 × 1 特征图（1 × 1 × C）

❹ 通过减少比例将滤波器数量（1 × 1 × C / r）减少

❺ 通过恢复滤波器数量（1 × 1 × C）进行维度恢复的激励操作

❻ 缩放操作，将挤压/激励输出与输入（W × H × C）相乘

使用 Idiomatic 程序重用设计模式在 GitHub 上提供了一个 SE-Net 的完整代码实现（[`mng.bz/vea7`](https://shortener.manning.com/vea7)）。

## 摘要

+   DenseNet 中的特征重用通过将残差块的输入到输出的特征图拼接代替矩阵加法。这种分类提高了现有 SOTA 模型的准确性。

+   在 DenseNet 中使用 1 × 1 卷积来学习是针对特定数据集进行上采样和下采样的最佳方式。

+   在 Xception 中将空间可分离卷积进一步重构为深度可分离卷积，进一步降低计算成本，同时保持表示等价。

+   在 SE-Net 中添加 squeeze-excite-scale 模式，同时仅增加少量参数，可以提高准确性。
