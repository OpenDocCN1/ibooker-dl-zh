
# 密集连接网络与多层感知机

*密集连接网络*和*多层感知机*这两个术语在很大程度上可以互换使用，但要注意密集连接网络可以只包含一个层，而多层感知机必须至少包含一个隐藏层和一个输出层。为了简洁起见，我们将使用*MLP*这个术语来指代主要由密集层构建的所有模型。此类网络专门用于无序向量数据（例如，在钓鱼网站检测问题和房价预测问题中的数值特征）。每个密集层试图对输入特征的所有可能的对和该层的输出激活之间的关系进行建模。这通过密集层的核与输入向量之间的矩阵乘法（然后加上偏差向量和激活函数）来实现。每个输出激活受到每个输入特征的影响是这些层和建立在它们上面的网络被称为*密集连接*（或被一些作者称为*完全连接*）的原因。这与其他类型的架构（卷积网络和循环神经网络）形成对比，在这些架构中，一个输出元素仅依赖于输入数据中的一部分元素。

MLP 最常用于分类数据（例如，输入特征是属性列表，比如在钓鱼网站检测问题中）。它们也经常被用作大多数分类和回归神经网络的最终输出阶段，这些网络可以包含卷积或循环层作为特征提取器，将特征输入馈送到这样的 MLP 中。例如，我们在第四章和第五章中介绍的二维卷积网络都以一个或两个密集层结尾，我们在第九章中介绍的循环网络也是如此。

## 输出层激活函数的选择

让我们简要回顾一下在监督学习中如何选择 MLP 的输出层激活函数以适应不同类型的任务。

### 二元分类

要进行二元分类，你的 MLP 的最后一个密集层应该正好有一个单元，并使用 sigmoid 激活函数。在训练这样一个二元分类器 MLP 时，应该使用`binaryCrossentropy`作为损失函数。你的训练数据中的例子应该有二元标签（0 或 1 的标签）。具体来说，TensorFlow.js 的代码如下：

```js
import * as tf from '@tensorflow/tfjs';

const model = tf.sequential();
model.add(tf.layers.dense({units: 32, activation: 'relu', inputShape:
     [numInputFeatures]}));
model.add(tf.layers.dense({units: 32, activation: 'relu'}));
model.add(tf.layers.dense({units: 1: activation: 'sigmoid'}));
model.compile({loss: 'binaryCrossentropy', optimizer: 'adam'});
```

### 单标签多类别分类

要进行单标签多类别分类（其中每个例子在多个候选类别中只有一个类别），请在你的层堆叠的末尾加上一个包含 softmax 激活函数的密集层，单位数量等于类别的数量。如果你的目标是独热编码，请使用`categoricalCrossentropy`作为损失函数；如果它们是整数索引，请使用`sparseCategoricalCrossentropy`。例如：

```js
const model = tf.sequential();
model.add(tf.layers.dense({units: 32, activation: 'relu', inputShape:
     [numInputFeatures]}));
model.add(tf.layers.dense({units: 32, activation: 'relu'}));
model.add(tf.layers.dense({units: numClasses: activation: 'softmax'}));
model.compile({loss: 'categoricalCrossentropy', optimizer: 'adam'});
```

### 多标签多类别分类

为了执行多标签多类别分类（每个示例可能具有多个正确类别），将你的层堆栈结束为一个包含 sigmoid 激活和单位数量等于所有候选类别数量的密集层。使用`binaryCrossentropy`作为损失函数。你的目标应该是 k-hot 编码：

```js
const model = tf.sequential();
model.add(tf.layers.dense({units: 32, activation: 'relu', inputShape:
     [numInputFeatures]}));
model.add(tf.layers.dense({units: 32, activation: 'relu'}));
model.add(tf.layers.dense({units: numClasses: activation: 'sigmoid'}));
model.compile({loss: 'binaryCrossentropy', optimizer: 'adam'});
```

### 回归

为了对连续值向量执行回归，将你的层堆栈结束为一个具有单位数量等于你尝试预测的值数量（通常只有一个数字，比如房价或温度值）的密集层，并使用默认的线性激活函数。多个损失函数可用于回归。最常用的是`meanSquaredError`和`meanAbsoluteError`：

```js
const model = tf.sequential();
model.add(tf.layers.dense({units: 32, activation: 'relu', inputShape:
     [numInputFeatures]}));
model.add(tf.layers.dense({units: 32, activation: 'relu'}));
model.add(tf.layers.dense({units: numClasses}));
model.compile({loss: 'meanSquaredError', optimizer: 'adam'});
```

## 卷积网络

卷积层通过将相同的几何变换应用于输入张量中的不同空间位置（补丁）来查看局部空间模式。这导致了是平移不变的表示，使卷积层高度数据有效和模块化。这个想法适用于任何维度的空间：1D（序列），2D（图像或类似于非图像数量的表示，如声音频谱图），3D（体积）等等。你可以使用`tf.layers.conv1d`层来处理序列，使用`conv2d`层来处理图像，使用`conv3d`层来处理体积。

卷积网络由一系列卷积和池化层组成。池化层允许你对数据进行空间降采样，这对于保持特征图的合理大小是必需的，因为特征数量增加，同时也允许后续层“看到”卷积网络输入图像的更大空间范围。卷积网络通常以展平层或全局池化层结束，将空间特征图转换为向量，然后可以通过一系列密集层（MLP）处理以实现分类或回归输出。

很可能常规卷积很快就会被一个等价但更快更高效的替代品所取代（或完全取代）：深度可分离卷积（`tf.layers.separableConv2d`层）。当你从头开始构建一个网络时，强烈推荐使用深度可分离卷积。可分离卷积层可以用作`tf.layers.conv2d`的即插即用替代品，结果是一个更小更快的网络，在其任务上表现同样好或更好。以下是一个典型的图像分类网络（单标签多类别分类，在本例中）。其拓扑结构包含重复的卷积-池化层组的模式：

```js
const model = tf.sequential();
model.add(tf.layers.separableConv2d({
    filters: 32, kernelSize: 3, activation: 'relu',
    inputShape: [height, width, channels]}));
model.add(tf.layers.separableConv2d({
         filters: 64, kernelSize: 3, activation: 'relu'}));
model.add(tf.layers.maxPooling2d({poolSize: 2}));

model.add(tf.layers.separableConv2d({
         filters: 64, kernelSize: 3, activation: 'relu'}));
model.add(tf.layers.separableConv2d({
         filters: 128, kernelSize: 3, activation: 'relu'}));
model.add(tf.layers.maxPooling2d({poolSize: 2}));

model.add(tf.layers.separableConv2d({
    filters: 64, kernelSize: 3, activation: 'relu'}));
model.add(tf.layers.separableConv2d({
    filters: 128, kernelSize: 3, activation: 'relu'}));
model.add(tf.layers.globalAveragePooling2d());
model.add(tf.layers.dense({units: 32, activation: 'relu'}));
model.add(tf.layers.dense({units: numClasses, activation: 'softmax'}));

model.compile({loss: 'categoricalCrossentropy', optimizer: 'adam'});
```

## 循环网络

RNN 通过一次处理一个时间戳的输入序列并始终保持状态来工作。状态通常是一个向量或一组向量（几何空间中的一个点）。在不是时间上不变的序列（例如，时间序列数据，其中最近的过去比遥远的过去更重要）的情况下，应优先使用 RNN，而不是 1D 卷积网络。

TensorFlow.js 提供了三种 RNN 层类型：`simpleRNN`、`GRU`和`LSTM`。对于大多数实际目的，您应该使用`GRU`或`LSTM`。`LSTM`是这两者中更强大的，但也更消耗计算资源。您可以将`GRU`视为`LSTM`的简化和更便宜的替代品。

为了将多个 RNN 层堆叠在一起，除了最后一层之外的每一层都需要配置为返回其输出的完整序列（每个输入时间步对应一个输出时间步）。如果不需要堆叠 RNN 层，通常 RNN 层只需返回最后一个输出，其中包含有关整个序列的信息。

以下是使用单个 RNN 层与密集层一起执行向量序列的二进制分类的示例：

```js
const model = tf.sequential();
model.add(tf.layers.lstm({
  units: 32,
  inputShape: [numTimesteps, numFeatures]
}));
model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
model.compile({loss: 'binaryCrossentropy', optimizer: 'rmsprop'});
```

接下来是一个带有 RNN 层堆叠的模型，用于向量序列的单标签多类别分类：

```js
const model = tf.sequential();
model.add(tf.layers.lstm({
  units: 32,
  returnSequences: true,
  inputShape: [numTimesteps, numFeatures]
}));
model.add(tf.layers.lstm({units: 32, returnSequences: true}));
model.add(tf.layers.lstm({units: 32}));
model.add(tf.layers.dense({units: numClasses, activation: 'softmax'}));
model.compile({loss: 'categoricalCrossentropy', optimizer: 'rmsprop'});
```

## 用于减轻过拟合和提高收敛性的层和正则化器

除了上述主要的基本层类型之外，还有一些其他类型的层适用于广泛的模型和问题类型，并协助训练过程。没有这些层，许多机器学习任务的最新准确性不会像今天这样高。例如，`dropout`和`batchNormalization`层经常插入到 MLP、卷积网络和 RNN 中，以帮助模型在训练过程中更快地收敛并减少过拟合。以下示例显示了包含`dropout`层的回归 MLP：

```js
const model = tf.sequential();
model.add(tf.layers.dense({
  units: 32,
  activation: 'relu',
  inputShape: [numFeatures]
}));
model.add(tf.layers.dropout({rate: 0.25}));
model.add(tf.layers.dense({units: 64, activation: 'relu'}));
model.add(tf.layers.dropout({rate: 0.25}));
model.add(tf.layers.dense({units: 64, activation: 'relu'}));
model.add(tf.layers.dropout({rate: 0.25}));
model.add(tf.layers.dense({
  units: numClasses,
  activation: 'categoricalCrossentropy'
}));
model.compile({loss: 'categoricalCrossentropy', optimizer: 'rmsprop'});
```

# 从 TensorFlow.js 使用预训练模型

当您要解决的机器学习问题特定于您的应用程序或数据集时，从头开始构建和训练模型是正确的方法，而 TensorFlow.js 使您能够做到这一点。然而，在某些情况下，您面临的问题是通用的，存在预训练模型，这些模型要么完全符合您的要求，要么只需进行轻微调整即可满足您的需求。来自 TensorFlow.js 和第三方开发人员的预训练模型集合。这些模型提供了干净且易于使用的 API。它们还作为 npm 包打包得很好，您可以方便地依赖它们在您的 JavaScript 应用程序（包括 Web 应用程序和 Node.js 项目）中。

在适当的使用案例中使用这些预训练模型可以大大加快您的开发速度。由于不可能列出所有基于 TensorFlow.js 的预训练模型，因此我们只会调查我们所知道的最流行的那些。以`@tensorflow-models/`为前缀的软件包是由 TensorFlow.js 团队维护的第一方软件包，而其余的是第三方开发者的工作。

`@tensorflow-models/mobilenet`是一个轻量级图像分类模型。它能够根据输入图像输出 1,000 个 ImageNet 类别的概率分数。它适用于在网页中为图像标记、从网络摄像头输入流中检测特定内容，以及涉及图像输入的迁移学习任务。虽然`@tensorflow-models/mobilenet`关注通用的图像类别，但也有第三方软件包用于更具特定领域的图像分类。例如，`nsfwjs`将图像分类为包含色情和其他不当内容与安全内容，这对家长监控、安全浏览等应用非常有用。

如我们在第五章中讨论的那样，目标检测与图像分类不同之处在于，它不仅输出图像中包含的物体，还输出它们在图像坐标系中的*位置*。`@tensorflow-models/coco-ssd`是一个能够检测 90 种对象的目标检测模型。对于每个输入图像，它都能够检测出可能有重叠边界框的多个目标对象（图 13.1，A 面）。

## 图 13.1。几个使用 TensorFlow.js 构建的预训练 npm 软件包模型的屏幕截图。

A 面：`@tensorflow-models/coco-ssd`是一个多目标对象检测器。B 面：`face-api.js`用于实时人脸和面部特征点检测（通过 Vincent Mühler 的许可从[`github.com/justadudewhohacks/face-api.js`](https://github.com/justadudewhohacks/face-api.js)复制）。C 面：`handtrack.js`实时跟踪一个或两只手的位置（通过 Victor Dibia 的许可从[`github.com/victordibia/handtrack.js/`](https://github.com/victordibia/handtrack.js/)复制）。D 面：`@tensorflow-models/posenet`使用实时图像输入检测人体的骨骼关键点。E 面：`@tensorflow-models/toxicity`检测并标记任何英文文本输入中的七种不当内容。

![](img/13fig01_alt.jpg)

对于网络应用程序，特定类型的对象特别受到关注，因为它们有可能实现新颖有趣的计算机人类交互。这些包括人脸、手和整个身体。针对每一种类型，都有基于 TensorFlow.js 的专门的第三方模型。对于人脸，`face-api.js`和`handsfree`都支持实时人脸跟踪和检测面部特征点（如眼睛或嘴巴；图 13.1，面板 B）。对于手部，`handtrack.js`可以实时跟踪一个或两只手的位置（图 13.1，面板 C）。对于整个身体，`@tensorflow-models/posenet`实现了高精度、实时的骨架关键点检测（如肩膀、肘部、臀部和膝盖；图 13.1，面板 D）。

对于音频输入模态，`@tensorflow-models/speech-commands`提供了一个预训练模型，可以实时检测浏览器的 WebAudio API 中的 18 个英文单词。虽然这不像大词汇连续语音识别那样强大，但它仍然可以在浏览器中实现一系列基于语音的用户交互。
