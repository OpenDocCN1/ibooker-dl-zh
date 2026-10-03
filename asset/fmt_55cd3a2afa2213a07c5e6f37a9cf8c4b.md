
上一段中所说的内容同样适用于浏览器环境。你保存的文件可以用来在网页中重建模型。重建后的模型支持完整的`tf.LayersModel()`工作流程，但有一个警告：如果你重新训练整个模型，由于增强卷积网络的尺寸较大，速度会特别慢且效率低下。在`tfjs-node`和浏览器中加载模型唯一根本不同的是，你在浏览器中应该使用除`file://`之外的其他 URL 方案。通常，你可以将`model.json`和`weights.bin`文件作为静态资产文件放在 HTTP 服务器上。假设你的主机名是`localhost`，你的文件在服务器路径`my/models/`下可见；你可以使用以下行在浏览器中加载模型：

```js
const loadedModel =
    await tf.loadLayersModel('http:///localhost/my/models/model.json');
```

在浏览器中处理基于 HTTP 的模型加载时，`tf.loadLayersModel()`在底层调用浏览器内置的 fetch 函数。因此，它具有以下特性和属性：

+   支持`http://`和`https://`。
+   支持相对服务器路径。事实上，如果使用相对路径，则可以省略 URL 的`http://`或`https://`部分。例如，如果你的网页位于服务器路径`my/index.html`，你的模型的 JSON 文件位于`my/models/model.json`，你可以使用相对路径`model/model.json`：

    ```js
    const loadedModel = await tf.loadLayersModel('models/model.json');
    ```

+   若要为 HTTP/HTTPS 请求指定额外选项，应该使用`tf.io.browserHTTPRequest()`方法代替字符串参数。例如，在模型加载过程中包含凭据和标头，你可以这样做：

    ```js
    const loadedModel = await tf.loadLayersModel(tf.io.browserHTTPRequest(
      'http://foo.bar/path/to/model.json',
      {credentials: 'include', headers: {'key_1': 'value_1'}}));
    ```

## 4.4 语音识别：在音频数据上应用卷积网络

到目前为止，我们已经向您展示了如何使用卷积网络执行计算机视觉任务。但是人类的感知不仅仅是视觉。音频是感知数据的一个重要模态，并且可以通过浏览器 API 进行访问。如何识别语音和其他类型声音的内容和意义？值得注意的是，卷积网络不仅适用于计算机视觉，而且在音频相关的机器学习中也以显著的方式发挥作用。

在本节中，您将看到我们如何使用类似于我们为 MNIST 构建的卷积网络来解决一个相对简单的音频任务。该任务是将短语音片段分类到 20 多个单词类别中。这个任务比您可能在亚马逊 Echo 和 Google Home 等设备中看到的语音识别要简单。特别是，这些语音识别系统涉及比本示例中使用的词汇量更大的词汇。此外，它们处理由多个词连续发音组成的连续语音，而我们的示例处理逐个单词发音。因此，我们的示例不符合“语音识别器”的条件；相反，更准确地描述它为“单词识别器”或“语音命令识别器”。然而，我们的示例仍然具有实际用途（如无需手动操作的用户界面和可访问性功能）。此外，本示例中体现的深度学习技术实际上是更高级语音识别系统的基础。^([9])

> ⁹
>
> Ronan Collobert、Christian Puhrsch 和 Gabriel Synnaeve，“Wav2Letter: 一种基于端到端卷积网络的语音识别系统”，2016 年 9 月 13 日提交，[`arxiv.org/abs/1609.03193`](https://arxiv.org/abs/1609.03193)。

### 4.4.1 声谱图：将声音表示为图像

与任何深度学习应用一样，如果您想要理解模型的工作原理，首先需要了解数据。要理解音频卷积网络的工作原理，我们需要首先查看声音是如何表示为张量的。请回忆高中物理课上的知识，声音是空气压力变化的模式。麦克风捕捉到空气压力变化并将其转换为电信号，然后计算机的声卡可以将其数字化。现代 Web 浏览器提供了*WebAudio* API，它与声卡通信并提供对数字化音频信号的实时访问（在用户授权的情况下）。因此，从 JavaScript 程序员的角度来看，声音就是一组实值数字的数组。在深度学习中，这种数字数组通常表示为 1D 张量。

你可能会想，迄今为止我们见过的这种卷积网络是如何在 1D 张量上工作的？它们不是应该操作至少是 2D 的张量吗？卷积网络的关键层，包括`conv2d`和`maxPooling2d`，利用了 2D 空间中的空间关系。事实证明声音*可以*被表示为称为*声谱图*的特殊类型的图像。声谱图不仅使得可以在声音上应用卷积网络，而且在深度学习之外还具有理论上的解释。

如图 4.12 所示，频谱图是一个二维数组，可以以与 MNIST 图像基本相同的方式显示为灰度图像。水平维度是时间，垂直维度是频率。频谱图的每个垂直切片是一个短时间窗口内的声音的*频谱*。频谱是将声音分解为不同频率分量的过程，可以粗略地理解为不同的“音高”。就像光可以通过棱镜分解成多种颜色一样，声音可以通过称为*傅里叶变换*的数学操作分解为多个频率。简而言之，频谱图描述了声音的频率内容如何在一系列连续的短时间窗口（通常约为 20 毫秒）内变化。

#### 图 4.12 “zero” 和 “yes” 这两个孤立口语单词的示例频谱图

频谱图是声音的联合时间-频率表示。你可以将频谱图视为声音的图像表示。沿着时间轴的每个切片（图像的一列）都是时间的短时刻（帧）；沿着频率轴的每个切片（图像的一行）对应于特定的窄频率范围（音调）。图像的每个像素的值表示给定时间点上给定频率区段的声音相对能量。本图中的频谱图被渲染为较暗的灰色，对应着较高的能量。不同的语音有不同的特征。例如，类似于“z”和“s”这样的咝音辅音以在 2–3 kHz 以上频率处集中的准稳态能量为特征；像“e”和“o”这样的元音以频谱的低端（< 3 kHz）中的水平条纹（能量峰值）为特征。在声学中，这些能量峰值被称为*共振峰*。不同的元音具有不同的共振峰频率。所有这些不同语音的独特特征都可以被深度卷积神经网络用于识别单词。

![](img/04fig11_alt.jpg)

谱图对于以下原因是声音的合适表示。首先，它们节省空间：谱图中的浮点数通常比原始波形中的浮点值少几倍。其次，在宽泛的意义上，谱图对应于生物学中的听力工作原理。内耳内部的一种名为耳蜗的解剖结构实质上执行了傅里叶变换的生物版本。它将声音分解成不同的频率，然后被不同组听觉神经元接收。第三，谱图表示可以更容易地区分不同类型的语音。这在图 4.12 的示例语音谱图中可以看到：元音和辅音在谱图中都有不同的特征模式。几十年前，在机器学习被广泛应用之前，从谱图中检测不同的元音和辅音的人们实际上尝试手工制作规则。深度学习为我们节省了这种手工制作的麻烦和泪水。

让我们停下来思考一下。看一看图 4.1 中的 MNIST 图片和图 4.12 中的声音谱图，你应该能够理解这两个数据集之间的相似之处。两个数据集都包含在二维特征空间中的模式，一双经过训练的眼睛应该能够区分出来。两个数据集都在特征的具体位置、大小和细节上呈现一定的随机性。最后，两个数据集都是多类别分类任务。虽然 MNIST 有 10 个可能的类别，我们的声音命令数据集有 20 个类别（从 0 到 9 的 10 个数字，“上”，“下”，“左”，“右”，“前进”，“停止”，“是”，“否”，以及“未知”词和背景噪音的类别）。正是这些数据集本质上的相似性使得卷积神经网络非常适用于声音命令识别任务。

但是这两个数据集也有一些显著的区别。首先，声音命令数据集中的音频录音有些噪音，可以从图 4.12 中的示例谱图中看到不属于语音声音的黑色像素点。其次，声音命令数据集中的每个谱图的尺寸为 43×232，与单个 MNIST 图像的 28×28 大小相比显著较大。谱图的尺寸在时间和频率维度之间是不对称的。这些差异将体现在我们将在音频数据集上使用的卷积神经网络中。

定义和训练声音命令卷积神经网络的代码位于`tfjs-models`存储库中。您可以使用以下命令访问代码：

```js
git clone https://github.com/tensorflow/tfjs-models.git
cd speech-commands/training/browser-fft
```

模型的创建和编译封装在`model.ts`中的`createModel()`函数中。

#### 4.8 章节的声音命令谱图分类的卷积神经网络

```js
function createModel(inputShape: tf.Shape, numClasses: number) {
  const model = tf.sequential();
  model.add(tf.layers.conv2d({                                           ***1***
    filters: 8,
    kernelSize: [2, 8],
    activation: 'relu',
    inputShape
  }));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  model.add(      tf.layers.conv2d({
        filters: 32,
        kernelSize: [2, 4],
        activation: 'relu'
      }));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  model.add(
      tf.layers.conv2d({
        filters: 32,
        kernelSize: [2, 4],
        activation: 'relu'
      }));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  model.add(
      tf.layers.conv2d({
        filters: 32,
        kernelSize: [2, 4],
        activation: 'relu'
      }));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [1, 2]}));
  model.add(tf.layers.flatten());                                        ***2***
  model.add(tf.layers.dropout({rate: 0.25}));                            ***3***
  model.add(tf.layers.dense({units: 2000, activation: 'relu'}));
  model.add(tf.layers.dropout({rate: 0.5}));
  model.add(tf.layers.dense({units: numClasses, activation: 'softmax'}));

  model.compile({                                                        ***4***
    loss: 'categoricalCrossentropy',
    optimizer: tf.train.sgd(0.01),
    metrics: ['accuracy']
  });
  model.summary();
  return model;
}
```

+   ***1*** `conv2d`+`maxPooling2d`的重复模式
+   ***2*** 多层感知器开始
+   ***3*** 使用`dropout`减少过拟合
+   ***4*** 配置多类别分类的损失和指标

我们的音频卷积网络的拓扑结构看起来很像 MNIST 卷积网络。顺序模型以多个重复的`conv2d`层与`maxPooling2d`层组合开始。模型的卷积-池化部分在一个展平层结束，在其上添加了 MLP。MLP 有两个密集层。隐藏的密集层具有`relu`激活，最终（输出）层具有适合分类任务的`softmax`激活。模型编译为在训练和评估期间使用`categoricalCrossentropy`作为损失函数并发出准确度指标。这与 MNIST 卷积网络完全相同，因为两个数据集都涉及多类别分类。音频卷积网络还显示出与 MNIST 不同的一些有趣之处。特别是，`conv2d`层的`kernelSize`属性是矩形的（例如，`[2, 8]`）而不是方形的。这些值被选择为与频谱图的非方形形状匹配，该频谱图的频率维度比时间维度大。

要训练模型，首先需要下载语音命令数据集。该数据集源自谷歌 Brain 团队工程师 Pete Warden 收集的语音命令数据集（请参阅 [www.tensorflow.org/tutorials/sequences/audio_recognition](http://www.tensorflow.org/tutorials/sequences/audio_recognition)）。它已经转换为浏览器特定的频谱图格式：

```js
curl -fSsL https://storage.googleapis.com/learnjs-data/speech-
     commands/speech-commands-data- v0.02-browser.tar.gz  -o speech-commands-
     data-v0.02-browser.tar.gz &&
tar xzvf speech-commands-data-v0.02-browser.tar.gz
```

这些命令将下载并提取语音命令数据集的浏览器版本。一旦数据被提取，您就可以使用以下命令启动训练过程：

```js
yarn
yarn train \
    speech-commands-data-browser/ \
    /tmp/speech-commands-model/
```

`yarn train`命令的第一个参数指向训练数据的位置。以下参数指定了模型的 JSON 文件将保存的路径，以及权重文件和元数据 JSON 文件的路径。就像我们训练增强的 MNIST 卷积网络时一样，音频卷积网络的训练也发生在`tfjs-node`中，有可能利用 GPU。由于数据集和模型的大小都比 MNIST 卷积网络大，训练时间会更长（大约几个小时）。如果您有 CUDA GPU 并且稍微更改命令以使用`tfjs-node-gpu`而不是默认的`tfjs-node`（仅在 CPU 上运行），您可以显著加快训练速度。要做到这一点，只需在上一个命令中添加标志`--gpu`：

```js
yarn train \
  --gpu \
  speech-commands-data-browser/ \
  /tmp/speech-commands-model/
```

当训练结束时，模型应该达到约 94% 的最终评估（测试）准确率。

训练过的模型保存在上一个命令中指定的路径中。与我们用`tfjs-node`训练的 MNIST 卷积网络一样，保存的模型可以在浏览器中加载以提供服务。然而，您需要熟悉 WebAudio API，以便能够从麦克风获取数据并将其预处理为模型可用的格式。为了方便起见，我们编写了一个封装类，不仅可以加载经过训练的音频卷积网络，还可以处理数据输入和预处理。如果您对音频数据输入流水线的机制感兴趣，可以在`tfjs-model`Git 仓库中的`speech-commands/src`文件夹中研究底层代码。这个封装类可以通过 npm 的`@tensorflow-models/speech-commands`名称使用。Listing 4.9 展示了如何使用封装类在浏览器中进行在线语音命令识别的最小示例。

在`tfjs-models`仓库的`speech-commands/demo`文件夹中，您可以找到一个相对完整的示例，该示例展示了如何使用该软件包。要克隆并运行该演示，请在`speech-commands`目录下运行以下命令：

```js
git clone https://github.com/tensorflow/tfjs-models.git
cd tfjs-models/speech-commands
yarn && yarn publish-local
cd demo
yarn && yarn link-local && yarn watch
```

`yarn watch`命令将自动在默认的网页浏览器中打开一个新的标签页。要看到语音命令识别器的实际效果，请确保您的计算机已准备好麦克风（大多数笔记本电脑都有）。每次识别到词汇表中的一个单词时，屏幕上将显示该单词以及包含该单词的一秒钟的频谱图。所以，这是基于浏览器的单词识别，由 WebAudio API 和深度卷积网络驱动。当然，它没有能力识别带有语法的连接语音？这将需要其他类型的能够处理序列信息的神经网络模块的帮助。我们将在第八章中介绍这些模块。

#### Listing 4.9 `@tensorflow-models/speech-commands`模块的示例用法
