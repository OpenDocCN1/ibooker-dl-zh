
重新加载网页通常会清除张量，但长时间运行的 Node.js 服务器将不得不监视和验证张量和模型是否被处理。

# 我们的第一个 TensorFlow Hub 模型

现在你已经正确地编码、加载和处理了少量数据通过一个自定义模型，你应该花一点时间挑战自己。在这一部分，你将加载一个规模更大的模型从 TensorFlow Hub，并处理一张图片。井字棋是九个值的输入，而大多数图片是包含数千个值的张量。

你将要加载的模型是目前最大和最令人印象深刻的模型之一，Inception v3。Inception 模型是一个令人印象深刻的网络，最初在 2015 年创建。这第三个版本已经训练了数十万张图片。这个模型有 91.02 MB，可以对 1,001 种不同的对象进行分类。来自第二章的 Chapter Challenge 中的 MobileNet-wrapped NPM 包很棒，但不像你即将使用的模型那样强大。

## 探索 TFHub

Google 已经开始免费托管像 Inception v3 这样的模型在其自己的 CDN 上。对于这种大型模型大小的情况，拥有一个可靠且令人印象深刻的版本化 CDN 对于像我们经常为 JavaScript 做的模型非常有用。您可以在 [*https://tfhub.dev*](https://tfhub.dev) 上访问数百个经过训练并准备就绪的 TensorFlow 和 TensorFlow.js 模型。TensorFlow.js 有一种特殊的方式来识别您的模型是否托管在 TFHub 上；我们只需在确定了模型 URL 后，在配置中添加 `{ fromTFHub: true }`。

当您浏览 TFHub 时，您可以看到各种发布者和每个模型的解释。这些解释很关键，因为正如我们已经确定的那样，模型对于输入和输出的期望是非常具体的。您可以在 [与 Inception v3 相关的 TFHub 页面](https://oreil.ly/Utstp) 上了解更多信息。这个模型是由 Google 构建的，提供的版本经过了广泛的训练。如果您渴望获取更多信息，不妨浏览一下 [关于该模型的发表论文](https://arxiv.org/abs/1512.00567)。

在 TFHub 页面上，您可以获得使用模型的这两个关键见解。首先，预期的输入图片尺寸应为 299 x 299，值应为 `0-1`，并且应该像我们在之前的井字棋示例中一样进行批处理。其次，模型返回的结果是一个具有 1,001 个值的单维张量，最大的值最有可能（类似于井字棋返回的九个值）。这可能听起来有点混乱，但该页面使用了一些基于统计的术语来表达这一点：

> 输出是一批 logits 向量。Logits 中的索引是原始训练中分类的 num_classes = 1001 个类。

返回一个数值结果是有用的，但是像往常一样，我们需要将其映射回一个有用的标签。在井字棋中，我们将索引映射到棋盘上的位置，而在这种情况下，我们将值的索引映射到相应的标签，这些标签遵循相同的索引。TFHub 页面 [分享了一个 TXT 文件](https://oreil.ly/bzUeD)，其中包含了所有必要标签的正确顺序，您将使用这些标签创建一个数组来解释预测结果。

## 连接 Inception v3

现在您知道 Inception v3 模型可以对照片进行分类，并且您已经了解了输入和输出规范。这就像是井字棋问题的一个更大版本。然而，会有新的障碍。例如，打印 1,001 个数字并不会提供有用的信息。您需要使用 `topk` 将巨大的张量解析回一个有用的上下文中。

以下代码可在 GitHub 仓库的 [*chapter5/simple/simple-tfhub*](https://oreil.ly/X7TpN) 文件夹中找到。该代码依赖于一个具有 `id` `mystery` 的神秘图片。理想情况下，AI 可以为我们解决这个谜题：

```js
tf.ready().then(() => {
  const modelPath =
    "https://tfhub.dev/google/tfjs-model/imagenet/inception_v3/classification/3/default/1"; // ①
  tf.tidy(() => {
    tf.loadGraphModel(modelPath, { fromTFHub: true }).then((model) => { // ②
      const mysteryImage = document.getElementById("mystery");
      const myTensor = tf.browser.fromPixels(mysteryImage);
      // Inception v3 expects an image resized to 299x299
      const readyfied = tf.image
        .resizeBilinear(myTensor, [299, 299], true) // ③
        .div(255) // ④
        .reshape([1, 299, 299, 3]); // ⑤

      const result = model.predict(readyfied); // ⑥
      result.print();  // ⑦

      const { values, indices } = tf.topk(result, 3); // ⑧
      indices.print(); // ⑨

      // Let's hear those winners
      const winners = indices.dataSync();
      console.log(` 10
         First place ${INCEPTION_CLASSES[winners[0]]},
         Second place ${INCEPTION_CLASSES[winners[1]]},
         Third place ${INCEPTION_CLASSES[winners[2]]}
      `);
    });
  });
});
```

① 这是 Inception 模型的 TFHub 的 URL。

② 加载图模型并将 `fromTFHub` 设置为 true。

③ 图片被调整为 299 x 299。

④ 将 `fromPixels` 的结果转换为介于 0 和 1 之间的值（对数据进行归一化）。

⑤ 将 3D 张量转换为单批次 4D 张量，就像模型期望的那样。

⑥ 对图片进行预测。

⑦ 打印内容太多被截断了。

⑧ 恢复前三个值作为我们的猜测。

⑨ 打印前三个预测索引。

⑩ 将索引映射到它们的标签并打印出来。`INCEPTION_CLASSES` 是一个标签数组，映射到模型输出。

在本章的相关代码中，您会发现三幅图片，您可以将其设置为本节中的神秘图片。Inception v3 令人印象深刻地正确识别了所有三幅图片。查看 图 5-4 中捕获的结果。

![Inception 正确识别磁带播放器](img/ltjs_0504.png)

###### 图 5-4\. Inception v3 图像的分类结果

从照片中可以看出，Inception 的第一个选择是“磁带播放器”，我认为这非常准确。其次，它看到了一个“磁带播放器”，老实说我不知道这和“磁带播放器”有什么不同，但我不是超级模型。最后，第三高的值是“收音机”，这就是我会说的。

你通常不需要像这样的大型模型，但随着新模型被添加到 TFHub，你知道你有选择。偶尔浏览现有模型。你会看到很多关于图像分类的模型。对图像进行分类是 AI 入门中比较令人印象深刻的任务之一，但为什么要止步于此呢？

# 我们的第一个叠加模型

到目前为止，你一直在处理简单的输出模型。井字棋识别你的下一步，Inception 对照片进行分类，为了全面，你将在电影中展示 AI 的经典视觉效果，即在照片中识别物体的边界框。AI 不是对整个照片进行分类，而是在照片中突出显示特定的边界框，就像图 5-5 中那样。

![气球图像的边界框](img/ltjs_0505.png)

###### 图 5-5\. 边界框叠加

通常，模型的边界框输出相当复杂，因为它处理各种类别和重叠框。通常，模型会让你使用一些数学方法来正确清理结果。与其处理这些，不如专注于在 TensorFlow.js 中绘制预测输出中的单个矩形。有时这被称为*对象定位*。

这个最终练习的模型将是一个宠物脸部检测器。该模型将尽力为我们提供一个边界坐标集，指示它认为宠物脸部位于何处。通常不难说服人们看可爱的狗和猫，但这个模型可能有各种应用。一旦你有了宠物脸部的位置，你可以使用这些数据来训练额外的模型，比如识别宠物或检查它们可爱的鼻子是否需要 boop。你懂的...科学！

## 定位模型

这个模型是在一个名为[Oxford-IIIT 宠物数据集](https://oreil.ly/Pz0D9)上训练的。这个小巧的、大约 2MB 的模型期望一个 256 x 256 的`Float32`输入 RGB 宠物图像，并输出四个数字来识别围绕宠物脸部的边界框。1D 张量中的四个数字是左上角点和右下角点。

这些点表示为 0 到 1 之间的值，作为图像的百分比。你可以使用模型结果信息定义一个矩形，如图 5-6 所示。

![4 个值如何变成两个点的显示](img/ltjs_0506.png)

###### 图 5-6\. 四个值变成两个点

代码的开头将与之前的代码类似。你将首先将图像转换为张量，然后通过模型运行。以下代码可以在 GitHub 仓库的[*chapter5/simple/simple-object-localization*](https://oreil.ly/zkSfM)中找到。

```js
const petImage = document.getElementById("pet");
const myTensor = tf.browser.fromPixels(petImage);
// Model expects 256x256 0-1 value 3D tensor
const readyfied = tf.image
  .resizeNearestNeighbor(myTensor, [256, 256], true)
  .div(255)
  .reshape([1, 256, 256, 3]);

const result = model.predict(readyfied);
// Model returns top left and bottom right
result.print();
```

## 标记检测

现在你可以将结果坐标绘制为图像上的矩形。在 TensorFlow.js 中绘制检测是一个常见的任务。在图像上绘制张量结果的基本方法需要你将图像放在一个容器中，然后在图像上方放置一个绝对位置的画布。现在当你在画布上绘制时，你将在图像上绘制。¹ 从侧面看，布局将类似于图 5-7。

![DOM 的 3D 视图](img/ltjs_0507.png)

###### 图 5-7\. 画布的堆叠视图

对于这节课，CSS 已经直接嵌入到 HTML 中以方便。图像和画布布局如下：

```js
<div style="position: relative; height: 80vh"> <!-- ① -->
  <img id="pet" src="/dog1.jpg" height="100%" />
  <canvas
    id="detection"
    style="position: absolute; left: 0;"
  ><canvas/> <!-- ② -->
</div>
```

① 包含的`div`是相对定位的，并且锁定在页面高度的 80%处。

② 画布以绝对位置放置在图像上。

对于简单的矩形，您可以使用画布上下文的`strokeRect`方法。`strokeRect`方法不像模型返回的那样需要两个点。它需要一个起点，然后是宽度和高度。要将模型点转换为宽度和高度，您只需减去每个顶点以获得距离。图 5-8 显示了这种计算的可视化表示。

![计算宽度和高度的解释](img/ltjs_0508.png)

###### 图 5-8。宽度和高度是 X 和 Y 之间的差异计算

使用起点、覆盖矩形的宽度和高度，您可以用几行代码在画布上按比例绘制它。记住，张量输出是一个百分比，需要在每个维度上进行缩放。

```js
// Draw box on canvas
const detection = document.getElementById("detection");
const imgWidth = petImage.width;
const imgHeight = petImage.height;
detection.width = imgWidth; // ①
detection.height = imgHeight;
const box = result.dataSync(); // ②
const startX = box[0] * imgWidth; // ③
const startY = box[1] * imgHeight;
const width = (box[2] - box[0]) * imgWidth; // ④
const height = (box[3] - box[1]) * imgHeight;
const ctx = detection.getContext("2d");
ctx.strokeStyle = "#0F0";
ctx.lineWidth = 4;
ctx.strokeRect(startX, startY, width, height); // ⑤
```

① 使检测画布与其所覆盖的图像大小相同。

② 获取边界框结果。

③ 将起点 X 和 Y 缩放回图像。

④ 通过从 X[1]减去 X[2]来找到框的宽度，然后通过图像宽度进行缩放。Y[1]和 Y[2]也是如此。

⑤ 现在使用画布的 2D 上下文来绘制所需的矩形。

结果是在给定点处完美放置的边界框。自己看看图 5-9。

![识别宠物脸部位置](img/ltjs_0509.png)

###### 图 5-9。宠物脸部定位

在运行此项目时的一个误解可能是您经历的检测和绘制很慢。这是错误的。很明显，当页面加载时，边界框出现之前会有延迟；然而，您正在经历的延迟包括加载模型并将其加载到某种加速内存中（有时称为*模型预热*）。尽管这有点超出了本章的目标，但如果您调用`model.predict`并再次绘制，您会在微秒内看到结果。您在本节中创建的画布+TensorFlow.js 结构可以轻松支持桌面计算机上每秒 60 帧以上。

具有大量边界框和标签的模型使用类似的`strokeRect`调用来勾画识别对象的位置。有各种各样的模型，它们各自识别图像的各个方面。在 TensorFlow.js 世界中，修改画布以在图像上绘制信息的实践非常有用。

# 章节回顾

了解模型的输入和输出是关键。在本章中，您最终看到了数据的全部过程。您转换了输入，将其传递给经过训练的模型，并解释了结果。模型可以接受各种各样的输入并提供同样广泛的输出。现在，无论模型需要什么，您都有一些令人印象深刻的经验可以借鉴。

## 章节挑战：可爱的脸

想象一下，我们的宠物脸部定位是更大过程中的第一步。假设您正在识别宠物脸部，然后将宠物脸部传递给另一个模型，该模型将寻找舌头以查看宠物是否发热和喘气。通常会像这样在管道中组织多个模型，每个模型都调整到自己特定的目的。

根据上一段代码中宠物脸部的位置，编写额外的代码来提取宠物的脸并为需要 96 x 96 图像输入的模型做准备。您的答案将是一个批量裁剪，如图 5-10。

![只有狗的脸](img/ltjs_0510.png)

###### 图 5-10。目标[1, 96, 96, 3]张量，只包含脸部

尽管这个练习是为了裁剪宠物的脸以供第二个模型使用，但它也可以很容易地成为一个“宠物匿名化器”，需要您模糊宠物的脸。浏览器中的人工智能应用是无限的。

你可以在附录 B 中找到这个挑战的答案。

## 复习问题

让我们回顾一下你在本章编写的代码中学到的教训。花点时间回答以下问题：

1.  在 TensorFlow.js 中可以加载哪些类型的模型？

1.  你需要知道一个模型被分成了多少个碎片吗？

1.  除了公共 URL 之外，还有哪些地方可以加载模型？

1.  `loadLayersModel` 返回什么？

1.  如何清除已加载模型的内存？

1.  Inception v3 模型的预期输入形状是什么？

1.  使用哪个画布上下文方法可以绘制一个空矩形？

1.  从 TFHub 加载模型时，你必须向加载方法传递哪个参数？

这些问题的解决方案可以在附录 A 中找到。

¹ 你不一定要使用画布；如果你愿意，你可以移动一个 DOM 对象，但是画布提供了简单和复杂的动画，速度很快。
