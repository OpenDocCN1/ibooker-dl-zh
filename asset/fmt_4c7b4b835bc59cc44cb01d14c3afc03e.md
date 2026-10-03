
该模型首先通过将图像输入展平以将它们连接到神经网络，然后有一个`64`和一个`8`单元层。最后一层是九种可能的骰子配置。

这个模型在几个时代内就能达到近乎完美的准确率。这对于生成的数据来说是很有希望的，但在下一节中，我们将看到它在实际图像中的表现如何。

# 网站界面

现在您已经有了一个经过训练的模型，是时候用非生成数据进行测试了。肯定会有一些错误，但如果模型表现得不错，这将是相当成功的！

您的网站需要告诉需要使用多少个骰子，然后将输入图像分成相同数量的块。这些块将被调整大小为 12 x 12 的输入（就像我们的训练数据），然后在图像上运行模型进行预测。在图 12-8 中显示的示例中，一个 X 的图像被告知要转换为四个骰子。因此，图像被切割成四个象限，然后对每个象限进行预测。它们应该理想地将骰子对齐以绘制 X。

![将 TensorFlow 标志切割成 32 x 32 块之前和之后](img/ltjs_1208.png)

###### 图 12-8。将 TensorFlow 标志切割成 32 x 32 块

一旦您获得了预测结果，您可以重建一个由指定图像张量组成的新张量。

###### 注意

这些图像是在 0 和 1 上进行训练的。这意味着，为了期望得到一个体面的结果，您的输入图像也应该由 0 和 1 组成。颜色甚至灰度都会产生虚假的结果。

应用程序代码的核心应该看起来像这样：

```js
const dicify = async () => {
  const modelPath = '/dice-model/model.json'
  const dModel = await tf.loadLayersModel(modelPath)

  const grid = await cutData("input")
  const predictions = await predictResults(dModel, grid)
  await displayPredictions(predictions)

  tf.dispose([dModel, predictions])
  tf.dispose(grid)
}
```

结果的预测是您经典的“数据输入，数据输出”模型行为。最复杂的部分将是`cutData`和`displayPredictions`方法。在这里，您的张量技能将大放异彩。

## 切成块

`cutData`方法将利用`tf.split`，它沿着一个轴将张量分割为 N 个子张量。您可以通过使用`tf.split`沿着每个轴将图像分割成一个补丁或图像网格来进行预测。

```js
const numDice = 32
const preSize = numDice * 10
const cutData = async (id) => {
  const img = document.getElementById(id)
  const imgTensor = tf.browser.fromPixels(img, 1) // ①
  const resized = tf.image.resizeNearestNeighbor( // ②
    imgTensor, [preSize,preSize]
  )
  const cutSize = numDice
  const heightCuts = tf.split(resized, cutSize)   // ③
  const grid = heightCuts.map((sliver) =>         // ④
    tf.split(sliver, cutSize, 1))

  return grid
}
```

①

您只需要将图像的灰度版本从像素转换过来。

②

图像被调整大小，以便可以被所需数量的骰子均匀分割。

③

图像沿着第一个轴（高度）被切割。

④

然后将这些列沿着宽度轴切割，以创建一组张量。

`grid`变量现在包含一个图像数组。在需要时，您可以调整图像大小并堆叠它们进行预测。例如，图 12-9 是一个切片网格，因为 TensorFlow 标志的黑白切割将创建许多较小的图像，这些图像将被转换为骰子。

![将 TensorFlow 标志切割成 27x27 块](img/ltjs_1209.png)

###### 图 12-9。黑白 TensorFlow 标志的切片

## 重建图像

一旦您有了预测结果，您将想要重建图像，但您将希望将原始块替换为它们预测的骰子。

从预测答案重建和创建大张量的代码可能如下所示：

```js
const displayPredictions = async (answers) => {
  tf.tidy(() => {
    const diceTensors = diceData.map( // ①
      (dt) => tf.tensor(dt)
    )
    const { indices } = tf.topk(answers)
    const answerIndices = indices.dataSync()

    const tColumns = []
    for (let y = 0; y < numDice; y++) {
      const tRow = []
      for (let x = 0; x < numDice; x++) {
        const curIndex = y * numDice + x       // ②
        tRow.push(diceTensors[answerIndices[curIndex]])
      }
      const oneRow = tf.concat(tRow, 1)        // ③
      tColumns.push(oneRow)
    }
    const diceConstruct = tf.concat(tColumns)  // ④
    // Print the reconstruction to the canvas
    const can = document.getElementById('display')
    tf.browser.toPixels(diceConstruct, can)    // ⑤
  })
}
```

①

要绘制的`diceTensors`从`diceData`中加载并转换。

②

要从 1D 返回到 2D，需要为每一行计算索引。

③

行是通过沿着宽度轴进行连接而创建的。

④

列是通过沿着默认（高度）轴连接行来制作的。

⑤

哒哒！新构建的张量可以显示出来了。

如果你加载了一个黑白图像并处理它，现在是真相的时刻。每个类别生成了大约 200 张图像是否足够？

我将`numDice`变量设置为 27。一个 27 x 27 的骰子图像是相当低分辨率的，需要在亚马逊上花费大约 80 美元。让我们看看加上 TensorFlow 标志会是什么样子。图 12-10 展示了结果。

![TensorFlow 标志转换为 27 x 27 骰子之前和之后](img/ltjs_1210.png)

###### 图 12-10。TensorFlow 标志转换为 27 x 27 骰子

它有效！一点也不错。你刚刚教会了一个 AI 如何成为一个艺术家。如果你增加骰子的数量，图像会变得更加明显。

# 章节回顾

使用本章的策略，我训练了一个 AI 来处理红白骰子。我没有太多耐心，所以我只为一个朋友制作了一个 19x19 的图像。结果相当令人印象深刻。我花了大约 30 分钟将所有的骰子放入图 12-11 中显示的影子盒中。如果没有印刷说明，我想我不会冒这个风险。

![19 x 19 成品图像。](img/ltjs_1211.png)

###### 图 12-11。完成的 19 x 19 红白骰子带背光

你可以走得更远。哪个疯狂的科学家没有自己的肖像？现在你的肖像可以由骰子制成。也许你可以教一个小机器人如何为你摆放骰子，这样你就可以建造满是数百磅骰子的巨大画框（见图 12-12）。

![一个人看着一堵用骰子做成的图像的墙](img/ltjs_1212.png)

###### 图 12-12。完美的疯狂科学肖像

你可以继续改进数据并获得更好的结果，你不仅仅局限于普通的黑白骰子。你可以利用你的 AI 技能用装饰性骰子、便利贴、魔方、乐高积木、硬币、木片、饼干、贴纸或其他任何东西来绘画。

虽然这个实验对于 1.0 版本来说是成功的，但我们已经确定了无数个实验，可以让你改进你的模型。

## 章节挑战：简单如 01、10、11

现在你有了一个强大的新模型，可以成为由黑色`0`和白色`1`像素组成的任何照片的艺术家。不幸的是，大多数图像，即使是灰度图像，也有中间值。如果有一种方法可以高效地将图像转换为黑白就好了。

将图像转换为二进制黑白被称为*二值化*。计算机视觉领域有各种各样的令人印象深刻的算法，可以最好地将图像二值化。让我们专注于最简单的方法。

在这个章节挑战中，使用`tf.where`方法来检查像素是否超过给定的阈值。使用该阈值，你可以将灰度图像的每个像素转换为`1`或`0`。这将为你的骰子模型准备正常的图形输入。

通过几行代码，你可以将成千上万种光的变化转换为黑白像素，如图 12-13 所示。

![一个头骨被转换成黑白像素。](img/ltjs_1213.png)

###### 图 12-13。二值化的头骨

你可以在附录 B 中找到这个挑战的答案。

## 复习问题

让我们回顾一下你在本章编写的代码中学到的知识。花点时间回答以下问题：

1.  TensorFlow.js 的哪个方法允许你将张量分解为一组相等的子张量？

1.  用于创建数据的稍微修改的替代品以扩大数据集的过程的名称是什么？

1.  为什么 Gant Laborde 如此了不起？

这些练习的解决方案可以在附录 A 中找到。

¹如果你想了解更多关于暗箱的知识，请观看纪录片[*Tim's Vermeer*](https://oreil.ly/IrjNM)。
