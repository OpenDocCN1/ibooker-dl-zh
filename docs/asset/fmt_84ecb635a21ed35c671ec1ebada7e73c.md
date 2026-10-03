
```js
async function makePrediction(canvas, model) {
  const drawTensor = tf.browser.fromPixels(canvas, 1) // ①
  const resize = tf.image.resizeNearestNeighbor(drawTensor, [28,28], true) // ②

  const batched = resize.expandDims() // ③
  const results = await model.predict(batched)
  const predictions = await results.array()

  // Display
  displayResults(predictions[0]) // ④
  // Cleanup
  tf.dispose([drawTensor, resize, batched, results])
}
```

①

当你读取画布时，不要忘记标识你只对单个通道感兴趣；否则，你需要在继续之前将 3D 张量转换为 1D 张量。

②

使用最近邻算法将图像调整为 28 x 28 的大小，以输入到模型中。最近邻引起的像素化在这里并不重要，所以这是一个明智的选择，因为它比 `resizeBilinear` 更快。

③

模型期望一个批次数据，所以准备数据作为一个批次的数据。这将创建一个 `[1, 28, 28, 1]` 的输入张量。

④

预测结果已经作为一个包含 10 个数字的批次返回到 JavaScript。想出一种创造性的方式来展示结果。

现在你已经得到了结果，你可以以任何你喜欢的格式展示答案。我个人是按照房间组织了分数，并用它们来设置标签的不透明度。这样，你可以在每画一条线时得到反馈。标签的不透明度取决于值 `0-1`，这与 softmax 预测结果非常契合。

```js
function displayResults(predictions) {
  // Get Scores
  const ravenclaw = predictions[0] + predictions[2] + predictions[3]
  const gryffindor = predictions[1] + predictions[9]
  const hufflepuff = predictions[4] + predictions[8]
  const slytherin = predictions[6] + predictions[7]
  const deatheater = predictions[5]

  document.getElementById("ravenclaw").style.opacity = ravenclaw
  document.getElementById("gryffindor").style.opacity = gryffindor
  document.getElementById("hufflepuff").style.opacity = hufflepuff
  document.getElementById("slytherin").style.opacity = slytherin

  // Harry Potter fans will enjoy this one
  if (deatheater > 0.9) {
    alert('DEATH EATER DETECTED!!!')
  }
}
```

你可能会想知道拉文克劳是否有轻微的数学优势，因为它由更多类别组成，你是对的。在所有条件相同的情况下，一组完全随机的线更有可能被分类为拉文克劳，因为它拥有大多数类别。然而，当图像不是随机的时，这在统计上是不显著的。如果你希望模型只有九个类别，可以移除 `bird` 并重新训练，以创建最平衡的分类谱。

###### 提示

如果你有兴趣确定哪些类别可能存在问题或混淆，你可以使用视觉报告工具，如混淆矩阵或 [t-SNE](https://oreil.ly/sBio5) 算法。这些工具对于评估训练数据特别有帮助。

我强烈建议你从 [*chapter10/simple/simplest-draw*](https://oreil.ly/emOWR) 加载本章的代码，并测试一下你的艺术技能！我的鸟类绘画将我分类到了拉文克劳，如 图 10-15 所示。

![网页正确识别出一只鸟](img/ltjs_1015.png)

###### 图 10-15\. 一个 UI 和绘画杰作

我能够糟糕地画出并被正确分类到其他可能的房间中。然而，我不会再用我的“艺术”来惩罚你。

# 章节回顾

你已经在视觉数据上训练了一个模型。虽然这个数据集仅限于灰度图，但你所学到的经验可以适用于任何图像数据集。有很多优秀的图像数据集可以与你创建的模型完美配合。我们将在接下来的两章中详细介绍。

我为本章中的 [绘画识别特色](https://oreil.ly/jnlhb) 创建了一个更加复杂的页面。

## 章节挑战：保存魔法

如果您最感兴趣的是获得最高验证准确性模型，那么您的最佳模型版本很可能不是最后一个版本。例如，如果您查看图 10-16，90.3%的验证准确性会丢失，最终验证模型为 89.6%。

对于本章挑战，与其保存模型的最终训练版本，不如添加一个回调函数，当验证准确性达到新的最佳记录时保存模型。这种代码非常有用，因为它允许您运行多个时期。随着模型过拟合，您将能够保留最佳的通用模型用于生产。

![验证与训练准确性](img/ltjs_1016.png)

###### 图 10-16。评估哪个准确性更重要

您可以在附录 B 中找到此挑战的答案。

## 复习问题

让我们回顾一下你在本章编写的代码中学到的教训。花点时间回答以下问题：

1.  卷积层有许多可训练的*什么*，可以帮助提取图像的特征？

1.  控制卷积窗口大小的属性名称是什么？

1.  如果你希望卷积结果与原始图像大小相同，应该将填充设置为什么？

1.  真或假？在将图像插入卷积层之前，必须将其展平。

1.  在 81 x 81 图像上，步幅为 3 的最大池 3 x 3 的输出大小将是多少？

1.  如果您要对数字 12 进行独热编码，您是否有足够的信息来这样做？

这些练习的解决方案可以在附录 A 中找到。

¹ YouTube 上的 [3Blue1Brown](https://oreil.ly/zuGzT) 的视频和讲座是任何想要深入了解卷积的人的绝佳起点。

² TensorFlow.js 中还有其他可用于实验的池化方法。
