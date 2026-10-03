
如果将 NMS 的 IoU 设置为 0.5，则任何与得分更高的框共享 50% 面积的框将被删除。这对于消除与同一对象重叠的框非常有效。但是，对于彼此重叠并应该有两个边界框的两个对象可能会出现问题。这对于具有不幸角度的真实对象是一个问题，因为它们的边界框将相互抵消，您只会得到两个实际对象的一个检测。对于这种情况，您可以启用一个名为 [Soft-NMS](https://arxiv.org/pdf/1704.04503.pdf) 的 NMS 的高级版本，它将降低重叠框的分数而不是删除它们。如果它们在被降低后的分数仍然足够高，检测结果将存活并获得自己的边界框，即使 IoU 非常高。图 6-9 使用 Soft-NMS 正确识别了与极高交集的两个对象。

![Soft-NMS 示例](img/ltjs_0609.png)

###### 图 6-9。即使是真实世界中重叠的对象也可以使用 Soft-NMS 进行检测

Soft-NMS 最好的部分是它内置在 TensorFlow.js 中。我建议您为所有目标检测需求使用这个 TensorFlow.js 函数。在这个练习中，您将使用内置的方法，名为 `tf.image.nonMaxSuppressionWithScoreAsync`。TensorFlow.js 内置了许多 NMS 算法，但 `tf.image.nonMaxSuppressionWithScoreAsync` 具有两个优点，使其非常适合使用：

+   `WithScore` 提供 Soft-NMS 支持。
+   `Async` 可以阻止 GPU 锁定 UI 线程。

在使用非异步高级方法时要小心，因为它们可能会锁定整个 UI。如果出于任何原因想要移除 Soft-NMS 方面，可以将最后一个参数（Soft-NMS Sigma）设置为零，然后您就得到了传统的 NMS。

```js
const nmsDetections = await tf.image.nonMaxSuppressionWithScoreAsync(
  justBoxes, // shape [numBoxes, 4]
  justValues, // shape [numBoxes]
  maxBoxes, // Stop making boxes when this number is hit
  iouThreshold, // Allowed overlap value 0 to 1
  detectionThreshold, // Minimum detection score allowed
  1 // 0 is normal NMS, 1 is max Soft-NMS
);
```

只需几行代码，您就可以将 SSD 结果澄清为几个清晰的检测结果。

结果将是一个具有两个属性的对象。`selectedIndices` 属性将是一个张量，其中包含通过筛选的框的索引，`selectedScores` 将是它们对应的分数。您可以循环遍历所选结果并绘制边界框。

```js
const chosen = await nmsDetections.selectedIndices.data(); // ①
chosen.forEach((detection) => {
  ctx.strokeStyle = "#0F0";
  ctx.lineWidth = 4;
  const detectedIndex = maxIndices[detection]; // ②
  const detectedClass = CLASSES[detectedIndex]; // ③
  const detectedScore = scores[detection];
  const dBox = boxes[detection];
  console.log(detectedClass, detectedScore); // ④

  // No negative values for start positions
  const startY = dBox[0] > 0 ? dBox[0] * imgHeight : 0; // ⑤
  const startX = dBox[1] > 0 ? dBox[1] * imgWidth : 0;
  const height = (dBox[2] - dBox[0]) * imgHeight;
  const width = (dBox[3] - dBox[1]) * imgWidth;
  ctx.strokeRect(startX, startY, width, height);
});
```

① 从结果中高得分的框的索引创建一个普通的 JavaScript 数组。

② 从先前的 `topk` 调用中获取最高得分的索引。

③ 将类别作为数组导入以匹配给定的结果索引。这种结构就像上一章中 Inception 示例中的代码一样。

④ 记录在画布中被框定的内容，以便验证结果。

⑤ 禁止负数，以便框至少从帧开始。否则，一些框将从左上角被切断。

返回的检测数量各不相同，但受限于 NMS 中设置的规格。示例代码导致了五个正确的检测结果，如图 6-10 所示。

![非最大抑制结果](img/ltjs_0610.png)

###### 图 6-10。干净的 Soft-NMS 检测结果

循环中的控制台日志打印出五个检测结果分别为三个“人”检测、一个“酒杯”和一个“餐桌”。将图 6-11 中的五个日志与图 6-10 中的五个边界框进行比较。

![非最大抑制结果日志](img/ltjs_0611.png)

###### 图 6-11。结果日志类别和置信水平

UI 已经取得了很大进展。覆盖层应该能够识别检测和它们的置信度百分比是合理的。普通用户不知道要查看控制台以查看日志。

# 添加文本覆盖

您可以以各种花式方式向画布添加文本，并使其识别相关的边界框。在此演示中，我们将回顾最简单的方法，并将更美观的布局留给读者作为任务。

可以使用画布 2D 上下文的 `fillText` 方法向画布绘制文本。您可以通过重复使用绘制框时使用的 `X, Y` 坐标将文本定位在每个框的左上角。

有两个绘制文本时需要注意的问题：

+   文本与背景之间很容易出现低对比度。
+   与同时绘制的框相比，同时绘制的文本可能会被后绘制的框覆盖。

幸运的是，这两个问题都很容易解决。

## 解决低对比度

创建可读标签的典型方法是绘制一个背景框，然后放置文本。正如您所知，`strokeRect` 创建一个没有填充颜色的框，所以不应该感到意外的是 `fillRect` 绘制一个带有填充颜色的框。

矩形应该有多大？一个简单的答案是将矩形绘制到检测框的宽度，但不能保证框足够宽，当框非常宽时，这会在结果中创建大的阻挡条。唯一有效的解决方案是测量文本并相应地绘制框。文本高度可以通过利用上下文 `font` 属性来设置，宽度可以通过 `measureText` 确定。

最后，您可能需要考虑从绘图位置减去字体高度，以便将文本绘制在框内而不是在框上方，但上下文已经有一个属性可以设置以保持简单。`context.textBaseline` 属性有各种选项。图 6-12 显示了每个可能属性选项的起始点。

![文本基线选项](img/ltjs_0612.png)

###### 图 6-12。将 `textBaseline` 设置为 `top` 可以保持文本在 X 和 Y 坐标内

现在你知道如何绘制一个填充矩形到适当的大小并将标签放在内部。您可以将这些方法结合在您的 `forEach` 循环中，您在其中绘制检测结果。标签绘制在每个检测的左上角，如图 6-13 所示。

![显示绘制结果](img/ltjs_0613.png)

###### 图 6-13。标签与每个框一起绘制

重要的是文本在背景框之后绘制，否则框将覆盖文本。对于我们的目的，标签将使用略有不同颜色的绿色绘制，而不是边界框。

```js
// Draw the label background.
ctx.fillStyle = "#0B0";
ctx.font = "16px sans-serif"; // ①
ctx.textBaseline = "top"; // ②
const textHeight = 16;
const textPad = 4; // ③
const label = `${detectedClass} ${Math.round(detectedScore * 100)}%`;
const textWidth = ctx.measureText(label).width;
ctx.fillRect(  // ④
  startX,
  startY,
  textWidth + textPad,
  textHeight + textPad
);
// Draw the text last to ensure it's on top.
ctx.fillStyle = "#000000";  // ⑤
ctx.fillText(label, startX, startY); // ⑥
```

① 设置标签使用的字体和大小。

② 设置 `textBaseline` 如上所述。

③ 添加一点水平填充以在 `fillRect` 渲染中使用。

④ 使用相同的 `startX` 和 `startY` 绘制矩形，这与绘制边界框时使用的相同。

⑤ 将 `fillStyle` 更改为黑色以进行文本渲染。

⑥ 最后，绘制文本。这可能也应该略微填充。

现在每个检测都有一个几乎可读的标签。但是，根据您的图像，您可能已经注意到了一些问题，我们现在将解决。

## 解决绘制顺序

尽管标签是绘制在框的上方，但框是在不同的时间绘制的，可以轻松重叠一些现有标签文本，使它们难以阅读甚至不可能阅读。如您在图 6-14 中所见，由于重叠检测，餐桌百分比很难阅读。

![上下文重叠问题](img/ltjs_0614.png)

###### 图 6-14。上下文绘制顺序重叠问题

解决这个问题的一种方法是遍历检测结果并绘制框，然后再进行第二次遍历并绘制文本。这将确保文本最后绘制，但代价是需要在两个连续循环中遍历检测结果。

作为替代方案，您可以使用代码处理这个问题。您可以设置上下文 `globalCompositeOperation` 来执行各种令人惊奇的操作。一个简单的操作是告诉上下文在现有内容的上方或下方渲染，有效地设置 z 顺序。

`strokeRect` 调用可以设置为 `globalCompositeOperation` 为 `destination-over`。这意味着任何存在于目标中的像素将获胜并放置在添加的内容上方。这有效地在任何现有内容下绘制。

然后，在绘制标签时，将 `globalCompositionOperation` 返回到其默认行为，即 `source-over`。这会将新的源像素绘制在任何现有绘图上。如果在这两种操作之间来回切换，您可以确保您的标签是最优先的，并在主循环内处理所有内容。

总的来说，绘制边界框、标签框和标签的单个循环如下所示：

```js
chosen.forEach((detection) => {
  ctx.strokeStyle = "#0F0";
  ctx.lineWidth = 4;
  ctx.globalCompositeOperation='destination-over'; // ①
  const detectedIndex = maxIndices[detection];
  const detectedClass = CLASSES[detectedIndex];
  const detectedScore = scores[detection];
  const dBox = boxes[detection];

  // No negative values for start positions
  const startY = dBox[0] > 0 ? dBox[0] * imgHeight : 0;
  const startX = dBox[1] > 0 ? dBox[1] * imgWidth : 0;
  const height = (dBox[2] - dBox[0]) * imgHeight;
  const width = (dBox[3] - dBox[1]) * imgWidth;
  ctx.strokeRect(startX, startY, width, height);
  // Draw the label background.
  ctx.globalCompositeOperation='source-over'; // ②
  ctx.fillStyle = "#0B0";
  const textHeight = 16;
  const textPad = 4;
  const label = `${detectedClass} ${Math.round(detectedScore * 100)}%`;
  const textWidth = ctx.measureText(label).width;
  ctx.fillRect(
    startX,
    startY,
    textWidth + textPad,
    textHeight + textPad
  );
  // Draw the text last to ensure it's on top.
  ctx.fillStyle = "#000000";
  ctx.fillText(label, startX, startY);
});
```

① 在任何现有内容下绘制。

② 在任何现有内容上绘制。

结果是一个动态的人类可读结果，您可以与您的朋友分享（参见图 6-15）。

![完全工作的目标检测](img/ltjs_0615.png)

###### 图 6-15。使用 `destination-over` 修复重叠问题

# 连接到网络摄像头

所有这些速度的好处是什么？正如前面提到的，选择 SSD 而不是 R-CNN，选择 MobileNet 而不是 Inception，以及一次绘制画布而不是两次。当你加载页面时，它看起来相当慢。似乎至少需要四秒才能加载和渲染。

是的，把一切都放在适当的位置需要一点时间，但在内存分配完毕并且模型下载完成后，你会看到一些相当显著的速度。是的，足以在你的网络摄像头上运行实时检测。

加快流程的关键是运行设置代码一次，然后继续运行检测循环。这意味着你需要将这节课的庞大代码库分解；否则，你将得到一个无法使用的界面。为简单起见，你可以按照示例 6-1 中所示分解项目。

##### 示例 6-1。分解代码库

```js
async function doStuff() {
  try {
    const model = await loadModel()  // ①
    const mysteryVideo = document.getElementById('mystery') // ②
    const camDetails = await setupWebcam(mysteryVideo) // ③
    performDetections(model, mysteryVideo, camDetails) // ④
  } catch (e) {
    console.error(e) // ⑤
  }
}
```

① 加载模型时最长的延迟应该首先发生，且仅发生一次。

② 为了效率，你可以一次捕获视频元素，并将该引用传递到需要的地方。

③ 设置网络摄像头应该只发生一次。

④ `performDetections` 方法可以在检测网络摄像头中的内容并绘制框时无限循环。

⑤ 不要让所有这些 `awaits` 吞没错误。

## 从图像到视频的转换

从静态图像转换为视频实际上并不复杂，因为将所见内容转换为张量的困难部分由 `tf.fromPixels` 处理。`tf.fromPixels` 方法可以读取画布、图像，甚至视频元素。因此，复杂性在于将 `img` 标签更改为 `video` 标签。

你可以通过更换标签来开始。原始的 `img` 标签：

```js
<img id="mystery" src="/dinner.jpg" height="100%" />
```

变成以下内容：

```js
<video id="mystery" height="100%" autoplay></video>
```

值得注意的是，视频元素的宽度/高度属性稍微复杂，因为有输入视频的宽度/高度和实际客户端的宽度/高度。因此，所有使用 `width` 的计算都需要使用 `clientWidth`，同样，`height` 需要使用 `clientHeight`。如果使用错误的属性，框将不对齐，甚至可能根本不显示。

## 激活网络摄像头

为了我们的目的，我们只会设置默认的网络摄像头。这对应于示例 6-1 中的第四点。如果你对 `getUserMedia` 不熟悉，请花点时间分析视频元素如何连接到网络摄像头。这也是你可以将画布上下文设置移动到适应视频元素的时间。
