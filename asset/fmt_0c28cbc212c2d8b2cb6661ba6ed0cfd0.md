
调整图像张量的大小以用于输入是大多数模型的常见做法。这意味着任何与期望输入严重不成比例的图像，如全景照片，当调整大小以用于输入时可能表现糟糕。

TensorFlow.js 有两种优秀的方法用于调整图像大小，并且两者都支持图像批处理：`image.resizeNearestNeighbor`和`image.resizeBilinear`。我建议您在进行任何视觉调整时使用`image.resizeBilinear`，并将`image.resizeNearestNeighbor`保留用于当图像的特定像素值不能被破坏或插值时。速度上有一点小差异，`image.resizeNearestNeighbor`比`image.resizeBilinear`快大约 10 倍，但差异仍然以每次调整的毫秒数来衡量。

直白地说，`resizeBilinear`会模糊，而`resizeNearestNeighbor`会像素化，当它们需要为新数据进行外推时。让我们使用这两种方法放大图像并进行比较。您可以在[*simple/simple-image-manipulation/resize.html*](https://oreil.ly/ieQLD)中查看此示例。

```js
// Simple Tensor Flip
const newSize = [768, 560] // 4x larger // ①
const littleGantImage = document.getElementById("littleGant");
const nnCanvas = document.getElementById("nnCanvas");
const blCanvas = document.getElementById("blCanvas");
const gantTensor = tf.browser.fromPixels(littleGantImage);

const nnResizeTensor = tf.image.resizeNearestNeighbor( // ②
  gantTensor,
  newSize,
  true // ③
)
tf.browser.toPixels(nnResizeTensor, nnCanvas).then(() => {
  nnResizeTensor.dispose();
})

const blResizeTensor = tf.image.resizeBilinear( // ④
  gantTensor,
  newSize,
  true // ⑤
)
const blResizeTensorInt = blResizeTensor.asType('int32') // ⑥
tf.browser.toPixels(blResizeTensorInt, blCanvas).then(() => {
  blResizeTensor.dispose();
  blResizeTensorInt.dispose();
})

// All done with ya
gantTensor.dispose();
```

① 将图像大小增加 4 倍，以便您可以看到这两者之间的差异。

② 使用最近邻算法调整大小。

③ 第三个参数是`alignCorners`；请始终将其设置为 true。¹

④ 使用双线性算法调整大小。

⑤ 始终将此设置为`true`（参见*3*）。

⑥ 截至目前，`resizeBilinear`返回一个`float32`，你需要进行转换。

如果你仔细观察图 4-9 中的结果，你会看到最近邻的像素呈现锐利的像素化，而双线性的呈现柔和的模糊效果。

![调整大小方法](img/ltjs_0409.png)

###### 图 4-9\. 使用调整大小方法的表情符号（有关图像许可证，请参见附录 C）

###### 警告

使用最近邻算法调整大小可能会被恶意操纵。如果有人知道你最终的图像尺寸，他们可以构建一个看起来只在那个调整大小时不同的邪恶图像。这被称为*对抗性预处理*。更多信息请参见[*https://scaling-attacks.net*](https://scaling-attacks.net)。

如果你想看到鲜明对比，你应该尝试使用两种方法调整本章开头创建的 4 x 3 图像的大小。你能猜到哪种方法会在新尺寸上创建一个棋盘格，哪种方法不会吗？

## 裁剪图像张量

在我们最后一轮的基本图像张量任务中，我们将裁剪一幅图像。我想指出，就像我们之前的镜像练习一样，有一种适用于批量裁剪大量图像的版本，称为`image.cropAndResize`。知道这种方法的存在，你可以利用它来收集和规范化图像的部分用于训练，例如，抓取照片中检测到的所有人脸并将它们调整到相同的输入尺寸以供模型使用。

目前，你只需从 3D 张量中裁剪出一些张量数据的简单示例。如果你想象这在空间中，就像从一个更大的矩形蛋糕中切出一个小矩形薄片。

通过给定切片的起始位置和大小，你可以在任何轴上裁剪出你想要的任何部分。你可以在 GitHub 上的[*simple/simple-image-manipulation/crop.html*](https://oreil.ly/QDmBD)找到这个例子。要裁剪单个图像，请使用以下代码：

```js
// Simple Tensor Crop
const startingPoint = [0, 40, 0]; // ①
const newSize = [265, 245, 3]; // ②
const lemonadeImage = document.getElementById("lemonade");
const lemonadeCanvas = document.getElementById("lemonadeCanvas");
const lemonadeTensor = tf.browser.fromPixels(lemonadeImage);

const cropped = tf.slice(lemonadeTensor, startingPoint, newSize) // ③
tf.browser.toPixels(cropped, lemonadeCanvas).then(() => {
  cropped.dispose();
})
lemonadeTensor.dispose();

```

① 从下方`0`像素开始，向右`40`像素，并且在红色通道上。

② 获取接下来的`265`像素高度，`245`像素宽度，以及所有三个 RGB 值。

③ 将所有内容传入`tf.slice`方法。

结果是原始图像的精确裁剪，你可以在图 4-10 中看到。

![使用切片裁剪张量](img/ltjs_0410.png)

###### 图 4-10\. 使用`tf.slice`裁剪单个图像张量

## 新的图像工具

你刚刚学会了三种最重要的图像操作方法，但这并不意味着你的能力有所限制。新的 AI 模型将需要新的图像张量功能，因此，TensorFlow.js 和辅助库不断添加用于处理和处理图像的方法。现在，你可以更加自如地在单个和批量形式中利用和依赖这些工具。

# 章节回顾

从可编辑张量中编码和解码图像使你能够进行逐像素的操作，这是很少有人能做到的。当然，你已经学会了为了我们在 AI/ML 中的目标而学习视觉张量，但事实上，如果你愿意，你可以尝试各种疯狂的图像操作想法。如果你愿意，你可以做以下任何一种：

+   铺设一个你自己设计的像素图案

+   从另一幅图像中减去一幅图像以进行艺术设计

+   通过操纵像素值在图像中隐藏一条消息

+   编写分形代码或其他数学可视化

+   去除背景图像颜色，就像绿幕一样

在本章中，你掌握了创建、加载、渲染、修改和保存大型结构化数据张量的能力。处理图像张量不仅简单，而且非常有益。你已经准备好迎接任何挑战。

## 章节挑战：排序混乱

使用您在本章和之前章节学到的方法，您可以用张量做一些非常令人兴奋和有趣的事情。虽然这个挑战没有我能想到的特定实用性，但它是对您所学内容的有趣探索。作为对所学课程的练习，请思考以下问题：

> 如何生成一个随机的 400 x 400 灰度张量，然后沿一个轴对随机像素进行排序？

如果您完成了这个挑战，生成的张量图像将会像图 4-11 那样。

![一个随机噪声张量排序](img/ltjs_0411.png)

###### 图 4-11\. 沿宽度轴排序的 400 x 400 随机性

您可以使用本书中学到的方法来解决这个问题。如果遇到困难，请查阅[TensorFlow.js 在线文档](https://js.tensorflow.org/api/latest)。在文档中搜索关键词将指引您正确方向。

您可以在附录 B 中找到这个挑战的答案。

## 复习问题

让我们回顾一下您在本章编写的代码中学到的知识。请花点时间回答以下问题：

1.  如果一个图像张量包含值`0-255`，为了正确渲染它需要什么类型的数据？

1.  一个 2 x 2 的红色`Float32`在张量形式中会是什么样子？

1.  `tf.fill([100, 50, 1], 0.2)`会创建什么样的图像张量？

1.  真或假：要保存一个 RGBA 图像，您必须使用一个四阶图像张量。

1.  真或假：`randomUniform`如果给定相同的输入，将会创建相同的输出。

1.  在浏览器中将图像转换为张量应该使用什么方法？

1.  在 Node.js 中对 PNG 进行编码时，第二个参数应该使用什么数字以获得最大压缩？

1.  如果您想要将图像张量上下翻转，您该如何做？

1.  哪个更快？

    1.  循环遍历一组图像并调整它们的大小

    1.  将一组图像作为四阶张量进行批处理并调整整个张量的大小

1.  以下结果的秩和大小是多少：

    ```js
    [.keep-together]#`tf.slice(myTensor, [0,0,0], [20, 20, 3])`?#
    ```

这些练习的解决方案可以在附录 A 中找到。

¹ TensorFlow 对`alignCorners`的实现存在错误，可能会有[问题](https://oreil.ly/Ir9Gy)。
