
1. 在`package.json`中，将`@tensorflow/tfjs-node`依赖项替换为`@tensorflow/tfjs-node-gpu`，但保持相同的版本号，因为这两个软件包的发布是同步的。

2. 再次运行`yarn`，这将下载包含 TensorFlow.js 用于 CUDA 数学运算的共享库。

3. 在`main.js`中，将该行替换为

   ```js
   require('@tensorflow/tfjs-node');
   ```

   使用

   ```js
   require('@tensorflow/tfjs-node-gpu');
   ```

4. 再次开始训练

   ```js
   node main.js
   ```

如果步骤正确完成，你的模型将在 CUDA GPU 上迅猛地训练，在速度上通常是 CPU 版本（`tfjs-node`）的五倍。与在浏览器中训练相同模型相比，使用`tfjs-node`的 CPU 或 GPU 版本训练速度都显著提高。

## 在 tfjs-node 中为 MNIST 训练增强的 convnet

一旦在 20 个周期内完成训练，模型应该显示出约 99.6% 的最终测试（或评估）准确度，这超过了我们在 section 4.2 中取得的 99.0% 的先前结果。那么，导致准确度提高的是这个基于节点的模型和基于浏览器的模型之间的区别是什么呢？毕竟，如果你使用训练数据在`tfjs-node`和 TensorFlow.js 的浏览器版本中训练相同的模型，你应该得到相同的结果（除了随机权重初始化的影响）。为了回答这个问题，让我们来看看基于节点的模型的定义。模型是在文件`model.js`中构建的，这个文件由`main.js`导入。

### 列表 4.5. 在 Node.js 中定义一个更大的 MNIST convnet

```js
const model = tf.sequential();
model.add(tf.layers.conv2d({
  inputShape: [28, 28, 1],
  filters: 32,
  kernelSize: 3,
  activation: 'relu',
}));
model.add(tf.layers.conv2d({
  filters: 32,
  kernelSize: 3,
  activation: 'relu',
}));
model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
model.add(tf.layers.conv2d({
  filters: 64,
  kernelSize: 3,
  activation: 'relu',
}));
model.add(tf.layers.conv2d({
  filters: 64,
  kernelSize: 3,
  activation: 'relu',
}));
model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
model.add(tf.layers.flatten());
model.add(tf.layers.dropout({rate: 0.25}));                    ***1***
model.add(tf.layers.dense({units: 512, activation: 'relu'}));
model.add(tf.layers.dropout({rate: 0.5}));
model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

model.summary();
model.compile({
  optimizer: 'rmsprop',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});
```

- ***1*** 添加了 dropout 层以减少过拟合

模型的摘要如下：

```js
_________________________________________________________________
Layer (type)                 Output shape              Param #
=================================================================
conv2d_Conv2D1 (Conv2D)      [null,26,26,32]           320
_________________________________________________________________
conv2d_Conv2D2 (Conv2D)      [null,24,24,32]           9248
_________________________________________________________________
max_pooling2d_MaxPooling2D1  [null,12,12,32]           0
_________________________________________________________________
conv2d_Conv2D3 (Conv2D)      [null,10,10,64]           18496
_________________________________________________________________
conv2d_Conv2D4 (Conv2D)      [null,8,8,64]             36928
_________________________________________________________________
max_pooling2d_MaxPooling2D2  [null,4,4,64]             0
_________________________________________________________________
flatten_Flatten1 (Flatten)   [null,1024]               0
_________________________________________________________________
dropout_Dropout1 (Dropout)   [null,1024]               0
_________________________________________________________________
dense_Dense1 (Dense)         [null,512]                524800
_________________________________________________________________
dropout_Dropout2 (Dropout)   [null,512]                0
_________________________________________________________________
dense_Dense2 (Dense)         [null,10]                 5130
=================================================================
Total params: 594922
Trainable params: 594922
Non-trainable params: 0
_________________________________________________________________
```

这些是我们的`tfjs-node`模型和基于浏览器的模型之间的关键区别：

- 基于节点的模型具有四个`conv2d`层，比基于浏览器的模型多一个。
- 基于节点的模型的 hidden dense 层比基于浏览器的模型的对应层单元更多（512 与 100）。
- 总体而言，基于节点的模型的权重参数约为基于浏览器的模型的 18 倍。
- 基于节点的模型在`flatten`层和`dense`层之间插入了两个`dropout`层。

列表中的前三个差异使基于节点的模型具有比基于浏览器的模型更高的容量。这也是使基于节点的模型在浏览器中训练速度无法接受的原因。正如我们在第三章中学到的那样，更大的模型容量意味着更大的过拟合风险。第四个差异增加了过拟合风险的减轻，即包括 dropout 层。

## 使用 dropout 层来减少过拟合

Dropout 是你在本章中遇到的另一个新的 TensorFlow.js 层类型之一。它是减少深层神经网络过拟合最有效和广泛使用的方式之一。它的功能可以简单地描述为：

- 在训练阶段（在`Model.fit()`调用期间），它随机将输入张量的一部分作为零（或“丢失”），并且其输出张量是 dropout 层的输出张量。对于本示例来说，dropout 层只有一个配置参数：dropout 比例（例如，如列表 4.5 所示的两个`rate`字段）。例如，假设一个 dropout 层被配置为有一个 0.25 的 dropout 比例，并且输入张量是值为`[0.7, -0.3, 0.8, -0.4]`的 1D 张量，则输出张量可以是`[0.7, -0.3, 0.0, 0.4]`，其中 25% 的输入张量元素随机选择并设置为值 0。在反向传播期间，dropout 层的梯度张量也会受到类似的归 0 影响。

- 在推理期间（在`Model.predict()`和`Model.evaluate()`调用期间），dropout 层不会随机将输入张量中的元素置零。相反，输入会简单地作为输出传递，不会发生任何改变（即，一个恒等映射）。

图 4.11 展示了一个带有二维输入张量的 dropout 层在训练和测试时的工作示例。

### 图 4.11. dropout 层的一个示例

在这个示例中，输入张量是 2D 的，形状为`[4, 2]`。dropout 层的比例被配置为 0.25，导致在训练阶段随机选择输入张量中 25%（即 8 个中的两个）的元素并将它们设置为零。在推理阶段，该层充当一个简单的传递层。

![](img/04fig10.jpg)

这种简单算法是对抗过拟合最有效的方法之一似乎很奇怪。为什么它有效呢？Geoff Hinton，是 dropout 算法的发明者（神经网络中的许多其他内容也是他的创造），他说他受到了一些银行用于防止员工欺诈的机制的启发。用他自己的话说，

> *我去了我的银行。出纳员不停地换，我问其中一个为什么。他说他不知道，但他们经常被调动。我想这一定是因为需要员工之间的合作才能成功欺诈银行。这让我意识到，随机移除每个示例上的不同子集神经元将防止共谋，从而减少过拟合。*

将这种深度学习的术语引入，向层的输出值引入噪声会破坏不重要的偶然模式，这些模式与数据中的真实模式不相关（Hinton 称之为“共谋”）。在本章末尾的练习 3 中，您应该尝试从`model.js`中的基于节点的卷积网络中移除两个 dropout 层，然后再次训练模型，并查看由此导致的训练、验证和评估准确度的变化。

清单 4.6 显示了我们用于训练和评估增强型卷积网络的关键代码。如果您将此处的代码与清单 4.2 中的代码进行比较，您就会欣赏到这两个代码块之间的相似之处。两者都围绕着`Model.fit()`和`Model.evaluate()`调用。语法和样式相同，只是在如何呈现或显示损失值、准确度值和训练进度上有所不同（终端与浏览器）。

这展示了 TensorFlow.js 的一个重要特性，这是一个跨越前端和后端的 JavaScript 深度学习框架：

> *就创建和训练模型而言，在 TensorFlow.js 中编写的代码与您是在 web 浏览器还是在 Node.js 中工作无关。*

### 列表 4.6. 在 tfjs-node 中训练和评估增强型卷积网络

```js
  await model.fit(trainImages, trainLabels, {
    epochs,
    batchSize,
    validationSplit
  });

  const {images: testImages, labels: testLabels} = data.getTestData();
  const evalOutput = model.evaluate                                     ***1***
      testImages, testLabels);
  console.log('\nEvaluation result:');
  console.log(
      `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; `+
      `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`);
```

- ***1*** 使用模型未见过的数据评估模型

### 4.3.2. 从 Node.js 保存模型并在浏览器中加载

训练模型会消耗 CPU 和 GPU 资源，并需要一些时间。您不希望浪费训练的成果。如果不保存模型，下次运行`main.js`时，您将不得不从头开始。本节展示了如何在训练后保存模型，并将保存的模型导出为磁盘上的文件（称为*检查点*或*工件*）。我们还将向您展示如何在浏览器中导入检查点，将其重新构建为模型，并用于推断。`main.js`中`main()`函数的最后一部分是以下清单中的保存模型代码。

#### 列表 4.7. 在 tfjs-node 中将训练好的模型保存到文件系统中

```js
  if (modelSavePath != null) {
    await model.save(`file://${modelSavePath}`);
    console.log(`Saved model to path: ${modelSavePath}`);
  }
```

`model`对象的`save()`方法用于将模型保存到文件系统上的目录中。该方法接受一个参数，即以`file://`开头的 URL 字符串。请注意，由于我们使用的是`tfjs-node`，所以可以将模型保存在文件系统上。TensorFlow.js 的浏览器版本也提供了`model.save()`API，但不能直接访问机器的本地文件系统，因为浏览器出于安全原因禁止了这样做。如果我们在浏览器中使用 TensorFlow.js，则必须使用非文件系统保存目标（例如浏览器的本地存储和 IndexedDB）。这些对应于`file://`以外的 URL 方案。

`model.save()`是一个异步函数，因为它通常涉及文件或网络输入输出。因此，在`save()`调用上使用`await`。假设`modelSavePath`的值为`/tmp/tfjs-node-mnist`；在`model.save()`调用完成后，您可以检查目录的内容，

```js
ls -lh /tmp/tfjs-node-mnist
```

这可能打印出类似以下的文件列表：

```js
-rw-r--r-- 1 user group 4.6K Aug 14 10:38 model.json
     -rw-r--r-- 1 user group 2.3M Aug 14 10:38 weights.bin
```

在那里，你可以看到两个文件：

- `model.json`是一个包含模型保存拓扑的 JSON 文件。这里所说的“拓扑”包括形成模型的层类型、它们各自的配置参数（例如卷积层的`filters`和 dropout 层的`rate`），以及层之间的连接方式。对于 MNIST 卷积网络来说，连接是简单的，因为它是一个顺序模型。我们将看到连接模式不太平凡的模型，这些模型也可以使用`model.save()`保存到磁盘上。

- 除了模型拓扑，`model.json`还包含模型权重的清单。该部分列出了所有模型权重的名称、形状和数据类型，以及权重值存储的位置。这将我们带到第二个文件：`weights.bin`。正如其名称所示，`weights.bin`是一个存储所有模型权重值的二进制文件。它是一个没有标记的平面二进制流，没有标明个体权重的起点和终点。这些“元信息”在`model.json`中的权重清单部分中可用于。

要使用`tfjs-node`加载模型，您可以使用`tf.loadLayersModel()`方法，指向`model.json`文件的位置（未在示例代码中显示）：

```js
const loadedModel = await tf.loadLayersModel('file:///tmp/tfjs-node-mnist');
```

`tf.loadLayersModel()`通过反序列化`model.json`中保存的拓扑数据来重建模型。然后，`tf.loadLayersModel()`使用`model.json`中的清单读取`weights.bin`中的二进制权重值，并将模型的权重强制设置为这些值。与`model.save()`一样，`tf.loadLayersModel()`是异步的，所以我们在这里调用它时使用`await`。一旦调用返回，`loadedModel`对象在所有意图和目的上等同于使用 listings 4.5 和 4.6 中的 JavaScript 代码创建和训练的模型。你可以通过调用其`summary()`方法打印模型的摘要，通过调用其`predict()`方法执行推理，通过使用`evaluate()`方法评估其准确性，甚至通过使用`fit()`方法重新训练它。如果需要，也可以再次保存模型。重新训练和重新保存加载的模型的工作流程将在我们讨论第五章的迁移学习时相关。
