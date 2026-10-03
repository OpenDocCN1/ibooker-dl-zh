
# 一元和二元操作与张量连接

一元和二元操作是张量输入张量输出（TITO）的，它们以一个或多个张量作为输入，并返回一个张量作为输出。TensorFlow.js 中的一些常用操作不是 TITO，因为它们将张量与另一个非张量参数一起作为输入。`tf.concat()` 可能是这类函数中最常用的函数。它允许你将多个形状兼容的张量连接成一个单一的张量。只有当张量的形状满足某些约束条件时才能进行连接。例如，可以将 `[5, 3]` 张量和 `[4, 3]` 张量沿第一个轴合并以得到 `[9, 3]` 张量，但如果它们的形状分别为 `[5, 3]` 和 `[4, 2]`，则无法将它们组合在一起！在给定的形状合法性的情况下，你可以使用 `tf.concat()` 函数来连接张量。例如，以下代码沿第一个轴连接一个全零 `[2, 2]` 张量和一个全一 `[2, 2]` 张量，则得到一个 `[4, 2]` 张量，其中“顶部”是全零，而“底部”则是全一：

```js
> const x = tf.zeros([2, 2]);
> const y = tf.ones([2, 2]);
> tf.concat([x, y]).print();
Tensor
    [[0, 0],
     [0, 0],
     [1, 1],
     [1, 1]]
```

由于两个输入张量的形状相同，可以以不同的方式对它们进行连接：即沿第二个轴进行连接。轴可以作为第二个输入参数传递给 `tf.concat()`。这将给我们一个 `[2, 4]` 张量，在这个张量中，左半部分都是零，右半部分都是一：

```js
> tf.concat([x, y], 1).print();
Tensor
    [[0, 0, 1, 1],
     [0, 0, 1, 1]]
```

## 张量切片

除了将多个张量连接成一个以外，有时我们希望执行“反向”操作，检索张量的一部分。例如，假设你创建了一个形状为 `[3, 2]` 的二维张量（矩阵），

```js
> const x = tf.randomNormal([3, 2]);
> x.print();
Tensor
    [[1.2366893 , 0.6011682 ],
     [-1.0172369, -0.5025602],
     [-0.6265425, -0.0009868]]
```

而你想要获取矩阵的第二行。为此，可以使用 `tf.slice()` 的链式版本：

```js
> x.slice([1, 0], [1, 2]).print();
Tensor
     [[-1.0172369, -0.5025602],]
```

`slice()` 的第一个参数指示我们想要的输入张量部分从第一个维度的索引 1 和第二个维度的索引 0 开始。换句话说，它应该从第二行和第一列开始，因为我们在这里处理的二维张量是一个矩阵。第二个参数指定所需输出的形状：`[1, 2]` 或在矩阵语言中，1 行 2 列。

如您所看到的，通过查看打印的值，我们成功地检索了 3×2 矩阵的第二行。输出的形状与输入的秩相同（2），但第一个维度的大小为 1。在这种情况下，我们检索第二个维度的全部（所有列）和第一个维度的子集（行的一部分）。这是一种特殊情况，可以使用更简单的语法实现相同的效果：

```js
> x.slice(1, 1).print();
Tensor
     [[-1.0172369, -0.5025602],]
```

在这个更简单的语法中，我们只需要指定请求的块沿第一个维度的起始索引和大小。如果将第二个输入参数传递为 1 而不是 2，则输出将包含矩阵的第一行和第二行：

```js
> x.slice(1, 2).print();
Tensor
    [[-1.0172369, -0.5025602],
     [-0.6265425, -0.0009868]]
```

正如你可能猜到的那样，这种更简单的语法与批处理约定有关。它使得从批处理张量中获取单个示例的数据更容易。

但是如果我们想要访问矩阵的 *列* 呢？在这种情况下，我们将不得不使用更复杂的语法。例如，假设我们想要矩阵的第二列。可以通过以下方式实现

```js
> x.slice([0, 1], [-1, 1]).print();
Tensor
    [[0.6011682 ],
     [-0.5025602],
     [-0.0009868]]
```

这里，第一个参数（`[0, 1]`）是表示我们想要的切片的起始索引的数组。它是沿第一维的第一个索引和第二维的第二个索引。更简单地说，我们希望我们的切片从第一行和第二列开始。第二个参数（`[-1, 1]`）指定了我们想要的切片的大小。第一个数字（-1）表示我们想要沿第一维的所有索引（我们想要所有起始行），而第二个数字（1）表示我们只想要沿第二维的一个索引（我们只想要一列）。结果是矩阵的第二列。

看一下 `slice()` 的语法，你可能已经意识到 `slice()` 不仅限于检索行或列。事实上，如果开始索引和大小数组被正确指定，它足够灵活，可以让你检索输入二维张量中的任何“子矩阵”（矩阵内的任何连续矩形区域）。更一般地，对于秩大于 0 的张量，`slice()` 允许你检索输入张量中的任何连续子张量。我们将这留作附录末尾的练习。

## 张量堆叠与解堆

除了 `tf.slice()` 和 `tf.concat()`，另外两个经常用于将张量分割成部分或将多个张量合并成一个的操作是 `tf.unstack()` 和 `tf.stack()`。 `tf.unstack()` 将张量沿着第一维分割成多个“pieces”。每个片段在第一维上的尺寸为 1。例如，我们可以使用 `tf.unstack()` 的链式 API：

```js
> const x = tf.tensor2d([[1, 2], [3, 4], [5, 6]]);
> x.print();
Tensor
    [[1, 2],
     [3, 4],
     [5, 6]]
> const pieces = x.unstack();
> console.log(pieces.length);
  3
> pieces[0].print();
Tensor
    [1, 2]
> pieces[1].print();
Tensor
    [3, 4]
> pieces[2].print();
Tensor
    [5, 6]
```

你可能已经注意到，`unstack()` 返回的“pieces”比输入张量的秩少一。

`tf.stack()` 是 `tf.unstack()` 的反向操作。顾名思义，它将具有相同形状的多个张量“堆叠”到一个新张量中。根据先前的示例代码片段，我们将片段重新堆叠在一起：

```js
> tf.stack(pieces).print();
Tensor
    [[1, 2],
     [3, 4],
     [5, 6]]
```

`tf.unstack()` 用于从批处理张量中获取与各个示例对应的数据；`tf.stack()` 用于将各个示例的数据合并成一个批处理张量。

# B.3. TensorFlow.js 中的内存管理：`tf.dispose()` 和 `tf.tidy()`

在 TensorFlow.js 中，如果你直接处理张量对象，你需要对它们执行内存管理。特别是在创建和使用张量后，张量需要被释放，否则它将继续占用分配给它的内存。如果未释放的张量数量过多或者总大小过大，它们最终将导致浏览器标签页耗尽 WebGL 内存或导致 Node.js 进程耗尽系统或 GPU 内存（取决于是否使用 tfjs-node 的 CPU 或 GPU 版本）。TensorFlow.js 不会自动对用户创建的张量进行垃圾回收。[[5]](#app02fn5) 这是因为 JavaScript 不支持对象终结。TensorFlow.js 提供了两个内存管理函数：`tf.dispose()` 和 `tf.tidy()`。

> ⁵

> 

> 然而，在 TensorFlow.js 函数和对象方法中创建的张量由库本身管理，因此你不需要担心在调用这些函数或方法时包装它们在 `tf.tidy()` 中。其中的示例函数包括 `tf.confusionMatrix()`、`tf.Model.predict()` 和 `tf.Model.fit()`。

例如，考虑使用 `for` 循环对 TensorFlow.js 模型进行重复推理的示例：

```js
const model = await tf.loadLayersModel(                                    ***1***
    'https://storage.googleapis.com/tfjs-models/tfjs/iris_v1/model.json'); ***1***
const x = tf.randomUniform([1, 4]);                                        ***2***
for (let i = 0; i < 3; ++i) {
  const y = model.predict(x);
  y.print();
  console.log(`# of tensors: ${tf.memory().numTensors}` );                 ***3***
}
```

+   ***1*** 从网络上加载预先训练好的模型

+   ***2*** 创建一个虚拟输入张量

+   ***3*** 检查当前已分配张量的数量

输出将如下所示

```js
Tensor
     [[0.4286409, 0.4692867, 0.1020722],]
# of tensors: 14
Tensor
     [[0.4286409, 0.4692867, 0.1020722],]
# of tensors: 15
Tensor
     [[0.4286409, 0.4692867, 0.1020722],]
# of tensors: 16
```

正如你在控制台日志中看到的那样，每次调用 `model.predict()` 都会生成一个额外的张量，在迭代结束后不会被释放。如果允许 `for` 循环运行足够数量的迭代，它最终会导致内存不足错误。这是因为输出张量 `y` 没有被正确释放，导致张量内存泄漏。有两种方法可以修复这个内存泄漏。

在第一种方法中，你可以在不再需要输出张量时调用 `tf.dispose()`：

```js
for (let i = 0; i < 3; ++i) {
  const y = model.predict(x);
  y.print();
  tf.dispose(y);                                          ***1***
  console.log(`# of tensors: ${tf.memory().numTensors}` );
}
```

+   ***1*** 在使用后释放输出张量

在第二种方法中，你可以在 `for` 循环的主体部分使用 `tf.tidy()`：

```js
for (let i = 0; i < 3; ++i) {
  tf.tidy(() => {                          ***1***
    const y = model.predict(x);
    y.print();
    console.log(`# of tensors: ${tf.memory().numTensors}` );
  });
}
```

+   ***1*** `tf.tidy()` 自动释放传递给它的函数中创建的所有张量，除了由该函数返回的张量。

无论选用哪种方法，你应该看到迭代中分配的张量数量变为常数，表明不再有张量内存泄漏。哪种方法应该优先使用呢？通常，你应该使用 `tf.tidy()`（第二种方法），因为它消除了跟踪需要释放哪些张量的需要。`tf.tidy()` 是一个智能函数，它释放传递给它的匿名函数中创建的所有张量（除了由该函数返回的张量-稍后再说），即使这些张量没有绑定到任何 JavaScript 对象。例如，假设我们稍微修改先前的推理代码，以便使用 `argMax()` 获得获胜类别的索引：

```js
const model = await tf.loadLayersModel(
    'https://storage.googleapis.com/tfjs-models/tfjs/iris_v1/model.json');
const x = tf.randomUniform([1, 4]);
for (let i = 0; i < 3; ++i) {
  const winningIndex =
           model.predict(x).argMax().dataSync()[0];
  console.log(`winning index: ${winningIndex}`);
  console.log(`# of tensors: ${tf.memory().numTensors}` );
}
```

当这段代码运行时，你会发现它每次迭代泄漏两个张量：

```js
winning index: 0
# of tensors: 15
winning index: 0
# of tensors: 17
winning index: 0
# of tensors: 19
```

为什么每次迭代泄漏两个张量？因为这行代码：

```js
   const winningIndex =
       model.predict(x).argMax().dataSync()[0];
```

生成两个新的张量。第一个是 `model.predict()` 的输出，第二个是 `argMax()` 的返回值。这两个张量都没有绑定到任何 JavaScript 对象上。它们被创建后立即使用。这两个张量在某种意义上“丢失”——没有 JavaScript 对象可供您用来引用它们。因此，`tf.dispose()` 不能用于清理这两个张量。但是，`tf.tidy()` 仍然可以用来修复内存泄漏，因为它会对新张量执行簿记，无论它们是否绑定到 JavaScript 对象上：

```js
const model = await tf.loadLayersModel(
    'https://storage.googleapis.com/tfjs-models/tfjs/iris_v1/model.json');
const x = tf.randomUniform([1, 4]);
for (let i = 0; i < 3; ++i) {
  tf.tidy(() => {                                                  ***1***
    const winningIndex = model.predict(x).argMax().dataSync()[0];
    console.log(`winning index: ${winningIndex}`);
    console.log(`# of tensors: ${tf.memory().numTensors}` );       ***1***
  });
}
```

+   ***1*** `tf.tidy()` 会自动处理传递给它作为参数的匿名函数中创建的张量，即使这些张量没有绑定到 JavaScript 对象上。

`tf.tidy()` 的示例用法操作的是不返回任何张量的函数。如果该函数返回张量，则不希望将它们处理掉，因为它们需要在后续使用。这种情况在使用 TensorFlow.js 提供的基本张量操作编写自定义张量操作时经常遇到。例如，假设我们想编写一个函数来计算输入张量的标准化值——即，减去平均值并将标准差缩放为 1 的张量：

```js
function normalize(x) {
  const mean = x.mean();
  const sd = x.norm(2);
  return x.sub(mean).div(sd);
}
```

这个实现有什么问题？^([6]) 就内存管理而言，它泄漏了三个张量：1）均值，2）SD 和 3）一个更微妙的泄漏：`sub()` 调用的返回值。为了修复内存泄漏问题，我们将函数体包装在 `tf.tidy()` 中：

> ⁶

> 

> 这个实现还有其他问题。例如，它没有对输入张量进行健全性检查，以确保它至少有两个元素，使 SD 不为零，否则将导致除以零和无限结果。但是这些问题与此处的讨论无直接关联。

```js
function normalize(x) {
  return tf.tidy(() => {
    const mean = x.mean();
    const sd = x.norm(2);
    return x.sub(mean).div(sd);
       });
}
```

在这里，`tf.tidy()` 为我们完成了三个操作：

+   它会自动处理在匿名函数中创建但未被它返回的张量，包括之前提到的所有泄漏。我们在之前的例子中已经看到这一点了。

+   它检测到 `div()` 调用的输出由匿名函数返回，因此将其转发到自己的返回值。

+   在此期间，它将避免处理该特定张量，以便它可以在 `tf.tidy()` 调用之外使用。

如我们所见，`tf.tidy()` 是一种智能而强大的内存管理函数。它在 TensorFlow.js 代码库中被广泛使用。在本书的示例中，您还将经常看到它。然而，它有以下重要限制：作为参数传递给 `tf.tidy()` 的匿名函数 *不* 能是异步的。如果您有一些需要内存管理的异步代码，您应该使用 `tf.dispose()` 并手动跟踪待处理的张量。在这种情况下，您可以使用 `tf.memory().numTensor` 来检查泄漏张量的数量。一个好的做法是编写单元测试来断言不存在内存泄漏。

# B.4. 计算梯度

这一部分适用于对在 TensorFlow.js 中执行导数和梯度计算感兴趣的读者。对于本书中的大多数深度学习模型，导数和梯度的计算都是在 `model.fit()` 和 `model.fitDataset()` 中自动完成的。然而，对于某些问题类型，比如在第七章中找到卷积滤波器的最大激活图像和在第十一章中的 RL，需要显式地计算导数和梯度。TensorFlow.js 提供了支持此类用例的 API。让我们从最简单的情景开始，即一个接受单个输入张量并返回单个输出张量的函数：
