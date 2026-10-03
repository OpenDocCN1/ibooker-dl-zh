
+   `data.js` — 下载和访问波士顿房价数据集所需的结构的实现
+   `ui.js` — 实现将 UI 元素与操作连接的 UI 钩子的文件；绘图配置的规范
+   `normalization.js` — 数值例程，例如从数据中减去均值
+   `package.json` — 标准的 npm 包定义，描述了构建和运行此演示所需的依赖项（例如 TensorFlow.js！）

请注意，我们不遵循将 HTML 文件和 JavaScript 文件放在特定类型的子目录中的标准做法。这种模式在更大的存储库中是最佳做法，但对于我们将在本书中使用的较小示例或您可以在 [github.com/tensorflow/tfjs-examples](http://github.com/tensorflow/tfjs-examples) 找到的示例，它更多地是混淆而不是澄清。

要运行此演示，请使用 Yarn：

```js
yarn && yarn watch
```

这将在您的浏览器中打开一个指向 `localhost` 上的端口的新标签，该端口将运行示例。如果您的浏览器没有自动反应，可以在命令行中导航到输出的 URL。点击标记为“Train Linear Regressor”的按钮将触发构建线性模型并将其拟合到波士顿房价数据的过程，然后在每个周期后输出训练和测试数据集的损失的动态图表，如图 2.11 所示。

##### 图 2.11。tfjs-examples 中的波士顿房价线性回归示例

![](img/02fig11_alt.jpg)

本节的其余部分将介绍构建这个波士顿房价线性回归 Web 应用演示的重要要点。我们首先将回顾数据是如何收集和处理的，以便与 TensorFlow.js 一起使用。然后我们将重点关注模型的构建、训练和评估；最后，我们将展示如何在网页上使用模型进行实时预测。

#### 2.3.3\. 访问波士顿房价数据

在我们的第一个项目中，在清单 2.1 中，我们将数据硬编码为 JavaScript 数组，并使用`tf.tensor2d`函数将其转换为张量。硬编码对于小型演示来说没问题，但显然不适用于更大的应用程序。一般来说，JavaScript 开发人员会发现他们的数据位于某个 URL（可能是本地）的某种序列化格式中。例如，波士顿房屋数据以 CSV 格式公开且免费提供，可以从 Google Cloud 的以下 URL 中获取：

+   [`storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/train-data.csv`](https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/train-data.csv)
+   [`storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/train-target.csv`](https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/train-target.csv)
+   [`storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/test-data.csv`](https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/test-data.csv)
+   [`storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/test-target.csv`](https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/test-target.csv)

数据已经通过将样本随机分配到训练和测试部分而进行了预拆分。大约有三分之二的样本在训练拆分中，剩下的三分之一用于独立评估经过训练的模型。此外，对于每个拆分，目标特征已经与其他特征分开成为 CSV 文件，导致了表 2.2 中列出的四个文件名。

##### 表 2.2\. 波士顿房屋数据集的拆分和内容的文件名

|  |  | 特征（12 个数字） | 目标（1 个数字） |
| --- | --- | --- | --- |
| **训练-测试拆分** | **训练** | `train-data.csv` | `train-target.csv` |
|  | **测试** | `test-data.csv` | `test-target.csv` |

为了将这些数据引入我们的应用程序，我们需要能够下载这些数据并将其转换为适当类型和形状的张量。波士顿房屋项目在 `data.js` 中定义了一个名为`BostonHousingDataset`的类，用于此目的。该类抽象了数据集流操作，提供了一个 API 来检索原始数据作为数字矩阵。在内部，该类使用了公共开源 Papa Parse 库（[www.papaparse.com](http://www.papaparse.com)）来流式传输和解析远程 CSV 文件。一旦文件已加载和解析，库就会返回一个数字数组的数组。然后，使用与第一个示例中相同的 API 将其转换为张量，如下清单所示，这是`index.js`中的一个略微简化的示例，重点放在相关部分上。

##### 清单 2.7\. 在 index.js 中将波士顿房屋数据转换为张量

```js
// Initialize a BostonHousingDataset object defined in data.js.
const bostonData = new BostonHousingDataset();
const tensors = {};

// Convert the loaded csv data, of type number[][] into 2d tensors.
export const arraysToTensors = () => {
  tensors.rawTrainFeatures = tf.tensor2d(bostonData.trainFeatures);
  tensors.trainTarget = tf.tensor2d(bostonData.trainTarget);
  tensors.rawTestFeatures = tf.tensor2d(bostonData.testFeatures);
  tensors.testTarget = tf.tensor2d(bostonData.testTarget);
}

// Trigger the data to load asynchronously once the page has loaded.
let tensors;
document.addEventListener('DOMContentLoaded', async () => {
  await bostonData.loadData();
  arraysToTensors();
}, false);
```

#### 2.3.4\. 精确定义波士顿房屋问题

现在我们可以以我们想要的形式访问我们的数据，现在是时候更准确地澄清我们的任务了。我们说我们想要从其他字段预测 MEDV，但是我们将如何确定我们的工作是否做得好呢？我们如何区分一个好模型和一个更好的模型呢？

我们在第一个例子中使用的度量标准`meanAbsoluteError`将所有错误都视为平等。如果只有 10 个样本，并且我们对所有 10 个样本进行预测，并且我们在其中的第 10 个样本上完全正确，但在其他 9 个样本上偏差为 30，则`meanAbsoluteError`将为 3（因为 30/10 为 3）。如果我们的预测对每个样本都偏差为 3，那么`meanAbsoluteError`仍然为 3。这个“错误的平等性”原则可能似乎是唯一显然正确的选择，但是选择除`meanAbsoluteError`之外的损失度量有很好的理由。

另一种选择是将大错误的权重赋予小错误。我们可以不是取绝对误差的平均值，而是取*平方*误差的平均值。

在进行有关这 10 个样本的案例研究时，这种均方误差（MSE）方法看到了在每个示例上偏差为 3 时（10 × 3² = 90）比在一个示例上偏差为 30 时（1 × 30² = 900）较低的损失。由于对大错误的敏感性，平方误差比绝对误差更敏感于样本异常值。将模型拟合以最小化 MSE 的优化器将更喜欢系统地犯小错误的模型，而不是偶尔给出非常糟糕估计的模型。显然，这两种错误度量都会更喜欢根本没有错误的模型！但是，如果您的应用可能对非常不正确的异常值敏感，那么 MSE 可能比 MAE 更好。选择 MSE 或 MAE 的其他技术原因，但它们在此时并不重要。在本例中，我们将使用 MSE 来增加变化，但 MAE 也足够。

在我们继续之前，我们应该找到损失的基准估计。如果我们不知道从一个非常简单的估计中得出的误差，那么我们就没有能力从一个更复杂的模型中评估它。我们将使用平均房地产价格作为我们的“最佳天真猜测”，并计算总是猜测该值时的误差。

##### 列表 2.8\. 计算猜测平均价格的基线损失

```js
export const computeBaseline = () => {
  const avgPrice = tf.mean(tensors.trainTarget);             ***1***
  console.log(`Average price: ${avgPrice.dataSync()[0]}`);

  const baseline =
      tf.mean(tf.pow(tf.sub(
          tensors.testTarget, avgPrice), 2));                ***2***

  console.log(
      `Baseline loss: ${baseline.dataSync()[0]}`);           ***3***
};
```

+   ***1*** 计算平均价格
+   ***2*** 计算测试数据上的平均平方误差。`sub()`、`pow` 和 `mean()` 调用是计算平均平方误差的步骤。
+   ***3*** 打印出损失值

因为 TensorFlow.js 通过在 GPU 上进行调度来优化其计算，所以张量可能并不总是可供 CPU 访问。在列表 2.8 中对`dataSync`的调用告诉 TensorFlow.js 完成张量的计算，并将值从 GPU 拉到 CPU 中，以便可以打印出来或以其他方式与非 TensorFlow 操作共享。

当执行时，列表 2.8 中的代码将在控制台中产生以下输出：

```js
Average price: 22.768770217895508
Baseline loss: 85.58282470703125
```

这告诉我们，天真的误差率大约为 85.58。如果我们构建一个总是输出 22.77 的模型，该模型在测试数据上将达到 85.58 的 MSE。再次注意，我们在训练数据上计算指标，并在测试数据上评估它，以避免不公平的偏见。

平均*平方*误差为 85.58，所以我们应该取平方根得到平均误差。85.58 的平方根大约是 9.25。因此，我们可以说我们期望我们的（常量）估计平均偏离（上下）约 9.25。根据表 2.1 的数值，以千美元为单位，估计一个常量意味着我们会偏离约 9,250 美元。如果这对我们的应用程序足够好，我们可以停止！明智的机器学习从业者知道何时避免不必要的复杂性。让我们假设我们的价格估计应用程序需要比这更接近。我们将通过拟合我们的数据来查看是否可以获得比 85.58 更好的 MSE 的线性模型。

#### 2.3.5。稍微偏离数据标准化

查看波士顿房屋的特征，我们会看到各种值。NOX 的范围在 0.4 到 0.9 之间，而 TAX 则从 180 到 711。为了拟合线性回归，优化器将尝试找到每个特征的权重，使特征的累加乘以权重大约等于房屋价格。请记住，为了找到这些权值，优化器正在寻找，遵循权重空间中的梯度。如果某些特征与其他特征的比例相差很大，那么某些权重将比其他权重敏感得多。向一个方向的一个非常小的移动将比另一个方向的一个非常大的移动更改输出。这可能导致不稳定，并使得难以拟合模型。

为了对抗这一点，我们将首先*标准化*我们的数据。这意味着我们将缩放我们的特征，使它们的平均值为零，标准差为单位。这种标准化方法很常见，也可以被称为*标准转换*或*z-score 标准化*。做这种操作的算法很简单——我们首先计算每个特征的平均值，并从原始值中减去，使得该特征的平均值为零。然后我们计算特征的标准差与减去的平均值，并进行除法。在伪代码中，

```js
normalizedFeature = (feature - mean(feature)) / std(feature)
```

例如，当特征是`[10, 20, 30, 40]`时，标准化后的版本大约是`[-1.3, -0.4, 0.4, 1.3]`，很明显的平均值为零；肉眼看，标准差大约为一。在波士顿房屋的例子中，标准化代码被分解到一个单独的文件中，`normalization.js`，其内容在列表 2.9 中。在这里，我们看到两个函数，一个用于计算所提供的二维张量的平均值和标准差，另一个用于在提供预先计算的平均值和标准差的情况下标准化张量。

##### 列表 2.9。数据规范化：零均值，单位标准差

```js
/**
 * Calculates the mean and standard deviation of each column of an array.
 *
 * @param {Tensor2d} data Dataset from which to calculate the mean and
 *                        std of each column independently.
 *
 * @returns {Object} Contains the mean and std of each vector
 *                   column as 1d tensors.
 */
export function determineMeanAndStddev(data) {
  const dataMean = data.mean(0);
  const diffFromMean = data.sub(dataMean);
  const squaredDiffFromMean = diffFromMean.square();
  const variance = squaredDiffFromMean.mean(0);
  const std = variance.sqrt();
  return {mean, std};
}
/**
 * Given expected mean and standard deviation, normalizes a dataset by
 * subtracting the mean and dividing by the standard deviation.
 *
 * @param {Tensor2d} data: Data to normalize.
 *    Shape: [numSamples, numFeatures].
 * @param {Tensor1d} mean: Expected mean of the data. Shape [numFeatures].
 * @param {Tensor1d} std: Expected std of the data. Shape [numFeatures]
 *
 * @returns {Tensor2d}: Tensor the same shape as data, but each column
 * normalized to have zero mean and unit standard deviation.
 */
export function normalizeTensor(data, dataMean, dataStd) {
  return data.sub(dataMean).div(dataStd);
}
```

让我们稍微深入一下这些函数。函数`determineMeanAndStddev`将`data`作为输入，这是一个秩 2 张量。按照惯例，第一个维度是*样本*维度：每个索引对应一个独立，唯一的样本。第二个维度是*特征*维度：其 12 个元素对应于 12 个输入特征（如 CRIM，ZN，INDUS 等）。由于我们要独立计算每个特征的平均值，因此调用

```js
const dataMean = data.mean(0);
```

这个调用中的`0`表示平均值是在第 0 维度（第一维度）上计算的。记得`data`是一个二维张量，因此有两个维度（或轴）。第一个轴，即“批处理”轴，是样本维度。当我们沿着该轴从第一个到第二个到第三个元素移动时，我们引用不同的样本，或者在我们的情况下，不同的房地产部分。第二个维度是特征维度。当我们在该维度的第一个元素移动到第二个元素时，我们引用不同的特征，例如 CRIM，ZN 和 INDUS，来自表 2.1。当我们沿轴 0 取平均值时，我们正在沿样本方向取平均值。结果是具有仅保留特征轴的秩 1 张量。我们拥有每个特征的平均值。如果我们改为沿轴 1 取平均值，我们仍会得到一个秩 1 张量，但剩余轴将是样本维度。这些值将对应于每个房地产部分的平均值，这在我们的应用程序中没有意义。在使用轴进行计算时，请注意在正确方向上进行计算，因为这是常见的错误来源。

果然，如果我们在这里设置一个断点^([6])，我们可以使用 JavaScript 控制台来探索计算出的平均值，我们看到的平均值非常接近我们为整个数据集计算的值。这意味着我们的训练样本是代表性的：

> ⁶
> 
