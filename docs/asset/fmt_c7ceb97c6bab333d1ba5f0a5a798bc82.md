
`metric_sparse_categorical_crossentropy`

`metric_sparse_top_k_categorical_accuracy`

`metric_specificity_at_sensitivity`

`metric_squared_hinge`

`metric_sum`

`metric_top_k_categorical_accuracy`

`metric_true_negatives`

`metric_true_positives`

在本书中，你将看到许多这些选项的具体应用。

### 3.8.4 选择损失函数

选择正确的损失函数非常重要：你的网络会尽其所能采取任何捷径来最小化损失，因此，如果目标与手头任务的成功不完全相关，你的网络最终会做出你可能不希望的事情。想象一下，通过 SGD 训练的愚蠢的全能 AI，其目标函数选择不当：“最大化所有活着的人的平均幸福感。”为了简化其工作，这个 AI 可能选择杀死除了少数人之外的所有人，并专注于剩下人的幸福感，因为平均幸福感不受剩余人数的影响。这可能不是你想要的结果！请记住，你构建的所有神经网络在降低损失函数方面都会同样无情，所以明智地选择目标，否则你将不得不面对意想不到的副作用。

幸运的是，对于常见问题，如分类、回归和序列预测，您可以遵循简单的指南来选择正确的损失。例如，对于两类分类问题，您将使用二元交叉熵，对于多类分类问题，您将使用分类交叉熵，依此类推。只有当您处理真正的新研究问题时，您才需要开发自己的损失函数。在接下来的几章中，我们将明确详细说明选择哪些损失函数适用于各种常见任务。

### 3.8.5 理解 `fit()` 方法

`compile()` 之后是 `fit()`。`fit()` 方法实现了训练循环本身。以下是其关键参数：

- 训练所需的*数据*（输入和目标）。通常以 R 数组、张量或 TensorFlow 数据集对象的形式传递。您将在接下来的章节中更多了解 `tfdatasets` API。
- 训练的*周期*数：训练循环应该遍历传递的数据的次数。
- 在每个迷你批次梯度下降的周期中使用的批次大小：用于计算一个权重更新步骤的梯度的训练示例数

**清单 3.21 使用 R 数组调用 `fit()`**

```
history <- model %>%
  fit(inputs,➊
      目标,➋
      epochs = 5,➌
      batch_size = 128)➍
```

➊ **输入示例，作为 R 数组**

➋ **相应的训练目标，作为 R 数组**

➌ **训练循环将遍历数据五次。**

➍ **训练循环将遍历 128 个示例的批次数据。**

对 `fit()` 的调用返回一个历史对象。此对象包含一个指标属性，它是每个周期的“损失”和特定指标名称的值的命名列表：

```
str(history$metrics)
长度为 2 的列表
$ 损失               : num [1:5] 14.2 13.6 13.1 12.6 12.1
$ 二元准确度: num [1:5] 0.55 0.552 0.554 0.557 0.559
```

### 3.8.6 监控验证数据上的损失和指标

机器学习的目标不是获得在训练数据上表现良好的模型，这很容易——你只需跟随梯度。目标是获得在一般情况下表现良好的模型，特别是在模型以前从未遇到过的数据点上表现良好。仅仅因为一个模型在训练数据上表现良好，并不意味着它在以前从未见过的数据上也会表现良好！例如，您的模型可能仅仅*记忆*了您的训练样本和它们的目标之间的映射，这对于预测模型以前从未见过的数据的目标将是无用的。在第五章中，我们将更详细地讨论这一点。

为了密切关注模型在新数据上的表现，将一部分训练数据作为*验证数据*是标准做法：您不会在这些数据上训练模型，但会使用它来计算损失值和指标值。您可以通过在 `fit()` 中使用 `validation_data` 参数来实现此目的。像训练数据一样，验证数据可以作为 R 数组或 TensorFlow Dataset 对象传递。

**清单 3.22 使用 `validation_data` 参数**

```
model <- keras_model_sequential() %>%
  layer_dense(1)

model %>% compile(optimizer_rmsprop(learning_rate = 0.1),
                  loss = loss_mean_squared_error(),
                  metrics = metric_binary_accuracy())

n_cases <- dim(inputs)[1]
num_validation_samples <- round(0.3 * n_cases)➊
val_indices <-
  sample.int(n_cases, num_validation_samples)➋
val_inputs <- inputs[val_indices, ]
val_targets <- targets[val_indices, , drop = FALSE]➌
training_inputs <- inputs[-val_indices, ]
training_targets <-
  targets[-val_indices, , drop = FALSE]➌

model %>% fit(
  training_inputs,
  training_targets,➍
  epochs = 5,
  batch_size = 16,
  validation_data = list(val_inputs, val_targets)➎
)
```

➊ **将 30% 的训练输入和目标保留用于验证（我们将排除这些样本进行训练，并保留它们以计算验证损失和指标）。**

➋ **生成 `num_validation_samples` 个随机整数，范围在 `[1, n_cases]`。**

➌ **传递 `drop = FALSE` 以防止 R 数组 `[` 方法删除大小为 1 的维度，而是返回形状为 `(num_validation_samples, 1)` 的数组。**

➍ **训练数据，用于更新模型的权重**

➎ **验证数据，仅用于监控验证损失和指标**

在验证数据上的损失值称为*验证损失*，以区别于*训练损失*。请注意，严格保持训练数据和验证数据分开是至关重要的：验证的目的是监控模型学习的内容是否实际上对新数据有用。如果任何验证数据在训练期间被模型看到，您的验证损失和指标将是错误的。

请注意，如果您想在训练完成后计算验证损失和指标，您可以调用 `evaluate()` 方法：

```
loss_and_metrics <- evaluate(model, val_inputs, val_targets,
                             batch_size = 128)
```

`evaluate()` 将在传递的数据上以批次（批量大小）迭代，并返回数字向量，其中第一个条目是验证损失，后续条目是验证指标。如果模型没有指标，则仅返回验证损失（长度为 1 的 R 向量）。

### 3.8.7 推断：在训练后使用模型

一旦您训练好了模型，您会想要将其用于在新数据上进行预测。这被称为*推断*。为了做到这一点，一个朴素的方法是简单地调用模型：

```
predictions <- model(new_inputs)➊
```

➊ **接受一个 R 数组或 TensorFlow 张量，并返回一个 TensorFlow 张量。**

然而，这将一次性处理 `new_inputs` 中的所有输入，如果你要处理大量数据可能不可行（特别是可能需要比你的 GPU 拥有的内存更多）。

进行推断的更好方法是使用 `predict()` 方法。它将在小批量数据上迭代，并返回一个 R 数组的预测。与调用模型不同的是，它还可以处理 TensorFlow Dataset 对象：

```
predictions <- model %>%
  predict(new_inputs, batch_size = 128)➊
```

➊ **接受一个 R 数组或 TF Dataset 并返回一个 R 数组。**

例如，如果我们对先前训练的线性模型使用 `predict()` 在一些验证数据上，我们会得到对应于模型对每个输入样本的预测的标量分数：

```
predictions <- model %>%
  predict(val_inputs, batch_size = 128)

head(predictions, 10)
          [,1]
[1,]  -0.11416233
[2,]   0.43776459
[3,]  -0.02436411
[4,]  -0.19723934
[5,]  -0.24584538
[6,]  -0.18628466
[7,]  -0.06967193
[8,]   0.19761485
[9,]  -0.28266442
[10,]   0.43299851
```

目前，这就是你需要了解的关于 Keras 模型的全部内容。你已经准备好在下一章中使用 Keras 解决真实世界的机器学习问题了。

## 摘要

- TensorFlow 是一个工业强度的数值计算框架，可以在 CPU、GPU 或 TPU 上运行。它可以自动计算任何可微表达式的梯度，可以分布到多个设备，并且可以导出程序到各种外部运行时——甚至 JavaScript。
- Keras 是使用 TensorFlow 进行深度学习的标准 API。这是本书中我们将使用的内容。
- TensorFlow 的关键对象包括张量、变量、张量操作和梯度带。
- Keras 中的核心类型是 `Layer`。一个*层*封装了一些权重和一些计算。层被组装成*模型*。
- 在开始训练模型之前，你需要选择一个*优化器*，一个*损失*，和一些*度量标准*，你可以通过 `model %>% compile()` 方法来指定。
- 要训练一个模型，你可以使用 `fit()` 方法，它为你运行小批量梯度下降。你还可以用它来监视你在*验证数据*上的损失和指标，验证数据是模型在训练过程中不见的一组输入。
- 一旦你的模型训练好了，你就可以使用 `model %>% predict()` 方法在新输入上生成预测。
