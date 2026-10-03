
+ `SparseCategoricalCrossentropy`
+ `BinaryCrossentropy`
+ `MeanSquaredError`
+ `KLDivergence`
+ `CosineSimilarity`
+ 等等。

指标：

+ `CategoricalAccuracy`
+ `SparseCategoricalAccuracy`
+ `BinaryAccuracy`
+ `AUC`
+ `Precision`
+ `Recall`
+ 等等。

在本书中，您将看到许多这些选项的具体应用。

### 3.6.4 选择损失函数

为正确的问题选择正确的损失函数非常重要：你的网络会尽其所能缩小损失，因此如果目标与当前任务的成功并不完全相关，你的网络最终可能会执行一些你不希望的操作。想象一下，通过使用这个选择不当的目标函数（“最大化所有活着人类的平均幸福感”）进行 SGD 训练的愚蠢、全能的 AI。为了简化工作，这个 AI 可能选择杀死除少数人外的所有人类，并专注于剩下人的幸福感——因为平均幸福感不受剩余人数的影响。这可能不是你想要的结果！请记住，你构建的所有神经网络都会像这样无情地降低它们的损失函数，因此明智地选择目标，否则你将面临意想不到的副作用。

幸运的是，对于常见问题如分类、回归和序列预测，你可以遵循简单的准则来选择正确的损失函数。例如，对于两类分类问题，你将使用二元交叉熵，对于多类分类问题，你将使用分类交叉熵，依此类推。只有在处理真正新的研究问题时，你才需要开发自己的损失函数。在接下来的几章中，我们将明确详细地介绍为各种常见任务选择哪些损失函数。

### 3.6.5 理解 `fit()` 方法

在 `compile()` 之后是 `fit()`。`fit()` 方法实现了训练循环本身。以下是它的关键参数：

+ 用于训练的*数据*（输入和目标）。通常会以 NumPy 数组或 TensorFlow `Dataset` 对象的形式传递。你将在接下来的章节中更多地了解 `Dataset` API。
+ 训练的*轮数*：训练循环应该迭代传递的数据多少次。
+ 在每个迷你批次梯度下降的 epoch 中使用的批次大小：用于计算一次权重更新步骤的训练示例数量。

第 3.23 节 使用 NumPy 数据调用 `fit()`

```py
history = model.fit(
    inputs,          # ❶
    targets,         # ❷
    epochs=5,        # ❸
    batch_size=128   # ❹
)
```

❶ 输入示例，作为 NumPy 数组

❷ 相应的训练目标，作为 NumPy 数组

❸ 训练循环将在数据上迭代 5 次。

❹ 训练循环将以 128 个示例的批次迭代数据。

调用 `fit()` 返回一个 `History` 对象。该对象包含一个 `history` 字段，它是一个将诸如 `"loss"` 或特定指标名称映射到每个 epoch 值列表的字典。

```py
>>> history.history
{"binary_accuracy": [0.855, 0.9565, 0.9555, 0.95, 0.951],
 "loss": [0.6573270302042366,
          0.07434618508815766,
          0.07687718723714351,
          0.07412414988875389,
          0.07617757616937161]}
```

### 3.6.6 监控验证数据上的损失和指标

机器学习的目标不是获得在训练数据上表现良好的模型，这很容易——你只需遵循梯度。目标是获得在一般情况下表现良好的模型，特别是在模型从未遇到过的数据点上表现良好。仅仅因为一个模型在训练数据上表现良好并不意味着它会在从未见过的数据上表现良好！例如，你的模型可能最终只是*记忆*训练样本和它们的目标之间的映射，这对于预测模型从未见过的数据的目标是无用的。我们将在第五章中更详细地讨论这一点。

为了监视模型在新数据上的表现，通常会将训练数据的一个子集保留为*验证数据*：你不会在这些数据上训练模型，但会用它们来计算损失值和指标值。你可以通过在 `fit()` 中使用 `validation_data` 参数来实现这一点。与训练数据类似，验证数据可以作为 NumPy 数组或 TensorFlow `Dataset` 对象传递。

第 3.24 节 使用 `validation_data` 参数

```py
model = keras.Sequential([keras.layers.Dense(1)])
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.1),
              loss=keras.losses.MeanSquaredError(),
              metrics=[keras.metrics.BinaryAccuracy()])

indices_permutation = np.random.permutation(len(inputs))        # ❶
shuffled_inputs = inputs[indices_permutation]                   # ❶
shuffled_targets = targets[indices_permutation]                 # ❶

num_validation_samples = int(0.3 * len(inputs))                 # ❷
val_inputs = shuffled_inputs[:num_validation_samples]           # ❷
val_targets = shuffled_targets[:num_validation_samples]         # ❷
training_inputs = shuffled_inputs[num_validation_samples:]      # ❷
training_targets = shuffled_targets[num_validation_samples:]    # ❷
model.fit(
    training_inputs,                                            # ❸
    training_targets,                                           # ❸
    epochs=5,
    batch_size=16,
    validation_data=(val_inputs, val_targets)                   # ❹
)
```

❶ 为了避免在验证数据中只有一个类的样本，使用随机索引排列来对输入和目标进行洗牌。

❷ 保留 30%的训练输入和目标用于验证（我们将排除这些样本进行训练，并保留它们来计算验证损失和指标）。

❸ 用于更新模型权重的训练数据

❹ 仅用于监控验证损失和指标的验证数据

在验证数据上的损失值称为“验证损失”，以区别于“训练损失”。请注意，保持训练数据和验证数据严格分开是至关重要的：验证的目的是监测模型学习的内容是否实际上对新数据有用。如果模型在训练过程中看到任何验证数据，您的验证损失和指标将是有缺陷的。

请注意，如果您想在训练完成后计算验证损失和指标，可以调用 `evaluate()` 方法：

```py
loss_and_metrics = model.evaluate(val_inputs, val_targets, batch_size=128)
```

`evaluate()` 将在传递的数据上以批量（大小为 `batch_size`）迭代，并返回一个标量列表，其中第一个条目是验证损失，后一个条目是验证指标。如果模型没有指标，只返回验证损失（而不是列表）。

### 3.6.7 推断：在训练后使用模型

一旦您训练好模型，您将想要使用它在新数据上进行预测。这被称为*推断*。为此，一个简单的方法就是简单地 `__call__()` 模型：

```py
predictions = model(new_inputs)    # ❶
```

❶ 接受一个 NumPy 数组或 TensorFlow 张量，并返回一个 TensorFlow 张量

然而，这将一次性处理 `new_inputs` 中的所有输入，如果你要处理大量数据可能不可行（特别是可能需要比你的 GPU 更多的内存）。

进行推断的更好方法是使用 `predict()` 方法。它将以小批量迭代数据，并返回一个预测的 NumPy 数组。与 `__call__()` 不同，它还可以处理 TensorFlow 的 `Dataset` 对象。

```py
predictions = model.predict(new_inputs, batch_size=128)     # ❶
```

❶ 接受一个 NumPy 数组或数据集，并返回一个 NumPy 数组

例如，如果我们对之前训练过的线性模型使用 `predict()` 在一些验证数据上，我们会得到对应于模型对每个输入样本的预测的标量分数：

```py
>>> predictions = model.predict(val_inputs, batch_size=128)
>>> print(predictions[:10])
[[0.3590725 ]
 [0.82706255]
 [0.74428225]
 [0.682058  ]
 [0.7312616 ]
 [0.6059811 ]
 [0.78046083]
 [0.025846  ]
 [0.16594526]
 [0.72068727]]
```

目前，这就是您需要了解的关于 Keras 模型的全部内容。您已经准备好在下一章节中使用 Keras 解决真实世界的机器学习问题了。

## 摘要

+ TensorFlow 是一个工业强度的数值计算框架，可以在 CPU、GPU 或 TPU 上运行。它可以自动计算任何可微表达式的梯度，可以分布到许多设备，还可以将程序导出到各种外部运行时，甚至 JavaScript。
+ Keras 是使用 TensorFlow 进行深度学习的标准 API。这是我们将在整本书中使用的。
+ TensorFlow 的关键对象包括张量、变量、张量操作和梯度带。
+ Keras 的核心类是 `Layer`。一个*层*封装了一些权重和一些计算。层被组装成*模型*。
+ 在开始训练模型之前，您需要选择一个*优化器*，一个*损失*和一些*指标*，您可以通过 `model.compile()` 方法指定。
+ 要训练一个模型，您可以使用 `fit()` 方法，它为您运行小批量梯度下降。您还可以使用它来监视您在*验证数据*上的损失和指标，这是模型在训练过程中没有看到的一组输入。
+ 一旦您的模型训练完成，您可以使用 `model.predict()` 方法在新输入上生成预测。
