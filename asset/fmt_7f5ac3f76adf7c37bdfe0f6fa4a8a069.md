
```py
from keras.regularizers import l2

model = keras.Sequential(
    [
        layers.Dense(16, kernel_regularizer=l2(0.002), activation="relu"),
        layers.Dense(16, kernel_regularizer=l2(0.002), activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
history_l2_reg = model.fit(
    train_data,
    train_labels,
    epochs=20,
    batch_size=512,
    validation_split=0.4,
)
```

列表 5.12：将 L2 权重正则化添加到模型中

`l2(0.002)` 表示层中每个权重矩阵的系数都将添加 `0.002 * weight_coefficient_value ** 2` 到模型的总体损失中。请注意，由于这种惩罚*仅在训练时添加*，因此该模型的损失在训练时比测试时高得多。

图 5.20 展示了 L2 正则化惩罚的效果。如您所见，具有 L2 正则化的模型比参考模型对过拟合的抵抗力更强，尽管两个模型具有相同数量的参数：见图 5.20：

![图片](img/a8ba8a378919e7d2c88c940518b94422.png)

图 5.20：L2 权重正则化对验证损失的影响

作为 L2 正则化的替代方案，您可以使用以下 Keras 权重正则化器之一。

```py
from keras import regularizers

# L1 regularization
regularizers.l1(0.001)
# Simultaneous L1 and L2 regularization
regularizers.l1_l2(l1=0.001, l2=0.001)
```

代码列表 5.13：Keras 中可用的不同权重正则化器

注意，权重正则化通常用于较小的深度学习模型。大型深度学习模型往往过度参数化，对权重值施加约束对模型容量和泛化能力的影响不大。在这些情况下，更倾向于使用不同的正则化技术：*dropout*。

#### 添加 dropout

*Dropout*，由多伦多大学的 Geoffrey Hinton 及其学生开发，是神经网络中最有效且最常用的正则化技术之一。Dropout 在层中的应用包括在训练期间随机 *丢弃*（置零）一定数量的层输出特征。假设一个给定的层在训练期间对于一个给定的输入样本通常会返回一个向量 `[0.2, 0.5, 1.3, 0.8, 1.1]`。在应用 dropout 后，这个向量将会有一些随机分布的零值：例如，`[0, 0.5, 1.3, 0, 1.1]`。*dropout 率* 是被置零的特征的比例；它通常设置在 0.2 到 0.5 之间。在测试时，不丢弃任何单元；相反，层的输出值按 dropout 率的因子进行缩放，以平衡训练时比测试时更活跃的单元数量。

考虑一个包含层输出 `layer_output` 的 NumPy 矩阵，其形状为 `(batch_size, features)`。在训练时，我们随机将矩阵中的一部分值置零：

```py
# At training time, drops out 50% of the units in the output
layer_output *= np.random.randint(low=0, high=2, size=layer_output.shape)
```

在测试时，我们将输出按 dropout 率进行缩放。这里，我们按 0.5 缩放（因为我们之前丢弃了一半的单元）：

```py
# At test time
layer_output *= 0.5
```

注意，这个过程可以通过在训练时执行这两个操作，并在测试时保持输出不变来实现，这在实践中通常是这样做的方式（见图 5.21）：

```py
# At training time
layer_output *= np.random.randint(low=0, high=2, size=layer_output.shape)
# Note that we're scaling up rather scaling down in this case.
layer_output /= 0.5
```

![图片](img/efa2213fd7e01bbc65bc2b5e8326ebaf.png)

图 5.21：在训练时应用于激活矩阵的 dropout，训练过程中进行缩放。在测试时，激活矩阵保持不变。

这种技术可能看起来很奇怪且随意。为什么这有助于减少过拟合？Hinton 表示，他受到了，包括但不限于，银行使用的欺诈预防机制的启发：

> 我去银行。出纳员一直在更换，我询问其中一位为什么。他说他不知道，但他们经常被调动。我想这肯定是因为这需要员工之间的合作才能成功欺诈银行。这让我意识到，在每一个例子中随机移除不同子集的神经元可以防止阴谋，从而减少过拟合。

核心思想是在层的输出值中引入噪声可以打破不重要的偶然模式（Hinton 称之为**阴谋**），如果没有噪声，模型将开始记忆这些模式。

在 Keras 中，你可以通过`Dropout`层在模型中引入 dropout，该层应用于层之前的输出。让我们在 IMDB 模型中添加两个`Dropout`层，看看它们在减少过拟合方面做得如何。

```py
model = keras.Sequential(
    [
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
history_dropout = model.fit(
    train_data,
    train_labels,
    epochs=20,
    batch_size=512,
    validation_split=0.4,
)
```

列表 5.14：在 IMDB 模型中添加 dropout

图 5.22 显示了结果的图表。这比参考模型有明显的改进。它似乎比 L2 正则化工作得更好，因为最低的验证损失已经得到改善：

![](img/e72c698c6b306d49d152835add95f515.png)

图 5.22：dropout 对验证损失的影响

总结一下，这些是在神经网络中最大化泛化并防止过拟合的最常见方法：

+   获取更多或更好的训练数据
+   开发更好的特征
+   降低模型的容量
+   添加权重正则化（适用于较小的模型）
+   添加 dropout

## 摘要

+   机器学习模型的目的在于**泛化**：在从未见过的输入上执行准确。这比看起来要难。
+   深度神经网络通过学习一个可以成功在训练样本之间进行**插值**的参数模型来实现泛化。这样的模型可以说已经学会了训练数据的**潜在流形**。这就是为什么深度学习模型只能理解在训练期间所见到的非常接近的输入。
+   机器学习中的基本问题是**优化与泛化之间的张力**：为了实现泛化，你必须首先对训练数据有一个良好的拟合，但提高你的模型对训练数据的拟合最终会损害泛化。每一个深度学习最佳实践都涉及管理这种张力。
+   深度学习模型能够泛化的能力源于它们能够学会近似其数据的**潜在流形**，因此可以通过插值理解新的输入。
+   在开发模型的过程中，能够准确评估模型的一般化能力至关重要。你可以使用一系列评估方法，从简单的保留验证到 K 折交叉验证和带洗牌的迭代 K 折交叉验证。请记住，始终为最终模型评估保留一个完全独立的测试集，因为你的验证数据可能已经泄露到模型中。
+   当你开始构建模型时，你的目标是首先实现一个具有一定一般化能力且可以过拟合的模型。实现这一目标的最佳实践包括调整学习率和批量大小、使用更好的架构先验、增加模型容量或简单地延长训练时间。
+   当你的模型开始过拟合时，你的目标转变为通过*模型正则化*来提高一般化能力。你可以减少模型容量，添加 dropout 或权重正则化，并使用提前停止。当然，更大的或更好的数据集始终是帮助模型实现一般化的首要方法。

### 脚注

1.  马克·吐温甚至称其为“人类已知最美味的水果。” [[↩]](#footnote-link-1)
