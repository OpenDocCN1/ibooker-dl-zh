
![Image](img/f0160-01.jpg)

在前面的清单中，`regularizer_l2(0.002)`表示层中权重矩阵中的每个系数将会添加`0.002 * weight_coefficient_value ^ 2`到模型的总损失中。请注意，因为这个惩罚只在训练时*添加*，所以该模型的损失在训练时会比在测试时高得多。

图 5.19 显示了 L2 正则化惩罚的影响。正如你所见，具有 L2 正则化的模型比参考模型更能抵抗过拟合，即使两个模型具有相同数量的参数。

### 清单 5.14 生成图表以演示 L2 权重正则化的效果

```
plot(NULL,
main = "L2 权重正则化对验证损失的影响",
xlab = "Epochs",
xlim = c(1, history_original$params$epochs),
ylab = "验证损失",
ylim = range(c(history_original$metrics$val_loss,
history_l2_reg $metrics$val_loss)),
panel.first = abline(v = 1:history_original$params$epochs,
lty = "dotted", col = "lightgrey"))
lines(history_original$metrics$val_loss, lty = 2)
lines(history_l2_reg $metrics$val_loss, lty = 1)
legend("左上角", lty = 2:1,
legend = c("原始模型的验证损失",
"L2 正则化模型的验证损失"))
```

![Image](img/f0161-01.jpg)

**图 5.19 L2 权重正则化对验证损失的影响**

作为 L2 正则化的替代方案，你可以使用以下 Keras 权重正则化器之一。

### 清单 5.15 Keras 中可用的不同权重正则化器

```
regularizer_l1(0.001)➊
regularizer_l1_l2(l1 = 0.001, l2 = 0.001)
<keras.regularizers.L1 object at 0x7f81cc3df340>➋
<keras.regularizers.L1L2 object at 0x7f81cc651c40>
```

➊ **L1 正则化**

➋ **同时使用 L1 和 L2 正则化**

注意，权重正则化更常用于较小的深度学习模型。大型深度学习模型往往过度参数化，对权重值施加约束并不会对模型容量和泛化性产生太大影响。在这些情况下，更喜欢使用不同的正则化技术：dropout。

### 添加 dropout

*Dropout*是神经网络中最有效、最常用的正则化技术之一；它是由杰弗·欣顿和他在多伦多大学的学生们开发的。应用于一层的 Dropout 在训练期间包括随机地*退出*（设为零）一定数量的该层输出特征。假设给定的层在训练期间为给定的输入样本返回一个向量`c(0.2, 0.5, 1.3, 0.8, 1.1)`。应用 Dropout 后，该向量将随机分布一些零条目，例如，`c(0, 0.5, 1.3, 0, 1.1)`。*退出率*是被置零的特征的比例；通常设置在 0.2 和 0.5 之间。在测试时，不会退出任何单元；相反，该层的输出值会按退出率缩小，以平衡更多单元处于活动状态的事实。

考虑一个包含一层输出的矩阵，`layer_output`，形状为`(batch_size, features)`。在训练时，我们随机将矩阵中的一部分值置零：

```
zero_out <- random_array(dim(layer_output)) < .5
layer_output[zero_out] <- 0
```

➊ **在训练时，退出输出中的 50%单元**

在测试时，我们通过退出率来缩放输出。在这里，我们以 0.5 的比例缩放（因为我们之前退出了一半的单元）：

```
layer_output <- layer_output * .5
```

➊ **在测试时**

请注意，这个过程可以通过在训练时执行两个操作并在测试时保持输出不变来实现，这通常是实践中的实现方式（参见图 5.20）：

```
layer_output[random_array(dim(layer_output)) < dropout_rate] <- 0
layer_output <- layer_output / .5
```

➊ **在训练时。请注意，在这种情况下，我们是放大而不是缩小。**

![图片](img/f0162-01.jpg)

**图 5.20 Dropout 应用于训练期间的激活矩阵，训练期间发生重新缩放。在测试时，激活矩阵保持不变。**

这种技术可能看起来奇怪而武断。为什么这有助于减少过拟合？欣顿说他受到了银行使用的一种防欺诈机制的启发。用他自己的话说，“我去了我的银行。出纳员不停地换，我问其中一个为什么。他说他不知道，但他们经常变动。我想这一定是因为成功欺诈银行需要员工之间的合作。这让我意识到，在每个例子中随机删除不同子集的神经元将防止阴谋，从而减少过拟合。”其核心思想是，向一层的输出值引入噪音可以打破无意义的模式（欣顿称之为*阴谋*），如果没有噪音存在，模型将开始记忆。

在 Keras 中，您可以通过`layer_dropout`在模型中引入 Dropout，它将 Dropout 应用于该层之前的输出。让我们在 IMDB 模型中添加两个`layer_dropout`，看看它们在减少过拟合方面的效果如何。

### 图 5.16 向 IMDB 模型添加 Dropout

```
model <- keras_model_sequential() %>%
layer_dense(16,
layer_dropout(0.5) %>%
layer_dense(16, activation = "relu") %>%
layer_dropout(0.5) %>%
layer_dense(1, activation = "sigmoid")
model %>% compile(optimizer = "rmsprop",
损失 = "binary_crossentropy",
指标 = "accuracy")
history_dropout <- model %>% fit(
训练数据, 训练标签,
epochs = 20, batch_size = 512,
validation_split = 0.4
)
plot(history_dropout)
```

![图像](img/f0163-01.jpg)

图 5.21 显示了结果的图表。这是对参考模型的明显改进，它似乎也比 L2 正则化工作得更好，因为达到的最低验证损失也有所改善。

### 图 5.17 生成一个演示 Dropout 对验证损失的效果的图表

```
plot(NULL,
主标题 = "Dropout 对验证损失的效果",
x 轴标签 = "Epochs", x 轴范围 = c(1, history_original$params$epochs),
y 轴标签 = "验证损失",
y 轴范围 = range(c(history_original$metrics$val_loss,
history_dropout $metrics$val_loss)),
面板首先 = abline(v = 1:history_original$params$epochs,
lty = "dotted", col = "lightgrey"))
lines(history_original$metrics$val_loss, lty = 2)
lines(history_dropout $metrics$val_loss, lty = 1)
说明框("左上角", lty = 1:2,
图例 = c("Dropout 正则化模型的验证损失",
"原始模型的验证损失"))
```

![图像](img/f0164-01.jpg)

**图 5.21 Dropout 对验证损失的效果**

总结一下，以下是在神经网络中最常见的最大化泛化和防止过拟合的方法：

+   获得更多的训练数据，或更好的训练数据。
+   发展更好的特征。
+   减少模型的容量。
+   添加权重正则化（适用于较小的模型）。
+   添加 Dropout

## 总结

+   机器学习模型的目的是*泛化*：在以前未见过的输入上准确执行。这比看起来更难

+   深度神经网络通过学习一个能够成功*插值*训练样本之间的参数化模型来实现泛化——这样的模型可以说已经学习了训练数据的“潜在流形”。这就是为什么深度学习模型只能理解与它们在训练中看到的非常接近的输入的原因。

+   机器学习中的基本问题是*优化和泛化之间的紧张关系*：为了实现泛化，必须首先获得对训练数据的良好拟合，但是改善模型对训练数据的拟合会在一段时间后开始损害泛化能力。每一个深度学习最佳实践都处理了这种紧张的方式。

+   深度学习模型的泛化能力来自于它们成功学习近似其数据的*潜在流形*的事实，因此可以通过插值来理解新的输入。

+   在开发模型时，能够准确评估模型的泛化能力至关重要。你可以使用一系列评估方法，从简单的留出验证到*K*-折交叉验证和迭代*K*-折交叉验证与洗牌。请记住始终保留一个完全独立的测试集用于最终模型评估，因为验证数据泄露到模型中可能已经发生。

+   当您开始研究一个模型时，您的目标首先是获得具有一定泛化能力并且可以过拟合的模型。做到这一点的最佳实践包括调整学习率和批量大小，利用更好的架构先验，增加模型容量，或者简单地延长训练时间。

+   当您的模型开始过拟合时，您的目标转向通过*模型正则化*来提高泛化能力。您可以降低模型的容量，添加 dropout 或权重正则化，并使用早停技术。自然地，一个更大或更好的数据集始终是帮助模型泛化的首选方法。

1.  ¹ 马克·吐温甚至称其为“人类所知最美味的水果。”
