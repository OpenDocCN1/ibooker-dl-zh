
```py
epochs = range(1, len(average_mae_history) + 1)
plt.plot(epochs, average_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show() 
```

代码列表 4.31：绘制验证分数

![](img/3a62bf8f25afa58de847c1fe58574fe5.png)

图 4.10：按 epoch 验证的 MAE

由于缩放问题，可能难以阅读图表：前几个 epoch 的验证 MAE 比后续的值高得多。让我们省略前 10 个数据点，这些数据点与曲线的其余部分处于不同的尺度。

```py
truncated_mae_history = average_mae_history[10:]
epochs = range(10, len(truncated_mae_history) + 10)
plt.plot(epochs, truncated_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show() 
```

代码列表 4.32：绘制验证分数，排除前 10 个数据点

![](img/e5f84d5faf3a31c710e45f85a1a04052.png)

图 4.11：按 epoch 验证的 MAE，排除前 10 个数据点

根据这个图（见图 4.11），验证 MAE 在 120-140 个周期后（这个数字包括我们省略的 10 个周期）不再显著提高。超过这个点，你开始过拟合。

一旦你完成了模型其他参数的调整（除了周期数，你还可以调整中间层的大小），你可以在所有训练数据上使用最佳参数训练一个最终的生产模型，然后查看其在测试数据上的性能。

```py
# Gets a fresh, compiled model
model = get_model()
# Trains it on the entirety of the data
model.fit(x_train, y_train, epochs=130, batch_size=16, verbose=0)
test_mean_squared_error, test_mean_absolute_error = model.evaluate(
    x_test, y_test
) 
```

列表 4.33：训练最终模型

这是最终结果：

```py
>>> round(test_mean_absolute_error, 3)
0.31
```

我们平均偏差约为 $31,000。

### 在新数据上生成预测

当在二分类模型上调用`predict()`时，我们为每个输入样本检索了一个介于 0 和 1 之间的标量分数。对于我们的多类分类模型，我们为每个样本检索了所有类别的概率分布。现在，使用这个标量回归模型，`predict()`返回模型对样本价格的猜测，以十万美元为单位：

```py
>>> predictions = model.predict(x_test)
>>> predictions[0]
array([2.834494], dtype=float32)
```

测试集中的第一个区域预测的平均房价约为 $283,000。

### 总结

这是你应该从这个标量回归示例中吸取的教训：

+   回归使用与分类不同的损失函数。均方误差（MSE）是回归中常用的损失函数。

+   类似地，用于回归的评估指标与用于分类的指标不同；自然地，准确性的概念不适用于回归。MAE 是一个常见的回归指标。

+   当输入数据中的特征具有不同范围的价值时，每个特征应该作为预处理步骤独立缩放。

+   当可用的数据很少时，使用 K 折验证是可靠评估模型的好方法。

+   当可用的训练数据很少时，最好使用具有少量中间层（通常只有一个或两个）的小型模型，以避免严重的过拟合。

## 摘要

+   在向量数据上最常见的三种机器学习任务是二分类、多类分类和标量回归。每个任务使用不同的损失函数：

    +   `binary_crossentropy` 用于二分类

    +   `categorical_crossentropy` 用于多类分类

    +   `mean_squared_error` 用于标量回归

+   在将原始数据输入神经网络之前，你通常需要预处理原始数据。

+   当你的数据具有不同范围的特征时，作为预处理的一部分，独立地对每个特征进行缩放。

+   随着训练的进行，神经网络最终开始过拟合，并在从未见过的数据上获得更差的结果。

+   如果你没有太多训练数据，使用只有一个或两个中间层的较小模型，以避免严重的过拟合。

+   如果你的数据分为许多类别，如果你使中间层太小，可能会导致信息瓶颈。

+   当你处理少量数据时，K 折交叉验证可以帮助你可靠地评估你的模型。
