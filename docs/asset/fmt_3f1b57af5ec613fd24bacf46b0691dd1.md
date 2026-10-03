
# 训练与评估

## 训练过程

```py
Train for 704 steps, validate for 79 steps
Epoch 1/50
704/704 [==============================] - 196s 279ms/step - loss: 14.6223 
➥ - final_loss: 4.9449 - aux1_loss: 4.8074 - aux2_loss: 4.8700 - 
➥ final_accuracy: 0.0252 - aux1_accuracy: 0.0411 - aux2_accuracy: 0.0347 
➥ - val_loss: 13.3207 - val_final_loss: 4.5473 - val_aux1_loss: 4.3426 - 
➥ val_aux2_loss: 4.4308 - val_final_accuracy: 0.0595 - val_aux1_accuracy: 
➥ 0.0860 - val_aux2_accuracy: 0.0765
...
Epoch 50/50
704/704 [==============================] - 196s 279ms/step - loss: 0.6361 - 
➥ final_loss: 0.2271 - aux1_loss: 0.1816 - aux2_loss: 0.2274 - 
➥ final_accuracy: 0.9296 - aux1_accuracy: 0.9411 - aux2_accuracy: 0.9264 
➥ - val_loss: 27.6959 - val_final_loss: 7.9506 - val_aux1_loss: 10.4079 - 
➥ val_aux2_loss: 9.3375 - val_final_accuracy: 0.2703 - val_aux1_accuracy: 
➥ 0.2318 - val_aux2_accuracy: 0.2361
```

**注意**：在一台配备 Intel Core i5 处理器和 NVIDIA GeForce RTX 2070 8GB 显卡的机器上，训练大约需要 2 小时 45 分钟。您可以通过减少训练周期的数量来减少训练时间。

## 测试模型

最后，我们将在测试数据上测试训练好的模型（即 `val` 文件夹中的数据）。您可以通过调用`model.evaluate()`并传递测试数据生成器（`test_gen_aux`）和测试集的步数（迭代次数）来轻松获取模型的测试性能：

```py
model = load_model(os.path.join('models','inception_v1_base.h5'))
test_res = model.evaluate(test_gen_aux, steps=get_steps_per_epoch(200*50, 
➥ batch_size))
test_res_dict = dict(zip(model.metrics_names, test_res))
```

您将会得到以下输出：

```py
196/196 [==============================] - 17s 88ms/step - loss: 27.7303 - 
➥ final_loss: 7.9470 - aux1_loss: 10.3892 - aux2_loss: 9.3941 - 
➥ final_accuracy: 0.2700 - aux1_accuracy: 0.2307 - aux2_accuracy: 0.2367
```

## 结果分析

我们可以看到模型达到了约 30% 的验证和测试准确率以及惊人的约 94% 的训练准确率。这清楚地表明我们没有完全避免过拟合。但这并不完全是坏消息。三十％的准确率意味着模型在验证和测试集中识别了约 3,000/10,000 张图像。就纯粹的数据量而言，这相当于 200 个类别中的 60 个。

**注意**：过拟合的模型就像一个把所有答案都背下来的学生，而泛化的模型是一个努力理解将在考试中测试的概念的学生。背诵答案的学生只会在考试中表现良好，在现实世界中会失败，而理解概念的学生可以将他们的知识推广到考试和现实世界中。

## 过拟合的原因

过度拟合可能会出现几种原因：

+   模型架构对我们拥有的数据集来说并不是最佳的。
+   需要更多的正则化来减少过拟合，比如 `dropout` 和批归一化。
+   我们没有使用已经在类似数据上训练过的预训练模型。

我们将在下一章中解决这些问题，看到事情会有多大改善将会令人兴奋。

## 练习 4

如果用一个包含 50,000 个样本且批次大小为 250 的数据集训练模型 10 个周期，那么你会训练模型多少次迭代？假设输入和标签分别存储在变量 `x` 和 `y` 中，填写`model.fit()`中的必要参数以根据这个规范训练模型。在不使用数据生成器时，可以使用 `batch_size` 参数设置批次大小，并在`model.fit()`中忽略 `steps_per_epoch` 参数（自动推断）。

# 总结

+   探索性数据分析（EDA）是机器学习生命周期中必须在开始任何建模之前执行的关键步骤。
+   分析数据的方面越多，效果越好。
+   Keras 数据生成器可以用来从磁盘读取图像并将它们加载到内存中以训练模型。
+   Inception 网络 v1 是用于图像分类的最新计算机视觉模型之一，旨在减少过拟合和深度模型的内存需求。
+   Inception 网络 v1 包括一个 stem、若干 inception 块和辅助输出。
+   Inception 块是 Inception 网络中的一层，包含多个具有不同卷积核大小的子卷积层，而辅助输出确保模型训练的平稳性。
+   训练模型时，会使用三个数据集：训练集、验证集和测试集。
+   通常情况下，我们会在训练数据上训练模型多个周期，在每个周期结束时在验证集上评估性能。最后，在训练结束后，我们会在测试数据集上评估性能。
+   过拟合的模型就像一个把所有答案都记住的学生，它在训练数据上表现很好，但在将知识概括应用于未见数据分析时效果很差。

# 练习答案

## 练习 1

```py
def get_img_minimum(path):
    img = np.array(Image.open(path))
    return np.min(img)

df["minimum"] = df["filepath"].apply(lambda x: get_img_minimum(x))
```

## 练习 2

```py
def data_gen_corrupt(gen):
    for x,y in gen:
        if np.random.normal()>0:
            y = 0
        yield x,(y,y,y)
```

## 练习 3

```py
def poolception(x):
    out1 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
    out2 = MaxPool2D(pool_size=(5,5), strides=(2,2), padding='same')(out1)
    out3 = AvgPool2D(pool_size=(3,3), strides=(2,2), padding='same')(out2)
    out = Concatenate(axis=-1)([out1, out2, out3])
    return out
```

## 练习 4

总迭代次数 = （数据集大小/批次大小）* 周期数 = （50,000/250）* 10 = 2,000

```py
model.fit(x=x, y=y, batch_size=250, epochs=10)
```
