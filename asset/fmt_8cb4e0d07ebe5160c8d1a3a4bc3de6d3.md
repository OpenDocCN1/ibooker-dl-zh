
# 自定义评估指标与模型训练

## 自定义评估指标

### 平均准确率指标

以下代码实现了平均准确率（Mean Accuracy）评估指标，它继承自`tf.keras.metrics.Mean`：

```py
class MeanAccuracyMetric(tf.keras.metrics.Mean):

  def __init__(self, num_classes, name='mean_accuracy', **kwargs):
    super(MeanAccuracyMetric, self).__init__(name=name, **kwargs)    

  def update_state(self, y_true, y_pred, sample_weight=None):

    smooth = 1            

    y_true.set_shape([None, y_pred.shape[1], y_pred.shape[2]])             ❶

    y_true = tf.reshape(y_true, [-1])                                      ❶
    y_pred = tf.reshape(tf.argmax(y_pred, axis=-1),[-1])                   ❶

    valid_mask = tf.reshape((y_true <= num_classes - 1), [-1])             ❶

    y_true = tf.boolean_mask(y_true, valid_mask)                           ❶
    y_pred = tf.boolean_mask(y_pred, valid_mask)                           ❶

    conf_matrix = tf.cast(                                                            
        tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes), 
➥ 'float32'                                                               ❷
    )
    true_pos = tf.linalg.diag_part(conf_matrix)                            ❸

    mean_accuracy = tf.reduce_mean(
        (true_pos + smooth)/(tf.reduce_sum(conf_matrix, axis=1) + smooth)  ❹
    )

    super(MeanAccuracyMetric, self).update_state(mean_accuracy)            ❺
```

❶ 初始设置  
❷ 使用`y_true`和`y_pred`计算混淆矩阵。  
❸ 获取真正的正例（对角线上的元素）。  
❹ 使用每个类别的真正正例和真正类别计数计算平均准确度。  
❺ 使用`update_state()`函数计算`mean_accuracy`的平均值。

`MeanAccuracyMetric`将从`tf.keras.metrics.Mean`分支出来，它计算给定值序列的平均值。计划是在`update_state()`函数中计算`mean_accuracy`，然后将该值传递给父类的`update_state()`函数，以便得到平均准确度的平均值。首先，我们执行我们之前讨论的`y_true`和`y_pred`的初始设置和清理。

![08-13](img/08-13.png)

图 8.13 五类分类问题的混淆矩阵示意图。阴影区域表示真正的正例。

然后，从预测和真实目标计算混淆矩阵（图 8.13）。对于一种*n*种分类问题（即，具有*n*个可能类别的分类问题），混淆矩阵被定义为一个*n*×*n*矩阵。这里，位置（*i*，*j*）处的元素表示预测为属于第*i*个类别的实例实际上属于第*j*个类别。图 8.13 描绘了这种类型的混淆矩阵。我们可以通过提取对角线（即，所有 1 <= *i* <= *n*的矩阵中的(*i*，*i*)元素）来获取真阳性。现在我们可以通过两个步骤计算平均准确度：

1. 对于所有类别，对真阳性计数进行逐元素除法。这将产生一个向量，其元素表示每个类别的准确性。
2. 计算步骤 1 产生的向量平均值。

最后，我们将平均准确度传递给其父对象的`update_state()`函数。

### 平均交并比指标

*平均交并比*(mean intersection over union)是用于分割任务的流行评估度量，与我们之前讨论的 Dice loss 密切相关，因为它们都使用交和并的概念来计算最终结果（见下一个列表）。

列表 8.19 实现平均交并比度量

```py
class MeanIoUMetric(tf.keras.metrics.MeanIoU):

  def __init__(self, num_classes, name='mean_iou', **kwargs):
    super(MeanIoUMetric, self).__init__(num_classes=num_classes, name=name, **kwargs)    

  def update_state(self, y_true, y_pred, sample_weight=None):

    y_true.set_shape([None, y_pred.shape[1], y_pred.shape[2]])
    y_true = tf.reshape(y_true, [-1])

    y_pred = tf.reshape(tf.argmax(y_pred, axis=-1),[-1])

    valid_mask = tf.reshape((y_true <= num_classes - 1), [-1])

    # Get pixels corresponding to valid mask
    y_true = tf.boolean_mask(y_true, valid_mask)
    y_pred = tf.boolean_mask(y_pred, valid_mask)

    super(MeanIoUMetric, self).update_state(y_true, y_pred)    ❶
```

❶ 在`y_true`和`y_pred`的初始设置之后，我们只需调用父对象的`update_state()`函数即可。

平均交并比的计算已经在`tf.keras.metrics.MeanIoU`中找到。因此，我们将使用它作为我们的父类。我们所需做的就是为`y_true`和`y_pred`进行前述设置，然后调用父对象的`update_state()`函数。平均交并比计算为

![08_13a](img/08_13a.png)

在该计算中使用的各种元素如图 8.14 所示。

![08-14](img/08-14.png)

图 8.14 混淆矩阵，以及如何用它来计算假阳性，假阴性和真阳性

我们现在理解了可用于我们的损失函数和评估指标，并已经实现了它们。我们可以将这些损失编译到模型中：

```py
deeplabv3.compile(
    loss=ce_dice_loss_from_logits(num_classes), 
    optimizer=optimizer, 
    metrics=[
        MeanIoUMetric(num_classes), 
        MeanAccuracyMetric(num_classes), 
        PixelAccuracyMetric(num_classes)
    ])
```

记住，我们从前面删除的卷积块中存储了权重。现在我们已经编译了模型，我们可以使用以下语法将权重复制到新模型中：

```py
# Setting weights for newly added layers
for k, w *in* w_dict.items():    
    deeplabv3.get_layer(k).set_weights(w)
```

我们现在开始使用数据管道和我们定义的模型来训练模型。

### 练习 4

您正在设计一个新的损失函数，用于计算`y_true`和`y_pred`之间的不相交并集。两个集合 A 和 B 之间的不相交并集是在 A 或 B 中但不在交集中的元素的集合。您知道可以通过`y_true`和`y_pred`的元素乘法计算交并通过`y_true`和`y_pred`的元素加法计算。编写函数的等式，以根据`y_true`和`y_pred`计算不相交并集。

## 8.5 训练模型

您已经进入产品第一次迭代的最后阶段。现在是时候充分利用您获得的数据和知识了（即，训练模型）。我们将训练模型进行 25 个周期，并监视像素精度、平均精度和平均 IoU 指标。在训练过程中，我们将衡量验证数据集的性能。

训练模型是最容易的部分，因为我们已经完成了导致训练的艰苦工作。现在只需要在我们刚刚定义的 DeepLab v3 上使用正确的参数调用`fit()`，如下面的列表所示。

图 8.20 训练模型

```py
if *not* os.path.exists('eval'):
    os.mkdir('eval')

csv_logger = tf.keras.callbacks.CSVLogger(
    os.path.join('eval','1_pretrained_deeplabv3.log')                      ❶

monitor_metric = 'val_loss'
mode = 'min' if 'loss' *in* monitor_metric else 'max'                        ❷
print("Using metric={} and mode={} for EarlyStopping".format(monitor_metric, mode))
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor=monitor_metric, factor=0.1, patience=3, mode=mode, min_lr=1e-8 ❸
)
es_callback = tf.keras.callbacks.EarlyStopping(
    monitor=monitor_metric, patience=6, mode=mode                          ❹
)

# Train the model
deeplabv3.fit(                                                             ❺
    x=tr_image_ds, steps_per_epoch=n_train,
    validation_data=val_image_ds, validation_steps=n_valid, 
    epochs=epochs, callbacks=[lr_callback, csv_logger, es_callback])
```

❶ 训练记录器  
❷ 自动根据指标名称设置后续回调函数的模式。  
❸ 学习率调度器  
❹ 提前停止回调  
❺ 在使用验证集进行学习率调整和提前停止的同时训练模型。

首先，如果不存在，我们将定义一个名为`eval`的目录。训练日志将保存在这个目录中。接下来，我们定义三个不同的回调函数在训练过程中使用：

+ `csv_logger`—记录训练损失/指标和验证损失/指标
+ `lr_callback`—如果验证损失在三个周期内没有减少，则将学习率减小 10 倍
+ `es_callback`—如果验证损失在六个周期内没有减少，则执行提前停止

注意 在一台配备英特尔 Core i5 处理器和 NVIDIA GeForce RTX 2070 8GB 显卡的机器上，训练大约需要 45 分钟来运行 25 个周期。

有了这个，我们使用以下参数调用`deeplabv3.fit()`：

+ `x`—生成训练实例的`tf.data`流水线（设置为`tr_image_ds`）。
+ `steps_per_epoch`—每个周期的步数。这是通过计算训练实例的数量并将其除以批次大小获得的（设置为`n_train`）。
+ `validation_data`—生成验证实例的`tf.data`流水线。这是通过计算验证实例的数量并将其除以批次大小来获得的（设置为`val_image_ds`）。
+ `epochs`—周期数（设置为`epochs`）。
+ `callbacks`—我们之前设置的回调函数（设置为`[lr_callback, csv_logger, es_callback]`）。

在模型训练完成后，我们将在测试集上进行评估。我们还将可视化模型生成的分割结果。

### 练习 5

您有一个包含 10,000 个样本的数据集，并将其分成 90% 的训练数据和 10% 的验证数据。您使用的训练批次大小为 10，验证批次大小为 20。单个周期中将有多少个训练和验证步骤？

## 8.6 评估模型

让我们花点时间回顾一下我们到目前为止所做的事情。我们定义了一个数据管道来读取图像并将它们准备为模型的输入和目标。然后我们定义了一个称为 DeepLab v3 的模型，它使用预训练的 ResNet-50 作为其主干网络，并使用称为空洞空间金字塔池的特殊模块来预测最终的分割掩模。然后我们定义了任务特定的损失和指标，以确保我们可以使用各种指标评估模型。之后，我们对模型进行了训练。现在是最终揭晓的时候了。我们将在一个未见过的测试数据集上评估性能，看看模型的表现如何。我们还将可视化模型输出，并将其与真实目标进行比较，将它们并排绘制出来。

我们可以在未见过的测试图像上运行模型，并评估其性能。为此，我们执行以下操作：

```py
deeplabv3.evaluate(test_image_ds, steps=n_valid)
```

测试集的大小与验证集相同，因为我们将在`val.txt`中列出的图像分成两个相等的验证集和测试集。这将返回大约

+ 62% 的平均 IoU
+ 87% 的平均准确率
+ 91% 的像素准确率

考虑到我们的情况，这些得分非常可观。我们的训练数据集包含不到 1500 张分割图像。使用这些数据，我们能够训练出一个在大小约为 725 的测试数据集上达到约 62% 平均 IoU 的模型。

### 什么是技术发展的最新成果？

Pascal VOC 2012 的最新性能报告显示约 90% 的平均 IoU ([`mng.bz/o2m2`](http://mng.bz/o2m2))。然而，这些模型比我们在这里使用的模型要大得多且复杂得多。此外，它们通常使用一个称为语义边界数据集（SBD）的辅助数据集进行训练（该数据集在论文 [`mng.bz/nNve`](http://mng.bz/nNve) 中介绍）。这将使训练数据点数量增加到 10000 多个（几乎是我们当前训练集大小的七倍）。

您可以通过直观检查模块生成的一些结果来进一步研究模型。毕竟，我们正在开发一种视觉模型。因此，我们不应仅仅依靠数字来做出决定和结论。在确定结论之前，我们还应该对结果进行视觉分析。

### 基于 U-Net 的网络的结果会是什么样的呢？

在为 DeepLab v3 模型提供类似条件的情况下，使用预训练的 ResNet-50 模型作为编码器构建的 U-Net 模型仅能达到约 32.5% 的平均 IoU、78.5% 的平均准确率和 81.5% 的像素准确率。实现细节在`ch08`文件夹的 Jupyter 笔记本中提供。

在一台搭载 Intel Core i5 处理器和 NVIDIA GeForce RTX 2070 8GB 显卡的机器上，训练 25 个周期大约需要 55 分钟。

U-Net 模型的详细解释见附录 B。

为了完成这个调查，我们将从测试集中随机选择一些样本，并要求模型对每个图像进行分割地图的预测。然后我们将将结果并排绘制在一起，以确保我们的模型工作得很好（图 8.15）。

![08-15](img/08-15.png)

图 8.15 对比真实标注目标和模型预测结果。可以看出，该模型在将不同背景的物体分离方面表现很好。

我们可以看到，除非是一张非常困难的图像（例如左上角的图像，其中有一个被栅栏遮挡的汽车），我们的模型表现得非常好。它可以以高准确率识别我们分析的样本中几乎所有的图像。可在笔记本中找到显示图像的代码。

这就是我们关于图像分割的讨论。在接下来的几章中，我们将讨论几个自然语言处理问题。

### 练习 6

您被给予的是

+ 一个模型（称为`model`）
+ 名为`batch_image`的图像批次（已预处理并准备好馈送给模型）
+ 相应的目标批次，`batch_targets`（以独热编码格式表示的真实分割掩码）

编写一个名为`get_top_bad_examples(model, batch_images, batch_targets, n)`的函数，它将返回批次图像中损失最高（最难的）图像的前`n`个索引。对于给定的预测掩码和目标掩码，可以将元素逐个相乘的和作为给定图像的损失。

您可以使用`model.predict()`函数对`batch_images`进行预测，并返回与`batch_targets`相同大小的预测掩码。一旦计算出批次的损失（`batch_loss`），您可以使用`tf.math.top_k(batch_loss, n)`函数获取具有最高值的元素的索引。`tf.math.top_k()`返回一个元组，包含给定向量的前`n`个值和索引，按顺序排列。

## 总结

+ 分割模型分为两大类别：语义分割和实例分割。
+ `tf.data` API 提供了多种功能来实现复杂的数据流水线，例如使用自定义的 NumPy 函数，使用`tf.data.Dataset.map()`执行快速转换以及使用`prefetch`和`cache`等 I/O 优化技术。
+ DeepLab v3 是一种常用的分割模型，它使用了预训练的 ResNet-50 模型作为骨干网络，并通过在卷积操作的权重之间插入孔（即零）来增加感受野。
