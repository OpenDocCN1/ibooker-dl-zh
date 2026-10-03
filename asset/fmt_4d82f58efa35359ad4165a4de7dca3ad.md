
```py
array([[9.9994004e-01, 9.4704428e-06, 3.8405190e-10, 5.0486942e-05,
        1.0701914e-08],
       [5.9500107e-06, 3.1842374e-06, 3.5622744e-08, 9.9999082e-01,
        3.0683900e-08],
       [9.9994218e-01, 5.9974178e-07, 5.8693445e-10, 5.7049790e-05,
        9.6709634e-08],
       ...,
       [3.1268091e-06, 9.9986601e-01, 1.5343730e-06, 1.2935932e-04,
        2.7383029e-09],
       [4.8439368e-05, 1.9247003e-05, 1.8034354e-01, 1.6394027e-02,
        8.0319476e-01],
       [4.9799957e-07, 9.9232978e-01, 3.5823192e-08, 7.6697678e-03,
        1.7666844e-09]], dtype=float32)
```

每行代表了图像类别的概率分布。对于第一张图像，最高概率为`1.0701914e-08`（在上述代码中突出显示），位于该行的最后位置，对应于该行的索引`4`（请记住，索引的编号从`0`开始）。

现在，您需要使用以下代码找到每行中最高概率出现的位置：

```py
predicted_idx = tf.math.argmax(prediction, axis = -1)
```

如果您使用`print`命令显示结果，您将看到以下内容：

```py
print (predicted_idx)

<tf.Tensor: shape=(731,), dtype=int64, numpy=
array([0, 3, 0, 1, 0, 4, 4, 1, 2, 3, 4, 1, 4, 0, 4, 3, 1, 4, 4, 0,
       …
       3, 2, 1, 4, 1])>
```

现在，对该数组中的每个元素应用`idx_labels`的查找。对于每个元素，使用一个函数：

```py
def find_label(idx):
    return idx_labels[idx]
```

要将函数应用于 NumPy 数组的每个元素，您需要对函数进行矢量化：

```py
find_label_batch = np.vectorize(find_label)
```

然后将此矢量化函数应用于数组中的每个元素：

```py
result = find_label_batch(predicted_idx)
```

最后，将结果与图像文件夹和文件名并排输出，以便保存以供报告或进一步调查。您可以使用 Python pandas DataFrame 操作来实现这一点：

```py
import pandas as pd
predicted_label = result_class.tolist()
file_name = valid_generator.filenames

results=pd.DataFrame({"File":file_name,
                      "Prediction":predicted_label})
```

让我们看看`results`数据框，它是`731`行×`2`列。

|   | 文件 | 预测 |
| --- | --- | --- |
| 0 | daisy/100080576_f52e8ee070_n.jpg | daisy | 雏菊 |
| 1 | daisy/10140303196_b88d3d6cec.jpg | sunflowers | 向日葵 |
| 2 | daisy/10172379554_b296050f82_n.jpg | daisy | 雏菊 |
| 3 | daisy/10172567486_2748826a8b.jpg | dandelion | 玛丽金花 |
| 4 | daisy/10172636503_21bededa75_n.jpg | daisy | 雏菊 |
| ... | ... | ... |
| 726 | tulips/14068200854_5c13668df9_m.jpg | sunflowers | 向日葵 |
| 727 | tulips/14068295074_cd8b85bffa.jpg | roses | 玫瑰 |
| 728 | tulips/14068348874_7b36c99f6a.jpg | dandelion | 郁金香 |
| 729 | tulips/14068378204_7b26baa30d_n.jpg | tulips | 郁金香 |
| 730 | tulips/14071516088_b526946e17_n.jpg | dandelion | 玛丽金花 |

## 评估：创建混淆矩阵

混淆矩阵通过比较模型输出和实际情况来评估分类结果，是了解模型表现的最简单方法。让我们看看如何创建混淆矩阵。

您将使用 pandas Series 作为构建混淆矩阵的数据结构：

```py
y_actual = pd.Series(valid_generator.classes)
y_predicted = pd.Series(predicted_idx)
```

然后，您将再次利用 pandas 生成矩阵：

```py
pd.crosstab(y_actual, y_predicted, rownames = ['Actual'],
colnames=['Predicted'], margins=True)
```

图 4-4 显示了混淆矩阵。每行代表了实际花标签的分布情况。例如，看第一行，您会注意到总共有`126`个样本实际上是类别`0`，即雏菊。模型正确地将这些图像中的`118`个预测为类别`0`；四个被错误分类为类别`1`，即蒲公英；一个被错误分类为类别`2`，即玫瑰；三个被错误分类为类别`3`，即向日葵；没有被错误分类为类别`4`，即郁金香。

![花卉图像分类的混淆矩阵](img/t2pr_0404.png)

###### 图 4-4\. 花卉图像分类的混淆矩阵

接下来，使用`sklearn`库为每个图像类别提供统计报告：

```py
from sklearn.metrics import classification_report
report = classification_report(truth, predicted_results)
print(report)
              precision    recall  f1-score   support

           0       0.90      0.94      0.92       126
           1       0.93      0.87      0.90       179
           2       0.85      0.86      0.85       128
           3       0.85      0.88      0.86       139
           4       0.86      0.86      0.86       159

    accuracy                           0.88       731
   macro avg       0.88      0.88      0.88       731
weighted avg       0.88      0.88      0.88       731
```

这个结果表明，当对雏菊（类别`0`）进行分类时，该模型的性能最佳，`f1`分数为`0.92`。在对玫瑰（类别`2`）进行分类时，其性能最差，`f1`分数为`0.85`。“支持”列显示了每个类别中的样本量。

## 总结

您刚刚完成了一个使用来自 TensorFlow Hub 的预训练模型的示例项目。您添加了必要的输入层，执行了数据归一化和标准化，训练了模型，并对一批图像进行了评分。

这个经验表明了满足模型的输入和输出要求的重要性。同样重要的是，要密切关注预训练模型的输出格式。（这些信息都可以在 TensorFlow Hub 网站上的模型文档页面找到。）最后，您还需要创建一个函数，将模型的输出映射到纯文本，以使其具有意义并可解释。

# 使用 tf.keras.applications 模块进行预训练模型

另一个为您自己使用找到预训练模型的地方是`tf.keras.applications`模块（请参阅[可用模型列表](https://oreil.ly/HQJBl)）。当 Keras API 在 TensorFlow 中可用时，该模块成为 TensorFlow 生态系统的一部分。

每个模型都带有预训练的权重，使用它们和使用 TensorFlow Hub 一样简单。Keras 提供了方便地微调模型所需的灵活性。通过使模型中的每一层可访问，`tf.keras.applications`让您可以指定哪些层要重新训练，哪些层保持不变。

## 使用 tf.keras.applications 实现模型

与 TensorFlow Hub 一样，您只需要一行代码从 Keras 模块加载一个预训练模型：

```py
base_model = tf.keras.applications.ResNet101V2(
input_shape = (224, 224, 3), 
include_top = False, 
weights = 'imagenet')
```

注意`include_top`输入参数。请记住，您需要为自己的数据添加一个输出层。通过将`include_top`设置为`False`，您可以为分类输出添加自己的`Dense`层。您还将从`imagenet`初始化模型权重。

然后将`base_model`放入一个顺序架构中，就像您在 TensorFlow Hub 示例中所做的那样：

```py
model2 = tf.keras.Sequential([
  base_model,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(NUM_CLASSES, 
  activation = 'softmax', 
  name = 'flower_class')
])
```

添加`GlobalAveragePooling2D`，将输出数组平均为一个数值，然后将其发送到最终的`Dense`层进行预测。

现在编译模型并像往常一样启动训练过程：

```py
model2.compile(
  optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
  loss=tf.keras.losses.CategoricalCrossentropy(
  from_logits=True, label_smoothing=0.1),
  metrics=['accuracy']
)

model2.fit(
    train_generator,
    epochs=5, steps_per_epoch=steps_per_epoch,
    validation_data=valid_generator,
    validation_steps=validation_steps)
```

要对图像数据进行评分，请按照您在“将输出映射到纯文本格式”中所做的步骤进行。

## 从 tf.keras.applications 微调模型

如果您希望通过释放一些基础模型的层进行训练来尝试您的训练例程，您可以轻松地这样做。首先，您需要找出基础模型中有多少层，并将基础模型指定为可训练的：

```py
print("Number of layers in the base model: ", 
       len(base_model.layers))
base_model.trainable = True

Number of layers in the base model:  377
```

如所示，在这个版本的 ResNet 模型中，有`377`层。通常我们从模型末尾附近的层开始重新训练过程。在这种情况下，将第`370`层指定为微调的起始层，同时保持在第`300`层之前的权重不变：

```py
fine_tune_at = 370

for layer in base_model.layers[: fine_tune_at]:
  layer.trainable = False
```

然后使用`Sequential`类将模型组合起来：

```py
model3 = tf.keras.Sequential([
  base_model,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(NUM_CLASSES, 
  activation = 'softmax', 
  name = 'flower_class')
])
```

###### 提示

您可以尝试使用`tf.keras.layers.Flatten()`而不是`tf.keras.layers.GlobalAveragePooling2D()`，看看哪一个给您一个更好的模型。

编译模型，指定优化器和损失函数，就像您在 TensorFlow Hub 中所做的那样：

```py
model3.compile(
  optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
  loss=tf.keras.losses.CategoricalCrossentropy(
  from_logits=True, 
  label_smoothing=0.1),
  metrics=['accuracy']
)
```

启动训练过程：

```py
fine_tune_epochs = 5
steps_per_epoch = train_generator.samples // 
train_generator.batch_size
validation_steps = valid_generator.samples // 
valid_generator.batch_size
model3.fit(
    train_generator,
    epochs=fine_tune_epochs, 
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_generator,
    validation_steps=validation_steps)
```

由于您已经释放了更多基础模型的层进行重新训练，这个训练可能需要更长时间。训练完成后，对测试数据进行评分，并按照“将输出映射到纯文本格式”和“评估：创建混淆矩阵”中描述的方式比较结果。

# 结束

在这一章中，您学习了如何使用预训练的深度学习模型进行迁移学习。有两种方便的方式可以访问预训练模型：TensorFlow Hub 和`tf.keras.applications`模块。两者都简单易用，具有优雅的 API 和风格，可以快速开发模型。然而，用户需要正确地塑造他们的输入数据，并提供一个最终的`Dense`层来处理模型输出。

有大量免费可访问的预训练模型，具有丰富的库存，您可以使用它们来处理自己的数据。利用迁移学习来利用它们，让您花费更少的时间来构建、训练和调试模型。
