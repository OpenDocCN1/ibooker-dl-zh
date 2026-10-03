
# 模型名称中的“41”是什么意思？

按照惯例，预训练的卷积神经网络通常根据它们的“深度”来命名。在这种情况下，`41`表示我们的`Xception`模型有`41`个可训练层（卷积和密集层）堆叠在一起。这是我们迄今为止在书中使用的“最深层”模型，差距很大。

在我们能够使用这个模型之前，还有一个缺失的部分需要补充。每个预训练的卷积神经网络（`ConvNet`）在预训练之前都会对图像进行一些缩放和调整大小。确保我们的输入图像*匹配*是很重要的；否则，我们的模型将需要重新学习如何从具有完全不同输入范围的图像中提取特征。与其跟踪哪些预训练模型使用`[0, 1]`的像素值输入范围，哪些使用`[-1, 1]`的范围，我们不如使用一个名为`ImageConverter`的`KerasHub`层，它将把我们的图像缩放到与我们的预训练检查点匹配。它具有与骨干类相同的特殊`from_preset()`构造函数。

```py
preprocessor = keras_hub.layers.ImageConverter.from_preset(
    "xception_41_imagenet",
    image_size=(180, 180),
) 
```

列表 8.24：实例化与`Xception`模型配合使用的预处理

在这个阶段，你可以选择两种方法继续：

+   在你的数据集上运行卷积基础，将其输出记录到磁盘上的`NumPy`数组中，然后使用这些数据作为独立、密集连接的分类器的输入，类似于你在第四章和第五章中看到的。这个解决方案运行速度快且成本低，因为它只需要为每个输入图像运行一次卷积基础，而卷积基础是整个流程中最昂贵的部分。但出于同样的原因，这种技术不允许你使用数据增强。

+   通过在输入数据上添加`Dense`层来扩展你现有的模型（`conv_base`），并从头到尾运行整个模型。这将允许你使用数据增强，因为每次模型看到输入图像时，每个输入图像都会通过卷积基础。但出于同样的原因，这种技术比第一种技术昂贵得多。

我们将介绍这两种技术。让我们通过设置第一个所需的代码来逐步进行：记录`conv_base`在你的数据上的输出，并使用这些输出作为新模型的输入。

## 无数据增强的快速特征提取

我们将首先通过调用`conv_base`模型的`predict()`方法来提取特征，作为我们的训练、验证和测试数据集。让我们遍历我们的数据集以提取预训练模型的特征。

```py
def get_features_and_labels(dataset):
    all_features = []
    all_labels = []
    for images, labels in dataset:
        preprocessed_images = preprocessor(images)
        features = conv_base.predict(preprocessed_images, verbose=0)
        all_features.append(features)
        all_labels.append(labels)
    return np.concatenate(all_features), np.concatenate(all_labels)

train_features, train_labels = get_features_and_labels(train_dataset)
val_features, val_labels = get_features_and_labels(validation_dataset)
test_features, test_labels = get_features_and_labels(test_dataset) 
```

列表 8.25：提取图像特征和相应的标签

重要的是，`predict()`只期望图像，而不是标签，但我们的当前数据集生成的批次包含图像及其标签。

提取的特征当前形状为`(samples, 6, 6, 2048)`：

```py
>>> train_features.shape
(2000, 6, 6, 2048)
```

在这一点上，你可以定义你的密集连接分类器（注意正则化时使用`dropout`），并在你刚刚记录的数据和标签上对其进行训练。

```py
inputs = keras.Input(shape=(6, 6, 2048))
# Averages spatial dimensions to flatten the feature map
x = layers.GlobalAveragePooling2D()(inputs)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="feature_extraction.keras",
        save_best_only=True,
        monitor="val_loss",
    )
]
history = model.fit(
    train_features,
    train_labels,
    epochs=10,
    validation_data=(val_features, val_labels),
    callbacks=callbacks,
) 
```

代码列表 8.26：定义和训练密集连接分类器

训练非常快，因为你只需要处理两个`Dense`层——即使是在`CPU`上，一个`epoch`也只需不到 1 秒。

让我们看看训练期间的损失和准确率曲线（见图 8.13）。

```py
import matplotlib.pyplot as plt

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, "r--", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "r--", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show() 
```

代码列表 8.27：绘制结果

![](img/605acfe8452fb67a1808d09146951f69.png) ![](img/df8ababe79bc755a8d10cdfc5323b506.png)

图 8.13：平面特征提取的训练和验证指标

你达到了略高于 98% 的验证准确率——比你在上一节中使用从头开始训练的小模型所达到的准确率要好得多。然而，这种比较有点不公平，因为`ImageNet`包含许多狗和猫的实例，这意味着我们的预训练模型已经具备了完成任务所需的精确知识。当你使用预训练特征时，这种情况并不总是如此。

然而，这些图表也表明你几乎从一开始就过度拟合——尽管使用了相当大的`dropout`率。这是因为这项技术没有使用数据增强，这对于防止使用小型图像数据集时的过度拟合是至关重要的。

让我们检查测试准确率：

```py
test_model = keras.models.load_model("feature_extraction.keras")
test_loss, test_acc = test_model.evaluate(test_features, test_labels)
print(f"Test accuracy: {test_acc:.3f}") 
```

我们得到了 98.1% 的测试准确率——与从头开始训练模型相比，这是一个非常好的改进！

## 特征提取与数据增强相结合

现在，让我们回顾我们提到的第二个用于特征提取的技术，它速度较慢且成本更高，但允许你在训练期间使用数据增强：创建一个将`conv_base`与新的密集分类器链式连接的模型，并在输入上从头到尾进行训练。

要做到这一点，我们首先冻结卷积基。*冻结*一个层或一组层意味着防止它们在训练期间更新权重。在这里，如果你不这样做，那么卷积基之前学习到的表示将在训练期间被修改。因为顶部的`Dense`层是随机初始化的，非常大的权重更新将通过网络传播，实际上破坏了之前学习到的表示。

在`Keras`中，你可以通过将层的`trainable`属性设置为`False`来冻结一个层或模型。

```py
import keras_hub

conv_base = keras_hub.models.Backbone.from_preset(
    "xception_41_imagenet",
    trainable=False,
) 
```

代码列表 8.28：创建冻结的卷积基

将`trainable`设置为`False`将清空层或模型的可训练权重列表。

```py
>>> conv_base.trainable = True
>>> # The number of trainable weights before freezing the conv base
>>> len(conv_base.trainable_weights)
154
>>> conv_base.trainable = False
>>> # The number of trainable weights after freezing the conv base
>>> len(conv_base.trainable_weights)
0
```

代码列表 8.29：冻结前后的可训练权重列表

现在，我们只需创建一个新的模型，将冻结的卷积基础和密集分类器链接在一起，如下所示：

```py
inputs = keras.Input(shape=(180, 180, 3))
x = preprocessor(inputs)
x = conv_base(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256)(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
) 
```

使用这种设置，只有你添加的两个`Dense`层的权重将被训练。总共是四个权重张量：每个层两个（主权重矩阵和偏置向量）。请注意，为了使这些更改生效，你必须首先编译模型。如果你在编译后修改权重的可训练性，那么你应该重新编译模型，否则这些更改将被忽略。

让我们训练我们的模型。我们将重用我们的增强数据集`augmented_train_dataset`。多亏了数据增强，模型开始过度拟合需要更长的时间，因此我们可以训练更多的轮次——让我们做 30 轮：

```py
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="feature_extraction_with_data_augmentation.keras",
        save_best_only=True,
        monitor="val_loss",
    )
]
history = model.fit(
    augmented_train_dataset,
    epochs=30,
    validation_data=validation_dataset,
    callbacks=callbacks,
) 
```

让我们再次绘制结果（见图 8.14）。这个模型达到了 98.2% 的验证准确率。

![图 8.14](img/8e3a8a5c36838e7499ef2d4ccb9849c0.png) ![图 8.15](img/ae553ac8e7101ab0653eeff0e1edba20.png)

图 8.14：使用数据增强进行特征提取的训练和验证指标

让我们检查测试准确率。

```py
test_model = keras.models.load_model(
    "feature_extraction_with_data_augmentation.keras"
)
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}") 
```

列表 8.30：在测试集上评估模型

我们得到了 98.4% 的测试准确率。这并没有比之前的模型有改进，这有点令人失望。这可能表明我们的数据增强配置并不完全符合测试数据的分布。让我们看看我们能否在我们的最新尝试中做得更好。

## 微调预训练模型

另一种广泛使用的模型重用技术，与特征提取互补的是*微调*（见图 8.15）。微调包括解冻用于特征提取的冻结模型基础，并联合训练模型的新添加部分（在这种情况下，全连接分类器）和基础模型。这被称为*微调*，因为它略微调整了被重用的模型的更抽象的表示，使其更相关于当前的问题。

我们之前提到，为了能够在顶部训练一个随机初始化的分类器，首先需要冻结预训练的卷积基础。出于同样的原因，只有在顶部的分类器已经训练好的情况下，才能微调卷积基础。如果分类器尚未训练，那么在训练过程中通过网络传播的错误信号将太大，并且之前由正在微调的层学到的表示将被破坏。因此，微调网络的步骤如下：

1.  在已经训练好的基础网络上添加你的自定义网络。
2.  解冻基础网络。
3.  训练你添加的部分。
4.  解冻基础网络。
5.  联合训练这两个层以及你添加的部分。

注意，你不应该解冻“批量归一化”层（`BatchNormalization`）。批量归一化及其对微调的影响将在下一章中解释。

当进行特征提取时，你已经完成了前三个步骤。让我们继续进行第 4 步：你将解冻你的`conv_base`。

让我们以一个非常低的学习率开始微调模型。使用低学习率的原因是，你希望限制你对正在微调的层的表示所做的修改的幅度。过大的更新可能会损害这些表示。

```py
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    metrics=["accuracy"],
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="fine_tuning.keras",
        save_best_only=True,
        monitor="val_loss",
    )
]
history = model.fit(
    augmented_train_dataset,
    epochs=30,
    validation_data=validation_dataset,
    callbacks=callbacks,
) 
```

列表 8.31：微调模型

你现在可以最终在测试数据上评估这个模型了（见图 8.15）：

```py
model = keras.models.load_model("fine_tuning.keras")
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}") 
```

![图片 1](img/a4eed2da3b1ba70dcee533725f83fcc8.png) ![图片 2](img/6549d0428cc9b3aa20c7b92abef1c8e1.png)

![图 8.15](img/#figure-8-15)：微调的训练和验证指标

在这里，你得到了 98.6% 的测试准确率（再次强调，你的结果可能在半个百分点之内）。在围绕这个数据集的原始`Kaggle`竞赛中，这将是一份顶尖的结果。然而，这种比较并不完全公平，因为你使用了已经包含有关猫和狗先前知识的预训练特征，而当时的竞争对手无法使用这些特征。

在积极的一面，通过使用现代深度学习技术，你只用到了竞赛中可用训练数据的一小部分（大约 10%）就达到了这个结果。与训练 2,000 个样本相比，能够在 20,000 个样本上进行训练有着巨大的差异！

现在你已经拥有了一套处理图像分类问题（特别是小型数据集）的强大工具。

## 摘要

+   卷积神经网络在计算机视觉任务中表现出色。即使是在非常小的数据集上，从头开始训练一个卷积神经网络也是可能的，并且可以得到相当不错的结果。
+   卷积神经网络通过学习一系列模块化的模式和概念来表示视觉世界，从而工作。
