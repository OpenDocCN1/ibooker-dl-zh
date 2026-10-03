
```py
>>> conv_base.summary()
Model: "vgg16" 
_________________________________________________________________
Layer (type)                 Output Shape              Param # 
=================================================================
input_19 (InputLayer)        [(None, 180, 180, 3)]     0 
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 180, 180, 64)      1792 
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 180, 180, 64)      36928 
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 90, 90, 64)        0 
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 90, 90, 128)       73856 
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 90, 90, 128)       147584 
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 45, 45, 128)       0 
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 45, 45, 256)       295168 
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 45, 45, 256)       590080 
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 45, 45, 256)       590080 
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 22, 22, 256)       0 
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 22, 22, 512)       1180160 
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 22, 22, 512)       2359808 
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 22, 22, 512)       2359808 
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 11, 11, 512)       0 
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 11, 11, 512)       2359808 
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 11, 11, 512)       2359808 
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 11, 11, 512)       2359808 
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 5, 5, 512)         0 
=================================================================
Total params: 14,714,688 
Trainable params: 14,714,688 
Non-trainable params: 0 
_________________________________________________________________
```

最终的特征图形状为`(5, 5, 512)`。这是我们将在其上放置一个密集连接分类器的特征图。

在这一点上，我们可以有两种方式继续：

+ 运行卷积基在我们的数据集上，将其输出记录到磁盘上的 NumPy 数组中，然后使用这些数据作为输入到一个独立的、与本书第四章中看到的类似的密集连接分类器。这种解决方案运行快速且成本低，因为它只需要为每个输入图像运行一次卷积基，而卷积基是整个流程中最昂贵的部分。但出于同样的原因，这种技术不允许我们使用数据增强。

+ 通过在`conv_base`顶部添加`Dense`层来扩展我们的模型，并在输入数据上端对端地运行整个模型。这将允许我们使用数据增强，因为每个输入图像在模型看到时都会经过卷积基。但出于同样的原因，这种技术比第一种要昂贵得多。

我们将涵盖这两种技术。让我们逐步了解设置第一种技术所需的代码：记录`conv_base`在我们的数据上的输出，并使用这些输出作为新模型的输入。

## 无数据增强的快速特征提取

我们将通过在训练、验证和测试数据集上调用`conv_base`模型的`predict()`方法来提取特征作为 NumPy 数组。

让我们迭代我们的数据集以提取 VGG16 特征。

**列表 8.20 提取 VGG16 特征和相应标签**

```py
import numpy as np

def get_features_and_labels(dataset):
    all_features = []
    all_labels = []
    for images, labels in dataset:
        preprocessed_images = keras.applications.vgg16.preprocess_input(images)
        features = conv_base.predict(preprocessed_images)
        all_features.append(features)
        all_labels.append(labels)
    return np.concatenate(all_features), np.concatenate(all_labels)

train_features, train_labels =  get_features_and_labels(train_dataset)
val_features, val_labels =  get_features_and_labels(validation_dataset)
test_features, test_labels =  get_features_and_labels(test_dataset)
```

重要的是，`predict()`只期望图像，而不是标签，但我们当前的数据集产生的批次包含图像和它们的标签。此外，`VGG16`模型期望使用`keras.applications.vgg16.preprocess_input`函数预处理输入，该函数将像素值缩放到适当的范围。

提取的特征目前的形状为`(samples, 5, 5, 512)`：

```py
>>> train_features.shape
(2000, 5, 5, 512)
```

在这一点上，我们可以定义我们的密集连接分类器（注意使用了 dropout 进行正则化），并在我们刚刚记录的数据和标签上对其进行训练。

**列表 8.21 定义和训练密集连接分类器**

```py
inputs = keras.Input(shape=(5, 5, 512))
x = layers.Flatten()(inputs)               # ❶
x = layers.Dense(256)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="feature_extraction.keras",
        save_best_only=True,
        monitor="val_loss")
]
history = model.fit(
    train_features, train_labels,
    epochs=20,
    validation_data=(val_features, val_labels),
    callbacks=callbacks)
```

❶ 注意在将特征传递给密集层之前使用了 Flatten 层。

训练非常快，因为我们只需要处理两个`Dense`层——即使在 CPU 上，一个时代也不到一秒。

让我们在训练过程中查看损失和准确率曲线（见图 8.13）。

![](img/08-13.png)

**图 8.13 普通特征提取的训练和验证指标**

**列表 8.22 绘制结果**

```py
import matplotlib.pyplot as plt
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, "bo", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()
```

我们达到了约 97%的验证准确率——比我们在前一节使用从头开始训练的小模型取得的结果要好得多。然而，这有点不公平的比较，因为 ImageNet 包含许多狗和猫实例，这意味着我们预训练的模型已经具有了完成任务所需的确切知识。当您使用预训练特征时，情况并不总是如此。

然而，图表也表明我们几乎从一开始就过拟合了——尽管使用了相当大的 dropout 率。这是因为这种技术没有使用数据增强，而数据增强对于防止小图像数据集过拟合是至关重要的。

## 结合数据增强的特征提取

现在让我们回顾一下我提到的第二种特征提取技术，这种技术速度较慢，成本较高，但允许我们在训练过程中使用数据增强：创建一个将`conv_base`与新的密集分类器连接起来的模型，并在输入上端对端地进行训练。

为了做到这一点，我们首先要*冻结卷积基*。*冻结*一层或一组层意味着在训练过程中阻止它们的权重被更新。如果我们不这样做，卷积基先前学到的表示将在训练过程中被修改。因为顶部的`Dense`层是随机初始化的，非常大的权重更新会通过网络传播，有效地破坏先前学到的表示。

在 Keras 中，通过将其`trainable`属性设置为`False`来冻结一个层或模型。

**列表 8.23 实例化和冻结 VGG16 卷积基**

```py
conv_base  = keras.applications.vgg16.VGG16(
    weights="imagenet",
    include_top=False)
conv_base.trainable = False
```

将`trainable`设置为`False`会清空层或模型的可训练权重列表。

**列表 8.24 在冻结前后打印可训练权重列表**

```py
>>> conv_base.trainable = True
>>> print("This is the number of trainable weights " 
          "before freezing the conv base:", len(conv_base.trainable_weights))
This is the number of trainable weights before freezing the conv base: 26 
>>> conv_base.trainable = False
>>> print("This is the number of trainable weights " 
          "after freezing the conv base:", len(conv_base.trainable_weights))
This is the number of trainable weights after freezing the conv base: 0 
```

现在我们可以创建一个新模型，将

1. 一个数据增强阶段

2. 我们冻结的卷积基础

3. 一个密集分类器

**列表 8.25 向卷积基添加数据增强阶段和分类器**

```py
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
    ]
)

inputs = keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs)                      # ❶
x = keras.applications.vgg16.preprocess_input(x)   # ❷
x = conv_base(x)
x = layers.Flatten()(x)
x = layers.Dense(256)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])
```

❶ 应用数据增强。

❷ 应用输入值缩放。

使用这种设置，只有我们添加的两个`Dense`层的权重将被训练。总共有四个权重张量：每层两个（主要权重矩阵和偏置向量）。请注意，为了使这些更改生效，您必须首先编译模型。如果在编译后修改权重的可训练性，那么您应该重新编译模型，否则这些更改将被忽略。

让我们训练我们的模型。由于数据增强，模型开始过拟合的时间会更长，所以我们可以训练更多的 epochs——让我们做 50 个。

注意：这种技术足够昂贵，只有在您可以访问 GPU（例如 Colab 中提供的免费 GPU）时才应尝试——在 CPU 上无法实现。如果无法在 GPU 上运行代码，则应采用前一种技术。
