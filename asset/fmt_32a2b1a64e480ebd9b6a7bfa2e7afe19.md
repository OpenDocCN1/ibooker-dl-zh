
```py
DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, strides=1,
                        padding="same", kernel_initializer="he_normal",
                        use_bias=False)

class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            tf.keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                tf.keras.layers.BatchNormalization()
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)
```

正如您所看到的，这段代码与图 14-19 非常接近。在构造函数中，我们创建所有需要的层：图中右侧的主要层和左侧的跳过层（仅在步幅大于 1 时需要）。然后在`call()`方法中，我们让输入经过主要层和跳过层（如果有的话），然后我们添加两个输出并应用激活函数。

现在我们可以使用`Sequential`模型构建一个 ResNet-34，因为它实际上只是一长串的层——现在我们有了`ResidualUnit`类，可以将每个残差单元视为一个单独的层。代码与图 14-18 非常相似：

```py
model = tf.keras.Sequential([
    DefaultConv2D(64, kernel_size=7, strides=2, input_shape=[224, 224, 3]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
])
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters

model.add(tf.keras.layers.GlobalAvgPool2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation="softmax"))
```

这段代码中唯一棘手的部分是将`ResidualUnit`层添加到模型的循环：正如前面解释的，前 3 个 RU 有 64 个滤波器，然后接下来的 4 个 RU 有 128 个滤波器，依此类推。在每次迭代中，当滤波器的数量与前一个 RU 中的数量相同时，我们必须将步幅设置为 1；否则，我们将其设置为 2；然后我们添加`ResidualUnit`，最后我们更新`prev_filters`。

令人惊讶的是，我们只需大约 40 行代码，就可以构建赢得 ILSVRC 2015 挑战的模型！这既展示了 ResNet 模型的优雅之处，也展示了 Keras API 的表现力。实现其他 CNN 架构会有点长，但并不难。不过，Keras 内置了几种这些架构，为什么不直接使用呢？

# 使用 Keras 中的预训练模型

通常，您不必手动实现标准模型，如 GoogLeNet 或 ResNet，因为在`tf.keras.applications`包中只需一行代码即可获得预训练网络。

例如，您可以使用以下代码加载在 ImageNet 上预训练的 ResNet-50 模型：

```py
model = tf.keras.applications.ResNet50(weights="imagenet")
```

就这些！这将创建一个 ResNet-50 模型，并下载在 ImageNet 数据集上预训练的权重。要使用它，您首先需要确保图像的尺寸正确。ResNet-50 模型期望 224×224 像素的图像（其他模型可能期望其他尺寸，如 299×299），因此让我们使用 Keras 的`Resizing`层（在第十三章中介绍）来调整两个示例图像的大小（在将它们裁剪到目标纵横比之后）：

```py
images = load_sample_images()["images"]
images_resized = tf.keras.layers.Resizing(height=224, width=224,
                                          crop_to_aspect_ratio=True)(images)
```

预训练模型假定图像以特定方式预处理。在某些情况下，它们可能期望输入被缩放为 0 到 1，或者从-1 到 1 等等。每个模型都提供了一个`preprocess_input()`函数，您可以用它来预处理您的图像。这些函数假设原始像素值的范围是 0 到 255，这在这里是正确的：

```py
inputs = tf.keras.applications.resnet50.preprocess_input(images_resized)
```

现在我们可以使用预训练模型进行预测：

```py
>>> Y_proba = model.predict(inputs)
>>> Y_proba.shape
(2, 1000)
```

像往常一样，输出`Y_proba`是一个矩阵，每行代表一个图像，每列代表一个类别（在本例中有 1,000 个类别）。如果您想显示前*K*个预测结果，包括类别名称和每个预测类别的估计概率，请使用`decode_predictions()`函数。对于每个图像，它返回一个包含前*K*个预测结果的数组，其中每个预测结果表示为一个包含类别标识符、其名称和相应置信度分数的数组：

```py
top_K = tf.keras.applications.resnet50.decode_predictions(Y_proba, top=3)
for image_index in range(len(images)):
    print(f"Image #{image_index}")
    for class_id, name, y_proba in top_K[image_index]:
        print(f" {class_id} - {name:12s}{y_proba:.2%}")
```

输出如下所示：

```py
Image #0
  n03877845 - palace       54.69%
  n03781244 - monastery    24.72%
  n02825657 - bell_cote    18.55%
Image #1
  n04522168 - vase         32.66%
  n11939491 - daisy        17.81%
  n03530642 - honeycomb    12.06%
```

正确的类别是 palace 和 dahlia，因此模型对第一张图像是正确的，但对第二张图像是错误的。然而，这是因为 dahlia 不是 1,000 个 ImageNet 类之一。考虑到这一点，vase 是一个合理的猜测（也许这朵花在花瓶里？），daisy 也不是一个坏选择，因为 dahlias 和 daisies 都属于同一菊科家族。

正如您所看到的，使用预训练模型创建一个相当不错的图像分类器非常容易。正如您在表 14-3 中看到的，`tf.keras.applications`中提供了许多其他视觉模型，从轻量级快速模型到大型准确模型。

但是，如果您想要为不属于 ImageNet 的图像类别使用图像分类器，那么您仍然可以通过使用预训练模型来进行迁移学习获益。

# 用于迁移学习的预训练模型

如果您想构建一个图像分类器，但没有足够的数据来从头开始训练它，那么通常可以重用预训练模型的较低层，正如我们在第十一章中讨论的那样。例如，让我们训练一个模型来对花的图片进行分类，重用一个预训练的 Xception 模型。首先，我们将使用 TensorFlow Datasets（在第十三章中介绍）加载花卉数据集：

```py
import tensorflow_datasets as tfds

dataset, info = tfds.load("tf_flowers", as_supervised=True, with_info=True)
dataset_size = info.splits["train"].num_examples  # 3670
class_names = info.features["label"].names  # ["dandelion", "daisy", ...]
n_classes = info.features["label"].num_classes  # 5
```

请注意，您可以通过设置`with_info=True`来获取有关数据集的信息。在这里，我们获取数据集的大小和类的名称。不幸的是，只有一个`"train"`数据集，没有测试集或验证集，所以我们需要拆分训练集。让我们再次调用`tfds.load()`，但这次将前 10%的数据集用于测试，接下来的 15%用于验证，剩下的 75%用于训练：

```py
test_set_raw, valid_set_raw, train_set_raw = tfds.load(
    "tf_flowers",
    split=["train[:10%]", "train[10%:25%]", "train[25%:]"],
    as_supervised=True)
```

所有三个数据集都包含单独的图像。我们需要对它们进行批处理，但首先我们需要确保它们都具有相同的大小，否则批处理将失败。我们可以使用`Resizing`层来实现这一点。我们还必须调用`tf.keras.applications.xception.preprocess_input()`函数，以适当地预处理图像以供 Xception 模型使用。最后，我们还将对训练集进行洗牌并使用预取：

```py
batch_size = 32
preprocess = tf.keras.Sequential([
    tf.keras.layers.Resizing(height=224, width=224, crop_to_aspect_ratio=True),
    tf.keras.layers.Lambda(tf.keras.applications.xception.preprocess_input)
])
train_set = train_set_raw.map(lambda X, y: (preprocess(X), y))
train_set = train_set.shuffle(1000, seed=42).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(lambda X, y: (preprocess(X), y)).batch(batch_size)
test_set = test_set_raw.map(lambda X, y: (preprocess(X), y)).batch(batch_size)
```

现在每个批次包含 32 个图像，所有图像都是 224×224 像素，像素值范围从-1 到 1。完美！

由于数据集不是很大，一点数据增强肯定会有所帮助。让我们创建一个数据增强模型，将其嵌入到我们的最终模型中。在训练期间，它将随机水平翻转图像，稍微旋转它们，并调整对比度：

```py
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode="horizontal", seed=42),
    tf.keras.layers.RandomRotation(factor=0.05, seed=42),
    tf.keras.layers.RandomContrast(factor=0.2, seed=42)
])
```

###### 提示

`tf.keras.preprocessing.image.ImageDataGenerator`类使从磁盘加载图像并以各种方式增强它们变得容易：您可以移动每个图像，旋转它，重新缩放它，水平或垂直翻转它，剪切它，或者应用任何您想要的转换函数。这对于简单的项目非常方便。然而，`tf.data`管道并不复杂，通常更快。此外，如果您有 GPU 并且将预处理或数据增强层包含在模型内部，它们将在训练过程中受益于 GPU 加速。

接下来让我们加载一个在 ImageNet 上预训练的 Xception 模型。通过设置`include_top=False`来排除网络的顶部。这将排除全局平均池化层和密集输出层。然后我们添加自己的全局平均池化层（将其输入设置为基础模型的输出），然后是一个具有每个类别一个单元的密集输出层，使用 softmax 激活函数。最后，我们将所有这些包装在一个 Keras `Model`中：

```py
base_model = tf.keras.applications.xception.Xception(weights="imagenet",
                                                     include_top=False)
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(n_classes, activation="softmax")(avg)
model = tf.keras.Model(inputs=base_model.input, outputs=output)
```

如第十一章中解释的，通常冻结预训练层的权重是一个好主意，至少在训练开始时是这样的：

```py
for layer in base_model.layers:
    layer.trainable = False
```

###### 警告

由于我们的模型直接使用基础模型的层，而不是`base_model`对象本身，设置`base_model.trainable=False`不会产生任何效果。

最后，我们可以编译模型并开始训练：

```py
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
history = model.fit(train_set, validation_data=valid_set, epochs=3)
```

###### 警告

如果你在 Colab 上运行，请确保运行时正在使用 GPU：选择运行时→“更改运行时类型”，在“硬件加速器”下拉菜单中选择“GPU”，然后点击保存。可以在没有 GPU 的情况下训练模型，但速度会非常慢（每个时期几分钟，而不是几秒）。

在训练模型几个时期后，其验证准确率应该达到 80%以上，然后停止提高。这意味着顶层现在已经训练得相当好，我们准备解冻一些基础模型的顶层，然后继续训练。例如，让我们解冻第 56 层及以上的层（这是 14 个残差单元中第 7 个的开始，如果列出层名称，你会看到）：

```py
for layer in base_model.layers[56:]:
    layer.trainable = True
```

不要忘记在冻结或解冻层时编译模型。还要确保使用更低的学习率以避免破坏预训练权重：
