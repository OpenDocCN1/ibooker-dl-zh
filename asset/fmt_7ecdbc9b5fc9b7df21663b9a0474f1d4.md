
# 使用预构建模型与迁移学习

## 11.1 TF.Keras 预构建模型

以下代码示例展示了如何使用 TF.Keras 中的预构建模型：

```
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense

model = ResNet50(include_top=False, input_shape=(100, 100, 3), pooling='avg') ❶

outputs = Dense(20, activation='softmax')(model.output)                       ❷
model = Model(model.input, outputs)

model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])                                           ❸
```

❶ 获取输入形状为 (100,100,3) 且没有最终分类器的预构建模型

❷ 添加了 20 个类别的分类器

❸ 编译模型以进行训练

对于大多数 TF.Keras 预构建模型，瓶颈层是一个全局平均池化层。这个层既作为特征图的最终池化层，又作为一个展平操作，将特征图转换为 1D 向量。在某些情况下，我们可能想用我们自己的自定义最终池化/展平层替换这个层。在这种情况下，我们要么指定参数 `pooling=None`，要么不指定它，这是默认设置。那么我们为什么要这样做呢？

为了回答这个问题，让我们回到我们的面条菜肴。假设当您训练模型时，您得到了 92% 的准确率，并希望做得更好。首先，您决定添加图像增强。嗯，我们可能不会考虑水平翻转，因为面条菜肴永远不会被倒着看到！同样，垂直翻转可能也不会有帮助，因为面条碗相当均匀（没有镜像）。我们可以跳过旋转，因为面条碗相当均匀，我们跳过缩放，因为相机到菜肴的位置是固定的。嗯，所以您问，还有什么？

关于移动碗的位置怎么样，因为碗在结账和扔掉柜台时都会移动？您这样做并得到了 94% 的准确率。但您希望更高的准确率。凭直觉，我们推测可能特征信息保留得不够，当每个最终特征图通过默认的 `GlobalAveragePooling2D` 池化减少到一个像素，然后展平成一个 1D 向量时。您查看您的模型摘要，看到最终特征图的大小是 4 × 4。因此，您决定取消默认池化，并用步长为 2 的 `MaxPooling2D` 替换它，这样每个特征图将减少到 2 × 2，4 个像素而不是一个像素，然后进行展平成一个 1D 向量。

在这个代码示例中，我们用最大池化 (`outputs = MaxPooling2D(model.outputs)`) 和展平 (`outputs = Flatten(outputs)`) 替换了瓶颈层，用于我们的 20 种面条菜肴分类器：

```
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model

model = ResNet50(include_top=False, input_shape=(100, 100, 3), pooling=None) ❶

outputs = MaxPooling2D(model.output)                                         ❷
outputs = Flatten()(ouputs)                                                  ❷

outputs = Dense(20, activation='softmax')(outputs)                           ❸

model = Model(model.input, outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
```

❶ 获取输入形状为 (100,100,3) 且不带分类器组的预建模型

❷ 将特征图池化并展平成一个 1D 向量

❸ 添加了一个 20 类别的分类器

在本节中，我们介绍了 TF.Keras 的预建模型和预训练模型。总结一下，预建模型是一个现有的模型，通常基于 SOTA 架构，其输入形状和任务组是可重新配置的，且权重未经过训练。预建模型通常用于从头开始训练模型，具有可重用性和可重新配置以适应您的数据集和任务的优势。缺点是架构可能没有针对您的数据集/任务进行调整，因此最终得到的模型在尺寸和准确性方面可能都不够高效。

预训练模型本质上与预建模型相同，只是权重已经使用另一个数据集（如 ImageNet 数据集）进行了预训练。预训练模型用于即插即用预测或迁移学习，具有通过代表性学习重用快速训练新数据集/任务并减少数据量的优势。缺点是预训练的代表性学习可能不适合您的数据集/任务领域。

在下一节中，我们将使用来自 TensorFlow Hub 存储库的预建模型介绍相同的概念。

## 11.2 TF Hub 预建模型

*TensorFlow Hub*，或 *TF Hub*，是一个开源公共仓库，包含预构建和预训练模型，比 TF.Keras 更为广泛。TF.Keras 的预构建/预训练模型适合学习和练习迁移学习，但在生产目的上提供的选项过于有限。TF Hub 包含大量预构建的 SOTA 架构、广泛的任务类别、特定领域的预训练权重以及超出 TensorFlow 组织直接提供的模型之外的公共提交。

本节涵盖了图像分类的预构建模型。TF Hub 为每个模型提供两个版本，具体描述如下：

+ 用于特定类别的图像分类的模块。这个过程与预训练模型相同。

+ 用于提取图像特征向量（瓶颈值）的模块，用于在自定义图像分类器中使用。这些分类器与 TF.Keras 中描述的新分类器相同。

我们将使用两个预构建模型，一个用于开箱即用的分类，另一个用于迁移学习。我们将从 TensorFlow Hub 的预构建模型开源仓库中下载这些模型，该仓库位于 [www.tensorflow.org/hub](https://www.tensorflow.org/hub)。

要使用 TF Hub，您首先需要安装 `tensorflow_hub` Python 模块：

```
pip install tensorflow_hub
```

在您的 Python 脚本中，通过导入 `tensorflow_hub` 模块来访问 TF Hub：

```
import tensorflow_hub as hub
```

您现在已设置好下载我们两个模型。

### 11.2.1 使用 TF Hub 预训练模型

与 TF.Keras 相比，TF Hub 在可加载的模型格式类型方面非常灵活：

+ *TF2.x SavedModel* — 在本地、REST 或云上的微服务、桌面/笔记本电脑或工作站中使用。

+ *TF Lite* — 在移动或内存受限的 IoT 设备上的应用程序服务中使用。

+ *TF.js* — 在客户端浏览器应用程序中使用。

+ *Coral* — 优化用于在 Coral Edge/IoT 设备上作为应用程序服务使用。

本节将仅涵盖 TF 2.x 的 SavedFormat 模型。要加载一个模型，您需要执行以下操作：

1. 获取 TF Hub 仓库中图像分类器模型的 URL。

2. 使用 `hub.KerasLayer()` 从指定的 URL 指定的仓库中检索模型数据。

3. 通过使用 TF.Keras sequential API 从模型数据构建一个 TF.Keras SavedModel。

4. 将输入形状指定为 (224, 224, 3)，这与预训练模型在 ImageNet 数据库上训练的输入形状相匹配。

```
model_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/classification/4" ❶

model = tf.keras.Sequential([hub.KerasLayer(model_url,
                             input_shape=(224,224,3))])                       ❷
```

❶ TF Hub 仓库中 ResNet50 v2 模型数据的存储位置

❷ 从模型数据检索并构建 SavedModel 格式的模型

当您执行 `model.summary()` 时，输出将如下所示：

```
Layer (type)                 Output Shape              Param #   
=================================================================
keras_layer_7 (KerasLayer)   (None, 1001)              25615849  
=================================================================
Total params: 25,615,849
Trainable params: 0
Non-trainable params: 25,615,849
```

现在，您可以使用该模型进行预测，这被称为*推理*。图 11.4 描述了使用 TF Hub ImageNet 预训练模型进行预测的以下步骤：

1. 获取 ImageNet 的标签（类别名称）信息，以便我们将预测的标签（数字索引）转换为类别名称。

2. 预处理图像以预测以下内容：

   + 将图像输入调整大小以匹配模型的输入：(224, 224, 3)。

   + 标准化图像数据：除以 255。

3. 对图像调用 `predict()`。

4. 使用 `np.argmax()` 返回最高概率的标签索引。

5. 将预测的标签索引转换为相应的类名。

![](img/CH11_F04_Ferlitsch.png)

列表 11.4 使用 TF Hub 的 ImageNet 预训练模型预测标签，然后使用 ImageNet 映射显示预测的类名

这里是这些五个步骤的一个示例实现。

```
path = tf.keras.utils.get_file('ImageNetLabels.txt',
'https://storage.googleapis.com/download.tensorflow.org/data/
ImageNetLabels.txt')           
imagenet_labels = np.array(open(path).read().splitlines())    ❶

import cv2
import numpy as np
data = cv2.imread('apple.png')                                ❷
data = cv2.resize(data, (224, 224))                           ❷
data = (data / 255.0).astype(np.float32)                      ❷

p = model.predict(np.asarray([data]))                         ❸
y = np.argmax(p)                                              ❸

print(imagenet_labels[y])                                     ❹
```

❶ 获取从 ImageNet 标签索引到类名的转换

❷ 预处理图像以进行预测

❸ 使用模型进行预测

❹ 将预测的标签索引转换为类名

### 11.2.2 新的分类器

对于为预训练模型构建新的分类器，我们加载相应的模型 URL，表示为模型的*特征向量*版本。这个版本加载了预训练模型，但没有模型顶部或分类器。这允许你添加自己的顶部或任务组。模型的输出是输出层。我们还可以指定一个与 TF Hub 模型默认输入形状不同的新输入形状。

以下是一个加载预训练 ResNet50 v2 模型特征向量版本的示例实现，我们将添加自己的任务组件以训练 CIFAR-10 模型。由于我们的 CIFAR-10 输入大小与 TF Hub 的 ResNet50 v2 版本不同，其大小为 (224, 224, 3)，因此我们还可以选择指定输入形状：

1. 获取 TF Hub 存储库中图像分类器模型的 URL。

2. 使用 `hub.KerasLayer()` 从由 URL 指定的存储库中检索模型数据。

3. 为 CIFAR-10 数据集指定新的输入形状为 (32, 32, 3)。

```
f_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"  ❶

f_layer = hub.KerasLayer(f_url, input_shape=(32,32,3))                     ❷
```

❶ TF Hub 存储库中 ResNet50 v2 特征向量版本模型数据的存储位置

❷ 将模型数据作为 TF.Keras 层检索并设置输入形状

这里是构建 CIFAR-10 新分类器的一个示例实现，格式为 SavedModel：

1. 使用顺序 API 创建 SavedModel。

   + 将预训练的 ResNet v2 的特征向量版本指定为模型底部。

   + 指定一个有 10 个节点（每个 CIFAR-10 类别一个）的密集层作为模型顶部。

2. 编译模型。

```
model = tf.keras.Sequential([
                             f_layer,
                             Dense(10, activation='softmax')
                            ])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
              metrics=['acc'])
```

当你执行 `model.summary()` 时，输出将如下所示：

```
Layer (type)                 Output Shape              Param #   
=================================================================
keras_layer_4 (KerasLayer)   (None, 2048)              23561152  
_________________________________________________________________
dense_2 (Dense)              (None, 10)                20490     
=================================================================
Total params: 23,581,642
Trainable params: 20,490
Non-trainable params: 23,561,152
```

到目前为止，我们已经涵盖了使用预训练模型进行即插即用预测和使用可重新配置的预构建模型进行更方便的新模型训练。接下来，我们将介绍如何使用和重新配置预训练模型以实现更高效的训练并减少新任务所需的数据。

## 11.3 领域间的迁移学习

在迁移学习中，我们使用预训练模型完成一个任务，并重新训练分类器和/或微调层以完成新任务。这个过程与我们刚刚在预构建模型上构建新分类器类似，但除此之外，模型是从头开始完全训练的。

迁移学习有两种一般方法：

+ *相似任务* — 预训练数据集和新数据集来自相似的域（例如水果到蔬菜）。

+ *不同任务* — 预训练数据集和新数据集来自不同的域（例如水果和卡车/面包车）。

### 11.3.1 相似任务

如本章前面所讨论的，在决定方法时，我们查看源（预训练）图像域和目标（新）域的相似性。越相似，我们可以重用更多现有底层而无需重新训练。例如，如果我们有一个在水果上训练的模型，那么预训练模型的底层所有层很可能可以重用而无需重新训练来构建一个用于识别蔬菜的新模型。

我们假设在底层学习到的粗略和详细特征对于新分类器将是相同的，并且可以在进入最顶层（的）分类之前直接重用。让我们考虑一些我们可以推测水果和蔬菜来自非常相似域的原因。两者都是天然食品。虽然水果通常在地面上生长，而蔬菜在地下生长，但它们在形状和质地上有相似的物理特性，以及如茎和叶等装饰。

当源域和目标域具有这种高水平相似性时，我们通常可以用新的分类器层替换现有的最顶层分类器层，冻结底层层，并仅训练分类器层。由于我们不需要学习其他层的权重/偏差，因此我们可以用大量更少的数据和更少的周期来训练新域的模型。

虽然拥有更多数据总是更好的，但相似源域和目标域之间的迁移学习提供了使用大量更小数据集进行训练的能力。关于数据集最小尺寸的两个最佳实践如下：

+ 每个类别（标签）的大小是源数据集的 10%。

+ 每个类别（标签）至少有 100 张图片。

与*新分类器*所示的方法相反，我们在训练之前修改代码以冻结所有位于最顶层分类器层之前的层。冻结可以防止这些层（的）权重/偏差在分类器（最顶层）层的训练期间被更新（重新训练）。在 TF.Keras 中，每个层都有 `trainable` 属性，默认为 `True`。

图 11.5 描述了预训练模型分类器层的重新训练；以下是步骤：

1. 使用具有预训练权重/偏差的预构建模型（ImageNet 2012）。

2. 从预构建模型中删除现有的分类器（最顶层）。

3. 冻结剩余的层。
