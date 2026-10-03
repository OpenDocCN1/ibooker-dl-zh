
# 在 TensorBoard 中查看训练结果

在导航到 TensorBoard 地址或通过 Google Cloud 打开会话后，您应该看到一个类似于图 10-12 的页面。鉴于脚本仅每五分钟保存一次摘要，可能需要一段时间才能在图表中找到有用的内容。图 10-12 显示了训练一天以上后的结果。最重要的图表称为`clone_loss`；它显示了与日志输出中显示的相同损失值的进展。正如您在此示例中所看到的，它波动很大，但随着时间的推移总体趋势是向下的。如果在训练几个小时后没有看到这种进展，这表明您的模型没有收敛到一个好的解决方案，您可能需要调试数据集或训练参数出现的问题。

TensorBoard 在打开时默认显示 SCALARS 选项卡，但在训练期间可能有用的另一个部分是 IMAGES（图 10-13）。这显示了模型当前正在训练的图片的随机选择，包括任何扭曲和其他预处理。在图中，您可以看到图像已被翻转，并且在馈送到模型之前已被转换为灰度。这些信息并不像损失图表那样重要，但可以用来确保数据集符合您的期望，并且在训练进行时看到示例更新是有趣的。

![Tensorboard 中的训练图表](img/timl_1012.png)

###### 图 10-12. TensorBoard 中的图表

![Tensorboard 中的训练图片](img/timl_1013.png)

###### 图 10-13. TensorBoard 中的图片

# 评估模型

损失函数与模型训练的好坏相关，但它不是一个直接可理解的度量标准。我们真正关心的是模型正确检测到多少人，但要让它计算这一点，我们需要运行一个单独的脚本。您不需要等到模型完全训练，可以检查`--train_dir`文件夹中任何检查点的准确性。要执行此操作，请运行以下命令：

```cpp
! python models/research/slim/eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=vww_96_grayscale/model.ckpt-698580 \
    --dataset_dir=data/visualwakewords \
    --dataset_name=visualwakewords \
    --dataset_split_name=val \
    --model_name=mobilenet_v1_025 \
    --preprocessing_name=mobilenet_v1 \
    --use_grayscale=True \
    --train_image_size=96
```

您需要确保`--checkpoint_path`指向有效的检查点数据集。检查点存储在三个单独的文件中，因此值应为它们的公共前缀。例如，如果您有一个名为`model.ckpt-5179.data-00000-of-00001`的检查点文件，则前缀将是`model.ckpt-5179`。脚本应该生成类似于以下内容的输出：

```cpp
INFO:tensorflow:Evaluation [406/406]
I0929 22:52:59.936022 140225887045056 evaluation.py:167] Evaluation [406/406]
eval/Accuracy[0.717438412]eval/Recall_5[1]
```

这里的重要数字是准确率。它显示了被正确分类的图片的比例，在这种情况下是 72%，转换为百分比后。如果按照示例脚本进行，您应该期望完全训练的模型在一百万步后达到约 84%的准确率，并显示约 0.4 的损失。

# 将模型导出到 TensorFlow Lite

当模型训练到您满意的准确度时，您需要将结果从 TensorFlow 训练环境转换为可以在嵌入式设备上运行的形式。正如我们在之前的章节中看到的，这可能是一个复杂的过程，而`tf.slim`也添加了一些自己的特点。

## 导出到 GraphDef Protobuf 文件

Slim 每次运行其脚本时都会从`model_name`生成架构，因此要在 Slim 之外使用模型，需要将其保存为通用格式。我们将使用 GraphDef protobuf 序列化格式，因为 Slim 和 TensorFlow 的其余部分都能理解它：

```cpp
! python models/research/slim/export_inference_graph.py \
    --alsologtostderr \
    --dataset_name=visualwakewords \
    --model_name=mobilenet_v1_025 \
    --image_size=96 \
    --use_grayscale=True \
    --output_file=vww_96_grayscale_graph.pb
```

如果成功，您应该在主目录中有一个新的`vww_96_grayscale_graph.pb`文件。这包含了模型中操作的布局，但尚未包含任何权重数据。

## 冻结权重

将训练好的权重与操作图一起存储的过程称为*冻结*。这将所有图中的变量转换为常量，加载它们的值后从检查点文件中。接下来的命令使用了百万次训练步骤的检查点，但您可以提供任何有效的检查点路径。图冻结脚本存储在主 TensorFlow 存储库中，因此在运行此命令之前，您需要从 GitHub 下载这个脚本：

```cpp
! git clone https://github.com/tensorflow/tensorflow
! python tensorflow/tensorflow/python/tools/freeze_graph.py \
    --input_graph=vww_96_grayscale_graph.pb \
    --input_checkpoint=vww_96_grayscale/model.ckpt-1000000 \
    --input_binary=true --output_graph=vww_96_grayscale_frozen.pb \
    --output_node_names=MobilenetV1/Predictions/Reshape_1
```

之后，您应该看到一个名为`vww_96_grayscale_frozen.pb`的文件。

## 量化和转换为 TensorFlow Lite

量化是一个棘手而复杂的过程，仍然是一个活跃的研究领域，因此将我们迄今为止训练的浮点图转换为 8 位实体需要相当多的代码。您可以在第十五章中找到更多关于量化是什么以及它是如何工作的解释，但在这里我们将向您展示如何在我们训练的模型中使用它。大部分代码是准备示例图像以馈送到训练网络中，以便测量典型使用中激活层的范围。我们依赖`TFLiteConverter`类来处理量化并将其转换为我们需要用于推理引擎的 TensorFlow Lite FlatBuffer 文件：

```cpp
import tensorflow as tf
import io
import PIL
import numpy as np

def representative_dataset_gen():

  record_iterator = tf.python_io.tf_record_iterator
      (path='data/visualwakewords/val.record-00000-of-00010')

  count = 0
  for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)
    image_stream = io.BytesIO
        (example.features.feature['image/encoded'].bytes_list.value[0])
    image = PIL.Image.open(image_stream)
    image = image.resize((96, 96))
    image = image.convert('L')
    array = np.array(image)
    array = np.expand_dims(array, axis=2)
    array = np.expand_dims(array, axis=0)
    array = ((array / 127.5) - 1.0).astype(np.float32)
    yield([array])
    count += 1
    if count > 300:
        break

converter = tf.lite.TFLiteConverter.from_frozen_graph \
    ('vww_96_grayscale_frozen.pb', ['input'],  ['MobilenetV1/Predictions/ \
 Reshape_1'])
converter.inference_input_type = tf.lite.constants.INT8
converter.inference_output_type = tf.lite.constants.INT8
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen

tflite_quant_model = converter.convert()
open("vww_96_grayscale_quantized.tflite", "wb").write(tflite_quant_model)
```

## 转换为 C 源文件

转换器会写出一个文件，但大多数嵌入式设备没有文件系统。为了从我们的程序中访问序列化数据，我们必须将其编译到可执行文件中并存储在闪存中。最简单的方法是将文件转换为 C 数据数组，就像我们在之前的章节中所做的那样：

```cpp
# Install xxd if it is not available
! apt-get -qq install xxd
# Save the file as a C source file
! xxd -i vww_96_grayscale_quantized.tflite > person_detect_model_data.cc
```

现在，您可以用您训练过的版本替换现有的`person_detect_model_data.cc`文件，并能够在嵌入式设备上运行您自己的模型。

# 为其他类别训练

COCO 数据集中有 60 多种不同的对象类型，因此自定义模型的一种简单方法是在构建训练数据集时选择其中一种而不是`person`。以下是一个查找汽车的示例：

```cpp
! python models/research/slim/datasets/build_visualwakewords_data.py \
   --logtostderr \
   --train_image_dir=coco/raw-data/train2014 \
   --val_image_dir=coco/raw-data/val2014 \
   --train_annotations_file=coco/raw-data/annotations/instances_train2014.json \
   --val_annotations_file=coco/raw-data/annotations/instances_val2014.json \
   --output_dir=coco/processed_cars \
   --small_object_area_threshold=0.005 \
   --foreground_class_of_interest='car'
```

您应该能够按照与人员检测器相同的步骤进行操作，只需在以前的`data/visualwakewords`路径处用新的`coco/processed_cars`路径替换。

如果您感兴趣的对象类型在 COCO 中不存在，您可能可以使用迁移学习来帮助您在您收集的自定义数据集上进行训练，即使它很小。虽然我们还没有分享这方面的示例，但您可以查看[*tinymlbook.com*](http://tinymlbook.com)以获取有关这种方法的更新。

# 理解架构

[MobileNets](https://oreil.ly/tK57G)是一系列旨在提供尽可能少的权重参数和算术运算的良好准确性的架构。现在有多个版本，但在我们的情况下，我们使用原始的 v1，因为它在运行时需要的 RAM 最少。该架构背后的核心概念是*深度可分离卷积*。这是经典 2D 卷积的一种变体，以更高效的方式工作，而几乎不损失准确性。常规卷积根据在输入的所有通道上应用特定大小的滤波器来计算输出值。这意味着每个输出中涉及的计算数量是滤波器的宽度乘以高度，再乘以输入通道的数量。深度卷积将这个大计算分解为不同的部分。首先，每个输入通道通过一个或多个矩形滤波器进行滤波，以产生中间值。然后使用逐点卷积来组合这些值。这显著减少了所需的计算量，并且在实践中产生了与常规卷积类似的结果。

MobileNet v1 是由 14 个这些深度可分离卷积层堆叠而成，其中包括一个平均池化层，然后是一个全连接层，最后是一个 softmax 层。我们指定了一个*宽度乘数*为 0.25，这样可以将每次推断的计算量减少到约 6000 万次，通过将每个激活层中的通道数量缩减 75%与标准模型相比。本质上，它在操作上与普通的卷积神经网络非常相似，每一层都在学习输入中的模式。较早的层更像是边缘识别滤波器，识别图像中的低级结构，而较后的层将这些信息合成为更抽象的模式，有助于最终的对象分类。

# 总结

使用机器学习进行图像识别需要大量的数据和大量的处理能力。在本章中，您学习了如何从头开始训练模型，只提供数据集，并将该模型转换为适用于嵌入式设备的形式。

这种经验应该为您解决产品所需解决的机器视觉问题奠定了良好的基础。计算机能够看到并理解周围世界仍然有些神奇，所以我们迫不及待地想看看您会有什么创意！
