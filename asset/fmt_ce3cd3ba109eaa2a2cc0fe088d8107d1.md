
*超参数*是指定用于控制模型训练过程的值或属性。想法是您希望使用这些值的组合来训练模型，并确定最佳组合。知道哪种组合最好的唯一方法是尝试这些值，因此您希望以有效的方式迭代所有组合并将选择范围缩小到最佳组合。超参数通常应用于模型架构，比如深度学习神经网络中密集层中的节点数。超参数可以应用于训练例程，比如执行反向传播的优化器（您在第八章中学到的）在训练过程中。它们还可以应用于*学习率*，指定您希望使用增量更改来更新模型的权重和偏差的程度，从而确定反向传播在训练过程中更新权重和偏差的步长有多大。综合起来，超参数可以是数值（节点数、学习率）或非数值（优化器名称）。

截至 TensorFlow 分发 2.4.1 版本，Keras Tuner 库尚未成为标准分发的一部分。这意味着您需要安装它。您可以在命令终端或 Google Colab 笔记本中运行此安装语句：

```py
pip install keras-tuner
```

如果在 Jupyter Notebook 单元格中运行它，请使用此版本：

```py
!pip install keras-tuner
```

感叹号（！）告诉笔记本单元格将其解释为命令而不是 Python 代码。

安装完成后，您将像往常一样导入它。请注意，当您安装库时，库名称是带连字符的，但在导入语句中不使用连字符：

```py
import kerastuner as kt
```

从 Keras Tuner 的角度来看，超参数有三种数据类型：整数、项目选择（一组离散值或对象）和浮点数。

## 整数列表作为超参数

通过示例更容易看到如何使用 Keras Tuner，所以假设您想尝试在密集层中使用不同数量的节点。首先定义可能的数字，然后将该列表传递给密集层的定义：

```py
hp_node_count = hp.Int('units', min_value = 16, max_value = 32, 
step = 8)
tf.keras.layers.Dense(units = hp_node_count, activation = 'relu')
```

在上述代码中，`hp.Int`定义了一个别名：`hp_node_count`。这个别名包含一个整数列表（16、24 和 32），您将其作为`units`传递给`Dense`层。您的目标是看哪个数字效果最好。

## 项目选择作为超参数

设置超参数的另一种方法是将所有选择放在一个列表中作为离散项或选择。这可以通过`hp.Choice`函数实现：

```py
hp_units = hp.Choice('units', values = [16, 18, 21])
```

以下是通过名称指定激活函数的示例：

```py
hp_activation = hp.Choice('dense_activation', 
                values=['relu', 'tanh', 'sigmoid'])
```

## 浮点值作为超参数

在许多情况下，您可能希望在训练例程中尝试不同的小数值。如果您想为优化器选择学习率，这是非常常见的。要这样做，请使用以下命令：

```py
hp_learning_rate = hp.Float('learning_rate', 
 min_value = 1e-4, 
 max_value = 1e-2, 
 step = 1e-3)
optimizer=tf.keras.optimizers.SGD(
 lr=hp_learning_rate, 
 momentum=0.5)
```

# 端到端超参数调整

超参数调整是一个耗时的过程。它涉及尝试多种组合，每种组合都经过相同数量的 epochs 训练。对于“蛮力”方法，您需要循环遍历每个超参数组合，*然后*启动训练例程。

使用 Keras Tuner 的优势在于其提前停止实现：如果特定组合似乎没有改善结果，它将终止训练例程并转移到下一个组合。这有助于减少总体训练时间。

接下来，您将看到如何使用一种称为*超带搜索*的策略执行和优化超参数搜索。超带搜索利用训练过程中的逐渐减少原则。在每次循环中，算法会对所有超参数组合的模型表现进行排名，并丢弃参数组合中较差的一半。下一轮中，较好的组合将获得更多的处理器核心和内存。这将一直持续到最后一种组合保留下来，消除所有但最佳超参数组合。

这有点像体育比赛中的[淘汰赛](https://oreil.ly/jq8Lb)：每一轮和每一场比赛都会淘汰排名较低的球队。然而，在超带搜索中，失败的球队在比赛完成之前就被宣布失败了。这个过程会一直持续到冠军赛，冠军赛中排名第一的球队最终成为冠军。这种策略比蛮力方法要节约得多，因为每种组合都会被训练到完整的 epochs，这会消耗大量的训练资源和时间。

让我们将您在上一章中使用的 CIFAR-10 图像分类数据集中学到的知识应用起来。

## 导入库和加载数据

我建议使用 Google Colab 笔记本电脑和 GPU 来运行此示例中的代码。截至 TensorFlow 2.4.1，Keras Tuner 尚未成为 TensorFlow 分发的一部分，因此在 Google Colab 环境中，您需要运行`pip install`命令：

```py
pip install -q -U keras-tuner
```

安装完成后，将其与所有其他库一起导入：

```py
import tensorflow as tf
import kerastuner as kt
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pylab as plt
import os
from datetime import datetime
print(tf.__version__)
```

它将显示当前版本的 TensorFlow——在本例中为 2.4.1。然后在一个单元格中加载和归一化图像：

```py
(train_images, train_labels), (test_images, test_labels) = 
datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, 
test_images / 255.0
```

现在提供一个标签列表：

```py
# Plain-text name in alphabetical order. 
https://www.cs.toronto.edu/~kriz/cifar.html
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat',
               'deer','dog', 'frog', 'horse', 'ship', 'truck']
```

接下来，通过将图像和标签合并为张量将图像转换为数据集。将测试数据集分为两组——前 500 张图像（用于训练期间的验证）和其他所有图像（用于测试）：

```py
validation_dataset = tf.data.Dataset.from_tensor_slices(
 (test_images[:500], test_labels[:500]))

test_dataset = tf.data.Dataset.from_tensor_slices(
 (test_images[500:], test_labels[500:]))

train_dataset = tf.data.Dataset.from_tensor_slices(
 (train_images, train_labels))
```

要确定每个数据集的样本大小，请执行以下命令：

```py
train_dataset_size = len(list(train_dataset.as_numpy_iterator()))
print('Training data sample size: ', train_dataset_size)

validation_dataset_size = len(list(validation_dataset.
as_numpy_iterator()))
print('Validation data sample size: ', validation_dataset_size)

test_dataset_size = len(list(test_dataset.as_numpy_iterator()))
print('Test data sample size: ', test_dataset_size)
```

您应该会得到类似以下输出：

```py
Training data sample size:  50000
Validation data sample size:  500
Test data sample size:  9500
```

接下来，为了利用分布式训练，定义一个`MirroredStrategy`对象来处理分布式训练：

```py
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(
strategy.num_replicas_in_sync))
```

在您的 Colab 笔记本中，您应该看到以下输出：

```py
Number of devices: 1
```

现在设置样本批处理参数：

```py
BUFFER_SIZE = 10000
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
STEPS_PER_EPOCH = train_dataset_size // BATCH_SIZE_PER_REPLICA
VALIDATION_STEPS = 1
```

对所有数据集进行洗牌和批处理：

```py
train_dataset = train_dataset.repeat().shuffle(BUFFER_SIZE).batch(
BATCH_SIZE)
validation_dataset = validation_dataset.shuffle(BUFFER_SIZE).batch(
validation_dataset_size)
test_dataset = test_dataset.batch(test_dataset_size)
```

现在您可以创建一个函数来包装模型架构：

```py
def build_model(hp):
  model = tf.keras.Sequential()
  # Node count for next layer as hyperparameter
  hp_node_count = hp.Int('units', min_value=16, max_value=32, 
      step=8)
  model.add(tf.keras.layers.Conv2D(filters = hp_node_count,
      kernel_size=(3, 3),
      activation='relu',
      name = 'conv_1',
      kernel_initializer='glorot_uniform',
      padding='same', input_shape = (32,32,3)))
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(tf.keras.layers.Flatten(name = 'flat_1'))
  # Activation function for next layer as hyperparameter
  hp_AF = hp.Choice('dense_activation', 
      values = ['relu', 'tanh'])
  model.add(tf.keras.layers.Dense(256, activation=hp_AF,
      kernel_initializer='glorot_uniform',
      name = 'dense_1'))
  model.add(tf.keras.layers.Dense(10, 
      activation='softmax',
      name = 'custom_class'))

  model.build([None, 32, 32, 3])
  # Compile model with optimizer
  # Learning rate as hyperparameter
  hp_LR = hp.Float('learning_rate', 1e-2, 1e-4)

  model.compile(
     loss=tf.keras.losses.SparseCategoricalCrossentropy(
          from_logits=True),
       optimizer=tf.keras.optimizers.Adam(
          learning_rate=hp_LR),
       metrics=['accuracy'])

  return model
```

这个函数与您在第九章中看到的函数之间有一些主要区别。该函数现在期望一个输入对象`hp`。这意味着该函数将由名为`hp`的超参数调整对象调用。

在模型架构中，第一层`conv_1`的节点计数通过使用`hp_node_count`进行超参数搜索。层`dense_1`的激活函数也通过使用`hp_AF`进行超参数搜索进行声明。最后，`optimizer`中的学习率通过`hp_LR`进行超参数搜索进行声明。此函数返回具有声明的超参数的模型。

接下来，使用`kt.Hyperband`定义一个对象（`tuner`），该对象将`build_model`函数作为输入：

```py
tuner = kt.Hyperband(hypermodel = build_model,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='hp_dir',
                     project_name='hp_kt')
```

您传递以下输入以定义`tuner`对象：

`超模型`

定义模型架构和优化器的函数。

`目标`

用于评估模型性能的训练指标。

`max_epochs`

模型训练的最大时期数。

`因子`

每个框架中的时期和模型数量的减少因子。排名在前 1/因子的模型被选中并进入下一轮训练。如果因子是 2，那么前一半将进入下一轮。如果因子是 4，那么前一四分之一将进入下一轮。

`目录`

将结果写入的目标目录，例如每个模型的检查点。

`project_name`

在目标目录中保存的所有文件的前缀。

在这里，您可以定义一个提前停止，如果验证准确性在五个时期内没有改善，则停止训练：

```py
early_stop = tf.keras.callbacks.EarlyStopping(
 monitor='val_accuracy', 
 patience=5)
```

现在您已准备好通过 Hyperband 算法启动搜索。当搜索完成时，它将打印出最佳超参数：

```py
tuner.search(train_dataset,
             steps_per_epoch = STEPS_PER_EPOCH,
             validation_data = validation_dataset,
             validation_steps = VALIDATION_STEPS,
             epochs = 15,
             callbacks = [early_stop]
             )
# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units 
in conv_1 layer is {best_hps.get('units')} and the optimal 
learning rate for the optimizer is {best_hps.get('learning_rate')} 
and the optimal activation for dense_1 layer
is {best_hps.get('dense_activation')}.
""")
```

正如您所看到的，搜索后，`best_hps`保存了有关最佳超参数值的所有信息。

当您在带有 GPU 的 Colab 笔记本中运行此示例时，通常需要大约 10 分钟才能完成。期望看到类似以下的输出：
