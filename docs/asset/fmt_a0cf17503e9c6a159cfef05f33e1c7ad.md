
此外，我们将在训练期间引入另一个增强步骤，随机对比图像（与论文中使用的相同）。不幸的是，您无法使用`ImageDataGenerator`实现这两个步骤中的任何一个。好消息是，自`TensorFlow 2.2`以来，引入了几个新的图像预处理层（[`mng.bz/yvzy`](http://mng.bz/yvzy)）。我们可以像模型中的任何其他层一样使用这些层。例如，我们像以前一样从输入开始：

```py
inp = Input(shape=(64,64,3))
```

然后导入`RandomCrop`和`RandomContrast`层，并按如下方式使用它们：

```py
from tensorflow.keras.layers.experimental.preprocessing import RandomCrop, 
➥ RandomContrast
# Cropping the image to a 56x56 sized image
crop_inp = RandomCrop(56, 56, seed=random_seed)(inp)
# Provide a random contrast between 0.7 and 1.3 where 1.0 is the original 
➥ contrast
crop_inp = RandomContrast(0.3, seed=random_seed)(crop_inp)
```

最终模型如下所示。

列表 7.10 最终 Minception 模型

```py
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dropout, 
➥ AvgPool2D, Dense, Concatenate, Flatten, BatchNormalization, Activation
➥ from tensorflow.keras.layers.experimental.preprocessing import RandomCrop, 
➥ RandomContrast
from tensorflow.keras.models import Model
from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger

inp = Input(shape=(64,64,3))                                       ❶

crop_inp = RandomCrop(56, 56, seed=random_seed)(inp)               ❷
crop_inp = RandomContrast(0.3, seed=random_seed)(crop_inp)         ❸

stem_out = stem(crop_inp)                                          ❹

inc_a = inception_resnet_a(stem_out, [(32,),(32,32), (32, 48, 64, 
➥ 384),(384,)], initializer=init)                                 ❺

red = reduction(inc_a, [(256,256,384),(384,)], initializer=init)   ❻

inc_b1 = inception_resnet_b(red, [(192,),(128,160,192),(1152,)], 
➥ initializer=init)                                               ❼
inc_b2 = inception_resnet_b(inc_b1,  [(192,),(128,160,192),(1152,)], 
➥ initializer=init)                                               ❼

avgpool1 = AvgPool2D((4,4), strides=(1,1), padding='valid')(inc_b2)
flat_out = Flatten()(avgpool1)
dropout1 = Dropout(0.5)(flat_out)
out_main = Dense(200, activation='softmax',  kernel_initializer=init, 
➥ name='final')(flat_out)                                         ❽

minception_resnet_v2 = Model(inputs=inp, outputs=out_main)         ❾
minception_resnet_v2.compile(loss=’categorical_crossentropy’, 
➥ optimizer='adam', metrics=['accuracy'])                         ❿
```

❶ 定义 64 × 64 输入层。

❷ 对输入进行随机裁剪（仅在训练期间激活随机性）。

❸ 在输入上执行随机对比度调整（仅在训练期间激活随机性）。

❹ 定义干部的输出。

❺ 定义 Inception-ResNet 块（类型 A）。

❻ 定义减少层。

❼ 定义 2 个 Inception-ResNet 块（类型 B）。

❽ 定义最终预测层。

❾ 定义模型。

❿ 使用分类交叉熵损失和 adam 优化器编译模型。

最后，我们的 Minception 模型已经准备就绪。它接收一个 64 × 64 × 3 大小的输入（与我们实现的其他模型相同）。然后在训练期间随机（在验证/测试期间居中）裁剪图像并应用随机对比度调整（在训练期间）。这些都会自动处理。接下来，处理后的输入进入网络的干部部分，产生输出干部输出，然后进入类型 A 的 Inception-ResNet 块并流入减少块。接下来，我们有两个连续的 Inception-ResNet 类型 B 块。然后是一个平均池化层，一个扁平化层，将除批次维度之外的所有维度压缩为 1。然后在输出上应用 50% 丢失率的 dropout 层。最后，具有 softmax 激活的 200 个节点的密集层生成最终输出。最后，使用分类交叉熵损失和 adam 优化器编译模型。

这结束了我们对 Minception 模型的讨论。您想知道这将如何提升我们模型的性能吗？在下一节中，我们将训练我们定义的 Minception 模型。

### 7.2.6 训练 Minception

现在我们开始训练模型了。训练过程与您已经为 Inception Net v1 模型所做的非常相似，只有一个区别。我们将使用学习率缩减计划进一步减少过拟合并改善泛化能力。在此示例中，如果模型的性能在预定义的持续时间内没有改善，学习率调度器将减少学习率（请参见下一个清单）。

清单 7.11 训练 Minception 模型

```py
import time
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from functools import partial

n_epochs=50

es_callback = EarlyStopping(monitor='val_loss', patience=10)              ❶
csv_logger = CSVLogger(os.path.join('eval','3_eval_minception.log'))      ❷
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto'    ❸
)

history = model.fit(                                                      ❹
    train_gen_aux, validation_data=valid_gen_aux, 
    steps_per_epoch=get_steps_per_epoch(int(0.9*(500*200)), batch_size),
    validation_steps=get_steps_per_epoch(int(0.1*(500*200)), batch_size),
    epochs=n_epochs, 
    callbacks=[es_callback, csv_logger, lr_callback]
)
```

❶ 设置了早停回调

❷ 设置了 CSV 记录器以记录指标

❸ 设置学习率控制回调

❹ 训练模型

在训练深度网络时，使用学习率计划而不是固定学习率非常常见。通常，我们通过在模型训练开始时使用较高的学习率，然后随着模型的进展逐渐减小学习率来获得更好的性能。这是因为，在优化过程中，当模型收敛时，您应该使步长变小（即学习率）。否则，较大的步长会使模型表现不稳定。我们可以在观察到指标没有增加时智能地执行此过程，并在固定间隔内减小学习率。在 Keras 中，您可以通过回调`ReduceLROnPlateau` ([`mng.bz/M5Oo`](http://mng.bz/M5Oo)) 轻松将此纳入模型训练中：

```py
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto'
)
```

使用回调时，您需要设置以下关键字参数：

+   `monitor`——定义观察的指标。在我们的示例中，我们将根据验证损失决定何时降低学习率。

+   `factor`——减少学习率的乘法因子。如果学习率为 0.01，0.1 的因子，这意味着在减少时学习率将为 0.001。

+   `patience`——与早停类似，等待多少个时期在指标没有改善的情况下降低学习率。

+   `mode`——与早停类似，指标的最小化/最大化是否应被视为改进。

当您训练模型时，您应该得到以下输出：

```py
Train for 703 steps, validate for 78 steps
Epoch 1/50
703/703 [==============================] - 158s 224ms/step - loss: 4.9362 - 
➥ accuracy: 0.0544 - val_loss: 13.1802 - val_accuracy: 0.0246
...
Epoch 41/50
702/703 [============================>.] - ETA: 0s - loss: 2.5830 - 
➥ accuracy: 0.6828
Epoch 00041: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
703/703 [==============================] - 136s 194ms/step - loss: 2.5831 - 
➥ accuracy: 0.6827 - val_loss: 3.4446 - val_accuracy: 0.4316
...
Epoch 47/50
702/703 [============================>.] - ETA: 0s - loss: 2.3371 - 
➥ accuracy: 0.7859
Epoch 00047: ReduceLROnPlateau reducing learning rate to 1.0000000474974514e-05.
703/703 [==============================] - 139s 197ms/step - loss: 2.3372 - 
➥ accuracy: 0.7859 - val_loss: 3.2988 - val_accuracy: 0.4720
...
Epoch 50/50
703/703 [==============================] - 137s 194ms/step - loss: 2.3124 - 
➥ accuracy: 0.7959 - val_loss: 3.3133 - val_accuracy: 0.4792
```

太棒了！通过调整模型架构，我们获得了巨大的准确性提升。我们现在有一个模型，在验证集上的准确率约为 50%（相当于准确识别了 100/200 个类别的对象，或者每个类别的图像有 50% 被准确分类）。您可以在输出中看到`ReduceLROnPlateau`回调所进行的干预。

最后，我们使用以下方式保存模型

```py
if not os.path.exists('models'):
    os.mkdir("models")
model.save(os.path.join('models', 'minception_resnet_v2.h5'))
```

接下来，我们可以在测试集上衡量模型的性能：

```py
# Load the model from disk
model = load_model(os.path.join('models','minception_resnet_v2.h5'))

# Evaluate the model
test_res = model.evaluate(test_gen_aux, steps=get_steps_per_epoch(500*50, 
➥ batch_size))
```

这应该在测试集上达到 51%的准确率。这是非常令人兴奋的消息。通过更多关注模型结构，我们几乎将之前模型的性能提升了一倍。

这是一个很好的教训，教会了我们模型架构在深度学习中的至关重要的作用。有一个误解认为深度学习是解决一切问题的灵丹妙药。不是的。例如，你不应该期望任何随意组合在一起的架构能够像一些公开发表的最新技术结果那样好。获得一个表现良好的深度网络可能是对超参数进行了数天甚至数周的优化和凭经验的选择的结果。

在下一节中，我们将利用迁移学习更快地达到更高程度的准确性。我们将下载一个预训练模型，并在特定数据集上进行微调。

注意 在一台配备 Intel Core i5 和 NVIDIA GeForce RTX 2070 8GB 的机器上，训练大约需要 1 小时 54 分钟来运行 50 个 epoch。

练习 3

你有以下卷积块，用于实现图像分类器：

```py
def my_conv_block(input, activation): 
    out_1 = tf.keras.layers.Conv2D(n_filters[0][2], (3,3), strides=(1,1), 
                    kernel_initializer=initializer, padding='same')(input)
    out_final = tf.keras.layers.BatchNormalization()(out_1)
    out_final = tf.keras.layers.Activation(activation)(out_final)
    return out_final  
```

你想做以下两个改变：

+   在应用激活后引入批标准化

+   从卷积层的输出到批标准化层的输出创建一个残差连接。

## 7.3 如果你无法击败它们，就加入它们：使用预训练网络增强性能

到目前为止，你已经开发了一个很好的图像分类模型，它使用各种方法防止过拟合。公司一直很满意，直到你的老板宣布镇上出现了一个表现比你开发的模型更好的新竞争对手的消息。传言是他们有一个大约 70%准确率的模型。所以，你和你的同事又回到了起点。你相信一种特殊的技术，称为迁移学习，可以帮助。具体来说，你打算使用一个在原始 ImageNet 图像分类数据集上已经训练过的 Inception-ResNet v2 的预训练版本；在 tiny-imagenet-200 数据集上对这个模型进行微调将比到目前为止实现的所有模型都提供更高的准确性。

如果你想接近最新技术水平，你必须尽一切可能获得帮助。开始这个探索的一个好方法是使用预训练模型，然后针对你的任务进行微调。预训练模型是已经在类似任务上训练过的模型。这个过程属于*迁移学习*的概念。例如，你可以很容易找到在 ILSVRC 任务上预训练过的模型。

### 7.3.1 迁移学习：在深度神经网络中重用现有知识

迁移学习是一个庞大的话题，需要单独的章节（甚至一本书）来讨论。迁移学习有许多变体。要理解迁移学习的不同方面，请参考[`ruder.io/transfer-learning/`](https://ruder.io/transfer-learning/)。一种方法是使用预训练模型并针对要解决的任务进行微调。该过程如图 7.14 所示。

![07-14](img/07-14.png)

图 7.14 迁移学习的工作原理。首先，我们从一个在解决与我们感兴趣的任务类似/相关的较大数据集上预训练的模型开始。然后，我们传输模型权重（除了最后一层），并在现有权重之上拟合一个新的预测层。最后，我们在新任务上进行微调。

首先，你在一个你已经拥有大型标记数据集的任务上训练模型（称为预训练任务）。例如，在图像分类中，你有几个大型标记数据集，包括 ImageNet 数据集。一旦你在大型数据集上训练了一个模型，你就会得到网络的权重（除了最后的预测层），并拟合一个匹配新任务的新预测层。这给了网络解决新任务的一个非常好的起点。然后，你可以用较小的数据集解决新任务，因为你已经在类似的较大数据集上训练了模型。
