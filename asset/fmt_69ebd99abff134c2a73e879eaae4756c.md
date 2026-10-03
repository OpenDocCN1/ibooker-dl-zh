
ELMo 包含卷积和循环（特别是*长短期记忆* [LSTM]）元素，而合适命名的 BERT 是基于 transformers 的。这些术语在第一章中介绍过，并将在后续章节中更详细地讨论。我们采用了最简单的迁移学习微调形式，在对应的预训练嵌入之上，通过我们前几节的标签数据集训练了一个单独的密集分类层。

## 3.2.1 语言模型的嵌入 (ELMo)

语言模型嵌入（ELMo）模型以热门的*Sesame Street*角色命名，是最早证明在一般 NLP 任务中将预训练的语言模型知识转移的有效性的模型之一。该模型被训练以预测一系列单词中的下一个单词，在非监督的大型语料库上可以进行，结果显示得到的权重可以推广到各种其他 NLP 任务。我们将不会在本节详细讨论该模型的架构——我们将在后面的章节中讨论该问题。在这里，我们专注于建立直觉，但足够提到的是，该模型利用了字符级卷积来构建每个单词标记的初步嵌入，接着是双向 LSTM 层，将上下文信息引入模型产生的最终嵌入中。

简要介绍了 ELMo 之后，让我们开始为两个运行示例数据集中的每个数据集训练它。ELMo 模型可以通过 TensorFlow Hub 获得，TensorFlow Hub 提供了一个简单的平台用于共享 TensorFlow 模型。我们将使用使用 TensorFlow 作为后端的 Keras 来构建我们的模型。为了使 TensorFlow Hub 模型可以被 Keras 使用，我们需要定义一个自定义的 Keras 层，以正确的格式实例化它。下面的代码段展示了如何实现这个功能。

**代码段 3.4：将 TensorFlow Hub ELMo 实例化为自定义的 Keras 层**

```py
import tensorflow as tf                                                 ❶
import tensorflow_hub as hub
from keras import backend as K
import keras.layers as layers
from keras.models import Model, load_model
from keras.engine import Layer
import numpy as np

sess = tf.Session()                                                     ❷
K.set_session(sess)

class ElmoEmbeddingLayer(Layer):                                        ❸
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable=True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo =hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))      ❹

        self.trainable_weights += 
                         K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))  ❺
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                      as_dict=True,
                      signature='default',
                      )['default']
        return result

    def compute_output_shape(self, input_shape):                        ❻
        return (input_shape[0], self.dimensions)
```

❶ 导入所需的依赖项

❷ 初始化会话

❸ 创建一个自定义层，允许我们更新权重

❹ 从 TensorFlow Hub 下载预训练的 ELMo 模型

❺ 提取可训练参数——ELMo 模型层加权平均值中的四个权重；更多详细信息请参阅之前的 TensorFlow Hub 链接

❻ 指定输出的形状

在使用这个函数来训练模型之前，我们需要对我们的预处理数据进行一些调整，以适应这个模型结构。特别是，回想一下，我们为传统模型组装了变量`raw_data`的词袋表示，该变量由第 2.7 节的代码生成，并且这是一个包含每封电子邮件的单词标记列表的 NumPy 数组。在这种情况下，我们将使用第 3.5 节的函数和代码将每个这样的列表合并成一个单一的文本字符串。这是 ELMo TensorFlow Hub 模型期望的输入格式，我们很高兴满足要求。

**注意**：深度学习实践中，由于人工神经网络极具发现重要性和非重要性的神奇能力，通常不需要去除停用词这一步骤。但在我们的情况下，因为我们试图比较不同模型类型在这个问题上的优点和缺点，对所有算法应用相同的预处理是有意义的，同时也可以说是正确的方法。然而，需要注意的是，ELMo 和 BERT 都是在包含停用词的语料库上进行预训练的。

**列表 3.5 将数据转换为 ELMo TensorFlow Hub 模型所需的形式**

```py
def convert_data(raw_data,header):                              ❶
    converted_data, labels = [], []
    for i in range(raw_data.shape[0]):
        out = ' '.join(raw_data[i])                             ❷
        converted_data.append(out)
        labels.append(header[i])
    converted_data = np.array(converted_data, dtype=object)[:, np.newaxis]

    return converted_data, np.array(labels)

raw_data, header = unison_shuffle(raw_data, header)             ❸

idx = int(0.7*data_train.shape[0])
train_x, train_y = convert_data(raw_data[:idx],header[:idx])    ❹
test_x, test_y = convert_data(raw_data[idx:],header[idx:])      ❺
```

❶ 将数据转换为正确的格式

❷ 将每封邮件的标记连接成一个字符串

❸ 首先对原始数据进行洗牌

❹ 将 70%的数据转换为训练数据

❺ 将剩余的 30%的数据转换为测试数据

在将数据转换为正确的格式之后，我们使用下一个列表中的代码构建和训练 Keras ELMo TensorFlow Hub 模型。

**列表 3.6 使用列表 3.4 中定义的自定义层构建 ELMo Keras 模型**

```py
def build_model(): 
  input_text = layers.Input(shape=(1,), dtype="string")
  embedding = ElmoEmbeddingLayer()(input_text)
  dense = layers.Dense(256, activation='relu')(embedding)    ❶
  pred = layers.Dense(1, activation='sigmoid')(dense)        ❷

  model = Model(inputs=[input_text], outputs=pred)

  model.compile(loss='binary_crossentropy', optimizer='adam',
                                 metrics=['accuracy'])       ❸
  model.summary()                                            ❹

  return model

# Build and fit
model = build_model()
model.fit(train_x,                                           ❺
          train_y,
          validation_data=(test_x, test_y),
          epochs=5,
          batch_size=32)
```

❶ 输出 256 维特征向量的新层

❷ 分类层

❸ 损失、度量和优化器选择

❹ 显示模型架构以进行检查

❺ 对模型进行五个时期的拟合

在这里需要注意几点，因为这是我们第一次接触深度学习设计的一些详细方面。首先，注意到我们在预训练的 ELMo 嵌入之上添加了一个额外的层，产生了 256 维的特征向量。我们还添加了一个输出维数为 1 的分类层。激活函数`sigmoid`将其输入转换为 0 到 1 之间的区间，并且在本质上是图 2.6 中的逻辑曲线。我们可以将其输出解释为正类别的概率，并且当它超过某个预先指定的阈值（通常为 0.5）时，我们可以将相应的网络输入分类为正类别。

模型在整个数据集上进行了五个“主要步骤”或时期的拟合。列表 3.6 中的 Keras 代码语句`model.summary()`打印出模型的详细信息，并产生以下输出：

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         (None, 1)                 0         
_________________________________________________________________
elmo_embedding_layer_2 (Elmo (None, 1024)              4         
_________________________________________________________________
dense_3 (Dense)              (None, 256)               262400    
_________________________________________________________________
dense_4 (Dense)              (None, 2)                 514       
=================================================================
Total params: 262,918
Trainable params: 262,918
Non-trainable params: 0
```

我们注意到——不深入进一步的细节，因为这将在第四章中讨论——在这种情况下，大多数可训练参数（大约 26 万个）来自于我们添加在自定义 ELMo 模型之上的层。这是我们第一个使用迁移学习的实例：在 ELMo 的创建者共享的预训练模型之上学习一对新的层。对于大多数神经网络实验而言，使用强大的 GPU 是非常重要的，并且`batch_size`参数的值（指定每一步向 GPU 输入的数据量）对于收敛速度非常重要。它将根据所使用的 GPU 或缺少 GPU 而有所不同。在实践中，可以增加这个参数的值，直到典型问题实例的收敛速度不再因增加而获益，或者 GPU 的内存在算法的一次迭代中已经不足以容纳单个数据批次为止。此外，在处理多 GPU 的情况下，已经实验证明¹批大小的最佳扩展计划与 GPU 的数量呈线性关系。

通过 Kaggle Kernel（请看我们的配套 GitHub 存储库²获得 Kaggle 笔记本链接）上的一个免费的 NVIDIA Tesla K80 GPU，在我们的电子邮件数据集上，我们在图 3.3 中显示了前五个时代的典型运行性能。我们发现`batch_size`为 32 在那个环境中能很好地工作。

![03_03](img/03_03.png)

**图 3.3** 显示了在电子邮件分类示例中训练 ELMo 模型的前五个时代的验证和训练准确性比分的趋势。

每个时代需要大约 10 秒来完成——这些信息是由我们的代码打印的。我们看到第四个时代达到了约 97.3%的验证准确性（意味着在不到一分钟的时间内达到了结果）。这个性能与 Logistic Regression 方法的性能相当，后者只稍好一点，为 97.7%（也见表 3.1）。我们注意到这个算法的行为是随机的——它的运行在每次运行时都表现出不同的行为。因此，即使在与我们使用的类似的架构上，你自己的收敛性也会有所不同。实践中通常尝试几次运行算法，并在随机和不同的结果中选择最佳的参数集。最后，我们注意到训练和验证准确性的背离表明了过拟合的开始，正如图中所示的那样。这证实了将通过增加超参数`maxtokenlen`指定的标记长度以及通过`maxtokens`指定的每封电子邮件的标记数量来增加信号量的数量可能会进一步提高性能的假设。自然地，通过打开`Nsamp`来增加每类样本的数量也应该有助于提高性能。

对于 IMDB 示例，ELMo 模型代码产生了图 3.4 所示的收敛输出。

![03_04](img/03_04.png)

**图 3.4** 显示了在 IMDB 电影评论分类示例上训练 ELMo 模型的前五个时代的验证和训练准确性比分的趋势。

每个时代再次需要大约 10 秒，并且在第二个时代不到一分钟的时间内就实现了约 70%左右的验证准确性。我们将看到如何在本章的下一个和最后一个部分提高这些模型的性能。请注意，在第三个以及之后的时代，可以观察到一些过拟合的证据，因为训练准确性继续提高——对数据的拟合改善了，而验证准确性仍然较低。

## 3.2.2 双向编码器表示来自转换（BERT）

双向编码器表示来自变换器（BERT）模型也是以流行的*Sesame Street*角色命名的，以向 ELMo 开始的趋势致敬。在撰写本文时，其变体在将预训练语言模型知识转移到下游自然语言处理任务方面取得了一些最佳性能。该模型同样被训练来预测词语序列中的词语，尽管确切的*masking*过程略有不同，将在本书后面详细讨论。它也可以在非常大的语料库上以无监督的方式进行，并且生成的权重同样适用于各种其他自然语言处理任务。可以说，要熟悉自然语言处理中的迁移学习，熟悉 BERT 也是不可或缺的。

就像我们对 ELMo 所做的那样，在本节中我们将避免完全详细讨论这个深度学习模型的架构——我们将在后面的章节中涵盖这个话题。在这里提一下，模型利用字符级卷积来构建词元的初步嵌入，然后是基于变换器的编码器，其中包含自注意层，为模型提供周围单词的上下文。变换器在功能上取代了 ELMo 所采用的双向 LSTM 的作用。回顾第一章中，变换器相对于 LSTM 在训练可扩展性方面具有一些优势，我们可以看到这个模型背后的一些动机。同样，我们将使用带有 TensorFlow 后端的 Keras 来构建我们的模型。

简要介绍了 BERT 之后，让我们继续为两个运行示例数据集中的每一个训练它。BERT 模型也可以通过 TensorFlow Hub 获得。为了使 hub 模型能够被 Keras 使用，我们同样定义了一个自定义 Keras 层，以正确的格式实例化它，如下一个清单所示。

[使用自定义 Keras 层实例化 TensorFlow Hub BERT](https://wiki.example.org/instantiating_tensorflow_hub_bert_as_custom_keras_layer)
