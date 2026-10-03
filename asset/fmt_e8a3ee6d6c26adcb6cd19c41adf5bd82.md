
第一个层是一个嵌入层，它将所有单词 ID 变为嵌入。每有一个单词 ID（`vocab_size + num_oov_buckets`），嵌入矩阵就有一行，每有一个嵌入维度，嵌入矩阵就有一列（这个例子使用了 128 个维度，这是一个可调的超参数）。模型输入是 2D 张量，形状为`[批次大小, 时间步]`，嵌入层的输出是一个 3D 张量，形状为`[批次大小, 时间步, 嵌入大小]`。

模型剩下的部分就很简单了：有两个`GRU`层，第二个只返回最后时间步的输出。输出层只有一个神经元，使用 sigmoid 激活函数，输出评论是正或负的概率。然后编译模型，利用前面准备的数据集来训练几个周期。

### 遮掩

在训练过程中，模型会学习到填充标记要被忽略掉。但这其实是已知的。为什么不告诉模型直接忽略填充标记，将精力集中在真正重要的数据中呢？只需一步就好：创建嵌入层时加上`mask_zero=True`。这意味着填充标记（其 ID 为 0）可以被接下来的所有层忽略。

其中的原理，是嵌入层创建了一个等于`K.not_equal(inputs, 0)`（其中`K = keras.backend`）遮掩张量：这是一个布尔张量，形状和输入相同，只要词 ID 有 0，它就等于`False`，否则为`True`。模型自动将这个遮掩张量向前传递给所有层，只要时间维度保留着。所以在例子中，尽管两个`GRU`都接收到了遮掩张量，但第二个`GRU`层不返回序列（只返回最后一个时间步），遮掩张量不会传递到紧密层。每个层处理遮掩的方式不同，但通常会忽略被遮掩的时间步（遮掩为`False`的时间步）。例如，当循环神经层碰到被遮掩的时间步时，就只是从前一时间步复制输出而已。如果遮掩张量一直传递到输出（输出为序列的模型），则遮掩也会作用到损失上，所以遮掩时间步不会贡献到损失上（它们的损失为 0）。

> 警告：基于英伟达的 cuDNN 库，`LSTM`层和`GRU`层针对 GPU 有优化实现。但是，这个实现不支持遮挡。如果你的模型使用了遮挡，则这些层会回滚到（更慢的）默认实现。注意优化实现还需要使用几个超参数的默认值：`activation`、`recurrent_activation`、`recurrent_dropout`、`unroll`、`use_bias`、`reset_after`。

所有接收遮挡的层必须支持遮挡（否则会抛出异常）。包括所有的循环层、`TimeDistributed`层和其它层。所有支持遮挡的层必须有等于`True`的属性`supports_masking`。如果想实现自定义的支持遮挡的层，应该给`call()`方法添加`mask`参数。另外，要在构造器中设定`self.supports_masking = True`。如果第一个层不是嵌入层，可以使用`keras.layers.Masking`层：它设置遮挡为`K.any(K.not_equal(inputs, 0), axis=-1)`，意思是最后一维都是 0 的时间步，会被后续层遮挡。

对于`Sequential`模型，使用遮挡层，并自动向前传递遮挡是最佳的。但复杂模型上不能这么做，比如将`Conv1D`层与循环层混合使用时。对这种情况，需要使用函数式 API 或子类化 API 显式计算遮挡张量，然后将其传给需要的层。例如，下面的模型等价于前一个模型，除了使用函数式 API 手动处理遮挡张量：

```py
K = keras.backend
inputs = keras.layers.Input(shape=[None])
mask = keras.layers.Lambda(lambda inputs: K.not_equal(inputs, 0))(inputs)
z = keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size)(inputs)
z = keras.layers.GRU(128, return_sequences=True)(z, mask=mask)
z = keras.layers.GRU(128)(z, mask=mask)
outputs = keras.layers.Dense(1, activation="sigmoid")(z)
model = keras.Model(inputs=[inputs], outputs=[outputs]) 
```

训练几个周期之后，这个模型的表现就相当不错了。如果使用`TensorBoard()`调回，可以可视化 TensorBoard 中的嵌入是怎么学习的：可以看到`awesome`和`amazing`这样的词渐渐聚集于嵌入空间的一边，而`awful`、`terrible`这样的词聚集到另一边。一些词可能不会像预期那样是正面的，比如`good`，可能所有负面评论含有`not good`。模型只基于 25000 个词就能学会词嵌入，让人印象深刻。如果训练集有几十亿的规模，效果就更好了。但可惜没有，但可以利用在其它大语料（比如，维基百科文章）上训练的嵌入，就算不是影评也可以？毕竟，`amazing`这个词在哪种语境的意思都差不多。另外，甚至嵌入是在其它任务上训练的，也可能有益于情感分析：因为`awesome`和`amazing`有相似的意思，即使对于其它任务（比如，预测句子中的下一个词），它们也倾向于在嵌入空间聚集，所以对情感分析也是有用的。所以看看能否重复利用预训练好的词嵌入。

### 复用预训练的词嵌入

在 TensorFlow Hub 上可以非常方便的找到可以复用的预训练模型组件。这些模型组件被称为模块。只需浏览 [TF Hub 仓库](https://links.jianshu.com/go?to=https%3A%2F%2Ftfhub.dev%2F)，找到需要的模型，复制代码到自己的项目中就行，模块可以自动下载下来，包含预训练权重，到自己的模型中。

例如，在情感分析模型中使用`nnlm-en-dim50`句子嵌入模块，版本 1：

```py
import tensorflow_hub as hub

model = keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1",
                   dtype=tf.string, input_shape=[], output_shape=[50]),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="Adam",
              metrics=["accuracy"]) 
```

`hub.KerasLayer`从给定的 URL 下载模块。这个特殊的模块是“句子编码器”：它接收字符串作为输入，将每句话编码为一个独立向量（这个例子中是 50 维度的向量）。在内部，它将字符串解析（空格分隔），然后使用预训练（训练语料是 Google News 7B，一共有 70 亿个词）的嵌入矩阵来嵌入每个词。然后计算所有词嵌入的平均值，结果是句子嵌入。我们接着可以添加两个简单的紧密层来创建一个出色的情感分析模型。默认，`hub.KerasLayer`是不可训练的，但创建时可以设定`trainable=True`，就可以针对自己的任务微调了。

> 警告：不是所有的 TF Hub 模块都支持 TensorFlow 2。

然后，就可以加载 IMDb 影评数据集了，不需要预处理（但要做批次和预提取），直接训练模型就成：

```py
datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
train_size = info.splits["train"].num_examples
batch_size = 32
train_set = datasets["train"].batch(batch_size).prefetch(1)
history = model.fit(train_set, epochs=5) 
```

注意到，TF Hub 模块的 URL 的末尾指定了是模型的版本 1。版本号可以保证当有新的模型版本发布时，不会破坏自己的模型。如果在浏览器中输入这个 URL，能看到这个模块的文档。TF Hub 会默认将下载文件缓存到系统的临时目录。你可能想将文件存储到固定目录，以免每次系统清洗后都要下载。要这么做的话，设置环境变量`TFHUB_CACHE_DIR`就成（比如，`os.environ["TFHUB_CACHE_DIR"] = "./my_tfhub_cache"`）。

截至目前，我们学习了时间序列、用 Char-RNN 生成文本、用 RNN 做情感分析、训练自己的词嵌入或复用预训练词嵌入。接下来看看另一个重要的 NLP 任务：神经网络机器翻译（NMT），我们先使用纯粹的编码器-解码器模型，然后使用注意力机制，最后看看 Transformer 架构。

## 用编码器-解码器做机器翻译

看一个简单的[神经网络机器翻译模型](https://links.jianshu.com/go?to=https%3A%2F%2Fhoml.info%2F103)，它能将英语翻译为法语（见图 16-3）。

简而言之，英语句子输入进编码器，解码器输出法语。注意，法语翻译也作为解码器的输入，但向后退一步。换句话说，解码器将前一步的输出再作为输入（不用管它输出什么）。对于第一个词，给它加上一个序列开始（SOS）标记，序列结尾加上序列结束（EOS）标记。

英语句子在输入给编码器之前，先做了翻转。例如，`I drink milk`翻转为`milk drink I`。这样能保证英语句子的第一个词是最后一个输入给编码器的，通常也是解码器要翻译的第一个词。

每个单词首先用它的 ID 来表示（例如，288 代表`milk`）。然后，嵌入层返回单词嵌入。单词嵌入才是输入给编码器和解码器的。

![](img/aef8aa3fde805533d8461794f1548f17.png)

图 16-3 一个简单的机器翻译模型

在每一步，解码器输出一个输出词典中每个单词的分数，然后 softmax 层将分数变为概率。例如，在第一步，`Je`的概率可能为 20%，`Tu`的概率可能为 1%，等等。概率最高的词作为输出。这特别像一个常规分类任务，所以可以用`"sparse_categorical_crossentropy"`损失训练模型，跟前面的 Char-RNN 差不多。

在做推断时，没有目标语句输入进解码器。相反的，只是输入解码器前一步的输出，见图 16-4（这需要一个嵌入查找表，图中没有展示）。

![](img/1e702f782691c1e3f0d0a177fb02d497.png)

图 16-4 在推断时，将前一步的输出作为输入

好了，现在知道整体的大概了。但要实现模型的话，还有几个细节要处理：

*   目前假定所有（编码器和解码器的）输入序列的长度固定。但很显然句子长度是变化的。因为常规张量的形状固定，它们只含有相同长度的句子。可以用遮挡来处理；但如果句子的长度非常不同，就不能像之前情感分析那样截断（因为想要的是完整句子的翻译）。可以将句子放进长度相近的桶里（一个桶放 1 个词到 6 个词的句子，一个桶放 7 个词到 12 个词的句子，等等），给短句子加填充，使同一个桶中的句子长度相同（见`tf.data.experimental.bucket_by_sequence_length()`函数）。例如，`I drink milk`变为`<pad> <pad> <pad> milk drink I`。

*   要忽略所有在 EOS 标记后面的输出，这些输出不能影响损失（遮挡起来）。例如，如果模型输出`Je bois du lait <eos> oui`，忽略最后一个词对损失的影响。

*   如果输出词典比较大（这个例子就是这样），输出每个词的概率会非常慢。如果目标词典有 50000 个发语词，则解码器要输出 50000 维的向量，在这个向量上计算 softmax 非常耗时。一个方法是只查看模型对正确词和非正确词采样的对数概率输出，然后根据这些对数概率计算一个大概的损失。这个采样 softmax 方法是 [Sébastien Jean 在 2015 年提出的](https://links.jianshu.com/go?to=https%3A%2F%2Fhoml.info%2F104)。在 TensorFlow 中，你可以在训练时使用`tf.nn.sampled_softmax_loss()`，在推断时使用常规 softmax 函数（推断时不能使用采样 softmax，因为需要知道目标）。

TensorFlow Addons 项目涵盖了许多序列到序列的工具，可以创建准生产的编码器-解码器。例如，下面的代码创建了一个基本的编码器-解码器模型，相似于图 16-3：

```py
import tensorflow_addons as tfa

encoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
decoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
sequence_lengths = keras.layers.Input(shape=[], dtype=np.int32)

embeddings = keras.layers.Embedding(vocab_size, embed_size)
encoder_embeddings = embeddings(encoder_inputs)
decoder_embeddings = embeddings(decoder_inputs)

encoder = keras.layers.LSTM(512, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_embeddings)
encoder_state = [state_h, state_c]

sampler = tfa.seq2seq.sampler.TrainingSampler()

decoder_cell = keras.layers.LSTMCell(512)
output_layer = keras.layers.Dense(vocab_size)
decoder = tfa.seq2seq.basic_decoder.BasicDecoder(decoder_cell, sampler,
                                                 output_layer=output_layer)
final_outputs, final_state, final_sequence_lengths = decoder(
    decoder_embeddings, initial_state=encoder_state,
    sequence_length=sequence_lengths)
Y_proba = tf.nn.softmax(final_outputs.rnn_output)

model = keras.Model(inputs=[encoder_inputs, decoder_inputs, sequence_lengths],
                    outputs=[Y_proba]) 
```

这个代码很简单，但有几点要注意。首先，创建`LSTM`层时，设置`return_state=True`，以便得到最终隐藏态，并将其传给解码器。因为使用的是 LSTM 单元，它实际返回两个隐藏态（短时和长时）。`TrainingSampler`是 TensorFlow Addons 中几个可用的采样器之一：它的作用是在每一步告诉解码器，前一步的输出是什么。在推断时，采样器是实际输出的标记嵌入。在训练时，是前一个目标标记的嵌入：这就是为什么使用`TrainingSampler`的原因。在实际中，一个好方法是，一开始用目标在前一时间步的嵌入训练，然后逐渐过渡到实际标记在前一步的输出。这个方法是 Samy Bengio 在 [2015 年的一篇论文](https://links.jianshu.com/go?to=https%3A%2F%2Fhoml.info%2Fscheduledsampling)中提出的。`ScheduledEmbeddingTrainingSampler`可以随机从目标或实际输出挑选，你可以在训练中逐渐调整概率。

### 双向 RNN

在每个时间步，常规循环层在产生输出前，只会查看过去和当下的输入。换句话说，循环层是遵循因果关系的，它不能查看未来。这样的 RNN 在预测时间序列时是合理的，但对于许多 NLP 任务，比如机器翻译，在编码给定词时，最好看看后面的词是什么。比如，对于这几个短语`the Queen of the United Kingdom`、`the queen of hearts`、`the queen bee`：要正确编码`queen`，需要向前看。要实现的话，可以对于相同的输入运行两个循环层，一个从左往右读，一个从右往左读。然后将每个时间步的输出结合，通常是连起来。这被称为双向循环层（见图 16-5）。

要在 Keras 中实现双向循环层，可以在`keras.layers.Bidirectional`层中包一个循环层。例如，下面的代码创建了一个双向`GRU`层：

```py
keras.layers.Bidirectional(keras.layers.GRU(10, return_sequences=True)) 
```

> 笔记：`Bidirectional`层会创建一个`GRU`层的复制（但方向相反），会运行两个层，并将输出连起来。因此`GRU`层有 10 个神经元，`Bidirectional`层在每个时间步会输出 20 个值。

![](img/498e45d619e08c8fc7acff449bfb27de.png)

图 16-5 双向循环层

### 集束搜索
