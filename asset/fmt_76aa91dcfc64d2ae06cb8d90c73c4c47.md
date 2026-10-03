
模型的第二部分是`Transformer`解码器。就像`RNN`解码器一样，它读取目标序列中的令牌`1…N`并尝试预测令牌`N + 1`。至关重要的是，在执行此操作时，它使用神经注意力来识别编码源句子中与当前正在尝试预测的目标令牌最相关的令牌——也许与人类翻译者所做的事情类似。回想一下查询-键-值模型：在`Transformer`解码器中，目标序列充当用于更密切关注源序列不同部分的注意力“查询”的角色（源序列扮演了键和值的角色）。

### Transformer 解码器

图 11.14 展示了完整的序列到序列`Transformer`。看一下解码器的内部：你会认识到它看起来非常类似于`Transformer`编码器，只是在应用于目标序列的自注意力块和退出块的密集层之间插入了一个额外的注意力块。

![Image](img/f0393-01.jpg)

**图 11.14 `TransformerDecoder`类似于`TransformerEncoder`，不同之处在于它具有一个额外的注意力块，其中键和值是由`TransformerEncoder`编码的源序列。编码器和解码器共同形成一个端到端的`Transformer`。**

让我们来实现它。就像对于`TransformerEncoder`一样，我们将创建一个新的层类。在我们关注`call()`方法之前，这个方法是发生动作的地方，让我们先定义类构造函数，包含我们将需要的层。

**`TransformerDecoder` 列表 11.33**

```
layer_transformer_decoder <- new_layer_class(

classname = "TransformerDecoder",

initialize = function(embed_dim, dense_dim, num_heads, …) {

super$initialize(…)

self$embed_dim <- embed_dim

self$dense_dim <- dense_dim

self$num_heads <- num_heads

self$attention_1 <- layer_multi_head_attention(num_heads = num_heads,

key_dim = embed_dim)

self$attention_2 <- layer_multi_head_attention(num_heads = num_heads,

key_dim = embed_dim)

self$dense_proj <- keras_model_sequential() %>%

layer_dense(dense_dim, activation = "relu") %>%

layer_dense(embed_dim)

self$layernorm_1 <- layer_layer_normalization()

self$layernorm_2 <- layer_layer_normalization()

self$layernorm_3 <- layer_layer_normalization()

self$supports_masking <- TRUE➊

},

get_config = function() {

config <- super$get_config()

for (name in c("embed_dim", "num_heads", "dense_dim"))

config[[name]] <- self[[name]]

config

},
```

➊ **这个属性确保层将其输入掩码传播到其输出；在`Keras`中，掩码必须明确地选择。如果你将一个掩码传递给一个不实现`compute_mask()`并且不公开这个`supports_masking`属性的层，那就是一个错误。**

`call()`方法几乎是从图 11.14 的连通性图的直观渲染。但是还有一个额外的细节我们需要考虑：*因果填充*。因果填充对于成功训练序列到序列`Transformer`是绝对关键的。与`RNN`不同，`RNN`逐步查看其输入，因此在生成输出步骤`N`（这是目标序列中的标记`N`+1）时只能访问步骤`1…N`的信息，`TransformerDecoder`是无序的：它一次查看整个目标序列。如果允许它使用其整个输入，它将简单地学会将输入步骤`N`+1复制到输出的位置`N`。因此，该模型将实现完美的训练准确度，但当进行推断时，它将完全无用，因为超出`N`的输入步骤是不可用的。

解决方法很简单：我们将屏蔽成对注意力矩阵的上半部分，以防止模型关注未来的任何信息——只有目标序列中的标记`1…N`的信息应该在生成目标标记`N`+1时使用。为此，我们将在我们的`TransformerDecoder`中添加一个`get_causal_attention_mask(inputs)`方法，以检索我们可以传递给我们的`MultiHeadAttention`层的注意力屏蔽。

**列出 11.34 `TransformerDecoder`生成因果掩码的方法**

```
get_causal_attention_mask = function(inputs) {

c(batch_size, sequence_length, .) %<-%➊

tf$unstack(tf$shape(inputs))

x <- tf$range(sequence_length)➋

i <- x[, tf$newaxis]

j <- x[tf$newaxis, ]

mask <- tf$cast(i >= j, "int32")➌ ➍

tf$tile(mask[tf$newaxis, , ],

tf$stack(c(batch_size, 1L, 1L)))➎

},
```

➊ **第三个轴是`encoding_length`；我们在这里不使用它。**

➋ **整数序列`[0, 1, 2, … sequence_length-1]`**

➌ **在我们的`>=`操作中使用`Tensor`广播。将`dtype bool`转换为`int32`。**

➍ **掩码是一个形状为`(sequence_length, sequence_length)`的方阵，其中下三角有`1`，其他地方为`0`。例如，如果`sequence_length`是`4`，那么掩码是：**

```
tf.Tensor([[1 0 0 0]

[1 1 0 0]

[1 1 1 0]

[1 1 1 1]], shape=(4, 4), dtype=int32)
```

➎ **向掩码添加一个批量维度，然后沿着批量维度复制（`rep()`）`batch_size`次。返回的张量形状为`(batch_size, sequence_length, sequence_length)`。**

现在我们可以写下实现解码器前向传递的完整`call()`方法。

**11.35 列出了`TransformerDecoder`的前向传递**

```
call = function(inputs, encoder_outputs, mask = NULL) {

causal_mask <- self$get_causal_attention_mask(inputs)➊

if (is.null(mask))➋

mask <- causal_mask

else

mask %<>% { tf$minimum(tf$cast(.[, tf$newaxis, ], "int32"),

causal_mask) }➌

inputs %>%

{ self$attention_1(query = ., value = ., key = .,

attention_mask = causal_mask) + . } %>%➍

self$layernorm_1() %>%➎

{ self$attention_2(query = .,

value = encoder_outputs,➏

key = encoder_outputs,➏

attention_mask = mask) + . } %>%➐

self$layernorm_2() %>%➑

{ self$dense_proj(.) + . } %>%➒

self$layernorm_3()

})
```

➊ **检索因果掩码。**

➋ **在调用中提供的掩码是填充掩码（它描述目标序列中的填充位置）。**

➌ **将填充掩码与因果屏蔽组合。**

➍ **将因果屏蔽传递给第一个注意力层，该注意力层在目标序列上执行自我注意力。**

➎ **将带有残差的`attention_1()`输出传递给`layernorm_1()`。**

➏ **在调用中使用`encoder_output`作为`attention_2()`的`value`和`key`参数。**

➐ **将组合屏蔽传递给第二个注意力层，该层将源序列与目标序列相关联。**

➑ **将带有残差的`attention_2()`输出传递给`layernorm_2()`。**

➒ **将带有残差的`dense_proj()`输出加起来并传递到`layernorm_3()`。**

### 将所有内容组合到一起：面向机器翻译的 Transformer

端到端`Transformer`是我们将要训练的模型。它将源序列和目标序列映射到目标序列的下一个步骤。它直接组合了我们迄今为止构建的部分：`PositionalEmbedding`层，`TransformerEncoder`和`TransformerDecoder`。请注意，`TransformerEncoder`和`TransformerDecoder`都形状不变，因此您可以堆叠许多它们来创建更强大的编码器或解码器。在我们的示例中，我们将坚持每个部分单独一个实例。

**图 11.36 端到端 Transformer**

```
embed_dim <- 256

dense_dim <- 2048

num_heads <- 8

encoder_inputs <- layer_input(shape(NA), dtype = "int64", name = "english")

encoder_outputs <- encoder_inputs %>%

layer_positional_embedding(sequence_length, vocab_size, embed_dim) %>%

layer_transformer_encoder(embed_dim, dense_dim, num_heads)➊

transformer_decoder <-

layer_transformer_decoder(NULL, embed_dim, dense_dim, num_heads)➋

decoder_inputs <- layer_input(shape(NA), dtype = "int64", name = "spanish")

decoder_outputs <- decoder_inputs %>%

layer_positional_embedding(sequence_length, vocab_size, embed_dim) %>%

transformer_decoder(., encoder_outputs) %>%➌

layer_dropout(0.5) %>%

layer_dense(vocab_size, activation = "softmax")➍

transformer <- keras_model(list(encoder_inputs, decoder_inputs),

decoder_outputs)
```

➊ **编码源句子。**

➋ **对于第一个参数传递`NULL`，以便直接创建并返回一个层实例，而不是将其与任何内容组合。**

➌ **编码目标句子并将其与编码的源句子结合起来。**

➍ **针对每个输出位置预测一个单词。**

现在我们准备训练我们的模型，我们得到了`67％`的准确率，比基于`GRU`的模型高得多。

**图 11.37 训练序列到序列的 Transformer**

```
transformer %>%

compile(optimizer = "rmsprop",

loss = "sparse_categorical_crossentropy",

metrics = "accuracy")

transformer %>%

fit(train_ds, epochs = 30, validation_data = val_ds)
```

最后，让我们尝试使用我们的模型来翻译来自测试集的从未见过的英语句子。设置与我们用于序列到序列`RNN`模型相同；唯一改变的是我们将`seq2seq_rnn`替换为`transformer`，并且删除了我们配置的`target_vectorization()`层添加的额外标记。

**列表 11.38 使用我们的 Transformer 模型翻译新句子**

```
tf_decode_sequence <- tf_function(function(input_sentence) {

withr::local_options(tensorflow.extract.style = "python")

tokenized_input_sentence <- input_sentence %>%

as_tensor(shape = c(1, 1)) %>%

source_vectorization()

spa_vocab <- as_tensor(spa_vocab)

decoded_sentence <- as_tensor("[start]", shape = c(1, 1))

for (i in tf$range(as.integer(max_decoded_sentence_length))) {

tokenized_target_sentence <-

target_vectorization(decoded_sentence)[, NA:-1]➊

next_token_predictions <-➋

transformer(list(tokenized_input_sentence,

tokenized_target_sentence))

sampled_token_index <- tf$argmax(next_token_predictions[0, i, ])

sampled_token <- spa_vocab[sampled_token_index]➌

decoded_sentence <-

tf$strings$join(c(decoded_sentence, sampled_token),

separator = " ")

if (sampled_token == "[end]")➍

break

}

decoded_sentence

})

for (i in sample.int(nrow(test_pairs), 20)) {

c(input_sentence, correct_translation) %<-% test_pairs[i, ]

cat(input_sentence, "\n")

cat(input_sentence %>% as_tensor() %>%

tf_decode_sequence() %>% as.character(), "\n")

cat("-\n")

}
```

➊ **删除最后一个标记；“python”样式不包括切片结尾。**

➋ **随机抽取下一个标记。**

➌ **将下一个标记预测转换为字符串，并附加到生成的句子中。**

➍ **退出条件**

主观上，`Transformer`似乎比基于`GRU`的翻译模型表现要好得多。它仍然是一个玩具模型，但是是一个更好的玩具模型。

**列表 11.39 Transformer 翻译模型的一些示例结果**

```
这是我小的时候学会的一首歌。

[start] esta es una canción que aprendí cuando

![Image](img/common01.jpg) era chico [end]➊

-

她会弹钢琴。

[start] ella puede tocar piano [end]

-

我不是你认为的那个人。

[start] no soy la persona que tú creo que soy [end]

-

昨晚可能下了一点雨。

[start] puede que llueve un poco el pasado [end]
```

➊ **尽管源句子没有明确性别，但这个翻译假设了一个男性说话者。请记住，翻译模型经常会对其输入数据做出不合理的假设，这会导致算法偏见。在最糟糕的情况下，模型可能会产生与当前处理的数据无关的记忆信息。**

这就结束了关于自然语言处理的这一章节——你刚刚从最基础的知识到了一个完全成熟的`Transformer`，可以将英语翻译成西班牙语。教会机器理解语言是你可以添加到自己技能集合中的最新超能力。

## 总结

+ 有两种`NLP`模型：处理单词集或`N`-gram而不考虑其顺序的*词袋模型*，以及处理单词顺序的*序列模型*。词袋模型由稠密层构成，而序列模型可以是`RNN`、一维卷积网络或`Transformer`。

+ 在文本分类方面，训练数据中的样本数量与每个样本的平均单词数之间的比例可以帮助您确定是使用词袋模型还是序列模型。

+ *词嵌入*是语义关系建模为代表这些词的向量之间的距离关系的向量空间。

+ *序列到序列学习*是一种通用、强大的学习框架，可用于解决许多`NLP`问题，包括机器翻译。序列到序列模型由一个编码器（处理源序列）和一个解码器（通过查看编码器处理的源序列的过去标记来尝试预测目标序列中的未来标记）组成。

+ *神经注意力*是一种创建上下文感知词表示的方法。这是`Transformer`架构的基础。

+ *Transformer*架构由`TransformerEncoder`和`TransformerDecoder`组成，在序列到序列的任务上产生了出色的结果。前半部分，`TransformerEncoder`，也可以用于文本分类或任何单输入`NLP`任务。

1. ¹ Yoshua Bengio 等人，《神经概率语言模型》，《机器学习研究杂志》（2003）。
