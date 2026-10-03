
`embedded_positions <- self$position_embeddings(positions)`

`embedded_tokens + embedded_positions`

},

`compute_mask = function(inputs, mask = NULL)` {

`inputs != 0`

},

`get_config = function()` {

`config <- super$get_config()`

`for(name in c("output_dim", "sequence_length", "input_dim"))`

`config[[name]] <- self[[name]]`

`config`

}

)

**位置嵌入的一个缺点是需要提前知道序列长度。**

**为标记索引准备一个 `layer_embedding()`。**

**为标记位置准备另一个。**

`tf$shape(inputs)[-1]` 切片出形状的最后一个元素，即嵌入维度的大小。（`tf$shape()` 返回张量的形状。）

`tf$range()` 类似于 R 中的 `seq()`，生成整数序列：`[0, 1, 2, …, limit - 1]`。

将两个嵌入向量相加。

像 `layer_embedding()`一样，这个层应该能够生成一个蒙版，这样我们就可以忽略输入中的填充 0。`compute_mask()`方法将由框架自动调用，并且蒙版将传播到下一层。

实现序列化以便我们可以保存模型。

你会像使用常规的 `layer_embedding()`一样使用这个 `layer_positional_embedding()`。让我们看看它的作用！

### 将所有内容整合在一起：一个文本分类 Transformer

只需将旧的 `layer_embedding()`替换为我们的位置感知版本，就可以开始考虑单词顺序。

列表 11.25 结合 Transformer 编码器和位置嵌入

`vocab_size <- 20000`

`sequence_length <- 600`

`embed_dim <- 256`

`num_heads <- 2`

`dense_dim <- 32`

`inputs <- layer_input(shape(NULL), dtype = "int64")`

`outputs <- inputs %>%`

`layer_positional_embedding(sequence_length, vocab_size, embed_dim) %>%`

`layer_transformer_encoder(embed_dim, dense_dim, num_heads) %>%`

`layer_global_average_pooling_1d() %>%`

`layer_dropout(0.5) %>%`

`layer_dense(1, activation = "sigmoid")`

`model <-`

`keras_model(inputs, outputs) %>%`

`compile(optimizer = "rmsprop",`

`loss = "binary_crossentropy",`

`metrics = "accuracy")`

`model`

![图片](img/f0380-01.jpg)

`callbacks <- list(`

`callback_model_checkpoint("full_transformer_encoder.keras",`

`save_best_only = TRUE)`

`)`

`model %>% fit(`

`int_train_ds,`

`validation_data = int_val_ds,`

`epochs = 20,`

`callbacks = callbacks`

`)`

`model <- load_model_tf(`

`"full_transformer_encoder.keras",`

`custom_objects = list(layer_transformer_encoder,`

`layer_positional_embedding))`

`cat(sprintf(`

`"测试准确率：%.3f\n", evaluate(model, int_test_ds)["accuracy"]))`

测试准确率：0.886

看这里！我们达到了 88.6%的测试准确率——这种改进证明了单词顺序信息对文本分类的价值。这是我们迄今为止最好的序列模型！然而，它仍然比词袋模型差一档。

### 11.4.4 何时使用序列模型而不是词袋模型

你可能会听说词袋模型方法已过时，而基于 Transformer 的序列模型是前进的道路，无论你看的是什么任务或数据集。这绝对不是这种情况：在许多情况下，在词袋模型之上放置一小叠稠密层仍然是一个完全有效和相关的方法。事实上，在本章节中我们在 IMDB 数据集上尝试的各种技术中，迄今为止表现最佳的是词袋模型！那么，何时您应该在另一种方法上更倾向于另一种方法？

2017 年，我和我的团队对多种不同类型的文本数据集上各种文本分类技术的性能进行了系统分析，我们发现了一个惊人而令人惊讶的经验法则，用于决定是选择词袋模型还是序列模型（[`mng.bz/AOzK`](http://mng.bz/AOzK)）——一种黄金常数。事实证明，当面对一个新的文本分类任务时，你应该密切关注训练数据中样本数量与每个样本平均字数之间的比率（见图 11.11）。如果这个比率很小——小于 1,500——那么二元模型的表现会更好（而且作为额外奖励，它的训练和迭代速度也会更快）。如果这个比率高于 1,500，则应选择序列模型。换句话说，当有大量训练数据可用且每个样本相对较短时，序列模型的效果最佳。

![图像](img/f0381-01.jpg)

**图 11.11 选择文本分类模型的一个简单启发式方法：训练样本数与每个样本平均字数之间的比率**

所以，如果你要分类的是 1,000 字长的文档，而你有 100,000 个这样的文档（比例为 100），你应该选择一个二元模型。如果你要分类的是平均长度为 40 个字的推文，而你有 50,000 条这样的推文（比例为 1,250），你也应该选择一个二元模型。但如果你的数据集大小增加到 500,000 条推文（比例为 12,500），那就选择一个 Transformer 编码器。那 IMDB 电影评论分类任务呢？我们有 20,000 个训练样本，平均字数为 233，所以我们的经验法则指向一个二元模型，这证实了我们在实践中的发现。

这在直觉上是有道理的：序列模型的输入代表了一个更丰富、更复杂的空间，因此需要更多的数据来映射出这个空间；与此同时，一组简单的术语是一个如此简单的空间，以至于你可以只用几百或几千个样本来训练顶部的逻辑回归。此外，样本越短，模型就越不能丢弃其中包含的任何信息——特别是，单词顺序变得更加重要，丢弃它可能会产生歧义。句子“这部电影太棒了”和“这部电影是一颗炸弹”有非常接近的单字表示，这可能会让词袋模型感到困惑，但序列模型可以告诉哪一个是消极的，哪一个是积极的。对于更长的样本，单词统计将变得更可靠，而从单词直方图中就能更明显地看出主题或情感。

现在，请记住，这个启发式规则是专门为文本分类而开发的。它不一定适用于其他自然语言处理任务——例如，对于机器翻译来说，相比于循环神经网络，Transformer 在处理非常长的序列时表现得特别出色。我们的启发式规则也只是一个经验法则，而不是科学定律，所以请期望它大部分时间都有效，但不一定总是有效。

## 11.5 超越文本分类：序列到序列学习

你现在拥有了处理大多数自然语言处理任务所需的所有工具。然而，你只看到这些工具在单一问题上的应用：文本分类。这是一个极其流行的用例，但自然语言处理远不止于此。在这一部分，你将通过学习*序列到序列模型*来深化你的专业知识。

序列到序列模型接受一个序列作为输入（通常是一个句子或段落），并将其转换成另一个序列。这是许多最成功的自然语言处理应用程序的核心任务之一：

+   *机器翻译*—将源语言的段落转换成目标语言的等效段落。

+   *文本摘要*—将长篇文档转换成保留最重要信息的较短版本。

+   *问答*—将输入的问题转换成答案。

+   *聊天机器人*—将对话提示转换成对该提示的回复，或将对话历史转换成对话中的下一个回复。

+   *文本生成*—将文本提示转换成完成提示的段落。

+   等等。

序列到序列模型的一般模板描述在图 11.12 中。在训练过程中：

+   *编码器*模型将源序列转换为中间表示。

+   *解码器*通过查看之前的标记（1 到 i - 1）和编码后的源序列来训练，以预测目标序列中的下一个标记 i。

![图片](img/f0383-01.jpg)

**图 11.12 序列到序列学习：源序列经过编码器处理，然后发送到解码器。解码器查看目标序列至今，并预测偏移一个步骤的目标序列。在推断过程中，我们逐个目标标记地生成并将其送回解码器。**

在推断中，我们无法访问目标序列——我们试图从头开始预测它。我们将不得不逐个标记地生成它：

1. 我们从编码器中获得编码后的源序列。

2. 解码器首先查看编码后的源序列以及一个初始的“种子”标记（例如字符串“[start]”），并用它来预测序列中的第一个真实标记。

3. 到目前为止预测的序列被送回解码器，解码器生成下一个标记，依此类推，直到生成一个停止标记（如字符串“[end]”）。

到目前为止，你学到的所有东西都可以重新用于构建这种新型模型。让我们深入了解。

### 11.5.1 一个机器翻译示例

我们将在一个机器翻译任务上演示序列到序列建模。机器翻译正是 Transformer 开发的初衷！我们将从循环序列模型开始，并将跟进完整的 Transformer 架构。我们将使用[`www.manythings.org/anki/.`](http://www.manythings.org/anki/)上提供的英语到西班牙语翻译数据集。让我们下载它：

`download.file(`

`"http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip",`

`destfile = "spa-eng.zip")`

`zip::unzip("spa-eng.zip")`

文本文件每行包含一个示例：一个英语句子，后跟一个制表符，然后是相应的西班牙句子。让我们使用 `readr::read_tsv()`，因为我们有制表符分隔的值：

`text_file <- "spa-eng/spa.txt"`

`text_pairs <- text_file %>%`

`readr::read_tsv(col_names = c("english", "spanish"),`

`col_types = c("cc")) %>%`

`within(spanish %<>% paste("[start]", ., "[end]"))`

**使用 `read_tsv()`读取文件（制表符分隔的值）。**

**每行包含一个英语短语及其西班牙语翻译，用制表符分隔。**

**两字符列**

**我们在西班牙语句子前加上“[start]”，并在后面加上“[end]”，以匹配图 11.12 中的模板。**

我们的 `text_pairs` 看起来是这样的：

`str(text_pairs[sample(nrow(text_pairs), 1), ])`

`tibble [1 × 2] (S3: tbl_df/tbl/data.frame)`

`$ english: chr "I’m staying in Italy."`

`$ spanish: chr "[start] Me estoy quedando en Italia. [end]"`

让我们对它们进行洗牌并将它们拆分成通常的训练、验证和测试集：

`num_test_samples <- num_val_samples <-`

`round(0.15 * nrow(text_pairs))`

`num_train_samples <- nrow(text_pairs) - num_val_samples - num_test_samples`

`pair_group <- sample(c(`

`rep("train", num_train_samples),`

`rep("test", num_test_samples),`

`rep("val", num_val_samples)`

`))`

`train_pairs <- text_pairs[pair_group == "train", ]`

`test_pairs <- text_pairs[pair_group == "test", ]`

`val_pairs <- text_pairs[pair_group == "val", ]`

接下来，让我们准备两个单独的 `TextVectorization` 层：一个用于英语，一个用于西班牙语。我们需要定制字符串的预处理方式：

+   我们需要保留我们插入的“[start]”和“[end]”标记。默认情况下，字符[和]将被剥离，但我们希望保留它们，以便我们可以区分单词“start”和起始标记“[start]”。

+   标点符号在不同语言之间是不同的！在西班牙语 `Text-Vectorization` 层中，如果我们要剥离标点字符，我们还需要剥离字符¿。

请注意，对于非玩具翻译模型，我们将把标点字符视为单独的标记，而不是剥离它们，因为我们希望能够生成正确标点的句子。在我们的情况下，为了简单起见，我们将摆脱所有标点符号。

我们为西班牙语 `TextVectorization` 层准备了一个自定义字符串标准化函数：它保留了 [ 和 ]，但剥离了 ¿、¡ 和 [:punct:] 类中的所有其他字符。（[:punct:] 类的双重否定会互相抵消，就好像根本没有否定一样。然而，外部否定正则表达式分组让我们能够明确排除 [:punct:] 正则表达式类中的 [ 和 ]。我们使用 | 添加了其他不在 [:punct:] 字符类中的特殊字符，比如 ¡ 和 ¿。）

第 11.26 节 将英语和西班牙语文本对转为向量

`punctuation_regex <- "[^[:^punct:][\\]]|[¡¿]"`

`library(tensorflow)`

`custom_standardization <- function(input_string) {`

`input_string %>%`

`tf$strings$lower() %>%`

`tf$strings$regex_replace(punctuation_regex, "")`

`}`

`input_string <- as_tensor("[start] ¡corre! [end]")`

`custom_standardization(input_string)`

`tf.Tensor(b’[start] corre [end]’, shape=(), dtype=string)`

**基本上，就是 [[:punct:]]，除了省略了 "[" 和 "]"，添加了 "¿" 和 "¡"。**

**注意：这次我们使用张量操作。这允许函数被追踪到 TensorFlow 图中。**

**保留了 [start] 和 [end] 的 []，并去除了 ¡ 和 !。**

> **警告** TensorFlow 正则表达式与 R 正则引擎有细微差异。如果您需要高级正则表达式，请查阅源文档：[`github.com/google/re2/wiki/Syntax`](https://www.github.com/google/re2/wiki/Syntax)。

`vocab_size <- 15000`

`sequence_length <- 20`

`source_vectorization <- layer_text_vectorization(`

`max_tokens = vocab_size,`

`output_mode = "int",`

`output_sequence_length = sequence_length`

`)`

`target_vectorization <- layer_text_vectorization(`

`max_tokens = vocab_size,`

`output_mode = "int", output_sequence_length = sequence_length + 1,`

`standardize = custom_standardization`

`)`

`adapt(source_vectorization, train_pairs$english)`

`adapt(target_vectorization, train_pairs$spanish)`

**为了简单起见，我们将只考虑每种语言中的前 15,000 个单词，并将句子限制在 20 个词以内。**

**英语层**

**西班牙语层**

**生成西班牙句子，多了一个额外的标记，因为在训练过程中我们需要将句子向前偏移一步。**

**学习每种语言的词汇。**

最后，我们可以将我们的数据转为 TF Dataset 流水线。我们希望它返回一个对，(inputs, target)，其中 inputs 是一个带有两个条目的命名列表，英语句子（编码器输入）和西班牙句子（解码器输入），target 是西班牙句子向前偏移一步。

第 11.27 节 为翻译任务准备数据集

`format_pair <- function(pair) {`

`eng <- source_vectorization(pair$english)`

`spa <- target_vectorization(pair$spanish)`

`inputs <- list(english = eng,`

`spanish = spa[NA:-2])`

`targets <- spa[2:NA]`

`list(inputs, targets)`

`}`

`batch_size <- 64`

`library(tfdatasets)`

`make_dataset <- function(pairs) {`

`tensor_slices_dataset(pairs) %>%`

`dataset_map(format_pair, num_parallel_calls = 4) %>%`

`dataset_cache() %>%`
