
# 第三章：Transformer 剖析

在第二章中，我们看到了微调和评估 Transformer 所需的内容。现在让我们来看看它们在内部是如何工作的。在本章中，我们将探索 Transformer 模型的主要构建模块以及如何使用 PyTorch 实现它们。我们还将提供如何在 TensorFlow 中进行相同操作的指导。我们首先将专注于构建注意力机制，然后添加必要的部分来使 Transformer 编码器工作。我们还将简要介绍编码器和解码器模块之间的架构差异。通过本章结束时，您将能够自己实现一个简单的 Transformer 模型！

虽然通常不需要对 Transformer 架构有深入的技术理解来使用![nlpt_pin01](img/nlpt_pin01.png)Transformer 并为您的用例微调模型，但对于理解和应对 Transformer 的局限性并在新领域中使用它们可能会有所帮助。

本章还介绍了 Transformer 的分类法，以帮助您了解近年来出现的各种模型。在深入代码之前，让我们先来看一下开启 Transformer 革命的原始架构的概述。

## Transformer 架构

正如我们在第一章中所看到的，原始 Transformer 基于“编码器-解码器”架构，这种架构被广泛用于诸如机器翻译之类的任务，其中一个词序列被翻译成另一种语言。这种架构由两个组件组成：

- **编码器**：将输入的令牌序列转换为嵌入向量序列，通常称为“隐藏状态”或“上下文”
- **解码器**：使用编码器的隐藏状态迭代地生成一个令牌序列的输出，每次生成一个令牌

如图 3-1 所示，编码器和解码器本身由几个构建模块组成。

![transformer-encoder-decoder](img/nlpt_0301.png)

###### 图 3-1。Transformer 的编码器-解码器架构，编码器显示在图的上半部分，解码器显示在图的下半部分

我们很快将详细查看每个组件，但我们已经可以在图 3-1 中看到一些特征，这些特征表征了 Transformer 架构：

+ 输入文本被标记化，并使用我们在第二章中遇到的技术转换为“令牌嵌入”。由于注意力机制不知道令牌的相对位置，我们需要一种方法将一些关于令牌位置的信息注入到输入中，以建模文本的顺序性质。因此，令牌嵌入与包含每个令牌的位置信息的“位置嵌入”相结合。
+ 编码器由一堆“编码器层”或“块”组成，类似于在计算机视觉中堆叠卷积层。解码器也是如此，它有自己的一堆“解码器层”。
+ 编码器的输出被馈送到每个解码器层，然后解码器生成一个最可能的下一个令牌序列的预测。这一步的输出然后被反馈到解码器中以生成下一个令牌，依此类推，直到达到特殊的序列结束（EOS）令牌。在图 3-1 的示例中，想象一下解码器已经预测了“Die”和“Zeit”。现在它将这两个作为输入以及所有编码器的输出来预测下一个令牌，“fliegt”。在下一步中，解码器获得“fliegt”作为额外的输入。我们重复这个过程，直到解码器预测出 EOS 令牌或者达到最大长度。

Transformer 架构最初是为机器翻译等序列到序列任务设计的，但编码器和解码器块很快被改编为独立的模型。虽然有数百种不同的 transformer 模型，但它们大多属于以下三种类型之一：

- **仅编码器**：这些模型将文本输入序列转换为丰富的数值表示，非常适合文本分类或命名实体识别等任务。BERT 及其变种，如 RoBERTa 和 DistilBERT，属于这类架构。在这种架构中，对于给定标记的表示取决于左侧（标记之前）和右侧（标记之后）的上下文。这经常被称为*双向注意力*。
- **仅解码器**：给定一个文本提示，比如“谢谢午餐，我吃了一个...”，这些模型将通过迭代预测最有可能的下一个词来自动完成序列。GPT 模型系列属于这一类。在这种架构中，对于给定标记的表示仅取决于左侧上下文。这经常被称为*因果*或*自回归注意力*。
- **编码器-解码器**：这些模型用于对一个文本序列到另一个文本序列的复杂映射进行建模；它们适用于机器翻译和摘要任务。除了我们已经看到的 Transformer 架构，它结合了编码器和解码器，BART 和 T5 模型也属于这一类。

###### 注意

实际上，解码器-仅模型与仅编码器模型的应用区别有些模糊。例如，像 GPT 系列中的解码器-仅模型可以被用于翻译等任务，这些任务传统上被认为是序列到序列任务。同样，像 BERT 这样的仅编码器模型可以应用于通常与编码器-解码器或仅解码器模型相关的摘要任务。¹

现在您已经对 Transformer 架构有了高层次的理解，让我们更仔细地看看编码器的内部工作。

## 编码器

正如我们之前看到的，transformer 的编码器由许多相邻堆叠的编码器层组成。如图 3-2 所示，每个编码器层接收一系列嵌入，并通过以下子层进行处理：

+ 多头自注意力层
+ 应用于每个输入嵌入的全连接前馈层

每个编码器层的输出嵌入与输入的大小相同，我们很快就会看到编码器堆栈的主要作用是“更新”输入嵌入，以产生编码一些上下文信息的表示。例如，如果“keynote”或“phone”这样的词靠近“apple”，那么“apple”这个词将被更新为更“公司化”而不是更“水果化”。

![encoder-zoom](img/nlpt_0302.png)

###### 图 3-2\. 放大到编码器层

每个子层也使用跳跃连接和层归一化，这是训练深度神经网络的标准技巧。但要真正理解 transformer 的工作原理，我们必须深入了解。让我们从最重要的构建模块开始：自注意力层。

### 自注意力

正如我们在第一章中讨论的那样，注意力是一种机制，它允许神经网络为序列中的每个元素分配不同数量的权重或“注意力”。对于文本序列，元素是*标记嵌入*，就像我们在第二章中遇到的那样，其中每个标记都被映射到某个固定维度的向量。例如，在 BERT 中，每个标记表示为一个 768 维向量。“自注意力”中的“自”指的是这些权重是针对同一集合中的所有隐藏状态计算的，例如，编码器的所有隐藏状态。相比之下，与循环模型相关的注意力机制涉及计算每个编码器隐藏状态对于给定解码时间步的解码器隐藏状态的相关性。

自注意力的主要思想是，我们可以使用整个序列来计算每个嵌入的*加权平均*，而不是为每个标记使用固定的嵌入。另一种表述方法是，给定标记嵌入序列<math alttext="x 1 comma ellipsis comma x Subscript n Baseline"><mrow><msub><mi>x</mi> <mn>1</mn></msub> <mo>,</mo> <mo>...</mo> <mo>,</mo> <msub><mi>x</mi> <mi>n</mi></msub></mrow></math>，自注意力会产生一个新嵌入序列<math alttext="x prime 1 comma ellipsis comma x prime Subscript n"><mrow><msubsup><mi>x</mi> <mn>1</mn> <mo>'</mo></msubsup> <mo>,</mo> <mo>...</mo> <mo>,</mo> <msubsup><mi>x</mi> <mi>n</mi> <mo>'</mo></msubsup></mrow></math>，其中每个<math alttext="x prime Subscript i"><msubsup><mi>x</mi> <mi>i</mi> <mo>'</mo></msubsup></math>都是所有<math alttext="x Subscript j"><msub><mi>x</mi> <mi>j</mi></msub></math>的线性组合：

<math alttext="x prime Subscript i Baseline equals sigma-summation Underscript j equals 1 Overscript n Endscripts w Subscript j i Baseline x Subscript j" display="block"><mrow><msubsup><mi>x</mi> <mi>i</mi> <mo>'</mo></msubsup> <mo>=</mo> <munderover><mo>∑</mo> <mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow> <mi>n</mi></munderover> <msub><mi>w</mi> <mrow><mi>j</mi><mi>i</mi></mrow></msub> <msub><mi>x</mi> <mi>j</mi></msub></mrow></math>

系数<math alttext="w Subscript j i"><msub><mi>w</mi> <mrow><mi>j</mi><mi>i</mi></mrow></msub></math>被称为*注意力权重*，并且被归一化，使得<math alttext="sigma-summation Underscript j Endscripts w Subscript j i Baseline equals 1"><mrow><msub><mo>∑</mo> <mi>j</mi></msub> <msub><mi>w</mi> <mrow><mi>j</mi><mi>i</mi></mrow></msub> <mo>=</mo> <mn>1</mn></mrow></math>。要了解为什么对标记嵌入进行平均可能是一个好主意，请考虑当你看到“flies”这个词时会想到什么。你可能会想到讨厌的昆虫，但如果你得到更多的上下文，比如“time flies like an arrow”，那么你会意识到“flies”是指动词。同样，我们可以通过以不同的比例组合所有标记嵌入，也许通过给“time”和“arrow”的标记嵌入分配更大的权重<math alttext="w Subscript j i"><msub><mi>w</mi> <mrow><mi>j</mi><mi>i</mi></mrow></msub></math>，来创建一个包含这个上下文的“flies”的表示。以这种方式生成的嵌入称为*上下文化嵌入*，并且早于像 ELMo 这样的语言模型中的 transformers 的发明。2 显示了该过程的图表，我们在其中说明了如何通过自注意力，根据上下文，可以生成“flies”的两种不同表示。

![上下文化嵌入](img/nlpt_0303.png)

###### 图 3-3。显示了自注意力如何将原始标记嵌入（上部）更新为上下文化嵌入（下部），以创建包含整个序列信息的表示。

现在让我们看看如何计算注意力权重。

#### 缩放点积注意力

有几种实现自注意力层的方法，但最常见的是来自 Transformer 架构的“缩放点积注意力”。实现这种机制需要四个主要步骤：

1. 将每个标记嵌入投影到称为查询、键和值的三个向量中。
2. 计算注意力分数。我们使用相似性函数来确定查询和键向量之间的关系程度。正如其名称所示，缩放点积注意力的相似性函数是点积，通过嵌入的矩阵乘法进行高效计算。相似的查询和键将具有较大的点积，而那些没有共同之处的将几乎没有重叠。这一步的输出被称为注意力分数，对于具有 n 个输入标记的序列，有一个相应的 n×n 的注意力分数矩阵。
3. 计算注意力权重。点积通常会产生任意大的数，这可能会使训练过程不稳定。为了处理这个问题，首先将注意力分数乘以一个缩放因子来归一化它们的方差，然后通过 softmax 进行归一化，以确保所有列的值总和为 1。得到的 n×n 矩阵现在包含了所有的注意力权重，wji。
4. 更新标记嵌入。一旦计算出注意力权重，我们将它们乘以值向量 v1，直到 vn，以获得嵌入的更新表示 xi'。

我们可以使用一个称为“BertViz for Jupyter”的巧妙库来可视化注意力权重的计算。这个库提供了几个函数，可以用来可视化 Transformer 模型中注意力的不同方面。为了可视化注意力权重，我们可以使用`neuron_view`模块，它跟踪权重的计算过程，以展示查询和键向量是如何组合产生最终权重的。由于 BertViz 需要访问模型的注意力层，我们将使用 BertViz 的模型类来实例化我们的 BERT 检查点，然后使用`show()`函数来为特定的编码器层和注意力头生成交互式可视化。请注意，您需要点击左侧的“+”来激活注意力可视化：

```py
from transformers import AutoTokenizer
from bertviz.transformers_neuron_view import BertModel
from bertviz.neuron_view import show

model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = BertModel.from_pretrained(model_ckpt)
text = "time flies like an arrow"
show(model, "bert", tokenizer, text, display_mode="light", layer=0, head=8)
```

![](img/nlpt_03in01.png)

从可视化中，我们可以看到查询和键向量的值被表示为垂直条带，其中每个条带的强度对应于其大小。连接线的权重根据标记之间的注意力而加权，我们可以看到“flies”的查询向量与“arrow”的键向量有最强的重叠。

让我们通过实现计算缩放点积注意力的操作图来更详细地了解这个过程，如图 3-4 所示。

![缩放点积注意力中的操作](img/nlpt_0304.png)

###### 图 3-4. 缩放点积注意力中的操作

在本章中，我们将使用 PyTorch 来实现 Transformer 架构，但 TensorFlow 中的步骤是类似的。我们提供了两个框架中最重要函数的映射，详见表 3-1。

**表 3-1。本章中使用的 PyTorch 和 TensorFlow（Keras）类和方法**

| PyTorch | TensorFlow (Keras) | 创建/实现 |
| --- | --- | --- |
| `nn.Linear` | `keras.layers.Dense` | 一个密集的神经网络层 |
| `nn.Module` | `keras.layers.Layer` | 模型的构建模块 |
| `nn.Dropout` | `keras.layers.Dropout` | 一个 dropout 层 |
| `nn.LayerNorm` | `keras.layers.LayerNormalization` | 层归一化 |
| `nn.Embedding` | `keras.layers.Embedding` | 一个嵌入层 |
| `nn.GELU` | `keras.activations.gelu` | 高斯误差线性单元激活函数 |
| `nn.bmm` | `tf.matmul` | 批量矩阵乘法 |
| `model.forward` | `model.call` | 模型的前向传播 |

我们需要做的第一件事是对文本进行标记化，因此让我们使用我们的标记器提取输入 ID：

```py
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
inputs.input_ids
```

```py
tensor([[ 2051, 10029,  2066,  2019,  8612]])
```

正如我们在第二章中看到的，句子中的每个标记都被映射到标记器词汇表中的唯一 ID。为了保持简单，我们还通过设置`add_special_tokens=False`来排除了`[CLS]`和`[SEP]`标记。接下来，我们需要创建一些密集的嵌入。在这种情况下，“密集”意味着嵌入中的每个条目都包含非零值。相比之下，在第二章中看到的 one-hot 编码是“稀疏”的，因为除一个之外的所有条目都是零。在 PyTorch 中，我们可以通过使用`torch.nn.Embedding`层来实现这一点，该层充当每个输入 ID 的查找表：

```py
from torch import nn
from transformers import AutoConfig

config = AutoConfig.from_pretrained(model_ckpt)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
token_emb
```

```py
Embedding(30522, 768)
```

在这里，我们使用了`AutoConfig`类来加载与`bert-base-uncased`检查点相关联的*config.json*文件。在![nlpt_pin01](img/nlpt_pin01.png) Transformers 中，每个检查点都分配了一个配置文件，指定了诸如`vocab_size`和`hidden_size`之类的各种超参数，例如我们的示例中显示每个输入 ID 将映射到`nn.Embedding`中存储的 30,522 个嵌入向量之一，每个嵌入向量的大小为 768。`AutoConfig`类还存储其他元数据，例如标签名称，用于格式化模型的预测。

请注意，此时的标记嵌入与它们的上下文无关。这意味着在上一个示例中的“flies”等同义词（拼写相同但含义不同的单词）具有相同的表示。随后的注意力层的作用将是混合这些标记嵌入，以消除歧义并使用上下文的内容来形成每个标记的表示。

现在我们有了查找表，我们可以通过输入 ID 生成嵌入：

```py
inputs_embeds = token_emb(inputs.input_ids)
inputs_embeds.size()
```

```py
torch.Size([1, 5, 768])
```

这给我们提供了一个形状为`[batch_size, seq_len, hidden_dim]`的张量，就像我们在第二章中看到的一样。我们将推迟位置编码，因此下一步是创建查询、键和值向量，并使用点积作为相似性函数来计算注意力分数。
