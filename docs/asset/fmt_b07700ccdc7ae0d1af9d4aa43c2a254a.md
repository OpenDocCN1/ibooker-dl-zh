
```py
import torch
from math import sqrt

query = key = value = inputs_embeds
dim_k = key.size(-1)
scores = torch.bmm(query, key.transpose(1,2)) / sqrt(dim_k)
scores.size()
```

```py
torch.Size([1, 5, 5])
```

这创建了一个每个批次样本的 5 × 5 注意力分数矩阵。我们将在后面看到，查询、键和值向量是通过将独立的权重矩阵 `W_{Q,K,V}` 应用于嵌入来生成的，但现在为了简单起见，我们将它们保持相等。在缩放的点积注意力中，点积被嵌入向量的大小缩放，以便在训练过程中不会得到太多的大数，这可能会导致我们接下来将应用的 softmax 饱和。

###### 注

`torch.bmm()` 函数执行*批量矩阵-矩阵乘积*，简化了注意力分数的计算，其中查询和键向量的形状为 `[batch_size, seq_len, hidden_dim]`。如果忽略批处理维度，我们可以通过简单地转置键张量的形状为 `[hidden_dim, seq_len]`，然后使用矩阵乘积来收集所有点积在 `[seq_len, seq_len]` 矩阵中。由于我们希望对批中的所有序列独立地执行此操作，我们使用 `torch.bmm()`，它接受两个矩阵批次，并将第一个批次中的每个矩阵与第二个批次中的相应矩阵相乘。

现在让我们应用 softmax：

```py
import torch.nn.functional as F

weights = F.softmax(scores, dim=-1)
weights.sum(dim=-1)
```

```py
tensor([[1., 1., 1., 1., 1.]], grad_fn=<SumBackward1>)
```

最后一步是将注意力权重乘以值：

```py
attn_outputs = torch.bmm(weights, value)
attn_outputs.shape
```

```py
torch.Size([1, 5, 768])
```

就是这样——我们已经完成了实现简化形式的自注意力的所有步骤！请注意，整个过程只是两次矩阵乘法和一个 softmax，因此你可以将“自注意力”看作是一种花哨的平均形式。

让我们将这些步骤封装成一个我们以后可以使用的函数：

```py
def scaled_dot_product_attention(query, key, value):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)
```

我们的注意机制使用相等的查询和键向量将为上下文中相同的单词分配非常大的分数，特别是对于当前单词本身：查询与自身的点积始终为 1。但实际上，一个单词的含义更多地受到上下文中的补充单词的影响，而不是相同的单词，例如，“flies”的含义更好地通过“time”和“arrow”的信息来定义，而不是通过另一个“flies”的提及。我们如何促进这种行为呢？

让模型通过使用三个不同的线性投影为一个标记的查询、键和值创建不同的向量集。

### 多头注意力

在我们简单的例子中，我们只是使用嵌入“原样”来计算注意力分数和权重，但这远非全部。实际上，自注意力层对每个嵌入应用三个独立的线性变换，以生成查询、键和值向量。这些变换将嵌入投影到不同的空间，并且每个投影都携带其自己的可学习参数集，这使得自注意力层能够关注序列的不同语义方面。

同时拥有*多个*线性投影集也被证明是有益的，每个代表一个所谓的*注意头*。结果的*多头注意力层*在图 3-5 中有所说明。但为什么我们需要多个注意头呢？原因是一个头的 softmax 倾向于主要关注相似性的某个方面。拥有多个头允许模型同时关注多个方面。例如，一个头可以关注主谓交互，而另一个可以找到附近的形容词。显然，我们不会手工将这些关系编码到模型中，它们完全是从数据中学习到的。如果你熟悉计算机视觉模型，你可能会看到它与卷积神经网络中的滤波器的相似之处，其中一个滤波器可以负责检测脸部，另一个可以在图像中找到车轮。

![多头注意力](img/nlpt_0305.png)

###### 图 3-5。多头注意力

让我们首先编写一个单个注意力头来实现这一层：

```py
class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state):
        attn_outputs = scaled_dot_product_attention(
            self.q(hidden_state), self.k(hidden_state), self.v(hidden_state))
        return attn_outputs
```

在这里，我们初始化了三个独立的线性层，它们对嵌入向量进行矩阵乘法，以产生形状为 `[batch_size, seq_len, head_dim]` 的张量，其中 `head_dim` 是我们投影到的维度的数量。虽然 `head_dim` 不一定要小于标记的嵌入维度（`embed_dim`），但实际上它被选择为 `embed_dim` 的倍数，以便每个头部的计算是恒定的。例如，BERT 有 12 个注意力头，因此每个头的维度是 768 / 12 = 64。

现在我们有了一个单独的注意力头，我们可以将每个头的输出连接起来，以实现完整的多头注意力层：

```py
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x
```

注意，注意力头的连接输出也通过最终的线性层，以产生适合下游前馈网络的形状为 `[batch_size, seq_len, hidden_dim]` 的输出张量。为了确认，让我们看看多头注意力层是否产生了我们输入的预期形状。我们在初始化 `MultiHeadAttention` 模块时传递了之前从预训练 BERT 模型加载的配置。这确保我们使用与 BERT 相同的设置：

```py
multihead_attn = MultiHeadAttention(config)
attn_output = multihead_attn(inputs_embeds)
attn_output.size()
```

```py
torch.Size([1, 5, 768])
```

它起作用了！在注意力这一部分结束时，让我们再次使用 BertViz 来可视化单词“flies”两种不同用法的注意力。在这里，我们可以使用 BertViz 的 `head_view()` 函数，通过计算预训练检查点的注意力，并指示句子边界的位置：

```py
from bertviz import head_view
from transformers import AutoModel

model = AutoModel.from_pretrained(model_ckpt, output_attentions=True)

sentence_a = "time flies like an arrow"
sentence_b = "fruit flies like a banana"

viz_inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt')
attention = model(**viz_inputs).attentions
sentence_b_start = (viz_inputs.token_type_ids == 0).sum(dim=1)
tokens = tokenizer.convert_ids_to_tokens(viz_inputs.input_ids[0])

head_view(attention, tokens, sentence_b_start, heads=[8])
```

![](img/nlpt_03in02.png)

这个可视化将注意力权重显示为连接正在更新嵌入的标记（左）与正在被关注的每个单词（右）的线条。线条的强度表示注意力权重的强度，深色线条表示接近 1 的值，淡色线条表示接近 0 的值。

在这个例子中，输入由两个句子组成，而 `[CLS]` 和 `[SEP]` 标记是 BERT 的分词器中的特殊标记，我们在第二章中遇到过。从可视化中我们可以看到的一件事是，注意力权重在属于同一句子的单词之间最强，这表明 BERT 可以知道它应该关注同一句子中的单词。然而，对于单词“flies”，我们可以看到 BERT 已经确定了第一个句子中“arrow”是重要的，第二个句子中是“fruit”和“banana”。这些注意力权重使模型能够区分“flies”作为动词或名词，取决于它出现的上下文！

现在我们已经涵盖了注意力，让我们来看看如何实现编码器层中缺失的位置逐层前馈网络。

## 前馈层

编码器和解码器中的前馈子层只是一个简单的两层全连接神经网络，但有一个变化：它不是将整个嵌入序列作为单个向量处理，而是*独立地*处理每个嵌入。因此，这一层通常被称为*位置逐层前馈层*。您可能还会看到它被称为具有大小为 1 的核的一维卷积，通常是由具有计算机视觉背景的人（例如，OpenAI GPT 代码库使用这种命名方式）。文献中的一个经验法则是，第一层的隐藏大小应为嵌入大小的四倍，最常用的是 GELU 激活函数。这是假设大部分容量和记忆发生的地方，也是在扩大模型规模时最常扩展的部分。我们可以将其实现为一个简单的 `nn.Module`，如下所示：

```py
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x
```

请注意，诸如 `nn.Linear` 之类的前馈层通常应用于形状为 `(batch_size, input_dim)` 的张量，其中它独立地作用于批处理维度的每个元素。这实际上对除了最后一个维度之外的任何维度都是真实的，因此当我们传递形状为 `(batch_size, seq_len, hidden_dim)` 的张量时，该层将独立地作用于批处理和序列中的所有令牌嵌入，这正是我们想要的。让我们通过传递注意力输出来测试一下：

```py
feed_forward = FeedForward(config)
ff_outputs = feed_forward(attn_outputs)
ff_outputs.size()
```

```py
torch.Size([1, 5, 768])
```

我们现在已经拥有了创建一个完整的 transformer 编码器层的所有要素！唯一剩下的决定是在哪里放置跳过连接和层归一化。让我们看看这如何影响模型架构。

## 添加层归一化

正如前面提到的，Transformer 架构使用了*层归一化*和*跳过连接*。前者将批处理中的每个输入归一化为零均值和单位方差。跳过连接将一个张量传递到模型的下一层而不进行处理，并将其添加到处理过的张量中。在将层归一化放置在 transformer 的编码器或解码器层中时，文献中采用了两种主要选择：

后层归一化

这是 Transformer 论文中使用的安排；它将层归一化放置在跳过连接之间。这种安排很难从头开始训练，因为梯度可能会发散。因此，你经常会看到一个称为“学习率预热”的概念，在训练过程中学习率会逐渐从一个小值增加到某个最大值。

层前归一化

这是文献中最常见的安排；它将层归一化放置在跳过连接的范围内。这在训练过程中往往更加稳定，通常不需要任何学习率预热。

两种安排的区别在图 3-6 中有所说明。

![Transformer 层归一化](img/nlpt_0306.png)

###### 图 3-6. Transformer 编码器层中层归一化的不同安排

我们将使用第二种安排，因此我们可以简单地将我们的构建模块连接在一起：

```py
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        # Apply layer normalization and then copy input into query, key, value
        hidden_state = self.layer_norm_1(x)
        # Apply attention with a skip connection
        x = x + self.attention(hidden_state)
        # Apply feed-forward layer with a skip connection
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x
```

现在让我们用我们的输入嵌入来测试一下：

```py
encoder_layer = TransformerEncoderLayer(config)
inputs_embeds.shape, encoder_layer(inputs_embeds).size()
```

```py
(torch.Size([1, 5, 768]), torch.Size([1, 5, 768]))
```

我们现在已经从头开始实现了我们的第一个 transformer 编码器层！然而，我们设置编码器层的方式有一个问题：它们对令牌的位置是完全不变的。由于多头注意力层实际上是一种花哨的加权和，令牌位置的信息会丢失。⁴

幸运的是，有一个简单的技巧可以使用位置嵌入来纳入位置信息。让我们看看。

## 位置嵌入

位置嵌入基于一个简单但非常有效的思想：用一个与位置相关的值模式增强令牌嵌入，这些值排列在一个向量中。如果该模式对每个位置都是特征性的，那么每个堆栈中的注意力头和前馈层可以学习将位置信息纳入它们的转换中。
