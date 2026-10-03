
有三种类型的 Transformer：仅编码器 Transformer、仅解码器 Transformer 和编码器-解码器 Transformer。我们在这章和下一章中使用的是编码器-解码器 Transformer，但你将在本书的后面有机会亲自探索仅解码器 Transformer。

仅包含编码器的 Transformer 由图 9.4 左侧所示的 N 个相同的编码器层组成，并且能够将序列转换为抽象的连续向量表示。例如，BERT 就是一个仅包含 12 个编码器层的编码器-only Transformer。仅包含编码器的 Transformer 可以用于文本分类，例如。如果两个句子具有相似的向量表示，我们可以将这两个句子分类到同一类别。另一方面，如果两个序列具有非常不同的向量表示，我们可以将它们放入不同的类别中。

仅解码器 Transformer 也由 N 个相同的层组成，每个层都是图 9.4 右侧所示的解码器层。例如，ChatGPT 是一个包含许多解码器层的仅解码器 Transformer。仅解码器 Transformer 可以根据提示生成文本，例如。它提取提示中单词的语义含义并预测最可能的下一个标记。然后它将标记添加到提示的末尾并重复此过程，直到文本达到一定长度。

我们之前讨论的机器翻译 Transformer 是编码器-解码器 Transformer 的一个例子。它们对于处理复杂任务（如文本到图像生成或语音识别）是必需的。编码器-解码器 Transformer 结合了编码器和解码器的优点。编码器在处理和理解输入数据方面效率高，而解码器在生成输出方面表现卓越。这种组合使模型能够有效地理解复杂的输入（如文本或语音）并生成复杂的输出（如图像或转录文本）。

## 9.2 构建编码器

我们将开发并训练一个针对机器语言翻译设计的编码器-解码器 Transformer。本项目中的编码来自 Chris Cui 的将中文翻译成英文的工作（[`mng.bz/9o1o`](https://mng.bz/9o1o)）和 Alexander Rush 的德语到英语翻译项目（[`mng.bz/j0mp`](https://mng.bz/j0mp)）。

本节讨论了如何在 Transformer 中构建编码器。具体来说，我们将深入探讨构建每个编码器层内各种子层的过程以及实现多头自注意力机制。

### 9.2.1 注意力机制

尽管存在不同的注意力机制，但我们将使用 SDPA，因为它被广泛使用且有效。SDPA 注意力机制使用查询、键和值来计算序列中元素之间的关系。它分配分数以显示一个元素与序列中所有元素（包括该元素本身）的关系。

与使用一组查询、键和值向量不同，Transformer 模型使用了一个称为多头注意力的概念。我们的 256 维查询、键和值向量被分成 8 个头，每个头都有一组查询、键和值向量，维度为 32（因为 256/8 = 32）。每个头关注输入的不同部分或方面，使模型能够捕捉更广泛的信息，并对输入数据形成更详细和上下文化的理解。例如，多头注意力允许模型捕捉双关语笑话中“bank”一词的多种含义，“为什么这条河这么富饶？因为它有两个河岸。”

为了实现这一点，我们在本地模块 `ch09util` 中定义了一个 `attention()` 函数。从本书的 GitHub 仓库（[`github.com/markhliu/DGAI`](https://github.com/markhliu/DGAI)）下载文件 `ch09util.py` 并将其存储在您计算机的 `/utils/` 目录中。`attention()` 函数的定义如下所示。

列表 9.1 基于查询、键和值计算注意力

```py
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, 
              key.transpose(-2, -1)) / math.sqrt(d_k)       ①
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)        ②
    p_attn = nn.functional.softmax(scores, dim=-1)          ③
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn              ④
```

① 缩放后的注意力分数是查询和键的点积，乘以 `d_k` 的平方根。

② 如果有掩码，则隐藏序列中的未来元素

③ 计算注意力权重

④ 返回注意力和注意力权重

`attention()` 函数接受查询、键和值作为输入，并计算注意力和注意力权重，正如我们在本章前面讨论的那样。缩放后的注意力分数是查询和键的点积，乘以键的维度的平方根 `d_k`。我们应用 softmax 函数到缩放后的注意力分数以获得注意力权重。最后，注意力是注意力权重和值的点积。

让我们使用我们的运行示例来展示多头注意力的工作原理（见图 9.7）。"你好吗？"的嵌入是一个大小为 (1, 6, 256) 的张量，正如我们在上一节中解释的（在我们将位置编码添加到词嵌入之后）。请注意，1 表示批处理中有一个句子，句子中有六个标记而不是四个，因为我们向序列的开始和结束添加了 BOS 和 EOS。这个嵌入通过三个线性层传递，以获得查询 `Q`、键 `K` 和值 `V`，每个的大小相同 (1, 6, 256)。这些被分成八个头，现在每个头的大小为 (1, 6, 256/8 = 32)。根据前面定义的注意力函数，将这些集合中的每个都应用注意力函数，产生八个注意力输出，每个的大小也是 (1, 6, 32)。然后，我们将八个注意力输出连接成一个单一的注意力，结果是大小为 (1, 6, 32 × 8 = 256) 的张量。最后，这个组合注意力通过另一个大小为 256 × 256 的线性层，导致 `MultiHeadAttention()` 类的输出。这个输出保持了原始输入的维度，即 (1, 6, 256)。

![图片](img/CH09_F07_Liu.png)

图 9.7 多头注意力的一个示例。此图以短语“你好吗？”的多头自注意力计算为例。我们首先将嵌入通过三个神经网络传递，以获得查询 `Q`、键 `K` 和值 `V`，每个的大小为 (1, 6, 256)。我们将它们分成八个头，每个头都有一组 `Q`、`K` 和 `V`，大小为 (1, 6, 32)。我们在每个头中计算注意力。然后，将八个头的注意力向量合并回一个单一的注意力向量，大小为 (1, 6, 256)。

这在以下本地模块的代码列表中实现。

列表 9.2 计算多头注意力

```py
from copy import deepcopy
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([deepcopy(
            nn.Linear(d_model, d_model)) for i in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)  
        query, key, value = [l(x).view(nbatches, -1, self.h,
           self.d_k).transpose(1, 2)    
         for l, x in zip(self.linears, (query, key, value))]    ①
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout) ②
        x = x.transpose(1, 2).contiguous().view(
            nbatches, -1, self.h * self.d_k)                    ③
        output = self.linears[-1](x)                            ④
        return output
```

① 将输入通过三个线性层传递以获得 `Q`、`K`、`V`，并将它们分成多头

② 计算每个头的注意力和注意力权重

③ 将多头注意力向量连接成一个单一的注意力向量

④ 将输出通过一个线性层

每个编码器层和解码器层也包含一个前馈子层，这是一个两层全连接神经网络，其目的是增强模型在训练数据集中捕获和学习的复杂特征的能力。此外，神经网络独立处理每个嵌入。它不将嵌入序列视为单个向量。因此，我们通常称其为位置感知的前馈网络（或一维卷积网络）。为此，我们在本地模块中定义了一个 `PositionwiseFeedForward()` 类，如下所示：

```py
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        h1 = self.w_1(x)
        h2 = self.dropout(h1)
        return self.w_2(h2) 
```

`PositionwiseFeedForward()` 类使用两个关键参数定义：`d_ff`，表示前馈层的维度，和 `d_model`，表示模型的维度大小。通常，`d_ff` 被选择为 `d_model` 的四倍。在我们的例子中，`d_model` 是 256，因此我们将 `d_ff` 设置为 256 * 4 = 1024。与模型大小相比扩大隐藏层的方法是 Transformer 架构中的标准做法。这增强了网络在训练数据集中捕获和学习的复杂特征的能力。

### 9.2.2 创建编码器

要创建一个编码器层，我们首先定义以下 `EncoderLayer()` 类和 `SublayerConnection()` 类。

列表 9.3 定义编码器层的类

```py
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([deepcopy(
        SublayerConnection(size, dropout)) for i in range(2)])
        self.size = size  
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))     ①
        output = self.sublayer[1](x, self.feed_forward)     ②
        return output
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
        output = x + self.dropout(sublayer(self.norm(x)))   ③
        return output  
```

① 每个编码器层中的第一个子层是一个多头自注意力网络。

② 每个编码器层中的第二个子层是一个前馈网络。

③ 每个子层都经过残差连接和层归一化。

每个编码器层由两个不同的子层组成：一个是 `MultiHeadAttention()` 类中概述的多头自注意力层，另一个是 `PositionwiseFeedForward()` 类中指定的简单、位置感知的全连接前馈网络。此外，这两个子层都包含层归一化和残差连接。正如第六章所述，残差连接涉及通过一系列变换（在此上下文中是注意力或前馈层）传递输入，然后将输入添加到这些变换的输出中。残差连接的方法被用来对抗梯度消失问题，这是非常深的网络中常见的挑战。在 Transformers 中，残差连接的另一个好处是提供一个通道，将位置编码（仅在第一层之前计算）传递到后续层。

层归一化与我们在第四章中实现的批归一化有些类似。它将层中的观察值标准化为零均值和单位标准差。为了在本地模块中实现这一点，我们定义了 `LayerNorm()` 类，该类执行层归一化如下：

```py
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True) 
        std = x.std(-1, keepdim=True)
        x_zscore = (x - mean) / torch.sqrt(std ** 2 + self.eps)
        output = self.a_2*x_zscore+self.b_2
        return output 
```

在前面的 `LayerNorm()` 类中的 `mean` 和 `std` 值是每个层中输入的均值和标准差。`LayerNorm()` 类中的 `a_2` 和 `b_2` 层将 `x_zscore` 扩展回输入 `x` 的形状。

现在我们可以通过堆叠六个编码器层来创建一个编码器。为此，我们在本地模块中定义了 `Encoder()` 类：

```py
from copy import deepcopy
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList(
            [deepcopy(layer) for i in range(N)])
        self.norm = LayerNorm(layer.size)
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
            output = self.norm(x)
        return output
```

在这里，`Encoder()` 类使用两个参数定义：`layer`，它是一个编码器层，如列表 9.3 中定义的 `EncoderLayer()` 类中的编码器层，以及 `N`，编码器中编码器层的数量。`Encoder()` 类接受输入 `x`（例如，一批英语短语）和掩码（以屏蔽序列填充，如我在第十章中解释的）以生成输出（捕获英语短语意义的向量表示）。

这样，你就创建了一个编码器。接下来，你将学习如何创建一个解码器。

## 9.3 构建编码器-解码器 Transformer

现在你已经了解了如何在 Transformer 中创建编码器，让我们继续到解码器。在本节中，你将首先学习如何创建解码器层。然后，你将堆叠 N = 6 个相同的解码器层来形成一个解码器。

然后，我们创建了一个包含五个组件的编码器-解码器 Transformer：`encoder`、`decoder`、`src_embed`、`tgt_embed` 和 `generator`，我将在本节中解释这些。

### 9.3.1 创建解码器层

每个解码器层由三个子层组成：（1）一个多头自注意力层，（2）第一个子层的输出与编码器输出的交叉注意力，以及（3）一个前馈网络。这三个子层都包含一个层归一化和残差连接，类似于我们在编码器层中做的。此外，解码器堆叠的多头自注意力子层被掩码，以防止位置关注后续位置。掩码迫使模型使用序列中的先前元素来预测后续元素。我将在稍后解释掩码多头自注意力是如何工作的。为了实现这一点，我们在本地模块中定义了 `DecoderLayer()` 类。

列表 9.4 创建解码器层
