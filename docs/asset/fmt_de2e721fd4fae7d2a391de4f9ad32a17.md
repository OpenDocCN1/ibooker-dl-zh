
图 4.12 表明，通过跳过一层或多层，捷径连接为梯度流动到网络中创建了一条替代的、更短的路径，这是通过将某一层的输出添加到后续层的输出中实现的。这就是为什么这些连接也被称为跳过连接。它们在训练过程中反向传递时保持梯度流动起着至关重要的作用。

在以下列表中，我们实现了图 4.12 中的神经网络，以查看我们如何在`forward`方法中添加快捷连接。

##### 列表 4.5：用于说明快捷连接的神经网络

```py
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([       #1
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), 
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), 
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), 
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), 
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), 
                          GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)         #2
            if self.use_shortcut and x.shape == layer_output.shape:    #3
                x = x + layer_output
            else:
                x = layer_output
        return x
```

#1 实现五个层

#2 计算当前层的输出

#3 检查是否可以应用快捷连接

代码实现了一个包含五个层的深度神经网络，每个层由一个`Linear`层和一个`GELU`激活函数组成。在正向传播过程中，我们迭代地将输入通过层传递，如果将`self.use_shortcut`属性设置为`True`，则可选地添加快捷连接。

让我们使用此代码初始化一个没有快捷连接的神经网络。每个层将被初始化，以便它接受一个具有三个输入值的示例，并返回三个输出值。最后一层返回一个单一的输出值：

```py
layer_sizes = [3, 3, 3, 3, 3, 1]  
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123)                            #1
model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=False
)
```

#1 指定初始权重的随机种子以实现可重复性

接下来，我们实现一个函数，该函数在模型的反向传播过程中计算梯度：

```py
def print_gradients(model, x):
    output = model(x)             #1
    target = torch.tensor([[0.]])

    loss = nn.MSELoss()
    loss = loss(output, target)    #2

    loss.backward()          #3

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")
```

#1 正向传播

#2 根据目标和输出之间的接近程度计算损失

#3 反向传播以计算梯度

此代码指定了一个损失函数，该函数计算模型输出与用户指定的目标（这里，为了简单起见，是值 0）的接近程度。然后，在调用`loss.backward()`时，PyTorch 计算模型中每个层的损失梯度。我们可以通过`model.named_parameters()`遍历权重参数。假设我们有一个 3 × 3 的权重参数矩阵，对于给定的层。在这种情况下，该层将具有 3 × 3 的梯度值，我们打印这些 3 × 3 梯度值的平均值绝对梯度，以获得每个层的单个梯度值，以便更容易地比较层之间的梯度。

简而言之，`.backward()`方法是 PyTorch 中的一个方便的方法，它计算损失梯度，这在模型训练期间是必需的，而不需要我们自己实现梯度计算的数学，从而使得与深度神经网络的工作变得更加容易。

注意：如果您不熟悉梯度以及神经网络训练的概念，我建议阅读附录 A 中的 A.4 和 A.7 节。

让我们现在使用`print_gradients`函数并将其应用于没有跳过连接的模型：

```py
print_gradients(model_without_shortcut, sample_input)
```

输出是

```py
layers.0.0.weight has gradient mean of 0.00020173587836325169
layers.1.0.weight has gradient mean of 0.0001201116101583466
layers.2.0.weight has gradient mean of 0.0007152041653171182
layers.3.0.weight has gradient mean of 0.001398873864673078
layers.4.0.weight has gradient mean of 0.005049646366387606
```

`print_gradients`函数的输出显示，随着我们从最后一层`(layers.4)`向第一层`(layers.0)`前进，梯度值会变小，这是一种称为*梯度消失问题*的现象。

让我们现在实例化一个具有跳过连接的模型，看看它与没有快捷连接的模型相比如何：

```py
torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=True
)
print_gradients(model_with_shortcut, sample_input)
```

输出是

```py
layers.0.0.weight has gradient mean of 0.22169792652130127
layers.1.0.weight has gradient mean of 0.20694105327129364
layers.2.0.weight has gradient mean of 0.32896995544433594
layers.3.0.weight has gradient mean of 0.2665732502937317
layers.4.0.weight has gradient mean of 1.3258541822433472
```

最后一个层`(layers.4`)的梯度值仍然比其他层要大。然而，随着我们向第一层`(layers.0`)前进，梯度值趋于稳定，并不会缩小到一个极小的值。

总之，快捷连接对于克服深度神经网络中梯度消失问题的限制至关重要。快捷连接是像 LLM 这样的大型模型的核心构建块，它们将有助于通过确保在下一章中训练 GPT 模型时层间梯度流的连续性来促进更有效的训练。

接下来，我们将把之前涵盖的所有概念（层归一化、GELU 激活、前馈模块和快捷连接）连接到 transformer 块中，这是我们需要编码 GPT 架构的最后一个构建块。

## 4.5 在 transformer 块中连接注意力和线性层

现在，让我们实现*transformer 块*，这是 GPT 和其他 LLM 架构的基本构建块。这个块在 1.24 亿参数的 GPT-2 架构中重复了 12 次，结合了我们之前涵盖的几个概念：多头注意力、层归一化、dropout、前馈层和 GELU 激活。稍后，我们将把这个 transformer 块连接到 GPT 架构的其余部分。

![figure](img/4-13.png)

##### 图 4.13 transformer 块的示意图。输入标记已嵌入到 768 维向量中。每一行对应一个标记的向量表示。transformer 块输出的向量与输入具有相同的维度，然后可以被输入到 LLM 的后续层中。

图 4.13 展示了一个结合了多个组件的 transformer 块，包括第三章中提到的掩码多头注意力模块（见章节 3）以及我们之前实现过的`FeedForward`模块（见第 4.3 节）。当 transformer 块处理一个输入序列时，序列中的每个元素（例如，一个单词或子词标记）由一个固定大小的向量表示（在这种情况下，768 维）。transformer 块内的操作，包括多头注意力和前馈层，被设计成以保留其维度的这种方式转换这些向量。

理念在于多头注意力块中的自注意力机制识别并分析输入序列中元素之间的关系。相比之下，前馈网络在每一个位置单独修改数据。这种组合不仅使对输入的理解和处理更加细腻，而且增强了模型处理复杂数据模式的整体能力。

我们可以在代码中创建`TransformerBlock`。

##### 列表 4.6 GPT 的 transformer 块组件

```py
from chapter03 import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
 #1
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut      #2

        shortcut = x         #3
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut      #4
        return x
```

#1 注意力块的快捷连接

#2 将原始输入添加回

#3 前馈块的快捷连接

#4 添加原始输入回

给定的代码在 PyTorch 中定义了一个 `TransformerBlock` 类，该类包含一个多头注意力机制 (`MultiHeadAttention`) 和一个前馈网络 (`FeedForward`)，这两个组件都是基于提供的配置字典 (`cfg`) 配置的，例如 `GPT_CONFIG_124M`。

在这两个组件之前应用层归一化 (`LayerNorm`)，并在它们之后应用 dropout 以正则化模型并防止过拟合。这被称为 *Pre-LayerNorm*。较老的架构，如原始的 Transformer 模型，在自注意力机制和前馈网络之后应用层归一化，称为 *Post-LayerNorm*，这通常会导致更差的训练动态。

该类还实现了前向传递，其中每个组件之后都跟着一个快捷连接，该连接将块的输入添加到其输出中。这个关键特性有助于在训练期间通过网络流动梯度，并提高深度模型的学习（参见第 4.4 节）。

使用我们之前定义的 `GPT_CONFIG_124M` 字典，让我们实例化一个 Transformer 模块并给它一些样本数据：

```py
torch.manual_seed(123)
x = torch.rand(2, 4, 768)                   #1
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
```

#1 创建形状为 [batch_size, num_tokens, emb_dim] 的样本输入

输出结果为

```py
Input shape: torch.Size([2, 4, 768])
Output shape: torch.Size([2, 4, 768])
```

如我们所见，Transformer 模块在其输出中保持了输入维度，这表明 Transformer 架构在处理数据序列时不会改变它们的形状。

在 Transformer 模块架构中保持形状不是偶然的，而是其设计的一个关键方面。这种设计使其能够有效地应用于广泛的序列到序列任务，其中每个输出向量直接对应于一个输入向量，保持一对一的关系。然而，输出是一个上下文向量，它封装了整个输入序列的信息（参见第三章）。这意味着尽管序列的物理维度（长度和特征大小）在通过 Transformer 模块时保持不变，但每个输出向量的内容被重新编码以整合整个输入序列的上下文信息。

在实现了 Transformer 模块之后，我们现在拥有了实现 GPT 架构所需的所有构建块。如图 4.14 所示，Transformer 模块结合了层归一化、前馈网络、GELU 激活和快捷连接。正如我们最终将看到的，这个 Transformer 模块将成为 GPT 架构的主要组成部分。

![figure](img/4-14.png)

##### 图 4.14 构建 GPT 架构所需的构建块。黑色勾选表示我们已完成的块。

## 4.6 编写 GPT 模型

我们以对称为`DummyGPTModel`的 GPT 架构的大图概述开始本章。在这个`DummyGPTModel`代码实现中，我们展示了 GPT 模型的输入和输出，但其构建块仍然是一个黑盒，使用`DummyTransformerBlock`和`DummyLayerNorm`类作为占位符。

现在我们将`DummyTransformerBlock`和`DummyLayerNorm`占位符替换为我们之前编写的真实`TransformerBlock`和`LayerNorm`类，以组装 GPT-2 原始 1.24 亿参数版本的完整工作版本。在第五章中，我们将预训练 GPT-2 模型，在第六章中，我们将从 OpenAI 加载预训练的权重。

在我们用代码组装 GPT-2 模型之前，让我们看看它的整体结构，如图 4.15 所示，其中包括我们迄今为止所涵盖的所有概念。正如我们所见，变换块在整个 GPT 模型架构中被重复多次。在 1.24 亿参数的 GPT-2 模型中，它被重复 12 次，我们通过`GPT_CONFIG_124M`字典中的`n_layers`条目来指定。在具有 15.42 亿参数的最大 GPT-2 模型中，变换块被重复 48 次。

![figure](img/4-15.png)

##### 图 4.15 展示了 GPT 模型架构的概述，显示了数据通过 GPT 模型流动的过程。从底部开始，标记化文本首先被转换为标记嵌入，然后添加位置嵌入。这些组合信息形成一个张量，通过中心的一系列变换块（每个包含多头注意力和具有 dropout 和层归一化的前馈神经网络层）传递，这些变换块堆叠在一起并重复 12 次。

最终变换块输出的数据在到达线性输出层之前会经过最后的层归一化步骤。这一层将变换器的输出映射到一个高维空间（在这种情况下，50,257 维，对应于模型的词汇量大小），以预测序列中的下一个标记。

现在我们来编写图 4.15 中的架构。

##### 列表 4.7 GPT 模型架构实现
