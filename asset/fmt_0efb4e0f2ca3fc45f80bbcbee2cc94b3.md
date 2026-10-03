
# GPTModel 类

```py
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        #1
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```

#1 设备设置将允许我们在 CPU 或 GPU 上训练模型，具体取决于输入数据所在的设备。

多亏了`TransformerBlock`类，`GPTModel`类相对较小且紧凑。

这个`GPTModel`类的`__init__`构造函数使用通过 Python 字典`cfg`传入的配置初始化标记和位置嵌入层。这些嵌入层负责将输入标记索引转换为密集向量并添加位置信息（见第二章）。

接下来，`__init__`方法创建了一个等于`cfg`中指定层数的`TransformerBlock`模块的顺序堆叠。在变换器块之后，应用了一个`LayerNorm`层，标准化变换器块的输出以稳定学习过程。最后，定义了一个没有偏置的线性输出头，它将变换器的输出投影到分词器的词汇空间，为词汇表中的每个标记生成 logits。

前向方法接收一个输入标记索引的批次，计算它们的嵌入，应用位置嵌入，将序列通过变换器块，对最终输出进行归一化，然后计算 logits，表示下一个标记的非归一化概率。我们将在下一节将这些 logits 转换为标记和文本输出。

现在，让我们使用传递给`cfg`参数的`GPT_CONFIG_124M`字典初始化 124 百万参数的 GPT 模型，并用我们之前创建的批文本输入进行喂养：

```py
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)
```

这段代码打印了输入批次的内容，然后是输出张量：

```py
Input batch:
 tensor([[6109,  3626,  6100,   345],      #1
         [6109,  1110,  6622,   257]])     #2

Output shape: torch.Size([2, 4, 50257])
tensor([[[ 0.3613,  0.4222, -0.0711,  ...,  0.3483,  0.4661, -0.2838],
         [-0.1792, -0.5660, -0.9485,  ...,  0.0477,  0.5181, -0.3168],
         [ 0.7120,  0.0332,  0.1085,  ...,  0.1018, -0.4327, -0.2553],
         [-1.0076,  0.3418, -0.1190,  ...,  0.7195,  0.4023,  0.0532]],

        [[-0.2564,  0.0900,  0.0335,  ...,  0.2659,  0.4454, -0.6806],
         [ 0.1230,  0.3653, -0.2074,  ...,  0.7705,  0.2710,  0.2246],
         [ 1.0558,  1.0318, -0.2800,  ...,  0.6936,  0.3205, -0.3178],
         [-0.1565,  0.3926,  0.3288,  ...,  1.2630, -0.1858,  0.0388]]],
       grad_fn=<UnsafeViewBackward0>)
```

#1 文本 1 的标记 ID

#2 文本 2 的标记 ID

如我们所见，输出张量的形状为`[2, 4, 50257]`，因为我们输入了两个每个包含四个标记的文本。最后一个维度，`50257`，对应于分词器的词汇表大小。稍后，我们将看到如何将这些 50,257 维输出向量中的每一个转换回标记。

在我们继续编写将模型输出转换为文本的函数之前，让我们花更多的时间来分析模型架构本身的大小。使用`numel()`方法，即“元素数量”，我们可以收集模型参数张量中的总参数数量：

```py
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
```

结果是

```py
Total number of parameters: 163,009,536
```

现在，一个好奇的读者可能会注意到一个差异。之前，我们提到初始化了一个 124 百万参数的 GPT 模型，那么为什么实际的参数数量是 163 百万呢？

原因是一个称为*权重绑定*的概念，它在原始 GPT-2 架构中使用。这意味着原始 GPT-2 架构在其输出层中重新使用了标记嵌入层的权重。为了更好地理解，让我们看一下我们之前通过`GPTModel`初始化在`model`上的标记嵌入层和线性输出层的形状：

```py
print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)
```

如我们从打印输出中可以看到，这两个层的权重张量具有相同的形状：

```py
Token embedding layer shape: torch.Size([50257, 768])
Output layer shape: torch.Size([50257, 768])
```

由于分词器词汇表中有 50,257 个行数，标记嵌入和输出层非常大。让我们根据权重绑定从总 GPT-2 模型计数中减去输出层参数计数：

```py
total_params_gpt2 = (
    total_params - sum(p.numel()
    for p in model.out_head.parameters())
)
print(f"Number of trainable parameters "
      f"considering weight tying: {total_params_gpt2:,}"
)
```

输出是

```py
Number of trainable parameters considering weight tying: 124,412,160
```

如我们所见，该模型现在只有 124 百万参数大，与原始的 GPT-2 模型大小相匹配。

权重绑定可以减少模型的总体内存占用和计算复杂度。然而，根据我的经验，使用独立的标记嵌入层和输出层可以获得更好的训练和模型性能；因此，我们在`GPTModel`实现中使用了独立的层。对于现代大型语言模型（LLMs）也是如此。然而，我们将在第六章中重新审视并实现权重绑定概念，那时我们将从 OpenAI 加载预训练的权重。

## 练习 4.1 前馈和注意力模块中的参数数量

计算并比较包含在前馈模块中的参数数量和包含在多头注意力模块中的参数数量。

最后，让我们计算我们的`GPTModel`对象中 1.63 亿个参数的内存需求：

```py
total_size_bytes = total_params * 4       #1
total_size_mb = total_size_bytes / (1024 * 1024)     #2
print(f"Total size of the model: {total_size_mb:.2f} MB")
```

#1 计算总大小（假设为 float32，每个参数 4 字节）

#2 转换为兆字节

结果是

```py
Total size of the model: 621.83 MB
```

总之，通过计算我们的`GPTModel`对象中 1.63 亿个参数的内存需求，并假设每个参数是一个 32 位的浮点数，占用 4 字节，我们发现模型的总大小为 621.83 MB，这说明了即使是相对较小的 LLMs 也需要相对较大的存储容量。

现在我们已经实现了`GPTModel`架构，并看到它输出形状为`[batch_size, num_tokens, vocab_size]`的数值张量，让我们编写代码将这些输出张量转换为文本。

## 练习 4.2 初始化更大的 GPT 模型

我们初始化了一个参数数量为 1.24 亿的 GPT 模型，被称为“GPT-2 small”。除了更新配置文件外，不进行任何代码修改，使用`GPTModel`类实现 GPT-2 medium（使用 1,024 维嵌入，24 个 transformer 块，16 个多头注意力头），GPT-2 large（1,280 维嵌入，36 个 transformer 块，20 个多头注意力头），以及 GPT-2 XL（1,600 维嵌入，48 个 transformer 块，25 个多头注意力头）。作为额外奖励，计算每个 GPT 模型中的参数总数。

# 4.7 生成文本

我们现在将实现将 GPT 模型的张量输出转换回文本的代码。在我们开始之前，让我们简要回顾一下像 LLM 这样的生成模型是如何逐个单词（或标记）生成文本的。

![figure](img/4-16.png)

## 图 4.16 LLM 逐个标记生成文本的逐步过程

从初始输入上下文（“你好，我是”）开始，模型在每次迭代中预测后续标记，并将其附加到下一次预测的输入上下文中。如图所示，第一次迭代添加了“a”，第二次添加了“model”，第三次添加了“ready”，逐步构建句子。

图 4.16 说明了 GPT 模型根据输入上下文（例如“你好，我是。”）生成文本的逐步过程。每次迭代，输入上下文都会增长，允许模型生成连贯且上下文适当的文本。到第六次迭代时，模型已经构建了一个完整的句子：“你好，我是一个准备帮助的模型。”我们已经看到，我们当前的`GPTModel`实现输出形状为`[batch_size, num_token, vocab_size]`的张量。现在的问题是：GPT 模型是如何从这些输出张量生成文本的？

GPT 模型从输出张量到生成文本的过程涉及几个步骤，如图 4.17 所示。这些步骤包括解码输出张量、根据概率分布选择标记，并将这些标记转换为可读文本。

![figure](img/4-17.png)

## 图 4.17 通过展示标记生成过程中的单个迭代，展示了 GPT 模型中文本生成的机制

这个过程首先将输入文本编码为标记 ID，然后将这些 ID 输入到 GPT 模型中。模型的输出随后被转换回文本，并附加到原始输入文本上。

图 4.17 中详细说明了下一标记生成过程，展示了 GPT 模型根据其输入生成下一个标记的单个步骤。在每一步中，模型输出一个矩阵，其中包含表示潜在下一个标记的向量。与下一个标记对应的向量被提取出来，并通过`softmax`函数转换为概率分布。在包含结果概率分数的向量中，找到最高值的索引，这对应于标记 ID。然后，这个标记 ID 被解码回文本，生成序列中的下一个标记。最后，这个标记被附加到之前的输入上，形成新的输入序列，用于后续迭代。这个逐步过程使模型能够顺序生成文本，从初始输入上下文构建连贯的短语和句子。

在实践中，我们重复这个过程许多次迭代，例如图 4.16 所示，直到达到用户指定的生成标记数。在代码中，我们可以将标记生成过程实现如下所示。

## 列表 4.8 GPT 模型生成文本的函数

```py
def generate_text_simple(model, idx,                 #1
                         max_new_tokens, context_size): 
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]    #2
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]                    #3
        probas = torch.softmax(logits, dim=-1)           #4
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)    #5
        idx = torch.cat((idx, idx_next), dim=1)     #6

    return idx
```

#1 `idx`是当前上下文中索引的 (batch, n_tokens) 数组。

#2 如果当前上下文超过支持的上下文大小，则裁剪当前上下文，例如，如果 LLM 只支持 5 个标记，而上下文大小为 10，则只使用最后 5 个标记作为上下文

#3 仅关注最后一个时间步，因此 (batch, n_token, vocab_size) 变为 (batch, vocab_size)

#4 `probas`的形状为 (batch, vocab_size)。

#5 `idx_next`的形状为 (batch, 1)。

#6 将采样索引添加到运行序列中，其中`idx`的形状为 (batch, n_tokens+1)

此代码演示了使用 PyTorch 实现语言模型生成循环的简单实现。它迭代指定数量的新 token 以生成，裁剪当前上下文以适应模型的最大上下文大小，计算预测，然后根据最高概率预测选择下一个 token。

要编写`generate_text_simple`函数，我们使用`softmax`函数将 logits 转换为概率分布，然后通过`torch.argmax`识别具有最高值的位。`softmax`函数是单调的，这意味着它在转换为输出时保留了输入的顺序。因此，在实践中，softmax 步骤是多余的，因为 softmax 输出张量中得分最高的位置与 logit 张量中的相同位置。换句话说，我们可以直接对 logits 张量应用`torch.argmax`函数并得到相同的结果。然而，我提供了转换的代码，以说明将 logits 转换为概率的完整过程，这可以增加额外的直观性，以便模型生成最可能的下一个 token，这被称为*贪婪解码*。

当我们在下一章实现 GPT 训练代码时，我们将使用额外的采样技术来修改 softmax 输出，使得模型不总是选择最可能的 token。这为生成的文本引入了变化性和创造性。

使用`generate_text_simple`函数逐个生成 token ID 并将其附加到上下文中的这个过程在图 4.18 中进一步说明。（每个迭代的 token ID 生成过程在图 4.17 中详细说明。）我们以迭代的方式生成 token ID。例如，在迭代 1 中，模型被提供了对应于“Hello, I am,”的 token，预测下一个 token（ID 为 257，即“a”），并将其附加到输入中。这个过程重复进行，直到模型在六次迭代后生成完整的句子“Hello, I am a model ready to help”。

![figure](img/4-18.png)

## 图 4.18 token 预测周期的六次迭代

其中模型以一系列初始 token ID 作为输入，预测下一个 token，并将此 token 附加到下一个迭代的输入序列中。（token ID 也被转换成相应的文本以更好地理解。）

现在，让我们尝试使用`"Hello, I am"`上下文作为模型输入的`generate_text_simple`函数。首先，我们将输入上下文编码为 token ID：

```py
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)    #1
print("encoded_tensor.shape:", encoded_tensor.shape)
```

#1 添加批处理维度
