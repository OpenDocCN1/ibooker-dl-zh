
# 神经机器翻译模型实现

## 数据整理函数

```py
def nmt_collate_fn(batch):
    src_texts = [pair['source_text'] for pair in batch]
    tgt_texts = [f"<s> {pair['target_text']} </s>" for pair in batch]
    src_encodings = nmt_tokenizer.encode_batch(src_texts)
    tgt_encodings = nmt_tokenizer.encode_batch(tgt_texts)
    src_token_ids = torch.tensor([enc.ids for enc in src_encodings])
    tgt_token_ids = torch.tensor([enc.ids for enc in tgt_encodings])
    src_mask = torch.tensor([enc.attention_mask for enc in src_encodings])
    tgt_mask = torch.tensor([enc.attention_mask for enc in tgt_encodings])
    inputs = NmtPair(src_token_ids, src_mask,
                     tgt_token_ids[:, :-1], tgt_mask[:, :-1])
    labels = tgt_token_ids[:, 1:]
    return inputs, labels

batch_size = 32
nmt_train_loader = DataLoader(nmt_train_set, batch_size=batch_size,
                              collate_fn=nmt_collate_fn, shuffle=True)
nmt_valid_loader = DataLoader(nmt_valid_set, batch_size=batch_size,
                              collate_fn=nmt_collate_fn)
nmt_test_loader = DataLoader(nmt_test_set, batch_size=batch_size,
                             collate_fn=nmt_collate_fn)
```

`nmt_collate_fn()`函数首先从给定的批次中提取所有英语和西班牙语文本。在这个过程中，它还在每个西班牙语文本的开头添加一个 SoS 标记，以及在结尾添加一个 EoS 标记。然后，它使用我们的 BPE 标记器对英语和西班牙语文本进行标记化。接下来，将输入序列和注意力掩码转换为张量，并包装在`NmtPair`中。重要的是，该函数从解码器输入中删除 EoS 标记，并从解码器目标中删除 SoS 标记。例如，输入可能包含“<s> Me gusta el fútbol”的标记 ID，而目标可能包含“Me gusta el fútbol </s>”。

## 模型定义

现在我们已经准备好构建我们的翻译模型。它就像图 14-5 一样，只是编码器和解码器共享相同的`nn.Embedding`层，编码器和解码器的`nn.GRU`模块各包含两层：

```py
class NmtModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, pad_id=0, hidden_dim=512,
                 n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.encoder = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers,
                              batch_first=True)
        self.decoder = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers,
                              batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, pair):
        src_embeddings = self.embed(pair.src_token_ids)
        tgt_embeddings = self.embed(pair.tgt_token_ids)
        src_lengths = pair.src_mask.sum(dim=1)
        src_packed = pack_padded_sequence(
            src_embeddings, lengths=src_lengths.cpu(),
            batch_first=True, enforce_sorted=False)
        _, hidden_states = self.encoder(src_packed)
        outputs, _ = self.decoder(tgt_embeddings, hidden_states)
        return self.output(outputs).permute(0, 2, 1)

torch.manual_seed(42)
vocab_size = nmt_tokenizer.get_vocab_size()
nmt_model = NmtModel(vocab_size).to(device)
```

这个模型中的几乎所有内容都应该看起来很熟悉：它与我们的先前模型非常相似。我们在构造函数中创建模块，然后`forward()`方法将输入序列（英语和西班牙语）嵌入，打包英语嵌入并通过编码器传递，然后它将西班牙语嵌入传递给解码器，同时传递编码器的最后一个隐藏状态（所有`nn.GRU`层）。最后，解码器的输出通过输出`nn.Linear`层传递，最终输出通过重新排列以确保类别维度（包含标记对数）是第二个维度，因为`nn.CrossEntropyLoss`和`Accuracy`度量标准期望如此，正如我们之前所看到的。

> **注意**
> 
> 在 NMT 中最常用的度量标准是*bilingual evaluation understudy*（BLEU）分数，它将模型生成的每个翻译与人类生成的几个良好翻译进行比较。它计算出现在任何目标翻译中的*n*-gram（*n*个单词的序列）的数量，并调整分数以考虑生成的*n*-gram 在目标翻译中的频率。它通过 TorchMetric 的`BLEUScore`类实现。

我们本可以打包西班牙语嵌入，但那样的话，解码器的输出将是打包的序列，我们得在将它们传递到输出层之前填充它们。我们避免了这种复杂性，因为我们只需配置损失函数，使其在目标为填充标记时忽略输出标记，如下所示：

```py
xentropy = nn.CrossEntropyLoss(ignore_index=0)  # ignore <pad> tokens
```

现在，您可以训练这个模型（例如，使用`Nadam`优化器，`lr = 0.001`进行 10 个 epoch），这将花费相当长的时间。实际上，当你考虑到模型一次学习两种语言的事实时，它并不长！

## 翻译辅助函数

在训练过程中，让我们编写一个小的辅助函数，使用我们的模型将一些英语文本翻译成西班牙语。它将首先使用英语文本调用编码器，并为解码器提供一个单个 SoS 标记。解码器将只为翻译中的第一个标记输出 logits。然后我们的函数将选择最可能的标记（即具有最高 logit 的标记），并将其添加到解码器输入中，然后再次调用模型以获取下一个标记。它将重复此过程，每次添加一个标记，直到模型输出一个 EoS 标记：

```py
def translate(model, src_text, max_length=20, pad_id=0, eos_id=3):
    tgt_text = ""
    token_ids = []
    for index in range(max_length):
        batch, _ = nmt_collate_fn([{"source_text": src_text,
                                    "target_text": tgt_text}])
        with torch.no_grad():
            Y_logits = model(batch.to(device))
            Y_token_ids = Y_logits.argmax(dim=1)  # find the best token IDs
            next_token_id = Y_token_ids[0, index]  # take the last token ID

        next_token = nmt_tokenizer.id_to_token(next_token_id)
        tgt_text += " " + next_token
        if next_token_id == eos_id:
            break
    return tgt_text
```

> **注意**
> 
> 这种实现方法是可行的，但它根本就没有优化。我们可以在英语文本上只运行一次编码器，也可以在每个时间步只运行一次解码器，而不是在每次迭代中在整个增长文本上运行它。

让我们尝试翻译一些文本！

```py
>>> nmt_model.eval()
>>> translate(nmt_model, "I like soccer.")
' Me gusta el fútbol . </s>'
```

哈哈，成功了！我们从头开始构建了一个模型，可以将英语翻译成西班牙语。

如果你玩我们的翻译模型，你会发现它在处理短文本时通常表现良好，但处理长句时却真的很困难。例如：

```py
>>> longer_text = "I like to play soccer with my friends."
>>> translate(nmt_model, longer_text)
' Me gusta jugar con mis amigos . </s>'
```

翻译说“我喜欢和我的朋友们一起玩”。哎呀，没有提到足球。那么我们如何改进这个模型呢？一种方法是通过增加训练集的大小，并在编码器和解码器中添加更多的`nn.GRU`层。你还可以使编码器双向（但不是解码器，否则它就不再是因果的，它会在每个时间步看到完整的翻译，而不是仅仅看到之前的标记）。另一种可以极大地提高翻译模型在推理时性能的技术是*带宽搜索*。

# 带宽搜索

要将英语文本翻译成西班牙语，我们需要多次调用我们的模型，每次生成一个单词。不幸的是，这意味着当模型犯了一个错误时，它会一直带着这个错误完成整个翻译，这可能会导致更多的错误，从而使翻译越来越差。例如，假设我们想要翻译“我喜欢足球”，模型正确地以“我”开始，但随后预测的是“gustan”（复数）而不是“gusta”（单数）。这个错误是可以理解的，因为“Me gustan”在很多情况下是翻译“I like”的正确方式。一旦模型犯了这个错误，它就会一直使用“gustan”。然后它合理地添加了“los”，这是“the”的复数形式。但由于模型从未在训练数据中看到“los fútbol”（足球是单数，不是复数），模型试图找到一些合理的东西来添加，并且根据上下文，它添加了“jugadores”，这意味着“球员”。所以“我喜欢足球”被翻译成了“I like the players”。一个错误引发了一系列错误。

我们如何给模型一个机会去纠正它之前犯的错误？最常用的解决方案之一是**光束搜索**：它跟踪一个包含 k 个最有希望的输出序列的短列表（比如，前三个），在每次解码步骤中，它尝试通过一个单词扩展每个序列，并只保留 k 个最可能的序列。参数 k 被称为**光束宽度**。

例如，假设你使用该模型，通过具有三个宽度光束的光束搜索来翻译句子“我喜欢足球”（参见图 14-7）。在第一个解码步骤中，模型将为翻译句子中每个可能的首个单词输出一个估计概率。假设前三个单词是“Me”（估计概率为 75%），“a”（3%）和“como”（1%）。这就是我们到目前为止的短列表。接下来，我们使用模型为每个句子找到下一个单词。对于第一个句子（“Me”），模型可能为单词“gustan”输出 36%的概率，为单词“gusta”输出 32%的概率，为单词“encanta”输出 16%，等等。请注意，这些实际上是**条件**概率，给定句子以“Me”开头。对于第二个句子（“a”），模型可能为单词“mi”输出 50%的条件概率，等等。假设词汇表有 10,000 个标记，我们将为每个句子得到 10,000 个概率。

接下来，我们计算我们考虑的 30,000 个双词句子的概率（3 × 10,000）。我们通过将每个单词的估计条件概率乘以完成该句子的估计概率来完成这项工作。例如，句子“Me”的估计概率为 75%，而单词“gustan”（在第一个单词是“Me”的情况下）的估计条件概率为 36%，因此句子“Me gustan”的估计概率为 75% × 36% = 27%。在计算所有 30,000 个双词句子的概率后，我们只保留前三个。在这个例子中，它们都以单词“Me”开头：“Me gustan”（27%），“Me gusta”（24%）和“Me encanta”（12%）。目前，“Me gustan”领先，但“Me gusta”尚未被淘汰。

![展示具有三个宽度光束的搜索过程图，显示以“Me”开头并经过不同阶段的西班牙语句子构造的概率](img/hmls_1407.png)

**图 14-7. 具有三个宽度光束的搜索**

然后我们重复同样的过程：我们使用模型预测这三个句子中的每一个句子中的下一个单词，并计算我们考虑的 30,000 个三词句子的概率。也许现在排名前三的是“Me gustan los”（10%）、“Me gusta el”（8%）和“Me gusta mucho”（2%）。在下一步中，我们可能会得到“Me gusta el fútbol”（6%）、“Me gusta mucho el”（1%）和“Me gusta el deporte”（0.2%）。请注意，“Me gustan”已被排除，正确的翻译现在就在眼前。我们通过更明智地使用模型，而没有进行任何额外的训练，就提高了编码器-解码器模型的表现。

本章的笔记本包含一个非常简单的`beam_search()`函数，如果你感兴趣的话，但通常你可能会想使用 Transformers 库中`GenerationMixin`类提供的实现。这就是 Transformers 库中的文本生成模型获得`generate()`方法的地方：它接受一个`num_beams`参数，你可以将其设置为所需的 beam 宽度，如果你想使用 beam 搜索。它还提供了一个`do_sample`参数，它将使用模型输出的概率分布随机采样下一个标记，就像我们之前在 char-RNN 模型中做的那样。还支持其他生成策略，并且可以组合使用（更多详情请见[*https://homl.info/hfgen*](https://homl.info/hfgen)）。

就这样，你可以为相当短的句子获得相当好的翻译。例如，以下翻译是正确的：

```py
>>> beam_search(nmt_model, longer_text, beam_width=3)
' Me gusta jugar al fútbol con mis amigos . </s>'
```

不幸的是，这个模型在翻译长句子方面仍然会很糟糕：

```py
>>> longest_text = "I like to play soccer with my friends at the beach."
>>> beam_search(nmt_model, longest_text, beam_width=3)
' Me gusta jugar con jugar con los jug adores de la playa . </s>'
```

这翻译成“我喜欢和海滩上的玩家一起玩”。这并不完全正确。再次强调，问题源于 RNNs 有限的短期记忆。*注意力机制*是解决这一问题的颠覆性创新。

# 注意力机制

考虑从单词“soccer”到其翻译“fútbol”的路径，如图 14-5 所示：它相当长！这意味着这个单词（以及所有其他单词）的代表需要在实际使用之前跨越许多步骤。我们能不能使这条路径更短？

这是 Dzmitry Bahdanau 等人于 2014 年发表的一篇里程碑式[论文](https://homl.info/attention)⁠^(20)的核心思想，其中作者介绍了一种技术，允许解码器在每个时间步集中关注由编码器编码的适当单词。例如，在解码器需要输出单词“fútbol”的时间步，它将集中关注单词“soccer”。这意味着从输入单词到其翻译的路径现在要短得多，因此 RNNs 的短期记忆限制的影响要小得多。注意力机制彻底改变了神经机器翻译（以及深度学习），特别是在长句子（例如，超过 30 个单词）方面，显著提高了当前的最佳水平。

图 14-8 显示了我们的编码器-解码器模型，并添加了注意力机制：

+   在左侧，你有编码器和解码器（我在这个图中将编码器设置为双向的，因为这通常是一个好主意）。

+   我们现在将所有编码器的输出发送到解码器，而不是像之前那样将编码器的最终隐藏状态以及每个时间步的先前目标单词发送到解码器（这仍然会做，但在图中没有显示）。

+   由于解码器不能一次处理所有这些编码器输出，因此它们需要被聚合：在每个时间步，解码器的记忆单元计算所有编码器输出的加权总和。这决定了解码器在这个步骤将关注哪些单词。
