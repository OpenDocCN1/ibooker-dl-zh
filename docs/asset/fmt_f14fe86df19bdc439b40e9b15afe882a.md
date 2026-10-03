
```py
import transformers

gpt2_tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
gpt2_encoding = gpt2_tokenizer(train_reviews[:3], truncation=True,
                               max_length=500)
```

注意我们像使用函数一样使用分词器对象。结果是类型为 `BatchEncoding` 的类似字典的对象。你可以使用 `"input_ids"` 键来获取标记 ID。它返回一个标记 ID 的 Python 列表列表。例如，让我们看看第一个编码评论的前 10 个标记 ID，并使用分词器来解码它们，使用其 `decode()` 方法：

```py
>>> gpt2_token_ids = gpt2_encoding["input_ids"][0][:10]
>>> gpt2_token_ids
[14247, 35030, 1690, 423, 257, 1688, 8046, 13, 484, 1690]
>>> gpt2_tokenizer.decode(gpt2_token_ids)
'stage adaptations often have a major fault. they often'
```

如果你更愿意使用预训练的 WordPiece 分词器，你可以重用任何使用 WordPiece 预训练的 LLM 的分词器，例如 BERT（另一个流行的 NLP 模型，代表 Transformer 的双向编码表示）。这个分词器有一个填充标记（与之前的分词器不同，因为 GPT-2 不需要它），所以当编码一批评论时，我们可以指定 `padding=True`：像往常一样，最短的文本将使用填充标记填充到最长的文本长度。这允许我们还可以指定 `return_tensors="pt"` 来获取 PyTorch 张量而不是 Python 列表列表的标记 ID：非常方便！所以让我们编码前三个 IMDb 评论：

```py
bert_tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
bert_encoding = bert_tokenizer(train_reviews[:3], padding=True,
                               truncation=True, max_length=500,
                               return_tensors="pt")
```

### 小贴士

名称 `"bert-base-uncased"` 指的是一个 *模型检查点*：这个特定的检查点是大小写不敏感的 BERT 模型，在英语文本上预训练。其他检查点也可用，例如 `"bert-large-cased"`，如果你需要一个更大且大小写敏感的 BERT 模型，或者 `"bert-base-multilingual-uncased"`，如果你需要一个在超过 100 种语言上预训练的无大小写模型。目前我们只是使用模型的分词器。

结果的标记 ID 和注意力掩码是很好的填充张量：

```py
>>> bert_encoding["input_ids"]
tensor([[ 101, 2754,17241, 2411, 2031, [...],  102, 0, 0, 0, [...], 0, 0, 0],
 [ 101, 1005, 1996, 8305, 1005, [...],  102, 0, 0, 0, [...], 0, 0, 0],
 [ 101, 7929, 1010, 2021, 2515, [...], 1012,  1019,  1013,  1019,  102]])
>>> bert_encoding["attention_mask"]
tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, [...], 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, [...], 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, [...], 1, 1, 1, 1, 1, 1, 1, 1, 1]])
```

注意，每个标记 ID 序列都以标记 101 (`[CLS]`) 开头，并以标记 102 (`[SEP]`) 结尾（忽略填充标记）。这些标记是 BERT 模型所需的（正如我们将在第十五章中看到的那样），但除非您的模型也需要它们，否则您可以通过在调用分词器时设置 `add_special_tokens=False` 来删除它们。

那么，预训练的单词 LM 分词器怎么样呢？嗯，许多模型都是使用 Unigram LM 训练的，例如 ALBERT、T5 或 XML-R 模型，仅举几个例子。例如：

```py
albert_tokenizer = transformers.AutoTokenizer.from_pretrained("albert-base-v2")
albert_encoding = albert_tokenizer(train_reviews[:3], padding=True, [...])
```

Transformers 库还提供了一个对象，可以将您的自己的分词器（来自 Tokenizers 库）包装起来，并给它与预训练的分词器（来自 Transformers 库）相同的 API。例如，让我们包装我们之前训练过的 BPE 分词器：

```py
hf_tokenizer = transformers.PreTrainedTokenizerFast(
    tokenizer_object=bpe_tokenizer)
hf_encodings = hf_tokenizer(train_reviews[:3], padding=True, [...])
```

这样，我们就拥有了所有需要的分词工具，所以让我们回到情感分析。

## 构建和训练情感分析模型

我们的 sentiment analysis 模型必须使用分词评论的批次进行训练。然而，我们创建的数据集并没有处理分词。一个选择是更新它们（例如，使用 `map()` 方法），但处理数据加载器中的分词同样简单。为此，我们可以通过 `collate_fn` 参数将一个函数传递给 `DataLoader` 构造函数：数据加载器将为每个批次调用此函数，传递一个数据集样本列表。我们的函数将接受这个批次，对评论进行分词，如果需要则截断和填充它们，并返回一个包含 PyTorch 张量（标记 ID 和注意力掩码）的 `BatchEncoding` 对象，以及另一个包含标签的张量。对于分词，我们将简单地使用我们刚刚加载的预训练 WordPiece 分词器：

```py
def collate_fn(batch, tokenizer=bert_tokenizer):
    reviews = [review["text"] for review in batch]
    labels = [[review["label"]] for review in batch]
    encodings = tokenizer(reviews, padding=True, truncation=True,
                          max_length=200, return_tensors="pt")
    labels = torch.tensor(labels, dtype=torch.float32)
    return encodings, labels

batch_size = 256
imdb_train_loader = DataLoader(imdb_train_set, batch_size=batch_size,
                               collate_fn=collate_fn, shuffle=True)
imdb_valid_loader = DataLoader(imdb_valid_set, batch_size=batch_size,
                               collate_fn=collate_fn)
imdb_test_loader = DataLoader(imdb_test_set, batch_size=batch_size,
                              collate_fn=collate_fn)
```

现在我们已经准备好创建我们的情感分析模型：

```py
class SentimentAnalysisModel(nn.Module):
    def __init__(self, vocab_size, n_layers=2, embed_dim=128, hidden_dim=64,
                 pad_id=0, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim,
                                  padding_idx=pad_id)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers,
                          batch_first=True, dropout=dropout)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, encodings):
        embeddings = self.embed(encodings["input_ids"])
        _outputs, hidden_states = self.gru(embeddings)
        return self.output(hidden_states[-1])
```

如您所见，这个模型与我们之前的莎士比亚模型非常相似，但有一些重要的区别：

+   在创建 `nn.Embedding` 层时，我们将 `padding_idx` 参数设置为我们的填充 ID。这确保填充 ID 被嵌入为不可训练的零向量，以减少填充标记对损失的影响。

+   由于这是一个序列到向量模型，而不是序列到序列模型，我们只需要顶层 GRU 层的最后一个输出来进行最终预测（通过输出 `nn.Linear` 层）。我们可以使用 `outputs[:, -1]` 而不是 `hidden_states[-1]`，因为它们是相等的。

+   输出的 `nn.Linear` 层只有一个输出维度，因为它是一个二分类模型。最终输出将是一个二维张量，包含一个列，每条评论有一个 logit，正面评论为正，负面评论为负。

+   `forward()` 方法接受一个 `BatchEncoding` 对象作为输入，其中包含标记 ID（可能已填充和截断）。

我们可以使用 `nn.BCEWithLogitsLoss` 训练这个模型，因为这是一个二元分类任务。它在验证集上的准确率接近 85%，这相当不错，尽管最好的模型达到了人类水平，略高于 90% 的准确率。可能无法进一步提高；因为许多评论是模糊的，分类它们感觉就像抛硬币一样。

我们模型的一个问题是我们没有完全忽略填充标记。确实，如果一条评论以许多填充标记结束，`nn.GRU` 模块将不得不处理它们，等到它处理完所有这些标记时，它可能已经忘记了评论的内容。为了避免这种情况，我们可以使用一个 *打包序列* 而不是常规张量。打包序列是一种特殊的数据结构，用于高效地表示一批长度可变的序列。你可以使用 `pack_padded_sequence()` 函数将包含填充序列的张量转换为打包序列对象，相反，你可以在需要将打包序列对象转换为填充张量时使用 `pad_packed_sequence()` 函数：

```py
>>> from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
>>> sequences = torch.tensor([[1, 2, 0, 0], [5, 6, 7, 8]])
>>> packed = pack_padded_sequence(sequences, lengths=(2, 4),
...                               enforce_sorted=False, batch_first=True)
...
>>> packed
PackedSequence(data=tensor([5, 1, 6, 2, 7, 8]), [...])
>>> padded, lengths = pad_packed_sequence(packed, batch_first=True)
>>> padded, lengths
(tensor([[1, 2, 0, 0],
 [5, 6, 7, 8]]),
 tensor([2, 4]))
```

默认情况下，`pack_padded_sequence()` 函数假定批次中的序列按从长到短的顺序排列。如果不是这种情况，你必须设置 `enforce_sorted=False`。此外，该函数还假定时间维度在批次维度之前。如果批次维度是第一个，你必须设置 `batch_first=True`。

PyTorch 的循环层支持打包序列：它们高效地处理序列，在每个序列的末尾停止。所以让我们更新我们的情感分析模型以使用打包序列。在 `forward()` 方法中，只需将 `self.gru(embeddings)` 行替换为以下代码：

```py
lengths = encodings["attention_mask"].sum(dim=1)
packed = pack_padded_sequence(embeddings, lengths=lengths.cpu(),
                              batch_first=True, enforce_sorted=False)
_outputs, hidden_states = self.gru(packed)
```

这段代码首先计算批次中每个序列的长度，就像我们之前做的那样，然后打包嵌入张量，并将打包序列传递给 `nn.GRU` 模块。有了这个，模型将正确处理序列，而不会被任何填充标记所打扰。实际上，在创建 `nn.Embedding` 层时，你不再需要设置 `padding_idx`，但这并不妨碍，并且使调试变得更容易，所以我更喜欢保留它。

提高我们模型的另一种方法是让它从两个方向查看评论：从左到右，然后从右到左。让我们看看这是如何工作的。

### 注意

如果你将一个打包序列传递给 `nn.GRU` 模块，它的输出也将是一个打包序列，在你将其传递给下一层之前，你需要将其转换回填充张量。幸运的是，我们不需要这些输出用于我们的情感分析模型，只需要隐藏状态。

## 双向 RNN

在每个时间步，一个常规循环层只查看过去和当前的输入，然后生成其输出。换句话说，它是*因果的*，这意味着它不能看到未来。这种类型的 RNN 在预测时间序列或在序列到序列（seq2seq）模型的解码器中是有意义的。但对于文本分类或 seq2seq 模型的编码器中的任务，通常更倾向于在编码给定单词之前查看下一个单词。

例如，考虑短语“the right arm”，“the right person”，和“the right to speak”：为了正确编码单词“right”，你需要查看前方。一种解决方案是在相同的输入上运行两个循环层，一个从左到右读取单词，另一个从右到左读取，然后在每个时间步将它们的输出合并，通常是通过连接它们。这就是*双向循环层*所做的事情（参见图 14-4）。

![图示双向循环层，显示双向处理输入并合并输出。](img/hmls_1404.png)

###### 图 14-4\. 双向循环层

要使我们的情感分析模型双向，我们只需在创建 `nn.GRU` 层时设置 `bidirectional=True`（这也适用于 `nn.RNN` 和 `nn.LSTM` 模块）。

然而，一旦我们这样做，我们必须对我们的模型进行一些调整。特别是，我们必须将输出 `nn.Linear` 层的输入维度加倍，因为隐藏状态的大小将加倍：

```py
self.output = nn.Linear(2 * hidden_dim, 1)
```

我们还必须在将结果传递到输出层之前，连接 GRU 顶层的前向和后向隐藏状态。为此，我们可以将 `forward()` 方法的最后一行（即 `return self.output(hidden_states[-1])`）替换为以下代码：

```py
n_dims = self.output.in_features
top_states = hidden_states[-2:].permute(1, 0, 2).reshape(-1, n_dims)
return self.output(top_states)
```

让我们看看中间行是如何工作的：

+   到目前为止，`nn.GRU` 模块返回的隐藏状态的形状是 [*层数*，*批量大小*，*隐藏大小*]，在我们的例子中是 [2, 256, 64]。但是当我们设置 `bidirectional=True` 时，我们将第一个维度的尺寸加倍，所以我们现在有一个形状为 [4, 256, 64] 的形状：张量包含第 1 层前向、第 1 层后向、第 2 层前向和第 2 层后向的隐藏状态。由于我们只想获取顶层的前向和后向隐藏状态，我们必须获取 `hidden_states[-2:]`。

+   我们还需要连接前向和后向状态。一种方法是通过使用 `permute(1, 0, 2)` 对顶层隐藏状态的前两个维度进行置换，以获得形状 [256, 2, 64]，然后使用 `reshape(-1, n_dims)`（其中 `n_dims` 等于 128）重塑结果，以获得所需的形状：[256, 2 * 64]。

### 注意
