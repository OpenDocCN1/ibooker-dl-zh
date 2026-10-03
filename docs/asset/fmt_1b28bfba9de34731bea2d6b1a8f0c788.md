
```py
out_en_ids=[[en_word_dict.get(w,UNK) for w in s] for s in en_tokens]
out_fr_ids=[[fr_word_dict.get(w,UNK) for w in s] for s in fr_tokens]
sorted_ids=sorted(range(len(out_en_ids)),
                  key=lambda x:len(out_en_ids[x]))
out_en_ids=[out_en_ids[x] for x in sorted_ids]
out_fr_ids=[out_fr_ids[x] for x in sorted_ids]
```

接下来，我们将数值表示放入批次进行训练：

```py
import numpy as np

batch_size=128
idx_list=np.arange(0,len(en_tokens),batch_size)
np.random.shuffle(idx_list)

batch_indexs=[]
for idx in idx_list:
    batch_indexs.append(np.arange(idx,min(len(en_tokens),
                                          idx+batch_size)))
```

注意，我们在将观察结果放入批次之前，已经根据英文短语的长度对训练数据集中的观察结果进行了排序。这种方法确保了每个批次中的观察结果具有可比的长度，从而减少了填充的需要。因此，这种方法不仅减少了训练数据的总体大小，还加速了训练过程。

为了将批次中的序列填充到相同的长度，我们定义了以下函数：

```py
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)                                                   ①
    padded_seq = np.array([np.concatenate([x, [padding] * (ML - len(x))])
        if len(x) < ML else x for x in X])                        ②
    return padded_seq
```

① 找出批次中最长序列的长度。

② 如果批次比最长的序列短，则在序列末尾添加 0。

函数 `seq_padding()` 首先在批次中识别最长的序列。然后，它将零添加到较短的序列的末尾，以确保批次中的每个序列都与这个最大长度匹配。

为了节省空间，我们在本地模块 `ch09util.py` 中创建了一个 `Batch()` 类，您在上一个章节中已下载（见图 10.1）。

列表 10.4 在本地模块中创建一个 `Batch()` 类

```py
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
class Batch:
    def __init__(self, src, trg=None, pad=0):
        src = torch.from_numpy(src).to(DEVICE).long()
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)              ①
        if trg is not None:
            trg = torch.from_numpy(trg).to(DEVICE).long()
            self.trg = trg[:, :-1]                              ②
            self.trg_y = trg[:, 1:]                             ③
            self.trg_mask = make_std_mask(self.trg, pad)        ④
            self.ntokens = (self.trg_y != pad).data.sum()
```

① 创建一个源掩码以隐藏句子末尾的填充

② 为解码器创建输入

③ 将输入向右移动一个标记并将其用作输出

④ 创建一个目标掩码

![](img/CH10_F01_Liu.png)

图 10.1 `Batch()` 类的作用是什么？`Batch()` 类接受两个输入：`src` 和 `trg`，分别代表源语言和目标语言的索引序列。它向训练数据添加了几个属性：`src_mask`，用于隐藏填充的源掩码；`modified trg`，解码器的输入；`trg_y`，解码器的输出；`trg_mask`，用于隐藏填充和未来标记的目标掩码。

`Batch()` 类处理一批英语和法语短语，将它们转换为适合训练的格式。为了使这个解释更具体，以英语短语“How are you?”及其法语对应短语“Comment êtes-vous?”为例。`Batch()` 类接收两个输入：`src`，代表“How are you?”中标记的索引序列，以及 `trg`，代表“Comment êtes-vous?”中标记的索引序列。这个类生成一个张量 `src_mask`，用于隐藏句子末尾的填充。例如，句子“How are you?”被分解成六个标记：`['BOS', 'how', 'are', 'you', '?', 'EOS']`。如果这个序列是长度为八个标记的批次的一部分，则在末尾添加两个零。`src_mask` 张量指示模型在这种情况下忽略最后的两个标记。

`Batch()` 类还准备了 Transformer 解码器的输入和输出。以法语短语“Comment êtes-vous?”为例，它被转换成六个标记：`['BOS', 'comment', 'et', 'es-vous', '?', 'EOS']`。这些前五个标记的索引作为解码器的输入，命名为 `trg`。接下来，我们将这个输入向右移动一个标记以形成解码器的输出，`trg_y`。因此，输入包含 `['BOS', 'comment', 'et', 'es-vous', '?']` 的索引，而输出则包含 `['comment', 'et', 'es-vous', '?', 'EOS']` 的索引。这种方法与我们第八章讨论的内容相似，旨在迫使模型根据前面的标记预测下一个标记。

`Batch()` 类还生成了一个用于解码器输入的掩码，`trg_mask`。这个掩码的目的是隐藏输入中的后续标记，确保模型仅依赖于先前标记进行预测。这个掩码是由 `make_std_mask()` 函数生成的，该函数定义在本地模块 `ch09util` 中：

```py
import numpy as np
def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape),k=1).astype('uint8')
    output = torch.from_numpy(subsequent_mask) == 0
    return output
def make_std_mask(tgt, pad):
    tgt_mask=(tgt != pad).unsqueeze(-2)
    output=tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    return output 
```

`subsequent_mask()` 函数为序列生成一个特定的掩码，指示模型仅关注实际序列，并忽略末尾的填充零，这些填充零仅用于标准化序列长度。另一方面，`make_std_mask()` 函数构建了一个针对目标序列的标准掩码。这个标准掩码具有双重作用，即隐藏填充零和目标序列中的后续标记。

接下来，我们导入 `Batch()` 类从本地模块，并使用它来创建训练数据批次：

```py
from utils.ch09util import Batch

class BatchLoader():
    def __init__(self):
        self.idx=0
    def __iter__(self):
        return self
    def __next__(self):
        self.idx += 1
        if self.idx<=len(batch_indexs):
            b=batch_indexs[self.idx-1]
            batch_en=[out_en_ids[x] for x in b]
            batch_fr=[out_fr_ids[x] for x in b]
            batch_en=seq_padding(batch_en)
            batch_fr=seq_padding(batch_fr)
            return Batch(batch_en,batch_fr)
        raise StopIteration
```

`BatchLoader()` 类创建用于训练的数据批次。列表中的每个批次包含 128 对，其中每对包含一个英语短语及其对应的法语翻译的数值表示。

## 10.2 词嵌入和位置编码

在上一节进行分词之后，英语和法语短语被表示为一系列的索引。在本节中，您将使用词嵌入将这些索引（本质上是一维热向量）转换为紧凑的向量表示。这样做可以捕捉短语中标记的语义信息和相互关系。词嵌入还可以提高训练效率：与庞大的热向量相比，词嵌入使用连续的、低维向量来减少模型的复杂性和维度。

注意力机制同时处理短语中的所有标记，而不是按顺序处理。这提高了其效率，但本身并不允许它识别标记的序列顺序。因此，我们将通过使用不同频率的正弦和余弦函数，将位置编码添加到输入嵌入中。

### 10.2.1 词嵌入

英语和法语短语的数值表示涉及大量的索引。为了确定每种语言所需的唯一索引的确切数量，我们可以计算 `en_word_dict` 和 `fr_word_dict` 字典中唯一元素的数量。这样做会生成每种语言词汇表中唯一标记的总数（我们将在后面将它们作为 Transformer 的输入使用）：

```py
src_vocab = len(en_word_dict)
tgt_vocab = len(fr_word_dict)
print(f"there are {src_vocab} distinct English tokens")
print(f"there are {tgt_vocab} distinct French tokens")
```

输出是

```py
there are 11055 distinct English tokens
there are 11239 distinct French tokens
```

在我们的数据集中，有 11,055 个唯一的英语标记和 11,239 个唯一的法语标记。对这些使用一维热编码会导致训练时参数数量过高。为了解决这个问题，我们将采用词嵌入，它将数值表示压缩成连续的向量，每个向量的长度为 `d_model = 256`。

这是通过使用定义在本地模块 `ch09util` 中的 `Embeddings()` 类来实现的：

```py
import math

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        out = self.lut(x) * math.sqrt(self.d_model)
        return out
```

之前定义的 `Embeddings()` 类使用了 PyTorch 的 `Embedding()` 类。它还将输出乘以 `d_model` 的平方根，即 256。这种乘法是为了抵消在计算注意力分数过程中发生的除以 `d_model` 的平方根。`Embeddings()` 类降低了英语和法语短语的数值表示的维度。我们在第八章详细讨论了 PyTorch 的 `Embedding()` 类是如何工作的。

### 10.2.2 位置编码

为了准确表示输入和输出中元素序列的顺序，我们在本地模块中引入了 `PositionalEncoding()` 类。

列表 10.5 计算位置编码的类

```py
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):       ①
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model, device=DEVICE)
        position = torch.arange(0., max_len, 
                                device=DEVICE).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0., d_model, 2, device=DEVICE)
            * -(math.log(10000.0) / d_model))
        pe_pos = torch.mul(position, div_term)
        pe[:, 0::2] = torch.sin(pe_pos)                        ②
        pe[:, 1::2] = torch.cos(pe_pos)                        ③
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  

    def forward(self, x):
        x=x+self.pe[:,:x.size(1)].requires_grad_(False)        ④
        out=self.dropout(x)
        return out
```

① 初始化类，允许最大 5,000 个位置

② 将正弦函数应用于向量的偶数索引

③ 将余弦函数应用于向量的奇数索引

④ 将位置编码添加到词嵌入中

`PositionalEncoding()` 类使用正弦函数对偶数索引进行编码，使用余弦函数对奇数索引进行编码来生成序列位置的向量。需要注意的是，在 `PositionalEncoding()` 类中，包含了 `requires_grad_(False)` 参数，因为这些值不需要进行训练。它们在所有输入中保持不变，并且在训练过程中不会改变。

例如，来自英语短语 `['BOS', 'how', 'are', 'you', '?', 'EOS']` 的六个标记的索引首先通过一个词嵌入层进行处理。这一步将这些索引转换为一个维度为 (1, 6, 256) 的张量：1 表示批处理中只有一个序列；6 表示序列中有 6 个标记；256 表示每个标记由一个 256 价值的向量表示。在完成词嵌入过程后，`PositionalEncoding()` 类被用来计算对应于标记 `['BOS', 'how', 'are', 'you', '?', 'EOS']` 的索引的位置编码。这样做是为了向模型提供有关每个标记在序列中位置的信息。更好的是，我们可以通过以下代码块告诉你前六个标记的位置编码的确切值：

```py
from utils.ch09util import PositionalEncoding
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pe = PositionalEncoding(256, 0.1)                              ①
x = torch.zeros(1, 8, 256).to(DEVICE)                          ②
y = pe.forward(x)                                              ③
print(f"the shape of positional encoding is {y.shape}")
print(y)                                                       ④
```

① 实例化 `PositionalEncoding()` 类并将模型维度设置为 256

② 创建一个词嵌入并将其填充为零

③ 通过向词嵌入添加位置编码来计算输入嵌入

④ 打印出输入嵌入，由于词嵌入被设置为零，因此与位置编码相同

我们首先创建一个 `PositionalEncoding()` 类的实例 `pe`，将模型维度设置为 256，并将 dropout 率设置为 0.1。由于这个类的输出是词嵌入和位置编码的和，我们创建一个填充为零的词嵌入并将其输入到 `pe` 中：这样输出就是位置编码。

运行前面的代码块后，你会看到以下输出：
