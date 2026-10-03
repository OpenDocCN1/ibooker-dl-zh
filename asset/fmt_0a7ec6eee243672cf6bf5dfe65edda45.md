
在本节中，你将开始使用 PyTorch 的内置 LSTM 层构建一个 LSTM 模型。这个模型将从词嵌入层开始，将每个索引转换为一个 128 维的密集向量。你的训练数据将通过这个嵌入层，然后输入到 LSTM 层。这个 LSTM 层被设计成按顺序处理序列的元素。在 LSTM 层之后，数据将进入一个线性层，其输出大小与你的词汇表大小相匹配。LSTM 模型生成的输出本质上是 logits，作为 softmax 函数的输入来计算概率。

一旦你构建了 LSTM 模型，下一步将涉及使用你的训练数据来训练这个模型。这个训练阶段对于提高模型理解和生成与提供的数据一致的模式的能力至关重要。

### 8.4.1 构建 LSTM 模型

在列表 8.3 中，我们定义了一个`WordLSTM()`类，作为我们的 LSTM 模型，用于训练以生成《安娜·卡列尼娜》风格的文本。该类定义如下所示。

**列表 8.3 定义`WordLSTM()`类**

```py
from torch import nn
device="cuda" if torch.cuda.is_available() else "cpu"
class WordLSTM(nn.Module):
    def __init__(self, input_size=128, n_embed=128,
             n_layers=3, drop_prob=0.2):
        super().__init__()
        self.input_size=input_size
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_embed = n_embed
        vocab_size=len(word_to_int)
        self.embedding=nn.Embedding(vocab_size,n_embed)    ①
        self.lstm = nn.LSTM(input_size=self.input_size,
            hidden_size=self.n_embed,
            num_layers=self.n_layers,
            dropout=self.drop_prob,batch_first=True)       ②
        self.fc = nn.Linear(input_size, vocab_size)    

    def forward(self, x, hc):
        embed=self.embedding(x)
        x, hc = self.lstm(embed, hc)                       ③
        x = self.fc(x)
        return x, hc      

    def init_hidden(self, n_seqs):                         ④
        weight = next(self.parameters()).data
        return (weight.new(self.n_layers,
                           n_seqs, self.n_embed).zero_(),
                weight.new(self.n_layers,
                           n_seqs, self.n_embed).zero_())  
```

① 训练数据首先通过嵌入层。

② 使用 PyTorch `LSTM()` 类创建一个 LSTM 层

③ 在每个时间步中，LSTM 层使用前一个标记和隐藏状态来预测下一个标记和下一个隐藏状态。

④ 初始化输入序列中第一个标记的隐藏状态

之前定义的 `WordLSTM()` 类有三个层：词嵌入层、LSTM 层和最终的线性层。我们将 `n_layers` 参数的值设置为 3，这意味着 LSTM 层将三个 LSTM 堆叠在一起形成一个堆叠的 LSTM，最后两个 LSTM 将前一个 LSTM 的输出作为输入。`init_hidden()` 方法在模型使用序列的第一个元素进行预测时将隐藏状态填充为零。在每个时间步中，输入是当前标记和前一个隐藏状态，输出是下一个标记和下一个隐藏状态。

**`torch.nn.Embedding()` 类的工作原理**

PyTorch 中的 `torch.nn.Embedding()` 类用于在神经网络中创建嵌入层。嵌入层是一个可训练的查找表，它将整数索引映射到密集的、连续的向量表示（嵌入）。

当你创建 `torch.nn.Embedding()` 的实例时，你需要指定两个主要参数：`num_embeddings`，词汇表的大小（唯一标记的总数），以及 `embedding_dim`，每个嵌入向量的大小（输出嵌入的维度性）。

在内部，该类创建一个形状为 `(num_embeddings, embedding_dim)` 的矩阵（或查找表），其中每一行对应于特定索引的嵌入向量。最初，这些嵌入是随机初始化的，但在训练过程中通过反向传播进行学习和更新。

当你将索引张量传递给嵌入层（在网络的前向传递过程中）时，它会在查找表中查找相应的嵌入向量并返回它们。关于该类的更多信息可在 PyTorch [`mng.bz/n0Zd`](https://mng.bz/n0Zd) 获取。

我们创建 `WordLSTM()` 类的实例，并将其用作我们的 LSTM 模型，如下所示：

```py
model=WordLSTM().to(device)
```

当创建 LSTM 模型时，权重是随机初始化的。当我们使用 `(x, y)` 对来训练模型时，LSTM 通过调整模型参数学习根据序列中的所有先前标记预测下一个标记。正如我们在图 8.2 中所展示的，LSTM 学习根据当前标记和当前隐藏状态预测下一个标记和下一个隐藏状态，这是所有先前标记信息的总结。

我们使用学习率为 0.0001 的 Adam 优化器。损失函数是交叉熵损失，因为这是一个多类别分类问题：模型试图从包含 12,778 个选择的字典中预测下一个标记：

```py
lr=0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_func = nn.CrossEntropyLoss()
```

现在 LSTM 模型已经构建，我们将使用之前准备好的训练数据批次来训练模型。

### 8.4.2 训练 LSTM 模型

在每个训练周期中，我们遍历训练集中的所有数据批次`(x, y)`。LSTM 模型接收输入序列 `x`，并生成一个预测输出序列 *ŷ*。这个预测与实际输出序列 `y` 进行比较，以计算交叉熵损失，因为我们实际上在这里进行多类别分类。然后我们调整模型的参数以减少这个损失，就像我们在第二章中分类服装物品时做的那样。

虽然我们可以将我们的数据分为训练集和验证集，直到在验证集上不再看到进一步的改进（就像我们在第二章中所做的那样）来训练模型，但我们的主要目标是掌握 LSTM 模型的工作原理，而不一定是实现最佳参数调整。因此，我们将模型训练 50 个周期。

**列表 8.4 训练 LSTM 模型以生成文本**

```py
model.train()

for epoch in range(50):
    tloss=0
    sh,sc = model.init_hidden(batch_size)
    for i, (x,y) in enumerate(loader):                      ①
        if x.shape[0]==batch_size:
            inputs, targets = x.to(device), y.to(device)
            optimizer.zero_grad()
            output, (sh,sc) = model(inputs, (sh,sc))        ②
            loss = loss_func(output.transpose(1,2),targets) ③ 
            sh,sc=sh.detach(),sc.detach()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()                                ④
            tloss+=loss.item()
        if (i+1)%1000==0:
            print(f"at epoch {epoch} iteration {i+1}\
            average loss = {tloss/(i+1)}")
```

① 遍历训练数据中的所有`(x,y)`批次

② 使用模型来预测输出序列

③ 将预测结果与实际输出进行比较，并计算损失

④ 调整模型参数以最小化损失

在前面的代码列表中，`sh` 和 `sc` 一起构成了隐藏状态。特别是，细胞状态 `sc` 作为传送带，在许多时间步中携带信息，每个时间步都会添加或删除信息。`sh` 是 LSTM 单元在给定时间步的输出。它包含有关当前输入的信息，并用于将信息传递给序列中的下一个 LSTM 单元。

如果您有 CUDA 支持的 GPU，这次训练大约需要 6 个小时。如果您只使用 CPU，可能需要一两天，具体取决于您的硬件。或者您可以从我的网站下载预训练的权重：[`mng.bz/vJZa`](https://mng.bz/vJZa)。

接下来，我们将训练好的模型权重保存在本地文件夹中：

```py
import pickle

torch.save(model.state_dict(),"files/wordLSTM.pth")
with open("files/word_to_int.p","wb") as fb:
    pickle.dump(word_to_int, fb)
```

字典 `word_to_int` 也保存在您的计算机上，这是一个实用的步骤，确保您可以在不重复分词过程的情况下，使用训练好的模型生成文本。

## 8.5 使用训练好的 LSTM 模型生成文本

现在您已经有一个训练好的 LSTM 模型，您将在这个部分学习如何使用它来生成文本。目标是查看训练好的模型是否可以通过迭代预测下一个标记来生成语法正确且连贯的文本。您还将学习如何使用温度和 top-K 采样来控制生成文本的创造性。

在使用训练好的 LSTM 模型生成文本时，我们以提示作为模型的初始输入。我们使用训练好的模型来预测最可能的下一个标记。将下一个标记附加到提示后，我们将新的序列输入到训练好的模型中，再次预测下一个标记。我们重复这个过程，直到序列达到一定的长度。

### 8.5.1 通过预测下一个标记来生成文本

首先，我们从本地文件夹中加载训练好的模型权重和字典 `word_to_int`：

```py
model.load_state_dict(torch.load("files/wordLSTM.pth",
                                    map_location=device))
with open("files/word_to_int.p","rb") as fb:
    word_to_int = pickle.load(fb)
int_to_word={v:k for k,v in word_to_int.items()}
```

书籍的 GitHub 仓库中也提供了`word_to_int.p`文件。我们将字典`word_to_int`中的键和值的顺序进行交换，以创建字典`int_to_word`。

要使用训练好的 LSTM 模型生成文本，我们需要一个提示作为生成文本的起点。我们将默认提示设置为“Anna and the”。一个简单的方法是限制生成的文本长度，例如 200 个标记：一旦达到所需长度，我们就要求模型停止生成。

下面的列表定义了一个`sample()`函数，用于根据提示生成文本。

**列表 8.5：用于生成文本的`sample()`函数**

```py
import numpy as np
def sample(model, prompt, length=200):
    model.eval()
    text = prompt.lower().split(' ')
    hc = model.init_hidden(1)
    length = length - len(text)                              ①
    for i in range(0, length):
        if len(text)<= seq_len:
            x = torch.tensor([[word_to_int[w] for w in text]])
        else:
            x = torch.tensor([[word_to_int[w] for w \
in text[-seq_len:]]])                                        ②
        inputs = x.to(device)
        output, hc = model(inputs, hc)                       ③
        logits = output[0][-1]
        p = nn.functional.softmax(logits, dim=0).detach().cpu().numpy()
        idx = np.random.choice(len(logits), p=p)             ④
        text.append(int_to_word[idx])                        ⑤
    text=" ".join(text)
    for m in ",.:;?!$()/_&%*@'`":
        text=text.replace(f" {m}", f"{m} ")
    text=text.replace('"  ', '"')
    text=text.replace("'  ", "'")
    text=text.replace('" ', '"')
    text=text.replace("' ", "'")
    return text  
```

① 确定需要生成多少个标记

② 输入是当前序列；如果它超过 100 个标记，则对其进行修剪

③ 使用训练模型进行预测

④ 根据预测概率选择下一个标记

⑤ 将预测的下一个标记添加到序列中并重复

`sample()`函数接受三个参数。第一个是您将要使用的训练好的 LSTM 模型。第二个是文本生成的起始提示，可以是任何长度的短语，用引号括起来。第三个参数指定要生成的文本长度，以标记为单位，默认值为 200 个标记。

在函数内部，我们首先从总期望长度中减去提示中的标记数量，以确定需要生成的标记数量。在生成下一个标记时，我们考虑当前序列的长度。如果它少于 100 个标记，我们将整个序列输入到模型中；如果它超过 100 个标记，则只使用序列的最后 100 个标记作为输入。然后将这个输入送入训练好的 LSTM 模型来预测下一个标记，我们将这个预测的标记添加到当前序列中。我们继续这个过程，直到序列达到所需长度。

当生成下一个标记时，模型使用 NumPy 中的 `random.choice(len(logits), p = p)` 方法。在这里，该方法的第一参数表示选择范围，在这个例子中是 `len(logits) = 12778`。这意味着模型将随机选择一个从 0 到 12,777 的整数，每个整数对应词汇表中的一个不同标记。第二个参数 `p` 是一个包含 12,778 个元素的数组，其中每个元素表示从词汇表中选择相应标记的概率。在这个数组中，概率较高的标记更有可能被选中。

让我们使用“Anna and the prince”作为提示（确保在使用自己的提示时在标点符号前加上空格）来生成一段文本：

```py
torch.manual_seed(42)
np.random.seed(42)
print(sample(model, prompt='Anna and the prince'))  
```

在这里，我将 PyTorch 和 NumPy 中的随机种子数固定为 42，以防你想要重现结果。生成的段落是
