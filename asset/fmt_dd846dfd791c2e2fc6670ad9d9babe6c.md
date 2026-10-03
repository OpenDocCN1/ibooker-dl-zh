
```py
class LMModel3(Module):
    def __init__(self, vocab_sz, n_hidden):
        self.i_h = nn.Embedding(vocab_sz, n_hidden)
        self.h_h = nn.Linear(n_hidden, n_hidden)
        self.h_o = nn.Linear(n_hidden,vocab_sz)
        self.h = 0

    def forward(self, x):
        for i in range(3):
            self.h = self.h + self.i_h(x[:,i])
            self.h = F.relu(self.h_h(self.h))
        out = self.h_o(self.h)
        self.h = self.h.detach()
        return out

    def reset(self): self.h = 0
```

无论我们选择什么序列长度，这个模型将具有相同的激活，因为隐藏状态将记住上一批次的最后激活。唯一不同的是在每一步计算的梯度：它们将仅在过去的序列长度标记上计算，而不是整个流。这种方法称为*时间穿梭反向传播*（BPTT）。

# 术语：时间穿梭反向传播

将一个神经网络有效地视为每个时间步长一个层（通常使用循环重构），并以通常的方式在其上计算梯度。为了避免内存和时间不足，我们通常使用*截断* BPTT，每隔几个时间步“分离”隐藏状态的计算历史。

要使用`LMModel3`，我们需要确保样本按照一定顺序进行查看。正如我们在第十章中看到的，如果第一批的第一行是我们的`dset[0]`，那么第二批应该将`dset[1]`作为第一行，以便模型看到文本流动。

`LMDataLoader`在第十章中为我们做到了这一点。这次我们要自己做。

为此，我们将重新排列我们的数据集。首先，我们将样本分成`m = len(dset) // bs`组（这相当于将整个连接数据集分成，例如，64 个大小相等的部分，因为我们在这里使用`bs=64`）。`m`是每个这些部分的长度。例如，如果我们使用整个数据集（尽管我们实际上将在一会儿将其分成训练和验证），我们有：

```py
m = len(seqs)//bs
m,bs,len(seqs)
```

```py
(328, 64, 21031)
```

第一批将由样本组成

```py
(0, m, 2*m, ..., (bs-1)*m)
```

样本的第二批

```py
(1, m+1, 2*m+1, ..., (bs-1)*m+1)
```

等等。这样，每个时期，模型将在每批次的每行上看到大小为`3*m`的连续文本块（因为每个文本的大小为 3）。

以下函数执行重新索引：

```py
def group_chunks(ds, bs):
    m = len(ds) // bs
    new_ds = L()
    for i in range(m): new_ds += L(ds[i + m*j] for j in range(bs))
    return new_ds
```

然后，我们在构建`DataLoaders`时只需传递`drop_last=True`来删除最后一个形状不为`bs`的批次。我们还传递`shuffle=False`以确保文本按顺序阅读：

```py
cut = int(len(seqs) * 0.8)
dls = DataLoaders.from_dsets(
    group_chunks(seqs[:cut], bs),
    group_chunks(seqs[cut:], bs),
    bs=bs, drop_last=True, shuffle=False)
```

我们添加的最后一件事是通过`Callback`对训练循环进行微调。我们将在第十六章中更多地讨论回调；这个回调将在每个时期的开始和每个验证阶段之前调用我们模型的`reset`方法。由于我们实现了该方法来将模型的隐藏状态设置为零，这将确保我们在阅读这些连续文本块之前以干净的状态开始。我们也可以开始训练更长一点：

```py
learn = Learner(dls, LMModel3(len(vocab), 64), loss_func=F.cross_entropy,
                metrics=accuracy, cbs=ModelResetter)
learn.fit_one_cycle(10, 3e-3)
```

| epoch | train_loss | valid_loss | accuracy | time |
| --- | --- | --- | --- | --- |
| 0 | 1.677074 | 1.827367 | 0.467548 | 00:02 |
| 1 | 1.282722 | 1.870913 | 0.388942 | 00:02 |
| 2 | 1.090705 | 1.651793 | 0.462500 | 00:02 |
| 3 | 1.005092 | 1.613794 | 0.516587 | 00:02 |
| 4 | 0.965975 | 1.560775 | 0.551202 | 00:02 |
| 5 | 0.916182 | 1.595857 | 0.560577 | 00:02 |
| 6 | 0.897657 | 1.539733 | 0.574279 | 00:02 |
| 7 | 0.836274 | 1.585141 | 0.583173 | 00:02 |
| 8 | 0.805877 | 1.629808 | 0.586779 | 00:02 |
| 9 | 0.795096 | 1.651267 | 0.588942 | 00:02 |

这已经更好了！下一步是使用更多目标并将它们与中间预测进行比较。

## 创建更多信号

我们当前方法的另一个问题是，我们仅为每三个输入单词预测一个输出单词。因此，我们反馈以更新权重的信号量不如可能的那么大。如果我们在每个单词后预测下一个单词，而不是每三个单词，将会更好，如图 12-5 所示。

![RNN 在每个标记后进行预测](img/dlcf_1205.png)

###### 图 12-5。RNN 在每个标记后进行预测

这很容易添加。我们需要首先改变我们的数据，使得因变量在每个三个输入词后的每个三个词中都有。我们使用一个属性`sl`（用于序列长度），并使其稍微变大：

```py
sl = 16
seqs = L((tensor(nums[i:i+sl]), tensor(nums[i+1:i+sl+1]))
         for i in range(0,len(nums)-sl-1,sl))
cut = int(len(seqs) * 0.8)
dls = DataLoaders.from_dsets(group_chunks(seqs[:cut], bs),
                             group_chunks(seqs[cut:], bs),
                             bs=bs, drop_last=True, shuffle=False)
```

查看`seqs`的第一个元素，我们可以看到它包含两个相同大小的列表。第二个列表与第一个相同，但偏移了一个元素：

```py
[L(vocab[o] for o in s) for s in seqs[0]]
```

```py
[(#16) ['one','.','two','.','three','.','four','.','five','.'...],
 (#16) ['.','two','.','three','.','four','.','five','.','six'...]]
```

现在我们需要修改我们的模型，使其在每个单词之后输出一个预测，而不仅仅是在一个三个词序列的末尾：

```py
class LMModel4(Module):
    def __init__(self, vocab_sz, n_hidden):
        self.i_h = nn.Embedding(vocab_sz, n_hidden)
        self.h_h = nn.Linear(n_hidden, n_hidden)
        self.h_o = nn.Linear(n_hidden,vocab_sz)
        self.h = 0

    def forward(self, x):
        outs = []
        for i in range(sl):
            self.h = self.h + self.i_h(x[:,i])
            self.h = F.relu(self.h_h(self.h))
            outs.append(self.h_o(self.h))
        self.h = self.h.detach()
        return torch.stack(outs, dim=1)

    def reset(self): self.h = 0
```

这个模型将返回形状为`bs x sl x vocab_sz`的输出（因为我们在`dim=1`上堆叠）。我们的目标的形状是`bs x sl`，所以在使用`F.cross_entropy`之前，我们需要将它们展平：

```py
def loss_func(inp, targ):
    return F.cross_entropy(inp.view(-1, len(vocab)), targ.view(-1))
```

我们现在可以使用这个损失函数来训练模型：

```py
learn = Learner(dls, LMModel4(len(vocab), 64), loss_func=loss_func,
                metrics=accuracy, cbs=ModelResetter)
learn.fit_one_cycle(15, 3e-3)
```

| epoch | train_loss | valid_loss | accuracy | time |
| --- | --- | --- | --- | --- |
| 0 | 3.103298 | 2.874341 | 0.212565 | 00:01 |
| 1 | 2.231964 | 1.971280 | 0.462158 | 00:01 |
| 2 | 1.711358 | 1.813547 | 0.461182 | 00:01 |
| 3 | 1.448516 | 1.828176 | 0.483236 | 00:01 |
| 4 | 1.288630 | 1.659564 | 0.520671 | 00:01 |
| 5 | 1.161470 | 1.714023 | 0.554932 | 00:01 |
| 6 | 1.055568 | 1.660916 | 0.575033 | 00:01 |
| 7 | 0.960765 | 1.719624 | 0.591064 | 00:01 |
| 8 | 0.870153 | 1.839560 | 0.614665 | 00:01 |
| 9 | 0.808545 | 1.770278 | 0.624349 | 00:01 |
| 10 | 0.758084 | 1.842931 | 0.610758 | 00:01 |
| 11 | 0.719320 | 1.799527 | 0.646566 | 00:01 |
| 12 | 0.683439 | 1.917928 | 0.649821 | 00:01 |
| 13 | 0.660283 | 1.874712 | 0.628581 | 00:01 |
| 14 | 0.646154 | 1.877519 | 0.640055 | 00:01 |

我们需要训练更长时间，因为任务有点变化，现在更加复杂。但我们最终得到了一个好结果...至少有时候是这样。如果你多次运行它，你会发现在不同的运行中可以得到非常不同的结果。这是因为实际上我们在这里有一个非常深的网络，这可能导致非常大或非常小的梯度。我们将在本章的下一部分看到如何处理这个问题。

现在，获得更好模型的明显方法是加深：在我们基本的 RNN 中，隐藏状态和输出激活之间只有一个线性层，所以也许我们用更多的线性层会得到更好的结果。

# 多层 RNNs

在多层 RNN 中，我们将来自我们递归神经网络的激活传递到第二个递归神经网络中，就像图 12-6 中所示。

![2 层 RNN](img/dlcf_1206.png)

###### 图 12-6\. 2 层 RNN

展开的表示在图 12-7 中显示（类似于图 12-3）。

![2 层展开的 RNN](img/dlcf_1207.png)

###### 图 12-7\. 2 层展开的 RNN

让我们看看如何在实践中实现这一点。

## 模型

我们可以通过使用 PyTorch 的`RNN`类来节省一些时间，该类实现了我们之前创建的内容，但也给了我们堆叠多个 RNN 的选项，正如我们之前讨论的那样：

```py
class LMModel5(Module):
    def __init__(self, vocab_sz, n_hidden, n_layers):
        self.i_h = nn.Embedding(vocab_sz, n_hidden)
        self.rnn = nn.RNN(n_hidden, n_hidden, n_layers, batch_first=True)
        self.h_o = nn.Linear(n_hidden, vocab_sz)
        self.h = torch.zeros(n_layers, bs, n_hidden)

    def forward(self, x):
        res,h = self.rnn(self.i_h(x), self.h)
        self.h = h.detach()
        return self.h_o(res)

    def reset(self): self.h.zero_()
```

```py
learn = Learner(dls, LMModel5(len(vocab), 64, 2),
                loss_func=CrossEntropyLossFlat(),
                metrics=accuracy, cbs=ModelResetter)
learn.fit_one_cycle(15, 3e-3)
```

| epoch | train_loss | valid_loss | accuracy | time |
| --- | --- | --- | --- | --- |
| 0 | 3.055853 | 2.591640 | 0.437907 | 00:01 |
| 1 | 2.162359 | 1.787310 | 0.471598 | 00:01 |
| 2 | 1.710663 | 1.941807 | 0.321777 | 00:01 |
| 3 | 1.520783 | 1.999726 | 0.312012 | 00:01 |
| 4 | 1.330846 | 2.012902 | 0.413249 | 00:01 |
| 5 | 1.163297 | 1.896192 | 0.450684 | 00:01 |
| 6 | 1.033813 | 2.005209 | 0.434814 | 00:01 |
| 7 | 0.919090 | 2.047083 | 0.456706 | 00:01 |
| 8 | 0.822939 | 2.068031 | 0.468831 | 00:01 |
| 9 | 0.750180 | 2.136064 | 0.475098 | 00:01 |
| 10 | 0.695120 | 2.139140 | 0.485433 | 00:01 |
| 11 | 0.655752 | 2.155081 | 0.493652 | 00:01 |
| 12 | 0.629650 | 2.162583 | 0.498535 | 00:01 |
| 13 | 0.613583 | 2.171649 | 0.491048 | 00:01 |
| 14 | 0.604309 | 2.180355 | 0.487874 | 00:01 |

现在这令人失望...我们之前的单层 RNN 表现更好。为什么？原因是我们有一个更深的模型，导致激活爆炸或消失。

## 激活爆炸或消失

在实践中，从这种类型的 RNN 创建准确的模型是困难的。如果我们调用`detach`的频率较少，并且有更多的层，我们将获得更好的结果 - 这使得我们的 RNN 有更长的时间跨度来学习和创建更丰富的特征。但这也意味着我们有一个更深的模型要训练。深度学习发展中的关键挑战是如何训练这种类型的模型。

这是具有挑战性的，因为当您多次乘以一个矩阵时会发生什么。想想当您多次乘以一个数字时会发生什么。例如，如果您从 1 开始乘以 2，您会得到序列 1、2、4、8，...在 32 步之后，您已经达到 4,294,967,296。如果您乘以 0.5，类似的问题会发生：您会得到 0.5、0.25、0.125，...在 32 步之后，它是 0.00000000023。正如您所看到的，即使是比 1 稍高或稍低的数字，经过几次重复乘法后，我们的起始数字就会爆炸或消失。

因为矩阵乘法只是将数字相乘并将它们相加，重复矩阵乘法会发生完全相同的事情。这就是深度神经网络的全部内容 - 每一层都是另一个矩阵乘法。这意味着深度神经网络很容易最终得到极大或极小的数字。

这是一个问题，因为计算机存储数字的方式（称为*浮点数*）意味着随着数字远离零点，它们变得越来越不准确。来自优秀文章“关于浮点数你从未想知道但却被迫了解”的图 12-8 中的图表显示了浮点数的精度如何随着数字线变化。
