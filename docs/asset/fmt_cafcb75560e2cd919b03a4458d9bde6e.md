
# 探索单词频率

```py
word_freq = word_frequency(training_sentences, word_index)
print(word_freq)
```

你将看到这样的结果：一个包含每个单词频率的字典，从最常用的单词开始，然后继续下去。以下是一些单词：

```py
{'new': 1318, 'trump': 1117, 'man': 1075, 'not': 634, 'just': 501, 
 'will': 484, 'one': 469, 'year': 440, …
```

如果你想绘制这个图表，你可以遍历列表中的每个项目，并将*x*值设置为你的位置序号（第一个项目为 1，第二个项目为 2，以此类推）。然后*y*值将是`newlist[item]`，你可以使用`matplotlib`来绘制。以下是代码：

```py
import matplotlib.pyplot as plt
from collections import OrderedDict
newlist = (OrderedDict(sorted(word_freq.items(), key=lambda t: t[1], 
                       reverse=True)))

xs=[]
ys=[]
curr_x = 1
for item in newlist:
  xs.append(curr_x)
  curr_x=curr_x+1
  ys.append(newlist[item])

print(ys)
plt.plot(xs,ys)
```

结果显示在图 6-6 中。

![图 6-6](img/aiml_0606.png)

###### 图 6-6\. 探索单词频率

这条“曲棍球棒”曲线告诉我们，很少的单词被多次使用，而大多数单词被很少使用。但每个单词都被有效地同等加权，因为每个单词都有一个“条目”在嵌入中。鉴于我们的训练集相对于验证集来说相对较大，我们最终处于一个情况，即训练集中有许多单词在验证集中不存在。

你可以通过更改在调用`plt.show`之前图表的轴来放大数据。例如，要查看*x*轴上 300 到 10,000 的单词量，以及*y*轴上 0 到 100 的刻度，你可以使用以下代码：

```py
plt.plot(xs,ys)
plt.axis([300,10000,0,100])
plt.show()
```

结果显示在图 6-7 中。

![单词频率 300–10,000](img/aiml_0607.png)

###### 图 6-7\. 300 到 10,000 的单词频率

语料库中大约有 25,000 个单词，代码被设置为对所有这些单词进行训练！但如果我们查看位置 2,000 及以后的单词，这超过了我们词汇量的 90%，我们会发现它们在整个语料库中每个单词的使用次数都不到 20 次！

这可能解释了过拟合，所以合理的下一步是看看我们是否可以减少我们正在训练的词汇量。在`build_vocab`辅助函数中，我们可以添加一个参数来指定我们感兴趣的词汇量的最大值，如下所示：

```py
def build_vocab(sentences, max_vocab_size=10000):
    counter = Counter()
    for text in sentences:
        counter.update(tokenize(text))

# Take only the top max_vocab_size-1 most frequent words 
# (leave room for special tokens)
    most_common = counter.most_common(max_vocab_size – 2)  
    # -2 for <pad> and <unk>

    # Create vocabulary with indices starting from 2
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
    vocab['<pad>'] = 0  # Add padding token
    vocab['<unk>'] = 1  # Add unknown token
    return vocab
```

然后，在构建我们的`word_index`时，我们可以指定一个我们感兴趣探索的最大词汇量：

```py
vocab_size = 2000
word_index = build_vocab(training_sentences, max_vocab_size=vocab_size)
```

嵌入层已经根据词汇量初始化，因此模型架构不需要改变。实际上，随着词汇量的减少，学习到的参数数量急剧下降，给我们提供了一个更简单的网络，它学习得更快：

```py

==========================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================
TextClassificationModel                  [32, 1]                   --
├─Embedding: 1-1                         [32, 100, 100]            200,100
├─AdaptiveAvgPool1d: 1-2                 [32, 100, 1]              --
├─Linear: 1-3                            [32, 24]                  2,424
├─ReLU: 1-4                              [32, 24]                  --
├─Linear: 1-5                            [32, 1]                   25
├─Sigmoid: 1-6                           [32, 1]                   --
==========================================================================
Total params: 202,549
Trainable params: 202,549
Non-trainable params: 0
Total mult-adds (M): 6.48
==========================================================================
Input size (MB): 0.03
Forward/backward pass size (MB): 2.57
Params size (MB): 0.81
Estimated Total Size (MB): 3.40
==========================================================================
```

模型从 240 万个参数缩减到只有 202,549 个。

在重新训练和探索较小的模型后，我们可以看到结果已经改变。

图 6-8 显示了准确率指标。现在，训练集的准确率大约是 82%，验证集的准确率大约是 76%。它们彼此更接近，没有发散，这是一个好兆头，表明我们已经消除了大部分过拟合。

![图 6-8](img/aiml_0608.png)

###### 图 6-8。两千个单词词汇量的准确率

这在一定程度上得到了图 6-9 中的损失图的加强。验证集上的损失在上升，但比之前慢得多，所以减少词汇量以防止训练集对可能在训练集中仅出现一次的低频单词过拟合似乎已经奏效。

值得尝试不同的词汇量大小，但请记住，词汇量太小也可能导致过拟合。你需要找到一个平衡点。在这种情况下，我选择出现 20 次或以上的单词纯粹是随意的。

![图 6-9](img/aiml_0609.png)

###### 图 6-9。两千个单词词汇量的损失

## 探索嵌入维度

对于这个例子，我随意选择了 16 维的嵌入维度。在这种情况下，单词被编码为 16 维空间中的向量，其方向表示它们的整体意义。但 16 是一个好数字吗？在我们的词汇量中只有两千个单词的情况下，它可能有点高，导致方向的高稀疏度。

###### 注意

我认为思考稀疏性的最佳方式是将它们投影到三维空间中。想象一下地球，有一千个向量从核心指向地表上的一个点。这些向量在三维空间中，*x*，*y*，和*z*。它们覆盖的表面积很大，但如果许多向量缺少*x*和*y*，这意味着它们只是零，那么许多向量将指向(0, 0, *z*)，而地球表面的大部分将保持未触及！因此，将缺乏独特性。

研究表明，嵌入大小的最佳实践是使其成为词汇大小的四次方根。2000 的四次方根是 6.687，所以让我们探索如果我们将其四舍五入并将嵌入维度更改为 7 会发生什么。

你可以在图 6-10 中看到训练 100 个 epoch 的结果。训练集的准确率稳定在大约 83%，验证集的准确率稳定在大约 77%。尽管有一些波动，但线条相当平稳，表明模型已经收敛。这与图 6-8 中的结果没有太大不同，但减少嵌入维度使得模型训练速度显著加快。

![图 6-10](img/aiml_0610.png)

###### 图 6-10\. 七维度的训练与验证准确率

图 6-11 显示了训练和验证中的损失。虽然最初看起来损失在约 20 个 epoch 时上升，但它很快平缓下来。再次，这是一个好兆头！

![图 6-11](img/aiml_0611.png)

###### 图 6-11\. 七维度的训练与验证损失

现在维度已经减少，我们可以对模型架构进行一些更多的调整。

## 探索模型架构

在前几节的优化之后，模型架构看起来是这样的：

```py
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim=24):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
```

想到的一件事是维度——`GlobalAveragePooling1D`层现在只发出 7 个维度，但它们被输入到 24 个神经元的隐藏层中，这是过度的。让我们看看当这减少到 8 个神经元并训练 100 个 epoch 时会发生什么。

你可以在图 6-12 中看到训练与验证的准确率。与使用 24 个神经元的图 6-7 相比，整体结果相当相似，但模型的训练速度有所提高。

![图 6-12](img/aiml_0612.png)

###### 图 6-12\. 减少密集架构准确率结果

图 6-13 中的损失曲线显示了类似的结果。

通过遵循这些练习，我们能够显著减少模型架构，在提高质量和减轻过拟合的同时减少参数数量。但还有一些事情我们可以做——从 dropout 开始。

![图 6-13](img/aiml_0613.png)

###### 图 6-13\. 减少密集架构损失结果

## 使用 dropout

减少过拟合的常见技术是在密集神经网络中添加 dropout。我们早在第三章中探讨了卷积神经网络，所以在这里直接探讨它对过拟合的影响似乎很吸引人。但在这个案例中，我想等到词汇量、嵌入大小和架构复杂性都得到解决。这些变化往往比使用 dropout 有更大的影响，而且我们已经看到了一些很好的结果。

现在我们已经将架构简化到中间密集层只有八个神经元，dropout 的影响可能已经被最小化——但让我们还是探索一下。以下是更新后的模型架构代码，添加了 0.25 的 dropout（相当于八个神经元中的两个）：

```py
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim=8, 
                       dropout_rate=0.25):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)  # Add dropout layer
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)  # Change for pooling layer
        x = self.global_pool(x).squeeze(2)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x
```

图 6-14 显示了训练 100 个 epoch 时的准确率结果。这次，我们看到训练准确率和验证准确率正在收敛，训练准确率现在低于之前。同样，图 6-15 中的损失曲线也显示了收敛，所以虽然 dropout 使我们的网络略微*降低*了准确率，但它似乎更好地泛化了。

但在宣布胜利之前，请务必谨慎！仔细检查曲线显示，损失已经很好地收敛，但它们*确实*比之前更高。使用 dropout 时，训练损失高于 0.5，而没有 dropout 时大约是 0.3。它也呈下降趋势，所以值得实验看看更长的训练时间是否能产生更好的结果。

![图 6-14](img/aiml_0614.png)

###### 图 6-14\. 添加 dropout 后的准确率

![图 6-15](img/aiml_0615.png)

###### 图 6-15\. 添加 dropout 后的损失

你还可以看到，模型正在回归到之前随时间增加验证损失的模式。这并不像之前那么糟糕，但它正朝着错误的方向发展。

在这种情况下，当神经元非常少时，引入 dropout 可能并不是一个好的主意。尽管如此，这个工具仍然值得保留在你的工具箱中，所以请确保记住它，用于比这个更复杂的架构。

## 使用正则化

*正则化*是一种通过减少权重的极化来帮助防止过拟合的技术。如果某些神经元的权重太重，正则化会有效地惩罚它们。广义而言，有两种类型的正则化：

L1 正则化

这通常被称为*最小绝对收缩*和*选择算子*（lasso）正则化。它有效地帮助我们计算层中的结果时忽略零或接近零的权重。

L2 正则化

这通常被称为*岭回归*，因为它通过取它们的平方来推动值分开。这往往放大了非零值与零或接近零值之间的差异，从而产生脊效应。

这两种方法也可以结合成有时被称为*弹性*正则化的方法。

对于我们正在考虑的 NLP 问题，L2 正则化最常使用。它可以作为`optimizer`的`weight_decay`属性添加。以下是一个示例：

```py
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), 
                       amsgrad=False, weight_decay=0.01)
```

这将应用`weight_decay`为`0.01`。（通常，这里的值会在 0.01 和 0.001 之间）。另外，使用 PyTorch 你可以定义不同层的不同权重衰减，通过在`Adam`声明调用中指定它们，如下所示：

```py
# Different weight decay for different layers
optimizer = torch.optim.Adam([
# L2 reg on fc1
        {'params': model.fc1.parameters(), 'weight_decay': 0.01},      
    # No L2 reg on other layers
{'params': [p for name, p in model.named_parameters() 
            if 'fc1' not in name]}  
], lr=0.0001)
```

在这样一个简单模型中添加正则化的影响并不特别大，但它确实在一定程度上平滑了我们的训练损失和验证损失。对于这种情况来说可能有些过度，但就像 dropout 一样，了解如何使用正则化来防止模型过度专业化是一个好主意。

## 其他优化考虑因素
