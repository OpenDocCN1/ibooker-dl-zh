
虽然我们做出的修改已经使我们得到了一个拟合度更低的改进模型，但还有其他超参数你可以进行实验。例如，我们选择将最大句子长度设为 100 个单词，但这完全是随意的，可能不是最优的。探索语料库并查看更好的句子长度可能是一个好主意。以下是一段代码，它查看句子并按从低到高的顺序绘制每个句子的长度：

```py
xs=[]
ys=[]
current_item=1
for item in sentences:
  xs.append(current_item)
  current_item=current_item+1
  ys.append(len(item))
newys = sorted(ys)

import matplotlib.pyplot as plt
plt.plot(xs,newys)
plt.show()
```

见图 6-16 以了解此结果。

![图片](img/aiml_0616.png)

###### 图 6-16\. 探索句子长度

在 26,000+的总语料库中，少于 200 个句子的长度超过 100 个单词，因此选择这个作为最大长度，我们引入了大量的不必要的填充，从而影响了模型的表现。将最大长度减少到 85 个单词仍然可以保留 26,000 个句子（99%以上），同时大幅减少填充。

## 整合所有内容

将所有前面的优化措施生效，并重新训练模型三百个 epoch，你可以在图 6-17 中看到训练和验证准确率的结果。鉴于它们的曲线大致匹配，这表明我们已朝着避免过拟合迈出了巨大的步伐，并且我们有一个正在有效学习的网络。

同样，训练和验证损失曲线在三百个 epoch 中显示出显著相似性，如图 6-18 所示，这表明优化是防止模型过拟合的正确方向。

![图片](img/aiml_0617.png)

###### 图 6-17\. 优化后的训练和验证准确率

![图片](img/aiml_0618.png)

###### 图 6-18\. 优化后的训练和验证损失

## 使用模型对句子进行分类

现在你已经创建了模型，训练了它，并优化了它以消除导致过拟合的许多问题，下一步就是运行模型并检查其结果。为此，你需要创建一个新句子的数组。例如，考虑以下内容：

```py
test_sentences = [
             "granny starting to fear spiders in the garden might be real", 
             "game of thrones season finale showing this sunday night", 
             "PyTorch book will be a best seller"]
```

你可以使用创建词汇表时使用的相同分词器来对这些序列进行编码：

```py
print(texts_to_sequences(test_sentences, word_index))

```

使用这个分词器很重要，因为它包含了网络训练时使用的单词的标记！

打印语句的输出将是前面句子的序列：

```py
[
[1, 803, 753, 1, 1, 312, 97], 
[123, 1183, 160, 1, 1, 1543, 152], 
[1, 235, 7, 47, 1]
]
```

这里有很多`1`标记（“<OOV>”），因为像*奶奶*和*蜘蛛*这样的词没有出现在字典中。由于停用词已被删除，序列也变短了。

接下来，在你可以将序列传递给模型之前，你需要将它们放入模型期望的形状——即期望的长度。你可以用与训练模型时相同的方式使用`pad_sequences`来完成这个操作：

```py
padded = pad_sequences(sequences, max_len)
```

这将输出长度为`85`的句子序列，因此第一个序列的输出如下：

```py
[1, 803, 753, 1, 1, 312, 97, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0]
```

这是一个非常短的句子，所以它被填充到了 85 个字符，其中包含了很多零！

现在你已经填充并分词了句子，使其符合模型对输入维度的期望，现在是时候将它们传递给模型并获取预测结果了。

这涉及多个步骤。首先，将填充的序列转换为输入张量：

```py
# Convert to tensor
input_ids = torch.tensor(padded, dtype=torch.long).to(device)
```

接下来，将模型置于评估模式以获取预测，然后只需将`input_ids`传递给它以获取输出：

```py
# Get predictions
model.eval()
with torch.no_grad():
    outputs = model(input_ids)
```

结果将以列表的形式返回并打印出来，高值表示可能存在讽刺。以下是我们的样本句子的结果：

```py
tensor([[0.5516],
        [0.0765],
        [0.0987]], device='cuda:0')
```

尽管第一个句子（“奶奶开始害怕花园里的蜘蛛可能是真的”）有很多停用词，并且被填充了很多零，但它的得分很高，这表明其中存在一定程度的讽刺。其他两个句子的得分要低得多，表明它们中讽刺的可能性较低。

要获取概率，你可以调用`squeeze()`方法来检索张量值。如果你想将阈值与比较以获得预测——例如，高于 0.5 表示讽刺，低于 0.5 表示没有讽刺——则可以使用如下代码：

```py
probabilities = outputs.squeeze().cpu().numpy()
predictions = (probabilities >= threshold).astype(int)
```

根据你的网络调整，你也可以确定你认为适当的阈值应该是什么。以 0.5 的阈值运行，我们得到以下结果：

```py
Text: granny starting to fear spiders in the garden might be real
Probability: 0.5516
Classification: `Sarcastic`
Confidence: 0.5516
--------------------------------------------------------------------------

Text: game of thrones season finale showing this sunday night
Probability: 0.0765
Classification: Not Sarcastic
Confidence: 0.9235
--------------------------------------------------------------------------

Text: PyTorch book will be a best seller
Probability: 0.0987
Classification: `Not` `Sarcastic`
Confidence: 0.9013
--------------------------------------------------------------------------
```

因此，通过这些测试句子，我们开始得到一个很好的迹象，表明我们的网络正在按预期运行。你应该用其他数据来测试它，看看你是否能将其破坏，如果你能持续地破坏它，那么就是时候尝试不同的模型架构，从现有的工作网络中利用迁移学习，或者探索使用预训练的嵌入体了。

我们将在下一节中了解这一点，但在那之前，我想向你展示你可以如何可视化这个网络学习到的自定义嵌入。

# 可视化嵌入

要可视化嵌入，你可以使用一个名为[嵌入投影仪](http://projector.tensorflow.org)的在线工具。它预先加载了许多现有数据集，但在本节中，你将看到如何使用这个工具通过模型训练的数据来可视化。

但首先，你需要一个函数来反转单词索引。它目前将单词作为标记，将键作为值，但你需要将其反转，以便你可以在投影仪上绘制单词值。以下是执行此操作的代码：

```py
reverse_word_index = dict([(value, key)
for (key, value) in word_index.items()])

```

你还需要提取嵌入中的向量权重：

```py
embedding_weights = model.embedding.weight.data.cpu().numpy()
print(embedding_weights.shape)

```

如果你遵循了本章中的优化，这个输出的结果将是`(2000,7)`，因为我们使用了 2,000 个单词的词汇量和 7 个维度的嵌入。如果你想探索一个单词及其向量细节，你可以使用如下代码：

```py
print(reverse_word_index[2])
print(embedding_weights[2])

```

这将产生以下输出：

```py
new
[–0.27116913 –1.3026129   1.6390767   0.4922502  –0.6025921   1.4584142
  0.05054485]

```

因此，单词“new”由一个在轴上有那些七个系数的向量表示。

嵌入投影仪使用两个制表符分隔的值（TSV）文件，一个用于向量维度，一个用于元数据。此代码将为你生成它们：

```py
import io
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = embedding_weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

```

或者，如果你使用的是 Google Colab，你可以使用以下代码或从文件面板下载 TSV 文件：

```py
try:
  from google.colab import files
except ImportError:
  pass
else:
  files.download('vecs.tsv')
  files.download('meta.tsv')

```

一旦你有这些文件，你可以在投影仪上按下“加载”按钮来可视化嵌入（见图 6-19）。

![](img/aiml_0619.png)

###### 图 6-19\. 使用嵌入投影仪

你还可以在结果对话框中推荐的位置使用向量和元 TSV 文件，然后点击投影仪上的“球化数据”。这将导致单词在球体中聚集，并将为你提供这个分类器二进制性质的清晰可视化。它只训练了讽刺和非讽刺句子，所以单词倾向于聚集在某个标签附近（见图 6-20）。

![](img/aiml_0620.png)

###### 图 6-20\. 可视化讽刺嵌入

屏幕截图并不能完全展示这些内容——你应该亲自尝试！你可以旋转中心球体，探索每个“极点”上的单词，看看它们对整体分类的影响，你还可以在右侧面板中选择单词并显示相关的单词。玩一玩，实验一下！

# 使用预训练嵌入

使用自己训练的嵌入的一个替代方案是使用别人代为预训练的嵌入。你可以在许多来源找到这些嵌入，包括 Kaggle 和 Hugging Face。你甚至可以在研究结果的旁边找到预训练的嵌入。这样一组预训练的嵌入是[斯坦福 GloVe 嵌入](https://oreil.ly/s1YWw)，我们将在下面探讨这些内容。

注意，然而，当使用预训练的嵌入时，你也应该考虑更新和更改你的分词器，以匹配预训练嵌入所使用的任何规则。

例如，对于 GloVE 预训练嵌入——它仅仅是一个包含单词及其在 50 到 300 个维度上预训练嵌入的大型文本文件——用于分词单词的规则与我们所使用的原始数据的手工分词器略有不同。因此，对于 GloVe，你应该考虑所有单词都转换为小写或数字归一化到 0 这样的规则。

一旦你完成这个操作（我在下载中提供了 GloVe 的代码，并在下一章中对其进行了更详细的讨论），那么只需将预训练嵌入的权重加载到你的模型定义中即可，如下所示：

```py
# Initialize embedding layer
self.embedding = nn.Embedding(vocab_size, embedding_dim)

# Load pretrained embeddings if provided
if pretrained_embeddings is not None:
    self.embedding.weight.data.copy_(pretrained_embeddings)
    if freeze_embeddings:
        self.embedding.weight.requires_grad = False
```

如果你不想从这些嵌入中学习，只想使用它们，那么你应该将`freeze_embeddings`设置为`True`。否则，网络将使用预加载的嵌入权重作为起点进行微调。

这个模型在训练中会迅速达到峰值准确度，并且它不会像我们之前看到的那样过度拟合。经过三百个 epoch 的准确度显示，训练和验证非常一致（参见图 6-22）。损失值也是一致的，这表明我们在前几百个 epoch 中拟合得非常好。然而，它们也开始发散（参见图 6-22）。

另一方面，值得注意的是，整体准确度（大约 70%）相当低，考虑到抛硬币有 50%的机会猜对！因此，虽然使用预训练嵌入可以加快训练速度并减少过度拟合，但你应该了解它们有什么用，以及它们可能并不总是最适合你的场景。因此，你可能需要探索适当的优化方法或替代方案。

![图片](img/aiml_0621.png)

###### 图 6-21\. 使用 GloVe 嵌入的准确度指标

![图片](img/aiml_0622.png)

###### 图 6-22\. 使用 GloVe 嵌入的损失指标

# 摘要

在本章中，你构建了第一个能够理解文本情感的模型。它是通过从第五章获取分词文本并将其映射到向量来实现的。然后，使用反向传播，它根据包含该句子的标签学习每个向量的适当“方向”。最后，它能够使用一组单词的所有向量来构建句子中情感的概念。

你还探索了优化模型以避免过拟合的方法，并看到了表示你单词的最终向量的整洁可视化。但尽管这是一种很好的句子分类方法，它只是简单地将每个句子视为一堆单词。其中没有固有的序列，而在确定句子的真正含义时，单词出现的顺序非常重要。

因此，考虑如果我们能通过考虑序列来改进我们的模型，那将是个不错的想法。我们将在下一章中通过介绍一种新的层类型：一个*循环*层，这是循环神经网络的基础，来探讨这一点。
