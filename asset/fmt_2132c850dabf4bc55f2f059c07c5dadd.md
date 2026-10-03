
安装完包后，我们可以下载数据集并定义一个分词器。Torchtext 通过`torchtext.datasets`和`torchtext.data.utils`子模块提供了许多自然语言处理（NLP）数据集和分词器。我们将使用内置的 IMDb 数据集和 PyTorch 提供的标准`'basic_english'`分词器。

```py
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer

# Load dataset
train_iter = IMDB(split=('train'))

# Define tokenizer and build vocabulary
tokenizer = get_tokenizer('basic_english')
```

到目前为止，我们一直在使用 PyTorch 的映射式数据集。Torchtext 将 NLP 数据集返回为可迭代式数据集，这对于流式数据更为合适。接下来，我们需要基于训练数据集创建一个词汇表，并修剪词汇表，只包括最常见的 30,000 个单词。然后，我们需要将每个输入序列填充到 500 个单词的长度，并处理标签。

```py
from torchtext.vocab import build_vocab_from_iterator

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# build vocab from iterator and add a list of any special tokens
text_vocab = build_vocab_from_iterator(yield_tokens(train_iter),
                                       specials=['<unk>', '<pad>'])
text_vocab.set_default_index(text_vocab['<unk>'])
```

如上所示，Torchtext 提供了一个函数`build_vocab_from_iterator`来创建词汇表。但是，该函数期望以标记列表作为输入，其中`next(train_iter)`会返回一个元组`(label_string, review_string)`。为了满足这一要求，我们定义了一个函数，在数据集被迭代时产生标记。最后，我们添加了未知和填充的特殊标记，并设置了默认值。

接下来，我们需要实际修剪词汇表并填充评论序列，以及将标签字符串`'neg'`或`'pos'`转换为数字。我们通过为标签和评论字符串定义一个流水线函数来实现这一点：

```py
def text_pipeline(x, max_size=512):
   text = tokenizer(x)

   # reduce vocab size
   pruned_text = []
   for token in text:
     if text_vocab.get_stoi()[token] >= 30000:
       token = '<unk>'
     pruned_text.append(token)

   # pad sequence or truncate
   if len(pruned_text) <= max_size:
     pruned_text += ['<pad>'] * (max_size - len(pruned_text))
   else:
     pruned_text = pruned_text[0:max_size]
   return text_vocab(pruned_text)

label_pipeline = lambda x: (0 if (x == 'neg') else 1)
```

`text_pipeline`函数将输入转换为 500 维向量。每个向量对应一个电影评论，其中向量的第 i^th 个分量对应于评论中第 i^th 个单词在我们的全局词典中的索引，该词典包含 30,000 个单词。为了完成数据准备，我们创建了一个特殊的 Python 类，用于从底层数据集中提供所需大小的小批量数据。

我们可以使用 PyTorch 中内置的`DataLoader`类来对数据集进行批处理采样。在这样做之前，我们需要定义一个函数`collate_batch`，告诉`DataLoader`如何预处理每个批次：

```py
def collate_batch(batch):
  label_list, text_list = [], []
  for label, review in batch:
    label_list.append(label_pipeline(label))
    text_list.append(text_pipeline(review))
  return (torch.tensor(label_list, dtype=torch.long),
          torch.tensor(text_list, dtype=torch.int32))
```

`collate_batch`函数只是通过每个相应的管道运行标签和评论字符串，并将批处理作为张量元组`(labels_batch, reviews_batch)`返回。一旦定义了`collate_fn`，我们只需使用 IMDb 构造函数加载数据集，并使用`DataLoader`构造函数配置数据加载器：

```py
from torch.utils.data import DataLoader

train_iter, val_iter = IMDB(split=('train','test'))
trainloader = DataLoader(train_iter,
                         batch_size = 4,
                         shuffle=False,
                         collate_fn=collate_batch)
valloader = DataLoader(val_iter,
                       batch_size = 4,
                       shuffle=False,
                       collate_fn=collate_batch)
```

我们使用`torchtext.datasets.IMDB` Python 类来为我们在训练情感分析模型时使用的训练和验证集提供服务。

现在数据已经准备就绪，我们将逐步构建情感分析模型。首先，我们将希望将输入评论中的每个单词映射到一个单词向量。为此，我们将利用一个嵌入层，正如您可能从第八章中回忆的那样，这是一个简单的查找表，存储与每个单词对应的嵌入向量。

与以前的例子不同，我们将学习词嵌入作为一个单独的问题（即通过构建 Skip-Gram 模型），我们将通过将嵌入矩阵视为完整问题中的参数矩阵来同时学习词嵌入和情感分析问题。我们通过使用 PyTorch 原语来管理嵌入来实现这一点（请记住，`input`代表一次完整的小批量，而不仅仅是一个电影评论向量）：

```py
import torch.nn as nn

embedding = nn.Embedding(
                      num_embeddings=30000,
                      embedding_dim=512,
                      padding_idx=text_vocab.get_stoi()['<pad>'])
```

然后，我们将嵌入层的结果传递给使用我们在前一节中看到的原语构建带有丢失的 LSTM。LSTM 的实现可以如下实现：

```py
class TextClassifier(nn.Module):
  def __init__(self):
    super(TextClassifier,self).__init__()
    self.layer_1 = nn.Embedding(
                      num_embeddings=30000,
                      embedding_dim=512,
                      padding_idx=1)                      
    self.layer_2 = nn.LSTMCell(input_size=512, hidden_size=512)
    self.layer_3 = nn.Dropout(p=0.5)
    self.layer_4 = nn.Sequential(
                      nn.Linear(512, 2),
                      nn.Sigmoid(),
                      nn.BatchNorm1d(2))

  def forward(self, x):
    x = self.layer_1(x)
    x = x.permute(1,0,2)
    h = torch.rand(x.shape[1], 512)
    c = torch.rand(x.shape[1], 512)
    for t in range(x.shape[0]):
      h, c = self.layer_2(x[t], (h,c))
      h = self.layer_3(h)
    return self.layer_4(h)
```

我们最后使用一个批归一化的隐藏层，与我们在以前的例子中一次又一次使用的隐藏层相同。将所有这些组件串联在一起，我们可以通过调用`TextClassifier`来构建模型：

```py
model = TextClassifier()
```

我们省略了设置摘要统计信息、保存中间快照和创建会话所涉及的其他样板，因为它与本书中构建的其他模型相同（请参阅[GitHub 存储库](https://github.com/darksigma/Fundamentals-of-Deep-Learning-Book)）。然后，我们可以使用 TensorBoard 运行和可视化我们模型的性能（参见图 9-23）。

![](img/fdl2_0923.png)

###### 图 9-23。我们电影评论情感模型的训练成本、验证成本和准确率

在训练开始时，模型稍微稳定性不佳，而在训练结束时，模型明显开始过拟合，训练成本和验证成本明显分歧。然而，在其最佳性能时，模型表现相当有效，并且在测试集上大约达到 86%的准确率。恭喜！您已经构建了您的第一个 RNN。

## 使用循环神经网络解决 seq2seq 任务

现在我们已经建立了对 RNN 的深入理解，我们准备重新审视 seq2seq 问题。我们在本章开始时以一个 seq2seq 任务的示例开始：将句子中的单词序列映射到 POS 标签序列。解决这个问题是可行的，因为我们不需要考虑长期依赖性来生成适当的标签。但是有几个 seq2seq 问题，例如语言之间的翻译或为视频创建摘要，其中长期依赖性对模型的成功至关重要。这就是 RNN 的用武之地。

seq2seq 的 RNN 方法看起来很像我们在上一章中讨论的自动编码器。seq2seq 模型由两个独立的网络组成。第一个网络称为*编码器*网络。编码器网络是一个循环网络（通常使用 LSTM 单元），它消耗整个输入序列。编码器网络的目标是生成对输入的简洁理解，并将其总结为由编码器网络的最终状态表示的一个单一思想。然后我们使用一个*解码器*网络，其起始状态由编码器网络的最终状态初始化，逐个标记地生成目标输出序列。在每一步中，解码器网络将其上一个时间步的输出作为当前时间步的输入。整个过程在图 9-24 中可视化。

![](Images/fdl2_0924.png)

###### 图 9-24。我们如何使用编码器/解码器循环网络架构来解决 seq2seq 问题

在这个设置中，我们试图将一句英语句子翻译成法语。我们对输入句子进行标记，并使用一个嵌入（类似于我们在前一节构建的情感分析模型中的方法），逐个单词作为编码器网络的输入。在句子结束时，我们使用一个特殊的“序列结束”（EOS）标记来指示输入序列的结束。然后我们取编码器网络的隐藏状态，并将其用作解码器网络的初始化。解码器网络的第一个输入是 EOS 标记，输出被解释为预测的法语翻译的第一个单词。从那时起，我们使用解码器网络的输出作为下一个时间步的输入。直到解码器网络发出 EOS 标记作为其输出，此时我们知道网络已经完成了对原始英语句子的翻译。我们将在本章后面剖析这个网络的实际开源实现（通过一些增强和技巧来提高准确性）。

seq2seq RNN 架构也可以重新用于学习序列的良好嵌入。例如，2015 年 Kiros 等人发明了*skip-thought 向量*的概念，借鉴了自动编码器框架和第八章中讨论的 Skip-Gram 模型的结构特征。skip-thought 向量通过将一段话分成一组由连续句子组成的三元组来生成。作者使用了一个编码器网络和两个解码器网络，如图 9-25 所示。

![](Images/fdl2_0925.png)

###### 图 9-25。skip-thought seq2seq 架构用于生成整个句子的嵌入表示

编码器网络消耗了我们想要生成简洁表示的句子（存储在编码器网络的最终隐藏状态中）。然后是解码步骤。第一个解码器网络将以该表示作为其自己隐藏状态的初始化，并尝试重建出出现在输入句子之前的句子。第二个解码器网络将尝试出现在输入句子之后的句子。整个系统在这些三元组上端到端地进行训练，一旦完成，就可以用来生成看似连贯的文本段落，同时提高关键句子级分类任务的性能。

以下是从原始论文中摘录的故事生成示例：

```py
she grabbed my hand .
"come on . "
she fluttered her back in the air .
"i think we're at your place . I ca n't come get you . "
he locked himself back up
" no . she will . "
kyrian shook his head
```

现在我们已经了解了如何利用 RNN 来解决 seq2seq 问题，我们几乎准备好尝试构建自己的模型了。然而，在那之前，我们还有一个重要的挑战要解决，在下一节中我们将直面讨论 seq2seq RNN 中的注意力概念。

## 用注意力增强循环网络

让我们更深入地思考翻译问题。如果你曾尝试学习一门外语，你会知道在完成翻译时有几个有用的步骤。首先，阅读完整的句子以理解你想要传达的概念是有帮助的。然后，逐字逐句地写出翻译，每个词都逻辑地跟在你之前写的词后面。但翻译的一个重要方面是，当你构成新句子时，你经常会参考原始文本，关注与当前翻译相关的特定部分。在每一步中，你都在关注原始“输入”的最相关部分，以便能够做出关于下一个要写在页面上的词的最佳决定。

回想一下我们对 seq2seq 的方法。通过消耗完整的输入并将其总结为“思考”隐藏在其隐藏状态中，编码器网络有效地实现了翻译过程的第一部分。通过使用先前的输出作为当前输入，解码器网络实现了翻译过程的第二部分。然而，我们的 seq2seq 方法尚未捕捉到*注意力*这一现象，这是我们需要工程化的最后一个构建块。

当前，在给定时间步骤*t*，解码器网络的唯一输入是在时间步骤<t-1>的输出。给解码器网络一些关于原始句子的视觉信息的一种方法是让解码器访问编码器网络的所有输出（我们之前完全忽略了）。这些输出对我们很有趣，因为它们代表编码器网络在看到每个新标记后内部状态的演变。这种策略的一个提议实现如图 9-26 所示。这个尝试失败了，因为它未能动态选择要关注的输入的最相关部分。

![](img/fdl2_0926.png)

###### 图 9-26。在 seq2seq 架构中尝试工程化注意力能力

然而，这种方法存在一个关键缺陷。问题在于，在每个时间步骤，解码器以完全相同的方式考虑编码器网络的所有输出。然而，在翻译过程中，人类显然不是这样做的。在翻译不同部分时，我们会关注原始文本的不同方面。关键的认识在于，仅仅让解码器访问所有输出是不够的。相反，我们必须设计一种机制，使解码器网络能够动态关注编码器输出的特定子集。

我们可以通过改变连接操作的输入来解决这个问题，使用 Bahdanau 等人 2015 年的提议作为灵感。而不是直接使用编码器网络的原始输出，我们对编码器的输出进行加权操作。我们利用解码器网络在时间<t-1>的状态作为加权操作的基础。

权重操作在图 9-27 中进行了说明。首先，我们为编码器的每个输出创建一个标量（一个单一的数字，而不是张量）相关性分数。该分数是通过计算每个编码器输出与时间<t 减 1>的解码器状态之间的点积来生成的。然后，我们使用 softmax 操作对这些分数进行归一化。最后，我们使用这些归一化的分数来分别缩放编码器的输出，然后将它们插入连接操作中。关键在于，为每个编码器输出计算的相对分数表示该特定编码器输出对于解码器在时间步*t*上的决策有多重要。事实上，正如我们将在后面看到的，我们可以通过检查 softmax 的输出来可视化哪些输入部分对于每个时间步的翻译最为相关。

![](img/fdl2_0927.png)

###### 图 9-27. 根据上一个时间步的解码器网络的隐藏状态实现的动态注意力机制的我们原始提议的修改

掌握了将注意力引入 seq2seq 架构的策略后，我们终于准备好使用 RNN 模型将英语句子翻译成法语。但在我们开始之前，值得注意的是，注意力在超越语言翻译的问题中非常适用。在语音转文本问题中，注意力可以帮助算法动态关注音频的相应部分，同时将音频转录为文本。同样，注意力可以用于改进图像字幕算法，帮助字幕算法在撰写字幕时专注于输入图像的特定部分。每当输入的特定部分与正确生成相应输出的部分高度相关时，注意力都可以显著提高性能。

## 剖析神经翻译网络
