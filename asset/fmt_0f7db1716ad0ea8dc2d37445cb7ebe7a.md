

## 第十四章 学习像莎士比亚一样写作：长短期记忆

**本章**

+ 字符语言模型
+ 截断反向传播
+ 梯度消失和梯度爆炸
+ RNN 反向传播的一个玩具示例
+ 长短期记忆（LSTM）单元

> “主啊，这些凡人多么愚蠢！”
> 
> *莎士比亚《仲夏夜之梦》*

### 字符语言模型

#### 让我们用 RNN 解决一个更具挑战性的任务

在第十二章和第十三章的结尾，你训练了简单的循环神经网络（RNN），这些网络学习了一个简单的序列预测问题。但你是在一个玩具数据集上训练的，这个数据集是通过规则合成的短语。

在本章中，你将尝试在一个更具挑战性的数据集上进行语言建模：莎士比亚的作品。而且，与上一章中学习根据前面的单词预测下一个单词不同，该模型将训练在字符上。它需要学习根据观察到的先前字符预测下一个字符。这就是我的意思：

```
import sys,random,math
from collections import Counter
import numpy as np
import sys

np.random.seed(0)

f = open('shakespear.txt','r')
raw = f.read()                  *1*
f.close()

vocab = list(set(raw))
word2index = {}
for i,word in enumerate(vocab):
    word2index[word]=i
indices = np.array(list(map(lambda x:word2index[x], raw)))
```

+ ***1* 来自 [`karpathy.github.io/2015/05/21/rnn-effectiveness/`](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)**

与第十二章和第十三章的词汇表由数据集中的单词组成不同，现在词汇表由数据集中的字符组成。因此，数据集也被转换为一个索引列表，这些索引对应于字符而不是单词。之上是`indices` NumPy 数组：

```
embed = Embedding(vocab_size=len(vocab),dim=512)
model = RNNCell(n_inputs=512, n_hidden=512, n_output=len(vocab))

criterion = CrossEntropyLoss()
optim = SGD(parameters=model.get_parameters() + embed.get_parameters(),
            alpha=0.05)
```

这段代码看起来都很熟悉。它初始化嵌入维度为 8，RNN 隐藏状态的大小为 512。输出权重初始化为 0（这不是规则，但我觉得这样效果更好）。最后，你初始化交叉熵损失和随机梯度下降优化器。

### 截断反向传播的需要

#### 通过 100,000 个字符进行反向传播是不可行的

阅读 RNN 代码的更具挑战性的方面之一是为输入数据而进行的批处理逻辑。之前的（更简单的）神经网络有一个这样的内部`for`循环（粗体部分）：

```
for iter in range(1000):
    batch_size = 100
    total_loss = 0

    hidden = model.init_hidden(batch_size=batch_size)

    for t in range(5):
        input = Tensor(data[0:batch_size,t], autograd=True)
        rnn_input = embed.forward(input=input)
        output, hidden = model.forward(input=rnn_input, hidden=hidden)

    target = Tensor(data[0:batch_size,t+1], autograd=True)
    loss = criterion.forward(output, target)
    loss.backward()
    optim.step()
    total_loss += loss.data
    if(iter % 200 == 0):
        p_correct = (target.data == np.argmax(output.data,axis=1)).mean()
        print_loss = total_loss / (len(data)/batch_size)
        print("Loss:",print_loss,"% Correct:",p_correct)
```

你可能会问，“为什么迭代到 5？”实际上，之前的语料库没有超过六个单词的例子。它读取了五个单词，然后尝试预测第六个。

更重要的是反向传播步骤。考虑当你对一个简单的前馈网络进行 MNIST 数字分类时：梯度总是反向传播到整个网络，对吧？它们一直反向传播，直到达到输入数据。这允许网络调整每个权重，试图学习如何根据整个输入示例正确预测。

这里的循环例子也没有不同。你通过五个输入示例进行前向传播，然后，当你稍后调用`loss.backward()`时，它将梯度反向传播回网络到输入数据点。你可以这样做，因为你一次没有输入那么多数据点。但是莎士比亚数据集有 10 万个字符！这对于每个预测进行反向传播来说太多了。你怎么办？

你不需要！你向后传播一个固定数量的步骤到过去，然后停止。这被称为*截断反向传播*，并且是行业标准。你向后传播的长度成为另一个可调参数（就像批量大小或 alpha）。

### 截断反向传播

#### 从技术上讲，它削弱了神经网络的最高理论极限

使用截断反向传播的缺点是它缩短了神经网络可以学习记住事物的距离。基本上，在比如说五个时间步长之后切断梯度，意味着神经网络无法学习记住过去超过五个时间步长的事件。

严格来说，情况比这更复杂。在 RNN 的隐藏层中，从过去超过五个时间步长可能会意外地保留一些残留信息，但神经网络不能使用梯度来特别请求模型从六个时间步长之前保留信息以帮助当前预测。因此，在实践中，神经网络不会学习基于过去超过五个时间步长的输入信号进行预测（如果截断设置为五个时间步长）。在实践中，对于语言建模，截断变量被称为`bptt`，它通常设置在 16 到 64 之间：

```
batch_size = 32
bptt = 16
n_batches = int((indices.shape[0] / (batch_size)))
```

截断反向传播的另一个缺点是它使得小批量逻辑变得稍微复杂一些。要使用截断反向传播，你假装你有一个大数据集，而不是大小为`bptt`的一堆小数据集。你需要相应地分组数据集：

```
trimmed_indices = indices[:n_batches*batch_size]
batched_indices = trimmed_indices.reshape(batch_size, n_batches)
batched_indices = batched_indices.transpose()

input_batched_indices = batched_indices[0:-1]
target_batched_indices = batched_indices[1:]

n_bptt = int(((n_batches-1) / bptt))
input_batches = input_batched_indices[:n_bptt*bptt]
input_batches = input_batches.reshape(n_bptt,bptt,batch_size)
target_batches = target_batched_indices[:n_bptt*bptt]
target_batches = target_batches.reshape(n_bptt, bptt, batch_size)
```

这里有很多事情在进行中。最上面一行使得数据集成为`batch_size`和`n_batches`之间的一个偶数倍。这样做是为了当你将其分组为张量时，它是平方的（或者你也可以用 0 填充数据集以使其成为平方）。第二行和第三行重新塑形数据集，使得每一列是初始`indices`数组的一个部分。我将展示这部分，就像`batch_size`被设置为 8（为了可读性）：

```
print(raw[0:5])
print(indices[0:5])
```

```
'That,'
array([ 9, 14, 2, 10, 57])
```

这些是莎士比亚数据集中的前五个字符。它们拼写出字符串“That，”。接下来是`batched_indices`中变换后的前五行的输出：

```
print(batched_indices[0:5])
```

```
array([[ 9, 43, 21, 10, 10, 23, 57, 46],
       [14, 44, 39, 21, 43, 14, 1, 10],
       [ 2, 41, 39, 54, 37, 21, 26, 57],
       [10, 39, 57, 48, 21, 54, 38, 43],
       [57, 39, 43, 1, 10, 21, 21, 33]])
```

我已经用粗体突出显示了第一列。看看短语“That，”的索引是否在左边的第一列？这是一个标准构造。有八个列的原因是`batch_size`是 8。这个张量随后被用来构建一个更小的数据集列表，每个数据集的长度为`bptt`。

你可以在这里看到输入和目标是如何构建的。注意，目标索引是输入索引偏移一行（因此网络预测下一个字符）。再次注意，在这个打印输出中`batch_size`设置为 8，这使得阅读更容易，但实际上你将其设置为 32。

```
print(input_batches[0][0:5])

print(target_batches[0][0:5])

array([[ 9, 43, 21, 10, 10, 23, 57, 46],
       [14, 44, 39, 21, 43, 14, 1, 10],
       [ 2, 41, 39, 54, 37, 21, 26, 57],
       [10, 39, 57, 48, 21, 54, 38, 43],
       [57, 39, 43, 1, 10, 21, 21, 33]])

array([[14, 44, 39, 21, 43, 14, 1, 10],
       [ 2, 41, 39, 54, 37, 21, 26, 57],
       [10, 39, 57, 48, 21, 54, 38, 43],
       [57, 39, 43, 1, 10, 21, 21, 33],
       [43, 43, 41, 60, 52, 12, 54, 1]])
```

如果这对你来说现在还不明白，也不要担心。这和深度学习理论关系不大；这只是设置 RNN 时一个特别复杂的部分，你有时会遇到。我想花几页纸来解释它。

#### 让我们看看如何使用截断反向传播进行迭代

以下代码展示了截断反向传播的实际应用。注意，它看起来与第十三章中的迭代逻辑非常相似。唯一的真正区别是，你会在每个步骤生成一个`batch_loss`；并且每进行一次`bptt`步骤后，你都会进行反向传播并更新权重。然后你继续像什么都没发生一样读取数据集（甚至使用之前的相同隐藏状态，它仅在每轮中重置）：

```
def train(iterations=100):
    for iter in range(iterations):
        total_loss = 0
        n_loss = 0

        hidden = model.init_hidden(batch_size=batch_size)
        for batch_i in range(len(input_batches)):

            hidden = Tensor(hidden.data, autograd=True)
            loss = None
            losses = list()
            for t in range(bptt):
                input = Tensor(input_batches[batch_i][t], autograd=True)
                rnn_input = embed.forward(input=input)
                output, hidden = model.forward(input=rnn_input,
                                               hidden=hidden)
               target = Tensor(target_batches[batch_i][t], autograd=True)
                batch_loss = criterion.forward(output, target)
                losses.append(batch_loss)
                if(t == 0):
                    loss = batch_loss
                else:
                    loss = loss + batch_loss
            for loss in losses:
                ""
            loss.backward()
            optim.step()
            total_loss += loss.data
            log = "\r Iter:" + str(iter)
            log += " - Batch "+str(batch_i+1)+"/"+str(len(input_batches))
            log += " - Loss:" + str(np.exp(total_loss / (batch_i+1)))
            if(batch_i == 0):
                log += " - " + generate_sample(70,'\n').replace("\n"," ")
            if(batch_i % 10 == 0 or batch_i-1 == len(input_batches)):
                sys.stdout.write(log)
        optim.alpha *= 0.99
        print()
train()

 Iter:0 - Batch 191/195 - Loss:148.00388828554404
 Iter:1 - Batch 191/195 - Loss:20.588816924127116 mhnethet tttttt t t t
                                     ....
 Iter:99 - Batch 61/195 - Loss:1.0533843281265225 I af the mands your
```

### 输出样本的一个示例

#### 通过从模型的预测中采样，你可以写出莎士比亚的作品！

以下代码使用训练逻辑的子集来使用模型进行预测。你将预测存储在一个字符串中，并将字符串版本作为输出返回给函数。生成的样本看起来非常像莎士比亚的作品，甚至包括对话角色：

```
def generate_sample(n=30, init_char=' '):
    s = ""
    hidden = model.init_hidden(batch_size=1)
    input = Tensor(np.array([word2index[init_char]]))
    for i in range(n):
        rnn_input = embed.forward(input)
        output, hidden = model.forward(input=rnn_input, hidden=hidden)
        output.data *= 10                                              *1*
        temp_dist = output.softmax()
        temp_dist /= temp_dist.sum()

        m = (temp_dist > np.random.rand()).argmax()                    *2*
        c = vocab[m]
        input = Tensor(np.array([m]))
        s += c
    return s
print(generate_sample(n=2000, init_char='\n'))
```

+ ***1* 样本采样的温度；越高 = 越贪婪**
+ ***2* 预测样本**

