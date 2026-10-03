
```python
import sys,random,math
from collections import Counter
import numpy as np
import sys

np.random.seed(0)

f = open('shakespear.txt','r')
raw = f.read()
f.close()

vocab = list(set(raw))
word2index = {}
for i,word in enumerate(vocab):
    word2index[word]=i
indices = np.array(list(map(lambda x:word2index[x], raw)))

embed = Embedding(vocab_size=len(vocab),dim=512)
model = LSTMCell(n_inputs=512, n_hidden=512, n_output=len(vocab))
model.w_ho.weight.data *= 0                                         *1*

criterion = CrossEntropyLoss()
optim = SGD(parameters=model.get_parameters() + embed.get_parameters(),
            alpha=0.05)

batch_size = 16
bptt = 25
n_batches = int((indices.shape[0] / (batch_size)))

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
min_loss = 1000
```

+   ***1* 这似乎有助于训练。**

## 训练 LSTM 字符语言模型

### 训练逻辑也没有发生太多变化

从标准的 RNN 逻辑中，你唯一需要做的真正改变是截断反向传播逻辑，因为每个时间步长有两个隐藏向量而不是一个。但这只是一个相对较小的修复（粗体）。我还增加了一些使训练更简单的功能（`alpha` 随时间缓慢减少，并且有更多的日志记录）：

```python
for iter in range(iterations):
    total_loss, n_loss = (0, 0)
     hidden = model.init_hidden(batch_size=batch_size)
     batches_to_train = len(input_batches)

     for batch_i in range(batches_to_train):

         hidden = (Tensor(hidden[0].data, autograd=True),
                   Tensor(hidden[1].data, autograd=True))
          losses = list()

          for t in range(bptt):
              input = Tensor(input_batches[batch_i][t], autograd=True)
              rnn_input = embed.forward(input=input)
             output, hidden = model.forward(input=rnn_input, hidden=hidden)

              target = Tensor(target_batches[batch_i][t], autograd=True)
              batch_loss = criterion.forward(output, target)

              if(t == 0):
                  losses.append(batch_loss)
              else:
                  losses.append(batch_loss + losses[-1])
          loss = losses[-1]

          loss.backward()
          optim.step()

         total_loss += loss.data / bptt
         epoch_loss = np.exp(total_loss / (batch_i+1))
         if(epoch_loss < min_loss):
             min_loss = epoch_loss
             print()
         log = "\r Iter:" + str(iter)
         log += " - Alpha:" + str(optim.alpha)[0:5]
         log += " - Batch "+str(batch_i+1)+"/"+str(len(input_batches))
         log += " - Min Loss:" + str(min_loss)[0:5]
         log += " - Loss:" + str(epoch_loss)
         if(batch_i == 0):
             s = generate_sample(n=70, init_char='T').replace("\n"," ")
             log += " - " + s
         sys.stdout.write(log)
     optim.alpha *= 0.99
```

## 调整 LSTM 字符语言模型

### 我花费了大约两天的时间调整这个模型，并且它是在夜间训练完成的

这里是此模型的某些训练输出。请注意，训练这个模型花费了非常长的时间（有很多参数）。我还不得不多次训练它，以找到这个任务的良好调整（学习率、批量大小等），并且最终的模型是在夜间（8 小时）训练完成的。一般来说，你训练的时间越长，你的结果就会越好。

```text
I:0 - Alpha:0.05 - Batch 1/249 - Min Loss:62.00 - Loss:62.00 - eeeeeeeeee
                                   ...
I:7 - Alpha:0.04 - Batch 140/249 - Min Loss:10.5 - Loss:10.7 - heres, and
                                   ...
I:91 - Alpha:0.016 - Batch 176/249 - Min Loss:9.900 - Loss:11.9757225699
```

```python
def generate_sample(n=30, init_char=' '):
    s = ""
    hidden = model.init_hidden(batch_size=1)
    input = Tensor(np.array([word2index[init_char]]))
    for i in range(n):
        rnn_input = embed.forward(input)
        output, hidden = model.forward(input=rnn_input, hidden=hidden)
        output.data *= 15
        temp_dist = output.softmax()
        temp_dist /= temp_dist.sum()

        m = output.data.argmax()           *1*
        c = vocab[m]
        input = Tensor(np.array([m]))
        s += c
    return s
print(generate_sample(n=500, init_char='\n'))
```

+   ***1* 取最大预测**

```text
Intestay thee.

SIR:
It thou my thar the sentastar the see the see:
Imentary take the subloud I
Stall my thentaring fook the senternight pead me, the gakentlenternot
they day them.

KENNOR:
I stay the see talk :
Non the seady!

Sustar thou shour in the suble the see the senternow the antently the see
the seaventlace peake,
I sentlentony my thent:
I the sentastar thamy this not thame.
```

## 摘要

### LSTM 是极其强大的模型

LSTM 学习生成莎士比亚语言的分布不容小觑。语言是一个极其复杂的统计分布，学习起来非常困难，而 LSTM 能够做得如此出色（在撰写本文时，它们是广泛领先的最佳方法）仍然让我（以及其他一些人）感到困惑。这个模型的小型变体要么是，要么最近一直是各种任务中的最佳状态，并且与词嵌入和卷积层一起，无疑将是我们长期以来的首选工具。
