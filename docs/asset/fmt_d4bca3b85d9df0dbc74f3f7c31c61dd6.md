
# 识别直接相关性的最小架构

识别直接相关性的最小架构是一个两层网络，其中网络有一个直接从输入层连接到输出层的单权重矩阵。但我们的网络有一个隐藏层。这引发了一个问题，这个隐藏层有什么作用？

基本上，隐藏层是将前一层的数据点分组为 *n* 组（其中 *n* 是隐藏层中神经元的数量）。每个隐藏神经元接收一个数据点并回答问题：“这个数据点是否在我的组中？”随着隐藏层的不断学习，它会寻找其输入的有用分组。什么是有用的分组？

如果一个输入数据点分组能够做两件事，那么它是有用的。首先，分组必须对预测输出标签有用。如果它对输出预测没有用，*相关性总结永远不会引导网络找到该组*。这是一个非常有价值的认识。神经网络研究的大部分内容都是关于找到训练数据（或为网络制造的其他信号，以便它能够人为地预测）以找到对任务有用的分组（例如预测电影评论星级）。我们将在稍后讨论这一点。

其次，如果分组是数据中你关心的实际现象，那么它是有用的。不良的分组只是记住数据。好的分组能够捕捉到有用的语言现象。

例如，在预测电影评论是正面还是负面时，理解“糟糕”和“不糟糕”之间的差异是一种强大的分组。如果有一个神经元在看到“糟糕”时关闭，在看到“不糟糕”时打开，这将是一个强大的分组，供下一层使用以做出最终预测。但由于神经网络输入是评论的词汇，“它很棒，不糟糕”与“它很糟糕，不棒”创造了完全相同的`layer_1`值。因此，网络非常不可能创建一个理解否定意义的隐藏神经元。

基于某种语言模式测试层是否相同或不同，是了解架构是否可能使用相关性总结找到该模式的一个很好的第一步。如果你可以构建两个具有相同隐藏层的例子，一个有你感兴趣的模式，另一个没有，那么网络不太可能找到该模式。

正如你刚刚学到的，隐藏层在本质上将前一层的数据分组。在细粒度层面，每个神经元将数据点分类为属于或不属于其组。在更高层面，如果两个数据点（电影评论）属于许多相同的组，则它们是相似的。最后，如果连接到各种隐藏神经元的权重（每个单词的组亲和力度量）相似，则两个输入（单词）是相似的。有了这些知识，在前一个神经网络中，你应该在连接到隐藏神经元的单词权重中观察到什么？

## 你应该在连接单词和隐藏神经元的权重中看到什么？

这里有一个提示：具有相似预测能力的单词应该属于相似的组（隐藏神经元配置）。这对连接每个单词到每个隐藏神经元的权重意味着什么？

这里是答案。与相似标签（正面或负面）相关的单词将会有类似的权重将它们连接到各种隐藏神经元。这是因为神经网络学会将它们归入相似的隐藏神经元，以便最终层（`weights_1_2`）可以做出正确的正面或负面预测。

您可以通过选择一个特别积极或消极的词语，并搜索具有最相似权值的其他词语来观察这一现象。换句话说，您可以取每个词语，并查看哪些其他词语与每个隐藏神经元（每个组）具有最相似的权值连接。

![图片](img/f0198-01.jpg)

“好”的三个粗体权值形成了“好”的嵌入。它们反映了“好”这个术语是每个组（隐藏神经元）的成员程度。具有相似预测能力的词语具有相似的词嵌入（权值）。

属于相似组的词语将具有相似的正负标签预测能力。因此，属于相似组且具有相似权值（权重）的词语也将具有相似的含义。从抽象的角度来看，在神经网络中，一个神经元与同一层中其他神经元的含义相似，当且仅当它与下一层和/或前一层的连接权重相似。

## 比较词嵌入

### 如何可视化权重相似性？

对于每个输入词语，您可以通过选择`weights_0_1`中对应的行来选择从该词语到各个隐藏神经元的权值列表。行中的每个条目代表从该行词语到每个隐藏神经元的权值。因此，为了确定哪些词语与目标术语最相似，您需要比较每个词语的向量（矩阵的行）与目标术语的向量。选择的比较方法是称为*欧几里得距离*，如下面的代码所示：

```
from collections import Counter
import math

def similar(target='beautiful'):
    target_index = word2index[target]
    scores = Counter()
    for word,index in word2index.items():
        raw_difference = weights_0_1[index] - (weights_0_1[target_index])
        squared_difference = raw_difference * raw_difference
        scores[word] = -math.sqrt(sum(squared_difference))

     return scores.most_common(10)
```

这使得您可以根据网络轻松查询最相似的词语（神经元）：

```
print(similar('beautiful'))           print(similar('terrible'))

[('beautiful', -0.0),                 [('terrible', -0.0),
 ('atmosphere', -0.70542101298),       ('dull', -0.760788602671491),
 ('heart', -0.7339429768542354),       ('lacks', -0.76706470275372),
 ('tight', -0.7470388145765346),       ('boring', -0.7682894961694),
 ('fascinating', -0.7549291974),       ('disappointing', -0.768657),
 ('expecting', -0.759886970744),       ('annoying', -0.78786389931),
 ('beautifully', -0.7603669338),       ('poor', -0.825784172378292),
 ('awesome', -0.76647368382398),       ('horrible', -0.83154121717),
 ('masterpiece', -0.7708280057),       ('laughable', -0.8340279599),
 ('outstanding', -0.7740642167)]       ('badly', -0.84165373783678)]
```

如您所预期的那样，每个词语最相似的术语是它自己，其次是具有与目标术语相似有用性的词语。再次，正如您所预期的那样，因为网络只有两个标签（`正面`和`负面`），输入术语根据它们倾向于预测的标签分组。

这是一种标准的关联总结现象。它试图根据预测的标签在网络上创建相似的表现（`layer_1`值），以便能够预测正确的标签。在这种情况下，副作用是输入到`layer_1`的权重根据输出标签分组。

关联总结现象的关键启示是对这一现象的直观理解。它始终试图说服隐藏层基于应该预测的标签保持相似。

## 神经元的含义是什么？

### 意义完全基于预测的目标标签

注意，不同词语的含义并没有完全反映您可能如何将它们分组。与“美丽”最相似的术语是“气氛”。这是一个宝贵的教训。为了预测电影评论是正面还是负面，这些词语具有几乎相同的意义。但在现实世界中，它们的含义却截然不同（例如，一个是形容词，另一个是名词）。

```
print(similar('beautiful'))           print(similar('terrible'))

[('beautiful', -0.0),                 [('terrible', -0.0),
 ('atmosphere', -0.70542101298),       ('dull', -0.760788602671491),
 ('heart', -0.7339429768542354),       ('lacks', -0.76706470275372),
 ('tight', -0.7470388145765346),       ('boring', -0.7682894961694),
 ('fascinating', -0.7549291974),       ('disappointing', -0.768657),
 ('expecting', -0.759886970744),       ('annoying', -0.78786389931),
 ('beautifully', -0.7603669338),       ('poor', -0.825784172378292),
 ('awesome', -0.76647368382398),       ('horrible', -0.83154121717),
 ('masterpiece', -0.7708280057),       ('laughable', -0.8340279599),
 ('outstanding', -0.7740642167)]       ('badly', -0.84165373783678)]
```

这一认识非常重要。网络中（神经元）的含义是基于目标标签定义的。神经网络中的所有内容都是基于试图正确预测的相关性总结进行语境化的。因此，尽管你和我对这些单词非常了解，但神经网络对任务之外的所有信息一无所知。

你如何让网络学习关于神经元（在这种情况下，是单词神经元）的更细微的信息？好吧，如果你给它输入和目标数据，这些数据需要更细微的语言理解，它就会有理由学习各种术语的更细微的解释。

你应该使用神经网络来预测什么，以便它为单词神经元学习更有趣的权重值？你将用于学习单词神经元更有趣的权重值的学习任务是一个美化的填空任务。为什么使用这个？首先，有几乎无限的训练数据（互联网），这意味着神经网络有几乎无限的信号来学习关于单词的更细微的信息。此外，能够准确地填充空白至少需要一些关于现实世界的上下文概念。

例如，在以下示例中，空白处更可能是被“anvil”（砧）还是“wool”（羊毛）正确填充？让我们看看神经网络能否解决这个问题。

![](img/f0200-01.jpg)

## 填空

### 通过拥有更丰富的学习信号来学习单词的更丰富含义

这个例子几乎与上一个例子使用完全相同的神经网络，只有一些修改。首先，你不会预测针对电影评论的单个标签，而是将每个（五个单词）短语中的每个单词（焦点词）移除，并尝试训练一个网络来根据剩余的短语确定被移除的单词的身份。其次，你将使用一种称为*负采样*的技巧来使网络训练更快。

考虑到为了预测哪个术语缺失，你需要为每个可能的单词提供一个标签。这将需要数千个标签，这将导致网络训练缓慢。为了克服这一点，让我们在每次前向传播步骤中随机忽略每个标签的大部分（即，假装它们不存在）。虽然这看起来可能是一种粗略的近似，但在实践中这是一种效果很好的技术。以下是本例的预处理代码：

```
import sys,random,math
from collections import Counter
import numpy as np

np.random.seed(1)
random.seed(1)
f = open('reviews.txt')
raw_reviews = f.readlines()
f.close()

tokens = list(map(lambda x:(x.split(" ")),raw_reviews))
wordcnt = Counter()
for sent in tokens:
    for word in sent:
        wordcnt[word] -= 1
vocab = list(set(map(lambda x:x[0],wordcnt.most_common())))

word2index = {}
for i,word in enumerate(vocab):
    word2index[word]=i

concatenated = list()
input_dataset = list()
for sent in tokens:
    sent_indices = list()
    for word in sent:
        try:
             sent_indices.append(word2index[word])
             concatenated.append(word2index[word])
        except:
            ""
    input_dataset.append(sent_indices)
concatenated = np.array(concatenated)

random.shuffle(input_dataset)

alpha, iterations = (0.05, 2)
hidden_size,window,negative = (50,2,5)

weights_0_1 = (np.random.rand(len(vocab),hidden_size) - 0.5) * 0.2
weights_1_2 = np.random.rand(len(vocab),hidden_size)*0

layer_2_target = np.zeros(negative+1)
layer_2_target[0] = 1

def similar(target='beautiful'):
  target_index = word2index[target]

  scores = Counter()
  for word,index in word2index.items():
    raw_difference = weights_0_1[index] - (weights_0_1[target_index])
    squared_difference = raw_difference * raw_difference
    scores[word] = -math.sqrt(sum(squared_difference))
  return scores.most_common(10)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

for rev_i,review in enumerate(input_dataset * iterations):
  for target_i in range(len(review)):

    target_samples = [review[target_i]]+list(concatenated\                   *1*
      [(np.random.rand(negative)*len(concatenated)).astype('int').tolist()]) *1*

       left_context = review[max(0,target_i-window):target_i]
       right_context = review[target_i+1:min(len(review),target_i+window)]

       layer_1 = np.mean(weights_0_1[left_context+right_context],axis=0)
       layer_2 = sigmoid(layer_1.dot(weights_1_2[target_samples].T))
       layer_2_delta = layer_2 - layer_2_target
       layer_1_delta = layer_2_delta.dot(weights_1_2[target_samples])

       weights_0_1[left_context+right_context] -= layer_1_delta * alpha
       weights_1_2[target_samples] -= np.outer(layer_2_delta,layer_1)*alpha

  if(rev_i % 250 == 0):
    sys.stdout.write('\rProgress:'+str(rev_i/float(len(input_dataset)
        *iterations)) + "   " + str(similar('terrible')))
  sys.stdout.write('\rProgress:'+str(rev_i/float(len(input_dataset)
        *iterations)))
print(similar('terrible'))

  Progress:0.99998 [('terrible', -0.0), ('horrible', -2.846300248788519),
  ('brilliant', -3.039932544396419), ('pathetic', -3.4868595532695967),
  ('superb', -3.6092947961276645), ('phenomenal', -3.660172529098085),
  ('masterful', -3.6856112636664564), ('marvelous', -3.9306620801551664),
```

+   ***1* 只预测随机子集，因为预测每个词汇都非常昂贵**

## 意义来源于损失

使用这个新的神经网络，你可以主观地看到单词嵌入的聚类方式有所不同。以前单词是根据预测“积极”或“消极”标签的可能性进行聚类的，而现在它们是根据在同一短语中出现的可能性进行聚类的（有时不考虑情感）。

| Predicting POS/NEG | Fill in the blank |
| --- | --- |
