
```py
def is_grammatical(sentence):
    word1, word2, word3, word4, word5, word6 = sentence
    state = init_state()

    state = process_main_verb(w1 *
        process_adverb(w1 *
            process_relative_clause_verb(w1 *
                process_relative_clause_subject(w1 *
                    process_main_subject(w1 *
                        process_article(w1 * state + w2 * word1 + b) +
                    w2 * word2 + b) +
                w2 * word3 + b) +
            w2 * word4 + b) + 
        w2 * word5 + b) + 
    w2 * word6 + b)

    return state
```

为了识别输入确实是一句语法正确的英语句子（或一句句子的前缀），RNN 需要保留有关主语（“书”）的信息在状态中，直到看到动词（“were”）而不会被中间的任何东西（“我昨天读了”）分散注意力。在先前的伪代码中，状态由函数调用的返回值表示，因此关于主题的信息（`process_main_subject` 的返回值）需要在链中传播到达最外层函数（`process_main_verb`）。这开始听起来像是一项困难的任务。

当涉及训练该 RNN 时，情况并不好。RNN 和其他任何神经网络都使用称为*反向传播*的算法进行训练。反向传播是一种过程，在该过程中，神经网络的组成部分与先前的组成部分通信，以便调整参数以最小化损失。对于这个特定的示例，它是如何工作的。首先，您查看结果，即 `is_grammatical` 的返回值()并将其与您期望的内容进行比较。这两者之间的差称为*损失*。最外层函数 `is_grammatical()` 基本上有四种方式来减少损失，使其输出更接近所需内容：1) 调整 `w1`，同时固定嵌套函数 `process_adverb()` 的返回值，2) 调整 `w2`，3) 调整 `b`，或 4) 调整 `process_adverb()` 的返回值，同时固定参数。调整参数（`w1`、`w2` 和 `b`）很容易，因为函数知道调整每个参数对其返回值的确切影响。然而，调整上一个函数的返回值是不容易的，因为调用者不知道函数内部的工作原理。因此，调用者告诉上一个函数（被调用方）调整其返回值以最小化损失。请参见图 4.7，了解损失如何向后传播到参数和先前的函数。

![CH04_F07_Hagiwara](img/CH04_F07_Hagiwara.png)

图 4.7 损失的反向传播

嵌套的函数调用重复这个过程并玩转电话游戏，直到消息传递到最内层函数。到那个时候，因为消息需要经过许多层，它变得非常微弱和模糊（或者如果有误解则非常强大和扭曲），以至于内部函数很难弄清楚自己做错了什么。

技术上讲，深度学习文献将此称为*梯度消失问题*。*梯度*是一个数学术语，对应于每个函数从下一个函数接收到的信息信号，该信号指示它们应该如何改进其过程（如何更改其魔法常数）。反向电话游戏，其中消息从最终函数（=损失函数）向后传递，称为反向传播。我不会涉及这些术语的数学细节，但至少在概念上理解它们是有用的。

由于梯度消失问题，简单的循环神经网络（Simple RNNs）难以训练，在实践中现在很少使用。

### 4.2.2 长短期记忆（LSTM）

之前提到的嵌套函数处理语法信息的方式似乎太低效了。毕竟，为什么外部函数（`is_grammatical`）不直接告诉负责的特定函数（例如，`process_main_subject`）出了什么问题，而不是玩电话游戏呢？它不能这样做，因为每次函数调用后消息都可以完全改变其形状，这是由于 `w2` 和 `f()`。最外层函数无法仅从最终输出中告诉哪个函数负责消息的哪个部分。

我们如何解决这个低效性呢？与其每次通过激活函数传递信息并完全改变其形状，不如在每一步中添加和减去与正在处理的句子部分相关的信息？例如，如果 `process_main_subject()` 可以直接向某种“记忆”中添加有关主语的信息，并且网络可以确保记忆通过中间函数完整地传递，`is_grammatical()` 就会更容易告诉前面的函数如何调整其输出。

*长短期记忆*单元（LSTMs）是基于这一观点提出的一种 RNN 单元。LSTM 单元不是传递状态，而是共享“记忆”，每个单元都可以从中删除旧信息并/或添加新信息，有点像制造工厂中的装配线。具体来说，LSTM RNN 使用以下函数来更新状态：

```py
def update_lstm(state, word):
    cell_state, hidden_state = state

    cell_state *= forget(hidden_state, word)
    cell_state += add(hidden_state, word)

    hidden_state = update_hidden(hidden_state, cell_state, word)

    return (cell_state, hidden_state)
```

![CH04_F08_Hagiwara](img/CH04_F08_Hagiwara.png)

图 4.8 LSTM 更新函数

尽管与其简单版本相比，这看起来相对复杂，但是如果你将其分解为子组件，就不难理解这里正在发生的事情，如下所述并在图 4.8 中显示：

+   LSTM 状态包括两个部分——细胞状态（“记忆”部分）和隐藏状态（“心理表征”部分）。
+   函数 `forget()` 返回一个介于 0 和 1 之间的值，因此乘以这个数字意味着从 `cell_state` 中擦除旧的记忆。要擦除多少由 `hidden_state` 和 `word`（输入）决定。通过乘以介于 0 和 1 之间的值来控制信息流动称为*门控*。LSTM 是第一个使用这种门控机制的 RNN 架构。
+   函数 `add()` 返回添加到记忆中的新值。该值再次是由 `hidden_state` 和 `word` 决定的。
+   最后，使用一个函数更新 `hidden_state`，该函数的值是从前一个隐藏状态、更新后的记忆和输入单词计算得出的。

我通过隐藏一些数学细节在 `forget()`、`add()` 和 `update_hidden()` 函数中抽象了更新函数，这些细节对于这里的讨论不重要。如果你对深入了解 LSTM 感兴趣，我建议你阅读克里斯·奥拉在此主题上撰写的精彩博文（[`colah.github.io/posts/2015-08-Understanding-LSTMs/`](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)）。

因为 LSTMs 有一个在不同时间步保持不变的单元状态，除非显式修改，它们更容易训练并且相对表现良好。因为你有一个共享的“记忆”，函数正在添加和删除与输入句子的不同部分相关的信息，所以更容易确定哪个函数做了什么以及出了什么问题。来自最外层函数的错误信号可以更直接地到达负责函数。

术语说明：LSTM 是此处提到的一种特定类型的架构，但人们使用 “LSTMs” 来表示带有 LSTM 单元的 RNN。此外，“RNN” 常常用来指代“简单 RNN”，在第 4.1.3 节中介绍。在文献中看到“RNNs”时，你需要注意它们使用的确切架构。

### 4.2.3 门控循环单元（GRUs）

另一种 RNN 架构称为*门控循环单元*（GRUs），它使用门控机制。GRUs 的理念与 LSTMs 相似，但 GRUs 仅使用一组状态而不是两组。GRUs 的更新函数如下所示：

```py
def update_gru(state, word): 
    new_state = update_hidden(state, word)

    switch = get_switch(state, word)

    state = swtich * new_state + (1 - switch) * state

    return state
```

GRUs 不使用擦除或更新内存，而是使用切换机制。单元首先从旧状态和输入计算出新状态。然后计算切换值，一个介于 0 和 1 之间的值。根据切换值选择新状态和旧状态之间的状态。如果它是 0，旧状态保持不变。如果它是 1，它将被新状态覆盖。如果它在两者之间，状态将是两者的混合。请参见图 4.9，了解 GRU 更新函数的示意图。

![CH04_F09_Hagiwara](img/CH04_F09_Hagiwara.png)

图 4.9 GRU 更新函数

请注意，与 LSTMs 相比，GRUs 的更新函数要简单得多。实际上，它的参数（魔术常数）比 LSTMs 需要训练的参数少。因此，GRUs 比 LSTMs 更快地训练。

最后，尽管我们介绍了两种不同类型的 RNN 架构，即 LSTM 和 GRU，但在社区中并没有一致的共识，哪种类型的架构对于所有应用最好。你通常需要将它们视为超参数，并尝试不同的配置。幸运的是，只要你使用现代深度学习框架如 PyTorch 和 TensorFlow，就很容易尝试不同类型的 RNN 单元。

## 4.3 准确率、精确率、召回率和 F-度量

在第 2.7 节，我简要地讨论了一些我们用于评估分类任务性能的指标。在我们继续实际构建一个句子分类器之前，我想进一步讨论我们将要使用的评估指标——它们的含义以及它们实际上衡量的内容。

### 4.3.1 准确率

准确率可能是我们所讨论的所有评估指标中最简单的。在分类设置中，准确率是你的模型预测正确的实例的比例。例如，如果有 10 封电子邮件，而你的垃圾邮件过滤模型正确地识别了其中的 8 封，那么你的预测准确率就是 0.8，或者 80%（见图 4.10）。

![CH04_F10_Hagiwara](img/CH04_F10_Hagiwara.png)

图 4.10 计算准确率

虽然简单，但准确率并不是没有局限性。具体来说，在测试集不平衡时，准确率可能会误导。一个*不平衡*的数据集包含多个类别标签，它们的数量差异很大。例如，如果一个垃圾邮件过滤数据集不平衡，可能包含 90% 的非垃圾邮件和 10% 的垃圾邮件。在这种情况下，即使一个愚蠢的分类器把一切都标记为非垃圾邮件，也能够达到 90% 的准确率。例如，如果一个“愚蠢”的分类器在图 4.10 中将所有内容都分类为“非垃圾邮件”，它仍然会达到 70% 的准确率（10 个实例中的 7 个）。如果你孤立地看这个数字，你可能会被误导以为分类器的性能实际上很好。当你使用准确率作为指标时，将其与假想的、愚蠢的分类器（*多数投票*）作为基准进行比较总是一个好主意。

### 4.3.2 精确率和召回率

剩下的指标——精确率、召回率和 F-度量——是在二元分类设置中使用的。二元分类任务的目标是从另一个类别（称为*负类*）中识别出一个类别（称为*正类*）。在垃圾邮件过滤设置中，正类是垃圾邮件，而负类是非垃圾邮件。

图 4.11 中的维恩图包含四个子区域：真正例、假正例、假负例和真负例。真正例（TP）是被预测为正类（= 垃圾邮件）并且确实属于正类的实例。假正例（FP）是被预测为正类（= 垃圾邮件）但实际上不属于正类的实例。这些是预测中的噪音，也就是被误认为垃圾邮件并最终出现在你的电子邮件客户端的垃圾邮件文件夹中的无辜非垃圾邮件。

另一方面，假阴性（FN）是被预测为负类但实际上属于正类的实例。这些是通过垃圾邮件过滤器漏过的垃圾邮件，最终出现在你的收件箱中。最后，真阴性（TN）是被预测为负类并且确实属于负类的实例（即出现在你的收件箱中的非垃圾邮件）。

精确率是模型将正确分类为正例的实例的比例。例如，如果你的垃圾邮件过滤器将三封邮件标记为垃圾邮件，并且其中有两封确实是垃圾邮件，则精确率将为 2/3，约为 66%。

召回率与精确率有些相反。它是你的模型在数据集中被正确识别为正例的正例占比。再以垃圾邮件过滤为例，如果你的数据集中有三封垃圾邮件，而你的模型成功识别了其中两封邮件为垃圾邮件，则召回率将为 2/3，约为 66%。

图 4.11 显示了预测标签和真实标签之间以及召回率和精确率之间的关系。

![CH04_F11_Hagiwara](img/CH04_F11_Hagiwara.png)

图 4.11 精确率和召回率

### 4.3.3 F-测量

你可能已经注意到了精确率和召回率之间的权衡。想象一下有一个非常谨慎的垃圾邮件过滤器。它只有在几千封邮件中输出一封邮件为垃圾邮件，但当它输出时，它总是正确的。这不是一个困难的任务，因为一些垃圾邮件非常明显 - 如果它们的文本中包含“v1@gra”这个词，并且是从垃圾邮件黑名单中的人发送的，将其标记为垃圾邮件应该是相当安全的。这个垃圾邮件过滤器的精确率是多少？100%。同样，还有另一个非常粗心的垃圾邮件过滤器。它将每封电子邮件都分类为垃圾邮件，包括来自同事和朋友的电子邮件。它的召回率是多少？100%。这两个垃圾邮件过滤器中的任何一个有用吗？几乎没有！

正如你所看到的，只关注精确率或召回率而忽视另一个是不好的做法，因为它们之间存在权衡。这就好比你在节食时只关注体重。你减了 10 磅？太棒了！但是如果你身高是 7 英尺呢？并不是很好。你需要同时考虑身高和体重-太多是多少取决于另一个变量。这就是为什么有像 BMI（身体质量指数）这样的衡量标准，它同时考虑了这两个指标。同样，研究人员提出了一种叫做 F-测量的度量标准，它是精确率和召回率的平均值（更准确地说是调和平均值）。通常使用的是一个叫做 F1-测量的特殊案例，它是 F-测量的等权版本。在分类设置中，衡量并尝试最大化 F-测量是一种很好的做法。

## 4.4 构建 AllenNLP 训练流程

在本节中，我们将重新审视第二章中构建的情感分析器，并详细讨论如何更详细地构建其训练流程。尽管我已经展示了使用 AllenNLP

要运行本节中的代码，您需要导入必要的类和模块，如下面的代码片段所示（本节中的代码示例也可以通过 Google Colab 访问，[`www.realworldnlpbook.com/ch2.html#sst-nb`](http://www.realworldnlpbook.com/ch2.html#sst-nb)）。

```py
from itertools import chain
from typing import Dict

import numpy as np
import torch
import torch.optim as optim
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.training.trainer import GradientDescentTrainer
from allennlp_models.classification.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
```

### 4.4.1 实例和字段

如第 2.2.1 节所述，实例是机器学习算法进行预测的原子单位。数据集是同一形式实例的集合。大多数 NLP 应用的第一步是读取或接收一些数据（例如从文件或通过网络请求）并将其转换为实例，以便 NLP/ML 算法可以使用它们。

AllenNLP 支持一个称为 `DatasetReader` 的抽象，它的工作是读取一些输入（原始字符串、CSV 文件、来自网络请求的 JSON 数据结构等）并将其转换为实例。AllenNLP 已经为 NLP 中使用的主要格式提供了广泛的数据集读取器，例如 CoNLL 格式（在语言分析的流行共享任务中使用）和 Penn Treebank（一种流行的用于句法分析的数据集）。要读取 Standard Sentiment Treebank，可以使用内置的 `StanfordSentimentTreeBankDatasetReader`，我们在第二章中已经使用过了。您还可以通过覆盖 `DatasetReader` 的一些核心方法来编写自己的数据集阅读器。

AllenNLP 类 `Instance` 表示一个单独的实例。一个实例可以有一个或多个字段，这些字段保存某种类型的数据。例如，情感分析任务的实例有两个字段——文本内容和标签——可以通过将字段字典传递给其构造函数来创建，如下所示：

```py
Instance({'tokens': TextField(tokens),
          'label': LabelField(sentiment)})
```

在这里，我们假设您已经创建了 `tokens`（一个标记列表）和 `sentiment`（一个与情感类别对应的字符串标签），并从读取输入文件中获取了它们。根据任务，AllenNLP 还支持其他类型的字段。

`DatasetReader` 的 `read()` 方法返回一个实例迭代器，使您能够枚举生成的实例并对其进行可视化检查，如下面的代码片段所示：
