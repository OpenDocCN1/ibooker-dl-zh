
# 随机交换

*随机交换*增强接受一个句子，然后在其中 *n* 次交换单词，每次迭代都在先前交换的句子上进行。这里是一个实现：

```py
def random_swap(sentence, n=5):
    length = range(len(sentence))
    for _ in range(n):
        idx1, idx2 = random.sample(length, 2)
        sentence[idx1], sentence[idx2] = sentence[idx2], sentence[idx1]
    return sentence
```

我们根据句子的长度随机抽取两个随机数，然后一直交换直到达到 *n*。

EDA 论文中的技术在使用少量标记示例（大约 500 个）时平均提高了约 3% 的准确性。如果你的数据集中有 5000 个以上的示例，该论文建议这种改进可能会降至 0.8% 或更低，因为模型从更多可用数据量中获得更好的泛化能力，而不是从 EDA 提供的改进中获得。

## 回译

另一种流行的增强数据集的方法是*回译*。这涉及将一个句子从我们的目标语言翻译成一个或多个其他语言，然后将它们全部翻译回原始语言。我们可以使用 Python 库 `googletrans` 来实现这个目的。在写作时，你可以使用 `pip` 安装它，因为它似乎不在 `conda` 中：

```py
pip install googletrans
```

然后，我们可以将我们的句子从英语翻译成法语，然后再翻译回英语：

```py
import googletrans
import googletrans.Translator

translator = Translator()

sentences = ['The cat sat on the mat']

translation_fr = translator.translate(sentences, dest='fr')
fr_text = [t.text for t in translations_fr]
translation_en = translator.translate(fr_text, dest='en')
en_text = [t.text for t in translation_en]
print(en_text)

>> ['The cat sat on the carpet']
```

这样我们就得到了一个从英语到法语再到英语的增强句子，但让我们再进一步，随机选择一种语言：

```py
import random

available_langs = list(googletrans.LANGUAGES.keys())
tr_lang = random.choice(available_langs)
print(f"Translating to {googletrans.LANGUAGES[tr_lang]}")

translations = translator.translate(sentences, dest=tr_lang)
t_text = [t.text for t in translations]
print(t_text)

translations_en_random = translator.translate(t_text, src=tr_lang, dest='en')
en_text = [t.text for t in translations_en_random]
print(en_text)
```

在这种情况下，我们使用 `random.choice` 来选择一个随机语言，将句子翻译成该语言，然后再翻译回来。我们还将语言传递给 `src` 参数，以帮助谷歌翻译的语言检测。试一试，看看它有多像*电话*这个老游戏。

你需要了解一些限制。首先，一次只能翻译最多 15,000 个字符，尽管如果你只是翻译句子的话，这应该不会是太大的问题。其次，如果你要在一个大型数据集上使用这个方法，你应该在云实例上进行数据增强，而不是在家里的电脑上，因为如果谷歌封禁了你的 IP，你将无法正常使用谷歌翻译！确保你一次发送几批数据而不是整个数据集。这也应该允许你在谷歌翻译后端出现错误时重新启动翻译批次。

## 增强和 torchtext

到目前为止，你可能已经注意到我所说的关于增强的一切都没有涉及 `torchtext`。遗憾的是，这是有原因的。与 `torchvision` 或 `torchaudio` 不同，`torchtext` 并没有提供转换管道，这有点让人恼火。它确实提供了一种执行预处理和后处理的方式，但这只在标记（单词）级别上操作，这对于同义词替换可能足够了，但对于像回译这样的操作并没有提供足够的控制。如果你尝试在增强中利用这些管道，你应该在预处理管道中而不是后处理管道中进行，因为在后处理管道中你只会看到由整数组成的张量，你需要通过词汇规则将其映射到单词。

出于这些原因，我建议不要浪费时间试图把 `torchtext` 搞得一团糟来进行数据增强。相反，使用诸如回译之类的技术在 PyTorch 之外进行增强，生成新数据并将其输入模型，就像它是*真实*数据一样。

增强已经讨论完毕，但在结束本章之前，我们应该解决一个悬而未决的问题。

## 迁移学习？

也许你会想知道为什么我们还没有谈论迁移学习。毕竟，这是一个关键技术，可以帮助我们创建准确的基于图像的模型，那么为什么我们不能在这里做呢？事实证明，在 LSTM 网络上实现迁移学习有点困难。但并非不可能。我们将在第九章中回到这个主题，你将看到如何在基于 LSTM 和 Transformer 的网络上实现迁移学习。

# 结论

在这一章中，我们涵盖了一个文本处理流程，涵盖了编码和嵌入，一个简单的基于 LSTM 的神经网络用于执行分类，以及一些针对基于文本数据的数据增强策略。到目前为止，您有很多可以尝试的内容。在标记化阶段，我选择将每条推文都转换为小写。这是自然语言处理中的一种流行方法，但它确实丢弃了推文中的潜在信息。想想看：“为什么这不起作用？”对我们来说甚至更暗示了负面情绪，而不是“为什么这不起作用？”但是在它进入模型之前，我们已经丢弃了这两条推文之间的差异。因此，一定要尝试在标记化文本中保留大小写敏感性。尝试从输入文本中删除停用词，看看是否有助于提高准确性。传统的自然语言处理方法非常强调删除停用词，但我经常发现当保留输入中的停用词时，深度学习技术可以表现得更好（这是我们在本章中所做的）。这是因为它们为模型提供了更多的上下文信息，而将句子简化为*仅包含*重要单词的情况可能会丢失文本中的细微差别。

您可能还想改变嵌入向量的大小。更大的向量意味着嵌入可以捕捉更多关于其建模的单词的信息，但会使用更多内存。尝试从 100 到 1,000 维的嵌入，并查看它如何影响训练时间和准确性。

最后，您也可以尝试使用 LSTM。我们使用了一种简单的方法，但您可以增加 `num_layers` 以创建堆叠的 LSTM，增加或减少层中隐藏特征的数量，或设置 `bidirectional=true` 以创建双向 LSTM。将整个 LSTM 替换为 GRU 层也是一个有趣的尝试；它训练速度更快吗？准确性更高吗？尝试实验并看看您会发现什么！

与此同时，我们将从文本转向音频领域，使用 `torchaudio`。

# 进一步阅读

+   [“长短期记忆”](https://oreil.ly/WKcxO) 作者：S. Hochreiter 和 J. Schmidhuber（1997 年）

+   [“使用 RNN 编码器-解码器学习短语表示进行统计机器翻译”](https://arxiv.org/abs/1406.1078) 作者：Kyunghyun Cho 等（2014 年）

+   [“双向 LSTM-CRF 模型用于序列标记”](https://arxiv.org/abs/1508.01991) 作者：Zhiheng Huang 等（2015 年）

+   [“注意力机制是您所需要的一切”](https://arxiv.org/abs/1706.03762) 作者：Ashish Vaswani 等（2017 年）

¹ 请注意，使用 CNN 也可以做到这些事情；在过去几年中，已经进行了大量深入研究，以将基于 CNN 的网络应用于时间域。我们不会在这里涵盖它们，但[“时间卷积网络：动作分割的统一方法”](https://arxiv.org/abs/1608.08242) 作者：Colin Lea 等（2016 年）提供了更多信息。还有 seq2seq！

² 参见[“在向量空间中高效估计单词表示”](https://arxiv.org/abs/1301.3781) 作者：Tomas Mikolov 等（2013 年）

³ 参见[“EDA：用于提升文本分类任务性能的简单数据增强技术”](https://arxiv.org/abs/1901.11196) 作者：Jason W. Wei 和 Kai Zou（2019 年）
