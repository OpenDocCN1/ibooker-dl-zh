
# 时间序列的循环神经网络与卷积神经网络

## 模型构建与训练

以下代码定义了一个结合卷积层与循环层的降采样模型，用于处理时间序列预测任务。

```py
class DownsamplingModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=4, stride=2)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        Z = X.permute(0, 2, 1)  # treat time as a spatial dimension
        Z = self.conv(Z)
        Z = Z.permute(0, 2, 1)  # swap back time & features dimensions
        Z = torch.relu(Z)
        Z, _states = self.gru(Z)
        return self.linear(Z)

torch.manual_seed(42)
dseq_model = DownsamplingModel(input_size=5, hidden_size=32, output_size=14)
dseq_model = dseq_model.to(device)
```

通过缩短时间序列，卷积层有助于`GRU`层检测更长的模式，因此我们可以将窗口长度加倍到 112 天。请注意，我们还必须从目标中裁剪掉前三个时间步：实际上，核的大小是 4，因此卷积层的第一个输出将基于输入时间步 0 到 3，因此第一个预测必须针对时间步 4 到 17（而不是时间步 1 到 14）。此外，由于步长，我们必须将目标下采样 2 倍。为此，我们需要一个新的`Dataset`类，因此让我们创建`Seq2SeqDataset`类的子类：

```py
class DownsampledDataset(Seq2SeqDataset):
    def __getitem__(self, idx):
        window, target = super().__getitem__(idx)
        return window, target[3::2]  # crop the first 3 targets and downsample

window_length = 112
dseq_train_set = DownsampledDataset(rail_train, window_length)
dseq_train_loader = DataLoader(dseq_train_set, batch_size=32, shuffle=True)
[...]  # create the datasets and data loaders for the validation and test sets
```

现在模型可以像往常一样进行训练。我们已经成功混合了卷积层和循环层。但是，如果我们只使用一维卷积层并完全丢弃循环层会怎样呢？

## WaveNet

在[2016 年的一篇论文](https://homl.info/wavenet)⁠^(16)中，Aaron van den Oord 和其他 DeepMind 研究人员介绍了一种名为*WaveNet*的新架构。他们堆叠了 1D 卷积层，每层将膨胀率（每个神经元的输入之间的距离）加倍：第一层卷积层一次只能看到两个时间步，而下一层可以看到四个时间步（其感受野长度为四个时间步），下一层可以看到八个时间步，以此类推（见图 13-14)。这样，较低层学习短期模式，而较高层学习长期模式。由于膨胀率加倍，网络可以非常高效地处理极长的序列。

![展示具有堆叠的 1D 卷积层和递增膨胀率的 WaveNet 架构的示意图，说明该模型如何以不同规模处理时间步以高效生成音频。](img/hmls_1314.png)

###### 图 13-14\. WaveNet 架构

论文的作者实际上堆叠了 10 个具有膨胀率 1、2、4、8、…、256、512 的卷积层，然后他们又堆叠了另一组 10 个相同的层（也具有膨胀率 1、2、4、8、…、256、512），然后再次堆叠了另一个相同的 10 层组。他们通过指出，一个具有这些膨胀率的单个 10 层卷积层堆叠将像一个具有 1,024 个大小内核的超高效卷积层（除了更快、更强大，并且使用显著更少的参数）。他们还在每一层之前用与膨胀率相等的零填充输入序列，以在整个网络中保持相同的序列长度。在左侧而不是两侧进行填充很重要，因为它确保卷积层在做出预测时不会看到未来。这使得它成为一个因果模型。

让我们实现一个简化的 WaveNet 来处理之前相同的序列。⁠^(17) 我们将首先创建一个自定义的`CausalConv1d`模块，它就像一个`nn.Conv1d`模块一样工作，除了输入在左侧通过适当的数量进行填充，以确保序列保持相同的长度：

```py
import torch.nn.functional as F

class CausalConv1d(nn.Conv1d):
    def forward(self, X):
        padding = (self.kernel_size[0] - 1) * self.dilation[0]
        X = F.pad(X, (padding, 0))
        return super().forward(X)
```

在此代码中，我们继承自`nn.Conv1d`类，并重写了`forward()`方法。在其中，我们计算所需的左填充大小，并在调用基类的`forward()`方法之前使用`pad()`函数填充序列。`pad()`函数接受两个参数：要填充的张量（`X`），以及一个表示在最后一个维度（即时间维度）左右填充多少个整数的元组。

现在我们已经准备好构建 WaveNet 模型本身：

```py
class WavenetModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        layers = []
        for dilation in (1, 2, 4, 8) * 2:
            conv = CausalConv1d(input_size, hidden_size, kernel_size=2,
                                dilation=dilation)
            layers += [conv, nn.ReLU()]
            input_size = hidden_size
        self.convs = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        Z = X.permute(0, 2, 1)
        Z = self.convs(Z)
        Z = Z.permute(0, 2, 1)
        return self.output(Z)

torch.manual_seed(42)
wavenet_model = WavenetModel(input_size=5, hidden_size=32, output_size=14)
wavenet_model = wavenet_model.to(device)
```

在构造函数中，我们创建了八个具有不同膨胀率的`CausalConv1d`层（1, 2, 4, 8，然后再次是 1, 2, 4, 8），每个层后面都跟着 ReLU 激活函数。我们将所有这些模块链接在一个`nn.Sequential`模块`self.convs`中。我们还创建了输出`nn.Linear`层。在正向方法中，我们交换输入的最后两个维度，就像我们之前做的那样，然后通过卷积层传递它们，然后交换最后两个维度回到原始顺序，并通过输出层传递结果。多亏了因果填充，每个卷积层都输出一个与输入序列长度相同的序列，因此我们用于训练的目标可以是完整的 112 天序列；无需裁剪或下采样。因此，我们可以使用为`Seq2SeqModel`（即`seq_train_loader`和`seq_valid_loader`）构建的数据加载器来训练模型。

我们在本节中讨论的模型在乘客预测任务上提供了相似的性能，但它们可能因任务和可用数据量的不同而有很大差异。在 WaveNet 论文中，作者在各种音频任务上实现了最先进的性能（因此得名该架构），包括文本到语音任务，产生跨越多种语言的非常逼真的声音。他们还使用该模型一次生成一个音频样本来生成音乐。当你意识到一个音频秒可以包含数万个时间步时，这一成就就更加令人印象深刻——即使是 LSTMs 和 GRUs 也无法处理如此长的序列。

###### 警告

如果你评估我们最好的芝加哥乘客模型在 2020 年开始的测试期间，你会发现它们的性能远低于预期！为什么？嗯，那正是 Covid-19 大流行开始的时候，这对公共交通产生了巨大影响。如前所述，这些模型只有在它们从过去学到的模式在未来继续存在时才会表现良好。无论如何，在将模型部署到生产之前，请确保它在最近的数据上表现良好。一旦它进入生产，请确保定期监控其性能。

这样，你现在可以处理各种时间序列！在第十四章中，我们将继续探索 RNN，并了解它们如何处理各种 NLP 任务。

# 练习

1.  你能想到几个序列到序列 RNN 的应用吗？序列到向量 RNN 和向量到序列 RNN 呢？

2.  RNN 层的输入必须有多少维？每个维度代表什么？它的输出呢？

3.  你如何在 PyTorch 中构建一个深度序列到序列 RNN？

4.  假设你有一个每日的单变量时间序列，你想使用 RNN 预测接下来七天。你应该使用哪种架构？

5.  训练 RNN 时主要有哪些困难？你如何处理它们？

6.  你能画出 LSTM 单元的架构吗？

7.  为什么你想要在一个 RNN 中使用 1D 卷积层？

8.  你可以使用哪种神经网络架构来对视频进行分类？

9.  尝试调整`Seq2SeqModel`模型以预测未来 14 天内铁路和公交客流量。现在，模型需要预测 28 个值而不是 14 个。

10. 下载[Bach 合唱曲](https://homl.info/bach)数据集并解压它。它由约翰·塞巴斯蒂安·巴赫创作的 382 首合唱曲组成。每首合唱曲长 100 到 640 个时间步，每个时间步包含 4 个整数，其中每个整数对应于钢琴上一个音符的索引（除了值 0，表示没有音符演奏）。训练一个模型——循环的、卷积的或两者结合——可以预测合唱曲的下一个时间步（四个音符），给定一个合唱曲的时间步序列。然后使用这个模型一次生成一个音符的巴赫风格音乐：你可以通过给模型一个合唱曲的开始部分并要求它预测下一个时间步，然后将这些时间步添加到输入序列中并要求模型预测下一个音符，依此类推。同时，确保查看[Google 的 Coconet 模型](https://homl.info/coconet)，该模型被用于一个关于巴赫的漂亮的 Google 涂鸦。

11. 训练一个针对[QuickDraw 数据集](https://homl.info/quickdraw)的分类模型，该数据集包含数百万个各种物体的草图。首先，下载几个类别的简化数据（例如，*ant.ndjson*、*axe.ndjson*和*bat.ndjson*）。每个 NDJSON 文件每行包含一个 JSON 对象，你可以使用 Python 的`json.loads()`函数来解析它。这将给你一个草图列表，其中每个草图都表示为一个 Python 字典。在每个字典中，`"drawing"`条目包含一系列笔触。你可以将这个列表转换为维度为[*笔触*、*x 坐标*、*y 坐标*]的 3D 浮点张量。由于 RNN 以单个序列作为输入，你需要将每个草图的笔触全部连接成一个单独的序列。最好添加一个额外的特征，以便 RNN 知道当前笔触的进度（例如，从 0 到 1）。换句话说，模型将接收一个序列，其中每个时间步长有三个特征：笔的*x*和*y*坐标，以及当前笔触的进度比。

12. 创建一个包含你说“是”或“否”的短音频录音的数据集，并在其上训练一个二进制分类循环神经网络（RNN）。例如，你可以：

    1.  使用 Audacity 等音频录制软件，尽可能多地录制你说“是”的声音，每个词之间有短暂的停顿。为“否”这个词也创建一个类似的录音。尽量涵盖你在现实生活中可能以各种方式发音的这些词。

    2.  使用 TorchAudio 库中的`torchaudio.load()`函数加载每个 WAV 文件。这将返回一个包含音频的张量，以及一个表示每秒样本数的整数。音频张量的形状为[*通道*，*样本*]：单声道一个通道，立体声两个通道。通过在通道维度上平均来将立体声转换为单声道。

    3.  通过在静音处分割，将每段录音分割成单个单词。你可以使用`torchaudio.transforms.Vad`转换（语音活动检测）来完成此操作。

    4.  由于序列如此之长，直接在它们上训练 RNN 很困难，因此首先将音频转换为频谱图很有帮助。为此，你可以使用`torchaudio.transforms.MelSpectrogram`转换，这对于语音来说非常适合。输出是一个显著更短的序列，具有更多的通道。

    5.  现在尝试在你的是/否数据集上构建和训练一个二进制分类 RNN！考虑将你的数据集和模型与世界分享（例如，通过 Hugging Face Hub）。

这些练习的解决方案可在本章笔记本的末尾找到，在[*https://homl.info/colab-p*](https://homl.info/colab-p)。

## 脚注

^(1) 注意，许多研究人员更喜欢在 RNN 中使用双曲正切（tanh）激活函数，而不是 ReLU 激活函数。例如，参见 Vu Pham 等人 2013 年的论文“Dropout Improves Recurrent Neural Networks for Handwriting Recognition”。基于 ReLU 的 RNN 也是可能的，如 Quoc V. Le 等人 2015 年的论文“A Simple Way to Initialize Recurrent Networks of Rectified Linear Units”所示。

^(2) Michael I. Jordan，“Attractor Dynamics and Parallelism in a Connectionist Sequential Machine”，*第八届认知科学学会年度会议论文集*（1986 年）。

^(3) Jeffrey L. Elman，“Finding Structure in Time”，*认知科学*，第 14 卷，第 2 期（1990 年）。

^(4) Nal Kalchbrenner 和 Phil Blunsom，“Recurrent Continuous Translation Models”，*2013 年实证自然语言处理会议论文集*（2013 年）：1700–1709。

^(5) 芝加哥交通管理局的最新数据可在[芝加哥数据门户](https://homl.info/ridership)找到。

^(6) 关于 ACF-PACF 方法的更多细节，请查看 Jason Brownlee 的这篇非常好的[帖子](https://homl.info/arimatuning)。

^(7) 注意，验证期从 2019 年 1 月 1 日开始，因此第一个预测是在 2019 年 2 月 26 日，八周后。当我们评估基线模型时，我们使用了从 2019 年 3 月 1 日开始进行的预测，但这应该足够接近。

^(8) 我们不能使用`ahead_model`，因为它需要铁路和公交客流量作为输入，但它只预测铁路客流量。

^(9) César Laurent 等人，“批量归一化循环神经网络”，*IEEE 国际声学、语音和信号处理会议论文集* (2016): 2657–2661.

^(10) 来自动画电影 *海底总动员* 和 *海底奇缘* 的角色，患有短期记忆丧失。

^(11) Sepp Hochreiter 和 Jürgen Schmidhuber，“长短期记忆”，*神经计算* 9, 第 8 期 (1997): 1735–1780.

^(12) Haşim Sak 等人，“基于长短期记忆的循环神经网络架构用于大词汇量语音识别”，arXiv 预印本 arXiv:1402.1128 (2014).

^(13) Wojciech Zaremba 等人，“循环神经网络正则化”，arXiv 预印本 arXiv:1409.2329 (2014).

^(14) Kyunghyun Cho 等人，“使用 RNN 编码器-解码器学习短语表示”，*2014 年自然语言处理实证方法会议论文集* (2014): 1724–1734.

^(15) 参见 Klaus Greff 等人，“LSTM：一次搜索空间探险”，[“LSTM: A Search Space Odyssey”](https://homl.info/98)，*IEEE 交易杂志：神经网络与学习系统* 28, 第 10 期 (2017): 2222–2232。这篇论文似乎表明所有 LSTM 变体表现大致相同。

^(16) Aaron van den Oord 等人，“WaveNet：原始音频的生成模型”，arXiv 预印本 arXiv:1609.03499 (2016).

^(17) 完整的 WaveNet 使用了一些额外的技巧，例如类似于 ResNet 中的跳跃连接，以及类似于在 GRU 单元中发现的 *门控激活单元*。
