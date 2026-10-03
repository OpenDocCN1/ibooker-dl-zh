
![CH10_F13_Hagiwara](img/CH10_F13_Hagiwara.png)

图 10.13 二元交叉熵损失（正确标签为 1）

如果您更关心模型在少数类上的表现，可以调整这个损失。具体而言，您可以更改这个损失的形状（通过简单地将其乘以一个常数），只针对那个类别，以便当模型在少数类上犯错时，它会产生更大的损失。图 10.14 中的一条调整后的损失曲线就是顶部的那条。这种加权与上采样少数类具有相同的效果，尽管修改损失的计算成本更低，因为您不需要实际增加训练数据量。

![CH10_F14_Hagiwara](img/CH10_F14_Hagiwara.png)

图 10.14 加权二元交叉熵损失

在 `PyTorch` 和 `AllenNLP` 中实现损失权重很容易。`PyTorch` 的二元交叉熵实现 `BCEWithLogitsLoss` 已经支持为不同类别使用不同的权重。您只需要将 `pos_weight` 参数作为权重传递，如下所示：

```py
>>> import torch
>>> import torch.nn as nn

>>> input = torch.randn(3)
>>> input
tensor([-0.5565,  1.5350, -1.3066])

>>> target = torch.empty(3).random_(2)
>>> target
tensor([0., 0., 1.])

>>> loss = nn.BCEWithLogitsLoss(reduction='none')
>>> loss(input, target)
tensor([0.4531, 1.7302, 1.5462])

>>> loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(2.))
>>> loss(input, target)
tensor([0.4531, 1.7302, 3.0923])
```

在这段代码片段中，我们随机生成预测值（`input`）和真实值（`target`）。总共有三个实例，其中两个属于类别 0（多数类），一个属于类别 1（少数类）。我们先使用 `BCEWithLogitsLoss` 对象计算不加权的损失，这将返回三个损失值，每个实例一个。然后，我们通过传递权重 2 来计算加权损失——这意味着如果目标类别是正类（类别 1），则错误预测将被惩罚两倍。请注意，对应于类别 1 的第三个元素是非加权损失函数返回值的两倍。

## 10.5 超参数调整

在本章的最后一节，我们将讨论超参数调整。*超参数*是有关模型和训练算法的参数。这个术语与*参数*相对，参数是模型用于从输入中作出预测的数字。这就是我们在本书中一直称之为“魔术常数”的内容——它们类似于编程语言中的常数，尽管它们的确切值被优化自动调整，以使预测尽可能接近所需输出。

正确调整超参数对于许多机器学习模型正常工作并发挥其最高潜力至关重要，机器学习从业者花费大量时间来调整超参数。知道如何有效地调整超参数对于提高在构建自然语言处理和机器学习系统时的生产力有着巨大的影响。

### 10.5.1 超参数示例

超参数是“元”级别的参数——与模型参数不同，它们不用于进行预测，而是用于控制模型的结构以及模型的训练方式。例如，如果你正在处理词嵌入或者一个 `RNN`，那么用于表示单词的隐藏单元（维度）的数量就是一个重要的超参数。使用的 `RNN` 层数是另一个超参数。除了这两个超参数（隐藏单元和层数）之外，我们在第九章中介绍的 `Transformer` 模型还有一些其他参数，比如注意力头的数量和前馈网络的维度。甚至你使用的架构类型，例如 `RNN` 与 `Transformer`，也可以被视为一个超参数。

此外，您使用的优化算法也可能有超参数。例如，在许多机器学习设置中最重要的超参数之一——学习率（第 9.3.3 节），确定了每个优化步骤中调整模型参数的程度。迭代次数（通过训练数据集的次数）也是一个重要的超参数。

到目前为止，我们对这些超参数几乎没有给予任何关注，更不用说优化它们了。然而，超参数对机器学习模型的性能有着巨大的影响。事实上，许多机器学习模型都有一个“甜蜜点”超参数，使它们最有效，而使用超参数集在这个点之外可能会使模型表现不佳。

许多机器学习从业者通过手动调整超参数来调整超参数。这意味着你从一组看起来合理的超参数开始，并在验证集上测量模型的性能。然后，您稍微改变一个或多个超参数，并再次测量性能。您重复这个过程几次，直到达到“高原”，在这里任何超参数的更改都只提供了边际改进。

这种手动调整方法的一个问题是它是缓慢和随意的。假设你从一组超参数开始。你如何知道接下来应该调整哪些参数，以及多少？你如何知道何时停止？如果你有调整广泛的机器学习模型的经验，你可能对这些模型如何响应某些超参数更改有一些“直觉”，但如果没有，那就像在黑暗中射击一样。超参数调整是一个非常重要的主题，机器学习研究人员一直致力于寻找更好和更有组织的方法来优化它们。

### 10.5.2 网格搜索 vs. 随机搜索

我们明白手动优化超参数效率低下，但是我们应该如何进行优化呢？我们有两种更有组织的调整超参数的方式——网格搜索和随机搜索。

在*网格搜索*中，你只需尝试优化的超参数值的每种可能组合。例如，假设你的模型只有两个超参数——`RNN` 层数和嵌入维度。你首先为这两个超参数定义合理的范围，例如，层数为 `[1, 2, 3]`，维度为 `[128, 256, 512]`。然后，网格搜索会对每种组合进行模型验证性能的测量——`(1, 128)`, `(1, 256)`, `(1, 512)`, `(2, 128)`, . . . , `(3, 512)`——并简单选择表现最佳的组合。如果你将这些组合绘制在二维图上，它看起来像一个网格（见图 10.15 的示例），这就是为什么称之为*网格搜索*。

网格搜索是优化超参数的一种简单直观的方式。然而，如果你有很多超参数和/或它们的范围很大，这种方法就会失控。可能的组合数量是指数级的，这使得在合理的时间内探索所有组合变得不可能。

![CH10_F15_Hagiwara](img/CH10_F15_Hagiwara.png)

图 10.15 网格搜索与随机搜索的超参数调优比较。（摘自 Bergstra 和 Bengio，2012；[`www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf.`](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)）

比*网格搜索*更好的替代方案是*随机搜索*。在随机搜索中，你不是尝试每种可能的超参数值的组合，而是随机抽样这些值，并在指定数量的组合（称为*试验*）上测量模型的性能。例如，在上述示例中，随机搜索可以选择 `(2, 87)`, `(1, 339)`, `(2, 101)`, `(3, 254)` 等，直到达到指定数量的试验为止。请参见图 10.15 的示例（右侧）。

除非你的超参数搜索空间非常小（就像第一个示例一样），如果你想要高效地优化超参数，通常建议使用随机搜索而不是网格搜索。为什么？在许多机器学习设置中，并非每个超参数都是相等的——通常只有少数几个超参数实际上对性能有影响，而其他许多超参数则不然。网格搜索会浪费大量计算资源来寻找并不真正重要的超参数组合，同时无法详细探索那些真正重要的少数超参数（图 10.15，左侧）。另一方面，随机搜索可以在性能重要的轴上探索许多可能的点（图 10.15，右侧）。请注意，随机搜索可以通过在相同的试验数量下在 x 轴上探索更多点来找到更好的模型（总共九个试验）。

### 10.5.3 使用 Optuna 进行超参数调优

好的，我们已经介绍了一些调整超参数的方法，包括手动、网格和随机搜索，但是在实践中应该如何实现呢？你可以随时编写自己的 `for` 循环（或者在网格搜索的情况下是 `for-loops`），尽管如果你需要为每个模型和任务编写这种样板代码，这将很快变得令人厌倦。

超参数优化是一个普遍的主题，许多机器学习研究人员和工程师一直在致力于改进算法和软件库。例如，`AllenNLP` 有自己的库叫做*Allentune*（[`github.com/allenai/allentune`](https://github.com/allenai/allentune)），你可以很容易地将其与 `AllenNLP` 的训练流程集成起来。然而，在本节的剩余部分中，我将介绍另一个超参数调整库叫做*Optuna*（[`optuna.org/`](https://optuna.org/)），并展示如何将其与 `AllenNLP` 一起使用以优化你的超参数。`Optuna` 实现了最先进的算法，可以高效地搜索最优超参数，并与包括 `TensorFlow`、`PyTorch` 和 `AllenNLP` 在内的广泛的机器学习框架集成。

首先，我们假设你已经安装了 `AllenNLP`（1.0.0+）和 `AllenNLP` 的 `Optuna` 插件。你可以通过运行以下命令来安装它们：

```py
pip install allennlp
pip install allennlp_optuna
```

此外，根据官方文档的指示（[`github.com/himkt/allennlp -optuna`](https://github.com/himkt/allennlp-optuna)），你需要运行下面的代码来注册 `AllenNLP` 的插件：

```py
echo 'allennlp_optuna' >> .allennlp_plugins
```

我们将使用第二章中构建的基于 `LSTM` 的分类器对斯坦福情感树库数据集进行分类。你可以在书的代码库中找到 `AllenNLP` 的配置文件（[`www.realworldnlpbook.com/ch10.html#config`](http://www.realworldnlpbook.com/ch10.html#config)）。注意，你需要引用变量（`std.extVar`）以便 `Optuna` 可以控制参数。具体来说，你需要在配置文件的开头定义它们：

```py
local embedding_dim = std.parseJson(std.extVar('embedding_dim'));
local hidden_dim = std.parseJson(std.extVar('hidden_dim'));
local lr = std.parseJson(std.extVar('lr'));
```

然后，你需要告诉 `Optuna` 要优化哪些参数。你可以通过编写一个 JSON 文件（`hparams.json` ([`www.realworldnlpbook.com/ch10.html# hparams`](http://www.realworldnlpbook.com/ch10.html#hparams)）来实现这一点。你需要指定你希望 `Optuna` 优化的每个超参数及其类型和范围，如下所示：

```py
[
    {
        "type": "int",
        "attributes": {
            "name": "embedding_dim",
            "low": 64,
            "high": 256
        }
    },
    {
        "type": "int",
        "attributes": {
            "name": "hidden_dim",
            "low": 64,
            "high": 256
        }
    },
    {
        "type": "float",
        "attributes": {
            "name": "lr",
            "low": 1e-4,
            "high": 1e-1,
            "log": true
        }
    }
]
```

接下来，调用这个命令来开始优化：

```py
allennlp tune \
    examples/tuning/sst_classifier.jsonnet \
    examples/tuning/hparams.json \
    --include-package examples \
    --serialization-dir result \
    --study-name sst-lstm \
    --n-trials 20 \
    --metrics best_validation_accuracy \
    --direction maximize
```

注意我们正在运行 20 次试验（`--n-trials`），以最大化验证准确性（`--metrics best_validation_accuracy`）作为度量标准（`--direction maximize`）。如果你没有指定度量标准和方向，`Optuna` 默认尝试最小化验证损失。

这将需要一些时间，但是在所有试验完成后，你将看到以下优化的一行摘要：

```py
Trial 19 finished with value: 0.3469573115349682 and parameters: {'embedding_dim': 120, 'hidden_dim': 82, 'lr': 0.00011044322486693224}. Best is trial 14 with value: 0.3869209809264305.
```

最后，`Optuna` 支持广泛的优化结果可视化，包括非常好的等高线图（[`www.realworldnlpbook.com/ch10.html# contour)`](http://www.realworldnlpbook.com/ch10.html#contour)，但在这里我们将简单地使用其基于 Web 的仪表板快速检查优化过程。你只需要按照以下命令从命令行调用其仪表板：

```py
optuna dashboard --study-name sst-lstm --storage sqlite:///allennlp_optuna.db
```

现在，你可以访问 `http://localhost:5006/dashboard` 来查看仪表板，如图 10.16 所示。

![CH10_F16_Hagiwara](img/CH10_F16_Hagiwara.png)

图 10.16 Optuna 仪表板显示了每个试验的参数评估指标。

从这个仪表板上，你不仅可以迅速看到最优试验是第 14 次试验，而且可以看到每次试验的最优超参数。

## 摘要

+   实例被排序、填充和批量化以进行更有效的计算。

+   子单词分词算法（如 `BPE`）将单词拆分成比单词更小的单元，以减轻神经网络模型中的词汇外问题。

+   正则化（如 `L2` 和 `dropout`）是一种用于鼓励机器学习中模型简单性和可泛化性的技术。

+   你可以使用数据上采样、下采样或损失权重来解决数据不平衡问题。

+   超参数是关于模型或训练算法的参数。可以通过手动、网格或随机搜索进行优化。更好的是，使用超参数优化库，如 `Optuna`，它与 `AllenNLP` 集成得很容易。

^(1.)Wu 等人，“谷歌神经机器翻译系统：填补人机翻译之间的差距”（2016）。[`arxiv.org/abs/1609.08144`](https://arxiv.org/abs/1609.08144)。

^(2.)Kudo，“Subword Regularization：使用多个子单词提高神经网络翻译模型”（2018）。[`arxiv.org/abs/1804.10959`](https://arxiv.org/abs/1804.10959)。

^(3.)Sennrich 等人，“使用子单词单元进行稀有词的神经机器翻译”（2016）。[`arxiv.org/abs/1508.07909`](https://arxiv.org/abs/1508.07909)。

^(4.)参见[`www.derczynski.com/papers/archive/BPE_Gage.pdf`](https://www.derczynski.com/papers/archive/BPE_Gage.pdf)。

^(5.)Chawla 等人，“SMOTE：合成少数类过采样技术”（2002）。[`arxiv.org/abs/1106.1813`](https://arxiv.org/abs/1106.1813)。
