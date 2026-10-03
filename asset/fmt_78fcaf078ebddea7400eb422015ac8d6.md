
# 10 NEAT：增强拓扑结构的神经进化

本章涵盖

+ 构建进化的增强拓扑结构网络
+ 可视化增强拓扑结构的神经进化网络
+ 锻炼增强拓扑结构的神经进化能力
+ 锻炼增强拓扑结构的神经进化以对图像进行分类
+ 揭示物种形成在神经进化中的作用

在过去几章的内容中，我们探讨了生成对抗网络和自编码器网络的进化优化。与之前的章节类似，在这些练习中，我们在深度学习网络周围叠加或包裹了进化优化。在本章中，我们脱离了 Python 中的分布式进化算法（DEAP）和 Keras，来探索一个名为*增强拓扑结构的神经进化*（NEAT）的神经进化框架。

NEAT 是由肯·斯坦利在 2002 年开发的，当时他在德克萨斯大学奥斯汀分校。当时，遗传算法（进化计算）和深度学习（高级神经网络）是平等的，并且都被认为是人工智能的下一个大事物。斯坦利的 NEAT 框架吸引了众多人的注意，因为它将神经网络与进化相结合，不仅优化了超参数、权重参数和架构，而且还优化了实际的神经网络连接本身。

图 10.1 显示了常规深度学习网络和进化 NEAT 网络之间的比较。在图中，已经添加和删除了新的连接，并且在进化 NEAT 网络中删除和/或改变了节点的位置。注意这与我们之前仅仅改变深度学习连接层中节点数量的努力有何不同。

![](img/CH10_F01_Lanham.png)

图 10.1 深度学习网络和 NEAT 网络的比较

NEAT 通过允许网络的神经连接和节点数量进化，将进化深度学习网络的概念推向了极致。由于 NEAT 还内部进化每个节点的网络权重，因此消除了使用微积分计算误差反向传播的复杂性。这使得 NEAT 网络能够进化到一些复杂的交织和互联的图。它甚至允许网络进化循环连接，正如我们在下一章深入探讨 NEAT 时所看到的。

在本章中，我们探讨了 NEAT 的基本原理，并深入研究了名为 NEAT-Python 的 Python 实现。这个框架很好地抽象了设置进化和深度学习系统的细节。以下列表标识了 NEAT 的每个组件，这些组件涵盖了本书中使用的许多其他方法：

+ *参数的神经进化*——NEAT 中的权重和参数作为系统的一部分进化。在第六章中，我们使用进化来改变神经网络的权重。
+ *架构的神经进化*——NEAT 进化网络层，结构本身通过进化进行适应。参见第七章，其中我们介绍了使用遗传算法进行神经进化架构的内容，以了解更多相关信息。
+ *超参数优化*——NEAT 不使用学习率、优化器或其他标准深度学习辅助工具。因此，它不需要优化这些参数。然而，正如我们将看到的，NEAT 引入了几个超参数来控制网络进化。

在下一节中，我们从基础知识开始，开始探索 NEAT。

## 10.1 使用 NEAT-Python 探索 NEAT

本章中我们首先考虑的笔记本设置了一个 NEAT 网络来解决经典的一阶 XOR 问题。NEAT 提供了几个配置选项，这个第一个练习展示了其中一些最重要的选项。打开您的网络浏览器，让我们开始查看一些代码。

在 Google Colab 中打开 `EDL_10_1_NEAT_XOR.ipynb` 笔记本。如需帮助，请参阅附录。通过选择菜单中的“运行”>“运行所有”来运行笔记本中的所有单元格。

NEAT-Python 仍在积极开发中，截至写作时，最佳实践是直接从 GitHub 仓库安装它，而不是从 PyPi 包安装。笔记本中的第一个代码单元格使用 `pip` 在第一行执行此操作，如下所示。然后，下一行使用 `import neat` 导入包。

列表 10.1 `EDL_10_1_NEAT_XOR.ipynb`：安装 NEAT-Python

```
!pip install 
➥ git+https://github.com/CodeReclaimers/neat-python.git   ❶

#then import 
import neat                                                ❷
```

❶ 从 GitHub 仓库安装

❷ 导入包

滚动到下一个单元格，显示数据的设置，分为 `xor_inputs`（X）和 `xor_outputs`（Y），如下所示。

列表 10.2 `EDL_10_1_NEAT_XOR.ipynb`：数据设置

```
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), 
➥ (1.0, 1.0)]                                     ❶
xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     
➥ (0.0,)]                                         ❷
```

❶ 输入：X

❷ 输出：Y

接下来，正如我们之前多次做的那样，我们构建一个评估函数来计算进化 NEAT 网络的“适应度”。这个概念现在应该相当熟悉了，代码与之前的练习相似。与 DEAP 不同，评估函数接受一组称为“基因组”的“基因”序列。函数遍历“基因组”，并为每个计算“适应度”，首先分配一些最大“适应度”，如列表 10.3 所示。然后，它使用 `FeedForwardNetwork.create` 函数创建一个经典的前馈网络的新版本，传入 `genome` 和配置。然后，通过使用 `net.activate` 函数对所有数据进行测试，传入 `X` 或 `xi` 值之一，并产生 `Y` 输出。每次激活输入后，输出与预期的输出 `xo` 进行比较，从 `genome.fitness` 中减去均方误差（MSE）。最后，`eval_genomes` 函数的结果更新了每个进化“基因组”的当前“适应度”。

列表 10.3 `EDL_10_1_NEAT_XOR.ipynb`：创建评估函数

```
def eval_genomes(genomes, config):
  for genome_id, genome in genomes:                 ❶
    genome.fitness = 4.0                            ❷
    net = neat.nn.FeedForwardNetwork.create
    ➥ (genome, config)                             ❸
    for xi, xo in zip(xor_inputs, xor_outputs):     ❹
      output = net.activate(xi)            
      genome.fitness -= (output[0] - xo[0]) ** 2    ❺
```

❶ 遍历基因组

❷ 分配最大适应度

❸ 从基因组创建 NEAT 网络

❹ 遍历数据

❺ 计算均方误差（MSE）然后从适应度中减去

下一个单元格设置了用于配置和运行 NEAT 代码的配置文件。NEAT-Python（NP）主要是由配置驱动的，你可以更改或微调几个选项。为了保持简单，我们只回顾列表 10.4 中的主要选项：前两个部分。`config` 首先设置 `fitness` 标准、`fitness` 阈值、种群大小和 `reset` 选项。之后，设置默认的 `genome` 配置，首先是激活选项，在这种情况下，只是 `sigmoid` 函数。NEAT 允许你从多个激活函数中选择选项，这些选项用于内部互联节点和输出。

列表 10.4 `EDL_10_1_NEAT_XOR.ipynb`：配置设置

```
%%writefile config                    ❶

[NEAT]                                ❷
fitness_criterion     = max
fitness_threshold     = 3.99
pop_size              = 150           ❸
reset_on_extinction   = False

[DefaultGenome]                       ❹
# node activation options
activation_default      = sigmoid   
activation_mutate_rate  = 0.0
activation_options      = sigmoid     ❺
```

❶ 将单元格内容写入名为 config 的文件

❷ 一般配置参数

❸ 设置要进化的个体数量

❹ 基因配置参数

❺ 默认激活函数

笔记本中的最后一个代码单元格包含我们进化 NEAT 网络所需的所有代码。我们首先回顾列表 10.5 中的前几行。代码首先加载配置并设置基因、繁殖、物种和停滞的基础假设。我们将在本章和下一章中介绍这些默认值。之后，从 `config` 创建 `genome` 种群。然后，向 `population` 对象添加报告器 `StdOutReporter` 以跟踪进化过程。注意 `population` 对象 `p` 是如何成为进化的焦点，以及它与 DEAP 的不同之处。

列表 10.5 `EDL_10_1_NEAT_XOR.ipynb`：设置 NEAT 进化

```
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config')               ❶

p = neat.Population(config)                  ❷

p.add_reporter(neat.StdOutReporter(False))   ❸
```

❶ 从配置文件中加载配置

❷ 创建种群

❸ 在进化过程中添加报告器以查看结果

通过在 `population` 对象上调用 `run` 函数，简单地运行或执行种群的进化，如下所示。进化完成后，代码将打印出获胜者，即具有最佳 `fitness` 的 `genome`，并输出对 XOR 输入的预测。

列表 10.6 `EDL_10_1_NEAT_XOR.ipynb`：进化种群

```
winner = p.run(eval_genomes)                           ❶

print('\nBest genome:\n{!s}'.format(winner))           ❷

print('\nOutput:')                                     ❸
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
for xi, xo in zip(xor_inputs, xor_outputs):
    output = winner_net.activate(xi)
    print(" input {!r}, expected output {!r}, got {!r}".format(xi, xo, 
    ➥ output))
```

❶ 对种群执行进化

❷ 打印出最佳基因

❸ 使用基因预测 XOR 并显示

运行此示例的输出如图 10.2 所示。与 DEAP 不同，NP 使用 `fitness` 阈值的概念来控制进化迭代次数。如果你还记得，在 `config` 设置中，我们将 `fitness_threshold` 设置为 `3.99`（见列表 10.4）。图中还显示了网络配置和权重的文本输出。当然，这并不是容易可视化的东西，但我们将在未来的章节中介绍。在图的下端，你可以看到 XOR 输入被正确预测的程度。

![图片](img/CH10_F02_Lanham.png)

图 10.2 在 XOR 上进化 NEAT 网络的最终输出

这个练习演示了我们可以如何快速设置 NEAT 进化来创建一个能够预测 XOR 函数的网络。正如你所看到的，代码中抽象了很多细节，但希望在这个阶段，你理解了进化是如何被应用的内部工作原理的一些内容。除了增强节点拓扑结构之外，我们之前使用 DEAP 完成了所有内部操作。

### 10.1.1 学习练习

使用以下学习练习来了解更多关于 NEAT 的信息：

1. 在列表 10.4 中更改 `population` 大小（`pop_size`），然后重新运行笔记本。`population` 大小如何影响进化？
2. 在列表 10.4 中减小 `fitness_threshold`，然后重新运行笔记本，看看这会对结果产生什么影响。
3. 更改输入或输出以匹配另一个函数，或者编写一个函数来创建列表 10.2 中的输出。然后，重新运行笔记本以查看近似新函数的结果。

从这个基本介绍开始，我们将在下一节中探索可视化进化的 NEAT 网络看起来是什么样子。

## 10.2 可视化进化的 NEAT 网络

现在我们已经将 NEAT-Python 的基本设置完成，我们可以看看添加一些有用的工具到我们的工具箱中。可视化 NEAT 网络对于理解网络架构是如何形成的非常有用。它还突出了网络在拟合或欠拟合问题上的表现如何。

在本节中，我们采用之前的笔记本示例，并添加了可视化进化的最佳 `genome` 网络的能力。我们还仔细研究了评估 `fitness` 函数是如何开发的。

在 Google Colab 中打开 `EDL_10_2_NEAT_XOR_Visualized.ipynb` 笔记本。如需帮助，请参阅附录。通过选择菜单中的“运行”>“运行所有”来运行笔记本中的所有单元格。

跳到加载 `config` 后的代码单元格开始。所有这些代码通常都在 `population` 类中处理，但我们提取了这个小节，如列表 10.7 所示，以突出构建评估函数。在 NP 中，所有 `genomes` 都需要一个键或唯一标识符，这里我们任意使用 `fred`。然后，根据默认类型从 `config` 创建一个 `genome`——在这个例子中，是一个 `DefaultGenome`。之后，使用 `genome_config` 通过 `genome.configure_new` 配置 `genome`。最后，通过 `FeedForwardNetwork.create` 传递 `genome` 和 `config` 创建了一个新的 `fred` 1.0 随机网络。

列表 10.7 `EDL_10_2_NEAT_XOR_Visualized.ipynb`：创建 `genome` 网络

```
key = "fred"                                              ❶
genome = config.genome_type(key)                          ❷
genome.configure_new(config.genome_config)                ❸

net = neat.nn.FeedForwardNetwork.create(genome, config)   ❹
```

❶ 为基因组分配一个键

❷ 创建基因组类型

❸ 从配置中配置基因组

❹ 从基因组创建前馈网络

接下来，网络 `net` 被 `评估`，使用 `for` 循环遍历数据，从最大 `适应度` 中减去 MSE，如列表 10.8 所示。回想一下，在实际评估函数列表 10.4 中，代码也遍历了整个 `基因组` 的 `种群`。为了简单起见，我们在这里只 `评估` 了 `基因组` `fred`。此代码块的输出显示了网络的输入和输出以及总 `适应度`。

列表 10.8 `EDL_10_2_NEAT_XOR_Visualized.ipynb`：评估 `基因组`

```
fitness = 4                         ❶
for x, y in zip(X, Y):              ❷
  output = net.activate(x)          ❸
  print(output, y)
  fitness -= (output[0]-y[0])**2    ❹

print(fitness)
```

❶ 分配最大适应度值

❷ 遍历 x 和 y 值

❸ 在输入上激活网络

❹ 计算 MSE 然后减去

在设置 `fred` 之后，我们继续到 `draw_net` 函数。此函数已直接从 NEAT 示例中提取，并使用 Graphviz 绘制一个进化的网络。您可以自己查看代码，但在这里我们不会关注具体细节。相反，我们想看看如何调用该函数以及它生成的内容。

下一步是调用 `draw_net` 函数；它首先在字典中命名输入和输出节点。之后，通过传递 `config`、`genome` 和主要节点（输入和输出）的名称来调用 `draw_net` 函数。我们传递值 `True` 以可视化输出，如下所示。

列表 10.9 `EDL_10_2_NEAT_XOR_Visualized.ipynb`：调用 `draw_net` 函数

```
node_names = {-1: 'X1', -2: 'X2', 0: 'Output'}        ❶
draw_net(config, genome, True, node_names=node_names) ❷
```

❶ 命名输入和输出节点。

❷ 使用 True 调用函数以查看。

图 10.3 显示了我们所称的 `fred` 的基本和未进化的 `基因组` 的输出。如图所示，网络是一个非常简单的单节点网络，有两个输入，分别标记为 `X1` 和 `X2`。

![](img/CH10_F03_Lanham.png)

图 10.3 一个初始的 NEAT 网络可视化

到目前为止，NEAT 的 `种群` 应该已经进化，我们可以再次调用 `draw_net` 函数，这次传入获胜的 `基因组` `winner`。在 `种群` 上调用 `run` 会输出获胜者，或最佳 `基因组`。然后，从获胜者创建网络以展示激活。接下来，使用获胜者的 `基因组` 调用 `draw_net` 来可视化网络，如下所示。

列表 10.10 `EDL_10_2_NEAT_XOR_Visualized.ipynb`：可视化获胜 `基因组`
