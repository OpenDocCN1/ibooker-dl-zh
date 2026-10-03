
```py
# In[47]:
class NetDropout(nn.Module):
    def __init__(self, n_chans1=32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv1_dropout = nn.Dropout2d(p=0.4)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3,
                               padding=1)
        self.conv2_dropout = nn.Dropout2d(p=0.4)
        self.fc1 = nn.Linear(8 * 8 * n_chans1 // 2, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = self.conv1_dropout(out)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = self.conv2_dropout(out)
        out = out.view(-1, 8 * 8 * self.n_chans1 // 2)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out
```

注意，在训练期间通常会激活 dropout，而在生产中评估经过训练的模型时，会绕过 dropout，或者等效地将概率分配为零。这通过`Dropout`模块的`train`属性来控制。请记住，PyTorch 允许我们通过调用在两种模式之间切换

```py
model.train()
```

或

```py
model.eval()
```

在任何`nn.Model`子类上。调用将自动复制到子模块，因此如果其中包含`Dropout`，它将在后续的前向和后向传递中相应地行为。

## 保持激活在适当范围内：批量归一化

在 2015 年，谷歌的 Sergey Ioffe 和 Christian Szegedy 发表了另一篇具有开创性意义的论文，名为“批量归一化：通过减少内部协变量转移加速深度网络训练”（[`arxiv.org/abs/1502.03167`](https://arxiv.org/abs/1502.03167)）。该论文描述了一种对训练有多种有益影响的技术：使我们能够增加学习率，使训练不那么依赖初始化并充当正则化器，从而代替了 dropout。

批量归一化背后的主要思想是重新缩放网络的激活输入，以便小批量具有某种理想的分布。回顾学习的机制和非线性激活函数的作用，这有助于避免输入到激活函数过于饱和部分，从而杀死梯度并减慢训练速度。

在实际操作中，批量归一化使用小批量样本中在该中间位置收集的均值和标准差来移位和缩放中间输入。正则化效果是因为模型始终将单个样本及其下游激活视为根据随机提取的小批量样本的统计数据而移位和缩放。这本身就是一种*原则性*的增强。论文的作者建议使用批量归一化消除或至少减轻了对 dropout 的需求。

在 PyTorch 中，批量归一化通过`nn.BatchNorm1D`、`nn.BatchNorm2d`和`nn.BatchNorm3d`模块提供，取决于输入的维度。由于批量归一化的目的是重新缩放激活的输入，自然的位置是在线性变换（在这种情况下是卷积）和激活之后，如下所示：

```py
# In[49]:
class NetBatchNorm(nn.Module):
    def __init__(self, n_chans1=32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=n_chans1)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3,
                               padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=n_chans1 // 2)
        self.fc1 = nn.Linear(8 * 8 * n_chans1 // 2, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = self.conv1_batchnorm(self.conv1(x))
        out = F.max_pool2d(torch.tanh(out), 2)
        out = self.conv2_batchnorm(self.conv2(out))
        out = F.max_pool2d(torch.tanh(out), 2)
        out = out.view(-1, 8 * 8 * self.n_chans1 // 2)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out
```

与 dropout 一样，批量归一化在训练和推断期间需要有不同的行为。实际上，在推断时，我们希望避免特定输入的输出依赖于我们向模型呈现的其他输入的统计信息。因此，我们需要一种方法来进行归一化，但这次是一次性固定归一化参数。

当处理小批量时，除了估计当前小批量的均值和标准差之外，PyTorch 还更新代表整个数据集的均值和标准差的运行估计，作为近似值。这样，当用户指定时

```py
model.eval()
```

如果模型包含批量归一化模块，则冻结运行估计并用于归一化。要解冻运行估计并返回使用小批量统计信息，我们调用`model.train()`，就像我们对待 dropout 一样。

### 深入学习更复杂的结构：深度

早些时候，我们谈到宽度作为第一个要处理的维度，以使模型更大，从某种意义上说，更有能力。第二个基本维度显然是*深度*。由于这是一本深度学习书，深度是我们应该关注的东西。毕竟，深层模型总是比浅层模型更好，不是吗？嗯，这取决于情况。随着深度增加，网络能够逼近的函数的复杂性通常会增加。就计算机视觉而言，一个较浅的网络可以识别照片中的人的形状，而一个更深的网络可以识别人、头部上半部分的脸和脸部内的嘴巴。深度使模型能够处理分层信息，当我们需要理解上下文以便对某些输入进行分析时。

还有另一种思考深度的方式：增加深度与增加网络在处理输入时能够执行的操作序列的长度有关。这种观点--一个执行顺序操作以完成任务的深度网络--对于习惯于将算法视为“找到人的边界，寻找边界上方的头部，寻找头部内的嘴巴”等操作序列的软件开发人员可能是迷人的。

#### 跳过连接

深度带来了一些额外的挑战，这些挑战阻碍了深度学习模型在 2015 年之前达到 20 层或更多层。增加模型的深度通常会使训练更难收敛。让我们回顾反向传播，并在非常深的网络环境中思考一下。损失函数对参数的导数，特别是早期层中的导数，需要乘以许多其他数字，这些数字来自于损失和参数之间的导数操作链。这些被乘以的数字可能很小，生成越来越小的数字，或者很大，由于浮点近似而吞噬较小的数字。归根结底，长链的乘法将使参数对梯度的贡献*消失*，导致该层的训练无效，因为该参数和类似的其他参数将无法得到适当更新。

2015 年 12 月，Kaiming He 和合著者提出了*残差网络*（ResNets），这是一种使用简单技巧的架构，使得非常深的网络能够成功训练（[`arxiv.org/abs/1512.03385`](https://arxiv.org/abs/1512.03385)）。该工作为从几十层到 100 层深度的网络打开了大门，超越了当时计算机视觉基准问题的最新技术。我们在第二章中使用预训练模型时遇到了残差网络。我们提到的技巧是：使用*跳跃连接*来绕过一组层，如图 8.11 所示。

![](img/CH08_F11_Stevens2_GS.png)

图 8.11 我们具有三个卷积层的网络架构。跳跃连接是`NetRes`与`NetDepth`的区别所在。

跳跃连接只是将输入添加到一组层的输出中。这正是在 PyTorch 中所做的。让我们向我们简单的卷积模型添加一层，并让我们使用 ReLU 作为激活函数。带有额外一层的香草模块如下所示：

```py
# In[51]:
class NetDepth(nn.Module):
    def __init__(self, n_chans1=32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv2d(n_chans1 // 2, n_chans1 // 2,
                               kernel_size=3, padding=1)
        self.fc1 = nn.Linear(4 * 4 * n_chans1 // 2, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = F.max_pool2d(torch.relu(self.conv2(out)), 2)
        out = F.max_pool2d(torch.relu(self.conv3(out)), 2)
        out = out.view(-1, 4 * 4 * self.n_chans1 // 2)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out
```

向这个模型添加一个类 ResNet 的跳跃连接相当于将第一层的输出添加到第三层的输入中的`forward`函数中：

```py
# In[53]:
class NetRes(nn.Module):
    def __init__(self, n_chans1=32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv2d(n_chans1 // 2, n_chans1 // 2,
                               kernel_size=3, padding=1)
        self.fc1 = nn.Linear(4 * 4 * n_chans1 // 2, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = F.max_pool2d(torch.relu(self.conv2(out)), 2)
        out1 = out
        out = F.max_pool2d(torch.relu(self.conv3(out)) + out1, 2)
        out = out.view(-1, 4 * 4 * self.n_chans1 // 2)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out
```

换句话说，我们将第一个激活的输出用作最后一个的输入，除了标准的前馈路径。这也被称为*恒等映射*。那么，这如何缓解我们之前提到的梯度消失问题呢？

想想反向传播，我们可以欣赏到在深度网络中的跳跃连接，或者一系列跳跃连接，为深层参数到损失创建了一条直接路径。这使得它们对损失的梯度贡献更直接，因为对这些参数的损失的偏导数有机会不被一长串其他操作相乘。

已经观察到跳跃连接对收敛特别是在训练的初始阶段有益。此外，深度残差网络的损失景观比相同深度和宽度的前馈网络要平滑得多。

值得注意的是，当 ResNets 出现时，跳跃连接并不是新鲜事物。Highway 网络和 U-Net 使用了各种形式的跳跃连接。然而，ResNets 使用跳跃连接的方式使得深度大于 100 的模型易于训练。

自 ResNets 出现以来，其他架构已经将跳跃连接提升到了一个全新水平。特别是 DenseNet，提出通过跳跃连接将每一层与数个下游层连接起来，以较少的参数实现了最先进的结果。到目前为止，我们知道如何实现类似 DenseNets 的东西：只需将早期中间输出算术地添加到下游中间输出中。

#### 在 PyTorch 中构建非常深的模型

我们谈到了在卷积神经网络中超过 100 层。我们如何在 PyTorch 中构建该网络而不至于在过程中迷失方向？标准策略是定义一个构建块，例如`(Conv2d，ReLU，Conv2d) + 跳跃连接`块，然后在`for`循环中动态构建网络。让我们看看实践中是如何完成的。我们将创建图 8.12 中所示的网络。

![](img/CH08_F12_Stevens2_GS.png)

图 8.12 我们带有残差连接的深度架构。在左侧，我们定义了一个简单的残差块。如右侧所示，我们将其用作网络中的构建块。

我们首先创建一个模块子类，其唯一任务是为一个*块*提供计算--也就是说，一组卷积、激活和跳跃连接：

```py
# In[55]:
class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans, kernel_size=3,
                              padding=1, bias=False)            # ❶
        self.batch_norm = nn.BatchNorm2d(num_features=n_chans)
        torch.nn.init.kaiming_normal_(self.conv.weight,
                                      nonlinearity='relu')      # ❷
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)
        torch.nn.init.zeros_(self.batch_norm.bias)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        return out + x
```

❶ BatchNorm 层会抵消偏差的影响，因此通常会被省略。

❷ 使用自定义初始化。`kaiming_normal_`使用正态随机元素进行初始化，标准差与 ResNet 论文中计算的一致。批量归一化被初始化为产生初始具有 0 均值和 0.5 方差的输出分布。

由于我们计划生成一个深度模型，我们在块中包含了批量归一化，因为这将有助于防止训练过程中梯度消失。我们现在想生成一个包含 100 个块的网络。这意味着我们需要做一些严肃的剪切和粘贴吗？一点也不；我们已经有了想象这个模型可能是什么样子的所有要素。

首先，在`__init__`中，我们创建包含一系列`ResBlock`实例的`nn.Sequential`。`nn.Sequential`将确保一个块的输出被用作下一个块的输入。它还将确保块中的所有参数对`Net`可见。然后，在`forward`中，我们只需调用顺序遍历 100 个块并生成输出：
