

# 11 策略梯度与 actor-critic 方法

在本章

+ 你将了解一类可以直接优化其性能的深度强化学习方法，而无需价值函数。
+ 你将学习如何使用价值函数使这些算法变得更好。
+ 你将实现使用多个进程同时进行以实现非常快速学习的深度强化学习算法。

没有什么比逆境更好。每一次失败、每一次心碎、每一次损失，都包含着自己的种子，自己的教训，告诉你如何在下一次做得更好。

—— 马尔科姆·X 美国穆斯林牧师和人权活动家

在本书中，我们探讨了在价值函数的帮助下可以找到最优和近似最优策略的方法。然而，所有这些算法在学习价值函数时，我们需要的却是策略。

在本章中，我们探索光谱的另一端以及中间部分的内容。我们开始探讨直接优化策略的方法。这些方法被称为*基于策略*或*策略梯度*方法，它们参数化一个策略并调整它以最大化预期回报。

在介绍了基础策略梯度方法之后，我们探索了一类结合的方法，这些方法同时学习策略和价值函数。这些方法被称为 actor-critic，因为选择动作的策略可以被视为 actor，而评估策略的价值函数可以被视为 critic。actor-critic 方法在许多深度强化学习基准测试中通常比单独的基于值或策略梯度方法表现更好。了解这些方法可以使你应对更具挑战性的问题。

这些方法结合了你在前三章中学到的关于学习价值函数的知识，以及在本章第一部分学到的关于学习策略的知识。actor-critic 方法在多样化的深度强化学习基准测试中通常能取得最先进的性能。

![图片](img/11_01.png)

基于策略、基于值和 actor-critic 方法

## REINFORCE：基于结果的策略学习

在本节中，我们开始介绍基于策略的方法的使用动机，首先是一个介绍；然后我们讨论使用这类方法时可以期待的一些优点；最后，我们介绍了最简单的策略梯度算法，称为*rEINFORCE*。

### 策略梯度方法简介

我首先想强调的是，在策略梯度方法中，与基于值的方法不同，我们试图最大化性能目标。在基于值的方法中，主要关注的是学习评估策略。为此，目标是最小化预测值和目标值之间的损失。更具体地说，我们的目标是匹配给定策略的真实动作值函数，因此我们参数化了一个值函数，并最小化了预测值和目标值之间的均方误差。请注意，我们没有真正的目标值，而是使用蒙特卡洛方法中的实际回报或自举方法中的预测回报。

相反，在基于策略的方法中，目标是最大化参数化策略的性能，因此我们执行梯度上升（或对负性能执行常规梯度下降）。很明显，代理的性能是从初始状态期望的总折现奖励，这与从给定策略的所有初始状态的期望状态值函数是同一回事。

| ![数学展示](img/icons_Math.png) | 展示数学：基于值的方法与基于策略的方法的目标 |
| --- | --- |
|  | ![侧边栏 01](img/11_01_Sidebar01.png) |
| ŘŁ | 带有强化学习口音的基于值的方法与基于策略的方法与策略梯度方法与演员-评论家方法 |
|  | **基于值的方法：**指的是学习值函数并仅学习值函数的算法。Q-learning、SARSA、DQN 等都是基于值的方法。**基于策略的方法：**指的是优化策略的广泛算法，包括如遗传算法这样的黑盒优化方法。**策略梯度方法：**指的是在参数化策略的性能梯度上解决优化问题的方法，你将在本章学习这些方法。**演员-评论家方法：**指的是学习策略和值函数的方法，主要是在值函数通过自举学习并用作随机策略梯度评分的情况下。你将在本章和下一章学习这些方法。 |

### 策略梯度方法的优势

学习参数化策略的主要优势是，策略现在可以是任何可学习的函数。在基于值的方法中，我们处理离散动作空间，主要是因为我们计算动作的最大值。在高维动作空间中，这个最大值可能过于昂贵。此外，在连续动作空间的情况下，基于值的方法受到严重限制。

另一方面，基于策略的方法可以更容易地学习随机策略，这反过来又具有多个额外的优点。首先，学习随机策略意味着在部分可观察环境中具有更好的性能。直觉是，因为我们可以学习任意动作的概率，所以智能体对马尔可夫假设的依赖性较低。例如，如果智能体无法区分几个状态及其发出的观察结果，最佳策略通常是按照特定概率随机行动。

![11_02.png](img/11_02.png)

学习随机策略可能帮助我们摆脱困境

有趣的是，尽管我们在学习随机策略，但没有任何东西阻止学习算法接近确定性策略。这与基于价值的方法不同，在训练过程中，我们必须通过某种概率强制进行探索以确保最优性。在具有随机策略的基于策略的方法中，探索被嵌入到学习函数中，并且在训练过程中对于给定状态收敛到确定性策略是可能的。

学习随机策略的另一个优点是，与表示价值函数相比，函数逼近表示策略可能更加直接。有时价值函数包含的信息过多，对于真正需要的信息来说是不必要的。可能计算状态或状态-动作对的精确价值是复杂或不必要的。

![11_03.png](img/11_03.png)

学习策略可能是一个更容易、更易于推广的问题来解决

最后要提到的一个优点是，由于策略是用连续值参数化的，动作概率会随着学习参数的变化而平滑地变化。因此，基于策略的方法通常具有更好的收敛特性。正如你从之前的章节中记得的那样，基于价值的方法容易产生振荡甚至发散。其中一个原因是价值函数空间中微小的变化可能意味着动作空间中显著的变化。动作的显著差异可以产生全新的轨迹，因此造成不稳定性。

在基于价值的方法中，我们使用激进的算子来改变价值函数；我们对 Q 值估计取最大值。在基于策略的方法中，我们相反地跟随随机策略相对于梯度，这只会逐渐和平滑地改变动作。如果你直接跟随策略的梯度，你将保证收敛到至少一个局部最优解。

| ![Python](img/icons_Python.png) | 我会说 Python 随机策略的离散动作空间 1/2 |
| --- | --- |
|  | 

```
class FCDAP(nn.Module): ①
    def __init__(self,
               input_dim,                                ②
                 output_dim,
                 hidden_dims=(32,32),
               init_std=1,
                 activation_fc=F.relu):
        super(FCDAP, self).__init__()
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear( ③
            input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList() ④
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(
                hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear( ⑤
            hidden_dims[-1], output_dim)
    def forward(self, state): ⑥
        x = state
        if not isinstance(x, torch.Tensor): ⑦
            x = torch.tensor(x, dtype=torch.float32)
            x = x.unsqueeze(0)
        x = self.activation_fc(self.input_layer(x)) ⑧
        for hidden_layer in self.hidden_layers: ⑨
            x = self.activation_fc(hidden_layer(x))
        return self.output_layer(x) ⑩
```

① 这个类，`FCDAP` 代表全连接离散动作策略。② 参数允许你指定一个全连接架构、激活函数以及权重和偏差的最大幅度。③ `__init__` 函数在输入和第一个隐藏层之间创建线性连接。④ 然后，它创建所有隐藏层之间的连接。⑤ 最后，它将最终隐藏层连接到输出节点，创建输出层。⑥ 这里我们有处理正向功能的方法。⑦ 首先，我们确保状态是我们期望的类型和形状，然后我们才能将其通过网络。⑧ 接着，我们将格式正确的状态传递到输入层，然后通过激活函数。⑨ 然后，我们将第一个激活的输出传递到一系列隐藏层和相应的激活函数。⑩ 最后，我们获得输出，即 logits，对动作的偏好。 |

| ![](img/icons_Python.png) | 我会说 Python 离散动作空间的随机策略 1/2 |
| --- | --- |
|  | 

```
       return self.output_layer(x) ⑪
    def full_pass(self, state): ⑫
        logits = self.forward(state) ⑬
        dist = torch.distributions.Categorical(logits=logits) ⑭
        action = dist.sample() ⑭
        logpa = dist.log_prob(action).unsqueeze(-1) ⑮
        entropy = dist.entropy().unsqueeze(-1) ⑯
        is_exploratory = action != np.argmax( \ ⑰
                                       logits.detach().numpy())
        return action.item(), is_exploratory.item(), \        ⑱
                                                 logpa, entropy
    def select_action(self, state): ⑲
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item()
    def select_greedy_action(self, state): ⑳
        logits = self.forward(state)
        return np.argmax(logits.detach().numpy())
```

⑪ 这一行重复了上一页的最后一行。⑫ 这里我们进行完整的正向传播。这是一个方便的函数，用于获取概率、动作以及训练所需的一切。⑬ 正向传播返回的是 logits，即对动作的偏好。⑭ 接下来，我们从概率分布中采样动作。⑮ 然后，计算该动作的对数概率并将其格式化为训练格式。⑯ 这里我们计算策略的熵。⑰ 在这里，为了统计，我们确定所选策略是探索性的还是不是。⑱ 最后，我们返回一个可以直接传递到环境中的动作，一个标志表示动作是否是探索性的，动作的对数概率以及策略的熵。⑲ 这是一个辅助函数，当我们只需要采样动作时使用。⑳ 这一个是为了根据策略选择贪婪动作。 |

### 直接学习策略

直接优化策略的主要优势是，嗯，这是正确的目标。我们学习一个直接优化价值函数的策略，而不学习价值函数，也不考虑环境的动态。这是如何可能的？让我来展示给你看。

| ![](img/icons_Math.png) | 展示数学推导策略梯度 |
| --- | --- |
|  | ![](img/11_03_Sidebar05.png) |

### 降低策略梯度的方差

有一种方法可以计算策略梯度，而无需了解任何关于环境转移函数的信息。这个算法增加了轨迹中所有动作的 log 概率，与完整回报的优良程度成正比。换句话说，我们首先收集一个完整的轨迹并计算完整的折现回报，然后使用这个分数来加权轨迹中每个动作的对数概率：*A*[*t*][*t*]，*A*[*t*+1]，...，*A*[T–1]。

![](img/11_04.png)

让我们只使用由动作产生的奖励

| ![数学图标](img/icons_Math.png) | 展示数学 Reducing the variance of the policy gradient |
| --- | --- |
|  | ![示例图片](img/11_04_Sidebar06.png) |
| 0001 | 一点历史介绍 REINFORCE 算法 |
|  | Ronald J. Williams 于 1992 年在一篇题为“Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning”的论文中介绍了 REINFORCE 算法系列。1986 年，他与 Geoffrey Hinton 等人合著了一篇名为“Learning representations by back-propagating errors”的论文，这引发了当时人工神经网络（ANN）研究的发展。 |
| ![Python 图标](img/icons_Python.png) | 我会说 Python REINFORCE |

