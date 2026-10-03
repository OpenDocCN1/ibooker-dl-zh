
# REINFORCE 算法

```
class REINFORCE(): ①
    <...>
    def optimize_model(self):
        T = len(self.rewards)
        discounts = np.logspace(0, T, num=T, base=self.gamma, ②
                                endpoint=False)
        returns = np.array( ③
                [np.sum(discounts[:T-t] * self.rewards[t:]) \ ④
                                            for t in range(T)])
        <...>
        policy_loss = -(discounts * returns * \             ⑤⑥
                                    self.logpas).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward() ⑦
        self.policy_optimizer.step()
    def interaction_step(self, state, env): ⑧
        action, is_exploratory, logpa, _ = \
                             self.policy_model.full_pass(state)
        new_state, reward, is_terminal, _ = env.step(action)
        <...>
        return new_state, is_terminal
class REINFORCE():                                            ⑨
    <...>
    def train(self, make_env_fn, make_env_kargs, seed, gamma, ⑩
              max_minutes, max_episodes, goal_mean_100_reward):
        for episode in range(1, max_episodes + 1): ⑪
            state, is_terminal = env.reset(), False ⑫
            <...>                                             ⑫
            self.logpas, self.rewards = [], [] ⑫
            for step in count(): ⑬
                state, is_terminal = \
                             self.interaction_step(state, env)⑭
                if is_terminal:
                    break ⑭
            self.optimize_model() ⑮
    def evaluate(self, eval_policy_model,                     ⑯
                 eval_env, n_episodes=1,
               greedy=True):
        rs = []
        for _ in range(n_episodes):
            <...>
            for _ in count():
                if greedy:
                    a = eval_policy_model.\
                     select_greedy_action(s)
                else:
                  a = eval_policy_model.select_action(s)
                s, r, d, _ = eval_env.step(a)
                <...>
        return np.mean(rs), np.std(rs)
```

① 这就是 REINFORCE 算法。当你看到`<...>`时，这意味着为了简化，代码已被删除。请前往该章节的笔记本以获取完整的代码。② 首先，我们像所有蒙特卡洛方法一样计算折扣。具有这些参数的`logspace`函数返回每一步的伽玛序列；例如，`[1, 0.99, 0.9801, ...]`。③ 接下来，我们计算所有时间步的折扣回报总和。④ 为了强调，这是在每一时间步的回报，从时间步 0 的初始状态到终端 T-1 之前。⑤ 注意这里我们使用的是数学上正确的策略梯度更新，这并不是你通常能找到的。额外的折扣假设我们试图优化从初始状态期望的折扣回报，因此，在场景中较晚的回报会得到折扣。⑥ 这是策略损失；它是选择动作的对数概率，加权之后是选择该动作后获得的回报。注意，由于 PyTorch 默认使用梯度下降，并且我们希望最大化性能，所以我们使用性能的负平均值来翻转函数。把它看作是在性能上进行梯度上升。此外，我们考虑了折扣策略梯度，因此我们将回报乘以折扣。⑦ 在这三个步骤中，我们首先在优化器中将梯度置零，然后进行反向传播，然后沿着梯度的方向进行步进。⑧ 这个函数获取要传递给环境的动作以及所有训练所需的变量。⑨ 仍在探索 REINFORCE 类的功能⑩ 训练方法是训练代理的入口点。⑪ 我们首先通过循环遍历场景。⑫ 对于每个新的场景，我们初始化训练和统计所需的变量。⑬ 然后，对每个时间步执行以下操作。⑭ 首先，我们收集经验，直到达到终端状态。⑮ 然后，我们使用场景中所有时间步的批次运行一个优化步骤。⑯ 我还希望让你看到我在评估期间选择策略的方式。我并不是选择贪婪策略，而是从学习到的随机策略中进行采样。这里正确的做法取决于环境，但采样是安全的赌注。

# VPG：学习一个值函数

你在上一个部分学到的 REINFORCE 算法在简单问题中表现良好，并且有收敛保证。但由于我们使用完整的蒙特卡洛回报来计算梯度，其方差是一个问题。在本节中，我们将讨论几种处理这种方差的方法，这些方法被称为*纯策略梯度*或*带有基线的 REINFORCE*。

## 进一步减少策略梯度的方差

REINFORCE 是一个原则性的算法，但它具有很高的方差。你可能还记得第五章中关于蒙特卡洛目标的讨论，但让我们再重申一遍。沿着轨迹随机事件的累积，包括从初始状态分布中采样的初始状态——转换函数概率，但现在在本章中是具有随机策略的——是动作选择添加到混合中的随机性。所有这些随机性都累积在回报中，使其成为一个高方差信号，难以解释。

减少方差的一种方法是用部分回报而不是完整回报来改变动作的对数概率。我们已经实现了这一改进。但另一个问题是动作对数概率会随着回报的比例而变化。这意味着，如果我们收到一个显著的正面回报，导致该回报的动作概率会大幅增加。而如果回报是显著的负值，那么概率会大幅降低。

然而，想象一个像小车-杆这样的环境，其中所有奖励和回报都是正的。为了准确地区分可接受的动作和最佳动作，我们需要大量的数据。否则，方差很难被抑制。如果我们能够不用噪声回报，而用某种允许我们在相同状态下区分动作值的东西，那就方便多了。还记得吗？

| ![](img/icons_Memory.png) | 刷新我的记忆使用策略梯度方法中的估计优势 |
| --- | --- |
|  | ![](img/11_04_Sidebar10.png) |

## 学习值函数

正如你在上一页看到的，我们可以通过使用动作优势函数的估计值来进一步减少策略梯度的方差，而不是实际回报。使用优势函数可以在一定程度上使分数围绕零中心；优于平均的动作有正分数，而低于平均的动作有负分数。前者会降低概率，而后者会增加概率。

我们将做的是确切如此。现在，让我们创建两个神经网络，一个用于学习策略，另一个用于学习状态值函数 V。然后，我们使用状态值函数和回报来计算优势函数的估计值，就像我们接下来看到的那样。

![](img/11_05.png)

两个神经网络，一个用于策略，一个用于值函数

| ŘŁ | 带有强化学习口音的 REINFORCE、纯策略梯度、基线、演员-评论家 |
| --- | --- |
|  | 一些有先前 DRL 经验的人可能会想，这是所谓的“actor-critic”吗？它正在学习策略和值函数，所以看起来应该是这样的。不幸的是，这是那些“RL 口音”会混淆新手的那些概念之一。原因如下。首先，根据 RL 的一个创始人 Rich Sutton，策略梯度方法近似性能度量的梯度，无论它们是否学习近似值函数。然而，DRL 中最杰出的代表人物之一，Sutton 的前学生 David Silver，不同意这种说法。他说，基于策略的方法不会额外学习值函数，只有 actor-critic 方法才会。但是，Sutton 进一步解释说，只有使用自举学习值函数的方法才能被称为 actor-critic，因为自举会给值函数添加偏差，从而使其成为一个“critic”。我喜欢这种区分；因此，本书中介绍的 REINFORCE 和 VPG 不被视为 actor-critic 方法。但请注意术语，它并不一致。 |

## 鼓励探索

对策略梯度方法的一个基本改进是向损失函数中添加熵项。我们可以从许多不同的角度来解释熵，从从分布中采样可以获得的信息量到对集合进行排序的方式数。

![](img/11_06.png)

我喜欢将熵想得很简单。均匀分布，即样本均匀分布，具有高熵，实际上是最高的。例如，如果你有两个样本，并且两者都有 50% 的概率被抽取，那么熵是两个样本集可能达到的最高值。如果你有四个样本，每个样本有 25% 的概率，熵是相同的，是四个样本集可能达到的最高值。相反，如果你有两个样本，一个有 100% 的概率，另一个有 0%，那么熵是最低的，总是零。在 PyTorch 中，使用自然对数来计算熵而不是二进制对数。这主要是因为自然对数使用欧拉数 e，使数学更“自然”。然而，从实际的角度来看，这并没有区别，效果是相同的。在有两个动作的 cart-pole 环境中，熵介于 0 和 0.6931 之间。

在策略梯度方法中使用熵的方法是将负加权熵添加到损失函数中，以鼓励动作分布均匀。这样，具有均匀分布动作的策略，即产生最高熵的策略，有助于最小化损失。另一方面，收敛到单个动作，即熵为零，不会减少损失。在这种情况下，智能体最好收敛到最优动作。

| ![](img/icons_Math.png) | 使用 VPG 的数学损失 |
| --- | --- |
|  | ![](img/11_06_Sidebar12.png) |

| ![](img/icons_Python.png) | 我会说 Python 状态值函数神经网络模型 |
| --- | --- |

```
class FCV(nn.Module): ①
    def __init__(self,
                 input_dim,
                 hidden_dims=(32,32), ②
                 activation_fc=F.relu):
        super(FCV, self).__init__()
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(input_dim,          ③
                                     hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear( ④
                hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear( ⑤
            hidden_dims[-1], 1) ⑤
    def forward(self, state): ⑥
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32) ⑦
            x = x.unsqueeze(0)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers: ⑧
            x = self.activation_fc(hidden_layer(x))
        return self.output_layer(x) ⑨
```

① 这是状态价值函数神经网络。它与过去我们使用的 Q 函数网络类似。② 注意我为你留下了方便的超参数，你可以随意调整。③ 在这里，我们在输入节点和第一个隐藏层之间创建线性连接。④ 在这里，我们在隐藏层之间创建连接。⑤ 在这里，我们将最后一个隐藏层连接到输出层，输出层只有一个节点，表示状态的价值。⑥ 这是前向传递函数。⑦ 这是按照我们期望的格式化输入。⑧ 进行完整的正向传递...⑨ ...并返回状态的价值。

| ![Python 图标](img/icons_Python.png) | 我会说 PythonVanilla 策略梯度，即带有基线的 REINFORCE |
| --- | --- |

```
class VPG(): ①
    <...>                                                       ①
    def optimize_model(self):
        T = len(self.rewards)
        discounts = np.logspace(0, T, num=T, base=self.gamma,
                                endpoint=False)
        returns = np.array( ②
[np.sum(discounts[:T-t] * self.rewards[t:]) for t in range(T)])  ③
        value_error = returns - self.values
        policy_loss = -( ④
         discounts * value_error.detach() * self.logpas).mean()
        entropy_loss = -self.entropies.mean() ⑤
        loss = policy_loss + \
                        self.entropy_loss_weight * entropy_loss
        self.policy_optimizer.zero_grad()
        loss.backward() ⑥
        torch.nn.utils.clip_grad_norm_(
                               self.policy_model.parameters(),
                               self.policy_model_max_grad_norm)
        self.policy_optimizer.step() ⑦
        value_loss = value_error.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_( ⑧
                                self.value_model.parameters(),
                                self.value_model_max_grad_norm)
        self.value_optimizer.step()
```

① 这是 VPG 算法。我移除了很多代码，所以如果你想要完整的实现，请查看该章节的笔记本。② 这是一个计算从时间步 0 到 T 的折扣奖励总和的非常方便的方法。③ 我想强调的是，这个循环正在遍历所有步骤从 0，然后 1，2，3 一直到最后的状态*T*，并计算从该状态开始的回报，即从时间步 t 到最终状态*T*的折扣奖励总和。④ 首先，计算价值误差；然后使用它来评分动作的对数概率。然后，将这些概率折扣以与折扣策略梯度兼容。然后，使用负平均值。⑤ 计算熵，并将其添加到损失中。⑥ 现在，我们优化策略。将优化器置零，进行反向传播，然后（如果需要）裁剪梯度。⑦ 我们移动优化器。⑧ 最后，我们优化价值函数神经网络。

# A3C：并行策略更新

VPG 对于简单问题来说是一个相当稳健的方法；它大部分是无偏的，因为它使用无偏的目标来学习策略和价值函数。也就是说，它使用蒙特卡洛回报，这是在环境中直接体验到的完整实际回报，没有任何自举。整个算法中唯一的偏差是因为我们使用了函数逼近，这本身是有偏差的，但由于 ANN 只是用作基线以减少实际回报的方差，因此引入的偏差很小，如果有的话。
