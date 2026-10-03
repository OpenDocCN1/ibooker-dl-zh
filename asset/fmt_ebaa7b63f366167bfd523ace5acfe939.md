
# 12 高级演员-评论家方法

## 在本章中

- 你将了解更多高级深度强化学习方法，这些方法至今仍然是深度强化学习中的最先进算法进步。
- 你将学习解决各种深度强化学习问题，从具有连续动作空间的问题到具有高维动作空间的问题。
- 你将从头开始构建最先进的演员-评论家方法，并打开理解与人工通用智能相关的高级概念的大门。

批评可能并不令人愉快，但它是必要的。它起着与人体疼痛相同的作用。它引起了人们对事物不健康状态的注意。

— 温斯顿·丘吉尔 英国政治家、军官、作家和英国首相

在上一章中，你了解了一种不同的、更直接的技术来解决深度强化学习问题。你首先被介绍到策略梯度方法，其中智能体通过直接近似来学习策略。在纯策略梯度方法中，我们不使用价值函数作为寻找策略的代理，实际上，我们根本不使用价值函数。我们而是直接学习随机策略。

然而，你很快注意到价值函数仍然可以发挥重要作用，并使策略梯度方法得到改进。因此，你被引入了演员-评论家方法。在这些方法中，智能体学习策略和价值函数。通过这种方法，你可以利用一个函数近似的优点来减轻另一个近似的缺点。例如，在某些环境中，学习策略可能比学习足够准确的价值函数更直接，因为动作空间中的关系可能比价值的关系更紧密。尽管如此，即使精确地知道状态的价值可能更复杂，但粗略的近似对于减少策略梯度目标方差是有用的。正如你在上一章中所探索的，学习价值函数并将其用作基线或用于计算优势可以显著减少用于策略梯度更新的目标方差。此外，减少方差通常会导致学习速度更快。

然而，在上一章中，我们专注于使用价值函数作为更新随机策略的评论家。我们为学习价值函数使用了不同的目标，并以几种不同的方式并行化了工作流程。然而，算法以相同的一般方式使用学习到的价值函数来训练策略，并且学习到的策略具有相同的属性，因为它是一个随机策略。我们只是触及了使用学习策略和价值函数的表面。在本章中，我们将更深入地探讨演员-评论家方法的范式，并在四个不同的具有挑战性的环境中进行训练：摆锤、跳跃者、猎豹和月球着陆器。正如你很快就会看到的，除了是更具挑战性的环境之外，大多数这些环境都具有连续的动作空间，这是我们第一次面对的，并且需要使用独特的策略模型。

为了解决这些环境，我们首先探索了学习确定性策略的方法；也就是说，当呈现相同的状态时，返回相同动作的策略，即被认为是最优的动作。我们还研究了一系列改进，使确定性策略梯度算法成为迄今为止解决深度强化学习问题的最先进方法之一。然后，我们探索了一种演员-评论家方法，它不是使用损失函数中的熵，而是直接使用价值函数方程中的熵。换句话说，它最大化回报以及策略的长期熵。最后，我们以一个算法结束，该算法通过将策略的更新限制为小的变化，从而允许更稳定的策略改进步骤。策略中的小变化使得策略梯度方法在性能上表现出稳定且通常是单调的改进，从而在几个 DRL 基准测试中实现了最先进的性能。

## DDPG：逼近确定性策略

在本节中，我们探讨了一种名为深度确定性策略梯度（DDPG）的算法。DDPG 可以被视为一个近似的 DQN，或者更好的说法，是一个适用于连续动作空间的 DQN。DDPG 使用了 DQN 中发现的许多相同技术：它使用重放缓冲区以离线方式训练动作值函数，并使用目标网络来稳定训练。然而，DDPG 还训练了一个逼近最优动作的策略。正因为如此，DDPG 是一个限制在连续动作空间中的确定性策略梯度方法。

### DDPG 借鉴了 DQN 的许多技巧

首先，将 DDPG 想象成一个与 DQN 具有相同架构的算法。训练过程也是类似的：智能体以在线方式收集经验，并将这些在线经验样本存储到重放缓冲区中。在每一步，智能体从重放缓冲区中抽取一个迷你批次，通常是随机均匀抽取的。然后，智能体使用这个迷你批次来计算自举的*TD*目标并训练一个 Q 函数。

DQN 和 DDPG 之间的主要区别在于，虽然 DQN 使用目标 Q 函数通过 argmax 获取贪婪动作，而 DDPG 使用一个目标确定性策略函数，该函数被训练来近似贪婪动作。与我们在 DQN 中使用下一个状态的 Q 函数的 argmax 获取贪婪动作不同，在 DDPG 中，我们直接使用策略函数来近似下一个状态中的最佳动作。然后，在两者中，我们都使用该动作与 Q 函数结合来获取最大值。

| ![](img/icons_Math.png) | 展示 MathDQN 与 DDPG 的价值函数目标 |
| --- | --- |
|  | ![](img/12_00_Sidebar01.png) |

| ![](img/icons_Python.png) | 我会说 PythonDDPG 的 Q 函数网络 |
| --- | --- |

```
class FCQV(nn.Module):                                           ①
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=(32,32),
                 activation_fc=F.relu):
        super(FCQV, self).__init__()
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])  ②
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            in_dim = hidden_dims[i]
            if i == 0:                                           ③
                in_dim += output_dim
            hidden_layer = nn.Linear(in_dim, hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)        ④
       <...>
    def forward(self, state, action):                            ⑤
        x, u = self._format(state, action)
        x = self.activation_fc(self.input_layer(x))
        for i, hidden_layer in enumerate(self.hidden_layers):
            if i == 0:
                x = torch.cat((x, u), dim=1)                     ⑥
            x = self.activation_fc(hidden_layer(x))              ⑦
        return self.output_layer(x)                              ⑧
```

① 这是 DDPG 中使用的 Q 函数网络。② 我们像往常一样开始构建架构。③ 这里有一个例外。我们将第一隐藏层的维度增加到输出维度。④ 注意网络的输出是一个节点，表示状态-动作对的价值。⑤ 前向传播开始如预期。⑥ 但是，我们在第一隐藏层直接将动作连接到状态。⑦ 然后，继续如预期。⑧ 最后，返回输出。

### 学习确定性策略

现在，我们需要添加到这个算法中的一个东西是策略网络。我们希望训练一个网络，能够在给定状态下给出最佳动作。该网络必须相对于动作是可微分的。因此，动作必须是连续的，以便进行高效的基于梯度的学习。目标是简单的；我们可以使用策略网络 `mu` 的期望 Q 值。也就是说，智能体试图找到最大化这个值的动作。注意，在实践中，我们使用最小化技术，因此最小化这个目标的负值。

| ![](img/icons_Math.png) | 展示 MathDDPG 的确定性策略目标 |
| --- | --- |
|  | ![](img/12_00_Sidebar03.png) |

还要注意，在这种情况下，我们不需要使用目标网络，而是使用在线网络，即策略网络（动作选择部分）和价值函数（动作评估部分）。此外，鉴于我们需要为训练价值函数采样状态的小批量，我们可以使用这些相同的状态来训练策略网络。

| 0001 | 一点历史介绍 DDPG 算法 |
| --- | --- |
|  | DDPG 是在 2015 年一篇题为“使用深度强化学习的连续控制”的论文中提出的。这篇论文由当时在谷歌 DeepMind 担任研究科学家的蒂莫西·利利克拉普（Timothy Lillicrap）等人撰写。自 2016 年以来，蒂姆一直担任谷歌 DeepMind 的高级研究科学家和伦敦大学学院的客座教授。蒂姆为 DeepMind 的几篇其他论文做出了贡献，例如 A3C 算法、AlphaGo、AlphaZero、Q-Prop 和星际争霸 II 等。最有趣的事实之一是，蒂姆在认知科学和系统神经科学方面有背景，而不是传统计算机科学路径进入深度强化学习。 |

| ![](img/icons_Python.png) | 我会说 PythonDDPG 的确定性策略网络 |
| --- | --- |

```
class FCDP(nn.Module):                                           ①
    def __init__(self,
                 input_dim,
                 action_bounds,
                 hidden_dims=(32,32),
                 activation_fc=F.relu,                           ②
                 out_activation_fc=F.tanh):
        super(FCDP, self).__init__()
        self.activation_fc = activation_fc
        self.out_activation_fc = out_activation_fc
        self.env_min, self.env_max = action_bounds               ③
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])  ④
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i],
                                     hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1],
                                      len(self.env_max))         ④
    def forward(self, state):                                    ⑤
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))              ⑥
        for hidden_layer in self.hidden_layers:                  ⑦
            x = self.activation_fc(hidden_layer(x))
        x = self.output_layer(x)                                 ⑧
        x = self.out_activation_fc(x)                            ⑨
        return self.rescale_fn(x)                                ⑩
```

① 这是 DDPG 中使用的策略网络：全连接确定性策略。② 注意这次输出层的激活函数不同。我们使用 `tanh` 激活函数将输出压缩到(-1, 1)范围内。③ 我们需要获取动作的最小值和最大值，以便将网络的输出(-1, 1)重新缩放到预期的范围。④ 架构正如预期：输入状态，输出动作。⑤ 前向传播也很直接。⑥ 输入⑦ 隐藏⑧ 输出⑨ 注意，然而，我们使用输出激活函数激活输出。⑩ 同样重要的是，我们将动作从-1 到 1 的范围重新缩放到特定于环境的范围。`rescale_fn` 函数在此处未显示，但你可以去笔记本中查看详细信息。

| ![](img/icons_Python.png) | 我会说 PythonDDPG 的模型优化步骤 |
| --- | --- |

```
    def optimize_model(self, experiences):                     ①
        states, actions, rewards, \
                  next_states, is_terminals = experiences
        batch_size = len(is_terminals)
        argmax_a_q_sp = self.target_policy_model(next_states)
        max_a_q_sp = self.target_value_model(next_states,
                                             argmax_a_q_sp)    ②
        target_q_sa = rewards + self.gamma * max_a_q_sp * \
                                             (1 - is_terminals)
        q_sa = self.online_value_model(states, actions)
        td_error = q_sa - target_q_sa.detach()                 ③
        value_loss = td_error.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()                       ④
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(
                          self.online_value_model.parameters(),
                          self.value_max_grad_norm)
        self.value_optimizer.step()
        argmax_a_q_s = self.online_policy_model(states)
        max_a_q_s = self.online_value_model(states,
                                            argmax_a_q_s)      ⑤
        policy_loss = -max_a_q_s.mean()                        ⑥
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(                        ⑦
                         self.online_policy_model.parameters(),
                         self.policy_max_grad_norm)
        self.policy_optimizer.step()
```

① `optimize_model` 函数接收一个经验的小批量。② 使用它，我们根据策略预测的下一个状态的最大值、根据策略的动作和根据 Q 函数的值来计算目标。③ 然后我们获取预测值，计算误差和损失。注意我们使用目标网络和在线网络的位置。④ 优化步骤与其他网络类似。⑤ 接下来，我们获取在线策略预测的迷你批量中的状态的动作，然后使用这些动作通过在线价值网络获取价值估计。⑥ 接下来，我们获取策略损失。⑦ 最后，我们将优化器置零，对损失进行反向传播，裁剪梯度，并更新优化器。

### 探索确定性策略

在 DDPG 中，我们训练确定性贪婪策略。在一个理想的世界里，这种策略接受一个状态并返回该状态的最优动作。但是，在一个未经训练的策略中，返回的动作不会足够准确，但仍然是确定的。如前所述，智能体需要平衡利用知识和探索。但是，由于 DDPG 智能体学习的是确定性策略，它不会在策略上进行探索。想象一下，智能体很固执，总是选择相同的动作。为了解决这个问题，我们必须进行离策略探索。因此，在 DDPG 中，我们在策略选择的动作中注入高斯噪声。
