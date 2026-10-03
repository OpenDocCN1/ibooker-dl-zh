
# 策略梯度法与演员-评论家方法

## A2C 训练过程

```python
class A2C():
    def train(self, make_envs_fn, make_env_fn, ①
              make_env_kargs, seed, gamma, max_minutes,
              max_episodes, goal_mean_100_reward):
        envs = self.make_envs_fn(make_env_fn, ②
                     make_env_kargs, self.seed,
                   self.n_workers)
        <...>
        self.ac_model = self.ac_model_fn(nS, nA) ③
        self.ac_optimizer = self.ac_optimizer_fn(
                           self.ac_model, self.ac_optimizer_lr)
        states = envs.reset() ④
        for step in count(start=1):
            states, is_terminals = \                            ⑤
                            self.interaction_step(states, envs)
            if is_terminals.sum() or \
                      step - n_steps_start == self.max_n_steps:
                past_limits_enforced = envs._past_limit()
                failure = np.logical_and(is_terminals,
                          np.logical_not(past_limits_enforced)) ⑤
                next_values = self.ac_model.evaluate_state( ⑤
                    states).detach().numpy() * (1 - failure)
                self.rewards.append(next_values)
                self.values.append(torch.Tensor(next_values))
                self.optimize_model()
                self.logpas, self.entropies = [], []
                self.rewards, self.values = [], []              ⑥
                n_steps_start = step
```

① 这是我们如何使用多处理器环境进行训练的方式。② 这里，看看如何创建，基本上，向量化的环境。③ 这里我们创建一个单一模型。这是一个具有策略和价值输出的演员-评论员模型。④ 看看，我们重置多处理器环境并获取一系列状态。⑤ 主要的是我们现在处理堆栈。⑥ 但，在其核心，一切都是相同的。

| ![Python 图标](img/icons_Python.png) | 我会说 PythonA2C 优化模型逻辑 |
| --- | --- |

## A2C 优化过程

```python
class A2C(): ①
    def optimize_model(self):
        T = len(self.rewards)
        discounts = np.logspace(0, T, num=T, base=self.gamma,
                                endpoint=False)
        returns = np.array(
                     [[np.sum(discounts[:T-t] * rewards[t:, w])
                             for t in range(T)] \             ②
                             for w in range(self.n_workers)]) ②
        np_values = values.data.numpy()
        tau_discounts = np.logspace(0, T-1, num=T-1,
                    base=self.gamma*self.tau, endpoint=False)
        advs = rewards[:-1] + self.gamma * np_values[1:] \    ③
                                               - np_values[:-1]
        gaes = np.array(
            [[np.sum(tau_discounts[:T-1-t] * advs[t:, w]) \
                for t in range(T-1)]
     for w in range(self.n_workers)]) ④
        discounted_gaes = discounts[:-1] * gaes
        value_error = returns - values                        ⑤
        value_loss = value_error.pow(2).mul(0.5).mean()
        policy_loss = -(discounted_gaes.detach() * \
                                                 logpas).mean()
        entropy_loss = -entropies.mean()
        loss = self.policy_loss_weight * policy_loss + \      ⑤
                self.value_loss_weight * value_loss + \
                self.entropy_loss_weight * entropy_loss
        self.ac_optimizer.zero_grad()
        loss.backward() ⑥
        torch.nn.utils.clip_grad_norm_(
            self.ac_model.parameters(),
          self.ac_model_max_grad_norm)
        self.ac_optimizer.step()
```

① 这是我们如何在 A2C 中优化模型的方式。② 主要要注意的是，我们现在使用的是每个工作者的时间步向量矩阵。③ 一些操作以完全相同的方式进行，令人惊讶。④ 对于某些操作，我们只需要添加一个循环来包含所有工作者。⑤ 看看我们如何构建单个损失函数。⑥ 最后，我们优化单个神经网络。

| ![细节图标](img/icons_Details.png) | 细节之处 Running all policy-gradient methods in the CartPole-v1 environment |
| --- | --- |

## 实验配置细节

为了展示策略梯度算法，并使与之前章节中探索的基于价值的方法的比较更容易，我使用了与基于价值方法实验相同的配置进行了实验。以下是详细信息：

**REINFORCE：**

+   运行具有 4-128-64-2 节点的策略网络，使用 Adam 优化器，学习率 `lr` 为 0.0007。
+   在每个回合结束时使用蒙特卡洛回报进行训练。没有基线。

**VPG（使用蒙特卡洛基线的 REINFORCE）：**

+   与 REINFORCE 相同的策略网络，但现在我们在损失函数中添加了一个熵项，权重为 0.001，并裁剪梯度范数为 1。
+   我们现在学习一个价值函数并将其用作基线，而不是作为评论员。这意味着 MC 回报在没有重新启动的情况下使用，价值函数仅减少回报的规模。价值函数使用 4-256-128-1 网络，RMSprop 优化器，学习率为 0.001。虽然没有梯度裁剪，但这是可能的。

**A3C：**

+   我们以完全相同的方式训练策略网络和价值网络。
+   我们现在每 50 步最多（或当达到终端状态时）重新启动回报。这是一个演员-评论员方法。
+   我们使用八个工人，每个工人都有网络的副本，并执行 Hogwild!更新。

**GAE：**

+   与其他算法相同的超参数。
+   主要区别在于 GAE 添加了一个 `tau` 超参数来折现优势。我们在这里使用 0.95 作为 `tau`。请注意，代理风格具有相同的 *n* 步自举逻辑，这可能不会使这是一个纯粹的 GAE 实现。通常，你会看到一次处理完整的批次。它仍然表现相当不错。

**A2C：**

+   A2C 确实改变了大多数超参数。首先，我们有一个单一的网络：4-256-128-3（2 和 1）。使用 Adam 进行训练，学习率为 0.002，梯度范数为 1。
+   策略权重为 1.0，价值函数为 0.6，熵为 0.001。
+   我们采用 10 步自举，八个工人，以及 0.95 的 `tau`。

这些算法并没有独立调优；我确信它们可以做得更好。

| ![图片](img/icons_Tally.png) | CartPole-v1 环境上的策略梯度法和演员-评论家方法汇总 |
| --- | --- |
| ![图片](img/11_09_Sidebar34.png) | |

## 概述

在本章中，我们概述了策略梯度法和演员-评论家方法。首先，我们用几个理由来介绍考虑策略梯度法和演员-评论家方法。你了解到直接学习策略是强化学习方法的真正目标。你了解到通过学习策略，我们可以使用随机策略，这在部分可观察环境中可能比基于价值的方法表现更好。你了解到尽管我们通常学习随机策略，但没有任何东西阻止神经网络学习确定性策略。

你还学习了四种算法。首先，我们研究了 REINFORCE 及其如何是改进策略的直接方法。在 REINFORCE 中，我们可以使用完整的回报或奖励到去作为改进策略的分数。

然后，你学习了 vanilla 策略梯度，也称为带有基线的 REINFORCE。在这个算法中，我们使用蒙特卡洛回报作为目标来学习一个价值函数。然后，我们使用价值函数作为基线，而不是作为评论家。在 VPG 中我们不进行自举；相反，我们使用奖励到去，就像在 REINFORCE 中一样，并减去学习到的价值函数以减少梯度的方差。换句话说，我们使用优势函数作为策略分数。

我们还研究了 A3C 算法。在 A3C 中，我们自举价值函数，无论是为了学习价值函数还是为了评分策略。更具体地说，我们使用 *n* 步回报来改进模型。此外，我们使用多个演员-学习者，每个演员-学习者都执行策略，评估回报，并使用 Hogwild! 方法更新策略和价值模型。换句话说，工人更新无锁模型。

我们接着学习了 GAE，以及这是如何估计类似于 *TD*(*λ*) 和 *λ*-回报的优势的方法。GAE 使用所有 *n*-步优势的指数加权混合，以创建一个更稳健的优势估计，可以轻松调整以使用更多的自举和因此偏差，或者实际的回报和因此方差。

最后，我们学习了 A2C 以及移除 A3C 的异步部分可以得到一个无需实现自定义优化器的相当算法。

到现在为止，

+   理解基于值、基于策略、策略梯度法和演员-评论家方法之间的主要区别
+   可以自己实现基本的策略梯度法和演员-评论家方法
+   可以调整策略梯度法和演员-评论家算法以通过各种环境

| ![](img/icons_Tweet.png) | 可分享的工作在自己的工作上努力，并分享你的发现 |
| --- | --- |

## 可分享的工作

这里有一些关于如何将你所学的内容提升到下一个层次的想法。如果你愿意，与世界分享你的结果，并确保查看其他人做了什么。这是一个双赢的局面，希望你能充分利用它。

+   **#gdrl_ch11_tf01:** 在本章的早期，我提到了一个虚构的雾蒙蒙的湖环境，但嘿，它之所以是虚构的，是因为你还没有实现它，对吧？现在就去实现一个雾蒙蒙的湖环境，以及一个雾蒙蒙的冰冻湖环境。对于这个环境，确保传递给智能体的观察与环境的实际内部状态不同。例如，如果智能体位于单元格 3，内部状态是保密的，智能体只能观察到它位于一个雾蒙蒙的单元格中。在这种情况下，所有雾蒙蒙的单元格应该发出相同的观察，这样智能体就无法判断自己的位置。实现这个环境后，测试只能学习确定性策略的 DRL 智能体（如前几章所述），以及可以学习随机策略的智能体（如本章所述）。你将需要对观察进行 one-hot 编码，以便将其传递到神经网络中。创建一个包含环境的 Python 包，以及一个包含有趣的测试和结果的 Notebook。

+   **#gdrl_ch11_tf02:** 在本章中，我们仍然使用 CartPole-v1 环境作为测试平台，但你应该知道环境交换应该是简单的。首先，在类似的环境中测试相同的智能体，例如 LunarLander-v2 或 MountainCar-v0。注意使其相似的是观察是低维连续的，动作是低维离散的。其次，在不同的环境中测试它们，这些环境具有高维观察或动作。

+   **#gdrl_ch11_tf03:** 在每一章中，我都使用最后的标签作为通用的标签。请随意使用这个标签来讨论与本章相关的任何其他内容。没有比你自己创造的任务更令人兴奋的作业了。确保分享你为自己设定的研究内容和你的结果。

用你的发现写一条推文，@提及我 @mimoralea（我会转发），并使用列表中的特定标签来帮助感兴趣的人找到你的结果。没有对错之分；你分享你的发现并检查他人的发现。利用这个机会社交，做出贡献，让自己更受关注！我们正在等待你的到来！以下是一条推文示例：“嘿，@mimoralea。我创建了一篇博客文章，列出了学习深度强化学习的资源列表。查看它在这里 <link>。#gdrl_ch01_tf01”我会确保转发并帮助他人找到你的作品。
