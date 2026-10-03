
## 多臂老虎机（MAB）算法

```py
def ucb(c):
    num_plays = np.zeros(K)                                                       ①
    sum_rewards = np.zeros(K)                                                     ②
    total_regret = 0                                                              ③

    for i in range(K):                                                            ④
        reward = sample_payoff(i)
        num_plays[i] += 1
        sum_rewards[i] += reward

    for t in range(K, num_trials):                                                ⑤
        ucb_values = sum_rewards / num_plays + c * np.sqrt(np.log(t) / num_plays) ⑥
        selected_machine = np.argmax(ucb_values)
        reward = sample_payoff(selected_machine)
        num_plays[selected_machine] += 1
        sum_rewards[selected_machine] += reward
        optimal_reward = max_reward
        regret = optimal_reward - reward
        total_regret += regret

    average_regret = total_regret / num_trials
    return average_regret
```

① 初始化一个数组来跟踪每个投币机的游戏次数。  
② 初始化一个数组来累计从每个投币机获得的奖励总和。  
③ 初始化每个老虎机的总后悔值。  
④ 每次玩一次老虎机以初始化。  
⑤ 继续使用 UCB 策略进行游戏。  
⑥ 计算 UCB 值。

以下代码片段用于运行策略，计算平均后悔值，并打印结果：

```py
avg_regret_explore = explore_only()
avg_regret_exploit = exploit_only_greedy()
avg_regret_epsilon_greedy = epsilon_greedy(0.1)                    ①
avg_regret_ucb = ucb(2)                                            ②

print(f"Average Regret - Explore only Strategy: {round(avg_regret_ ③
explore,4)}")                                                      ③
print(f"Average Regret - Exploit only Greedy Strategy:
➥    {round(avg_regret_exploit,4)}")                              ③
print(f"Average Regret - Epsilon-greedy Strategy:
➥    {round(avg_regret_epsilon_greedy,4)}")                       ③
print(f"Average Regret - UCB Strategy: {round(avg_regret_ucb,4)}") ③
```

① 设置 epsilon-greedy 策略的 epsilon 值，并运行策略。  
② 设置 UCB 策略的探索参数 c 的值，并运行策略。  
③ 打印结果

由于代码中包含随机抽样，您的结果可能会有所不同，但运行代码将产生类似以下的结果：

```py
Average Regret - Explore only Strategy: 2.246
Average Regret - Exploit only Greedy Strategy: 0.048
Average Regret - Epsilon-greedy Strategy: 0.3466
Average Regret - UCB Strategy: 0.0378
```

MAB 算法和概念在各种现实场景中找到应用，这些场景涉及不确定性下的决策和探索-利用权衡。MABs 的现实应用示例包括但不限于资源分配（动态分配资源到不同选项以最大化性能）、在线广告（动态分配广告印象到不同选项，并学习哪些广告产生最高的点击率，这是用户点击广告的概率）、实验设计和临床试验（优化患者分配到不同治疗选项）、内容推荐（为用户个性化内容推荐）和网站优化（优化不同的设计选项）。

如您在图 12.1 中看到的，多臂老虎机（MABs）可以分为非上下文和上下文 MABs。与之前解释的非上下文 MABs 相比，上下文多臂老虎机（CMAB）使用环境中包含的上下文信息。在 CMAB 中，学习器反复观察上下文，选择一个动作，并接收与所选动作相关的奖励或损失作为反馈。CMAB 算法使用补充信息，称为*侧信息*或*上下文*，在现实场景中做出明智的决策。例如，在卡车选择问题中，共享的上下文是配送路线的类型（城市或州际）。第 12.5 节展示了如何使用 CMAB 来解决这个问题，作为组合动作的示例。

## 上下文多臂老虎机应用

上下文多臂老虎机（CMABs）在个性化推荐和在线广告等领域找到了应用，其中上下文可以是一个用户的信息。例如，亚马逊展示了如何使用内置的 Vowpal Wabbit（VW）容器在 SageMaker 上开发和部署 CMAB 工作流程（[`mng.bz/y8rE`](http://mng.bz/y8rE)）。这些 CMABs 可以用于为用户个性化内容，如内容布局、广告、搜索、产品推荐等。此外，基于 CMAB 并使用 VW 进行营销决策优化的可扩展算法决策平台 WayLift 被开发出来（[`mng.bz/MZom`](http://mng.bz/MZom)）。

## 12.2 基于强化学习的优化

强化学习（RL）可以通过将问题构造成马尔可夫决策过程（MDP）并应用强化学习算法来寻找一个最优策略，从而用于组合优化问题，以获得最佳可能的解决方案。在强化学习中，智能体通过在一个环境中学习做出连续决策，以最大化累积奖励的概念。这个过程涉及到找到一个最优策略，将状态映射到动作，以最大化预期的长期奖励。

强化学习与优化的收敛性最近已成为一个活跃的研究领域，吸引了学术界和工业界的广泛关注。研究人员正在积极探索将强化学习的优势应用于高效有效地解决复杂优化问题的方法。例如，第 11.7 节中提出的通用端到端管道可以用于解决组合优化问题，如旅行商问题（TSP）、车辆路径问题（VRP）、可满足性问题（SAT）、最大切割（MaxCut）、最大独立集（MIS）等。此管道包括使用监督学习或强化学习训练模型。列表 11.4 展示了如何使用 Joshi、Laurent 和 Bresson [3]所描述的监督学习或强化学习方法预训练的机器学习模型来解决 TSP。

一篇由 Nazari 等人撰写的论文中提出了一种使用强化学习解决车辆路径问题（VRP）的端到端框架[4]。在 Delarue、Anderson 和 Tjandraatmadja 的论文[5]中，也使用强化学习处理了带能力的车辆路径问题（CVRP）。另一篇名为 RLOR 的框架由 Wan、Li 和 Wang 在论文[6]中描述，它是一个适用于 CVRP 和 TSP 等路由问题的深度强化学习灵活框架。Alabbasi、Ghosh 和 Aggarwal 在论文[7]中描述了一个无模型的分布式强化学习算法 DeepPool，通过与环境交互来学习最优调度策略，用于拼车应用。DeepFreight 是另一个用于货运交付问题的无模型强化学习算法，由 Jiayu 等人在论文[8]中描述。它将问题分解为两个紧密协作的组件：卡车调度和包裹匹配。货运交付系统的关键目标是最大化在特定时间限制内服务的请求数量，并在此过程中最小化整个车队的总燃油消耗。MOVI 是另一种用于大规模出租车调度问题的无模型方法，由 Oda 和 Joe-Wong 在论文[9]中描述。强化学习也被用来优化交通信号控制（TSC），作为一种缓解拥堵的方法。在 Ruan 等人撰写的论文[10]（我是合著者之一）中，模拟了一个收集于中国杭州的真实路口的真实交通数据模型，并使用不同的基于强化学习的交通信号控制器进行模拟。我们还提出了一种多智能体强化学习模型，以在混合交通场景中提供宏观和微观控制[11]。实验结果表明，所提出的方法在吞吐量、平均速度和安全性等几个指标上优于其他基线。

李和 Malik 在论文[12]中描述了一个名为“学习优化”的优化算法框架。问题被表述为一个强化学习问题，其中任何优化算法都可以表示为一个策略。使用引导策略搜索，并针对不同类别的凸和非凸目标函数训练了自主优化器。这些自主优化器比手工设计的优化器收敛更快或达到更好的最优解。这与第 11.6 节中描述的摊销优化概念有些相似，但使用了强化学习。

Toll 等人在论文[13]中描述了基于强化学习的调度方法，用于一个 10 层建筑中的四电梯系统。电梯调度问题是一个组合优化问题，涉及在多层建筑中高效调度多个电梯来服务乘客的请求。如第 1.4.2 节所述，在这种情况下可能的状态数量为 10⁴（电梯位置）× 2⁴⁰（电梯按钮）× 2¹⁸（大厅呼叫按钮）= 2.88 x 10²¹ 种不同状态。

`Stable-Baselines3` (SB3)为几个与 OpenAI Gym 兼容的自定义 RL 环境提供了强化学习算法的可靠实现。要使用 pip 安装 Stable Baselines3，请执行`pip install stable-baselines3`。SB3 中 RL 算法实现的示例包括优势演员-评论家（A2C）、软演员-评论家（SAC）、深度确定性策略梯度（DDPG）、深度 Q 网络（DQN）、事后经验重放（HER）、双延迟 DDPG（TD3）和近端策略优化（PPO）。SB3 中可用的环境和项目包括以下内容：

+ *移动环境*—一个用于无线移动网络中自主协调的体育馆环境。它允许模拟涉及多个基站中移动用户的各种场景。此环境在第 12.4 节中使用。
+ *gym-electric-motor*—一个用于模拟和控制电动传动系统的 OpenAI Gym 环境。此环境在本章的练习 6 中使用（见附录 C）。
+ *高速公路环境*—一个用于在不同场景中（如高速公路、合并、环岛、停车、交叉口和赛道）进行自动驾驶决策的环境。
+ *机器人深度强化学习中的广义状态依赖探索（gSDE）*—一种直接在真实机器人上训练 RL 代理的探索方法。
+ *RL Reach*—一个用于运行可定制机器人抓取任务的 RL 实验的平台。
+ *RL Baselines3 动物园*—一个用于训练、评估 RL 代理、调整超参数、绘制结果和记录视频的框架。
+ *Furuta 悬臂机器人*—一个构建和训练旋转倒立摆（也称为 Furuta 悬臂）的项目。
+ *UAV_Navigation_DRL_AirSim*—一个用于在复杂未知环境中训练无人机导航策略的平台。
+ *触觉健身房*—RL 环境，主要使用模拟触觉传感器作为观察的主要来源。
+ *SUMO-RL*—一个用于实例化带有城市交通模拟（SUMO）的交通信号控制的 RL 环境的接口。
+ *PyBullet Gym*—用于单代理和多代理四旋翼机控制强化学习的环境。

以下章节提供了如何使用 RL 方法处理组合动作控制问题的示例。

## 12.3 使用 A2C 和 PPO 平衡 CartPole

让我们考虑一个经典控制任务，其目标是通过左右移动车辆来平衡杆在车上，如图 12.8 所示。这个任务可以被认为是一个优化问题，其目标是通过找到一种最佳策略，尽可能长时间地平衡杆在车上，来最大化累积奖励。智能体需要学习如何做出决策（采取行动）以最大化奖励信号。智能体在不同的状态下探索不同的行动，并随着时间的推移学习哪些行动能带来更高的奖励。通过根据观察到的奖励迭代更新其策略，智能体旨在优化其决策过程并找到每个状态的最佳行动。

![图片](img/CH12_F08_Khamis.png)

图 12.8 CartPole 平衡问题

这个环境的描述由四个变量组成：

+ 车辆位置（连续）—这表示车辆沿*x*轴的位置。值范围从 –4.8 到 4.8。
+ 车辆速度（连续）—这表示车辆沿*x*轴的速度。值范围从 –inf 到 inf。
+ 杆角度（连续）—这表示杆从垂直位置的角度。值范围从 –0.418 到 0.418 弧度或 –23.95° 到 23.95° 度。
+ 杆角速度（连续）—这表示杆的角速度。值范围从 –inf 到 inf。

CartPole 环境中的动作空间是离散的，由两个可能的动作组成：

+ 行动 0—将车辆向左移动。
+ 行动 1—将车辆向右移动。

每当杆保持直立时，智能体在每个时间步长都会获得 +1 的奖励。如果满足以下条件之一，则剧集结束：

+ 杆的角度超过 ±12 度远离垂直位置。
+ 车辆位置超过 ±2.4 个单位远离中心。
+ 这一幕达到最大时间步长限制（通常是 200 步）。

在 CartPole 环境中，目标是尽可能长时间地平衡杆在车上，以最大化累积奖励。让我们看看使用第 12.1.4 节中讨论的收益演员-评论员（A2C）算法学习平衡 CartPole 的最佳策略的代码。

如列表 12.2 所示，我们首先导入必要的库：

+ `gym` 是 OpenAI Gym 库，用于处理强化学习环境。
+ `torch` 是用于构建和训练神经网络的 PyTorch 库。
+ `torch.nn` 是一个模块，提供定义神经网络所需的工具。
+ `torch.nn.functional` 包含各种激活和损失函数。
+ `torch.optim` 包含用于训练神经网络的优化算法。
+ `tqdm` 提供进度条以跟踪训练进度。
+ 使用 `seaborn` 进行可视化。

列表 12.2 使用 A2C 算法平衡 CartPole

```py
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import seaborn as sns
```

接下来，我们使用 PyTorch 创建演员和评论家网络。`Actor` 类是 PyTorch 中的 `nn.Module` 的子类，代表策略网络，而 `__init__` 方法定义了演员网络的架构。使用了三个全连接层（`fc1`、`fc2`、`fc3`）。`forward` 方法通过网络进行前向传递，应用激活函数（ReLU）并使用 softmax 函数返回动作概率：

```py
class Actor(nn.Module):

    def __init__(self, state_dim,  action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)

    def forward(self, state):
        x1 = F.relu(self.fc1(state))
        x2 = F.relu(self.fc2(x1))
        action_probs = F.softmax(self.fc3(x2), dim=-1)
        return action_probs
```

`Critic` 类也是 `nn.Module` 的子类，代表价值网络。`__init__` 方法定义了评论家网络的架构，类似于演员网络。前向方法通过网络进行前向传递，应用激活函数（ReLU）并返回预测的价值：

```py
class Critic(nn.Module):

    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, state):
        x1 = F.relu(self.fc1(state))
        x2 = F.relu(self.fc2(x1))
        value = self.fc3(x2)
        return value
```

作为延续，以下代码片段用于使用 OpenAI Gym 库创建 CartPole 环境的实例，并检索有关环境的重要信息：
