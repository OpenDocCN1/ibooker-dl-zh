
在我们的在线游戏中，许多玩家是公会成员。公会是一群玩家，他们聚集资源并协调他们的游戏玩法，例如一起完成支线任务。我们的模型假设玩家进行的*游戏内购买*数量也取决于他们是否在公会中；同一公会的成员聚集资源，许多资源是他们必须购买的虚拟物品。

假设你运行了一个修改后的初始数据库查询版本。查询产生了与表 7.1 中看到的相同的观测数据，但这次它包括了一个额外的列，表示*公会成员资格*。同样，我们在表 7.5 中看到了六名玩家（与表 7.1 中显示的玩家相同）。

### 表 7.5 与表 7.1 相同的观测数据，但增加了*公会成员资格*列

| 用户 ID | 支线任务参与度 | 公会成员资格 | 游戏内购买 |
| --- | --- | --- | --- |
| 71d44ad5 | 高 | 成员 | 156.77 |
| e6397485 | 低 | 非成员 | 34.89 |
| 87a5eaf7 | 高 | 非成员 | 12.93 |
| c5d78ca4 | 低 | 成员 | 215.74 |
| d3b2a8ed | 高 | 成员 | 201.07 |
| dc85d847 | 低 | 非成员 | 34.89 |

我们将使用 Pyro 在这个观测数据上构建一个因果图模型。为此，我们需要对因果马尔可夫核进行建模：给定**公会成员资格**的**公会成员**、**支线任务参与度**以及给定**公会成员资格**和**支线任务参与度**的**游戏内购买**的概率分布。在我们的 Pyro 模型中，我们需要为这些变量指定一些典型分布并估计它们的参数。

## 估计参数和构建模型

Pyro 可以联合估计我们每个因果马尔可夫核的参数，就像它可以估计复杂神经网络架构中的参数一样。但使用日常数据科学分析，通过利用第二章中讨论的**参数模块化**概念，一次估计每个核的参数会使我们的工作更简单。让我们从**公会成员资格**开始。

### 列表 7.5 估计**公会成员资格**的概率分布

```py
import pandas as pd
full_obs_url = (     #1
    "https://raw.githubusercontent.com/altdeep/causalML/master/"     #1
    "datasets/sidequests_and_purchases_full_obs.csv"    #1
)    #1
df = pd.read_csv(full_obs_url)      #1
membership_counts = df['Guild Membership'].value_counts()    #2
dist_guild_membership = membership_counts / sum(membership_counts)     #2
print(dist_guild_membership)    #2
```

#1 从数据库查询中加载数据到 pandas DataFrame 中。

#2 计算成员与非成员的比例。

这将打印出以下结果：

```py
nonmember    0.515
member       0.485
Name: Guild Membership, dtype: float64
```

这些是数据中公会成员与非成员的比例。我们可以将这些作为估计玩家是成员或非成员的概率。如果我们直接使用这些比例，它们将是概率的最大似然估计，但为了简单起见，我们将将其设置为 50/50（成为成员的概率为.5）。

接下来，我们将对给定**公会成员资格**的**支线任务参与度**条件概率分布（CPD）做同样的处理。

### 列表 7.6 估计**支线任务参与度**给定**公会成员资格**的 CPD

```py
member_subset = df[(df['Guild Membership'] == 'member')]     #1
member_engagement_counts = (   #1
    member_subset['Side-quest Engagement'].value_counts()   #1
)    #1
dist_engagement_member = (    #1
    member_engagement_counts / sum(member_engagement_counts)   #1
)    #1
print(dist_engagement_member)    #1

nonmember_subset = df[(df['Guild Membership'] == 'nonmember')]    #2
nonmember_engagement_counts = (   #2
    nonmember_subset['Side-quest Engagement'].value_counts()    #2
)     #2
dist_engagement_nonmember = (    #2
    nonmember_engagement_counts /   #2
sum(nonmember_engagement_counts)    #2
)     #2
print(dist_engagement_nonmember)     #2
```

#1 计算当玩家是公会成员时，**支线任务参与度**水平（“高”与“低”）的概率分布。

#2 计算当玩家不是公会成员时，**支线任务参与度**水平（“高”与“低”）的概率分布。

列表 7.6 打印了以下输出，显示了公会成员的**支线任务参与度**水平比例：

```py
high    0.797938
low     0.202062
```

以下比例适用于非公会成员：

```py
high    0.184466
low     0.815534
```

同样，我们将对这些结果进行四舍五入。公会成员有 80%的可能性在支线任务中高度参与，而非成员只有 20%的可能性高度参与。

最后，对于**公会成员资格**和**支线任务参与度**的每个组合，我们将计算**游戏内购买**的样本均值和标准差。当我们编码因果模型中**游戏内购买**的因果马尔可夫核时，我们将使用这些样本统计量作为典型分布中均值和位置参数的估计。

### 列表 7.7 按参与度和**公会成员资格**级别计算购买统计数据

```py
purchase_dist_nonmember_low_engagement = df[     #1
    (df['Guild Membership'] == 'nonmember') &    #1
    (df['Side-quest Engagement'] == 'low')    #1
].drop(     #1
  ['User ID', 'Side-quest Engagement', 'Guild Membership'], axis=1   #1  
).agg(['mean', 'std'])    #1
print(round(purchase_dist_nonmember_low_engagement, 2))    #1

purchase_dist_nonmember_high_engagement = df[     #2
    (df['Guild Membership'] == 'nonmember') &    #2
    (df['Side-quest Engagement'] == 'high')    #2
].drop(     #2
  ['User ID', 'Side-quest Engagement', 'Guild Membership'], axis=1     #2
).agg(['mean', 'std'])    #2
print(round(purchase_dist_nonmember_high_engagement, 2))    #2

purchase_dist_member_low_engagement = df[     #3
    (df['Guild Membership'] == 'member') &    #3
    (df['Side-quest Engagement'] == 'low')    #3
].drop(     #3
  ['User ID', 'Side-quest Engagement', 'Guild Membership'], axis=1     #3
).agg(['mean', 'std'])     #3
print(round(purchase_dist_member_low_engagement, 2))     #3

purchase_dist_member_high_engagement = df[     #4
    (df['Guild Membership'] == 'member') &    #4
    (df['Side-quest Engagement'] == 'high')    #4
].drop(    #4
  ['User ID', 'Side-quest Engagement', 'Guild Membership'], axis=1    #4
).agg(['mean', 'std'])     #4
print(round(purchase_dist_member_high_engagement, 2))   #4
```

#1 估计具有低支线任务参与度的非公会成员的游戏内购买样本均值和标准差。

#2 估计具有高支线任务参与度的非公会成员的游戏内购买样本均值和标准差。

#3 估计低支线任务参与度的公会成员的游戏内购买的样本均值和标准差。

#4 估计高支线任务参与度的公会成员的游戏内购买的样本均值和标准差。

对于低*支线任务参与度*的非公会成员，我们有这些结果：

```py
      In-game Purchases
mean              37.95
std               23.80
```

对于高*支线任务参与度*的非公会成员，我们有

```py
      In-game Purchases
mean              54.92
std                4.92
```

对于低*支线任务参与度*的公会成员，我们有

```py
      In-game Purchases
mean             223.71
std                5.30
```

对于高*支线任务参与度*的公会成员，我们有

```py
      In-game Purchases
mean             125.53
std               53.44
```

最后，在第 7.8 列表中，我们使用这些各种统计量作为在 Pyro 中构建的因果图模型中的参数估计。

### 列表 7.8 在 Pyro 中构建*游戏内购买*的因果模型

```py
import pyro
from torch import tensor
from pyro.distributions import Bernoulli, Normal

def model():
    p_member = tensor(0.5)    #1
    is_guild_member = pyro.sample(    #1
        "Guild Membership",     #1
        Bernoulli(p_member)     #1
    )     #1
    p_engaged = (tensor(0.8)*is_guild_member +    #2
                 tensor(.2)*(1-is_guild_member))    #2
    is_highly_engaged = pyro.sample(    #2
        "Side-quest Engagement",     #2
        Bernoulli(p_engaged)     #2
    )     #2
    get_purchase_param = lambda param1, param2, param3, param4: (     #3
        param1 * (1-is_guild_member) * (1-is_highly_engaged) +     #3
        param2 * (1-is_guild_member) * (is_highly_engaged) +   #3
        param3 * (is_guild_member)   * (1-is_highly_engaged) +    #3
        param4 * (is_guild_member)   * (is_highly_engaged)     #3
    )     #3
    μ = get_purchase_param(37.95, 54.92, 223.71, 125.50)     #4
    σ = get_purchase_param(23.80, 4.92, 5.30, 53.49)    #4
    in_game_purchases = pyro.sample(    #4
        "In-game Purchases",   #4
        Normal(μ, σ)     #4
    )   #4
    guild_membership = "member" if is_guild_member else "nonmember"     #5
    engagement = "high" if is_highly_engaged else "low"   #5
    in_game_purchases = float(in_game_purchases)     #5
    return guild_membership, engagement, in_game_purchases     #5
```

#1 公会成员与非成员的概率为.5。使用这个概率，我们从伯努利分布中生成了一个公会成员资格值（成员为 1，非成员为 0）。

#2 我们从伯努利分布（高为 1，低为 0）中生成了一个支线任务参与度的值，其参数取决于公会成员资格。

#3 用于计算游戏内购买参数的辅助函数

#4 我们使用在观测数据中找到的样本均值来指定游戏内购买的正态分布的位置参数。

#5 与均值参数一样，我们使用在数据中找到的标准差来指定游戏内购买的正态分布的尺度参数。

为了确认 Pyro 模型编码了一个因果 DAG，我们可以运行`pyro.render_model(model)`，这将产生图 7.4。

![figure](img/CH07_F04_Ness.png)

### 图 7.4 调用`pyro.render_model`与因果模型的结果

## 利用概率编程的参数灵活性

注意我们在 Pyro 模型中建模变量的选择灵活性。例如，在建模*游戏内购买*的分布时，我们使用了正态分布，但我们可以使用其他分布。例如，*游戏内购买*不能是负数，因此我们可以选择一个仅对正数定义的规范分布，而不是对负数和正数都定义的正态分布。这对于低*支线任务参与度*的非公会成员特别有用，因为从均值为 37.95 和尺度参数为 23.80 的正态分布生成将大约有 5.5%的概率生成负值。然而，我们选择在这个情况下稍微偷懒，使用正态分布，因为几个负数的*游戏内购买*对我们的分析结果影响不大。

重点是，概率编程工具如 Pyro 为我们提供了参数灵活性，与 pgmpy 等工具不同。利用这种灵活性来反映你对 DGP 的假设是一种好的做法。

## Pyro 的干预抽象

Pyro 在 `pyro.do` 中提供了一个表示干预的抽象。它接受一个模型并返回一个新的模型，该模型反映了干预。列表 7.9 展示了我们可以如何使用 `pyro.do` 将之前的模型转换为反映将 *侧任务参与度* 设置为“高”和“低”的干预的模型。

### 列表 7.9 使用 `pyro.do` 表示干预

```py
int_engaged_model = pyro.do(    #1
    model,    #1
    {"Side-quest Engagement": tensor(1.)}   #1
)    #1
int_unengaged_model = pyro.do(     #2
    model,    #2
    {"Side-quest Engagement": tensor(0.)}     #2
)    #2
```

#1 将 *侧任务参与度* 设置为 1.0（即，“高”）。这将返回一个新的模型。

#2 将 *侧任务参与度* 设置为 0.0（即，“低”）。这将返回一个新的模型。

现在我们有两个新的模型：一个将 *侧任务参与度* 设置为“高”，另一个将其设置为“低”。如果我们的原始模型是正确的，从每个新干预模型中生成 500 个示例，并将它们合并成 1000 个示例，实际上 *模拟* 了实验。记住，我们仅使用表 7.4 中展示的观察数据来估计这个因果模型的参数。如果我们能在观察数据上训练一个模型，并使用它来准确模拟实验结果，那么我们就无需实际运行实验。

列表 7.10 使用 `int_engaged_model` 和 `int_unengaged_model` 来模拟实验数据。我们可以通过比较此模拟数据的汇总统计与实际实验数据的汇总统计来确认模拟的有效性。

### 列表 7.10 使用 `pyro.do` 干预模拟实验数据
