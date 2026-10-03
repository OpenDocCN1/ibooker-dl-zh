
# 第七章：遗传算法

本章涵盖

+ 介绍基于种群的优化算法
+ 理解进化计算
+ 理解遗传算法的不同组成部分
+ 在 Python 中实现遗传算法

假设你正在进行一次寻宝任务，你不想独自搜索并空手而归。你可能会决定与一群朋友合作并共享信息。这种方法遵循基于种群的搜索策略，其中多个代理参与搜索过程。

在这次协作过程中，你可能会注意到一些猎人表现优于其他人。在这种情况下，你可能会选择只保留表现最好的猎人，并用新招募的成员替换表现不佳的猎人。这个过程类似于遗传算法等进化算法的工作原理，其中最适应的个体生存下来并将它们的特征传递给下一代。

在本章中，我们将介绍并讨论二进制编码的遗传算法作为一种进化计算算法。我们将探讨该算法的不同元素及其实现细节。下一章将讨论遗传算法的其他变体，如灰度编码遗传算法、实值遗传算法和基于排列的遗传算法。

## 7.1 基于种群的元启发式算法

基于种群的元启发式算法（P-元启发式），如遗传算法、粒子群优化和蚁群优化，利用多个代理来搜索最优或近似最优的全局解。由于这些算法从多样化的初始种群开始，它们自然更侧重于探索，从而有可能找到可能被基于轨迹（S-元启发式）算法遗漏的更好解决方案，后者更侧重于利用。

基于种群的元启发式算法可以根据其灵感来源分为两大类：*进化计算算法*和*群体智能算法*，如图 7.1 所示。

![图片](img/CH07_F01_Khamis.png)

图 7.1 元启发式算法

进化计算（EC）算法，正如其名所示，是受生物进化过程的启发。这些算法使用一组潜在解的种群，这些种群通过遗传操作，如变异和交叉，来创造新的后代，这些后代可能具有更好的适应度值。选择过程决定了种群中哪些个体被选中进行繁殖并创造下一代。遗传算法（GA）、差分进化（DE）、遗传编程（GP）、进化编程（EP）、进化策略（ES）、文化算法（CA）和协同进化（CoE）是进化计算算法的例子。

另一方面，群体智能（SI）算法受到蚂蚁、蜜蜂和鸟类等社会生物集体行为的启发，将在本书的第四章中讨论。这些算法使用一个相互作用的代理种群来寻找解决方案。它们使用各种机制，如通信、合作和自组织，来优化搜索过程。群体智能算法的例子包括粒子群优化（PSO）、蚁群优化（ACO）、人工蜂群（ABC）、萤火虫算法（FA）、蝙蝠算法（BA）和狼搜索算法（WSA）。

无论是进化计算还是群体智能算法，它们都是基于种群的算法，从候选解的初始种群开始搜索最优或近似最优解。初始种群的质量和多样性显著影响算法的性能和效率。一个构建良好的初始种群为搜索过程提供了一个良好的起点，并有助于算法快速收敛到搜索空间的希望区域。相反，一个构建不良的初始种群可能导致过早收敛到次优解，可能使算法陷入次优区域，或者可能需要更长的时间收敛到解。为了确保探索和利用之间的良好平衡，初始种群应该是多样化的，并覆盖广泛的潜在解。

在 El-Ghazali Talbi 的《元启发式算法：从设计到实现》[1]中，提供了基于种群元启发式算法的不同初始化策略的比较，基于三个关键方面：多样性、计算成本和初始解的质量。初始解可以通过伪随机过程或准随机搜索生成。初始解也可以顺序生成（顺序多样性）或并发生成（并行多样性）以实现非常高的多样性。启发式方法涉及使用局部搜索或贪婪方法来生成初始解。

如表 7.1 所示，伪随机策略提供了适中的多样性、低计算成本和低质量的初始解。准随机策略表现出更高的多样性，具有可比的计算成本和低质量的初始解。顺序多样性和并行多样性都表现出非常高的多样性，但前者产生适中的计算成本，而后者具有低计算成本；两种方法都导致低质量的初始解。相比之下，使用诸如局部搜索或贪婪启发式等启发式方法可以产生高质量的初始解，但多样性和计算成本较低。

表 7.1 基于种群的元启发式算法的初始化策略

| 初始化策略 | 多样性 | 计算成本 | 初始解的质量 |
| --- | --- | --- | --- |
| 伪随机 | 中等 | 低 | 低 |
| 近似随机 | 高 | 低 | 低 |
| 顺序多样化 | 非常高 | 中等 | 低 |
| 并行多样化 | 非常高 | 低 | 低 |
| 启发式（例如，局部搜索或贪婪启发式） | 低 | 高 | 高 |

使用随机化方法生成初始种群通常是有益的，其中候选者是从搜索空间的不同区域抽取的样本，以最大化找到最优解的机会。下面的列表展示了我们如何使用 Python 来采样初始解。让我们首先生成 200 个伪随机数。

列表 7.1 在 Python 中生成初始种群

```py
import math
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(6345245)                  ①

N=200                                    ②
P_random_pseudo=np.random.rand(N,N)      ③
```

① 为随机数生成器设置种子。
② 样本数量
③ 伪随机采样

注意：随机数本质上不可预测，伪随机数是确定的但看起来是随机的，而近似随机数是具有均匀分布模式的确定数。

`ghalton`库中的广义 Halton 数生成器可以用来生成近似随机数。这种方法基于 Halton 序列，它使用互质数作为其基数。你可以如下使用广义 Halton 数生成器：

```py
!pip install ghalton
import ghalton

sequencer = ghalton.GeneralizedHalton(7,23)
P_random_quasi = np.array(sequencer.get(N))
```

Box-Muller 变换用于从均匀分布的随机数对生成独立的标准正态分布随机数对。Box-Muller 是一种二维高斯采样方法，可以使用如下方式：

```py
u1 = np.random.uniform(size=(N))                        ①
u2 = np.random.uniform(size=(N))                        ①

P_BM_x = np.sqrt(-2*np.log(u1))*np.cos(2*math.pi*u2)    ②
P_BM_y = np.sqrt(-2*np.log(u1))*np.sin(2*math.pi*u2)    ②
```

① 生成 0 到 1 之间的均匀分布值。
② 使用 Box-Muller 计算 x 和 y 值。

Box-Muller 变换的一个缺点是它倾向于将值聚集在平均值周围，因为它依赖于均匀分布。此外，计算平方根可能很昂贵。

中心极限定理（CLT）采样是另一种采样方法，其中样本均值的分布随着样本量的增大而近似正态分布，而不管总体分布如何。以下代码片段展示了如何实现这种方法：

```py
import random

P_CLT_x=[2.0 * math.sqrt(N) * (sum(random.randint(0,1) for x in range(N)) / N - 0.5)
➥ for x in range(N)]
P_CLT_y=[2.0 * math.sqrt(N) * (sum(random.randint(0,1) for x in range(N)) / N - 0.5)
➥ for x in range(N)]
```

Sobol 低偏差序列（LDS）是 `sobol_seq` 包中可用的一种近似随机采样方法。这种方法生成一系列点，这些点在样本空间中均匀分布，使得相邻点之间的间隔尽可能小。它可以如下使用：

```py
!pip install sobol_seq
import sobol_seq
P_sobel=sobol_seq.i4_sobol_generate(2,N)
```

拉丁超立方体采样是一种并行多样化方法，其中搜索空间被分解成 25 个块，每个块中伪随机地生成一个解。下面展示了在 `pyDOE`（实验设计）Python 包中使用拉丁超立方体采样方法的示例：

```py
!pip install pyDOE
from pyDOE import *
P_LHS=lhs(2, samples=N, criterion='center')
```

让我们可视化所有这些采样方法，以便我们可以很好地了解它们之间的差异：

```py
f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,8))
f, (ax3,ax4) = plt.subplots(ncols=2, figsize=(18,8))
f, (ax5, ax6) = plt.subplots(ncols=2, figsize=(18,8))
ax1.scatter(P_random_pseudo[:,0], P_random_pseudo[:,1], color="gray")
ax2.scatter(P_random_quasi[:100], P_random_quasi[100:], color="red")
ax3.scatter(P_BM_x, P_BM_y, color="green")
ax4.scatter(P_CLT_x, P_CLT_y, color="cyan")
ax5.scatter(P_sobel[:,0], P_sobel[:,1], color="magenta")
ax6.plot(P_LHS[:,0], P_LHS[:,1], "o")

ax1.set_title("Pseudo-random")
ax2.set_title("Quasi-random")
ax3.set_title("Box-Muller")
ax4.set_title("Central Limit Theorem")
ax5.set_title("Sobol")
ax6.set_title("Latin Hypercube")
plt.show()
```

运行此代码生成图 7.2 所示的图表。在此图中，候选解已使用各种采样方法从可行搜索空间中采样，每个点代表一个不同的解。可以通过观察点之间的间隙及其分散程度来评估每种采样方法达到的多样性水平。

![](img/CH07_F02_Khamis.png)

图 7.2 生成初始种群的各种采样方法

如附录 A（见 liveBook）所述，有几个 Python 包用于进化计算。在本章中，我们将重点介绍使用 `pymoo`：Python 中的多目标优化。`Pymoo` 提供了不同的采样方法来创建初始种群或初始搜索点。例如，随机采样和拉丁超立方采样。作为列表 7.1 的延续，以下代码片段展示了 `pymoo` 中的随机采样：

```py
!pip install -U pymoo
from pymoo.core.problem import Problem                         ①
from pymoo.operators.sampling.rnd import FloatRandomSampling   ②
from pymoo.util import plotting                                ③

problem = Problem(n_var=2, xl=0, xu=1)                         ④

sampling = FloatRandomSampling()                               ⑤

X = sampling(problem, 200).get("X")                            ⑥
plotting.plot(X, no_fill=True)                                 ⑦
```

① 导入问题类的实例。
② 导入随机采样方法。
③ 导入可视化方法。
④ 创建一个有两个变量的问题，并指定下限和上限。
⑤ 创建随机采样器的实例。
⑥ 生成 200 个随机解/个体。
⑦ 可视化生成的个体。

以下代码使用拉丁超立方采样生成并可视化 200 个初始解：

```py
from pymoo.operators.sampling.lhs import LHS      ①

sampling = LHS()

X = sampling(problem, 200).get("X")
plotting.plot(X, no_fill=True)
```

① 导入拉丁超立方采样模块。

如果解的形式是排列，可以如下生成随机排列：

```py
per1=np.random.permutation(10)                               ①
print(per1)

per2 = np.array([5, 4, 9, 0, 1, 2, 6, 8, 7, 3])              ②
np.random.shuffle(per2)                                      ②
print(per2)

pop_init = np.arange(50).reshape((10,5))                     ③
np.random.permutation(pop_init)                              ③

from itertools import combinations                           ④
size=5                                                       ④
ones=2                                                       ④

for pos in map(set, combinations(range(size), ones)):        ④
     print([int(i in pos) for i in range(size)], sep='\n')   ④
```

① 随机排列一个序列，或返回一个排列的范围。
② 随机打乱一个序列。
③ 初始解的种群作为实值排列
④ 初始解的种群以二进制排列的形式，其中二进制字符串的位数和每个二进制字符串中 1 的个数

您也可以使用以下代码在两点之间生成一个随机路径：

```py
import osmnx as ox
import random
from collections import deque
from optalgotools.structures import Node

G = ox.graph_from_place("University of Toronto")
fig, ax = ox.plot_graph(G)

def randomized_search(G, source, destination):                  ①
    origin = Node(graph = G, osmid = source)
    destination = Node(graph = G, osmid = destination)

    route = [] 
    frontier = deque([origin])
    explored = set()
    while frontier:
        node = random.choice(frontier)                          ②
        frontier.remove(node)
        explored.add(node.osmid)

        for child in node.expand():
            if child not in explored and child not in frontier:
                if child == destination:
                    route = child.path()
                    return route
                frontier.append(child)

    raise Exception("destination and source are not on same component")

random_route = randomized_search(G, 24959528, 1480794706)       ③

fig, ax = ox.plot_graph_route(G, random_route)                  ④
```

① 这是一个典型的带有打乱边界的图搜索。
② 这部分是随机化部分。
③ 在两个节点之间生成随机路径。
④ 可视化随机路径。

前面的代码通过打乱边界节点修改了一个典型的图搜索算法。这意味着扩展的候选者是“随机”的，这意味着在重复调用时会产生不同的路径。一些生成的随机路径如图 7.3 所示。

![](img/CH07_F03_Khamis.png)

图 7.3 生成随机初始路径

在下一节中，我将介绍基于种群的元启发式算法——进化计算。

## 7.2 介绍进化计算

进化可以被视为一种优化过程，因为它涉及随着时间的推移逐渐改善生物体的特征，从而适应动态变化和竞争的环境，并增强在这些环境中生存的能力。在本节中，我将概述生物进化的基本概念。理解这些原理对于深入了解进化计算至关重要。

### 7.2.1 生物学基础简述

*细胞核*是任何活细胞的核心部分，其中包含遗传信息。这种遗传信息存储在*染色体*中，每条染色体都是由脱氧核糖核酸（DNA）构成的，它携带用于所有生物体生长、发育、功能和繁殖的遗传信息。人类在其每个细胞中都有 23 对染色体，即总共 46 条染色体。每条染色体由许多不同的部分组成，称为*基因*，这些基因负责编码个体的特定属性。决定这些属性的基因的变异形式，位于染色体上的特定位置，称为等位基因。每个基因在染色体上都有一个独特的位置，称为*座位*。所有基因的组合称为*基因型*，它为生物体提供遗传蓝图，决定个体的特征和特性的潜力。*表型*一词指的是生物体的可观察的物理、行为和生理特征，这些特征是其基因型和环境的相互作用的结果。
