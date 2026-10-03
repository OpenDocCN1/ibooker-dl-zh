
```py
import networkx as nx
import osmnx
import matplotlib.pyplot as plt
from optalgotools.algorithms.graph_search import Kruskal

reference = (43.661667, -79.395)
G = osmnx.graph_from_point(
    reference, dist=1000, clean_periphery=True, simplify=True, network_
type="drive"
)                                                                     ①
fig, ax = osmnx.plot_graph(G)

undir_G = G.to_undirected()                                           ②
sorted_edges = sorted(undir_G.edges(data=True), key=lambda t:         ②
➥ t[2].get("length",1))                                              ②

mst = Kruskal(G, sorted_edges=True, edges= sorted_edges,
➥ graph_type=nx.MultiDiGraph)                                        ③

highlight_edges = ['b' if e in mst.edges else 'r' for e in G.edges]   ④
edge_alphas = [1 if e in mst.edges else 0.25 for e in G.edges]        ④
osmnx.plot_graph(
    G, 
    edge_linewidth=2, 
    edge_color=highlight_edges, 
    edge_alpha=edge_alphas
)                                                                     ④
plt.show()                                                            ④
```

① 使用驾驶网络仅关注可驾驶的道路。这可以防止代码使用道路和人行道组合构建 MST。

② 获取无向图副本，并按边长对道路网络边进行排序。

③ 使用预排序列表并指定图类型，从`optalgotools`调用`Kruskal`算法。

④ 通过突出显示包含的边来可视化最小生成树（MST）。

图 4.11 显示了`Kruskal`算法生成的结果 MST。图中的道路网络图可能看起来像一个大连通分量，但实际上并非如此。图中似乎连接相邻节点的单行道实际上并没有连接（你可以从 A 到 B，但不能反过来）。我们通过使用`NetworkX`中的`to_undirected`函数将有向图转换为无向图来克服这个问题。

列表中使用的`Kruskal`算法版本与泥泞城市问题中使用的相同。我们将其从`optalgotools`导入以减少所需的代码量。

![图片](img/CH04_F11_Khamis.png)

图 4.11 `Kruskal`算法生成的 MST。所有包含在 MST 中的边都被突出显示。MST 中没有环，且树的总权重最小化。

MST 在现实世界中有着广泛的应用，包括网络设计、图像分割、聚类和设施定位问题。当处理涉及预算的问题时，如规划网络，MST 特别有用，因为它们允许所有节点以最低的总成本连接。如前所述，信息搜索算法可以用来找到本节所述的 MST，以及下一节讨论的最短路径算法。

## 4.3 最短路径算法

信息搜索算法可以通过使用关于问题的知识（领域特定知识）来剪枝搜索，以找到两个节点之间的最短路径。这种知识以启发式函数的形式给出对目标距离的估计。信息搜索算法的例子包括岗位攀登、束搜索、最佳优先、A* 和收缩层次。以下小节将详细讨论这些算法。

### 4.3.1 岗位攀登算法

假设你正在试图在浓雾中攀登山顶。上山和下山的路径只有一条，但你不确定确切的山顶位置。因此，你只能通过查看你上一步之后的进展来判断你是否已经上山或下山。你如何知道你已经到达山顶？一个不错的猜测是当你不再上山时！

从一个已知（非优化）的函数解或初始状态开始，岗位攀登算法检查该解决方案的邻居，并选择更优化的邻居。这个过程会重复进行，直到找不到更好的解决方案，此时算法终止。

岗位攀登算法是一种局部贪婪搜索算法，它试图通过结合领域特定知识或启发式信息来提高深度优先搜索的效率，因此它可以被认为是一种有信息的深度优先搜索算法。算法 4.2 中应用于图搜索的岗位攀登算法的伪代码假设为最小化问题。

算法 4.2 岗位攀登算法

```py
Inputs: Source node, Destination node
Output: Route from source to destination

Initialize current ← random route from source to destination
Initialize neighbours ← children of current

While min(neighbours) > current do
   Set current ← min(neighbours)
   Update neighbours ← children of current
Return current as the route from source to destination
```

该算法在将节点后继者（根据它们的启发式值）添加到待扩展列表之前，对它们进行排序。这种算法在内存和计算开销方面要求非常低，因为它只是简单地记住当前的后继者作为当前正在工作的路径。它是一种非穷举技术；它不会检查整个树，因此其性能将相对较快。然而，尽管这种算法在凸问题上相对表现良好，但具有多个局部最大值的函数通常会得到一个不是全局最大值的答案。当存在高原（一组所有解决方案都同样优化的局部解决方案）时，它的表现也较差。

如图 4.12 所示，根据初始状态，岗位攀登算法可能会陷入局部最优。一旦它达到山顶，算法将停止，因为任何新的后继者都会在山脚下。这类似于在雾中爬山，达到一个较小的顶峰，并认为你已经到达了主峰。

![图片](img/CH04_F12_Khamis.png)

图 4.12 根据初始状态，岗位攀登算法可能会陷入局部最优。一旦它达到一个顶峰，算法将停止，因为任何新的后继者都会在山脚下。

简单的岗位攀登、最陡上升岗位攀登、随机岗位攀登和随机重启岗位攀登都是岗位攀登算法的变体，如图 4.13 所示。

![图片](img/CH04_F13_Khamis.png)

图 4.13 岗位攀登算法的变体

*简单爬山法*逐个检查邻近节点，并选择第一个优化目标函数的邻近节点作为下一个要探索的节点。*最陡上升或最陡下降爬山法*是简单爬山算法的一种变体，它首先检查当前状态的所有邻近节点，并选择一个最接近目标状态的邻近节点。*随机爬山法*是爬山算法的一个随机版本，它随机选择一个邻近节点，而不检查所有邻近节点。此算法根据该邻近节点的改进量决定是否移动到该邻近节点或检查另一个节点。随机重启爬山法或首次选择爬山法遵循尝试策略，并迭代搜索节点，在每一步中选择最佳节点，直到找到目标。如果陷入局部最大值，它将从新的随机初始状态重新开始过程。与其他爬山变体相比，如果存在高原、局部最优和脊，此算法更有可能到达目的地。

梯度下降算法

*梯度下降算法*在机器学习中被广泛用于训练模型和进行预测。梯度下降和爬山算法是两种基本不同的算法，不能相互混淆。与爬山不同，梯度下降可以看作是向山谷底部徒步。梯度下降是一种迭代算法，它观察局部邻居的斜率，并朝着斜率最陡或负梯度方向移动以优化连续可微函数。在最大化问题的情形下，*梯度上升*则朝着正梯度方向移动以优化目标函数。

如果函数是凸函数（即，如果任何局部最小值也是全局最小值）并且学习率选择得当，梯度下降算法通常收敛到全局最小值。爬山算法是一种启发式贪婪算法，容易陷入局部最优。它主要用于离散优化问题，如旅行商问题，因为它不需要目标函数是可微的。

假设我们有一个如图 4.14 所示的简单图。源节点是 S，目标节点是 G。

![图片](img/CH04_F14_Khamis.png)

图 4.14 以图的形式表示的 8 个兴趣点（POIs）道路网络

此图可以通过找到一个包含原始图所有顶点且连通且无环的生成树来转换为树，如图 4.15 所示。

![图片](img/CH04_F15_Khamis.png)

图 4.15 以树的形式表示的 8 个兴趣点（POIs）道路网络

如您所见，从 S 到 G 有多种路径，每种路径的成本不同。遵循爬山算法，S 和 G 之间的最短路径将是 S→A→C→E→G，如图 4.16 所示。

![图片](img/CH04_F16_Khamis.png)

图 4.16 使用爬山算法在 S 和 G 之间找到的最短路径

现在我们使用 8 个拼图问题来说明爬山方法。图 4.17 展示了爬山搜索的进展，使用不包括空白瓷砖的错位瓷砖数量作为启发式信息 h(n)。例如，在第 2 步中，瓷砖 1、4、6 和 7 放置错误，所以 h = 4。在第 3 步中，瓷砖 1 和 4 放置错误，所以 h = 2。

![图片](img/CH04_F17_Khamis.png)

图 4.17 使用爬山算法解决 8 个拼图问题。在每次迭代中，算法探索相邻状态，寻找最小启发式值。

列表 4.3 展示了 Python 中爬山算法的简单实现。代码通过最小化下一个节点的启发式值来选择探索的节点。更复杂的版本，涉及生成最短路径，将在本章末尾介绍。

列表 4.3 使用爬山法解决 8 个拼图问题

```py
import matplotlib.pyplot as plt

def Hill_Climbing(origin, destination, cost_fn):
    costs = [cost_fn(origin)]
    current = origin
    route = [current]
    neighbours = current.expand()                           ①
    shortest = min(neighbours, key=lambda s: cost_fn(s))    ②
    costs.append(cost_fn(shortest))
    route.append(shortest)

    while cost_fn(shortest) < cost_fn(current):
        current = shortest
        neighbours = current.expand()
        shortest = min(neighbours, key=lambda s: cost_fn(s))
        costs.append(cost_fn(shortest))
        route.append(shortest)
        if shortest == destination:                         ③
            break

    return route, costs

def misplaced_tiles(state: State):                          ④
    flat = state.flatten()                                  ④
    goal = range(len(flat))                                 ④
    return sum([0 if goal[i] == flat[i] else 1 for i in range(len(flat))])    
```

① 可以使用`expand()`函数生成当前状态的邻居节点。

② “最近”的邻居是成本函数最低的（即错位瓷砖最少）。

③ 如果达到目标状态，则终止算法。

④ 计算并返回不在目标位置上的错位瓷砖数量。

注意：`State`类和`visualize`函数在完整的列表中定义，可在本书的 GitHub 仓库中找到。

以下代码片段定义了拼图的初始状态和目标状态，并使用爬山法来解决拼图问题：

```py
init_state = [[1,4,2],
              [3,7,5],
              [6,0,8]]

goal_state = [[0,1,2],
              [3,4,5],
              [6,7,8]]

init_state = State(init_state)
goal_state = State(goal_state)

if not init_state.is_solvable():                             ①
    print("This puzzle is not solvable.")
else:
    solution, costs = Hill_Climbing(init_state, goal_state,
    ➥ misplaced_tiles)                                      ②
    plt.xticks(range(len(costs)))                            ③
    plt.ylabel("Misplaced tiles")                            ③
    plt.title("Hill climbing: 8 Puzzle")                     ③
    plt.plot(costs)                                          ③
    visualize(solution)                                      ③
```

① 检查是否有解决方案。

② 使用爬山法解决拼图问题

③ 绘制搜索进度，并可视化解决方案。

输出显示在图 4.18 中。

![图片](img/CH04_F18_Khamis.png)

图 4.18 显示了 8 个拼图问题爬山解决方案的状态。每个后续状态都是通过将其成本与邻居相比最小化来选择的。由于 8 个拼图问题不仅有一个明确定义的目标状态，而且这个目标状态是可以实现的，算法的终止条件（达到目标）与“山峰”的“顶峰”相吻合（在这种情况下是一个山谷，因为它是一个最小化问题）。

由于 8 个拼图问题使用启发式作为成本，它本质上变成了一个最小化问题。这种实现与标准爬山算法的不同之处在于，由于总可以找到解决方案，并且图是完全连接的（您可以通过一些瓷砖移动的组合从任何状态转换到另一个状态），算法最终保证找到最优解。更复杂的问题通常会生成接近最优的解决方案。

### 4.3.2 光束搜索算法

束搜索算法试图最小化广度优先算法的内存需求，因此它可以被视为一种启发式广度优先算法。在爬山法在整个运行过程中保持一个最佳状态的同时，束搜索在内存中保持*W*个状态，其中*W*是束宽度。在每次迭代中，它为每个状态生成邻居并将它们放入与原始束中的状态相同的池中。然后，它在每个级别上从池中选择最佳的*W*个状态成为新的束，其余状态被丢弃。这个过程然后重复。该算法在每个级别上仅扩展前*W*个有希望的节点。

这是一个非穷尽搜索，但也是一个危险的过程，因为可能会错过目标状态。由于这是一个局部搜索算法，它也容易陷入局部最优。当*w*等于每个级别的节点数时，束搜索与 BFS 相同。因为存在可能丢弃导致最优解的状态的风险，所以束搜索是不完整的（它们可能不会以解结束）。

算法 4.3 展示了将束搜索算法应用于图搜索的伪代码。

算法 4.3 束搜索算法

```py
Inputs: A source node, a destination node, and beam width w
Output: A route from source to destination

Initialize Seen ← nil
Initialize beam ← random w routes from source to destination
Add beam to seen
Initialize pool ← children of routes in the beam with consideration of seen + beam
Initialize last_beam ← nil
While beam is not last_beam do
   Update last_beam ← beam
   Update beam ← the best w routes from pool
   If last_beam == beam then break
   Add beam to seen
   Update pool ← children of routes in the beam + beam

Return optimal route in beam
```

在 2.3.1 节中，您看到 BFS 具有指数复杂度*O*(*b^d*)，其中*b*代表每个节点的最大分支因子，*d*是必须扩展以达到目标深度的深度。在束搜索的情况下，我们只在任何深度探索*w*×*b*个节点，与 BFS 相比，节省了许多不必要的节点。然而，找到最佳状态或路线需要排序，如果*w*×*b*是一个很大的数字，那么排序将是耗时的。可以使用束阈值来处理这个问题，其中根据启发式函数*h*(*n*)在某个阈值内选择最佳节点，并将所有超出此阈值的节点剪枝掉。

重新审视具有 8 个兴趣点的简单路由问题（图 4.14）并遵循束搜索算法（*w* = *2*），S 和 G 之间的最短路径将是 S-A-C-E-G，如图 4.19 所示。

![图片](img/CH04_F19_Khamis.png)
