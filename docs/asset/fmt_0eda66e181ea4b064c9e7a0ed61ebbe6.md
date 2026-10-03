
## 4.4 将启发式搜索应用于路由问题

让我们再次看看第 3.5 节中介绍的多伦多大学路由问题。我们需要找到从女王公园的国王爱德华七世骑马雕像到信息科技中心的最短路径。搜索空间由一个道路网络表示，其中交叉口和兴趣点（包括起点和终点）是节点，边用于表示带有权重（例如，距离、旅行时间、燃料消耗、转弯次数等）的道路段。让我们看看我们如何可以使用本章讨论的启发式搜索算法找到最短路径。

### 4.4.1 路由的爬山搜索

列表 4.7 使用了来自`optalgotools.routing`的两个函数，这些函数生成随机和子路线。虽然实际的 HC 算法是确定的，但随机化的初始路线意味着在不同的运行中可以得到不同的结果。为了应对这种情况，我们将使用更高的`n`值，这允许更广泛的子路线多样性，因此更有可能得到最优（或近似最优）的解决方案。

**列表 4.7 U of T 路由使用爬山算法**

```py
def Hill_Climbing(G, origin, destination, n=20):
    time_start = process_time()                                        ①
    costs = []                                                         ①

    current = randomized_search(G, origin.osmid, destination.osmid)    ②
    costs.append(cost(G, current))

    neighbours = list(islice(get_child(G, current), n))                ③
    space_required = getsizeof(neighbours)
    shortest = min(neighbours, key=lambda route: cost(G, route))

    while cost(G, shortest) < cost(G, current):
        current = shortest
        neighbours = list(islice(get_child(G, current), n))
        shortest = min(neighbours, key=lambda route: cost(G, route))
        costs.append(cost(G, current))

    route = current
    time_end = process_time()
    return Solution(route, time_end - time_start, space_required, costs)
```

① 跟踪时间和成本以进行比较。

② 随机生成一个初始路线。

③ 获取`k`个邻居（子节点）。

虽然列表 4.7 中的实现是确定的，但初始路线仍然是随机的。这意味着在不同的运行中可能会得到不同的结果。爬山搜索将返回一些相当不错的结果，因为路线函数中局部最优点的数量很少。然而，较大的搜索空间自然会有更多的局部最大值和平台，HC 算法会很快陷入困境。

图 4.47 显示了一个最终解决方案为 806.892 米，这恰好与第三章中 Dijkstra 算法生成的结果相同（一个最优解）。

![图片](img/CH04_F47_Khamis.png)

图 4.47 使用爬山算法生成的最短路径解决方案。这里显示的解决方案使用了一个`n`值为 100，这增加了总处理时间，但返回了更好、更一致的结果。

### 4.4.2 路由的束搜索

路由的束搜索将与 HC 搜索具有类似的格式，但不同之处在于在每个迭代中都会保留一个用于比较的“束”解决方案。4.8 列表的完整代码，包括图初始化和可视化，可以在本书的 GitHub 仓库中找到。

**列表 4.8 使用光束搜索算法的 U of T 路由**

```py
def get_beam(G, beam, n=20):
    new_beam = []
    for route in beam:
        neighbours = list(islice(get_child(G, route), n))                  ①
        new_beam.extend(neighbours)
    return new_beam

def Beam_Search(G, origin, destination, k=10, n=20):
    start_time = process_time()
    seen = set()                                                           ②
    costs = []                                                             ②
    beam = [randomized_search(G, origin.osmid, destination.osmid) for _ in ②
    ➥  range(k)]                                                          ②

    for route in beam:                                                     ③
        seen.add(tuple(route))

    pool = []
    children = get_beam(G, beam, n)
    costs.append([cost(G, r) for r in beam])
    for r in children:
        if tuple(r) in seen:
            continue
        else:
            pool.append(r)
            seen.add(tuple(r))
    pool += beam
    space_required = getsizeof(pool)
    last_beam = None
    while beam != last_beam:                                               ④
        last_beam = beam
        beam = heapq.nsmallest(k, pool, key=lambda r: cost(G, r))

        for route in beam:
            seen.add(tuple(route))

        pool = []
        children = get_beam(G, beam, n)
        costs.append([cost(G, r) for r in beam])
        for r in children:
            if tuple(r) in seen:
                continue
            else:
                pool.append(r)
                seen.add(tuple(r))
        pool += beam
        space_required = (
            getsizeof(pool) if getsizeof(pool) > space_required else
            ➥ space_required
        )
    route = min(beam, key=lambda r: cost(G, r))                            ⑤
    end_time = process_time()
    return Solution(
        route, end_time - start_time, space_required, np.rot90(costs))     ⑥
```

① 为光束中的每条路线生成子路线。

② 初始化空集合以跟踪已访问节点和路径成本。

③ 已看到的路线必须转换为元组，以便它们是可哈希的，可以存储在集合中。

④ 在生成新光束不再找到更好的解决方案之前，在每个迭代中保留`k`条最佳路线。

⑤ 最终路线是最后光束中的最佳路线。

⑥ 返回最终路线、其成本、处理时间和所需空间。

光束搜索在路由中特别昂贵，因为它们需要为每个光束生成多个子路线。像 HC 一样，生成更多子路线会导致搜索空间的更广泛渗透，因此更有可能返回一个接近或达到最优解的解决方案。图 4.48 显示了由光束搜索生成的最终解决方案。

![图片](img/CH04_F48_Khamis.png)

图 4.48 使用光束搜索算法的最短路径。此解决方案使用`k` = 10 和`n` = 20，这意味着为光束中的每条路线生成 20 条路线，并为每个新光束保留前 10 条路线。较低的`k`和`n`值将提高处理时间，但会降低生成接近最优或最优解决方案的可能性。

### 4.4.3 路由的 A*

下一个列表显示了我们可以如何使用 A*搜索找到两个感兴趣点之间的最短路线。

**列表 4.9 使用 A*的 U of T 路由**

```py
import osmnx
from optalgotools.routing import (cost, draw_route, astar_heuristic)
from optalgotools.structures import Node
from optalgotools.algorithms.graph_search import A_Star
from optalgotools.utilities import haversine_distance

reference = (43.661667, -79.395)                                       ①

G = osmnx.graph_from_point(reference, dist=300, clean_periphery=True,
➥ simplify=True)                                                      ②

origin = (43.664527, -79.392442)                                       ③
destination = (43.659659, -79.397669)                                  ④

origin_id = osmnx.distance.nearest_nodes(G, origin[1], origin[0])      ⑤ 
destination_id = osmnx.distance.nearest_nodes(G, destination[1],       ⑤
➥ destination[0])                                                     ⑤

origin = Node(graph=G, osmid=origin_id)                                ⑥
destination = Node(graph=G, osmid=destination_id)                      ⑥

solution = A_Star(G, origin, destination, astar_heuristic,             ⑦
➥ heuristic_kwargs={"measuring_dist": haversine_distance})            ⑦
route = solution.result
print(f"Cost: {cost(G,route)} m")                                      ⑧
print(f"Process time: {solution.time} s")                              ⑧
print(f"Space required: {solution.space} bytes")                       ⑧
print(f"Explored nodes: {solution.explored}")                          ⑧
draw_route(G,route) 
```

① 定义参考点以获取地图。

② 从参考点获取地图数据。

③ 定义起点坐标。

④ 定义终点坐标。

⑤ 找到最近的节点。

⑥ 创建节点对象。

⑦ 运行 A*算法。

⑧ 打印结果。

在预处理阶段，收缩层次消耗了大量的处理时间，但一个正确剪枝的图（即节点收缩顺序良好）可以允许进行更快的查询。虽然在一个有 21 个节点的图上，搜索空间的微小减少是可以忽略的，但某些图可以被剪枝高达 40%，在查询时可以节省显著的成本和时间。在 4.6 列表的例子中，从节点 1 到节点 5 的搜索探索了 11 个节点，而正常双向 Dijkstra 的原始节点是 16 个。这几乎减少了 33%！

```py
import networkx as nx

shortcuts = {}
shortest_paths = dict(nx.all_pairs_dijkstra_path_length(G,
➥ weight="weight"))
current_G = G.copy()                                                     ①
for node in G.nodes:
    current_G.remove_node(node)                                          ②
    current_shortest_paths = dict(
        nx.all_pairs_dijkstra_path_length(current_G, weight="weight")
    )                                                                    ③
    for u in current_shortest_paths:
        if u == node:
            continue
        SP_contracted = current_shortest_paths[u]
        SP_original = shortest_paths[u]
        for v in SP_contracted:
            if u == v or node == v:
                continue
            if (
                SP_contracted[v] != SP_original[v]
                and G.has_edge(node, u)
                and G.has_edge(node, v)
            ):
                G.add_edge(u, v, weight=SP_original[v],contracted=True)  ④
                shortcuts[(u,v)] = node                                  ④
```

① 复制主图，以便节点只从副本中移除，而不是从主图中移除。

② 通过从复制的图中移除节点来收缩节点。

③ 重新计算最短路径矩阵，现在节点已被收缩。

④ 添加一个快捷边来替换改变后的最短路径，并跟踪它，以便在以后查询时可以解包它。

你会注意到，前面的代码为每个收缩创建了两个快捷边，一个从`P`到`v`，另一个从`v`到`u`的反向边。由于我们使用的是无向图，这种重复没有影响，因为边（`u`, `v`）与边（`v`, `u`）相同。

查询生成的图需要一个简单的修改后的双向 Dijkstra 搜索，其中如果邻居节点在层次结构中低于当前节点，则取消资格进行扩展。为了本书的目的，我们将使用`networkx.algorithms.shortest_paths.weighted.bidirectional_dijkstra`，略有变化（只有比当前节点层次结构高的节点可以被探索）。作为列表 4.6 的延续，以下代码片段显示了查询过程。修改算法的完整代码可以在书中 GitHub 仓库的列表 4.6 中找到：

```py
sln = bidirectional_dijkstra(G, 1, 5, hierarchy, weight="weight")      ①

uncontracted_route = [sln.result[0]]                                   ②
for u, v in zip(sln.result[:-1], sln.result[1:]): ]]                   ②
    if (u, v) in shortcuts: ]]                                         ②
        uncontracted_route.append(shortcuts[(u, v)])                   ②
    uncontracted_route.append(v)                                       ②
```

① 使用 NetworkX 运行双向 Dijkstra 算法。

② 解包任何标记为收缩的边，并生成解包路线。

前面的代码将生成一个可以如图 4.46 可视化的解包路线。

![图片](img/CH04_F46_Khamis.png)

图 4.46 解包收缩边后的解决方案路径。双向 Dijkstra 算法返回的原始路线通过收缩边（8,5），然后解包为（8,3）和（3,5）。
