
```py
Inputs: A graph with edges, a source node, a destination node
Output: Shortest path from source to destination in the graph

Initialize *priority_queue* ← *source*
Initialize *found* ← False
Initialize *seen* ← *source*

While *priority_queue* is not empty and *found* is False do
    Set *node* ← priority_queue.pop()
    Update *seen* ← *node*
    Update node_cost ← cumulative distance from source
    If *node* is *destination* then
        Update *route* ← *node*.route()
        Update *found* ← True
    For child in node.expand() do
        If child in priority_queue then
            If child.priority < priority_queue[child].priority then
               Set priority_queue[child].priority = child.priority
        Else
            Update priority_queue ← child
        Update priority_queue[child].priority ← node_cost
Return route
```

UCS 是 Dijkstra 算法的一个变体，对于大型图非常有用，因为它耗时较少，空间要求也较少。与 Dijkstra 算法在开始时将所有节点以无限代价添加到队列中不同，UCS 逐渐填充优先队列。例如，考虑在图中找到每对节点之间最短路径的问题。随着图的大小和复杂性的增长，很快就会明显看出 UCS 更有效率，因为它不需要事先知道整个图。表 3.6 显示了 Dijkstra 算法和 UCS 在不同大小图上的处理时间差异。这些数字是在没有多进程或多线程的情况下，在 3.60 GHz 的 Intel Core i9-9900K 上使用 `Comparison.ipynb` 中的代码收集的，该代码可在书籍的 GitHub 仓库中找到。

**表 3.6 UCS 与 Dijkstra 算法比较**

| 图大小 = | V | + | E | | Dijkstra 时间 | 统一代价搜索(UCS)时间 |
| --- | --- | --- | --- | --- | --- |
| 108 | 0.25 s | 0.14 s |
| 628 | 84.61 s | 58.23 s |
| 1,514 | 2,082.97 s | 1,360.98 s |

注意，在我们的 8 数码问题上进行 UCS 搜索需要每个状态的距离属性（默认为 1），并且总体上生成相当不错的结果（大约使用 6.2 KB 的空间和探索了 789 个状态）。重要的是要注意，因为边的长度都相等，UCS 无法优先探索新节点。因此，解决方案失去了最短路径算法的优势，即优化更紧凑解决方案的能力。在下一章中，你将看到计算这些状态之间人工距离的方法，最终快速生成解决方案并最小化所需的步骤数。

### 3.4.3 双向 Dijkstra 搜索

双向搜索同时应用正向搜索和反向搜索。如图 3.33 所示，它从初始源状态 S→G 正向搜索，并从最终目标状态 G→S 反向搜索，直到它们相遇。

![](img/CH03_F33_Khamis.png)

**图 3.33 双向 Dijkstra 算法**

如图 3.33 所示，Dijkstra 搜索空间为*C*[1] = 4πr²，双向 Dijkstra 搜索空间由*C*[2] + *C*[3] = 2πr²表示。这意味着我们减少了大约两倍的搜索空间。以下算法显示了双向 Dijkstra 算法的步骤。

**算法 3.5 双向 Dijkstra 算法**

```py
Inputs: A graph, a source node, a destination node
Output: Shortest path from source to destination in the graph

Initialize *frontier_f* ← initialized with source
Initialize *frontier_b* ← initialized with destination
Initialize *explored_f* ← empty
Initialize *explored_b* ← empty
Initialize *found* ← False
Initialize *collide* ← False
Initialize *altr_expand* ← False

While *frontier_f* is not empty and *frontier_b* is not empty and not *collide* and not *found* do
    If altr_expand then
        Set node ← frontier_f.pop()
        Add node to explored_f
        For child in node.expand() do
            If child in explored_f then continue
            If child is destination then
                Update route ← child.route()
                Update found ← True
            If child in explored_b then
                Update route ← child.route() + reverse(overlapped.route())
                Update collide ← True
            Add child to frontier_f
        Update altr_expand ← not altr_expand
    Else
        Update node ← frontier_b.pop()
        Add node to explored_b
        For child in node.expand() do
            If child in explored_b then continue
            If child is origin then
               Update route ← child.route()
               Update found ← True
            If child in explored_f then
               Update route ← reverse(child.route()) + overlapped.route()
               Update collide ← True
            Add child to frontier_b
        Update altr_expand ← not altr_expand
Return route
```

这种方法由于涉及的时间复杂度更高，因此更有效率。例如，具有常数分支因子*b*和深度*d*的 BFS 搜索的整体空间复杂度为*O*(b^d*)。然而，通过以只有一半深度(*d*/2)运行两个相反方向的 BFS 搜索，空间复杂度变为*O*(b^d^(1/2) + b^d^(1/2))，或者简单地 O(b^d^(1/2))，这显著降低了。

图 3.34 显示了在探索多伦多市 50,841 个节点时，Dijkstra 算法和双向 Dijkstra 算法之间的差异。

![](img/CH03_F34_Khamis.png)

**图 3.34 Dijkstra 算法与双向 Dijkstra 算法——从左向右的前向探索和从右向左的后向探索**

## 3.5 将盲目搜索应用于路由问题

逻辑谜题游戏和简单的网格路由问题有助于理解算法的工作原理。然而，是时候看看一些使用这些算法的真实世界示例和结果了。例如，想象一下，当你正在多伦多的皇后公园参观爱德华七世骑马雕像时，你突然想起你有一个在多伦多大学的信息技术中心（Bahen Centre）的会议。我最初在讨论本章开头道路网络图时提出了这个问题。在考虑这个问题时，我们将做出以下假设：

+   由于你的手机电量耗尽，你无法打开导航应用或打电话求助。
+   你知道你的目的地在多伦多某个地方，但你不知道它相对于你的起始位置在哪里。（在后面的章节中，你将学习到知道目的地方向如何在极短的时间内生成近似最优解。）
+   一旦你开始使用一条规则来规划你的目的地，你就会坚持这条规则。

让我们看看我们如何可能使用 BFS、DFS、Dijkstra 算法、UCS 和双向 Dijkstra 算法来模拟我们的路径搜索技巧。这个示例的代码位于本书的 GitHub 仓库中（`Comparison.ipynb`）。图 3.35 至图 3.37 展示了这些盲目搜索算法生成的路径。

![图 3.35](img/CH03_F35_Khamis.png)

**图 3.35 使用 BFS 生成的最短路径。** BFS 在移动到下一层之前先搜索每一层。当图不是很宽且解决方案靠近根节点时，这种方法效果最佳。

![图 3.36](img/CH03_F36_Khamis.png)

**图 3.36 使用 DFS 生成的最短路径。** DFS 在回溯之前尽可能深入地搜索图。当图不是很深且解决方案位于根节点较远的地方时，这种方法效果最佳。

![图 3.34](img/CH03_F37_Khamis.png)

**图 3.37 使用 Dijkstra 算法、UCS 和双向 Dijkstra 算法生成的最短路径。** 这三种算法都会产生相同的解决方案（最优路由），但它们在内存使用和节点探索方面会有不同的处理方式。

值得注意的是，`NetworkX` 中的 `dijkstra_path` 函数使用 Dijkstra 方法来计算图中两个节点之间的最短加权路径。我们的 `optalgo-tools` 包还提供了不同图搜索算法的实现，例如 BFS、DFS、Dijkstra 算法、UCS 和双向 Dijkstra 算法。`optalgotools` 中对 Dijkstra 算法的实现已经修改，以便与我们的 OSM 数据一起工作，因为从地图生成的图自然会有自环和并行边。并行边可能导致不是最短可用路径的路线，因为路线长度在很大程度上取决于在生成特定路径时选择了哪条并行边。在图 3.38 中，从 0 到 2 的最短路径可能返回长度为 7，如果计算该路径时选择了连接 0 和 1 的顶部边，而选择底部边时长度为 3。

![图 3.38](img/CH03_F38_Khamis.png)

**图 3.38 并行边可能存在问题，因为找到最短路径取决于在图探索过程中选择了哪条并行边。**

自环也会给原始的 Dijkstra 算法带来麻烦。如果一个图中包含自环，到某个节点的最短路径可能来自该节点本身。在那个时刻，我们就无法生成一条路径（图 3.39）。

![图 3.39](img/CH03_F39_Khamis.png)

**图 3.39 自环可能会打断父子节点链，这阻止我们在找到解决方案后重新追踪路径。**

这两个问题通常容易解决但非同寻常，需要避免。对于并行边，我们选择权重最低（长度最短）的边，并丢弃任何其他并行边。对于自环，我们可以完全忽略该环，因为在大多数路由问题中不存在负权重环（道路的长度不能为负），而正权重环不能成为最短路径的一部分。此外，本书中使用的 Dijkstra 算法在找到目标节点时终止，而传统的实现方式只有在找到从根节点到所有其他节点的最短路径时才结束。

表 3.7 比较了 BFS、DFS、Dijkstra 算法和 UCS 在路径长度、处理时间、所需空间和已探索节点数方面的差异。从这些结果中可以看出，Dijkstra 算法、UCS 和双向 Dijkstra 算法产生最优结果，时间和空间成本各不相同。虽然 BFS 和 DFS 都能在最短时间内找到可行解，但提供的解不是最优的，在 DFS 的情况下甚至是不合理的。另一方面，DFS 需要事先知道整个图，这既昂贵又不太实用。为特定问题选择合适的搜索算法很大程度上涉及确定处理时间和空间需求之间的理想平衡。在后面的章节中，我们将探讨产生近似最优解的算法，这些算法通常在最优解既不可能找到又不太实用时使用。请注意，所有这些解决方案都是可行的；它们都能从点 A 到点 B 产生一个有效（如果有时复杂）的路径。

**表 3.7 比较了 BFS、DFS、Dijkstra 算法和 UCS，其中*b*是分支因子，*m*是搜索树的最大深度，*d*是最浅的图深度，*E*是边的数量，*V*是顶点的数量。**

| 算法 | 成本（米） | 处理时间（秒） | 空间（字节） | 已探索节点 | 最坏情况时间 | 最坏情况空间 | 最优性 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BFS | 955.962 | 0.015625 | 1,152 | 278 | *O*(*b^d*) | *O*(*b^d*) | No |
| DFS | 3347.482 | 0.015625 | 1,152 | 153 | *O*(*b^m*) | *O*(*bm*) | No |
| Dijkstra’s | 806.892 | 0.0625 | 3,752 | 393 | *O*(&#124;*E*&#124; + &#124;*V*&#124; log &#124;*V*&#124;) | *O*(&#124;*V*&#124;) | Yes |
| UCS | 0.03125 | 592 | 393 | *O*((*b* + &#124;*E*&#124;) * *d*) | *O*(*b^d*) | Yes |
| 双向 Dijkstra’s | 0.046875 | 3,752 | 282 | *O*(*b^d*^(/2)) | *O*(*b^d*^(/2)) | Yes |

在下一章中，我们将探讨如果我们利用领域特定知识而不是盲目搜索，如何优化搜索。我们将直接深入研究信息搜索方法，看看我们如何使用这些算法来解决最小生成树和最短路径问题。

## 摘要

+   传统的图搜索算法（盲目和信息搜索算法）是确定性搜索算法，它们探索图是为了一般发现或显式搜索。
+   图是一种非线性数据结构，由顶点和边组成。
+   盲（无信息）搜索是一种搜索方法，其中不使用关于搜索空间的信息。
+   广度优先搜索（BFS）是一种图遍历算法，它在考虑下一层的节点之前，先检查搜索树同一层上的所有节点。
+   深度优先搜索（DFS）是一种图遍历算法，它从根节点或初始节点或顶点开始，尽可能跟随一条分支，然后回溯以探索其他分支，直到找到解决方案或所有路径都耗尽。
+   深度限制搜索（DLS）是具有预定深度限制的 DFS 的约束版本，防止它探索超过一定深度的路径。
+   迭代加深搜索（IDS），或迭代加深深度优先搜索（IDDFS），通过增加深度限制直到达到目标，结合了 DFS 的空间效率和 BFS 的快速搜索。
+   Dijkstra 算法解决了加权图中具有非负边成本的单一源最短路径问题。
+   广度优先搜索（UCS）是 Dijkstra 算法的一个变体，它使用最低的累积成本来找到从源到目的地的路径。如果所有边的路径成本相同，则它与 BFS 算法等价。
+   双向搜索（BS）是正向搜索和反向搜索的结合。它同时从起点正向搜索和从目标反向搜索。
+   选择搜索算法涉及确定目标平衡，包括时间复杂度、空间复杂度以及搜索空间的前知等因素。
