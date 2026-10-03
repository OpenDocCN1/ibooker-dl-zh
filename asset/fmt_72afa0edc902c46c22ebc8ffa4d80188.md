
TS 中纳入的一种自适应方法是允许短期记忆（禁忌期）的长度动态变化，并在指示器识别出有希望的领域时加强搜索，或者在改进似乎最小或检测到局部最优时促进多样化。例如，你可以为禁忌期设置一个下限`L[min]`和一个上限`L[max]`。然后，如果在上一个迭代中解决方案有所改进，你可以通过减去 1 来递减禁忌期，以便搜索将集中在潜在改进的区域。如果在上一个迭代中解决方案恶化，你可以通过增加 1 来递增禁忌期，以引导搜索远离显然不良的区域，如图 6.7 所示。`L[min]`和`L[max]`的值可以在每特定次数的迭代后随机更改。

![](img/CH06_F07_Khamis.png)

图 6.7 动态控制禁忌期

反应性 TS 通过自动学习最优禁忌期来防止循环发生[4]。在此方法中，考虑了两种可能的反应机制。一种称为**立即反应机制**，通过增加禁忌期来阻止额外的重复。经过若干次`R`立即反应后，几何增长足以打破任何极限循环。第二种机制，称为**逃逸机制**，计算重复多次（超过`REP`次）的移动次数。当这个数字超过预定义的阈值`REP`时，强制执行多样化的逃逸运动。在创建禁忌搜索的适应性版本时，也可以考虑其他算法参数，例如应用基于频率的记忆或渴望标准。

现在你已经很好地理解了禁忌搜索的各个组成部分，让我们来探讨这个算法如何被用来解决各种优化问题。

## 6.3 解决约束满足问题

`n`后问题是一个经典的谜题，它涉及将`n`个棋后放置在`n`×`n`的棋盘上，使得没有两个棋后相互威胁。换句话说，没有两个棋后应该共享同一行、列或对角线。这是一个不定义显式目标函数的约束满足问题（CSP）。假设我们正在尝试使用禁忌搜索解决 7 后问题。在此问题中，图 6.8a 中显示的初始随机配置中的冲突数为 4：{Q1–Q2}，{Q2–Q6}，{Q4–Q5}和{Q6–Q7}。

![](img/CH06_F08_Khamis.png)

图 6.8 7 后问题的 TS 初始化。在左侧，虚线显示了皇后之间的 4 次冲突。在中间，C 代表放置皇后 Q 的列。在右侧，*表示给出最佳邻近解的交换。

图 6.8a 中的初始解可以表示为图 6.8b 中所示的顺序。可以通过如图 6.8c 所示的方式交换生成多个候选邻近解。交换（Q1,Q7）、（Q2,Q3）、（Q2,Q6）和（Q5,Q6）给出相同的值，所以我们可以假设（Q1,Q7）是任意选择的移动，以给出一个新解，如图 6.9 所示。在初始迭代中，Q1 被放置在第 4 列，Q7 被放置在第 2 列。交换 Q1 和 Q7 意味着将 Q1 放置在第 2 列，Q7 放置在第 4 列。

![图片](img/CH06_F09_Khamis.png)

图 6.9 一个 7 皇后问题—TS 迭代 1

碰撞次数现在减少到 2，分别是{Q2-Q6}和{Q4-Q5}。禁忌结构更新如图 6.9c 所示，禁止最近执行的交换（Q1,Q7）进行三次迭代，假设禁忌期为 3。

在下一次迭代中，可以通过交换 Q2 和 Q4 生成其他邻近解，如图 6.10 所示。交换（Q2,Q4）给出一个新候选解，因为它减少了 1 次碰撞。此解决方案相关的碰撞数是 1。禁忌结构更新，搜索继续。

![图片](img/CH06_F10_Khamis.png)

图 6.10 一个 7 皇后问题—TS 迭代 2

在下一次迭代（图 6.11）中，选择交换（Q1,Q3）作为给出新解决方案的移动。

![图片](img/CH06_F11_Khamis.png)

图 6.11 一个 7 皇后问题—TS 迭代 3

在新的迭代（图 6.12）中，选择交换（Q5,Q7）。图 6.12a 中的`T`表示禁忌活跃移动。

![图片](img/CH06_F12_Khamis.png)

图 6.12 一个 7 皇后问题—TS 迭代 4

在下一次迭代中，选择交换（Q4,Q7）（图 6.13）。

![图片](img/CH06_F13_Khamis.png)

图 6.13 一个 7 皇后问题—TS 迭代 5

在下一次迭代中，由于改进的交换是禁忌活跃的，我们可以应用渴望标准来选择交换（Q1,Q3），因为在此交换出禁忌列表之前只剩下一个迭代（图 6.14）。

![图片](img/CH06_F14_Khamis.png)

图 6.14 一个 7 皇后问题—TS 迭代 6

基于此解决方案，板配置将如图 6.15 所示。这是各种可能解决方案之一。

![图片](img/CH06_F15_Khamis.png)

图 6.15 通过手动迭代生成的 7 皇后解决方案

让我们探讨如何使用 Python 通过禁忌搜索来解决这个问题。首先，我们将导入以下 Python 库，用于随机数生成、多维数组和绘图。然后，我们将定义一个函数来生成`n`-皇后板的随机配置，基于预定义的板大小。

列表 6.1 解决 7 皇后问题

```py
import random
import numpy as np
import matplotlib.pyplot as plt
def get_initial_state(board_size):
    queens = list(range(board_size))
    random.shuffle(queens)
    return queens
```

假设板大小为 7，调用此函数返回一个随机板配置，如`[0, 4, 1, 5, 6, 2, 3]`。这意味着 Q1、Q2、Q3、Q4、Q5、Q6 和 Q7 分别放置在第 1、5、2、6、7、3 和 4 列。

然后，我们可以定义一个函数来计算棋盘上相互攻击的皇后的数量。此函数定义如下：

```py
def num_attacking_queens(queens):
    board_size = len(queens)
    num_attacks = 0
    for i in range(board_size):
        for j in range(i + 1, board_size):
            if queens[i]==queens[j] or abs(queens[i] - queens[j]) == j - i:
                num_attacks += 1
    return num_attacks 
```

接下来，我们可以创建一个函数来确定最佳可能的移动，该移动可以减少棋盘上的攻击次数，同时确保该移动当前不在禁忌列表上（即，不是禁忌活跃的）。此函数定义如下：

```py
def get_best_move(queens, tabu_list):
    board_size = len(queens)
    best_move = None
    best_num_attacks = board_size * (board_size - 1) // 2
    for i in range(board_size):
        for j in range(board_size):
            if queens[i] != j:
                new_queens = queens.copy()
                new_queens[i] = j
                if str(new_queens) not in tabu_list:
                    num_attacks = num_attacking_queens(new_queens)
                    if num_attacks < best_num_attacks:
                        best_move = (i, j)
                        best_num_attacks = num_attacks
    return best_move
```

如你所注意到的，最佳攻击次数初始化为最大攻击次数，即`n * (n – 1) / 2`。在 7 皇后问题中，这个数字是 7 * 6 / 2 = 21。

我们还需要实现一个函数，根据预定义的禁忌任期更新禁忌列表。以下是此函数的定义：

```py
def update_tabu_list(tabu_list, tabu_tenure, move):
    tabu_list.append(str(move))
    if len(tabu_list) > tabu_tenure:
        tabu_list.pop(0)
```

以下函数执行禁忌搜索的步骤，接受如最大迭代次数、禁忌任期、在得出解决方案陷入停滞的结论之前，最大无改进移动次数以及初始解决方案等输入参数：

```py
def tabu_search(num_iterations, tabu_tenure, max_non_improvement, queens):
    num_non_improvement = 0
    best_queens = queens
    best_num_attacks = num_attacking_queens(queens)
    tabu_list = []

    for i in range(num_iterations):
        move = get_best_move(queens, tabu_list)
        if move is not None:
            queens[move[0]] = move[1]
            update_tabu_list(tabu_list, tabu_tennure, move)
            num_attacks = num_attacking_queens(queens)
            if num_attacks < best_num_attacks:
                best_queens = queens
                best_num_attacks = num_attacks
                num_non_improvement = 0
        else:
            num_non_improvement += 1
            if num_non_improvement >= max_non_improvement:
                break

    return best_queens, num_attacks
```

对于 7 大小的棋盘，最大迭代次数为 2,000，禁忌任期是 10，在考虑解决方案陷入停滞之前，最大无改进移动次数是 50。调用禁忌搜索给出解决方案`[5, 1, 4, 0, 3, 6, 2]`，如图 6.16 所示。

![](img/CH06_F16_Khamis.png)

图 6.16 由 Python 代码生成的 7 皇后解决方案

该实现的完整代码可以在本书 GitHub 仓库的列表 6.1 中找到。代码中使用的迭代次数作为停止标准。作为练习，你可以修改代码以添加一个停止标准，一旦找到零攻击的解决方案，就终止搜索。

`n`皇后问题是一个离散问题，因为它涉及到在离散棋盘上找到棋后可行配置。在下一节中，我们将探讨禁忌搜索如何应用于连续问题，以函数优化形式。

## 6.4 解决连续问题

作为连续问题的示例，让我们从函数优化开始。Himmelblau 函数（`f(x,y) = (x² + y – 11)² + (x + y² – 7)²`），以 David Mautner Himmelblau（1924–2011）的名字命名，是一个多模态函数，常被用作优化算法的测试问题。它是一个非凸函数，在`(3.0, 2.0)`，`(–2.805118, 3.131312)`，`(–3.779310, –3.283186)`，和`(3.584428, –1.848126)`有四个相同的局部最小值，如图 6.17 所示。

![](img/CH06_F17_Khamis.png)

图 6.17 Himmelblau 函数在`(3.0, 2.0)`，`(–2.805118, 3.131312)`，`(–3.779310, –3.283186)`，和`(3.584428, –1.848126)`有四个相同的局部最小值。

我们`optalgotools`包中提供了一个通用的 Tabu 搜索的 Python 实现。在这个实现中，使用哈希表或字典作为索引数据结构来实现 Tabu 结构。哈希表是一组键值对，没有重复的键。它可以用来快速检索数据，无论数据量有多大，因为它具有大`O(1)`的添加、获取和删除函数。

通用 TS 求解器接受以下参数：

+   最大迭代次数（默认`max_iter=1000`）
+   Tabu tenure（默认`tabu_tenure=1000`）
+   Neighborhood size（默认`neighbor_size=10`）
+   Aspiration criteria（默认`use_aspiration=True`）
+   剩余迭代次数以退出 Tabu（默认`aspiration_limit=None`）
+   Incorporating frequency-based memory（默认`use_longterm=False`）

下面的列表展示了如何使用在`optalgotools`中实现的通用 Tabu 搜索求解器来解决 Himmelblau 函数的最小化问题。

列表 6.2 使用 Tabu 搜索解决 Himmelblau 函数

```py
import numpy as np
from optalgotools.algorithms import TabuSearch                          ①
from optalgotools.problems import ProblemBase, ContinuousFunctionBase   ②

def Himmelblau(x,y):
       return (((x**2+y-11)**2) + (((x+y**2-7)**2)))                    ③

Himmelblau_bounds = np.asarray([[-6, 6], [-6, 6]])                      ④

Himmelblau_obj = ContinuousFunctionBase(Himmelblau, Himmelblau_bounds)  ⑤

ts = TabuSearch(max_iter=100, tabu_tenure=5, neighbor_size=50, use_aspiration=True,
➥ aspiration_limit=2, use_longterm=False, debug=1)                     ⑥
ts.run(Himmelblau_obj)                                                  ⑦
```

① 从`optalgotools`导入通用的 Tabu 搜索求解器。
② 导入连续问题库
③ 定义目标函数。
④ 定义边界。
⑤ 创建一个连续函数对象。
⑥ 定义 TS 求解器。添加`debug = 1`以打印初始和最终解。
⑦ 运行求解器。

运行此代码将为 Himmelblau 函数提供一个潜在解：

```py
Tabu search is initialized:
current value = 148.322
Tabu search is done: 
curr iter: 100, curr best value: 0.005569730862620958, curr best: sol: [3.00736837 1.98045825], found at iter: 21
```

适当的调整各种算法参数可以使你找到最优或近似最优解。附录 B 中提供了其他几个优化测试函数。你可以通过修改列表 6.2 来尝试不同的函数。

接下来，让我们看看 Tabu 搜索如何解决旅行商问题。

## 6.5 解决 TSP 和路由问题

让我们看看如何使用在 Google OR-Tools 中实现的 Tabu 搜索来解决 TSP 的 Berlin52 实例。这个数据集包含柏林市的 52 个地点（[`comopt.ifi.uni-heidelberg.de/software/TSPLIB95/STSP.html`](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/STSP.html)）。问题的目标是找到最短的旅行路线，该路线访问每个地点一次，然后返回起点。Berlin52 数据集的最短路线是 7,542，如前一章所述。

我们首先导入 TSP 问题类、OR-Tools 约束编程求解器和定义 OR-Tools 路由库中使用的各种枚举（`enums`）的协议缓冲模块。然后，我们将从`optalgotools`中实现的通用`tsp`类创建一个`tsp`对象。我们将提取感兴趣的点、节点或城市并计算成对距离。成对距离将被转换为整数，以满足 OR-Tools 的要求。然后，我们将问题数据以字典的形式存储。在这个字典中，`distance_matrix`将表示数据集中感兴趣点之间的成对距离。

列表 6.3 使用 OR-Tools 禁忌搜索解决 Belin52 TSP
