
一个冷却计划：`cooling_schedule='geometric'`（可用选项：`'linear'`、`'geometric'`、`'logarithmic'`、`'exponential'`、`'linear_inverse'`）

一个冷却因子：`cooling_alpha=0.9`

一个调试选项：`debug=1`（`debug=1` 打印初始和最终解；`debug=2` 提供手动迭代跟踪）

随意更改这些设置并观察它们对算法性能的影响。

## 列表 5.2 使用 SA 进行连续函数优化

```py
import random
import math
import numpy as np
from optalgotools.algorithms import SimulatedAnnealing
from optalgotools.problems import ProblemBase, ContinuousFunctionBase

def simple_example(x):                                                       ①
    return (x-6)**2                                                          ①

simple_example_bounds = np.asarray([[0, 31]])                                ①
simple_example_obj = ContinuousFunctionBase(simple_example, simple_example_  ①
➥ simple_example_bounds)                                                    ①
sa = SimulatedAnnealing(max_iter=1000, max_iter_per_temp=100,                ①
➥ initial_ temp=1000, final_temp=0.0001, cooling_schedule='geometric',      ①
➥ cooling_alpha=0.9, debug=1)                                               ①
sa.run(simple_example_obj)                                                   ①

def Bohachevsky(x_1, x_2):                                                   ②
    return x_1**2 +2*(x_2**2)-0.3*np.cos(3*np.pi*x_1)-0.4*np.cos(4*np.       ② 
pi*x_2)+0.7                                                                  ②

Bohachevsky_bounds = np.asarray([[-100, 100], [-100, 100]])                  ②
Bohachevsky_obj = ContinuousFunctionBase(Bohachevsky, Bohachevsky_bounds, 5) ②
sa.run(Bohachevsky_obj)                                                      ②

def bukin(x_1, x_2):                                                         ③
    return 100*math.sqrt(abs(x_2-0.01*x_1**2)) + 0.01 * abs(x_1 + 10)        ③

bukin_bounds = np.asarray([[-15, -5], [-3, 3]])                              ③
bukin_obj = ContinuousFunctionBase(bukin, bukin_bounds, 5)                   ③
sa.run(bukin_obj)                                                            ③

def gramacy_and_lee(x):                                                      ④
    return math.sin(10*pi*x)/(2*x) + (x-1)**4                                ④

gramacy_and_lee_bounds = np.asarray([[0.5, 2.5]])                            ④
gramacy_and_lee_obj = ContinuousFunctionBase(gramacy_and_lee, gramacy_and_   ④
➥ lee_bounds, .1)                                                           ④
sa.run(gramacy_and_lee_obj)                                                  ④

def griewank(*x):                                                            ⑤
    x = np.asarray(x)                                                        ⑤
    return np.sum(x**2/4000) - np.prod(np.cos(x/np.sqrt(np.asarray(range(1,  ⑤
    ➥ len(x)+1))))) + 1                                                     ⑤

griewank_bounds = np.asarray([[-600, 600]])                                  ⑤
griewank_1d=ContinuousFunctionBase(griewank, griewank_bounds, 10)            ⑤
sa.run(griewank_1d)                                                          ⑤

griewank_bounds_2d = np.asarray([[-600, 600]]*2)                             ⑥
griewank_2d=ContinuousFunctionBase(griewank, griewank_bounds_2d,             ⑥
➥ (griewank_bounds_2d[:, 1] - griewank_bounds_2d[:, 0])/10)                 ⑥
sa.run(griewank_2d)                                                          ⑥

griewank_bounds_3d = np.asarray([[-600, 600]]*3)                             ⑦
griewank_3d=ContinuousFunctionBase(griewank, griewank_bounds_3d,             ⑦
➥ (griewank_bounds_3d[:, 1] - griewank_bounds_3d[:, 0])/10)                 ⑦
sa.run(griewank_3d)                                                          ⑦
```

① 二次函数 SA 的解决方案

② 基于 Bohachevsky SA 的解决方案

③ 基于 Bukin SA 的解决方案

④ 基于 Gramacy & Lee SA 的解决方案

⑤ 基于 Griewank 1D SA 的解决方案

⑥ 基于 Griewank 2D SA 的解决方案

⑦ Griewank 3D SA-based solution

这是 Bukin 函数输出的一个示例：

```py
Simulated annealing is initialized:
current value = 60.73784664253138, current temp=1000
Simulated Annealing is done: 
curr iter: 154, curr best value: 0.6093437608551259, curr temp:9.97938882337113e-05, curr best: sol: [-14.63282848   2.14122839]
global minimum: x = -14.6328, 2.1412, f(x) = 0.6093
```

如所示，模拟退火（SA）能够处理不同的多维、非线性函数优化问题。这个随机全局优化算法能够适应目标函数的景观并避免陷入局部最小值。然而，在多维函数如 Griewank 2D 和 3D 的情况下，SA 需要时间才能收敛。以下章节将展示 SA 如何处理如数独和 TSP 之类的离散问题。

## 5.4 解数独

数独，也称为苏杜克，是有史以来最受欢迎的数字谜题之一。这个游戏是从一个称为*拉丁方*的数学概念改编而来的。数独谜题的第一个版本是由一位名叫霍华德·加恩斯的退休建筑师创造的，于 20 世纪 70 年代末作为《Dell 铅笔谜题和文字游戏》中的谜题出现。该游戏随后于 1984 年在日本以“数独”的名字出现，这个名字是从日语“数は独り立ちます”缩写而来，意味着数字（或数字）必须保持单一。如今，数独游戏在全球范围内都很受欢迎，并发表在游戏网站、谜题手册和报纸上。

数独游戏可以被视为一个约束满足问题（CSP），通过正确填充一个 9 × 9 的网格，使得每一列、每一行以及九个 3 × 3 的子网格（又称“盒子”、“区块”或“区域”）都包含从 1 到 9 的所有数字。任何一行、一列或 3 × 3 的子网格都不应包含超过一个相同的数字（从 1 到 9）。

除了娱乐之外，数独在现实生活中的应用还包括发展心理学和隐写术。例如，有几项研究表明，解决数独或填字游戏或其他脑力游戏可能有助于保持你的大脑年轻 10 年，并可以减缓如阿尔茨海默病等疾病的发展。数独还可以用作提高解决问题的能力、批判性思维和注意力的工具。最后，隐写术是将图像、消息、文件或其他秘密数据隐藏在非秘密事物中的技术。在秘密数据传输应用中，数字图像可以用来隐藏秘密数据。然后，使用数独谜题根据特别设计的参考矩阵修改封面图像中的选定像素对，以插入秘密数字。

### 拉丁方

拉丁方是由 10 世纪的阿拉伯数字学家发明的，他们处理数字的神秘力量。13 世纪的伊斯兰护身符，称为 wafq majazi，被发现，它们被绘制在 16 世纪阿拉伯医学文本的边缘。名字“拉丁”是受到著名的瑞士数学家莱昂哈德·欧拉（1707–1783）的启发，他在方阵中使用拉丁字母作为符号。

*拉丁方*是一个*n* × *n*的数组，填充了*n*个不同的数字、符号或颜色，以这样的方式排列，即任何正交（行或列）都不会包含相同的数字、符号或颜色两次。这里展示了一个 4 × 4 拉丁方的例子：

![](img/CH05_F10_UN03_Khamis.png)

拉丁方与*幻方*不同。幻方是一个正整数 1, 2, ..., *n*²的平方数组，这些数按照这样的方式排列，即任何水平、垂直或主对角线上的*n*个数的和总是相同的数。数独基于拉丁方。事实上，任何数独谜题的解决方案都是一个拉丁方。KenKen 和 KenDoku 是基于拉丁方增强版本的其它数字谜题，需要一定程度的算术技能。

通常来说，数独的搜索空间非常庞大。存在 6.671 × 10²¹种可能的可解数独网格，它们会产生一个唯一的结果[5]。根据大英百科全书，如果地球上每个人每秒解决一个数独谜题，他们要到大约 30,992 年才能全部解决。然而，去除对称性，例如旋转、反射、列和行的排列以及数字的交换，本质上不同的数独网格数量减少到 5,472,730,538 ≈5.473 × 10⁹[6]。广义的 *n* × *n* 数独问题是一个 NP 完全问题。然而，一些实例，例如标准的 9 × 9 数独，并不是 NP 完全的。存在常数时间算法可以解决一些 9 × 9 数独的实例，在*O*(1)时间内解决，因为每个 9 × 9 数独都可以在一个有限的字典或查找表中列出、枚举和索引，以找到解决方案。然而，这些算法无法处理任意的广义 *n* × *n* 数独问题。

回溯法、舞链法和克罗克的纸笔法是解决数独的常见算法，尤其是当问题规模较小时。*回溯法*主要是一种经典的深度优先搜索，它会测试整个分支，直到该分支违反规则或返回一个解决方案。

*舞链法*（DLX），由唐纳德·克努斯于 2000 年发明，使用算法 X 来解决数独谜题，将其作为精确覆盖问题处理。在精确覆盖问题中，给定一个二元矩阵（即仅由 0 和 1 组成的矩阵），需要找到一组行，每列恰好包含一个 1。算法 X，一种递归搜索算法，通过回溯法应用于解决精确覆盖问题。

在*克罗克的铅笔和纸算法*中，每个单元格中所有可能的数字都被列出。这个数字列表被称为单元格的标记。然后我们尝试找出是否有行、列或块中只有一个可能的值贯穿整个行、列或块。一旦找到，我们就用这个数字填写这个单元格，并更新任何受影响的行、列或框的标记。下一步是找到先发制人的集合。正如克罗克的论文中描述的那样，先发制人的集合由集合[1,2,…,9]中的数字组成，是一个大小为*m*的集合，2 ≤ *m* ≤ 9，其数字是*m*个单元格的潜在占用者，其中“独家”意味着集合[1,2,…,9]中的其他数字（除了先发制人集合的成员）不可能是那些*m*个单元格的潜在占用者。最后一步是消除先发制人集合外的可能数字。

### 回溯

回溯算法通常通过递归用于解决搜索和优化问题。回溯算法增量地构建一个可行解或一组可行解。给定一个 9×9 数独板，算法按照深度优先遍历顺序访问所有空单元格，增量地填写数字，并在找不到有效数字时回溯。以下图展示了 9×9 数独谜题的回溯算法步骤。

![](img/CH05_F10_UN04_Khamis.png)

9×9 数独的回溯步骤

下一个列表展示了如何使用 SA 算法解决 9×9 数独。

### 列表 5.3 使用 SA 解决数独

```py
from optalgotools.algorithms import SimulatedAnnealing                 ①
from optalgotools.problems import Sudoku                               ②

sa = SimulatedAnnealing(max_iter=100000, max_iter_per_temp=1000, 
➥ initial_temp=500, final_temp=0.001, cooling_schedule='geometric',
➥ cooling_alpha=0.9, debug=1)                                         ③

Sudoku_hard = [
    [9, 0, 0, 1, 0, 0, 0, 5, 4],
    [0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 5, 0, 0, 9, 0, 0, 3],
    [0, 9, 0, 0, 3, 5, 0, 4, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [4, 1, 0, 2, 6, 0, 0, 8, 0],
    [7, 0, 0, 3, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 4, 0, 0, 0, 0],
    [3, 5, 0, 0, 0, 1, 0, 0, 6],
]
sudoku_prob = Sudoku(Sudoku_hard)    
sudoku_prob.print()                                                   ④
                                                                      ④
sudoku_prob.solve_backtrack()                                         ⑤
sa.run(sudoku_prob, 0)                                                ⑥
```

① 导入 SA 求解器。

② 导入数独问题。

③ 使用所选参数创建 SA 求解器。

④ 创建一个硬 9×9 数独（可用变体包括简单、容易、中等、困难、邪恶）。

⑤ 使用回溯算法解决数独。

⑥ 使用 SA 解决数独。

您可以通过更改谜题配置来尝试不同的数独变体。在简单的数独问题中，单元格包含比中等或困难问题更多的预填数字。邪恶数独是谜题难度的最高级别。表 5.2 比较了 SA、回溯和 Python 线性规划（PuLP）库解决不同 9×9 数独实例所需的时间。PuLP 提供线性混合编程求解器。PuLP 中使用的默认求解器是 Cbc（COIN-OR 分支和切割），它是一个用于混合整数线性规划问题的开源求解器。有关 PuLP 的更多信息，请参阅附录 A。

### 表 5.2 SA 与回溯与 PuLP 在解决 9×9 数独谜题中的比较

| 找到解的时间 | 简单 | 容易 | 中等 | 困难 | 邪恶 |
| --- | --- | --- | --- | --- | --- |
