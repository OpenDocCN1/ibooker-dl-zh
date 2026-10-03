
```py
class AmortizedModel(nn.Module):
    def __init__(self, n_params):            ①
        super().__init__()
        self.base = nn.Sequential(

            nn.Linear(n_params, n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, 3)
        )                                    ②

    def forward(self, p):                    ③
        squeeze = p.ndim == 1
        if squeeze:
            p = p.unsqueeze(0)
        assert p.ndim == 2
        z = self.base(p)
        z = z / z.norm(dim=-1, keepdim=True)
        if squeeze:
            z = z.squeeze(0)
        return z
```

① 将用作神经网络输入的 c-凸函数中的参数数量

② 按顺序定义神经网络的层。

③ 定义摊销模型的前向传递，它将输入 `p`（参数向量）映射到球面上的一个点。

我们现在可以训练一个摊销模型来学习从参数向量到球面上点的映射。它使用具有不同随机种子的 `c_convex` 函数（`fs`）列表来生成训练数据。摊销模型使用 Adam 优化器进行训练，其进度使用 `tqdm` 进度条进行可视化。球面上的输出点存储在张量 `xs` 中：

```py
n_hidden = 128                                               ①
torch.manual_seed(0)                                         ②
model = AmortizedModel(n_params=n_params)                    ③
opt = torch.optim.Adam(model.parameters(), lr=5e-4)          ④

xs = []                                                      ⑤
num_iterations = 100

pbar = tqdm(range(num_iterations), desc="Training Progress")

for i in pbar:                                               ⑥
    losses = []                                              ⑦
    xis = []                                                 ⑦
    for f in fs:                                             ⑧
        pred_opt = model(f.params)
        xis.append(pred_opt)
        losses.append(f(pred_opt))
    with torch.no_grad():                                    ⑧
        xis = torch.stack(xis)
        xs.append(xis)
    loss = sum(losses)

    opt.zero_grad()
    loss.backward()
    opt.step()

    pbar.set_postfix({"Loss": loss.item()})

xs = torch.stack(xs, dim=1)
```

① 设置 `AmortizedModel` 神经网络的隐藏单元数量。

② 设置随机种子以确保训练过程的可重复性。

③ 创建 `AmortizedModel` 的一个实例。

④ 创建一个学习率为 0.0005 的 Adam 优化器来更新参数。

⑤ 存储每次训练迭代的球面上的输出点。

⑥ 训练循环

⑦ 存储每个 `c_convex` 函数及其对应的球面输出点（`xis`）的损失。

⑧ 遍历列表 `fs` 中的每个 `c_convex` 函数（`f`）。

训练完成后，所有预测的输出点沿新维度堆叠在球面上，从而得到一个形状为“迭代次数，`c_convex` 函数数量，3”的张量 `xs`。这个张量中的每个元素代表在训练的不同阶段由摊销模型预测的球面上的一个点。它生成了摊销模型和 `c_convex` 函数的训练进度可视化表示，如图 11.15 所示。

![图片](img/CH11_F15_Khamis.png)

图 11.15 训练的摊销模型输出的示例

列表 11.3 的完整版本可在本书的 GitHub 仓库中找到。它创建了一个天体坐标网格，评估网格上的 `c_convex` 函数和摊销模型，然后绘制函数、预测路径和球面上的最优点的等高线图。最优点是给出最小损失的点，前提是使用监督学习来训练摊销模型。

## 11.7 使用监督图机器学习解决 TSP

Joshi、Laurent 和 Bresson 在他们的“图神经网络用于旅行商问题”文章[20]中提出了一种通用的端到端流程来解决组合优化问题，如旅行商问题（TSP）、车辆路径问题（VRP）、可满足性问题（SAT）、最大切割（MaxCut）和最大独立集（MIS）。图 11.16 显示了使用机器学习解决 TSP 的步骤。

![图片](img/CH11_F16_Khamis.png)

图 11.16 组合优化问题的端到端流程

按照这种方法，我们首先以节点特征和节点之间的邻接矩阵的形式定义图问题。然后，基于消息传递方法，使用 GNN 或 GCN 生成低维图嵌入。使用多层感知器（MLPs）预测节点或边属于解决方案的概率。然后，应用图搜索，如束搜索（见第四章），通过边的概率分布搜索图，以找到可行的候选解决方案。应用模仿学习（监督学习）和探索学习（强化学习）。监督学习最小化最优解（在 TSP 的情况下，由 Concorde 等知名求解器获得）与模型预测之间的损失。强化学习方法使用策略梯度来最小化解码结束时模型预测的旅行长度。强化学习将在下一章讨论。

从头开始训练一个机器学习模型并将其应用于解决 TSP 需要大量的代码和数据预处理。列表 11.4 展示了如何使用预训练模型来解决 TSP 的不同实例。我们首先导入我们将使用的库和模块。这些库提供了处理数据、执行计算、可视化和优化的功能。Gurobi 库用于优化过程中的子回路消除以及计算一组点的降低成本（见附录 A）。我们设置 `CUDA_DEVICE_ORDER` 和 `CUDA_VISIBLE_DEVICES` 环境变量以控制 GPU 设备的可见性。

列表 11.4 使用监督机器学习解决 TSP

```py
import os
import math
import itertools
import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel

from learning_tsp.problems.tsp.problem_tsp import TSP
from learning_tsp.utils import load_model, move_to

from gurobipy import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
```

作为延续，以下 `opts` 类包含几个类级别属性，定义了以下选项和配置：

+   `dataset path` — 书中 GitHub 仓库中可用的 TSP 数据集。

+   `batch size` — 这决定了在训练或评估期间同时处理的 TSP 实例（问题）数量。它指定了将多少个 TSP 实例分组在一起并行处理。

+   `number of samples` — 这是每个 TSP 大小下的样本数量。

+   `neighbors` — 在 TSP 数据处理管道中使用，用于指定用于图稀疏化的最近邻的比例（百分比）。它通过为每个节点选择最近邻的子集来控制 TSP 图的连通性。

+   `knn strategy` — 这是在执行图稀疏化时确定最近邻数量的策略。在代码中，`'percentage'` 值表示最近邻的数量由 `neighbors` 参数确定，该参数指定了要考虑的邻居百分比。

+   `model` — 这是预训练机器学习模型的路径。所使用的模型是书中 GitHub 仓库中可用的预训练 GNN 模型。

+   `use_cuda` — 这检查系统上是否有 CUDA 可用。CUDA 是一个并行计算平台和编程模型，允许在 NVIDIA GPU 上高效执行计算。`torch.cuda.is_available()` 返回一个布尔值（true 或 false），指示 CUDA 是否可用。如果 CUDA 可用，则表示系统上存在兼容的 NVIDIA GPU，并且可以用于加速计算。

+   `device` — 这是用于计算的设备：

```py
class opts:
    dataset_path = "learning_tsp/data/tsp20-50_concorde.txt"
    batch_size = 16
    num_samples = 1280 

    neighbors = 0.20
    knn_strat = 'percentage'

    model = 
➥ "learning_tsp/pretrained/tspsl_20-50/sl-ar-var-20pnn-gnn-max_20200308T172931"

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
```

下一步是使用 `TSP` 类和以下参数创建一个数据集对象：

+   `filename` — 要使用的数据集的路径或文件名，由 `opts.dataset_path` 指定

+   `batch_size` — 每个批次中包含的样本数量，由 `opts.batch_size` 指定

+   `num_samples` — 要在数据集中包含的总样本数量，由 `opts.num_samples` 指定

+   `neighbors` — 表示用于图稀疏化的最近邻数量的值，由 `opts.neighbors` 指定

+   `knn_strat` — 选择最近邻的策略（`'percentage'` 或 `None`），由 `opts.knn_strat` 指定

+   `supervised` — 一个布尔值，指示数据集是否用于监督学习，设置为 `True`

`make_dataset` 方法创建 TSP 数据集类的实例，并用提供的参数初始化它，返回 `dataset` 对象：

```py
dataset = TSP.make_dataset(
    filename=opts.dataset_path, batch_size=opts.batch_size,
➥ num_samples=opts.num_samples, 
➥ neighbors=opts.neighbors, knn_strat=opts.knn_strat, supervised=True
)
```

以下行创建了一个数据加载器对象，它允许以批处理方式方便地迭代数据集，这在评估期间处理数据时很有用。上一行创建的 `dataset` 对象将用作数据源。您还可以提供其他可选参数来自定义数据加载器的行为，例如 `shuffle`（用于打乱数据）和 `num_workers`（用于指定数据加载的工作进程数）：

```py
dataloader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=False,
➥ num_workers=0)
```

现在我们可以加载训练好的模型并将其分配给 `model` 变量。如果模型被 `torch.nn.DataParallel` 包装，它将通过访问 `model.module` 提取底层模块。`DataParallel` 是 PyTorch 的一个包装器，它允许在多个 GPU 上并行执行模型。如果模型确实是 `DataParallel` 的实例，它将通过访问 `module` 属性提取底层模型模块。这一步是必要的，以确保在访问模型属性和方法时保持一致的行为。然后设置模型的解码类型为 `"greedy"`。这意味着在推理或评估期间，模型应使用贪婪解码策略来生成输出预测：

```py
model, model_args = load_model(opts.model, extra_logging=True) ①
model.to(opts.device)

if isinstance(model, DataParallel):                            ②
    model = model.module                                       ②

model.set_decode_type("greedy")                                ③

model.eval()                                                   ④
```

① 加载预训练模型。

② 提取底层模块。

③ 将模型的解码类型设置为 `"greedy"`。

④ 将模型的模式设置为评估模式

列表 11.4 的完整版本，包括可视化代码，可在本书的 GitHub 仓库中找到。图 11.17 显示了预训练 ML 模型为 TSP50 实例生成的输出。

![图片](img/CH11_F17_Khamis.png)

图 11.17 使用预训练 ML 模型的 TSP50 解决方案

该图显示了与 TSP 实例和模型预测相关的以下七个图表：

+   *Concorde* — 左上角的图表显示了 Concorde 求解器生成的真实解决方案，Concorde 是解决 TSP 实例以最优性为目的的分支和切割算法的高效实现。它显示 TSP 问题的节点作为由边连接的圆圈，代表 Concorde 计算出的最优回路。图表的标题表明了从 Concorde 获得的路程（成本）长度。

+   *1 - (降低成本)* — 第二个图表包含最短子回路，并使用 Gurobi 优化库显示了这些子回路中点的降低成本。它以红色线条显示 TSP 的边，边颜色表示降低的成本值。

+   *预测热图* — 第三个图表展示了模型对 TSP 问题的预测的热图可视化。它使用颜色尺度来表示边的预测概率，较高的概率以较深的阴影显示。

+   *贪婪解* — 第四幅图展示了 ML 模型使用贪婪解码策略生成的解决方案。它显示了通过边连接的 TSP 问题的节点，代表模型获得的路线。图的标题显示了模型计算出的路线长度（成本）。

+   *欧几里得距离（最大值归一化）* — 左下角的图是 TSP 问题中节点之间欧几里得距离的热力图可视化。它使用颜色尺度来表示距离，较浅的色调表示较小的距离。

+   *降低成本* — 中间下方的图是 TSP 问题中边降低成本的热力图表示。它以颜色尺度显示降低的成本，较低值以较浅的色调显示。

+   *1 - (模型预测)* — 右下角的图展示了模型对 TSP 问题预测的热力图可视化，类似于第三幅图。然而，在这种情况下，热力图通过从 1 减去模型的预测概率来显示“1 - (模型预测)”。较深的色调代表较低的概率，表明对边选择的信心更强。

本例演示了如何使用预训练的 GNN 模型来解决 TSP。图 11.17 显示了模型解决方案与 Concorde TSP 求解器对包含 50 个兴趣点的 TSP 实例结果的并排显示。更多信息及完整代码，包括模型训练步骤，可在“Learning the Travelling Salesperson Problem Requires Rethinking Generalization” GitHub 仓库[21]中找到。

## 11.8 使用无监督机器学习解决 TSP

作为无监督 ML 方法的一个例子，列表 11.5 展示了我们如何使用自组织映射（SOMs）来解决 TSP。我们首先导入我们将使用的库。一些辅助函数是从 Vicente 的博客文章[22]中描述的 `som-tsp` 实现导入的，用于读取 TSP 实例、获取邻域、获取路线、选择最近的候选者以及计算路线距离和绘制路线。我们从提供的 URL 读取 TSP 实例，并获取城市并将它们的坐标归一化到 [0, 1] 范围内。

列表 11.5 使用无监督学习解决 TSP
