
**结论** — GeoGrid 采用`ShaDowKHopSampler`的决定源于对每种技术优缺点的深入分析。通过将采样方法与实际数据相结合，他们旨在在图表示的粒度和相关性之间取得平衡。

现在我们已经掌握了批处理的概念，我们可以检查两种与采样协同工作的技术：并行处理和使用远程后端。

## 7.8 并行和分布式处理

批处理非常适合于接下来的两种方法，并行处理和使用远程后端，因为这些方法在数据分割时效果最佳。并行处理是一种通过在多个计算节点或多台机器上分散计算任务来训练机器学习模型的方法。在本节中，我们专注于在单台机器上的多个 GPU 之间分散模型训练 [13–17]。我们将使用 PyTorch 的`DistributedDataParallel`来实现这一点。

### DataParallel 和 DistributedDataParallel

在 PyTorch 领域，你将遇到两种并行化神经网络模型的主要选项：`DataParallel`和`DistributedDataParallel`。每种方法都有其优点和局限性，这对于做出明智的决定至关重要。

`DataParallel`是为单台机器上的多 GPU 设置量身定制的，但也有一些注意事项，例如在每个前向传递期间模型的复制会带来额外的计算成本。随着你的模型和数据规模的扩大，这些限制变得更加明显。

另一方面，`DistributedDataParallel`可以跨多台机器和 GPU 进行扩展。它通过为 GPU 间通信分配专用的 Compute Unified Device Architecture (CUDA) 缓冲区，并且通常产生更少的开销，从而优于`DataParallel`。这使得它非常适合大规模数据和复杂模型。

`DataParallel`和`DistributedDataParallel`都为在 PyTorch 中并行化你的模型提供了途径。了解它们各自的优势和劣势，使你能够选择最适合你特定机器学习挑战的技术。鉴于其在可扩展性和效率方面的优势，尤其是在复杂或大规模项目中，我们选择了`DistributedDataParallel`作为我们模型并行化的首选选项。

### 7.8.1 使用分布式数据并行

用简单的话说，分布式数据并行（DDP）是一种同时使用多个图形卡（GPU）训练机器学习模型的方法。想法是将数据和模型分布在不同的 GPU 上，执行计算，然后将结果汇总在一起。为了使这成为可能，你首先需要设置一个*进程组*，这仅仅是一种组织你使用的 GPU 的方式。与一些其他方法不同，DDP 不会自动分割你的数据；你必须自己完成这部分工作。

当你准备训练时，DDP 通过在所有 GPU 之间同步对模型所做的更新来提供帮助。这是通过共享梯度来完成的。因为所有 GPU 都获得了这些更新，它们都在帮助改进同一个模型，尽管它们正在处理不同的数据。

与在单个 GPU 上运行或使用更简单的并行方法相比，这种方法特别快速且高效。然而，有一些技术细节需要记住，例如如果你使用多台机器，确保你正确地加载和保存你的模型。训练的一般步骤如下：

+   *模型实例化* — 初始化用于训练的 GNN 模型。
+   *分布式模型设置* — 使用 PyTorch 的`DistributedDataParallel`包装模型，以准备分布式训练。
+   *训练循环* — 实现一个包含前向传播、计算损失、反向传播和更新模型参数的训练循环。
+   *进程同步* — 使用 PyTorch 的分布式通信包来同步所有进程，确保在进入下一步之前所有进程都已完成训练。这可以通过在移动到下一个 epoch 之前使用`dist.barrier()`来实现。一旦所有 epoch 都完成，它将销毁进程组。
+   *入口点守卫* — 使用`if __name__ == '__main__':`来指定数据集并启动分布式训练。这确保了只有在脚本直接运行时才会执行训练代码，而不是当它作为模块导入时。

使用分布式处理需要仔细处理同步点，以确保模型被正确训练。你还必须确保你的机器或集群有足够的资源来处理并行计算。

`Torch.distributed`支持分布式计算的多种后端。以下是最推荐的两种：

+   NVIDIA 集体通信库（`NCCL`）—Nvidia 的 NCCL 用于基于 GPU 的分布式训练。它为集体通信提供了优化的原语。
+   `Gloo` — Gloo 是由 Facebook 开发的集体通信库，提供广播、全归约等各种操作。这个库用于 CPU 训练。

### 7.8.2 DDP 代码示例

下面是使用 PyTorch 进行分布式训练的示例。为了简单起见，我们使用修改后的国家标准与技术研究院（MNIST）数据集训练一个简单的神经网络。在 GitHub 仓库中可以找到一个使用 GCN 在 Amazon 产品数据集上的示例。在那个例子中，我们不是使用 Google Colab 来运行代码，而是使用 Kaggle 笔记本，它具有双 GPU 系统。GCN 示例中的另一个区别是我们使用了`NeighborLoader`数据加载器，它使用`NeighborSampler`样本器。

让我们分解一下这段代码中发生的事情。GCN 版本基本上遵循同样的逻辑。

#### 分布式训练设置

脚本导入必要的模块，如`torch`、`torch.distributed`等。它使用`dist.init_process_group`初始化 DDP 环境。它使用 NCCL 设置通信，并指定本地主机地址和端口（`tcp://localhost:23456`）用于同步。

#### 准备模型和数据

代码定义了一个简单的`Flatten`层，这是神经网络的一部分，用于重塑其输入。数据转换和加载步骤使用 PyTorch 的 DataLoader 和 torchvision 数据集进行设置。加载的数据是 MNIST。

#### 训练函数

`train`是负责训练模型的函数。它遍历数据批次，执行正向和反向传递，并更新模型参数。

#### 主函数

在`main()`函数内部，每个进程（在本例中代表单个 GPU）设置其随机种子和设备（基于进程的排名的 CUDA 设备）。神经网络模型被定义为具有`Flatten`层和`Linear`层的顺序模型。然后使用`DistributedDataParallel`进行包装。定义了损失函数（`CrossEntropyLoss`）和优化器（`SGD`）。

#### 多进程启动

最后，脚本使用`mp.spawn`函数启动分布式训练。它在`world_size`数量的进程（基本上是两个 GPU）上运行`main()`。每个进程将在其数据子集上训练模型。

#### 运行训练

每个进程使用其数据子集训练模型，但梯度会在所有进程（GPU）之间同步，以确保处理器正在更新相同的全局模型。这个过程总结在图 7.3 中。

![图](img/7-3.png)

##### 图 7.3 使用多个处理器设备启动和运行训练的过程图

以下列表使用`DistributedDataParallel`模块训练神经网络。

##### 列表 7.2 使用 DDP 进行训练

```py
import torch 
import torch.distributed as dist 
import torch.multiprocessing as mp 
import torch.nn as nn 
from torch.nn.parallel import DistributedDataParallel    #1
from torch.utils.data import DataLoader    #2
from torchvision import datasets, transforms 

class Flatten(nn.Module): 
  def forward(self, input): 
    return input.view(input.size(0), -1) 

def train(model, trainloader, 
                 criterion, 
                 optimizer,
                 device):   #3
    model.train() 
    for batch_idx, (data, target) in enumerate(trainloader): 
      print(f'Process {device}, Batch {batch_idx}') 
       data, target = data.to(device), target.to(device) 
       optimizer.zero_grad() 
       output = model(data) 
       loss = criterion(output, target) 
       loss.backward() 
       optimizer.step() 

def main(rank, world_size):    #4
    filepath = '~/.pytorch/MNIST_data/'
    dist.init_process_group(   #5
    backend='nccl', 
    init_method='tcp://localhost:23456', 
    rank=rank,
    world_size=world_size    #6
    )

    torch.manual_seed(0)   #7
    device = torch.device(f'cuda:{rank}')    #8

    transform = transforms.Compose(
                        [transforms.ToTensor(),
                        transforms.Normalize((0.5,),
                        (0.5,))]
                        )

    trainset = datasets.MNIST(filepath ,                #9
                                download=True,          #9
                                train=True,             #9
                                transform=transform)    #9
    train_loader = DataLoader(trainset,           #10
                                batch_size=64,    #10
                                shuffle=True,     #10
                                num_workers=2)    #10

    model = nn.Sequential(Flatten(), nn.Linear(784, 10)).to(device) 
    model = DistributedDataParallel(model, device_ids=[rank])   #11

    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 

    train(model, train_loader, criterion, optimizer, device)    #12
```

#1 导入分布式训练的 DistributedDataParallel 类

#2 导入数据加载器实用工具

#3 定义主要训练函数

#4 定义分布式训练设置的主要函数

#5 初始化分布式进程组

#6 指定参与进程的总数

#7 设置随机种子以确保可重复性

#8 根据进程排名设置设备

#9 加载和转换 MNIST 数据集

#10 为训练数据创建数据加载器

#11 包装模型以进行分布式训练

#12 调用训练函数以启动训练过程

我们以 GeoGrid 的朋友们的另一个例子结束本节。

#### 示例

GeoGrid 有机会提交一个政府项目的概念验证，该项目旨在使用 GNN 进行复杂的环境建模。赢得这个合同可能使他们成为该领域的领导者，但他们面临着激烈的竞争。政府设定了一个紧张的截止日期来审查概念验证演示，这使得 GeoGrid 的情况变得紧张，因为它仍处于开发初期。

在一次团队会议上，焦点转向了一个关键的技术决策和一个重要的困境：在多个 GPU 上使用 DDP 训练的潜在用途。首席数据科学家看到了 DDP 加快训练时间的吸引力，这为政府项目提供了一个可能令人印象深刻的效率展示和准备情况。

另一方面，团队中的一位经验丰富的工程师心存担忧。尽管 DDP 有其优势，但它可能会引入问题，例如在 GPU 之间同步梯度时产生的计算开销。另一层复杂性来自其他团队成员，他们指出，他们的专用 GNN 算法尚未与 DDP 进行过测试。他们表达了对数据如何在 GPU 之间分配以及可能出现的不平衡和低效的担忧。其他担忧集中在开发和测试代码所需的时间上。

团队仔细权衡了这些因素。快速且按时完成演示将是理想的选择。然而，将 DDP 应用于他们特定的 GNN 模型可能带来的复杂性和未知因素可能会带来意外的延误和成本，可能使他们错过提交截止日期。

对模型开发的迭代性质进行了进一步考虑。在概念验证阶段，为了性能优化进行快速迭代至关重要。将 DDP 加入其中可能会使调试复杂化并延长开发周期：

+   *决策* — 最终，团队选择了一种谨慎的方法。他们决定进行为期一周的可行性研究，严格评估使用 DDP 对他们的 GNN 架构的影响。这将使他们能够根据经验数据做出明智的决定，这些数据跟踪了收敛时间和每个 epoch 的平均时间。将咨询 IT 部门以确保必要的计算资源仅用于这项关键研究。
+   *结论* — 推出 GNN 的决定通常高度依赖于数据、时间表和计算需求。可行性研究是决策过程中的重要部分，尤其是在确定计算需求时。

在下一节中，我们将探讨另一种基于采样的技术，该技术通过直接从远程存储系统中抽取数据来进行训练。

## 7.9 使用远程存储进行训练

本书在数据处理管道方面的一个突出方法是从数据存储系统中获取数据，然后通过将其转换为 GNN 平台使用的形式来预处理这些数据。在训练过程中，这些预处理好的数据被存储在内存中。

相比之下，当数据变得太大而无法放入内存时，一种方法是将预处理集成到训练过程中。我们不必预处理整个数据集，将其放入内存，然后进行训练，我们基本上可以直接从初始数据存储系统进行采样和批处理。通过我们 GNN 平台和数据源之间的接口，我们可以处理直接从数据源抽取的每个批次[18]。在 PyG 中，这被称为*远程后端*，并且设计为对所使用的特定后端无感知[19–22]。

好处在于，我们现在数据集的大小仅受我们数据库容量的限制。权衡如下：

+   我们需要做一些工作来设置远程后端，具体细节在本节中详细说明。
+   从远程后端抽取数据将引入 I/O 延迟。
+   集成远程后端会增加训练设置的复杂性。基本上，可能出现更多问题，并且需要调试的项目也会更多。

在 PyG 中，远程后端是通过存储和从图的两个方面进行采样实现的：使用`GraphStore`对结构信息（即，边）进行采样，以及使用`FeatureStore`对节点特征进行采样（在撰写本文时，边特征尚未支持）。对于存储图结构，PyG 团队建议使用图数据库作为后端，例如 Neo4J、TigerGraph、Kùzu 和 ArangoDB。同样，对于节点特征，PyG 团队建议使用键值数据库，例如 Memcached、LevelDB 和 RocksDB。实现远程后端的关键要素如下：

+   *远程数据源* — 存储您的图结构和节点特征的数据库。这个选择可能仅仅是您目前用来存储图的数据库系统。
+   *图存储对象* — `torch_geometric.data.GraphStore`对象存储图的边索引，从而实现节点采样。您自定义类的主要组件必须是与您数据库的连接，以及 CRUD（创建、读取、更新、删除）函数，包括`put_edge_index()`、`get_edge_index()`和`remove_edge_index()`。
+   *特征存储对象* — `torch_geometric.data.FeatureStore`管理图节点的特征。在图学习应用中，节点特征的大小被认为是一个主要的存储问题。与`GraphStore`类似，自定义实现包括连接到远程数据库和 CRUD 函数。
+   *采样器* — 一个与`GraphStore`链接的图采样器，通过`torch_geometric.sampler.BaseSampler`接口使用采样算法从输入节点生成子图。PyG 的默认采样器拉取边索引，将它们转换为压缩稀疏列 (CSC) 格式，并使用内存中的采样例程。自定义采样器可以通过实现`BaseSampler`类的`sample_from_nodes()`和`sample_from_edges()`方法来使用专门的`GraphStore`方法。这涉及到节点级和链接级采样。
+   *数据加载器* — 数据加载器的工作方式与前面章节中介绍的方式类似。这里的区别在于数据加载器使用`GraphStore`、`FeatureStore`和`sampler`对象，而不是通常的 PyG 数据对象。PyG 文档中的一个示例将在下一列表中展示。

##### 列表 7.3 使用远程后端的加载器对象

```py
loader = NodeLoader(
    data=(feature_store, graph_store),
    node_sampler=node_sampler,
    batch_size=20,
    input_nodes='paper',
)

for batch in loader:
    <training loop>
```
