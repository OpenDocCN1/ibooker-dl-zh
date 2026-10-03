
# 第六章：PyTorch 加速和优化

在前几章中，您学习了如何使用 PyTorch 的内置功能，并通过创建自己的自定义组件来扩展这些功能，从而使您能够快速设计新模型和算法来训练它们。

然而，当处理非常大的数据集或更复杂的模型时，将模型训练在单个 CPU 或 GPU 上可能需要非常长的时间——可能需要几天甚至几周才能获得初步结果。训练时间更长可能会变得令人沮丧，特别是当您想要使用不同的超参数配置进行许多实验时。

在本章中，我们将探讨使用 PyTorch 加速和优化模型开发的最新技术。首先，我们将看看如何使用张量处理单元（TPU）而不是 GPU 设备，并考虑在使用 TPU 时可以提高性能的情况。接下来，我将向您展示如何使用 PyTorch 的内置功能进行并行处理和分布式训练。这将为跨多个 GPU 和多台机器训练模型提供一个快速参考，以便在更多硬件资源可用时快速扩展您的训练。在探索加速训练的方法之后，我们将看看如何使用高级技术（如超参数调整、量化和剪枝）来优化您的模型。

本章还将提供参考代码，以便轻松入门，并提供我们使用的关键软件包和库的参考资料。一旦您创建了自己的模型和训练循环，您可以返回到本章获取有关如何加速和优化训练过程的提示。

让我们开始探讨如何在 TPU 上运行您的模型。

## 在 TPU 上的 PyTorch

随着深度学习和人工智能的不断部署，公司正在开发定制硬件芯片或 ASIC，旨在优化硬件中的模型性能。谷歌开发了自己的用于神经网络加速的 ASIC，称为 TPU。由于 TPU 是为神经网络设计的，它没有 GPU 的一些缺点，GPU 是为图形处理而设计的。谷歌的 TPU 现在可以作为谷歌云 TPU 的一部分供您使用。您还可以在 Google Colab 上运行 TPU。

在前几章中，我向您展示了如何使用 GPU 测试和训练您的深度模型。如果您的用例符合以下条件，您应该继续使用 CPU 和 GPU 进行训练：

+ 您有小型或中型模型以及小批量大小。
+ 您的模型训练时间不长。
+ 数据的进出是您的主要瓶颈。
+ 您的计算经常是分支的或主要是逐元素完成的，或者您使用稀疏内存访问。
+ 您需要使用高精度。双精度不适合在 TPU 上使用。

另一方面，有几个原因可能会导致您希望使用 TPU 而不是 GPU 进行训练。TPU 在执行密集向量和矩阵计算方面非常快速。它们针对特定工作负载进行了优化。如果您的用例符合以下情况，您应该强烈考虑使用 TPU：

+ 您的模型主要由矩阵计算组成。
+ 您的模型训练时间很长。
+ 您希望在 TPU 上运行整个训练循环的多次迭代。

在 TPU 上运行与在 CPU 或 GPU 上运行非常相似。让我们回顾一下如何在 GPU 上训练模型的以下代码：

```py
device = torch.device("cuda" if
  torch.cuda.is_available() else "cpu") # ①

model.to(device) # ②
for epoch in range(n_epochs):
  for data in trainloader:
    input, labels = data
    input = input.to(device) # ③
    labels = labels.to(device) # ③
    optimizer.zero_grad()

    output = model(input)

    loss = criterion(input, labels)
    loss.backward()
    optimizer.step()
```

① 如果有 GPU 可用，请将设备配置为 GPU。

② 将模型发送到设备。

③ 将输入和标签发送到 GPU。

换句话说，我们将模型、输入和标签移至 GPU，其余工作由系统完成。在 TPU 上训练网络几乎与在 GPU 上训练相同，只是您需要使用 *PyTorch/XLA*（加速线性代数）包，因为 TPU 目前不受 PyTorch 原生支持。

让我们在 Google Colab 上使用 Cloud TPU 训练我们的模型。打开一个新的 Colab 笔记本，并从运行时菜单中选择更改运行时类型。然后从“硬件加速器”下拉菜单中选择 TPU，如图 6-1 所示。Google Colab 提供免费的 Cloud TPU 系统，包括远程 CPU 主机和每个具有两个核心的四个 TPU 芯片。

![ptpr 0601](img/ptpr_0601.png)

###### 图 6-1。在 Google Colab 中使用 TPU

由于 Colab 默认未安装 PyTorch/XLA，我们需要首先安装它，使用以下命令。这将安装最新的“夜间”版本，但如果需要，您可以选择其他版本：

```py
&#33;curl 'https://raw.githubusercontent.com/pytorch' \
  '/xla/master/contrib/scripts/env-setup.py' \
  -o pytorch-xla-env-setup.py
&#33;python pytorch-xla-env-setup.py --version &#34;nightly&#34; # ①
```

<1>这些是打算在笔记本中运行的命令。在命令行上运行时，请省略“!”。

安装 PyTorch/XLA 后，我们可以导入该软件包并将数据移动到 TPU：

```py
import torch_xla.core.xla_model as xm

device = xm.xla_device()
```

请注意，我们这里不使用 `torch.cuda.is_available()`，因为它仅适用于 GPU。不幸的是，TPU 没有 `is_available()` 方法。如果您的环境未配置为 TPU，您将收到错误消息。

设备设置完成后，其余代码完全相同：

```py
model.to(device)
for epoch in range(n_epochs):
  for data in trainloader:
    input, labels = data
    input = input.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()

    output = model(input)

    loss = criterion(input, labels)
    loss.backward()
    optimizer.step()

print(output.device) # ①
# out: xla:1
```

① 如果 Colab 配置为 TPU，您应该看到 `xla:1`。

PyTorch/XLA 是一个用于 XLA 操作的通用库，可能支持除 TPU 之外的其他专用 ASIC。有关 PyTorch/XLA 的更多信息，请访问 [PyTorch/XLA GitHub 存储库](https://pytorch.tips/xla)。

在 TPU 上运行仍然存在许多限制，GPU 支持更加普遍。因此，大多数 PyTorch 开发人员将首先使用单个 GPU 对其代码进行基准测试，然后再探索使用单个 TPU 或多个 GPU 加速其代码。

我们已经在本书的前面部分介绍了如何使用单个 GPU。在下一节中，我将向您展示如何在具有多个 GPU 的机器上训练您的模型。

## 在多个 GPU 上的 PyTorch（单台机器）

在加速训练和开发时，充分利用您可用的硬件资源非常重要。如果您有一台本地计算机或网络服务器可以访问多个 GPU，本节将向您展示如何充分利用系统上的 GPU。此外，您可能希望通过在单个实例上使用云 GPU 来扩展 GPU 资源。这通常是在考虑分布式训练方法之前的第一级扩展。

在多个 GPU 上运行代码通常称为 *并行处理*。并行处理有两种方法：*数据* 并行处理和 *模型* 并行处理。在数据并行处理期间，数据批次在多个 GPU 之间分割，而每个 GPU 运行模型的副本。在模型并行处理期间，模型在多个 GPU 之间分割，数据批次被管道传送到每个部分。

数据并行处理在实践中更常用。模型并行处理通常保留用于模型不适合单个 GPU 的情况。我将在本节中向您展示如何执行这两种类型的处理。

### 数据并行处理

图 6-2 说明了数据并行处理的工作原理。在此过程中，每个数据批次被分成 *N* 部分（*N* 是主机上可用的 GPU 数量）。*N* 通常是 2 的幂。每个 GPU 持有模型的副本，并且为批次的每个部分计算梯度和损失。在每次迭代结束时，梯度和损失被合并。这种方法适用于较大的批次大小和模型适合单个 GPU 的用例。

PyTorch 可以使用 *单进程，多线程方法* 或使用 *多进程* 方法来实现数据并行处理。单进程，多线程方法只需要一行额外的代码，但在许多情况下性能不佳。

![ptpr 0602](img/ptpr_0602.png)

###### 图 6-2。数据并行处理

不幸的是，由于 Python 的全局解释器锁（GIL）在线程之间的争用、模型的每次迭代复制以及输入散布和输出收集引入的额外开销，多线程性能较差。您可能想尝试这种方法，因为它非常简单，但在大多数情况下，您可能会使用多进程方法。

#### 使用 nn.DataParallel 的多线程方法

PyTorch 的 `nn` 模块原生支持多线程的数据并行处理。您只需要在将模型发送到 GPU 之前将其包装在 `nn.DataParallel` 中，如下面的代码所示。在这里，我们假设您已经实例化了您的模型：

```py
if torch.cuda.device_count() > 1:
  print("This machine has",
        torch.cuda.device_count(),
        "GPUs available.")
  model = nn.DataParallel(model)

model.to("cuda")
```

首先，我们检查确保我们有多个 GPU，然后我们使用 `nn.DataParallel()` 在将模型发送到 GPU 之前设置数据并行处理。

这种多线程方法是在多个 GPU 上运行的最简单方式；然而，多进程方法通常在单台机器上表现更好。此外，多进程方法也可以用于跨多台机器运行，我们将在本章后面看到。

#### 使用 DDP 的多进程方法（首选）

最好使用多进程方法在多个 GPU 上训练您的模型。PyTorch 通过其 `nn.parallel.DistributedDataProcessing` 模块支持这一点。分布式数据处理（DDP）可以在单台机器上的多个进程或跨多台机器的多个进程中使用。我们将从单台机器开始。

有四个步骤需要修改您的代码：

1. 使用 *torch.distributed* 初始化一个进程组。
2. 使用 *torch.nn.to()* 创建一个本地模型。
3. 使用 *torch.nn.parallel* 将模型包装在 DDP 中。
4. 使用 *torch.multiprocessing* 生成进程。

以下代码演示了如何将您的模型转换为 DDP 训练。我们将其分解为步骤。首先，导入必要的库：

```py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel \
  import DistributedDataParallel as DDP
```

请注意，我们正在使用三个新库— *torch.distributed*、*torch.multiprocessing* 和 *torch.nn.parallel*。以下代码向您展示如何创建一个分布式训练循环：

```py
def dist_training_loop(rank,
                       world_size,
                       dataloader,
                       model,
                       loss_fn,
                       optimizer):
    dist.init_process_group("gloo",
                    rank=rank,
                    world_size=world_size) # ①

    model = model.to(rank) # ②
    ddp_model = DDP(model,
                    device_ids=[rank]) # ③
    optimizer = optimizer(
                  ddp_model.parameters(),
                  lr=0.001)

    for epochs in range(n_epochs):
      for input, labels in dataloader:
        input = input.to(rank)
        labels = labels.to(rank) # ④
        optimizer.zero_grad()
        outputs = ddp_model(input) # ⑤
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

    dist.destroy_process_group()
```

① 使用 `world_size` 进程设置一个进程组。

② 将模型移动到 ID 为 `rank` 的 GPU。

③ 将模型包装在 DDP 中。

④ 将输入和标签移动到 ID 为 `rank` 的 GPU。

⑤ 调用 DDP 模型进行前向传递。

DDP 将模型状态从 `rank0` 进程广播到所有其他进程，因此我们不必担心不同进程具有具有不同初始化权重的模型。

DDP 处理了低级别的进程间通信，使您可以将模型视为本地模型。在反向传播过程中，当 `loss.backward()` 返回时，DDP 会自动同步梯度并将同步的梯度张量放在 `params.grad` 中。

现在我们已经定义了进程，我们需要使用 `spawn()` 函数创建这些进程，如下面的代码所示：

```py
if __name__=="__main__":
  world_size = 2
  mp.spawn(dist_training_loop,
      args=(world_size,),
      nprocs=world_size,
      join=True)
```

在这里，我们将代码作为 `main` 运行，生成两个进程，每个进程都有自己的 GPU。这就是如何在单台机器上的多个 GPU 上运行数据并行处理。

###### 警告

GPU 设备不能在进程之间共享。

如果您的模型不适合单个 GPU 或者使用较小的批量大小，您可以考虑使用模型并行处理而不是数据并行处理。接下来我们将看看这个。

### 模型并行处理

图 6-3 展示了模型并行处理的工作原理。在这个过程中，模型被分割到同一台机器上的 *N* 个 GPU 中。如果我们按顺序处理数据批次，下一个 GPU 将始终等待前一个 GPU 完成，这违背了并行处理的目的。因此，我们需要对数据处理进行流水线处理，以便每个 GPU 在任何给定时刻都在运行。当我们对数据进行流水线处理时，只有前 *N* 个批次按顺序运行，然后每个后续运行会激活所有 GPU。

![ptpr 0603](img/ptpr_0603.png)

###### 图 6-3。模型并行处理

实现模型并行处理并不像数据并行处理那样简单，它需要您重新编写模型。您需要定义模型如何跨多个 GPU 分割以及数据在前向传递中如何进行流水线处理。通常通过为模型编写一个子类，具有特定数量的 GPU 的多 GPU 实现来完成这一点。

以下代码演示了 AlexNet 的双 GPU 实现：

```py
class TwoGPUAlexNet(AlexNet):
  def __init__(self):
    super(ModelParallelAlexNet, self).__init__(
              num_classes=num_classes,
              *args,
              **kwargs)
    self.features.to('cuda:0')
    self.avgpool.to('cuda:0')
    self.classifier.to('cuda:1')
    self.split_size = split_size

  def forward(self, x):
      splits = iter(x.split(self.split_size,
                    dim=0))
      s_next = next(splits)
      s_prev = self.seq1(s_next).to('cuda:1')
      ret = []

      for s_next in splits:
        s_prev = self.seq2(s_prev) # ①
        ret.append(self.fc(
            s_prev.view(s_prev.size(0), -1)))

        s_prev = self.seq1(s_next).to('cuda:1') # ②

      s_prev = self.seq2(s_prev)
      ret.append(self.fc(
            s_prev.view(s_prev.size(0), -1)))

      return torch.cat(ret)
```

① `s_prev` 在 `cuda:1` 上运行。

② `s_next` 在 `cuda:0` 上运行，可以与 `s_prev` 并行运行。
