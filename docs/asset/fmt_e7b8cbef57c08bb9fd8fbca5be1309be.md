
`Dataset`是一个高度可重用的类，并且可以支持基于映射样式和可迭代样式的数据集子类的各种机器学习用例。映射样式数据集是`PyTorch`框架中的原始数据集类，最适合在内存中，可以通过索引进行寻址的数据集。例如，如果您要通过将`PyTorch`的数据集子类化为`MapStyleDataset`来实现自己的映射样式数据集，那么您必须实现所需的`__getitem__`和`__len__`方法。

**列表 6.5 设计为子类的 PyTorch 映射样式数据集**

```py
import torch.utils.data.Dataset

class MapStyleDataset(Dataset):
    def __getitem__(self, index):    ❶
      ...
    def __len__(self):               ❷
      ...
```

❶ 映射样式接口方法，用于从数据集中检索特定项。. . .

❷ . . . 并返回整个数据集中项目的总数。

请注意，作为接口的一部分，映射样式数据集做出了两个假设：预计可以通过索引值❶访问数据集中的每个示例（项），并且随时可以了解和获取数据集的大小❷。

在大多数情况下，如果您使用的是内存数据集，您可以避免实现自己的映射样式数据集子类，而是重用`TensorDataset`类。`TensorDataset`也是`torch.utils.data`包的一部分，并通过包装张量或`NumPy` n 维数组来实现所需的数据集方法。例如，要为张量`X`和`y`中的示例数据值创建映射样式的训练数据集，可以直接将数据张量传递给`TensorDataset`❶。

**列表 6.6 简化批处理 PyTorch 张量的 TensorDataset**

```py
import torch as pt
from torch.utils.data import TensorDataset
pt.manual_seed(0)

X = pt.linspace(-5, 5, 100)
y = 2 * X + pt.randn(len(X))

train_ds = TensorDataset(y, X)   ❶
```

这使您可以使用 Python `[0]`语法从数据集中获取索引为 0 的示例，使用`__getitem__`方法，

```py
print(train_ds[0])
```

输出为

```py
(tensor(-11.1258), tensor(-5.))
```

并使用`train_ds`数据集上的`__len__`方法验证数据集的长度为 100，

```py
assert
 len(train_ds) == 100
```

其中断言中的布尔表达式的结果为`True`。

在使用数据集的实例时，实现梯度下降迭代的`PyTorch`代码不应直接访问数据集，而是使用`DataLoader`的实例作为与数据交互的接口。您将在使用`PyTorch`和`GPU`的即将到来的部分中了解更多有关使用`DataLoader`的理由。

`DataLoader`本质上是对数据集的包装器，因此通过包装前面描述的`train_ds`实例，可以创建`train_dl`使用

```py
from torch.utils.data import DataLoader
train_dl = DataLoader(train_ds)
```

注意，默认情况下，当`DataLoader`与映射样式的数据集一起使用时，`DataLoader`实例返回的每个批次的大小为 1，这意味着以下表达式的结果为`True`：

```py
len(next(iter(train_dl))) == 1
```

可以通过在实例化`DataLoader`时指定`batch_size`命名参数来轻松修改此默认行为，以便该表达式

```py
train_dl = DataLoader(train_ds, batch_size = 25)
len(next(iter(train_dl)))
```

评估结果为 25。

批量大小的两个值，1 和 25，都会生成完整的批次。尽管所有完整的批次的批量大小相同，但*不完整的批次*包含的示例数少于批量大小。具体来说，根据批量大小和数据集，不完整的批次可以包含至少

*dataset_size* mod *batch_size*

示例，或者在`Python`中，`dataset_size % batch_size`。

例如，当使用`batch_size`为 33 时，在`for`循环的第四次迭代过程中，以下代码生成一个批次大小为 1 的不完整批次：

```py
train_dl = DataLoader(train_ds, batch_size = 33)
for idx, (y_batch, X_batch) in enumerate(train_dl):
  print(idx, len(X_batch))
```

这将打印以下内容：

```py
0 33
1 33
2 33
3 1
```

处理不完整的批次没有普遍接受的技术。虽然有可能尝试防止不完整批次的问题，但在实践中可能没有太多价值：因为批次是在处理足够大卷的数据时使用的，以致于数据集太大而无法放入内存中，因此如果不完整的批次对机器学习模型的整体性能没有负面和可衡量的影响，那么可以选择忽略或删除它们。例如，`DataLoader`类提供了一个`drop_last`选项，这样您就可以忽略较小的、不完整的批次：

```py
train_dl = DataLoader(train_ds, batch_size = 33, drop_last=True)
for idx, (y_batch, X_batch) in enumerate(train_dl):
  print(idx, len(X_batch))
```

这将输出以下内容：

```py
0 33
1 33
2 33
```

尽管在指定批大小时应谨慎使用不完整批次的`drop_last`选项，特别是在使用占数据集的大比例的批次大小时。例如，假设批次大小不慎设置为![06-01_EQ09](img/06-01_EQ09.png)。由于此批次大小的选择产生两个批次，当使用`drop_last=True`选项时，两个中的单个不完整批次，批次大小为![06-01_EQ10](img/06-01_EQ10.png)，被丢弃，导致将近一半的数据集浪费！

可以通过训练多个周期并将数据集与自身连接起来，使用`batch_size`窗口作为*滚动窗口*在数据集上滑动，从而防止不完整的批次。采用这种技术时，训练周期的数量应该基于批次大小和数据集大小的最小公倍数（`lcm`）：

![06-01_EQ11](img/06-01_EQ11.png)

为了说明这种方法，假设仅为例子而工作的批次大小为 12，数据集大小为 33，则

```py
import math

def lcm(a, b):
    return a * b / math.gcd(a, b)

lcm(12, 33) / 33
```

输出 4.0，这表示训练数据集需要与自身连接四次，以实现所需的四个训练周期，以避免不完整的批次。

从数据集中选择批次是如何进行的？由于批次旨在统计代表数据集，批次中的示例应尽可能相互独立。这意味着确保批次中的出租车票价示例是从整个数据集中随机采样（无替换）的。在实践中，实现对数据集进行随机洗牌最直接的方法是使用`PySpark DataFrame API`的`shuffle()`方法。

由于批次需要统计代表数据集，您可能会倾向于重新使用基于第四章中发现的测试数据集大小的批次大小。尽管测试数据集大小指标作为批次大小的 *下限*，但重新使用其值并不是正确的决定，因为测试数据集大小是尽可能小而又统计代表开发数据集的。第十一章详细描述了如何使用在本章介绍的下限和上限值来选择正确的批量大小的原则性超参数调整方法。

## 6.6 用于批量梯度下降的 Dataset 和 DataLoader 类

本节说明了如何使用最小化示例应用`Dataset`和`DataLoader`类来教授这些概念，这些概念也适用于使用数据集批次的更复杂和更现实的机器学习问题。要使用`Dataset`和`DataLoader`执行使用批量的梯度下降的线性回归，您需要修改第 6.3 节的解决方案。

**列表 6.7 使用批量数据集的基本线性回归**

```py
import torch as pt
from torch.utils.data import TensorDataset, DataLoader
pt.manual_seed(0)

X = pt.linspace(-5, 5, 100)
y = 2 * X + pt.randn(len(X))

train_ds = TensorDataset(y, X)                  ❶
train_dl = DataLoader(train_ds, batch_size=1)   ❷

w = pt.empty(1, requires_grad = True)

def forward(X):
  y_pred =  X * w
  return y_pred

def loss(y_pred, y):
  mse_loss = pt.mean((y_pred - y) ** 2)
  return mse_loss

LEARNING_RATE = 0.003
optimizer = pt.optim.SGD([w], lr = LEARNING_RATE)

EPOCHS = 25
for _ in range(EPOCHS):
  for y_batch, X_batch in train_dl:             ❸
    y_pred = forward(X_batch)                   ❹
    mse = loss(y_pred, y_batch)                 ❺
    mse.backward()

    optimizer.step()
    optimizer.zero_grad()

print(w.item())
```

❶ 使用`TensorDataset`提供`y`和`X`的数据集接口。

❷ 使用批量大小为 1（默认）为数据集创建一个`DataLoader`。

❸ 在迭代`DataLoader`时解压数据批次中的每个元组。

❹ 对特征批次执行前向步骤以生成预测。

❺ 使用批次标签和预测计算损失。

这应该输出估计的`w`大约为 2.0812765834924307。一旦原始张量`y`和`X`被打包成映射样式的`TensorDataset`数据集 ❶，则产生的`train_ds`实例进一步包装使用`DataLoader`以生成`train_dl`。

要使用梯度下降的批次，对于每个周期，代码使用在`train_dl`中由`for`循环返回的单个批次执行 100 次梯度下降迭代 ❸。由于原始数据集包含 100 个示例，并且`DataLoader`的默认批量大小等于 1。由于批量大小为 1 会产生完整的批次（回想一下批次的定义为数据集的一部分），

![06-01_EQ12](img/06-01_EQ12.png)

或者，如果您使用批次大小为 25 的批次

```py
train_dl = DataLoader(train_ds, batch_size=25),
```

那么

![06-01_EQ13](img/06-01_EQ13.png)

列表 6.7 中的更改 ❹ 和 ❺ 很简单：代码不再使用原始数据集，而是使用`train_dl`实例返回的批次。

如果您将批量大小修改为生成不完整批次的值，例如通过指定批量大小为 51，

```py
train_dl = DataLoader(train_ds, batch_size=51)
```

内部`for`循环的第二次迭代将生成具有 49 个示例的批次，因为`DataLoader`默认允许不完整的批次。在这种特定情况下，这不是一个问题，因为具有`torch.Size([])`形状的模型参数可以与形状为`torch.Size([49])`的不完整批次进行广播。然而，通常情况下，您必须注意将模型参数的形状与批次的形状对齐。在第七章中，您将从对齐模型参数与 DC 出租车数据集批次形状的示例中学习。

## 总结

+   自动微分是简化复杂链式规则应用的基本算法。

+   可以使用`python`原生数据结构来演示如何为张量实现自动微分的基础知识。

+   `PyTorch`张量提供了全面的支持，用于对张量梯度进行自动微分。

+   `torch.optim`中的优化器是一系列用于使用梯度下降优化机器学习模型中参数的算法。

+   `PyTorch`张量的自动微分和优化器 API 在使用`PyTorch`进行机器学习时是核心内容。

+   `Dataset`和`DataLoader`接口类简化了在`PyTorch`代码中使用批处理的过程。

+   `TensorDataset`提供了一个可以直接使用的内存中数据集的实现。

^(1.)自动微分和反向传播的故事在这里有详细描述：[`people.idsia.ch/~juergen/who-invented-backpropagation.html`](http://people.idsia.ch/~juergen/who-invented-backpropagation.html)。

^(2.)在有关自动微分的复杂数学论文中，标量类被称为双重数，`grad`被称为伴随数。

^(3.)例如，`SGD`的文档可以从[`mng.bz/zEpB`](http://mng.bz/zEpB)获取，并包含指向相关研究论文和`SGD`实现详细信息的链接。

^(4.)基础的`Optimizer`类和它的构造函数在这里有文档说明：[`pytorch.org/docs/stable/optim.html#torch.optim.Optimizer`](https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer)。

^(5.)更精确地说，通过使用无替换的随机抽样，或者等效地对数据集中的示例顺序进行随机洗牌。

^(6.)根据批处理大小和数据集大小的最小公倍数增加纪元数，可以产生足够的训练纪元，以避免不完整的批次。也可以训练任意`lcm(batch_size, data set_size)`的倍数，以避免不完整批次。
