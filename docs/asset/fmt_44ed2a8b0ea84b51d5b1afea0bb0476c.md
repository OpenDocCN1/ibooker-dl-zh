
在本节的其余部分中，我们将首先使用`PyTorch`来熟悉各种`PyTorch`操作。我们建议您现在已经安装了`PyTorch`并准备好了`Python 3.5+`笔记本，并按照本节中的示例进行操作。我们还建议您完成本节后面的练习。

### 安装 PyTorch

第一步是通过在`pytorch.org`上选择您的系统首选项在您的机器上安装`PyTorch`。选择您的操作系统，然后选择包管理器（我们推荐`conda/pip`），然后选择您正在使用的`Python`版本（我们推荐`3.5+`）。这将生成命令，以便您执行安装`PyTorch`。在撰写本文时，`conda`环境的安装命令如下：

```py
conda install pytorch torchvision -c pytorch
```

> **注意**：如果您有一个支持 CUDA 的图形处理器单元（GPU），您还应该选择合适的 CUDA 版本。要了解更多细节，请参考`pytorch.org`上的安装说明。

### 创建张量

首先，我们定义一个辅助函数`describe(x)`，它总结了张量`x`的各种性质，例如张量的类型、张量的维数和张量的内容：

```py
Input[0]:
def describe(x):
  print("Type: {}".format(x.type()))
  print("Shape/size: {}".format(x.shape))
  print("Values: \n{}".format(x))
```

`PyTorch`允许我们使用`torch`包以许多不同的方式创建张量。创建张量的一种方法是通过指定一个随机张量的维数来初始化它，如示例 1-3 所示。

**示例 1-3：在 PyTorch 中使用`torch.Tensor`创建张量**

```py
Input[0]:
import torch
describe(torch.Tensor(2, 3))
Output[0]:
Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 3.2018e-05,  4.5747e-41,  2.5058e+25],
        [ 3.0813e-41,  4.4842e-44,  0.0000e+00]])
```

我们还可以创建一个张量通过随机初始化值区间上的均匀分布（`0,1`）或标准正态分布（从均匀分布随机初始化张量，说，是很重要的，正如您将看到的在第三章和第四章），见示例 1-4。

**示例 1-4：创建随机初始化的张量**

```py
Input[0]:
import torch
describe(torch.rand(2, 3))   # uniform random
describe(torch.randn(2, 3))  # random normal
Output[0]:
Type:  torch.FloatTensor
Shape/size:  torch.Size([2, 3])
Values:
 tensor([[ 0.0242,  0.6630,  0.9787],
        [ 0.1037,  0.3920,  0.6084]])

Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[-0.1330, -2.9222, -1.3649],
        [ 2.3648,  1.1561,  1.5042]])
```

我们还可以创建张量，所有张量都用相同的标量填充。对于创建 0 或 1 张量，我们有内置函数，对于填充特定值，我们可以使用`fill_()`方法。任何带有下划线（`_`）的`PyTorch`方法都是指就地（in place）操作；也就是说，它在不创建新对象的情况下就地修改内容，如示例 1-5 所示。

**示例 1-5：创建填充的张量**

```py
Input[0]:
import torch
describe(torch.zeros(2, 3))
x = torch.ones(2, 3)
describe(x)
x.fill_(5)
describe(x)
Output[0]:
Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 0.,  0.,  0.],
        [ 0.,  0.,  0.]])

Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]])

Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 5.,  5.,  5.],
        [ 5.,  5.,  5.]])
```

示例 1-6 演示了如何通过使用`Python`列表以声明的方式创建张量。

**示例 1-6：从列表创建和初始化张量**

```py
Input[0]:
x = torch.Tensor([[1, 2, 3],  
                  [4, 5, 6]])
describe(x)
Output[0]:
Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 1.,  2., 3.],
        [ 4.,  5., 6.]])
```

值可以来自列表（如前面的示例），也可以来自`NumPy`数组。当然，我们也可以从`PyTorch`张量变换到`NumPy`数组。注意，这个张量的类型是一个`double`张量，而不是默认的`FloatTensor`。这对应于`NumPy`随机矩阵的数据类型`float64`，如示例 1-7 所示。

**示例 1-7：从 NumPy 创建和初始化张量**

```py
Input[0]:
import torch
import numpy as np
npy = np.random.rand(2, 3)
describe(torch.from_numpy(npy))
Output[0]:
Type: torch.DoubleTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 0.8360,  0.8836,  0.0545],
        [ 0.6928,  0.2333,  0.7984]], dtype=torch.float64)
```

在处理使用`Numpy`格式数值的遗留库（legacy libraries）时，在`NumPy`和`PyTorch`张量之间切换的能力变得非常重要。

### 张量类型和大小

每个张量都有一个相关的类型和大小。使用`torch`时的默认张量类型。张量构造函数是`torch.FloatTensor`。但是，可以在初始化时指定张量，也可以在以后使用类型转换方法将张量转换为另一种类型（`float`、`long`、`double`等）。有两种方法可以指定初始化类型，一种是直接调用特定张量类型（如`FloatTensor`和`LongTensor`）的构造函数，另一种是使用特殊的方法`torch.tensor`，并提供`dtype`，如例 1-8 所示。

**示例 1-8：张量属性**

```py
Input[0]:
x = torch.FloatTensor([[1, 2, 3],  
                       [4, 5, 6]])
describe(x)
Output[0]:
Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 1.,  2.,  3.],
        [ 4.,  5.,  6.]])
Input[1]:
x = x.long()
describe(x)
Output[1]:
Type: torch.LongTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 1,  2,  3],
        [ 4,  5,  6]])
Input[2]:
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]], dtype=torch.int64)
describe(x)
Output[2]:
Type: torch.LongTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 1,  2,  3],
        [ 4,  5,  6]])
Input[3]:
x = x.float()
describe(x)
Output[3]:
Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 1.,  2.,  3.],
        [ 4.,  5.,  6.]])
```

我们利用张量对象的形状特性和尺寸方法来获取其尺寸的测量值。访问这些度量的两种方法基本上是相同的。在调试`PyTorch`代码时，检查张量的形状成为必不可少的工具。

### 张量操作

在创建了张量之后，可以像处理传统编程语言类型（如`+`、`-`、`*`和`/`）那样对它们进行操作。除了操作符，我们还可以使用`.add()`之类的函数，如示例 1-9 所示，这些函数对应于符号操作符。

**示例 1-9：张量操作：加法**

```py
Input[0]:
import torch
x = torch.randn(2, 3)
describe(x)
Output[0]:
Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 0.0461,  0.4024, -1.0115],
        [ 0.2167, -0.6123,  0.5036]])
Input[1]:
describe(torch.add(x, x))
Output[1]:
Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 0.0923,  0.8048, -2.0231],
        [ 0.4335, -1.2245,  1.0072]])
Input[2]:
describe(x + x)
Output[2]:
Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 0.0923,  0.8048, -2.0231],
        [ 0.4335, -1.2245,  1.0072]])
```

还有一些运算可以应用到张量的特定维数上。正如您可能已经注意到的，对于 2D 张量，我们将行表示为维度 0，列表示为维度 1，如示例 1-10 所示。

**示例 1-10：基于维度的张量操作**

```py
Input[0]:
import torch
x = torch.arange(6)
describe(x)
Output[0]:
Type: torch.FloatTensor
Shape/size: torch.Size([6])
Values:
tensor([ 0.,  1.,  2.,  3.,  4.,  5.])
Input[1]:
x = x.view(2, 3)
describe(x)
Output[1]:
Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 0.,  1.,  2.],
        [ 3.,  4.,  5.]])
Input[2]:
describe(torch.sum(x, dim=0))
Output[2]:
Type: torch.FloatTensor
Shape/size: torch.Size([3])
Values:
tensor([ 3.,  5.,  7.])
Input[3]:
describe(torch.sum(x, dim=1))
Output[3]:
Type: torch.FloatTensor
Shape/size: torch.Size([2])
Values:
tensor([  3.,  12.])
Input[4]:
describe(torch.transpose(x, 0, 1))
Output[4]:
Type: torch.FloatTensor
Shape/size: torch.Size([3, 2])
Values:
tensor([[ 0.,  3.],
        [ 1.,  4.],
        [ 2.,  5.]])
```

通常，我们需要执行更复杂的操作，包括索引、切片、连接和突变（indexing, slicing, joining and mutation）的组合。与`NumPy`和其他数字库一样，`PyTorch`也有内置函数，可以使此类张量操作非常简单。

### 索引，切片和连接

如果您是一个`NumPy`用户，那么您可能非常熟悉示例 1-11 中所示的`PyTorch`的索引和切片方案。

**示例 1-11：切片和索引张量**

```py
Input[0]:
import torch
x = torch.arange(6).view(2, 3)
describe(x)
Output[0]:
Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 0.,  1.,  2.],
        [ 3.,  4.,  5.]])
Input[1]:
describe(x[:1, :2])
Output[1]:
Type: torch.FloatTensor
Shape/size: torch.Size([1, 2])
Values:
tensor([[ 0.,  1.]])
Input[2]:
describe(x[0, 1])
Output[2]:
Type: torch.FloatTensor
Shape/size: torch.Size([])
Values:
1.0
```

示例 1-12 演示了`PyTorch`还具有用于复杂索引和切片操作的函数，您可能对有效地访问张量的非连续位置感兴趣。

**示例 1-12：复杂索引：张量的非连续索引**

```py
Input[0]:
indices = torch.LongTensor([0, 2])
describe(torch.index_select(x, dim=1, index=indices))
Output[0]:
Type: torch.FloatTensor
Shape/size: torch.Size([2, 2])
Values:
tensor([[ 0.,  2.],
        [ 3.,  5.]])
Input[1]:
indices = torch.LongTensor([0, 0])
describe(torch.index_select(x, dim=0, index=indices))
Output[1]:
Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 0.,  1.,  2.],
        [ 0.,  1.,  2.]])
Input[2]:
row_indices = torch.arange(2).long()
col_indices = torch.LongTensor([0, 1])
describe(x[row_indices, col_indices])
Output[2]:
Type: torch.FloatTensor
Shape/size: torch.Size([2])
Values:
tensor([ 0.,  4.])
```

注意索引（`indices`）是一个长张量；这是使用`PyTorch`函数进行索引的要求。我们还可以使用内置的连接函数连接张量，如示例 1-13 所示，通过指定张量和维度。

**示例 1-13：连接张量**
