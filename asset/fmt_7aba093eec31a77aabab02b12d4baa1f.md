
# 第五章：在 PyTorch 中实现神经网络

## PyTorch 简介

在本章中，您将学习 PyTorch 的基础知识，这是当今最流行的深度学习框架之一。PyTorch 由 Facebook 的人工智能研究实验室于 2016 年推出，并在随后的几年中迅速获得用户，无论是在工业界还是研究领域。PyTorch 被广泛采用的一个原因是其直观的、Pythonic 的感觉，这与深度学习从业者遵循的现有工作流程和编码范式自然契合。

特别是，本章将讨论 PyTorch 使用的数据结构，如何在 PyTorch 中定义神经模型，以及如何将数据与模型连接起来进行训练和测试。最后，我们在 PyTorch 中实现一个实际的示例——一个用于 MNIST 数字数据集的分类器，包括用于训练和测试分类器的代码。

## 安装 PyTorch

安装与 CPU 兼容的 PyTorch 版本相对简单。PyTorch 文档建议使用 `conda`，一个包管理系统。在 `conda` 中，您可以创建多个环境，其中环境是封装所有包安装的上下文。包的访问权限不会跨环境传递，这允许用户通过在各个环境中下载包来实现不同上下文的清晰分离。我们建议您为深度学习目的创建一个 `conda` 环境，以便在必要时切换到该环境。我们建议您参考 `conda` 文档，了解如何下载 `conda` 以及有关环境的进一步说明。

一旦您安装了 `conda`，创建了深度学习环境并切换到该环境，PyTorch 文档建议从命令行运行以下代码来下载 macOS 上与 CPU 兼容的 PyTorch 版本：

```py
conda install pytorch torchvision torchaudio -c pytorch
```

请注意，随此安装而来的还有 `torchvision` 和 `torchaudio`，它们分别是用于处理图像数据和音频数据的专门包。如果您使用的是 Linux 系统，文档建议从命令行中运行以下代码：

```py
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

现在，您可以转到 Python shell（仍然在您的深度学习环境中），以下命令应该可以无问题运行：

```py
import torch
```

在继续运行以下部分的代码之前，重要的是在 Python shell 中确保此命令无错误运行，因为它们都需要能够导入 PyTorch 包。

## PyTorch 张量

张量是 PyTorch 存储和操作数值信息的主要数据结构。张量可以被看作是数组和矩阵的泛化，我们在第一章中详细介绍了数组和矩阵。具体来说，张量作为 2D 矩阵和 1D 数组的泛化，可以存储多维数据，例如批量的三通道图像。请注意，这需要 4D 数据存储，因为每个图像是 3D 的（包括通道维度），还需要第四个维度来索引每个单独的图像。张量甚至可以表示超过 4D 空间的维度，尽管在实践中使用这样的张量是不常见的。

在 PyTorch 中，张量被广泛使用。它们用于表示模型的输入，模型内部的权重层以及模型的输出。张量上可以运行所有标准的线性代数操作，如转置、加法、乘法、求逆等。

### 张量初始化

我们如何初始化张量？我们可以从各种数据类型初始化张量。一些示例是 Python 列表和 Python 数值原语：

```py
arr = [1,2]
tensor = torch.tensor(arr)
val = 2.0
tensor = torch.tensor(val)
```

张量也可以从 NumPy 数组中初始化，这使得 PyTorch 可以轻松地集成到现有的数据科学和机器学习工作流中：

```py
import numpy as np
np_arr = np.array([1,2])
x_t = torch.from_numpy(np_arr)
```

此外，张量还可以通过一些常见的 PyTorch API 端点来形成：

```py
zeros_t = torch.zeros((2,3)) # Returns 2x3 tensor of zeros
ones_t = torch.ones((2,3)) # Returns 2x3 tensor of ones
rand_t = torch.randn((2,3)) # Returns 2x3 tensor of random numbers
```

### 张量属性

在我们刚刚看到的例子中，我们将一个元组作为每个函数调用的参数传递。元组中的索引数量是要创建的张量的维度，而每个索引处的数字表示该特定维度的期望大小。要访问张量的维度，我们可以调用其形状属性：

```py
zeros_t.shape # Returns torch.Size([2, 3])
```

在任何先前示例中调用形状属性应该返回与输入参数相同的元组，假设张量在此期间没有被显着修改。

张量还有哪些其他属性？除了维度，张量还存储有关存储的数据类型的信息：浮点数、复数、整数和布尔值。在每个类别中存在子类型，但我们不会在这里讨论每个子类型之间的区别。还要注意的是，张量不能包含各种数据类型的混合和匹配——单个张量中的所有数据必须是相同的数据类型。要访问张量的数据类型，我们可以调用其 `dtype` 属性：

```py
x_t = torch.tensor(2.0)
x_t.dtype # Returns torch.float32
```

此外，尽管我们尚未展示这一点，但我们可以在初始化期间设置张量的数据类型。扩展我们先前的一个示例：

```py
arr = [1,2]
x_t = torch.tensor(arr, dtype=torch.float32)
```

除了张量的数据类型和形状，我们还可以了解张量所分配的设备。这些设备包括著名的 CPU，它是任何计算机的标准设备，也是任何张量的默认存储设备，以及 GPU，即图形处理单元，它是专门用于图像空间的数据处理单元。GPU 通过在数百个小型专用核心上进行并行处理，大大加快了许多常见张量操作，例如乘法，因此使它们在大多数深度学习应用中非常有用。要访问张量设备，我们可以调用其设备属性：

```py
x_t.device # Returns device(type='cpu') by default
```

与数据类型类似，我们可以在初始化时设置张量的设备：

```py
# PyTorch will use GPU if it's available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
arr = [1,2]
x_t = torch.tensor(arr, dtype=torch.float32, device=device)
```

这是通过代码检查 GPU 是否可用并在可用时使用 GPU 的常见方法。如果 GPU 不可用，它将在没有错误的情况下使用 CPU。

如果您已经定义了具有一组特定属性的张量，并希望修改这些属性，可以使用 `to` 函数：

```py
x_t = x_t.to(device, dtype=torch.int)
```

最后，正如我们将在“PyTorch 中的梯度”中介绍的那样，PyTorch 张量可以使用参数 `requires_grad` 进行初始化，当设置为 `True` 时，将张量的梯度存储在名为 `grad` 的属性中。

### 张量操作

PyTorch API 为我们提供了许多可能的张量操作，从张量算术到张量索引。在本节中，我们将介绍一些更有用的张量操作——这些操作在深度学习应用中经常使用。

最基本的操作之一是将张量乘以某个标量 `c`。这可以通过以下代码实现：

```py
c = 10
x_t = x_t*c
```

这将导致标量与张量条目的逐元素乘积。张量加法和减法是最基本的张量操作之一。要做到这一点，我们可以通过 `+` 简单地添加张量。减法直接来自能够进行加法并将第二个张量乘以标量 -1：

```py
x1_t = torch.zeros((1,2))
x2_t = torch.ones((1,2))
x1_t + x2_t
# returns tensor([[1., 1.]])
```

结果是两个张量的逐元素求和。这可以看作是对于任何维度的矩阵加法的直接推广。请注意，这种直接推广隐含地假设了我们之前讨论过的矩阵加法的相同约束：被求和的两个张量具有相同的维度。类似地，PyTorch 将接受任何两个可广播的输入而不会出现问题，其中广播是一种通过将两个输入解析为共同形状的过程，可广播指的是这两个输入是否能够解析为共同形状。如果两个张量已经具有相同的形状，则不需要广播。有关 API 如何确定两个输入是否可广播以及在这种情况下如何执行广播的更多信息，请参阅 [PyTorch 文档](https://oreil.ly/rHEdO)。

张量乘法是另一个有用的操作，值得熟悉。当每个张量的维度小于或等于 2 时，张量乘法与矩阵和向量乘法相同。然而，张量乘法也适用于任意高维度的张量，只要这两个张量是兼容的。我们可以将高维度的张量乘法看作是批量矩阵乘法：想象我们有两个张量，第一个的形状为 `(2,1,2)`，第二个的形状为 `(2,2,2)`。我们可以进一步将第一个张量表示为长度为 2 的 1×2 矩阵列表，而第二个是长度为 2 的 2×2 矩阵列表。它们的乘积是一个长度为 2 的列表，其中产品的索引 *i* 是第一个张量的索引 *i* 和第二个张量的索引 *i* 的矩阵乘积，如图 5-1 所示。

![](img/fdl2_0501.png)

###### 图 5-1。为了帮助可视化一般张量乘法方法，这个图显示了重新堆叠之前发生的矩阵乘法。

将结果列表重新堆叠成一个 3D 张量，我们看到产品的形状为 `(2,1,2)`。现在，我们可以将这个推广到四维，其中我们不再想象有一个矩阵列表，而是将每个 4D 张量表示为矩阵的网格，产品的 `(i,j)` 索引是两个 4D 输入张量的 `(i,j)` 索引的矩阵乘积。我们用数学方式表示这一点：

<math alttext="upper P Subscript i comma j comma x comma z Baseline equals sigma-summation Underscript y Endscripts upper A Subscript i comma j comma x comma y Baseline asterisk upper B Subscript i comma j comma y comma z"><mrow><msub><mi>P</mi> <mrow><mi>i</mi><mo>,</mo><mi>j</mi><mo>,</mo><mi>x</mi><mo>,</mo><mi>z</mi></mrow></msub> <mo>=</mo> <msub><mo>∑</mo> <mi>y</mi></msub> <msub><mi>A</mi> <mrow><mi>i</mi><mo>,</mo><mi>j</mi><mo>,</mo><mi>x</mi><mo>,</mo><mi>y</mi></mrow></msub> <mo>*</mo> <msub><mi>B</mi> <mrow><mi>i</mi><mo>,</mo><mi>j</mi><mo>,</mo><mi>y</mi><mo>,</mo><mi>z</mi></mrow></msub></mrow></math>

这个过程可以推广到任何维度，假设两个输入张量遵循矩阵乘法的约束。与张量加法一样，也存在涉及广播的例外情况，但我们在这里不会详细介绍。有关广播的详细信息，请参阅 PyTorch 文档。要在 PyTorch 中对两个张量进行乘法，您可以使用 `torch.matmul` 函数：

```py
x1_t = torch.tensor([[1,2],[3,4]])
x2_t = torch.tensor([[1,2,3],[4,5,6]])
torch.matmul(x1_t, x2_t) # Returns tensor([[9,12,15],[19,26,33]])
```

除了张量上的算术运算，我们还可以对张量进行索引和切片。如果您之前有使用 NumPy 的经验，您会注意到 PyTorch 的索引非常相似，并且基于线性代数基础。如果您有一个 3D 张量，您可以通过以下代码访问位置 `(i,j,k)` 处的值：

```py
i,j,k = 0,1,1
x3_t = torch.tensor([[[3,7,9],[2,4,5]],[[8,6,2],[3,9,1]]])
print(x3_t)
# out:
# tensor([[[3, 7, 9],
#          [2, 4, 5]],
#         [[8, 6, 2],
#          [3, 9, 1]]])

x3_t[i,j,k]
# out:
# tensor(4)
```

要访问张量的更大切片，比如在 3D 张量中位置为 0 的矩阵，您可以使用以下代码：

```py
x3_t[0] # Returns the matrix at position 0 in tensor
x3_t[0,:,:] # Also returns the matrix at position 0 in tensor!
# out:
# tensor([[3, 7, 9],
#         [2, 4, 1]])
```

这两行代码被 PyTorch API 解释为等价的。这是因为使用单个索引器，比如 `x3_t[0]`，隐含地假设用户想要访问满足条件 *i* = 0 的所有索引 `(i,j,k)`（即原始 3D 张量中矩阵堆栈中的顶部矩阵）。使用 `:` 符号使这种隐含假设清晰明了，告诉 PyTorch 直接，用户不想在该维度上对数据进行子集操作。我们还可以使用 `:` 符号来对数据进行子集操作，例如：

```py
x3_t[0,1:3,:]
# returns tensor([[2, 4, 5]])
```

其中代码的最后一行被解释为：找到所有索引 `(i,j,k)`，使得 *i* = 0，*j* 大于或等于 1，并且 *j* 小于 3（`:` 遵循标准的 Python 列表索引约定，在定义范围的开始处是包含的，结束处是不包含的）。简单来说，我们想要访问原始 3D 张量中堆叠的矩阵的顶部矩阵的第二行和第三行。请注意，这种对 `:` 的使用与标准的 Python 列表索引一致。

除了访问张量的索引或切片之外，我们还可以将这些索引和切片设置为新值。在单索引的情况下，这很简单：

```py
x3_t[0,1,2] = 1

# out:
# tensor([[[3, 7, 9],
#          [2, 4, 1]],

#         [[8, 6, 2],
#          [3, 9, 1]]])
```

要设置张量的较大切片，最直接的方法是定义一个与切片具有相同维度的张量，并使用以下代码：

```py
x_t = torch.randn(2,3,4)
sub_tensor = torch.randn(2,4)
x_t[0,1:3,:] = sub_tensor
```

此外，通过广播，我们可以做如下操作：

```py
x_t[0,1:3,:] = 1
sub_tensor = torch.randn(1,4)
x_t[0,1:3,:] = sub_tensor
```

第一行将这两行的全部设置为 1，第二行将切片的两行都设置为传入的单行 `sub_tensor`。在下一节中，我们将展示如何在 PyTorch 中计算函数的梯度，以及如何访问这些梯度的值。

## PyTorch 中的梯度

作为回顾，让我们回顾一下微积分中的导数和偏导数。函数的偏导数，可以是简单的多变量多项式函数，也可以是复杂的神经网络，相对于函数的一个输入表示函数输出随着该输入值稍微变化的变化速率。因此，大幅度的导数表示输出在输入稍微变化时非常不稳定（当 *x* 是适度大小时，考虑 *f(x) = x¹⁰*），而小幅度的导数表示输出在输入稍微变化时相对稳定（考虑 *f(x) = x/10*）。如果函数接受多个输入，则梯度是由所有这些偏导数组成的向量：

<math alttext="f left-parenthesis x comma y comma z right-parenthesis equals x squared plus y squared plus z squared"><mrow><mi>f</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>,</mo> <mi>z</mi> <mo>)</mo></mrow> <mo>=</mo> <msup><mi>x</mi> <mn>2</mn></msup> <mo>+</mo> <msup><mi>y</mi> <mn>2</mn></msup> <mo>+</mo> <msup><mi>z</mi> <mn>2</mn></msup></mrow></math>

<math alttext="StartFraction normal partial-differential f Over normal partial-differential x EndFraction equals normal nabla Subscript x Baseline f left-parenthesis x comma y comma z right-parenthesis equals 2 x"><mrow><mfrac><mrow><mi>∂</mi><mi>f</mi></mrow> <mrow><mi>∂</mi><mi>x</mi></mrow></mfrac> <mo>=</mo> <msub><mi>∇</mi> <mi>x</mi></msub> <mi>f</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>,</mo> <mi>z</mi> <mo>)</mo></mrow> <mo>=</mo> <mn>2</mn> <mi>x</mi></mrow></math>
