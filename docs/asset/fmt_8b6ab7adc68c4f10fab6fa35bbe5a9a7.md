
```py
Input[0]:
import torch
x = torch.arange(6).view(2,3)
describe(x)
Output[0]:
Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 0.,  1.,  2.],
        [ 3.,  4.,  5.]])
Input[1]:
describe(torch.cat([x, x], dim=0))
Output[1]:
Type: torch.FloatTensor
Shape/size: torch.Size([4, 3])
Values:
tensor([[ 0.,  1.,  2.],
        [ 3.,  4.,  5.],
        [ 0.,  1.,  2.],
        [ 3.,  4.,  5.]])
Input[2]:
describe(torch.cat([x, x], dim=1))
Output[2]:
Type: torch.FloatTensor
Shape/size: torch.Size([2, 6])
Values:
tensor([[ 0.,  1.,  2.,  0.,  1.,  2.],
        [ 3.,  4.,  5.,  3.,  4.,  5.]])
Input[3]:
describe(torch.stack([x, x]))
Output[3]:
Type: torch.FloatTensor
Shape/size: torch.Size([2, 2, 3])
Values:
tensor([[[ 0.,  1.,  2.],
         [ 3.,  4.,  5.]],

        [[ 0.,  1.,  2.],
         [ 3.,  4.,  5.]]])
```

PyTorch 还在张量上实现了高效的线性代数操作，如乘法、逆和迹，如示例 1-14 所示。

**示例 1-14：张量上的线性代数：乘法**

```py
Input[0]:
import torch
x1 = torch.arange(6).view(2, 3)
describe(x1)
Output[0]:
Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 0.,  1.,  2.],
        [ 3.,  4.,  5.]])
Input[1]:
x2 = torch.ones(3, 2)
x2[:, 1] += 1
describe(x2)
Output[1]:
Type: torch.FloatTensor
Shape/size: torch.Size([3, 2])
Values:
tensor([[ 1.,  2.],
        [ 1.,  2.],
        [ 1.,  2.]])
Input[2]:
describe(torch.mm(x1, x2))
Output[2]:
Type: torch.FloatTensor
Shape/size: torch.Size([2, 2])
Values:
tensor([[  3.,   6.],
        [ 12.,  24.]])
```

到目前为止，我们已经研究了创建和操作恒定 PyTorch 张量对象的方法。就像编程语言（如 Python）变量封装一块数据，关于数据的额外信息（如内存地址存储，例如），PyTorch 张量处理构建计算图时所需的记账（bookkeeping）所需构建计算图对机器学习只是在实例化时通过启用一个布尔标志。

## 张量和计算图

PyTorch 张量类封装了数据（张量本身）和一系列操作，如代数操作、索引操作和整形操作。然而，如示例 1-15 所示，当`requires_grad`布尔标志被设置为`True`的张量，记账操作启用，可以追踪的梯度张量以及梯度函数，这两个需要基于促进梯度学习讨论“监督学习范式”。

**示例 1-15：为梯度记录创建张量**

```py
Input[0]:
import torch
x = torch.ones(2, 2, requires_grad=True)
describe(x)
print(x.grad is None)
Output[0]:
Type: torch.FloatTensor
Shape/size: torch.Size([2, 2])
Values:
tensor([[ 1.,  1.],
        [ 1.,  1.]])
True
Input[1]:
y = (x + 2) * (x + 5) + 3
describe(y)
print(x.grad is None)
Output[1]:
Type: torch.FloatTensor
Shape/size: torch.Size([2, 2])
Values:
tensor([[ 21.,  21.],
        [ 21.,  21.]])
True
Input[2]:
z = y.mean()
describe(z)
z.backward()
print(x.grad is None)
Output[2]:
Type: torch.FloatTensor
Shape/size: torch.Size([])
Values:
21.0
False
```

当您使用`requires_grad=True`创建张量时，您需要 PyTorch 来管理计算梯度的 bookkeeping 信息。首先，PyTorch 将跟踪向前传递的值。然后，在计算结束时，使用单个标量来计算向后传递。反向传递是通过对一个张量使用`backward()`方法来初始化的，这个张量是由一个损失函数的求值得到的。向后传递为参与向前传递的张量对象计算梯度值。

一般来说，梯度是一个值，它表示函数输出相对于函数输入的斜率。在计算图形设置中，模型中的每个参数都存在梯度，可以认为是该参数对误差信号的贡献。在 PyTorch 中，可以使用`.grad`成员变量访问计算图中节点的梯度。优化器使用`.grad`变量更新参数的值。

到目前为止，我们一直在 CPU 内存上分配张量。在做线性代数运算时，如果你有一个 GPU，那么利用它可能是有意义的。要利用 GPU，首先需要分配 GPU 内存上的张量。对 gpu 的访问是通过一个名为 CUDA 的专门 API 进行的。CUDA API 是由 NVIDIA 创建的，并且仅限于在 NVIDIA gpu 上使用。PyTorch 提供的 CUDA 张量对象在使用中与常规 cpu 绑定张量没有区别，除了内部分配的方式不同。

## CUDA 张量

PyTorch 使创建这些 CUDA 张量变得非常容易（示例 1-16），它将张量从 CPU 传输到 GPU，同时维护其底层类型。PyTorch 中的首选方法是与设备无关，并编写在 GPU 或 CPU 上都能工作的代码。在下面的代码片段中，我们首先使用`torch.cuda.is_available()`检查 GPU 是否可用，然后使用`torch.device`检索设备名。然后，将实例化所有未来的张量，并使用`.to(device)`方法将其移动到目标设备。

**示例 1-16：创建 CUDA 张量**

```py
Input[0]:
import torch
print (torch.cuda.is_available())
Output[0]:
True
Input[1]:
# preferred method: device agnostic tensor instantiation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)
Output[1]:
cuda
Input[2]:
x = torch.rand(3, 3).to(device)
describe(x)
Output[2]:
Type: torch.cuda.FloatTensor
Shape/size: torch.Size([3, 3])
Values:
tensor([[ 0.9149,  0.3993,  0.1100],
        [ 0.2541,  0.4333,  0.4451],
        [ 0.4966,  0.7865,  0.6604]], device='cuda:0')
```

要对 CUDA 和非 CUDA 对象进行操作，我们需要确保它们在同一设备上。如果我们不这样做，计算就会中断，如下面的代码片段所示。例如，在计算不属于计算图的监视指标时，就会出现这种情况。当操作两个张量对象时，确保它们在同一个设备上。如示例 1-17 所示。

**示例 1-17：混合 CUDA 张量和 CPU 绑定的张量**

```py
Input[0]
y = torch.rand(3, 3)
x + y
Output[0]
----------------------------------------------------------------------
RuntimeError                         Traceback (most recent call last)
      1 y = torch.rand(3, 3)
----> 2 x + y

RuntimeError: Expected object of type torch.cuda.FloatTensor but found type torch.FloatTensor for argument #3 'other'
Input[1]
cpu_device = torch.device("cpu")
y = y.to(cpu_device)
x = x.to(cpu_device)
x + y
Output[1]
tensor([[ 0.7159,  1.0685,  1.3509],
        [ 0.3912,  0.2838,  1.3202],
        [ 0.2967,  0.0420,  0.6559]])
```

请记住，将数据从 GPU 来回移动是非常昂贵的。因此，典型的过程包括在 GPU 上执行许多并行计算，然后将最终结果传输回 CPU。这将允许您充分利用 gpu。如果您有几个 CUDA 可见的设备（即，最佳实践是在执行程序时使用`CUDA_VISIBLE_DEVICES`环境变量，如下图所示：

```py
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py
```

在本书中我们不涉及并行性和多 gpu 训练，但是它们在缩放实验中是必不可少的，有时甚至在训练大型模型时也是如此。我们建议您参考 PyTorch 文档和讨论论坛，以获得关于这个主题的更多帮助和支持。

# 练习

掌握一个主题的最好方法是解决问题。这里有一些热身运动。许多问题将涉及到查阅官方文件[1]和寻找有用的功能。

1. Create a 2D tensor and then add a dimension of size 1 inserted at dimension 0.
2. Remove the extra dimension you just added to the previous tensor.
3. Create a random tensor of shape 5x3 in the interval [3, 7)
4. Create a tensor with values from a normal distribution (mean=0, std=1).
5. Retrieve the indexes of all the nonzero elements in the tensor `torch.Tensor([1, 1, 1, 0, 1])`.
6. Create a random tensor of size (3,1) and then horizontally stack 4 copies together.
7. Return the batch matrix-matrix product of two 3-dimensional matrices (`a=torch.rand(3,4,5)`, `b=torch.rand(3,5,4)`).
8. Return the batch matrix-matrix product of a 3D matrix and a 2D matrix (`a=torch.rand(3,4,5)`, `b=torch.rand(5,4)`).

# Solutions

9. `a = torch.rand(3, 3)` `a.unsqueeze(0)`
10. `a.squeeze(0)`
11. `3 + torch.rand(5, 3) * (7 - 3)`
12. `a = torch.rand(3, 3)` `a.normal_()`
13. `a = torch.Tensor([1, 1, 1, 0, 1])` `torch.nonzero(a)`
14. `a = torch.rand(3, 1)` `a.expand(3, 4)`
15. `a = torch.rand(3, 4, 5)` `b = torch.rand(3, 5, 4)` `torch.bmm(a, b)`
16. `a = torch.rand(3, 4, 5)` `b = torch.rand(5, 4)` `torch.bmm(a, b.unsqueeze(0).expand(a.size(0), * b.size()))`

# 总结

在这一章中，我们介绍了本书的目标——自然语言处理（NLP）和深度学习——并对监督学习范式进行了详细的理解。在本章的最后，您现在应该熟悉或至少了解各种术语，例如观察、目标、模型、参数、预测、损失函数、表示、学习/训练和推理。您还了解了如何使用单热编码对学习任务的输入（观察和目标）进行编码。我们还研究了基于计数的表示，如 TF 和 TF-IDF。我们首先了解了什么是计算图，静态和动态计算图，以及 PyTorch 张量操纵操作。在第二章中，我们对传统的 NLP 进行了概述。第二章，这一章应该为你奠定必要的基础，如果你对这本书的主题是新的，并为你的书的其余部分做准备。

重点是 TF-IDF

`词频（TF）= 某个词在文章中出现的次数 / 文章中的总词数`

`逆文档频率（IDF）= log(语料库的文档总数 / (包含该词的文档数 + 1))`

TF 应该很容易理解就是计算词频，IDF 衡量词的常见程度。为了计算 IDF 我们需要事先准备一个语料库用来模拟语言的使用环境，如果一个词越是常见，那么式子中分母越大，逆文档频率越接近 0。这里分母`+1`是为了避免分母为 0 的情况出现

`TF-IDF = 词频（TF）× 逆文档频率（IDF）`

TF-IDF 可以很好的实现提取文章中关键词的目的。
