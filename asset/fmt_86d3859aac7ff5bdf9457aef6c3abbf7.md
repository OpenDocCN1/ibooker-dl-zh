
图 3.5 张量的偏移、尺寸和步幅之间的关系。这里的张量是一个更大存储的视图，就像在创建更大的张量时可能分配的存储一样。

### 3.8.1 另一个张量存储的视图

通过提供相应的索引，我们可以获取张量中的第二个点：

```py
# In[21]:
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points[1]
second_point.storage_offset()

# Out[21]:
2

# In[22]:
second_point.size()

# Out[22]:
torch.Size([2])
```

结果张量在存储中的偏移为 2（因为我们需要跳过第一个点，它有两个项目），尺寸是`Size`类的一个实例，包含一个元素，因为张量是一维的。重要的是要注意，这与张量对象的`shape`属性中包含的信息相同：

```py
# In[23]:
second_point.shape

# Out[23]:
torch.Size([2])
```

步幅是一个元组，指示当索引在每个维度上增加 1 时，必须跳过存储中的元素数量。例如，我们的`points`张量的步幅是`(2, 1)`：

```py
# In[24]:
points.stride()

# Out[24]:
(2, 1)
```

在 2D 张量中访问元素`i, j`会导致访问存储中的`storage_offset + stride[0] * i + stride[1] * j`元素。偏移通常为零；如果这个张量是一个查看存储的视图，该存储是为容纳更大的张量而创建的，则偏移可能是一个正值。

`Tensor`和`Storage`之间的这种间接关系使得一些操作变得廉价，比如转置张量或提取子张量，因为它们不会导致内存重新分配。相反，它们包括为尺寸、存储偏移或步幅分配一个具有不同值的新`Tensor`对象。

当我们索引特定点并看到存储偏移增加时，我们已经提取了一个子张量。让我们看看尺寸和步幅会发生什么变化：

```py
# In[25]:
second_point = points[1]
second_point.size()

# Out[25]:
torch.Size([2])

# In[26]:
second_point.storage_offset()

# Out[26]:
2

# In[27]:
second_point.stride()

# Out[27]:
(1,)
```

底线是，子张量的维度少了一个，正如我们所期望的那样，同时仍然索引与原始`points`张量相同的存储。这也意味着改变子张量将对原始张量产生副作用：

```py
# In[28]:
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points[1]
second_point[0] = 10.0
points

# Out[28]:
tensor([[ 4.,  1.],
        [10.,  3.],
        [ 2.,  1.]])
```

这可能并不总是理想的，所以我们最终可以将子张量克隆到一个新的张量中：

```py
# In[29]:
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points[1].clone()
second_point[0] = 10.0
points

# Out[29]:
tensor([[4., 1.],
        [5., 3.],
        [2., 1.]])
```

### 3.8.2 在不复制的情况下转置

现在让我们尝试转置。让我们拿出我们的`points`张量，其中行中有单独的点，列中有*X*和*Y*坐标，并将其转向，使单独的点在列中。我们借此机会介绍`t`函数，这是二维张量的`transpose`的简写替代品：

```py
# In[30]:
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points

# Out[30]:
tensor([[4., 1.],
        [5., 3.],
        [2., 1.]])

# In[31]:
points_t = points.t()
points_t

# Out[31]:
tensor([[4., 5., 2.],
        [1., 3., 1.]])
```

*提示* 为了帮助建立对张量机制的扎实理解，可能是一个好主意拿起一支铅笔和一张纸，像图 3.5 中的图一样在我们逐步执行本节代码时涂鸦图表。

我们可以轻松验证这两个张量共享相同的存储

```py
# In[32]:
id(points.storage()) == id(points_t.storage())

# Out[32]:
True
```

它们只在形状和步幅上有所不同：

```py
# In[33]:
points.stride()

# Out[33]:
(2, 1)
# In[34]:
points_t.stride()

# Out[34]:
(1, 2)
```

这告诉我们，在`points`中将第一个索引增加 1（例如，从`points[0,0]`到`points[1,0]`）将会跳过存储中的两个元素，而增加第二个索引（从`points[0,0]`到`points[0,1]`）将会跳过存储中的一个元素。换句话说，存储按行顺序顺序保存张量中的元素。

我们可以将`points`转置为`points_t`，如图 3.6 所示。我们改变了步幅中元素的顺序。之后，增加行（张量的第一个索引）将沿着存储跳过一个元素，就像我们在`points`中沿着列移动一样。这就是转置的定义。不会分配新的内存：转置只是通过创建一个具有不同步幅顺序的新`Tensor`实例来实现的。

![](img/CH03_F06_Stevens2_GS.png)

图 3.6 张量的转置操作

### 3.8.3 高维度中的转置

在 PyTorch 中，转置不仅限于矩阵。我们可以通过指定应该发生转置（翻转形状和步幅）的两个维度来转置多维数组：

```py
# In[35]:
some_t = torch.ones(3, 4, 5)
transpose_t = some_t.transpose(0, 2)
some_t.shape

# Out[35]:
torch.Size([3, 4, 5])

# In[36]:
transpose_t.shape

# Out[36]:
torch.Size([5, 4, 3])

# In[37]:
some_t.stride()

# Out[37]:
(20, 5, 1)

# In[38]:
transpose_t.stride()

# Out[38]:
(1, 5, 20)
```

从存储中右起维度开始排列数值（即，对于二维张量，沿着行移动）的张量被定义为`contiguous`。连续张量很方便，因为我们可以有效地按顺序访问它们，而不需要在存储中跳跃（改善数据局部性会提高性能，因为现代 CPU 的内存访问方式）。当然，这种优势取决于算法的访问方式。

### 3.8.4 连续张量

PyTorch 中的一些张量操作仅适用于连续张量，例如我们将在下一章中遇到的`view`。在这种情况下，PyTorch 将抛出一个信息性异常，并要求我们显式调用`contiguous`。值得注意的是，如果张量已经是连续的，则调用`contiguous`不会做任何事情（也不会影响性能）。

在我们的例子中，`points`是连续的，而其转置则不是：

```py
# In[39]:
points.is_contiguous()

# Out[39]:
True

# In[40]:
points_t.is_contiguous()

# Out[40]:
False
```

我们可以使用`contiguous`方法从非连续张量中获得一个新的连续张量。张量的内容将保持不变，但步幅和存储将发生变化：

```py
# In[41]:
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_t = points.t()
points_t

# Out[41]:
tensor([[4., 5., 2.],
        [1., 3., 1.]])

# In[42]:
points_t.storage()

# Out[42]:
 4.0
 1.0
 5.0
 3.0
 2.0
 1.0
[torch.FloatStorage of size 6]

# In[43]:
points_t.stride()

# Out[43]:
(1, 2)

# In[44]:
points_t_cont = points_t.contiguous()
points_t_cont

# Out[44]:
tensor([[4., 5., 2.],
        [1., 3., 1.]])

# In[45]:
points_t_cont.stride()

# Out[45]:
(3, 1)

# In[46]:
points_t_cont.storage()

# Out[46]:
 4.0
 5.0
 2.0
 1.0
 3.0
 1.0
[torch.FloatStorage of size 6]
```

请注意，存储已经重新排列，以便元素按行排列在新存储中。步幅已更改以反映新布局。

作为复习，图 3.7 再次显示了我们的图表。希望现在我们已经仔细研究了张量是如何构建的，一切都会变得清晰。

![](img/CH03_F07_Stevens2_GS.png)

图 3.7 张量的偏移、大小和步幅之间的关系。这里的张量是一个更大存储的视图，就像在创建更大的张量时可能分配的存储一样。

## 3.9 将张量移动到 GPU

到目前为止，在本章中，当我们谈论存储时，我们指的是 CPU 上的内存。PyTorch 张量也可以存储在不同类型的处理器上：图形处理单元（GPU）。每个 PyTorch 张量都可以传输到 GPU 中的一个（或多个）以执行高度并行、快速的计算。将在张量上执行的所有操作都将使用 PyTorch 提供的 GPU 特定例程执行。

PyTorch 对各种 GPU 的支持

截至 2019 年中期，主要的 PyTorch 发行版只在支持 CUDA 的 GPU 上有加速。PyTorch 可以在 AMD 的 ROCm 上运行（[`rocm.github.io`](https://rocm.github.io)），主存储库提供支持，但到目前为止，您需要自行编译它。（在常规构建过程之前，您需要运行`tools/amd_build/build_amd.py`来转换 GPU 代码。）对 Google 的张量处理单元（TPU）的支持正在进行中（[`github.com/pytorch/xla`](https://github.com/pytorch/xla)），当前的概念验证可在 Google Colab 上公开访问：https://colab.research.google.com。在撰写本文时，不计划在其他 GPU 技术（如 OpenCL）上实现数据结构和内核。](https://colab.research.google.com)

### 3.9.1 管理张量的设备属性

除了`dtype`，PyTorch 的`Tensor`还有`device`的概念，即张量数据所放置的计算机位置。以下是我们如何通过为构造函数指定相应参数来在 GPU 上创建张量的方法：

```py
# In[64]:
points_gpu = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]], device='cuda')
```

我们可以使用`to`方法将在 CPU 上创建的张量复制到 GPU 上：

```py
# In[65]:
points_gpu = points.to(device='cuda')
```

这样做会返回一个新的张量，其中包含相同的数值数据，但存储在 GPU 的 RAM 中，而不是常规系统 RAM 中。现在数据存储在 GPU 上，当对张量执行数学运算时，我们将开始看到之前提到的加速效果。在几乎所有情况下，基于 CPU 和 GPU 的张量都暴露相同的用户接口，这样编写代码就更容易，不用关心重要的数值计算到底在哪里运行。

如果我们的机器有多个 GPU，我们还可以通过传递一个从零开始的整数来决定将张量分配到哪个 GPU 上，例如

```py
# In[66]:
points_gpu = points.to(device='cuda:0')
```

此时，对张量执行的任何操作，例如将所有元素乘以一个常数，都是在 GPU 上执行的：

```py
# In[67]:
points = 2 * points                        # ❶
points_gpu = 2 * points.to(device='cuda')  # ❷
```

❶ 在 CPU 上执行的乘法

❷ 在 GPU 上执行的乘法

请注意，`points_gpu`张量在计算结果后并没有返回到 CPU。这是这一行中发生的事情：

1. `points`张量被复制到 GPU 上。
2. 在 GPU 上分配一个新的张量，并用于存储乘法的结果。
3. 返回一个指向该 GPU 张量的句柄。

因此，如果我们还向结果添加一个常数

```py
# In[68]:
points_gpu = points_gpu + 4
```

加法仍然在 GPU 上执行，没有信息流向 CPU（除非我们打印或访问生成的张量）。为了将张量移回 CPU，我们需要在`to`方法中提供一个`cpu`参数，例如

```py
# In[69]:
points_cpu = points_gpu.to(device='cpu')
```

我们还可以使用`cpu`和`cuda`的简写方法，而不是`to`方法来实现相同的目标：

```py
# In[70]:
points_gpu = points.cuda()      # ❶
points_gpu = points.cuda(0)
points_cpu = points_gpu.cpu()
```

❶ 默认为 GPU 索引 0

还值得一提的是，通过使用`to`方法，我们可以通过同时提供`device`和`dtype`作为参数来同时更改位置和数据类型。

## 3.10 NumPy 互操作性

我们在这里和那里提到了 NumPy。虽然我们不认为 NumPy 是阅读本书的先决条件，但我们强烈建议您熟悉 NumPy，因为它在 Python 数据科学生态系统中无处不在。PyTorch 张量可以与 NumPy 数组之间进行非常高效的转换。通过这样做，我们可以利用围绕 NumPy 数组类型构建起来的 Python 生态系统中的大量功能。这种与 NumPy 数组的零拷贝互操作性归功于存储系统与 Python 缓冲区协议的工作（[`docs.python.org/3/c-api/buffer.html`](https://docs.python.org/3/c-api/buffer.html)）。

要从我们的`points`张量中获取一个 NumPy 数组，我们只需调用

```py
# In[55]:
points = torch.ones(3, 4)
points_np = points.numpy()
points_np

# Out[55]:
array([[1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.]], dtype=float32)
```

这将返回一个正确大小、形状和数值类型的 NumPy 多维数组。有趣的是，返回的数组与张量存储共享相同的底层缓冲区。这意味着`numpy`方法可以在基本上不花费任何成本地执行，只要数据位于 CPU RAM 中。这也意味着修改 NumPy 数组将导致源张量的更改。如果张量分配在 GPU 上，PyTorch 将把张量内容复制到在 CPU 上分配的 NumPy 数组中。

相反，我们可以通过以下方式从 NumPy 数组获得一个 PyTorch 张量

```py
# In[56]:
points = torch.from_numpy(points_np)
```

这将使用我们刚刚描述的相同的缓冲区共享策略。

*注意* PyTorch 中的默认数值类型是 32 位浮点数，而 NumPy 中是 64 位。正如在第 3.5.2 节中讨论的那样，我们通常希望使用 32 位浮点数，因此在转换后，我们需要确保具有`dtype torch .float`的张量。

## 3.11 广义张量也是张量

对于本书的目的，以及一般大多数应用程序，张量都是多维数组，就像我们在本章中看到的那样。如果我们冒险窥探 PyTorch 的内部，会有一个转折：底层数据存储方式与我们在第 3.6 节讨论的张量 API 是分开的。只要满足该 API 的约定，任何实现都可以被视为张量！

PyTorch 将调用正确的计算函数，无论我们的张量是在 CPU 还是 GPU 上。这是通过*调度*机制实现的，该机制可以通过将用户界面 API 连接到正确的后端函数来满足其他张量类型的需求。确实，还有其他种类的张量：有些特定于某些类别的硬件设备（如 Google TPU），而其他的数据表示策略与我们迄今所见的稠密数组风格不同。例如，稀疏张量仅存储非零条目，以及索引信息。图 3.8 左侧的 PyTorch 调度程序被设计为可扩展的；图 3.8 右侧所示的用于适应各种数字类型的后续切换是实现的固定方面，编码到每个后端中。

![](img/CH03_F08_Stevens2_GS.png)

图 3.8 PyTorch 中的调度程序是其关键基础设施之一。

我们将在第十五章遇到*量化*张量，它们作为另一种具有专门计算后端的张量类型实现。有时，我们使用的通常张量被称为*稠密*或*分步*，以区别于使用其他内存布局的张量。

与许多事物一样，随着 PyTorch 支持更广泛的硬件和应用程序范围，张量种类的数量也在增加。我们可以期待随着人们探索用 PyTorch 表达和执行计算的新方法，新的种类将继续出现。
