
# PyTorch 张量操作与广播

当添加 PyTorch 张量和兼容的基本 Python 数据类型（浮点数、整数或布尔值）时，PyTorch 会自动将后者转换为 PyTorch 标量张量（这称为*类型强制转换*）。因此，这些操作

```py
pt.arange(10) + 1
```

和

```py
pt.arange(10) + pt.tensor(1)
```

是等价的。

PyTorch API 的默认实现对张量执行不可变操作。因此，在开发 PyTorch 机器学习代码时，您必须记住加法操作以及其他由 PyTorch 重载的标准 Python 数学运算符都会返回一个新的张量实例。您可以轻松通过运行以下命令进行确认

```py
a = pt.arange(10)
id(a), id(a + 1)
```

PyTorch 还提供了一组可就地（in-place）操作的操作符，这些操作符可更改张量的值。这意味着 PyTorch 将直接在张量设备的内存中替换张量的值，而不是为张量分配新的 `PyObject` 实例。例如，使用 `add_` 方法

```py
a = pt.arange(10)
id(a), id(a.add_(1))
```

返回一个带有两个相同对象标识符的元组。

**注意**：在 PyTorch API 设计中，所有的就地（in place）更改张量的操作都使用 `_` 后缀，例如 `mul_` 表示就地乘法，`pow_` 表示就地幂运算，`abs_` 表示就地绝对值函数等等。⁹

在处理机器学习代码时，您肯定会发现自己不得不在非标量张量上执行常见的数学运算。例如，如果给定两个张量 `a` 和 `b`，则 PyTorch 找到 `a + b` 的值是什么意思？

**列出 5.3** 张量按元素加和，因为它们具有相同的形状

```py
a = pt.arange(10)
b = pt.ones(10)
```

正如您所期望的那样，因为 `a` 的值是

```py
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

`b` 的值是

```py
tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
```

他们的和等于

```py
tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
```

因此，`a + b` 相当于将张量 `a` 的每个元素递增 1。这个操作可以被描述为张量 `a` 和 `b` 的逐元素相加，因为对于张量 `a` 的每个元素，PyTorch 找到仅一个对应的张量 `b` 的元素索引值，并将它们相加产生输出。

如果尝试添加

```py
a = pt.ones([2, 5])
b = pt.ones(5)
```

在 tensor `a` 的逻辑中，元素按位相加没有立即意义，这意味着应将张量 `a` 的哪些元素按 1 递增？应将张量 `a` 的第一行，第二行还是两行都增加 1？

要理解此示例中加法的原理以及在操作中张量的形状不同时的其他情况，您需要熟悉*broadcasting*（¹⁰），PyTorch 在幕后执行它来生成 `a + b` 的下面的结果:

```py
tensor([[2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 2.]])
```

当操作中使用的张量的形状不相同时，PyTorch 尝试执行张量广播。在两个大小不同的张量上执行操作时，一些维度可能会被重用或广播以完成操作。如果给定两个张量 `a` 和 `b`，并且要在两个张量之间执行广播运算，可以通过调用 `can_broadcast` 来检查能否执行此操作。

**列出 5.4**，当 `can_broadcast` 返回 `true` 时，可以 broadcast

```py
def can_broadcast(a, b):
  return all( [x == y or x == 1 or y == 1 \
    for x, y in zip( a.shape[::-1], b.shape[::-1] ) ])
```

这个广播规则取决于张量的尾部维度，或者以相反顺序对齐的张量的维度。即使是将标量添加到张量的简单示例也涉及广播：当 `a = pt.ones(5)` 且 `b = pt.tensor(42)` 时，它们的形状分别为 `torch.Size([5])` 和 `torch.Size([])`。因此，标量必须像图 5.2 中所示一样广播五次到张量 `a`。

![05-02](img/05-02.png)

**图 5.2** 将标量 `b` 广播到秩为 1 的张量 `a`

广播不需要在内存中复制或复制张量数据；相反，从操作中使用的张量的值直接计算产生广播结果的张量的内容。有效地使用和理解广播可以帮助您减少张量所需的内存量，并提高张量操作的性能。

为了用更复杂的示例说明广播，其中 `a = pt.ones([2, 5])` 且 `b = pt.ones(5)`，请注意广播重复使用张量 `b` 的值（图 5.3 的右侧），以便在生成 `a + b` 张量时对齐结果的尾部维度，同时保留来自张量 `a` 的前导维度。

![05-03](img/05-03.png)

**图 5.3** 将秩为 1 的张量 `b` 广播到秩为 2 的张量 `a`

根据您目前所见的广播示例，您可能会错误地认为广播只发生在一个方向上：从操作中的一个张量到另一个张量。这是错误的。请注意，根据列表 5.4 中的规则，参与操作的两个张量都可以相互广播数据。例如，在图 5.4 中，张量 `a` 被广播到张量 `b` 三次（基于第一维），然后张量 `b` 的内容沿着相反的方向（沿第二维）广播，以产生尺寸为 `torch.Size([3, 2, 5])` 的结果张量 `a+b`。

![05-04](img/05-04.png)

**图 5.4** 广播是双向的，其中 `b` 的第一维广播到 `a`，`a` 的第二维广播到 `b`。

## PyTorch 张量 vs. 原生 Python 列表

在本节中，您将深入了解原生 Python 数据结构与 PyTorch 张量在内存使用方面的比较，并了解为什么 PyTorch 张量可以帮助您更有效地利用内存来处理机器学习用例。

大多数现代笔记本电脑使用的中央处理单元（CPU）的运行频率为 2 到 3 GHz。为了保持计算简单，让我们忽略一些现代处理器执行流水线功能的高级指令，并将 2 GHz 处理器频率解释为处理器大约需要半个纳秒（ns）来执行一条单个指令，例如执行加法运算并存储结果的指令。虽然处理器可以在少于 1 ns 的时间内执行一条指令，但处理器必须等待超过 100 倍的时间，从 50 到 100 ns 不等，才能从主计算机内存（动态随机访问存储器）中获取一段数据。当然，处理器使用的一些数据存储在缓存中，可以在单个数字纳秒内访问，但低延迟缓存的大小有限，通常以单个数字 MiB 为单位进行度量。[¹¹]

假设你正在编写一个计算机程序，需要对数据张量执行一些计算，比如处理一个包含 4,096 个整数的数组，并将每个整数加 1。为了使这样的程序获得高性能，可以使用较低级别的编程语言，如 C 或 C++，在计算机内存中为输入数据数组分配一个单一的块。例如，在 C 编程语言中，一个包含 4,096 个整数值的数组，每个整数值为 64 位，可以存储为某个连续内存范围内的 64 位值序列，例如从地址 `0x8000f` 到地址 `0x9000f`。[¹²]

假设所有 4,096 个整数值都在连续的内存范围内，那么这些值可以作为一个单一块从主存储器传输到处理器缓存中，有助于减少对值的加法计算的总延迟。如图 5.5 的左侧所示，C 整数仅占用足够的内存以存储整数值，以便可以将一系列 C 整数存储为可寻址的内存位置序列。请注意，数字 4,096 是有意选择的：因为 4,096 * 8（每个 64 位整数的字节数）= 32,768 字节是 2020 年 x86 处理器的常见 L1 缓存大小。这意味着每次需要刷新缓存并用另外 4,096 个整数重新填充缓存时，都会产生大约 100 ns 的延迟惩罚，这些整数需要从计算机内存中获取。

![05-05](img/05-05.png)

**图 5.5** C 整数的值（左侧）直接存储为可寻址的内存位置。Python 整数（右侧）存储为对通用 `PyObject_HEAD` 结构的地址引用，该结构指定了数据类型（整数）和数据值。

这种高性能方法不适用于本地 Python 整数或列表。在 Python 中，所有数据类型都以 Python 对象（`PyObjects`）的形式存储在计算机内存中。这意味着对于任何数据值，Python 分配内存来存储值以及称为 `PyObject_HEAD` 的元数据描述符（图 5.5 右侧），该描述符跟踪数据类型（例如，数据位描述整数还是浮点数）和支持元数据，包括一个引用计数器，用于跟踪数据值是否正在使用。对于浮点数和其他原始数据值，`PyObject` 元数据的开销可能会使存储数据值所需的内存量增加两倍以上。

从性能角度来看，情况更糟糕的是，Python 列表（例如，如图 5.6 左侧所示的 `list PyObject`）通过引用存储值到它们的 `PyObject` 内存地址（图 5.6 右侧）并且很少将所有值存储在连续的内存块中。由于 Python 列表存储的每个 `PyObject` 可能分散在计算机内存中的许多位置，所以在最坏的情况下，每个 Python 列表中的值可能会有 100 纳秒的潜在延迟惩罚，因为需要为每个值刷新并重新填充缓存。

![05-06](img/05-06.png)

**图 5.6** 中的整数在 Python 列表（`PyListObject`）中通过引用（内存地址）作为 `PyListObjects` 中的项目进行访问，需要额外的内存访问以查找每个整数的 `PyObject`。根据内存碎片化程度，单个整数的 `PyObjects` 可以分散在内存中，导致频繁的缓存未命中。

PyTorch 张量（以及其他库，如 NumPy）使用基于低级别 C 代码实现高性能数据结构，以克服较高级别的 Python 本地数据结构的低效率。具体而言，PyTorch 使用基于 C 的 ATen 张量库¹⁴，确保 PyTorch 张量使用友好的缓存、连续的内存块（在 ATen 中称为 *blobs*）存储底层数据，并提供了从 Python 到 C++ 的绑定，以支持通过 PyTorch Python API 访问数据。

为了说明性能差异，请看下面的代码片段，使用 Python 的 `timeit` 库来测量处理长度从 2 到大约 268 百万（2²⁸）的整数值列表的性能，并将列表中的每个值递增 1，

```py
import timeit
sizes = [2 ** i for i in range(1, 28)]

pylist = [ timeit.timeit(lambda: [i + 1 for i in list(range(size))],
                          number = 10) for size in sizes ]
```

使用类似的方法来测量递增张量数组中值所需的时间：

```py
pytorch = [ timeit.timeit(lambda: pt.tensor(list(range(size))) + 1,
                          number = 10) for size in sizes ]
```

如图 5.7 所示，可以通过将大小作为 x 轴、比率作为 y 轴的图形比较 PyTorch 张量与本地 Python 列表的性能，其中比率定义为：

```py
ratio = [pylist[i] / pytorch[i] for i in range(len(pylist))]
```

![05-07](img/05-07.png)

**图 5.7** Python 到 PyTorch 性能比的比例显示了增量操作基准测试的一致更快的 PyTorch 性能，从具有 10,000 个元素及更高数量的列表开始。

## 摘要

+   PyTorch 是一个面向深度学习的框架，支持基于高性能张量的机器学习算法。
+   PyTorch 张量将标量、数组（列表）、矩阵和更高维数组泛化为单个高性能数据结构。
+   使用多个张量的操作，包括张量加法和乘法，依赖于广播以对齐张量形状。
+   PyTorch 中基于 C/C++ 的张量比 Python 本地数据结构更节省内存，并且能够实现更高的计算性能。

## 参考链接

由特斯拉 AI 主管 Andrej Karpathy 提出的 PyTorch 机器学习用例：[`www.youtube.com/watch?v=oBklltKXtDE`](https://www.youtube.com/watch?v=oBklltKXtDE)。

OpenAI 因创建基于 PyTorch 的最先进自然语言处理 GPT 模型而闻名：[`openai.com/blog/openai-pytorch/`](https://openai.com/blog/openai-pytorch/)。

当然，在 Python 中可以使用切片符号，但这与本说明无关。

有关 PyTorch 中所有子类的完整列表，请参阅 `torch.Tensor` 文档：[` pytorch.org/docs/stable/tensors.html#torch-tensor`](https://pytorch.org/docs/stable/tensors.html#torch-tensor)。

PyTorch 支持的 `dtype` 值的全面列表可在[`mng.bz/YwyB`](http://mng.bz/YwyB)上找到。

`NestedTensor` 类作为 PyTorch 包在这里提供：[`github.com/pytorch/nestedtensor`](https://github.com/pytorch/nestedtensor)。

关于 PyTorch 张量数据类型的详细信息，请参阅[`pytorch.org/docs/stable/ tensors.html`](https://pytorch.org/docs/stable/tensors.html)。

有关 PyTorch 随机抽样工厂方法的详细文档，请访问[`mng.bz/GOqv`](http://mng.bz/GOqv)。

关于原地张量操作的详细参考，请访问[`pytorch.org/docs/stable/tensors.html`](https://pytorch.org/docs/stable/tensors.html) 并搜索“in-place”。

广播是各种计算库和软件包（如 NumPy、Octave 等）中常用的技术。有关 PyTorch 广播的更多信息，请访问[`pytorch.org/docs/stable/ notes/broadcasting.html`](https://pytorch.org/docs/stable/notes/broadcasting.html)。

了解每位计算机程序员都应该知道的延迟数字的全面解析，请访问[`gist.github.com/jboner/2841832`](https://gist.github.com/jboner/2841832)。

当然，现代计算机程序的实际内存地址不太可能具有像 `0x8000f` 或 `0x9000f` 的值；这些值仅用于说明目的。

^(13.)Python 使用此引用计数器来确定数据值是否不再使用，以便可以安全地释放用于数据的内存，并释放给其他数据使用。

^(14.)有关 ATen 文档，请访问[`pytorch.org/cppdocs/#aten`](https://pytorch.org/cppdocs/#aten)。
