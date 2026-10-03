
另一个不在 Python 中运行模型的即时加速仅在多线程环境中出现，但这时它可能是显著的：因为中间结果不是 Python 对象，计算不受所有 Python 并行化的威胁，即 GIL。这是我们之前考虑到的，并且当我们在服务器上使用跟踪模型时实现了这一点。

从经典的 PyTorch 执行一项操作后再查看下一项的方式转变过来，确实让 PyTorch 能够全面考虑计算：也就是说，它可以将计算作为一个整体来考虑。这为关键的优化和更高级别的转换打开了大门。其中一些主要适用于推断，而其他一些也可以在训练中提供显著的加速。

让我们通过一个快速示例来让你体会一下为什么一次查看多个操作会有益。当 PyTorch 在 GPU 上运行一系列操作时，它为每个操作调用一个子程序（在 CUDA 术语中称为*内核*）。每个内核从 GPU 内存中读取输入，计算结果，然后存储结果。因此，大部分时间通常不是用于计算，而是用于读取和写入内存。这可以通过仅读取一次，计算多个操作，然后在最后写入来改进。这正是 PyTorch JIT 融合器所做的。为了让你了解这是如何工作的，图 15.3 展示了长短期记忆（LSTM；[`en.wikipedia.org/wiki/ Long_short-term_memory`](https://en.wikipedia.org/wiki/Long_short-term_memory)）单元中进行的逐点计算，这是递归网络的流行构建块。

图 15.3 的细节对我们来说并不重要，但顶部有 5 个输入，底部有 2 个输出，中间有 7 个圆角指数表示的中间结果。通过在一个单独的 CUDA 函数中一次性计算所有这些，并将中间结果保留在寄存器中，JIT 将内存读取次数从 12 降低到 5，写入次数从 9 降低到 2。这就是 JIT 带来的巨大收益；它可以将训练 LSTM 网络的时间缩短四倍。这看似简单的技巧使得 PyTorch 能够显著缩小 LSTM 和在 PyTorch 中灵活定义的通用 LSTM 单元与像 cuDNN 这样提供的高度优化 LSTM 实现之间速度差距。

总之，使用 JIT 来避免 Python 的加速并不像我们可能天真地期望的那样大，因为我们被告知 Python 非常慢，但避免 GIL 对于多线程应用程序来说是一个重大胜利。JIT 模型的大幅加速来自 JIT 可以实现的特殊优化，但这些优化比仅仅避免 Python 开销更为复杂。

![](img/CH15_F03_Stevens2_GS.png)

图 15.3 LSTM 单元逐点操作。从顶部的五个输入，该块计算出底部的两个输出。中间的方框是中间结果，普通的 PyTorch 会将其存储在内存中，但 JIT 融合器只会保留在寄存器中。

### 15.3.2 PyTorch 作为接口和后端的双重性质

要理解如何摆脱 Python 的工作原理，有益的是在头脑中将 PyTorch 分为几个部分。我们在第 1.4 节中初步看到了这一点。我们的 PyTorch `torch.nn` 模块--我们在第六章第一次看到它们，自那以后一直是我们建模的主要工具--保存网络的参数，并使用功能接口实现：接受和返回张量的函数。这些被实现为 C++ 扩展，交给了 C++ 级别的自动求导启用层。 （然后将实际计算交给一个名为 ATen 的内部库，执行计算或依赖后端来执行，但这不重要。）

鉴于 C++ 函数已经存在，PyTorch 开发人员将它们制作成了官方 API。这就是 LibTorch 的核心，它允许我们编写几乎与其 Python 对应物相似的 C++ 张量操作。由于`torch.nn`模块本质上只能在 Python 中使用，C++ API 在一个名为`torch::nn`的命名空间中镜像它们，设计上看起来很像 Python 部分，但是独立的。

这将使我们能够在 C++ 中重新做我们在 Python 中做的事情。但这不是我们想要的：我们想要*导出*模型。幸运的是，PyTorch 还提供了另一个接口来访问相同的函数：PyTorch JIT。PyTorch JIT 提供了计算的“符号”表示。这个表示是*TorchScript 中间表示*（TorchScript IR，有时只是 TorchScript）。我们在第 15.2.2 节讨论延迟计算时提到了 TorchScript。在接下来的章节中，我们将看到如何获取我们 Python 模型的这种表示以及如何保存、加载和执行它们。与我们讨论常规 PyTorch API 时所述类似，PyTorch JIT 函数用于加载、检查和执行 TorchScript 模块也可以从 Python 和 C++ 中访问。

总结一下，我们有四种调用 PyTorch 函数的方式，如图 15.4 所示：从 C++ 和 Python 中，我们可以直接调用函数，也可以让 JIT 充当中介。所有这些最终都会调用 C++ 的 LibTorch 函数，从那里进入 ATen 和计算后端。

![](img/CH15_F04_Stevens2_GS.png)

图 15.4 调用 PyTorch 的多种方式

### 15.3.3 TorchScript

TorchScript 是 PyTorch 设想的部署选项的核心。因此，值得仔细研究它的工作原理。

创建 TorchScript 模型有两种简单直接的方式：追踪和脚本化。我们将在接下来的章节中分别介绍它们。在非常高的层面上，这两种方式的工作原理如下：

在*追踪*中，我们在第 15.2.2 节中使用过，使用样本（随机）输入执行我们通常的 PyTorch 模型。PyTorch JIT 对每个函数都有钩子（在 C++ autograd 接口中），允许它记录计算过程。在某种程度上，这就像在说“看我如何计算输出--现在你也可以这样做。”鉴于 JIT 仅在调用 PyTorch 函数（以及`nn.Module`）时才起作用，你可以在追踪时运行任何 Python 代码，但 JIT 只会注意到那些部分（尤其是对控制流一无所知）。当我们使用张量形状--通常是整数元组--时，JIT 会尝试跟踪发生的情况，但可能不得不放弃。这就是在追踪 U-Net 时给我们警告的原因。

在*脚本化*中，PyTorch JIT 查看我们计算的实际 Python 代码，并将其编译成 TorchScript IR。这意味着，虽然我们可以确保 JIT 捕获了程序的每个方面，但我们受限于编译器理解的部分。这就像在说“我告诉你如何做--现在你也这样做。”听起来真的像编程。

我们不是来讨论理论的，所以让我们尝试使用一个非常简单的函数进行追踪和脚本化，该函数在第一维上进行低效的加法：

```py
# In[2]:
def myfn(x):
    y = x[0]
    for i in range(1, x.size(0)):
        y = y + x[i]
    return y
```

我们可以追踪它：

```py
# In[3]:
inp = torch.randn(5,5)
traced_fn = torch.jit.trace(myfn, inp)
print(traced_fn.code)

# Out[3]:
def myfn(x: Tensor) -> Tensor:
  y = torch.select(x, 0, 0)                                                # ❶
  y0 = torch.add(y, torch.select(x, 0, 1), alpha=1)                        # ❷
  y1 = torch.add(y0, torch.select(x, 0, 2), alpha=1)
  y2 = torch.add(y1, torch.select(x, 0, 3), alpha=1)
  _0 = torch.add(y2, torch.select(x, 0, 4), alpha=1)
  return _0

TracerWarning: Converting a tensor to a Python index might cause the trace # ❸
to be incorrect. We can't record the data flow of Python values, so this
value will be treated as a constant in the future. This means the
trace might not generalize to other inputs!
```

❶ 在我们函数的第一行中进行索引

❷ 我们的循环--但完全展开并固定为 1...4，不管 x 的大小如何

❸ 令人害怕，但却如此真实！

我们看到了一个重要的警告--实际上，这段代码已经为五行修复了索引和添加，但对于四行或六行的情况并不会按预期处理。

这就是脚本化的用处所在：

```py
# In[4]:
scripted_fn = torch.jit.script(myfn)
print(scripted_fn.code)

# Out[4]:
def myfn(x: Tensor) -> Tensor:
  y = torch.select(x, 0, 0)
  _0 = torch.__range_length(1, torch.size(x, 0), 1)     # ❶
  y0 = y
  for _1 in range(_0):                                  # ❷
    i = torch.__derive_index(_1, 1, 1)
    y0 = torch.add(y0, torch.select(x, 0, i), alpha=1)  # ❸
  return y0
```

❶ PyTorch 从张量大小构建范围长度。

❷ 我们的 for 循环--即使我们必须采取看起来有点奇怪的下一行来获取我们的索引 i

❸ 我们的循环体，稍微冗长一点

我们还可以打印脚本化的图，这更接近 TorchScript 的内部表示：

```py
# In[5]:
xprint(scripted_fn.graph)
# end::cell_5_code[]

# tag::cell_5_output[]
# Out[5]:
graph(%x.1 : Tensor):
  %10 : bool = prim::Constant[value=1]()               # ❶
  %2 : int = prim::Constant[value=0]()
  %5 : int = prim::Constant[value=1]()
  %y.1 : Tensor = aten::select(%x.1, %2, %2)           # ❷
  %7 : int = aten::size(%x.1, %2)
  %9 : int = aten::__range_length(%5, %7, %5)          # ❸
  %y : Tensor = prim::Loop(%9, %10, %y.1)              # ❹
    block0(%11 : int, %y.6 : Tensor):
      %i.1 : int = aten::__derive_index(%11, %5, %5)
      %18 : Tensor = aten::select(%x.1, %2, %i.1)      # ❺
      %y.3 : Tensor = aten::add(%y.6, %18, %5)
      -> (%10, %y.3)
  return (%y)
```

❶ 看起来比我们需要的要冗长得多

❷ y 的第一个赋值

❸ 在看到代码后，我们可以识别出构建范围的方法。

❹ 我们的 for 循环返回它计算的值（y）。

❺ for 循环的主体：选择一个切片，并将其添加到 y 中

在实践中，您最常使用`torch.jit.script`作为装饰器的形式：

```py
@torch.jit.script
def myfn(x):
  ...
```

您也可以使用自定义的`trace`装饰器来处理输入，但这并没有流行起来。

尽管 TorchScript（语言）看起来像 Python 的一个子集，但存在根本性差异。如果我们仔细观察，我们会发现 PyTorch 已经向代码添加了类型规范。这暗示了一个重要的区别：TorchScript 是静态类型的--程序中的每个值（变量）都有且只有一个类型。此外，这些类型限于 TorchScript IR 具有表示的类型。在程序内部，JIT 通常会自动推断类型，但我们需要用它们的类型注释脚本化函数的任何非张量参数。这与 Python 形成鲜明对比，Python 中我们可以将任何内容分配给任何变量。

到目前为止，我们已经追踪函数以获取脚本化函数。但是我们很久以前就从仅在第五章中使用函数转向使用模块了。当然，我们也可以追踪或脚本化模型。然后，这些模型将大致表现得像我们熟悉和喜爱的模块。对于追踪和脚本化，我们分别将`Module`的实例传递给`torch.jit.trace`（带有示例输入）或`torch.jit.script`（不带示例输入）。这将给我们带来我们习惯的`forward`方法。如果我们想要暴露其他方法（这仅适用于`脚本化`）以便从外部调用，我们在类定义中用`@torch.jit.export`装饰它们。

当我们说 JIT 模块的工作方式与 Python 中的工作方式相同时，这包括我们也可以用它们进行训练。另一方面，这意味着我们需要为推断设置它们（例如，使用`torch.no_grad()`上下文），就像我们传统的模型一样，以使它们做正确的事情。

对于算法相对简单的模型--如 CycleGAN、分类模型和基于 U-Net 的分割--我们可以像之前一样追踪模型。对于更复杂的模型，一个巧妙的特性是我们可以在构建和追踪或脚本化模块时使用来自其他脚本化或追踪代码的脚本化或追踪函数，并且我们可以在调用`nn.Models`时追踪函数，但是我们需要将所有参数设置为不需要梯度，因为这些参数将成为追踪模型的常数。

由于我们已经看到了追踪，让我们更详细地看一个脚本化的实际示例。

### 15.3.4 脚本化追踪的间隙

在更复杂的模型中，例如用于检测的 Fast R-CNN 系列或用于自然语言处理的循环网络，像`for`循环这样的控制流位需要进行脚本化。同样，如果我们需要灵活性，我们会找到追踪器警告的代码片段。

代码清单 15.8 来自 utils/unet.py

```py
class UNetUpBlock(nn.Module):
    ...
    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]                            # ❶

    def forward(self, x, bridge):
        ...
        crop1 = self.center_crop(bridge, up.shape[2:])
 ...
```

❶ 追踪器在这里发出警告。

发生的情况是，JIT 神奇地用包含相同信息的 1D 整数张量替换了形状元组`up.shape`。现在切片`[2:]`和计算`diff_x`和`diff_y`都是可追踪的张量操作。然而，这并不能拯救我们，因为切片然后需要 Python `int`；在那里，JIT 的作用范围结束，给我们警告。

但是我们可以通过一种简单直接的方式解决这个问题：我们对`center_crop`进行脚本化。我们通过将`up`传递给脚本化的`center_crop`并在那里提取大小来略微更改调用者和被调用者之间的切割。除此之外，我们所需的只是添加`@torch.jit.script`装饰器。结果是以下代码，使 U-Net 模型可以无警告地进行追踪。

代码清单 15.9 从 utils/unet.py 重写的节选

```py
@torch.jit.script
def center_crop(layer, target):                         # ❶
    _, _, layer_height, layer_width = layer.size()
    _, _, target_height, target_width = target.size()   # ❷
    diff_y = (layer_height - target_height) // 2
    diff_x = (layer_width - target_width]) // 2
    return layer[:, :, diff_y:(diff_y + target_height),  diff_x:(diff_x + target_width)]                     # ❸

class UNetUpBlock(nn.Module):
    ...

    def forward(self, x, bridge):
        ...
        crop1 = center_crop(bridge, up)                 # ❹
  ...
```

❶ 更改签名，接受目标而不是目标大小

❷ 在脚本化部分内获取大小

❸ 索引使用我们得到的大小值。

❹ 我们调整我们的调用以传递上而不是大小。

我们可以选择的另一个选项--但我们这里不会使用--是将不可脚本化的内容移入在 C++ 中实现的自定义运算符中。TorchVision 库为 Mask R-CNN 模型中的一些特殊操作执行此操作。

## 15.4 LibTorch：在 C++ 中使用 PyTorch

我们已经看到了各种导出模型的方式，但到目前为止，我们使用了 Python。现在我们将看看如何放弃 Python 直接使用 C++。
