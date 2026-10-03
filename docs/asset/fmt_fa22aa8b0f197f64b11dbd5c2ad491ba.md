
# 自动微分框架实现详解

## 核心张量类

以下是实现自动微分功能的核心`Tensor`类代码：

```
import numpy as np
class Tensor (object):

    def __init__(self,data,
                 autograd=False,
                 creators=None,
                 creation_op=None,
                 id=None):

        self.data = np.array(data)
        self.creators = creators
        self.creation_op = creation_op
        self.grad = None
        self.autograd = autograd
        self.children = {}
        if(id is None):
            id = np.random.randint(0,100000)
        self.id = id

        if(creators is not None):
            for c in creators:
                if(self.id not in c.children):                        *1*
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1

    def all_children_grads_accounted_for(self):                       *2*
        for id,cnt in self.children.items():
            if(cnt != 0):
                return False
        return True

    def backward(self,grad=None, grad_origin=None):
        if(self.autograd):
            if(grad_origin is not None):
                if(self.children[grad_origin.id] == 0):               *3*
                    raise Exception("cannot backprop more than once")
                else:
                    self.children[grad_origin.id] -= 1

            if(self.grad is None):
                self.grad = grad                                      *4*
            else:
                self.grad += grad

            if(self.creators is not None and
               (self.all_children_grads_accounted_for() or
                grad_origin is None)):

                if(self.creation_op == "add"):
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)        *5*

    def __add__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data + other.data,
                          autograd=True,
                          creators=[self,other],
                          creation_op="add")
        return Tensor(self.data + other.data)

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())

a = Tensor([1,2,3,4,5], autograd=True)
b = Tensor([2,2,2,2,2], autograd=True)
c = Tensor([5,4,3,2,1], autograd=True)

d = a + b
e = b + c
f = d + e

f.backward(Tensor(np.array([1,1,1,1,1])))

print(b.grad.data == np.array([2,2,2,2,2]))
```

+   ***1* 跟踪张量有多少个子张量**

+   ***2* 检查张量是否从每个子张量接收到了正确数量的梯度**

+   ***3* 检查是否可以反向传播，或者你是否正在等待梯度，如果是，则递减计数器**

+   ***4* 从多个子节点累积梯度**

+   ***5* 开始实际的反向传播**

```
[ True  True  True  True  True]
```

此外，你创建了一个`self.children`计数器，在反向传播期间计算每个子节点接收到的梯度数量。这样，你也可以防止变量意外地从同一个子节点两次进行反向传播（这会抛出异常）。

第二个新增功能是一个具有相当冗长名称的新函数`all_children_grads_accounted_for()`。这个函数的目的是计算一个张量是否从其图中的所有子节点接收到了梯度。通常，每当在图中的中间变量上调用`.backward()`时，它会立即对其父节点调用`.backward()`。但由于一些变量从多个父节点接收梯度值，每个变量需要等待直到它具有局部的最终梯度后，再调用`.backward()`对其父节点进行反向传播。

如前所述，从深度学习理论的角度来看，这些概念并不新颖；这些都是深度学习框架试图面对的工程挑战。更重要的是，它们是你在标准框架中调试神经网络时将面临的那种挑战。在继续之前，花点时间玩一下这段代码，熟悉它。尝试删除不同的部分，看看它以各种方式崩溃。尝试两次调用`.backprop()`。

## 加法反向传播的工作原理

### 研究抽象以支持更多函数

到目前为止，这个框架已经达到了一个令人兴奋的地方！你现在可以通过将函数添加到`Tensor`类并添加其导数到`.backward()`方法来支持任意操作。对于加法，有以下方法：

```
def __add__(self, other):
    if(self.autograd and other.autograd):
        return Tensor(self.data + other.data,
                      autograd=True,
                      creators=[self,other],
                      creation_op="add")
    return Tensor(self.data + other.data)
```

对于通过加法函数的反向传播，以下是`.backward()`方法中的以下梯度传播：

```
           if(self.creation_op == "add"):
             self.creators[0].backward(self.grad, self)
             self.creators[1].backward(self.grad, self)
```

注意，在这个类中其他地方并没有处理加法。通用的反向传播逻辑被抽象化，所以所有必要的加法定义都包含在这两个地方。进一步注意，反向传播逻辑在每次加法中都会调用`.backward()`两次，一次针对每个参与加法的变量。因此，反向传播逻辑的默认设置是始终将反向传播应用于图中的每个变量。但有时，如果变量关闭了自动微分（`self.autograd` == `False`），则会跳过反向传播。这个检查是在`.backward()`方法中进行的：

![](img/f0240-01_alt.jpg)

尽管加法的反向传播逻辑将梯度反向传播到所有对其有贡献的变量，但除非将该变量的 `.autograd` 设置为 `True`，否则反向传播不会运行（对于 `self.creators[0]` 或 `self.creators[1]` 分别）。注意，在 `__add__()` 的第一行中，创建的张量（稍后是 `running.backward()`）的 `self.autograd` 仅当 `self.autograd` == `other.autograd` == `True` 时才为 `True`。

## 添加对否定的支持

### 修改加法以支持否定

现在加法功能已经正常工作，你应该能够复制并粘贴加法代码，进行一些修改，并为否定添加自动微分支持。让我们试试。`__add__` 函数的修改内容以粗体显示：

```
def __neg__(self):
    if(self.autograd):
        return Tensor(self.data * -1,
                      autograd=True,
                      creators=[self],
                      creation_op="neg")
    return Tensor(self.data * -1)
```

几乎所有内容都是相同的。你不接受任何参数，所以“other”参数已在多个地方被移除。让我们看看你应该添加到 `.backward()` 中的反向传播逻辑。`__add__` 函数反向传播逻辑的修改内容以粗体显示：

```
             if(self.creation_op == "neg"):
               self.creators[0].backward(self.grad.__neg__())
```

因为 `__neg__` 函数只有一个创建者，所以你只需要调用一次 `.backward()`。如果你想知道如何知道正确的梯度进行反向传播，请回顾第 4、5 和 6 章。你现在可以测试新的代码了：

```
a = Tensor([1,2,3,4,5], autograd=True)
b = Tensor([2,2,2,2,2], autograd=True)
c = Tensor([5,4,3,2,1], autograd=True)

d = a + (-b)
e = (-b) + c
f = d + e

f.backward(Tensor(np.array([1,1,1,1,1])))

print(b.grad.data == np.array([-2,-2,-2,-2,-2]))

[ True  True  True  True  True]
```

当你使用 `-b` 而不是 `b` 进行前向传播时，反向传播的梯度也会翻转符号。此外，你不需要对一般的反向传播系统做任何修改来使其工作。你可以根据需要创建新的函数。让我们添加一些吧！

## 添加对其他函数的支持

### 减法、乘法、求和、扩展、转置和矩阵乘法

使用你为加法和否定学到的相同思想，让我们添加几个其他函数的前向和反向传播逻辑：

```
def __sub__(self, other):
     if(self.autograd and other.autograd):
         return Tensor(self.data - other.data,
                       autograd=True,
                       creators=[self,other],
                       creation_op="sub")
     return Tensor(self.data - other.data)

 def __mul__(self, other):
     if(self.autograd and other.autograd):
         return Tensor(self.data * other.data,
                       autograd=True,
                       creators=[self,other],
                       creation_op="mul")
     return Tensor(self.data * other.data)

 def sum(self, dim):
     if(self.autograd):
         return Tensor(self.data.sum(dim),
                        autograd=True,
                        creators=[self],
                        creation_op="sum_"+str(dim))
     return Tensor(self.data.sum(dim))

 def expand(self, dim,copies):

     trans_cmd = list(range(0,len(self.data.shape)))
     trans_cmd.insert(dim,len(self.data.shape))
     new_shape = list(self.data.shape) + [copies]
     new_data = self.data.repeat(copies).reshape(new_shape)
     new_data = new_data.transpose(trans_cmd)

     if(self.autograd):
         return Tensor(new_data,
                       autograd=True,
                       creators=[self],
                       creation_op="expand_"+str(dim))
     return Tensor(new_data)

 def transpose(self):
     if(self.autograd):
         return Tensor(self.data.transpose(),
                       autograd=True,
                       creators=[self],
                       creation_op="transpose")
     return Tensor(self.data.transpose())

 def mm(self, x):
     if(self.autograd):
         return Tensor(self.data.dot(x.data),
                       autograd=True,
                       creators=[self,x],
                       creation_op="mm")
     return Tensor(self.data.dot(x.data))
```

我们之前讨论了所有这些函数的导数，尽管 `sum` 和 `expand` 可能看起来有些陌生，因为它们有新的名称。`sum` 在张量的一个维度上执行加法；换句话说，假设你有一个名为 `x` 的 2 × 3 矩阵：

```
            x = Tensor(np.array([[1,2,3],
                                [4,5,6]]))
```

`.sum(dim)` 函数在一个维度上求和。`x.sum(0)` 将得到一个 1 × 3 矩阵（长度为 3 的向量），而 `x.sum(1)` 将得到一个 2 × 1 矩阵（长度为 2 的向量）：

![](img/f0243-00_alt.jpg)

你使用 `expand` 来通过 `.sum()` 进行反向传播。这是一个复制数据沿一个维度的函数。给定相同的矩阵 `x`，沿第一个维度复制会得到两个张量副本：

![](img/f0243-01_alt.jpg)

为了明确起见， whereas `.sum()` 移除一个维度（2 × 3 -> 只剩 2 或 3），`expand` 添加一个维度。2 × 3 的矩阵变成了 4 × 2 × 3。您可以将其视为一个包含四个张量的列表，每个张量都是 2 × 3。但如果您扩展到最后一个维度，它将沿着最后一个维度复制，因此原始张量中的每个条目都变成了一个条目的列表：

![](img/f0243-02_alt.jpg)

因此，当您对一个在该维度有四个条目的张量执行 `.sum(dim=1)` 时，您需要在反向传播时对梯度执行 `.expand(dim=1, copies=4)`。

您现在可以将相应的反向传播逻辑添加到 `.backward()` 方法中：
