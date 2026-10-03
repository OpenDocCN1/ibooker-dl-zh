
```py
# In[22]:
%matplotlib inline
from matplotlib import pyplot as plt

t_p = model(t_un, *params)                    # ❶

fig = plt.figure(dpi=600)
plt.xlabel("Temperature (°Fahrenheit)")
plt.ylabel("Temperature (°Celsius)")
plt.plot(t_u.numpy(), t_p.detach().numpy())   # ❷
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
```

❶ 记住我们是在归一化的未知单位上进行训练。我们还使用参数解包。

❷ 但我们正在绘制原始的未知值。

我们在这里使用了一个名为参数解包的 Python 技巧：`*params`意味着将`params`的元素作为单独的参数传递。在 Python 中，这通常是用于列表或元组的，但我们也可以在 PyTorch 张量中使用参数解包，这些张量沿着主导维度分割。因此，在这里，`model(t_un, *params)`等同于`model(t_un, params[0], params[1])`。

此代码生成图 5.9。我们的线性模型似乎是数据的一个很好的模型。看起来我们的测量有些不稳定。我们应该给我们的验光师打电话换一副新眼镜，或者考虑退还我们的高级温度计。

![](img/CH05_F10_Stevens2_GS.png)

图 5.9 我们的线性拟合模型（实线）与输入数据（圆圈）的绘图

## 5.5 PyTorch 的 autograd：反向传播一切

在我们的小冒险中，我们刚刚看到了反向传播的一个简单示例：我们使用*链式法则*向后传播导数，计算了函数组合（模型和损失）相对于它们最内部参数（`w` 和 `b`）的梯度。这里的基本要求是，我们处理的所有函数都可以在解析上进行微分。如果是这种情况，我们可以一次性计算出相对于参数的梯度--我们之前称之为“损失变化率”。

即使我们有一个包含数百万参数的复杂模型，只要我们的模型是可微的，计算相对于参数的损失梯度就相当于编写导数的解析表达式并评估它们*一次*。当然，编写一个非常深层次的线性和非线性函数组合的导数的解析表达式并不是一件有趣的事情。这也不是特别快的过程。

### 5.5.1 自动计算梯度

这就是当 PyTorch 张量发挥作用时的时候，PyTorch 组件 *autograd* 就派上用场了。第三章介绍了张量是什么以及我们可以在它们上调用什么函数的全面概述。然而，我们遗漏了一个非常有趣的方面：PyTorch 张量可以记住它们的来源，即生成它们的操作和父张量，并且可以自动提供这些操作相对于它们的输入的导数链。这意味着我们不需要手动推导我们的模型；给定一个前向表达式，无论多么嵌套，PyTorch 都会自动提供该表达式相对于其输入参数的梯度。

#### 应用 autograd

此时，继续前进的最佳方式是重新编写我们的温度计校准代码，这次使用 autograd，并看看会发生什么。首先，我们回顾一下我们的模型和损失函数。

`code/p1ch5/2_autograd.ipynb`

```py
# In[3]:
def model(t_u, w, b):
    return w * t_u + b

# In[4]:
def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()
```

让我们再次初始化一个参数张量：

```py
# In[5]:
params = torch.tensor([1.0, 0.0], requires_grad=True)
```

#### 使用 grad 属性

注意张量构造函数中的 `requires_grad=True` 参数？该参数告诉 PyTorch 跟踪由于对 `params` 进行操作而产生的张量的整个家族树。换句话说，任何将 `params` 作为祖先的张量都将访问从 `params` 到该张量的链式函数。如果这些函数是可微的（大多数 PyTorch 张量操作都是可微的），导数的值将自动填充为 `params` 张量的 `grad` 属性。

一般来说，所有 PyTorch 张量都有一个名为 `grad` 的属性。通常，它是 `None`：

```py
# In[6]:
params.grad is None

# Out[6]:
True
```

我们只需开始一个 `requires_grad` 设置为 `True` 的张量，然后调用模型并计算损失，然后在 `loss` 张量上调用 `backward`：

```py
# In[7]:
loss = loss_fn(model(t_u, *params), t_c)
loss.backward()

params.grad

# Out[7]:
tensor([4517.2969,   82.6000])
```

此时，`params` 的 `grad` 属性包含了相对于每个元素的 params 的损失的导数。

当我们在参数 `w` 和 `b` 需要梯度时计算我们的 `loss` 时，除了执行实际计算外，PyTorch 还会创建带有操作（黑色圆圈）的 autograd 图，如图 5.10 顶部行所示。当我们调用 `loss.backward()` 时，PyTorch 沿着这个图的反向方向遍历以计算梯度，如图的底部行所示的箭头所示。

![](img/CH05_F11_Stevens2_GS.png)

图 5.10 模型的前向图和后向图，使用 autograd 计算

#### 累积 grad 函数

我们可以有任意数量的张量，其`requires_grad`设置为`True`，以及任意组合的函数。在这种情况下，PyTorch 会计算整个函数链（计算图）中损失的导数，并将其值累积在这些张量的`grad`属性中（图的叶节点）。

警告！*大坑在前方*。这是 PyTorch 新手——以及许多更有经验的人——经常会遇到的问题。我们刚刚写的是*累积*，而不是*存储*。

*警告* 调用`backward`会导致导数在叶节点*累积*。在使用参数更新后，我们需要*显式地将梯度清零*。

让我们一起重复：调用`backward`会导致导数在叶节点*累积*。因此，如果`backward`在之前被调用，损失会再次被评估，`backward`会再次被调用（就像在任何训练循环中一样），并且每个叶节点的梯度会累积（即求和）在上一次迭代计算的梯度之上，这会导致梯度的值不正确。

为了防止这种情况发生，我们需要在每次迭代时*显式地将梯度清零*。我们可以很容易地使用就地`zero_`方法来实现：

```py
# In[8]:
if params.grad is not None:
    params.grad.zero_()
```

*注意* 你可能会好奇为什么清零梯度是一个必需的步骤，而不是在每次调用`backward`时自动清零。这样做提供了更多在处理复杂模型中梯度时的灵活性和控制。

将这个提醒铭记在心，让我们看看我们启用自动求导的训练代码是什么样子，从头到尾：

```py
# In[9]:
def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        if params.grad is not None:                # ❶
            params.grad.zero_()

        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        loss.backward()

        with torch.no_grad():                      # ❷
            params -= learning_rate * params.grad

        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))

    return params
```

❶ 这可以在调用 `loss.backward()` 之前的循环中的任何时候完成。

❷ 这是一段有些繁琐的代码，但正如我们将在下一节看到的，实际上并不是问题。

请注意，我们更新`params`的代码并不像我们可能期望的那样直截了当。有两个特殊之处。首先，我们使用 Python 的`with`语句在`no_grad`上下文中封装更新。这意味着在`with`块内，PyTorch 自动求导机制应该*不要关注*：即，在前向图中不添加边。实际上，当我们执行这段代码时，PyTorch 记录的前向图在我们调用`backward`时被消耗掉，留下`params`叶节点。但现在我们想要在开始构建新的前向图之前更改这个叶节点。虽然这种用例通常包含在我们在第 5.5.2 节中讨论的优化器中，但当我们在第 5.5.4 节看到`no_grad`的另一个常见用法时，我们将更仔细地看一下。

其次，我们就地更新`params`。这意味着我们保留相同的`params`张量，但从中减去我们的更新。在使用自动求导时，我们通常避免就地更新，因为 PyTorch 的自动求导引擎可能需要我们将要修改的值用于反向传播。然而，在这里，我们在没有自动求导的情况下操作，保留`params`张量是有益的。在第 5.5.2 节中向优化器注册参数时，不通过将新张量分配给其变量名来替换参数将变得至关重要。

让我们看看它是否有效：

```py
# In[10]:
training_loop(
    n_epochs = 5000,
    learning_rate = 1e-2,
    params = torch.tensor([1.0, 0.0], requires_grad=True),  # ❶
    t_u = t_un,                                             # ❷
    t_c = t_c)

# Out[10]:
Epoch 500, Loss 7.860116
Epoch 1000, Loss 3.828538
Epoch 1500, Loss 3.092191
Epoch 2000, Loss 2.957697
Epoch 2500, Loss 2.933134
Epoch 3000, Loss 2.928648
Epoch 3500, Loss 2.927830
Epoch 4000, Loss 2.927679
Epoch 4500, Loss 2.927652
Epoch 5000, Loss 2.927647

tensor([  5.3671, -17.3012], requires_grad=True)
```

❶ 添加 `requires_grad=True` 至关重要

❷ 再次，我们使用了标准化的 `t_un` 而不是 `t_u`。

结果与我们之前得到的相同。对我们来说很好！这意味着虽然我们*能够*手动计算导数，但我们不再需要这样做。

### 5.5.2 自选优化器

在示例代码中，我们使用了*普通*梯度下降进行优化，这对我们简单的情况效果很好。不用说，有几种优化策略和技巧可以帮助收敛，特别是在模型变得复杂时。

我们将在后面的章节深入探讨这个主题，但现在是介绍 PyTorch 如何将优化策略从用户代码中抽象出来的正确时机：也就是我们已经检查过的训练循环。这样可以避免我们不得不手动更新模型的每个参数的样板繁琐工作。`torch`模块有一个`optim`子模块，我们可以在其中找到实现不同优化算法的类。这里是一个简略列表（`code/p1ch5/3_optimizers.ipynb`）：

```py
# In[5]:
import torch.optim as optim

dir(optim)

# Out[5]:
['ASGD',
 'Adadelta',
 'Adagrad',
 'Adam',
 'Adamax',
 'LBFGS',
 'Optimizer',
 'RMSprop',
 'Rprop',
 'SGD',
 'SparseAdam',
...
]
```

每个优化器构造函数的第一个输入都是参数列表（也称为 PyTorch 张量，通常将`requires_grad`设置为`True`）。所有传递给优化器的参数都会被保留在优化器对象内部，因此优化器可以更新它们的值并访问它们的`grad`属性，如图 5.11 所示。

![](img/CH05_F12_Stevens2_GS.png)

图 5.11（A）优化器如何保存参数的概念表示。（B）从输入计算损失后，（C）调用`.backward`会使参数上的`.grad`被填充。（D）此时，优化器可以访问`.grad`并计算参数更新。

每个优化器都暴露两个方法：`zero_grad`和`step`。`zero_grad`将在构造时将所有传递给优化器的参数的`grad`属性清零。`step`根据特定优化器实现的优化策略更新这些参数的值。

#### 使用梯度下降优化器

让我们创建`params`并实例化一个梯度下降优化器：

```py
# In[6]:
params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-5
optimizer = optim.SGD([params], lr=learning_rate)
```

这里 SGD 代表*随机梯度下降*。实际上，优化器本身就是一个标准的梯度下降（只要`momentum`参数设置为`0.0`，这是默认值）。术语*随机*来自于梯度通常是通过对所有输入样本的随机子集进行平均得到的，称为*小批量*。然而，优化器不知道损失是在所有样本（标准）上评估的还是在它们的随机子集（随机）上评估的，所以在这两种情况下算法实际上是相同的。

无论如何，让我们尝试一下我们新的优化器：

```py
# In[7]:
t_p = model(t_u, *params)
loss = loss_fn(t_p, t_c)
loss.backward()

optimizer.step()

params

# Out[7]:
tensor([ 9.5483e-01, -8.2600e-04], requires_grad=True)
```

在调用`step`时，`params`的值会被更新，而无需我们自己操作！发生的情况是，优化器查看`params.grad`并更新`params`，从中减去`learning_rate`乘以`grad`，与我们以前手动编写的代码完全相同。

准备将这段代码放入训练循环中？不！几乎让我们犯了大错--我们忘记了将梯度清零。如果我们在循环中调用之前的代码，梯度会在每次调用`backward`时在叶子节点中累积，我们的梯度下降会一团糟！这是循环准备就绪的代码，正确位置是在`backward`调用之前额外加上`zero_grad`：

```py
# In[8]:
params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)

t_p = model(t_un, *params)
loss = loss_fn(t_p, t_c)

optimizer.zero_grad()      # ❶
loss.backward()
optimizer.step()

params

# Out[8]:
tensor([1.7761, 0.1064], requires_grad=True)
```

❶ 与以前一样，这个调用的确切位置有些随意。它也可以在循环中较早的位置。

太棒了！看看`optim`模块如何帮助我们将特定的优化方案抽象出来？我们所要做的就是向其提供一个参数列表（该列表可能非常长，对于非常深的神经网络模型是必需的），然后我们可以忘记细节。

让我们相应地更新我们的训练循环：
