
# 第十六章：训练过程

> 原文：[`www.bookstack.cn/read/th-fastai-book/76501b0f60767008.md`](https://www.bookstack.cn/read/th-fastai-book/76501b0f60767008.md)
>
> 译者：[飞龙](https://github.com/wizardforcel)
>
> 协议：[CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/)

现在你知道如何为计算机视觉、自然图像处理、表格分析和协同过滤创建最先进的架构，也知道如何快速训练它们。所以我们完成了，对吧？还没有。我们仍然需要探索一下训练过程的更多内容。

我们在第四章中解释了随机梯度下降的基础：将一个小批量数据传递给模型，用损失函数将其与目标进行比较，然后计算这个损失函数对每个权重的梯度，然后使用公式更新权重：

```py
new_weight = weight - lr * weight.grad
```

我们在训练循环中从头开始实现了这个，看到 PyTorch 提供了一个简单的`nn.SGD`类，可以为我们的每个参数进行这个计算。在本章中，我们将构建一些更快的优化器，使用一个灵活的基础。但在训练过程中，我们可能还想要改变一些东西。对于训练循环的任何调整，我们都需要一种方法来向 SGD 的基础添加一些代码。fastai 库有一个回调系统来做到这一点，我们将教你所有相关知识。

让我们从标准的 SGD 开始建立一个基线；然后我们将介绍最常用的优化器。

## 建立基线

首先，我们将使用普通的 SGD 创建一个基线，并将其与 fastai 的默认优化器进行比较。我们将通过使用与第十四章中相同的`get_data`来获取 Imagenette：

```py
dls = get_data(URLs.IMAGENETTE_160, 160, 128)
```

我们将创建一个没有预训练的 ResNet-34，并传递任何接收到的参数：

```py
def get_learner(**kwargs):
    return cnn_learner(dls, resnet34, pretrained=False,
                    metrics=accuracy, **kwargs).to_fp16()
```

这是默认的 fastai 优化器，具有通常的 3e-3 学习率：

```py
learn = get_learner()
learn.fit_one_cycle(3, 0.003)
```

| epoch | train_loss | valid_loss | accuracy | time |
| --- | --- | --- | --- | --- |
| 0 | 2.571932 | 2.685040 | 0.322548 | 00:11 |
| 1 | 1.904674 | 1.852589 | 0.437452 | 00:11 |
| 2 | 1.586909 | 1.374908 | 0.594904 | 00:11 |

现在让我们尝试普通的 SGD。我们可以将`opt_func`（优化函数）传递给`cnn_learner`，以便让 fastai 使用任何优化器：

```py
learn = get_learner(opt_func=SGD)
```

首先要看的是`lr_find`：

```py
learn.lr_find()
```

```py
(0.017378008365631102, 3.019951861915615e-07)
```

![](img/dlcf_16in01.png)

看起来我们需要使用比我们通常使用的更高的学习率：

```py
learn.fit_one_cycle(3, 0.03, moms=(0,0,0))
```

| epoch | train_loss | valid_loss | accuracy | time |
| --- | --- | --- | --- | --- |
| 0 | 2.969412 | 2.214596 | 0.242038 | 00:09 |
| 1 | 2.442730 | 1.845950 | 0.362548 | 00:09 |
| 2 | 2.157159 | 1.741143 | 0.408917 | 00:09 |

因为用动量加速 SGD 是一个很好的主意，fastai 在`fit_one_cycle`中默认执行这个操作，所以我们用`moms=(0,0,0)`关闭它。我们很快会讨论动量。

显然，普通的 SGD 训练速度不如我们所希望的快。所以让我们学习一些技巧来加速训练！

## 通用优化器

为了构建我们加速的 SGD 技巧，我们需要从一个灵活的优化器基础开始。在 fastai 之前没有任何库提供这样的基础，但在 fastai 的开发过程中，我们意识到学术文献中看到的所有优化器改进都可以使用*优化器回调*来处理。这些是我们可以组合、混合和匹配在优化器中构建优化器步骤的小代码片段。它们由 fastai 的轻量级`Optimizer`类调用。这些是我们在本书中使用的两个关键方法在`Optimizer`中的定义：

```py
def zero_grad(self):
    for p,*_ in self.all_params():
        p.grad.detach_()
        p.grad.zero_()

def step(self):
    for p,pg,state,hyper in self.all_params():
        for cb in self.cbs:
            state = _update(state, cb(p, **{**state, **hyper}))
        self.state[p] = state
```

正如我们在从头开始训练 MNIST 模型时看到的，`zero_grad`只是循环遍历模型的参数并将梯度设置为零。它还调用`detach_`，这会删除任何梯度计算的历史，因为在`zero_grad`之后不再需要它。

更有趣的方法是`step`，它循环遍历回调（`cbs`）并调用它们来更新参数（如果`cb`返回任何内容，`_update`函数只是调用`state.update`）。正如你所看到的，`Optimizer`本身不执行任何 SGD 步骤。让我们看看如何将 SGD 添加到`Optimizer`中。

这是一个优化器回调，通过将`-lr`乘以梯度并将其添加到参数（当在 PyTorch 中传递`Tensor.add_`两个参数时，它们在相加之前相乘）来执行单个 SGD 步骤：

```py
def sgd_cb(p, lr, **kwargs): p.data.add_(-lr, p.grad.data)
```

我们可以使用`cbs`参数将这个传递给`Optimizer`；我们需要使用`partial`，因为`Learner`将调用这个函数来创建我们的优化器：

```py
opt_func = partial(Optimizer, cbs=[sgd_cb])
```

让我们看看这是否有效：

```py
learn = get_learner(opt_func=opt_func)
learn.fit(3, 0.03)
```

| epoch | train_loss | valid_loss | accuracy | time |
| --- | --- | --- | --- | --- |
| 0 | 2.730918 | 2.009971 | 0.332739 | 00:09 |
| 1 | 2.204893 | 1.747202 | 0.441529 | 00:09 |
| 2 | 1.875621 | 1.684515 | 0.445350 | 00:09 |

它正在工作！这就是我们如何在 fastai 中从头开始创建 SGD。现在让我们看看这个“动量”是什么。

## 动量

正如在第四章中所描述的，SGD 可以被看作站在山顶上，通过在每个时间点沿着最陡峭的斜坡方向迈出一步来往下走。但如果我们有一个球在山上滚动呢？在每个给定点，它不会完全按照梯度的方向前进，因为它会有*动量*。具有更多动量的球（例如，一个更重的球）会跳过小凸起和洞，更有可能到达崎岖山脉的底部。另一方面，乒乓球会卡在每一个小缝隙中。

那么我们如何将这个想法带到 SGD 中呢？我们可以使用移动平均值，而不仅仅是当前梯度，来进行我们的步骤：

```py
weight.avg = beta * weight.avg + (1-beta) * weight.grad
new_weight = weight - lr * weight.avg
```

这里`beta`是我们选择的一个数字，定义了要使用多少动量。如果`beta`为 0，第一个方程变为`weight.avg = weight.grad`，因此我们最终得到普通的 SGD。但如果它接近 1，所选择的主要方向是以前步骤的平均值。（如果你对统计学有一点了解，你可能会在第一个方程中认出*指数加权移动平均*，它经常用于去噪数据并获得潜在趋势。）

请注意，我们写`weight.avg`以突出显示我们需要为模型的每个参数存储移动平均值（它们都有自己独立的移动平均值）。

图 16-1 显示了一个单参数的噪声数据示例，其中动量曲线以红色绘制，参数的梯度以蓝色绘制。梯度增加，然后减少，动量很好地跟随总体趋势，而不会受到噪声的太大影响。

![显示动量示例的图表](img/dlcf_1601.png)

###### 图 16-1。动量的一个例子

如果损失函数有窄谷，我们需要导航：普通的 SGD 会使我们从一边反弹到另一边，而带有动量的 SGD 会将这些平均值平滑地滚动到一侧。参数`beta`确定我们使用的动量的强度：使用较小的`beta`，我们会保持接近实际梯度值，而使用较高的`beta`，我们将主要朝着梯度的平均值前进，直到梯度的任何变化使得该趋势移动。

使用较大的`beta`，我们可能会错过梯度改变方向并滚动到一个小的局部最小值。这是一个期望的副作用：直观地，当我们向模型展示一个新的输入时，它会看起来像训练集中的某个东西，但不会*完全*像它。它将对应于损失函数中接近我们在训练结束时得到的最小值的点，但不会*在*那个最小值。因此，我们宁愿在一个宽阔的最小值中进行训练，附近的点具有近似相同的损失（或者如果你喜欢的话，损失尽可能平坦的点）。图 16-2 显示了当我们改变`beta`时，图 16-1 中的图表如何变化。

![显示 beta 值如何影响动量的图表](img/dlcf_1602.png)

###### 图 16-2。不同 beta 值的动量

我们可以看到在这些示例中，`beta`太高会导致梯度的整体变化被忽略。在带动量的 SGD 中，通常使用的`beta`值为 0.9。

`fit_one_cycle`默认从 0.95 开始，逐渐调整到 0.85，然后在训练结束时逐渐移回到 0.95。让我们看看在普通 SGD 中添加动量后我们的训练情况如何。

要向我们的优化器添加动量，我们首先需要跟踪移动平均梯度，我们可以使用另一个回调来实现。当优化器回调返回一个`dict`时，它用于更新优化器的状态，并在下一步传回优化器。因此，这个回调将跟踪梯度平均值，存储在名为`grad_avg`的参数中：

```py
def average_grad(p, mom, grad_avg=None, **kwargs):
    if grad_avg is None: grad_avg = torch.zeros_like(p.grad.data)
    return {'grad_avg': grad_avg*mom + p.grad.data}
```

要使用它，我们只需在我们的步骤函数中用`grad_avg`替换`p.grad.data`：

```py
def momentum_step(p, lr, grad_avg, **kwargs): p.data.add_(-lr, grad_avg)
```

```py
opt_func = partial(Optimizer, cbs=[average_grad,momentum_step], mom=0.9)
```

`Learner`将自动调度`mom`和`lr`，因此`fit_one_cycle`甚至可以与我们自定义的`Optimizer`一起使用：

```py
learn = get_learner(opt_func=opt_func)
learn.fit_one_cycle(3, 0.03)
```

| epoch | train_loss | valid_loss | accuracy | time |
| --- | --- | --- | --- | --- |
| 0 | 2.856000 | 2.493429 | 0.246115 | 00:10 |
| 1 | 2.504205 | 2.463813 | 0.348280 | 00:10 |
| 2 | 2.187387 | 1.755670 | 0.418853 | 00:10 |

```py
learn.recorder.plot_sched()
```

![](img/dlcf_16in02.png)

我们仍然没有得到很好的结果，所以让我们看看我们还能做什么。

## RMSProp

RMSProp 是由 Geoffrey Hinton 在[他的 Coursera 课程“神经网络机器学习”第 6 讲 e](https://oreil.ly/FVcIE)中介绍的 SGD 的另一种变体。与 SGD 的主要区别在于它使用自适应学习率：每个参数都有自己特定的学习率，由全局学习率控制。这样，我们可以通过为需要大幅度改变的权重提供更高的学习率来加速训练，而对于已经足够好的权重，则提供较低的学习率。

我们如何决定哪些参数应该具有较高的学习率，哪些不应该？我们可以查看梯度来获取一个想法。如果一个参数的梯度一直接近于零，那么该参数将需要更高的学习率，因为损失是平的。另一方面，如果梯度到处都是，我们可能应该小心并选择一个较低的学习率以避免发散。我们不能简单地平均梯度来查看它们是否变化很多，因为大正数和大负数的平均值接近于零。相反，我们可以使用通常的技巧，即取绝对值或平方值（然后在平均后取平方根）。

再次，为了确定噪声背后的一般趋势，我们将使用移动平均值，具体来说是梯度的平方的移动平均值。然后，我们将通过使用当前梯度（用于方向）除以这个移动平均值的平方根来更新相应的权重（这样，如果它很低，有效的学习率将更高，如果它很高，有效的学习率将更低）：

```py
w.square_avg = alpha * w.square_avg + (1-alpha) * (w.grad ** 2)
new_w = w - lr * w.grad / math.sqrt(w.square_avg + eps)
```

`eps`（*epsilon*）是为了数值稳定性而添加的（通常设置为 1e-8），`alpha`的默认值通常为 0.99。

我们可以通过做与`avg_grad`类似的事情将其添加到`Optimizer`中，但是多了一个`**2`：

```py
def average_sqr_grad(p, sqr_mom, sqr_avg=None, **kwargs):
    if sqr_avg is None: sqr_avg = torch.zeros_like(p.grad.data)
    return {'sqr_avg': sqr_avg*sqr_mom + p.grad.data**2}
```

我们可以像以前一样定义我们的步骤函数和优化器：

```py
def rms_prop_step(p, lr, sqr_avg, eps, grad_avg=None, **kwargs):
    denom = sqr_avg.sqrt().add_(eps)
    p.data.addcdiv_(-lr, p.grad, denom)

opt_func = partial(Optimizer, cbs=[average_sqr_grad,rms_prop_step],
                   sqr_mom=0.99, eps=1e-7)
```

让我们试一试：

```py
learn = get_learner(opt_func=opt_func)
learn.fit_one_cycle(3, 0.003)
```

| epoch | train_loss | valid_loss | accuracy | time |
| --- | --- | --- | --- | --- |
| 0 | 2.766912 | 1.845900 | 0.402548 | 00:11 |
| 1 | 2.194586 | 1.510269 | 0.504459 | 00:11 |
| 2 | 1.869099 | 1.447939 | 0.544968 | 00:11 |

好多了！现在我们只需将这些想法结合起来，我们就有了 Adam，fastai 的默认优化器。

## Adam

Adam 将 SGD 与动量和 RMSProp 的思想结合在一起：它使用梯度的移动平均作为方向，并除以梯度平方的移动平均的平方根，为每个参数提供自适应学习率。

Adam 计算移动平均值的方式还有一个不同之处。它采用*无偏*移动平均值，即

```py
w.avg = beta * w.avg + (1-beta) * w.grad
unbias_avg = w.avg / (1 - (beta**(i+1)))
```

如果我们是第`i`次迭代（从 0 开始，就像 Python 一样）。这个除数`1 - (beta**(i+1))`确保无偏平均值看起来更像开始时的梯度（因为`beta < 1`，分母很快接近 1）。

将所有内容放在一起，我们的更新步骤看起来像这样：
