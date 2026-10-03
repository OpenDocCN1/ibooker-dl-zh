
```py
ds = L(enumerate(string.ascii_lowercase))
ds
```

```py
(#26) [(0, 'a'),(1, 'b'),(2, 'c'),(3, 'd'),(4, 'e'),(5, 'f'),(6, 'g'),(7,
 > 'h'),(8, 'i'),(9, 'j')...]
```

当我们将`Dataset`传递给`DataLoader`时，我们将得到许多批次，它们本身是表示独立和相关变量批次的张量元组：

```py
dl = DataLoader(ds, batch_size=6, shuffle=True)
list(dl)
```

```py
[(tensor([17, 18, 10, 22,  8, 14]), ('r', 's', 'k', 'w', 'i', 'o')),
 (tensor([20, 15,  9, 13, 21, 12]), ('u', 'p', 'j', 'n', 'v', 'm')),
 (tensor([ 7, 25,  6,  5, 11, 23]), ('h', 'z', 'g', 'f', 'l', 'x')),
 (tensor([ 1,  3,  0, 24, 19, 16]), ('b', 'd', 'a', 'y', 't', 'q')),
 (tensor([2, 4]), ('c', 'e'))]
```

我们现在准备为使用 SGD 的模型编写我们的第一个训练循环！

# 把所有东西放在一起

是时候实现我们在图 4-1 中看到的过程了。在代码中，我们的过程将为每个时期实现类似于这样的东西：

```py
for x,y in dl:
    pred = model(x)
    loss = loss_func(pred, y)
    loss.backward()
    parameters -= parameters.grad * lr
```

首先，让我们重新初始化我们的参数：

```py
weights = init_params((28*28,1))
bias = init_params(1)
```

`DataLoader`可以从`Dataset`创建：

```py
dl = DataLoader(dset, batch_size=256)
xb,yb = first(dl)
xb.shape,yb.shape
```

```py
(torch.Size([256, 784]), torch.Size([256, 1]))
```

我们将对验证集执行相同的操作：

```py
valid_dl = DataLoader(valid_dset, batch_size=256)
```

让我们创建一个大小为 4 的小批量进行测试：

```py
batch = train_x[:4]
batch.shape
```

```py
torch.Size([4, 784])
```

```py
preds = linear1(batch)
preds
```

```py
tensor([[-11.1002],
        [  5.9263],
        [  9.9627],
        [ -8.1484]], grad_fn=<AddBackward0>)
```

```py
loss = mnist_loss(preds, train_y[:4])
loss
```

```py
tensor(0.5006, grad_fn=<MeanBackward0>)
```

现在我们可以计算梯度了：

```py
loss.backward()
weights.grad.shape,weights.grad.mean(),bias.grad
```

```py
(torch.Size([784, 1]), tensor(-0.0001), tensor([-0.0008]))
```

让我们把所有这些放在一个函数中：

```py
def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward()
```

并测试它：

```py
calc_grad(batch, train_y[:4], linear1)
weights.grad.mean(),bias.grad
```

```py
(tensor(-0.0002), tensor([-0.0015]))
```

但是看看如果我们调用两次会发生什么：

```py
calc_grad(batch, train_y[:4], linear1)
weights.grad.mean(),bias.grad
```

```py
(tensor(-0.0003), tensor([-0.0023]))
```

梯度已经改变了！这是因为`loss.backward` *添加*了`loss`的梯度到当前存储的任何梯度中。因此，我们首先必须将当前梯度设置为 0：

```py
weights.grad.zero_()
bias.grad.zero_();
```

# 原地操作

PyTorch 中以下划线结尾的方法会*原地*修改它们的对象。例如，`bias.zero_`会将张量`bias`的所有元素设置为 0。

我们唯一剩下的步骤是根据梯度和学习率更新权重和偏差。当我们这样做时，我们必须告诉 PyTorch 不要对这一步骤进行梯度计算，否则当我们尝试在下一个批次计算导数时会变得混乱！如果我们将张量的`data`属性赋值，PyTorch 将不会对该步骤进行梯度计算。这是我们用于一个时期的基本训练循环：

```py
def train_epoch(model, lr, params):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        for p in params:
            p.data -= p.grad*lr
            p.grad.zero_()
```

我们还想通过查看验证集的准确性来检查我们的表现。要决定输出是否代表 3 或 7，我们只需检查它是否大于 0。因此，我们可以计算每个项目的准确性（使用广播，所以没有循环！）如下：

```py
(preds>0.0).float() == train_y[:4]
```

```py
tensor([[False],
        [ True],
        [ True],
        [False]])
```

这给了我们计算验证准确性的这个函数：

```py
def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean()
```

我们可以检查它是否有效：

```py
batch_accuracy(linear1(batch), train_y[:4])
```

```py
tensor(0.5000)
```

然后把批次放在一起：

```py
def validate_epoch(model):
    accs = [batch_accuracy(model(xb), yb) for xb,yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)
```

```py
validate_epoch(linear1)
```

```py
0.5219
```

这是我们的起点。让我们训练一个时期，看看准确性是否提高：

```py
lr = 1.
params = weights,bias
train_epoch(linear1, lr, params)
validate_epoch(linear1)
```

```py
0.6883
```

然后再做几次：

```py
for i in range(20):
    train_epoch(linear1, lr, params)
    print(validate_epoch(linear1), end=' ')
```

```py
0.8314 0.9017 0.9227 0.9349 0.9438 0.9501 0.9535 0.9564 0.9594 0.9618 0.9613
 > 0.9638 0.9643 0.9652 0.9662 0.9677 0.9687 0.9691 0.9691 0.9696
```

看起来不错！我们的准确性已经接近“像素相似性”方法的准确性，我们已经创建了一个通用的基础可以构建。我们的下一步将是创建一个将处理 SGD 步骤的对象。在 PyTorch 中，它被称为*优化器*。

## 创建一个优化器

因为这是一个如此通用的基础，PyTorch 提供了一些有用的类来使实现更容易。我们可以做的第一件事是用 PyTorch 的`nn.Linear`模块替换我们的`linear`函数。*模块*是从 PyTorch `nn.Module`类继承的类的对象。这个类的对象的行为与标准 Python 函数完全相同，您可以使用括号调用它们，它们将返回模型的激活。

`nn.Linear`做的事情与我们的`init_params`和`linear`一样。它包含了*权重*和*偏差*在一个单独的类中。这是我们如何复制上一节中的模型：

```py
linear_model = nn.Linear(28*28,1)
```

每个 PyTorch 模块都知道它有哪些可以训练的参数；它们可以通过`parameters`方法获得：

```py
w,b = linear_model.parameters()
w.shape,b.shape
```

```py
(torch.Size([1, 784]), torch.Size([1]))
```

我们可以使用这些信息创建一个优化器：

```py
class BasicOptim:
    def __init__(self,params,lr): self.params,self.lr = list(params),lr

    def step(self, *args, **kwargs):
        for p in self.params: p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs):
        for p in self.params: p.grad = None
```

我们可以通过传入模型的参数来创建优化器：

```py
opt = BasicOptim(linear_model.parameters(), lr)
```

我们的训练循环现在可以简化：

```py
def train_epoch(model):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        opt.step()
        opt.zero_grad()
```

我们的验证函数不需要任何更改：

```py
validate_epoch(linear_model)
```

```py
0.4157
```

让我们把我们的小训练循环放在一个函数中，让事情变得更简单：

```py
def train_model(model, epochs):
    for i in range(epochs):
        train_epoch(model)
        print(validate_epoch(model), end=' ')
```

结果与上一节相同：

```py
train_model(linear_model, 20)
```

```py
0.4932 0.8618 0.8203 0.9102 0.9331 0.9468 0.9555 0.9629 0.9658 0.9673 0.9687
 > 0.9707 0.9726 0.9751 0.9761 0.9761 0.9775 0.978 0.9785 0.9785
```

fastai 提供了`SGD`类，默认情况下与我们的`BasicOptim`做相同的事情：

```py
linear_model = nn.Linear(28*28,1)
opt = SGD(linear_model.parameters(), lr)
train_model(linear_model, 20)
```

```py
0.4932 0.852 0.8335 0.9116 0.9326 0.9473 0.9555 0.9624 0.9648 0.9668 0.9692
 > 0.9712 0.9731 0.9746 0.9761 0.9765 0.9775 0.978 0.9785 0.9785
```

fastai 还提供了`Learner.fit`，我们可以使用它来代替`train_model`。要创建一个`Learner`，我们首先需要创建一个`DataLoaders`，通过传入我们的训练和验证`DataLoader`：

```py
dls = DataLoaders(dl, valid_dl)
```

要创建一个`Learner`而不使用应用程序（如`cnn_learner`），我们需要传入本章中创建的所有元素：`DataLoaders`，模型，优化函数（将传递参数），损失函数，以及可选的任何要打印的指标：

```py
learn = Learner(dls, nn.Linear(28*28,1), opt_func=SGD,
                loss_func=mnist_loss, metrics=batch_accuracy)
```

现在我们可以调用`fit`：

```py
learn.fit(10, lr=lr)
```

| epoch | train_loss | valid_loss | batch_accuracy | time |
| --- | --- | --- | --- | --- |
| 0 | 0.636857 | 0.503549 | 0.495584 | 00:00 |
| 1 | 0.545725 | 0.170281 | 0.866045 | 00:00 |
| 2 | 0.199223 | 0.184893 | 0.831207 | 00:00 |
| 3 | 0.086580 | 0.107836 | 0.911187 | 00:00 |
| 4 | 0.045185 | 0.078481 | 0.932777 | 00:00 |
| 5 | 0.029108 | 0.062792 | 0.946516 | 00:00 |
| 6 | 0.022560 | 0.053017 | 0.955348 | 00:00 |
| 7 | 0.019687 | 0.046500 | 0.962218 | 00:00 |
| 8 | 0.018252 | 0.041929 | 0.965162 | 00:00 |
| 9 | 0.017402 | 0.038573 | 0.967615 | 00:00 |

正如您所看到的，PyTorch 和 fastai 类并没有什么神奇之处。它们只是方便的预打包部件，使您的生活变得更轻松！（它们还提供了许多我们将在未来章节中使用的额外功能。）

有了这些类，我们现在可以用神经网络替换我们的线性模型。

# 添加非线性

到目前为止，我们已经有了一个优化函数的一般过程，并且我们已经在一个无聊的函数上尝试了它：一个简单的线性分类器。线性分类器在能做什么方面受到限制。为了使其更复杂一些（并且能够处理更多任务），我们需要在两个线性分类器之间添加一些非线性（即与 ax+b 不同的东西）——这就是给我们神经网络的东西。

这是一个基本神经网络的完整定义：

```py
def simple_net(xb):
    res = xb@w1 + b1
    res = res.max(tensor(0.0))
    res = res@w2 + b2
    return res
```

就是这样！在`simple_net`中，我们只有两个线性分类器，它们之间有一个`max`函数。

在这里，`w1`和`w2`是权重张量，`b1`和`b2`是偏置张量；也就是说，这些参数最初是随机初始化的，就像我们在上一节中所做的一样：

```py
w1 = init_params((28*28,30))
b1 = init_params(30)
w2 = init_params((30,1))
b2 = init_params(1)
```

关键点是`w1`有 30 个输出激活（这意味着`w2`必须有 30 个输入激活，以便匹配）。这意味着第一层可以构建 30 个不同的特征，每个特征代表不同的像素混合。您可以将`30`更改为任何您喜欢的数字，以使模型更复杂或更简单。

那个小函数`res.max(tensor(0.0))`被称为*修正线性单元*，也被称为*ReLU*。我们认为我们都可以同意*修正线性单元*听起来相当花哨和复杂...但实际上，它不过是`res.max(tensor(0.0))`——换句话说，用零替换每个负数。这个微小的函数在 PyTorch 中也可以作为`F.relu`使用：

```py
plot_function(F.relu)
```

![](img/dlcf_04in16.png)

# Jeremy 说

深度学习中有大量行话，包括*修正线性单元*等术语。绝大多数这些行话并不比我们在这个例子中看到的一行代码更复杂。事实是，学术界为了发表论文，他们需要让论文听起来尽可能令人印象深刻和复杂。他们通过引入行话来实现这一点。不幸的是，这导致该领域变得比应该更加令人生畏和难以进入。您确实需要学习这些行话，因为否则论文和教程对您来说将毫无意义。但这并不意味着您必须觉得这些行话令人生畏。只需记住，当您遇到以前未见过的单词或短语时，它几乎肯定是指一个非常简单的概念。
