
```py
def _setup_decay(self) -> None:

    if not self.decay_type:
        return
    elif self.decay_type == 'exponential':
        self.decay_per_epoch = np.power(self.final_lr / self.lr,
                                   1.0 / (self.max_epochs-1))
    elif self.decay_type == 'linear':
        self.decay_per_epoch = (self.lr - self.final_lr) / (self.max_epochs-1)
```

然后，在每个时期结束时，我们将实际衰减学习率：

```py
def _decay_lr(self) -> None:

    if not self.decay_type:
        return

    if self.decay_type == 'exponential':
        self.lr *= self.decay_per_epoch

    elif self.decay_type == 'linear':
        self.lr -= self.decay_per_epoch
```

最后，在`fit`函数中，我们将在每个时期结束时从`Trainer`中调用`_decay_lr`函数：

```py
if self.optim.final_lr:
    self.optim._decay_lr()
```

让我们进行一些实验，看看这是否有助于改善训练。

## 实验：学习率衰减

接下来，我们尝试使用学习率衰减训练相同的模型架构。我们初始化学习率，使得整个运行过程中的“平均学习率”等于之前的学习率 0.1：对于线性学习率衰减的运行，我们将学习率初始化为 0.15，衰减到 0.05；对于指数衰减的运行，我们将学习率初始化为 0.2，衰减到 0.05。对于线性衰减运行：

```py
optimizer = SGDMomentum(0.15, momentum=0.9, final_lr=0.05, decay_type='linear')
```

我们有：

```py
Validation loss after 10 epochs is 0.403
Validation loss after 20 epochs is 0.343
Validation loss after 30 epochs is 0.282
Loss increased after epoch 40, final loss was 0.282, using the model from epoch 30
The model validation accuracy is: 95.91%
```

对于指数衰减的运行，有：

```py
optimizer = SGDMomentum(0.2, momentum=0.9, final_lr=0.05, decay_type='exponential')
```

我们有：

```py
Validation loss after 10 epochs is 0.461
Validation loss after 20 epochs is 0.323
Validation loss after 30 epochs is 0.284
Loss increased after epoch 40, final loss was 0.284, using the model from epoch 30
The model validation accuracy is: 96.06%
```

这些运行中“最佳模型”的损失分别为 0.282 和 0.284，明显低于之前的 0.338！

接下来：如何以及为什么更智能地初始化模型的权重。

# 权重初始化

正如我们在激活函数部分提到的，一些激活函数，如 sigmoid 和 Tanh，在其输入为 0 时具有最陡的梯度，随着输入远离 0，函数迅速变平。这可能会限制这些函数的有效性，因为如果许多输入的值远离 0，那么附加到这些输入的权重将在反向传播中接收到非常小的梯度。

这事实证明是我们现在处理的神经网络中的一个主要问题。考虑我们一直在研究的 MNIST 网络中的隐藏层。这一层将接收 784 个输入，然后将它们乘以一个权重矩阵，最终得到一些数量为`n`的神经元（然后可选择性地为每个神经元添加偏置）。图 4-8 显示了我们神经网络的隐藏层（具有 784 个输入）中这些`n`个值在通过`Tanh`激活函数之前和之后的分布。

![激活输入的分布](img/dlfs_0408.png)

###### 图 4-8. 激活函数和激活的输入分布

通过激活函数后，大多数激活要么是-1，要么是 1！这是因为每个特征在数学上被定义为：

<math display="block"><mrow><msub><mi>f</mi> <mi>n</mi></msub> <mo>=</mo> <msub><mi>w</mi> <mrow><mn>1</mn><mo>,</mo><mi>n</mi></mrow></msub> <mo>×</mo> <msub><mi>x</mi> <mn>1</mn></msub> <mo>+</mo> <mo>...</mo> <mo>+</mo> <msub><mi>w</mi> <mrow><mn>784</mn><mo>,</mo><mi>n</mi></mrow></msub> <mo>×</mo> <msub><mi>x</mi> <mn>784</mn></msub> <mo>+</mo> <msub><mi>b</mi> <mi>n</mi></msub></mrow></math>

由于我们将每个权重的方差初始化为 1（<math><mrow><mtext>Var</mtext> <mo>(</mo> <msub><mi>w</mi> <mrow><mi>i</mi><mo>,</mo><mi>j</mi></mrow></msub> <mo>)</mo> <mo>=</mo> <mn>1</mn></mrow></math> —和 <math><mrow><mtext>Var</mtext> <mo>(</mo> <msub><mi>b</mi> <mi>n</mi></msub> <mo>)</mo> <mo>=</mo> <mn>1</mn></mrow></math> ）以及<math><mrow><mtext>Var</mtext> <mrow><mo>(</mo> <msub><mi>X</mi> <mn>1</mn></msub> <mo>+</mo> <msub><mi>X</mi> <mn>2</mn></msub> <mo>)</mo></mrow> <mo>=</mo> <mtext>Var</mtext> <mrow><mo>(</mo> <msub><mi>X</mi> <mn>1</mn></msub> <mo>)</mo></mrow> <mo>+</mo> <mtext>Var</mtext> <mrow><mo>(</mo> <msub><mi>X</mi> <mn>2</mn></msub> <mo>)</mo></mrow></mrow></math> 对于独立的随机变量*X*[1]和*X*[2]，我们有：

<math display="block"><mrow><mtext>Var</mtext> <mo>(</mo> <msub><mi>f</mi> <mi>n</mi></msub> <mo>)</mo> <mo>=</mo> <mn>785</mn></mrow></math>

这使其标准差（<math><msqrt><mn>785</mn></msqrt></math>）略高于 28，这反映了我们在图 4-8 顶部看到的值的分布。

这告诉我们我们有一个问题。但问题是否仅仅是我们输入激活函数的特征不能“太分散”？如果是这个问题，我们可以简单地将特征除以某个值来减少它们的方差。然而，这引发了一个明显的问题：我们如何知道要将值除以多少？答案是，值应该根据输入到层中的神经元数量进行缩放。如果我们有一个多层神经网络，其中一层有 200 个神经元，下一层有 100 个，那么 200 个神经元层将传递前向值，其分布比 100 个神经元层更广。这是不希望的——我们不希望我们神经网络在训练期间学习的特征的*尺度*取决于向前传递的特征数量，原因与我们不希望我们网络的预测取决于输入特征的尺度相同。例如，如果我们将特征中的所有值乘以 2 或除以 2，我们的模型的预测不应受影响。

有几种方法可以纠正这个问题；这里我们将介绍最常见的一种，由前一段建议：我们可以*根据它们连接的层中的神经元数量调整权重的初始方差，以便在前向传递期间传递给下一层的值*和*在反向传递期间传递给前一层的值*大致具有相同的尺度。是的，我们也必须考虑反向传递，因为同样的问题也存在：层在反向传播期间接收的梯度的方差将直接取决于*下一个*层中的特征数量，因为这是将梯度向后发送到相关层的层。

## 数学和代码

那么，我们如何平衡这些问题？如果每一层有<math><msub><mi>n</mi> <mrow><mi>i</mi><mi>n</mi></mrow></msub></math>个神经元输入和<math><msub><mi>n</mi> <mrow><mi>o</mi><mi>u</mi><mi>t</mi></mrow></msub></math>个神经元输出，为了保持结果特征的方差在*前向传递*中恒定，每个权重的方差应该是：

<math display="block"><mfrac><mn>1</mn> <msub><mi>n</mi> <mrow><mi>i</mi><mi>n</mi></mrow></msub></mfrac></math>

同样，保持特征方差在向后传递过程中恒定的权重方差将是：

<math display="block"><mfrac><mn>1</mn> <msub><mi>n</mi> <mrow><mi>o</mi><mi>u</mi><mi>t</mi></mrow></msub></mfrac></math>

作为这些之间的折衷，最常被称为*Glorot 初始化*⁷涉及将每一层中权重的方差初始化为：

<math display="block"><mfrac><mn>2</mn> <mrow><msub><mi>n</mi> <mrow><mi>i</mi><mi>n</mi></mrow></msub> <mo>+</mo><msub><mi>n</mi> <mrow><mi>o</mi><mi>u</mi><mi>t</mi></mrow></msub></mrow></mfrac></math>

编码这很简单 - 我们为每一层添加一个`weight_init`参数，并将以下内容添加到我们的`_setup_layer`函数中：

```py
if self.weight_init == "glorot":
    scale = 2/(num_in + self.neurons)
else:
    scale = 1.0
```

现在我们的模型将如下所示：

```py
model = NeuralNetwork(
    layers=[Dense(neurons=89,
                  activation=Tanh(),
                  weight_init="glorot"),
            Dense(neurons=10,
                  activation=Linear(),
                  weight_init="glorot")],
            loss = SoftmaxCrossEntropy(),
seed=20190119)
```

对于每一层指定`weight_init="glorot"`。

## 实验：权重初始化

使用 Glorot 初始化运行与上一节相同的模型给出：

```py
Validation loss after 10 epochs is 0.352
Validation loss after 20 epochs is 0.280
Validation loss after 30 epochs is 0.244
Loss increased after epoch 40, final loss was 0.244, using the model from epoch 30
The model validation accuracy is: 96.71%
```

对于具有线性学习率衰减的模型，以及：

```py
Validation loss after 10 epochs is 0.305
Validation loss after 20 epochs is 0.264
Validation loss after 30 epochs is 0.245
Loss increased after epoch 40, final loss was 0.245, using the model from epoch 30
The model validation accuracy is: 96.71%
```

对于具有指数学习率衰减的模型。我们在这里看到损失再次显著下降，从我们之前实现的 0.282 和 0.284 下降到 0.244 和 0.245！请注意，通过所有这些更改，*我们并没有增加模型的大小或训练时间*；我们只是根据我们对神经网络尝试做的事情的直觉调整了训练过程，这是我在本章开头展示的。

在本章中，我们将介绍最后一种技术。作为动机，您可能已经注意到*本章中使用的所有模型都不是深度学习模型*；相反，它们只是具有一个隐藏层的神经网络。这是因为没有*dropout*，我们现在将学习的技术，深度学习模型很难有效训练而不会过拟合。

# Dropout

在本章中，我展示了几种修改我们神经网络训练过程的方法，使其越来越接近全局最小值。您可能已经注意到，我们还没有尝试看似最明显的事情：向我们的网络添加更多层或每层更多的神经元。原因是简单地向大多数神经网络架构添加更多“火力”可能会使网络更难以找到很好泛化的解决方案。这里的直觉是，虽然向神经网络添加更多容量使其能够模拟输入和输出之间更复杂的关系，但也有可能导致网络过度拟合训练数据的解决方案。*dropout 允许我们向神经网络添加容量，同时在大多数情况下减少网络过拟合的可能性*。

具体来说*dropout*是什么？

## 定义

Dropout 只是在训练的每个前向传递中随机选择一层中的一定比例*p*的神经元，并将它们设置为 0。这种奇怪的技巧降低了网络的容量，但在许多情况下，经验上确实可以防止网络过拟合。这在更深层次的网络中尤其如此，因为所学习的特征构造上是多层抽象远离原始特征。

尽管 dropout 可以帮助我们的网络在训练过程中避免过拟合，但当预测时，我们仍然希望为我们的网络提供“最佳机会”来进行正确的预测。因此，`Dropout`操作将有两种“模式”：一个“训练”模式，在该模式下应用 dropout，以及一个“推断”模式，在该模式下不应用 dropout。然而，这会导致另一个问题：将 dropout 应用于一层会平均减少传递的值的总体幅度，即如果以下层中的权重通常期望具有幅度*M*的值，则实际上得到的幅度为<math><mrow><mi>M</mi> <mo>×</mo> <mo>(</mo> <mn>1</mn> <mo>-</mo> <mi>p</mi> <mo>)</mo></mrow></math>。当在推断模式下运行网络时，我们希望模拟这种幅度变化，因此，除了删除 dropout，我们将乘以*所有*值乘以 1 - *p*。

为了更清楚，让我们编写代码。

## 实施

我们可以将 dropout 实现为一个`Operation`，我们将其附加到每一层的末尾。它将如下所示：

```py
class Dropout(Operation):

    def __init__(self,
                 keep_prob: float = 0.8):
        super().__init__()
        self.keep_prob = keep_prob

    def _output(self, inference: bool) -> ndarray:
        if inference:
            return self.inputs * self.keep_prob
        else:
            self.mask = np.random.binomial(1, self.keep_prob,
                                           size=self.inputs.shape)
            return self.inputs * self.mask

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        return output_grad * self.mask
```

在前向传播时，应用 dropout 时，我们保存一个代表被设置为 0 的单个神经元的“掩码”。然后，在反向传播时，我们将操作接收到的梯度乘以这个掩码。这是因为 dropout 使得输入值为 0 的梯度为 0（因为改变它们的值现在对损失没有影响），而其他梯度保持不变。

### 调整我们的框架的其余部分以适应 dropout

您可能已经注意到，在`_output`方法中我们包含了一个影响是否应用 dropout 的`inference`标志。为了正确调用此标志，我们实际上必须在整个训练过程中的几个其他地方添加它：

1.  `Layer`和`NeuralNetwork`的`forward`方法将接受`inference`作为参数（默认为`False`），并将该标志传递到每个`Operation`中，以便每个`Operation`在训练模式和推理模式下表现不同。
