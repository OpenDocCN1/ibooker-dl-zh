
`Model`类是`Layer`类的子类，因此模型可以像层一样定义和使用。但是模型具有一些额外的功能，包括当然包括`compile()`、`fit()`、`evaluate()`和`predict()`方法（以及一些变体），还有`get_layer()`方法（可以通过名称或索引返回模型的任何层）和`save()`方法（以及对`tf.keras.models.load_model()`和`tf.keras.models.clone_model()`的支持）。

###### 提示

如果模型提供的功能比层更多，为什么不将每个层都定义为模型呢？技术上您可以这样做，但通常更清晰的做法是区分模型的内部组件（即层或可重用的层块）和模型本身（即您将训练的对象）。前者应该是`Layer`类的子类，而后者应该是`Model`类的子类。

有了这些，您可以自然而简洁地构建几乎任何您在论文中找到的模型，使用顺序 API、函数 API、子类 API，甚至这些的混合。“几乎”任何模型？是的，还有一些事情我们需要看一下：首先是如何基于模型内部定义损失或指标，其次是如何构建自定义训练循环。

## 基于模型内部的损失和指标

我们之前定义的自定义损失和指标都是基于标签和预测（以及可选的样本权重）。有时您可能希望基于模型的其他部分（例如其隐藏层的权重或激活）定义损失。这可能对正则化目的或监视模型的某些内部方面很有用。

要基于模型内部定义自定义损失，可以根据模型的任何部分计算损失，然后将结果传递给`add_loss()`方法。例如，让我们构建一个由五个隐藏层堆叠加一个输出层组成的自定义回归 MLP 模型。这个自定义模型还将在最上面的隐藏层之上具有一个辅助输出。与这个辅助输出相关联的损失将被称为*重建损失*（参见第十七章）：它是重建和输入之间的均方差差异。通过将这个重建损失添加到主要损失中，我们将鼓励模型通过隐藏层尽可能保留更多信息，即使这些信息对于回归任务本身并不直接有用。在实践中，这种损失有时会改善泛化能力（它是一种正则化损失）。还可以使用模型的`add_metric()`方法添加自定义指标。以下是具有自定义重建损失和相应指标的自定义模型的代码：

```py
class ReconstructingRegressor(tf.keras.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [tf.keras.layers.Dense(30, activation="relu",
                                             kernel_initializer="he_normal")
                       for _ in range(5)]
        self.out = tf.keras.layers.Dense(output_dim)
        self.reconstruction_mean = tf.keras.metrics.Mean(
            name="reconstruction_error")

    def build(self, batch_input_shape):
        n_inputs = batch_input_shape[-1]
        self.reconstruct = tf.keras.layers.Dense(n_inputs)

    def call(self, inputs, training=False):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        reconstruction = self.reconstruct(Z)
        recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))
        self.add_loss(0.05 * recon_loss)
        if training:
            result = self.reconstruction_mean(recon_loss)
            self.add_metric(result)
        return self.out(Z)
```

让我们来看一下这段代码：

+   构造函数创建了一个具有五个密集隐藏层和一个密集输出层的 DNN。我们还创建了一个`Mean`流式指标，用于在训练过程中跟踪重建误差。

+   `build()`方法创建一个额外的密集层，用于重构模型的输入。它必须在这里创建，因为其单元数必须等于输入的数量，在调用`build()`方法之前这个数量是未知的。

+   `call()`方法通过所有五个隐藏层处理输入，然后将结果传递给重构层，该层生成重构。

+   然后`call()`方法计算重构损失（重构和输入之间的均方差），并使用`add_loss()`方法将其添加到模型的损失列表中。请注意，我们通过将重构损失乘以 0.05 来缩小重构损失（这是一个可以调整的超参数）。这确保了重构损失不会主导主要损失。

+   接下来，在训练过程中，`call()`方法更新重构度量并将其添加到模型中以便显示。这段代码示例实际上可以通过调用`self.add_metric(recon_loss)`来简化：Keras 将自动为您跟踪均值。

+   最后，`call()`方法将隐藏层的输出传递给输出层，并返回其输出。

在训练过程中，总损失和重构损失都会下降：

```py
Epoch 1/5
363/363 [========] - 1s 820us/step - loss: 0.7640 - reconstruction_error: 1.2728
Epoch 2/5
363/363 [========] - 0s 809us/step - loss: 0.4584 - reconstruction_error: 0.6340
[...]
```

在大多数情况下，到目前为止我们讨论的一切将足以实现您想构建的任何模型，即使是具有复杂架构、损失和指标。然而，对于一些架构，如 GANs（参见第十七章），您将不得不自定义训练循环本身。在我们到达那里之前，我们必须看看如何在 TensorFlow 中自动计算梯度。

## 使用自动微分计算梯度

要了解如何使用自动微分（参见第十章和附录 B）自动计算梯度，让我们考虑一个简单的玩具函数：

```py
def f(w1, w2):
    return 3 * w1 ** 2 + 2 * w1 * w2
```

如果你懂微积分，你可以分析地找到这个函数相对于`w1`的偏导数是`6 * w1` `+` `2 * w2`。你也可以找到它相对于`w2`的偏导数是`2 * w1`。例如，在点`(w1, w2)` `=` `(5, 3)`，这些偏导数分别等于 36 和 10，因此在这一点的梯度向量是（36，10）。但如果这是一个神经网络，这个函数会复杂得多，通常有数万个参数，通过手工分析找到偏导数将是一个几乎不可能的任务。一个解决方案是通过测量当你微调相应参数一点点时函数的输出如何变化来计算每个偏导数的近似值：

```py
>>> w1, w2 = 5, 3
>>> eps = 1e-6
>>> (f(w1 + eps, w2) - f(w1, w2)) / eps
36.000003007075065
>>> (f(w1, w2 + eps) - f(w1, w2)) / eps
10.000000003174137
```

看起来不错！这个方法运行得相当好，而且易于实现，但它只是一个近似值，重要的是你需要至少针对每个参数调用一次`f()`（不是两次，因为我们可以只计算一次`f(w1, w2)`）。每个参数至少调用一次`f()`使得这种方法在大型神经网络中变得难以处理。因此，我们应该使用反向模式自动微分。TensorFlow 使这变得非常简单：

```py
w1, w2 = tf.Variable(5.), tf.Variable(3.)
with tf.GradientTape() as tape:
    z = f(w1, w2)

gradients = tape.gradient(z, [w1, w2])
```

首先我们定义两个变量`w1`和`w2`，然后我们创建一个`tf.GradientTape`上下文，它将自动记录涉及变量的每个操作，最后我们要求这个磁带计算结果`z`相对于两个变量`[w1, w2]`的梯度。让我们看看 TensorFlow 计算的梯度：

```py
>>> gradients
[<tf.Tensor: shape=(), dtype=float32, numpy=36.0>,
 <tf.Tensor: shape=(), dtype=float32, numpy=10.0>]
```

太棒了！结果不仅准确（精度仅受浮点误差限制），而且`gradient()`方法只需通过记录的计算一次（按相反顺序），无论有多少变量，因此非常高效。就像魔术一样！

###### 提示

为了节省内存，在`tf.GradientTape()`块中只放入严格的最小值。或者，通过在`tf.GradientTape()`块内创建一个`with tape.stop_recording()`块来暂停记录。

在调用其`gradient()`方法后，磁带会立即被擦除，因此如果尝试两次调用`gradient()`，将会收到异常：

```py
with tf.GradientTape() as tape:
    z = f(w1, w2)

dz_dw1 = tape.gradient(z, w1)  # returns tensor 36.0
dz_dw2 = tape.gradient(z, w2)  # raises a RuntimeError!
```

如果您需要多次调用`gradient()`，您必须使磁带持久化，并在每次完成后删除它以释放资源：

```py
with tf.GradientTape(persistent=True) as tape:
    z = f(w1, w2)

dz_dw1 = tape.gradient(z, w1)  # returns tensor 36.0
dz_dw2 = tape.gradient(z, w2)  # returns tensor 10.0, works fine now!
del tape
```

默认情况下，磁带只会跟踪涉及变量的操作，因此，如果您尝试计算`z`相对于除变量以外的任何东西的梯度，结果将是`None`：

```py
c1, c2 = tf.constant(5.), tf.constant(3.)
with tf.GradientTape() as tape:
    z = f(c1, c2)

gradients = tape.gradient(z, [c1, c2])  # returns [None, None]
```

但是，您可以强制磁带监视任何您喜欢的张量，记录涉及它们的每个操作。然后，您可以计算相对于这些张量的梯度，就像它们是变量一样：

```py
with tf.GradientTape() as tape:
    tape.watch(c1)
    tape.watch(c2)
    z = f(c1, c2)

gradients = tape.gradient(z, [c1, c2])  # returns [tensor 36., tensor 10.]
```

在某些情况下，这可能很有用，比如如果您想要实现一个正则化损失，惩罚激活在输入变化很小时变化很大的情况：损失将基于激活相对于输入的梯度。由于输入不是变量，您需要告诉磁带监视它们。

大多数情况下，梯度磁带用于计算单个值（通常是损失）相对于一组值（通常是模型参数）的梯度。这就是反向模式自动微分的优势所在，因为它只需要进行一次前向传递和一次反向传递就可以一次性获得所有梯度。如果尝试计算向量的梯度，例如包含多个损失的向量，那么 TensorFlow 将计算向量总和的梯度。因此，如果您需要获取各个梯度（例如，每个损失相对于模型参数的梯度），您必须调用磁带的`jacobian()`方法：它将为向量中的每个损失执行一次反向模式自动微分（默认情况下全部并行）。甚至可以计算二阶偏导数（Hessians，即偏导数的偏导数），但在实践中很少需要（请参阅本章笔记本的“使用自动微分计算梯度”部分以获取示例）。

在某些情况下，您可能希望阻止梯度通过神经网络的某些部分进行反向传播。为此，您必须使用`tf.stop_gradient()`函数。该函数在前向传递期间返回其输入（类似于`tf.identity()`），但在反向传播期间不允许梯度通过（它的作用类似于常数）：

```py
def f(w1, w2):
    return 3 * w1 ** 2 + tf.stop_gradient(2 * w1 * w2)

with tf.GradientTape() as tape:
    z = f(w1, w2)  # the forward pass is not affected by stop_gradient()

gradients = tape.gradient(z, [w1, w2])  # returns [tensor 30., None]
```

最后，当计算梯度时，您可能偶尔会遇到一些数值问题。例如，如果在*x*=10^（-50）处计算平方根函数的梯度，结果将是无穷大。实际上，该点的斜率并不是无穷大，但它超过了 32 位浮点数的处理能力：

```py
>>> x = tf.Variable(1e-50)
>>> with tf.GradientTape() as tape:
...     z = tf.sqrt(x)
...
>>> tape.gradient(z, [x])
[<tf.Tensor: shape=(), dtype=float32, numpy=inf>]
```

为了解决这个问题，在计算平方根时，通常建议向*x*（例如 10^（-6））添加一个微小值。

指数函数也经常引起头痛，因为它增长非常快。例如，之前定义的`my_softplus()`的方式在数值上不稳定。如果计算`my_softplus(100.0)`，您将得到无穷大而不是正确的结果（约为 100）。但是可以重写该函数以使其在数值上稳定：softplus 函数被定义为 log(1 + exp(*z*))，这也等于 log(1 + exp(–|*z*|)) + max(*z*, 0)（请参阅数学证明的笔记本），第二种形式的优势在于指数项不会爆炸。因此，这是`my_softplus()`函数的更好实现：

```py
def my_softplus(z):
    return tf.math.log(1 + tf.exp(-tf.abs(z))) + tf.maximum(0., z)
```

在一些罕见的情况下，一个数值稳定的函数可能仍然具有数值不稳定的梯度。在这种情况下，你将不得不告诉 TensorFlow 使用哪个方程来计算梯度，而不是让它使用自动微分。为此，你必须在定义函数时使用`@tf.​cus⁠tom_gradient`装饰器，并返回函数的通常结果以及计算梯度的函数。例如，让我们更新`my_softplus()`函数，使其也返回一个数值稳定的梯度函数：

```py
@tf.custom_gradient
def my_softplus(z):
    def my_softplus_gradients(grads):  # grads = backprop'ed from upper layers
        return grads * (1 - 1 / (1 + tf.exp(z)))  # stable grads of softplus

    result = tf.math.log(1 + tf.exp(-tf.abs(z))) + tf.maximum(0., z)
    return result, my_softplus_gradients
```

如果你懂微积分（参见关于这个主题的教程笔记本），你会发现 log(1 + exp(*z*))的导数是 exp(*z*) / (1 + exp(*z*))。但这种形式是不稳定的：对于较大的*z*值，它最终会计算出无穷大除以无穷大，返回 NaN。然而，通过一点代数操作，你可以证明它也等于 1 - 1 / (1 + exp(*z*))，这是稳定的。`my_softplus_gradients()`函数使用这个方程来计算梯度。请注意，这个函数将接收到目前为止反向传播的梯度，一直到`my_softplus()`函数，并根据链式法则，我们必须将它们与这个函数的梯度相乘。

现在当我们计算`my_softplus()`函数的梯度时，即使对于较大的输入值，我们也会得到正确的结果。

恭喜！现在你可以计算任何函数的梯度（只要在计算时它是可微的），甚至在需要时阻止反向传播，并编写自己的梯度函数！这可能比你需要的灵活性更多，即使你构建自己的自定义训练循环。接下来你将看到如何做到这一点。

## 自定义训练循环

在某些情况下，`fit()`方法可能不够灵活以满足你的需求。例如，我们在第十章中讨论的[Wide & Deep 论文](https://homl.info/widedeep)使用了两种不同的优化器：一种用于宽路径，另一种用于深路径。由于`fit()`方法只使用一个优化器（在编译模型时指定的那个），实现这篇论文需要编写自己的自定义循环。

你可能也喜欢编写自定义训练循环，只是为了更有信心地确保它们确实按照你的意图执行（也许你对`fit()`方法的一些细节不确定）。有时候，让一切都显式化可能会感觉更安全。然而，请记住，编写自定义训练循环会使你的代码变得更长、更容易出错，并且更难维护。

###### 提示

除非你在学习或确实需要额外的灵活性，否则应该优先使用`fit()`方法而不是实现自己的训练循环，特别是如果你在团队中工作。

首先，让我们构建一个简单的模型。不需要编译它，因为我们将手动处理训练循环：
