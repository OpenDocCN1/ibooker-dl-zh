
```py
class BiasAdd(ParamOperation):
    '''
 Compute bias addition.
 '''

    def __init__(self,
                 B: ndarray):
        '''
 Initialize Operation with self.param = B.
 Check appropriate shape.
 '''
        assert B.shape[0] == 1

        super().__init__(B)

    def _output(self) -> ndarray:
        '''
 Compute output.
 '''
        return self.input_ + self.param

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
 Compute input gradient.
 '''
        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        '''
 Compute parameter gradient.
 '''
        param_grad = np.ones_like(self.param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])
```

最后，让我们做`sigmoid`：

```py
class Sigmoid(Operation):
    '''
 Sigmoid activation function.
 '''

    def __init__(self) -> None:
        '''Pass'''
        super().__init__()

    def _output(self) -> ndarray:
        '''
 Compute output.
 '''
        return 1.0/(1.0+np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
 Compute input gradient.
 '''
        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad
```

这只是实现了前一章描述的数学。

###### 注意

对于`sigmoid`和`ParamOperation`，在反向传播期间计算的步骤是：

```py
input_grad = <something> * output_grad
```

是我们应用链规则的步骤，以及`WeightMultiply`的相应规则：

```py
np.dot(output_grad, np.transpose(self.param, (1, 0)))
```

正如我在第一章中所说的，当涉及的函数是矩阵乘法时，这相当于链规则的类比。

现在我们已经准确定义了这些`Operation`，我们可以将它们用作定义`Layer`的构建块。

## 层蓝图

由于我们编写了`Operation`的方式，编写`Layer`类很容易：

+   `forward`和`backward`方法只涉及将输入依次通过一系列`Operation`向前传递 - 就像我们一直在图表中所做的那样！这是关于`Layer`工作最重要的事实；代码的其余部分是围绕这一点的包装，并且主要涉及簿记：

    +   在`_setup_layer`函数中定义正确的`Operation`系列，并在这些`Operation`中初始化和存储参数（这也将在`_setup_layer`函数中进行）

    +   在`forward`方法中存储正确的值在`self.input_`和`self.output`中

    +   在`backward`方法中执行正确的断言检查

+   最后，`_params`和`_param_grads`函数只是从层内的`ParamOperation`中提取参数及其梯度（相对于损失）。

所有这些看起来是这样的：

```py
class Layer(object):
    '''
 A "layer" of neurons in a neural network.
 '''

 def __init__(self,
                 neurons: int):
        '''
 The number of "neurons" roughly corresponds to the "breadth" of the
 layer
 '''
        self.neurons = neurons
        self.first = True
        self.params: List[ndarray] = []
        self.param_grads: List[ndarray] = []
        self.operations: List[Operation] = []

    def _setup_layer(self, num_in: int) -> None:
        '''
 The _setup_layer function must be implemented for each layer.
 '''
        raise NotImplementedError()

    def forward(self, input_: ndarray) -> ndarray:
        '''
 Passes input forward through a series of operations.
 '''
        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_

        for operation in self.operations:

            input_ = operation.forward(input_)

        self.output = input_

        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        '''
 Passes output_grad backward through a series of operations.
 Checks appropriate shapes.
 '''

        assert_same_shape(self.output, output_grad)

        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)

        input_grad = output_grad

        self._param_grads()

        return input_grad

    def _param_grads(self) -> ndarray:
        '''
 Extracts the _param_grads from a layer's operations.
 '''

        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self) -> ndarray:
        '''
 Extracts the _params from a layer's operations.
 '''

        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)
```

就像我们从抽象定义的`Operation`转向实现神经网络所需的特定`Operation`一样，让我们现在也实现该网络中的`Layer`。

## 密集层

我们称我们一直在处理的`Operation`为`WeightMultiply`，`BiasAdd`等等。到目前为止我们一直在使用的层应该叫什么？`LinearNonLinear`层？

这一层的一个定义特征是*每个输出神经元都是所有输入神经元的函数*。这就是矩阵乘法真正做的事情：如果矩阵是<math><msub><mi>n</mi> <mrow><mi>i</mi><mi>n</mi></mrow></msub></math>行乘以<math><msub><mi>n</mi> <mrow><mi>o</mi><mi>u</mi><mi>t</mi></mrow></msub></math>列，那么乘法本身计算的是<math><msub><mi>n</mi> <mrow><mi>o</mi><mi>u</mi><mi>t</mi></mrow></msub></math>个新特征，每个特征都是*所有*<math><msub><mi>n</mi> <mrow><mi>i</mi><mi>n</mi></mrow></msub></math>个输入特征的加权线性组合。² 因此，这些层通常被称为*全连接*层；最近，在流行的`Keras`库中，它们也经常被称为`Dense`层，这是一个更简洁的术语，传达了相同的概念。

既然我们知道该如何称呼它以及为什么，让我们根据我们已经定义的操作来定义`Dense`层 - 正如您将看到的，由于我们如何定义了我们的`Layer`基类，我们所需要做的就是在`_setup_layer`函数中将前一节中定义的`Operation`作为列表放入其中。

```py
class Dense(Layer):
    '''
 A fully connected layer that inherits from "Layer."
 '''
    def __init__(self,
                 neurons: int,
                 activation: Operation = Sigmoid()) -> None:
        '''
 Requires an activation function upon initialization.
 '''
        super().__init__(neurons)
        self.activation = activation

    def _setup_layer(self, input_: ndarray) -> None:
        '''
 Defines the operations of a fully connected layer.
 '''
        if self.seed:
            np.random.seed(self.seed)

        self.params = []

        # weights
        self.params.append(np.random.randn(input_.shape[1], self.neurons))

        # bias
        self.params.append(np.random.randn(1, self.neurons))

        self.operations = [WeightMultiply(self.params[0]),
                           BiasAdd(self.params[1]),
                           self.activation]

        return None
```

请注意，我们将默认激活函数设置为`Linear`激活函数，这实际上意味着我们不应用激活函数，只是将恒等函数应用于层的输出。

在`Operation`和`Layer`之上，我们现在应该添加哪些构建块？为了训练我们的模型，我们知道我们将需要一个`NeuralNetwork`类来包装`Layer`，就像`Layer`包装`Operation`一样。不明显需要哪些其他类，所以我们将直接着手构建`NeuralNetwork`，并在进行过程中找出我们需要的其他类。

# 神经网络类，也许还有其他类

我们的`NeuralNetwork`类应该能做什么？在高层次上，它应该能够*从数据中学习*：更准确地说，它应该能够接收代表“观察”（`X`）和“正确答案”（`y`）的数据批次，并学习`X`和`y`之间的关系，这意味着学习一个能够将`X`转换为非常接近`y`的预测`p`的函数。

鉴于刚刚定义的`Layer`和`Operation`类，这种学习将如何进行？回顾上一章的模型是如何工作的，我们将实现以下内容：

1.  神经网络应该接受`X`并将其逐步通过每个`Layer`（实际上是一个方便的包装器，用于通过许多`Operation`进行馈送），此时结果将代表`prediction`。

1.  接下来，应该将`prediction`与值`y`进行比较，计算损失并生成“损失梯度”，这是与网络中最后一个层（即生成`prediction`的层）中的每个元素相关的损失的偏导数。

1.  最后，我们将通过每个层将这个损失梯度逐步向后发送，同时计算“参数梯度”——损失对每个参数的偏导数，并将它们存储在相应的`Operation`中。

## 图

图 3-5 以`Layer`的术语捕捉了神经网络的描述。

![神经网络图](img/dlfs_0305.png)

###### 图 3-5。反向传播，现在以 Layer 而不是 Operation 的术语

## 代码

我们应该如何实现这一点？首先，我们希望我们的神经网络最终处理`Layer`的方式与我们的`Layer`处理`Operation`的方式相同。例如，我们希望`forward`方法接收`X`作为输入，然后简单地执行类似以下的操作：

```py
for layer in self.layers:
    X = layer.forward(X)

return X
```

同样，我们希望我们的`backward`方法接收一个参数——我们最初称之为`grad`——然后执行类似以下的操作：

```py
for layer in reversed(self.layers):
    grad = layer.backward(grad)
```

`grad`将从哪里来？它必须来自*损失*，一个特殊的函数，它接收`prediction`以及`y`，然后：

+   计算代表网络进行该`prediction`的“惩罚”的单个数字。

+   针对每个`prediction`中的元素，发送一个梯度与损失相关的反向梯度。这个梯度是网络中最后一个`Layer`将作为其`backward`函数输入接收的内容。

在前一章的示例中，损失函数是`prediction`和目标之间的平方差，相应地计算了`prediction`相对于损失的梯度。

我们应该如何实现这一点？这个概念似乎很重要，值得拥有自己的类。此外，这个类可以类似于`Layer`类实现，只是`forward`方法将产生一个实际数字（一个`float`）作为损失，而不是一个`ndarray`被发送到下一个`Layer`。让我们正式化这一点。

## 损失类

`Loss`基类将类似于`Layer`——`forward`和`backward`方法将检查适当的`ndarray`的形状是否相同，并定义两个方法，`_output`和`_input_grad`，任何`Loss`子类都必须定义：
