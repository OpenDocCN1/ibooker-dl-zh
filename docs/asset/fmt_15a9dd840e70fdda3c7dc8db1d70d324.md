
```py
class Loss(object):
    '''
 The "loss" of a neural network.
 '''

    def __init__(self):
        '''Pass'''
        pass

    def forward(self, prediction: ndarray, target: ndarray) -> float:
        '''
 Computes the actual loss value.
 '''
        assert_same_shape(prediction, target)

        self.prediction = prediction
        self.target = target

        loss_value = self._output()

        return loss_value

    def backward(self) -> ndarray:
        '''
 Computes gradient of the loss value with respect to the input to the
 loss function.
 '''
        self.input_grad = self._input_grad()

        assert_same_shape(self.prediction, self.input_grad)

        return self.input_grad

    def _output(self) -> float:
        '''
 Every subclass of "Loss" must implement the _output function.
 '''
        raise NotImplementedError()

    def _input_grad(self) -> ndarray:
        '''
 Every subclass of "Loss" must implement the _input_grad function.
 '''
        raise NotImplementedError()
```

与`Operation`类一样，我们检查损失向后发送的梯度与从网络的最后一层接收的`prediction`的形状是否相同：

```py
class MeanSquaredError(Loss):

    def __init__(self)
        '''Pass'''
        super().__init__()

    def _output(self) -> float:
        '''
 Computes the per-observation squared error loss.
 '''
        loss =
            np.sum(np.power(self.prediction - self.target, 2)) /
            self.prediction.shape[0]

        return loss

    def _input_grad(self) -> ndarray:
        '''
 Computes the loss gradient with respect to the input for MSE loss.
 '''

        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]
```

在这里，我们简单地编写均方误差损失公式的前向和反向规则。

这是我们需要从头开始构建深度学习的最后一个关键构建块。让我们回顾一下这些部分如何组合在一起，然后继续构建模型！

# 从零开始的深度学习

我们最终希望构建一个`NeuralNetwork`类，使用图 3-5 作为指南，我们可以用来定义和训练深度学习模型。在我们深入编码之前，让我们准确描述一下这样一个类会是什么样的，以及它将如何与我们刚刚定义的`Operation`、`Layer`和`Loss`类进行交互：

1. `NeuralNetwork`将具有`Layer`列表作为属性。`Layer`将如先前定义的那样，具有`forward`和`backward`方法。这些方法接受`ndarray`对象并返回`ndarray`对象。

2. 每个`Layer`在`_setup_layer`函数期间的`operations`属性中保存了一个`Operation`列表。

3. 这些`Operation`，就像`Layer`本身一样，有`forward`和`backward`方法，接受`ndarray`对象作为参数并返回`ndarray`对象作为输出。

4. 在每个操作中，`backward`方法中接收的`output_grad`的形状必须与`Layer`的`output`属性的形状相同。在`backward`方法期间向后传递的`input_grad`的形状和`input_`属性的形状也是如此。

5. 一些操作具有参数（存储在`param`属性中）；这些操作继承自`ParamOperation`类。`Layer`及其`forward`和`backward`方法的输入和输出形状的约束也适用于它们——它们接收`ndarray`对象并输出`ndarray`对象，`input`和`output`属性及其相应梯度的形状必须匹配。

6. `NeuralNetwork`还将有一个`Loss`。这个类将获取`NeuralNetwork`最后一个操作的输出和目标，检查它们的形状是否相同，并计算损失值（一个数字）和一个`ndarray` `loss_grad`，该`loss_grad`将被馈送到输出层，开始反向传播。

## 实现批量训练

我们已经多次介绍了逐批次训练模型的高级步骤。这些步骤很重要，值得重复：

1. 通过模型函数（“前向传递”）将输入馈送。

2. 计算代表损失的数字。

3. 使用链式法则和在前向传递期间计算的量，计算损失相对于参数的梯度。

4. 使用这些梯度更新参数。

然后我们将通过一批新数据并重复这些步骤。

将这些步骤转换为刚刚描述的`NeuralNetwork`框架是直接的：

1. 接收`X`和`y`作为输入，都是`ndarray`。

2. 逐个将`X`通过每个`Layer`向前传递。

3. 使用`Loss`生成损失值和损失梯度以进行反向传播。

4. 使用损失梯度作为网络`backward`方法的输入，该方法将计算网络中每一层的`param_grads`。

5. 在每一层上调用`update_params`函数，该函数将使用`NeuralNetwork`的整体学习率以及新计算的`param_grads`。

我们最终有了一个完整的神经网络定义，可以进行批量训练。现在让我们编写代码。

## 神经网络：代码

编写所有这些代码非常简单：

```py
class NeuralNetwork(object):
    '''
 The class for a neural network.
 '''
    def __init__(self, layers: List[Layer],
                 loss: Loss,
                 seed: float = 1)
        '''
 Neural networks need layers, and a loss.
 '''
        self.layers = layers
        self.loss = loss
        self.seed = seed
        if seed:
            for layer in self.layers:
                setattr(layer, "seed", self.seed)

    def forward(self, x_batch: ndarray) -> ndarray:
        '''
 Passes data forward through a series of layers.
 '''
        x_out = x_batch
        for layer in self.layers:
            x_out = layer.forward(x_out)

        return x_out

    def backward(self, loss_grad: ndarray) -> None:
        '''
 Passes data backward through a series of layers.
 '''

        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return None

    def train_batch(self,
                    x_batch: ndarray,
                    y_batch: ndarray) -> float:
        '''
 Passes data forward through the layers.
 Computes the loss.
 Passes data backward through the layers.
 '''

        predictions = self.forward(x_batch)

        loss = self.loss.forward(predictions, y_batch)

        self.backward(self.loss.backward())

        return loss

    def params(self):
        '''
 Gets the parameters for the network.
 '''
        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        '''
 Gets the gradient of the loss with respect to the parameters for the
 network.
 '''
        for layer in self.layers:
            yield from layer.param_grads
```

有了这个`NeuralNetwork`类，我们可以以更模块化、灵活的方式实现上一章中的模型，并定义其他模型来表示输入和输出之间的复杂非线性关系。例如，这里是如何轻松实例化我们在上一章中介绍的两个模型——线性回归和神经网络：³

```py
linear_regression = NeuralNetwork(
    layers=[Dense(neurons = 1)],
            loss = MeanSquaredError(),
            learning_rate = 0.01
            )

neural_network = NeuralNetwork(
    layers=[Dense(neurons=13,
                  activation=Sigmoid()),
            Dense(neurons=1,
                  activation=Linear())],
            loss = MeanSquaredError(),
            learning_rate = 0.01
            )
```

基本上我们已经完成了；现在我们只需反复将数据通过网络以便学习。然而，为了使这个过程更清晰、更容易扩展到下一章中将看到的更复杂的深度学习场景，定义另一个类来执行训练以及另一个类来执行“学习”，即根据反向传播计算的梯度实际更新`NeuralNetwork`参数。让我们快速定义这两个类。

# 训练器和优化器

首先，让我们注意这些类与我们在第二章中用于训练网络的代码之间的相似之处。在那里，我们使用以下代码来实现用于训练模型的前述四个步骤：

```py
# pass X_batch forward and compute the loss
forward_info, loss = forward_loss(X_batch, y_batch, weights)

# compute the gradient of the loss with respect to each of the weights
loss_grads = loss_gradients(forward_info, weights)

# update the weights
for key in weights.keys():
    weights[key] -= learning_rate * loss_grads[key]
```

这段代码位于一个`for`循环中，该循环反复将数据通过定义和更新我们的网络的函数。

有了我们现在的类，我们最终将在`Trainer`类中的`fit`函数内部执行这些操作，该函数将主要是对前一章中使用的`train`函数的包装。 （完整的代码在本章的[Jupyter Notebook](https://oreil.ly/2MV0aZI)中的书的 GitHub 页面上。）主要区别是，在这个新函数内部，前面代码块中的前两行将被替换为这一行：

```py
neural_network.train_batch(X_batch, y_batch)
```

更新参数将在以下两行中进行，这将在一个单独的`Optimizer`类中进行。最后，之前包围所有这些内容的`for`循环将在包围`NeuralNetwork`和`Optimizer`的`Trainer`类中进行。

接下来，让我们讨论为什么需要一个`Optimizer`类以及它应该是什么样子。

## 优化器

在上一章描述的模型中，每个`Layer`包含一个简单的规则，根据参数和它们的梯度来更新权重。正如我们将在下一章中提到的，我们可以使用许多其他更新规则，例如涉及梯度更新*历史*而不仅仅是在该迭代中传入的特定批次的梯度更新。创建一个单独的`Optimizer`类将使我们能够灵活地将一个更新规则替换为另一个，这是我们将在下一章中更详细地探讨的内容。

### 描述和代码

基本的`Optimizer`类将接受一个`NeuralNetwork`，每次调用`step`函数时，将根据它们当前的值、梯度和`Optimizer`中存储的任何其他信息来更新网络的参数：

```py
class Optimizer(object):
    '''
 Base class for a neural network optimizer.
 '''
    def __init__(self,
                 lr: float = 0.01):
        '''
 Every optimizer must have an initial learning rate.
 '''
        self.lr = lr

    def step(self) -> None:
        '''
 Every optimizer must implement the "step" function.
 '''
        pass
```

以下是我们迄今为止看到的简单更新规则的实际情况，即*随机梯度下降*：

```py
class SGD(Optimizer):
    '''
 Stochastic gradient descent optimizer.
 '''
    def __init__(self,
                 lr: float = 0.01) -> None:
        '''Pass'''
        super().__init__(lr)

    def step(self):
        '''
 For each parameter, adjust in the appropriate direction, with the
 magnitude of the adjustment based on the learning rate.
 '''
        for (param, param_grad) in zip(self.net.params(),
                                       self.net.param_grads()):

            param -= self.lr * param_grad
```

###### 注意

请注意，虽然我们的`NeuralNetwork`类没有`_update_params`方法，但我们依赖于`params()`和`param_grads()`方法来提取正确的`ndarray`以进行优化。

这就是基本的`Optimizer`类；接下来让我们来看一下`Trainer`类。

## 训练器

除了按照前面描述的方式训练模型外，`Trainer`类还将`NeuralNetwork`与`Optimizer`连接在一起，确保后者正确训练前者。您可能已经注意到，在前一节中，我们在初始化`Optimizer`时没有传入`NeuralNetwork`；相反，我们将在不久后初始化`Trainer`类时将`NeuralNetwork`分配为`Optimizer`的属性，使用以下代码行：

```py
setattr(self.optim, 'net', self.net)
```
