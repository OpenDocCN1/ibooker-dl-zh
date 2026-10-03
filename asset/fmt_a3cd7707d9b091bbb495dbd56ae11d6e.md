
# 模型代码优化与定制

## 初始模型实现

以下是一个简单的线性回归模型，使用 TensorFlow 实现：

```py
class Model:
  def __init__(self):

    # Model
    self.x = tf.placeholder(tf.float32,shape=[None,3])
    self.y_true = tf.placeholder(tf.float32,shape=None)
    self.w = tf.Variable([[0,0,0]],dtype=tf.float32)
    self.b = tf.Variable(0,dtype=tf.float32)

    init = tf.global_variables_initializer()
    self.sess = tf.Session()
    self.sess.run(init)

    self._output = None
    self._optimizer = None
    self._loss = None

  def fit(self,x_data,y_data):
    print(self.b.name)

    for step in range(10):
      self.sess.run(self.optimizer,{self.x: x_data, self.y_true: y_data})
      if (step % 5 == 4) or (step == 0):
        print(step, self.sess.run([self.w,self.b]))

  @property
  def output(self):
    if not self._output:
      y_pred = tf.matmul(self.w,tf.transpose(self.x)) + self.b
      self._output = y_pred
    return self._output

  @property
  def loss(self):
    if not self._loss:
      error = tf.reduce_mean(tf.square(self.y_true-self.output))
      self._loss= error
    return self._loss

  @property
  def optimizer(self):
    if not self._optimizer:
      opt = tf.train.GradientDescentOptimizer(0.5)
      opt = opt.minimize(self.loss)
      self._optimizer = opt
    return self._optimizer

lin_reg = Model()
lin_reg.fit(x_data,y_data)
lin_reg.fit(x_data,y_data)
```

运行上述代码，输出如下：

```py
Variable_89:0
0 [array([[ 0.32110521,  0.4908163 ,  0.09833425]], 
                                       dtype=float32), -0.18784374]
4 [array([[ 0.30250472,  0.49442694,  0.10041162]], 
                                       dtype=float32), -0.1999902]
9 [array([[ 0.30250433,  0.49442688,  0.10041161]], 
                                       dtype=float32), -0.19999036]

Variable_89:0
0 [array([[ 0.30250433,  0.49442688,  0.10041161]], 
                                       dtype=float32), -0.19999038]
4 [array([[ 0.30250433,  0.49442688,  0.10041161]], 
                                       dtype=float32), -0.19999038]
9 [array([[ 0.30250433,  0.49442688,  0.10041161]], 
                                       dtype=float32), -0.19999036]
```

将代码拆分为函数在某种程度上是多余的，因为相同的代码行在每次调用时都会重新计算。一个简单的解决方案是在每个函数的开头添加一个条件。在下一个代码迭代中，我们将看到一个更好的解决方法。

在这种情况下，不需要使用变量共享，因为变量被保留为模型对象的属性。此外，在调用两次训练方法`model.fit()`后，我们看到变量保持了它们的当前状态。

在本节的最后一批代码中，我们添加了另一个增强功能，创建一个自定义装饰器，自动检查函数是否已被调用。

## 改进的模型实现

我们可以做的另一个改进是将所有变量保存在字典中。这将使我们能够在每次操作后跟踪我们的变量，就像我们在第十章中看到的那样，当我们查看保存权重和模型时。

最后，添加了用于获取损失函数值和权重的额外函数：

```py
class Model:
  def __init__(self):

    # Model
    self.x = tf.placeholder(tf.float32,shape=[None,3])
    self.y_true = tf.placeholder(tf.float32,shape=None)

    self.params = self._initialize_weights()

    init = tf.global_variables_initializer()
    self.sess = tf.Session()
    self.sess.run(init)

    self.output
    self.optimizer
    self.loss

  def _initialize_weights(self):
    params = dict()
    params['w'] = tf.Variable([[0,0,0]],dtype=tf.float32)
    params['b'] = tf.Variable(0,dtype=tf.float32)
    return params

  def fit(self,x_data,y_data):
    print(self.params['b'].name)

    for step in range(10):
      self.sess.run(self.optimizer,{self.x: x_data, self.y_true: y_data})
      if (step % 5 == 4) or (step == 0):
        print(step, 
                self.sess.run([self.params['w'],self.params['b']]))

  def evaluate(self,x_data,y_data):
    print(self.params['b'].name)

    MSE = self.sess.run(self.loss,{self.x: x_data, self.y_true: y_data})
    return MSE

  def getWeights(self):
    return self.sess.run([self.params['b']])

  @property_with_check
  def output(self):
    y_pred = tf.matmul(self.params['w'],tf.transpose(self.x)) + \
        self.params['b']
    return y_pred

  @property_with_check
  def loss(self):
    error = tf.reduce_mean(tf.square(self.y_true-self.output))
    return error

  @property_with_check
  def optimizer(self):
    opt = tf.train.GradientDescentOptimizer(0.5)
    opt = opt.minimize(self.loss)
    return opt

lin_reg = Model()
lin_reg.fit(x_data,y_data)
MSE = lin_reg.evaluate(x_data,y_data)
print(MSE)

print(lin_reg.getWeights())
```

以下是输出：

```py
Variable_87:0
0 [array([[ 0.32110521,  0.4908163 ,  0.09833425]], 
                                        dtype=float32), -0.18784374]
4 [array([[ 0.30250472,  0.49442694,  0.10041162]], 
                                        dtype=float32), -0.1999902]
9 [array([[ 0.30250433,  0.49442688,  0.10041161]], 
                                        dtype=float32), -0.19999036]

Variable_87:0
0 [array([[ 0.30250433,  0.49442688,  0.10041161]], 
                                        dtype=float32), -0.19999038]
4 [array([[ 0.30250433,  0.49442688,  0.10041161]], 
                                        dtype=float32), -0.19999038]
9 [array([[ 0.30250433,  0.49442688,  0.10041161]], 
                                        dtype=float32), -0.19999036]
Variable_87:0
0.0102189
[-0.19999036] 
```

自定义装饰器检查属性是否存在，如果不存在，则根据输入函数设置它。否则，返回属性。使用`functools.wrap()`，这样我们就可以引用函数的名称：

```py
import functools

def property_with_check(input_fn):
  attribute = '_cache_' + input_fn.__name__

  @property
  @functools.wraps(input_fn)
  def check_attr(self):
    if not hasattr(self, attribute):
      setattr(self, attribute, input_fn(self))
    return getattr(self, attribute)

  return check_attr
```

这是一个相当基本的示例，展示了我们如何改进模型的整体代码。这种优化可能对我们简单的线性回归示例来说有些过度，但对于具有大量层、变量和特征的复杂模型来说，这绝对是值得的努力。

## 定制

到目前为止，我们使用了两个损失函数。在第二章中的分类示例中，我们使用了交叉熵损失，定义如下：

```py
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
```

相比之下，在前一节的回归示例中，我们使用了平方误差损失，定义如下：

```py
loss = tf.reduce_mean(tf.square(y_true-y_pred))
```

这些是目前在机器学习和深度学习中最常用的损失函数。本节的目的是双重的。首先，我们想指出 TensorFlow 在利用自定义损失函数方面的更一般能力。其次，我们将讨论正则化作为任何损失函数的扩展形式，以实现特定目标，而不考虑使用的基本损失函数。

### 自制损失函数

本书（以及我们的读者）以深度学习为重点来看待 TensorFlow。然而，TensorFlow 的范围更广泛，大多数机器学习问题都可以以一种 TensorFlow 可以解决的方式来表述。此外，任何可以在计算图框架中表述的计算都是从 TensorFlow 中受益的好候选。

主要特例是无约束优化问题类。这些问题在科学（和算法）计算中非常常见，对于这些问题，TensorFlow 尤为有用。这些问题突出的原因是，TensorFlow 提供了计算梯度的自动机制，这为解决这类问题的开发时间提供了巨大的加速。

一般来说，对于任意损失函数的优化将采用以下形式：

```py
def my_loss_function(key-variables...):
loss = ...
return loss

my_loss = my_loss_function(key-variables...)
gd_step = tf.train.GradientDescentOptimizer().minimize(my_loss)
```

任何优化器都可以用于替代`GradientDescentOptimizer`。

### 正则化

正则化是通过对解决方案的复杂性施加惩罚来限制优化问题（有关更多详细信息，请参见第四章中的注释）。在本节中，我们将看一下特定情况下，惩罚直接添加到基本损失函数中的附加形式。

例如，基于第二章中 softmax 示例，我们有以下内容：

```py
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))

y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)

cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))

total_loss = cross_entropy + LAMBDA * tf.nn.l2_loss(W)

gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(total_loss)
```

与第二章中的原始版本的区别在于，我们将`LAMBDA * tf.nn.l2_loss(W)`添加到我们正在优化的损失中。在这种情况下，使用较小的权衡参数`LAMBDA`值对最终准确性的影响很小（较大的值会有害）。在大型网络中，过拟合是一个严重问题，这种正则化通常可以拯救一命。

这种类型的正则化可以针对模型的权重进行，如前面的示例所示（也称为*权重衰减*，因为它会使权重值变小），也可以针对特定层或所有层的激活进行。

另一个因素是我们使用的函数——我们可以使用`l1`而不是`l2`正则化，或者两者的组合。所有这些正则化器的组合都是有效的，并在各种情境中使用。

许多抽象层使正则化的应用变得简单，只需指定滤波器数量或激活函数即可。例如，在 Keras（在第七章中审查的非常流行的扩展）中，我们提供了适用于所有标准层的正则化器，列在表 A-1 中。

**表 A-1. 使用 Keras 进行正则化**

| 正则化器 | 作用 | 示例 |
| --- | --- | --- |
| `l1` | `l1` 正则化权重 | `Dense(100, W_regularizer=l1(0.01))` |
| `l2` | `l2` 正则化权重 | `Dense(100, W_regularizer=l2(0.01))` |
| `l1l2` | 组合 `l1 + l2` 正则化权重 | `Dense(100, W_regularizer=l1l2(0.01))` |
| `activity_l1` | `l1` 正则化激活 | `Dense(100, activity_regularizer=activity_l1(0.01))` |
| `activity_l2` | `l2` 正则化激活 | `Dense(100, activity_regularizer=activity_l2(0.01))` |
| `activity_l1l2` | 组合 `l1 + l2` 正则化激活 | `Dense(100, activity_regularizer=activity_l1l2(0.01))` |

在模型过拟合时，使用这些快捷方式可以轻松测试不同的正则化方案。

### 编写自己的操作

TensorFlow 预装了大量本地操作，从标准算术和逻辑操作到矩阵操作、深度学习特定函数等等。当这些操作不够时，可以通过创建新操作来扩展系统。有两种方法可以实现这一点：

+   编写一个“从头开始”的 C++ 版本的操作
