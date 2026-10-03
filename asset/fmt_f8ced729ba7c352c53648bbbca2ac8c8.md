
```py
v.assign(2 * v)           # => [[2., 4., 6.], [8., 10., 12.]]
v[0, 1].assign(42)        # => [[2., 42., 6.], [8., 10., 12.]]
v[:, 2].assign([0., 1.])  # => [[2., 42., 0.], [8., 10., 1.]]
v.scatter_nd_update(indices=[[0, 0], [1, 2]], updates=[100., 200.])
                          # => [[100., 42., 0.], [8., 10., 200.]]
```

> 笔记：在实践中，很少需要手动创建变量，因为 Keras 有`add_weight()`方法可以自动来做。另外，模型参数通常会直接通过优化器更新，因此很少需要手动更新。

### 其它数据结构

TensorFlow 还支持其它几种数据结构，如下（可以参考笔记本的`Tensors and Operations`部分，或附录的 F）：

稀疏张量（`tf.SparseTensor`）

高效表示含有许多 0 的张量。`tf.sparse`包含有对稀疏张量的运算。

张量数组（`tf.TensorArray`）

是张量的列表。有默认固定大小，但也可以做成动态的。列表中的张量必须形状相同，数据类型也相同。

嵌套张量（`tf.RaggedTensor`）

张量列表的静态列表，张量的形状和数据结构相同。`tf.ragged`包里有嵌套张量的运算。

字符串张量

类型是`tf.string`的常规张量，是字节串而不是 Unicode 字符串，因此如果你用 Unicode 字符串（比如，Python3 字符串`café`）创建了一个字符串张量，就会自动被转换为 UTF-8（`b"caf\xc3\xa9"`）。另外，也可以用`tf.int32`类型的张量表示 Unicode 字符串，其中每项表示一个 Unicode 码（比如，`[99, 97, 102, 233]`）。`tf.strings`包里有字节串和 Unicode 字符串的运算，以及二者转换的运算。要注意`tf.string`是原子性的，也就是说它的长度不出现在张量的形状中，一旦将其转换成了 Unicode 张量（即，含有 Unicode 码的`tf.int32`张量），长度才出现在形状中。

集合

表示为常规张量（或稀疏张量）。例如`tf.constant([[1, 2], [3, 4]])`表示两个集合{1, 2}和{3, 4}。通常，用张量的最后一个轴的向量表示集合。集合运算可以用`tf.sets`包。

队列

用来在多个步骤之间保存张量。TensorFlow 提供了多种队列。先进先出（FIFO）队列`FIFOQueue`，优先级队列`PriorityQueue`，随机队列`RandomShuffleQueue`，通过填充的不同形状的批次项队列`PaddingFIFOQueue`。这些队列都在`tf.queue`包中。

有了张量、运算、变量和各种数据结构，就可以开始自定义模型和训练算法啦！

## 自定义模型和训练算法

先从简单又常见的任务开始，创建一个自定义的损失函数。

### 自定义损失函数

假如你想训练一个回归模型，但训练集有噪音。你当然可以通过清除或修正异常值来清理数据集，但是这样还不够：数据集还是有噪音。此时，该用什么损失函数呢？均方差可能对大误差惩罚过重，导致模型不准确。均绝对值误差不会对异常值惩罚过重，但训练可能要比较长的时间才能收敛，训练模型也可能不准确。此时使用 Huber 损失（第 10 章介绍过）就比 MSE 好多了。目前官方 Keras API 中没有 Huber 损失，但`tf.keras`有（使用类`keras.losses.Huber`的实例）。就算`tf.keras`没有，实现也不难！只需创建一个函数，参数是标签和预测值，使用 TensorFlow 运算计算每个实例的损失：

```py
def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss  = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)
```

> 警告：要提高性能，应该像这个例子使用向量。另外，如果想利用 TensorFlow 的图特性，则只能使用 TensorFlow 运算。

最好返回一个包含实例的张量，其中每个实例都有一个损失，而不是返回平均损失。这么做的话，Keras 可以在需要时，使用类权重或样本权重（见第 10 章）。

现在，编译 Keras 模型时，就可以使用 Huber 损失来训练了：

```py
model.compile(loss=huber_fn, optimizer="nadam")
model.fit(X_train, y_train, [...])
```

仅此而已！对于训练中的每个批次，Keras 会调用函数`huber_fn()`计算损失，用损失来做梯度下降。另外，Keras 会从一开始跟踪总损失，并展示平均损失。

在保存这个模型时，这个自定义损失会发生什么呢？

### 保存并加载包含自定义组件的模型

因为 Keras 可以保存函数名，保存含有自定义损失函数的模型也不成问题。当加载模型时，你需要提供一个字典，这个字典可以将函数名和真正的函数映射起来。一般说来，当加载一个含有自定义对象的模型时，你需要将名字映射到对象上：

```py
model = keras.models.load_model("my_model_with_a_custom_loss.h5",
                                custom_objects={"huber_fn": huber_fn})
```

对于刚刚的代码，在 -1 和 1 之间的误差被认为是“小”误差。如果要改变阈值呢？一个解决方法是创建一个函数，它可以产生一个可配置的损失函数：

```py
def create_huber(threshold=1.0):
    def huber_fn(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < threshold
        squared_loss = tf.square(error) / 2
        linear_loss  = threshold * tf.abs(error) - threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    return huber_fn
model.compile(loss=create_huber(2.0), optimizer="nadam")
```

但在保存模型时，`threshold`不能被保存。这意味在加载模型时（注意，给 Keras 的函数名是`Huber_fn`，不是创造这个函数的函数名），必须要指定`threshold`的值：

```py
model = keras.models.load_model("my_model_with_a_custom_loss_threshold_2.h5",
                                custom_objects={"huber_fn": create_huber(2.0)})
```

要解决这个问题，可以创建一个`keras.losses.Loss`类的子类，然后实现`get_config()`方法：

```py
class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss  = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}
```

> 警告：Keras API 目前只使用子类来定义层、模型、调回和正则器。如果使用子类创建其它组件（比如损失、指标、初始化器或约束），它们不能迁移到其它 Keras 实现上。可能 Keras API 经过更新，就会支持所有组件了。

逐行看下这段代码：

*   构造器接收`**kwargs`，并将其传递给父构造器，父构造器负责处理超参数：损失的`name`，要使用的、用于将单个实例的损失汇总的`reduction`算法。默认情况下是`"sum_over_batch_size"`，意思是损失是各个实例的损失之和，如果有样本权重，则做权重加权，再除以批次大小（不是除以权重之和，所以不是加权平均）。其它可能的值是`"sum"`和`None`。

*   `call()`方法接受标签和预测值，计算所有实例的损失，并返回。

*   `get_config()`方法返回一个字典，将每个超参数映射到值上。它首先调用父类的`get_config()`方法，然后将新的超参数加入字典（`{**x}`语法是 Python 3.5 引入的）。

当编译模型时，可以使用这个类的实例：

```py
model.compile(loss=HuberLoss(2.), optimizer="nadam")
```

保存模型时，阈值会一起保存；加载模型时，只需将类名映射到具体的类上：

```py
model = keras.models.load_model("my_model_with_a_custom_loss_class.h5",
                                custom_objects={"HuberLoss": HuberLoss})
```

保存模型时，Keras 调用损失实例的`get_config()`方法，将配置以 JSON 的形式保存在 HDF5 中。当加载模型时，会调用`HuberLoss`类的`from_config()`方法：这个方法是父类`Loss`实现的，创建一个类`Loss`的实例，将`**config`传递给构造器。

### 自定义激活函数、初始化器、正则器和约束

Keras 的大多数功能，比如损失、正则器、约束、初始化器、指标、激活函数、层，甚至是完整的模型，都可以用相似的方法做自定义。大多数时候，需要写一个简单的函数，带有合适的输入和输出。下面的例子是自定义激活函数（等价于`keras.activations.softplus()`或`tf.nn.softplus()`），自定义 Glorot 初始化器（等价于`keras.initializers.glorot_normal()`），自定义`ℓ1`正则化器（等价于`keras.regularizers.l1(0.01)`），可以保证权重都是正值的自定义约束（等价于`equivalent to keras.constraints.nonneg()`或`tf.nn.relu()`）：

```py
def my_softplus(z): # return value is just tf.nn.softplus(z)
    return tf.math.log(tf.exp(z) + 1.0)

def my_glorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2\. / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)

def my_l1_regularizer(weights):
    return tf.reduce_sum(tf.abs(0.01 * weights))

def my_positive_weights(weights): # return value is just tf.nn.relu(weights)
    return tf.where(weights < 0., tf.zeros_like(weights), weights)
```

可以看到，参数取决于自定义函数的类型。这些自定义函数可以如常使用，例如：

```py
layer = keras.layers.Dense(30, activation=my_softplus,
                           kernel_initializer=my_glorot_initializer,
                           kernel_regularizer=my_l1_regularizer,
                           kernel_constraint=my_positive_weights)
```

激活函数会应用到这个`Dense`层的输出上，结果会传递到下一层。层的权重会使用初始化器的返回值。在每个训练步骤，权重会传递给正则化函数以计算正则损失，这个损失会与主损失相加，得到训练的最终损失。最后，会在每个训练步骤结束后调用约束函数，经过约束的权重会替换层的权重。

如果函数有需要连同模型一起保存的超参数，需要对相应的类做子类，比如`keras.regularizers.Regularizer`，`keras.constraints.Constraint`，`keras.initializers.Initializer`，或 `keras.layers.Layer`（任意层，包括激活函数）。就像前面的自定义损失一样，下面是一个简单的`ℓ1`正则类，可以保存它的超参数`factor`（这次不必调用其父构造器或`get_config()`方法，因为它们不是父类定义的）：

```py
class MyL1Regularizer(keras.regularizers.Regularizer):
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, weights):
        return tf.reduce_sum(tf.abs(self.factor * weights))
    def get_config(self):
        return {"factor": self.factor}
```

注意，你必须要实现损失、层（包括激活函数）和模型的`call()`方法，或正则化器、初始化器和约束的`__call__()`方法。对于指标，处理方法有所不同。

### 自定义指标

损失和指标的概念是不一样的：梯度下降使用损失（比如交叉熵损失）来训练模型，因此损失必须是可微分的（至少是在评估点可微分），梯度不能在所有地方都是 0。另外，就算损失比较难解释也没有关系。相反的，指标（比如准确率）是用来评估模型的：指标的解释性一定要好，可以是不可微分的，或者可以在任何地方的梯度都是 0。

但是，在多数情况下，定义一个自定义指标函数和定义一个自定义损失函数是完全一样的。事实上，刚才创建的 Huber 损失函数也可以用来当指标（持久化也是同样的，只需要保存函数名`Huber_fn`就成）：

```py
model.compile(loss="mse", optimizer="nadam", metrics=[create_huber(2.0)])
```

对于训练中的每个批次，Keras 能计算该指标，并跟踪自周期开始的指标平均值。大多数时候，这样没有问题。但会有例外！比如，考虑一个二元分类器的准确性。第 3 章介绍过，准确率是真正值除以正预测数（包括真正值和假正值）。假设模型在第一个批次做了 5 个正预测，其中 4 个是正确的，准确率就是 80%。再假设模型在第二个批次做了 3 次正预测，但没有一个预测对，则准确率是 0%。如果对这两个准确率做平均，则平均值是 40%。但它不是模型在两个批次上的准确率！事实上，真正值总共有 4 个，正预测有 8 个，整体的准确率是 50%。我们需要的是一个能跟踪真正值和正预测数的对象，用该对象计算准确率。这就是类`keras.metrics.Precision`所做的：

```py
>>> precision = keras.metrics.Precision()
>>> precision([0, 1, 1, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1])
<tf.Tensor: id=581729, shape=(), dtype=float32, numpy=0.8>
>>> precision([0, 1, 0, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 0, 0, 0])
<tf.Tensor: id=581780, shape=(), dtype=float32, numpy=0.5>
```
