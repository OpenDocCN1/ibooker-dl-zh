
```py
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=
                                                           y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(STEPS):
        batch = mnist.train.next_batch(50)

        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], 
                                                           y_: batch[1],
                                                           keep_prob: 1.0})
            print "step {}, training accuracy {}".format(i, train_accuracy)

        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1],
                                        keep_prob: 0.5})

    X = mnist.test.images.reshape(10, 1000, 784)
    Y = mnist.test.labels.reshape(10, 1000, 10)
    test_accuracy = np.mean([sess.run(accuracy, 
                            feed_dict={x:X[i], y_:Y[i],keep_prob:1.0}) 
                            for i in range(10)])

print "test accuracy: {}".format(test_accuracy)
```

这个模型的性能已经相当不错，仅经过 5 个周期，准确率就超过了 99%，¹这相当于 5,000 步，每个步骤的迷你批次大小为 50。

有关多年来使用该数据集的模型列表以及如何进一步改进结果的一些想法，请查看[http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)。

# CIFAR10

*CIFAR10*是另一个在计算机视觉和机器学习领域有着悠久历史的数据集。与 MNIST 类似，它是一个常见的基准，各种方法都会被测试。[CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)是一个包含 60,000 张尺寸为 32×32 像素的彩色图像的数据集，每张图像属于以下十个类别之一：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船和卡车。

针对这个数据集的最先进的深度学习方法在分类这些图像方面与人类一样出色。在本节中，我们首先使用相对较简单的方法，这些方法将运行相对较快。然后，我们简要讨论这些方法与最先进方法之间的差距。

## 加载 CIFAR10 数据集

在本节中，我们构建了一个类似于用于 MNIST 的内置`input_data.read_data_sets()`的 CIFAR10 数据管理器。²

首先，[下载数据集的 Python 版本](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)并将文件提取到本地目录中。现在应该有以下文件：

+   `data_batch_1`, `data_batch_2`, `data_batch_3`, `data_batch_4`, `data_batch_5`

+   `test_batch`

+   `batches_meta`

+   `readme.html`

`data_batch_X`文件是包含训练数据的序列化数据文件，`test_batch`是一个类似的包含测试数据的序列化文件。`batches_meta`文件包含从数字到语义标签的映射。`.html`文件是 CIFAR-10 数据集网页的副本。

由于这是一个相对较小的数据集，我们将所有数据加载到内存中：

```py
class CifarLoader(object):
def __init__(self, source_files):
self._source = source_files
self._i = 0
self.images = None
self.labels = None

def load(self):
data = [unpickle(f) for f in self._source]
images = np.vstack([d["data"] for d in data])
n = len(images)
self.images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1)\
.astype(float) / 255
self.labels = one_hot(np.hstack([d["labels"] for d in data]), 10)
return self

def next_batch(self, batch_size):
x, y = self.images[self._i:self._i+batch_size], 
self.labels[self._i:self._i+batch_size]
self._i = (self._i + batch_size) % len(self.images)
return x, y
```

在这里我们使用以下实用函数：

```py
DATA_PATH="`/path/to/CIFAR10`"defunpickle(file):withopen(os.path.join(DATA_PATH,file),'rb')asfo:dict=cPickle.load(fo)returndictdefone_hot(vec,vals=10):n=len(vec)out=np.zeros((n,vals))out[range(n),vec]=1returnout
```

`unpickle()`函数返回一个带有`data`和`labels`字段的`dict`，分别包含图像数据和标签。`one_hot()`将标签从整数（范围为 0 到 9）重新编码为长度为 10 的向量，其中除了标签位置上的 1 之外，所有位置都是 0。

最后，我们创建一个包含训练和测试数据的数据管理器：

```py
class CifarDataManager(object):
def __init__(self):
self.train = CifarLoader(["data_batch_{}".format(i) 
for i in range(1, 6)])
.load()
self.test = CifarLoader(["test_batch"]).load()
```

使用 Matplotlib，我们现在可以使用数据管理器来显示一些 CIFAR10 图像，并更好地了解这个数据集中的内容：

```py
def display_cifar(images, size):
    n = len(images)
    plt.figure()
    plt.gca().set_axis_off()
    im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)])
                    for i in range(size)])
    plt.imshow(im)
    plt.show()

d = CifarDataManager()
print "Number of train images: {}".format(len(d.train.images))
print "Number of train labels: {}".format(len(d.train.labels))
print "Number of test images: {}".format(len(d.test.images))
print "Number of test images: {}".format(len(d.test.labels))
images = d.train.images
display_cifar(images, 10)
```

# Matplotlib

*Matplotlib*是一个用于绘图的有用的 Python 库，设计得看起来和行为类似于 MATLAB 绘图。这通常是快速绘制和可视化数据集的最简单方法。

`display_cifar()`函数的参数是`images`（包含图像的可迭代对象）和`size`（我们想要显示的图像数量），并构建并显示一个`size×size`的图像网格。这是通过垂直和水平连接实际图像来形成一个大图像。

在显示图像网格之前，我们首先打印训练/测试集的大小。CIFAR10 包含 50K 个训练图像和 10K 个测试图像：

```py
Number of train images: 50000
Number of train labels: 50000
Number of test images: 10000
Number of test images: 10000
```

在图 4-5 中生成并显示的图像旨在让人了解 CIFAR10 图像实际上是什么样子的。值得注意的是，这些小的 32×32 像素图像每个都包含一个完整的单个对象，该对象位于中心位置，即使在这种分辨率下也基本上是可识别的。

![CIFAR10](img/letf_0405.png)

###### 图 4-5。100 个随机的 CIFAR10 图像。

## 简单的 CIFAR10 模型

我们将从先前成功用于 MNIST 数据集的模型开始。回想一下，MNIST 数据集由 28×28 像素的灰度图像组成，而 CIFAR10 图像是带有 32×32 像素的彩色图像。这将需要对计算图的设置进行轻微调整：

```py
cifar = CifarDataManager()

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

conv1 = conv_layer(x, shape=[5, 5, 3, 32])
conv1_pool = max_pool_2x2(conv1)

conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
conv2_pool = max_pool_2x2(conv2)
conv2_flat = tf.reshape(conv2_pool, [-1, 8 * 8 * 64])

full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

y_conv = full_layer(full1_drop, 10)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, 
                                                               y_))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def test(sess):
X = cifar.test.images.reshape(10, 1000, 32, 32, 3)
Y = cifar.test.labels.reshape(10, 1000, 10)
acc = np.mean([sess.run(accuracy, feed_dict={x: X[i], y_: Y[i], 
                                             keep_prob: 1.0})
for i in range(10)])
print "Accuracy: {:.4}%".format(acc * 100)

with tf.Session() as sess:
sess.run(tf.global_variables_initializer())

for i in range(STEPS):
batch = cifar.train.next_batch(BATCH_SIZE)
sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], 
                                keep_prob: 0.5})

test(sess)
```

这第一次尝试将在几分钟内达到大约 70%的准确度（使用批量大小为 100，自然取决于硬件和配置）。这好吗？截至目前，最先进的深度学习方法在这个数据集上实现了超过 95%的准确度，但是使用更大的模型并且通常需要许多小时的训练。

这与之前介绍的类似 MNIST 模型之间存在一些差异。首先，输入由大小为 32×32×3 的图像组成，第三维是三个颜色通道：

```py
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
```

同样，在两次池化操作之后，我们这次剩下的是大小为 8×8 的 64 个特征图：

```py
conv2_flat = tf.reshape(conv2_pool, [-1, 8 * 8 * 64])
```

最后，为了方便起见，我们将测试过程分组到一个名为`test()`的单独函数中，并且我们不打印训练准确度值（可以使用与 MNIST 模型中相同代码添加回来）。

一旦我们有了一些可接受的基线准确度的模型（无论是从简单的 MNIST 模型还是从其他数据集的最先进模型中派生的），一个常见的做法是通过一系列的适应和更改来尝试改进它，直到达到我们的目的所需的内容。

在这种情况下，保持其他所有内容不变，我们将添加一个具有 128 个特征图和 dropout 的第三个卷积层。我们还将把完全连接层中的单元数从 1,024 减少到 512：

```py
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

conv1 = conv_layer(x, shape=[5, 5, 3, 32])
conv1_pool = max_pool_2x2(conv1)

conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
conv2_pool = max_pool_2x2(conv2)

conv3 = conv_layer(conv2_pool, shape=[5, 5, 64, 128])
conv3_pool = max_pool_2x2(conv3)
conv3_flat = tf.reshape(conv3_pool, [-1, 4 * 4 * 128])
conv3_drop = tf.nn.dropout(conv3_flat, keep_prob=keep_prob)

full_1 = tf.nn.relu(full_layer(conv3_drop, 512))
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

y_conv = full_layer(full1_drop, 10)
```

这个模型将需要稍长一点的时间来运行（但即使没有复杂的硬件，也不会超过一个小时），并且可以达到大约 75%的准确度。

这仍然与最佳已知方法之间存在相当大的差距。有几个独立适用的元素可以帮助缩小这个差距：

**模型大小**

对于这种数据集和类似数据集，大多数成功的方法使用更深的网络和更多可调参数。

**其他类型的层和方法**

通常与这里介绍的层一起使用的是其他类型的流行层，比如局部响应归一化。

**优化技巧**

更多关于这个的内容以后再说！

**领域知识**

利用领域知识进行预处理通常是很有帮助的。在这种情况下，这将是传统的图像处理。

**数据增强**

基于现有数据集添加训练数据可能会有所帮助。例如，如果一张狗的图片水平翻转，那么显然仍然是一张狗的图片（但垂直翻转呢？）。小的位移和旋转也经常被使用。

**重用成功的方法和架构**
