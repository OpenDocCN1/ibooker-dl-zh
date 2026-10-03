
# 第十章：使用 TensorFlow 导出和提供模型

在本章中，我们将学习如何使用简单和高级的生产就绪方法保存和导出模型。对于后者，我们介绍了 TensorFlow Serving，这是 TensorFlow 中最实用的用于创建生产环境的工具之一。我们将从快速概述两种简单的保存模型和变量的方法开始：首先是通过手动保存权重并重新分配它们，然后是使用`Saver`类创建训练检查点以及导出我们的模型。最后，我们将转向更高级的应用程序，通过使用 TensorFlow Serving 在服务器上部署我们的模型。

## 保存和导出我们的模型

到目前为止，我们已经学习了如何使用 TensorFlow 创建、训练和跟踪模型。现在我们将学习如何保存训练好的模型。保存当前权重状态对于明显的实际原因至关重要——我们不想每次都从头开始重新训练模型，我们也希望有一种方便的方式与他人分享我们模型的状态（就像我们在第七章中看到的预训练模型一样）。

在这一部分，我们将讨论保存和导出的基础知识。我们首先介绍了一种简单的保存和加载权重到文件的方法。然后我们将看到如何使用 TensorFlow 的`Saver`对象来保持序列化模型检查点，其中包含有关权重状态和构建图的信息。

### 分配加载的权重

在训练后重复使用权重的一个天真但实用的方法是将它们保存到文件中，稍后可以加载它们并重新分配给模型。

让我们看一些例子。假设我们希望保存用于 MNIST 数据的基本 softmax 模型的权重，我们从会话中获取它们后，将权重表示为 NumPy 数组，并以我们选择的某种格式保存它们：

```py
import numpy as np
weights = sess.run(W)
np.savez(os.path.join(path, 'weight_storage'), weights)
```

鉴于我们构建了完全相同的图，我们可以加载文件并使用会话中的`.assign()`方法将加载的权重值分配给相应的变量：

```py
loaded_w = np.load(path + 'weight_storage.npz')
loaded_w = loaded_w.items()[0][1]

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)
cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, 
                                                    labels=y_true))
gd_step = tf.train.GradientDescentOptimizer(0.5)\
                                            .minimize(cross_entropy)
correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

with tf.Session() as sess:
    # Assigning loaded weights
  sess.run(W.assign(loaded_w))
  acc = sess.run(accuracy, feed_dict={x: data.test.images, 
                                       y_true: data.test.labels})

print("Accuracy: {}".format(acc))

Out: 
Accuracy: 0.9199
```

接下来，我们将执行相同的过程，但这次是针对第四章中用于 MNIST 数据的 CNN 模型。在这里，我们有八组不同的权重：两个卷积层 1 和 2 的滤波器权重及其对应的偏置，以及两组全连接层的权重和偏置。我们将模型封装在一个类中，以便方便地保持这八个参数的更新列表。

我们还为要加载的权重添加了可选参数：

```py
if weights is not None and sess is not None:
  self.load_weights(weights, sess)
```

以及在传递权重时分配其值的函数：

```py
def load_weights(self, weights, sess):
  for i,w in enumerate(weights):
    print("Weight index: {}".format(i), 
                           "Weight shape: {}".format(w.shape))
    sess.run(self.parameters[i].assign(w))
```

在整个过程中：

```py
class simple_cnn:
  def __init__(self, x_image,keep_prob, weights=None, sess=None):

    self.parameters = []
    self.x_image = x_image

    conv1 = self.conv_layer(x_image, shape=[5, 5, 1, 32])
    conv1_pool = self.max_pool_2x2(conv1)

    conv2 = self.conv_layer(conv1_pool, shape=[5, 5, 32, 64])
    conv2_pool = self.max_pool_2x2(conv2)

    conv2_flat = tf.reshape(conv2_pool, [-1, 7*7*64])
    full_1 = tf.nn.relu(self.full_layer(conv2_flat, 1024))

    full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

    self.y_conv = self.full_layer(full1_drop, 10)

    if weights is not None and sess is not None:
      self.load_weights(weights, sess)

  def weight_variable(self,shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name='weights')

  def bias_variable(self,shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name='biases')

  def conv2d(self,x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], 
                                      padding='SAME')

  def max_pool_2x2(self,x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
               strides=[1, 2, 2, 1], padding='SAME')

  def conv_layer(self,input, shape):
    W = self.weight_variable(shape)
    b = self.bias_variable([shape[3]])
    self.parameters += [W, b]

    return tf.nn.relu(self.conv2d(input, W) + b)

  def full_layer(self,input, size):
    in_size = int(input.get_shape()[1])
    W = self.weight_variable([in_size, size])
    b = self.bias_variable([size])
    self.parameters += [W, b]
    return tf.matmul(input, W) + b

  def load_weights(self, weights, sess):
    for i,w in enumerate(weights):
      print("Weight index: {}".format(i), 
                              "Weight shape: {}".format(w.shape))
      sess.run(self.parameters[i].assign(w))
```

在这个例子中，模型已经训练好，并且权重已保存为`cnn_weights`。我们加载权重并将它们传递给我们的 CNN 对象。当我们在测试数据上运行模型时，它将使用预训练的权重：

```py
x = tf.placeholder(tf.float32, shape=[None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

sess = tf.Session()

weights = np.load(path + 'cnn_weight_storage.npz')
weights = weights.items()[0][1]
cnn = simple_cnn(x_image, keep_prob, weights, sess)

cross_entropy = tf.reduce_mean(
           tf.nn.softmax_cross_entropy_with_logits(
                                                logits=cnn.y_conv,
                                                 labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(cnn.y_conv, 1), 
                                     tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

X = data.test.images.reshape(10, 1000, 784)
Y = data.test.labels.reshape(10, 1000, 10)
test_accuracy = np.mean([sess.run(accuracy,
            feed_dict={x:X[i], y_:Y[i],keep_prob:1.0})
            for i in range(10)])

sess.close()

print("test accuracy: {}".format(test_accuracy))

Out: 
Weight index: 0 Weight shape: (5, 5, 1, 32)
Weight index: 1 Weight shape: (32,)
Weight index: 2 Weight shape: (5, 5, 32, 64)
Weight index: 3 Weight shape: (64,)
Weight index: 4 Weight shape: (3136, 1024)
Weight index: 5 Weight shape: (1024,)
Weight index: 6 Weight shape: (1024, 10)
Weight index: 7 Weight shape: (10,)

test accuracy: 0.990100026131
```

我们可以获得高准确度，而无需重新训练。

### Saver 类

TensorFlow 还有一个内置的类，我们可以用于与前面的示例相同的目的，提供额外有用的功能，我们很快就会看到。这个类被称为`Saver`类（在第五章中已经简要介绍过）。

`Saver`添加了操作，允许我们通过使用称为*检查点文件*的二进制文件保存和恢复模型的参数，将张量值映射到变量的名称。与前一节中使用的方法不同，这里我们不必跟踪我们的参数——`Saver`会自动为我们完成。

使用`Saver`非常简单。我们首先通过`tf.train.Saver()`创建一个 saver 实例，指示我们希望保留多少最近的变量检查点，以及可选的保留它们的时间间隔。

例如，在下面的代码中，我们要求只保留最近的七个检查点，并且另外指定每半小时保留一个检查点（这对于性能和进展评估分析可能很有用）：

```py
saver = tf.train.Saver(max_to_keep=7, 
                      keep_checkpoint_every_n_hours=0.5)
```

如果没有给出输入，那么默认情况下会保留最后五个检查点，并且`every_n_hours`功能会被有效地禁用（默认设置为`10000`）。

接下来，我们使用`saver`实例的`.save()`方法保存检查点文件，传递会话参数、文件保存路径以及步数（`global_step`），它会自动连接到每个检查点文件的名称中，表示迭代次数。在训练模型时，这会创建不同步骤的多个检查点。

在这个代码示例中，每 50 个训练迭代将在指定目录中保存一个文件：

```py
DIR="`*path/to/model*`"withtf.Session()assess:forstepinrange(1,NUM_STEPS+1):batch_xs,batch_ys=data.train.next_batch(MINIBATCH_SIZE)sess.run(gd_step,feed_dict={x:batch_xs,y_true:batch_ys})ifstep%50==0:saver.save(sess,os.path.join(DIR,"model"),global_step=step)
```

另一个保存的文件名为*checkpoint*包含保存的检查点列表，以及最近检查点的路径：

```py
model_checkpoint_path: "model_ckpt-1000"

all_model_checkpoint_paths: "model_ckpt-700"

all_model_checkpoint_paths: "model_ckpt-750"

all_model_checkpoint_paths: "model_ckpt-800"

all_model_checkpoint_paths: "model_ckpt-850"

all_model_checkpoint_paths: "model_ckpt-900"

all_model_checkpoint_paths: "model_ckpt-950"

all_model_checkpoint_paths: "model_ckpt-1000"
```

在下面的代码中，我们使用`Saver`保存权重的状态：
