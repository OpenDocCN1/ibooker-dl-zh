
```py
def model(data, train=False):
  """The Model definition."""
  # 2D convolution, with 'SAME' padding (i.e. the output feature map has
  # the same size as the input). Note that {strides} is a 4D array whose
  # shape matches the data layout: [image index, y, x, depth].
  conv = tf.nn.conv2d(data,
                      conv1_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
  # Bias and rectified linear non-linearity.
  relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
  # Max pooling. The kernel size spec {ksize} also follows the layout of
  # the data. Here we have a pooling window of 2, and a stride of 2.
  pool = tf.nn.max_pool(relu,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')
  conv = tf.nn.conv2d(pool,
                      conv2_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
  relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
  pool = tf.nn.max_pool(relu,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')
  # Reshape the feature map cuboid into a 2D matrix to feed it to the
  # fully connected layers.
  pool_shape = pool.get_shape().as_list()
  reshape = tf.reshape(
      pool,
      [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
  # Fully connected layer. Note that the '+' operation automatically
  # broadcasts the biases.
  hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
  # Add a 50% dropout during training only. Dropout also scales
  # activations such that no rescaling is needed at evaluation time.
  if train:
    hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
  return tf.matmul(hidden, fc2_weights) + fc2_biases
```

如前所述，网络的基本架构交替使用`tf.nn.conv2d`、`tf.nn.max_pool`和非线性，以及最后一个完全连接的层。为了正则化，在最后一个完全连接的层之后应用一个 dropout 层，但只在训练期间。请注意，我们将输入作为参数`data`传递给函数`model()`。

网络中仍需定义的唯一部分是占位符（示例 6-12）。我们需要定义两个占位符，用于输入训练图像和训练标签。在这个特定的网络中，我们还定义了一个用于评估的单独占位符，允许我们在评估时输入更大的批次。

### 示例 6-12。为架构定义占位符

```py
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64

train_data_node = tf.placeholder(
    tf.float32,
    shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
eval_data = tf.placeholder(
    tf.float32,
    shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
```

有了这些定义，我们现在已经处理了数据，指定了输入和权重，并构建了模型。我们现在准备训练网络（示例 6-13）。

### 示例 6-13。训练 LeNet-5 架构

```py
# Create a local session to run the training.
start_time = time.time()
with tf.Session() as sess:
  # Run all the initializers to prepare the trainable parameters.
  tf.global_variables_initializer().run()
  # Loop through training steps.
  for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
    # Compute the offset of the current minibatch in the data.
    # Note that we could use better randomization across epochs.
    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
    batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
    batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
    # This dictionary maps the batch data (as a NumPy array) to the
    # node in the graph it should be fed to.
    feed_dict = {train_data_node: batch_data,
                 train_labels_node: batch_labels}
    # Run the optimizer to update weights.
    sess.run(optimizer, feed_dict=feed_dict)
```

这个拟合代码的结构看起来与本书迄今为止看到的其他拟合代码非常相似。在每一步中，我们构建一个 feed 字典，然后运行优化器的一步。请注意，我们仍然使用小批量训练。

## 评估经过训练的模型

我们现在有一个正在训练的模型。我们如何评估训练模型的准确性？一个简单的方法是定义一个错误度量。与前几章一样，我们将使用一个简单的分类度量来衡量准确性（示例 6-14）。

### 示例 6-14。评估经过训练的架构的错误

```py
def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
      predictions.shape[0])
```

我们可以使用这个函数来评估网络在训练过程中的错误。让我们引入一个额外的方便函数，以批处理的方式评估任何给定数据集上的预测（示例 6-15）。这种便利是必要的，因为我们的网络只能处理固定批量大小的输入。

### 示例 6-15。一次评估一批数据

```py
def eval_in_batches(data, sess):
  """Get predictions for a dataset by running it in small batches."""
  size = data.shape[0]
  if size < EVAL_BATCH_SIZE:
    raise ValueError("batch size for evals larger than dataset: %d"
                     % size)
  predictions = numpy.ndarray(shape=(size, NUM_LABELS),
                              dtype=numpy.float32)
  for begin in xrange(0, size, EVAL_BATCH_SIZE):
    end = begin + EVAL_BATCH_SIZE
    if end <= size:
      predictions[begin:end, :] = sess.run(
          eval_prediction,
          feed_dict={eval_data: data[begin:end, ...]})
    else:
      batch_predictions = sess.run(
          eval_prediction,
          feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
      predictions[begin:, :] = batch_predictions[begin - size:, :]
  return predictions
```

现在我们可以在训练过程中的内部`for`循环中添加一些仪器（instrumentation），定期评估模型在验证集上的准确性。我们可以通过评分测试准确性来结束训练。示例 6-16 展示了添加了仪器的完整拟合代码。

### 示例 6-16。训练网络的完整代码，添加了仪器

```py
# Create a local session to run the training.
start_time = time.time()
with tf.Session() as sess:
  # Run all the initializers to prepare the trainable parameters.
  tf.global_variables_initializer().run()
  # Loop through training steps.
  for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
    # Compute the offset of the current minibatch in the data.
    # Note that we could use better randomization across epochs.
    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
    batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
    batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
    # This dictionary maps the batch data (as a NumPy array) to the
    # node in the graph it should be fed to.
    feed_dict = {train_data_node: batch_data,
                 train_labels_node: batch_labels}
    # Run the optimizer to update weights.
    sess.run(optimizer, feed_dict=feed_dict)
    # print some extra information once reach the evaluation frequency
    if step % EVAL_FREQUENCY == 0:
      # fetch some extra nodes' data
      l, lr, predictions = sess.run([loss, learning_rate,
                                     train_prediction],
                                    feed_dict=feed_dict)
      elapsed_time = time.time() - start_time
      start_time = time.time()
      print('Step %d (epoch %.2f), %.1f ms' %
            (step, float(step) * BATCH_SIZE / train_size,
             1000 * elapsed_time / EVAL_FREQUENCY))
      print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
      print('Minibatch error: %.1f%%'
            % error_rate(predictions, batch_labels))
      print('Validation error: %.1f%%' % error_rate(
          eval_in_batches(validation_data, sess), validation_labels))
      sys.stdout.flush()
  # Finally print the result!
  test_error = error_rate(eval_in_batches(test_data, sess),
                          test_labels)
  print('Test error: %.1f%%' % test_error)
```

## 读者的挑战

尝试自己训练网络。您应该能够达到<1%的测试错误！

# 回顾

在这一章中，我们向您展示了卷积网络设计的基本概念。这些概念包括构成卷积网络核心构建模块的卷积和池化层。然后我们讨论了卷积架构的应用，如目标检测、图像分割和图像生成。我们以一个深入的案例研究结束了这一章，向您展示了如何在 MNIST 手写数字数据集上训练卷积架构。

在第七章中，我们将介绍循环神经网络，另一个核心深度学习架构。与为图像处理而设计的卷积网络不同，循环架构非常适合处理顺序数据，如自然语言数据集。
