
# 第五章：文本 I：处理文本和序列，以及 TensorBoard 可视化

在本章中，我们将展示如何在 TensorFlow 中处理序列，特别是文本。我们首先介绍循环神经网络（RNN），这是一类强大的深度学习算法，特别适用于自然语言处理（NLP）。我们展示如何从头开始实现 RNN 模型，介绍一些重要的 TensorFlow 功能，并使用交互式 TensorBoard 可视化模型。然后，我们探讨如何在监督文本分类问题中使用 RNN 进行词嵌入训练。最后，我们展示如何构建一个更高级的 RNN 模型，使用长短期记忆（LSTM）网络，并如何处理可变长度的序列。

## 序列数据的重要性

我们在前一章中看到，利用图像的空间结构可以导致具有出色结果的先进模型。正如在那一章中讨论的那样，利用结构是成功的关键。正如我们将很快看到的，一种极其重要和有用的结构类型是顺序结构。从数据科学的角度来看，这种基本结构出现在许多数据集中，跨越所有领域。在计算机视觉中，视频是随时间演变的一系列视觉内容。在语音中，我们有音频信号，在基因组学中有基因序列；在医疗保健中有纵向医疗记录，在股票市场中有金融数据，等等（见图 5-1）。

![](img/letf_0501.png)

###### 图 5-1。序列数据的普遍性。

一种特别重要的具有强烈顺序结构的数据类型是自然语言——文本数据。利用文本中固有的顺序结构（字符、单词、句子、段落、文档）的深度学习方法处于自然语言理解（NLU）系统的前沿，通常将传统方法远远甩在后面。有许多类型的 NLU 任务需要解决，从文档分类到构建强大的语言模型，从自动回答问题到生成人类级别的对话代理。这些任务非常困难，吸引了整个学术界和工业界 AI 社区的努力和关注。

在本章中，我们专注于基本构建模块和任务，并展示如何在 TensorFlow 中处理序列，主要是文本。我们深入研究了 TensorFlow 中序列模型的核心元素，从头开始实现其中一些，以获得深入的理解。在下一章中，我们将展示更高级的文本建模技术，使用 TensorFlow，而在第七章中，我们将使用提供更简单、高级实现方式的抽象库来实现我们的模型。

我们从最重要和流行的用于序列（特别是文本）的深度学习模型类开始：循环神经网络。

## 循环神经网络简介

循环神经网络是一类强大且广泛使用的神经网络架构，用于建模序列数据。RNN 模型背后的基本思想是序列中的每个新元素都会提供一些新信息，从而更新模型的当前状态。

在前一章中，我们探讨了使用 CNN 模型进行计算机视觉的内容，讨论了这些架构是如何受到当前科学对人类大脑处理视觉信息方式的启发。这些科学观念通常与我们日常生活中对顺序信息处理方式的常识直觉非常接近。

当我们接收新信息时，显然我们的“历史”和“记忆”并没有被抹去，而是“更新”。当我们阅读文本中的句子时，随着每个新单词，我们当前的信息状态会被更新，这不仅取决于新观察到的单词，还取决于前面的单词。

在统计学和概率论中的一个基本数学构造，通常被用作通过机器学习建模顺序模式的基本构件是马尔可夫链模型。比喻地说，我们可以将我们的数据序列视为“链”，链中的每个节点在某种程度上依赖于前一个节点，因此“历史”不会被抹去，而是被延续。

RNN 模型也基于这种链式结构的概念，并且在如何确切地维护和更新信息方面有所不同。正如它们的名称所示，循环神经网络应用某种形式的“循环”。如图 5-2 所示，在某个时间点*t*，网络观察到一个输入*x[t]*（句子中的一个单词），并将其“状态向量”从上一个向量*h[t-1]*更新为*h[t]*。当我们处理新的输入（下一个单词）时，它将以某种依赖于*h[t]*的方式进行，因此依赖于序列的历史（我们之前看到的单词影响我们对当前单词的理解）。如图所示，这种循环结构可以简单地被视为一个长长的展开链，链中的每个节点执行相同类型的处理“步骤”，基于它从前一个节点的输出获得的“消息”。当然，这与先前讨论的马尔可夫链模型及其隐马尔可夫模型（HMM）扩展密切相关，这些内容在本书中没有讨论。

![](img/letf_0502.png)

###### 图 5-2。随时间更新的循环神经网络。

### 基础 RNN 实现

在本节中，我们从头开始实现一个基本的 RNN，探索其内部工作原理，并了解 TensorFlow 如何处理序列。我们介绍了一些强大的、相当低级的工具，TensorFlow 提供了这些工具用于处理序列数据，您可以使用这些工具来实现自己的系统。

在接下来的部分中，我们将展示如何使用更高级别的 TensorFlow RNN 模块。

我们从数学上定义我们的基本模型开始。这主要包括定义循环结构 - RNN 更新步骤。

我们简单的基础 vanilla RNN 的更新步骤是

*h[t]* = *tanh*(*W[x]**x[t]* + *W[h]h[t-1]* + *b*)

其中*W[h]*，*W[x]*和*b*是我们学习的权重和偏置变量，*tanh*(·)是双曲正切函数，其范围在[-1,1]之间，并且与前几章中使用的 sigmoid 函数密切相关，*x[t]*和*h[t]*是之前定义的输入和状态向量。最后，隐藏状态向量乘以另一组权重，产生出现在图 5-2 中的输出。

#### MNIST 图像作为序列

为了初尝序列模型的强大和普适性，在本节中，我们实现我们的第一个 RNN 来解决您现在熟悉的 MNIST 图像分类任务。在本章的后面，我们将专注于文本序列，并看看神经序列模型如何强大地操纵它们并提取信息以解决 NLU 任务。

但是，你可能会问，图像与序列有什么关系？

正如我们在上一章中看到的，卷积神经网络的架构利用了图像的空间结构。虽然自然图像的结构非常适合 CNN 模型，但从不同角度查看图像的结构是有启发性的。在前沿深度学习研究的趋势中，先进模型尝试利用图像中各种顺序结构，试图以某种方式捕捉创造每个图像的“生成过程”。直观地说，这一切归结为图像中相邻区域在某种程度上相关，并试图对这种结构建模。

在这里，为了介绍基本的 RNN 以及如何处理序列，我们将图像简单地视为序列：我们将数据中的每个图像看作是一系列的行（或列）。在我们的 MNIST 数据中，这意味着每个 28×28 像素的图像可以被视为长度为 28 的序列，序列中的每个元素是一个包含 28 个像素的向量（参见图 5-3）。然后，RNN 中的时间依赖关系可以被想象成一个扫描头，从上到下（行）或从左到右（列）扫描图像。

![](img/letf_0503.png)

###### 图 5-3。图像作为像素列的序列。

我们首先加载数据，定义一些参数，并为我们的数据创建占位符：

```py
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Define some parameters
element_size = 28
time_steps = 28
num_classes = 10
batch_size = 128
hidden_layer_size = 128

# Where to save TensorBoard model summaries
LOG_DIR = "logs/RNN_with_summaries"

# Create placeholders for inputs, labels
_inputs = tf.placeholder(tf.float32,shape=[None, time_steps,
                        element_size],
                                              name='inputs')
y = tf.placeholder(tf.float32, shape=[None, num_classes],
                                              name='labels')
```

`element_size`是我们序列中每个向量的维度，在我们的情况下是 28 个像素的行/列。`time_steps`是序列中这样的元素的数量。

正如我们在之前的章节中看到的，当我们使用内置的 MNIST 数据加载器加载数据时，它以展开的形式呈现，即一个包含 784 个像素的向量。在训练期间加载数据批次时（我们稍后将在本节中介绍），我们只需将每个展开的向量重塑为[`batch_size`, `time_steps`, `element_size`]：

```py
batch_x, batch_y = mnist.train.next_batch(batch_size)
# Reshape data to get 28 sequences of 28 pixels
batch_x = batch_x.reshape((batch_size, time_steps, element_size))
```

我们将`hidden_layer_size`设置为`128`（任意值），控制之前讨论的隐藏 RNN 状态向量的大小。

`LOG_DIR`是我们保存模型摘要以供 TensorBoard 可视化的目录。随着我们的学习，您将了解这意味着什么。

## TensorBoard 可视化

在本章中，我们还将简要介绍 TensorBoard 可视化。TensorBoard 允许您监视和探索模型结构、权重和训练过程，并需要对代码进行一些非常简单的添加。更多细节将在本章和本书后续部分提供。

最后，我们创建了适当维度的输入和标签占位符。

### RNN 步骤

让我们实现 RNN 步骤的数学模型。

我们首先创建一个用于记录摘要的函数，稍后我们将在 TensorBoard 中使用它来可视化我们的模型和训练过程（在这个阶段理解其技术细节并不重要）：

```py
# This helper function, taken from the official TensorFlow documentation,
# simply adds some ops that take care of logging summaries
def variable_summaries(var):
  with tf.name_scope('summaries'):
   mean = tf.reduce_mean(var)
   tf.summary.scalar('mean', mean)
   with tf.name_scope('stddev'):
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
   tf.summary.scalar('stddev', stddev)
   tf.summary.scalar('max', tf.reduce_max(var))
   tf.summary.scalar('min', tf.reduce_min(var))
   tf.summary.histogram('histogram', var)
```

接下来，我们创建在 RNN 步骤中使用的权重和偏置变量：

```py
# Weights and bias for input and hidden layer
with tf.name_scope('rnn_weights'):
  with tf.name_scope("W_x"):
    Wx = tf.Variable(tf.zeros([element_size, hidden_layer_size]))
    variable_summaries(Wx)
  with tf.name_scope("W_h"):
    Wh = tf.Variable(tf.zeros([hidden_layer_size, hidden_layer_size]))
    variable_summaries(Wh)
  with tf.name_scope("Bias"):
    b_rnn = tf.Variable(tf.zeros([hidden_layer_size]))
    variable_summaries(b_rnn)
```

### 使用 tf.scan() 应用 RNN 步骤

现在，我们创建一个函数，实现了我们在前一节中看到的基本 RNN 步骤，使用我们创建的变量。现在应该很容易理解这里使用的 TensorFlow 代码：

```py
def rnn_step(previous_hidden_state,x):

    current_hidden_state = tf.tanh(
      tf.matmul(previous_hidden_state, Wh) +
      tf.matmul(x, Wx) + b_rnn)

    return current_hidden_state
```

接下来，我们将这个函数应用到所有的 28 个时间步上：

```py
# Processing inputs to work with scan function
# Current input shape: (batch_size, time_steps, element_size)
processed_input = tf.transpose(_inputs, perm=[1, 0, 2])
# Current input shape now: (time_steps, batch_size, element_size)

initial_hidden = tf.zeros([batch_size,hidden_layer_size])
# Getting all state vectors across time
all_hidden_states = tf.scan(rnn_step,
              processed_input,
              initializer=initial_hidden,
              name='states')
```

在这个小的代码块中，有一些重要的元素需要理解。首先，我们将输入从`[batch_size, time_steps, element_size]`重塑为`[time_steps, batch_size, element_size]`。`tf.transpose()`的`perm`参数告诉 TensorFlow 我们想要交换的轴。现在，我们的输入张量中的第一个轴代表时间轴，我们可以通过使用内置的`tf.scan()`函数在所有时间步上进行迭代，该函数重复地将一个可调用（函数）应用于序列中的元素，如下面的说明所述。

## tf.scan()

这个重要的函数被添加到 TensorFlow 中，允许我们在计算图中引入循环，而不仅仅是通过添加更多和更多的操作复制来“展开”循环。更技术上地说，它是一个类似于 reduce 操作符的高阶函数，但它返回随时间的所有中间累加器值。这种方法有几个优点，其中最重要的是能够具有动态数量的迭代而不是固定的，以及用于图构建的计算速度提升和优化。

为了演示这个函数的使用，考虑以下简单示例（这与本节中的整体 RNN 代码是分开的）：

```py
import numpy as np
import tensorflow as tf

elems = np.array(["T","e","n","s","o","r", " ", "F","l","o","w"])
scan_sum = tf.scan(lambda a, x: a + x, elems)

sess=tf.InteractiveSession()
sess.run(scan_sum)
```

让我们看看我们得到了什么：

```py
array([b'T', b'Te', b'Ten', b'Tens', b'Tenso', b'Tensor', b'Tensor ',
       b'Tensor F', b'Tensor Fl', b'Tensor Flo', b'Tensor Flow'],
       dtype=object)
```

在这种情况下，我们使用`tf.scan()`将字符顺序连接到一个字符串中，类似于算术累积和。

### 顺序输出

正如我们之前看到的，在 RNN 中，我们为每个时间步获得一个状态向量，将其乘以一些权重，然后获得一个输出向量——我们数据的新表示。让我们实现这个：
