

# 十三、使用 TensorFlow 加载和预处理数据

> 译者：[@SeanCheney](https://www.jianshu.com/u/130f76596b02)

目前为止，我们只是使用了存放在内存中的数据集，但深度学习系统经常需要在大数据集上训练，而内存放不下大数据集。其它的深度学习库通过对大数据集做预处理，绕过了内存限制，但 TensorFlow 通过 Data API，使一切都容易了：只需要创建一个数据集对象，告诉它去哪里拿数据，以及如何做转换就行。TensorFlow 负责所有的实现细节，比如多线程、队列、批次和预提取。另外，Data API 和`tf.keras`可以无缝配合！

Data API 还可以从现成的文件（比如 CSV 文件）、固定大小的二进制文件、使用 TensorFlow 的 TFRecord 格式的文件（支持大小可变的记录）读取数据。TFRecord 是一个灵活高效的二进制格式，基于 Protocol Buffers（一个开源二进制格式）。Data API 还支持从 SQL 数据库读取数据。另外，许多开源插件也可以用来从各种数据源读取数据，包括谷歌的 BigQuery。

高效读取大数据集不是唯一的难点：数据还需要进行预处理，通常是归一化。另外，数据集中并不是只有数值字段：可能还有文本特征、类型特征，等等。这些特征需要编码，比如使用独热编码或嵌入（后面会看到，嵌入嵌入是用来标识类型或标记的紧密向量）。预处理的一种方式是写自己的自定义预处理层，另一种是使用 Kera 的标准预处理层。

本章中，我们会介绍 Data API，TFRecord 格式，以及如何创建自定义预处理层，和使用 Keras 的预处理层。还会快速学习 TensorFlow 生态的一些项目：

*   TF Transform (`tf.Transform`)：可以用来编写单独的预处理函数，它可以在真正训练前，运行在完整训练集的批模式中，然后输出到 TF 函数，插入到训练好的模型中。只要模型在生产环境中部署好了，就能随时预处理新的实例。

*   TF Datasets (TFDS)。提供了下载许多常见数据集的函数，包括 ImageNet，和数据集对象（可用 Data API 操作）。

## Data API

整个 Data API 都是围绕数据集`dataset`的概念展开的：可以猜得到，数据集表示一连串数据项。通常你是用的数据集是从硬盘里逐次读取数据的，简单起见，我们是用`tf.data.Dataset.from_tensor_slices()`创建一个存储于内存中的数据集：

```py
>>> X = tf.range(10)  # any data tensor
>>> dataset = tf.data.Dataset.from_tensor_slices(X)
>>> dataset
<TensorSliceDataset shapes: (), types: tf.int32> 
```

函数`from_tensor_slices()`取出一个张量，创建了一个`tf.data.Dataset`，它的元素是`X`的全部切片，因此这个数据集包括 10 项：张量 0、1、2、...、9。在这个例子中，使用`tf.data.Dataset.range(10)`也能达到同样的效果。

可以像下面这样对这个数据集迭代：

```py
>>> for item in dataset:
...     print(item)
...
tf.Tensor(0, shape=(), dtype=int32)
tf.Tensor(1, shape=(), dtype=int32)
tf.Tensor(2, shape=(), dtype=int32)
[...]
tf.Tensor(9, shape=(), dtype=int32) 
```

### 链式转换

有了数据集之后，通过调用转换方法，可以对数据集做各种转换。每个方法会返回一个新的数据集，因此可以将转换像下面这样链接起来（见图 13-1）：

```py
>>> dataset = dataset.repeat(3).batch(7)
>>> for item in dataset:
...     print(item)
...
tf.Tensor([0 1 2 3 4 5 6], shape=(7,), dtype=int32)
tf.Tensor([7 8 9 0 1 2 3], shape=(7,), dtype=int32)
tf.Tensor([4 5 6 7 8 9 0], shape=(7,), dtype=int32)
tf.Tensor([1 2 3 4 5 6 7], shape=(7,), dtype=int32)
tf.Tensor([8 9], shape=(2,), dtype=int32) 
```

![](img/252e2a32408349522b1f991945ebc47b.png)

图 13-1 链接数据集转换

在这个例子中，我们先在原始数据集上调用了`repeat()`方法，返回了一个重复了原始数据集 3 次的新数据集。当然，这步不会复制数据集中的数据三次（如果调用这个方法时没有加参数，新数据集会一直重复源数据集，必须让迭代代码决定何时退出）。然后我们在新数据集上调用了`batch()`方法，这步又产生了一个新数据集。这一步会将上一个数据集的分成 7 个一批次。最后，做一下迭代。可以看到，最后的批次只有两个元素，可以设置`drop_remainder=True`，丢弃最后的两项，将数据对齐。

> 警告：数据集方法不修改数据集，只是生成新的数据集而已，所以要做新数据集的赋值（即使用`dataset = ...`）。

还可以通过`map()`方法转换元素。比如，下面的代码创建了一个每个元素都翻倍的新数据集：

```py
>>> dataset = dataset.map(lambda x: x * 2) # Items: [0,2,4,6,8,10,12] 
```

这个函数可以用来对数据做预处理。有时可能会涉及复杂的计算，比如改变形状或旋转图片，所以通常需要多线程来加速：只需设置参数`num_parallel_calls`就行。注意，传递给`map()`方法的函数必须是可以转换为 TF 函数。

`map()`方法是对每个元素做转换的，`apply()`方法是对数据整体做转换的。例如，下面的代码对数据集应用了`unbatch()`函数（这个函数目前是试验性的，但很有可能加入到以后的版本中）。新数据集中的每个元素都是一个单整数张量，而不是批次大小为 7 的整数。

```py
>>> dataset = dataset.apply(tf.data.experimental.unbatch()) # Items: 0,2,4,... 
```

还可以用`filter()`方法做过滤：

```py
>>> dataset = dataset.filter(lambda x: x < 10) # Items: 0 2 4 6 8 0 2 4 6... 
```

`take()`方法可以用来查看数据：

```py
>>> for item in dataset.take(3):
...     print(item)
...
tf.Tensor(0, shape=(), dtype=int64)
tf.Tensor(2, shape=(), dtype=int64)
tf.Tensor(4, shape=(), dtype=int64) 
```

### 打散数据

当训练集中的实例是独立同分布时，梯度下降的效果最好（见第 4 章）。实现独立同分布的一个简单方法是使用`shuffle()`方法。它能创建一个新数据集，新数据集的前面是一个缓存，缓存中是源数据集的开头元素。然后，无论什么时候取元素，就会从缓存中随便随机取出一个元素，从源数据集中取一个新元素替换。从缓冲器取元素，直到缓存为空。必须要指定缓存的大小，最好大一点，否则随机效果不明显。不要查出内存大小，即使内存够用，缓存超过数据集也是没有意义的。可以提供一个随机种子，如果希望随机的顺序是固定的。例如，下面的代码创建并显示了一个包括 0 到 9 的数据集，重复 3 次，用大小为 5 的缓存做随机，随机种子是 42，批次大小是 7：

```py
>>> dataset = tf.data.Dataset.range(10).repeat(3) # 0 to 9, three times
>>> dataset = dataset.shuffle(buffer_size=5, seed=42).batch(7)
>>> for item in dataset:
...     print(item)
...
tf.Tensor([0 2 3 6 7 9 4], shape=(7,), dtype=int64)
tf.Tensor([5 0 1 1 8 6 5], shape=(7,), dtype=int64)
tf.Tensor([4 8 7 1 2 3 0], shape=(7,), dtype=int64)
tf.Tensor([5 4 2 7 8 9 9], shape=(7,), dtype=int64)
tf.Tensor([3 6], shape=(2,), dtype=int64) 
```

> 提示：如果在随机数据集上调用`repeat()`方法，默认下，每次迭代的顺序都是新的。通常这样没有问题，但如果你想让每次迭代的顺序一样（比如，测试或调试），可以设置`reshuffle_each_iteration=False`。

对于内存放不下的大数据集，这个简单的随机缓存方法就不成了，因为缓存相比于数据集就小太多了。一个解决方法是将源数据本身打乱（例如，Linux 可以用`shuf`命令打散文本文件）。这样肯定能提高打散的效果！即使源数据打散了，你可能还想再打散一点，否则每个周期可能还会出现同样的顺序，模型最后可能是偏的（比如，源数据顺序偶然导致的假模式）。为了将实例进一步打散，一个常用的方法是将源数据分成多个文件，训练时随机顺序读取。但是，相同文件中的实例仍然靠的太近。为了避免这点，可以同时随机读取多个文件，做交叉。在最顶层，可以用`shuffle()`加一个随机缓存。如果这听起来很麻烦，不用担心：Data API 都为你实现了，几行代码就行。

#### 多行数据交叉

首先，假设加载了加州房价数据集，打散它（除非已经打散了），分成训练集、验证集、测试集。然后将每个数据集分成多个 csv 文件，每个如下所示（每行包含 8 个输入特征加上目标中位房价）：

```py
MedInc,HouseAge,AveRooms,AveBedrms,Popul,AveOccup,Lat,Long,MedianHouseValue
3.5214,15.0,3.0499,1.1065,1447.0,1.6059,37.63,-122.43,1.442
5.3275,5.0,6.4900,0.9910,3464.0,3.4433,33.69,-117.39,1.687
3.1,29.0,7.5423,1.5915,1328.0,2.2508,38.44,-122.98,1.621
[...] 
```

再假设`train_filepaths`包括了训练文件路径的列表（还要`valid_filepaths`和`test_filepaths`）：

```py
>>> train_filepaths
['datasets/housing/my_train_00.csv', 'datasets/housing/my_train_01.csv',...] 
```

另外，可以使用文件模板，比如`train_filepaths = "datasets/housing/my_train_*.csv"`。现在，创建一个数据集，包括这些文件路径：

```py
filepath_dataset = tf.data.Dataset.list_files(train_filepaths, seed=42) 
```

默认，`list_files()`函数返回一个文件路径打散的数据集。也可以设置`shuffle=False`，文件路径就不打散了。

然后，可以调用`leave()`方法，一次读取 5 个文件，做交叉操作（跳过第一行表头，使用`skip()`方法）：

```py
n_readers = 5
dataset = filepath_dataset.interleave(
    lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
    cycle_length=n_readers) 
```

`interleave()`方法会创建一个数据集，它从`filepath_dataset`读 5 条文件路径，对每条路径调用函数（例子中是用的匿名函数）来创建数据集（例子中是`TextLineDataset`）。为了更清楚点，这一步总欧诺个由七个数据集：文件路径数据集，交叉数据集，和五个`TextLineDatasets`数据集。当迭代交叉数据集时，会循环`TextLineDatasets`，每次读取一行，知道数据集为空。然后会从`filepath_dataset`再获取五个文件路径，做同样的交叉，直到文件路径为空。

> 提示：为了交叉得更好，最好让文件有相同的长度，否则长文件的尾部不会交叉。

默认情况下，`interleave()`不是并行的，只是顺序从每个文件读取一行。如果想变成并行读取文件，可以设定参数`num_parallel_calls`为想要的线程数（`map()`方法也有这个参数）。还可以将其设置为`tf.data.experimental.AUTOTUNE`，让 TensorFlow 根据 CPU 自己找到合适的线程数（目前这是个试验性的功能）。看看目前数据集包含什么：

```py
>>> for line in dataset.take(5):
...     print(line.numpy())
...
b'4.2083,44.0,5.3232,0.9171,846.0,2.3370,37.47,-122.2,2.782'
b'4.1812,52.0,5.7013,0.9965,692.0,2.4027,33.73,-118.31,3.215'
b'3.6875,44.0,4.5244,0.9930,457.0,3.1958,34.04,-118.15,1.625'
b'3.3456,37.0,4.5140,0.9084,458.0,3.2253,36.67,-121.7,2.526'
b'3.5214,15.0,3.0499,1.1065,1447.0,1.6059,37.63,-122.43,1.442' 
```

忽略表头行，这是五个 csv 文件的第一行，随机选取的。看起来不错。但是也看到了，都是字节串，需要解析数据，缩放数据。

### 预处理数据

实现一个小函数来做预处理：

```py
X_mean, X_std = [...] # mean and scale of each feature in the training set
n_inputs = 8

def preprocess(line):
  defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
  fields = tf.io.decode_csv(line, record_defaults=defs)
  x = tf.stack(fields[:-1])
  y = tf.stack(fields[-1:])
  return (x - X_mean) / X_std, y 
```

逐行看下代码：

*   首先，代码假定已经算好了训练集中每个特征的平均值和标准差。`X_mean`和`X_std`是 1D 张量（或 NumPy 数组），包含八个浮点数，每个都是特征。

*   `preprocess()`函数从 csv 取一行，开始解析。使用`tf.io.decode_csv()`函数，接收两个参数，第一个是要解析的行，第二个是一个数组，包含 csv 文件每列的默认值。这个数组不仅告诉 TensorFlow 每列的默认值，还有总列数和数据类型。在这个例子中，是告诉 TensorFlow，所有特征列都是浮点数，缺失值默认为，但提供了一个类型是`tf.float32`的空数组，作为最后一列（目标）的默认值：数组告诉 TensorFlow 这一列包含浮点数，但没有默认值，所以碰到空值时会报异常。

*   `decode_csv()`函数返回一个标量张量（每列一个）的列表，但应该返回 1D 张量数组。所以在所有张量上调用了`tf.stack()`，除了最后一个。然后对目标值做同样的操作（让其成为只包含一个值，而不是标量张量的 1D 张量数组）。

*   最后，对特征做缩放，减去平均值，除以标准差，然后返回包含缩放特征和目标值的元组。

测试这个预处理函数：

```py
>>> preprocess(b'4.2083,44.0,5.3232,0.9171,846.0,2.3370,37.47,-122.2,2.782')
(<tf.Tensor: id=6227, shape=(8,), dtype=float32, numpy=
 array([ 0.16579159,  1.216324  , -0.05204564, -0.39215982, -0.5277444 ,
        -0.2633488 ,  0.8543046 , -1.3072058 ], dtype=float32)>,
 <tf.Tensor: [...], numpy=array([2.782], dtype=float32)>) 
```

很好，接下来将函数应用到数据集上。

### 整合

为了让代码可复用，将前面所有讨论过的东西编程一个小函数：创建并返回一个数据集，可以高效从多个 csv 文件加载加州房价数据集，做预处理、打散、选择性重复，做批次（见图 3-2）：

