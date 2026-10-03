

# 第三章：数据预处理

在本章中，您将学习如何为训练准备和设置数据。机器学习工作中最常见的数据格式是表格、图像和文本。与每种数据格式相关的常见实践技术，尽管如何设置数据工程流水线当然取决于您的问题陈述是什么以及您试图预测什么。

我将详细查看所有三种格式，使用具体示例来引导您了解这些技术。所有数据都可以直接读入 Python 运行时内存；然而，这并不是最有效的使用计算资源的方式。当讨论文本数据时，我将特别关注标记和字典。通过本章结束时，您将学会如何准备表格、图像和文本数据进行训练。

## 为训练准备表格数据

在表格数据集中，重要的是要确定哪些列被视为分类列，因为您必须将它们的值编码为类或类的二进制表示（独热编码），而不是数值值。表格数据集的另一个方面是多个特征之间的相互作用的潜力。本节还将查看 TensorFlow 提供的 API，以便更容易地建模列之间的交互。

通常会遇到作为 CSV 文件的表格数据集，或者仅仅作为数据库查询结果的结构化输出。在这个例子中，我们将从已经在 pandas DataFrame 中的数据集开始，并学习如何转换它并为模型训练设置它。我们将使用*泰坦尼克*数据集，这是一个开源的表格数据集，通常用于教学，因为其可管理的大小和可用性。该数据集包含每位乘客的属性，如年龄、性别、舱位等级，以及他们是否幸存。我们将尝试根据他们的属性或特征来预测每位乘客的生存概率。请注意，这是一个用于教学和学习目的的小数据集。实际上，您的数据集可能会更大。您可能会对一些输入参数的默认值做出不同的决定，并选择不同的默认值，所以不要对这个例子过于字面理解。

让我们从加载所有必要的库开始：

```py
import functools
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
```

从谷歌的公共存储中加载数据：

```py
TRAIN_DATA_URL = "https://storage.googleapis.com/
tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/
tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", 
TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)
```

现在看看`train_file_path`：

```py
print(train_file_path)

/root/.keras/datasets/train.csv
```

这个文件路径指向一个 CSV 文件，我们将其读取为一个 pandas DataFrame：

```py
titanic_df = pd.read_csv(train_file_path, header='infer')
```

图 3-1 展示了`titanic_df`作为 pandas DataFrame 的样子。

![泰坦尼克数据集作为 pandas DataFrame](img/t2pr_0301.png)

###### 图 3-1。*泰坦尼克*数据集作为 pandas DataFrame

### 标记列

如您在图 3-1 中所见，这些数据中既有数值列，也有分类列。目标列，或者用于预测的列，是“survived”列。您需要将其标记为目标，并将其余列标记为特征。

###### 提示

在 TensorFlow 中的最佳实践是将您的表格转换为流式数据集。这种做法确保数据的大小不会影响内存消耗。

为了做到这一点，TensorFlow 提供了函数`tf.data.experimental.make_csv_dataset`：

```py
LABEL_COLUMN = 'survived'
LABELS = [0, 1]

train_ds = tf.data.experimental.make_csv_dataset(
      train_file_path,
      batch_size=3,
      label_name=LABEL_COLUMN,
      na_value="?",
      num_epochs=1,
      ignore_errors=True)

test_ds = tf.data.experimental.make_csv_dataset(
      test_file_path,
      batch_size=3,
      label_name=LABEL_COLUMN,
      na_value="?",
      num_epochs=1,
      ignore_errors=True)
```

在前面的函数签名中，您指定要生成数据集对象的文件路径。`batch_size`被任意设置为一个较小的值（在本例中为 3），以便方便检查数据。我们还将`label_name`设置为“survived”列。对于数据质量，如果任何单元格中指定了问号（?），您希望将其解释为“NA”（不适用）。对于训练，将`num_epochs`设置为对数据集进行一次迭代。您可以忽略任何解析错误或空行。

接下来，检查数据：

```py
for batch, label in train_ds.take(1):
  print(label)
  for key, value in batch.items():
    print("{}: {}".format(key,value.numpy()))
```

它看起来类似于图 3-2。

![一个批次的泰坦尼克数据集](img/t2pr_0302.png)

###### 图 3-2. *泰坦尼克*数据集的一个批次

以下是训练范式消耗训练数据集的主要步骤：

1. 按特征类型指定列。
2. 决定是否嵌入或交叉列。
3. 选择感兴趣的列，可能作为一个实验。
4. 为训练范式创建一个“特征层”以供使用。

现在您已经将数据设置为数据集，可以根据其特征类型指定每列，例如数字或分类，如果需要的话可以进行分桶。如果唯一类别太多且降维会有帮助，还可以嵌入列。

让我们继续进行第 1 步。有四个数字列：`age`、`n_siblings_spouses`、`parch`和`fare`。五列是分类的：`sex`、`class`、`deck`、`embark_town`和`alone`。完成后，您将创建一个`feature_columns`列表来保存所有特征列。

以下是如何根据实际数字值严格指定数字列的方法，而不进行任何转换：

```py
feature_columns = []

# numeric cols
for header in ['age', 'n_siblings_spouses', 'parch', 'fare']:
feature_columns.append(feature_column.numeric_column(header))
```

请注意，除了使用`age`本身，您还可以将`age`分箱，例如按年龄分布的四分位数。但是分箱边界（四分位数）是什么？您可以检查 pandas DataFrame 中数字列的一般统计信息：

```py
titanic_df.describe()
```

图 3-3 显示了输出。

![泰坦尼克号数据集中数字列的统计](img/t2pr_0303.png)

###### 图 3-3。*泰坦尼克号*数据集中的数字列统计

让我们尝试为年龄设置三个分箱边界：23、28 和 35。这意味着乘客年龄将被分组为第一四分位数、第二四分位数和第三四分位数（如图 3-3 所示）：

```py
age = feature_column.numeric_column('age')
age_buckets = feature_column.
bucketized_column(age, boundaries=[23, 28, 35])
```

因此，除了“age”之外，您还生成了另一个列“age_bucket”。

为了了解每个分类列的性质，了解其中的不同值将是有帮助的。您需要使用每列中的唯一条目对词汇表进行编码。对于分类列，这意味着您需要确定哪些条目是唯一的：

```py
h = {}
for col in titanic_df:
  if col in ['sex', 'class', 'deck', 'embark_town', 'alone']:
    print(col, ':', titanic_df[col].unique())
    h[col] = titanic_df[col].unique()
```

结果显示在图 3-4 中。

![泰坦尼克号数据集中每个分类列中的唯一值](img/t2pr_0304.png)

###### 图 3-4。数据集中每个分类列的唯一值

您需要以字典格式跟踪这些唯一值，以便模型进行映射和查找。因此，您将对“sex”列中的唯一分类值进行编码：

```py
sex_type = feature_column.categorical_column_with_vocabulary_list(
      'Type', ['male' 'female'])
sex_type_one_hot = feature_column.indicator_column(sex_type)
```

然而，如果列表很长，逐个写出会变得不方便。因此，当您遍历分类列时，可以将每列的唯一值保存在 Python 字典数据结构`h`中以供将来查找。然后，您可以将唯一值作为列表传递到这些词汇表中：

```py
sex_type = feature_column.
categorical_column_with_vocabulary_list(
      'Type', h.get('sex').tolist())
sex_type_one_hot = feature_column.
indicator_column(sex_type)

class_type = feature_column.
categorical_column_with_vocabulary_list(
      'Type', h.get('class').tolist())
class_type_one_hot = feature_column.
indicator_column(class_type)

deck_type = feature_column.
categorical_column_with_vocabulary_list(
      'Type', h.get('deck').tolist())
deck_type_one_hot = feature_column.
indicator_column(deck_type)

embark_town_type = feature_column.
categorical_column_with_vocabulary_list(
      'Type', h.get('embark_town').tolist())
embark_town_type_one_hot = feature_column.
indicator_column(embark_town_type)

alone_type = feature_column.
categorical_column_with_vocabulary_list(
      'Type', h.get('alone').tolist())
alone_one_hot = feature_column.
indicator_column(alone_type)
```

您还可以嵌入“deck”列，因为有八个唯一值，比任何其他分类列都多。将其维度减少到 3：

```py
deck = feature_column.
categorical_column_with_vocabulary_list(
      'deck', titanic_df.deck.unique())
deck_embedding = feature_column.
embedding_column(deck, dimension=3)
```

减少分类列维度的另一种方法是使用*哈希特征列*。该方法根据输入数据计算哈希值，然后为数据指定一个哈希桶。以下代码将“class”列的维度减少到 4：

```py
class_hashed = feature_column.categorical_column_with_hash_bucket(
      'class', hash_bucket_size=4)
```

### 将列交互编码为可能的特征

现在来到最有趣的部分：您将找到不同特征之间的交互（这被称为*交叉列*），并将这些交互编码为可能的特征。这也是您的直觉和领域知识可以有益于您的特征工程努力的地方。例如，基于*泰坦尼克号*灾难的历史背景，一个问题是：一等舱的女性是否比二等或三等舱的女性更有可能生存？为了将这个问题重新表述为一个数据科学问题，您需要考虑乘客的性别和舱位等级之间的交互。然后，您需要选择一个起始维度大小来表示数据的变化性。假设您任意决定将变化性分成五个维度（`hash_bucket_size`）：

```py
cross_type_feature = feature_column.
crossed_column(['sex', 'class'], hash_bucket_size=5)
```

现在您已经创建了所有特征，需要将它们组合在一起，并可能进行实验以决定在训练过程中包含哪些特征。为此，您首先要创建一个列表来保存您想要使用的所有特征：

```py
feature_columns = []
```

然后，您将每个感兴趣的特征附加到列表中：

```py
# append numeric columns
for header in ['age', 'n_siblings_spouses', 'parch', 'fare']:
  feature_columns.append(feature_column.numeric_column(header))

# append bucketized columns
age = feature_column.numeric_column('age')
age_buckets = feature_column.
bucketized_column(age, boundaries=[23, 28, 35])
feature_columns.append(age_buckets)

# append categorical columns
indicator_column_names = 
['sex', 'class', 'deck', 'embark_town', 'alone']
for col_name in indicator_column_names:
  categorical_column = feature_column.
  categorical_column_with_vocabulary_list(
      col_name, titanic_df[col_name].unique())
  indicator_column = feature_column.
  indicator_column(categorical_column)
  feature_columns.append(indicator_column)

# append embedding columns
deck = feature_column.categorical_column_with_vocabulary_list(
      'deck', titanic_df.deck.unique())
deck_embedding = feature_column.
embedding_column(deck, dimension=3)
feature_columns.append(deck_embedding)

# append crossed columns
feature_columns.
append(feature_column.indicator_column(cross_type_feature))
```

现在创建一个特征层：

```py
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
```

这一层将作为您即将构建和训练的模型的第一（输入）层。这是您为模型的训练过程提供所有特征工程框架的方式。

### 创建一个交叉验证数据集

在开始训练之前，您需要为交叉验证目的创建一个小数据集。由于一开始只有两个分区（训练和测试），生成一个交叉验证数据集的一种方法是简单地将其中一个分区细分：

```py
val_df, test_df = train_test_split(test_df, test_size=0.4)
```

在这里，原始`test_df`分区的 40%被随机保留为`test_df`，剩下的 60%现在是`val_df`。通常，测试数据集是三个数据集中最小的，因为它们仅用于最终评估，而不用于模型训练。

现在您已经处理了特征工程和数据分区，还有最后一件事要做：使用数据集将数据流入训练过程。您将把三个 DataFrame（训练、验证和测试）分别转换为自己的数据集：

```py
batch_size = 32
labels = train_df.pop('survived')
working_ds = tf.data.Dataset.
from_tensor_slices((dict(train_df), labels))
working_ds = working_ds.shuffle(buffer_size=len(train_df))
train_ds = working_ds.batch(batch_size)
```

