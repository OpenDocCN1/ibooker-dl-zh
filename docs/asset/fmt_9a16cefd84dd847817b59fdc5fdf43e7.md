
```py
Input[0]
conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)

intermediate2 = conv2(intermediate1)
intermediate3 = conv3(intermediate2)

print(intermediate2.size())
print(intermediate3.size())
Output[0]
torch.Size([2, 32, 3])
torch.Size([2, 64, 1])
Input[1]
y_output = intermediate3.squeeze()
print(y_output.size())
Output[1]
torch.Size([2, 64])
```

在每次卷积中，通道维数的大小都会增加，因为通道维数是每个数据点的特征向量。张量实际上是一个特征向量的最后一步是去掉讨厌的尺寸为 1 维。您可以使用`squeeze()`方法来实现这一点。该方法将删除`size=1`的所有维度并返回结果。然后，得到的特征向量可以与其他神经网络组件（如线性层）一起使用来计算预测向量。

另外还有两种方法可以将张量简化为每个数据点的一个特征向量：将剩余的值压平为特征向量，并在额外维度上求平均值。这两种方法如示例 4-16 所示。使用第一种方法，只需使用 PyTorch 的`view()`方法将所有向量平展成单个向量。第二种方法使用一些数学运算来总结向量中的信息。最常见的操作是算术平均值，但沿特征映射维数求和和使用最大值也是常见的。每种方法都有其优点和缺点。扁平化保留了所有的信息，但会导致比预期（或计算上可行）更大的特征向量。平均变得与额外维度的大小无关，但可能会丢失信息。

**示例 4-16：用于归约到特征向量的两个相加方法**

```py
Input[0]
# Method 2 of reducing to feature vectors
print(intermediate1.view(batch_size, -1).size())

# Method 3 of reducing to feature vectors
print(torch.mean(intermediate1, dim=2).size())
# print(torch.max(intermediate1, dim=2).size())
# print(torch.sum(intermediate1, dim=2).size())
Output[0]
torch.Size([2, 80])
torch.Size([2, 16])
```

这种设计一系列卷积的方法是基于经验的：从数据的预期大小开始，处理一系列卷积，最终得到适合您的特征向量。虽然这种方法在实践中效果很好，但在给定卷积的超参数和输入张量的情况下，还有另一种计算张量输出大小的方法，即使用从卷积运算本身推导出的数学公式。

## Example: Classifying Surnames by Using a CNN

为了证明 CNN 的有效性，让我们应用一个简单的 CNN 模型来分类姓氏。这项任务的许多细节与前面的 MLP 示例相同，但真正发生变化的是模型的构造和向量化过程。模型的输入，而不是我们在上一个例子中看到的收缩的单热向量，将是一个单热的矩阵。这种设计将使 CNN 能够更好地“查看”字符的排列，并对在“示例：带有多层感知器的姓氏分类”中使用的收缩的单热编码中丢失的序列信息进行编码。

### The SurnameDataset

虽然姓氏数据集之前在“示例：带有多层感知器的姓氏分类”中进行了描述，但是我们建议您参考“姓氏数据集”来了解它的描述。尽管我们使用了来自“示例：带有多层感知器的姓氏分类”中的相同数据集，但在实现上有一个不同之处：数据集由单热向量矩阵组成，而不是一个收缩的单热向量。为此，我们实现了一个数据集类，它跟踪最长的姓氏，并将其作为矩阵中包含的行数提供给向量化器。列的数量是单热向量的大小（词汇表的大小）。示例 4-17 显示了对`SurnameDataset.__getitem__`的更改；我们显示对`SurnameVectorizer`的更改。在下一小节向量化。

我们使用数据集中最长的姓氏来控制单热矩阵的大小有两个原因。首先，将每一小批姓氏矩阵组合成一个三维张量，要求它们的大小相同。其次，使用数据集中最长的姓氏意味着可以以相同的方式处理每个小批量。

**示例 4-17：为传递最大姓氏长度而修改的`SurnameDataset`**

```py
class SurnameDataset(Dataset):
    # ... existing implementation from Section 4.2

    def __getitem__(self, index):
        row = self._target_df.iloc[index]

        surname_matrix = \
            self._vectorizer.vectorize(row.surname, self._max_seq_length)

        nationality_index = \
             self._vectorizer.nationality_vocab.lookup_token(row.nationality)

        return {'x_surname': surname_matrix,
                'y_nationality': nationality_index}
```

### `Vocabulary`，`Vectorizer`和`DataLoader`

在本例中，尽管词汇表和`DataLoader`的实现方式与“示例：带有多层感知器的姓氏分类”中的示例相同，但`Vectorizer`的`vectorize()`方法已经更改，以适应 CNN 模型的需要。具体来说，正如我们在示例 4-18 中的代码中所示，该函数将字符串中的每个字符映射到一个整数，然后使用该整数构造一个由单热向量组成的矩阵。重要的是，矩阵中的每一列都是不同的单热向量。主要原因是，我们将使用的`Conv1d`层要求数据张量在第 0 维上具有批量，在第 1 维上具有通道，在第 2 维上具有特性。

除了更改为使用单热矩阵之外，我们还修改了向量化器，以便计算姓氏的最大长度并将其保存为`max_surname_length`。

**示例 4-18：为 CNN 实现姓氏向量化器**

```py
class SurnameVectorizer(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""
    def vectorize(self, surname):
        """
        Args:
            surname (str): the surname
        Returns:
            one_hot_matrix (np.ndarray): a matrix of onehot vectors
        """

        one_hot_matrix_size = (len(self.character_vocab), self.max_surname_length)
        one_hot_matrix = np.zeros(one_hot_matrix_size, dtype=np.float32)

        for position_index, character in enumerate(surname):
            character_index = self.character_vocab.lookup_token(character)
            one_hot_matrix[character_index][position_index] = 1

        return one_hot_matrix

    @classmethod
    def from_dataframe(cls, surname_df):
        """Instantiate the vectorizer from the dataset dataframe

        Args:
            surname_df (pandas.DataFrame): the surnames dataset
        Returns:
            an instance of the SurnameVectorizer
        """
        character_vocab = Vocabulary(unk_token="@")
        nationality_vocab = Vocabulary(add_unk=False)
        max_surname_length = 0

        for index, row in surname_df.iterrows():
            max_surname_length = max(max_surname_length, len(row.surname))
            for letter in row.surname:
                character_vocab.add_token(letter)
            nationality_vocab.add_token(row.nationality)

        return cls(character_vocab, nationality_vocab, max_surname_length)
```

### 使用 CNN 重新实现`SurnameClassifier`

我们在本例中使用的模型是使用我们在“卷积神经网络”中介绍的方法构建的。实际上，我们在该部分中创建的用于测试卷积层的“人工”数据与姓氏数据集中使用本例中的向量化器的数据张量的大小完全匹配。正如您在示例 4-19 中所看到的，它与我们在“卷积神经网络”中引入的`Conv1d`序列既有相似之处，也有需要解释的新添加内容。具体来说，该模型类似于“卷积神经网络”，它使用一系列一维卷积来增量地计算更多的特征，从而得到一个单特征向量。

然而，本例中的新内容是使用`sequence`和 ELU PyTorch 模块。序列模块是封装线性操作序列的方便包装器。在这种情况下，我们使用它来封装`Conv1d`序列的应用程序。ELU 是类似于第 3 章中介绍的 ReLU 的非线性函数，但是它不是将值裁剪到 0 以下，而是对它们求幂。ELU 已经被证明是卷积层之间使用的一种很有前途的非线性（Clevert et al.， 2015）。

在本例中，我们将每个卷积的通道数与`num_channels`超参数绑定。我们可以选择不同数量的通道分别进行卷积运算。这样做需要优化更多的超参数。我们发现 256 足够大，可以使模型达到合理的性能。

**示例 4-19：基于 CNN 的`SurnameClassifier`**

```py
import torch.nn as nn
import torch.nn.functional as F

class SurnameClassifier(nn.Module):
    def __init__(self, initial_num_channels, num_classes, num_channels):
        """
        Args:
            initial_num_channels (int): size of the incoming feature vector
            num_classes (int): size of the output prediction vector
            num_channels (int): constant channel size to use throughout network
        """
        super(SurnameClassifier, self).__init__()

        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=initial_num_channels,
                      out_channels=num_channels, kernel_size=3),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                      kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                      kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                      kernel_size=3),
            nn.ELU()
        )
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x_surname, apply_softmax=False):
        """The forward pass of the classifier

        Args:
            x_surname (torch.Tensor): an input data tensor.
                x_surname.shape should be (batch, initial_num_channels,
                                           max_surname_length)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, num_classes)
        """
        features = self.convnet(x_surname).squeeze(dim=2)
        prediction_vector = self.fc(features)

        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector, dim=1)

        return prediction_vector
```

### 训练例程

训练程序包括以下似曾相识的操作序列：实例化数据集，实例化模型，实例化损失函数，实例化优化器，遍历数据集的训练分区和更新模型参数，遍历数据集的验证分区和测量性能，然后重复数据集迭代一定次数。此时，这是本书到目前为止的第三个训练例程实现，应该将这个操作序列内部化。对于这个例子，我们将不再详细描述具体的训练例程，因为它与“示例：带有多层感知器的姓氏分类”中的例程完全相同。但是，输入参数是不同的，您可以在示例 4-20 中看到。

**示例 4-20：CNN 姓氏分类器的输入参数**

```py
args = Namespace(
    # Data and Path information
    surname_csv="data/surnames/surnames_with_splits.csv",
    vectorizer_file="vectorizer.json",
    model_state_file="model.pth",
    save_dir="model_storage/ch4/cnn",
    # Model hyper parameters
    hidden_dim=100,
    num_channels=256,
    # Training hyper parameters
    seed=1337,
    learning_rate=0.001,
    batch_size=128,
    num_epochs=100,
    early_stopping_criteria=5,
    dropout_p=0.1,
    # Runtime omitted for space ...
)
```

### 模型评估和预测

要理解模型的性能，需要对性能进行定量和定性的度量。下面将描述这两个度量的基本组件。我们鼓励您扩展它们，以探索该模型及其所学习到的内容。

## 在测试集上评估

正如“示例：带有多层感知器的姓氏分类”中的示例与本示例之间的训练例程没有变化一样，执行评估的代码也没有变化。总之，调用分类器的`eval()`方法来防止反向传播，并迭代测试数据集。与 MLP 约 50% 的性能相比，该模型的测试集性能准确率约为 56%。尽管这些性能数字绝不是这些特定架构的上限，但是通过一个相对简单的 CNN 模型获得的改进应该足以让您在文本数据上尝试 CNNs。

## 为新的姓氏分类或获取最佳预测
