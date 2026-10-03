
###### 图 13-10\. 一个深度 RNN（左）通过时间展开（右）

使用 PyTorch 实现深度 RNN 很简单：只需在创建`nn.RNN`模块时将`num_layers`参数设置为所需的循环层数。例如，如果你在先前模型的构造函数中创建`nn.RNN`模块时设置`num_layers=3`，你将得到一个三层 RNN（其余代码保持不变）：

```py
self.rnn = nn.RNN(input_size, hidden_size, num_layers=3, batch_first=True)
```

如果你训练和评估这个模型，你会发现它达到了 29,273 的 MAE。这是我们迄今为止最好的模型！

## 多变量时间序列预测

神经网络的一个重要特性是其灵活性：特别是，它们可以几乎不改变其架构来处理多元时间序列。例如，让我们尝试使用铁路和公交数据作为输入来预测铁路时间序列。实际上，让我们也加入天气类型！由于我们总是可以提前知道明天是工作日、周末还是假日，我们可以将天气类型序列提前一天，这样模型就可以得到明天的天气类型作为输入。为了简单起见，我们将使用 Pandas 进行此处理：

```py
df_mulvar = df[["rail", "bus"]] / 1e6  # use both rail & bus series as input
df_mulvar["next_day_type"] = df["day_type"].shift(-1)  # we know tomorrow's type
df_mulvar = pd.get_dummies(df_mulvar, dtype=float)  # one-hot encode day type
```

现在`df_mulvar`是一个包含五个列的 DataFrame：铁路和公交数据，以及包含下一天类型 one-hot 编码的三个列（回想一下，有三种可能的天气类型，`W`、`A`和`U`）。接下来，我们可以像之前一样进行操作。首先，我们将数据分为三个时期，将其按一百万的比例缩小，并将其转换为张量：

```py
mulvar_train = torch.FloatTensor(df_mulvar["2016-01":"2018-12"].values / 1e6)
mulvar_valid = torch.FloatTensor(df_mulvar["2019-01":"2019-05"].values / 1e6)
mulvar_test = torch.FloatTensor(df_mulvar["2019-06":].values / 1e6)
```

然后，我们需要创建 PyTorch 数据集。如果我们使用`TimeSeriesDataset`，目标将包括第二天铁路和公交客流量，以及下一天类型的 one-hot 编码。由于我们目前只想预测铁路客流量，我们必须调整`TimeSeriesDataset`，只保留目标中的第一个值，即铁路客流量。一种方法是为`TimeSeriesDataset`创建一个新的`MulvarTimeSeriesDataset`类，该类扩展了`TimeSeriesDataset`类，并调整了`__getitem__()`方法以过滤目标：

```py
class MulvarTimeSeriesDataset(TimeSeriesDataset):
    def __getitem__(self, idx):
        window, target = super().__getitem__(idx)
        return window, target[:1]
```

接下来，我们可以创建数据集和数据加载器，就像我们之前做的那样：

```py
mulvar_train_set = MulvarTimeSeriesDataset(mulvar_train, window_length)
mulvar_train_loader = DataLoader(mulvar_train_set, batch_size=32, shuffle=True)
[...]  # create the datasets and data loaders for the validation and test sets
```

如果你查看数据加载器产生的批次，你会发现输入形状是[32, 56, 5]，目标形状是[32, 1]。完美！

因此，我们最终可以创建 RNN：

```py
torch.manual_seed(42)
mulvar_model = SimpleRnnModel(input_size=5, hidden_size=32, output_size=1)
mulvar_model = mulvar_model.to(device)
```

注意到这个模型与之前我们构建的`univar_model` RNN 完全相同，只是`input_size=5`：在每一个时间步，模型现在接收五个输入而不是一个。实际上，这个模型达到了验证集的 MAE 为 23,227。现在我们取得了很大的进步！

实际上，让 RNN 预测铁路和公交客流量并不太难。你只需要在`MulvarTimeSeriesDataset`类中返回`target[:2]`而不是`target[:1]`，并在创建`SimpleRnnModel`时设置`output_size=2`，这就是全部了！

正如我们在第九章中讨论的，使用单个模型执行多个相关任务通常比为每个任务使用单独的模型性能更好，因为为某个任务学习到的特征可能对其他任务也有用，并且因为需要在多个任务上表现良好，这可以防止模型过拟合（这是一种正则化形式）。然而，这取决于任务，在这个特定的情况下，预测公交和铁路客流量的多任务 RNN 并不像专门预测一个或另一个（使用所有五个列作为输入）的模型表现得那么好。尽管如此，它达到了铁路验证 MAE 为 26,441，公交车为 26,178，这仍然相当不错。

## 预测多个时间步长

到目前为止，我们只预测了下一个时间步长的值，但我们可以通过适当地更改目标（例如，为了预测两周后的客流量，我们可以将目标更改为 14 天前的值而不是 1 天前的值）来预测多个步骤。但如果我们想使用单个模型预测下一个 14 个值怎么办？

第一种选择是使用我们之前为铁路时间序列训练的`univar_model` RNN，预测下一个值，并将该值添加到输入中，就像预测的值实际上已经发生一样。然后我们再次使用模型来预测下一个值，依此类推，如下面的代码所示：

```py
n_steps = 14
univar_model.eval()
with torch.no_grad():
    X = rail_valid[:window_length].unsqueeze(dim=0).to(device)
    for step_ahead in range(n_steps):
        y_pred_one = univar_model(X)
        X = torch.cat([X, y_pred_one.unsqueeze(dim=0)], dim=1)

    Y_pred = X[0, -n_steps:, 0]
```

在此代码中，我们取验证期前 56 天的铁路客流量，使用`unsqueeze()`方法添加一个大小为 1 的批次维度（因为我们的`univar_model`期望 3D 输入），然后将张量移动到 GPU 上。现在`X`的形状是[1, 56, 1]。然后我们反复使用模型来预测下一个值，并将每个预测值追加到输入序列中，沿着时间轴（`dim=1`）。由于每个预测的形状为[1, 1]，我们必须再次使用`unsqueeze()`来添加一个大小为 1 的批次维度，然后才能将其连接到`X`。最后，`X`的形状变为[1, 56 + 14, 1]，我们的最终预测是`X`的最后 14 个值。生成的预测结果在图 13-11 中展示。

###### 警告

如果模型在某个时间步长出现错误，那么后续时间步长的预测也会受到影响：错误往往会累积。因此，最好只使用这种技术进行少数几个步骤。

![比较实际与预测数据点随时间变化的线图，突出显示逐个预测 14 个步骤的预测，并显示潜在的误差累积。](img/hmls_1311.png)

###### 图 13-11. 逐个预测 14 个步骤的预测

第二种选择是训练一个 RNN 一次性预测接下来的 14 个值。我们仍然可以使用序列到向量的模型，但它将输出 14 个值而不是 1 个。然而，我们首先需要将目标更改为包含下一个 14 个值的向量。为此，我们可以创建以下类：

```py
class ForecastAheadDataset(TimeSeriesDataset):
    def __len__(self):
        return len(self.series) - self.window_length - 14 + 1

    def __getitem__(self, idx):
        end = idx + self.window_length  # 1st index after window
        window = self.series[idx : end]
        target = self.series[end : end + 14, 0]  # 0 = rail ridership
        return window, target
```

###### 小贴士

我硬编码了数字 14，但在实际项目中你应该使其可配置（例如，就像`window_length`一样）。

这个类继承自`TimeSeriesDataset`类，并调整了其`__len__()`和`__getitem__()`方法。现在目标是一个包含下一个 14 个铁路客流量值的张量，而不仅仅是下一个值。我们可以再次创建一个训练集、一个验证集和一个测试集，基于我们之前构建的多变量时间序列：

```py
ahead_train_set = ForecastAheadDataset(mulvar_train, window_length)
ahead_train_loader = DataLoader(ahead_train_set, batch_size=32, shuffle=True)
[...]  # create the datasets and data loaders for the validation and test sets
```

最后，我们可以创建一个简单的 RNN，就像`mulvar_model`一样，但`output_size=14`：

```py
torch.manual_seed(42)
ahead_model = SimpleRnnModel(input_size=5, hidden_size=32, output_size=14)
ahead_model = ahead_model.to(device)
```

训练这个模型后，你可以一次性预测下一个 14 个值，如下所示：

```py
ahead_model.eval()
with torch.no_grad():
    window = mulvar_valid[:window_length]  # shape [56, 5]
    X = window.unsqueeze(dim=0)            # shape [1, 56, 5]
    Y_pred = ahead_model(X.to(device))     # shape [1, 14]
```

这种方法相当有效。它对下一天的预测显然比对未来 14 天的预测要好，但它不会像之前的方法那样累积误差。现在让我们看看序列到序列模型是否可以做得更好。

###### 小贴士

你可以将这两种方法结合起来，预测很多步的未来：一次性使用一个模型来预测接下来的 14 天，然后将预测结果附加到输入上，再次运行模型以获取接下来 14 天的预测，依此类推。⁠^(8)

## 使用序列到序列模型进行预测

我们不是只在最后一个时间步训练模型来预测接下来的 14 个值，而是可以在每个时间步都训练它来预测接下来的 14 个值。为了清楚起见，在时间步 0 时，模型将输出一个包含从时间步 1 到 14 的预测的向量，然后在时间步 1 时，模型将预测时间步 2 到 15，依此类推。换句话说，目标是连续窗口的序列，每个时间步向前移动一个时间步。对于每个输入窗口的目标不再是向量，而是一个与输入长度相同的序列，每个步骤包含一个 14 维向量。给定一个形状为[*batch size*, *window length*, *input size*]的输入批次，输出将具有形状[*batch size*, *window length*, *output_size*]。这不再是一个序列到向量的 RNN，而是一个序列到序列（或*seq2seq*）RNN。

###### 注意

可能会令人惊讶的是，目标中包含出现在输入中的值（除了最后一个时间步）。这不是作弊吗？幸运的是，根本不是：在每个时间步，RNN 只知道过去的时间步；它不能向前看。它被称为*因果*模型。

你可能想知道，当我们真正只对预测未来值感兴趣时，为什么还要训练一个 seq2seq 模型，这些未来值是在模型的最后一个时间步输出的。你是对的：训练完成后，实际上可以忽略所有输出，除了最后一个时间步的输出。这种技术的主要优势是，损失将包含每个时间步 RNN 输出的项，而不仅仅是最后一个时间步的输出。这意味着将有更多的错误梯度流经模型，并且由于它们将来自每个时间步的输出，而不是仅仅来自最后一个，因此它们不需要像以前那样在时间上流动得那么多。这既可以稳定训练，又可以加快收敛速度。此外，由于模型必须在每个时间步进行预测，它将看到不同长度的输入序列，这可以降低模型过度拟合训练期间使用的特定窗口长度的风险。好吧，至少这是我们的希望！让我们尝试将这种技术应用于铁路客流量时间序列。像往常一样，我们首先需要准备数据集：

```py
class Seq2SeqDataset(ForecastAheadDataset):
    def __getitem__(self, idx):
        end = idx + self.window_length  # 1st index after window
        window = self.series[idx : end]
        target_period = self.series[idx + 1 : end + 14, 0]
        target = target_period.unfold(dimension=0, size=14, step=1)
        return window, target
```

我们的新`Seq2SeqDataset`类继承自`ForecastAheadDataset`类，并重写了`__getitem__()`方法：输入窗口的定义和以前一样，但现在目标是每个时间步连续的窗口，每个时间步向前移动一个时间步。`unfold()`方法就是魔法所在：它从一个张量中产生滑动块。为此，它沿着给定的维度以给定的步数反复滑动，并提取给定大小的块。例如：

```py
>>> torch.tensor([0, 1, 2, 3, 4, 5]).unfold(dimension=0, size=4, step=1)
tensor([[0, 1, 2, 3],
 [1, 2, 3, 4],
 [2, 3, 4, 5]])
```

我们必须再次创建一个训练集、一个验证集和一个测试集，以及相应的数据加载器：

```py
seq_train_set = Seq2SeqDataset(mulvar_train, window_length)
seq_train_loader = DataLoader(seq_train_set, batch_size=32, shuffle=True)
[...]  # create the datasets and data loaders for the validation and test sets
```

最后，我们可以构建序列到序列模型：

```py
class Seq2SeqRnnModel(SimpleRnnModel):
    def forward(self, X):
        outputs, last_state = self.rnn(X)
        return self.output(outputs)

torch.manual_seed(42)
seq_model = Seq2SeqRnnModel(input_size=5, hidden_size=32, output_size=14)
seq_model = seq_model.to(device)
```

我们从`SimpleRnnModel`类继承，并重写了`forward()`方法。与之前只将线性`self.output`层应用于最后一个时间步的输出不同，我们现在将其应用于每个时间步的输出。这可能会让你感到惊讶，但到目前为止，我们只将`nn.Linear`层应用于形状为[*批大小*，*特征*]的 2D 输入，但这里的`outputs`张量形状为[*批大小*，*窗口长度*，*隐藏大小*]：它是 3D 的，而不是 2D！幸运的是，这可以正常工作，因为`nn.Linear`层将自动应用于每个时间步，因此模型的预测将具有形状[*批大小*，*窗口长度*，*输出大小*]：这正是我们需要的。

在底层，`nn.Linear`层依赖于`torch.matmul()`进行矩阵乘法。这个函数有效地支持乘以超过两个维度的数组。例如，你可以将形状为[2, 3, 5, 7]的数组与形状为[2, 3, 7, 11]的数组相乘。实际上，这两个数组都可以看作是 2×3 的矩阵网格，而`torch.matmul()`只是简单地乘以两个网格中对应的矩阵。由于将 5×7 矩阵与 7×11 矩阵相乘会产生一个 5×11 矩阵，最终结果是一个 2×3 的 5×11 矩阵网格，表示为一个形状为[2, 3, 5, 11]的张量。广播也被支持；例如，你可以将形状为[10, 56, 32]的数组与形状为[32, 14]的数组相乘：第一个数组中的十个 56×32 矩阵将分别与第二个数组中的相同的 32×14 矩阵相乘，你将得到一个形状为[10, 56, 14]的张量。这就是当你将 3D 输入传递给`nn.Linear`层时发生的事情。

###### 小贴士
