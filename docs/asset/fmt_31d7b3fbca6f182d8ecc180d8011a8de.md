

# 第十章\. 创建用于预测序列的机器学习模型

第九章介绍了序列数据和时间序列的属性，包括季节性、趋势、自相关和噪声。你创建了一个用于预测的合成序列，并探讨了如何进行基本的统计预测。

在接下来的几章中，你将学习如何使用机器学习进行预测。但在开始创建模型之前，你需要了解如何通过创建我们称之为*窗口数据集*的方式来构建时间序列数据，以便为预测模型进行训练。

为了理解为什么你需要这样做，考虑你在第九章中创建的时间序列。你可以在图 10-1 中看到它的一个图。

如果在任何时候你想预测时间`t`的值，你将希望将其作为时间`t`之前值的函数进行预测。例如，假设你想将时间序列在第 1,200 步的值作为其前 30 个值的函数进行预测。在这种情况下，时间步 1,170 到 1,199 的值将决定时间步 1,200 的值（参见图 10-2）。

![图片](img/aiml_1001.png)

###### 图 10-1\. 合成时间序列

![图片](img/aiml_1002.png)

###### 图 10-2\. 影响预测的前期值

现在，这开始看起来熟悉了：你可以将 1,170 到 1,199 的值视为你的*特征*，而 1,200 的值视为你的*标签*。如果你能将你的数据集整理到有特定数量的值作为特征，而后续的值作为标签的状态，并且如果你对数据集中的每个已知值都这样做，那么你最终将得到一组相当不错的特征和标签，你可以用它们来训练一个模型。

在对第九章中的时间序列数据集进行此操作之前，让我们创建一个非常简单的数据集，它具有所有相同的属性，但数据量要小得多。

## 创建窗口数据集

PyTorch 有很多用于操作数据的 API。例如，你可以使用`torch.arange(10)`来创建一个包含数字 0-9 的基本数据集，从而模拟一个时间序列。然后你可以将这个数据集转换为窗口数据集的开始。以下是代码：

```py
import torch

def create_sliding_windows(data, window_size, shift=1):
    # Convert input to tensor if it isn't already
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)

    # Calculate number of valid windows
    n = len(data)
    num_windows = max(0, (n - window_size) // shift + 1)

    # Create strided view of data
    windows = data.unfold(0, window_size, shift)

    return windows

# Example usage:
data = torch.arange(10)
windows = create_sliding_windows(data, window_size=5, shift=1)

# Print each window
for window in windows:
    print(window.numpy())
```

首先，它通过使用范围来创建数据集，这仅仅使数据集包含从 0 到`n - 1`的值，其中`n`在这个例子中是 10。

接下来，调用`create_sliding_windows`并传递参数`5`指定网络应将数据集分割成五个项目的窗口。指定`shift=1`会导致每个窗口相对于前一个窗口移动一个位置：第一个窗口将包含从 0 开始的五个项目，下一个窗口将包含从 1 开始的五个项目，等等。

运行此代码将给出以下结果：

```py
[0 1 2 3 4]
[1 2 3 4 5]
[2 3 4 5 6]
[3 4 5 6 7]
[4 5 6 7 8]
[5 6 7 8 9]
```

之前，您看到我们想要将此数据转换为训练数据，其中`n`个值定义了一个特征，并且有一个随后的值给出标签。您可以通过一些简单的 Python 列表切片来实现这一点，将每个窗口分成两部分：最后一个值之前的所有内容以及最后一个值本身。

这在张量数据上使用`unfold`技术，它创建了一个滑动窗口，将数据转换为之前概述的集合。

它还包含三个参数：

- **维度**（在这种情况下为 0）：这是展开的维度。
- **window_size**：这是每个滑动窗口的大小。
- **shift**：这是窗口之间的步长/步长大小。

这个过程给我们一个`x`和`y`数据集，如下所示：

```py
import torch

def create_sliding_windows_with_target(data, window_size, shift=1):
    # Convert input to tensor if it isn't already
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)

    # Create windows using unfold
    windows = data.unfold(0, window_size, shift)

    # Split each window into features
    features = windows[:, :-1]  # All elements except the last
    targets = windows[:, -1:]   # Just the last element

    return features, targets

# Example usage:
data = torch.arange(10)
features, targets = create_sliding_windows_with_target(data, window_size=5, 
                                                             shift=1)

# Print each window's features and target
for x, y in zip(features, targets):
    print(f"Features: {x.numpy()}, Target: {y.numpy()}")
```

现在的结果与您预期的相符。窗口中的前四个值可以被视为特征，随后的值是标签：

```py
[0 1 2 3] [4]
[1 2 3 4] [5]
[2 3 4 5] [6]
[3 4 5 6] [7]
[4 5 6 7] [8]
[5 6 7 8] [9]
```

现在，使用 PyTorch 的`TensorDataset`类型，您可以将其转换为数据集，并执行诸如洗牌和批处理等原生操作。

注意，在洗牌数据时，确保首先将验证集和测试集分开是一个好的做法。在时间序列数据中，在分割之前洗牌可能会导致训练集的信息泄露到测试集中，这会损害评估过程。

这里，它已经被洗牌并批量处理，批大小为 2：

```py
from torch.utils.data import TensorDataset, DataLoader

# Create dataset
data = torch.arange(10)
features, targets = create_sliding_windows_with_target(data, window_size=5, 
                                                             shift=1)

# Combine features and targets into a dataset
dataset = TensorDataset(features, targets)

# Create DataLoader with shuffling and batching
batch_size = 2
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Example iteration
for batch_features, batch_targets in dataloader:
    print(f"Batch features shape: {batch_features.shape}")
    print(f"Features:\n{batch_features}")
    print(f"Targets:\n{batch_targets}\n")
```

结果显示，第一个批次有两个`x`集合（分别从 5 和 0 开始）及其标签，第二个批次有两个`x`集合（分别从 1 和 3 开始）及其标签，依此类推：

```py
tensor([[5, 6, 7, 8],
        [0, 1, 2, 3]])
Targets:
tensor([[9],
        [4]])

Features:
tensor([[1, 2, 3, 4],
        [3, 4, 5, 6]])
Targets:
tensor([[5],
        [7]])

Features:
tensor([[4, 5, 6, 7],
        [2, 3, 4, 5]])
Targets:
tensor([[8],
        [6]])
```

使用这种技术，您现在可以将任何时间序列数据集转换为神经网络的一组训练数据。在下一节中，您将探索如何从第九章中的合成数据创建训练集。从那里，您将转向创建一个简单的 DNN，该 DNN 在数据上训练，并可用于预测未来值。

### 创建时间序列数据集的窗口版本

回顾一下，这是我们之前章节中用来创建合成时间序列数据集的代码：

```py
import numpy as np
def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

time = np.arange(4 * 365 + 1, dtype="float32")
series = trend(time, 0.1)
baseline = 10
amplitude = 20
slope = 0.09
noise_level = 5

series = baseline + trend(time, slope)
series += seasonality(time, period=365, amplitude=amplitude)
series += noise(time, noise_level, seed=42)
```

这将创建一个看起来像图 10-3 中的时间序列。如果您想修改它，请随意调整各种常数的值。

![](img/aiml_1003.png)

###### 图 10-3。绘制具有趋势、季节性和噪声的时间序列

一旦您有了序列，您就可以使用与上一节中类似的代码将其转换为窗口数据集：

```py
import torch
from torch.utils.data import TensorDataset, DataLoader

# Convert the numpy series to a PyTorch tensor
series_tensor = torch.tensor(series, dtype=torch.float32)

# Create windowed dataset with 30-day windows (predicting next day)
window_size = 30
features, targets = create_sliding_windows_with_target(
    series_tensor, window_size=window_size, shift=1)

# Create PyTorch Dataset and DataLoader
dataset = TensorDataset(features, targets)
batch_size = 32
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Print some information about the dataset
print(f"Series length: {len(series)}")
print(f"Number of windows: {len(features)}")
print(f"Feature shape: {features.shape}")  # Should be (num_windows, 
                                                        window_size-1)
print(f"Target shape: {targets.shape}")    # Should be (num_windows, 1)

# Show a few examples
print("\nFirst few windows:")
for i in range(3):
    print(f"\nWindow {i+1}:")
    print(f"Features (previous {window_size-1} days): {features[i].numpy()}")
    print(f"Target (next day): {targets[i].item():.2f}")
```

您可以在这里看到输出：

```py
Series length: 1461
Number of windows: 1432
Feature shape: torch.Size([1432, 29])
Target shape: torch.Size([1432, 1])

First few windows:

Window 1:
Features (previous 29 days): [32.48357  29.395714 33.40659  37.858486 
 29.14184  29.20528  38.32948  34.322147 28.183279 33.283253 28.287313 
 28.303862 31.864614 21.104889 22.057411 27.875519 25.622026 32.25094  
 26.127428 23.588236 37.95459  29.468477 30.900469 23.39905  27.755371 
 30.980967 24.615065 32.186863 27.23822 ]
Target (next day): 28.71

Window 2:
Features (previous 29 days): [29.395714 33.40659  37.858486 29.14184  
 29.20528  38.32948  34.322147 28.183279 33.283253 28.287313 28.303862 
 31.864614 21.104889 22.057411 27.875519 25.622026 32.25094  26.127428 
 23.588236 37.95459  29.468477 30.900469 23.39905  27.755371 30.980967 
 24.615065 32.186863 27.23822  28.710733]
Target (next day): 27.08

Window 3:
Features (previous 29 days): [33.40659  37.858486 29.14184  29.20528  
 38.32948  34.322147 28.183279 33.283253 28.287313 28.303862 31.864614 
 21.104889 22.057411 27.875519 25.622026 32.25094  26.127428 23.588236 
 37.95459  29.468477 30.900469 23.39905  27.755371 30.980967 24.615065 
 32.186863 27.23822  28.710733 27.083256]
Target (next day): 39.27
```

要使用这些数据训练模型，您将把序列分成训练集和验证集。在这种情况下，我们将通过分割列表并将其转换为`train_dataset`和`val_dataset`子集来训练 1,000 条记录。这使用了`torch.utils.data`中的`Subset`类。

然后，你可以使用`DataLoader`加载这些数据，就像我们在第三章中看到的那样。

