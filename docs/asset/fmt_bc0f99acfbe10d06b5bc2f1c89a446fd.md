

# 第十三章：时间序列预测

> 原文：[`deeplearningwithpython.io/chapters/chapter13_timeseries-forecasting`](https://deeplearningwithpython.io/chapters/chapter13_timeseries-forecasting)

本章探讨时间序列，其中时间顺序至关重要。我们将重点关注最常见且最有价值的时间序列任务：预测。使用最近的过去来预测近未来的能力非常强大，无论您是试图预测能源需求、管理库存还是简单地预测天气。

## 不同类型的时间序列任务

一个*时间序列*可以是任何通过定期测量获得的数据，例如股票的每日价格、城市的每小时电力消耗或商店的每周销售额。时间序列无处不在，无论是观察自然现象（如地震活动、河流中鱼类种群的变化或某地的天气）还是人类活动模式（如网站访问者、一个国家的 GDP 或信用卡交易）。与您迄今为止遇到的数据类型不同，处理时间序列需要理解系统的*动态*——它的周期性循环、随时间的变化趋势、常规状态和突然的峰值。

到目前为止，最常见的时间序列相关任务是*预测*：预测序列中接下来会发生什么。提前几小时预测电力消耗，以便您可以预测需求；提前几个月预测收入，以便您可以规划预算；提前几天预测天气，以便您可以规划日程。预测是本章的重点。但实际上，您可以用时间序列做很多事情，例如：

+   *异常检测*——在连续数据流中检测任何异常事件。公司网络上的异常活动？可能是攻击者。生产线上的异常读数？是时候让人类去看看了。异常检测通常通过无监督学习来完成，因为您通常不知道您在寻找哪种异常，因此您无法在特定的异常示例上进行训练。

+   *分类*——将一个或多个分类标签分配给时间序列。例如，给定网站访问者活动的时间序列，判断该访问者是机器人还是人类。

+   *事件检测*——在连续数据流中识别特定、预期的事件发生。特别有用的应用是“热词检测”，其中模型监控音频流并检测“OK, Google”或“Hey, Alexa”等语音。

在本章中，您将了解循环神经网络（RNNs）及其在时间序列预测中的应用。

## 一个温度预测示例

在本章中，我们所有的代码示例都将针对一个问题：根据最近过去由一组安装在建筑物屋顶的传感器记录的诸如大气压力和湿度等量的每小时测量值的时间序列，预测未来 24 小时的温度。正如你将看到的，这是一个相当具有挑战性的问题！

我们将使用这个温度预测任务来突出时间序列数据与迄今为止你遇到的数据集的基本不同之处，以表明密集连接的网络和卷积网络并不适合处理它，并展示一种新的机器学习技术，这种技术在处理这类问题上表现得非常出色：循环神经网络（RNNs）。

我们将使用德国耶拿马克斯·普朗克生物地球化学研究所气象站记录的气象时间序列数据集进行工作.^([[1]](#footnote-1)) 在这个数据集中，记录了 14 个不同的量（如温度、大气压力、湿度、风向等），每隔 10 分钟记录一次，持续了数年。原始数据可以追溯到 2003 年，但我们将要下载的数据子集限制在 2009-2016 年。

让我们开始下载和解压缩数据：

```py
!wget https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip
!unzip jena_climate_2009_2016.csv.zip 
```

让我们来看看数据。

```py
import os

fname = os.path.join("jena_climate_2009_2016.csv")

with open(fname) as f:
    data = f.read()

lines = data.split("\n")
header = lines[0].split(",")
lines = lines[1:]
print(header)
print(len(lines)) 
```

代码列表 13.1：检查耶拿气象数据集的数据

这输出了 420,551 行数据的计数（每行是一个时间步长：一个日期和 14 个与天气相关的值的记录），以及以下标题：

```py
["Date Time",
 "p (mbar)",
 "T (degC)",
 "Tpot (K)",
 "Tdew (degC)",
 "rh (%)",
 "VPmax (mbar)",
 "VPact (mbar)",
 "VPdef (mbar)",
 "sh (g/kg)",
 "H2OC (mmol/mol)",
 "rho (g/m**3)",
 "wv (m/s)",
 "max. wv (m/s)",
 "wd (deg)"] 
```

现在，将所有 420,551 行数据转换为 NumPy 数组：一个数组用于温度（以摄氏度为单位），另一个数组用于其余的数据——我们将使用这些特征来预测未来的温度。请注意，我们丢弃了“日期时间”列。

```py
import numpy as np

temperature = np.zeros((len(lines),))
raw_data = np.zeros((len(lines), len(header) - 1))

for i, line in enumerate(lines):
    values = [float(x) for x in line.split(",")[1:]]
    # We store column 1 in the temperature array.
    temperature[i] = values[1]
    # We store all columns (including the temperature) in the raw_data
    # array.
    raw_data[i, :] = values[:] 
```

代码列表 13.2：解析数据

图 13.1 显示了温度（以摄氏度为单位）随时间变化的图表。在这张图上，你可以清楚地看到温度的年度周期性——数据跨度为八年。

```py
from matplotlib import pyplot as plt

plt.plot(range(len(temperature)), temperature) 
```

代码列表 13.3：绘制温度时间序列图

![](img/149d51397c1deac9439c06ada3e22192.png)

图 13.1：数据集整个时间范围内的温度（ºC）

图 13.2 显示了温度数据的前 10 天的更窄的图表。因为数据每 10 分钟记录一次，所以每天有 24 × 6 = 144 个数据点。

```py
plt.plot(range(1440), temperature[:1440]) 
```

代码列表 13.4：绘制温度时间序列的前 10 天

![](img/255b274a022adb5680908111339b3299.png)

图 13.2：数据集前 10 天的温度（ºC）

在这张图上，你可以看到日周期性，特别是在最后四天尤为明显。同时请注意，这个 10 天的周期必须来自一个相当寒冷的冬季月份。

在我们的数据集中，如果您试图根据几个月的过去数据预测下一个月的平均温度，这个问题将很容易，因为数据的可靠年周期性。但是，从日的时间尺度来看，温度看起来要混乱得多。这个时间序列在日尺度上可预测吗？让我们来看看。

在我们所有的实验中，我们将使用数据的前 50%进行训练，接下来的 25%进行验证，最后的 25%进行测试。当处理时间序列数据时，使用比训练数据更近期的验证和测试数据非常重要，因为您试图根据过去预测未来，而不是相反，并且您的验证/测试拆分应该反映这种时间顺序。如果反转时间轴，某些问题可能会变得相当简单！

```py
>>> num_train_samples = int(0.5 * len(raw_data))
>>> num_val_samples = int(0.25 * len(raw_data))
>>> num_test_samples = len(raw_data) - num_train_samples - num_val_samples
>>> print("num_train_samples:", num_train_samples)
>>> print("num_val_samples:", num_val_samples)
>>> print("num_test_samples:", num_test_samples)
num_train_samples: 210225
num_val_samples: 105112
num_test_samples: 105114
```

列表 13.5：计算每个数据拆分的样本数量

### 准备数据

问题的确切表述如下：给定覆盖前五天且每小时采样一次的数据，我们能否预测 24 小时后的温度？

首先，让我们预处理数据，使其成为神经网络可以摄入的格式。这很简单：数据已经是数值的，因此您不需要进行任何向量化。但是，数据中的每个时间序列都在不同的尺度上（例如，大气压力，以毫巴为单位，约为 1,000，而 H2OC，以每摩尔毫摩尔为单位，约为 3）。我们将独立归一化每个时间序列，使它们都在相似的尺度上取小值。我们将使用前 210,225 个时间步作为训练数据，因此我们只计算这个数据分量的均值和标准差。

```py
mean = raw_data[:num_train_samples].mean(axis=0)
raw_data -= mean
std = raw_data[:num_train_samples].std(axis=0)
raw_data /= std 
```

列表 13.6：数据归一化

接下来，让我们创建一个`Dataset`对象，它提供过去五天的数据批次以及未来 24 小时的温度目标。由于数据集中的样本高度冗余（样本`N`和样本`N + 1`将有大部分时间步长是相同的），明确为每个样本分配内存将是浪费的。相反，我们将动态生成样本，同时只在内存中保留原始`raw_data`和`temperature`数组，不再需要其他任何东西。

我们可以轻松地编写一个 Python 生成器来完成这项工作，但 Keras 中有一个内置的数据集实用工具可以做到这一点（`timeseries_dataset_from_array()`），因此我们可以通过使用它来节省一些工作。您通常可以使用它来完成任何类型的时序预测任务。

我们将使用`timeseries_dataset_from_array`来实例化三个数据集：一个用于训练，一个用于验证，一个用于测试。

我们将使用以下参数值：

+   `sampling_rate = 6` — 观测将以每小时一个数据点的频率进行采样：我们将只保留六个数据点中的一个。

+   `sequence_length = 120` — 观测将回溯五天（120 小时）。

+   `delay = sampling_rate * (sequence_length + 24 - 1)` — 序列的目标将是序列结束后的 24 小时温度。

+   `start_index = 0`和`end_index = num_train_samples` — 对于训练数据集，仅使用前 50%的数据。

+   `start_index = num_train_samples`和`end_index = num_train_samples + num_val_samples` — 对于验证数据集，仅使用接下来的 25%的数据。

+   `start_index = num_train_samples + num_val_samples` — 对于测试数据集，使用剩余的样本。

```py
sampling_rate = 6
sequence_length = 120
delay = sampling_rate * (sequence_length + 24 - 1)
batch_size = 256

train_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0,
    end_index=num_train_samples,
)

val_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples,
)

test_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples + num_val_samples,
) 
```

列表 13.7：实例化用于训练、验证和测试的数据集

每个数据集都产生一个元组`(samples, targets)`，其中`samples`是包含 256 个样本的批次，每个样本包含 120 个连续小时的输入数据，而`targets`是相应的 256 个目标温度数组。请注意，样本是随机打乱的，因此批次中的连续序列（如`samples[0]`和`samples[1]`）不一定在时间上接近。

```py
>>> for samples, targets in train_dataset:
>>>     print("samples shape:", samples.shape)
>>>     print("targets shape:", targets.shape)
>>>     break
samples shape: (256, 120, 14)
targets shape: (256,)
```

列表 13.8：检查数据集

### 常识性、非机器学习基线

在您开始使用黑盒深度学习模型来解决温度预测问题之前，让我们尝试一种简单、常识性的方法。这将作为合理性检查，并建立一个基线，您必须超越这个基线才能证明更高级、基于机器学习模型的实用性。这种常识性基线在您面对一个尚未找到已知解决方案的新问题时可能很有用。一个经典的例子是不平衡分类任务，其中某些类别比其他类别更常见。如果您的数据集中包含 90%的类别 A 实例和 10%的类别 B 实例，那么对分类任务的常识性方法是，在呈现新样本时始终预测“A”。这样的分类器总体准确率为 90%，因此任何基于学习的算法都应该超越这个 90%的分数以证明其有用性。有时，这样的基本基线可能难以超越。

在这个情况下，可以安全地假设温度时间序列是连续的（明天的温度很可能接近今天的温度），并且具有每日周期性。因此，一种常识性的方法是始终预测 24 小时后的温度将与现在的温度相等。让我们使用以下定义的均方误差（MAE）指标来评估这种方法：

```py
np.mean(np.abs(preds - targets)) 
```

这里是评估循环。

```py
def evaluate_naive_method(dataset):
    total_abs_err = 0.0
    samples_seen = 0
    for samples, targets in dataset:
        # The temperature feature is at column 1, so `samples[:, -1,
        # 1]` is the last temperature measurement in the input
        # sequence. Recall that we normalized our features to retrieve
        # a temperature in Celsius degrees, we need to un-normalize it,
        # by multiplying it by the standard deviation and adding back
        # the mean.
        preds = samples[:, -1, 1] * std[1] + mean[1]
        total_abs_err += np.sum(np.abs(preds - targets))
        samples_seen += samples.shape[0]
    return total_abs_err / samples_seen

print(f"Validation MAE: {evaluate_naive_method(val_dataset):.2f}")
print(f"Test MAE: {evaluate_naive_method(test_dataset):.2f}") 
```

列表 13.9：计算常识性基线 MAE

这个常识性基线实现了 2.44 摄氏度的验证 MAE 和 2.62 摄氏度的测试 MAE。所以如果你总是假设 24 小时后的温度将与现在相同，你平均会差两度半。这并不太糟糕，但你可能不会基于这个启发式方法推出天气预报服务。现在，游戏规则是利用你对深度学习的知识做得更好。

