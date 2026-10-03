
```py
# Importing the required libraries
import pandas_datareader as pdr
import pandas as pd

# Setting the beginning and end of the historical data
start_date = '1950-01-01'
end_date   = '2022-12-01'

# Creating a dataframe and downloading the CPI data using its code name and its source 
cpi = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)

# Dropping the NaN values from the rows
cpi = cpi.dropna()

# Transforming the US CPI into a year-on-year measure
cpi = cpi.pct_change(periods = 12, axis = 0) * 100
cpi = cpi.dropna()

# Transforming the data frame to a series structure
cpi = cpi.iloc[:,0]

# Calculating autocorrelation with a lag of 1
print('Correlation with a lag of 1 = ', round(cpi.autocorr(lag = 1), 2))

# Calculating autocorrelation with a lag of 6
print('Correlation with a lag of 6 = ', round(cpi.autocorr(lag = 6), 2))

# Calculating autocorrelation with a lag of 12
print('Correlation with a lag of 12 = ', round(cpi.autocorr(lag = 12), 2))
```

滞后 12 意味着每个数据与十二个周期前的数据进行比较，然后计算一个计算度量。代码的输出如下：

```py
Correlation with a lag of 1 =  0.99
Correlation with a lag of 6 =  0.89
Correlation with a lag of 12 =  0.73
```

现在，在继续下一节之前，让我们回到信息论并讨论一个能够捕捉非线性关系的有趣相关系数。其中一种方式是最大信息系数（MIC）。

`最大信息系数`（MIC）是两个变量之间关联的非参数测量，旨在处理大型和复杂数据。它通常被视为传统相关性测量的更稳健的替代方法，如皮尔逊相关性和斯皮尔曼秩相关性。由`Reshef 等人`引入，MIC 使用了信息论中的概念，这些概念在第二章中已经介绍过。

MIC 通过计算列联表中的单元格数量来衡量两个变量之间的关联强度，这些单元格对于变量之间的关系是最具信息量的。MIC 值的范围从 0 到 1，数值越高表示关联越强。它可以处理高维数据，并且可以识别变量之间的非线性关系。然而，它是非方向性的，这意味着接近 1 的值仅表明两个变量之间存在强相关性，但并不说明相关性是正向还是负向。

###### 注意

换句话说，在将每个变量的范围划分为一组箱后，计算每个箱内两个变量之间的互信息。

然后，通过跨所有箱子中的最大互信息值来估计两个变量之间的关联强度。

让我们看一个展示 MIC 在检测非线性关系方面的强大性的实际示例。以下示例模拟了正弦和余弦时间序列。直观地看，从图 3-17 可以看出，两者之间存在滞后-领先关系。

![](img/dlf_0230.png)

###### 图 3-17。显示一种非线性关系形式的两个波浪序列的强度

以下 Python 代码片段创建了两个时间序列并绘制了图 3-17：

```py
# Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt

# Setting the range of the data
data_range = np.arange(0, 30, 0.1)

# Creating the sine and the cosine waves
sine = np.sin(data_range)
cosine = np.cos(data_range)

# Plotting
plt.plot(sine, color = 'black', label = 'Sine Function')
plt.plot(cosine, color = 'grey', linestyle = 'dashed', label = 'Cosine Function')
plt.grid()
plt.legend()
```

现在，任务是计算三个相关性度量并分析它们的结果。可以使用以下代码完成这项工作：

```py
# Importing the libraries
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from minepy import MINE

# Calculating the linear correlation measures
print('Correlation | Pearson: ', round(pearsonr(sine, cosine)[0], 3))
print('Correlation | Spearman: ', round(spearmanr(sine, cosine)[0], 3))

# Calculating the MIC
mine = MINE(alpha = 0.6, c = 15)
mine.compute_score(sine,cosine)
MIC = mine.mic()
print('Correlation | MIC: ', round(MIC, 3))
```

请注意，由于代码创建了一个数组（而不是数据框），在计算测量之前必须导入所需的库。这将在下一章中明确说明。以下是代码的输出：

```py
Correlation | Pearson:  0.035
Correlation | Spearman:  0.027
Correlation | MIC: 0.602
```

让我们解释一下结果：

+   `Pearson 相关性`：由于缺少非线性关联，这里没有任何类型的相关性。

+   `Spearman 相关性`：在这里也适用于极弱的相关性。

+   `MIC`：该测量返回了两者之间 0.60 的强关系，更接近现实。看起来 MIC 表明这两个波浪之间有强关系，尽管是非线性的。

###### 注意

您可能需要更新 Microsoft Visual C++（至少版本 14.0 或更高版本）以避免在尝试运行`minepy`库时出现任何错误。

如果正确使用，MIC 在经济分析、金融分析甚至寻找交易信号中非常有用。在这些复杂领域中非线性关系很常见，能够检测到这些关系可能会带来可观的优势。

###### 注意

相关性部分的主要要点如下：

+   相关性是用于计算两个变量之间线性关系程度的度量。它是一个介于 -1.0 和 1.0 之间的数字，-1.0 表示变量之间强烈的负相关关系，1.0 表示变量之间强烈的正相关关系。

+   有两种主要类型的相关性测量，Spearman 和 Pearson。它们都有各自的优点和局限性。

+   自相关是变量与其自身滞后值的相关性。例如，如果 Nvidia 的股票回报的自相关是正的，则表示趋势配置。

+   当您使用正确的工具时，相关性测量也可以指非线性关系，例如 MIC。

## 平稳性概念

平稳性是统计分析和机器学习中的关键概念之一。`平稳性`是指时间序列的统计特征（均值、方差等）随时间保持恒定。换句话说，当沿时间绘制数据时，没有可辨认的趋势。

不同的学习模型依赖于数据的平稳性，因为这是统计建模的基础之一，主要是为了简化。在金融领域，价格时间序列不是平稳的，因为它们显示具有不同方差（波动性）的趋势。看一下图 3-18，看看您能否检测到趋势。您会说这个时间序列是平稳的吗？

![](img/dlf_0211.png)

###### 图 3-18。模拟数据随时间变化的均值

自然地，答案是否定的，因为一个上升趋势明显正在进行中。这种状态对于统计分析和机器学习是不可取的。幸运的是，您可以对时间序列应用转换使其平稳。但首先，让我们看看如何以数学方式检查平稳性，因为视觉方式并不能证明任何事情。处理数据平稳性问题的正确过程是遵循以下步骤：

1. 使用本节中将看到的不同统计测试来检查平稳性。

2. 如果测试显示数据平稳性，您就可以准备使用数据进行不同的学习算法。如果测试显示数据不平稳，您必须继续进行下一步。

3. 应用您将在本节中看到的价格转换技术。

4. 使用相同测试在新转换的数据上重新检查平稳性。

5. 如果测试显示数据平稳性，那么您已成功转换了数据。否则，请重新进行转换并再次检查，直到您获得平稳数据。

###### 注意

升序或降序时间序列随时间变化的均值和方差，因此很可能是非平稳的。当然也有例外情况，稍后您将看到原因。

请记住，平稳性的目标是随时间稳定和恒定的均值和方差。因此，当您看图 3-19 时，您能说些什么？

![](img/dlf_0210.png)

###### 图 3-19。模拟数据在时间上均值围绕零

从视觉上看，数据似乎没有趋势，看起来围绕着一个稳定的均值波动，具有稳定的方差。第一印象是数据是平稳的。当然，这必须通过统计测试来证明。

第一个和最基本的测试是`增广迪基-富勒`（ADF）测试。ADF 测试使用假设检验来检验平稳性。

ADF 测试在数据中寻找单位根。`单位根`是非平稳数据的一个特性，在时间序列分析的背景下，指的是随机过程的一个特征，其中系列的根等于 1。简单来说，这意味着其统计特性，如均值和方差，会随时间变化。以下是你需要知道的内容：

+ 零假设假定存在单位根。这意味着如果你试图证明数据是平稳的，你要拒绝零假设（如第二章的假设检验部分所示）。

+ 因此，备择假设是不存在单位根，数据是平稳的。

+ 从测试中获得的 p 值必须小于所选的显著性水平（在大多数情况下，为 5%）。

让我们拿美国 CPI 的年度数据进行平稳性测试。以下代码片段使用 ADF 测试检查平稳性：

```py
# Importing the required libraries
from statsmodels.tsa.stattools import adfuller
import pandas_datareader as pdr

# Setting the beginning and end of the historical data
start_date = '1950-01-01'
end_date   = '2022-12-01'

# Creating a dataframe and downloading the CPI data using its code name and its source 
cpi = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)

# Dropping the NaN values from the rows
cpi = cpi.dropna()

# Transforming the US CPI into a year-on-year measure
cpi = cpi.pct_change(periods = 12, axis = 0) * 100
cpi = cpi.dropna()

# Applying the ADF test on the CPI data
adfuller(cpi) 
print('p-value: %f' % adfuller(cpi)[1])
```

代码的输出如下：

```py
p-value: 0.0152
```

假设 5%的显著性水平，似乎可以接受年度数据是平稳的（然而，如果你想更严格一些，使用 1%的显著性水平，那么 p 值表明数据是非平稳的）。无论如何，即使看图表也会让你感到困惑。请记住，在图 3-11 中，美国 CPI 的年度变化似乎是稳定的，但并不像是平稳数据。这就是为什么要使用数值和统计测试的原因。

现在，让我们做同样的事情，但省略年度变化。换句话说，对美国 CPI 原始数据应用代码，不考虑年度变化。以下是代码：

```py
# Importing the required libraries
from statsmodels.tsa.stattools import adfuller
import pandas_datareader as pdr

# Setting the beginning and end of the historical data
start_date = '1950-01-01'
end_date   = '2022-12-01'

# Creating a dataframe and downloading the CPI data using its code name and its source 
cpi = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)

# Dropping the NaN values from the rows
cpi = cpi.dropna()

# Applying the ADF test on the CPI data
adfuller(cpi) 
print('p-value: %f' % adfuller(cpi)[1])
```

代码的输出如下：

```py
p-value: 0.999
```

显然，p 值大于所有显著性水平，这意味着时间序列是非平稳的。让我们总结一下这些结果：

+ 似乎在美国 CPI 的年度变化方面，可以使用 5%的显著性水平拒绝零假设。数据集被假定为平稳。

+ 似乎在美国 CPI 的原始数值方面，无法使用 5%的显著性水平拒绝零假设。数据集被假定为非平稳。

当你绘制美国 CPI 的原始数值时，这一点变得明显，如图 3-20 所示。

![](img/dlf_5.png)

###### 图 3-20。美国 CPI 的绝对数值显示出明显的趋势性

你必须了解的另一个测试被称为`Kwiatkowski-Phillips-Schmidt-Shin`（KPSS），这也是一个旨在确定时间序列是平稳还是非平稳的统计测试。然而，KPSS 测试可以检测趋势时间序列中的平稳性，使其成为一个强大的工具。

趋势时间序列实际上可以是平稳的，条件是它们具有稳定的均值。

# 警告

ADF 测试有一个假设检验，主张非平稳性，另一个假设主张平稳性。另一方面，KPSS 测试有一个假设主张平稳性，另一个假设主张非平稳性。

在分析通货膨胀数据之前，让我们看看趋势时间序列如何可以是平稳的。记住，平稳性指的是稳定的均值和标准差，所以如果你有一个逐渐上升或下降的时间序列，具有稳定的统计特性，它可能是平稳的。下一个代码片段模拟了一个正弦波，然后加入了一丝趋势性：

```py
# Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt

# Creating the first time series using sine waves
length = np.pi * 2 * 5
sinewave = np.sin(np.arange(0, length, length / 1000))

# Creating the second time series using trending sine waves
sinewave_ascending = np.sin(np.arange(0, length, length / 1000))

# Defining the trend variable
a = 0.01

# Looping to add a trend factor
for i in range(len(sinewave_ascending)):

    sinewave_ascending[i] = a + sinewave_ascending[i]

    a = 0.01 + a
```

如图 3-21 所示绘制两个系列的图表显示了趋势正弦波似乎是稳定的。但让我们通过统计检验来证明这一点。

![](img/dlf_6.png)

###### 图 3-21. 具有趋势正弦波的正常正弦波模拟系列

图 3-21 是使用以下代码生成的（确保您已经使用前面的代码块定义了系列）：

```py
# Plotting the series
plt.plot(sinewave, label = 'Sine Wave', color = 'black')
plt.plot(sinewave_ascending, label = 'Ascending Sine Wave', color = 'grey')

# Calling the grid function for better interpretability
plt.grid()

# Calling the legend function to show the labels
plt.legend()

# Showing the plot
plt.show()
```

让我们对两个系列进行 ADF 测试，看看结果如何：

```py
# ADF testing | Normal sine wave
adfuller(sinewave) 
print('p-value: %f' % adfuller(sinewave)[1])

# ADF testing | Ascending sine wave
adfuller(sinewave_ascending) 
print('p-value: %f' % adfuller(sinewave_ascending)[1])
```

输出如下：
