
# 第三章：描述性统计和数据分析

*描述性统计*是描述数据并尽可能多地从中提取信息的领域。基本上，描述性统计可以像数据的代表一样，因为它概括了数据的倾向、行为和趋势。

交易和分析从这一领域的指标中借鉴了很多。在本章中，您将看到需要掌握数据分析的主要概念。我发现最好的教育工具是实际示例，因此我将使用一个经济数据集的示例来呈现本章。

让我们以来自美国的通胀数据为例。消费者价格指数（CPI）衡量城市消费者每月支付的一系列产品和服务的价格（这意味着每个月都会向公众发布一个新的观察值，从而形成一个连续的时间序列）。任何两个时间段之间的通胀率是通过价格指数的百分比变化来衡量的。例如，如果去年面包的价格是 1.00 美元，今天的价格是 1.01 美元，那么通胀率为 1.00%。

您可以使用的获取 CPI 数据的代码类似于您在第一章中用来获取 VIX 数据的代码。

```py
# Importing the required library
import pandas_datareader as pdr

# Setting the beginning and end of the historical data
start_date = '1950-01-01'
end_date   = '2023-01-23'

# Creating a dataframe and downloading the CPI data using its code name and its source
cpi = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)

# Printing the latest five observations of the dataframe
print(cpi.tail())

# Importing the required library
import pandas as pd

# Checking if there are NaN values in the CPI dataframe previously defined
count_nan = cpi['CPIAUCSL'].isnull().sum()

# Printing the result
print('Number of NaN values in the CPI dataframe: ' + str(count_nan))

# Dropping the NaN values from the rows
cpi = cpi.dropna()

# Transforming the CPI into a year-on-year measure 
cpi = cpi.pct_change(periods = 12, axis = 0) * 100
cpi = cpi.dropna()
```

到目前为止，您应该有一个包含 CPI 年度变化的数据框架。年度变化是 CPI 上最常见的转换，因为它清晰简单地衡量了总体价格水平的变化，足够长的时间来考虑短期波动和季节性影响（回想一下面包的例子）。

因此，CPI 的年度变化作为通胀总体趋势的一个标尺。它也简单易懂，可以与其他国家和历史时期进行比较，因此在政策制定者和经济学家中很受欢迎（尽管在不同国家之间的篮子中元素权重存在缺陷）。让我们从统计角度分析数据集。

## 中心趋势度量

*中心趋势*指的是将数据集总结为可以代表它们的值的计算。第一个和最为人熟知的中心趋势度量是平均值（平均数）。*平均值*就是值的总和除以它们的数量。它是数据集的中心点，很可能是最能代表它的值。平均值的数学公式如下：

<math alttext="x overbar equals StartFraction 1 Over n EndFraction sigma-summation Underscript n Overscript i equals 1 Endscripts x Subscript i Baseline equals StartFraction 1 Over n EndFraction left-parenthesis x 1 plus period period period plus x Subscript n Baseline right-parenthesis"><mrow><mover accent="true"><mi>x</mi> <mo>¯</mo></mover> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="false"><mfrac><mn>1</mn> <mi>n</mi></mfrac></mstyle> <msubsup><mo>∑</mo> <mrow><mi>n</mi></mrow> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow></msubsup> <msub><mi>x</mi> <mi>i</mi></msub> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="false"><mfrac><mn>1</mn> <mi>n</mi></mfrac></mstyle> <mrow><mo>(</mo> <msub><mi>x</mi> <mn>1</mn></msub> <mo>+</mo> <mo>.</mo> <mo>.</mo> <mo>.</mo> <mo>+</mo> <msub><mi>x</mi> <mi>n</mi></msub> <mo>)</mo></mrow></mrow></math>

让我们以两个数据集的简单示例为例。假设您想要计算数据集 A 和数据集 B 的平均值。您会如何做？

+   数据集 A = [1, 2, 3, 4, 5]
+   数据集 B = [1, 1, 1, 1]

数据集 A 包含 5 个值（数量），总和为 15。这意味着平均值等于 3。数据集 B 包含 4 个值，总和为 4。这意味着平均值等于 1。

###### 注意

当数据集中的所有值相同时，平均值与这些值相同。

图 3-1 显示了美国 CPI 过去二十年的年度值。较高的虚线是过去二十年计算的月度平均值。较低的虚线代表零，低于此线的是通货紧缩期。

![](img/dlf_0220.png)

###### 图 3-1。美国 CPI 过去二十年的年度变化

您可以使用以下代码创建图 3-1：

```py
# Calculating the mean of the CPI over the last 10 years
cpi_last_ten_years = cpi.iloc[-240:]
mean = cpi_last_ten_years["CPIAUCSL"].mean()

# Printing the result
print('The mean of the dataset: ' + str(mean), '%')

# Importing the required library
import matplotlib.pyplot as plt

# Plotting the latest observations in black with a label
plt.plot(cpi_last_ten_years[:], color = 'black', linewidth = 1.5, label = 'Change in CPI Year-on-Year')

# Plotting horizontal lines that represent the mean and the zero threshold
plt.axhline(y = mean, color = 'red', linestyle = 'dashed', label = '10-Year Mean')
plt.axhline(y = 0, color = 'blue', linestyle = 'dashed', linewidth = 1)

# Calling a grid to facilitate the visual component
plt.grid()

# Calling the legend function so it appears with the chart
plt.legend()
```

平均值的输出应该如下。

```py
The mean of the dataset: 2.4794 %
```

这意味着 CPI 每年变化的平均观察值约为 2.50%。尽管美联储没有明确的通胀目标，但普遍认为存在一致意见，即维持通胀年度变化在 2.00%左右，因此与历史观察值相差不远。由于 2021 年以来政治和经济动荡导致的最近高通胀数字，有必要回归到平均水平以稳定当前局势。这个例子为过去 10 年所谓的正常水平（~2.50%）提供了一个数值价值。

显然，在 2023 年初高通胀率（约 6.00%）的情况下，情况与正常有些偏离，但是偏离多少？这个问题将在下一节中讨论变异性度量时得到回答。现在，让我们继续讨论中心趋势。

下一个度量是*中位数*，简单来说就是将数据集分成两个相等的部分的值。换句话说，如果您按升序排列数据集，中间的值就是中位数。当数据分布中有很多离群值或偏斜时（可能会使均值偏差较大并且不够代表性），中位数就会被使用。

通常与计算中位数相关的有两个主题，第一个与包含偶数个值的数据集有关（例如，24 行），第二个与包含奇数个值的数据集有关（例如，47 行）：

计算偶数数据集的中位数

如果排列好的数据集中有偶数个值，中位数就是两个中间值的平均值。

计算奇数数据集的中位数

如果排列好的数据集中有一个奇数个值，中位数就是中间的值。

让我们以两个数据集的简单示例为例。假设您想要在数据集 A 和数据集 B 上计算中位数。您会如何做呢？

+   数据集 A = [1, 2, 3, 4, 5]
+   数据集 B = [1, 2, 3, 4]

数据集 A 包含五个值，是奇数个。这意味着中间的值就是中位数。在这种情况下，中位数是 3（注意它也是数据集的均值）。数据集 B 包含四个值，是偶数个。这意味着两个中间值的平均值就是中位数。在这种情况下，中位数是 2.5，即 2 和 3 的平均值。

图 3-2 显示了过去二十年美国 CPI 同比值。较高的虚线是过去二十年计算出的月中位数。较低的虚线代表零。基本上，这就像图 3-1，但是用中位数而不是均值绘制。

![](img/dlf_0225.png)

###### 图 3-2. 过去二十年美国 CPI 同比变化与中位数

您可以使用以下代码创建图 3-2：

```py
# Calculating the median of the dataset
median = cpi_last_ten_years["CPIAUCSL"].median() 

# Printing the result
print('The median of the dataset: ' + str(median), '%')

# Plotting the latest observations in black with a label
plt.plot(cpi_last_ten_years[:], color = 'black', linewidth = 1.5, label = 'Change in CPI Year-on-Year')

plt.axhline(y = median, color = 'red', linestyle = 'dashed', label = '10-Year Median')
plt.axhline(y = 0, color = 'blue', linestyle = 'dashed', linewidth = 1)

# Calling a grid to facilitate the visual component
plt.grid()

# Calling the legend function so it appears with the chart
plt.legend()
```

中位数的输出应该如下：

```py
The median of the dataset: 2.1143 %
```

显然，中位数受最近来自异常环境的离群值的影响较小。中位数约为 2.10%，更符合隐含目标 2.00%。

###### 注意

请记住，第六章将为您提供有关本章中看到的 Python 代码片段的所有必要信息，因此如果您对编码概念感到困惑，也不必担心。

本节中最后一个中心趋势度量是众数。*众数*是最常见的值（但在数据分析中使用最少）。

让我们以两个数据集的简单示例为例。假设您想要在以下数据集上计算众数。您会如何做呢？

+   数据集 A = [1, 2, 2, 4, 5]
+   数据集 B = [1, 2, 3, 4]
+   数据集 C = [1, 1, 2, 2, 3]

数据集 A 包含两次值为 2，这使其成为众数。数据集 B 没有众数，因为每个值只出现一次。数据集 C 是多峰的，因为它包含多个众数（即 1 和 2）。

###### 注意

众数在分类变量（如信用评级）中很有用，而不是连续变量（如价格和收益时间序列）

在分析时间序列时，您不太可能使用众数，因为均值和中位数更有用。以下是一些在金融分析中使用均值和中位数的示例：

+   计算价格数据的移动均值（平均值）以检测潜在趋势。
+   计算价格衍生指标的滚动中位数以了解其中性区域。
+   使用历史均值计算证券的预期收益。
+   通过比较均值和中位数来检查收益分布的正态性。

对于中心趋势指标的讨论非常重要，特别是均值和中位数不仅作为独立指标被广泛使用，而且作为更复杂的测量方法的组成部分。

###### 注意

本节的主要要点如下：

+   主要有三种中心趋势测量：均值、中位数和众数。
+   均值是总和除以数量，而中位数是将数据一分为二的值。众数是数据集中最频繁出现的值。

## 变异性测量

*变异性测量*描述了数据集中值相对于中心趋势测量的分散程度。最早也是最为人熟知的变异性测量是方差。*方差*描述了一组数字相对于它们的均值的变异性。方差公式背后的思想是确定每个数据点离均值有多远，然后平方这些偏差以确保所有数字都是正数（这是因为距离不能为负）。

找到方差的公式如下：

<math alttext="sigma squared equals StartFraction 1 Over n EndFraction sigma-summation Underscript n Overscript i equals 1 Endscripts left-parenthesis x Subscript i Baseline minus x overbar right-parenthesis squared"><mrow><msup><mi>σ</mi> <mn>2</mn></msup> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="false"><mfrac><mn>1</mn> <mi>n</mi></mfrac></mstyle> <msubsup><mo>∑</mo> <mrow><mi>n</mi></mrow> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow></msubsup> <msup><mrow><mo>(</mo><msub><mi>x</mi> <mi>i</mi></msub> <mo>-</mo><mover accent="true"><mi>x</mi> <mo>¯</mo></mover><mo>)</mo></mrow> <mn>2</mn></msup></mrow></math>

这个公式背后的直觉是计算每个数据点与均值之间的平方偏差的总和，从而给出不同距离的观察结果，然后计算这些距离观察结果的均值。

让我们以两个数据集的简单示例为例。假设您想要计算数据集 A 和数据集 B 的方差。您会如何做？

+   数据集 A = [1, 2, 3, 4, 5]
+   数据集 B = [5, 5, 5, 5]

第一步是计算数据集的均值，因为这是您将计算数据离散度的基准。数据集 A 的均值为 3。下一步是逐步使用方差公式如下：

<math alttext="left-parenthesis x 1 minus x overbar right-parenthesis squared equals left-parenthesis 1 minus 3 right-parenthesis squared equals 4"><mrow><msup><mrow><mo>(</mo><msub><mi>x</mi> <mn>1</mn></msub> <mo>-</mo><mover accent="true"><mi>x</mi> <mo>¯</mo></mover><mo>)</mo></mrow> <mn>2</mn></msup> <mo>=</mo> <msup><mrow><mo>(</mo><mn>1</mn><mo>-</mo><mn>3</mn><mo>)</mo></mrow> <mn>2</mn></msup> <mo>=</mo> <mn>4</mn></mrow></math>

<math alttext="left-parenthesis x 2 minus x overbar right-parenthesis squared equals left-parenthesis 2 minus 3 right-parenthesis squared equals 1"><mrow><msup><mrow><mo>(</mo><msub><mi>x</mi> <mn>2</mn></msub> <mo>-</mo><mover accent="true"><mi>x</mi> <mo>¯</mo></mover><mo>)</mo></mrow> <mn>2</mn></msup> <mo>=</mo> <msup><mrow><mo>(</mo><mn>2</mn><mo>-</mo><mn>3</mn><mo>)</mo></mrow> <mn>2</mn></msup> <mo>=</mo> <mn>1</mn></mrow></math>
