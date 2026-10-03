
# 计算四分位数与数据可视化

在计算四分位数时要小心，因为有许多方法使用不同的计算来处理相同的数据集。最重要的是在整个分析过程中始终使用一致的方法。在表 3-1 中用于计算四分位数的方法称为*图基的铰链*方法。默认情况下，当您想使用`pandas`计算四分位数时，默认方法是*线性插值*方法，这将给出不同的结果。

这些方法之间的主要区别在于有些方法可能更适合较小的数据集或具有不同分布特征的正常大小数据集。

## 注意

本节的关键要点如下：

+   正态分布是具有钟形曲线的连续概率分布。大多数数据聚集在平均值周围。正态分布曲线的均值、中位数和众数都相等。
+   偏度度量概率分布的不对称性。
+   峰度度量概率分布的尖峰度。过度峰度通常用于描述当前概率分布。
+   分位数将排列的数据集分成相等的部分。最著名的分位数是将数据分成四个相等部分的四分位数。
+   IQR 是第三四分位数和第一四分位数之间的差异。它不受异常值的影响，因此在数据分析中非常有帮助。

## 可视化数据

如果您还记得前一章，我提出了数据科学中的六个阶段过程。第四阶段涉及数据可视化。本节将向您展示几种以清晰的视觉方式呈现数据的方法，使您能够解释数据。

有许多常用于可视化数据的统计图类型，例如散点图和折线图。让我们讨论这些图并使用相同的通货膨胀数据创建它们。

第一种数据可视化方法是*散点图*，用于通过对应于变量交点的点来绘制两个变量之间的关系。让我们使用以下代码创建和可视化一个散点图：

```py
# Importing the required library
import pandas_datareader as pdr
import pandas as pd
import matplotlib.pyplot as plt

# Setting the beginning and end of the historical data
start_date = '1950-01-01'
end_date   = '2022-12-01'

# Creating a dataframe and downloading the CPI data using its code name and its source
cpi = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)

# Transforming the CPI into a year-on-year measure
cpi = cpi.pct_change(periods = 12, axis = 0) * 100

# Dropping the NaN values
cpi = cpi.dropna()

# Resetting the index
cpi = cpi.reset_index()

# Creating the chart
fig, ax = plt.subplots()
ax.scatter(cpi['DATE'], cpi['CPIAUCSL'], color = 'black', s = 8,  label = 'Change in CPI Year-on-Year')

# Calling the grid function for better interpretability
plt.grid()

# Calling the legend function to show the labels
plt.legend()

# Showing the plot
plt.show()
```

图 3-8 显示了时间散点图的结果。这意味着您将 CPI 数据作为第一个变量（y 轴），时间作为第二个变量（x 轴）。然而，散点图更常用于比较变量，因此去除时间变量可以提供更多见解。

![](img/dlf_0228.png)

**图 3-8。** 美国 CPI 与时间轴的散点图

如果您拿英国 CPI 年同比变化与美国 CPI 年同比变化进行比较，您将得到图 3-9。请注意两者之间的正相关性，因为一个变量的较高值与另一个变量的较高值相关联。相关性是一个关键指标，您将在下一节中详细了解。绘制图 3-9 的代码如下：

```py
# Setting the beginning and end of the historical data
start_date = '1995-01-01'
end_date   = '2022-12-01'

# Creating a dataframe and downloading the CPI data using its code name and its source
cpi_us = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)
cpi_uk = pdr.DataReader('GBRCPIALLMINMEI', 'fred', start_date, end_date)

# Dropping the NaN values from the rows
cpi_us = cpi_us.dropna()
cpi_uk = cpi_uk.dropna()

# Transforming the CPI into a year-on-year measure
cpi_us = cpi_us.pct_change(periods = 12, axis = 0) * 100
cpi_us = cpi_us.dropna()

cpi_uk = cpi_uk.pct_change(periods = 12, axis = 0) * 100
cpi_uk = cpi_uk.dropna()

# Creating the chart
fig, ax = plt.subplots()
ax.scatter(cpi_us['CPIAUCSL'], cpi_uk['GBRCPIALLMINMEI'], color = 'black', s = 8, label = 'Change in CPI Year-on-Year')

# Adding a few aesthetic elements to the chart
ax.set_xlabel('US CPI')
ax.set_ylabel('UK CPI')
ax.axvline(x = 0, color='black', linestyle = 'dashed', linewidth = 1)  # vertical line
ax.axhline(y = 0, color='black', linestyle = 'dashed', linewidth = 1)  # horizontal line
ax.set_ylim(-2,)

# Calling the grid function for better interpretability
plt.grid()

# Calling the legend function to show the labels
plt.legend()

# Showing the plot
plt.show()
```

![](img/dlf_0229.png)

**图 3-9。** 英国 CPI 与美国 CPI 的散点图

散点图在可视化数据之间的相关性时很有用。它们也易于绘制和解释。通常，当点散布在这样一种方式时，即可以画出一个向上倾斜的对角线来代表它们，就假定相关性是正的，因为每当 x 轴上的变量增加时，y 轴上的变量也会增加。

另一方面，当可以画出向下倾斜的对角线来代表不同变量时，可能存在负相关性。负相关意味着每当 x 轴上的变量移动时，y 轴上的变量可能会以相反的方式移动。图 3-10 绘制了两个通货膨胀数据之间的最佳拟合线（通过代码生成）。请注意它是向上倾斜的：

![](img/dlf_4.png)

**图 3-10。** 英国 CPI 与美国 CPI 的散点图及最佳拟合线

现在让我们转向另一种图表方法。*线图*基本上是散点图，它们被连接在一起，大多数情况下绘制在时间轴（x 轴）上。您已经在之前的图表中看到过线图，比如图 3-1 和图 3-2，因为它是绘图的最基本形式。

线图的优点在于其简单性和易于实现。它们还显示了系列随时间的演变，有助于检测趋势和模式。在第五章中，您将看到一个更复杂的金融时间序列绘图版本，称为*蜡烛图*。图 3-11 显示了自 1950 年以来美国 CPI 的基本线图：

![](img/dlf_1.png)

**图 3-11。** 美国 CPI 与时间轴的线图

要创建图 3-11，您可以使用以下代码片段：

```py
# Setting the beginning and end of the historical data
start_date = '1950-01-01'
end_date   = '2022-12-01'

# Creating a dataframe and downloading the CPI data using its code name and its source
cpi = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)

# Transforming the CPI into a year-on-year measure
cpi = cpi.pct_change(periods = 12, axis = 0) * 100

# Dropping the NaN values
cpi = cpi.dropna()

# Resetting the index
cpi = cpi.reset_index()

# Creating the chart
plt.plot(cpi['DATE'], cpi['CPIAUCSL'], color = 'black', label = 'Change in CPI Year-on-Year')

# Calling the grid function for better interpretability
plt.grid()

# Calling the legend function to show the labels
plt.legend()

# Showing the plot
plt.show()
```

接下来是*条形图*，用于显示变量（通常是分类变量）的分布。图 3-12 显示了自 2022 年初以来美国 CPI 的条形图：

![](img/dlf_0778.png)

**图 3-12。** 美国 CPI 与时间轴的条形图

要创建图 3-12，您可以使用以下代码片段：

```py
# Taking the values of the previous twelve months
cpi_one_year = cpi.iloc[-12:]

# Creating the chart
plt.bar(cpi_one_year['DATE'], cpi_one_year['CPIAUCSL'], color = 'black', label = 'Change in CPI Year-on-Year', width = 7)

# Calling the grid function for better interpretability
plt.grid()

# Calling the legend function to show the labels
plt.legend()

# Showing the plot
plt.show()
```

条形图易于实现，功能多样。然而，对于绘制美国 CPI 或股票价格等连续数据，它们可能存在局限性。当比例失调时，它们也可能具有误导性。由于它们会占用空间，因此不建议在大型数据集上使用条形图。基于后一原因，直方图更适合。

*直方图*是一种特定类型的条形图，用于通过使用条形来表示统计信息来显示连续数据的频率分布。它指示落入值类别或区间的观测数量。直方图的一个示例是图 3-13（以及上一节的图 3-7）：

![](img/dlf_2.png)

**图 3-13。** 美国 CPI 的直方图

```py
# Creating the chart
fig, ax = plt.subplots()
ax.hist(cpi['CPIAUCSL'], bins = 30, edgecolor = 'black', color = 'white', label = 'Change in CPI Year-on-Year',)

# Add vertical lines for better interpretation
ax.axvline(0, color='black')

# Calling the grid function for better interpretability
plt.grid()

# Calling the legend function to show the labels
plt.legend()

# Showing the plot
plt.show()
```

请注意，条形图是根据时间轴绘制的，而直方图没有时间范围，因为它是一组值，旨在显示整体分布点。从视觉上，您可以看到分布的正偏度。

## 注意

*分类变量*的一个示例是性别，而*连续变量*的一个示例是商品价格。

统计学中另一种经典绘图技术是著名的*箱线图*。它用于可视化连续变量的分布，同时包括中位数和四分位数，以及异常值。理解箱线图的方法如下：

+   箱子代表 IQR。箱子在第一四分位数和第三四分位数之间绘制。箱子的高度表示该范围内数据的传播。
+   箱子内部的线代表中位数。
+   须在箱子的顶部和底部延伸到仍在 1.5 倍 IQR 内的最高和最低数据点。这些数据点称为*异常值*，并在图中表示为单独的点。

图 3-14 显示了自 1950 年以来美国 CPI 的箱线图：

![](img/dlf_080.png)

**图 3-14。** 美国 CPI 的箱线图

您还可以绘制没有异常值的图（任何值距离箱子的任一端超过箱子长度的一倍半）。要创建图 3-14，您可以使用以下代码片段：

```py
# Taking the values of the last twenty years
cpi_last_ten_years = cpi.iloc[-240:]

# Creating the chart
fig, ax = plt.subplots()
ax.boxplot(cpi_last_ten_years['CPIAUCSL'])

# Calling the grid function for better interpretability
plt.grid()

# Calling the legend function to show the labels
plt.legend()

# Showing the plot
plt.show()
```

要从图中去除异常值，您只需使用以下调整：

```py
# Replace the corresponding code line with the following
ax.boxplot(cpi_last_ten_years['CPIAUCSL'], showfliers = False)
```

这将给您图 3-15：

![](img/dlf_0.81.png)

**图 3-15。** 美国 CPI 的箱线图，没有异常值

存在许多其他数据可视化技术，例如*热图*（通常与相关数据和温度映射一起使用）和*饼图*（通常用于预算和分割）。这完全取决于您需要理解的内容以及哪种更适合您的需求。例如，线图更适合只有一个特征的时间序列（例如，只有某种证券的收盘价可用）。直方图图更适合概率分布数据。

## 注意

让我们总结一下您需要记住的一切：

+   数据可视化取决于您想要进行的分析和解释的类型。某些图适合某些类型的数据。
+   在数值上确认之前，数据可视化有助于对数据进行初步解释。
+   处理金融时间序列时，您更有可能使用线图和蜡烛图。

## 相关性

*相关性*是用于计算两个变量之间线性关系程度的度量。它是一个介于-1.0 和 1.0 之间的数字，-1.0 表示变量之间有强烈的负相关关系，而 1.0 表示有强烈的正相关关系。

零值表示变量之间没有线性关联。然而，相关性并不意味着因果关系。如果两个变量朝着同一个方向移动，则它们被认为是相关的，但这并不意味着一个导致另一个移动，或者它们是由相同事件的结果移动的。

大多数人都同意一些资产具有自然的相关性。例如，由于它们都属于同一行业，并受到相同趋势和事件的影响，苹果和微软的股票是正相关的（这意味着它们的总体趋势是相同的）。图 3-16 显示了两只股票之间的图表。注意它们如何一起移动。

![](img/dlf_0569.png)

**图 3-16。** 自 2021 年以来的苹果和微软股价

两只股票的顶部和底部几乎同时出现。同样，由于美国和英国具有相似的经济驱动因素和影响，它们也很可能有正相关的通货膨胀数字。

通过视觉解释和数学公式来检查相关性。在看一个例子之前，让我们深入了解计算相关性的根源，这样您就知道它来自哪里以及它的局限性是什么。

## 注意

简而言之，要计算相关性，您需要测量两个变量散点图中的点与一条直线的接近程度。它们看起来越像一条直线，它们就越正相关，因此有了术语“线性相关”。

有两种主要的计算相关性的方法，即使用 Spearman 方法或 Pearson 方法。

皮尔逊相关系数是从两个变量的标准差和协方差计算出来的线性关联度量。但是，什么是协方差？

*协方差*计算两个变量均值之间的差异的平均值。如果两个变量有一起移动的倾向，协方差是正的，如果两个变量通常朝相反方向移动，协方差是负的。它的范围在无穷大和负无穷大之间，接近零的值表示没有线性相关。

计算变量*x*和*y*之间的协方差的公式如下：
