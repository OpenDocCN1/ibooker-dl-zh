
# 第二章：在 Altair 和 GitHub Copilot 中运行您的第一个数据故事

## 本章涵盖

+ 介绍 Altair
+ 相关用例：描述场景
+ 使用 Altair
+ 使用 Copilot

我的独特之处在于立即动手实践。如果我没有立即感受到，我就不会开心。要开始使用软件或任何其他东西，我立即会去“入门”部分。当我需要更多细节时，我会查阅文档。这一章节也是基于同样的想法产生的：立即了解事物的工作原理。我们将查看书中将要学习的大致但完整的草图。在本章中，我们将探讨 Altair 背后的基本概念，然后我们将实现一个实际的应用案例，这将使我们能够将原始数据集转化为故事。我们将逐步应用数据、信息、知识、智慧（DIKW）金字塔原则于 Altair，并逐步查看所取得的结果。在章节的第二部分，我们将使用 Copilot 自动化故事创建过程的一些步骤。我们将只关注 Copilot 作为生成式 AI 工具，以保持章节简单且流程易于理解。在接下来的章节中，我们将介绍 ChatGPT 和 DALL-E 到 DIKW 金字塔。

## 2.1 介绍 Altair

Vega-Altair 库（简称 Altair）是一个基于 Vega 和 Vega-Lite 可视化语法的声明式 Python 库，用于统计可视化。Vega 是一个用于创建表达性和交互式数据可视化的可视化语法。Vega-Lite 是建立在 Vega 之上的高级声明式语言，旨在通过简洁直观的语法简化常见数据可视化的创建。我们将在下一章中讨论 Vega 和 Vega-Lite。

Altair 是一个声明式库，这意味着您可以描述预期的图表结果，而不是需要手动编程每个步骤，定义一个简洁直观的语法来创建交互式可视化。我们使用声明式库来指定我们希望在图表中看到的内容。我们可以指定我们想要的数据和可视化类型，然后库会为我们自动创建可视化。声明式库与命令式库形成对比，后者更侧重于手动构建可视化（例如，指定所需的轴、大小、图例和标签）。Matplotlib 是命令式库的一个例子。

每个 Altair 图表至少包含三个主要元素：图表、标记和编码。我们将在接下来的页面中简要介绍这些内容。有关安装 Altair 和本章中描述的其他 Python 库的说明，请参阅附录 A。有关本章中使用到的 pandas DataFrame 方法的更多详细信息，请参阅附录 B。

### 2.1.1 图表

在 Altair 中，图表是一个对象，它作为在 Altair 中构建和自定义交互式可视化的起点。使用`alt.Chart()`方法输入您想要可视化的数据集，如列表 2.1 所示。您可以在本书的 GitHub 仓库中找到此示例（`02/bar-chart.py`）。首先，克隆本书的 GitHub 仓库：[`mng.bz/PZVP`](https://mng.bz/PZVP)。要开始使用 GitHub，请遵循本书附录 A 中描述的程序或 GitHub 官方文档（[`mng.bz/1Geg`](https://mng.bz/1Geg)）。

#### 列表 2.1 Altair 中的图表

```py
import altair as alt
import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie'],
'Age': [25, 30, 35],
'City': ['New York', 'Paris', 'London']
}                                        #1

df = pd.DataFrame(data)     #2

print(df)          #3

chart = alt.Chart(df)     #4
```

#1 为 DataFrame 创建数据

#2 从数据创建 DataFrame

#3 打印 DataFrame

#4 创建图表

注意：首先，导入所需的库，然后构建包含您的数据的 DataFrame。最后，将 DataFrame 传递给`alt.Chart()`。此示例没有创建任何可视化，因为我们没有指定我们想要表示的内容。请注意，代码仍然不完整，所以如果您尝试运行它，它将不会工作。请耐心等待。在下一节中，您将添加其他部分以运行它！

### 2.1.2 标记

标记定义了如何表示数据。标记的示例包括柱状图、线状图、面积图等。要指定标记，将其附加到 Altair 图表。例如，在下面的列表中，我们将指定我们想要使用柱状图。

#### 列表 2.2 在 Altair 中标记

```py
chart = alt.Chart(df).mark_bar()
```

注意：使用`mark_bar()`在 Altair 中创建柱状图。

图表尚未准备好，因为您需要其他部分才能显示它。然而，为了有一个概念，图 2.1 显示了最终图表将如何看起来。

![图](img/2-1.png)

##### 图 2.1 柱状图的示例

还有其他标记的示例，如`mark_line()`，用于绘制线状图（图 2.2）。此外，`mark_circle()`绘制散点图（图 2.3）。

![图](img/2-2.png)

##### 图 2.2 线状图的示例

![图](img/2-3.png)

##### 图 2.3 散点图的示例

### 2.1.3 编码

编码指定 DataFrame 列与其在 Altair 图表中的视觉表示之间的映射。在实践中，编码定义了数据在哪里在图表中表示，例如它们的位置、大小和颜色。在实践中，编码定义了数据属性如何映射到图表中的视觉属性。例如，您可以将数据集的*x*值编码为*x*轴上的位置，将*y*值编码为 y 轴上的位置。在 Altair 中，*通道*指的是一个视觉属性，如`x`、`y`、`color`、`size`和`shape`，可以用来在图表中表示数据属性。

每个通道都与一种数据类型相关联，它描述了属性包含的数据类型，例如对于数值数据使用*定量*，对于有序分类数据使用*有序*，或者对于无固有顺序的分类数据使用*名义*。例如，在下面的列表中，我们使用编码来指定 DataFrame 中必须用于 x 轴和 y 轴的列。

#### 列表 2.3 Altair 中的编码

```py
import pandas as pd
import altair as alt

data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Paris', 'London']
}                                   #1

df = pd.DataFrame(data)      #2

# Create the chart
chart = alt.Chart(df).mark_bar(
).encode(
    x = 'Age:Q',     
    y = 'Name:N' 
)

chart.save('bar-chart.xhtml')
```

#1 创建 DataFrame 的数据

#2 从数据创建 DataFrame

注意：在 Altair 中使用`encode()`定义编码。将`df`的`Age`变量放在 x 轴上，将`df`的`Name`变量放在 y 轴上。将`Age`解释为数量（`Q`），将`Name`解释为名义值（`N`）。要显示图表，请使用`save()`方法将其保存为 HTML 文件。或者，如果你使用的是 Jupyter Notebook，只需写出图表变量的名称即可在行内显示图表。

图 2.4 显示了列表 2.4 中代码生成的图表。现在你已经学习了 Altair 背后的基本概念，我们将回顾在 Altair 和 Copilot 中实现实际场景的先决条件。

![figure](img/2-4.png)

##### 图 2.4 由列表 2.3 生成的图表

#### 练习 1

使用列表 2.4 中的 DataFrame 绘制图 2.2（折线图）和图 2.3（散点图）。

##### 列表 2.4

```py
df = pd.DataFrame({
   'X' : [2,3,5,6,7,9,4,5,5,6],
   'Y' : [4,5,6,7,8,9,5,6,3,5]
})
```

练习的解答可以在本书的 GitHub 仓库下的`02/line-chart.py`和`02/scatter-plot.py`找到。

## 2.2 用例：描述场景

假设我们是在 2019 年底，在 COVID-19 大流行之前。你是一名葡萄牙酒店的数据分析师。你的老板经历了酒店游客到达人数的增加，并希望投资建造一个新的游泳池。在投资之前，他们要求你研究这一现象，以了解他们的投资是否会成功。你的任务是查看葡萄牙在近年来相对于全球趋势是否看到了游客到达人数的增加。

你开始搜索并找到一个关于旅游住宿设施到达情况的数据库（[`mng.bz/2KOX`](https://mng.bz/2KOX)），这是由欧盟统计局发布的公开数据。为了简化，在本章中，你只考虑一个数据集，但在实际应用场景中，你应该考虑更多的数据，包括访问葡萄牙的游客中游泳池的受欢迎程度、在酒店新建游泳池的成本以及新游泳池可能为酒店带来的潜在收入。

该数据集包含了自 2019 年以来所有欧洲国家旅游住宿设施到达的数量。为了简化，只需关注以下国家：葡萄牙、法国、意大利、德国、西班牙和英国。本章中描述的代码可以在 GitHub 仓库的`CaseStudies/tourist-arrivals`下找到，数据集在`CaseStudies/tourist-arrivals/source`下。

### 2.2.1 数据集

表 2.1 展示了我们将要分析的样本数据集。该数据集包含了意大利（IT）、法国（FR）、德国（DE）、葡萄牙（PT）、西班牙（ES）和英国（UK）从 1994 年到 2019 年的旅游接待设施到达情况。

##### 表 2.1 自 1994 年以来旅游接待设施到达情况

| 日期 | IT | FR | DE | PT | ES | UK |
| --- | --- | --- | --- | --- | --- | --- |
| 1990-01-01 | 2,543,920 | | 3,185,877 | 325,138 | 1723,786 | 1,776,000 |
| 1990-02-01 | 2,871,632 | | 3,588,879 | 381,539 | 1,885,718 | 2,250,000 |
| 1990-03-01 | 3,774,702 | | 4,272,437 | 493,957 | 2,337,847 | 2,662,000 |
| … | … | … | … | … | … | … |
| 2019-08-01 | 11,649,500 | 13,692,822 | 14,570,339 | 2,531,809 | 12,893,366 | 8,889,049 |
| 2019-09-01 | 9,888,817 | 11,684,845 | 14,373,815 | 2,263,748 | | 5,858,984 |
| 2019-10-01 | 7,692,388 | 10,401,793 | 13,780,441 | 1,995,942 | | 7,455,781 |

表格中某些国家和某些年份存在一些缺失值，因此如果我们想按年份合并数据，例如，我们应该注意，因为汇总不完整的年份可能会导致错误的结果。按照书中 GitHub 仓库中`from-data-to-information/raw-chart.py`文件中的描述，将数据集作为 pandas DataFrame 加载。

##### 列表 2.5 将数据集作为 pandas DataFrame 加载

```py
import pandas as pd

df = pd.read_csv('../source/tourist_arrivals_countries.csv', parse_dates=['Date'])
```

注意：使用 pandas 将数据集作为 DataFrame 加载。使用`parse_dates`参数将`Date`字段加载为日期。

现在，你已经准备好进行基本的数据探索。

### 2.2.2 数据探索

EDA（探索性数据分析）是提取数据洞察的必经阶段，但你不能使用其输出构建原始数据故事。实际上，EDA 的目的是分析和可视化数据以提取洞察并理解其潜在的模式、分布和关系。你甚至可以使用生成式 AI 进行 EDA。本书的范围不包括进行完整的数据探索。你可以参考本章末尾的参考文献，了解如何进行 EDA。然而，你可以使用现有的 EDA 库进行快速数据探索，例如 ydata-profiling（[`pypi.org/project/ydata-profiling/`](https://pypi.org/project/ydata-profiling/)）或 sweetviz（[`pypi.org/project/sweetviz/`](https://pypi.org/project/sweetviz/)）。在本章中，我们将使用 ydata-profiling。导入库，创建一个`ProfileReport()`对象，并将 pandas DataFrame 作为输入参数传递。最后，通过调用`to_file()`方法生成报告。

##### 列表 2.6 构建摘要报告

```py
from ydata_profiling import ProfileReport
import pandas as pd

df = pd.read_csv('../source/tourist_arrivals_countries.csv', parse_dates=['Date'])

eda = ProfileReport(df)
eda.to_file(output_file='eda.xhtml')
```

注意：使用 ydata-profiling 构建数据集的摘要报告。

因此，`to_file()`方法生成一个包含报告的 HTML 文件。图 2.5 显示了生成的报告快照。

![figure](img/2-5.png)

##### 图 2.5 ydata-profiling 生成的报告快照

报告包含许多部分，如图 2.5 右上角的菜单所示。报告可以帮助你完成以下任务：

+ *理解数据。* 报告提供了数据集的概述，包括观测值数量、变量和缺失值。使用这些信息来了解数据质量。
+ *识别数据类型。* 报告确定了每个变量的数据类型，并提供了它们分布的摘要，包括均值、中位数、标准差和范围等指标。使用这些信息来识别可能需要进一步预处理的变量。
+ *识别相关性。* 报告提供了一个相关矩阵、热图和散点图矩阵，可以帮助识别高度相关或相关的变量。
+ *识别分布。* 报告提供了分布图，包括直方图和密度图，以帮助识别每个变量的分布形状和范围。使用这些信息来理解数据中的潜在模式、趋势和数据质量问题。

在你的情况下，缺失值部分描述了一些国家（法国、西班牙和联合王国）的缺失值，如图 2.6 所示。

![图表](img/2-6.png)

##### 图 2.6 报告突出了法国、西班牙和联合王国的缺失值。

现在我们已经快速探索了数据集，你准备好从该数据集中构建一个数据故事了。让我们从第一种方法开始：使用 Altair。

## 2.3 第一种方法：Altair

为了将数据转化为故事，我们将逐步进行。首先，我们将构建一个基本图表，然后我们将逐步丰富它，添加将图表转化为独立故事的必要元素。我们将使用 DIKW 金字塔的原则来实现这种渐进式丰富：

+ 从数据到信息
+ 从信息到知识
+ 从知识到智慧

让我们从 DIKW 金字塔的第一步开始：将数据转化为信息。在本节中，我们将使用书中 GitHub 仓库`CaseStudies/tourist-arrivals`下的代码。

### 2.3.1 从数据到信息

将数据转化为信息意味着从数据中提取一些见解。首先，绘制以下代码中的原始图表和文件`from-data-to-information/raw-chart.py`。

##### 列表 2.7 在 Altair 中构建原始图表

```py
import altair as alt
import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv('../source/tourist_arrivals_countries.csv', parse_dates=['Date'])

df = pd.melt(df, id_vars='Date', value_name='Tourist Arrivals', var_name='Country')

chart = alt.Chart(df).mark_line().encode(
    x = 'Date:T',                       #1
    y = 'Tourist Arrivals:Q',       #2
    color=alt.Color('Country:N')    #3
)

chart.save('raw-chart.xhtml')
```

#1 使用符号：`:T`来指定数据类型是时间的

#2 使用符号：`:Q`来指定数据类型是定量的

#3 使用符号：`:N`来指定数据类型是名义的

注意：使用`mark_line()`属性在 Altair 中构建原始图表。在 y 轴上绘制游客到达数量，在 x 轴上绘制日期。

使用`melt()`函数将数据集进行转置（即，将数据行转换为列）。有关`melt()`函数的更多详细信息，请参阅附录 B。然后，构建图表。该图表使用三个通道：`x`和`y`描述轴，`color`按国家分组。对于每个通道，指定数据集中的列（例如，`Date`用于`x`通道）和数据类型（`T`用于时间数据，`Q`用于定量数据，`N`用于名义数据）。图 2.7 显示了生成的图表。

![图表](img/2-7.png)

##### 图 2.7 Altair 中生成的原始图表，没有任何操作
