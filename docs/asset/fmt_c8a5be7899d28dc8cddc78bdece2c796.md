
```py
import pandas as pd
import altair as alt

df = pd.DataFrame({
'X' : [1,2,3,4],
'Y' : [2,4,5,6],
'Z' : [3,4,5,6],
'H' : [5,6,8,9],
'M' : [3,4,5,3],
'Country' : ['USA', 'EU', 'EU', 'USA']
})

fields = df.columns.tolist()
fields.remove('Country')

chart = alt.Chart(df).mark_circle(color='#80C11E').encode(
    alt.X(alt.repeat("column"), type='quantitative'),
    alt.Y(alt.repeat("row"), type='quantitative')
).properties(
    width=100,
    height=100
).repeat(
    row=fields,
    column=fields
)

chart.save('repeat.xhtml')
```

注意：首先，创建 `pandas` `DataFrame`。然后，提取要重复的字段列表并将它们存储在`fields`变量中。接下来，定义`repeat()`方法，指定要重复的行和列。最后，使用刚刚定义的`row`和`column`变量作为图表编码的变量。

图 3.5 显示了列表 3.17 的输出。`Altair`已构建了 25 个图表，每个图表对应于`fields`变量中指定的`DataFrame`列的每个组合。

![图](img/3-5.png)

##### 图 3.5 通过编码变化的重复图表

### 3.2.5 交互性

*交互性*指的是通过用户输入（如悬停在数据点上、点击图例或拖动滑块）来操作可视化的能力。在`Altair`中，使用`interactive()`方法来启用缩放、平移和数据点悬停以显示工具提示的功能。列表 3.18 展示了如何使图表交互的示例，图 3.6 显示了生成的图表。您也可以在`03/altair/interactive.py`下找到完整的示例。

##### 列表 3.18 在 Altair 中构建交互式图表

```py
import altair as alt
import pandas as pd

df = pd.DataFrame(
    {"Country": ["Japan", "USA", "Germany", "Spain", "France", "Italy"], 
     "Medals": [4, 6, 10, 3, 7, 8], 
     "Region":["Asia","Americas","Europe","Europe","Europe","Europe"]}) 

chart = alt.Chart(df).mark_bar(color='#636466').encode(
    x='Country',
    y='Medals',
    tooltip=['Country', 'Medals', 'Region']
).properties(
    width=300, 
    title='A bar chart'
).interactive()

chart.save('interactive.xhtml')
```

注意：首先，创建`pandas` `DataFrame`。然后，构建`Altair`图表，在末尾附加`interactive()`方法。另外，添加一个工具提示，它接收要显示的`DataFrame`列的列表。

![图](img/3-6.png)

##### 图 3.6 启用交互性的条形图

### 3.2.6 配置

要配置单个图表的基本属性，请使用`properties()`方法。对于更复杂的属性和配置复合图表，您必须使用*全局配置*。`Altair`支持许多全局配置。在本节中，我们将描述轴、标题和视图。我们将等到书中稍后需要它们时再描述其他配置。

为了展示配置如何工作，我们将从图 3.7 中描述的图表开始，并通过以下列表中的代码生成。

##### 列表 3.19 定义基本图表

```py
import pandas as pd
import altair as alt

df = pd.DataFrame(
    {"Country": ["Japan", "USA", "Germany", "Spain", "France", "Italy"], 
     "Medals": [4, 6, 10, 3, 7, 8], 
     "Region":["Asia","Americas","Europe","Europe","Europe","Europe"]}) 

chart = alt.Chart(df).mark_bar(color='#636466').encode(
    x='Country',
    y='Medals'
).properties(width=300, title='A bar chart')

chart.save('chart.xhtml')
```

![图](img/3-7.png)

##### 图 3.7 默认配置构建的基本条形图

现在，让我们通过配置轴来开始修改基本图表。

#### 配置轴

要在全局级别配置轴，`Altair`支持三种方法：

+   `configure_axis()`—配置两个轴。

+   `configure_axisX()`—配置 x 轴。

+   `configure_axisY()`—配置 y 轴。

配置轴允许您自定义多个属性，例如标签、刻度和比例。以下列表显示了一个示例，您也可以在`03/altair/configure-axis.py`下找到。

##### 列表 3.20 配置轴

```py
chart = chart.configure_axis(
    labelAngle=0,         #1
    titleFontSize=14,      #2
    labelFontSize=12    #3
)

chart.save('configure-axis.xhtml')
```

#1 设置标签角度

#2 设置轴标题的字体大小

#3 设置轴标签的字体大小

注意：使用`configure_axis()`来配置轴的标签和标题。作为替代，使用`configure_axisX()`来配置 x 轴，而`configure_axisY()`来配置 y 轴。

图 3.8 显示了使用`configure_axis()`（在左侧）和不使用`configure_axis()`（在右侧）时的布局差异。

![figure](img/3-8.png)

##### 图 3.8 使用`configure_axis()`和不使用`configure_axis()`的布局之间的差异

#### 配置标题

使用`configure_title()`方法自定义图表的标题属性，如颜色、字体大小和字体族。以下列表显示了如何使用`configure_title()`方法的示例，你还可以在`03/altair/configure-title.py`下找到。

##### 列表 3.21 配置`title`

```py
chart = chart.configure_title(
    fontSize=20,             #1
    color='#80C11E',     #2
    offset=30,         #3
    anchor='start'     #4
)

chart.save('configure-title.xhtml')
```

#1 设置标题字体大小

#2 设置标题颜色

#3 设置标题与图表之间的距离

#4 设置标题锚点位置

注意：使用`configure_title()`配置标题属性。锚点必须是`None`、`start`、`middle`或`end`。

图 3.9 显示了使用`configure_title()`（左侧）和不使用`configure_title()`（右侧）的布局之间的差异。

![figure](img/3-9.png)

##### 图 3.9 使用`configure_title()`和不使用`configure_title()`的布局之间的差异

#### 配置视图

使用`configure_view()`方法自定义图表的一般视图。以下列表显示了`configure_view()`方法的使用示例，你还可以在`03/altair/configure-view.py`下找到。

##### 列表 3.22 配置`view`

```py
chart = chart.configure_view(
    strokeWidth=0,           #1
    stroke='#80C11E',    #2
    fill='#E0E0E0'    #3
)

chart.save('configure-view.xhtml')
```

#1 设置边框描边宽度

#2 设置边框描边颜色

#3 设置背景颜色

注意：使用`configure_view()`配置图表的一般视图。

图 3.10 显示了使用`configure_view()`（左侧）和不使用`configure_view()`（右侧）的布局之间的差异。

![figure](img/3-10.png)

##### 图 3.10 使用`configure_view()`和不使用`configure_view()`的布局之间的差异

现在你已经学习了`Altair`图表的基本组件，让我们继续下一步：实现一个案例研究。

## 3.3 案例研究

让我们假设你在全球人口分析中心（GDAC）工作，该中心希望研究北美的人口增长。为此研究，你将使用世界银行的`population`数据集（在 CC BY-4.0 许可下发布，[`data.worldbank.org/indicator/SP.POP.TOTL`](https://data.worldbank.org/indicator/SP.POP.TOTL)）；你可以在本书的 GitHub 仓库的`CaseStudies/population/source/population.csv`中找到数据集的清洗版本。该数据集包含从 1960 年到 2021 年所有国家的数据，如表 3.3 所示。国家名称列还包含大洲。你可以在本书的 GitHub 仓库的`CaseStudies/population/population.ipynb`下找到此案例研究的完整代码。在这种情况下，只需执行包含图表变量的单元格即可可视化`Altair`图表。

##### 表 3.3 人口数据集

| 国家名称 | 1960 | … | 2021 |

| --- | --- | --- | --- |

| 阿鲁巴 | 54,608 |  | 106,537 |

| … |  |  |  |

| 津巴布韦 | 3,806,310 |  | 15,993,524 |

你首先在以下列表中绘制一个原始图表，显示从 1960 年到 2021 年所有国家和大陆的趋势线。国家名称列很脏，所以生成的图表非常混乱，如图 3.11 所示。

##### 列表 3.23 绘制原始图表

```py
import pandas as pd
import altair as alt
alt.data_transformers.disable_max_rows()    #1

df = pd.read_csv('data/population.csv')
df = df.melt(id_vars='Country Name',       #2
             var_name='Year', 
             value_name='Population')
df['Year'] = df['Year'].astype('int')     

chart = alt.Chart(df).mark_line().encode(
    x = 'Year:Q',
    y = 'Population:Q',
    color = 'Country Name:N'
)               #3

chart
```

#1 使用此`Altair`函数处理超过 5,000 行的数据集

#2 将数据集展开成非透视表形式

#3 绘制图表

注意：使用`mark_line()`绘制图表。要绘制不同的国家，使用`color`通道。

![图表](img/3-11.png)

##### 图 3.11 原始数据的图表

该图表非常混乱，因为它提出了以下问题：国家太多，颜色太多，且没有专注于北美。为了解决这些问题，让我们使用 DIKW 金字塔，从第一步开始：将数据转化为信息。

### 3.3.1 从数据到信息

将数据转化为信息涉及从数据中提取洞察力。从图表的角度来看，这意味着使图表可读且易于理解。首先按大陆分组国家。数据集已经包含大陆的值。使用`pandas` `DataFrame`提供的`unique()`方法列出国家。有关`unique()`方法的更多详细信息，请参阅附录 B。请注意，国家列表中也包含大陆，因此我们可以构建一个大陆列表并基于该列表过滤`DataFrame`。然后，绘制生成的图表。

##### 列表 3.24 清理图表

```py
continents = ['Africa Eastern and Southern',
             'Africa Western and Central',
             'Middle East & North Africa',
             'Sub-Saharan Africa',
             'Europe & Central Asia',
             'Latin America & Caribbean',
             'North America',
             'Pacific island small states',
             'East Asia & Pacific']       #1

df = df[df['Country Name'].isin(continents)]   #2

chart = alt.Chart(df).mark_line().encode(
    x = 'Year:Q',
    y = 'Population:Q',
    color = 'Country Name:N'
)         #3

chart
```

#1 定义大陆列表

#2 根据列表过滤`DataFrame`

#3 绘制图表

注意：按大陆分组数据以减少线条数量。

图 3.12 显示了生成的图表，它更易于阅读。然而，它仍然存在一些问题：颜色太多，且没有专注于北美。在实践中，很难将北美与其他大陆进行比较。

![图表](img/3-12.png)

##### 图 3.12 专注于大陆的图表

通过专注于北美并将其他大陆按其平均值分组来更改图表，如下列表所示。

##### 列表 3.25 专注于北美

```py
mask = df['Country Name'].isin(['North America'])                 #1
df_mean = df[~mask].groupby(by='Year').mean().reset_index()    #2

df_grouped = pd.DataFrame({ 
    'Year' : df[mask]['Year'].values,
    'North America' : df[mask]['Population'].values, 
    'World': df_mean['Population'].values
})    #3

df_melt = df_grouped.melt(id_vars='Year', var_name='Continent', value_name='Population')
```

#1 定义一个仅选择北美的掩码

#2 计算掩码外大陆的平均值

#3 创建一个新的包含北美和其他大陆平均值的`DataFrame`

注意：专注于北美，并计算其他大陆的平均值。

然后，绘制图表。

##### 列表 3.26 绘制图表
