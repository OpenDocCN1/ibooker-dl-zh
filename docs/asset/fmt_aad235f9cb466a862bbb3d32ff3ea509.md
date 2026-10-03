
# 附录 C 其他图表类型

本附录描述了第六章未涵盖的一些最流行的图表。如果您想尝试新的图表，请将其作为参考。图表的代码在本书的 GitHub 仓库中描述，位于 06 目录下，此外还有其他不太流行的图表。让我们分别研究每种图表类型，从烹饪图表系列开始。

## C.1 环形图

环形图是一种圆形数据可视化类型，以环形形状显示数据。它与饼图类似，但中心有一个洞，创建了不同类别的比例或百分比的视觉表示。

将饼图转换为环形图，只需将 `innerRadius` 和 `outerRadius` 属性添加到 `mark_arc()` 方法中，如下面的列表所示。

##### 列表 C.1 生成环形图的代码

```py
import pandas as pd
import altair as alt

data = {
    'percentage': [0.7,0.3], 
    'label'     : ['70%','30%'], 'color'     : ['#81c01e','lightgray'] **}

df = pd.DataFrame(data)

chart = alt.Chart(df).mark_arc(
    innerRadius=100,
    outerRadius=150
).encode( 
    theta='percentage',
    color=alt.Color('color', scale=None),
    tooltip='label'
).properties(
    width=300,
    height=300
)**
```

**注意** 在 Altair 中生成饼图的代码。使用 `innerRadius` 和 `outerRadius` 将饼图转换为环形图。

如果您的环形图比较两个值，并且只想关注其中一个值，请将此值添加到环形图的中心，如图 C.1 所示。

![图](img/C-1.png)

##### 图 C.1 Altair 中的环形图，中心有标签

请 Copilot 生成环形图内的标签。

##### 列表 C.2 生成环形图内标签的提示

```py
# Add text to the center of the donut chart
# - Use df.head(1) to get the first row of the dataframe
# - Use the `label` column for text channel
# - Use the `color` column for color
```

**注意** 为 Copilot 生成环形图内标签的指令顺序。

下面的列表显示了生成的代码。

##### 列表 C.3 Copilot 生成的代码

```py
text = alt.Chart(df.head(1)).mark_text(
    align='center',
    baseline='middle',
    fontSize=60,
    fontWeight='bold'
).encode(
    text='label',
    color=alt.Color('color', scale=None)
).properties(
    width=300,
    height=300
)
```

**注意** 更改 `fontSize` 属性以增加或减少标签大小。

从数据故事讲述的角度来看，您可以连接多个环形图，如图 C.2 所示，以显示多个百分比值，例如，关于餐厅顾客喜欢哪些餐点的问卷调查结果。总百分比大于 100%，因为顾客可能喜欢多种餐点。

![图](img/C-2.png)

##### 图 C.2 显示问卷调查输出的多个环形图

在这个场景中，累计总百分比可能不是必需的。然而，值得注意的是，其他业务视角可能需要这样的要求。现在您已经学会了如何在 Altair 和 Copilot 中绘制烹饪图表，让我们继续到下一系列的图表：柱状图系列。

## C.2 柱状图系列

柱状图系列包括柱状图、多系列柱状图、金字塔图、堆积柱状图、100% 堆积柱状图和直方图。让我们更仔细地看看。

### C.2.1 柱状图

柱状图类似于条形图，但轴是倒置的；它在 x 轴上显示类别，在 y 轴上显示值。由于 x 轴也用于表示时间数据，因此您可以使用柱状图将时间段作为类别来描述。下面的列表显示了如何绘制柱状图。同时，将图表宽度增加到 600，为每个柱子留出更多空间。

##### 列表 C.4 创建柱状图的代码

```py
import pandas as pd
import altair as alt

chart = alt.Chart(df).mark_bar(
    color='#81c01e'
).encode(
    x=alt.X('Meal Type', sort='-y'),
    y='Number of Likes'
).properties(
    width=600,
    height=300
)

chart.save('column-chart.xhtml')
```

**备注**：反转 `x` 和 `y` 通道。同时，更改 `sort` 属性值。

图 C.3 表示生成的柱状图。

![图](img/C-3.png)

##### 图 C.3 柱状图

柱状图的 x 轴显示旋转的标签。此外，x 轴标题是不必要的。让我们开始清理。首先，在 Visual Studio Code 编辑器中格式化代码，然后写上注释 `# Rotate`，如图 C.4 所示。Copilot 将建议如何完成句子。按 Tab 键，将 45 度改为 0 度，然后按 Enter 键，然后 Copilot 将在你的注释下方的一行添加所需的代码（`axis=alt.Axis(labelAngle=0),`）。

![图](img/C-4.png)

##### 图 C.4 如何在代码中添加注释。Copilot 将建议下一步。

接下来，在 `labelAngle=0` 后添加一个逗号，并开始写单词 `title`。Copilot 会建议 `=None`。按 Tab 键确认。这个 Copilot 属性很棒！你可以在代码的任何位置添加注释，并请求 Copilot 为你生成新的代码！由于 Copilot 是一个生成式 AI 工具，可能在你的情况下，建议的提示可能不同。无论如何，你可以轻松地调整 Copilot 建议的代码以满足你的需求。

现在，让我们清理 y 轴。通过将 `y='Number of Likes'` 通道更改为 `y=alt.Y('Number of Likes', axis=alt.Axis(grid=False))` 来移除 y 轴。然后，在每根柱子的顶部添加点赞数。让我们使用 Copilot 来完成这项工作。开始写下以下文本：`# Add a text mark to the chart with the following options:.` Copilot 将建议下一步。通过按 Tab 键和 Enter 键确认。在某个时候，Copilot 将停止建议。按 Enter 键，Copilot 将为你生成代码。测试生成的代码。如果不满足你的要求，请手动更改它，或者再次使用 Copilot。在我的情况下，我不得不稍微改变 Copilot 的注释，以便生成所需的代码。以下列表显示了 Copilot 的指令。

##### 列表 C.5 Copilot 生成的提示并稍作修改

```py
# Add a text mark to the chart with the following options:
# - Use the `Number of Likes` column for y channel
# - Use the `Meal Type` column for x channel and sort by the number of likes
# - Set the color of the text to '#81c01e'
# - Set the text to the `Number of Likes` column
# - Set the font size to 14
# - Set the font weight to 600
# - Set the text baseline to 'center'
# - Set the text align to 'middle'
# - Set the y offset to 10
```

**备注**：请求 Copilot 生成在每根柱子顶部绘制标签的代码。

以下列表描述了生成的代码。

##### 列表 C.6 Copilot 生成的添加标签的代码

```py
text = alt.Chart(df).mark_text(
    color='#81c01e',
    fontSize=14,
    fontWeight=600,
    baseline='middle',
    align='center',
    dy=-10
).encode(
    x=alt.X('Meal Type',
            sort='-y',
            axis=alt.Axis(labelAngle=0,title=None),
    ),
    y='Number of Likes',
    text='Number of Likes'
)
```

**备注**：使用 `mark_text()` 方法在每根柱子顶部添加标签。设置文本属性，包括颜色、字体大小、粗细等。

最后，合并这两个图表。

##### 列表 C.7 合并图表以构建最终图表

```py
# Combine the bar chart and text mark into a single chart.
chart = chart + text

chart = chart.configure_view(
    strokeWidth=0
)

chart.save('column-chart.xhtml')
```

**备注**：编写提示 Copilot 生成合并图表的代码。

图 C.5 显示了最终的图表。

![图](img/C-5.png)

##### 图 C.5 清理后的柱状图

现在你已经学会了如何在 Altair 和 Copilot 中清理柱状图，让我们继续到下一个图表。在下一节中，我们将介绍具有多个系列的柱状图。

### C.2.2 具有多个系列的柱状图

到目前为止，我们只使用一个数据系列实现了柱状图（以及条形图）。然而，在许多情况下，您可能想要比较两个或多个数据系列。在这种情况下，您必须将 `column` 通道添加到您的图表中。考虑表 C.1 中显示的数据集，描述了 2022 年和 2023 年每种餐食类型的点赞数。

##### 表 C.1 包含两个数据系列的样本数据集，一个用于 2022 年，另一个用于 2023 年

| **餐食类型** | **2022 年点赞数** | **2023 年点赞数** |
| --- | --- | --- |
| **披萨** | 120 | 145 |
| **汉堡** | 95 | 88 |
| **意大利面** | 80 | 97 |
| **寿司** | 60 | 67 |
| **Salad** | 50 | 52 |
| **牛排** | 70 | 66 |
| **Tacos** | 90 | 78 |
| **冰淇淋** | 110 | 134 |
| **Curry** | 40 | 43 |
| **Sandwich** | 75 | 59 |

下面的列表描述了构建多系列柱状图的代码。

##### 列表 C.8 如何创建多系列柱状图

```py
import pandas as pd
import altair as alt

df = pd.read_csv('data/meals-by-year.csv')        #1

df = df.melt(id_vars=['Meal Type'],var_name='Year',value_name='Number of Likes')

chart = alt.Chart(df).mark_bar(
).encode(
    x=alt.X('Year', 
            # Rotate the labels by 0 degrees
            axis=alt.Axis(title=None, labels=False)
    ),
    y=alt.Y('Number of Likes',axis=alt.Axis(grid=False)),
    column=alt.Column('Meal Type',
                      header=alt.Header(
                          labelOrient='bottom',
                          title=None
                      )),
    color=alt.Color('Year',scale=alt.Scale(range=['lightgray','#81c01e']))
).properties(
    width=50,
    height=300
).configure_view(
    strokeWidth=0
)

chart.save('multiple-column-chart.xhtml')
```

#1 将数据文件 `data/meals.csv` 载入 pandas DataFrame。

**注意**：使用 `column` 通道将多个系列添加到您的图表中。默认情况下，柱状图将柱标签添加到图表顶部。要将它们移动到底部，请在 `alt.Column()` 通道中设置 `header` 属性。

图 C.6 显示了生成的图表。现在您已经学会了如何实现具有多个系列的柱状图，让我们继续下一个图表：金字塔图。

![图表](img/C-6.png)

##### 图 C.6 具有多个系列的柱状图

### C.2.3 金字塔图

*金字塔图*由两个背靠背的条形图组成，两侧代表对比类别，如男性和女性。水平轴表示数量，垂直轴显示比较的类别——通常是时间段。考虑表 C.2 中显示的数据集，描述了 2023 年 1 月至 12 月披萨和意大利面的订单数量。

##### 表 C.2 2023 年 1 月至 12 月披萨和意大利面的订单数量样本数据集

| **月份** | **披萨** | **意大利面** |
| --- | --- | --- |
| 一月 | 200 | 18 |
| 二月 | 220 | 19 |
| 三月 | 240 | 10 |
| 四月 | 230 | 20 |
| 五月 | 250 | 20 |
| 六月 | 280 | 40 |
| 七月 | 300 | 60 |
| 八月 | 320 | 80 |
| 九月 | 310 | 70 |
| 十月 | 290 | 50 |
| 十一月 | 270 | 30 |
| 十二月 | 240 | 20 |

折线图由三个主要元素组成：左侧部分、中间部分和右侧部分。左侧和右侧部分是条形图，左侧部分是颠倒的。中间部分则包含两个条形图的标签。我们不会为 Copilot 生成代码编写提示。相反，我们将使用它提出的建议，并直接展示生成的代码，而不提供 Copilot 建议的截图。我建议您尝试编写代码以实验 Copilot 的功能。

下面的列表描述了构建金字塔图左侧部分的代码，重点关注披萨。

##### 列表 C.9 如何创建金字塔图的左侧部分

```py
import pandas as pd
import altair as alt

# Load data data/orders.csv as a Pandas dataframe.
df = pd.read_csv('data/orders.csv')

months = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]

left_base = alt.Chart(df).encode(
    y=alt.Y('Month:N', axis=None, sort=months),
    x=alt.X('Pizza:Q', title='',sort=alt.SortOrder('descending'), axis=None),
)

left =  left_base.mark_bar(
    size=20,
    color='#c01e95'
).properties(
    title='Pizza',
    width=300,
    height=300
)

left_text = left_base.encode(
    text=alt.Text('Pizza:N'),
).mark_text(
    color='#c01e95',
    baseline='middle',
    align='right',
    dx=-10,
)
```

**注意**：构建基础图表（`left_chart`），然后使用它来构建柱状图和标签。将 `x` 通道中的 `sort` 属性设置为 `descending` 以将柱状图锚定在右侧。此外，在 `mark_text()` 方法中将 `align` 属性设置为 `right`。

现在，让我们构建金字塔图的中间部分。这部分包含标签。以下列表显示了代码。

##### 列表 C.10 如何创建金字塔图的中间部分
