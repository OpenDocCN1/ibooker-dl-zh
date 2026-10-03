

# 第十一章：发布数据故事

## 本章涵盖

+ 导出故事
+ 使用 Streamlit 发布故事
+ 发布故事的替代方法：Tableau、Power BI 和 Comet
+ 通过幻灯片展示您的数据故事

在这本书的整个过程中，我们构建了我们的数据故事。现在，在本书的最后一章，是时候向我们的观众展示它了。本章全部关于如何将您的数据故事传达给观众，并且以道德的方式进行。在本章的第一部分，我们将关注不同的导出数据故事的技术。接下来，我们将描述 Streamlit，这是一个与 Altair 完全集成的 Python 库。Streamlit 可以帮助您构建一个完整的独立网站，托管您的 Python 代码。然后，我们将探讨一些发布故事的替代方法，包括将其集成到一些流行的数据分析与可视化工具中的技术：Tableau 和 Power BI。之后，我们将描述如何将数据故事集成到 Comet 中，这是一个机器学习（ML）的实验平台。最后，我们将看到如何通过幻灯片展示您的数据故事。让我们从第一个要点开始：导出故事。

## 11.1 导出故事

Altair 提供了不同的格式来导出您的数据故事。在整个书中，我们使用了`chart.save('chart.xhtml')`，但 Altair 还支持其他格式，包括以下：

+ JPEG
+ PNG
+ SVG
+ JSON
+ PDF

要将图表保存为特定格式，只需将格式扩展名添加到您的文件中，例如`chart.save('chart.png')`用于 PNG 文件。对于 PNG 文件，您还可以通过一个额外的参数指定每英寸像素数（ppi）：`chart.save('chart.png', ppi=300)`。

您可能会想知道，为什么在这本书中，我们一直使用 HTML 格式而不是 PNG 或 SVG。主要原因在于 HTML 是唯一支持显示交互性的格式。此外，一旦您创建了 HTML 文件，Altair 还会在图表右上角提供默认操作菜单，让您可以将故事下载为其他格式，如图 11.1 所示。

![图](img/11-1.png)

##### 图 11.1 HTML 文件右上角的菜单允许您将故事导出为其他格式。

如果您不想显示操作菜单，您可以禁用它，如下面的列表所示。

##### 列表 11.1 禁用操作菜单

```py
chart.save('chart.xhtml', embed_options={'actions': False})
```

您从将原始数据转换为数据故事，通过 DIKW 金字塔、生成式 AI 和 Altair 的旅程已经完成。您的故事终于准备好向观众传播了！

在接下来的两个部分中，我们将探讨一些发布故事的替代技术。如果您不感兴趣，可以自由跳过。首先，我们将看到如何使用 Streamlit 发布故事，然后我们将看到如何将我们的故事嵌入到一些流行的工具中。让我们从 Streamlit 开始。

## 11.2 通过网络发布故事：Streamlit

*Streamlit* ([`streamlit.io/`](https://streamlit.io/)) 是一个开源框架，允许你使用 Python 快速构建网络应用。要开始使用 Streamlit，请参阅附录 A。与其它选项相比，使用 Streamlit 的主要优势是它与 Altair 完全集成。

在本节中，我们将了解如何使用 Streamlit 发布数据故事。Streamlit 使用特定的函数在网页上显示内容，例如 `st.Title()` 用于设置页面标题。

考虑第 11.2 列表中的示例。首先，导入 Streamlit 库。然后，像在 Altair 中一样构建你的图表。要在网页上的任何位置添加 Altair 图表，请使用 `st.altair_chart()`。

##### 列表 11.2 在 Streamlit 中构建应用

```py
import streamlit as st
import pandas as pd
import altair as alt

# Read the data
df = pd.read_csv('../data/population.csv')
df = df.melt(id_vars='Country Name', var_name='Year', value_name='Population')
df['Year'] = df['Year'].astype('int')

continents = ['Africa Eastern and Southern', 'Africa Western and Central',
              'Middle East & North Africa', 'Sub-Saharan Africa',
              'Europe & Central Asia', 'Latin America & Caribbean',
              'North America', 'East Asia & Pacific']
df = df[df['Country Name'].isin(continents)]

# Create Streamlit app
st.title('Population of Continents')

# Add a slider for year selection
selected_year = st.slider('Select a year', min_value=1960, max_value=2021, value=2021, step=1)
# Filter data based on selected year
filtered_df = df[df['Year'] == selected_year]
# Create Altair chart
chart = alt.Chart(filtered_df).mark_bar(color='#80C11E').encode(
    y=alt.Y('Country Name:O', sort='-x', title=''),
    x=alt.X('Population:Q', title='Population')
).properties(
    width=600,
    height=300
).configure_view(
    strokeWidth=0
).configure_axis(
    grid=False
)

# Display chart in Streamlit app
st.altair_chart(chart)
```

注意：首先导入所需的库。然后，将数据作为 pandas DataFrame 加载。接下来，像通常一样在 Altair 中构建图表。使用 `st.slider()` 在 Streamlit 中构建滑块。最后，使用 `st.altair_chart()` 在 Streamlit 应用中显示图表。

要启动应用，请在终端中运行以下命令：`streamlit app.py`，其中 app.py 是包含第 11.2 列表中代码的脚本的名称。图 11.2 显示了结果应用。

![figure](img/11-2.png)

##### 图 11.2 Streamlit 中的结果应用

你可以从本书的 GitHub 仓库中下载本节描述的代码，路径为 11/streamlit/app.py。现在你已经学会了如何在 Streamlit 中发布你的数据故事，让我们继续探讨一些其他发布故事的方法。如果你不感兴趣，可以直接跳到结论部分。

发布你的数据故事有多种方式，包括报告、演示文稿、HTML 页面等。在本节中，我们将介绍三种工具：

+ Tableau
+ Power BI
+ Comet

让我们从第一个工具开始：Tableau。

## 11.3 Tableau

*Tableau* ([`www.tableau.com/`](https://www.tableau.com/)) 是一款商业智能工具，帮助用户探索和分析数据。与 Streamlit 不同，Tableau 不需要任何编程技能，因为其界面完全可视化。如果你已经熟悉 Tableau 并用它来实现仪表板，结合 Tableau 和 Altair 可能很有用。你可以将 Altair 生成的数据故事集成到 Tableau 中，作为仪表板的一个附加项。

在使用 Tableau 之前，你必须将其下载到你的本地计算机上并支付许可证费用。Tableau 还为学生和教师提供试用版和完全免费的许可证。在本章的参考文献中，你可以找到一些有趣的资源，帮助你开始使用 Tableau。在本节中，我们描述了如何在 Tableau 中导入你的数据故事。如果你不感兴趣，可以直接跳过本节，转到下一节。考虑第八章 slider.py 中的示例和图 11.3 所示。

![figure](img/11-3.png)

##### 图 11.3 带有滑块的图表

要在 Tableau 中导入你的图表，执行以下操作：

1. 打开终端，将其指向包含 08/slider.py 的目录。运行命令 `python3 slider.py` 以生成 HTML 文件。
2. 在终端中，启动 Python 服务器如下：`python3 -m http.server`。
3. 在浏览器中指向 localhost:8000，并访问 slider.xhtml；您应该看到图 10.14 中显示的图表。移动滑块以检查代码是否正确工作。
4. 启动 Tableau Desktop 应用程序，导入 08/data/population.csv 文件，并创建一个新的仪表板。
5. 在 Tableau 中，从对象菜单双击网页，并在对话框中输入以下 URL，如图 11.4 所示：http://localhost:8000/slider.xhtml。

![图](img/11-4.png)

##### 图 11.4 Tableau 中的编辑 URL 对话框

现在，您应该在 Tableau 仪表板中看到您的 Altair 图表。既然您已经学会了如何在 Tableau 仪表板中包含您的数据故事，让我们继续下一个替代方案：在 Power BI 报告中包含您的数据故事。

注意：在某些情况下，网页导入器失败，因为它不支持 Altair 生成的 JavaScript 包装器以包含图表。在这种情况下，将图表导出为 .png 文件，并使用图像对象将其导入到 Tableau 中。

## 11.4 Power BI

*Power BI* ([`mng.bz/o0Pd`](https://mng.bz/o0Pd)) 是微软发布的另一个商业智能工具。Power BI 提供桌面和在线版本。与 Tableau 类似，您必须购买许可证才能使用它；然而，您可以使用试用版进行测试。在本章的参考文献中，您可以找到一些开始使用 Power BI 的资源。

与 Tableau 类似，如果您已经熟悉并使用 Power BI，则可能需要将 Altair 生成的数据故事集成到 Power BI 仪表板中。否则，使用其他解决方案，如 Streamlit，会更好。

在本节中，我们仅描述如何将 Altair 生成的图表导入到 Power BI 中。我们将使用在线版本。具体来说，我们将使用 Deneb 插件，该插件允许您导入用 Vega 编写的数据。从 Altair，您始终可以直接从 HTML 版本导出 Vega 或 Vega-Lite JSON 文件，如图 10.12 所示。

为了简单起见，在下面的示例中，我们将直接使用 GitHub 仓库中本书的 03/vega/json/spec.json 目录下的 Vega 代码。执行以下步骤：

1. 登录您的 Power BI 账户，然后点击新建报告 → 粘贴或手动输入您的数据 → 创建空白报告（图 11.5）。

![图](img/11-5.png)

##### 图 11.5 创建新报告的 Power BI 界面快照

2. 点击可视化 → （三个点）获取更多可视化 → Vega → Power BI 中的 Deneb：声明性可视化 → 添加（图 11.6）。

![图](img/11-6.png)

##### 图 11.6 Deneb 插件

3. 在可视化菜单中，选择 Deneb。

4. 在数据菜单中，将 X 和 Y 添加到 Deneb。点击 X 和 Y 附近的箭头，然后点击不汇总（图 11.7）。

![图](img/11-7.png)

##### 图 11.7 每个变量的选择菜单

5. 在应用左侧，你应该能看到 Deneb 框。在报告中点击 Deneb 框中的三个点，然后点击编辑 → Vega - 空白（图 11.8）。

![图](img/11-8.png)

##### 图 11.8 每个变量的选择菜单

6. 将 03/vega/json/spec.json 中的代码复制到 Deneb 编辑器中。你需要删除数据才能使其工作。以下列表显示了修改后的版本。接下来，点击创建。

##### 列表 11.3 图表的 Vega 表示

```py
{
  "description": "A basic line chart",
  "width": 400,
  "height": 200,
  "padding": 5,
 "data": [
 {
 "name": "dataset"
 }
 ],

  "scales": [
    {
      "name": "xscale",
      "domain": {"data": "dataset", "field": "X"},
      "range": "width"
    },
    {
      "name": "yscale",
      "domain": {"data": "dataset", "field": "Y"},
      "range": "height"
    }
  ],

  "axes": [
    { "orient": "bottom", "scale": "xscale" },
    { "orient": "left", "scale": "yscale" }
  ],

  "marks": [
    {
      "type": "line",
      "from": {"data":"dataset"},
      "encode": {
        "enter": {
          "x": {"scale": "xscale", "field": "X"},
          "y": {"scale": "yscale", "field": "Y"},
          "stroke": { "value": "#636466"}

        }
      }
    },
    {
      "type": "symbol",
      "from": {"data":"dataset"},
      "encode": {
        "enter": {
          "x": {"scale": "xscale", "field": "X"},
          "y": {"scale": "yscale", "field": "Y"},
          "shape": {"value": "circle"},
          "size" : {"value": 100},
          "fill": { "value": "#636466"}
        }
      }
    }
  ]
  }
```

注意：用导入到 Power BI 的数据集的链接替换数据部分。

图 11.9 显示了结果图表。现在你已经学会了如何在 Power BI 中导入 Vega 图表，让我们继续到最后一个工具：Comet。

![图](img/11-9.png)

##### 图 11.9 Power BI 中的结果图表

## 11.5 Comet

*Comet* ([`www.comet.com/site/`](https://www.comet.com/site/))是一个用于 ML 模型跟踪和测试的实验平台。在我的上一本书《Comet for Data Science》中，我详细描述了如何开始使用 Comet 以及如何跟踪 ML、深度学习（DL）、时间序列和自然语言处理任务。因此，有关 Comet 的更多详细信息，请参阅此书。

如果你的数据故事是完整数据科学项目的成果，包括模型实验和评估，你可能会决定将 Altair 生成的数据故事与 Comet 集成。由于 Comet 支持 Python，使用它导出你的数据故事不需要任何特别的努力。

在 Comet 中，你还可以创建报告。*报告*是一组文本和面板，显示实验结果。这是一个非常有用的功能，因为你可以使用一个工具进行实验和报告。与 Tableau 和 Power BI 不同，Comet 对个人用户是免费的，因此你不需要付费来访问服务。此外，Comet 支持许多编程语言，包括 Python，因此你可以继续使用 Python 来编写报告。

在本节中，我们将描述如何在 Comet 中包含一个简单的交互式图表。我们将使用由 08/slider.py 生成的 HTML 代码。我们将用 HTML/JavaScript 编写代码，但 Comet 也支持 Python。如果你没有 slider.xhtml 文件，打开终端，移动到 08 目录，并运行命令`python slider.py`。

执行以下步骤：

1. 登录 Comet，按照以下步骤开始一个新项目：新建项目 → 项目名称，描述，项目类型：实验跟踪。
2. 通过点击报告 → 新报告 → 添加面板 → 新来创建一个新的报告。
3. Comet SDK 应该会打开。默认情况下，它配置为在 JavaScript 中工作。如果你想用 Python 工作，在右上角点击 JavaScript 并选择 Python。
4. 在资源选项卡下，添加以下三个 JavaScript 脚本（图 11.10）：

    1. [`cdn.jsdelivr.net/npm/vega@5`](https://cdn.jsdelivr.net/npm/vega@5)

