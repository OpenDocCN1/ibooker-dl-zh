
在数据之前拥有知识可能看起来很奇怪。然而，在这里，我们所说的“知识”是指理解故事并将其置于特定背景所需的所有元素。所描述的映射表明了构建数据故事时的自然流程（图 5.11）。首先，确定故事的主要观点；然后设置背景；最后，设置下一步。

![figure](img/5-11.png)

##### 图 5.11 构建数据故事的流程

### 5.2.2 DIKW 在行动：完成无家可归者故事

本章开头描述的无家可归者例子停留在 DIKW 金字塔的信息层面，从数据故事的角度来看，它只包括主要观点。向图表添加背景涉及吸引观众的注意力，并使他们对你所描述的问题产生兴趣。在第二章中，你学习了在 Altair 中，你可以向图表添加文本注释，将信息转化为知识。此外，还有其他方法可以向图表添加背景，包括标签、标题、来源、图像、描述和比较。我们将在第六章中更详细地描述这些内容。

在无家可归者问题的案例中，你可以添加描述这些人们情况的文本，以及可能给涉及的人一个面孔的照片。我们将在第六章中看到如何将英雄添加到故事中。现在，只需给数据背后的人一个面孔就足够了。

在列表 5.4 中，我们将描述背景的文本作为图表副标题添加。使用`TitleParams()`函数来指定标题属性，包括副标题。将副标题设置为：

```py
Homelessness is a heartbreaking reality that leaves individuals and families without a stable home, leading to devastating consequences such as poor health and social isolation.
```

此外，添加简短描述，帮助读者关注数据：

```py
The chart describes the number of homeless people in Italy per 1,000 inhabitants, organized by region. The data is from 2021, the most recent year available (Source: ISTAT).
```

##### 列表 5.4 将背景作为副标题添加

```py
# Add context
chart = chart.properties(width=500,title=alt.TitleParams(
    text=["Together, Let's Make a Difference:","Support Our Project to Help the Homeless!"],
    subtitle=['Homelessness is a heartbreaking reality that leaves individuals and families without a stable home,','leading to devastating consequences such as poor health and social isolation.', 'The chart describes the number of homeless people in Italy per 1,000 inhabitants, organized by region. ', 'The data is from 2021, the most recent year available (Source: ISTAT).']],
    subtitleFontSize=18,
))
```

注意：在`properties()`方法中，使用`title`参数来设置图表标题。使用`TitleParams()`函数定义标题，该函数可以接收多个参数作为输入。示例传递了以下参数：标题（`text`）、副标题（`subtitle`）、副标题字体大小（`subTitleFontSize`）、标题字体大小（`titleFontSize`）。

列表 5.5 展示了如何向图表添加两张照片，并将它们与原始图表结合来构建仪表板。你必须在一个网络服务器上运行代码才能使其工作。如果你没有网络服务器，你可以从命令行运行一个本地和临时的网络服务器，在包含生成的 HTML 文件的目录中运行以下命令：`python -m http.server`。服务器应在端口 8000 上监听，并从启动它的目录中提供所有文件。将你的浏览器指向`http://localhost:8000/chart.xhtml`以访问文件`chart.xhtml`。

##### 列表 5.5 添加上下文作为图像

```py
image1 = alt.Chart(pd.DataFrame({'image_url': ['source/homeless1.png']})).mark_image(
    width=200,
    height=200,
).encode(
    url='image_url',
    x=alt.value(0),  # pixels from left
    y=alt.value(50)  # pixels from the top
)           #1

image2 = alt.Chart(pd.DataFrame({'image_url': ['source/homeless2.png']})).mark_image(
    width=200,
    height=200,
).encode(
    url='image_url',
    x=alt.value(0),  # pixels from left
    y=alt.value(300)
)                     #2

image1 + image2 | chart
```

#1 构建第一个图像。

#2 构建第二个图像

备注：使用`mark_image()`向图表添加图像。在传递给图表的 DataFrame 中设置图像文件的路径。使用`x`和`y`通道设置图像在图表中的位置。使用相同的图表绘制`image1`和`image2`（`+`运算符），并将主图表相对于`image1 + image2`绘制在右侧。

为了完成无家可归的故事，添加一个行动号召，包括下一步。在这个例子中，我们首先修改标题：*一起，让我们做出改变：支持我们的项目，帮助无家可归者!* 我们将在第九章中看到如何向图表添加行动号召的不同策略。图 5.12 显示了结果图表。

![figure](img/5-12.png)

##### 图 5.12 无家可归图表通过上下文（图像和副标题）得到丰富。

然后，我们在图表中添加一个新的部分，描述我们将如何使用资金（图 5.13）。例如，我们将用 35%的资金用于庇护和住房，25%用于职业培训，等等。

![figure](img/5-13.png)

##### 图 5.13 解释我们将如何使用资金的下一个步骤

以下列表显示了我们在图 5.13 中实现图表的方法。我们假设我们有一个 DataFrame 存储每个类别的资金（分配）百分比。

##### 列表 5.6 添加下一步

```py
import pandas as pd
import altair as alt

donuts = None

for index, row in ns.iterrows():
    curr_ns = pd.DataFrame(
        {'Category': ['A', 'B'],
         'Value': [row['Allocation'], 100-row['Allocation']]
        }                             #1
    )

    donut = alt.Chart(curr_ns).mark_arc(outerRadius=30, innerRadius=20).encode(
        theta=alt.Theta("Value:Q", stack=True),
        color=alt.Color("Category:N", scale=alt.Scale(range=['green', 'lightgray']), legend=None)         #2
    )

    title = alt.Chart(curr_ns).mark_text(text=row['Category'], y=0, size=16) #3

    text = alt.Chart(curr_ns).mark_text(text=f"{row['Allocation']}%", color=iColor, size=16)          #4

    donut = donut.properties(
        height=100,
        width=100
    )          #5

 #6
    if index == 0:
 #7
        donuts = title + donut + text
    else:
 #8
        donuts = alt.hconcat(donuts, title + donut + text)

 #9
donuts = donuts.properties(title='Our visionary plan to harness the funds')
```

#1 为当前行创建一个 DataFrame，用作图表的数据

#2 使用 Altair 生成饼图

#3 为饼图创建标题

#4 为饼图创建文本注释

#5 设置饼图的大小

#6 结合标题、饼图和文本注释

#7 如果是第一次迭代，则将组合图表分配给饼图

#8 如果不是第一次迭代，则将组合图表水平连接到现有的饼图中

#9 为最终的饼图设置标题

备注：使用`mark_arc()`将每个饼图实现为一个单独的图表。对于每个饼图，构建一个临时的 DataFrame（`current_ns`），其中包含实际分配值及其互补值（100 - 实际分配值）。然后，使用不同的颜色绘制当前分配值及其互补值。

图 5.14 显示了结果图表。如果 Angelica 将图 5.14 中的图表提交给委员会基金会，她可能会有更多获得资助的机会。现在你已经完成了无家可归的故事，让我们描述如何将生成式 AI 的力量融入 DIKW 金字塔。

![figure](img/5-14.png)

##### 图 5.14 无家可归故事的最终图表

## 5.3 将生成式 AI 融入 DIKW 金字塔

正如我们在第四章中讨论的，生成式 AI 可以帮助我们更进一步，通过使创建复杂和逼真的计算机生成内容成为可能，从而改变各个行业和领域。现在，有了这些知识，我们可以利用生成式 AI 的能力来增强人类创造力并自动化内容生成过程。

图 5.15 显示了我们可以将生成式 AI 工具融入 DIKW 金字塔的每个步骤：

+   *洞察提取* — 使用生成算法自动化构建图表的代码生成。

+   *添加背景* — 使用文本生成将基本文本转换为更吸引观众的文本。添加由 AI 生成的图像、声音和视频，为观众提供额外的背景信息。

+   *行动号召* — 使用文本生成来提出新想法和吸引人的文本。

![图](img/5-15.png)

##### 图 5.15 我们如何将生成式 AI 工具融入 DIKW 金字塔

重要的是要注意，列出的例子只是你可以将生成式 AI 融入 DIKW 金字塔的几种方式之一。还有无数其他方式可以利用这些工具，例如综合大量数据、开发个性化的预测模型，以及根据数据构建个性化的推荐。然而，在这本书中，我们将专注于将生成式 AI 融入 DIKW 金字塔的技术，因为它有助于构建数据故事。无论如何，我鼓励你跳出思维定式，探索将生成式 AI 应用于工作中的新方法和创新方式。随着如此多的潜力等待被解锁，可能性确实是无限的！现在，你已经学会了如何将生成式 AI 工具融入 DIKW 金字塔，让我们将这种策略应用于我们的案例研究：无家可归者的故事。

## 5.4 背后故事：无家可归者的故事

在前面的章节中，你看到了如何使用 GitHub Copilot 生成构建图表的代码。现在，是时候更进一步了。

再次考虑无家可归者的故事，特别是图 5.12。该图包含一个标题，定义了行动号召；一个副标题，设定背景；以及两张照片，增加了背景信息。虽然你可能认为我们提出了标题并创建或下载了图像，但实际上是 ChatGPT 和 DALL-E 共同合作生成内容！让我们看看我们是如何使用 ChatGPT 生成副标题以及 DALL-E 生成照片的。在接下来的章节中，我们将看到如何使用更深入的生成式 AI 工具进行数据叙事。

### 5.4.1 创建引人入胜的副标题

我们与 ChatGPT 进行了一次对话，以提取描述无家可归者生活状况的可能情境。这个情境应该是一个简短且引人入胜的句子。图 5.16 展示了对话中涉及的步骤。"Q"代表用户问题（这些将成为我们提示的基础），而"A"代表 ChatGPT 的回答。您可以在`05/genAI/SubtitleConversation.txt`中阅读完整的对话。

![figure](img/5-16.png)

##### 图 5.16 生成用于情境的文本的步骤

ChatGPT 在四个步骤后生成了文本。我们使用了以下策略来让 ChatGPT 生成所需的文本：

1.  *描述* — 要求 ChatGPT 一般性地描述您的问题。在这种情况下，要求 ChatGPT 一般性地描述无家可归的状况。作为回答，ChatGPT 生成了一段长文本。

2.  *缩短* — 要求 ChatGPT 对生成的文本进行摘要。

3.  *转换* — 要求 ChatGPT 使摘要对观众更具吸引力。

4.  *缩短* — 如果文本仍然很长，要求 ChatGPT 进一步缩减。

在没有意识到的情况下，我们将 DIKW 模型应用于 ChatGPT 的使用。从一段长文本（数据）开始，我们提取了信息（摘要），然后将其转化为知识和智慧（引人入胜的文本）。换句话说，当您与 ChatGPT 生成情境时，将对话组织成一个故事（图 5.17）。

![figure](img/5-17.png)

##### 图 5.17 数据故事结构与 ChatGPT 对话生成情境之间的映射

您可以将描述的过程推广到生成各种可能的副标题，然后选择适合您需求的那个。此外，您还可以添加特定的关键词，这有助于吸引目标受众并提高文本的可见性。我们将在第六章中看到如何将关键词添加到 ChatGPT 对话中。

### 5.4.2 生成图像

在您的图表中添加一张或多张图片或照片可以帮助给图表的主题赋予一个面孔。然而，图片常常受到版权保护，这意味着在没有获得所有者许可的情况下使用它们可能会导致法律后果。我们可以使用 DALL-E 和其他 AI 工具来创建独特且不受版权限制的图像。

在关于无家可归者的图表中，我们添加了两张代表无家可归者的照片。我们使用以下简单的提示来生成 DALL-E 的照片："一张无家可归者的照片"。DALL-E 生成了四张图片，我们选择了其中两张（图 5.18）。在第六章中，我们将看到更多高级的图像生成技术。

![figure](img/5-18.png)

##### 图 5.18 当提示 DALL-E 展示无家可归者照片时生成的照片

现在您已经学会了如何使用生成式 AI 工具将您的原始数据可视化图表转换为数据故事，让我们来看另一个例子，以巩固这些概念。

## 5.5 另一个例子：假新闻

想象一下`LatestNews`是一个重要的网站，它发布来自不同贡献者的新闻。在某个时刻，主编收到了不同读者的投诉，因为他们阅读了大量的虚假新闻。主编联系你分析`LatestNews`网站上的虚假新闻数量，并建议网站编辑注意那些最有可能为虚假的新闻类别。你已经收集了数据，并且你有表 5.2 所示的数据集。

##### 表 5.2 示例数据集的摘录

| 类别 | 虚假文章数量 | 文章总数 |
| --- | --- | --- |
| 政治 | 1,235 | 1,300 |
| 经济 | 1,456 | 1,678 |
| 司法 | 300 | 570 |
| 宗教 | 30 | 100 |

数据集显示了每个新闻类别的虚假新闻故事数量和文章总数。你首先绘制一个初步图表，注意每个类别的虚假文章百分比。图 5.19 显示了初步图表，列表 5.7 显示了相关的代码。你还可以在 GitHub 仓库的`CaseStudies/fake-news/raw-chart.py`下找到该代码。

![figure](img/5-19.png)

##### 图 5.19 显示按类别划分的虚假文章数量的原始图表

##### 列表 5.7 构建原始图表
