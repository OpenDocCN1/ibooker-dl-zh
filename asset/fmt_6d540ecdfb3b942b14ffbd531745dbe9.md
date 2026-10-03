
如果你的代码包含图像（我们将在下一章中看到），你可以在本地或 Web 服务器上运行它。如果你没有 Web 服务器，你可以从命令行运行一个本地和临时的 Web 服务器，在包含生成的 HTML 文件的目录中运行以下命令：`python -m http.server.` 服务器应该监听 8000 端口，并应该从它启动的目录中提供所有文件。将你的浏览器指向 `http://localhost:8000/chart.xhtml` 以访问 `chart.xhtml` 文件。图 2.13 显示了通过最终步骤，将知识转化为智慧所生成的图表。

![图](img/2-13.png)

##### 图 2.13 最终步骤产生的图表：将知识转化为智慧

图表包含了你老板问题的答案。如果你将这个图表与图 2.5 进行比较，你可能会注意到很大的不同：图 2.9 和图 2.10 中的图表显示，观众不需要进行任何计算或处理。他们只需阅读结果即可。相反，如果他们查看图 2.5 中的图表，观众必须自己提取信息。

有些人可能会争论在过程中丢失了很多信息，比如其他欧洲国家发生了什么，以及 1994 年至 2018 年之间发生了什么。因此，老板不会仅仅基于图表中显示的数字做出重大投资。这个观察是正确的。然而，我们简化了这个例子，以展示将数据转化为故事的工作流程。在更复杂的情况下，你必须考虑额外的因素，并构建一个适当的故事，其中每个因素都是你数据故事的单个场景。

为了了解 Altair 和 Matplotlib 之间的差异，我们将在下一节中使用 Matplotlib 构建相同的图表。如果你不熟悉 Matplotlib，可以直接跳转到 2.4 节，该节涵盖了第二种方法：使用 Copilot。

### 2.3.4 比较 Altair 和 Matplotlib

你可能会想知道为什么在一本关于 Altair 的书中会有关于 Matplotlib 的章节。答案很简单：本书中描述的方法，包括使用生成式 AI 技术和 DIKW 金字塔的技术，可以应用于所有数据可视化库，而不仅仅是 Altair。实际上，你也可以考虑在其他语言中使用它们，例如 R。作为例子，在本节中，我们简要比较了 Altair 和 Matplotlib，突出了它们之间的差异和相似之处。

Matplotlib 是一个流行的 Python 数据可视化库。很可能，它是你用来快速绘制数据的第一个 Python 库。如果你将 Matplotlib 作为 Altair 的替代品，请考虑 Matplotlib 是一个命令式库，因此你必须定义所有生成图表的步骤。在 Altair 中，你只需关注期望的输出。无论如何，本书中描述的所有与数据叙事相关的概念也适用于 Matplotlib，所以如果你更喜欢命令式方法，可以使用它。

你可以在本书的 GitHub 仓库 `CaseStudies/tourist-arrivals/matplotlib/chart.py` 中找到 Matplotlib 的完整代码，以及在下述列表中。

##### 列表 2.17 在 Matplotlib 中生成图表

```py
import matplotlib.pyplot as plt

df = pd.read_csv('../source/tourist_arrivals_countries.csv', parse_dates=['Date'])
df = pd.melt(df, id_vars='Date', value_name='Tourist Arrivals', var_name='Country')

df.loc[:, 'Year'] = df['Date'].dt.year     #1

df = df.groupby(['Year', 'Country'])['Tourist Arrivals'].sum().reset_index() #2

df = df[(df['Year'] >= 1994) & (df['Year'] <= 2018)]    #3

for country in df['Country'].unique():
    current = df[df['Country'] == country]['Tourist Arrivals']
    base = df[(df['Country'] == country) & (df['Year'] == 1994)]['Tourist Arrivals'].values[0]
    df.loc[df['Country'] == country, 'PI'] = (current - base)/ base*100    #4

fig, ax = plt.subplots(figsize=(10, 6))   #5

for country in df['Country'].unique():
    current = df[df['Country'] == country]['Tourist Arrivals']
    base = df[(df['Country'] == country) & (df['Year'] == 1994)]['Tourist Arrivals'].values[0]
    pi = (current - base)/ base*100
    ax.plot(df[df['Country'] == country]['Year'], pi, label=country)   #6

ax.set_title('Tourist Arrivals in Portugal (1994-2018)')    #7
ax.set_xlabel('')                                           #7
ax.set_ylabel('Percentage Increase since 1994')             #7

ax.legend()       #8

plt.savefig('chart.png', dpi=300)     #9
```

#1 从日期中提取年份

#2 按年份和国家分组

#3 过滤出 1994 年之前和 2018 年之后的年份

#4 添加一个新列，包含每个国家当前年份与 1994 年之间游客到达人数的差异

#5 创建图形和轴

#6 绘制数据

#7 设置标题和轴标签

#8 添加图例

#9 保存图表

备注：为每个国家构建单独的行。

图 2.14 显示了 Matplotlib 生成的图表。通过添加图表线条、数据点或标签，可以使图表更易于阅读和展示。

![figure](img/2-14.png)

##### 图 2.14 Matplotlib 生成的图表

Altair 的代码比 Matplotlib 的代码更冗长。然而，在 Matplotlib 中，你必须单独绘制每条线或文本，而在 Altair 中，你只需传递 DataFrame，库就会为你绘制线条或标记。现在你已经通过手工将数据转化为智慧，并看到了 Altair 和 Matplotlib 之间的差异，我们将转向另一种方法：使用 GitHub Copilot 半自动地构建图表。

## 2.4 第二种方法：Copilot

Copilot 允许你编写代码来自动实现某些功能。我们将通过 Visual Studio Code (VSC)的扩展来启用 Copilot。要安装 VSC，请参考其官方文档（[`code.visualstudio.com/docs/introvideos/basics`](https://code.visualstudio.com/docs/introvideos/basics)），并配置 Copilot，请参阅附录 A。

要使用 Copilot，只需有序地描述要实现的操作序列，Copilot 就会为我们完成剩余的工作，甚至提出不同的实现选项。在本节中，我们将使用 Copilot 构建图表的框架（即其主要部分）。然后，我们将手动添加细节，使用颜色和字体大小。使用 Copilot 的主要困难是用自然语言描述代码需要实现的内容。文本越清晰，Copilot 在实现上就越一致。

在本节中，我们将描述如何将前几节中编写的代码翻译成 Copilot 能理解的文本序列。我们将只关注将图表框架翻译成自然语言，而不考虑细节，如字体大小和颜色。在本书的仓库 `CaseStudies/tourist-arrivals/copilot/chart.py` 下找到 GitHub Copilot 生成的代码。

我们将把问题分解成步骤：

1.  加载和清理数据集

2.  计算百分比增加

3.  在 Altair 中绘制基本图表

4.  丰富图表

我们将在接下来的章节中详细讲解每个步骤。

### 2.4.1 加载和清理数据集

在使用数据集之前，你必须加载所需的库，以 pandas DataFrame 的形式打开数据集，并计算百分比增加。想法是为构建图 2.13 所示的图表准备数据。

打开 VSC，并写下以下注释。

##### 列表 2.18 加载和过滤数据集

```py
# Import the necessary libraries.
# Read the following dataset into a pandas DataFrame, '../source/tourist_
arrivals_countries.csv', and parse the Date field as a date.
# Filter out rows before 1994 and after 2018.
# Extract the year from the Date field and create a new column called Year.
# Group the data by Year and calculate the sum of tourist arrivals for each year.
```

注意：描述加载和过滤数据集的操作序列。

如果你按下最后一个注释的 Enter 键并开始输入 `import`，Copilot 将建议完成该语句。只需按下 Enter 键接受生成的代码。或者，将鼠标悬停在建议的文本上，打开 Copilot，如图 2.15 所示。

![图](img/2-15.png)

##### 图 2.15 如何在 VSC 中打开 Copilot

根据生成的代码，Copilot 可能会提出不同的解决方案，如图 2.16 所示。点击“接受解决方案”以接受一个解决方案。在我们的例子中，第二个解决方案实现了所描述的步骤。请注意，每次运行 Copilot 时，它可能会生成不同的解决方案，即使你重新运行相同的文本。

![图](img/2-16.png)

##### 图 2.16 Copilot 在 VSC 中提出的不同解决方案

### 2.4.2 计算百分比增加

下一步涉及从之前清理过的数据集开始构建一个新的数据集。新的数据集将包含葡萄牙和德国的百分比增加值。写下以下操作。

##### 列表 2.19 使用百分比增加构建新的 DataFrame

```py
# Add a new column to the DataFrame called PI containing the difference for each country between the number of tourist arrivals in the year and 1994.
```

注意：描述如何计算百分比增加。

与前一个案例一样，Copilot 可以提出多个解决方案。打开 Copilot 检查可能的解决方案，并接受最能实现你操作的方案。如果你找不到任何实现你文本的解决方案，通过添加更多细节重写文本。如果在重写文本后，你仍然找不到实现你操作的解决方案，接受最接近你要求的解决方案并根据你的需要进行修改。

### 2.4.3 在 Altair 中绘制基本图表

现在，我们将指导 Copilot 构建一个类似于我们在图 2.12 中看到的图表（除了机场数据集图表），关于百分比增加。

##### 列表 2.20 在 Altair 中构建图表

```py
# Use the Altair library to plot the PI column for each country versus the Year.
```

备注：描述图表应该如何看起来。

Copilot 可以提出多个解决方案。打开 Copilot 检查可能的解决方案，并接受最能实现你操作的方案。

### 2.4.4 丰富图表

指导 Copilot 添加上下文。此外，将生成的图表保存为名为 `output.xhtml` 的 HTML 文件。

##### 列表 2.21 丰富图表

```py
# Create a new chart with the following text: 'Thanks to the introduction\n of low-cost flights,\nPortugal has experienced\\nan increase\\nin tourist arrivals\nof over 200% in 25 years,\\neven surpassing the increase\\nin the other countries.' Use the \n as a line break to format the text. Set the font size to 14\. 
# Place the two graphs side by side. Set title to 'Tourist Arrivals in Portugal (1994-2018)'.
# Save the chart as an HTML. Name the file chart.xhtml.
```

备注：写上注释和标题，以便在图表中显示。

一旦代码完成，保存并运行脚本。图 2.17 显示了生成的图表。

![图](img/2-17.png)

##### 图 2.17 Copilot 生成的图表

图表是一个好的起点，但仍需改进。为了改进图表，你可以手动运行以下步骤：

1.  增加注释的字体大小。

2.  增加 PT 线的笔画宽度。

使用 Copilot 可能看似没有意义，因为一些元素仍然需要手动管理。然而，随着实践，你会意识到 Copilot 对于加快你代码中最简单部分的编写速度是多么有价值。Copilot 会记住你之前的代码，并在你使用它时适应你的编程风格。随着实践，它很可能会成为你的编程伴侣，就像对我一样！

在本章中，你学习了如何使用 Altair 和 Copilot 将原始数据集转化为数据故事。在下一章，我们将回顾 Altair 的基本概念。

## 摘要

+   Altair 是一个用于数据操作和可视化的声明式库。它提供了三个主要元素来构建图表：图表、标记和编码。使用图表元素来指定要绘制的数据集。使用标记来指定要绘制的图表类型，并使用编码来设置通道，如 x 轴和 y 轴。

+   要将原始数据转化为数据故事，逐步使用 DIKW 原则。首先清理你的数据集，然后只提取有意义的信息。绘制提取的信息。然后，添加上下文和行动号召作为注释。

+   Copilot 是一个用于加快代码生成的强大工具。要使 Copilot 生成你的代码，将问题分解成小步骤，并使用自然语言描述每个步骤。尽管 Copilot 并不完美，但它可以帮助你构建代码的框架。

+   使用 Copilot 自动生成你代码的框架，然后手动改进代码。

## 参考文献

### EDA

#### 通用方法

+   Gupta, D., Bhattacharyya, S., Khanna, A., and Sagar, K. (Eds.). (2020). *Intelligent Data Analysis: From Data Gathering to Data Comprehension*. John Wiley & Sons.

+   Rasmussen, R., Gulati, H., Joseph, C., Stanier, C., and Umegbolu, O. (2019). *Data Analyst: Careers in Data Analysis*. BCS Publishing.

+   Khalil, M. (2024). *Effective Data Analysis: Hard and Soft Skills*. Manning Publications.

#### Python

+   Mukhiya, S. K. and Ahmed, U. (2020). *Hands-On Exploratory Data Analysis with Python*. Packt Publishing.

+   Oluleye, A. (2023). *Exploratory Data Analysis with Python Cookbook*. Packt Publishing.

### 工具和库

+   *Altair: Declarative Visualization in Python*. (n.d.). [`altair-viz.github.io/`](https://altair-viz.github.io/)

+   *GitHub*. (n.d.). *GitHub Copilot: Your AI Pair Programmer*. [`github.com/features/copilot`](https://github.com/features/copilot).

### 其他

+   Dobruszkes, F. (2013). The Geography of European Low-Cost Airline Networks: A Contemporary Analysis. *Journal of Transport Geography*, *28*, 75–88.
