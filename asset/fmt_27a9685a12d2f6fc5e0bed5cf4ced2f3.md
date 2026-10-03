
……有很大可能性生成式 AI 能够解决这个问题。图 2.6 展示了按月聚合的评分随时间分布。

我们可以看到，2016 年 11 月和 12 月低分有显著上升，也许，只是也许，春季低分有周期性上升，但信号如此微弱，除非我们想要运行一些精确的预测分析，否则我们不会深入挖掘。

我们可以安全地得出结论，从这一部分来看，生成式 AI 可能会犯错误，但它们仍然可以帮助你编写和修改代码，例如进行时间序列分析，并且它们并不羞于使用真正先进的库和算法（Lomb-Scargle 是一种由生物学家开发的用于分析生物周期的算法，后来被天文学家采用来寻找星光观测中的模式）。我们相信，通过一点耐心和迭代的方法来明确方法要求和你的数据属性，生成式 AI 将成为你分析过程的最大促进者。

![图](img/CH02_F06_Siwiak3.png)

##### 图 2.6 随时间变化的订单分数分布—月度汇总

### 2.6.3 神秘变量调查

如果神秘变量还没有成为你作为数据分析师日常生活中的困扰，那么你可能会期待它们的出现。我们提到过，并且还会再次强调，没有任何东西能比坚实的元数据更有效，但像详尽的文档和数据管道这样的东西，在日益增加的时间压力下被视为一种奢侈。因此，你的一些数据可能来自文档记录不佳的系统。尽管我们强烈反对出于任何原因（除了为你的服务器室供暖）使用未经证实的数据，但有时可以推断出一些变量的含义，达到令人满意的确定性水平。

为了展示如何处理一个神秘变量的例子，让我们假装对我们示例数据中的一列的含义一无所知。让我们尝试深入了解`review_answer_timestamp`列。戴上我们的福尔摩斯帽子，让我们看看是否有其他列可能与它相关。仅凭直觉，我们将检查它与`review_creation_date`的关系。这两个列都与评论相关，并且在某种形式上与时间相关。

但是，两个列相关意味着什么呢？一种本能的反应可能是检查相关性，但在时间序列的情况下，这可能会是一个风险很大的提议。相关性衡量两个变量之间的线性关系，假设这种关系在整个数据集中是一致的。然而，这种假设在时间序列数据中往往不成立，因为自相关、趋势、季节性和外部冲击等因素。*自相关*指的是信号与其延迟版本的关联，这是时间序列数据的一个常见特征，可能会夸大两个时间序列之间的相关性测量。同样，*趋势*和*季节性*可以产生虚假的相关性，这并不反映真正的因果关系，而是反映了两个序列都共享的潜在模式的存在。此外，影响两个序列的*外部冲击*可能会暂时增加相关性，但并不表示有意义的或持久的关系。

在我们的案例中，关键因素是两列中的值都与订单相关，而这些订单本身是以时间序列的形式放置的，因此它们*将会*相关。无论这些列代表什么，它们的值都会随着订单的增长而增长。那么，我们该如何定义这种关系呢？

也许我们不会呢？我们已经测试了生成式 AI 在提出想法方面的能力，并发现其理解问题和提出解决方案的能力是值得称道的。

**![image](img/Init-MA.png)**你能提出我如何调查`review_answer_timestamp`和`review_creation_date`列之间的关系吗？

**![image](img/Response.png)**为了调查`review_answer_timestamp`和`review_creation_date`之间的关系，你可以计算这两个时间戳之间的时间差异，并分析时间差异的分布。这将让你了解通常需要多长时间来回答一条评论。

这里是分析这两列之间关系的逐步指南：

1. 计算从`review_answer_timestamp`到`review_creation_date`的时间差异。

```py
df_order_reviews['time_diff'] = 
↪(df_order_reviews['review_answer_timestamp'] – 
↪df_order_reviews['review_creation_date']).dt.total_seconds()
```

`time_diff`列现在包含秒为单位的时间差异。

2. 计算时间差异的基本统计量，以获得分布的概述。

```py
time_diff_summary = df_order_reviews['time_diff'].describe()
print(time_diff_summary)
```

3. 使用直方图或箱线图可视化时间差异的分布。

```py
import matplotlib.pyplot as plt
# Histogram
df_order_reviews['time_diff'].plot(kind='hist', bins=50)
plt.xlabel('Time Difference (Seconds)')
plt.ylabel('Frequency')
plt.title('Distribution of Time Difference Between Review Creation and 
↪Answer')
plt.show()

# Box plot
df_order_reviews['time_diff'].plot(kind='box', vert=False)
plt.xlabel('Time Difference (Seconds)')
plt.title('Box Plot of Time Difference Between Review Creation and
↪Answer')
plt.show()
```

通过分析`review_answer_timestamp`和`review_creation_date`之间时间差异的分布，你可以了解典型的响应时间，并识别数据中的任何异常值或趋势。

如果需要，请随意将时间差异转换为更合适的单位（例如，小时或天）。你还可以通过观察时间差异如何与其他因素（如评分或特定时间段，例如一周中的某一天或一年中的某月）变化来进一步研究这种关系。

检查两个调查列中日期之间时间差异的分布是完全合理的。让我们应用提出的代码。

时间差异的总结分析证明，默认的秒数时间范围可能不是分析中最有用的。

```py
count    9.922400e+04
mean     2.720730e+05
std      8.545003e+05
min      7.709000e+03
25%      8.682075e+04
50%      1.447155e+05
75%      2.681480e+05
max      4.481561e+07
Name: time_diff, dtype: float64
```

然而，我们的 AI 顾问提到，秒可能不是这种数据分析的最佳单位。幸运的是，将其转换为其他时间单位就像除以一个常数一样简单。由于懒惰，我们要求生成式 AI 修改代码。它甚至处理了适当的轴标题。结果图像展示在图 2.7 中。

根据时间分布来看，有些答案在几小时内提供，而有些则在数月后提供，这似乎表明某些人类交互推动了这种差异。

![figure](img/CH02_F07_Siwiak3.png)

##### 图 2.7 `review_answer_timestamp`和`review_creation_date`列之间时间差异的分布，以直方图（a）和箱线图（b）表示

让我们尝试应用逻辑。我们有一个评论创建和评论回答。由于数据来自在线平台，第一个直觉是用户在购买后创建评论，卖家在一段时间后回复评论者。

这个例子表明，生成式 AI 和人类专业知识的结合很容易被模糊的列命名约定和缺乏元数据所击败。实际上，`review_creation_date` 列包含发送调查给客户的日期，而 `review_answer_timestamp` 指示客户何时回答。这是始终确认数据创建者使用的命名约定的另一个原因。

### 2.6.4 数据协调

让我们把注意力转向每份订单的评论数量这一结构性问题。这里有一个技巧：你真的确定那些收集到的评论是关于什么的吗？获取同一产品的多个评论是一回事。很多人购买同一件商品并表达他们通常多样化的意见并不罕见。然而，获取同一订单的多个评论则需要你的注意，因为你需要决定在多个评分的情况下如何评估订单满意度。你会使用最后的评论还是平均值？

在我们的数据集中，有 547 个订单有多条评论。现在我们应该检查这些评论是否来自下单的客户，或者来自多个客户，这会令人担忧。为了回答这个问题，我们需要协调我们的数据并连接不同的来源以形成一个连贯的整体。

假设我们感到懒惰（这是一个合理的假设），并且我们想让我们的生成式 AI 承担繁重的工作。

**![image](img/Init-MA.png)** 我还有一个名为 `df_customers` 的 DataFrame。有两个列 `customer_id` 和 `customer_unique_id`。我该如何在之前讨论的 `df_order_reviews` 中创建一个额外的 `customer_unique_id` 列，并将其正确地分配给已经存在的 `customer_id`？

**![image](img/Response.png)** 要根据 `customer_id` 列将 `customer_unique_id` 列添加到 `df_order_reviews` DataFrame 中，你可以使用 pandas 中的 `merge()` 函数。以下是操作的步骤：

在 `customer_id` 列上合并 `df_order_reviews` 和 `df_customers` DataFrame。

```py
merged_df = df_order_reviews.merge(df_customers, on='customer_id', 
↪how='left')
```

这将执行左连接，意味着 `df_order_reviews` 中的所有行都将包含在 `merged_df` DataFrame 中，并且根据 `customer_id` 列匹配 `df_customers` 中的相应 `customer_unique_id` 值。

现在，`merged_df` DataFrame 包含了一个额外的 `customer_unique_id` 列，每个 `customer_id` 都被分配了正确的值。如果你想更新原始的 `df_order_reviews` DataFrame 而不是创建一个新的，你可以简单地将 `merged_df` 赋值给 `df_order_reviews`：

```py
df_orders_reviews = merged_df
```

现在有一段不错的代码，但没有权利工作！它试图将客户分配给 `df_order_reviews` DataFrame 中的订单。然而，`df_order_reviews` 中的任何列都无法直接完成这项操作。我们知道我们唯一实际的标识符是 `order_id` 列，因为我们做了作业并了解到 `customer_id` 实际上并不能识别客户。

客户永远是对的  生成式 AI 既不会质疑也不会纠正你。如果你让他们相信某件事，他们会假设你有你的理由。尽管，他们对怀疑的容忍度有限。我们试图说服生成式 AI 地球是平的，尽管他们为了讨论的目的同意保持这个假设，但他们时不时地提醒我们地球实际上是圆的。然而，他们的容忍度边界设置得相当远。

我们可以通过两步过程得到期望的结果，首先在 `customer_id` 字段上连接 `df_orders` 和 `df_customers`。这将给我们一个包含每个订单的 `customer_unique_id` 的 DataFrame。然后，我们可以在这个 DataFrame 上连接 `df_order_reviews` 的 `order_id` 字段。或者，我们也可以从在 `order_id` 字段上连接 `df_orders` 和 `df_order_reviews` 开始。然后，我们可以将得到的 DataFrame 连接到 `df_customers` 的 `customer_id` 字段。希望你现在能明白为什么初始的数据分析是至关重要的。你需要查看所有 DataFrame，才能有效地协调来自多个来源的数据。

连接、合并和连接表可能对你来说很直观，但对其他人来说，数据工程 101 中的这部分内容可能仍然值得关注。在过去，只知道“需要发生什么”是编写代码以执行所需操作所必需的，但并不充分。我们在许多数据分析师的职位描述中看到了计算机科学学位的要求。让我们利用这个机会来检查今天在编码经验上的差距是否可以通过生成式 AI 来弥补。

**![image](img/Init-MA.png)**我有一个名为 `df_order_reviews` 的 DataFrame，其中包含 `order_id` 列。我还有一个名为 `df_orders` 的 DataFrame，其中包含 `order_id` 和 `customer_id` 列。此外，我还有一个名为 `df_customers` 的 DataFrame，其中包含 `customer_id` 和 `customer_unique_id` 列。我该如何将 `df_customers` 中的有效 `customer_unique_id` 分配给 `df_order_reviews` 中的每个 `order_id`？

**![image](img/Response.png)**要将 `df_customers` 中的有效 `customer_unique_id` 分配给 `df_order_reviews` 中的每个 `order_id`，你首先需要将 `df_orders` 与 `df_customers` 合并，以创建包含 `order_id` 和 `customer_unique_id` 列的 DataFrame。然后，将这个新的 DataFrame 与 `df_order_reviews` 合并。以下是逐步操作的方法：

1. 在 `customer_id` 列上合并 `df_orders` 和 `df_customers` DataFrame。

```py
orders_customers = df_orders.merge(df_customers, on='customer_id', 
↪how='left')
```

现在，`orders_customers` DataFrame 包含 `order_id` 和 `customer_unique_id` 列。

2. 在 `order_id` 列上合并 `df_order_reviews` 和 `orders_customers` DataFrame。

```py
merged_df = df_order_reviews.merge(orders_customers[['order_id', 
↪'customer_unique_id']], on='order_id', how='left')
```

这将执行左连接，意味着 `df_order_reviews` 中的所有行都将包含在 `merged_df` DataFrame 中，并且根据 `order_id` 列匹配 `orders_customers` 中的相应 `customer_unique_id` 值。

现在，`merged_df` DataFrame 包含了一个额外的 `customer_unique_id` 列，为每个 `order_id` 分配了正确的值。如果你想更新原始的 `df_order_reviews` DataFrame 而不是创建一个新的，你只需将 `merged_df` 赋值给 `df_order_reviews` 即可：

```py
df_order_reviews = merged_df
```

现在，你的 `df_order_reviews` DataFrame 已经为每个 `order_id` 正确分配了 `customer_unique_id` 列。

这次，代码按预期工作。生成式 AI 确实可以为你提供代码；它们知道 *如何* 做这件事（即使有时需要严厉的推动才能让某件事工作）。关于 *什么* 需要发生，决定权仍然在你（即使最初的对话为你提供了建议）。

现在，为了完成这个练习的部分，让我们检查是否有多条评论来自不同的顾客；如果是这样，你可能需要通知你的 IT 部门，因为可能会出现错误地将评论分配给订单的情况。以下代码应该会给你所需的答案（而且，不，我们并没有自己编写这段代码）。

```py
unique_customers_per_order =
↪df_order_reviews.groupby('order_id')['customer_unique_id'].nunique()
multiple_customers_orders =
unique_customers_per_order[unique_customers_per_order > 1]
if len(multiple_customers_orders) > 0:
    print(f"There are {len(multiple_customers_orders)} orders with reviews 
    ↪from multiple customers.")
    print("\nOrder IDs and the number of unique customers:")
    print(multiple_customers_orders)
else:
    print("There are no orders with reviews from multiple customers.")
```

这段代码通过统计唯一的 `customer_unique_id` 值将 `df_order_reviews` DataFrame 进行分组，检查分配给 `order_id` 的唯一 `customer_unique_id` 值的计数是否大于 `1`，最后打印结果。所有这些多次评论都来自同一顾客。这可以节省你前往 IT 部门的行程，甚至允许分析关于订单的意见随时间变化的普遍性。

我们可以假设你现在要么足够了解你的数据，要么知道如何使用足够优雅和知识渊博的生成式 AI 模型来帮助你深入了解你的数据结构和属性。在下一章中，我们将专注于业务分析本身。

##### 询问生成式 AI 的事项

+ 来自 `<some_business_area>` 的数据通常的数据结构是什么？

+ 我应该在提供的 DataFrame 中测试哪些数据质量问题？
