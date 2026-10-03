
```py
df_orders_customers = df_orders.merge(df_customers[
↪'customer_id', 'customer_city', 'customer_state']], on='customer_id')
df_orders_customers_sellers = 
↪df_orders_customers.merge(df_order_items[
↪'order_id', 'seller_id']], on='order_id')
df_orders_customers_sellers = 
↪df_orders_customers_sellers.merge(
↪df_sellers[['seller_id', 'seller_city', 'seller_state']], 
↪on='seller_id')
```

### 可视化送达时间、客户位置和卖家位置之间的关系

对于这个目的，一个合适的可视化方法是散点图，我们可以将客户的州和卖家的州分别放在 x 轴和 y 轴上，并根据送达时间着色点。

```py
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(16, 12))
sns.scatterplot(data=df_orders_customers_sellers, x='customer_state', 
↪y='seller_state', hue='delivery_time', palette='coolwarm', alpha=0.6)
plt.xlabel("Customer State")
plt.ylabel("Seller State")
plt.title("Delivery Time by Customer and Seller State")
plt.show()
```

这个可视化将帮助我们识别基于客户和卖家位置的送达时间中的任何模式或趋势。请记住，这是一个高级概述，可能需要进一步分析才能得出更具体的结论。

有趣的是，这次我们收到了端到端的解决方案。从合并所需的 DataFrames，到特征工程（创建一个全新的列），再到可视化！这真是令人印象深刻。

在我们的 GitHub 仓库中的笔记本里，你会发现初始的可视化并不那么令人印象深刻。数据分布明显偏向低值，而高值异常值“劫持”了颜色范围的上部，使得大部分结果几乎无法区分。此外，两个轴上区域的顺序是随机的。然而，解决这些问题并不需要太多工作，代码就能处理这些问题。

#### 列表 3.3 创建对数刻度的热图

```py
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

*# Sort the data by customer_state and seller_state*
df_sorted = df_orders_customers_sellers.sort_values(
↪by=['customer_state', 'seller_state'])

*# Create the scatter plot*
plt.figure(figsize=(16, 12))

*# Use LogNorm for color normalization*
norm = mcolors.LogNorm(
↪vmin=df_orders_customers_sellers['delivery_time'].min() + 1, 
↪vmax=df_orders_customers_sellers['delivery_time'].max())

plt.scatter(x=df_sorted['customer_state'], y=df_sorted['seller_state'], 
↪c=df_sorted['delivery_time'], cmap='coolwarm', alpha=0.6, norm=norm)

plt.xlabel("Customer State")
plt.ylabel("Seller State")
plt.title("Log-scaled Delivery Time by Customer and Seller State")
plt.colorbar(label="Delivery Time (log-scaled)")

*# Get the unique state values and sort them*
sorted_states = sorted(df_sorted['customer_state'].unique())

# Set the same order for both x and y axes
ax = plt.gca()
ax.set_xticks(range(len(sorted_states)))
ax.set_xticklabels(sorted_states)
ax.set_yticks(range(len(sorted_states)))
ax.set_yticklabels(sorted_states)

plt.show()
```

应用生成式 AI 的建议使用对数刻度解决了我们遇到的“蓝色度”问题，区域的字母顺序使得找到相关对变得容易得多。正如你在图 3.10 中看到的，结果是完全可以分析的照片；我们将在第四章中进行实际分析。现在，只需知道确实存在根据区域的不同而不同的送达时间差异即可。

![figure](img/CH03_F10_Siwiak3.png)

##### 图 3.10 根据一个州到另一个州的送达时间散点图，按对数-正态比例着色，两个轴上元素的顺序相同

“当我们谈论品味和颜色时，没有什么可争辩的” 人类感知的一些元素可能难以被计算机程序察觉。如果做得恰当，像颜色分布或元素顺序这样的东西将增强人类吸收呈现信息的能力。你可以尝试从生成式 AI 中获得一些理论背景和有益的建议，但最终结果清晰度和可读性的责任在你。

在本节中，你了解到生成式 AI 可以为你提供执行描述性分析所需的所有必要元素的代码。它并非没有错误，但你必须记住，尽管是计算机程序，LLMs 并不执行实际的计算或转换。它们根据对数学推理中涉及的语言模式的理解来生成答案，而不是通过进行数值计算。提供的答案的准确性取决于相关数学模式在训练数据中的表示程度。然而，描述性分析相对简单。让我们看看我们的 AI 顾问在面对更具挑战性的任务时表现如何。

## 3.4 推断分析

在上一节中，你主要可视化了数据，在过程中以这种方式或那种方式对其进行分组。幸运的是（从商业决策的角度来看），或者不幸的是（从你的工作量角度来看），描述性分析不仅仅是条形图和折线图。甚至比散点图还要多。

假设你的组织最近实施了一种旨在增加产品销售的全新营销策略。该计划在具有不同市场动态和消费者行为的几个地区推出。为了评估策略的有效性，你需要确定的不只是实施后销售是否增加，还要确定观察到的变化是否可以直接归因于营销努力，而不是外部因素或仅仅是运气。此外，你还被要求确定在不同地区和不同人口群体中最有效的特定策略元素。

你需要打开的工具箱，用于解决更高级的问题，被称为推断统计或推理统计。*推断分析*是统计学的一个分支，它侧重于根据从样本中获得的数据对总体进行推断（根据你在样本中找到的内容对总体进行推断——它确实做到了它所说的，真的）。它建立在概率论的基础上，并采用各种统计测试和方法来估计总体参数、检验假设以及评估变量之间关系强度和方向。

以下章节将测试生成式 AI 引导我们穿越这个充满统计学陷阱的迷宫的能力。

### 3.4.1 开始之前

不幸的是，我们需要暂时戴上堂吉诃德帽（或头盔）。我们将向您展示的方法并不流行，你可能会合理地问自己，为什么我们不使用我们在大多数教科书或网站上都能找到的方法？这有几个原因。

首先，我们认为我们的角色是指导你们找到正确做事的方法，这里的正确意味着以允许你得出正确结论的方式分析你的数据。

第二个原因源于生成式 AI 的可用性。以前，你需要花费相当多的时间学习统计学的复杂性才能开始正确做事，而现在，你手头上有所有所需的知识和工具，可以以你感到舒适的方式解析和解释它。此外，这个工具还将提供执行分析的完整代码作为额外奖励。

#### 在推断分析中使用置信区间的必要性

在 3.1 节中，我们提到我们不同意生成式 AI 基于流行度选择评估统计分析结果的方法。我们意识到，对于大多数读者来说，我们以下分析所基于的概念可能很新，因此我们将在此进行概述。

我们已经提到，推断分析的作用是从样本中得出关于总体的结论。想象一下，我们想知道我们公司客户在某个产品类别上的平均每月支出（他们也可能在我们公司之外购买）。由于调查所有客户不可行，我们可能会决定随机选择 100 名客户并计算他们的平均支出。假设结果是 200 美元。

样本可能无法完美地代表我们的整个客户群。然后，我们会计算*置信区间*（CI）来考虑潜在的变异性。平均每月支出的 95%置信区间可能在 180 美元到 220 美元之间。这意味着我们可以有 95%的把握，所有客户在这个产品类别上的*真实*平均支出位于这个范围内。换句话说，如果我们重复抽样和置信区间计算 100 次，95 次计算出的置信区间将包含真实均值。

相比于更广泛使用的零假设显著性检验（NHST）或 p 值，置信区间提供了一定优势，因为它们为总体参数提供了一系列可能的值，从而允许更全面地理解效应大小及其不确定性。

因此，置信区间有助于更好地解释结果和更明智的决策。这就是为什么，尽管 ChatGPT 最初提出的基于普遍流行度的共识方法，我们还是修改了以下小节中将要讨论的问题的分析设计，并明确要求我们的 AI 顾问专注于使用置信区间的方法。

#### 关于数据完整性的重要说明

我们在这里使用推断统计，因为我们知道我们所拥有的数据是更大集合的一个样本。如果你处理的是完整的数据集，例如所有客户和所有交易，那么在历史数据分析的目的上，就没有必要计算置信区间，因为你的样本就是你的总体。然而，如果你想要将数据用于任何类型的预测分析，推断统计将非常有用。

我们现在可以重新投入到分析我们的电子商务数据中。

### 3.4.2 产品属性与运输成本之间的关系

我们 AI 顾问提出的“产品属性（例如，重量、尺寸）与间接成本（例如，运输成本）之间的关系是什么？”这个问题可能看起来有点天真，但它将帮助我们建立关于基于置信区间的线性回归能力的直觉。这个工具的重要性无法过高估计。你将在各种领域的工作中遇到它：

+   *经济学*—经济学家使用线性回归来理解经济指标之间的关系，例如 GDP 增长和失业率，或通货膨胀和利率。在这些回归中的置信区间有助于政策制定者在制定财政或货币政策决策时评估这些关系的可靠性。

+   *金融*—在金融领域，线性回归用于根据资本资产定价模型（CAPM）描述的资产风险和回报之间的关系进行建模。分析师使用置信区间来衡量β系数的精确度，β系数衡量资产回报对市场运动的敏感性，从而为投资策略提供信息。

+   *医疗保健*—研究人员使用线性回归来研究各种风险因素（例如，吸烟、饮食、身体活动）与健康状况（例如，心脏病、糖尿病）之间的关系。回归系数周围的置信区间提供了关于这些因素的重要性和可靠性的见解，指导公共卫生建议和患者治疗方案。

+   *市场营销*—市场营销分析师使用线性回归来理解市场营销策略的不同方面（例如，广告支出、促销活动）如何影响销售或客户行为。置信区间有助于评估这些关系的强度，优化营销预算，并定制策略以最大化投资回报率。

+   *房地产*—线性回归模型用于根据特征（如大小、位置、便利设施）预测房地产价格。预测周围的置信区间为潜在买家和卖家提供预期价格的范围，有助于谈判和决策过程。

+   *环境科学*—科学家应用线性回归来模拟人类活动对环境指标（如 CO[2]排放和全球温度上升）的影响，或环境变量对生物多样性的影响。置信区间有助于量化这些模型中的不确定性，这对于环境政策和保护策略至关重要。

+   *制造业*—在制造业中，线性回归用于模拟生产过程（例如温度、压力、材料特性）与产品质量或产量的关系。置信区间允许工程师了解这些过程中的变化性，并优化它们以获得更好的结果。

+   *教育*—研究人员使用线性回归来探索教育干预措施（例如班级规模、教学方法、技术使用）对学生表现的影响。置信区间提供了这些干预措施的有效性和可靠性的衡量标准，为教育政策和实践提供信息。

+   *体育分析*—线性回归在体育分析中应用于根据各种指标（例如得分、防守动作）评估球员的表现及其对比赛结果的影响。这些估计值周围的置信区间帮助球队做出关于球员训练和比赛策略的战略决策。

+   *技术与创新*—在科技公司中，线性回归可以分析产品或服务特征与用户参与度或满意度之间的关系。这些关系的置信区间指导产品开发、功能优先级和客户体验策略。

在我们看来，在决策过程中定量检查一件事对另一件事的依赖程度是不可或缺的。幸运的是，生成式 AI 可以帮助你快速高效地完成整个过程。

**![image](img/Init-MA.png)**如何检查产品属性（例如重量、尺寸）与运输成本之间是否存在关系？

**![image](img/Response.png)**为了调查产品属性（例如重量、尺寸）与运输成本之间的关系，你可以使用带有置信区间的线性回归模型。首先，让我们从`df_products`和`df_order_items`数据框中合并必要的数据：

```py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table

*# Merge the DataFrames*
merged_df = df_order_items.merge(df_products, on="product_id", ↪how="left")

*# Calculate product volume*
merged_df["product_volume"] = merged_df["product_length_cm"] * 
↪merged_df["product_height_cm"] * merged_df["product_width_cm"]

*# Filter out the relevant columns*
analysis_df = merged_df[
↪"freight_value", "product_weight_g", "product_volume"]]

*# Remove rows with NaN values*
analysis_df = analysis_df.dropna()
```

接下来，让我们为每个产品属性（重量和体积）与运输成本（运费价值）建立线性回归模型：

```py
*# Linear regression for weight*
X_weight = sm.add_constant(analysis_df["product_weight_g"])
y = analysis_df["freight_value"]
model_weight = sm.OLS(y, X_weight).fit()

*# Linear regression for volume*
X_volume = sm.add_constant(analysis_df["product_volume"])
y = analysis_df["freight_value"]
model_volume = sm.OLS(y, X_volume).fit()
```

现在，你可以可视化这些关系及其置信区间：

```py
*# Function to plot regression line with confidence interval*
def plot_regression_with_ci(X, y, model, xlabel, ylabel):
    fig, ax = plt.subplots()
    sns.regplot(x=X.iloc[:, 1], y=y, ax=ax, scatter_kws={'alpha': 0.3})
    st, data, ss2 = summary_table(model, alpha=0.05)
    predicted = data[:, 2]
    upper = data[:, 4]
    lower = data[:, 5]
    ax.plot(X.iloc[:, 1], predicted, '-', lw=2, color='red', 
    ↪label='Fitted Line')
    ax.fill_between(X.iloc[:, 1], lower, upper, color='red', alpha=0.15, 
    ↪label='95% CI')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.show()

*# Plot the relationship between weight and freight value*
plot_regression_with_ci(X_weight, y, model_weight, 
↪xlabel="Product Weight (g)", ylabel="Freight Value")

*# Plot the relationship between volume and freight value*
plot_regression_with_ci(X_volume, y, model_volume, 
↪xlabel="Product Volume (cm³)", ylabel="Freight Value")
```

通过检查图表及其 95%置信区间，你可以确定产品属性与运输成本之间是否存在关系。拟合线周围的狭窄置信区间表明变量之间存在更强的关系，而较宽的区间则表明关系中的不确定性更大。

我们的 AI 顾问提出了一个完整的解决方案。它构建了所需的 DataFrame，建立了回归模型，并提出了结果的可视化。后者结果比我们预期的要杂乱得多，但仍然在图 3.11 中展示。

![figure](img/CH03_F11_Siwiak3.png)
