
### 图 3.11 展示了（a）货运价值与产品重量以及（b）货运价值与体积之间的关系散点图

在图 3.11 中，在 30,000 克和 25,000 立方厘米处有一些有趣的数据点的垂直线。我们猜测这是由于巴西的快递公司使用的某些标准包装尺寸造成的。

值得讨论的另一件事是，置信区间如此之窄，以至于在图片中甚至都看不见。这意味着从统计学的角度来看，这条线最有可能就是我们画出的那条线——我们拥有如此多的数据，我们可以确信我们的估计精度。但这并不意味着这个回归分析立即就有用。

结果在回归线周围分布得很广。至少可以说，它的预测能力似乎很低。例如，如果我们有一个 20 公斤的包裹，我们可以估计其货运价值在 0 到 200 BRL 之间（这是我们的假设，即数据提供者使用的货币单位）。

仓鼠能叼多少木头？回归分析的主要价值在于它们能够根据一组独立变量预测因变量。为了评估回归模型的表现，我们可以使用各种指标，例如平均绝对误差（MAE）、平均平方误差（MSE）、均方根误差（RMSE）和 R²（确定系数）。这些指标提供了关于模型准确度、误差幅度以及独立变量解释的因变量变差比例的见解。此外，残差分析和诊断图可以用来检验模型的假设，例如线性、同方差性和残差的正态性。

我们承诺您将进行依赖性的定量分析，这次我们不会仅仅止步于绘制图表。以下是在分析回归数据（尤其是在其预测能力方面）时您可能想要查看的一些常用指标：

+   *平均绝对误差* *(MAE)* —预测值和实际值之间绝对差的平均值。它给出了预测错误程度的一个概念。较小的 MAE 表示更好的预测性能。

+   *均方误差* *(MSE)* —预测值和实际值之间平方差的平均值。它倾向于对大误差进行更多惩罚。较小的 MSE 表示更好的预测性能。

+   *均方根误差* *(RMSE)* —MSE 的平方根。它与因变量的单位相同，比 MSE 更易于解释。较小的 RMSE 表示更好的预测性能。

+   *R*²——R²衡量的是因变量方差中可以被自变量（或多个自变量）解释的比例。R²值越高，表明预测性能越好。

如往常一样，接受生成式 AI 的帮助是获取计算所有这些值的代码的最快方式。

### 列表 3.4 预测产品重量或体积对货运价值的影响

```py
from sklearn.metrics import mean_absolute_error, mean_squared_error, 
↪r2_score

*# Split the data into features (X) and target (y)*
X_weight = analysis_df["product_weight_g"].values.reshape(-1, 1)
X_weight = sm.add_constant(X_weight)  *# Add a constant for the intercept*
X_volume = analysis_df["product_volume"].values.reshape(-1, 1)
X_volume = sm.add_constant(X_volume)  *# Add a constant for the intercept*

y = analysis_df["freight_value"].values

*# Predict the target variable using the models*
y_pred_weight = model_weight.predict(X_weight)
y_pred_volume = model_volume.predict(X_volume)

*# Calculate the evaluation metrics for both models*
mae_weight = mean_absolute_error(y, y_pred_weight)
mse_weight = mean_squared_error(y, y_pred_weight)
rmse_weight = np.sqrt(mse_weight)
r2_weight = r2_score(y, y_pred_weight)

mae_volume = mean_absolute_error(y, y_pred_volume)
mse_volume = mean_squared_error(y, y_pred_volume)
rmse_volume = np.sqrt(mse_volume)
r2_volume = r2_score(y, y_pred_volume)

*# Print the evaluation metrics*
print(f"Product Weight Model: MAE={mae_weight:.2f}, MSE={mse_weight:.2f}, 
↪RMSE={rmse_weight:.2f}, R2={r2_weight:.2f}")
print(f"Product Volume Model: MAE={mae_volume:.2f}, MSE={mse_volume:.2f}, 
↪RMSE={rmse_volume:.2f}, R2={r2_volume:.2f}")
```

建议的代码应该提供以下结果：

```py
Product Weight Model: MAE=6.89, MSE=156.77, RMSE=12.52, R2=0.37
Product Volume Model: MAE=7.41, MSE=163.70, RMSE=12.79, R2=0.34
```

结果显示，大约 35%的货运价值变异性可以通过产品重量或体积来解释。看看将两者结合起来是否会增加可用的预测能力可能很有趣。

### 列表 3.5 预测产品重量和体积对货运价值的影响

```py
import pandas as pd
import numpy as np
import statsmodels.api as sm

*# Merge the DataFrames*
merged_df = df_order_items.merge(df_products, on="product_id", how="left")

*# Calculate product volume*
merged_df["product_volume"] = merged_df["product_length_cm"] * 
↪merged_df["product_height_cm"] * merged_df["product_width_cm"]

*# Filter out the relevant columns*
analysis_df = merged_df[
↪["freight_value", "product_weight_g", "product_volume"]]

*# Remove rows with NaN values*
analysis_df = analysis_df.dropna()

*# Define the independent variables (weight and volume)* 
*↪and the dependent variable (freight value)*
X = analysis_df[['product_weight_g', 'product_volume']]
y = analysis_df['freight_value']

*# Add a constant for the intercept*
X = sm.add_constant(X)

*# Fit the model*
model = sm.OLS(y, X).fit()

*# Print the summary*
print(model.summary())
```

结果可能看起来有些令人畏惧，但如果你敢于尝试，生成式 AI 应该能帮助你理解以下所有参数。

```py
 OLS Regression Results
===========================================================================
Dep. Variable:       freight_value   R-squared:                       0.399
Model:                         OLS   Adj. R-squared:                  0.399
Method:              Least Squares   F-statistic:                 3.744e+04
Date:             Sun, 09 Apr 2023   Prob (F-statistic):               0.00
Time:                     15:38:56   Log-Likelihood:            -4.4203e+05
No. Observations:           112632   AIC:                         8.841e+05
Df Residuals:               112629   BIC:                         8.841e+05
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
===========================================================================
                     coef     std err         t    P>|t|    [0.025   0.975]
---------------------------------------------------------------------------
const             13.7268    0.044      314.740    0.000    13.641  13.812
product_weight_g   0.0016    1.63e-05   101.023    0.000     0.002   0.002
product_volume     0.0002    2.61e-06    70.759    0.000     0.000   0.000
===========================================================================
Omnibus:                105287.730   Durbin-Watson:                   1.815
Prob(Omnibus):               0.000   Jarque-Bera (JB):         12678300.032
Skew:                        4.145   Prob(JB):                         0.00
Kurtosis:                   54.311   Cond. No.                     3.37e+04
===========================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors 
    ↪is correctly specified.
[2] The condition number is large, 3.37e+04\. This might indicate 
    ↪that there are strong multicollinearity or other numerical problems.
```

注释：你可以从处理金钱的人的角度，而不是从基础科学的角度，从 Joshua Angrist 和 Jörn-Steffen Pischke 的《Mostly Harmless Econometrics》（普林斯顿大学出版社，2009 年）一书中了解很多关于统计学中使用的测量和参数。

这里关键的信息应该是 R²，它告诉你几乎 40%的货运价值变异性可以通过产品重量和体积的变异性来解释。这意味着通过控制和优化这两个变量，你可以影响货运价值。如果货运价值是产品总成本的一个关键因素，这个结果可能导致这样的结论：尝试在不降低产品功能或质量的情况下减少产品的重量或尺寸是值得的。这可能导致货运成本的节约。

“乞丐没有选择权”——正如我们在第二章中解释的，结果的质量在很大程度上取决于数据的质量和可访问性。如果可以选择，你应该避免基于图 3.11 中看到的分布的数据提出业务解决方案。观察到的效果也不太显著。然而，关键短语是“如果可以选择”。有时你将没有更好的数据，而你的业务可能需要任何它能得到的支持。依赖于甚至不完美的结果，可能意味着让业务沉没和给予它战斗机会之间的区别。

我们将在第四章中花更多的时间从这个结果中得出结论。现在，我们将把注意力转回到审查分数上。

### 3.4.3 产品、交易、运输属性与审查分数之间的关系

如果你的老板看到我们在本节中将要做的——将我们 AI 顾问提出的三个研究问题合并为一个，他们将会为你感到骄傲：

+   “哪些因素（例如，交货时间、价格、产品属性）导致了平均审查分数的差异？”

+   “不同的支付方式和分期付款的数量如何影响订单完成率和客户满意度？”

+   “配送时间（实际与估计）对客户满意度和评论分数有什么影响？”

回答它们的方法是相同的，所以为什么浪费时间（也就是钱）做同样的工作三次呢？好吧，那只有在你不太关心结果的可靠性和准确性时才会有效。这是一个重要的教训：即使技术上可以做到某事，并不意味着你应该这么做。

少即是多  在回归分析中，你可能希望将更多的独立变量添加到模型中，以获得更好的结果，但这并不一定是正确的。过多的变量可能会使问题复杂化，换句话说，它们可能会导致模型过拟合或引起多重共线性问题。过拟合意味着你的模型开始拟合噪声，而不是潜在的模式，这通常发生在你拥有的参数数量相对于观测数过多的情况下。多重共线性发生在两个或更多预测变量高度相关时——估计系数可能会变得不稳定，并对数据的微小变化敏感。这种不稳定性使得可靠地解释哪些变量真正有影响以及它们对解释因变量的贡献程度变得困难。

我们对产品和交易过程只有有限的信息。我们还知道，大多数交易都是在交付时完成的。这应该使我们不要期望在找到这些因素与客户在评论中表达的幸福程度之间的相关性方面取得巨大成功。然而，我们认为研究这种关系的技术值得学习。

**![image](img/Init-MA.png)**我应该使用哪种推断统计方法来研究平均评论分数在不同产品类别中的变化，以及哪些因素（例如，配送时间及其与估计的差异、价格、产品和支付属性）导致这些差异？

**![image](img/Response.png)**要使用推断统计方法研究平均评论分数在不同产品类别中的变化以及配送时间、价格和产品属性等因素如何导致这些差异，你可以遵循以下步骤：

1.  准备数据：

    1.  创建包含必要列的 DataFrame：产品类别、评论分数、配送时间、价格和产品属性（例如，重量、尺寸）。

    1.  合并必要的 DataFrames 以获取这些信息。

    1.  确保没有缺失值，并在必要时将分类变量转换为数值格式。

1.  执行有序逻辑回归：

    1.  由于评论分数是一个有序变量（1-5），你可以使用有序逻辑回归来建模评论分数与感兴趣因素（产品类别、配送时间、价格和产品属性）之间的关系。

    1.  要做到这一点，你可以使用来自 mord 包的`mord.LogisticAT`模型。

1.  调整模型：

    1.  将数据分为训练集和测试集。

    1.  在训练数据上拟合有序逻辑回归模型。

1.  评估模型：

    1.  使用模型来预测测试数据上的评论分数。

    1.  使用混淆矩阵和其他指标（如精确度、召回率和 F1 分数）来计算预测的准确度。

    1.  分析模型的系数以确定每个因素在预测评论分数中的重要性。

这里有一个示例代码，帮助您开始：
