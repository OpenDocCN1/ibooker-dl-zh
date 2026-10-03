
# 使用有序逻辑回归分析评论分数

```py
import pandas as pd
import mord
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Prepare the data (assuming you have already merged necessary
# DataFrames into analysis_df)

# Convert categorical variables to numerical format if required
# (e.g., using pd.get_dummies() for product categories)

# Split the data into training and testing sets
X = analysis_df.drop("review_score", axis=1)
y = analysis_df["review_score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
↪random_state=42)

# Fit the ordinal logistic regression model
model = mord.LogisticAT(alpha=0)
model.fit(X_train, y_train)

# Predict the review scores on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the predictions
print("Accuracy:", accuracy_score(y_test, y_pred))

# Display the confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Analyze the model's coefficients
print("Coefficients:\n", model.coef_)
```

使用这种方法，您可以使用有序逻辑回归探索平均评论分数与各种因素之间的关系。

如果代码中的注释“假设您已经将必要的 DataFrames 合并到 `analysis_df` 中”没有引起您的警觉，那么我们到目前为止做得还不够好。每当您看到生成式 AI 方面的任何假设时，您都需要感到震惊，暗自咒骂，并仔细检查它们。我们不会包括我们试图具体说明生成式 AI 如何将 `analysis_df` 假想为的整个对话。您可以在列表 3.6 中看到最终的代码。

## 列表 3.6 准备 `analysis_df` DataFrame

```py
# Merge df_products with df_product_category_translation
products_merged = pd.merge(
↪df_products, df_product_category_translation, on='product_category_name')

# Merge df_orders with df_order_items
order_items_merged = pd.merge(df_orders, df_order_items, on='order_id')

# Merge the resulting DataFrame with df_customers
order_customer_merged = pd.merge(
↪order_items_merged, df_customers, on='customer_id')

# Merge the resulting DataFrame with products_merged
order_product_merged = pd.merge(
↪order_customer_merged, products_merged, on='product_id')

# Merge the resulting DataFrame with df_sellers
order_seller_merged = pd.merge(
↪order_product_merged, df_sellers, on='seller_id')

# Merge the resulting DataFrame with df_order_reviews
order_review_merged = pd.merge(
↪order_seller_merged, df_order_reviews, on='order_id')

# Merge the resulting DataFrame with df_order_payments
analysis_df = pd.merge(
↪order_review_merged, df_order_payments, on='order_id')

# Calculate delivery time in days
analysis_df['delivery_time'] = 
↪(analysis_df['order_delivered_customer_date'] – 
↪analysis_df['order_purchase_timestamp']).dt.days

# Calculate the difference between actual vs. estimated delivery time
analysis_df['delivery_time_misestimation'] = 
↪(pd.to_datetime(analysis_df['order_estimated_delivery_date']) – 
↪pd.to_datetime(analysis_df['order_delivered_customer_date'])).dt.days

# Calculate product volume
analysis_df['product_volume'] = analysis_df['product_length_cm'] * 
↪analysis_df['product_height_cm'] * analysis_df['product_width_cm']

# Apply one-hot encoding to the payment_type column
payment_type_dummies = pd.get_dummies(
↪analysis_df['payment_type'], prefix='payment_type')

# Concatenate the one-hot encoded columns with analysis_df
analysis_df = pd.concat([analysis_df, payment_type_dummies], axis=1)

# Drop unnecessary columns
analysis_df.drop(columns=['product_category_name', 'order_approved_at', 
↪'order_delivered_carrier_date', 'order_estimated_delivery_date', 
↪'shipping_limit_date', 'review_creation_date', 
↪'review_answer_timestamp', 'order_id', 'customer_id', 'order_status', 
↪'order_purchase_timestamp', 'order_delivered_customer_date', 
↪'order_item_id','product_id', 'seller_id', 
↪'product_category_name_english', 'seller_zip_code_prefix', 
↪'seller_city', 'seller_state', 'review_comment_message', 
↪'review_comment_title', 'review_id', 'product_length_cm', 
↪'product_height_cm', 'product_width_cm', 'customer_unique_id', 
↪'customer_zip_code_prefix', 'customer_city', 'customer_state', 
↪'payment_type'], inplace=True)

analysis_df = analysis_df.dropna()
```

前面代码生成的 DataFrame 将不含所有关于标识符、地理位置和时间戳的不必要信息，因为它们将不会用于这次特定的分析，并且会成为不必要的噪音。我们希望关注感兴趣的变量及其对评论分数的影响。

## 逻辑回归分析结果

逻辑回归分析的结果相对容易阅读。

```py
Accuracy: 0.43035067573535907
Confusion Matrix:
 [[  385    70   401  1637  1302]
 [   38    13    71   563   513]
 [   55    16   109  1342  1365]
 [   49    18   100  2909  3501]
 [   54    23   208  8021 11200]]
Coefficients:
 [-4.92618413e-02  1.63964377e-03  3.08760598e-03  1.38901936e-02
  1.93889038e-05  4.21370114e-01 -8.75650964e-06  3.59787432e-01
  2.93638614e-02 -1.72296023e-03  2.82130024e-02  9.50451240e-07
  1.32936583e-01  2.23209948e-01  1.31639907e-02 -5.95202359e-03]
```

这个准确度意味着您可以解释 43% 的评论分数的变异性。

精准无误吗？误读或高估模型的准确度可能代价高昂。混淆矩阵是一种表格布局，用于可视化分类算法的性能，通常用于监督学习。在二元预测中，它显示了模型做出的正确和错误预测的数量，并将它们与实际结果进行比较。矩阵分为四个象限，代表真实阳性（TP）、真实阴性（TN）、假阳性（FP；也称为 I 型错误）和假阴性（FN；也称为 II 型错误）。通过评估矩阵，我们可以估计 I 型错误的比率，即模型错误地将阳性实例分类为阴性时发生的情况，以及 II 型错误的比率，即模型错误地将阴性实例分类为阳性时发生的情况。您可以在第五章中找到有关错误类型的更多信息。

在一个多元混淆矩阵中，例如前面提到的矩阵，每一行代表实际类别，而每一列代表预测类别。矩阵主对角线上的元素对应于每个类别的正确分类实例，而矩阵的非对角线元素表示错误分类。为了更好地理解，让我们看看我们对评论的分类，并关注矩阵中的第三列，它代表对得分为三的预测。在 109 个案例中，分类器正确地分配了得分。它还将 71 个实际的三分误分类为二分，401 个误分类为一分，100 个误分类为四分，208 个误分类为五分。

为了使分析更容易，让我们请求生成式 AI 提供将接收到的结果与剩余变量连接的代码。

## 列表 3.7 将预测结果与变量名连接

```py
# Assuming you have already fitted the model and have the `model` variable
# as the instance of the fitted model, e.g.,
model = mord.LogisticAT().fit(X_train, y_train)

# Get the feature names from the analysis_df DataFrame
# Replace 'X' with the actual variable name containing your
# independent variables
feature_names = X.columns

# Get the coefficients from the model
coefficients = model.coef_

# Create a dictionary to link the coefficients to the column names
coefficient_dict = dict(zip(feature_names, coefficients))

# Print the coefficients along with the corresponding column names
for feature, coef in coefficient_dict.items():
    print(f"{feature}: {coef}")
```

前面的代码片段应该按预期工作并返回这个简单的答案：

```py
delivery_time: -0.04926184129232652
price: 0.0016396437688542174
freight_value: 0.003087605979194856
product_name_lenght: 0.013890193611842357
product_description_lenght: 1.938890383573905e-05
product_photos_qty: 0.4213701136980763
product_weight_g: -8.756509636657266e-06
payment_sequential: 0.35978743241417466
payment_installments: 0.02936386138061978
payment_value: -0.001722960231783981
delivery_time_misestimation: 0.02821300240733595
product_volume: 9.504512398147115e-07
payment_type_boleto: 0.1329365830670677
payment_type_credit_card: 0.22320994812520034
payment_type_debit_card: 0.013163990732791755
payment_type_voucher: -0.005952023594428278
```

从有序回归得到的系数的解释与从线性回归得到的结果的解释相似，如果不是那么直接。这些数字代表在保持其他变量不变的情况下，对应预测变量每增加一个单位，结果变量处于更高类别中的对数优势的变化。用更简单的话来说，当我们预测具有特定顺序的事物时，比如“低”、“中”和“高”，系数会告诉我们我们的预测如何随着输入的变化而变化。

例如，假设我们想要根据你睡了多少小时来预测幸福水平。系数将告诉我们，对于每多睡一个小时，成为“更快乐”的可能性（从“低”到“中”或从“中”到“高”）如何变化。所有这些都假设你的环境中没有其他变化。因此，如果这个数字（系数）是正的，这意味着多睡觉会增加成为“更快乐”的可能性。如果是负的，多睡觉会减少可能性。数字越大，影响越强。

从研究的角度来看，如果增加的睡眠是由于从压力大的工作中休假而导致的，那么评估睡眠对幸福的影响并不理想。尽管如此，我们并不总是拥有完美的测试条件。这里描述的方法可能允许你通过提供多个条件和你的情绪记录作为输入来识别睡眠的影响。

我们将在第四章讨论我们的系数如何转化为商业洞察。

## 3.4.4 卖家之间的销售业绩和客户满意度差异

在本节中，我们的重点将放在不同销售群体之间的差异上。我们将尝试回答这样的问题：“来自不同城市或州的卖家在销售业绩和客户满意度方面是否存在显著差异？”

由于我们已进入章节的后期，你可能急于进入下一章，让我们看看这种方法可以成功应用的地方。谁知道呢，也许这里有些内容会对你感兴趣：

- *医疗绩效分析* — 比较不同医院或医疗提供者之间的患者结果（例如，恢复率、满意度评分），以识别质量差异和改进领域。
- *教育成就差距* — 分析学校或地区之间学生表现（例如，考试成绩、毕业率）的差异，以揭示教育不平等并针对干预措施。
- *营销活动有效性* — 评估不同地区或人口统计的营销活动的影响，以确定哪些策略在增强品牌知名度或销售方面最有效。
- *人力资源管理* — 比较不同公司地点或部门的员工满意度和绩效指标，以识别最佳实践和需要关注的领域。
- *金融服务比较* — 分析不同分支机构的贷款批准率或违约率，以评估风险管理有效性和客户服务质量。
- *供应链效率* — 比较不同配送中心的交货时间或库存周转率，以优化物流和供应链管理。
- *产品质量控制* — 分析不同设施制造的产品退货率或客户投诉，以识别质量控制问题。
- *环境政策影响* — 比较实施环境政策前后不同地区的污染水平或保护成果，以评估其有效性。
- *零售连锁店绩效* — 评估不同门店位置的销售额或客户忠诚度评分，以识别表现良好的门店并了解其成功因素。
- *公共政策和社会项目* — 通过比较就业率或健康指标等结果指标，评估不同社区中社会项目（例如，失业援助、公共卫生倡议）的有效性。

在我们深入分析之前，我们需要向您介绍一种称为自助法的方法。
