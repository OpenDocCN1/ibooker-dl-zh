
```py
categorical = [
    'room_type',
    'neighbourhood_more_than_30', 
    'type_of_accommodation',
    'coordinates'
]
numeric = [
    'minimum_nights',
    'number_of_reviews', 
    'days_since_last_review',
    'reviews_per_month',
    'calculated_host_listings_count', 
    'availability_365',
    'score', 
    'number_of_reviews_ltm', 
    'number_of_reviews_ltm_ratio', 
    'number_of_bedrooms', 
    'number_of_beds',
    'number_of_baths', 
    'imperial_palace_distance', 
    'nearest_convenience_store',
    'nearest_train_station',
    'nearest_airport', 
    'nearest_bus_station',
    'nearest_subway'
]
binary = [
    'is_new',
    'is_studio',
    'has_shared_bath',
    'has_half_bath'
]
```

进一步检查分类特征是必要的，因为我们需要了解是否将它们视为高基数特征。在计算每个分类特征的唯一值数量后，我们可以更好地理解应该怎么做：

```py
for feat in categorical:
    print(f"{feat} has {X[feat].nunique()} unique values")
```

从结果中，我们可以确定可能唯一可以考虑为高基数分类的特征是`coordinates`特征，它有近 300 个唯一值。至于`neighbourhood_more_than_30`和`type_of_accomodation`，我们可以对它们应用顺序编码以进行基于树的建模，而对于线性模型，最好对这些特征应用独热编码（从而产生大约 50 个新的二进制特征）或目标编码：

```py
room_type has 4 unique values
neighbourhood_more_than_30 has 24 unique values
type_of_accommodation has 29 unique values
coordinates has 296 unique values
```

由于我们的示例围绕 XGBoost，并展示它如何处理类似问题，我们决定仅对`room_type`进行独热编码，对`neighbourhood_more_than_30`和`type_of_accomodation`进行顺序编码，对`coordinates`进行目标编码：

```py
onehot_encoding = ['room_type']
ordinal_encoding = ['neighbourhood_more_than_30', 'type_of_accommodation']
target_encoding = ['coordinates']
```

我们选择使用基于树的模型 XGBoost，这也解释了为什么我们保留所有数值特征不变。使用线性模型、统计标准化，以在使用正则化或广义线性模型时获得更好的收敛性，以及特征转换，以更好地拟合非线性，通常是标准做法。

在列表 7.13 中，我们定义了所有必要的特征转换，并将它们集成到 Scikit-learn 的列转换器中，这将作为管道的一部分，该管道还将包含我们选择的机器学习模型。还应注意，我们在定义列转换器时采取步骤来处理测试时可能意外出现的未知类别和缺失值。我们单热编码的策略是忽略新的未知类别。对于有序编码，分配给参数`unknown_value`的值，默认为`np.nan`，将用于编码未知类别。这意味着 XGBoost 模型将使用最频繁的分割来处理此类情况。其他机器学习算法可能会在类似情况下中断，这是 XGBoost 的优势。至于目标编码器，未知类别用目标均值替换。不要忘记安装`category_encoders`包。如果您的系统上不可用，请使用`pip install category_encoders`命令。

**列表 7.13 定义列转换**

```py
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from category_encoders.target_encoder import TargetEncoder

onehot_encoder = OneHotEncoder(handle_unknown='ignore')          ①
ordinal_enconder = OrdinalEncoder(handle_unknown="use_encoded_value",
                                  unknown_value=np.nan)          ②
target_encoder = TargetEncoder(
    cols=target_encoding, 
    handle_unknown="value", 
    smoothing=0.5
)                                                                ③

column_transform = ColumnTransformer(
    [('onehot_encoding', onehot_encoder, onehot_encoding),
     ('ordinal_encoding', ordinal_enconder, ordinal_encoding),
     ('target_encoding', target_encoder, target_encoding),
     ('numeric', 'passthrough', numeric),
     ('binary', 'passthrough', binary)],                         ④
    remainder='drop',                                            ⑤
    verbose_feature_names_out=True,                              ⑥
    sparse_threshold=0.0)                                        ⑦
```

① 创建一个单热编码器对象，具有在编码过程中忽略未知类别的选项

② 创建一个具有处理未知类别和将未知值替换为`np.nan`的有序编码器对象

③ 创建一个目标编码器对象，通过使用平均目标值进行编码并应用平滑参数为 0.5 来处理未知值

④ 创建一个列转换器对象，将指定的编码器应用于相应的列

⑤ 删除转换器中未指定的剩余列

⑥ 保留转换列的详细特征名称

⑦ 确保转换后的数据保持为密集数组

在运行代码列表后，我们可以立即测试转换我们拥有的数据并检查转换后的列名：

```py
Xt = column_transform.fit_transform(X, y)
column_transform.get_feature_names_out()
```

输出显示，现在特征前面有一个前缀，指出它们经历了什么转换。由单热编码创建的二进制特征也跟随着它们所代表的类别：

```py
array(['onehot_encoding__room_type_Entire home/apt',
       'onehot_encoding__room_type_Hotel room',
       'onehot_encoding__room_type_Private room',
       'onehot_encoding__room_type_Shared room',
       'ordinal_encoding__neighbourhood_more_than_30',
       'ordinal_encoding__type_of_accommodation',
       'target_encoding__coordinates', 'numeric__minimum_nights',
       'numeric__number_of_reviews', 'numeric__days_since_last_review',
       'numeric__reviews_per_month',
       'numeric__calculated_host_listings_count',
       'numeric__availability_365', 'numeric__score',
       'numeric__number_of_reviews_ltm',
       'numeric__number_of_reviews_ltm_ratio',
       'numeric__number_of_bedrooms', 'numeric__number_of_beds',
       'numeric__number_of_baths', 'numeric__imperial_palace_distance',
       'numeric__nearest_convenience_store',
       'numeric__nearest_train_station', 'numeric__nearest_airport',
       'numeric__nearest_bus_station', 'numeric__nearest_subway',
       'binary__is_new', 'binary__is_studio', 'binary__has_shared_bath',
       'binary__has_half_bath'], dtype=object)
```

作为最后一步，我们将处理后的特征和目标值存储到一个单独的 CSV 文件中。我们将在第十二章再次使用这些数据，那时我们将测试一个深度学习解决方案，并将其性能与本章训练的 XGBoost 模型进行比较。

```py
data = pd.DataFrame(
    Xt, 
    columns=column_transform.get_feature_names_out(),
    index=y.index
)
data = data.assign(target=y).reset_index()
data.to_csv("airbnb_tokyo.csv", index=False)
```

现在我们有了管道的数据处理部分，我们可以继续定义基线模型，然后最终定义 XGBoost 回归器。

### 7.2.3 构建基线模型

在机器学习中拥有基线模型有几个重要原因：

+ *比较性能*——基线模型作为基准，用于比较更复杂模型的性能，帮助你理解更多的复杂性是否真的增加了价值。

+ *检测过拟合*——通过比较你的高级模型与基线在未见数据上的性能，你可以确定高级模型是否过拟合，因为基线模型的表现会好得多。

+ *理解问题*——创建一个简单的基线模型，尤其是如果它是一个线性模型，迫使你更好地理解数据和问题。

+ *调试和验证*——基线模型可以帮助你验证你的数据预处理流程是否正确，因为变量的影响不会被模型的复杂性所掩盖。

+ *提供最小可行模型*——基线模型为当前问题提供了一个最小可行解决方案。

由于所有这些原因，我们不会立即跳入使用梯度提升模型训练我们的模型，因为我们期望它在问题上表现良好，但我们退一步，测试一个简单的线性模型。此外，在此期间，我们将尝试从一种我们可以轻松评估和比较的模型类型中获得预测。我们不仅通过交叉验证评估指标，还会采用交叉验证预测。这种方法通过在交叉验证的验证折叠中进行预测，为所有训练案例提供无偏预测。

在交叉验证过程中，评估指标会为每个折叠单独计算。这些指标代表模型在每个折叠上的性能。从交叉验证中报告的最终评估指标通常是单个折叠指标的均值（平均值或中位数）。这个聚合指标提供了模型在未见数据上的泛化性能估计。然而，如果我们使用交叉验证预测，我们则专注于模型在现有数据上的表现能力。实际上，交叉验证预测的主要用途是分析模型在不同数据部分上的预测。使用这些预测有助于我们了解模型在不同数据子集上的表现如何，并确定模型是否过拟合或欠拟合，因为我们可以将预测与预期的目标值进行比较。

**列表 7.14 带诊断图的线性回归基线模型**

```py
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

lm = LinearRegression(fit_intercept=False)                  ①
lm.fit(Xt, y)                                               ②

cv_splits = cv.split(
    X,
    y = X["neighbourhood_more_than_30"]
)                                                           ③
y_pred_cv = cross_val_predict(
    lm, Xt, y, cv=cv_splits
)                                                           ④
prediction_range = y_pred_cv.min()} - {y_pred_cv.max()
print(f"prediction range: {prediction_range}")              ⑤

r2 = r2_score(y, y_pred_cv)
rmse = np.sqrt(mean_squared_error(y, y_pred_cv))
mae = mean_absolute_error(y, y_pred_cv)                     ⑥

print(f'R-squared: {r2:.3f}')
print(f'RMSE: {rmse:.3f}')
print(f'MAE: {mae:.3f}')

plt.scatter(y, y_pred_cv)                                   ⑦
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Fit')
plt.axhline(
   0, color='orange', linestyle='--', label='Zero Line'
)                                                           ⑧
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression - Fitted Results')
plt.legend()
plt.show()
```

① 初始化一个没有截距的线性回归模型

② 将线性回归模型拟合到转换后的训练数据

③ 根据超过 30 个计数的邻域创建分层交叉验证分割

④ 执行交叉验证预测

⑤ 打印交叉验证预测的范围

⑥ 计算 R 平方、均方根误差和平均绝对误差评估指标以评估模型性能

⑦ 创建实际值与预测值之间的散点图

⑧ 在图表上绘制一条虚线橙色零线作为理想拟合的参考

运行代码后，我们获得了评估结果，我们可以立即注意到一些预测是负数。由于线性回归模型的预测没有界限，平均绝对误差（MAE）相当高（超过 12,000 日元），而 R 平方，一个典型的拟合度度量，衡量模型捕获目标方差的比例，仅为 0.32：

```py
prediction range: -34929.50241836217 - 136479.60736257263
R-squared: 0.320
RMSE: 17197.323
MAE: 12568.371
```

显然，模型的拟合并不特别令人印象深刻，我们可以在图 7.11 中得到证实，其中我们在 y 轴上表示交叉验证预测的散点图，与 x 轴上的预期目标值相对应。除了目标分布开始处的一些负预测外，我们还可以注意到预测如何偏离理想的拟合虚线，显示出平坦的趋势，这是欠拟合的明显迹象，以及如何有一些异常的预测。

![](img/CH07_F11_Ryan2.png)

**图 7.11 基线线性回归拟合结果与其理想值的对比图**

在检查结果的第一步中，我们要求预测值低于或等于零的百分比，这是一个不可行的预测，因为列表应该是正数：

```py
(y_pred_cv <= 0).sum() / len(y_pred_cv)
```

结果是极低的百分比，大约 0.5%：

```py
0.005178767055074196
```

理想情况下，我们的预测应该大于零，在一个线性模型中，这可以通过目标转换来实现——例如，对数转换。然而，基线模型的作用并不是成为一个完美的模型，而只是一个用来突出数据中的挑战，并为更复杂的模型提供有益比较的模型。

现在我们继续定位正异常值的行：
