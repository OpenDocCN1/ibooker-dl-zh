
# 第十二章：混合梯度提升和深度学习

本章涵盖

+ 对第七章中端到端梯度提升示例的回顾
+ 将第七章中梯度提升示例的结果与同一问题的深度学习解决方案进行比较
+ 将梯度提升模型与深度学习模型集成后的结果

在第七章中，我们对使用梯度提升的端到端示例进行了深入探索。我们探索了东京 Airbnb 列表的数据集，我们为定价回归任务创建了合适的特征，然后我们创建了一个基于此数据集的基线模型来预测价格。最后，应用我们在书中学到的方法，我们优化了基于此数据集训练的 `XGBoost` 模型，并检查了一些解释模型行为的方法。

在本章中，我们评估使用深度学习是否会导致同一问题上的不同结果和性能，确定哪种方法最有效，并发现如何使用和整合每种方法的优点和缺点。为此，我们首先回顾第七章中关于 Airbnb 东京问题的梯度提升方法。接下来，我们回顾一些我们可以采取的方法来将深度学习应用于同一问题，并分享我们选择的针对该问题的深度学习方法。最后，我们比较这两种解决方案，并确定在决定是否使用梯度提升或深度学习来解决表格数据问题时，我们可以从这种比较中学到什么，无论是回归问题还是分类问题。除了根据核心性能（例如每个解决方案的预测能力和推理时间）比较解决方案外，我们还探讨了两种解决方案在更多面向业务的指标（如维护成本、业务利益相关者的清晰度以及部署后的稳定性）方面的比较。

本章将本书中你看到的一些主题汇总在一起，结合我们在第七章中学到的 `XGBoost` 知识，以及我们在第一章和第八章中学到的如何使用深度学习处理表格数据的方法。

在第一章中，我们回顾了一些比较经典机器学习在表格数据应用与深度学习应用的论文。在本章中，我们将看到我们比较 `XGBoost` 与深度学习的结果与我们在第一章中引用的一篇论文的观察结果一致。在第八章中，我们评估了多种处理表格数据的深度学习方法。现在，我们将使用在那里学到的知识来帮助我们选择一个与 `XGBoost` 性能具有最佳竞争力的深度学习方法。

通过结合本书中探讨的两个主要主题（经典机器学习方法和对解决表格数据问题的深度学习方法），本章总结了本书到目前为止所涵盖的内容，并为将机器学习应用于表格数据的应用提供了指导。本章中展示的代码可在[`mng.bz/vKPp`](https://mng.bz/vKPp)找到。

## 12.1 第七章中梯度提升解决方案的回顾

在第七章中，我们为东京的 Airbnb 列表创建了一个数据集，分析了该数据集的关键特征，并创建了一个 `XGBoost` 模型来预测列表的价格。从 Inside Airbnb Network 网站([`insideairbnb.com/`](http://insideairbnb.com/))开始，我们下载了以下与东京市相关的文件：

+ `listings.csv`，其中包含东京 Airbnb 住宿的摘要列表和其他信息
+ `calendar.csv.gz`，一个包含 `calendar.csv` 的压缩文件，该数据集包含每个列表在给定年份的占用和价格信息

请记住，`listings.csv` 数据集包含以下列：

+ `id`—这是 Airbnb 上每个列表的唯一标识符。它是一个 `int64` 数据类型，意味着它是一个数值 ID 表示。在其他表中，它可以被称为 `listing_id`。
+ `name`—Airbnb 列表的描述。它属于 `object` 数据类型，通常表示字符串或文本。
+ `host_id`—这是 Airbnb 上每个房东的唯一标识符。它是一个 `int64` 数据类型。
+ `host_name`—拥有列表的房东的姓名。它属于 `object` 数据类型。
+ `neighbourhood_group`—此字段表示该社区所属的更广泛的区域或地区。它存储为 `float64` 数据类型，但需要注意的是，使用浮点数据类型来表示组或类别是不常见的。在这种情况下，浮点值的存在表明该字段的全部数据都是缺失值。
+ `neighbourhood`—列表所在的具体社区。它属于 `object` 数据类型。
+ `latitude`—列表位置的纬度坐标。它属于 `float64` 数据类型。
+ `longitude`—列表位置的经度坐标。它属于 `float64` 数据类型。
+ `room_type`—列表中提供的房间或住宿类型（例如，整个房屋/公寓、私人房间、共享房间）。它属于 `object` 数据类型。
+ `price`—租赁列表每晚的价格。它属于 `int64` 数据类型，表示一个整数值。
+ `minimum_nights`—预订列表所需的最少夜晚数。它属于 `int64` 数据类型。
+ `number_of_reviews`—列表收到的总评论数。它属于 `int64` 数据类型。
+ `last_review`—列表收到的最后一条评论的日期。数据类型为 `object`，可能代表日期和时间信息，但可能需要进一步解析才能有效使用。
+ `reviews_per_month`—列表每月的平均评论数量。数据类型为 `float64`。
+ `calculated_host_listings_count`—房东在 Airbnb 上的列表总数。数据类型为 `int64`。
+ `availability_365`—列表一年中可供预订的天数（365 天中的天数）。数据类型为 `int64`。
+ `number_of_reviews_ltm`—过去 12 个月内收到的评论数量。数据类型为 `int64`。
+ `license`—列表的许可证号码或相关信息。数据类型为 `object`，通常表示字符串或文本。

第七章中我们创建的模型的目标是预测新列表的价格。这实际上比我们在第八章中解决的 Airbnb 纽约问题对深度学习模型更具挑战性，正如表 12.1 中的比较所示。

表 12.1 Airbnb 纽约问题与 Airbnb 东京问题比较

|  | Airbnb 纽约 | Airbnb 东京 |
| --- | --- | --- |
| 数据集行数 | 48,000 | 10,000 |
| 数据集列数 | 18 | 31 |
| 目标 | 分类：预测价格是否超过或低于中位数价格 | 回归：预测价格 |

实际上，东京 Airbnb 数据集的记录数少于纽约 Airbnb 数据集的 25%，并且列数是纽约 Airbnb 数据集的两倍以上。数据点较少时，你可能需要更多地依赖领域专业知识（因此特征工程的作用）。列数较多意味着在训练过程中过拟合的风险更高，并且无论如何，你必须处理特征与目标变量本身之间更复杂的关系。

通常来说，经典机器学习方法在小数据集上可以获得比深度学习更好的结果。数据增强等技术可以减轻深度学习的这一缺点，但深度学习方法在行数少于数万的数据集上会面临挑战。关于为什么深度学习需要更多数据的研究尚不完整，但深度学习架构中的大量参数以及模型泛化至少需要一定数量的数据被认为是深度学习在处理小数据集问题时遇到困难的原因之一[例如，参见 Thompson 等人撰写的“深度学习的计算限制”（[`arxiv.org/pdf/2007.05558.pdf`](https://arxiv.org/pdf/2007.05558.pdf)），该文还论证了计算效率对于方法进步的必要性]。

东京 Airbnb 数据集中行数较少，这无疑给深度学习解决方案的成功实施带来了挑战。此外，数据集中列数较多，需要更广泛的数据预处理和特征工程，以确保生成的特征对建模目的相关且可用。由于我们遵循了与第七章相同的数据准备步骤，我们不应该担心需要处理更多列的特定应用，但值得记住，对于其他数据集来说，更多的列意味着需要制定有效的策略来处理缺失值、处理多重共线性（如我们在第二章中讨论的）以及选择最有信息量的特征。除了需要更多的数据准备外，更多的列还可能导致冗余或噪声特征，这可能会降低深度学习解决方案在该数据集上的有效性，尽管深度学习模型由于其能够捕捉特征与目标变量之间复杂关系的能力，通常对数据中的噪声更具有鲁棒性。

此外，我们用东京 Airbnb 数据集解决的问题是回归（预测列表价格），而不是二分类（预测给定列表的价格是否高于或低于中位数价格），这是我们尝试用纽约 Airbnb 数据集解决的问题，这具有不同的商业影响。如果目标是提供能够带来商业效益的解决方案，我们的解决方案应该尽可能正确地预测。在二分类问题中，解决方案是离散的（一个类别或另一个类别），从商业角度来看，解决方案做出正确预测的可能性比回归问题中看起来要高，在回归问题中，商业期望模型预测的价格与实际价格非常接近，但输出是连续值，可能显著不同于预期值。简而言之，从满足商业需求的角度来看，二分类问题看起来比回归问题更容易——因为它有一个明确的正确性阈值。

在第七章中，在完成对 Airbnb 东京数据集的一系列转换后，我们得到了以下一组数据集特征：

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

所有这些特征都是数值型的，我们也妥善处理了任何缺失值，以便能够使用线性模型基线。事实上，我们首先创建了一个线性回归模型作为基线，并为我们提供了一个衡量标准，可以用来比较 `XGBoost` 模型的进一步改进。

准备你的数据以进行线性回归或逻辑回归（取决于是否是回归或分类问题）会自动使你的模型准备好由神经网络处理。然而，虽然这很方便，但它可能会错过一些仅适用于神经网络的特定准备。例如，在线性模型中，分类特征通常使用 one-hot-encoding 处理，而在神经网络中，你可以在训练过程中使用编码层直接将分类值转换为数值。

在获得线性回归的基线结果后，我们对 `XGBoost` 模型进行了一系列优化。我们使用这个优化的 `XGBoost` 代码作为解决东京 Airbnb 问题的梯度提升解决方案的基础，我们将在本章中将它用于与深度学习的比较。列表 12.1 显示了我们将用于与深度学习解决方案比较的 `XGBoost` 代码。这个版本的 `XGBoost` 代码与第七章中使用的最终 `XGBoost` 代码非常接近。超参数与第七章中使用的笔记本的最佳超参数相匹配；[`mng.bz/4a6R`](https://mng.bz/4a6R))。在本章使用的代码版本中，预测结果保存在 `xgb_oof_preds` 数组中，以便进一步处理或与我们将从深度学习模型获得的预测一起使用。

列表 12.1 训练最终 `XGBoost` 模型的代码

```py
xgb_params =  {'booster': 'gbtree',                          ①
               'objective': 'reg:tweedie', 
               'n_estimators': 932, 
               'learning_rate': 0.08588055025922144, 
               'subsample': 0.9566295202123205, 
               'colsample_bytree': 0.6730567082779646, 
               'max_depth': 7, 
               'min_child_weight': 6, 
               'reg_lambda': 6.643211493348415e-06, 
               'reg_alpha': 7.024597970671363e-05, 
               'tweedie_variance_power': 1.6727891016980427}

from sklearn.metrics import r2_score  
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error              ②
from XGBoost import XGBRegressor
import numpy as np

xgb = XGBRegressor(**xgb_params)                             ③

cv_splits = cv.split(X, y=neighbourhood_more_than_30)        ④

r2_scores = []
rmse_scores = []
mae_scores = []
xgb_oof_preds = np.zeros(len(X))

for train_index, test_index in cv_splits:                    ⑤
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)                             ⑥
    xgb_oof_preds[test_index] = y_pred

    r2_scores.append(r2_score(y_test, y_pred))               ⑦
    rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae_scores.append(mean_absolute_error(y_test, y_pred))

print(f"Mean cv R-squared: {np.mean(r2_scores):.3f}")        ⑧
print(f"Mean cv RMSE: {np.mean(rmse_scores):.3f}")
Print(f"Mean cv MAE: {np.mean(mae_scores):.3f}")
```

① 设置超参数
② 导入所需的库
③ 使用指定的超参数设置 `XGBoost` 回归器
④ 基于 `neighbourhood_more_than_30` 特征定义交叉验证分割
⑤ 基于 `neighbourhood_more_than_30` 特征生成交叉验证分割
⑥ 执行交叉验证预测
⑦ 计算 R-squared、均方根误差和平均绝对误差评估指标以评估模型性能
⑧ 打印 R-squared、均方根误差和平均绝对误差的均值

对 `XGBoost` 解决方案进行的优化产生了显著优于线性回归基线的结果，如表 12.2 所示。

表 12.2 第七章结果总结

| 指标 | 线性回归基线 | 优化 XGBoost |
| --- | --- | --- |
| R-squared | 0.320 | 0.729 |
| 均方根误差 | 17197.323 | 10853.661 |
| 平均绝对误差 | 12568.371 | 6611.609 |

作为进一步的检查，你可以验证在第七章中跟踪的所有三个指标——即 R-squared、均方根误差（RMSE）和平均绝对误差（MAE）——优化的 `XGBoost` 模型在线性回归基线上的确总是有所改进。现在，以 `XGBoost` 模型作为我们的参考，我们将在本章的剩余部分探索使用相同的深度学习模型可以取得什么样的结果。

## 12.2 选择深度学习解决方案

在第八章中，我们回顾了一系列用于处理表格数据的不同深度学习框架，包括 Keras、fastai 以及其他专门为表格数据设计的库，如 TabNet。如果我们现在想要比较第七章中东京 Airbnb 问题的 `XGBoost` 解决方案，我们应该使用哪种深度学习方法？

作为提醒，我们分享了对使用纽约 Airbnb 数据集的不同深度学习方法的比较，该比较再次在表 12.3 中展示。

表 12.3 深度学习选项比较
