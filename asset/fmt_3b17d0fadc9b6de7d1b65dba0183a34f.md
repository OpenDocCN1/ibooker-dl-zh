
我们收到了两个案例：`5509` 和 `8307`：

```py
(array([5509, 8307]),)
```

我们还调查了负异常值：

```py
print(np.where(y_pred_cv < -25_000))
```

这里我们得到一个单一案例，`182`：

```py
(array([182]),)
```

在列表 7.15 中，我们定义了一个函数，可以帮助我们检查异常值。对于每个预测特征，这个函数打印出系数以及该案例中系数与特征值的乘积，从而明确每个特征对预测的贡献。

## 列表 7.15 系数检查

```py
def report_case(model, data, feature_names, case_no):
    case_values = data[case_no]                             ①
    coef_values = case_values * model.coef_                 ②
    for feature_name, value, coef_value in zip(
            feature_names, case_values, coef_values):
        print(f"{feature_name:50s}" +
              f"({value:10.2f}) : " +
              f"{coef_value:+0.2f}")                        ③
    print("-" * 80)
    print(" "*66 + f"{np.sum(coef_values):+0.2f}")          ④
```

① 从数据数组中提取指定案例编号的特征值

② 通过将每个特征的值与其模型中的相应系数相乘来计算每个特征的系数值

③ 遍历并打印特征名称、它们的值以及相应的系数值

④ 打印出计算出的系数值的总和

在我们的检查函数准备就绪后，我们可以开始检查案例 `8307`，它代表了一个预测中过于正大的异常值：

```py
report_case(model=lm, 
            data=Xt, 
            feature_names=column_transform.get_feature_names_out(), 
            case_no=8307)
```

以下是对案例 `8307` 的结果，很明显，使预测成为异常值的额外贡献是由于卧室数量（暗示这可能是宿舍）。这个高值将最终预测列表推高：

```py
onehot_encoding__room_type_Entire home/apt         (      1.00) : -8295.89
onehot_encoding__room_type_Hotel room              (      0.00) : -0.00
onehot_encoding__room_type_Private room            (      0.00) : -0.00
onehot_encoding__room_type_Shared room             (      0.00) : -0.00
ordinal_encoding__neighbourhood_more_than_30       (     12.00) : +576.48
ordinal_encoding__type_of_accommodation            (     20.00) : +2377.99
target_encoding__coordinates                       (  29649.71) : +26556.25
numeric__minimum_nights                            (      1.00) : -268.05
numeric__number_of_reviews                         (      0.00) : -0.00
numeric__days_since_last_review                    (     -1.00) : -0.66
numeric__reviews_per_month                         (     -1.00) : -172.50
numeric__calculated_host_listings_count            (     15.00) : +1470.92
numeric__availability_365                          (    354.00) : +16503.11
numeric__score                                     (     -1.00) : +524.08
numeric__number_of_reviews_ltm                     (      0.00) : -0.00
numeric__number_of_reviews_ltm_ratio               (      0.00) : +0.00
numeric__number_of_bedrooms                        (     18.00) : +64407.67
numeric__number_of_beds                            (     18.00) : +31283.70
numeric__number_of_baths                           (      2.00) : -1787.41
numeric__imperial_palace_distance                  (   2279.80) : -859.33
numeric__nearest_convenience_store                 (    149.84) : +549.07
numeric__nearest_train_station                     (    545.08) : -1043.20
numeric__nearest_airport                           (    389.85) : -137.44
numeric__nearest_bus_station                       (    322.04) : -266.55
numeric__nearest_subway                            (    221.93) : -17.29
binary__is_new                                     (      0.00) : -0.00
binary__is_studio                                  (      0.00) : +0.00
binary__has_shared_bath                            (      0.00) : -0.00
binary__has_half_bath                              (      0.00) : -0.00
----------------------------------------------------------------------------
                                                                  +131400.95
```

类似的问题是由于每个特征都是线性建模的。因此，特征的预测贡献是无界的，没有最大值或最小值，而是根据特征值增加或减少。通常，将非线性性和交互作用引入模型中的非线性性和交互作用可以缓解这些问题。现在让我们检查唯一的负异常值：

```py
report_case(model=lm, 
            data=Xt, 
            feature_names=column_transform.get_feature_names_out(), 
            case_no=182)
```

这里问题表现为夜数的最小值，这又太高，拉低了估计值。事实上，一些列表作为季节性住宿，通常为工人或学生，而不仅仅是短期住宿。模型确实过于简单，无法捕捉到这样的细微差别，而且，引入非线性性和交互作用可能会有所帮助：

```py
onehot_encoding__room_type_Entire home/apt         (      0.00) : -0.00
onehot_encoding__room_type_Hotel room              (      0.00) : -0.00
onehot_encoding__room_type_Private room            (      1.00) : -11573.69
onehot_encoding__room_type_Shared room             (      0.00) : -0.00
ordinal_encoding__neighbourhood_more_than_30       (      6.00) : +288.24
ordinal_encoding__type_of_accommodation            (     14.00) : +1664.59
target_encoding__coordinates                       (  27178.66) : +24343.02
numeric__minimum_nights                            (    120.00) : -32166.38
numeric__number_of_reviews                         (    122.00) : -1241.88
numeric__days_since_last_review                    (    132.00) : +87.20
numeric__reviews_per_month                         (      1.33) : +229.43
numeric__calculated_host_listings_count            (      4.00) : +392.25
numeric__availability_365                          (      0.00) : +0.00
numeric__score                                     (      4.98) : -2609.92
numeric__number_of_reviews_ltm                     (      4.00) : -18.33
numeric__number_of_reviews_ltm_ratio               (      0.03) : +54.17
numeric__number_of_bedrooms                        (      1.00) : +3578.20
numeric__number_of_beds                            (      0.00) : +0.00
numeric__number_of_baths                           (      0.00) : -0.00
numeric__imperial_palace_distance                  (  32506.70) : -12252.79
numeric__nearest_convenience_store                 (   5020.81) : +18397.51
numeric__nearest_train_station                     (   5689.32) : -10888.48
numeric__nearest_airport                           (  11438.81) : -4032.76
numeric__nearest_bus_station                       (   4999.17) : -4137.76
numeric__nearest_subway                            (  16976.69) : -1322.52
binary__is_new                                     (      0.00) : -0.00
binary__is_studio                                  (      0.00) : +0.00
binary__has_shared_bath                            (      0.00) : -0.00
binary__has_half_bath                              (      0.00) : -0.00
----------------------------------------------------------------------------
                                                                  -31209.90
```

总之，我们的基线模型已经向我们表明，成功解决东京 Airbnb 数据集提出的问题需要一个更好的拟合模式，该模式可以处理正预测（它们应该是必要的正数）并且可以表示住宿的特定特征（如大量卧室表示旅舍，高最低住宿夜数表示季节性租户）之间的非线性关系和交互作用。在下一小节中，我们将通过使用 XGBoost 模型一次性解决所有这些问题，该模型应该能够以更复杂和智能的方式处理这些数据。

## 7.2.4 构建第一个尝试性模型

首先，我们选择了一个 XGBoost 回归器，试图结合我们从之前的 EDA 和基线模型检查中获得的一些见解。我们决定使用 `gamma` 目标函数，这在回归问题中常用，用于建模正的连续变量，这些变量是正偏斜的。当目标变量始终为正，并且包含许多小值和一些较大值时，`gamma` 特别有用，因为它很好地处理了这种分布特征。

此外，由于我们的基线模型已经显示出欠拟合的迹象，并且没有正确处理交互作用或线性关系，我们决定对于组成增强集的决策树的最大深度不超过六，从而允许有足够数量的分割来处理大多数常见的数据特征。

在以下列表中，与之前的列表训练线性回归基线类似，我们训练了一个 XGBoost 回归器，并测试了其折叠交叉验证的预测结果。

## 列表 7.16 第一个 XGBoost 模型
