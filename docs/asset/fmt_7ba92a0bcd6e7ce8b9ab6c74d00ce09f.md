
# 第六章：高级特征处理方法

本章涵盖

+ 使用更高级的方法处理特征
+ 选择有用的特征以创建更轻便、更易于理解的模型
+ 优化超参数以使您的模型在性能上更加出色
+ 掌握梯度提升决策树的具体特性和选项

我们现在已经讨论了决策树，它们的特性，它们的局限性，以及所有它们的集成模型，无论是基于随机重采样的，如随机森林，还是基于提升的，如梯度提升。由于提升解决方案被认为是表格数据建模的当前最佳状态，我们详细解释了它是如何工作的以及如何优化其预测。特别是，我们介绍了几种可靠的梯度提升实现，`XGBoost`和`LightGBM`，它们正在证明是处理表格数据的科学家所能获得的最佳解决方案。

本章将涉及有关经典机器学习的更一般主题。然而，我们将专注于梯度提升决策树（GBDTs），特别是`XGBoost`。在本章中，我们将讨论更高级的特征处理方法，例如多元缺失值插补、将高基数分类特征转换为简单数值的目标编码，以及根据它们与目标的关系来确定如何转换或细化特征的一般方法。我们将提出几种方法来减少特征数量到基本要素，并根据可用的计算资源和您选择的模型来优化超参数。然后，本章将以仅与GBDTs相关的先进方法和选项部分结束。

## 6.1 特征处理

在处理现实世界的表格数据集时，您可能会遇到各种问题，如果我们不调整技术以应对数据的现实情况，我们之前讨论的所有方法都将产生不理想的结果。在这里，我们将考虑一些这样的问题，例如以最智能的方式处理缺失值，转换具有大量唯一值的分类特征，以及找到在训练模型后重新处理特征以挤出更多性能的方法。这当然不是详尽的列表，但它应该能帮助您练习发现问题和规划适当的方案。

如前一章所述，为了解释和举例，我们再次将依赖于Airbnb纽约市数据集来展示处理表格数据问题中最具挑战性的任务的实用示例。以下列表回顾了我们将再次在本章中使用的数据和一些关键函数和类。

**列表 6.1 回顾 Airbnb 纽约市数据集**

```py
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

data = pd.read_csv("./AB_NYC_2019.csv")
excluding_list = ['price', 'id', 'latitude', 'longitude', 'host_id', 
                  'last_review', 'name', 'host_name']            ①
low_card_categorical = [
    'neighbourhood_group',
    'room_type'
]                                                                ②
high_card_categorical = ['neighbourhood']                        ③
continuous = [
    'minimum_nights',
    'number_of_reviews',
    'reviews_per_month', 
    'calculated_host_listings_count',
    'availability_365'
]                                                                ④
target_mean = (
    (data["price"] > data["price"].mean())
    .astype(int)
)                                                                ⑤
target_median = (
    (data["price"] > data["price"].median())
    .astype(int)
)                                                                ⑥
target_multiclass = pd.qcut(
    data["price"], q=5, labels=False
)                                                                ⑦
target_regression = data["price"]                                ⑧
categorical_onehot_encoding = OneHotEncoder(handle_unknown='ignore')
categorical_ord_encoding = 
OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
numeric_standardization = Pipeline([('StandardScaler', StandardScaler()), 
                                    ('Imputer', 
SimpleImputer(strategy="constant", fill_value=0))])

column_transform = ColumnTransformer(
    [
        ('low_card_categories', 
         categorical_onehot_encoding, 
         low_card_categorical),
        ('high_card_categories', 
         categorical_ord_encoding, 
         high_card_categorical),
        ('numeric', 
         numeric_standardization, 
         continuous)
    ],
    remainder='drop',
    verbose_feature_names_out=True,
    sparse_threshold=0.0)                                        ⑨

lm_column_transform = ColumnTransformer(
    [
        ('low_card_categories', 
         categorical_onehot_encoding, 
         low_card_categorical),
        ('numeric', 
         numeric_standardization, 
         continuous)
    ],
    remainder='drop',
    verbose_feature_names_out=True,
    sparse_threshold=0.0)                                        ⑩
```

① 特征处理中排除的列列表

② 需要一元编码的低基数分类列列表

③ 需要顺序编码的高基数分类列列表

④ 列出连续特征列

⑤ 创建一个二元目标，指示价格是否高于平均值（不平衡的二元目标）

⑥ 创建一个二元目标，指示价格是否高于中位数（平衡的二元目标）

⑦ 通过将价格分箱为五个类别来创建多类目标

⑧ 将回归的目标设置为价格列

⑨ 创建一个列转换器，对不同的特征组应用不同的转换

⑩ 创建一个适合线性模型的列转换器

我们参考前一章中提供的代码解释的所有细节。唯一的补充是一个专门为线性模型设计的列转换器。这个转换器仅通过执行独热编码处理低基数分类特征，而将高基数分类特征排除在外。

### 6.1.1 多变量缺失数据填充

在你的表格数据集中存在缺失数据是一个阻碍问题，因为除了GBDTs解决方案，如`XGBoost`、`LightGBM`和`Scikit-learn`的`HistGradientBoosting`之外，经典的机器学习算法没有对缺失值的原生支持。此外，即使你选择的GBDTs算法可以处理缺失值，正如下一节所解释的，你仍然可能发现直接填充缺失值更有效，因为你可以在事先检查每个特征或特定案例是如何处理的。

在第二章中，我们讨论了简单的填充方法，例如使用平均值或中位数，以及构建缺失指示器的有用性，从而使得算法更容易发现存在的缺失模式。本节将提供更多关于这些技术和多元填充的细节。

首先，除非缺失案例依赖于未观测变量，例如你无法访问的特征，否则缺失数据可以归类为

+ *完全随机缺失*（MCAR）——在这种情况下，数据缺失与观测到的和未观测到的变量无关。缺失发生在数据集的随机位置。

+ *随机缺失*（MAR）——MAR假设观测变量，而不是未观测变量，可以解释缺失。换句话说，缺失的概率完全取决于观测数据。

当缺失案例依赖于缺失数据的未观测值时，你陷入了“缺失非随机”（MNAR）的情况，这需要相当专业的处理，但这本书不是主题。然而，假设你理解某些缺失非随机缺失数据背后的机制，例如当你没有在人口普查中获得关于太富有人（由于隐私）或太穷的人（由于普遍缺乏访问）的信息时。在这种情况下，你可以尝试收集一些暗示他们财富的新特征，以添加到你的数据集中，并回到“随机缺失”（MAR）的情况。

通常，你经常会遇到缺失数据是MCAR或MAR的情况。在这两种情况下，除了使用与MCAR完美配合的期望值进行简单填充外，你还可以通过多元填充更好地重建缺失数据。*多元填充*是一种使用数据集中预测变量之间的相关性来填充缺失值的方法。它涉及构建一系列模型，根据变量之间的关系来估计缺失值。在此方法中，每个模型将具有缺失值的特征视为目标变量（通过仅对其已知值进行建模），并使用剩余的特征作为预测变量。然后，使用得到的模型来确定用哪些值替换目标中的缺失值。你可以设置算法如何循环特征进行填充。你通常使用默认设置，从缺失数据较少的特征开始，逐步过渡到缺失值较多的特征，这是首选且最有效的方法。

为了处理预测变量中的缺失值，首先使用简单均值或其他基本填充方法进行初步填充。然后，通过多次迭代，通过结合填充模型的成果来细化初始估计。这种迭代过程持续进行，直到填充值达到稳定状态，进一步的迭代不会导致显著变化。`Scikit-learn`通过`IterativeImputer`实现多元填充（[`mng.bz/MDZQ`](https://mng.bz/MDZQ)）。受R MICE包（通过链式方程进行多元填充：[`mng.bz/avEj`](https://mng.bz/avEj)）的启发，它允许进行多元填充和多次填充，这是统计学和社会科学中常见的方法，在这种方法中，你得到的不是单个填充值，而是一系列可能的替代值的分布。通过将`sample_posterior`参数设置为True并使用不同的随机种子多次运行`IterativeImputer`，可以实现多次填充。

然而，在数据科学中的表格数据应用中，多元填充是首选的选择，因为它允许基于单一但精确的估计来构建模型。在我们的例子中，我们取Airbnb NYC数据集的连续特征，并随机删除5%的数据，从而模拟MCAR情况。之后，我们运行`SimpleImputer`，用均值和`IterativeImputer`替换缺失值。最后，我们使用平均绝对误差（MAE）比较每种方法重建的特征与原始值。

**列表 6.2 多元填充**

```py
from sklearn.experimental import (
    enable_iterative_imputer
)                                                             ①
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor

Xm = data[continuous].copy()                                  ②
missing_percentage = 0.05
np.random.seed(0)
mask = np.random.rand(*Xm.shape) < missing_percentage         ③
Xm[mask] = np.nan

simple_imputer = SimpleImputer()
Xm_si = simple_imputer.fit_transform(Xm)                      ④

rf = RandomForestRegressor(random_state=0, n_jobs=-1)         ⑤
multivariate_imputer = IterativeImputer(
    estimator=rf,
    max_iter=1,
    tol=0.01
)                                                             ⑥
Xm_mi = multivariate_imputer.fit_transform(Xm)                ⑦

mae = pd.DataFrame(
    {
        "simple": np.mean(
            np.abs(data[continuous] - Xm_si), axis=0
        ),
        "multivariate": np.mean(
            np.abs(data[continuous] - Xm_mi), axis=0
        )
     },
     index = continuous
)                                                             ⑧
print(mae)
```

① 导入`IterativeImputer`，它在`Scikit-learn`中仍然是实验性的，并且正在改进中

② 创建连续特征数据的副本

③ 创建一个掩码以随机标记缺失值

④ 使用具有均值填充策略的`SimpleImputer`实例

⑤ 实例化`RandomForestRegressor`进行迭代填充

⑥ 创建一个具有最大迭代次数`max_iter`和容忍度`tol`作为停止标准的`IterativeImputer`实例

⑦ 使用迭代插补来插补缺失数据

⑧ 计算插补数据和原始数据的平均绝对误差（MAE）

命令`print(mae)`提供的结果是一个表格，它比较了简单插补和多变量插补方法：

```py
                                       Simple        Multivariate
minimum_nights                         0.347355      0.260156
number_of_reviews                      1.327776      0.858506
reviews_per_month                      0.057980      0.036876
calculated_host_listings_count         0.579423      0.368567
availability_365                       6.025748      4.62264
```

比较结果表明，多变量方法，特别是`IterativeImputer`，在单次迭代后始终产生比简单插补方法更低的MAE值。这表明`IterativeImputer`在用更少的错误替换缺失值方面更有效。为了获得更好的估计，你可以将`max_iter`增加到更高的数值，并让算法根据容忍度值（用于检查结果是否稳定的容忍度阈值）来决定是否提前停止。增加`max_iter`将导致更长的插补时间，因为作为一个插补模型，我们正在使用随机森林算法。随机森林通常是处理多变量估计（在R社区中称为*MissForest*的方法）的最有效方式：[`rpubs.com/lmorgan95/MissForest`](https://rpubs.com/lmorgan95/MissForest)。然而，你可以通过简单地替换`IterativeImputer`中的`estimator`来选择基于线性模型或k近邻的更快方法：

+ `BayesianRidge`—简单地使用`BayesianRidge()`

+ `RandomForestRegressor`—对于随机树回归的森林，你可以设置`n_estimators`、`max_depth`和`max_features`来创建更浅的树，从而加速插补过程，例如`RandomForestRegressor(n_estimators=30, max_depth=6, max_samples=0.5)`

+ `Nystroem` + `Ridge`—一个通过组合不同的`Scikit-learn`命令（`make_pipeline(Nystroem(kernel="polynomial", degree=2, random_state=0), Ridge(alpha=1e3))`）来扩展二次多项式核和正则化线性回归的管道

+ `KNeighborsRegressor`—一个k近邻插补方法，你可以决定要考虑的邻居数量，例如`KNeighbors-Regressor(n_neighbors=5)`
