
```py
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

categorical_onehot_encoding = OneHotEncoder(
       handle_unknown='ignore')                       ①
numeric_passthrough = SimpleImputer(
       strategy="constant", fill_value=0)             ②
numeric_standardization = Pipeline([
       ("imputation", SimpleImputer(strategy="constant", fill_value=0)),
       ("standardizing", StandardScaler())
       ])                                             ③
```

① 将分类特征转换为独热编码格式
② 用零替换缺失的数值
③ 管道用零替换缺失的数值并标准化特征

在这个阶段，我们可以根据每个机器学习算法的需求，编写特定的转换管道来处理数据。例如，在这个例子中，我们设置了一个管道，该管道将低类别特征进行独热编码，并将数值特征的缺失值简单地填充为零。这样的管道是通过`ColumnTransformer`函数创建的，这是一个粘合函数，它同时结合了对不同特征集应用的操作。这是一种非常适合大多数机器学习模型的优秀转换策略：

```py
column_transform = ColumnTransformer(
                [('categories', 
                  categorical_onehot_encoding, 
                  low_card_categorical),              ①
                 ('numeric', 
                  numeric_passthrough, 
                  continuous),                        ②
                ],
    remainder='drop',                                 ③
    verbose_feature_names_out=False,                  ④
    sparse_threshold=0.0                              ⑤
)
```

① 管道的第一步：独热编码类别特征
② 管道的第二步：处理数值特征
③ 管道未处理的特征将从结果中删除。
④ 特征的名称保持与原始名称相同。
⑤ 结果始终是一个密集矩阵（即 NumPy 数组）

我们可以立即运行此代码片段并检查该管道如何转换我们的 Airbnb NYC 数据：

```py
X = column_transform.fit_transform(data)
print(type(X), X.dtype, X.shape)
```

结果是输出现在是一个由浮点数组成的 NumPy 数组，其形状增加到 13 列。实际上，由于独热编码，类别特征中的每个值都变成了一个单独的特征：

```py
<class 'numpy.ndarray'> float64 (48895, 13)
```

下一节将探讨表格数据的主要机器学习技术。每个技术都将伴随其列转换类，这些类将被集成到包含模型的管道中。

## 4.3 经典机器学习

为了解释经典机器学习技术对于表格数据的差异，我们首先将介绍算法的核心特性，然后通过一个代码片段展示其工作，在参考的表格问题上，即 Airbnb NYC 数据集上运行。以下是我们将在示例中使用的一些最佳实践，以允许不同方法的可重复性和可比性：

+ 我们定义了一个包含数据转换和建模的管道。
+ 我们设置一个错误度量，例如回归的均方根误差（RMSE）或分类的准确度，并使用相同的交叉验证策略进行测量。
+ 我们报告交叉验证估计误差的平均值和标准差——这对于确定模型在不同数据样本上是否具有恒定的性能至关重要。

在上一节中，我们介绍了 Scikit-learn 提供的不同工具，用于构建集成特征处理和机器学习模型的管道。在本节中，我们将介绍推荐的评估指标以及 Scikit-learn `cross_validate`命令如何进行交叉验证估计。

让我们先回顾一下*评估指标*。我们决定使用 RMSE，这是回归任务中常用的一个度量，以及准确率，这是当类别具有大致相同的样本大小时，平衡二元和多类分类问题的另一个标准度量。在随后的章节中，我们还将使用适合不平衡分类问题的度量，例如 ROC-AUC 和平均精度。

*交叉验证*是数据科学中当您打算估计机器学习模型在除训练数据外，从相同数据分布中抽取的任何数据上的预期性能时的实际标准。需要注意的是，交叉验证基于这样的想法，即您的数据可能会在未来发生变化，但不会发生根本性的变化，来估计您模型未来的性能。为了正确工作，模型期望您未来将使用相同的特征，并且它们将具有相同的唯一值（如果是一个分类特征），具有相似的分布（对于分类和数值特征），最重要的是，特征将与您的目标变量保持相同的关系。

假设数据分布在未来将保持一致这一假设通常并不成立，因为在现实世界中，经济动态、消费者市场以及社会和政治情况变化非常快。在现实世界中，您的模型可能会遇到概念漂移，即特征与目标之间的建模关系不再代表现实。因此，当处理新数据时，您的模型将表现不佳。交叉验证是评估模型在创建时的最佳工具，因为它基于您当时可用的信息，并且因为，如果设计得当，它不会受到您的机器学习模型过度拟合训练数据的能力的影响。即使与未来的性能相比，交叉验证的结果被证明是错误的，其有用性仍然成立，通常是因为潜在的数据分布已经发生变化。此外，如留一法或自助法等替代方法，随着计算成本的提高，提供了更好的估计，而如训练/测试集分割等更直接的方法在估计上则不太可靠。

在其最简单的形式中，*k 折交叉验证*（在 Scikit-learn 中通过`KFold`函数实现：[`mng.bz/VVM0`](https://mng.bz/VVM0)）基于将你的可用训练数据分成 k 个分区，并构建 k 个版本的你的模型，每次由不同的 k-1 个分区提供数据，然后在剩下的一个未使用的分区上进行测试（样本外性能）。k 个分数的平均值和标准差将提供一个估计及其不确定性水平的量化，用作对未来未见数据的预期性能的模型估计。图 4.9 展示了当 k 设置为 5 时的 k 折验证：每一行代表每个折叠中的数据分区。一个折叠的验证部分总是与其他的不同，而训练部分总是不同地组成。

![](img/CH04_F09_Ryan2.png)

图 4.9 五折交叉验证策略中数据在训练和验证之间的分布

设置 k 的正确值取决于你有多少可用训练数据，在它上训练模型需要多少计算成本，你收到的样本是否捕捉到了你想要建模的数据分布的所有可能变化，以及你打算为什么目的获取性能估计。作为一个一般性的经验法则，k 的值如 5 或 10 是最佳选择，其中 k = 10 更适合精确的性能评估，而 k = 5 是一个在模型、特征和超参数评估等活动中的精度和计算成本之间取得良好平衡的好值（因此它将用于我们的示例）。

为了对你的模型进行一般性能估计，你可以通过在`KFold`函数（或其变体，提供样本分层或对时间维度的控制：[`mng.bz/xKne`](https://mng.bz/xKne)）上进行一系列迭代来构建必要的交叉验证迭代，或者依赖`cross_validate`过程（[`mng.bz/AQyK`](https://mng.bz/AQyK)），它会为你处理所有事情并只返回结果。对于测试不同算法的目的，`cross_validate`非常方便，因为，给定适当的参数，它将生成一系列指标：

+ 交叉验证测试分数（样本外性能）
+ 交叉验证训练分数（样本内性能）
+ 适应时间和预测时间（用于评估计算成本）
+ 不同交叉验证折叠上的训练估计器

我们需要做的就是提供一个估计器，它可以是一个具有`fit`方法的 Scikit-learn 对象，包括预测器和目标，交叉验证策略，以及列表中的单个或多个评分函数。这个估计器应以可调用的形式提供，使用`make_scorer`命令创建（[`mng.bz/ZlAO`](https://mng.bz/ZlAO)）。在下一节中，我们将开始看到如何使用这些输入获取交叉验证的性能估计，从经典的机器学习算法如线性回归和逻辑回归开始。

### 4.3.1 线性回归和逻辑回归

在线性回归中，这是一种统计方法，通过拟合线性方程到观测数据来模拟因变量与一个或多个自变量之间的关系。你首先需要将所有特征转换为数值型，并将它们放入一个矩阵中，包括独热编码的类别特征。算法的目标是在列向量（系数）中找到最优的权重值，这样当乘以特征矩阵时，可以得到一个结果向量，最好地近似你的目标（预测）。换句话说，算法试图最小化目标与通过乘以权重向量得到的预测之间的残差平方和。在这个过程中，你可以考虑使用预测基线（所谓的截距或偏差）或对权重值施加约束，使它们只能是正数。

由于线性回归算法只是一个加权求和，你必须注意三个关键方面：

+ 确保没有缺失值，因为除非你将它们填充为某个值，否则它们不能用于乘法或加法运算。
+ 确保你已经处理了异常值，因为它们会影响算法在训练和预测中的工作。
+ 验证特征和目标尽可能线性相关（即，它们有良好的皮尔逊相关系数）：与目标弱相关的特征往往会给模型增加噪声，并且它们往往会使其欠拟合，甚至在数量多时，甚至过拟合。

由于加权特征的求和给出了预测，因此很容易确定对预测输出影响最大的因素以及每个特征如何对其做出贡献。观察每个特征的系数可以让你了解算法的行为。这种理解在你必须向监管机构或利益相关者解释模型如何工作时非常有价值，以及当你想从假设或领域专家知识的角度检查预测是否合理时。

然而，回归模型在幕后如何工作的简单方式也存在隐藏的陷阱。当数据中的两个或多个特征高度相关时，这在统计学上被称为“多重共线性”，回归模型中的解释可能会变得非常复杂，即使这两个特征都有效地对预测做出了贡献。通常，只有一个许多中的显著系数，而其他则取很小的值，好像它们与目标无关。实际上，情况往往相反，理解特征在回归预测中的作用相对容易，这可能导致重要的概念误解。

线性回归算法的另一个巨大优势是，由于它只是些乘法和加法，因此在任何软件平台上实现它都非常容易，甚至可以通过在脚本中手动编码来实现。其他机器学习算法更复杂，因此比线性回归更复杂的算法从头开始实现可能会容易出错和出现 bug。然而，尽管手动编码任何机器学习模型对于交付项目来说是不切实际的，但我们必须指出，手动编码任何机器学习模型可以是一种宝贵的学习经历，让你更深入地理解算法的内部工作原理，并使你更有能力在未来解决和优化类似模型的性能。我们在第五章中展示了某些算法的一些可管理的从头开始实现，以供学习之用。

我们将从一个线性回归模型的例子开始，该模型从头到尾应用于我们的 Airbnb 纽约市数据。这个例子遵循图 4.10 中提出的方案，这个方案我们将为每个经典机器学习算法进行复制，它基于 Scikit-learn 的管道和交叉验证评估函数。

![图片](img/CH04_F10_Ryan2.png)

图 4.10 经典机器学习算法示例组织方案

该方案相当线性。来自逗号分隔值文件的输入首先通过`ColumnTransformer`，这是数据准备部分，它根据列名对数据进行转换、丢弃数据或让它原样通过，然后是一个机器学习模型。这两个部分都被一个由`cross_validate`函数测试的管道所包裹，该函数执行交叉验证并记录计算时间、训练模型和一定数量的折叠上的性能。最后，选择结果来展示模型是如何工作的。此外，我们可以通过管道访问模型系数和权重，以获得更多关于我们测试的算法功能的见解。

应用此架构，我们只需在列表 4.3 中使用纯线性回归模型，因为此算法通常不需要指定任何参数。对于与模型可解释性相关的特殊应用，你可以指定`fit_intercept`为 false 以从模型中移除截距并仅从特征中得出所有预测，或者将正参数设置为 true 以仅获得正系数。

列表 4.3 线性回归

```py
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

column_transform = ColumnTransformer(
    [('categories', categorical_onehot_encoding, low_card_categorical),
     ('numeric', numeric_passthrough, continuous)],
    remainder='drop',
    verbose_feature_names_out=False,
    sparse_threshold=0.0)                                   ①

model = LinearRegression()                                  ②

model_pipeline = Pipeline(
    [('processing', column_transform),
     ('modeling', model)]                                   ③
)

cv = KFold(5, shuffle=True, random_state=0)                 ④
rmse =  make_scorer(mean_squared_error, 
                    squared=False)                          ⑤

cv_scores = cross_validate(estimator=model_pipeline, 
                           X=data[price_window], 
                           y=target_regression[price_window],
                           scoring=rmse,
                           cv=cv, 
                           return_train_score=True,
                           return_estimator=True)           ⑥

mean_cv = np.mean(cv_scores['test_score'])
std_cv = np.std(cv_scores['test_score'])
fit_time = np.mean(cv_scores['fit_time'])
score_time = np.mean(cv_scores['score_time'])
print(f"{mean_cv:0.3f} ({std_cv:0.3f})", 
      f"fit: {fit_time:0.2f}", 
      f"secs pred: {score_time:0.2f} secs")                 ⑦
```

① `ColumnTransformer`，将数据转换为数值特征并填充缺失数据
② 纯线性回归模型
③ 组装`ColumnTransformer`和模型的管道
④ 基于 5 折交叉验证和随机抽样的交叉验证策略
⑤ 从均方误差派生的评估指标函数
⑥ 自动交叉验证过程
⑦ 以评估指标、标准差、拟合度和预测时间为单位报告结果

运行列出的代码将产生以下 RMSE 结果：

```py
33.949 (0.274) fit: 0.06 secs pred: 0.01 secs
```

这是一个很好的结果，在极短的时间内获得（使用标准的 Google Colab 实例或 Kaggle 笔记本），可以作为更复杂尝试的基准。例如，如果你尝试运行列表 4.4 中的代码，你会意识到你可以通过更少但准确准备的特征获得类似的结果。这被称为*特征工程*，做这件事的有趣之处在于你可以获得更好的结果，或者用更少的特征获得相同的结果，这些特征对领域或商业专家来说更有意义。例如，我们在代码列表中通过生成与特定值相关的二进制特征、组合特征和使用对数函数转换它们来创建各种新的特征。

列表 4.4 线性回归的定制数据准备
