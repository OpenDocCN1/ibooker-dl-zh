
# 梯度提升回归器（GBRT）示例

首先，我们导入所需的库并生成一些带有噪声的二次函数数据：

```py
import numpy as np
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)  # y = 3x² + Gaussian noise

tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)
```

接下来，我们将在第一个预测器产生的残差错误上训练第二个`DecisionTreeRegressor`：

```py
y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=43)
tree_reg2.fit(X, y2)
```

然后我们将在第二个预测器产生的残差错误上训练第三个回归器：

```py
y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=44)
tree_reg3.fit(X, y3)
```

现在我们有一个包含三棵树的集成。它可以通过简单地将所有树的预测相加来对新实例进行预测：

```py
>>> X_new = np.array([[-0.4], [0.], [0.5]])
>>> sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
array([0.49484029, 0.04021166, 0.75026781])
```

图 7-9 在左列中表示这三棵树的预测，右列中表示集成的预测。在第一行中，集成只有一棵树，因此其预测与第一棵树的预测完全相同。在第二行中，新树是在第一棵树的残差错误上训练的。您可以看到集成的预测等于前两棵树的预测之和。类似地，在第三行中，另一棵树是在第二棵树的残差错误上训练的。您可以看到随着树被添加到集成中，集成的预测逐渐变得更好。

您可以使用 Scikit-Learn 的`GradientBoostingRegressor`类更轻松地训练 GBRT 集合（还有一个用于分类的`GradientBoostingClassifier`类）。就像`RandomForestRegressor`类一样，它有用于控制决策树增长的超参数（例如`max_depth`，`min_samples_leaf`），以及用于控制集合训练的超参数，比如树的数量（`n_estimators`）。以下代码创建了与前一个相同的集合：

```py
from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3,
                                 learning_rate=1.0, random_state=42)
gbrt.fit(X, y)
```

![mls3 0709](img/mls3_0709.png)

###### 图 7-9。在这个梯度提升的描述中，第一个预测器（左上角）被正常训练，然后每个连续的预测器（左中和左下）都是在前一个预测器的残差上进行训练；右列显示了结果集合的预测

`learning_rate`超参数缩放每棵树的贡献。如果将其设置为一个较低的值，比如`0.05`，则需要更多的树来拟合训练集，但预测通常会更好地泛化。这是一种称为*缩减*的正则化技术。图 7-10 显示了使用不同超参数训练的两个 GBRT 集合：左侧的集合没有足够的树来拟合训练集，而右侧的集合有大约适量的树。如果添加更多的树，GBRT 将开始过拟合训练集。

![mls3 0710](img/mls3_0710.png)

###### 图 7-10。GBRT 集合，预测器不足（左）和刚好足够（右）

要找到最佳数目的树，可以像往常一样使用`GridSearchCV`或`RandomizedSearchCV`进行交叉验证，但也有一种更简单的方法：如果将`n_iter_no_change`超参数设置为一个整数值，比如 10，那么`GradientBoostingRegressor`将在训练过程中自动停止添加更多的树，如果看到最后的 10 棵树没有帮助。这只是早停（在第四章中介绍），但需要一点耐心：它容忍在停止之前几次迭代没有进展。让我们使用早停来训练集合：

```py
gbrt_best = GradientBoostingRegressor(
    max_depth=2, learning_rate=0.05, n_estimators=500,
    n_iter_no_change=10, random_state=42)
gbrt_best.fit(X, y)
```

如果将`n_iter_no_change`设置得太低，训练可能会过早停止，模型会欠拟合。但如果设置得太高，它将过拟合。我们还设置了一个相当小的学习率和一个较高数量的估计器，但由于早停，训练集合中实际的估计器数量要低得多：

```py
>>> gbrt_best.n_estimators_
92
```

当设置了`n_iter_no_change`时，`fit()`方法会自动将训练集分成一个较小的训练集和一个验证集：这使得它可以在每次添加新树时评估模型的性能。验证集的大小由`validation_fraction`超参数控制，默认为 10%。`tol`超参数确定了仍然被视为微不足道的最大性能改进。默认值为 0.0001。

`GradientBoostingRegressor`类还支持一个`subsample`超参数，指定用于训练每棵树的训练实例的分数。例如，如果`subsample=0.25`，则每棵树都是在选择的 25%训练实例上随机训练的。现在你可能已经猜到了，这种技术以更低的方差换取更高的偏差。它还显著加快了训练速度。这被称为*随机梯度提升*。

## 基于直方图的梯度提升

Scikit-Learn 还提供了另一种针对大型数据集进行优化的 GBRT 实现：*基于直方图的梯度提升*（HGB）。它通过对输入特征进行分箱，用整数替换它们来工作。箱数由`max_bins`超参数控制，默认为 255，不能设置得比这更高。分箱可以大大减少训练算法需要评估的可能阈值数量。此外，使用整数可以使用更快速和更节省内存的数据结构。构建箱的方式消除了在训练每棵树时对特征进行排序的需要。

因此，这种实现的计算复杂度为*O*(*b*×*m*)，而不是*O*(*n*×*m*×log(*m*))，其中*b*是箱数，*m*是训练实例数，*n*是特征数。实际上，这意味着 HGB 在大型数据集上的训练速度可以比常规 GBRT 快数百倍。然而，分箱会导致精度损失，这起到了正则化的作用：根据数据集的不同，这可能有助于减少过拟合，也可能导致欠拟合。

Scikit-Learn 提供了两个 HGB 类：`HistGradientBoostingRegressor`和`HistGradientBoostingClassifier`。它们类似于`GradientBoostingRegressor`和`GradientBoostingClassifier`，但有一些显著的区别：

+   如果实例数大于 10,000，则自动启用提前停止。您可以通过将`early_stopping`超参数设置为`True`或`False`来始终启用或关闭提前停止。

+   不支持子采样。

+   `n_estimators`被重命名为`max_iter`。

+   唯一可以调整的决策树超参数是`max_leaf_nodes`、`min_samples_leaf`和`max_depth`。

HGB 类还有两个很好的特性：它们支持分类特征和缺失值。这在预处理方面简化了很多。但是，分类特征必须表示为从 0 到小于`max_bins`的整数。您可以使用`OrdinalEncoder`来实现。例如，以下是如何为第二章介绍的加利福尼亚住房数据集构建和训练完整管道的方法：

```py
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

hgb_reg = make_pipeline(
    make_column_transformer((OrdinalEncoder(), ["ocean_proximity"]),
                            remainder="passthrough"),
    HistGradientBoostingRegressor(categorical_features=[0], random_state=42)
)
hgb_reg.fit(housing, housing_labels)
```

整个管道和导入一样简短！不需要填充器、缩放器或独热编码器，非常方便。请注意，`categorical_features`必须设置为分类列的索引（或布尔数组）。在没有进行任何超参数调整的情况下，该模型的 RMSE 约为 47,600，这还算不错。

###### 提示

Python ML 生态系统中还提供了几种经过优化的梯度提升实现：特别是[XGBoost](https://github.com/dmlc/xgboost)、[CatBoost](https://catboost.ai)和[LightGBM](https://lightgbm.readthedocs.io)。这些库已经存在了几年。它们都专门用于梯度提升，它们的 API 与 Scikit-Learn 的非常相似，并提供许多附加功能，包括 GPU 加速；您一定要去了解一下！此外，[TensorFlow 随机森林库](https://tensorflow.org/decision_forests)提供了各种随机森林算法的优化实现，包括普通随机森林、极端树、GBRT 等等。

# 堆叠

本章我们将讨论的最后一种集成方法称为*堆叠*（缩写为[*堆叠泛化*](https://homl.info/29)）。它基于一个简单的想法：不要使用微不足道的函数（如硬投票）来聚合集成中所有预测器的预测，为什么不训练一个模型来执行这种聚合呢？图 7-11 展示了这样一个集成在新实例上执行回归任务。底部的三个预测器中的每一个都预测不同的值（3.1、2.7 和 2.9），然后最终的预测器（称为*混合器*或*元学习器*）将这些预测作为输入，做出最终的预测（3.0）。

![mls3 0711](img/mls3_0711.png)

###### 图 7-11。使用混合预测器聚合预测

要训练混合器，首先需要构建混合训练集。您可以对集成中的每个预测器使用`cross_val_predict()`，以获取原始训练集中每个实例的样本外预测（图 7-12），并将这些用作输入特征来训练混合器；目标可以简单地从原始训练集中复制。请注意，无论原始训练集中的特征数量如何（在本例中只有一个），混合训练集将包含每个预测器的一个输入特征（在本例中为三个）。一旦混合器训练完成，基本预测器将最后一次在完整的原始训练集上重新训练。

![mls3 0712](img/mls3_0712.png)

###### 图 7-12。在堆叠集成中训练混合器

实际上可以通过这种方式训练几个不同的混合器（例如，一个使用线性回归，另一个使用随机森林回归），以获得一个完整的混合器层，并在其上再添加另一个混合器以生成最终预测，如图 7-13 所示。通过这样做，您可能会挤出更多的性能，但这将在训练时间和系统复杂性方面付出代价。

![mls3 0713](img/mls3_0713.png)

###### 图 7-13。多层堆叠集成中的预测

Scikit-Learn 提供了两个用于堆叠集成的类：`StackingClassifier`和`StackingRegressor`。例如，我们可以用`StackingClassifier`替换本章开始时在 moons 数据集上使用的`VotingClassifier`：

```py
from sklearn.ensemble import StackingClassifier

stacking_clf = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ],
    final_estimator=RandomForestClassifier(random_state=43),
    cv=5  # number of cross-validation folds
)
stacking_clf.fit(X_train, y_train)
```

对于每个预测器，堆叠分类器将调用`predict_proba()`（如果可用）；如果不可用，它将退而求其次调用`decision_function()`，或者作为最后手段调用`predict()`。如果您没有提供最终的估计器，`StackingClassifier`将使用`LogisticRegression`，而`StackingRegressor`将使用`RidgeCV`。

如果您在测试集上评估这个堆叠模型，您会发现 92.8%的准确率，比使用软投票的投票分类器稍好，后者获得了 92%。

总之，集成方法是多才多艺、强大且相当简单易用的。随机森林、AdaBoost 和 GBRT 是您应该为大多数机器学习任务测试的第一批模型，它们在异构表格数据方面表现尤为出色。此外，由于它们需要非常少的预处理，因此非常适合快速搭建原型。最后，像投票分类器和堆叠分类器这样的集成方法可以帮助将系统的性能推向极限。

# 练习

1.  如果您在完全相同的训练数据上训练了五个不同的模型，并且它们都达到了 95%的精度，是否有可能将这些模型组合以获得更好的结果？如果可以，如何？如果不行，为什么？

1.  硬投票分类器和软投票分类器之间有什么区别？

1.  将一个装袋集成的训练分布到多个服务器上是否可以加快训练速度？那么对于粘贴集成、提升集成、随机森林或堆叠集成呢？

1.  什么是袋外评估的好处？

1.  什么使得额外树集成比常规随机森林更随机？这种额外的随机性如何帮助？额外树分类器比常规随机森林慢还是快？

1.  如果您的 AdaBoost 集成对训练数据拟合不足，应该调整哪些超参数，如何调整？

1.  如果您的梯度提升集成对训练集过拟合，应该增加还是减少学习率？

1.  加载 MNIST 数据集（在第三章中介绍），将其分为训练集、验证集和测试集（例如，使用 50,000 个实例进行训练，10,000 个用于验证，10,000 个用于测试）。然后训练各种分类器，如随机森林分类器、额外树分类器和 SVM 分类器。接下来，尝试将它们组合成一个集成，使用软投票或硬投票在验证集上优于每个单独的分类器。一旦找到一个，尝试在测试集上运行。它的表现比单个分类器好多少？

1.  运行上一个练习中的各个分类器，对验证集进行预测，并使用结果预测创建一个新的训练集：每个训练实例是一个向量，包含所有分类器对图像的预测集合，目标是图像的类别。在这个新的训练集上训练一个分类器。恭喜您——您刚刚训练了一个混合器，它与分类器一起形成了一个堆叠集成！现在在测试集上评估集成。对于测试集中的每个图像，使用所有分类器进行预测，然后将预测结果输入混合器以获得集成的预测。它与您之前训练的投票分类器相比如何？现在尝试使用`StackingClassifier`。性能更好吗？如果是，为什么？

这些练习的解决方案可以在本章笔记本的末尾找到，网址为[*https://homl.info/colab3*](https://homl.info/colab3)。

¹ 想象从一副牌中随机抽取一张卡片，写下来，然后将其放回到牌组中再抽取下一张卡片：同一张卡片可能被多次抽样。

² Leo Breiman，“Bagging 预测器”，*机器学习* 24, no. 2 (1996): 123–140。

³ 在统计学中，带替换的重采样被称为*自助法*。

⁴ Leo Breiman，“在大型数据库和在线分类中粘贴小投票”，*机器学习* 36, no. 1–2 (1999): 85–103。

⁵ 偏差和方差在第四章中介绍过。

⁶ `max_samples`也可以设置为 0.0 到 1.0 之间的浮点数，此时采样实例的最大数量等于训练集大小乘以`max_samples`。

⁷ 当*m*增长时，这个比率接近 1 – exp(–1) ≈ 63%。

⁸ Gilles Louppe 和 Pierre Geurts，“随机补丁上的集成”，*计算机科学讲义* 7523 (2012): 346–361。

⁹ 何天金，“用于构建决策森林的随机子空间方法”，*IEEE 模式分析与机器智能* 20, no. 8 (1998): 832–844。

¹⁰ 何天金，“随机决策森林”，*第三届文档分析与识别国际会议论文集* 1 (1995): 278。

¹¹ 如果您想要一个除决策树以外的袋子，`BaggingClassifier` 类仍然很有用。

¹² Pierre Geurts 等，“极端随机树”，*机器学习* 63, no. 1 (2006): 3–42。

¹³ Yoav Freund 和 Robert E. Schapire, “一个决策理论的在线学习泛化及其在 Boosting 中的应用”, *计算机与系统科学杂志* 55, no. 1 (1997): 119–139。
