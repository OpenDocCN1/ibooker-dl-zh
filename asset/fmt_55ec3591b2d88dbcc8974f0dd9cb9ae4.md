
除了这里提到的优势之外，还应注意的是，在某些场景下，GBDTs（梯度提升决策树）通常比 DNNs（深度神经网络）更受欢迎。一个原因是 GBDTs 需要更少的数据预处理，这使得它们更高效且易于实现。此外，在目标函数方面，GBDTs 与 DNNs 一样灵活。在两种情况下，都有许多可供选择，这在具有复杂优化目标的领域中特别有用。GBDTs 的另一个好处是，它们提供了更多控制决策树规则构建的方式，为用户提供了一定的透明度和可解释性。最后，GBDTs 在大多数情况下比 DNNs 训练得更快，并且根据它们的复杂性，它们也可以在合理的时间内进行预测，这在实时应用或时间敏感的任务中可能是一个关键因素。

现在你已经了解了梯度提升背后的基本概念及其在解决表格数据问题上的有效性，与深度学习相比，下一节将探讨一些其实现，从 Scikit-learn 提供的实现开始。

## 5.3 Scikit-learn 中的提升

Scikit-learn 为回归和分类任务提供了梯度提升算法。这些算法可以通过`GradientBoostingClassifier`（[`mng.bz/Ea9o`](https://mng.bz/Ea9o)）和`GradientBoostingRegressor`（[`mng.bz/N1VN`](https://mng.bz/N1VN)）类分别访问。

Scikit-learn 对梯度提升的实现是数据科学领域 Python 用户最早可用的选项之一。这个实现与 1999 年 Jerome Friedman 提出的算法原始提案非常相似[“Greedy Function Approximation: A Gradient Boosting Machine,” *Annals of Statistics* (2001): 1189–1232]。让我们在接下来的代码列表中看看实现的效果，我们将对 Airbnb NY 数据集上的分类器性能进行交叉验证，以预测列表价格是否高于或低于中位数。

**列表 5.10 Scikit-learn 梯度提升分类器**

```py
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
accuracy = make_scorer(accuracy_score)
cv = KFold(5, shuffle=True, random_state=0)
model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=4,
    min_samples_leaf=3, 
    random_state=0
)                                                            ①

model_pipeline = Pipeline(
    [('processing', column_transform),
     ('modeling', model)])                                   ②

cv_scores = cross_validate(
    estimator=model_pipeline, 
    X=data, 
    y=target_median,
    scoring=accuracy,
    cv=cv, 
    return_train_score=True,
    return_estimator=True
)                                                            ③

mean_cv = np.mean(cv_scores['test_score'])
std_cv = np.std(cv_scores['test_score'])
fit_time = np.mean(cv_scores['fit_time'])
score_time = np.mean(cv_scores['score_time'])
print(f"{mean_cv:0.3f} ({std_cv:0.3f})", 
      f"fit: {fit_time:0.2f}",
      f"secs pred: {score_time:0.2f} secs")                  ④
```

① 具有指定超参数的`GradientBoostingClassifier`模型

② 首先使用`column_transform`进行数据处理，然后拟合模型的管道

③ 使用定义的管道进行五折交叉验证，计算准确度分数，并返回额外信息

④ 打印交叉验证中准确度分数的平均值和标准差

使用之前使用的相同参数集，我们获得的效果略好于我们实现中获得的：

```py
0.826 (0.004) fit: 16.48 secs pred: 0.07 secs
```

Scikit-learn 实现的一个显著特点是训练过程可能需要相当长的时间，瓶颈可以归因于决策树，这是 Scikit-learn 本身使用的唯一支持构建序列集成模型。作为交换，你在参数上拥有一些灵活性，并且可以控制 GBDT 使用单个决策树的方式。例如，Scikit-learn 的梯度提升允许你

+   定义`init`函数。在我们的实现中，我们使用平均值作为第一个估计器。在这里，你可以使用任何你想要的估计器作为起点。由于梯度提升基于梯度下降，而梯度下降优化过程对起点敏感，因此在解决更复杂的数据问题时，这可能是一个优势。

+   通过训练具有指数损失的`GradientBoostingClassifier`（`loss="exponential"`）将算法回退到 Adaboost，这是启发序列集成的原始算法。

+   详细控制所使用的决策树的复杂性，这意味着你可以通过诸如

    +   `min_samples_split`用于分裂内部节点所需的最小样本数

    +   `min_sample_leaf`对于成为叶子节点所需的最小样本数

    +   `min_weight_fraction_leaf`对于在叶子节点所需的输入样本总权重中加权分数的最小值

    +   `max_depth`对于树的深度

    +   将`min_impurity_decrease`作为不纯度降低的阈值，用于决定是否分裂或停止树的生长

    +   `max_leaf_nodes`作为停止生长树之前可以达到的最大最终节点数

+   如果仅仅控制决策树的生长还不够，一旦它们生长出来，你可以使用`ccp_alpha`参数来减少它们的复杂性。此参数通过移除未通过复杂性测试的节点来从最终节点回退树（有关详细信息，请参阅[`mng.bz/DM9n`](https://mng.bz/DM9n)）。

+   对行（`subsample`）和列（`max_features`）进行子采样，这是一种特别有效的减少过拟合并提高训练模型泛化能力的方法。

此外，该实现还提供了对稀疏数据和支持早停的支持，这是一种防止 GBDT 和神经网络过拟合的程序，将在下一章中详细讨论。

此外，在提供的输出方面，此版本提供的支持相当令人印象深刻，使其成为理解并解释为什么你的 GBDT 模型做出某些预测的完美工具。例如，你可以访问所有使用的决策树，并要求从集成树中预测的原始值，无论是作为整体（`decision_function`方法）还是作为一系列步骤（`staged_decision_function`方法）。

最近，由于 XGBoost、LightGBM 和 Scikit-learn 的 HistGradientBoosting 提供的更快、更高效的解决方案，这种实现已被从业者较少使用。然而，如果您想控制梯度提升过程的某些方面，它仍然是一个有趣的选择。在下一节中，我们将探讨 XGBoost，并确定它如何成为解决您的表格数据问题的更强大选择。

### 5.3.1 应用提前停止以避免过拟合

原始 Scikit-learn 类梯度提升的展示提供了一个机会，介绍一个可以帮助控制过拟合的程序。该程序是提前停止，这是一种最初用于梯度下降的方法，用于限制在优化系数的进一步调整不会带来增强或导致解决方案泛化不良时迭代的数量。该方法也已被用于训练神经网络。在梯度提升中，它将梯度下降作为其优化过程的一部分，该方法可以帮助解决相同的问题：限制添加的决策树模型数量以减少计算负担并避免可能的过拟合。

提前停止的工作步骤如下：

1.  将训练数据集的一部分留出以形成验证集。

2.  在训练过程的每次迭代中，使用验证集评估产生的部分模型。

3.  记录并比较部分模型在验证集上的性能与之前的结果。

4.  如果模型的表现没有提高，算法会增加其自上次改进以来的迭代计数（通常称为*耐心*）。否则，它将重置计数。

5.  如果在一定的迭代次数内没有改进，则停止训练过程。

6.  否则，训练过程将继续进行另一轮迭代，除非所有指定的待提升决策树都已完成。此时，训练过程将停止。

图 5.8 展示了这个过程在流程图中的表示。

![图片](img/CH05_F08_Ryan2.png)

**图 5.8 描述了 GBDT 中提前停止如何工作的流程图。**

图 5.9 从验证指标的角度展示了完全相同的过程，该指标可以是损失或其他您希望获得最佳结果的指标。随着迭代的进行，观察到训练误差通常会降低。另一方面，验证误差在算法开始过拟合之前往往有一个最佳点。提前停止有助于捕捉验证误差的增加，监控验证误差的动态变化，让您能够回溯到任何过拟合发生之前的迭代。

![图片](img/CH05_F09_Ryan2.png)

**图 5.9 展示了在多次迭代使用 GBDT 时，验证误差和训练误差通常如何表现。**

在具有早停法的交叉验证过程中，针对每个折叠训练的模型可能会找到不同的停止点。当在整个数据集上训练时，您仍然可以依赖使用基于验证样本的早停法。因此，您只需设置一个高迭代次数，并观察训练何时停止。否则，您可以使用基于交叉验证中观察到的停止迭代次数的固定迭代次数。在这种情况下，您可以通过计算交叉验证中看到的所有停止点的平均值或中位数来确定要使用的提升树数量。然而，没有固定的规则，您可以选择使用第二大的值来实现激进的停止策略，或者使用第二小的值来实现保守的停止策略。

此外，考虑到训练数据比交叉验证期间更多，因此可以考虑增加提升树的数量。作为一个一般性指南，可以通过将提升树的数量增加一个百分比来达到这个目的，这个百分比等于交叉验证中使用的折叠数目的倒数。然而，由于没有一种适合所有情况的解决方案，可能需要进行实验以找到最适合您问题的最佳方法。

例如，我们再次运行之前的代码，这次设置更高的基本估计器数量以及两个参数，`validation_fraction`和`n_iter_no_change`，这些参数激活了早停过程。参数`validation_fraction`确定了用于验证的训练数据比例，并且仅在`n_iter_no_change`设置为一个整数时有效，该整数表示在停止过程之前，在验证集上测试模型时，应该经过多少次迭代而没有改进。

**列表 5.11 使用 GradientBoostingClassifier 应用早停法**

```py
model = GradientBoostingClassifier(
    n_estimators=1000,                                      ①
    learning_rate=0.1,
    validation_fraction=0.2,                                ②
    n_iter_no_change=10,                                    ③
    max_depth=4,
    min_samples_leaf=3, 
    random_state=0
)

model_pipeline = Pipeline(
    [('processing', column_transform),
     ('modeling', model)])

cv_scores = cross_validate(estimator=model_pipeline, 
                           X=data, 
                           y=target_median,
                           scoring=accuracy,
                           cv=cv, 
                           return_train_score=True,
                           return_estimator=True)

mean_cv = np.mean(cv_scores['test_score'])
std_cv = np.std(cv_scores['test_score'])
fit_time = np.mean(cv_scores['fit_time'])
score_time = np.mean(cv_scores['score_time'])
print(f"{mean_cv:0.3f} ({std_cv:0.3f})", 
      f"fit: {fit_time:0.2f} secs pred: {score_time:0.2f} secs")
iters = [cv_scores["estimator"][i].named_steps["modeling"].n_estimators_ 
         for i in range(5)]                                  ④
print(iters)                                                 ⑤
```

① 将迭代次数从之前的 300 提高到 1,000 的`GradientBoostingClassifier`模型

② 作为验证分数，`GradientBoostingClassifier`使用 20% 的训练数据用于验证。

③ 当验证集上没有改进时，`GradientBoostingClassifier`的训练将在 10 次迭代后停止。

④ 提取每个折叠估计器在训练期间使用的估计器数量

⑤ 打印每个折叠估计器的估计器数量列表

输出是

```py
0.826 (0.005) fit: 6.24 secs pred: 0.04 secs
[145, 109, 115, 163, 159]
```

由于交叉验证折叠的平均迭代次数为 268 次迭代，根据经验法则，在训练阶段使用所有可用数据时，我们建议将迭代次数增加 20%，固定为 322 次迭代。

在接下来的章节中，我们将介绍梯度提升的新实现，例如 XGBoost 和 LightGBM。我们还将展示如何使用它们实现早停法。

## 5.4 使用 XGBoost

XGBoost 在 Kaggle 竞赛中取得成功后获得了关注，例如在希格斯玻色子机器学习挑战赛（[`www.kaggle.com/c/higgs-boson`](https://www.kaggle.com/c/higgs-boson)）中，XGBoost 在竞赛论坛中被提出作为一种比 Scikit-learn 的梯度提升更快速、更准确的解决方案。自那时起，XGBoost 已被广泛应用于许多其他数据科学竞赛中，证明了其有效性和 Kaggle 竞赛作为介绍颠覆性能基准的创新的好场所的作用。Keras 是另一个在 Kaggle 竞赛中取得成功后被广泛采用的创新例子。在撰写本文时，XGBoost 软件包已更新并达到了 2.0.3 版本的里程碑，这是我们在这本书中使用的版本。

XGBoost 最初是由陈天奇作为研究项目提出的，后来在 Carlos Guestrin 的贡献下进一步发展，XGBoost 是一个作为开源软件提供的梯度提升框架。值得注意的是，与由微软赞助的 LightGBM 和其他由谷歌赞助的 Yggdrasil 决策森林（见[`mng.bz/lYV6`](https://mng.bz/lYV6)）等倡议不同，XGBoost 保持了完全独立，由分布式（深度）机器学习共同社区（[dmlc.github.io](https://dmlc.github.io/)）维护。

随着时间的推移，该框架经历了重大改进，现在提供了用于分布式处理和并行化的高级功能，使其能够处理大规模数据集。同时，XGBoost 也得到了广泛的应用，目前可在包括 C/C++、Python 和 R 在内的多种编程语言中使用。此外，该框架在多个数据科学平台上得到支持，如 H2O.ai 和 Apache Spark。

作为数据科学用户，你将立即注意到这个框架的几个关键特性，包括

+   处理各种输入数据类型的能力

+   支持自定义目标函数和评估函数

+   自动处理缺失值

+   简单支持 GPU 训练

+   适应单调性和特征交互约束

+   优化独立计算机上的多核和缓存

从系统性能的角度来看，显著的特点包括

+   网络并行训练，允许在机器集群上实现分布式计算

+   在构建树的过程中利用所有可用的 CPU 核心以实现并行化

+   处理不适合内存的大型数据集时的离核计算

然而，将“极端”XGBoost 算法与其他算法区分开来的是其创新的优化算法细节，包括被称为牛顿下降的梯度下降的变体、正则化项，以及其独特的特征分割和稀疏数据处理方法。以下部分简要总结了这些推动 XGBoost 性能的突破性技术。

现在，让我们尝试使用 Airbnb NYC 数据集上的分类任务来运行这个算法。在这种情况下，使用 XGBoost 的`XGBClassifier`。对于回归问题，你可以使用`XGBRegressor`类。然而，首先，你需要在你的系统上安装 XGBoost。要安装 XGBoost，你可以直接使用 pip 安装：

```py
pip install XGBoost
```

或者使用 conda 来完成这项工作：

```py
conda install -c conda-forge py-XGBoost
```
