
![说明不同决策阈值下精度/召回权衡的图；随着阈值的增加，精确度通常增加，而召回率降低。](img/hmls_0304.png)

###### 图 3-4。精度/召回权衡：图像按其分类器分数排序，高于所选决策阈值的被视为正例；阈值越高，召回率越低，但（通常）精确度越高

你可以调用分类器的`decision_function()`方法而不是调用其`predict()`方法，该方法为每个实例返回一个分数。然后你可以使用任何你想要的阈值来根据这些分数做出预测：

```py
>>> y_scores = sgd_clf.decision_function([some_digit])
>>> y_scores
array([2164.22030239])
>>> threshold = 0
>>> y_some_digit_pred = (y_scores > threshold)
>>> y_some_digit_pred
array([ True])
```

`SGDClassifier`使用阈值为 0，因此前面的代码返回的结果与`predict()`方法相同（即`True`）。让我们提高阈值：

```py
>>> threshold = 3000
>>> y_some_digit_pred = (y_scores > threshold)
>>> y_some_digit_pred
array([False])
```

这证实了提高阈值会降低召回率。图像实际上代表一个 5，当阈值是 0 时，分类器检测到它，但当阈值增加到 3,000 时，它就错过了它。

你如何决定使用哪个阈值？一个选择是使用`cross_val_predict()`函数来获取训练集中所有实例的分数，但这次指定你想要返回决策分数而不是预测：

```py
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")
```

使用这些分数，使用`precision_recall_curve()`函数计算所有可能阈值下的精确度和召回率（该函数添加一个最后的精确度为 1 和一个最后的召回率为 0，对应于无限大的阈值）：

```py
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
```

最后，使用 Matplotlib 将精确度和召回率作为阈值值的函数绘制出来（图 3-5）。让我们展示我们选择的 3,000 个阈值的阈值：

```py
plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
plt.vlines(threshold, 0, 1.0, "k", "dotted", label="threshold")
[...]  # beautify the figure: add grid, legend, axis, labels, and circles
plt.show()
```

![显示精度和召回率作为决策阈值函数的图表，说明它们之间的反向关系。](img/hmls_0305.png)

###### 图 3-5\. 精确度和召回率与决策阈值的关系

###### 注意

你可能会想知道为什么 图 3-5 中的精确度曲线比召回率曲线更崎岖。原因是当提高阈值时，精确度有时会下降（尽管总体上会上升）。要理解为什么，回顾 图 3-4 并注意当你从中心阈值开始并向右移动一个数字时会发生什么：精确度从 4/5（80%）下降到 3/4（75%）。另一方面，召回率只有在增加阈值时才会下降，这解释了为什么其曲线看起来很平滑。

在这个阈值值下，精确度接近 90%，召回率大约为 50%。另一种选择良好的精确度/召回率权衡的方法是将精确度直接与召回率绘制在一起，如图 图 3-6 所示（显示相同的阈值）：

```py
plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")
[...]  # beautify the figure: add labels, grid, legend, arrow, and text
plt.show()
```

![显示精确度-召回率曲线并标记精确度在召回率接近 80% 时急剧下降的点](img/hmls_0306.png)

###### 图 3-6\. 精确度与召回率

你可以看到，精确度确实在召回率大约为 80% 时开始急剧下降。你可能想在下降之前选择一个精确度/召回率的权衡——例如，在召回率大约为 60% 时。但当然，选择取决于你的项目。

假设你决定将目标设定为 90% 的精确度。你可以使用第一个图表来找到所需的阈值，但这并不非常精确。或者，你可以寻找至少提供 90% 精确度的最低阈值。为此，你可以使用 NumPy 数组的`argmax()`方法。这个方法返回最大值的第一个索引，在这个情况下意味着第一个`True`值：

```py
>>> idx_for_90_precision = (precisions >= 0.90).argmax()
>>> threshold_for_90_precision = thresholds[idx_for_90_precision]
>>> threshold_for_90_precision
np.float64(3370.0194991439557)
```

要进行预测（目前是在训练集上），而不是调用分类器的`predict()`方法，你可以运行以下代码：

```py
y_train_pred_90 = (y_scores >= threshold_for_90_precision)
```

让我们检查这些预测的精确度和召回率：

```py
>>> precision_score(y_train_5, y_train_pred_90)
0.9000345901072293
>>> recall_at_90_precision = recall_score(y_train_5, y_train_pred_90)
>>> recall_at_90_precision
0.4799852425751706
```

太好了，你有一个 90% 精确度的分类器！正如你所见，创建一个具有几乎任何所需精确度的分类器相当容易：只需设置一个足够高的阈值，就完成了。但是等等，不要这么快：如果其召回率太低，一个高精确度的分类器并不很有用！对于许多应用来说，48% 的召回率根本不会很好。

###### 小贴士

如果有人说，“让我们达到 99% 的精确度”，你应该问，“在什么召回率下？”

自从 Scikit-Learn 1.5 版本以来，有两个新的类可以帮助你更容易地调整决策阈值：

+   `FixedThresholdClassifier`类允许你包装一个二元分类器并手动设置所需的阈值。如果底层分类器有`predict_proba()`方法，那么阈值应该是一个介于 0 和 1 之间的值（默认是 0.5）。否则，它应该是一个决策分数，与模型`decision_function()`的输出相当（默认是 0）。

+   `TunedThresholdClassifierCV`类使用*k*-折交叉验证自动找到给定指标的优化阈值。默认情况下，它试图找到最大化模型**平衡精度**的阈值：这是每个类召回率的平均值。然而，你可以选择另一个指标进行优化（请参阅文档以获取完整选项列表）。

## ROC 曲线

接收者操作特征（ROC）曲线是另一种与二元分类器一起使用的常用工具。它与精确度/召回率曲线非常相似，但不是绘制精确度与召回率的对比图，而是 ROC 曲线绘制了**真正例率**（召回率的另一种名称）与**假正例率**（FPR）的对比图。FPR（也称为*漏报*）是错误地被分类为正的负实例的比例。它等于 1 – **真正例率**（TNR），即正确地将负实例分类为负的比例。TNR 也称为*特异性*。因此，ROC 曲线绘制了*灵敏度*（召回率）与 1 – *特异性*的对比图。

要绘制 ROC 曲线，你首先使用`roc_curve()`函数计算各种阈值值下的 TPR 和 FPR：

```py
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
```

然后，你可以使用 Matplotlib 绘制 FPR 与 TPR 的对比图。以下代码生成了图 3-7 中的图表。为了找到对应于 90%精度的点，我们需要查找所需阈值的索引。由于在这种情况下阈值按递减顺序列出，我们在第一行使用`<=`而不是`>=`：

```py
idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
plt.plot([fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")
[...]  # beautify the figure: add labels, grid, legend, arrow, and text
plt.show()
```

再次，这里有一个权衡：召回率（TPR）越高，分类器产生的假阳性（FPR）就越多。虚线表示纯随机分类器的 ROC 曲线；一个好的分类器应尽可能地远离这条线（朝向左上角）。

比较分类器的一种方法是通过测量曲线下的面积（AUC）。一个完美的分类器将具有等于 1 的 ROC AUC，而一个纯随机分类器将具有等于 0.5 的 ROC AUC。Scikit-Learn 提供了一个函数来估计 ROC AUC：

```py
>>> from sklearn.metrics import roc_auc_score
>>> roc_auc_score(y_train_5, y_scores)
np.float64(0.9604938554008616)
```

![ROC 曲线显示真正例率与假正例率的对比，突出显示的点表示 90%精度和 48%召回率的阈值。](img/hmls_0307.png)

###### 图 3-7\. 绘制所有可能阈值下假正例率与真正例率的 ROC 曲线；黑色圆圈突出显示所选比率（在 90%精度和 48%召回率处）

###### 小贴士

由于 ROC 曲线与精度/召回（PR）曲线非常相似，你可能会想知道如何决定使用哪一个。一般来说，当正类很少或你更关心假阳性而不是假阴性时，你应该更喜欢 PR 曲线。否则，使用 ROC 曲线。例如，查看之前的 ROC 曲线（以及 ROC AUC 分数），你可能会认为分类器非常好。但这主要是因为正例（5s）与负例（非-5s）相比很少。相比之下，PR 曲线清楚地表明分类器还有改进的空间：曲线实际上可以更接近右上角（再次查看图 3-6）。

现在我们创建一个`RandomForestClassifier`，我们可以将其 PR 曲线和 F[1]分数与`SGDClassifier`的进行比较：

```py
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
```

`precision_recall_curve()`函数需要每个实例的标签和分数，因此我们需要训练随机森林分类器并使其为每个实例分配一个分数。但由于其工作方式（我们将在第六章中介绍这一点），`RandomForestClassifier`类没有`decision_function()`方法。幸运的是，它有一个`predict_proba()`方法，该方法返回每个实例的估计类别概率，我们可以直接使用正类概率作为分数，这样`precision_recall_curve()`就可以工作了.^(4) 我们可以调用`cross_val_​predict()`函数，使用交叉验证训练`RandomForestClassifier`，并使其预测每个图像的类别概率，如下所示：

```py
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")
```

让我们查看训练集中前两张图像的估计类别概率：

```py
>>> y_probas_forest[:2]
array([[0.11, 0.89],
 [0.99, 0.01]])
```

模型预测第一张图像有 89%的概率是正例，它预测第二张图像有 99%的概率是负例。由于每张图像要么是正例要么是负例，每行中的估计概率总和为 100%。

###### 警告

这些是**估计**的概率，而不是实际概率。例如，如果你查看模型将所有估计概率在 50% 到 60% 之间的图像分类为正例的所有图像，大约 94% 的它们实际上是正例。因此，在这种情况下，模型的估计概率太低了——但模型也可能过于自信。`sklearn.calibration`包中的`CalibratedClassifierCV`类可以使用交叉验证校准估计概率，使它们更接近实际概率（请参阅笔记本中的代码示例）。在某些场景中，这很重要，例如医疗诊断、财务风险评估或欺诈检测。

第二列包含正类的估计概率，因此让我们将它们传递给`precision_recall_curve()`函数：

```py
y_scores_forest = y_probas_forest[:, 1]
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(
    y_train_5, y_scores_forest)
```

现在我们准备绘制 PR 曲线。绘制第一个 PR 曲线也很有用，以便比较它们（图 3-8）：

```py
plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2,
         label="Random Forest")
plt.plot(recalls, precisions, "--", linewidth=2, label="SGD")
[...]  # beautify the figure: add labels, grid, and legend
plt.show()
```

![精度-召回曲线比较了 RandomForestClassifier 和 SGDClassifier 的性能，RandomForest 曲线更接近右上角，表明性能更优。](img/hmls_0308.png)

###### 图 3-8\. 比较 PR 曲线：随机森林分类器优于 SGD 分类器，因为它的 PR 曲线更接近右上角，并且具有更大的 AUC

如图 3-8 所示，`RandomForestClassifier`的 PR 曲线看起来比`SGDClassifier`的更好：它更接近右上角。它的 F[1]分数和 ROC AUC 分数也明显更好：

```py
>>> y_train_pred_forest = y_probas_forest[:, 1] >= 0.5  # positive proba ≥ 50%
>>> f1_score(y_train_5, y_train_pred_forest)
0.9274509803921569
>>> roc_auc_score(y_train_5, y_scores_forest)
0.9983436731328145
```

尝试测量精度和召回分数：你应该找到大约 99.0% 的精度和 87.3% 的召回率。还不错！

你现在知道如何训练二元分类器，选择适合你任务的适当指标，使用交叉验证评估你的分类器，选择适合你需求的精度/召回权衡，并使用多个指标和曲线来比较各种模型。你已经准备好尝试检测不仅仅是 5s 了。

# 多类别分类

与二元分类器区分两个类别不同，*多类别分类器*（也称为*多项式分类器*）可以区分超过两个类别。

一些 Scikit-Learn 分类器（例如，`LogisticRegression`，`RandomForestClassifier`和`GaussianNB`）能够原生地处理多个类别。其他的是严格的二元分类器（例如，`SGDClassifier`和`SVC`）。然而，你可以使用各种策略来使用多个二元分类器执行多类别分类。

将数字图像分类到 10 个类别（从 0 到 9）的一种方法是训练 10 个二元分类器，每个数字一个（一个 0 检测器，一个 1 检测器，一个 2 检测器，以此类推）。然后当你想要对一张图像进行分类时，你从每个分类器那里获取该图像的决策分数，并选择输出最高分数的类别。这被称为*一对余下所有*（OvR）策略，有时也称为*一对所有*（OvA）。
