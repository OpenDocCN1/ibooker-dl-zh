
为了理解这种权衡，让我们看看`SGDClassifier`是如何做出分类决策的。对于每个实例，它根据*决策函数*计算得分。如果该得分大于阈值，则将实例分配给正类；否则将其分配给负类。图 3-4 显示了一些数字，从最低得分的左侧到最高得分的右侧。假设*决策阈值*位于中间箭头处（两个 5 之间）：你会发现在该阈值右侧有 4 个真正例（实际为 5），以及 1 个假正例（实际上是 6）。因此，使用该阈值，精度为 80%（5 个中的 4 个）。但在 6 个实际为 5 的情况下，分类器只检测到 4 个，因此召回率为 67%（6 个中的 4 个）。如果提高阈值（将其移动到右侧的箭头处），假正例（6）变为真负例，从而增加精度（在这种情况下最高可达 100%），但一个真正例变为假负例，将召回率降低到 50%。相反，降低阈值会增加召回率并降低精度。

![mls3 0304](img/mls3_0304.png)

###### 图 3-4。精度/召回率权衡：图像按其分类器得分排名，高于所选决策阈值的图像被视为正例；阈值越高，召回率越低，但（一般而言）精度越高

Scikit-Learn 不允许直接设置阈值，但它确实让您访问它用于做出预测的决策得分。您可以调用分类器的`decision_function()`方法，而不是调用`predict()`方法，该方法返回每个实例的得分，然后根据这些得分使用任何阈值进行预测：

```py
>>> y_scores = sgd_clf.decision_function([some_digit])
>>> y_scores
array([2164.22030239])
>>> threshold = 0
>>> y_some_digit_pred = (y_scores > threshold)
array([ True])
```

`SGDClassifier`使用阈值等于 0，因此前面的代码返回与`predict()`方法相同的结果（即`True`）。让我们提高阈值：

```py
>>> threshold = 3000
>>> y_some_digit_pred = (y_scores > threshold)
>>> y_some_digit_pred
array([False])
```

这证实了提高阈值会降低召回率。图像实际上代表一个 5，当阈值为 0 时分类器检测到它，但当阈值增加到 3,000 时却错过了它。

如何决定使用哪个阈值？首先，使用`cross_val_predict()`函数获取训练集中所有实例的分数，但是这次指定要返回决策分数而不是预测：

```py
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")
```

使用这些分数，使用`precision_recall_curve()`函数计算所有可能阈值的精度和召回率（该函数添加最后一个精度为 0 和最后一个召回率为 1，对应于无限阈值）：

```py
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
```

最后，使用 Matplotlib 绘制精度和召回率作为阈值值的函数（图 3-5）。让我们展示我们选择的 3,000 的阈值：

```py
plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
plt.vlines(threshold, 0, 1.0, "k", "dotted", label="threshold")
[...]  # beautify the figure: add grid, legend, axis, labels, and circles
plt.show()
```

![mls3 0305](img/mls3_0305.png)

###### 图 3-5. 精度和召回率与决策阈值

###### 注意

你可能会想知道为什么图 3-5 中的精度曲线比召回率曲线更加崎岖。原因是当你提高阈值时，精度有时会下降（尽管通常会上升）。要理解原因，请回顾图 3-4，注意当你从中心阈值开始，将其向右移动一个数字时会发生什么：精度从 4/5（80%）下降到 3/4（75%）。另一方面，当增加阈值时，召回率只能下降，这解释了为什么其曲线看起来平滑。

在这个阈值下，精度接近 90%，召回率约为 50%。选择一个良好的精度/召回率折衷的另一种方法是直接绘制精度与召回率的图表，如图 3-6 所示（显示相同的阈值）：

```py
plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")
[...]  # beautify the figure: add labels, grid, legend, arrow, and text
plt.show()
```

![mls3 0306](img/mls3_0306.png)

###### 图 3-6. 精度与召回率

你可以看到精度在约 80%的召回率处开始急剧下降。你可能希望在该下降之前选择一个精度/召回率折衷，例如在约 60%的召回率处。但是，选择取决于你的项目。

假设你决定追求 90%的精度。你可以使用第一个图表找到需要使用的阈值，但这不太精确。或者，你可以搜索给出至少 90%精度的最低阈值。为此，你可以使用 NumPy 数组的`argmax()`方法。这将返回最大值的第一个索引，这在这种情况下意味着第一个`True`值：

```py
>>> idx_for_90_precision = (precisions >= 0.90).argmax()
>>> threshold_for_90_precision = thresholds[idx_for_90_precision]
>>> threshold_for_90_precision
3370.0194991439557
```

要进行预测（目前只是在训练集上），而不是调用分类器的`predict()`方法，你可以运行这段代码：

```py
y_train_pred_90 = (y_scores >= threshold_for_90_precision)
```

让我们检查这些预测的精度和召回率：

```py
>>> precision_score(y_train_5, y_train_pred_90)
0.9000345901072293
>>> recall_at_90_precision = recall_score(y_train_5, y_train_pred_90)
>>> recall_at_90_precision
0.4799852425751706
```

太棒了，你有一个 90%的精度分类器！正如你所看到的，几乎可以轻松地创建一个任意精度的分类器：只需设置足够高的阈值，就可以了。但是等等，不要那么快——如果召回率太低，高精度的分类器就不太有用！对于许多应用程序来说，48%的召回率根本不好。

###### 提示

如果有人说：“让我们达到 99%的精度”，你应该问：“召回率是多少？”

## ROC 曲线

*接收者操作特征*（ROC）曲线是与二元分类器一起使用的另一个常见工具。它与精度/召回率曲线非常相似，但是 ROC 曲线不是绘制精度与召回率，而是绘制*真正例率*（召回率的另一个名称）与*假正例率*（FPR）。FPR（也称为*误报率*）是被错误分类为正例的负实例的比率。它等于 1 减去*真负例率*（TNR），即被正确分类为负例的负实例的比率。TNR 也称为*特异性*。因此，ROC 曲线绘制*灵敏度*（召回率）与 1-*特异性*。

要绘制 ROC 曲线，首先使用`roc_curve()`函数计算各种阈值的 TPR 和 FPR：

```py
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
```

然后，您可以使用 Matplotlib 绘制 FPR 与 TPR。以下代码生成图 3-7 中的图。要找到对应于 90%精确度的点，我们需要查找所需阈值的索引。由于在这种情况下，阈值按降序列出，因此我们在第一行上使用`<=`而不是`>=`：

```py
idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
plt.plot([fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")
[...]  # beautify the figure: add labels, grid, legend, arrow, and text
plt.show()
```

![mls3 0307](img/mls3_0307.png)

###### 图 3-7。ROC 曲线绘制了所有可能阈值的假阳性率与真阳性率之间的关系；黑色圆圈突出显示了选择的比率（在 90%精确度和 48%召回率处）

再次存在权衡：召回率（TPR）越高，分类器产生的假阳性（FPR）就越多。虚线代表纯随机分类器的 ROC 曲线；一个好的分类器尽可能远离该线（朝向左上角）。

比较分类器的一种方法是测量*曲线下面积*（AUC）。完美的分类器的 ROC AUC 等于 1，而纯随机分类器的 ROC AUC 等于 0.5。Scikit-Learn 提供了一个估计 ROC AUC 的函数：

```py
>>> from sklearn.metrics import roc_auc_score
>>> roc_auc_score(y_train_5, y_scores)
0.9604938554008616
```

###### 提示

由于 ROC 曲线与精确率/召回率（PR）曲线非常相似，您可能想知道如何决定使用哪个。作为经验法则，当正类别很少或您更关心假阳性而不是假阴性时，应优先选择 PR 曲线。否则，请使用 ROC 曲线。例如，查看先前的 ROC 曲线（以及 ROC AUC 分数），您可能会认为分类器非常好。但这主要是因为与负例（非 5）相比，正例（5）很少。相比之下，PR 曲线清楚地表明分类器有改进的空间：曲线实际上可以更接近右上角（再次参见图 3-6）。

现在让我们创建一个`RandomForestClassifier`，我们可以将其 PR 曲线和 F[1]分数与`SGDClassifier`的进行比较：

```py
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
```

`precision_recall_curve()`函数期望每个实例的标签和分数，因此我们需要训练随机森林分类器并使其为每个实例分配一个分数。但是，由于`RandomForestClassifier`类的工作方式（我们将在第七章中介绍），它没有`decision_function()`方法。幸运的是，它有一个`predict_proba()`方法，为每个实例返回类概率，并且我们可以将正类别的概率作为分数，因此它将正常工作。⁴我们可以调用`cross_val_predict()`函数，使用交叉验证训练`RandomForestClassifier`，并使其为每个图像预测类概率，如下所示：

```py
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")
```

让我们看看训练集中前两个图像的类概率：

```py
>>> y_probas_forest[:2]
array([[0.11, 0.89],
 [0.99, 0.01]])
```

模型预测第一幅图像为正面的概率为 89%，并且预测第二幅图像为负面的概率为 99%。由于每幅图像要么是正面要么是负面，因此每行中的概率总和为 100%。

###### 警告

这些是*估计*概率，而不是实际概率。例如，如果您查看模型将其分类为正面的所有图像，其估计概率在 50%到 60%之间，大约 94%实际上是正面的。因此，在这种情况下，模型的估计概率要低得多，但模型也可能过于自信。`sklearn.calibration`包含工具，可以校准估计的概率，使其更接近实际概率。有关更多详细信息，请参阅[本章笔记本](https://homl.info/colab3)中的额外材料部分。

第二列包含正类别的估计概率，因此让我们将它们传递给`precision_recall_curve()`函数：

```py
y_scores_forest = y_probas_forest[:, 1]
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(
    y_train_5, y_scores_forest)
```

现在我们准备绘制 PR 曲线。为了查看它们的比较，也有必要绘制第一个 PR 曲线（图 3-8）： 

```py
plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2,
         label="Random Forest")
plt.plot(recalls, precisions, "--", linewidth=2, label="SGD")
[...]  # beautify the figure: add labels, grid, and legend
plt.show()
```

![mls3 0308](img/mls3_0308.png)

###### 图 3-8。比较 PR 曲线：随机森林分类器优于 SGD 分类器，因为其 PR 曲线更接近右上角，并且具有更大的 AUC

正如您在图 3-8 中所看到的，`RandomForestClassifier`的 PR 曲线看起来比`SGDClassifier`的要好得多：它更接近右上角。它的 F[1]分数和 ROC AUC 分数也显著更好：

```py
>>> y_train_pred_forest = y_probas_forest[:, 1] >= 0.5  # positive proba ≥ 50%
>>> f1_score(y_train_5, y_pred_forest)
0.9242275142688446
>>> roc_auc_score(y_train_5, y_scores_forest)
0.9983436731328145
```

尝试测量精确度和召回率得分：您应该会发现大约 99.1%的精确度和 86.6%的召回率。还不错！

您现在知道如何训练二元分类器，选择适合您任务的适当度量标准，使用交叉验证评估您的分类器，选择适合您需求的精确度/召回率折衷，并使用多种指标和曲线比较各种模型。您已经准备好尝试检测不仅仅是数字 5 了。

# 多类分类

而二元分类器区分两个类别，*多类分类器*（也称为*多项分类器*）可以区分两个以上的类别。

一些 Scikit-Learn 分类器（例如`LogisticRegression`、`RandomForestClassifier`和`GaussianNB`）能够本地处理多个类别。其他严格的二元分类器（例如`SGDClassifier`和`SVC`）。然而，有各种策略可用于使用多个二元分类器执行多类分类。

创建一个能够将数字图像分类为 10 个类别（从 0 到 9）的系统的一种方法是训练 10 个二元分类器，每个数字一个（一个 0 检测器，一个 1 检测器，一个 2 检测器，依此类推）。然后，当您想要对一幅图像进行分类时，您会从每个分类器中获取该图像的决策分数，并选择输出最高分数的类别。这被称为*一对剩余*（OvR）策略，有时也称为*一对所有*（OvA）。

另一种策略是为每对数字训练一个二元分类器：一个用于区分 0 和 1，另一个用于区分 0 和 2，另一个用于 1 和 2，依此类推。这被称为*一对一*（OvO）策略。如果有*N*个类别，您需要训练*N*×（*N* - 1）/ 2 个分类器。对于 MNIST 问题，这意味着训练 45 个二元分类器！当您想要对一幅图像进行分类时，您必须通过所有 45 个分类器并查看哪个类别赢得了最多的决斗。OvO 的主要优势在于每个分类器只需要在包含它必须区分的两个类别的训练集部分上进行训练。

一些算法（如支持向量机分类器）随着训练集的大小而扩展得很差。对于这些算法，OvO 更受青睐，因为在小训练集上训练许多分类器比在大训练集上训练少数分类器要快。然而，对于大多数二元分类算法，OvR 更受青睐。

Scikit-Learn 会检测到您尝试将二元分类算法用于多类分类任务时，并根据算法自动运行 OvR 或 OvO。让我们尝试使用`sklearn.svm.SVC`类中的支持向量机分类器（参见第五章）。我们只会在前 2,000 幅图像上进行训练，否则会花费很长时间：
