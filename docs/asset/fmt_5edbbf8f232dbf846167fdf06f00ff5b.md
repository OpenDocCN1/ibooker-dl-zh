

# 第三章：开始基线：基准测试和优化

本章内容包括

+ 分析一对自然语言处理（NLP）问题
+ 使用关键的传统方法建立问题基线
+ 使用代表性的深度预训练语言模型 ELMo 和 BERT 进行基线

在本章中，我们继续直接深入解决 NLP 问题，这是我们在上一章开始的。我们继续追求建立一套具体 NLP 问题的基线，稍后我们将能够利用这些基线来衡量从越来越复杂的迁移学习方法中获得的逐渐改进。我们完成了我们在第二章开始的练习，那里我们介绍了一对实际问题，预处理了相应的数据，并通过探索一些广义线性方法开始了基线。特别是，我们介绍了电子邮件垃圾邮件和互联网电影数据库（IMDB）电影评论分类示例，并使用了逻辑回归和支持向量机（SVM）来建立它们的基线。

在本章中，我们探讨基于决策树和基于神经网络的方法。我们研究的基于决策树的方法包括随机森林和梯度提升机。关于基于神经网络的方法，我们将最简单形式的迁移学习应用到了一对最近流行的深度预训练语言模型 ELMo 和 BERT 上。这项工作只涉及在目标数据集上对每个网络的最后几层进行微调。这项活动将作为本书主题的应用性实践介绍，即 NLP 的迁移学习。此外，我们探索通过超参数调优来优化模型的性能。

我们将在下一节中探讨基于决策树的方法。

## 3.1 基于决策树的模型

决策树是一种决策支持工具，它将决策及其后果建模为*树*——一个图，其中任意两个节点都由一条路径连接。树的另一种定义是将输入值转换为输出类别的流程图。有关这种类型模型的更多详细信息，请参阅第一章。

在本节中，我们将两种最常见的基于决策树的方法——随机森林和梯度提升机——应用到我们的两个正在运行的问题上。

### 3.1.1 随机森林（RFs）

*随机森林*（RFs）通过生成大量专门的树并收集它们的输出，为应用决策树提供了一种实用的机器学习方法。RFs 非常灵活和广泛适用，通常是从逻辑回归后从业者尝试的第二种算法用于建立基线。有关 RFs 及其历史背景的更详细讨论，请参阅第一章。

让我们使用流行的库 `scikit-learn` 来构建我们的分类器，如下所示。

**列表 3.1 训练和测试随机森林分类器**

```py
from sklearn.ensemble import RandomForestClassifier          ❶

clf = RandomForestClassifier(n_jobs=1, random_state=0)       ❷

start_time = time.time()                                     ❸
clf.fit(train_x, train_y)
end_time = time.time()
print("Training the Random Forest Classifier took %3d seconds"%(end_time-start_time))

predicted_labels = clf.predict(test_x)

acc_score = accuracy_score(test_y, predicted_labels)

print("The RF testing accuracy score is::")
print(acc_score)
```

❶ 加载 `scikit` 的随机森林分类器库

❷ 创建一个随机森林分类器

❸ 训练分类器，以了解训练特征与训练响应变量的关系

用这段代码在我们的实验中，对电子邮件示例数据进行 RF 分类器的训练只需不到一秒钟的时间，并且达到了 0.945 的准确率分数。类似地，在 IMDB 示例上进行训练也只需不到一秒钟，并且达到了 0.665 的准确率分数。这个练习进一步证实了上一章的最初猜测，即 IMDB 评论问题比电子邮件分类问题更难。

### 3.1.2 梯度提升机（GBMs）

这种基于决策树的机器学习算法的变体迭代地学习新的基于决策树的模型，以解决前次迭代模型的弱点。在撰写本文时，它们被普遍认为是解决非感知机器学习问题的最佳类方法。不幸的是，它们确实存在一些缺点，包括较大的模型大小、过拟合的风险更高以及比其他一些决策树模型更少的可解释性。

训练梯度提升机（GBM）分类器的代码显示在下一个列表中。同样，我们使用 `scikit-learn` 中这些模型的实现。请注意，Python 库 `XGBoost` 中的实现被普遍认为更具有内存效率，并且更容易扩展/并行化。

**列表 3.2 训练/测试梯度提升机分类器**

```py
from sklearn.ensemble import GradientBoostingClassifier        ❶
from sklearn import metrics                                   ❷
from sklearn.model_selection import cross_val_score

def modelfit(alg, train_x, train_y, predictors, test_x, performCV=True, cv_folds=5):
    alg.fit(train_x, train_y)                                 ❸
    predictions = alg.predict(train_x)                        ❹
    predprob = alg.predict_proba(train_x)[:,1]
    if performCV:                                             ❺
        cv_score = cross_val_score(alg, train_x, train_y, cv=cv_folds, scoring='roc_auc')

    print("\nModel Report")                                  ❻
    print("Accuracy : %.4g" % metrics.accuracy_score(train_y,predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(train_y, predprob))
    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % 
(np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))

    return alg.predict(test_x),alg.predict_proba(test_x)      ❼
```

❶ GBM 算法

❷ 附加的 `sklearn` 函数

❸ 在整体数据上拟合算法

❹ 预测训练集

❺ 执行 k 折交叉验证

❻ 打印模型报告

❼ 预测测试数据

注意，在列表 3.2 中，除了通常的训练准确率分数外，我们还报告了 *k 折交叉验证* 和 *受试者工作特征曲线*（ROC）下的 *曲线下面积* 来评估模型。这是必要的，因为 GBMs 特别容易过拟合，报告这些指标有助于我们监控这种风险。另一个原因是，这个练习可以让你复习这些概念。

更具体地说，k 折交叉验证（默认值为 `k=5` 折）将训练数据集随机分为 k 个分区或折叠，并在 k-1 个分区上训练模型，同时在剩余的第 k 个分区上评估/验证性能，重复这个过程 k 次，每个分区都作为验证集。然后，它使用这些 k 次评估迭代的统计数据报告性能。这个过程允许我们减少模型在数据集的某些部分过拟合和在其他部分表现不佳的风险。

简而言之，过拟合是指将太多参数拟合到太少的数据中。这种情况会损害模型对新数据的泛化能力，并且通常表现为改善训练指标但验证指标没有改善。可以通过收集更多数据、简化模型以减少训练参数的数量以及本书中我们将重点介绍的其他方法来减轻这种情况。

以下代码可用于调用函数并在两个运行示例中对其进行评估：

```py
gbm0 = GradientBoostingClassifier(random_state=10)
start_time = time.time()
test_predictions, test_probs = modelfit(gbm0, train_x, train_y, predictors, test_x)
end_time = time.time()
print("Training the Gradient Boosting Classifier took %3d seconds"%(end_time-start_time))

predicted_labels = test_predictions
acc_score = accuracy_score(test_y, predicted_labels)
print("The Gradient Boosting testing accuracy score is::")
print(acc_score)
```

对于电子邮件垃圾邮件分类示例，这产生了以下结果：

```py
Model Report
Accuracy : 0.9814
AUC Score (Train): 0.997601
CV Score : Mean - 0.9854882 | Std - 0.006275645 | Min - 0.9770558 | Max - 0.9922158
Training the Gradient Boosting Classifier took 159 seconds
The Gradient Boosting testing accuracy score is::

```

对于 IMDB 电影评论分类示例，这产生了以下结果：

```py
Model Report
Accuracy : 0.8943
AUC Score (Train): 0.961556
CV Score : Mean - 0.707521 | Std - 0.03483452 | Min - 0.6635249 | Max - 0.7681968
Training the Gradient Boosting Classifier took 596 seconds
The Gradient Boosting testing accuracy score is::
0.665   
```

乍一看，看着这些结果，人们可能会倾向于认为与我们之前看过的先前方法相比，GBM 数值实验更昂贵——在 Kaggle 上 IMDB 示例完成需要将近 10 分钟。然而，我们必须考虑到当进行 k 折交叉验证练习时，模型会被训练 `k=5` 次以获得更可靠的性能估计。因此，每次训练大约需要两分钟——这并不像不考虑 k 折交叉验证而推断出的训练时间增加那么剧烈。

我们可以看到一些过拟合的证据——第一个示例的测试准确度低于 k 折训练准确度。此外，在 IMDB 示例中，k 折交叉验证分数明显低于整个数据集的训练分数，强调了在这种模型类型中使用 k 折交叉验证方法跟踪过拟合的重要性。我们在本章的倒数第二节中讨论了一些进一步提高分类器准确性的方法。

那么 ROC 曲线究竟是什么呢？它是假正率（FPR）与真正率（TPR）的曲线图，是用来评估和调整分类器的重要特征。它显示了这些分类器重要特性的权衡，当决策阈值——预测置信度开始被分类为给定类别的成员的概率值——在 0 和 1 之间变化时。现在可以使用以下代码来绘制这条曲线：

```py
test_probs_max = []                                                        ❶
for i in range(test_probs.shape[0]):
    test_probs_max.append(test_probs[i,test_y[i]])

fpr, tpr, thresholds = metrics.roc_curve(test_y, np.array(test_probs_max))  ❷

import matplotlib.pyplot as plt                                            ❸
fig,ax = plt.subplots()
plt.plot(fpr,tpr,label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Email Example')
plt.legend(loc="lower right")
plt.show()
```

❶ 我们首先需要找到每个示例的最大概率。

❷ 计算 ROC 曲线值

❸ 使用 `matplotlib` 库生成标记的 ROC 曲线图

邮件分类示例的结果 ROC 曲线如图 3.1 所示。斜率为 1 的直线代表 FPR 与 TPR 之间的权衡对应于随机机会。ROC 曲线越远离此线的左侧，分类器性能越好。因此，ROC 曲线下面积可用作性能的度量。

![03_01](img/03_01.png)

**图 3.1 邮件分类示例的 ROC 曲线**

决策树方法的一个重要特性是它们可以提供特征的重要性得分，这可以用来检测给定数据集中最重要的特征。我们通过在列表 3.2 函数的返回语句之前插入几行代码来做到这一点，如列表 3.3 中所示。

**列表 3.3 梯度提升机分类代码和特征重要性分数**

```py
from sklearn.ensemble import GradientBoostingClassifier               ❶
from sklearn import metrics                                          ❷
from sklearn.model_selection import cross_val_score

def modelfit(alg, train_x, train_y, predictors, test_x, performCV=True, cv_folds=5):
    alg.fit(train_x, train_y)                                        ❸
    predictions = alg.predict(train_x)                               ❹
    predprob = alg.predict_proba(train_x)[:,1]
    if performCV:                                                     ❺
        cv_score = cross_val_score(alg, train_x, train_y, cv=cv_folds, scoring='roc_auc')

    print("\nModel Report")                                          ❻
    print("Accuracy : %.4g" % metrics.accuracy_score(train_y,predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(train_y, predprob))
    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % 
(np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))

    feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False) 
    feat_imp[:10].plot(kind='bar',title='Feature Importances')        ❼

    return alg.predict(test_x),alg.predict_proba(test_x)              ❽
```

❶ GBM 算法

❷ 其他 `sklearn` 函数

❸ 在整体数据上拟合算法

❹ 预测训练集

❺ 执行 k 折交叉验证

❻ 打印模型报告

❼ 添加新的代码来计算特征的重要性

❽ 预测测试数据

对于 IMDB 示例，这产生了图 3.2 的绘图。我们看到像“worst”和“awful”这样的词对分类决策非常重要，这在定性上是有意义的，因为可以想象负面评论家使用这些词。另一方面，“loved”这样的词可能会被积极的评论者使用。

注意：重要性分数在这个示例中似乎效果很好，但不应该一味地相信它们。例如，已经广泛认识到这些重要性分数可能对连续变量以及高基数分类变量有偏见。

现在我们开始应用一些神经网络模型到我们两个运行的示例中，可以说神经网络是当今自然语言处理中最重要的模型类别之一。

## 3.2 神经网络模型

正如我们在第一章讨论的那样，神经网络是处理感知问题（如计算机视觉和自然语言处理）最重要的机器学习算法类别之一。

在本节中，我们将在本章和上一章中基线化的两个示例问题上训练两个代表性的预训练神经网络语言模型。我们考虑来自语言模型的嵌入（ELMo）和来自 transformers 的双向编码器表示（BERT）。

![03_02](img/03_02.png)

**图 3.2 在 IMDB 分类示例中由梯度提升机分类器发现的各种标记的重要性分数**

