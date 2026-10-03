
```py
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

numeric_standardization = Pipeline([
       ("imputation", SimpleImputer(strategy="constant", fill_value=0)),
       ("standardizing", StandardScaler())
       ])

accuracy = make_scorer(accuracy_score)                            ①
cv = KFold(5, shuffle=True, random_state=0)                       ②
model = SVC(
    C=1.0,
    kernel='rbf',
    gamma='scale',
    probability=False
)                                                                 ③

column_transform = ColumnTransformer(
    [('categories', categorical_onehot_encoding, low_card_categorical),
     ('numeric', numeric_standardization, continuous)],
    remainder='drop',
    verbose_feature_names_out=False,
    sparse_threshold=0.0)                                         ④

model_pipeline = Pipeline(
    [('processing', column_transform),
     ('modeling', model)])                                        ⑤

cv_scores = cross_validate(estimator=model_pipeline,
                           X=data,
                           y=target_median,
                           scoring=accuracy,
                           cv=cv,
                           return_train_score=True,
                           return_estimator=True)                 ⑥

mean_cv = np.mean(cv_scores['test_score'])
std_cv = np.std(cv_scores['test_score'])
fit_time = np.mean(cv_scores['fit_time'])
score_time = np.mean(cv_scores['score_time'])
print(f"{mean_cv:0.3f} ({std_cv:0.3f})",
      f"fit: {fit_time:0.2f}",
      f"secs pred: {score_time:0.2f} secs")                       ⑦
```

① 使用`accuracy_score`度量创建一个评分函数

② 创建一个具有洗牌和固定随机状态的五折交叉验证迭代器

③ 创建一个具有指定超参数的支持向量机分类器实例

④ 定义一个`ColumnTransformer`以预处理特征，对低基数分类特征应用独热编码，对数值特征进行标准化

⑤ 创建一个管道，按顺序应用列转换

⑥ 使用定义的管道在数据上执行交叉验证，使用准确率评分

⑦ 打印测试分数的均值和标准差

结果相当有趣，通过调整超参数可能还会变得更好：

```py
0.821 (0.004) fit: 102.28 secs pred: 9.80 secs
```

然而，训练单个折叠所需的时间与所有之前的机器学习算法相比过于冗长。在本附录的下一节中，我们将讨论如何使用 GPU 卡加速过程，同时仍然使用`Scikit-learn` API。

## B.3 使用 GPU 进行机器学习

由于深度学习在数据科学领域的迅速崛起，GPU 现在在本地和云计算中都得到了广泛应用。以前，你只听说过 GPU 在 3D 游戏、图形处理渲染和动画中的应用。由于它们便宜且擅长快速矩阵乘法任务，学者和实践者迅速将 GPU 用于神经网络计算。`RAPIDS`是由 NVIDIA（GPU 顶级制造商之一）开发的一系列用于在 GPU 上执行数据科学全光谱的包，而不仅仅是深度学习。`RAPIDS`包承诺帮助机器学习管道的各个阶段，从端到端。这对许多经典机器学习算法来说是一个变革，特别是对于 SVMs，它是处理涉及噪声、异常值和大型数据集（特别是如果特征多线性或稀疏）的复杂任务的最可靠选择。在`RAPIDS`包（表 B.1）中，所有命令都采用了现有的 API 作为它们的命令。这确保了包的即时市场采用，对于用户来说，无需重新学习轮子的工作方式。

**表 B.1 Rapids 包**

| Rapids 包 | 任务 | API 模拟 |
| --- | --- | --- |
| cuPy | 数组操作 | NumPy |
| cuDF | 数据处理 | pandas |
| cuML | 机器学习 | Scikit-learn |

本节将重点介绍如何轻松地将你的`Scikit-learn`算法替换为`RAPIDS cuML`包。目前，此包包括线性模型、k-NN 和 SVMs 的实现，以及聚类和降维。以下列表显示了测试支持向量分类器（使用我们在上一节中尝试的径向基函数核）的`RAPIDS`实现（使用 P100 GPU）的代码。

**列表 B.3 RAPIDS cuML 支持向量分类器**

```py
from cuml.svm import SVC
from sklearn.metrics import accuracy_score

accuracy = make_scorer(accuracy_score)                            ①
cv = KFold(5, shuffle=True, random_state=0)                       ②
model = SVC(
    C=1.0,
    kernel='rbf',
    gamma='scale',
    probability=False
)                                                                 ③

column_transform = ColumnTransformer(
    [('categories', categorical_onehot_encoding, low_card_categorical),
     ('numeric', numeric_standardization, continuous)],
    remainder='drop',
    verbose_feature_names_out=False,
    sparse_threshold=0.0)                                         ④

model_pipeline = Pipeline(
    [('processing', column_transform),
     ('modeling', model)])                                        ⑤

cv_scores = cross_validate(estimator=model_pipeline,
                           X=data,
                           y=target_median,
                           scoring=accuracy,
                           cv=cv,
                           return_train_score=True,
                           return_estimator=True)                 ⑥

mean_cv = np.mean(cv_scores['test_score'])
std_cv = np.std(cv_scores['test_score'])
fit_time = np.mean(cv_scores['fit_time'])
score_time = np.mean(cv_scores['score_time'])
print(f"{mean_cv:0.3f} ({std_cv:0.3f})",
      f"fit: {fit_time:0.2f}",
      f"secs pred: {score_time:0.2f} secs")                       ⑦
```

① 使用`accuracy_score`度量创建评分函数

② 创建一个具有洗牌和固定随机状态的五折交叉验证迭代器

③ 从 GPU 加速的`cuML`库中创建一个支持向量分类器实例，并指定超参数

④ 定义一个`ColumnTransformer`来预处理特征，对低基数分类特征应用独热编码，对数值特征应用标准化

⑤ 创建一个管道，按顺序将列转换和模型应用于数据

⑥ 使用定义的管道在数据上执行交叉验证，并使用准确率评分

⑦ 打印测试分数的均值和标准差

我们获得的结果是

```py
0.821 (0.004) fit: 4.09 secs pred: 0.11 secs
```

如您所见，我们通过重用相同的代码但依赖`cuML`获得了相同的结果。然而，处理时间已从每个文件夹 102 秒降低到每个文件夹 4 秒。如果您计算时间节省，那将是 25 倍的速度提升。确切的表现效益取决于您使用的 GPU 型号；GPU 越强大，结果越快，因为这与 GPU 卡从 CPU 内存传输数据以及处理矩阵乘法的速度有关。

基于标准 GPU（公众可访问的通用 GPU）上的此类性能，我们最近看到了将表格数据与深度学习模型（如文本或图像）的大嵌入融合的应用。SVMs（支持向量机）与众多特征（但不超过示例中的数量）和稀疏值（许多零值）配合良好。在这种情况下，SVMs 可以轻松获得最先进的结果，超越当时其他更受欢迎的表格算法——即 XGBoost 和其他梯度提升实现，以及端到端深度学习解决方案，后者在没有足够案例提供时表现较弱。

拥有 GPU 并使您的代码适应使用`RAPIDS`算法，使得某些经典的表格机器学习算法再次具有竞争力，这通常基于机器学习中没有免费午餐的原则（关于没有免费午餐定理的更多细节可在[`www.no-free-lunch.org/`](http://www.no-free-lunch.org/)找到）。考虑到您的项目限制（例如，您可能无法在项目环境中获得某些资源），如果可行，永远不要排除在先验测试中将您的问题与所有可用算法进行比较。
