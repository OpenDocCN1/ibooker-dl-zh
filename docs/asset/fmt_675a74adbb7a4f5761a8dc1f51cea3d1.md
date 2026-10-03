
## 聚类相似性特征

```py
>>> similarities[:3].round(2)
array([[0\.  , 0.14, 0\.  , 0\.  , 0\.  , 0.08, 0\.  , 0.99, 0\.  , 0.6 ],
 [0.63, 0\.  , 0.99, 0\.  , 0\.  , 0\.  , 0.04, 0\.  , 0.11, 0\.  ],
 [0\.  , 0.29, 0\.  , 0\.  , 0.01, 0.44, 0\.  , 0.7 , 0\.  , 0.3 ]])
```

图 2-19 显示了 *k*-means 找到的 10 个聚类中心。根据它们与最近聚类中心的地理相似性，地区被着色。如您所见，大多数聚类位于人口稠密和昂贵的地区。

![mls3 0219](img/mls3_0219.png)

###### 图 2-19. 高斯 RBF 相似度到最近的聚类中心

## 转换管道

如您所见，有许多数据转换步骤需要按正确顺序执行。幸运的是，Scikit-Learn 提供了 `Pipeline` 类来帮助处理这样的转换序列。这是一个用于数值属性的小管道，它将首先填充然后缩放输入特征：

```py
from sklearn.pipeline import Pipeline

num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler()),
])
```

`Pipeline` 构造函数接受一系列步骤定义的名称/估计器对（2 元组）列表。名称可以是任何您喜欢的内容，只要它们是唯一的且不包含双下划线（`__`）。稍后在我们讨论超参数调整时，它们将很有用。所有估计器必须是转换器（即，它们必须具有 `fit_transform()` 方法），除了最后一个，它可以是任何东西：一个转换器，一个预测器或任何其他类型的估计器。

###### 提示

在 Jupyter 笔记本中，如果 `import sklearn` 并运行 `sklearn.set_config(display="diagram")`，则所有 Scikit-Learn 估计器都将呈现为交互式图表。这对于可视化管道特别有用。要可视化 `num_pipeline`，请在最后一行中运行一个包含 `num_pipeline` 的单元格。单击估计器将显示更多详细信息。

如果您不想为转换器命名，可以使用 `make_pipeline()` 函数；它将转换器作为位置参数，并使用转换器类的名称创建一个 `Pipeline`，名称为小写且没有下划线（例如，`"simpleimputer"`）：

```py
from sklearn.pipeline import make_pipeline

num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
```

如果多个转换器具有相同的名称，将在它们的名称后附加索引（例如，`"foo-1"`，`"foo-2"`等）。

当您调用管道的 `fit()` 方法时，它会按顺序在所有转换器上调用 `fit_transform()`，将每次调用的输出作为下一次调用的参数，直到达到最终的估计器，对于最终的估计器，它只调用 `fit()` 方法。

管道公开与最终估计器相同的方法。在这个例子中，最后一个估计器是 `StandardScaler`，它是一个转换器，因此管道也像一个转换器。如果调用管道的 `transform()` 方法，它将顺序应用所有转换到数据。如果最后一个估计器是预测器而不是转换器，则管道将具有 `predict()` 方法而不是 `transform()` 方法。调用它将顺序应用所有转换到数据，并将结果传递给预测器的 `predict()` 方法。

让我们调用管道的 `fit_transform()` 方法，并查看输出的前两行，保留两位小数：

```py
>>> housing_num_prepared = num_pipeline.fit_transform(housing_num)
>>> housing_num_prepared[:2].round(2)
array([[-1.42,  1.01,  1.86,  0.31,  1.37,  0.14,  1.39, -0.94],
 [ 0.6 , -0.7 ,  0.91, -0.31, -0.44, -0.69, -0.37,  1.17]])
```

如您之前所见，如果要恢复一个漂亮的 DataFrame，可以使用管道的 `get_feature_names_out()` 方法：

```py
df_housing_num_prepared = pd.DataFrame(
    housing_num_prepared, columns=num_pipeline.get_feature_names_out(),
    index=housing_num.index)
```

管道支持索引；例如，`pipeline[1]` 返回管道中的第二个估计器，`pipeline[:-1]` 返回一个包含除最后一个估计器之外的所有估计器的 `Pipeline` 对象。您还可以通过 `steps` 属性访问估计器，该属性是名称/估计器对的列表，或者通过 `named_steps` 字典属性访问估计器，该属性将名称映射到估计器。例如，`num_pipeline["simpleimputer"]` 返回名为 `"simpleimputer"` 的估计器。

到目前为止，我们已经分别处理了分类列和数值列。有一个单一的转换器可以处理所有列，对每一列应用适当的转换会更方便。为此，您可以使用 `ColumnTransformer`。例如，以下 `ColumnTransformer` 将 `num_pipeline`（我们刚刚定义的）应用于数值属性，将 `cat_pipeline` 应用于分类属性：

```py
from sklearn.compose import ColumnTransformer

num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
               "total_bedrooms", "population", "households", "median_income"]
cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])
```

首先我们导入 `ColumnTransformer` 类，然后定义数值和分类列名的列表，并为分类属性构建一个简单的管道。最后，我们构建一个 `ColumnTransformer`。它的构造函数需要一个三元组（3 元组）的列表，每个三元组包含一个名称（必须是唯一的，不包含双下划线）、一个转换器和一个应用转换器的列名（或索引）列表。

###### 提示

如果你想要删除列，可以指定字符串 `"drop"`，如果你想要保留列不变，可以指定 `"passthrough"`。默认情况下，剩余的列（即未列出的列）将被删除，但是如果你想要这些列被处理方式不同，可以将 `remainder` 超参数设置为任何转换器（或 `"passthrough"`）。

由于列名的列举并不是很方便，Scikit-Learn 提供了一个 `make_column_selector()` 函数，返回一个选择器函数，你可以用它自动选择给定类型的所有特征，比如数值或分类。你可以将这个选择器函数传递给 `ColumnTransformer`，而不是列名或索引。此外，如果你不关心命名转换器，你可以使用 `make_column_transformer()`，它会为你选择名称，就像 `make_pipeline()` 一样。例如，以下代码创建了与之前相同的 `ColumnTransformer`，只是转换器自动命名为 `"pipeline-1"` 和 `"pipeline-2"`，而不是 `"num"` 和 `"cat"`：

```py
from sklearn.compose import make_column_selector, make_column_transformer

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),
)
```

现在我们准备将这个 `ColumnTransformer` 应用到房屋数据中：

```py
housing_prepared = preprocessing.fit_transform(housing)
```

太棒了！我们有一个预处理管道，它接受整个训练数据集，并将每个转换器应用于适当的列，然后水平连接转换后的列（转换器绝不能改变行数）。再次返回一个 NumPy 数组，但你可以使用 `preprocessing.get_feature_names_out()` 获取列名，并像之前一样将数据包装在一个漂亮的 DataFrame 中。

###### 注意

`OneHotEncoder` 返回一个稀疏矩阵，而 `num_pipeline` 返回一个密集矩阵。当存在稀疏和密集矩阵混合时，`ColumnTransformer` 会估计最终矩阵的密度（即非零单元格的比例），如果密度低于给定阈值（默认为 `sparse_threshold=0.3`），则返回稀疏矩阵。在这个例子中，返回一个密集矩阵。

你的项目进展得很顺利，你几乎可以开始训练一些模型了！现在你想创建一个单一的管道，执行到目前为止你已经尝试过的所有转换。让我们回顾一下管道将做什么以及为什么：

+   数值特征中的缺失值将被中位数替换，因为大多数 ML 算法不希望有缺失值。在分类特征中，缺失值将被最频繁的类别替换。

+   分类特征将被独热编码，因为大多数 ML 算法只接受数值输入。

+   将计算并添加一些比率特征：`bedrooms_ratio`、`rooms_per_house` 和 `people_per_house`。希望这些特征与房屋价值中位数更好地相关，并帮助 ML 模型。

+   还将添加一些集群相似性特征。这些特征可能对模型比纬度和经度更有用。

+   具有长尾的特征将被其对数替换，因为大多数模型更喜欢具有大致均匀或高斯分布的特征。

+   所有数值特征将被标准化，因为大多数 ML 算法更喜欢所有特征具有大致相同的尺度。

构建执行所有这些操作的管道的代码现在应该对你来说很熟悉：

```py
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())
preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)  # one column remaining: housing_median_age
```

如果你运行这个 `ColumnTransformer`，它将执行所有转换并输出一个具有 24 个特征的 NumPy 数组：

```py
>>> housing_prepared = preprocessing.fit_transform(housing)
>>> housing_prepared.shape
(16512, 24)
>>> preprocessing.get_feature_names_out()
array(['bedrooms__ratio', 'rooms_per_house__ratio',
 'people_per_house__ratio', 'log__total_bedrooms',
 'log__total_rooms', 'log__population', 'log__households',
 'log__median_income', 'geo__Cluster 0 similarity', [...],
 'geo__Cluster 9 similarity', 'cat__ocean_proximity_<1H OCEAN',
 'cat__ocean_proximity_INLAND', 'cat__ocean_proximity_ISLAND',
 'cat__ocean_proximity_NEAR BAY', 'cat__ocean_proximity_NEAR OCEAN',
 'remainder__housing_median_age'], dtype=object)
```

# 选择并训练模型

终于！你确定了问题，获取了数据并对其进行了探索，对训练集和测试集进行了抽样，并编写了一个预处理管道来自动清理和准备数据以供机器学习算法使用。现在你准备好选择和训练一个机器学习模型了。

## 在训练集上训练和评估

好消息是，由于之前的所有步骤，现在事情将变得容易！你决定训练一个非常基本的线性回归模型来开始：

```py
from sklearn.linear_model import LinearRegression

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)
```

完成了！你现在有一个可用的线性回归模型。你可以在训练集上尝试一下，查看前五个预测值，并将其与标签进行比较：

```py
>>> housing_predictions = lin_reg.predict(housing)
>>> housing_predictions[:5].round(-2)  # -2 = rounded to the nearest hundred
array([243700., 372400., 128800.,  94400., 328300.])
>>> housing_labels.iloc[:5].values
array([458300., 483800., 101700.,  96100., 361800.])
```

好吧，它起作用了，但并不总是：第一个预测结果相差太远（超过 20 万美元！），而其他预测结果更好：两个相差约 25%，两个相差不到 10%。记住你选择使用 RMSE 作为性能指标，所以你想使用 Scikit-Learn 的 `mean_squared_error()` 函数在整个训练集上测量这个回归模型的 RMSE，将 `squared` 参数设置为 `False`：

```py
>>> from sklearn.metrics import mean_squared_error
>>> lin_rmse = mean_squared_error(housing_labels, housing_predictions,
...                               squared=False)
...
>>> lin_rmse
68687.89176589991
```
