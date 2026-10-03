
```py
from sklearn.cluster import KMeans

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]
```

## 小贴士

您可以通过将实例传递给`sklearn.utils.estimator_checks`包中的`check_estimator()`来检查您的自定义估计器是否遵守 Scikit-Learn 的 API。有关完整 API，请访问[*https://scikit-learn.org/stable/developers*](https://scikit-learn.org/stable/developers)。

正如您将在第八章中看到的那样，*k*-means 是一种聚类算法，它可以在数据中定位聚类。例如，我们可以用它来找到加利福尼亚州人口最多的地区。*k*-means 搜索的聚类数量由`n_clusters`超参数控制。`KMeans`的`fit()`方法支持一个可选的`sample_weight`参数，允许用户指定样本的相对权重。例如，如果我们想使聚类偏向富裕地区，我们可以传递中位数收入。训练后，聚类中心可通过`cluster_centers_`属性访问。*k*-means 是一个随机算法，这意味着它依赖于随机性来定位聚类，因此如果您想得到可重复的结果，您必须设置`random_state`参数。如您所见，尽管任务复杂，但代码相当简单。现在让我们使用这个自定义转换器：

```py
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]])
```

此代码创建了一个`ClusterSimilarity`转换器，将簇的数量设置为 10。然后它使用训练集中每个地区的纬度和经度调用`fit_transform()`（您可以尝试通过每个地区的中位数收入来加权每个地区，看看这如何影响簇）。转换器使用 *k*-means 定位簇，然后测量每个地区与所有 10 个簇中心的 Gaussian RBF 相似性。结果是一个矩阵，每行代表一个地区，每列代表一个簇。让我们看看前三个行，四舍五入到两位小数：

```py
>>> similarities[:3].round(2)
array([[0.46, 0\.  , 0.08, 0\.  , 0\.  , 0\.  , 0\.  , 0.98, 0\.  , 0\.  ],
 [0\.  , 0.96, 0\.  , 0.03, 0.04, 0\.  , 0\.  , 0\.  , 0.11, 0.35],
 [0.34, 0\.  , 0.45, 0\.  , 0\.  , 0\.  , 0.01, 0.73, 0\.  , 0\.  ]])
```

图 2-19 显示了由 *k*-means 找到的 10 个簇中心。地区根据其与最近簇中心的地理相似性着色。请注意，大多数簇都位于高人口密度区域。

![基于高斯 RBF 相似性的地理聚类散点图，高人口密度区域突出显示](img/hmls_0219.png)

### 图 2-19\. 到最近簇中心的 Gaussian RBF 相似性

## 转换器管道

如您所见，有许多需要按正确顺序执行的数据转换步骤。幸运的是，Scikit-Learn 提供了`Pipeline`类来帮助处理这样的转换序列。以下是一个用于数值属性的简单管道示例，它将首先填充缺失值然后缩放输入特征：

```py
from sklearn.pipeline import Pipeline

num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler()),
])
```

`Pipeline`构造函数接受一个定义步骤序列的名称/估计器对（2-元组）列表。名称可以是任何您喜欢的，只要它们是唯一的并且不包含双下划线（`__`）。它们将在我们讨论超参数调整时很有用。估计器必须全部是转换器（即它们必须有`fit_transform()`方法），除了最后一个，可以是任何东西：一个转换器、一个预测器或任何其他类型的估计器。

### 小贴士

在 Jupyter 笔记本中，如果你`import` `sklearn`并运行`sklearn.​set_config(display="diagram")`，所有 Scikit-Learn 估计器都将被渲染为交互式图表。这对于可视化管道特别有用。要可视化`num_pipeline`，请运行一个以`num_pipeline`为最后一行的单元格。点击估计器将显示更多详细信息。

如果你不想命名转换器，可以使用方便的`make_pipeline()`函数代替；它接受转换器作为位置参数，并使用转换器类的名称创建一个`Pipeline`，名称为小写且不带下划线（例如，`"simpleimputer"`）：

```py
from sklearn.pipeline import make_pipeline

num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
```

如果多个转换器具有相同的名称，则会在其名称后附加一个索引（例如，`"foo-1"`，`"foo-2"`等）。

当你调用管道的`fit()`方法时，它将对所有转换器依次调用`fit_transform()`，将每个调用的输出作为下一个调用的参数，直到它达到最终的估计器，对于这个估计器，它只调用`fit()`方法。

管道公开了与最终估计器相同的方法。在这个例子中，最后一个估计器是一个`StandardScaler`，它是一个转换器，因此管道也像转换器一样工作。如果你调用管道的`transform()`方法，它将按顺序应用所有转换到数据上。如果最后一个估计器是一个预测器而不是转换器，那么管道将有一个`predict()`方法而不是`transform()`方法。调用它将按顺序应用所有转换到数据上，并将结果传递给预测器的`predict()`方法。

让我们调用管道的`fit_transform()`方法，并查看输出前两行，结果四舍五入到小数点后两位：

```py
>>> housing_num_prepared = num_pipeline.fit_transform(housing_num)
>>> housing_num_prepared[:2].round(2)
array([[-1.42,  1.01,  1.86,  0.31,  1.37,  0.14,  1.39, -0.94],
 [ 0.6 , -0.7 ,  0.91, -0.31, -0.44, -0.69, -0.37,  1.17]])
```

如你之前所见，如果你想恢复一个漂亮的 DataFrame，你可以使用管道的`get_feature_names_out()`方法：

```py
df_housing_num_prepared = pd.DataFrame(
    housing_num_prepared, columns=num_pipeline.get_feature_names_out(),
    index=housing_num.index)
```

管道支持索引；例如，`pipeline[1]`返回管道中的第二个估计器，而`pipeline[:-1]`返回一个不包含最后一个估计器的`Pipeline`对象。你也可以通过`steps`属性访问估计器，它是一个包含名称/估计器对的列表，或者通过`named_steps`字典属性访问，它将名称映射到估计器。例如，`num_pipeline["simpleimputer"]`返回名为`"simpleimputer"`的估计器。

到目前为止，我们已经分别处理了分类列和数值列。如果有一个能够处理所有列的单个转换器，并且对每个列应用适当的转换，将会更方便。为此，你可以使用`ColumnTransformer`。例如，以下`ColumnTransformer`将应用我们刚刚定义的`num_pipeline`到数值属性，并将`cat_pipeline`应用到分类属性：

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

首先，我们导入`ColumnTransformer`类，然后定义数值和分类列名的列表，并为分类属性构建一个简单的管道。最后，我们构建一个`ColumnTransformer`。其构造函数需要一个包含三元组（3-tuples）的列表，每个三元组包含一个名称（必须是唯一的，且不能包含双下划线）、一个转换器和应用于转换器的列名（或索引）列表。

### 小贴士

如果你不想使用转换器，你可以指定字符串`"drop"`来删除列，或者如果你想保持列不变，可以指定`"passthrough"`。默认情况下，剩余的列（即未列出的列）将被删除，但你可以将`remainder`超参数设置为任何转换器（或设置为`"passthrough"`)，如果你希望这些列以不同的方式处理。

由于列出所有列名不太方便，Scikit-Learn 提供了一个`make_column_selector`类，你可以使用它来自动选择给定类型的所有特征，例如数值或分类。你可以将选择器传递给`ColumnTransformer`而不是列名或索引。此外，如果你不关心命名转换器，你可以使用`make_column_transformer()`，它将为你选择名称，就像`make_pipeline()`做的那样。例如，以下代码创建了一个与之前相同的`ColumnTransformer`，但转换器的名称自动命名为`"pipeline-1"`和`"pipeline-2"`而不是`"num"`和`"cat"`：

```py
from sklearn.compose import make_column_selector, make_column_transformer

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),
)
```

现在我们已经准备好将这个`ColumnTransformer`应用到住房数据上了：

```py
housing_prepared = preprocessing.fit_transform(housing)
```

太好了！我们有一个预处理管道，它接受整个训练数据集，并将每个转换器应用于适当的列，然后水平地连接转换后的列（转换器绝不能改变行数）。再次强调，这返回一个 NumPy 数组，但你可以使用`preprocessing.get_feature_names_out()`来获取列名，并将数据包裹在一个漂亮的 DataFrame 中，就像我们之前做的那样。

### 注意

`OneHotEncoder`返回一个稀疏矩阵，而`num_pipeline`返回一个密集矩阵。当存在这种稀疏和密集矩阵的混合时，`ColumnTransformer`会估计最终矩阵的密度（即非零单元格的比例），如果密度低于给定的阈值（默认为`sparse_threshold=0.3`），则返回一个稀疏矩阵。在这个例子中，它返回一个密集矩阵。

你的项目进展得非常顺利，你几乎准备好训练一些模型了！你现在想要创建一个单一的管道，该管道将执行你迄今为止实验过的所有转换。让我们回顾一下管道将做什么以及为什么：

+   数值特征中的缺失值将通过用中位数替换它们来填充，因为大多数机器学习算法不期望存在缺失值。在分类特征中，缺失值将被最频繁出现的类别所替换。

+   分类特征将被进行独热编码，因为大多数机器学习算法只接受数值输入。

+   将计算并添加一些比率特征：`bedrooms_ratio`、`rooms_per_house`和`people_per_house`。希望这些特征能更好地与中位数房价相关联，从而帮助机器学习模型。

+   将添加一些聚类相似度特征。这些特征可能比经纬度对模型更有用。

+   尾部较长的特征将被它们的对数所替代，因为大多数模型更喜欢具有大致均匀或高斯分布的特征。

+   所有数值特征都将进行标准化，因为大多数机器学习算法更喜欢所有特征具有大致相同的尺度。

构建执行所有这些操作的管道的代码现在应该对你来说很熟悉了：

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
