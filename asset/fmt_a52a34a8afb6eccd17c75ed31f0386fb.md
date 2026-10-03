
到目前为止，我们考虑了纯随机抽样方法。如果你的数据集足够大（尤其是相对于属性数量），这通常是可行的，但如果不这样，你可能会引入显著的抽样偏差。当调查公司的员工决定给 1000 人打电话，问他们一些问题时，他们不会在电话簿中随机挑选 1000 人。他们试图确保这 1000 人能代表整个群体，针对他们想要询问的问题。例如，根据美国人口普查局的数据，51.6%的投票年龄公民是女性，而 48.4%是男性，因此在美国进行的一项良好调查会试图在样本中保持这一比例：516 名女性和 484 名男性（至少如果答案可能因性别而异的话）。这被称为*分层抽样*：人口被划分为同质子群体，称为*层*，并从每个层中抽取适当数量的实例，以确保测试集能代表整体人口。如果进行调查的人使用纯随机抽样，那么抽取一个女性比例低于 49%或高于 54%的测试集的概率会超过 10%。无论如何，调查结果可能会非常偏颇。

假设你已经和一些专家交谈过，他们告诉你中位数收入是预测中位数房价的一个重要属性。你可能想确保测试集能代表整个数据集中各种收入类别。由于中位数收入是一个连续的数值属性，你首先需要创建一个收入类别属性。让我们更仔细地看看中位数收入直方图（回到图 2-8）：大多数中位数收入值集中在 1.5 到 6 之间（即，$15,000–$60,000），但有些中位数收入远远超过 6。在你的数据集中为每个层有足够数量的实例是很重要的，否则对层重要性的估计可能会偏颇。这意味着你不应该有太多的层，并且每个层应该足够大。以下代码使用`pd.cut()`函数创建了一个有五个类别（从 1 到 5 标记）的收入类别属性；类别 1 的范围是 0 到 1.5（即，低于$15,000），类别 2 从 1.5 到 3，依此类推：

```py
housing_full["income_cat"] = pd.cut(housing_full["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1, 2, 3, 4, 5])
```

这些收入类别在图 2-9 中表示：

```py
cat_counts = housing_full["income_cat"].value_counts().sort_index()
cat_counts.plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
plt.show()
```

现在你可以根据收入类别进行分层抽样了。Scikit-Learn 在`sklearn.model_selection`包中提供了一些分割器类，它们实现了将数据集分割为训练集和测试集的各种策略。每个分割器都有一个`split()`方法，它返回对相同数据的不同训练/测试分割的迭代器。

![展示五个收入类别中地区分布的条形图，其中第二和第三个类别的计数最高](img/hmls_0209.png)

###### 图 2-9. 收入类别直方图

为了更精确，`split()`方法返回的是训练和测试*索引*，而不是数据本身。如果你想要更好地估计模型的性能，多个分割可能会有用，正如我们在本章后面讨论交叉验证时将看到的。例如，以下代码生成了 10 个不同的分层分割的相同数据集：

```py
from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
for train_index, test_index in splitter.split(housing_full,
                                              housing_full["income_cat"]):
    strat_train_set_n = housing_full.iloc[train_index]
    strat_test_set_n = housing_full.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])
```

目前，你可以只使用第一个分割：

```py
strat_train_set, strat_test_set = strat_splits[0]
```

或者，由于分层抽样相当常见，你可以使用带有`stratify`参数的`train_test_split()`函数来通过更短的方式获得单个分割：

```py
strat_train_set, strat_test_set = train_test_split(
    housing_full, test_size=0.2, stratify=housing_full["income_cat"],
    random_state=42)
```

让我们看看这是否如预期那样工作。你可以从查看测试集中的收入类别比例开始：

```py
>>> strat_test_set["income_cat"].value_counts() / len(strat_test_set)
income_cat
3    0.350533
2    0.318798
4    0.176357
5    0.114341
1    0.039971
Name: count, dtype: float64
```

使用类似的代码，你可以测量整个数据集中收入类别的比例。图 2-10 比较了整体数据集、使用分层抽样生成的测试集以及使用纯随机抽样生成的测试集中的收入类别比例。如图所示，使用分层抽样生成的测试集的收入类别比例几乎与整个数据集中的比例相同，而使用纯随机抽样生成的测试集则存在偏差。

![比较整体数据集、分层抽样和随机抽样中收入类别比例的表格，突出分层抽样中的低误差](img/hmls_0210.png)

###### 图 2-10. 分层抽样与纯随机抽样的采样偏差比较

你将不再使用`income_cat`列，所以你可以将其删除，将数据恢复到其原始状态：

```py
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
```

我们在测试集生成上花费了大量时间，这是有充分理由的：这是机器学习项目中经常被忽视但至关重要的一个部分。此外，许多这些想法将在我们后面讨论交叉验证时变得有用。现在，是时候进入下一阶段：探索数据。

# 探索和可视化数据以获得洞察

到目前为止，你只是快速浏览了数据，以获得对所处理数据类型的一般了解。现在目标是深入一些。

首先，确保你已经将测试集放在一边，并且你只探索训练集。此外，如果训练集非常大，你可能想要采样一个探索集，以便在探索阶段轻松快速地进行操作。在这种情况下，训练集相当小，所以你可以直接在完整集上工作。由于你将要对整个训练集的各种转换进行实验，你应该制作一个原始副本，以便之后可以恢复：

```py
housing = strat_train_set.copy()
```

## 可视化地理数据

由于数据集包括地理信息（纬度和经度），创建所有地区的散点图以可视化数据是个好主意（图 2-11）：

```py
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True)
plt.show()
```

![散点图显示地理数据点，经度在 x 轴上，纬度在 y 轴上，分布类似于加利福尼亚](img/hmls_0211.png)

###### 图 2-11. 数据的地理散点图

这看起来确实像加利福尼亚，但除此之外，很难看出任何特定的模式。将`alpha`选项设置为`0.2`使得可视化数据点密集的地方变得容易多了（图 2-12）：

```py
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
plt.show()
```

现在好多了：你可以清楚地看到高密度区域，即旧金山湾区以及洛杉矶和圣地亚哥周围，以及中央谷地中一条相当高密度区域的漫长线条（特别是在萨克拉门托和弗雷斯诺周围）。

我们的大脑在图片中寻找模式方面非常擅长，但你可能需要调整可视化参数以使模式更加突出。

![散点图显示按纬度和经度分布的住房位置，用较深的蓝色簇表示高密度区域](img/hmls_0212.png)

###### 图 2-12. 更好的可视化，突出显示高密度区域

接下来，你查看房价（图 2-13）。每个圆的半径代表该地区的总人口（选项`s`），颜色代表价格（选项`c`）。在这里，你使用一个预定义的颜色映射（选项`cmap`）称为`jet`，范围从蓝色（低值）到红色（高价）：⁠^(8)

```py
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
             s=housing["population"] / 100, label="population",
             c="median_house_value", cmap="jet", colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))
plt.show()
```

这张图片告诉你，房价与位置（例如，靠近海洋）和人口密度有很大关系，你可能已经知道了。聚类算法应该有助于检测主要簇并添加测量簇中心接近度的特征。海洋接近度属性也可能很有用，尽管在加利福尼亚北部，沿海地区的房价并不太高，因此这不是一个简单的规则。

![散点图显示加利福尼亚房价按位置分布，红色表示昂贵地区，蓝色表示较便宜的地区；较大的圆圈代表人口密度较大的地区](img/hmls_0213.png)

###### 图 2-13. 加利福尼亚房价：红色表示昂贵，蓝色表示便宜，较大的圆圈表示人口较多的地区

## 寻找相关性

由于数据集不是很大，你可以轻松地使用`corr()`方法计算每对数值属性之间的*标准相关系数*（也称为*皮尔逊相关系数*）：

```py
corr_matrix = housing.corr(numeric_only=True)
```

现在，你可以查看每个属性与中值房价的相关程度：

```py
>>> corr_matrix["median_house_value"].sort_values(ascending=False)
median_house_value    1.000000
median_income         0.688380
total_rooms           0.137455
housing_median_age    0.102175
households            0.071426
total_bedrooms        0.054635
population           -0.020153
longitude            -0.050859
latitude             -0.139584
Name: median_house_value, dtype: float64
```

相关系数范围从-1 到 1。当它接近 1 时，表示存在强烈的正相关；例如，当中等收入上升时，中等房价往往会上升。当系数接近-1 时，表示存在强烈的负相关；你可以看到纬度与中等房价之间存在轻微的负相关（即，价格有轻微的下降趋势，当你向北走时）。最后，接近 0 的系数表示没有线性相关性。

检查属性之间的相关性还有另一种方法，即使用 Pandas 的`scatter_matrix()`函数，该函数将每个数值属性与每个其他数值属性进行绘图。由于现在有 9 个数值属性，你会得到 81 个绘图，这不会适合在一页纸上显示——因此你决定专注于几个似乎与中等住房价值最相关的有希望的属性（图 2-14）：

```py
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()
```

![散点矩阵显示中等房价、中等收入、总房间数和住房中位数年龄的双变量比较，对角线上有直方图](img/hmls_0214.png)

###### 图 2-14. 这个散点矩阵将每个数值属性与每个其他数值属性进行对比，同时在主对角线上（从左上角到底右角）显示每个数值属性的值直方图

如果 Pandas 将每个变量与其自身进行绘图，主对角线将充满直线，这不会很有用。因此，Pandas 显示每个属性的直方图（其他选项也可用；有关更多详细信息，请参阅 Pandas 文档）。

观察相关性散点图，似乎预测中等房价最有希望的属性是中等收入，因此你放大了该散点图（图 2-15）：

```py
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1, grid=True)
plt.show()
```

![散点图显示中等收入与中等房价之间的相关性，突出显示上升趋势和特定值处的可见价格上限](img/hmls_0215.png)

###### 图 2-15. 中等收入与中等房价对比

此图揭示了一些事情。首先，相关性确实相当强；你可以清楚地看到上升趋势，尽管数据有噪声。其次，你之前注意到的价格上限在$500,000 处清楚地显示为一条水平线。但该图还揭示了其他不太明显的直线：大约在$450,000 处的一条水平线，另一条大约在$350,000 处，可能还有一条大约在$280,000 处，以及一些更低的。你可能想要尝试删除相应的区域，以防止你的算法学习到这些数据异常。

注意，相关系数仅衡量线性相关性（“当*x*增加时，*y*通常增加/减少”）。它可能完全忽略非线性关系（例如，“当*x*接近 0 时，*y*通常增加”）。图 2-16 显示了各种数据集及其相关系数。注意，底部所有图表的相关系数都等于 0，尽管它们的轴显然*不是*独立的：这些都是非线性关系的例子。此外，第二行显示了相关系数等于 1 或-1 的例子；请注意，这与斜率无关。例如，你的英寸身高与你的英尺身高或纳米身高有 1 的相关系数。

![各种数据集的散点图，显示相关系数在某些情况下可以指示强烈的线性关系，但无法捕捉非线性关系，系数范围从-1 到 1。](img/hmls_0216.png)

###### 图 2-16. 各种数据集的标准相关系数（来源：维基百科；公有领域图片）

## 尝试属性组合

希望前几节能给你一些启发，让你了解一些探索数据并获取洞察的方法。你识别出了一些可能需要清理的数据异常，在将数据输入机器学习算法之前，你发现了属性之间的有趣相关性，特别是与目标属性的相关性。你还注意到一些属性具有偏右分布，因此你可能想要对它们进行转换（例如，通过计算它们的对数或平方根）。当然，每个项目的里程数可能会有很大差异，但总体思路是相似的。

在准备数据用于机器学习算法之前，你可能还想尝试各种属性组合。例如，如果你不知道该地区有多少家庭，那么该地区总房间数并不是很有用。你真正想要的是每户家庭的房间数。同样，仅凭卧室总数本身也不是很有用：你可能想将其与总房间数进行比较。而且，每户家庭的人口数也是一个有趣的属性组合。你可以按照以下方式创建这些新属性：

```py
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]
```

然后你再次查看相关矩阵：

```py
>>> corr_matrix = housing.corr(numeric_only=True)
>>> corr_matrix["median_house_value"].sort_values(ascending=False)
median_house_value    1.000000
median_income         0.688380
rooms_per_house       0.143663
total_rooms           0.137455
housing_median_age    0.102175
households            0.071426
total_bedrooms        0.054635
population           -0.020153
people_per_house     -0.038224
longitude            -0.050859
latitude             -0.139584
bedrooms_ratio       -0.256397
Name: median_house_value, dtype: float64
```

嘿，不错！新的`bedrooms_ratio`属性与总房间数或卧室总数相比，与中位数房价的相关性要强得多。这是一个强烈的负相关，所以看起来卧室/房间比例较低的房屋往往更贵。每户家庭的房间数也比该地区总房间数更有信息量——显然，房屋越大，价格越高。

###### 警告
