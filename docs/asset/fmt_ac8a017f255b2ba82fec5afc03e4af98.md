
如果系数的值为`+1`，则意味着该观察点与相邻聚类之间存在距离。

因此，我们期望获得系数的最高值以获得良好的聚类解决方案。

1. **Dunn 指数**也可用于衡量聚类的效果。它使用了点 2 和点 3 中定义的聚类间距离和聚类内距离测量，并由下面的方程（方程 2.6）给出

##### （方程 2.6）

```
Dunn 指数 = min（聚类间距离）/max（聚类内距离）
```

显然，我们会努力最大化 Dunn 指数的值。为了实现这一点，分子应尽可能大，意味着聚类之间相距较远，而分母应尽可能低，表明聚类非常健壮且紧密排列。

现在我们已经检验了衡量算法性能的方法。我们现在将转向寻找 k-means 聚类的最佳“k”值。

### 2.3.3 寻找最佳的“k”值

选择最优的聚类数量并不容易。正如我们之前所说，最好的聚类是当聚类数量等于观察数量时——但是正如我们在上一节中所学到的，这在实际中是不可能的。但是我们必须将聚类数量“k”作为算法的输入提供。

##### 图 2.10 肘方法寻找最优聚类数量。红色圆圈表示转折点。但最终的聚类数量取决于业务逻辑，通常根据业务知识合并/拆分聚类。维护聚类的便利性在其中也扮演了至关重要的角色

![02_08](img/02_08.png)

或许找到“k”的最优值最广泛使用的方法是*Elbow Method*。在这种方法中，我们计算不同“k”值的簇内平方和或 WCSS。该过程与上一节中讨论的相同。然后，将 WCSS 绘制在图表上，针对不同的“k”值。在我们观察到一个弯曲或肘部的地方，如（图 2.10）所示，它就是数据集的最优簇数。注意（图 2.10）中描绘的锋利边缘。

![](img/tgt.png) 测验 - 回答这些问题来检查你的理解。书的结尾有答案。

1. K-means 聚类不需要作为输入的簇数- 真 或 假
2. Knn 和 k-means 聚类是一回事 - 真 或 假
3. 描述一种可能的找到最优“k”值的过程

但这并不意味着这是我们建议用于业务问题的最终簇数。根据落入每个簇的观测数量，几个簇可能被合并或分解成子簇。我们还考虑创建簇所需的计算成本。簇的数量越多，计算成本和所需时间就越大。

我们也可以使用前面讨论过的 Silhouette Coefficient 找到最优的簇数。

##### 注意：

探讨合并几个簇或拆分几个簇的业务逻辑是至关重要的。最终，解决方案必须在实际的业务场景中实施。

通过这个，我们已经研究了 k-means 聚类的点点滴滴 - 数学概念和过程，各种距离度量以及确定最佳 k 值。现在我们将研究 k-means 算法为我们提供的优势。

### 2.3.4 k-means 聚类的优缺点

k-means 算法是一个非常流行和广泛实施的聚类解决方案。该解决方案提供以下优势：

+ 与其他算法相比，它简单易懂且相对容易实现。距离测量计算使得即使是非统计背景的用户也很容易理解。
  + 如果维度的数量很大，k-means 算法比其他聚类算法更快，创建的簇更紧凑。因此，如果维度的数量相当大，则更倾向于使用它。
  + 它很快适应新的观察结果，并且能够非常好地概括各种形状和大小的簇。
+ 解决方案通过一系列重新计算的迭代产生结果。大多数情况下使用欧氏距离度量，这使得计算成本较低。它还确保算法一定会收敛并产生结果。

K-means 在现实生活中的业务问题中被广泛使用。尽管 k-means 聚类有明显的优点，但我们确实面临着算法的某些挑战：

+ 选择最佳聚类数目并不容易。我们必须将其作为输入提供。使用不同的“k”值，结果会完全不同。选择最佳“k”值的过程在上一节中已经探讨过。
  + 解决方案取决于质心的初始值。由于质心是随机初始化的，因此每次迭代的输出都将不同。因此，建议运行多个解决方案的版本并选择最佳的一个。
  + 该算法对异常值非常敏感。它们可能会破坏最终结果，因此在开始聚类之前，我们必须处理异常值。我们还可以实现 k 均值算法的其他变体，如*k-modes*聚类，以应对异常值的问题。我们将在后续章节讨论处理异常值的方法。
+ 由于 k 均值聚类的基本原理是计算距离，因此该解决方案不直接适用于分类变量。换句话说，我们不能直接使用分类变量，因为我们可以计算数字值之间的距离，但不能对分类变量进行数学计算。为了解决这个问题，我们可以使用独热编码将分类变量转换为数字变量，这是我们在本章末尾讨论的内容之一。

尽管存在这些问题，k 均值聚类是最常用的聚类解决方案之一，因其简单性和易于实现。还有一些不同版本的 k 均值算法，如 k-medoids、k-中位数等，有时用于解决所面临的问题。

1. 如其名，**k-中位数聚类**是基于数据集中位数而不是 k 均值中心的。这增加了计算时间，因为只有在数据排序之后才能找到中位数。但与此同时，k 均值对异常值很敏感，而 k 中位数对它们的影响较小。
2. 接下来，我们有**k-medoids 聚类**作为 k 均值算法的变体之一。Medoids 与均值类似，只是它们始终来自同一数据集，并且在难以获得均值的情况下实施，比如图像。Medoid 可以被认为是簇中最核心的点，与簇中的所有其他成员最不相似。K-medoids 选择实际观测值作为中心，而不是 k 均值，其中质心甚至可能不是数据的一部分。与 k 均值聚类算法相比，它对异常值的敏感性较低。

还有其他版本，如 kmeans++、小批量 k 均值等。一般来说，在工业界，大多数聚类解决方案都使用 k 均值。如果结果不理想或计算时间太长，您可以探索其他选项，如 kmeans++、小批量 k 均值等。此外，使用不同的距离测量指标可能会为 k 均值算法产生不同的结果。

本节结束了我们对 k-means 聚类算法的讨论。是时候进入实验室，开发真正的 Python 代码了！

### 2.3.5 使用 Python 实现 k-means 聚类

我们现在将为 k-means 聚类创建一个 Python 解决方案。在这种情况下，我们使用了链接中的数据集

`github.com/vverdhan/UnsupervisedLearningWithPython/tree/main/Chapter2`

这个数据集包含了四种车型的特征信息。基于车辆的特征，我们将把它们分成不同的群。

**第 1 步：** 将库和数据集导入到一个数据框中。在这里，`vehicles.csv`是输入数据文件。如果数据文件不在与 Jupyter 笔记本相同的文件夹中，您需要提供文件的完整路径。`Dropna`用于删除可能存在的缺失值。

```py
import pandas as pd
vehicle_df = pd.read_csv('vehicle.csv').dropna()
```

**第 2 步：** 对数据进行一些初始检查，比如形状、信息、前五行、类别分布等。这是为了确保我们已经加载了完整的数据集，并且在加载数据集时没有损坏。`Shape`命令将给出数据中的行数和列数，`info`将描述所有变量及其类型，`head`将显示前 5 行。`value_counts`显示`class`变量的分布。换句话说，`value_counts`返回唯一值的计数。

```py
vehicle_df.shape
vehicle_df.info()
vehicle_df.head()
pd.value_counts(vehicle_df['class'])
```

**第 3 步：** 让我们为变量`“class”`生成两个图表。数据集中的汽车示例更多，而对于公共汽车和货车，数据是平衡的。我们使用 matplotlib 库来绘制这些图表。图表的输出如下所示。

```py
import matplotlib.pyplot as plt
%matplotlib inline
pd.value_counts(vehicle_df["class"]).plot(kind='bar')
pd.value_counts(vehicle_df['class']).hist(bins=300)
```

![02_08a](img/02_08a.png)

**第 4 步：** 现在我们将检查数据集中是否有任何缺失的数据点。我们的数据集中没有缺失的数据点，因为我们已经处理过了。

```py
vehicle_df.isna().sum()
```

##### 注意

我们将在后面的章节中讨论处理缺失值的方法，因为删除缺失值通常不是最好的方法。

**第 5 步：** 现在我们将对数据集进行标准化。对于聚类来说，标准化数据集是一个很好的实践。这是很重要的，因为不同的维度可能处于不同的尺度，如果某个维度的值自然上比其他维度的值要大得多，那么一个维度可能会在距离计算中占据主导地位。下面使用`zscore`和`StandardScaler()`函数来实现。请参考书的附录，了解`zscore`和`StandardScaler()`函数之间的区别。

```py
vehicle_df_1 = vehicle_df.drop('class', axis=1)
from scipy.stats import zscore
vehicle_df_1_z = vehicle_df_1.apply(zscore)
from sklearn.preprocessing import StandardScaler
import numpy as np
sc = StandardScaler()
X_standard = sc.fit_transform(vehicle_df_1)
```

**第 6 步：** 我们现在要快速查看数据集，生成一个散点图。该图显示了我们在上一步中创建的`X_standard`的所有数据点的分布。

```py
plt.scatter(X_standard[:,0], X_standard[:,1])
plt.show()
```

![02_08b](img/02_08b.png)

**步骤 7：** 我们现在将执行 k-means 聚类。首先，我们必须使用肘部法选择最佳聚类数。从 sklearn 库中，我们导入 KMeans。在一个 for 循环中，我们对聚类值从 1 到 10 进行迭代。换句话说，算法将创建 1、2、3、4 到 10 个聚类，然后生成结果供我们选择最优 k 值。

在下面的代码片段中，模型对象包含了在上一步生成的 X_standard 上适配的 KMeans 算法的输出。这里，欧几里得距离被用作距离度量。

```py
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
clusters=range(1,10)
meanDistortions=[]
for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(X_standard)
    prediction=model.predict(X_standard)
    meanDistortions.append(sum(np.min(cdist(X_standard, model.cluster_centers_, 'euclidean'), axis=1)) / X_standard
                           .shape[0])
plt.plot(clusters, meanDistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Selecting k with the Elbow Method')
```

![02_08c](img/02_08c.png)

**步骤 8：** 正如我们所观察到的，最佳聚类数为 3\. 这是一个尖锐转折点，在图表中清晰可见。我们将使用 3 个聚类数进行 k-means 聚类。尽管这里的数字 3 没有特别之处，但它最适合这个数据集，我们也可以使用 4 或 5 个聚类数。`random_state` 是一个用于确定质心初始化的参数。我们将其设置为一个值以使随机性变得确定性。

```py
kmeans = KMeans(n_clusters=3, n_init = 15, random_state=2345) 
kmeans.fit(X_standard)
```

**步骤 9：** 获取聚类的质心

```py
centroids = kmeans.cluster_centers_
centroids
```

**步骤 10：** 现在我们将使用质心，以便它们可以按列进行分析。

```py
centroid_df = pd.DataFrame(centroids, columns = list(X_standard) )
```

**步骤 11：** 我们现在将创建一个仅用于创建标签的 `dataframe`，然后将其转换为分类变量。

```py
dataframe_labels = pd.DataFrame(kmeans.labels_ , columns = list(['labels']))
dataframe_labels['labels'] = dataframe_labels['labels'].astype('category')
```

**步骤 12：** 在这一步中，我们将两个 `dataframes` 进行连接

```py
dataframe_labeled = vehicle_df_1.join(dataframe_labels)
```

**步骤 13：** 执行 group by 操作以创建用于分析的数据框

```py
dataframe_analysis = (dataframe_labeled.groupby(['labels'] , axis=0)).head(1234)
dataframe_labeled['labels'].value_counts()  
```

**步骤 14：** 现在，我们将为我们定义的聚类创建可视化。这是使用 `mpl_toolkits` 库完成的。逻辑很容易理解。数据点根据相应的标签着色。其余的步骤与调整标签、标题、刻度等有关的细节。由于在绘图中不可能绘制所有的 18 个变量，我们选择了 3 个变量来显示在图中。

```py
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8, 6))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=20, azim=60)
kmeans.fit(vehicle_df_1_z)
labels = kmeans.labels_
ax.scatter(vehicle_df_1_z.iloc[:, 0], vehicle_df_1_z.iloc[:, 1], vehicle_df_1_z.iloc[:, 3],c=labels.astype(np.float), edgecolor='k')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Length')
ax.set_ylabel('Height')
ax.set_zlabel('Weight')
ax.set_title('3D plot of KMeans Clustering on vehicles dataset')
```

![02_08d](img/02_08d.png)

我们也可以用多个其他值对上述代码进行测试。我们已经创建了使用不同值的代码。出于空间考虑，我们将不同值的测试代码放在了 github 位置。

在上面的例子中，我们首先对数据集进行了小型的探索性分析。

##### 注意

探索性数据分析（EDA）是实现稳健机器学习解决方案和成功项目的关键。在随后的章节中，我们将为数据集创建详细的 EDA。

接下来是确定最佳聚类数，本例中为三。然后我们实施了 k-means 聚类。您应该迭代不同的初始化 k-means 解决方案并比较结果，迭代不同的 k 值，并可视化以分析数据点的移动。在本章后面，我们将使用 Python 创建层次聚类的相同数据集。

基于质心的聚类是最推荐的解决方案之一，因为其逻辑较少，易于实现，灵活性高且易于维护。每当我们需要聚类作为解决方案时，大多数情况下我们都会从创建一个作为基准的 k 均值聚类解决方案开始。该算法非常受欢迎，通常是聚类的首选解决方案之一。然后我们会测试并迭代其他算法。

这标志着对基于质心的聚类算法的讨论结束。我们现在将继续前进，讨论连接性解决方案，并在下一节中讨论层次聚类。

## 2.4 基于连接性的聚类

“物以类聚”是连接性聚类中遵循的原则。核心概念是-彼此连接的对象相似。因此，根据这些对象之间的连接性，它们被组合成簇。图 2.11 中展示了这种表示的一个示例，我们可以迭代地对观察进行分组。例如，我们从所有事物开始，分成生物和非生物等等。这种表示最好使用右侧的图表展示，称为*Dendrogram*。

##### 图 2.11 层次聚类利用迭代地对相似对象进行分组。右侧是聚类的可视化表示，称为树状图。

![02_09](img/02_09.png)

由于存在树状结构，连接性聚类有时被称为*层次*聚类。

层次聚类很好地符合人类直觉，因此我们容易理解它。与 k 均值聚类不同，在层次聚类中，我们不必输入最终聚类的数量，但该方法确实需要一个终止条件，即聚类应何时停止。同时，层次聚类不建议聚类的最佳数量。从生成的层次结构/树状图中，我们必须自己选择最佳的聚类数。在接下来的部分中，当我们为其创建 Python 代码时，我们将更多地探讨它。

层次聚类可以通过图 2.12 来理解，如下所示。这里第一个节点是根节点，然后迭代地分裂成节点和子节点。每当一个节点不能再进一步分裂时，它被称为终端节点或*叶子*。

##### 图 2.12 层次聚类具有一个根节点，分裂为节点和子节点。不能再进一步分裂的节点称为叶子。在自底向上的方法中，将合并叶子节点

![02_10](img/02_10.png)

由于将观察结果合并成簇的过程或逻辑不止一个，我们可以生成大量的树状图，如下所示的(Equation 2.7)：

##### (Equation 2.7)

```
树状图数量 = (2n-3)!/[2^((n-2)) (n-2)!]
```
