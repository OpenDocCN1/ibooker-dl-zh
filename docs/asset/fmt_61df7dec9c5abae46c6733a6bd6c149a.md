
当期望步骤完成后，我们将执行最大化步骤或 M 步骤。在这个步骤中，我们将使用方程 5.7 中的公式更新`µ`、`∑`和`f`的值。回想一下，在 k-means 聚类中，我们只是简单地取数据点的均值然后继续。这里我们做的是类似的事情，尽管我们使用了上一步计算的概率或期望。

这三个值可以使用以下方程计算。方程 5.7 是计算所有点的协方差`∑[j]`，然后按照方程 5.11 中所示，由该点由高斯`j`生成的概率进行加权。数学证明超出了本书的范围。

![figure](img/verdhan-ch5-eqs-11x.png)

##### (5.11)

均值`µ[j]`，由方程 5.12 确定。在这里，我们确定所有点的均值，并按照该点由高斯`j`生成的概率进行加权。

![figure](img/verdhan-ch5-eqs-12x.png)

##### (5.12)

同样，密度或强度是通过方程 5.13 计算的，其中我们将每个点由高斯`j`生成的所有概率相加，然后除以点的总数`N`。

![figure](img/verdhan-ch5-eqs-13x.png)

##### (5.13)

根据这些值，推导出`∑`、`µ`和`f`的新值，这个过程会一直持续到模型收敛。当我们能够最大化似然函数时停止。

这是一个复杂的过程。我们介绍了它，以便您深入了解统计算法背后的发生情况。Python 实现比数学概念要简单得多。

##### 练习 5.3

回答这些问题来检查你的理解：

1. 高斯分布的均值等于 1，标准差等于 0。对还是错？
2. GMM 模型不考虑数据的协方差。对还是错？

### 5.6.2 GMM 的 Python 实现

我们首先导入数据，然后使用 k-means 和 GMM 进行比较。我们遵循以下步骤：

1. 导入所有库和数据集：

```py
import pandas as pd
data = pd.read_csv('vehicle.csv')
import matplotlib.pyplot as plt
```

2. 从数据集中删除任何 NA 值：

```py
data = data.dropna()
```

3. 配置一个`kmeans`算法。我们将簇的数量保持为 5。请注意，我们并没有说这是簇的理想数量。簇的数量只是为了说明目的。我们声明一个变量 k-means 并使用五个簇。接下来拟合数据集：

```py
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
kmeans.fit(data)
```

4. 绘制聚类。首先，对数据集进行预测，然后将值添加到数据框作为一个新列。然后，用不同颜色表示不同聚类的数据绘制。书的打印版不会显示不同的颜色，但 Python 代码的输出会。同样的输出也可以在 GitHub 仓库中找到。

输出如下（见图 5.22）：

```py
pred = kmeans.predict(data)
frame = pd.DataFrame(data)
frame['cluster'] = pred

color=['red','blue','orange', 'brown', 'green']
for k in range(0,5):
    data = frame[frame["cluster"]==k]
    plt.scatter(data["compactness"],data["circularity"],c=color[k])
plt.show()
```

![figure](img/CH05_F22_Verdhan.png)

##### 图 5.22 配置`kmeans`算法后绘制聚类的结果

5. 配置一个 GMM 模型。请注意，代码与 k-means 算法相同，只是算法的名称从 k-means 更改为`GaussianMixture`：

```py
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=5)
gmm.fit(data)

#predictions from gmm
labels = gmm.predict(data)
frame = pd.DataFrame(data)
frame['cluster'] = labels
```

6. 绘制结果。输出如下（见图 5.23）：

```py
color=['red','blue','orange', 'brown', 'green']
for k in range(0,5):
    data = frame[frame["cluster"]==k]
    plt.scatter(data["compactness"],data["circularity"],c=color[k])
plt.show()
```

![figure](img/CH05_F23_Verdhan.png)

##### 图 5.23 配置 GMM 算法后绘制聚类的结果

7. 使用不同的聚类值运行代码以观察差异。在下面的图中，左边的是具有两个聚类的 k-means，而右边的是具有两个聚类的 GMM。在这两种聚类方法中，有几个点被分类为不同的类别。书的打印版不会显示不同的颜色，但 Python 代码的输出会。同样的输出也可以在 GitHub 仓库中找到（见图 5.24）。

![figure](img/CH05_UN01_Verdhan.png)![figure](img/CH05_F24_Verdhan.png)

##### 图 5.24 具有两个聚类的 k-means（左）和具有两个聚类的 GMM（右）

高斯分布是使用最广泛的数据分布之一。如果我们比较 k-means 和 GMM 模型，我们会看到 k-means 不考虑数据的正态分布。在 k-means 中，也没有考虑各种数据点之间的关系。

注意：k-means 是一种基于距离的算法；GMM 是一种基于分布的算法。

简而言之，使用 GMM 模型创建聚类是有利的，尤其是在我们拥有重叠数据集时。它对于金融和价格建模、基于自然语言处理解决方案等都是一种有用的技术。

## 5.7 总结思考

在本章中，我们探讨了三种复杂的聚类算法。你可能觉得数学概念有点沉重。确实如此，但它们提供了对过程的更深入理解。这些算法并不一定是每个问题的最佳选择。理想情况下，在现实世界的商业问题中，我们首先从经典的聚类算法（k-means、层次和 DBSCAN）开始。如果我们没有得到可接受的结果，我们可以尝试更复杂的算法。

许多时候，数据科学问题被等同于算法的选择，这其实并不正确。算法确实是整个解决方案的重要组成部分，但并非唯一。在现实世界的数据集中，有很多变量，数据量也相当大。数据中有很多噪声。在选择算法时，我们应该考虑到所有这些因素。算法的维护和更新也是需要考虑的因素。所有这些方面都在本书的最后一章中详细讨论。

## 5.8 实践步骤和推荐阅读

以下提供了一些下一步行动的建议和一些有用的阅读材料：

+ 在第二章中，我们使用了各种技术进行聚类。使用那里的数据集，执行谱聚类、GMM 和 FCM 聚类以比较结果。第二章末尾提供的数据集可用于聚类。

+ 从 Kaggle 获取聚类用信用卡数据集([`mng.bz/oKwd`](https://mng.bz/oKwd))和之前使用过的著名 Iris 数据集的数据([`www.kaggle.com/uciml/iris`](https://www.kaggle.com/uciml/iris))。

+ 参考亨利·赫克斯莫尔的书籍《计算网络科学》来学习数学概念。

+ 从以下链接获取谱聚类论文并学习：

    + 关于谱聚类：分析和算法：[`mng.bz/nRwa`](https://mng.bz/nRwa)

    + 基于特征值选择的谱聚类：[`mng.bz/vKw7`](https://mng.bz/vKw7)

    + 谱聚类的数学原理及其与主成分分析的等价性：[`arxiv.org/pdf/2103.00733v1.pdf`](https://arxiv.org/pdf/2103.00733v1.pdf)

+ 从以下链接获取 GMM 论文并探索：

    + “高维面板数据模型的 GMM 估计法”：[`mng.bz/4agw`](https://mng.bz/4agw)

    + “复合高斯混合模型在数据流中的应用”：[`ieeexplore.ieee.org/document/5620507`](https://ieeexplore.ieee.org/document/5620507)

+ 从以下链接获取 FCM 论文并学习：

    + “FCM：模糊 c 均值聚类算法”：[`mng.bz/QDXG`](https://mng.bz/QDXG)

    + 模糊 c 均值聚类技术综述：[`www.ijedr.org/papers/IJEDR1704186.pdf`](https://www.ijedr.org/papers/IJEDR1704186.pdf)

    + “模糊 C 均值和可能性 C 均值聚类算法的实现、聚类趋势分析和聚类验证”：[`arxiv.org/pdf/1809.08417.pdf`](https://arxiv.org/pdf/1809.08417.pdf)

## 摘要

+ 谱聚类关注聚类中的数据点亲和力而非位置。它适用于复杂的数据形状，而传统的算法如 k-means 可能不足以满足需求。

+ 谱聚类利用图理论和连通性，依赖于特征值、拉普拉斯矩阵和亲和矩阵。

+ 该过程包括计算度、邻接矩阵、拉普拉斯矩阵和 Fielder 向量以进行聚类。

+ K-means 聚类使用质心，而谱聚类的重点是连通性和数据点相似性。

+ 由于矩阵运算，谱聚类可能需要大量的计算资源，因此适用于较小的数据集。

+ 模糊聚类允许数据点属于多个簇，为数据项引入了“隶属度”。

+ FCM 是模糊聚类中的一个关键算法，它通过隶属度和通过超参数`m`来控制模糊度。

+ GMM 使用高斯分布进行软聚类，并考虑数据集的协方差。

+ GMM 适用于重叠数据集，并且与 k-means 不同，它考虑数据点之间的关系。

+ 在 GMM 中，EM 技术被用于迭代估计参数。

+ GMM 模型在金融建模、自然语言处理以及重叠数据的案例中具有优势。

+ 模糊聚类和 GMM 是软聚类方法，允许对数据点进行详细的隶属度和概率分配。

+ 谱聚类支持图像分割、语音分析和文本分析等应用，而不需要假设数据形状约束。
