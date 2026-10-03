
```py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline

clf = make_pipeline(PCA(random_state=42),
                    RandomForestClassifier(random_state=42))
param_distrib = {
    "pca__n_components": np.arange(10, 80),
    "randomforestclassifier__n_estimators": np.arange(50, 500)
}
rnd_search = RandomizedSearchCV(clf, param_distrib, n_iter=10, cv=3,
                                random_state=42)
rnd_search.fit(X_train[:1000], y_train[:1000])
```

让我们看看找到的最佳超参数：

```py
>>> print(rnd_search.best_params_)
{'randomforestclassifier__n_estimators': 465, 'pca__n_components': 23}
```

有趣的是最佳组件数量是多么低：我们将一个 784 维数据集减少到只有 23 维！这与我们使用了随机森林这一强大模型有关。如果我们改用线性模型，例如`SGDClassifier`，搜索会发现我们需要保留更多维度（约 70 个）。

## 用于压缩的 PCA

降维后，训练集占用的空间大大减少。例如，将 PCA 应用于 MNIST 数据集并保留 95%的方差后，我们只剩下 154 个特征，而不是原始的 784 个特征。因此，数据集现在不到原始大小的 20%，而且我们仅损失了 5%的方差！这是一个合理的压缩比率，很容易看出这种大小的减小会极大地加快分类算法的速度。

还可以通过应用 PCA 投影的逆变换将缩减的数据集解压缩回 784 维。这不会给您原始数据，因为投影丢失了一些信息（在丢弃的 5%方差内），但它可能接近原始数据。原始数据和重构数据（压缩然后解压缩）之间的均方距离称为*重构误差*。

`inverse_transform()`方法让我们将降维后的 MNIST 数据集解压缩回 784 维：

```py
X_recovered = pca.inverse_transform(X_reduced)
```

图 8-9 展示了原始训练集中的一些数字（左侧），以及压缩和解压缩后的相应数字。您可以看到存在轻微的图像质量损失，但数字仍然基本完整。

![mls3 0809](img/mls3_0809.png)

###### 图 8-9。保留 95%方差的 MNIST 压缩

逆变换的方程式显示在方程 8-3 中。

##### 方程 8-3。PCA 逆变换，回到原始维度数量

<math display="block"><mrow><msub><mi mathvariant="bold">X</mi> <mtext>recovered</mtext></msub> <mo>=</mo> <msub><mi mathvariant="bold">X</mi> <mrow><mi>d</mi><mtext>-proj</mtext></mrow></msub> <msup><mrow><msub><mi mathvariant="bold">W</mi> <mi>d</mi></msub></mrow> <mo>⊺</mo></msup></mrow></math>

## 随机 PCA

如果将`svd_solver`超参数设置为`"randomized"`，Scikit-Learn 将使用一种称为*随机 PCA*的随机算法，快速找到前*d*个主成分的近似值。其计算复杂度为*O*(*m* × *d*²) + *O*(*d*³)，而不是完整 SVD 方法的*O*(*m* × *n*²) + *O*(*n*³)，因此当*d*远小于*n*时，它比完整 SVD 快得多：

```py
rnd_pca = PCA(n_components=154, svd_solver="randomized", random_state=42)
X_reduced = rnd_pca.fit_transform(X_train)
```

###### 提示

默认情况下，`svd_solver` 实际上设置为 `"auto"`：Scikit-Learn 在 max(*m*, *n*) > 500 且 `n_components` 是小于 min(*m*, *n*) 的 80%的整数时，自动使用随机化 PCA 算法，否则使用完整的 SVD 方法。因此，即使删除了 `svd_solver="randomized"` 参数，上述代码也会使用随机化 PCA 算法，因为 154 < 0.8 × 784。如果要强制 Scikit-Learn 使用完整的 SVD 以获得稍微更精确的结果，可以将 `svd_solver` 超参数设置为 `"full"`。

## 增量 PCA

PCA 的前述实现的一个问题是，为了使算法运行，整个训练集必须适合内存。幸运的是，已经开发出了*增量 PCA*（IPCA）算法，允许您将训练集分成小批次，并逐个小批次馈送这些数据。这对于大型训练集以及在线应用 PCA（即，随着新实例的到来而进行）非常有用。

以下代码将 MNIST 训练集分成 100 个小批次（使用 NumPy 的 `array_split()` 函数），并将它们馈送给 Scikit-Learn 的 `IncrementalPCA` 类来将 MNIST 数据集的维度降低到 154 维，就像以前一样。请注意，您必须对每个小批次调用 `partial_fit()` 方法，而不是对整个训练集调用 `fit()` 方法：

```py
from sklearn.decomposition import IncrementalPCA

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
    inc_pca.partial_fit(X_batch)

X_reduced = inc_pca.transform(X_train)
```

或者，您可以使用 NumPy 的 `memmap` 类，它允许您像在内存中一样操作存储在磁盘上的二进制文件中的大数组；该类仅在需要时将数据加载到内存中。为了演示这一点，让我们首先创建一个内存映射（memmap）文件，并将 MNIST 训练集复制到其中，然后调用 `flush()` 来确保任何仍在缓存中的数据被保存到磁盘。在现实生活中，`X_train` 通常不会完全适合内存，因此您需要逐块加载它并将每个块保存到 memmap 数组的正确部分：

```py
filename = "my_mnist.mmap"
X_mmap = np.memmap(filename, dtype='float32', mode='write', shape=X_train.shape)
X_mmap[:] = X_train  # could be a loop instead, saving the data chunk by chunk
X_mmap.flush()
```

接下来，我们可以加载 memmap 文件并像常规 NumPy 数组一样使用它。让我们使用 `IncrementalPCA` 类来降低其维度。由于该算法在任何给定时间仅使用数组的一小部分，内存使用保持在控制之下。这使得可以调用通常的 `fit()` 方法而不是 `partial_fit()` 方法，这非常方便：

```py
X_mmap = np.memmap(filename, dtype="float32", mode="readonly").reshape(-1, 784)
batch_size = X_mmap.shape[0] // n_batches
inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
inc_pca.fit(X_mmap)
```

###### 警告

只有原始的二进制数据保存在磁盘上，因此在加载时需要指定数组的数据类型和形状。如果省略形状，`np.memmap()` 将返回一个一维数组。

对于非常高维的数据集，PCA 可能太慢了。正如您之前看到的，即使使用随机化 PCA，其计算复杂度仍然是 *O*(*m* × *d*²) + *O*(*d*³)，因此目标维数 *d* 不应太大。如果您处理的数据集具有成千上万个特征或更多（例如，图像），那么训练可能会变得非常缓慢：在这种情况下，您应该考虑使用随机投影。

# 随机投影

正如其名称所示，随机投影算法使用随机线性投影将数据投影到较低维度的空间。这听起来可能很疯狂，但事实证明，这样的随机投影实际上很可能相当好地保留距离，正如 William B. Johnson 和 Joram Lindenstrauss 在一个著名的引理中数学上证明的那样。因此，两个相似的实例在投影后仍然保持相似，而两个非常不同的实例在投影后仍然保持非常不同。

显然，你丢弃的维度越多，丢失的信息就越多，距离就会变得更加扭曲。那么你如何选择最佳的维度数量呢？Johnson 和 Lindenstrauss 提出了一个方程，确定保留的最小维度数量，以确保——高概率下——距离不会改变超过给定的容差。例如，如果你有一个包含*m*=5,000 个实例，每个实例有*n*=20,000 个特征的数据集，你不希望任意两个实例之间的平方距离变化超过*ε*=10%，⁶，那么你应该将数据投影到*d*维度，其中*d*≥4 log(*m*) / (½ *ε*² - ⅓ *ε*³)，即 7,300 维度。这是一个相当显著的降维！请注意，这个方程不使用*n*，它只依赖于*m*和*ε*。这个方程由`johnson_lindenstrauss_min_dim()`函数实现：

```py
>>> from sklearn.random_projection import johnson_lindenstrauss_min_dim
>>> m, ε = 5_000, 0.1
>>> d = johnson_lindenstrauss_min_dim(m, eps=ε)
>>> d
7300
```

现在我们可以生成一个形状为[*d*, *n*]的随机矩阵**P**，其中每个项都是从均值为 0，方差为 1 / *d*的高斯分布中随机抽样的，然后用它将数据集从*n*维度投影到*d*维度：

```py
n = 20_000
np.random.seed(42)
P = np.random.randn(d, n) / np.sqrt(d)  # std dev = square root of variance

X = np.random.randn(m, n)  # generate a fake dataset
X_reduced = X @ P.T
```

这就是全部！它简单高效，无需训练：算法需要创建随机矩阵的唯一信息是数据集的形状。数据本身根本没有被使用。

Scikit-Learn 提供了一个`GaussianRandomProjection`类，可以做我们刚才做的事情：当你调用它的`fit()`方法时，它使用`johnson_lindenstrauss_min_dim()`来确定输出的维度，然后生成一个随机矩阵，存储在`components_`属性中。然后当你调用`transform()`时，它使用这个矩阵来执行投影。在创建转换器时，如果你想调整*ε*，可以设置`eps`（默认为 0.1），如果你想强制特定的目标维度*d*，可以设置`n_components`。以下代码示例给出了与前面代码相同的结果（你也可以验证`gaussian_rnd_proj.components_`等于`P`）：

```py
from sklearn.random_projection import GaussianRandomProjection

gaussian_rnd_proj = GaussianRandomProjection(eps=ε, random_state=42)
X_reduced = gaussian_rnd_proj.fit_transform(X)  # same result as above
```

Scikit-Learn 还提供了第二个随机投影转换器，称为`SparseRandomProjection`。它以相同的方式确定目标维度，生成相同形状的随机矩阵，并执行相同的投影。主要区别在于随机矩阵是稀疏的。这意味着它使用的内存要少得多：在前面的例子中，约 25 MB，而不是将近 1.2 GB！而且它也更快，无论是生成随机矩阵还是降维：在这种情况下，大约快 50%。此外，如果输入是稀疏的，转换会保持稀疏（除非你设置`dense_output=True`）。最后，它享有与之前方法相同的保持距离性质，降维的质量是可比较的。简而言之，通常最好使用这个转换器而不是第一个，特别是对于大型或稀疏数据集。

稀疏随机矩阵中非零项的比率*r*称为其*密度*。默认情况下，它等于<math><mn>1</mn><mo>/</mo><msqrt><mi>n</mi></msqrt></math>。有了 20,000 个特征，这意味着随机矩阵中大约 141 个单元格中只有 1 个是非零的：这是相当稀疏的！如果你愿意，你可以将`density`超参数设置为另一个值。稀疏随机矩阵中的每个单元格有一个概率*r*是非零的，每个非零值要么是–*v*，要么是+*v*（两者概率相等），其中*v*=<math><mn>1</mn><mo>/</mo><msqrt><mi>d</mi><mi>r</mi></msqrt></math>。

如果你想执行逆变换，你首先需要使用 SciPy 的`pinv()`函数计算组件矩阵的伪逆，然后将减少的数据乘以伪逆的转置：

```py
components_pinv = np.linalg.pinv(gaussian_rnd_proj.components_)
X_recovered = X_reduced @ components_pinv.T
```

###### 警告

如果组件矩阵很大，计算伪逆可能需要很长时间，因为`pinv()`的计算复杂度是*O*(*dn*²)（如果*d*<*n*），否则是*O*(*nd*²)。

总之，随机投影是一种简单、快速、内存高效且令人惊讶地强大的降维算法，尤其在处理高维数据集时应该牢记。

###### 注意

随机投影并不总是用于降低大型数据集的维度。例如，Sanjoy Dasgupta 等人的一篇[2017 年论文](https://homl.info/flies)显示，果蝇的大脑实现了一种类似随机投影的模拟，将密集的低维嗅觉输入映射到稀疏的高维二进制输出：对于每种气味，只有少数输出神经元被激活，但相似的气味会激活许多相同的神经元。这类似于一个名为*局部敏感哈希*（LSH）的著名算法，通常用于搜索引擎以将相似的文档分组。

# LLE

[*局部线性嵌入*（LLE）](https://homl.info/lle)是一种*非线性降维*（NLDR）技术。它是一种流形学习技术，不依赖于投影，不同于 PCA 和随机投影。简而言之，LLE 首先衡量每个训练实例与其最近邻的线性关系，然后寻找训练集的低维表示，以便最好地保留这些局部关系（稍后详细介绍）。这种方法使其特别擅长展开扭曲的流形，尤其是在噪声不多时。

以下代码制作了一个瑞士卷，然后使用 Scikit-Learn 的`LocallyLinearEmbedding`类来展开它：

```py
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding

X_swiss, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
X_unrolled = lle.fit_transform(X_swiss)
```

变量`t`是一个包含每个实例沿瑞士卷轴位置的 1D NumPy 数组。在这个示例中我们没有使用它，但它可以用作非线性回归任务的目标。

生成的 2D 数据集显示在图 8-10 中。如您所见，瑞士卷完全展开，实例之间的距离在局部上得到很好保留。然而，在较大尺度上距离并没有得到保留：展开的瑞士卷应该是一个矩形，而不是这种被拉伸和扭曲的带状。尽管如此，LLE 在对流形建模方面做得相当不错。

![mls3 0810](img/mls3_0810.png)

###### 图 8-10。使用 LLE 展开的瑞士卷
