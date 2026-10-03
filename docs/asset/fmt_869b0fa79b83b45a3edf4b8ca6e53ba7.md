
# 用 t-SNE 可视化图像聚类

让我们通过可视化整个数据集来提升游戏！

为了做到这一点，我们需要降低特征向量的维度，因为不可能在两个维度（纸张）中绘制一个 2,048 维向量（特征长度）。t-分布随机邻居嵌入（t-SNE）算法将高维特征向量降至 2D，提供数据集的鸟瞰视图，有助于识别聚类和附近图像。t-SNE 难以扩展到大型数据集，因此通过主成分分析（PCA）降低维度然后调用 t-SNE 是一个好主意：

```py
# Perform PCA over the features
num_feature_dimensions=100      # Set the number of features
pca = PCA(n_components = num_feature_dimensions)
pca.fit(featureList)
feature_list_compressed = pca.transform(featureList)

# For speed and clarity, we'll analyze about first half of the dataset.
selected_features = feature_list_compressed[:4000]
selected_class_ids = class_ids[:4000]
selected_filenames = filenames[:4000]

tsne_results =
TSNE(n_components=2,verbose=1,metric='euclidean')
    .fit_transform(selected_features)

# Plot a scatter plot from the generated t-SNE results
colormap = plt.cm.get_cmap('coolwarm')
scatter_plot = plt.scatter(tsne_results[:,0],tsne_results[:,1], c =
               selected_class_ids, cmap=colormap)
plt.colorbar(scatter_plot)
plt.show()
```

我们将在后面的部分更详细地讨论 PCA。为了扩展到更大的维度，使用均匀流形逼近和投影（UMAP）。

图 4-8 展示了相似类别的聚类，以及它们是如何靠近彼此的。

![t-SNE 可视化图像特征聚类，每个聚类用相同颜色表示一个对象类别](img/00136.jpeg)

###### 图 4-8。t-SNE 可视化图像特征聚类，每个聚类用相同颜色表示一个对象类别

图 4-8 中的每种颜色表示不同的类别。为了使其更加清晰，我们可以使用另一个辅助函数`plot_images_in_2d()`来绘制这些集群中的图像，就像图 4-9 中所演示的那样。

![t-SNE 可视化显示图像集群；相似的图像在同一集群中](img/00036.jpeg)

###### 图 4-9\. t-SNE 可视化显示图像集群；相似的图像在同一集群中

很棒！有一个明显划分的人脸、花朵、老式汽车、船只、自行车的集群，以及一个稍微分散的陆地和海洋动物集群。有很多图像重叠在一起，这使得图 4-9 有点令人困惑，所以让我们尝试使用辅助函数`tsne_to_grid_plotter_manual()`将 t-SNE 绘制为清晰的瓷砖，其结果可以在图 4-10 中看到。

```py
tsne_to_grid_plotter_manual(tsne_results[:,0], tsne_results[:,1],
                            selected_filenames)
```

![t-SNE 可视化与瓷砖图像；相似的图像彼此靠近](img/00011.jpeg)

###### 图 4-10\. t-SNE 可视化与瓷砖图像；相似的图像彼此靠近

这绝对更清晰了。我们可以看到相似的图像在人脸、椅子、自行车、飞机、船只、笔记本电脑、动物、手表、花朵、倾斜的尖塔、老式汽车、锚标志和相机的集群中共同定位，都靠近自己的同类。物以类聚！

###### 提示

2D 集群很棒，但在 3D 中可视化它们会看起来更加出色。如果它们可以旋转、缩放，并且可以使用鼠标进行操作而无需编码，那将更好。如果数据可以以交互方式搜索，显示其邻居，那就更加分数。[TensorFlow Embedding projector](https://projector.tensorflow.org)在基于浏览器的 GUI 工具中实现了所有这些功能以及更多。来自图像和文本数据集的预加载嵌入对于更好地理解嵌入的强大性能是有帮助的。正如图 4-11 所示，看到深度学习发现约翰·列侬、齐柏林飞艇和埃里克·克莱普顿恰好在英语中与披头士乐队在类似的语境中使用是令人欣慰的。

![TensorFlow Embedding projector 显示了常见 10,000 个英语单词的 3D 表示，并突出显示了与“披头士”相关的单词](img/00320.jpeg)

###### 图 4-11\. TensorFlow Embedding projector 显示了 10,000 个常见英语单词的 3D 表示，并突出显示了与“披头士”相关的单词

# 提高相似性搜索的速度

有几个机会可以提高相似性搜索步骤的速度。对于相似性搜索，我们可以利用两种策略：要么减少特征长度，要么使用更好的算法在特征之间进行搜索。让我们分别检查这两种策略。

## 特征向量长度

理想情况下，我们期望搜索的数据量越小，搜索速度就应该越快。回想一下，ResNet-50 模型提供了 2048 个特征。每个特征都是一个 32 位浮点数，每个图像由一个 8 KB 的特征向量表示。对于一百万张图像，这相当于将近 8 GB。想象一下在 8 GB 的特征中进行搜索会有多慢。为了更好地了解我们的情况，表 4-1 给出了我们从不同模型中获得的特征长度。

表 4-1\. 不同 CNN 模型的 Top 1%准确率和特征长度

| **模型** | **瓶颈特征长度** | **在 ImageNet 上的 Top-1%准确率** |
| --- | --- | --- |
| VGG16 | 512 | 71.5% |
| VGG19 | 512 | 72.7% |
| MobileNet | 1024 | 66.5% |
| InceptionV3 | 2048 | 78.8% |
| ResNet-50 | 2048 | 75.9% |
| Xception | 2048 | 79.0% |

###### 注意

在幕后，`tf.keras.applications`中提供的许多模型产生数千个特征。例如，InceptionV3 产生形状为 1 x 5 x 5 x 2048 的特征，这相当于 2048 个 5 x 5 卷积的特征图，总共有 51200 个特征。因此，通过使用平均或最大池化层来减少这个大向量变得至关重要。池化层将每个卷积（例如 5 x 5 层）压缩为一个值。这可以在模型实例化期间定义如下：

```py
model = InceptionV3(weights='imagenet', include_top=False,
input_shape = (224,224,3), pooling='max')
```

对于产生大量特征的模型，通常会发现所有代码示例都使用这个池化选项。表 4-2 显示了不同模型中最大池化对特征数量的影响前后效果。

表 4-2。不同模型在池化前后的特征数量

| **模型** | **池化前特征数量** | **池化后特征数量** |
| --- | --- | --- |
| ResNet-50 | [1,1,1,2048] = 2048 | 2048 |
| InceptionV3 | [1,5,5,2048] = 51200 | 2048 |
| MobileNet | [1,7,7,1024] = 50176 | 1024 |

正如我们所看到的，几乎所有模型都生成大量特征。想象一下，如果我们能将特征减少到仅 100 个（减少了 10 到 20 倍！），而不会影响结果的质量，搜索速度会有多快。除了尺寸之外，这对于大数据场景来说是一个更大的改进，因为数据可以一次性加载到 RAM 中，而不是定期加载部分数据，从而提供更大的加速。PCA 将帮助我们实现这一点。

## 使用 PCA 减少特征长度

PCA 是一个统计过程，质疑代表数据的特征是否同等重要。一些特征是否足够冗余，以至于即使删除这些特征，我们仍然可以获得类似的分类结果？PCA 被认为是降维的常用技术之一。请注意，它并不消除冗余特征；相反，它生成一组新的特征，这些特征是输入特征的线性组合。这些线性特征彼此正交，这就是为什么所有冗余特征都不存在。这些特征被称为*主成分*。

执行 PCA 非常简单。使用`scikit-learn`库，执行以下操作：

```py
import sklearn.decomposition.PCA as PCA
num_feature_dimensions=100
pca = PCA(n_components = num_feature_dimensions)
pca.fit(feature_list)
feature_list_compressed = pca.transform(feature_list)
```

PCA 还可以告诉我们每个特征的相对重要性。第一个维度具有最大的方差，随着维度的增加，方差不断减小：

```py
# Explain the importance of first 20 features
print(pca.explained_variance_ratio_[0:20])
```

```py
[ 0.07320023  0.05273142   0.04310822 0.03494248  0.02166119  0.0205037
  0.01974325  0.01739547   0.01611573 0.01548918  0.01450421  0.01311005
  0.01200541  0.0113084    0.01103872 0.00990405  0.00973481  0.00929487
  0.00915592  0.0089256 ]
```

嗯，为什么我们从原始的 2048 个维度中选择了 100 个维度？为什么不是 200 个？PCA 代表我们的原始特征向量，但是在减少维度。每个新维度在表示原始向量方面的回报递减（即，新维度可能不太能解释数据），并占用宝贵的空间。我们可以在原始数据解释得有多好与我们想要减少多少之间取得平衡。让我们可视化前 200 个维度的重要性。

```py
pca = PCA(200)
pca.fit(feature_list)
matplotlib.style.use('seaborn')
plt.plot(range(1,201),pca.explained_variance_ratio_,'o--', markersize=4)
plt.title ('Variance for each PCA dimension')
plt.xlabel('PCA Dimensions')
plt.ylabel('Variance')
plt.grid(True)
plt.show()
```

图 4-12 展示了结果。

![每个 PCA 维度的方差](img/00262.jpeg)

###### 图 4-12。每个 PCA 维度的方差

个体方差将告诉我们新增特征的重要性。例如，在前 100 个维度之后，额外的维度并不增加太多方差（几乎等于 0），可以忽略不计。甚至在检查准确性之前，可以安全地假设具有 100 个维度的 PCA 将是一个强大的模型。另一种看待这个问题的方式是通过找到累积方差来可视化原始数据有多少是由有限数量的特征解释的（参见图 4-13）。

```py
plt.plot(range(1,201),pca.explained_variance_ratio_.cumsum(),'o--', markersize=4)
plt.title ('Cumulative Variance with each PCA dimension')
plt.xlabel('PCA Dimensions')
plt.ylabel('Variance')
plt.grid(True)
plt.show()
```

![每个 PCA 维度的累积方差](img/00237.jpeg)

###### 图 4-13。每个 PCA 维度的累积方差

如预期的那样，添加 100 个维度（从 100 到 200）仅增加了 0.1 的方差，并开始逐渐趋于平稳。作为参考，使用完整的 2,048 个特征将导致累积方差为 1。

PCA 中的维度数量是一个重要的参数，我们可以根据手头的问题进行调整。直接证明一个好的阈值的一种方法是找到特征数量与其对准确性和速度的影响之间的良好平衡：

```py
pca_dimensions = [1,2,3,4,5,10,20,50,75,100,150,200]
pca_accuracy = []
pca_time = []

for dimensions in pca_dimensions:
    # Perform PCA
    pca = PCA(n_components = dimensions)
    pca.fit(feature_list)
    feature_list_compressed = pca.transform(feature_list[:])
    # Calculate accuracy over the compressed features
    accuracy, time_taken = accuracy_calculator(feature_list_compressed[:])
    pca_time.append(time_taken)
    pca_accuracy.append(accuracy)
    print("For PCA Dimensions = ", dimensions, ",\tAccuracy = ",accuracy,"%",
",\tTime = ", pca_time[-1])
```

我们使用图表图 4-14 来可视化这些结果，并看到在一定数量的维度之后，增加维度并不会导致更高的准确性：

```py
plt.plot(pca_time, pca_accuracy,'o--', markersize=4)
for label, x, y in zip(pca_dimensions, pca_time,pca_accuracy):
    plt.annotate(label, xy=(x, y), ha='right', va='bottom')
plt.title ('Test Time vs Accuracy for each PCA dimension')
plt.xlabel('Test Time')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
```

![每个 PCA 维度的测试时间与准确性](img/00191.jpeg)

###### 图 4-14。每个 PCA 维度的测试时间与准确性

正如图中所示，在超过 100 个维度的特征长度之后，准确性几乎没有提高。与原始数据（2,048）相比，几乎少了 20 倍的维度（100），这在几乎任何搜索算法中都提供了极高的速度和更少的时间，同时实现了类似（有时略有更好）的准确性。因此，100 将是这个数据集的理想特征长度。这也意味着前 100 个维度包含了关于数据集的大部分信息。

使用这种减少的表示有许多好处，如有效利用计算资源、去除噪音、由于维度较少而实现更好的泛化，以及对在这些数据上学习的机器学习算法的性能改进。通过将距离计算减少到最重要的特征，我们还可以稍微提高结果的准确性。这是因为以前所有的 2,048 个特征在距离计算中都是平等贡献的，而现在，只有最重要的 100 个特征发挥作用。但更重要的是，它使我们摆脱了“维度灾难”。观察到，随着维度数量的增加，两个最接近点和两个最远点之间的欧氏距离比 tend to become 1。在非常高维的空间中，来自真实世界数据集的大多数点似乎相互之间的距离相似，欧氏距离度量开始无法区分相似和不相似的项目。PCA 有助于恢复理智。

您还可以尝试不同的距离，如闵可夫斯基距离、曼哈顿距离、杰卡德距离和加权欧氏距离（其中权重是每个特征的贡献，如`pca.explained_variance_ratio_`中所解释的）。

现在，让我们将注意力转向使用这个减少后的特征集来使我们的搜索更快。

# 使用近似最近邻扩展相似性搜索

我们想要什么？最近邻。我们的基准是什么？暴力搜索。虽然在两行中实现起来很方便，但它会遍历每个元素，因此随着数据大小（项目数量以及维度数量）的增加而呈线性增长。将我们的特征向量从 2,048 的长度缩减到 100，不仅会使数据大小减少 20 倍，还会使使用暴力搜索时的速度增加 20 倍。PCA 确实是值得的！

假设相似性搜索一个包含 10,000 张图像的小集合，现在用 100 个特征长度向量表示，大约需要 1 毫秒。即使对于 10,000 个项目来说这看起来很快，但在一个真实的生产系统中，也许有更大的数据，比如 10 百万个项目，这将需要超过一秒的时间来搜索。我们的系统可能无法每秒每个 CPU 核心处理超过一个查询。如果您从用户那里每秒收到 100 个请求，即使在机器的多个 CPU 核心上运行（并为每个线程加载搜索索引），您也需要多台机器才能处理流量。换句话说，低效的算法意味着花费大量的硬件成本。

蛮力搜索是我们进行每次比较的基准。在大多数算法方法中，蛮力搜索是最慢的方法。现在我们已经设定了基准，我们将探索近似最近邻算法。与蛮力搜索方法保证正确结果不同，近似算法*通常*能够得到正确结果，因为它们是...嗯，近似值。大多数算法提供某种形式的调整来平衡正确性和速度。可以通过与蛮力基准结果进行比较来评估结果的质量。

## 近似最近邻基准
