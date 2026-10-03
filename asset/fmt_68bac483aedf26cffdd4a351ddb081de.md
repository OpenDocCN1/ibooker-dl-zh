
为了管理标记和未标记的数据，初始化并填充了布尔掩码`labelled_mask`和`unlabelled_mask`。`labelled_mask`对于选为标记的索引设置为`True`；这些是相应节点的真实标签。同样，`unlabelled_mask`设置为`False`。这些掩码将数据集分割为训练和评估部分，确保算法在正确的数据子集上正确训练和验证。

### 列表 2.5 半监督问题的预处理

```py
labels = []
for node, data in gml_graph.nodes(data=True):   #1
    if data['value'] == 'c':
        labels.append('right')
    elif data['value'] == 'l':
        labels.append('left')
    else:  
        labels.append('neutral')
labels = np.array(labels)

random.seed(52)   #2

indices = list(range(len(labels)))   #3

labelled_percentage = 0.2    #4

labelled_indices = random.sample(indices, \
int(labelled_percentage * len(labels)))   #5

labelled_mask = np.zeros(len(labels), dtype=bool)   #6
unlabelled_mask = np.ones(len(labels), dtype=bool)

labelled_mask[labelled_indices] = True   #7
unlabelled_mask[labelled_indices] = False

labelled_labels = labels[labelled_mask]   #8
unlabelled_labels = labels[unlabelled_mask]

label_mapping = {'left': 0, 'right': 1, 'neutral': 2}   #9
numeric_labels = np.array([label_mapping[label] for label in labels])
```

#1 提取标签并处理中性值

#2 用于可重复性的随机种子

#3 所有节点的索引

#4 保留 20%的数据作为标记数据

#5 选择保留标记的索引子集

#6 初始化标签数据和未标记数据的掩码

#7 更新掩码

#8 使用掩码分割数据集

#9 将标签转换为数值形式

现在我们对模型训练数据进行转换，如列表 2.6 所示。对于由 GNN 生成的嵌入，`X_train_gnn`和`y_train_gnn`被分配了嵌入数组和相应的数字标签，这些标签通过一个`labelled_mask`进行过滤。这个掩码是一个布尔数组，指示图中哪些节点是标签子集的一部分，确保只有具有已知标签的数据点包含在训练集中。

对于 N2V 嵌入，采用类似的方法，并添加预处理步骤以对齐嵌入与其对应的标签。每个节点的嵌入按`books_graph`中节点出现的顺序聚合到 NumPy 数组`X_n2v`中。这确保了嵌入与其标签之间的一致性，对于监督学习任务是一个关键步骤。随后，`X_train_n2v`和`y_train_n2v`被填充了 N2V 嵌入和标签，再次应用`labelled_mask`来过滤标记数据点。

### 列表 2.6 预处理：构建训练数据

```py
X_train_gnn = gnn_embeddings[labelled_mask]   #1
Y_train_gnn = numeric_labels[labelled_mask]  

X_n2v = np.array([embeddings[str(node)] \
for node in gml_graph.nodes()])  #2
X_train_n2v = X_n2v[labelled_mask]              #3
y_train_n2v = numeric_labels[labelled_mask]     #3
```

#1 用于 GNN 嵌入

#2 用于 N2V 嵌入

#3 确保 N2V 嵌入与标签顺序相同

对于 N2V 嵌入，不需要额外的对齐步骤，因为 GNN 模型在以结构化方式处理整个图时，内在地维护了节点的顺序。因此，GNN 的输出嵌入自然地按照输入图的节点顺序排序。

相比之下，N2V 通过从每个节点开始的独立随机游走来生成嵌入，生成的嵌入的顺序不一定与原始图数据结构中节点的顺序相匹配。因此，需要一个显式的对齐步骤来确保每个 N2V 嵌入与其从图中提取的对应标签正确关联。这一步骤对于监督学习任务至关重要，在这些任务中，将特征（嵌入）与标签的正确匹配对于模型训练和评估至关重要。对于这个任务，我们使用属性`index_to_key`，它包含节点标识符，按照它们在模型中处理和存储的顺序排列。

## 2.3.2 随机森林分类

在我们的数据准备就绪后，我们使用第 2.1 节和第 2.2 节中的 GNN 和 N2V 嵌入作为`RandomForestClassifier`的输入特征，如列表 2.7 所示。

### 列表 2.7 预处理：构建训练数据

```py
clf_gnn = RandomForestClassifier()   #1
clf_gnn.fit(X_train_gnn, y_train_gnn)

clf_n2v = RandomForestClassifier()  #2
clf_n2v.fit(X_train_n2v, y_train_n2v)
```

#1 GNN 嵌入的分类器

#2 N2V 嵌入的分类器

这种方法使我们能够直接比较嵌入的预测能力，如表 2.2 所示。

### 表 2.2 分类性能

| 嵌入类型 | 准确率 | F1 分数 |
| --- | --- | --- |
| GNN | 83.33% | 82.01% |
| N2V | 84.52% | 80.72% |

对于这个基本的分类练习，我们将使用两个基本指标来评估我们模型的性能：

+   *准确率*—这个指标衡量模型在所有预测中正确预测的比例。它提供了一个直接的评估，即分类器正确识别书籍政治倾向的频率。例如，准确率为 84.52%意味着模型在大约 100 次中有 85 次正确预测了书籍的倾向。

+   *F1 分数*—这是一个更细微的指标，它平衡了精确度和召回率，在数据不平衡的情况下尤其有用——这意味着类别并不均匀地表示。它提供了精确度（真实正预测数除以总正预测数）和召回率（真实正预测数除以实际正总数）的调和平均值。更高的 F1 分数表明模型在正确识别不同类别的存在和不存在方面表现出稳健的性能，同时最小化错误正例和错误负例。

性能指标显示，当在`RandomForestClassifier`中使用时，N2V 嵌入产生了略高的准确率 84.52%，而 GNN 嵌入为 83.33%。然而，GNN 嵌入实现了略好的 F1 分数 82.01%，而 N2V 嵌入为 80.72%。这种细微的差异强调了两种嵌入类型之间的潜在权衡：虽然 N2V 提供了略好的整体预测准确率，但 GNN 嵌入可能在多数和少数类别上提供更平衡的性能。

通常，GNN 的归纳性质为学习不同大小图的网络节点表示提供了一个稳健的框架。即使在较小的图上，GNN 也能有效地学习节点之间的底层模式和交互，这从更高的 F1 分数中可以看出，这表明在分类任务中精确度和召回率之间的平衡更好。

在这个背景下，GNN 和 N2V 嵌入的选择也可能取决于分析的具体目标和最感兴趣的性能指标。如果优先考虑实现尽可能高的准确率，并且数据集不太可能显著扩展，那么 N2V 可能是更合适的选择。相反，如果任务重视精确度和召回率之间的平衡，并且有将学习到的模型应用于类似但新的图的可能性，那么 GNN 提供了宝贵的灵活性和稳健性，即使对于较小的数据集也是如此。在将 N2V 和 GNN 嵌入用作随机森林模型的输入之后，接下来让我们研究当我们将它们用作全端到端 GNN 模型的输入时会发生什么。

## 2.3.3 全端到端模型中的嵌入

在上一节中，我们使用了 GNN 和 N2V 嵌入作为静态输入到传统的机器学习模型中，即随机森林分类器。在这里，我们使用一个端到端的 GNN 模型来解决相同的标签预测问题。通过“端到端”，我们指的是在预测标签的同时生成嵌入。这意味着这里的嵌入不会是静态的，因为随着 GNN 的学习，它将更新节点嵌入。

为了构建这个模型，我们将使用之前相同的工具——`books_gml` 数据集和 `SimpleGNN` 架构。我们将稍微修改 GNN，通过在末尾添加 log `softmax` 激活，以方便对三标签分类问题的输出。我们还将稍微修改 `SimpleGNN` 类的输出，使我们能够观察到嵌入以及预测输出。我们的过程包括以下内容：

+   数据准备

+   模型/架构修改

+   建立训练循环

+   研究性能

+   研究嵌入预训练和训练后

### 数据准备

假设我们使用 `books_gml` 数据集，将其转换为 PyG 框架内使用的进程保持不变。我们将训练两种版本的数据：一种使用随机初始化的节点特征，另一种使用 N2V 嵌入的节点特征。

### 模型修改

我们使用相同的 `SimpleGNN` 类及其修改。首先，在这个 `SimpleGNN` 类的增强版本中，我们扩展了其功能，为每个节点提供预测输出。这是通过将第二个 GCN 层产生的嵌入应用 log `softmax` 激活来实现的。log `softmax` 输出为每个节点的潜在类别提供了一个归一化的对数概率分布，用于分类任务。

第二，我们引入双重输出。该方法返回两个值：来自 `conv2` 层的原始嵌入，它捕获节点表示，以及这些嵌入的 log `softmax`。为了观察嵌入和预测，我们将 `forward` 方法返回两者。除了这个双层模型之外，我们还添加了两个层到这个架构中，以形成一个四层模型进行比较，如列表 2.8 所示。

### 列表 2.8 预处理：构建训练数据

```py
class SimpleGNN_inference(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        # First Graph Convolutional layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # Second Graph Convolutional layer
        x = self.conv2(x, edge_index)
        predictions = F.log_softmax(x, dim=1)    #1

        return x, predictions   #2
```

#1 通过将最终卷积层通过 log softmax 预测类别

#2 该类返回最后一个嵌入和预测。

### 建立训练循环

我们在半监督学习环境中为 GNN 模型编写训练循环，如列表 2.9 所示。这个循环遍历指定数量的 epoch，其中 epoch 代表整个训练数据集的一次完整遍历。在每个 epoch 中，模型的参数被更新以最小化损失函数，该函数量化了预测输出与训练集中节点实际标签之间的差异。对于那些熟悉深度学习训练循环编程的人来说，这应该非常熟悉。对于那些需要快速提醒的人来说，以下描述了初始化和运行训练循环的一些关键步骤：

+   *优化器初始化*—当创建优化器时，它会使用一个特定的学习率进行初始化。例如，在这里我们使用 Adam 优化器，初始学习率为 0.01。

+   *归零梯度*—`optimizer.zero_grad()` 确保在每次更新之前重置梯度，防止它们在 epoch 之间累积。

+   *模型前向传播*—模型处理节点特征（`data.x`）和图结构（`data.edge_index`）以产生输出预测。在半监督设置中，并非所有节点都有标签，因此模型的输出包括对标记节点和无标记节点的预测。

+   *应用训练掩码*—`out_masked` `=` `out[data.train_mask]` 对模型的输出应用掩码，以仅选择对应于标记节点的预测。这在半监督学习中至关重要，因为在半监督学习中，只有一部分节点具有已知的标签。

+   *损失计算和反向传播*—损失函数 `loss_fn` 将选定的预测（`out_masked`）与标记节点的真实标签（`train_labels`）进行比较。`loss.backward()` 调用计算损失函数相对于模型参数的梯度，然后通过 `optimizer.step()` 更新这些参数。

+   *日志记录*—训练循环在固定间隔（在这种情况下为每 10 轮）打印损失，以监控训练进度。

### 列表 2.9 训练循环

```py
for epoch in range(3000):    #1
    optimizer.zero_grad()

    _, out = model(data.x, data.edge_index)   #2

    out_masked = out[data.train_mask]   #3

    loss = loss_fn(out_masked, train_labels)   #4
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:    #5
        print(f'Epoch {epoch}, Log Loss: {loss.item()}')
```

#1 训练轮数

#2 将节点特征和 edge_index 传递给模型

#3 应用训练掩码以仅选择标记节点的输出

#4 仅使用标记节点计算损失

#5 每 10 轮打印一次损失

此过程迭代地细化模型的参数，以提高其在数据集标记部分上的预测能力，目标是学习一个能够很好地泛化到无标记节点，甚至可能到新的、未见过的数据的模型。

### GNN 结果：随机化与 N2V 节点特征

让我们比较分类任务，比较随机节点特征与 N2V 节点特征的 GNN 性能，如表 2.3 所示。

#### 表 2.3 使用不同节点特征的 GNN 模型的分类性能

| 模型 | GNN 准确率 | GNN F1 分数 |
| --- | --- | --- |
| 双层，随机化特征 | 82.27% | 82.14% |
| 双层，N2V 特征 | 87.79% | 88.10% |
| 四层，随机化特征 | 86.58% | 86.90% |
| 四层，N2V 特征 | 88.99% | 89.29% |

表格总结了不同 GNN 模型的性能，基于它们的准确率和 F1 分数。它突出了使用 N2V 特征的 GNN 在所有模型配置中一致优于使用随机化特征。具体来说，具有 N2V 特征的四层 GNN 实现了最高的准确率和 F1 分数，这表明从 N2V 嵌入中提取的有意义节点表示的有效性。如果我们对节点具有更多具体信息，就像我们在第三章中所做的那样，GNN 嵌入可能会进一步提高 GNN 模型的准确率。

### 结果：GNN 与随机森林

我们现在比较本节中 GNN 模型的性能与上一节中随机森林模型的性能（见表 2.4）。

#### 表 2.4 GNN 模型和随机森林模型分类性能的比较

| 模型 | 数据输入 | 准确率 | F1 分数 |
| --- | --- | --- | --- |
| 随机森林 | GNN 嵌入 | 83.33% | 82.01% |
| 随机森林 | N2V 嵌入 | 84.52% | 80.72% |
| 两层简单 GNN | 具有随机节点特征的图 | 82.27% | 82.14% |
| 两层简单 GNN | 使用 n2v 嵌入作为节点特征的图 | 87.79% | 88.10% |
| 四层简单 GNN | 随机节点特征的图 | 86.58% | 86.90% |
| 四层简单 GNN | 使用 n2v 嵌入作为节点特征的图 | 88.99% | 89.29% |

图 2.10 可视化表 2.4 的结果。总体而言，GNN 模型优于随机森林模型。

![figure](img/2-10.png)

#### 图 2.10 比较随机森林与 GNN 分类性能的图表。只有一个 GNN 模型被随机森林超越：在具有随机节点特征的图数据上训练的两层模型在准确率方面被超越。

当比较 GNN 模型与随机森林模型的性能时，我们可以得出几个观察结果。当在由 GNN 透传或 N2V 嵌入得到的嵌入上进行训练时，随机森林达到与两层简单 GNN 模型相当的准确率。然而，在考虑 F1 分数时，两个 GNN 模型都优于随机森林。值得注意的是，四层简单 GNN 模型，尤其是在使用 N2V 嵌入作为特征时，比随机森林模型表现出显著更好的性能，展示了更高的准确率和 F1 分数。

这表明，虽然随机森林在准确率方面可能优于简单的 GNN 架构，如两层模型，但在 F1 分数方面，更复杂的 GNN 架构表现出更优越的性能，尤其是在使用 N2V 等复杂的节点嵌入时。因此，在随机森林和 GNN 之间进行选择时，应考虑准确率、F1 分数以及模型架构的复杂性和输入特征的性质，以实现给定任务和数据集的最佳性能。
