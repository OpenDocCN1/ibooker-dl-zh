
通常，`GraphsTuple` 是一种灵活的数据结构，可以以多种方式存储数据。我们不必为每个数据分割创建一个 `GraphsTuple`，而是可以构建一个包含训练集和评估数据集的单个图，使用 `nodes` 属性来指定每个节点属于哪个集合。

## 准备样本对

图构建完成后，我们使用 `prepare_pairs` 方法获取模型将分类为连接或不连接的药物-药物对——包括正负样本：

```py
def prepare_pairs(
    self, graph: int, pos_pairs: jax.Array, neg_pairs: jax.Array | None = None
) -> Pairs:
    """Prepares positive and negative edge pairs."""
    if neg_pairs is None:
        neg_pairs = self.infer_negative_pairs(graph)
    return Pairs(pos=pos_pairs, neg=neg_pairs)
```

对于评估数据集，准备样本对很简单，因为我们可以直接使用 OGBL 数据集提供的正负样本对。

###### 注意

你可能会想知道为什么我们不使用我们刚刚创建的图中的边来生成正样本对。原因是，由于我们使图无向，每个正样本对被表示了两次，这可能导致评估过程中的冗余和错误。

对于训练数据集，准备样本对稍微复杂一些，因为负样本对没有提供，必须使用 `infer_negative_pairs` 方法推断：

```py
def infer_negative_pairs(self, graph: jraph.GraphsTuple) -> jax.Array:
    """Infers negative edge pairs in a graph."""
    # Initialize a matrix where all possible edges are marked as potential
    # negative edges (1).
    neg_adj_mask = np.ones((graph.n_node[0], graph.n_node[0]), dtype=np.uint8)

    # Mask out existing edges in the graph (set to 0).
    neg_adj_mask[graph.senders, graph.receivers] = 0

    # Use the upper triangular part of the matrix to avoid duplicate pairs and
    # self-loops.
    neg_adj_mask = np.triu(neg_adj_mask, k=1)
    neg_pairs = jnp.array(neg_adj_mask.nonzero()).T  # Extract indices.
    return neg_pairs
```

该方法首先使用 NumPy 构建一个邻接矩阵，并用零初始化。然后，它将所有现有边标记为 1。为了识别负边，该方法翻转矩阵值，使得连接变为 0，非连接变为 1。最后，它仅保留矩阵的上三角部分（不包括对角线），以避免自环和重复样本对。剩余的非零条目被转换为负节点对的边列表。

由于图稀疏性，生成的负样本对远多于正样本对。这种不平衡可能是有利的，因为它提供了更多负边的样本。然而，如前所述，负样本对的采样方式会显著影响性能，因为有些对很容易预测为未连接。在训练过程中，我们需要仔细选择一个公平的负样本子集。

###### 注意

使用邻接矩阵方法假设它可以适应内存。如果这不可行，生成负节点对的替代方法包括采样一个非全面子集或使用依赖于稀疏邻接矩阵的高效实现。

正负样本对随后被封装在一个名为 `Pairs` 的数据类中，我们将在训练过程中进一步探讨它。这是一个简单的数据类，用于存储正负样本对数组，并包括在学习过程中对样本进行子采样和在批量中访问样本的实用工具。

## 图的子集

为了有效地探索模型如何从图中学习，能够创建数据集的一个较小子集非常有用。这允许我们在实验中使用更易于管理的图大小。子集方法正是这样做的：

```py
def subset(
    self,
    dataset_splits: dict[str, Dataset],
    rng: jax.Array,
    node_limit: int,
    keep_original_ids: bool = False,
) -> dict[str, Dataset]:
    """Creates subset of dataset splits by sampling a fixed number of nodes."""
    # Get a random subset of node_ids.
    node_ids = jax.random.choice(
        rng, dataset_splits["train"].n_nodes, (node_limit,), replace=False
    )

    # Subset every dataset split by the same node_ids.
    dataset_subset_splits = {}
    for name, dataset in dataset_splits.items():
        dataset_subset_splits[name] = dataset.subset(node_ids, keep_original_ids)

    return dataset_subset_splits
```

它根据指定的 `node_limit` 选择节点子集，并在所有数据集分割（即训练、验证、测试）中一致地应用此子集。默认情况下，子集图重新编号节点 ID 以创建一个更小、更紧凑的图。然而，您可以通过将 `keep_original_ids` 参数设置为 `True` 来保留来自完整数据集的原始节点 ID。

## 数据集类

最后，我们已经准备好所有组件来创建一个 `Dataset` 类，它将允许灵活地探索图及其注释：

```py
@dataclass
class Dataset:
    """Graph dataset with nodes, pairs, and optional annotations."""

    n_nodes: int
    graph: jraph.GraphsTuple
    pairs: Pairs
    annotation: pd.DataFrame = field(default_factory=pd.DataFrame)
```

在这里，我们只展示了 `Dataset` 字段，但该类还提供了几个有用的方法。值得注意的是，它在幕后处理图的子集和注释管理。虽然我们鼓励您在线探索代码以更好地了解其功能，但理解每个细节并不是开始训练所必需的。

# 构建原型

让我们从简单开始。我们需要构建一个预测节点之间链接的模型。我们将仅使用图的连通性——没有节点特征或注释来完成这项工作。令人惊讶的是，仅凭这种连通性信息本身就可以非常强大，用于学习哪些节点对可能连接在一起。

虽然链接预测可以被视为一个二元分类任务（即连接与不连接），但它与典型的分类问题在关键方面有所不同。模型的输入不是一个单独的节点或特征向量，而是一对节点，并且预测取决于它们在图中的结构关系。

我们的原型模型将由几个关键组件组成，其中一些直接在模型中定义，而其他则作为训练过程的一部分进行处理：

+   模型组件：
    +   邻域编码：生成反映局部图结构的节点嵌入
    +   链接预测：使用这些嵌入来评估节点对之间连接的可能性

+   训练组件：
    +   负采样：选择未连接的节点对以与真实边进行对比
    +   损失函数：计算训练信号以优化模型性能

我们首先关注最关键的部分：我们如何编码每个节点的局部邻域。

## 节点编码器

对于我们的模型来说，最具影响力的选择可能是我们如何编码节点的邻域。为此，我们使用一个受 GraphSAGE 启发的实现：

```py
class NodeEncoder(nn.Module):
    """Encodes nodes into embeddings using a two-layer GraphSAGE model."""

    n_nodes: int
    embedding_dim: int
    last_layer_self: bool
    degree_norm: bool
    dropout_rate: float

    def setup(self):
        """Initializes node embeddings, which cover the full graph's n_nodes."""
        self.node_embeddings = nn.Embed(
            num_embeddings=self.n_nodes,
            features=self.embedding_dim,
            embedding_init=jax.nn.initializers.glorot_uniform(),
        )

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, is_training: bool) -> jax.Array:
        """Encodes the nodes of a graph into embeddings."""
        # Graph can be a subgraph and thus we use a subset of embeddings
        x = self.node_embeddings(graph.nodes["gid"])

        # First convolutional layer.
        x = SAGEConv(
            self.embedding_dim, with_self=True, degree_norm=self.degree_norm
        )(graph, x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not is_training)(x)

        # Second convolutional layer.
        x = SAGEConv(
            self.embedding_dim,
            with_self=self.last_layer_self,
            degree_norm=self.degree_norm,
        )(graph, x)

        return x
```

模块的主要输入参数包括：

`n_nodes`

定义原始图中的节点总数。

`embedding_dim`

指定节点嵌入的维度。这控制了模型可以多么丰富地表示邻域信息。较低的值（例如，16 或 32）可能会限制表达能力，而较高的值（例如，128 或 256）提供了更多的容量，但代价是增加了计算量。在实践中，较小的图可以支持较高的嵌入维度，而较大的图则受益于较低的值。

`dropout_rate`

设置在训练期间随机失活的神经元比例，以减少过拟合。

`last_layer_self` 和 `degree_norm`

配置图卷积行为的一些方面，这些方面将在下一节中更详细地描述。

GraphSAGE 的核心是 `node_embeddings` 矩阵。在 `setup` 方法中，这个矩阵被初始化为一个可学习的参数，使用 `nn.Embed`，其形状为 `[n_nodes, embedding_dim]`。这些嵌入使用 `glorot_uniform` 方法初始化，以促进稳定的训练动态。在训练过程中，这些嵌入通过聚合每个节点的邻居信息来更新，逐渐编码更高阶的结构模式。目标是让这些嵌入收敛到反映节点对之间连接可能性的表示。

编码器的主要逻辑在 `__call__` 方法中实现。这里的一个重要（尽管是硬编码的）设计选择是 `SAGEConv` 层的数量，它定义了应用多少轮消息传递：

+   在一层中，每个节点从其直接邻居那里聚合信息。
+   在两层中，每个节点可以访问到两跳远的邻居信息。

因此，层的数量控制了每个节点的感受野。层之间，ReLU 激活引入了非线性，并且应用 dropout 进行正则化以防止过拟合。

## 图卷积

我们现在已经到达了我们模型的核心架构组件：`SAGEConv` 层。让我们深入探讨它：

```py
class SAGEConv(nn.Module):
    """GraphSAGE convolutional layer with optional self-loops."""

    embedding_dim: int
    with_self: bool
    degree_norm: bool

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, x) -> jax.Array:
        n_nodes = self.get_n_nodes(graph)

        # Add self-loops if enabled.
        if self.with_self:
            senders, receivers = self._add_self_edges(graph, n_nodes)
        else:
            senders, receivers = graph.senders, graph.receivers

        # Aggregate node features from neighbors.
        if not self.degree_norm:
            x_updated = jraph.segment_mean(
                x[senders], receivers, num_segments=n_nodes
            )
        else:

            def get_degree(n):
                return jax.ops.segment_sum(jnp.ones_like(senders), n, n_nodes)

            x_updated = self.normalize_by_degree(x, get_degree(senders))
            x_updated = jraph.segment_mean(
                x_updated[senders], receivers, num_segments=n_nodes
            )
            x_updated = self.normalize_by_degree(x_updated, get_degree(receivers))

        # Combine node and neighbor embeddings by concatenation.
        combined_embeddings = jnp.concatenate([x, x_updated], axis=-1)

        return nn.Dense(self.embedding_dim)(combined_embeddings)

    @staticmethod
    def _add_self_edges(
        graph: jraph.GraphsTuple, n_nodes: int
    ) -> tuple[jax.Array, jax.Array]:
        """Adds self-loops to the graph."""
        all_nodes = jnp.arange(n_nodes)
        senders = jnp.concatenate([graph.senders, all_nodes])
        receivers = jnp.concatenate([graph.receivers, all_nodes])
        return senders, receivers

    @staticmethod
    def normalize_by_degree(x: jax.Array, degree: jax.Array) -> jax.Array:
        """Normalizes node features by the square root of the degree."""
        # We set the the degree to a minimum of 1.
        return x * jax.lax.rsqrt(jnp.maximum(degree, 1.0))[:, None]

    @staticmethod
    def get_n_nodes(graph):
        """Returns the number of nodes in the graph in a jittable way."""
        return tree.tree_leaves(graph.nodes)[0].shape[0]
```

在设置 `SAGEConv` 层时，我们指定嵌入维度（使用 `embedding_dim`），是否添加自环（`with_self`），以及是否应用度数归一化（`degree_norm`）。后两个选项是可选的，因为它们对模型性能的影响取决于数据集的特征，如大小和连接模式。启用或禁用这些功能可以显著影响模型行为。

每个 `SAGEConv` 层执行以下关键步骤：

可选地添加自环

允许每个节点在聚合时考虑其自身的嵌入

聚合邻域嵌入

收集并平均每个节点邻居的嵌入

可选地按度数进行归一化

根据邻居的度数缩放邻居的贡献，减少高度连接节点的偏差

将嵌入与邻居结合

将原始节点嵌入与聚合的邻居嵌入合并

###### 注意

当阅读 `SAGEConv` 卷积层的实现时，你可能想知道：*卷积在哪里？*
