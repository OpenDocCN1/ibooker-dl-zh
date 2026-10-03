
# 将数据集转换为 PyG 张量

为此，我们从一个早期部分生成的同构邻接列表开始。使用 NetworkX，我们将其转换为 NetworkX `graph`对象。从那里，我们使用 PyG 的`from_networkx`函数将其转换为 PyG `data`对象。

# 将训练/测试分割应用于转换后的数据对象

为此，我们使用第一步中的索引。

我们希望展示多种安排训练数据以供摄入的方式。因此，对于 GCN，我们将整个数据集通过模型运行，而在 GAT 示例中，我们将训练数据进行分批。

## 列表 4.4 转换训练数据的数据类型

```py
from torch_geometric.transforms import NormalizeFeatures

split = 0.2                                        #1
indices = np.arange(len(features))                 #1
xtrain, xtest, ytrain, ytest, idxtrain, idxtest\
 = train_test_split(features labels,indices, \
stratify=labels, test_size = split, \
random_state = 99)                                 #2

g = nx.Graph(homogenous)                                            #3
print(f'Number of nodes: {g.number_of_nodes()}')
print(f'Number of edges: {g.number_of_edges()}')
print(f'Average node degree: {len(g.edges) / len(g.nodes):.2f}')
data = from_networkx(g)                                            
data.x = torch.tensor(features).float()                            
data.y = torch.tensor(labels)                                      
data.num_node_features = data.x.shape[-1]                          
data.num_classes = 1 #binary classification                        

A = set(range(len(labels)))                                  #4
data.train_mask = torch.tensor([x in idxtrain for x in A])   #4
data.test_mask = torch.tensor([x in idxtest for x in A])     #4
```

#1 建立训练/测试分割。我们将只使用索引变量。

#2 建立训练/测试分割。我们将只使用索引变量。

#3 将邻接表转换为 PyG 数据对象

#4 在数据对象中建立训练/测试分割

预处理完成后，我们准备应用 GCN 和 GAT 解决方案。我们在第三章详细介绍了 GCN 架构。在列表 4.5 中，我们建立了一个两层 GCN，在 1,000 个 epoch 上训练。我们选择两层是因为第三章的见解，一般来说，低模型深度可以提高性能并防止过平滑。

## 列表 4.5 GCN 定义和训练

```py
class GCN(torch.nn.Module):       #1
    def __init__(self, hidden_layers = 64):
        super().__init__()
        torch.manual_seed(2022)
        self.conv1 = GCNConv(data.num_node_features, hidden_layers)
        self.conv2 = GCNConv(hidden_layers, 1)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return torch.sigmoid(x)

device = torch.device("cuda"\
 if torch.cuda.is_available() \
else "cpu")              #2
print(device)
model = GCN()
model.to(device)
data.to(device)

lr = 0.01
epochs = 1000

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
criterion = torch.nn.BCELoss()

losses = []
for e in range(epochs):      #3
    model.train()
    optimizer.zero_grad()
    out = model(data)                          #4

    loss = criterion(out[data.train_mask], \
    data.y[data.train_mask].\
    reshape(-1,1).float())                    
    loss.backward()
    losses.append(loss.item())

    optimizer.step()

    ypred = model(data).clone().cpu()
    pred = data.y[data.test_mask].clone().cpu().detach().numpy()
    true = ypred[data.test_mask].detach().numpy()
    acc = roc_auc_score(pred,true)

    print(f'Epoch {e} | Loss {loss:6.2f} \
    | Accuracy = {100*acc:6.3f}% \
    | # True Labels =\ {ypred.round().sum()}')
fpr, tpr, _ = roc_curve(pred,true)      #5
```

#1 定义了两层 GCN 架构

#2 实例化模型并将模型和数据放在 GPU 上

#3 训练循环

#4 对于每个 epoch，我们将整个数据对象通过模型，然后使用训练掩码来计算损失。

#5 计算假阳性率（fpr）和真阳性率（tpr）

## 应用解决方案

值得注意的是我们训练中使用了掩码。虽然我们使用训练掩码中的节点来建立损失，但在前向传播时，我们必须将整个图通过模型传递。为什么是这样？与在独立数据点（例如表格数据集的行）上工作的传统机器学习模型不同，GNNs 在图结构数据上操作，其中节点之间的关系至关重要。在训练 GCN 时，每个节点的嵌入是基于其邻居的信息更新的。因为这个消息传递过程涉及到从节点的局部邻域中聚合信息，所以模型需要访问整个图结构，以便它可以正确准确地计算这些聚合并有效地执行此过程。

因此，在训练过程中，尽管我们只对某些节点的预测感兴趣（那些在训练集中的节点），但通过将整个图通过模型传递确保了考虑了所有必要上下文。如果只将图的一部分通过模型传递，网络将缺乏传播消息和有效更新节点表示所需的所有完整信息。

GCN 的 100 个 epoch 的训练会得到 94.37%的准确率。通过引入图数据，我们看到了与 XGBoost 模型相比的渐进改进。表 4.7 比较了模型性能水平。

### 表 4.7 四个基线模型的 AUC

| 模型 | AUC |
| --- | --- |
| 逻辑回归 | 75.90% |
| XGBoost | 94.17% |
| 多层感知器 | 85.93% |
| GCN | 94.37% |

总结来说，我们已看到，使用 GNN 模型包含图结构信息与仅基于特征或表格模型相比，略微提高了性能。很明显，XGBoost 模型即使没有使用图结构也展示了令人印象深刻的成果。然而，GCN 模型略微更好的性能凸显了 GNN 在利用图数据中嵌入的关系信息方面的潜力。

在我们研究的下一阶段，我们的注意力将转向图注意力网络（GATs）。GATs 具有专门针对在消息传递步骤中如何权衡邻居重要性进行学习的注意力机制。这可能会提供更好的模型性能。在下一节中，我们将深入了解训练 GAT 模型的细节，并将它们的成果与我们已经建立的基线进行比较。让我们继续进行 GAT 模型的训练。

# 4.4 训练 GAT 模型

为了训练我们的 GAT 模型，我们将应用两种 PyG 实现（GAT 和 GATv2）[2]。在本节中，我们将直接进入模型的训练过程，而不讨论对于机器学习模型来说注意力意味着什么以及为什么它是有帮助的。然而，关于注意力及其为何可能就是你所需要的简要概述，请参阅第 4.5 节。

我们将训练两种不同的 GAT 模型。这两个模型都遵循相同的基本思想——我们将用注意力机制替换我们的 GCN 中的聚合操作，以学习模型应该最关注哪些消息（节点特征）。第一个——GATConv——是对第三章中 GCN 的简单扩展，加入了注意力机制。第二个是对此模型稍作修改的 GATv2Conv。这个模型与 GATConv 相同，除了它解决了原始实现中的一个限制，即注意力机制在单个 GNN 层上是静态的。相反，对于 GATv2Conv，注意力机制在层之间是动态的。

再次强调，原始的 GAT 模型仅在每次训练循环中通过使用单个节点和邻域特征来计算注意力权重一次，并且这些权重在所有层中都是静态的。在 GATv2 中，注意力权重是在节点特征通过层变换时计算的。这允许 GATv2 更具表现力，学习在整个训练模型中强调节点邻域的影响。

由于引入了注意力机制，这两个模型引入了显著的计算开销。为了解决这个问题，我们在训练循环中引入了小批量处理。

## 4.4.1 邻域加载器和 GAT 模型

从实现的角度来看，先前研究的卷积模型和我们的 GAT 模型之间有一个关键的区别，那就是 GAT 模型的内存需求要大得多 [9]。这是因为 GAT 需要对每个注意力头和每条边进行注意力分数的计算。这反过来又需要 PyTorch 的 `autograd` 方法在内存中保留可以显著扩展的张量，这取决于边的数量、头的数量和（两倍）节点特征的数量。

为了解决这个问题，我们可以将我们的图分成批次，并将这些批次加载到训练循环中。这与我们之前对 GCN 模型的做法形成对比，我们当时在一个单独的批次（整个图）上训练。PyG 的 `NeighborLoader`（在其 `dataloader` 模块中）允许这种小批量训练，我们在这个列表 4.6 中提供了相应的实现代码。（PyG 函数 `NeighborLoader` 基于“在大型图上的归纳表示学习”论文 [10]。）`NeighborLoader` 的关键输入参数是

+   `num_neighbors`——将被采样的邻居节点数量，乘以迭代次数（即 GNN 层）。在我们的例子中，我们指定在两次迭代中采样 1,000 个节点。

+   `batch_size`——每个批次选择的节点数量。在我们的例子中，我们将批量大小设置为 `128`。

### 列表 4.6 为 GAT 设置 `NeighborLoader`

```py
from torch_geometric.loader import NeighborLoader

batch_size = 128
loader = NeighborLoader(
    data,
    num_neighbors=[1000]*2,   #1
    batch_size=batch_size,   #2
    input_nodes=data.train_mask)

sampled_data = next(iter(loader))
print(f'Checking that batch size is \
{batch_size}: {batch_size == \
sampled_data.batch_size}')
print(f'Percentage fraud in batch: \
{100*sampled_data.y.sum()/\
len(sampled_data.y):.4f}%')
sampled_data
```

#1 在两次迭代中为每个节点采样 1,000 个邻居

#2 使用批量大小来采样训练节点

在创建我们的 GAT 模型时，相对于我们的 GCN 类别，有两个关键的改变。首先，因为我们是在批量训练，所以我们想应用一个批量归一化层。批量归一化是一种用于将神经网络中每一层的输入归一化到均值为 0 和标准差为 1 的技术。这有助于通过减少内部协变量偏移来稳定和加速训练过程，允许使用更高的学习率，并提高模型的总体性能。

其次，我们注意到我们的 GAT 层有一个额外的输入参数——`heads`——它表示多头注意力的数量。在我们的例子中，我们的第一个 `GATConv` 层有两个头，如列表 4.7 所指定。

第二个 `GATConv` 层，即输出层，有一个头。在这个 GAT 模型中，因为我们希望最终层对每个节点都有一个单一的表现形式来完成我们的任务，所以我们使用一个头。多个头会导致输出混乱，出现多个节点表现形式。

### 列表 4.7 基于 GAT 的架构

```py
class GAT(torch.nn.Module):
    def __init__(self, hidden_layers=32, heads=1, dropout_p=0.0):
        super().__init__()
        torch.manual_seed(2022)
        self.conv1 = GATConv(data.num_node_features,\
 hidden_layers, heads, dropout=dropout_p)                            #1
        self.bn1 = nn.BatchNorm1d(hidden_layers*heads)     #2
        self.conv2 = GATConv(hidden_layers * heads, \
1, dropout=dropout_p)                                               

    def forward(self, data, dropout_p=0.0):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)                                   
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return torch.sigmoid(x)
```

#1 GAT 层有一个 `heads` 参数，它决定了每一层中的注意力机制数量。在这个实现中，第一层（conv1）使用多个头进行更丰富的特征提取，而最终的输出层（conv2）使用一个头将学习到的信息聚合为每个节点的单一输出。

#2 因为正在执行小批量训练，所以添加了一个批量归一化层。

我们对 GAT 的训练程序与单批次的 GCN 相似，我们将在以下列表中提供，但现在我们需要为每个批次进行嵌套循环。

### 列表 4.8 GAT 的训练循环
