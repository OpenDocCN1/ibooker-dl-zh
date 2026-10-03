
```py
def calculate_loss(self, inputs,
    is_train=False,
    teacher_forcing=True,
    return_edges=False,
    return_logits=False):

    encoder_results = self.encoder(inputs)
    logits = encoder_results['logits']
    hard_sample = not is_train
    edges = F.gumbel_softmax\
            (logits.view(-1, self.num_edge_types),
            tau=self.gumbel_temp,
            hard=hard_sample).view\
                    (logits.shape)  #1

    output = self.decoder(inputs[:, :-1], edges)

    if len(inputs.shape) == 3: \
target = inputs[:, 1:] 
    else:
        Target = inputs[:, 1:, :, :]

    loss_nll = F.mse_loss(\
output, target) / (2 * \
self.prior_variance)  #2

    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-16)  #3
    loss_kl = (probs * \
(log_probs - torch.log(\
torch.tensor(1.0 /  #4
    self.num_edge_types)))).\
sum(-1).mean() 

    loss = loss_nll + loss_kl

    return loss, loss_nll, loss_kl, logits, output
```

#1 使用 PyTorch 的功能 API 计算 Gumbel-Softmax，在代码中导入为 `F`

#2 高斯分布的负对数似然 (NLL)

#3 添加一个小的常数以避免取零的对数

#4 与均匀分类分布的 KL 散度

最后，我们需要我们的模型能够预测传感器的未来轨迹。预测图未来状态的代码在列表 6.12 中给出。一旦我们的编码器和解码器被训练，这是一个相对简单的函数。我们传递当前图给编码器，这返回一个表示是否存在边缘的潜在表示。然后我们使用 Gumbel-Softmax 将这些概率转换为合适的分布，并将其传递给解码器。解码器的输出是我们的预测。我们可以直接获取预测，或者获取预测和是否存在边缘。

##### 列表 6.12 预测未来

```py
def predict_future(self, inputs, prediction_steps, 
   return_edges=False, 
   return_everything=False): #1
    encoder_dict = self.encoder(inputs) #1
    logits = encoder_dict['logits'] 
    edges = nn.functional.gumbel_softmax(  #2
        logits.view(-1, \
        self.num_edge_types),  
        tau=self.gumbel_temp,\
        hard=True).view(logits.shape\ 
        ) 
    tmp_predictions, decoder_state =\
       self.decoder(  #3
       inputs[:, :-1],  #3
       edges,  #3
       return_state=True  #3
    )  #3
    predictions = self.decoder(  #4
       inputs[:, -1].unsqueeze(1),   #4
       edges,   #4
       prediction_steps=prediction_steps,   #4
       teacher_forcing=False,  #4
       state=decoder_state  #4
       ) #4
    if return_everything:  #5
        predictions = torch.cat([\  #4
           tmp_predictions,\  #5
           Predictions\  #5
           ], dim=1)  #5

    return (predictions, edges)\
       if return_edges else predictions  #6
```

#1 运行编码器以获取边缘类型的 logits

#2 将 Gumbel-Softmax 应用到边缘上

#3 运行解码器以获取初始预测和解码器状态

#4 使用最后一个输入和解码器状态来预测未来步骤

#5 如有必要，则连接初始和未来预测

#6 如果指定，则返回预测和边缘

这是 NRI 模型的基础。我们有一个编码器，它将我们的初始节点数据转换为边缘概率。边缘概率传递给我们的解码器，解码器根据最可能的图表示预测条件下的未来轨迹。我们的编码器将是一个简单的多层感知器 (MLP)，它处理图数据。我们的解码器需要能够做出未来预测，因此我们将使用 RNN 来实现这一点，具体是我们在第 6.2.2 节中讨论的相同的 GRU 模型。接下来，让我们认识一下编码器和解码器网络，这样我们就可以将模型应用于数据并查看其性能。

### 6.4.1 编码姿态数据

现在我们已经了解了我们 NRI 模型的不同部分，让我们定义我们的编码器。这个编码器将作为瓶颈来简化我们的问题。编码后，我们将剩下边数据的低维表示，所以我们在这个阶段不需要担心时间数据。然而，通过一起提供我们的时间数据，我们将时间结构转移到我们的潜在空间中。具体来说，编码器从输入数据中提取时间模式和关系，并在压缩的低维表示中保留这些信息。这使得解码更容易，使我们的姿态预测问题更容易解决。

实现编码器有几个子集。首先，我们传递输入数据，它由不同帧、不同实验中的不同传感器组成。然后，编码器将此数据，*x*，执行消息传递步骤，将边数据转换为节点数据，然后再转换回边数据。然后，边数据再次转换为节点数据，在潜在空间中进行编码。这相当于三个消息传递步骤，从边到节点，从边到边，然后再从边到节点。重复的转换对于通过重复消息传递进行信息聚合和捕获图中的高阶交互是有用的。通过在节点和边之间重复转换，模型能够意识到局部和全局结构信息。

在整本书中，我们探讨了如何使用消息传递将节点或边特征转换为节点或边的复杂表示。这些是所有 GNN 方法的核心。NRI 模型与我们之前探索的方法略有不同，因为消息是在节点和边之间传递，而不是节点到节点或边到边。为了明确这些步骤的作用，我们将从 PyG 转向，并用纯 PyTorch 编写我们的模型。

在列表 6.13 中，我们展示了我们编码器的基类，它需要几个关键特性。首先，请注意，我们还没有描述将要用于编码数据的实际神经网络。我们将很快介绍这一点。相反，我们有两个消息传递函数，`edge2node`和`node2edge`，以及一个编码函数，`one_hot_recv`。

##### 列表 6.13 编码器基类

```py
class BaseEncoder(nn.Module):
    def __init__(self, num_vars):
        super(BaseEncoder, self).__init__()
        self.num_vars = num_vars
        edges = torch.ones(num_vars)\
- torch.eye(num_vars)  #1
        self.send_edges, self.\
recv_edges = torch.where(edges)  #2

        one_hot_recv = torch.nn.functional.one_hot(  #3
           self.recv_edges,  #3
           num_classes=num_vars  #3
                                           ) #3
        self.edge2node_mat = \
nn.Parameter(one_hot_recv.\
float().T, requires_grad=False)  #4

    def node2edge(self, node_embeddings):
        send_embed = \
node_embeddings[:, self.send_edges]  #5
        recv_embed = \
node_embeddings[:, self.recv_edges] 
        return torch.\
cat([send_embed, recv_embed], dim=2)  #6

    def edge2node(self, edge_embeddings):
        incoming = torch.\
matmul(self.edge2node_mat, edge_embeddings)  #7
        return incoming / (self.num_vars - 1)  #8
```

#1 创建表示变量之间边的矩阵

#2 查找存在边的索引

#3 创建接收边的 one-hot 表示

#4 创建边到节点转换的参数张量

#5 提取发送者和接收者的嵌入

#6 连接发送者和接收者的嵌入

#7 将边嵌入与边到节点矩阵相乘

#8 正则化传入的嵌入

我们编码器类的第一步是构建一个邻接矩阵。在这里，我们假设图是完全连接的，这意味着所有节点都与其他所有节点相连，但与自己不相连。`node2edge`函数接受节点嵌入数据并识别这些消息发送的方向。图 6.12 展示了我们构建邻接矩阵的一个示例。

![figure](img/6-12.png)

##### 图 6.12 展示了为具有三个节点的完全连接图创建邻接矩阵的示例。左边的矩阵代表一个完全连接的图，中间的矩阵代表单位矩阵，右边的矩阵显示了减去单位矩阵后的最终邻接矩阵。这导致一个每个节点都与其他每个节点相连但没有自环的图。

下一个函数调用通过返回包含连接节点行和列的两个向量来确定哪些节点正在发送或接收数据。回想一下，在邻接矩阵中，行代表接收节点，列代表发送节点。输出结果是

```py
send_edges = tensor([0, 0, 1, 1, 2, 2])
recv_edges = tensor([1, 2, 0, 2, 0, 1])
```

我们可以理解为，行 0 的节点向列 1 和 2 的节点发送数据，依此类推。这使我们能够提取节点之间的边。一旦我们构建了节点嵌入，我们就使用发送和接收数据将节点数据转换为边。这就是`node2edge`函数的原理。

我们需要的下一个函数是如何根据我们的`edge_embeddings`构建`edge2node`。我们首先构建一个`edge2node`矩阵。在这里，我们使用一种 one-hot 编码方法，将接收到的边转换为 one-hot 编码表示。具体来说，我们创建一个矩阵，其中每一行表示该类别（接收节点）是否存在。对于我们的简单三个节点案例，接收边的 one-hot 编码方法如图 6.13 所示。

然后，我们将这个矩阵转置以交换行和列，这样维度将是（节点数，边数），并将其转换为 PyTorch 参数，以便我们可以对其进行微分。一旦我们有了`edge2node`矩阵，我们就将其与边嵌入相乘。我们的边嵌入将是形状为（边数，嵌入大小）的对象，这样将`edge2node`矩阵与边嵌入相乘就给我们一个形状为（节点数，嵌入大小）的对象。这些就是我们的新节点嵌入！最后，我们通过可能的节点数来归一化这个矩阵，以确保数值稳定性。

这一部分是理解模型中消息传递步骤的关键。（关于消息传递的更多信息，请回顾第二章和第三章。）正如所讨论的，一旦我们有一种在节点、边或两者的组合之间传递消息的原则性方法，我们就将这些嵌入应用于神经网络以获得非线性表示。为此，我们需要定义我们的嵌入架构。完整的编码器代码在列表 6.14 中给出。

![figure](img/6-13.png)

##### 图 6.13 展示了在具有三个节点的全连接图中，每个节点 incoming edges 的一热编码矩阵，位于左侧。每一行对应一条边，每一列对应一个节点。位置（i, j）处的 1 表示边 i 指向节点 j。这个矩阵用于在编码器基类的`edge2node`函数中将边嵌入转换为节点嵌入，使模型能够为每个节点聚合来自 incoming edges 的信息。在这个图结构中，节点 0、1 和 2 各自向其他两个节点发送消息，从而产生总共六个有向边。三个节点图的示意图位于右侧。

`RefMLPEncoder`在列表 6.14 中展示。这个编码器使用四个 MLP 进行消息处理，每个 MLP 都具备指数线性单元（ELU）激活和批归一化（在`RefNRIMLP`中定义，本章代码库中展示）。

备注：指数线性单元（ELU）是一种有用的激活函数，可用于平滑多层输出并防止梯度消失。与 ReLU 不同，ELU 在负输入时内置了更平滑的梯度，并允许有负输出。

网络的最后一部分（`self.fc_out`）是一系列带有 ELU 激活的线性层，以输出所需嵌入或预测的线性层结束。这个序列的最后一层是一个全连接层。

##### 列表 6.14 NRI MLP 编码器

```py
class RefMLPEncoder(BaseEncoder):
    def __init__(self, 
            num_vars=31, 
            input_size=6, 
            input_time_steps=50, 
            encoder_mlp_hidden=256, 
            encoder_hidden=256, 
            num_edge_types=2, 
            encoder_dropout=0.):
        super(RefMLPEncoder, self).__init__(num_vars)
        inp_size = input_size * input_time_steps
        hidden_size = encoder_hidden
        num_layers = 3
        self.input_time_steps = input_time_steps

        self.mlp1 = RefNRIMLP\
(inp_size, hidden_size, \
hidden_size, encoder_dropout)  #1
        self.mlp2 = RefNRIMLP\
(hidden_size*2, hidden_size,\
 hidden_size, encoder_dropout) 
        self.mlp3 = RefNRIMLP\
(hidden_size, hidden_size,\
 hidden_size, encoder_dropout) 
        mlp4_inp_size = hidden_size * 2
        self.mlp4 = RefNRIMLP\
(mlp4_inp_size, hidden_size,\
 hidden_size, encoder_dropout)

        layers = [nn.Linear\
(hidden_size, encoder_mlp_hidden), \
nn.ELU(inplace=True)]  #2
        layers += [nn.Linear\
(encoder_mlp_hidden, \
encoder_mlp_hidden),\ 
nn.ELU(inplace=True)] \
* (num_layers - 2) 
        layers.append(nn.\
Linear(encoder_mlp_hidden, \
num_edge_types)) 
        self.fc_out = nn.Sequential(*layers) 
        self.init_weights()
```

#1 定义了 MLP 层。`RefNRIMLP`是一个 2 层全连接 ELU 网络，带有批归一化。

#2 定义了最终的完全连接层
