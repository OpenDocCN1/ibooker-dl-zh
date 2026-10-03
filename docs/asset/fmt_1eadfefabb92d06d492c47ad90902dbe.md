
在这里，我们定义与编码器相关的架构细节。如前所述，我们使用`num_vars`变量表示 31 个传感器。特征数量为 6，这是网络的`input_size`。我们的训练和验证集的时间步数为 50，编码器网络的大小将为 256。`edge_types`的数量为 2，我们假设权重没有 dropout。然后我们初始化我们的网络，这些是典型的 MLP，在共享的代码库中描述。网络包括一个批归一化层和两个全连接层。一旦定义了网络，我们也预先初始化权重，如列表 6.15 所示。在这里，我们遍历所有不同的层，然后使用 Xavier 初始化方法初始化权重。这确保了层中的梯度都大致处于相似的比例，从而降低了损失迅速发散的风险——即爆炸。当我们像这里这样组合具有不同架构的多个网络时，这是一个重要的步骤。我们还设置了初始偏置为 0.1，这有助于提高训练的稳定性。

### 列表 6.15 权重初始化

```py
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):  #1
                nn.init.xavier_normal_(m.weight.data)  #2
                m.bias.data.fill_(0.1)  #3
```

#1 仅适用于线性层

#2 使用 Xavier 正态初始化权重

#3 将偏置设置为 0.1

最后，我们需要定义我们的前向传递方法，如列表 6.16 所示。这就是我们的消息传递步骤发生的地方。

### 列表 6.16 编码器前向传递

```py
   def forward(self, inputs, state=None, return_state=False):
       if inputs.size(1) > self.input_time_steps:
           inputs = inputs[:, -self.input_time_steps:]
       elif inputs.size(1) < self.input_time_steps:
           begin_inp = inputs[:, 0:1].expand(
           -1, 
           self.input_time_steps-inputs.size(1),
           -1, -1
           )
           inputs = torch.cat([begin_inp, inputs], dim=1) #1

       x = inputs.transpose(1, 2).contiguous()  #1
       x = x.view(inputs.size(0), inputs.size(2), -1) 

       x = self.mlp1(x)  #2
       x = self.node2edge(x)  #3
       x = self.mlp2(x)  #4

       x = self.edge2node(x)  #5
       x = self.mlp3(x)

       x = self.node2edge(x)  #6
       x = self.mlp4(x)

       result =  self.fc_out(x)  #7
       result_dict = {
          'logits': result,
          'state': inputs,
           }
       return result_dict
```

#1 新形状：`[num_sims, num_atoms, num_timesteps*num_dims]`

#2 通过第一个 MLP 层（每个节点两个 ELU 网络层）

#3 将节点嵌入转换为边嵌入

#4 通过第二个 MLP 层

#5 将边嵌入转换回节点嵌入

#6 再次将节点嵌入转换为边嵌入

#7 最终的全连接层以获取 logits

我们的编码器允许我们的模型将我们的传感器图的帧集转换为边概率的潜在表示。接下来，让我们探索如何构建一个解码器，该解码器使用最近的传感器数据将潜在边概率转换为轨迹。

## 6.4.2 使用 GRU 解码姿态数据

为了将潜在表示转换为未来帧，我们需要考虑轨迹的时间演化。为此，我们训练一个解码器网络。在这里，我们将遵循 NRI 论文[2]的原始结构，并使用 GRU 作为我们的 RNN。

我们在 6.2.2 节中介绍了 GRU 的概念。为了快速回顾，门控循环单元（GRU）是一种使用门控过程来允许 RNN 捕获数据中长期行为的 RNN 类型。它们由两种类型的门组成——重置门和更新门。

对于 NRI 模型，我们将 GRU 应用于我们的边，而不是整个图。更新门将用于确定根据接收到的数据，应该更新节点隐藏状态的多少，而重置门决定应该删除或“忘记”多少。换句话说，我们将使用 GRU 根据编码器网络中的边类型概率预测节点的未来状态。

让我们一步一步地看看我们是如何构建这个步骤的。我们的解码器初始化代码在列表 6.17 中给出。首先，我们注意到传递给这个网络的一些变量。我们再次定义我们图中变量或节点的数量，31 个，以及输入特征的个数，6 个。我们假设权重没有 dropout，并且每层的隐藏大小为 64。再次明确，我们的解码器应该预测两种不同类型的边。在预测时，我们也会跳过第一种边类型，因为这表示没有边。

一旦我们定义了输入参数，我们就可以引入网络架构。第一层是一个简单的线性网络，需要具有两倍的输入维度，以考虑由编码器提供的均值和方差，并且我们为每种边类型定义了这个网络。然后我们定义第二层以进一步提高我们网络的表达能力。这两个线性层的输出传递给我们的 RNN，即 GRU。在这里，我们必须使用自定义 GRU 来考虑节点数据和边数据。GRU 的输出传递给三个更多的神经网络层以提供未来预测。最后，我们需要定义我们的`edge2node`矩阵以及发送和接收节点，就像我们处理编码器时那样。

### 列表 6.17 RNN 解码器

```py
   class GraphRNNDecoder(nn.Module):
       def __init__(self, 
           num_vars=31, 
           input_size=6, 
           decoder_dropout=0., 
           decoder_hidden=64, 
           num_edge_types=2, 
           skip_first=True):
           super(GraphRNNDecoder, self).__init__()
           self.num_vars = num_vars
           self.msg_out_shape = decoder_hidden
           self.skip_first_edge_type = skip_first
           self.dropout_prob = decoder_dropout
           self.edge_types = num_edge_types

           self.msg_fc1 = nn.ModuleList\
([nn.Linear(2 * decoder_hidden,\
 decoder_hidden) for _ in \
range(self.edge_types)])  #1
           self.msg_fc2 = nn.ModuleList\
([nn.Linear(decoder_hidden, decoder_hidden)\
 for _ in range(self.edge_types)])

           self.custom_gru = CustomGRU\
(input_size, decoder_hidden)  #2

           self.out_fc1 = nn.Linear\
(decoder_hidden, decoder_hidden)  #3
           self.out_fc2 = nn.Linear(decoder_hidden, decoder_hidden)
           self.out_fc3 = nn.Linear(decoder_hidden, input_size)

           self.num_vars = num_vars
           edges = np.ones(num_vars) - np.eye(num_vars)
           self.send_edges = np.where(edges)[0]
           self.recv_edges = np.where(edges)[1]
           self.edge2node_mat = \
                torch.FloatTensor\
                (encode_onehot(self.recv_edges))
           self.edge2node_mat = self.edge2node_mat.cuda(non_blocking=True)
```

#1 与边相关的层

#2 GRU 层

#3 全连接层

在列表 6.18 中，我们提供了我们的 GRU 架构。这个网络的整体架构与典型的 GRU 结构相同。我们定义了三个隐藏层，这些层代表了由`hidden_r`和`input_r`定义的重置门，由`hidden_i`和`input_i`定义的更新门，以及由`hidden_h`和`input_h`定义的激活网络。然而，正向网络需要考虑来自编码器消息传递输出的聚合消息。这可以在正向传递中看到。我们将`agg_msgs`中的边概率以及输入节点数据传递，这些数据结合起来返回未来预测。这可以在我们的基础 NRI 类中的`predict_future`代码中看到：

```py
      predictions = self.decoder(inputs[:, -1].unsqueeze(1), edges,
      prediction_steps=prediction_steps, teacher_forcing=False, 
      state=decoder_state)
```

我们的解码器接收我们图的最后一个时间帧。从编码器输出的边数据也传递给解码器。

### 列表 6.18 自定义 GRU 网络

```py
   class CustomGRU(nn.Module):
       def __init__(self,input_size, n_hid,num_vars=31):
           super(CustomGRU, self).__init__()
           self.num_vars = num_vars
           self.hidden_r = nn.Linear
(n_hid, n_hid, bias=False)  #1
           self.hidden_i = nn.Linear\
(n_hid, n_hid, bias=False) 
           self.hidden_h = nn.Linear\
(n_hid, n_hid, bias=False) 

           self.input_r = nn.Linear\
(input_size, n_hid, bias=True)  #2
           self.input_i = nn.Linear(\
input_size, n_hid, bias=True) 
           self.input_n = nn.Linear\
(input_size, n_hid, bias=True) 

       def forward(self, inputs, agg_msgs, hidden):
           inp_r = self.input_r(inputs)\
.view(inputs.size(0), self.num_vars, -1)
           inp_i = self.input_i(inputs)\
.view(inputs.size(0), self.num_vars, -1)
           inp_n = self.input_n(inputs)\
.view(inputs.size(0), self.num_vars, -1)

           r = torch.sigmoid(inp_r + \
self.hidden_r(agg_msgs))  #3
           i = torch.sigmoid(inp_i + \
self.hidden_i(agg_msgs))  #4
           n = torch.tanh(inp_n + \
r*self.hidden_h(agg_msgs))  #5
           hidden = (1 - i)*n + i*hidden  #6

           return hidden
```

#1 定义重置、输入和新门的隐藏层变换

#2 定义重置、输入和新门的输入层变换

#3 计算重置门激活

#4 计算输入门激活

#5 计算新门激活

#6 隐藏状态更新

解码器网络的输出是未来的预测时间步。为了更好地理解这一点，让我们看看解码器的正向传递方法，如列表 6.19 所示。我们的正向传递接收输入和采样边来构建预测。还有四个额外的参数有助于控制行为。首先，我们定义一个`teacher_forcing`变量。教学强制是一种在训练序列模型时常用的典型方法，例如 RNN。如果这是真的，我们使用真实值（真实图）来预测下一个时间帧。当这是假的时，我们使用模型上一个时间步的输出。这确保了模型在训练期间不会被错误的预测所误导。接下来，我们包括一个`return_state`变量，它允许我们访问解码器网络提供的隐藏表示。当我们预测未来图演化时，我们使用这个变量，如这里所示：

```py
     tmp_predictions, decoder_state = \
        self.decoder(inputs[:, :-1], edges, 
        return_state=True)
     predictions = self.decoder\
        (inputs[:, -1].unsqueeze(1), edges, 
        prediction_steps=prediction_steps, \
        teacher_forcing=False, state=decoder_state)
```

现在我们来讨论预测过程。首先，我们预测一个临时预测集。然后，我们使用隐藏表示来预测所需的所有未来步骤。当我们想要预测多个时间步时，这特别有用，正如我们在该模型的测试阶段所展示的。这由`prediction_steps`变量控制，它告诉我们 RNN 要循环多少次，即我们想要预测多少未来的时间步。最后，我们有一个`state`变量，用于控制传递给解码器的信息。当它为空时，我们初始化一个零张量，以便没有信息被传递。否则，我们将使用之前时间步的信息。

### 列表 6.19 解码器正向传递
