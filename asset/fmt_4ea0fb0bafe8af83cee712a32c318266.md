
我们可以通过查看概率矩阵`p`来轻松验证屏蔽是否起作用。它必须是一个下三角矩阵。

```
>>> tf.Tensor(
    [[0\. 1\. 1\. 1\. 1\. 1\. 1.]
     [0\. 0\. 1\. 1\. 1\. 1\. 1.]
     [0\. 0\. 0\. 1\. 1\. 1\. 1.]
     [0\. 0\. 0\. 0\. 1\. 1\. 1.]
     [0\. 0\. 0\. 0\. 0\. 1\. 1.]
     [0\. 0\. 0\. 0\. 0\. 0\. 1.]
     [0\. 0\. 0\. 0\. 0\. 0\. 0.]], shape=(7, 7), dtype=float32)
```

```
layer = SelfAttentionLayer(512)
h, p = layer(x, x, x, mask)
print(p.numpy())
```

给出：

```
>>> [[1\.    0\.    0\.    0\.    0\.    0\.    0\.   ]
     [0.37  0.63  0\.    0\.    0\.    0\.    0\.   ]
     [0.051 0.764 0.185 0\.    0\.    0\.    0\.   ]
     [0.138 0.263 0.072 0.526 0\.    0\.    0\.   ]
     [0.298 0.099 0.201 0.11  0.293 0\.    0\.   ]
     [0.18  0.344 0.087 0.25  0.029 0.108 0\.   ]
     [0.044 0.044 0.125 0.284 0.351 0.106 0.045]]
```

现在，在计算值时，模型无法看到或关注到它在处理当前单词时尚未看到的单词。

### 5.2.7 多头注意力

原始 Transformer 论文中讨论了一种称为多头注意力的方法，它是自注意力层的扩展。一旦理解了自注意机制，这个想法就很简单。多头注意力创建多个并行的自注意力头。这样做的动机是，当模型有机会为输入序列学习多个注意力模式（即多组权重）时，它的性能更好。

记住，在单个注意力头中，所有的查询、键和值的维度都设置为 512。换句话说，

`d[q] = d[k] = d[v] = 512`

使用多头注意力，假设我们使用八个注意力头，

`d[q] = d[k] = d[v] = 512/8 = 64`

然后将所有注意力头的最终输出连接起来，形成最终输出，它的维度将为 64 × 8 = 512

`H = Concat(h¹, h², ... , h⁸)`

其中`h^i`是第`i`个注意力头的输出。使用刚刚实现的`SelfAttentionLayer`，代码变为：

```
multi_attn_head = [SelfAttentionLayer(64) for i in range(8)]
outputs = [head(x, x, x)[0] for head in multi_attn_head]
outputs = tf.concat(outputs, axis=-1)
print(outputs.shape)
```

得到：

```
>>> (1, 7, 512)
```

如你所见，它仍然具有之前的相同形状（没有多个头）。然而，此输出是使用多个头进行计算的，这些头的维度比原始的自注意层要小。

### 5.2.8 全连接层

与我们刚刚学习的内容相比，全连接层更加简单。到目前为止，自注意力层产生了一个`n × d[v]`大小的输出（忽略批处理维度）。全连接层将输入数据进行以下转换：

`h[1] = ReLU(xW[1] + b[1])`

这里，`W[1]`是一个`d[v] × d[ff1]`的矩阵，`b[1]`是一个`d[ff1]`大小的向量。因此，这个操作产生一个`n × d[ff1]`大小的张量。结果输出传递到另一层，进行以下计算：

`h[2] = h[1] W[2] + b[2]`

这里`W[2]`是一个`d[ff1] × d[ff2]`大小的矩阵，`b[2]`是一个`d[ff2]`大小的向量。该操作得到一个大小为`n × d[ff2]`的张量。在 TensorFlow 中，我们可以将这些计算再次封装成一个可重用的 Keras 层（见下一个列表）。

**列表 5.3 全连接子层**

```
import tensorflow as tf

class FCLayer(layers.Layer):
    def __init__(self, d1, d2):
        super(FCLayer, self).__init__()
        self.d1 = d1                                                       ❶
        self.d2 = d2                                                       ❷

    def build(self, input_shape):
        self.W1 = self.add_weight(                                         ❸
            shape=(input_shape[-1], self.d1), initializer='glorot_uniform',❸
            trainable=True, dtype='float32'                                ❸
        )
        self.b1 = self.add_weight(                                         ❸
            shape=(self.d1,), initializer='glorot_uniform',                ❸
            trainable=True, dtype='float32'                                ❸
        )        
        self.W2 = self.add_weight(                                         ❸
            shape=(input_shape[-1], self.d2), initializer='glorot_uniform',❸
            trainable=True, dtype='float32'                                ❸
        )
        self.b2 = self.add_weight(                                         ❸
            shape=(self.d2,), initializer='glorot_uniform',                ❸
            trainable=True, dtype='float32'                                ❸
        )  

    def call(self, x):
        ff1 = tf.nn.relu(tf.matmul(x,self.W1)+self.b1)                     ❹
        ff2 = tf.matmul(ff1,self.W2)+self.b2                               ❺
        return ff2
```

❶ 第一个全连接计算的输出维度

❷ 第二个全连接计算的输出维度

❸ 分别定义`W1`、`b1`、`W2`和`b2`。我们使用`glorot_uniform`作为初始化器。

❹ 计算第一个全连接计算

❺ 计算第二个全连接计算

在这里，你可以使用`tensorflow.keras.layers.Dense()`层来实现此功能。然而，我们将使用原始的 TensorFlow 操作进行练习，以熟悉低级 TensorFlow。在这个设置中，我们将改变`FCLayer`，如下面的列表所示。

**列表 5.4 使用 Keras Dense 层实现的全连接层**

```
import tensorflow as tf
import tensorflow.keras.layers as layers

class FCLayer(layers.Layer):

    def __init__(self, d1, d2):
        super(FCLayer, self).__init__()
        self.dense_layer_1 = layer.Dense(d1, activation='relu')  ❶
        self.dense_layer_2 = layers.Dense(d2)                    ❷

    def call(self, x):
        ff1 = self.dense_layer_1(x)                              ❸
        ff2 = self.dense_layer_2(ff1)                            ❹
        return ff2
```

❶ 在子类化层的`__init__`函数中定义第一个全连接层

❷ 定义第二个稠密层。注意我们没有指定激活函数。

❸ 调用第一个稠密层以获取输出

❹ 使用第一个稠密层的输出调用第二个稠密层以获取最终输出

现在你知道了 Transformer 架构中进行的计算以及如何使用 TensorFlow 实现它们。但请记住，原始 Transformer 论文中解释了各种细微的细节，我们还没有讨论。这些细节大多将在后面的章节中讨论。

**练习 2**

假设你被要求尝试一种新型的多头注意力机制。与其将较小头的输出（大小为 64）连接起来，而是将输出（大小为 512）相加。使用`SelfAttentionLayer`编写 TensorFlow 代码以实现此效果。您可以使用`tf.math.add_n()`函数按元素对张量列表求和。

### 5.2.9 将所有内容放在一起

让我们将所有这些元素放在一起创建一个 Transformer 网络。首先让我们创建一个编码器层，其中包含一组`SelfAttentionLayer`对象（每个头一个）和一个`FCLayer`（请参阅下一个列表）。

**列表 5.5 编码器层**

```
import tensorflow as tf

class EncoderLayer(layers.Layer):

    def __init__(self, d, n_heads):
        super(EncoderLayer, self).__init__()
        self.d = d
        self.d_head = int(d/n_heads) 
        self.n_heads = n_heads
        self.attn_heads = [
            SelfAttentionLayer(self.d_head) for i in range(self.n_heads)
        ]                                           ❶
        self.fc_layer = FCLayer(2048, self.d)       ❷

    def call(self, x):
        def compute_multihead_output(x):            ❸
            outputs = [head(x, x, x)[0] for head in self.attn_heads] 
            outputs = tf.concat(outputs, axis=-1)
            return outputs

        h1 = compute_multihead_output(x)            ❹
        y = self.fc_layer(h1)                       ❺

        return y
```

❶ 创建多个注意力头。每个注意力头具有`d/n_heads`大小的特征维度。

❷ 创建完全连接的层，其中中间层有 2,048 个节点，最终子层有`d`个节点。

❸ 创建一个函数，给定一个输入来计算多头注意力输出。

❹ 使用定义的函数计算多头注意力。

❺ 获取层的最终输出。

在初始化`EncoderLayer`时，`EncoderLayer`接受两个参数：`d`（输出的维度）和`n_heads`（注意力头的数量）。然后，在调用层时，传递一个单一的输入`x`。首先计算注意力头（`SelfAttentionLayer`）的关注输出，然后是完全连接层（`FCLayer`）的输出。这就包装了编码器层的关键点。接下来，我们创建一个解码器层（请参阅下一个列表）。

**列表 5.6 解码器层**

```
import tensorflow as tf

class DecoderLayer(layers.Layer):

    def __init__(self, d, n_heads):
        super(DecoderLayer, self).__init__()
        self.d = d
        self.d_head = int(d/n_heads)
        self.dec_attn_heads = [
            SelfAttentionLayer(self.d_head) for i in range(n_heads)
        ]                                                           ❶
        self.attn_heads = [
            SelfAttentionLayer(self.d_head) for i in range(n_heads)
        ]                                                           ❷
        self.fc_layer = FCLayer(2048, self.d)                       ❸

    def call(self, de_x, en_x, mask=None):
        def compute_multihead_output(de_x, en_x, mask=None):        ❹
            outputs = [
                head(en_x, en_x, de_x, mask)[0] for head in 
➥ self.attn_heads]                                                 ❺
            outputs = tf.concat(outputs, axis=-1)
            return outputs

        h1 = compute_multihead_output(de_x, de_x, mask)             ❻
        h2 = compute_multihead_output(h1, en_x)                     ❼
        y = self.fc_layer(h2)                                       ❽
        return y
```

❶ 创建处理解码器输入的注意力头。

❷ 创建同时处理编码器输出和解码器输入的注意力头。

❸ 最终完全连接的子层

❹ 计算多头注意力的函数。此函数接受三个输入（解码器的先前输出、编码器输出和可选的掩码）。

❺ 每个头将函数的第一个参数作为查询和键，并将函数的第二个参数作为值。

❻ 计算第一个受关注的输出。这仅查看解码器输入。

❼ 计算第二个受关注的输出。这将查看先前的解码器输出和编码器输出。

❽ 通过完全连接的子层将输出计算为层的最终输出。

解码器层与编码器层相比有几个不同之处。它包含两个多头注意力层（一个被屏蔽，一个未被屏蔽）和一个全连接层。首先计算第一个多头注意力层（被屏蔽）的输出。请记住，我们会屏蔽任何超出当前已处理的解码器输入的解码器输入。我们使用解码器输入来计算第一个注意力层的输出。然而，第二层中发生的计算有点棘手。做好准备！第二个注意力层将编码器网络的最后一个被关注的输出作为查询和键；然后，为了计算值，使用第一个注意力层的输出。将这一层看作是一个混合器，它混合了被关注的编码器输出和被关注的解码器输入。

有了这个，我们可以用两个编码器层和两个解码器层创建一个简单的 Transformer 模型。我们将使用 Keras 函数式 API（见下一个列表）。

**列表 5.7 完整的 Transformer 模型**
