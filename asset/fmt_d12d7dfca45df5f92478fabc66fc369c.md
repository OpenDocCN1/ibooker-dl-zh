
让我们使用更强大（但计算成本更高）的 ESM2 模型，具有 640 维嵌入，并使用`store_sequence_embeddings`函数存储每个分割的嵌入：

```py
model_checkpoint = "facebook/esm2_t30_150M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = EsmModel.from_pretrained(model_checkpoint)

for split, df in sequence_splits.items():
  store_sequence_embeddings(
    sequence_df=df,
    store_prefix=assets(f"proteins/datasets/protein_dataset_{split}"),
    tokenizer=tokenizer,
    model=model,
  )
```

嵌入存储后，我们可以在需要时将其加载回内存。以下是从模型学习的训练数据集的预览：

```py
train_df = load_sequence_embeddings(
  assets("proteins/datasets/protein_dataset_train"),
  model_checkpoint=model_checkpoint,
)

print(train_df)
```

输出：

```py
         EntryID             Sequence  Length  GO:0000166  GO:0000287  ...  \
0     A0A0C4DG62  MAHVGSRKRSRSRSRS...     218           0           0  ...   
1     A0A1B0GTB2  MVITSENDEDRGGQEK...      48           0           0  ...   
2         A0AVI4  MDSPEVTFTLAYLVFA...     362           0           0  ...   
...          ...                  ...     ...         ...         ...  ...   
3571      Q9Y6W5  MPLVTRNIEPRHLCRQ...     498           0           0  ...   
3572      Q9Y6W6  MPPSPLDDRVVVALSR...     482           0           0  ...   
3573      Q9Y6Y9  MLPFLFFSTLFSSIFT...     160           0           0  ...   

        ME:636    ME:637    ME:638    ME:639    ME:640  
0     0.062926  0.040286  0.030008 -0.033614  0.023891  
1     0.129815 -0.044294  0.023842 -0.020635  0.125583  
2     0.153848 -0.075747  0.024440 -0.123321  0.020945  
...        ...       ...       ...       ...       ...  
3571 -0.001535 -0.084161 -0.014317 -0.141801 -0.040719  
3572  0.120192 -0.086032 -0.016481 -0.108710 -0.077937  
3573  0.114847 -0.028570  0.084638  0.038610  0.087047  

[3574 rows x 946 columns]
```

你会注意到一系列标记为`ME:1`到`ME:640`的列。这些代表 ESM2 模型最终层的平均池化隐藏状态——实际上是每个蛋白质序列的固定长度数值摘要。这些嵌入捕获了预训练期间学习的生化结构和信息，并将作为我们分类器的输入特征。

这个数据框成为`convert_to_tfds`函数的输入，我们定义了这个函数，以便更容易为每个分割准备数据集：

```py
import tensorflow as tf

def convert_to_tfds(
  df: pd.DataFrame,
  embeddings_prefix: str = "ME:",
  target_prefix: str = "GO:",
  is_training: bool = False,
  shuffle_buffer: int = 50,
) -> tf.data.Dataset:
  """Convert embedding DataFrame into a TensorFlow dataset."""
  dataset = tf.data.Dataset.from_tensor_slices(
    {
      "embedding": df.filter(regex=f"^{embeddings_prefix}").to_numpy(),
      "target": df.filter(regex=f"^{target_prefix}").to_numpy(),
    }
  )
  if is_training:
    dataset = dataset.shuffle(shuffle_buffer).repeat()
  return dataset
```

现在我们使用我们的`convert_to_tfds`函数从训练数据框构建一个 TensorFlow 兼容的数据集：

```py
train_ds = convert_to_tfds(train_df, is_training=True)
```

从这些数据集中获取一批数据很简单。我们只需将数据集分批，将其转换为 NumPy 迭代器，然后通过调用`next`来检索一批：

```py
batch_size = 32

batch = next(train_ds.batch(batch_size).as_numpy_iterator())
batch["embedding"].shape, batch["target"].shape
```

输出：

```py
((32, 640), (32, 303))
```

这些形状确认了每个输入都是一个 640 维的嵌入向量（来自 ESM2 模型），每个目标是一个 303 维的二进制向量，表示每个分子功能标签的存在或不存在。

## 小贴士

因为训练数据集包括`.repeat()`，它通过循环数据无限期地产生批次。这在训练中很有用，因为我们希望多次循环数据集。相比之下，验证和测试数据集不会被重复——因此它们的批次最终会耗尽，这正是我们在评估期间想要的，因为在评估期间每个示例只应被看到一次。

为了简化数据集设置，我们将整个管道封装到一个单一的辅助函数`build_dataset`中：

```py
def build_dataset(
  store_file_prefix: str, model_checkpoint: str
) -> dict[str, tf.data.Dataset]:
  """Build train/valid/test TensorFlow datasets from stored embeddings."""
  dataset_splits = {}

  for split in ["train", "valid", "test"]:
    dataset_splits[split] = convert_to_tfds(
      df=load_sequence_embeddings(
        store_file_prefix=f"{store_file_prefix}_{split}",
        model_checkpoint=model_checkpoint,
      ),
      is_training=(split == "train"),
    )
  return dataset_splits
```

这个函数从磁盘加载保存的平均嵌入，并为所有三个分割构建`tf.data.Dataset`对象，这些对象已准备好用于训练：

```py
dataset_splits = build_dataset(
  assets("proteins/datasets/protein_dataset"), model_checkpoint=model_checkpoint
)
```

因此，我们现在已经完全预处理了数据，并准备好用于训练模型。

# 训练模型

我们现在将训练一个简单的[Flax](https://oreil.ly/MjH5C)线性模型，基于平均蛋白质嵌入。回想一下，每个蛋白质序列的长度是可变的，但我们已经将它们转换成了固定大小的嵌入。我们的目标是预测每个蛋白质执行的可能分子功能中的哪一个。这是一个*多标签分类*问题，意味着每个蛋白质可能同时与几个功能标签相关联。

在这个设置中，我们将训练一个轻量级的 MLP（多层感知器）——一个具有非线性函数的密集层堆叠。重要的是，我们并没有微调原始的 ESM2 模型：它保持冻结状态，我们的模型只是在它的嵌入之上进行学习。

这里是模型代码：

```py
import flax.linen as nn
from flax.training import train_state

class Model(nn.Module):
  """Simple MLP for protein function prediction."""

  num_targets: int
  dim: int = 256

  @nn.compact
  def __call__(self, x):
    """Apply MLP layers to input features."""
    x = nn.Sequential(
      [
        nn.Dense(self.dim * 2),
        jax.nn.gelu,
        nn.Dense(self.dim),
        jax.nn.gelu,
        nn.Dense(self.num_targets),
      ]
    )(x)
    return x

  def create_train_state(self, rng: jax.Array, dummy_input, tx) -> TrainState:
    """Initialize model parameters and return a training state."""
    variables = self.init(rng, dummy_input)
    return TrainState.create(
      apply_fn=self.apply, params=variables["params"], tx=tx
    )
```

关于这个非常轻量级的模型的一些说明：

+   它使用`nn.Sequential`堆叠层，这对于这个简单的模型来说，保持了定义的清晰和可读性。
+   我们使用 GELU（高斯误差线性单元）激活函数，它是一个平滑的非线性替代品，用于 ReLU。
+   最后一层是一个`nn.Dense`层，将投影到功能标签的数量（`num_targets`）。它返回 logits，而不是概率——因此我们将在损失函数内部应用合适的激活（如 sigmoid），将这些 logits 转换为预测概率。
+   这个模型在 ESM2 嵌入之上冻结——这意味着它不会更新转换器的权重。它只学习将固定嵌入映射到功能标签。这是高效且可解释的，并且在训练期间减少了内存使用。

你可能也注意到了，我们为模型类添加了一个便利函数`create_train_state`，用于创建训练状态。这个函数将模型初始化、参数注册和优化器设置封装成一个单一的`TrainState`对象。这特别有用，因为它允许我们在所有需要的元素——模型、用于形状推断的虚拟输入和优化器配置——都 readily available 的时候构建训练状态。

让我们根据训练数据框中 GO 项列的数量，实例化具有正确数量输出目标的模型：

```py
targets = list(train_df.columns[train_df.columns.str.contains("GO:")])

mlp = Model(num_targets=len(targets))
```

这个模型现在已准备好接受训练，以预测蛋白质参与的分子功能，使用预计算的嵌入作为输入。

## 定义训练循环

当模型和数据集准备就绪后，我们现在可以定义一个函数来执行单个训练步骤。这个步骤包括：

+   模型前向传递
+   计算损失
+   计算梯度
+   使用这些梯度更新模型参数

这里是如何实现的：

```py
@jax.jit
def train_step(state, batch):
  """Run a single training step and update model parameters."""

  def calculate_loss(params):
    """Compute sigmoid cross-entropy loss from logits."""
    logits = state.apply_fn({"params": params}, x=batch["embedding"])
    loss = optax.sigmoid_binary_cross_entropy(logits, batch["target"]).mean()
    return loss

  grad_fn = jax.value_and_grad(calculate_loss, has_aux=False)
  loss, grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  return state, loss
```

在这个设置中：

+   我们使用 sigmoid 激活和二元交叉熵损失，这适用于多标签分类。logits 通过 sigmoid 激活，而不是 softmax——因为我们希望对每个可能的蛋白质功能进行独立的是/否预测。记住，每个蛋白质可能同时具有许多功能。
+   `@jax.jit`编译训练步骤以获得更好的性能。

接下来，让我们实现一些指标来评估模型除了损失之外的表现，使用`sklearn`中的工具：
