
```py
@partial(jax.jit, static_argnums=(3,))
def train_step(
  state: TrainStateWithBatchNorm,
  batch: dict[str, list[Any]],
  rng_dropout: jax.Array,
  num_classes: int,
) -> tuple[TrainStateWithBatchNorm, dict[str, jax.Array]]:
  """Performs a single training step and returns updated state and metrics."""

  def calculate_loss(params, images, labels):
    variables, kwargs = {"params": params}, {"mutable": []}
    if state.batch_stats is not None:
      variables.update({"batch_stats": state.batch_stats})
      kwargs.update({"mutable": ["batch_stats"]})
    logits, updates = state.apply_fn(
      variables,
      x=images,
      is_training=True,
      rngs={"dropout": rng_dropout},
      **kwargs,
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(
      logits, labels
    ).mean()
    return loss, (logits, updates)

  grad_fn = jax.value_and_grad(calculate_loss, has_aux=True)
  (loss, (logits, updates)), grads = grad_fn(
    state.params, batch["images"], batch["labels"]
  )
  state = state.apply_gradients(grads=grads)
  if state.batch_stats is not None:
    state = state.replace(batch_stats=updates["batch_stats"])

  metrics = {
    "loss": loss,
    **compute_metrics(batch["labels"], logits, num_classes),
  }

  return state, metrics
```

在`calculate_loss`内部：

- 从批次中提取图像和标签。
- 模型执行正向传递以计算 logits（每个类别的未归一化预测分数）。
- 使用`optax.softmax_cross_entropy_with_integer_labels`将预测的 logits 与真实标签进行比较。
- 计算平均交叉熵损失——在批次的示例上平均。
- 然后根据这个损失计算梯度，并应用于更新模型的参数。

一个重要的细节：如果模型使用批归一化（如 ResNet 所做的那样），我们还需要更新批归一化层跟踪的运行统计信息。`calculate_loss`函数通过条件性地包括和更新变量集合中的`batch_stats`条目来处理这两种情况——无论是否使用批归一化。

这种结构使得训练步骤通用且可重用，适用于我们定义的所有模型，无论其架构复杂性或归一化层。

### 评估步骤

评估步骤与训练步骤类似，但更简单，因为它不更新模型权重。它的唯一目的是在验证（或测试）批次上计算性能指标。

```py
@partial(jax.jit, static_argnums=(2,))
def eval_step(
  state: TrainStateWithBatchNorm, batch: dict[str, Any], num_classes: int
):
  """Evaluates model performance on a batch and computes metrics."""
  variables, kwargs = {"params": state.params}, {}
  if state.batch_stats is not None:
    variables.update({"batch_stats": state.batch_stats})
    kwargs.update({"mutable": False})

  logits = state.apply_fn(
    variables, x=batch["images"], is_training=False, **kwargs
  )
  loss = optax.softmax_cross_entropy_with_integer_labels(
    logits, batch["labels"]
  ).mean()

  metrics = {
    "loss": loss,
    **compute_metrics(batch["labels"], logits, num_classes),
  }
  return metrics
```

这就是它的工作原理：

- 模型以`is_training=False`运行前向传播，禁用了训练时的行为，如 dropout 和更新批归一化统计信息。
- 如果使用批归一化，存储的`batch_stats`将包含在变量集合中——但它们不会被更新。相反，我们依赖于训练期间收集的运行平均值（激活的平均值和方差的指数移动平均值）。
- 使用与训练期间相同的损失函数将 logits 与真实标签进行比较。
- 除了计算损失之外，该函数还使用`compute_metrics`返回性能指标，包括加权精确度和召回率（稍后解释）。

通过保持评估状态无状态和无副作用——也就是说，避免对模型参数或内部统计信息的任何更新——我们确保验证分数在运行之间保持一致、可靠和可比较。

### 评估指标

在训练期间，我们通常监控损失，但在评估期间，我们还将计算*精确度*和*召回率*，以更全面地了解模型在皮肤病变分类任务上的性能。

让我们先讨论这些指标在二分类情况下的工作方式——例如，区分黑色素瘤和非黑色素瘤——然后推广到我们真正关心的多分类设置：

- *精确度*衡量预测的正例中有多少实际上是正确的（例如黑色素瘤）：

  <mrow><mtext>精确度</mtext> <mo>=</mo> <mfrac><mrow><mtext>真阳性</mtext><mtext>正例</mtext></mrow> <mrow><mtext>真阳性</mtext><mtext>正例</mtext><mtext>+</mtext><mtext>假阳性</mtext><mtext>正例</mtext></mrow></mfrac></mrow>

  高精确度意味着假阳性很少；因此，当模型标记黑色素瘤时，它很可能是正确的。这在医疗环境中很重要，可以避免不必要的后续程序和患者焦虑。

- *召回率*（也称为灵敏度）衡量有多少实际的黑色素瘤病例被正确识别：

  <mrow><mtext>召回率</mtext> <mo>=</mo> <mfrac><mrow><mtext>真正阳性</mtext><mtext>+</mtext><mtext>真正阳性</mtext><mtext>+</mtext><mtext>假阴性</mtext></mrow></mfrac></mrow>

高召回率意味着很少错过黑色素瘤病例：这对于确保早期检测至关重要。在许多临床环境中，召回率被优先考虑，因为错过一个真实病例（即做出错误的阴性预测）比错误地标记一个良性病例更有害。

在我们的案例中，我们处理的是多个类别，而不仅仅是黑色素瘤与非黑色素瘤之间的比较。为了总结所有类别的性能，我们使用*加权精确率*和*加权召回率*。这些指标在计算每类值的同时，还考虑了类别不平衡——具有更多示例的类别对最终分数的贡献更大。

我们将使用加权精确率和召回率作为我们的主要评估指标，并在训练过程中将它们绘制在训练和验证分割上。以下是我们的`compute_metrics`代码：

```py
@partial(jax.jit, static_argnums=(2,))
def compute_metrics(
  y_true: jax.Array, logits: jax.Array, n_labels: jax.Array
) -> dict[str, jax.Array]:
  """Computes weighted precision and recall metrics from logits and labels."""
  y_scores = jax.nn.softmax(logits)
  y_pred = jnp.argmax(y_scores, axis=1)
  metrics = {
    "recall_weighted": recall_score(
      y_true, y_pred, n_labels, average="weighted"
    ),
    "precision_weighted": precision_score(
      y_true, y_pred, n_labels, average="weighted"
    ),
  }
  return metrics
```

此函数使用预测的 logits 来计算 softmax 分数，选择最可能的类别，然后根据真实标签和预测标签计算加权精确率和召回率。

###### 警告

加权指标简洁且有用，但可能会掩盖问题——例如，如果模型在罕见类别上表现不佳，这可能在整体分数中不会明显显示。始终检查*每类指标*，这可以突出可能需要加权某些类别的损失函数或重新采样数据集以更好地平衡类别代表性的弱点。

你还可能在分类任务中遇到 F1 分数。F1 分数是精确率和召回率的调和平均值——它提供了一个平衡两者的单一数字。在这里，我们将坚持只绘制精确率和召回率，因为它们通常是更容易解释的指标，并让你诊断模型是否在假阳性或假阴性方面有更大的困难。

### 更快的评估指标

你可能已经注意到`compute_metrics`函数已经被 JIT 编译。标准的评估指标，例如来自`sklearn`或`scipy`的指标，通常不与 JIT 兼容，这可能会在训练过程中成为瓶颈。

为了解决这个问题，我们实现了 JIT 友好的精确率和召回率版本。这些自定义函数已经过测试，以匹配其标准对应版本的输出，但运行速度显著更快——尤其是在频繁的批量评估期间。

指标是在每个批次上计算的，并使用`MetricsLogger`进行记录，该记录器汇总结果以提供整个数据集的性能摘要。

###### 备注

在大规模生产模型中，非 JIT 指标的额外开销可能可以忽略不计。但对我们这里使用的相对较小的模型和快速的迭代周期来说，JIT 兼容性的加速效果是明显的。

## 创建多类数据集

最后，我们可以使用我们的`DatasetBuilder`构建用于训练和评估本章中所有模型的训练集。在实验中使用相同的训练集设置允许我们进行干净的比较，并将性能差异归因于模型选择而不是数据变化。

我们还提前初始化并分割一个随机种子，这样我们可以在模型初始化和训练运行中重用一致的种子——这有助于确保可重复性。

```py
from dlfb.cancer.dataset.preprocessors import skew

rng = jax.random.PRNGKey(seed=42)
rng, rng_dataset, rng_init, rng_train = jax.random.split(rng, num=4)

dataset_splits = DatasetBuilder(
  data_dir=assets("cancer/datasets/raw"),
).build(
  rng=rng_dataset,
  preprocessors=[skew, crop, resnet],
  image_size=(224, 224, 3),
  splits={"train": 0.70, "valid": 0.20, "test": 0.10},
)

num_classes = dataset_splits["train"].num_classes
```

实现了三种预处理选项：

`crop`

专注于图像的中心区域，病变通常位于此处。虽然简单有效，但它可能会丢弃可能有用的边缘特征，如毛发、纹理或色素变化。

`skew`

通过将整个图像调整到正方形而不裁剪，尽可能保留病变的上下文——这意味着拉伸或压缩原始的宽高比。由于大多数输入图像的宽度大于高度，裁剪到正方形可能会切掉可能有用的周围组织。倾斜通过扭曲图像以适应正方形形状，从而保留所有像素。

虽然图像会略微变形，但它可能仍然有助于模型捕捉更广泛的环境线索，如皮肤纹理或边缘特征。

`resnet`

应用 ResNet 模型所需的标准化预处理，包括调整大小到预期的输入形状和归一化像素值。确保与预训练的 ResNet 架构兼容。

这些预处理步骤是模块化的，可以轻松地替换。我们可以将这些选择之间的选择视为实验中另一个可调的超参数。

## 训练基线模型

让我们先训练基线`SimpleCnn`模型。这里没有什么花哨的——但重点是展示即使是原始架构也能从原始图像中学习到惊人的内容。以下是训练设置：

```py
import optax

from dlfb.cancer.train.handlers.samplers import repeating_sampler
from dlfb.cancer.utils import decay_mask

learning_rate = 0.001
num_steps = 2000

state, metrics = train(
  state=SimpleCnn(num_classes=num_classes).create_train_state(
    rng=rng_init,
    dummy_input=dataset_splits["train"].get_dummy_input(),
    tx=optax.adamw(learning_rate, weight_decay=0.0, mask=decay_mask),
  ),
  rng=rng_train,
  dataset_splits=dataset_splits,
  num_steps=num_steps,
  batch_size=32,
  preprocessor=crop,
  sampler=repeating_sampler,
  augmentor=None,
  eval_every=100,
  store_path=assets("cancer/models/baseline"),
)
```

关于这个“无装饰”设置的几点快速说明：

- 我们使用`optax.adamw`，但将`weight_decay`设置为`0.0`，这使得它等同于普通的`optax.adam`。这意味着没有应用权重衰减正则化（我们也不必担心`mask`参数）。
- 预处理器设置为裁剪。
- 我们正在使用`repeating_sampler`来循环遍历数据。这并不会重新采样图像以平衡类别的频率。
- 没有应用数据增强（`augmentor=None`）。

###### 注意

虽然我们指定了`optax.adamw`优化器，但我们最初没有使用权重衰减。为什么？在所有我们的模型中统一定义它确保了一致性，并使超参数调整变得简单得多。稍后，我们将激活这个占位符以帮助减少模型过拟合。

现在让我们通过绘制其指标随训练步骤的变化来评估模型的性能。结果如图图 5-16 所示。

```py
from dlfb.cancer.inspect import plot_learning

plot_learning(metrics);
```

![](img/dlfb_0516.png)

###### 图 5-16\. `SimpleCnn`模型的评估指标。左图显示了`train`和`valid`数据集分割的损失演变，而右图跟踪了`valid`数据集的（加权）精确率和召回率。

我们看到这个基线模型在验证集上达到了大约 0.4 的加权精确度和召回率——对于这样一个简单的架构来说还不错。这为我们提供了一个参考点，可以与更先进的模型进行比较。

训练指标远高于验证指标，表明过度拟合严重。我们还观察到验证损失随时间增加，这与过度拟合一致。

评估模型性能的另一种有用方法是混淆矩阵，如图 图 5-17 所示。这提供了预测与真实标签对齐的详细视图——包括哪些类别经常被混淆：

```py
from dlfb.cancer.inspect import plot_classified_images
from dlfb.cancer.train import get_predictions

predictions = get_predictions(state, dataset_splits["valid"], resnet, 32)
plot_classified_images(
  predictions, dataset_splits["valid"], crop, max_images=8
);
```

![图片](img/dlfb_0517.png)

###### 图 5-17\. 在九类皮肤病变分类任务上`SimpleCnn`模型的混淆矩阵。行代表真实标签，列代表预测标签。对角线单元格显示正确预测；对角线外的单元格表示错误分类。每个单元格还包括用于定性检查的示例验证图像，并在右下角显示预测计数。每个类别都提供了精确度（P:) 和召回率（R:) 分数。

以下是如何阅读此图的指导——我们将在本章中一直使用：

- 对角线是好的：这些代表被正确分类的图像。一个理想的模型只会对角线上的预测。
- 对角线外的部分是坏的：这些是错误分类。
