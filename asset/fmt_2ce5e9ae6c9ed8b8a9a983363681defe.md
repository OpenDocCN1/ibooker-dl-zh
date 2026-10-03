
一个小的额外变化：损失函数直接在`train_step`中定义。这通常是为了可读性和封装而做的，尤其是在损失依赖于额外的可变模型状态（如`batch_stats`）或 dropout 时，这两者现在都是前向传递所必需的：

```
@jax.jit
def train_step(state, batch, rng_dropout: jax.Array):
  """Run a training step and update parameters."""

  def calculate_loss(params, batch):
    """Make predictions on batch and compute binary cross-entropy loss."""
    logits, updates = state.apply_fn(
      {"params": params, "batch_stats": state.batch_stats},
      x=batch["sequences"],
      is_training=True,
      rngs={"dropout": rng_dropout},
      mutable=["batch_stats"],
    )

    loss = optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()

    return loss, updates

  grad_fn = jax.value_and_grad(calculate_loss, has_aux=True)
  (loss, updates), grads = grad_fn(state.params, batch)
  state = state.apply_gradients(grads=grads)
  state = state.replace(batch_stats=updates["batch_stats"])

  metrics = {"loss": loss}

  return state, metrics
```

为了确认一切连接正确，我们可以对单个批次进行过拟合——也就是说，在同一个批次上运行几个步骤，并检查损失是否下降：

```
# Overfit on one batch.
for i in range(5):
  rng, rng_dropout = jax.random.split(rng, 2)
  state, metrics = train_step(state, batch, rng_dropout)
  print(f"Step {i} loss: {metrics['loss']}")
```

输出：

```
Step 0 loss: 0.6932974457740784
Step 1 loss: 0.25415313243865967
Step 2 loss: 0.08513301610946655
Step 3 loss: 0.02221144177019596
Step 4 loss: 0.019464049488306046
```

看起来不错——损失迅速下降。这表明模型能够拟合数据。

然而，仅损失并不是评估分类模型的最佳方式。我们希望监控额外的指标，如准确率和 ROC 曲线下的面积（auROC）。我们在评估步骤中计算这些指标：

```
def eval_step(state, batch):
  """Evaluate model on a single batch."""
  logits = state.apply_fn(
    {"params": state.params, "batch_stats": state.batch_stats},
    x=batch["sequences"],
    is_training=False,
    mutable=False,
  )
  loss = optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()
  metrics = {
    "loss": loss.item(),
    **compute_metrics(batch["labels"], logits),
  }
  return metrics

def compute_metrics(y_true: np.ndarray, logits: np.ndarray):
  """Compute accuracy and auROC for model predictions."""
  metrics = {
    "accuracy": accuracy_score(y_true, nn.sigmoid(logits) >= 0.5),
    "auc": roc_auc_score(y_true, logits).item(),
  }
  return metrics
```

注意，由于`scikit-learn`函数与 JAX 不兼容，评估步骤没有使用`@jax.jit`装饰。

让我们看看运行`eval_step`的输出：

```
# Evaluate the batch.
metrics = eval_step(state, batch)
print(metrics)
```

输出：

```
{'loss': 0.45877009630203247, 'accuracy': 0.9375, 'auc': 1.0}
```

现在我们已经实现了训练和评估步骤，让我们定义一个完整的训练循环。`train`函数接受初始化的模型状态、训练和验证数据集以及一些配置参数：

```
@restorable
def train(
  state: TrainStateWithBatchNorm,
  rng: jax.Array,
  dataset_splits: dict[str, tf.data.Dataset],
  num_steps: int,
  eval_every: int = 100,
) -> tuple[TrainStateWithBatchNorm, Any]:
  """Train a model and log metrics over steps."""
  metrics = MetricsLogger()
  train_batches = dataset_splits["train"].as_numpy_iterator()

  steps = tqdm(range(num_steps))  # Steps with progress bar.
  for step in steps:
    steps.set_description(f"Step {step + 1}")

    rng, rng_dropout = jax.random.split(rng, 2)
    train_batch = next(train_batches)
    state, batch_metrics = train_step(state, train_batch, rng_dropout)
    metrics.log_step(split="train", **batch_metrics)

    if step % eval_every == 0:
      for eval_batch in dataset_splits["valid"].as_numpy_iterator():
        batch_metrics = eval_step(state, eval_batch)
        metrics.log_step(split="valid", **batch_metrics)
      metrics.flush(step=step)

    steps.set_postfix_str(metrics.latest(["loss"]))

  return state, metrics.export()
```

训练循环与之前相当相似，但作为一个快速回顾，以下是它所做的工作：

+   使用进度条遍历训练步骤
+   运行`train_step`来更新模型参数
+   定期使用`eval_step`在验证集上进行评估
+   使用自定义`MetricsLogger`在每个步骤记录和存储指标

由于我们为每个 TF 训练一个模型，我们将使用一个实用函数来加载按 TF 名称分割的数据集：

```
def load_dataset_splits(
  path, transcription_factor, batch_size: int | None = None
):
  """Load TF dataset splits (train, valid, test) as TensorFlow datasets."""
  dataset_splits = {}
  for split in ["train", "valid", "test"]:
    dataset = load_dataset(
      sequence_db=f"{path}/{transcription_factor}_{split}_sequences.csv"
    )
    ds = convert_to_tfds(dataset, batch_size, is_training=(split == "train"))
    dataset_splits.update({split: ds})
  return dataset_splits
```

此函数加载给定`transcription_factor`的训练、验证和测试分割，并将它们转换为 TensorFlow 数据集，以便进行批处理。

现在我们已经为每个 TF 训练一个模型做好了所有准备：

```
prefix = assets("dna/datasets")
tf_metrics = {}

# Train one model per transcription factor.
for transcription_factor in transcription_factors:
  # Load data for this TF.
  dataset_splits = load_dataset_splits(
    assets("dna/datasets"), transcription_factor, batch_size
  )
  rng = jax.random.PRNGKey(42)
  rng, rng_init, rng_train = jax.random.split(rng, 3)
  dummy_batch = next(dataset_splits["train"].as_numpy_iterator())["sequences"]

  # Create train state.
  state = ConvModelV2().create_train_state(
    rng=rng_init,
    dummy_input=dummy_batch,
    tx=optax.adam(scheduler),
  )

  # Train the model.
  _, metrics = train(
    state=state,
    rng=rng_train,
    dataset_splits=dataset_splits,
    num_steps=num_steps,
    eval_every=100,
    store_path=assets(f"dna/models/{transcription_factor}"),
  )

  # Store metrics.
  tf_metrics.update({transcription_factor: metrics})
```

训练完成后，让我们看看图 3-17 中仅 CTCF 转录因子性能的情况：

```
from dlfb.dna.inspect import plot_learning

tf = "CTCF"
plot_learning(tf_metrics[tf], tf);
```

![](img/dlfb_0317.png)

###### 图 3-17\. CTCF 转录因子的学习曲线。左侧面板显示了训练和验证损失随时间的变化。右侧面板跟踪使用验证集准确率和 auROC 的模型分类性能。性能在训练过程中稳步提高并平滑收敛。

我们的模型在预测 CTCF 是否结合给定的 DNA 序列方面表现良好，训练和验证损失下降，验证 auROC 接近完美的 1.0 分——这与 CTCF 相对于其他一些 TFs 来说更容易预测的事实一致。

现在，让我们可视化所有 10 个 TFs 的训练曲线。我们首先处理记录的指标：

```
from dlfb.utils.metric_plots import to_df

# Extract metrics logged per transcription factor.
tf_df = []
for tf, metrics in tf_metrics.items():
  tf_df.append(to_df(metrics).assign(TF=tf))
tf_df = pd.concat(tf_df)

# Determine order of best performance.
auc_df = tf_df[(tf_df["metric"] == "auc") & (tf_df["split"] == "valid")]
max_auc_by_tf = auc_df.groupby("TF")["mean"].max()
tf_order = max_auc_by_tf.sort_values(ascending=False).index.tolist()
tf_df["TF"] = pd.Categorical(tf_df["TF"], categories=tf_order, ordered=True)
```

然后，我们可以按 auROC 性能（最佳性能的 TFs 排在前面）排序，在图 3-18 中可视化它们的学习曲线：

```
sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 2.5})
sns.set_style("ticks", {"axes.grid": True})
g = sns.relplot(
  data=tf_df,
  x="round",
  y="mean",
  hue="split",
  style="metric",
  kind="line",
  col="TF",
  col_order=tf_order,
  col_wrap=4,
  alpha=0.8,
  palette=DEFAULT_SPLIT_COLORS,
  dashes=True,
)
g.set_axis_labels("Step", "Value")
g.set(ylim=(0, 1));
```

![](img/dlfb_0318.png)

###### 图 3-18\. 所有 10 个 TFs 的学习曲线。每个面板显示了训练步骤中的训练和验证损失（实线）、验证准确率（虚线）和验证 auROC（点线）。TFs 按峰值 auROC 性能排序。虽然一些 TFs，如 CTCF 和 ATF2，可以快速达到接近完美的性能，但其他如 ZNF24 和 BACH1 则更难以建模。

这些不同的 TF 结合模型是否做得很好？知道什么构成了“足够好”的 auROC 可能很困难，特别是在生物环境中，标签噪声和数据集复杂性可能差异很大。幸运的是，由于我们的数据集来自已发表的基准，我们可以直接将我们的模型结果与原始论文中报告的结果进行比较。

在[源论文的图 3](https://oreil.ly/QciKO)中，我们可以看到在不同 TFs（转录因子）之间，auROC 分数差异很大，有些甚至在使用来自更高级架构（如使用大规模预训练的基因组语言模型（gLMs））的信息时也难以准确预测。论文中的图表有助于为基于 one-hot 编码 DNA 训练的基线 CNNs 建立性能上限。有趣的是，论文指出，预训练的 gLM 表示并不总是优于使用 one-hot 输入的传统方法，这表明简单模型在这些 TF 结合任务上仍然具有竞争力。

以下是从论文图表中得出的关键要点：

+   CTCF 和 ATF2 是最可预测的转录因子，一热编码和预训练模型都实现了超过 0.95 的 auROC 分数。这些转录因子具有强大的保守结合基序，易于模型学习。
+   REST、MAX 和 ELK1 显示中等难度，auROC 在 0.83 到 0.85 之间。
+   ZNF24、BACH1、ARID3 和 GABPA 通常更难预测，其 auROC 值在 0.75 到 0.80 之间徘徊。

让我们现在看看我们的模型是如何比较的。以下打印出我们为每个 TF 独立训练的 CNN 达到的峰值验证 auROC：

```
print(max_auc_by_tf.sort_values(ascending=False))
```

输出：

```
TF
ATF2     0.985182
CTCF     0.980479
SRF      0.849944
           ...   
ARID3    0.769149
BACH1    0.767122
ZNF24    0.761506
Name: mean, Length: 10, dtype: float64
```

我们的结果与论文中的排名和分数非常接近——特别是在哪些转录因子更容易或更难预测方面。这种一致性提供了令人放心的外部验证，表明我们的训练设置正在正常工作，我们的模型正在学习有意义的内容。

让我们现在探讨如何通过更表达性的架构进一步推动我们的 TF 结合预测结果。

# 高级技术

在我们介绍更复杂的模型组件之前，让我们首先通过将其重构为可重用构建块来提高我们模型架构的清晰度和模块化。这使得我们的代码更容易阅读、扩展和维护。

我们将通过创建两个辅助模块`ConvBlock`和`MLPBlock`来模块化我们的卷积和 MLP 层：
