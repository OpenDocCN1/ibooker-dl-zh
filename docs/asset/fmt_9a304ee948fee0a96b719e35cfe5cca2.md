
我们将改变`embedding_dim`参数，并为每个值训练新的模型以评估其对性能的影响。由于之前的实验表明更长的训练时间会产生更好的结果，我们还将在这个扫描中增加`epoch`的数量：

```py
embedding_dims = [64, 128, 256, 512]
model_params = {
  "n_nodes": dataset_splits["train"].n_nodes,
  "last_layer_self": False,
  "degree_norm": False,
  "dropout_rate": 0.3,
}
training_params = {
  "rng": rng_train,
  "dataset_splits": dataset_splits,
  "num_epochs": 500,
  "eval_every": 25,
  "loss_fn": auc_loss,
  "norm_loss": True,
}
```

下面的循环自动在一系列`embedding_dim`值上执行训练，并将评估指标存储起来以供后续分析：

```py
from dlfb.utils.metric_plots import to_df

all_metrics = []
for embedding_dim in embedding_dims:
  model = DdiModel(**{"embedding_dim": embedding_dim, **model_params})
  _, metrics = train(
    state=model.create_train_state(
      rng=rng_init,
      dummy_input={
        "graph": dataset_splits["train"].graph,
        "pairs": dataset_splits["train"].pairs.get_dummy_input(),
      },
      tx=optax.adam(0.001),
    ),
    **training_params,
    store_path=assets(f"graphs/models/sweep_embedding_dim:{embedding_dim}"),
  )
  df = to_df(metrics).assign(**{"embedding_dim": embedding_dim})
  all_metrics.append(df)
all_metrics_df = pd.concat(all_metrics, axis=0)
```

如图 4-11 所示，嵌入大小与模型性能之间存在明显的关系：较大的嵌入向量往往会产生更好的验证指标。这可能是由于高维嵌入提供了更大的能力来捕捉图中复杂的关系和模式：

```py
import seaborn as sns
from matplotlib import pyplot as plt

from dlfb.utils.metric_plots import DEFAULT_SPLIT_COLORS

data = all_metrics_df
data = data[(data["metric"] == "hits@20")]
data = data.groupby(["metric", "split", "embedding_dim"], as_index=False)[
  "mean"
].max()
data = data.sort_values(by=["split", "mean"])

plt.figure(figsize=(7, 3.5))
sns.barplot(
  data=data,
  x="embedding_dim",
  y="mean",
  hue="split",
  palette=DEFAULT_SPLIT_COLORS,
)
plt.ylim(0.5, 1)
plt.xlabel("Embedding Dimensions")
plt.ylabel("Maximum Hits@20");
```

![](img/dlfb_0411.png)

###### 图 4-11. 具有不同嵌入维度的模型实现的最大 Hits@20，突出了嵌入大小对性能的影响。

将嵌入维度设置为 512 的模型在训练集上似乎实现了完美的 Hits@20 大约 ~1，这表明它具有足够的容量来完全（甚至过度）拟合训练数据。

###### 注意

增加嵌入大小并延长训练时间，如本节所述，会带来更高的计算成本。这种权衡是否值得取决于可用的资源以及提高模型性能在您特定用例中的相对重要性。

### 调整多个超参数

对于这个实验，我们通过同时扫描多个模型超参数来扩大搜索范围。这基于之前的实验，但探索了超参数空间更广泛的区域，希望发现性能更好的配置。

我们将调整以下参数：

+   Dropout 率：0、0.3 或 0.5
+   最后卷积层中的自环：是否包含自环（`last_layer_self`：`True` 或 `False`）
+   度规范化：是否通过节点嵌入的度数进行规范化（`degree_norm`：`True` 或 `False`）
+   链接预测器中的 MLP 层数：1、2 或 3（`n_mlp_layers`）

对于这个实验，我们将嵌入维度固定在 512，因为它在早期的扫描中是表现最好的设置之一。虽然嵌入大小为 256 也表现相当好，并且可以降低计算成本，但我们选择 512 以期待扩展到更大的图。更大的嵌入大小提供了更大的能力来模拟复杂的关系模式。

让我们设置扫描：

```py
import itertools

model_params = {
  "n_nodes": dataset_splits["train"].n_nodes,
  "embedding_dim": 512,
}

model_params_sweep = {
  "dropout_rate": [0, 0.3, 0.5],
  "last_layer_self": [True, False],
  "degree_norm": [True, False],
  "n_mlp_layers": [1, 2, 3],
}
keys, values = zip(*model_params_sweep.items())
model_param_combn = [
  dict(zip(keys, combo)) for combo in itertools.product(*values)
]
print(pd.DataFrame(model_param_combn))
```

输出：

```py
    dropout_rate  last_layer_self  degree_norm  n_mlp_layers
0   0.0           True             True         1
1   0.0           True             True         2
2   0.0           True             True         3
..  ...           ...              ...          ...
33  0.5           False            False        1
34  0.5           False            False        2
35  0.5           False            False        3

[36 rows x 4 columns]
```

为了训练具有每个参数组合的模型，我们采用以下方法：

```py
def name_from_params(params: dict) -> str:
  """Generates a string from a parameters dictionary"""
  return "_".join([f"{k}:{v}" for k, v in params.items()])

all_metrics = []

for combn in model_param_combn:
  model = DdiModel(**{**combn, **model_params})
  _, metrics = train(
    state=model.create_train_state(
      rng=rng_init,
      dummy_input={
        "graph": dataset_splits["train"].graph,
        "pairs": dataset_splits["train"].pairs.get_dummy_input(),
      },
      tx=optax.adam(0.001),
    ),
    **training_params,
    store_path=assets(f"graphs/models/sweep_all_{name_from_params(combn)}"),
  )
  df = to_df(metrics).assign(**combn)
  all_metrics.append(df)

all_metrics_df = pd.concat(all_metrics, axis=0)
```

与早期的实验类似，这个循环自动化了在所有参数组合上训练模型的过程。评估每个模型的性能，并记录如 Hits@20 等指标以供后续分析。

然后，我们计算每个参数组合和分割的最大 Hits@20 指标，并生成用于编码器中卷积层配置的更易读的表示。这种系统性的探索帮助我们确定优化模型的最有效的超参数组合。

然后，我们提取每个参数组合和分割的最大 Hits@20 指标：

```py
def conv_layer_annot(row):
  if row["last_layer_self"] and row["degree_norm"]:
    return "with self-edges and norm"
  elif row["last_layer_self"]:
    return "with self-edges, no norm"
  elif row["degree_norm"]:
    return "with norm, no self-edges"
  else:
    return "no self-edges and no norm"

data = all_metrics_df
data = data[(data["metric"] == "hits@20")]
data = data.groupby(
  ["metric", "split", *list(model_params_sweep.keys())], as_index=False
)["mean"].max()
data = data.sort_values(by=["split", "mean"])
data["conv_layer"] = data.apply(conv_layer_annot, axis=1)
```

接下来，在图 4-12 中，我们可视化了模型在不同超参数组合下的性能概述。这个图表包含了很多信息，所以让我们首先明确如何解读它。每一对点代表单个模型配置的训练和验证 Hits@20 分数。我们的目标是识别那些验证性能高且与训练性能接近的配置——这表明模型具有良好的泛化能力且没有过拟合。这一点尤为重要，因为数据集是根据蛋白质靶点划分的，这意味着验证集包含的药物靶标与训练过程中看到的靶标不同。

###### 小贴士

在训练和验证集上表现相似的模式更有可能具有良好的泛化能力。优先考虑这些配置可以帮助指导稳健的模型选择。

```py
fig = sns.relplot(
  data=data,
  x="conv_layer",
  y="mean",
  row="dropout_rate",
  col="n_mlp_layers",
  hue="split",
  palette=DEFAULT_SPLIT_COLORS,
  facet_kws=dict(margin_titles=True, despine=False),
  height=2,
)
fig.figure.subplots_adjust(wspace=0.1, hspace=0)
for ax in fig.axes.flat:
  for label in ax.get_xticklabels():
    label.set_rotation(45)
    label.set_ha("right")
fig.set_axis_labels("", "maximum Hits@20");
```

![图片](img/dlfb_0412.png)

###### 图 4-12. 在我们的超参数网格中，每个配置达到的最大 Hits@20 分数。行对应于`dropout_rate`（0.0，0.3，0.5）；列变化了链接预测器中 MLP 层的数量（1-3）。每个 x 轴类别切换自环和度归一化。最佳泛化来自最小模型（最左侧，第一行）。

在这个图表中有很多模型，但左上角的一个模型特别突出：没有 dropout 的最佳性能模型。它使用了一个简单的卷积设置（没有自环，没有归一化）和一个单层链接预测器。值得注意的是，这样一个最小配置表现如此之好，表明简单的模型仍然可以非常有效。

从图表中观察到的其他现象：

+   度归一化几乎在所有地方都会造成损害。两个“带归一化”的列相对于它们的“无归一化”对应列，验证性能都有明显下降。
+   深度 MLP 头部更容易过拟合。从一层到三层的 MLP 层，训练-验证差距扩大，并且很少能产生更高的峰值验证分数。
+   Dropout 并不总是有帮助，尽管一些具有高 dropout 的模型表现良好，这表明它也不是严格有害的。
+   自环几乎没有影响。两个“带自环”的类别与它们的“无自环”双胞胎紧密跟踪，表明这个图已经传达了足够的互反信息。总体而言，任务似乎相当宽容——许多配置在验证中超过了 0.8，但最简单、容量最低的模型仍然是最可靠的选择。

现在，让我们检查最佳性能配置的学习曲线，如图 4-13 所示。

```py
from dlfb.utils.metric_plots import from_df

metrics = all_metrics_df
metrics = from_df(
  metrics[
    (metrics["dropout_rate"] == 0.0)
    & (metrics["last_layer_self"] == False)
    & (metrics["degree_norm"] == False)
    & (metrics["n_mlp_layers"] == 1)
  ]
)

plot_learning(metrics);
```

回想一下，之前的指标代表的是最大值——这意味着它们可能不在训练的同一时间点发生，并且如果继续训练，不能排除最终过拟合的可能性。绘制完整的训练曲线提供了额外的见解。在图 4-13 中，我们看到训练集上有强大的学习效果，但配置似乎容易过拟合，如果没有提前停止。这一点从训练和验证指标随时间拉大的差距中可以看出。

![图片](img/dlfb_0413.png)

###### 图 4-13. 最佳配置的损失（左）和 Hits@20（右）的学习曲线。模型实现了接近完美的训练性能，但验证指标达到平台期后下降，表明过拟合。

让我们再看看图 4-14 中另一个高性能模型的训练曲线。此配置来自超参数空间的不同区域：它在链接预测器中包含更多层，并应用高 dropout 率以鼓励泛化。它还使用了不同的卷积设置。值得注意的是，训练和验证曲线更接近，这表明该模型学习到了更具泛化性的表示，并且不太容易过拟合：

```py
metrics = all_metrics_df
metrics = from_df(
  metrics[
    (metrics["dropout_rate"] == 0.5)
    & metrics["last_layer_self"]
    & (metrics["degree_norm"] == False)
    & (metrics["n_mlp_layers"] == 2)
  ]
)

plot_learning(metrics);
```

![图片](img/dlfb_0414.png)

###### 图 4-14. 另一个高性能模型的损失（左）和 Hits@20（右）的学习曲线。此配置展示了改进的泛化能力，训练和验证指标保持稳定且紧密一致。

虽然只关注顶级指标以识别最佳配置很有吸引力，但检查训练动态可以提供对模型性能的更深入见解。基于最大指标和学习曲线行为，我们推荐这个最新的模型，因为它性能强大且稳定。

###### 注意

有许多其他超参数可以探索——包括我们模型和训练循环中已经存在的，以及尚未考虑的。例如，我们可以尝试不同的负采样策略，改变负对正对的比例，或者增加图卷积层的深度。然而，对于这个较小的数据集，当前的性能已经足够强大，我们认为进一步调整是不必要的。

## 在更大的数据集上训练

最后，我们在一个更大的 DDI 数据集上训练了一个模型——从使用大约十分之一的数据增加到大约一半的数据。

如前所述，最佳模型配置通常取决于数据集大小和图连接性。在这个更大的数据集中，额外的超参数探索表明*度归一化*对于强大性能至关重要。此外，0.3 的适度 dropout 率在此设置中在正则化和学习容量之间取得了最佳平衡。

这是完整的`DdiModel`设置：
