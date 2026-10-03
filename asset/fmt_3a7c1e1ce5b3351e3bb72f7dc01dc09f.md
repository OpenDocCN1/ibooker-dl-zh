
`fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 4))`

`sns.heatmap(ax=ax[0], data=valid_true_df, yticklabels=False, xticklabels=False, cmap="flare")`
`ax[0].set_title("True functional annotations by protein.")`
`ax[0].set_xlabel("Functional category")`

`sns.heatmap(ax=ax[1], data=valid_prob_df, yticklabels=False, xticklabels=False, cmap="flare")`
`ax[1].set_title("Predicted functional annotations by protein.")`
`ax[1].set_xlabel("Functional category");`

![图片](img/dlfb_0214.png)

###### 图 2-14. 蛋白质功能预测的热图概述。左侧面板显示了验证集中每个蛋白质的真实功能注释，而右侧面板显示了模型预测的概率。这两个矩阵都是稀疏的，垂直带反映了常见功能标签。

这种可视化相当放大且高层次，但它有助于建立对整体模型行为的直觉：

- 一些蛋白质功能在数据集中出现频率较高（表现为垂直条纹），模型倾向于对这些功能进行相对较好的预测。
- 稀有功能更难捕捉——模型经常完全错过它们，导致预测热图中出现稀疏或空列。
- 一些函数被过度预测，表现为许多蛋白质中的微弱垂直线，这表明模型对这些类别过于自信。
- 预测矩阵中的许多单元格显示中间色调，这反映了更不确定的概率（不是自信的接近 0 或接近 1）。

我们现在将从这个定性观点转向定量观点，通过评估每个蛋白质功能的模型性能来进行：

```py
metrics_by_function = {}
for function in targets:
  metrics_by_function[function] = compute_metrics(
    valid_true_df[function].values, valid_prob_df[function].values
  )

overview_valid = (
  pd.DataFrame(metrics_by_function)
  .T.merge(go_term_descriptions, left_index=True, right_on="term")
  .set_index("term")
  .sort_values("auprc", ascending=False)
)
print(overview_valid)
```

输出：

```py
            accuracy    recall  precision     auprc     auroc  \
term                                                            
GO:0004930  0.958858  0.000000   0.000000  0.948591  0.982272   
GO:0004888  0.945424  0.177215   1.000000  0.849885  0.968354   
GO:0003824  0.848027  0.731591   0.819149  0.849362  0.909372   
...              ...       ...        ...       ...       ...   
GO:0003774  0.000000  0.000000   0.000000  0.000000  0.000000   
GO:0051015  0.000000  0.000000   0.000000  0.000000  0.000000   
GO:1902936  0.000000  0.000000   0.000000  0.000000  0.000000   

                    description  
term                             
GO:0004930  G protein-couple...  
GO:0004888  transmembrane si...  
GO:0003824   catalytic activity  
...                         ...  
GO:0003774  cytoskeletal mot...  
GO:0051015  actin filament b...  
GO:1902936  phosphatidylinos...  

[303 rows x 6 columns]
```

这项分析揭示了模型在蛋白质功能上的性能存在很大差异。例如，模型在`GO:0004930`（G 蛋白偶联受体活性）等函数上表现良好，但在其他函数上，如`GO:0003774`（细胞骨架马达活性）上则表现不佳。然而，解释这些结果需要谨慎：一些指标可能基于非常少的验证示例，并且对于在训练过程中代表性不足的函数，性能自然会有限。频繁函数的高分可能仅仅反映了充足的训练数据，而罕见函数的低分可能是预期的。

让我们更仔细地看看蛋白质功能在训练数据中出现的频率与模型在验证集中预测它的好坏之间是否存在关系：

```py
# Compute number of occurrences of each function in the training set.
overview_valid = overview_valid.merge(
  pd.DataFrame(train_df[targets].sum(), columns=["train_n"]),
  left_index=True,
  right_index=True,
)
print(overview_valid)
```

输出：

```py
            accuracy    recall  precision     auprc     auroc  \
GO:0004930  0.958858  0.000000   0.000000  0.948591  0.982272   
GO:0004888  0.945424  0.177215   1.000000  0.849885  0.968354   
GO:0003824  0.848027  0.731591   0.819149  0.849362  0.909372   
...              ...       ...        ...       ...       ...   
GO:0003774  0.000000  0.000000   0.000000  0.000000  0.000000   
GO:0051015  0.000000  0.000000   0.000000  0.000000  0.000000   
GO:1902936  0.000000  0.000000   0.000000  0.000000  0.000000   

                    description  train_n  
GO:0004930  G protein-couple...      138  
GO:0004888  transmembrane si...      228  
GO:0003824   catalytic activity     1210  
...                         ...      ...  
GO:0003774  cytoskeletal mot...        5  
GO:0051015  actin filament b...       17  
GO:1902936  phosphatidylinos...       18  

[303 rows x 7 columns]
```

乍一看，似乎预测性能更高的函数（例如，更高的 auPRC）也往往有更多的训练示例。在图 2-15 中，我们通过散点图更清楚地可视化这种关系：

```py
fig = sns.scatterplot(
  x="train_n", y="auprc", data=overview_valid, alpha=0.5, s=50, color="grey"
)
fig.set_xlabel("# Train instances")
fig.set_ylabel("Validation auPRC");
```

![图片](img/dlfb_0215.png)

###### 图 2-15. 蛋白质功能训练频率与预测性能（auPRC）之间的关系。训练集中常见的功能往往被模型更准确地预测。

这个图显示了明显的趋势：在训练集中出现频率更高的蛋白质功能，在验证集中被模型更准确地预测（以 auPRC 衡量）。这与预期相符——机器学习模型通常在代表性良好的类别上表现更好。这也突出了类别不平衡的挑战：罕见的功能往往预测不佳，这不一定是因为生物复杂性，而是因为模型学习的数据有限。

但我们如何知道特定的 auPRC 分数是否真的很好？例如，对于某种蛋白质功能，auPRC 值为 0.8 可能听起来很有希望——但这是否比随机更好？它是否有意义？为了解释这些分数，我们需要一些东西来比较它们。

## 评估模型的有用性

为了使我们的评估有据可依，我们将把我们的模型与两个简单的基线进行比较：

**抛硬币**

对于每个蛋白质功能，以相等的概率随机预测 0 或 1。这为我们提供了一个完全无知的基础。

**比例猜测**

对于每个函数，预测其出现概率等于其在训练集中的频率。这反映了先验的类别分布知识，但没有任何学习。

这些基线有助于将模型的性能置于上下文中。如果我们的训练模型没有优于这些简单的启发式方法，这可能是一个迹象，表明它可能没有从数据中学习到有意义的结构。

这里是基线的实现：

```py
def make_coin_flip_predictions(
  valid_true_df: pd.DataFrame, targets: list[str]
) -> pd.DataFrame:
  """Make random coin flip predictions for each protein function."""
  predictions = np.random.choice([0.0, 1.0], size=valid_true_df.shape)
  return pd.DataFrame(predictions, columns=targets, index=valid_true_df.index)

def make_proportional_predictions(
  valid_true_df: pd.DataFrame, train_df: pd.DataFrame, targets: list[str]
) -> pd.DataFrame:
  """Make random protein function predictions proportional to frequency."""
  percent_1_train = dict(train_df[targets].mean())
  proportional_preds = []
  for target_column in targets:
    prob_1 = percent_1_train[target_column]
    prob_0 = 1 - prob_1
    proportional_preds.append(
      np.random.choice([0.0, 1.0], size=len(valid_true_df), p=[prob_0, prob_1])
    )
  return pd.DataFrame(
    np.stack(proportional_preds).T, columns=targets, index=valid_true_df.index
  )
```

这些基线应该给我们提供简单但信息丰富的参考点。现在，让我们将这些预测方法应用于我们的训练模型：

```py
prediction_methods = {
  "coin_flip_baseline": make_coin_flip_predictions(valid_true_df, targets),
  "proportional_guess_baseline": make_proportional_predictions(
    valid_true_df, train_df, targets
  ),
  "model": valid_prob_df,
}
```

现在，让我们以与我们的模型完全相同的方式评估基线——通过计算每个蛋白质的指标并取平均值：

```py
metrics_by_method = {}
for method, preds_df in prediction_methods.items():
  metrics_by_method[method] = pd.DataFrame(
    [
      compute_metrics(valid_true_df.iloc[i], preds_df.iloc[i])
      for i in range(len(valid_true_df))
    ]
  ).mean()

print(pd.DataFrame(metrics_by_method))
```

输出：

```py
           coin_flip_baseline  proportional_guess_baseline     model
accuracy             0.500916             0.956447          0.978569
recall               0.499229             0.093555          0.128532
precision            0.023883             0.079994          0.424301
auprc                0.025307             0.039701          0.412350
auroc                0.500027             0.535605          0.882679
```

我们的模型在所有指标上都明显优于两个基线——特别是在精确度、auPRC 和 auROC 方面。这是预期的，因为训练模型利用实际的序列特征做出更明智的预测。如前所述，准确性在这个设置中不是一个可靠的指标，即使是简单的比例猜测也由于类别不平衡而达到欺骗性的高准确性。

模型的性能提升主要来自精确度的显著提高，而召回率的提升则更为适度。这意味着当模型做出预测时，它擅长正确识别正例，但它往往错过许多真实正例——它谨慎且倾向于预测“无功能”。

这突出了一个关键权衡：模型保守但准确。根据您的应用，您可能希望调整这种行为——例如，通过降低决策阈值以提高召回率，如前所述。

接下来，我们将通过单个蛋白质功能分解模型的优缺点，并与两个基线进行比较。这使我们能够看到模型预测得好的具体功能，以及它遇到困难的地方：

```py
auprc_by_function = {}

for method, preds_df in prediction_methods.items():
  metrics_by_function = {}

  for function in targets:
    metrics_by_function[function] = compute_metrics(
      valid_true_df[function], preds_df[function]
    )

  auprc_by_function[method] = (
    pd.DataFrame(metrics_by_function)
    .T.merge(go_term_descriptions, left_index=True, right_on="term")
    .set_index("term")
    .sort_values("auprc", ascending=False)
  )["auprc"].to_dict()
```

在图 2-16 中，我们将函数级别的 auPRC 分数可视化为条形图，以突出模型处理最好的功能类别：

```py
best_performing = (
  pd.DataFrame(auprc_by_function)
  .merge(go_term_descriptions, left_index=True, right_on="term")
  .set_index("term")
  .sort_values("model", ascending=False)
  .head(20)
  .melt("description")
)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(
  x="description",
  y="value",
  hue="variable",
  data=best_performing,
)
ax.set_title("The model's 20 best performing protein functions")
ax.set_ylabel("Validation auPRC")
plt.xticks(rotation=90);
```

![](img/dlfb_0216.png)

###### 图 2-16. 模型在验证集上按 auPRC 排名的前 20 个蛋白质功能。条形图显示了模型实现的 auPRC，与两个简单基线（即抛硬币和比例猜测）进行比较。
