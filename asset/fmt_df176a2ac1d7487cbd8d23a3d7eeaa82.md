
| 0 | [1, 1, 1, 1, 1, 1, 1] | [0, 10699, 11, 15, 16104, 1388, 2] | [IGN, B-ORG, IGN, I-ORG, I-ORG, I-ORG, IGN] | [0.0, 0.014679872, 0.0, 0.009469474, 0.010393422, 0.01293836, 0.0] | [I-ORG, B-ORG, I-ORG, I-ORG, I-ORG, I-ORG, I-ORG] | [<s>, ▁Ham, a, ▁(, ▁Unternehmen, ▁), </s>] |

每一列都包含一个标记、标签、预测标签等的列表，让我们逐个查看这些标记。`pandas.Series.explode()`函数允许我们在一行中为原始行列表中的每个元素创建一行。由于一行中的所有列表长度相同，我们可以并行进行操作。我们还丢弃了我们命名为`IGN`的填充标记，因为它们的损失无论如何都是零。最后，我们将仍然是`numpy.Array`对象的损失转换为标准浮点数：

```py
df_tokens = df.apply(pd.Series.explode)
df_tokens = df_tokens.query("labels != 'IGN'")
df_tokens["loss"] = df_tokens["loss"].astype(float).round(2)
df_tokens.head(7)
```

| attention_mask | input_ids | labels | loss | predicted_label | input_tokens |
| --- | --- | --- | --- | --- | --- |
| 1 | 10699 | B-ORG | 0.01 | B-ORG | ▁Ham |
| 1 | 15 | I-ORG | 0.01 | I-ORG | ▁( |
| 1 | 16104 | I-ORG | 0.01 | I-ORG | ▁Unternehmen |
| 1 | 1388 | I-ORG | 0.01 | I-ORG | ▁) |
| 1 | 56530 | O | 0.00 | O | ▁WE |
| 1 | 83982 | B-ORG | 0.34 | B-ORG | ▁Luz |
| 1 | 10 | I-ORG | 0.45 | I-ORG | ▁a |

有了这样的数据，我们现在可以按输入标记对其进行分组，并聚合每个标记的损失、计数、均值和总和。最后，我们按损失的总和对聚合数据进行排序，并查看验证集中累积损失最多的标记：

```py
(
    df_tokens.groupby("input_tokens")[["loss"]]
    .agg(["count", "mean", "sum"])
    .droplevel(level=0, axis=1)  # Get rid of multi-level columns
    .sort_values(by="sum", ascending=False)
    .reset_index()
    .round(2)
    .head(10)
    .T
)
```

|  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 输入标记 | ▁ | ▁der | ▁in | ▁von | ▁/ | ▁und | ▁( | ▁) | ▁'' | ▁A |
| 计数 | 6066 | 1388 | 989 | 808 | 163 | 1171 | 246 | 246 | 2898 | 125 |
| 平均值 | 0.03 | 0.1 | 0.14 | 0.14 | 0.64 | 0.08 | 0.3 | 0.29 | 0.02 | 0.44 |
| 总和 | 200.71 | 138.05 | 137.33 | 114.92 | 104.28 | 99.15 | 74.49 | 72.35 | 59.31 | 54.48 |

我们可以在这个列表中观察到几种模式：

+   空格标记具有最高的总损失，这并不奇怪，因为它也是列表中最常见的标记。然而，它的平均损失要低得多。这意味着模型不会在对其进行分类时遇到困难。
+   像“in”、“von”、“der”和“und”这样的词出现相对频繁。它们经常与命名实体一起出现，有时也是它们的一部分，这解释了为什么模型可能会混淆它们。
+   括号、斜杠和单词开头的大写字母很少见，但平均损失相对较高。我们将进一步调查它们。

我们还可以对标签 ID 进行分组，并查看每个类别的损失：

```py
(
    df_tokens.groupby("labels")[["loss"]]
    .agg(["count", "mean", "sum"])
    .droplevel(level=0, axis=1)
    .sort_values(by="mean", ascending=False)
    .reset_index()
    .round(2)
    .T
)
```

|  | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 标签 | B-ORG | I-LOC | I-ORG | B-LOC | B-PER | I-PER | O |
| 计数 | 2683 | 1462 | 3820 | 3172 | 2893 | 4139 | 43648 |
| 平均值 | 0.66 | 0.64 | 0.48 | 0.35 | 0.26 | 0.18 | 0.03 |
| 总和 | 1769.47 | 930.94 | 1850.39 | 1111.03 | 760.56 | 750.91 | 1354.46 |

我们看到`B-ORG`的平均损失最高，这意味着确定组织的开始对我们的模型构成了挑战。

我们可以通过绘制标记分类的混淆矩阵来进一步分解这一点，在那里我们看到组织的开始经常与随后的`I-ORG`标记混淆：

```py
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()
```

```py
plot_confusion_matrix(df_tokens["labels"], df_tokens["predicted_label"],
                      tags.names)
```

![](img/nlpt_04in01.png)

从图中我们可以看出，我们的模型往往最容易混淆`B-ORG`和`I-ORG`实体。否则，它在分类其余实体方面表现相当不错，这可以从混淆矩阵近似对角线的性质中清楚地看出。

现在我们已经在标记级别上检查了错误，让我们继续看一下损失较大的序列。为了进行这种计算，我们将重新审视我们的“未爆炸”的`DataFrame`，并通过对每个标记的损失求和来计算总损失。为此，让我们首先编写一个帮助我们显示带有标签和损失的标记序列的函数：

```py
def get_samples(df):
    for _, row in df.iterrows():
        labels, preds, tokens, losses = [], [], [], []
        for i, mask in enumerate(row["attention_mask"]):
            if i not in {0, len(row["attention_mask"])}:
                labels.append(row["labels"][i])
                preds.append(row["predicted_label"][i])
                tokens.append(row["input_tokens"][i])
                losses.append(f"{row['loss'][i]:.2f}")
        df_tmp = pd.DataFrame({"tokens": tokens, "labels": labels,
                               "preds": preds, "losses": losses}).T
        yield df_tmp

df["total_loss"] = df["loss"].apply(sum)
df_tmp = df.sort_values(by="total_loss", ascending=False).head(3)

for sample in get_samples(df_tmp):
    display(sample)
```

|  | 0 | 1 | 2 | 3 | 4 | ... | 13 | 14 | 15 | 16 | 17 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 标记 | ▁'' | 8 | . | ▁Juli | ▁'' | ... | n | ischen | ▁Gar | de | </s> |
| 标签 | B-ORG | IGN | IGN | I-ORG | I-ORG | ... | IGN | IGN | I-ORG | IGN | IGN |
| 预测 | O | O | O | O | O | ... | I-ORG | I-ORG | I-ORG | I-ORG | O |
| 损失 | 7.89 | 0.00 | 0.00 | 6.88 | 8.05 | ... | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 |

|  | 0 | 1 | 2 | 3 | 4 | ... | 14 | 15 | 16 | 17 | 18 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 标记 | ▁' | ▁'' | ▁Τ | Κ | ▁'' | ... | k | ▁'' | ▁' | ala | </s> |
| 标签 | O | O | O | IGN | O | ... | IGN | I-LOC | I-LOC | IGN | IGN |
| 预测 | O | O | B-ORG | O | O | ... | O | O | O | O | O |
| 损失 | 0.00 | 0.00 | 3.59 | 0.00 | 0.00 | ... | 0.00 | 7.66 | 7.78 | 0.00 | 0.00 |

|  | 0 | 1 | 2 | 3 | 4 | ... | 10 | 11 | 12 | 13 | 14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 标记 | ▁联合 | ▁国家 | ▁多 | 维 | ▁整合 | ... | ▁中央 | ▁非洲 | ▁共和国 | </s> |
| 标签 | B-PER | I-PER | I-PER | IGN | I-PER | ... | I-PER | I-PER | I-PER | I-PER | IGN |
| 预测 | B-ORG | I-ORG | I-ORG | I-ORG | I-ORG | ... | I-ORG | I-ORG | I-ORG | I-ORG | I-ORG |
| 损失 | 6.46 | 5.59 | 5.51 | 0.00 | 5.11 | ... | 4.77 | 5.32 | 5.10 | 4.87 | 0.00 |

显然，这些样本的标签出现了问题；例如，联合国和中非共和国分别被标记为一个人！与此同时，第一个例子中的“8. Juli”被标记为一个组织。原来 PAN-X 数据集的注释是通过自动化过程生成的。这样的注释通常被称为“银标准”（与人工生成的注释的“金标准”相对），并且并不奇怪，自动化方法未能产生合理的标签。事实上，这种失败模式并不是自动方法的特有现象；即使在人类仔细注释数据时，当注释者的注意力分散或者他们简单地误解句子时，也会出现错误。

我们早些时候注意到的另一件事是，括号和斜杠的损失相对较高。让我们看一些带有开括号的序列的例子：

```py
df_tmp = df.loc[df["input_tokens"].apply(lambda x: u"\u2581(" in x)].head(2)
for sample in get_samples(df_tmp):
    display(sample)
```

|  | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| 标记 | ▁Ham | a | ▁( | ▁Unternehmen | ▁) | </s> |
| 标签 | B-ORG | IGN | I-ORG | I-ORG | I-ORG | IGN |
| 预测 | B-ORG | I-ORG | I-ORG | I-ORG | I-ORG | I-ORG |
| 损失 | 0.01 | 0.00 | 0.01 | 0.01 | 0.01 | 0.00 |

|  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 标记 | ▁Kesk | kül | a | ▁( | ▁Mart | na | ▁) | </s> |
| 标签 | B-LOC | IGN | IGN | I-LOC | I-LOC | IGN | I-LOC | IGN |
| 预测 | B-LOC | I-LOC | I-LOC | I-LOC | I-LOC | I-LOC | I-LOC | I-LOC |
| 损失 | 0.02 | 0.00 | 0.00 | 0.01 | 0.01 | 0.00 | 0.01 | 0.00 |

通常情况下，我们不会将括号及其内容包括在命名实体的一部分，但这似乎是自动提取标注文档的方式。在其他例子中，括号中包含地理位置的说明。虽然这确实也是一个位置，但我们可能希望在注释中将其与原始位置断开。这个数据集由不同语言的维基百科文章组成，文章标题通常包含括号中的某种解释。例如，在第一个例子中，括号中的文本表明哈马是一个“Unternehmen”，或者在英语中是公司。当我们推出模型时，了解这些重要细节是很重要的，因为它们可能对模型所属的整个流水线的下游性能产生影响。

通过相对简单的分析，我们已经确定了我们的模型和数据集的一些弱点。在实际用例中，我们将在这一步上进行迭代，清理数据集，重新训练模型，并分析新的错误，直到我们对性能感到满意。

在这里，我们分析了单一语言的错误，但我们也对跨语言的性能感兴趣。在下一节中，我们将进行一些实验，看看 XLM-R 中的跨语言转移效果如何。

# 跨语言转移

现在我们已经在德语上对 XLM-R 进行了微调，我们可以通过`Trainer`的`predict()`方法评估它对其他语言的转移能力。由于我们计划评估多种语言，让我们创建一个简单的函数来为我们执行这些操作：

```py
def get_f1_score(trainer, dataset):
    return trainer.predict(dataset).metrics["test_f1"]
```

我们可以使用这个函数来检查测试集的性能，并在`dict`中跟踪我们的分数：

```py
f1_scores = defaultdict(dict)
f1_scores["de"]["de"] = get_f1_score(trainer, panx_de_encoded["test"])
print(f"F1-score of [de] model on [de] dataset: {f1_scores['de']['de']:.3f}")
```

```py
F1-score of [de] model on [de] dataset: 0.868
```

这对于 NER 任务来说是相当不错的结果。我们的指标大约在 85%左右，我们可以看到模型似乎在`ORG`实体上遇到了最大的困难，可能是因为这些在训练数据中最不常见，而且 XLM-R 的词汇表中许多组织名称都很少见。其他语言呢？为了热身，让我们看看我们在德语上进行微调的模型在法语上的表现如何：

```py
text_fr = "Jeff Dean est informaticien chez Google en Californie"
tag_text(text_fr, tags, trainer.model, xlmr_tokenizer)
```

|  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
