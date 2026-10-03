
```py
from transformers import pipeline

pipe = pipeline("fill-mask", model="bert-base-uncased")
```

接下来，让我们构建一个小电影描述，并在其中添加一个带有屏蔽词的提示。提示的目标是引导模型帮助我们进行分类。`fill-mask`管道返回填充屏蔽位置的最有可能的标记：

```py
movie_desc = "The main characters of the movie madacascar \
are a lion, a zebra, a giraffe, and a hippo. "
prompt = "The movie is about [MASK]."

output = pipe(movie_desc + prompt)
for element in output:
    print(f"Token {element['token_str']}:\t{element['score']:.3f}%")
```

```py
Token animals:  0.103%
Token lions:    0.066%
Token birds:    0.025%
Token love:     0.015%
Token hunting:  0.013%
```

显然，模型只预测与动物相关的标记。我们也可以反过来，而不是获取最有可能的标记，我们可以查询管道获取几个给定标记的概率。对于这个任务，我们可以选择`cars`和`animals`，所以我们可以将它们作为目标传递给管道：

```py
output = pipe(movie_desc + prompt, targets=["animals", "cars"])
for element in output:
    print(f"Token {element['token_str']}:\t{element['score']:.3f}%")
```

```py
Token animals:  0.103%
Token cars:     0.001%
```

毫不奇怪，对于标记为`cars`的预测概率要远远小于`animals`。让我们看看这是否也适用于更接近汽车的描述：

```py
movie_desc = "In the movie transformers aliens \
can morph into a wide range of vehicles."

output = pipe(movie_desc + prompt, targets=["animals", "cars"])
for element in output:
    print(f"Token {element['token_str']}:\t{element['score']:.3f}%")
```

```py
Token cars:     0.139%
Token animals:  0.006%
```

它确实可以！这只是一个简单的例子，如果我们想确保它运行良好，我们应该进行彻底的测试，但它说明了本章讨论的许多方法的关键思想：找到一种方法，使预训练模型适应另一个任务，而无需对其进行训练。在这种情况下，我们设置了一个提示，其中包含一个掩码，以便我们可以直接使用掩码语言模型进行分类。让我们看看是否可以通过调整一个在更接近文本分类的任务上进行了微调的模型来做得更好：*自然语言推理*（NLI）。

使用掩码语言模型进行分类是一个不错的技巧，但是我们可以通过使用一个在更接近分类的任务上训练过的模型来做得更好。有一个称为*文本蕴涵*的巧妙代理任务符合要求。在文本蕴涵中，模型需要确定两个文本段落是否可能相互跟随或相互矛盾。模型通常是使用诸如多种体裁 NLI 语料库（MNLI）或跨语言 NLI 语料库（XNLI）等数据集进行蕴涵和矛盾的检测。³

这些数据集中的每个样本由三部分组成：前提、假设和标签，标签可以是`蕴涵`、`中性`或`矛盾`中的一个。当假设文本在前提下必然为真时，分配`蕴涵`标签。当假设在前提下必然为假或不合适时，使用`矛盾`标签。如果这两种情况都不适用，则分配`中性`标签。参见表 9-1 中的示例。

**表 9-1。MLNI 数据集中的三个类**

| 前提 | 假设 | 标签 |
| --- | --- | --- |
| 他最喜欢的颜色是蓝色。 | 他喜欢重金属音乐。 | `中性` |
| 她觉得这个笑话很搞笑。 | 她认为这个笑话一点都不好笑。 | `矛盾` |
| 这所房子最近建造。 | 这所房子是新的。 | `蕴涵` |

现在，事实证明我们可以劫持一个在 MNLI 数据集上训练的模型，构建一个分类器而无需任何标签！关键思想是将我们希望分类的文本视为前提，然后将假设制定为：

> “这个例子是关于{label}的。”

在这里我们插入标签的类名。蕴涵分数告诉我们前提很可能是关于那个主题的，我们可以依次为任意数量的类运行这个。这种方法的缺点是我们需要为每个类执行一次前向传播，这使得它比标准分类器效率低。另一个稍微棘手的方面是标签名称的选择可能对准确性产生很大影响，通常最好选择具有语义含义的标签。例如，如果标签只是`Class 1`，模型就不知道这可能意味着什么，以及这是否构成了矛盾或蕴涵。

![nlpt_pin01](img/nlpt_pin01.png) Transformers 具有内置的零样本分类 MNLI 模型。我们可以通过管道初始化它如下：

```py
from transformers import pipeline

pipe = pipeline("zero-shot-classification", device=0)
```

设置`device=0`确保模型在 GPU 上运行，而不是默认的 CPU，以加快推理速度。要对文本进行分类，我们只需要将其传递给管道，并附上标签名称。此外，我们可以设置`multi_label=True`以确保返回所有分数，而不仅仅是单标签分类的最大值：

```py
sample = ds["train"][0]
print(f"Labels: {sample['labels']}")
output = pipe(sample["text"], all_labels, multi_label=True)
print(output["sequence"][:400])
print("\nPredictions:")

for label, score in zip(output["labels"], output["scores"]):
    print(f"{label}, {score:.2f}")
```

```py
Labels: ['new model']
Add new CANINE model

#  New model addition

## Model description

Google recently proposed a new **C**haracter **A**rchitecture with **N**o
tokenization **I**n **N**eural **E**ncoders architecture (CANINE). Not only the
title is exciting:

> Pipelined NLP systems have largely been superseded by end-to-end neural
modeling, yet nearly all commonly-used models still require an explicit tokeni

Predictions:
new model, 0.98
tensorflow or tf, 0.37
examples, 0.34
usage, 0.30
pytorch, 0.25
documentation, 0.25
model training, 0.24
tokenization, 0.17
pipeline, 0.16
```

###### 注意

由于我们使用的是子词分词器，我们甚至可以将代码传递给模型！分词可能不太高效，因为零样本管道的预训练数据集只包含了一小部分代码片段，但由于代码也由许多自然词组成，这并不是一个大问题。此外，代码块可能包含重要信息，例如框架（PyTorch 或 TensorFlow）。

我们可以看到模型非常确信这段文本是关于一个新模型的，但它也为其他标签产生了相对较高的分数。零射击分类的一个重要方面是我们所处的领域。我们在这里处理的文本非常技术化，大部分是关于编码的，这使它们与 MNLI 数据集中的原始文本分布相当不同。因此，这对于模型来说是一个具有挑战性的任务并不奇怪；它可能对某些领域的工作效果比其他领域要好得多，这取决于它们与训练数据的接近程度。

让我们编写一个函数，通过零射击管道将单个示例传递，并通过运行`map()`将其扩展到整个验证集：

```py
def zero_shot_pipeline(example):
    output = pipe(example["text"], all_labels, multi_label=True)
    example["predicted_labels"] = output["labels"]
    example["scores"] = output["scores"]
    return example

ds_zero_shot = ds["valid"].map(zero_shot_pipeline)
```

现在我们有了分数，下一步是确定应该为每个示例分配哪组标签。我们可以尝试一些选项：

+   定义一个阈值，并选择高于阈值的所有标签。
+   选择具有最高分数的前*k*个标签。

为了帮助我们确定哪种方法最好，让我们编写一个`get_preds()`函数，应用其中一种方法来获取预测：

```py
def get_preds(example, threshold=None, topk=None):
    preds = []
    if threshold:
        for label, score in zip(example["predicted_labels"], example["scores"]):
            if score >= threshold:
                preds.append(label)
    elif topk:
        for i in range(topk):
            preds.append(example["predicted_labels"][i])
    else:
        raise ValueError("Set either `threshold` or `topk`.")
    return {"pred_label_ids": list(np.squeeze(mlb.transform([preds])))}
```

接下来，让我们编写第二个函数`get_clf_report()`，它从具有预测标签的数据集中返回 Scikit-learn 分类报告：

```py
def get_clf_report(ds):
    y_true = np.array(ds["label_ids"])
    y_pred = np.array(ds["pred_label_ids"])
    return classification_report(
        y_true, y_pred, target_names=mlb.classes_, zero_division=0,
        output_dict=True)
```

有了这两个函数，让我们从增加几个值的*k*开始，然后绘制验证集上的微观和宏观* F * [1]-分数：

```py
macros, micros = [], []
topks = [1, 2, 3, 4]
for topk in topks:
    ds_zero_shot = ds_zero_shot.map(get_preds, batched=False,
                                    fn_kwargs={'topk': topk})
    clf_report = get_clf_report(ds_zero_shot)
    micros.append(clf_report['micro avg']['f1-score'])
    macros.append(clf_report['macro avg']['f1-score'])
```

```py
plt.plot(topks, micros, label='Micro F1')
plt.plot(topks, macros, label='Macro F1')
plt.xlabel("Top-k")
plt.ylabel("F1-score")
plt.legend(loc='best')
plt.show()
```

![](img/nlpt_09in03.png)

从图中我们可以看到，通过选择每个示例的最高分数的标签（top 1）获得了最佳结果。这也许并不奇怪，因为我们数据集中的大多数示例只有一个标签。现在让我们将其与设置阈值进行比较，这样我们可能可以对每个示例进行多个标签的预测：

```py
macros, micros = [], []
thresholds = np.linspace(0.01, 1, 100)
for threshold in thresholds:
    ds_zero_shot = ds_zero_shot.map(get_preds,
                                    fn_kwargs={"threshold": threshold})
    clf_report = get_clf_report(ds_zero_shot)
    micros.append(clf_report["micro avg"]["f1-score"])
    macros.append(clf_report["macro avg"]["f1-score"])
```

```py
plt.plot(thresholds, micros, label="Micro F1")
plt.plot(thresholds, macros, label="Macro F1")
plt.xlabel("Threshold")
plt.ylabel("F1-score")
plt.legend(loc="best")
plt.show()
```

![](img/nlpt_09in04.png)

```py
best_t, best_micro = thresholds[np.argmax(micros)], np.max(micros)
print(f'Best threshold (micro): {best_t} with F1-score {best_micro:.2f}.')
best_t, best_macro = thresholds[np.argmax(macros)], np.max(macros)
print(f'Best threshold (micro): {best_t} with F1-score {best_macro:.2f}.')
```

```py
Best threshold (micro): 0.75 with F1-score 0.46.
Best threshold (micro): 0.72 with F1-score 0.42.
```

这种方法的表现略逊于 top-1 的结果，但我们可以在这张图中清楚地看到精确度/召回率的权衡。如果我们将阈值设置得太低，那么会有太多的预测，这会导致精确度低。如果我们将阈值设置得太高，那么我们几乎不会进行任何预测，这会产生低召回率。从图中我们可以看到，阈值约为 0.8 是两者之间的最佳平衡点。

由于 top-1 方法表现最佳，让我们使用它来将零射击分类与朴素贝叶斯在测试集上进行比较：

```py
ds_zero_shot = ds['test'].map(zero_shot_pipeline)
ds_zero_shot = ds_zero_shot.map(get_preds, fn_kwargs={'topk': 1})
clf_report = get_clf_report(ds_zero_shot)
for train_slice in train_slices:
    macro_scores['Zero Shot'].append(clf_report['macro avg']['f1-score'])
    micro_scores['Zero Shot'].append(clf_report['micro avg']['f1-score'])
```

```py
plot_metrics(micro_scores, macro_scores, train_samples, "Zero Shot")
```

![](img/nlpt_09in05.png)

将零射击管道与基线进行比较，我们观察到两件事：

1.  如果我们少于 50 个有标签的样本，零射击管道将轻松胜过基线。
2.  即使超过 50 个样本，零射击管道的性能在考虑微观和宏观* F * [1]-分数时仍然优越。微观* F * [1]-分数的结果告诉我们，基线在频繁类别上表现良好，而零射击管道在这些类别上表现出色，因为它不需要任何示例来学习。

###### 注意

您可能会注意到本节中存在一个小悖论：尽管我们谈论处理没有标签的情况，但我们仍然使用验证集和测试集。我们使用它们来展示不同的技术，并使结果可以相互比较。即使在实际用例中，收集一些有标签的示例进行快速评估也是有意义的。重要的一点是，我们没有根据数据调整模型的参数；相反，我们只是调整了一些超参数。

如果您发现在自己的数据集上难以获得良好的结果，可以尝试以下几种方法来改进零射击管道：

+   管道的工作方式使其对标签的名称非常敏感。如果名称不太合理或与文本不容易联系起来，那么管道可能表现不佳。可以尝试使用不同的名称或并行使用几个名称，并在额外的步骤中对它们进行聚合。
