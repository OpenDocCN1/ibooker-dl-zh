
```py
n_answers = 3
preds = pipe.run(query=query, top_k_retriever=3, top_k_reader=n_answers,
                 filters={"item_id": [item_id], "split":["train"]})

print(f"Question: {preds['query']} \n")
for idx in range(n_answers):
    print(f"Answer {idx+1}: {preds['answers'][idx]['answer']}")
    print(f"Review snippet: ...{preds['answers'][idx]['context']}...")
    print("\n\n")
```

```py
Question: Is it good for reading?

Answer 1: I mainly use it for book reading
Review snippet: ... is my third one.  I never thought I would want a fire for I
mainly use it for book reading.  I decided to try the fire for when I travel I
take my la...

Answer 2: the larger screen compared to the Kindle makes for easier reading
Review snippet: ...ght enough that I can hold it to read, but the larger screen
compared to the Kindle makes for easier reading. I love the color, something I
never thou...

Answer 3: it is great for reading books when no light is available
Review snippet: ...ecoming addicted to hers! Our son LOVES it and it is great
for reading books when no light is available. Amazing sound but I suggest good
headphones t...
```

太好了，现在我们有了一个端到端的 Amazon 产品评论 QA 系统！这是一个很好的开始，但请注意第二和第三个答案更接近实际问题。为了做得更好，我们需要一些指标来量化检索器和读取器的性能。我们接下来将看一下这一点。

# 改进我们的 QA 管道

尽管最近关于 QA 的研究大部分集中在改进阅读理解模型上，但实际上，如果检索器一开始就找不到相关文档，你的读取器有多好并不重要！特别是，检索器为整个 QA 系统的性能设定了一个上限，因此确保它做得好很重要。考虑到这一点，让我们首先介绍一些常见的指标来评估检索器，以便我们可以比较稀疏和密集表示的性能。

## 评估检索器

评估检索器的常见指标是*召回率*，它衡量了检索到的所有相关文档的比例。在这种情况下，“相关”只是指答案是否出现在文本段落中，因此给定一组问题，我们可以通过计算答案出现在检索器返回的前*k*个文档中的次数来计算召回率。

在 Haystack 中，有两种评估检索器的方法：

+   使用检索器内置的`eval()`方法。这可以用于开放域和封闭域的问答，但不能用于像 SubjQA 这样的数据集，其中每个文档都与一个产品配对，我们需要为每个查询按产品 ID 进行过滤。

+   构建一个自定义的`Pipeline`，将检索器与`EvalRetriever`类结合在一起。这样可以实现自定义指标和查询流程。

###### 注意

与召回率相辅相成的指标是*平均精度*（mAP），它奖励能够将正确答案排在文档排名中较高位置的检索器。

由于我们需要对每个产品进行召回率评估，然后在所有产品中进行聚合，我们将选择第二种方法。流水线图中的每个节点都代表一个通过`run()`方法获取一些输入并产生一些输出的类：

```py
class PipelineNode:
    def __init__(self):
        self.outgoing_edges = 1

    def run(self, **kwargs):
        ...
        return (outputs, "outgoing_edge_name")
```

这里的`kwargs`对应于图中前一个节点的输出，在`run()`方法中进行操作，以返回下一个节点的输出的元组，以及出边的名称。唯一的其他要求是包括一个`outgoing_edges`属性，指示节点的输出数量（在大多数情况下`outgoing_edges=1`，除非流水线中有根据某些标准路由输入的分支）。

在我们的情况下，我们需要一个节点来评估检索器，因此我们将使用`EvalRetriever`类，其`run()`方法跟踪哪些文档具有与地面真相匹配的答案。有了这个类，我们可以通过在代表检索器本身的节点之后添加评估节点来构建`Pipeline`图：

```py
from haystack.pipeline import Pipeline
from haystack.eval import EvalDocuments

class EvalRetrieverPipeline:
    def __init__(self, retriever):
        self.retriever = retriever
        self.eval_retriever = EvalDocuments()
        pipe = Pipeline()
        pipe.add_node(component=self.retriever, name="ESRetriever",
                      inputs=["Query"])
        pipe.add_node(component=self.eval_retriever, name="EvalRetriever",
                      inputs=["ESRetriever"])
        self.pipeline = pipe

pipe = EvalRetrieverPipeline(es_retriever)
```

注意每个节点都有一个`name`和一个`inputs`列表。在大多数情况下，每个节点都有一个单独的出边，因此我们只需要在`inputs`中包含前一个节点的名称。

现在我们有了评估流水线，我们需要传递一些查询及其相应的答案。为此，我们将答案添加到我们文档存储中的专用`label`索引。Haystack 提供了一个`Label`对象，以标准化的方式表示答案跨度及其元数据。为了填充`label`索引，我们将首先通过循环遍历测试集中的每个问题，并提取匹配的答案和附加的元数据来创建`Label`对象的列表：

```py
from haystack import Label

labels = []
for i, row in dfs["test"].iterrows():
    # Metadata used for filtering in the Retriever
    meta = {"item_id": row["title"], "question_id": row["id"]}
    # Populate labels for questions with answers
    if len(row["answers.text"]):
        for answer in row["answers.text"]:
            label = Label(
                question=row["question"], answer=answer, id=i, origin=row["id"],
                meta=meta, is_correct_answer=True, is_correct_document=True,
                no_answer=False)
            labels.append(label)
    # Populate labels for questions without answers
    else:
        label = Label(
            question=row["question"], answer="", id=i, origin=row["id"],
            meta=meta, is_correct_answer=True, is_correct_document=True,
            no_answer=True)
        labels.append(label)
```

如果我们查看其中一个标签：

```py
print(labels[0])
```

```py
{'id': 'e28f5e62-85e8-41b2-8a34-fbff63b7a466', 'created_at': None, 'updated_at':
None, 'question': 'What is the tonal balance of these headphones?', 'answer': 'I
have been a headphone fanatic for thirty years', 'is_correct_answer': True,
'is_correct_document': True, 'origin': 'd0781d13200014aa25860e44da9d5ea7',
'document_id': None, 'offset_start_in_doc': None, 'no_answer': False,
'model_id': None, 'meta': {'item_id': 'B00001WRSJ', 'question_id':
'd0781d13200014aa25860e44da9d5ea7'}}
```

我们可以看到问题-答案对，以及一个包含唯一问题 ID 的`origin`字段，以便我们可以根据问题过滤文档存储。我们还将产品 ID 添加到`meta`字段中，以便我们可以按产品筛选标签。现在我们有了标签，我们可以按以下方式将它们写入 Elasticsearch 上的`label`索引：

```py
document_store.write_labels(labels, index="label")
print(f"""Loaded {document_store.get_label_count(index="label")} \
question-answer pairs""")
```

```py
Loaded 358 question-answer pairs
```

接下来，我们需要建立一个映射，将我们的问题 ID 和相应的答案传递给流水线。为了获取所有标签，我们可以使用文档存储中的`get_all_labels_aggregated()`方法，该方法将聚合与唯一 ID 相关联的所有问题-答案对。该方法返回一个`MultiLabel`对象的列表，但在我们的情况下，由于我们按问题 ID 进行过滤，所以只会得到一个元素。我们可以按以下方式建立一个聚合标签列表：

```py
labels_agg = document_store.get_all_labels_aggregated(
    index="label",
    open_domain=True,
    aggregate_by_meta=["item_id"]
)
print(len(labels_agg))
```

```py
330
```

通过查看其中一个标签，我们可以看到与给定问题相关联的所有答案都聚合在一个`multiple_answers`字段中：

```py
print(labels_agg[109])
```

```py
{'question': 'How does the fan work?', 'multiple_answers': ['the fan is really
really good', "the fan itself isn't super loud. There is an adjustable dial to
change fan speed"], 'is_correct_answer': True, 'is_correct_document': True,
'origin': '5a9b7616541f700f103d21f8ad41bc4b', 'multiple_document_ids': [None,
None], 'multiple_offset_start_in_docs': [None, None], 'no_answer': False,
'model_id': None, 'meta': {'item_id': 'B002MU1ZRS'}}
```

现在我们有了评估检索器的所有要素，让我们定义一个函数，将与每个产品相关联的每个问题-答案对传递到评估流水线中，并在我们的`pipe`对象中跟踪正确的检索：

```py
def run_pipeline(pipeline, top_k_retriever=10, top_k_reader=4):
    for l in labels_agg:
        _ = pipeline.pipeline.run(
            query=l.question,
            top_k_retriever=top_k_retriever,
            top_k_reader=top_k_reader,
            top_k_eval_documents=top_k_retriever,
            labels=l,
            filters={"item_id": [l.meta["item_id"]], "split": ["test"]})
```

```py
run_pipeline(pipe, top_k_retriever=3)
print(f"Recall@3: {pipe.eval_retriever.recall:.2f}")
```

```py
Recall@3: 0.95
```

太好了，它起作用了！请注意，我们选择了一个特定的值作为`top_k_retriever`，以指定要检索的文档数。一般来说，增加这个参数将提高召回率，但会增加向读者提供更多文档并减慢端到端流程的速度。为了指导我们选择哪个值，我们将创建一个循环遍历几个*k*值并计算每个*k*在整个测试集上的召回率的函数：

```py
def evaluate_retriever(retriever, topk_values = [1,3,5,10,20]):
    topk_results = {}

    for topk in topk_values:
        # Create Pipeline
        p = EvalRetrieverPipeline(retriever)
        # Loop over each question-answers pair in test set
        run_pipeline(p, top_k_retriever=topk)
        # Get metrics
        topk_results[topk] = {"recall": p.eval_retriever.recall}

    return pd.DataFrame.from_dict(topk_results, orient="index")

es_topk_df = evaluate_retriever(es_retriever)
```

如果我们绘制结果，我们可以看到随着*k*的增加，召回率如何提高：

```py
def plot_retriever_eval(dfs, retriever_names):
    fig, ax = plt.subplots()
    for df, retriever_name in zip(dfs, retriever_names):
        df.plot(y="recall", ax=ax, label=retriever_name)
    plt.xticks(df.index)
    plt.ylabel("Top-k Recall")
    plt.xlabel("k")
    plt.show()

plot_retriever_eval([es_topk_df], ["BM25"])
```

![](img/nlpt_07in02.png)

从图中可以看出，在<math alttext="k equals 5"><mrow><mi>k</mi> <mo>=</mo> <mn>5</mn></mrow></math>附近有一个拐点，从<math alttext="k equals 10"><mrow><mi>k</mi> <mo>=</mo> <mn>10</mn></mrow></math>开始我们几乎可以完美地召回。现在让我们来看看使用密集向量技术检索文档。

### 密集通道检索
