
```py
# Compute mean protein embeddings for each location.
protein_embeddings = {
  loc: get_mean_embeddings(sequences_by_function[loc], tokenizer, model)
  for loc in ["extracellular", "membrane"]
}

# Reformat data.
labels, embeddings = [], []
for location, embedding in protein_embeddings.items():
  labels.extend([location] * embedding.shape[0])
  embeddings.append(embedding)
  print(f"{location}: {embedding.shape}")
```

输出：

```py
extracellular: (20, 320)
membrane: (20, 320)
```

现在每一组 20 个采样的蛋白质现在表示为一个`(20, 320)`嵌入矩阵。这意味着对于每个序列——无论其原始长度如何——我们都获得一个固定大小的 320 维向量。这些向量对应于序列中所有标记的最终隐藏层激活的平均值，并且应该包含关于蛋白质整体结构的某些信息。

为了可视化这些嵌入如何与蛋白质定位相关，我们使用 t-SNE（一种用于可视化高维数据的常用方法）将它们投影到二维空间。图 2-11 显示，细胞外和膜蛋白在这个空间中倾向于形成不同的簇。

```py
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

embeddings_tsne = TSNE(n_components=2, random_state=42).fit_transform(
  np.vstack(embeddings)
)
embeddings_tsne_df = pd.DataFrame(
  {
    "first_dimension": embeddings_tsne[:, 0],
    "second_dimension": embeddings_tsne[:, 1],
    "location": np.array(labels),
  }
)

fig = sns.scatterplot(
  data=embeddings_tsne_df,
  x="first_dimension",
  y="second_dimension",
  hue="location",
  style="location",
  s=50,
  alpha=0.7,
)
plt.title("tSNE of Protein Embeddings")
fig.set_xlabel("First Dimension")
fig.set_ylabel("Second Dimension");
```

![图片](img/dlfb_0211.png)

###### 图 2-11. 来自小型 ESM2 模型的 320 维嵌入的二维 t-SNE 投影。即使在这个轻量级模型中，我们也观察到细胞外和膜蛋白倾向于形成单独的簇，这表明嵌入包含与细胞定位相关的信息。

虽然分离并不完美，但有一个明显的趋势：细胞外蛋白倾向于在嵌入空间的一个不同区域聚集，与膜蛋白不同。模型仅从序列中捕捉到这一点是非常引人注目的。这表明学习到的嵌入反映了生物上有意义的模式——即使没有对细胞位置的任何明确监督。

在完成初步探索之后，我们现在转向本章的核心机器学习任务：预测蛋白质功能。让我们首先准备数据集。

# 准备数据

许多机器学习书籍和博客文章都尽可能快地直接跳到令人兴奋的部分——训练和评估模型。但在实践中，训练通常只是整个工作流程的一小部分。大量时间花在理解、清理和结构化数据上。当模型出现问题的时候，根本原因通常在数据中。因此，我们不会从虚无中给你一个完美的 CSV 文件，而是会一步一步地展示数据准备过程——从现实世界的资源开始，逐步完成将其转化为模型可以使用的过程。

我们的目标是微调一个模型来从序列中预测蛋白质功能，这意味着组装一个`(蛋白质序列，蛋白质功能)`对的数据库。幸运的是，生物学家已经开发了定义蛋白质功能的系统框架，并且已经存在经过整理的数据集。最广泛使用的资源之一是[CAFA（功能注释关键评估）](https://oreil.ly/87EN_)挑战，这是一个由社区驱动的比赛，团队构建模型来预测蛋白质功能。我们将使用 CAFA 数据作为我们的原材料，但我们仍然需要自己处理和结构化它。

###### 注意

如果你熟悉 AlphaFold 和蛋白质结构预测，你可能听说过同名的 CASP（结构预测关键评估），它在蛋白质结构社区中扮演着类似的角色。这类公开基准在推动计算生物学广泛问题上的进步中起到了关键作用。

现在，让我们探索 CAFA 数据集。

## 加载 CAFA3 数据

CAFA 已经进行了几轮，但 CAFA3 数据集是最新的公开可用数据集。我们首先从[CAFA 网站](https://oreil.ly/87EN_)下载了“CAFA3 Targets”和“CAFA3 Training Data”文件。让我们先加载标签文件，它告诉我们每个蛋白质的功能注释：

```py
labels = pd.read_csv(
  assets("proteins/datasets/train_terms.tsv.zip"), sep="\t", compression="infer"
)
print(labels)
```

输出：

```py
            EntryID        term aspect
0        A0A009IHW8  GO:0008152    BPO
1        A0A009IHW8  GO:0034655    BPO
2        A0A009IHW8  GO:0072523    BPO
...             ...         ...    ...
5363860      X5M5N0  GO:0005515    MFO
5363861      X5M5N0  GO:0005488    MFO
5363862      X5M5N0  GO:0003674    MFO

[5363863 rows x 3 columns]
```

这个数据框包含三列：

`EntryID`

蛋白质的 UniProt ID

`term`

描述特定蛋白质功能的 GO 访问号

`aspect`

函数所属的 GO 类别；介绍中描述的三种功能类型之一：生物过程（BPO）、分子功能（MFO）和细胞组分（CCO）

`term`列只包含 GO 访问号。为了使这些更易于理解，我们最好知道它们对应的人类可读描述。这些信息没有直接包含在 CAFA 文件中，但可以通过[基因本体下载页面](https://oreil.ly/uNhm2)获得。本体以`.obo`文件格式存储为图形格式，我们可以使用`obonet` Python 库来解析它。以下是检索术语描述的方法：

```py
import obonet

def get_go_term_descriptions(store_path: str) -> pd.DataFrame:
  """Return GO term to description mapping, downloading if needed."""
  if not os.path.exists(store_path):
    url = "https://current.geneontology.org/ontology/go-basic.obo"
    graph = obonet.read_obo(url)

    # Extract GO term IDs and names from the graph nodes.
    id_to_name = {id: data.get("name") for id, data in graph.nodes(data=True)}
    go_term_descriptions = pd.DataFrame(
      zip(id_to_name.keys(), id_to_name.values()),
      columns=["term", "description"],
    )
    go_term_descriptions.to_csv(store_path, index=False)

  else:
    go_term_descriptions = pd.read_csv(store_path)
  return go_term_descriptions
```

如果该功能已存在于本地文件中，则该函数将加载注释，否则将下载并缓存它们：

```py
go_term_descriptions = get_go_term_descriptions(
  store_path=assets("proteins/datasets/go_term_descriptions.csv")
)
print(go_term_descriptions)
```

输出：

```py
             term          description
0      GO:0000001  mitochondrion in...
1      GO:0000002  mitochondrial ge...
2      GO:0000006  high-affinity zi...
...           ...                  ...
40211  GO:2001315  UDP-4-deoxy-4-fo...
40212  GO:2001316  kojic acid metab...
40213  GO:2001317  kojic acid biosy...

[40214 rows x 2 columns]
```

然后，我们可以将可读的术语描述合并回标签数据框：

```py
labels = labels.merge(go_term_descriptions, on="term")
labels
```

输出：

```py
            EntryID        term aspect          description
0        A0A009IHW8  GO:0008152    BPO    metabolic process
1        A0A009IHW8  GO:0034655    BPO  nucleobase-conta...
2        A0A009IHW8  GO:0072523    BPO  purine-containin...
...             ...         ...    ...                  ...
4933955      X5M5N0  GO:0005515    MFO      protein binding
4933956      X5M5N0  GO:0005488    MFO              binding
4933957      X5M5N0  GO:0003674    MFO   molecular_function

[4933958 rows x 4 columns]
```

在本章中，我们将特别关注分子功能（`MFO`）——也就是说，蛋白质在生化层面的作用。稍后，你可能希望将本章的方法扩展到包括其他两个 GO 类别。

让我们看看在数据集中哪些分子功能被标注得最频繁：

```py
labels = labels[labels["aspect"] == "MFO"]
print(labels["description"].value_counts())
```

输出：

```py
description
molecular_function                            78637
binding                                       57380
protein binding                               47987
                                              ...  
kaempferide 7-O-methyltransferase activity        1
protopine 6-monooxygenase activity                1
costunolide 3beta-hydroxylase activity            1
Name: count, Length: 6973, dtype: int64
```

我们已经可以看到功能注释的分布非常倾斜。一些术语——如`molecular_function`、`binding`和`protein binding`——出现了数万次，而其他术语只出现一次。像`molecular_function`这样的标签可能过于通用，提供的有意义信息很少，对机器学习没有帮助。我们将在后续步骤中过滤掉这些标签。

接下来，让我们加载与每个蛋白质 ID 相关的蛋白质序列。这些信息存储在文件*train_sequences.fasta*中，这是表示蛋白质和 DNA 等生物序列的标准格式。我们可以使用 BioPython 的`SeqIO`模块将*.fasta*文件解析为我们能处理的形式。

###### 备注

简单地说：没有人一开始就知道 BioPython 的`SeqIO`模块是什么，或者*.fasta*文件是如何工作的，或者 GO 注释意味着什么——这是完全正常的。在生物学和机器学习的交叉领域工作意味着不断遇到新的工具和术语。频繁查阅新术语和工具不仅是可以接受的，而且是预期的。

我们将把*.fasta*序列转换为 pandas 数据框，以便更容易地操作：

```py
from Bio import SeqIO

sequences_file = assets("proteins/datasets/train_sequences.fasta")
fasta_sequences = SeqIO.parse(open(sequences_file), "fasta")

data = []
for fasta in fasta_sequences:
  data.append(
    {
      "EntryID": fasta.id,
      "Sequence": str(fasta.seq),
      "Length": len(fasta.seq),
    }
  )
sequence_df = pd.DataFrame(data)
print(sequence_df)
```

输出：

```py
           EntryID             Sequence  Length
0           P20536  MNSVTVSHAPYTITYH...     218
1           O73864  MTEYRNFLLLFITSLS...     354
2           O95231  MRLSSSPPRGPQQLSS...     258
...            ...                  ...     ...
142243      Q5RGB0  MADKGPILTSVIIFYL...     448
142244  A0A2R8QMZ5  MGRKKIQITRIMDERN...     459
142245  A0A8I6GHU0  HCISSLKLTAFFKRSF...     138

[142246 rows x 3 columns]
```

我们还计算了每个序列的长度，因为蛋白质长度可能差异很大，这些信息将在稍后过滤数据时很有用。

一个重要的细节：CAFA 数据集包括来自许多不同生物体的蛋白质。为了隔离人类蛋白质，我们将使用下载中提供的关联分类文件：

```py
taxonomy_file = assets("proteins/datasets/train_taxonomy.tsv.zip")
taxonomy = pd.read_csv(taxonomy_file, sep="\t", compression="infer")
print(taxonomy)
```

输出：

```py
           EntryID  taxonomyID
0           Q8IXT2        9606
1           Q04418      559292
2           A8DYA3        7227
...            ...         ...
142243  A0A2R8QBB1        7955
142244      P0CT72      284812
142245      Q9NZ43        9606

[142246 rows x 2 columns]
```

此文件包含每个蛋白质的物种分类 ID (`taxonomyID`)，基于 NCBI 的物种分类系统。我们将将其合并到我们的序列数据框中，并仅保留`taxonomyID == 9606`的蛋白质，这对应于*Homo sapiens*：

```py
sequence_df = sequence_df.merge(taxonomy, on="EntryID")
sequence_df = sequence_df[sequence_df["taxonomyID"] == 9606]
```

现在让我们概述一下我们过滤后的数据集中独特蛋白质和分子功能术语的数量：
