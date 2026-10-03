

# 第十一章：虚拟筛选工作流程示例

虚拟筛选可以提供一种高效且具有成本效益的方法，用于确定药物发现项目的起始点。我们可以使用计算方法来虚拟评估数百万甚至数千万分子，而不是进行昂贵的实验高通量筛选（HTS）。虚拟筛选方法通常分为两类，即基于结构的虚拟筛选和基于配体的虚拟筛选。

在基于结构的虚拟筛选中，使用计算方法来识别最佳适合蛋白质中的一个被称为结合位点的空腔的分子。分子结合到蛋白质结合位点中通常会抑制蛋白质的功能。例如，酶类蛋白质催化各种生理化学反应。通过识别和优化这些酶过程的抑制剂，科学家已经能够开发用于肿瘤学、炎症、感染和其他治疗领域的广泛疾病的治疗方法。

在基于配体的虚拟筛选中，我们搜索与一个或多个已知分子功能类似的分子。我们可能正在寻找改进现有分子的功能，避免与已知分子相关的药理学风险，或开发新的知识产权。基于配体的虚拟筛选通常从通过各种实验方法确定的一组已知分子开始。然后使用计算方法基于实验数据开发模型，并使用该模型虚拟筛选大量分子以找到新的化学起始点。

在本章中，我们将通过一个虚拟筛选工作流程的实际示例。我们将检查用于执行虚拟筛选各个组件的代码，以及整个分析过程中所做决策背后的思考过程。在这种特定情况下，我们将进行基于配体的虚拟筛选。我们将使用已知结合到特定蛋白质的一组分子，以及一组假定不结合的分子，来训练一个卷积神经网络，以识别具有结合到目标的潜力的新分子。

## 为预测建模准备数据集

作为第一步，我们将构建一个图卷积模型，以预测分子抑制一种名为 ERK2 的蛋白质的能力。这种蛋白质，也称为有丝分裂原活化蛋白激酶 1 或 MAPK1，在调节细胞增殖的信号通路中发挥重要作用。ERK2 已被牵涉到多种癌症，并且 ERK2 抑制剂目前正在临床试验中用于非小细胞肺癌和黑色素瘤（皮肤癌）。

我们将训练模型区分一组 ERK2 活性化合物和一组假化合物。活性和假化合物来源于[DUD-E 数据库](http://dud.docking.org/)，该数据库旨在测试预测模型。在实践中，我们通常会从科学文献或来自欧洲生物信息学研究所（EBI）的[ChEMBL 数据库](https://www.ebi.ac.uk/chembl/)等生物活性分子数据库中获取活性和非活性分子。为了生成最佳模型，我们希望假分子的属性分布与我们的活性化合物相似。假设这不是这种情况，非活性化合物的分子量低于活性化合物。在这种情况下，我们的分类器可能只是简单地训练以区分低分子量化合物和高分子量化合物。这样的分类器在实践中的效用非常有限。

为了更好地了解数据集，让我们检查一下我们的活性和假分子的一些计算属性。为了构建可靠的模型，我们需要确保活性分子的属性与假分子的属性相似。

首先，让我们导入必要的库：

```py
from rdkit import Chem             # RDKit libraries for chemistry functions 
from rdkit.Chem import Draw        # Drawing chemical structures 
import pandas as pd                # Dealing with data in tables 
from rdkit.Chem import PandasTools # Manipulating chemical data 
from rdkit.Chem import Descriptors # Calculating molecular descriptors 
from rdkit.Chem import rdmolops    # Additional molecular properties 
import seaborn as sns              # Making graphs 
```

在这个练习中，分子使用 SMILES 字符串表示。有关 SMILES 的更多信息，请参阅第四章。现在我们可以将一个 SMILES 文件读入 Pandas 数据框，并将一个 RDKit 分子添加到数据框中。虽然输入的 SMILES 文件在技术上不是 CSV 文件，但只要我们指定分隔符，Pandas 的`read_CSV()`函数就可以读取它，这种情况下分隔符是空格：

```py
active_df = pd.read_CSV("mk01/actives_final.ism",header=None,sep=" ")
active_rows,active_cols = active_df.shape
active_df.columns = ["SMILES","ID","ChEMBL_ID"]
active_df["label"] = ["Active"]*active_rows
PandasTools.AddMoleculeColumnToFrame(active_df,"SMILES","Mol")
```

让我们定义一个函数，将计算的属性添加到数据框中：

```py
def add_property_columns_to_df(df_in):
    df_in["mw"] = [Descriptors.MolWt(mol) for mol in df_in.Mol]
    df_in["logP"] = [Descriptors.MolLogP(mol) for mol in df_in.Mol]
    df_in["charge"] = [rdmolops.GetFormalCharge(mol) for mol in df_in.Mol]
```

有了这个函数，我们可以计算分子的分子量、LogP 和形式电荷。这些属性编码了分子的大小、从油性物质（辛醇）到水的分配能力，以及分子是否带有正电荷或负电荷。一旦我们有了这些属性，我们就可以比较活性和诱饵集的分布：

```py
add_property_columns_to_df(active_df)
```

让我们查看数据框的前几行，以确保数据框的内容与输入文件匹配（见表 11-1）：

```py
active_df.head()
```

表 11-1。`active_df`数据框的前几行。

|   | SMILES | ID | ChEMBL_ID | label |
|---|--------|----|-----------|-------|
| 0 | Cn1ccnc1Sc2ccc(cc2Cl)Nc3c4cc(c(cc4ncc3C#N)OCCCN5CCOCC5)OC | 168691 | CHEMBL318804 | Active |
| 1 | C[C@@]12C@@Hn3c4ccccc4c5c3c6n2c7ccccc7c6c8c5C(=O)NC8)NC)OC | 86358 | CHEMBL162 | Active |
| 2 | Cc1cnc(nc1c2cc([nH]c2)C(=O)NC@Hc3cccc(c3)Cl)Nc4cccc5c4OC(O5)(F)F | 575087 | CHEMBL576683 | Active |
| 3 | Cc1cnc(nc1c2cc([nH]c2)C(=O)NC@Hc3cccc(c3)Cl)Nc4cccc5c4OCO5 | 575065 | CHEMBL571484 | Active |
| 4 | Cc1cnc(nc1c2cc([nH]c2)C(=O)NC@Hc3cccc(c3)Cl)Nc4cccc5c4CCC5 | 575047 | CHEMBL568937 | Active |

现在让我们对诱饵分子做同样的事情：

```py
decoy_df = pd.read_CSV("mk01/decoys_final.ism",header=None,sep=" ")
decoy_df.columns = ["SMILES","ID"]
decoy_rows, decoy_cols = decoy_df.shape
decoy_df["label"] = ["Decoy"]*decoy_rows
PandasTools.AddMoleculeColumnToFrame(decoy_df,"SMILES","Mol")
add_property_columns_to_df(decoy_df)
```

为了构建模型，我们需要一个包含活性和诱饵分子的单个数据框。我们可以使用 Pandas 的`append`函数将两个数据框添加到一起，创建一个名为`tmp_df`的新数据框：

```py
tmp_df = active_df.append(decoy_df)
```

计算了活性和诱饵集的属性后，我们可以比较两组分子的属性。为了进行比较，我们将使用小提琴图。小提琴图类似于箱线图。小提琴图提供了频率分布的镜像、水平视图。理想情况下，我们希望看到活性和诱饵集的分布相似。结果显示在图 11-1 中：

```py
sns.violinplot(tmp_df["label"],tmp_df["mw"])
```

![](img/dlls_1101.png)

###### 图 11-1。活性和诱饵集的分子量小提琴图。

对这些图的检查显示，两组分子的分子量分布大致相等。诱饵集中有更多低分子量的分子，但在每个小提琴图中间显示的箱子的中心位置是相似的。

我们可以使用小提琴图对 LogP 分布进行类似的比较（图 11-2）。同样，我们可以看到分布是相似的，诱饵分子中有更多分布在分布的低端：

```py
sns.violinplot(tmp_df["label"],tmp_df["logP"])
```

![活性和诱饵集的 LogP 小提琴图。](img/dlls_1102.png)

###### 图 11-2。活性和诱饵集的 LogP 小提琴图。

最后，我们对分子的形式电荷进行相同的比较（图 11-3）：

```py
sns.violinplot(new_tmp_df["label"],new_tmp_df["charge"])
```

![活性和诱饵集的形式电荷小提琴图。](img/dlls_1103.png)

###### 图 11-3。活性和诱饵集的形式电荷小提琴图。

在这种情况下，我们看到了显著的差异。所有活性分子都是中性的，电荷为 0，而一些诱饵是带电的，电荷为+1 或-1。让我们看看多少比例的诱饵分子是带电的。我们可以通过创建一个只包含带电分子的新数据框来做到这一点：

```py
charged = decoy_df[decoy_df["charge"] != 0]
```

Pandas 数据框有一个名为`shape`的属性，返回数据框中的行数和列数。因此，`shape`属性中的元素`[0]`将是行数。让我们将带电分子的数据框中的行数除以假分子数据框中的总行数：

```py
charged.shape[0]/decoy_df.shape[0]
```

这返回 0.162。正如我们在小提琴图中看到的，大约 16%的假分子是带电的。这似乎是因为活性和假分子集合没有以一致的方式准备。我们可以通过修改假分子的化学结构来中和它们的电荷来解决这个问题。幸运的是，我们可以很容易地使用[RDKit Cookbook](https://www.rdkit.org/docs/Cookbook.html)中的`NeutraliseCharges()`函数来做到这一点：

```py
from neutralize import NeutraliseCharges
```

为了避免混淆，我们创建一个新的数据框，其中包含假分子的 SMILES 字符串、ID 和标签：

```py
revised_decoy_df = decoy_df[["SMILES","ID","label"]].copy()
```

有了这个新的数据框，我们可以用分子的中性形式的字符串替换原始的 SMILES 字符串。`NeutraliseCharges`函数返回两个值。第一个是分子中性形式的 SMILES 字符串，第二个是一个布尔变量，指示分子是否发生了变化。在下面的代码中，我们只需要 SMILES 字符串，所以我们使用`NeutraliseCharges`返回的元组的第一个元素。

```py
revised_decoy_df["SMILES"] = [NeutraliseCharges(x)[0] for x in revised_decoy_df["SMILES"]]
```

一旦我们替换了 SMILES 字符串，我们就可以向我们的新数据框添加一个分子列，并再次计算属性：

```py
PandasTools.AddMoleculeColumnToFrame(revised_decoy_df,"SMILES","Mol")
add_property_columns_to_df(revised_decoy_df)
```

然后我们可以将带活性分子的数据框追加到修订后的中性假分子的数据框中：

```py
new_tmp_df = active_df.append(revised_decoy_df)
```

接下来，我们可以生成一个新的箱线图，比较活性分子的电荷分布与我们中和的假分子的电荷分布（图 11-4）：

```py
sns.violinplot(new_tmp_df["label"],new_tmp_df["charge"])
```

![我们修订后的假分子集带电分布的小提琴图。](img/dlls_1104.png)

###### 图 11-4。我们修订后的假分子集带电分布的小提琴图。

经过对图表的检查，我们发现假分子集中现在几乎没有带电分子。我们可以使用之前使用的相同技术来创建一个仅包含带电分子的数据框。然后我们使用这个数据框来确定集合中剩余的带电分子数量：

```py
charged = revised_decoy_df[revised_decoy_df["charge"] != 0]
charged.shape[0]/revised_decoy_df.shape[0]
```

现在的结果是 0.003。我们已将带电分子的比例从 16%降低到 0.3%。现在我们可以确信我们的活性和假分子集合相当平衡。

为了将这些数据集与 DeepChem 一起使用，我们需要将分子写出为一个 CSV 文件，其中对于每个分子，包含 SMILES 字符串、ID、名称和一个整数值，指示化合物是活性的（标记为 1）还是非活性的（标记为 0）：

```py
active_df["is_active"] = [1] * active_df.shape[0]
revised_decoy_df["is_active"] = [0] * revised_decoy_df.shape[0]
combined_df = active_df.append(revised_decoy_df)[["SMILES","ID","is_active"]]
combined_df.head()
```

前五行显示在表 11-2 中。

表 11-2。我们新的合并数据框的前几行

|   | SMILES | ID | is_active |
|---|--------|----|-----------|
| 0 | Cn1 ccnc1Sc2ccc(cc2Cl}Nc3c4cc(c(cc4ncc3C#N}OCCCN5CCOCC5)OC | 168691 | 1 |
| 1 | C[C@@]12C@@Hn3c4ccccc4c5c3c6n2c7ccccc7c6c8c5C(=O)NC8)NC)OC | 86358 | 1 |
| 2 | Cc1cnc(nc1c2cc([nH]c2)C(=0) NC@Hc3cccc(c3}Cl}Nc4cccc5c4OC(O5)(F)F | 575087 | 1 |
| 3 | CCc1cnc(nc1c2cc([nH]c2)C(=O)NC@Hc3cccc(c3}Cl}Nc4cccc5c4OCO5 | 575065 | 1 |
| 4 | Cc1cnc(nc1c2cc([nH]c2)C(=0) NC@Hc3cccc(c3}Cl}Nc4cccc5c4CCC5 | 575047 | 1 |

本节的最后一步是将我们的新`combined_df`保存为 CSV 文件。`index=False`选项使 Pandas 不在第一列中包含行号：

```py
combined_df.to_csv("dude_erk1_mk01.CSV",index=False)
```

## 训练预测模型

现在我们已经处理了格式，我们可以使用这些数据来训练一个图卷积模型。首先，我们需要导入必要的库。其中一些库在第一节中已经导入，但让我们假设我们从上一节创建的 CSV 文件开始：

```py
import deepchem as dc                      # DeepChem libraries
from deepchem.models import GraphConvModel # Graph convolutions 
import numpy as np                         # NumPy for numeric operations 
import sys                                 # Error handling 
import pandas as pd                        # Data table manipulation 
import seaborn as sns                      # Seaborn library for plotting 
from rdkit.Chem import PandasTools         # Chemical structures in Pandas 
```

现在让我们定义一个函数来创建一个`GraphConvModel`。在这种情况下，我们将创建一个分类模型。由于我们将在以后的不同数据集上应用该模型，因此最好创建一个目录来存储模型。您需要将目录更改为您的文件系统上可访问的内容：

```py
def generate_graph_conv_model():
    batch_size = 128
    model = GraphConvModel(1, batch_size=batch_size,
                           mode='classification',
                           model_dir="/tmp/mk01/model_dir")
    return model
```

为了训练模型，我们首先读取在上一节中创建的 CSV 文件：




# 构建预测模型

现在我们已经加载了数据集，让我们构建一个模型。我们将创建训练集和测试集来评估模型的性能。在这种情况下，我们将使用`RandomSplitter`（DeepChem 还提供许多其他分割器，如`ScaffoldSplitter`，它通过化学支架划分数据集，以及`ButinaSplitter`，它首先对数据进行聚类，然后将数据集分割，使不同的聚类最终出现在训练集和测试集中）：

```py
splitter = dc.splits.RandomSplitter()
```

随着数据集的拆分，我们可以在训练集上训练模型，然后在验证集上测试该模型。在这一点上，我们需要定义一些指标并评估我们模型的性能。在这种情况下，我们的数据集是不平衡的：我们有少量活性化合物和大量非活性化合物。鉴于这种差异，我们需要使用一个反映不平衡数据集性能的指标。适合这种数据集的一个指标是马修斯相关系数（MCC）：

```py
metrics = [
dc.metrics.Metric(dc.metrics.matthews_corrcoef, np.mean,
mode="classification")]
```

为了评估我们模型的性能，我们将执行 10 次交叉验证，其中我们在训练集上训练模型并在验证集上验证：

```py
training_score_list = []
validation_score_list = []
transformers = []
cv_folds = 10
for i in range(0, cv_folds):
model = generate_graph_conv_model()
res = splitter.train_valid_test_split(dataset)
train_dataset, valid_dataset, test_dataset = res
model.fit(train_dataset)
train_scores = model.evaluate(train_dataset, metrics,
transformers)
training_score_list.append(
train_scores["mean-matthews_corrcoef"])
validation_scores = model.evaluate(valid_dataset,
metrics,
transformers)
validation_score_list.append(
validation_scores["mean-matthews_corrcoef"])
print(training_score_list)
print(validation_score_list)
```

为了可视化我们模型在训练和测试数据上的性能，我们可以制作箱线图。结果显示在图 11-5 中：

```py
sns.boxplot(
["training"] * cv_folds + ["validation"] * cv_folds,
training_score_list + validation_score_list)
```

![训练集和验证集的分数的箱线图。](img/dlls_1105.png)

###### 图 11-5。训练集和验证集的分数的箱线图。

图表表明，如预期的那样，训练集上的性能优于验证集上的性能。然而，验证集上的性能仍然相当不错。在这一点上，我们可以对我们模型的性能感到自信。

可视化我们模型的结果也是有用的。为了做到这一点，我们将为验证集生成一组预测：

```py
pred = [x.flatten() for x in model.predict(valid_dataset)]
```

为了使处理更容易，我们将使用预测创建一个 Pandas 数据框架：

```py
pred_df = pd.DataFrame(pred,columns=["neg","pos"])
```

我们可以轻松地将活动类别（1 = 活性，0 = 非活性）和我们预测的分子的 SMILES 字符串添加到数据框架中：

```py
pred_df["active"] = [int(x) for x in valid_dataset.y]
pred_df["SMILES"] = valid_dataset.ids
```

查看数据框架的前几行总是一个好主意，以确保数据是合理的。表 11-3 显示了结果。

**表 11-3。包含预测的数据框架的前几行**

|   | neg | pos | active | SMILES |
|---|---|---|---|---|
| 0 | 0.906081 | 0.093919 | 1 | Cn1ccnc1Sc2ccc(cc2Cl)Nc3c4cc(c(cc4ncc3C#N)OCCC... |
| 1 | 0.042446 | 0.957554 | 1 | Cc1cnc(nc1c2cc([nH]c2)C(=O)NC@Hc3cccc(c3... |
| 2 | 0.134508 | 0.865492 | 1 | Cc1cccc(c1)C@@HNC(=O)c2cc(c[nH]2)c3c(cnc... |
| 3 | 0.036508 | 0.963492 | 1 | Cc1cnc(nc1c2cc([nH]c2)C(=O)NC@Hc3ccccc3)... |
| 4 | 0.940717 | 0.059283 | 1 | c1c\2c([nH]c1Br)C(=O)NCC/C2=C/3\C(=O)N=C(N3)N |

创建箱线图使我们能够比较活性和非活性分子的预测值（请参见图 11-6）。

```py
sns.boxplot(pred_df.active,pred_df.pos)
```

![预测分子的正分数。](img/dlls_1106.png)

###### 图 11-6。预测分子的正分数。

我们的模型表现非常好：我们可以看到活性和非活性分子之间有明显的区分。在构建预测模型时，通常重要的是检查被预测为活性的非活性分子（假阳性）以及被预测为非活性的活性分子（假阴性）。看起来只有一个我们的活性分子得到了低正分数。为了更仔细地观察，我们将创建一个新的数据框架，其中包含所有得分<0.5 的活性分子：

```py
false_negative_df = pred_df.query("active == 1 & pos < 0.5").copy()
```

为了检查数据框架中分子的化学结构，我们使用 RDKit 中的 PandasTools 模块：

```py
PandasTools.AddMoleculeColumnToFrame(false_negative_df,
"SMILES", "Mol")
```

让我们看看新的数据框架（图 11-7）：

```py
false_negative_df
```

![假阴性预测。](img/dlls_1107.png)

###### 图 11-7. 假阴性预测。

为了充分利用这个数据框架中的信息，我们需要对药物化学有一些了解。查看假阴性分子的化学结构并将其与真阳性分子的化学结构进行比较通常是有益的。这可能会提供一些关于为什么分子没有被正确预测的原因的见解。通常情况下，假阴性分子可能与任何真阳性分子都不相似。在这种情况下，值得进行额外的文献搜索，以增加训练集中分子的多样性。

我们可以使用类似的方法来检查假阳性分子，这些分子是非活性的，但得到了>0.5 的正分数。再次与真阳性分子的化学结构进行比较可能会提供信息：

```py
false_positive_df = pred_df.query(
    "active == 0 & pos > 0.5").copy()
PandasTools.AddMoleculeColumnToFrame(false_positive_df,
                                     "SMILES", "Mol")
false_positive_df
```

![一个假阳性分子。](img/dlls_1108.png)

###### 图 11-8. 一个假阳性分子。

在模型训练阶段，我们的目标是评估模型的性能。因此，我们在部分数据上训练模型，并在其余数据上验证模型。现在我们已经评估了性能，我们希望生成最有效的模型。为了做到这一点，我们将在所有数据上训练模型：

```py
model.fit(dataset)
```

这给我们一个准确度得分为 91%。最后，我们将模型保存到磁盘上，以便将来进行预测使用：

```py
model.save()
```

# 为模型预测准备数据集

现在我们已经创建了一个预测模型，我们可以将这个模型应用到一组新的分子上。在许多情况下，我们会基于文献数据构建一个预测模型，然后将该模型应用到我们想要筛选的一组分子上。我们想要筛选的分子可能来自内部数据库或商业可用的筛选集合。例如，我们将使用我们创建的预测模型来筛选来自 ZINC 数据库的 100,000 个化合物的小样本，这是一个包含超过 10 亿种商业可用分子的集合。

在进行虚拟筛选时，一个潜在的困难源可能是存在可能干扰生物测定的分子。在过去的 25 年里，科学界内的许多团体已经制定了一套规则，用于识别可能具有反应性或问题的分子。这些规则集中的几个，被编码为 SMARTS 字符串，已经被负责维护 ChEMBL 数据库的团体收集起来。这些规则集已经通过一个名为`rd_filters.py`的 Python 脚本提供。在这个例子中，我们将使用`rd_filters.py`来识别来自 ZINC 数据库的 100,000 个分子集合中可能存在问题的分子。

`rd_filters.py`脚本和相关数据文件可以在我们的[GitHub 存储库](https://github.com/deepchem/DeepLearningLifeSciences)上找到。

通过使用`-h`标志调用脚本可以获得脚本的可用模式和参数。

```py
rd_filters.py -h
```

```py
Usage:
rd_filters.py $ **filter --in INPUT_FILE --prefix PREFIX [--rules RULES_FILE_NAME] 
[--alerts ALERT_FILE_NAME][--np NUM_CORES]**
rd_filters.py $ **template --out TEMPLATE_FILE [--rules RULES_FILE_NAME]**
Options:
--in INPUT_FILE input file name
--prefix PREFIX prefix for output file names
--rules RULES_FILE_NAME name of the rules JSON file
--alerts ALERTS_FILE_NAME name of the structural alerts file
--np NUM_CORES the number of cpu cores to use (default is all)
--out TEMPLATE_FILE parameter template file name 
```

要在我们的输入文件上调用脚本，该文件名为`zinc_100k.smi`，我们可以指定输入文件和输出文件名的前缀。`filter`参数调用脚本处于`filter`模式，其中它识别可能有问题的分子。`--prefix`参数指示输出文件名将以前缀`zinc`开头。

```py
rd_filters.py filter --in zinc_100k.smi --prefix zinc
```

```py
using 24 cores
Using alerts from Inpharmatica
Wrote SMILES for molecules passing filters to zinc.smi
Wrote detailed data to zinc.CSV
68752 of 100000 passed filters 68.8%
Elapsed time 15.89 seconds 
```

输出指示以下内容：

+   脚本正在运行在 24 个核心上。它在多个核心上并行运行，可以使用`-np`标志选择核心数。

+   脚本正在使用“Inpharmatica”规则集。该规则集涵盖了在生物测定中被证明有问题的大量化学功能。除了 Inpharmatica 规则集外，脚本还有其他七个规则集可用。有关更多信息，请参阅`rd_filters.py`文档。

+   通过过滤器的分子的 SMILES 字符串已写入名为`zinc.smi`的文件。当我们使用预测模型时，我们将使用此作为输入。

+   特定结构警报触发的化合物详细信息已写入名为`zinc.CSV`的文件。

+   69%的分子通过了过滤器，31%被认为有问题。

查看 31%的分子被拒绝的原因是很有启发性的。这可以让我们知道是否需要调整任何过滤器。我们将使用一些 Python 代码来查看输出的前几行，如表 11-4 所示。

```py
import pandas as pd
df = pd.read_CSV("zinc.CSV")
df.head()
```

**表 11-4。从 zinc.CSV 创建的数据框的前几行**

|   | SMILES | 名称 | 过滤器 | 分子量 | LogP | HBD |
|---|---|---|---|---|---|---|
| 0 | CN(CCO)CC@@HCn1cnc2c1c(=O)n(C)c(=O)n2C | ZINC000000000843 | Filter82_pyridinium >0 | 311.342 | –2.2813 | 2 |
| 1 | O=c1[nH]c(=O)n([C@@H]2CC@@HC@HO2)cc1Br | ZINC000000001063 | Filter82_pyridinium >0 | 307.100 | –1.0602 | 3 |
| 2 | Cn1c2ncn(CC(=O)N3CCOCC3)c2c(=O)n(C)c1=O | ZINC000000003942 | Filter82_pyridinium >0 | 307.310 | –1.7075 | 0 |
| 3 | CN1C(=O)CC@HCN3CCCC3)CC2)C1=O | ZINC000000036436 | OK | 308.382 | –1.0163 | 0 |
| 4 | CC(=O)NCC@H[C@H]1O[C@H]2OC(C)(C)O[C@H]2C... | ZINC 000000041101 | OK | 302.327 | -1.1355 | 3 |

数据框有六列：

**SMILES**

每个分子的 SMILES 字符串。

**名称**

分子名称，如输入文件中列出的。

**过滤器**

分子被拒绝的原因，或者如果分子未被拒绝则为“OK”。

**MW**

分子的分子量。默认情况下，具有大于 500 的分子量的分子将被拒绝。

**LogP**

分子的计算辛醇/水分配系数。默认情况下，具有大于五的 LogP 值的分子将被拒绝。

**HBD**

氢键供体的数量。默认情况下，具有超过 5 个氢键供体的分子将被拒绝。

我们可以使用 Python 的`collections`库中的`Counter`类来识别哪些过滤器负责移除最多的分子（参见表 11-5）：

```py
from collections import Counter
count_list = list(Counter(df.FILTER).items())
count_df = pd.DataFrame(count_list,columns=["Rule","Count"])
count_df.sort_values("Count",inplace=True,ascending=False)
count_df.head()
```

**表 11-5。前 5 个过滤器选择的分子数量统计**

|   | 规则 | 计数 |
|---|---|---|
| 1 | OK | 69156 |
| 6 | Filter41_12_dicarbonyl > O | 19330 |
| 0 | Filter82_pyridinium > O | 7761 |
| 10 | Filter93_acetyl_urea > O | 1541 |
| 11 | Filter78_bicyclic_lmide > O | 825 |

表中的第一行标记为“OK”，表示未被任何过滤器淘汰的分子数量。从中我们可以看到，我们输入的分子中有 69,156 个通过了所有过滤器。最多的分子数量（19,330）被拒绝，因为它们含有 1,2-二羰基基团。这种类型的分子可能会与丝氨酸和半胱氨酸等蛋白质残基反应并形成共价键。我们可以通过在`rd_filters.py`分发的`filter_collection.CSV`文件中查找字符串“Filter41_12_dicarbonyl”来找到用于识别这些分子的 SMARTS 模式。SMARTS 模式是“*C(=O)C(=O)*”，表示：

+   连接到任何原子

+   碳双键氧，连接到

+   碳双键氧，连接到

+   任何原子。

查看数据并确保一切按预期工作总是好的。我们可以使用 RDKit 的`MolsToGridImage()`函数的`highlightAtomLists`参数来突出显示 1,2-二羰基功能（参见图 11-9）：



进入`/path/to/xxx`目录，找到`xxx.json`。

在表达式`cvar = avar + bvar`中，加法运算符（`+`）将`avar`与`bvar`相加，得到它们的和`cvar`

在`List.of(arg0, arg1, arg2)`中，`List`接口的工厂方法`of()`接受一系列的元素，返回包含它们的只读列表。

之后我们这样调用`cmd`命令：`cmd arg0 arg1 arg2`。

```
if (condVar > someVal) {console.log("xxx")}
```
