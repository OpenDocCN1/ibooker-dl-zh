
在开始使用这些方法之前，阅读像“如何有效地使用 t-SNE”（[`distill.pub/2016/misread-tsne/`](https://distill.pub/2016/misread-tsne/)）或“理解 UMAP”（[`mng.bz/AQxK`](https://mng.bz/AQxK)）这样的文章，这些文章将为你提供在使用方法时采取适当预防措施所需的额外信心。另一个注意事项是，这两种方法都是计算密集型的，从大型和复杂的数据集中获得降维可能需要相当长的时间。然而，最近，NVIDIA 还基于 CUDA 和 GPU 技术开发了其 RAPIDS 套件（[`developer.nvidia.com/rapids`](https://developer.nvidia.com/rapids)），这可以显著减少从 UMAP 和 t-SNE 获得结果所需的时间，使它们在密集的 EDA 探索中更加有效。列表 2.9 展示了使用 Scikit-learn 包中的 t-SNE 实现分析示例数据集的代码。我们将在本书的后面部分介绍 NVIDIA RAPIDS 的实现，以及它们用于处理表格数据的其他工具。

## 列表 2.9 绘制数据集的 t-SNE 低维投影

```py
from sklearn.manifold import TSNE                           ①
import matplotlib.pyplot as plt                             ②

tsne = TSNE(n_components=2, 
            perplexity=30., 
            init="random", 
            learning_rate="auto",
            random_state=42)                                ③

X = data[numeric_feats + ordinal_feats].fillna(
            data[numeric_feats + ordinal_feats].mean())     ④
projection_2D = tsne.fit_transform(X)                       ⑤

plt.figure(figsize=(15, 15))
plt.scatter(projection_2D[:, 0], projection_2D[:, 1],
            edgecolor='none', 
            alpha=0.80, 
            s=10) 
plt.show()                                                  ⑥
```

① 导入 Scikit-learn 中可用的 t-SNE 类

② 导入 matplotlib 的 pyplot 用于图表绘制

③ 将 t-SNE 设置为在二维投影结果；其他参数保持默认设置。

④ 仅使用数值和有序特征；缺失值用均值替换，因为 t-SNE 类需要完整的输入数据矩阵。

⑤ 通过 `fit_transform`，投影基于提供的数据创建，并应用于数据本身。

⑥ 将 t-SNE 变换后的数据绘制为二维散点图。

图 2.13 展示了 t-SNE 变换后的数据结果图。

![](img/CH02_F13_Ryan2.png)

## 图 2.13 t-SNE 二维变换后的点分布

代码片段仅适用于数值特征，在所有缺失值被均值替换后（t-SNE 和 UMAP 都不能处理缺失数据），将数据压缩成散点图可以绘制的二维表示。在这个数据集中，结果看起来非常规律，如果我们认为这是为特定目的收集的精选示例集合，那么就不会有太多惊喜：所有数据点最终都排列在一个曲线云中，整体给人一种根据特定标准逐步、规律选择示例的印象。

这标志着我们对 Auto MPG 数据集的探索结束。下一章将讨论机器学习算法，并探讨通常应用于表格数据集的基本经典模型。通过举例，我们将指出每个算法在数据以及你可能在项目中遇到的特征方面的优势和劣势。

## 摘要

+ 尽管在领域和组织之间存在巨大差异，但表格数据集中的共同特征使得可以概述最佳的数据处理和建模实践。

+ 表格数据集中的行关联所表示的单位，当它们不是独立同分布（non-IID）时，会有局限性和机会需要注意。非独立同分布数据可能会影响数据科学中常用的程序，如自助法、子采样和交叉验证。

+ 在列中可以找到不同类型的数据：数值型（包括浮点数和整数）、序数型（整数）、分类型（低和高基数，即具有低或高数量的不同标签），以及日期：每种类型都需要不同的数据处理和分析方法。我们建议掌握 pandas 包（及其 DataFrame 数据结构）来处理表格数据中发现的这些不同类型的特征。

+ 不同的数据异常与列中数据的类型有关：常数或准常数特征、重复或高度共线性特征、无关特征、稀有类别和其他不一致性、缺失数据和信息泄露。对于这些中的每一个，都有特定的补救措施。

+ 如果您寻找公开存储库（如 UCI 机器学习存储库或 Kaggle 数据集）或咨询 Google 数据集搜索引擎，找到和获取表格数据相对容易。

+ 探索性数据分析（EDA）在帮助您清理组织内部或从网络收集的数据方面发挥着重要作用。您可以使用值计数和描述、直方图、箱线图、相关矩阵和低维投影（如 t-SNE）来揭示数据的结构和问题。
