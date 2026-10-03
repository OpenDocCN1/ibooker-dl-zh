
```py
from pgmpy.inference import VariableElimination
from pgmpy.models import NaiveBayes

model_L_given_CST = NaiveBayes()    #1
model_L_given_CST.fit(data_disc, 'L')   #1
infer_L_given_CST = VariableElimination(model_L_given_CST)   #1
#1
def p_L_given_CST(L_val, C_val, S_val, T_val):#1
    result_out = infer_L_given_CST.query(   #1
        variables=["L"],    #1
        evidence={'C': C_val, 'S': S_val, 'T': T_val},    #1
        show_progress=False    #1
    )    #1
    var_outcomes = result_out.state_names["L"]    #1
    var_values = result_out.values    #1
    prob = dict(zip(var_outcomes, var_values))    #1
    return prob[L_val]  #1
```

#1 我们将在 `pgmpy` 中使用朴素贝叶斯分类器来计算给定 `C`、`S` 和 `T` 的值的 `L` 的概率值。在这种情况下，我们将使用变量消除。

#2 我们将在 `pgmpy` 中使用朴素贝叶斯分类器来计算给定 `C`、`S` 和 `T` 的值的 `L` 的概率值。在这种情况下，我们将使用变量消除。

现在，我们将对 `P(s|c)` 执行相同的操作。

#### 列表 4.12 适配 `P(s|c)` 的朴素贝叶斯分类器

```py
model_S_given_C = NaiveBayes()    
model_S_given_C.fit(data_disc, 'S')    
infer_S_given_C = VariableElimination(model_S_given_C)    
def p_S_given_C(S_val, C_val):    
    result_out = infer_S_given_C.query(    
        variables=['S'],    
        evidence={'C': C_val},    
        show_progress=False    
    )    
    var_names = result_out.state_names["S"]    
    var_values = result_out.values    
    prob = dict(zip(var_names, var_values))    
    return prob[S_val]
```

现在，我们将这些值组合起来以实现 `h(L, T, C)` 函数。以下代码使用 `for` 循环对 `S` 进行求和。

#### 列表 4.13 合并模型以创建 `h(L, T, C)`

```py
def h_function(L, C, T):    #1
    summ = 0     #2
    for s in ["Low", "Med", "High"]:   #2
        summ += p_L_given_CST(L, C, s, T) * p_S_given_C(s, C)    #2
    return summ
```

#1 实现 `h(L, C, T)`。

#2 实现对 `P(l|c,s,t) * P(s|c)` 在 `s` 上的求和。

现在，我们将计算集合 `{C, T, L}` 的全部结果。给定这些结果，我们可以使用前面的函数计算每个组合的 `h(L, C, T)`。

#### 列表 4.14 计算 `C`、`T` 和 `L` 的结果组合

```py
ctl_outcomes = pd.DataFrame(
    [     #1
        (C, T, L)     #1
        for C in ["Low", "Med", "High"]    #1
        for T in ["Low", "High"]    #1
        for L in ["False", "True"]     #1
    ],    #1
    columns = ['C', 'T', 'L']     #1
)
```

#1 计算 `L`、`C` 和 `T` 的每个可能结果组合的这些值。首先，我们使用列表推导式创建一个包含所有组合的 `DataFrame`。

打印这些值显示了 `C`、`T` 和 `L` 的所有结果组合。

```py
       C     T      L
0    Low   Low  False
1    Low   Low   True
2    Low  High  False
3    Low  High   True
4    Med   Low  False
5    Med   Low   True
6    Med  High  False
7    Med  High   True
8   High   Low  False
9   High   Low   True
10  High  High  False
11  High  High   True
```

对于这些结果中的每一个，我们将应用 `h(L, C, T)`。

#### 列表 4.15 为 `C`、`T`、`L` 的每个结果计算 `h(L, C, T)`

```py
h_dist = ctl_outcomes.assign(    
    h_func = ctl_outcomes.apply(    
        lambda row: h_function(    
            row['L'], row['C'], row['T']), axis = 1    
    )    
)    
print(h_dist)
```

现在，对于 `C`、`T` 和 `L` 的每个联合结果，我们都有一个 `h(L, C, T)` 的值。

```py
       C     T      L    h_func
0    Low   Low  False  0.392395
1    Low   Low   True  0.607605
2    Low  High  False  0.255435
3    Low  High   True  0.744565
4    Med   Low  False  0.522868
5    Med   Low   True  0.477132
6    Med  High  False  0.369767
7    Med  High   True  0.630233
8   High   Low  False  0.495525
9   High   Low   True  0.504475
10  High  High  False  0.344616
11  High  High   True  0.655384
```

最后，我们将这个 `h_func` 分布合并到数据集中，使得对于我们的数据中的每一行，我们都能得到一个 `h(L, C, T)` 的值。

#### 列表 4.16 合并以获取数据集中每行的 `h(L, C, T)` 值

```py
df_mod = data_disc.merge(h_dist, on=['C', 'T', 'L'], how='left')    #1
print(df_mod)
```

#1 添加一个表示变量 `h(L, C, T)` 的列。

我们通过 `print(df_mod)` 看到结果：

```py
       C     S     T      L    h_func
0   High   Med   Low   True  0.504475
1    Med  High  High  False  0.369767
2    Med  High  High   True  0.630233
3    Med  High  High   True  0.630233
4    Med  High  High   True  0.630233
..   ...   ...   ...    ...       ...
95   Low  High  High   True  0.744565
96  High  High  High  False  0.344616
97   Low   Low   Low  False  0.392395
98  High   Low   Low  False  0.495525
99   Low  High  High   True  0.744565

[100 rows x 5 columns]
```

功能约束表明 `C` 和 `h(L, C, T)` 应该是独立的，因此我们可以查看 `h_func` 列和 `C` 列之间的独立性证据。由于我们对 `C` 进行了离散化，我们计算 `h(L, C, T)` 的结果在技术上也是离散的，因此我们可以使用卡方检验。但是，从理论上讲，`h(L, C, T)` 是连续的，所以我们将使用箱线图来可视化两个变量之间的依赖性。功能约束表明 `C` 和 `h(L, C, T)` 应该是独立的，因此我们将使用一个箱线图，该图将 `h(L, C, T)` 的值与 `C` 的值进行对比，以直观地检查 `C` 和 `h(L, C, T)` 是否看起来是独立的。

#### 列表 4.17 可视化 `C` 和 `h(L, C, T)` 之间的独立性

```py
df_mod.boxplot("h_func", "C")
```

这生成了图 4.21。

![figure](img/CH04_F21_Ness.png)

##### 图 4.21 在 `x` 轴上显示成本 (`C`) 的箱线图可视化，以及在 `y` 轴上显示函数 `h(L, C, T)`（标记为“乘积和”）。对于每个 `C` 值的乘积和分布的重叠支持了功能约束断言，即这两个量是独立的。

图 4.21 中的 `x` 轴表示不同的成本水平（低、中、高）。`y` 轴表示总和的值。图 4.21 是一个箱线图；每个箱子代表给定 `C` 值的总和乘积的分布。箱子的顶部和底部是分布的四分位数，箱子中间的线是中位数，较短的横向线是最大值和最小值（对于低成本，中位数、上四分位数和最大值非常接近）。总的来说，看起来总和乘积的分布在不同成本水平上变化不大；这正是独立性应有的样子。

我们还可以使用方差分析（ANOVA）方法推导出 `p` 值，这次使用 F 检验而不是卡方检验。以下代码使用 `statsmodels` 库运行 ANOVA 测试。

注意： “PR(>F)”表示对于一个给定变量（在我们的例子中是 `C`）看到 F 统计量的概率至少与从数据中计算出的 F 统计量一样大，假设该变量与 `sum_product`（即 `p` 值）独立。

#### 列表 4.18 使用 ANOVA 评估独立性

```py
from statsmodels.formula.api import ols
import statsmodels.api as sm

model = ols('h_func ~ C', data=df_mod).fit()    #1
aov_table = sm.stats.anova_lm(model, typ=2)   #1
print(aov_table["PR(>F)"]["C"])   #1

model = ols('h_func ~ T', data=df_mod).fit()    #2
aov_table = sm.stats.anova_lm(model, typ=2)    #2
print(aov_table["PR(>F)"]["T"])    #2

model = ols('h_func ~ L', data=df_mod).fit()     #3
aov_table = sm.stats.anova_lm(model, typ=2)  #3
print(aov_table["PR(>F)"]["L"])   #3
```

#1 使用 `statmodels` 库进行 ANOVA 的食谱

#2 返回高 `p` 值，支持（未能证伪）`h(L, C, T)` 和 `C` 独立的断言

#3 仅作为一个合理性检查，我们运行相同的测试来查看 `h(L, C, T)` 是否看起来独立于 `T` 和 `L`。与 `C` 不同，`T` 和 `L` 不应该独立于 `h(L, C, T)`，正如预期的那样，这些测试返回的 `p` 值要小得多，表明存在依赖性。

我们使用 `print(aov_table["PR(>F)"]["C"])` 打印出 `C` 的 `p` 值，得到约 0.1876。这个 `p` 值表明我们不能拒绝独立性的零假设，所以看起来数据支持约束。我们也对 `T` 和 `L` 进行了相同的测试，正如预期的那样，这些值要小得多，表明存在依赖性。它们都低于常见的 0.1 阈值，在这个阈值下，标准假设检验将拒绝 `h(L, C, T)` 独立于 `T` 和 `L` 的假设。

### 4.5.4 关于可测试含义的最终思考

一个有向无环图（DAG）的 d 分离和功能约束意味着，如果 DAG 是 DGP 的良好因果模型，那么在联合概率分布中应该存在某些条件独立性。我们可以通过运行条件独立性统计测试来证伪 DAG。

更普遍地说，因果模型可以对潜在的联合概率分布有不同的数学含义，其中一些可以进行测试。例如，如果你的模型假设原因和效果之间的关系是线性的，你可以在数据中寻找非线性证据（我们将在第六章中了解更多关于功能因果假设的内容）。当然，你也可以通过实验来证伪你的模型含义（我们将在第七章中看到）。

我们在因果建模方面越擅长，我们在测试和证伪我们的因果模型方面就越擅长。但请记住，不要让测试中的统计和数学细微差别分散你的注意力，你的目标是获得一个足够好的模型，并继续进行你的目标因果推断。

## 4.6 因果发现（及其风险）简介

在之前的流程中，我们提出了一个因果有向图（DAG），考虑了 DAG 对观测联合分布的潜在影响（如条件独立性），然后使用数据对这些影响进行测试。如果我们反过来操作会怎样？如果我们分析数据以寻找因果约束的统计证据，然后构建一个与这些约束一致的因果 DAG 呢？

这描述了*因果发现*的任务：从数据中统计学习因果 DAG。在本节中，我将提供因果发现的简要介绍，并涵盖你需要知道的内容，以便使用这类算法。

##### 警惕因果发现的虚假承诺

因果发现算法通常被宣传为神奇的工具，可以将任何数据集（无论质量如何有限）转换为因果 DAG。这种虚假承诺阻碍了建模 DGP（而不是数据）和证伪候选模型的思维模式。这也是为什么在实践中很难找到一致的发现用例。本节将采用一种方法，即阐述发现算法的工作原理和失败的地方，而不是列出一系列算法。我将通过关于如何有效地将这些算法融入你的分析工作流程的建议来结束。

我们将从支撑发现算法的关键思想概述开始。

### 4.6.1 因果发现方法

因果发现有几种方法。一些算法（通常称为*基于约束*算法）做的是我刚才建议的事情——从数据中条件独立性的证据中逆向工程一个图。其他算法（通常称为*基于评分*算法）将因果 DAG 转化为数据的解释模型，并找到具有高拟合度评分的因果 DAG。还有一种方法是假设因果 DAG 中父节点和子节点之间功能关系的额外约束，正如我们在第六章中将要看到的结构因果模型。

可能的 DAG 空间是一个离散空间。一类方法试图将这个空间软化成一个连续空间，并使用连续优化技术。深度学习自动微分库的流行加速了这一趋势。

由于 DAG 的空间可能相当大，因此引入先验知识以约束该空间的大小是有用的。这通常表现为指定必须存在的边或必须不存在的边，或者使用图结构的贝叶斯先验。

一些因果发现算法可以处理实验数据。这需要告诉算法哪些变量是由实验者设置的（或者从第七章开始，我们将说这些变量是“干预”的）。

要开始使用 Python 进行因果发现，我推荐使用 PyWhy 库进行因果发现，如 `causal-learn` 和 `DoDiscover`。

### 4.6.2 因果发现、因果忠实性和潜在变量假设

因果马尔可夫性质假设如果我们的 DAG 是真实的，那么该 DAG 中的 d 分离意味着变量联合概率中的条件独立性陈述：

![figure](img/ness-ch4-eqs-2x.png)

*因果忠实性*（或简称“忠实性”）是相反的陈述——联合分布中的条件独立性意味着图中 d 分离：

![figure](img/ness-ch4-eqs-3x.png)

许多因果发现算法依赖于忠实性成立的假设。这可能并不成立。

#### 发现与忠实性违反

在 4.4 节中，我们使用了马尔可夫性质来测试一个候选 DAG；给定一个对于 DAG 成立的 d 分离语句，我们运行一个统计测试来检查该 d 分离所隐含的条件独立性的经验证据。

想象一下，你想要通过逆向构建你的图。你在数据中检测到条件独立性的实例证据，然后你将候选 DAG 的空间限制为与隐含的 d 分离一致的 DAG。你这样迭代进行，直到你缩小了候选 DAG 的空间。一些发现算法执行某种版本的此过程，并且那些执行此过程的算法依赖于忠实性假设。

注意：将条件独立性的证据与 d 分离相匹配的算法通常被称为“基于约束”的发现算法。一个著名的例子是 PC 算法。基于约束的算法找到与因果性经验证据一致的 DAG。
