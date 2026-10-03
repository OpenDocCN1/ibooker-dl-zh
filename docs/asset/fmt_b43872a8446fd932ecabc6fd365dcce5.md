
我们可以通过将`chi_square`函数的`boolean`参数设置为`True`直接跳到测试的结果。然后，如果*p*-值大于显著性值（有利于条件独立性），函数将返回`True`；否则返回`False`（有利于依赖性）。

### 列表 4.5 带布尔结果的卡方测试

```py
from pgmpy.estimators.CITests import chi_square     #1
significance = .05     #2
result = chi_square(     #3
    X="E", Y="T", Z=["O", "R"],   #3
    data=data,    #3
    boolean=True,    #3
    significance_level=significance  #3   
)    #3
print(result)
```

#1 导入卡方测试函数。

#2 将显著性水平设置为`.05`。

#3 当`boolean`参数设置为`True`时，测试返回一个简单的`True`或`False`结果。如果*p*值大于显著性值，则表示支持条件独立性，它将返回`True`。否则，它返回`False`，表示支持依赖性。

这将打印出结果`True`。现在让我们遍历从运输图可以推导出的所有 d-separation 声明，并逐一测试它们。以下脚本将打印出每个 d-separation 声明以及相应的条件独立性测试的结果。

### 列表 4.6 对每个 d-separation 声明运行卡方测试

```py
from pprint import pprint
from pgmpy.base import DAG
from pgmpy.independencies import IndependenceAssertion

dag = DAG([
    ('A', 'E'),
    ('S', 'E'),
    ('E', 'O'),
    ('E', 'R'),
    ('O', 'T'),
    ('R', 'T')
])
dseps = dag.get_independencies()    

def test_dsep(dsep):
    test_outputs = []
    for X in list(dsep.get_assertion()[0]):
        for Y in list(dsep.get_assertion()[1]):
            Z = list(dsep.get_assertion()[2])
            test_result = chi_square(
                X=X, Y=Y, Z=Z,
                data=data,
                boolean=True,
                significance_level=significance
            )
            assertion = IndependenceAssertion(X, Y, Z)
            test_outputs.append((assertion, test_result))
    return test_outputs

results = [test_dsep(dsep) for dsep in dseps.get_assertions()]
results = dict([item for sublist in results for item in sublist])
pprint(results)
```

结果是一个包含 d-separation 声明及其是否支持（或未能反驳）该声明的数据证据的列表。

```py
{(O ⊥ A | R, E, T, S): True,
 (S ⊥ R | E, T, A): True,
 (S ⊥ O | E, T, A): True,
 (T ⊥ S | R, O, A): True,
 (S ⊥ O | R, E): True,
 (R ⊥ O | E): False,
 (S ⊥ O | E, A): True,
 (S ⊥ R | E, A): True,
 (S ⊥ R | E, T, O, A): True,
 (S ⊥ R | E, O, A): True,
 (O ⊥ A | E, T): True,
 (S ⊥ O | R, E, T): True,
 (R ⊥ O | E, S): False, 
 …
 (T ⊥ A | E, S): True}
```

我们可以计算通过测试的数量。

### 列表 4.7 计算通过测试的 d-separation 的比例

```py
num_pass = sum(results.values())
num_dseps = len(dseps.independencies)
num_fail = num_dseps - num_pass
print(num_fail / num_dseps)
```

这里我们得到`0.2875`。这意味着 29% 的 d-separation 在数据中缺乏相应的条件独立性证据。

这个数字似乎很高，但正如我们将在 4.4 节中看到的，这个统计量取决于数据的大小和其他因素。我们将想要将其与其他候选 DAG 的结果进行比较。目前，下一步是检查这些明显的依赖性案例，其中我们的 DAG 表示应该存在条件独立性。如果依赖性的证据很强，我们需要考虑如何改进我们的因果 DAG 来解释它。

之前，我使用了`chi_square`函数，该函数使用卡方测试分布构建特定的测试统计量——用于计算*p*-值的分布。卡方分布只是另一种典型分布，类似于正态分布或伯努利分布。卡方分布经常用于离散变量，因为在离散设置中，有几个测试统计量要么具有卡方分布，要么随着数据量的增加而越来越接近卡方分布。总的来说，独立性测试具有各种测试统计量和不同的测试分布。pgmpy 通过调用 SciPy 的 stats 库提供了几种选项。

一个常见的担忧是测试做出了强烈的假设。例如，一些连续变量之间的条件独立性测试假设变量之间的任何依赖性都是*线性的*。另一种方法是使用*排列*测试，这是一种算法，它构建*p*-值而不依赖于典型测试分布*。排列测试做出的假设较少，但计算成本较高。

### 4.3.3 一些测试比其他测试更重要

之前的分析测试了由因果 DAG 暗示的所有 d-separation。但有些 d-separation 可能比其他 d-separation 更重要。一些依赖关系和条件独立性关系对于下游因果推断分析至关重要，而其他关系则根本不影响该分析。

例如，考虑图 4.17，我们在第 3.3 节中已经讨论过。我们添加变量*Z*到图中，因为我们可能希望将其用作估计因果效应的“工具变量”。

![figure](img/CH04_F17_Ness.png)

#### 图 4.17 中包含*Z*，*X*[0] 和 *X*[1]，因为它们在分析*U*对*Y*的因果效应中发挥了有用的作用。它们的作用取决于条件独立性，并且测试它们确实能够扮演这些角色是很重要的。

我们将在第十一章中深入讨论工具变量。现在，只需说，为了*Z*成为工具变量，它必须与*W*[0]、*W*[1] 和 *W*[2] 独立。因此，我们将特别注意测试这个假设。

## 4.4 条件独立性测试的注意事项

正如我提到的，条件独立性测试可能是测试你提出的因果 DAG 中隐含的条件独立性约束的最明显的方法。然而，使用统计测试来测试因果 DAG 存在一些注意事项。根据我的经验，这些问题可能会让分析师偏离他们回答因果问题的最终目标。在本节中，我将强调一些这些注意事项，并提出一些条件独立性测试的替代方案。主要启示是，统计测试是构建你的 DAG 的重要工具，但与任何统计方法一样，它不是万能的（这是可以的）。

### 4.4.1 统计测试总是存在一些错误的可能性

我提到，使用 d-separation 时，我们不应该“把地图当作地形”；d-separation 与条件独立性不是一回事。相反，如果你的模型是因果性的良好表示，d-separation*意味着*条件独立性。

同样，条件独立性并不等同于条件独立性的统计证据。DGP 的因果结构对联合概率分布施加了条件独立性约束。但你无法“看到”联合分布及其包含的独立性；你只能“看到”（并对从该分布中抽取的数据进行统计测试）。

就像预测、分类或任何其他统计模式识别过程一样，检测数据中这些独立性的方法可能会出错。你可能会得到假阴性，即一对变量实际上是条件独立的，但统计独立性测试却得出它们是相关的结论。你可能会得到假阳性，即统计独立性测试发现一对变量是条件独立的，而实际上它们并不是。

### 4.4.2 使用传统的 CI 测试来测试因果 DAG 是有缺陷的

我认为所提出的用于反驳的条件独立性检验是“有缺陷的”，因为它们违反了科学中统计假设检验的精神。假设你认为你在股票价格中发现了某种模式。你倾向于认为这种模式不仅仅是巧合，因为如果是巧合，你可以从中赚钱。为了严谨，不受到你的偏见的影响，你的备择假设应该是这种模式是真实的且可利用的，而零假设则是它只是随机噪声。频率派测试假设零假设是真实的，并给你一个*p*值，这个*p*值量化了随机噪声形成至少与你发现的模式一样强的模式的概率。这个测试迫使你拒绝这种模式是真实的，除非那个*p*值非常小。大多数主流的统计测试库都是为了这种用例设计的。

当你提出一个因果模型时，你也倾向于相信它是真实的。但因果模型会诱导条件独立性，根据定义，这些是*不存在*的模式。在这种情况下，零假设和备择假设应该互换；备择假设应该是你的模型是正确的，没有模式（并且数据中的任何模式证据只是虚假的相关性），而零假设应该是存在模式。可以实施这样的假设检验，但这在数学上并不简单，而且大多数主流的统计库，如 SciPy，不支持这种用例。

妥协是使用传统的测试，其中零假设将条件独立性更多地指定为一种理论上的严谨分析，而不是一种*启发式*——一种经验性的问题解决技术，它可能不是最优的，但足以达到足够好的解决方案。

### 4.4.3 *p*值随数据大小变化

传统条件独立性检验的结论取决于一个显著性阈值。如果*p*值低于这个阈值，你倾向于认为存在依赖性；如果它高于这个阈值，你倾向于认为存在条件独立性。阈值的选择有点随意；人们倾向于选择常见的值，如 0.1、0.05 或 0.01。

问题在于*p*值统计量随着数据大小的变化而变化。在其他条件相同的情况下，随着数据大小的增加，*p*值会降低。换句话说，数据越大，事情开始看起来越有依赖性。如果你有一个大型数据集，*p*值更有可能低于那个任意的阈值，数据看起来就像是在反驳你的 DAG 所暗示的条件独立性，即使这种条件独立性是真实的。

为了说明，4.3.2 节中关于*E* ⊥ *T* | *O*, *R* 的检验有 30 个数据点，产生了 0.56 的 *p* 值。在我们的数据中，*E* ⊥ *T* | *O*, *R* 是真实情况（通过模拟获得），因此如果测试得出与*E* ⊥ *T* | *O*, *R* 相反的结论，那是因为测试的统计问题，而不是数据的质量。下面的自助统计分析将展示随着数据集大小的增加，p 值的估计如何下降。

首先，我们将编写一个`sample_p_value`函数，用于为给定数据大小抽取一个*p*值。下一个函数`estimate_p_value`将重复进行此抽样并计算平均*p*值、90%置信区间以及 p 值低于显著性阈值的概率，即拒绝*E* ⊥ *T* | *O*, *R* 正确结论的概率。

#### 列表 4.8 *E* **⊥** *T* | *O*, *R* 测试对样本大小的敏感性自助分析

```py
from numpy import mean, quantile

def sample_p_val(data_size, data, alpha):     #1
    bootstrap_data = data.sample(n=data_size, replace=True)    #1
    result = chi_square(    #1
        X="E", Y="T", Z=["O", "R"],    #1
        data=bootstrap_data,     #1
        boolean=False,     #1
        significance_level = alpha     #1
    )    #1
    p_val = result[1]    #1
    return p_val     #1

def estimate_p_val(data_size, data=fulldata, boot_size=1000, α=.05):    #2
    samples = [  #2
        sample_p_val(data_size, data=fulldata, alpha=α)  #2
        for _ in range(boot_size)     #2
    ]    #2
    positive_tests = [p_val > significance for p_val in samples]    #3
    prob_conclude = mean(positive_tests)    #4
    p_estimate = mean(samples)    #4
    quantile_05, quantile_95 = quantile(samples, [.05, .95])    #5
    lower_error = p_estimate - quantile_05     #5
    higher_error = quantile_95 - p_estimate    #5
    return p_estimate, lower_error, higher_error, prob_conclude

data_size = range(30, 1000, 20)     #6
result = list(zip(*[estimate_p_val(size) for size in data_size]))  #6
```

#1 给定一定大小的数据集，此函数从完整数据集中随机抽取相应数量的行。然后运行卡方独立性检验并返回 p 值。

#2 此函数执行一个“自助”过程，为给定数据大小抽取 1,000 个*p*值，并计算平均*p*值和 90% *p*值置信区间。

#3 计算支持条件独立性的测试结论的概率。

#4 计算 p 值的平均值以获得自助平均值。

#5 计算第 5 和第 95 个百分位数以获得 90%的自助置信区间。

#6 运行自助分析。

最后，我们将可视化结果。我们将绘制数据大小与给定数据大小得到的*p*值的平均值和 90%置信区间的对比图。我们还将绘制在显著性水平为.05 的情况下，支持真实假设（*E* ⊥ *T* | *O*, *R*）的概率如何依赖于数据大小。

#### 列表 4.9 可视化条件独立性测试对数据大小的依赖关系

```py
import numpy as np
import matplotlib.pyplot as plt

p_vals, lower_bars, higher_bars, probs_conclude_indep = result    #1
plt.title('Data size vs. p-value (Ind. of E & T | O & R)')     #2
plt.xlabel("Number of examples in data")     #2
plt.ylabel("Expected p-value")   #2
error_bars = np.array([lower_bars, higher_bars])    #2
plt.errorbar(   #2
    data_size,    #2
    p_vals,     #2
    yerr=error_bars,    #2
    ecolor="grey",    #2
    elinewidth=.5   #2
)     #2
plt.hlines(significance, 0, 1000, linestyles="dashed")     #2
plt.show()
plt.title('Probability of favoring independence given data size')    #3
plt.xlabel("Number of examples in data")     #3
plt.ylabel("Probability of test favoring conditional independence")    #3
plt.plot(data_size, probs_conclude_indep)  #3
```

#1 运行自助分析以获取 p 值的分位数和得出独立性支持的结论的概率。

#2 绘制数据大小与 p 值的关系图。在较大的数据集大小下，预期的 p 值会低于阈值。

#3 绘制数据大小与在.05 显著性水平下支持独立性的概率的关系图。

图 4.18 显示了第一个图表。下降的曲线是不同数据大小下的预期*p*值，垂直线表示显示 90%自助置信区间的误差线。当我们得到 1,000 个数据点的数据集时，预期的*p*值低于阈值，这意味着测试倾向于得出*E* ⊥ *T* | *O*, *R* 是错误的结论。

![figure](img/CH04_F18_Ness.png)

#### 图 4.18 条件独立性测试的样本量与预期*p*-值的关系图（实线）。垂直线是误差线；它们表示 90%的 bootstrap 置信区间。水平虚线是 0.05 的显著性水平，在此之上我们倾向于接受条件独立性的零假设，在此之下我们拒绝它。随着样本量的增加，我们最终会越过这条线。因此，我们的反驳分析结果取决于数据规模。

注意到置信区间的下限在 1,000 之前就已经跨过了显著性阈值，这表明即使在更小的数据规模下，我们也有很好的机会拒绝*E* ⊥ *T* | *O*, *R* 的真实结论。这在图 4.19 中更为明显，随着数据规模的增加，得出支持真实结论的概率降低。

![figure](img/CH04_F19_Ness.png)
