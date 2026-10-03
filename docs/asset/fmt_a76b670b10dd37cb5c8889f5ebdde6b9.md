
当你处理高维输入（如图像）时，这项技术特别有用，因为它可以显著加快训练速度。同时采样训练实例和特征被称为[*随机补丁*方法](https://homl.info/22)。⁠^(8) 保持所有训练实例（通过设置`bootstrap=False`和`max_samples=1.0`），但采样特征（通过设置`bootstrap_features`为`True`和/或`max_features`为一个小于`1.0`的值）被称为[*随机子空间*方法](https://homl.info/23)。⁠^(9)

采样特征导致预测器多样性更高，以略微增加的偏差换取更低的方差。

## 随机森林

正如我们讨论过的，一个 [随机森林](https://homl.info/24)⁠^(10) 是决策树的集成，通常通过 bagging 方法（或有时是粘贴）训练，通常将 `max_samples` 设置为训练集的大小。您不需要构建一个 `BaggingClassifier` 并传递一个 `DecisionTreeClassifier`，而是可以使用 `RandomForestClassifier` 类，它更方便且针对决策树进行了优化⁠^(11)（类似地，还有一个 `RandomForestRegressor` 类用于回归任务）。以下代码使用所有可用的 CPU 核心，训练了一个包含 500 棵树、每棵树最多 16 个叶节点的随机森林分类器：

```py
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16,
                                 n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)

y_pred_rf = rnd_clf.predict(X_test)
```

除了少数例外，`RandomForestClassifier` 具有与 `DecisionTreeClassifier` 相同的所有超参数（用于控制树的生长方式），以及 `BaggingClassifier` 的所有超参数，用于控制集成本身。

`RandomForestClassifier` 类在生长树时引入了额外的随机性：在分割节点时，它不是搜索最佳特征（参见第五章），而是在特征的一个随机子集中搜索最佳特征。默认情况下，它采样 $StartRoot n EndRoot$ 个特征（其中 *n* 是特征的总数）。该算法导致树多样性增加，这（再次）以更高的偏差换取了更低的方差，通常会产生更好的整体模型。因此，以下 `BaggingClassifier` 等同于之前的 `RandomForestClassifier`：

```py
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(max_features="sqrt", max_leaf_nodes=16),
    n_estimators=500, n_jobs=-1, random_state=42)
```

### 额外树

当在随机森林中生长一棵树时，在每个节点上仅考虑随机子集的特征进行分割（如前所述）。通过为每个特征使用随机阈值而不是搜索最佳可能的阈值（如常规决策树所做的那样），可以使树变得更加随机。为此，在创建 `DecisionTreeClassifier` 时，只需将 `splitter="random"` 设置即可。

这样的极端随机树森林被称为 [*极端随机树*](https://homl.info/25)⁠^(12)（或简称为 *extra-trees*）集成。再次强调，这种技术以更多的偏差换取了更低的方差，因此，如果您遇到过拟合，尤其是在有噪声和/或高维数据集的情况下，它们可能比常规随机森林表现更好。与常规随机森林相比，额外树分类器训练得也更快，因为找到每个节点上每个特征的最佳可能阈值是随机森林中生长树时最耗时的任务之一。

您可以使用 Scikit-Learn 的 `ExtraTreesClassifier` 类创建一个额外的树分类器。它的 API 与 `RandomForestClassifier` 类相同，只是 `bootstrap` 默认为 `False`。同样，`ExtraTreesRegressor` 类的 API 与 `RandomForestRegressor` 类相同，只是 `bootstrap` 默认为 `False`。

> **提示**
> 
> 正如决策树类别一样，最近 Scikit-Learn 版本的随机森林类别和额外树类别原生支持缺失值，无需使用填充器。

### 特征重要性

随机森林的另一个优点是它们使得测量每个特征的相对重要性变得容易。Scikit-Learn 通过查看使用该特征的树节点在森林中所有树上的平均纯度降低程度来衡量一个特征的重要性。更精确地说，这是一个加权平均值，其中每个节点的权重等于与之关联的训练样本数量（参见第五章 Chapter 5）。

Scikit-Learn 在训练后自动为每个特征计算这个分数，然后调整结果，使得所有重要性的总和等于 1。你可以通过`feature_importances_`变量访问结果。例如，以下代码在鸢尾花数据集（在第四章 Chapter 4 中介绍）上训练了一个`RandomForestClassifier`，并输出了每个特征的重要性。看起来最重要的特征是花瓣长度（44%）和宽度（42%），而与花瓣长度和宽度相比，萼片长度和宽度则相对不重要（分别为 11%和 2%）：

```py
>>> from sklearn.datasets import load_iris
>>> iris = load_iris(as_frame=True)
>>> rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42)
>>> rnd_clf.fit(iris.data, iris.target)
>>> for score, name in zip(rnd_clf.feature_importances_, iris.data.columns):
...     print(round(score, 2), name)
...
0.11 sepal length (cm)
0.02 sepal width (cm)
0.44 petal length (cm)
0.42 petal width (cm)
```

类似地，如果你在 MNIST 数据集（在第三章 Chapter 3 中介绍）上训练一个随机森林分类器，并绘制每个像素的重要性，你将得到图 6-6 中所示的形象。

![显示 MNIST 像素重要性热图，颜色表示根据随机森林分类器特征显著性的颜色，其中黄色表示高重要性，红色表示低重要性。](img/hmls_0606.png)

**图 6-6\. MNIST 像素重要性（根据随机森林分类器）**

随机森林非常适合快速了解哪些特征真正重要，特别是如果你需要执行特征选择。

## 提升

*提升*（最初称为*假设提升*）指的是任何可以将多个弱学习器组合成强学习器的集成方法。大多数提升方法的一般思想是顺序训练预测器，每个预测器都试图纠正其前驱。有许多提升方法可用，但最流行的是[*AdaBoost*](https://homl.info/26)⁠^(13)（即*自适应提升*）和*梯度提升*。让我们从 AdaBoost 开始。

### AdaBoost

新预测器纠正其前驱的一种方法是对前驱欠拟合的训练实例给予更多的关注。这导致新预测器越来越多地关注困难案例。这是 AdaBoost 使用的技术。

例如，当训练一个 AdaBoost 分类器时，算法首先训练一个基分类器（例如决策树）并使用它对训练集进行预测。然后算法增加被错误分类的训练实例的相对权重。接着它使用更新后的权重训练第二个分类器，并在训练集上再次进行预测，更新实例权重，依此类推（参见 图 6-7）。

![说明 AdaBoost 顺序训练过程的图，展示了实例权重和预测如何通过阶段更新。](img/hmls_0607.png)

**图 6-7\. AdaBoost 顺序训练与实例权重更新**

图 6-8 展示了五个连续预测器在月亮数据集上的决策边界（在这个例子中，每个预测器是一个具有 RBF 内核的高度正则化的 SVM 分类器）。⁠^(14) 第一个分类器错误地分类了许多实例，因此它们的权重被提升。因此，第二个分类器在这些实例上做得更好，依此类推。右侧的图表示相同的预测器序列，除了学习率减半（即，在每次迭代中错误分类的实例权重提升较少）。如您所见，这种顺序学习技术与梯度下降有一些相似之处，除了 AdaBoost 不是调整单个预测器的参数以最小化成本函数，而是将预测器添加到集成中，逐渐使其变得更好。

![展示五个连续 SVM 分类器在月亮数据集上的决策边界的两个图，学习率分别为 1 和 0.5，说明了学习率对 AdaBoost 性能的影响。](img/hmls_0608.png)

**图 6-8\. 连续预测器的决策边界**

> **警告**
> 
> 这种顺序学习技术有一个重要的缺点：由于每个预测器只能在之前的预测器被训练和评估之后才能进行训练，因此无法并行化。因此，它的扩展性不如 bagging 或 pasting。

一旦所有预测器都经过训练，集成进行预测的方式与 bagging 或 pasting 非常相似，除了预测器的权重取决于它们在加权训练集上的整体准确性。

让我们更仔细地看看 AdaBoost 算法。每个实例权重 *w*^((*i*)) 初始设置为 1/*m*，因此它们的总和为 1。首先训练一个预测器，并在训练集上计算其加权错误率 *r*[1]；参见 方程式 6-1。

#### 方程式 6-1\. 第 j 个预测器的加权错误率

<mrow><msub><mi>r</mi><mi>j</mi></msub> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="true"><mrow><munderover><mo>∑</mo> <mstyle scriptlevel="0" displaystyle="false"><mrow><mfrac linethickness="0pt"><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mrow><msubsup><mover accent="true"><mi>y</mi><mo>^</mo></mover> <mi>j</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup> <mo>≠</mo> <msup><mi>y</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup></mrow></mfrac></mrow></mstyle> <mi>m</mi></munderover> <msup><mi>w</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup></mrow></mstyle> <mtext>其中</mtext> <msubsup><mover accent="true"><mi>y</mi><mo>^</mo></mover> <mi>j</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup> <mtext>是</mtext> <mtext>第</mtext> <msup><mi>j</mi><mtext>个</mtext></msup> <mtext>预测器的</mtext> <mtext>预测结果</mtext><mtext>针对</mtext> <mtext>第</mtext> <msup><mi>i</mi> <mtext>个</mtext></msup> <mtext>实例</mtext></mrow>

预测器的权重 *α*[*j*] 然后通过方程 6-2 计算，其中 *η* 是学习率超参数（默认为 1）。⁠^(15) 预测器越准确，其权重越高。如果只是随机猜测，那么其权重将接近零。然而，如果它通常错误（即，不如随机猜测准确），那么其权重将是负数。

#### 方程 6-2\. 预测器权重

$dollar-sign alpha Subscript j Baseline equals eta log StartFraction 1 minus r Subscript j Baseline Over r Subscript j Baseline EndFraction dollar-sign$

接下来，AdaBoost 算法更新实例权重，使用方程 6-3，这增加了错误分类实例的权重，并鼓励下一个预测器更加关注它们。

#### 方程 6-3\. 权重更新规则

$dollar-sign StartLayout 1st Row 1st Column Blank 2nd Column for i equals 1 comma 2 comma period period period comma m 2nd Row 1st Column Blank 2nd Column w Superscript left-parenthesis i right-parenthesis Baseline left-arrow StartLayout Enlarged left-brace 1st Row  w Superscript left-parenthesis i right-parenthesis Baseline if ModifyingAbove y With caret Superscript left-parenthesis i right-parenthesis Baseline equals y Superscript left-parenthesis i right-parenthesis Baseline 2nd Row  w Superscript left-parenthesis i right-parenthesis Baseline exp left-parenthesis alpha Subscript j Baseline right-parenthesis if ModifyingAbove y With caret Superscript left-parenthesis i right-parenthesis Baseline not-equals y Superscript left-parenthesis i right-parenthesis EndLayout EndLayout dollar-sign$

然后将所有实例权重归一化，以确保它们的总和再次为 1（即，它们被除以 $sigma-summation Underscript i equals 1 Overscript m Endscripts w Superscript left-parenthesis i right-parenthesis$ )。

最后，使用更新后的权重训练一个新的预测器，整个过程重复进行：计算新预测器的权重，更新实例权重，然后训练另一个预测器，依此类推。算法在达到所需预测器数量或找到完美预测器时停止。

为了进行预测，AdaBoost 简单地计算所有预测器的预测值，并使用预测器权重 *α*[*j*] 对其进行加权。预测类别是获得加权投票多数的类别（见方程 6-4）。

#### 方程 6-4\. AdaBoost 预测

<mrow><mover accent="true"><mi>y</mi> <mo>^</mo></mover> <mrow><mo>(</mo> <mi mathvariant="bold">x</mi> <mo>)</mo></mrow> <mo>=</mo> <munder><mo form="prefix">argmax</mo> <mi>k</mi></munder> <mrow><munderover><mo>∑</mo> <mfrac linethickness="0pt"><mstyle scriptlevel="1" displaystyle="false"><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow></mstyle> <mstyle scriptlevel="1" displaystyle="false"><mrow><msub><mover accent="true"><mi>y</mi> <mo>^</mo></mover> <mi>j</mi></msub> <mrow><mo>(</mo><mi mathvariant="bold">x</mi><mo>)</mo></mrow><mo>=</mo><mi>k</mi></mrow></mstyle></mfrac> <mi>N</mi></munderover> <msub><mi>α</mi> <mi>j</mi></msub></mrow> <mtext>其中</mtext> <mi>N</mi> <mtext>是</mtext> <mtext>预测器</mtext> <mtext>的数量</mtext></mrow>

Scikit-Learn 使用 AdaBoost 的多分类版本，称为 [*SAMME*](https://homl.info/27)⁠^(16)（代表 *使用多类指数损失函数的阶跃式加性建模*）。当只有两个类别时，SAMME 等同于 AdaBoost。

以下代码使用 Scikit-Learn 的 `AdaBoostClassifier` 类（正如你所期望的，还有一个 `AdaBoostRegressor` 类）训练了一个基于 30 个 *决策树桩* 的 AdaBoost 分类器。决策树桩是一个 `max_depth=1` 的决策树——换句话说，由一个决策节点和两个叶节点组成的树。这是 `AdaBoostClassifier` 类的默认基础估计器。
