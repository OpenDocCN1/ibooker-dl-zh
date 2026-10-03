
```py
from copy import deepcopy
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

X_train, y_train, X_valid, y_valid = [...]  # split the quadratic dataset

preprocessing = make_pipeline(PolynomialFeatures(degree=90, include_bias=False),
                              StandardScaler())
X_train_prep = preprocessing.fit_transform(X_train)
X_valid_prep = preprocessing.transform(X_valid)
sgd_reg = SGDRegressor(penalty=None, eta0=0.002, random_state=42)
n_epochs = 500
best_valid_rmse = float('inf')

for epoch in range(n_epochs):
    sgd_reg.partial_fit(X_train_prep, y_train)
    y_valid_predict = sgd_reg.predict(X_valid_prep)
    val_error = root_mean_squared_error(y_valid, y_valid_predict)
    if val_error < best_valid_rmse:
        best_valid_rmse = val_error
        best_model = deepcopy(sgd_reg)
```

此代码首先添加多项式特征，并将训练集和验证集的所有输入特征进行缩放（代码假设您已将原始训练集分割成较小的训练集和验证集）。然后，它创建一个没有正则化且学习率较小的`SGDRegressor`模型。在训练循环中，它调用`partial_fit()`而不是`fit()`，以执行增量学习。在每个 epoch，它测量验证集上的 RMSE。如果它低于迄今为止看到的最低 RMSE，它将在`best_model`变量中保存模型的副本。此实现实际上并没有停止训练，但它让您在训练后回退到最佳模型。请注意，使用`copy.deepcopy()`复制模型，因为它复制了模型的超参数和学习的参数。相比之下，`sklearn.base.clone()`只复制模型的超参数。

# 逻辑回归

如第一章所述，一些回归算法可以用于分类（反之亦然）。*逻辑回归*（也称为*logit 回归*）通常用于估计实例属于特定类别的概率（例如，这封电子邮件是垃圾邮件的概率是多少？）。如果估计的概率大于给定的阈值（通常是 50%），则模型预测该实例属于该类别（称为*正类*，标记为“1”），否则它预测它不属于（即，它属于*负类*，标记为“0”）。这使得它成为一个二元分类器。

## 估计概率

那么，逻辑回归是如何工作的呢？就像线性回归模型一样，逻辑回归模型计算输入特征的加权总和（加上一个偏差项），但它不像线性回归模型那样直接输出结果，而是输出这个结果的*逻辑值*（参见方程式 4-14）。

##### 方程式 4-14\. 逻辑回归模型估计概率（向量形式）

\[mrow\]\[mover accent="true"\]\[mi\]p\[mo\]^\[mo\]\[msub\]\[mi\]h\[mi mathvariant="bold"\]θ\[msub\]\[mrow\]\mo\\[mo\]\[mrow\]\[mo\]=\[mi\]σ\[mrow\]\mo\\[mo\]\[mrow\]\[mo\]\]

逻辑函数——表示为*σ*(·)—是一个*S 型函数*（即，*S*形），输出一个介于 0 和 1 之间的数字。它定义如方程式 4-15 和图 4-22 所示。

##### 方程式 4-15\. 逻辑函数

<mrow><mi>σ</mi> <mrow><mo>(</mo> <mi>t</mi> <mo>)</mo></mrow> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="true"><mfrac><mn>1</mn> <mrow><mn>1</mn><mo>+</mo><mo form="prefix">exp</mo><mo>(</mo><mo lspace="0%" rspace="0%">-</mo><mi>t</mi><mo>)</mo></mrow></mfrac></mstyle></mrow>![说明逻辑函数的图形，显示输入值增加时从 0 到 1 上升的 S 形曲线。](img/hmls_0422.png)

###### 图 4-22\. 逻辑函数

一旦逻辑回归模型估计了实例 **x** 属于正类的概率 <mover><mi>p</mi><mo>^</mo></mover> = *h***θ**，它就可以轻松地做出预测 *ŷ*（参见 方程 4-16）。

##### 方程 4-16\. 使用 50% 阈值概率的逻辑回归模型预测

<mrow><mover accent="true"><mi>y</mi> <mo>^</mo></mover> <mo>=</mo> <mfenced separators="" open="{" close=""><mtable><mtr><mtd columnalign="left"><mn>0</mn></mtd> <mtd columnalign="left"><mrow><mtext>if</mtext> <mover accent="true"><mi>p</mi> <mo>^</mo></mover> <mo><</mo> <mn>0.5</mn></mrow></mtd></mtr> <mtr><mtd columnalign="left"><mn>1</mn></mtd> <mtd columnalign="left"><mrow><mtext>if</mtext> <mover accent="true"><mi>p</mi> <mo>^</mo></mover> <mo>≥</mo> <mn>0.5</mn></mrow></mtd></mtr></mtable></mfenced></mrow>

注意到当 *t* < 0 时，*σ*(*t*) < 0.5，而当 *t* ≥ 0 时，*σ*(*t*) ≥ 0.5，因此使用默认 50% 概率阈值的逻辑回归模型在 **θ**^⊺ **x** 为正时预测 1，为负时预测 0。

###### 注意

分数 *t* 通常被称为 *logit*。这个名字来源于 logit 函数，定义为 logit(*p*) = log(*p* / (1 – *p*))，它是逻辑函数的逆函数。确实，如果你计算估计概率 *p* 的 logit，你会发现结果是 *t*。logit 也被称为 *log-odds*，因为它是在正类估计概率与负类估计概率之间的比值的对数。

## 训练和损失函数

现在你已经知道了逻辑回归模型如何估计概率和做出预测。但是它是如何训练的呢？训练的目标是设置参数向量 **θ**，使得模型对正实例（*y* = 1）估计高概率，对负实例（*y* = 0）估计低概率。这个想法由 方程 4-17 中单个训练实例 **x** 的损失函数所体现。

##### 方程 4-17\. 单个训练实例的损失函数

<mrow><mi>c</mi><mo>(</mo><mi mathvariant="bold">θ</mi><mo>)</mo></mrow><mo>=</mo><mrow><mfenced open="{" close=""><mtable><mtr><mtd><mo>-</mo><mi>log</mi><mo>(</mo><mover accent="true"><mi>p</mi><mo>^</mo></mover><mo>)</mo></mtd><mtd><mtext>if </mtext><mi>y</mi><mo>=</mo><mn>1</mn></mtd></mtr><mtr><mtd><mo>-</mo><mi>log</mi><mo>(</mo><mn>1</mn><mo>-</mo><mover accent="true"><mi>p</mi><mo>^</mo></mover><mo>)</mo></mtd><mtd><mtext>if </mtext><mi>y</mi><mo>=</mo><mn>0</mn></mtd></mtr></mtable></mfenced></mrow>

这个代价函数是有意义的，因为当 *t* 接近 0 时，-log(*t*) 会变得非常大，所以如果模型估计一个正例的概率接近 0，代价就会很大；同样，如果模型估计一个负例的概率接近 1，代价也会很大。另一方面，当 *t* 接近 1 时，-log(*t*) 接近 0，所以如果估计的概率对于一个负例接近 0 或对于一个正例接近 1，代价就会接近 0，这正是我们想要的。

整个训练集上的代价函数是所有训练实例的平均代价。它可以写成单个表达式，称为 *对数损失*，如方程 4-18 所示。

##### 方程 4-18\. 逻辑回归代价函数（对数损失）

<mrow><mi>J</mi><mo>(</mo><mi mathvariant="bold">θ</mi><mo>)</mo></mrow><mo>=</mo><mrow><mo>-</mo><mfrac><mn>1</mn><mi>m</mi></mfrac><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><mfenced open="[" close="]"><mrow><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mi>l</mi><mi>o</mi><mi>g</mi><mfenced><msup><mover accent="true"><mi>p</mi><mo>^</mo></mover><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup></mfenced><mo>+</mo><mo>(</mo><mn>1</mn><mo>-</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo><mi>l</mi><mi>o</mi><mi>g</mi><mfenced><mrow><mn>1</mn><mo>-</mo><msup><mover accent="true"><mi>p</mi><mo>^</mo></mover><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup></mrow></mfenced></mrow></mfenced></mrow>

###### 警告

对数损失并不是随意提出的。可以通过数学方法（使用贝叶斯推理）证明，最小化这个损失会导致模型具有最大的似然性，假设实例围绕其类别的均值呈高斯分布。当你使用对数损失时，这就是你隐含的假设。这个假设越错误，模型就会越有偏差。同样，当我们使用均方误差（MSE）来训练线性回归模型时，我们隐含地假设数据是纯线性的，加上一些高斯噪声。所以，如果数据不是线性的（例如，如果是二次的）或者噪声不是高斯分布的（例如，如果异常值不是指数级稀少的），那么模型就会存在偏差。

坏消息是，没有已知的闭式方程来计算最小化此成本函数的 **θ** 值（没有正常方程的等价物）。但好消息是，这个成本函数是凸的，所以梯度下降（或任何其他优化算法）保证能够找到全局最小值（如果学习率不是太大，并且等待足够长的时间）。关于第 *j* 个模型参数 *θ*[*j*] 的成本函数的偏导数由公式 4-19 给出。

##### 公式 4-19\. 逻辑成本函数的偏导数

<mrow><mstyle scriptlevel="0" displaystyle="true"><mfrac><mi>∂</mi> <mrow><mi>∂</mi><msub><mi>θ</mi> <mi>j</mi></msub></mrow></mfrac></mstyle> <mtext>J</mtext> <mrow><mo>(</mo> <mi mathvariant="bold">θ</mi> <mo>)</mo></mrow> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="true"><mfrac><mn>1</mn> <mi>m</mi></mfrac></mstyle> <munderover><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>m</mi></munderover> <mfenced separators="" open="(" close=")"><mi>σ</mi> <mrow><mo>(</mo> <msup><mi mathvariant="bold">θ</mi> <mo>⊺</mo></msup> <msup><mi mathvariant="bold">x</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup> <mo>)</mo></mrow> <mo>-</mo> <msup><mi>y</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup></mfenced> <msubsup><mi>x</mi> <mi>j</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup></mrow>

这个公式看起来非常像公式 4-6：对于每个实例，它计算预测误差并将其乘以第 *j* 个特征值，然后计算所有训练实例的平均值。一旦你有了包含所有偏导数的梯度向量，你就可以在批量梯度下降算法中使用它。就是这样：你现在知道如何训练一个逻辑回归模型。对于随机梯度下降，你会一次取一个实例，而对于小批量梯度下降，你会一次使用一个小批量。

## 决策边界

我们可以使用鸢尾花数据集来说明逻辑回归。这是一个著名的包含三种不同物种（*Iris setosa*，*Iris versicolor*，和 *Iris virginica*）的 150 朵鸢尾花萼片和花瓣长度和宽度的数据集（见图 4-23）。

![三种鸢尾花物种的照片：_Iris virginica_，_Iris versicolor_，和 _Iris setosa_，展示了花瓣和萼片的特征。](img/hmls_0423.png)

###### 图 4-23\. 三种鸢尾花植物的鲜花⁠^(14)

让我们尝试构建一个分类器，仅基于花瓣宽度特征来检测 *Iris virginica* 类型。第一步是加载数据并快速查看：

```py
>>> from sklearn.datasets import load_iris
>>> iris = load_iris(as_frame=True)
>>> list(iris)
['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names',
 'filename', 'data_module']
>>> iris.data.head(3)
 sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                5.1               3.5                1.4               0.2
1                4.9               3.0                1.4               0.2
2                4.7               3.2                1.3               0.2
>>> iris.target.head(3)  # note that the instances are not shuffled
0    0
1    0
2    0
Name: target, dtype: int64
>>> iris.target_names
array(['setosa', 'versicolor', 'virginica'], dtype='<U10')
```

接下来我们将分割数据，并在训练集上训练一个逻辑回归模型：

```py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = iris.data[["petal width (cm)"]].values
y = iris.target_names[iris.target] == 'virginica'
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
```

让我们看看模型对花瓣宽度从 0 厘米到 3 厘米的鲜花估计的概率(图 4-24)⁠^(15)
