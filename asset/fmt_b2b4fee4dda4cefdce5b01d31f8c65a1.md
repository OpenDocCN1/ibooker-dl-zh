
# 多保真度优化与高斯过程扩展实践

## A.8 第十一章：多保真度优化

在本练习中，我们将多保真度优化技术应用于一个基准测试问题。我们首先设置必要的张量：

```py
fidelities = torch.tensor([0.1, 0.3, 1.0])
bounds_full = torch.cat(
    [
        bounds,
        torch.tensor([fidelities.min(), fidelities.max()]).unsqueeze(-1)
    ],
    dim=1
)
```

1. 将线性成本模型的固定成本设置为 0.2，权重设置为 1：

```py
from botorch.models.cost import AffineFidelityCostModel

cost_model = AffineFidelityCostModel(fixed_cost=0.2)
```

将每次实验中我们的预算限制设置为 10，并且重复实验的次数也设置为 10：

```py
budget_limit = 10
num_repeats = 10
```

2. 从 Sobol 序列中绘制的候选者数量设置为 5,000，并在使用辅助函数优化给定策略的收购分数时，使用 100 次重启和 500 次原始样本：

```py
num_samples = 5000

num_restarts = 100
raw_samples = 500
```

3. 重新定义辅助函数 `get_final_recommendation`，以找到后验平均值最大化器：

```py
from botorch.acquisition.fixed_feature import 
➥FixedFeatureAcquisitionFunction
from botorch.acquisition import PosteriorMean
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed

def get_final_recommendation(model):
    post_mean_policy = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=3,                ❶
        columns=[2],        ❶
        values=[1],
    )

    final_x, _ = optimize_acqf(
        post_mean_policy,
        bounds=bounds,
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )

    return torch.cat([final_x, torch.ones(1, 1)], dim=1)
```

❶ 必要的更改

4. 在我们的优化问题上运行多保真度 MES 策略及其单保真度版本，并使用练习 1 中描述的方法绘制每个策略的平均优化进展和误差条。图 A.19 显示了两种策略之间的比较。

![](img/A-19.png)

图 A.19 显示了单一和多层次 MES 策略在 Branin 函数上的平均优化进展。多层次策略再次优于单一层次策略。

## A.9 第十一章：同时优化多个目标

在本练习中，我们将我们学到的多目标优化技术应用于优化飞机的气动结构设计问题。这个练习让我们能够观察多维问题中 Expected Hypervolume Improvement (EHVI) 策略的表现：

1. 我们按照以下方式复制了目标函数的代码：

```py
def objective1(X):                                                  ❶
  X_copy = X.detach().clone()                                       ❶
  X_copy[:, [2, 3]] = 1 - X_copy[:, [2, 3]]                         ❶
  X_copy = X_copy * 10 - 5                                          ❶
  return (                                                          ❶
    -0.005                                                          ❶
    * (X_copy ** 4 - 16 * X_copy ** 2 + 5 * X_copy)                 ❶
    ➥.sum(dim=-1)                                                  ❶
      + 3                                                           ❶
  )                                                                 ❶

def objective2(X):                                                  ❷
  X = X * 20 - 10                                                   ❷
  part1 = (X[..., 0] - 1) ** 2                                      ❷
  i = X.new(range(2, 5))                                            ❷
  part2 = torch.sum(i * (2.0 * X[..., 1:] ** 2 - X[..., :-1]) ** 2, ❷
  ➥ dim=-1)                                                        ❷
  return (part1 + part2) / 100_000 - 2                              ❷
```

❶ 第一个目标函数

❷ 第二个目标函数，来自第八章练习 2 中的代码取反

2. 我们按照以下方式实现辅助函数：

```py
def joint_objective(X):
    return torch.vstack(
        [
            objective1(X).flatten(),
            objective2(X).flatten(),
        ]
    ).transpose(-1, -2)
```

3. 我们声明搜索空间的边界：

```py
bounds = torch.tensor([[0.0] * 4, [1.0] * 4])
```

4. 我们声明参考点：

```py
ref_point = torch.tensor([-1.5, -2.0])
```

5. 类实现和辅助函数可以使用与 `CH08/03 - 练习 2.ipynb` 中相同的代码实现。

6. 我们设置实验设置：

```py
num_queries = 50
num_repeats = 10
```

7. 我们按照 `CH11/02 - 多目标 BayesOpt loop.ipynb` 中相同的方式实现了两种 BayesOpt 策略。图 A.20 展示了两种策略在 10 个实验中的综合表现。EHVI 策略再次优于交替 EI 策略。

![](img/A-20.png)

图 A.20 显示两个贝叶斯优化策略根据查询次数的平均超体积和误差棒。EHVI 策略始终优于交替 EI 策略。

## A.10 第十二章：将高斯过程扩展到大数据集

本练习展示了在真实数据集加利福尼亚州房价上从普通 GP 模型转换为 VGP 模型时的效率提升。我们的目标是观察 VGP 在真实世界环境中的计算优势。

完成以下步骤：

1. 我们使用 Pandas 库读入数据集：

```py
import pandas as pd

df = pd.read_csv("../data/housing.csv")
```

将 Pandas dataframe 读入后，应该与图 A.21 中的输出类似。

![](img/A-21.png)

图 A.21 显示了房价数据集作为 Pandas dataframe。这是本练习的训练集。

2. 我们创建散点图如下所示：

```py
plt.figure(figsize=(8, 6))
plt.scatter(df.longitude, df.latitude, c=np.log(df.median_house_value))
plt.colorbar();
```

可视化应该类似于图 A.22。

![](img/A-22.png)

图 A.22 房价数据集显示为散点图

3. 为了提取我们的训练特征，我们使用 `torch.from_numpy()` 方法将 NumPy 数组转换为 PyTorch 张量：

```py
train_x = torch.from_numpy(df.drop(["median_house_value"], axis=1).values)
```

4. 我们同样对房价的对数进行了这样的操作，这是我们的训练标签：

```py
train_y = torch.from_numpy(
    df.median_house_value.values
).log().to(train_x.dtype)
```

5. 我们将训练标签 `train_y` 标准化如下：

```py
train_y = (train_y - train_y.mean()) / train_y.std()
```

6. 我们如下实现 GP 模型：

```py
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()   ❶
        self.covar_module = gpytorch.kernels.ScaleKernel(  ❷
            gpytorch.kernels.MaternKernel(
                nu=2.5,                                    ❷
                ard_num_dims=train_x.shape[1]              ❷
            )                                              ❷
        )                                                  ❷

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
```

❶ 常数均值函数

❷ 具有输出比例的 ARD Matern 5/2 核函数

7. 使用以下代码制作一个噪声至少为 0.1 的似然函数：

```py
likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints
    ➥.GreaterThan(1e-1)                    ❶
)
```

❶ 约束强制噪声至少为 0.1。

8. 我们使用梯度下降训练先前实现的 GP 模型如下：

```py
model = GPModel(train_x, train_y, likelihood)

optimizer = torch.optim.Adam(model.parameters(),
➥lr=0.01)                                        ❶
mll = gpytorch.mlls.ExactMarginalLogLikelihood
➥(likelihood, model)                             ❷

model.train()                                     ❸
likelihood.train()                                ❸

for i in tqdm(range(10)):
    optimizer.zero_grad()

    output = model(train_x)
    loss = -mll(output, train_y)

    loss.backward()
    optimizer.step()
```

❶ 梯度下降优化器 Adam

❷ （负）边际对数似然损失函数

❸ 启用训练模式

在 MacBook 上，总训练时间为 24 秒。

9. 我们按以下步骤实现了 VGP 模型：

```py
class ApproximateGPModel(gpytorch.models.ApproximateGP):
  def __init__(self, inducing_points):
    variational_distribution =                    ❶
    ➥gpytorch.variational                        ❶
    ➥.NaturalVariationalDistribution(            ❶
        inducing_points.size(0)                   ❶
    )                                             ❶
    variational_strategy = gpytorch.variational   ❶
    ➥.VariationalStrategy(                       ❶
        self,                                     ❶
        inducing_points,                          ❶
        variational_distribution,                 ❶
        learn_inducing_locations=True,            ❶
    )                                             ❶
    super().__init__(variational_strategy)
    self.mean_module = gpytorch.means.ConstantMean()
    self.covar_module = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(
            nu=2.5,
            ard_num_dims=inducing_points.shape[1]
        )
    )

  def forward(self, x):
    ...                                           ❷
```

❶ 变分参数

❷ 与 GP 相同

10. 这个 VGP 是这样训练的：
