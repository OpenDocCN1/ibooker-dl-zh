
```py
def objective(trial):
  hparams = {
    "seed": trial.suggest_int('seed', 0, pt.iinfo(pt.int32).max - 1),

    "num_features": "8",

    "batch_norm_linear_layers": \
      str(trial.suggest_int('batch_norm_linear_layers', 0, 1)),

    "optimizer": trial.suggest_categorical('optimizer', ['Adam', 'SGD']),

    "lr": trial.suggest_loguniform('lr', 0.009, 0.07),

    "num_hidden_neurons": \
      str([trial.suggest_categorical(f"num_hidden_layer_{layer}_neurons",
            [7, 11]) for layer in \
              range(trial.suggest_categorical('num_layers', [2, 3]))]),

    "batch_size": trial.suggest_int('batch_size', 30, 50, log = True),

    "max_batches": trial.suggest_int('max_batches', 30, 50, log = True)
  }
  model, trainer = build(DcTaxiModel(**hparams),
    train_glob = 'https://raw.githubusercontent.com/osipov/smlbook/
➥                   master/train.csv',
    val_glob = 'https://raw.githubusercontent.com/osipov/smlbook/
➥                   master/valid.csv')

  return trainer.callback_metrics['train_val_rmse'].item()

import optuna
from optuna.samplers import TPESampler                       ❶
study = \
  optuna.create_study(direction = 'minimize',                ❷
                      sampler = TPESampler( seed = 42 ),)

study.optimize(objective, n_trials = 100)                    ❸
```

❶ 配置研究以最小化 `DcTaxiModel` 的 MSE 损失。

❷ 使用 TPE 算法进行 HPO。

❸ 使用 100 次试验开始 HPO。

在执行列表 11.5 中的代码后，`study.optimize` 方法完成了 100 次 HPO 试验。使用以下方式可获得各个试验的详细信息：

```py
study_df = study.trials_dataframe().sort_values(by='value',
                                                ascending = True)
study_df[:5][['number', 'value', 'params_seed']],
```

应返回一个类似以下值的 pandas 数据帧：

| number | value | params_seed |
| --- | --- | --- |
| 96 | 2.390541 | 1372300804 |
| 56 | 7.403345 | 1017301131 |
| 71 | 9.006614 | 939699871 |
| 74 | 9.139935 | 973536326 |
| 94 | 9.817746 | 1075268021 |

其中，数字列指定由 Optuna 建议的试验的索引，值是由目标方法中的 `trainer.callback_metrics['train_val_rmse'].item()` 返回的损失函数的相应值，`params_seed` 是用于初始化 `DcTaxiModel` 的模型参数（权重）的种子值。

## 11.3.2 在 Optuna 中可视化 HPO 研究

本节介绍了本章中使用三种不同 Optuna 可视化执行的 HPO 研究，并比较了这些可视化图在 HPO 方面的相关性。

完成的研究实例也可以使用 Optuna 可视化包进行可视化。虽然全面概述 Optuna 中各种可视化的范围超出了本书的范围，但我发现自己在一系列机器学习模型中一直重复使用三个可视化图。这些可视化图将在本节的其余部分按重要性下降的顺序进行解释。

超参数重要性图揭示了关于超参数对目标函数相对影响的惊人信息。在列表中包含种子超参数特别有用，以评估某些超参数是否比仅用于模型初始化的随机变量具有更多或更少的重要性。比随机种子更重要的超参数值得进一步研究，而比随机变量不重要的超参数应该降低优先级。

要创建重要性图，您可以使用：

```py
optuna.visualization.plot_param_importances(study)
```

这应该会生成类似于图 11.1 的条形图。

![11-01](img/11-01.png)

图 11.1 重要性图有助于指导后续 HPO 迭代。

一旦您确定了要更详细探讨的一组超参数，下一步就是在平行坐标图上绘制它们。您可以使用以下命令实例化此图：

```py
optuna.visualization.plot_parallel_coordinate(study,
    params=["lr", "batch_size", "num_hidden_layer_0_neurons"])
```

这会绘制学习率（`lr`）、批量大小和 `num_hidden_layer_0_neurons` 超参数之间的关系。请注意，在图 11.2 中，线条代表各个试验配置，颜色较深的线条对应于目标函数值较低的试验。因此，通过某个超参数的一系列较深线条穿过一定区间，表明该超参数区间值得更仔细检查，可能需要进行另一次 HPO 迭代。

![11-02](img/11-02.png)

图 11.2 平行坐标图有助于确定超参数值的影响区间。

到目前为止描述的图中，轮廓图在生成有关研究结果的见解方面排在最后。由于轮廓图仅限于可视化超参数值对的图形，您会发现自己生成多个轮廓图，通常基于从重要性图或平行坐标图中选择的超参数。例如，要绘制批量大小、学习率和目标函数之间的关系，您可以运行：

```py
optuna.visualization.plot_contour(study, params=["batch_size", "lr"])
```

这应该生成类似于图 11.3 的图形。

![11-03](img/11-03.png)

图 11.3 轮廓图有助于分析与损失函数相关的超参数对。

## 摘要

+ Optuna 是一个超参数优化框架，具有与本机 Python 集成，可支持机器学习模型实验中的复杂超参数配置。

+ 当使用 HPO 用于跨量级的超参数范围时，采用对数均匀采样是有用的，以确保样本在范围内均匀分布而不是倾斜分布。

+ 在执行 HPO 试验之后，Optuna 的可视化功能有助于分析试验结果和相关的超参数。

^(1.)无梯度算法不需要计算损失（或任何目标）函数的梯度来优化函数参数。换句话说，无梯度的超参数优化可以在目标函数没有可计算梯度的情况下进行优化。

^(2.)最初广泛引用的论文《Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift》介绍了批归一化，并可从[`arxiv.org/abs/1502.03167`](https://arxiv.org/abs/1502.03167)获取。
