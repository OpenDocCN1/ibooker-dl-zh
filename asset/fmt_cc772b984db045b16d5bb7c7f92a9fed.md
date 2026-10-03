
```
lw = 1.5                                                      ①
f, ax = plt.subplots(1, 2, figsize=(15, 5))                   ②
ax[0].plot(fpr_male[1], tpr_male[1],                          ③
        linestyle='-', color='b',                             ③
        lw=lw,                                                ③
        label='Male (Area = %0.2f)' % roc_auc_male[1])        ③
ax[0].plot(fpr_female[1], tpr_female[1],                      ④
        linestyle='--', color='g',                            ④
        lw=lw,                                                ④
        label='Female (Area = %0.2f)' % roc_auc_female[1])    ④
ax[1].plot(fpr_white[1], tpr_white[1],                        ⑤
        linestyle='-', color='c',                             ⑤
        lw=lw,                                                ⑤
        label='White (Area = %0.2f)' % roc_auc_white[1])      ⑤
ax[1].plot(fpr_black[1], tpr_black[1],                        ⑥
        linestyle='--', color='r',                            ⑥
        lw=lw,                                                ⑥
        label='Black (Area = %0.2f)' % roc_auc_black[1])      ⑥
ax[0].legend()                                                ⑦
ax[1].legend()                                                ⑦
ax[0].set_ylabel('True Positive Rate')                        ⑦
ax[0].set_xlabel('False Positive Rate')                       ⑦
ax[1].set_ylabel('True Positive Rate')                        ⑦
ax[1].set_xlabel('False Positive Rate')                       ⑦
ax[0].set_title('ROC Curve (Gender)')                         ⑦
ax[1].set_title('ROC Curve (Race)')                           ⑦
```

① 设置线图的线宽

② 初始化一个由一排两列组成的 Matplotlib 图表

③ 在第一列中，绘制了男性成人的 ROC 曲线

④ 在第一列中，绘制了女性成人的 ROC 曲线

⑤ 在第二列中，绘制了白人成人的 ROC 曲线

⑥ 在第二列中，绘制了黑人成人的 ROC 曲线

⑦ 注释并标记了图表

结果图表显示在图 8.14 中。图表的第一列比较了两个性别组（男性和女性）的 ROC 曲线。第二列比较了两个种族组（白人和黑人）的 ROC 曲线。两个图表的曲线下面积在图例中显示。我们可以看到，男性成人的 AUC 为 0.89，女性成人的 AUC 为 0.92。差异大约有 3%偏向女性成人。另一方面，白人成人的 AUC 为 0.9，黑人成人的 AUC 为 0.92。差异大约有 2%偏向黑人成人。不幸的是，与人口平衡不同，法律或研究界没有关于考虑模型公平性时应使用哪些阈值的指南。在本章中，如果差异在统计上显著，我们将使用机会平等和赔率的概念将模型视为不公平。我们将在 8.3.1 节中通过置信区间查看这些差异是否显著。

![](img/CH08_F14_Thampi.png)

图 8.14 性别和种族的接收器操作特征（ROC）曲线

### 8.2.3 其他公平性概念

最常用的公平性概念是人口平衡和机会平等/赔率。但是，为了提醒，让我们也看看以下其他公平性概念：

+   *预测质量平等*—不同组之间的预测质量没有差异。预测质量可以是模型的准确度或任何其他性能指标，如 F1。

+   *治疗平等*—模型对各组平等对待，其中假阳性率存在平衡。假阳性率被量化为假阴性数与假阳性数的比率。

+   *通过无意识实现的公平性*—可以通过不明确使用受保护属性作为预测特征来实现公平性。在一个理想的世界里，模型使用的其他特征与受保护属性不相关，但这几乎总是不成立。因此，通过无意识实现公平性不能保证。我们将在 8.4.1 节中看到这一点。

+   *反事实公平性*—如果一个模型在反事实世界中，如果该个体属于另一个受保护群体，它仍然做出相同的预测，那么该模型对该个体是公平的。

我们可以将所有公平性概念分为两类——*组公平性*和*个体公平性*。组公平性确保模型对不同受保护群体是公平的。对于成人收入数据集，受保护群体包括性别、种族和年龄。另一方面，个体公平性确保模型对相似个体做出相似的预测。对于成人收入数据集，个体可以根据其教育水平、原籍国或每周工作时间等因素相似，仅举几个例子。表 8.3 显示了不同公平性概念的所属类别。

表 8.3 组合和个体公平性概念

| 公平性概念 | 描述 | 类别 |
| --- | --- | --- |
| 人口统计平等 | 不同受保护群体在正预测率上的平等 | 组 |
| 机会和赔率的平等 | 不同受保护群体在真阳性率和假阳性率上的平等 | 组 |
| 预测质量平等 | 不同受保护群体在预测质量上的平等 | 组 |
| 处理平等 | 不同受保护群体在错误预测率上的平等 | 组 |
| 通过无意识实现公平性 | 通过不明确将受保护属性作为预测特征来实现公平性 | 个体 |
| 反事实公平性 | 在反事实世界中，如果该个体属于另一个受保护群体，则对个体的预测相似 | 个体 |

## 8.3 可解释性和公平性

在本节中，我们将学习如何使用可解释性技术来检测模型导致的歧视来源。歧视的来源可以大致分为以下两组：

+   *通过输入特征进行歧视*——可以追溯到输入特征的公平性问题。

+   *通过表示进行歧视*——难以追溯到输入特征的公平性问题，尤其是对于处理图像和文本等输入的深度学习模型。在这种情况下，我们可以将歧视的来源追溯到模型学习到的深度表示。

### 8.3.1 通过输入特征进行歧视

让我们首先通过输入特征来观察歧视。当我们查看第 8.2 节中的各种公平概念时，我们通过处理模型输出看到，随机森林模型在人口统计学差异和机会平等/概率公平的公平性度量上是不公平的。我们如何通过追溯到输入来解释这些公平性度量？我们可以利用 SHAP 来完成这个目的。正如我们在第四章和第 8.1.2 节中看到的，SHAP 将模型输出分解为每个输入的 Shapley 值。这些 Shapley 值与模型输出具有相同的单位——如果我们对所有特征求和 Shapley 值，我们将得到一个与模型输出相匹配的值，该值衡量预测正结果的概率。我们在第 8.1.2 节中看到了这个说明。因为输入特征的 Shapley 值之和等于模型输出，所以我们可以将保护群体之间模型输出（以及，相应地，公平性度量）的差异归因于每个输入的 Shapley 值的差异。这就是如何将任何歧视或公平性问题追溯到输入的方法。

让我们通过代码来观察这一过程。以下代码片段定义了一个辅助函数，用于生成保护群体之间的 SHAP 差异，并可用于生成模型输出中由输入引起的差异的可视化：

```
def generate_shap_group_diff(df_X,                                  ①
                             y,                                     ②
                             shap_values,                           ③
                             notion='demographic_parity',           ④
                             protected_group='gender',              ⑤
                             trace_to_input=False):                 ⑥
    if notion not in ['demographic_parity', 
    ➥ 'equality_of_opportunity']:                                  ⑦
        return None                                                 ⑦
    if protected_group not in ['gender', 'race']:                   ⑦
        return None                                                 ⑦
    if notion == 'demographic_parity':                              ⑧
        flabel = 'Demographic parity difference'                    ⑧
    if notion == 'equality_of_opportunity':                         ⑨
        flabel = 'Equality of opportunity difference'               ⑨
        positive_label_indices = np.where(y == 1)[0]                ⑨
        df_X = df_X.iloc[np.where(y == 1)[0]]                       ⑨
        shap_values = shap_values[np.where(y == 1)[0],:]            ⑨
    if protected_group == 'gender':                                 ⑩
        pg_label = 'men v/s women'                                  ⑩
        mask = df_X['gender'].values == 1                           ⑩
    if protected_group == 'race':                                   ⑪
        pg_label = 'white v/s black'                                ⑪
        mask = df_X['race'].values == 1                             ⑪
    glabel = f"{flabel}\nof model output for {pg_label}"            ⑫
    xmin = -0.8                                                     ⑬
    xmax = 0.8                                                      ⑬
    if trace_to_input:                                              ⑭
        shap.group_difference_plot(shap_values,                     ⑭
                                   mask,                            ⑭
                                   df_X.columns,                    ⑭
                                   xmin=xmin,                       ⑭
                                   xmax=xmax,                       ⑭
                                   xlabel=glabel,                   ⑭
                                   show=False)                      ⑭
    else:                                                           ⑮
        shap.group_difference_plot(shap_values.sum(1),              ⑮
                                   mask,                            ⑮
                                   xmin=xmin,                       ⑮
                                   xmax=xmax,                       ⑮
                                   xlabel=glabel,                   ⑮
                                   show=False)                      ⑮
```

① 生成 SHAP 群体差异图的辅助函数，该函数接受六个输入。输入 1：特征值的 DataFrame

② 输入 2：目标值的向量

③ 输入 3：为输入特征生成的 SHAP 值

④ 输入 4：公平概念可以是人口统计学差异或机会平等/概率公平

⑤ 输入 5：可以是性别或种族的保护群体

⑥ 输入 6：标志，指示是否将歧视的来源追溯到输入

⑦ 对于不支持公平概念和保护群体，返回 None

⑧ 设置人口统计学差异概念的标签

⑨ 设置标签并仅处理机会平等概念的实际情况

⑩ 设置性别保护群体的标签和掩码

⑪ 设置种族保护群体的标签和掩码

⑫ 设置可视化标签

⑬ 限制可视化到 xmin 和 xmax

⑭ 当 trace_to_input 设置为 True 时创建可视化

⑮ 当 trace_to_input 设置为 False 时创建可视化

我们首先将使用辅助函数来检查模型输出中性别保护群体的人口统计学差异，如下代码示例所示。请注意，我们只关注测试集中的模型预测。`shap_values`变量包含数据集中所有输入和成人的 Shapley 值。我们是在第 8.1.2 节中生成的，你可以在与本书相关的 GitHub 仓库中找到源代码：

```
test_indices = X_test.index.values                         ①
generate_shap_group_diff(X_test,                           ②
                         y_test,                           ②
                         shap_values[1][test_indices,:],   ②
                         notion='demographic_parity',      ②
                         protected_group='gender',         ②
                         trace_to_input=False)             ②
```

① 提取测试集中输入的索引

② 调用辅助函数以适当的输入生成 SHAP 图

结果可视化如图 8.15 所示。请注意，差异可以是正的也可以是负的。如果差异为正，则模型偏向于男性成年人，如果差异为负，则模型偏向于女性成年人。从图 8.15 中我们可以看到，随机森林模型在预测正例（即薪水>50K）时对男性成年人的预测更多，因此偏向于男性成年人。

![](img/CH08_F15_Thampi.png)

图 8.15 模型输出中性别的人口统计学差异

为了确定导致男性和女性成年人之间人口统计学差异的原因，我们可以使用以下代码片段将其追溯到输入特征：
