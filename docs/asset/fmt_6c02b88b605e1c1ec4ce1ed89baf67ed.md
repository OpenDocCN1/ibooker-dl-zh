
`selectElementRoute = document.querySelector('#route');`
`selectElementDirection = document.querySelector('#direction');`
`selectElementYear = document.querySelector('#year');`
`selectElementMonth = document.querySelector('#month');`
`selectElementDaym = document.querySelector('#daym');`
`selectElementDay = document.querySelector('#day');`
`selectElementHour = document.querySelector('#hour');`
`route_string = \`
`selectElementRoute.options\`
`[selectElementRoute.selectedIndex].value`
`direction_string = \`
`selectElementDirection.options\`
`[selectElementDirection.selectedIndex].value`
`year_string = \`
`selectElementYear.options\`
`[selectElementYear.selectedIndex].value`
`month_string = \`
`selectElementMonth.options\`
`[selectElementMonth.selectedIndex].value`
`daym_string = \`
`selectElementDaym.options\`
`[selectElementDaym.selectedIndex].value`
`day_string = \`
`selectElementDay.options\`
`[selectElementDay.selectedIndex].value`
`hour_string = \`
`selectElementHour.options\`
`[selectElementHour.selectedIndex].value`

更新 `home.html` 中的 `getOption()` JavaScript 函数，以构建包含模型评分参数的目标 URL。以下代码片段展示了 `getOption()` 中定义街车延误预测部署目标 URL 的语句：

```
window.output = \
prefix.concat("route=",route_string,"&direction=",\
direction_string,"&year=",year_string,"&month=",\
month_string,"&daym=",daym_string,"&day=",\
day_string,"&hour=",hour_string)
```

更新 `flask_server.py` 中 `show_prediction.html` 的视图函数，以构建你希望在 `show_prediction.html` 中显示的字符串，用于每个预测结果：

```
if pred[0][0] >= 0.5:
    predict_string = "yes, delay predicted"
else:
    predict_string = "no delay predicted"
```

通过这些更改，你应该能够使用第八章中描述的 Web 部署来简单地部署你的新模型。

## 9.11 使用 Facebook Messenger 部署模型

如果你选择使用第八章中描述的 Facebook Messenger 部署方法来解决你的新问题，你将需要更新 `actions.py` 中的评分代码，包括将 `score_cols` 设置为训练模型所使用的列名。你还需要更新设置默认值的代码，以防用户在评分时没有提供这些值。通过这些更改，你将拥有准备就绪的 Python 代码，用于使用训练好的模型评分新的数据点。

使用 Rasa 与 Facebook Messenger 进行部署时，Python 代码并不是全部。你还需要在 Facebook Messenger 中有一个自然的界面，为此，你需要创建一个简单的 Rasa 模型。你可以从街车延误部署的示例中了解如何指定 `nlu.md` 文件中的句子级 Rasa 训练示例和在 `stories.md` 文件中的多句子 Rasa 示例。定义正确的 slot 集合更具挑战性。我建议你为 `score_cols` 中的每个列名创建一个 slot。你可以在 `domain.yml` 文件中定义这些 slot。为了简化，你可以将每个 slot 的类型设置为 `text`。最好避免不必要的 slot，所以如果你是从街车延误示例中复制 `domain.yml` 文件作为起点，请在定义新的 slot 值之前清除现有的 slot 值。

要完成新模型部署的其余部分，你将遵循第八章中部署街车延误预测模型时遵循的以下步骤子集：

1. 创建一个名为 `new_deploy` 的目录。

2. 在 `deploy` 目录中运行以下命令以设置基本的 Rasa 环境：

```
rasa init
```

3. 分别将你的训练模型的 `h5` 文件和 pipelines 的 `pkl` 文件复制到 `models` 和 `pipelines` 目录。

4. 将 `new_deploy` 目录中的 `actions.py` 文件替换为你为新的部署更新的 `actions.py` 文件。

5. 将 `data` 子目录中的 `nlu.md` 和 `stories.md` 文件替换为你为新的部署创建的 `nlu.md` 和 `stories.md` 文件。

6. 将 `new_deploy` 目录中的 `domain.yml` 文件替换为你的新部署的 `domain.yml` 文件。

7. 将 repo 中的 `custom_classes.py` 和 `endpoints.yml` 文件复制到新的 `deploy` 目录。

8. 将 `deploy_config.yml` 配置文件复制到新的 `deploy` 目录，并更新 pipeline 和模型文件名参数以匹配步骤 3 中复制的文件，如下所示。

列表 9.3 需要在部署配置文件中更新的参数

```
general:
   debug_on: False
   logging_level: "WARNING" # switch to control logging - WARNING for full
   ➥ logging; ERROR to minimize logging
   BATCH_SIZE: 1000
file_names:
   pipeline1_filename: <your pipeline1 pkl file>     ❶ 
   pipeline2_filename: <your pipeline2 pkl file>
   model_filename: <your trained model file>         ❷ 
```

❶ 替换为步骤 3 中复制到 `pipelines` 目录的 pipeline 文件名。

❷ 替换为步骤 3 中复制到 `models` 目录的训练模型 `h5` 文件名。

9. 在你的 `new_deploy` 目录中运行以下命令以在 Rasa 的 Python 环境中调用 `actions.py`：

```
rasa run actions
```

10. 在你安装 `ngrok` 的目录中，调用 `ngrok` 以使你的 `localhost` 在端口 `5005` 上可供 Facebook Messenger 使用。以下是 Windows 的命令；注意 `ngrok` 输出中的 HTTPS 转发 URL：

```
.\ngrok http 5005
```

11. 在 `new_deploy` 目录中运行以下命令以训练 Rasa 模型：

```
rasa train
```

12. 按照以下链接中的说明 `http://mng.bz/oRRN` 添加一个新的 Facebook 页面。您可以使用在第八章中创建的相同 Facebook 应用进行这次新部署。注意记录页面访问令牌和应用程序密钥；您需要在第 13 步更新 `credentials.yml` 文件，并使用这些值。

13. 更新 `new_deploy` 目录中的 `credentials.yml` 文件，以设置验证令牌（您选择的字符串值）、密钥和页面访问令牌（在第 12 步的 Facebook 设置期间提供）：

```
facebook:  
  verify: <verify token that you choose>
  secret: <app secret from Facebook app setup>
  page-access-token: <page access token from Facebook app setup>
```

14. 在 `new_deploy` 目录中运行以下命令以启动 Rasa 服务器，使用第 13 步在 `credentials.yml` 中设置的凭据：

```
rasa run —credentials credentials.yml
```

15. 在第八章中创建的 Facebook 应用中，选择“消息传递”->“设置”，滚动到 Webhooks 部分，并点击“编辑回调 URL”。将回调 URL 值的前一部分替换为在第 10 步调用 `ngrok` 时记录的 HTTPS 转发 URL。在第 13 步中设置的 verify token（您选择的字符串值）输入到“验证令牌”字段中，然后点击“验证并保存”。

16. 在 Facebook 消息传递（移动或 Web 应用）中，搜索第 12 步创建的 Facebook 页面的 ID，并输入查询以确认您的部署模型是可访问的。

我们已经审查了将用于电车延误问题的方法适应新结构数据集所需的步骤。现在，当您想要将深度学习应用于结构化数据时，您已经有了应用本书中代码示例所需的新问题。

## 9.12 将本书中的方法适应不同的数据集

为了使将本书中的方法应用于新的问题领域更容易，我们将通过将用于电车延误预测问题的代码适应新数据集的过程。我们不会处理整个端到端问题——只从初始数据集到最小训练的深度学习模型。

我们希望找到一个足够大的数据集，以便给深度学习一个公平的机会（记录数至少在数千条），但又不能太大，以至于数据量成为练习的主要焦点。我检查了流行的机器学习竞赛网站 Kaggle，以找到一个适合的表格结构化数据问题。我发现预测纽约市 Airbnb 物业价格的 [`www.kaggle.com/dgomonov/new-york-city-airbnb-open-data`](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data) 问题有一个有趣的数据集，可能适合适应本书中描述的方法。图 9.18 显示了该数据集的数据片段。

![CH09_F18_Ryan](img/CH09_F18_Ryan.png)

图 9.18 Airbnb 纽约市数据集的示例记录

该数据集包含略少于 89,000 条记录，分布在 16 列中，因此数据集的大小和复杂性是合适的。让我们检查每一列的内容：

+ `id` — 列表的数字标识符

+ `name` — 列表的描述

+ `host_id` — 与列表关联的主人的数字标识符

+ `host_name` — 与列表关联的主人的名字

+ `neighbourhood_group` — 列表所在的纽约市行政区：曼哈顿、布鲁克林、皇后区、布朗克斯或斯塔滕岛

+ `neighbourhood` — 列表所在的社区

+ `latitude` — 列表的纬度

+ `longitude` — 列表的经度

+ `room_type` — 列表的房间类型：整个住宅、私人房间或共享房间

+ `price` — 列表的价格（深度学习模型的目标）

+ `minimum_nights` — 列表可以预订的最少夜数

+ `number_of_reviews` — Airbnb 网站上可用的列表评论数量

+ `last_review` — 列表最近一次评论的日期

+ `reviews_per_month` — 列表每月的平均评论数量

+ `calculated_host_listings_count` — 与该列表关联的主人的列表数量

+ `availability_365` — 该列表一年中可供出租的比例

我们得到了这些列类型的一个有趣的混合：

+ *连续* — `price`、`minimum_nights`、`number_of_reviews`、`reviews_per_month`、`calculated_host_listings_count`、`availability_365`

+ *分类* — `neighbourhood_group`、`neighbourhood`、`room_type`、`host_id`

+ *文本* — `name` 和可能 `host_name`

除了这些容易分类的列之外，还有 `id`（从模型训练的角度来看并不有趣）、`longitude` 和 `latitude`（在本练习中，我们将依赖 `neighborhood` 来确定位置），以及 `last_review`（在本练习中，我们不会使用此列）。

与电车延误数据集相比，Airbnb 纽约市数据集的混乱程度要低得多。首先，让我们看看每个列中缺失值和唯一值的数量（图 9.19）。

与电车延误数据集相比，Airbnb 纽约市数据集具有较少的缺失值列。此外，所有分类列（`neighbourhood_group`、`neighbourhood` 和 `host_name`）似乎都有合理的值数量。相比之下，电车延误数据集中的方向、位置和路线列都包含了一些无关的值。例如，原始电车延误数据集中的方向列有 15 个不同的值，但只有 5 个有效值。

Airbnb 纽约数据集中相对较少的杂乱突显了从 Kaggle 可用的数据集的问题之一。尽管这些数据集对于练习机器学习的各个方面很有用，尽管参加 Kaggle 的比赛是学习的好方法，但用于比赛的精心策划和清洗的数据集并不能替代真实世界的数据集。正如您在第三章和第四章中看到的，真实世界的数据集在意外的方式下很杂乱，并且需要非平凡的工作来准备训练深度学习模型。Airbnb 数据集所需的数据准备量有限（将 CSV 文件导入 Pandas dataframe 并用默认值替换缺失值）包含在 `airbnb_data_preparation` 笔记本中。

![CH09_F19_Ryan](img/CH09_F19_Ryan.png)

图 9.19 Airbnb 纽约数据集的列特征

下一列表显示了与此示例相关的存储库中的文件。

列表 9.4 与 Airbnb 定价预测示例相关的代码存储库中的代码

```
├── data 
│           AB_NYC_2019_remove_bad_values_jun21_2020.pkl     ❶ 
│ 
├── notebooks
│           airbnb_data_preparation.ipynb                    ❷ 
│           airbnb_data_preparation_config.yml               ❸ 
│           airbnb_model_training.ipynb                      ❹ 
│           airbnb_model_training_config.yml                 ❺ 
```

❶ `airbnb_data_preparation` 笔记本保存的清洗后的数据集输出

❷ 数据准备笔记本

❸ 数据准备配置文件

❹ 模型训练笔记本

❺ 模型训练配置文件

`airbnb_data_preparation` 笔记本保存了清洗后的 dataframe 的 pickle 版本，该版本可以用作输入到深度学习模型训练笔记本 `airbnb_model_training`。这个笔记本是街车延误预测问题模型训练笔记本的简化版本。这个简单模型的目标是 Airbnb 物业的价格是否会低于（`0`）或高于（`1`）平均价格。与本书主要示例中使用的版本相比，这个笔记本的关键变化包括以下内容：

+ 列表列的成员资格（分类、连续和文本列）在配置文件 `airbnb_model_training_config.yml` 中设置，而不是在笔记本本身中（见下一列表）。

+ 列表 9.5 Airbnb 价格预测模型列类别的参数
