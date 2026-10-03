
```py
>>> cv_res = pd.DataFrame(grid_search.cv_results_)
>>> cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
>>> [...]  # change column names to fit on this page, and show rmse = -score
>>> cv_res.head()  # note: the 1st column is the row ID
 n_clusters  max_features  split0  split1  split2  mean_test_rmse
12          15             6   42725   43708   44335           43590
13          15             8   43486   43820   44900           44069
6           10             4   43798   44036   44961           44265
9           10             6   43710   44163   44967           44280
7           10             6   43710   44163   44967           44280
```

最佳模型的平均测试 RMSE 分数为 43,590，这比之前使用默认超参数值得到的分数（47,038）要好。恭喜你，你已经成功微调了你的最佳模型！

## 随机搜索

当你探索相对较少的组合时，例如在之前的例子中，网格搜索方法是可以接受的，但通常`RandomizedSearchCV`更可取，尤其是在超参数搜索空间较大时。这个类可以像`GridSearchCV`类一样使用，但它不是尝试所有可能的组合，而是在每次迭代中评估固定数量的组合，为每个超参数选择一个随机值。这听起来可能有些令人惊讶，但这种方法有几个优点：

+   如果你的某些超参数是连续的（或者离散但有许多可能的值），并且你让随机搜索运行，比如说，1,000 次迭代，那么它将为这些超参数中的每一个探索 1,000 个不同的值，而网格搜索只会探索你为每个超参数列出的少数几个值。

+   假设一个超参数实际上并没有太大的影响，但你还没有意识到这一点。如果它有 10 个可能的值，并且你将其添加到你的网格搜索中，那么训练将需要更长的时间。但如果将其添加到随机搜索中，它将不会产生任何影响。

+   如果有 6 个超参数需要探索，每个有 10 个可能的值，那么网格搜索除了训练模型一百万次外，没有其他选择，而随机搜索可以运行任何你选择的迭代次数。

对于每个超参数，你必须提供一个可能的值列表，或者一个概率分布：

```py
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {'preprocessing__geo__n_clusters': randint(low=3, high=50),
                  'random_forest__max_features': randint(low=2, high=20)}

rnd_search = RandomizedSearchCV(
    full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
    scoring='neg_root_mean_squared_error', random_state=42)

rnd_search.fit(housing, housing_labels)
```

Scikit-Learn 还提供了 `HalvingRandomSearchCV` 和 `HalvingGridSearchCV` 超参数搜索类。它们的目标是更有效地使用计算资源，要么是为了更快地训练，要么是为了探索更大的超参数空间。以下是它们的工作原理：在第一轮中，使用网格方法或随机方法生成许多超参数组合（称为“候选人”）。然后，使用这些候选人来训练模型，并像往常一样使用交叉验证来评估这些模型。然而，训练使用有限的资源，这大大加快了第一轮的速度。默认情况下，“有限资源”意味着模型是在训练集的小部分上训练的。然而，其他限制也是可能的，例如，如果模型有一个需要设置的超参数，可以减少训练迭代次数。一旦评估了所有候选人，只有最好的候选人才会进入第二轮，在那里它们可以获得更多的资源来竞争。经过几轮后，最终候选人将使用全部资源进行评估。这可能会节省您一些调整超参数的时间。

## 集成方法

另一种微调您系统的方法是尝试组合表现最好的模型。通常，组（或“集成”）的表现会比单个最佳模型更好——就像随机森林的表现比它们依赖的个别决策树更好一样——特别是如果个别模型犯下非常不同类型的错误。例如，您可以训练和微调一个 *k*-最近邻模型，然后创建一个集成模型，该模型仅预测随机森林预测和该模型预测的均值。我们将在第六章中更详细地介绍这个主题。

## 分析最佳模型及其错误

通过检查最佳模型，您通常可以从问题中获得很好的见解。例如，`RandomForestRegressor` 可以指示每个属性对准确预测的相对重要性：

```py
>>> final_model = rnd_search.best_estimator_  # includes preprocessing
>>> feature_importances = final_model["random_forest"].feature_importances_
>>> feature_importances.round(2)
array([0.07, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01, 0.19, [...], 0\.  , 0.01])
```

让我们将这些重要性分数按降序排列，并在它们对应的属性名称旁边显示：

```py
>>> sorted(zip(feature_importances,
...            final_model["preprocessing"].get_feature_names_out()),
...        reverse=True)
...
[(np.float64(0.18599734460509476), 'log__median_income'),
 (np.float64(0.07338850855844489), 'cat__ocean_proximity_INLAND'),
 (np.float64(0.06556941990883976), 'bedrooms__ratio'),
 (np.float64(0.053648710076725316), 'rooms_per_house__ratio'),
 (np.float64(0.04598870861894749), 'people_per_house__ratio'),
 (np.float64(0.04175269214442519), 'geo__Cluster 30 similarity'),
 (np.float64(0.025976797232869678), 'geo__Cluster 25 similarity'),
 (np.float64(0.023595895886342255), 'geo__Cluster 36 similarity'),
 [...]
 (np.float64(0.0004325970342247361), 'cat__ocean_proximity_NEAR BAY'),
 (np.float64(3.0190221102670295e-05), 'cat__ocean_proximity_ISLAND')]
```

根据这些信息，您可能想要尝试删除一些不太有用的特征（例如，显然只有一个 `ocean_proximity` 类别真正有用，因此您可以尝试删除其他类别）。

###### 小贴士

`sklearn.feature_selection.SelectFromModel` 转换器可以自动为您删除最不实用的特征：当您将其拟合时，它会训练一个模型（通常是随机森林），查看其 `feature_importances_` 属性，并选择最实用的特征。然后当您调用 `transform()` 时，它会删除其他特征。

您还应该查看您的系统犯的具体错误，然后尝试理解为什么它会犯这些错误以及什么可以修复问题：添加额外特征或去除无信息特征，清理异常值等。

现在也是检查*模型公平性*的好时机：它不仅应该在平均意义上表现良好，还应该在各种类型的地区上表现良好，无论是农村还是城市，富裕还是贫穷，北方还是南方，少数民族还是不是，等等。这需要详细的*偏差分析*：为每个类别创建你的验证集的子集，并分析你的模型在这些子集上的性能。这是一项大量工作，但非常重要：如果你的模型在某个地区的整个类别上表现不佳，那么在问题解决之前，它可能不应该部署，或者至少不应该用于该类别的预测，因为它可能会造成比好处更大的伤害。

## 在测试集上评估你的系统

在调整你的模型一段时间后，你最终拥有了一个表现足够好的系统。你现在可以准备在测试集上评估最终模型。这个过程没有什么特别之处；只需从你的测试集中获取预测器和标签，运行你的`final_model`来转换数据并做出预测，然后评估这些预测：

```py
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

final_predictions = final_model.predict(X_test)

final_rmse = root_mean_squared_error(y_test, final_predictions)
print(final_rmse)  # prints 41445.533268606625
```

在某些情况下，这样的泛化误差的点估计可能不足以让你决定发布：如果它只是比当前生产中的模型好 0.1%，你会怎么做？你可能想了解这个估计的精确度。为此，你可以使用`scipy.stats.bootstrap()`计算泛化误差的 95%*置信区间*。你得到一个相当大的区间，从 39,521 到 43,702，而你之前的点估计 41,445 大致位于这个区间的中间：

```py
from scipy import stats

def rmse(squared_errors):
    return np.sqrt(np.mean(squared_errors))

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
boot_result = stats.bootstrap([squared_errors], rmse,
                              confidence_level=confidence, random_state=42)
rmse_lower, rmse_upper = boot_result.confidence_interval
```

如果你做了很多超参数调整，性能通常会略低于你使用交叉验证测量的性能。这是因为你的系统最终会微调以在验证数据上表现良好，并且可能不会在未知数据集上表现得一样好。在这个例子中，测试 RMSE 低于验证 RMSE，但这并不是这种情况；当这种情况发生时，你必须抵制调整超参数以使测试集上的数字看起来更好的诱惑；这种改进不太可能推广到新数据。

现在是项目预发布阶段。有效地展示你的解决方案是区分优秀数据科学家和良好数据科学家的关键。你应该创建简洁的报告（Markdown，PDFs，幻灯片），可视化关键见解（例如，使用 Matplotlib 或其他工具如 SeaBorn 或 Tableau），并根据受众调整你的信息：对同行使用技术性语言，对利益相关者使用高级语言。提供有影响力和易于记忆的陈述（例如，“中位数收入是房价的第一大预测因素”）。强调你学到了什么，什么有效，什么无效，做了哪些假设，以及你系统的局限性是什么。

您的结果应该是可复制的（尽可能多）：使代码对您的团队可访问（例如，通过 GitHub），添加一个结构化的*README*文件来指导技术人员完成安装步骤。提供清晰的笔记本（例如，Jupyter）带有代码、解释和结果，编写干净、注释良好的代码。定义一个*requirements.txt*或*environment.yml*文件，包含所有必需的库及其精确版本（或创建一个 Docker 镜像）。为随机生成器设置种子，并移除任何其他变异来源。

在这个加利福尼亚住房示例中，系统的最终性能并不比专家的价格估计好多少，专家的估计通常误差高达 30%，但仍然可能是一个好主意来推出它，特别是如果这可以为专家节省一些时间，让他们可以从事更有趣和更有生产力的任务。

# 启动、监控和维护您的系统

完美，您获得了批准来启动！现在您需要为您的解决方案准备生产环境（例如，完善代码、编写文档和测试等）。然后您可以将模型部署到您的生产环境。最基本的方法是只是保存您训练的最佳模型，将文件转移到您的生产环境，并加载它。要保存模型，您可以使用`joblib`库，如下所示：

```py
import joblib

joblib.dump(final_model, "my_california_housing_model.pkl")
```

###### 小贴士

保存您实验过的每个模型通常是一个好主意，这样您可以轻松地回到任何您想要的模型。您还可以保存交叉验证分数，也许还有验证集上的实际预测。这将允许您轻松地比较不同模型类型的分数，并比较它们犯的错误类型。

一旦您的模型转移到生产环境，您就可以加载它并使用它。为此，您必须首先导入模型所依赖的任何自定义类和函数（这意味着将代码转移到生产环境），然后使用`joblib`加载模型并使用它进行预测：

```py
import joblib
[...]  # import KMeans, BaseEstimator, TransformerMixin, rbf_kernel, etc.

def column_ratio(X): [...]
def ratio_name(function_transformer, feature_names_in): [...]
class ClusterSimilarity(BaseEstimator, TransformerMixin): [...]

final_model_reloaded = joblib.load("my_california_housing_model.pkl")

new_data = [...]  # some new districts to make predictions for
predictions = final_model_reloaded.predict(new_data)
```

例如，模型可能被用于网站内部：用户将输入关于新区域的一些数据，然后点击“估算价格”按钮。这将发送包含数据的查询到网络服务器，服务器再将它转发到您的网络应用，最后您的代码将简单地调用模型的`predict()`方法（您希望在服务器启动时加载模型，而不是每次使用模型时都加载）。或者，您可以将模型封装在一个专用的网络服务中，您的网络应用可以通过 REST API 查询这个服务⁠^(14)（参见图 2-21）。这使得在不中断主应用的情况下升级模型到新版本变得更加容易。它还简化了扩展，因为您可以根据需要启动尽可能多的网络服务，并将来自您的网络应用的网络服务请求进行负载均衡。此外，它还允许您的网络应用使用任何编程语言，而不仅仅是 Python。

![展示用户与网页应用程序交互的图表，该应用程序将输入发送到托管模型的网络服务，并接收预测结果。](img/hmls_0221.png)

###### 图 2-21. 作为网络服务部署的模型及其被网页应用程序使用的场景

另一种流行的策略是将您的模型部署到云端，例如在 Google 的 Vertex AI（以前称为 Google Cloud AI Platform 和 Google Cloud ML Engine）上：只需使用`joblib`保存您的模型并将其上传到 Google Cloud Storage（GCS），然后转到 Vertex AI 并创建一个新的模型版本，将其指向 GCS 文件。就这样！这为您提供了一个简单的网络服务，它会为您处理负载均衡和扩展。它接受包含输入数据的 JSON 请求（例如，某个地区的输入数据），并返回包含预测结果的 JSON 响应。然后您可以在您的网站（或您正在使用的任何生产环境中）使用此网络服务。

但部署并不是故事的结束。您还需要编写监控代码，定期检查系统的实时性能，并在性能下降时触发警报。它可能下降得非常快，例如，如果您的基础设施中的某个组件损坏，但请注意，它也可能非常缓慢地衰减，这可能会长时间不被注意。这种情况很常见，因为数据漂移：如果模型是用去年的数据训练的，那么它可能无法适应今天的数据。

因此，您需要监控模型的实时性能。但您如何做到这一点呢？嗯，这取决于情况。在某些情况下，模型的表现可以从下游指标中推断出来。例如，如果您的模型是推荐系统的一部分，并且它建议用户可能感兴趣的产品，那么监控每天销售的建议产品数量很容易。如果这个数字下降了（与非推荐产品相比），那么首要嫌疑人就是模型。这可能是因为数据管道损坏，或者可能需要用新鲜数据重新训练模型（我们将在稍后讨论）。

然而，您可能还需要人工分析来评估模型的表现。例如，假设您训练了一个图像分类模型（我们将在第三章中探讨这些内容）来检测生产线上的各种产品缺陷。在数千个有缺陷的产品被发送到您的客户之前，您如何获得模型性能下降的警报？一个解决方案是将模型分类的所有图片样本（特别是模型不太确定的图片）发送给人工评分员。根据任务的不同，评分员可能是专家，也可能是非专业人士，例如众包平台（例如，Amazon Mechanical Turk）上的工人。在某些应用中，他们甚至可以是用户本人，例如通过调查或重新设计的验证码进行响应。⁠^(15)
