
最后但同样重要的是，`Vena`使用托管在 Web 服务器组件上的 Web UI 来监视工作流程的执行。因为调度程序/控制器组件和工作器实时向元数据数据库报告状态，所以 Web UI 始终显示最新的工作流程状态。

### 9.2.3 工作流程编排设计原则

因为我们已经看到了工作流编排系统在内部和外部的工作方式，现在是时候研究使编排系统在深度学习场景中出色的设计原则了。我们希望您可以将这些原则作为指导，来改进您的系统或评估开源方法。

注意：在深度学习系统中，工作流编排系统是最复杂的组件之一，涉及到大量的工程工作。所以，在最初的几个版本中，不必过于担心使您的系统与这些原则完全匹配。

**原则 1：重要性**

工作流编排本质上是一种作业调度的挑战，所以任何编排系统的底线都是提供一个可靠的工作流执行体验。一个有效的工作流应该总是能够正确、重复地按计划执行。

**原则 2：可用性**

在深度学习环境中，编排系统的可用性衡量标准是是否优化了数据科学家的工作效率。在一个编排系统中，数据科学家的大部分交互工作都是工作流的创建、测试和监控。因此，一个用户友好的编排系统应该让用户能够轻松地创建、监控和排除故障。

**原则 3：可扩展性**

为了适应各种深度学习基础设施，人们应该能够轻松定义自己的任务操作符和执行器，而不用担心它们部署在哪里。编排系统应该提供适合您环境的抽象级别，无论是 Amazon EC2 还是 Kubernetes。

**原则 4：隔离性**

可能发生两种关键的隔离：工作流创建隔离和工作流执行隔离。工作流创建隔离意味着在创建工作流时，人们不能相互干扰。例如，如果 `Vena`提交了一个无效的工作流有向无环图（DAG），或者发布了一个在其他工作流中被引用的共享库的新版本，那么现有的工作流不应受到影响。

工作流执行隔离意味着每个工作流在一个独立的环境中运行。工作流之间不应有资源竞争，并且一个工作流的失败不会影响其他工作流的执行。

**原则 5：扩展性**

一个好的编排系统应该解决以下两个扩展性问题：处理大量同时运行的工作流以及处理大型扩展性工作流。同时运行的工作流扩展性通常指，给定足够的计算资源——例如，向工作组中添加更多的工作节点——编排系统可以满足无限数量的并发工作流执行。此外，系统应始终为每个工作流保持服务级别协议（SLA）。例如，工作流应在其预定时间执行，且不得晚于 2 秒，无论有多少其他工作流正在执行。

对于单一的大型工作流扩展，系统应该鼓励用户不必担心性能，这样他们就可以专注于可读性强、直接明了的代码和简单的操作。当工作流执行达到限制时——例如，训练运算符执行时间过长——编排系统应该提供一些水平并行运算符，例如分布式训练运算符，以解决单个工作流性能问题。

深度学习编排的主要扩展思想是我们应该在系统级别解决性能问题，并避免要求用户考虑可扩展性编写代码。这可能导致可读性下降、调试困难和操作负担增加。

**原则 6：人本支持，既适用于原型设计，也适用于生产环境**

连接数据科学家本地原型代码到生产工作流的能力是深度学习特有的要求。这是我们用来评估编排系统是否适合深度学习系统的关键指标。

一个为深度学习设计的编排系统将尊重深度学习项目开发是从原型到生产的迭代持续努力的事实。因此，它将不遗余力地帮助数据科学家将他们的本地原型代码转换为生产工作流。

## 9.3 巡回开源工作流编排系统

在本节中，我们将介绍三种经过实战验证的工作流编排系统：`Airflow`、`Argo Workflows`和`Metaflow`。这三个开源系统在 IT 行业得到了广泛应用，并得到了活跃社区的支持。除了一般介绍外，我们还将从深度学习项目开发的角度评估这些工作流系统。

为了进行公正的比较，我们在 `Airflow`、`Argo Workflows`和`Metaflow`中为相同工作流实现伪代码。基本上，如果有新数据，我们首先转换数据并将其保存到数据库的新表中，然后通知数据科学团队。此外，我们希望工作流每天运行。

### 9.3.1 Airflow

`Airflow`（[`airflow.apache.org/docs/apache-airflow/stable/index.html`](https://airflow.apache.org/docs/apache-airflow/stable/index.html)）于 2014 年在 Airbnb 创建，现在是 Apache 基金会的一部分。`Airflow`是一个平台，用于以编程方式编写、调度和监视工作流。`Airflow`并不是为深度学习用例设计的；它最初是为了编排越来越复杂的 ETL（抽取、转换、加载）管道（或数据管道）而构建的。但由于 `Airflow`具有良好的可扩展性、生产质量和 GUI 支持，它被广泛应用于许多其他领域，包括深度学习。截至本书撰写时，`Airflow`是最受欢迎的编排系统。

**典型用例**

在 `Airflow`中构建工作流程需要两个步骤。首先，定义工作流 DAG 和任务。其次，在 DAG 中声明任务之间的依赖关系。`Airflow` DAG 本质上是 Python 代码。看以下清单，了解我们的示例工作流在 `Airflow`中是如何实现的。

**清单 9.1 一个示例 Airflow 工作流定义**

```py
# declare the workflow DAG. 
with DAG(dag_id="data_process_dag",
        schedule_interval="@daily",
        default_args=default_args,
        template_searchpath=[f"{os.environ['AIRFLOW_HOME']}"],
        catchup=False) as dag:

 # define tasks of the workflow, each code section below is a task 

   is_new_data_available = FileSensor(       ❶
       task_id="is_new_data_available",
       fs_conn_id="data_path",
       filepath="data.csv",
       .. .. ..
   )

 # define data transformation task
   transform_data = PythonOperator( 
       task_id="transform_data",
       python_callable=transform_data        ❷
   )

 # define table creation task
   create_table = PostgresOperator(          ❸
       task_id="create_table",
       sql='''CREATE TABLE IF NOT EXISTS invoices (
               .. .. ..
               );''',
       postgres_conn_id='postgres',
       database='customer_data'
   )

   save_into_db = PythonOperator(
       task_id='save_into_db',
       python_callable=store_in_db
   )

   notify_data_science_team = SlackWebhookOperator(
       task_id='notify_data_science_team',
       http_conn_id='slack_conn',
       webhook_token=slack_token,
       message="Data Science Notification \n"
       .. .. ..
   )

# Step two, declare task dependencies in the workflow
  is_new_data_available >> transform_data
  transform_data >> create_table >> save_into_db
  save_into_db >> notify_data_science_team
  save_into_db >> create_report

# The actual data transformation logic, which is referenced
# in the "transform_data" task.
def transform_data(*args, **kwargs):
   .. .. ..
```

❶ 检查是否有新文件到达

❷ 实际逻辑是在 `transform_data`函数中实现的。

❸ `PostgresOperator`是预定义的 `Airflow`运算符，用于与 postgres db 交互。

在代码清单 9.1 中，我们看到示例工作流 DAG 包含多个任务，如 `create_table`和`save_into_db`。在 `Airflow`中，任务被实现为运算符。有许多预定义和社区管理的运算符，例如 `MySqlOperator`、`SimpleHttpOperator`和 Docker 运算符。

`Airflow`的预定义运算符帮助用户实现任务而无需编码。您还可以使用 `PythonOperator`运行自定义的 Python 函数。一旦工作流 DAG 被构建并且所有代码被部署到 `Airflow`，我们可以使用 UI 或以下 CLI 命令来检查工作流执行状态；以下是一些示例 shell 命令：

```py
airflow dags list                             ❶

airflow tasks list data_process_dag           ❷

airflow tasks list data_process_dag --tree    ❸
```

❶ 打印所有活动的 DAG

❷ 打印 `data_process_dag` DAG 中任务的列表

❸ 打印 `data_process_dag` DAG 中任务的层次结构

如果您想了解更多关于 `Airflow`的信息，您可以查看其架构概述文档和教程 ([`mng.bz/Blpw`](http://mng.bz/Blpw))。

**关键功能**

`Airflow`提供以下关键功能：

+   *DAGs* — `Airflow`通过 **DAGs** 抽象复杂的工作流程，工作流 DAG 是通过 Python 库实现的。

+   *程序化工作流管理* — `Airflow`支持动态创建任务，并允许创建复杂的动态工作流。

+   *出色的内置运算符帮助构建自动化* — `Airflow`提供了许多预定义的运算符，帮助用户实现任务而无需编码。

+   *可靠的任务依赖性和执行管理* — `Airflow`在每个任务中都内置了自动重试策略，并提供了不同类型的传感器来处理运行时依赖关系，例如检测任务完成、工作流运行状态变更和文件存在。

+   *可扩展性* — `Airflow`使其传感器、钩子和运算符完全可扩展，这使得它能够从大量社区贡献的运算符中受益。`Airflow`还可以通过添加定制运算符轻松集成到不同的系统中。

+   *监控和管理界面* — `Airflow`提供了一个强大的用户界面，用户可以快速了解工作流/任务执行状态和历史。用户还可以从界面触发和清除任务或工作流运行。

+   *生产质量* — `Airflow`提供了许多有用的工具，用于在生产环境中维护服务，如任务日志搜索、扩展、报警和 restful API。

**限制**

尽管 `Airflow`是一个出色的工作流编排工具，但在深度学习场景中使用时仍然存在一些缺点：

+   *数据科学家入门时的高前期成本* — `Airflow`对于实现不受内置运算符支持的任务具有陡峭的学习曲线。此外，没有一种简单的方法进行工作流本地测试。

+   *将深度学习原型代码移至生产环境时的高摩擦力* — 当我们将 `Airflow`应用于深度学习时，数据科学家必须将他们的本地模型训练代码转换为 `Airflow` DAG。这是额外的工作，对于数据科学家来说是一种不愉快的体验，特别是考虑到如果我们直接从模型训练代码构建工作流程 DAG，这是可以避免的。

+   *在 Kubernetes 上操作时的高复杂性* — 在 Kubernetes 上部署和操作 `Airflow`并不简单。如果您希望采用一个编排系统在 Kubernetes 上运行，`Argo Workflows`是一个更好的选择。

### 9.3.2 Argo Workflows

`Argo Workflows`是一个开源的、容器原生的工作流引擎，用于在 Kubernetes 上编排并行工作流程/任务。`Argo Workflows`解决了与 `Airflow`相同的问题，但采用了不同的方式；它采用了 Kubernetes 本地方法。

`Argo Workflows`和 `Airflow`之间最大的区别在于 `Argo Workflows`在 Kubernetes 上本地构建。更具体地说，`Argo Workflows`中的工作流程和任务以 Kubernetes 自定义资源定义（CRD）对象实现，并且每个任务（步骤）都作为 Kubernetes pod 执行。请参阅图 9.6 以获得高级系统概述。

![](img/09-06.png)

**图 9.6 Argo Workflows 中的工作流程及其步骤是作为 Kubernetes pod 执行的。**

在图 9.6 中，`Vena`（数据科学家）首先将工作流程及其步骤/任务定义为 Kubernetes CRD 对象，通常表示为 YAML 文件。然后她将工作流提交到 `Argo Workflows`，其控制器在 Kubernetes 集群内创建 CRD 对象。接下来，Kubernetes pod 动态启动以按工作流程顺序运行工作流程步骤/任务。

您还可以注意到，每个步骤的执行完全由容器和 pod 隔离；每个步骤使用文件来表示其输入和输出值。`Argo Workflows`会自动将依赖文件挂载到步骤的容器中。

Kubernetes pod 创建的任务隔离是 `Argo Workflows`的一个巨大优势。简单性也是人们选择 `Argo Workflows`的另一个原因。如果您了解 Kubernetes，`Argo`的安装和故障排除都很简单。我们可以使用 `Argo Workflows`命令或标准的 Kubernetes CLI 命令来调试系统。

**典型的使用案例**

为了更好地理解，让我们看一个 `Argo Workflows`的例子。在本节中，我们使用 `Argo Workflows`来自动化我们在之前 `Airflow`部分看到的相同数据处理工作。工作流程包括首先检查新数据，转换数据，将其保存到数据库中的新表中，然后通过 Slack 通知数据科学家团队。请参阅以下代码清单以查看 `Argo Workflows`的定义。

**代码清单 9.2 Argo Workflows 的示例工作流程，包含一系列步骤**
