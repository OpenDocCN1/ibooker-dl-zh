
```py
public void getTrainingStatus(GetTrainingStatusRequest request) {
  int jobId = request.getJobId();
  .. .. ..  
  if (store.finalizedJobs.containsKey(jobId)) {           ❶
    job = store.finalizedJobs.get(jobId);
    status = job.isSuccess() ? TrainingStatus.succeed 
        : TrainingStatus.failure;

  } else if (store.launchingList.containsKey(jobId)) {    ❷
    job = store.launchingList.get(jobId);
    status = TrainingStatus.launch;

  } else if (store.runningList.containsKey(jobId)) {      ❸
    job = store.runningList.get(jobId);
    status = TrainingStatus.running;
  } else {                                                ❹
    TrainingJobMetadata metadata = store.jobQueue.get(jobId);
    status = TrainingStatus.waiting;
       .. .. ..
   }
   .. .. ..
}
```

❶ 在已完成的作业列表中搜索作业

❷ 在启动作业列表中搜索作业

❸ 在运行中的作业列表中搜索作业

❹ 作业仍在等待作业队列中。

因为 Docker 作业跟踪器实时将作业移动到相应的作业列表中，我们可以依靠使用作业队列类型来确定训练作业的状态。

### 3.3.6 意图分类模型训练代码

到目前为止，我们一直在处理训练服务代码。现在让我们看看最后一部分，模型训练代码。请不要被这里的深度学习算法吓到。这个代码示例的目的是向您展示一个具体的示例，说明训练服务如何与模型训练代码交互。图 3.9 描绘了示例意图分类训练代码的工作流程。

![](img/03-09.png)

图 3.9 意图分类训练代码工作流程首先从环境变量中读取所有输入参数，然后下载数据集，处理数据集，并启动训练循环。最后，它上传输出模型文件。

我们的示例训练代码训练一个三层神经网络以执行意图分类。首先，它从由我们的训练服务传递的环境变量中获取所有输入参数（请参阅第 3.3.4 节）。输入参数包括超参数（epoch 数、学习率等）、数据集下载设置（MinIO 服务器地址、数据集 ID、版本哈希）和模型上传设置。接下来，训练代码下载和解析数据集，并开始迭代学习过程。在最后一步中，代码将生成的模型和训练指标上传到元数据存储中。以下代码清单突出显示了前面提到的主要步骤（`train-service/text-classification/train.py` 和 `train-service/text-classification/Dockerfile`）。

清单 3.7 意图分类模型训练代码和 Docker 文件

```py
# 1\. read all the input parameters from 
# environment variables, these environment 
# variables are set by training service - docker job tracker.
EPOCHS = int_or_default(os.getenv('EPOCHS'), 20)
.. .. ..
TRAINING_DATA_PATH = os.getenv('TRAINING_DATA_PATH')

# 2\. download training data from dataset management
client.fget_object(TRAINING_DATA_BUCKET, 
  TRAINING_DATA_PATH + "/examples.csv", "examples.csv")
client.fget_object(TRAINING_DATA_BUCKET, 
  TRAINING_DATA_PATH + "/labels.csv", "labels.csv")

# 3\. prepare dataset
.. .. ..
train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(split_test_, batch_size=BATCH_SIZE,
                            shuffle=True, collate_fn=collate_batch)

# 4\. start model training
for epoch in range(1, EPOCHS + 1):
   epoch_start_time = time.time()
   train(train_dataloader)
   .. .. ..

print('Checking the results of test dataset.')
accu_test = evaluate(test_dataloader)
print('test accuracy {:8.3f}'.format(accu_test))

# 5\. save model and upload to metadata store.
.. .. ..
client.fput_object(config.MODEL_BUCKET, 
  config.MODEL_OBJECT_NAME, model_local_path)
artifact = orca3_utils.create_artifact(config.MODEL_BUCKET, 
  config.MODEL_OBJECT_NAME)
.. .. ..
```

注意 我们希望我们的示例训练代码演示了深度学习训练代码遵循常见模式。通过容器化以及传递参数的清晰协议，训练服务可以执行各种训练代码，而不论训练框架或模型架构。

### 3.3.7 训练作业管理

在第 3.1.2 节中，我们提到良好的训练服务应该解决计算隔离并提供按需计算资源（原则 4）。这种隔离有两重含义：训练过程的执行隔离和资源消耗隔离。由于我们使用 Docker 对训练过程进行容器化，所以这个执行隔离是由 Docker 引擎保证的。但是资源消耗隔离仍然需要我们自己处理。

想象一下，来自不同团队的三个用户（A、B 和 C）向我们的训练服务提交训练请求。如果用户 A 提交了 100 个训练请求，然后用户 B 和 C 都各自提交了一个请求，那么用户 B 和 C 的请求将在等待作业队列中等待一段时间，直到用户 A 的所有训练请求完成。当我们将训练集群视为每个人的游戏场时，这就是发生的情况：一个重度使用情况会主导作业调度和资源消耗。

为了解决资源竞争问题，我们需要在训练集群内为不同团队和用户设置边界。我们可以在训练集群内创建机器池以实现资源消耗隔离。每个团队或用户可以分配到一个专用的机器池，每个池都有自己的 GPU 和机器，池的大小取决于项目需求和训练使用情况。此外，每个机器池可以有一个专门的作业队列，因此重度用户不会影响其他用户。图 3.9 展示了这种方法的运作方式。

注意 资源隔离方法，像我们刚刚提到的服务器池方法，在资源利用方面可能不够高效。例如，服务器池 A 可能非常忙，而服务器池 B 可能处于空闲状态。可以定义每个服务器池的大小为一个范围，而不是一个固定数字，例如最小 5 台服务器、最大 10 台服务器，以提高资源利用率。然后可以应用额外的逻辑，以在服务器之间进行移动或提供新服务器。

实现图 3.10 的理想方法是使用 Kubernetes。Kubernetes 允许您创建由相同物理集群支持的多个虚拟集群，称为命名空间。Kubernetes 命名空间是一个消耗非常少系统资源的轻量级机器池。

![](img/03-10.png)

图 3.10 在训练集群内创建机器池，以为不同用户设置资源消耗边界。

如果您正在使用 Kubernetes 管理服务环境和计算集群，那么设置此类隔离非常容易。首先，您需要创建一个拥有资源配额的命名空间，例如 CPU 数量、内存大小和 GPU 数量；然后，在训练服务中定义用户及其命名空间的映射关系。

现在，当用户提交一个训练请求时，训练服务首先通过检查请求中的用户信息找到正确的命名空间，然后调用 Kubernetes API 将训练可执行文件放置在该命名空间中。由于 Kubernetes 实时跟踪系统的使用情况，它知道一个命名空间是否有足够的容量，如果命名空间已满，它将拒绝作业启动请求。

正如您所见，通过使用 Kubernetes 管理训练集群，我们可以将资源容量跟踪和资源隔离管理从训练服务中卸载出来。这是选择 Kubernetes 构建深度学习训练集群管理的一个原因。

### 3.3.8 故障排除指标

在此示例服务中，我们没有演示指标的使用。通常，指标是用于评估、比较和跟踪性能或生产的定量评估的常见度量。对于深度学习训练，我们通常定义两种类型的指标：模型训练执行指标和模型性能指标。

模型训练执行指标包括资源饱和率、训练作业执行可用性、平均训练作业执行时间和作业失败率。我们检查这些指标，以确保训练服务健康运行，并且用户的日常活动正常。例如，我们期望服务可用性超过 99.99% ，训练作业失败率小于 0.1% 。

模型性能指标衡量模型学习的质量。它包括每个训练迭代（epoch）的损失值和评估分数，以及最终的模型评估结果，如准确率、精度和 F1 分数。

对于与模型性能相关的指标，我们需要以更有组织的方式存储这些指标，以便能够使用统一的方法轻松搜索信息并比较不同训练运行之间的性能。我们将在第八章对此进行更详细的讨论。

### 3.3.9 支持新算法或新版本

现在让我们讨论如何将更多的训练代码接入到我们的示例训练服务中。在当前的实现中，我们使用请求中的 `algorithm` 变量来定义用户训练请求与训练代码之间的映射关系，使用 `algorithm` 变量在请求中找到对应的训练镜像。底层规则是 `algorithm` 变量必须等于 Docker 镜像名称，否则训练服务无法找到正确的镜像来运行模型训练。

以我们的意图分类训练为例。首先，我们需要将意图训练 Python 代码 Docker 化为 Docker 镜像并将其命名为“intent-classification”。然后，当用户使用`algorithm = 'intent-classification'`参数发送训练请求时，Docker 作业跟踪器将使用算法名称（即 intent-classification）在本地 Docker 仓库中搜索“intent-classification”训练镜像并将镜像运行为训练容器。 

这种方法肯定过于简化了，但它演示了我们如何与数据科学家一起定义将用户训练请求映射到实际训练代码的正式协议。在实践中，训练服务应该提供一组 API，允许数据科学家以自助方式注册训练代码。

一种可能的方法是在数据库中定义算法名称和训练代码映射，添加一些 API 来管理这个映射。建议的 API 可以是 

+   `createAlgorithmMapping(string algorithmName, string image, string version)` 

+   `updateAlgorithmVersion(string algorithmName, string image, string version)`

如果数据科学家想要添加新的算法类型，他们可以调用`createAlgorithmMapping` API 来向训练服务注册新的训练图像和新的算法名称。我们的用户只需要在训练请求中使用这个新的算法名称即可使用这个新的算法开始模型训练。 

如果数据科学家想要发布现有算法的新版本，他们可以调用`updateAlgorithmVersion` API 来更新映射。我们的用户仍然会有相同的算法名称（如意图分类），发送请求，但他们不会意识到训练代码在幕后升级到不同的版本。同时，值得指出的是，服务的公共 API 不会受到添加新训练算法的影响；只有一个新参数值被使用。

## 3.4. Kubeflow 训练操作符：开源方法

看完我们的示例训练服务后，让我们看看一个开源的训练服务。在本节中，我们将讨论来自 Kubeflow 项目的一组开源训练操作符。这些训练操作符可立即使用，并且可以在任何 Kubernetes 集群中独立设置。

Kubeflow 是一个成熟、开源的机器学习系统，适用于生产环境。我们在附录 B.4 中简要介绍了它，以及亚马逊 SageMaker 和 Google Vertex AI。我们推荐使用 Kubeflow、可扩展、可分布式和稳健性高的训练操作符。我们将首先讨论高级系统设计，然后讨论如何将这些训练操作符集成到自己的深度学习系统中。

什么是 Kubeflow？

Kubeflow 是一个开源的机器学习平台（源自谷歌），用于开发和部署生产级别的机器学习模型。你可以将 Kubeflow 视为 Amazon SageMaker 的开源版本，但它原生运行在 Kubernetes 上，因此它是云无关的。Kubeflow 将完整的机器学习功能集成到一个系统中——从 Notebooks 和管道到训练和服务。

即使你不打算使用它，我强烈建议你关注 Kubeflow 项目。Kubeflow 是一个设计良好且相当先进的深度学习平台；它的功能列表涵盖了整个机器学习生命周期。通过审查其用例、设计和代码，你将深入了解现代深度学习平台。

此外，因为 Kubeflow 是在 Kubernetes 的原生基础上构建的，你可以轻松地在你的本地或生产环境中设置整个系统。如果你不感兴趣借鉴整个系统，你也可以移植其中一些组件——如训练操作器或超参数优化服务——它们可以自行在任何 Kubernetes 环境中开箱即用。

### 3.4.1 Kubeflow 训练操作器

Kubeflow 提供了一组训练操作器，例如 TensorFlow 操作器、PyTorch 操作器、MXNet 操作器和 MPI 操作器。这些操作器涵盖了所有主要训练框架。每个操作器都有知识可以启动和监视用特定类型的训练框架编写的训练代码（容器）。

如果你计划在 Kubernetes 集群中运行模型训练，并希望设置自己的训练服务以减少操作成本，Kubeflow 训练操作器是完美的选择。以下是三个原因：

+   *轻松安装和低维护*——Kubeflow 操作器开箱即用；通过几行 Kubernetes 命令，你可以使它们在你的集群中工作。

+   *兼容大多数训练算法和框架*——只要你将训练代码容器化，就可以使用 Kubeflow 操作器执行它。

+   *易于集成到现有系统*——由于 Kubeflow 训练操作器遵循 Kubernetes 操作器设计模式，因此你可以使用 Kubernetes 的声明性 HTTP API 提交训练作业请求并检查作业运行状态和结果。你也可以使用 RESTful 查询与这些操作器交互。

### 3.4.2 Kubernetes 操作器/控制器模式

Kubeflow 训练操作器遵循 Kubernetes 操作器（控制器）设计模式。如果我们理解了此模式，那么运行 Kubeflow 训练操作器并阅读其源代码就很简单了。图 3.11 显示了控制器模式的设计图。

![](img/03-11.png)

图 3.11 Kubernetes 操作器/控制器模式运行无限控制循环，观察某些 Kubernetes 资源的实际状态（在右侧）和期望状态（在左侧），并尝试将实际状态移动到期望状态。

Kubernetes 中的所有内容都围绕着资源对象和控制器构建。Kubernetes 的资源对象，如 Pods、Namespaces 和 ConfigMaps，是持久化实体（数据结构），代表着集群的状态（期望的和当前的）。控制器是一个控制循环，它对实际的系统资源进行更改，以将您的集群从当前状态带到更接近期望状态，这在资源对象中定义。

注意 Kubernetes pod 是您可以在 Kubernetes 中创建和管理的最小部署单元。Pod 可以被视为运行一个或多个 Docker 容器的“逻辑主机”。有关 Kubernetes 概念的详细解释，例如 Namespaces 和 ConfigMaps，可以在官方网站找到：[`kubernetes.io/docs/concepts/`](https://kubernetes.io/docs/concepts/)

例如，当用户应用 Kubernetes 命令来创建一个 pod 时，它将在集群中创建一个 pod 资源对象（一个数据结构），其中包含所需的状态：两个 Docker 容器和一个磁盘卷。当控制器检测到这个新的资源对象时，它将在集群中提供实际的资源，并运行两个 Docker 容器并附加磁盘。接下来，它将更新 pod 资源对象的最新实际状态。用户可以查询 Kubernetes API 来获取此 pod 资源对象的更新信息。当用户删除此 pod 资源对象时，控制器将删除实际的 Docker 容器，因为所需状态已更改为零。
