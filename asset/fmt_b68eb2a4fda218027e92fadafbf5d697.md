
## 方法一：通过命令行 CLI 工具打包模型

```py
torch-model-archiver --model-name intent_classification --version 1.0 \
 --model-file torchserve_model.py --serialized-file \
    workspace/MiniAutoML/{model_id}/model.pth \
 --handler torchserve_handler.py --extra-files \
workspace/MiniAutoML/{model_id}/vocab.pth,
➥ workspace/MiniAutoML/{model_id}/manifest.json
```

## 方法二：在训练代码中打包模型

```py
model_archiver.archive(model_name=archive_model_name, 
  handler_file=handler, model_state_file=model_local_path,
  extra_files=extra_files, model_version=config.MODEL_SERVING_VERSION,
  dest_path=config.JOB_ID)
```

### 7.2.7 在 Kubernetes 中扩展规模

在我们的示例服务中，为了演示目的，我们运行单个 `TorchServe` 容器作为预测后端，但这在生产环境中并非如此。扩展 `TorchServe` 面临的挑战如下：

+   负载均衡器使 `TorchServe` 模型注册变得困难。在 `TorchServe` 中，模型文件需要先注册到 `TorchServe` 服务器，然后才能使用。但是在生产环境中，`TorchServe` 实例被放置在网络负载均衡器后，因此我们只能将预测请求发送到负载均衡器，让它将请求路由到随机的 `TorchServe` 实例。在这种情况下，由于我们无法指定哪个 `TorchServe` 实例为哪个模型提供服务，注册模型变得困难。负载均衡器向我们隐藏了 `TorchServe` 实例。

+   每个 `TorchServe` 实例都需要有一个用于加载模型的模型存储目录，并且在可以注册之前，模型文件需要放在模型存储目录中。有多个 `TorchServe` 实例会使模型文件复制变得难以管理，因为我们需要知道每个 `TorchServe` 实例的 IP 地址或 DNS 来复制模型文件。

+   我们需要在 `TorchServe` 实例之间平衡模型。让每个 `TorchServe` 实例加载每个模型文件是一个糟糕的想法；这将极大浪费计算资源。我们应该将负载均匀分配到不同的 `TorchServe` 实例上。

要解决这些挑战并扩展 `TorchServe` 后端，我们可以在 Kubernetes 中引入“边车”模式。图 7.7 描绘了整体概念。

![](img/07-07.png)

图 7.7 在 Kubernetes 中的 `TorchServe` pod 中添加代理容器以扩展 `TorchServe`。

图 7.7 中的建议是在每个 `TorchServe` pod 中与 `TorchServe` 容器一起添加代理容器（作为边车）。我们不直接调用 `TorchServe` API，而是将预测请求发送到代理容器。代理容器中的代理 API 将隐藏 `TorchServe` 模型管理细节，包括模型下载和模型注册。它将准备 `TorchServe` 容器以服务任意模型。

添加代理容器后，模型服务工作流程（图 7.7）如下。首先，预测请求落在代理容器上。其次，代理下载模型文件并将其输入到共享磁盘（模型存储库）。第三，代理将模型注册到 `TorchServe` 容器并将推理请求转换为 `TorchServe` 格式。第四，`TorchServe` 容器运行模型服务并将结果返回给代理。最后，代理容器将预测响应返回给用户。

通过添加代理容器，我们无需担心将预测请求发送到未注册该模型的 `TorchServe` 实例。代理容器（边车）将确保 `TorchServe` 容器准备好处理任何预测请求，方法是将模型文件复制到模型存储库并注册模型。这也简化了资源管理工作，因为现在我们可以简单地依赖负载均衡器将预测负载（模型）在 `TorchServe` pod 之间分配。此外，通过在所有 `TorchServe` pod 之间共享磁盘，我们可以为所有 `TorchServe` 实例共享模型存储库，从而减少了模型下载时间并节省了网络带宽。

边车模式：运行模型服务器的常用方法。

在第 7.4 节中，我们将介绍其他几种模型服务器方法，例如 TensorFlow serving 和 Triton。尽管这些模型服务器的实现方式不同，但它们的设计思想是相似的。它们都采用黑盒方法，并需要特定的模型格式和一些模型管理来启用模型服务。

图 7.7 中的旁路模式是在 Kubernetes pod 中运行这些不同模型服务器容器的常见解决方案。代理容器封装了模型服务器的所有特殊要求，并且只暴露通用模型服务 API。

## 7.3 模型服务器 vs. 模型服务

在设计模型服务应用程序时，选择模型服务器方法和模型服务方法之间是我们需要做出的第一个决定。当我们选择不当时，我们的服务应用程序要么难以使用和维护，要么构建所需时间过长。

我们已经在第六章（第 6.2 节和第 6.3 节）中回顾了这两种方法之间的差异，但这是一个非常关键的选择，值得再次审查。现在你已经看到了每种方法的具体示例，这些想法可能更容易理解。

通过在第 7.1 节和第 7.2 节中讨论的两个示例服务，可以清楚地看出，模型服务器方法避免了为特定模型类型构建专用后端预测器的工作量。相反，它可以直接使用，并且可以为任意模型提供服务，而不管模型实现了哪种算法。因此，模型服务器方法似乎应该始终是最佳选择。但这并不是真的；选择模型服务器或模型服务应该取决于用例和业务需求。

对于单一应用场景，在实践中，模型服务方法更容易构建和维护。模型服务后端预测器相当容易构建，因为模型服务代码是训练代码的简化版本。这意味着我们可以轻松地将模型训练容器转换为模型服务容器。一旦构建完成，模型服务方法更容易维护，因为我们完全拥有代码，并且工作流程简单。对于模型服务器方法，无论我们选择开源、预构建的模型服务器还是构建自己的服务器，设置系统的过程都很复杂。要学会并运营和维护系统需要花费大量的精力。

对于模型服务平台场景，系统需要支持多种不同类型的模型时，模型服务器方法无疑是最佳选择。当你为 500 种不同类型的模型构建模型服务系统时，如果选择模型服务器方法，你只需要一个单一类型的预测器后端来支持所有模型。相比之下，使用模型服务方法，你将需要 500 种不同的模型预测器！管理计算资源和进行所有这些预测器的维护工作非常困难。

我们的建议是，当你初学时使用模型服务方法，因为它更简单、更容易。当你需要在你的服务系统中支持超过 5 到 10 种类型的模型或应用时，你可以转向模型服务器方法。

## 7.4 开源模型服务工具巡回

有很多开源的模型服务工具可以使用。这些选择非常多，但是选择这么多有时会让人感到困惑。为了帮助您更轻松地做出选择，我们将向您介绍一些流行的模型服务工具，包括 TensorFlow Serving、`TorchServe`、Triton 和 KServe。所有这些工具都可以立即使用，并适用于生产用例。

因为这里描述的每一个工具都有详尽的文档，所以我们将保持讨论在一个通用的层面，只看它们的总体设计、主要特征和适当使用情况。这些信息应该足以作为一个起点，让你自己深入探索。

### 7.4.1 TensorFlow Serving

TensorFlow Serving ([`www.tensorflow.org/tfx/guide/serving`](https://www.tensorflow.org/tfx/guide/serving)) 是一个可自定义的独立网页系统，用于在生产环境中提供 TensorFlow 模型。TensorFlow Serving 采用模型服务器方法；它可以使用相同的服务器架构和 API 为所有类型的 TensorFlow 模型提供服务。

#### 特征

TensorFlow Serving 提供以下特点：

+   可以为多个模型或相同模型的多个版本提供服务。

+   与 TensorFlow 模型具有开箱即用的整合。

+   自动发现新的模型版本，支持不同的模型文件源。

+   具有统一的 gRPC 和 HTTP 端点，用于模型推断。

+   支持批量预测请求和性能调优。

+   具有可扩展的设计，可以在版本策略和模型加载上进行自定义。

#### 高级架构

在 TensorFlow Serving 中，一个模型由一个或多个可用服务程序组成。可用服务程序是用于执行计算的基础对象（例如查找或推断）；它是 TensorFlow Serving 中的中心抽象。来源是用于查找和提供可用服务程序的插件模块。装载器标准是用于装载和卸载可用服务程序的 API。管理器处理可用服务程序的整个生命周期，包括加载、卸载和提供可用服务程序。

![](img/07-08.png)

图 7.8 显示了 TensorFlow Serving 的架构和模型服务的生命周期。蓝色 = 最深的灰色；绿色 = 较浅的灰色；黄色 = 最浅的灰色。（来源：TensorFlow；[`mng.bz/KlNj`](http://mng.bz/KlNj)）

图 7.8 说明了向客户提供可提供服务的工作流程。首先，源插件为特定的可用服务程序创建一个加载器，加载器包含加载可用服务程序的元数据。第二，源从文件系统（模型库）中找到一个可用服务程序；它通知 `DynamicManager` 可用服务程序的版本和加载器。第三，基于预定义的版本策略，`DynamicManager` 确定是否加载模型。最后，客户端发送一个预测请求给可用服务程序，`DynamicManager` 返回一个句柄，以便客户端可以执行模型。

#### TensorFlow Serving 模型文件

TensorFlow Serving 要求模型以 SavedModel ([`mng.bz/9197`](http://mng.bz/9197)) 格式保存。我们可以使用 `tf.saved_model.save(model, save_path)` API 来实现这个目的。一个保存的模型是一个包含了序列化签名和运行它们所需的状态的目录，包括变量值和词汇表。例如，一个保存的模型目录有两个子目录，`assets` 和 `variables`，以及一个文件，`saved_model.pb`。

`assets` 文件夹包含了 TensorFlow 图使用的文件，比如用于初始化词汇表的文本文件。`variables` 文件夹包含了训练检查点。`saved_model.pb` 文件存储了实际的 TensorFlow 程序，或者说模型，以及一组命名的签名，每个签名标识了一个接受张量输入并产生张量输出的函数。

#### 模型服务

因为 TensorFlow 的 SavedModel 文件可以直接加载到 TensorFlow Serving 进程中，所以运行模型服务非常简单。一旦服务进程启动，我们可以将模型文件复制到 TensorFlow Serving 的模型目录中，然后立即发送 gRPC 或 REST 预测请求。让我们来看下面的预测示例：

```py
# 1\. Save model in training code 
MODEL_DIR='tf_model'
version = "1"
export_path = os.path.join(MODEL_DIR, str(version))
model.save(export_path, save_format="tf")

# 2\. Start tensorflow serving service locally as a docker container
docker run -p 8501:8501 
--mount type=bind,source=/workspace/tf_model,target=/models/model_a/ 
-e MODEL_NAME=model_a -t tensorflow/serving
--model_config_file_poll_wait_seconds=60
--model_config_file=/models/model_a/models.config 

# 3\. Send predict request to local tensorflow serving docker container 
# The request url pattern to call a specific version of a model is
   /v1/models/<model name>/versions/<version number> 
json_response = requests.post('http:/ /localhost:8501/
 ➥ v1/models/model_a/versions/1:predict', 
  data=data, headers=headers)
```

为了将多个模型和同一模型的多个版本加载到服务服务器中，我们可以在模型配置中配置模型的版本，如下所示：

```py
model_config_list {
  config{
    name: 'model_a'
    base_path: '/models/model_a/'
    model_platform: 'tensorflow'
    model_version_policy{
      specific{
        versions:2    ❶
        versions:3    ❷
      }
    }
  }
  config{
    name: 'model_b'
    base_path: '/models/model_b/'
    model_platform: 'tensorflow'
  }
}
```

❶ 在 `/models/model_a/versions/2` 找到模型 v2

❷ 在 `/models/model_a/versions/3` 找到模型 v3

在这个配置中，我们定义了两个模型，`model_a` 和 `model_b`。因为 `model_a` 有一个 `model_version_policy`，所以两个版本（v2 和 v3）都被加载并可以提供请求服务。默认情况下，模型的最新版本将被提供服务，所以当检测到 `model_b` 的新版本时，旧版本将被新版本替换。

#### 回顾

TensorFlow Serving 是用于 TensorFlow 模型的生产级模型服务解决方案；它支持 REST、gRPC、GPU 加速、小批量处理和边缘设备上的模型服务。虽然 TensorFlow Serving 在高级指标、灵活的模型管理和部署策略方面存在不足，但如果你只有 TensorFlow 模型的话，它仍然是一个不错的选择。

TensorFlow Serving 的主要缺点是它是一个供应商锁定的解决方案；它只支持 TensorFlow 模型。如果你正在寻找一个训练框架无关的方法，那么 TensorFlow Serving 将不是你的选择。

### 7.4.2 TorchServe

`TorchServe` ([`pytorch.org/serve/`](https://pytorch.org/serve/)) 是一个性能出色、灵活且易于使用的工具，用于为 PyTorch eager 模式和 torchscripted 模型提供服务（torchscripted 模型是 PyTorch 模型的一种中间表示，可以在高性能环境（如 C++）中运行）。与 TensorFlow Serving 类似，`TorchServe` 采用模型服务器方法为所有类型的 PyTorch 模型提供服务，并提供统一的 API。不同之处在于 `TorchServe` 提供了一组管理 API，使模型管理非常方便灵活。例如，我们可以以编程方式注册和注销模型或模型的不同版本。我们还可以为模型和模型的不同版本扩展和缩小服务工作程序。

#### 高层架构

一个 `TorchServe` 服务器由三个组件组成：前端、后端和模型存储。前端处理 `TorchServe` 的请求/响应。它还管理模型的生命周期。后端是一组负责在模型上运行实际推断的模型工作程序。模型存储是一个包含所有可加载模型的目录；它可以是云存储文件夹或本地主机文件夹。图 7.9 显示了 `TorchServing` 实例的高级架构。

![](img/07-09.png)

图 7.9 `TorchServe` 架构图（来源：Kuldeep Singh，“使用 `TorchServe` 将命名实体识别模型部署到生产环境”，Analytics Vidhya，2020）

图 7.9 描绘了两个工作流程：模型推断和模型管理。对于模型推断，首先，用户将预测请求发送到模型的推断端点，例如 `/predictions/{model_name}/{version}`。然后，推断请求被路由到已加载模型的工作进程之一。接下来，工作进程将从模型存储中读取模型文件，并让模型处理器加载模型，预处理输入数据，并运行模型以获得预测结果。

对于模型管理，用户需要在可以访问模型之前注册模型。这可以通过使用管理 API 来完成。我们还可以为模型调整工作进程计数的规模。我们将在即将到来的示例使用部分中看到一个示例。

#### 功能

`TorchServe` 提供以下功能：

+   可以为多个模型或同一模型的多个版本提供服务

+   将模型推断的 gRPC 和 HTTP 端点统一起来

+   支持批量预测请求和性能调优

+   支持将 PyTorch 模型和 Python 函数组合成顺序和并行管道的工作流程

+   提供管理 API 来注册/注销模型和调整工作进程的规模

+   处理模型版本控制，用于 A/B 测试和实验

#### Torch 服务模型文件

纯 PyTorch 模型不能直接加载到 Torch 服务服务器中。`TorchServe` 要求将其所有模型打包成 `.mar` 文件。请参阅 7.2.6 节，了解如何创建 `.mar` 文件的详细示例。

#### 模型服务

下面的代码片段列出了使用 `TorchServe` 运行模型推理的五个一般步骤。有关具体示例，您可以查看我们的示例意图分类预测器的 README 文档（[`mng.bz/WA8a`](http://mng.bz/WA8a)）：
