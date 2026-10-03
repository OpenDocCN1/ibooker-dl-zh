
列表 10.9 突出了我们需要对`show-prediction.html`视图函数进行的更新。这些更改将输入值打包成 Vertex AI 端点部署所期望的格式，并通过`predict_custom_trained_model_sample()`函数在 Vertex AI 端点调用模型。

## 列表 10.9 `show-prediction.html`端点视图函数

```py
def show_prediction():
    scoring_dict = {}
    for col in config['scoring_columns_cat']:
        print("value for "+col+" is: "+str(request.args.get(col))) 
        scoring_dict[col] = str(request.args.get(col))
    for col in config['scoring_columns_cont']:
        scoring_dict[col] = float(request.args.get(col))
    scoring_dict['size_type_bin'] = str(request.args.get('size_type'))+' 1'
    scoring_dict.pop('size_type')                            ①
    input_dict = {name: [value] for name, \
value in scoring_dict.items()}                               ②
    print("input_dict: ",input_dict)
    predictions = predict_custom_trained_model_sample(
    project = config['endpoint']['project'],
    endpoint_id = config['endpoint']['endpoint_id'],
    location = config['endpoint']['location'],
    instances = input_dict)                                  ③
    prob = tf.nn.sigmoid(predictions[0])
```

① 从特征列表中删除`size_type`

② 将`scoring_dict`中的值转换为值列表

③ 调用`predict_custom_trained_model_sample`

列表 10.9 显示了对`show-prediction.html`视图函数的以下更新：

+   对端点的调用必须具有精确正确的特征和值列表，不能缺少特征或额外特征。由于我们不直接使用`size_type`与模型一起，我们需要使用`scoring_dict.pop('size_type')`语句显式地从特征和值的字典中删除它。

+   端点期望以与我们在 10.4.4 节中直接在控制台练习端点所使用的 JSON 相同的格式从字典中获取特征和值。这意味着字典中的值需要转换为值列表，每个列表中恰好包含一个值。此语句将字典转换为值列表：

```py
input_dict = {name: [value] for name, value in scoring_dict.items()}
```

+   `predict_custom_trained_model_sample()`函数需要使用指定项目、端点和我们想要预测的特征键值对的参数来调用。

除了对 Flask 服务器模块的这些更新外，我们还可以删除从本地文件系统加载模型的语句，因为我们在这个解决方案中不使用本地模型。

注意：使用端点的 Flask 服务器模块的更新版本可在 [`mng.bz/Xx5a`](https://mng.bz/Xx5a) 获取。

要运行此版本的 Flask 服务器模块，请按照以下步骤操作：

1. 通过在运行 Flask 服务器模块的本地系统上的命令行中运行以下命令，通过此应用程序授权您用于访问 Google Cloud 的账户：

```py
gcloud auth application-default login
```

2. 启动 Flask 服务器模块：

```py
python flask_endpoint_deploy.py
```

如果您遇到与 protobuf（协议缓冲区）级别相关的错误，请尝试以下命令：

```py
pip install protobuf==3.20.*n
```

此命令将 protobuf 级别调整为端点所需的精确级别。

注意：在本应用的目的上，您不需要了解 protobufs，但如果您对此感兴趣，可以查看文档：[`protobuf.dev/`](https://protobuf.dev/).

3. 当 Flask 服务器模块运行时，在浏览器中转到`localhost:5000`以测试部署。

现在我们已经看到了模型的两种部署方式：一种是从本地系统完全运行的 Web 部署，另一种是使用从 Vertex AI 端点提供模型的 Web 部署。

## 10.5.3 将模型部署到端点的益处

如果模型在简单的 Web 部署或部署到端点时得到相同的结果，您可能会问，将模型部署到端点有什么意义。将模型部署到端点可以带来许多好处：

+   您可以扩展端点实例以处理额外的负载。

+   您可以将多个模型部署到同一端点。想象一下这样一个场景，您需要用另一个模型替换生产环境中的一个模型。如果您将这两个模型部署到同一端点，您可以在不更改应用程序的情况下，逐渐调整流量分配到新模型的比例，而不会对应用程序的用户造成突兀的变化。

+   您可以将相同的模型部署到不同的端点，从而优化机器资源。例如，如果您有一个生产和开发环境，您可以将相同的模型部署到两个端点，生产环境使用更高规格的机器资源，而开发环境使用更便宜的机器资源。

+   使用端点，您可以使用 Vertex AI 监控来检测偏差（训练数据与在生产中应用模型时看到的数据之间的分布差异）和漂移（模型在生产中看到的数据分布随时间的变化）。模型监控可以帮助确保模型的性能不会随时间下降，并且模型在生产中应用的数据的变化不会意外发生。有关监控的更多详细信息，请参阅[`mng.bz/yWdd`](https://mng.bz/yWdd)上的文档。

图 10.36 显示了多个模型部署到同一端点和单个模型部署到多个端点的示例。

在图 10.36 中，有两种场景：

+   *同一端点部署多个模型*——模型 A 是当前生产中的版本，模型 B 是我们希望引入生产的下一级模型。通过调整 X 和 Y 的值，我们可以控制流量分配到每个模型的比例，逐渐增加分配到新模型级别的比例。

+   *同一模型部署到多个端点*——端点 1 具有基本的计算资源，端点 2 具有足够的计算资源来处理我们当前的生产负载。通过这种方式，我们可以优化系统的资源成本，以满足多组用户的需求。

![](img/CH10_F36_Ryan2.png)

图 10.36 模型和端点之间的关系

注意：有关 Vertex AI 中模型部署的更多详细信息，请参阅 [`mng.bz/MDlB`](https://mng.bz/MDlB) 的文档。

## 10.6 Gemini for Google Cloud：Google Cloud 中的生成式 AI 辅助

在你完成本章步骤的过程中，你可能遇到了障碍或需要澄清问题。为了取得进展，你可能查阅了 Google Cloud 的文档，在 Stack Overflow 中搜索，或向同事寻求帮助。除了这些传统的辅助来源外，Google Cloud 还包括一个集成、由生成式 AI 驱动的辅助来源：Gemini for Google Cloud。

Gemini for Google Cloud 是一套针对 Google Cloud 的生成式 AI 能力。Gemini for Google Workspace 是一套针对 Google Workspace（Google Docs、Sheets、Slides）的配套生成式 AI 能力（参见 [`workspace.google.com/solutions/ai/`](https://workspace.google.com/solutions/ai/)）。在本章和第十一章中，我们将展示如何通过 Gemini for Google Cloud 使用生成式 AI 来简化表格数据模型的部署，并自动化机器学习管道的一些脚本。Gemini for Google Cloud 提供了各种能力，包括以下内容：

+   回答有关 Google Cloud 的问题

+   从文本生成代码（包括 SQL 和各种编程语言，如 Python、Java 和 JavaScript）

+   解释代码

注意：有关 Gemini for Google Cloud 的更详细概述，请参阅 Gemini for Google Cloud 文档：[`cloud.google.com/gemini/docs/overview`](https://cloud.google.com/gemini/docs/overview)。

在本节中，我们将介绍如何设置 Gemini for Google Cloud 并使用它来回答有关 Google Cloud 的问题。我们还将讨论本章中完成的一些操作，以在 Vertex AI 端点部署我们的模型。在第十一章中，我们将回到 Gemini for Google Cloud，展示如何使用它来生成和解释代码。

Google Cloud 不是唯一利用生成式 AI 来简化平台使用并自动化开发工作流程中某些步骤的云平台。Azure 中的 Copilot 和 AWS 中的 Code Whisperer 是基于生成式 AI 的功能，每个都提供了 Gemini for Google Cloud 提供的部分好处。

### 10.6.1 设置 Gemini for Google Cloud

以下是为设置 Gemini for Google Cloud 提供的说明：

+   为项目设置 Gemini for Google Cloud：[`cloud.google.com/gemini/docs/quickstart`](https://cloud.google.com/gemini/docs/quickstart)

+   设置 Gemini Code Assist，这是 Gemini for Google Cloud 中提供 AI 辅助开发的部分，用于 Google Cloud 的 Cloud Shell 编辑器：[`mng.bz/avMm`](https://mng.bz/avMm)

一旦为项目设置了 Gemini for Google Cloud，你将在工具栏中看到 Gemini for Google Cloud 图标（参见图 10.37）。

![](img/CH10_F37_Ryan2.png)

图 10.37 Gemini for Google Cloud 图标

现在我们已经完成了 Gemini for Google Cloud 的基本设置，我们将在下一节中看到如何使用它来回答关于 Google Cloud 的问题。

### 10.6.2 使用 Gemini for Google Cloud 回答关于 Google Cloud 的问题

Gemini for Google Cloud 在 Google Cloud 中表现出多种方式，包括：

+   *在 Google Cloud 中可用的聊天面板中*——图 10.38 展示了 Gemini for Google Cloud 的聊天面板。

![图片](img/CH10_F38_Ryan2.png)

图 10.38 Gemini for Google Cloud 聊天面板

+   *在 Google Cloud 支持的多种 IDE 中，包括 VS Code、Cloud Workstations 和 Cloud Shell Editor*——图 10.39 展示了 Gemini for Google Cloud 如何从 Cloud Shell Editor 中的注释生成一个简单的 Python 函数。

![图片](img/CH10_F39_Ryan2.png)

图 10.39 Gemini for Google Cloud 在 Cloud Shell Editor 中生成 Python

+   *在 Big Query 和 Spanner 的查询编辑器中*——图 10.40 展示了 Gemini for Google Cloud 如何从 Spanner 查询编辑器中的注释生成 SQL。

![图片](img/CH10_F40_Ryan2.png)

图 10.40 Gemini for Google Cloud 在 Spanner 查询编辑器中生成 SQL

现在我们已经回顾了 Gemini for Google Cloud 的适用示例，接下来让我们更深入地考察 Gemini for Google Cloud 的聊天面板，看看它是如何被用来获取关于 Google Cloud 的问题答案。您可以通过点击 Gemini 图标（见图 10.37）来打开 Gemini for Google Cloud 聊天面板，如图 10.41 所示。

![图片](img/CH10_F41_Ryan2.png)

图 10.41 Gemini for Google Cloud 聊天面板

通过在提示字段中输入指令提示并点击发送按钮来尝试使用 Gemini for Google Cloud 聊天面板，如图 10.42 所示。

![图片](img/CH10_F42_Ryan2.png)

图 10.42 Gemini for Google Cloud 聊天面板发送按钮

Gemini for Google Cloud 聊天面板在 Google Cloud 中随时可用，您可以使用它来提出关于 Google Cloud 的各种问题。例如，假设我们想要创建一个 Cloud Storage 存储桶，但我们忘记了本章前面列出的步骤。图 10.43 显示了 Gemini for Google Cloud 对提示“如何创建 Google 存储桶”的响应。

![图片](img/CH10_F43_Ryan2.png)

图 10.43 Gemini for Google Cloud 对“如何创建 Google Cloud 存储桶”的响应

Gemini for Google Cloud 专门训练以提供关于使用 Google Cloud 的有用答案。通过给出与本章某些部分相关的提示来尝试使用 Gemini for Google Cloud。例如，尝试“如何将自定义模型导入 Vertex AI”或“如何将模型部署到 Vertex AI 端点”，并查看 Gemini for Google Cloud 的响应与本章类似部分中列出的步骤有何不同。

## 摘要

+   在简单的 Web 应用程序中部署您的模型可以给您一个关于其在生产中的特性的感觉。

+   我们可以创建一个简单的 Web 应用程序来部署我们在第九章训练的模型。这个应用程序包括一个 Flask 服务器模块和两个简单的网页。

+   公共云环境提供了一个部署模型的环境，允许你扩展容量并控制可用性。

+   为了在 Google Cloud 中部署我们在第九章训练的模型，我们需要创建一个 Google Cloud 项目，创建一个 Google Cloud 存储桶，并将模型上传到存储桶中。

+   一旦我们完成了准备 Google Cloud 模型部署的步骤，我们就可以将训练好的模型部署到 Vertex AI 端点。

+   我们可以通过对我们在本章开头创建的 Web 应用程序中的 Flask 模块进行一些简单的更新来测试模型的 Vertex AI 端点部署。

+   将模型部署到 Vertex AI 的端点使部署更加稳健。特别是，我们可以指定适合我们应用程序的机器资源，提供不同级别的模型组合，并监控模型在生产中的性能。

+   Gemini for Google Cloud 提供了生成式 AI 能力，这些能力对于在 Google Cloud 中的模型开发和部署任务非常有帮助。
