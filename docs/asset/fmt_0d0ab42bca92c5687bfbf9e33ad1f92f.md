
一旦你在 Cloud Shell 编辑器中启用了 Gemini 代码助手，就在 Cloud Shell 编辑器中打开一个新的 Python 文件，并输入`get_pipeline_config`函数的签名和介绍性注释，如下所示。

列表 11.9 `get_pipeline_config` 的签名

```py
def get_pipeline_config(path_to_yaml):
    '''ingest the config yaml file
    Args:
        path_to_yaml: yaml file containing parameters for the pipeline script

    Returns:
        config: dictionary containing parameters read from the config file
    '''
```

注意，此代码片段不包括函数的逻辑。

要让 Gemini for Google Cloud 为此函数生成代码以完成它，只需按 Enter 键。如图 11.39 所示，Gemini for Google Cloud 以斜体生成临时代码。

![](img/CH11_F39_Ryan2.png)

图 11.39 Gemini for Google Cloud 生成的第一组临时代码

按 Tab 键接受此临时代码，然后再次按 Enter 键以获取下一组生成的代码，如图 11.40 所示。

![](img/CH11_F40_Ryan2.png)

图 11.40 Gemini for Google Cloud 生成的第二组临时代码

再次按 Tab 键以接受这组第二组临时代码。结果函数如下所示。

列表 11.10 `get_pipeline_config` 函数

```py
def get_pipeline_config(path_to_yaml):
    '''ingest the config yaml file
    Args:
        path_to_yaml: yaml file containing parameters for the pipeline script

    Returns:
        config: dictionary containing parameters read from the config file
    '''
    with open(path_to_yaml) as file:                     ①
        config = yaml.safe_load(file)

    return config                                        ②
```

① Gemini for Google Cloud 生成的第一组代码

② Gemini for Google Cloud 生成的第二组代码

列表 11.10 中的代码与`get_pipeline_config`函数的手写代码不完全相同，如下所示。

列表 11.11 `get_pipeline_config` 函数：手写版本

```py
def get_pipeline_config(path_to_yaml):
    '''ingest the config yaml file
    Args:
        path_to_yaml: yaml file containing parameters for the pipeline script

    Returns:
        config: dictionary containing parameters read from the config file
    '''
    print("path_to_yaml "+path_to_yaml)
    try:                                                  ①
        with open (path_to_yaml, 'r') as c_file:          ②
            config = yaml.safe_load(c_file)
    except Exception as e:
        print('Error reading the config file')
    return config
```

① 手写代码包括文件打开操作的异常处理

② 手写代码包括文件打开时的 'r' 参数

将列表 11.10 中 Gemini for Google Cloud 生成的代码与列表 11.11 中的手写代码进行比较，我们可以看到两个差异：

+   手写代码包括异常处理以处理打开配置文件时的问题。

+   手写代码在文件打开操作中包括`'r'`参数。

`get_pipeline_config`函数很简单，但无论如何，Gemini for Google Cloud 都能够为该函数生成可工作的代码。

对于 Gemini for Google Cloud 代码生成的额外考虑

+   你不必一次性接受 Gemini for Google Cloud 生成的所有临时代码。要逐个接受临时代码标记，请按 CTRL + 右箭头以接受单个标记。

+   要拒绝整个临时代码生成并重新开始，请按 ESC 键，整个临时代码集将被清除。

+   当您多次使用完全相同的输入请求 Gemini for Google Cloud 生成代码时，并不能保证生成的代码完全相同。例如，在`get_pipeline_config`示例中，有时 Gemini for Google Cloud 会分两步生成函数，如图中所示，有时它会一次性生成整个函数，包括`return`语句。

现在我们已经使用生成式 AI 生成代码，在下一节中，我们将看到如何用它来解释代码。

### 11.4.3 使用 Gemini for Google Cloud 解释 ML 管道代码

现在我们已经看到了 Gemini for Google Cloud 生成代码的例子，让我们来练习它解释代码的能力。

要让 Gemini for Google Cloud 解释一段代码片段，请将以下列表中的代码（管道脚本的`main`函数）复制到 Cloud Shell 编辑器中的新文件中。

列表 11.12 `get_pipeline_config` 函数

```py
if __name__ == '__main__':
    start_time = time.time()
    # load pipeline config parameters
    config = get_pipeline_config('pipeline_config.yml')
    # all the arguments sent to the training 
    #script run in the container are sent via
    # a yaml file in Cloud Storage whose URI is the single argument sent
    model_args = ['--config_bucket', config['config_bucket_path']]
    print("model_args: ",model_args)
    # create a CustomTrainingJob object
    job = create_job(config)
    # define TabularDataset object to use in running CustomTrainingJob
    dataset_path = \
'projects/'+config['project_id']+\
'/locations/'+config['region']+\
'/datasets/'+config['dataset_id']
    ds = aiplatform.TabularDataset(dataset_path)
    # run the CustomTrainingJob object to get a trained model
    model = run_job(job, ds, model_args,config)
    print("deployment starting")
    # deploy model to a Vertex AI endpoint
    if config['deploy_model']:
        deploy_model(model,config)
    print("pipeline completed")
    # show time taken by script
    print("--- %s seconds ---" % (time.time() - start_time))
```

一旦您将列表 11.13 中的代码粘贴到文件中，请选择它，然后从 Cloud Shell 编辑器工具栏中选择 Gemini for Google Cloud Smart Actions 图标（见图 11.41）。

![图片](img/CH11_F41_Ryan2.png)

图 11.41 Gemini for Google Cloud Smart Actions 图标

在出现的菜单中，选择“解释”，如图 11.42 所示。

![图片](img/CH11_F42_Ryan2.png)

图 11.42 Gemini for Google Cloud Smart Actions 图标

当您选择此选项时，代码的解释将出现在 Cloud Shell 编辑器的左侧面板中，如图 11.43 所示。

![图片](img/CH11_F43_Ryan2.png)

图 11.43 代码解释

Gemini for Google Cloud 的代码解释能力可以应用于各种代码，包括 Python、Java 和 JavaScript 代码。您可以使用代码解释来理解不熟悉的代码，并为您的代码推荐文档。

到目前为止，在本节中，我们已经看到了如何使用 Gemini for Google Cloud 中的生成式 AI 功能来回答问题、生成代码和解释代码。在下一小节中，我们将看到如何使用 Gemini for Google Cloud 来帮助总结日志条目。

### 11.4.4 使用 Gemini for Google Cloud 总结日志条目

Google Cloud 包含一个日志，您可以使用它来跟踪环境的行为并调试问题。然而，有时日志条目可能难以解释。Gemini for Google Cloud 可以通过总结来帮助您理解日志条目的要点。在本小节中，我们将介绍如何使用 Gemini for Google Cloud 充分利用 Google Cloud 日志。

为了练习 Gemini for Google Cloud 的这一功能，我们将尝试使用 Vertex AI 中的基础模型调优。基础模型调优允许我们使用 JSONL（JSON Lines：[`jsonlines.org/`](https://jsonlines.org/)）数据集中的数据集对预训练模型进行调优。有关在 Vertex AI 中调优文本模型的更多详细信息，请参阅文档：[`mng.bz/5goO`](https://mng.bz/5goO)。

为了准备本节中的示例，在您在第十章中创建的云存储桶中创建一个名为`staging`的新文件夹。

在 Google Cloud 控制台的 Vertex AI 中，选择 Vertex AI Studio -> 语言。在语言页面上，选择调整和精炼，然后选择创建调整模型，如图 11.44 所示。

![图片](img/CH11_F44_Ryan2.png)

图 11.44 Vertex AI Studio 语言页面

在创建调整模型页面的调整方法面板中：

+   在“调整模型名称”字段中指定您的模型名称。

+   在输出目录字段中指定您在本节开头创建的存档文件夹的 URI。

+   点击“继续”。

见图 11.45。

![图片](img/CH11_F45_Ryan2.png)

图 11.45 创建调整模型页面的调整方法面板

在创建调整模型页面的调整数据集面板中，执行以下操作：

+   在云存储中选择“现有文件”。

+   在“云存储文件路径”字段中输入此样本 JSONL 文件的 URI`cloud-samples-data/vertex-ai/model-evaluation/peft_train_sample.jsonl`。有关 JSONL 样本的详细信息，请参阅文档：[`mng.bz/6ene`](https://mng.bz/6ene)。

+   点击“开始调整”。

见图 11.46。

![图片](img/CH11_F46_Ryan2.png)

图 11.46 创建调整模型页面中的调整数据集面板

一旦点击“开始调整”，你将看到一系列调整后的模型列表，其中你的模型状态显示为“运行中”，如图 11.47 所示。

![图片](img/CH11_F47_Ryan2.png)

图 11.47 调整作业状态

当调整作业完成时，状态将更改为“成功”，如图 11.48 所示。

![图片](img/CH11_F48_Ryan2.png)

图 11.48 显示完成调整作业状态的调整作业状态

如果调整作业没有成功，那没关系。这个特定练习的目标是检查一个错误，所以如果由于某些原因操作失败，那是可以的。

调整作业完成后，在控制台顶部的搜索字段中输入“logs explorer”以进入日志资源管理器页面。此页面提供了检查由 Google Cloud 生成的日志的许多选项。现在，我们只想查看一个错误。要查看错误，请如图 11.49 所示，在日志资源管理器页面的左下角选择“错误”。

![图片](img/CH11_F49_Ryan2.png)

图 11.49 显示完成调整作业状态的调整作业状态

页面底部的查询结果面板显示了错误，如图 11.50 所示。

![图片](img/CH11_F50_Ryan2.png)

图 11.50 显示错误的查询结果面板

选择这些错误条目之一以展开它，并点击如图 11.51 所示的“解释此日志条目”。

![图片](img/CH11_F51_Ryan2.png)

图 11.51 展开的错误条目

在右侧，Gemini for Google Cloud 显示了错误解释，如图 11.52 所示。

![图片](img/CH11_F52_Ryan2.png)

图 11.52 错误解释

Gemini 为 Google Cloud 提供的解释总结了日志中的嵌套条目，使其更容易阅读和解释。请注意，您将看到的解释将取决于您从日志中选择的错误。

### 11.4.5 在 Vertex AI 中调整基础模型

在前一小节中，我们看到了如何使用 Gemini for Google Cloud 的生成式 AI 功能来解释错误日志。值得仔细看看我们触发的操作，以生成我们可以用 Gemini for Google Cloud 检查的日志。以下是我们的操作总结：

+   我们从 Vertex AI 中可用的基础模型之一`text-bison`开始。此模型旨在处理各种自然语言任务，如内容创作和分类。有关`text-bison`的更多详细信息，请参阅文档：[`mng.bz/oKrZ`](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text)。

+   我们对`text-bison`基础模型进行了监督式调整，以适应特定的用例——我们的用例是分类医学转录。要了解更多关于 Vertex AI 中基础模型的监督式调整信息，请参阅文档：[`mng.bz/nR15`](https://mng.bz/nR15)。

+   我们用于调整的数据集包含与转录分类配对的医学诊断转录，如下所示。

列表 11.13 调整数据集的示例记录

```py
{
"input_text":"TRANSCRIPT: \nREASON FOR CONSULTATION: , 
Loculated left effusion, 
multilobar pneumonia.\n\n LABEL:",                      ①
"output_text":"Consult - History and Phy."              ②
}
```

① 医学转录

② 分类

+   此数据集的 URI 为`gs://cloud-samples-data/vertex-ai/model-evaluation/peft_train_sample.jsonl`。

调整过程完成后，您可以在 Vertex AI Studio 中通过选择语言 -> 调整和精炼，然后选择之前小节中调整的模型的行中的测试来测试调整后的模型，如图 11.53 所示。

![图片](img/CH11_F53_Ryan2.png)

图 11.53 在 Vertex AI Studio 中选择调整后的模型

提示编辑器以选定的调整后的模型作为模型打开，如图 11.54 所示。

![图片](img/CH11_F54_Ryan2.png)

图 11.54 选择调整后的模型的提示编辑器

通过在提示字段中输入以下文本并点击提交来训练调整后的模型：

```py
TRANSCRIPT: \nIMPRESSION:  ,EEG during wakefulness, 
drowsiness, and sleep with synchronous 
video monitoring demonstrated no evidence 
of focal or epileptogenic activity.\n\n LABEL:
```

注意响应，如图 11.55 所示。

![图片](img/CH11_F55_Ryan2.png)

图 11.55 睡眠医学响应

现在将模型改回基础模型`text-bison@001`，如图 11.56 所示，然后再次点击提交。

![图片](img/CH11_F56_Ryan2.png)
