
将`<your bucket name>`替换为您选择的全球唯一存储桶名称。

为了部署 API，我们需要像之前一样使用 Serverless Framework。打开命令行，`cd`到`chapter7/text-analysis/text-analysis-api`目录，并运行：

```
$ npm install
$ serverless deploy
```

这将创建文档图像存储桶，设置 API 网关，并部署我们的两个 Lambda 函数。一旦部署，Serverless 将输出两个函数的网关 URL，这些 URL 将类似于下一条列表中所示。

**列表 7.6 端点 URL**

```
endpoints:
GET - https://63tat1jze6.execute-api.eu-west-1.amazonaws.com/dev/upload    ❶
POST - https://63tat1jze6.execute-api.eu-west-1.amazonaws.com/dev/analyze  ❷
functions:
upload: c7textanalysis-dev-upload
analyze: c7textanalysis-dev-analyze
```

❶ 上传 URL

❷ 分析 URL

我们将使用这些 URL 来调用我们的文本分析 API。

### 7.2.5 测试 API

现在我们已经部署了我们的 API，是时候用一些真实数据来测试它了。我们刚刚部署的服务能够读取和识别文档中的文本字段，例如账单或护照。我们在`data`子目录中提供了一些示例护照图像，其中之一如图 7.9 所示。当然，这些是由模拟数据组成的。

![图片](img/CH07_F09_Elger.png)

图 7.9 示例护照

为了测试 API，我们首先需要更新我们的`.env`文件。在文本编辑器中打开文件，并添加两个 URL 和存储桶名称，如下所示，使用您特定的名称。

**列表 7.7 环境文件**

```
TARGET_REGION=eu-west-1
CHAPTER7_IMAGE_BUCKET=<your bucket name>
CHAPTER7_ANALYZE_URL=<your analyze url>       ❶
CHAPTER7_GETUPLOAD_URL=<your upload url>      ❷
```

❶ 替换为分析 URL

❷ 替换为上传 URL

接下来，`cd`到`chapter7/text-analysis/client`目录。在`data`子目录中有一些示例图像。在`index.js`中有练习客户端的代码。要运行代码，打开命令行并执行：

```
$ npm install
$ node index.js
```

客户端代码将使用 API 将示例文档上传到图像存储桶，然后调用我们的`analyze` API。`analyze` API 将调用 Textract 来分析图像并将结果返回给我们的客户端。最后，客户端代码将解析输出 JSON 结构并挑选出几个关键字段，将它们显示在控制台上。你应该看到类似以下列表的输出。

**列表 7.8 客户端输出**

```
{
  "passportNumber": "340020013 (confidence: 99.8329086303711)",
  "surname": "TRAVELER (confidence: 75.3625717163086)",
  "givenNames": "HAPPY (confidence: 96.09229278564453)",
  "nationality": "UNITED STATES OF AMERICA (confidence: 82.67759704589844)",
  "dob": "01 Jan 1980 (confidence: 88.6818618774414)",
  "placeOfBirth": "WASHINGTON D.C. U.S.A. (confidence: 84.47944641113281)",
  "dateOfIssue": "06 May 2099 (confidence: 88.30438995361328)",
  "dateOfExpiration": "05 May 2019 (confidence: 88.60911560058594)"
}
```

重要的是要注意，Textract 正在应用多种技术来为我们提取这些信息。首先，它执行一个*光学字符识别（OCR）*分析来识别图像中的文本。作为分析的一部分，它保留了识别字符的坐标信息，将它们分组到块和行中。然后，它使用坐标信息将表单字段关联为名称-值对。

为了准确起见，我们需要向 Textract 提供高质量的图像：图像质量越好，分析结果越好。你可以通过创建或下载自己的低质量图像并将这些图像传递给 API 来测试这一点。你应该会发现 Textract 在低质量图像中难以识别相同的字段。

列表 7.8 显示了 Textract 识别的字段，以及一个置信度级别。大多数人工智能服务都会返回一些相关的置信度级别，而作为服务的消费者，我们则需要弄清楚我们应该如何处理这个数字。例如，如果我们的用例对错误非常敏感，那么可能只接受 99% 或更好的置信度级别是正确的。置信度较低的成果应发送给人类进行验证或修正。然而，许多商业用例可以容忍较低的准确性。这种判断非常特定于领域，应涉及业务和技术利益相关者。

考虑你自己的组织中的业务流程：是否有可以通过此类分析自动化的领域？你是否需要从客户提供的文档中收集和输入信息？也许你可以通过调整此示例以满足自己的需求来改进该流程。

### 7.2.6 删除 API

在进入下一节之前，我们需要删除 API 以避免产生额外费用。为此，请使用`cd`命令进入`chapter7/text-analysis/text-analysis-api`目录，并运行：

```
$ source ../.env && aws s3 rm s3://${CHAPTER7_IMAGE_BUCKET} --recursive
$ serverless remove
```

这将删除桶中所有上传的图像并拆除堆栈。

## 7.3 带有 Kinesis 的 AI 驱动数据处理管道

对于我们的第二个示例，我们将构建一个数据处理管道。这个管道将通过异步 API 公开，并作为我们的模式 2 示例。在构建这个示例时，我们将详细探讨包括 Kinesis、Translate 和 Comprehend 在内的一系列新服务和技术的使用：

+   Kinesis 是亚马逊的实时流服务，用于创建数据和处理视频的管道。

+   Translate 是亚马逊的机器驱动语言翻译服务。

+   Comprehend 是亚马逊的自然语言处理（NLP）服务，可用于执行情感分析或关键词检测等任务。

考虑零售和电子商务领域。大型零售店可能有多个产品部门，如“户外”、“汽车”、“宠物”等。客户服务是零售贸易的重要组成部分。特别是，快速有效地回应客户投诉非常重要，因为如果处理得当，它可以把一个不满的客户转变为品牌倡导者。问题是客户有多个渠道可以投诉，包括网站产品评论、电子邮件、Twitter、Facebook、Instagram、博客文章等等。

不仅有许多渠道可以放置产品反馈，全球零售商还必须处理多种语言的反馈。尽管需要人类来处理客户，但检测所有这些渠道和地理区域的负面反馈是适合人工智能驱动解决方案的。

我们的示例系统将是一个具有 AI 功能的管道，可以用于过滤来自多个渠道和多种语言的反馈。我们管道的目标是在检测到关于他们产品之一的负面反馈时，向适当的部门发出警报。

这个具有 AI 功能的管道增强了零售商的数字能力，同时不会直接干扰业务线系统。

我们在图 7.10 中展示了我们的管道。管道的起始处，原始数据被发送到一个收集 API；这些数据可以来自多个来源，例如 Twitter 流、Facebook 评论、入站电子邮件以及其他社交媒体渠道。API 将原始文本输入到 Kinesis 流中。

![](img/CH07_F10_Elger.png)

图 7.10 处理管道

AWS 提供了两种关键的流技术：管理的 Kafka（MSK）和 Kinesis。在这些技术中，Kinesis 是最容易使用的，因此我们将专注于它来构建这个系统。流中的数据会触发下游的 Lambda 函数，该函数使用 Comprehend 来确定入站文本的语言。如果语言不是英语，Lambda 函数将使用 AWS Translate 进行即时翻译，然后再将其发送到管道中。下一个下游的 Lambda 函数将对翻译后的文本进行情感分析，使用 Comprehend。如果检测到积极情感，则不会对消息进行进一步处理。然而，如果情感非常负面，文本将被发送到使用 AWS Comprehend 构建的客户分类器。该分类器分析文本并尝试确定与消息相关的产品部门。一旦确定了部门，消息就可以被发送到适当的团队，以便他们处理负面评论。在这种情况下，我们将输出结果到 S3 存储桶。

通过这种方式结合使用 AI 服务，这样的管道可以为企业节省巨大的成本，因为反馈的过滤和分类是自动完成的，无需团队人员。

**Kinesis 与 Kafka 的比较**

直到最近，选择 Kinesis 而不是 Kafka 的一个原因是 Kafka 需要在 EC2 实例上安装、设置和管理。随着 AWS 管理的 Kafka（MSK）的发布，这种情况已经改变。尽管关于 Kafka 优点的全面讨论超出了本书的范围，但我们想指出，这项技术具有高度可扩展性和多功能性。我们建议，如果您正在构建一个需要大量流处理的系统，您应该更深入地研究 Kafka。

即使考虑到 MSK，Kinesis 仍然更完全地集成到 AWS 堆栈中，并且更容易快速启动，因此我们将使用它作为示例系统。Kinesis 可以用几种方式使用：

+   Kinesis Video Streams——用于视频和音频内容

+   Kinesis Data Streams——用于通用数据流

+   Kinesis Data Firehose——支持将 Kinesis 数据流式传输到 S3、Redshift 或 Elasticsearch 等目标

+   Kinesis Analytics——支持使用 SQL 进行实时流处理

在本章中，我们使用 Kinesis Data Streams 构建我们的管道。

### 7.3.1 获取代码

我们管道的代码位于书籍仓库中`chapter7/pipeline`目录下。该目录包含以下子目录，它们对应于处理过程中的每个阶段：

+   `pipeline-api`——包含系统的 API Gateway 设置

+   `translate`——包含语言检测和翻译服务

+   `sentiment`——包含情感分析代码

+   `training`——包含帮助训练自定义分类器的实用脚本

+   `classify`——包含触发我们的自定义分类器的代码

+   `driver`——包含用于测试管道的代码

与前面的示例一样，在部署之前，我们将简要描述每个服务的代码。一旦我们所有的单元都已部署，我们将对管道进行端到端测试。让我们从简单的第一步开始，部署 API。

### 7.3.2 部署 API

API 的代码位于`chapter7/pipeline/pipeline-api`目录中，包括一个`serverless.yml`文件和一个简单的 API。Serverless 配置定义了一个单一的`ingest`方法，该方法将 API 发布的数据推送到 Kinesis。Kinesis 流也在 Serverless 配置中定义，如下所示。

**列表 7.9 `serverless.yml` Kinesis 定义**

```
resources:
  Resources:
    KinesisStream:                    ❶
      Type: AWS::Kinesis::Stream
      Properties:
        Name: ${env:CHAPTER7_PIPELINE_TRANSLATE_STREAM}
        ShardCount: ${env:CHAPTER7_PIPELINE_SHARD_COUNT}
```

❶ 定义 Kinesis 流

API 的代码非常简单，它只是将传入的数据转发到 Kinesis 流。API 接受传入的 JSON `POST`请求，并期望格式如下一列表所示。

**列表 7.10 管道 API 的 JSON 数据格式**

```
{
  originalText: ...                      ❶
  source: 'twitter' | 'facebook'...      ❷
  originator: '@pelger'                  ❸
}
```

❶ 原始文本

❷ 反馈来源

❸ 反馈发起者的 ID

在部署 API 之前，我们需要设置我们的环境。我们在`chapter7/pipeline`目录中提供了一个名为`default-environment.env`的模板`.env`文件。在`chapter7/pipeline`目录中创建此文件的副本，文件名为`.env`。该文件应包含下一列表中概述的内容。

**列表 7.11 管道的环境文件**

```
TARGET_REGION=eu-west-1
CHAPTER7_PIPELINE_SHARD_COUNT=1                          ❶
CHAPTER7_PIPELINE_TRANSLATE_STREAM=c7ptransstream        ❷
CHAPTER7_PIPELINE_SENTIMENT_STREAM=c7psentstream         ❸
CHAPTER7_PIPELINE_CLASSIFY_STREAM=c7pclassifystream      ❹
CHAPTER7_PIPELINE_TRANSLATE_STREAM_ARN=...
CHAPTER7_PIPELINE_SENTIMENT_STREAM_ARN=...
CHAPTER7_PIPELINE_CLASSIFY_STREAM_ARN=...
CHAPTER7_CLASSIFIER_NAME=chap7classifier
CHAPTER7_CLASSIFIER_ARN=...
...
```

❶ Kinesis 分片数量

❷ Kinesis 翻译流名称

❸ Kinesis 情感流名称

❹ Kinesis 分类流名称

接下来，我们可以在`chapter7/pipeline/pipeline-api`目录中打开命令行，并执行以下操作来部署 API。

```
$ npm install
$ serverless deploy
```

这将创建我们的第一个 Kinesis 流，以及我们的摄取 API。图 7.11 说明了 API 部署后的管道状态。高亮部分表示到目前为止已部署的内容。

![图 7.11](img/CH07_F11_Elger.png)

图 7.11 API 部署后的管道

在部署后，框架将输出我们 API 的 URL。在进入下一阶段之前，请将其添加到`.env`文件中，如下所示，并用您的具体值替换。

**列表 7.12 API 部署后的`.env`文件中的附加条目**

```
CHAPTER7_PIPELINE_API=<your API url>
```

## 7.4 使用 Translate 进行即时翻译

在数据摄取后的我们管道的第一个阶段是检测语言，如果需要则翻译成英语。这些任务由我们的翻译服务处理，其代码位于`chapter8/pipeline/translate`目录中。Serverless 配置相当标准，除了主要处理函数是由我们在 API 部署中定义的 Kinesis 流触发的。这将在下面的列表中展示。

**列表 7.13 由 Kinesis 触发的处理程序**

```
functions:
  translate:
    handler: handler.translate
    events:
      - stream:                 ❶
          type: kinesis
          arn: ${env:CHAPTER7_PIPELINE_TRANSLATE_STREAM_ARN}
          batchSize: 100
          startingPosition: LATEST
          enabled: true
          async: true
```

❶ 连接到流。

该配置定义了一个第二个 Kinesis 流，我们的情感服务将连接到它，并设置了适当的权限来发布到流和调用所需的翻译服务。这将在下面的列表中展示。

**列表 7.14 处理程序 IAM 权限**

```
- Effect: Allow
      Action:
        - comprehend:DetectDominantLanguage      ❶
         - translate:TranslateText               ❷
         - kinesis:PutRecord                     ❸
         - kinesis:PutRecords
      Resource: "*"
```

❶ Comprehend 权限

❷ 翻译权限

❸ Kinesis 权限

我们翻译服务的代码在`handler.js`中，由我们的 API 定义的 Kinesis 流中的数据触发。这作为事件参数中 Base64 编码记录的一个块。下一个列表展示了我们的服务如何消费这些记录。

**列表 7.15 翻译服务**
