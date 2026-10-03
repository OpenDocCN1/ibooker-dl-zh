

# 9 使用 AI 从大型数据集中提取价值

本章涵盖了

+ 使用 Amazon Comprehend 进行命名实体识别（NER）
+ 理解 Comprehend 的操作模式（异步、批量、同步）
+ 使用异步 Comprehend 服务
+ 使用 S3 通知触发 Lambda 函数
+ 使用死信队列处理 Lambda 中的错误
+ 处理 Comprehend 的结果

第八章处理了从网站收集非结构化数据用于机器学习分析的问题。本章在第八章的无服务器网页爬虫基础上进行构建。这次，我们关注的是使用机器学习从我们收集的数据中提取有意义的见解。如果你没有完成第八章的内容，你应该现在回去完成它，然后再继续本章，因为我们将在网页爬虫的基础上直接构建。如果你已经熟悉了这些内容，我们可以直接进入并添加信息提取部分。

## 9.1 使用 AI 从网页中提取重要信息

让我们回顾一下第八章场景的宏伟愿景——寻找要参加的相关开发者会议。我们希望建立一个系统，让人们能够搜索他们感兴趣的和会议演讲者。在第八章的网页爬虫中，我们构建了一个系统，解决了这个场景的第一部分——收集会议数据。

然而，我们不想让用户手动搜索我们收集的所有非结构化网站文本。相反，我们希望向他们展示会议、事件地点和日期以及可能在这些会议上发表演讲的人名单。

从非结构化文本中提取这些有意义的资料是一个非同小可的问题——至少，在最近管理人工智能服务进步之前是这样的。

让我们再次回顾第八章场景的需求概述图。这次，我们突出显示本章的相关部分。

![图片](img/CH09_F01_Elger.png)

图 9.1 本章处理从我们已收集的数据中提取事件和演讲者信息。

### 9.1.1 理解问题

从非结构化文本中提取重要信息的问题被称为 *命名实体识别 (NER)*。一个 *命名实体* 可以是人、地点或组织。它也可以指日期和数值。NER 是一个具有挑战性的问题，也是许多研究的话题。这绝对不是一个完全解决的问题。由于其结果不能保证达到 100% 的准确性，我们必须考虑这一点。根据应用情况，可能需要人工结果检查。例如，假设你有一个需要检测文本中位置的系统。现在假设文本中的一句话提到了阿波罗 11 号指令舱，"哥伦比亚"。NER 可能将其识别为地点而不是航天器的一部分！每个命名实体识别系统都会为每个识别结果提供一个可能性分数，而这个值永远不会达到 100%。

对于我们的会议事件信息提取场景，我们的目标是从网站数据中提取人名、地点和日期。然后这些信息将被存储并供用户访问。

### 9.1.2 扩展架构

我们即将设计和部署一个无服务器系统，从会议网页中提取所需信息。让我们看看本章的架构组件，使用第一章中提到的标准无服务器架构中的类别。这如图 9.2 所示。

![](img/CH09_F02_Elger.png)

图 9.2 服务器端无实体提取系统架构。系统由使用步骤函数编排的同步 Lambda 函数组成。数据存储在第八章中介绍的项存储 S3 存储桶中。S3 的存储桶通知触发我们的异步服务。

与前几章相比，服务和通信渠道的种类较少。从 AWS 服务角度来看，本章将相对简单。新引入的方面包括 Amazon Comprehend 功能和 S3 事件通知作为数据处理触发器。

## 9.2 理解 Comprehend 的实体识别 API

Amazon Comprehend 支持多种实体识别接口。在我们进一步详细说明数据如何通过系统流动之前，让我们花些时间来了解 Comprehend 的工作原理以及它可能对我们架构产生的影响。

Amazon Comprehend 中的三个实体识别接口在表 9.1 中概述。

表 9.1 Amazon Comprehend 操作模式

| API | 描述 | 限制 |
| --- | --- | --- |
| 按需实体识别 | 分析单个文本片段。结果同步返回。 | 仅支持最多 5,000 个字符。 |
| 批量实体识别 | 分析多个文本片段。结果同步返回。 | 每个文档最多 25 个，每个文档最多 5,000 个字符。 |
| 异步实体识别 | 多个大型文本被分析。文本从 S3 读取，结果异步写入 S3。 | 每秒仅一个请求，每份文档 100 KB，所有文档最大 5 GB。 |

有关 Comprehend 限制的完整详细信息，请参阅亚马逊 Comprehend 的 *指南和限制* 文档。1

为了我们的目的，我们希望分析超过 5,000 个字符的文档，因此我们必须选择异步操作模式。此模式要求我们使用两个 API：*StartEntitiesDetectionJob* 用于启动分析，以及 *DescribeEntitiesDetectionJob* 如果我们希望轮询作业状态。

Comprehend 以数组的形式返回实体识别结果。每个数组元素包含以下属性：

`Type`--已识别的实体类型：`PERSON`、`LOCATION`、`ORGANIZATION`、`COMMERCIAL_ITEM`、`EVENT`、`DATE`、`QUANTITY`、`TITLE` 或 `OTHER`。

+ `Score`--分析结果中的置信度分数。这是一个介于 `0` 和 `1` 之间的值。
+ `Text`--已识别的实体文本。
+ `BeginOffset`--实体在文本中的起始偏移量。
+ `EndOffset`--实体在文本中的结束偏移量。

为了了解 Comprehend 的工作方式，让我们使用 AWS 命令行界面运行一次性的测试。使用 shell 是熟悉任何新 AWS 服务的一种有用方式。

提示：第二章和附录 A 介绍了 AWS CLI。除了正常的 AWS CLI 外，亚马逊还发布了一个名为 *AWS Shell* 的交互式版本([`github.com/awslabs/aws-shell`](https://github.com/awslabs/aws-shell))。它支持交互式帮助和命令自动完成。如果您使用 AWS CLI 来学习和探索新服务，查看 AWS Shell 是值得的。

我们将分析代码仓库中 `chapter8-9/sample-text/apollo.txt` 下的样本文本。文本段落取自维基百科上的 *阿波罗 11 号* 页面。2 样本文本将在下一列表中展示。

列表 9.1 实体识别样本文本：`apollo.txt`

```
Apollo 11 was the spaceflight that first landed humans on the Moon. 
Commander Neil Armstrong and lunar module pilot Buzz Aldrin formed the 
American crew that landed the Apollo Lunar Module Eagle on July 20, 1969, 
at 20:17 UTC. Armstrong became the first person to step onto the lunar 
surface six hours and 39 minutes later on July 21 at 02:56 UTC; Aldrin 
joined him 19 minutes later. They spent about two and a quarter hours 
together outside the spacecraft, and they collected 47.5 pounds (21.5 kg) 
of lunar material to bring back to Earth. Command module pilot Michael 
Collins flew the command module Columbia alone in lunar orbit while they 
were on the Moon's surface. Armstrong and Aldrin spent 21 hours 31 minutes 
on the lunar surface at a site they named Tranquility Base before lifting 
off to rejoin Columbia in lunar orbit.
```

我们可以使用以下命令在 CLI 上运行按需实体识别：

```
export INPUT_TEXT=`cat apollo.txt`

aws comprehend detect-entities --language-code=en --text $INPUT_TEXT > results.json
```

此命令的输出，保存到 `results.json`，表明 Comprehend 如何为实体识别任务提供分析结果。表 9.2 以表格格式显示了此命令的一些结果。

表 9.2 Comprehend 实体识别样本结果

| 类型 | 文本 | 分数 | 起始偏移量 | 结束偏移量 |
| --- | --- | --- | --- | --- |
| `ORGANIZATION` | 阿波罗 11 号 | 0.49757930636405900 | 0 | 9 |
| `LOCATION` | 月球 | 0.9277622103691100 | 62 | 66 |
| `PERSON` | 尼尔·阿姆斯特朗 | 0.9994082450866700 | 78 | 92 |
| `PERSON` | 巴兹·奥尔德林 | 0.9906044602394100 | 116 | 127 |
| `OTHER` | 美国 | 0.6279735565185550 | 139 | 147 |
| `ORGANIZATION` | 阿波罗 | 0.23635128140449500 | 169 | 175 |
| `COMMERCIAL_ITEM` | 月球模块鹰 | 0.7624998688697820 | 176 | 194 |
| `DATE` | "1969 年 7 月 20 日" | 0.9936476945877080 | 198 | 211 |
| `QUANTITY` | 第一人称 | 0.8917713761329650 | 248 | 260 |
| `QUANTITY` | 大约两个半小时 | 0.9333438873291020 | 395 | 424 |
| `QUANTITY` | 21.5 公斤 | 0.995818555355072 | 490 | 497 |
| `LOCATION` | 地球 | 0.9848601222038270 | 534 | 539 |
| `PERSON` | 迈克尔·柯林斯 | 0.9996771812438970 | 562 | 577 |
| `LOCATION` | 哥伦比亚 | 0.9617793560028080 | 602 | 610 |

很明显，通过 Comprehend 实体识别可以以非常小的努力获得非常准确的结果。

为了从会议网站爬取的每个网页中获得这些结果，我们将使用异步实体识别 API。这意味着我们必须处理以下 API 的特性：

+ 在 Comprehend 上进行的实体识别作业在异步模式下运行时间较长。每个作业可能需要 5 到 10 分钟。这比同步作业要长得多，但权衡是异步作业可以处理更大的文档。
+ 为了避免触碰到 API 的限速限制，我们将避免每秒发送超过一个请求，并将多个网页提交给每个任务。
+ 亚马逊 Comprehend 中的异步 API 将结果写入配置的 S3 存储桶。我们将通过 S3 存储桶上的通知触发器来处理结果。

第八章中的网络爬虫为每个网页在 S3 存储桶中写了一个文本文件（`page.txt`）。为了开始实体识别，我们将在 S3 的单独的临时文件夹中创建这个文件的副本。这样，我们可以检查临时文件夹中的新文本文件以供处理。当处理开始时，我们将从临时区域删除文件。原始文件（`page.txt`）将永久保留在`sites`文件夹中，因此如果以后需要，它仍然可用于进一步处理。

让我们继续实现一个简单的服务，该服务将在临时区域创建文本文件的副本。

## 9.3 准备信息提取数据

包含准备处理文件的临时区域将是项目存储 S3 存储桶中的一个目录，称为`incoming-texts`。我们将使用 S3 通知触发器来响应从网络爬虫到达存储桶中的新`page.txt`文件。然后，每个文件将被复制到`incoming-texts/`。

### 9.3.1 获取代码

准备服务的代码位于`chapter8-9/preparation-service`目录中。此目录包含一个`serverless.yml`文件。在部署和测试准备服务之前，我们将逐步解释其内容。

### 9.3.2 创建 S3 事件通知

准备服务主要由一个具有事件通知的简单函数组成。让我们详细探索`serverless.yml`，看看它是如何工作的。接下来的列表显示了该文件的摘录，其中我们遇到了第一个 S3 存储桶事件通知。

列表 9.2 准备服务 `serverless.yml` 文件摘录

```
service: preparation-service
frameworkVersion: '>=1.47.0'                       ❶

plugins:
  - serverless-prune-plugin                        ❷
  - serverless-pseudo-parameters                   ❸
  - serverless-dotenv-plugin                       ❹
...

provider:
  ...
  iamRoleStatements:
    - Effect: Allow
      Action:
        - s3:GetObject
        - s3:PutObject
        - s3:ListBucket                             ❺

      Resource:
        - arn:aws:s3:::${env:ITEM_STORE_BUCKET}/*
...
functions:
  prepare:
    handler: handler.prepare                        ❻
    events:
      - s3:
          bucket: ${env:ITEM_STORE_BUCKET}
          event: s3:ObjectCreated:*
          rules:
            - suffix: page.txt                      ❼
          existing: true                            ❽
```

❶ 要使用 S3 现有的存储桶作为事件触发器，您必须使用 Serverless Framework 1.47.0 或更高版本。3 这一行强制执行了该要求。

❷ 每次我们部署时，都会为每个 Lambda 函数创建一个新的版本。serverless-prune-plugin 负责在版本积累时移除旧的 Lambda 函数版本。4

❸ 我们希望在配置中使用带有伪参数（如${AWS::AccountId}）的 CloudFormation 子函数，5，但此语法与 Serverless Framework 的变量语法冲突。6 serverless-pseudo-parameters7 通过允许我们使用更简单的语法（#{AWS::AccountId}）来解决此问题。

❹ 正如前几章所述，我们使用 serverless-dotenv-plugin8 从.env 文件中加载环境变量。

❺ 我们为函数授予读取和写入项目存储桶的权限。

❻ S3 事件处理函数在 handler.js 中定义。函数名称被导出为 prepare。

❼ Lambda 触发器被定义为 S3 通知。通知将匹配在桶中创建的任何带有 page.txt 后缀的对象（文件）。

❽ 这确保了 Serverless Framework 不会尝试创建桶。相反，它将在现有的项目存储桶上创建一个通知触发器。

我们刚刚声明了函数、其资源以及准备处理器的触发器。现在我们可以继续实现这个函数。

CloudFormation 和 S3 通知触发器

CloudFormation 是一种定义基础设施即代码的绝佳方式，它支持逻辑上分组资源，并在任何失败事件发生时提供回滚功能。然而，一个缺点是，与 AWS SDK 创建所有资源类型相比，CloudFormation 不够灵活。

这的一个例子是与桶通知一起使用。使用 CloudFormation，只有在创建桶资源时才能添加通知。9 我们更希望能够在我们的系统中为任何服务添加现有桶的通知。

Serverless Framework 为这个问题提供了一个很好的解决方案。通过使用具有属性`existing:` `true`的`s3`事件类型，框架在底层使用 AWS SDK 向现有桶添加一个新的通知。这是通过使用*CloudFormation 自定义资源*实现的，这是一种有用的解决方案，当官方 CloudFormation 支持不足以满足您的需求时。有关自定义资源的更多信息，请参阅 AWS 文档。10

### 9.3.3 实现准备处理程序

准备服务`handler`模块的目标是执行任何必要的处理，以便文本准备好进行实体识别。在我们的案例中，这仅仅是把文本放入正确的文件夹，并使用正确的文件名进行处理。准备服务的`handler`模块如下所示。

列表 9.3 准备服务`handler.js`摘录

