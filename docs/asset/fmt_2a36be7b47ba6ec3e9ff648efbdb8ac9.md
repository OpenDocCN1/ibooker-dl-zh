
# 服务配置

```
service: resources                                             ❶
 frameworkVersion: ">=1.30.0"
custom:                                                        ❷
   bucket: ${env:CHAPTER2_BUCKET}
  crawlerqueue: Chap2CrawlerQueue
  analysisqueue: Chap2AnalysisQueue
  region: ${env:AWS_DEFAULT_REGION, 'eu-west-1'}
  accountid: ${env:AWS_ACCOUNT_ID}

provider:                                                      ❸
   name: aws
  runtime: nodejs12.x
  stage: dev
  region: ${env:AWS_DEFAULT_REGION, 'eu-west-1'}

resources:
  Resources:
    WebAppS3Bucket:                                            ❹
       Type: AWS::S3::Bucket
      Properties:
        BucketName: ${self:custom.bucket}
        AccessControl: PublicRead
        WebsiteConfiguration:
          IndexDocument: index.html
          ErrorDocument: index.html
    WebAppS3BucketPolicy:                                      ❺
       Type: AWS::S3::BucketPolicy
      Properties:
        Bucket:
          Ref: WebAppS3Bucket
        PolicyDocument:
          Statement:
            - Sid: PublicReadGetObject
              Effect: Allow
              Principal: "*"
              Action:
                - s3:GetObject
              Resource: arn:aws:s3:::${self:custom.bucket}/*
    Chap2CrawlerQueue:                                         ❻
       Type: "AWS::SQS::Queue"
      Properties:
        QueueName: "${self:custom.crawlerqueue}"
    Chap2AnalysisQueue:
      Type: "AWS::SQS::Queue"
      Properties:
        QueueName: "${self:custom.analysisqueue}"
```

❶ 服务名称

❷ 自定义定义

❸ 特定提供者

❹ 存储桶定义

❺ 存储桶策略

❻ 队列定义

提示：Serverless 使用*YAML*文件格式进行配置。YAML 代表*YAML Ain’t Markup Language*；您可以在本网站上找到有关 YAML 的更多信息：[`yaml.org/`](http://yaml.org/)。

如果一开始看起来令人不知所措，请不要担心。我们将在这本书中一直使用 Serverless Framework，因此这些配置文件将变得非常熟悉。让我们看看这个文件的总体结构。

提示：Serverless Framework 及其配置的完整文档可以在项目的主要网站上找到：[`serverless.com/framework/docs/`](https://serverless.com/framework/docs/)。

服务器无配置被分解为几个顶级部分。其中关键的部分是

+   `custom`--定义在配置中其他地方使用的属性。

+   `provider`--定义框架中特定提供者的配置。在本例中，我们使用 AWS 作为提供者；然而，该框架支持多个云平台

+   `functions`--定义服务实现的功能端点。在本例中，我们没有要定义的功能，因此本例中不存在此部分。

+   `resources`--定义云平台上的支持资源。在本例中，我们定义了两个 SQS 队列和一个 S3 存储桶。当我们部署此配置时，Serverless Framework 将为我们创建队列和存储桶。

注意：还有许多其他工具可以用来部署云资源，例如 AWS CloudFormation 或 Hashicorp 的 Terraform，这两者都是管理基础设施代码的出色工具。如果你有一个基础设施密集型项目，我们建议调查这些工具。对于这本书，我们将几乎完全使用无服务器框架。我们还注意到，无服务器框架在 AWS 底层使用 CloudFormation；我们将在附录 E 中更详细地介绍这一点。

在我们继续部署资源之前，我们需要确定一个存储桶名称。AWS 存储桶名称空间是全球性的，因此你应该选择一个可用的名称，并在你的 shell 中添加一个额外的环境变量`CHAPTER2_BUCKET`，就像我们设置 AWS 环境变量一样：

```
export CHAPTER2_BUCKET=<YOUR BUCKET NAME>
```

将`<YOUR BUCKET NAME>`替换为你选择的唯一名称。现在我们已经准备好了，让我们继续部署资源。在`chapter2-3/resources`目录的命令行中运行

```
$ serverless deploy
```

无服务器将自动部署我们的资源，你应该会看到类似以下列表的输出。

**列表 2.3 无服务器部署输出**

```
Serverless: Packaging service...
Serverless: Creating Stack...
Serverless: Checking Stack create progress...
.....
Serverless: Stack create finished...
Serverless: Uploading CloudFormation file to S3...
Serverless: Uploading artifacts...
Serverless: Validating template...
Serverless: Updating Stack...
Serverless: Checking Stack update progress...
..............
Serverless: Stack update finished...
Service Information
service: resources
stage: dev
region: eu-west-1
stack: resources-dev
api keys:
  None
endpoints:
  None
functions:
  None
```

无服务器已经为我们创建了一个 S3 存储桶和一个 SQS 队列。现在我们已经有了支持的基础设施，我们可以继续进行实际实现了！

## 2.4 实现异步服务

在完成基本设置后，我们可以继续编写我们的第一个服务。在本节中，我们将组合爬虫和分析异步服务，并在隔离状态下测试它们。

### 2.4.1 爬虫服务

首先，让我们看看`crawler-service`的代码。图 2.8 展示了该服务内部的流程。

![图片](img/CH02_F08_Elger.png)

**图 2.8 爬虫服务**

当消息被放置在`crawler`队列中时，会调用`crawler-service`。消息包含服务要爬取的目标 URL。一旦被调用，爬虫会抓取指定 URL 的 HTML 页面，并解析出图像标签。然后，对于每个图像依次，它会将图像下载到 S3 文件夹中。最后，一旦所有图像都下载完毕，它会向`analysis`队列发送一个`analyze`消息，包括分析 URL 的域名以进行进一步处理。

爬虫服务的代码位于`chapter2-3/crawler-service`。进入这个目录，你应该会看到以下列出的文件：

```
handler.js
images.js
package.json
serverless.yml
```

为了了解此服务使用的资源及其整体结构，我们首先应该查看文件`serverless.yml`，其中包含下一列表中显示的配置。

**列表 2.4 爬虫服务的`serverless.yml`**

```
service: crawler-service
frameworkVersion: ">=1.30.0"
custom:
  bucket: ${env:CHAPTER2_BUCKET}                              ❶
  crawlerqueue: Chap2CrawlerQueue                             ❷
  analysisqueue: Chap2AnalysisQueue
  region: ${env:AWS_DEFAULT_REGION, 'eu-west-1'}
  accountid: ${env:AWS_ACCOUNT_ID}                            ❸

provider:
  name: aws
  runtime: nodejs12.x
  stage: dev
  region: ${env:AWS_DEFAULT_REGION, 'eu-west-1'}
  iamRoleStatements:
    - Effect: Allow                                           ❹
      Action:
        - s3:PutObject
      Resource: "arn:aws:s3:::${self:custom.bucket}/*"
    - Effect: Allow
      Action:
        - sqs:ListQueues
      Resource: "arn:aws:sqs:${self:provider.region}:*:*"
    - Effect: Allow                                           ❺
      Action:
        - sqs:ReceiveMessage
        - sqs:DeleteMessage
        - sqs:GetQueueUrl
      Resource: "arn:aws:sqs:*:*:${self:custom.crawlerqueue}"
    - Effect: Allow                                           ❻
      Action:
        - sqs:SendMessage
        - sqs:DeleteMessage
        - sqs:GetQueueUrl
      Resource: "arn:aws:sqs:*:*:${self:custom.analysisqueue}"

functions:
  crawlImages:                                                ❼
    handler: handler.crawlImages
    environment:
      BUCKET: ${self:custom.bucket}
      ANALYSIS_QUEUE: ${self:custom.analysisqueue}
      REGION: ${self:custom.region}
      ACCOUNTID: ${self:custom.accountid}
    events:
      - sqs:                                                  ❽
          arn: "arn:aws:sqs:${self:provider.region}:${env:AWS_ACCOUNT_ID} \
          :${self:custom.crawlerqueue}"
```

❶ S3 存储桶名称

❷ SQS 队列名称

❸ 本地环境中的账户 ID

❹ S3 权限

❺ 允许从爬虫队列接收

❻ 允许向分析队列发送

❼ 定义处理函数的入口点

❽ 由爬虫队列触发的函数

此配置的效果是定义并部署我们的爬虫服务函数到 AWS，并允许它通过资源配置中部署的爬虫 SQS 队列触发。关键部分如下

+   `custom`--定义在配置中其他地方使用的属性。

+   `provider`--在此配置中的提供者部分设置 AWS 权限，允许服务访问 SQS 队列，并授予它写入我们 S3 存储桶的权限。

+   `functions`--此部分定义了服务 Lambda。处理器设置引用了实现，我们将在稍后查看。事件条目将函数连接到我们之前部署的 SQS 爬虫队列。最后，环境块定义了将可用于我们的函数的环境变量。

注意：`iamRoleStatements`块中定义的权限直接映射到 AWS 身份和访问管理（IAM）模型。关于这方面的完整文档可以在 AWS 上找到，网址为[`aws.amazon.com/iam`](https://aws.amazon.com/iam)。

与我们资源之前的`serverless.yml`文件不同，此文件没有定义任何资源。这是因为我们选择在服务范围之外定义我们的资源。一般来说，一个很好的经验法则是，全局或共享资源应该部署在公共资源堆栈中；用于单个服务的资源应该与该特定服务一起部署。

在无服务器 YAML 文件中的“资源”部分定义了在部署时将创建的资源。依赖于此资源的其他服务必须在资源创建后部署。我们发现将全局资源放在单独的配置中是最好的做法。

让我们现在看看爬虫的主要实现文件`handler.js`。在文件顶部，我们包含了一些模块，如下所示。

**列表 2.5 爬虫 handler.js 所需模块**

```
const request = require('request') 1((CO3-1))          ❶
const urlParser = require('url')                       ❷
const AWS = require('aws-sdk')                         ❸
const s3 = new AWS.S3()
const sqs = new AWS.SQS({region: process.env.REGION})
const images = require('./images')()                   ❹
```

❶ request 是一个实现功能齐全的 HTTP 客户端的 node 模块。

❷ url 是一个核心 node 模块，它理解如何解析 URL。

❸ 包含 AWS SDK 节点模块。在这种情况下，我们实例化了 S3 和 SQS 对象，以便分别与我们的 S3 存储桶和队列进行接口。

❹ ./images 指的是文件 images.js 中的我们自己的模块。

进入此服务的主要入口点是`crawlImages`。此函数接受三个参数：`event`、`context`和`cb`。下面的代码展示了这一部分。

**列表 2.6 爬虫服务入口点**

```
module.exports.crawlImages = function (event, context, cb) {
  asnc.eachSeries(event.Records, (record, asnCb) => {           ❶
    let { body } = record

    try {
      body = JSON.parse(body)
    } catch (exp) {
      return asnCb('message parse error: ' + record)
    }

    if (body.action === 'download' && body.msg && body.msg.url) {
      const udomain = createUniqueDomain(body.msg.url)
      crawl(udomain, body.msg.url, context).then(result => {    ❷
        queueAnalysis(udomain, body.msg.url, context).
        then(result => {                                        ❸
          asnCb(null, result)
        })
      })
    } else {
      asnCb('malformed message')
    }
  }, (err) => {
    if (err) { console.log(err) }
    cb()
  })
}
```

❶ 遍历消息

❷ 爬取图像的 URL。

❸ 向 SQS 发送消息以触发分析

函数接受以下三个参数：

1.  `event`--提供有关正在处理的当前事件的详细信息。在这种情况下，事件对象包含从 SQS 队列中取出的记录数组。

1.  `context`--由 AWS 用于提供调用上下文信息，例如可用内存量、执行时间和客户端调用上下文。

1.  `cb`--回调函数。处理完成后，应由处理程序调用此函数并传递结果。

回调和异步 I/O
