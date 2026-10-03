

# 3 构建无服务器图像识别系统，第二部分

本章涵盖了

+ 构建简单的 AI as a Service 系统
+ 消费人工智能图像识别服务
+ 实现同步和异步服务
+ 部署用户界面
+ 部署到云端

在本章中，我们将继续构建我们在第二章开始的服务器端无服务器图像识别系统。我们将添加我们的图像识别服务，该服务将调用 AWS Rekognition 来为我们完成繁重的工作。一旦完成，我们将为我们的系统构建一个简单的前端，以便我们可以测试我们的图像识别能力。

如果你还没有完成第二章的内容，你应该在继续本章之前返回并完成它，因为我们将直接基于那里开始的工作进行构建。如果你对第二章的内容感到满意，我们可以直接从我们离开的地方继续，并部署分析服务。

## 3.1 部署异步服务

在第二章中，我们设置了我们的开发环境并部署了爬虫服务。在本章中，我们将继续部署系统的其余部分，从分析服务开始。

### 3.1.1 分析服务

让我们来看看`analysis-service`。与`crawler-service`类似，当我们的 S3 存储桶中有可供分析图像时，该服务会由 Analysis SQS 队列的消息触发。该服务的逻辑概述如图 3.1 所示。

![](img/CH03_F01_Elger.png)

图 3.1 分析服务

从本质上讲，`analysis-service`在下载的图像和亚马逊 Rekognition 服务之间形成了一座桥梁。`crawler-service`下载的每一张图像都会输入到 Rekognition 中，并返回一组标签。每个标签都是一个描述图像中由模型识别的对象的词，以及一个置信度水平（一个介于 0 和 100 之间的数字，其中 100 表示对图像标签的完全置信）。

在此分析之后，该服务处理返回的数据以创建一组词频，这些词频可以输入到词云生成器中。背后的想法是尝试在给定 URL 上可用的图像之间视觉上确定一个共同的主题。

让我们来看看`analysis-service`中的代码，从`serverless.yml`配置开始，看看这是如何实现的。这将在下一列表中展示。

图 3.1 分析服务 `serverless.yml`

```
service: analysis-service
custom:
  bucket: ${env:CHAPTER2_BUCKET}
  ...

provider:
  ...
  iamRoleStatements:
    - Effect: "Allow"               ❶
      Action:
        - "rekognition:*"
      Resource: "*"
  ...

functions:
  analyzeImages:                    ❷
    handler: handler.analyzeImages
    ...
```

❶ 允许访问 Rekognition API。

❷ 定义主入口点

我们应该注意，这个服务的`serverless.yml`配置文件与之前的非常相似。主要区别在于它允许 Lambda 函数访问 Rekognition API。让我们看看这个接口是如何工作的。`analysis-service`代码中的`handler.js`文件实现了这个接口。

以下列表显示了`analysis-service`的`require`语句。

图 3.2 分析服务 require

```
const AWS = require('aws-sdk')       ❶
const s3 = new AWS.S3()              ❷
const rek = new AWS.Rekognition()    ❸
```

❶ AWS SDK 模块已加载。

❷ S3 接口是为了处理存储桶及其对象而创建的。

❸ 创建 Rekognition 接口。

下一个列表显示了如何在`analyzeImageLabels`函数中使用 Rekognition 对象。

图 3.3 使用 Rekognition API

```
function analyzeImageLabels (imageBucketKey) {
  const params = {                                ❶
    Image: {
      S3Object: {
        Bucket: process.env.BUCKET,
        Name: imageBucketKey
      }
    },
    MaxLabels: 10,
    MinConfidence: 80
  }
  return new Promise((resolve, reject) => {
    rek.detectLabels(params, (err, data) => {    ❷
      if (err) {
        return resolve({image: imageBucketKey, labels: [], err: err})
      }
      return resolve({image: imageBucketKey,
        labels: data.Labels})                    ❸
    })
  })
}
```

❶ 创建 Rekognition 调用参数。

❷ 调用 Rekognition 的 detectLabel API。

❸ 承诺解析为结果。

这个简单的函数实现了很多功能！它触发了一个图像识别 AI 服务对存储在 S3 存储桶中的图像文件进行运行，然后返回一组结果以供进一步处理。所有这些都在相当少的代码中完成！

应该注意的是，我们可以用 Rekognition 做很多事情；然而，出于这个代码的目的，我们只是使用了默认设置。我们将在后面的章节中更详细地探讨这一点。

小贴士：Rekognition 不仅适用于视频，也适用于静态图像，并且可以用于检测图像中的各种特征，如微笑或皱眉的脸，图像中的文本，以及知名人士。你能想到哪些对你的最终用户有益的应用？例如，我们最近用它来检测图像中的地址和邮政编码。

`analysis-service`的最终列表显示了`wordCloudList`函数。该函数计算所有检测到的标签中单词出现的次数。

图 3.4 `wordCloudList`计算

```
function wordCloudList (labels) {            ❶
  let counts = {}
  let wcList = []

  labels.forEach(set => {                    ❷
    set.labels.forEach(lab => {
      if (!counts[lab.Name]) {
        counts[lab.Name] = 1
      } else {
        counts[lab.Name] = counts[lab.Name] + 1
      }
    })
  })

  Object.keys(counts).forEach(key => {       ❸
    wcList.push([key, counts[key]])
  })
  return wcList
}
```

❶ 函数接受一个标签对象的数组。

❷ 遍历每个标签对象中设置的标签集以计算标签出现次数。

❸ 将计数映射转换为表示为两个元素数组的单词-计数对的数组。

让我们继续使用 Serverless Framework 部署分析服务：

```
$ cd analysis-service
$ npm install
$ serverless deploy
```

一旦部署成功完成，我们可以通过 AWS 控制台将一个测试消息排队到 SQS 中，重新运行我们的系统。请继续这样做，发送与之前相同的 JSON 消息：

```
{
  "action": "download",
  "msg": {
    "url": "http://ai-as-a-service.s3-website-eu-west-1.amazonaws.com"
  }
}
```

这将导致爬虫服务运行。一旦完成，爬虫将向分析 SQS 队列发送一条消息，请求对下载的图像进行分析，这将触发分析服务。最终结果将是一组标签添加到 S3 中的`status.json`文件中。如果你打开这个文件，你应该会看到以下类似的列表。

图 3.5 `wordCloudList`计算结果

```
{
  "url": "http://ai-as-a-service.s3-website-eu-west-1.amazonaws.com",
  "stat": "analyzed",
  "downloadResults": [         ❶
    {
      "url": "http://ai-as-a-service.s3-website-eu-west-1.amazonaws.com/cat1.png",
      "stat": "ok"
    },
    ...
  ],
  "analysisResults": [         ❷
    {
      "image": "ai-as-a-service.s3-website-eu-west-1.amazonaws.com/cat1.png",
      "labels": [
        {
          "Name": "Cat",
          "Confidence": 99.03962707519531
        }
      ]
      ...
  ],
  "wordCloudList": [           ❸
    [ "Cat", 3 ],
    [ "Dog", 3 ],
    ....
  ]
}
```

❶ 图像下载结果

❷ 图像分析结果

❸ 单词云计算

对于一个更完整的系统，我们可能会考虑将此信息存储在数据库或键/值存储中；然而，对于这个第一个演示，S3 就足够好了。此状态文件用于驱动前端和 UI 服务，我们现在将关注这些服务。

## 3.2 实现同步服务

在这个系统中，同步服务由 UI 服务和前端组成。前端在浏览器中渲染和执行，而 UI 服务作为三个 Lambda 函数执行。

### 3.2.1 UI 服务

图 3.2 概述了 UI 服务的操作。

如上图所示，UI 服务公开了三个端点：

+ `url/list` 列出已提交进行分析的所有 URL。
+ `image/list` 列出针对特定 URL 已分析的所有图像。
+ `url/analyze` 提交 URL 以进行分析。

Serverless 框架允许我们在单个配置文件中定义多个 Lambda 函数，我们已经在 UI 服务的配置中使用了这一点。让我们看一下 UI 服务的`serverless.yml`文件，它将在下一个列表中展示。

![](img/CH03_F02_Elger.png)

图 3.2 UI 服务

图 3.6 UI 服务 `serverless.yml`

```
service: ui-service
frameworkVersion: ">=1.30.0"
plugins:
  - serverless-domain-manager                        ❶
custom:
  bucket: ${env:CHAPTER2_BUCKET}
  queue: Chap2Queue
  region: ${env:AWS_DEFAULT_REGION, 'eu-west-1'}
  domain: ${env:CHAPTER2_DOMAIN}
  accountid: ${env:AWS_ACCOUNT_ID}
  customDomain:                                      ❷
    domainName: 'chapter2api.${self:custom.domain}'
    stage: dev
    basePath: api
    certificateName: '*.${self:custom.domain}'
    createRoute53Record: true
    endpointType: regional

provider:
  name: aws
  runtime: nodejs12.x
  region: ${env:AWS_DEFAULT_REGION, 'eu-west-1'}
  iamRoleStatements:                                 ❸
  ...

functions:
  analyzeUrl:                                        ❹
    handler: handler.analyzeUrl
    environment:
     ...
    events:
      - http:
          path: url/analyze
          method: post
  listUrls:                                          ❺
    handler: handler.listUrls
    ...
  listImages:                                        ❻
     handler: handler.listImages
    ....
```

❶ 域插件

❷ 自定义域名设置

❸ 角色权限

❹ 分析 URL Lambda HTTP POST

❺ 列出 URL Lambda

❻ 列出图像 Lambda

此配置在之前的配置文件之上引入了一些新元素。首先，我们使用了一个自定义插件--`serverless-domain-manager`。我们使用它来帮助我们为我们的服务设置一个自定义域名。如果您还记得，在第二章的开始，我们在 Route53 中设置了一个域名并创建了一个通配符证书。一会儿，我们将使用此域名来为我们的 UI 服务。

在此阶段，配置中的权限部分应该看起来很熟悉。函数部分略有不同，因为它有三个条目。请注意，每个条目都是相似的，因为它与一个`HTTP`事件相关联。这告诉 Serverless 将函数绑定到 API Gateway，并通过给定的路由使函数可用。自定义域名条目用于为服务创建 DNS 条目，并将其连接到 API Gateway。我们将在一会儿部署此服务，但首先让我们看看实现，它在`handler.js`文件中。

在现在应该已经熟悉的模式中，我们引入 AWS SDK 并创建服务所需的对象，在这种情况下是 S3 和 SQS。这将在以下列表中展示。

图 3.7 UI 服务 `require`

```
const urlParser = require('url')    ❶
const AWS = require('aws-sdk')      ❷
const s3 = new AWS.S3()
const sqs = new AWS.SQS({region: process.env.REGION})
```

❶ 加载 url 节点模块以进行 URL 解析。

❷ 加载 AWS SDK 并实例化 S3 和 SQS 接口。

该服务定义了三个入口点，它们将被部署为三个单独的 Lambda 函数。`listUrl`函数将在下一个列表中提供。

图 3.8 `listUrls`函数

```
module.exports.listUrls = (event, context, cb) => {     ❶
  const params = {
    Bucket: process.env.BUCKET,
    Delimiter: '/',
    MaxKeys: 1000
  }

  s3.listObjectsV2(params, (err, data) => {             ❷
    let promises = []
    if (err) { return respond(500, {stat: 'error', details: err}, cb) }

    data.CommonPrefixes.forEach(prefix => {
      promises.push(readStatus(prefix.Prefix))
    })
    Promise.all(promises).then(values => {
      let result = []
      values.forEach(value => {
        result.push({url: value.url, stat: value.stat})
      })
      respond(200,
        {stat: 'ok', details: result}, cb)              ❸
    })
  })
}
```

❶ 入口点

❷ 列出 S3 对象

❸ 返回 URL 列表

注意，此函数的入口点与我们的所有其他服务完全相同，尽管在这种情况下，该函数将通过 API Gateway 作为 HTTP GET 请求执行。该函数相当简单，它只是列出我们 S3 存储桶顶层的一组文件夹，并将列表作为 JSON 数组返回。

我们的`listImages`函数更加简单，因为它从 S3 读取`status.json`文件并返回用于显示的内容，所以我们在这里不会详细说明它。相反，让我们看一下以下代码中的`analyzeUrl`函数。

图 3.9 `analyzeUrl`

