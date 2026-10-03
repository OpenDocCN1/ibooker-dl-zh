
## 沙盒环境的规范

服务规范由两部分组成：

- 服务二进制的类型。截至 TFX 版本 0.22，仅支持 TensorFlow Serving。
- 服务平台的类型，可以是：
  - Kubernetes
  - 本地 Docker 容器

此示例展示了使用 TensorFlow Serving 和 Kubernetes 集群指定服务规范的最小要求：

```
from tfx.proto.infra_validator_pb2 import ServingSpec

serving_spec = ServingSpec(tensorflow_serving=TensorflowServing(tags=['latest']),
                           kubernetes=KubernetesConfig()
                          )
```

TFX 关于 `ServingSpec` 的文档目前很少，并将您重定向到 GitHub 存储库中的 protobuf 定义（[`mng.bz/6NqA`](http://mng.bz/6NqA)）以获取更多信息。

## 提供预测服务

现在我们有了新的祝福模型，我们将探讨如何将模型部署到生产环境中以提供预测服务。生产模型通常用于按需（实时）或批量预测。

批量预测与从部署的模型中进行的按需（实时）预测有何不同？有一个关键的区别，但除此之外，它们在结果上基本上是相同的：

- **按需（实时）**——为整个实例集（一个或多个数据项）进行按需预测，并实时返回结果
- **批量预测服务**——在后台为整个实例集进行排队（批量）预测，并在准备好时将结果存储在云存储桶中

### 按需（实时）服务

对于按需预测，例如通过交互式网站进行的在线请求，模型被部署到一个或多个计算实例上，并接收作为 HTTP 请求的预测请求。一个预测请求可以包含一个或多个单个预测；每个预测通常被称为*实例*。您可以有单实例请求，其中用户只想对一张图像进行分类，或者多实例请求，其中模型将为多张图像返回预测。

假设模型接收单实例请求：用户提交一个图像并希望得到一个预测，如分类或图像标题。这些都是通过互联网实时到达的按需请求。例如，它们可能来自用户浏览器中运行的 Web 应用程序，或者服务器上的后端应用程序作为微服务获取预测。

图 14.19 展示了这个过程。在这个描述中，模型包含在一个服务二进制文件中，该文件由一个 Web 服务器、服务功能和受祝福的模型组成。Web 服务器接收预测请求作为一个 HTTP 请求包，提取请求内容，并将内容传递给服务功能。服务功能随后将内容预处理成受祝福模型输入层期望的格式和形状，然后输入到受祝福模型中。受祝福模型将预测返回给服务功能，服务功能执行任何后处理以进行最终交付，然后将其返回给 Web 服务器，Web 服务器将后处理的预测作为 HTTP 响应包返回。

![图片](img/CH14_F19_Ferlitsch.png)

图 14.19 一个在生产二进制文件上的模型通过互联网接收按需预测请求

如您在图 14.19 中看到的，在客户端，一个或多个预测请求被传递给一个 Web 客户端。Web 客户端随后将创建一个单实例或多实例的预测 HTTP 请求包。预测请求被编码，通常为 base64，以确保通过互联网安全传输，并放置在 HTTP 请求包的内容部分。

Web 服务器接收 HTTP 请求，解码内容部分，并将单个或多个预测请求传递给服务功能。

现在我们来深入探讨服务函数的目的和构建方式。通常，在客户端，内容（如图片、视频、文本和结构化数据）以原始格式发送到服务二进制文件，而不进行任何预处理。当网络服务器接收到请求后，它会从请求包中提取内容并将其传递给服务函数。在传递给服务函数之前，内容可能需要进行解码，例如 base64 解码。

假设内容是一个包含单个实例请求的内容，例如 JPG 或 PNG 格式的压缩图像。假设模型的输入层是不压缩的图像字节，格式为多维数组，例如 TensorFlow 张量或 NumPy 数组。至少，服务函数必须执行模型之外的任何预处理（例如，预茎）。假设模型没有预茎，服务函数需要执行以下操作：

- 确定图像数据的压缩格式，例如从 MIME 类型。
- 将图像解压缩为原始字节。
- 将原始字节重塑为高度 × 宽度 × 通道（例如，RGB）。
- 将图像调整大小以匹配模型的输入形状。
- 对像素数据进行缩放，以进行归一化或标准化。

接下来是一个图像分类模型的服务函数示例实现，其中图像数据的预处理发生在模型上游，没有预茎。在这个例子中，`serving_fn()` 方法通过将方法分配为模型的签名 `serving_default` 在服务二进制文件中的网络服务器上注册。我们在服务函数中添加了装饰器 `@tf.function`，该装饰器指示 AutoGraph 编译器将 Python 代码转换为静态图，然后可以在 GPU 上与模型一起运行。在这个例子中，假设网络服务器将提取的预测请求内容（在这种情况下，JPG 压缩字节）作为 TensorFlow 字符串传递。对 `tf.saved_model.save()` 的调用将服务函数保存到与模型相同的存储位置，该位置由参数 `export_path` 指定。

现在我们来看看这个服务函数的主体。在下面的代码示例中，我们假设服务二进制文件中的网络服务器从 HTTP 请求包中提取内容，解码 base64 编码，并将内容（压缩的 JPG 图像字节）作为 TensorFlow 字符串数据类型 `tf.string` 传递。然后服务函数执行以下操作：

- 调用一个预处理函数 `preprocess_fn()`，将 JPG 图像解码为原始字节，并调整大小和缩放以匹配底层模型的输入层，作为一个多维 TensorFlow 数组。
- 将多维 TensorFlow 数组传递给底层模型 `m_call()`。
- 将底层模型返回的预测 `prob` 返回给网络服务器。
- 服务二进制文件中的 Web 服务器将预测结果打包到 HTTP 响应数据包中，返回给 Web 客户端。

```
@tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
def serving_fn(bytes_inputs):                                     ❶
    images = preprocess_fn(bytes_inputs)                          ❷
    prob = m_call(**images)                                       ❸
    return prob                                                   ❹

tf.saved_model.save(model, export_path, signatures={              ❺
    'serving_default': serving_fn,                                ❺
})
```

❶ 定义接收通过服务二进制文件的 Web 服务器内容的服务函数

❷ 将内容转换为与底层模型输入层匹配的方法

❸ 将预处理数据传递给底层模型进行预测。

❹ 预测结果被返回到服务二进制文件的 Web 服务器，作为 HTTP 响应返回。

❺ 将服务函数作为静态图与底层模型保存

以下是一个服务函数预处理步骤的示例实现。在这个例子中，函数 `preprocess_fn()` 接收来自 Web 服务器的 base64 解码后的 TensorFlow 字符串，并执行以下操作：

- 调用 TensorFlow 静态图操作 `tf.io.decode_jpeg()` 将输入解压缩为一个解压缩图像，作为多维 TensorFlow 数组。
- 调用 TensorFlow 静态图操作 `tf.image.convert_image_dtype()` 将整数像素值转换为 32 位浮点值，并将值缩放到 0 到 1 的范围（归一化）。
- 调用 TensorFlow 静态图操作 `tf.image.resize()` 将图像调整大小以适应模型的输入形状。在这个例子中，那将是 (192, 192, 3)，其中值 3 是通道数。
- 将预处理后的图像数据传递给底层模型的输入层，该输入层由层的签名 `numpy_inputs` 指定。

```
def _preprocess(bytes_input):
    decoded = tf.io.decode_jpeg(bytes_input, channels=3)             ❶
    decoded = tf.image.convert_image_dtype(decoded, tf.float32)      ❷
    resized = tf.image.resize(decoded, size=(192, 192))              ❸
    return resized

@tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
def preprocess_fn(bytes_inputs):
    with tf.device("cpu:0"):
        decoded_images = tf.map_fn(_preprocess, bytes_inputs, 
        dtype=tf.float32)                                           ❹
    return {"numpy_inputs": decoded_images}                         ❺
```

❶ 将 TensorFlow 字符串解码为编码的 JPG，并将其转换为 TensorFlow 多维解压缩图像原始字节数据。

❷ 将像素转换为 32 位浮点值并缩放

❸ 将图像调整大小以适应底层模型的输入形状

❹ 预处理请求中的每个图像

❺ 将预处理图像传递给底层模型的输入层

以下是对底层模型调用的示例实现，此处进行了描述：

- 参数 `model` 是编译后的 TF.Keras 模型，其中 `call()` 方法是模型的前向馈预测方法。
- `get_concrete_function()` 方法构建了一个围绕底层模型的包装器，用于执行。包装器提供了从服务函数中的静态图切换到底层模型中的动态图的接口。

```
m_call = tf.function(model.call).get_concrete_function([tf.TensorSpec(shape=
            [None, 192, 192, 3], dtype=tf.float32, name="numpy_inputs")])
```

### 批量预测

**批量预测**与部署模型进行按需预测不同。在按需预测中，你创建一个服务二进制文件和服务平台来部署模型；我们称之为*端点*。然后你将模型部署到该端点。最后，用户向端点发出按需（实时）预测请求。

相比之下，批量预测从创建一个用于预测的批量作业开始。作业服务随后为批量预测请求分配资源，并将结果返回给调用者。然后作业服务释放请求的资源。

批量预测通常用于不需要立即响应的情况，因此响应可以延迟；需要处理的预测数量巨大（数百万）；并且只需要为处理批量分配计算资源。

例如，考虑一家金融机构，在银行日结束时有一百万笔交易，并且它有一个模型可以预测未来 10 天的存款和现金余额。由于预测是时间序列的，逐笔发送交易到实时预测服务是没有意义且效率低下的。相反，在银行日结束时，交易数据被提取（例如，从 SQL 数据库中）并作为一个单独的批量作业提交。然后为服务二进制文件和平台分配计算资源，处理作业，并释放服务二进制文件和平台（释放资源）。

图 14.20 展示了一个批量预测服务。此过程有五个主要步骤：

1. 累积的数据被提取并打包成批量请求，例如从 SQL 数据库中提取。
2. 批量请求已排队，队列管理器确定计算资源需求和优先级。
3. 当批量作业准备就绪时，它将从队列中出队到调度器。
4. 调度器为服务二进制文件和平台分配资源，然后提交批量作业。
5. 批量作业完成后，结果被存储，调度器释放分配的计算资源。

![](img/CH14_F20_Ferlitsch.png)

图 14.20 一个队列和调度器根据每个作业协调服务二进制文件和平台的分配和释放。

接下来，我们将介绍如何在 TFX 中部署模型以进行按需和批量预测。

### TFX 部署管道组件

在 TFX 中，部署管道由组件 `Pusher` 和 `Bulk Inference` 以及一个服务二进制文件和平台组成。服务平台可以是基于云的、本地化的、边缘设备或基于浏览器的。对于基于云的模型，推荐的服务平台是 TensorFlow Serving。

图 14.21 展示了 TFX 部署管道的组件。`Pusher` 组件用于部署模型以进行按需预测或批量预测。`Bulk Inference` 组件处理批量预测。

![](img/CH14_F21_Ferlitsch.png)

图 14.21 一个 TFX 部署管道可以部署模型以进行按需服务和/或批量预测。

#### Pusher

以下是一个示例实现，展示了实例化 `Pusher` 组件以部署模型到服务二进制文件所需的最小要求：

- `model`——要部署到服务二进制文件和平台的训练模型（在这种情况下，来自 `Trainer` 组件的当前训练模型实例）
- `push_destination`——在服务二进制文件中安装模型的目录位置

```
from tfx.components import Pusher
from tfx.proto import pusher_pb2

pusher = Pusher(model=trainer.outputs['model'],                  ❶
                push_destination=pusher_pb2.PushDestination(     ❷

filesystem=pusher_pb2.PushDestination.FileSystem(

                base_directory=serving_model_dir                 ❸
                                                           )
               )
```

❶ 要部署的训练模型

❷ 部署模型的二进制文件目标

❸ 在服务二进制文件中的目录位置安装模型

在生产环境中，我们通常将其纳入部署管道中，只有当它是新的受祝福模型时才部署模型。以下是一个示例实现，其中只有当模型是新的受祝福模型时才部署模型：

- `model`——来自 `Trainer` 组件的当前训练的模型
- `model_blessing`——来自 `Evaluator` 组件的当前受祝福的模型

在这个例子中，只有在模型和受祝福的模型是相同的模型实例时，才会部署模型：

```
pusher = Pusher(model=trainer.outputs['model'],                  ❶
                model_blessing=evaluator.outputs['blessing'],    ❷
                push_destination=pusher_pb2.PushDestination( 

                filesystem=pusher_pb2.PushDestination.FileSystem(

                base_directory=serving_model_dir  
                                                            )
                                                           )
               )
```

❶ 当前训练的模型

❷ 当前受祝福的模型实例

接下来，我们将介绍在 TFX 中进行批量预测。

#### 批量推理器

`BulkInferrer` 组件执行批量预测服务，TFX 文档将其称为*批量推理*。以下代码是使用当前训练的模型进行批量预测的最小参数的示例实现：

- `examples`——用于进行预测的示例。在这种情况下，它们来自 `ExampleGen` 组件的一个实例。
- `model`——用于批量预测的模型（在这种情况下，当前训练的模型）。
- `inference_result`——存储批量预测结果的位置。

```
from tfx.components import BulkInferrer

bulk_inferrer = BulkInferrer(examples=examples_gen.outputs['examples'],  ❶
                             model=trainer.outputs['model'],             ❷
                             inference_result=location                   ❸
                            )
```

❶ 批量预测的示例

❷ 用于批量预测的模型

❸ 存储预测结果的位置

以下是一个示例实现，仅当当前训练的模型是受祝福的模型时，使用当前训练的模型进行批量预测的最小参数。在这个例子中，只有在当前训练的模型和受祝福的模型实例相同时，才会执行批量预测。
