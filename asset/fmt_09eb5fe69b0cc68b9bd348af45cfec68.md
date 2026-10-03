
```py
distribution = tf.distribute.experimental.MultiWorkerMirroredStrategy()

with distribution.scope():
    mirrored_model = tf.keras.Sequential([...])
    mirrored_model.compile([...])

batch_size = 100 # must be divisible by the number of replicas
history = mirrored_model.fit(X_train, y_train, epochs=10) 
```

这就是前面用的代码，只是这次我们使用的是`MultiWorkerMirroredStrategy`（未来版本中，`MirroredStrategy`可能既处理单机又处理多机）。当在第一个工作器上运行脚本时，它会阻塞所有 AllReduce 步骤，最后一个工作器启动后，训练就开始了。可以看到工作器以相同的速度前进（因为每步使用的同步）。

你可以从两个 AllReduce 实现选择做分布策略：基于 gRPC 的 AllReduce 算法用于网络通信，和 NCCL 实现。最佳算法取决于工作器的数量、GPU 的数量和类型和网络。默认，TensorFlow 会选择最佳算法，但是如果想强制使用某种算法，将`CollectiveCommunication.RING`或`CollectiveCommunication.NCCL`（出自`tf.distribute.experimental`）传给策略构造器。

如果想用带有参数服务器的异步数据并行，可以将策略变为`ParameterServerStrategy`，添加一个或多个参数服务器，给每个任务配置`TF_CONFIG`。尽管工作器是异步的，每个工作器的复制是同步工作的。

最后，如果你能用 Google Cloud 的 TPU，可以如下创建`TPUStrategy`：

```py
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.tpu.experimental.initialize_tpu_system(resolver)
tpu_strategy = tf.distribute.experimental.TPUStrategy(resolver) 
```

> 提示：如果是研究员，可以免费试用 TPU，见[这里](https://links.jianshu.com/go?to=https%3A%2F%2Ftensorflow.org%2Ftfrc)。

现在就可以在多机多 GPU 训练模型了。如果想训练一个大模型，需要多个 GPU 多台服务器，要么买机器，要么买云虚拟机。云服务更便宜。

### 在 Google Cloud AI Platform 上训练大任务

如果你想用 Google AI Platform，可以用相同的代码部署训练任务，平台会管理 GPU VM。

要启动任务，你需要命令行工具`gcloud`，它属于 [Google Cloud SDK](https://links.jianshu.com/go?to=https%3A%2F%2Fcloud.google.com%2Fsdk%2F)。可以在自己的机器上安装 SDK，或在 GCP 上使用 Google Cloud Shell。这是可以在浏览器中使用的终端；运行在免费的 Linux VM（Debian）上，SDK 已经安装配置好了。Cloud Shell 可以在 GCP 上任何地方使用：只要点击页面右上的图标 Activate Cloud Shell（见图 19-22）。

![](img/35d9fcd8b373f316881a7a6c79e7c9a7.png)

图 19-22 启动 Google Cloud Shell

如果想在自己机器上安装 SDK，需要运行`gcloud init`启动：需要登录 GCP 准许权限，选择想要的 GCP 项目，还有想运行的地区。`gcloud`命令可以使用 GCP 所有功能。不用每次访问网页接口，可以写脚本开启或停止虚拟机、部署模型或做任意 GCP 动作。

运行训练任务之前，你需要写训练代码，和之前的分布设置一样（例如，使用`ParameterServerStrategy`）。AI 平台会为每个 VM 设置`TF_CONFIG`。做好之后，就可以在 TF 集群部署运行了，命令行如下：

```py
$ gcloud ai-platform jobs submit training my_job_20190531_164700 \
    --region asia-southeast1 \
    --scale-tier PREMIUM_1 \
    --runtime-version 2.0 \
    --python-version 3.5 \
    --package-path /my_project/src/trainer \
    --module-name trainer.task \
    --staging-bucket gs://my-staging-bucket \
    --job-dir gs://my-mnist-model-bucket/trained_model \
    --
    --my-extra-argument1 foo --my-extra-argument2 bar 
```

浏览这些选项。命令行启动名为`my_job_20190531_164700`的训练任务，地区是`asia-southeast1`，级别是`PREMIUM_1`：对应 20 个工作器和 11 个参数服务器（查看[其它等级](https://links.jianshu.com/go?to=https%3A%2F%2Fhoml.info%2Fscaletiers)）。所有 VM 基于 AI Platform 的 2.0 运行时（VM 配置包括 TensorFlow 2.0 和其它包）和 Python 3.5。训练代码位于字典`/my_project/src/trainer`，命令`gcloud`会自动绑定 PIP 包，并上传到 GCS 的`gs://my-staging-bucket`。然后，AI Platform 会启动几个 VM，部署这些包，运行`trainer.task`模块。最后，参数`--job-dir`和其它参数（即，分隔符`--`后面的参数）会传给训练程序：主任务会使用参数`--job-dir`在 GCS 上保存模型，在这个例子中，是在`gs://my-mnist-model-bucket/trained_model`。就是这样。在 GCP 控制台中，你可以打开导航栏，下滑到`Artificial Intelligence`，打开`AI Platform → Jobs`。可以看到在运行的任务，如果点击，可以看到图展示了每个任务的 CPU、GPU 和 RAM。点击`View Logs`，可以使用 Stackdriver 查看详细日志。

> 笔记：如果将训练数据放到 GCS 上，可以创建`tf.data.TextLineDataset`或`tf.data.TFRecordDataset`来访问：用 GCS 路径作为文件名（例如，`gs://my-data-bucket/my_data_001.csv`）。这些数据集依赖包`tf.io.gfile`访问文件：支持本地文件和 GCS 文件（要保证服务账号可以使用 GCS）。

如果想探索几个超参数的值，可以用参数指定超参数值，执行多个任务。但是，日过想探索许多超参数，最好使用 AI Platform 的超参数调节服务。

### 在 AI Platform 上做黑盒超参数调节

AI Platform 提供了强大的贝叶斯优化超参数调节服务，称为 [Google Vizier](https://links.jianshu.com/go?to=https%3A%2F%2Fhoml.info%2Fvizier)。要使用，创建任务时要传入 YAML 配置文件（`--config tuning.yaml`）。例如，可能如下：

```py
trainingInput:
  hyperparameters:
    goal: MAXIMIZE
    hyperparameterMetricTag: accuracy
    maxTrials: 10
    maxParallelTrials: 2
    params:
      - parameterName: n_layers
        type: INTEGER
        minValue: 10
        maxValue: 100
        scaleType: UNIT_LINEAR_SCALE
      - parameterName: momentum
        type: DOUBLE
        minValue: 0.1
        maxValue: 1.0
        scaleType: UNIT_LOG_SCALE 
```

它告诉 AI Platform，我们的目的是最大化指标`"accuracy"`，任务会做最多 10 次试验（每次试验都从零开始训练），最多并行运行 2 个试验。我们想调节两个超参数：`n_layers`（10 到 100 间的整数），和`momentum`（0.1 和 1.0 之间的浮点数）。参数`scaleType`指明了先验：`UNIT_LINEAR_SCALE`是扁平先验（即，没有先验偏好），`UNIT_LOG_SCALE`的先验是最优值靠近最大值（其它可能的先验是`UNIT_REVERSE_LOG_SCALE`，最佳值靠近最小值）。

`n_layers`和`momentum`参数会作为命令行参数传给训练代码。问题是训练代码如何将指标传回给 AI Platform，以便决定下一个试验使用什么超参数？AI Platform 会监督输出目录（通过`--job-dir`指定）的每个包含指标`"accuracy"`概括的事件文件（或是其它`hyperparameterMetricTag`指定的名字），读取这些值。训练代码使用`TensorBoard()`调回，就可以开始了。

任务完成后，每次试验中使用的超参数值和结果准确率会显示在任务的输出中（在`AI Platform → Jobs page`）。

> 笔记：AI Platform 还可以用于在大量数据上执行模型：每个工作器从 GCS 读取部分数据，做预测，并保存在 GCS 上。

现在就可以用各种分布策略规模化创建先进的神经网络架构了，可以用自己的机器，也可以用云 —— 还可以用高效贝叶斯优化微调超参数。

## 练习

1.  `SavedModel`包含什么？如何检查内容？

2.  什么时候使用 TF Serving？它有什么特点？可以用什么工具部署 TF Serving？

3.  如何在多个 TF Serving 实例上部署模型？

4.  为什么使用 gRPC API 而不是 REST API，查询 TF Serving 模型？

5.  在移动和嵌入设备上运行，TFLite 减小模型的大小有什么方法？

6.  什么是伪量化训练，有什么用？

7.  什么是模型并行和数据并行？为什么推荐后者？

8.  在多台服务器上训练模型时，可以使用什么分布策略？如何进行选择？

9.  训练模型（或任意模型），部署到 TF Serving 或 Google Cloud AI Platform 上。写客户端代码，用 REST API 或 gRPC API 做查询。更新模型，部署新版本。客户端现在查询新版本。回滚到第一个版本。

10.  用一台机器多个 GPU、`MirroredStrategy`策略，训练模型（如果没有 GPU，可以使用带有 GPU 的 Colaboratory，创建两个虚拟 GPU）。再用`CentralStorageStrategy`训练一次，比较训练时间。

11.  在 Google Cloud AI Platform 训练一个小模型，使用黑盒超参数调节。

参考答案见附录 A。
