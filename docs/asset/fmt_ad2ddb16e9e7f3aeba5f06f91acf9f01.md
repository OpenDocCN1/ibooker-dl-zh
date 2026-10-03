
```
hyperparams: lr 0.01 bs 32 epochs 10
Epoch 1/10
200/200 [==============================] - 1s 3ms/step - loss: 1.9392 - acc: 0.2973
Epoch 2/10
200/200 [==============================] - 1s 3ms/step - loss: 1.6730 - acc: 0.4130
...
Epoch 10/10
200/200 [==============================] - 1s 3ms/step - loss: 1.3809 - acc: 0.5170

hyperparams: lr 0.005 bs 64 epochs 10
Epoch 1/10
200/200 [==============================] - 1s 3ms/step - loss: 1.2248 - acc: 0.5704
Epoch 2/10
200/200 [==============================] - 1s 3ms/step - loss: 1.2740 - acc: 0.5510
...
Epoch 10/10
200/200 [==============================] - 1s 3ms/step - loss: 1.1876 - acc: 0.5853

hyperparams: lr 0.0025 bs 128 epochs 10
Epoch 1/10
200/200 [==============================] - 1s 4ms/step - loss: 1.1186 - acc: 0.6063
Epoch 2/10
200/200 [==============================] - 1s 3ms/step - loss: 1.1434 - acc: 0.5997
...
Epoch 10/10
200/200 [==============================] - 1s 3ms/step - loss: 1.1156 - acc: 0.6129
```

### 14.1.2 使用 `tf.Strategy` 进行分布式喂养

TensorFlow 模块`tf.distribute.Strategy`提供了一个方便且封装的接口，为你完成所有工作，用于在同一个计算实例上的多个 GPU 之间或多个 TPU 之间进行分布式训练。它实现了本章前面描述的同步参数服务器。此 TensorFlow 模块针对 TensorFlow 模型的分布式训练以及并行 Google TPUs 上的分布式训练进行了优化。

当在单个计算实例上使用多个 GPU 进行训练时，你使用 `tf.distribute.MirrorStrategy`，而当在 TPU 上进行训练时，你使用 `tf.distribute.TPUStrategy`。在本章中，除了指出你将使用 `tf.distribute.experimental.ParameterServerStrategy` 以实现跨网络的异步参数服务器之外，我们不会涵盖跨机器的分布式训练。跨多个机器的分布式训练设置相对复杂，可能需要单独的一章。如果你在构建 TensorFlow 模型，并且训练过程中需要大量的并行处理以满足业务目标，我建议使用这种方法，并学习 TensorFlow 文档。

这是我们在单机上设置分布式训练运行的步骤，该机器具有多个 CPU 或 GPU：

1. 实例化一个分布策略。
2. 在分布策略的作用域内
   - 创建模型。
   - 编译模型。
3. 训练模型。

这些步骤可能看起来有些不合常理，因为我们是在构建和编译模型时设置分布策略，而不是在训练时。在 TensorFlow 中，模型构建需要知道它将使用分布式训练策略进行训练。截至本文撰写时，TensorFlow 团队最近发布了一个新的实验版本，其中分布策略可以在不编译模型的情况下设置。

以下是实现前面三个步骤和两个子步骤的代码，这些步骤在此处描述：

1. 我们定义函数 `create_model()` 来创建用于训练的模型实例。
2. 我们实例化分布策略：`strategy = tf.distribute.MirrorStrategy()`。
3. 我们设置分布上下文：`with strategy.scope()`。
4. 在分布上下文中，我们创建模型的实例：`model = create_model()`。然后我们编译它：`model.compile()`。
5. 最后，我们训练模型。

```
def create_model():                                               ❶
    model = Sequential([ Conv2D(16, (3, 3), activation='relu', 
                                input_shape=(32, 32, 3)),
                         Conv2D(32, (3, 3), strides=(2, 2), 
                                activation='relu'),
                         MaxPooling2D((2, 2), strides=2),
                         Flatten(),
                         Dense(10, activation='softmax')
                       ])
    return model

strategy = tf.distribute.MirroredStrategy()                       ❷

with strategy.scope():                                            ❸
    model = create_model()                                        ❸
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')    

model.fit(dataset, epochs=10, steps_per_epoch=200)                ❹
```

❶ 创建模型实例的函数  
❷ 实例化分布策略  
❸ 在分布策略的作用域内创建和编译模型  
❹ 训练模型

你可能会问，我能否使用已经构建好的模型？答案是：不可以；你必须在分布策略的作用域内构建模型。例如，以下代码将导致错误，提示模型未在分布策略的作用域内构建：

```
model = create_model()          ❶
with strategy.scope():
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
```

❶ 模型未在分布策略的作用域内构建

再次，你可能还会问：我已经有一个预先构建或预训练的模型，它不是为分布策略构建的；我还能进行分布式训练吗？这里的答案是：可以。如果你有一个保存到磁盘的 TF.Keras 模型，当你使用 `load_model()` 将其加载回内存时，它将隐式地构建模型。以下是从预训练模型设置分布策略的示例实现：

```
with strategy.scope():
    model = tf.keras.models.load_model('my_model')                          ❶
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
```

❶ 从磁盘加载时模型会隐式重建

同样，当从模型存储库加载预构建模型时，存在隐式的加载和相应的隐式构建。以下代码序列是加载`tf.keras.applications`内置模型存储库中模型的示例，其中模型被隐式重建：

```
with strategy.scope():
   model = tf.keras.applications.ResNet50()                                ❶
   model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
```

❶ 从存储库加载时模型被隐式重建

默认情况下，镜像策略将使用计算实例上的所有 GPU。您可以使用`num_replicas_in_sync`属性获取将要使用的 GPU 或 CPU 核心数。您还可以明确设置要使用的 GPU 或核心。在以下代码示例中，我们将分布策略设置为使用两个 GPU：

```
strategy = tf.distribute.MirroredStrategy(['/gpu:0', '/gpu:1'])
print("GPUs:", strategy.num_replicas_in_sync)
```

以下代码示例生成了以下输出：

```
INFO:tensorflow:Using MirroredStrategy with devices ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1')
GPUs: 2
```

### 14.1.3 使用 TFX 进行模型喂养

第十三章涵盖了 TFX 端到端生产管道的数据管道部分。本节涵盖了训练管道组件的相应 TFX 模型喂养方面，作为另一种实现。图 14.6 描述了训练管道组件及其与数据管道的关系。训练管道由以下组件组成：

+ *训练器*—训练模型
+ *调优器*—调整超参数（例如，学习率）
+ *评估器*—评估模型的客观指标，例如准确性，并将结果与基线（例如，上一个版本）进行比较
+ *基础设施评估器*—在部署前在沙盒服务环境中测试模型

![图片](img/CH14_F06_Ferlitsch.png)

图 14.6 TFX 训练管道由调优器、训练器、评估器和基础设施评估器组件组成。

**编排**

让我们回顾一下 TFX 和管道的一般好处。如果我们单独执行训练/部署模型中的每个步骤，我们将其称为*任务感知架构*。每个组件都了解自己，但不知道连接的组件或之前执行的历史。

TFX 实现了*编排*。在编排中，一个管理接口监督每个组件的执行，记住过去组件的执行情况，并维护历史记录。如第十三章所述，每个组件的输出是工件；这些是执行的结果和历史。在编排中，这些工件或对其的引用被存储为元数据。对于 TFX，元数据以关系格式存储，因此可以通过 SQL 数据库进行存储和访问。

让我们更深入地探讨编排的好处，然后我们将介绍 TFX 中模型喂养的工作方式。通过编排，如图 14.7 所示，我们可以做以下事情：

+ 在另一个组件（或多个组件）完成后执行组件的调度。例如，我们可以在从训练数据生成特征模式后调度数据转换的执行。
+ 当组件的执行相互不依赖时，并行调度组件的执行。例如，在数据转换完成后，我们可以并行调度超参数调整和训练。
+ 如果组件的执行没有变化（缓存），则重用组件先前执行中的工件。例如，如果训练数据没有变化，转换组件的缓存工件（即转换图）可以在不重新执行的情况下重用。
+ 为每个组件提供不同的计算引擎实例。例如，数据管道组件可能配置在 CPU 计算实例上，而训练组件配置在 GPU 计算实例上。
+ 如果任务支持分布，例如调整和训练，则可以将任务分布到多个计算实例上。
+ 将组件的工件与组件先前执行的工件进行比较。例如，评估组件可以将模型的指标（例如，准确率）与先前训练的模型版本进行比较。
+ 通过能够向前和向后移动通过生成的工件来调试和审计管道的执行。

![图片](img/CH14_F07_Ferlitsch.png)

图 14.7 编排器摄取表示为图的管道，并配置实例和调度任务。

**训练组件**

`Trainer`组件支持训练 TensorFlow 估计器、TF.Keras 模型和其他自定义训练循环。由于 TensorFlow 2.*x*建议逐步淘汰估计器，我们将仅关注配置 TF.Keras 模型的训练组件，并向其提供数据。训练组件需要以下最小参数：

+ `module_file`—这是用于自定义训练模型的 Python 脚本。它必须包含一个`run_fn()`函数作为训练的入口点。
+ `examples`—用于训练模型的示例，这些示例来自`ExampleGen`组件的输出，`example_gen.outputs['examples']`。
+ `schema`—数据集模式，它来自`SchemaGen`组件的输出，`schema_gen['schema']`。
+ `custom_executor_spec`—自定义训练的执行器，它将在`module_file`中调用`run_fn()`函数。

```
from tfx.components import Trainer
from tfx.components.base import executor_spec                              ❶
from tfx.components.trainer import GenericExecutor                         ❶

trainer = Trainer(
    module_file=module_file,                                               ❷
    examples=example_gen.outputs['examples'],                              ❸
    schema=schema_gen.outputs['schema'],                                   ❹
    custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor)  ❺
)
```

❶ 自定义训练的导入  
❷ 自定义训练的 Python 脚本  
❸ 在训练过程中向模型提供数据的训练数据源  
❹ 从数据集中推断出的模式  
❺ 自定义训练的自定义执行器

如果训练数据需要由`Transform`组件进行预处理，我们需要设置以下两个参数：

+ `transformed_examples`—设置为`Transform`组件的输出，`transform.outputs['transformed_examples']`。
+ `transform_graph`—由`Transform`组件产生的静态转换图，`transform.outputs['transformed_graph']`。

```
trainer = Trainer(
    module_file=module_file,
    transformed_examples=transform.outputs['transformed_examples'],    ❶
    transform_graph=transform.outputs['transform_graph'],              ❶
    schema=schema_gen.outputs['schema'],
    custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor)
)
```

❶ 训练数据从`Transform`组件馈送到静态转换图。

通常，我们希望将其他超参数传递到训练模块中。这些可以通过将 `train_args` 和 `eval_args` 作为附加参数传递给 `Trainer` 组件。这些参数被设置为键/值对列表，并转换为 Google 的 protobuf 格式。以下代码传递了训练和评估的步骤数：

```
from tfx.proto import trainer_pb2                                          ❶
trainer = Trainer(
    module_file=module_file,
    transformed_examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],   
    schema=schema_gen.outputs['schema'],
    custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
    train_args=trainer_pb2.TrainArgs(num_steps=10000),                     ❷
    eval_args=trainer_pb2.EvalArgs(num_steps=5000)                         ❷
)
```

❶ 导入用于传递超参数的 TFX protobuf 格式  
❷ 将超参数作为 protobuf 消息传递到 Trainer 组件中

现在，让我们看看自定义 Python 脚本中 `run_fn()` 函数的基本要求。`run_fn()` 的参数由传递给 `Trainer` 组件的参数构建，并且作为属性访问。在以下示例实现中，我们执行以下操作：

+ 提取训练的总步骤数：`training_args.train_steps`。
+ 提取每个 epoch 后的验证步骤数：`training_args.eval_steps`。
