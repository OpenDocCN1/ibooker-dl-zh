
# TensorBoard 词向量可视化

## 设置嵌入

```py
from tensorboard.plugins import projector

log_dir=os.path.join('logs', 'embeddings')
weights = tf.Variable(df_common.values)                          ❶

checkpoint = tf.train.Checkpoint(embedding=weights)              ❷
checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))         ❷

with open(os.path.join(log_dir, 'metadata.tsv'), 'w') as f:      ❸
    for w in df_common.index:
        f.write(w+'\n')

config = projector.ProjectorConfig()                             ❹
embedding = config.embeddings.add()
embedding.metadata_path = 'metadata.tsv'                         ❺
projector.visualize_embeddings(log_dir, config)
```

❶ 用我们捕获的嵌入创建一个`tf.Variable`。

❷ 将嵌入保存为 TensorFlow 检查点。

❸ 保存元数据（一个 TSV 文件），其中每个与嵌入对应的单词被附加为新行。

❹ 创建一个特定于投影仪和嵌入的配置（有关详细信息，请参阅文本）。

❺ 设置元数据路径，以便 TensorBoard 可以在可视化中包含它。

## 配置投影仪

要可视化来自保存的 TensorFlow 检查点和元数据（即，保存的单词向量对应的标记），我们使用`tensorboard.plugins.projector`对象。然后我们定义一个`ProjectorConfig`对象和一个嵌入配置。我们将保留它们的默认配置，这适合我们的问题。当调用`config.embeddings.add()`时，它将生成一个使用默认配置的嵌入配置（类型为`EmbeddingInfo`对象）。`ProjectorConfig`包含诸如以下信息：

+   `model_checkpoint_directory` —— 包含嵌入的检查点的目录

`EmbeddingInfo`包含

+   `tensor_name` —— 如果嵌入使用了特殊的张量名称

+   `metadata_path` —— 包含嵌入标签的 TSV 文件的路径

要查看可用配置的完整列表，请参考 [`mng.bz/J2Zo`](http://mng.bz/J2Zo) 上的文件。在其当前状态下，投影仪的配置不支持太多的定制。因此，我们将保持默认设置。我们将在`EmbeddingInfo`配置中设置一个配置，即`metadata_path`。我们将`metadata_path`设置为包含令牌的文件，最后将其传递给`projecter.visualize_embeddings()`函数。我们给它一个日志目录，投影仪将自动检测 TensorFlow 检查点并加载它。

## 启动 TensorBoard

我们一切都准备就绪。在您的笔记本上，执行以下行以打开 TensorBoard：

```py
%tensorboard --logdir logs/embeddings/ --port 6007
```

要可视化词向量，它们需要在`--logdir`指向的确切目录中（即不在嵌套文件夹中）。因此，我们需要一个新的 TensorBoard 服务器。这行代码将在端口 6007 上打开一个新的 TensorBoard 服务器。图 14.13 描述了在 TensorBoard 中显示的内容。

### 关于`%tensorboard`魔术命令的有趣事实

`%tensorboard`魔术命令足够智能，能够知道何时打开新的 TensorBoard 服务器以及何时不需要。如果您一遍又一遍地执行相同的命令，它将重用现有的 TensorBoard。但是，如果您执行带有不同`--logdir`或`--port`的命令，它将打开一个新的 TensorBoard 服务器。

![14-13](img/14-13.png)

图 14.13 在 TensorBoard 上的词向量视图。您可以选择使用哪种降维算法（以及参数）来获取词向量的二维或三维表示。在可视化中悬停在点上将显示由该点表示的单词。

## 深入分析

您可以在可视化中悬停在显示的点上，它们将显示它们代表的语料库中的哪个单词。您可以通过切换维度控制器来可视化二维或三维空间中的词向量。您可能想知道我们选择的词向量。它们最初有 50 个维度 —— 我们如何在二维或三维空间中可视化这样高维度的数据呢？有一套降维算法可以为我们做到这一点。一些示例是 t-SNE ([`mng.bz/woxO`](http://mng.bz/woxO))，PCA（主成分分析; [`mng.bz/ZAmZ`](http://mng.bz/ZAmZ))，以及 UMAP（Uniform Manifold Approximation and Projection; [`arxiv.org/pdf/1802.03426.pdf`](https://arxiv.org/pdf/1802.03426.pdf))。参考附带的链接以了解更多关于这些算法的信息。

您可以在 TensorBoard 上做的不仅仅是词向量的简单可视化。您可以通过突出显示可视化中的特定单词进行更详细的分析。为此，您可以使用正则表达式。例如，图 14.14 中显示的可视化是使用正则表达式`(?:fred|larry|mrs\.|mr\.|michelle|sea|denzel|beach|comedy|theater|idiotic|sadistic|marvelous|loving|gorg|bus|truck|lugosi)`生成的。

![14-14](img/14-14.png)

图 14.14 在可视化中搜索单词。您可以使用正则表达式来搜索单词的组合。

这就结束了我们关于 TensorBoard 的讨论。在下一章中，我们将讨论 TensorFlow 如何帮助我们轻松创建机器学习流水线并部署模型。

## 练习 5

如果您想在 TensorBoard 中显示单词向量时包含唯一标识符，而不仅仅是单词本身，例如，您想要看到“loving; 218”而不是“loving”，其中 218 是给予该单词的唯一标识符。为此，您需要更改写入`metadata.tsv`文件的内容。不仅仅是单词，每行上都写一个用分号分隔的递增 ID。例如，如果单词是`["a", "b", "c"]`，那么新行应该是`["a;1", "b;2", "c;3"]`。您如何进行更改？

## 摘要

+   TensorBoard 是一个用于可视化数据（例如图像）和实时跟踪模型性能的优秀工具。

+   在使用 Keras 构建模型时，您可以使用方便的`tf.keras.callbacks.TensorBoard()`回调来记录模型性能、层激活直方图等。

+   如果您有自定义指标想要记录到 TensorBoard 中，您可以在`tf.summary`命名空间中使用相应的数据类型（例如，如果您想要记录随时间变化的模型精度等，可以使用`tf.summary.scalar()`）。

+   每次将信息记录到 TensorBoard 中的会话称为一次运行。您应该为不同的运行制定一个可读且健壮的命名约定。一个好的命名约定应该捕捉您所做的主要更改以及运行执行的日期/时间。

+   TensorBoard Profile 提供了各种各样的性能分析结果（使用 NVIDIA 的`libcupti`库），例如模型训练过程中各个子任务所花费的时间（例如，设备计算时间、主机计算时间、输入时间等）、模型使用的内存以及各种操作是如何进行的顺序视图。

+   TensorBoard 是一个用于可视化高维数据（如图像和单词向量）的强大工具。

## 练习答案

### 练习 1

```py
image_writer = tf.summary.create_file_writer(image_logdir)

with image_writer.as_default():
    for bi, batch in enumerate(steps_image_batches):
        tf.summary.image(
            "batch_{}".format(bi), 
            batch, 
            max_outputs=10, 
            step=bi
        )
```

### 练习 2

```py
log_dir = "./logs "

classif_model.compile(
    loss='binary_crossentropy', 
    optimizer='adam', 
    metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, profile_batch=0
)

classif_model.fit(tr_ds, validation_data=v_ds, epochs=10, callbacks=[tb_callback])
```

### 练习 3

```py
writer = tf.summary.create_file_writer(log_dir)

x_n_minus_1 = 1
x_n_minus_2 = 0

with writer.as_default():        
    for i in range(100):
        x_n = x_n_minus_1 + x_n_minus_2
        x_n_minus_1 = x_n
        x_n_minus_2 = x_n_minus_1

        tf.summary.scalar("fibonacci", x_n, step=i)

        writer.flush()
```

### 练习 4

1.  主机上正在进行大量的计算。这可能是因为设备（例如，GPU）的内存不足。使用混合精度训练将有助于缓解这个问题。此外，可能有太多无法在 GPU 上运行的非 TensorFlow 代码。为此，使用更多的 TensorFlow 操作并将这样的代码转换为 TensorFlow 将获得加速。

2.  内核启动时间增加了。这可能是因为工作负载严重受限于 CPU。在这种情况下，我们可以合并`TF_GPU_THREAD_MODE`环境变量，并将其设置为`gpu_private`。这将确保有几个专用线程用于为 GPU 启动内核。

3.  输出时间显著偏高。这可能是因为频繁向磁盘写入过多输出。为解决此问题，我们可以考虑将数据在内存中保存更长时间，并仅在少数时刻将其刷新到磁盘上。

### 练习 5

```py
log_dir=os.path.join('logs', 'embeddings')

weights = tf.Variable(df_common.values)
checkpoint = tf.train.Checkpoint(embedding=weights)
checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

with open(os.path.join(log_dir, 'metadata.tsv'), 'w') as f:
    for i, w in enumerate(df_common.index):
        f.write(w+'; '+str(i)+'\n')
```
