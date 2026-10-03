
使用此变体，`DistilBERT`可用于解决多项选择问题。输入包括一个问题和几个答案。这些通常组合成单个序列（即，`[CLS] [Question] [SEP] [Answer 1] [SEP] [Answer 2] [SEP]`等），并且模型被要求预测问题的最佳答案，通常通过将`[CLS]`标记的表示传递给分类层来完成，该分类层将预测正确答案的索引（例如，第一个答案、第二个答案等）。

**TFDistilBertForTokenClassification**

命名实体识别或词性标注等任务需要模型为输入序列中的每个标记预测一个标签（例如，人物、组织等）。对于这样的任务，可以简单地使用这种类型的模型。

**TFDistilBertForQuestionAnswering**

这是我们场景中将使用的模型。我们有一个问题和一个上下文，模型需要预测答案（或答案在上下文中的起始/结束位置）。可以使用此模块解决此类问题。

表 13.1 总结了此侧边栏中的模型及其用法。

表 13.1 Hugging Face 的 transformers 库中不同模型及其用法总结

![13_table2](img/13_table2.png)

博客[`jalammar.github.io/illustrated-bert/`](http://jalammar.github.io/illustrated-bert/)提供了一张非常详细的图表，展示了 BERT-like 模型如何用于不同的任务（如“特定任务模型”部分所述）。

对这些模型的训练和评估将非常类似于您如果将它们用作 Keras 模型时的使用方式。只要数据格式符合模型期望的正确格式，您就可以对这些模型调用`model.fit()`、`model.predict()`或`model.evaluate()`。

要训练这些模型，您还可以使用高级`Trainer`（[`mng.bz/EWZd`](http://mng.bz/EWZd)）。我们将在稍后更详细地讨论`Trainer`对象。

这一部分是我在使用该库时的一个警示。通常，一旦定义了模型，你可以像使用 Keras 模型一样使用它。这意味着你可以使用`tf.data.Dataset`调用`model.fit()`并训练模型。当训练模型时，TensorFlow 和 Keras 期望模型输出为张量或张量元组。然而，Transformer 模型的输出是特定对象（继承自`transformers.file_utils.ModelOutput`对象），如 [`mng.bz/N6Rn`](http://mng.bz/N6Rn) 中所述。这将引发类似以下的错误：

```py
TypeError: The two structures don't have the same sequence type. 
Input structure has type <class 'tuple'>, while shallow structure has type 
<class 'transformers.modeling_tf_outputs.TFQuestionAnsweringModelOutput'>.
```

为了修复这个问题，`transformers`库允许你设置一个名为`return_dict`的特定配置，并确保模型返回一个元组，而不是一个对象。然后，我们定义一个`Config`对象，该对象具有`return_dict=False`，并用新的`config`对象覆盖模型的默认`config`。例如，对于`DistilBERT`模型，可以这样做：

```py
from transformers import DistilBertConfig, TFDistilBertForQuestionAnswering

config = DistilBertConfig.from_pretrained(
    "distilbert-base-uncased", return_dict=False
)
model = TFDistilBertForQuestionAnswering.from_pretrained(
    "distilbert-base-uncased", config=config
)
```

不幸的是，我无法通过使用这种配置使模型表现出我预期的行为。这表明，在编写代码时，即使使用一流的库，你也需要预料到可能出现的问题。最简单的解决方法是让`transformers`库输出一个`ModelOutput`对象，并编写一个包装函数，该函数将提取该对象的所需输出，并从这些输出创建一个`tf.keras.Model`。以下清单中的函数为我们执行了这个任务。

[将模型包装在 tf.keras.models.Model 对象中以防止错误](https://wiki.example.org/wrapping_the_model_in_a_tf_keras_model_object_to_prevent_errors)

```py
def tf_wrap_model(model):
    """ Wraps the huggingface's model with in the Keras Functional API """

    # Define inputs
    input_ids = tf.keras.layers.Input(
        [None,], dtype=tf.int32, name="input_ids"
    )                                                  ❶
    attention_mask = tf.keras.layers.Input(
      [None,], dtype=tf.int32, name="attention_mask"   ❷
    )

   out = model([input_ids, attention_mask])            ❸

   wrap_model = tf.keras.models.Model(
     [input_ids, attention_mask], 
     outputs=(out.start_logits, out.end_logits)        ❹
   )

   return wrap_model
```

❶ 定义一个输入层，该层将接收一个令牌序列的批次。

❷ 定义一个输入，用于编码时返回的 attention mask。

❸ 给定输入 id 和 attention_mask 获取模型输出。

❹ 定义一个包装模型，该模型以定义的输入作为输入，并输出开始和结束索引预测层的对数。

你可以通过调用清单 13.8 中的函数来生成生成校正输出的模型：

```py
model_v2 = tf_wrap_model(model)
```

最后，使用损失函数（稀疏分类交叉熵，因为我们使用整数标签）、度量（稀疏准确性）和优化器（Adam 优化器）对模型进行编译：

```py
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
acc = tf.keras.metrics.SparseCategoricalAccuracy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

model_v2.compile(optimizer=optimizer, loss=loss, metrics=[acc])
```

## 视觉中的 Transformer

Transformer 模型通常在自然语言处理领域表现出色。直到最近，人们才开始努力了解它们在计算机视觉领域的位置。然后谷歌和 Facebook AI 发表了几篇重要论文，研究了 Transformer 模型如何在计算机视觉领域中使用。

### 视觉 Transformer（ViT）

想法是将图像分解成 16×16 个小块，并将每个小块视为一个单独的令牌。每个图像路径被展平为一个 1D 向量，并使用位置编码机制对其位置进行编码，类似于原始的 Transformer。需要注意的是，原始 Transformer 中的位置编码是 1D 的。然而，图像是 2D 的。作者认为 1D 的位置编码已足够，且 1D 和 2D 位置编码之间没有太大的性能差异。一旦图像被分成 16×16 的块并展平，每个图像就可以被表示为一个令牌序列，就像文本输入序列一样。

然后，该模型以一种自监督的方式在一个名为 JFT-300M（[`paperswithcode.com/dataset/jft-300m`](https://paperswithcode.com/dataset/jft-300m)）的视觉数据集中进行预训练。在视觉中制定自监督任务并不是一件微不足道的事情，就像在 NLP 中那样。在 NLP 中，您可以简单地形成一个目标，即预测文本序列中的掩蔽令牌。但是，在计算机视觉的上下文中，令牌是一系列连续的值（归一化的像素值）。因此，`ViT`预先训练以预测给定图像补丁的平均三位 RGB 颜色。每个通道（即红色，绿色和蓝色）都用三位表示（即每个位的值为 0 或 1），这给出 512 个可能或类。换句话说，对于给定的图像，补丁将被随机遮盖（使用与 BERT 相同的方法），并且要求模型预测该图像补丁的平均三位 RGB 颜色。

首先，Dosovitskiy 等人在 2020 年 10 月发表了题为“一张图片值 16x16 个单词：用于大规模图像识别的 transformers”（An Image Is Worth 16X16 Words: Transformers for Image Recognition at Scale）的文章（[`arxiv.org/pdf/2010.11929.pdf`](https://arxiv.org/pdf/2010.11929.pdf)）。这可以被认为是走向视觉变换器的第一步。在这篇论文中，作者将原始 Transformer 模型适应于计算机视觉，对架构进行了最小限度的修改。这个模型被称为视觉变换器（`ViT`）。

在预训练后，可以通过在`ViT`的顶部适配一个分类或回归头来针对特定任务的问题进行微调，就像 BERT 一样。`ViT`还在序列的开头有`[CLS]`令牌，该令牌将用作插入到`ViT`上面的下游视觉模型的输入表示。以下图示说明了`ViT`的机制。

`ViT`的原始代码是使用一个称为 Jax 的框架编写的（[`github.com/google-research/vision_transformer`](https://github.com/google-research/vision_transformer)）。然而，该模型有几个第三方 TensorFlow 包装器（例如[`github.com/emla2805/vision-transformer`](https://github.com/emla2805/vision-transformer)）。如果使用第三方版本，请确保阅读代码并验证正确性，因为第三方与原始研究团队没有关联。

![13-09-unnumb-1](img/13-09-unnumb-1.png)

视觉 Transformer（ViT）模型架构及其在下游图像分类任务中的使用方式

### 统一 Transformer（UniT）

随后，Facebook AI 的一篇更具突破性的论文问世，题为“Transformer 就是你所需要的一切：统一 Transformer 实现多模态多任务学习”（作者：Hu 等）（[`arxiv.org/pdf/2102.10772.pdf`](https://arxiv.org/pdf/2102.10772.pdf)）。该模型被称为统一 Transformer（`UniT`）。`UniT`可以在计算机视觉和自然语言处理领域执行大量任务，只需改变分类头即可。

即使模型很复杂，整体上还是很直观的。有三个 Transformer 模型。一个 Transformer 对图像输入（如果有）进行编码，以生成图像的编码表示。下一个 Transformer 对文本输入（如果有）进行编码。最后，另一个 Transformer 将任务索引作为输入，获取嵌入并将其传递给一个跨自注意力层，该层以连接的图像和文本编码作为查询和键，并将任务嵌入（经过自注意力层后）作为值。这类似于 Transformer 解码器在其编码器-解码器注意力层中使用最后一个编码器输出生成查询和键，并使用解码器的输入作为值。该模型在下一页的图中表示。

`UniT`在涉及八个数据集的七项任务上进行了评估。这些任务包括对象检测、视觉问答、对象注释和四个仅语言任务。这四个仅语言任务来自 GLUE 基准数据集，其中包括以下内容：

**任务**

*对象检测*—模型预测图像中存在的对象的矩形坐标（数据集 COCO 和 Visual Genome Detection [VGD]）。

*视觉问答*（VQAv2）—给定一幅图像和一个问题，然后模型预测问题的答案，该答案可以在图像中找到（数据集：VQAv2）。

*视觉蕴涵任务*—一种视觉蕴涵任务，其中给定一幅图像和一个文本序列，模型预测句子是否语义上蕴含图像（SNLI-VE）。

*问题自然语言推理*（QNLI）—通过从给定上下文中提取答案来回答问题。

*Quora 问题对*（QQP）—从给定的问题对中识别重复的问题。

*文本蕴涵*—文本蕴涵着重于预测句子 A 是否蕴含/与句子 B 矛盾/与句子 B 中性（数据集 MNLI）。

*情感分析*—为给定的评论/文本预测情感（正面/负面/中性）（数据集 Stanford Sentiment Treebank [SST-2]）。

![13-09-unnumb-2](img/13-09-unnumb-2.png)

UniT 的整体架构。该模型由三个组件组成：图像编码器、文本编码器和任务解码器。最后，有几个分类头安装在任务解码器之上。

接下来，我们将训练我们刚刚定义的模型。

### 13.3.4 训练模型

我们一直在耐心地解决这个问题，最终，我们可以训练我们刚刚定义的模型，使用之前创建的`train_dataset`数据集，并使用`valid_dataset`来监视模型的准确度：

```py
model_v2.fit(
    train_dataset, 
    validation_data=valid_dataset,    
    epochs=3
)
```

这将打印以下输出：
