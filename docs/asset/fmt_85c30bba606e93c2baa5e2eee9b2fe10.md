
```py
response = predictor.predict(data="In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.")
```

响应内容应该类似于这样：

```py
b'In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English. The unicorns said they would take a stroll in the direction of scientists over the next month or so.\n\n\n\n\nWhen contacted by Animal Life and Crop.com, author Enrique Martinez explained how he was discovered and how the unicorns\' journey has surprised him. According to Martinez, the experience makes him more interested in research and game development.\n"This is really what I want to see this year, and in terms of medical research, I want to see our population increase."<|endoftext|>'
```

恭喜！我们刚刚完成了我们的旅程——我们从第二章开始构建一个 ML 模型，并在本章中一直部署到了云平台。

## 11.4 解释和可视化模型预测

人们经常谈论标准化数据集上的指标和排行榜表现，但分析和可视化模型预测和内部状态对于现实世界中的自然语言处理应用非常重要。尽管深度学习模型在其所做的事情上可能非常出色，在某些自然语言处理任务上甚至达到了人类水平的性能，但这些深度模型是黑盒，很难知道它们为什么会做出某些预测。

因为这种（有些令人不安的）深度学习模型的属性，人工智能中的一个日益增长的领域称为*可解释人工智能*（XAI）正在努力开发方法来解释机器学习模型的预测和行为。解释机器学习模型对于调试非常有用——如果您知道它为什么做出某些预测，它会给您很多线索。在一些领域，如医疗应用和自动驾驶汽车，使机器学习模型可解释对于法律和实际原因至关重要。在本章的最后一节中，我们将介绍一个案例研究，在该案例研究中，我们使用*语言可解释性工具*（LIT）（[`pair-code.github.io/lit/`](https://pair-code.github.io/lit/)）来可视化和解释自然语言处理模型的预测和行为。

LIT 是由 Google 开发的开源工具包，提供了一个基于浏览器的界面，用于解释和可视化 ML 预测。请注意，它是框架不可知的，这意味着它可以与任何选择的基于 Python 的 ML 框架一起使用，包括 AllenNLP 和 Hugging Face Transformers。LIT 提供了一系列功能，包括以下内容：

+   *显著性图*——以彩色可视化输入的哪部分对达到当前预测起到了重要作用

+   *聚合统计信息*——显示诸如数据集指标和混淆矩阵等聚合统计信息

+   *反事实*——观察模型对生成的新样本的预测如何变化

在本节的其余部分，让我们选择我们训练的 AllenNLP 模型之一（第九章中基于 BERT 的情感分析模型）并通过 LIT 进行分析。LIT 提供了一组可扩展的抽象，如数据集和模型，以使使用任何基于 Python 的 ML 模型更加轻松。

首先，让我们安装 LIT。可以通过以下 pip 调用一次性安装它：

```py
pip install lit-nlp
```

接下来，您需要使用 LIT 定义的抽象类包装您的数据集和模型。让我们创建一个名为 `run_lit.py` 的新脚本，并导入必要的模块和类，如下所示：

```py
import numpy as np

from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor
from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types

from examples.sentiment.sst_classifier import LstmClassifier
from examples.sentiment.sst_reader import StanfordSentimentTreeBankDatasetReaderWithTokenizer
```

下面的代码展示了如何为 LIT 定义一个数据集。在这里，我们创建了一个仅包含四个硬编码示例的玩具数据集，但在实践中，您可能想要读取要探索的真实数据集。记得定义返回数据集类型规范的 `spec()` 方法：

```py
class SSTData(lit_dataset.Dataset):
    def __init__(self, labels):
        self._labels = labels
        self._examples = [
            {'sentence': 'This is the best movie ever!!!', 'label': '4'},
            {'sentence': 'A good movie.', 'label': '3'},
            {'sentence': 'A mediocre movie.', 'label': '1'},
            {'sentence': 'It was such an awful movie...', 'label': '0'}
        ]

    def spec(self):
        return {
            'sentence': lit_types.TextSegment(),
            'label': lit_types.CategoryLabel(vocab=self._labels)
        }
```

现在，我们已经准备好定义主要模型了，如下所示。

列表 11.2 定义 LIT 的主要模型

```py
class SentimentClassifierModel(lit_model.Model):
    def __init__(self):
        cuda_device = 0
        archive_file = 'model/model.tar.gz'
        predictor_name = 'sentence_classifier_predictor'

        archive = load_archive(                                       ❶
            archive_file=archive_file,
            cuda_device=cuda_device
        )

        predictor = Predictor.from_archive(archive, predictor_name=predictor_name)

        self.predictor = predictor                                    ❷
        label_map = archive.model.vocab.get_index_to_token_vocabulary('labels')
        self.labels = [label for _, label in sorted(label_map.items())]

    def predict_minibatch(self, inputs):
        for inst in inputs:
            pred = self.predictor.predict(inst['sentence'])           ❸
            tokens = self.predictor._tokenizer.tokenize(inst['sentence'])
            yield {
                'tokens': tokens,
                'probas': np.array(pred['probs']),
                'cls_emb': np.array(pred['cls_emb'])
            }

    def input_spec(self):
        return {
            "sentence": lit_types.TextSegment(),
            "label": lit_types.CategoryLabel(vocab=self.labels, required=False)
        }

    def output_spec(self):
        return {
            "tokens": lit_types.Tokens(),
            "probas": lit_types.MulticlassPreds(parent="label", vocab=self.labels),
            "cls_emb": lit_types.Embeddings()
        }
```

❶ 加载 AllenNLP 存档

❷ 提取并设置预测器

❸ 运行预测器的 `predict` 方法

在构造函数（`__init__`）中，我们正在从存档文件中加载一个 AllenNLP 模型，并从中创建一个预测器。我们假设您的模型放在 `model/model.tar.gz` 下，并且硬编码了其路径，但根据您的模型位置随意修改此路径。

模型预测是在 `predict_minibatch()` 中计算的。给定输入（简单地是数据集实例的数组），它通过预测器运行模型并返回结果。请注意，预测是逐个实例进行的，尽管在实践中，您应考虑批量进行预测，因为这会提高对较大输入数据的吞吐量。该方法还返回用于可视化嵌入的预测类别的嵌入（作为 `cls_emb`），这将用于可视化嵌入（图 11.5）。

![CH11_F05_Hagiwara](img/CH11_F05_Hagiwara.png)

图 11.5 LIT 可以显示显著性图、聚合统计信息和嵌入，以分析您的模型和预测。

最后，这是运行 LIT 服务器的代码：

```py
model = SentimentClassifierModel()
models = {"sst": model}
datasets = {"sst": SSTData(labels=model.labels)}

lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
lit_demo.serve()
```

运行上面的脚本后，转到 `http:/./localhost:5432/` 在你的浏览器上。你应该会看到一个类似于图 11.5 的屏幕。你可以看到一系列面板，对应于有关数据和预测的各种信息，包括嵌入、数据集表和编辑器、分类结果以及显著性图（显示通过一种名为 *LIME* 的自动方法计算的标记贡献）³。

可视化和与模型预测进行交互是了解模型工作原理以及如何改进的好方法。

## 11.5 从这里开始去哪里

在本书中，我们只是浅尝了这个广阔而悠久的自然语言处理领域的表面。如果你对进一步学习 NLP 的实践方面感兴趣，*Natural Language Processing in Action*，作者是 Hobson Lane 和其他人（Manning Publications，2019），以及 *Practical Natural Language Processing*，作者是 Sowmya Vajjala 和其他人（O’Reilly，2020），可以成为下一个很好的步骤。*Machine Learning Engineering*，作者是 Andriy Burkov（True Positive Inc.，2020），也是学习机器学习工程主题的好书。

如果你对学习 NLP 的数学和理论方面更感兴趣，我建议你尝试一些流行的教材，比如 *Speech and Language Processing*，作者是 Dan Jurafsky 和 James H. Martin（Prentice Hall，2008）⁴，以及 *Introduction to Natural Language Processing*，作者是 Jacob Eisenstein（MIT Press，2019）。虽然 *Foundations of Statistical Natural Language Processing*，作者是 Christopher D. Manning 和 Hinrich Schütze（Cambridge，1999），有点过时，但它也是一本经典教材，可以为你提供广泛的 NLP 方法和模型打下坚实的基础。

也要记住，你通常可以免费在网上找到很棒的资源。一个免费的 AllenNLP 课程，“A Guide to Natural Language Processing with AllenNLP”（[`guide .allennlp.org/`](https://guide.allennlp.org/)），以及 Hugging Face Transformers 的文档（[`huggingface.co/transformers/index.html`](https://huggingface.co/transformers/index.html)）是学习这些库的深入了解的好地方。

最后，学习 NLP 最有效的方法实际上是自己动手。如果您的兴趣、工作或任何涉及处理自然语言文本的事情存在问题，请考虑您在本书中学到的任何技术是否适用。这是一个分类、标记还是序列到序列的问题？您使用哪些模型？您如何获得训练数据？您如何评估您的模型？如果您没有 NLP 问题，不用担心——请前往 Kaggle，在那里您可以找到许多与 NLP 相关的竞赛，您可以在处理真实世界问题时“动手”并获得 NLP 经验。NLP 会议和研讨会经常举办共享任务，参与者可以在共同任务、数据集和评估指标上进行竞争，这也是一个很好的学习方法，如果您想深入研究 NLP 的某个特定领域。

## 概要

+   在现实世界的 NLP/ML 系统中，机器学习代码通常只是一个小部分，支持着复杂的基础设施，用于数据收集、特征提取以及模型服务和监控。

+   NLP 模块可以开发为一次性脚本、批量预测服务或实时预测服务。

+   重要的是要对模型和数据进行版本控制，除了源代码。要注意训练和测试时间之间的训练服务偏差。

+   您可以使用 TorchServe 轻松提供 PyTorch 模型，并将其部署到 Amazon SageMaker。

+   可解释性人工智能是一个新的领域，用于解释和解释机器学习模型及其预测。您可以使用 LIT（语言可解释性工具）来可视化和解释模型预测。

^(1.)GCP: [`cloud.google.com/compute/docs/gpus/install-drivers-gpu;`](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu) AWS: [`docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html`](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html)。

^(2.)还有另一个叫做 AllenNLP Interpret 的工具包（[`allennlp.org/interpret`](https://allennlp.org/interpret)），它提供了一套类似的功能，用于理解 NLP 模型，尽管它专门设计用于与 AllenNLP 模型进行交互。

^(3.)Ribeiro 等人，“‘为什么我要相信你？’: 解释任何分类器的预测”（2016 年）。[`arxiv.org/abs/1602.04938`](https://arxiv.org/abs/1602.04938)。

^(4.)你可以免费阅读第三版（2021 年）的草稿，网址为[`web.stanford.edu/~jurafsky/slp3/`](https://web.stanford.edu/~jurafsky/slp3/)。
