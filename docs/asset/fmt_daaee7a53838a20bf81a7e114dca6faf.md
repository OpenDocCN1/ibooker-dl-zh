
或者，你可以使用 Hugging Face `Evaluate library` 提供的指标：它们被设计得与 `Transformers` 库配合得很好。或者，尽管 `Trainer API` 不支持来自 `TorchMetrics` 库的流式指标，但你仍然可以在一个函数中包装它们来使用。

接下来，我们必须在 `TrainingArguments` 对象中指定我们的训练配置：

```py
from transformers import TrainingArguments

train_args = TrainingArguments(
    output_dir="my_imdb_model", num_train_epochs=2,
    per_device_train_batch_size=128, per_device_eval_batch_size=128,
    eval_strategy="epoch", logging_strategy="epoch", save_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="accuracy",
    report_to="none")
```

我们指定日志和模型检查点必须保存在`my_imdb_model`目录中；训练应该运行 2 个 epoch（如果你想增加这个数字）；训练和评估的批量大小都是 128（你可以根据你的 VRAM 量进行调整）；我们希望在每个 epoch 结束时进行评估、记录和保存；并且根据验证准确率，在训练结束时加载最佳模型。最后，`report_to`参数允许你指定一个或多个训练代码将日志报告给的工具，例如 TensorBoard 或 [Weights & Biases (W&B)](https://wandb.ai)。这可以用来可视化学习曲线。为了简单起见，我将`report_to="none"`设置为关闭报告。

最后，我们创建一个`Trainer`对象，并将模型、训练参数、训练集和验证集、评估函数以及一个负责填充的数据合并器传递给它。最后，我们调用训练器的`train()`方法，这样就完成了！模型在仅经过两个 epoch 后，在验证集上的准确率达到了大约 90%：

```py
from transformers import DataCollatorWithPadding, Trainer

trainer = Trainer(
    bert_for_binary_clf, train_args, train_dataset=tok_imdb_train_set,
    eval_dataset=tok_imdb_valid_set, compute_metrics=compute_accuracy,
    data_collator=DataCollatorWithPadding(bert_tokenizer))
train_output = trainer.train()
```

太好了，你现在知道如何下载像 BERT 这样的预训练模型并在自己的数据集上进行微调了！但如果你根本没有任何数据集，只想使用已经针对情感分析微调过的预训练模型怎么办？为此，你可以使用`pipelines API`。

## Hugging Face Pipelines

`Transformers` 库提供了一个非常方便的 API，用于下载和使用各种任务的预训练管道。每个管道都包含一个预训练模型以及相应的预处理和后处理模块。例如，让我们创建一个情感分析管道，并在训练集中的前 10 条 IMDb 评论上运行它：

```py
>>> from transformers import pipeline
>>> model_name = "distilbert-base-uncased-finetuned-sst-2-english"
>>> classifier_imdb = pipeline("sentiment-analysis", model=model_name,
...                            truncation=True, max_length=512)
...
>>> classifier_imdb(train_reviews[:10])
[{'label': 'POSITIVE', 'score': 0.9996108412742615},
 {'label': 'POSITIVE', 'score': 0.9998623132705688},
 [...]
 {'label': 'POSITIVE', 'score': 0.9978922009468079},
 {'label': 'NEGATIVE', 'score': 0.9997020363807678}]
```

好吧，这几乎不可能更简单了，对吧？只需通过指定任务和要使用的模型以及一些其他参数来创建一个管道，然后就可以开始了！在这个例子中，每条评论都会得到一个`"POSITIVE"`或`"NEGATIVE"`标签，以及一个等于模型对该标签估计概率的分数。这个特定的模型在验证集上的准确率实际上达到了 88.2%，这是一个相当不错的成绩。以下是一些需要注意的点：

+   如果你没有指定模型，`pipeline()`函数将使用所选任务的默认模型。对于情感分析，在撰写本文时，它是我们选择的模型：这是一个 DistilBERT 模型——BERT 的缩小版——带有非大小写分词器，在英语维基百科和一组英语书籍上训练，并在斯坦福情感树库 v2 (SST 2)任务上进行微调。

+   如果你有 GPU，管道会自动使用它。如果你有多个 GPU，你可以通过将管道的`device`参数设置为 GPU 索引来指定使用哪一个。

+   `Transformers` 库中的模型默认总是处于评估模式（无需调用`eval()`）。

+   分数是针对所选标签的，而不是针对正类。特别是，由于这是一个二元分类任务，分数不能低于 0.5（否则模型会选择另一个标签）。

我们选择的是适用于通用情感分析的模型，例如电影评论，但其他模型更适合特定的用例，例如社交媒体帖子（例如，在大型推文数据集上训练，然后在情感分析数据集上进行微调）。为了找到最适合您用例的最佳模型，您可以在 [*Hugging Face Hub*](https://huggingface.co/models) 上搜索可用的模型列表。然而，仅“文本分类”类别中就有超过 80,000 个模型可供选择，因此您需要使用过滤器来缩小选项范围。特别是，从任务开始过滤，并按趋势或最受欢迎的模型排序。如果需要，您还可以继续按语言和数据集进行过滤。优先选择来自信誉良好的来源的模型（例如，来自 huggingface、facebook、google、cardiffnlp 等用户的模型），并且如果模型包含可执行代码，请确保您信任该用户（如果您信任，请在调用`pipeline()`函数时设置`trust_remote_code=True`）。

除了情感分析之外，还有许多其他文本分类任务。例如，在多体裁自然语言推理（MultiNLI）数据集上微调的模型可以将一对文本（每个文本以分隔标记`[SEP]`结束）分类为三个类别：矛盾（如果文本相互矛盾）、蕴涵（如果第一个文本蕴涵第二个文本），或者中性。例如：

```py
>>> model_name = "huggingface/distilbert-base-uncased-finetuned-mnli"
>>> classifier_mnli = pipeline("text-classification", model=model_name)
>>> classifier_mnli([
...     "She loves me. [SEP] She loves me not. [SEP]",
...     "Alice just woke up. [SEP] Alice is awake. [SEP]",
...     "I like dogs. [SEP] Everyone likes dogs. [SEP]"])
...
[{'label': 'contradiction', 'score': 0.9717152714729309},
 {'label': 'entailment', 'score': 0.9119168519973755},
 {'label': 'neutral', 'score': 0.9509281516075134}]
```

通过管道 API 还可以通过许多其他 NLP 任务，例如问答、摘要、句子相似度、文本生成、标记分类、翻译等。而且这还远不止于此！还有许多计算机视觉任务，例如图像分类、图像分割、目标检测、图像到文本、文本到图像、深度估计，甚至音频任务，例如音频分类、语音到文本、文本到语音等。请确保查看 [*https://huggingface.co/tasks*](https://huggingface.co/tasks) 上的完整列表。

###### 警告

在下载模型之前，请确保您信任托管平台（例如，Hugging Face Hub）和模型的作者：模型可能包含可执行代码，这可能是有害的。它也可能产生有偏见的输出，或者它可能使用受版权保护或敏感的私人数据进行训练，这些数据可能泄露给您的用户，或者它甚至可能包含*有毒权重*，这可能导致它仅对某些类型的输入产生有害内容（例如，宣传），而其他情况下则表现正常。

是时候回顾一下了。到目前为止，我们已经探讨了使用 char-RNN 进行文本生成，以及使用各种子词标记化方法、预训练嵌入甚至整个预训练模型进行情感分析。在这个过程中，我们讨论了嵌入、标记化器和 Hugging Face 库。在下一节中，我们将探索另一个重要的 NLP 任务：*神经机器翻译*（NMT）。具体来说，我们将构建一个能够将英语翻译成西班牙语的编码器-解码器模型，并了解如何使用束搜索和注意力机制来提高其性能。¡Vamos!

# 用于神经机器翻译的编码器-解码器网络

让我们从相对简单的 [序列到序列 NMT 模型](https://homl.info/nmtmodel)⁠^(15) 开始，该模型将英语文本翻译成西班牙语（见图 图 14-5）。

![编码器-解码器网络图，展示了将“我喜欢足球”从英语翻译成西班牙语为“我喜欢足球”的序列到序列模型](img/hmls_1405.png)

###### 图 14-5. 一个简单的机器翻译模型

简而言之，架构如下：英语文本作为输入传递给编码器，解码器输出西班牙语翻译。请注意，在训练期间，西班牙语翻译也被用作解码器的输入，但向后移动了一步。换句话说，在训练期间，解码器被提供的输入是它在前一步应该输出的标记，而不考虑它实际上输出了什么。这被称为*教师强制*——一种显著加快训练速度并提高模型性能的技术。对于第一个标记，解码器被提供起始序列（SoS，也称为序列开始，BoS）标记（`"<s>"`），并且解码器被期望以结束序列（EoS）标记（`"</s>"`）结束文本。

每个标记最初由其 ID 表示（例如，标记“足球”的 ID 为`4553`）。接下来，一个`nn.Embedding`层返回标记嵌入。这些标记嵌入随后被输入到编码器和解码器中。

在每一步，解码器的密集输出层（即一个`nn.Linear`层）为输出词汇表中的每个标记（即西班牙语）输出一个 logit 分数。如果你将这些 logit 通过 softmax 函数传递，你会得到每个可能标记的估计概率。例如，在第一步，单词“我”可能有一个 7%的概率，“你”可能有一个 1%的概率，依此类推。这非常像一项常规的分类任务，实际上，我们将使用`nn.CrossEntropyLoss`来训练模型，就像我们在 char-RNN 模型中所做的那样。

注意，在推理时间（训练后），你将没有目标文本可以输入到解码器中。相反，你需要输入它在前一步刚刚输出的单词，如图 图 14-6 所示（这需要嵌入查找，图中未显示）。

![展示序列循环神经网络细胞图的示意图，其中每个细胞在推理期间输出一个标记，该标记作为下一个细胞的输入。](img/hmls_1406.png)

###### 图 14-6. 在推理时间，解码器将前一个时间步输出的单词作为输入

###### 小贴士

在一篇 [2015 年的论文](https://homl.info/scheduledsampling)⁠^(16) 中，Samy Bengio 等人提出，在训练过程中逐渐从向解码器提供前一个*目标*标记切换到提供前一个*输出*标记。

让我们构建和训练这个模型！首先，我们需要下载一个英语/西班牙语文本对的语料库。为此，我们将使用 `Dataset` 库从*Tatoeba Challenge*语料库下载英语/西班牙语文本对。The [Tatoeba 项目](https://tatoeba.org) 是由 Trang Ho 于 2006 年启动的一个语言学习倡议，贡献者创建了一个包含许多语言文本对的巨大集合。Tatoeba Challenge 语料库是由赫尔辛基大学的研究人员创建的，用于评估机器翻译系统，使用从 Tatoeba 项目提取的数据。训练集相当大，因此我们将使用验证集作为我们的训练集，留出 20%用于验证。我们还将下载测试集：

```py
nmt_original_valid_set, nmt_test_set = load_dataset(
    path="ageron/tatoeba_mt_train", name="eng-spa",
    split=["validation", "test"])
split = nmt_original_valid_set.train_test_split(train_size=0.8, seed=42)
nmt_train_set, nmt_valid_set = split["train"], split["test"]
```

数据集中的每个样本都是一个包含英语文本及其西班牙语翻译的字典。例如：

```py
>>> nmt_train_set[0]
{'source_text': 'Tom tried to break up the fight.',
 'target_text': 'Tom trató de disolver la pelea.',
 'source_lang': 'eng',
 'target_lang': 'spa'}
```

我们需要对这个文本进行分词。我们可以为英语和西班牙语使用不同的分词器，但这两种语言有很多共同词汇（例如，animal, color, hotel, hospital, idea, radio, motor），以及许多类似的子词（例如，pre, auto, inter, uni），因此使用一个共同的分词器是有意义的。让我们在所有训练文本上训练一个 BPE 分词器，包括英语和西班牙语：

```py
def train_eng_spa():  # a generator function to iterate over all training text
    for pair in nmt_train_set:
        yield pair["source_text"]
        yield pair["target_text"]

max_length = 256
vocab_size = 10_000
nmt_tokenizer_model = tokenizers.models.BPE(unk_token="<unk>")
nmt_tokenizer = tokenizers.Tokenizer(nmt_tokenizer_model)
nmt_tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
nmt_tokenizer.enable_truncation(max_length=max_length)
nmt_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
nmt_tokenizer_trainer = tokenizers.trainers.BpeTrainer(
    vocab_size=vocab_size, special_tokens=["<pad>", "<unk>", "<s>", "</s>"])
nmt_tokenizer.train_from_iterator(train_eng_spa(), nmt_tokenizer_trainer)
```

让我们来测试这个分词器：

```py
>>> nmt_tokenizer.encode("I like soccer").ids
[43, 401, 4381]
>>> nmt_tokenizer.encode("<s> Me gusta el fútbol").ids
[2, 396, 582, 219, 3356]
```

完美！现在让我们创建一个小的实用类，它将包含分词后的英语文本（即*源*标记 ID 序列），以及相应的分词后的西班牙语目标（即*目标*标记 ID 序列），再加上相应的注意力掩码。为此，我们可以创建一个`namedtuple`基类（即具有命名字段的元组），并扩展它以添加一个`to()`方法，这将使将所有这些张量移动到 GPU 变得容易：

```py
from collections import namedtuple

fields = ["src_token_ids", "src_mask", "tgt_token_ids", "tgt_mask"]
class NmtPair(namedtuple("NmtPairBase", fields)):
    def to(self, device):
        return NmtPair(self.src_token_ids.to(device), self.src_mask.to(device),
                       self.tgt_token_ids.to(device), self.tgt_mask.to(device))
```

接下来，让我们创建数据加载器：
