
# 8.4.4 XLM

最后，作为一个有趣的例子，我们将尝试多语言语言生成。`XLM`（跨语言语言模型）是由 Facebook AI Research 的研究人员提出的基于 Transformer 的跨语言语言模型，可以生成和编码多种语言的文本。通过学习如何编码多语言文本，模型可以用于不同语言之间的迁移学习。我们将在第九章介绍迁移学习。

您可以通过以下方式初始化分词器和模型，并使用预训练权重进行初始化：

```py
tokenizer = AutoTokenizer.from_pretrained('xlm-clm-enfr-1024')
model = AutoModelWithLMHead.from_pretrained('xlm-clm-enfr-1024')
```

在这里，我们加载一个使用英语和法语（`enfr`）进行训练的 `XLM` 模型（`xlm`），使用*因果语言模型*（`CLM`）*目标*（`clm`）进行训练。`CLM` 只是以更高级的方式描述我们在本章中所做的内容——根据前缀预测下一个标记。`XLM` 通常用于对多语言文本进行编码，用于一些下游任务，如文本分类和机器翻译，但我们只是将其用作生成文本的语言模型。有关使用 `XLM` 生成多语言文本的代码片段，请参见清单 8.2。您可以再次重用大部分之前的代码片段，尽管您还需要指定您正在使用的语言（请注意 `lang = 0` 行）。此外，在这里，我们通过仅提供 `BOS` 标记（其索引为零）从头开始生成文本。

**清单 8.2 生成多语言文本与 XLM**

```py
generated = [0] # start with just <s>
context = torch.tensor([generated])
lang = 0 # English

for i in range(100):
    langs = torch.zeros_like(context).fill_(lang)
    output = model(context, langs=langs)
    token = sample_token(output)

    generated.append(token.item())
    context = torch.tensor([generated])

print(tokenizer.decode(generated))
```

运行这个代码后，我得到了以下结果：

```py
<s>and its ability to make decisions on the basis of its own. " </s>The government has taken no decisions on that matter, " Mr Hockey said. </s>A lot of the information is very sensitive. </s>The new research and information on the Australian economy, which is what we're going to get from people, and the information that we are going to be looking at, we're going to be able to provide and we 'll take it forward. </s>I'm not trying to make sure we're not
```

然后，让我们将语言更改为 1（表示法语），并再次运行相同的代码片段，这将给出下一段文本：

```py
<s></s>En revanche, les prix des maisons individuelles n' ont guère augmenté ( - 0,1 % ). </s>En mars dernier, le taux de la taxe foncière, en légère augmentation à la hausse par rapport à février 2008\. </s>" Je n' ai jamais eu une augmentation " précise ". </s>" Je me suis toujours dit que ce n' était pas parce que c' était une blague. </s>En effet, j' étais un gars de la rue " </s>Les jeunes sont des gens qui avaient beaucoup d' humour... "
```

尽管这种生成质量不如我们之前实验的 `GPT-2` 那么好，但是看到一种单一模型可以同时生成英语和法语的文本非常令人耳目一新。如今，构建基于 Transformer 的多语言 NLP 模型以解决多种语言的 NLP 问题和任务越来越普遍。这也得益于 Transformer 对语言复杂性建模的强大能力。

# 8.5 案例研究：拼写检查器

在本章的最后一节中，我们将使用 Transformer 构建一个实用的 NLP 应用——拼写检查器。在现代世界中，拼写检查器无处不在。你的 Web 浏览器可能装备有一个拼写检查器，它会在拼写错误的单词下划线提示你。许多字处理器和编辑器也默认运行拼写检查器。一些应用程序（包括 Google Docs 和 Microsoft Word）甚至指出简单的语法错误。你是否想知道它们是如何工作的？我们将学习如何将其作为 NLP 问题进行规划、准备数据集、训练和改进模型。

## 8.5.1 拼写纠正作为机器翻译

拼写检查器接收这样一个文本："tisimptant too spll chck ths dcment"，检测任何拼写和语法错误，并修复所有错误："It's important to spell-check this document." 如何使用自然语言处理技术解决这个任务？这些系统如何实现？

最简单的方法是将输入文本分词为单词，并检查每个单词是否在字典中。如果不在，你可以查找距离最近的有效单词并替换它。可以使用一些度量（如编辑距离）来计算距离。重复这个过程，直到没有需要更正的单词。这种逐个单词修正的算法被许多拼写检查器广泛使用，因为它很简单。

然而，这种类型的拼写检查器有几个问题。首先，就像示例中的第一个单词"tisimptant"一样，您如何知道句子的哪一部分实际上是一个单词？我副本中默认的微软 Word 拼写检查器指出它是"disputant"的拼写错误，尽管对于任何英语使用者来说，它实际上是两个（或更多）单词的拼写错误是显而易见的。用户还可能拼写标点符号（包括空格），这使得一切都变得复杂。其次，仅仅因为某个单词在词典中存在，并不意味着它就没有错误。例如，示例中的第二个单词"too"是"to"的拼写错误，但两者都是任何英语词典中都有的有效单词。您如何判断前者在这种情况下是错误的呢？第三，所有这些决定都是在没有上下文的情况下做出的。我尝试过的一个拼写检查器在这个例子中显示"thus"是替换"ths"的候选词之一。然而，从这个上下文（名词之前）来看，"this"是一个更合适的候选词是显而易见的，尽管"this"和"thus"都与"ths"相隔一个编辑距离，这意味着根据编辑距离来看，它们都是同样有效的选项。

通过添加一些启发式规则，您可以解决其中一些问题。例如，"too"更有可能是动词之前"to"的拼写错误，"this"更有可能出现在名词之前而不是"thus"。但这种方法显然不具备可扩展性。还记得第 1.1.2 节中可怜的初级开发者吗？语言广阔而充满异常。您不能仅仅通过编写这些简单单词的规则来处理语言的全部复杂性。即使您能够为这些简单单词编写规则，您又如何知道"tisimptant"实际上是两个单词呢？您会尝试在每个可能的位置拆分这个单词，看拆分后的单词是否与现有单词相似吗？如果输入的是一种没有空格的语言，比如中文和日语，会怎么样呢？

此时，您可能意识到这种"拆分和修复"的方法行不通。一般来说，在设计自然语言处理应用程序时，您应该从以下三个方面考虑：

- **任务**——正在解决什么任务？是分类、序列标注还是序列到序列问题？
- **模型**——您将使用什么模型？是前馈网络、循环神经网络还是 Transformer？
- **数据集**——您从哪里获取数据集来训练和验证您的模型？

根据我的经验，如今绝大多数自然语言处理应用程序都可以通过结合这些方面来解决。拼写检查器呢？因为它们以一段文本作为输入，并生成修复后的字符串，如果我们将其作为一个 `Seq2Seq` 任务使用 Transformer 模型来解决将会最直接。换句话说，我们将建立一个机器翻译系统，将带有拼写/语法错误的嘈杂输入转换为干净、无误的输出，如图 8.16 所示。您可以将这两个方面看作是两种不同的"语言"（或英语的"方言"）。

![CH08_F16_Hagiwara](img/CH08_F16_Hagiwara.png)

**图 8.16 将拼写检查器训练为将"嘈杂"的句子翻译成"干净"的句子的 MT 系统**

此时，您可能会想知道我们从哪里获取数据集。这通常是解决现实世界自然语言处理问题中最重要（也是最困难）的部分。幸运的是，我们可以使用公共数据集来完成这项任务。让我们深入研究并开始构建一个拼写检查器。

## 8.5.2 训练拼写检查器

我们将使用 GitHub Typo Corpus（[`github.com/mhagiwara/github-typo-corpus`](https://github.com/mhagiwara/github-typo-corpus)）作为训练拼写检查器的数据集。这个数据集是由我和我的合作者创建的，其中包含数十万个从 GitHub 自动收集的"打字错误"编辑。这是迄今为止最大的拼写错误及其校正数据集，这使得它成为训练拼写检查器的理想选择。

在准备数据集和训练模型之前，我们需要做出一个决定，那就是选择模型操作的原子语言单位。许多自然语言处理模型使用令牌作为最小单位（即，`RNN`/`Transformer` 被馈送一个令牌序列），但越来越多的自然语言处理模型使用*单词或句子片段*作为基本单位（第 10.4 节）。对于拼写校正，我们应该使用什么作为最小的单位？与许多其他自然语言处理模型一样，起初使用单词作为输入听起来像是一个很好的"默认"选择。然而，正如我们之前所看到的，令牌的概念并不适用于拼写校正——用户可能会弄乱标点符号，如果您正在处理令牌，这会使一切过于复杂。更重要的是，因为自然语言处理模型需要操作一个固定的词汇表，所以拼写校正器的词汇表需要包含训练期间遇到的每个单词的每个拼写错误。这将使得训练和维护这样一个自然语言处理模型变得不必要昂贵。

由于这些原因，我们将使用*字符*作为拼写检查器的基本单位，就像在第 5.6 节中一样。使用字符有几个优点——它可以保持词汇表的大小相当小（通常对于具有小字母表集的语言，如英语，不到一百个）。即使是充满打字错误的嘈杂数据集，您也不必担心膨胀您的词汇表，因为打字错误只是字符的不同排列。您还可以将标点符号（甚至空白符）视为词汇表中的字符之一。这使得预处理步骤非常简单，因为您不需要任何语言工具包（如标记器）来执行此操作。

**注意**：使用字符并非没有缺点。其中一个主要问题是使用它们会增加序列的长度，因为你需要将所有内容分解为字符。这使得模型变得庞大且训练速度变慢。

首先，让我们为训练拼写检查器准备数据集。构建拼写检查器所需的所有必要数据和代码都包含在此代码库中：[`github.com/mhagiwara/xfspell`](https://github.com/mhagiwara/xfspell)。经过分词和拆分的数据集位于 `data/gtc` 目录下（如 `train.tok.fr`、`train.tok.en`、`dev.tok.fr`、`dev.tok.en`）。后缀 `en` 和 `fr` 是机器翻译中常用的约定，其中"fr"表示"外语"，"en"表示英语，因为许多机器翻译研究项目最初是由希望将某种外语翻译为英语的人发起的。这里，我们将"fr"和"en"仅仅解释为"拼写纠错前的嘈杂文本"和"拼写纠错后的纠正文本"。

图 8.17 显示了根据 GitHub Typo Corpus 创建的拼写纠错数据集的摘录。请注意，文本被分割成单个字符，甚至包括空格（由"_"替换）。所有不在通用字母表（大写字母、小写字母、数字和一些常见标点符号）内的字符都被替换为"#"。您可以看到数据集包含各种纠正，包括简单的拼写错误（`pubilc`->`public` 在第 670 行，`HYML`->`HTML` 在第 672 行），更复杂的错误（`mxnet` 一词替换成 `mxnet is not` 在第 681 行，`22th`->`22nd` 在第 682 行），甚至不带任何更正的行（第 676 行）。这看起来是训练拼写检查器的一个好资源。

![CH08_F17_Hagiwara](img/CH08_F17_Hagiwara.png)

**图 8.17 拼写纠错的训练数据**

训练拼写检查器（或任何其他 `Seq2Seq` 模型）的第一步是对数据集进行预处理。因为数据集已经分割和格式化，你只需要运行 `fairseq-preprocess` 将数据集转换为二进制格式，操作如下：

```py
fairseq-preprocess --source-lang fr --target-lang en \
    --trainpref data/gtc/train.tok \
    --validpref data/gtc/dev.tok \
    --destdir bin/gtc
```

然后，您可以使用以下代码立即开始训练模型。

**列表 8.3 训练拼写检查器**

```py
fairseq-train \
    bin/gtc \
    --fp16 \
    --arch transformer \
    --encoder-layers 6 --decoder-layers 6 \
    --encoder-embed-dim 1024 --decoder-embed-dim 1024 \
    --encoder-ffn-embed-dim 4096 --decoder-ffn-embed-dim 4096 \
    --encoder-attention-heads 16 --decoder-attention-heads 16 \
    --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.997)' --adam-eps 1e-09 --clip-norm 25.0 \
    --lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 16000 \
    --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
    --weight-decay 0.00025 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
    --max-tokens 4096 \
    --save-dir models/gtc01 \
    --max-epoch 40
```

您不需要担心这里的大多数超参数——这组参数对我来说效果还不错，尽管可能还有其他参数组合效果更好。但是，您可能想注意一些与模型大小相关的参数，即：

- 层数（`-[encoder|decoder]-layers`）
- 自注意力的嵌入维度（`-[encoder|decoder]-embed-dim`）
- 前馈层的嵌入维度（`-[encoder/decoder]-ffn-embed-dim`）
- 注意力头数（`-[encoder|decoder]-attention-heads`）
