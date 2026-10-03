
从最初创建一组简单的词袋转换字典开始，我们已经取得了巨大的进步，对此我们感到非常高兴。标记化本质上包括两个主要步骤：一个步骤是将文本分割开，另一个步骤是将它转换成数字。最明显的标记化形式是将字符串根据空白字符分割，然后根据词到整数的字典将其转换为数字。

这对大多数印欧语系语言使用者来说是有意义的，但我们不能推荐这种方法，因为它基于两个预设的假设：字母表和空白字符。当你遇到不使用字母表的语言，比如中文时，你会怎么做？当你遇到像匈牙利语或土耳其语这样的语言，它们不使用与英语相同的空白字符时，你会怎么做？或者代码——空白字符对 Python 的语法至关重要，它不仅仅是分隔符；它还具有语义意义。这就是为什么多语言模型几乎在所有情况下都能在相同任务上优于单语言模型：它们被迫在没有简单标记化障碍的情况下学习更深层次的意义表示。因此，让我们看看一些适用于 UTF-8 编码语言的更深入的方法。

这里是当前所有流行的基础标记化选项的示例：

+ *基于词*——“Johannes Gutenberg”变为`['Johannes', 'Gutenberg']`。
+ *基于字符* — “莎士比亚”变为`['S','h','a','k','e','s','p','e', 'a','r','e']`。
+ *基于子词* — “The quick red Delphox jumped over the lazy brown Emolga”变为`['the','quick','red','delph','ox','jump','ed','over','the', 'laz','y','brown','emol','ga']`

让我们逐一看看它们。

### 基于单词

基于单词的标记器通常在空白处分割，但还有其他方法，比如使用正则表达式、字典或标点符号。例如，基于标点的处理方法会将“ It’s the truth!”分割为`['It', '`'`', 's', 'the', 'truth', '!']`，这比仅基于空白分割提供了稍微更好的上下文。NLTK 中的`TreebankWordTokenizer`是一个正则表达式标记器的例子。基于单词的标记器相对容易实现，但需要我们保持一个难以管理的巨大字典，以映射编码每一个可能的单词。这是不合理的，所以通常你会在模型遇到未识别的单词时实现字典截断并返回未知标记，使其工作。这使得标记器在许多任务上表现不佳，如代码、名称和实体识别，以及跨领域的泛化。

### 基于字符

基于字符的编码方法是最直接且最容易实现的，因为我们是在 UTF-8 字符编码上进行分割。使用这种方法，我们只需要极小的字典来将字符映射到数字，这意味着我们可以避免未知标记和相关问题的需要。然而，它伴随着主要的信息损失，并且无法保留文本的相关语法、语义或形态。

### 基于子词

就像灰姑娘和三只熊的故事一样，基于字符的标记器太硬，基于单词的标记器太软，而基于子词的标记器则恰到好处。基于子词的标记器已被证明是最佳选择，它是前两种方法的混合体。我们能够使用像基于字符的标记器一样的小型字典，但丢失的语义像基于单词的标记器那样少。它甚至还有额外的优势，包括一些形态信息。然而，关于单词应该如何分割以及在哪里分割的问题尚未解决，并且存在许多不同的方法和途径。最佳方法的选择将像所有其他与 LLMs 相关的事物一样，取决于任务。如果你没有明确的目标来指导你想要做什么，那么将来可能会有后果。

创建子词字典主要使用三种算法：BPE、WordPiece 和 Unigram。此外，SentencePiece，它是三种方法的结合，可以显式处理空白，也非常常见。讨论它们的工作原理超出了本书的范围，但作为一个专注于生产的书籍，你应该知道最流行的子词标记化方法是 BPE（GPT-x）和 SentencePiece（LlamaX）。

在列表 4.10 中，我们将介绍如何训练 BPE 和 SentencePiece 的定制版本，以便你能够直面（几乎）任何数据集。在阅读代码时，请注意我们在哪里训练标记器。特别是，你将想要调整三个关键参数：`vocab_size`、`min_frequency`和`special_tokens`。更大的词汇量意味着你的标记器将更加健壮，并且可能更擅长处理更多语言，但它会增加计算复杂性。最小频率决定了特定子词标记在数据集中出现多少次之前才会被添加到字典中。较大的值可以防止罕见且可能不重要的标记填充我们的字典，并防止我们学习到重要的罕见标记。最后，特殊标记相对简单，包括我们特别关注于模型训练的句法标记。

#### 列表 4.10：训练自己的子词标记器

```py
import os
from pathlib import Path

import transformers
from tokenizers import ByteLevelBPETokenizer, SentencePieceBPETokenizer
from tokenizers.processors import BertProcessing

paths = [str(x) for x in Path("./data/").glob("**/*.txt")]      #1
bpe_tokenizer = ByteLevelBPETokenizer()   #2

bpe_tokenizer.train(                      #2
    files=paths,                          #2
    vocab_size=52_000,                    #2
    min_frequency=2,                      #2
    show_progress=True,                   #2
    special_tokens=[                      #2
        "<s>",                            #2
        "<pad>",                          #2
        "</s>",                           #2
        "<unk>",                          #2
        "<mask>",                         #2
    ], #2
) #2

token_dir = "./chapters/chapter_4/tokenizers/bytelevelbpe/"
if not os.path.exists(token_dir):
    os.makedirs(token_dir)
bpe_tokenizer.save_model(token_dir)

bpe_tokenizer = ByteLevelBPETokenizer(
    f"{token_dir}vocab.json",
    f"{token_dir}merges.txt",
)

example_text = "This sentence is getting encoded by a tokenizer."
print(bpe_tokenizer.encode(example_text).tokens)  
# ['This', 'Ġsentence', 'Ġis', 'Ġgetting', 'Ġenc', \
# 'oded', 'Ġby', 'Ġa', 'Ġto', 'ken', 'izer', '.']
print(bpe_tokenizer.encode(example_text).ids)
# [2666, 5651, 342, 1875, 4650, 10010, 504, 265, \ 
# 285, 1507, 13035, 18]

bpe_tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", bpe_tokenizer.token_to_id("</s>")),
    ("<s>", bpe_tokenizer.token_to_id("<s>")),
)
bpe_tokenizer.enable_truncation(max_length=512)

special_tokens = [
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<cls>",
    "<sep>",
    "<mask>",
]
sentencepiece_tokenizer = SentencePieceBPETokenizer()    #3

sentencepiece_tokenizer.train(                           #3
    files=paths,                                         #3
    vocab_size=4000,                                     #3
    min_frequency=2,                                     #3
    show_progress=True,                                  #3
    special_tokens=special_tokens,                       #3
) #3

token_dir = "./chapters/chapter_4/tokenizers/sentencepiece/"
if not os.path.exists(token_dir):
    os.makedirs(token_dir)
sentencepiece_tokenizer.save_model(token_dir)

tokenizer = transformers.PreTrainedTokenizerFast(
    tokenizer_object=sentencepiece_tokenizer,
    model_max_length=512,
    special_tokens=special_tokens,
)                                     #4
tokenizer.bos_token = "<s>"
tokenizer.bos_token_id = sentencepiece_tokenizer.token_to_id("<s>")
tokenizer.pad_token = "<pad>"
tokenizer.pad_token_id = sentencepiece_tokenizer.token_to_id("<pad>")
tokenizer.eos_token = "</s>"
tokenizer.eos_token_id = sentencepiece_tokenizer.token_to_id("</s>")
tokenizer.unk_token = "<unk>"
tokenizer.unk_token_id = sentencepiece_tokenizer.token_to_id("<unk>")
tokenizer.cls_token = "<cls>"
tokenizer.cls_token_id = sentencepiece_tokenizer.token_to_id("<cls>")
tokenizer.sep_token = "<sep>"
tokenizer.sep_token_id = sentencepiece_tokenizer.token_to_id("<sep>")
tokenizer.mask_token = "<mask>"
tokenizer.mask_token_id = sentencepiece_tokenizer.token_to_id("<mask>")
tokenizer.save_pretrained(token_dir)   #5

print(tokenizer.tokenize(example_text))
# ['_This', '_s', 'ent', 'ence', '_is', '_', 'g', 'et', 'tin', 'g', '_'
# 'en', 'co', 'd', 'ed', '_', 'b', 'y', '_a', '_', 't', 'ok', 'en', 
# 'iz', 'er', '.']

print(tokenizer.encode(example_text))
# [814, 1640, 609, 203, 1810, 623, 70, \
# 351, 148, 371, 125, 146, 2402, 959, 632]
```

#1 初始化用于训练的文本

#2 训练一个字节对编码标记器

#3 训练一个 SentencePiece 标记器

#4 转换

#5 并保存以备后用！

在 BPE 和 SentencePiece 这两种方法中，我们发现我们几乎同样地使用它们。这主要取决于我们正在微调或作为特定项目基础的模型。在算法上，我们更倾向于 SentencePiece，因为它往往能提高在它之上训练的模型在几乎所有测试中的评估分数，而且它也更接近我们作为人类与形态学互动的方式。

总的来说，标记化会丢失信息，就像将语音转换为文本一样——即，词序（句法）和意义（语义）。关于一个数字是什么以及它如何与字母不同的所有信息在标记化后都完全消失了。为了规避潜在的语义和句法问题，我们需要为这些特征中的每一个创建一个近似值，并找出如何用数学方法在抽象中表示它们，以便将那种意义重新插入到标记化的向量中。为此，我们使用了嵌入。

### 4.4.2 嵌入

嵌入为标记化过程中生成的向量提供了意义。标记化文本只是将几乎任意分配（基于出现频率）的数字分配给字典，但至少它是一个模型可以摄入的格式。嵌入是下一步，其中创建并查找位置和语义编码，为模型提供额外的上下文，以便在如何（可能）完成所给任务方面做出决策。

嵌入存在不完美之处，原因有很多，但最相关的问题可能是这个理论问题：能否仅使用该集合的子集来表示该集合？在这种情况下，第一个集合是语言，一个或多个，第二个集合是数字、浮点数和数字。数学是用于描述我们接受为真的公理性事物的语言的一个子集。以英语字母表为例：能否仅使用 26 个字母中的某些部分来表示整个字母表？显然不能，但如果原始集合和子集都是无限的，会怎样呢？能否仅使用 0 到 1 之间的十进制数来表示所有数字？鉴于第一个是可数的无限集合，第二个是不可数的无限集合，答案是肯定的，这对语言建模领域应该是一个鼓舞人心的消息。

既然我们已经讨论了为什么不应该完全盲目地依赖嵌入，嵌入就是大多数企业在 LLMs 中寻求的东西。您不需要一个 1.7T 参数的模型来处理客户询问您的定价或通过您的文档进行搜索。正如我们在第二章中讨论的，如果比较的两个嵌入都是由同一模型在相同维度空间中创建的，嵌入具有通过距离进行比较的天生优势。这为各种快速计算和检索打开了大门，您永远不必担心如何托管一个巨大的模型，因为您可以在 CPU 上运行一个较小的嵌入模型，处理数百个标记只需毫秒。

目前，嵌入应用中最受欢迎且最酷的应用之一是检索增强生成（RAG），在这种应用中，你存储与模型整体任务相关的数据，并在提示时将数据的一部分提供给更大的模型，以改善结果。假设我们将 RAG 应用于波士顿住房数据集，并尝试预测新房屋的价值。在这种情况下，我们可以将那所房子的嵌入数据与该地区最相似的房屋进行比较，并在无需评估师验证的情况下生成有根据的评估，只要您检索到的嵌入是最新更新的。

嵌入可用于数十种不同的任务，并且是取自模型最终隐藏状态表示的结果。您模型的每一层都是一个潜在的选择，但普遍共识是在任何解码或最终线性层或 softmax 之前，从最终层提取表示。列表 4.11 提供了一个从 PyTorch 和 Hugging Face 模型中提取嵌入的实际示例。最佳实践规定，您应该使用计划用于推理的任何嵌入模型从文档中提取嵌入，尤其是如果这些嵌入最终将存储在 VectorDB 中。在创建我们的嵌入后，我们展示了如何对结果进行简单的相似度搜索，这是 RAG 系统的基础。

#### 列表 4.11 示例嵌入
