
# 第五章 自然语言处理简介

自然语言处理（NLP）是人工智能中的一种技术，它涉及对语言的理解。它包括编程技术来创建可以理解语言、分类内容，甚至生成和创建新语言作品的模型。它也是大型语言模型（LLMs）如 GPT、Gemini 和 Claude 的底层基础。我们将在后面的章节中探讨 LLMs，但首先，我们将在接下来的几章中探讨更基础的 NLP，以便为你即将学习的内容做好准备。

还有许多服务使用 NLP 来创建应用程序，如聊天机器人，但这不是本书的范围——相反，我们将探讨 NLP 的基础以及如何建模语言，以便你可以训练神经网络来理解和分类文本。在后面的章节中，你还将学习如何使用 ML 模型的预测元素来写一些诗歌。这不仅仅是为了好玩——这也是学习如何使用支撑生成式 AI 的基于 transformer 的模型的一个先导。

我们将从这个章节开始，探讨如何将语言分解为数字，以及如何使用这些数字在神经网络中使用。

## 将语言编码为数字

最终，计算机处理的是数字，因此为了处理语言，你需要将其转换为数字，这个过程称为*编码*。

你可以用多种方式将语言编码为数字。最常见的是按字母编码，就像在程序中存储字符串时自然发生的那样。然而，在内存中，你并不存储字母`a`，而是存储它的编码——可能是 ASCII 或 Unicode 值或其他。例如，考虑单词`listen`。你可以用 ASCII 将其编码为数字 76, 73, 83, 84, 69 和 78。这是好的，因为现在你可以用数字来表示这个单词。但是，再考虑一下单词`silent`，它是`listen`的字母重组。相同的数字代表这个单词，尽管顺序不同，这可能会使构建一个理解文本的模型变得更加困难。

可能更好的选择是使用数字来编码整个单词，而不是它们内部的字母。在这种情况下，`silent`可以是数字`x`，而`listen`可以是数字`y`，它们不会相互重叠。

使用这种技术，考虑一个句子，比如“我爱我的狗。”你可以用数字`[1, 2, 3, 4]`来编码它。如果你想要编码“我爱我的猫”，你可以用`[1, 2, 3, 5]`来编码。到现在，你可能已经到了可以判断这两个句子具有相似意义的地步，因为它们的数字相似——换句话说，`[1, 2, 3, 4]`看起来非常像`[1, 2, 3, 5]`。

代表单词的数字也被称为*标记*，因此这个过程被称为*标记化*。你将在下一节中了解如何在代码中实现它。

### 开始使用标记化

PyTorch 生态系统包含许多用于标记化的库，它将单词转换为标记。你可能在代码示例中看到的一个常见标记器是`torchtext`，但自 2023 年以来它已经被弃用。所以，在使用它时要小心，尤其是因为 PyTorch 版本在进步，但它没有。因此，一些替代方案是使用自定义标记器、来自其他地方的预训练标记器，或者（令人惊讶的是）来自 Keras 生态系统的标记器。

#### 使用自定义标记器

为了给你一个简单的例子，这里有一些我用来创建自定义标记器的代码，将一个小语料库（两个句子）的单词转换为标记：

```py
import torch

sentences = [
    'Today is a sunny day',
    'Today is a rainy day'
]

# Tokenization function
def tokenize(text):
    return text.lower().split()

# Build the vocabulary
def build_vocab(sentences):
    vocab = {}
    for sentence in sentences:
        tokens = tokenize(sentence)
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab) + 1 
    return vocab

# Create the vocabulary index
vocab = build_vocab(sentences)

print("Vocabulary Index:", vocab)
```

注意：词汇“corpus”通常用来表示一组你将用于训练的文本项。它字面上是你将用于训练模型和创建标记器的文本“主体”。

这个输出的结果如下：

```py
Vocabulary Index: {'today': 1, 'is': 2, 'a': 3, 'sunny': 4, 'day': 5, 'rainy': 6}
```

如你所见，标记器做了一个非常简单的任务，创建了一个包含我的词汇表的单词列表，每次它遇到一个独特的单词，它就会将其添加到列表中。所以第一句话，“今天是个晴天”，为五个单词生成了五个标记：“今天”、“是”、“一个”、“晴天”和“天”。第二句话与这些单词中的大多数相同，除了“雨天”是例外，所以它变成了第六个标记。

另一方面，你可以想象，对于一个非常大的语料库，这个过程会非常慢。

#### 使用 Hugging Face 的预训练标记器

考虑到这一点，我将使用 Hugging Face 的`transformers`库和其中的预构建标记器。在这种情况下，因为`transformers`库支持许多语言模型，而这些语言模型需要标记器来处理它们的文本语料库，所以这个在数百万个单词上训练的标记器是免费供你使用的。它的覆盖范围比你自己创建的更大，而且免费且易于使用！

如果你还没有这个库，你可以使用以下命令安装它：

```py
!pip install transformers
```

现在，让我们用一个简单的例子来看看它的实际应用：

```py
from transformers import BertTokenizerFast

sentences = [
    'Today is a sunny day',
    'Today is a rainy day'
]

# Initialize the tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Tokenize the sentences and encode them
encoded_inputs = tokenizer(sentences, padding=True, truncation=True, 
                           return_tensors='pt')

# To see the tokens for each input (helpful for understanding the output)
tokens = [tokenizer.convert_ids_to_tokens(ids) 
           for ids in encoded_inputs["input_ids"]]

# To get the word index similar to Keras' tokenizer
word_index = tokenizer.get_vocab()

print("Tokens:", tokens)
print("Token IDs:", encoded_inputs['input_ids'])
print("Word Index:", dict(list(word_index.items())[:10]))  
# show only the first 10 for brevity
```

它的输出看起来像这样：

```py
Tokens: [['[CLS]', 'today', 'is', 'a', 'sunny', 'day', '[SEP]'], 
         ['[CLS]', 'today', 'is', 'a', 'rainy', 'day', '[SEP]']]

Token IDs: tensor([
        [  101,  2651,  2003,  1037, 11559,  2154,   102],
        [  101,  2651,  2003,  1037, 16373,  2154,   102]])

Word Index: {'protestant': 8330, 'initial': 3988, '##pt': 13876, 
             'charters': 23010, '243': 22884, 'ref': 25416, '##dies': 18389, 
             '##uchi': 15217, 'sainte': 16947, 'annette': 22521}
```

现在，让我们来分解这个过程。我们首先从`transformers`库中导入`BertTokenizerFast`。这可以通过多种预训练的标记器进行初始化，我们选择了`'bert-base-uncased'`。你可能想知道这究竟是什么！好吧，这里的想法是我想使用一个预训练的标记器，它们通常与训练它们的模型配对。BERT（代表来自转换器的双向编码器表示）是由谷歌在一个大型语料库上训练的模型，拥有 30,000 个词汇。你可以在 Hugging Face 模型仓库中找到这样的模型，当你深入到一个模型时，你通常会看到获取其标记器的转换器代码。例如，看看[这个我使用的页面](https://oreil.ly/Ok7L9)——虽然我没有使用这个模型，但我仍然可以获取它的标记器而不是创建一个自定义的。

在这种情况下，我们创建一个`tokenizer`对象并指定它可以分词的单词数量。这将是从单词语料库中生成的最大标记数。我们这里有一个非常小的语料库，只包含六个独特的单词，所以我们将在指定的最大值一百以下。

一旦我有了分词器，我就可以直接将文本传递给它：

```py
# Tokenize the sentences and encode them
encoded_inputs = tokenizer(sentences, padding=True, truncation=True, 
                           return_tensors='pt')
```

我们将在本章稍后探索填充和截断，但就目前而言，你应该注意`return_tensors='pt'`参数。这对于我们 PyTorch 开发者来说是一个很好的便利，因为返回值将是`torch.Tensor`对象，这对于我们来说很容易处理。

BERT 模型在原始分词上使用了许多覆盖，例如`attention_masking`，这意味着它使用每个单词的`IDs`而不是原始标记。这超出了本章的范围，但对你当前的影响是，如果你不需要所有这些。如果你只需要标记，你必须以下列方式提取标记，注意你的句子在 BERT 分词器中被编码为`input_Ids`：

```py
# To see the tokens for each input (helpful for understanding the output)
tokens = [tokenizer.convert_ids_to_tokens(ids) 
           for ids in encoded_inputs["input_ids"]]
```

一旦你完成了这个步骤，你就可以轻松地打印出以下`Tokens`集合：

```py
Tokens: [['[CLS]', 'today', 'is', 'a', 'sunny', 'day', '[SEP]'], 
         ['[CLS]', 'today', 'is', 'a', 'rainy', 'day', '[SEP]']]
```

现在，你可能想知道`[CLS]`和`[SEP]`是什么——以及 BERT 模型是如何被训练成期望句子以`[CLS]`（用于*分类器*）开始，以`[SEP]`（用于*分隔符*）结束或分隔的。这两个表达式分别被分词为值 101 和 102，所以当你打印出你句子的分词值时，你会看到这个：

```py
Token IDs: tensor([
        [  101,  2651,  2003,  1037, 11559,  2154,   102],
        [  101,  2651,  2003,  1037, 16373,  2154,   102]])
```

从这个，你可以推导出*今天*在 BERT 中是标记 2651，*是*是标记 2003 等。

因此，这完全取决于你如何处理这个问题。对于使用小型数据集进行学习，自定义分词器可能就足够了。但一旦你开始处理更大的数据集，你可能希望选择一个预训练的分词器。在这种情况下，你可能必须处理一些开销——所以在本章的其余部分，我将使用自定义代码来分词和预处理文本，而不需要像 BERT 分词器那样的开销。

无论哪种方式，一旦你的句子中的单词被分词，下一步就是将你的句子转换成数字列表，其中数字是单词作为键的值。这个过程被称为*序列化*。

#### 将句子转换为序列

现在你已经看到了如何将单词转换为数字，下一步是将句子编码为数字序列，你可以这样做：

```py
def text_to_sequence(text, vocab):
    return [vocab.get(token, 0) for token in tokenize(text)]  
# 0 for unknown words
```

然后，你会得到表示三个句子的序列。记住单词索引是这样的：

```py
Vocabulary Index: {'today': 1, 'is': 2, 'a': 3, 'sunny': 4, 'day': 5, 'rainy': 6}
```

输出将看起来像这样：

```py
[1, 2, 3, 4, 5]
[1, 2, 3, 6, 5]
```

然后，你可以用数字替换单词，你会看到句子是有意义的。

现在，考虑如果你在一组数据上训练神经网络会发生什么。典型的模式是，你有一组用于训练的数据，但你知道它不会覆盖 100%的需求，但你希望尽可能多地覆盖。在 NLP 的情况下，你的训练数据中可能有成千上万的单词，它们在许多不同的上下文中使用，但你不能在所有可能的上下文中都有每个可能的单词。所以当你向你的神经网络展示一些新的、之前未见过的文本，其中包含之前未见过的单词时，可能会发生什么？你已经猜到了——网络会感到困惑，因为它根本没有任何这些单词的上下文，因此，它给出的任何预测都会受到负面影响。

##### 使用词汇表外的分词

你可以使用的一个工具来处理这些情况是*词汇表外*（OOV）*分词*，它可以帮助你的神经网络理解包含之前未见过的文本的数据的上下文。例如，给定之前的示例语料库，假设你想要处理这样的句子：

```py
test_data = [
    'Today is a snowy day',
    'Will it be rainy tomorrow?'
]
```

记住，你并不是将这个输入添加到现有的文本语料库（你可以将其视为你的训练数据），而是在考虑一个预训练网络可能会如何看待这段文本。比如说，你用你已经使用过的单词和现有的分词器来分词，就像这样：

```py
for test_sentence in test_data:
  test_seq = text_to_sequence(test_sentence, vocab)
  print(test_seq)
```

然后，你的结果将看起来像这样：

```py
[1, 2, 3, 0, 5]
[0, 0, 0, 6, 0]
```

因此，新的句子，将分词换回单词，将是“今天是<UNK>的一天”和“<UNK> <UNK> <UNK>雨天<UNK>。”

在这里，我使用标签`<UNK>`（代表*未知*）来表示 0 号分词。如果你查看我之前展示的`text_to_sequence`代码，它使用`0`来表示不在其字典中的单词。当然，你可以使用任何你喜欢的值。

##### 理解填充

在训练神经网络时，你通常需要所有数据都具有相同的形状。回想一下，在早期章节中，当你用图像训练时，你需要重新格式化图像以具有相同的宽度和高度。对于文本，你面临相同的问题——一旦你将单词分词并将句子转换为序列，它们的长度都可能不同。但为了使它们具有相同的大小和形状，你可以使用*填充*。

我们到目前为止使用的所有句子都是由五个单词组成的，所以你可以看到我们的序列是五个分词。但如果你有一些较长的句子会怎样呢？比如说，一些句子有 5 个单词，一些有 8 个单词，一些有 10 个单词。为了让神经网络处理它们，它们需要具有相同的长度！你可以通过延长较短的句子将所有内容转换为 10 个单词，通过截断较长的句子的一部分将所有内容转换为 5 个单词，或者遵循其他策略！

为了探索填充，让我们向语料库添加另一个更长的句子：

```py
sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?',
    'I really enjoyed walking in the snow today'
]
```

当你进行序列化时，你会看到你的数字列表长度不同。还请注意，如果你没有重新分词来构建新的词汇表，后两个句子将充满零：

```py
[1, 2, 3, 4, 5]
[1, 2, 3, 6, 5]
[2, 0, 4, 0]
[0, 0, 0, 0, 0, 0, 0, 1]
```

所以，别忘了调用：

```py
vocab = build_vocab(sentences)
```

然后，你将为你的分词器中的新单词获得新的标记，所以输出将看起来像这样：

```py
[1, 2, 3, 4, 5]
[1, 2, 3, 6, 5]
[2, 7, 4, 8]
[9, 10, 11, 12, 13, 14, 15, 1]
```

记住，在你之前章节中训练神经网络时，你的神经网络输入层需要图像具有一致的大小和形状。在 NLP 的大部分情况下也是如此。（对于所谓的*ragged tensors*有一个例外，但这超出了本章的范围。）因此，我们需要一种方法来使我们的句子长度相同。

这里有一个简单的填充函数：
