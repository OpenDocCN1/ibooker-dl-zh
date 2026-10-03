
# 电子邮件分类数据预处理

## 数据加载与采样

```py
Successfully retrieved message body from e-mails!
                                                   0
0  FROM:MR. JAMES NGOLA.\nCONFIDENTIAL TEL: 233-27-587908.\nE-MAIL: (james_ngola2002@maktoob.com).\n\nURGENT BUSINESS ASSISTANCE AND PARTNERSHIP.\n\n\nDEAR FRIEND,\n\nI AM ( DR.) JAMES NGOLA, THE PERSONAL ASSISTANCE TO THE LATE CONGOLESE (PRESIDENT LAURENT KABILA) WHO WAS ASSASSINATED BY HIS BODY G...
1  Dear Friend,\n\nI am Mr. Ben Suleman a custom officer and work as Assistant controller of the Customs and Excise department Of the Federal Ministry of Internal Affairs stationed at the Murtala Mohammed International Airport, Ikeja, Lagos-Nigeria.\n\nAfter the sudden death of the former Head of s...
2  FROM HIS ROYAL MAJESTY (HRM) CROWN RULER OF ELEME KINGDOM \nCHIEF DANIEL ELEME, PHD, EZE 1 OF ELEME.E-MAIL \nADDRESS:obong_715@epatra.com  \n\nATTENTION:PRESIDENT,CEO Sir/ Madam. \n\nThis letter might surprise you because we have met\nneither in person nor by correspondence. But I believe\nit is...
3  FROM HIS ROYAL MAJESTY (HRM) CROWN RULER OF ELEME KINGDOM \nCHIEF DANIEL ELEME, PHD, EZE 1 OF ELEME.E-MAIL \nADDRESS:obong_715@epatra.com  \n\nATTENTION:PRESIDENT,CEO Sir/ Madam. \n\nThis letter might surprise you because we have met\nneither in person nor by correspondence. But I believe\nit is...
4  Dear sir, \n \nIt is with a heart full of hope that I write to seek your help in respect of the context below. I am Mrs. Maryam Abacha the former first lady of the former Military Head of State of Nigeria General Sani Abacha whose sudden death occurred on 8th of June 1998 as a result of cardiac ...
```

在加载了两个数据集之后，我们现在准备从每个数据集中抽样电子邮件到一个单独的 DataFrame 中，该 DataFrame 将代表覆盖两类电子邮件的整体数据集。在这样做之前，我们必须决定从每个类别中抽取多少样本。理想情况下，每个类别中的样本数量将代表野外电子邮件的自然分布——如果我们希望我们的分类器在部署时遇到 60% 的垃圾邮件和 40% 的非垃圾邮件，那么 600 和 400 的比率可能是有意义的。请注意，数据的严重不平衡，例如 99% 的非垃圾邮件和 1% 的垃圾邮件，可能会过度拟合以大多数时间预测非垃圾邮件，这是在构建数据集时需要考虑的问题。由于这是一个理想化的实验，我们没有任何关于类别自然分布的信息，我们将假设是 50/50 的分布。我们还需要考虑如何对电子邮件进行标记化，即将电子邮件分割成文本的子单元——单词、句子等等。首先，我们将标记化为单词，因为这是最常见的方法。我们还必须决定每封电子邮件的最大标记数和每个标记的最大长度，以确保偶尔出现的极长电子邮件不会拖慢分类器的性能。我们通过指定以下通用超参数来完成所有这些工作，稍后将通过实验调整以根据需要提高性能：

```py
Nsamp = 1000         ❶
maxtokens = 50       ❷
maxtokenlen = 20     ❸
```

❶ 每个类别生成的样本数——垃圾邮件和非垃圾邮件

❷ 每个文档的最大标记数

❸ 每个标记的最大长度

有了这些指定的超参数，我们现在可以为全局训练数据集创建一个单独的 DataFrame。让我们利用这个机会执行剩余的预处理任务，即删除停用词、标点符号和标记化。

## 标记化

让我们通过定义一个函数来对邮件进行标记化，将它们分割成单词，如下列表所示。

**列表 2.4 将每封电子邮件标记化为单词**

```py
def tokenize(row):
    if row in [None,'']:
        tokens = ""
    else:
        tokens = str(row).split(" ")[:maxtokens]    ❶
    return tokens
```

❶ 按空格分割每个电子邮件字符串，以创建单词标记列表。

## 正则表达式预处理

再次查看前两页的电子邮件，我们发现它们包含大量的标点符号，并且垃圾邮件往往是大写的。为了确保分类仅基于语言内容进行，我们定义了一个函数，用于从电子邮件中删除标点符号和其他非单词字符。我们通过使用 Python 的 *regex* 库来使用正则表达式实现这一点。我们还通过使用 Python 字符串函数 `.lower()` 将单词转换为小写来规范化单词。预处理函数如下列表所示。

**列表 2.5 从电子邮件中删除标点符号和其他非单词字符**

```py
import re
def reg_expressions(row):
    tokens = []
    try:
        for token in row:
            token = token.lower()
            token = re.sub(r'[\W\d]', "", token)    ❶
            token = token[:maxtokenlen]             ❷
            tokens.append(token)
    except:
        token = ""
        tokens.append(token)
    return tokens
```

❶ 匹配并移除任何非单词字符。

❷ 截断标记

## 停用词移除

最后，让我们定义一个函数来移除*停用词*—在语言中频繁出现但对分类没有用的词。这包括诸如"the"和"are"等词，在流行的库 NLTK 中提供了一个被广泛使用的列表，我们将使用它。停用词移除函数在下一个清单中展示。请注意，NLTK 还有一些用于去除标点的方法，作为清单 2.5 所做的替代方法。

**列表 2.6 移除停用词**

```py
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = stopwords.words('english')    

def stop_word_removal(row):
    token = [token for token in row if token not in stopwords]    ❶
    token = filter(None, token)                                   ❷
    return token
```

❶ 这就是从标记列表中实际移除停用词的地方。

❷ 同样移除空字符串—''，None 等等

## 组合预处理步骤

现在我们将把所有这些功能整合在一起，构建代表两个类别的单一数据集。该过程在下一个代码清单中演示。在那段代码中，我们将合并的结果转换为 NumPy 数组，因为这是许多我们将使用的库所期望的输入数据格式。

**列表 2.7 将预处理步骤组合在一起构建电子邮件数据集**

```py
import random

EnronEmails = bodies_df.iloc[:,0].apply(tokenize)              ❶
EnronEmails = EnronEmails.apply(stop_word_removal)
EnronEmails = EnronEmails.apply(reg_expressions)
EnronEmails = EnronEmails.sample(Nsamp)                        ❷

SpamEmails = fraud_bodies_df.iloc[:,0].apply(tokenize)
SpamEmails = SpamEmails.apply(stop_word_removal)
SpamEmails = SpamEmails.apply(reg_expressions)
SpamEmails = SpamEmails.sample(Nsamp)

raw_data = pd.concat([SpamEmails,EnronEmails], axis=0).values  ❸
```

❶ 应用预定义的处理函数

❷ 从每个类别中抽取正确数量的电子邮件样本

❸ 转换为 NumPy 数组

现在让我们来看一眼结果，确保事情正在按预期进行：

```py
print("Shape of combined data represented as NumPy array is:")
print(raw_data.shape)
print("Data represented as NumPy array is:")
print(raw_data)
```

这产生了如下输出：

```py
Shape of combined data represented as NumPy array is:
(2000, )
Data represented as NumPy array is:
'got' ... ]
 ['dear', 'friend' ' my' ...]
 ['private’, 'confidential' 'friend', 'i' ... ]
 ...
```

我们看到生成的数组已经将文本分割成了单词单位，正如我们想要的。

让我们创建相应的与这些电子邮件对应的头，包括`Nsamp`=1000 封垃圾邮件，然后是`Nsamp`=1000 封非垃圾邮件，如下所示：

```py
Categories = ['spam','notspam']
header = ([1]*Nsamp)
header.extend(([0]*Nsamp))
```

现在我们已经准备将这个 NumPy 数组转换为可以实际输入到分类算法中的数值特征。

## 将电子邮件文本转换为数字

在本章中，我们首先采用了通常被认为是最简单的将单词*向量化*的方法，即将它们转换为数字向量——*袋词模型*。该模型简单地计算每封电子邮件中包含的单词标记的频率，从而将其表示为这种频率计数的向量。我们在清单 2.8 中提供了组装电子邮件的词袋模型的函数。请注意，通过这样做，我们只保留出现超过一次的标记，如变量`used_tokens`所捕获的那样。这使我们能够将向量维度保持得比其他情况下低得多。还请注意，可以使用流行的库 scikit-learn 中的各种内置矢量化器来实现这一点（我们的 Jupyter 笔记本展示了如何做到这一点）。但是，我们专注于清单 2.8 中所展示的方法，因为我们发现这比使用实现相同功能的黑匣子函数更具说明性。我们还注意到，scikit-learn 的向量化方法包括计算任意*n*个单词序列或*n-gram*的出现次数，以及*tf-idf*方法—如果有生疏的话，这些是您应该复习的重要基本概念。在这里展示的问题中，当使用这些向量化方法时，我们并未注意到与使用词袋模型方法相比的改进。

**清单 2.8 组装词袋表示**

```py
def assemble_bag(data):
    used_tokens = []
    all_tokens = []

    for item in data:
        for token in item:
            if token in all_tokens:             ❶
                if token not in used_tokens:
                    used_tokens.append(token)
            else:
                all_tokens.append(token)

    df = pd.DataFrame(0, index = np.arange(len(data)), columns = used_tokens)

    for i, item in enumerate(data):             ❷
        for token in item:
            if token in used_tokens:
                df.iloc[i][token] += 1    
    return df
```

❶ 如果标记之前已经见过，将其附加到用过的标记输出列表

❷ 创建 Pandas DataFrame 计数词汇单词的频率——对应于每封电子邮件的列——对应于行

定义了`assemble_bag`函数之后，让我们使用它来实际执行向量化并将其可视化如下：

```py
EnronSpamBag = assemble_bag(raw_data)
print(EnronSpamBag)
predictors = [column for column in EnronSpamBag.columns]
```

输出 DataFrame 的一个片段如下所示：

```py
     fails  report s  events   may   compliance  stephanie  
0         0         0        0     0           0          0  
1         0         0        0     0           0          0  
2         0         0        0     0           0          0  
3         0         0        0     0           0          0  
4         0         0        0     0           0          0  
...     ...       ...      ...   ...         ...        ...  
1995      1         2        1     1           1          0  
1996      0         0        0     0           0          0  
1997      0         0        0     0           0          0  
1998      0         0        0     0           0          1  
1999      0         0        0     0           0          0  

[2000 rows x 5469 columns]
```

列标签指示词袋模型的词汇中的单词，每行中的数字条目对应于我们数据集中 2000 封电子邮件中每个此类单词的频率计数。请注意，这是一个极为稀疏的 DataFrame——它主要由值`0`组成。

## 数据洗牌与训练/测试分割

将数据集完全向量化后，我们必须记住它与类别无关的洗牌；也就是说，它包含`Nsamp` = 1000 封垃圾邮件，然后是相同数量的非垃圾邮件。根据如何拆分此数据集——在我们的情况下，通过选择前 70％用于训练，剩余部分用于测试——这可能导致训练集仅由垃圾邮件组成，这显然会导致失败。为了在数据集中创建类样本的随机排序，我们需要与标头/标签列表一起洗牌数据。下一个清单中显示了实现此目的的函数。同样，可以使用内置的 scikit-learn 函数实现相同的效果，但我们发现下一个清单中显示的方法更具说明性。

**清单 2.9 与标头/标签列表一起洗牌数据**

```py
def unison_shuffle_data(data, header):
    p = np.random.permutation(len(header))
    data = data[p]
    header = np.asarray(header)[p]
    return data, header
```

作为准备电子邮件数据集以供基线分类器训练的最后一步，我们将其分割为独立的训练和测试，或验证集。这将允许我们评估分类器在未用于训练的一组数据上的性能——这是机器学习实践中必须确保的重要事情。我们选择使用 70％ 的数据进行训练，然后进行 30％ 的测试/验证。下面的代码调用了同步洗牌函数，然后执行了训练/测试分割。生成的 NumPy 数组变量`train_x`和`train_y`将直接传递给本章后续部分中的分类器：
