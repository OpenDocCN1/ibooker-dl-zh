
# 微调数据准备与模型应用

## 数据格式与准备工具

```py
{"prompt": "<prompt text>", "completion": "<completion text>"}
{"prompt": "<prompt text>", "completion": "<completion text>"}
{"prompt": "<prompt text>", "completion": "<completion text>"}
…
```

`JSONL`文件是一个文本文件，每行代表一个单独的`JSON`对象。您可以使用它来高效地存储大量数据。`OpenAI`提供了一个工具，帮助您生成这个训练文件。该工具可以接受各种文件格式作为输入（`CSV`、`TSV`、`XLSX`、`JSON`或`JSONL`），只要它们包含提示和完成列/键，并且输出一个准备好发送进行微调过程的训练`JSONL`文件。该工具还会验证并提供建议，以改善您的数据质量。

在您的终端中使用以下代码行运行此工具：

```py
$ openai tools fine_tunes.prepare_data -f <LOCAL_FILE>
```

该应用程序将提出一系列建议，以改善最终文件的结果；您可以接受或拒绝这些建议。您还可以指定选项`-q`，自动接受所有建议。

### 注意

当您执行`pip install openai`时，此`openai`工具已安装并在您的终端中可用。

如果您有足够的数据，该工具将询问是否有必要将数据分成训练集和验证集。这是一种推荐的做法。算法将使用训练数据在微调过程中修改模型的参数。验证集可以衡量模型在未用于更新参数的数据集上的性能。

对`LLM`进行微调有赖于使用高质量的示例，最好由专家审查。在使用预先存在的数据集进行微调时，确保数据经过筛查，排除冒犯性或不准确的内容，或者如果数据集太大无法手动审核所有条目，则检查随机样本。

## 使您的数据可用

一旦您准备好带有训练示例的数据集，您需要将其上传到`OpenAI`服务器。`OpenAI API`提供了不同的功能来操作文件。以下是最重要的功能：

上传文件：

```py
openai.File.create(
    file=open("out_openai_completion_prepared.jsonl", "rb"),
    purpose='fine-tune'
)
```

- 有两个必填参数：`file`和`purpose`。将`purpose`设置为`fine-tune`。这将验证用于微调的下载文件格式。此函数的输出是一个字典，您可以从中检索`id`字段中的`file_id`。目前，总文件大小可达`1GB`。如需更多，请联系`OpenAI`。

删除文件：

```py
openai.File.delete("file-z5mGg(...)")
```

- 一个参数是必需的：`file_id`。

列出所有上传的文件：

```py
openai.File.list()
```

- 例如，在开始微调过程时，检索文件的`ID`可能会有所帮助。

## 创建一个经过精细调整的模型

对上传的文件进行微调是一个简单的过程。端点`openai.FineTune.create()`在`OpenAI`服务器上创建一个作业，以从给定数据集中细化指定的模型。此函数的响应包含排队作业的详细信息，包括作业的状态、`fine_tune_id`和过程结束时模型的名称。

主要输入参数在表 4-1 中描述。

表 4-1。`openai.FineTune.create()`的参数

| 字段名称 | 类型 | 描述 |
| --- | --- | --- |
| `training_file` | 字符串 | 这是包含上传文件的`file_id`的唯一必填参数。您的数据集必须格式化为`JSONL`文件。每个训练示例都是一个具有`prompt`和`completion`键的`JSON`对象。 |
| `model` | 字符串 | 这指定了用于微调的基础模型。您可以选择`ada`、`babbage`、`curie`、`davinci`或先前调整的模型。默认的基础模型是`curie`。 |
| `validation_file` | 字符串 | 这包含具有验证数据的上传文件的`file_id`。如果提供此文件，数据将用于在微调过程中定期生成验证指标。 |
| `suffix` | 字符串 | 这是一个最多 40 个字符的字符串，添加到您的自定义模型名称中。 |

### 列出微调作业

可以通过以下函数在`OpenAI`服务器上获取所有微调作业的列表：

```py
openai.FineTune.list()
```

结果是一个包含所有精细调整模型信息的字典。

### 取消微调作业

可以通过以下函数立即中断在`OpenAI`服务器上运行的作业：

```py
openai.FineTune.cancel()
```

此功能只有一个必填参数：`fine_tune_id`。`fine_tune_id`参数是一个以`ft-`开头的字符串；例如，`ft-Re12otqdRaJ(...) `。它是在使用`openai.FineTune.cre⁠ate()`函数创建作业后获得的。如果您丢失了`fine_tune_id`，可以使用`openai.FineTune.list()`来检索它。

## 微调应用

微调提供了一种强大的方式来增强各种应用程序中模型的性能。本节将介绍几种已经有效部署微调的用例。从这些例子中获得灵感！也许您在您的用例中有相同类型的问题。再次提醒，微调比基于提示工程的其他技术更昂贵，因此在大多数情况下并不是必需的。但是当需要时，这种技术可以显著改善您的结果。

### 法律文件分析

在这种情况下，`LLM`用于处理法律文件并提取有价值的信息。这些文件通常使用特定行话编写，这使得非专业人士难以理解这些类型的文本。我们已经在第一章中看到，当在统一律师考试上进行测试时，`GPT-4`的得分位于 90 分位数。在这种情况下，微调可以使模型专门针对特定领域和/或允许其协助非专业人士参与法律流程。通过在特定主题的法律语料库上对`LLM`进行微调，或者针对特定类型的最终用户，模型可以更好地处理法律语言的复杂性，并更擅长执行与特定类型最终用户相关的任务。

通过使用大量数据对`LLM`进行微调，以分析法律文件，可以显著提高模型在这些任务中的性能，使其能够更好地处理通常超出提示工程技术能力范围的法律语言的细微差别。

### 自动代码审查

在这种用例中，微调可以帮助模型分析并提出开发人员编写的代码的改进建议。这需要在包含代码片段和注释的数据集上训练模型，使模型能够处理特定语言的语法、语义和最佳实践。请注意，这个用例类似于`GitHub`的`Copilot`工具，该工具旨在通过实时提供代码和整个函数的建议来帮助开发人员编写代码。您可以使用微调来训练模型执行特定任务，如代码审查，并为自己的代码构建一个符合特定标准或专门用于特定编程框架的项目。

作为这种用例的输入文件的一个例子，您的`JSONL`文件可以包含代码和相应的审查评论对。这将帮助模型学习如何提供准确和相关的代码质量反馈。`JSONL`文件的一行可能是这样的：

```py
{"prompt": "Review the following Python code: 
    'def sum(a, b): return a + b\nresult = sum(5, '5')'",
"completion": "Type error: The 'sum' function is adding an integer and a string. 
    Consider converting the string to an integer using int() before passing it
    to the function."}
```

### 财务文件摘要

在财务文件合成中，经过微调的`LLM`可以生成财务文件的简洁准确摘要，如收益报告、财务报表和分析报告。通过在与财务记录明确相关的数据集上微调语言模型，生成的模型可以更准确地理解这些文件的术语和上下文。例如，该模型可以将央行利率上调的详细报告压缩成简洁而富有信息的摘要。

要调整现有模型以获得更好的财务文件摘要，您需要已经有大量样本摘要。然后，您需要向模型提供一个包含财务文件及其理想摘要对的`JSONL`文件。例如：

```py
{"prompt": "The central bank announced a 0.25% increase in the benchmark 
    interest rate, citing concerns about inflationary pressures. This move is
    expected to affect borrowing costs for consumers and businesses.",
"completion": "Bank increases rate 0.25% for inflation worry. May affect people,
    business borrowing costs."}
```

### 技术文件翻译

使用微调来翻译技术文件可以显著提高基础模型的性能，与少量样本学习相比。主要原因是技术文件通常包含专业词汇和复杂的句子结构，少量样本学习无法有效处理。基础模型是未经调整的`GPT-3`模型，如`InstructGPT-3`模型，使用强化学习和人类反馈技术。要使用它们，需要进行微调。要调整现有的基础模型，您需要准备一个包含训练数据的`JSONL`文件。对于技术文件翻译的用例，该文件的内容将包括将技术文本翻译成目标语言的内容。

### 针对利基主题的新闻文章生成

在这种用例中，经过微调的模型可以为高度专业化的主题生成高质量、引人入胜和与上下文相关的新闻文章，这些主题基础模型可能没有足够的数据来准确生成。与所有其他用例一样，您需要创建一个训练数据集，以使您的模型专门用于撰写文章。为此，您需要准备许多关于特定利基主题的文章。这些数据将用于创建包含提示-完成对的`JSONL`文件。以下是一个小例子：

```py
{"prompt": "Write an introductory article about a new environmentally friendly
    cryptocurrency: 'EcoCoin: The Green Cryptocurrency Taking
    the Market by Storm'",
"completion": "As concerns over the environmental impact of cryptocurrency
    mining (...) mining process and commitment to sustainability."}
```

## 为电子邮件营销活动生成和微调合成数据

在这个例子中，我们将为电子邮件营销机构制作一个文本生成工具，利用定向内容为企业创建个性化的电子邮件营销活动。这些电子邮件旨在吸引受众并推广产品或服务。

假设我们的机构有一个支付处理行业的客户，他们要求帮助他们运行直接电子邮件营销活动，为电子商务提供新的支付服务。电子邮件营销机构决定为这个项目使用微调技术。我们的电子邮件营销机构将需要大量数据来进行微调。

在我们的情况下，我们需要为演示目的合成生成数据，正如您将在下一小节中看到的。通常，最好的结果是使用人类专家的数据，但在某些情况下，合成数据生成可能是一个有用的解决方案。

### 创建合成数据集

在以下示例中，我们从`GPT-3.5 Turbo`创建人工数据。为此，我们将在提示中指定要将促销句子发送给特定商家以销售电子商务服务。商家的特征是活动领域、商店所在城市和商店的大小。我们通过将提示发送到之前定义的`chat_completion`函数中的`GPT-3.5 Turbo`来获得促销句子。

我们通过定义三个列表来开始我们的脚本，分别对应于商店类型、商店所在的城市和商店的大小：

```py
l_sector = ['Grocery Stores', 'Restaurants', 'Fast Food Restaurants',
              'Pharmacies', 'Service Stations (Fuel)', 'Electronics Stores']
l_city = ['Brussels', 'Paris', 'Berlin']
l_size = ['small', 'medium', 'large'] 
```

然后我们在一个字符串中定义第一个提示。在此提示中，角色、上下文和任务都很明确，因为它们是使用本章前面描述的提示工程技术构建的。在此字符串中，大括号中的三个值将在代码中稍后替换为相应的值。这个第一个提示用于生成合成数据：

```py
f_prompt = """ 
Role: You are an expert content writer with extensive direct marketing 
experience. You have strong writing skills, creativity, adaptability to 
different tones and styles, and a deep understanding of audience needs and
preferences for effective direct campaigns.
Context: You have to write a short message in no more than 2 sentences for a
direct marketing campaign to sell a new e-commerce payment service to stores. 
The target stores have the following three characteristics:
- The sector of activity: {sector} `- The city where the stores are located:` `{city}`
`- The size of the stores:` `{size}` ``Task: Write a short message for the direct marketing campaign. Use the skills`
`defined in your role to write this message! It is important that the message`
`you create takes into account the product you are selling and the`
`characteristics of the store you are writing to.`
`"""``
```

以下提示仅包含三个变量的值，用逗号分隔。它不用于创建合成数据；仅用于微调：

```py
f_sub_prompt = "{sector}, {city}, {size}"
```

然后是代码的主要部分，它迭代我们之前定义的三个值列表。我们可以看到循环中的代码块很简单。我们用两个提示的大括号中的值替换为适当的值。变量`prompt`与函数`chat_completion`一起使用，以生成保存在`response_txt`中的广告。然后将`sub_prompt`和`response_txt`变量添加到*out_openai_completion.csv*文件中，这是我们微调的训练集。

```py
df = pd.DataFrame()
for sector in l_sector:
    for city in l_city:
        for size in l_size:
            for i in range(3):  ## 3 times each
                prompt = f_prompt.format(sector=sector, city=city, size=size)
                sub_prompt = f_sub_prompt.format(
                    sector=sector, city=city, size=size
                )
                response_txt = chat_completion(
                    prompt, model="gpt-3.5-turbo", temperature=1
                )
                new_row = {"prompt": sub_prompt, "completion": response_txt}
                new_row = pd.DataFrame([new_row])
                df = pd.concat([df, new_row], axis=0, ignore_index=True)
df.to_csv("out_openai_completion.csv",  index=False)
```

请注意，对于每种特征组合，我们生成三个示例。为了最大化模型的创造力，我们将温度设置为`1`。在此脚本结束时，我们有一个存储在*out_openai_completion.csv*文件中的`Pandas`表。它包含 162 个观察结果，其中有两列包含提示和相应的完成。这个文件的前两行如下：
