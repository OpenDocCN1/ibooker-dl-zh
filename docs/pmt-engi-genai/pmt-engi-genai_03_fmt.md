
# 第三章：使用 ChatGPT 进行文本生成的标准实践

简单的提示技巧将帮助你最大化 LLM 的输出和格式。你将从定制提示开始，以探索用于文本生成的所有常见实践。

## 生成列表

自动生成列表功能非常强大，使你能够专注于更高层次的任务，同时 GPT 可以代表你自动生成、优化、重新排序和去重列表。

输入：

```py
Generate a list of Disney characters.
```

输出：

```py
Sure, here is a list of some popular Disney characters:

1\. Mickey Mouse
2\. Minnie Mouse
...
30\. Bagheera (The Jungle Book)
```

GPT-4 完全能够提供角色列表。然而，这种方法存在一些陷阱：

+ GPT 决定提供 30 个示例作为编号列表，列表项之间由 `\n` 字符分隔。然而，如果你的下游 Python 代码期望在项目符号处分割，那么你可能会得到不理想的结果或运行时错误。

+ GPT 提供了前置说明；移除任何前置/后续说明将使解析输出更容易。

+ 列表大小未受控制，留给了语言模型。

+ 一些角色的名字在括号内带有他们对应电影的名称——例如，*Bagheera (The Jungle Book)*——而另一些则没有。这使得提取名字变得更加困难，因为你需要移除电影标题。

+ 在基于我们期望结果的 LLM 生成过程中没有应用任何过滤或选择。

以下是一个优化的提示。

输入：

```py
Generate a bullet-point list of 5 male Disney characters.
Only include the name of the character for each line.
Never include the film for each Disney character.
Only return the Disney characters, never include any commentary.

Below is an example list:

* Aladdin
* Simba
* Beast
* Hercules
* Tarzan
```

输出：

```py
* Woody
* Buzz Lightyear
* Stitch
* Jack Sparrow
* Prince Charming
```

## 提供示例

简单地重新措辞你的提示以包含示例 *(few-shot prompting)* 可以极大地影响期望的输出。

通过优化提示，你已实现以下效果：

+ 将列表限制为固定大小五项

+ 仅生成男性角色

+ 正确地使用项目符号格式化列表

+ 移除了任何前置说明

简单列表适用于大多数任务；然而，它们结构化程度较低，对于某些任务，从 GPT-4 输出中获得嵌套数据结构是有益的。

三种典型数据结构包括：

+ 嵌套文本数据（层级列表）

+ JSON

+ YAML

### 层级列表生成

层级列表在期望的输出是嵌套时非常有用。一个很好的例子是详细的文章结构。

输入：

```py
Generate a hierarchical and incredibly detailed article outline on:

What are the benefits of data engineering.

See an example of the hierarchical structure below:

Article Title: What are the benefits of digital marketing?

* Introduction
    a. Explanation of digital marketing
    b. Importance of digital marketing in today's business world
* Increased Brand Awareness
    a. Definition of brand awareness
    b. How digital marketing helps in increasing brand awareness
```

输出结果：

```py
Article Title: What are the benefits of data engineering?

* Introduction
    a. Explanation of data engineering
    b. Importance of data engineering in today’s data-driven world

...(10 sections later)...

* Conclusion
    a. Importance of data engineering in the modern business world
    b. Future of data engineering and its impact on the data ecosystem
```

为了在前面的输出中生成有效的文章大纲，你包括了两个关键短语：

层级结构

建议文章大纲需要生成嵌套结构。

极其详细

指导语言模型生成更大的输出。其他可以包含以产生相同效果的词语包括 *非常长* 或通过指定大量子标题，*包括至少 10 个顶级标题*。

> **注意**
> 请求语言模型生成固定数量的项目并不保证语言模型会生成相同长度的输出。例如，如果你请求 10 个标题，你可能会只收到 8 个。因此，你的代码应该验证是否存在 10 个标题，或者能够灵活处理来自 LLM 的不同长度。

因此，你已经成功生成了一个分层文章概要，但如何将字符串解析成结构化数据呢？

让我们用 Python 探索示例 3-1，其中你之前已成功对 OpenAI 的 GPT-4 进行了 API 调用。这里使用了两个正则表达式来从 `openai_result` 中提取标题和子标题。Python 中的 `re` 模块用于处理正则表达式。

##### 示例 3-1\. [解析分层列表](https://oreil.ly/A0otS)

```py
import re

# openai_result = generate_article_outline(prompt)
# Commented out to focus on a fake LLM response, see below:

openai_result = '''
* Introduction
 a. Explanation of data engineering
 b. Importance of data engineering in today’s data-driven world
* Efficient Data Management
 a. Definition of data management
 b. How data engineering helps in efficient data management
* Conclusion
 a. Importance of data engineering in the modern business world
 b. Future of data engineering and its impact on the data ecosystem
'''

# Regular expression patterns
heading_pattern = r'\* (.+)'
subheading_pattern = r'\s+[a-z]\. (.+)'

# Extract headings and subheadings
headings = re.findall(heading_pattern, openai_result)
subheadings = re.findall(subheading_pattern, openai_result)

# Print results
print("Headings:\n")
for heading in headings:
    print(f"* {heading}")

print("\nSubheadings:\n")
for subheading in subheadings:
    print(f"* {subheading}")
```

此代码将输出：

```py
Headings:
- Introduction
- Efficient Data Management
- Conclusion

Subheadings:
- Explanation of data engineering
- Importance of data engineering in today’s data-driven world
- Definition of data management
- How data engineering helps in efficient data management
- Importance of data engineering in the modern business world
- Future of data engineering and its impact on the data ecosystem
```

正则表达式的使用允许进行高效的模式匹配，使得处理输入文本的变化（如是否存在前导空格或制表符）成为可能。让我们探索这些模式是如何工作的：

+ `heading_pattern = r'\* (.+)'`

此模式旨在提取主要标题，由以下部分组成：

+ `\*` 匹配标题开头处的星号 `(*)` 符号。反斜杠用于转义星号，因为在正则表达式中星号具有特殊含义（前一个字符出现零次或多次）。

+ 星号之后将匹配一个空格字符。

+ `(.+)`: 匹配一个或多个字符，括号创建了一个捕获组。`.` 是一个通配符，匹配除换行符之外的任何字符，而 `+` 是一个量词，表示前面的元素（在这种情况下是点）出现一次或多次。

通过应用此模式，你可以轻松地将所有主要标题提取到一个列表中，而不包含星号。

+ `subheading_pattern = r'\s+[a-z]\. (.+)`

`subheading pattern` 将匹配 `openai_result` 字符串中的所有子标题：

+ `\s+` 匹配一个或多个空白字符（空格、制表符等）。`+` 表示前面的元素（在这种情况下是 `\s`）出现一次或多次。

+ `[a-z]` 匹配从 *a* 到 *z* 的单个小写字母。

+ `\.` 匹配点字符。反斜杠用于转义点，因为在正则表达式中点具有特殊含义（匹配除换行符之外的任何字符）。

+ *点号之后将匹配一个空格字符。*

+ `(.+)` 匹配一个或多个字符，括号创建了一个捕获组。`.` 是一个通配符，匹配除换行符之外的任何字符，而 `+` 是一个量词，表示前面的元素（在这种情况下是点）出现一次或多次。

此外，`re.findall()` 函数用于在输入字符串中查找所有非重叠模式的匹配项，并将它们作为列表返回。然后打印提取的标题和子标题。

因此，你现在能够从分层文章概要中提取标题和子标题；然而，你可以进一步细化正则表达式，以便每个标题都与相应的 `subheadings` 相关联。

在示例 3-2 中，正则表达式略有修改，以便每个子标题直接与其相应的子标题相关联。

##### 示例 3-2\. [将分层列表解析为 Python 字典](https://oreil.ly/LcMtv)

```py
import re

openai_result = """
* Introduction
 a. Explanation of data engineering
 b. Importance of data engineering in today’s data-driven world
* Efficient Data Management
 a. Definition of data management
 b. How data engineering helps in efficient data management
 c. Why data engineering is important for data management
* Conclusion
 a. Importance of data engineering in the modern business world
 b. Future of data engineering and its impact on the data ecosystem
"""

section_regex = re.compile(r"\* (.+)")
subsection_regex = re.compile(r"\s*([a-z]\..+)")

result_dict = {}
current_section = None

for line in openai_result.split("\n"):
    section_match = section_regex.match(line)
    subsection_match = subsection_regex.match(line)

    if section_match:
        current_section = section_match.group(1)
        result_dict[current_section] = []
    elif subsection_match and current_section is not None:
        result_dict[current_section].append(subsection_match.group(1))

print(result_dict)
```

这将输出：

```py
{
    "Introduction": [
        "a. Explanation of data engineering",
        "b. Importance of data engineering in today’s data-driven world"
    ],
    "Efficient Data Management": [
        "a. Definition of data management",
        "b. How data engineering helps in efficient data management"
    ],
    "Conclusion": [
        "a. Importance of data engineering in the modern business world",
        "b. Future of data engineering and its impact on the data ecosystem"
    ]
}
```

部分标题正则表达式 `r'\* (.+)'` 匹配一个星号后跟一个空格和更多字符。括号捕获星号和空格后面的文本，以便在代码中稍后使用。

子部分正则表达式 `r'\s*([a-z]\..+)'` 以 `\s*` 开头，它匹配零个或多个空白字符（空格或制表符）。这允许正则表达式匹配带有或不带有前导空格或制表符的子部分。接下来的部分 `([a-z]\..+)` 匹配一个小写字母后跟一个点和一个或多个字符。括号捕获整个匹配的子部分文本，以便在代码中稍后使用。

`for` 循环遍历输入字符串 `openai_result` 的每一行。当遇到与部分标题正则表达式匹配的行时，循环将匹配的标题设置为当前部分，并在 `result_dict` 字典中为其分配一个空列表作为其值。当一行与子部分正则表达式匹配时，匹配的子部分文本被追加到对应当前部分的列表中。

因此，循环逐行处理 *输入字符串*，将行分类为部分标题或子部分，并构建所需的字典结构。

### 避免使用正则表达式的时机

当你努力从 LLM 响应中提取更多结构化数据时，仅依靠正则表达式会使控制流*变得越来越复杂*。然而，还有其他格式可以轻松地促进从 LLM 响应中解析结构化数据。两种常见的格式是 *.json* 和 *.yml* 文件。

### 生成 JSON

让我们从实验一些提示设计开始，这将指导一个大型语言模型返回 JSON 响应。

输入：

```py
Compose a very detailed article outline on "The benefits of learning code" with a
JSON payload structure that highlights key points.

Only return valid JSON.

Here is an example of the JSON structure:
{
    "Introduction": [
        "a. Explanation of data engineering",
        "b. Importance of data engineering in today’s data-driven world"],
    ...
    "Conclusion": [
        "a. Importance of data engineering in the modern business world",
        "b. Future of data engineering and its impact on the data ecosystem"]
}
```

输出：



```py
{
    "Introduction": [
        "a. Overview of coding and programming languages",
        "b. Importance of coding in today's technology-driven world"],
    ...
    "Conclusion": [
        "a. Recap of the benefits of learning code",
        "b. The ongoing importance of coding skills in the modern world"]
}
```

# 给出指示并提供示例

注意，在前面的提示中，你已经提供了关于任务类型、格式和示例 JSON 输出的指示。

当处理 JSON 时，你可能会遇到的一些常见错误包括无效的有效负载，或者 JSON 被三重反引号包裹（```py），例如：

Output:

```

当然，这是 JSON：

```pyjson
{"Name": "John Smith"} # valid payload
{"Name": "John Smith", "some_key":} # invalid payload
```

```py

理想情况下，你会希望模型像这样响应：

Output:

```

{"Name": "John Smith"}

```py

这很重要，因为对于第一个输出，你必须在`json`之后拆分，然后解析包含有效 JSON 的字符串的确切部分。有几点值得添加到你的提示中，以改进 JSON 解析：

```

你必须遵循以下原则：

* 仅返回有效的 JSON

* 永远不要包含反引号符号，例如：`

* 响应将通过`json.loads()`解析，因此它必须是有效的 JSON。

```py

现在让我们检查如何解析 [JSON output with Python](https://oreil.ly/MoJHn)：

```

import json

# openai_json_result = generate_article_outline(prompt)

openai_json_result = """

{

"Introduction": [

"a. 编码和编程语言的概述",

"b. 当今技术驱动世界中编码的重要性"],

"Conclusion": [

"a. 回顾学习代码的好处",

"b. 现代世界中编码技能的持续重要性"]

}

"""

parsed_json_payload = json.loads(openai_json_result)

print(parsed_json_payload)

'''{'Introduction': ['a. 编码和编程语言的概述',

"b. 当今技术驱动世界中编码的重要性"],

'Conclusion': ['a. 回顾学习代码的好处',

'b. The ongoing importance of coding skills in the modern world']}'''

```py

干得好，你已经成功解析了一些 JSON。

如所示，当请求以有效 JSON 格式响应时，从 LLM 响应中结构化数据是简化的。与之前演示的正则表达式解析相比，这种方法更不繁琐且更直接。

那么可能出什么问题呢？

* 语言模型意外地在响应中添加了额外文本，例如`json output:`，而你的应用程序逻辑只处理有效的 JSON。

* 生成的 JSON 无效，解析时失败（要么因为大小，要么仅仅因为没有转义某些字符）。

稍后你将研究策略，以优雅地处理此类边缘情况。

## YAML

`*.yml`文件是一种结构化数据格式，与`*.json`相比提供了不同的优势：

无需转义字符

YAML 的缩进模式消除了对花括号、方括号和逗号来表示结构的需求。这可以带来更干净、更不易出错的文件，因为括号或标点不匹配的风险更小。

可读性

YAML 设计为人类可读，与 JSON 相比具有更简单的语法和结构。这使得你更容易创建、阅读和编辑提示，尤其是在处理复杂或嵌套结构时。

注释

与 JSON 不同，YAML 支持注释，允许你直接在文件中向提示添加注释或解释。当在团队中工作或在一段时间后重新查看提示时，这非常有用，因为它允许更好的理解和协作。

输入：

```

- 以下是你将找到的当前 yaml 模式。

- 您可以根据用户查询更新数量。

- 根据以下模式过滤用户查询，如果不匹配则

如果没有剩余的项目，则返回`"No Items"`。

- 如果有部分匹配，则只返回匹配的项目

在以下模式内：

# 模式：

- 项目：苹果片

数量：5

单位：个

- 项目：牛奶

数量：1

单位：加仑

- 项目：面包

数量：2

单位：条

- 项目：鸡蛋

数量：1

单位：打

用户查询："5 个苹果片，2 打鸡蛋。"

根据以下模式，请仅返回有效的.yml

查询。如果没有匹配项，则返回`"No Items"`。不要提供任何

注释或解释。

```py

Output:

```

- 项目：苹果片

数量：5

单位：个

- 项目：鸡蛋

数量：2

单位：打

```py

注意前面的示例中，LLM 如何能够从`User Query`字符串中推断出正确的`*.yml`格式。

此外，你给了 LLM 一个机会，要么：

* 返回有效的`*.yml`响应

* 返回过滤后的`*.yml`响应

如果过滤后没有剩余的`*.yml`项目，则返回*No Items*。

# 过滤 YAML 负载

你可能决定使用相同的提示来清理/过滤`*.yml`负载。

首先，让我们关注一个包含相对于我们期望的`schema`既有效又无效的`schema`的负载。`Apple slices`符合标准；然而，`Bananas`不存在，你应该期望`User Query`被适当地过滤。

输入：

```

# 用户查询：

- 项目：苹果片

数量：5

单位：个

- 项目：香蕉

数量：3

单位：个

```py

Output:

```

# 更新的 yaml 列表

- 项目：苹果片

数量：5

单位：个

```py

在前面的示例中，你已经成功地将用户的负载与一组标准进行了过滤，并将语言模型用作*推理引擎*。

通过在提示中向 LLM 提供一组指令，响应与人类手动清理数据时的做法密切相关。

输入提示促进了将更多控制流任务委托给语言学习模型（LLM），这些任务通常需要用 Python 或 JavaScript 等编程语言编写代码。

图 3-1 提供了 LLM 处理用户查询时应用的逻辑的详细概述。

![Using an LLM to determine the control flow of an application instead of directly using code.](img/pega_0301.png)

###### 图 3-1\. 使用 LLM 而不是代码来确定应用程序的控制流

# 处理 YAML 中的无效负载

一个完全无效的负载可能看起来像这样：

输入：

```

# 用户查询：

- 项目：香蕉

数量：3

单位：个

```py

Output:

```

没有项目

```py

正如预期的那样，LLM 返回了`No Items`，因为`User Query`中的项目都没有与先前定义的`schema`匹配。

让我们创建一个 Python 脚本，优雅地容纳 LLM 返回的各种类型的结果。脚本的核心部分将集中在：

* 为可能因三种 LLM 响应场景而发生的每种错误类型创建自定义异常

* 解析提议的 schema

* 对响应运行一系列自定义检查，以便你可以确保 YML 响应可以安全地传递给下游软件应用程序/微服务

你可以定义六个特定的错误来处理所有边缘情况：

```

class InvalidResponse(Exception):

    pass

class InvalidItemType(Exception):

    pass

class InvalidItemKeys(Exception):

    pass

class InvalidItemName(Exception):

    pass

class InvalidItemQuantity(Exception):

    pass

class InvalidItemUnit(Exception):

    pass

```py

然后将先前提议的`YML schema`作为字符串提供：

```

# 提供的模式

schema = """

- 项目：苹果片

数量：5

单位：个

- 项目：牛奶

数量：1

单位：加仑

- 项目：面包

数量：2

单位：条

- 项目：鸡蛋

数量：1

单位：打

"""

```py

导入`yaml`模块并创建一个名为`validate_response`的自定义解析函数，该函数允许你轻松确定 LLM 输出是否有效：

```

导入 yaml

def validate_response(response, schema):

    # 解析模式

    schema_parsed = yaml.safe_load(schema)

    最大数量 = 10

    # 检查响应是否为列表

    if not isinstance(response, list):

        raise InvalidResponse("响应不是列表")

    # 检查列表中的每个项目是否是字典

    for item in response:

        if not isinstance(item, dict):

            raise InvalidItemType('''项目不是字典''')

        # 检查每个字典是否具有 "item"，"quantity" 和 "unit" 键

        if not all(key in item for key in ("item", "quantity", "unit")):

            raise InvalidItemKeys("项目没有正确的键")

        # 检查与每个键关联的值是否是正确的类型

        if not isinstance(item["item"], str):

            raise InvalidItemName("项目名称不是字符串")

        if not isinstance(item["quantity"], int):

            raise InvalidItemQuantity("项目数量不是整数")

        if not isinstance(item["unit"], str):

            raise InvalidItemUnit("项目单位不是字符串")

        # 检查与每个键关联的值是否是正确的值

        if item["item"] not in [x["item"] for x in schema_parsed]:

            raise InvalidItemName("项目名称不在模式中")

        if item["quantity"] >  最大数量:

            raise InvalidItemQuantity(f'''项目数量大于

            {最大数量}''')

        if item["unit"] not in ["pieces", "dozen"]:

            raise InvalidItemUnit("项目单位不是个或打")

```py

为了测试这些边缘情况，下面你将找到几个模拟的 LLM 响应：

```

# 模拟响应
```



`fake_response_1 = """`

```
- 项目：苹果片
  数量：5
  单位：个
- 项目：鸡蛋
  数量：2
  单位：打
"""
```

`fake_response_2 = """`

```
# 更新后的 yaml 列表
- 项目：苹果片
  数量：5
  单位：件
"""
```

`fake_response_3 = """Unmatched"""`

```py
Finally, now you can:
*   Use `yaml.safe_load(response)` to safely parse the *.yml* schema
*   Call the `validate_response` function for each LLM response to test it against custom *.yml* validation logic
```

# 解析伪造的响应

`response_1_parsed = yaml.safe_load(fake_response_1)`

`response_2_parsed = yaml.safe_load(fake_response_2)`

`response_3_parsed = yaml.safe_load(fake_response_3)`

# 验证响应是否符合模式

```py
try:
    validate_response(response_1_parsed, schema)
    print("响应 1 是有效的")
except Exception as e:
    print("响应 1 是无效的:", str(e))

try:
    validate_response(response_2_parsed, schema)
    print("响应 2 是有效的")
except Exception as e:
    print("响应 2 是无效的:", str(e))

try:
    validate_response(response_3_parsed, schema)
    print("响应 3 是有效的")
except Exception as e:
    print("响应 3 是无效的:", str(e))
```

响应 1 是有效的

响应 2 是有效的

响应 3 是无效的：响应不是列表

```py
# Diverse Format Generation with ChatGPT

ChatGPT is not only capable of generating *.json* or *.yml* files but can also support many other data formats. For example, it can aid you in creating a mermaid diagram or generating a mock *.csv* document. Its versatility stretches to even generating code, conversations, and scripts.

Let’s explore in detail how ChatGPT can produce diverse formatted content. Mermaid is a markdown-like scripting language for generating charts, diagrams, and other visualizations.
```

Input:

```
使用 mermaid 语法创建一个简单的食物订购流程图
从“选择食物”开始，以“付费用餐”结束的过程
包含“添加到购物车”和“确认购物车”的步骤。
```

```py
Output:

graph TD
    选择食物[选择食物] --> 添加到购物车[添加到购物车]
    添加到购物车 --> 确认购物车[确认购物车]
    确认购物车 --> 付费用餐[付费用餐]
```

This output, when input to a mermaid diagram renderer, will create a precise flowchart reflecting the food ordering process (Figure 3-2).

![A simple mermaid diagram output from an LLM.](img/pega_0302.png)

###### Figure 3-2. A streamlined flow diagram created using mermaid syntax via an LLM

## Mock CSV Data

ChatGPT can also generate mock CSV data that can be utilized for testing purposes or other uses.

Input:

```
为五名学生生成以下字段的样本 CSV 数据：
姓名，年龄，年级。
```

```py
Output:

姓名，年龄，年级
约翰，16，A
阿历克斯，17，B
艾玛，16，A
索菲亚，16，B
李姆，17，A
```

The example generated is a well-formatted CSV file, which can be loaded directly into many tools that read CSV data.

Remember, GPT models rely heavily on the format you specify. For instance, specifying CSV implies the AI should use commas as separators. Be clear about the desired format for optimal results.

You’ve explored different ways of extracting structured data from language models including regular expressions, JSON, YML, and other formats. You can also now use LLMs to direct parts of your application’s control flow.

# Explain It like I’m Five

The *Explain It like I’m Five* prompt focuses on explaining a given section of text *as if you were talking to a five-year-old child.*

This style of prompting is very effective and is a great way to convert technical documents into simple summaries that anyone can understand.

Let’s apply this prompt to an abstract of [“Tumor Microenvironment: Recent Advances in Various Cancer Treatments”](https://oreil.ly/MDOsz).

Input:

```
像我五岁时一样解释以下文本。就像我五岁一样解释。
文本：这是一篇关于不同类型癌症治疗的评论。
我们旨在分析肿瘤微环境和最近的治疗应用趋势
对于几种类型的治疗效果和有效性
癌症的。传统上，癌症治疗基于
新生细胞。
...(为简洁起见省略)...
```

```py
Output:

癌症是一种疾病，使你的身体细胞以不正常的方式生长和分裂
那是不正常的。医生有不同的方法来治疗癌症，比如
切掉它，给你药，或者使用特殊光线来杀死
癌细胞...
```

The ChatGPT output from the preceding example is much more digestible than the abstract.

# Universal Translation Through LLMs

Additionally, language models can potentially act as *universal translators* due to their proficiency in understanding and generating human-like text across multiple languages.

Let’s investigate this with a simple example:

1.  Generate some simple text.
2.  Change the simple text and make it difficult to read.
3.  Then transform the same text back into an easy to read format in Spanish (instead of English).
4.  Revert the simplistic text back into English.

Input:

```
创建一些简单的文本。
```

```py
Output:

太阳缓缓升起，用橙色和粉色的色调描绘天空。鸟儿
清新的微风穿过树木，带着新鲜草和盛开的花的香气。生命
树木，带着新鲜草和盛开的花的香气。生命
感觉到宁静和希望的觉醒。
...(为简洁起见省略)...
```

Input:

```
现在将文本变得极其复杂且难以阅读。
```

```py
Output:

在天体上升的虚幻领域，其中辉煌
太阳的球体开始缓慢上升，这是一个精美现象
展开，在广阔的天空上产生了一曲色彩绚丽的交响乐
天空的广阔冠冕。鸟类居民，赋予
甜美的声音，表达他们的欢乐问候，点缀
带有幸福之音的听觉织物。微风，温柔地
在树木哨兵中飘荡，带着它轻柔的潮流
鲜绿叶片和盛开的花朵的香气。
...(为简洁起见省略)...
```

Input:

```
现在将文本转换成简单、易于阅读的西班牙语格式
```

```py
Output:

```py
Input:

```py
Output:

```py
As you can see, part of the meaning is lost while doing the translation, and this is to be expected. However, large parts of the original text’s intent remains, demonstrating that LLMs can act as *highly effective translators* for languages that have adequate resources on the internet. For languages that do not have a good amount of resources, it will yield bad results.

The same logic applies to coding languages. LLMs are very good at generating code for established programming languages such as Python and JavaScript but perform worse for newer coding languages and packages.

The boundaries between different forms of information are becoming *increasingly fluid*. The essence of information itself is evolving, allowing for effortless transformations of summaries into stories, poems, or other creative expressions, ultimately enriching our understanding and engagement with the content.

*Diffusion models* are a unique class of generative models utilized in machine learning, specifically designed to produce new images that mimic those found in the training set.

Moreover, when you combine language models with diffusion models, it enables seamless transitions between text, video, and other modalities. This makes it even simpler for you to convey complex ideas across various formats, facilitating a more accessible and comprehensive experience.

# Ask for Context

LLMs are not only capable of generating text but can also act as simple agents with a limited amount of *reasoning capability.* This allows you to write a prompt asking the language model to either:

*   Return a valid result to a question or statement
*   Ask for more context to appropriately answer the question

In this section, you’ll learn about the importance of *asking for context* when working with LLMs such as GPT-4. We will start with an example of a prompt that doesn’t provide enough context, resulting in a less useful response.

Then, we will provide a better prompt that encourages the model to ask for additional context if needed. Finally, we will use the additional context provided to generate a more informed response.

Input:

```py
Output:

```py
In the preceding prompt, the model does not have enough context to make a meaningful recommendation. Instead, you can ask ChatGPT for a list of recommended points that would help it to make an effective decision.

Input:

```py
Output:

```py
After prompting ChatGPT about how to make an informed decision, now you’re aware of what to include within your prompt to help the language model with deciding.

Input:

```
- 数据完整性和约束：强制执行严格的外键
  关系和复杂的数据验证规则。
- 事务和并发：需要处理高水平的
  在保持数据一致性的同时进行并发读写操作。
- 成熟度和稳定性：优先考虑建立良好且广泛使用的
  经过测试的数据库以实现长期稳定性。
- 开发和运营便捷性：采用敏捷开发方法，需要
  用于快速原型设计和迭代模式设计。
- 成本：预算限制需要使用开源解决方案，
  无需额外的许可或支持费用。
如果您需要更多上下文，请具体说明什么可以帮助您做出
以做出更好的决定。
```

```py
Output:

基于这些考虑，MongoDB 和 PostgreSQL 都有其
强调数据一致性的重要性，但考虑到复杂的数据，
查询和数据完整性约束，PostgreSQL 似乎是一个更
适合您项目的选择。
```



在这个最终示例中，模型利用提供的额外上下文，给出了关于使用 PostgreSQL 的明智建议。通过在必要时主动询问上下文，像 ChatGPT 和 GPT-4 这样的大语言模型（LLM）能够提供更有价值、更准确的回答。

图 3-3 展示了*询问上下文*如何改变大语言模型的决策过程。在收到用户输入后，模型首先评估当前提供的上下文是否充分。如果不够充分，它会提示用户提供更详细的信息，这凸显了模型对丰富上下文输入的依赖。一旦获取到足够的上下文，大语言模型便会生成一个信息充分且相关的响应。

![The decision process of an LLM with asking for context.](img/pega_0303.png)

###### 图 3-3\. 大语言模型在询问上下文时的决策过程

# 默认允许大语言模型询问更多上下文

你可以通过包含以下关键短语，将“允许大语言模型询问更多上下文”设为默认行为：*如果你需要更多上下文，请说明哪些信息能帮助你做出更好的决策。*

在本节中，你已经看到大语言模型如何充当利用环境上下文进行决策的智能体。通过根据模型的建议迭代地优化提示词，我们最终会达到一个模型拥有*足够上下文以做出明智决策*的状态。

这个过程凸显了在提示词中提供充分上下文的重要性，以及在必要时准备好询问更多信息的必要性。通过这样做，你可以利用 GPT-4 等大语言模型的能力，做出更准确、更有价值的推荐。

在像 GPT-4 这样的基于智能体的系统中，能够询问更多上下文并提供最终答案，对于做出明智的决策至关重要。[AutoGPT](https://oreil.ly/l3Ihy) 是一个多智能体系统，它有一个自我评估步骤，会自动检查在当前提示词上下文下任务是否能够完成。该技术使用了“行动者-批评者”关系，即在执行前分析现有的提示词上下文，看它是否可以进一步优化。

# 文本风格解耦

*文本风格解耦*是提示工程中一项强大的技术，它允许你从给定文档中提取并分离出特定的文本特征，例如语气、长度、词汇和结构。

这使你能够创建与原始文档具有相似特征的新内容，确保在各种形式的沟通中保持风格和语气的一致性。

这种一致性对于需要在不同渠道和平台上以统一声音进行沟通的企业和组织至关重要。该技术的好处包括：

**提升品牌一致性**

通过确保所有内容遵循相似的风格，组织可以强化其品牌形象并保持统一的形象。

**简化内容创作**

通过提供一套清晰的指南，写作者和内容创作者可以更容易地制作出符合所需风格的材料。

**适应性**

文本风格解耦允许在保留核心信息和语气的同时，将现有内容轻松适配到新的格式或风格。

文本风格解耦的过程包括*识别所需的文本特征*或创建一个元提示（用于创建提示词的提示词）来提取这些特征，然后使用提取出的特征来指导新内容的生成。

# 识别所需的文本特征

要成功地对文本风格进行解耦，你首先必须识别要从输入文档中提取的特定特征。常见的文本特征包括：

**语气**

文本传达的整体情绪或态度，例如正式、随意、幽默或权威。

**长度**

内容的期望字数或大致篇幅。

**词汇和措辞**

词语和表达方式的选择，包括行业特定术语、口语以及复杂程度。

此外，你甚至可以创建一个元提示词，动态地向你展示这些特征。

输入：

```
分析以下文本并确定核心功能，这些功能将允许
进一步的文档以模仿相同的风格、一致性和格式
如下文所述。
提供所有功能作为将用作风格写作指南的所有内容，
用于未来文档。

分析以下文本：

人工智能正在迅速改变我们周围的世界，
开辟新的可能性并颠覆行业。从
从医疗保健到娱乐，人工智能正在使我们的生活变得更好，更加
方便之处不胜枚举。然而，权力越大，责任越大，
责任。我们必须确保人工智能技术得到开发和
以道德和负责任的方式部署，以保护我们的隐私和维护我们的
价值观。
```

输出：

```
风格写作指南：

语气
- 在整个文档中保持信息性和乐观的语气。
- 强调主题的积极影响，同时承认
任何潜在的关注或责任。

长度：
...

词汇：
...

结构：
...

内容：
- 专注于主题的变革性质和积极影响，
事项。
- 突出特定行业或领域，其中主题有所
已经产生了影响。
- 承认任何相关的责任或潜在风险，
强调道德和负责任实践的需要。
- 强调隐私保护和维护价值观的重要性。
与主题相关。
```

# 使用提取的特征生成新内容

从输入文档中提取出所需特征后，你可以利用这些信息来指导未来的内容生成。请记住，通过在提示词中指定所需风格，确保 AI 模型将其输出与提取的特征相结合。例如：

- `Write a new blog post on [topic] using the same tone of voice, length, vocabulary, and structure as the previously analyzed text.`

通过将此技术与*参考文本*（作为事实依据的文档）相结合，你可以创作出可信、品牌化的内容，且几乎不需要修改。

# 使用大语言模型提取特定文本特征

你可以轻松定制提示词，引导大语言模型从文档中提取特定的文本特征。这不仅限于为文案目的分析文本。例如，通过为 LLM 制定精确的指令，可以实现从文本中识别实体或辨别情感。

输入：

```
分析以下文本以识别和列出提到的实体：

人工智能正在迅速改变我们周围的世界，
开辟新的可能性并颠覆行业。从
从医疗保健到娱乐，人工智能正在使我们的生活变得更好，更加
方便之处不胜枚举。然而，权力越大，责任越大，
责任。我们必须确保人工智能技术得到开发和
以道德和负责任的方式部署，以保护我们的隐私和
维护我们的价值观。
```

输出：

```
实体：
- 人工智能（AI）
- 医疗保健
- 娱乐
- 隐私
- 价值观
```

# 摘要生成

在信息过载的时代，将大量文本压缩成简洁、易于理解的摘要的能力已成为一项基本技能。随着我们生产和消费的数据量持续增长，对高效摘要技术的需求也显著增加。

人工智能在这一领域取得了重大进展，提供了强大的工具来生成保留最关键信息同时舍弃不太相关细节的摘要。

让我们探讨如何使用语言模型进行摘要。

输入：

```
总结以下文本：

第一方当事人，以下称为
"购买者"，以及第二方当事人，以下称为
作为完全了解各自法律
根据适用的法定
规定和普通法原则，已依法
约束力，以下称为“协议”，目的是实现涉及货物转让的交易，
目的，以实现涉及货物转让的交易，
以下称为“货物”，在购买者和卖方之间，
应对教育系统日益增长的需求。

我们郑重声明和确认以下内容：
...
```

输出：

```
这是一个对购买者和卖方都具有法律约束力的协议
用于货物转让。协议将序言作为
是一个不可分割的部分，但在任何冲突的情况下，适用的
协议普遍适用。解释条款规定，单数词语
包括复数和单数，以及指代性别的词语包括
所有性别。对法定规定的提及包括任何修正或
替换。
```



Summarization is an invaluable application of AI, enabling users to quickly extract key insights from lengthy articles, reports, or research papers. This process can help individuals make informed decisions, save time, and prioritize their reading. AI-generated summaries can also facilitate information sharing among teams, allowing for more efficient collaboration and communication.

# Summarizing Given Context Window Limitations

For documents larger than an LLM can handle in a single API request, a common approach is to chunk the document, summarize each chunk, and then combine these summaries into a final summary, as shown in Figure 3-4.

![.A summarization pipeline that uses text splitting and multiple summarization steps.](img/pega_0304.png)

###### Figure 3-4. A summarization pipeline that uses text splitting and multiple summarization steps

Additionally, people may require different types of summaries for various reasons, and this is where AI summarization comes in handy. As illustrated in the preceding diagram, a large PDF document could easily be processed using AI summarization to generate distinct summaries tailored to individual needs:

Summary A

Provides key insights, which is perfect for users seeking a quick understanding of the document’s content, enabling them to focus on the most crucial points

Summary B

On the other hand, offers decision-making information, allowing users to make informed decisions based on the content’s implications and recommendations

Summary C

Caters to collaboration and communication, ensuring that users can efficiently share the document’s information and work together seamlessly

By customizing the summaries for different users, AI summarization contributes to increased information retrieval for all users, making the entire process more efficient and targeted.

Let’s assume you’re only interested in finding and summarizing information about the advantages of digital marketing. Simply change your summarization prompt to `Provide a concise, abstractive summary of the above text. Only summarize the advantages: ...`

AI-powered summarization has emerged as an essential tool for quickly distilling vast amounts of information into concise, digestible summaries that cater to various user needs. By leveraging advanced language models like GPT-4, AI summarization techniques can efficiently extract key insights and decision-making information, and also facilitate collaboration and communication.

As the volume of data continues to grow, the demand for effective and targeted summarization will only increase, making AI a crucial asset for individuals and organizations alike in navigating the Information Age.

# Chunking Text

LLMs continue to develop and play an increasingly crucial role in various applications, as the ability to process and manage large volumes of text becomes ever more important. An essential technique for handling large-scale text is known as *chunking.*

*Chunking* refers to the process of breaking down large pieces of text into smaller, more manageable units or chunks. These chunks can be based on various criteria, such as sentence, paragraph, topic, complexity, or length. By dividing text into smaller segments, AI models can more efficiently process, analyze, and generate responses.

Figure 3-5 illustrates the process of chunking a large piece of text and subsequently extracting topics from the individual chunks.

![Topic extraction process after chunking text.](img/pega_0305.png)

###### Figure 3-5. Topic extraction with an LLM after chunking text

## Benefits of Chunking Text

There are several advantages to chunking text, which include:

Fitting within a given context length

LLMs only have a certain amount of input and output tokens, which is called a *context length*. By reducing the input tokens you can make sure the output won’t be cut off and the initial request won’t be rejected.

Reducing cost

Chunking helps you to only retrieve the most important points from documents, which reduces your token usage and API costs.

Improved performance

Chunking reduces the processing load on LLMs, allowing for faster response times and more efficient resource utilization.

Increased flexibility

Chunking allows developers to tailor AI responses based on the specific needs of a given task or application.

## Scenarios for Chunking Text

Chunking text can be particularly beneficial in certain scenarios, while in others it may not be required. Understanding when to apply this technique can help in optimizing the performance and cost efficiency of LLMs.

### When to chunk

Large documents

When dealing with extensive documents that exceed the maximum token limit of the LLM

Complex analysis

In scenarios where a detailed analysis is required and the document needs to be broken down for better comprehension and processing

Multitopic documents

When a document covers multiple topics and it’s beneficial to handle them individually

### When not to chunk

Short documents

When the document is short and well within the token limits of the LLM

Simple analysis

In cases where the analysis or processing required is straightforward and doesn’t benefit from chunking

Single-topic documents

When a document is focused on a single topic and chunking doesn’t add value to the processing

## Poor Chunking Example

When text is not chunked correctly, it can lead to reduced LLM performance. Consider the following paragraph from a news article:

```
Google My Business 个人资料，以便在人们寻找

今年，这一举措受到了家长和教师的普遍欢迎。该

额外的资金将用于改善学校基础设施，雇佣更多

这是另一个句子。

认为增加的幅度不足以应对教育系统日益增长的需求。

print(chunk)
```

When the text is fragmented into isolated words, the resulting list lacks the original context:

```
["The", "local", "council", "has", "decided", "to", "increase", "the",

"预算"，...]
```

The main issues with this poor chunking example include:

Loss of context

By splitting the text into individual words, the original meaning and relationships between the words are lost. This makes it difficult for AI models to understand and respond effectively.

Increased processing load

Processing individual words requires more computational resources, making it less efficient than processing larger chunks of text.

As a result of the poor chunking in this example, an LLM may face several challenges:

* Difficulty understanding the main ideas or themes of the text
* Struggling to generate accurate summaries or translations
* Inability to effectively perform tasks such as sentiment analysis or text `classification`

By understanding the pitfalls of poor chunking, you can apply prompt engineering principles to improve the process and achieve better results with AI language models.

Let’s explore an improved chunking example using the same news article paragraph from the previous section; you’ll now chunk the text by sentence:

```
["""当地议会决定增加教育预算

今年增长 10%，这一举措受到了家长和教师的普遍欢迎。

""",

"""额外的资金将用于改善学校基础设施，

雇佣更多教师，并为学生提供更好的资源。",

"""然而，一些批评者认为增加的幅度不足以

"""]
```



# Divide Labor and Evaluate Quality

Define the granularity at which the text should be chunked, such as by sentence, paragraph, or topic. Adjust parameters like the number of tokens or model temperature to optimize the chunking process.

By chunking the text in this manner, you could insert whole sentences into an LLM prompt with the most relevant sentences.

# Chunking Strategies

There are many different chunking strategies, including:

**Splitting by sentence**

Preserves the context and structure of the original content, making it easier for LLMs to understand and process the information. Sentence-based chunking is particularly useful for tasks like summarization, translation, and sentiment analysis.

**Splitting by paragraph**

This approach is especially effective when dealing with longer content, as it allows the LLM to focus on one cohesive unit at a time. Paragraph-based chunking is ideal for applications like document analysis, topic modeling, and information extraction.

**Splitting by topic or section**

This method can help AI models better identify and understand the main themes and ideas within the content. Topic-based chunking is well suited for tasks like text classification, content recommendations, and clustering.

**Splitting by complexity**

For certain applications, it might be helpful to split text based on its complexity, such as the reading level or technicality of the content. By grouping similar complexity levels together, LLMs can more effectively process and analyze the text. This approach is useful for tasks like readability analysis, content adaptation, and personalized learning.

**Splitting by length**

This technique is particularly helpful when working with very long or complex documents, as it allows LLMs to process the content more efficiently. Length-based chunking is suitable for applications like large-scale text analysis, search engine indexing, and text preprocessing.

**Splitting by tokens using a tokenizer**

Utilizing a tokenizer is a crucial step in many natural language processing tasks, as it enables the process of splitting text into individual tokens. Tokenizers divide text into smaller units, such as words, phrases, or symbols, which can then be analyzed and processed by AI models more effectively. You’ll shortly be using a package called `tiktoken`, which is a bytes-pair encoding tokenizer (BPE) for chunking.

Table 3-1 provides a high-level overview of the different chunking strategies; it’s worth considering what matters to you most when performing chunking.

Are you more interested in preserving semantic context, or would naively splitting by length suffice?

**Table 3-1. Six chunking strategies highlighting their advantages and disadvantages**

| Splitting strategy | Advantages | Disadvantages |
| --- | --- | --- |
| Splitting by sentence | Preserves context, suitable for various tasks | May not be efficient for very long content |
| Splitting by paragraph | Handles longer content, focuses on cohesive units | Less granularity, may miss subtle connections |
| Splitting by topic | Identifies main themes, better for classification | Requires topic identification, may miss fine details |
| Splitting by complexity | Groups similar complexity levels, adaptive | Requires complexity measurement, not suitable for all tasks |
| Splitting by length | Manages very long content, efficient processing | Loss of context, may require more preprocessing steps |
| Using a tokenizer: Splitting by tokens | Accurate token counts, which helps in avoiding LLM prompt token limits | Requires tokenization, may increase computational complexity |

By choosing the appropriate chunking strategy for your specific use case, you can optimize the performance and accuracy of AI language models.

# Sentence Detection Using SpaCy

*Sentence detection*, also known as sentence boundary disambiguation, is the process used in NLP that involves identifying the start and end of sentences within a given text. It can be particularly useful for tasks that require preserving the context and structure of the original content. By splitting the text into sentences, LLMs can better understand and process the information for tasks such as summarization, translation, and sentiment analysis.

Splitting by sentence is possible using NLP libraries such as [spaCy](https://spacy.io). Ensure that you have spaCy installed in your Python environment. You can install it with `pip install spacy`. Download the `en_core_web_sm` model using the command `python -m spacy download en_core_web_sm`.

In Example 3-3, the code demonstrates sentence detection using the spaCy library in Python.

**Example 3-3. [Sentence detection with spaCy](https://oreil.ly/GKDnc)**

```py
import spacy

nlp = spacy.load("en_core_web_sm")

text = "这是一个句子。这是另一个句子。"

doc = nlp(text)

对于 doc.sents 中的 sent：
    print(sent.text)
```

Output:

```
这是一个句子。

当地议会决定将教育预算增加 10%
```

First, you’ll import the spaCy library and load the English model (`en_core_web_sm`) to initialize an `nlp` object. Define an input text with two sentences; the text is then processed with `doc = nlp(text)`, creating a `doc` object as a result. Finally, the code iterates through the detected sentences using the `doc.sents` attribute and prints each sentence.

# Building a Simple Chunking Algorithm in Python

After exploring many chunking strategies, it’s important to build your intuition by writing a simple chunking algorithm from scratch.

Example 3-4 shows how to chunk text based on the length of characters from the blog post “Hubspot - What Is Digital Marketing?” This file can be found in the Github repository at [content/chapter_3/hubspot_blog_post.txt](https://oreil.ly/30rlQ).

To correctly read the `hubspot_blog_post.txt` file, make sure your current working directory is set to the [content/chapter_3](https://oreil.ly/OHurh) GitHub directory. This applies for both running the Python code or launching the Jupyter Notebook server.

**Example 3-4. [Character chunking](https://oreil.ly/n3sNy)**

```py
with open("hubspot_blog_post.txt", "r") as f：
    text = f.read()

chunks = [text[i : i + 200] for i in range(0, len(text), 200)]

将内容分成块：
    print("-" * 20)
    老师，并为学生提供更好的资源。然而，一些批评者
```

Output:

```
对于许多本地企业来说，搜索引擎优化策略是优化

卖方，根据本文件规定的条款和条件，双方

与您提供的产品或服务相关。

--------------------

您提供的内容。

对于位于乔治亚州亚特兰大的本地书店 For Keeps Bookstore，它已经优化了其

Google My Business 个人资料以进行本地搜索引擎优化，以便在查询时显示。

“atlanta bookstore。”

--------------------

...(为简洁起见省略)...
```



首先，使用`open`函数打开文本文件`hubspot_blog_post.txt`，并将其内容读取到变量`text`中。然后，通过列表推导式创建一个块列表，其中每个`chunk`是`text`中200个字符的子串。

接着，使用`range`函数为每个200字符的子串生成索引，并通过`i:i+200`切片表示法从`text`中提取子串。

最后，遍历`chunks`列表中的每个块，并使用`print`将其输出到控制台。

可以看到，由于分块实现相对简单且仅基于长度，句子甚至单词之间会出现间隙。

因此，我们认为良好的NLP分块应具备以下特性：

*   保留完整的单词，理想情况下保留句子和说话者表达的上下文要点
*   处理句子跨多页的情况，例如从第1页到第2页
*   为每个`chunk`提供足够的令牌数，以便总输入令牌数能适当适配任何LLM的给定令牌上下文窗口

# 滑动窗口分块

*滑动窗口分块*是一种根据指定数量的字符、令牌或单词，将文本数据划分为重叠块（或*窗口*）的技术。

但滑动窗口到底是什么？

想象一下透过一个小窗口查看长文本。这个窗口每次只能显示固定数量的项目。当您将窗口从文本开头滑动到结尾时，您会看到*重叠的文本块*。这种机制构成了滑动窗口方法的本质。

每个窗口大小由*固定数量的字符、令牌或单词*定义，而*步长*决定窗口每次滑动的距离。

在图3-6中，窗口大小为4个单词，步长为1，第一个块将包含文本的前4个单词。然后窗口向右滑动1个单词以创建第二个块，其中包含第2到第5个单词。

此过程重复直到到达文本末尾，确保每个块与前后块重叠，以保留一些共享上下文。

![pega 0306](img/pega_0306.png)

###### 图3-6\. 滑动窗口，窗口大小为4，步长为1

由于步长为1，块之间存在大量重复信息，同时块之间丢失信息的风险也大大降低。

这与图3-7形成鲜明对比，图3-7的窗口大小为4个单词，步长为2。您会注意到，由于步长增加了100%，块之间共享的信息量大大减少。

![pega 0307](img/pega_0307.png)

###### 图3-7\. 滑动窗口，窗口大小为4，步长为2

如果准确性和保留语义上下文比最小化令牌输入或对LLM的请求次数更重要，您可能需要更大的重叠。

示例3-5展示了如何使用Python的`len()`函数实现滑动窗口。`len()`函数为我们提供给定文本字符串中的总字符数（而非单词数），这有助于定义滑动窗口的参数。

##### 示例3-5\. [滑动窗口](https://oreil.ly/aCkDo)

```
def sliding_window(text, window_size, step_size):

    如果 window_size > len(text)或 step_size < 1：

        return []

    return [text[i:i+window_size] for i

    in range(0, len(text) - window_size + 1, step_size)]

text = "This is an example of sliding window text chunking."

window_size = 20

step_size = 5

chunks = sliding_window(text, window_size, step_size)

for idx, chunk in enumerate(chunks):

    print(f"Chunk {idx + 1}: {chunk}")
```

此代码输出：

```
Chunk 1: This is an example o
Chunk 2: is an example of sli
Chunk 3:  example of sliding
Chunk 4: ple of sliding windo
Chunk 5: f 滑动窗口文本
Chunk 6: ding 窗口文本
Chunk 7: window text chunking
```

在提示工程的背景下，滑动窗口方法相比固定分块方法提供了多项优势。它允许LLM保留更高程度的上下文，因为块之间存在重叠，并提供了一种与句子检测相比保留上下文的替代方法。

# 文本分块包

在使用GPT-4等LLM时，始终注意最大上下文长度：

*   `maximum_context_length = input_tokens + output_tokens`

有多种分词器可用于将文本分解为可管理的单元，最流行的包括NLTK、spaCy和tiktoken。

[NLTK](https://oreil.ly/wTmI7)和[spaCy](https://oreil.ly/c4MvQ)都提供全面的文本处理支持，但您将重点关注tiktoken。

# 使用Tiktoken进行文本分块

[Tiktoken](https://oreil.ly/oSpVe)是一种快速的*字节对编码（BPE）*分词器，可将文本分解为子词单元，专为OpenAI的模型设计。Tiktoken比同类开源分词器提供更快的性能。

作为使用GPT-4应用程序的开发人员，使用tiktoken为您提供几个关键优势：

准确的令牌分解

将文本划分为令牌至关重要，因为GPT模型将文本解释为单个令牌。识别文本中的令牌数量有助于您判断文本是否过长而无法由模型处理。

有效的资源利用

拥有正确的令牌计数使您能够高效管理资源，尤其是在使用OpenAI API时。了解确切的令牌数量有助于您调节和优化API使用，在成本和资源使用之间保持平衡。

# 编码

编码定义了将文本转换为令牌的方法，不同模型使用不同的编码。Tiktoken支持OpenAI模型常用的三种编码：

| 编码名称 | OpenAI模型 |
| --- | --- |
| cl100k_base | GPT-4, GPT-3.5-turbo, text-embedding-ada-002 |
| p50k_base | Codex模型, text-davinci-002, text-davinci-003 |
| r50k_base (或gpt2) | GPT-3模型如davinci |

## 理解字符串的令牌化

在英语中，令牌的长度可以变化，从单个字符如*t*，到整个单词如*great*。这是由于令牌化的适应性特性，它可以适应复杂脚本语言中比字符更短的令牌，或适应无空格语言中比单词更长的令牌，或短语作为单个单元的情况。

令牌中包含空格并不罕见，例如`"is"`而不是`"is "`或`" "+"is"`。这种做法有助于保持原始文本格式，并能捕捉特定的语言特征。

###### 注意

要轻松检查字符串的令牌化，您可以使用[OpenAI Tokenizer](https://oreil.ly/K6ZQK)。

您可以使用`pip install tiktoken`从[PyPI安装tiktoken](https://oreil.ly/HA2QD)。在以下示例中，您将看到如何轻松地将文本编码为令牌并将令牌解码为文本：

```
# 1\. 导入包：
import tiktoken

# 2\. 使用 tiktoken.get_encoding() 加载编码
encoding = tiktoken.get_encoding("cl100k_base")

# 3\. 使用 encoding.encode() 将一些文本转换为标记
# 而将标记转换为文本时使用 encoding.decode()
print(encoding.encode("Learning how to use Tiktoken is fun!"))
print(encoding.decode([1061, 15009, 374, 264, 2294, 1648,
311, 4048, 922, 15592, 0]))

# [48567, 1268, 311, 1005, 73842, 5963, 374, 2523, 0]
# "Data engineering is a great way to learn about AI!"
```

此外，让我们编写一个函数，给定`text_string`和`encoding_name`，对文本进行令牌化并计算令牌数量。

```
def count_tokens(text_string: str, encoding_name: str) -> int:
    """
返回使用给定编码的文本字符串中的标记数量。

Args:
text: 要进行标记化的文本字符串。
encoding_name: 要用于标记化的编码的名称。

Returns:
文本字符串中的标记数量。

Raises:
ValueError: 如果编码名称不被识别。
"""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text_string))
```



# 4. 使用该函数计算文本字符串中的标记数量

```py
text_string = "Hello world! This is a test."
print(count_tokens(text_string, "cl100k_base"))
```

这段代码输出 `8`。

# 估算 Chat API 调用的标记使用量

ChatGPT 模型（如 `GPT-3.5-turbo` 和 `GPT-4`）在标记使用方式上与之前的补全模型类似。然而，基于消息的结构使得对话的标记计数更具挑战性：

```py
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """返回由消息列表使用的标记数量。"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("警告：模型未找到。使用 cl100k_base 编码。")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # 每条消息后跟随
        # <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # 如果有名称，则省略角色
    elif "gpt-3.5-turbo" in model:
        print('''警告：gpt-3.5-turbo 可能会随时间更新。返回
num_tokens_assuming_gpt-3.5-turbo-0613.''')
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print('''警告：gpt-4 可能会随时间更新。
返回假设使用 gpt-4-0613 的标记数量。''')
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() 未实现对于模型
            {model}."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # 每个回复都预先填充
    return num_tokens
```

示例 3-6 突出了向任何聊天模型（目前是 `GPT-3x` 和 `GPT-4`）发起请求所需的具体结构。

通常，聊天历史首先以 `system` 消息结构化，然后由 `user` 和 `assistant` 之间的交替交流接续。

##### 示例 3-6. OpenAI 上 Chat Completions API 的负载

```py
{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
            "role": "system",
            "content": "您是一个有帮助的、遵循模式的助手。"
        },
        {
            "role": "user",
            "content": "当然！以下是一个基本 Flask \"Hello World\"应用程序的概述："
        },
        {
            "role": "assistant",
            "name": "example_assistant",
            "content": "导入 Flask 模块：导入 Flask 模块"
        },
        {
            "role": "user",
            "name": "example_user",
            "content": "用 Flask 类表示您的 Web 应用程序。"
        },
        {
            "role": "assistant",
            "content": "创建 Flask 应用程序的实例：创建一个 Flask 应用程序的实例。"
        },
        {
            "role": "user",
            "content": "定义路由和视图函数：..."
        },
        {
            "role": "assistant",
            "content": "运行应用程序：启动 Flask 开发服务器以运行应用程序。"
        }
    ]
}
```

```py
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run()
```

`"role": "system"` 描述了一条系统消息，对于*提供提示指令*很有用。它提供了一种调整助手角色或提供有关其交互方式的明确指令的方法。但重要的是要理解，系统命令不是必需的，没有系统命令时模型的默认行为可能非常接近“你是一个有帮助的助手”的行为。

您可以拥有的角色是 `["system", "user", "assistant"]`。

`"content": "Some content"` 是您放置提示或语言模型响应的地方，具体取决于消息的角色。它可以是 `"assistant"`、`"system"` 或 `"user"`。

# 情感分析

*情感分析*是一种广泛使用的 NLP 技术，有助于识别、提取和理解文本中表达的情感、观点或情绪。通过利用像 `GPT-4` 这样的 LLM 的力量，情感分析已成为各行各业企业、研究人员和开发人员的重要工具。

情感分析的主要目标是确定文本中传达的态度或情绪基调，无论是积极的、消极的还是中立的。这些信息可以为消费者对产品或服务的看法提供有价值的见解，帮助监控品牌声誉，甚至有助于预测市场趋势。

以下是创建有效情感分析提示的几种提示工程技术：

**输入：**

```py
text = "我绝对喜欢这部手机的造型，但电池寿命相当令人失望。"
```

**输出：**

```py
这段文字是积极的还是消极的？或者中立：我绝对喜欢这部手机的造型，但电池
寿命相当令人失望。
```

尽管 `GPT-4` 识别出“混合语气”，但该结果是提示中几个缺点的结果：

**缺乏清晰度**

提示未明确定义所需的输出格式。

**示例不足**

提示未包含任何积极、消极或中性情感的示例，这有助于引导 LLM 理解它们之间的区别。

**没有关于处理混合情感的指导**

提示未指定如何处理文本包含积极和消极情感混合的情况。

**输入：**

```py
{
    "role": "system",
    "content": "使用以下示例作为指南："
},
{
    "role": "user",
    "content": "请将以下文本的情感分类为积极、消极或中立："
},
{
    "role": "assistant",
    "name": "example_assistant",
    "content": "事物协同工作将增加收入。"
},
{
    "role": "user",
    "content": "客户交付成果需要花费大量时间。"
},
{
    "role": "assistant",
    "name": "example_assistant",
    "content": "这次晚期的转变意味着我们没有带宽来触及基于增加杠杆的机会。"
}
```

**输出：**

```py
积极的一面是 "我绝对喜欢这部手机的造型，"
而消极的部分是 "电池寿命相当令人失望。"
```

**示例：**

```py
import unittest

# 提供一个 Flask 中简单 "Hello World" 路由的测试用例示例
class TestHelloWorld(unittest.TestCase):
    def test_hello(self):
        # 测试代码
        pass
```

```py
# 使用该函数计算标记数量
example_messages = [
    {
        "role": "system",
        "content": "您是一个有帮助的助手。"
    },
    {
        "role": "user",
        "content": "你好！"
    }
]

for model in ["gpt-3.5-turbo-0301", "gpt-4-0314"]:
    print(f"{num_tokens_from_messages(example_messages, model)}")
```

```py
# 情感分析示例
text = "我绝对喜欢这部手机的造型，但电池寿命相当令人失望。"
# 输出：积极的一面是 "我绝对喜欢这部手机的造型，" 而消极的部分是 "电池寿命相当令人失望。"
```



# 提示工程与情感分析

## 改进后的提示词

这个提示词更好，因为它：

- **提供清晰的指令**：明确说明了任务，即把给定文本的情感分类为积极、消极或中性三个类别之一。
- **提供示例**：为每个情感类别提供了示例，有助于理解上下文和期望的输出。
- **定义输出格式**：指定输出应为单个单词，确保响应简洁且易于理解。

## 改进情感分析的技术

为了提高情感分析的准确性，预处理输入文本是至关重要的一步，这包括以下内容：

- **移除特殊字符**：表情符号、话题标签和标点等特殊字符可能会干扰基于规则的情感算法判断。此外，机器学习与深度学习模型可能无法识别这些字符，从而导致错误分类。
- **转换为小写**：将所有字符转换为小写有助于保持一致性。例如，模型会将`Happy`和`happy`视为不同的单词，这可能导致重复和不准确。
- **拼写纠正**：拼写错误可能导致误解和错误分类。构建一个拼写检查流程可以显著减少此类错误并改善结果。

对于行业或领域特定的文本，在提示词中嵌入领域特定内容有助于引导大语言模型（LLM）理解文本的框架和情感。这能提高分类的准确性，并加深对特定术语和表达方式的理解。

## 情感分析的局限性与挑战

尽管大语言模型（LLM）取得了进步，并且应用了提示工程技巧，但情感分析仍面临一些局限性和挑战：

- **处理讽刺和反语**：LLM难以检测文本中的讽刺和反语，因为这通常需要理解上下文和人类容易识别的微妙线索。误解讽刺或反语的陈述可能导致不准确的情感分类。
- **识别特定上下文的情感**：在处理特定上下文的情感（如领域特定术语或文化表达相关的情感）时，情感分析可能具有挑战性。如果没有适当的指导或领域特定示例，LLM可能难以准确分类这些情况下的情感。

# 从少到多（Least to Most）

提示工程中的*从少到多*技术是一种强大的方法，用于在给定主题上顺序生成或提取越来越详细的知识。这种方法在处理复杂主题或需要高度细节时特别有效。

从少到多使用一个*提示链*，其中每个新提示都基于上一个答案。这种逐步方法有助于每次收集更详细的信息，使得深入探讨任何主题都更加容易。

该技术也可以应用于代码生成，正如一个Flask `Hello World`应用示例所展示的那样。

## 规划架构

在深入架构之前，让我们简要了解一下什么是Flask。[Flask](https://oreil.ly/7N-bs)是Python中一个轻量级的Web应用框架，广泛用于快速创建Web应用且代码量极少。（此处仅用于演示目的，并未包含在本书的[*requirements.txt*文件](https://oreil.ly/TRK0i)中。）

现在，让我们要求一个LLM为一个简单的Flask “Hello World”应用概述基本架构。

**输入：**

```py
我绝对喜欢这部手机的造型，但电池寿命相当
```

**输出：**

```py
- 中立
```

**输入：**

```py
积极的："我绝对喜欢这部手机的造型！"
中立："我喜欢这个产品，但它的电池寿命很短。"
概述一个简单的 Flask "Hello World"应用程序的基本架构。
"name": "example_user",
```

**输出：**

```py
当然！以下是一个使用 Flask 的 "Hello World" 路由的简单代码片段：
中立
num_tokens += tokens_per_name
```

## 编写各个函数

在编码之前，让我们澄清一下Flask路由是什么：它是一个链接到URL模式的函数，决定了用户在特定网页上看到的内容。接下来，我们将为Flask中的简单“Hello World”路由提供代码片段。

**输入：**

```py
if __name__ == '__main__':
```

**输出：**

```py
将公司术语翻译成普通英语。'''
提供一个 Flask 中 "Hello World" 路由的测试用例示例。
"content": "新的协同效应将有助于推动收入增长。",
实现视图函数：...
def hello_world():
    {
},
    "role": "system",
```

**代码：**

```py
In this code, we import the Flask module and create a Flask application instance named `app`. We then define a route using the `@app.route` decorator, specifying the root URL path `'/'`. The `hello_world` function is associated with this route, and it returns the string `'Hello, World!'` as the response.

The Flask application will start by calling `app.run()` when your Python script is executed directly.
```

## 添加测试

最后，让我们要求它为Flask应用中的“Hello World”路由生成一个测试用例。

**输入：**

```py
这段文字的语气是混合的，因为它包含了积极和消极的内容。
只返回以下单词中的一个：
```

**输出：**

```py
prompt tokens counted by num_tokens_from_messages().''')
消极的："电池寿命相当令人失望。"
example token count from the function defined above
```

**测试代码：**

```py
from flask import Flask
from flask.testing import FlaskClient

# 创建 Flask 应用程序
app = Flask(__name__)

# 创建路由
@app.route('/')
def hello_world():
    return 'Hello, World!'

class FlaskAppTestCase(unittest.TestCase):
    def setUp(self):
        # 创建测试客户端
        self.app = app.test_client()

    def test_hello_world_route(self):
        # 向路由发送 GET 请求
        response = self.app.get('/')
        # 检查响应状态码
        self.assertEqual(response.status_code, 200)
        # 检查响应数据
        self.assertEqual(response.data, b'Hello, World!')

if __name__ == '__main__':
    unittest.main()
```



通过使用“从少到多”的提示技术，你可以逐步构建处理当前任务所需的知识和理解。

## 从少到多技术的好处

这种方法对于复杂任务特别有用，因为它允许大型语言模型（LLM）生成相关知识，这些知识随后将作为未来任务的上下文。

让我们深入探讨在各种其他场景中使用这种方法的好处：

**渐进式探索**

将复杂问题分解为较小的任务，可以让LLM在每一步提供更详细和准确的信息。在处理新主题或多方面问题时，这种方法尤其有帮助。

**灵活性**

从少到多技术提供了处理问题不同方面的灵活性。它使您能够根据需要调整方向、探索替代解决方案或深入特定领域。

**提高理解力**

通过将任务分解为更小的步骤，LLM可以以更易于消化的格式提供信息，使您更容易理解和跟进。

**协作学习**

该技术促进了您与LLM之间的协作，因为它鼓励迭代过程，不断优化输出并调整您的响应以达到预期结果。

## 从少到多技术的挑战

**过度依赖先前生成的知识**

使用先前的聊天历史来存储状态，可能会因上下文长度限制而导致较大任务忘记其初始任务/输出。

**依赖先前的提示**

由于每个提示都建立在前面的输出之上，因此必须确保先前提示的质量和响应能为下一步提供足够的信息。

# 评估质量

在设计提示的过程中，务必评估每个先前LLM响应的质量。下一个任务的性能取决于前一个任务所提供信息的质量。

# 角色提示

*角色提示*是一种让AI在生成响应时承担特定角色或性格的技术。这有助于引导AI的响应风格和内容，更好地符合用户的需求或期望结果。

###### 注意

[Awesome ChatGPT prompts](https://oreil.ly/8pf40) 提供了大量可用的角色提示列表。

让我们让ChatGPT扮演MongoDB的技术评论员。

输入：

```
我想让你扮演技术评论员。我会给你一个文章的名字
技术和您将提供深入评论 - 包括
优点、缺点、特性以及与其他市场上技术的比较。
我的第一项建议请求是：MongoDB。
```

输出：

```
MongoDB 是一种流行的 NoSQL、面向文档的数据库管理系统
设计用于处理高容量数据存储和检索。

特性：
1. 面向文档的存储：MongoDB 以灵活、类似 JSON 的方式存储数据
格式称为 BSON，这使得存储复杂的数据结构变得容易
数组和嵌套对象。
...

优点：
1. 高性能：MongoDB 的索引和缓存功能提供
快速读写操作。
...

缺点：
1. 对事务的支持有限：虽然 MongoDB 支持多文档事务，
如果在 ACID 事务中找不到答案，它可能不是应用程序的最佳选择
需要复杂的交易操作。
...

与其他技术的比较：
1. Couchbase，另一个流行的 NoSQL 数据库，提供与 MongoDB 类似的功能
MongoDB，包括灵活的模式和基于 JSON 的存储。然而，
Couchbase 更专注于分布式缓存，可能更适合
适用于需要低延迟数据访问的应用。

结论：
MongoDB 是一种强大且灵活的 NoSQL 数据库解决方案，适用于
各种应用，尤其是那些需要高性能和
可扩展性。其灵活的数据模型、丰富的查询语言和强大的
一致性使其成为开发人员中的热门选择。
```

这个提示是角色提示的一个极好例子，因为它明确定义了AI应承担的角色（技术评论员），并设定了期望的响应类型（对MongoDB的深入评论）。

# 给予方向

在制作提示时，考虑为AI分配一个特定角色。这为响应设置了正确的上下文，从而产生更专注和相关的输出。

# 角色提示的好处

角色提示有助于缩小AI的响应范围，确保更专注、上下文适当和量身定制的结果。它还可以通过推动AI从独特的角度思考和回应来增强创造力。

# 角色提示的挑战

角色提示可能带来某些挑战。根据分配的角色，可能存在偏见或刻板印象的风险。分配刻板角色可能导致生成有偏见的响应，这可能损害可用性或冒犯个人。此外，在扩展交互过程中保持角色的一致性可能很困难。模型可能会偏离主题或提供与分配角色无关的信息。

# 评估质量

持续检查LLM响应的质量，尤其是在使用角色提示时。监控AI是否坚持分配的角色，或者是否偏离主题。

# 何时使用角色提示

角色提示在以下情况下特别有用：

**引出特定专业知识**

如果您需要需要领域知识或专业知识的响应，角色提示可以帮助引导LLM生成更明智和准确的响应。

**定制响应风格**

分配角色可以帮助LLM生成匹配特定语气、风格或视角的响应，例如正式、随意或幽默的响应。

**鼓励创造性响应**

角色提示可用于创建虚构场景或通过分配诸如讲故事者、小说角色或历史人物等角色来生成富有想象力的答案。

**探索多样视角**：如果您想探索某个主题的不同观点，角色提示可以通过让AI承担各种角色或人物来提供帮助，从而更全面地理解主题。

**增强用户参与度**：角色提示可以通过让LLM承担与用户产生共鸣的角色或人物，使互动更具吸引力和娱乐性。

如果您使用OpenAI，那么添加角色的最佳位置是在聊天模型的`System Message`中。

# GPT 提示策略

到目前为止，您已经涵盖了几种提示策略，包括请求上下文、文本风格捆绑、从少到多和角色提示。

让我们再介绍几种策略，从利用适当的参考文本管理潜在的幻觉，到为LLM提供关键的*思考时间*，再到理解*任务分解*的概念——我们有很多内容供您探索。

这些方法旨在显著提高AI输出的精度，并得到了[OpenAI](https://oreil.ly/QZE8n)的推荐。此外，每种策略都利用了第1章中讨论的一个或多个提示工程原则。

## 利用参考避免幻觉

避免基于文本的幻觉的第一种方法是指示模型*仅使用参考文本回答*。

通过向AI模型提供有关给定查询的准确和相关信息的参考，可以引导模型使用这些信息来生成其响应。

输入：

```
参考三引号内的文章以回答查询。
您必须遵循以下原则：
- 如果在这些文章中找不到答案，只需
return "I could not find an answer".
"""
B2B 客户往往有更长的决策过程，因此决策周期更长
销售漏斗。关系建立策略对这些
客户，而 B2C 客户往往对短期优惠反应更好
和消息。
"""
示例响应：
- 我找不到答案。
- 是的，B2B 客户往往有更长的决策过程，因此
更长的销售漏斗。
```

输出：

```
是的，B2B 客户往往有更长的决策过程，这导致
决策周期更长。
```

如果您向相同的参考文本询问这个问题：

输入：

```
...剩余提示...
问题：B2C 销售是否更划算？
```

输出：

```
我找不到答案。
```

# 给予方向并指定格式

上述提示非常出色，因为它既指示了模型如何找到答案，也为任何无法回答的问题设置了特定的响应格式。

考虑到GPT的上下文窗口受限，可能需要一种动态检索与所提查询相关信息的方法来利用此策略。

另一种方法是指导模型在其响应中*包含引用*。当输入增强了相关信息时，可以引导模型通过引用所提供文档的章节来在响应中包含引用。这种方法还有一个额外的好处，即输出中的引用可以通过*匹配给定文档中的字符串来自动验证*。

输入：

```
您将提供由三引号分隔的文档，并
```



## Give GPTs “Thinking Time”

Often, by explicitly guiding an LLM to *derive solutions from first principles* before reaching a verdict, you can garner more accurate responses. Providing an LLM with *thinking time* can often lead to better results.

Input:

```
```

Output:

```
```

In some cases, when using GPT-4 or other generative AI models, you may not want the model’s reasoning process to be visible to the user.

You can achieve this by asking an LLM to *generate an inner monologue*. This is particularly useful in tutoring applications, where revealing the model’s reasoning might give away the solution prematurely.

## The Inner Monologue Tactic

The *inner monologue tactic* instructs the model to structure parts of the output that should be hidden from the user in a specific format. This makes it easy to remove these parts before presenting the final output to the user.

Here’s how you can utilize this tactic to answer user queries effectively.

Input:

```
指导他们找到正确答案，而不直接给出答案。

这个部分也应该在三个引号内。

第四步：交付提示

如果有错误，向用户展示上一步的提示。

记住不要在这个步骤中使用三引号。而不是使用标题

一样，使用 `提示:` 这样的标题

问题陈述：一副牌有 52 张牌：13 张红心、13 张方块、13

梅花、13 张黑桃。如果从牌组中抽取两张牌而不进行

抽牌，那么抽出的牌中至少有一张红心的概率是多少

和另一张是梅花吗？

用户解决方案：（13/52）*（13/51）= 169/2652 = 13/204
```

Output:

```
"""

第一步 - 我的解决方案：

首先，计算从牌组中抽取 1 张红心和 1 张梅花的概率

52 张牌的牌组，不进行抽牌。

在第一次抽取中，抽取红心的概率是 13/52。

对于第二次抽取，在移除 1 张牌（假设是红心）后，剩余的

总牌数变为 51 张。因此，现在抽取梅花的概率是

是 13/51。

将这两个概率相乘，得到两个事件同时发生的概率

按顺序发生，即 P(红心然后梅花) = (13/52)*(13/51)。

假设第一张抽出的牌是梅花，然后重复计算

第二次抽取的牌是红心。记住，在这种情况下，顺序很重要，

我们处理的是独立事件。

"""

"""

第二步 - 与用户解决方案的比较：

将你的解决方案与用户解决方案进行比较。用户的解决方案只

考虑先抽红心然后抽梅花的顺序。他们

没有考虑到另一种序列（梅花然后红心），并且

因此，解决方案是不完整的。

"""

"""

第三步 - 提示准备：

由于用户没有考虑两种序列（红心然后梅花

以及梅花然后红心），这是对考虑两种可能

序列将有助于引导用户走向正确的方向。这可能会

方向而不直接透露解决方案。

"""

# 第四步 - 提示交付：

# 提示：记住两张牌可以以任何顺序抽取。考虑

两种可能的顺序：红心然后梅花和梅花然后红心。
```

## Self-Eval LLM Responses

Another tactic you can use is to *critque a generated LLM output* and ask whether the LLM missed any information or important facts. You’re essentially asking an LLM to *evaluate itself* based on its previous output.

First, create a prompt to generate a `"Hello World"` function.

Input:

```
生成一个 "Hello World" Python 函数。
```

Output:

```
是的，这是一个非常简单的 Python 函数，它将 "Hello, World!" 打印到

控制台：

def print_hello_world():

    print("Hello, World!")

打印 "Hello, World!" 的函数。

当你运行这个函数时，它将打印字符串 "Hello, World!" 到

控制台。
```

The LLM returns a simple Python function called `print_hello_world()` that prints the traditional “Hello, World!” greeting.

Then make a second LLM request with the previous chat history, asking for the initial output to be improved.

Input:

```
...之前的聊天记录...

给定前面的函数，你能用其他方式再次改进它吗？

也许默认值或类型提示？
```

Output:

```
我们可以添加类型提示来清楚地表明该函数不接受任何

参数且不返回任何内容。我们还可以添加一个文档字符串来

描述函数的功能。

这里是改进后的代码：

def print_hello_world() -> None:

    """

    将 "Hello, World!" 打印到标准输出。

    """

    print("Hello, World!")

如果你想使函数更灵活，可以添加一个默认

允许你打印不同的消息：

def print_message(message: str = "Hello, World!") -> None:

    """

    打印给定的消息到标准输出。如果没有提供消息

    , 默认为 "Hello, World!"。

    Args:

        message (str, optional): 要打印的消息。

        默认为 "Hello, World!"。

    """

    print(message)
```

Consequently GPT-4 returns an upgraded version of the function, now furnished with Python type hints and a default argument. This enriches the function with greater flexibility, allowing it to print not just `"Hello, World!"` but any user-specified message.

These prompt-response exchanges illustrate how you can easily refine generated LLM outputs until you’re satisfied with the final output.

###### Note

It’s possible to critique an LLM’s response multiple times, until no further refinement is provided by the LLM.

# Classification with LLMs

Classifying, in the context of AI, refers to the process of predicting the class or category of a given data point or sample. It’s a common task in machine learning where models are trained to assign predefined labels to unlabeled data based on learned patterns.

LLMs are powerful assets when it comes to classification, even with zero or only a small number of examples provided within a prompt. Why? That’s because LLMs, like GPT-4, have been previously trained on an extensive dataset and now possess a degree of reasoning.

There are two overarching strategies in solving classification problems with LLMs: *zero-shot learning* and *few-shot learning*.

**Zero-shot learning**

In this process, the LLM classifies data with exceptional accuracy, without the aid of any prior specific examples. It’s akin to acing a project without any preparation—impressive, right?

**Few-shot learning**

Here, you provide your LLM with a small number of examples. This strategy can significantly influence the structure of your output format and enhance the overall classification accuracy.

**Why is this groundbreaking for you?**

Leveraging LLMs lets you sidestep lengthy processes that traditional machine learning processes demand. Therefore, you can quickly prototype a classification model, determine a base level accuracy, and create immediate business value.

###### Warning

Although an LLM can perform classification, depending upon your problem and training data you might find that using a traditional machine learning process could yield better results.

# Building a Classification Model

Let’s explore a few-shot learning example to determine the sentiment of text into either `'Compliment'`, `'Complaint'`, or `'Neutral'`.

```
Given the statement, classify it as either "Compliment", "Complaint", or

"中立":

1. "太阳正在照耀。" - 中立

2. "你们的支持团队太棒了！" - 表扬

3. "我使用你们的软件时遇到了糟糕的经历。" - 投诉

您必须遵循以下原则：

- 只返回单个分类词。响应应该是

"Compliment", "Complaint", or "Neutral".

- 在 """定界符内的文本上执行分类。

"""用户界面直观。”

Classification:
```

```
表扬
```

Several good use cases for LLM classification include:

**Customer reviews**

Classify user reviews into categories like “Positive,” “Negative,” or “Neutral.” Dive deeper by further identifying subthemes such as “Usability,” “Customer Support,” or “Price.”

**Email filtering**

Detect the intent or purpose of emails and classify them as “Inquiry,” “Complaint,” “Feedback,” or “Spam.” This can help businesses prioritize responses and manage communications efficiently.

**Social media sentiment analysis**

Monitor brand mentions and sentiment across social media platforms. Classify posts or comments as “Praise,” “Critic,” “Query,” or “Neutral.” Gain insights into public perception and adapt marketing or PR strategies accordingly.

**News article categorization**

Given the vast amount of news generated daily, LLMs can classify articles by themes or topics such as “Politics,” “Technology,” “Environment,” or “Entertainment.”

**Résumé screening**

For HR departments inundated with résumés, classify them based on predefined criteria like “Qualified,” “Overqualified,” “Underqualified,” or categorize by expertise areas such as “Software Development,” “Marketing,” or “Sales.”

###### Warning

Be aware that exposing emails, résumés, or sensitive data does run the risk of data being leaked into OpenAI’s future models as training data.

# Majority Vote for Classification

Utilizing multiple LLM requests can help in reducing the variance of your classification labels. This process, known as *majority vote*, is somewhat like choosing the most common fruit out of a bunch. For instance, if you have 10 pieces of fruit and 6 out of them are apples, then apples are the majority. The same principle goes for choosing the majority vote in classification labels.

By soliciting several classifications and taking the *most frequent classification*, you’re able to reduce the impact of potential outliers or unusual interpretations from a single model inference. However, do bear in mind that there can be significant downsides to this approach, including the increased time required and cost for multiple API calls.

Let’s classify the same piece of text three times, and then take the majority vote:

```
from openai import OpenAI

import os
```



```python
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
```

```python
base_template = """

Given the statement, classify it as either "Compliment", "Complaint", or

"中立":

1. "太阳正在照耀。" - 中立

2. "你们的支持团队太棒了！" - 表扬

3. "我使用你们的软件时遇到了糟糕的经历。" - 投诉

您必须遵循以下原则：

- 只返回单个分类词。响应应该是

"表扬", "投诉", 或 "中立"。

- 在 '''定界符内的文本上执行分类。

'''{content}'''

Classification:

"""
```

```python
responses = []

for i in range(0, 3):

    response = client.chat.completions.create(

        model="gpt-4",

        messages=[{"role": "system",

            "content": base_template.format(content='''外面下雨了，

但我过得非常愉快，我只是不明白人们

live, 我非常难过！'''),}],)

    responses.append(response.choices[0].message.content.strip())
```

```python
def most_frequent_classification(responses):

    # Use a dictionary to count occurrences of each classification

    count_dict = {}

    for classification in responses:

        count_dict[classification] = count_dict.get(classification, 0) + 1

    # 返回具有最大计数的分类

    return max(count_dict, key=count_dict.get)

print(most_frequent_classification(responses))  # 预期输出: 中立
```

调用`most_frequent_classification(responses)`函数应能准确识别出`'Neutral'`（中立）作为主导情感。至此，您已学会如何使用`OpenAI`包进行多数投票分类。

# 标准评估

在第 1 章中，我们使用了一个基于人工的评估系统，采用简单的点赞/点踩评分机制，以衡量响应符合预期的频率。人工评分既昂贵又繁琐，需要合格的人员来判断质量或识别错误。虽然可以将这项工作外包给诸如[Mechanical Turk](https://www.mturk.com)等平台上的低成本评分者，但设计一个能获得有效结果的任务本身就可能耗时且容易出错。一种日益普遍的做法是使用更复杂的`LLM`来评估较小模型的响应。

关于`LLM`能否作为有效评估者，现有证据褒贬不一：一些研究[声称`LLM`能达到人类水平的评估能力](https://oreil.ly/nfc3f)，另一些研究则[指出了`LLM`评估方式中的不一致性](https://oreil.ly/ykkzY)。根据我们的经验，`GPT-4`是一个有用的评估工具，在多种任务中都能提供一致的结果。特别是，`GPT-4`在评估较小、较不复杂的模型（如`GPT-3.5-turbo`）的响应时，表现出色且可靠。在下面的示例中，我们使用`GPT-3.5-turbo`生成对某个问题的简洁和冗长两种答案，并准备用`GPT-4`进行评分。

输入：

```python
from openai import OpenAI

import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

responses = []

for i in range(10):

    # concise if even, verbose if odd

    style = "concise" if i % 2 == 0 else "verbose"

    if style == "concise":

        prompt = f"""返回一个 {style} 答案到

following question: 生命的意义是什么？”"""

    else:

        prompt = f"""返回以下问题的答案

question: 生命的意义是什么？”"""

    response = client.chat.completions.create(

        # using GPT-3.5 Turbo for this example

        model="gpt-3.5-turbo",

        messages=[{"role": "user",

            "content": prompt}])

    responses.append(

        response.choices[0].message.content.strip())

system_prompt = """您正在评估提示的简洁性

response from a chatbot.

如果响应简洁，则只回复 1，如果不简洁，则回复 0。

and a 0 if it is not.

"""

ratings = []

for idx, response in enumerate(responses):

    rating = client.chat.completions.create(

        模型="gpt-4"，

        消息=[{"角色": "系统",

            "内容": system_prompt},

            {"角色": "系统",

            "内容": response}]

    ratings.append(

        rating.choices[0].message.content.strip())

for idx, rating in enumerate(ratings):

    样式 = "简洁" if idx % 2 == 0 else "冗长"

    print(f"样式：{style}, ", f"评分：{rating}")
```

输出：

```text
样式：简洁，评分：1

样式：冗长，评分：0

样式：简洁，评分：1

样式：冗长，评分：0

样式：简洁，评分：1

样式：冗长，评分：0

样式：简洁，评分：1

样式：冗长，评分：0

样式：简洁，评分：1

样式：冗长，评分：0
```



This script is a Python program that interacts with the OpenAI API to generate and evaluate responses based on their conciseness. Here’s a step-by-step explanation:

1. `responses = []` creates an empty list named `responses` to store the responses generated by the OpenAI API.
2. The `for` loop runs 10 times, generating a response for each iteration.
3. Inside the loop, `style` is determined based on the current iteration number (`i`). It alternates between “concise” and “verbose” for even and odd iterations, respectively.
4. Depending on the `style`, a `prompt` string is formatted to ask, “What is the meaning of life?” in either a concise or verbose manner.
5. `response = client.chat.completions.create(...)` makes a request to the OpenAI API to generate a response based on the `prompt`. The model used here is specified as “gpt-3.5-turbo.”
6. The generated response is then stripped of any leading or trailing whitespace and added to the `responses` list.
7. `system_prompt = """You are assessing..."""` sets up a prompt used for evaluating the conciseness of the generated responses.
8. `ratings = []` initializes an empty list to store the conciseness ratings.
9. Another `for` loop iterates over each response in `responses`.
10. For each response, the script sends it along with the `system_prompt` to the OpenAI API, requesting a conciseness evaluation. This time, the model used is “gpt-4.”
11. The evaluation rating (either 1 for concise or 0 for not concise) is then stripped of whitespace and added to the `ratings` list.
12. The final `for` loop iterates over the `ratings` list. For each rating, it prints the `style` of the response (either “concise” or “verbose”) and its corresponding conciseness `rating`.

For simple ratings like conciseness, GPT-4 performs with near 100% accuracy; however, for more complex ratings, it’s important to spend some time evaluating the evaluator. For example, by setting test cases that contain an issue, as well as test cases that do not contain an issue, you can identify the accuracy of your evaluation metric. An evaluator can itself be evaluated by counting the number of false positives (when the LLM hallucinates an issue in a test case that is known not to contain an issue), as well as the number of false negatives (when the LLM misses an issue in a test case that is known to contain an issue). In our example we generated the concise and verbose examples, so we can easily check the rating accuracy, but in more complex examples you may need human evaluators to validate the ratings.

# Evaluate Quality

Using GPT-4 to evaluate the responses of less sophisticated models is an emerging standard practice, but care must be taken that the results are reliable and consistent.

Compared to human-based evaluation, LLM-based or synthetic evaluation typically costs an order of magnitude less and completes in a few minutes rather than taking days or weeks. Even in important or sensitive cases where a final manual review by a human is necessary, rapid iteration and A/B testing of the prompt through synthetic reviews can save significant time and improve results considerably. However, the cost of running many tests at scale can add up, and the latency or rate limits of GPT-4 can be a blocker. If at all possible, a prompt engineer should first test using programmatic techniques that don’t require a call to an LLM, such as simply measuring the length of the response, which runs near instantly for close to zero cost.

# Meta Prompting

*Meta prompting* is a technique that involves the creation of text prompts that, in turn, generate other text prompts. These text prompts are then used to generate new assets in many mediums such as images, videos, and more text.

To better understand meta prompting, let’s take the example of authoring a children’s book with the assistance of GPT-4\. First, you direct the LLM to generate the text for your children’s book. Afterward, you invoke meta prompting by instructing GPT-4 to produce prompts that are suitable for image-generation models. This could mean creating situational descriptions or specific scenes based on the storyline of your book, which then can be given to AI models like Midjourney or Stable Diffusion. These image-generation models can, therefore, deliver images in harmony with your AI-crafted children’s story.

Figure 3-8 visually describes the process of meta prompting in the context of crafting a children’s book.

![Creating image prompts from an LLM that will later be used by MidJourney for image creation.](img/pega_0308.png)

###### Figure 3-8\. Utilizing an LLM to generate image prompts for MidJourney’s image creation in the process of crafting a children’s book

Meta prompts offer a multitude of benefits for a variety of applications:

**Image generation from product descriptions**

Meta prompts can be employed to derive an image generation prompt for image models like [Midjourney](https://www.midjourney.com), effectively creating a visual representation of product descriptions.

**Generating style/feature prompts**

Let’s consider you are a copywriter needing to develop a unique style guide prompt from a couple of blog posts. Given each client has a distinctive tone and style, it’s beneficial to utilize a *meta prompt* that encapsulates all the varied features, rather than producing a single prompt output.

**Optimizing prompts to achieve specific goals**

A common approach is to ask ChatGPT or another language model to refine or improve `Prompt A` in order to attain `Goal 1`, given `Prompt A` and `Goal 1`. This method aids in identifying any missed input features, that by adding could assist the language model in generating a more detailed and desirable response.

Let’s experiment with creating a meta prompt for the copywriting example.

**Input:**

```
作为 GPT 提示工程师，你唯一负责的是产生
高效的大型语言模型文本提示，给定一个任务。
你必须遵循以下原则：
- 仅返回单个提示，永远不要返回提示的输出。
- 你必须正确地完成任务。
- 你必须根据任务适当地进行提示或创建新的提示。
任务。
- 列出任何重要功能，以便另一个提示工程师
可以轻松地将输入复制到提示中。
任务：你必须为 ChatGPT 创建多个博客文章的写作指南。
在 ChatGPT 提示中包含相关的功能，这将允许 ChatGPT
为了模仿作者。你必须决定最重要的功能是什么
模仿作者的风格等。你将只应用此风格到
新的博客文章。
博客文章文本：你需要除了技术
技能以在技术领域取得成功。软技能被
软件专业人士与同行协作...
```

**Output:**

```
提示：
根据提供的广泛文本，你的任务是创建一系列
按照作者相同的风格和结构撰写博客文章。作为指南
你应该遵循以下特点：
1. 清晰、直接的语言：你的写作应该是清晰和直接的
语言，将复杂主题分解成易于理解
概念。
2. 使用子标题：为了为你的博客文章提供结构，将它们
使用相关的子标题。这有助于引导读者并使你的
文本更易于阅读。
...
9. 技术术语的包含：在讨论技术主题时，
使用适当的术语语言以确保准确性和可信度。
然而，务必提供解释或定义，以便更复杂
术语。
```

As previously mentioned, it can be beneficial to *critique the model’s output* and to ask for a refined prompt either via the interface or API.

ChatGPT can perform web searches and can make use of the content of web pages, along with the previously generated copywriting style guide. Let’s provide two URLs that ChatGPT will read to enhance the output.

**Input:**

```
根据以下风格指南和这两篇博客文章，我想让你创建一系列
改进写作风格提示，以便 LLM 或文案人员可以写作
有效地为我客户服务。
...之前生成的写作指南...
URL 1：https://understandingdata.com/posts/how-to-clean-and-process-data
URL 2：https://understandingdata.com/posts/how-to-scrape-twitter-data
```

Figure 3-9 shows the output.

![pega 0309](img/pega_0309.png)

###### Figure 3-9\. ChatGPT refining a meta prompt by two URL web pages

```
基于风格指南和两篇博客文章，以下是一个改进的
写作风格提示：
为客户创建博客文章时，请遵循以下指南：
1. 清晰、直接的语言：你的写作应该是清晰和直接的
主题。以易于理解的方式呈现信息。目标是你的
观众可以理解概念，而无需先前的知识。
...
8. 技术术语的使用：使用适当的技术语言
讨论复杂主题时。然而，确保提供定义
或对这些术语的解释，以保持文本的可访问性。
模仿这些博客文章的写作风格：
"什么是 API？它们如何与数据工程相关？": 这篇博客
这篇文章使用清晰直接的语言解释了 API 的概念。它在
提供了详细解释和常用术语的实用示例
APIs。它将 API 的技术概念与业务增长联系起来，
提高生产力。信息被很好地分类，使用子标题如
'API 的类型。'
"如何轻松设置带有 API 的 Google Cloud 项目"：这篇博客文章是一篇
设置 Google Cloud 项目的实用指南。它被分解成
编号步骤，每个步骤都有一个清晰的子标题。文章使用详细的
解释并包括如截图等实用示例。它还
使用清晰直接的语言引导读者通过这个过程。
```



元提示提供了一种动态和创新的方式，以利用生成式 AI 模型的力量，促进复杂、多方面的提示以及生成其他提示的提示的创建。它扩大了应用范围，从文本和图像生成到风格和功能提示，以及针对特定目标的优化。随着你继续完善和探索元提示的潜力，它承诺将改变你利用、交互和从使用 `LLMs` 中受益的方式。

# 摘要

在阅读这一章之后，你现在意识到给出清晰的指示和示例以生成所需输出是多么重要。此外，你通过使用 `Python` 中的正则表达式从分层列表中提取结构化数据获得了实践经验，并且你学习了如何利用嵌套数据结构如 `JSON` 和 `YAML` 来生成强大、可解析的输出。

你已经学习了几个最佳实践和有效的提示工程技巧，包括著名的“像对我五岁孩子解释”，角色提示和元提示技巧。在下一章中，你将学习如何使用一个流行的 `LLM` 包 `LangChain`，它将帮助你创建更高级的提示工程工作流程。
