
```
from reasoning_from_scratch.ch02 import (
    generate_text_basic_stream_cache
)

prompt = (  #1
    r"If $a+b=3$ and $ab=\tfrac{13}{6}$, "
    r"what is the value of $a²+b²$?"
)

input_token_ids_tensor = torch.tensor(  #2
    tokenizer.encode(prompt),
    device=device
).unsqueeze(0)  #3

all_token_ids = []

for token in generate_text_basic_stream_cache(  #4
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=2048,
    eos_token_id=tokenizer.eos_token_id
):
    token_id = token.squeeze(0)  #5
    decoded_id = tokenizer.decode(token_id.tolist())
    print(  #6
        decoded_id,   
        end="",
        flush=True
    )
    all_token_ids.append(token_id)

all_tokens = tokenizer.decode(all_token_ids)  #7
```

- `#1` 将数学问题定义为字符串提示。
- `#2` 将提示转换为模型可以处理的 token ID。
- `#3` 增加批量维度。
- `#4` 从模型逐次生成输出 token。
- `#5` 移除批量维度。
- `#6` 生成时立即打印 token。
- `#7` 将完整生成序列解码为文本。

在这段代码中，我们首先将一个简单的数学问题编码为模型可以处理的 token ID。然后模型以流式方式逐个生成 token。我们在 token 出现时立即打印，以便在生成过程中就能读取输出。同时，我们将生成的 token 收集到一个列表中，以便之后将它们解码为最终的答案字符串。这种同时流式化和收集 token 的模式非常方便，因为它让我们可以实时监控生成过程，同时仍能保存完整的答案文本（在`all_tokens`中）供后续处理。

代码清单 3.2 中代码生成的响应如下：

```
To find the value of \( a² + b² \) given that \( a + b = 3 \) 
and \( ab = \frac{13}{6} \), we can use the following algebraic identity:

\[
a² + b² = (a + b)² - 2ab
\]

**Step 1:** Substitute the given values into the equation.

\[
a² + b² = (3)² - 2 \left( \frac{13}{6} \right)
\]

[...]  #1

**Final Answer:**

\[
\boxed{\dfrac{14}{3}}
\]
```

- `#1` 为简洁起见有所删减

如您所见，尽管我们使用的是基础模型，但它提供了类似推理模型的解释。这可能是因为 Qwen3 团队在预训练期间包含了思维链数据，正如其技术报告所述。但即使该模型表现出一些类似推理模型的行为，添加额外的推理方法仍可进一步增强这些能力。（请注意，生成的响应可能因你在 CPU、CUDA 或 MPS 设备上执行代码而有所不同。）

如果你不熟悉数学中常用的 LaTeX 语法，前面的响应可能难以解读。在这种情况下，你可以使用 IPython 的`Latex`类来渲染它：

```
from IPython.display import Latex, display
display(Latex(all_tokens))
```

在代码笔记本中运行上述代码的结果如图 3.4 所示。

![](img/CH03_F04_Raschka2.png)

### 图 3.4 渲染后的响应，包含分步计算和最终的方框答案

图 3.4 所示的最终答案确实是该问题的正确答案。

## 3.3 实现一个包装器以便更轻松地生成文本

我们已经加载了预训练的 LLM 并设置了文本生成功能（如图 3.5 所示）。现在，为了方便后面的章节，我们将为文本生成函数创建一个包装器（见代码清单 3.3），这样我们只需传入模型、分词器、提示以及一些额外设置，而不必每次都重复分词和输入准备步骤。

![](img/CH03_F05_Raschka2.png)

### 图 3.5 在本节中，我们将验证器工作流程的第 1 步和第 2 步包装成一个可复用的文本生成辅助函数。该包装器负责提示处理和流式生成，并返回模型的原始 LaTeX 输出，而无需每次都重复分词和输入准备。

### 代码清单 3.3 流式文本生成的包装器

```
def generate_text_stream_concat(
    model, tokenizer, prompt, device, max_new_tokens,
    verbose=False,
):
    input_ids = torch.tensor(                    #1
        tokenizer.encode(prompt), device=device  
        ).unsqueeze(0)                           

    generated_ids = []
    for token in generate_text_basic_stream_cache(  #2
        model=model,                                
        token_ids=input_ids,                        
        max_new_tokens=max_new_tokens,              
        eos_token_id=tokenizer.eos_token_id,        
    ):                                              
        next_token_id = token.squeeze(0)
        generated_ids.append(next_token_id.item())


        if verbose: #3
            print(
                tokenizer.decode(next_token_id.tolist()),
                end="",
                flush=True
            )

    return tokenizer.decode(generated_ids)  #4
```

- `#1` 将提示文本编码为 token ID 并放到设备上。
- `#2` 使用带缓存的生成方式逐次流式输出 token。
- `#3` 可选：在生成时打印 token。
- `#4` 将所有生成的 ID 解码为最终的文本字符串。

这个包装函数处理了文本生成的完整流程：它对输入提示进行分词，从模型流式生成新 token，然后将结果解码为最终的字符串。

如前所述，可选的`verbose`标志允许我们实时查看生成的 token。该函数的用法如下：

```
generated_text = generate_text_stream_concat(
    model, tokenizer, prompt, device,
    max_new_tokens=2048,
    verbose=True  #1
)
```

`#1` 使用`False`将禁止逐 token 的实时打印。

这会打印出你在第 3.2 节中看到的完全相同的结果：

```
[...]  #1

**Final Answer:**

\[
\boxed{\dfrac{14}{3}}
\]
```

`#1` 为简洁起见已缩短

## 3.4 提取最终答案框

现在我们已经加载并准备好模型，可以进入有趣的部分：评估模型。

值得重申的是，本书采用从头开始（from-scratch）的方法，自然包含一些详细的、有时甚至乏味的步骤。这是有意为之。我们希望自己构建评估流水线，以便更好地理解其工作原理，而不是仅仅调用预定义的包装函数和构建模块。

在上一节中，你看到模型在答案框（answer box）中返回了最终答案（在原始文本中写作`r"""\boxed{\dfrac{14}{3}}"""`）。我们并没有强制要求这种格式；模型很可能是因为框式答案（boxed answers）是数学基准和训练数据中的常见约定才这样输出的。这种行为不一定证明模型对我们稍后要运行的评估任务过拟合，但它反映了预训练模型经常在网上遇到许多问题格式，并学会复现这些风格约定。一般来说，可以合理地假设：模型训练时互联网上可获得的任何信息都属于训练数据的一部分。

虽然这里没有必要，但当我们在 MATH-500 数据集上评估模型时，我们会添加一个特定的提示，指示模型以这种框式形式返回答案。（MATH-500 是一个经过整理的 500 道问题集，广泛用作推理模型的基准数据集。）这一步很直接，每当我们需要将模型响应转换为可以大规模自动评分的形式时，它都会再次出现。这一常见约定使评估在模型之间更具一致性，并简化了数据提取。

我们现在将编写代码来执行框式答案内容的提取。本节实现图 3.6 中的步骤 3。（下一节将实现步骤 4 的标准化方法。）

![](img/CH03_F06_Raschka2.png)

**图 3.6** 说明如何从大语言模型输出中提取框式结果

由于你的模型可能因硬件不同而产生与我略有差异的响应，以下代码使用硬编码的占位答案，而不是用模型生成响应（但我们会假装这个答案是由模型生成的）。在后面的章节中，我们将重新使用该模型，让它为 MATH-500 数据集中的任务生成答案：

```
model_answer = (
r"""... some explanation...
**Final Answer:**

\[                     #1
\boxed{\dfrac{14}{3}}   #1
\]                      #1
""")
```

`#1` 我们要提取的答案框

注意：我们使用的是原始字符串（`r"""..."""`而不是普通字符串`"""..."""`）。原始字符串使得处理反斜杠（`\`）字符更加容易，否则反斜杠会被视为转义序列，并要求每个反斜杠都写成两个。

接下来，让我们定义一个函数，从`model_answer`中提取答案框。

**清单 3.4** 提取答案框

```
def get_last_boxed(text):
    boxed_start_idx = text.rfind(r"\boxed")  #1
    if boxed_start_idx == -1:
        return None

    current_idx = boxed_start_idx + len(r"\boxed")  #2

    while current_idx < len(text) and text[current_idx].isspace():#3
        current_idx += 1


    if current_idx >= len(text) or text[current_idx] != "{":#4
        return None

    current_idx += 1
    brace_depth = 1
    content_start_idx = current_idx

    while current_idx < len(text) and brace_depth > 0:#5
        char = text[current_idx]
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        current_idx += 1

    if brace_depth != 0:  #6
        return None

    return text[content_start_idx:current_idx-1]  #7
```

- `#1` 查找`\boxed`的最后一次出现位置。
- `#2` 获取`\boxed`之后的位置。
- `#3` 跳过`\boxed`后的任何空白字符。
- `#4` 期望一个左花括号`{`。
- `#5` 解析嵌套的花括号。
- `#6` 处理不平衡的花括号。
- `#7` 提取最外层花括号内的内容。

清单 3.4 中的`get_last_boxed`辅助实用函数从模型输出中提取最后一个`\boxed{...}`表达式的内容。更具体地说，它扫描最终的`\boxed`，跳过空白，检查花括号，并处理任何嵌套，从而捕获预期的答案字符串。

虽然这看起来有点繁琐，但有了这个解析器，当我们在像 MATH-500 这样的数据集上运行评估时，它会带来回报，因为提取正确的最终答案是衡量模型推理能力的第一步。

现在，让我们在该模型答案上测试这个函数：

```
extracted_answer = get_last_boxed(model_answer)
print(extracted_answer)
```

此函数调用的输出是`"\dfrac{14}{3}"`，这正是我们想要提取的框式答案。

### 渲染数学公式

我们可以通过前面介绍的`Latex`类渲染数学公式。或者，对于没有伴随答案文本的独立数学公式，我们可以使用更简单的`Math`类：

```
from IPython.display import Math
display(Math(r"\dfrac{14}{3}"))
```

这将分数渲染为

![](img/26.jpg)

前面的`get_last_boxed`函数正确地提取了文本，但我们可以通过`extract_final_candidate`函数（如下面的清单所示）使答案提取更加稳健，该函数处理了最终答案框缺失或不完整的情况。

**清单 3.5** 提取最终答案候选

```
import re

RE_NUMBER = re.compile(  #1
    r"-?(?:\d+/\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)

def extract_final_candidate(text, fallback="number_then_full"):

    result = ""  #2

    if text:  #3
        boxed = get_last_boxed(text.strip())
        if boxed:
            result = boxed.strip().strip("$ ")

        elif fallback in ("number_then_full", "number_only"):#4
            m = RE_NUMBER.findall(text)
            if m:
                result = m[-1]  #5
            elif fallback == "number_then_full":

                result = text  #6
    return result
```

- `#1` 用于从文本中提取数值的正则表达式
- `#2` 如果什么都没有匹配到，则返回默认值
- `#3` 如果存在，优先选择最后一个框式表达式。
- `#4` 如果没有框式表达式，则尝试回退。
- `#5` 使用最后一个数字。
- `#6` 如果找不到数字，则返回完整文本。

如果找不到框式答案，`extract_final_candidate`函数会提供以下`fallback`设置：

- `"number_then_full"`（默认）：选择最后一个简单数字；否则选择整个文本。
- `"number_only"`：选择最后一个简单数字；否则返回空字符串`""`。
- `"none"`：仅提取框式内容；否则返回空字符串`""`。

对于回退设置，清单 3.5 中的代码通过 Python 的`re`库使用了*正则表达式*（简称 regex）。正则表达式是一种在文本中搜索模式的方法。在我们的例子中，该正则表达式模式被设计为识别数字，包括分数、小数和科学计数法。虽然正则表达式语法看起来令人生畏，但你无需担心它在这里如何工作。重要的是，这为我们提供了一个可靠的工具，在没有框式答案时从模型输出中提取最后一个数字候选。

让我们在模型答案上尝试一下：

```
print(extract_final_candidate(model_answer))
```

这会正确返回`"\dfrac{14}{3}"`。

现在，让我们尝试一些额外的示例。首先，这里有一个带框的候选答案：

```
print(extract_final_candidate(r"\boxed{ 14/3. }"))
```

这会正确返回`"14/3."`，去除多余空白但不去除标点。标点字符将由我们稍后实现的等价性检查正确处理。

接下来，让我们尝试一个不带框的候选，它应该会触发回退设置：

```
print(extract_final_candidate("abc < > 14/3 abc"))
```

得益于默认的回退设置，它会在答案中找到最后一个数字并正确返回`"14/3"`。
