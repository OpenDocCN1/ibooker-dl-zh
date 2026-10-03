
### 3.2.4 控制随机性

如前一章所述，`temperature`设置影响生成输出的随机性。较低的温度会产生更重复和确定性的响应，而较高的温度会产生更多创新的响应。从根本上说，没有正确的设置——一切都取决于用例。

对于企业来说，更有创意的输出是在对多样化输出感兴趣并创建用于内容生成、故事、诗歌、歌词、笑话等用例的文本时。这些通常需要创造力。然而，企业在需要更可靠和精确的答案的用例，如发票生成、提案、代码生成等时，需要更多。这些设置适用于每个 API 调用，因此在同一工作流程中结合不同的温度级别是可能的。

如前例所示，我们建议创意响应的温度设置为 0.8。相反，建议将设置调整为 0.2 以获得更可预测的响应。通过一个例子，让我们看看这些设置如何改变输出，并观察多次调用之间的变化。

当温度设置为 0.8 时，我们从三个连续调用中收到了以下响应。输出如预期的那样改变，提供了本章中看到的建议。重要的是要注意，我们不需要进行三个单独的 API 调用。我们可以在单个 API 调用中将`n`参数设置为 3 以生成多个响应。以下是我们的 API 调用示例：

```py
response = client.completions.create(
    model="gpt-35-turbo",
    prompt=prompt_startphrase,
    temperature=0.8,
    max_tokens=100,
    n=3,
    stop=None)
```

以下列表展示了三种响应的创意生成。

#### 列表 3.9 温度设置为 0.8 时的完成输出

```py
{
  "choices": [
    {
      "finish_reason": "content_filter",
      "index": 0,            #1
      "logprobs": null,
      "text": "",
      "content_filter_results"={...}
    },
    {
      "finish_reason": "stop",
      "index": 1,            #2
      "logprobs": null,
      "text": "\n\n1\. Pawsitively Professional Pet Styling\n 
                 ↪2\. Fur-Ever Friendly Pet Groomers \n 
                 ↪3\. Tailored TLC Pet Care",
      "content_filter_results"={...}
    },
    {
      "finish_reason": "stop",
      "index": 2,            #3
      "logprobs": null,
      "text": "\n\n1\. Pawsitively Professional Pet Salon \n 
                 ↪2\. Friendly Fur-ternity Pet Care \n 
                 ↪3\. Personalized Pup Pampering Place",
      "content_filter_results"={...}
    }
  ],
  "created": 1689123394,
  "id": "cmpl-7bIRe6Ponn8y1198flJFfagq64r2E",
  "model": "gpt35",
  "object": "text_completion",
  "usage": {
    "completion_tokens": 96,
    "prompt_tokens": 32,
    "total_tokens": 128
  }
}
```

#1 第一个响应：被内容过滤器阻止

#2 第三个响应

#3 具有非常不同生成文本的最终响应

让我们更改设置以使其更加确定，并再次运行。请注意，API 调用中唯一的更改是`temperature=0.2`。输出是可预测和确定的，三个回答之间生成的文本非常相似。

#### 列表 3.10 温度为 0.2 时的完成输出

```py
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,                #1
      "logprobs": null,
      "text": "\n\n1\. Pawsitively Professional Pet Salon\n 
                 ↪2\. Friendly Furr Salon\n 
                 ↪3\. Personalized Pet Pampering",
      "content_filter_results"={...}
    },
    {
      "finish_reason": "stop",
      "index": 1,                #2
      "logprobs": null,
      "text": "\n\n1\. Pawsitively Professional Pet Salon\n 
                 ↪2\. Friendly Fur-Ever Pet Salon\n 
                 ↪3\. Personalized Pet Pampering Salon",
      "content_filter_results"={...}
    },
    {
      "finish_reason": "stop",
      "index": 2,               #3
      "logprobs": null,
      "text": "\n\n1\. Pampered Paws Pet Salon\n 
                 ↪2\. Friendly Fur Salon\n
                 ↪3\. Professional Pet Pampering"
    }
  ],
  ...
}
```

#1 三个回答中的一个

#2 三个回答中的两个；生成的文本非常相似

#3 最终的回答，生成的文本非常相似

温度值可以高达 2，但并不建议设置得那么高，因为模型开始产生更多幻觉并创建无意义的文本。如果我们想要更多的创造力，通常希望它设置为 0.8，最多为 1.2。让我们看看当温度值变为 1.8 时的例子。在这个例子中，我们甚至没有完成第三次生成，因为我们达到了标记限制并停止了生成。

#### 列表 3.11 温度为 1.8 时的完成输出

```py
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,                #1
      "logprobs": null,
      "text": "\n\n1\. ComfortGroom Pet Furnishing \n2\. Pampered TreaBankant Carers \n3\. Toptech Sunny Haven Promotion.",
      "content_filter_results"={...}
    },
    {
      "finish_reason": "stop",
      "index": 1,               #2
      "logprobs": null,
      "text": "\n\n1: Naturalistov ClearlywowGroomingz 
                    ↪Pet Luxusia \n2: VipalMinderers Pet 
                    ↪Starencatines grooming \n3: Brisasia 
                    ↪Crownsnus Take Care Buddsroshesipalising",
      "content_filter_results"={...}
    },
    {
      "finish_reason": "length",
      "index": 2,
      "logprobs": null,
      "text": "\n\n1\. TrustowStar Pet Salon\n 
              ↪2\. Hartipad TailTagz Grooming & Styles\n 
              ↪3\. LittleLoft Millonista Cosmania DipSavez 
                 ↪Hubopolis ShineBright Princessly 
                 ↪Prosnoiffarianistics Kensoph Cowlosophy 
                 ↪Expressionala Navixfordti Mundulante Effority 
                 ↪DivineSponn BordloveDV EnityzBFA Prestageinato 
                 ↪SuperGold Cloutoilyna Critinarillies 
                 ↪Prochromomumphance Toud",
                 ↪"content_filter_results"={...}
    }
  ],
  ...
}
```

#1 三个回答中的一个，名称不是很清晰

#2 第二和第三个回答，名称不清晰

### 3.2.5 使用 top_p 控制随机性

用于管理随机性的`temperature`参数的替代方法是`top_p`参数。它对生成的影响与温度参数相同，但它使用了一种称为*核心采样*的不同技术。本质上，核心采样只允许概率等于或低于`top_p`值的标记被视为生成的一部分。

核心采样通过从具有最高累积概率的一小部分最可能出现的单词中挑选单词来创建文本。`top_p`值决定了这个小组的大小，基于单词出现在其中的总概率。小组的大小可以根据下一个单词的概率而变化。核心采样可以帮助避免重复，并生成比其他方法更多样化和清晰的文本。

例如，如果我们把`top_p`值设置为 0.9，那么只有组成 90%概率分布的标记会被采样用于文本生成。这使我们能够避免最后 10%，这部分通常非常随机且多样化，最终导致无意义的幻觉。

`top_p`的值较低时，模型在生成时选择较少的标记，因此更加一致且创造力较低。相反，较高的值使生成更加有创造力和多样化，因为它有更大的标记集可供操作。较大的值也使其更容易出现错误和随机性。`top_p`的确切值取决于用例；在大多数情况下，`top_p`的理想值介于 0.7 和 0.95 之间。我们应该更改温度属性或`top_p`，但不能同时更改两者。表 3.5 概述了两者之间的关系。

#### 表 3.5 温度和`top_p`之间的关系

| 温度 | top_p | 影响 |
| --- | --- | --- |
| 低  | 低  | 生成符合常见语言模式的可预测文本 |
| 低  | 高  | 生成的文本可预测，但偶尔会出现不太常见的单词或短语 |
| 高  | 低  | 生成的文本通常连贯，但具有创造性和意外的词汇使用 |
| 高  | 高  | 生成的文本高度多样且不可预测，具有各种词汇选择和想法；具有非常创造性和多样化的输出，但可能包含许多错误 |

让我们看看一些针对特定场景的高级 API 选项。

## 3.3 高级完成 API 选项

现在我们已经检查了完成 API 的基本结构并了解了它们的工作原理，我们需要考虑完成 API 的更高级方面。其中许多可能看起来并不复杂，但它们给系统架构增加了许多更多责任，从而复杂了整体实现。

### 3.3.1 流式完成

完成 API 允许流式响应，提供在信息准备好后立即访问信息的能力，而不是等待完整响应。对于企业来说，在某些情况下，实时内容生成和较低延迟是关键，流式传输在这种情况下可能很重要。此功能可以通过及时处理传入的响应来增强用户体验。

要从 API 的角度启用流式传输，将`stream`参数修改为`true`。默认情况下，此可选参数设置为`false`。

流式传输使用服务器发送事件（SSE），这需要在客户端实现。SSE 是一种标准协议，允许服务器在建立初始连接后继续向客户端传输数据。它是从服务器到客户端的长期单向连接。SSE 提供诸如低延迟、减少带宽消耗和简单的配置设置等优势。

列表 3.12 演示了如何调整我们的示例以利用流式传输。尽管 API 修改很简单，但描述和请求的多个生成（使用`n`属性）已调整。这使得我们可以人为地生成更多文本，从而更容易观察流式生成。

#### 列表 3.12 流式完成

```py
import os
import sys
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint=os.getenv("AOAI_ENDPOINT"),
    api_version="2024-05-01-preview",
    api_key=os.getenv("AOAI_KEY"))

prompt_startphrase = "Suggest three names and a tagline  
↪which is at least 3 sentences for a new pet salon business. 
↪The generated name ideas should evoke positive emotions and the 
↪followingkey features: Professional, friendly, Personalized Service."   #1

for response in client.completions.create(         #2
    model="gpt-35-turbo",
    prompt=prompt_startphrase,
    temperature=0.8,
    max_tokens=500,
    stream=True,                                   #3
    stop=None):
    for choice in response.choices:       #4
        sys.stdout.write(str(choice.text)+"\n")
        sys.stdout.flush()
```

#1 稍微调整了提示以添加描述

#2 我们需要在客户端处理流式响应。

#3 启用流式传输

#4 我们需要遍历数组并处理多个生成。

在管理流式调用时，我们必须特别注意`finish_reason`属性。随着消息的流式传输，每个消息都表现为标准完成，其中的文本代表新生成的标记。在这些情况下，`finish_reason`保持为 null。然而，最后一条消息不同；它的`finish_reason`可以是`stop`或`length`，具体取决于触发它的原因。

#### 列表 3.13 流式完成原因

```py
...
{
  "finish_reason": null,
  "index": 0,
  "logprobs": null,
  "text": " Pet"
}
{
  "finish_reason": null,
  "index": 0,
  "logprobs": null,
  "text": " Pam"
}
{
  "finish_reason": null,
  "index": 0,
  "logprobs": null,
  "text": "pering"
}
{
  "finish_reason": "stop",
  "index": 0,
  "logprobs": null,
  "text": ""
}
```

### 3.3.2 影响标记概率：logit_bias

`logit_bias`参数是我们影响输出完成的一种方式。在 API 中，此参数允许我们操纵模型在响应中生成的某些标记（可以是单词或短语）的概率。它被称为`logit_bias`，因为它直接影响了模型在生成过程中为每个潜在标记计算的 log odds 或 logits。偏差值被添加到这些 log-odds 中，在将它们转换为概率之前，改变模型可以从中选择的标记的最终分布。

此功能的重要性在于其引导模型输出的能力。比如说我们正在创建一个聊天机器人，并希望它避免某些单词或短语。我们可以使用`logit_bias`降低这些标记被模型选择的概率。相反，如果我们希望模型优先考虑某些单词或短语，我们可以使用`logit_bias`增加它们的可能性。此参数的范围是从-100 到 100，它作用于单词的标记。将标记设置为-100 实际上禁止它在生成中出现，而将其设置为 100 则使其成为独家。

要使用`logit_bias`，我们提供了一个字典，其中键是标记，值是需要应用于这些标记的偏差。要获取标记，我们使用`tiktoken`库。一旦你有了适当的标记，你可以分配一个正偏差使其更有可能出现，或者分配一个负偏差使其不太可能出现，如图 3.2 所示。这些块显示了不同标记在不同禁止或独家生成概率下的概率程度。对标记值的较小更改会增加或减少这些标记在生成输出中的概率。

![figure](img/CH03_F02_Bahree.png)

#### 图 3.2 `logit_bias`参数
