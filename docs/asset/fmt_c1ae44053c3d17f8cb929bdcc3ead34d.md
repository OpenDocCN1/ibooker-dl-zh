
| 最近最少使用 (LRU) | 跟踪缓存使用情况随时间变化，并移除最近最少访问的项目 | 当最近访问的项目更有可能再次被访问时 |
| --- | --- | --- |
| 最少使用 (LFU) | 跟踪缓存使用情况随时间变化，并移除最少访问的项目 | 当应首先移除较少使用的项目时 |
| 最近最常使用 (MRU) | 跟踪缓存使用情况随时间变化，并移除最近访问的项目 | 很少使用，当最近访问的项目不太可能再次被访问时 |
| 随机替换 (RR) | 从缓存中随机移除一个项目 | 简单快捷，在不影响性能的情况下使用 |

选择合适的淘汰策略将取决于你的用例和应用需求。通常，你可以从 `LRU` 策略开始，然后再切换到其他替代方案。

现在，你应该更有信心实现适用于文档检索或模型响应的语义缓存机制。接下来，让我们了解上下文或提示缓存，它根据输入优化对模型的查询。

### 上下文/提示缓存

*上下文缓存*，也称为 *提示缓存*，是一种适用于在小型请求中重复引用大量上下文场景的缓存机制。它旨在重用频繁使用的提示的预计算注意力状态，从而消除每次新请求时对整个输入上下文的冗余重新计算的需求。

当你的服务涉及以下情况时，你应该考虑使用上下文缓存：

+   具有广泛系统指令和长对话的聊天机器人
+   对长视频文件进行重复分析
+   对大型文档集进行重复查询
+   经常进行代码仓库分析或错误修复
+   文档摘要、与书籍、论文、文档、播客文稿和其他长篇内容进行交流
+   在提示中提供大量示例（即，上下文学习）

这种类型的缓存可以帮助你通过缓存大型上下文标记来显著减少标记使用成本。据 Anthropic 称，提示缓存可以将成本降低高达 90%，并将延迟降低高达 85%。

提出这种技术的 [提示缓存论文](https://oreil.ly/augpd) 的作者还声称：

> 我们发现，提示缓存显著降低了首次标记的时间延迟，尤其是在基于文档的问答和推荐等较长的提示中。改进范围从基于 GPU 推理的 8 倍到基于 CPU 推理的 60 倍，同时保持输出准确性，且无需修改模型参数。

图 10-3 可视化了上下文缓存系统架构。

![bgai 1003](img/bgai_1003.png)

###### 图 10-3\. 上下文缓存的系统架构

在撰写本文时，OpenAI 自动为所有 API 请求实现提示缓存，无需进行任何代码更改或额外费用。示例 10-12 展示了在使用 Anthropic API 时如何使用提示缓存。

##### 示例 10-12\. 使用 Anthropic API 进行上下文/提示缓存

```py
from anthropic import Anthropic

client = Anthropic()

response = client.messages.create(
    model="claude-3-7-sonnet-20250219", ![1](img/1.png)
    max_tokens=1024,
    system=
        {
            "type": "text",
            "text": "You are an AI assistant",
        },
        {
            "type": "text",
            "text": "<the entire content of a large document>",
            "cache_control": {"type": "ephemeral"}, ![2
        },
    ],
    messages=[{"role": "user", "content": "Summarize the documents in ..."}],
)
print(response)
```

![1](img/1.png)(#co_optimizing_ai_services_CO8-1)

提示缓存仅适用于包括 Claude 3.5 Sonnet、Claude 3 Haiku 和 Claude 3 Opus 在内的少数模型。

![2](img/2.png)(#co_optimizing_ai_services_CO8-2)

使用`cache_control`参数在多个 API 调用之间重用大文档内容，而无需每次都处理它。

在幕后，Anthropic 客户端将`anthropic-beta: prompt-caching-2024-07-31`添加到请求头中。

在撰写本文时，`ephemeral`是唯一支持的缓存类型，对应 5 分钟的缓存生命周期。

###### 注意

一旦您采用上下文缓存，您就在请求中引入了状态性，通过在它们之间保留标记。这意味着您在一个请求中提交的数据将影响后续请求，因为模型提供者服务器可以使用缓存的上下文来保持交互之间的连续性。

使用 Gemini API 的上下文缓存功能，您只需向模型提供一次内容，缓存输入标记，并在未来的请求中引用这些缓存的标记。

使用这些缓存的标记可以为您节省大量费用，如果您避免在高量级上重复传递相同的标记语料库。缓存成本将取决于输入标记的大小和期望的 TTL 存储时长。

###### 小贴士

当您缓存一组标记时，您可以指定 TTL 持续时间，即标记在自动删除之前应存在的时长。默认情况下，TTL 通常设置为 1 小时。

你可以在示例 10-13 中看到如何使用缓存系统指令。你还需要 Gemini API Python SDK：

```py
$ pip install google-generativeai
```

##### 示例 10-13\. 使用 Google Gemini API 进行上下文缓存

```py
import datetime
import google.generativeai as genai
from google.generativeai import caching

genai.configure(api_key="your_gemini_api_key")

corpus = genai.upload_file(path="corpus.txt")
cache = caching.CachedContent.create(
    model='models/gemini-1.5-flash-001',
    display_name='fastapi', ![1](img/1.png)
    system_instruction=(
        "You are an expert AI engineer, and your job is to answer "
        "the user's query based on the files you have access to."
    ),
    contents=[corpus], ![2](img/2.png)
    ttl=datetime.timedelta(minutes=5),
)

model = genai.GenerativeModel.from_cached_content(cached_content=cache)
response = model.generate_content(
    [
        (
            "Introduce different characters in the movie by describing "
            "their personality, looks, and names. Also list the timestamps "
            "they were introduced for the first time."
        )
    ]
)
```

![1](img/1.png)(#co_optimizing_ai_services_CO9-1)

提供一个显示名称作为缓存键或标识符。

![2](img/2.png)(#co_optimizing_ai_services_CO9-2)

将语料库传递给上下文缓存系统。上下文缓存的最小大小为 32,768 个标记。

如果您运行示例 10-13 并打印`response.usage_metadata`，您应该收到以下输出：

```py
>> print(response.usage_metadata)

prompt_token_count: 696219
cached_content_token_count: 696190
candidates_token_count: 214
total_token_count: 696433
```

注意比较`prompt_token_count`和`cached_content_token_count`时，现在有多少`prompt_token_count`被缓存了。`candidates_token_count`指的是来自模型的输出或响应标记的数量，这不受缓存系统的影响。

###### 警告

Gemini 模型在缓存标记和常规输入标记之间没有区别。缓存的内容将被添加到提示的前面。这就是为什么使用缓存时提示标记计数不会减少。

使用上下文缓存，你不会看到响应时间的显著减少，但可以显著降低运营成本，因为你避免了重新发送大量的系统提示和上下文标记。因此，这种缓存策略最适合当你有一个大上下文要处理时——例如，当你批量处理包含大量指令和示例的文件时。

###### 注意

使用相同的上下文缓存和提示不能保证模型响应的一致性，因为 LLM 的响应是非确定性的。上下文缓存不会缓存任何输出。

上下文缓存仍然是研究的一个活跃领域。如果你想避免任何供应商锁定，这个领域已经有一些进展，例如开源工具[*MemServe*](https://oreil.ly/PXm6B)，它通过弹性内存池实现了上下文缓存。

除了缓存之外，你还可以通过使用如*模型量化*等技术来审查你的选项，以减少模型大小，从而加快响应时间。

## 模型量化

如果你打算自己提供 LLM 等模型的服务，你应该考虑如果可能的话对模型进行*量化*（即压缩/缩小）。通常，开源模型存储库也会提供量化版本，你可以直接下载和使用，而无需自己进行量化过程。

*模型量化*是对模型权重和激活的调整过程，其中通过使用原始参数分布上的缩放因子进行微调操作，将高精度模型参数统计投影到低精度值。然后你可以使用低精度执行所有关键推理操作，之后可以将输出转换为高精度，以保持质量同时提高性能。

降低精度也会降低内存存储需求，理论上降低能耗，并通过整数运算加速矩阵乘法等操作。这也使得模型能够在仅支持整数数据类型的嵌入式设备上运行。

图 10-4 展示了完整的量化过程。

![bgai 1004](img/bgai_1004.png)

###### 图 10-4\. 量化过程

你可以在 GPU 内存消耗上节省超过几个 GB，因为低精度数据类型，如 8 位整数，每个参数所需的 RAM 比 32 位浮点数据类型少得多。

### 精度与质量权衡

图 10-5 比较了未量化和量化的模型。

![bgai 1005](img/bgai_1005.png)

###### 图 10-5\. 量化

由于每个高精度的 32 位浮点参数消耗 4 字节 GPU 内存，一个 1B 参数的模型仅用于推理就需要 4 GB 的内存。如果您计划重新训练或微调相同的模型，您至少需要 24 GB 的 GPU VRAM。这是因为每个参数还需要存储诸如梯度、训练优化器状态、激活和临时内存空间等信息，总共消耗额外的 24 字节。这估计比仅加载模型权重所需的内存需求高 6 倍。因此，同样的 1B 模型将需要 24 GB 的 GPU，而即使是最好的、最昂贵的消费级显卡，如 NVIDIA RTX 4090，也可能难以满足这一需求。

除了标准的 32 位浮点数外，您可以选择以下任何一种格式：

+   *16 位浮点数 (FP16)* 可以将内存使用量减半，而对模型输出质量的影响不大。
+   *8 位整数 (INT8)* 在内存上提供巨大的节省，但质量损失很大。
+   *16 位脑浮点数 (BFLOAT16)* 与 FP32 具有相似的范围，平衡了内存和质量之间的权衡。
+   *4 位整数 (INT4)* 在内存效率和计算精度之间提供平衡，使其适合低功耗设备。
+   *1 位整数 (INT1)* 使用最低精度数据类型，以最大程度地减少模型大小。目前正在进行创建高质量 [1 位 LLMs](https://oreil.ly/QH9nH) 的研究。

为了比较，表 10-2 展示了量化 Llama 家族模型时模型大小的减少。

表 10-2\. 量化对 Llama 模型大小的影响^(a)

| 模型 | 原始 | FP16 | 8 位 | 6 位 | 4 位 | 2 位 |
| --- | --- | --- | --- | --- | --- | --- |
| Llama 2 70B | 140 GB | 128.5 GB | 73.23 GB | 52.70 GB | 36.20 GB | 28.59 GB |
| Llama 3 8B | 16.07 GB | 14.97 GB | 7.96 GB | 4.34 GB | 4.34 GB | 2.96 GB |

| ^(a) 来源：[Llama.cpp GitHub 仓库](https://oreil.ly/9iYtL) 和 [Tom Jobbins 的 Hugging Face Llama 2 70B 模型卡片](https://oreil.ly/BMDtR) |

###### 小贴士

除了用于适应模型的 GPU VRAM 外，您还需要额外的 5 到 8 GB GPU VRAM 来处理模型加载时的开销。

根据当前的研究状况，仅使用整数数据类型 INT4 和 INT1 来保持精度是一个挑战，而使用 INT32 或 FP16 的性能提升并不显著。因此，最流行的低精度数据类型是用于推理的 INT8。

根据研究 [`oreil.ly/C7Lz3`](https://oreil.ly/C7Lz3)，使用仅整数算术进行推理将比使用浮点数更高效。然而，将浮点数量化为整数可能很复杂。例如，INT8 只能表示 256 个值，而 float32 可以表示广泛的值。

### 浮点数

要了解为什么将 32 位浮点数投影到其他格式可以节省如此多的 GPU 内存，让我们看看它是如何分解的。

一个 32 位浮点数由以下类型的位组成：

+   *符号位* 表示一个数是正数还是负数
+   *指数位* 控制数字的量级
+   *尾数位* 保留实际数字，以确定其精度（也称为 *分数位*）

您可以在 图 10-6 中看到上述浮点数的位可视化。

![bgai 1006](img/bgai_1006.png)

###### 图 10-6\. 32 位浮点数、16 位浮点数和 bfloat16 数字中的位

当您将 FP32 数字投影到其他格式时，实际上是在将其挤压到更小的范围内，丢失了大部分尾数位，并调整了指数位，但精度损失不大。您可以通过参考 图 10-7 来看到这种现象的实际操作。

![bgai 1007](img/bgai_1007.png)

###### 图 10-7\. 将浮点数量化为整数的量化过程

事实上，[关于预训练 LLM 量化策略的研究](https://oreil.ly/Swfz7) 表明，具有 4 位量化的 LLM 可以保持与未量化版本相似的性能。然而，虽然量化可以节省内存，但它也可能降低 LLM 的推理速度。

### 如何量化预训练的 LLM

量化是通过权重调整来压缩大型模型的过程。有一种称为 [*GPTQ*](https://oreil.ly/rHYKZ) 的技术可以在大约 4 个 GPU 小时内量化具有 1750 亿参数的 LLM，将权重位宽降低到每权重 3 或 4 位，与未压缩模型相比，精度下降可以忽略不计。

Hugging Face `transformers` 和 `optimum` 库的作者与 `auto-gptq` 库的开发者紧密合作，为开源 LLM 提供了一个简单的 API，用于应用 GPTQ 量化。Optimum 是一个库，它提供了使用不同工具进行量化的 API。

使用 GPTQ 量化，您可以将您喜欢的语言模型量化为 8、4、3 或甚至 2 位，而性能下降不大，同时保持大多数 GPT 硬件支持的更快推理速度。您可以通过 示例 10-14 了解如何在您的 GPU 上量化预训练模型。

运行 示例 10-14 所需安装的依赖项包括以下内容：

```py
$ pip install auto-gptq optimum transformers accelerate
```

##### 示例 10-14\. 使用 Hugging Face 和 AutoGPTQ 库进行 GPTQ 模型量化

```py
import torch
from optimum.gptq import GPTQQuantizer
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "facebook/opt-125m" ![1](img/1.png)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16
)

quantizer = GPTQQuantizer(
    bits=4,
    dataset="c4", ![2](img/2.png)
    block_name_to_quantize="model.decoder.layers", ![3](img/3.png)
    model_seqlen=2048, ![4](img/4.png)
)
quantized_model = quantizer.quantize_model(model, tokenizer)
```

![1](img/#co_optimizing_ai_services_CO10-1)

在量化之前，加载 `facebook/opt-125m` 预训练模型的 `float16` 版本。

![2](img/#co_optimizing_ai_services_CO10-2)
