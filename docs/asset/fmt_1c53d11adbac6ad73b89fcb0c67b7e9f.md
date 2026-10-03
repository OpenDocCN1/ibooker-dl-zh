
# Qwen3 基础模型与推理方法

Qwen3 既提供基础模型（我们推理模型开发的重点），也提供官方的推理变体，我们可以将其作为评估的参考。

> **注意：** “Qwen3”的规范拼写不包含空格，而例如“Llama 3”则包含空格。

本着“从零开始”构建的精神，本书使用了我在纯 PyTorch 中编写、且完全不依赖外部 LLM 库的 Qwen3 自定义重实现。该重实现的重点是代码的可读性和易调整性，以便你日后可能为自身实验对其进行修改。尽管是从零构建，该实现仍与原始预训练的 Qwen3 模型权重完全兼容。

本书并不深入探讨 Qwen3 的代码实现。仅该主题本身就足以独立成书，类似我的 *《构建大语言模型（从零开始）》* 一书。相反，本书着重于在基础模型（此处即 Qwen3）之上实现推理方法。如果你对 Qwen3 及模型代码的更详细信息感兴趣，可参阅附录 C。

**注意：** 此重实现的 Qwen3 LLM 完全在本地运行，与任何其他以 PyTorch 实现的神经网络一样。不涉及服务端组件或外部 API 调用。所有模型使用均在你自己的机器上完成，你的数据留在你的设备中。如果你关心隐私，我们使用的设置可以确保对 LLM 的输入和输出拥有完全控制。

在加载模型之前，我们可以指定将使用的设备，即 CPU 或 GPU。下面的代码会自动选择最可用的设备。

### 清单 2.1 自动获取设备

```python
def get_device(enable_tensor_cores=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA CUDA GPU")

        if enable_tensor_cores:
            major, minor = map(int, torch.__version__.split(".")[:2])
            if (major, minor) >= (2, 9):
                torch.backends.cuda.matmul.fp32_precision = "tf32"
                torch.backends.cudnn.conv.fp32_precision = "tf32"
            else:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")

    elif torch.xpu.is_available():
        device = torch.device("xpu")
        print("Using Intel GPU")

    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device
```

请注意，如果你拥有较新的 NVIDIA GPU（基于 Ampere 架构或更新），当 `enable_tensor_cores=True` 时，`get_device()` 函数会自动启用 Tensor Cores 以加快矩阵乘法。这可能会略微改变浮点舍入，但不会明显影响本书中的结果，而在新显卡上的速度优势是值得的。在非 NVIDIA 设备上，这些设置会被忽略。

使用清单 2.1 中的代码，我们可以按如下方式获取设备：

```python
device = get_device()
```

虽然 GPU 通常能带来显著的速度和性能提升，但为了兼容性和调试，在 CPU 上运行本章剩余代码也可能会有所帮助。你可以通过显式设置设备来临时覆盖自动选择：

```python
device = torch.device("cpu")
```

当你完成本章并验证代码在 CPU 上能正常运行后，可以移除或注释掉手动覆盖并重新运行代码。如果你的系统有 GPU，那么此后应能看到更好的性能。

**注意：** 本章剩余部分的代码在搭载 Apple M4 CPU 的 Mac Mini 上执行。本章末尾包含与 Apple Silicon M4 GPU 和 NVIDIA H100 GPU 的性能对比。

在加载模型并将其放到所选 `device` 上之前，我们首先需要下载 Qwen3 0.6B 的权重。需要这些文件才能正确初始化预训练模型。在上一节中，我们调用了带有 `tokenizer_only=True` 的 `download_qwen3_small()` 函数，仅下载了分词器文件。这里，我们使用相同的函数，但设置 `tokenizer_only=False`，以便也下载模型权重：

```python
download_qwen3_small(kind="base", tokenizer_only=False, out_dir="qwen3")
```

输出如下：

```
qwen3-0.6B-base.pth: 100% (1433 MiB / 1433 MiB)
✓ qwen3/tokenizer-base.json already up-to-date
```

（分词器旁边有复选标记，因为我们在上一节已下载了它。）

既然我们已经下载了模型权重，就可以实例化一个 `Qwen3Model` 类，并通过 PyTorch 的 `load_state_dict` 方法加载预训练权重：

```python
from reasoning_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B

model_path = Path("qwen3") / "qwen3-0.6B-base.pth"
model = Qwen3Model(QWEN_CONFIG_06_B)  #1
model.load_state_dict(torch.load(model_path))  #2
model.to(device)  #3
```

- `#1` 实例化一个 Qwen3 模型，使用随机权重作为占位符。
- `#2` 将预训练权重加载到模型中。
- `#3` 将模型转移到指定设备（例如 “cuda”）。

请注意，如果设备设置为 `"cpu"`，则 `model.to(device)` 操作会被跳过，因为模型默认已位于 CPU 内存中。

执行上述代码后，你应该看到以下输出。（如果你不是在 Jupyter 笔记本等交互式环境中运行代码，则需要运行 `print(model)` 才能看到此输出。）

```
Qwen3Model(
  (tok_emb): Embedding(151936, 1024)
  (trf_blocks): ModuleList(
    (0-27): 28 x TransformerBlock(
      (att): GroupedQueryAttention(
        (W_query): Linear(in_features=1024, out_features=2048, bias=False)
        (W_key): Linear(in_features=1024, out_features=1024, bias=False)
        (W_value): Linear(in_features=1024, out_features=1024, bias=False)
        (out_proj): Linear(in_features=2048, out_features=1024, bias=False)
        (q_norm): RMSNorm()
        (k_norm): RMSNorm()
      )
      (ff): FeedForward(
        (fc1): Linear(in_features=1024, out_features=3072, bias=False)
        (fc2): Linear(in_features=1024, out_features=3072, bias=False)
        (fc3): Linear(in_features=3072, out_features=1024, bias=False)
      )
      (norm1): RMSNorm()
      (norm2): RMSNorm()
    )
  )
  (final_norm): RMSNorm()
  (out_head): Linear(in_features=1024, out_features=153936, bias=False)
)
```

该输出是 Qwen3 0.6B 基础模型架构的概要，由 PyTorch 打印。它突出了模型的核心组件：嵌入层、28 个变压器块的堆叠，以及最终的线性投影头。每个变压器块包括一个分组查询注意力机制和一个多层前馈网络，并贯穿归一化层。

这些组件也在图 2.8 中进行了说明，适合熟悉 LLM 架构的读者，但本书并不要求深入理解该架构。由于我们不是修改基础模型本身，而是在其之上构建推理方法，因此你可以安全地暂时将架构视为黑盒。如果你感兴趣，可以在附录 C 中找到有关这些组件（例如 RMSNorm）的更多信息。

![](img/CH02_F08_Raschka2.png)

### 图 2.8 前面实例化的 Qwen3 0.6B 模型架构概览。输入文本被分词化并经过嵌入层，随后是 28 个重复的变压器块。每个块包含分组查询注意力、前馈层和 RMS 归一化。模型以最终归一化和线性输出层结束。箭头显示数据流经模型的方式。

现在我们已经加载了一个预训练模型，它应当能够生成连贯的文本。在下一节中，我们将编写一个文本生成函数，该函数将分词化数据输入模型，并以人类可读的形式返回输出。

## 2.6 理解顺序式 LLM 文本生成过程

既然我们已经加载了预训练的 LLM，我们的目标是编写一个函数，利用该 LLM 来生成文本。这个函数将为本书后续实现的推理改进方法奠定基础，如图 2.9 所示。

![](img/CH02_F09_Raschka2.png)

### 图 2.9 本书中开发推理模型的四个关键阶段概述。本节解释 LLM 中文本生成背后的主要概念，这将使我们能够实现一个使用预训练 LLM 的文本生成函数。

你可能已经知道，LLM 中的文本生成是一个顺序过程，LLM 一次生成一个词（或 token）。这通常也被称为*自回归*文本生成。图 2.10 展示了一个大致的流程，每一步生成一个输出 token（位于顶行），并同时将其作为输入提示的一部分反馈给模型。

### 聊天机器人界面

聊天界面只是将底层的下一个 token 预测过程（如图 2.10 所示）包装在一个对话循环中。聊天机器人模型仍然一次预测一个 token，但系统会将整个对话历史反馈给模型，这样每一步的新回复都会在轮次之间具有上下文感知和连贯性。本书专注于单轮对话，即当前答案独立于之前的答案。如果你有兴趣，可以在附录 G 中找到带有回答历史的聊天界面实现。

![](img/CH02_F10_Raschka2.png)

#### 图 2.10 LLM 中的顺序（自回归）文本生成。在每次迭代中，模型根据输入和先前生成的 token 生成下一个 token，这些 token 会累积地反馈给模型，以产生连贯的输出。

如果我们放大图 2.10 中的某一次迭代，可以看到 LLM 在一次前向传播中处理完整的输入序列，并为每个输入位置生成一个预测。这意味着如果我们有六个输入 token，模型会返回六个对应的预测，如图 2.11 所示。该图使逐位置的结构一目了然，但模型输出通常不仅仅是输入提示向后移动一个位置。相反，每个位置都会产生一个关于可能的下一个 token 的分布。对于文本生成，我们只使用最后一个位置的预测，因为它能告诉我们接下来该添加哪个新 token。

在实现一个文本生成函数（该函数在自回归文本生成过程的每次迭代中使用图 2.11 所示的概念）之前，让我们看一个代码示例来进一步说明图 2.11。我们将重用第 2.4 节中的“解释大型语言模型。”提示：

![](img/CH02_F11_Raschka2.png)

#### 图 2.11 LLM 处理完整的输入序列，并为每个输入位置生成一个下一个 token 预测。这些预测在概念上向前移动了一步，因为每个位置都预测其后的 token。该插图按位置对齐以帮助理解，但输出并不是字面上的输入提示向后移动一个位置。在生成过程中，我们只保留最后一个预测，并用它一次一个 token 地继续扩展输入提示。

```python
prompt = "Explain large language models."
input_token_ids_list = tokenizer.encode(prompt)
print(f"Number of input tokens: {len(input_token_ids_list)}")

input_tensor = torch.tensor(input_token_ids_list)  #1
input_tensor_fmt = input_tensor.unsqueeze(0)  #2
input_tensor_fmt = input_tensor_fmt.to(device)

with torch.inference_mode():
    output_tensor = model(input_tensor_fmt)  #3
output_tensor_fmt = output_tensor.squeeze(0)  #4
print(f"Formatted Output tensor shape: {output_tensor_fmt.shape}")
```

- `#1` 将 Python 列表转换为 PyTorch 张量。
- `#2` 增加一个额外的维度。
- `#3` 生成输出。
- `#4` 移除多余的维度。

上述代码产生以下输出：

```
Number of input tokens: 6
Formatted Output tensor shape: torch.Size([6, 151936])
```

如我们所见，我们将六个输入 token 输入模型，模型返回一个 6×151,936 维的矩阵。这个矩阵中的 6 对应六个输入 token。第二个维度 151,936 对应模型支持的词汇表大小。例如，六个 token 中的每一个都由一个包含 151,936 个值的向量表示。我们可以将这些向量中的值视为词汇表中每个可能单词的分数，其中最高分数对应的单词或子词（在包含 151,936 个条目的词汇表中）最有可能被选为生成的 token。

### 压缩和扩展张量

张量是 *n* 维的广义矩阵。许多 PyTorch 函数和模型组件期望特定维度的张量，因此能够添加或移除维度对于使数据与这些操作兼容非常重要。

PyTorch 中的 `.squeeze()` 和 `.unsqueeze()` 操作通过移除或添加大小为 1 的维度来改变张量的形状。这对于重塑张量以匹配模型的期望通常很有用。例如，模型可能期望输入张量具有两个维度（例如，行和列），以便处理批量输入（参见附录 E）。但如果输入只是一个行向量，我们可以使用 `.unsqueeze(0)` 添加一个额外的维度使其兼容：

```python
example = torch.tensor([1, 2, 3])
print(example)
print(example.unsqueeze(0))
```

这会返回以下结果：

```
tensor([1, 2, 3])
tensor([[1, 2, 3]])
```

在这里，`.unsqueeze(0)` 在位置 0 添加了一个新维度，将一维张量转换为形状为 `(1, 3)` 的二维张量。相反，`.squeeze(0)` 会从位置 0 移除大小为 1 的维度：

```python
example = torch.tensor([[1, 2, 3]])
print(example)
print(example.squeeze(0))
```

这会返回以下结果：

```
tensor([[1, 2, 3]])
tensor([1, 2, 3])
```

当你想要移除不需要的多余维度时，这非常有用。

为了获得下一个生成的单词，我们提取这个 6×151,936 维矩阵的最后一行，找到该行中最大值分数对应的 token ID，然后通过分词器将该 token ID 转换回文本，如图 2.12 所示。

![](img/CH02_F12_Raschka2.png)

#### 图 2.12 在单次文本生成迭代中，LLM 输出的原始分数如何被转换为 token ID 及其对应的文本表示的详细说明。

让我们看看如何在代码中将 LLM 输出矩阵转换为生成的文本 token（如图 2.12 所示）。

请注意，对于文本生成，我们以推理模式而非训练模式运行模型，因此 PyTorch 不需要跟踪梯度。如图 2.11 所示，较早的位置对应的是针对提示中已有 token 的预测。只有最后一个位置预测了我们想要追加的下一个未见过的 token，我们可以通过 `[-1]` 索引获得：

```python
last_token = output_tensor_fmt[-1]
print(last_token)
```

在上一步中使用 `torch.inference_mode()` 可以避免存储在生成期间不需要的梯度信息。这减少了内存使用，并且通常会提高速度。

这会打印出与最后一个 token 对应的 151,936 个值：

```
tensor([ 7.3750,  2.0312,  8.0000,  ..., -2.5469, -2.5469, -2.5469],
       dtype=torch.bfloat16)
```

`dtype=torch.bfloat16` 后缀表示这些分数以 `bfloat16` 存储，这是一种降低精度的格式，常用于降低内存使用并提高 LLM 推理效率。根据你的硬件和设置，你可能会看到不同的 `dtype`。
