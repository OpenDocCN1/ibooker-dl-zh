
```py
from torch.ao.quantization import get_default_qat_qconfig

model = nn.Sequential(nn.Linear(10, 100), nn.ReLU(), nn.Linear(100, 1))
model.qconfig = get_default_qat_qconfig(engine)
torch.ao.quantization.prepare_qat(model, inplace=True)
train(model, optimizer, [...])  # train the model normally
torch.ao.quantization.convert(model.eval(), inplace=True)
```

导入之后，我们创建我们的模型，将其`qconfig`属性设置为所选量化引擎的默认 QAT 配置对象，然后我们调用`prepare_qat()`函数向模型添加伪量化操作。这一步还会添加观察者以确定激活值的正常范围。接下来，我们可以正常训练模型。最后，我们将模型切换到评估模式，并调用`convert()`函数来真正量化它。

### 小贴士

QAT 不需要在整个训练过程中使用：你可以使用预训练模型，仅使用 QAT 对其进行几轮微调，通常使用较低的学习率以避免损坏预训练权重。

我们已经看到了如何使用 PyTorch 的`torch.ao`包实现 PTQ 和 QAT。然而，它主要是为 CPU 设计的。如果你想在内存不太够的 GPU 上运行 LLM 怎么办？一个选项是使用具有日益增长的 GPU 支持的 TorchAO 库。另一个选项是使用 bitsandbytes 库：让我们现在讨论它。

## 使用 bitsandbytes 库对 LLMs 进行量化

由 Tim Dettmers 创建的 bitsandbytes 库（bnb）旨在使在有限 VRAM 的 GPU 上训练和运行大型模型变得更加容易。为此，它提供了以下功能：

+   量化工具，包括 4 位量化、块量化等
+   内存高效的流行优化器版本，如 Adam 或 AdamW，它们在 8 位张量上操作
+   为 8 位或 4 位量化模型专门编写的自定义 CUDA 内核，以实现最大速度

### 警告

bitsandbytes 库是为 Nvidia GPU 设计的。它还对 CPU 和 AMD GPU 提供了一些有限的支持。

例如，让我们看看如何实现将训练后的静态量化降低到 4 位。如果你使用 Colab，你必须首先使用`%pip install bitsandbytes`安装 bitsandbytes 库，然后运行以下代码：

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",
                                             quantization_config=bnb_config)
```

此代码首先从 Transformers 库（在第十四章中介绍）导入必要的类，然后创建一个`BitsAndBytesConfig`对象，我将在稍后解释。最后，它下载了一个预训练模型（在这种情况下是一个名为 TinyLlama 的 11 亿参数版本的 Llama，用于聊天微调），并指定了所需的量化配置。

在底层，Transformers 库使用 bitsandbytes 库将模型权重量化到 4 位，就像它们被加载到 GPU 上一样：不需要额外的步骤。你现在可以使用这个模型正常生成文本（见第十五章）。在推理过程中，每当需要某些权重时，它们会即时反量化到由`bnb_4bit_compute_dtype`参数指定的类型（在这种情况下为`bfloat16`），并且计算以更高的精度执行。一旦不再需要反量化后的权重，它们就会被丢弃，因此内存使用保持较低。

在这个例子中，`BitsAndBytesConfig`对象指定使用`bfloat16`进行计算的*4 位正常浮点数*（NF4）量化。NF4 是一种非线性 4 位方案，其中 16 个可能的整数值中的每一个代表-1 和+1 之间的特定浮点值。与线性量化中均匀分布不同，这些值对应于以零为中心的正态分布的分位数：这意味着它们在零附近更接近。这提高了准确性，因为模型权重往往遵循以零为中心的正态分布，因此在零附近有更多的精度是有帮助的。

NF4 作为[QLoRA](https://homl.info/qlora)⁠^(7)的一部分被引入，这是一种量化冻结预训练模型的方法，然后使用 LoRA 适配器（见第十七章）进行微调，以及激活检查点（见第十二章）。这种方法大大减少了 VRAM 使用量和计算量：作者成功使用单个具有 48GB RAM 的 GPU 微调了一个 65 亿参数的模型，只略有精度下降。尽管激活检查点总体上减少了 VRAM 使用量，但它会导致处理长序列批次时的内存峰值。为了处理这种峰值，QLoRA 作者还引入了*分页优化器*，它利用 Nvidia 统一内存：CUDA 驱动程序在需要时自动将数据页从 GPU VRAM 移动到 CPU RAM。最后，作者还使用了*双重量化*，这意味着量化参数本身也被量化，以节省更多 VRAM。

关于 Hugging Face 生态系统中 4 位量化更详细的介绍，请查看 QLoRA 作者和其他贡献者写的这篇[优秀文章](https://huggingface.co/blog/4bit-transformers-bitsandbytes)。

# 使用预量化模型

许多流行的预训练模型已经量化并发布在网上，特别是在 Hugging Face Hub 上。例如，Tom Jobbins，更广为人知的是他的 Hugging Face 用户名 TheBloke，已经发布了数千个可用的量化模型，见[*https://huggingface.co/TheBloke*](https://huggingface.co/TheBloke)。许多这些模型都是使用以下现代方法之一进行量化的：

*生成式预训练量化* (GPTQ)

[GPTQ](https://homl.info/gptq)⁠^(8)是一种后训练量化方法，通常降低到 4 位，将量化视为一个优化问题。GPTQ 逐层进行，一次一层，优化 4 位权重以最小化层原始输出（即使用全精度权重）和近似输出（即使用 4 位权重）之间的均方误差。一旦找到最优的 4 位权重，近似输出就传递到下一层，整个过程一直重复到输出层。在推理过程中，每当需要时，权重就会被反量化。GPTQ 只量化权重，不量化激活：这被称为*仅权重量化*，这对于推理来说非常好，但不适用于训练。您可以使用[Hugging Face Optimum 库](https://huggingface.co/docs/optimum)或[GPTQModel 库](https://github.com/ModelCloud/GPTQModel)使用 GPTQ 量化您的模型。

*激活感知权重量化* (AWQ)

[AWQ](https://homl.info/awq)⁠^(9)旨在提高块状权重仅量化（通常是 4 位量化）的准确性。其思路是保留最重要的权重的精度。为了识别这些所谓的*显著权重*，算法将校准数据集通过模型运行，并找到每个量化组的最大激活（例如，最大的 0.1%到 1%的激活），相应的权重被认为是显著的。作者观察到，使用 float16 存储显著权重可以显著降低模型的总困惑度（一个常见的指标，等于交叉熵的指数）。然而，混合 4 位和 16 位权重对硬件不友好，因此 AWQ 使用另一种方法来保留显著权重的精度：他们简单地通过某个因子放大它们，并在模型中添加一个操作来缩小相应的激活（但这个操作通常可以融合到前面的操作中）。AWQ 而不是使用固定的缩放因子，执行对最优因子的搜索，从而实现最低的量化误差。要实现 AWQ，你可以使用 Hugging Face Optimum 库。

使用*GPT-Generated Unified Format*（GGUF）的 Llama.cpp 量化

[GGUF](https://homl.info/gguf)是一种二进制文件格式，旨在高效地存储 LLMs。它由 llama.cpp 的创造者 Georgi Gerganov 引入，并取代了之前的文件格式，如 GGML、GGMF 和 GGJT。一个 GGUF 文件包括权重、分词器、特殊标记、模型架构、词汇量大小和其他元数据。Llama.cpp 提供了量化器（例如，使用`quantize`工具）将模型权重转换为 GGUF 支持的量化格式之一，例如 Q4_K_M。Q4 代表 4 位量化，K 代表每块量化（通常取决于所选格式，每块 32 或 64 个权重），M 表示此量化级别的中等大小和精度（其他选项是 S = 小型和 L = 大型）。还有更近期的更高效的量化选项，如重要性感知量化（IQ），它使用各种技术来提高准确性（例如，非线性量化），以及三值量化（TQ）。

### 注意

在 Hugging Face Hub 上，每个存储库都由 Git 支持，因此它有分支和提交。当你调用`from_pretrained()`时，模型是从默认分支获取的，这几乎总是`main`。但是量化模型通常放在不同的分支中。当调用`from_pretrained()`时，你可以通过使用`revision`参数来选择分支、标签，甚至提交哈希。检查模型卡片以获取可用文件和版本的列表。对于 GGUF 模型，你必须使用`gguf_file`参数指定文件名。

总之，降低精度、混合精度训练和量化可能是允许大型模型在有限硬件上运行的最重要工具。但还有更多，包括以下内容：

+   你可以在训练之前调整模型的架构，例如通过减少层数、每层的神经元数量，或者通过在层之间共享权重（例如，在第十五章中介绍的 ALBERT 模型中就是这样做的）。
+   如果你有一个大型训练好的模型，你可以通过移除其中的一些权重来缩小它，例如那些幅度最小的权重，或者那些对损失影响最小的权重。你也可以移除整个通道、层或注意力头。这被称为*模型剪枝*，你可以使用`torch.nn.utils.prune`模块或 Hugging Face Optimum 库来实现它。
+   如我们在第十五章中看到的，你也可以使用一个大型训练好的模型作为教师来训练一个较小的模型：这被称为蒸馏。
+   训练好的模型也可以通过融合其部分层来缩小，移除冗余。例如，一个批归一化层（在第十一章[deep_chapter]中介绍）执行线性操作，所以如果它紧接在一个线性层之后，你可以将这两个层融合成一个单一的线性层。同样，你也可以将一个卷积层和一个批归一化层融合成一个单一的卷积层。这只能在训练后进行，因为批归一化层必须在训练期间计算运行平均值。你可以使用`torch.quantization.fuse_modules()`函数或 Hugging Face Optimum 库来实现层融合。无论如何，确保在量化你的模型之前融合层：层越少，量化噪声就越少。
+   你可以使用低秩近似，其中一个大矩阵被两个较小矩阵的乘积所替代。例如，将一个大的线性层，如`Linear(10_000, 20_000)`，替换为两个线性层`Linear(10_000, 100)`和`Linear(100, 20_000)`。这将从大约两亿个参数减少到仅仅三百万个，同时也极大地减少了计算。中间维度（本例中的 100）是一个可以调整的超参数，以平衡准确性和模型大小。这种技术可以通过使用 SVD（参见笔记本中的示例）对权重矩阵进行分解来在训练后执行。

尝试这些技术：缩小模型！

### 注意

第十七章和附录 C、D 和 E 可在网上找到[*https://homl.info*](https://homl.info)。

^(1) 通常，-0 和+0 被认为是相等的，但某些操作会给出不同的结果，例如 1 / -0 = -infinity，而 1 / +0 = +infinity。

^(2) 一些高性能计算应用会禁用非正常数值，因为它们会减慢计算速度，而归一化数值通常足够（例如，归一化的 fp32 可以表示±1.2e^(–38)这样小的数值）。

^(3) “*M*”代表**尾数**，这是一个常被用作分数同义词的术语。不幸的是，它也被用作有效数字的同义词，导致了一些混淆。这就是为什么 IEEE 754 标准不再使用“尾数”这个术语。

^(4) P. Micikevicius 等人，“混合精度训练”，arXiv 预印本 2017，ICLR (2018)。

^(5) PyTorch 实现了**限制对称量化**，这意味着它排除了可能的最小有符号整数（例如，8 位整数的 -128）以确保范围是对称的（例如，-127 到 +127）。一些其他实现允许完整的有符号字节范围（从 -128 到 +127）：这被称为**非限制对称量化**。这些实现也在 方程 B-2 的分母中减去 0.5 而不是 1。

^(6) Benoit Jacob 等人，“用于高效整数算术推理的神经网络量化与训练”，arXiv 预印本 arXiv:1712.05877 (2017)”。

^(7) Tim Dettmers 等人，“QLORA：量化 LLM 的有效微调”，arXiv 预印本 arXiv:2305.14314 (2023)。

^(8) Elias Frantar 等人，“GPTQ：生成预训练变换器的准确后训练量化”，arXiv 预印本 arXiv:2210.17323 (2022)。

^(9) Ji Lin 等人，“AWQ：用于 LLM 压缩和加速的激活感知权重量化”，arXiv 预印本 arXiv:2306.00978 (2023)。
