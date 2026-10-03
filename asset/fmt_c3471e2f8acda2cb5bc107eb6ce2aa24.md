
如果你想修改输入文本处理的方式会发生什么？例如，假设你发现了你想改进的分词过程中的某些内容，并且你在数据加载器中修改了输入文本的分词方式。你更新了数据加载器代码，重新训练了模型，并部署了模型。然而，你忘记了在预测器中更新相应的分词代码，实际上在训练和服务之间创建了一个输入分词方式不一致的差异。这在图 11.3 中有所说明。

![CH11_F03_Hagiwara](img/CH11_F03_Hagiwara.png)

图 11.3 训练-服务偏差是由于训练和服务之间输入处理方式的差异而引起的。

修复这个问题的最佳方法——甚至更好的是，第一次就预防它发生——是在训练和服务基础设施之间尽可能共享特征管道。在 AllenNLP 中的一种常见做法是在数据集读取器中实现一个名为`_text_to_instance()`的方法，它接受一个输入并返回一个实例。通过确保数据集读取器和预测器都引用同一个方法，你可以尽量减少管道之间的差异。

在 NLP 中，输入文本被分词并转换为数字值的事实使得调试模型变得更加困难。例如，一个在分词中明显的错误，你可以用肉眼轻松发现，但如果一切都是数字值，那么很难识别。一个好的做法是将一些中间结果记录到一个日志文件中，以便稍后检查。

最后，请注意，神经网络在训练和服务之间的一些行为是不同的。一个显著的例子是`dropout`，一个我们在第 10.3.1 节中简要介绍过的正则化方法。简而言之，`dropout`通过在神经网络中随机屏蔽激活值来对模型进行正则化。这在训练中是有道理的，因为通过去除激活，模型学会根据可用值做出稳健的预测。但是，请记住在服务时关闭它，因为你不希望你的模型随机丢弃神经元。PyTorch 模型实现了`train()`和`eval()`等方法，可以在训练和预测模式之间切换，从而影响像`dropout`这样的层的行为。如果你手动实现了训练循环，请记住调用`model.eval()`来禁用`dropout`。好消息是，诸如 AllenNLP 之类的框架可以自动处理这个问题，只要你使用它们的默认训练器。

### 11.2.3 监控

与其他软件服务一样，部署的 ML 系统应该持续监控。除了通常的服务器指标（例如，CPU 和内存使用率）之外，您还应该监视与模型的输入和输出相关的指标。具体来说，您可以监视一些高级统计信息，如输入值和输出标签的分布。正如前面提到的，逻辑错误是一种导致模型产生错误结果但不会崩溃的错误类型，在 ML 系统中最常见且最难找到。监控这些高级统计信息可以更容易地找到它们。像 PyTorch Serve 和 Amazon SageMaker（在第 11.3 节讨论）这样的库和平台默认支持监控。

### 11.2.4 使用 GPU

训练大型现代 ML 模型几乎总是需要像 GPU 这样的硬件加速器。回想一下第二章中，我们将海外工厂比作了 GPU 的类比，GPU 设计用于并行执行大量的算术运算，如向量和矩阵的加法和乘法。在本小节中，我们将介绍如何使用 GPU 加速 ML 模型的训练和预测。

如果您没有自己的 GPU 或以前从未使用过基于云的 GPU 解决方案，免费“尝试” GPU 的最简单方法是使用 Google Colab。转到其 URL（[`colab.research.google.com/`](https://colab.research.google.com/)），创建一个新笔记本，转到“运行时”菜单，并选择“更改运行时类型”。这将弹出如图 11.4 所示的对话框。

![CH11_F04_Hagiwara](img/CH11_F04_Hagiwara.png)

图 11.4 Google Colab 允许您选择硬件加速器的类型。

选择 GPU 作为硬件加速器的类型，并在代码块中输入`!nvidia-smi`并执行它。将显示一些关于您的 GPU 的详细信息，如下所示：

```py
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.56       Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   39C    P8     9W /  70W |      3MiB / 15109MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+

```

`nvidia-smi`命令（简称 Nvidia 系统管理接口）是一个方便的工具，用于检查机器上 Nvidia GPU 的信息。从上面的代码片段中，您可以看到驱动程序和 CUDA（一种用于与 GPU 交互的 API 和库）的版本、GPU 类型（Tesla T4）、可用和已使用的内存（15109 MiB 和 3 MiB），以及当前使用 GPU 的进程列表（没有）。这个命令的最典型用法是检查当前进程使用了多少内存，因为在 GPU 编程中，如果您的程序使用的内存超过了可用内存，很容易出现内存不足的错误。

如果你使用云基础设施，比如 AWS（Amazon Web Services）和 GCP（Google Cloud Platform），你会发现有很多虚拟机模板，可以用来快速创建支持 GPU 的云实例。例如，GCP 提供了 Nvidia 官方的 GPU 优化图像，可以用作模板来启动 GPU 实例。AWS 提供了深度学习 AMIs（Amazon Machine Images），预先安装了基本的 GPU 库，如 CUDA，以及深度学习库，如 PyTorch。使用这些模板时，你不需要手动安装必要的驱动程序和库——你可以直接开始构建你的 ML 应用程序。请注意，尽管这些模板是免费的，但你需要为基础设施付费。支持 GPU 的虚拟机的价格通常比 CPU 机器高得多。在长时间运行之前，请确保检查它们的价格。

如果你要从头开始设置 GPU 实例，你可以找到详细的说明 ¹ 来设置必要的驱动程序和库。要使用本书中介绍的库（即，AllenNLP 和 Transformers）构建 NLP 应用程序，你需要安装 CUDA 驱动程序和工具包，以及支持 GPU 的 PyTorch 版本。

如果你的机器有 GPU，你可以通过在 AllenNLP 配置文件中指定`cuda_device`来启用 GPU 加速，如下所示：

```py
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1.0e-5
        },
        "num_epochs": 20,
        "patience": 10,
        "cuda_device": 0
}
```

这告诉训练器使用第一个 GPU 训练和验证 AllenNLP 模型。

如果你要从头开始编写 PyTorch 代码，你需要手动将模型和张量转移到 GPU 上。用比喻来说，这就像是你的材料被运往海外工厂的集装箱船上。首先，你可以指定要使用的设备（GPU ID），并调用张量和模型的`to()`方法在设备之间移动它们。例如，你可以使用以下代码片段在使用 Hugging Face Transformers 的 GPU 上运行文本生成：

```py
device = torch.device('cuda:0')
tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
model = AutoModelWithLMHead.from_pretrained("gpt2-large")

generated = tokenizer.encode("On our way to the beach ")
context = torch.tensor([generated])

model = model.to(device)
context = context.to(device)
```

其余的与我们在第 8.4 节中使用的代码相同。

## 11.3 案例研究：提供和部署 NLP 应用程序

在本节中，我们将对一个案例研究进行概述，在其中，我们使用 Hugging Face 构建了一个 NLP 模型。具体地说，我们将使用预训练的语言生成模型（DistilGPT2），使用 TorchServe 进行服务，并使用 Amazon SageMaker 部署到云服务器。

### 11.3.1 用 TorchServe 提供模型

如你所见，部署 NLP 应用程序不仅仅是编写 ML 模型的 API。你需要考虑许多与生产相关的问题，包括如何使用多个 worker 并行化模型推理来处理高流量，如何存储和管理多个 ML 模型的不同版本，如何一致地处理数据的预处理和后处理，并且如何监视服务器的健康状况以及数据的各种指标。

由于这些问题如此常见，机器学习从业者一直在研究用于服务和部署机器学习模型的通用平台。在本节中，我们将使用 TorchServe ([`github.com/pytorch/serve`](https://github.com/pytorch/serve))，这是一个由 Facebook 和 Amazon 共同开发的用于服务 PyTorch 模型的易于使用的框架。TorchServe 附带了许多功能，可以解决前面提到的问题。

TorchServe 可通过以下方式安装：

```py
pip install torchserve torch-model-archiver
```

在这个案例研究中，我们将使用一个名为 *DistilGPT2* 的预训练语言模型。DistilGPT2 是使用一种称为 *知识蒸馏* 的技术构建的 GPT-2 的较小版本。知识蒸馏（或简称 *蒸馏*）是一种机器学习技术，其中一个较小的模型（称为 *学生*）被训练成以模仿一个较大模型（称为 *教师*）产生的预测。这是训练一个产生高质量输出的较小模型的绝佳方式，通常比从头开始训练一个较小模型产生更好的模型。

首先，让我们通过运行以下命令从 Hugging Face 仓库下载预训练的 DistilGPT2 模型。请注意，您需要安装 Git Large File Storage ([`git-lfs.github.com/`](https://git-lfs.github.com/))，这是一个用于处理 Git 下大文件的 Git 扩展：

```py
git lfs install
git clone https://huggingface.co/distilgpt2
```

这将创建一个名为`distilgpt2`的子目录，其中包含`config.json`和`pytorch_model.bin`等文件。

接下来，您需要为 TorchServe 编写一个处理程序，这是一个轻量级的包装类，指定了如何初始化您的模型、预处理和后处理输入以及对输入进行推断。清单 11.1 显示了用于服务 DistilGPT2 模型的处理程序代码。实际上，处理程序中的任何内容都不特定于我们使用的特定模型（DistilGPT2）。只要使用 Transformers 库，您就可以将相同的代码用于其他类似 GPT-2 的模型，包括原始的 GPT-2 模型。

清单 11.1 TorchServe 的处理程序
