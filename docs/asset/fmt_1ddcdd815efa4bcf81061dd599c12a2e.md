
在实现数据加载器创建步骤之前，我们必须简要谈谈`custom_collate_fn`的`device`设置。`custom_collate_fn`包括将输入和目标张量（例如，`torch.stack(inputs_lst).to(device)`）移动到指定设备的代码，这可以是`"cpu"`或`"cuda"`（用于 NVIDIA GPU）或可选的`"mps"`（用于配备 Apple Silicon 芯片的 Mac）。

注意：使用`"mps"`设备可能与本章内容中的内容产生数值差异，因为 PyTorch 对 Apple Silicon 的支持仍然是实验性的。

在之前的训练循环中，我们将数据移动到了目标设备（例如，当`device="cuda"`时的 GPU 内存）。将此作为 collate 函数的一部分提供了优势，即在训练循环之外作为后台进程执行此设备传输过程，防止它阻塞模型训练时的 GPU。

以下代码初始化了`device`变量：

```py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.backends.mps.is_available():   #1
#     device = torch.device("mps")"      
print("Device:", device)
```

#1 取消注释这两行以在 Apple Silicon 芯片上使用 GPU

这将根据您的机器打印出`"Device:` cpu`或`"Device:` cuda"。

接下来，为了在将`custom_collate_fn`插入 PyTorch 的`DataLoader`类时重用选择的设备设置，我们使用 Python 的`functools`标准库中的`partial`函数创建一个带有预填充设备参数的新函数版本。此外，我们将`allowed_max_length`设置为`1024`，这将截断数据到 GPT-2 模型支持的上下文最大长度，该模型我们稍后将对其进行微调：

```py
from functools import partial

customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)
```

接下来，我们可以像之前一样设置数据加载器，但这次，我们将使用我们的自定义 collate 函数进行批处理过程。

### 列表 7.6 初始化数据加载器

```py
from torch.utils.data import DataLoader

num_workers = 0      #1
batch_size = 8

torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)
```

#1 如果您的操作系统支持并行 Python 进程，您可以尝试增加这个数字。

让我们检查由训练加载器生成的输入和目标批次的维度：

```py
print("Train loader:")
for inputs, targets in train_loader:
    print(inputs.shape, targets.shape)
```

输出如下（为了节省空间已截断）：

```py
Train loader:
torch.Size([8, 61]) torch.Size([8, 61])
torch.Size([8, 76]) torch.Size([8, 76])
torch.Size([8, 73]) torch.Size([8, 73])
...
torch.Size([8, 74]) torch.Size([8, 74])
torch.Size([8, 69]) torch.Size([8, 69])
```

此输出显示，第一个输入和目标批次具有 8 × 61 的维度，其中 8 代表批次大小，61 是此批次中每个训练示例中的标记数。第二个输入和目标批次具有不同的标记数——例如，76。多亏了我们的自定义 collate 函数，数据加载器能够创建不同长度的批次。在下文中，我们将加载一个预训练的 LLM，然后我们可以使用这个数据加载器对其进行微调。

## 7.5 加载预训练的 LLM

我们在准备指令微调数据集上花费了大量时间，这是监督微调过程中的一个关键方面。许多其他方面与预训练相同，使我们能够重用早期章节中大部分的代码。

在开始指令微调之前，我们必须首先加载一个我们想要微调的预训练 GPT 模型（见图 7.15），这是一个我们之前已经执行过的过程。然而，与之前使用最小的 1240 万参数模型不同，我们这次加载了具有 3.55 亿参数的中等大小模型。选择这个模型的原因是，1240 万参数模型在容量上过于有限，无法通过指令微调获得令人满意的结果。具体来说，较小的模型缺乏学习并保留高质量指令遵循任务所需的复杂模式和细微行为所必需的容量。

![图](img/7-15.png)

#### 图 7.15 指令微调 LLM 的三个阶段过程。在数据集准备之后，指令遵循的 LLM 微调过程开始于加载一个预训练 LLM，这作为后续训练的基础。

加载我们的预训练模型需要与我们在第 5.5 节中预训练数据时相同的代码，以及在第 6.4 节中对其进行分类微调时的代码，只是我们现在指定 `"gpt2-medium (355M)"` 而不是 `"gpt2-small (124M)"`。

注意：执行此代码将启动下载中等大小的 GPT 模型，该模型的存储需求约为 1.42 吉字节。这大约是小型模型所需存储空间的三倍。

### 列表 7.7 加载预训练模型

```py
from gpt_download import download_and_load_gpt2
from chapter04 import GPTModel
from chapter05 import load_weights_into_gpt

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

settings, params = download_and_load_gpt2(
    model_size=model_size, 
    models_dir="gpt2"
)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval();
```

执行代码后，将下载几个文件：

```py
checkpoint: 100%|██████████| 77.0/77.0 [00:00<00:00, 156kiB/s]
encoder.json: 100%|██████████| 1.04M/1.04M [00:02<00:00, 467kiB/s]
hparams.json: 100%|██████████| 91.0/91.0 [00:00<00:00, 198kiB/s]
model.ckpt.data-00000-of-00001: 100%|██████████| 1.42G/1.42G 
[05:50<00:00, 4.05MiB/s]
model.ckpt.index: 100%|██████████| 10.4k/10.4k [00:00<00:00, 18.1MiB/s]
model.ckpt.meta: 100%|██████████| 927k/927k [00:02<00:00, 454kiB/s]
vocab.bpe: 100%|██████████| 456k/456k [00:01<00:00, 283kiB/s]
```

现在，让我们花一点时间评估预训练 LLM 在验证任务中的一个表现，通过将其输出与预期响应进行比较。这将使我们获得一个基准理解，了解模型在未经微调的情况下直接执行指令遵循任务的表现如何，并有助于我们后来欣赏微调的效果。我们将使用验证集的第一个示例进行此评估：

```py
torch.manual_seed(123)
input_text = format_input(val_data[0])
print(input_text)
```

指令内容如下：

```py
Below is an instruction that describes a task. Write a response that 
appropriately completes the request.

### Instruction:
Convert the active sentence to passive: 'The chef cooks the meal every day.'
```

接下来，我们使用与第五章中预训练模型相同的`generate`函数来生成模型的响应：

```py
from chapter05 import generate, text_to_token_ids, token_ids_to_text

token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256,
)
generated_text = token_ids_to_text(token_ids, tokenizer)
```

`generate`函数返回组合的输入和输出文本。这种行为之前很方便，因为预训练 LLM 主要被设计为文本补全模型，其中输入和输出被连接起来以创建连贯且易于阅读的文本。然而，当评估模型在特定任务上的性能时，我们通常只想关注模型生成的响应。

为了隔离模型的响应文本，我们需要从`generated_text`的开始减去输入指令的长度：

```py
response_text = generated_text[len(input_text):].strip()
print(response_text)
```

此代码从`generated_text`的开头移除输入文本，只留下模型的生成响应。然后应用`strip()`函数来移除任何前导或尾随空白字符。输出如下：

```py
### Response:

The chef cooks the meal every day.

### Instruction:

Convert the active sentence to passive: 'The chef cooks the
```

此输出显示，预训练模型尚不能正确地遵循给定的指令。虽然它确实创建了一个响应部分，但它只是重复了原始输入句子和部分指令，未能将主动句转换为请求的被动语态。因此，我们现在实施微调过程，以提高模型理解和适当响应此类请求的能力。

## 7.6 在指令数据上微调 LLM

是时候微调 LLM 以处理指令了（图 7.16）。我们将使用上一节中加载的预训练模型，并使用本章前面准备好的指令数据集进一步训练它。当我们在本章开头实现指令数据集处理时，我们已经完成了所有艰苦的工作。对于微调过程本身，我们可以重用第五章中实现的损失计算和训练函数：

![figure](img/7-16.png)

#### 图 7.16 对 LLM 指令微调的三阶段过程。在第 5 步中，我们在之前准备好的指令数据集上训练之前加载的预训练模型。

```py
from chapter05 import (
    calc_loss_loader,
    train_model_simple
)
```

在我们开始训练之前，让我们计算训练集和验证集的初始损失：

```py
model.to(device)
torch.manual_seed(123)

with torch.no_grad():
    train_loss = calc_loss_loader(
        train_loader, model, device, num_batches=5
    )
    val_loss = calc_loss_loader(
        val_loader, model, device, num_batches=5
)

print("Training loss:", train_loss)
print("Validation loss:", val_loss)
```

初始损失值如下；如前所述，我们的目标是使损失最小化：

```py
Training loss: 3.825908660888672
Validation loss: 3.7619335651397705
```

### 处理硬件限制

使用和训练一个更大的模型，如 GPT-2 中等（355 百万参数），比较小的 GPT-2 模型（124 百万参数）计算量更大。如果您因硬件限制而遇到问题，可以通过将`CHOOSE_MODEL = "gpt2-medium (355M)"`更改为`CHOOSE_MODEL = "gpt2-small (124M)"`（见第 7.5 节）来切换到较小的模型。或者，为了加快模型训练，考虑使用 GPU。本书代码库中的以下补充部分列出了使用云 GPU 的几种选项：[`mng.bz/EOEq`](https://mng.bz/EOEq)。

下表提供了在包括 CPU 和 GPU 在内的各种设备上训练每个模型的参考运行时间，对于 GPT-2。在兼容的 GPU 上运行此代码无需代码更改，并且可以显著加快训练速度。对于本章中显示的结果，我使用了 GPT-2 中等模型，并在 A100 GPU 上对其进行训练。

| 模型名称 | 设备 | 两个 epoch 的运行时间 |
| --- | --- | --- |
| gpt2-medium (355M) | CPU (M3 MacBook Air) | 15.78 分钟 |
| gpt2-medium (355M) | GPU (NVIDIA L4) | 1.83 分钟 |
| gpt2-medium (355M) | GPU (NVIDIA A100) | 0.86 分钟 |
| gpt2-small (124M) | CPU (M3 MacBook Air) | 5.74 分钟 |
| gpt2-small (124M) | GPU (NVIDIA L4) | 0.69 分钟 |
| gpt2-small (124M) | GPU (NVIDIA A100) | 0.39 分钟 |

模型和数据加载器准备就绪后，我们现在可以开始训练模型。列表 7.8 中的代码设置了训练过程，包括初始化优化器、设置 epoch 数量以及定义评估频率和起始上下文，以便在训练期间根据我们在 7.5 节中查看的第一个验证集指令（`val_data[0]`）评估生成的 LLM 响应。

### 列表 7.8 指令微调预训练的 LLM
