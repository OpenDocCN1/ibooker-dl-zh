
## 图 5.16 保存模型

在训练和检查模型后，保存模型通常很有帮助，这样我们就可以稍后使用或继续训练它（步骤 6）。

幸运的是，保存 PyTorch 模型相对简单。推荐的方法是使用`torch.save`函数保存模型的`state_dict`，这是一个将每一层映射到其参数的字典：

```py
torch.save(model.state_dict(), "model.pth")
```

`"model.pth"`是保存`state_dict`的文件名。`.pth`扩展名是 PyTorch 文件的约定，尽管技术上我们可以使用任何文件扩展名。

然后在通过`state_dict`保存模型权重之后，我们可以将模型权重加载到一个新的`GPTModel`模型实例中：

```py
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
```

如第四章所述，dropout 通过在训练过程中随机“丢弃”层中的神经元来帮助防止模型过度拟合训练数据。然而，在推理过程中，我们不希望随机丢弃网络学习到的任何信息。使用`model.eval()`将模型切换到评估模式进行推理，禁用`model`的 dropout 层。如果我们计划稍后继续预训练模型——例如，使用本章前面定义的`train_model_simple`函数——保存优化器状态也是推荐的。

自适应优化器，如 AdamW，为每个模型权重存储额外的参数。AdamW 使用历史数据来动态调整每个模型参数的学习率。没有它，优化器会重置，模型可能学习不佳，甚至无法正确收敛，这意味着它将失去生成连贯文本的能力。使用`torch.save`，我们可以保存模型和优化器的`state_dict`内容：

```py
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    }, 
    "model_and_optimizer.pth"
)
```

然后，我们可以通过首先使用`torch.load`加载保存的数据，然后使用`load_state_dict`方法来恢复模型和优化器状态：

```py
checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train();
```

### 练习 5.4

保存权重后，在新的 Python 会话或 Jupyter 笔记本文件中加载模型和优化器，并使用`train_model_simple`函数继续进行一个 epoch 的预训练。

## 5.5 从 OpenAI 加载预训练权重

在之前，我们使用一个包含短篇小说集的有限数据集训练了一个小的 GPT-2 模型。这种方法使我们能够专注于基础，而无需大量的时间和计算资源。

幸运的是，OpenAI 公开分享了他们的 GPT-2 模型的权重，从而消除了我们自己在大型语料库上重新训练模型所需的数十万甚至数百万美元的投资。因此，让我们将这些权重加载到我们的`GPTModel`类中，并使用该模型进行文本生成。在这里，*权重*指的是存储在 PyTorch 的`Linear`和`Embedding`层的`.weight`属性中的权重参数，例如。我们在训练模型时通过`model.parameters()`访问过它们。在第六章中，我们将重用这些预训练的权重来微调模型以进行文本分类任务，并遵循类似于 ChatGPT 的说明。

注意，OpenAI 最初通过 TensorFlow 保存了 GPT-2 的权重，我们必须安装 TensorFlow 才能在 Python 中加载权重。以下代码将使用名为`tqdm`的进度条工具来跟踪下载过程，我们同样需要安装它。

您可以通过在终端中执行以下命令来安装这些库：

```py
pip install tensorflow>=2.15.0  tqdm>=4.66
```

下载代码相对较长，主要是样板代码，并不很有趣。因此，我们不会在讨论从互联网上获取文件的 Python 代码上浪费宝贵空间，而是直接从本章的在线仓库下载`gpt_download.py` Python 模块：

```py
import urllib.request
url = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch05/"
    "01_main-chapter-code/gpt_download.py"
)
filename = url.split('/')[-1]
urllib.request.urlretrieve(url, filename)
```

接下来，在将此文件下载到 Python 会话的本地目录后，您应该简要检查此文件的内容，以确保它已正确保存并包含有效的 Python 代码。

现在，我们可以按照以下方式从`gpt_download.py`文件中导入`download_and_load_gpt2`函数，这将把 GPT-2 架构设置（`settings`）和权重参数（`params`）加载到我们的 Python 会话中：

```py
from gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2"
)
```

执行此代码将下载与`124M`参数 GPT-2 模型相关的以下七个文件：

```py
checkpoint: 100%|███████████████████████████| 77.0/77.0 [00:00<00:00, 
                                                         63.9kiB/s]
encoder.json: 100%|█████████████████████████| 1.04M/1.04M [00:00<00:00,
                                                           2.20MiB/s]
hprams.json: 100%|██████████████████████████| 90.0/90.0 [00:00<00:00,
                                                         78.3kiB/s]
model.ckpt.data-00000-of-00001: 100%|███████| 498M/498M [01:09<00:00,
                                                         7.16MiB/s]
model.ckpt.index: 100%|█████████████████████| 5.21k/5.21k [00:00<00:00,
                                                           3.24MiB/s]
model.ckpt.meta: 100%|██████████████████████| 471k/471k [00:00<00:00, 
                                                         2.46MiB/s]
vocab.bpe: 100%|████████████████████████████| 456k/456k [00:00<00:00,
                                                         1.70MiB/s]
```

注意：如果下载代码对您不起作用，可能是由于间歇性互联网连接、服务器问题或 OpenAI 分享开源 GPT-2 模型权重的变化方式。在这种情况下，请访问本章的在线代码仓库[`github.com/rasbt/LLMs-from-scratch`](https://github.com/rasbt/LLMs-from-scratch)，以获取替代和更新的说明，并通过 Manning 论坛提出进一步的问题。

假设上一段代码已执行完成，让我们检查`settings`和`params`的内容：

```py
print("Settings:", settings)
print("Parameter dictionary keys:", params.keys())
```

内容如下

```py
Settings: {'n_vocab': 50257, 'n_ctx': 1024, 'n_embd': 768, 'n_head': 12,
           'n_layer': 12}
Parameter dictionary keys: dict_keys(['blocks', 'b', 'g', 'wpe', 'wte'])
```

`settings`和`params`都是 Python 字典。`settings`字典存储了 LLM 架构设置，类似于我们手动定义的`GPT_CONFIG_124M`设置。`params`字典包含实际的权重张量。请注意，我们只打印了字典键，因为打印权重内容会占用太多的屏幕空间；然而，我们可以通过打印整个字典`print(params)`或通过选择单个张量，例如通过相应的字典键来检查这些权重张量，例如嵌入层权重：

```py
print(params["wte"])
print("Token embedding weight tensor dimensions:", params["wte"].shape)
```

令牌嵌入层的权重是

```py
[[-0.11010301 ... -0.1363697   0.01506208   0.04531523]
 [ 0.04034033 ...  0.08605453  0.00253983   0.04318958]
 [-0.12746179  ...  0.08991534 -0.12972379 -0.08785918]
 ...
 [-0.04453601 ...   0.10435229  0.09783269 -0.06952604]
 [ 0.1860082  ...  -0.09625227  0.07847701 -0.02245961]
 [ 0.05135201 ...   0.00704835  0.15519823  0.12067825]]
Token embedding weight tensor dimensions: (50257, 768)
```

我们通过`download_and_load_gpt2(model_size="124M", ...)`设置下载并加载了最小的 GPT-2 模型权重。OpenAI 还分享了更大模型的权重：`355M`、`774M`和`1558M`。这些不同尺寸的 GPT 模型的总体架构是相同的，如图 5.17 所示，除了不同的架构元素重复的次数不同，以及嵌入大小不同。本章剩余的代码也与这些更大的模型兼容。

![figure](img/5-17.png)

### 图 5.17 GPT-2 LLM 有几种不同的模型尺寸，从 1.24 亿到 1.558 亿参数不等。核心架构是相同的，唯一的区别是嵌入大小以及注意力头和 Transformer 块等单个组件重复的次数。

在将 GPT-2 模型权重加载到 Python 中之后，我们仍然需要将它们从`settings`和`params`字典转移到我们的`GPTModel`实例中。首先，我们创建一个字典，列出图 5.17 中不同 GPT 模型尺寸之间的差异：

```py
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
```

假设我们感兴趣的是加载最小的模型`"gpt2-small"`（124M）。我们可以使用`model_configs`表中的相应设置来更新我们之前定义并使用的完整长度的`GPT_CONFIG_124M`：

```py
model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
```

仔细的读者可能还记得我们之前使用的是 256 标记长度，但 OpenAI 的原始 GPT-2 模型是在 1,024 标记长度下训练的，因此我们必须相应地更新`NEW_CONFIG`：

```py
NEW_CONFIG.update({"context_length": 1024})
```

此外，OpenAI 在多头注意力模块的线性层中使用了偏置向量来实现查询、键和值矩阵的计算。偏置向量在 LLMs 中不再常用，因为它们不会提高建模性能，因此是不必要的。然而，由于我们正在使用预训练的权重，我们需要匹配设置以保持一致性并启用这些偏置向量：

```py
NEW_CONFIG.update({"qkv_bias": True})
```

现在我们可以使用更新的`NEW_CONFIG`字典来初始化一个新的`GPTModel`实例：

```py
gpt = GPTModel(NEW_CONFIG)
gpt.eval()
```

默认情况下，`GPTModel`实例使用随机权重进行预训练。使用 OpenAI 模型权重的最后一步是用我们加载到`params`字典中的权重覆盖这些随机权重。为此，我们首先定义一个小的`assign`实用函数，该函数检查两个张量或数组（`left`和`right`）是否具有相同的维度或形状，并返回正确的张量作为可训练的 PyTorch 参数：

```py
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                          "Right: {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right))
```

接下来，我们定义一个`load_weights_into_gpt`函数，该函数将`params`字典中的权重加载到`GPTModel`实例`gpt`中。

#### 列表 5.5 将 OpenAI 权重加载到我们的 GPT 模型代码中
