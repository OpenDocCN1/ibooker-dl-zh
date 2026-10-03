
`SpamDataset`类从我们之前创建的 CSV 文件中加载数据，使用`tiktoken`的 GPT-2 分词器对文本进行分词，并允许我们将序列*填充*或*截断*到由最长序列或预定义的最大长度决定的统一长度。这确保了每个输入张量的大小相同，这对于创建我们接下来实现的训练数据加载器中的批次是必要的：

```py
train_dataset = SpamDataset(
    csv_file="train.csv",
    max_length=None,
    tokenizer=tokenizer
)
```

最长序列长度存储在数据集的`max_length`属性中。如果你想知道最长序列中标记的数量，可以使用以下代码：

```py
print(train_dataset.max_length)
```

代码输出`120`，表明最长序列中不超过 120 个标记，这是文本消息的常见长度。考虑到其上下文长度限制，模型可以处理最多 1,024 个标记的序列。如果你的数据集中包含更长的文本，你可以在创建训练数据集时传递`max_length=1024`，以确保数据不超过模型支持的输入（上下文）长度。

接下来，我们将验证集和测试集填充到最长训练序列的长度。重要的是，任何超过最长训练示例长度的验证集和测试集样本都会使用我们之前定义的`SpamDataset`代码中的`encoded_text[:self.max_length]`进行截断。这种截断是可选的；如果验证集和测试集中没有超过 1,024 个标记的序列，则可以将`max_length=None`设置为两者：

```py
val_dataset = SpamDataset(
    csv_file="validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
test_dataset = SpamDataset(
    csv_file="test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
```

### 练习 6.1 增加上下文长度

将输入填充到模型支持的标记数最大值，并观察它如何影响预测性能。

使用数据集作为输入，我们可以像处理文本数据时一样实例化数据加载器。然而，在这种情况下，目标表示类别标签，而不是文本中的下一个标记。例如，如果我们选择批大小为 8，每个批次将包含八个长度为 120 的训练示例以及每个示例对应的类别标签，如图 6.7 所示。

![图](img/6-7.png)

#### 图 6.7 显示了一个包含八个文本消息的训练批次，这些文本消息以标记 ID 表示。每个文本消息由 120 个标记 ID 组成。一个类别标签数组存储了与文本消息对应的八个类别标签，这些标签可以是`0`（“非垃圾邮件”）或`1`（“垃圾邮件”）。

下面的代码创建训练、验证和测试集数据加载器，这些加载器以 8 个批次的容量加载文本消息和标签。

#### 列表 6.5 创建 PyTorch 数据加载器

```py
from torch.utils.data import DataLoader

num_workers = 0      #1
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)
```

#1 此设置确保与大多数计算机的兼容性。

为了确保数据加载器正在正常工作，并且确实返回了预期大小的批次，我们遍历训练加载器，然后打印最后一个批次的张量维度：

```py
for input_batch, target_batch in train_loader:
    pass
print("Input batch dimensions:", input_batch.shape)
print("Label batch dimensions", target_batch.shape)
```

输出是

```py
Input batch dimensions: torch.Size([8, 120])
Label batch dimensions torch.Size([8])
```

如我们所见，输入批次由八个训练示例组成，每个示例有 120 个标记，正如预期的那样。标签张量存储了与八个训练示例对应的类别标签。

最后，为了了解数据集的大小，让我们打印每个数据集中的总批次数：

```py
print(f"{len(train_loader)} training batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} test batches")
```

每个数据集中的批次数是

```py
130 training batches
19 validation batches
38 test batches
```

现在我们已经准备好了数据，我们需要为微调准备模型。

## 6.4 使用预训练权重初始化模型

我们必须为分类微调准备模型以识别垃圾邮件。我们首先初始化我们的预训练模型，如图 6.8 所示。

![图](img/6-8.png)

#### 图 6.8 对 LLM 进行分类微调的三阶段过程。在完成阶段 1，准备数据集后，我们现在必须初始化 LLM，然后我们将对其进行微调以分类垃圾邮件。

要开始模型准备过程，我们使用与预训练未标记数据相同的配置：

```py
CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"
BASE_CONFIG = {
    "vocab_size": 50257,          #1
    "context_length": 1024,       #2
    "drop_rate": 0.0,             #3
    "qkv_bias": True              #4
}
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
```

#1 词汇量大小

#2 上下文长度

#3 Dropout 率

#4 查询-键-值偏差

接下来，我们从`gpt_download.py`文件中导入`download_and_load_gpt2`函数，并重用预训练中的`GPTModel`类和`load_weights_into_gpt`函数（参见第五章）将下载的权重加载到 GPT 模型中。

#### 列表 6.6 加载预训练的 GPT 模型

```py
from gpt_download import download_and_load_gpt2
from chapter05 import GPTModel, load_weights_into_gpt

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(
    model_size=model_size, models_dir="gpt2"
)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()
```

在将模型权重加载到`GPTModel`之后，我们重用第四章和第五章中的文本生成实用函数，以确保模型生成连贯的文本：

```py
from chapter04 import generate_text_simple
from chapter05 import text_to_token_ids, token_ids_to_text

text_1 = "Every effort moves you"
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,
    context_size=BASE_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))
```

以下输出显示模型生成了连贯的文本，这表明模型权重已正确加载：

```py
Every effort moves you forward.
The first step is to understand the importance of your work
```

在我们开始将模型作为垃圾邮件分类器进行微调之前，让我们看看模型是否已经通过提示指令来对垃圾邮件消息进行分类：

```py
text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_2, tokenizer),
    max_new_tokens=23,
    context_size=BASE_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))
```

模型输出是

```py
Is the following text 'spam'? Answer with 'yes' or 'no': 'You are a winner
you have been specially selected to receive $1000 cash 
or a $2000 award.'
The following text 'spam'? Answer with 'yes' or 'no': 'You are a winner
```

根据输出，很明显模型在遵循指令方面有困难。这个结果是预期的，因为它只经过了预训练，缺乏指令微调。因此，让我们为分类微调准备模型。

## 6.5 添加分类头

我们必须修改预训练的 LLM，以便为分类微调做准备。为此，我们用一个小型的输出层替换了原始的输出层，该输出层将隐藏表示映射到 50,257 个词汇，新的输出层映射到两个类别：`0`（“非垃圾邮件”）和`1`（“垃圾邮件”），如图 6.9 所示。我们使用与之前相同的模型，只是替换了输出层。

![figure](img/6-9.png)

#### 图 6.9 通过改变其架构来适应 GPT 模型进行垃圾邮件分类。最初，模型的线性输出层将 768 个隐藏单元映射到 50,257 个标记的词汇表。为了检测垃圾邮件，我们用一个新的输出层替换了这个层，该层将相同的 768 个隐藏单元映射到仅两个类别，代表“垃圾邮件”和“非垃圾邮件”。

#### 输出层节点

从技术上讲，我们可以使用单个输出节点，因为我们处理的是一个二元分类任务。然而，这将需要修改损失函数，正如我在“损失函数学习——在 PyTorch 中优化负对数似然和交叉熵”中讨论的那样。因此，我们选择了一个更通用的方法，其中输出节点的数量与类别的数量相匹配。例如，对于将新闻文章分类为“技术”、“体育”或“政治”的三个类别问题，我们将使用三个输出节点，依此类推。

在尝试图 6.9 中显示的修改之前，让我们通过`print(model)`打印出模型架构：

```py
GPTModel(
  (tok_emb): Embedding(50257, 768)
  (pos_emb): Embedding(1024, 768)
  (drop_emb): Dropout(p=0.0, inplace=False)
  (trf_blocks): Sequential(
...
    (11): TransformerBlock(
      (att): MultiHeadAttention(
        (W_query): Linear(in_features=768, out_features=768, bias=True)
        (W_key): Linear(in_features=768, out_features=768, bias=True)
        (W_value): Linear(in_features=768, out_features=768, bias=True)
        (out_proj): Linear(in_features=768, out_features=768, bias=True)
        (dropout): Dropout(p=0.0, inplace=False)
      )
      (ff): FeedForward(
        (layers): Sequential(
          (0): Linear(in_features=768, out_features=3072, bias=True)
          (1): GELU()
          (2): Linear(in_features=3072, out_features=768, bias=True)
        )
      )
      (norm1): LayerNorm()
      (norm2): LayerNorm()
      (drop_resid): Dropout(p=0.0, inplace=False)
    )
  )
  (final_norm): LayerNorm()
  (out_head): Linear(in_features=768, out_features=50257, bias=False)
)
```

这个输出清晰地展示了我们在第四章中提出的架构。正如之前讨论的那样，`GPTModel`由嵌入层组成，随后是 12 个相同的*transformer blocks*（为了简洁起见，只显示了最后一个块），然后是一个最终的`LayerNorm`和输出层`out_head`。

接下来，我们将`out_head`替换为一个新的输出层（见图 6.9），我们将对其进行微调。

#### 微调选定层与所有层

由于我们从一个预训练模型开始，没有必要微调所有模型层。在基于神经网络的自然语言模型中，底层通常捕捉适用于广泛任务和数据集的基本语言结构和语义。因此，仅微调最后几层（即靠近输出的层），这些层更具体于细微的语言模式和特定任务的特性，通常足以使模型适应新任务。一个很好的副作用是，仅微调少量层在计算上更有效率。感兴趣的读者可以在附录 B 中找到更多关于应该微调哪些层的信息，包括实验。

为了让模型准备好进行分类微调，我们首先将模型**冻结**，这意味着我们使所有层不可训练：

```py
for param in model.parameters():
    param.requires_grad = False
```

然后，我们替换输出层（`model.out_head`），它最初将层输入映射到 50,257 维，即词汇表的大小（见图 6.9）。

#### 列表 6.7 添加分类层

```py
torch.manual_seed(123)
num_classes = 2
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"], 
    out_features=num_classes
)
```

为了使代码更通用，我们使用`BASE_CONFIG["emb_dim"]`，在`"gpt2-small` `(124M)"`模型中等于 768。因此，我们也可以使用相同的代码来处理更大的 GPT-2 模型变体。

这个新的`model.out_head`输出层默认将其`requires_grad`属性设置为`True`，这意味着它是模型中唯一将在训练期间更新的层。技术上，仅训练我们刚刚添加的输出层就足够了。然而，正如我在实验中发现的那样，微调额外的层可以显著提高模型的预测性能。（更多细节请参阅附录 B。）我们还配置了最后一个转换器块和最终的`LayerNorm`模块，该模块将此块连接到输出层，如图 6.10 所示。

![figure](img/6-10.png)
