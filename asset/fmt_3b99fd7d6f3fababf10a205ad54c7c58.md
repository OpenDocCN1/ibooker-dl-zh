
```py
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
config_small = AutoConfig.from_pretrained("gpt2", vocab_size=len(tokenizer))
model_small = AutoModelForCausalLM.from_config(config_small)
```

```py
print(f'GPT-2 size: {model_size(model_small)/1000**2:.1f}M parameters')
```

```py
GPT-2 size: 111.0M parameters
```

同时，让我们也保存到 Hub 上，以便轻松共享和重复使用：

```py
model_small.save_pretrained("models/" + model_ckpt + "-small", push_to_hub=True,
                            organization=org)
```

现在我们有两个可以训练的模型，我们需要确保在训练期间能够有效地向它们提供输入数据。

## 实现数据加载器

为了能够以最大效率进行训练，我们将希望为我们的模型提供填充其上下文的序列。例如，如果我们的模型的上下文长度为 1,024 个标记，我们在训练期间始终希望提供 1,024 个标记的序列。但是，我们的一些代码示例可能比 1,024 个标记短或长。为了向我们的模型提供`sequence_length`的完整序列，我们因此应该丢弃最后一个不完整的序列或对其进行填充。然而，这将使我们的训练略微不那么高效，并迫使我们处理填充和屏蔽填充的标记标签。我们的计算资源比数据资源多得多，所以我们将采取简单和高效的方式。我们可以使用一个小技巧来确保我们不会丢失太多的尾随段：我们可以对几个示例进行标记化，然后连接它们，用特殊的序列结束标记分隔，以获得一个非常长的序列。最后，我们将这个序列分成相等大小的块，如图 10-5 所示。通过这种方法，我们最多只会丢失数据的一小部分。

![CLM 的预处理](img/nlpt_1005.png)

###### 图 10-5\. 通过连接几个标记化示例并在分块之前使用 EOS 标记来准备因果语言建模的不同长度序列

例如，我们可以通过将输入字符串的字符长度定义为大约一百个完整序列的 tokenized 示例来确保我们有大约一百个完整序列：

```py
input_characters = number_of_sequences * sequence_length * characters_per_token
```

其中：

+   `input_characters`是输入到我们的 tokenizer 的字符串的字符数。
+   `number_of_sequences`是我们希望从我们的 tokenizer 中获得的（截断的）序列数（例如，100）。
+   `sequence_length`是 tokenizer 返回的每个序列的标记数（例如，1,024）。
+   `characters_per_token`是我们首先需要估计的每个输出标记的平均字符数。

如果我们输入一个具有`input_characters`字符的字符串，我们平均会得到`number_of_sequences`个输出序列，我们可以轻松计算通过丢弃最后一个序列而丢失了多少输入数据。如果`number_of_sequences=100`，这意味着我们大致堆叠了 100 个序列，并且最多丢失了最后一个元素，这可能太短或太长。这最多相当于我们数据集的 1%。同时，这种方法确保我们不会通过截断大多数文件结尾引入偏差。

让我们首先估计数据集中每个标记的平均字符长度：

```py
examples, total_characters, total_tokens = 500, 0, 0
dataset = load_dataset('transformersbook/codeparrot-train', split='train',
                       streaming=True)

for _, example in tqdm(zip(range(examples), iter(dataset)), total=examples):
    total_characters += len(example['content'])
    total_tokens += len(tokenizer(example['content']).tokens())

characters_per_token = total_characters / total_tokens
```

```py
print(characters_per_token)
```

```py
3.6233025034779565
```

有了这些，我们就有了创建自己的`IterableDataset`所需的一切（这是 PyTorch 提供的一个辅助类），用于为模型准备恒定长度的输入。我们只需要继承自`IterableDataset`并设置`__iter__()`函数，以产生我们刚刚讨论过的逻辑的下一个元素：

```py
import torch
from torch.utils.data import IterableDataset

class ConstantLengthDataset(IterableDataset):

    def __init__(self, tokenizer, dataset, seq_length=1024,
                 num_of_sequences=1024, chars_per_token=3.6):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.input_characters = seq_length * chars_per_token * num_of_sequences

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.input_characters:
                    m=f"Buffer full: {buffer_len}>={self.input_characters:.0f}"
                    print(m)
                    break
                try:
                    m=f"Fill buffer: {buffer_len}<{self.input_characters:.0f}"
                    print(m)
                    buffer.append(next(iterator)["content"])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    iterator = iter(self.dataset)

            all_token_ids = []
            tokenized_inputs = self.tokenizer(buffer, truncation=False)
            for tokenized_input in tokenized_inputs["input_ids'"]:
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])

            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    yield torch.tensor(input_ids)
```

`__iter__()`函数构建了一个字符串缓冲区，直到它包含足够的字符。缓冲区中的所有元素都被标记化并与 EOS 标记连接，然后在`all_token_ids`中的长序列中以`seq_length`大小的片段进行分块。通常，我们需要注意掩码来堆叠不同长度的填充序列，并确保在训练过程中忽略填充。我们已经通过仅提供相同（最大）长度的序列来处理了这一点，因此我们在这里不需要掩码，只返回`input_ids`。让我们测试一下我们的可迭代数据集：

```py
shuffled_dataset = dataset.shuffle(buffer_size=100)
constant_length_dataset = ConstantLengthDataset(tokenizer, shuffled_dataset,
                                                num_of_sequences=10)
dataset_iterator = iter(constant_length_dataset)

lengths = [len(b) for _, b in zip(range(5), dataset_iterator)]
print(f"Lengths of the sequences: {lengths}")
```

```py
Fill buffer: 0<36864
Fill buffer: 3311<36864
Fill buffer: 9590<36864
Fill buffer: 22177<36864
Fill buffer: 25530<36864
Fill buffer: 31098<36864
Fill buffer: 32232<36864
Fill buffer: 33867<36864
Buffer full: 41172>=36864
Lengths of the sequences: [1024, 1024, 1024, 1024, 1024]
```

很好，这正如预期的那样工作，我们为模型获得了恒定长度的输入。现在我们有了模型的可靠数据源，是时候构建实际的训练循环了。

###### 提示

请注意，在创建`ConstantLengthDataset`之前，我们对原始数据集进行了洗牌。由于这是一个可迭代的数据集，我们不能在开始时对整个数据集进行洗牌。相反，我们设置了一个大小为`buffer_size`的缓冲区，并在从数据集获取元素之前对该缓冲区中的元素进行洗牌。

## 定义训练循环

现在我们已经有了所有的元素来编写我们的训练循环。训练自己的语言模型的一个明显的限制是我们将使用的 GPU 的内存限制。即使在现代显卡上，也无法在合理的时间内训练 GPT-2 规模的模型。在本教程中，我们将实现*数据并行*，这将帮助我们利用多个 GPU 进行训练。幸运的是，我们可以使用![nlpt_pin01](img/nlpt_pin01.png) Accelerate 来使我们的代码可扩展。![nlpt_pin01](img/nlpt_pin01.png) Accelerate 库旨在使分布式训练和更改用于训练的基础硬件变得容易。我们还可以使用`Trainer`进行分布式训练，但![nlpt_pin01](img/nlpt_pin01.png) Accelerate 为我们提供了对训练循环的完全控制，这正是我们想要在这里探索的。

![nlpt_pin01](img/nlpt_pin01.png) Accelerate 提供了一个简单的 API，可以使训练脚本在混合精度和任何类型的分布式设置（单个 GPU、多个 GPU 和 TPU）下运行。然后，相同的代码可以在本地机器上无缝运行以进行调试，也可以在强大的训练集群上进行最终训练运行。您只需要对原生 PyTorch 训练循环进行少量更改：

```py
  import torch
  import torch.nn.functional as F
  from datasets import load_dataset
+ from accelerate import Accelerator

- device = 'cpu'
+ accelerator = Accelerator()

- model = torch.nn.Transformer().to(device)
+ model = torch.nn.Transformer()
  optimizer = torch.optim.Adam(model.parameters())
  dataset = load_dataset('my_dataset')
  data = torch.utils.data.DataLoader(dataset, shuffle=True)
+ model, optimizer, data = accelerator.prepare(model, optimizer, data)

  model.train()
  for epoch in range(10):
      for source, targets in data:
-         source = source.to(device)
-         targets = targets.to(device)
          optimizer.zero_grad()
          output = model(source)
          loss = F.cross_entropy(output, targets)
-         loss.backward()
+         accelerator.backward(loss)
          optimizer.step()
```

变革的核心部分是对`prepare()`的调用，它确保了模型、优化器和数据加载器都准备好并分布在基础设施上。对 PyTorch 训练循环的这些微小更改使您能够轻松地在不同的基础设施上扩展训练。有了这个想法，让我们开始构建我们的训练脚本并定义一些辅助函数。首先，我们设置训练的超参数并将它们包装在`Namespace`中以便轻松访问：

```py
from argparse import Namespace

# Commented parameters correspond to the small model
config = {"train_batch_size": 2, # 12
          "valid_batch_size": 2, # 12
          "weight_decay": 0.1,
          "shuffle_buffer": 1000,
          "learning_rate": 2e-4, # 5e-4
          "lr_scheduler_type": "cosine",
          "num_warmup_steps": 750, # 2000
          "gradient_accumulation_steps": 16, # 1
          "max_train_steps": 50000, # 150000
          "max_eval_steps": -1,
          "seq_length": 1024,
          "seed": 1,
          "save_checkpoint_steps": 50000} # 15000

args = Namespace(**config)
```

接下来，我们为训练设置日志记录。由于我们是从头开始训练模型，训练运行将需要一段时间并且需要昂贵的基础设施。因此，我们希望确保所有相关信息都被存储并且易于访问。`setup_logging()`方法设置了三个级别的日志记录：使用标准 Python [`Logger`](https://oreil.ly/P9Xrm)、[TensorBoard](https://oreil.ly/kY5ri)和[Weights & Biases](https://oreil.ly/BCC3k)。根据您的偏好和用例，您可以在这里添加或删除日志记录框架：
