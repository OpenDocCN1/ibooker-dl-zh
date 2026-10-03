
我们的软提示调整涉及将软嵌入添加到实际输入数据的嵌入之前，所以我们这样做：

```py
inputs_embeds = torch.cat([prompt_embeddings, embeddings], dim=1)
```

BERT 使用`attention_mask`来过滤掉在训练或推理时间我们不关心的标记，通常是填充标记。但我们要让 BERT 关注所有的软提示标记，所以我们将它们的注意力掩码设置为全 1，然后将它们附加到训练数据的传入注意力掩码（`s`）中。以下是代码：

```py
prompt_attention_mask = torch.ones(batch_size, self.n_tokens, 
                                   device=attention_mask.device)

attention_mask = torch.cat([prompt_attention_mask, attention_mask], 
                            dim=1)
```

现在我们已经完成了所有的调整，我们需要将数据传递给模型以进行优化和计算损失：

```py
return self.bert(
    inputs_embeds=inputs_embeds,
    attention_mask=attention_mask,
    labels=labels,
    return_dict=True
)
```

接下来，我们将看到这些数据如何在训练循环中使用。

## 训练模型

这次训练的关键是我们将进行完整的、正常的训练循环，但在特殊情况下。在这种情况下，我们之前冻结了 BERT 模型中的所有内容，*除了*我们定义的作为模型参数的软提示。因此，例如，我们定义优化器如下：

```py
optimizer = AdamW(model.parameters(), lr=1e-2)
```

在这种情况下，我们使用标准代码，告诉它调整模型的参数。但唯一可调整的是软提示，所以这应该很快！

注意，学习率的值相当大。这有助于系统快速学习，但在实际系统中，你可能希望该值更小——或者至少是可调整的，从大的值开始，然后在后续的周期中减小。

现在，让我们进入训练阶段。首先，我们将设置训练循环：

```py
num_epochs = 3

# Perform the training
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
```

### 管理数据批次

对于每个批次，我们将获取列（`input_ids`和`attention_masks`）以及标签，并将它们传递给模型：

```py
for batch in tqdm(train_dataloader, 
                  desc=f'Training Epoch {epoch + 1}'):
    batch = {k: v.to(device) for k, v in batch.items()}
    labels = batch.pop('label')
    outputs = model(**batch, labels=labels)
```

这看起来与本书中较早的部分略有不同，但它基本上在做同样的事情。`tqdm`代码只是提供了一个状态栏，因为我们正在训练。我们按批次读取数据，但我们希望数据与模型位于同一设备上。例如，如果模型在 GPU 上运行，我们希望它能够访问 GPU 的内存。因此，此行将遍历每一列，读取键并将值传递到设备：

```py
batch = {k: v.to(device) for k, v in batch.items()}
```

它重新定义了读入的批次，以确保数据与模型位于同一设备上。但我们不希望标签在批次中，因为模型期望它们单独输入，所以我们使用`pop()`方法将它们从批次中移除：

```py
labels = batch.pop('label')
```

现在，我们可以使用`**batch`的缩写来将输入值的集合（在这种情况下，是`input_ids`和`attention_mask`）传递给模型的 forward 方法，并像这样与标签一起解包字典：

```py
outputs = model(**batch, labels=labels)
```

### 处理损失

前向传递将数据发送到模型并返回损失。我们使用这个来更新我们的总损失，然后我们可以进行反向传递：

```py
loss = outputs.loss
total_train_loss += loss.item()

loss.backward()
```

随着梯度的反向流动，优化器现在可以开始工作。

### 优化损失

记住`model.parameters()`只会管理*可训练未冻结*的参数，我们现在可以调用优化器。我在这里添加了*梯度裁剪*来使训练更有效率，但其余的只是调用优化器的下一步，然后清零梯度以便我们下次可以使用：

```py
clip_grad_norm_(model.parameters(), max_grad_norm)  # Add here
optimizer.step()
optimizer.zero_grad()
```

###### 注意

*梯度裁剪*背后的想法是，在反向传播过程中，梯度有时可能太大，优化器可能会采取非常大的步骤。这可能导致一个称为*梯度爆炸*的问题，其中值的改变隐藏了可能学习到的细微差别。但是，如果梯度值变得太大，裁剪会将其缩小，在这种情况下，它们甚至可能不是必需的。

## 训练期间的评估

我们还有一组测试数据，因此我们可以评估模型在整个训练周期中的表现。在每个 epoch 中，一旦完成前向和反向传递并且模型参数被重置，我们可以将模型切换到评估模式，然后开始将所有测试数据通过它以获取推理。我们还将比较推理结果与实际标签以计算准确率：

```py
model.eval()
val_accuracy = []
total_val_loss = 0
```

然后，我们将有类似的代码——但这次，它将是读取评估批次，将它们转换为带有标签的输出，并从模型中获取预测和损失值：

```py
with torch.no_grad():
    for batch in tqdm(eval_dataloader, desc='Validating'):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop('label')

        outputs = model(**batch, labels=labels)
        total_val_loss += outputs.loss.item()

        predictions = torch.argmax(outputs.logits, dim=-1)
        val_accuracy.extend((predictions == labels).cpu().numpy())
```

一旦我们计算了这些值，那么在每个 epoch 结束时，我们可以报告它们和训练损失。

## 报告训练指标

在每个 epoch 的训练中，我们计算了训练损失，因此我们现在可以获取所有记录的平均值。我们可以用验证损失做同样的事情，当然还有验证准确率，然后报告它们：

```py
avg_train_loss = total_train_loss / len(train_dataloader)
avg_val_loss = total_val_loss / len(eval_dataloader)
val_accuracy = np.mean(val_accuracy)

print(f"\nEpoch {epoch + 1}:")
print(f"Average training loss: {avg_train_loss:.4f}")
print(f"Average validation loss: {avg_val_loss:.4f}")
print(f"Validation accuracy: {val_accuracy:.4f}")
```

进行三个 epoch 的训练给出以下结果：

```py
Training Epoch 1: 100%|██████████| 79/79 [01:01<00:00, 1.28it/s]
Validating: 100%|██████████| 40/40 [00:27<00:00, 1.44it/s]

Epoch 1:
Average training loss: 0.6559
Average validation loss: 0.6037
Validation accuracy: 0.8036
Training Epoch 2: 100%|██████████| 79/79 [01:01<00:00, 1.28it/s]
Validating: 100%|██████████| 40/40 [00:27<00:00, 1.44it/s]

Epoch 2:
Average training loss: 0.6112
Average validation loss: 0.5854
Validation accuracy: 0.8386
Training Epoch 3: 100%|██████████| 79/79 [01:01<00:00, 1.28it/s]
Validating: 100%|██████████| 40/40 [00:27<00:00, 1.44it/s]

Epoch 3:
Average training loss: 0.5799
Average validation loss: 0.5270
Validation accuracy: 0.8736
```

这是在 Colab 上的 A100 上完成的，有 40 Gb 的 GRAM，正如你所看到的，每个 epoch 的训练只需大约 1 分钟，评估只需 30 秒。

到最后，平均训练损失从大约 0.65 下降到 0.58。准确率为 0.8736。因此，很可能是过拟合，因为我们只训练了三个 epoch。

## 保存提示嵌入

这种方法的真正好处是，当你完成时，你可以简单地保存提示嵌入。你还可以在稍后加载它们以进行推理，正如你将在下一节中看到的：

```py
torch.save(model.prompt_embeddings, "imdb_prompt_embeddings.pt")
```

我觉得这真的很酷的是，这个文件相对较小（61 K），并且它不需要你以任何方式修改底层模型。因此，在应用中，你可能会有一系列这样的提示调整文件，并且可以根据需要热插拔和替换它们，这样你就可以拥有多个可以协调的模型，这是代理解决方案的基础。

## 使用模型进行推理

要使用提示调整模型进行推理，你只需用软提示定义模型，然后，而不是训练它们，从磁盘加载预训练的软提示。我们将在本节中探讨这一点。如果你不想训练自己的模型，那么[在下载](https://github.com/lmoroney/PyTorch-Book-FIles)中，我提供了一种经过 30 个 epoch 训练的模型版本的软提示，而不是 3 个 epoch。

为了更整洁的封装，我创建了一个类，它类似于我们用于训练的类，但仅用于推理。我称它为`PromptTunedBERTInference`，以下是它的初始化器：

```py
class PromptTunedBERTInference:
    def __init__(self, model_name="bert-base-uncased", 
                       prompt_path="imdb_prompt_embeddings.pt"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model =  
                  AutoModelForSequenceClassification.from_pretrained(
                                            model_name, num_labels=2)
        self.model.eval()
        self.prompt_embeddings = torch.load(prompt_path)
        self.device = 
           torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
```

它与可训练的初始化器非常相似，除了几个重要的点。首先是因为我们*仅*使用它进行推理，我已经将其设置为评估模式：

```py
self.model.eval()
```

第二点是我们不需要训练嵌入并执行所有相关操作——相反，我们只需从指定的路径加载它们：

```py
self.prompt_embeddings = torch.load(prompt_path)
```

就这样！如你所见，它相当轻量级且非常直接。它不会有`forward`函数，因为我们没有在训练它，但让我们添加一个`predict`函数来封装使用它的推理。

### `predict`函数

`predict`函数的职责是接收我们想要进行推理的字符串（字符串），对其进行标记化（它们），然后将它们与前置的软标记一起传递给模型。让我们逐块查看代码。

首先，让我们定义它，并让它接受它将随后进行标记化的文本：

```py
def predict(self, text):
    inputs = self.tokenizer(text, padding=True, 
                        truncation=True,
                        max_length=512-self.prompt_embeddings.shape[1],
                        return_tensors="pt")
    inputs = {k: v.to(self.device) for k, v in inputs.items()}
```

文本将被标记化到最大长度，减去软提示的大小，然后输入中的每个项目将被加载到一个字典中。请注意，标记器将为文本返回多个列——通常是标记和注意力掩码——因此我们将遵循这种方法将它们转换成一组易于我们以后处理的键值对。

现在我们有了输入，是时候将它们传递给模型以获取我们的推理结果了。我们首先将`torch`置于`no_grad()`模式，因为我们不感兴趣于训练梯度。然后我们将为我们的每个标记获取嵌入：

```py
with torch.no_grad():
    embeddings = self.model.bert.embeddings.word_embeddings(
                                        inputs['input_ids'])

    batch_size = embeddings.shape[0]

    prompt_embeds = self.prompt_embeddings.expand(
                         batch_size, –1, –1).to(self.device)

    inputs_embeds = torch.cat([prompt_embeds, embeddings], dim=1)
```

我们有一个由标记器生成的输入注意力掩码，但没有软提示的注意力掩码。所以，让我们创建一个，并将其附加到输入注意力掩码上：

```py
attention_mask = inputs['attention_mask']
prompt_attention = torch.ones(batch_size, self.prompt_embeddings.shape[1],
                            device=self.device)
attention_mask = torch.cat([prompt_attention, attention_mask], dim=1)
```

现在我们已经准备好了所有东西，我们可以将我们的数据传递给模型以获取我们的推理结果：

```py
outputs = self.model(inputs_embeds=inputs_embeds,
                   attention_mask=attention_mask)
```

输出将是两个神经元的 logits，一个表示积极情绪，另一个表示消极。然后我们可以对它们进行 Softmax 以获得预测：

```py
probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
return {"prediction": outputs.logits.argmax(-1).item(),
       "confidence": probs.max(-1).values.item()}
```

### 使用示例

使用这个类进行预测是非常直接的。我们创建类的实例，并传递一个字符串给它以获取结果。结果将包含一个预测和一个置信度值，然后我们可以将其输出。
