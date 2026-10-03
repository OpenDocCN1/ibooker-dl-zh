
## 图 6.10 GPT 模型包含 12 个重复的转换器块

与输出层一起，我们将最终的`LayerNorm`和最后一个转换器块设置为可训练。其余的 11 个转换器块和嵌入层保持不可训练。

为了使最终的`LayerNorm`和最后一个转换器块可训练，我们将它们的`requires_grad`分别设置为`True`：

```py
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True
```

## 练习 6.2 微调整个模型

而不是仅微调最后的转换器块，微调整个模型并评估其对预测性能的影响。

尽管我们添加了一个新的输出层并将某些层标记为可训练或不可训练，我们仍然可以像以前一样使用这个模型。例如，我们可以给它一个与之前使用的示例文本相同的示例文本：

```py
inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)
print("Inputs:", inputs)
print("Inputs dimensions:", inputs.shape)    #1
```

> **#1** 形状：`(batch_size, num_tokens)`

打印输出显示，前面的代码将输入编码成一个包含四个输入标记的张量：

```py
Inputs: tensor([[5211,  345,  423,  640]])
Inputs dimensions: torch.Size([1, 4])
```

然后，我们可以像往常一样将编码后的标记 ID 传递给模型：

```py
with torch.no_grad():
    outputs = model(inputs)
print("Outputs:\n", outputs)
print("Outputs dimensions:", outputs.shape)
```

输出张量看起来如下：

```py
Outputs:
 tensor([[[-1.5854,  0.9904],
          [-3.7235,  7.4548],
          [-2.2661,  6.6049],
          [-3.5983,  3.9902]]])
Outputs dimensions: torch.Size([1, 4, 2])
```

相似的输入之前会产生一个`[1, 4, 50257]`的输出张量，其中`50257`代表词汇量。输出行的数量对应于输入标记的数量（在这种情况下，四个）。然而，每个输出的嵌入维度（列数）现在为 2，而不是 50,257，因为我们替换了模型的输出层。

记住，我们感兴趣的是微调这个模型以返回一个类别标签，指示模型输入是“垃圾邮件”还是“非垃圾邮件”。我们不需要微调所有四行输出；相反，我们可以专注于单个输出标记。特别是，我们将专注于最后一行，对应于最后一个输出标记，如图 6.11 所示。

![figure](img/6-11.png)

### 图 6.11 GPT 模型，具有四个标记示例输入和输出

由于修改后的输出层，输出张量由两列组成。在微调模型进行垃圾邮件分类时，我们只对最后一行对应最后一个标记感兴趣。

要从输出张量中提取最后一个输出标记，我们使用以下代码：

```py
print("Last output token:", outputs[:, -1, :])
```

这会打印：

```py
Last output token: tensor([[-3.5983,  3.9902]])
```

我们仍然需要将值转换为类别标签预测。但首先，让我们了解为什么我们特别关注最后一个输出标记。

我们已经探讨了注意力机制，它建立了每个输入标记与每个其他输入标记之间的关系，以及*因果注意力掩码*的概念，这在 GPT-like 模型中常用（见第三章）。这个掩码限制了标记的关注点为其当前位置及其之前的那些位置，确保每个标记只能受到自身和前面标记的影响，如图 6.12 所示。

![figure](img/6-12.png)

### 图 6.12 因果注意力机制

其中输入标记之间的注意力得分以矩阵格式显示。空单元格表示由于因果注意力掩码而屏蔽的位置，防止标记关注未来的标记。单元格中的值代表注意力得分；最后一个标记`time`是唯一一个为所有先前标记计算注意力得分的标记。

在图 6.12 中设置的因果注意力掩码下，序列中的最后一个标记积累了最多的信息，因为它是有权访问所有先前标记数据的唯一标记。因此，在我们的垃圾邮件分类任务中，我们在微调过程中关注这个最后一个标记。

我们现在准备好将最后一个标记转换为类别标签预测，并计算模型的初始预测准确率。随后，我们将微调模型以进行垃圾邮件分类任务。

## 练习 6.3 调整第一个与最后一个标记

尝试微调第一个输出标记。注意与微调最后一个输出标记相比，预测性能的变化。

## 6.6 计算分类损失和准确率

在我们微调模型之前，只剩下一个小的任务：我们必须实现微调期间使用的模型评估函数，如图 6.13 所示。

![图](img/6-13.png)

### 图 6.13 对 LLM 进行分类微调的三个阶段过程

我们已经完成了前六个步骤。我们现在准备进行第二阶段最后一步：实现评估模型在微调前后以及分类垃圾邮件性能的函数。

在实现评估工具之前，让我们简要讨论一下我们如何将模型输出转换为类别标签预测。我们之前通过将 50,257 个输出转换为通过`softmax`函数的概率，然后通过`argmax`函数返回最高概率的位置来计算 LLM 生成的下一个标记的标记 ID。我们在这里采取相同的方法来计算模型对于给定输入输出“垃圾邮件”或“非垃圾邮件”预测，如图 6.14 所示。唯一的区别是我们处理的是二维输出而不是 50,257 维输出。

![图](img/6-14.png)

### 图 6.14 对应于最后一个标记的模型输出被转换为每个输入文本的概率分数

通过查找最高概率分数的索引位置来获得类别标签。由于模型尚未经过训练，模型错误地预测了垃圾邮件标签。

让我们通过一个具体的例子来考虑最后一个标记的输出：

```py
print("Last output token:", outputs[:, -1, :])
```

对应于最后一个标记的张量值是：

```py
Last output token: tensor([[-3.5983,  3.9902]])
```

我们可以获取类别标签：

```py
probas = torch.softmax(outputs[:, -1, :], dim=-1)
label = torch.argmax(probas)
print("Class label:", label.item())
```

在这种情况下，代码返回`1`，意味着模型预测输入文本是“垃圾邮件。”在这里使用`softmax`函数是可选的，因为最大的输出直接对应于最高的概率分数。因此，我们可以简化代码而不使用 softmax：

```py
logits = outputs[:, -1, :]
label = torch.argmax(logits)
print("Class label:", label.item())
```

这个概念可以用来计算分类准确率，它衡量的是数据集中正确预测的百分比。

为了确定分类准确率，我们将基于`argmax`的预测代码应用于数据集中的所有示例，并通过定义一个`calc_accuracy_loader`函数来计算正确预测的比例。

### 列表 6.8 计算分类准确率

```py
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]     #1
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (
                (predicted_labels == target_batch).sum().item()
            )

        else:
            break
    return correct_predictions / num_examples
```

> **#1** 最后输出标记的对数

让我们使用该函数来确定跨各种数据集的分类准确率，这些数据集是从 10 批次中估计出来的，以提高效率：

```py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

torch.manual_seed(123)
train_accuracy = calc_accuracy_loader(
    train_loader, model, device, num_batches=10
)
val_accuracy = calc_accuracy_loader(
    val_loader, model, device, num_batches=10
)
test_accuracy = calc_accuracy_loader(
    test_loader, model, device, num_batches=10
)

print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")
```

通过`device`设置，如果可用带有 Nvidia CUDA 支持的 GPU，则模型自动在 GPU 上运行，否则在 CPU 上运行。输出是：

```py
Training accuracy: 46.25%
Validation accuracy: 45.00%
Test accuracy: 48.75%
```

如我们所见，预测准确率接近随机预测，在这种情况下为 50%。为了提高预测准确率，我们需要微调模型。

然而，在我们开始微调模型之前，我们必须定义我们将在训练中优化的损失函数。我们的目标是最大化模型的垃圾邮件分类准确性，这意味着前面的代码应该输出正确的类别标签：`0`表示非垃圾邮件，`1`表示垃圾邮件。

由于分类准确性不是一个可微分的函数，我们使用交叉熵损失作为最大化准确性的代理。因此，`calc_loss_batch`函数保持不变，但有一个调整：我们只优化最后一个标记，`model(input_batch)[:, -1, :]`，而不是所有标记，`model(input_batch)`：

```py
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]     #1
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss
```

> **#1** 最后输出标记的 logits

我们使用`calc_loss_batch`函数计算从先前定义的数据加载器中获得的单个批次的损失。为了计算数据加载器中所有批次的损失，我们像以前一样定义`calc_loss_loader`函数。

### 列表 6.9 计算分类损失

```py
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:                                        #1
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches
```

> **#1** 确保批次数不超过数据加载器中的批次数

与计算训练准确性类似，我们现在计算每个数据集的初始损失：

```py
with torch.no_grad():                 #1
    train_loss = calc_loss_loader(
        train_loader, model, device, num_batches=5
    )
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)
print(f"Training loss: {train_loss:.3f}")
print(f"Validation loss: {val_loss:.3f}")
print(f"Test loss: {test_loss:.3f}")
```

> **#1** 为了效率，禁用梯度跟踪，因为我们还没有开始训练

初始损失值是：

```py
Training loss: 2.453
Validation loss: 2.583
Test loss: 2.322
```

接下来，我们将实现一个训练函数来微调模型，这意味着调整模型以最小化训练集损失。最小化训练集损失将有助于提高分类准确性，这是我们整体目标。

## 6.7 在监督数据上微调模型

我们必须定义并使用训练函数来微调预训练的 LLM 并提高其垃圾邮件分类的准确性。如图 6.15 所示的训练循环，与我们用于预训练的整体训练循环相同；唯一的区别是我们计算分类准确性而不是生成样本文本来评估模型。

![figure](img/6-15.png)

### 图 6.15 PyTorch 中训练深度神经网络的典型训练循环

包括几个步骤，迭代训练集中的批次几个时期。在每个循环中，我们计算每个训练集批次的损失以确定损失梯度，我们使用这些梯度来更新模型权重以最小化训练集损失。

实现图 6.15 中所示概念的训练函数也与用于预训练模型的`train_model_simple`函数非常相似。唯一的两个区别是，我们现在跟踪看到的训练示例数量（`examples_seen`）而不是标记数量，并在每个时期后计算准确性而不是打印样本文本。

### 列表 6.10 微调模型以分类垃圾邮件
