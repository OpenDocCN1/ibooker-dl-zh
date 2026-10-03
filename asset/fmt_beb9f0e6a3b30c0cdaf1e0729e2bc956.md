
```py
def train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []   #1
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):    #2
        model.train()             #3

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()                      #4
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()                          #5
            optimizer.step()                          #6
            examples_seen += input_batch.shape[0]    #7
            global_step += 1

 #8
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                )

 #9
        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )

        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen
```

#1 初始化跟踪损失和已看到示例的列表

#2 主要训练循环

#3 将模型设置为训练模式

#4 重置上一批迭代中的损失梯度

#5 计算损失梯度

#6 使用损失梯度更新模型权重

#7 新增：跟踪示例而不是标记

#8 可选评估步骤

#9 计算每个轮数后的准确率

`evaluate_model`函数与我们用于预训练的函数相同：

```py
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss
```

接下来，我们初始化优化器，设置训练轮数，并使用`train_classifier_simple`函数开始训练。在 M3 MacBook Air 笔记本电脑上训练大约需要 6 分钟，在 V100 或 A100 GPU 上不到半分钟：

```py
import time

start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 5

train_losses, val_losses, train_accs, val_accs, examples_seen = \
    train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50,
        eval_iter=5
    )

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")
```

训练过程中我们看到的输出如下：

```py
Ep 1 (Step 000000): Train loss 2.153, Val loss 2.392
Ep 1 (Step 000050): Train loss 0.617, Val loss 0.637
Ep 1 (Step 000100): Train loss 0.523, Val loss 0.557
Training accuracy: 70.00% | Validation accuracy: 72.50%
Ep 2 (Step 000150): Train loss 0.561, Val loss 0.489
Ep 2 (Step 000200): Train loss 0.419, Val loss 0.397
Ep 2 (Step 000250): Train loss 0.409, Val loss 0.353
Training accuracy: 82.50% | Validation accuracy: 85.00%
Ep 3 (Step 000300): Train loss 0.333, Val loss 0.320
Ep 3 (Step 000350): Train loss 0.340, Val loss 0.306
Training accuracy: 90.00% | Validation accuracy: 90.00%
Ep 4 (Step 000400): Train loss 0.136, Val loss 0.200
Ep 4 (Step 000450): Train loss 0.153, Val loss 0.132
Ep 4 (Step 000500): Train loss 0.222, Val loss 0.137
Training accuracy: 100.00% | Validation accuracy: 97.50%
Ep 5 (Step 000550): Train loss 0.207, Val loss 0.143
Ep 5 (Step 000600): Train loss 0.083, Val loss 0.074
Training accuracy: 100.00% | Validation accuracy: 97.50%
Training completed in 5.65 minutes.
```

我们随后使用 Matplotlib 绘制训练和验证集的损失函数。

##### 列表 6.11 绘制分类损失

```py
import matplotlib.pyplot as plt

def plot_values(
        epochs_seen, examples_seen, train_values, val_values,
        label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

 #1
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(
        epochs_seen, val_values, linestyle="-.",
        label=f"Validation {label}"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

 #2
    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)    #3
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()             #4
    plt.savefig(f"{label}-plot.pdf")
    plt.show()

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)
```

#1 将训练和验证损失与轮数对比绘图

#2 为示例创建第二个 x 轴

#3 隐藏的绘图用于对齐刻度

#4 调整布局以留出空间

图 6.16 绘制了结果损失曲线。

![figure](img/6-16.png)

##### 图 6.16 展示了模型在五个训练轮数中的训练和验证损失。训练损失（用实线表示）和验证损失（用虚线表示）在第一个轮数急剧下降，并逐渐稳定在第五个轮数。这种模式表明良好的学习进度，并表明模型从训练数据中学习，同时很好地泛化到未见过的验证数据。

如基于图 6.16 中急剧下降的斜率所示，模型正在很好地从训练数据中学习，几乎没有过拟合的迹象；也就是说，训练集和验证集损失之间没有明显的差距。

##### 选择训练轮数

之前，当我们开始训练时，我们将训练轮数设置为五次。训练轮数取决于数据集和任务的难度，没有通用的解决方案或建议，尽管通常五个轮数是一个良好的起点。如果在最初的几个轮数后模型出现过拟合（如损失图所示，见图 6.16），你可能需要减少轮数。相反，如果趋势线表明验证损失可以通过进一步训练而改善，你应该增加轮数。在这个具体案例中，五个轮数是一个合理的数字，因为没有出现早期过拟合的迹象，验证损失接近 0。

使用相同的`plot_values`函数，现在让我们绘制分类准确率：

```py
epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))

plot_values(
    epochs_tensor, examples_seen_tensor, train_accs, val_accs,
    label="accuracy"
)
```

图 6.17 显示了结果准确率。模型在第四和第五个轮数后实现了相对较高的训练和验证准确率。重要的是，我们之前在使用`train_classifier_simple`函数时设置了`eval_iter=5`，这意味着我们的训练和验证性能估计仅基于五个批次以提高训练效率。

![figure](img/6-17.png)

##### 图 6.17 训练准确率（实线）和验证准确率（虚线）在早期 epoch 中大幅增加，然后达到平台期，几乎完美的准确率得分为 1.0。两条线在整个 epoch 中的接近表明模型并没有过度拟合训练数据。

现在，我们必须通过运行以下代码，在整个数据集上计算训练、验证和测试集的性能指标，这次不定义`eval_iter`值：

```py
train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)

print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")
```

结果准确率值是

```py
Training accuracy: 97.21%
Validation accuracy: 97.32%
Test accuracy: 95.67%
```

训练集和测试集的性能几乎相同。训练集和测试集准确率之间的微小差异表明训练数据过拟合最小。通常，验证集准确率略高于测试集准确率，因为模型开发通常涉及调整超参数以在验证集上表现良好，这可能不会像在测试集上那样有效地推广。这种情况很常见，但通过调整模型的设置，如增加 dropout 率（`drop_rate`）或优化器配置中的`weight_decay`参数，可以最大限度地减少差距。

## 6.8 使用 LLM 作为垃圾邮件分类器

在微调和评估了模型之后，我们现在准备好对垃圾邮件进行分类（见图 6.18）。让我们使用我们的基于 GPT 的微调垃圾邮件分类模型。下面的`classify_review`函数遵循与我们在之前实现的`SpamDataset`中使用的类似的数据预处理步骤。然后，在将文本处理成标记 ID 之后，该函数使用模型预测一个整数类标签，类似于我们在第 6.6 节中实现的方法，然后返回相应的类名。

![图像](img/6-18.png)

##### 图 6.18 对 LLM 进行分类微调的三阶段过程。步骤 10 是第三阶段的最后一步——使用微调后的模型对新的垃圾邮件消息进行分类。

##### 列表 6.12 使用模型对新的文本进行分类

```py
def classify_review(
        text, model, tokenizer, device, max_length=None,
        pad_token_id=50256):
    model.eval()

    input_ids = tokenizer.encode(text)          #1
    supported_context_length = model.pos_emb.weight.shape[0]

    input_ids = input_ids[:min(              #2
        max_length, supported_context_length
    )]

    input_ids += [pad_token_id] * (max_length - len(input_ids))    #3

    input_tensor = torch.tensor(
        input_ids, device=device
    ).unsqueeze(0)              #4

    with torch.no_grad():                                #5
        logits = model(input_tensor)[:, -1, :]     #6
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"     #7
```

#1 准备模型输入

#2 如果序列太长则截断

#3 将序列填充到最长序列

#4 添加批量维度

#5 模型推理不跟踪梯度

#6 最后输出标记的 logits

#7 返回分类结果

让我们在一个示例文本上尝试这个`classify_review`函数：
