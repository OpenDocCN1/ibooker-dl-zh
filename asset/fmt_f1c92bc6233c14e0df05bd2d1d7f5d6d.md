
```py
for i in range(50):                                    ①
    tloss=0
    for imgs,labels in binary_train_loader:            ②
        imgs=imgs.reshape(-1,28*28)                    ③
        imgs=imgs.to(device)
        labels=torch.FloatTensor(\
          [x if x==0 else 1 for x in labels])          ④
        labels=labels.reshape(-1,1).to(device)
        preds=binary_model(imgs)    
        loss=loss_fn(preds,labels)                     ⑤
        optimizer.zero_grad()
        loss.backward()                                ⑥
        optimizer.step()
        tloss+=loss.detach()
    tloss=tloss/n
    print(f"at epoch {i}, loss is {tloss}")
```

① 训练 50 个 epoch

② 遍历所有批次

③ 在将张量移动到 GPU 之前将图像展平

④ 将标签转换为 0 和 1

⑤ 计算损失

⑥ 反向传播

在 PyTorch 中训练深度学习模型时，`loss.backward()`计算损失相对于每个模型参数的梯度，从而实现反向传播，而`optimizer.step()`根据这些计算的梯度更新模型参数以最小化损失。为了简单起见，我们训练模型 50 个 epoch（一个 epoch 是指使用训练数据训练模型一次）。在下一节中，你将使用验证集和提前停止类来确定训练的 epoch 数。在二分类中，我们将目标标签标记为 0 和 1。由于我们只保留了标签为 0 和 9 的 T 恤和踝靴，我们在列表 2.6 中将它们转换为 0 和 1。因此，两种服装类别的标签分别为 0 和 1。

如果你使用 GPU 进行训练，这个过程可能需要几分钟。如果你使用 CPU 进行训练，则所需时间更长，但训练时间应该不到一小时。

### 2.3.3 测试二分类模型

训练好的二分类模型的预测结果是一个介于 0 和 1 之间的数字。我们将使用`torch.where()`方法将预测结果转换为 0 和 1：如果预测概率小于 0.5，我们将预测结果标记为 0；否则，我们将预测结果标记为 1。然后我们将这些预测结果与实际标签进行比较，以计算预测的准确率。在下面的列表中，我们使用训练好的模型对测试数据集进行预测。

列表 2.7 计算预测的准确率

```py
import numpy as np
results=[]
for imgs,labels in binary_test_loader:                     ①
    imgs=imgs.reshape(-1,28*28).to(device)
    labels=(labels/9).reshape(-1,1).to(device)
    preds=binary_model(imgs)
    pred10=torch.where(preds>0.5,1,0)                      ②
    correct=(pred10==labels)                               ③
    results.append(correct.detach().cpu()\
      .numpy().mean())                                     ④
accuracy=np.array(results).mean()                          ⑤
print(f"the accuracy of the predictions is {accuracy}")
```

① 遍历测试集中的所有批次

② 使用训练好的模型进行预测

③ 将预测与标签进行比较

④ 在批次中计算准确率

⑤ 在测试集中计算准确率

我们遍历测试集中所有批次的数据。训练好的模型会输出一个概率，表示图像是否为踝靴。然后我们根据 0.5 的截止值，使用`torch.where()`方法将概率转换为 0 或 1。转换后，预测结果要么是 0（即 T 恤），要么是 1（踝靴）。我们将预测结果与实际标签进行比较，看模型正确预测的次数。结果显示，在测试集中预测的准确率为 87.84%。

## 2.4 多分类分类

在本节中，我们将使用 PyTorch 构建一个深度神经网络来将服装项目分类到 10 个类别之一。然后我们将使用 Fashion MNIST 数据集训练模型。最后，我们将使用训练好的模型进行预测，并查看其准确率。我们首先创建一个验证集并定义一个提前停止类，以便我们可以确定何时停止训练。

### 2.4.1 验证集和提前停止

当我们构建和训练一个深度神经网络时，有许多超参数可以选择（例如学习率和训练的 epoch 数）。这些超参数影响模型的性能。为了找到最佳超参数，我们可以创建一个验证集来测试不同超参数下模型的性能。

为了给您一个例子，我们将在多类别分类中创建一个验证集，以确定训练的最佳 epoch 数。我们之所以在验证集而不是在训练集中这样做，是为了避免过拟合，即模型在训练集中表现良好，但在样本外测试（即未见过的数据）中表现不佳。

在这里，我们将训练数据集的 60,000 个观测值分为一个训练集和一个验证集：

```py
train_set,val_set=torch.utils.data.random_split(\
    train_set,[50000,10000])
```

原始训练集现在变成了两个集合：包含 50,000 个观测值的新训练集和包含剩余 10,000 个观测值的验证集。

我们使用 PyTorch *utils*包中的`DataLoader`类将训练集、验证集和测试集转换为三个数据迭代器，以批量形式：

```py
train_loader=torch.utils.data.DataLoader(
    train_set,    
    batch_size=batch_size,   
    shuffle=True)   
val_loader=torch.utils.data.DataLoader(
    val_set,    
    batch_size=batch_size,   
    shuffle=True)
test_loader=torch.utils.data.DataLoader(
    test_set,    
    batch_size=batch_size,   
    shuffle=True)
```

接下来，我们定义一个`EarlyStop()`类并创建该类的实例。

列表 2.8 `EarlyStop()`类用于确定何时停止训练

```py
class EarlyStop:
    def __init__(self, patience=10):         ①
        self.patience = patience
        self.steps = 0
        self.min_loss = float('inf')
    def stop(self, val_loss):                ②
        if val_loss < self.min_loss:         ③
            self.min_loss = val_loss
            self.steps = 0
        elif val_loss >= self.min_loss:      ④
            self.steps += 1
        if self.steps >= self.patience:
            return True
        else:
            return False
stopper=EarlyStop()
```

① 将 patience 的默认值设置为 10

② 定义了 stop()方法

③ 如果达到新的最小损失，则更新 min_loss 的值

④ 计算自上次最小损失以来的 epoch 数

`EarlyStop()`类用于确定验证集中的损失在最后`patience=10`个 epoch 中是否停止了改进。我们将`patience`参数的默认值设置为 10，但您在实例化类时可以选择不同的值。`patience`的值衡量了自上次模型达到最小损失以来您希望训练的 epoch 数。`stop()`方法记录最小损失和自最小损失以来的 epoch 数，并将该数字与`patience`的值进行比较。如果自最小损失以来的 epoch 数大于`patience`的值，则该方法返回`True`。

### 2.4.2 构建和训练一个多类别分类模型

Fashion MNIST 数据集包含 10 种不同的服装类别。因此，我们创建一个多类别分类模型来对它们进行分类。接下来，您将学习如何创建这样的模型并对其进行训练。您还将学习如何使用训练好的模型进行预测并评估预测的准确性。在下面的列表中，我们使用 PyTorch 创建多类别分类的神经网络。

列表 2.9 创建一个多类别分类模型

```py
model=nn.Sequential(
    nn.Linear(28*28,256),
    nn.ReLU(),
    nn.Linear(256,128),
    nn.ReLU(),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64,10)                         ①
    ).to(device)                             ②
```

① 输出层有 10 个神经元。

② 输出层不应用 softmax 激活

与上一节中我们创建的二分类模型相比，这里做了一些修改。首先，输出现在有 10 个值，代表数据集中 10 种不同的服装类型。其次，我们将最后一隐藏层的神经元数量从 32 改为 64。创建深度神经网络的一个经验法则是从一层到下一层逐渐增加或减少神经元数量。由于输出神经元的数量从二分类中的 1 增加到多类别分类中的 10，我们将第二到最后一层的神经元数量从 32 改为 64 以匹配增加。然而，64 这个数字并没有什么特殊之处：如果你在第二到最后一层使用，比如说，100 个神经元，你将得到相似的结果。

我们将使用 PyTorch 的`nn.CrossEntropyLoss()`类作为我们的损失函数，该函数将`nn.LogSoftmax()`和`nn.NLLLoss()`合并到一个单独的类中。有关详细信息，请参阅此处文档：[`mng.bz/pxd2`](https://mng.bz/pxd2)。特别是，文档中提到，“此标准计算输入 logits 和目标之间的交叉熵损失。”这解释了为什么我们在前面的列表中没有应用 softmax 激活。在本书的 GitHub 仓库中，我演示了如果我们使用`nn.LogSoftmax()`在模型中，并使用`nn.NLLLoss()`作为损失函数，我们将获得相同的结果。

因此，`nn.CrossEntropyLoss()`类将在对输出应用 softmax 激活函数之前，将 10 个数字压缩到[0, 1]的范围内进行对数运算。在二分类中，输出上首选的激活函数是 sigmoid，而在多类别分类中是 softmax。此外，softmax 激活后的 10 个数字加起来等于 1，这可以解释为对应于 10 种不同服装项目的概率。我们将使用与上一节二分类中相同的学习率和优化器。

```py
lr=0.001
optimizer=torch.optim.Adam(model.parameters(),lr=lr)
loss_fn=nn.CrossEntropyLoss()
```

我们定义`train_epoch()`如下：

```py
def train_epoch():
    tloss=0
    for n,(imgs,labels) in enumerate(train_loader):    
        imgs=imgs.reshape(-1,28*28).to(device)
        labels=labels.reshape(-1,).to(device)
        preds=model(imgs)    
        loss=loss_fn(preds,labels)
        optimizer.zero_grad()
        loss.backward()    
        optimizer.step()
        tloss+=loss.detach()
    return tloss/n
```

该函数用于训练模型一个 epoch。代码与我们之前在二分类中看到的是相似的，只是标签从 0 到 9，而不是两个数字（0 和 1）。

我们还定义了一个`val_epoch()`函数：

```py
def val_epoch():
    vloss=0
    for n,(imgs,labels) in enumerate(val_loader):    
        imgs=imgs.reshape(-1,28*28).to(device)
        labels=labels.reshape(-1,).to(device)
        preds=model(imgs)    
        loss=loss_fn(preds,labels)    
        vloss+=loss.detach()
    return vloss/n
```

该函数使用模型对验证集中的图像进行预测，并计算数据批次平均损失。

我们现在训练多类别分类器：

```py
for i in range(1,101):    
    tloss=train_epoch()
    vloss=val_epoch()
    print(f"at epoch {i}, tloss is {tloss}, vloss is {vloss}")
    if stopper.stop(vloss)==True:             
        break  
```

我们最多训练 100 个 epoch。在每个 epoch 中，我们首先使用训练集来训练模型。然后我们计算验证集中每个批次的平均损失。我们使用`EarlyStop()`类通过查看验证集中的损失来确定是否应该停止训练。如果损失在最后 10 个 epoch 中没有改善，则停止训练。经过 19 个 epoch 后，训练停止。

如果你使用 GPU 进行训练，训练大约需要 5 分钟，这比二分类中的训练过程要长，因为我们现在训练集中的观察结果更多（10 种服装而不是只有 2 种）。

模型的输出是一个包含 10 个数字的向量。我们使用`torch.argmax()`根据最高概率为每个观察结果分配一个标签。然后，我们将预测标签与实际标签进行比较。为了说明预测是如何工作的，让我们看看测试集中前五张图像的预测结果。

列表 2.10 在五张图像上测试训练好的模型

```py
plt.figure(dpi=300,figsize=(5,1))
for i in range(5):                                          ①
    ax=plt.subplot(1,5, i + 1)
    img=test_set[i][0]    
    label=test_set[i][1]
    img=img/2+0.5    
    img=img.reshape(28, 28)    
    plt.imshow(img, cmap="binary")
    plt.axis('off')
    plt.title(text_labels[label]+f"; {label}", fontsize=8)
plt.show()
for i in range(5):
    img,label = test_set[i]                                 ②
    img=img.reshape(-1,28*28).to(device)
    pred=model(img)                                         ③
    index_pred=torch.argmax(pred,dim=1)                     ④
    idx=index_pred.item()
    print(f"the label is {label}; the prediction is {idx}") ⑤ 
```

① 在测试集中绘制前五张图像及其标签

② 获取测试集中的第 i 张图像和标签

③ 使用训练好的模型进行预测

④ 使用 torch.argmax()方法获取预测标签

⑤ 打印出实际标签和预测标签

我们以 1 × 5 的网格形式绘制测试集中的前五种服装。然后，我们使用训练好的模型对每种服装进行预测。预测结果是一个包含 10 个值的张量。`torch.argmax()`方法返回张量中最高概率的位置，我们将其用作预测标签。最后，我们打印出实际标签和预测标签以进行比较，看看预测是否正确。运行前面的代码列表后，你应该会看到图 2.3 中的图像。

![](img/CH02_F03_Liu.png)

图 2.3 测试数据集中的前五种服装及其相应的标签。每种服装都有一个文本标签和一个介于 0 到 9 之间的数字标签。

图 2.3 显示了测试集中的前五种服装分别是踝靴、套头衫、裤子、裤子和外套，分别对应数字标签 9、2、1、1 和 6。

运行列表 2.10 中的代码后的输出如下：
