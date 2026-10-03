
我们使用`run_epoch`函数在将模型置于正确模式后执行训练步骤，该函数记录训练结果。然后，如果提供了`test_loader`，我们切换到`model.eval()`模式并进入`with torch.no_grad()`上下文，这样我们就不会以任何方式修改模型，并可以检查其在保留数据上的性能。我们分别使用前缀`"train"`和`"test"`来表示训练和测试运行的结果。

最后，我们有一个新的训练函数将结果转换为 pandas `DataFrame`，这将使我们以后更容易访问和查看它们：

```
return pd.DataFrame.from_dict(results)
```

使用这个新的改进代码，让我们在月亮数据集上重新训练我们的模型。由于准确率是我们真正关心的，我们导入 scikit-learn 中的准确度度量。让我们包括 F1 分数度量来展示代码如何同时处理两个不同的度量：

```
from sklearn.metrics import accuracy_score 
from sklearn.metrics import f1_score
```

我们还希望更好地评估并包括一个验证集。由于月亮数据是合成的，我们可以轻松创建一个新的验证数据集。与其像以前那样进行 200 个训练周期，让我们生成一个更大的训练集：

```
X_train, y_train = make_moons(n_samples=8000, noise=0.4) 
X_test, y_test = make_moons(n_samples=200, noise=0.4) 
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
➥ torch.tensor(y_train, dtype=torch.long)) 
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), 
➥ torch.tensor(y_test, dtype=torch.long)) 

training_loader = DataLoader(train_dataset, shuffle=True) 
testing_loader = DataLoader(test_dataset)
```

我们已经有了重新训练模型所需的一切。我们将使用`model.pt`作为保存模型结果的存储位置。我们所需做的只是声明一个新的`model`对象并调用我们的新`train_simple_network`函数：

```
model = nn.Sequential( 
    nn.Linear(2, 30), 
    nn.Tanh(), 
    nn.Linear(30, 30), 
    nn.Tanh(), 
    nn.Linear(30, 2),
) 
results_pd = train_simple_network(model, loss_func, training_loader,
➥ epochs=5, test_loader=testing_loader, checkpoint_file=’model.pt’,
➥ score_funcs={’Acc’:accuracy_score,’F1’: f1_score})
```

是时候看看一些结果了。首先，让我们看看我们是否可以加载我们的检查点`model`而不是使用我们已训练的模型。要加载一个`model`，我们首先需要定义一个*新*的`model`，它具有与原始模型相同的所有子模块，并且它们都需要相同的大小。这是必要的，以确保所有权重都匹配。如果我们保存了一个具有 30 个神经元的第二隐藏层的模型，我们需要有一个具有 30 个神经元的新的模型；否则，将会有太多或太少，并且会发生错误。

我们使用`torch.load`和`torch.save`函数的一个原因是因为它们提供的`map_location`参数。这为我们处理从数据到正确计算设备加载模型提供了便利。一旦我们加载了结果字典，我们就可以使用`load_state_dict`函数将原始模型的状态恢复到这个新对象中。然后我们可以将模型应用于数据，并看到我们得到了相同的结果：

```
model_new = nn.Sequential( 
nn.Linear(2, 30), 
nn.Tanh(), 
nn.Linear(30, 30), 
nn.Tanh(), 
nn.Linear(30, 2),
)

visualize2DSoftmax(X_test, y_test, model_new, title="Initial Model") plt.show()

checkpoint_dict = torch.load(’model.pt’, map_location=device)

model_new.load_state_dict(checkpoint_dict[’model_state_dict’])

visualize2DSoftmax(X_test, y_test, model_new, title="Loaded Model") plt.show()
```

![图片](img/CH02_UN14_Raff.png)

你可以很容易地看出，初始模型并不给出很好的预测，因为它的权重是随机值且未经过训练。如果你运行几次代码，你应该会看到许多略有不同但同样无用的结果。但当我们把之前的`model`状态加载到`model_new`中时，我们得到了我们预期的清晰结果。

注意：在这个例子中，我们只加载了模型的州因为只做了预测；除非你想继续训练，否则不需要优化器。如果你想继续训练，你需要在你的代码中添加一行`optimizer.load_state_dict(checkpoint['optimizer_state_dict'])`。

我们的新训练函数被编写为在每次周期结束后返回一个包含模型信息的 pandas `DataFrame`对象。这为我们提供了一些有价值的信息，我们可以轻松地可视化。例如，我们可以快速绘制训练和验证准确率作为完成周期的函数：

```
    sns.lineplot(x=’epoch’, y=’train Acc’, data=results_pd, label=’Train’) 
    sns.lineplot(x=’epoch’, y=’test Acc’, data=results_pd, label=’Test’)

[29]: <AxesSubplot:xlabel='epoch', ylabel='train Acc'>
```

![图片](img/CH02_UN16_Raff.png)

现在很容易看出，通过使用更多的数据，我们的模型在大约两个周期内达到了噪声训练数据的顶峰。提供了两个分数函数，所以让我们看看 F1 分数作为实际训练时间（秒）的函数。如果我们想要比较两个不同的模型学习速度的快慢，这将在未来更有用：

```
    sns.lineplot(x=’total time’, y=’train F1’, data=results_pd, label=’Train’) 
    sns.lineplot(x=’total time’, y=’test F1’, data=results_pd, label=’Test’)

[30]: <AxesSubplot:xlabel='total time', ylabel='train F1'>
```

![图片](img/CH02_UN17_Raff.png)

对于这个玩具数据集，F1 分数和准确率给出了非常相似的分数，因为这两个类别具有相似的行为，并且在大小上平衡。你应该注意到的更有趣的趋势是，训练准确率在上升后趋于稳定，但验证准确率随着每个训练周期的上下波动而更加崎岖。*这是正常的*。模型将开始过度拟合训练数据，这使得其性能看起来稳定，并在完成学习并开始记忆更具有挑战性的数据点时缓慢上升。由于验证数据是分开的，这些可能对新的数据有益或有害的小变化对模型来说是未知的，因此它无法调整以使它们始终正确。我们保留一个单独的验证或测试集是至关重要的，这样我们就可以看到模型在新的数据上实际表现的较少偏差的视图。

## 2.5 批量训练

如果你查看上一张图的 x 轴，当我们把 F1 分数作为训练时间的函数来绘制时，你可能注意到，只用*d* = 2 个特征，在仅仅 8,000 个数据点上训练一个模型几乎需要一分钟。鉴于这个漫长的训练时间，我们如何才能扩大到更大的数据集呢？

我们需要在数据*批次*上进行训练。数据批次只是数据的一个更大的组。假设我们有以下数据集，包含*N* = 4 个项目：

![图片](img/CH02_F07_Raff_EQ09.png)

我们当前的代码，在一个周期内，将执行四次更新：针对数据集中的每个项目。这就是为什么它被称为*随机梯度下降*（SGD）。*随机*这个词是行话，意思是“随机的”，但通常有一些潜在的目的或无形的手在驱动这种随机性。SGD 名称中的随机部分来自于我们只使用部分洗牌后的数据来计算梯度，而不是整个数据集。由于它是洗牌的，我们每次都会得到不同的结果。

如果我们将所有 N 个数据点通过模型，并在整个数据集上计算损失，即 ∇ Σ*[i]^N*[=1] ℓ(*f*(***x**[i]*), *y[i]*)，我们得到的是*真实*梯度。这也可以通过一次性处理所有数据而不是逐个处理数据点来使我们的训练更加*计算效率高*。因此，我们不是将形状为(*d*)的向量作为输入传递给`model` *f*(⋅)，而是传递一个形状为(*N*,*d*)的矩阵。PyTorch 模块默认为此情况设计；我们只需要一种方法来告诉 PyTorch 将我们的数据分组到更大的批次中。结果证明，`DataLoader`具有内置的此功能，通过可选的`batch_size`参数。如果没有指定值，则默认为`batch_size=1`。如果我们将其设置为`batch_size=len(train_dataset)`，我们执行真正的梯度下降：

```
training_loader = DataLoader(train_dataset, batch_size=len(train_dataset),
➥ shuffle=True) 
testing_loader = DataLoader(test_dataset, batch_size=len(test_dataset)) 
model_gd = nn.Sequential( 
    nn.Linear(2, 30), 
    nn.Tanh(), 
    nn.Linear(30, 30), 
    nn.Tanh(), 
    nn.Linear(30, 2),
) 
results_true_gd = train_simple_network(model_gd, loss_func, 
➥ training_loader, epochs=5, test_loader=testing_loader, 
➥ checkpoint_file=’model.pt’, 
➥ score_funcs={’Acc’:accuracy_score,’F1’: f1_score})
```

五个训练周期仅用了 0.536 秒。显然，一次性在*更多*数据上训练使我们能够从现代 GPU 提供的并行性中受益。但如果我们绘制准确率，我们会看到训练梯度下降(*B*=*N*)产生了一个不太准确的模型：

```
    sns.lineplot(x=’total time’, y=’test Acc’, data=results_pd,
    ➥ label=’SGD, B=1’) 
    sns.lineplot(x=’total time’, y=’test Acc’, data=results_true_gd,
    ➥ label=’GD, B=N’)

[32]: <AxesSubplot:xlabel='total time', ylabel='test Acc'>
```

![图片](img/CH02_UN19_Raff.png)

让我们通过一个玩具示例来解释为什么会发生这种情况。图 2.8 显示了一个我们正在优化的函数，如果我们使用梯度下降（它查看*所有*数据），我们采取的步骤将引导我们走向正确的方向。但是每一步都很昂贵，所以我们只能采取少数几步。这个例子显示我们总共进行了四次更新/步骤，对应于四个周期。

![图片](img/CH02_F09_Raff.png)

图 2.8 左边的图像显示了数据四个周期的梯度下降。这意味着它只能前进四步，但每一步都是正确的方向。在右边，SGD 通过仅查看一些数据在每个周期进行多次更新。这意味着每一步都是嘈杂的，并不总是指向正确的方向，但它通常是朝着有用的方向，因此 SGD 通常可以在更少的周期内朝着目标取得更多进展。

当我们使用 SGD 时，我们每个周期执行 N 次更新，因此我们得到更多更新或步骤，对于固定数量的周期。但是，由于使用每个更新仅一个数据点的*随机*或随机行为，我们采取的步骤是嘈杂的。它们并不总是指向正确的方向。更大的总步骤数最终使我们更接近答案，但代价是运行时间的增加，因为我们失去了同时处理所有数据的计算效率。

我们在实践中使用的解决方案是在这两个极端之间取得平衡。让我们选择一个足够大的批量大小，以便更有效地使用 GPU，但又要足够小，以便我们仍然能够在每个 epoch 中进行更多的更新。我们用 B 表示批量大小；对于大多数应用，你会发现*B* ∈ [32,256]是一个不错的选择。另一个好的经验法则是使批量大小尽可能大，以便可以放入 GPU 内存，并添加更多的训练 epoch，直到模型收敛。这需要做更多的工作，因为随着你开发你的网络并做出改变，你可以放入 GPU 的最大批量大小可能会改变。

注意：因为我们只使用验证数据来*评估*我们的模型表现，而不是更新模型的权重，所以用于验证数据的批量大小没有特定的权衡。我们只需增加批量大小到运行最快的程度即可。无论测试数据使用的批量大小如何，结果都将相同。在实践中，大多数人为了简单起见，会使用相同的批量大小进行训练和测试数据。

这里是代码：

```
training_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) 
model_sgd = nn.Sequential( 
    nn.Linear(2, 30), 
    nn.Tanh(), 
    nn.Linear(30, 30), 
    nn.Tanh(), 
    nn.Linear(30, 2),
)
results_batched = train_simple_network(model_sgd, loss_func, 
➥ training_loader, epochs=5, test_loader=testing_loader, 
➥ checkpoint_file=’model.pt’, 
➥ score_funcs={’Acc’:accuracy_score,’F1’: f1_score})
```

现在，如果我们把结果作为时间的函数来绘制，我们会看到绿色的线给出了两者之间的最佳结果。它只运行了 1.044 秒，并且几乎达到了相同的准确度。你会发现使用这样的数据批量几乎没有缺点，并且在现代深度学习中是首选的方法。

```
    sns.lineplot(x=’total time’, y=’test Acc’, data=results_pd, label=’SGD, B=1’) 
    sns.lineplot(x=’total time’, y=’test Acc’, data=results_true_gd,
    ➥ label=’GD, B=N’) 
    sns.lineplot(x=’total time’, y=’test Acc’, data=results_batched,
    ➥ label=’SGD, B=32’)

[35]: <AxesSubplot:xlabel='total time', ylabel='test Acc'>
```

![](img/CH02_UN20_Raff.png)

## 练习

在 Manning 在线平台 Inside Deep Learning Exercises 上分享和讨论你的解决方案([`liveproject.manning.com/project/945`](https://liveproject.manning.com/project/945))。一旦你提交了自己的答案，你将能够看到其他读者提交的解决方案，并看到哪些是作者认为最好的。

1.  数据的输入范围可以对神经网络产生重大影响。这适用于输入*和*输出，例如回归问题。尝试将 scikit-learn 的`StandardScaler`应用于本章开始时的玩具回归问题的目标 y，并在其上训练一个新的神经网络。改变输出的尺度是否有助于或损害模型的预测？

1.  曲线下方的面积（AUC）指标在 scikit-learn 中并不遵循标准模式，因为它需要`y_pred`是一个形状为(*N*)的向量，而不是形状为(*N*,2)的矩阵。为 AUC 编写一个包装函数，使其与我们的`train_simple_network`函数兼容。

1.  编写一个新的函数`resume_simple_network`，从磁盘加载`checkpoint_file`，恢复`optimizer`和`model`的状态，并继续训练到指定的总 epoch 数。所以如果模型在 20 个 epoch 后保存，而你指定了 30 个 epoch，它应该只进行 10 个额外的 epoch 训练。

1.  在进行实验时，我们可能希望回到之前的不同时代的模型版本，特别是如果我们试图确定某些奇怪行为开始出现的时间。修改`train_simple_network`函数，添加一个新参数`checkpoint_every_x`，该参数在每个`x`个时期保存一个不同文件名的模型版本。这样，你可以回退并加载特定版本，而不会用每个时期的模型填满你的硬盘驱动器。

1.  深度学习的“深度”部分指的是神经网络中的层数。尝试向用于`make_moons`分类问题的模型中添加更多层（最多 20 层）。更多的层如何影响性能？

1.  尝试更改`make_moons`分类问题中隐藏层中使用的神经元数量。它如何影响性能？
