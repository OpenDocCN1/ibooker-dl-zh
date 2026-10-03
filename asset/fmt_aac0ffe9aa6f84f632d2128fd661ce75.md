
# 构建简单 GAN 模型

## 定义 `simpleGAN` 函数

以下代码定义了一个简单的生成对抗网络（GAN）构建函数，返回生成器 `G` 和判别器 `D`：

```
def simpleGAN(latent_d, neurons, out_shape, sigmoidG=False, leak=0.2): 
""" 
This function will create a simple GAN for us to train. It will return 
➥ a tuple (G, D), holding the generator and discriminator network 
➥ respectively. 
latent_d: the number of latent variables we will use as input to the 
➥ generator G. 
neurons: how many hidden neurons to use in each hidden layer 
out_shape: the shape of the output of the discriminator D. This should 
➥ be the same shape as the real data. 
sigmoidG: true if the generator G should end with a sigmoid activation, 
➥ or False if it should just return unbounded activations 
"""

G = nn.Sequential( fcLayer(latent_d, neurons, leak), 
    fcLayer(neurons, neurons, leak), 
    fcLayer(neurons, neurons, leak), 
    nn.Linear(neurons, abs(np.prod(out_shape)) ), ❶ 
    View(out_shape)                               ❷ 
)

if sigmoidG:                                      ❸ 
G = nn.Sequential(G, 
    nn.Sigmoid()) 
D = nn.Sequential( nn.Flatten(), 
    fcLayer(abs(np.prod(out_shape)), neurons, leak), 
    fcLayer(neurons, neurons, leak), 
    fcLayer(neurons, neurons, leak), 
    nn.Linear(neurons, 1 )                        ❹ 
) 
return G, D
```

❶ `np.prod` 函数将形状中的每个值相乘，从而得到所需输出的总数。`abs` 函数消除了批维度中“–1”的影响。

❷ 将输出重塑为 D 期望的任何形状

❸ 有时我们希望或不希望 G 返回一个 sigmoid 值（即，[0,1]），所以我们将其放在一个条件语句中。

❹ D 有一个输出用于二分类问题。

使用此函数，我们可以通过调用 `simpleGAN` 函数快速定义新的 G 和 D 模型：

```
G, D = simpleGAN(latent_d, neurons, out_shape, sigmoidG=True)
```

## GAN 的起始配方

GAN 因其难以训练而臭名昭著。第一次尝试往往效果不佳，可能需要大量手动调整才能使其工作并产生像本章开头示例那样的清晰结果。网上有很多技巧可以帮助 GAN 训练良好，但其中一些仅适用于特定类型的 GAN。其他则更可靠，适用于多种架构。以下是我尝试构建任何 GAN 时的建议：

+   使用具有大泄漏值（如 *α* = 0.1 或 *α* = 0.2）的 LeakyReLU 激活函数。对于我们判别器没有消失梯度这一点尤为重要，因为训练 G 的梯度必须首先完全通过 D！较大的泄漏值有助于避免这个问题。

+   使用 LN 而不是批归一化（BN）。有些人发现他们用 BN 可以得到最好的结果。在其他情况下，我们可以 *证明* BN 会导致问题。因此，我更喜欢从 LN 开始，它在训练 GAN 时具有更一致和可靠的性能。如果我在努力获得最后一丝性能时，我会尝试仅在生成器中将 LN 替换为 BN。

+   使用特定于 Adam 优化器的学习率 *η* = 0.0001 和 *β*[1] = 0 以及 *β*[2] = 0.9。这比 Adam 的正常默认值慢，但更适合 GAN。这是我最后会尝试更改以获得更好结果的事情。

## 实现 GAN 训练循环

我们如何训练我们的两个网络？每个网络 G 和 D 都有自己的优化器，我们将轮流使用它们！因此，我们*不*使用我们在本书的大部分内容中使用的相同的`train_network`函数。这是学习框架及其提供的工具很重要的一个原因：并不是所有东西都可以轻易抽象出来，并且与*任何*你未来可能想要训练的神经网络一起工作。

首先，让我们做一些设置。让我们将我们的模型移动到 GPU 上，指定我们的二元交叉熵损失函数（因为真实与伪造是一个二元问题），并为每个网络设置两个不同的优化器：

```
G.to(device) 
D.to(device)

loss_func = nn.BCEWithLogitsLoss()             ❶ 

real_label = 1                                 ❷

fake_label = 0                                 ❷

optimizerD = torch.optim.AdamW(D.parameters(), ❸ 
➥ lr=0.0001, betas=(0.0, 0.9)) 
optimizerG = torch.optim.AdamW(G.parameters(), ❸
➥ lr=0.0001, betas=(0.0, 0.9))
```

❶ 初始化 BCEWithLogitsLoss 函数。BCE 损失用于二元分类问题，我们的问题就是这样（真实与伪造）。

❷ 在训练期间建立真实和伪造标签的约定

❸ 为 G 和 D 设置 Adam 优化器

接下来，我们抓取 MNIST 作为我们的数据集。在本章中，我们不会真正使用测试集，因为我们专注于生成新的数据：

```
train_data = torchvision.datasets.MNIST("./", train=True, 
➥ transform=transforms.ToTensor(), download=True) 
test_data = torchvision.datasets.MNIST("./", train=False, 
➥ transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, 
➥ drop_last=True) 
test_loader = DataLoader(test_data, batch_size=batch_size)
```

现在我们需要训练这个 GAN。我们将这个过程分为两个步骤，一个用于 D，一个用于 G，如图 9.6 所示。

![图片](img/CH09_F06_Raff.png)

图 9.6 GAN 的训练步骤。在几个 epoch 中，我们运行这两个步骤。在第 1 步中，我们在真实和伪造数据上计算判别器损失，然后使用其优化器更新 D 的权重。在第 2 步中，我们更新生成器。由于这两个步骤都需要 G 的输出，我们在两个步骤之间重用*G*(**z**)。

希望到这一点，你已经开始感到舒适地混合代码和数学符号。这张图几乎列出了我们成功实现生成对抗网络（GAN）所需的所有细节！让我们将其翻译成完整的代码。首先，我们有`for`循环和一些设置。我们使用两个数组来存储每个批次的损失，以便在训练后查看。损失函数对于 GAN 来说可以特别有信息量。我们还创建了*y*[真实]和*y*[伪造]，这两个变量在两个步骤中都会使用，并将数据移动到正确的设备上。以下代码展示了这一点：

```
G_losses = [] D_losses = []

for epoch in tqdm(range(num_epochs)): 
    for data, class_label in tqdm(train_loader, leave=False): 
        real_data = data.to(device)                     ❶ 
        y_real = torch.full((batch_size,1), real_label, ❶ 
        ➥ dtype=torch.float32, device=device) 
        y_fake = torch.full((batch_size,1), fake_label, ❶
        ➥ dtype=torch.float32, device=device)
```

❶ 准备批次并制作标签

现在我们可以进行第一步：更新判别器。实现这一点不需要做太多工作。我们计算完每个错误组件后调用 `backward`，作为一个轻微的效率优化。但我们仍然稍后把错误加在一起以保存组合错误。这里的一个大技巧是你应该注意到的，就是在将假图像传递给 D 时使用 `fake.detach()`。`detach()` 方法返回一个相同对象的新版本，该版本不会进一步传递梯度。我们这样做是因为 `fake` 对象是使用 `G` 计算的，所以天真地使用 `fake` 对象在我们的计算中会导致我们输入一个对 `G` 有利的梯度，因为我们正在计算判别器的损失！由于第一步只是应该改变 `D` 和此时的 `G` 梯度对 `G` 的目标（它想打败判别器！）是有害的，所以我们调用 `.detach()` 以确保 `G` 不获得任何梯度。

这里是代码：

```
D.zero_grad() 
errD_real = loss_func(D(real_data), y_real)          ❶
errD_real.backward()

z = torch.randn(batch_size, latent_d, device=device) ❷ 

fake = G(z)                                          ❹ 

errD_fake = loss_func(D(fake.detach()), y_fake)      ❺

errD_fake.backward()                                 ❻❸ 

errD = errD_real + errD_fake                         ❼❸

optimizerD.step()                                    ❽❸
```

❶ 真实数据

❷ 使用全假批次进行训练并生成一个潜在向量批次 *z* ∼ 𝒩(![](img/vec_0.png), 1)。

❸ 使用 G 生成一个假图像批次，并使用 D 对假批次进行分类。我们将此保存以在第二步中重用。

❹ **x**[fake] = *G*(**z**)

❺ 在全假批次上计算 D 的损失；注意使用 fake.detach()。ℓ(*D*(*x*[real]),*y*[real])

❻ 计算该批次的梯度

❼ 将全真实和全假批次的梯度相加

❽ 更新 D

我们循环体的最后一部分是第二步：更新生成器 G。我们重用 `fake` 对象，这样我们就不需要创建一个新的对象，这可以节省我们的时间。由于我们还想改变 G，所以我们直接使用原始的 `fake` 对象——没有调用 `.fake()`。这里需要的代码非常少，我们将 G 和 D 的错误追加到一个列表中，以便稍后绘制它们：

```
G.zero_grad() 
errG = loss_func(D(fake), y_real) ❶ 
errG.backward()                   ❷ 
optimizerG.step()                 ❸

G_losses.append(errG.item()) 
D_losses.append(errD.item())
```

❶ 基于此输出计算 G 的损失：ℓ(*D*(**x**[fake]), *y*[real])

❷ 为 G 计算梯度

❸ 更新 G

## 检查结果

运行该代码成功训练了一个 GAN。因为潜在的 z 来自高斯分布（**z** ∼ 𝒩(![](img/vec_0.png), **I** ) ），我们可以轻松地采样它们并计算 *G*(*z*) 来获取合成数据。我们还可以查看 *D*(*G*(*z*)) 的值，以了解判别器对每个样本真实性的看法。我们训练模型的方式是，值为 1 表示判别器认为输入肯定是真实的，而 0 表示判别器认为它肯定是假的。以下代码将一些新的潜在值采样到一个名为 `noise` 的变量中，这是你在 GAN 中将看到的另一个常见名称。我们用 G 生成假数字并计算它们看起来有多真实的分数：

```
with torch.no_grad(): 
    noise = torch.randn(batch_size, latent_d, device=device) ❶ 
    fake_digits = G(noise) 
    scores = torch.sigmoid(D(fake_digits)) 
    fake_digits = fake_digits.cpu() scores =
    scores.cpu().numpy().flatten()
```

❶ **z** ∼ 𝒩(![](img/vec_0.png), **I**)

接下来是一段 Matplotlib 代码，用于绘制所有生成的图像，每个数字上方都有红色的分数。该图会根据我们使用的批量大小自动调整大小。该代码快速计算可以从给定批次填充的最大正方形图像。我们将`scores`作为一个可选参数，因为我们的未来 GANs 可能不会有这种类型的分数：

```
def plot_gen_imgs(fake_digits, scores=None): 
    batch_size = fake_digits.size(0) 
    fake_digits = fake_digits.reshape(-1,
    ➥ fake_digits.size(-1), fake_digits.size(-1)) 
    i_max = int(round(np.sqrt(batch_size))) 
    j_max = int(np.floor(batch_size/float(i_max))) 
    f, axarr = plt.subplots(i_max,j_max, figsize=(10,10)) 
    for i in range(i_max): 
        for j in range(j_max): 
            indx = i*j_max+j 
            axarr[i,j].imshow(fake_digits[indx,:].numpy(), cmap=’gray’, 
            vmin=0, vmax=1) 
            axarr[i,j].set_axis_off() 
            if scores is not None:
                axarr[i,j].text(0.0, 0.5, str(round(scores[indx],2)), 
                ➥ dict(size=20, color=’red’)) 
plot_gen_imgs(fake_digits, scores)
```

❶ 此代码假设我们正在处理黑白图像。

![图片](img/CH09_UN04_Raff.png)

我们已经生成了合成数据，其中一些看起来相当不错！一些样本看起来不太真实，这是可以接受的。学习生成模型的问题本质上比学习判别模型更具挑战性。如果你再次运行此代码，你应该会得到新的结果并开始注意到一些问题。

你可能会注意到的第一个模式是生成器更喜欢一些数字而不是其他数字。当我运行这个时，我通常看到很多 0s、3s 和 8s 被生成。其他数字非常罕见。第二，判别器几乎总是正确地称生成的样本为伪造的。这两个问题相关。

几个数字的重复是一个问题，因为这意味着我们的生成器没有对整个分布进行建模。这使得其整体输出不太真实，即使单个数字看起来不错。判别器在检测伪造方面非常出色是一个潜在问题，因为我们使用了二元交叉熵（BCE）损失！由于 BCE 涉及 sigmoid，如果 D 变得非常好并预测样本是恶意的概率为 0%，这将导致 G 的梯度消失。这是因为 G 的梯度来自 D 预测的相反方向。如果 D 预测得非常完美，G 就无法学习。

D 可能会通过完美的预测赢得游戏的危险很重要，因为 D 和 G 之间的游戏是不公平的。我说的不公平是什么意思？让我们看看判别器和生成器在训练过程中的损失。我们可以快速绘制 G 和 D 的损失，并将它们进行比较，看看它们的进展情况：

```
plt.figure(figsize=(10,5)) 
plt.title("Generator and Discriminator Loss During Training") 
plt.plot(G_losses,label="G") 
plt.plot(D_losses,label="D") 
plt.xlabel("iterations") 
plt.ylabel("Loss") 
plt.legend() 
plt.show()
```

![图片](img/CH09_UN05_Raff.png)
