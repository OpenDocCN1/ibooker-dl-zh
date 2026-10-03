
D 和 G 是非对称的。我们可以使 D 的网络比 G 小，以帮助 G 在竞争中取得优势。即使在 WGAN-GP 中的梯度惩罚下，D 仍然有一个更容易的学习任务。我发现的一个有用的经验法则是 D 的层数是 G 的三分之二。然后在一个训练运行中绘制 D 和 G 的损失，以查看 D 是否陷入困境（损失没有下降），此时应该变得更大，或者 G 是否陷入困境（损失没有增加），此时也许 D 应该缩小或 G 应该变得更大。

使用`nn.AvgPool2d`而不是最大池化切换到平均池化。这有助于最大化梯度流动，因为每个像素都对答案有相同的贡献，所以所有像素都得到梯度的一部分。

使用`AdaptiveAvgPool`或`AdaptiveMaxPool`函数结束你的网络，这最容易做到。这可以帮助 D 不依赖于值的精确位置，而我们所真正关心的是内容的整体外观。

以下代码将这些想法纳入 D 的定义中。再次，我们使用 LN，因此我们需要跟踪高度和宽度，从全尺寸开始然后缩小。我们最终使用 4 × 4 作为自适应池化的大小，因此下一个线性层有 4 × 4 = 16 个值每个通道作为输入。如果我们对 64 × 64 或 256 × 256 的图像做这件事，我可能会将自适应池化提升到 7 × 7 或 9 × 9：

```
D = nn.Sequential( 
    cnnLayer(1, n_filters, k_size, 28, leak), 
    cnnLayer(n_filters, n_filters, k_size, 28, leak), 
    nn.AvgPool2d(2),         ❶
    cnnLayer(n_filters, n_filters, k_size, 28//2, leak), 
    cnnLayer(n_filters, n_filters, k_size, 28//2, leak), 
    nn.AvgPool2d(2), 
    cnnLayer(n_filters, n_filters, 3, 28//4, leak), 
    cnnLayer(n_filters, n_filters, 3, 28//4, leak), 
    nn.AdaptiveAvgPool2d(4), ❷ 
    nn.Flatten(), 
    nn.Linear(n_filters*4**2,256), 
    nn.LeakyReLU(leak), 
    nn.Linear(256,1), 
)
```

❶ 为了避免稀疏梯度，我们使用平均池化而不是最大池化。

❷ 这是一种自适应池化，因此我们知道在这个点上大小是 4 × 4，以便在池化时更加激进（通常对卷积 GANs 有帮助）并且使编码更容易。

## 训练和检查我们的卷积 GAN

通过将`View`逻辑移动到网络而不是训练代码中，我们可以为我们的 CNN GANs 重复使用`train_wgan`函数。以下代码训练它们：

```
D_losses, G_losses = train_wgan(D, G, train_loader, latent_d_conv, 
➥ epochs=15, device=device)

G = G.eval() 
D = D.eval()
```

十个时期对于训练一个 GAN 来说不是很多时间，但接下来我们可视化了一些 CNN GANs 的随机样本，数字看起来比之前好得多。再次，卷积方法在图像上工作得更好并不令人惊讶，但我们确实需要学习一些额外的技巧来使这工作得很好：

```
with torch.no_grad(): 
    noise = torch.randn(batch_size, latent_d_conv, device=device) 
    fake_digits = G(noise) 
    scores = D(fake_digits)

    fake_digits = fake_digits.cpu() scores =
    scores.cpu().numpy().flatten() 
plot_gen_imgs(fake_digits)
```

![图片](img/CH09_UN13_Raff.png)

更多的训练轮数会改善我们的 WGAN-GP 吗？以下代码再次绘制了每个批次中 G 和 D 损失的平滑版本。有一个明显的趋势，G 的得分在增加，这表明 G 在生成方面变得更好；但 D 的得分却保持平稳（没有变好或变坏）。更多的训练可能会改善这一点，但这并不保证。如果 D 的得分在下降（因为 D 想要负值），使得图表呈漏斗状（前 300 次迭代就有这种形状），那就更好了。这意味着 G 和 D 都在改进，我们更有信心更多的训练轮数会改善 GAN 的结果：

```
plt.figure(figsize=(10,5)) 
plt.title("Conv-WGAN Generator and Discriminator Loss") 
plt.plot(np.convolve(G_losses, np.ones((100,))/100, mode=’valid’) ,label="G") 
plt.plot(np.convolve(D_losses, np.ones((100,))/100, mode=’valid’) ,label="D") 
plt.xlabel("iterations") 
plt.ylabel("Loss") 
plt.legend() 
plt.show()
```

![图片](img/CH09_UN14_Raff.png)

## 9.5 条件 GAN

另一种帮助改善 GAN 训练的方法是创建一个*条件 GAN*。条件 GAN 是监督学习而不是无监督学习，因为我们使用了属于每个数据点 x 的标签 y。如图 9.7 所示，这是一个相当简单的改变。

![图片](img/CH09_F07_Raff.png)

图 9.7 展示了条件生成对抗网络（GAN）的例子，其中 D（判别器）出现了错误，而 G（生成器）表现良好。唯一的改变是我们将数据的标签 y 作为输入同时提供给 G 和 D。G 通过获取标签来知道它需要创建什么，而 D 通过获取标签来知道它在寻找什么。

我们不是要求模型从 x 预测 y，而是告诉 G 生成 y 的一个示例。你可以将正常的 GAN 视为要求模型生成任何真实数据，而条件 GAN 则要求模型生成我们可以将其分类为 y 的真实数据。这通过提供额外信息帮助模型。它不需要自己确定有多少个类别，因为通过提供标签 y，你告诉模型 G 要生成什么类型的数据。为了实现这一点，我们还向判别器 D 提供了关于 y 的信息。

另一种思考方式是，条件 GAN 允许我们学习一种*一对一*映射。我们之前所有的神经网络都是*一对一*的。对于任何输入 x，都有一个正确的输出 y。但如图 9.8 所示，条件模型让我们为任何有效的输入创建多个有效的输出。

![图片](img/CH09_F08_Raff.png)

图 9.8 展示了条件 GAN 作为一对一映射的例子。输入是“猫”，右侧的输出显示了多个有效的输出。为了获得这种多样化的输出，潜在变量 z 为 G 提供了创建多个输出的方法。

使用条件 GAN 进行一对一映射也允许我们开始操纵 G 生成的结果。给定一个单个的潜在向量 z，我们可以要求模型使用*G*(**z**∣*y*=1)生成一个 1，或者使用*G*(**z**∣*y*=3)生成一个 3。如果我们使用相同的 z 值做这两件事，生成的 1 和 3 将具有相似的特征，如线条粗细和倾斜。

### 9.5.1 实现条件 GAN

我们可以不费吹灰之力实现条件 GAN。要做到这一点，我们首先需要修改 G 和 D 以接受两个输入：潜在变量 z 和标签 y。让我们定义一个`ConditionalWrapper`类，它为我们完成这项工作。我们的方法是`ConditionalWrapper`将接受一个正常的 G 或 D 网络，并将其用作子网络，类似于我们实现 U-Net 的方式。`ConditionalWrapper`接受 z 和 y，并将它们组合成一个新的潜在值ẑ。然后我们将新的潜在值ẑ传递给原始网络（G 或 D）。

以下`Module`定义实现了这个想法。在构造函数中，我们创建了一个`nn.Embedding`层，将标签 y 转换为与 z 相同大小的向量。我们还创建了一个`combiner`网络，它接受一个大小是潜在向量 z 两倍的输入，并返回一个与 z 相同大小的输出。这个网络故意很小，没有超过两个隐藏层，所以它刚好足够组合我们的两个输入值 z 和 y。

`forward`函数随后可以以非常少的步骤进行操作。它将 y 嵌入到一个向量中。由于 y 的嵌入和 z 具有相同的形状，我们可以将它们连接在一起以形成一个双倍大小的输入。这个输入进入`combiner`，然后直接进入原始`net`（G 或 D）。

下面是代码：

```
class ConditionalWrapper(nn.Module): 
    def __init__(self, input_shape, neurons, classes, main_network, 
    ➥ leak=0.2): 
        """ 
        input_shape: the shape that the latent variable z 
        ➥ should take. 
        neurons: neurons to use in hidden layers 
        classes: number of classes in labels y 
        main_network: either the generator G or discriminator D 
        """ 
        super().__init__() self.input_shape = input_shape 
        self.classes = classes 
        input_size = abs(np.prod(input_shape))                   ❶ 

        self.label_embedding = nn.Embedding(classes, input_size) ❷

        self.combiner = nn.Sequential(                           ❸ 
            nn.Flatten(), 

            fcLayer(input_size*2, input_size, leak=leak),        ❹ 
            nn.Linear(input_size, input_size),                   ❺ 
            nn.LeakyReLU(leak), 

            View(input_shape),                                   ❻
            nn.LayerNorm(input_shape[1:]), 
        ) 
            self.net = main_network

    def forward(self, x, condition=None):                        ❼ 
        if condition is None:                                    ❽
            condition = torch.randint(0, self.classes, size=(x.size(0),), 
            ➥ device=x.get_device())

        embd = self.label_embedding(condition)                   ❾ 
        embd = embd.view(self.input_shape)                       ❿ 
        x = x.view(self.input_shape) 
        x_comb = torch.cat([x, embd], dim=1)                     ⓫ 
        return self.net(self.combiner(x_comb))                   ⓬
```

❶ 确定潜在形状的潜在参数数量

❷ 创建一个嵌入层以将标签转换为向量

❸ 在前向函数中，我们将标签和原始数据连接成一个向量。然后这个`combiner`接受这个非常大的张量，并创建一个大小仅为原始`input_shape`的新张量。这完成了将条件信息（来自标签嵌入）合并到潜在向量中的工作。

❹ 一个全连接层

❺ 第二个全连接层，但首先应用线性层和激活函数

❻ 因此我们可以重塑输出并基于目标输出形状进行归一化。这使得条件包装器对线性模型和卷积模型非常有用。

❼ 前向函数接收一个输入并产生一个输出。

❽ 如果没有给出标签，我们可以随机选择一个。

❾ 将标签嵌入并按需重塑

❿ 确保标签嵌入和数据 x 具有相同的形状，这样我们就可以将它们连接起来

⓫ 将潜在输入与嵌入的标签连接

⓬ 返回网络在组合输入上的结果

### 9.5.2 训练条件 GAN

使用此代码，将我们的普通全连接生成对抗网络（GANs）转换为条件 GANs 变得容易。以下代码片段创建了一个新的全连接 GAN。我们唯一改变的是定义`classes=10`的数量，并使用我们新的`ConditionalWrapper`独立包装 G 和 D：

```
latent_d = 128 
out_shape = (-1, 1, 28, 28) 
in_shape = (-1, latent_d) 
classes = 10 
G, D = simpleGAN(latent_d, neurons, out_shape, sigmoidG=True)

G = ConditionalWrapper(in_shape, neurons, classes, G) 
D = ConditionalWrapper(out_shape, neurons, classes, D)
```

现在我们需要一种方法来训练这些条件模型，因为`train_wgan`函数不使用标签。这是一个简单的改动。我们可以定义一个新的函数`train_c_wgan`，它具有完全相同的代码，除了每次我们在代码中有`G(noise)`时，我们将其更改为`G(noise, class_label)`。同样，每次我们看到类似`D(real)`的内容时，我们将其更改为`D(real, class_label)`。就是这样——我们只需在每次使用`G`或`D`时添加`, class_label)`！这为我们提供了创建条件 GANs 的工具和代码。以下代码块训练了我们刚刚定义的模型：

```
D_losses, G_losses = train_c_wgan(D, G, train_loader, latent_d, 
➥ epochs=20, device=device)

G = G.eval() 
D = D.eval()
```

### 9.5.3 使用条件 GANs 控制生成

让我们可视化这个新 GAN 的结果，并展示我们如何同时控制这个过程。以下代码生成 10 个不同的潜在向量 z，并为每个 z 值创建 10 个副本。然后我们创建一个`labels`张量，从 0 到 9 计数，覆盖所有 10 个类别。这样，我们创建 *G*(**z**∣*y*=0), *G*(**z**∣*y*=1), …, *G*(**z**∣*y*=9)。

每个数字都是在使用相同的潜在向量生成。因为条件控制生成的类别，潜在 z 被迫学习 *风格*。如果你查看每一行，你将看到在每种情况下，一些基本风格在所有输出中都被保持，无论类别如何。

以下是代码：

```
with torch.no_grad(): 
    noise = torch.randn(10, latent_d, device=device). ❶ 
    ➥ repeat((1,10)).view(-1, latent_d) labels =
    torch.fmod(torch.arange(0, noise.size(0),         ❷ 
    ➥ device=device), classes) 
    fake_digits = G(noise, labels)                    ❸ 
    scores = D(fake_digits, labels) 
    fake_digits = fake_digits.cpu() 
    scores = scores.cpu().numpy().flatten() 
plot_gen_imgs(fake_digits)                            ❹
```

❶ 生成 10 个潜在噪声向量，并重复 10 次。我们重用相同的潜在代码。

❷ 从 0 到 9 计数，然后回到 0。这重复了 10 次。

❸ 使用相同的潜在噪声生成 10 张图像，但每次都更改标签。

❹ 当我们绘制结果时，我们应该看到从 0 到 9 的数字网格，其中每一行使用相同的潜在向量，并具有相似的可视属性。

![图片](img/CH09_UN15_Raff.png)

注意，如果一行开头的 0 有粗线，那么该行中所有的数字都有粗线。如果 0 向右倾斜，那么所有的数字都向右倾斜。这就是条件 GANs 让我们控制生成内容的方式。我们可以将这种方法扩展到更多条件属性，这将使我们能够更多地控制 G 生成输出内容和方式。但这需要大量的标记输出，而这现在总是有可能的。

## 9.6 遍历 GANs 的潜在空间

我们已经看到 GANs 可以在创建假数据方面做得非常出色，随着训练的进行，潜在表示 z 开始学习关于数据的有趣属性。即使我们没有标签，这也是正确的。但我们也可以通过改变潜在向量 z 本身来控制 GAN 生成的结果。这样做可以让我们通过仅标记少量图像来操纵图像，从而确定如何正确地改变 z。

### 9.6.1 从 Hub 获取模型
