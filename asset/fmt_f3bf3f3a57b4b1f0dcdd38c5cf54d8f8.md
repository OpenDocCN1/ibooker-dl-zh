
基本上，这个函数执行的功能与调用`.backward()`相同，但它返回一个新张量作为对象，而不是将其存储在`.grad`字段中。如果你不理解这个函数，那也无所谓，因为它的用途很专业。但我会给那些感兴趣的人提供一个快速的高级解释。`outputs=output`告诉PyTorch我们会调用`.backward()`的内容，而`inputs=mixed`告诉PyTorch导致这个结果的初始输入。`grad_outputs=torch.ones_like(output)`给PyTorch提供一个初始值来开始梯度计算，将其设置为全1，这样我们最终得到所有部分的期望梯度。选项`create_graph=True, retain_graph=True`告诉PyTorch我们希望自动微分在结果上工作（这做了梯度的梯度）：

```
eps_shape = [batch_size]+[1]*(len(data.shape)-1)               ❶ 
eps = torch.rand(eps_shape, device=device) 
mixed = eps*real + (1-eps)*fake 
output = D(mixed)

grad = torch.autograd.grad(outputs=output, inputs=mixed, 
➥ grad_outputs=torch.ones_like(output), create_graph=True, 
➥ retain_graph=True, only_inputs=True, allow_unused=True)[0]

D_grad_penalty = ((grad.norm(2, dim=1) - 1) ** 2)

errD = (D_failure-D_success).mean() + D_grad_penalty.mean()*10 ❷
errD.backward() 
optimizerD.step()                                              ❸
```

❶ 现在计算用于计算梯度惩罚项的张量。

❷ 计算D的损失

❸ 更新D

计算WGAN-GP损失

当`grad`变量最终到手时，我们可以快速计算总损失。`*10`是控制惩罚强度的λ项。我将其硬编码为10，因为这在这个方法中是一个很好的默认值，但更好的编码会将`lambda`作为函数的参数，默认值为10。

第二步在代码的最后一段中计算G的更新，我们首先重新将梯度归零。这是一个防御性编码步骤，因为跟踪我们何时以及何时没有改变梯度是很棘手的。然后我们在`noise`变量中采样一个新的潜在变量z，计算`-D(G(noise))`，取平均值，然后进行更新：

```
D.zero_grad()                                            ❺ 
G.zero_grad()                                            ❶

noise = torch.randn(batch_size, latent_d, device=device) ❷ 
output = -D(G(noise)) errG = output.mean()               ❸ 
errG.backward()                                          ❹
optimizerG.step()                                        ❺
```

❶ 第2步：-D(G(z))

❷ 由于我们刚刚更新了D，执行一个所有伪造批次通过D的前向传递。

❸ 基于此输出计算G的损失

❹ 计算G的梯度

❺ 更新G

我们通过记录G和D各自的损失来结束函数，以便稍后查看，并将这些损失作为函数调用的结果返回：

```
    D_losses.append(errD.item()) 
    G_losses.append(errG.item()) 
return D_losses, G_losses
```

现在我们有了这个新的`train_wgan`函数，我们可以尝试在早期用九个高斯分布在3 × 3网格上的玩具问题上训练一个新的Wasserstein GAN。我们只需调用以下代码片段：

```
G, D = simpleGAN(latent_d, 512, (-1, 2)) 
train_wgan(D, G, toy_loader, latent_d, epochs=20, device=device) 
G, D = G.eval(), D.eval()
```

减少模式坍塌的结果

以下代码生成一些新的样本。结果并不完美，但它们比我们之前得到的好得多。GAN已经学会了覆盖更广泛的输入数据，分布的形状更接近我们知道的真值。我们可以一般地看到所有九个模式都被表示出来，这是好的，尽管其中一些不如其他强烈：

```
    with torch.no_grad(): 
        noise = torch.randn(X.shape[0], latent_d, device=device) 
        fake_samples_w = G(noise).cpu().numpy() 
    plt.figure(figsize=(10,10)) 
    ax = sns.kdeplot(x=fake_samples_w[:,0], y=fake_samples_w[:,1], 
    ➥ shade=True, thresh=-0.001) 
    plt.xlim(-1.5, 1.5) 
    plt.ylim(-1.5, 1.5)

[27]: (-1.5, 1.5)
```

![图片](img/CH09_UN10_Raff.png)

模式坍塌问题并没有通过Wasserstein方法得到解决，但它得到了极大的改善。Wasserstein方法以及大多数GAN的一个缺点是它需要更多的迭代才能收敛。提高WGAN训练的一个常见技巧是对于生成器的每次更新，更新判别器五次，这进一步增加了获得最佳可能结果所需的总epoch数。对于大多数问题，你应该进行大约200到400个epoch来训练一个真正好的GAN。即使使用WGAN，如果你再次训练它，你经常会看到它坍塌。这是训练GAN的臭名昭著的一部分，但更多的数据和更多的训练epoch将提高其性能。

话虽如此，让我们回到我们最初的MNIST问题，看看我们的WGAN-GP是否提高了结果。我们重新定义潜在维度大小为128，输出形状与MNIST相匹配。我们再次为G使用sigmoid输出，因为MNIST的值都在[0,1]的范围内。以下代码使用我们的新方法进行训练：

```
latent_d = 128 
out_shape = (-1, 1, 28, 28) 
G, D = simpleGAN(latent_d, neurons, out_shape, sigmoidG=True)

D_losses, G_losses = train_wgan(D, G, train_loader, latent_d, epochs=40, 
➥ device=device)

G = G.eval() 
D = D.eval()
```

注意：为什么更新判别器D的次数要多于更新生成器G的次数？这样判别器就有更多机会更新并捕捉到生成器G的行为。这是可取的，因为G只能变得和判别器D一样好。由于复杂性惩罚，这并没有给D带来太大的不公平优势。

现在我们生成一些合成数据，但我们不会查看分数：分数不再是概率，因此很难将其解释为单个值。这就是为什么我们将`scores`作为`plot_gen_imgs`的可选参数：

```
with torch.no_grad(): 
    noise = torch.randn(batch_size, latent_d, device=device) 
    fake_digits = G(noise) 
    scores = D(fake_digits) 
    fake_digits = fake_digits.cpu() 
    scores = scores.cpu().numpy().flatten() 
plot_gen_imgs(fake_digits)
```

![图片](img/CH09_UN11_Raff.png)

总体而言，我们的MNIST GAN样本看起来比以前好得多。你应该能在大多数样本中找到所有10个数字的例子，即使有些数字有点难看或不规则。你也应该看到单个特定数字内的样式更加多样化！这些都是我们GAN质量的主观巨大改进。一个缺点是，我们的0、3和8现在没有我们之前在模式崩溃时那么清晰：我们的WGAN-GP可能需要更多的训练周期，而且它现在正在做*更多*的事情。当原始GAN崩溃时，它可以使用整个网络来表示仅三个数字。现在每个数字占用的网络更少，因为我们正在表示所有10个数字。这也是我建议将生成的结果与真实数据比较以决定结果是否良好的原因之一。高质量的崩溃结果可能会让你对GAN的质量产生错误的看法。

我们还可以再次绘制判别器和生成器的损失。以下代码执行此操作，并使用卷积来平滑平均损失，以便我们可以关注趋势。在解释WGAN-GP的损失时，请记住，生成器G希望有一个大的损失，而判别器D希望有一个小的损失：

```
plt.figure(figsize=(10,5)) 
plt.title("Generator and Discriminator Loss During Training") 
plt.plot(np.convolve(G_losses, np.ones((100,))/100, mode=’valid’) ,label="G") 
plt.plot(np.convolve(D_losses, np.ones((100,))/100, mode=’valid’) ,label="D") 
plt.xlabel("iterations") 
plt.ylabel("Loss") 
plt.legend() 
plt.show()
```

![图片](img/CH09_UN12_Raff.png)

解释这个图，G随着训练的进行而改进，这是好事，也是继续训练的理由。但是，判别器D似乎已经停止了改进，这可能是一个迹象，表明D现在不足以学习更好的表示。这是一个问题，因为G的质量只有当D在任务上变得更好时才会提高。我们可以做出的一个主要改变是实现卷积GAN，这样生成器（和判别器）就能获得卷积的结构先验，并且（我们希望）两者都能做得更好！

## 9.4 卷积GAN

与大多数深度学习应用一样，要改善GAN的结果，最佳方法是选择一个适合你问题的架构。由于我们正在处理图像，我们可以使用卷积架构来提高我们的结果。这需要对潜在表示z进行一些调整。我们的代码为z中的D维度创建了一个形状为(*B*,*D*)的向量。对于卷积G，我们的潜在应该是什么形状？

### 9.4.1 设计卷积生成器

一种方法是创建一个形状为(*B*,*C*′,*W*′,*H*′)的潜在张量，这样它已经像图像一样形状，但没有正确的输出形状(*B*,*C*,*W*,*H*)。我们使这个潜在的张量的宽度和高度比真实数据小，并使用转置卷积将其扩展到真实数据的大小。但对于*C*′，我们希望使用比原始数据更多的通道。这样，模型就可以学会将潜在通道解释为具有不同类型的意义，而且只有一个或三个通道可能太小。

以下代码设置了训练*卷积生成对抗网络*的参数。首先，我们确定`start_size`，它将是初始宽度W′和高度H′。我们将W′设置为W/4，这样我们就可以通过转置卷积进行两轮扩展。如果我们不使用一些丑陋的代码，就不能再小了，因为那会让我们降到28/4=7像素：再次除以2将是3.5像素，这不是我们容易处理的事情。接下来，我们定义`latent_channels` *C*′为16。这是一个超参数，我简单地选择了一个较小的值，如果我的结果不好，我会增加它。由于我们有一个形状为(*B*,*C*′,*W*′,*H*′)的潜在变量，我们定义`in_shape`元组来表示它，作为我们一直在使用的`out_shape`元组的对应物。我们的网络将使用`in_shape`来重塑G的输入；`out_shape`仍将用于重塑G的输出和D的输入：

```
start_size = 28//4                              ❶ 
latent_channels = 16

latent_d_conv = latent_channels*(start_size**2) ❷ 
in_shape = (-1, latent_channels, start_size, start_size )
```

❶ 初始宽度和高度，以便我们可以进行两轮转置卷积

❷ 潜在空间中需要的值数量

此代码在设置我们问题的变量同时，也便于调整以适应新的问题。如果你有更大的图像，可以尝试使用超过两轮的扩展和超过16个潜在通道，这可以通过更改前两行轻松实现。

实现卷积辅助函数

在定义我们的卷积架构之前，我们需要几个变量。我们以`n_filters=32`个过滤器开始我们的生成器，并使用LeakyReLU激活函数的漏率0.2。我们还定义了核大小为`k_size=5`。通常，我们使用3x3的核大小来增加CNN的深度。对于GAN，我经常发现稍微大一点的核大小可以改善结果。特别是，如果我们使用步长的倍数作为核大小，我们的转置卷积将会有更平滑的输出，所以我们给它们一个单独的核大小`k_size_t=4`。这有助于确保当转置卷积扩展输出时，输出中的每个位置都会得到其值的均匀数量的贡献：

```
n_filters = 32                                                ❶ 

k_size= 5                                                     ❷ 

k_size_t = 4                                                  ❸ 
leak = 0.2

def cnnLayer(in_channels, out_channels, filter_size,          ❹ 
➥ wh_size, leak=0.2): 
        return nn.Sequential( 
        nn.Conv2d(in_channels, out_channels, filter_size, 
        ➥ padding=filter_size//2), 
        nn.LeakyReLU(leak), 
        nn.LayerNorm([out_channels, wh_size, wh_size]), 
    ) 
def tcnnLayer(in_channels, out_channels, wh_size, leak=0.2):  ❺ 
    return nn.Sequential( 
        nn.ConvTranspose2d(in_channels, out_channels, k_size_t, 
        ➥ padding=1, output_padding=0, stride=2), 
        nn.LeakyReLU(leak), 
        nn.LayerNorm([out_channels, wh_size, wh_size]), 
    )
```

❶ 潜在空间中的通道数

❷ 默认用于卷积生成对抗网络的核大小

❸ 转置卷积的默认核大小

❹ 创建隐藏卷积层的辅助函数

❺ 与`cnnLayer`类似，但我们使用转置卷积来扩展大小

实现卷积生成对抗网络

现在我们定义我们的CNN GAN。对于G，我们从`View(in_shape)`开始，这样潜在向量z就变成了我们指定的所需输入形状：(*B*,*C*′,*W*′,*H*′)。接下来是一系列的卷积、激活和LN操作。记住，对于CNN，我们需要跟踪输入的宽度和高度，这就是为什么我们将MNIST的高度28x28编码到我们的架构中。我们遵循在每次转置卷积前后使用两到三个CNN层的简单模式：

```
G = nn.Sequential( 
    View(in_shape), 
    cnnLayer(latent_channels, n_filters, k_size, 28//4, leak), 
    cnnLayer(n_filters, n_filters, k_size, 28//4, leak), 
    cnnLayer(n_filters, n_filters, k_size, 28//4, leak), 
    tcnnLayer(n_filters, n_filters//2, 28//2, leak), 
    cnnLayer(n_filters//2, n_filters//2, k_size, 28//2, leak), 
    cnnLayer(n_filters//2, n_filters//2, k_size, 28//2, leak), 
    tcnnLayer(n_filters//2, n_filters//4, 28, leak), 
    cnnLayer(n_filters//4, n_filters//4, k_size, 28, leak), 
    cnnLayer(n_filters//4, n_filters//4, k_size, 28, leak), 
    nn.Conv2d(n_filters//4, 1, k_size, padding=k_size//2), 
    nn.Sigmoid(), 
)
```

### 9.4.2 设计卷积判别器

接下来，我们实现判别器D。这需要从我们的正常设置中做出一些更多的改变。再次，这些是一些可以帮助改进你的GANs的技巧：
