
```py
def train_gan(generator, discriminator, train_loader, codings_dim, n_epochs=20,
              g_lr=1e-3, d_lr=5e-4):
    criterion = nn.BCELoss()
    generator_opt = torch.optim.NAdam(generator.parameters(), lr=g_lr)
    discriminator_opt = torch.optim.NAdam(discriminator.parameters(), lr=d_lr)
    for epoch in range(n_epochs):
        for real_images, _ in train_loader:
            real_images = real_images.to(device)
            pred_real = discriminator(real_images)
            batch_size = real_images.size(0)
            ones = torch.ones(batch_size, 1, device=device)
            real_loss = criterion(pred_real, ones)
            codings = torch.randn(batch_size, codings_dim, device=device)
            fake_images = generator(codings).detach()
            pred_fake = discriminator(fake_images)
            zeros = torch.zeros(batch_size, 1, device=device)
            fake_loss = criterion(pred_fake, zeros)
            discriminator_loss = real_loss + fake_loss
            discriminator_opt.zero_grad()
            discriminator_loss.backward()
            discriminator_opt.step()

            codings = torch.randn(batch_size, codings_dim, device=device)
            fake_images = generator(codings)
            for p in discriminator.parameters():
                p.requires_grad = False
            pred_fake = discriminator(fake_images)
            generator_loss = criterion(pred_fake, ones)
            generator_opt.zero_grad()
            generator_loss.backward()
            generator_opt.step()
            for p in discriminator.parameters():
                p.requires_grad = True
```

如前所述，你可以在每个迭代中看到两个阶段：首先判别器执行梯度下降步骤，然后轮到生成器。我们为每个使用一个单独的优化器。让我们更详细地看看：

## 第一阶段

我们向判别器输入一批真实图像，并计算等于一的标签给出的损失；确实，我们想让判别器预测这些图像是真实的。然后我们生成一些随机编码，并将它们输入到生成器中以生成一些假图像。注意，我们对这些图像调用`detach()`，因为我们不希望梯度下降影响生成器在这个阶段。然后我们将这些假图像传递给判别器，并计算等于零的标签给出的损失；我们想让判别器预测这些图像是假的。判别器的总损失是`real_loss`加上`fake_loss`。最后，我们执行梯度下降步骤，改进判别器。

## 第二阶段

我们使用生成器生成一些假图像，并将它们传递给判别器，就像我们刚才做的那样。然而，这次我们不对假图像调用`detach()`，因为我们想训练生成器。此外，我们通过将每个参数`p`的`p.required_grad`设置为`False`来使判别器不可训练。然后我们使用等于一的标签来计算损失：确实，我们想让生成器欺骗判别器，所以想让判别器错误地预测这些是真实图像。最后，我们为生成器执行梯度下降步骤，并再次使判别器可训练。

就这样！训练完成后，你可以从高斯分布中随机采样一些编码，并将它们输入到生成器中，以生成新的图像：

```py
generator.eval()
codings = torch.randn(n_images, codings_dim, device=device)
with torch.no_grad():
    generated_images = generator(codings)
```

如果你展示了生成的图像（见图 18-16），你将看到在第一个训练周期结束时，它们已经开始看起来像（非常嘈杂的）Fashion MNIST 图像。

![试图描绘 Fashion MNIST 物品的嘈杂、黑白图像，由 GAN 在训练一个周期后生成。](img/hmls_1816.png)

###### 图 18-16\. GAN 在训练一个周期后生成的图像

很不幸，图像的质量永远不会比这更好，你甚至可能会发现某些时期，GAN 似乎忘记了它学到的内容。这是为什么？好吧，事实证明，训练 GAN 可能具有挑战性。让我们看看原因。

## 训练 GAN 的困难

在训练过程中，生成器和判别器不断在零和游戏中试图智胜对方。随着训练的进行，游戏可能最终陷入一种博弈论学家称之为纳什均衡的状态，以数学家约翰·纳什的名字命名。这种情况发生在没有任何玩家会因改变自己的策略而变得更好，假设其他玩家不会改变他们的策略。例如，当每个人都开车在道路的左侧时，就会达到纳什均衡：没有司机会因为成为唯一一个改变车道的人而变得更好。当然，还有第二个可能的纳什均衡：当每个人都开车在*右侧*的道路上。不同的初始状态和动态可能导致一个均衡或另一个均衡。在这个例子中，一旦达到均衡，就有一个单一的优化策略（即与其他人一样开车），但纳什均衡可以涉及多个竞争策略（例如，捕食者追逐猎物，猎物试图逃跑，并且双方都不会因改变策略而变得更好）。

那么，这如何适用于 GAN？嗯，GAN 论文的作者证明了 GAN 只能达到一个纳什均衡：那就是生成器产生完全逼真的图像，而判别器被迫猜测（50%真实，50%伪造）。这个事实非常鼓舞人心，因为它似乎意味着你只需要训练 GAN 足够长的时间，它最终会达到这个均衡，给你一个完美的生成器。不幸的是，事情并没有那么简单：没有任何保证均衡会被达到。

最大的困难被称为*模式崩溃*：当生成器的输出逐渐变得不那么多样化。这怎么可能发生呢？假设生成器在制作令人信服的鞋子方面比其他任何类别都做得更好。它将用鞋子稍微欺骗判别器，这将鼓励它产生更多鞋子的图像。渐渐地，它将忘记如何产生其他任何东西。同时，判别器将看到的唯一伪造图像将是鞋子，因此它也会忘记如何区分其他类别的伪造图像。最终，当判别器设法区分伪造的鞋子与真实的鞋子时，生成器将被迫转移到另一个类别。然后它可能擅长衬衫，忘记鞋子，判别器也会跟随。GAN 可能逐渐在几个类别之间循环，但从未真正擅长任何一个（参见图 18-17 的顶部行）。

![说明 GAN 训练中模式崩溃的图，显示顶部行中的聚类伪造示例与底部行中成功训练的多样化结果](img/hmls_1817.png)

###### 图 18-17\. 训练 GAN 时的模式坍塌（顶部行）与无模式坍塌的成功训练（底部行）

此外，由于生成器和判别器不断相互对抗，它们的参数最终可能会振荡并变得不稳定。训练可能一开始进行得很顺利，然后突然由于这些不稳定性而突然发散，没有任何明显的原因。由于许多因素会影响这些复杂的动态，GANs 对超参数非常敏感：你可能需要花费大量精力来微调它们。

这些问题自 2014 年以来一直让研究人员非常忙碌。关于这个主题已经发表了多篇论文，其中一些提出了新的成本函数⁠^(13)（尽管谷歌研究人员在 2018 年的一篇论文中 14 对其效率提出了质疑）或用于稳定训练或避免模式坍塌问题的技术。例如，一种流行的技术称为*经验回放*，它包括在每个迭代中将生成器产生的图像存储在回放缓冲区中（逐渐丢弃较旧的生成图像）并使用真实图像加上从该缓冲区中抽取的假图像来训练判别器（而不是仅使用当前生成器产生的假图像）。这减少了判别器过度拟合最新生成器输出的可能性。另一种常见的技术称为*小批量判别*：它衡量批次中图像之间的相似性，并将此统计数据提供给判别器，以便它可以轻松拒绝缺乏多样性的整个批次假图像。这鼓励生成器产生更多样化的图像，减少模式坍塌的可能性（见图 18-17 的底部行）。

简而言之，这是一个非常活跃的研究领域，直到最近都取得了很大的进展：从基于卷积层的*深度卷积生成对抗网络*（DCGANs）（见笔记本中的示例），到能够生成高分辨率图像的*渐进式增长生成对抗网络*，或允许用户对图像生成过程进行精细控制的*风格生成对抗网络*（StyleGANs），GANs 似乎有着光明的未来。但是，当扩散模型也开始产生令人惊叹的图像，并且具有更稳定的训练过程和更多样化的图像时，GANs 很快就被边缘化了。因此，现在让我们将注意力转向扩散模型。

# 扩散模型

扩散模型背后的思想已经存在很多年了，但它们首次以现代形式在斯坦福大学和加州大学伯克利分校的 Jascha Sohl-Dickstein 等人于 2015 年发表的一篇[论文](https://homl.info/diffusion)⁠^(15)中得到正式化。作者将统计力学的工具应用于模拟一个扩散过程，类似于牛奶在茶杯中扩散的过程。核心思想是训练一个模型来学习逆向过程：从完全混合的状态开始，逐渐“去混合”茶中的牛奶。利用这个想法，他们在图像生成方面获得了有希望的结果，但由于当时的 GANs 生成的图像更加令人信服，而且速度更快，扩散模型并没有得到太多的关注。

然后，在 2020 年，来自加州大学伯克利分校的[Jonathan Ho 等人](https://homl.info/ddpm)成功构建了一个能够生成高度逼真图像的扩散模型，他们将这个模型称为*去噪扩散概率模型*（DDPM）。⁠^(16)几个月后，OpenAI 的研究员 Alex Nichol 和 Prafulla Dhariwal 发表了一篇[2021 年的论文](https://homl.info/ddpm2)⁠^(17)，分析了 DDPM 架构并提出了几个改进，使得 DDPM 最终能够击败 GANs：DDPM 的训练比 GANs 容易得多，而且生成的图像更加多样化，质量更高。DDPM 的主要缺点，正如你将看到的，是它们生成图像所需的时间非常长，与 GANs 或 VAEs 相比。

那么，DDPM 究竟是如何工作的呢？好吧，假设你从一个猫的图片（如图 18-18 中的图片）开始，记为**x**[0]，在每一个时间步长*t*，你向图像中添加一点高斯噪声，均值为 0，方差为*β*[*t*]（一个标量）。这种噪声对每个像素是独立的（使用相同的均值和方差）：我们称之为*各向同性*。你首先得到图像**x**[1]，然后**x**[2]，以此类推，直到猫被噪声完全掩盖，无法看见。最后一个时间步长记为*T*。在原始的 DDPM 论文中，作者使用了*T* = 1,000，并且他们安排方差*β*[*t*]的方式使得猫的信号在时间步长 0 和*T*之间线性衰减。在改进的 DDPM 论文中，*T*被提高到 4,000，方差的时间表被调整以在开始和结束时变化得更慢。简而言之，我们是在逐渐将猫淹没在噪声中：这被称为*正向过程*。

![去噪扩散概率模型（DDPM）的示意图，展示了正向和逆向过程，其中猫的图像逐渐被噪声掩盖，直到细节丢失，说明了噪声是如何添加和移除的。](img/hmls_1818.png)

###### 图 18-18\. 正向过程*q*和逆向过程*p*

随着我们在正向过程中添加越来越多的高斯噪声，像素值的分布变得越来越像高斯分布。我遗漏的一个重要细节是，像素值在每一步都会稍微重新缩放，缩放因子为 $StartRoot 1 minus beta Subscript t Baseline EndRoot$ 。这确保了像素值的平均值逐渐接近 0，因为缩放因子略小于 1（想象一下反复将一个数字乘以 0.99）。这也确保了方差将逐渐收敛到 1。这是因为像素值的标准差也会被 $StartRoot 1 minus beta Subscript t Baseline EndRoot$ 缩放，因此方差会被缩放为 1 – *β*[*t*]（即缩放因子的平方）。但由于我们在每一步都添加了具有方差 *β*[*t*] 的高斯噪声，方差不能缩小到 0。并且由于高斯分布的方差相加，方差必须收敛到 1 – *β*[*t*] + *β*[*t*] = 1。

正向扩散过程总结在 方程 18-5 中。这个方程不会教你关于正向过程的新知识，但它有助于理解这种数学符号，因为它在机器学习论文中经常被使用。这个方程定义了给定 **x**[*t*–1] 作为具有平均值 **x**[*t*–1] 乘以缩放因子的高斯分布，并且协方差矩阵等于 *β*[*t*]**I** 的概率分布 *q*。这是单位矩阵 **I** 乘以 *β*[*t*]，这意味着噪声是各向同性的，方差为 *β*[*t*]。

##### 方程 18-5\. 正向扩散过程的概率分布 *q*

$q left-parenthesis bold x Subscript t Baseline vertical-bar bold x Subscript t minus 1 Baseline right-parenthesis equals script upper N left-parenthesis StartRoot 1 minus beta Subscript t Baseline EndRoot bold x Subscript t minus 1 Baseline comma beta Subscript t Baseline bold upper I right-parenthesis$

有趣的是，对于正向过程有一个快捷方式：在不需要首先计算 **x**[1]，**x**[2]，…，**x**[*t*–1] 的情况下，可以采样图像 **x**[*t*]。确实，由于多个独立高斯分布的总和也是一个高斯分布，所有噪声可以在一次操作中添加。如果我们定义 *α*[*t*] = 1 – *β*[*t*]，并且 *α̅*[*t*] = *α*[*1*] × *α*[*2*] × …​× *α*[*t*] = $alpha overbar Subscript t Baseline equals product Underscript i equals 1 Overscript t Endscripts alpha Subscript t$ ，那么我们可以使用 方程 18-6 来计算 **x**[*t*]。这就是我们将要使用的方程，因为它要快得多。

##### 方程 18-6\. 正向扩散过程的快捷方式

$q left-parenthesis bold x Subscript t Baseline vertical-bar bold x 0 right-parenthesis equals script upper N left-parenthesis StartRoot alpha overbar Subscript t Baseline EndRoot bold x 0 comma left-parenthesis 1 minus alpha overbar Subscript t Baseline right-parenthesis bold upper I right-parenthesis$

当然，我们的目标不是在噪声中淹死猫。相反，我们想要创造许多新的猫！我们可以通过训练一个能够执行 *反向过程* 的模型来实现这一点：从 **x**[*t*] 到 **x**[*t*–1]。然后我们可以用它来从图像中移除一小部分噪声，并重复操作多次，直到所有噪声都消失。这不是一个仅依赖于相邻像素的基本噪声过滤器：相反，当噪声被移除时，它会被根据训练数据替换为逼真的像素。例如，如果我们在一个包含许多猫图像的数据集上训练模型，那么我们可以给它一张完全由高斯噪声组成的图片，模型将逐渐使一只全新的猫出现（参见 图 18-18）。

好的，那么让我们开始编码！我们首先需要做的是编码前向过程。为此，我们首先需要实现方差调度。我们如何控制猫消失的速度？在每一个时间步 *t*，像素值会被乘以 $StartRoot 1 minus beta Subscript t Baseline EndRoot$，并添加均值为 0 和方差 *β*[*t*] 的噪声（如前所述）。因此，图像方差中来自原始猫图像的部分在每个步骤中会以 *α*[*t*] = 1 – \beta_t 的因子缩小。经过 *t* 个时间步后，它将缩小到因子 *α̅*[*t*] = *α*[*1*] × *α*[*2*] × …​ × *α*[*t*]。我们想要调度这个“猫信号”因子 *α̅*[*t*]，使其在时间步 0 和 *T* 之间逐渐从 1 缩小到 0。在改进的 DDPM 论文中，作者根据 方程 18-7 调度 *α̅*[*t*]。这个调度在 图 18-19 中表示。

##### 方程 18-7\. 前向扩散过程的方差调度方程
