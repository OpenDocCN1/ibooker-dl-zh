
```py
def generate_ddim(model, batch_size=32, num_steps=50, eta=0.85):
    model.eval()
    with torch.no_grad():
        xt = torch.randn([batch_size, 1, 28, 28], device=device)
        times = torch.linspace(T - 1, 0, steps=num_steps + 1).long().tolist()
        for t, t_prev in zip(times[:-1], times[1:]):
            t_batch = torch.full((batch_size, 1), t, device=device)
            sample = DiffusionSample(xt, t_batch)
            eps_pred = model(sample)
            x0 = ((xt - (1 - alpha_bars[t]).sqrt() * eps_pred)
                  / (alpha_bars[t].sqrt()))
            abar_t_prev = alpha_bars[t_prev]
            variance = eta * (1 - abar_t_prev) / (1 - alpha_bars[t]) * betas[t]
            sigma_t = variance.sqrt()
            pred_dir = (1 - abar_t_prev - sigma_t**2).sqrt() * eps_pred
            noise = torch.randn_like(xt)
            xt = abar_t_prev.sqrt() * x0 + pred_dir + sigma_t * noise

        return torch.clamp((xt + 1) / 2, 0, 1)  # from [–1, 1] range to [0, 1]

X_gen_ddim = generate_ddim(diffusion_model, num_steps=500)
```

这次生成将只需几秒钟，并将产生如图图 18-20 所示的图像。诚然，它们并不非常令人印象深刻，但我们只在 Fashion MNIST 上训练了模型几分钟。尝试在更大的数据集上训练它几个小时，以获得更令人印象深刻的结果。

![使用在 Fashion MNIST 数据集上有限训练的扩散模型生成的低分辨率、黑白服装物品网格。](img/hmls_1820.png)

###### 图 18-20\. 由 DDIM 加速扩散生成的图像

自 2020 年以来，扩散模型取得了巨大的进步。特别是，2021 年 12 月由 Robin Rombach 等人发表的一篇论文介绍了*潜在扩散模型*，其中扩散过程发生在潜在空间，而不是像素空间。为了实现这一点，使用了一个强大的自动编码器将每个训练图像压缩到一个更小的潜在空间，其中扩散过程发生，然后使用自动编码器来解压缩最终的潜在表示，生成输出图像。这大大加快了图像生成速度，并显著减少了训练时间和成本。重要的是，生成的图像质量非常出色。

此外，研究人员还采用了各种条件化技术，通过文本提示、图像或其他任何输入来引导扩散过程。这使得快速生成任何你想要的图像成为可能。你也可以使用输入图像来条件化图像生成过程。这使许多应用成为可能，例如扩展画布——将输入图像扩展到其边界之外，或者修复画布——在图像中填充洞。

最后，一个名为*Stable Diffusion*（SD）的强大预训练潜在扩散模型在 2022 年 8 月由慕尼黑大学与 StabilityAI、Runway 等几家公司的合作开源，得到了 EleutherAI 和 LAION 的支持。现在任何人都可以在几秒钟内免费生成令人惊叹的图像，甚至在普通的笔记本电脑上也可以。例如，你可以使用 Hugging Face Diffusers 库来加载 SD（例如，turbo 变体），创建一个从文本到图像的图像生成管道，并生成一只猩猩读书的图像：

```py
from diffusers import AutoPipelineForText2Image

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo", variant="fp16", dtype=torch.float16)
pipe.to(device)
prompt = "A closeup photo of an orangutan reading a book"
torch.manual_seed(26)
image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
```

![一只猩猩在仔细阅读一本打开的书，展示了 Stable Diffusion 模型的创造性能力。](img/hmls_1821.png)

###### 图 18-21\. 使用 Diffusers 库生成的 Stable Diffusion 图像

可能性是无限的！

在下一章中，我们将转向深度学习的另一个完全不同的分支：深度强化学习。

# 练习

1.  自动编码器主要用于哪些主要任务？

1.  假设你想训练一个分类器，你有很多未标记的训练数据，但只有几千个标记实例。自动编码器如何帮助？你会如何进行？

1.  如果一个自编码器完美地重建了输入，它是否必然是一个好的自编码器？你如何评估自编码器的性能？

1.  什么是欠完备和过完备的自编码器？一个过度欠完备的自编码器的主要风险是什么？过完备的自编码器的主要风险又是什么？

1.  你如何在堆叠自编码器中绑定权重？这样做有什么意义？

1.  什么是生成模型？你能命名一种生成自编码器的类型吗？

1.  什么是 GAN？你能列举一些 GAN 可以大放异彩的任务吗？

1.  训练 GAN 时主要有哪些困难？

1.  扩散模型擅长什么？它们的主要局限性是什么？

1.  尝试使用去噪自编码器来预训练一个图像分类器。你可以使用 MNIST（最简单的选项），或者一个更复杂的图像数据集，如[CIFAR10](https://homl.info/122)，如果你想要更大的挑战。无论你使用什么数据集，请遵循以下步骤：

    1.  将数据集分为训练集和测试集。在完整的训练集上训练一个深度去噪自编码器。

    1.  检查图像是否得到了相当好的重建。可视化激活编码层中每个神经元的图像。

    1.  构建一个分类深度神经网络（DNN），重用自编码器的底层。仅使用训练集中的 500 张图像进行训练。是否有预训练的情况下表现更好？

1.  在你选择的数据集上训练一个变分自编码器，并使用它来生成图像。或者，你可以尝试找到一个你感兴趣的无标签数据集，看看你是否可以生成新的样本。

1.  训练一个 DCGAN 来处理你选择的图像数据集，并使用它来生成图像。添加经验回放并看看这是否有帮助。

1.  在你喜欢的图像数据集（例如，`torchvision.datasets.Flowers102`）上训练一个扩散模型，并生成漂亮的图像。接下来，将图像类别作为额外的输入添加到模型中，并重新训练它：你现在应该能够控制生成图像的类别。

这些练习的解决方案可在本章笔记本的末尾找到，在[*https://homl.info/colab-p*](https://homl.info/colab-p)。

^(1) William G. Chase 和 Herbert A. Simon，“国际象棋中的感知”，*认知心理学* 4, 第 1 期 (1973): 55–81.

^(2) 提示：一种方法是为给定的数据集创建一个自定义的 `AutoencoderDataset` 类，该类包装给定数据集并将目标替换为输入。

^(3) Yoshua Bengio 等人，“深度网络贪婪层逐层训练”，*第 19 届国际神经网络信息处理系统会议论文集* (2006): 153–160.

^(4) Jonathan Masci 等人，“用于分层特征提取的堆叠卷积自编码器”，*第 21 届国际人工神经网络会议论文集* 1 (2011): 52–59.

^(5) Pascal Vincent 等人, “使用去噪自编码器提取和组合鲁棒特征”, *第 25 届国际机器学习会议论文集* (2008): 1096–1103.

^(6) Pascal Vincent 等人, “堆叠去噪自编码器：使用局部去噪标准在深度网络中学习有用的表示”, *机器学习研究杂志* 11 (2010): 3371–3408.

^(7) Diederik Kingma 和 Max Welling, “自动编码器变分贝叶斯”, arXiv 预印本 arXiv:1312.6114 (2013).

^(8) 这两个特性使得 VAEs 与 RBMs 非常相似，但它们更容易训练，采样过程也更快（使用 RBMs，你需要等待网络稳定到“热平衡”状态后才能采样新的实例）。

^(9) Chris J. Maddison 等人, “具体分布：离散随机变量的连续松弛”, arXiv 预印本 arXiv:1611.00712 (2016).

^(10) Eric Jang 等人, “使用 Gumbel-Softmax 进行分类重参数化”, arXiv 预印本 arXiv:1611.01144 (2016).

^(11) Aaron van den Oord 等人, “神经离散表示学习”, arXiv 预印本 arXiv:1711.00937 (2017).

^(12) Ian Goodfellow 等人, “生成对抗网络”, *第 27 届国际神经网络信息处理系统会议论文集* 2 (2014): 2672–2680.

^(13) 想要比较主要的 GAN 损失，可以查看 Hwalsuk Lee 的这个优秀的 [GitHub 项目](https://homl.info/ganloss)。

^(14) Mario Lucic 等人, “GANs 是否平等？一项大规模研究”, *第 32 届国际神经网络信息处理系统会议论文集* (2018): 698–707.

^(15) Jascha Sohl-Dickstein 等人, “使用非平衡热力学进行深度无监督学习”, arXiv 预印本 arXiv:1503.03585 (2015).

^(16) Jonathan Ho 等人, “去噪扩散概率模型” (2020).

^(17) Alex Nichol 和 Prafulla Dhariwal, “改进的去噪扩散概率模型” (2021).

^(18) Olaf Ronneberger 等人, “U-Net：用于生物医学图像分割的卷积网络”, arXiv 预印本 arXiv:1505.04597 (2015).

^(19) Jiaming Song 等人, “去噪扩散隐式模型”, arXiv 预印本 arXiv:2010.02502 (2020).

^(20) Robin Rombach, Andreas Blattmann 等人, “使用潜在扩散模型进行高分辨率图像合成”, arXiv 预印本 arXiv:2112.10752 (2021).
