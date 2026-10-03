
```py
from torch import nn

class BasicUNet(nn.Module):
    """A minimal UNet implementation."""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = torch.nn.ModuleList([
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
        ])
        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2),
        ])
        self.act = nn.SiLU() # The activation function
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x)) # Through the layer and the activation function
            if i < 2: # For all but the third (final) down layer:
              h.append(x) # Storing output for skip connection
              x = self.downscale(x) # Downscale ready for the next layer

        for i, l in enumerate(self.up_layers):
            if i > 0: # For all except the first up layer
              x = self.upscale(x) # Upscale
              x += h.pop() # Fetching stored output (skip connection)
            x = self.act(l(x)) # Through the layer and the activation function

        return x
```

在 MNIST 上使用这种架构训练的扩散模型产生以下样本（代码包含在补充材料中，这里为了简洁起见而省略）：

## 改进 UNet

这个简单的 UNet 适用于这个相对简单的任务，但远非理想。那么，我们可以做些什么来改进它呢？

+ 添加更多参数。可以通过在每个块中使用多个卷积层，通过在每个卷积层中使用更多的滤波器，或者通过使网络更深来实现。
+ 添加残差连接。使用 ResBlocks 而不是常规卷积层可以帮助模型学习更复杂的功能，同时保持训练稳定。
+ 添加归一化，如批归一化。批归一化可以帮助模型更快、更可靠地学习，确保每一层的输出都围绕 0 中心，并具有标准差为 1。
+ 添加正则化，如 dropout。Dropout 有助于防止模型过度拟合训练数据，这在处理较小的数据集时非常重要。
+ 添加注意力。通过引入自注意力层，我们允许模型在不同时间集中关注图像的不同部分，这有助于学习更复杂的功能。类似变压器的注意力层的添加也可以增加可学习参数的数量，这有助于模型学习更复杂的功能。缺点是注意力层在更高分辨率时计算成本要比常规卷积层高得多，因此我们通常只在较低分辨率（即 UNet 中的较低分辨率块）使用它们。
+ 添加一个额外的输入用于时间步长，这样模型可以根据噪音水平调整其预测。这称为时间步长调节，几乎所有最近的扩散模型都在使用。我们将在下一章中更多地了解有条件的模型。

作为对比，在 diffusers 库中使用 UNet 实现时，在 MNIST 上的结果如下，该库包含了上述所有改进：

###### 警告

这一部分可能会在未来通过结果和更多细节进行扩展。我们还没有开始训练具有不同改进的变体！

## 替代架构

最近，一些替代架构已被提出用于扩散模型。这些包括：

+ 变压器。Peebles 和 Xie 的 DiT 论文（“具有变压器的可扩展扩散模型”）显示了基于变压器的架构可以用于训练扩散模型，并取得了很好的结果。然而，变压器架构的计算和内存需求仍然是非常高分辨率的挑战。
+ [Simple Diffusion paper](https://arxiv.org/abs/2301.11093)链接中的 *UViT* 架构旨在兼顾两者的优点，通过用一大堆变压器块替换 UNet 的中间层。该论文的一个关键见解是，将大部分计算集中在 UNet 的较低分辨率块上，可以更有效地训练高分辨率扩散模型。对于非常高的分辨率，他们使用称为小波变换的东西进行一些额外的预处理，以减少输入图像的空间分辨率，同时通过使用额外的通道尽可能保留更多信息，再次减少在更高空间分辨率上花费的计算量。
+ 循环接口网络。[RIN paper](https://arxiv.org/abs/2212.11972)（Jabri 等）采用类似的方法，首先将高分辨率输入映射到更易处理和低维的“潜在”表示，然后通过一堆变压器块进行处理，然后解码回到图像。此外，RIN 论文引入了“循环”概念，其中信息从上一个处理步骤传递给模型，这对于扩散模型旨在执行的迭代改进可能是有益的。

尚不清楚基于变压器的方法是否会完全取代 UNet 作为扩散模型的首选架构，还是像 UViT 和 RIN 架构这样的混合方法将被证明是最有效的。

# 深入：目标和预处理

我们已经谈到扩散模型接受嘈杂的输入并“学会去噪”它。乍一看，你可能会认为网络的自然预测目标是图像的去噪版本，我们将其称为 `x0`。然而，在代码中，我们将模型预测与用于创建嘈杂版本的单位方差噪声进行了比较（通常称为 epsilon 目标，`eps`）。这两者在数学上看起来是相同的，因为如果我们知道噪声和时间步长，我们可以推导出 `x0`，反之亦然。虽然这是真的，但目标的选择对不同时间步的损失有一些微妙的影响，因此模型学习最佳去噪哪个噪声水平。为了获得一些直觉，让我们在不同的时间步上可视化一些不同的目标：

![image](img/cell-26-output-1.png)

在极低的噪声水平下，`x0` 目标是非常容易的，而准确预测噪声几乎是不可能的。同样，在极高的噪声水平下，`eps` 目标是容易的，而准确预测去噪图像几乎是不可能的。两种情况都不理想，因此引入了其他目标，使模型在不同的时间步预测 `x0` 和 `eps` 的混合。[“用于快速采样扩散模型的渐进蒸馏。”](https://arxiv.org/abs/2202.00512)中引入的 `v` 目标是其中之一，它被定义为 `v = sqrt(ᾱ) · ε + sqrt(1 - ᾱ) · x0`。Karras 等人在 [EDM paper](https://arxiv.org/abs/2206.00364) 中通过一个称为 `c_skip` 的参数引入了类似的想法，并将不同的扩散模型公式统一到一个一致的框架中。如果您对了解不同目标、缩放和其他不同扩散模型公式的微妙之处感兴趣，我们建议阅读他们的论文以获得更深入的讨论。

# 项目时间：训练您自己的扩散模型

现在您已经了解了扩散模型的基础知识，现在是时候自己训练一些了！本章的补充材料包括一个笔记本，指导您如何在自己的数据集上训练扩散模型的过程。在您进行操作时，请回顾本章，看看不同部分是如何相互配合的。笔记本还包括许多建议的更改，以更好地探索不同的模型架构和训练策略如何影响结果。

# 总结

在本章中，我们看到了迭代改进的想法如何应用于训练扩散模型，使其能够将噪音转化为美丽的图像。您已经看到了一些设计选择，这些选择是创建成功的扩散模型所必需的，并希望通过训练自己的模型来实践这些选择。在下一章中，我们将看看一些更先进的技术，这些技术已经被开发出来，以改进扩散模型的性能，并赋予它们非凡的新能力！

# 参考文献

Ho, Jonathan, Ajay Jain, and Pieter Abbeel. “Denoising diffusion probabilistic models.” Advances in Neural Information Processing Systems 33 (2020): 6840-6851.

Ronneberger, O., Fischer, P. and Brox, T., 2015. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18 (pp. 234-241). Springer International Publishing.

Bansal, Arpit, Eitan Borgnia, Hong-Min Chu, Jie S. Li, Hamid Kazemi, Furong Huang, Micah Goldblum, Jonas Geiping, and Tom Goldstein. “Cold diffusion: Inverting arbitrary image transforms without noise.” arXiv preprint arXiv:2208.09392 (2022).

Hoogeboom, Emiel, Jonathan Heek, and Tim Salimans. “simple diffusion: End-to-end diffusion for high resolution images.” arXiv preprint arXiv:2301.11093 (2023).

Chang, Huiwen, Han Zhang, Jarred Barber, A. J. Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang et al. “Muse: Text-To-Image Generation via Masked Generative Transformers.” arXiv preprint arXiv:2301.00704 (2023).

Chang, Huiwen, Han Zhang, Lu Jiang, Ce Liu, and William T. Freeman. “Maskgit: Masked generative image transformer.” In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11315-11325. 2022.

Rampas, Dominic, Pablo Pernias, Elea Zhong, and Marc Aubreville. “Fast Text-Conditional Discrete Denoising on Vector-Quantized Latent Spaces.” arXiv preprint arXiv:2211.07292 (2022).

Chen, Ting “On the Importance of Noise Scheduling for Diffusion Models.” arXiv preprint arXiv:2301.10972 (2023).

Peebles, William, and Saining Xie. “Scalable Diffusion Models with Transformers.” arXiv preprint arXiv:2212.09748 (2022).

Jabri, Allan, David Fleet, and Ting Chen. “Scalable Adaptive Computation for Iterative Generation.” arXiv preprint arXiv:2212.11972 (2022).

Salimans, Tim, and Jonathan Ho. “Progressive distillation for fast sampling of diffusion models.” arXiv preprint arXiv:2202.00512 (2022).

Karras, Tero, Miika Aittala, Timo Aila, and Samuli Laine. “Elucidating the design space of diffusion-based generative models.” arXiv preprint arXiv:2206.00364 (2022).
