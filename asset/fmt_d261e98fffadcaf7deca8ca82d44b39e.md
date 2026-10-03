
以下 PyTorch 代码初始化了每个组件，紧密模仿了原始论文中官方 PyTorch 实现的各个版本。第一个组件，直到最大池，作用于原始输入，接下来的每个组件之间只需要降采样。这是因为在每个组件内，每个`ResidualBlock`的输入和输出维度相同。虽然我们在本节中不会明确展示，但`kernel_size`为 3，`stride`为 1，`padding`为 1 的组合确保每个特征图的大小从头到尾保持不变。此外，由于每个组件内特征图的数量保持不变，所有维度最终保持不变：

```py
class ResNet34(nn.Module):
  def __init__(self):
    super(ResNet34, self).__init__()

    self.conv1 = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=7,
                stride=2, padding=3, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3,
                   stride=2, padding=1)
    )

    # Note that each ResidualBlock has 2 conv layers
    # 3 blocks in a row, 6 conv layers
    self.comp1 = nn.Sequential(
      ResidualBlock(64, 64),
      ResidualBlock(64, 64),
      ResidualBlock(64, 64)
    )

    # 4 blocks in a row, 8 conv layers
    downsample1 = nn.Sequential(
      nn.Conv2d(64, 128, kernel_size=1,
             stride=1, bias=False),
      nn.BatchNorm2d(128)
    )
    self.comp2 = nn.Sequential(
      ResidualBlock(64, 128, downsample=downsample1),
      ResidualBlock(128, 128),
      ResidualBlock(128, 128),
      ResidualBlock(128, 128)
    )

    # 6 blocks in a row, 12 conv layers
    downsample2 = nn.Sequential(
      nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False),
      nn.BatchNorm2d(256)
    )
    self.comp3 = nn.Sequential(
      ResidualBlock(128, 256, downsample=downsample2),
      ResidualBlock(256, 256),
      ResidualBlock(256, 256),
      ResidualBlock(256, 256),
      ResidualBlock(256, 256),
      ResidualBlock(256, 256),
    )

    # 3 blocks in a row, 6 conv layers
    downsample3 = nn.Sequential(
      nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False),
      nn.BatchNorm2d(512)
    )
    self.comp4 = nn.Sequential(
      ResidualBlock(256, 512, downsample=downsample3),
      ResidualBlock(512, 512),
      ResidualBlock(512, 512)   
    )

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    # ImageNet classifier: 1000 classes
    self.fc = nn.Linear(512, 1000)

  def forward(self, inp):
    out = self.conv1(inp)

    out = self.comp1(out)
    out = self.comp2(out)
    out = self.comp3(out)
    out = self.comp4(out)

    out = self.avgpool(out)
    out = torch.flatten(out, 1)
    out = self.fc(out)

    return out
```

在下一节中，我们将介绍关于神经风格转移的计算机视觉领域的一些最新进展。

# 利用卷积滤波器复制艺术风格

在过去的几年里，我们还开发了利用卷积网络以更有创意方式的算法。其中一个算法被称为*神经风格*。¹⁷ 神经风格的目标是能够将任意照片呈现为以著名艺术家的风格绘制的样子。这似乎是一项艰巨的任务，如果没有卷积网络，我们不太清楚如何解决这个问题。然而，聪明地操作卷积滤波器可以在这个问题上产生惊人的结果。

让我们拿一个预训练的卷积网络。我们处理三幅图像。前两幅是内容源***p***和风格源***a***。第三幅图像是生成的图像***x***。我们的目标是推导一个错误函数，我们可以反向传播，当最小化时，将完美地结合所需照片的内容和所需艺术作品的风格。

我们先从内容开始。如果网络中的一层有<math alttext="k Subscript l"><msub><mi>k</mi> <mi>l</mi></msub></math>个滤波器，那么它会产生<math alttext="k Subscript l"><msub><mi>k</mi> <mi>l</mi></msub></math>个特征图。让我们称每个特征图的大小为<math alttext="m Subscript l"><msub><mi>m</mi> <mi>l</mi></msub></math>，即特征图的高乘以宽。这意味着该层所有特征图中的激活可以存储在大小为<math alttext="k Subscript l Baseline times m Subscript l"><mrow><msub><mi>k</mi> <mi>l</mi></msub> <mo>×</mo> <msub><mi>m</mi> <mi>l</mi></msub></mrow></math>的矩阵***F***^((*l*))中。我们还可以用矩阵***P***^((*l*))表示照片的所有激活，用矩阵***X***^((*l*))表示生成图像的所有激活。我们使用原始 VGGNet 的`relu4_2`：

*E*content = ∑ij) – ***X***[ij]^((*l*)))²

现在我们可以尝试处理风格。为此，我们构建一个被称为*Gram 矩阵*的矩阵，它代表了给定层中特征图之间的相关性。这些相关性代表了纹理和感觉，这些纹理和感觉在所有特征中都是共同的，无论我们正在查看哪些特征。构建 Gram 矩阵，其大小为<math alttext="k 下标 l 乘以 k 下标 l"><mrow><msub><mi>k</mi> <mi>l</mi></msub> <mo>×</mo> <msub><mi>k</mi> <mi>l</mi></msub></mrow></math>，对于给定的图像，可以按照以下步骤完成：

**G**^((*l*))[ij] = ∑[c = 0]^(m[l]) **F**^((*l*))[*ic*] **F**^((*l*))[*jc*]

我们可以计算艺术品矩阵***A***^((*l*))和生成图像矩阵***G***^((*l*))的 Gram 矩阵。然后我们可以将误差函数表示为：

<math alttext="upper E Subscript s t y l e Baseline left-parenthesis bold a comma bold x right-parenthesis equals StartFraction 1 Over 4 k Subscript l Superscript 2 Baseline m Subscript l Superscript 2 Baseline EndFraction sigma-summation Underscript l equals 1 Overscript upper L Endscripts sigma-summation Underscript i j Endscripts StartFraction 1 Over upper L EndFraction left-parenthesis upper A Subscript i j Superscript left-parenthesis l right-parenthesis Baseline minus upper G Subscript i j Superscript left-parenthesis l right-parenthesis Baseline right-parenthesis squared"><mrow><msub><mi>E</mi> <mrow><mi>s</mi><mi>t</mi><mi>y</mi><mi>l</mi><mi>e</mi></mrow></msub> <mrow><mo>(</mo> <mi>𝐚</mi> <mo>,</mo> <mi>𝐱</mi> <mo>)</mo></mrow> <mo>=</mo> <mfrac><mn>1</mn> <mrow><mn>4</mn><msubsup><mi>k</mi> <mi>l</mi> <mn>2</mn></msubsup> <msubsup><mi>m</mi> <mi>l</mi> <mn>2</mn></msubsup></mrow></mfrac> <msubsup><mo>∑</mo> <mrow><mi>l</mi><mo>=</mo><mn>1</mn></mrow> <mi>L</mi></msubsup> <msub><mo>∑</mo> <mrow><mi>i</mi><mi>j</mi></mrow></msub> <mfrac><mn>1</mn> <mi>L</mi></mfrac> <msup><mfenced separators="" open="(" close=")"><msubsup><mi>A</mi> <mrow><mi>i</mi><mi>j</mi></mrow> <mrow><mo>(</mo><mi>l</mi><mo>)</mo></mrow></msubsup> <mo>-</mo><msubsup><mi>G</mi> <mrow><mi>i</mi><mi>j</mi></mrow> <mrow><mo>(</mo><mi>l</mi><mo>)</mo></mrow></msubsup></mfenced> <mn>2</mn></msup></mrow></math>

在这里，我们平等地权衡每个平方差（通过除以我们想要包含在风格重建中的层数）。具体来说，我们使用原始 VGGNet 的`relu1_1`、`relu2_1`、`relu3_1`、`relu4_1`和`relu5_1`层。为了简洁起见，我们省略了有关 TensorFlow 代码的完整讨论，但结果如图 7-22 所示，再次非常壮观。我们混合了麻省理工学院标志性圆顶和列昂尼德·阿弗莫夫的*雨公主*的照片。

![](img/fdl2_0722.png)

###### 图 7-22。将雨公主与麻省理工学院圆顶的照片混合的结果¹⁸

# 学习卷积滤波器用于其他问题领域

尽管本章中的示例侧重于图像识别，但卷积网络在几个其他问题领域中也很有用。图像分析的自然延伸是视频分析。事实上，使用五维张量（包括时间作为一个维度）并应用三维卷积是将卷积范式扩展到视频的一种简单方法。¹⁹ 卷积滤波器还成功地用于分析听力图。²⁰ 在这些应用中，卷积网络在听力图输入上滑动，以预测另一侧的音素。

不太直观的是，卷积网络还在自然语言处理中找到了一些用途。我们将在后面的章节中看到一些例子。卷积网络的更奇特用途包括教授算法玩棋盘游戏，以及分析生物分子以进行药物发现。我们还将在本书的后面章节讨论这两个例子。

# 总结

在本章中，我们学习了如何构建分析图像的神经网络。我们发展了卷积的概念，并利用这个想法创建了可分析简单和更复杂自然图像的可操作网络。我们在 TensorFlow 中构建了几个这样的卷积网络，并利用各种图像处理流水线和批量归一化使训练网络更快速和更稳健。最后，我们可视化了卷积网络的学习过程，并探索了技术的其他有趣应用。

由于我们能够想出有效的方法将图像表示为张量，因此图像分析变得容易。在其他情况下（例如自然语言），如何将输入数据表示为张量并不太清楚。为了作为新深度学习模型的一个基石来解决这个问题，我们将在下一章中开发一些关键概念，包括向量嵌入和表示学习。

¹ Hubel, David H., and Torsten N. Wiesel. “Receptive Fields and Functional Architecture of Monkey Striate Cortex.” *The Journal of Physiology* 195.1 (1968): 215-243.

² Cohen, Adolph I. “Rods and Cones.” *Physiology of Photoreceptor Organs*. Springer Berlin Heidelberg, 1972\. 63-110.

³ Viola, Paul, and Michael Jones. “Rapid Object Detection using a Boosted Cascade of Simple Features.” Computer Vision and Pattern Recognition, 2001\. CVPR 2001\. *Proceedings of the 2001 IEEE Computer Society Conference* on. Vol. 1\. IEEE, 2001.

⁴ Deng, Jia, et al. “ImageNet: A Large-Scale Hierarchical Image Database.” *Computer Vision and Pattern Recognition*, 2009\. CVPR 2009\. IEEE Conference. IEEE, 2009.

⁵ Perronnin, Florent, Jorge Sénchez, and Yan Liu Xerox. “Large-Scale Image Categorization with Explicit Data Embedding.” *Computer Vision and Pattern Recognition* (CVPR), 2010 IEEE Conference. IEEE, 2010.

⁶ Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. “ImageNet Classification with Deep Convolutional Neural Networks.” *Advances in Neural Information Processing Systems*. 2012.

⁷ LeCun, Yann, et al. “Handwritten Digit Recognition with a Back-Propagation Network.” *Advances in Neural Information Processing Systems*. 1990.

⁸ Hubel, David H., and Torsten N. Wiesel. “Receptive Fields of Single Neurones in the Cat’s Striate Cortex.” *The Journal of Physiology* 148.3 (1959): 574-591.

⁹ Graham, Benjamin. “Fractional Max-Pooling.” *arXiv Preprint arXiv*:1412.6071 (2014).

¹⁰ Simonyan, Karen, and Andrew Zisserman. “Very Deep Convolutional Networks for Large-Scale Image Recognition.” *arXiv Preprint arXiv*:1409.1556 (2014).

¹¹ S. Ioffe, C. Szegedy. “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.” *arXiv Preprint arXiv*:1502.03167\. 2015.

¹² Wu et. al. “Group Normalization.” 2018\. *https://arxiv.org/abs/1803.08494*.

¹³ Krizhevsky, Alex, and Geoffrey Hinton. “Learning Multiple Layers of Features from Tiny Images.” University of Toronto (2009).
