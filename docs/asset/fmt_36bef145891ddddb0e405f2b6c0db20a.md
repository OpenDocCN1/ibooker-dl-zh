
![在 Colab 中运行的 GitHub 笔记本的屏幕截图](img/00278.jpeg)

###### 图 1-14。在 Colab 中运行的 GitHub 笔记本的屏幕截图

1. Colab 很棒，但我已经有一台为玩<插入视频游戏名称>而购买的强大计算机。我应该如何设置我的环境？

   理想的设置涉及 Linux，但 Windows 和 macOS 也可以。对于大多数章节，您需要以下内容：

   + Python 3 和 PIP
   + `tensorflow`或`tensorflow-gpu` PIP 包（版本 2 或更高）
   + Pillow

   我们喜欢保持事情整洁和自包含，因此我们建议使用 Python 虚拟环境。每当您安装软件包或运行脚本或笔记本时，都应该使用虚拟环境。

   如果您没有 GPU，您的设置就完成了。

   如果您有 NVIDIA GPU，您需要安装适当的驱动程序，然后安装 CUDA，然后安装 cuDNN，然后安装`tensorflow-gpu`软件包。如果您使用 Ubuntu，有一个比手动安装这些软件包更简单的解决方案，即使用[Lambda Stack](https://oreil.ly/93oon)一行代码安装整个环境。

   或者，您可以使用 Anaconda Distribution 安装所有软件包，这对 Windows、Mac 和 Linux 同样有效。

2. 我在哪里可以找到本书中使用的代码？

   您可以在[*http://PracticalDeepLearning.ai*](http://PracticalDeepLearning.ai)找到可直接运行的示例。

3. 阅读本书的最低先决条件是什么？

   博士学位包括微积分、统计分析、变分自动编码器、运筹学等领域...绝对*不*是必需的，才能阅读这本书（让您有点紧张，是吧？）。一些基本的编码技能、熟悉 Python、健康的好奇心和幽默感应该会在吸收材料的过程中大有裨益。尽管初级水平的移动开发理解（使用 Swift 和/或 Kotlin）会有所帮助，但我们设计的示例是自给自足且足够简单，可以由以前从未编写过移动应用的人部署。

4. 我们将使用哪些框架？

   Keras + TensorFlow 用于训练。逐章探讨不同的推理框架。

5. 我读完这本书后会成为专家吗？

   如果您跟着学习，您将掌握从训练到推理再到性能最大化等各种主题的知识。尽管这本书主要关注计算机视觉，您也可以将相同的知识应用到其他领域，如文本、音频等，并且能够迅速上手。

6. 本章前面的猫是谁？

   那是 Meher 的猫，Vader。他将在本书中多次客串。不用担心，他已经签署了模特释放表。

7. 我可以联系您吗？

   当然。如果有任何问题、更正或其他问题，请给我们发送电子邮件至 PracticalDLBook@gmail.com，或在 Twitter 上@PracticalDLBook 发推文。

如果您正在阅读盗版副本，请考虑我们对您感到失望。

² [Robert Geirhos 等人](https://arxiv.org/pdf/1811.12231.pdf)
