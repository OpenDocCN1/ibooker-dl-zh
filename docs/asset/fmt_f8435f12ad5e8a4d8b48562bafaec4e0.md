
# 附录 B 参考文献及进一步阅读

## 第一章

如彭博团队通过从零开始预训练金融数据的 GPT 版本所展示，定制构建的 LLM 能够超越通用 LLM，同时在通用 LLM 基准测试中保持良好的性能：

+ Wu 等人撰写的《“BloombergGPT：用于金融的大型语言模型”》（2023），[`arxiv.org/abs/2303.17564`](https://arxiv.org/abs/2303.17564)

现有的 LLM 可以进行适配和微调，以超越通用 LLM，谷歌研究和谷歌 DeepMind 团队在医疗环境中展示了这一点：

+ Singhal 等人撰写的《“使用大型语言模型实现专家级医学问答”》（2023），[`arxiv.org/abs/2305.09617`](https://arxiv.org/abs/2305.09617)

以下论文提出了原始的 Transformer 架构：

+ Vaswani 等人撰写的《“注意力即一切”》（2017），[`arxiv.org/abs/1706.03762`](https://arxiv.org/abs/1706.03762)

关于原始编码器风格的 Transformer，即 BERT，请参阅：

+ Devlin 等人撰写的《“BERT：用于语言理解的深度双向 Transformer 的预训练”》（2018），[`arxiv.org/abs/1810.04805`](https://arxiv.org/abs/1810.04805)

描述解码器风格的 GPT-3 模型的论文，该模型启发了现代 LLM，并将作为本书中从头实现 LLM 的模板，该论文是：

+ Brown 等人撰写的《“语言模型是少样本学习者”》（2020），[`arxiv.org/abs/2005.14165`](https://arxiv.org/abs/2005.14165)

以下涵盖了用于图像分类的原版视觉 Transformer，这表明 Transformer 架构不仅限于文本输入：

+ Dosovitskiy 等人撰写的《“一张图片等于 16x16 个单词：大规模图像识别的 Transformer”》（2020），[`arxiv.org/abs/2010.11929`](https://arxiv.org/abs/2010.11929)

以下实验（但不太流行）的 LLM 架构作为例子，说明并非所有 LLM 都必须基于 Transformer 架构：

+ Peng 等人撰写的《“RWKV：为 Transformer 时代重新发明 RNN”》（2023），[`arxiv.org/abs/2305.13048`](https://arxiv.org/abs/2305.13048)

+ Poli 等人撰写的《“鬣狗等级：向更大型的卷积语言模型迈进”》（2023），[`arxiv.org/abs/2302.10866`](https://arxiv.org/abs/2302.10866)

+ Gu 和 Dao 撰写的《“Mamba：具有选择性状态空间的线性时间序列建模”》（2023），[`arxiv.org/abs/2312.00752`](https://arxiv.org/abs/2312.00752)

Meta AI 的模型是 GPT 类模型的一种流行实现，与 GPT-3 和 ChatGPT 相比，它是公开可用的：

+ Touvron 等人撰写的《“Llama 2：开放基础和微调的聊天模型”》（2023），[`arxiv.org/abs/2307.092881`](https://arxiv.org/abs/2307.092881)

对于对第 1.5 节中数据集引用的详细信息感兴趣的读者，本文描述了由 Eleuther AI 精心整理的公开可用 *The Pile* 数据集：

+ “The Pile：用于语言建模的 800GB 多样化文本数据集”（2020）由 Gao 等人所著，[`arxiv.org/abs/2101.00027`](https://arxiv.org/abs/2101.00027)

以下论文提供了 InstructGPT 微调 GPT-3 的参考，这在第 1.6 节中提到，将在第七章中更详细地讨论：

+ “通过人类反馈训练语言模型以遵循指令”（2022）由 Ouyang 等人所著，[`arxiv.org/abs/2203.02155`](https://arxiv.org/abs/2203.02155)

## 第二章

对于那些对讨论和比较嵌入空间与潜在空间以及向量表示的通用概念感兴趣的读者，可以在我的书的第一章中找到更多信息：

+ 机器学习 Q 和 AI（2023）由 Sebastian Raschka 所著，[`leanpub.com/machine-learning-q-and-ai`](https://leanpub.com/machine-learning-q-and-ai)

以下论文更深入地讨论了字节对编码作为分词方法的使用：

+ “使用子词单元进行罕见词的神经机器翻译”（2015）由 Sennrich 等人所著，[`arxiv.org/abs/1508.07909`](https://arxiv.org/abs/1508.07909)

OpenAI 开源了用于训练 GPT-2 的字节对编码分词器的代码：

+ [`github.com/openai/gpt-2/blob/master/src/encoder.py`](https://github.com/openai/gpt-2/blob/master/src/encoder.py)

OpenAI 提供了一个交互式的 Web 用户界面，以展示 GPT 模型中的字节对分词器是如何工作的：

+ [`platform.openai.com/tokenizer`](https://platform.openai.com/tokenizer)

对于那些对从零开始编码和训练 BPE 分词器感兴趣的读者，Andrej Karpathy 的 GitHub 仓库`minbpe`提供了一个最小化和可读的实现：

+ “一个 BPE 分词器的最小实现”，[`github.com/karpathy/minbpe`](https://github.com/karpathy/minbpe)

对于那些对研究某些其他流行 LLM 使用的替代分词方案感兴趣的读者，可以在 SentencePiece 和 WordPiece 论文中找到更多信息：

+ “SentencePiece：用于神经文本处理的一个简单且语言无关的子词分词器和反分词器”（2018）由 Kudo 和 Richardson 所著，[`aclanthology.org/D18-2012/`](https://aclanthology.org/D18-2012/)

+ “快速 WordPiece 分词”（2020）由 Song 等人所著，[`arxiv.org/abs/2012.15524`](https://arxiv.org/abs/2012.15524)

## 第三章

对于那些想要了解更多关于 RNN 和语言翻译中 Bahdanau 注意力机制的读者，可以在以下论文中找到详细见解：

+ “一个 BPE 分词器的最小实现”，[`github.com/karpathy/minbpe`](https://github.com/karpathy/minbpe)

自注意力作为缩放点积注意力在原始的 Transformer 论文中被引入：

+ “Attention Is All You Need”（2017）由 Vaswani 等人所著，[`arxiv.org/abs/1706.03762`](https://arxiv.org/abs/1706.03762)

FlashAttention 是自注意力机制的高效实现，通过优化内存访问模式加速计算过程。从数学上讲，FlashAttention 与标准自注意力机制相同，但优化了计算过程以提高效率：

+ “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness” (2022) by Dao et al., [`arxiv.org/abs/2205.14135`](https://arxiv.org/abs/2205.14135)

+ “FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning” (2023) by Dao, [`arxiv.org/abs/2307.08691`](https://arxiv.org/abs/2307.08691)

PyTorch 实现了一个支持 FlashAttention 以提高效率的自注意力和因果注意力的函数。此函数处于测试阶段，可能会发生变化：

+ `scaled_dot_product_attention`文档：[`mng.bz/NRJd`](https://mng.bz/NRJd)

PyTorch 也实现了一个基于`scaled_dot_product`函数的高效的`MultiHeadAttention`类：

+ `MultiHeadAttention`文档：[`mng.bz/DdJV`](https://mng.bz/DdJV)

Dropout 是一种在训练过程中随机从神经网络中丢弃单元（及其连接）的正则化技术，以防止过拟合：

+ “Dropout: A Simple Way to Prevent Neural Networks from Overfitting” (2014) by Srivastava et al., [`jmlr.org/papers/v15/srivastava14a.html`](https://jmlr.org/papers/v15/srivastava14a.html)

虽然在实践中最常见的自注意力变体是基于缩放点积注意力的多头注意力，但作者发现，即使没有值权重矩阵和投影层，也可以实现良好的性能：

+ “Simplifying Transformer Blocks” (2023) by He and Hofmann, [`arxiv.org/abs/2311.01906`](https://arxiv.org/abs/2311.01906)

## 第四章

以下论文介绍了一种通过标准化隐藏层中神经元的总和输入来稳定隐藏状态动态的神经网络的技术，与之前发表的方法相比，显著减少了训练时间：

+ “Layer Normalization” (2016) by Ba, Kiros, and Hinton, [`arxiv.org/abs/1607.06450`](https://arxiv.org/abs/1607.06450)

Post-LayerNorm，在原始 Transformer 模型中使用，在自注意力和前馈网络之后应用层归一化。相比之下，Pre-LayerNorm，如 GPT-2 和更新的 LLMs 中采用，在这些组件之前应用层归一化，这可能导致更稳定的训练动态，并且已经证明在某些情况下可以提高性能，如以下论文所述：

+ “On Layer Normalization in the Transformer Architecture” (2020) by Xiong et al., [`arxiv.org/abs/2002.04745`](https://arxiv.org/abs/2002.04745)

+ “ResiDual: Transformer with Dual Residual Connections” (2023) by Tie et al., [`arxiv.org/abs/2304.14802`](https://arxiv.org/abs/2304.14802)

在现代 LLM 中使用的 LayerNorm 的一个流行变体是 RMSNorm，因为它提高了计算效率。这种变体通过仅使用输入的均方根来归一化输入，而不在平方之前减去均值，从而简化了归一化过程。这意味着在计算尺度之前不会对数据进行中心化。RMSNorm 在以下内容中描述得更详细：

+ “均方根层归一化”（2019）由 Zhang 和 Sennrich 撰写，[`arxiv.org/abs/1910.07467`](https://arxiv.org/abs/1910.07467)

高斯误差线性单元（GELU）激活函数结合了经典 ReLU 激活函数和正态分布的累积分布函数的特性，以建模层输出，允许在深度学习模型中进行随机正则化和非线性：

+ “高斯误差线性单元（GELUs）”（2016）由 Hendricks 和 Gimpel 撰写，[`arxiv.org/abs/1606.08415`](https://arxiv.org/abs/1606.08415)

GPT-2 论文介绍了一系列基于 transformer 的、大小各异的 LLM——1.24 亿、3.55 亿、7.74 亿和 15 亿个参数：

+ “语言模型是无监督多任务学习者”（2019）由 Radford 等人撰写，[`mng.bz/DMv0`](http://mng.bz/DMv0)

OpenAI 的 GPT-3 基本上使用了与 GPT-2 相同的架构，除了最大的版本（1750 亿）比最大的 GPT-2 模型大 100 倍，并且使用了更多的数据进行训练。感兴趣的读者可以参考 OpenAI 的官方 GPT-3 论文以及 Lambda Labs 的技术概述，该概述计算在单个 RTX 8000 消费级 GPU 上训练 GPT-3 需要 665 年：

+ “语言模型是少样本学习者”（2023）由 Brown 等人撰写，[`arxiv.org/abs/2005.14165`](https://arxiv.org/abs/2005.14165)

+ “OpenAI 的 GPT-3 语言模型：技术概述”，[`lambdalabs.com/blog/demystifying-gpt-3`](https://lambdalabs.com/blog/demystifying-gpt-3)

NanoGPT 是一个代码仓库，它以简约而高效的方式实现了 GPT-2 模型，类似于本书中实现的模型。虽然本书中的代码与 nanoGPT 不同，但这个仓库激发了将大型 GPT Python 父类实现重组为较小子模块的灵感：

+ “NanoGPT，一个用于训练中等大小 GPT 的仓库，[`github.com/karpathy/nanoGPT`](https://github.com/karpathy/nanoGPT)

一篇信息丰富的博客文章展示了，当上下文大小小于 32,000 个标记时，大多数 LLM 的计算都是在前馈层而不是注意力层中进行的：

+ “在长（上下文）运行中”由 Harm de Vries 撰写，[`www.harmdevries.com/post/context-length/`](https://www.harmdevries.com/post/context-length/)

## 第五章

关于详细说明损失函数和应用对数变换以使其更容易进行数学优化的信息，请参阅我的讲座视频：

+ L8.2 逻辑回归损失函数，[`www.youtube.com/watch?v=GxJe0DZvydM`](https://www.youtube.com/watch?v=GxJe0DZvydM)

作者提供的以下讲座和代码示例解释了 PyTorch 的交叉熵函数在底层是如何工作的：

+ L8.7.1 OneHot 编码和多类别交叉熵，[`www.youtube.com/watch?v=4n71-tZ94yk`](https://www.youtube.com/watch?v=4n71-tZ94yk)

+ 理解 PyTorch 中的 Onehot 编码和交叉熵，[`mng.bz/o05v`](https://mng.bz/o05v)

以下两篇论文详细介绍了用于预训练 LLM 的数据集、超参数和架构细节：

+ “Pythia：用于分析大型语言模型训练和扩展的套件”（2023）由 Biderman 等人撰写，[`arxiv.org/abs/2304.01373`](https://arxiv.org/abs/2304.01373)

+ “OLMo：加速语言模型科学”（2024）由 Groeneveld 等人撰写，[`arxiv.org/abs/2402.00838`](https://arxiv.org/abs/2402.00838)

以下为本书提供的补充代码，包含从 Project Gutenberg 项目准备 60,000 本公共领域书籍用于 LLM 训练的说明：

+ 在 Project Gutenberg 数据集上预训练 GPT，[`mng.bz/Bdw2`](https://mng.bz/Bdw2)

第五章讨论了 LLM 的预训练，附录 D 涵盖了更高级的训练函数，例如线性预热和余弦退火。以下论文发现，类似的技巧可以成功应用于继续预训练已经预训练的 LLM，并附带额外的技巧和见解：

+ “简单且可扩展的策略以持续预训练大型语言模型”（2024）由 Ibrahim 等人撰写，[`arxiv.org/abs/2403.08763`](https://arxiv.org/abs/2403.08763)

BloombergGPT 是通过对通用和特定领域文本语料库进行训练创建的特定领域 LLM 的示例，特别是在金融领域：

+ “BloombergGPT：用于金融的 LLM”（2023）由 Wu 等人撰写，[`arxiv.org/abs/2303.17564`](https://arxiv.org/abs/2303.17564)

GaLore 是一个旨在使 LLM 预训练更高效的研究项目。所需的代码更改仅限于将训练函数中的 PyTorch 的`AdamW`优化器替换为`galore-torch`Python 包提供的`GaLoreAdamW`优化器：

+ “GaLore：通过梯度低秩投影提高 LLM 训练效率”（2024）由 Zhao 等人撰写，[`arxiv.org/abs/2403.03507`](https://arxiv.org/abs/2403.03507)
