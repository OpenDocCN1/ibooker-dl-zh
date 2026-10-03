
# 练习与参考资料

## 练习

1. 浏览 Keras 网站上关于[“使用双编码器进行自然语言图像搜索”](https://homl.info/dualtuto)的示例。您将学习如何构建一个能够在同一嵌入空间内表示图像和文本的模型。这使得可以使用文本提示搜索图像，就像 OpenAI 的 CLIP 模型一样。

2. 使用 Hugging Face Transformers 库下载一个预训练的语言模型，能够生成文本（例如，GPT），并尝试生成更具说服力的莎士比亚文本。您需要使用模型的`generate()`方法——请参阅 Hugging Face 的文档以获取更多详细信息。

这些练习的解决方案可在本章笔记本的末尾找到，网址为[*https://homl.info/colab3*](https://homl.info/colab3)。

## 参考文献与注释

艾伦·图灵，“计算机机器和智能”，*心灵* 49（1950 年）：433-460。

当然，单词*chatbot*出现得更晚。图灵称其测试为*模仿游戏*：机器 A 和人类 B 通过文本消息与人类审问者 C 聊天；审问者提出问题以确定哪一个是机器（A 还是 B）。如果机器能够愚弄审问者，那么它通过了测试，而人类 B 必须尽力帮助审问者。

由于输入窗口重叠，因此在这种情况下*时代*的概念并不那么清晰：在每个时代（由 Keras 实现），模型实际上会多次看到相同的字符。

Alec Radford 等人，“学习生成评论和发现情感”，arXiv 预印本 arXiv:1704.01444（2017 年）。

Rico Sennrich 等人，“使用子词单元进行稀有词的神经机器翻译”，*计算语言学年会第 54 届年会论文集* 1（2016 年）：1715-1725。

⁶ Taku Kudo，“子词规范化：改进神经网络翻译模型的多个子词候选”，arXiv 预印本 arXiv:1804.10959（2018）。

⁷ Taku Kudo 和 John Richardson，“SentencePiece：用于神经文本处理的简单且语言无关的子词标记器和去标记器”，arXiv 预印本 arXiv:1808.06226（2018）。

⁸ Yonghui Wu 等人，“谷歌的神经机器翻译系统：弥合人类和机器翻译之间的差距”，arXiv 预印本 arXiv:1609.08144（2016）。

⁹ 不规则张量在第十二章中被介绍，详细内容在附录 C 中。

¹⁰ Matthew Peters 等人，“深度上下文化的词表示”，*2018 年北美计算语言学分会年会论文集：人类语言技术* 1（2018）：2227–2237。

¹¹ Jeremy Howard 和 Sebastian Ruder，“文本分类的通用语言模型微调”，*计算语言学年会第 56 届年会论文集* 1（2018）：328–339。

¹² Daniel Cer 等人，“通用句子编码器”，arXiv 预印本 arXiv:1803.11175（2018）。

¹³ Ilya Sutskever 等人，“使用神经网络进行序列到序列学习”，arXiv 预印本（2014）。

¹⁴ Samy Bengio 等人，“使用循环神经网络进行序列预测的计划抽样”，arXiv 预印本 arXiv:1506.03099（2015）。

¹⁵ 这个数据集由[Tatoeba 项目](https://tatoeba.org)的贡献者创建的句子对组成。网站作者选择了约 120,000 个句子对[*https://manythings.org/anki*](https://manythings.org/anki)。该数据集在创作共用署名 2.0 法国许可下发布。其他语言对也可用。

¹⁶ 在 Python 中，如果运行`a, *b = [1, 2, 3, 4]`，那么`a`等于`1`，`b`等于`[2, 3, 4]`。

¹⁷ Sébastien Jean 等人，“在神经机器翻译中使用非常大的目标词汇”，*计算语言学年会第 53 届年会和亚洲自然语言处理联合国际会议第 7 届年会论文集* 1（2015）：1–10。

¹⁸ Dzmitry Bahdanau 等人，“通过联合学习对齐和翻译的神经机器翻译”，arXiv 预印本 arXiv:1409.0473（2014）。

¹⁹ Minh-Thang Luong 等人，“基于注意力的神经机器翻译的有效方法”，*2015 年自然语言处理经验方法会议论文集*（2015）：1412–1421。

²⁰ Ashish Vaswani 等人，“注意力就是一切”，*第 31 届国际神经信息处理系统会议论文集*（2017）：6000–6010。

²¹ 由于Transformer使用时间分布密集层，可以说它使用了核大小为 1 的 1D 卷积层。

²² 这是“注意力就是一切”论文中的图 1，经作者的亲切许可再现。

²³ 如果您使用最新版本的 TensorFlow，可以使用不规则张量。

²⁴ 这是“注意力机制是你所需要的一切”中图 2 的右侧部分，经作者亲切授权复制。

²⁵ 当您阅读本文时，这很可能会发生变化；请查看[Keras 问题＃16248](https://github.com/keras-team/keras/issues/16248)获取更多详细信息。当这种情况发生时，将不需要设置`attention_mask`参数，因此也不需要创建`encoder_pad_mask`。

²⁶ 目前`Z + skip`不支持自动屏蔽，这就是为什么我们不得不写`tf.keras.layers.Add()([Z, skip])`的原因。再次强调，当您阅读本文时，情况可能已经发生变化。

²⁷ Alec Radford 等人，“通过生成式预训练改进语言理解”（2018 年）。

²⁸ 例如，“简在朋友的生日派对上玩得很开心”意味着“简喜欢这个派对”，但与“每个人都讨厌这个派对”相矛盾，与“地球是平的”无关。

²⁹ Jacob Devlin 等人，“BERT：深度双向 Transformer 的预训练”，*2018 年北美计算语言学协会会议论文集：人类语言技术* 1（2019 年）。

³⁰ 这是论文中的图 1，经作者亲切授权复制。

³¹ Alec Radford 等人，“语言模型是无监督多任务学习者”（2019 年）。

³² William Fedus 等人，“Switch Transformers: 通过简单高效的稀疏性扩展到万亿参数模型”（2021 年）。

³³ Victor Sanh 等人，“DistilBERT，Bert 的精简版本：更小、更快、更便宜、更轻”，arXiv 预印本 arXiv:1910.01108（2019 年）。

³⁴ Mariya Yao 在这篇文章中总结了许多这些模型：[*https://homl.info/yaopost*](https://homl.info/yaopost)。

³⁵ Colin Raffel 等人，“探索统一文本到文本 Transformer 的迁移学习极限”，arXiv 预印本 arXiv:1910.10683（2019 年）。

³⁶ Aakanksha Chowdhery 等人，“PaLM: 使用路径扩展语言建模”，arXiv 预印本 arXiv:2204.02311（2022 年）。

³⁷ Jason Wei 等人，“思维链提示引发大型语言模型的推理”，arXiv 预印本 arXiv:2201.11903（2022 年）。

³⁸ Kelvin Xu 等人，“展示、关注和叙述：带有视觉注意力的神经图像字幕生成”，*第 32 届国际机器学习会议论文集*（2015 年）：2048–2057。

³⁹ 这是论文中图 3 的一部分。经作者亲切授权复制。

⁴⁰ Marco Tulio Ribeiro 等人，“‘为什么我应该相信你？’：解释任何分类器的预测”，*第 22 届 ACM SIGKDD 国际知识发现与数据挖掘会议论文集*（2016 年）：1135–1144。

⁴¹ Nicolas Carion 等人，“使用 Transformer 进行端到端目标检测”，arXiv 预印本 arxiv:2005.12872（2020 年）。

⁴² Alexey Dosovitskiy 等人，“一幅图像价值 16x16 个词：大规模图像识别的 Transformer”，arXiv 预印本 arxiv:2010.11929（2020 年）。

⁴³ Hugo Touvron 等人，“训练数据高效的图像 Transformer 和通过注意力蒸馏”，arXiv 预印本 arxiv:2012.12877（2020 年）。

⁴⁴ Andrew Jaegle 等人，“Perceiver: 带有迭代注意力的通用感知”，arXiv 预印本 arxiv:2103.03206（2021）。

⁴⁵ Mathilde Caron 等人，“自监督视觉 Transformer 中的新兴属性”，arXiv 预印本 arxiv:2104.14294（2021）。

⁴⁶ Xiaohua Zhai 等人，“缩放视觉 Transformer”，arXiv 预印本 arxiv:2106.04560v1（2021）。

⁴⁷ Mitchell Wortsman 等人，“模型汤：多个微调模型的权重平均提高准确性而不增加推理时间”，arXiv 预印本 arxiv:2203.05482v1（2022）。

⁴⁸ Alec Radford 等人，“从自然语言监督中学习可转移的视觉模型”，arXiv 预印本 arxiv:2103.00020（2021）。

⁴⁹ Aditya Ramesh 等人，“零样本文本到图像生成”，arXiv 预印本 arxiv:2102.12092（2021）。

⁵⁰ Aditya Ramesh 等人，“具有 CLIP 潜变量的分层文本条件图像生成”，arXiv 预印本 arxiv:2204.06125（2022）。

⁵¹ Jean-Baptiste Alayrac 等人，“Flamingo：用于少样本学习的视觉语言模型”，arXiv 预印本 arxiv:2204.14198（2022）。

⁵² Scott Reed 等人，“通用主体代理”，arXiv 预印本 arxiv:2205.06175（2022）。
