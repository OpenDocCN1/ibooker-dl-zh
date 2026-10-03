
^(14) 也有一些模型生成词嵌入，而不是文档嵌入，例如`word2vec`（Mikolov 等人，[“Efficient Estimation of Word Representations in Vector Space”](https://arxiv.org/abs/1301.3781)，*arXiv*，v3，2013 年 9 月 7 日）和`GloVe`（Pennington 等人，[“GloVe: Global Vectors for Word Representation”](https://oreil.ly/O5QTX)，斯坦福大学自然语言处理组（博客），2014 年）。

^(15) “AI 裁判”这个术语不要与 AI 在法庭上作为裁判的使用场景混淆。

^(16) 2017 年，我在 NeurIPS 研讨会[MEWR](https://x.com/chipro/status/937384141791698944)（无参考文本的机器翻译评估指标）上发表了演讲，这是一种利用更强的语言模型来自动评估机器翻译的方法。遗憾的是，我从未继续这条研究路线，因为生活让我分心。

^(17) 在某些情况下，评估可能占据预算的大部分，甚至超过响应生成。

^(18) 检查点与抽样相同。

^(19) Saito 等人（2023）发现，人类也倾向于更喜欢较长的响应，但程度要小得多。

^(20) 这种技术有时被称为*自我批评*或*自我提问*。

^(21) `BLEURT`分数的范围令人困惑。它大约在[-2.5 和 1.0](https://github.com/google-research/bleurt/issues/1)之间。这突显了 AI 裁判中标准模糊性的挑战：分数范围可以是任意的。

^(22) 例如使用[李克特量表](https://en.wikipedia.org/wiki/Likert_scale)。

^(23) 即使 Chatbot Arena 停止使用 Elo 评级算法，其开发者在一段时间内仍然继续将他们的模型评级称为“Elo 分数”。他们通过缩放得到的 Bradley-Terry 分数，使其看起来像 Elo 分数。这种缩放相当复杂。每个分数乘以 400（Elo 使用的比例）并加到 1000（初始 Elo 分数）。然后对这个分数进行重新缩放，使得 Llama-13b 模型的分数为 800。

^(24) 随着 Chatbot Arena 越来越受欢迎，试图操纵它的情况也变得越来越常见。虽然没有人向我承认他们试图操纵排名，但几位模型开发者告诉我，他们相信他们的竞争对手试图操纵它。
