# 附录 A 参考文献与延伸阅读

## A.1 第 1 章：理解推理模型

### A.1.1 参考文献

这是 OpenAI o1 模型的公告文章，o1 被认为是首个基于 LLM 的推理模型：

* OpenAI，“OpenAI o1-preview 简介”（2024 年 9 月 12 日），[`openai.com/index/introducing-openai-o1-preview/`](https://openai.com/index/introducing-openai-o1-preview/)

DeepSeek-R1 是首个开源推理模型，并附带了一份全面的技术报告，表明推理来自具有可验证奖励的强化学习（这一主题将在第 5 章中更详细地讨论）：

* DeepSeek-AI、Daya Guo、Dejian Yang 等，“DeepSeek-R1：通过强化学习激励 LLM 中的推理能力”（2025 年 1 月 22 日），[`arxiv.org/abs/2501.12948`](https://arxiv.org/abs/2501.12948)

这是 OpenAI CEO 对未来模型推理（“思维链”）能力的评论：“我们接下来将发布 GPT-4.5，这是我们在内部称为 Orion 的模型，也是我们最后一个非思维链模型。”

* Sam Altman，X 平台上的帖子（2025 年 2 月 12 日），[`x.com/sama/status/1889755723078443244`](https://x.com/sama/status/1889755723078443244)

苹果公司 AI 研究人员的一篇研究论文发现，推理模型是复杂（但能力很强）的模式匹配器：

* Parshin Shojaee、Iman Mirzadeh、Keivan Alizadeh 等，“思考的错觉：从问题复杂性的视角理解推理模型的优势与局限”（2025 年 6 月），[`machinelearning.apple.com/research/illusion-of-thinking`](https://machinelearning.apple.com/research/illusion-of-thinking)

以下这本书是逐步实现和训练大语言模型的深入指南：

* Sebastian Raschka，*从零开始构建大语言模型*（Manning，2024），[`mng.bz/orYv`](http://mng.bz/orYv)

### A.1.2 延伸阅读

以下文章介绍了 DeepSeek-R1 的工作原理，深入探讨了 LLM 中推理的基础：

* Sebastian Raschka，“理解推理 LLM”（2025 年 2 月 5 日），[`magazine.sebastianraschka.com/p/understanding-reasoning-llms`](https://magazine.sebastianraschka.com/p/understanding-reasoning-llms)

## A.2 第 2 章：使用预训练 LLM 生成文本

### A.2.1 参考文献

你可以在以下 URL 找到 `uv` Python 包和项目管理器的官方安装页面：

* “安装 `uv`”，[`docs.astral.sh/uv/getting-started/installation/`](https://docs.astral.sh/uv/getting-started/installation/)

以下是两个用户友好且流行的支持 GPU 的云计算平台：

* Lightning AI，[`lightning.ai/`](https://lightning.ai/)
* Google Colab，[`colab.research.google.com/`](https://colab.research.google.com/)

以下文章提供了 Qwen3 资源，包括额外的基准测试结果以及与其他模型的比较：

* “Qwen3：思考更深，行动更快”（2025 年 4 月 28 日），[`qwen.ai/blog?id=qwen3`](https://qwen.ai/blog?id=qwen3)
* An Yang、Anfeng Li、Baosong Yang 等，“Quen3 技术报告”（2025 年 5 月 14 日），[`arxiv.org/abs/2505.09388`](https://arxiv.org/abs/2505.09388)

如果你对不同序列长度的 KV 缓存大小感到好奇，可以在这里找到一个便捷的计算器应用：

* KV 缓存大小计算器，[`lmcache.ai/kv\_cache\_calculator.html`](https://lmcache.ai/kv_cache_calculator.html)

### A.2.2 延伸阅读

这是一个 PyTorch 教程，适合刚接触 PyTorch 或想复习的读者：

* Sebastian Raschka，“一小时 PyTorch：从张量到在多 GPU 上训练神经网络教程”（2025 年 7 月 1 日），[`sebastianraschka.com/teaching/pytorch-1h`](https://sebastianraschka.com/teaching/pytorch-1h)

以下是关于分词（tokenization）的其他资源：

* Sebastian Raschka，*从零开始构建大语言模型*，第 2 章（Manning，2024），[`mng.bz/M96o`](https://mng.bz/M96o)
* Sebastian Raschka，“从零开始实现字节对编码（BPE）分词器”（2025 年 1 月 17 日），[`sebastianraschka.com/blog/2025/bpe-from-scratch.html`](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html)

如果你对更深入的 PyTorch 内容感兴趣，我推荐以下两本书：

* Howard Huang、Eli Stevens、Luca Antiga 和 Thomas Viehmann，*使用 PyTorch 进行深度学习*（Manning，2026），[`www.manning.com/books/deep-learning-with-pytorch-second-edition`](https://www.manning.com/books/deep-learning-with-pytorch-second-edition)
* Sebastian Raschka、Yuxi (Hayden) Liu 和 Vahid Mirjalili，*使用 PyTorch 和 Scikit-Learn 进行机器学习*（Packt Publishing，2022），[`www.amazon.com/Machine-Learning-PyTorch-Scikit-Learn-learning/dp/1801819319/`](https://www.amazon.com/Machine-Learning-PyTorch-Scikit-Learn-learning/dp/1801819319/)

## A.3 第 3 章：评估推理模型

### A.3.1 参考文献

MATH-500 数据集源自 MATH 数据集，该数据集由以下论文提出，包含代数、几何、概率、数论等领域的 12,500 个问题：

* Dan Hendrycks、Collin Burns、Saurav Kadavath 等，“使用 MATH 数据集衡量数学问题求解能力”（2021 年 3 月 5 日），[`arxiv.org/abs/2103.03874`](https://arxiv.org/abs/2103.03874)

MATH-500 划分（由原始 MATH 数据集创建）由以下论文提出：

* Hunter Lightman、Vineet Kosaraju、Yura Burda 等，“让我们逐步验证”（2023 年 5 月 31 日），[`arxiv.org/abs/2305.20050`](https://arxiv.org/abs/2305.20050)

### A.3.2 延伸阅读

如果你对 SymPy 感兴趣，这是一个用于数学和符号计算的 Python 库（本书不需要用到），你可以通过以下官方教程了解更多：

* SymPy，“入门教程”，[`docs.sympy.org/latest/tutorials/intro-tutorial/index.html`](https://docs.sympy.org/latest/tutorials/intro-tutorial/index.html)

以下文章展示了一个系统（此处为微调后的 LLM）的示例，该系统还会评估中间推理步骤：

* Shijie Xia、Xuefeng Li、Yixin Liu 等，“超越准确率评估数学推理”，第 61 届计算语言学协会年会论文集，第 1 卷：长论文（2023），6858–6877，[`arxiv.org/pdf/2404.05692`](https://arxiv.org/pdf/2404.05692)

以下是一个大规模数据集，包含 80 万个针对 MATH 数据集问题模型生成解的步骤级正确性标签：

* Hunter Lightman、Vineet Kosaraju、Yura Burda 等，“让我们逐步验证”（2023 年 5 月 31 日），[`arxiv.org/abs/2305.20050`](https://arxiv.org/abs/2305.20050)

一篇描述 LLM 评估成本不断上升的文章发现，在七个流行基准上评估 o1 等推理模型的成本约为 1,500 美元：

* Kyle Wiggers，“AI‘推理’模型的崛起正在让基准测试变得更加昂贵”（2025 年 4 月 10 日），[`techcrunch.com/2025/04/10/the-rise-of-ai-reasoning-models-is-making-benchmarking-more-expensive/`](https://techcrunch.com/2025/04/10/the-rise-of-ai-reasoning-models-is-making-benchmarking-more-expensive/)

以下是一份 2025 年关于 LLM 基准的全面综述：

* Shiwen Ni、Guhong Chen、Shuaimin Li 等，“大语言模型基准综述”（2025 年 8 月 21 日），[`arxiv.org/abs/2508.15361`](https://arxiv.org/abs/2508.15361)

一项近期研究项目表明，与其仅仅依赖确定性和符号化的验证器，小型推理模型本身也可以成功地用作其他推理模型的验证器：

* Ding Chen, Qingchen Yu, Pengyuan Wang, et al., “xVerify：用于推理模型评估的高效答案验证器”（2025 年 4 月 14 日），[`arxiv.org/abs/2504.10481`](https://arxiv.org/abs/2504.10481)

## A.4 第 4 章：通过推理时扩展提升推理能力

### A.4.1 参考文献

以下论文正式描述了思维链提示。请注意，该论文建议将提示词修改为“让我们一步一步思考”。在我的实验中，我发现使用 Qwen3 基础模型时，“逐步解释”的效果更好，这就是我们在第 4 章中使用它的原因：

* Takeshi Kojima, Shixiang Shane Gu, Machel Reid, et al., “大型语言模型是零样本推理器”（2022 年 5 月 24 日），[`arxiv.org/abs/2205.11916`](https://arxiv.org/abs/2205.11916)

以下论文重点研究了自洽性采样，并进行了额外的对比研究：

* Xuezhi Wang, Jason Wei, Dale Schuurmans, et al., “自洽性改善了语言模型中的思维链推理”（2022 年 3 月 21 日），[`arxiv.org/abs/2203.11171`](https://arxiv.org/abs/2203.11171)

### A.4.2 延伸阅读

以下文章对其他推理扩展方法进行了概述和讨论：

* Sebastian Raschka, “LLM 推理模型的推理现状”（2025 年 3 月 8 日），[`magazine.sebastianraschka.com/p/state-of-llm-reasoning-and-inference-scaling`](https://magazine.sebastianraschka.com/p/state-of-llm-reasoning-and-inference-scaling)

## A.5 第 5 章：通过自我精炼实现推理时扩展

### A.5.1 参考文献

谷歌将其专有 Gemini 3 模型背后的方法保密，但根据最近的一项公告，我们可以推测它使用了类似于自洽性或 best-of-`N` 的推理扩展技术：“我们正通过 Gemini 3 Deep Think 进一步突破智能的边界。该模式通过同时探索多个假设来解决问题，从而显著提高推理能力。”

* Google DeepMind（Gemini 的开发者）于 X 平台发布的公开公告（2025 年 12 月 4 日），https://x.com/GoogleDeepMind/status/1996658401233842624?s=20

DeepSeekMath-V2 论文表明，自洽性扩展可以显著提高答案的准确性，并且通过将自洽性与他们的自我精炼版本（在以下论文的图 2 中称为 `Best@32`）相结合，该模型在多项数学竞赛中取得了金牌级别的表现：

* Zhihong Shao, Yuxiang Luo, Chengda Lu, et al., “DeepSeekMath-V2：迈向可自我验证的数学推理”（2025 年 11 月 27 日），[`arxiv.org/abs/2511.22570v1`](https://arxiv.org/abs/2511.22570v1)

与其在自洽性中使用多数投票，我们可以使用评分函数对不同答案进行排序并选择最佳答案。这种方法被称为 best-of-`N`。不过，在适用的情况下，多数投票通常会给出更好的结果：

* Junlin Wang, Shang Zhu, Jon Saad-Falcon, et al., “深入思考，快速思考：探究无验证器的推理时扩展方法的效率”（2025 年 4 月 18 日），[`arxiv.org/abs/2504.14047`](https://arxiv.org/abs/2504.14047)

### A.5.2 延伸阅读

以下短文解释了概率与似然之间的区别：

* Sebastian Raschka, “似然和概率有什么区别？” [`sebastianraschka.com/faq/docs/probability-vs-likelihood.html`](https://sebastianraschka.com/faq/docs/probability-vs-likelihood.html)

## A.6 第 6 章：使用强化学习训练推理模型

### A.6.1 参考文献

InstructGPT 论文证明了 RLHF 的有效性，并在将 RLHF 推广为 LLM 的标准对齐和微调方法方面发挥了重要作用：

* Long Ouyang, Jeff Wu, Xu Jiang, et al., “通过人类反馈训练语言模型遵循指令”（2022 年 3 月 4 日），[`arxiv.org/abs/2203.02155`](https://arxiv.org/abs/2203.02155)

以下 DeepSeekMath 论文介绍了 GRPO 算法：

* Zhihong Shao, Peiyi Wang, Qihao Zhu, et al., “DeepSeekMath：推动开放语言模型中的数学推理极限”（2024 年 2 月 5 日），[`arxiv.org/abs/2402.03300`](https://arxiv.org/abs/2402.03300)

DeepSeek-R1 论文表明，仅通过强化学习（即采用 GRPO 的 RLVR），LLM 中就能涌现出强大的推理行为。这一点在 R1-Zero 变体中体现得最为明显，但将该方法与多阶段训练流程相结合，可以产生更好的推理模型：

* DeepSeek-AI, Daya Guo, Dejian Yang, et al., “DeepSeek-R1：通过强化学习激励 LLM 的推理能力”（2025 年 1 月 22 日），[`arxiv.org/abs/2501.12948`](https://arxiv.org/abs/2501.12948)

### A.6.2 延伸阅读

以下文章对涉及 RLVR 的 DeepSeek-R1 训练流程进行了全面讲解：

* Sebastian Raschka, “理解推理 LLM：构建和优化推理模型的方法与策略”（2025 年 2 月 5 日），[`magazine.sebastianraschka.com/p/understanding-reasoning-llms`](https://magazine.sebastianraschka.com/p/understanding-reasoning-llms)

这篇文章比较了在 LLM 上下文中用于强化学习的 GRPO 和 PPO：

* Sebastian Raschka, “LLM 推理的强化学习现状：理解 GRPO 以及来自推理模型论文的新见解”（2025 年 4 月 19 日），[`magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training`](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training)

## A.7 第 7 章：针对强化学习改进 GRPO

### A.7.1 参考文献

这是引入裁剪策略比率的原始 PPO 论文，我们在此也使用它来稳定 GRPO：

* John Schulman, Filip Wolski, Prafulla Dhariwal, et al., “近端策略优化算法”（2017 年 7 月 20 日），[`arxiv.org/abs/1707.06347`](https://arxiv.org/abs/1707.06347)

以下论文对 GRPO 算法提出了改进建议：

* Qiying Yu, Zheng Zhang, Ruofei Zhu, et al., “DAPO：一个大规模开源 LLM 强化学习系统”（2025 年 3 月 18 日），[`arxiv.org/abs/2503.14476`](https://arxiv.org/abs/2503.14476)
* “理解类似 R1-Zero 的训练：一个批判性视角”（2025 年 3 月 26 日，本文介绍了 Dr. GRPO），[`arxiv.org/abs/2503.20783`](https://arxiv.org/abs/2503.20783)
* Feng Yao, Liyuan Liu, Dinghuai Zhang, et al., “你高效的 RL 框架暗地里带来了离策略 RL 训练”（2025 年 8 月 5 日，本文介绍了 VERL），[`fengyao.notion.site/off-policy-rl`](https://fengyao.notion.site/off-policy-rl)
* DeepSeek-AI, Aixin Liu, Aoxue Mei, et al., “DeepSeek-V3.2：推动开放大语言模型的前沿”（2025 年 12 月 2 日），[`arxiv.org/abs/2512.02556`](https://arxiv.org/abs/2512.02556)
* Shih-Yang Liu, Xin Dong, Ximing Lu, et al., “GDPO：面向多奖励 RL 优化的组奖励解耦归一化策略优化”（2026 年 1 月 8 日），[`arxiv.org/abs/2601.05242`](https://arxiv.org/abs/2601.05242)
* Chujie Zheng, Shixuan Liu, Mingze Li, et al., “组序列策略优化”（2025 年 7 月 24 日，本文介绍了 GSPO），[`arxiv.org/abs/2507.18071`](https://arxiv.org/abs/2507.18071)
* Aili Chen, Aonian Li, Bangwei Gong, et al., “MiniMax-M1：利用 Lightning Attention 高效扩展测试时计算”（2025 年 6 月 16 日，本文介绍了 CISPO），[`arxiv.org/abs/2506.13585`](https://arxiv.org/abs/2506.13585)

### A.7.2 延伸阅读

以下文章对 PPO（最初用于 RLHF 的算法）和 GRPO 进行了比较：

* Sebastian Raschka，《大型语言模型推理的强化学习现状》(2025 年 4 月 19 日)，[`magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training`](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training)

以下文章是一篇很好的技术深度剖析，讨论了各种 GRPO 改进：

* Cameron R. Wolfe，《GRPO++：让强化学习真正起作用的技巧》(2026 年 1 月 5 日)，[`cameronrwolfe.substack.com/p/grpo-tricks`](https://cameronrwolfe.substack.com/p/grpo-tricks)

## A.8 第 8 章：蒸馏推理模型以实现高效推理

### A.8.1 参考文献

这是开创性的知识蒸馏论文，推广了硬蒸馏与软蒸馏目标相结合的方法：

* Geoffrey Hinton, Oriol Vinyals, and Jeff Dean，《蒸馏神经网络中的知识》(2015 年 3 月 9 日)，[`arxiv.org/abs/1503.02531`](https://arxiv.org/abs/1503.02531)

DeepSeek-R1 论文描述了启发本章的推理蒸馏方法，即大型教师模型生成推理轨迹，然后用这些轨迹训练较小的学生模型：

* DeepSeek-AI, Daya Guo, Dejian Yang, et al.，《DeepSeek-R1：通过强化学习激励大型语言模型的推理能力》(2025 年 1 月 22 日)，[`arxiv.org/abs/2501.12948`](https://arxiv.org/abs/2501.12948)

以下关于蒸馏大型语言模型的论文报告了精心设计的软蒸馏目标取得了强劲结果：

* Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang，《MiniLLM：大型语言模型的知识蒸馏》(2023 年 6 月 14 日)，[`arxiv.org/abs/2306.08543`](https://arxiv.org/abs/2306.08543)

### A.8.2 延伸阅读

以下章节提供了更多关于监督微调（硬蒸馏中使用的技术）以及处理批量训练示例时掩码操作的细节：

* Sebastian Raschka, *从零开始构建大型语言模型*, 第 7 章 (Manning, 2024)，[`www.manning.com/books/build-a-large-language-model-from-scratch`](https://www.manning.com/books/build-a-large-language-model-from-scratch)

以下文章提供了推理模型训练管道的实用演练，包括蒸馏和 RLVR：

* Sebastian Raschka，《理解推理型大语言模型：构建和优化推理模型的方法与策略》(2025 年 2 月 5 日)，[`magazine.sebastianraschka.com/p/understanding-reasoning-llms`](https://magazine.sebastianraschka.com/p/understanding-reasoning-llms)

## A.9 附录 F：大语言模型评估的常见方法

### A.9.1 参考文献

这是引入流行的多项选择 MMLU 数据集的原始论文：

* Dan Hendrycks, Collin Burns, Steven Basart, et al.，《测量大规模多任务语言理解》(2020 年 9 月 7 日)，[`arxiv.org/abs/2009.03300`](https://arxiv.org/abs/2009.03300)

以下维基百科文章提供了 Elo 评分系统的详细描述：

* Elo 评分系统，[`en.wikipedia.org/wiki/Elo\_rating\_system`](https://en.wikipedia.org/wiki/Elo_rating_system)

以下论文描述了广受欢迎的大语言模型排行榜背后的原始方法论：

* Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, et al.，《Chatbot Arena：通过人类偏好评估大语言模型的开放平台》(2024 年 3 月 7 日)，[`arxiv.org/abs/2403.04132`](https://arxiv.org/abs/2403.04132)

### A.9.2 延伸阅读

以下论文讨论了诸如 LM Arena 等排行榜存在的问题：

* Shivalika Singh, Yiyang Nan, Alex Wang, et al.，《排行榜幻觉》(2025 年 4 月 29 日)，[`arxiv.org/abs/2504.20879`](http://arxiv.org/abs/2504.20879)

以下文章更详细地描述了 OpenAI 最新的开源模型 gpt-oss：

* Sebastian Raschka，《从 GPT-2 到 gpt-oss：分析架构演进》(2025 年 8 月 9 日)，[`magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the`](https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the)

以下综述比较了不同的 LLM 评判者方法：

* Jiawei Gu, Xuhui Jiang, Zhichao Shi, et al.，《LLM-as-a-Judge 综述》(2024 年 11 月 23 日)，[`arxiv.org/abs/2411.15594`](https://arxiv.org/abs/2411.15594)

以下论文描述的方法是一个微调后充当评判者的小型 LLM 示例：

* Mahesh Deshwal and Apoorva Chawla，《PHUDGE：Phi-3 作为可扩展评判者》(2024 年 5 月 12 日)，[`arxiv.org/abs/2405.08029`](https://arxiv.org/abs/2405.08029)
