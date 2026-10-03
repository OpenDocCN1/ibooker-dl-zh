
在许多应用中，你需要找到与某些其他文本相似的文字。例如，在亚马逊寻找产品时，你可能希望找到类似的产品。*向量数据库*通过找到具有相似意义的文本来简化这一过程。即使没有任何关键词完全匹配，向量数据库也能将“带有 USB 充电器的笔记本电脑背包”的亚马逊搜索与标记为“技术友好型日背包、17 英寸、内置电源银行”的项目相连接。

对于许多搜索应用，你不需要向量数据库。除非你的数据集非常大（在训练 LLM 或运行全球电子商务网站时，情况通常如此），请记住特斯拉前 AI 总监、OpenAI 创始团队一员的安德烈·卡帕西（Andrej Karpathy）的建议[建议](https://oreil.ly/ez8fn)：在开始一个项目时，你可能需要的只是一个免费的 NumPy Python 库。

向量数据库被设计用来存储嵌入并快速执行与给定嵌入接近的嵌入的搜索。想象一下，你有一堆数据——可能是文本、图像，或者两者的混合。第一步是将这些数据向量化。一旦完成，数据库就可以以易于找到相似数据点的方式存储这些向量。这个过程被称为*索引*。当提交文本查询时，查询也会被向量化，然后数据库通过执行[*最近邻搜索*](https://oreil.ly/oq2Du)来计算提交的查询与存储记录的距离。这意味着你可以在向量空间中找到与给定查询最相似的项目。

向量数据库允许你存储与你的数据和其嵌入向量相关的元数据。你可以在执行向量相似度搜索之前使用它来缩小搜索范围。这可以显著提高查询效率，尤其是在大型数据集上。例如，如果你已经知道你的查询语言，并且你的元数据中有一个语言字段，你可以使用它来提高性能。

当选择向量数据库时，除了成本之外，还需要考虑可扩展性、容错性和索引技术的可用性。这些特性在表 4-1 中详细说明。

**表 4-1\. 选择向量数据库**

| 特征 | 描述 | 常见指标 | 索引技术（影响） |
| --- | --- | --- | --- |
| 可扩展性 | 处理不断增加的数据量和查询负载的能力 | 吞吐量（每秒查询数）延迟（查询执行时间）存储容量 | 水平扩展能力（影响吞吐量）分片策略（影响查询效率） |
| 容错性 | 在故障期间保持可用性的能力 | 正常运行时间百分比恢复时间（MTTR） | 数据复制（确保数据可用性）高可用性（HA）功能（最小化停机时间） |
| 索引技术 | 在高维向量空间中进行高效搜索的方法 | 搜索精度（检索相关向量）搜索速度 | 度量树（HNSW、VP-Tree）：适用于中等数据和维度 Hashing（LSH）：在近似最近邻搜索中更快（与精度权衡）倒排文件（IVF）：如果你可以使用多个元数据过滤器，可以提高速度 |

## 维护新鲜数据

预处理流程开发的第 10 步是确保数据保持最新。将索引与底层数据的实时变化同步可能会很具挑战性，方法从“当我询问时告诉我是否有任何变化”到“无论何时发生变化都立即告诉我”连续不断。*轮询*是最简单的方法：你定期查询源数据，并将最新的快照与数据库中已有的数据进行比较。这种方法易于理解，当你的数据量或新鲜度要求不高时，它也能正常工作，但它会浪费资源，即使没有任何变化也会反复询问。它还引入了与查询频率相等的延迟。如果你每年刷新一次数据，至少需要一年时间才能检测到变化。

一个更精确的替代方案是*变更数据捕获*（CDC），它直接连接到源的事务或提交日志或有关新鲜度的某些元数据（如文档的最后修改日期）。你不必在整个文档集中寻找差异，而是可以读取已更改的文档列表并仅更新它们。CDC 仍然涉及拉取数据，但它消除了猜测并最小化了检查所花费的带宽，这些检查会导致无意义的更新。

当数据的生产者能够积极推送信息时，你就进入了事件驱动和流式处理的领域。在*事件驱动*更新模型中，数据的所有者发送诸如“产品描述已更改”或“文章已更新”的消息。每个事件都是自包含的，并且可以立即触发数据库中的更新。

## 生成微调数据集

如果你只执行预处理数据部分中描述的 10 个步骤，你将拥有足够的数据来执行你 LLM 的预训练步骤。然而，对于几乎所有实际应用，你都需要你的 LLM 除了猜测隐藏的单词之外还能做其他事情。LLM 最常见的任务是回答问题，为此，你需要提供给它一个问题和预期答案的列表。这被称为*指令数据集*或*微调训练数据集*。

创建微调训练数据集有四种主要方法：

**手动整理**

这是最实际的方法。你和你的团队自己整理和设计数据集，仔细选择和制作每条指令以满足你的需求。这个过程很慢且劳动密集，但它能让你掌握控制权，当你需要高度特定或针对特定任务定制的数据集时，这种控制权会带来回报。

**收集和改进现有的开源数据集**

当已经有有价值的数据存在时，为什么还要重新发明轮子呢？你可以从开源数据集中提取，对它们进行精炼和改进，以更好地满足你的需求。这是一个不牺牲质量的捷径，尤其是在与一些战略性的增强相结合时。通过微调现有内容，你正在利用社区的集体工作来加速你自己的进步。

**使用 LLM 生成**

你可以使用 LLM 生成一个指令调整数据集。为此，你需要将你的数据块存储在向量数据库中。这个过程将在下一节中详细描述。

**混合方法**

最后，别忘了结合这些方法的力量。你可以将手动创建、开源数据集整理和模型生成结合起来，全面覆盖你的需求。简单来说，就是将所有方法都使用上，将每个数据集附加到另一个数据集上。这种混合方法为你提供了灵活性——让你可以根据手头的任务，充分利用每种方法的优点。

微调数据集通常分为两大类。*通用指令微调数据集*范围广泛，覆盖许多领域的各种任务。它们的目的是提高模型遵循通用指令的能力，使其更加灵活。数据集越广泛，您的模型在理解和执行各种指令方面的能力就越强。另一方面，*特定领域指令微调数据集*则专注于特定领域，为专业领域构建。例如，一个医疗指令数据集将训练模型处理诊断或医疗辅助等任务。通过专注于特定领域，您正在增强模型在该特定领域的专业知识。

## 自动生成指令微调数据集

**第 1 步：预处理和向量化**

我们推荐使用[LlamaIndex](https://oreil.ly/1uMXc)来处理您的大量文本数据。LlamaIndex 可以执行之前概述的一般流程中的许多预处理步骤，如分词和清理。它还可以为您的语料库中的每个片段生成高质量的向量表示。您可以将这些文档向量存储在不同的数据库中。

**第 2 步：构建检索机制**

创建一个简单的程序，给定一个问题，从您在第 1 步中创建的向量数据库中检索最相关的关联块。您可以使用 LlamaIndex 的现成`VectorIndexRetriever`功能来完成此操作。

**第 3 步：生成问题**

将文档（而不是内容块）提交给现有的 LLM，并要求它生成可以由文档回答的问题集。您可能需要将文档分成多个部分，但它们可以比数据库中的内容块大得多。例如，一个内容块通常将有 8,000 个字符（以便为其生成嵌入），而 GPT-4o 可以为包含大约 400,000 个字符的文档生成问题列表。

**第 4 步：让现有的 LLM 决定最佳答案**

将在第 3 步中生成的每个问题提交到第 2 步中编写的程序。结果将是一个与问题密切相关的内容块列表。将问题和答案列表发送给现有的 LLM，并要求它选择包含最佳答案的块，并使用该块生成一个清晰、完整的答案。您可以要求 LLM 以 JSON 格式的记录形式生成输出，格式为`{"instruction": <question>, "input": "", "output": <answer>}`。重复此循环，直到您获得所需数量的示例。

在生成问题和答案后，进行基本卫生检查：通过计算文本级别的余弦相似度来去重几乎相同的问答对，过滤掉引入了检索段落中不存在的信息的答案，然后手动检查一小部分随机样本，以确保自动过滤器校准正确。如果你想使数据集更丰富，可以提示 LLM 提出更深入的后续问题，或者以有效的 JSON 等约束格式返回答案——这两种策略都教会了微调模型处理更复杂的指令。

当资源稀缺时，整个流程可以缩减。对于一个小型语料库，你可能可以使用 BERT 等轻量级库预先计算嵌入，跳过检索步骤，让 LLM 从一个段落中生成问题和答案，并通过余弦相似度验证答案与源文本保持接近。

# 结论

本章讨论了大型语言模型（LLM）的端到端数据工程流程。虽然 LLM 的数据工程仍是一个新兴领域，但本章提供的提示和指南应该为你提供了每个步骤的良好基础，以便你可以针对特定的用例优化你的流程。

# 参考文献

Chang, Ernie, et al. [“Scaling Parameter-Constrained Language Models with Quality Data”](https://oreil.ly/4v-i1), arXiv, October 2024.

Chardet. n.d. [“Chardet: The Universal Character Encoding Detector”](https://oreil.ly/NU-hl), accessed May 21, 2025.

Codd, E. F. [“A Relational Model of Data for Large Shared Data Banks”](https://oreil.ly/14TGS), *Communications of the ACM*13 (6): 377–87 (1970).

Common Crawl. n.d. [Common Crawl](https://oreil.ly/TaXZi), accessed May 21, 2025.

Dodge, Jesse, et al. [“Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus”](https://oreil.ly/g1Gek) arXiv, September 2021.

Gao, Yunfan, et al. [“Retrieval-Augmented Generation for Large Language Models: A Survey”](https://oreil.ly/9D0yi), arXiv, March 27, 2024.

Hoffmann, Jordan, et al. [“Training Compute-Optimal Large Language Models”](https://oreil.ly/F2I7y), arXiv, March 2022.

Kaplan, Jared, et al. [“Scaling Laws for Neural Language Models”](https://oreil.ly/IwsIC), arXiv, January 2020.

Lee, Cinoo, et al. [“People Who Share Encounters with Racism Are Silenced Online by Humans and Machines, but a Guideline-Reframing Intervention Holds Promise”](https://oreil.ly/e7va9), *Proceedings of the National Academy of Sciences* 121 (38): e2322764121 (2024).

LlamaIndex. n.d. [“Vector Stores”](https://oreil.ly/05_gK), accessed May 21, 2025.

Ma, Yingwei, et al. [“At Which Training Stage Does Code Data Help LLMs Reasoning?”](https://oreil.ly/ZVMyR), arXiv, September 2023.

Nguyen, Thuat, et al., [“CulturaX: A Cleaned, Enormous, and Multilingual Dataset for Large Language Models in 167 Languages”](https://oreil.ly/Hfb9j), arXiv, September 2023.

OpenAI 平台. 未知日期. [“向量嵌入”](https://oreil.ly/zj-2l), 访问日期：2025 年 5 月 21 日.

Pemistahl. 未知日期. [lingua-py](https://oreil.ly/VIIBH), 访问日期：2025 年 5 月 21 日.

Penedo, Guilherme, 等人. [“Falcon LLM 的 RefinedWeb 数据集：仅使用网页数据就超越精心挑选的语料库”](https://oreil.ly/ZBNxW), arXiv, 2023 年 6 月.

Reis, Joe 和 Matt Housley. [*《数据工程基础》*](https://learning.oreilly.com/library/view/fundamentals-of-data/9781098108298/), O’Reilly, 2022 年.

Salley, C. 等. [“为用户分析师提供 OLAP：一项 IT 命令”](https://oreil.ly/u549E) (1998 年).

Wang, Zige, 等人. [“大型语言模型的数据管理：综述”](https://oreil.ly/JWI_9), arXiv, 2024 年 8 月.

WARC 规范. 未知日期. [“WARC 格式 1.0”](https://oreil.ly/QnaDE), 访问日期：2025 年 5 月 21 日.

Xu, Yipei, 等人. [“源提示：从多个来源的多样化语料库上对语言模型进行协调预训练”](https://oreil.ly/8-tjT), arXiv, 2023 年 11 月.

Xue, Fuzhao, 等人. [“重复与否：在令牌危机下扩展 LLM 的见解”](https://oreil.ly/5AAg5), arXiv, 2023 年 10 月.

Yang, Rui, 等人. [“RAGVA：在实践中构建基于检索增强生成的虚拟助手”](https://oreil.ly/Zv3UP), arXiv, 2025 年 2 月.

# 进一步阅读

Gao, Leo, 等人. [“Pile：用于语言模型的多样化文本数据集，800GB”](https://oreil.ly/H9Ycv), arXiv, 2020 年 12 月.
