
阅读器微调后，让我们现在将其在测试集上的性能与我们的基线模型进行比较：

```py
reader_eval["Fine-tune on SQuAD + SubjQA"] = evaluate_reader(reader)
plot_reader_eval(reader_eval)
```

![](img/nlpt_07in05.png)

哇，领域适应将我们的 EM 分数提高了六倍以上，并且*F*[1]-score 增加了一倍多！此时，您可能会想知道为什么我们不直接在 SubjQA 训练集上对预训练的语言模型进行微调。一个原因是 SubjQA 中只有 1,295 个训练示例，而 SQuAD 有超过 100,000 个，因此我们可能会遇到过拟合的挑战。尽管如此，让我们看看天真的微调会产生什么结果。为了公平比较，我们将使用与在 SQuAD 上微调基线时使用的相同语言模型。与以前一样，我们将使用`FARMReader`加载模型：

```py
minilm_ckpt = "microsoft/MiniLM-L12-H384-uncased"
minilm_reader = FARMReader(model_name_or_path=minilm_ckpt, progress_bar=False,
                           max_seq_len=max_seq_length, doc_stride=doc_stride,
                           return_no_answer=True)
```

接下来，我们进行一轮微调：

```py
minilm_reader.train(data_dir=".", use_gpu=True, n_epochs=1, batch_size=16,
             train_filename=train_filename, dev_filename=dev_filename)
```

并在测试集上进行评估：

```py
reader_eval["Fine-tune on SubjQA"] = evaluate_reader(minilm_reader)
plot_reader_eval(reader_eval)
```

![](img/nlpt_07in06.png)

我们可以看到，直接在 SubjQA 上对语言模型进行微调的结果比在 SQuAD 和 SubjQA 上进行微调的性能要差得多。

## 警告

处理小数据集时，最佳做法是在评估 Transformer 时使用交叉验证，因为它们可能容易过拟合。您可以在[FARM 存储库](https://oreil.ly/K3nK8)中找到如何使用 SQuAD 格式的数据集执行交叉验证的示例。

## 评估整个 QA 管道

现在我们已经了解了如何分别评估阅读器和检索器组件，让我们将它们联系在一起，以衡量我们管道的整体性能。为此，我们需要为我们的检索器管道增加阅读器及其评估的节点。我们已经看到，在`k = 10`时，我们几乎可以完美地召回，因此我们可以固定这个值，并评估这对阅读器性能的影响（因为现在它将收到每个查询的多个上下文，而不是 SQuAD 风格的评估）：

```py
# Initialize retriever pipeline
pipe = EvalRetrieverPipeline(es_retriever)
# Add nodes for reader
eval_reader = EvalAnswers()
pipe.pipeline.add_node(component=reader, name="QAReader",
              inputs=["EvalRetriever"])
pipe.pipeline.add_node(component=eval_reader, name="EvalReader",
              inputs=["QAReader"])
# Evaluate!
run_pipeline(pipe)
# Extract metrics from reader
reader_eval["QA Pipeline (top-1)"] = {
    k:v for k,v in eval_reader.__dict__.items()
    if k in ["top_1_em", "top_1_f1"]}
```

然后，我们可以将模型预测文档中检索器返回的答案的前 1 个 EM 和*F*[1]分数与图 7-12 中的进行比较。

![](img/nlpt_0712.png)

**图 7-12。** EM 和*F*[1]分数的阅读器与整个 QA 管道的比较

从这个图中，我们可以看到检索器对整体性能的影响。特别是，与匹配问题-上下文对相比，整体性能有所下降，这是在 SQuAD 风格的评估中所做的。这可以通过增加阅读器被允许预测的可能答案数量来避免。

到目前为止，我们只从上下文中提取了答案范围，但一般情况下，答案的各个部分可能分散在整个文档中，我们希望我们的模型能够将这些片段综合成一个连贯的答案。让我们看看如何使用生成式 QA 来成功完成这项任务。

# 超越抽取式 QA

在文档中提取答案作为文本段的一个有趣的替代方法是使用预训练语言模型生成它们。这种方法通常被称为*抽象*或*生成型 QA*，它有潜力产生更好措辞的答案，综合了多个段落中的证据。虽然这种方法比抽取式 QA 不够成熟，但这是一个快速发展的研究领域，所以很可能在您阅读本文时，这些方法已经被广泛应用于工业中！在本节中，我们将简要介绍当前的技术水平：*检索增强生成*（RAG）。

RAG 通过将读者替换为*生成器*并使用 DPR 作为检索器来扩展了我们在本章中看到的经典检索器-阅读器架构。生成器是一个预训练的序列到序列变换器，如 T5 或 BART，它接收来自 DPR 的文档的潜在向量，然后根据查询和这些文档迭代生成答案。由于 DPR 和生成器是可微分的，整个过程可以端到端地进行微调，如图 7-13 所示。

![RAG Architecture](img/nlpt_0713.png)

**图 7-13。** RAG 架构用于对检索器和生成器进行端到端微调（由 Ethan Perez 提供）

为了展示 RAG 的工作原理，我们将使用之前的`DPRetriever`，因此我们只需要实例化一个生成器。有两种类型的 RAG 模型可供选择：

*RAG-Sequence*

使用相同的检索文档来生成完整的答案。特别是，从检索器中获取的前*k*个文档被馈送到生成器，生成器为每个文档产生一个输出序列，然后对结果进行边际化以获得最佳答案。

*RAG-Token*

可以使用不同的文档来生成答案中的每个标记。这允许生成器从多个文档中综合证据。

由于 RAG-Token 模型往往比 RAG-Sequence 模型表现更好，我们将使用在 NQ 上进行微调的标记模型作为我们的生成器。在 Haystack 中实例化生成器类似于实例化阅读器，但是我们不是为上下文中的滑动窗口指定`max_seq_length`和`doc_stride`参数，而是指定控制文本生成的超参数：

```py
from haystack.generator.transformers import RAGenerator

generator = RAGenerator(model_name_or_path="facebook/rag-token-nq",
                        embed_title=False, num_beams=5)
```

这里的`num_beams`指定了在波束搜索中使用的波束数量（文本生成在第五章中有详细介绍）。与 DPR 检索器一样，我们不嵌入文档标题，因为我们的语料库始终根据产品 ID 进行过滤。

接下来要做的事情是使用 Haystack 的`GenerativeQAPipeline`将检索器和生成器联系起来：

```py
from haystack.pipeline import GenerativeQAPipeline

pipe = GenerativeQAPipeline(generator=generator, retriever=dpr_retriever)
```

**注意**

在 RAG 中，查询编码器和生成器都是端到端训练的，而上下文编码器是冻结的。在 Haystack 中，`GenerativeQAPipeline`使用`RAGenerator`的查询编码器和`DensePassageRetriever`的上下文编码器。

现在让我们通过输入一些关于之前的亚马逊 Fire 平板电脑的查询来尝试一下 RAG。为了简化查询，我们将编写一个简单的函数，该函数接受查询并打印出前几个答案：

```py
def generate_answers(query, top_k_generator=3):
    preds = pipe.run(query=query, top_k_generator=top_k_generator,
                     top_k_retriever=5, filters={"item_id":["B0074BW614"]})
    print(f"Question: {preds['query']} \n")
    for idx in range(top_k_generator):
        print(f"Answer {idx+1}: {preds['answers'][idx]['answer']}")
```

好的，现在我们准备好测试一下：

```py
generate_answers(query)
```

```py
Question: Is it good for reading?

Answer 1:  the screen is absolutely beautiful
Answer 2:  the Screen is absolutely beautiful
Answer 3:  Kindle fire
```

这个答案并不算太糟糕，但它确实表明问题的主观性使生成器感到困惑。让我们尝试一些更加客观的问题：

```py
generate_answers("What is the main drawback?")
```

```py
Question: What is the main drawback?

Answer 1:  the price
Answer 2:  no flash support
Answer 3:  the cost
```

这更有意义！为了获得更好的结果，我们可以在 SubjQA 上对 RAG 进行端到端的微调；我们将把这留作练习，但如果您有兴趣探索，可以在![nlpt_pin01](img/nlpt_pin01.png) [Transformers repository](https://oreil.ly/oZz4S)中找到一些脚本来帮助您入门。

# 结论

好吧，这是 QA 的一个风风火火的介绍，你可能还有很多问题想要得到答案（双关语！）。在本章中，我们讨论了 QA 的两种方法（提取式和生成式），并检查了两种不同的检索算法（BM25 和 DPR）。在这个过程中，我们看到领域适应可以是一个简单的技术，可以显著提高我们的 QA 系统的性能，并且我们看了一些用于评估这种系统的最常见的指标。尽管我们专注于封闭领域的 QA（即电子产品的单一领域），但本章中的技术可以很容易地推广到开放领域的情况；我们建议阅读 Cloudera 出色的 Fast Forward [QA 系列](https://oreil.ly/Fd6lc)来了解其中的内容。

在野外部署 QA 系统可能是一个棘手的问题，我们的经验是，其中相当大一部分价值来自首先为最终用户提供有用的搜索功能，然后是一个提取组件。在这方面，读者可以以新颖的方式使用，超出了按需回答用户查询的范围。例如，[Grid Dynamics](https://oreil.ly/CGLh1)的研究人员能够使用他们的读者自动提取客户目录中每种产品的优缺点。他们还表明，可以通过创建查询，如“什么样的相机？”以零-shot 方式使用读者来提取命名实体。鉴于其幼稚和微妙的失败模式，我们建议只有在其他两种方法耗尽后才探索生成式 QA。解决 QA 问题的这种“需求层次结构”在图 7-14 中有所说明。

![QA 金字塔](img/nlpt_0714.png)

**图 7-14。** QA 需求层次结构

展望未来，一个令人兴奋的研究领域是*多模态 QA*，它涉及对文本、表格和图像等多种模态进行 QA。正如 MultiModalQA 基准所描述的那样，¹⁷这样的系统可以使用户回答跨越不同模态的信息的复杂问题，比如“著名的两根手指的画是什么时候完成的？”另一个具有实际业务应用的领域是在*知识图谱*上进行 QA，其中图的节点对应于现实世界的实体，它们的关系由边定义。通过将事实编码为（*主语*，*谓语*，*宾语*）三元组，可以使用图来回答关于缺失元素的问题。有一个将 Transformer 与知识图谱相结合的例子，请参见[Haystack 教程](https://oreil.ly/n7lZb)。另一个有前途的方向是*自动生成问题*，作为一种使用未标记数据或数据增强进行无监督/弱监督训练的方式。最近的两个例子包括 Probably Answered Questions（PAQ）基准和跨语言设置的合成数据增强的论文。¹⁸

在本章中，我们已经看到，为了成功地将 QA 模型应用于实际用例，我们需要应用一些技巧，比如实现一个快速检索管道，以便在几乎实时进行预测。尽管这听起来可能不像什么，但想象一下，如果您必须等待几秒钟才能获得谷歌搜索结果，您的体验会有多不同——几秒钟的等待时间可能决定了您的 Transformer 应用程序的命运。在下一章中，我们将看一些加速模型预测的方法。

¹尽管在这种特殊情况下，每个人都同意 Drop C 是最好的吉他调音。

²J. Bjerva 等人，[“SubjQA: A Dataset for Subjectivity and Review Comprehension”](https://arxiv.org/abs/2004.14283)，（2020）。

³ 正如我们将很快看到的，还有一些*无法回答*的问题，旨在产生更健壮的模型。

⁴ D. Hendrycks 等人，[“CUAD: 专家注释的法律合同审查 NLP 数据集”](https://arxiv.org/abs/2103.06268)，（2021）。

⁵ P. Rajpurkar 等人，[“SQuAD：超过 10 万个文本理解问题”](https://arxiv.org/abs/1606.05250)，（2016）。

⁶ P. Rajpurkar, R. Jia, and P. Liang，[“知道你不知道的：SQuAD 的无法回答的问题”](https://arxiv.org/abs/1806.03822)，（2018）。

⁷ T. Kwiatkowski 等人，“自然问题：问答研究的基准”，*计算语言学协会交易* 7（2019 年 3 月）：452-466，[*http://dx.doi.org/10.1162/tacl_a_00276*](http://dx.doi.org/10.1162/tacl_a_00276)。

⁸ W. Wang 等人，[“MINILM: 用于任务不可知的预训练 Transformer 的深度自注意压缩”](https://arxiv.org/abs/2002.10957)，（2020）。

⁹ 请注意，`token_type_ids`并不是所有 Transformer 模型都具有的。对于 MiniLM 等类似 BERT 的模型，`token_type_ids`在预训练期间也用于整合下一个句子预测任务。

¹⁰ 有关如何提取这些隐藏状态的详细信息，请参阅第二章。

¹¹ 如果一个向量的大部分元素都是零，那么它就是稀疏的。

¹² 该指南还提供了 macOS 和 Windows 的安装说明。

¹³ 有关使用 TF-IDF 和 BM25 进行文档评分的深入解释，请参阅 D. Jurafsky 和 J.H. Martin（Prentice Hall）的《语音和语言处理》第 3 版第二十三章。

¹⁴ V. Karpukhin 等人，[“用于开放领域问答的密集通道检索”](https://arxiv.org/abs/2004.04906)，（2020）。

¹⁵ D. Yogatama 等人，[“学习和评估通用语言智能”](https://arXiv.org/abs/1901.11373)，（2019）。

¹⁶ P. Lewis 等人，[“用于知识密集型 NLP 任务的检索增强生成”](https://arxiv.org/abs/2005.11401)，（2020）。

¹⁷ A. Talmor 等人，[“MultiModalQA：文本、表格和图像的复杂问答”](https://arxiv.org/abs/2104.06039)，（2021）。

¹⁸ P. Lewis 等人，[“PAQ: 6500 万个可能被问到的问题及其用途”](https://arxiv.org/abs/2102.07033)，（2021）；A. Riabi 等人，[“用于零样本跨语言问答的合成数据增强”](https://arxiv.org/abs/2010.12643)，（2020）。
