
第一次运行此代码时，请期待花费十分钟或更长时间下载包含 70 亿模型权重的 10 GB 文件，即使在快速的互联网连接下也是如此。在我们的 5G 家庭网络上，我们花费了 8 分钟。当 `fastchat` 脚本下载 Vicuna 后，它将为您提供一个命令行界面（CLI），您可以在其中与 Vicuna 进行 AMA。^([66]) 如果您再次运行 Vicuna，则它将准备好在您的 `$HOME/.cache` 目录中，与您所有其他 Hugging Face Hub 模型一起使用。

```py
USER: Are you able to use common sense and logic to reason about the world and physical things?
ASSISTANT: As an artificial intelligence language model, I am designed to process and generate human-like text based on the data I was trained on, and I do not have the ability to perceive, sense or interact with the physical world directly. Therefore, I don't have access to the common sense and logic that humans use to reason about the world and physical things.
```

与 Vicuna 的此次交互的完整记录可在 GitLab 上的 `nlpia2` 包中找到。^([67])

如果您的笔记本电脑具有足够的 RAM 来运行 LLaMa-2，则您也可能可以运行 Vicuna。

### 10.2.14 AI 伦理与 AI 安全

在本章中，你学到了有关 AI 和大型语言模型造成的伤害很多。希望你已经想出了自己的想法，来帮助减轻这些伤害。设计、构建和使用自主算法的工程师开始关注这些算法造成的伤害以及它们的使用方式。如何通过最小化伤害来道德使用算法，被称为 *AI 伦理*。而那些最大程度减少或减轻这些伤害的算法通常被称为道德 AI。

您可能也听说过 *AI 控制问题* 或 *AI 安全*，并可能对其与 AI 伦理有所不同感到困惑。AI 安全是关于我们如何避免被未来的“机器统治者”有意或无意地消灭的问题。致力于 AI 安全的人们正在试图减轻由超级智能智能机器造成的长期存在风险。许多最大的 AI 公司的首席执行官已经公开表示了对这个问题的担忧。

> 减轻 AI 对灭绝风险的风险应该是与其他社会规模风险（如大流行和核战争）并列的全球优先事项。
>
> —— AI 安全中心

这一句话对于人工智能公司的业务如此重要，以至于超过 100 位人工智能公司的高级经理签署了这封公开信。尽管如此，许多这些公司并没有将重要资源、时间或公共宣传用于解决这一问题。许多最大的公司甚至都不愿意签署这种含糊的不承诺性声明。Open AI、微软和 Anthropic 签署了这封信，但苹果、特斯拉、Facebook、Alphabet（谷歌）、亚马逊等许多其他人工智能巨头却没有。

目前公开存在关于*人工智能安全*与*人工智能伦理*的紧急性和优先级的辩论。一些思想领袖，如尤瓦尔·赫拉利和 Yoshua Bengio，完全专注于人工智能安全，即约束或控制假设的超智能 AGI。其他一些不那么知名的思想领袖将他们的时间和精力集中在算法和人工智能当前造成的更直接的伤害上，换句话说，人工智能伦理。处于劣势的人群对人工智能的不道德使用尤为脆弱。当公司将用户数据货币化时，他们从最无力承受损失的人那里获取权力和财富。当技术被用来创建和维护垄断时，这些垄断将淘汰小企业、政府计划、非营利组织和支持弱势群体的个人。^([68])

所以，你关心这些紧迫的话题中的哪一个呢？有没有一些重叠的事情，你可以同时做，既可以减少对人类的伤害，又可以防止我们长期灭绝？也许*可解释的人工智能*应该是你帮助创造“道德和安全人工智能”的方法列表的首要选择。可解释的人工智能是指一种算法的概念，该算法能够解释其如何以及为何做出决策，特别是当这些决策是错误或有害的时候。你将在下一章学习的信息提取和知识图概念是构建可解释的人工智能的基础工具之一。而且，可解释的、扎根的人工智能不太可能通过生成事实不正确的陈述或论点来传播错误信息。如果你能找到帮助解释机器学习算法如何做出其有害预测和决策的算法，你可以利用这种理解来防止那种伤害。

## 10.3 自我测试

+   这一章的生成模型与您在上一章中看到的 BERT 模型有何不同？

+   我们将这本书的句子索引为 Longformer-based 阅读理解问答模型的上下文。如果您使用维基百科的部分内容作为上下文，情况会变得更好还是更糟？整个维基百科文章呢？

+   什么是向量搜索和语义搜索中最快的索引算法？（提示，这是一个诡计问题）

+   为 100 篇维基百科文章中提取的句子中的二元组（bigrams）拟合一个 Scikit-Learn 的`CountVectorizer`。计算你的计数向量中跟随第一个词的所有第二个词的条件概率，并使用 Python 的`random.choice`函数来自动完成句子的下一个词。与使用 LLM（如 Llama 2）自动完成你的句子相比，这个方法效果如何？

+   你会使用哪些方法或测试来帮助量化 LLM 的智能？衡量人类智能的最新基准是什么，它们对评估 LLM 或 AI 助手有用吗？

+   判断你的判断力：创建一个最聪明的开源 LLM 排名列表，你能想到的。现在访问 LMSYS 竞技场（`chat.lmsys.org`）并至少进行 5 轮评判。将你的排名列表与 LMSYS 排行榜上的官方 Elo 排名进行比较（`huggingface.co/spaces/lmsys/chatbot-arena-leaderboard`）。你的 LLM 排名有多少是错乱的？

+   你能解开《关于随机鹦鹉的危险：语言模型是否太大？》这篇论文的最后作者“Shmargaret Shmitchell”的谜团吗？她是谁？你能做些什么来支持她和她的合著者在 AI 研究的诚实和透明方面的斗争吗？

## 10.4 总结

+   像 GPT-4 这样的大型语言模型可能看起来很智能，但其答案背后的“魔法”是以概率方式选择下一个标记来生成。

+   调整你的生成模型将帮助你生成特定领域的内容，尝试不同的生成技术和参数可以提高输出的质量。

+   近似最近邻算法和库是寻找基于你的答案的信息的有用工具。

+   检索增强生成结合了语义搜索和生成模型的优点，创建了基于事实的 AI，可以准确回答问题。

+   LLM 在到目前为止的超过一半的自然语言理解问题上失败了，而扩大 LLM 规模并没有帮助。

[[1]](#_footnoteref_1) 由 Katharine Miller 撰写的《AI 的表面上的新兴能力是一种幻觉》2023 年（`mng.bz/z0l6`）

[[2]](#_footnoteref_2) 用于估计 ML 模型环境影响的工具（`mlco2.github.io/impact/`）

[[3]](#_footnoteref_3) 由 Carole-Jean Wu 等人撰写的《可持续 AI：环境影响、挑战与机遇》2022 年（`arxiv.org/pdf/2111.00364.pdf`）

[[4]](#_footnoteref_4) 由 Dmytro Nikolaiev 撰写的《数百万背后：估计大语言模型的规模》（`mng.bz/G94A`）

[[5]](#_footnoteref_5) GPT-2 的维基百科文章（`en.wikipedia.org/wiki/GPT-2`）

[[6]](#_footnoteref_6) 由 Terry Yue Zhuo 等人撰写的《通过越狱来对抗 ChatGPT：偏见、稳健性、可靠性和毒性》2023 年（`arxiv.org/abs/2301.12867`）

[[7]](#_footnoteref_7) Clever Hans 维基百科文章（ `en.wikipedia.org/wiki/Clever_Hans`）

[[8]](#_footnoteref_8) 维基百科关于社交媒体“点赞”按钮的有害影响的文章（ `en.wikipedia.org/wiki/Facebook_like_button#Criticism`）

[[9]](#_footnoteref_9) Techdirt 文章解释了 ChatGPT 如何放大了错误信息（ `www.techdirt.com/2023/07/19/g-o-media-execs-full-speed-ahead-on-injecting-half-cooked-ai-into-a-very-broken-us-media/`）

[[10]](#_footnoteref_10) 谷歌在蒂姆尼特·格布鲁与她的合著者艾米丽·M·本德尔、安吉丽娜·麦克米兰-梅杰和谢玛格丽特·施密特尔（一个化名？蒂姆尼特与姓米切尔的盟友）要求发布《关于随机鹦鹉的危险...》时解雇了她（ `dl.acm.org/doi/pdf/10.1145/3442188.3445922?uuid=f2qngt2LcFCbgtaZ2024`）

[[11]](#_footnoteref_11) 《GPT-4 技术报告》（ `arxiv.org/pdf/2303.08774.pdf`）

[[12]](#_footnoteref_12) "Emergent Abilities of Large Language Models" 一书附录 E 中提取的非紧急能力表格，作者为 Jason Wei 等人（ `arxiv.org/abs/2206.07682`）

[[13]](#_footnoteref_13) GitLab 上的 MathActive 包（ `gitlab.com/tangibleai/community/mathactive`）

[[14]](#_footnoteref_14) 推特现在称为 X，评分和推荐系统在新管理下变得更加有毒和不透明。

[[15]](#_footnoteref_15) SpaCy 基于规则的匹配文档（ `spacy.io/usage/rule-based-matching`）

[[16]](#_footnoteref_16) GitHub 上的 ReLM 项目（ `github.com/mkuchnik/relm`）

[[17]](#_footnoteref_17) GitHub 上的 lm-evaluation-harness 项目（ `github.com/EleutherAI/lm-evaluation-harness`）

[[18]](#_footnoteref_18) PyPi 上的正则表达式包（ `pypi.org/project/regex/`）

[[19]](#_footnoteref_19) GitHub 上的 Guardrails-ai 项目（ `github.com/ShreyaR/guardrails`）

[[20]](#_footnoteref_20) GitHub 上 `guardrails-ai` 的源代码（ `github.com/ShreyaR/guardrails`）

[[21]](#_footnoteref_21) Andy Zou 等人的《对齐语言模型的通用和可迁移的对抗性攻击》（ `llm-attacks.org/`）

[[22]](#_footnoteref_22) Ars Technica 新闻文章（ `arstechnica.com/information-technology/2023/02/ai-powered-bing-chat-spills-its-secrets-via-prompt-injection-attack/`）

[[23]](#_footnoteref_23) 来自 Antrhopic.AI 的 Jared Kaplan 等人的《神经语言模型的缩放定律》（ `arxiv.org/abs/2001.08361`）

[[24]](#_footnoteref_24) Huggingface 模型输出文档（ `huggingface.co/docs/transformers/main_classes/output`）

[[25]](#_footnoteref_25) 在 Huggingface 上的 *"Tokenizer 摘要"* (`huggingface.co/docs/transformers/tokenizer_summary`)

[[26]](#_footnoteref_26) 如何生成文本：使用变换器进行语言生成的不同解码方法 (`huggingface.co/blog/how-to-generate`)

[[27]](#_footnoteref_27) 搜索引擎的维基百科文章 (`en.wikipedia.org/wiki/Search_engine`)

[[28]](#_footnoteref_28) (`www.elastic.co/elasticsearch/`)

[[29]](#_footnoteref_29) `solr.apache.org/`

[[30]](#_footnoteref_30) Meilisearch GitHub 仓库 (`github.com/meilisearch/meilisearch`)

[[31]](#_footnoteref_31) GitLab 如何在可扩展到数百万用户的软件中使用 PostgreSQL 的三字母索引 (`about.gitlab.com/blog/2016/03/18/fast-search-using-postgresql-trigram-indexes/`)

[[32]](#_footnoteref_32) 有关使用 FAISS 库的绝佳资源 (`www.pinecone.io/learn/series/faiss/`)

[[33]](#_footnoteref_33) (`github.com/dataplayer12/Fly-LSH`)

[[34]](#_footnoteref_34) 使用分层可导航小世界图进行高效且鲁棒的近似最近邻搜索 (`arxiv.org/ftp/arxiv/papers/1603/1603.09320.pdf`)

[[35]](#_footnoteref_35) (`en.wikipedia.org/wiki/Six_degrees_of_separation`)

[[36]](#_footnoteref_36) 在 PyPi 上的局部最优产品量化 (`pypi.org/project/lopq/`)

[[37]](#_footnoteref_37) 由 Jeff Johnson、Matthijs Douze、Herve' Jegou 撰写的使用 GPU 进行十亿级相似度搜索的论文 (`arxiv.org/pdf/1702.08734.pdf`)

[[38]](#_footnoteref_38) `weaviate.io/blog/ann-algorithms-hnsw-pq`

[[39]](#_footnoteref_39) `github.com/spotify/annoy`

[[40]](#_footnoteref_40) Faiss GitHub 仓库: (`github.com/facebookresearch/faiss`)

[[41]](#_footnoteref_41) NMSlib GitHub 仓库 (`github.com/nmslib/nmslib`)

[[42]](#_footnoteref_42) GitHub 上的 ANN Benchmarks 仓库 (`github.com/erikbern/ann-benchmarks/`)

[[43]](#_footnoteref_43) OpenSearch k-NN 文档 (`opensearch.org/docs/latest/search-plugins/knn`)

[[44]](#_footnoteref_44) (`github.com/jina-ai/jina`)

[[45]](#_footnoteref_45) `github.com/deepset-ai/haystack`

[[46]](#_footnoteref_46) (`github.com/neuml/txtai`)

[[47]](#_footnoteref_47) Haystack 在 Windows 上的安装说明 (`docs.haystack.deepset.ai/docs/installation`)

[[48]](#_footnoteref_48) 关于 `faiss_index_factor_str` 选项的 Haystack 文档 (`github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index`)

[[49]](#_footnoteref_49) BART：由 Mike Lewis 等人于 2019 年提出的用于自然语言生成、翻译和理解的去噪序列到序列预训练（`arxiv.org/abs/1910.13461`）。

[[50]](#_footnoteref_50) （`docs.streamlit.io/`）。

[[51]](#_footnoteref_51) （`share.streamlit.io/`）。

[[52]](#_footnoteref_52) "It’s A Trap" 的 Know Your Meme 文章（`knowyourmeme.com/memes/its-a-trap`）。

[[53]](#_footnoteref_53) （`theintercept.com/2018/08/01/google-china-search-engine-censorship/`）。

[[54]](#_footnoteref_54) "It Takes a Village to Combat a Fake News Army" 的 Zachary J. McDowell & Matthew A Vetter 文章（`journals.sagepub.com/doi/pdf/10.1177/2056305120937309`）。
