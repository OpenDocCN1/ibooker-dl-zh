
如果这一切看起来有些模糊，那是因为它确实是——如果提起版权诉讼，这些单一因素都不是铁的规则，它们都会被权衡。在转向已经对 LLMs 开发者提起的诉讼之前，让我们首先考察一个类似地基于使用大量互联网版权文本的案例：*Authors Guild v. Google* [[51]](https://www.copyright.gov/fair-use/summaries/authorsguild-google-2dcir2015.pdf)。

2015 年，谷歌与几家主要研究图书馆合作，将他们的藏书数字化——大约有 2000 万卷。这家科技巨头通过合作伙伴关系访问这些书籍，扫描它们，并允许人们搜索文本片段，所有这些都是在没有版权所有者许可的情况下，也没有支付版税。这个案件上诉到了第二巡回上诉法院，该法院同意了下级法院的意见，即谷歌的数字化努力构成了合理使用，因为搜索功能使公众能够访问他们否则无法获得的书本信息，并且尽管谷歌使用了书籍的全文，但他们只返回了匹配文本的片段，而不是使整本书可用。这种使用全部源材料为基本不同的工具的概念与 LLMs 的训练类似。

通常来说，我们迄今为止讨论的 LLMs（大型语言模型）似乎受到合理使用的保护，因为模型与任何文档都非常不同，因此使用这些材料是具有变革性的。然而，问题复杂化了，用户已经证明有时可以让 LLMs 逐字逐句地重复文本。由于 LLMs 的概率性质，很难一致地展示“记忆”源材料的例子。由于对 LLMs 确切学习内容的理解不足，即使是它们的开发者也不太可能确切地说出模型何时会逐字逐句地复制短语或文本。尽管如此，根据*Authors Guild v. Google*的先例，LLMs 被认为合理使用的可能性似乎相当大。

大型语言模型（LLMs）并不是唯一在版权方面引起轰动的生成模型——如前所述，还有令人印象深刻的生成模型能够创建所有类型的合成媒体，包括图像、音频和视频。其中一些最受欢迎的模型，包括 Midjourney 和 Stable Diffusion，是文本到图像的模型：用户可以描述他们想要的图片外观，模型将为他们生成。

就像大型语言模型（LLMs）一样，生成式图像模型在从互联网收集的大量数据上进行训练。与像 Common Crawl 这样的文本数据集一样，也存在常见的图像数据集，例如由非营利组织大规模人工智能开放网络（LAION）编纂的包含 58 亿张图像的 LAION-5B 数据集。LAION-5B 被 Stability AI（Stable Diffusion 的开发者）和其他公司使用；它由公开在线可用的图像组成，包括股票照片和编辑摄影。一位德国摄影师在发现他的部分股票照片被用于 LAION-5B 后，要求将其删除；LAION 回应称，满足此类请求是不可能的，因为数据库中只包含图像链接，没有存储任何内容，他们无法轻易识别出哪些图像来自他的作品集。德国版权法——像许多国家一样——允许在数据“合法可访问”且之后删除的情况下进行数据挖掘，但生成模型的兴起使得这个问题受到了更多的审查[[52]](https://www.vice.com/en/article/pkapb7/a-photographer-tried-to-get-his-photos-removed-from-an-ai-dataset-he-got-an-invoice-instead)]。Stability AI 后来宣布，他们将尊重包括在 LAION 数据集中的艺术家的工作中的退出请求[[53]](https://arstechnica.com/information-technology/2022/12/stability-ai-plans-to-let-artists-opt-out-of-stable-diffusion-3-image-training/)]。

Stability AI 目前正在因使用 Getty 收藏中的超过 1200 万张照片而受到 Getty Images 的起诉[[52]](https://www.vice.com/en/article/pkapb7/a-photographer-tried-to-get-his-photos-removed-from-an-ai-dataset-he-got-an-invoice-instead)]，[[54](https://fingfx.thomsonreuters.com/gfx/legaldocs/byvrlkmwnve/GETTY%20IMAGES%20AI%20LAWSUIT%20complaint.pdf)]。在诉状中，原告写道：

Getty Images 在近三十年的时间里，以巨大的成本精心挑选了数亿个高质量的视觉资产……其中许多图像是由 Getty Images 的摄影师作为雇佣作品创作的，其他则是由 Getty Images 从第三方获得，并转让了其相关的版权，剩余的则是由 Getty Images 的数百个内容合作伙伴或数十万贡献摄影师许可的，他们依赖 Getty Images 为他们产生的许可收入。[55](https://fingfx.thomsonreuters.com/gfx/legaldocs/byvrlkmwnve/GETTY%20IMAGES%20AI%20LAWSUIT%20complaint.pdf)

其隐含意义很明确：生成式 AI 模型对 Getty 和作为行业的股票摄影构成了生存威胁。Getty 希望得到对其贡献和感知到的版权侵权的赔偿，但正如大型文本数据集一样，很难确定模型从任何单一图像中保留了多少信息，而且，再次强调，Stability AI 的使用似乎具有变革性。

有趣的是，由于训练数据中的一个瑕疵，Getty 可能拥有更强的诉讼理由：投诉进一步指控以下内容：

通常，Stable Diffusion 生成的输出包含 Getty Images 水印的修改版本，这导致了对图像来源的混淆，并错误地暗示与 Getty Images 有关联。虽然通过使用 Stable Diffusion 生成的一些输出在美学上令人愉悦，但其他输出质量较低，有时从奇异到丑陋不等。Stability AI 将 Getty Images 的标志融入低质量、不吸引人或有冒犯性的图像中，进一步违反了联邦和州商标法。[55](https://fingfx.thomsonreuters.com/gfx/legaldocs/byvrlkmwnve/GETTY%20IMAGES%20AI%20LAWSUIT%20complaint.pdf)

如果图像上出现 Getty Images 的水印，那么 Stable Diffusion 或其用户可能会被发现在侵犯商标权，尽管 Stability AI 无疑会迅速采取措施解决这个问题。总的来说，这是一个相对未经验证的法律法规领域。

当一个模型不仅学会了人类艺术家捕捉的图像，而且还编码了该艺术家的风格时，事情变得更加复杂。除了生成逼真的渲染效果外，像 Midjourney 和 Stable Diffusion 这样的生成模型还能够以特定风格创作艺术品，如第“生成 AI 在创意工作流程”节所述。风格通常不受版权保护，但很容易理解艺术家们可能会认为这种模仿会贬低或削弱他们的作品。Sarah Andersen 是一位著名的漫画家，她在“Sarah’s Scribbles”系列中发布网络漫画，她写了一篇关于她经历另类右翼网络暴民通过编辑文字和帧来改变其意义的《纽约时报》观点文章。图 4.8 展示了由 AI 工具以她的艺术风格生成的艺术品示例——文字明显混乱，但 Andersen 的一些视觉元素仍然存在。“当我检查了[`haveibeentrained.com`](https://haveibeentrained.com)网站时，这是一个允许人们搜索 LAION 数据集的网站，我的许多作品都在那里，以至于填满了我的整个桌面屏幕，”Andersen 证实，并担心 AI 工具会被用来再次扭曲她的创作[[56]](https://www.nytimes.com/2022/12/31/opinion/sarah-andersen-how-algorithim-took-my-work.xhtml)]。

Andersen 是三位原告之一，与 Karla Ortiz 和 Kelly McKernan 一起，对 Midjourney、Stability AI 和 DeviantArt 提起了集体诉讼。像 Andersen 一样，Ortiz 和 McKernan 同样发现这些工具能够以令人感到个人侵犯的方式生成他们风格的图像。“他们用我们的作品训练了这些模型。他们剥夺了我们决定是否想成为其中一部分的权利，”Ortiz 说[[56]](https://www.nytimes.com/2022/12/31/opinion/sarah-andersen-how-algorithim-took-my-work.xhtml)] [[57](https://www.buzzfeednews.com/article/pranavdixit/ai-art-generators-lawsuit-stable-diffusion-midjourney)]。

![图片](img/CH04_F08_Dhamani.png)

图 4.8 使用开源图像生成模型并以提示“Sarah Andersen 网络漫画”创建的 AI 图像

虽然 Andersen、Ortiz 和 McKernen 的诉讼结果尚待观察，但这些工具仍在世界各地被用来生成和实验新的艺术形式。公平使用的宽容结构意味着，对现状的任何重大改变都需要为在训练 AI 模型中使用树立新的先例。然而，与此同时，我们谈论的许多数据集和模型已经是开源的，这意味着任何人都可以训练自己的模型或创建现有模型的新版本。无论任何特定公司是否改变其数据集构建程序，或者最终支付损害赔偿金或许可费，从漫画到音乐到诗歌，AI 生成的艺术都将持续存在。

## 开源和许可

我们已经提到，由于生产 LLMs 需要巨大的数据和计算规模，这项工作到目前为止主要留给了一些主要的技术公司和一些资金充足的初创公司。然而，由于开源社区的出现，这种情况正在改变。“开源”指的是软件的源代码对公众开放并可重用和修改。更重要的是，开源是一种运动，其倡导者认为开源软件是公共产品，通过更多的协作和参与以及降低进入门槛，可以产生更好的软件。同样，开放数据运动的倡导者建议，当数据广泛可访问时，公众将更加知情，因此政府和非营利组织、科学研究以及其他实体收集或产生的数据应该可以自由使用和构建。

“开源”指的是软件的源代码对公众开放并可重用和修改。

如果有什么的话，生成式图像模型在这方面已经领先于 LLMs。由于对计算机视觉模型的高度兴趣，自从斯坦福大学计算机科学教授 Fei-Fei Li 开始了一个名为 ImageNet 的项目以来，学者们就已经编制了大量的图像数据集。2006 年，Li 有一个先见之明的想法，即计算机视觉的最大进步不一定来自新的、更好的算法，而是来自更好的（以及更大的）数据。她开始创建一个数据库，ImageNet，它最终将包含数百万张描绘数百种事物的图片：动物、家庭用品、地形，以及许多其他类别。经过最初的许多怀疑之后，ImageNet 成为所有计算机视觉模型衡量其结果的标准。它不仅启动了目标检测问题（现在在 ImageNet 上被认为已经“解决”，因为最先进的模型可以近乎完美地执行），而且还迎来了一个为训练和测试模型共享基准数据集的时代。关于 ImageNet 的影响，Li 说：“现在有各种各样的数据集如雨后春笋般涌现和绽放，从视频到语音、游戏到一切。”当然，这也证实了她最初假设的证明点，后来 LLMs 的成功也证实了这一点 [[58]](https://qz.com/1034972/the-data-that-changed-the-direction-of-ai-research-and-possibly-the-world)。

因此，从自然语言到图像和视频等各个问题领域，对数据的贪婪是有益的。像后来的数据集一样，ImageNet 是由互联网上的图片组成的，然后由 Amazon Mechanical Turk（一个众包平台）上的工人进行标注。通过编写少量代码，人们可以通过程序访问网页并复制其内容来编译文本和图像数据。这种做法被称为*网络爬虫*，只要数据是公开可用的，就被反复发现是合法的[[59](https://techcrunch.com/2022/04/18/web-scraping-legal-court/)]——因此几乎任何你在网上浏览到的内容都可以。例如，任何被搜索引擎索引的网站都会被爬虫抓取。一些运营网站的公司，包括 Reddit、Twitter 和 Stack Overflow，这些网站是大型语言模型（LLM）频繁的数据来源，已经公开表示计划向 AI 开发者收费以使用这些数据，尽管在实践中的具体形式尚不清楚——最有可能的是，他们会出售无需爬取的数据集[[60](https://www.zdnet.com/article/stack-overflow-joins-reddit-and-twitter-in-charging-ai-companies-for-training-data/)]。维护网站的人可以添加一个 robots.txt 文件，这本质上是一组针对爬虫的指令，告诉爬虫它可以抓取哪些页面以及哪些页面不应该抓取。在实践中，robots.txt 文件只是建议性的，恶意程序可以轻易地忽略它们。

尽管公开可用的网络内容法律限制不多，但代码和数据都有相应的许可。一些开源许可明确允许所有类型的衍生使用。例如，麻省理工学院许可证（MIT License）是一种宽松的软件许可证——实际上，这是 GitHub 上最受欢迎的许可证之一——允许在专有软件中重用[[61]](https://github.blog/2015-03-09-open-source-license-usage-on-github-com/)]。其他许可证仅允许非商业用途的重用；还有一些许可证可能允许在注明出处的情况下重用，或者满足其他条件。代码和数据许可证在法律上是可执行的[[62]](https://www.nytimes.com/2008/08/14/technology/14commons.xhtml)]。

代码许可证是软件开发商对微软、GitHub 和 OpenAI 提起的集体诉讼中的核心问题，该诉讼涉及 LLM 工具 Copilot。Copilot 基于 OpenAI 的 GPT-3 模型的变体，特别适用于编写代码，并且是在数千个 GitHub 仓库上训练的。与版权问题一样，关于使用此代码进行 LLM 训练的诉讼正在进行中；不清楚依赖许可而不是合理使用将如何运作。该案的原告认为这种使用相当于“前所未有的软件盗版”，而被告则表示，原告通过要求“禁令和数十亿美元的意外之财”来“分享他们自愿分享的软件”，正在破坏开源的原则。[63](https://www.theverge.com/2023/1/28/23575919/microsoft-openai-github-dismiss-copilot-ai-copyright-lawsuit)

同时，像 Hugging Face 这样的公司对开源原则持乐观态度，它们构建和托管免费使用的模型和数据集。[64](https://huggingface.co/)。与任何著名 AI 实验室无关的人也能够访问这个快速迭代和共享的生态系统，并在某些情况下改进最先进的结果。这伴随着一定的风险，因为任何旨在减少某些危害的限制都可能被下游用户移除。这将更难防止复制内容的创作或执行现有的版权。

尽管如此，我们有理由相信这些问题不会扼杀创造力，反而会促进它。互联网活动家和作家 Cory Doctorow 长期以来一直批评版权，指出尽管这些权利的条款随着时间的推移而变得更长、更广泛，但创作者并没有从中获利——购买他们版权的公司[65](https://doctorow.medium.com/copyright-wont-solve-creators-generative-ai-problem-92d7adbcc6e6)。Doctorow 对进一步扩大版权以防止生成模型访问这些作品进行训练持怀疑态度，他写道：

基本上，机器学习系统会摄入大量的作品，分析它们，找出它们之间的统计相关性，然后利用这些相关性来创作新的作品。这就像是每个创作者都会做的事情的数学化版本：分析他们所钦佩的作品是如何制作的，以便他们可以创作出自己的新作品。如果你翻阅一本艺术书籍，分析你喜欢的画作中的色彩方案或鼻与额头之间的比例，你并没有侵犯版权。我们不应该创造一个新的权利来决定谁有权深入思考你的创意作品并从中学习——这样的权利将使下一代创作者无法（合法地）学习他们的技艺。[65](https://doctorow.medium.com/copyright-wont-solve-creators-generative-ai-problem-92d7adbcc6e6)

人们可能合理地就是否以及如何在大规模模型上使用受版权保护的数据存在分歧。可以肯定的是，随着这些案件的继续发展和先例的建立，我们将从法律角度获得更多的清晰度。但早期的艺术家也对摄影的发明感到担忧，因为没有人会继续绘画或购买画作，因为他们无法在描绘现实方面与相机竞争。相反，艺术家继续绘画，但他们通过自己的解释和表达来传达场景 [[66]](https://www.thecollector.com/how-photography-transformed-art/)。似乎生成模型将成为另一种媒介，而永远不会完全满足人类对美的需求，也不会取代人类对创造力的冲动。

## 摘要

+   综合媒体，或更具体地说，AI 生成媒体，是一个涵盖使用 AI 创建或修改的内容的总称，这些内容包括文本、图像、视频、声音和数据。

+   术语*深度伪造*——由“深度学习”和“伪造”组合而成——有时与视觉合成媒体同义使用，但它通常带有负面含义。

+   自动编码器使用神经网络来压缩和解压缩图像，它们通常用于人脸交换技术。

+   GANs 由两个神经网络组成——一个生成器和一个人工智能判别器。生成器存在是为了创建新的数据，例如图像，判别器通过将图像与训练数据集进行比较来验证图像的真实性，以确定虚假图像和真实图像之间的差异。

+   综合媒体正在使内容创作和创意对每个人来说都更加民主化，同时引领新一波创意和艺术。

+   生成式 AI 也臭名昭著地被用于创建虚假/错误信息内容、名人色情视频、报复色情或网络性骚扰、欺诈和间谍活动。

+   为了对抗深度伪造，采用一种综合方法来检测 AI 生成媒体，包括技术解决方案、媒体素养和教育，以及适当的立法来规范技术的使用是至关重要的。
