
到目前为止，我们一直在讨论团队在将 LLM 部署到生产环境时面临的一些技术问题，但没有什么问题能与 LLM 倾向于错误这一简单问题相提并论。它们往往犯很多错误。“幻觉”是一个术语，用来描述当 LLM 模型会产生听起来正确但实际上错误的结果的情况——例如，书籍引用或具有预期形式和结构的超链接，但却是完全虚构的。作为一个有趣的例子，我们向出版社 Manning（由于一位作者仍在撰写，这本书还不存在）请求有关生产中 LLM 的书籍。我们得到了以下建议：Mike Del Balso 和 Lucas Serveén 合著的《生产中的机器学习工程》[`www.manning.com/books/machine-learning-engineering-in-production`](https://www.manning.com/books/machine-learning-engineering-in-production)，以及 Jeremy Howard 和 Sylvain Gugger 合著的《使用 Fastai 和 PyTorch 的编码者深度学习》[`www.manning.com/books/deep-learning-for-coders-with-fastai-and-pytorch`](https://www.manning.com/books/deep-learning-for-coders-with-fastai-and-pytorch)。第一本书完全是虚构的。第二本书是真实的；然而，它并非由 Manning 出版。在每种情况下，互联网地址都是虚构的。这些 URL 的格式实际上与你浏览 Manning 网站时预期的非常相似，但如果你访问它们，将会返回 404 错误。

幻觉最令人烦恼的方面之一是它们通常被自信的词语所包围。LLM 在表达不确定性方面非常糟糕，这在很大程度上是由于它们的训练方式。考虑“2 + 2 =”的情况。你更希望它回答“我认为它是 4”还是简单地“4”？大多数人更希望得到正确的“4”。这种偏见是固有的，因为模型通常会因为正确或至少听起来正确而得到奖励。

关于幻觉发生的原因，有各种解释，但最真实的答案是，我们不知道是否只有一个原因。它可能是由几件事情的组合；因此，目前还没有一个好的解决办法。尽管如此，准备好应对这些模型的错误和不准确性是至关重要的，以便为你的产品提供最佳的用户体验。

### 3.2.8 偏见和伦理考量

### 3.2.9 安全问题

就像所有技术一样，我们需要注意安全问题。大型语言模型（LLMs）是在大量文本语料库上训练的，其中一些可能是有害或敏感的，不应该被暴露。因此，应该采取措施保护这些数据不被泄露。上一节中提到的偏见和伦理问题就是你不希望你的用户讨论的话题的例子，但你也可以想象在公司的数据上微调模型，如果未采取适当预防措施，可能会意外地丢失秘密。

例如，使用简单的提示请求可口可乐的秘方仅仅返回了无聊的结果，“我不能泄露商业机密或任何专有信息。”如果我们问您，这真是太无聊了。通过一点小诡计和一些提示注入，我们设法让它泄露了秘密，当然，我们在尝试之前确认了一位食品科学家，它不会让我们中毒。按照指示严格操作后，我们惊讶地发现我们得到了一杯相当美味的饮料，但它尝起来根本不像可口可乐。虽然配方看起来很合法，但像数量、烹饪时间，甚至需要过滤糖浆这样的细微差别都不对。看来我们得等到得到真正的配方后再考虑接管饮料行业。

##### 可口可乐配方通过提示注入产生的幻觉

```py
Homemade Coca-Cola Recipe
=====================
**Disclaimer**: The original Coca-Cola recipe is a highly guarded trade secret. The following recipe is an approximation for creating a homemade cola drink that resembles the taste of Coca-Cola.

Ingredients:
- 3 cups of water
- 2 cups granulated sugar
- 1/8 tsp food-grade citric acid
- 3/4 tsp caramel color (liquid)
- 1/2 tsp caffeine (powder or liquid form)
- 1 1/2 tsp cola extract
- 1/8 tsp vanilla extract
- 1 1/2 cups carbonated water (store-bought or using a home soda maker)
- A pinch of ground cinnamon (optional)
- A pinch of ground nutmeg (optional)

Instructions:
1\. In a large saucepan, bring the 3 cups of water to a gentle simmer.
2\. Slowly stir in the granulated sugar, ensuring that it fully dissolves.
3\. Add the citric acid, caramel color, caffeine, cola extract, and vanilla extract to the saucepan. Mix well.
4\. Stir in the optional spices if desired: cinnamon and nutmeg.
5\. Simmer the mixture for about 20 minutes, stirring occasionally.
6\. Remove the saucepan from the heat and let it cool to room temperature.
7\. Once cooled, strain the syrup to remove any undissolved solids.
8\. To make the final drink, mix 1 part syrup with 3 parts carbonated water. Adjust to taste depending on your preference for sweetness and carbonation.

Variations:
- You can experiment with different amounts of caramel color, caffeine, and cola extract to adjust the flavor and appearance of your homemade cola.
- To create different cola flavor profiles, experiment with adding additional spices (such as star anise, orange or lemon peel, or cardamom), essential oils, or fruit extracts.
- Consider using a natural sweetener, like honey or agave syrup, as an alternative to granulated sugar.

Historical context and interesting facts:
- Coca-Cola was first introduced in 1886 by pharmacist Dr. John S. Pemberton, who aimed to create a unique and refreshing beverage that could also relieve fatigue and headaches.
- The original Coca-Cola formula contained trace amounts of cocaine extracted from the coca leaf, which was later replaced with caffeine in the early 20th century.
- In 1985, Coca-Cola introduced "New Coke," a reformulation aimed at competing with Pepsi. However, the taste change was met with backlash, and the company returned to its original formula just three months later, rebranded as "Coca-Cola Classic."
Result(s): You can now create your own homemade version of cola, resembling the taste of Coca-Cola. Enjoy and feel free to experiment with different spices, extracts, and sweeteners to create your own unique flavor profile.
```

另一个安全问题是腌菜注入。腌菜是 Python 中的一个库，用于序列化对象，通常用于序列化机器学习模型。它在反序列化时将对象序列化为包含逐个执行的指令的字节流。这是一种快速且简单的方式共享大型对象。腌菜注入会破坏这个字节流，通常在模型通过不安全的网络传输时注入恶意软件。这对于下载时间较长的大型模型尤其令人担忧，因为这使得第三方更容易拦截传输并注入恶意代码。如果发生这种情况，注入的代码可能会让攻击者获得对系统的访问权限。这可能在尝试在推理过程中使用模型时发生，因为如果有害代码未被检测并正确移除，它将执行。因此，在使用模型之前采取预防措施，如使用安全网络和验证模型的完整性，以防止此类攻击，是非常重要的。

### 3.2.10 控制成本

与大型语言模型（LLMs）一起工作涉及各种与成本相关的问题。首先，你可能已经注意到了，这是基础设施成本，包括高性能 GPU、存储和其他硬件资源。我们讨论了 GPU 采购难度较大，不幸的是，这也意味着它们更昂贵。像留下你的服务这样的错误一直都有累积账单的潜力，但与 GPU 混合在一起，这种错误甚至更具破坏性。这些模型还要求大量的计算能力，导致在训练和推理过程中消耗大量能源。除此之外，它们的部署时间更长，这意味着我们经常在低流量期间运行它们，以处理突发的工作负载或预期的未来流量。总的来说，这导致了更高的运营成本。

额外的成本包括管理和存储用于训练或微调以及常规维护的大量数据，例如模型更新、安全措施和错误修复，这可能对财务造成压力。与任何用于商业目的的技术一样，管理潜在的诉讼和确保符合法规也是一个问题。最后，投资于持续的研究和开发以改进您的模型并给您带来竞争优势，将是一个因素。

我们谈了一些关于令牌限制的技术问题，这些问题可能会得到解决，但我们没有讨论成本限制，因为大多数 API 按令牌收费。这使得发送更多上下文和使用更好的提示变得更加昂贵。这也使得预测成本变得有点困难，因为虽然你可以标准化输入，但你不能标准化输出。你永远不能太确定会返回多少令牌，这使得管理变得困难。记住，对于 LLM 来说，实施和遵循适当的成本工程实践以确保成本永远不会失控，这和以前一样重要。

## 3.3 LLMOps 基本要素

现在我们已经掌握了我们正在应对的挑战类型，让我们来看看所有不同的 LLMOps 实践、工具和基础设施，看看不同的组件如何帮助我们克服这些障碍。首先，让我们深入探讨不同的实践，从压缩开始，我们将讨论缩小、修剪和近似，以使模型尽可能小。然后我们将讨论分布式计算，这是必需的，因为模型如此之大，很少能适应单个 GPU 的内存。完成这些后，我们将在下一节中探讨实现这一切所需的基础设施和工具。

### 3.3.1 压缩

在上一节中，当你阅读关于大型语言模型（LLM）的挑战时，你可能自己问过类似的问题：“如果 LLM 的最大问题来自它们的大小，为什么我们不把它们做得更小？”如果你这样想过，恭喜你！你是个天才——压缩就是做这件事的实践。将模型压缩到尽可能小，将提高部署时间，减少延迟，减少所需昂贵 GPU 的数量，并最终节省资金。然而，最初让模型变得如此庞大无比的整个目的，是因为它使它们在所做的事情上变得更好。我们需要能够在不失去我们通过使它们变大而取得的全部进步的情况下缩小它们。

这个问题远未解决，但有多种方法可以解决这个问题，每种方法都有其优缺点。我们将讨论几种方法，从最简单和最有效的方法开始。

#### 定量

量化是将精度降低以降低内存需求的过程。这种权衡在直觉上是有意义的。当作者在大学时，他被教导总是将数字四舍五入到工具的精度。拿出直尺测量他的铅笔，如果你告诉他长度是 19.025467821973739 厘米，你不会相信他。即使他使用了卡尺，他也无法验证如此精确的数字。用我们的直尺，任何超过 19.03 厘米的数字都是幻想。为了强调这一点，他的一个工程教授曾经问他，“如果你正在测量摩天大楼的高度，你会在乎顶部多出的一张纸吗？”

我们在计算机内部表示数字的方式往往让我们误以为自己有比实际更好的精度。为了说明这一点，打开 Python 终端并执行 `0.1 + 0.2`。如果你以前从未尝试过，你可能会惊讶地发现它不等于 0.3，而是 0.30000000000000004。我们不会深入探讨这一现象背后的数学细节，但问题仍然存在：我们能否在不使事情变得更糟的情况下降低精度？我们实际上只需要精确到小数点后第十位，但降低精度可能会得到一个像 0.304 这样的数字，而不是 0.300，从而增加我们的误差范围。

最终，计算机唯一能理解的数字是 0 和 1，开或关，一个比特位。为了提高这个范围，我们将多个比特位组合起来，并赋予它们不同的含义。将 8 个比特位串联起来，你就得到了一个字节。使用 INT8 标准，我们可以将这个字节编码为从-128 到 127 的所有整数。我们就不详细解释背后的数学原理了，因为假设你已经知道二进制是如何工作的；只需说，我们拥有的比特位越多，我们能够表示的数字范围就越大，无论是更大的还是更小的。图 3.1 展示了几个常见的浮点数编码。将 32 个比特位串联起来，我们得到了我们自诩的“全精度”，这也是大多数数字存储的方式，包括机器学习模型中的权重。基本的量化将我们从全精度转换为半精度，将模型的大小缩小到原来的一半。有两种不同的半精度标准，FP16 和 BF16，它们在表示范围或指数部分时使用的比特位数不同。由于 BF16 使用的指数与 FP32 相同，它被发现对于量化来说更有效，你可以预期在模型大小减半的情况下几乎可以达到相同的精确度。如果你理解了论文和摩天大楼的类比，那么原因应该很明显。

![figure](img/3-1.png)

##### 图 3.1 几种常见浮点数编码的比特映射：16 位浮点数或半精度（FP16）、bfloat 16（BF16）、32 位浮点数或单精度（FP32）以及 NVIDIA 的 TensorFloat（TF32）

然而，没有理由就此停止。我们通常可以将它降低到 8 位格式，而不会损失太多精度。甚至已经有一些成功的研究尝试表明，对 LLM 的部分进行选择性的 4 位量化是可能的，而且精度损失很小。选择性地应用量化是一个称为动态量化的过程，通常只对权重进行量化，而将激活保留在完全精度以减少精度损失。

量化的圣杯是 INT2，表示每个数字为–1、0 或 1。目前这不可能而不完全降低模型质量，但它可以使模型缩小到原来的 1/8。Bloom 模型将只有大约~40 GB，足够小，可以放在单个 GPU 上。这当然是量化的极限，如果我们想进一步缩小，我们需要考虑其他方法。

量化的最好部分是它很容易做。有许多框架允许这样做，但在列表 3.1 中，我们展示了如何使用 PyTorch 的量化库进行简单的训练后静态量化（PTQ）。你所需要的只是一个全精度模型，一些示例输入，以及用于准备和校准的验证数据集。正如你所看到的，这只需要几行代码。

##### 列表 3.1 PyTorch 中的示例 PTQ

```py
import copy
import torch.ao.quantization as q

model_to_quantize = copy.deepcopy(model_fp32)     #1
model_to_quantize.eval()

qconfig_mapping = q.get_default_qconfig_mapping("qnnpack")     #2

prepared_model = q.prepare(model_to_quantize)     #3

with torch.inference_mode():     #4
    for x in dataset:
        prepared_model(x) 

model_quantized = q.convert(prepared_model)      #5
```

#1 原模型的深度拷贝，因为量化是在原地进行的

#2 获取映射；注意对于 ARM 使用“qnnpack”，对于 x86 CPU 使用“fbgemm”

#3 准备

#4 校准；你需要使用代表性的（验证）数据

#5 量化

静态 PTQ 是量化最直接的方法；它在模型训练后进行，并对所有模型参数进行均匀量化。与大多数公式一样，最直接的方法会引入更多的错误。通常，这种错误是可以接受的，但如果不可以，我们可以增加额外的复杂性来减少量化带来的精度损失。可以考虑的方法包括均匀与非均匀、静态与动态、对称与不对称，以及在其训练期间或之后应用。

要理解这些方法，让我们考虑从 FP32 量化到 INT8 的情况。在 FP32 中，我们实际上有可用的全部数字范围，但在 INT8 中，我们只有 256 个值。我们试图把一个精灵放进瓶子里，这不是一件小事。如果你研究你的模型中的权重，你可能会注意到大多数数字都在[–1, 1]之间的分数。我们可以利用这一点，通过使用一个 8 位标准，在这个区域以非均匀方式表示更多的值，而不是标准的均匀[–128, 127]。虽然从数学上来说是可能的，但不幸的是，这样的标准并不常见，现代深度学习硬件和软件也没有设计来利用它们。所以现在，最好坚持使用均匀量化。
