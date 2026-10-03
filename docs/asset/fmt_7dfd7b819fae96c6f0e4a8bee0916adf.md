
本章演示了如何克服在生产环境中部署 LLM 时出现的主要约束。本章中的五种模式应对了生产约束的不同方面：计算开销、硬件利用率、延迟和记忆。关于小语言模型（模式 24）的小节向你展示了如何通过模型蒸馏和量化技术来降低计算开销。关于提示缓存（模式 25）的小节演示了如何消除冗余处理并降低成本和延迟。关于推理优化（模式 26）的小节涵盖了连续批处理和投机解码等先进技术，以最大化硬件利用率。关于推理分布测试（模式 27）的小节提供了验证基于 LLM 的应用程序性能所需的指标，以及当它在某些性能方面表现不足时可以采取的措施。关于长期记忆（模式 28）的小节帮助你在不触及上下文窗口限制的情况下长时间维护用户历史。

表 8-2 总结了我们本章讨论的模式。

**表 8-2. 满足约束的模式**

| 模式 | 问题 | 解决方案 | 使用场景 |
| --- | --- | --- | --- |
| **小语言模型（SLM）（模式 24）** | 你正在使用的基础模型带来了过多的延迟或成本。 | 使用小型基础模型，通过量化（降低模型参数的精度）、蒸馏（缩小知识范围）或投机编码（用更大的模型作为后备）来适应成本和延迟约束，同时不过度牺牲质量。 | 窄范围知识应用、成本降低、边缘设备部署、更快的推理需求以及 GPU 受限环境 |
| **提示缓存（模式 25）** | 用户请求遵循重复查询的模式。重新计算相同的响应会浪费资源并增加成本。 | 针对相同或相似的提示，复用先前生成的响应（客户端缓存）和/或模型内部状态（服务端缓存）。相似性可以基于提示含义（语义缓存）或重叠（前缀缓存）。 | 具有重复查询的应用、成本优化、需要快速响应的交互式应用以及多租户系统 |
| **推理优化（模式 26）** | 自托管 LLM 带来了 GPU 约束和硬件利用率挑战。实时应用需要更快的响应时间。 | 该模式通过采用连续批处理（从队列中拉取请求，并在 GPU 核心可用时立即将其放入）、投机解码（只要较小模型能够计算下一组 token，就高效地计算，并用大型模型作为后备）和/或提示压缩（预处理提示以使其更短）来提高模型推理效率。 | 自托管 LLM 部署、实时应用、GPU 内存受限环境以及高吞吐量服务场景 |
| **降级测试（模式 27）** | 你需要指标来帮助识别服务质量何时下降以及应用程序受哪些约束限制。 | 一组核心指标——首 token 时间（TTFT）、端到端请求延迟（EERL）、每秒 token 数（TPS）和每秒请求数（RPS）——以及各种可扩展性和韧性指标，可以帮助识别服务质量下降。有针对性的干预可以帮助改进特定指标。 | 预生产测试、性能验证、瓶颈识别、容量规划、持续监控和优化 |
| **长期记忆（模式 28）** | LLM 应用需要通过将相关历史附加到每个提示中来模拟过去交互的记忆，但在长对话中，由于上下文窗口的限制，这种方法可能变得昂贵且低效。 | LLM 应用使用各种类型的记忆——工作记忆、情景记忆、程序性记忆和语义记忆——分别用于维持上下文、回忆过去的交互、个性化响应和保留关键事实。 | 聊天机器人、多步骤工作流、个性化和处理大型文档 |

# 第 9 章 设置安全防护

GenAI 应用总是存在一定程度的风险。这是因为它们构建在基础模型之上，而基础模型是一种非确定性技术，有可能提供不准确或幻觉式的答案。基础模型也是一种通用技术，因此它们的响应不一定总是符合你的期望。

在本章中，我们将讨论四种模式，它们可以帮助你为 GenAI 应用程序设置安全防护。**模板生成**（模式 29）适用于未经人工审核就发送内容的风险非常高、但人工审核又无法应对通信量的场景。**组装重排**（模式 30）适用于内容需要以吸引人的方式呈现、但动态生成内容所带来的风险过高的场景。**自我检查**（模式 31）可以帮助你以经济高效的方式识别潜在的幻觉。最后，**安全护栏**（模式 32）是一种包罗万象的方式，可以为你的核心 GenAI 应用设置防护，确保它们在合乎道德、法律和功能的参数范围内运行。

## 模式 29：模板生成

模板生成模式通过预生成可离线审核的模板，减少了需要人工审核的项目数量。在推理时，应用程序只需对已审核的模板执行确定性的字符串替换即可。这使得最终响应无需额外审核即可安全地发送给消费者。

### 问题

LLM 是一项强大的技术，但它们并非确定性的，因此其响应总存在不准确或有害的风险。

例如，假设你是一家旅行社运营商，正在为购买了旅行套餐的顾客生成感谢信。你希望这些感谢信个性化、可读性强，甚至可能支持多种语言。你很想使用 LLM 来生成这些信件，但你深知这将使你的品牌面临相当大的风险。如果信件中包含不当语言，或试图推销不当或有争议的物品，该怎么办？你可以增加人工审核步骤，但面对每天可能数千笔的订单，人工审核的成本会变得很高。

有没有一种方法既能使用 LLM 生成感谢信，又能避免每封信都要经过人工审核的开销呢？

### 解决方案

与其直接用 LLM 生成感谢信，不如使用 LLM 为感谢信生成模板（参见图 9-1）。这些模板可以由人工审核并适当编辑。你还可以使用少样本学习以及第 2 章到第 8 章中的许多模式，来确保生成的内容有据可依、符合你的品牌风格等，从而将编辑工作量降到最低。

![图示说明生成和利用可定制感谢信模板的过程，包括人工审核步骤以及 LLM 的使用，确保在最后阶段实现最小化编辑和错误减少。](img/gaid_0901.png)

*图 9-1. 模板生成：创建少量可经人工审核的模板，然后在推理时直接使用*

在推理时，应用程序只需将值填入模板并发送信件即可。由于这一过程是确定性的且受限的，在此阶段引入事实错误或不良内容的可能性极低。

### 示例

你经营一家旅行套餐公司，每次预订后不久，你都会以带队导游的名义发送一封个性化感谢信，因为这往往能减少取消订单的数量。[本示例的代码可在 GitHub 上获取](https://oreil.ly/AbUr1)。

#### 预生成循环

你可以为你提供的每种旅游目的地、套餐类型和语言组合，预生成模板化的感谢信，并将这些模板存储在数据库中以便快速检索：

```
DESTINATIONS=[
    "Toledo, Spain",
    "Avila & Segovia",
    "Escorial Monastery"
]
PACKAGE_TYPES=[
    "Family",
    "Individual",
    "Group",
    "Singles"
]
LANGUAGES=[
    "English",
    "Polish"
]
for dest in DESTINATIONS:
   for package_type in PACKAGE_TYPES:
      for lang in LANGUAGES:
          template = create_template(dest, package_type, lang)
          db.insert(dest, package_type, lang, template)
```

你的业务提供三个目的地、四种套餐类型以及两种语言的旅行产品。因此，你将生成 3 × 4 × 2 = 24 个模板。你可以通过提示一个基础模型来创建每个模板。

#### 生成模板

一些占位符（例如语言）将被预生成循环替换，但生成的模板中仍会包含像客户姓名这样的占位符。这些剩余的占位符需要在发送前被填充：

```
def create_template(tour_destination, package_type, language):
    prompt=f"""
    You are a tour guide working on behalf of Tours GenAI S.L. Write a 
personalized letter in {language} to a customer who has purchased a 
{package_type} tour package to visit {tour_destination}. Sound excited to see
them and lead them on the tour. Explain some of the highlights of what they will
see there and some of the things they can do while there. In the letter, use
[CUSTOMER_NAME] to indicate the place to be replaced by their name and
[TOUR_GUIDE] to indicate the place to be replaced by your name.
    """
    template = zero_shot(GEMINI, prompt)
    # ask humans to edit/confirm ...
    template = human_edit_confirm(template)
    return template
```

为一个西班牙托莱多英语之旅生成的模板可能如下所示：

> 亲爱的 `[CUSTOMER_NAME]`，
>
> 我非常激动地欢迎你来到托莱多！我是 `[TOUR_GUIDE]`，将担任你家庭之旅的导游。我非常高兴能向你及你的家人展示这座令人惊叹的城市。
>
> ...
>
> 这里提前预览一下等待你的精彩内容：
>
> * **宏伟的托莱多大教堂：** 一座哥特式建筑的杰作，定会让你叹为观止。
> * **托莱多城堡：** 一座坚固的要塞，可以饱览城市全景。
>
> ...
>
> 我期待与你和你的家人见面，并与你们分享我对托莱多的热情。准备好迎接一场难忘的冒险吧！
>
> 期待很快见到你，
>
> `[TOUR_GUIDE]`

另一方面，一封用于西班牙阿维拉和塞哥维亚之旅的波兰语信件可能这样开头：^(1)

> Szanowni Państwo, `[CUSTOMER_NAME]`!
>
> Z ogromną radością witam Państwa w imieniu Tours GenAI S.L.! Jestem `[TOUR_GUIDE]` i będę miał przyjemność być Państwa przewodnikiem podczas rodzinnej wycieczki do Avili i Segowii!

#### 推理

每当有旅行产品被购买，或确认了该次旅行的导游后，你的应用程序将调用电子邮件服务，并传入旅行详情。应用程序将从数据库中检索相应的模板，并用会话中的字符串替换占位符，从而获得邮件正文：

```
booked_tour = ...
template = db.retrieve(booked_tour.destination,
                       booked_tour.package_type,
                       booked_tour.language)
email_body = template.replace(
               "[CUSTOMER_NAME]", booked_tour.customer_name
             ).replace(
               "[TOUR_GUIDE]", booked_tour.tour_guide.name
             )
# send out email
```

### 注意事项

模板生成帮助你在生成内容的场景中，避免对每份内容进行人工审核所产生的成本和时间延迟。只要所需模板的数量是可管理的，它就能很好地工作。如果组合数量过多，请考虑**组装重排**（本章中的模式 30）。模板生成的另一个替代方案是使用**安全护栏**（本章中的模式 32），但这会显著增加工程复杂性。

您可以在广泛的个性化和定制场景中，将模板生成与机器学习（ML）结合使用。例如，您可以使用预生成的模板构建一组个性化落地页，然后利用 ML 来决定在特定营销活动或用户画像中采用哪些模板。对于个性化推荐，您可以使用传统推荐引擎来选择向用户展示哪些产品，并为该产品提取相应的预生成内容（图像和文本）。

如果您正在为不同的客户旅程定制网站，也可以采用这种将预生成与 ML 相结合的方法。在这种情况下，您可以预先生成落地页，并使用*倾向模型*（一种预测事件（如购买）发生可能性的 ML 模型）来选择下一个最佳行动。

### 参考

*邮件合并*是一项通过将主模板与独立来源的数据结合，为批量分发个性化文档或电子邮件的功能。它[可追溯至](https://oreil.ly/uf-fS)20 世纪 80 年代名为 WordStar 的文字处理器。在 2024 年，Valliappa Lakshmanan 在一篇关于[平衡创造性与风险](https://oreil.ly/h5lpl)的 GenAI 应用文章中，提出了使用 LLM 创建这些模板而非最终文档的想法。

## 模式 30：组装式重格式化

组装式重格式化通过将内容创建任务分解为两个低风险步骤，来降低内容不准确或产生幻觉的风险。第一步涉及使用低幻觉方法（如 OCR、RAG、工具调用（模式 21）和模板生成（模式 29））来组装原始数据。第二步涉及使用 LLM 对组装后的内容进行重格式化，因为改写和总结等任务相对不太可能引入不准确信息。

### 问题

假设您正在为电子商务网站创建产品目录。产品目录需要对潜在买家和搜索引擎都具有吸引力。网站上有成千上万的产品页面，因此您希望使用 LLM 来生成目录的网页。

在这种情况下，幻觉或不准确内容会带来什么风险？例如，如果 LLM 生成的相机目录页面（该相机配有锂电池）写着电池是碱性电池，您将公司置于何种风险等级？锂电池不允许放入托运行李，因为它们可能在密闭空间中引发无法控制的火灾。如果您网站上的相机在飞机货舱中引燃或助长火灾怎么办？如果航空公司因为乘客行李中有易燃物品而拒绝其登机怎么办？我们希望您同意，这个看似简单的电池类型错误所带来的风险相当高，因此对于这个用例来说，动态生成风险过高。

有没有一种方法既能获得 LLM 生成的益处，又不会承担 LLM 可能引入不准确或幻觉信息的风险？

### 解决方案

组装式重格式化模式在以下情况下有效：您能够识别产品中若产生幻觉则会带来不可接受风险的特征，并且您可以使用低风险方法（例如从数据库读取或使用文档数据提取系统）来组装这些特征。

通过将所有对应于产品属性的文本附加在一起，或将它们放入某个预定义结构中来创建最终结果，所生成的文本是准确的，但读起来不太令人愉悦。

一旦您获得了这段准确的文本，将其放入提示的上下文中，并让 LLM 重新措辞、改写或重格式化（参见图 9-2）。由这些 LLM 任务生成的文本，其风险往往远低于从头生成的文本，且生成的文本将更加流畅，更符合该内容的用途。

![Diagram illustrating a two-step process: \&quot;Assemble\&quot; collects raw data from a database and RAG system, followed by \&quot;Reformat\&quot; where an LLM refines this data into the desired content form.](img/gaid_0902.png)

###### 图 9-2. 组装式重格式化通过将任务分解为两个低风险步骤，降低了与内容创建相关的风险

### 示例

对于产品目录案例（[完整代码在 GitHub 上](https://oreil.ly/47nBo)），您可以将要收集的原始数据定义为一个数据类：

```
@dataclass
class CatalogContent:
    part_name: str = Field("Common name of part")
    part_id: str = Field("unique part id in catalog")
    part_description: str = Field("""One paragraph description of part, which is 
one of the parts of a paper machine, explaining its purpose.""")
    failure_modes: list[str] = Field("""list of common reasons why customer might
need to replace this part.""")
    warranty_period: int = Field("""number of years that the part is under 
warranty""")
    price: str = Field("price of part")
```

您可以，从数据库中检索零件名称、ID、保修期和价格，但从设备手册中获取零件描述和失效模式（见上述斜体文本）。如果您使用 LLM 进行此类数据提取，请使用较低的温度，可能是 0.1 或 0，以降低不准确的风险。

对于一个用于造纸机器的名为*湿端*的零件，目录内容可能如下所示：

```
CatalogContent
part_name='wet_end'
part_id='X34521PL'
part_description='The wet end of a paper machine is the section where the paper 
web is formed. It is arguably the most important section of the machine.'
failure_modes=['Web breaks', 'Uneven sheet formation', 'Poor drainage'], 
warranty_period=3
price='$23295'
```

从可信来源提取的失效模式已用斜体标出。通过将后续生成基于这三种失效模式，LLM 将始终包含关于这三个问题的注意事项，并且也不会包含造纸机制造商可能不希望公开的其他失效模式。

一旦您获得了提取的内容，您可以将其传递给一个更具创造性的写作引擎，该引擎将对其进行重格式化以满足您的期望目标，例如搜索引擎优化和/或对潜在买家具有说服力。
