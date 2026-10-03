

# 第十二章：使用商业领域知识微调 LLMs

本章涵盖

+ LLM 的微调过程
+ 准备用于微调的数据集
+ 使用微调工具更好地理解过程

尽管大型语言模型（LLMs）在各个行业中的影响在主流媒体中得到了广泛的报道，但 LLMs 的普遍性和受欢迎程度已经为 AI 开源社区带来了一场静悄悄的革命。通过开放协作的精神和大型科技公司的支持，微调 AI 模型的能力越来越容易为 AI 爱好者所获得。这个机会产生了一个充满活力的社区，该社区正在尝试和分享各种过程和工具，这些过程和工具可以更好地理解微调是如何工作的，以及我们如何自己或团队合作来调整模型。

微调的主题非常广泛，深入探讨每个重要细节可能需要一本整本书。然而，通过利用开源社区创建和为开源社区创建的模型、数据集、平台和工具，我们可以对微调过程有一个欣赏。这些开源资源可以为我们准备一个未来，在那个未来中，我们可能会在我们的组织中微调我们的模型，以帮助构建基于上下文的 LLMs。然而，微调不仅关乎我们采取的方法，也关乎我们使用的工具。因此，在本章中，我们将详细了解微调过程的每个重要部分，并学习如何微调我们的模型。

## 12.1 探索微调过程

在我们学习更多关于微调工具之前，我们首先应该讨论微调包含的内容，并反思我们希望实现的目标。正如我们将看到的，微调涉及一系列步骤，这些步骤在更广泛的过程中发挥着重要作用。了解我们希望实现的目标不仅有助于我们评估微调后的模型，而且还能指导我们的微调方法。微调过程的每一步都包括不同的活动和挑战。尽管我们可能无法涵盖所有细节，但我们将学习到足够多的知识，以了解我们在微调模型时会发生什么，以及我们可能面临的挑战。

### 12.1.1 微调过程的地图

正如我们在第十章中发现的，微调是将已经经过某种训练的现有模型（称为基础模型）进一步使用额外的数据集进行训练的过程

+ 使模型更能适应上下文信息
+ 改变模型的声音
+ 帮助它响应特定的查询或指令

就像基础模型需要经过一系列步骤进行训练一样，在微调会话期间也有一系列步骤需要遵循。图 12.1 总结了这些步骤以及它们是如何相互关联的。

![图片](img/CH12_F01_Winteringham2.png)

图 12.1 精调过程中所采取的不同步骤的视觉表示

图 12.1 不一定涵盖微调的每一个细微之处，但它捕捉了我们通常期望成功创建一个微调模型的核心步骤。在本章中，我们将更详细地检查每个步骤，但首先让我们反思一下这个过程可能最重要的部分，即确定我们希望通过微调实现的目标。

### 12.1.2 目标设定

在微调 LLM 时，我们可能犯的最大错误之一就是没有明确的想法，不知道我们希望微调后的 LLM 做什么。未能为微调模型应该帮助解决的问题设定明确的目标可能会影响微调过程的各个方面，从数据准备到测试。鉴于 LLM 的不确定性，这并不意味着围绕我们期望从 LLM 中获得的具体信息来设定目标。但我们必须问自己我们想要什么类型的行为，以及它如何融入更广泛的背景。

为了说明这一点，让我们考虑两个不同的目标。一个是创建一个代码补全工具，该工具已经在我们的代码库上微调，以使用 LLM 的生成能力，同时不向第三方泄露知识产权。另一个是一个基于问答/聊天的 LLM，它为用户提供支持，该 LLM 已经在支持文档和客户数据上微调，以帮助回答问题。根据我们想要追求的目标，我们需要考虑以下细节：

+ *使用什么数据集？* 对于我们的代码补全场景，我们可能希望创建一个由逻辑部分组成的代码数据集，以微调我们的模型。我们也可能对使用包含开源代码的其他数据集感兴趣。这与问答聊天场景不同，在问答聊天场景中，我们会创建一个包括帮助指南和文档的数据集。

+ *使用什么模型？* 在撰写本文时，Hugging Face，一个 AI 开源社区网站，目前托管了 500,000+个不同的模型，这些模型都旨在服务于不同的目的，如生成、分类和翻译。在考虑我们的微调目标时，我们需要选择适合我们需求的模型。在我们的代码场景中，我们可能会选择一个已经在大量代码数据集上训练过的模型，这使得它更容易与我们的代码库进一步微调。在我们的问答/聊天 LLM 场景中，我们可能会想要一个已经训练成能够作为有能力的基于聊天的 LLM 的模型。

+ *使用多大的模型？* 我们需要问自己的另一个问题是我们需要多大型的模型？根据我们想要的模型大小，通常由其参数大小定义，也将决定我们的硬件需求。这里有一个必须考虑的权衡。如果我们希望我们的模型在响应时准确，那么就需要大量的硬件来托管和运行更大的模型。如果我们预算有限，或者我们部署模型的位置性能不佳，那么我们可能需要考虑一个可能不那么准确或对我们的请求响应不那么快的较小模型。

什么是 Hugging Face？

Hugging Face 是一个 AI 开源社区的平台，允许成员托管、部署和协作他们的 AI 工作。值得注意的是，Hugging Face 提供了一个托管数据集和模型的地方，以及通过其空间功能让其他人部署和交互 AI 应用的机会。还有其他付费功能，如自动训练，旨在使微调更容易，以及增加硬件空间以部署更复杂和资源密集型模型。类似于 GitHub 除了提供托管代码的能力外，还允许团队协作，Hugging Face 提供了一个分享和从 AI 社区成员那里学习的地方，并共同参与未来的 AI 项目。

这绝对不是考虑因素的详尽列表，随着我们进一步探索微调，我们将了解到我们需要决定的不同选项。幸运的是（如果你有预算），已经创建并实施了许多工具，使微调模型的实验更容易、更快。因此，尽管在开始微调之旅时有一个目标是很不错的，但我们仍然可以自由地更换模型、数据集等，以了解如何为特定情境创建最优化微调模型。

## 12.2 执行微调会话

考虑到微调模型目标的重要性，在本章中，我们将尝试使用项目的代码库来微调一个模型，以创建一个可以支持未来分析的支持模型。在实际情况中，这意味着拥有一个模型，当被问及我们的代码库时，可以给出与我们的项目相关的上下文敏感的回答。为此，我们将使用开源项目 `restful-booker-platform` 的代码（[`mng.bz/vJRJ`](https://mng.bz/vJRJ)）。

### 12.2.1 准备训练数据

尽管微调不需要像预训练模型那样大量的数据，但微调的成功在很大程度上依赖于我们使用的数据的大小和质量，这引出了我们需要什么类型的数据的问题。为了帮助我们理解数据集的大小、范围和格式，我们可以从像 Hugging Face 这样的网站中学到很多，那里存储着开源数据集（[`huggingface.co/datasets`](https://huggingface.co/datasets)）。在撰写本文时，一些值得注意的数据集包括：

+ *堆栈*—从 GitHub 等开源项目上抓取的 5.46 亿行代码示例（[`huggingface.co/datasets/bigcode/the-stack`](https://huggingface.co/datasets/bigcode/the-stack)）
+ *羊驼*—使用现有的大型语言模型（LLM）生成的 52,000 行合成数据（[`huggingface.co/datasets/tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca)）
+ *OpenOrca*—290 万行问题和回答数据（[`huggingface.co/datasets/Open-Orca/OpenOrca`](https://huggingface.co/datasets/Open-Orca/OpenOrca)）

查看这些数据集的每一个，我们可以看到它们包含不同类型的信息，从代码示例到问题和回答，以不同的方式创建。例如，The Stack 这样的数据集是基于从互联网上抓取的真实信息，而 Alpaca 是由人工智能合成的。我们还可以看到，与列表中的其他数据集相比，Alpaca 是一个规模较小的数据集，但这并不意味着它没有用。

什么是合成数据？

在人工智能训练和微调的背景下，合成数据是指人工生成看起来像真实用户数据，但实际上并非基于真实生活数据的过程。使用合成数据是训练和微调人工智能的有用技术，因为它可以提供进行训练或微调所需的数据。尽管如此，使用合成数据也存在副作用。首先，生成测试数据会有成本。例如，`gretel.ai`、`mostly.ai` 和 `tonic.ai` 等工具提供数据生成工具，但它们是有价格的。其次，也许更重要的是，研究表明，在纯合成数据上训练模型可能会影响模型响应的质量。这很有道理，因为真实数据将具有难以在人工智能生成数据中模拟的变异性随机性。

因此，在制定我们的要求时，我们需要考虑我们想要什么，目前有什么可用，以及我们可能需要自己构建什么。让我们回到我们的代码助手和问答模型示例。对于我们的代码助手 LLM，我们可能需要一个主要以代码为基础的数据集，而我们的问答模型则需要以自然语言编写的数据，并包含以键值格式（问题作为键，答案作为值）编写的问题和答案。正如我们所看到的，我们的目标指导我们关于所需数据类型的决策，但随之而来的是更多的问题，例如数据从何而来以及我们将如何格式化它。

我们已经看到，来自 Hugging Face（Kaggle 的网址[`www.kaggle.com/datasets`](https://www.kaggle.com/datasets)也是一个优秀的数据集来源）等网站上有许多公开的数据集。但如果我们试图微调一个模型，使其更符合我们的上下文，那么我们很可能会希望使用属于我们组织的数据来调整它。因此，在决定如何将其转换为适合训练的格式之前，我们需要进行一项关于可用数据、其质量以及我们如何获取它的练习。考虑 Alpaca 数据集的结构格式([`huggingface.co/datasets/tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca))。该数据集包括以下四个列：指令、输入、输出和文本。正如我们将在下一步学习的那样，根据我们微调模型的方式，我们将需要数据集的不同方面。例如，如果我们想微调一个问答模型，我们至少需要指令和输出列，以帮助调整它，使其能够回答预期的问题类型和回答类型。

挑战在于将原始数据转换为像我们在 Alpaca 数据集中看到的那种结构化格式。例如，如果我们考虑我们的问答模型场景，我们可能希望用我们的文档和支持文档来训练它。其中一些原始数据可能以问答格式存在，例如常见问题解答（FAQs），但我们的大部分数据可能不会这么直接。因此，我们需要找出一种方法来解析我们的数据，使其适合我们的数据集结构。更复杂的是，我们还需要自动完成这项工作，以生成足够的数据量，使其有用。手动完成也是一项选择，但可能会很昂贵。

数据准备案例研究

我们将要用于微调会话的数据集展示了我们在构建小型数据集时可能遇到的挑战。在我们的微调会话中，我们将使用一个之前创建的数据集，该数据集可以在 Hugging Face 上找到，网址为[`mng.bz/4pXa`](https://mng.bz/4pXa)。该数据集是一个 JSONL 格式的文档，包含 `restful-booker-platform` (RBP)的 Java 部分的解析部分，以及生成的指令，其中一部分如下所示（为了便于阅读，已添加换行符）：

```py
{"instruction": "What is the class declaration for
➥BrandingServiceIT","output": "@ExtendWith
➥@SpringBootTest
➥@ActiveProfiles
public class BrandingServiceIT { }"}
{"instruction": "How does the method setup Restito work for
➥BrandingServiceIT?","output": "
➥@BeforeEach
➥public void setupRestito(){
➥whenHttp(server).match(post("/auth/validate"))
➥.then(status(HttpStatus.OK_200));
➥}"}
```

这段代码片段将为我们提供调整所需的所有数据，但在我们开始之前，让我们探索一下它是如何制作的。考虑以下类，然后问问自己你会如何以逻辑的方式将这段代码拆分以进行微调？

```py
package com.automationintesting.api;

import com.automationintesting.db.BrandingDB;
import com.automationintesting.model.db.Branding;
import com.automationintesting.model.service.BrandingResult;
import com.automationintesting.requests.AuthRequests;
import com.automationintesting.service.BrandingService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import javax.validation.Valid;
import java.sql.SQLException;

@RestController
public class BrandingController {

    @Autowired
    private BrandingService brandingService;

    @RequestMapping(value = "/", method = RequestMethod.GET)
    public ResponseEntity<Branding> getBranding() throws SQLException {
        Branding branding = brandingService.getBrandingDetails();

        return ResponseEntity.ok(branding);
    }

    @RequestMapping(value = "/", method = RequestMethod.PUT)
    public ResponseEntity<?> updateBranding(@Valid @RequestBody
    ➥Branding branding, @CookieValue(value ="token", required = false) 
    ➥String token) throws SQLException {
        BrandingResult brandingResult = 
        ➥brandingService.updateBrandingDetails(branding, token);

        return ResponseEntity.status(brandingResult.getHttpStatus())
        ➥.body(brandingResult.getBranding());
    }

}
```

你会选择按文件逐个微调、按行逐个微调，还是其他方式？在尝试测试基于 RBP 代码库的模型微调的第一尝试中，我选择了逐行的方法。它创建了一个脚本，该脚本将遍历项目中的每个文件，并将文件的每一行添加到其行中，从而生成一个类似于以下示例表的数据表：

```py
id, content
1,  @RequestMapping(value = "/", method = RequestMethod.GET)
2,  public ResponseEntity<Branding> getBranding() throws
3,  SQLException {
4,  Branding branding = brandingService.getBrandingDetails();
5,  return ResponseEntity.ok(branding);
6,  }
```

这种方法的缺点是，尽管解析和存储数据很容易，但我最终得到的条目缺乏上下文，这意味着我调整的是像`}`或`@Autowired`这样的条目。这些条目并没有提供很多关于 RBP 项目的上下文或细节，并且引发了一个新的问题。可以与这些类型的条目配对的指令类型是什么？如果你还记得，在基于指令的微调过程中，我们会发送一个指令（有时附带额外的输入数据），然后比较响应与我们的预期输出。我们添加到像`}`这样的条目中的指令类型将不包含任何关于我们上下文的提示，并可能对微调模型产生响应异常和不希望的方式。这正是我在尝试基于逐行材料调整模型时发生的情况。

相反，我选择的方法（并在数据集中可以找到）是将代码分解成逻辑部分。这意味着，而不是逐行切割事物，一个文件将基于 Java 类中的不同属性进行切割。例如，从之前分享的类中选择的切片在示例表中看起来如下：

```py
id, content
1,  @RestController
public class BrandingController { }
2,  @Autowired
    private BrandingService brandingService;
3,  @RequestMapping(value = "/", method = RequestMethod.GET)
    public ResponseEntity<Branding> getBranding() throws SQLException {
        Branding branding = brandingService.getBrandingDetails();

        return ResponseEntity.ok(branding);
    }
```

