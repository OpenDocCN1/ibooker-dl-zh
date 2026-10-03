
# 使用 LLM 生成单元测试

![MW 标志](img/logo-MW.png)

你是一位有经验的 Java 软件开发测试人员，请为以下方法创建单元测试：

然后我们提供它需要的代码来提供单元测试：

![MW 标志](img/logo-MW.png)

```py
public class AuthService {

   public HttpStatus deleteToken(Token token) throws SQLException {
        Boolean successfulDeletion = authDB.deleteToken(token);

        if(successfulDeletion){
            return HttpStatus.OK;
        } else {
            return HttpStatus.NOT_FOUND;
        }
    }
}
```

将此发送到 ChatGPT，返回以下结果：

![OpenAI 标志](img/logo-openai.png)

```py
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class TokenManagerTest {

    @Test
    void deleteToken_SuccessfulDeletion_ReturnsOK() throws SQLException {
        // Arrange
        TokenManager tokenManager = new TokenManager();
        Token token = new Token();

        // Act
        HttpStatus result = tokenManager.deleteToken(token);

        // Assert
        assertEquals(HttpStatus.OK, result);
    }
}
```

这看起来像是合理的单元测试 Java 代码，但如果将其添加到单元检查套件中，它将失败，因为生产代码中提供的 `authDB.deleteToken` 方法没有被正确处理。具体来说，如果我们想让这个单元检查工作，我们需要模拟 `authDB.deleteToken`。

现在，如果我们再次运行提示，但这次在输出最终答案之前先评估其解决方案，我们会得到不同的结果。所以首先我们将提示更改为一个新的提示，要求 LLM 确认检查的结果是否有效：

![MW 标志](img/logo-MW.png)

你是一位有经验的 Java 开发者，在测试中编写单元测试。为以下方法创建一个单元测试，该测试将被共享。在输出单元测试之前，检查每个单元检查的断言是否返回一个确定的结果。

我们随后提供了将要编写单元测试的代码：

![MW 标志](img/logo-MW.png)

```py
public class AuthService {

   public HttpStatus deleteToken(Token token) throws SQLException {
        Boolean successfulDeletion = authDB.deleteToken(token);

        if(successfulDeletion){
            return HttpStatus.OK;
        } else {
            return HttpStatus.NOT_FOUND;
        }
    }

}
```

将此发送到 ChatGPT 返回了以下结果：

![OpenAI 标志](img/logo-openai.png)

```py
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.sql.SQLException;

public class TokenDeletionTest {

    @Mock
    private AuthDB authDB;

    @Test
    public void testDeleteToken_SuccessfulDeletion() throws SQLException {

        // Mocking the behavior of authDB.deleteToken() to return true
        Token token = new Token();
        Mockito.when(authDB.deleteToken(token)).thenReturn(true);
        AuthService authService = new AuthService(authDB);
        HttpStatus result = authService.deleteToken(token);

        // Assertion: The method should return HttpStatus.OK
        Assertions.assertEquals(HttpStatus.OK, result);
    }
}
```

这次，我们可以看到，因为我们要求 LLM 在返回结果之前评估其解决方案，所以输出单元检查使用 `Mockito` 模拟 `authDB.deleteToken`。因此，如果我们观察到 LLMs 输出错误解决方案或它们开始产生幻觉，我们可以添加一个指令先评估解决方案，以最大限度地减少幻觉的发生。

## 活动 2.6

创建一个提示，要求 LLM 解决一个问题。观察其输出，看看它产生的解决方案是否正确。然后添加指令让 LLM 评估该解决方案。发生了什么？解决方案是否改变？这是否是一个改进？

## 2.5 与各种大型语言模型（LLMs）合作

到目前为止，我们一直在广义上讨论 LLMs，而在之前的例子中使用 OpenAI 的 ChatGPT 来演示它们的一般工作方式。然而，ChatGPT 只是我们可以使用的许多不同 LLMs 之一。因此，在我们结束本章之前，让我们熟悉一下 LLMs 之间的不同之处，并了解一些目前流行的模型和社区，以便我们增加找到适合工作的正确 LLM 的机会。

### 2.5.1 比较 LLMs

什么使一个 LLM 变得优秀？我们如何确定一个模型是否值得使用？这些问题并不容易回答。LLMs 的复杂性质、它们的训练方式以及所使用的数据使得这些系统对深入分析关闭，这损害了一些研究人员试图改进或阐明的一个领域。然而，这并不意味着我们不应该在 LLMs 的关键方面以及它们如何影响它们方面进行自我教育。我们可能不是试图探索 LLMs 深层内部运作的 AI 研究人员，但我们现在是或将成为它们的用户，并希望知道我们投入的资源是否在给我们带来价值。因此，为了帮助我们分解一些术语，并给我们一些关于 LLMs 如何不同的基础，让我们回顾一下在 LLMs 领域中讨论的一些关键属性。

#### 参数数量

如果你查看不同的 LLM，你可能会听到关于 LLM 拥有 1750 亿或 1 万亿参数数量的讨论。这有时可能感觉像是营销术语，但参数数量并不影响 LLM 的性能。参数数量本质上指的是模型中存在的统计权重数量。每个单独的权重都提供了构成 LLM 的统计谜题的一部分。所以，粗略地说，一个 LLM 拥有的参数越多，其性能越好。参数数量也可以让我们对成本有所感知。参数数量越高，运行成本就越高，并且可能会有部分成本转嫁给用户。

#### 训练数据

LLM 需要大量的数据进行训练，因此数据的大小和质量将对 LLM 的质量产生影响。如果我们希望 LLM 在响应请求时准确无误，仅仅投入尽可能多的数据是不够的。它需要的是能够以合理方式影响模型概率的数据。例如，我们在本章前面探讨的 Reddit 示例，其中 `r/counting` 子版块被用来训练 ChatGPT，导致它以奇怪的方式产生幻觉，这表明更多的数据并不一定是更好的。然而，与参数数量类似，一个 LLM 训练时所使用的高质量数据越多，其性能可能越好。挑战在于了解 LLM 训练所使用的数据——这是企业 AI 创造者热衷于保密的事情。

#### 可扩展性和集成

就像任何其他工具一样，如果 LLM 能够提供其核心能力之外的其他功能，其价值可以进一步提高，例如集成到现有系统或进一步训练以满足我们的特定需求。可集成和扩展 LLM 的功能很大程度上取决于谁负责了训练。

例如，OpenAI 提供了付费 API 访问其模型的服务。但除了允许你通过简单的提示调整输出的指令功能外，没有进一步微调和部署其 GPT 模型用于私人使用的功能。与此相比，Meta 的 `LlaMa` 模型已被开源，允许 AI 社区下载并根据自身需求进一步训练，尽管他们必须建立自己的基础设施来部署该模型。

随着大型语言模型（LLM）平台的增长，我们将看到它们不仅对提示的响应能力有所提升，而且其周边功能和访问性也将得到改进。因此，在评估要与之合作的内容时，有必要牢记这些功能。

#### 响应质量

不可否认，最重要的考虑因素是 LLM 提供的响应是否易于阅读、有用，并且尽可能没有幻觉（或尽可能接近免费）。尽管参数数量和训练数据等标准是衡量 LLM 性能的有用指标，但了解我们想要使用 LLM 做什么，然后确定每个 LLM 如何响应我们的提示并帮助我们解决具体问题，这取决于我们。我们面临的挑战并不都需要市场上最大、最昂贵的 LLM。因此，花时间尝试不同的模型、比较它们的输出，然后自己做出判断是很重要的。例如，OpenAI 的 GPT 模型在代码示例方面的表现优于 Google Gemini。这些细节是通过实验和观察发现的。

我们所探讨的标准绝对不是详尽的列表，但它们表明，一旦我们超越了 LLMs 最初的光环，还有更多关于 LLMs 需要考虑的地方。不同的 LLMs 表现不同，帮助我们应对各种挑战。因此，让我们来看看一些目前更受欢迎的模型和平台。

### 2.5.2 检查流行的 LLMs

自从 OpenAI 发布 ChatGPT 以来，各种组织发布的 LLMs 数量激增。这并不是说这些模型和相关工作在 ChatGPT 发布之前不存在，但公众的关注确实加剧了，越来越多的营销和发布公告都集中在发布 LLM 产品的公司。以下是一些自 2022 年底以来发布的更常见/流行的 LLMs。

#### 跟踪大型语言模型（LLMs）

值得注意的是，LLMs 及其相关功能的发布情况极其灵活，发展速度相当快。因此，我们可能会发现，从 2024 年中期写作时到您阅读这本书的时间，我们将探索的内容可能会有所不同。幸运的是，像 LLM Models ([`llmmodels.org/`](https://llmmodels.org/))这样的网站分享了最新的列表以供审查。然而，这个列表表明，LLM 领域的一些知名公司值得探索。

#### OpenAI

在写作时，OpenAI 是提供 LLMs 使用的最普遍的组织。尽管 OpenAI 已经研究 LLM 模型有一段时间了，2020 年发布了 GPT-3 模型，但直到 2022 年 11 月发布 ChatGPT，才引发了人们对 LLMs 的兴趣和使用的热潮。

OpenAI 提供了一系列不同的 LLM 模型，但最突出的是 `GPT-3.5-Turbo` 和 `GPT-4o`，您可以在[`platform.openai.com/docs/models/overview`](https://platform.openai.com/docs/models/overview)了解更多信息。这两个模型被用作*基础*模型或可以进一步训练以用于特定目的的模型，用于 ChatGPT、GitHub Copilot 和 Microsoft Bing AI 等一系列产品。

除了他们的模型，OpenAI 还提供了一系列功能，例如直接访问他们的 `GPT-3.5-Turbo` 和 `GPT-4` 模型的 API，以及与 ChatGPT 集成的应用程序集合（如果您订阅了他们的 plus 会员）。到目前为止，它是最受欢迎的 LLM（至少目前是这样），并引发了一波组织发布他们自己的 LLM 的竞赛。尽管我们已经探索了一些与 ChatGPT 相关的提示，但您始终可以访问并实验 ChatGPT，网址为[`ChatGPT.com/`](https://ChatGPT.com/)。

##### 坚持使用 OpenAI

尽管有许多不同的 LLM（大型语言模型）我鼓励您使用，但为了保持一致性，我们将坚持使用 `ChatGPT-3.5-Turbo`。它不一定是最强大的 LLM，但它是分布最广泛的——而且是免费的。话虽如此，如果您想尝试使用其他 LLM 模型来测试这些提示，请随意。然而，请记住，它们的响应可能会与本书中分享的内容不同。

#### Gemini

无怪乎，Google 也在生成式 AI 市场中占有一席之地，他们有一系列名为 Gemini 的 LLM 模型。截至撰写本文时，他们最强大的模型是 `Gemini 1.5 Pro`，但他们还提供其他版本的模型，如 `Gemini 1.5 Flash` 和 `Gemini 1.0 Pro`。鉴于是 Google，每个模型的参数数量不是公开信息，但它们与其他 LLM 模型在性能上相对可比较。

与 OpenAI 类似，Google 通过他们的 Google Cloud 平台提供对 Gemini 模型的访问（[`ai.google.dev/`](https://ai.google.dev/)），并且最近开始提供类似于 OpenAI ChatGPT 应用程序的应用程序，增加了与其他 Google Suite 工具（如 Google Drive 和 Gmail）的集成。您可以在[`gemini.google.com/app`](https://gemini.google.com/app)访问并实验 Gemini。

#### LLaMa

LLaMa 这个名字代表一系列模型，首次由 Meta 在 2023 年 7 月发布。LLaMa 与 OpenAI 的 GPT 模型和 Google 的 Gemini 模型的不同之处在于 LLaMa 是开源的。除了开源许可外，LLaMa 还提供多种大小：分别是 80 亿和 700 亿参数。这些大小及其可访问性意味着 LLaMa 已被 AI 社区采纳为流行的基础模型。然而，这种可访问性的另一面是，Meta 不提供公共平台来训练和运行 LLaMa 的版本。因此，必须个人获取数据集和基础设施才能使用。

关于 LLaMa 的更多详细信息可以在以下链接中找到：

+   [`ai.meta.com/blog/meta-llama-3/`](https://ai.meta.com/blog/meta-llama-3/)
+   [`www.llama.com/llama-downloads/`](https://www.llama.com/llama-downloads/)

#### Hugging Face

与我们列表中的其他条目不同，Hugging Face 不提供专有模型，而是促进了一个包含各种不同模型的 AI 社区，其中大多数是开源的。查看他们提供的模型索引页[`huggingface.co/models`](https://huggingface.co/models)，我们可以看到来自不同公司和研究实验室的数十万个不同训练的模型。Hugging Face 还提供了用于训练的数据集、应用程序和文档，允许读者深入了解模型是如何构建的。所有这些资源都是为了使 AI 社区能够访问预训练模型，对其进行调整，并针对特定用途进行进一步训练，这是我们将在本书的第三部分进一步探讨的内容。

LLM 的市场在短时间内显著增长，无论是商业上还是开源领域，这与其他软件开发领域类似，积极关注新出现的 LLM 可能是有益的。然而，这可能会让人感到不知所措，并且不一定能够跟上所有同时发生的事情。因此，我们不妨选择在我们想要使用 LLM 解决特定问题时去探索 LLM。拥有问题可以帮助我们确定哪些工具最适合我们。

## 活动 2.7

要么选择本章中的一个早期提示词，要么创建自己的提示词并提交给不同的 LLM。注意每个 LLM 的响应，并进行比较。其中一些是否感觉更像是对话？它们如何处理接收或发送代码示例？在你看来，哪一些提供了最好的响应？

## 2.6 创建提示词库

提示词的一个好处是，一旦创建，就可以重复使用。因此，针对不同角色和任务的提示词集合在网上大量出现。例如，以下是一些我最近看到的分享的集合：

+   Awesome ChatGPT Prompts，GitHub ([`github.com/f/awesome-ChatGPT-prompts`](https://github.com/f/awesome-ChatGPT-prompts))
+   50 个面向开发者的 ChatGPT 提示词，Dev.to ([`mng.bz/w5D7`](https://mng.bz/w5D7))
+   ChatGPT 速查表，Hackr.io ([`mng.bz/q0PK`](https://mng.bz/q0PK))

这个列表远非详尽无遗，样本集合也不一定与测试相关，但它们值得浏览，以了解他人是如何创建提示词的，同时也为我们提供了确定哪些提示词有效、哪些无效的机会。

虽然公开共享的提示集合可能很有用，但我们很可能会创建出适用于特定情境的提示。因此，养成将有益的提示存储在某种类型的存储库中，以便我们和其他人可以快速使用的习惯是值得的。你将存储这些提示的位置将取决于它们的使用目的和对象。如果是供公众使用，那么共享提示存储库或添加到现有集合中可能是有价值的。如果我们是在开发公司产品的同时创建和使用它们，那么我们需要像对待我们的生产代码一样对待它们，并将它们存储在私密的地方，以免违反任何关于知识产权的政策。最后，我们还可以考虑版本控制，这样我们就可以随着我们对与 LLMs 合作的学习以及 LLMs 本身的演变来调整和跟踪提示。

无论它们存储在哪里，目的是创建一个提示存储库，这些提示快速且易于访问，以便一旦为特定活动创建了一个提示，就可以快速多次重用，从而我们可以从它们中获得尽可能多的价值来提高我们的生产力。

## 活动 2.8

创建一个可以存储未来提示的空间供你和你的团队使用。

## 使用本书中的提示
