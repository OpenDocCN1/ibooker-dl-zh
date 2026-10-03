

# 第九章：AI 代理作为测试助手

本章涵盖

+ 理解如何使用 LLMs 构建 AI 代理
+ 创建一个基本的 AI 代理来展示其价值

在前面的几章中，我们看到了大型语言模型（LLMs）如何帮助我们进行测试。我们还学习了如何使用各种提示技术充分利用 LLMs，并编制出在需要时可以使用的不同提示。这是一个很好的位置，但如果我们能将我们的新理解再向前迈进一步，创建定制的 AI 测试助手会怎样呢？

随着 LLMs 的进步，创建 AI 代理、可以自主与其他系统交互、收集数据、分析信息并适应适当响应以实现目标的应用程序的机会也在增加。在 AI 领域，代理可以以许多方式实现，但目标往往相同——创建我们可以分配任务以解决的东西。设计和构建 AI 代理的范围很广，但在这章中，我们将更多地了解它们的潜力和在生成式 AI 背景下的工作方式。我们还将创建我们自己的基本测试数据 AI 代理，以展示这项技术的力量和潜力。

## 9.1 理解 AI 代理和 LLMs

当我们讨论 AI 代理时，我们需要明确我们所说的意思。AI 代理可以根据我们工作的 AI 领域以不同的方式实现。但这并不意味着无论其工作方式如何，代理都没有预期的行为。因此，在我们开始实现我们的代理之前，让我们首先定义这些预期行为可能的样子，然后讨论在生成式 AI 的背景下如何构建代理。

### 9.1.1 什么是 AI 代理？

要理解什么是代理，我们必须更多地关注其中注入的特性，而不是它的实现。没有明确的特性列表表明代理应该满足，但以下属性通常是存在的：

+ *以目标为导向*——代理必须能够接收一个它可以最终实现的目标。目标本身可以是具体的，例如“为我预订 X 酒店的房间”，或者更抽象的，例如“根据当前趋势确定下个月最成功的股票”。无论目标范围如何，都需要某种方向来帮助我们评估代理采取的行动是否使其更接近完成任务。

+ *具有感知能力*——代理还必须能够与更广泛的世界互动。这可能意味着提取信息，或与系统交互以产生结果。例如，可能包括向 Web API 发出 HTTP 请求以获取用于处理的数据，或运行 Web 自动化代码以完成预订表单。无论我们希望我们的代理实现什么目标，我们都必须赋予它与世界外部互动的能力，以便它能为我们解决问题。

+ *具有自主性*——也许最重要的是，代理必须能够自主决定如何解决问题。它不必遵循我们设定的清晰、算法化的路径，而是可以选择和决定要执行的任务及其顺序。这就是代理的核心要素通常所在的地方。通过评估它被赋予的目标，它可以与世界互动，执行任务，并评估这些任务是否与其设定的目标一致。

+ *具有适应性*——最后，代理还必须能够从行动中学习。例如，能够完成视频游戏的代理是通过从它们犯的错误中学习来做到这一点的。但这也可以仅仅意味着对在特定时间点检索到的特定信息做出反应。与自主性类似，设定的目标在确定代理在执行特定任务时是否成功时发挥作用。如果没有，我们希望我们的代理能够对这种失败做出反应，而不是重复它或绕过它来实现其设定的目标。

这并不是一个详尽的特性列表。根据代理被要求解决的问题将决定哪些特性更重要。然而，提供的例子让我们对 AI 代理可能的样子及其行为有了概念。因此，代理是一块可以执行相对复杂任务并为我们解决问题的自主软件。

### 9.1.2 代理如何与 LLM 协同工作

那么，在使用 LLM（大型语言模型）驱动代理时，这些特征是如何产生的呢？答案是，“通过使用一个名为*函数调用*的 LLM 功能，”这使我们能够将代码封装成函数，当 LLM 被提示完成一个任务时，它可以为我们触发这些函数。为了帮助我们更好地理解这个过程，图 9.1 概述了函数调用的一般工作流程。

![图片](img/CH09_F01_Winteringham2.png)

图 9.1 使用函数调用的 LLM 代理概述

如图中所述，函数调用通过向 LLM 提供提示和封装的代码来实现。然后 LLM 可以确定运行哪个函数以达到我们在提示中设定的目标。在每一个函数内部，都会有我们构建的代码，这些代码将以某种形式处理信息或与外界进行交互。例如，一个函数可能从 Web API 中提取数据，抓取网页，或从传感器收集信息。一个函数也可能从 LLM 接收参数，在将数据发送回 LLM 用于未来使用之前对其进行进一步处理。

使我们的代理具有自主特性的，是 LLM 确定何时以及使用什么数据调用哪些函数的能力。我们创建的每个函数都提供了一些额外的指令，以帮助 LLM 确定函数内部我们的代码做了什么。当被分配一个任务去完成时，LLM 可以处理初始提示，选择首先调用的正确函数，并将函数返回的任何信息存储到提示本身中。

这种自主性是区分代理与执行一系列函数的工具的关键，这些函数与不同的 API 或服务交互。代理可能被赋予大量各种函数，这些函数以不同的方式处理信息和与世界交互，仅使用必要的函数来解决问题。

## 9.2 创建 AI 测试助手

既然我们已经了解了代理是什么以及它们在生成式 AI 环境中的工作方式，让我们自己构建一个。在这个例子中，我们将构建一个 AI 代理，当给出指令时，它会为我们读取和创建数据。这可能看起来是一个相对简单的代理，但要深入了解代理的高级主题，可能需要另一整本书。然而，通过这个例子，我们将更好地理解如何使用 LLMs 构建代理，以便我们可以将我们的提示工程和代理构建提升到下一个层次。我们将逐步创建我们的 AI 代理，但您可以在以下链接查看我们的完成版本以供参考：[`mng.bz/aVlJ`](https://mng.bz/aVlJ)。

### 9.2.1 设置我们的虚拟 AI 代理

我们首先创建一个 Maven 项目，并在我们的`pom.xml`文件中添加以下依赖项：

```xml
<dependencies>

    <dependency>                                                    ❶
        <groupId>dev.langchain4j</groupId>                          ❶
        <artifactId>langchain4j-open-ai</artifactId>                ❶
        <version>0.31.0</version>                                   ❶
    </dependency>                                                   ❶

    <dependency>                                                    ❷
        <groupId>dev.langchain4j</groupId>                          ❷
        <artifactId>langchain4j</artifactId>                        ❷
        <version>0.31.0</version>                                   ❷
    </dependency>                                                   ❷

    <dependency>                                                    ❸
        <groupId>com.h2database</groupId>                           ❸
        <artifactId>h2</artifactId>                                 ❸
        <version>2.2.224</version>                                  ❸
    </dependency>                                                   ❸
</dependencies>                                                     ❸
```

❶ 快速连接到 OpenAI 平台的函数

❷ 创建我们的 AI 代理的 AI 服务

❸ 数据库用于与我们的代理交互

我们将使用 LangChain4J（[`docs.langchain4j.dev`](https://docs.langchain4j.dev)）来管理与我们 LLM 的通信以及我们希望代理执行的必要函数。当实现这些函数时，这会变得更加清晰，但首先，让我们通过创建一个名为`DataAssistant`的新类并添加一个`main`函数来建立我们与 LLM 的连接：

```java
public class DataAssistant {

    public static void main(String[] args) {

    }

}
```

现在我们可以更新类，添加必要的代码来向 gpt-3.5-turbo 模型发送一个基本提示：

```java
public class DataAssistant {

    static interface DataAssistantService {                          ❶
        String sendPrompt(String userPrompt);                        ❶
    }                                                                ❶

    public static void main(String[] args) {                         ❶

        OpenAiChatModel model = OpenAiChatModel                      ❷
                .builder()                                           ❷
                .apiKey("API-KEY-HERE")                              ❷
                .modelName(OpenAiChatModelName.GPT_3_5_TURBO)        ❷
                .build();                                            ❷

        DataAssistantService dataAssistantChat =                     ❸
   ➥ AiServices.builder(DataAssistantService.class)                 ❸
                .chatLanguageModel(model)                            ❸
                .build();                                            ❸

        String response = dataAssistantChat.sendPrompt("Hi, can you  ❹
        ➥introduce yourself?");                                     ❹
        System.out.printf(response);                                 ❹
    }                                                                ❹

}                                                                    ❹
```

❶ 创建一个接口，将我们的 AI 服务添加进去

❷ 设置访问我们的 OpenAI 模型和模型偏好

❸ 使用`AiServices.builder`将我们的模型添加到`DataAssistantService`中

❹ 向 OpenAI 发送基本提示，存储响应并输出

代码中的`OpenAiChatModel`部分确定了我们想要连接的模型以及我们的授权方法。在`.apiKey`方法中，我们提供了一个来自 OpenAI 的 API 密钥，该密钥可以通过他们位于[`platform.openai.com/api-keys`](https://platform.openai.com/api-keys)的 API 密钥页面生成。然后，在设置`DataAssistantService`时，我们使用`AiServices`库将模型作为参数提供。这使得我们可以将模型选择和 AI 服务分开，从而可以轻松地更改我们想要使用的模型。`DataAssistantService`接口有助于配置我们想要用来发送提示的方法，以及如果我们希望的话，添加其他高级功能，例如系统提示，这些提示将上下文化我们想要发送的用户提示，一旦我们的服务建立。我们很快就会看到`AiServices`是如何发挥其作用的，但到目前为止，我们可以通过运行它并获取以下类似的响应来测试我们实现的代码：

| ![](img/logo-openai.png) | 你好，我是一个由 OpenAI 创建的语言模型 AI 助手。我在这里帮助你回答问题，并协助你获取可能需要的任何信息。今天我能帮您什么忙？ |
| --- |

现在我们已经连接到我们的 LLM，我们可以开始构建我们想要我们的代理触发的任务，以给我们的助手一些自主权。为此，我们创建了一个名为`DataAssistantTools`的新类，并添加了以下代码：

```java
public class DataAssistantTools {

    @Tool("Create room records")
    public void createRooms(@P("Amount of room records to create")
    ➥int count) {
        System.out.println("You want me to create " + count + "
        ➥rooms.");
    }

    @Tool("Create booking records")
    public void createBookings(@P("Amount of booking records to
    ➥create") int count) {
        System.out.println("You want me to create " + count + "
        ➥bookings.");
    }

    @Tool("Show results of database")
    public void displayDatabase() {
        System.out.println("I'll then share current database
        ➥details");
    }

}
```

我们现在已经创建了三个函数，我们的 LLM 可以在提示时选择触发。但是 LLM 是如何确定在给定时间触发哪个函数的呢？这是通过使用`LangChain4J`提供的`@Tool`注解来实现的。`@Tool`注解不仅标识了要触发的函数，而且还向 LLM 用自然语言说明了代码的作用，这样 LLM 就可以确定它是否是一个值得调用的函数。例如，我们的第一个工具具有`@Tool("Create room records")`的注解。如果我们向我们的 LLM 发送一个提示，包括我们的工具，要求它创建一些房间，那么 LLM 将确定我们的工具应该被执行。如果我们发送一个完全不同的消息的提示，那么工具可能不会被使用。我们很快就会看到这一点，但首先，让我们更新我们的`AiServices`构建器，使其包含我们新创建的`DataAssistantTools`：

```java
DataAssistantService dataAssistantChat =
➥AiServices.builder(DataAssistantService.class)
                .chatLanguageModel(model)

                .tools(new DataAssistantTools())                      ❶
                .build();                                             ❶

        while(true){                                                  ❷
            Scanner scanner = new Scanner(System.in);                 ❷
            System.out.println("What do you need?");                  ❷

            String query = scanner.nextLine();                        ❷
            String response = dataAssistantChat.sendPrompt(query);    ❷
            System.out.println(response);                             ❷
        }                                                             ❷
```

❶ 通过构建器中的`tools()`方法添加我们的工具

❷ 设置一个扫描器以保持应用运行并接收提示

