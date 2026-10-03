
# 从 JSON 文件 `OAI_CONFIG_LIST` 加载您的 LLM 配置

此代理直接与 LLM（大型语言模型）进行交流。

此代理代理用户与助手之间的对话。

设置终止消息允许代理迭代。

通过 `user_proxy` 与助手发起聊天以完成任务。

通过在 VS Code 的调试器中运行文件（F5）来运行代码。列表 4.6 中的代码使用一个简单的任务来演示代码编写。列表 4.7 展示了几个可供选择示例。这些编码任务也是作者评估 LLM 在编码能力上的常规基准之一。

## 列表 4.7 简单编码任务示例

```py
write a Python function to check if a number is prime
code a classic sname game using Pygame                   #1
code a classic asteroids game in Python using Pygame  #1
```

#1 要享受对这些任务进行迭代的乐趣，请在 Windows 上使用 Windows Subsystem for Linux (WSL)，或使用 Docker。

几秒钟后代码开始运行，助手将向代理提供一个解决方案。此时，代理将提示您提供反馈。按 Enter 键，实际上是没有提供反馈，这将提示代理运行代码以验证其是否按预期运行。

令人印象深刻的是，代理甚至可以获取提示来安装所需的包，例如 Pygame。然后它会运行代码，您将在终端或新窗口或浏览器中看到输出。如果代码在新的窗口/浏览器中打开，您可以玩游戏或使用界面。

注意，在 Windows 上，生成的窗口/浏览器不会关闭，并且需要退出整个程序。为了避免这个问题，请通过 Windows Subsystem for Linux (WSL)或 Docker 运行代码。AutoGen 明确建议使用 Docker 作为代码执行代理，如果您熟悉容器，这是一个不错的选择。

无论哪种方式，在代理生成和运行代码后，列表 4.6 中之前设置的 `working_dir` 文件夹现在应该有一个包含代码的 Python 文件。这将允许你随意运行代码，进行更改，甚至要求改进，正如我们将看到的。在下节中，我们将探讨如何提高编码代理的能力。

### 使用代理评论家增强代码输出

多代理系统的一个强大优势是，在完成任务时可以自动分配多个角色/角色。生成或帮助编写代码可以是任何开发者的一个极好的优势，但如果该代码也被审查和测试呢？在下一个练习中，我们将向我们的代理系统添加另一个代理评论家来帮助编码任务。打开 `autogen_coding_critic.py`，如下列所示。

#### 列表 4.8 `autogen_coding_critic.py`

```py
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")

user_proxy = UserProxyAgent(
    "user",
    code_execution_config={
        "work_dir": "working",
        "use_docker": False,
        "last_n_messages": 1,
    },
    human_input_mode="ALWAYS",
    is_termination_msg=lambda x: 
x.get("content", "").rstrip().endswith("TERMINATE"),
)

engineer = AssistantAgent(
    name="Engineer",
    llm_config={"config_list": config_list},
    system_message="""
    You are a profession Python engineer, known for your expertise in 
software development.
    You use your skills to create software applications, tools, and 
games that are both functional and efficient.
    Your preference is to write clean, well-structured code that is easy 
to read and maintain.    
    """,      #1
)

critic = AssistantAgent(
    name="Reviewer",
    llm_config={"config_list": config_list},
    system_message="""
    You are a code reviewer, known for your thoroughness and commitment 
to standards.
    Your task is to scrutinize code content for any harmful or 
substandard elements.
    You ensure that the code is secure, efficient, and adheres to best 
practices.
    You will identify any issues or areas for improvement in the code 
and output them as a list.
    """,      #2
)

def review_code(recipient, messages, sender, config):      #3
    return f"""
            Review and critque the following code.

            {recipient.chat_messages_for_summary(sender)[-1]['content']}
            """                       #3                    

user_proxy.register_nested_chats(      #4
    [
        {
            "recipient": critic,
            "message": review_code,
            "summary_method": "last_msg",
            "max_turns": 1,
        }
    ],
    trigger=engineer,                 #4
)
task = """Write a snake game using Pygame."""

res = user_proxy.initiate_chat(
    recipient=engineer, 
    message=task, 
    max_turns=2, 
    summary_method="last_msg"      #5
)
```

#1 这次，助手被给予一个系统/角色消息。

#2 创建了一个具有背景的第二个助手评论家代理。

#3 一个自定义函数帮助提取评论家审查的代码。

#4 评论家和工程师之间创建了一个嵌套聊天。

#5 代理代理以最大延迟和明确的摘要方法启动聊天。

在 VS Code 中以调试模式运行 `autogen_coding_critic.py` 文件，并观察代理之间的对话。这次，在代码返回后，评论家也会被触发进行响应。然后，评论家会添加注释和建议来改进代码。

嵌套聊天对于支持和控制代理交互效果很好，但我们在下节中会看到一个更好的方法。在此之前，我们将在下一节中回顾 AutoGen 缓存的重要性。

### 理解 AutoGen 缓存

作为可对话的多代理平台，AutoGen 在聊天迭代中可能会消耗许多令牌。如果你要求 AutoGen 通过复杂或新颖的问题，你甚至可能会在 LLM 上遇到令牌限制；因此，AutoGen 支持几种减少令牌使用的方法。

AutoGen 使用缓存来存储进度并减少令牌使用。缓存默认启用，你可能已经遇到过。如果你检查当前工作文件夹，你会注意到一个 `.cache` 文件夹，如图 4.8 所示。缓存允许代理在被打断后继续对话。

![figure](img/4-8.png)

#### 图 4.8 AutoGen 缓存和工作文件夹

在代码中，你可以控制代理运行的缓存文件夹，如图 4.9 所示。通过将 `initiate_chat` 调用用 `with` 语句包装，你可以控制缓存的位置和种子。这将允许你通过仅设置前一个缓存的 `cache_seed` 来保存并返回未来的长时间运行的 AutoGen 任务。

#### 列表 4.9 设置缓存文件夹

```py
with Cache.disk(cache_seed=42) as cache:     #1
    res = user_proxy.initiate_chat(
        recipient=engineer,
        message=task,
        max_turns=2,
        summary_method="last_msg",
        cache=cache,      #2
    )
```

#1 设置 `seed_cache` 表示单个位置。

#2 将缓存作为参数设置

这种缓存能力允许你从之前的缓存位置继续操作并捕获之前的运行。这也可以是一种展示和检查代理对话如何生成结果的好方法。在下一节中，我们将探讨另一种 AutoGen 支持的群聊对话模式。

## 与代理和 AutoGen 的群聊

聊天委托和嵌套聊天或对话的一个问题是信息的传达。如果你曾经玩过电话游戏，你亲自见证了这一点，并体验到了信息在迭代中如何迅速变化。对于代理来说，这当然也不例外，通过嵌套或顺序对话聊天可以改变任务甚至期望的结果。

##### 电话游戏

电话游戏是一种既有趣又有教育意义的游戏，它展示了信息和连贯性的丧失。孩子们排成一排，第一个孩子只能听到一个信息。然后，孩子们轮流将信息口头传递给下一个孩子，以此类推。最后，最后一个孩子将信息宣布给整个小组，通常这个信息与最初的信息相差甚远。

为了应对这一点，AutoGen 提供了一个群聊机制，通过这个机制代理可以参与共享对话。这允许代理回顾所有过去的对话，并在长期和复杂任务上更好地协作。

图 4.9 显示了嵌套群聊和协作群聊之间的差异。我们在上一节中使用了嵌套聊天功能来构建嵌套代理聊天。在本节中，我们使用群聊来提供更协作的体验。

![图](img/4-9.png)

#### 图 4.9 可对话代理的嵌套群聊与群聊之间的差异

打开`autogen_coding_group.py`，查看列表 4.10 中的相关部分。代码与之前的练习类似，但现在引入了`GroupChat`和`GroupChatManager`。代理和消息都存储在群聊中，类似于 Slack 或 Discord 等应用中的消息频道。聊天管理器协调消息响应以减少对话重叠。

#### 列表 4.10 `autoget_coding_group.py`（相关部分）

```py
user_proxy = UserProxyAgent(
    "user",
    code_execution_config={
        "work_dir": "working",
        "use_docker": False,
        "last_n_messages": 3,
    },
    human_input_mode="NEVER",     #1
)

llm_config = {"config_list": config_list}

engineer = AssistantAgent(…      #2

critic = AssistantAgent(…       #2

groupchat = GroupChat(agents=[user_proxy, 
                              engineer, 
                              critic], 
                              messages=[], 
                              max_round=20)      #3
manager = GroupChatManager(groupchat=groupchat, 
                           llm_config=llm_config)     #4

task = """Write a snake game using Pygame."""

with Cache.disk(cache_seed=43) as cache:
    res = user_proxy.initiate_chat(
        recipient=manager,
        message=task,
        cache=cache,
    )
```

#1 人类输入现在设置为永不，因此没有人类反馈。

#2 代码省略，但请参考文件中角色的更改

#3 此对象持有所有代理的连接并存储消息。

#4 管理员作为调解者一样协调对话。

运行这个练习，你会看到代理是如何协作的。工程师现在将从评论者那里获取反馈并执行操作以解决评论者的建议。这也允许代理参与所有对话。

群组对话是加强代理协作能力的好方法，因为它们在完成任务时进行协作。然而，它们也更为冗长且代价高昂。当然，随着大型语言模型（LLMs）的成熟，它们的上下文标记窗口的大小和标记处理的价格也在增加。随着标记窗口的增加，对标记消耗的担忧可能会最终消失。

AutoGen 是一个强大的多代理平台，可以通过网页界面或代码体验。无论你的偏好如何，这个代理协作工具是构建代码或其他复杂任务的优秀平台。当然，它并不是唯一的平台，正如你将在下一节中看到的，我们将探索一个新来者，即 CrewAI。

## 使用 CrewAI 构建代理团队

CrewAI 在多代理系统领域相对较新。AutoGen 最初是从研究中发展起来的，然后进行了扩展，而 CrewAI 是以企业系统为前提构建的。因此，该平台在某种程度上更健壮，但在某些方面不太容易扩展。

使用 CrewAI，你可以构建一个专注于特定任务目标的代理团队。与 AutoGen 不同，CrewAI 不需要使用用户代理代理，而是假设代理只在自己之间工作。

图 4.10 展示了 CrewAI 平台的主要元素，它们如何连接在一起以及它们的主要功能。它展示了一个具有通用研究人员和作家代理的顺序处理代理系统。代理被分配的任务可能包括工具或记忆来协助它们。

![figure](img/4-10.png)

#### 图 4.10 CrewAI 系统的组成

CrewAI 支持两种主要的处理形式：顺序和分层。图 4.10 通过迭代给定的代理及其相关任务来展示顺序处理过程。在下一节中，我们将深入研究一些代码来设置团队并使用它来完成目标并创造一个有趣的笑话。

### 创建一个由 CrewAI 代理组成的玩笑者团队

与 AutoGen 相比，CrewAI 需要更多的设置，但这也允许更多的控制和额外的指南，这些指南提供了更具体的上下文来指导代理完成给定的任务。这并非没有问题，但它确实提供了比 AutoGen 更多的控制。

在 VS Code 中打开 `crewai_introduction.py` 并查看列表 4.11 所示的顶部部分。配置代理需要许多设置，包括角色、目标、冗长性、记忆、前传、委派，甚至工具（未显示）。在这个例子中，我们使用了两个代理：一个高级玩笑研究人员和一个玩笑作家。

#### 列表 4.11 `crewai_introduction.py`（代理部分）
