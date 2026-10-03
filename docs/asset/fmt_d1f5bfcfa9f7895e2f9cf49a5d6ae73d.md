
```py
import streamlit as st
import requests
import json

url = "http://localhost:8000/generate"    #1

st.title("Chatbot with History")

if "chat_history" not in st.session_state:      #2
    st.session_state.chat_history = []

for chat in st.session_state.chat_history:      #3
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

if user_input := st.chat_input("Your question here"):     #4
    with st.chat_message("user"):                   #5
        st.markdown(user_input)

    st.session_state.chat_history.append(          #6
        {"role": "user", "content": user_input}
    )

    with st.chat_message("assistant"):     #7
        placeholder = st.empty()
        full_response = ""

        prompt = "You are an assistant who helps the user. "     #8
        "Answer their questions as accurately as possible. Be concise. "
        history = [
            f'{ch["role"]}: {ch["content"]}'
            for ch in st.session_state.chat_history
        ]
        prompt += " ".join(history)
        prompt += " assistant: "
        data = json.dumps({"prompt": prompt})

        with requests.post(url, data=data, stream=True) as r:     #9
            for line in r.iter_lines(decode_unicode=True):
                full_response += line.decode("utf-8")
                placeholder.markdown(full_response + "▌")     #10
        placeholder.markdown(full_response)

    st.session_state.chat_history.append(      #11
        {"role": "assistant", "content": full_response}
    )
```

#1 指向你的模型 API

#2 在会话状态中创建聊天历史

#3 Δ显示历史聊天

#4 对用户做出响应。注意：我们使用 walrus 操作符（`:=`）在同时确保用户输入不为 `None` 的情况下分配用户输入。

#5 Δ显示用户输入

#6 将用户输入添加到聊天历史

#7 流响应

#8 格式化提示，添加聊天历史以提供额外上下文

#9 发送请求

#10 添加闪烁的光标以模拟输入

#11 将 LLM 响应添加到聊天历史

图 8.4 是捕获我们刚刚构建的 LLM 应用的屏幕截图。虽然我们的第一个例子相当丑陋，但你可以看到 Streamlit 自动创建了一个很好的用户界面，包括完成细节，如用户的人脸图片和我们的 LLM 助手的机器人脸。你也会注意到模型正在接收并理解对话历史——尽管给出的响应很糟糕。如果我们想要得到更好的响应，有一件事要确保的是，你的 LLM 已经在对话数据上进行了训练。

![图](img/8-4.png)

##### 图 8.4 我们的 Streamlit 应用利用聊天历史的屏幕截图

当然，利用历史记录会导致一些问题。第一个问题是用户可以与我们的机器人进行相对较长的对话，但我们仍然受限于可以输入到模型中的令牌长度，输入越长，生成所需的时间就越长。在某个时候，历史记录会变得过长。解决这个问题的最简单方法是用较新的消息替换较旧的消息。当然，我们的模型可能会在对话开始时忘记重要的细节或指令，但人类在对话中也倾向于有近期偏差，所以这通常是可以接受的——当然，除了人类倾向于期望计算机永远不会忘记任何事情这一点。

一个更稳健的解决方案是使用 LLM 来总结聊天历史，并将总结作为我们用户查询的上下文，而不是使用完整的聊天历史。LLM 通常非常擅长从大量文本中突出显示重要的信息片段，因此这可以是一种有效地压缩对话的方法。压缩可以按需进行，也可以作为后台进程运行。图 8.5 展示了聊天历史压缩的总结工作流程。

![figure](img/8-5.png)

##### 图 8.5 使用总结进行聊天历史压缩的应用程序流程图

你还可以探索其他策略，以及混合和匹配多种方法。另一个想法是将每个聊天嵌入，并执行搜索以查找相关的先前聊天消息以添加到提示上下文中。但无论你选择如何缩短聊天历史，随着对话的持续进行或提示和响应的增大，细节很可能会丢失或被遗忘。

### 8.1.3 聊天机器人交互功能

与 LLM 聊天机器人聊天并不像与你的朋友聊天。一方面，聊天机器人总是随时待命，等待我们与之交谈，因此我们可以期待立即得到回应。用户在收到反馈之前不应该有机会向我们的机器人发送多条消息进行垃圾邮件式攻击。但让我们面对现实，在现实世界中，可能会有连接问题或网络状况不佳，服务器可能会过载，以及许多可能导致请求失败的其他原因。这些差异促使我们以不同的方式与聊天机器人互动，并且我们应该确保为我们的用户提供几个功能来改善他们的体验。现在让我们考虑其中的一些：

+   *回退响应*——当发生错误时提供的响应。为了保持整洁，你将希望确保每个用户查询在聊天历史中的 LLM 响应比例为 1:1。回退响应确保我们的聊天历史整洁，并为用户提供最佳行动方案的说明，例如过几分钟再尝试。说到这一点，你还应该考虑在收到响应时禁用提交按钮，以防止异步对话和聊天历史顺序混乱导致的奇怪问题。

+   *停止按钮* — 在响应过程中中断。大型语言模型（LLM）往往很健谈，在回答用户问题后还会继续回应。它经常误解问题并开始错误地回答。在这些情况下，最好给用户提供一个停止按钮，以便他们可以中断模型并继续操作。这个按钮是一个简单的成本节约功能，因为我们通常以某种方式按输出令牌付费。

+   *重试按钮* — 重新发送最后一个查询并替换响应。LLM 有一定的随机性，这对于创意写作来说可能很好，但这也意味着它们可能对之前已经正确响应多次的提示做出不利的回应。由于我们将 LLM 聊天历史添加到新的提示中以提供上下文，因此重试按钮允许用户尝试获得更好的结果，并保持对话朝着正确的方向进行。在重试时，调整我们的提示超参数可能是有意义的，例如，每次用户重试时都降低温度。这可以帮助将响应推向用户可能期望的方向。当然，如果他们因为糟糕的网络连接而重试，这可能不是最好的选择，因此您需要仔细考虑调整。

+   *删除按钮* — 删除聊天历史的一部分。如前所述，聊天历史被用作未来响应的上下文，但并非每个响应都能立即被识别为不良。我们经常看到误导性的信息。例如，在编码时使用的聊天助手可能会幻想不存在的功能或方法，这可能导致对话走向难以恢复的路径。当然，根据您的需求，解决方案可能是一个软删除，我们只从前端和提示空间中删除它，但不从后端删除。

+   *反馈表单* — 收集用户体验反馈的一种方式。如果您正在训练或微调自己的 LLM，这些数据非常有价值，因为它可以帮助您的团队在下一个训练迭代中改进结果。这些数据在使用强化学习与人类反馈（RLHF）时通常可以很容易地应用。当然，您不会想直接应用它，但首先需要清理和过滤掉恶意的回复。即使您没有在训练，它也可以帮助您的团队做出切换模型、改进提示和识别边缘情况的决策。

在列表 8.3 中，我们展示了如何使用 Gradio 设置简单的聊天机器人应用程序。Gradio 是一个开源库，用于快速创建数据科学演示和 Web 应用程序的可定制用户界面。它因其易于与 Jupyter 笔记本集成而非常受欢迎，这使得在熟悉的环境中创建界面和编辑 Web 应用程序变得容易。要使用 Gradio 创建聊天机器人，我们将使用`ChatInterface`并给它一个函数来执行我们的 API 请求。您会注意到 Gradio 期望历史记录是`generate`函数的一部分，而流式传输只是确保函数是一个生成器的问题。

##### 什么是 Gradio？

Gradio 是一个开源的 Python 库，允许您快速创建围绕您的机器学习模型的定制 UI 组件。它提供了一个简单的界面来构建您的模型的无 HTML、CSS 或 JavaScript 代码的交互式网络应用程序。使用 Gradio，您可以创建用于模型的输入表单，显示结果，甚至可以通过网络界面与他人共享您的模型。

##### 列表 8.3 本地 LLM Gradio 聊天应用，具有停止、重试和撤销功能

```py
import gradio as gr
import requests
import json

url = "http://localhost:8000/generate"      #1

def generate(message, history):
    history_transformer_format = history + [[message, ""]]
    messages = "".join(
        [
            "".join(["\n<human>:" + h, "\n<bot>:" + b])
            for h, b in history_transformer_format
        ]
    )
    data = json.dumps({"prompt": messages})

    full_response = ""
    with requests.post(url, data=data, stream=True) as r:   #2
        for line in r.iter_lines(decode_unicode=True):
            full_response += line.decode("utf-8")
            yield full_response + "▌"              #3
        yield full_response

gr.ChatInterface(generate, theme="soft").queue().launch()
```

#1 指向您的模型 API

#2 发送请求

#3 添加闪烁的光标以模拟输入

您可以看到这段代码是多么简单，需要的行数非常少。Gradio 为我们做了所有繁重的工作。您可能还在想我们的交互功能在哪里。好消息是，Gradio 会自动为我们添加大部分这些功能。您不相信吗？请查看图 8.6 中我们刚刚创建的应用程序。

![图](img/8-6.png)

##### **图 8.6 我们 Gradio 应用的截图，包括交互功能停止、重试和撤销，以提供更好的使用便捷性**

##### Chainlit：专为 LLM 设计的应用程序构建器

我们已经向您展示了如何使用几种不同的工具构建 LLM 应用程序：Streamlit、Gradio，甚至是纯 HTML 和 JavaScript。有许多优秀的工具，我们无法对每一个都给予个人关注。但我们认为，许多读者可能会对我们接下来要介绍的工具 Chainlit 感兴趣。Chainlit 是一个专门为构建 LLM 应用程序而构建的工具，它自带大多数功能，包括这里未讨论的功能，如主题、CSS 定制、身份验证和云托管。它可能是快速启动和运行的最快方式之一。

您可以为您的应用程序添加的每一项生活品质改进都将帮助它在竞争中脱颖而出，并可能为您节省金钱。出于同样的原因，您应该考虑使用标记计数器，我们将在下一节中介绍。

### 8.1.4 标记计数

您可以收集的最基本但非常有价值的信息之一是提交的标记数。由于 LLM 有标记限制，我们需要确保用户的提示不会超过这些限制。及时且频繁地提供反馈将提供更好的用户体验。没有人愿意输入一个长的查询，然后发现提交后太长了。

计算标记数也使我们能够更好地提示工程和改进结果。例如，在一个问答机器人中，如果用户的提问特别短，我们可以通过扩展我们的检索增强生成（RAG）系统将返回的搜索结果数量来添加更多上下文。如果他们的提问很长，我们则希望限制它，并确保我们有足够的空间来附加我们的上下文。

Tiktoken 就是这样的一个库，用于帮助完成这项任务。它是一个专为 OpenAI 模型构建的极快 BPE 分词器。该包已移植到多种语言，包括用于 Golang 的 `tiktoken-go`、用于 Rust 的 `tiktoken-rs` 以及几个其他语言。在下一节中，我们将展示如何使用它的基本示例。它已经针对速度进行了优化，这使得我们能够快速编码和计数 token，这正是我们所需要的。

##### 列表 8.4 使用 tiktoken 计数 token

```py
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")
print(encoding.encode("You're users chat message goes here."))     
# [2675, 2351, 3932, 6369, 1984, 5900, 1618, 13]
def count_tokens(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))

num_tokens = count_tokens("You're users chat message goes here.")
print(num_tokens)       
# 8
```

当然，那些没有跳过前面的读者会认识到使用 tiktoken 存在一些问题，主要是因为它是针对 OpenAI 的编码器构建的。如果你使用自己的分词器（我们推荐这样做），它将不会非常准确。我们见过一些开发者——出于懒惰或不知道更好的解决方案——仍然将其用于其他模型。通常，当使用 tiktoken 的结果为使用类似 BPE 分词器的其他模型计数时，他们看到每 1,000 个 token 内的计数在±5–10 个 token 之间。对他们来说，速度和延迟的增益足以弥补不准确，但这都是口头相传，所以请带着怀疑的态度看待。

如果你使用的是不同类型的分词器，如 SentencePiece，通常最好是创建自己的 token 计数器。例如，我们在第十章的项目中就是这样做的。正如你可以猜到的，代码遵循了相同的模式，即编码字符串和计数 token。难点在于将其移植到需要运行计数器的语言。为此，就像我们第 6.1.1 节讨论的那样，将分词器编译成任何其他 ML 模型。

### 8.1.5 RAG 应用

RAG 是向你的 LLM 添加上下文和外部知识以改进结果准确性的绝佳方式。在上一个章节中，我们讨论了它在后端系统中的应用。在这里，我们将从前端的角度来讨论它。你的 RAG 系统可以在任何一边设置，每边都有其自身的优缺点。

在后端设置 RAG 确保了所有用户都能获得一致的经验，并让我们作为开发者对上下文数据如何使用有更大的控制权。它还为存储在向量数据库中的数据提供了一定的安全性，因为只有通过 LLM 才能被最终用户访问。当然，通过巧妙的提示注入，它仍然可能被爬取，但与简单地允许用户直接查询你的数据相比，它仍然要安全得多。

RAG 更常在前端设置，因为这样做允许开发者将任何通用的 LLM 插入业务上下文。如果你在运行时给模型提供你的数据集，你不需要在数据集上微调模型。因此，RAG 成为了一个为我们的 LLM 应用添加个性和功能，而不仅仅是确保结果准确性和减少幻觉的工具系统。

在 6.1.8 节中，我们向您展示了如何设置一个 RAG 系统；现在我们将向您展示如何利用它进行高效的查询增强。在列表 8.5 中，我们向您展示如何访问和使用我们之前设置的向量存储。我们将继续使用 OpenAI 和 Pinecone，这是我们上一个例子中的内容。我们还将使用 LangChain，这是一个 Python 框架，我们在上一章中发现了它，以帮助创建 LLM 应用。

##### 列表 8.5 前端 RAG
