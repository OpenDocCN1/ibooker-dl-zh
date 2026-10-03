
示例 3-1 是一个良好的起点；你仍然可以在你的 CPU 上加载此模型并在合理的时间内获得响应。然而，`TinyLlama` 可能也不会像其更大的版本表现得那么好。对于生产工作负载，你将希望使用更大的模型以获得更好的输出质量和性能。

你现在可以在控制器函数内使用 `load_model` 和 `predict` 函数^(10)，然后添加一个路由处理装饰器，通过端点提供模型，如 示例 3-2 所示。

### 示例 3-2\. 通过 FastAPI 端点提供语言模型

```py
# main.py

from fastapi import FastAPI
from models import load_text_model, generate_text

app = FastAPI()

@app.get("/generate/text") ![1](img/1.png)
def serve_language_model_controller(prompt: str) -> str: ![2](img/2.png)
    pipe = load_text_model() ![3](img/3.png)
    output = generate_text(pipe, prompt) ![4](img/4.png)
    return output ![5](img/5.png)
```

![1](img/#co_ai_integration_and_model_serving_CO2-1)

创建一个 `FastAPI` 服务器并添加一个 `/generate` 路由处理程序来提供模型服务。

![2](img/#co_ai_integration_and_model_serving_CO2-2)

`serve_language_model_controller` 负责从请求查询参数中获取提示。

![3](img/#co_ai_integration_and_model_serving_CO2-3)

模型已加载到内存中。

![4](img/#co_ai_integration_and_model_serving_CO2-4)

控制器将查询传递给模型以执行预测。

![5](img/#co_ai_integration_and_model_serving_CO2-5)

`FastAPI` 服务器将输出作为 HTTP 响应发送给客户端。

一旦 `FastAPI` 服务启动并运行，你可以访问位于 `http://localhost:8000/docs` 的 Swagger 文档页面来测试你的新端点：

```py
http://localhost:8000/generate/text?prompt="What is FastAPI?"
```

如果你在一个 CPU 上运行代码示例，它将花费大约一分钟的时间从模型那里收到响应，如 图 3-11 所示。

![bgai 0311](img/bgai_0311.png)

#### 图 3-11\. TinyLlama 的响应

对于在您自己的计算机上运行的 CPU 上的小型语言模型（SLM）来说，这不是一个糟糕的响应，除了 `TinyLlama` 幻觉了 `FastAPI` 使用 `Flask`。这是一个错误的陈述；`FastAPI` 使用 `Starlette` 作为底层 Web 框架，而不是 `Flask`。

*幻觉* 指的是不基于训练数据或现实的输出。尽管开源 SLMs（如 `TinyLlama`）已经在令人印象深刻的数量（3 万亿）的标记上进行了训练，但一小部分模型参数可能限制了它们在数据中学习真实情况的能力。此外，一些未经筛选的训练数据也可能被使用，这两者都可能导致更多幻觉实例的发生。

#### 警告

在提供语言模型时，始终让您的用户知道使用外部来源对输出进行事实核查，因为语言模型可能会 *产生幻觉* 并产生不正确的陈述。

你现在可以使用 Python 的 Web 浏览器客户端以比使用命令行客户端更多的交互性来测试你的服务。

一个优秀的 Python 包 [Streamlit](https://oreil.ly/9BXmn)，可以让你轻松快速地开发用户界面，几乎不需要任何努力就能为你的 AI 服务创建美观且可定制的 UI。

### 将 FastAPI 与 Streamlit UI 生成器连接

`Streamlit` 允许你轻松创建用于测试和原型设计的聊天用户界面。你可以使用 `pip` 安装 `streamlit` 包：

```py
$ pip install streamlit
```

示例 3-3 展示了如何开发一个简单的 UI 来连接你的服务。

#### 示例 3-3\. Streamlit 聊天 UI 消费 FastAPI 的 `/generate` 端点

```py
# client.py

import requests
import streamlit as st

st.title("FastAPI ChatBot") ![1](img/1.png)

if "messages" not in st.session_state:
    st.session_state.messages = [] ![2](img/2.png)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) ![3](img/3.png)

if prompt := st.chat_input("Write your prompt in this input field"): ![4](img/4.png)
    st.session_state.messages.append({"role": "user", "content": prompt}) ![5](img/5.png)

    with st.chat_message("user"):
        st.text(prompt) ![6](img/6.png)

    response = requests.get(
        f"http://localhost:8000/generate/text", params={"prompt": prompt}
    ) ![7](img/7.png)
    response.raise_for_status() ![8](img/8.png)

    with st.chat_message("assistant"):
        st.markdown(response.text) ![9](img/9.png)
```

![1](img/1.png)(#co_ai_integration_and_model_serving_CO3-1)

为你的应用程序添加一个标题，该标题将被渲染到 UI 中。

![2](img/2.png)(#co_ai_integration_and_model_serving_CO3-2)

初始化聊天并跟踪聊天历史。

![3](img/3.png)(#co_ai_integration_and_model_serving_CO3-3)

在应用重运行时显示聊天历史中的聊天消息。

![4](img/4.png)(#co_ai_integration_and_model_serving_CO3-4)

等待用户通过聊天输入字段提交提示。

![5](img/5.png)(#co_ai_integration_and_model_serving_CO3-5)

将用户或助手的消息添加到聊天历史中。

![6](img/6.png)(#co_ai_integration_and_model_serving_CO3-6)

在聊天消息容器中显示用户消息。

![7](img/7.png)(#co_ai_integration_and_model_serving_CO3-7)

向你的 `FastAPI` 端点发送带有提示作为查询参数的 `GET` 请求以从 `TinyLlama` 生成响应。

![8](img/8.png)(#co_ai_integration_and_model_serving_CO3-8)

验证响应是否正常。

![9](img/9.png)(#co_ai_integration_and_model_serving_CO3-9)

在聊天消息容器中显示助手消息。

你现在可以启动你的 `Streamlit` 客户端应用程序:^(11)

```py
$ streamlit run client.py
```

你现在应该能够像 图 3-12 中所示的那样在 `Streamlit` 中与 `TinyLlama` 交互。所有这些都可以通过几个简短的 Python 脚本实现。

![bgai 0312](img/bgai_0312.png)

#### 图 3-12\. Streamlit 客户端

图 3-13 展示了我们迄今为止开发的解决方案的整体系统架构。

![bgai 0313](img/bgai_0313.png)

#### 图 3-13\. FastAPI 服务系统架构

#### 警告

虽然 示例 3-3 中的解决方案非常适合原型设计和测试模型，但它不适合生产工作负载，因为在这种情况下，多个用户需要同时访问模型。这是因为，在当前设置中，每次处理请求时，模型都会被加载到内存中并卸载。加载/卸载大型模型到内存中既慢又阻塞 I/O。

你刚刚构建的 `TinyLlama` 服务使用了一个 *解码器* 变换器，该变换器针对对话和聊天用例进行了优化。然而，[关于变换器的原始论文](https://oreil.ly/RqztC) 介绍了一个由编码器和解码器组成的架构。

你现在应该对语言模型的内部工作原理以及如何在 `FastAPI` 网络服务器中打包它们更有信心。

语言模型只是所有生成模型中的一小部分。接下来的几节将扩展你的知识，包括生成音频、图像和视频的模型的功能和用途。

我们可以先从音频模型开始工作。

## 音频模型

在 GenAI 服务中，音频模型对于创建交互性和逼真的声音非常重要。与你现在熟悉的、专注于处理和生成文本的文本模型不同，音频模型可以处理音频信号。有了它们，你可以合成语音、生成音乐，甚至为虚拟助手、自动配音、游戏开发和沉浸式音频环境等应用创建音效。

`Suno AI` 创建的 `Bark` 模型是功能最强大的文本到语音和文本到音频模型之一。这个基于变换器的模型可以生成逼真的多语言语音和音频，包括音乐、背景噪音和音效。

`Bark` 模型由四个模型串联作为管道，从文本提示中合成音频波形，如图 图 3-15 所示。

![bgai 0315](img/bgai_0315.png)

#### 图 3-15\. Bark 合成管道

1. 语义文本模型

一个因果（顺序）自回归变换器模型接受标记化的输入文本，并通过语义标记捕获意义。自回归模型通过重复使用自己的先前输出来预测序列中的未来值。

2. 粗略声学模型

一个因果自回归变换器接收语义模型的输出并生成初始音频特征，这些特征缺乏更精细的细节。每个预测都基于语义标记序列中的过去和现在信息。

3. 精细声学模型

一个非因果自动编码器变换器通过生成剩余的音频特征来精炼音频表示。由于粗略声学模型已经生成了整个音频序列，精细模型不需要是因果的。

4. Encodec 音频编解码器模型

模型解码来自所有先前生成的音频代码的输出音频数组。

`Bark` 通过将精炼的音频特征解码成最终音频输出（以语音、音乐或简单的音频效果的形式）来合成音频波形。

示例 3-4 展示了如何使用小的 `Bark` 模型。

#### 示例 3-4\. 从 Hugging Face 仓库下载并加载小的 Bark 模型

```py
# schemas.py

from typing import Literal

VoicePresets = Literal["v2/en_speaker_1", "v2/en_speaker_9"] ![1](img/1.png)

# models.py
import torch
import numpy as np
from transformers import AutoProcessor, AutoModel, BarkProcessor, BarkModel
from schemas import VoicePresets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_audio_model() -> tuple[BarkProcessor, BarkModel]:
    processor = AutoProcessor.from_pretrained("suno/bark-small", device=device) ![2](img/2.png)
    model = AutoModel.from_pretrained("suno/bark-small", device=device) ![3](img/3.png)
    return processor, model

def generate_audio(
    processor: BarkProcessor,
    model: BarkModel,
    prompt: str,
    preset: VoicePresets,
) -> tuple[np.array, int]:
    inputs = processor(text=[prompt], return_tensors="pt",voice_preset=preset) ![4](img/4.png)
    output = model.generate(**inputs, do_sample=True).cpu().numpy().squeeze() ![5](img/5.png)
    sample_rate = model.generation_config.sample_rate ![6](img/6.png)
    return output, sample_rate
```

![1](img/1.png)[#co_ai_integration_and_model_serving_CO4-1]

使用 `Literal` 类型指定支持的语音预设选项。

![2](img/2.png)[#co_ai_integration_and_model_serving_CO4-2]

下载小型的 `Bark` 处理器，该处理器为核心模型准备输入文本提示。

![3](img/3.png)(#co_ai_integration_and_model_serving_CO4-3)

下载 `Bark` 模型，该模型将用于生成输出音频。这两个对象在后续的音频生成中都将被需要。

![4](img/4.png)(#co_ai_integration_and_model_serving_CO4-4)

使用说话人语音预设嵌入对文本提示进行预处理，并使用`return_tensors="pt"`返回一个标记化输入的 Pytorch 张量数组。

![5](img/5.png)(#co_ai_integration_and_model_serving_CO4-5)

生成包含合成音频信号随时间变化的振幅值的音频数组。

![6](img/6.png)(#co_ai_integration_and_model_serving_CO4-6)

从生成音频的模型配置中获取采样率，该采样率可以用于生成音频。

当您使用模型生成音频时，输出是一个表示音频信号在每个时间点上的*振幅*（或强度）的浮点数序列。

要播放此音频，需要将其转换为可以发送到扬声器的数字格式。这涉及到以固定速率采样音频信号并将振幅值量化为固定数量的位。`soundfile`库可以通过使用*采样率*生成音频文件来帮助您。采样率越高，采样的样本越多，这提高了音频质量，但也增加了文件大小。

您可以使用`pip`安装`soundfile`音频库来使用`pip`写入音频文件：

```py
$ pip install soundfile
```

示例 3-5 展示了如何将音频内容流式传输到客户端。

#### 示例 3-5\. 返回生成音频的 FastAPI 端点
