
| HTTP 请求-响应 | 简单的请求-响应模型，无状态协议，所有网络客户端和服务器都支持 | 实时更新延迟高，频繁服务器到客户端数据传输效率低 | RESTful API，实时更新不是关键点的网络服务 |

| 短/定期轮询 | 客户端定期以间隔请求数据，易于实现 | 当没有新数据时，资源浪费，延迟取决于轮询间隔 | 需要更新不频繁的应用，简单的近实时仪表板，提交作业的状态更新 |

| 长轮询 | 对于实时更新比短轮询更高效，保持开放连接直到有数据可用 | 服务器端可能资源密集，管理多个连接复杂 | 实时通知，较老的聊天应用 |

| 服务器发送事件 | 单个持久连接用于更新，内置重连和事件 ID 支持 | 仅从服务器到客户端的单向通信 | 实时流，聊天应用，实时分析仪表板 |

| WebSocket | 全双工通信，低延迟，支持二进制数据 | 实现和管理更复杂，服务器需要 WebSocket 支持 | 多玩家游戏，聊天应用，协作编辑工具，视频会议和网络研讨会应用，实时转录和翻译应用 |

在详细审查了实时通信机制之后，让我们通过使用这两种机制实现自己的流式端点来更深入地了解 SSE 和 WebSocket。在下一节中，你将学习如何实现与这两种技术一起工作的流式端点。

# 实现 SSE 端点

在第三章中，你学习了关于 LLMs 的内容，它们是*自回归*模型，根据先前输入预测下一个标记。在每个生成步骤之后，输出标记被附加到输入中并通过模型再次传递，直到生成`<stop>`标记以中断循环。你不必等待循环完成，可以将生成的输出标记作为数据流直接转发给用户。

模型提供者通常会提供一个选项，允许你使用`stream=True`将输出模式设置为数据流。设置此选项后，模型提供者可以返回一个数据生成器而不是最终输出给你，你可以直接将其传递到你的 FastAPI 服务器进行流式传输。

要演示这一功能，请参考示例 6-2，它使用`openai`库实现了一个异步数据生成器。

###### 小贴士

要运行示例 6-2，你需要在 Azure 门户上创建 Azure OpenAI 的实例并创建一个模型部署。注意 API 端点、密钥和模型部署名称。对于示例 6-2，你可以使用`2023-05-15` api 版本。

##### 示例 6-2\. 实现用于流式响应的 Azure OpenAI 异步聊天客户端

```py
# stream.py

import asyncio
import os
from typing import AsyncGenerator
from openai import AsyncAzureOpenAI

class AzureOpenAIChatClient: ![1](img/1.png)
    def __init__(self):
        self.aclient = AsyncAzureOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            api_version=os.environ["OPENAI_API_VERSION"],
            azure_endpoint=os.environ["OPENAI_API_ENDPOINT"],
            azure_deployment=os.environ["OPENAI_API_DEPLOYMENT"],
        )

    async def chat_stream(
        self, prompt: str, model: str = "gpt-3.5-turbo"
    ) -> AsyncGenerator[str, None]: ![2](img/2.png)
        stream = await self.aclient.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
            stream=True, ![3](img/3.png)
        )

        async for chunk in stream:
            yield f"data: {chunk.choices[0].delta.content or ''}\n\n" ![4](img/4.png)
            await asyncio.sleep(0.05) ![5](img/5.png)

        yield f"data: [DONE]\n\n"

azure_chat_client = AzureOpenAIChatClient()
```

![1](img/1.png)[#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO2-1]

创建一个异步的`AzureOpenAIChatClient`以与 Azure OpenAI API 进行交互。聊天客户端需要 API 端点、部署名称、密钥和版本才能运行。

![2](img/2.png)[#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO2-2]

定义一个`chat_stream`异步生成器方法，用于从 API `yield`每个输出令牌。

![3](img/3.png)[#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO2-3]

设置`stream=True`以从 API 接收输出流，而不是一次性接收完整响应。

![4](img/4.png)[#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO2-4]

遍历流，`yield`每个输出令牌，如果`delta.content`为空，则返回一个空字符串。每个令牌应带有`data:`子串前缀，以便浏览器可以使用`EventSource` API 正确解析内容。

![5](img/5.png)[#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO2-5]

减慢流速率以减少对客户端的背压。

在示例 6-2 中，你创建了一个`AsyncAzureOpenAI`的实例，这允许你通过你的私有 Azure 环境中的 API 与 Azure OpenAI 模型进行聊天。

通过设置`stream=True`，`AsyncAzureOpenAI`返回一个数据流（一个异步生成器函数），而不是完整的模型响应。你可以遍历数据流，并使用`data:`前缀`yield`令牌，以符合 SSE 规范。这将允许浏览器使用广泛可用的`EventSource`网络 API 自动解析流内容。^([3])

###### 警告

当公开流式端点时，你需要考虑客户端可以多快消费你发送给他们的数据。一个良好的做法是像在示例 6-2 中看到的那样减少流速率，以减少对客户端的背压。你可以通过在不同的设备上使用不同的客户端测试你的服务来调整节流。

## 使用 GET 请求的 SSE

你现在可以通过将聊天流传递给 FastAPI 的`StreamingResponse`作为`GET`端点来实现 SSE 端点，如示例 6-3 所示。

##### 示例 6-3\. 使用 FastAPI 的`StreamingResponse`实现一个 SSE 端点

```py
# main.py

from fastapi.responses import StreamingResponse
from stream import azure_chat_client

...

@app.get("/generate/text/stream") ![1](img/1.png)
async def serve_text_to_text_stream_controller(
    prompt: str,
) -> StreamingResponse:
    return StreamingResponse( ![2](img/2.png)
        azure_chat_client.chat_stream(prompt), media_type="text/event-stream"
    )
```

![1](img/#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO3-1)

使用`GET`方法实现一个 SSE 端点，以便在浏览器中使用`EventSource` API。

![2](img/#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO3-2)

将聊天流生成器传递给`StreamingResponse`，以便将生成的输出流直接转发给客户端。根据 SSE 规范设置`media_type=text/event-stream`，以便浏览器可以正确处理响应。

在服务器上设置了`GET`端点后，你可以在客户端创建一个简单的 HTML 表单，通过`EventSource`接口消费 SSE 流，如示例 6-4 所示。

###### 小贴士

示例 6-4 不使用任何 JavaScript 库或 Web 框架。然而，有一些库可以帮助你在你选择的任何框架（如 React、Vue 或 SvelteKit）中实现`EventSource`连接。

##### 示例 6-4\. 使用浏览器`EventSource` API 在客户端实现 SSE

```py
{# pages/client-sse.html #} <!DOCTYPE html>
<html lang="en">
<head>
    <title>SSE with EventSource API</title>
</head>
<body>
<button id="streambtn">Start Streaming</button>
<label for="messageInput">Enter your prompt:</label>
<input type="text" id="messageInput" placeholder="Enter your prompt"> ![1](img/1.png)
<div style="padding-top: 10px" id="responseContainer"></div> ![2](img/2.png)

<script>
    let source;
    const button = document.getElementById('streambtn');
    const container = document.getElementById('container');
    const input = document.getElementById('messageInput');

    function resetForm(){
        input.value = '';
        container.textContent = '';
    }

    function handleOpen() {
        console.log('Connection was opened');
    }
    function handleMessage(e){
        if (e.data === '[DONE]') {
            source.close();
            console.log('Connection was closed');
            return;
        }

        container.textContent += e.data;
    }
    function handleClose(e){
        console.error(e);
        source.close()
    }

    button.addEventListener('click', function() { ![3](img/3.png)
        const message = input.value;
        const url = 'http://localhost:8000/generate/text/stream?prompt=' +
            encodeURIComponent(message);
        resetForm() ![4](img/4.png)

        source = new EventSource(url); ![5](img/5.png)
        source.addEventListener('open', handleOpen, false);
        source.addEventListener('message', handleMessage, false);
        source.addEventListener('error', handleClose, false); ![6](img/6.png)
    });

</script>
</body>
</html>
```

![1](img/#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO4-1)

创建一个简单的 HTML 输入和按钮以启动 SSE 请求。

![2](img/#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO4-2)

创建一个空容器，用作流内容的接收器。

![3](img/#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO4-3)

监听按钮`clicks`并运行 SSE 回调。

![4](img/#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO4-4)

重置先前内容的内容表单和响应容器。

![5](img/#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO4-5)

创建一个新的`EventSource`对象并监听连接状态变化以处理事件。

![6](img/#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO4-6)

当 SSE 连接打开时记录到控制台。通过将消息内容渲染到响应容器中处理每个消息，直到接收到`[DONE]`消息，这表示连接现在应该关闭。此外，如果发生任何错误，关闭连接并将错误记录到浏览器的控制台。

在示例 6-4 中实现了 SSE 客户端后，你现在可以使用它来测试你的 SSE 端点。然而，你需要首先提供 HTML 服务。

创建一个`pages`目录，然后将 HTML 文件放入其中。然后*挂载*该目录到你的 FastAPI 服务器上，以将其内容作为静态文件提供服务，如示例 6-5 所示。通过挂载，FastAPI 负责将 API 路径映射到每个文件，这样你就可以从与服务器相同的源通过浏览器访问它们。

##### 示例 6-5\. 在服务器上挂载 HTML 文件作为静态资源

```py
# main.py

from fastapi.staticfiles import StaticFiles

app.mount("/pages", StaticFiles(directory="pages"), name="pages") ![1](img/1.png)
```

![1](img/#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO5-1)

将`pages`目录挂载到`/pages`以将其内容作为静态资源提供服务。一旦挂载，你就可以通过访问`*<origin>*/pages/*<filename>*`来访问每个文件。

通过实现示例 6-5，你从与你的 API 服务器相同的源提供服务 HTML。这避免了触发浏览器的 CORS 安全机制，该机制可能会阻止到达服务器的出站请求。

你现在可以通过访问`http://localhost:8000/pages/sse-client.html`来访问 HTML 页面。

### 跨源资源共享

如果你尝试直接在你的浏览器中打开示例 6-4 HTML 文件并点击开始流式传输按钮，你会注意到没有任何反应。你可以检查浏览器的网络标签来查看出站请求发生了什么。

经过一些调查，你应该会注意到你的浏览器已经阻止了对服务器的出站请求，因为它的预检**跨源资源共享**（CORS）检查与你的服务器失败了。
