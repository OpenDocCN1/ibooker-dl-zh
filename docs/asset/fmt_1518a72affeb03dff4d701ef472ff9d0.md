
# 网络爬虫与 RAG 模块实现

## 网络爬虫实现

以下代码实现了一个异步网络爬虫，用于从维基百科页面提取文本内容：

```py
# scraper.py

import asyncio
import re

import aiohttp
from bs4 import BeautifulSoup
from loguru import logger

def extract_urls(text: str) -> list[str]:
    url_pattern = r"(?P<url>https?:\/\/[^\s]+)" ![1](img/1.png)
    urls = re.findall(url_pattern, text) ![2](img/2.png)
    return urls

def parse_inner_text(html_string: str) -> str:
    soup = BeautifulSoup(html_string, "lxml")
    if content := soup.find("div", id="bodyContent"): ![3](img/3.png)
        return content.get_text()
    logger.warning("Could not parse the HTML content")
    return ""

async def fetch(session: aiohttp.ClientSession, url: str) -> str:
    async with session.get(url) as response: ![4](img/4.png)
        html_string = await response.text()
        return parse_inner_text(html_string)

async def fetch_all(urls: list[str]) -> str:
    async with aiohttp.ClientSession() as session: ![5](img/5.png)
        results = await asyncio.gather(
            *[fetch(session, url) for url in urls], return_exceptions=True
        )
    success_results = [result for result in results if isinstance(result, str)]
    if len(results) != len(success_results): ![6](img/6.png)
        logger.warning("Some URLs could not be fetched")
    return " ".join(success_results)
```

![1](img/#co_achieving_concurrency_in_ai_workloads_CO5-1)

一个简单的正则表达式模式，将 URL 捕获到名为`url`的命名组中，并匹配`http`和`https`协议。为了简单起见，此模式对 URL 的定义较为宽松，不验证域名或路径的结构，也不考虑 URL 中的查询字符串或锚点。

![2](img/2.png)[(#co_achieving_concurrency_in_ai_workloads_CO5-2)]

在文本中找到所有非重叠的正则表达式模式匹配。

![3](img/3.png)[(#co_achieving_concurrency_in_ai_workloads_CO5-3)]

使用`bs4` Beautiful Soup 包解析 HTML 字符串。在维基百科页面中，文章内容嵌套在一个具有`id="bodyContent"`的`div`容器中，因此解析逻辑假设只有维基百科 URL 会被传入。你可以更改此逻辑以处理其他 URL，或者只需使用`soup.getText()`来获取 HTML 中嵌套的任何文本内容。然而，请注意，如果你以这种方式解析原始 HTML，解析内容中会有很多噪声，这可能会混淆 LLM。

![4](img/4.png)[(#co_achieving_concurrency_in_ai_workloads_CO5-4)]

给定一个`aiohttp`会话和一个 URL，执行异步的`get`请求。创建一个`response`异步上下文管理器，并在该上下文管理器内`await`响应。

![5](img/5.png)[(#co_achieving_concurrency_in_ai_workloads_CO5-5)]

给定一个 URL 列表，创建一个客户端会话异步上下文管理器以异步执行多个获取调用。由于`fetch()`是一个协程函数（即它使用`await`关键字），`fetch_all()`需要在`asyncio.gather()`内部运行多个`fetch()`协程，以便在事件循环上异步执行。

![6](img/6.png)[(#co_achieving_concurrency_in_ai_workloads_CO5-6)]

检查所有 URL 是否已成功获取，如果没有，则发出警告。

你现在拥有了实现`/generate/text`端点中网络爬虫功能的所需实用爬虫函数。

接下来，将文本到文本处理器升级为使用爬虫函数，通过依赖项以异步方式执行，如示例 5-7 所示。

### 示例 5-7：将网络爬虫功能作为依赖项注入 FastAPI LLM 处理器

```py
# dependencies.py

from fastapi import Body
from loguru import logger

from schemas import TextModelRequest
from scraper import extract_urls, fetch_all

async def get_urls_content(body: TextModelRequest = Body(...)) -> str: ![1](img/1.png)
    urls = extract_urls(body.prompt)
    if urls:
        try:
            urls_content = await fetch_all(urls)
            return urls_content
        except Exception as e:
            logger.warning(f"Failed to fetch one or several URls - Error: {e}")
    return ""

# main.py

from fastapi import Body, Depends, Request
from dependencies import construct_prompt
from schemas import TextModelResponse

@app.post("/generate/text", response_model_exclude_defaults=True) ![2](img/2.png)
async def serve_text_to_text_controller(
    request: Request,
    body: TextModelRequest = Body(...),
    urls_content: str = Depends(get_urls_content) ![3](img/3.png)
) -> TextModelResponse:
    ... # rest of controller logic
    prompt = body.prompt + " " + urls_content
    output = generate_text(models["text"], prompt, body.temperature)
    return TextModelResponse(content=output, ip=request.client.host)
```

![1](img/1.png)[(#co_achieving_concurrency_in_ai_workloads_CO6-1)]

实现一个`get_urls_content` FastAPI 依赖项，它从请求体中获取用户提示并找到所有 URL。然后它返回所有 URL 的内容作为一个长字符串。该依赖项内置异常处理，以处理任何 I/O 错误，通过返回一个空字符串并在服务器上记录警告。

![2](img/2.png)[(#co_achieving_concurrency_in_ai_workloads_CO6-2)]

当在 FastAPI 中使用`aiohttp`时，你不需要自己管理事件循环，因为 FastAPI 作为一个异步框架，会处理事件循环。你可以定义你的端点为一个异步函数，并在处理器内部或通过依赖项（如本例所示）使用`aiohttp`来发送异步 HTTP 请求。

![3](img/3.png)[(#co_achieving_concurrency_in_ai_workloads_CO6-3)]

通过 FastAPI 的`Depends`类将`get_urls_content`依赖项调用的结果注入到处理器中。在这里使用 FastAPI 依赖项使得控制器逻辑保持小巧、简洁和可读。

现在，在浏览器中运行 Streamlit 客户端，尝试您的新特性。图 5-6 展示了我的实验。

![bgai 0506](img/bgai_0506.png)

#### 图 5-6\. 请自托管 TinyLlama 模型总结维基百科文章

恭喜！您已经学会了如何构建一个简单的非阻塞网络爬虫，与您自己的 LLM 一起工作。在这个小型项目中，您利用`re`包匹配用户提示中的 URL 模式，然后使用`aiohttp`库异步并发地抓取多个页面。然后，您使用`BeautifulSoup`包通过获取抓取的 HTML 字符串中`bodyContent` ID 的`div`容器的文本内容来解析维基百科文章的内容。对于其他网站或内部公司网页，您始终可以修改解析逻辑以进行适当的解析。最后，您将整个抓取逻辑封装在 FastAPI 依赖项中，内置异常处理，以便在升级文本模型服务处理程序时使用依赖注入。

请记住，您的爬虫无法处理具有动态布局且由服务器渲染的复杂页面。在这种情况下，您可以在您的网络爬虫中添加一个无头浏览器来导航动态页面。

此外，抓取外部网站的内容将具有挑战性，因为大多数网站可能实施了如*IP 封锁*或*CAPTCHAs*等常见的防爬措施。与外部网站保持*数据质量*和*一致性*也是一个持续性的挑战，因为您可能需要定期更新您的抓取脚本以确保准确和可靠的数据提取。

您现在应该对构建需要通过异步网络请求与网络交互的 GenAI 服务感到更加得心应手。

接下来，我们将通过构建“与您的文档对话”功能来查看其他 I/O 异步交互，例如与数据库和文件系统的交互。

这个功能允许用户通过 Streamlit 界面上传文档到您的服务。上传文档的内容随后被提取、处理并保存在数据库中。随后，在用户与 LLM 交互期间，一个异步检索系统从数据库中检索语义相关的内容，然后用于增强 LLM 提供的上下文。

这个过程被称为 RAG，我们将为您的 LLM 构建一个模块。

## 项目：与文档对话（RAG）

在这个项目中，我们将把一个 RAG 模块构建到您的 GenAI 服务中，让您亲身体验异步与外部系统（如数据库和文件系统）交互的过程。

您可能对 RAG 模块的用途及其必要性感到好奇。RAG 是一种简单的方法，用于通过为知识密集型任务提供自定义数据源来增强 LLM 提示的上下文.^(5)这是一种有效的技术，可以在不进行复杂且昂贵的 LLM 微调的情况下，将 LLM 的响应基于数据中的事实。

组织渴望使用自己的 LLM 实施 RAG，因为它允许员工通过 LLM 与他们庞大的内部知识库互动。通过 RAG，企业期望内部知识库、系统和程序对需要它们来回答问题的人变得可访问和易于获取。这种对公司信息库的访问预期将提高生产力，减少查找信息的时间和成本，并提高任何企业的利润。

然而，LLM 容易生成不符合用户指令的响应。换句话说，LLM 可以使用基于事实或现实的信息或数据来*虚构*响应。

这些虚构可能是因为模型依赖于其训练数据中的模式，而不是直接访问外部、最新和事实性的数据。LLM 可以以自信的方式呈现不正确或无意义的答案、编造的故事或缺乏事实依据的声明。

因此，对于更复杂和知识密集型任务，您希望您的 LLM 访问外部知识源以完成任务。这可以确保生成响应的事实一致性并提高其可靠性。图 5-7 展示了整个流程。

![bgai 0507](img/bgai_0507.png)

#### 图 5-7\. RAG

在这个项目中，您将为您的 LLM 服务构建一个简单的 RAG 模块，以便用户可以上传和与他们的文档进行交流。

### 注意

关于 RAG 有很多要了解的内容。这足以用每天出版的新论文填满几本教科书，每天都有新技术和算法被发布。

我建议查看其他关于 LLM 的出版物，以了解 RAG 流程和高级 RAG 技术。

RAG 的管道包括以下阶段：

1. *提取* 从文件系统中的文档提取以将文本内容分块加载到内存中。

2. *转换* 通过清理、分割和准备文本内容，以便将其传递到嵌入模型中，以生成表示块语义意义的嵌入向量。

3. *存储* 将嵌入向量及其元数据（如来源和文本块）存储在 Qdrant 等向量存储中。

4. *检索* 通过对用户查询 LLM 的语义搜索来检索语义相关的嵌入向量。原始文本块——存储为检索向量的元数据——然后用于增强（即，增强 LLM 收到的初始提示中的上下文）。

5. *生成* LLM 响应时绕过查询和检索的块（即上下文）直接提交给 LLM 以获取响应。

您可以在图 5-8 中看到完整的管道。

![bgai 0508](img/bgai_0508.png)

#### 图 5-8\. RAG 管道

您可以将 图 5-8 中所示的流程图应用到您现有的服务中。图 5-9 展示了启用 RAG 的“与您的文档对话”服务的系统架构。

![bgai 0509](img/bgai_0509.png)

#### 图 5-9\. 与您的文档对话系统架构

图 5-9 概述了用户通过 Streamlit 接口上传的文档的存储方式，然后用于处理并存储到数据库中，以便稍后检索以增强 LLM 提示。

在 图 5-9 中实现 RAG 系统之前的第一步是在 Streamlit 客户端和您的后端 API 中包含文件上传功能。

使用 FastAPI 的 `UploadFile` 类，您可以分块接收用户的文档并保存到文件系统或任何其他文件存储解决方案，例如 blob 存储。这里需要注意的重要事项是，通过异步编程，这种 I/O 操作是非阻塞的，FastAPI 的 `UploadFile` 类支持这种异步操作。

### 小贴士

由于用户可能上传大文档，FastAPI 的 `UploadFile` 类支持 *分块* 存储，一次存储一个上传的文档。

这将防止您的服务内存被阻塞。您还希望通过禁止用户上传超过一定大小的文档来保护您的服务。

示例 5-8 展示了如何实现异步文件上传功能。

### 小贴士

您需要安装 `aiofiles` 包以异步上传文件，同时安装 `python-multipart` 以从 HTML 表单接收上传的文件：

```py
$ pip install aiofiles python-multipart
```

### 示例 5-8\. 实现异步文件上传端点
