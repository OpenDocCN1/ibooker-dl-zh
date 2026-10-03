
# 第十章\. 优化 AI 服务

在本章中，你将学习如何通过提示工程、模型量化和缓存机制进一步优化你的服务。

## 优化技术

优化 AI 服务的目标是提高输出质量或性能（延迟、吞吐量、成本等）。

与性能相关的优化包括以下内容：

+   使用批处理 API
+   缓存（关键字、语义、上下文或提示）
+   模型量化

与质量相关的优化包括以下内容：

+   使用结构化输出
+   提示工程
+   模型微调

让我们更详细地回顾每一个。

### 批处理

通常，你希望 LLM 同时处理多个条目。最明显的解决方案是每个条目提交多个 API 调用。然而，这种明显的方法可能成本高昂且速度慢，可能会导致你的模型提供商限制你的速率。

在这种情况下，你可以利用两种不同的技术通过 LLM 批处理你的数据：

+   更新你的结构化输出模式以同时返回多个示例
+   识别和使用为批处理设计的好模型提供商 API

第一个解决方案要求你更新你的 Pydantic 模型或模板提示以请求每个请求的输出列表。在这种情况下，你可以在少量请求中批处理你的数据，而不是每个条目一个请求。

第一个解决方案的实现示例如示例 10-1 所示。

##### 示例 10-1\. 更新解析多个项目的结构化输出模式

```py
from pydantic import BaseModel

class BatchDocumentClassification(BaseModel):
    class Category(BaseModel):
        document_id: str
        category: list[str]

    categories: list[Category] ![1](img/1.png)
```

![1](img/#co_optimizing_ai_services_CO1-1)

更新 Pydantic 模型以包含一个 `Category` 模型的列表。

现在，你可以将新的模式与文档标题列表一起传递给 OpenAI 客户端，以在单个 API 调用中处理多个条目。然而，如果可用，使用批处理 API 可能是一个替代方案，也可能是最佳解决方案。

幸运的是，像 OpenAI 这样的模型提供商已经为这种用例提供了相关的 API。在底层，这些提供商可能会运行任务队列以在后台处理任何单个批处理作业，同时在你获取结果之前提供状态更新。

与直接使用标准端点相比，你将能够发送异步的请求组，成本更低（使用 OpenAI 可降低高达 50%1），享受更高的速率限制，并保证完成时间。批处理作业服务非常适合处理不需要即时响应的作业，例如在后台使用 OpenAI LLM 解析、分类或翻译大量文档。

要提交批处理作业，你需要一个 `jsonl` 文件，其中每行包含对 API 的单个请求的详细信息，如示例 10-2 所示。同样，如本例所示，要创建 JSONL 文件，你可以遍历你的条目并动态生成文件。

##### 示例 10-2\. 从条目创建 JSONL 文件

```py
import json
from uuid import UUID

def create_batch_file(
    entries: list[str],
    system_prompt: str,
    model: str = "gpt-4o-mini",
    filepath: str = "batch.jsonl",
    max_tokens: int = 1024,
) -> None:
    with open(filepath, "w") as file:
        for _, entry in enumerate(entries, start=1):
            request = {
                "custom_id": f"request-{UUID()}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {"role": "user", "content": entry},
                    ],
                    "max_tokens": max_tokens,
                },
            }
            file.write(json.dumps(request) + "\n")
```

一旦创建，您可以将文件提交给批处理 API 进行处理，如示例 10-3 所示。

##### 示例 10-3\. 使用 OpenAI 批处理 API 处理批处理作业

```py
from loguru import logger
from openai import AsyncOpenAI
from openai.types import Batch

client = AsyncOpenAI()

async def submit_batch_job(filepath: str) -> Batch:
    if ".jsonl" not in filepath:
        raise FileNotFoundError(f"JSONL file not provided at {filepath}")

    file_response = await client.files.create(
        file=open(filepath, "rb"), purpose="batch"
    )

    batch_job_response = await client.batches.create(
        input_file_id=file_response.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "document classification job"},
    )
    return batch_job_response

async def retrieve_batch_results(batch_id: str):
    batch = await client.batches.retrieve(batch_id)
    if (
        status := batch.status == "completed"
        and batch.output_file_id is not None
    ):
        file_content = await client.files.content(batch.output_file_id)
        return file_content
    logger.warning(f"Batch {batch_id} is in {status} status")
```

您现在可以利用离线批处理端点一次处理多个条目，并保证有保证的周转时间和显著的成本节约。

除了利用结构化输出和批处理 API 优化您的服务外，您还可以利用缓存技术显著加快服务器的响应时间和资源成本。

### 缓存

在 GenAI 服务中，您通常会依赖于需要大量计算或长时间处理的数据/模型响应。如果您有多个用户请求相同的数据，重复相同的操作可能是浪费的。相反，您可以使用缓存技术来存储和检索频繁访问的数据，以通过加快响应时间、减少服务器负载、节省带宽和运营成本来优化您的服务。

例如，在一个公共 FAQ 聊天机器人中，用户提出的问题大多相同，您可能希望更长时间地重用缓存的响应。另一方面，对于更个性化和动态的聊天机器人，您可以频繁刷新（即，使缓存失效）缓存的响应。

###### 小贴士

您应该始终根据数据的性质和可接受的陈旧程度来考虑缓存刷新的频率。

对 GenAI 服务最相关的缓存策略包括：

+   关键字缓存
+   语义缓存
+   上下文或提示缓存

让我们更详细地回顾一下。

#### 关键字缓存

如果您只需要一个简单的缓存机制来存储函数或端点响应，您可以使用 *关键字缓存*，它涉及根据输入查询的精确匹配作为键值对缓存响应。

在 FastAPI 中，`fastapi-cache` 等库可以帮助您通过几行代码在任意函数或端点上实现关键字缓存。FastAPI 缓存还为您提供了附加存储后端（如 Redis）以集中管理缓存存储的选项。

###### 小贴士

或者，您可以使用像 `cachetools` 这样的底层包来实现自己的自定义缓存机制，使用缓存存储。

要开始，您只需在应用程序生命周期内初始化和配置缓存系统，如示例 10-4 所示。您可以使用以下命令安装 FastAPI 缓存：

```py
$ pip install "fastapi-cache2[redis]"
```

##### 示例 10-4\. 配置 FastAPI 缓存生命周期

```py
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis import asyncio as aioredis

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    redis = aioredis.from_url("redis://localhost")
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache") ![1](img/1.png)
    yield

app = FastAPI(lifespan=lifespan)
```

![1](img/1.png)(#co_optimizing_ai_services_CO2-1)

使用 `RedisBackend` 初始化 `FastAPICache`，使其不解码响应，以便缓存数据以字节（二进制）形式存储。这是因为解码响应会通过改变原始响应格式来破坏缓存。

一旦配置了缓存系统，您就可以装饰您的函数或端点处理器以缓存它们的输出，如示例 10-5 所示。

##### 示例 10-5\. 函数和端点结果缓存

```py
from fastapi import APIRouter
from fastapi_cache.decorator import cache

router = APIRouter(prefix="/generate", tags=["Resource"])

@cache()
async def classify_document(title: str) -> str:
    ...

@router.post("/text")
@cache(expire=60) ![1](img/1.png)
async def serve_text_to_text_controller():
    ...
```

![1](img/#co_optimizing_ai_services_CO3-1)

`cache()` 装饰器必须始终放在最后。通过设置 `expires=60` 来在 60 秒内使缓存失效，以重新计算输出。

在示例 10-5 中显示的 `cache()` 装饰器注入了 `Request` 和 `Response` 对象的依赖项，以便它可以向输出响应添加缓存控制头。这些缓存控制头指示客户端如何通过指定一组指令（即指令）在其端缓存响应。

这些是在发送响应时的一些常见缓存控制指令：

`max-age`

定义响应被视为新鲜的最大时间量（以秒为单位）

`no-cache`

强制重新验证，以便客户端检查与服务器之间的持续更新

`no-store`

完全阻止缓存

`private`

在私有缓存中存储响应（例如，浏览器中的本地缓存）

响应可以包含如 `Cache-Control: max-age=180, private` 这样的缓存控制头，以设置这些指令.^(2)

由于关键字缓存基于精确匹配，因此它更适合期望频繁重复匹配输入的函数和 API。然而，在接受变量用户查询的 GenAI 服务中，您可能需要考虑其他缓存机制，这些机制在返回缓存响应时依赖于输入的意义。这正是语义缓存可以证明有用的地方。

#### 语义缓存

*语义缓存* 是一种缓存机制，根据相似输入返回存储的值。

在底层，系统使用编码器和嵌入向量来捕获输入的语义和意义。然后，它在存储的关键值对之间执行相似性搜索，以返回缓存响应。

与关键字缓存相比，相似输入可以返回相同的缓存响应。输入到系统中的内容不必完全相同才能被识别为相似。即使这些输入具有不同的句子结构或表述，或包含不准确之处，它们仍然会被捕获为相似，以携带相同的意义。并且，请求的是相同的响应。以下查询被认为是携带相同意图的相似查询：

+   您如何使用 FastAPI 构建生成性服务？
+   开发用于 GenAI 的 FastAPI 服务的流程是什么？

该缓存系统通过减少 API 调用至 [30–40%](https://oreil.ly/gjGz6)（即，60–70% 缓存命中率）来显著降低成本^(3)，具体取决于用例和用户规模的大小。例如，接收大量用户常见问题的问答 RAG 应用可以使用语义缓存将 API 调用减少 69%。

在典型的 RAG 系统中，有两个地方拥有缓存可以减少资源密集型和耗时操作：

+   *在 LLM 之前* 返回缓存响应而不是生成新的响应
+   *在向量存储之前*，用缓存文档丰富提示而不是搜索和检索新鲜文档

当将语义缓存组件集成到您的 RAG 系统中时，您应该考虑返回缓存响应是否会对您的应用程序的用户体验产生负面影响。例如，如果缓存 LLM 响应，由于它们的高度语义相似性，以下两个查询都会返回相同的响应，导致语义缓存系统将它们视为几乎相同：

+   用 100 个字总结此文本
+   用 50 个字总结此文本

这使得您的服务感觉像是在对查询没有响应。由于您可能仍然希望在您的应用程序中获取不同的 LLM 输出，我们将为您的 RAG 系统实现一个文档检索语义缓存。图 10-1 展示了完整的系统架构。

![bgai 1001](img/bgai_1001.png)

###### 图 10-1\. RAG 系统架构中的语义缓存

让我们先从从头开始实现语义缓存系统开始，然后我们将回顾如何将功能卸载到外部库，如 `gptcache`。

##### 从零开始构建语义缓存服务

您可以通过实现以下组件来实施一个语义缓存系统：

+   缓存存储客户端
+   文档向量存储客户端
+   嵌入模型

示例 10-6 展示了如何实现缓存存储客户端。

##### 示例 10-6\. 缓存存储客户端
