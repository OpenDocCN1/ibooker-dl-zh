
要运行示例 9-10，您需要一个 Redis 数据库来存储用户 API 使用数据：

```py
$ pip install coredis
$ docker pull redis
$ docker run \
  --name rate-limit-redis-cache \
  -d \
  -p 6379:6379 \
  redis
```

### 示例 9-10：在多个实例中添加集中使用内存存储（Redis）

```py
from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware

app.state.limiter = Limiter(storage_uri="redis://localhost:6379")
app.add_middleware(SlowAPIMiddleware)
```

您现在有一个按预期在多个实例上工作的速率限制 API。

您可以通过实现自己的限制器，该限制器由`limits`包支持，来解决这个问题。或者，您可以通过*负载均衡器*、*反向代理*或*API 网关*来应用速率限制。

每个解决方案都可以在基础设施层执行请求路由、速率限制、协议转换和流量监控，同时执行速率限制。如果您不需要定制的速率限制逻辑，那么在您的用例中，外部应用速率限制可能更合适。

## 限制 WebSocket 连接

不幸的是，`slowapi`包在撰写本文时也不支持限制异步和 WebSocket 端点。

由于 WebSocket 连接可能是长期存在的，你可能想要限制通过套接字发送的数据转换速率。你可以依赖外部包，如`fastapi-limiter`，来限制 WebSocket 连接，如图示例 9-11 所示。

### 示例 9-11：使用`fastapi_limiter`包限制 WebSocket 连接的速率

```py
from contextlib import asynccontextmanager
import redis
from fastapi import Depends, FastAPI
from fastapi.websockets import WebSocket
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import WebSocketRateLimiter

...

@asynccontextmanager
async def lifespan(_: FastAPI): ![1](img/1.png)
    redis_connection = redis.from_url("redis://localhost:6379", encoding="utf8")
    await FastAPILimiter.init(redis_connection)
    yield
    await FastAPILimiter.close()

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket, user_id: int = Depends(get_current_user) ![2](img/2.png)
):
    ratelimit = WebSocketRateLimiter(times=1, seconds=5)
    await ws_manager.connect(websocket)
    try:
        while True:
            prompt = await ws_manager.receive(websocket)
            await ratelimit(websocket, context_key=user_id) ![3](img/3.png)
            async for chunk in azure_chat_client.chat_stream(prompt, "ws"):
                await ws_manager.send(chunk, websocket)
    except WebSocketRateLimitException:
        await websocket.send_text(f"Rate limit exceeded. Try again later")
    finally:
        await ws_manager.disconnect(websocket)
```

![1](img/1.png)[#co_securing_ai_services_CO7-1]

使用 Redis 存储后端配置`FastAPILimiter`应用程序的生命周期。

![2](img/2.png)[#co_securing_ai_services_CO7-2]

配置 WebSocket 速率限制器，允许每秒一个请求。

![3](img/3.png)[#co_securing_ai_services_CO7-3]

使用用户的 ID 作为速率限制的唯一标识符。

示例 9-11 展示了如何限制给定用户的活跃 WebSocket 连接数。

除了限制 WebSocket 端点之外，你还可能想要限制你的 GenAI 模型的数据流速率。接下来，让我们看看如何节流实时数据流。

## 节流实时流

当与实时流一起工作时，你可能需要减慢流速率，以便给客户端足够的时间来消费流，并提高多个客户端之间的流吞吐量。此外，节流可以帮助你管理网络带宽、服务器负载和资源利用率。

在流生成层应用*节流*，如图示例 9-12，如果你的服务处于压力之下，这是一种有效的管理吞吐量的方法。

### 示例 9-12：节流流

```py
class AzureOpenAIChatClient:
    def __init__(self, throttle_rate = 0.5): ![1](img/1.png)
        self.aclient = ...
        self.throttle_rate = throttle_rate

    async def chat_stream(
            self, prompt: str, mode: str = "sse", model: str = "gpt-3.5-turbo"
    ) -> AsyncGenerator[str, None]:
        stream = ...  # OpenAI chat completion stream
        async for chunk in stream:
            await asyncio.sleep(self.throttle_rate) ![2](img/2.png)
            if chunk.choices[0].delta.content is not None:
                yield (
                    f"data: {chunk.choices[0].delta.content}\n\n"
                    if mode == "sse"
                    else chunk.choices[0].delta.content
                )
                await asyncio.sleep(0.05)

        if mode == "sse":
            yield f"data: [DONE]\n\n"
```

![1](img/1.png)[#co_securing_ai_services_CO8-1]

设置固定的节流速率或根据使用情况动态调整。

![2](img/2.png)[#co_securing_ai_services_CO8-2]

减慢流速率而不阻塞事件循环。

你可以使用节流流在 SSE 或 WebSocket 端点内使用。或者，你可以根据你自己的自定义策略限制每个活动的 WebSocket 连接数。

除了实时流的本地节流之外，你还可以在基础设施层利用*流量整形*。

使用防护措施、速率限制和节流，应该为保护你的服务免受滥用和误用提供足够的障碍。

在下一节中，你将了解更多优化技术，这些技术可以帮助你降低延迟，提高响应质量和吞吐量，同时降低你的 GenAI 服务的成本。

# 摘要

本章提供了 GenAI 服务的攻击向量全面总结以及如何防范对抗性尝试、误用和滥用。

你已经学会了在评估和内容过滤机制的同时实现输入和输出防护栏，以调节服务使用。在防护栏的基础上，你还开发了 API 速率限制和节流保护来管理服务器负载并防止滥用。

在下一章中，我们将学习通过各种技术优化 AI 服务，例如缓存、批量处理、模型量化、提示工程和模型微调。

^(1) 受到[《OpenAI 烹饪手册》中的“如何实现 LLM 防护栏”](https://oreil.ly/UQV6i)的启发。
