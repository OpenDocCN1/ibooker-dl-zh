
# Alembic 迁移文件示例

以下是一个 Alembic 迁移文件的示例，用于创建 `conversations` 和 `messages` 两张表：

```py
# alembic/versions/24c35f32b152.py

from datetime import UTC, datetime
import sqlalchemy as sa
from alembic import op

"""
Revision ID: 2413cf32b712 Revises:
Create Date: 2024-07-11 12:30:17.089406
"""

# revision identifiers, used by Alembic.
revision = "24c35f32b152"
down_revision = None
branch_labels = None

def upgrade():
    op.create_table(
        "conversations",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("title", sa.String, nullable=False),
        sa.Column("model_type", sa.String, index=True, nullable=False),
        sa.Column(
            "created_at", sa.DateTime, default=datetime.now(UTC), nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime,
            default=datetime.now(UTC),
            onupdate=datetime.now(UTC),
            nullable=False,
        ),
    )

    op.create_table(
        "messages",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "conversation_id",
            sa.BigInteger,
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            index=True,
            nullable=False,
        ),
        sa.Column("prompt_content", sa.Text, nullable=False),
        sa.Column("response_content", sa.Text, nullable=False),
        sa.Column("prompt_tokens", sa.Integer, nullable=True),
        sa.Column("response_tokens", sa.Integer, nullable=True),
        sa.Column("total_tokens", sa.Integer, nullable=True),
        sa.Column("is_success", sa.Boolean, nullable=True),
        sa.Column("status_code", sa.Integer, nullable=True),
        sa.Column(
            "created_at", sa.DateTime, default=datetime.now(UTC), nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime,
            default=datetime.now(UTC),
            onupdate=datetime.now(UTC),
            nullable=False,
        ),
    )

def downgrade():
    op.drop_table("messages")
    op.drop_table("conversations")
```

现在你已经更新了第一个迁移文件，你就可以运行它来对数据库进行操作：

```py
$ alembic upgrade head
```

如果您需要回滚操作，可以运行`alembic downgrade`。

Alembic 在底层执行的操作是生成运行或回滚迁移所需的原始 SQL，并在数据库中创建一个`alembic_versions`表。它使用这个表来跟踪已经应用在您的数据库上的迁移，这样再次运行`alembic upgrade head`命令就不会执行任何重复的迁移。

如果在任何情况下，您的数据库模式和迁移历史发生了偏移，您始终可以删除`versions`目录中的文件并截断`alembic_revision`表。然后重新初始化 Alembic，以全新的环境开始对现有数据库进行操作。

## 警告

在使用迁移文件迁移数据库后，请确保将其提交到 Git 仓库。避免在迁移数据库后重新编辑迁移文件，因为 Alembic 将通过与其版本控制表交叉检查来跳过现有的迁移。

如果迁移文件已经运行，它将不会检测其内容的变化。

要更新数据库模式，请创建一个新的迁移文件。

按照上述工作流程，现在不仅可以版本控制数据库模式，还可以在应用程序需求变化时管理生产环境中的更改。

# 在处理实时流时存储数据

现在，您应该能够实现自己的 CRUD 端点，以检索和修改数据库中的用户对话和消息记录。

一个悬而未决的问题是，如何在数据流式传输端点中处理事务，例如 LLM 将流式输出到客户端。

您无法将数据流式传输到传统的关系型数据库，因为确保流式事务的 ACID 兼容性将是一项挑战。相反，您将希望尽快执行标准数据库操作，一旦 FastAPI 服务器向客户端返回响应。这正是 FastAPI 的后台任务可以解决的问题，正如您在示例 7-15 中看到的那样。

## 示例 7-15. 存储 LLM 输出流的内容

```py
# main.py

from itertools import tee
from database import DBSessionDep
from entities import Message
from fastapi import BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from repositories.conversations import Conversation
from repositories.messages import MessageRepository
from sqlalchemy.ext.asyncio import AsyncSession

async def store_message( ![1](img/1.png)
    prompt_content: str,
    response_content: str,
    conversation_id: int,
    session: AsyncSession,
) -> None:
    message = Message(
        conversation_id=conversation_id,
        prompt_content=prompt_content,
        response_content=response_content,
    )
    await MessageRepository(session).create(message)

@app.get("/text/generate/stream")
async def stream_llm_controller(
    prompt: str,
    background_task: BackgroundTasks,
    session: DBSessionDep,
    conversation: Conversation = Depends(get_conversation), ![2](img/2.png)
) -> StreamingResponse:
    # Invoke LLM and obtain the response stream
    ...
    stream_1, stream_2 = tee(response_stream) ![3](img/3.png)
    background_task.add_task(
        store_message, prompt, "".join(stream_1), conversation.id, session
    ) ![4](img/4.png)
    return StreamingResponse(stream_2)
```

![1](img/1.png)[#co_integrating_databases_into_ai_services_CO12-1]

创建一个函数来存储与对话相关的消息。

![2](img/2.png)[#co_integrating_databases_into_ai_services_CO12-2]

检查对话记录是否存在，并在依赖关系中获取它。

![3](img/3.png)[#co_integrating_databases_into_ai_services_CO12-3]

创建 LLM 流的两个单独副本，一个用于`StreamingResponse`，另一个用于后台任务处理。

![4](img/4.png)[#co_integrating_databases_into_ai_services_CO12-4]

在`StreamingResponse`完成后创建一个后台任务来存储消息。

在示例 7-15 中，您允许 FastAPI 完全将 LLM 响应流式传输到客户端。

无论你使用的是 SSE 还是 WebSocket 端点，都不会有任何区别。一旦请求和响应完全流式传输，调用一个后台任务，传入完整的流式响应内容。在后台任务中，你可以在请求发送后运行一个函数来存储消息，并带有完整的 LLM 响应内容。

使用相同的方法，你甚至可以根据第一条消息的内容为对话生成一个标题。为此，你可以再次调用 LLM，使用对话中第一条消息的内容，请求为对话提供一个合适的标题。一旦生成了对话标题，你可以在数据库中创建对话记录，如示例 7-16 所示。

## 示例 7-16. 使用 LLM 根据初始用户提示生成对话标题

```py
from entities import Conversation
from openai import AsyncClient
from repositories.conversations import ConversationRepository
from sqlalchemy.ext.asyncio import AsyncSession

async_client = AsyncClient(...)

async def create_conversation(
    initial_prompt: str, session: AsyncSession
) -> Conversation:
    completion = await async_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Suggest a title for the conversation "
                           "based on the user prompt",
            },
            {
                "role": "user",
                "content": initial_prompt,
            },
        ],
        model="gpt-3.5-turbo",
    )
    title = completion.choices[0].message.content
    conversation = Conversation(
        title=title,
        # add other conversation properties
        ...
    )
    return await ConversationRepository(session).create(conversation)
```

在 FastAPI 中使用 SQLAlchemy 与 Alembic 是一种经过验证和测试的方法来处理关系型数据库，因此你更有可能找到很多关于集成这些技术的资源。

SQLAlchemy ORM 和 Alembic 都允许你与数据库交互并控制其模式的变化。

# 摘要

在本章中，你深入探讨了将数据库集成到你的 FastAPI 应用程序中以存储和检索用户对话的关键方面。

你学会了如何识别何时需要数据库以及如何为你的项目选择合适的数据库类型，无论是关系型还是非关系型。通过理解关系型数据库的底层机制和非关系型数据库的使用场景，你现在可以做出关于数据库选择的明智决策。

你还探索了与关系型数据库一起工作的开发工作流程、工具和最佳实践。这包括学习提高查询性能和效率的技术，以及管理数据库模式变化的策略。此外，你在团队工作中获得了关于管理代码库、数据库模式和数据漂移的见解。

随着你继续前进，下一章将指导你实现用户管理、身份验证和授权机制。这将进一步增强你的应用程序的安全性并提升用户体验，基于你在本章中建立的稳固数据库基础。

^(1) 请参阅此 [reddit 讨论线程](https://oreil.ly/OMaOT)。
