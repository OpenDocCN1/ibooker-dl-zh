

# 第七章\. 将数据库集成到 AI 服务中

在本章中，你将集成数据库到当前的 API 服务中，以存储和检索用户交互。

本章假设你具备使用数据库和结构化查询语言（SQL）的基本经验，因此它不会涵盖 SQL 编程和数据库工作流程的各个方面。相反，你将学习在将数据库集成到与 LLM 等 GenAI 模型交互的 FastAPI 应用程序时，更高层次的数据库概念、开发工作流程和最佳实践。

作为其中的一部分，你将了解关系型（SQL）数据库与无关系型（NoSQL）数据库在应用程序开发中的作用，并能够自信地选择适合你用例的正确数据库。接下来，你将了解更多关于关系型数据库及其相关工具（如对象关系映射器（ORMs）和数据库迁移工具）的功能。最后，作为一个动手练习，你将使用 `SQLAlchemy` 和 `Alembic` 将数据库集成到现有的应用程序中，以存储和检索与 LLM 的用户对话。

到本章结束时，你将在选择、配置和解决 GenAI 应用程序中的数据库相关问题方面更有信心。

## 数据库的作用

在构建后端服务时，你通常需要一个数据库来持久化应用程序状态和存储用户数据。在其他情况下，你的应用程序可能不需要数据库，你不应该尝试添加一个，因为任何数据库集成都可能显著增加你服务的复杂性。

在以下几种情况下，你可以选择不使用数据库：

1. 你的应用程序可以在每个新的用户会话启动时从全新状态开始。
2. 重新计算应用程序数据是直接且资源高效的。
3. 应用程序数据足够小，可以存储在内存中。
4. 你的应用程序能够容忍由于服务器错误、重启或其他意外事件等原因导致的数据丢失。
5. 不同的用户会话或应用程序实例不需要共享数据。
6. 你所需的数据可以直接从外部系统、GenAI 模型和其他应用程序 API 中获取，而不是从你自己的数据库中获取。
7. 用户愿意等待为每个新的会话或操作重新计算数据。
8. 你的服务需求允许数据持久化存储在磁盘上的文件、浏览器存储或外部云存储，而不是数据库。使用这些替代方案，你的服务可以容忍数据存储和检索的可靠性不如数据库高效。
9. 你正在构建一个概念验证，需要不惜一切代价避免项目延迟或复杂性。

一个符合上述标准的应用程序示例是一个仅用于演示目的的*GenAI 图像生成器*。

在这个示例中，您不需要存储任何生成的图像，并且您始终可以从一个全新的状态重新启动或使用应用程序。此外，应用程序不需要知道用户是谁。此外，不需要在会话之间共享数据。此外，如果发生服务器错误，由于您可以即时重新生成新的图像，数据丢失的影响最小。

如您所见，至少有几种情况您不需要数据库来构建自己的 GenAI 服务。然而，您可能会想知道何时您确实需要数据库。

要确定何时需要数据库，您需要了解数据库的作用。简而言之，您可以使用它们以高效格式存储、组织和管理工作数据，从而便于检索、操作和分析。此外，数据库还配备了诸如恢复/备份、并发访问管理、索引、缓存和基于角色的访问控制等关键功能，以及许多其他功能，使它们成为任何显示、生成和消耗数据的服务的不可或缺的组成部分。

在下一节中，我们将以显著详细的程度检查数据库的内部运作机制，重点是关系型数据库，本章的实践示例将聚焦于此。通过对数据库内部结构的详细了解，您可以设计出完全优化且适用于生产的 GenAI API。这将使您能够将繁重的工作负载委托给专门为处理数据密集型任务而设计的数据库引擎。

## 数据库系统

既然您已经了解了何时可以利用数据库，那么让我们更深入地了解您可以使用的不同数据库以及它们的工作原理。

您可以通过将数据库分为两大类来构建数据库的心理模型：*关系型*（SQL）和*非关系型*（NoSQL）数据库。

*SQL*与*NoSQL*的分类基于这样一个事实：关系型数据库使用 SQL 的各种方言作为其主要查询语言，而非关系型数据库通常附带自己的专用查询语言。

对于这两种数据库系统类别，您可以采用一种心理模型来理解这些系统的结构。SQL 和 NoSQL 数据库系统通常由以下内容组成：

+ 最高级别的*服务器*，它托管整个数据库基础设施并消耗系统资源（CPU、RAM 和存储）。
+ 服务器内部的一个或多个*数据库*，它们作为*逻辑容器*，持有相关数据。
+ 数据库内部的一个或多个*模式*（取决于数据库软件），它作为*蓝图*，定义了数据的完整结构和各种结构对象，如索引、逻辑约束、触发器等。然而，与关系型数据库不同，NoSQL 数据库服务器可能不会使用严格的模式。
+ 在数据库内部（作为模式的一部分）创建的零个或多个*表*（SQL）或*集合*（NoSQL），它们将相关数据分组。
+ 每个集合（作为文档）或表格（作为行）中可以有零个或多个*项目*，它们代表特定的记录或实体。

图 7-1 展示了上述分解。

![bgai 0701](img/bgai_0701.png)

###### 图 7-1\. 数据库系统分解

采用如图 7-1 所示的思维模型将帮助您在数据库系统种类不断增多的同时，能够预期到类似的底层机制存在。这种熟悉感有望减少您采用不同数据库系统时的学习曲线。

接下来，让我们简要回顾一下 SQL 和 NoSQL 数据库系统，以便您在构建 API 和服务时能更好地理解它们的使用场景、特性和局限性。

为了帮助您构建关联性和非关联性数据库的思维模型，请参阅图 7-2。

![bgai 0702](img/bgai_0702.png)

###### 图 7-2\. 数据库类型

您还可以使用表 7-1 中的摘要作为本章将要涵盖的数据库类型的参考。

表 7-1\. 数据库比较

| 类型 | 数据模型 | 示例 | 使用场景 |
| --- | --- | --- | --- |
| Key-value stores | 键值对 | Redis, DynamoDB, Memcached | 缓存，会话管理 |
| Graph stores | 节点和边 | Neo4j, Amazon Neptune, ArangoDB | 社交网络，推荐引擎，欺诈检测 |
| Document stores | 文档 | MongoDB, CouchDB, Amazon DocumentDB | 内容管理，电子商务，实时分析 |
| Vector stores | 高维向量 | Pinecone, Weaviate | 推荐系统，图像/文本搜索，机器学习模型存储 |
| Wide-column family stores | 表格，具有行和列 | Apache Cassandra, HBase, ScyllaDB | 时间序列数据，实时分析，日志记录 |

现在您已经对每个常见的关联性和非关联性数据库有了广泛的概述，您可以可视化一个利用这些数据库的实时世界 GenAI 服务。

想象一下，您正在构建一个支持 RAG 的 LLM 服务，它可以与知识库进行交互。这个知识库中的文档相互关联，因此您决定实现一个 RAG 图来捕捉更丰富的上下文。为了实现 RAG 图，您将您的服务与基于图的数据库集成。

现在，为了检索相关的文档块，您还需要将它们嵌入到向量数据库中。作为这一过程的一部分，您还需要一个关系数据库来监控使用情况，并存储用户数据和会话历史。

由于用户可能会提出常见问题，您还决定通过提前生成多个输出来缓存 LLM 的响应。因此，您还向您服务中集成了键值存储。

最后，你希望管理员能够通过版本控制提示的能力来控制系统提示。因此，你将内容管理系统作为提示管理器添加到你的解决方案中。然而，由于提示模板经常发生变化，你也决定集成一个文档数据库。

如你所见，每种数据库类型最终都会解决你复杂 RAG 启用应用程序中的特定问题。一个存储你的后端和用户数据，另一个捕获文档关系，一个存储你的文档嵌入，另一个帮助你存储提示的灵活模式，最后一个帮助你返回缓存输出。

你可以通过 图 7-3 中的可视化了解这些数据库如何协同工作以实现解决方案。

![bgai 0703](img/bgai_0703.png)

###### 图 7-3\. 使用各种数据库类型一起

现在你已经了解了你的 GenAI 服务如何与各种数据库集成，在下一节中，我们将专注于向你的服务添加一个关系型数据库。

## 项目：在关系型数据库中存储与 LLM 的用户对话

在上一节中，我们介绍了添加数据持久性到你的应用程序相关的核心数据库概念。

在本节中，你将集成一个关系型数据库到你的 GenAI 服务中，以便你可以在数据库中存储与 LLM 的用户对话历史。作为这项工作的一个部分，你还将学习管理数据库模式更改和数据迁移的最佳实践、工具和开发工作流程。

对于这个项目，我们将安装一个开源、免费且经过实战检验的 Postgres 关系型数据库，许多企业都在使用它。为了开始，让我们使用 `docker run` 命令下载并运行 Postgres 容器，如 示例 7-1 所示。

##### 示例 7-1\. 下载并运行 Postgres 数据库容器

```py
$ docker run -p 5432:5432  \  ![1](img/1.png) ![2](img/2.png)
	-e POSTGRES_USER=fastapi \
	-e POSTGRES_PASSWORD=mysecretpassword \
	-e POSTGRES_DB=backend_db \
	-e PGDATA=/var/lib/postgresql/data \ ![3](img/3.png)
    -v "$(pwd)"/dbstorage:/var/lib/postgresql/data \ ![4](img/4.png)
    postgres:latest ![1](img/1.png)
```

![1](img/1.png)（[将数据库集成到 AI 服务中](https://wiki.example.org/co_integrating_databases_into_ai_services_CO1-1)）

从 Docker 仓库下载并运行最新的 `postgres` 关系型数据库镜像。

![2](img/2.png)（[将数据库集成到 AI 服务中](https://wiki.example.org/co_integrating_databases_into_ai_services_CO1-2)）

运行 `postgres` 镜像，然后将容器端口 `5432` 映射到主机机器上的相同端口。

![3](img/3.png)（[将数据库集成到 AI 服务中](https://wiki.example.org/co_integrating_databases_into_ai_services_CO1-3)）

使用几个环境变量运行容器，这些变量指定了默认数据库管理员用户名和密码、数据库名称以及容器内的 DBMS 数据位置。

![4](img/4.png)（[将数据库集成到 AI 服务中](https://wiki.example.org/co_integrating_databases_into_ai_services_CO1-4)）

将 Postgres 数据库存储挂载到主机机器文件系统中的当前工作目录下的 `dbstorage` 文件夹。

接下来，让我们安装 `sqlalchemy`、`alembic` 和 `psycopg3` 包：

```py
$ pip install alembic sqlalchemy psycopg3
```

这些经过实战检验的包允许您通过 Python 直接与 Postgres 关系型数据库通信。`psycopg3`是 Python 中流行的 PostgreSQL 数据库适配器，而 SQLAlchemy 是一个 SQL 工具包和 ORM，允许您在 Python 中运行 SQL 查询。

最后，`alembic`包是由 SQLAlchemy 开发者创建的用于与 SQLAlchemy 一起使用的数据库迁移工具。数据迁移工作流程类似于 Git 版本控制系统，但针对您的数据库模式。它允许您管理对模式的更改和更新，从而避免任何数据损坏，跟踪随时间的变化，并在需要时撤销任何更改。

### 定义 ORM 模型

在 Python 中查询数据库的第一步是使用 SQLAlchemy 类定义您的 ORM 模型，如示例 7-2 所示。您可以使用图 8-4 中提到的 ERD 图中的数据模式。

###### 注意

在实现身份验证和授权机制时，您将在下一章中添加`user`表。

##### 示例 7-2\. 定义数据库 ORM 模型

```py
# entities.py

from datetime import UTC, datetime
from sqlalchemy import ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase): ![1](img/1.png)
    pass

class Conversation(Base): ![2](img/2.png)
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column() ![3](img/3.png)
    model_type: Mapped[str] = mapped_column(index=True) ![4](img/4.png)
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.now(UTC), onupdate=datetime.now(UTC) ![5](img/5.png)
    )

    messages: Mapped[list["Message"]] = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan" ![6](img/6.png)
    )

class Message(Base): ![7](img/7.png)
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True ![6](img/6.png)
    )
    prompt_content: Mapped[str] = mapped_column()
    response_content: Mapped[str] = mapped_column()
    prompt_tokens: Mapped[int | None] = mapped_column()
    response_tokens: Mapped[int | None] = mapped_column()
    total_tokens: Mapped[int | None] = mapped_column()
    is_success: Mapped[bool | None] = mapped_column()
    status_code: Mapped[int | None] = mapped_column() ![8](img/8.png) ![9](img/9.png)
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.now(UTC), onupdate=datetime.now(UTC)
    )

    conversation: Mapped["Conversation"] = relationship(
        "Conversation", back_populates="messages"
    )
```

![1](img/#co_integrating_databases_into_ai_services_CO2-1)

声明一个声明性基类，用于创建 SQLAlchemy 模型的 ORM 引擎。

![2](img/#co_integrating_databases_into_ai_services_CO2-2)

创建`Conversation`模型，指定表列、主键和二级索引。

![3](img/#co_integrating_databases_into_ai_services_CO2-3)

使用`mapped_column()`从`Mapped`提供的类型提示中推导列类型。

![4](img/#co_integrating_databases_into_ai_services_CO2-4)

如果您想根据模型类型更快地过滤会话，则对`model_type`进行索引。

![5](img/#co_integrating_databases_into_ai_services_CO2-5)

指定日期时间列的默认值和更新操作。

![6](img/#co_integrating_databases_into_ai_services_CO2-6)

指示如果通过`CASCADE DELETE`操作删除会话，则必须删除所有孤儿消息。

![7](img/#co_integrating_databases_into_ai_services_CO2-7)

创建`Message`模型，指定表列、主键、二级索引、表关系和外键。

![8](img/#co_integrating_databases_into_ai_services_CO2-9)

`messages`表将包含 LLM 提示和响应、使用令牌和成本，以及状态码和成功状态。

