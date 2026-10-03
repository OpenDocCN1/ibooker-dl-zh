
你从先前消息中提取的关键事实并不一定只来自用户档案信息。例如，假设当前用户问：“我抵达冰岛当天可以做些什么？”这时，检索他们告诉过你的最近一次旅行的相关信息会很有帮助。主要基于内容进行存储和检索信息的能力就是*语义记忆*。它与主要基于最近性的情景记忆不同。

在 `LangGraph` 中，要存储记忆，你可以执行以下操作：

```
store = PostgresStore(
    connection_string="postgresql://.../dbname",
)
trip_memories_ns = (user_id, "trip_memories") # namespace
memory = {"trip": {"from": "SEA", "to": "KEF", "depart_time": ...}}
memory_id = hash(json.dumps(memory, sort_keys=True))
store.put(trip_memories_ns, memory_id, memory)
```

然后，如果用户问起与旅行相关的问题，应用就可以在记忆存储中搜索他们最近的行程：

```
most_recent_trip = in_memory_store.search(trip_memories_ns)[-1]  
```

它也可以查看他们在一定时间范围内的所有近期行程，或者那些涉及冰岛的行程。什么内容相关，其判定逻辑由你决定。

## 示例

假设你正在开发一个帮助用户预订未来旅行的 AI 助手。长期记忆将帮助它记住相关上下文，例如座位偏好、偏好的航空公司或酒店连锁，以及已访问过的目的地。你可以在我们的 [GitHub 仓库](https://oreil.ly/jI8aG) 中找到完整示例。

### Mem0

对于长期记忆，你将使用 [Mem0](https://mem0.ai)，它实现了 Chhikara 等人于 2025 年在论文 [“Mem0：构建具有可扩展长期记忆的生产级 AI 代理”](https://arxiv.org/abs/2504.19413)中提出的低延迟记忆管理方法。`Mem0` 会从持续进行的对话中动态提取、整合并检索相关信息，形成基于图的记忆表示。它能够表示前面“解决方案”一节中讨论的全部四种记忆类型。

首先，你需要使用向量存储、嵌入模型、LLM 和数据库来配置 `Mem0`（我们稍后会讨论 `Mem0` 如何使用这些组件）：

```
# Mem0 Configuration
config = {
    "vector_store": {
        "provider": "chroma", # Database for semantic search
        "config": {
            "collection_name": "mem0_basic_example",
            "path": "/tmp/chroma_db", # Where to store memories
        }
    },
    "llm": {
        "provider": "openai", # OPENAI_API_KEY needs to be set in env
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0.1,
        }
    },
    "embedder": {
        "provider": "openai", # Convert text to vectors
        "config": {
            "model": "text-embedding-3-small"
        }
    }
    "history_db_path": os.path.join(temp_dir, "history.db")
}
 
memory = Memory.from_config(config)
```

然后，你可以调用 `memory.add` 将一个或多个对话轮次添加到 `Mem0`，如下所示：

```
# When you have a conversation
conversation = [
    {"role": "user", "content": """I'm looking to travel from Seattle to 
Reykjavik. What's the best way to do this?"""},
    {"role": "assistant", "content": """The best way to travel from Seattle to 
Reykjavik is to fly. The flight is about 10 hours long."""}
]
 
# add conversation to memory
memory.add(conversation, user_id="megan")
```

在将来与用户的对话中，你的应用可以查询长期记忆存储，并将检索到的信息作为额外上下文添加到模型请求中。要查询记忆存储，请调用 `memory.search`：

```
# When user asks something later
query = "What are my options to travel to Reykjavik?"
relevant_memories = memory.search(query=query, user_id="megan", limit=3)
 
# Returns: ["Interested in travel from Seattle to Reykjavik"]
```

你现在可以将搜索返回的记忆注入到 LLM 请求的请求上下文中。

就这么简单！如你所见，`Mem0`（`LangMem` 是另一个）等记忆框架抽象出了记忆检索和排序的具体细节。让我们来看看幕后发生了什么。

#### 添加记忆的工作原理

当你调用 `memory.add(messages, user_id="megan")` 时，长期记忆会与该用户关联，并跨会话持久化。你还可以指定其他类型的元数据：

```
client.add(messages, user_id="megan", metadata={"food": "vegan"})
```

`user_id` 不仅仅用于人类用户。如果你希望 AI 助手为不同用户提供一致的回答并跨会话记住细节，你也可以为它们分配用户 ID。

如果你只想在单个会话中保留消息，传入 `run_id`——这就是实现短期记忆的方式：

```
client.add(messages, user_id="megan", run_id="iceland-trip-planning")
```

当你使用 `memory.add` 将对话添加到记忆时，[`Mem0` 使用配置的 LLM](https://oreil.ly/eSxgh) 从对话中提取相关信息，识别重要的实体和关系，将新信息与现有数据进行比较，并识别/解决矛盾。这些提取出的关系信息存储在配置的数据库中。

例如，为了从对话中提取关键细节，`Mem0` 可能会这样提示 LLM：

```
# Internally converts to prompt for analysis
analysis_prompt = """
Analyze this conversation and extract important facts about the user:
 
{messages}
 
Extract only memorable, personal information. Ignore general knowledge.
"""
```

然后 LLM 会返回类似这样的内容：

```
User wants to travel from Seattle to Reykjavik.
```

预处理的主要目标是保留相关信息，如个人事实（“我喜欢去雷克雅未克旅行。”）、偏好（“我爱吃披萨。”）、计划（“我要去冰岛旅行。”）和关系（“Megan 是客户。”）。同时，LLM 会忽略无关信息，如闲聊（“今天天气不错。”）、常见问候（“你好，最近怎么样？”）以及容易重建的信息，如常识（“巴黎是法国的首都。”）和数学问题（“2 + 2 等于几？”）。

然后使用嵌入模型对相关信息进行嵌入：

```
# Convert memory text to numerical vector
embedding_response = openai_embeddings.create(
    model="text-embedding-3-small",
    input="User wants to travel from Seattle to Reykjavik"
)
 
# Gets back a 1536-dimensional vector like:
# [0.023, -0.891, 0.445, ..., 0.123]
```

该嵌入存储在配置的向量存储中（这里我们指定了 `ChromaDB`）。向量存储经过优化，可以快速找到相似的向量，从而实现高效的相似性查找。这些数据库中的向量搜索会检索相关的语义信息，以帮助回答未来的用户查询：

```
chroma_client.add(
    embeddings=[embedding_vector],
    documents=["User wants to travel from Seattle to Reykjavik"],
    metadatas=[{
        "user_id": "megan",
        "created_at": "2025-07-15T10:30:00Z",
        "memory_id": "uuid-12345",
        "category": "customer_info"
    }],
    ids=["memory_uuid_12345"]
)
```

处理记忆的工具通常将数据组织为图结构或存储在关系数据库中。这种方法有助于管理程序性记忆元素，如用户偏好和联系。将语义记忆与程序性记忆相结合，或将图数据库与向量数据库相结合，能产生良好的效果。具体来说，这种组合提供了合适的上下文，从而便于检索相关用户内容：

```
history_db.execute("""
    INSERT INTO memory_history 
    (memory_id, user_id, action, content, timestamp)
    VALUES (?, ?, 'ADD', ?, ?)
""", ["uuid-12345", "megan", memory_text, timestamp])
 
# Potentially Key-Value store for quick lookups
kv_store["user:megan:destination"] = "Reykjavik"
kv_store["user:megan:origin"] = "Seattle"
```

此时，框架已在长期记忆中存储了关于客户 Megan 的知识对象。但是，当你在未来的对话中调用 `memory.search(query, user_id="megan")` 时会发生什么呢？

#### 搜索记忆的工作原理

`memory.search` 操作通过在向量数据库中搜索与输入查询相似的近期消息来检索记忆。`Mem0` 还会识别被引用的实体，并在数据库中搜索相关的消息和用户偏好。通过这种方式，调用 `memory.search` 可以检索情景记忆、语义记忆和程序性记忆。这些结果根据与查询的相关性进行排序，你还可以指定元数据过滤器（有关详细信息，请参阅 [Mem0 文档](https://docs.mem0.ai/introduction)）。

让我们详细看一下。

首先，用户查询被转换为嵌入，如下面的代码片段所示：

```
query_embedding = openai_embeddings.create(
    model="text-embedding-3-small", 
    input="Where does Megan like to travel to?"
)
# Returns vector: [0.156, -0.234, 0.789, ..., 0.445]
```

将查询转换为嵌入后，`Mem0` 可以在向量存储中查询相似的知识对象：

```
# ChromaDB performs cosine similarity search
search_results = chroma_client.query(
    query_embeddings=[query_vector],
    where={"user_id": "megan"},  # Filter to user's memories only
    n_results=3,                 # Limit results
    include=["documents", "metadatas", "distances"]
)
 
# Returns memories ranked by similarity:
# [
#   {"document": "wants to travel to Reykjavik", "distance": 0.15},
#   {"document": "Iceland", "distance": 0.23},
#   {"document": "Blue Lagoon", "distance": 0.31}
# ]
```

程序性记忆可以存储在 KV 数据库（如 `Redis`）、图数据库（如 `Neo4j`）或关系数据库（如 `Postgres` 或 `SQLite`）中。根据你配置 `Mem0` 的方式，它会使用适当的查询机制：

```
# Key-Value Database: Fast exact lookups
kv_store["user:megan:name"] → "Megan"
kv_store["user:megan:destination"] → "Reykjavik"
kv_store["user:megan:origin"] → "Seattle"
# Graph Database: Relationship queries
graph_db.query(
   "MATCH (megan:Person)-[:WORKS_AT]->(company:Company) RETURN company")
# → Finds Megan's workplace connections
# Relational Database: Audit trail
history_db.query(
   "SELECT * FROM memory_history WHERE user_id='megan' ORDER BY timestamp")
# → Shows how Megan's memories evolved over time
```

你可以使用 *过滤器* 来控制返回的记忆——过滤器是你对记忆附加的额外条件。例如，你可以设置一个过滤器，使模型在为用户预订航班时提取与食物偏好相关的记忆：

```
query = "Book a flight from SEA to KEF on June 6"
filters = {
    "AND": [
        {
            "categories": {
                "contains": "food_preferences"
            }
        }
    ]
}
client.search(query, user_id="megan", filters=filters)
```

这就是 `Mem0` 如何通过一个简单的接口支持全部四种类型的记忆，该接口抽象掉了许多查询处理和存储操作。

## 注意事项

像 `Mem0` 这样的长期记忆框架结合了多种基础设施组件，例如向量数据库、关系数据库以及各种 LLM 配置。对于生产环境用例，我们强烈建议将其部署为微服务，而不是直接使用 Python 模块。可以从 [Mem0 服务器](https://oreil.ly/FaCvD) 入手。

为了避免降低用户体验，你可以在后台线程中将消息和记忆写入历史记录。

根据你正在构建的应用程序，你可能需要跟踪四种记忆中的一种或多种。工作记忆对于维持当前对话上下文很重要。情景记忆用于查找特定交互中的消息。程序性记忆对于学习用户模式和偏好很重要。语义记忆对于管理从先前对话中提取的事实和知识至关重要。工作记忆在聊天机器人中至关重要，情景记忆在涉及多步骤工作流的应用和任务中至关重要，程序性记忆对于个性化至关重要，而语义记忆对于处理大型文档至关重要。

一般来说，你应该使用语义记忆而非情景记忆。情景记忆会按原样存储所有消息，并在运行时依赖查找相关消息；虽然存储成本低廉，但检索相关消息会带来相当大的延迟。由缓存未命中导致的记忆错误也很难排查，因此与其按原样存储消息，不如从消息中提取值得记住的信息，只存储这些记忆，这样数量更少，并且可以更确定性地搜索。

## 参考资料

[记忆类型](https://arxiv.org/abs/2309.02427)由 Sumers 等人（2023）提出。Wang 等人（2023）提出了一种潜在空间方法，为[增强 LLM](https://arxiv.org/abs/2306.07174)添加长期记忆。他们在保持原始 LLM 不变的情况下，使用一个自适应残差侧网络作为记忆核心检索器。在撰写本文时，记忆管理的前沿是由 Chhikara 等人（2025）开发的[可扩展 Mem0 方法](https://arxiv.org/abs/2504.19413)，该方法在带标签的图中动态提取和更新信息。

[RevisionDojo](https://oreil.ly/EYNuW) 在其个性化辅导产品中使用 `Mem0` 来引用先前的工作、定制解释并动态调整难度级别。

## 总结
