
## 混合检索示例

以下示例演示了如何结合 FAISS 和 Whoosh 实现混合检索：

```py
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT
import faiss
import numpy as np

# Initialize FAISS index for dense retrieval
dimension = 128
dense_index = faiss.IndexFlatL2(dimension)

# Simulate sparse retrieval with Whoosh
schema = Schema(content=TEXT(stored=True))
ix = create_in("index", schema)
writer = ix.writer()

writer.add_document(content="This is a test document.")
writer.add_document(content="Another document for retrieval.")
writer.commit()

# Query for dense and sparse retrieval
def retrieve_dense(query_vector):
    return dense_index.search(np.array([query_vector]), k=5)

def retrieve_sparse(query):
    searcher = ix.searcher()
    results = searcher.find("content", query)
    return [hit['content'] for hit in results]

query_vector = np.random.rand(1, dimension).astype('float32')
sparse_query = "document"

# Perform combined retrieval
dense_results = retrieve_dense(query_vector)
sparse_results = retrieve_sparse(sparse_query)

# Combine dense and sparse results
combined_results = dense_results + sparse_results
print("Combined results:", combined_results)
```

在这个例子中，`FAISS`处理基于密集向量的检索，而`Whoosh`处理基于关键词的稀疏搜索。然后将结果合并，提供语义和精确匹配检索，这可以提高系统响应的整体准确性和完整性。

## 缓存嵌入

对于频繁查询的数据，而不是重新计算嵌入，可以使用*嵌入缓存*让系统存储嵌入，并在后续查询中重用它们。如果查询的嵌入已经存储在缓存中，系统将检索它们；否则，它将计算嵌入并将它们存储以供将来使用。这减少了重新处理相同数据的需求，显著降低了响应时间并提高了效率。

这里是一个如何实现嵌入缓存（embedding caching）的例子：

```py
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('MiniLM')

# Check if embeddings are cached
def get_embeddings(query):
    cache_file = "embedding_cache.pkl"

    # Check if cache exists
    try:
        embeddings_cache = joblib.load(cache_file)
    except FileNotFoundError:
        embeddings_cache = {}

    # If query is not in cache, compute and cache the embeddings
    if query not in embeddings_cache:
        embedding = model.encode([query])
        embeddings_cache[query] = embedding
        joblib.dump(embeddings_cache, cache_file)  # Save cache to disk

    return embeddings_cache[query]

# Query
query = "What is the capital of France?"
embedding = get_embeddings(query)
print("Embedding for the query:", embedding)
```

## 键值（Key-Value）缓存

*键值（KV）缓存*与嵌入缓存的工作方式类似。它存储键值对的结果，其中键是一个查询或中间结果，值是对应的响应或计算结果。这允许系统检索预先计算的结果，而不是每次处理重复查询时都重新计算。键值缓存加快了检索和生成的速度，尤其是在大规模、高流量系统中。

在 RAG 系统中，键值（KV）缓存通常在检索阶段应用，以加快查询-响应周期。在生成阶段，模型可能使用缓存的文档和响应的版本或部分来构建其最终输出。

让我们看看如何在 Python 中实现它：

```py
import redis
import numpy as np
from sentence_transformers import SentenceTransformer

# Step 1\. Initialize Redis client
r = redis.Redis(host='localhost', port=6379, db=0)

# Step 2\. Initialize sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Step 3\. Function to get embeddings and cache them in Redis
def get_embeddings_from_cache_or_compute(query):
    cache_key = f"embedding:{query}"  # Key to store the query embeddings

    # Check if the embedding exists in the cache
    cached_embedding = r.get(cache_key)

    if cached_embedding:
        print("Cache hit, returning cached embedding")
        return np.frombuffer(cached_embedding, dtype=np.float32)
    else:
        print("Cache miss, computing and storing embedding")
        embedding = model.encode([query])
        r.set(cache_key, embedding.tobytes())  # Store embedding in Redis
        return embedding

# Step 4\. Query the system
query = "What is the capital of France?"
embedding = get_embeddings_from_cache_or_compute(query)
print("Embedding:", embedding)
```

在第 1 步中，你连接到本地运行的`Redis`实例，以存储用于快速查找的关键值对。

然后第 3 步规定，当接收到查询时，代码会检查该查询的嵌入是否已经缓存到`Redis`中，通过检查键（`embedding:<query>`）。如果缓存包含嵌入（称为*缓存命中*），则直接检索并返回它们。如果没有（称为*缓存未命中*），则使用`SentenceTransformer`计算嵌入，然后存储。嵌入使用`tobytes()`存储在`Redis`中，以确保它们可以以相同的格式检索。

通过减少重新计算嵌入或模型响应的需求，键值（KV）缓存可以帮助降低计算成本，并减少检索和生成组件的负担，确保系统即使在重负载下也能保持响应。

## 可扩展性和可重用性

可扩展性和可重用性对于处理高流量系统至关重要。在大规模环境中，有效地扩展你的基础设施的能力是至关重要的。*分布式推理编排*允许系统在流量增加时将负载分配到多个节点，每个节点处理整体请求的一部分。这减少了任何单台机器过载的可能性。

`Kubernetes`通常用于通过自动化任务分配和根据需要调整资源来管理扩展过程。

可重用组件使您更容易扩展和管理您的管道。由于它们不需要进行重大修改，因此可以快速在不同服务或项目中复制。这在需要不断更新和迭代的环境中尤为重要。`ZenML`和类似的编排工具允许您创建可重用管道，您可以在不破坏整个系统的情况下对其进行修改或扩展。随着您构建新的模型或添加新的任务，您可以重用现有组件以保持一致性并减少开发时间。

分布式推理编排和可重用组件协同工作，以确保您的系统既可扩展又易于维护。当流量激增或出现新的用例时，了解您可以依赖现有基础设施来处理需求是很重要的。这使得整个系统在面对新挑战时更具弹性和敏捷性。

可扩展性和可重用性不仅仅是锦上添花的功能，对于高流量 LLM 系统来说是必需的功能。分布式推理编排确保您的系统可以扩展以满足需求，而可重用组件使您更容易随着时间的推移维护和扩展系统。共同作用，它们允许高效有效地处理大规模 LLM 部署。

## 结论

正确的堆栈将取决于您的项目目标。如果您需要灵活性和有技术资源来管理设置，开源工具非常出色。对于优先考虑速度和简单性的团队，托管服务是完美的选择。在承诺使用某个堆栈之前，仔细评估您的需求，因为正确的选择将节省时间、提高性能，并帮助您更有效地部署。
