
# Python 代码摄取与 RAG 检索

## 1. 导入依赖与连接 Milvus

```py
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

from transformers import AutoTokenizer
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from tqdm.auto import tqdm
from uuid import uuid4

connections.connect("default", host="localhost", port="19530")    #1
```

#1 连接到 Milvus

## 2. 定义数据摄取类

```py
class PythonCodeIngestion:
    def __init__(
        self,
        collection,
        python_code=None,
        embedder=None,
        tokenizer=None,
        text_splitter=None,
        batch_limit=100,
    ):
        self.collection = collection
        self.python_code = python_code or load_dataset(
            "iamtarun/python_code_instructions_18k_alpaca",
            split="train",
        )
        self.embedder = embedder or SentenceTransformer(
            "krlvi/sentence-t5-base-nlpl-code_search_net"
        )
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            "Deci/DeciCoder-1b"
        )
        self.text_splitter = (
            text_splitter
            or RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=20,
                length_function=self.token_length,
                separators=["\n\n", "\n", " ", ""],
            )
        )
        self.batch_limit = batch_limit

    def token_length(self, text):
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    def get_metadata(self, page):
        return {
            "instruction": page["instruction"],
            "input": page["input"],
            "output": page["output"],
        }

    def split_texts_and_metadatas(self, page):
        basic_metadata = self.get_metadata(page)
        prompts = self.text_splitter.split_text(page["prompt"])
        metadatas = [
            {"chunk": j, "prompt": prompt, **basic_metadata}
            for j, prompt in enumerate(prompts)
        ]
        return prompts, metadatas

    def upload_batch(self, texts, metadatas):
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeddings = self.embedder.encode(texts)
        self.collection.insert([ids, embeddings, metadatas])

    def batch_upload(self):
        batch_texts = []
        batch_metadatas = []

        for page in tqdm(self.python_code):
            texts, metadatas = self.split_texts_and_metadatas(page)

            batch_texts.extend(texts)
            batch_metadatas.extend(metadatas)

            if len(batch_texts) >= self.batch_limit:
                self.upload_batch(batch_texts, batch_metadatas)
                batch_texts = []
                batch_metadatas = []

        if len(batch_texts) > 0:
            self.upload_batch(batch_texts, batch_metadatas)

        self.collection.flush()
```

现在我们已经创建了我们的摄取类，我们可以继续进行管道操作。首先，如果这是我们第一次运行它，我们需要创建我们的集合。集合在其他数据库中就像一个表，或者在 Pinecone 中就像一个索引。我们将定义我们的模式，它只是一个 ID 字段、我们的嵌入字段和一个元数据字段，该字段包含自由形式的 JSON。一旦设置好，我们将使用我们的`PythonCodeIngestion`类上传我们的数据。

接下来，我们需要创建我们的搜索索引。我们将使用的索引类型是`IVF_FLAT`，这是 Milvus 中最基本的索引，将嵌入空间分割成`nlist`数量的聚类。这通过首先将我们的搜索嵌入与聚类中心进行比较，然后与它最近的聚类中的嵌入进行比较来加速相似性搜索。我们还将使用`L2`作为我们的度量类型，这意味着我们将使用欧几里得距离。这些是常见的设置，但对我们来说不需要任何特殊设置。Milvus 在构建索引时支持更多的选项，我们鼓励您查看他们的文档：

```py
if __name__ == "__main__":
    collection_name = "milvus_llm_example"
    dim = 768

    if utility.has_collection(collection_name):     #1
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(
            name="ids",
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=False,
            max_length=36,
        ),
        FieldSchema(
            name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim
        ),
        FieldSchema(name="metadata", dtype=DataType.JSON),
    ]

    schema = CollectionSchema(
        fields, f"{collection_name} is collection of python code prompts"
    )

    print(f"Create collection {collection_name}")
    collection = Collection(collection_name, schema)

    collection = Collection(collection_name)    #2
    print(collection.num_entities)

    python_code_ingestion = PythonCodeIngestion(collection)    #3
    python_code_ingestion.batch_upload()
    print(collection.num_entities)

    search_index = {                #4
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},      #5
    }
    collection.create_index("embeddings", search_index)
```

#1 如果不存在则创建集合

#2 连接到集合并显示其大小

#3 数据摄取并显示数据摄取后的统计数据

#4 构建搜索索引

#5 聚类的数量

## 3. 测试查询

现在我们已经设置好了一切，我们可以继续下一步。但在那之前，让我们通过运行一个查询来测试它。我们想要确保我们的数据和索引给我们提供了合理的搜索结果。使用 Milvus，我们首先将集合加载到内存中，并使用我们的嵌入器将我们的查询转换为嵌入。接下来，我们将定义一些搜索参数。再次，`L2`代表欧几里得距离，而`nprobe`参数表示要搜索的簇数量。在我们的情况下，在 128 个我们设置的簇中，我们将搜索与我们的查询嵌入最近的 10 个簇。最后，在实际搜索中，我们将结果限制为三个最佳匹配，并返回与我们的查询一起的元数据字段：

```py
    collection.load()      #1

    query = (             #2
        "Construct a neural network model in Python to classify "
        "the MNIST data set correctly."
    )
    search_embedding = python_code_ingestion.embedder.encode(query)
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},     #3
    }
    results = collection.search(
        [search_embedding],
        "embeddings",
        search_params,
        limit=3,
        output_fields=["metadata"],
    )
    for hits in results:
        for hit in hits:
            print(hit.distance)
            print(hit.entity.metadata["instruction"])
```

#1 在进行搜索之前，你需要将数据加载到内存中。

#2 进行查询

#3 搜索的簇数量

你可以看到，对于我们的查询，搜索结果正在返回来自我们数据集的强候选者：

```py
    # 0.7066953182220459
    # Create a neural network in Python to identify
    # hand-written digits from the MNIST dataset.
    # 0.7366453409194946
    # Create a question-answering system using Python
    # and Natural Language Processing.
    # 0.7389795184135437
    # Write a Python program to create a neural network model that can
    # classify handwritten digits (0-9) with at least 95% accuracy.
```

## 4. 使用 RAG

现在我们已经设置了我们的 VectorDB 并加载了数据，让我们更新我们的 API 以从我们的 RAG 系统中检索结果并将上下文注入到我们的提示中。

在本节中，我们将更新列表 10.1 以包含我们的检索代码。在列表 10.3 中，我们不会重复之前所做的所有事情，考虑到时间和空间，我们只需展示需要添加的新部分。在伴随本书的仓库中，如果你在理解哪些部分应该放在哪里时遇到困难，你将能够找到将一切组合在一起的代码。首先，在脚本接近顶部的地方，我们需要添加我们的导入，连接到我们的 Milvus 服务，并加载我们的嵌入模型。

### 列表 10.3 将 RAG 添加到我们的 API

```py
from contextlib import asynccontextmanager

from pymilvus import (
    connections,
    Collection,
)
from sentence_transformers import SentenceTransformer
connections.connect("default", host="localhost", port="19530")    #1

collection_name = "milvus_llm_example"
collection = Collection(collection_name)

embedder = SentenceTransformer(                  #2
    "krlvi/sentence-t5-base-nlpl-code_search_net"
)
embedder = embedder.to(device)
```

#1 连接到 Milvus

#2 加载我们的嵌入模型

接下来，我们将添加一些便利函数，包括一个标记计数器和 FastAPI 生命周期，以确保我们加载和释放我们的 Milvus 集合从内存。由于我们添加了一个生命周期，请确保更新 FastAPI 调用：

```py
def token_length(text):
    tokens = tokenizer([text], return_tensors="pt")
    return tokens["input_ids"].shape[1]

@asynccontextmanager
async def lifespan(app: FastAPI):
    collection.load()                #1
    yield
    collection.release()            #2

app = FastAPI(lifespan=lifespan)      #3
```

#1 启动时加载集合

#2 关闭时从内存中释放集合

#3 运行 FastAPI

现在我们已经设置好了一切，我们可以进入下一步——运行查询并更新我们的`generate`端点的提示。第一部分应该看起来很熟悉，因为我们刚刚做过。我们将编码用户的提示并在我们的集合中搜索最近的邻居。我们使用与之前相同的所有搜索参数，除了一个。我们将限制从`3`增加到`5`，以便可能将更多示例添加到我们的提示中。接下来，我们将这些结果格式化为几个提示示例数据集。然后我们创建指令提示并格式化用户的输入。

我们几乎到了可以结合我们的指令、示例和用户提示的阶段；然而，我们需要确保我们的示例不会占用太多空间。使用一个利用我们的标记计数器的`for`循环，我们将过滤掉任何不符合我们上下文窗口的示例。有了这个，我们现在可以结合所有内容，为我们的 DeciCoder 模型创建最终的提示。
