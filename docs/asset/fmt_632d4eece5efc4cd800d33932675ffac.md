
```py
load_dotenv()      #1
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("No API key found. Please check your .env file.")
client = OpenAI(api_key=api_key)     #1            

def get_embedding(text, model="text-embedding-ada-002"):     #2
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text],
              model=model).data[0].embedding                #2

# Sample documents (omitted)

embeddings = [get_embedding(doc) for doc in documents]    #3
print(embeddings_array.shape)

embeddings_array = np.array(embeddings)    #4

pca = PCA(n_components=3)   #5
reduced_embeddings = pca.fit_transform(embeddings_array)
```

#1 将所有项目用逗号和空格连接起来。

#2 使用 OpenAI 客户端创建嵌入

#3 为每个 1536 维度的文档生成嵌入

#4 将嵌入转换为 NumPy 数组以进行 PCA

#5 将维度减少到 3 以进行绘图

当使用 OpenAI 模型对文档进行嵌入时，它将文本转换为一个 1536 维度的向量。我们无法可视化这么多维度，因此我们使用主成分分析（PCA）这种降维技术，将 1536 维度的向量转换为 3 维。

图 8.6 显示了在 VS Code 中运行文件生成的输出。通过将嵌入减少到 3D，我们可以绘制输出以显示语义相似的文档是如何分组的。

![figure](img/8-6.png)

##### 图 8.6 3D 嵌入，显示相似语义文档是如何分组的

你可以选择使用哪种嵌入模型或服务。OpenAI 的嵌入模型被认为是通用语义相似度方面最好的。这使得这些模型成为大多数记忆和检索应用的标准。通过我们对于如何使用嵌入将文本向量化并存储在向量数据库中的理解，我们可以在下一节中继续一个更实际的例子。

### 8.3.4 从 Chroma 查询文档嵌入

我们可以将所有部分结合起来，通过使用一个名为 Chroma DB 的本地向量数据库来查看一个完整的示例。存在许多向量数据库选项，但 Chroma DB 是一个优秀的本地向量存储，适用于开发或小规模项目。还有许多更健壮的选项，你可以在以后考虑。

列表 8.5 展示了`document_query_chromadb.py`文件中的新和相关的代码部分。注意，结果是根据距离评分，而不是根据相似度评分。余弦距离由以下公式确定：

余弦距离(A,B) = 1 – 余弦相似度(A,B)

这意味着余弦距离的范围从 0（最相似）到 2（语义上相反）。

##### 列表 8.5 `document_query_chromadb.py`（相关代码部分）

```py
embeddings = [get_embedding(doc) for doc in documents]     #1
ids = [f"id{i}" for i in range(len(documents))]           #1

chroma_client = chromadb.Client()               #2
collection = chroma_client.create_collection(
                       name="documents")       #2
collection.add(     #3
    embeddings=embeddings,
    documents=documents,
    ids=ids
)

def query_chromadb(query, top_n=2):      #4
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n
    )
    return [(id, score, text) for id, score, text in
            zip(results['ids'][0],
                results['distances'][0], 
                results['documents'][0])]

while True:     #5
    query = input("Enter a search query (or 'exit' to stop): ")
    if query.lower() == 'exit':
        break
    top_n = int(input("How many top matches do you want to see? "))
    search_results = query_chromadb(query, top_n)

    print("Top Matched Documents:")
    for id, score, text in search_results:
        print(f"""
ID:{id} TEXT: {text} SCORE: {round(score, 2)}
""")    #5

    print("\n")
###Output
Enter a search query (or 'exit' to stop): dogs are lazy
How many top matches do you want to see? 3
Top Matched Documents:
ID:id7 TEXT: The dog is lazy but the brown fox is quick! SCORE: 0.24
ID:id5 TEXT: The brown fox is quick and the blue dog is lazy! SCORE: 0.28
ID:id2 TEXT: The quick brown fox jumps over the lazy dog. SCORE: 0.29
```

#1 为每个文档生成嵌入并分配一个 ID

#2 创建 Chroma DB 客户端和集合

#3 将文档嵌入添加到集合中

#4 查询数据存储并返回最相关的 n 个文档

#5 用户输入循环和输出相关文档/分数的输入循环

如前文场景所示，你现在可以使用语义意义而不是仅仅关键词或短语来查询文档。这些场景现在应该为理解检索模式在底层是如何工作的提供背景。在下一节中，我们将看到如何使用 LangChain 来应用检索模式。

## 8.4 使用 LangChain 构建 RAG

LangChain 最初是一个专注于抽象多个数据源和向量存储检索模式的开源项目。它已经演变得更多，但基础层面上，它仍然为检索实现提供了优秀的选项。

图 8.7 展示了 LangChain 中的一个流程图，该图标识了存储文档以供检索的过程。这些相同的步骤可以全部或部分复制以实现记忆检索。文档检索和记忆检索之间的关键区别在于来源以及内容是如何被转换的。

![figure](img/8-7.png)

##### 图 8.7 显示了存储文档以供后续检索的加载、转换、嵌入和存储步骤

我们将探讨如何使用 LangChain 实现这些步骤，并理解伴随此实现的细微差别和细节。在下一节中，我们将首先使用 LangChain 分割和加载文档。

### 8.4.1 使用 LangChain 分割和加载文档

检索机制通过添加与请求相关的特定信息来增强给定提示的上下文。例如，你可能需要关于本地文档的详细信息。在早期的语言模型中，由于标记限制，将整个文档作为提示的一部分提交不是一个选项。

今天，我们可以将整个文档提交给许多商业 LLM，如 GPT-4 Turbo，作为提示请求的一部分。然而，结果可能不会更好，并且可能会因为标记数量的增加而成本更高。因此，更好的选择是将文档分割，并使用相关部分请求上下文——这正是 RAG 和记忆所做的事情。

分割文档对于将内容分解成语义上和具体相关的部分至关重要。图 8.8 显示了如何分解包含老母鸡童谣的 HTML 文档。通常，将文档分割成上下文语义块需要仔细考虑。

![图](img/8-8.png)

##### 图 8.8 理想情况下文档如何分割成块以获得更好的语义和上下文意义

理想情况下，当我们将文档分割成块时，它们应按相关性和语义意义分解。虽然 LLM 或代理可以帮助我们做这件事，但我们将查看 LangChain 中当前的工具包选项，用于分割文档。在本章的后面部分，我们将查看一个可以帮助我们在嵌入内容时进行语义划分的语义函数。

对于下一个练习，在 VS Code 中打开`langchain_load_splitting.py`，如列表 8.6 所示。此代码显示了我们在上一节列表 8.5 中留下的地方。这次我们不是使用样本文档，而是加载这次的老母鸡童谣。

##### 列表 8.6 `langchain_load_splitting.py`（部分和输出）

```py
From langchain_community.document_loaders 
                     ↪ import UnstructuredHTMLLoader    #1
from langchain.text_splitter import RecursiveCharacterTextSplitter
#previous code

loader = UnstructuredHTMLLoader(
                   "sample_documents/mother_goose.html")   #2
data = loader.load    #3

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=25,     #4
    length_function=len,
    add_start_index=True,
)
documents = text_splitter.split_documents(data)

documents = [doc.page_content 
                ↪ for doc in documents] [100:350]   #5

embeddings = [get_embedding(doc) for doc in documents]     #6
ids = [f"id{i}" for i in range(len(documents))]
###Output
Enter a search query (or 'exit' to stop): **who kissed the girls and made** 
**them cry?**
How many top matches do you want to see? 3
Top Matched Documents:
ID:id233 TEXT: And chid her daughter,
        And kissed my sister instead of me. SCORE: 0.4…
```

#1 新的 LangChain 导入

#2 将文档作为 HTML 加载

#3 加载文档

#4 将文档分割成 100 个字符长、25 个字符重叠的文本块

#5 仅嵌入 250 个块，这更便宜且更快

#6 返回每个文档的嵌入

注意在列表 8.6 中，HTML 文档被分割成 100 个字符的块，并且有 25 个字符的重叠。这种重叠允许文档的部分不会切断特定的想法。我们选择这个分割器进行这个练习，因为它易于使用、设置和理解。

好吧，在 VS Code 中运行`langchain_load_splitting.py`文件（F5）。输入一个查询，看看你得到什么结果。列表 8.6 中的输出显示了给定特定示例的良好结果。请记住，我们只嵌入 250 个文档块以降低成本并使练习简短。当然，你总是可以尝试嵌入整个文档或使用较小的输入文档示例。

构建适当的检索最关键的因素可能是文档分割的过程。你可以使用多种方法来分割文档，包括多种并发方法。超过一种方法会通过分割文档为同一文档的多个嵌入视图。在下一节中，我们将检查一种更通用的文档分割技术，使用标记和标记化。

### 8.4.2 使用 LangChain 按标记分割文档

*分词* 是将文本分割成单词标记的过程。一个单词标记代表文本中的一个简洁元素，一个标记可以是一个像 *hold* 这样的单词，甚至是一个像左花括号 ({) 这样的符号，具体取决于什么是有意义的。

使用分词技术分割文档为语言模型如何解释文本提供了一个更好的基础，以及语义相似性。分词技术还允许移除无关字符，如空白字符，使文档的相似性匹配更加相关，并通常提供更好的结果。

对于下一个代码练习，请在 VS Code 中打开 `langchain_token_splitting.py` 文件，如图 8.7 所示。现在我们使用分词技术将文档分割成大小不等的部分。这种不均匀的大小是由于原始文档中大量空白区域造成的。

##### 列表 8.7 `langchain_token_splitting.py`（相关新代码）

```py
loader = UnstructuredHTMLLoader("sample_documents/mother_goose.html")
data = loader.load()
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=50, chunk_overlap=10      #1
)

documents = text_splitter.split_documents(data)
documents = [doc for doc in documents][8:94]      #2

db = Chroma.from_documents(documents, OpenAIEmbeddings())

def query_documents(query, top_n=2):
    docs = db.similarity_search(query, top_n)      #3
    return docs
###Output
Created a chunk of size 68, 
which is longer than the specified 50
Created a chunk of size 67, 
which is longer than the specified 50     #4
Enter a search query (or 'exit' to stop): 
                     who kissed the girls and made them cry?
How many top matches do you want to see? 3
Top Matched Documents:
Document 1: GEORGY PORGY

        Georgy Porgy, pudding and pie,
        Kissed the girls and made them cry.
```

#1 更新为 50 个标记和 10 个标记的重叠

#2 仅选择包含押韵的文档

#3 使用数据库的相似性搜索

#4 由于空白字符而分割成不规则大小的块

在 VS Code 中运行 `langchain_token_splitting.py` 代码（按 F5）。您可以使用上次使用的查询或您自己的查询。注意，结果比上一个练习明显更好。然而，结果仍然可疑，因为查询使用了几个相同顺序的相似词。

一个更好的测试是尝试一个语义上相似的短语，但使用不同的词，并检查结果。代码仍在运行时，输入一个新的查询短语：`为什么` `女孩们` `在哭泣？` 列表 8.8 显示了执行该查询的结果。如果您自己运行此示例并向下滚动输出，您将看到乔治·波吉出现在返回的第二或第三份文档中。

##### 列表 8.8 查询：谁让女孩们哭泣？

```py
Enter a search query (or 'exit' to stop): Who made the girls cry?
How many top matches do you want to see? 3
Top Matched Documents:
Document 1: WILLY, WILLY

        Willy, Willy Wilkin…
```

这个练习展示了如何使用各种检索方法来返回语义上的文档。在这个基础上，我们可以看到 RAG 如何应用于知识和记忆系统。下一节将讨论 RAG 在应用于代理和代理系统知识时的应用。

## 8.5 将 RAG 应用于构建代理知识

代理中的知识包括使用 RAG 在非结构化文档中进行语义搜索。这些文档可以是 PDF 文件、Microsoft Word 文档以及所有文本，包括代码。代理知识还包括使用非结构化文档进行问答、参考查找、信息增强和其他未来模式。

Nexus，是与本书一起开发并在上一章中介绍的代理平台，为代理提供了完整的知识和记忆系统。在本节中，我们将揭示知识系统是如何工作的。
