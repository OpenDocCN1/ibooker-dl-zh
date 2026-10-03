
```py
from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Two prompts
physics_template = """You are a very smart physics professor. You are great at 
 answering questions about physics in a concise and easy-to-understand manner. 
 When you don't know the answer to a question, you admit that you don't know.

Here is a question:
{query}"""

math_template = """You are a very good mathematician. You are great at answering 
 math questions. You are so good because you are able to break down hard 
 problems into their component parts, answer the component parts, and then 
 put them together to answer the broader question.

Here is a question:
{query}"""

# Embed prompts
embeddings = OpenAIEmbeddings()
prompt_templates = [physics_template, math_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)

# Route question to prompt
@chain
def prompt_router(query):
    # Embed question
    query_embedding = embeddings.embed_query(query)
    # Compute similarity
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    # Pick the prompt most similar to the input question
    most_similar = prompt_templates[similarity.argmax()]
    return PromptTemplate.from_template(most_similar)

semantic_router = (
    prompt_router
    | ChatOpenAI()
    | StrOutputParser()
)

print(semantic_router.invoke("What's a black hole"))
```

*JavaScript*

```py
import {cosineSimilarity} from '@langchain/core/utils/math'
import {ChatOpenAI, OpenAIEmbeddings} from '@langchain/openai'
import {PromptTemplate} from '@langchain/core/prompts'
import {RunnableLambda} from '@langchain/core/runnables';

const physicsTemplate = `You are a very smart physics professor. You are great 
 at answering questions about physics in a concise and easy-to-understand 
 manner. When you don't know the answer to a question, you admit that you 
 don't know.

Here is a question:
{query}`

const mathTemplate = `You are a very good mathematician. You are great at 
 answering math questions. You are so good because you are able to break down 
 hard problems into their component parts, answer the component parts, and 
 then put them together to answer the broader question.

Here is a question:
{query}`

const embeddings = new OpenAIEmbeddings()

const promptTemplates = [physicsTemplate, mathTemplate]
const promptEmbeddings = await embeddings.embedDocuments(promptTemplates)

const promptRouter = RunnableLambda.from(query => {
  // Embed question
  const queryEmbedding = await embeddings.embedQuery(query)
  // Compute similarity
  const similarities = cosineSimilarity([queryEmbedding], promptEmbeddings)[0]
  // Pick the prompt most similar to the input question
  const mostSimilar = similarities[0] > similarities[1] 
    ? promptTemplates[0] 
    : promptTemplates[1]
  return PromptTemplate.fromTemplate(mostSimilar)
})

const semanticRouter = promptRouter.pipe(new ChatOpenAI())

await semanticRouter.invoke("What's a black hole")
```

现在您已经看到了如何将用户的查询路由到相关的数据源，让我们讨论构建健壮 RAG 系统时的第三个主要问题：“我们如何将自然语言转换为目标数据源的查询语言？”

# 查询构建

如前所述，RAG 是一种有效的策略，可以根据查询从基于向量的存储中嵌入和检索相关的非结构化数据。但大多数可用于生产应用程序的数据是结构化的，通常存储在关系数据库中。此外，嵌入在向量存储中的非结构化数据还包含具有重要信息的结构化元数据。

*查询构建*是将自然语言查询转换为与您交互的数据库或数据源的查询语言的过程。参见图 3-10。

![数据类型的示意图  自动生成的描述](img/lelc_0310.png)

###### 图 3-10\. 各种数据源的查询语言示意图

例如，考虑以下查询：*1980 年关于外星人的电影是什么？*这个问题包含一个可以通过嵌入（*外星人*）检索的非结构化主题，但它还包含潜在的有序组件（*年份 == 1980*）。

以下章节将更深入地探讨查询构建的各种形式。

## 文本到元数据过滤器

大多数向量存储都提供了基于元数据限制向量搜索的能力。在嵌入过程中，我们可以将元数据键值对附加到索引中的向量上，然后在查询索引时指定过滤器表达式。

LangChain 提供了一个 `SelfQueryRetriever`，它抽象了这种逻辑，并使得将自然语言查询转换为针对各种数据源的格式化查询变得更加容易。自查询利用一个 LLM 根据用户的查询和预定义的元数据模式提取和执行相关的元数据过滤器：

*Python*

```py
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI

fields = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="A 1-10 rating for the movie", type="float"
    ),
]
description = "Brief summary of a movie"

llm = ChatOpenAI(temperature=0)

retriever = SelfQueryRetriever.from_llm(
    llm, db, description, fields,
)

print(retriever.invoke(
    "What's a highly rated (above 8.5) science fiction film?"))
```

*JavaScript*

```py
import { ChatOpenAI } from "@langchain/openai";
import { SelfQueryRetriever } from "langchain/retrievers/self_query";
import { FunctionalTranslator } from "@langchain/core/structured_query";

/**
 * First, we define the attributes we want to be able to query on.
 * in this case, we want to be able to query on the genre, year, director, 
 * rating, and length of the movie.
 * We also provide a description of each attribute and the type of the attribute.
 * This is used to generate the query prompts.
 */
const fields = [
  {
    name: "genre",
    description: "The genre of the movie",
    type: "string or array of strings",
  },
  {
    name: "year",
    description: "The year the movie was released",
    type: "number",
  },
  {
    name: "director",
    description: "The director of the movie",
    type: "string",
  },
  {
    name: "rating",
    description: "The rating of the movie (1-10)",
    type: "number",
  },
  {
    name: "length",
    description: "The length of the movie in minutes",
    type: "number",
  },
];
const description = "Brief summary of a movie";

const llm = new ChatOpenAI();
const attributeInfos = fields.map((field) => new AttributeInfo(field.name,  
  field.description, field.type));

const selfQueryRetriever = SelfQueryRetriever.fromLLM({
  llm,
  db,
  description,
  attributeInfo: attributeInfos,
  /**
 * We need to use a translator that translates the queries into a
 * filter format that the vector store can understand. LangChain provides one 
 * here.
 */
  structuredQueryTranslator: new FunctionalTranslator(),
});

await selfQueryRetriever.invoke(
  "What's a highly rated (above 8.5) science fiction film?"
);
```

这导致检索器将用户查询分成：

+   首先应用于每个文档元数据的过滤器

+   用于在文档上进行语义搜索的查询

要做到这一点，我们必须描述我们的文档元数据包含哪些字段；该描述将包含在提示中。检索器接下来将执行以下操作：

1.  将查询生成提示发送到 LLM。

1.  解析来自 LLM 输出的元数据过滤器和重写的搜索查询。

1.  将 LLM 生成的元数据过滤器转换为适合我们向量存储的格式。

1.  对向量存储执行相似度搜索，仅匹配元数据通过生成的过滤器的文档。

## 文本到 SQL

SQL 和关系数据库是结构化数据的重要来源，但它们不直接与自然语言交互。尽管我们可以简单地使用 LLM 将用户的查询翻译成 SQL 查询，但错误的空间很小。

这里有一些有效的文本到 SQL 转换策略：

数据库描述

为了使 SQL 查询具体化，LLM 必须提供数据库的准确描述。一种常见的文本到 SQL 提示采用本文和其他论文中报道的想法：为每个表提供 `CREATE TABLE` 描述，包括列名和类型.^(5) 我们还可以提供一些（例如，三个）表中的示例行。

少样本示例

通过提供问题-查询匹配的少样本示例来填充提示，可以提高查询生成的准确性。这可以通过在提示中简单地附加标准静态示例来实现，以指导代理如何根据问题构建查询。

见 图 3-11 以了解过程的视觉表示。

![机器的特写，描述自动生成](img/lelc_0311.png)

###### 图 3-11\. 将用户查询转换为 SQL 查询

这里有一个完整的代码示例：

*Python*

```py
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI

# replace this with the connection details of your db
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
llm = ChatOpenAI(model="gpt-4", temperature=0)

# convert question to sql query
write_query = create_sql_query_chain(llm, db)

# Execute SQL query
execute_query = QuerySQLDatabaseTool(db=db)

# combined
chain = write_query | execute_query

# invoke the chain
chain.invoke('How many employees are there?');
```

*JavaScript*
