
## 表 6.3 查询 `chef` 的按类别划分的上下文相关术语列表

| 查询：`chef` |  |
| --- | --- |
| `上下文：cooking 0.37731` `关键词：`<br>`chef 0.93239`<br>`chefs 0.5151`<br>`www.pamperedchef.com 0.41292`<br>`kitchen 0.39127`<br>`restaurant 0.38975`<br>`cooking 0.38332`<br>`chef's 0.37392`<br>`professional 0.36688`<br>`nakiri 0.36599`<br>`pampered 0.34736` | `上下文：devops 0.34959` `关键词：`<br>`chef 0.87653`<br>`puppet 0.79142`<br>`docs.chef.io 0.7865`<br>`ansible 0.73888`<br>`www.chef.io 0.72073`<br>`learn.chef.io 0.71902`<br>`default.rb 0.70194`<br>`configuration 0.68296`<br>`inspec 0.65237`<br>`cookbooks 0.61503` |

作为最后的例子，表 6.3 展示了查询 `chef` 的消歧。前两个上下文都显示了合理的正相关度分数，表明这两种含义都可能是解释。虽然“烹饪”上下文的分数（`0.37731`）略高于“devops”上下文（`0.34959`），但在选择这两种含义之间时，尽可能考虑用户上下文仍然很重要。在“devops”上下文中，`chef` 的含义与用于构建和部署服务器的 Chef 配置管理软件相关（相关术语包括 `puppet` 和 `ansible`），而在“烹饪”上下文中，它指的是准备食物的人（“烹饪”、“口味”、“餐厅”、“食材”）。Chef 软件从烹饪领域汲取灵感，作为准备和提供软件的隐喻，因此看到 `cookbooks` 这样的术语出现在“devops”类别中并不令人惊讶。

用于消歧查询的特定搜索引擎 SKG 请求可以通过调用 `print_disambigutaion_request` 函数来查看。这有助于理解和直接在配置的搜索引擎或向量数据库上运行内部 SKG 请求。为这个 `chef` 查询含义消歧函数调用打印的 Solr 特定 SKG 请求语法如下所示。

### 列表 6.4 为查询 `chef` 的 Solr SKG 消歧请求

```py
print_disambigutaion_request("chef", context_limit=2)
```

结果：

```py
{"limit": 0,
 "params": {"q": "*",
            "fore": "{!${defType} v=$q}",
            "back": "*",
            "defType": "edismax",
            "f0_0_query": "chef"}, #1
 "facet": {
   "f0_0": {
     "type": "query",
     "query": "{!edismax qf=body v=$f0_0_query}", #1
     "field": "body",
     "sort": {"relatedness": "desc"},
     "facet": {"relatedness": {"type": "func",
                               "func": "relatedness($fore,$back)"},
       "f1_0": {
         "type": "terms",
         "field": "category", #2
         "mincount": 5, "limit": 2,
         "sort": {"relatedness": "desc"},
         "facet": {"relatedness": {"type": "func",
                                   "func": "relatedness($fore,$back)"},
           "f2_0": {
             "type": "terms",  #3
             "field": "body", #3
             "mincount": 5, "limit": 10,
             "sort": {"relatedness": "desc"},
             "facet": {"relatedness":{"type": "func",
                                      "func": "relatedness($fore,$back)"}}
}}}}}}}
```

#1 起始节点是对 `chef` 的查询。

#2 第一次 SKG 遍历从类别字段中找到与起始节点最相关的术语。这些类别是消歧上下文。

#3 最后一次 SKG 遍历找到与消歧上下文相关的正文字段中的术语。

这是用于消歧具有 `context_limit` 为 `2` 的查询 `chef` 的内部 Solr SKG 请求。请求将特定于配置的任何搜索引擎或向量数据库，如果没有 SKG 功能，将回退到 Solr。有关更改配置的搜索引擎的说明，请参阅附录 B。

通过结合查询分类、术语消歧和查询扩展，SKG 可以为您的 AI 搜索引擎提供增强的特定领域和高度上下文化的语义搜索能力。我们将在第七章中进一步探讨这些技术的应用，当时我们将它们应用于一个实际的语义搜索应用中。

## 6.3 从查询信号中学习相关短语

到目前为止，你已经看到了如何使用你的内容作为知识图谱来发现相关术语、分类查询和消歧术语。虽然这些技术很强大，但它们完全依赖于你文档的质量。在本章的其余部分，我们将探讨关于你的领域的另一个主要知识来源——用户信号（查询、点击和后续操作）。通常，用户信号可以导致与文档内容相似，甚至更有用的见解，用于解释查询。

作为从真实用户行为中学习特定领域术语的起点，让我们考虑一下你的查询日志代表什么。对于你搜索引擎的每个查询，查询日志包含运行搜索的人的标识符、运行的查询以及查询的时间戳。这意味着如果单个用户搜索多个术语，你可以将这些搜索分组，并且也知道术语输入的顺序。

虽然这并不总是正确的，但一个合理的假设是，如果有人在很短的时间内输入两个不同的查询，那么第二个查询很可能是第一个查询的细化或关于相关主题的。图 6.2 演示了你在查询日志中可能找到的某个用户的真实搜索序列。

![图](img/CH06_F02_Grainger.png)

### 图 6.2 某特定用户的查询日志中典型的搜索序列

当查看这些查询时，我们直观地理解到 `iphond` 是 `iphone` 的拼写错误，`iphone accesories` 是 `iphone accessories` 的拼写错误，以及 `iphone`、`pink phone case` 和 `pink iphone case` 都是相关查询。我们将在后面的章节中处理这些拼写错误，但现在我们可以考虑这些也是相关术语。

虽然依赖单个用户的信号来推断两个查询相关并不明智，但许多用户之间的相似查询模式表明可能存在关联关系。正如我们在 5.4.5 节中展示的那样，查询可以扩展以包含相关术语来提高召回率。在本节中，我们将探讨学习相关查询的技术，首先通过挖掘查询日志，然后通过交叉引用产品交互日志。

### 6.3.1 从查询日志中挖掘相关查询

在我们开始挖掘用户信号以获取相关查询之前，让我们首先将我们的信号转换为更简单的格式以便处理。以下列表提供了从我们的通用信号结构到简单结构的转换，该结构将每个查询术语的出现映射到搜索该术语的用户。

#### 列表 6.5 将信号映射到关键词、用户对

```py
signals_collection = engine.get_collection("signals")
create_view_from_collection(signals_collection, "signals") #1
query = """SELECT LOWER(searches.target) AS keyword, searches.user
           FROM signals AS searches  #2
           WHERE searches.type='query'"""  #2
spark.sql(query).createOrReplaceTempView("user_searches")  #2
print_keyword_user_pairs()
```

#1 将信号语料库中的所有文档加载到 Spark 视图中

#2 从查询信号中选择关键词和用户数据

输出：

```py
Number of keyword user pairs: 725459

Keyword user pairs derived from signals:
User "u10" searched for "joy stick"
User "u10" searched for "xbox"
User "u10" searched for "xbox360"
```

您可以从这个列表中看到，有超过 725,000 个查询被表示出来。我们的目标是根据有多少用户输入了这两个查询来找到相关查询对。两个查询在不同用户的查询日志中共同出现的频率越高，这些查询被认为越相关。

下一个列表显示了每个查询对，其中两个查询都是由同一用户搜索的，以及搜索了这两个查询的用户数量（`users_cooc`）。

#### 列表 6.6 查询的总出现次数和共同出现次数

```py
query = """SELECT k1.keyword AS keyword1, k2.keyword AS keyword2,
           COUNT(DISTINCT k1.user) users_cooc  #1
           FROM user_searches k1
           JOIN user_searches k2 ON k1.user = k2.user
           WHERE k1.keyword > k2.keyword  #2
           GROUP BY k1.keyword, k2.keyword"""  #3
spark.sql(query).createOrReplaceTempView("keywords_users_cooc")
query = """SELECT keyword, COUNT(DISTINCT user) users_occ FROM
           user_searches GROUP BY keyword"""
spark.sql(query).createOrReplaceTempView("keywords_users_oc")
print_keyword_cooccurrences()
```

#1 统计同时搜索了 `k1` 和 `k2` 的用户数量

#2 限制关键词对仅有一个排列，以避免重复的查询对

#3 在用户字段上将 `user_searches` 视图与自身连接，以找到所有由同一用户搜索的关键词对

输出：

```py
+-----------+---------+
|    keyword|users_occ|
+-----------+---------+
|     lcd tv|     8449|
|       ipad|     7749|
|hp touchpad|     7144|
|  iphone 4s|     4642|
|   touchpad|     4019|
|     laptop|     3625|
|    laptops|     3435|
|      beats|     3282|
|       ipod|     3164|
| ipod touch|     2992|
+-----------+---------+

Number of co-occurring keyword searches: 244876

+-------------+---------------+----------+
|     keyword1|       keyword2|users_cooc|
+-------------+---------------+----------+
|green lantern|captain america|        23|
|    iphone 4s|         iphone|        21|
|       laptop|      hp laptop|        20|
|         thor|captain america|        18|
|         bose|          beats|        17|
|    iphone 4s|       iphone 4|        17|
|   skullcandy|          beats|        17|
|      laptops|         laptop|        16|
|      macbook|            mac|        16|
|         thor|  green lantern|        16|
+-------------+---------------+----------+
```

在列表 6.6 中，第一个查询产生了最常搜索的关键词，如结果所示。虽然这些可能是最受欢迎的查询，但它们并不一定是与其他查询共同出现频率最高的查询。第二个查询产生了至少由同一用户搜索过一次的两个查询对的总数（244,876）。最后一个查询按流行度对这些查询对进行排名。这些顶级查询对高度相关。

注意，然而，最上面的结果只有 `23` 个共同出现的用户，这意味着数据点的数量稀疏，并且列表下方的噪声可能会更多。在下一节中，我们将探讨一种沿不同轴（产品交互）组合信号的技术，这有助于解决稀疏性问题。

虽然直接将搜索数量聚合为用户共同出现的情况有助于找到最受欢迎的查询对，但搜索的流行度并不是唯一有用的度量标准，用于寻找相关性。关键词“和”和“的”高度共同出现，同样，“手机”、“电影”、“电脑”和“电子产品”也是高度共同出现的，因为它们都是许多人搜索的通用词。为了进一步关注术语之间关系强度，而独立于它们的个人流行度，我们可以使用一种称为*点互信息*的技术。

*点互信息*（PMI）是衡量任何两个事件之间关联度的指标。在自然语言处理的情况下，PMI 预测两个词共同出现的可能性，因为它们是相关的，而不是因为它们偶然共同出现的可能性。可以用于计算和归一化 PMI 的公式有很多，但我们将使用一种称为 PMI^k 的变体，其中 `k = 2`，这种变体比 PMI 在保持分数一致方面做得更好，无论词频如何。

计算 PMI²的公式显示在图 6.3 中。

![figure](img/grainger-ch6-eqs-0x.png)

#### 图 6.3 PMI²分数

在我们的实现中，`k1` 和 `k2` 代表我们想要比较的两个不同的关键词。`P(k1,k2)` 表示相同用户同时搜索这两个关键词的频率，而 `P(k1)` 和 `P(k2)` 分别表示用户只搜索第一个关键词或第二个关键词的频率。直观上，如果关键词出现的频率比它们随机出现的频率更高，那么它们的 PMI² 分数将更高。分数越高，这些术语在语义上相关的可能性就越大。

下面的列表展示了在我们共现查询对数据集上进行的 PMI² 计算。

#### 列表 6.7 用户搜索的 PMI² 计算

```py
query = """
SELECT k1.keyword AS k1, k2.keyword AS k2, k1_k2.users_cooc,
k1.users_occ AS n_users1, k2.users_occ AS n_users2,
LOG(POW(k1_k2.users_cooc, 2) /  #1
    (k1.users_occ * k2.users_occ)) AS pmi2  #1
FROM keywords_users_cooc AS k1_k2
JOIN keywords_users_oc AS k1 ON k1_k2.keyword1 = k1.keyword
JOIN keywords_users_oc AS k2 ON k1_k2.keyword2 = k2.keyword"""
spark.sql(query).createOrReplaceTempView("user_related_keywords_pmi")

spark.sql("""SELECT k1, k2, users_cooc, n_users1,
                    n_users2, ROUND(pmi2, 3) AS pmi2
             FROM user_related_keywords_pmi
             WHERE users_cooc > 5 ORDER BY pmi2 DESC, k1 ASC""").show(10)
```

#1 PMI 计算

输出：
