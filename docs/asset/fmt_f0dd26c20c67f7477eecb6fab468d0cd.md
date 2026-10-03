
# 第八章：信号增强模型

## 本章涵盖

+ 聚合用户信号以创建基于流行度的排名模型
+ 对噪声查询输入进行信号归一化
+ 在众包信号中打击信号垃圾邮件
+ 应用时间衰减以优先考虑近期信号
+ 将多种信号类型混合到一个模型中
+ 选择查询时增强或索引时增强

在第四章中，我们介绍了三种不同类别的反射智能：信号增强（流行相关性）、协同过滤（个性化相关性）和排序学习（泛化相关性）。在本章中，我们将深入探讨这些中的第一个，实现信号增强以增强您最热门的查询和文档的相关性排名。

在大多数搜索引擎中，相对较少的查询往往占很大一部分总查询量。这些热门查询，称为*头部查询*，也倾向于产生更多信号（例如电子商务用例中的点击和购买），这有助于对顶级搜索结果的热度做出更强的推断。

信号增强模型直接利用这些更强的推理，并确保您最重要的和最高可见度的查询得到最佳调整以返回最相关的文档。

## 8.1 基本信号增强

在 4.2.2 节中，我们在 RetroTech 数据集上构建了我们的第一个信号增强模型，显著提高了最常搜索和点击的搜索结果的相关性。在本节中，我们将快速回顾创建简单信号增强模型的过程，我们将在接下来的章节中在此基础上解决一些更高级的需求。

您会记得从 4.2.2 节中，信号增强模型聚合了文档上的有用用户行为信号（如点击信号），这些信号是特定查询的结果。我们使用对`ipad`的搜索，并根据该搜索结果中该文档被点击的次数来增强每个文档。图 8.1 展示了查询`ipad`在应用信号增强前后的搜索结果。

![图](img/CH08_F01_Grainger.png)

##### 图 8.1 应用信号增强模型前后的对比。信号增强通过将最受欢迎的项目推到搜索结果顶部来提高相关性。

导致图 8.1 中相关性提高的信号增强模型是一个基本的信号增强模型。它查看给定查询中每个曾经被点击的文档，并应用与该查询在该文档上的过去点击总数相等的增强。

尽管本章 4.2.2 节中介绍的基本信号增强模型极大地提高了相关性，但它不幸地容易受到一些数据偏差甚至操纵的影响。在 8.2 节中，我们将讨论一些去除信号中的噪声的技术，以最大化信号增强模型的质量并降低不希望偏差的可能性。

## 8.2 归一化信号

在聚合之前对传入的用户查询进行标准化非常重要，以便将变体视为相同的查询。鉴于最终用户可以将任何任意文本作为查询输入，聚合的信号本质上是有噪声的。第四章中的基本信号增强模型（在第 8.1 节中总结）没有进行标准化。它为每个查询和文档对生成聚合的增强，但由于传入的查询尚未标准化为共同的形式，查询的变体将被视为完全不同的查询。以下列表生成了一个列表，列出了在搜索结果中增强最受欢迎的 iPad 型号的顶级查询。

##### 列表 8.1 聚合信号并检索相关查询

```py
def create_boosting_collection(collection_name):
  basic_signals_aggregation_query = """  #1
  SELECT q.target AS query, c.target AS doc,  #1
  COUNT(c.target) AS boost  #1
  FROM signals c LEFT JOIN signals q  #1
  ON c.query_id = q.query_id  #1
  WHERE c.type = 'click' AND q.type = 'query'  #1
  GROUP BY q.target, doc  #1
  ORDER BY boost DESC  #1
  """
  collection = engine.get_collection(collection_name)  #2
  return aggregate_signals(collection, "basic_signals_boosts",  #2
                           basic_signals_aggregation_query)  #2

def search_for_boosts(query, collection, query_field="query"):  #3
  boosts_request = {"query": query,  #3
                    "query_fields": [query_field],  #3
                    "return_fields": ["query", "doc", "boost"], #3
                    "limit": 20, 3((CO1-16)) #3
                    "order_by": [("boost", "desc")]}  #3
  response = collection.search(**boosts_request)  #3
  return response["docs"] #3

signals_boosting_collection = create_boosting_collection("signals")
query = "885909457588"  #4
signals_docs = search_for_boosts(query, signals_boosting_collection, "doc") #5
show_raw_boosted_queries(signals_docs)  #5
```

#1 定义信号聚合查询
#2 从信号集合运行聚合到基本信号增强集合
#3 加载指定查询和集合的信号增强
#4 最受欢迎的 iPad 型号
#5 返回指定文档的顶级信号增强列表

输出：

```py
Raw Boosted Queries
"iPad" : 1050
"ipad" : 966
"Ipad" : 829
"iPad 2" : 509
"ipad 2" : 347
"Ipad2" : 261
"ipad2" : 238
"Ipad 2" : 213
"I pad" : 203
"i pad" : 133
"IPad" : 77
"Apple" : 76
"I pad 2" : 60
"apple ipad" : 55
"Apple iPad" : 53
"ipads" : 43
"tablets" : 42
"apple" : 41
"iPads" : 38
"i pad 2" : 38
```

此列表通过其查询聚合所有信号，并将每个查询及其出现次数存储到一个新的集合中。您可以从输出中看到，在基本信号增强模型中存在许多相同查询的变体。变体的最大罪魁祸首似乎是对大小写敏感，因为我们看到`iPad`、`ipad`、`Ipad`和`IPad`作为常见变体。空格似乎也是另一个问题，有`ipad` `2`与`i pad` `2`与`ipad2`之间的区别。我们甚至在`ipad`与`ipads`中看到了单数与复数表示。

关键词搜索字段通常将查询标准化为不区分大小写，使用词干分析来忽略术语的复数形式，并在单词之间的大小写变化和字母到数字的转换处分割。同样，对信号进行标准化也很有用，因为保持不同查询术语和增强的变体，这些变体对于搜索引擎来说无法区分，可能是适得其反的。未能标准化术语会分散信号的价值，因为信号被分散到具有较低增强的相同关键词的变体中，而不是合并成更有意义的查询并具有更强的增强。

在信号聚合之前，您需要确定查询标准化应该有多复杂，但即使只是将传入的查询转换为小写以使信号聚合不区分大小写，也能走很长的路。以下列表演示了与之前相同的信号聚合基本操作，但这次首先将查询转换为小写。

##### 列表 8.2 不区分大小写的信号聚合

```py
normalized_signals_aggregation_query = """ 
SELECT LOWER(q.target) AS query, #1
c.target AS doc, COUNT(c.target) AS boost   #2
FROM signals c LEFT JOIN signals q ON c.query_id = q.query_id 
WHERE c.type = 'click' AND q.type = 'query' 
GROUP BY LOWER(q.target), doc   #2
ORDER BY boost DESC
""" 

normalized_collection = \ 
  aggregate_signals(signals_collection, "normalized_signals_boosts", 
                    normalized_signals_aggregation_query) 

query = "885909457588"  #3
signals_documents = search_for_boosts(query, normalized_collection, "doc") 
show_raw_boosted_queries(signals_documents)
```

#1 通过将每个查询转换为小写来标准化情况
#2 按标准化查询分组增加了这些查询的信号计数，增加了信号增强
#3 最受欢迎的 iPad 型号

输出：

```py
Raw Boosted Queries
"ipad" : 2939
"ipad 2" : 1104
"ipad2" : 540
"i pad" : 341
"apple ipad" : 152
"ipads" : 123
"apple" : 118
"i pad 2" : 99
"tablets" : 67
"tablet" : 61
"ipad 1" : 52
"apple ipad 2" : 27
"hp touchpad" : 26
"ipaq" : 20
"i pad2" : 19
"wi" : 19
"apple computers" : 18
"apple i pad" : 15
"ipad 2 16gb" : 15
"samsung galaxy" : 14
```

原始增强查询列表看起来已经整洁多了！不仅冗余更少，而且请注意，信号增强的强度已经增加，因为更多的信号被归因于查询的规范形式（小写版本）。

将查询转换为小写，也许删除空白或多余的字符，通常在聚合信号之前对查询进行充分规范就足够了。然而，本节的重要启示是，信号增强模型越强，你就能确保在聚合时将相同的查询视为相同。

查询中的变化并不是我们需要担心数据中的唯一类型的噪声。在下一节中，我们将讨论如何克服由垃圾邮件引起的潜在的重大问题。

## 8.3 拒绝信号垃圾邮件

任何时候我们使用如点击信号这样的众包数据来影响搜索引擎的行为时，我们需要问自己“数据输入如何被操纵以产生不理想的结果？”在本节中，我们将展示如何通过点击信号垃圾邮件来操纵搜索结果，以及如何阻止它。

### 8.3.1 使用信号垃圾邮件来操纵搜索结果

让我们想象有一个用户，出于某种原因，非常讨厌*星球大战*，并认为最新的电影完全是垃圾。他们感情如此强烈，以至于他们想要确保任何搜索`star wars`都能返回作为顶级搜索结果的物理垃圾桶。这个用户对搜索引擎略知一二，并注意到你的杀手级相关性算法似乎在使用用户信号和信号增强。图 8.2 显示了查询`star` `wars`的默认响应，信号增强将最受欢迎的产品带到搜索结果的顶部。

![figure](img/CH08_F02_Grainger.png)

##### 图 8.2 在查询`star wars`时最受欢迎的搜索结果，信号增强已开启。在没有恶意信号垃圾邮件的情况下，这些是预期的结果。

用户决定既然你的搜索引擎排名基于流行项目，他们将通过大量搜索`star wars`来垃圾邮件搜索引擎。他们将跟随每次搜索后对找到的星战主题垃圾桶的虚假点击，试图使垃圾桶出现在搜索结果的顶部。

为了模拟这种场景，我们将在以下列表中运行一个简单的脚本，以生成 5,000 个`star wars`查询和在该查询后对垃圾桶进行 5,000 次相应的点击。

##### 列表 8.3 生成垃圾查询和点击信号

```py
signals_collection = engine.get_collection("signals")
spam_user = "u8675309"
spam_query = "star wars"
spam_signal_boost_doc_upc = "45626176"  #1

signal_docs = []
for num in range(5000):  #2
  query_id = f"u8675309_0_{num}"  #2
  query_signal = {   #2
    "query_id": query_id,  #2
    "user": spam_user,  #2
    "type": "query",  #2
    "target": spam_query,  #2
    "signal_time": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),  #2
    "id": f"spam_signal_query_{num}"}  #2
  click_signal = {   #2
    "query_id": query_id,  #2
    "user": spam_user,  #2
    "type": "click",  #2
    "target": spam_signal_boost_doc_upc,  #2
    "signal_time": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),  #2
    "id": f"spam_signal_click_{num}"}  #2
  signal_docs.extend([click_signal, query_signal])  #2

signals_collection.add_documents(signal_docs)   #2

spam_signals_collection = \  #3
  aggregate_signals(signals_collection, "signals_boosts_with_spam", #3
    normalized_signals_aggregation_query)  #3
```

#1 将垃圾邮件想要移至搜索结果顶部的内容文档
#2 生成并发送 5,000 个查询和点击信号到搜索引擎
#3 生成并发送 5,000 个查询和点击信号到搜索引擎
#4 运行信号聚合以生成包括垃圾信号的信号增强模型

列表 8.3 向我们的搜索引擎发送数千个垃圾查询和点击信号，模拟了如果用户搜索并点击特定的搜索结果数千次我们会看到的结果。该列表然后重新运行基本的信号聚合。

为了看到恶意用户垃圾点击行为对我们搜索结果的影响，以下列表运行了查询 `star wars`，现在结合了操纵后的信号增强模型。

##### 列表 8.4 受垃圾用户信号影响的搜索结果

```py
def boosted_product_search_request(query, collection, boost_field=None):
  signals_documents = search_for_boosts(query, collection)  #1
  signals_boosts = create_boosts_query(signals_documents)  #1
  boosted_request = product_search_request(query)  #2
  if boost_field:  #2
    signals_boosts = (boost_field, signals_boosts)  #2
  boosted_request["query_boosts"] = signals_boosts  #2
  return boosted_request

query = '"star wars"'
boosted_request = boosted_product_search_request(query,
                    spam_signals_collection, "upc")
response = products_collection.search(**boosted_request)
display_product_search(query, response["docs"])
```

#1 从包含垃圾信号的信号增强模型中加载信号增强
#2 使用信号增强模型提升星球大战查询

图 8.3 显示了从列表 8.4 生成的新的操纵后的搜索结果，其中星球大战垃圾桶位于顶部位置。

![图](img/CH08_F03_Grainger.png)

##### 图 8.3 用户通过向搜索引擎发送虚假信号来操纵搜索结果。用户只需多次点击就能修改顶部结果。

垃圾邮件发送者成功了，这些被操纵的搜索结果现在将出现在每个在 RetroTech 网站上搜索 `star wars` 的后续访客面前！看起来我们需要使我们的信号增强模型更加健壮，以对抗这种信号垃圾。

### 8.3.2 通过基于用户的过滤来对抗信号垃圾

如果你打算使用如用户信号这样的众包数据来影响搜索引擎排名，那么采取步骤最小化用户操纵基于信号的排名算法的能力是很重要的。

为了对抗我们刚刚展示的星球大战垃圾桶问题，最简单的技术是确保同一用户重复点击只获得信号增强聚合中的一个“投票”。这样，无论恶意用户点击一次还是一百万次，他们的点击只算作一个信号，因此对信号增强模型没有实质性的影响。以下列表重新设计了信号聚合查询，只计算每个用户的唯一点击信号。

##### 列表 8.5 去除噪声用户信号
