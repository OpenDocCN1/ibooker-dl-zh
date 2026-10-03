
# 量化嵌入

量化嵌入允许你以牺牲一些理想上很小量的召回率为代价，显著提高存储效率和查询速度，这对于大规模搜索系统至关重要。例如，你可以通过将每个特征转换为 8 位整数（`Int8`）来减少 32 位浮点数（`Float32`）向量的内存使用量 75%。以这种方式压缩每个维度的单个数值（标量）称为*标量量化*。这通常可以在不显著影响召回率的情况下完成，并且当你有大量向量需要存储和搜索时特别有用。你甚至可以将每个特征量化到单个比特——一种称为*二进制量化*的技术——如果与使用更高精度向量在顶部-*N* 结果上重新排序的步骤结合使用，仍然可以保持相对较高的召回率。图 13.9 使用包含本书封面图像的向量（假设向量的维度代表图像中的像素）直观地展示了标量量化和二进制量化的概念。

![图](img/CH13_F09_Grainger.png)

##### 图 13.9 数据量化：全精度、降低标量精度和二进制精度

在图中，你可以看到原始图像（未量化）、标量量化图像（使用减少的颜色调色板，映射到相似的范围但颜色/精度较少）和二进制量化图像（每个像素要么是黑色要么是白色）。你会注意到标量量化图像仍然保留了原始图像的大部分重要细节，而二进制量化版本仍然可以清楚地识别，尽管丢失了一些重要数据（标题中的某些字符和颜色）。

在本节中，我们将介绍标量量化、二进制量化和另一种称为乘积量化的第三种量化类型。我们还将介绍一种称为马特罗什卡表示学习的多层嵌入方法，它可以用来动态切换已生成的嵌入的精度级别，而无需额外的量化或重新训练。

## 13.7.1 标量量化

标量量化是量化最简单的形式，其中嵌入向量中的每个值独立映射到较低精度的表示。考虑以下两个向量：

```py
[ -1.2345679,  2.2345679, 100.45679 ]  #4 bytes = 32 bits
[ -1.234,      2.234,     100.44 ]     #2 bytes = 16 bits
```

第一个向量是一个 32 位浮点表示，第二个是一个 16 位浮点表示，每个都四舍五入到最大可靠的精度。第二个向量需要的内存减少了 50%（2 字节对 4 字节），同时仍然表示大约相同的价值，只是精度较低。

这种降低精度是标量量化的一个简单例子，将更高精度的值映射到需要更少内存来存储和更少计算来处理的较低精度表示。

但如果我们想进一步压缩向量到单个字节（甚至几个比特）——我们还能做到这一点并保留大部分召回率吗？答案是肯定的，我们通常通过将浮点值范围映射到 `Int8` 来实现这一点，如图 13.10 所示。

![图](img/CH13_F10_Grainger.png)

##### 图 13.10 从 Float32 到 Int8 的标量量化

在图中，顶部的曲线表示浮点数范围内的值分布。由于我们从 32 位量化到 8 位，我们将 `Float32` 值的范围映射到更小的范围 -128 到 127（如果使用无符号整数，则为 0 到 255）。根据所使用的量化算法，通常会尝试尽可能充分地利用新的范围，通过*钳位*（限制范围到最小和最大值）以及利用原始向量中值的密度，将它们更均匀地映射到新的、量化的范围内。

让我们实现一个标量量化示例，看看这种优化对索引大小、召回率和搜索速度有什么影响。存在多个用于执行标量量化的库。您可以使用`sentence_transformers.quantization`模块，或者您的搜索引擎或语言模型可能内置了自己的量化实现。我们将利用 FAISS 库来构建我们的量化索引，并使用`sentence_transformers`库和 FAISS 的组合来构建我们的每个量化示例。FAISS（Facebook AI Similarity Search）是一个开源库，旨在高效地进行密集向量的相似性搜索和聚类。它与我们在本章早期用于语义搜索的 NMSLIB 库类似，但它有一些额外的功能，包括内置的量化支持。FAISS 在密集向量搜索的生产系统中得到广泛应用，并且是实现量化索引的一个很好的选择。

我们稍后会通过一个示例来展示如何使用所选搜索引擎的`collection`抽象进行搜索，但由于并非每个搜索引擎都支持所有量化模式，并且每个搜索引擎都有不同的开销和性能特征，我们将使用 FAISS 进行基准测试。

现在我们将构建一个具有全精度 `Float32` 嵌入的 FAISS 索引。然后我们将将其作为基准与各种量化嵌入索引进行比较，以比较索引大小、搜索速度和召回率。列表 13.20 显示了创建全精度 `Float32` 嵌入并将它们索引到 FAISS 索引中的代码。

##### 列表 13.20 使用 FAISS 索引全精度嵌入

```py
from sentence_transformers.quantization import quantize_embeddings

model = SentenceTransformer(
          "mixedbread-ai/mxbai-embed-large-v1",  #1
          similarity_fn_name=SimilarityFunction.DOT_PRODUCT,
          truncate_dim=1024)  #2

def index_full_precision_embeddings(doc_embeddings, name):
  index = faiss.IndexFlatIP(doc_embeddings.shape[1])  #3
  index.add(doc_embeddings)  #4
  faiss.write_index(index, name)  #5
  return index

def get_outdoors_embeddings(model):
  outdoors_dataframe = load_dataframe("data/outdoors/posts.csv")
  post_texts = [post["title"] + " " + post["body"]
                for post in outdoors_dataframe.collect()]
  return numpy.array(
    get_embeddings(post_texts, model, "outdoors_mrl_normed"))

doc_embeddings = get_outdoors_embeddings(model)  #6
full_index = index_full_precision_embeddings(  #7
               doc_embeddings, "full_embeddings")  #7
```

#1 该模型创建支持所有即将到来的优化技术的嵌入。

#2 原始嵌入将具有 1,024 个维度。

#3 `IndexFlatIP` 是一个简单、未优化的索引，支持不同的嵌入格式。

#4 将文档添加到索引中

#5 将索引写入磁盘

#6 为户外数据集生成嵌入

#7 创建一个全精度（`Float32`）的 FAISS 索引

在这个列表中，我们使用 `mixedbread-ai/mxbai-embed-large-v1` 模型为户外数据集计算嵌入表示，该模型产生高质量的嵌入表示，与我们的所有量化技术都很好地配合工作，并且支持玛特罗什卡表示学习，我们将在本章后面探讨。然后我们将这些嵌入索引到一个完整精度（`Float32`）的 FAISS 索引中，我们将很快将其用作基准来评估各种量化技术的性能。

对于我们的基准测试，我们还需要将一些测试查询编码成嵌入表示，这将在列表 13.21 中展示。

##### 列表 13.21 生成查询嵌入和完整索引基准

```py
def get_test_queries():
  return ["tent poles", "hiking trails", "mountain forests",
          "white water", "best waterfalls", "mountain biking",
          "snowboarding slopes", "bungee jumping", "public parks"]

queries = get_test_queries()  #1
query_embeddings = model.encode(queries,  #2
                     convert_to_numpy=True,  #2
                     normalize_embeddings=True)  #2

full_results = time_and_execute_search( #3
                 full_index, "full_embeddings",  #3
                 query_embeddings, k=25)  #3
display_statistics(full_results)  #4
```

#1 获取基准测试的测试查询

#2 为每个查询生成嵌入表示

#3 为完整精度（`Float32`）索引生成搜索时间、索引大小和召回率统计

#4 显示基准测试统计信息

输出：

```py
full_embeddings search took: 7.621 ms
full_embeddings index size: 75.6 MB
Recall: 1.0
```

在前面的列表中，我们定义了一个查询列表，我们将使用它来基准测试我们的完整精度索引与各种向量搜索优化策略。然后我们对列表 13.20 中的完整精度索引调用`time_and_execute_search`函数（为了简洁省略），然后将结果传递给`display_statistics`函数（同样为了简洁省略），该函数显示搜索时间、索引大小和召回率统计。

这为我们即将推出的量化（或优化）索引提供了一个比较基准。列表 13.22 展示了我们将使用来比较其他索引策略结果的两个附加函数的实现：`evaluate_search`函数和`evaluate_rerank_search`函数。

##### 列表 13.22 用于基准测试优化搜索方法的函数

```py
def evaluate_search(full_index, optimized_index,  #1
                    optimized_index_name,  #1
                    query_embeddings,  #1
                    optimized_query_embeddings,  #1
                    k=25, display=True, log=False):  #1
  full_results = time_and_execute_search(  #2
                   full_index, "full_embeddings",  #2
                   query_embeddings, k=k)  #2
  optimized_results = time_and_execute_search(  #2
                        optimized_index,  #2
                        optimized_index_name,  #2
                        optimized_query_embeddings, k=k)  #2
  if display:
    display_statistics(optimized_results, full_results)
  return optimized_results, full_results

def evaluate_rerank_search(full_index, optimized_index,  #3
                           query_embeddings,  #3
                           optimized_embeddings,  #3
                           k=50, limit=25):  #3
  results, full_results = evaluate_search(
                            full_index,
                            optimized_index, None,
                            query_embeddings,
                            optimized_embeddings,
                            display=False, k=k)

  doc_embeddings = get_outdoors_embeddings(model)  #4
  rescore_scores, rescore_ids = [], []
  for i in range(len(results["results"])):
    embedding_ids = results["faiss_ids"][i]
    top_k_embeddings = [doc_embeddings[id]  #5
                        for id in embedding_ids]  #5
    query_embedding = query_embeddings[i] #5
    scores = query_embedding @ \ #5
             numpy.array(top_k_embeddings).T  #5
    indices = scores.argsort()[::-1][:limit]  #6
    top_k_indices = embedding_ids[indices]  #6
    top_k_scores = scores[indices]  #6
    rescore_scores.append(top_k_scores)  #6
    rescore_ids.append(top_k_indices)  #6

  results = generate_search_results(rescore_scores, rescore_ids)  #7
  recall = calculate_recall(full_results["results"], results)  #8
  print(f"Reranked recall: {recall}")
```

#1 从每个索引中检索前 25 个结果并进行比较

#2 计算优化索引与完整精度索引的查询速度、索引大小和召回率

#3 与`evaluate_search`相同，但请求 k=50 个结果（默认值）并使用完整精度嵌入重新排序这些结果

#4 为户外数据集中的每个文档生成嵌入表示

#5 在查询嵌入和前 k 个嵌入之间执行点积操作

#6 按点积分数对结果进行排序以重新排序

#7 辅助函数（省略）用于将 ID 和分数与其他文档字段结合，以返回

#8 计算重新排序后的结果与完整精度索引的召回率

`evaluate_search`函数在内部对完整精度索引和量化索引都调用`time_and_execute_search`函数，并将结果传递给`display_statistics`函数以比较和显示搜索时间、索引大小和召回率统计。

然后，`evaluate_rerank_search`函数在重新排序使用全精度嵌入从量化索引中获取的顶级-*N* 结果后，再次计算召回率。虽然量化可以大幅减少内存和搜索时间，但我们会看到它也降低了召回率，这意味着一些*应该*返回的结果没有返回。但是，通过过度请求并使用全精度嵌入（通常在量化搜索后从磁盘加载，而不是与索引一起保留在内存中）重新排序仅顶级-*N* 结果，我们可以恢复大部分丢失的召回率。

我们将在每个后续列表中展示量化召回率和重新排序的量化召回率，以展示全精度搜索、量化搜索和重新排序的量化搜索之间的权衡。对于我们的第一个量化示例，让我们实现 `Int8` 标量量化。

##### 列表 13.23 使用 FAISS 创建 Int8 量化嵌入索引

```py
def index_int8_embeddings(doc_embeddings, name):
  int8_embeddings = quantize_embeddings(  #1
                      doc_embeddings, precision="int8")  #1
  print("Int8 embeddings shape:", int8_embeddings.shape)
  index = faiss.IndexFlatIP(int8_embeddings.shape[1]) #2
  index.add(int8_embeddings)  #3
  faiss.write_index(index, name) #4
  return index

int8_index_name = "int8_embeddings"
int8_index = index_int8_embeddings(doc_embeddings, int8_index_name)

quantized_queries = quantize_embeddings(  #5
  query_embeddings,  #5
  calibration_embeddings=doc_embeddings,  #5
  precision="int8")  #5

evaluate_search(full_index, int8_index,  #6
                int8_index_name, query_embeddings,  #6
                quantized_queries)  #6
evaluate_rerank_search(full_index, int8_index,  #7
          query_embeddings, quantized_queries)  #7
```
