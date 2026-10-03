

# 第八章：使 transformers 在生产中更高效

在之前的章节中，您已经看到了 transformers 如何被微调以在各种任务上产生出色的结果。然而，在许多情况下，准确性（或者您正在优化的任何指标）是不够的；如果您的最先进模型太慢或太大，无法满足应用程序的业务需求，那么它就不是很有用。一个明显的替代方案是训练一个更快、更紧凑的模型，但模型容量的减少通常会伴随着性能的下降。那么当您需要一个快速、紧凑但高度准确的模型时，您该怎么办呢？

在本章中，我们将探讨四种互补的技术，可以用来加速预测并减少您的 transformer 模型的内存占用：*知识蒸馏*、*量化*、*修剪*和使用 Open Neural Network Exchange (ONNX)格式和 ONNX Runtime (ORT)进行*图优化*。我们还将看到其中一些技术如何结合起来产生显著的性能提升。例如，这是 Roblox 工程团队在他们的文章[“我们如何在 CPU 上扩展 BERT 以处理 10 亿+日请求”](https://oreil.ly/QdNIk)中采取的方法，正如图 8-1 所示，他们发现结合知识蒸馏和量化使他们的 BERT 分类器的延迟和吞吐量提高了 30 倍以上！

![在 Roblox 扩展 BERT](img/nlpt_0801.png)

###### 图 8-1\. Roblox 如何通过知识蒸馏、动态填充和权重量化扩展 BERT（照片由 Roblox 员工 Quoc N. Le 和 Kip Kaehler 提供）

为了说明与每种技术相关的好处和权衡，我们将以意图检测为案例研究；这是基于文本的助手的重要组成部分，低延迟对于实时维持对话至关重要。在学习的过程中，您将学习如何创建自定义训练器，执行高效的超参数搜索，并了解实施最前沿研究所需的内容，使用![nlpt_pin01](img/nlpt_pin01.png) Transformers。让我们开始吧！

## 以意图检测为案例研究

假设我们正在尝试为公司的呼叫中心构建一个基于文本的助手，以便客户可以在不需要与人类代理交谈的情况下请求其账户余额或进行预订。为了理解客户的目标，我们的助手需要能够将各种自然语言文本分类为一组预定义的动作或*意图*。例如，客户可能会发送以下关于即将到来的旅行的消息：

> 嘿，我想在 11 月 1 日到 11 月 15 日在巴黎租一辆车，我需要一辆 15 座位的面包车。

我们的意图分类器可以自动将此分类为*租车*意图，然后触发一个动作和响应。为了在生产环境中具有鲁棒性，我们的分类器还需要能够处理*超出范围*的查询，即客户提出不属于任何预定义意图的查询，系统应该产生一个回退响应。例如，在图 8-2 中显示的第二种情况中，客户询问有关体育的问题（超出范围），文本助手错误地将其分类为已知的范围内意图之一，并返回发薪日的响应。在第三种情况下，文本助手已经被训练来检测超出范围的查询（通常标记为一个单独的类），并告知客户它可以回答关于哪些主题的问题。

![超出范围的查询](img/nlpt_0802.png)

###### 图 8-2\. 人类（右）和基于文本的助手（左）之间的三次交流，涉及个人理财（由 Stefan Larson 等人提供）

作为基准，我们微调了一个 BERT-base 模型，在 CLINC150 数据集上达到了约 94%的准确性。这个数据集包括 150 个意图和 10 个领域（如银行和旅行）中的 22,500 个范围内查询，还包括属于`oos`意图类别的 1,200 个范围外查询。在实践中，我们还会收集自己的内部数据集，但使用公共数据是快速迭代和生成初步结果的好方法。

让我们从 Hugging Face Hub 下载我们微调的模型，并将其包装成文本分类的管道：

```py
from transformers import pipeline

bert_ckpt = "transformersbook/bert-base-uncased-finetuned-clinc"
pipe = pipeline("text-classification", model=bert_ckpt)
```

现在我们有了一个管道，我们可以传递一个查询以从模型获取预测的意图和置信度分数：

```py
query = """Hey, I'd like to rent a vehicle from Nov 1st to Nov 15th in
Paris and I need a 15 passenger van"""
pipe(query)
```

```py
[{'label': 'car_rental', 'score': 0.549003541469574}]
```

很好，`car_rental`意图是有意义的。现在让我们看看创建一个基准，我们可以用来评估我们基准模型的性能。

## 创建性能基准

与其他机器学习模型一样，在生产环境中部署 transformers 涉及在几个约束条件之间进行权衡，最常见的是：

*模型性能*

我们的模型在反映生产数据的精心设计的测试集上表现如何？当错误的成本很高时（最好通过人为干预来减轻），或者当我们需要对数百万个示例进行推断，并且模型指标的小幅改进可以转化为大幅增益时，这一点尤为重要。

*延迟*

我们的模型能够多快地提供预测？我们通常关心实时环境中的延迟，这些环境处理大量流量，就像 Stack Overflow 需要一个分类器来快速[检测网站上不受欢迎的评论](https://oreil.ly/cf7QX)一样。

*内存*

我们如何部署像 GPT-2 或 T5 这样需要占用几 GB 磁盘存储和内存的百亿参数模型？内存在移动设备或边缘设备中扮演着特别重要的角色，因为模型必须在没有强大的云服务器的情况下生成预测。

未能解决这些约束条件可能会对应用程序的用户体验产生负面影响。更常见的是，可能会导致运行昂贵的云服务器的成本激增，而这些服务器可能只需要处理少量请求。为了探索如何使用各种压缩技术优化这些约束条件，让我们从创建一个简单的基准开始，该基准可以测量给定管道和测试集的每个数量：

```py
class PerformanceBenchmark:
    def __init__(self, pipeline, dataset, optim_type="BERT baseline"):
        self.pipeline = pipeline
        self.dataset = dataset
        self.optim_type = optim_type

    def compute_accuracy(self):
        # We'll define this later
        pass

    def compute_size(self):
        # We'll define this later
        pass

    def time_pipeline(self):
        # We'll define this later
        pass

    def run_benchmark(self):
        metrics = {}
        metrics[self.optim_type] = self.compute_size()
        metrics[self.optim_type].update(self.time_pipeline())
        metrics[self.optim_type].update(self.compute_accuracy())
        return metrics
```

我们定义了一个`optim_type`参数，以跟踪我们在本章中将涵盖的不同优化技术。我们将使用`run_benchmark()`方法将所有指标收集到一个字典中，键由`optim_type`给出。

让我们现在通过在测试集上计算模型的准确性来为这个类添加一些具体内容。首先，我们需要一些数据进行测试，所以让我们下载用于微调基准模型的 CLINC150 数据集。我们可以通过以下方式从 Hub 获取数据集：![nlpt_pin01](img/nlpt_pin01.png)。

```py
from datasets import load_dataset

clinc = load_dataset("clinc_oos", "plus")
```

在这里，`plus`配置是指包含超出范围的训练示例的子集。CLINC150 数据集中的每个示例都包括`text`列中的查询及其对应的意图。我们将使用测试集来对我们的模型进行基准测试，所以让我们看一下数据集的一个示例：

```py
sample = clinc["test"][42]
sample
```

```py
{'intent': 133, 'text': 'transfer $100 from my checking to saving account'}
```

意图以 ID 的形式提供，但我们可以通过访问数据集的`features`属性轻松获取到字符串的映射（反之亦然）：

```py
intents = clinc["test"].features["intent"]
intents.int2str(sample["intent"])
```

```py
'transfer'
```

现在我们对 CLINC150 数据集的内容有了基本的了解，让我们实现`PerformanceBenchmark`的`compute_accuracy()`方法。由于数据集在意图类别上是平衡的，我们将使用准确性作为我们的度量标准。我们可以通过以下方式使用![nlpt_pin01](img/nlpt_pin01.png)数据集加载这个度量标准：

```py
from datasets import load_metric

accuracy_score = load_metric("accuracy")
```

准确度指标期望预测和参考（即，真实标签）是整数。我们可以使用管道从`text`字段中提取预测，然后使用我们的`intents`对象的`str2int()`方法将每个预测映射到其相应的 ID。以下代码在返回数据集的准确度之前收集所有的预测和标签。让我们也将其添加到我们的`PerformanceBenchmark`类中：

```py
def compute_accuracy(self):
    """This overrides the PerformanceBenchmark.compute_accuracy() method"""
    preds, labels = [], []
    for example in self.dataset:
        pred = self.pipeline(example["text"])[0]["label"]
        label = example["intent"]
        preds.append(intents.str2int(pred))
        labels.append(label)
    accuracy = accuracy_score.compute(predictions=preds, references=labels)
    print(f"Accuracy on test set - {accuracy['accuracy']:.3f}")
    return accuracy

PerformanceBenchmark.compute_accuracy = compute_accuracy
```

接下来，让我们使用 PyTorch 的`torch.save()`函数来计算我们模型的大小，将模型序列化到磁盘上。在内部，`torch.save()`使用 Python 的`pickle`模块，可以用来保存从模型到张量到普通 Python 对象的任何东西。在 PyTorch 中，保存模型的推荐方式是使用它的`state_dict`，这是一个 Python 字典，将模型中的每一层映射到它的可学习参数（即，权重和偏置）。让我们看看我们基准模型的`state_dict`中存储了什么：

```py
list(pipe.model.state_dict().items())[42]
```

```py
('bert.encoder.layer.2.attention.self.value.weight',
 tensor([[-1.0526e-02, -3.2215e-02,  2.2097e-02,  ..., -6.0953e-03,
           4.6521e-03,  2.9844e-02],
         [-1.4964e-02, -1.0915e-02,  5.2396e-04,  ...,  3.2047e-05,
          -2.6890e-02, -2.1943e-02],
         [-2.9640e-02, -3.7842e-03, -1.2582e-02,  ..., -1.0917e-02,
           3.1152e-02, -9.7786e-03],
         ...,
         [-1.5116e-02, -3.3226e-02,  4.2063e-02,  ..., -5.2652e-03,
           1.1093e-02,  2.9703e-03],
         [-3.6809e-02,  5.6848e-02, -2.6544e-02,  ..., -4.0114e-02,
           6.7487e-03,  1.0511e-03],
         [-2.4961e-02,  1.4747e-03, -5.4271e-02,  ...,  2.0004e-02,
           2.3981e-02, -4.2880e-02]]))
```

我们可以清楚地看到每个键/值对对应于 BERT 中的特定层和张量。因此，如果我们用以下方式保存我们的模型：

```py
torch.save(pipe.model.state_dict(), "model.pt")
```

我们可以使用 Python 的`pathlib`模块中的`Path.stat()`函数来获取有关底层文件的信息。特别是，`Path("model.pt").stat().st_size`将给出模型的大小（以字节为单位）。让我们将所有这些放在`compute_size()`函数中，并将其添加到`PerformanceBenchmark`中：

```py
import torch
from pathlib import Path

def compute_size(self):
    """This overrides the PerformanceBenchmark.compute_size() method"""
    state_dict = self.pipeline.model.state_dict()
    tmp_path = Path("model.pt")
    torch.save(state_dict, tmp_path)
    # Calculate size in megabytes
    size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)
    # Delete temporary file
    tmp_path.unlink()
    print(f"Model size (MB) - {size_mb:.2f}")
    return {"size_mb": size_mb}

PerformanceBenchmark.compute_size = compute_size
```

最后，让我们实现`time_pipeline()`函数，以便我们可以计算每个查询的平均延迟时间。对于这个应用程序，延迟时间指的是将文本查询输入到管道中并从模型返回预测意图所需的时间。在内部，管道还会对文本进行标记化，但这比生成预测快了大约一千倍，因此对整体延迟时间的贡献可以忽略不计。衡量代码片段的执行时间的一个简单方法是使用 Python 的`time`模块中的`perf_counter()`函数。这个函数比`time.time()`函数具有更好的时间分辨率，非常适合获取精确的结果。

我们可以使用`perf_counter()`通过传递我们的测试查询来计时我们的管道，并计算开始和结束之间的毫秒时间差：

```py
from time import perf_counter

for _ in range(3):
    start_time = perf_counter()
    _ = pipe(query)
    latency = perf_counter() - start_time
    print(f"Latency (ms) - {1000 * latency:.3f}")
```

```py
Latency (ms) - 85.367
Latency (ms) - 85.241
Latency (ms) - 87.275
```

这些结果展示了延迟时间的相当大的差异，并且表明通过管道的单次计时可能每次运行代码时都会得到完全不同的结果。因此，我们将收集多次运行的延迟时间，然后使用得到的分布来计算均值和标准差，这将让我们对数值的差异有一个概念。以下代码实现了我们需要的功能，并包括了在执行实际计时运行之前预热 CPU 的阶段：

