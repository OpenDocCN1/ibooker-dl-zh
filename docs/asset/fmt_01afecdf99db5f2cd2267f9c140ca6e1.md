
# 文本生成与束搜索解码

## 使用 Transformers 库进行文本生成

```py
import pandas as pd

input_txt = "Transformers are the"
input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device)
iterations = []
n_steps = 8
choices_per_step = 5

with torch.no_grad():
    for _ in range(n_steps):
        iteration = dict()
        iteration["Input"] = tokenizer.decode(input_ids[0])
        output = model(input_ids=input_ids)
        # Select logits of the first batch and the last token and apply softmax
        next_token_logits = output.logits[0, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        sorted_ids = torch.argsort(next_token_probs, dim=-1, descending=True)
        # Store tokens with highest probabilities
        for choice_idx in range(choices_per_step):
            token_id = sorted_ids[choice_idx]
            token_prob = next_token_probs[token_id].cpu().numpy()
            token_choice = (
                f"{tokenizer.decode(token_id)} ({100 * token_prob:.2f}%)"
            )
            iteration[f"Choice {choice_idx+1}"] = token_choice
        # Append predicted next token to input
        input_ids = torch.cat([input_ids, sorted_ids[None, 0, None]], dim=-1)
        iterations.append(iteration)

pd.DataFrame(iterations)
```

|  | 输入 | 选择 1 | 选择 2 | 选择 3 | 选择 4 | 选择 5 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Transformers are the | most (8.53%) | only (4.96%) | best (4.65%) | Transformers (4.37%) | ultimate (2.16%) |
| 1 | Transformers are the most | popular (16.78%) | powerful (5.37%) | common (4.96%) | famous (3.72%) | successful (3.20%) |
| 2 | Transformers are the most popular | toy (10.63%) | toys (7.23%) | Transformers (6.60%) | of (5.46%) | and (3.76%) |
| 3 | Transformers are the most popular toy | line (34.38%) | in (18.20%) | of (11.71%) | brand (6.10%) | line (2.69%) |
| 4 | Transformers are the most popular toy line | in (46.28%) | of (15.09%) | , (4.94%) | on (4.40%) | ever (2.72%) |
| 5 | Transformers are the most popular toy line in | the (65.99%) | history (12.42%) | America (6.91%) | Japan (2.44%) | North (1.40%) |
| 6 | Transformers are the most popular toy line in the | world (69.26%) | United (4.55%) | history (4.29%) | US (4.23%) | U (2.30%) |
| 7 | Transformers are the most popular toy line in the world | , (39.73%) | . (30.64%) | and (9.87%) | with (2.32%) | today (1.74%) |

通过这种简单的方法，我们能够生成句子"Transformers are the most popular toy line in the world"。有趣的是，这表明 GPT-2 已经内化了一些关于变形金刚媒体特许经营的知识，这是由两家玩具公司（孩之宝和 Takara Tomy）创造的。我们还可以看到每一步的其他可能的延续，这显示了文本生成的迭代性质。与其他任务（如序列分类）不同，在那些任务中，单次前向传递就足以生成预测，而在文本生成中，我们需要逐个解码输出标记。

实现贪婪搜索并不太难，但我们希望使用来自 ![nlpt_pin01](img/nlpt_pin01.png) Transformers 的内置`generate()`函数来探索更复杂的解码方法。为了重现我们的简单示例，让我们确保抽样被关闭（默认情况下关闭，除非您从加载检查点的特定模型配置中另有说明），并指定`max_new_tokens`为新生成标记的数量：

```py
input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device)
output = model.generate(input_ids, max_new_tokens=n_steps, do_sample=False)
print(tokenizer.decode(output[0]))
```

```py
Transformers are the most popular toy line in the world,
```

现在让我们尝试一些更有趣的东西：我们能否重现 OpenAI 的独角兽故事？与之前一样，我们将使用分词器对提示进行编码，并为`max_length`指定一个较大的值，以生成更长的文本序列。

```py
max_length = 128
input_txt = """In a shocking finding, scientist discovered \
a herd of unicorns living in a remote, previously unexplored \
valley, in the Andes Mountains. Even more surprising to the \
researchers was the fact that the unicorns spoke perfect English.\n\n `"""`
input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device)
output_greedy = model.generate(input_ids, max_length=max_length,
                               do_sample=False)
print(tokenizer.decode(output_greedy[0]))
```

```py
In a shocking finding, scientist discovered a herd of unicorns living in a
remote, previously unexplored valley, in the Andes Mountains. Even more
surprising to the researchers was the fact that the unicorns spoke perfect
English.

The researchers, from the University of California, Davis, and the University of
Colorado, Boulder, were conducting a study on the Andean cloud forest, which is
home to the rare species of cloud forest trees.

The researchers were surprised to find that the unicorns were able to
communicate with each other, and even with humans.

The researchers were surprised to find that the unicorns were able
```

嗯，前几句与 OpenAI 的示例有很大不同，有趣的是涉及到不同的大学被认为是发现者！我们还可以看到贪婪搜索解码的一个主要缺点：它倾向于产生重复的输出序列，在新闻文章中显然是不可取的。这是贪婪搜索算法的一个常见问题，它可能无法给出最佳解决方案；在解码的上下文中，它可能会错过整体概率更高的词序列，只是因为高概率的词恰好是由低概率的词前导的。

幸运的是，我们可以做得更好——让我们来看一种被称为*束搜索解码*的流行方法。

> **注意**
> 尽管贪婪搜索解码在需要多样性的文本生成任务中很少使用，但它对于生成像算术这样需要确定性和事实正确的输出的短序列是有用的。⁴ 对于这些任务，您可以通过提供格式为`"5 + 8 => 13 \n 7 + 2 => 9 \n 1 + 0 =>"`的几个以换行符分隔的示例来对 GPT-2 进行条件设置。

## 束搜索解码

束搜索不是在每一步解码时选择具有最高概率的标记，而是跟踪前*b*个最有可能的下一个标记，其中*b*被称为*束*或*部分假设*的数量。下一组束是通过考虑现有集合的所有可能的下一个标记扩展，并选择*b*个最有可能的扩展来选择的。这个过程重复进行，直到达到最大长度或 EOS 标记，并且通过根据它们的对数概率对*b*束进行排名来选择最有可能的序列。束搜索的一个示例显示在图 5-4 中。

![束搜索](img/nlpt_0504.png)

**图 5-4。** 具有两个束的束搜索

为什么我们要使用对数概率而不是概率本身来对序列进行评分？计算序列的总体概率<math alttext="upper P left-parenthesis y 1 comma y 2 comma ellipsis comma y Subscript t Baseline vertical-bar bold x right-parenthesis"><mrow><mi>P</mi> <mo>(</mo> <msub><mi>y</mi> <mn>1</mn></msub> <mo>,</mo> <msub><mi>y</mi> <mn>2</mn></msub> <mo>,</mo> <mo>...</mo> <mo>,</mo> <msub><mi>y</mi> <mi>t</mi></msub> <mo>|</mo> <mi>𝐱</mi> <mo>)</mo></mrow></math>涉及计算条件概率<math alttext="upper P left-parenthesis y Subscript t Baseline vertical-bar y Subscript t Baseline comma bold x right-parenthesis"><mrow><mi>P</mi> <mo>(</mo> <msub><mi>y</mi> <mi>t</mi></msub> <mo>|</mo> <msub><mi>y</mi> <mrow><mo><</mo><mi>t</mi></mrow></msub> <mo>,</mo> <mi>𝐱</mi> <mo>)</mo></mrow></math>的*乘积*是一个原因。由于每个条件概率通常是在[ <math alttext="0 comma 1"><mrow><mn>0</mn> <mo>,</mo> <mn>1</mn></mrow></math> ]范围内的一个小数，取它们的乘积可能导致总体概率很容易下溢。这意味着计算机不能准确地表示计算结果。例如，假设我们有一个包含<math alttext="t equals 1024"><mrow><mi>t</mi> <mo>=</mo> <mn>1024</mn></mrow></math>个标记的序列，并慷慨地假设每个标记的概率为 0.5。这个序列的总体概率是一个极小的数：

```py
0.5 ** 1024
```

```py
5.562684646268003e-309
```

这导致了数值不稳定，因为我们遇到了下溢。我们可以通过计算一个相关的术语，即对数概率来避免这种情况。如果我们对联合和条件概率应用对数，然后借助对数的乘法规则，我们得到：

<math alttext="log upper P left-parenthesis y 1 comma ellipsis y Subscript t Baseline vertical-bar bold x right-parenthesis equals sigma-summation Underscript t equals 1 Overscript upper N Endscripts log upper P left-parenthesis y Subscript t Baseline vertical-bar y Subscript t Baseline comma bold x right-parenthesis" display="block"><mrow><mo form="prefix">log</mo> <mi>P</mi> <mrow><mo>(</mo> <msub><mi>y</mi> <mn>1</mn></msub> <mo>,</mo> <mo>...</mo> <msub><mi>y</mi> <mi>t</mi></msub> <mo>|</mo> <mi>𝐱</mi> <mo>)</mo></mrow> <mo>=</mo> <munderover><mo>∑</mo> <mrow><mi>t</mi><mo>=</mo><mn>1</mn></mrow> <mi>N</mi></munderover> <mo form="prefix">log</mo> <mi>P</mi> <mrow><mo>(</mo> <msub><mi>y</mi> <mi>t</mi></msub> <mo>|</mo> <msub><mi>y</mi> <mrow><mo><</mo><mi>t</mi></mrow></msub> <mo>,</mo> <mi>𝐱</mi> <mo>)</mo></mrow></mrow></math>

换句话说，我们之前看到的概率乘积变成了对数概率的总和，这样就不太可能遇到数值不稳定性的问题。例如，计算与之前相同示例的对数概率为：

```py
import numpy as np

sum([np.log(0.5)] * 1024)
```

```py
-709.7827128933695
```

这是一个我们可以轻松处理的数字，这种方法对于更小的数字仍然有效。因为我们只想比较相对概率，所以我们可以直接使用对数概率进行比较。

让我们计算并比较贪婪搜索和束搜索生成的文本的对数概率，看看束搜索是否可以提高整体概率。由于 ![nlpt_pin01](img/nlpt_pin01.png) Transformers 模型返回给定输入标记的下一个标记的非归一化 logits，我们首先需要对 logits 进行归一化，以创建整个词汇表上每个标记的概率分布。然后我们需要选择仅出现在序列中的标记概率。以下函数实现了这些步骤：

```py
import torch.nn.functional as F

def log_probs_from_logits(logits, labels):
    logp = F.log_softmax(logits, dim=-1)
    logp_label = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logp_label
```

这给我们了单个标记的对数概率，所以要得到序列的总对数概率，我们只需要对每个标记的对数概率求和：

```py
def sequence_logprob(model, labels, input_len=0):
    with torch.no_grad():
        output = model(labels)
        log_probs = log_probs_from_logits(
            output.logits[:, :-1, :], labels[:, 1:])
        seq_log_prob = torch.sum(log_probs[:, input_len:])
    return seq_log_prob.cpu().numpy()
```
