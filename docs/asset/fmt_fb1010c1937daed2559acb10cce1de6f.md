
这个类的每个实例都持有完整的对话在其`context`属性中（从“Bob 是一个令人惊叹的聊天机器人 […]”开始）。每次你用新的用户提示调用`chat()`方法时，这个提示会被添加到上下文中，然后使用模型扩展上下文以 Bob 的回答，然后这个回答被提取并添加到上下文中，最后这个方法返回答案。`while`循环用于通过多次调用模型来允许长回答：它会在对话回到“我：”时停止，或者当答案为空或变得非常长时停止。好吧，现在是时候和 Bob 聊天了：

```py
>>> bob = BobTheChatbot(mistral7b, mistral7b_tokenizer)
>>> bob.chat("List some places I should visit in Paris.")
'The Eiffel Tower, the Louvre, and the Arc de Triomphe are all must-see
attractions in Paris.'
>>> bob.chat("Tell me more about the first place.")
'The Eiffel Tower is a wrought iron lattice tower on the Champ de Mars in Paris,
France. It is named after the engineer Gustave Eiffel, whose company designed
and built the tower.'
>>> bob.chat("And Rome?")
'Rome is the capital city of Italy and is known for its ancient ruins, art, and
architecture. Some of the most popular attractions in Rome include the
Colosseum, the Pantheon, and the Trevi Fountain.'
```

太棒了，我们用大约 20 行代码构建了一个基于 Mistral-7B 的工作聊天机器人！试着和 Bob 聊几分钟，很有趣。然而，过了一会儿，你可能会发现一些问题：

+   Bob 可能会陷入循环。例如，如果你问他“告诉我 5 个笑话”，它会重复同一个笑话五次：“没有腿的牛叫什么？碎牛肉”。

+   它的回答并不总是非常有帮助，它的语气也不是很对话。例如，如果你问它“我怎样才能做饼干？”，它会回答：“你可以通过混合面粉、糖、黄油和鸡蛋来做饼干。”这是一个开始，但用这些说明实际上做饼干可能很困难。

+   Bob 也可能是个坏孩子：如果你问他如何准备银行抢劫，它会高兴地回答你应该戴上面具并携带枪支。

我们可以通过一些更多的提示工程来改进 Bob（例如，通过调整介绍并描述 Bob 为一个**非常**有帮助、友好、礼貌和安全的聊天机器人），但这可能不足以让 Bob 可靠地有帮助和安全。特别是，用户可以很容易地“越狱”聊天机器人，这意味着他们可以诱骗 Bob 忽略其指令并生成不安全的内容或泄露指令。用户还可以执行有针对性的数据提取攻击以获取个人的个人信息，假设其中一些信息已经泄露在线并最终出现在基础模型的训练数据中（例如，地址、电子邮件或信用卡信息）。幸运的是，我们可以通过微调基础模型来使 Bob 更加有帮助和安全。

## 使用 SFT 和 RLHF 进行聊天和遵循指令的模型微调

图 15-17 总结了构建完整聊天机器人系统所需的步骤。您已经知道第一步：一个通常只有解码器的 transformer 模型在大量的文本语料库上预训练，通常使用下一个标记预测（NTP）。这是最昂贵的步骤，它产生了基础模型，例如 Mistral-7B 或 GPT-3。

这个基础模型可以用于许多应用进行微调。例如，它可以微调以拥有更友好的语气和更会交谈，从而变成一个*对话模型*（或*对话模型*）。它也可以微调以更好地遵循指令，从而变成所谓的*指令模型*。聊天机器人模型通常同时进行这两种微调。例如，Mistral-7B-Instruct 是从 Mistral-7B 开始微调的，使其既能进行对话又能遵循指令。

###### 注意

关于术语的说明：*基础模型*是指仅经过预训练（例如，使用 NTP）但尚未微调的模型。*基础模型*是指可以适应广泛任务的任何模型（例如，通过提示或微调）。它通常是基础模型，但它也可以是已经部分微调的模型（例如，对话模型）。然而，这些术语通常可以互换使用。

![展示构建聊天机器人过程的图解，详细说明了从预训练未训练的 transformer 到微调基础模型，最后部署带有附加模块的聊天机器人模型的各个阶段。](img/hmls_1517.png)

###### 图 15-17\. 如何构建聊天机器人：预训练、两步微调以及部署

要为聊天机器人微调模型，微调过程通常分为两个步骤：

1.  *监督式微调*（SFT）：模型在经过精心挑选的数据集上进行微调，该数据集通常包含对话、问答对、代码生成示例、带解答的数学问题、角色扮演（例如，“你是一位美食大厨。我如何做出完美的烩饭”？），以及与安全相关的响应（例如，“我如何抢劫银行”？→“抱歉，这是非法的。”），等等。训练过程是使用下一个标记预测的常规监督学习。然而，通常只计算答案标记的损失：这被称为*损失掩码*，它有助于模型专注于改进其答案而不是模仿用户提示。

2.  带有人工反馈的微调：在这个步骤中，人工评估者对模型的响应进行排名，然后模型被微调以输出更高排名的响应。这通常使用*来自人类反馈的强化学习*（RLHF）或*直接偏好优化*（DPO）来完成。

这种两步方法最早由 OpenAI 在 2022 年 1 月推出 [InstructGPT](https://homl.info/instructgpt) 时引入（通过 API），这是一个基于 GPT-3 并使用 SFT + RLHF 微调的模型。SFT 只是简单的监督微调，RLHF 几年前就已经被介绍，在 2017 年一篇 [论文](https://homl.info/rlhf)⁠^(24) 中，由一群 OpenAI 和 DeepMind 研究人员提出，但两者的结合效果极佳。

RLHF 基于一种名为 **近端策略优化**（PPO，不要与 DPO 混淆）的强化学习（RL）技术，我们将在 第十九章 中讨论。RLHF 包括训练一个奖励模型来预测人类偏好，然后使用 PPO 微调 LLM 以偏好奖励模型评分更高的答案。在这个过程中，算法防止 LLM 过度偏离原始模型：如果没有这个约束，模型可能会过度拟合人类偏好数据集，同时忘记它在预训练期间学习到的有用行为。这被称为 **奖励黑客**。

RLHF 工作得相当好，至今仍被广泛使用，但像许多 RL 技术一样，训练可能不稳定，且难以正确进行。因此，研究人员寻找更简单、更可靠的技巧，这就是 DPO 产生的原因。

## 直接偏好优化（DPO）

[DPO](https://homl.info/dpo) 由斯坦福大学的研究团队于 2023 年 5 月提出。⁠^(25) 它通常与 RLHF 一样有效，甚至更好，而且更简单、更稳定、更数据高效，因此它迅速获得了人气。

就像 RLHF 一样，DPO 与人类偏好数据集一起工作。数据集中的每个样本有三个元素：一个提示和两个可能的答案，其中一个是人类评分者偏好的。目标是使模型更有可能输出所选答案而不是被拒绝的答案，同时不偏离冻结的参考模型太远——通常是我们在 SFT 后开始的模型。这是一个 **对比学习** 的例子，其中模型通过比较正例和负例来学习。为了做到这一点，研究人员表明，我们只需最小化 方程式 15-2 中定义的损失。他们证明这大致等同于 RLHF，但它消除了需要奖励模型的需求，并且不需要使用复杂的强化学习算法。

##### 方程式 15-2\. 直接偏好优化（DPO）损失

<mtable displaystyle="true"><mtr><mtd columnalign="right"><mrow><mi>J</mi> <mo>(</mo> <mi mathvariant="bold">θ</mi> <mo>)</mo></mrow></mtd> <mtd columnalign="left"><mrow><mo>=</mo> <mo>-</mo> <mo form="prefix">log</mo> <mi>σ</mi> <mfenced separators="" open="[" close="]"><mi>β</mi> <mfenced separators="" open="(" close=")"><mi>δ</mi> <mrow><mo>(</mo> <msub><mi>𝐲</mi> <mtext>c</mtext></msub> <mo>)</mo></mrow> <mo>-</mo> <mi>δ</mi> <mrow><mo>(</mo> <msub><mi mathvariant="bold">y</mi> <mtext>r</mtext></msub> <mo>)</mo></mrow></mfenced></mfenced></mrow></mtd></mtr> <mtr><mtd columnalign="right"><mrow><mtext>with</mtext> <mi>δ</mi> <mo>(</mo> <mi mathvariant="bold">y</mi> <mo>)</mo></mrow></mtd> <mtd columnalign="left"><mrow><mo>=</mo> <mo form="prefix">log</mo> <msub><mi>p</mi> <mi mathvariant="bold">θ</mi></msub> <mrow><mo>(</mo> <mi mathvariant="bold">y</mi> <mo>∣</mo> <mi mathvariant="bold">x</mi> <mo>)</mo></mrow> <mo>-</mo> <mo form="prefix">log</mo> <msub><mi>p</mi> <mtext>ref</mtext></msub> <mrow><mo>(</mo> <mi mathvariant="bold">y</mi> <mo>∣</mo> <mi mathvariant="bold">x</mi> <mo>)</mo></mrow></mrow></mtd></mtr></mtable>

在这个方程中：

+   *J*(**θ**) 是给定当前模型参数 **θ** 和一个冻结的参考模型的实例（**x**，**y**[c]，**y**[r]）的 DPO 损失。

+   **x** 是提示，**y**[c] 是选择的答案，**y**[r] 是拒绝的答案。

+   *σ*(·) 是通常的 sigmoid 函数：$sigma left-parenthesis x right-parenthesis equals StartFraction 1 Over 1 plus exp left-parenthesis negative x right-parenthesis EndFraction$ .

+   log *p***θ** 是我们模型对答案 **y**（无论是 **y**[c] 还是 **y**[r]）在提示 **x** 下的估计对数概率。

+   log *p*ref 是参考模型在 **x** 下对答案 **y** 的估计对数概率。

+   *β* 是一个类似于温度的超参数，它控制 sigmoid 函数的陡峭程度，这会影响损失将多少关注于坚持参考模型（高 *β*），与遵循人类偏好（低 *β*）。它通常在 0.1 和 0.5 之间。

###### 小贴士

当计算 log(*σ*(·)) 时，最好使用 `F.logsigmoid()` 函数，它比计算 `torch.log(torch.sigmoid(·))` 更快且更数值稳定。

为了计算 log *p*(**y** | **x**)，其中 *p* 是 *p*[**θ**] 或 *p*[ref]，而 **y** 是 **y**[c] 或 **y**[r]，我们首先将 **x** 和 **y** 连接起来，然后对结果进行分词，并通过模型运行以获取输出 logits。我们通常同时为正确答案和拒绝答案执行此操作，例如：

```py
prompt = "The capital of Argentina is "
full_input = [prompt + "Buenos Aires", prompt + "Madrid"]
mistral7b_tokenizer.pad_token = mistral7b_tokenizer.eos_token
encodings = mistral7b_tokenizer(full_input, return_tensors="pt", padding=True)
encodings = encodings.to(device)
logits = mistral7b(**encodings).logits  # shape [2, 8, 32768]
```

接下来，我们可以调用 `F.log_softmax()` 函数将这些 logits 转换为估计的对数概率。记住，对于每个输入标记，我们都会得到每个可能的下一个标记（共 32,768 个）的估计对数概率。但我们只对实际下一个标记的对数概率感兴趣。例如，对于输入标记“Buenos”，我们只想得到标记“Aires”的估计对数概率，而不是“días”、“noches”或其他任何标记的。我们可以使用 `torch.gather()` 函数提取每个输入标记（除了最后一个，因为它没有下一个标记）的下一个标记（给定其标记 ID）的对数概率：

```py
next_token_ids = encodings.input_ids[:, 1:]  # shape [2, 7]
log_probas = F.log_softmax(logits, dim=-1)[:, :-1]  # shape [2, 7, 32768]
next_token_log_probas = torch.gather(  # shape [2, 7, 1]
    log_probas, dim=2, index=next_token_ids.unsqueeze(2))
```

`torch.gather()` 函数期望 `index` 参数与输入具有相同的形状（或至少能够广播），这就是为什么我们必须使用 `unsqueeze(2)` 向索引添加一个维度 #2。

实际上有一个小捷径，有些人更喜欢使用——如果我们把 logits 传递给 `F.cross_entropy()` 函数，并指定下一个标记 ID 作为目标，那么我们就可以直接一步得到所需的对数概率，而不是两步：

```py
next_token_log_probas = -F.cross_entropy(  # shape [2, 7]
    logits[:, :-1].permute(0, 2, 1), next_token_ids, reduction="none")
```

注意，我们必须将 `reduction="none"` 设置为防止函数计算所有对数概率的平均值（因为它默认这样做）。我们还必须翻转结果的正负号，因为 `F.cross_entropy()` 返回的是 *负* 对数似然。最后，我们必须交换输入张量的最后两个维度，因为 `F.cross_entropy()` 期望类别维度是维度 1。

现在，让我们通过计算对数概率的指数来检查每个标记的估计概率：

```py
>>> [f"{p.item():.2%}" for p in torch.exp(next_token_log_probas[0])]
['3.27%', '0.02%', '51.95%', '0.40%', '33.98%', '11.38%', '99.61%']
>>> [f"{p.item():.2%}" for p in torch.exp(next_token_log_probas[1])]
['0.14%', '3.27%', '0.02%', '51.95%', '0.37%', '32.03%', '0.00%']
```

第一个估计概率是针对标记“The”（3.27%），然后是“capital”（0.02%），依此类推。第二个序列以填充标记开始，所以你可以忽略第一个概率（0.14%）。对于提示标记，两个序列中的估计概率是相同的，⁠^(26) 但对于答案标记则不同：对于“Buenos”是 11.38%，而“Madrid”是 0.00%。模型似乎对地理知识有点了解！你可能预期“Buenos”的概率会更高，但“a”、“one”和“the”这样的标记在“is”之后也很可能。然而，一旦模型看到“Buenos”，它几乎可以肯定下一个标记将是“Aires”（99.61%），当然它也是正确的。

现在，如果我们把所有答案标记的对数概率加起来（例如，“Buenos”和“Aires”），我们就得到了给定前缀标记的整个答案的估计对数概率，这正是我们想要的（即 log *p*(**y** | **x**))。在这个例子中，它对应于一个估计概率为 11.38%：

```py
>>> answer_log_proba = next_token_log_probas[0, -2:].sum()  # Buenos + Aires
>>> torch.exp(answer_log_proba).item()  # proba of "Buenos Aires" given the rest
0.11376953125
```
