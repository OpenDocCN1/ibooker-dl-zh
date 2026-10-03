
请务必进行更多实验，以了解模型的优缺点。例如，您可以尝试使用我们的示例句子提示模型，“他不想在手机上谈论细胞，因为他认为这很无聊。”我们发现这在创意写作空间和技术写作空间中都是一个可信的应用，`max_length`设置为较小的数值。对许多作者来说，它已经是一个可信的辅助工具。在撰写本文时，我们只能想象 GPT-3 能够做到什么。未来确实非常令人兴奋。

玩弄文本生成后，让我们看看是否可以以某种方式使用它来创建聊天机器人。

### 7.2.3 聊天机器人的应用

直觉上应该能够无需对此应用进行重大修改即可采用 GPT。幸运的是，微软的人员已经通过模型 DialoGPT 完成了这一点，该模型最近也被包含在 transformers 库中。它的架构与 GPT 相同，只是增加了特殊标记，以指示对话中参与者的回合结束。在看到这样的标记后，我们可以将参与者的新贡献添加到启动上下文文本中，并通过直接应用 GPT 来生成聊天机器人的响应，迭代重复这个过程。自然地，预训练的 GPT 模型在会话文本上进行了微调，以确保响应是适当的。作者们使用 Reddit 主题进行了微调。

让我们继续构建一个聊天机器人吧！在这种情况下，我们不会使用管道，因为在撰写本文时，该模型尚未通过该 API 公开。这使我们能够对比调用这些模型进行推理的不同方法，这对你来说是一个有用的练习。

首先要做的事情是通过以下命令加载预训练模型和分词器：

```py
from transformers import GPT2LMHeadModel, GPT2Tokenizer      ❶
import torch                                                 ❷

tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-medium")
```

❶ 请注意，DialoGPT 模型使用 GPT-2 类。

❷ 我们在这里使用 Torch 而不是 TensorFlow，因为 transformers 文档中默认选择的是 Torch 平台。

此处值得强调几点。首先，请注意我们使用的是 GPT-2 模型类，这与我们先前讨论的 DialoGPT 作为该架构的直接应用是一致的。另外，请注意我们可以与这些 GPT 特定的模型类交换使用`AutoModelWithLMHead`和`AutoTokenizer`类。这些实用程序类会检测用于加载指定模型的最佳类别，例如，在这种情况下，它们将检测到最佳要使用的类别为`GPT2LMHeadModel`和`GPT2Tokenizer`。浏览 transformers 库文档时，你可能会遇到这些实用程序类，了解它们的存在对你的代码更一般化是有好处的。最后请注意，这里使用的是 GPT 的“LMHead”版本。这意味着从普通 GPT 得到的输出将通过一个线性层和一个归一化层，然后转换成一个维度等于词汇表大小的概率向量。最大值对应于模型正确训练的情况下下一个最有可能的令牌。

与我们加载的预训练 DialoGPT 模型进行对话的代码如列表 7.1 所示。我们首先指定最多五个回应的最大数量。然后，我们编码用户在每个轮次的对话，将对话添加到聊天历史记录中，并将其传递给加载的预训练 DialoGPT 模型，以生成下一个响应。

**列表 7.1 与预训练 DialoGPT 模型进行对话，最多五个聊天机器人响应**

```py
conversation_length = 5                                                                                           ❶
for step in range(conversation_length):
    new_user_inputs_ids = tokenizer.encode(input("User: ") + tokenizer.eos_token, return_tensors='pt')            ❷
    bot_input_ids = torch.cat([chat_history_ids, new_user_inputs_ids], dim=1)                                     ❸
     if step > 0 else new_user_inputs_ids   
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)        ❹
    print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))               ❺
```

❶ 五行聊天

❷ 编码新用户输入，添加一个句子结束标记，并返回张量

❸ 将新输入添加到聊天历史记录中

❹ 使用聊天历史记录作为上下文生成最多 `max_length` 令牌的响应

❺ 显示响应

一个人可能整天都和这个机器人玩耍！我们很开心地向它提问各种问题，并以各种方式提示它。我们在这里包括了其中一个交流以供您娱乐：

```py
User: Hi there.
DialogGPT: Hi there.
User: How are you today?
DialogGPT: I'm good, how are you?
User: Good! How much money do you have?
DialogGPT: I have about 100k.
User: What will you spend it on?
DialogGPT: I'm not sure, I'm not sure what I want to spend it on.
User: Make a decision, life is short.
DialogGPT: I'm going to go with a lot of things.
```

接下来这短暂对话的另一端很可能是一个人，不是吗？那是否意味着它通过了图灵测试？并非如此，下面的交流说明了这一点：

```py
User: Hi robot.
DialogGPT: Hello, human.
User: Huh?
DialogGPT: I'm a bot.
User: OK, what is your name?
DialogGPT: Robot. I'm a robot.
User: All right then.
DialogGPT: Robot. I'm a robot.
User: Say something else.
DialogGPT: Robot. I'm a robot.
```

当你增加允许的对话轮次数量时，你会发现机器人会陷入重复的与话题无关的回复中。这类似于 GPT 开放式文本生成随着生成文本长度的增加变得更加荒谬。改善这一点的一个简单方法是保持固定的局部上下文大小，其中模型只受到该上下文内的对话历史的提示。当然，这意味着对话不总是考虑整个对话的上下文——这是必须对任何给定应用进行实验探索的一个权衡。

想象一下 GPT-3 在这些问题上的表现会有多好，是不是令人兴奋？在本书的最后一章中，我们将简要讨论更多关于 GPT-3 的细节，并介绍一个最近推出的更小但同样值得关注的开源替代品：EleutherAI 的 GPT-Neo。它已经可以在 transformers 库中使用，并且可以通过将 `model` 字符串设置为 EleutherAI 提供的模型名称之一来直接使用。[¹³]我们还附上了一个伴随笔记本，在其中展示了它在本章练习中的应用。经过检查，你应该会发现它的性能更好，但自然也会有显著更高的成本（最大模型的权重超过 10 GB！）。

在下一章中，我们将讨论 transformers 家族中可能最重要的成员——BERT。

## 总结

+   transformers 架构使用自注意力机制来构建文本的双向上下文。这使得它成为了近期在自然语言处理中占主导地位的语言模型。

+   transformers 允许对序列中的令牌进行独立处理。这比处理顺序的双向 LSTM 实现了更大的并行性。

+   transformers 是翻译应用的不错选择。

+   在训练过程中，生成预训练 transformers 使用因果建模目标。这使得它成为文本生成的首选模型，例如聊天机器人应用。

1. A. Vaswani 等人，“Attention Is All You Need”，NeurIPS（2017）。

2. A. Radford 等人，“通过生成预训练来改善语言理解”，arXiv（2018）。

3. M. E. Peters 等人，“BERT：用于语言理解的深度双向 transformers 的预训练”，NAACL-HLT（2019）。

4. [`github.com/google-research/bert/blob/master/multilingual.md`](https://github.com/google-research/bert/blob/master/multilingual.md)

5. A. Radford 等人，“通过生成预训练来改善语言理解”，arXiv（2018）。

6. A. Vaswani 等人，“Attention Is All You Need”，NeurIPS（2017）。

7. [`github.com/jessevig/bertviz`](https://github.com/jessevig/bertviz)

8. [`huggingface.co/Helsinki-NLP`](https://huggingface.co/Helsinki-NLP)

9. [`opus.nlpl.eu/JW300.php`](http://opus.nlpl.eu/JW300.php)

10. [`marian-nmt.github.io/`](https://marian-nmt.github.io/)

11. A. Radford 等人，“通过生成式预训练提高语言理解能力”，*arXiv*（2018）。

12. Y. Zhang 等人，“DialoGPT：面向对话回应生成的大规模生成式预训练”，*arXiv*（2019）。

13. [`huggingface.co/EleutherAI`](https://huggingface.co/EleutherAI)
