
```py
$ paste data/chatbot/selfdialog.train.tok.fr data/chatbot/selfdialog.train.tok.en | head 
...
Have you played in a band ?    What type of band ?
What type of band ?    A rock and roll band .
A rock and roll band .    Sure , I played in one for years .
Sure , I played in one for years .    No kidding ?
No kidding ?    I played in rock love love .
I played in rock love love .    You played local ?
You played local ?    Yes
Yes    Would you play again ?
Would you play again ?    Why ?
...
```

如您所见，每一行都包含一个话语（在左侧）和对其的回应（在右侧）。请注意，此数据集与我们在第 6.3.1 节中使用的西班牙语-英语平行语料库具有相同的结构。下一步是运行`fairseq-preprocess`命令将其转换为二进制格式，如下所示：

```py
$ fairseq-preprocess \
    --source-lang fr \
    --target-lang en \
    --trainpref data/chatbot/selfdialog.train.tok \
    --validpref data/chatbot/selfdialog.valid.tok \
    --destdir data/chatbot-bin \
    --thresholdsrc 3 \
    --thresholdtgt 3
```

再次，这与我们为西班牙语翻译器示例运行的内容类似。只需注意您指定的源语言—我们在这里使用的是`fr`而不是`es`。

### 6.6.3 训练和运行聊天机器人

现在，聊天机器人的训练数据准备好了，让我们从这些数据中训练一个`Seq2Seq`模型。您可以像下面这样调用`fairseq-train`命令，几乎使用与上次相同的参数：

```py
$ fairseq-train \
    data/chatbot-bin \
    --arch lstm \
    --share-decoder-input-output-embed \
    --optimizer adam \
    --lr 1.0e-3 \
    --max-tokens 4096 \
    --save-dir data/chatbot-ckpt
```

与先前一样，请注意验证损失随每个时代的变化。当我尝试时，验证损失在约五个时代内减小，然后开始缓慢上升。在观察到验证损失趋于稳定后，随时按`Ctrl + C`停止训练命令。`Fairseq`将自动将最佳模型（按验证损失测量）保存为`checkpoint_best.pt`。

最后，您可以通过调用`fairseq-interactive`命令来运行聊天机器人模型，如下所示：

```py
$ fairseq-interactive \
    data/chatbot-bin \
    --path data/chatbot-ckpt/checkpoint_best.pt \
    --beam 5 \
    --source-lang fr \
    --target-lang en
```

与前面的示例一样，您可以输入源句并与聊天机器人对话，让它将它们“翻译”为另一种语言！以下是我与我训练的模型进行的一部分对话（为了清晰起见，我添加了粗体）。再次强调，您的结果可能不同：

```py
Hi !
S-0     Hi !
H-0     -1.011711835861206      Hey .
P-0     -0.6860 -2.1352 -0.2140
How are you ?
S-1     How are you ?
H-1     -0.6552960276603699     Doing good , you ?
P-1     -0.6472 -0.8651 -1.0960 -0.9257 -0.3337 -0.0640
Good .
S-2     Good .
H-2     -0.8540937900543213     Do you have a favorite team ?
P-2     -2.3027 -0.0550 -1.7088 -0.4403 -0.1763 -2.0262 -0.1150 -0.0085
Yeah I like the Seahawks .
S-3     Yeah I like the Seahawks .
H-3     -0.6036218404769897     Do you have a favorite team ?
P-3     -2.7614 -0.0525 -1.0905 -0.3178 -0.3213 -0.1049 -0.1701 -0.0106
How about you ?
S-4     How about you ?
H-4     -1.2329542636871338     I 'm a big fan of them .
P-4     -0.7648 -1.8435 -1.6808 -1.2868 -0.2102 -0.1362 -3.5273 -1.0031 -0.6438
What 's your favorite movie ?
S-5     What 's your favorite movie ?
H-5     -0.8238655924797058     Oh , that 's a tough one . I 'd have to say Ghost Busters .
P-5     -3.2622 -0.3213 -1.4815 -0.1012 -0.3417 -0.3069 -0.2432 -0.6954 -1.0858 -2.3246 -0.4296 -0.0039 -0.0905 -2.6429 -0.1676 -0.3853 -0.1221
Oh yeah , I like that movie too .
S-6     Oh yeah , I like that movie too .
H-6     -1.0659444332122803     Yeah , that 's a good one .
P-6     -2.0782 -0.1724 -2.5089 -0.9560 -1.5034 -0.9040 -0.4471 -0.5435 -0.4801
```

在这个例子中，对话看起来很自然。因为自我对话语料库是通过限制可能的对话主题集来构建的，所以如果您保持在这些主题（电影、体育、音乐等）上，对话更有可能顺利进行。

然而，一旦您开始谈论不熟悉的话题，聊天机器人就会对其答案失去信心，如下所示：

```py
What 's your name ?
S-0     What 's your name ?
H-0     -0.9807574152946472 I do n't know , but I do n't think I 've                           heard of them .
P-0     -1.4680 -2.2938 -0.0635 -1.0242 -1.2039 -0.5201 -0.3381 -2.2612 -0.1665 -1.6033 -0.6112 -1.5515 -0.8739 -0.8453 -1.0143 -0.4702 -0.3637
What do you do ?
S-1     What do you do ?
H-1     -1.170507788658142 I do n't know .
P-1     -0.9807 -2.1846 -0.3276 -0.9733 -1.3650 -1.1919
Are you a student ?
S-2     Are you a student ?
H-2     -0.9505285024642944 I 'm not sure .
P-2     -1.5676 -1.5270 -0.6944 -0.2493 -0.8445 -0.8204
```

这是一个众所周知的现象—一个简单的基于`Seq2Seq`的聊天机器人很快就会退化到生成“我不知道”和“我不确定”等模板回答，每当被问及它不熟悉的事物时。这与我们训练这个聊天机器人的方式有关。因为我们训练模型是为了使其在训练数据中最小化损失，它能采取的最佳策略是生成适用于尽可能多的输入句子的东西。非常通用的短语，比如“我不知道”，可以成为许多问题的答案，所以这是一个安全的策略，可以减少损失！

### 6.6.4 后续步骤

虽然我们的聊天机器人可以为许多输入产生逼真的响应，但它离完美还有很远的路要走。它处理不好的一个问题是专有名词。当你询问需要具体答案的问题时，就会看到这一点，比如：

```py
What 's your favorite show ?
S-0     What 's your favorite show ?
H-0     -0.9829921722412109     I would have to say <unk> .
P-0     -0.8807 -2.2181 -0.4752 -0.0093 -0.0673 -2.9091 -0.9338 -0.3705
```

这里，`<unk>`是未知词的通用特殊符号。聊天机器人正在尝试回答某些问题，但该问题在训练数据中出现的频率太低，以至于不能被视为独立的词。这是一般简单`NMT`系统中出现的问题。因为模型需要将有关单词的所有信息压缩成 200 多维的数字向量，所以许多细节和类似单词之间的区别都被牺牲了。想像一下将关于你所在城市所有餐厅的信息压缩成一个 200 维的向量！

此外，我们训练的聊天机器人没有任何 "记忆" 或任何上下文概念。你可以通过问一系列相关问题来测试这一点，如下所示：

```py
Do you like Mexican food ?
S-0     Do you like Mexican food ?
H-0     -0.805641770362854      Yes I do .
P-0     -1.0476 -1.1101 -0.6642 -0.6651 -0.5411
Why do you like it ?
S-1     Why do you like it ?
H-1     -1.2453081607818604     I think it 's a great movie .
P-1     -0.7999 -2.1023 -0.7766 -0.7130 -1.4816 -2.2745 -1.5750 -1.0524 -0.4324
```

在第二个问题中，聊天机器人在理解上下文方面有困难，并产生了完全无关的响应。要正确回答这样的问题，模型需要理解代词 "it" 指的是前面的名词，即本例中的 "Mexican food"。`NLP`系统在现实世界中解决哪些提及指向哪些实体的任务被称为*共指消解*。系统还需要维护某种类型的记忆，以跟踪到目前为止在对话中讨论了哪些内容。

最后，在本章中讨论的简单`Seq2Seq`模型在处理长句子方面并不擅长。如果您回顾一下图 6.2，就会理解这一点 - 模型使用`RNN`读取输入语句，并使用固定长度的句子表示向量表示有关句子的所有内容，然后从该向量生成目标语句。无论输入是“Hi！”还是“The quick brown fox jumped over the lazy dog.”，句子表示都会成为瓶颈，特别是对于更长的输入。因此，在 2015 年左右，直到发明了一种称为 *注意力* 的机制来解决这个问题之前，神经`MT`模型无法击败传统的基于短语的统计`MT`模型。我们将在第八章详细讨论注意力。

## 概括

+   `Seq2Seq`模型使用编码器和解码器将一个序列转换为另一个序列。

+   你可以使用`fairseq`框架在一小时内构建工作中的翻译系统。

+   `Seq2Seq`模型使用解码算法生成目标序列。贪心解码每一步都最大化概率，而束搜索则尝试同时考虑多个假设来寻找更好的解决方案。

+   用于自动评估翻译系统的一个指标叫做`BLEU`。

+   通过使用`Seq2Seq`模型和对话数据集，可以构建一个简单的聊天机器人。

^(1.) 详细信息请参阅 Oriol Vinyals 等人的“Grammar as a Foreign Language”（2015 年；[`arxiv.org/abs/1412.7449`](https://arxiv.org/abs/1412.7449)）。

^(2.) 请注意每行开头的`$`是由 shell 渲染的，您无需输入它。
