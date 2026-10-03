
如你所见，较短且更简洁的答案比更长或更啰嗦的回答获得了更高的（负得较少的）序列级对数概率。唯一例外是最后一个答案，它获得了最低分。请注意，这是唯一一个包含错误数字答案值（`38` 而不是 `83`）的答案，因此这也符合直觉。

总的来说，这说明了求和的对数概率如何自然地倾向于简洁的输出，这与它们在 GRPO 中于序列级别应用奖励和优势时所起的作用一致。

## 6.9 通过 GRPO 损失从优势到策略更新

我们现在将实现 GRPO 流水线的第五个阶段，在该阶段中，先前计算的优势和对数概率被组合成策略梯度损失。我们将在稍后作为训练循环的一部分实现后续的权重更新（阶段 6），如图 6.17 所示。

![](img/CH06_F17_Raschka2.png)

**图 6.17** GRPO 流水线的第五个阶段计算策略梯度损失，我们将用它来更新模型。阶段 6（模型权重更新）将在稍后作为训练循环的一部分实现。

首先，我们将把上一节中的输出序列对数概率列表转换为 PyTorch 张量。然后，我们将通过将每个输出序列的序列级对数概率乘以其对应的优势，在所有输出序列上取平均，并应用负号来计算策略梯度损失。由于 PyTorch 优化器被定义为*最小化*损失，因此天然以最大化问题形式书写的目标必须翻转符号。

**代码清单 6.13** 计算策略梯度损失

```
logps = torch.stack(rollout_logps)
pg_loss =-(advantages.detach() * logps).mean()
print(logps)
print(pg_loss)
```

这将打印出以下对数概率：

```
tensor([
-7.9243, -20.1546, -16.6130, -23.3677],
grad_fn=<StackBackward0>)
```

得到的策略梯度损失值如下：

```
tensor(-2.5764, grad_fn=<NegBackward0>)
```

请注意，我们对`advantages`使用了`.detach()`，因为它们在策略更新期间被视为固定的学习信号。这可以防止梯度通过优势计算回流，确保只有影响对数概率的模型参数得到更新。

在策略梯度方法中，我们希望最大化输出序列的优势加权对数概率，因为高优势输出序列的更高对数概率会改进策略。通过将该目标乘以 −1，我们将最大化问题转化为等价的`最小化`问题。最小化负目标产生的参数更新与最大化原始目标完全相同，同时还与 PyTorch 的优化器实现兼容。

### 最大化与最小化

为了用具体例子说明符号翻转，假设我们要最大化的目标是简单的标量值`𝑓(𝑥) = 3`。

在最大化视角下，值越大越好，因此 3 优于 2。PyTorch 优化器是最小化的，所以它们会试图*减小*这个值，这与我们的期望相反。

现在，假设我们翻转符号并将损失定义为`𝐿(𝑥) = –𝑓(𝑥) = –3`。最小化`𝐿(𝑥)`现在等价于最大化`𝑓(𝑥)`。例如，如果`𝐿(𝑥)`从 −2 减小到 −3，那么`𝑓(𝑥)`从 2 增大到 3。

### 策略梯度损失的数学定义

如果你觉得通过数学符号更容易理解这些计算，那么 GRPO 中使用的策略梯度损失可以写成如下形式：

![公式图像](img/eq-chapter-6-svg-eq-209.png)

这里，

- `N`表示输出序列的数量。

- `𝑦𝑡(𝑖), …, 𝑦𝑇ᵢ(𝑖)`是第`i`个生成长度为`𝑇𝑖`的响应的标记。

- `𝑦<𝑡(𝑖)`表示该响应中所有先前生成的标记。

- `𝑥(𝑖)`是对应的输入提示。

- `𝐴𝑖`是整个输出序列的优势。

内层求和计算一个输出序列的序列级对数概率，而外层平均则计算所有输出序列的优势加权对数概率。

## 6.10 将一切整合到一个 GRPO 函数中

我们现在已经完成了本章最具挑战性的部分，并逐一介绍了每个 GRPO 阶段。接下来，我们将把图 6.18 所示的五个 GRPO 阶段组合成一个`compute_grpo_loss`函数，稍后将在完整训练循环中使用它。（请注意，这里的“阶段”一词指的是 GRPO 损失计算的概念组成部分，基于我们逐步讲解 GRPO 算法的方式；它并不是外部优化循环中的训练步骤。）

![](img/CH06_F18_Raschka2.png)

**图 6.18** 完整的 GRPO 工作流程，其中（1）为一个提示生成多个输出序列，（2）为它们分配正确性奖励，（3）将它们转换为组相对优势，并且（4）将它们与对数概率结合，以（5）计算策略梯度损失。损失梯度（6）将在下一节中计算并用于更新模型。

下面的代码清单展示了`compute_grpo_loss`函数，它结合了我们之前讨论的图 6.18 中的所有阶段。

**代码清单 6.14** 组合所有 GRPO 阶段

```
def compute_grpo_loss(
    model,
    tokenizer,
    example,
    device,
    num_rollouts=2,
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.9,
):
    assert num_rollouts >= 2
    roll_logps, roll_rewards, samples = [], [], []
    prompt = render_prompt(example["problem"])

    was_training = model.training
    model.eval()

    for _ in range(num_rollouts):

        token_ids, prompt_len, text = sample_response(#1
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        reward = reward_rlvr(text, example["answer"])#2

        logp = sequence_logprob(model, token_ids, prompt_len)#3

        roll_logps.append(logp)
        roll_rewards.append(reward)
        samples.append(
            {
                "text": text,
                "reward": reward,
                "gen_len": token_ids.numel() - prompt_len,
            }
        )

    if was_training:
        model.train()

    rewards = torch.tensor(roll_rewards, device=device)#4

    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)#5

    logps = torch.stack(roll_logps)#6

    pg_loss = -(advantages.detach() * logps).mean()#7
    loss = pg_loss  # In the next chapter we add a KL term here

    return {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "rewards": roll_rewards,
        "advantages": advantages.detach().cpu().tolist(),
        "samples": samples,
        "loss_tensor": loss,
    }
```

- `#1` 阶段 1：生成输出序列

- `#2` 阶段 2.1：计算奖励

- `#3` 阶段 4.1：计算对数概率

- `#4` 阶段 2.2：收集所有奖励

- `#5` 阶段 3：计算优势

- `#6` 阶段 4.2：收集所有对数概率

- `#7` 阶段 5：计算策略梯度损失

`compute_grpo_loss`函数相对不言自明，因为它遵循我们在前几节中手动实现的相同阶段。但请注意，在第 1 阶段之后，代码直接进入阶段 2 和 4，而不是阶段 3 和 4。这纯粹是一种实现选择，通过避免多个嵌套的`for`循环来简化代码结构，同时保持相同的操作逻辑顺序。

还要注意，`pg_loss`（策略梯度损失）与我们的示例中的整体损失相同。我在这里有意省略了 KL 损失项；为了完整性，我们将在第 7 章中添加它。如前所述，GRPO 在省略 KL 损失项时通常在数学问题上表现更好，这就是我们在此采用这个简化目标的原因。

接下来，让我们在 MATH 训练数据集中的一个示例上尝试新的`compute_grpo_loss`函数，以确保它能正常工作。出于测试目的，我们只采样少量轨迹（`num_rollouts=2`），并生成相对较少的 token（`max_new_tokens=256`）。该函数仍可能需要几秒钟才能返回结果。

### 清单 6.15 在 MATH 训练示例上计算 GRPO 阶段

```
torch.manual_seed(123)

stats = compute_grpo_loss(
    model=model,
    tokenizer=tokenizer,
    example=math_train[4],
    device=device,
    num_rollouts=2,
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.9
)

pprint(stats)
```

输出如下：

```
{'advantages': [0.0, 0.0],
 'loss': -0.0,
 'loss_tensor': tensor(-0., grad_fn=<NegBackward0>),
 'pg_loss': -0.0,
 'rewards': [0.0, 0.0],
 'samples': {'gen_len': 4, 'reward': 0.0, 'text': ' 14<|endoftext|>'}, 
             {'gen_len': 256,
              'reward': 0.0,
              'text': ' 4\n'
                      'To solve the problem, let's break it down step by step'
                      '...'}
```

正如你从`'samples'`字段中的`'rewards'`和`'text'`条目所看到的，模型回答错误。如第 6.4 节所讨论的，正确的答案必须包含`\boxed{6}`。由于不满足此条件，奖励为零，这反过来导致零优势和零损失。在这种情况下，如果我们正在训练模型，梯度将为零，模型参数将不会被更新。

## 6.11 实现 GRPO 训练循环

我们现在已经具备了实现完整 GRPO 训练循环所需的全部组件。在本章的最后一节中，我们将把所有内容整合起来，使用 GRPO 算法通过 RLVR 训练模型，如图 6.19 所示。

![#### 图 6.19 我们现在将实现训练循环来更新模型权重。训练循环由八个主要阶段组成，如图 6.20 所示。这些阶段中的大多数是典型的 PyTorch 深度神经网络（包括 LLM）训练循环的标准组成部分。唯一针对推理模型特有的阶段是阶段 4，在该阶段中，我们计算 GRPO 损失，而不是许多其他类型的深度神经网络和 LLM 预训练中使用的标准分类损失。![](img/CH06_F20_Raschka2.png)

### 图 6.20 训练循环概览。整体结构遵循标准的深度学习训练循环。关键区别在于损失的计算方式；损失不是通过标准监督目标获得的，而是通过 GRPO 阶段（阶段 4）获得的。

在逐一讨论这八个阶段之前，先看看它们在代码中是如何实现的会很有帮助。清单 6.16 展示了图 6.20 所示的完整训练循环。

### 清单 6.16 实现 RLVR 训练循环
