
`format_reward = reward_format(token_ids, prompt_len)` # NEW#2

`reward = rlvr_reward + format_reward_weight * format_reward` # NEW #2

# ...

`#1` 一个设置参数，用于调节格式奖励相对于现有正确性奖励的强度。  
`#2` 新增的格式奖励代码行，将之前的奖励修改为同时包含正确性奖励和格式奖励。

此外，我们还可以修改 `render_prompt`，引导模型显式生成这些 `` 词元。

##### 清单 7.14 针对 ` tags.\n"

```
        "Then write the final result on a new line in the exact format:\n"
        "\\boxed{ANSWER}\n\n"
        f"Question:\n{prompt}\n\nAnswer:"
    )
    return template
```

理论上，清单 7.11 到 7.14 中的修改应当足以训练一个带有额外格式奖励的推理模型。现在，我们来看看它在实践中的效果，并讨论一些额外的注意事项。

### 7.6.2 训练模型输出 `` 词元（如清单 7.14 所示），结果会很差，因为基础模型从未见过这些词元。引入这些新词元的正确方式，是先在与 `` 词元相关的数据上对模型进行预训练或指令微调（指令微调将在下一章介绍）。

在本章中，由于我们聚焦于强化学习，我们将训练现有的模型推理变体，它已经熟悉 ` thinking` 词元。这与我们在第 3 章中介绍的推理变体相同。

你可以按如下方式运行该脚本：

```
uv run 7_6_plus_format_reward.py \
    --steps 500 \
    --max_new_tokens 1024
```

注意，该脚本已配置为使用推理模型变体，而不是基础模型。

你可以下载本次运行的日志文件做进一步分析，因为运行该实验可能消耗大量资源且成本较高：

```
download_from_github(
    "ch07/02_logs/7_6_plus_format_reward_metrics.csv"
)
plot_grpo_metrics(
    "7_6_plus_format_reward_metrics.csv",
    columns=["loss", "reward_avg", "avg_response_len", "eval_acc"]
)
```

得到的图如图 7.19 所示。可以看到，模型一开始有所改进，但随后在评估准确率方面变差，而平均奖励大致保持恒定。评估准确率的下降似乎与之后模型生成的响应长度变短有关。

![](img/CH07_F19_Raschka2.png)

##### 图 7.19 使用格式奖励进行 GRPO 训练运行的基本指标

为了进一步探究，我们来看一些附加指标：

```
plot_grpo_metrics(
    "7_6_plus_format_reward_metrics.csv",
    columns=["reward_avg", "format_reward_avg", "adv_std", "entropy_avg"],
)
```

结果如图 7.20 所示。模型性能下降的一个可能解释是：模型仅仅因为遵循 ` thinking` 标记格式就获得了过多奖励，如 `format_reward_avg` 图所示。换句话说，模型没有充分关注答案的正确性（这反映在 `eval_acc` 下降，而 `reward_avg` 仍然很高）。

![](img/CH07_F20_Raschka2.png)

##### 图 7.20 使用格式奖励进行 GRPO 训练运行的附加指标

解决这个问题的一种方法是降低格式奖励的强度，例如将默认的 `format_reward_weight` 从 `1.0` 降到 `0.1`。另一种方法是让格式奖励变为条件式，即只在答案也正确时才施加该奖励。在当前设置中，无论答案正确与否，模型都会获得格式奖励。

#### 关于 `loss` 和 `kl_loss` 的说明

本次实验使用了 `7_6_plus_format_reward.py`，其默认设置为 `--kl_coeff 0.0` 和 `--inner_epochs 1`。由于 `--kl_coeff` 被设置为 `0.0`，KL 惩罚被禁用，因此整个运行过程中记录的 `kl_loss` 一直为 `0`。

如果你想使用与第 7.5 节相同的设置启用 KL 正则化，请运行以下命令：

```
uv run 7_6_plus_format_reward.py \
    --steps 500 \
    --max_new_tokens 1024 \
    --kl_coeff 0.001 \
    --inner_epochs 2
```

在默认的 7.6 设置下，记录的 `loss`（图 7.19）也为 0。这是因为该运行仅使用一次内部更新（`--inner_epochs 1`），并将模型与同一步骤中自身的副本进行比较。因此，策略比率初始为 1，又因为优势被归一化为均值为 0，所以标量策略损失计算结果为 0。

因此，在这种情况下，`loss = 0` 是由默认设置导致的计算假象。它主要反映了该运行使用单次内部 epoch 且没有 KL 项，这使得记录的标量损失不具有参考意义。

这里更有用的监控指标是诸如 `reward_avg`、`format_reward_avg`、`adv_std` 和 `entropy_avg` 等。

#### 练习 7.2：让格式奖励变为条件式

修改格式奖励的计算方式：

```
reward = rlvr_reward + format_reward_weight * format_reward
```
