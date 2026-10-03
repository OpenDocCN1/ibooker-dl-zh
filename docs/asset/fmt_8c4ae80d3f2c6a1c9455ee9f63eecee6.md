
运行以下命令来评估模型：

```
uv run evaluate_math500.py \
--dataset_size 500 \
--which_model base \
--checkpoint_path "qwen3-0.6B-rlvr-grpo-step00050.pth"
```

如果你不是 `uv` 用户，请将 `uv run` 替换为 `python`。

表 6.1 将原始基础模型和推理模型（第 1 行和第 2 行）与我们在本章中通过 GRPO 训练的基础模型（第 3–5 行）进行了比较。

**表 6.1** 不同基础模型和推理模型在 MATH-500 任务上的准确率

| | 方法 | 步数 | 最大 token 数 | rollout 数量 | 准确率 | 平均 token 数 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 基础模型（第 3 章） | - | - | - | 15.2% | 78.85 |
| 2 | 推理模型（第 3 章） | - | - | - | 48.2% | 1369.79 |
| 3 | GRPO（本章） | 50 | 256 | 4 | 43.2% | 560.22 |
| 4 | GRPO（本章） | 50 | 512 | 4 | 45.6% | 579.81 |
| 5 | GRPO（本章） | 50 | 512 | 8 | 47.4% | 586.11 |

表 6.1 中的准确率列指的是我们在第 3 章中使用的完整 500 样本 MATH-500 数据集上的准确率。

最大 token 数列标识训练期间每个 rollout 允许的 token 数量。如果该数值为 512，这会鼓励 LLM 在 512 个 token 的限制内给出最终的 `boxed` 答案；否则，它在训练期间将不会获得奖励。评估代码以 2,048 的最大 token 限制运行，从而允许 LLM 在评估期间生成更长的响应。

平均 token 数列显示了在 MATH-500 数据集上评估时的平均响应长度。如你所见，与参考推理模型（第 2 行）相比，我们的 GRPO 模型平均生成的响应更短，这符合预期，因为训练期间存在 token 长度限制。

请注意，DeepSeek-R1 团队观察到，随着模型开始编写中间（思维链）解释，响应在训练过程中会变长。为了最大化准确率，在训练期间不宜过于激进地限制 token 长度。较长的 token 长度，尤其是在与多次 rollout 结合时，需要更多的计算资源，这就是我们在本章中将它们限制在较低值的原因。

如表 6.1 所示，通过 GRPO 训练推理模型可获得 47.4% 的准确率（第 5 行），这接近官方 Qwen3 0.6B 推理模型在 MATH-500 上的准确率。模型的准确率可以通过增加响应长度、每个提示采样更多 rollout 以及进一步训练更多步数来继续提高。下一章将介绍用于监控和改进 GRPO 训练结果的实用技巧与方法。

我们仅训练了 50 步，这似乎已经足以在基础模型中解锁推理行为。在我的实验中，训练更长时间不一定能提高性能，甚至可能降低准确率，因为原始的 GRPO 公式在更长的运行中可能不稳定。

另外值得记住的是，本章实现了一个没有 KL 正则化项的简化 GRPO 变体。在下一章中，我们将回到完整的 GRPO 目标，重新加入 KL 项，并讨论使训练在更长运行中更稳定的常见改进。

如果你感兴趣，可以在 [`huggingface.co/rasbt/qwen3-from-scratch-grpo-checkpoints/tree/main/grpo_original_no_kl`](https://huggingface.co/rasbt/qwen3-from-scratch-grpo-checkpoints/tree/main/grpo_original_no_kl) 找到并下载从 50 到 9,000 的额外检查点。

## 本章小结

* 强化学习（RL）可用于在人类偏好标签和可验证奖励上训练 LLM。
* RL 通常作为预训练基础模型之上的后训练阶段应用，也可以插入 LLM 流程的不同阶段，包括推理训练和偏好微调。
* 基于人类反馈的强化学习（RLHF）通过两阶段设置来优化人类偏好：先从排序后的响应训练一个奖励模型，然后使用奖励分数更新 LLM。
* 基于可验证奖励的强化学习（RLVR）通过将学习得到的奖励模型替换为确定性、自动计算的验证器（例如，数学答案检查），从而简化了 RLHF。
* GRPO 是一种策略优化算法，它将验证器奖励转化为参数更新。由于 GRPO 直接使用序列级奖励优化模型，而无需单独的价值模型，因此特别方便。
* GRPO 是用于 LLM 的其他 RL 算法的一种更节省资源的替代方案，因为它避免了训练单独的价值模型，而是从一组采样 rollout 的内部比较中推导学习信号。
* "rollout" 指的是模型对提示词的完整回答（补全）；在后续步骤中，奖励、优势和 log 概率都是从该 rollout 计算得出的。
* 奖励由验证器计算，只有当最终答案既正确又能以所需格式（如 `"\boxed{}"`）提取时，验证器才给予奖励。
* 原始奖励通过将每个 rollout 的奖励相对于组均值和标准差进行归一化，转化为优势值。
* GRPO 还依赖于序列级 log 概率，这些概率通过对生成的答案 token 的 token log 概率求和得到。
* 序列 log 概率与优势值一起构成了 GRPO 中核心的策略梯度目标。
* 完整的 GRPO 损失计算被合并到一个函数中，该函数执行 rollout 采样、奖励计算、优势计算、log 概率计算和策略梯度损失计算。
* 其周围的训练循环是一个标准的深度学习循环，主要区别在于损失来自 GRPO 而不是传统的分类损失。
* 训练非常消耗资源，因为每一步都需要生成多个可能很长的 rollout，但即使是短暂的 GRPO 运行也能将 MATH-500 的准确率从 15% 提高到 47%。

# 7 改进 GRPO 以用于强化学习

## 本章涵盖

* 解读训练曲线和评估指标
* 防止模型利用奖励信号
* 用额外的响应格式奖励扩展任务正确性奖励

此前，我们端到端地实现了用于基于可验证奖励的强化学习（RLVR）的 GRPO 算法。现在，如图 7.1 所示，我们将从该基线出发，重点关注当我们运行更长时间的训练时会发生什么。

![](img/CH07_F01_Raschka2.png)

**图 7.1** 本书所涵盖主题的图示。本章将更深入地介绍用于基于可验证奖励的强化学习的 GRPO 算法。

具体来说，我们将讨论哪些指标值得跟踪（除了奖励和准确率之外），如何及早发现失败模式，以及为什么即使在代码"正确"的情况下训练也可能变得不稳定。事实证明，基础 GRPO 可能导致训练不稳定，因此本章还将介绍推理模型训练中使用的实用 GRPO 扩展和修复方法。

## 7.1 改进 GRPO

我们在上一章中实现了 GRPO（组相对策略优化，Group Relative Policy Optimization）；现在我们将重新审视训练过程并更深入地分析它。我们还将重新审视上一章中省略的 KL 损失项，并讨论一些在实际训练运行中变得重要的实用技巧和算法选择。这些主题总结在图 7.2 的本章概览中。

![](img/CH07_F02_Raschka2.png)

**图 7.2** 本章概览，展示了本章涵盖的不同主题

本章中的示例基于实际实验，但结果应谨慎解读。要想就个别设置的效果得出有力结论，每个实验都需要重复多次并对结果取平均，因为采样和优化中的随机性可能导致不同运行之间出现明显差异。即便如此，这里展示的示例也足以说明主要思想和相关的权衡取舍。

**注意：** 本章侧重于使用 GRPO 的额外技术细节和实际考虑。如果你觉得本章内容过于技术化，可以跳过，因为下一章并不依赖本章内容。

## 7.2 跟踪 GRPO 性能指标

在第 6 章中，我们运行了一个简短的 GRPO 训练循环，并简要检查了结果。例如，一次简短运行的输出结构如下：

```
[Step 1/50] loss=-0.0000 reward_avg=0.000 avg_resp_len=5.5
[Step 2/50] loss=-0.0000 reward_avg=0.000 avg_resp_len=6.8
[Step 3/50] loss=0.3592 reward_avg=0.250 avg_resp_len=7.8
[Step 4/50] loss=2.7401 reward_avg=0.250 avg_resp_len=56.5
[Step 5/50] loss=3.3214 reward_avg=0.500 avg_resp_len=251.2
# ...
```

让我们从这里继续，讨论使用 GRPO 训练时应跟踪哪些指标，以及它们如何帮助解释和调试训练行为。

### 7.2.1 执行一次 GRPO 训练运行

第 6 章中的代码旨在实现完整的 GRPO 流程与保持实现紧凑可读之间取得平衡。为了方便起见，本书的补充材料包含一个等效脚本，其中包含相同的代码，并且可以从终端运行（稍后详述）：[`mng.bz/oZxN`](https://mng.bz/oZxN)。

对于本章介绍的每一项 GRPO 改进，我们将使用补充材料中的相关参考脚本，以避免在正文中重复大段代码。清单 7.1 中的辅助函数将从补充材料下载相关脚本并保存到本地，供你在本章中使用。这样我们就可以避免重复冗长的代码段落，并把重点放在 GRPO 实现的主要变化上。

**清单 7.1** 下载补充材料的辅助函数

```
from pathlib import Path
import requests

def download_from_github(rel_path, out=None):
    github_raw_base = (  #1
        "https://raw.githubusercontent.com/rasbt/"
        "reasoning-from-scratch/refs/heads/main/"
    )

    rel_path = Path(rel_path)

    out = Path(out) if out is not None else Path(rel_path.name)#2

    if out.exists():  #3
        size_kb = out.stat().st_size / 1e3
        print(f"{out}: {size_kb:.1f} KB (cached)")
        return out


    r = requests.get(github_raw_base + rel_path.as_posix())#4
    r.raise_for_status()

    out.write_bytes(r.content)
    size_kb = out.stat().st_size / 1e3
    print(f"{out}: {size_kb:.1f} KB")
```

`#1` 基础 URL

`#2` 使用 URL 文件名作为默认输出文件名。

`#3` 如果文件已存在于本地，则跳过下载。

`#4` 下载该文件。

使用上述辅助函数，你可以按如下方式下载第 6 章中带有 GRPO 训练代码的脚本：

```
download_from_github(
    "ch06/02_rlvr_grpo_scripts_intro/rlvr_grpo_original_no_kl.py"
)
```

你应该看到以下输出：`rlvr_grpo_original_no_kl.py: 13.4 KB`。如果下载的文件比这里列出的大小小得多，则说明下载可能未正确完成。在这种情况下，请仔细检查 URL，并参阅故障排除指南获取更多建议：[`github.com/rasbt/reasoning-from-scratch/blob/main/troubleshooting.md`](https://github.com/rasbt/reasoning-from-scratch/blob/main/troubleshooting.md)。生成的脚本可以从终端运行，如图 7.3 所示。

![](img/CH07_F03_Raschka2.png)

**图 7.3** 终端中 GRPO 训练运行的输出，显示了多项训练统计数据，例如损失、平均奖励、每秒 token 数吞吐量和平均响应长度

你也可以通过在 Jupyter Notebook 的代码单元格中运行训练脚本，方法是加上前缀 `!`，即 `!uv run rlvr_grpo_original_no_kl.py`。

如果你不是 `uv` 用户，请将图 7.3 中显示的 `uv run` 替换为 `python`，如下所示：

```
python rlvr_grpo_original_no_kl.py \
--steps 500 \
--max_new_tokens 1024
```

**提示：** 你可以在前面的代码执行命令中添加 `--show_eta`，以显示脚本在机器上总运行时间的估计值。

如果你不希望自己运行训练代码（鉴于其计算成本，这也是合理的），补充材料提供了本次运行的日志文件。你可以按如下方式下载它们：

```
download_from_github(
    "ch07/02_logs/ch06_rlvr_grpo_original_no_kl_metrics.txt"
)
download_from_github(
    "ch07/02_logs/ch06_rlvr_grpo_original_no_kl_metrics.csv"
)
```

第一个文件以 `.txt` 结尾，是纯文本输出文件，显示与图 7.3 类似的输出统计信息。第二个文件以 `.csv` 结尾，是一个逗号分隔值文件，为这些信息提供了更结构化的格式，从而更容易提取和绘制数据以进行进一步分析。

### 7.2.2 检查 GRPO 训练运行

为了检查训练运行，我们可以使用 Matplotlib 绘制日志文件中的结果。我们将定义一个在本章中反复使用的绘图函数。

**清单 7.2** 用于可视化日志文件训练结果的绘图函数

```
import csv
import matplotlib.pyplot as plt


def moving_average(values, window_fraction=0.25):

    window_size = max(1, int(window_fraction * len(values)))#1
    smoothed = []

    for i in range(len(values)):
        start_idx = max(0, i - window_size + 1)
        window_mean = sum(values[start_idx : i + 1]) / (i - start_idx + 1)
        smoothed.append(window_mean)

    return smoothed


def plot_grpo_metrics(csv_path, columns, save_as=None):
    data = {name: {"steps": [], "values": []} for name in columns}

    with Path(csv_path).open(newline="") as f: #2
        reader = csv.DictReader(f)
        for row in reader:
            if not row or not row.get("step"):
                continue

            step = int(row["step"])  #3

            for name in columns:
                value_str = row.get(name)
                if value_str:
                    data[name]["steps"].append(step)
                    data[name]["values"].append(float(value_str))


    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(6, 4))#4
    axes = axes.ravel()

    for i, name in enumerate(columns):
        steps = data[name]["steps"]
        values = data[name]["values"]

        if not values: #5
            fig.delaxes(axes[i])
            continue

        if name == "eval_acc":  #6
            axes[i].bar(steps, values, width=20)
        else:
            axes[i].plot(steps, values, alpha=0.4)
            axes[i].plot(steps, moving_average(values))

        axes[i].set_ylabel(name)

    for j in (2, 3):
        if axes[j] in fig.axes:
            axes[j].set_xlabel("Step")

    plt.tight_layout()
    if save_as is not None:
        plt.savefig(save_as)
    plt.show()
```

`#1` 平滑嘈杂的训练信号，以揭示训练中的长期趋势。

`#2` 打开并读取 CSV 日志文件。

`#3` 使用训练步数作为所有指标共享的 x 轴。
