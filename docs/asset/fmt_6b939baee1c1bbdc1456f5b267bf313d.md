
# RLVR GRPO 训练循环

## 代码实现

以下是使用 GRPO 算法实现 RLVR 训练循环的完整代码：

```python
import time

def train_rlvr_grpo(
    model,
    tokenizer,
    math_data,
    device,
    steps=None,
    num_rollouts=2,
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.9,
    lr=1e-5,
    checkpoint_every=50,
    checkpoint_dir=".",
    csv_log_path=None,

):
    if steps is None:
        steps = len(math_data)


    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)#1
    model.train()
    current_step = 0

    if csv_log_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_log_path = f"train_rlvr_grpo_metrics_{timestamp}.csv"
    csv_log_path = Path(csv_log_path)

    try:

        for step in range(steps):#2


            optimizer.zero_grad()#3

            current_step = step + 1
            example = math_data[step % len(math_data)]


            stats = compute_grpo_loss(#4
                model=model,
                tokenizer=tokenizer,
                example=example,
                device=device,
                num_rollouts=num_rollouts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )


            stats["loss_tensor"].backward()#5


            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)#6


            optimizer.step()#7


            reward_avg = torch.tensor(stats["rewards"]).mean().item()#8
            step_tokens = sum(
                sample["gen_len"] for sample in stats["samples"]
            )
            avg_response_len = (
                step_tokens / len(stats["samples"]) if stats["samples"] else 0.0
            )
            append_csv_metrics(
                csv_log_path, current_step, steps,
                stats["loss"], reward_avg, avg_response_len,
            )
            print(
                f"[Step {current_step}/{steps}] "
                f"loss={stats['loss']:.4f} "
                f"reward_avg={reward_avg:.3f} "
                f"avg_resp_len={avg_response_len:.1f}"
            )


            if checkpoint_every and current_step % checkpoint_every == 0:#9
                ckpt_path = save_checkpoint(
                    model=model,
                    checkpoint_dir=checkpoint_dir,
                    step=current_step,
                )
                print(f"Saved checkpoint to {ckpt_path}")


    except KeyboardInterrupt:#10
        ckpt_path = save_checkpoint(
            model=model,
            checkpoint_dir=checkpoint_dir,
            step=max(1, current_step),
            suffix="interrupt",
        )
        print(f"\nKeyboardInterrupt. Saved checkpoint to {ckpt_path}")
        return model

    return model


def save_checkpoint(model, checkpoint_dir, step, suffix=""):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"-{suffix}" if suffix else ""
    ckpt_path = (
        checkpoint_dir /
        f"qwen3-0.6B-rlvr-grpo-step{step:05d}{suffix}.pth"
    )
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def append_csv_metrics(#11
    csv_log_path, step_idx, total_steps,
    loss, reward_avg, avg_response_len,
):
    if not csv_log_path.exists():
        csv_log_path.write_text(
            "step,total_steps,loss,reward_avg,avg_response_len\n",
            encoding="utf-8",
        )
    with csv_log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"{step_idx},{total_steps},{loss:.6f},{reward_avg:.6f},"
            f"{avg_response_len:.6f}\n"
        )
```

## 训练循环的阶段说明

1. **阶段 1**：初始化`optimizer`（`model`已在函数外部初始化）
2. **阶段 2**：遍历训练步骤
3. **阶段 3**：重置损失梯度（最佳实践是在每一步开始时执行此操作）
4. **阶段 4**：计算 GRPO 损失
5. **阶段 5**：反向传播以计算损失梯度
6. **阶段 6**：裁剪大梯度以提高训练稳定性
7. **阶段 7**：使用损失梯度更新模型权重
8. **阶段 8**：收集奖励、平均响应长度和损失
9. **阶段 9**：保存模型检查点
10. **阶段 10**：如果我们提前中断训练，保存模型检查点
11. **阶段 11**：用于将结果保存到 CSV 文件的实用函数

## 训练循环概述

该函数使用 GRPO 实现了完整的 RLVR 训练循环。如前所述，除第 4 阶段（GRPO 损失计算）外，其结构镜像了常规的 PyTorch 训练循环：我们重置梯度、反向传播损失、可选地裁剪梯度、通过优化器步骤更新参数、记录指标，并定期保存检查点。（我们保存模型检查点，即模型权重的快照，以便日后加载、评估和使用，我们将在下一节中讨论。）

> **提示：** 有关在 PyTorch 中训练神经网络的通用介绍，请参阅我的文章《PyTorch 一小时：从张量到在多个 GPU 上训练神经网络》的第 3-8 节（[`sebastianraschka.com/teaching/pytorch-1h/`](https://sebastianraschka.com/teaching/pytorch-1h/)）。

与其他标准训练循环相比，这里的关键区别在于损失的构建方式。大多数神经网络（包括分类器和 LLM 预训练）使用的是标准监督目标，而这里的训练信号是通过 GRPO 算法作为 RLVR 过程的一部分来计算的。

## 运行训练

现在，让我们运行代码并训练模型。之前，我们将 PyTorch 设备硬编码为`"cpu"`，以便所有中间结果与书中显示的结果紧密匹配，因为`"mps"`和`"cuda"`设备可能会引入微小的浮点差异。不过，在 CPU 上训练非常慢。

为了方便起见，下面的代码清单中包含了在调用`train_rlvr_grpo()`函数之前自动选择并激活相应设备的两行代码。例如，只需运行下面清单中的代码，它就会自动切换到`"cuda"`或`"mps"`。

**代码清单 6.17 训练模型**

```python
device = get_device()
model.to(device)

torch.manual_seed(0)

train_rlvr_grpo(
    model=model,
    tokenizer=tokenizer,
    math_data=math_train,
    device=device,
    steps=50,
    num_rollouts=4,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
    lr=1e-5,
    checkpoint_every=5,
    checkpoint_dir=".",
)
```

### 参数设置说明

在检查结果之前，我们先简要讨论这些设置。`temperature`和`top_p`的设置处于常见范围内，这有助于鼓励答案的多样性，同时仍能为此模型生成连贯的文本。您可以更改这些设置，观察它们对结果的影响（常用范围是 0.7-0.9）。

步数相对较少，为 50 步，尽管数据集包含 12,000 条记录。这只是为了保持合理的运行时间，并且可以增加。不过，即使只有 50 步，也足以获得良好的结果。

学习率（`lr`）可以调整，但它处于合理范围内，并且效果良好。

rollouts 的数量（`num_rollouts=4`）和允许的 token 数量（`max_new_tokens=512`）相对较小，以减少资源需求。如果您遇到内存不足的错误，可以进一步减小这些值，例如为了测试目的设置`num_rollouts=2`和`max_new_tokens=64`。

使用当前的`checkpoint_every=5`设置，模型每 5 步保存一次，每个检查点大约需要 1.5 GB 的磁盘空间。对于更长时间的运行，我建议将此数字设置为 50 或 100。这里故意设置得较小，以便测试。另请注意，如果训练运行被手动中断（例如通过`Interrupt the kernel`按钮中断 Jupyter notebook 的执行），则应创建一个额外的检查点。

### 运行输出示例

让我们简要看一下运行输出（我在第 7 步中断了 50 步的运行）：

```text
Using Apple Silicon GPU (MPS)
[Step 1/50] loss=-0.0000 reward_avg=0.000 avg_resp_len=88.0 
[Step 2/50] loss=-0.0000 reward_avg=0.000 avg_resp_len=7.5 
[Step 3/50] loss=-0.0000 reward_avg=0.000 avg_resp_len=6.5 
[Step 4/50] loss=0.0909 reward_avg=0.250 avg_resp_len=6.5 
[Step 5/50] loss=1.1001 reward_avg=0.500 avg_resp_len=300.5 
Saved checkpoint to qwen3-0.6B-rlvr-grpo-step00005.pth

KeyboardInterrupt. Saved checkpoint to qwen3-0.6B-rlvr-grpo-step00006-interrupt.pth 
```

您可以看到损失和奖励值都在波动，这在 RL 训练中是正常的。

我们打印了平均奖励，它表示采样响应中正确的比例。使用我们的`num_rollouts=4`设置，`reward_avg`为 0.5 意味着 4 个 rollouts 中有 2 个获得了正奖励。类似地，0.25 和 0.0 分别表示有 1 个或 0 个正确响应。

不应过度解读损失值。`reward_avg=0.000`的步骤会产生接近零的损失，因为所有 rollouts 都获得相同的奖励，导致组相对优势消失，梯度信号很少甚至没有。较大的损失幅度，例如第 3 步（见前面输出的第 4 行），只是反映了 rollouts 之间更大的相对差异，这对于 GRPO 风格的目标函数来说是典型的，尤其是在训练开始时。

理想情况下，我们希望看到两个主要趋势随时间变化：

1. 随着模型学习生成更准确的响应，平均奖励在多个步骤上求平均时应增加。
2. 推理准确性应提高（我们将在下一节中评估这一点）。

本书的在线代码包含该脚本的一个更复杂的版本，其中还包含一个选项，可以定期在 MATH-500 测试数据集的子集上评估模型，以查看模型在目标任务上是否有所改进：[`github.com/rasbt/reasoning-from-scratch/tree/main/ch06/02\_rlvr\_grpo\_scripts\_intro`](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch06/02_rlvr_grpo_scripts_intro)。

### 批量训练

强化学习可能相对资源密集，因为我们必须通过 LLM 为每个训练步骤生成多个（长）rollouts。因此，该实现不支持批处理。

如果您可以使用多个 GPU，您可以使用此代码的可选版本，该版本支持批处理和多个 GPU：[`github.com/rasbt/reasoning-from-scratch/tree/main/ch06/02\_rlvr\_grpo\_scripts\_intro`](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch06/02_rlvr_grpo_scripts_intro)。此版本训练模型更快。

## 6.12 加载和评估保存的模型检查点

现在我们将讨论如何加载上一节中 RLVR 训练运行的保存检查点（见图 6.21）。我们还将使用 MATH-500 数据集评估模型。

![](img/CH06_F21_Raschka2.png)

**图 6.21** 我们现在将加载保存的模型检查点并对其进行评估。

保存的检查点可以使用第 2 章中描述的 PyTorch state dict 方法加载，其中`model_path`指向相应的`.pth`检查点文件：

```python
model.load_state_dict(torch.load("qwen3-0.6B-rlvr-grpo-step00050.pth"))
```

### 下载 RLVR 检查点

如果您不愿意因运行成本而在本地运行 GRPO 训练循环，您可以改为下载我的预训练检查点。这些检查点可以手动获取，也可以直接从 Python 使用以下辅助函数获取，该函数会下载选定的检查点，并使其可用于评估或进一步实验：

```python
from reasoning_from_scratch.qwen3 import download_qwen3_grpo_checkpoints
download_qwen3_grpo_checkpoints(grpo_type="no_kl", step="00050")
```

请注意，此检查点是在与代码清单 6.17 类似的设置下创建的，只是将`num_rollouts=4`更改为`num_rollouts=8`。

这些检查点与第 3 章中介绍的模型评估工具完全兼容，这些工具允许我们使用相同的 MATH-500 验证流程来评估经过强化学习训练的模型。

为方便起见，你可以复用第 3 章补充材料中提供的评估脚本（[`github.com/rasbt/reasoning-from-scratch/blob/main/ch03/02_math500-verifier-scripts/evaluate_math500.py`](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch03/02_math500-verifier-scripts/evaluate_math500.py)），通过指定所需的检查点路径，在 MATH-500 数据集上运行评估。例如，要评估`qwen3-0.6B-rlvr-grpo-step00050.pth`检查点文件，可以按如下方式运行上述脚本：
