
# 设置日志记录

```py
from torch.utils.tensorboard import SummaryWriter
import logging
import wandb

def setup_logging(project_name):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y%H:%M:%S", level=logging.INFO, handlers=[
        logging.FileHandler(f"log/debug_{accelerator.process_index}.log"),
        logging.StreamHandler()])
    if accelerator.is_main_process: # We only want to set up logging once
        wandb.init(project=project_name, config=args)
        run_name = wandb.run.name
        tb_writer = SummaryWriter()
        tb_writer.add_hparams(vars(args), {'0': 0})
        logger.setLevel(logging.INFO)
        datasets.utils.logging.set_verbosity_debug()
        transformers.utils.logging.set_verbosity_info()
    else:
        tb_writer = None
        run_name = ''
        logger.setLevel(logging.ERROR)
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    return logger, tb_writer, run_name
```

每个 worker 都会获得一个独特的`accelerator.process_index`，我们将其与`FileHandler`一起使用，将每个 worker 的日志写入单独的文件。我们还使用`accelerator.is_main_process`属性，该属性仅对主 worker 为`true`。我们确保我们不会多次初始化 TensorBoard 和 Weights & Biases 记录器，并且我们降低其他 worker 的日志级别。我们返回自动生成的唯一`wandb.run.name`，稍后我们将用它来命名 Hub 上的实验分支。

# 记录指标

我们还将定义一个函数来使用 TensorBoard 和 Weights & Biases 记录指标。我们再次在这里使用`accelerator.is_main_process`来确保我们只记录一次指标，而不是为每个工作进程记录一次：

```py
def log_metrics(step, metrics):
    logger.info(f"Step {step}: {metrics}")
    if accelerator.is_main_process:
        wandb.log(metrics)
        [tb_writer.add_scalar(k, v, step) for k, v in metrics.items()]
```

# 创建数据加载器

接下来，让我们编写一个函数，使用我们全新的`ConstantLengthDataset`类为训练和验证集创建数据加载器：

```py
from torch.utils.data.dataloader import DataLoader

def create_dataloaders(dataset_name):
    train_data = load_dataset(dataset_name+'-train', split="train",
                              streaming=True)
    train_data = train_data.shuffle(buffer_size=args.shuffle_buffer,
                                    seed=args.seed)
    valid_data = load_dataset(dataset_name+'-valid', split="validation",
                              streaming=True)

    train_dataset = ConstantLengthDataset(tokenizer, train_data,
                                          seq_length=args.seq_length)
    valid_dataset = ConstantLengthDataset(tokenizer, valid_data,
                                          seq_length=args.seq_length)

    train_dataloader=DataLoader(train_dataset, batch_size=args.train_batch_size)
    eval_dataloader=DataLoader(valid_dataset, batch_size=args.valid_batch_size)
    return train_dataloader, eval_dataloader
```

最后，我们将数据集包装在`DataLoader`中，它还处理批处理。![nlpt_pin01](img/nlpt_pin01.png) Accelerate 将负责将批次分发给每个工作进程。

# 参数分组

我们需要实现的另一个方面是优化。我们将在主循环中设置优化器和学习率调度，但在这里我们定义一个辅助函数来区分应接收权重衰减的参数。一般来说，偏置和 LayerNorm 权重不受权重衰减的影响：

```py
def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [{'params': params_with_wd, 'weight_decay': args.weight_decay},
            {'params': params_without_wd, 'weight_decay': 0.0}]
```

# 评估函数

最后，我们希望不时在验证集上评估模型，因此让我们添加一个评估函数，我们可以调用它来计算评估集上的损失和困惑度：

```py
def evaluate():
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch, labels=batch)
        loss = outputs.loss.repeat(args.valid_batch_size)
        losses.append(accelerator.gather(loss))
        if args.max_eval_steps > 0 and step >= args.max_eval_steps: break
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = torch.tensor(float("inf"))
    return loss.item(), perplexity.item()
```

困惑度衡量模型的输出概率分布预测目标标记的能力。因此，较低的困惑度对应更好的性能。请注意，我们可以通过对模型输出的交叉熵损失取指数来计算困惑度。特别是在训练开始时，当损失仍然很高时，计算困惑度时可能会出现数值溢出。我们捕获这个错误，并在这些情况下将困惑度设置为无穷大。

# 与 Hugging Face Hub 集成

在将所有内容整合到训练脚本中之前，我们还有一个额外的函数要使用。正如您现在所知，Hugging Face Hub 在内部使用 Git 来存储和版本化模型和数据集。使用`huggingface_hub`库中的`Repository`类，您可以以编程方式访问存储库并进行拉取、分支、提交或推送。我们将在我们的脚本中使用这个功能来在训练过程中不断推送模型检查点到 Hub。

# 主训练脚本

现在我们已经准备好编写训练脚本的核心部分了，因为我们已经有了所有这些辅助函数：

```py
set_seed(args.seed)

# Accelerator
accelerator = Accelerator()
samples_per_step = accelerator.state.num_processes * args.train_batch_size

# Logging
logger, tb_writer, run_name = setup_logging(project_name.split("/")[1])
logger.info(accelerator.state)

# Load model and tokenizer
if accelerator.is_main_process:
    hf_repo = Repository("./", clone_from=project_name, revision=run_name)
model = AutoModelForCausalLM.from_pretrained("./", gradient_checkpointing=True)
tokenizer = AutoTokenizer.from_pretrained("./")

# Load dataset and dataloader
train_dataloader, eval_dataloader = create_dataloaders(dataset_name)

# Prepare the optimizer and learning rate scheduler
optimizer = AdamW(get_grouped_params(model), lr=args.learning_rate)
lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer,
                             num_warmup_steps=args.num_warmup_steps,
                             num_training_steps=args.max_train_steps,)
def get_lr():
    return optimizer.param_groups[0]['lr']

# Prepare everything with our `accelerator` (order of args is not important)
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader)

# Train model
model.train()
completed_steps = 0
for step, batch in enumerate(train_dataloader, start=1):
    loss = model(batch, labels=batch).loss
    log_metrics(step, {'lr': get_lr(), 'samples': step*samples_per_step,
                       'steps': completed_steps, 'loss/train': loss.item()})
    loss = loss / args.gradient_accumulation_steps
    accelerator.backward(loss)
    if step % args.gradient_accumulation_steps == 0:
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        completed_steps += 1
    if step % args.save_checkpoint_steps == 0:
        logger.info('Evaluating and saving model checkpoint')
        eval_loss, perplexity = evaluate()
        log_metrics(step, {'loss/eval': eval_loss, 'perplexity': perplexity})
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained("./")
            hf_repo.push_to_hub(commit_message=f'step {step}')
        model.train()
    if completed_steps >= args.max_train_steps:
        break

# Evaluate and save the last checkpoint
logger.info('Evaluating and saving model after training')
eval_loss, perplexity = evaluate()
log_metrics(step, {'loss/eval': eval_loss, 'perplexity': perplexity})
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
if accelerator.is_main_process:
    unwrapped_model.save_pretrained("./")
    hf_repo.push_to_hub(commit_message=f'final model')
```

这是一个相当大的代码块，但请记住，这是您在分布式基础设施上训练复杂的大型语言模型所需的所有代码。让我们稍微解构一下脚本，并突出显示最重要的部分：

## 模型保存

我们从模型存储库中运行脚本，并在开始时检出一个以来自 Weights & Biases 的`run_name`命名的新分支。稍后，我们在每个检查点处提交模型并将其推送到 Hub。通过这种设置，每个实验都在一个新分支上，每个提交代表一个模型检查点。请注意，我们需要调用`wait_for_everyone()`和`unwrap_model()`来确保在存储模型时模型得到正确同步。

## 优化

对于模型优化，我们使用`AdamW`，并在线性预热期之后使用余弦学习率调度。对于超参数，我们紧密遵循 GPT-3 论文中描述的类似规模模型的参数。⁸

## 评估

我们在每次保存模型时（即每个`save_checkpoint_steps`和训练后）在评估集上评估模型。除了验证损失，我们还记录验证困惑度。

## 梯度累积和检查点

所需的批处理大小即使在最新的 GPU 上运行时也无法适应 GPU 的内存。因此，我们实现了梯度累积，它在多次向后传递中收集梯度，并在累积足够的梯度后进行优化。在第六章中，我们看到了如何使用`Trainer`来实现这一点。对于大型模型，即使一个批次也无法完全适应单个 GPU。使用一种称为*梯度检查点*的方法，我们可以交换一些内存占用量，以换取大约 20%的训练减速。⁹ 这使我们能够将大型模型适应单个 GPU。

可能仍然有点模糊的一个方面是在多个 GPU 上训练模型意味着什么。根据模型的大小和数据量，有几种分布式训练模型的方法。![nlpt_pin01](img/nlpt_pin01.png) Accelerate 使用的方法称为[`DataDistributedParallelism` (DDP)](https://oreil.ly/m4iNm)。这种方法的主要优势在于它允许您使用更大的批量大小更快地训练模型，这些批量大小无法适应任何单个 GPU。该过程在 Figure 10-6 中有图示。

![DDP](img/nlpt_1006.png)
