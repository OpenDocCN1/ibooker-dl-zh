
每个方块的右下角的数字表示有多少个示例落入该单元格（在验证集中）。许多混淆矩阵只是在这里报告计数，但我们还包括实际图像，这样你就可以开始了解不同病变类型的外观。

精确度（`P:`）显示在每个列（每个预测类别）上方，而召回率（`R:`）显示在每个行（每个真实类别）上。

在之前的训练图中报告的加权精确率和召回率并不是这些值的简单平均值——它们是类别加权的，这意味着更常见的类别对整体指标贡献更多。

考虑到这一点，具体解释图 5-17 中的混淆矩阵，我们可以看到：

- 这个模型并不出色——有许多对角线外的条目。
- 模型通常在更常见的类别上表现更好，而在罕见类别上表现较差——我们稍后会尝试重采样来解决这个问题。
- 它在血管病变（右下角）上表现最好，这些病变在视觉上明显且易于识别，即使是未经训练的眼睛也能识别。
- 模型在色素性良性角化病、痣和黑色素瘤上也相对表现良好——尽管还有很大的改进空间。
- 一个关键问题是，黑色素瘤常常被错误地分类为痣或其他良性类别。考虑到检测黑色素瘤的临床重要性，这一点尤其令人担忧。我们将密切关注是否更先进的模型能减少这些错误。

考虑到这个基线，让我们将注意力转向`ResNetFromScratch`模型。

## 训练 `ResNetFromScratch` 模型

记住，这个模型遵循 ResNet 架构，但始于随机初始化的权重——这意味着与使用预训练权重的模型相比，它可能需要更多的时间来达到强大的性能。

在训练之前，我们需要设置数据集和模型实例。我们意识到类别不平衡和数据集大小较小，并且我们已经准备了像`balanced_sampler`和`rich_augmentor`这样的技术来缓解这些问题。然而，为了更好地理解它们各自的影响，我们将首先在没有任何增强或类别平衡的情况下运行训练。

对于这个模型，我们不需要对图像应用`resnet`预处理，因为我们是从零开始训练，并且不需要与预训练的 ResNet 权重兼容。

现在，让我们训练我们的模型：

```py
state, metrics = train(
  state=ResNetFromScratch(num_classes=num_classes).create_train_state(
    rng=rng_init,
    dummy_input=dataset_splits["train"].get_dummy_input(),
    tx=optax.adamw(learning_rate, weight_decay=0.0, mask=decay_mask),
  ),
  rng=rng_train,
  dataset_splits=dataset_splits,
  num_steps=num_steps,
  batch_size=32,
  preprocessor=crop,  # No preprocessing.
  sampler=repeating_sampler,  # No balancing.
  augmentor=None,  # No augmentation.
  eval_every=100,
  store_path=assets("cancer/models/resnet_from_scratch"),
)
```

如前所述，让我们通过绘制模型在训练步骤上的指标来评估模型性能。结果如图 5-18 所示：

```py
plot_learning(metrics);
```

![](img/dlfb_0518.png)

###### 图 5-18. `ResNetFromScratch`多类皮肤病变分类模型在训练和验证步骤中的模型性能。

结果很有希望，并且已经显示出与基线相比的改进——精确率和召回率指标显著提高。这是有道理的，因为 ResNet 架构通常对图像任务很强。

重要的是，训练和验证指标稳步增加，并且保持紧密一致。同样，训练和验证损失随时间下降，表明模型没有显著过拟合。事实上，训练曲线表明模型仍在学习——因此，随着更多训练步骤的增加，我们可能会取得更好的结果。

为了看到哪些类别从这种改进的模型中受益最大，让我们再次查看图 5-19 中的混淆矩阵：

```py
predictions = get_predictions(state, dataset_splits["valid"], resnet, 32)
plot_classified_images(
  predictions, dataset_splits["valid"], crop, max_images=8
);
```

![](img/dlfb_0519.png)

###### 图 5-19. 在九类皮肤病变分类任务中`ResNetFromScratch`模型的混淆矩阵。

一些观察结果：

- 总体改进：与基线模型相比，我们在大多数类别中看到了更好的召回率和精确率。模型通常更加自信和准确。对角线整体上看起来更强，表明有更多的正确预测。
- 血管病变现在几乎被完美分类（右下角）。这些病变在视觉上很独特，模型显然注意到了这一点。
- 痣和黑色素瘤在召回率方面都有显著的提高。然而，正如这些类别之间的非对角线单元格所示，两者之间仍然存在一些混淆。
- 痣性良性角化病在精确度和召回率方面都有显著改善，与基线相比有明显的跳跃。
- 日光性角化病和脂溢性角化病现在永远不会被模型预测——导致这些类别的精确度值为`NaN`。这很奇怪，是未来模型需要重新审视的问题。

我们现在转向`FinetunedHeadResNet`，它结合了预训练的 ImageNet 特征。这应该给我们一个更强的模型，特别是考虑到我们的训练数据集很小。

## 训练 `FinetunedHeadResNet` 模型

这次，我们需要小心图像预处理。由于我们使用的是在 ImageNet 上预先训练的 ResNet 模型的预训练权重，我们的输入图像必须以与用于 ImageNet 的相同方式预处理。

###### 警告

如果图像偏差太大——例如，如果它们缩放不同或归一化不一致——预训练的特征可能不再适用。这可能会迫使模型在训练过程中过度纠正，从而减少迁移学习的益处。

为了处理这个问题，我们应用`resnet`预处理函数，确保输入图像以与预训练的 ResNet 主干期望匹配的方式进行调整大小和归一化：

```py
IMAGE_PROCESSOR = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

def resnet(image: jax.Array) -> jax.Array:
  """Preprocess from pretrained model with transpose for compatibility."""
  image = IMAGE_PROCESSOR(image, return_tensors="jax", do_rescale=True)
  image = image["pixel_values"]
  image = convert_nchw_to_nhwc(image)
  return image
```

我们现在准备训练`FinetunedHeadResNet`模型，该模型冻结预训练的主干，仅训练分类头。为了隔离迁移学习的影响，我们保持设置最小——在这个阶段不应用正则化、增强或类别重平衡。

```py
state, metrics = train(
  state=FinetunedHeadResNet(num_classes=num_classes).create_train_state(
    rng=rng_init,
    dummy_input=dataset_splits["train"].get_dummy_input(),
    tx=optax.adamw(learning_rate, weight_decay=0.0, mask=decay_mask),
  ),
  rng=rng_train,
  dataset_splits=dataset_splits,
  num_steps=num_steps,
  batch_size=32,
  preprocessor=resnet,
  sampler=repeating_sampler,
  augmentor=None,
  eval_every=100,
  store_path=assets("cancer/models/resnet_just_head"),
)
```

让我们检查图 5-20 中模型的性能。

```py
plot_learning(metrics);
```

![](img/dlfb_0520.png)

###### 图 5-20. `FinetunedHeadResNet`多类皮肤病变分类模型在训练和验证步骤中的模型性能。

一些观察结果：

- 损失曲线：训练损失如预期地稳步下降，但验证损失在大约 250 步之后又开始上升。这是过拟合的经典迹象——模型对训练集的信心越来越强，但将其推广到新数据的能力开始下降。
- 训练与验证差距：尽管比`SimpleCnn`基线小，但训练和验证指标之间仍然存在很大的差距。
- 精确度和召回率：有趣的是，尽管验证损失在增加，但验证精确度和召回率指标相对稳定。这表明，尽管模型的 softmax 置信度可能恶化，但它的实际分类决策在训练过程中并没有显著恶化——至少目前还没有。
- 指标停滞：验证性能在大约 250 步时趋于平稳。这很可能反映了我们只训练分类头——模型的大部分（ResNet 主干）仍然保持冻结。这种有限的灵活性可能限制了性能，尽管早期有所提升。

值得注意的是，这里的验证性能并没有比我们从零开始训练 ResNet 所达到的性能明显更好（尽管请注意`FinetunedHeadResNet`训练得多快）。这可能表明仅训练分类头不足以提供足够的灵活性，使模型能够正确适应我们的数据集。相对于验证性能而言，训练性能显著更高也指向了过拟合。虽然我们可以尝试通过正则化或增强来解决此问题，但我们将转而进行我们预期将是一个显著更强的方法——微调整个网络。

## 训练 `FinetunedResNet` 模型

接下来，我们将训练一个更新所有预训练权重的模型——而不仅仅是分类头。这种方法被称为完全微调，允许模型逐渐调整其内部特征表示，以更好地匹配我们的数据集：

```py
state, metrics = train(
  state=FinetunedResNet(num_classes=num_classes).create_train_state(
    rng=rng_init,
    dummy_input=dataset_splits["train"].get_dummy_input(),
    tx=optax.adamw(learning_rate, weight_decay=0.0, mask=decay_mask),
  ),
  rng=rng_train,
  dataset_splits=dataset_splits,
  num_steps=num_steps,
  batch_size=32,
  preprocessor=resnet,
  sampler=repeating_sampler,
  augmentor=None,
  eval_every=100,
  store_path=assets("cancer/models/resnet50_basic"),
)
```

让我们检查图 5-21 中模型的性能：

```py
plot_learning(metrics);
```

![图片](img/dlfb_0521.png)

###### 图 5-21. 初始（非优化）`FinetunedResNet`多类皮肤病变分类模型在训练和验证步骤中的模型性能。

从这些图中我们可以得出以下几点结论：

- 与先前模型的比较：在验证指标方面，该模型明显优于基线`SimpleCnn`、`ResNetFromScratch`和`FinetunedHeadResNet`模型。
- 精确度和召回率：验证指标始终高于先前模型——经过训练后，加权精确度和召回率达到约 0.65–0.7。然而，训练和验证性能之间仍然存在明显的差距，这表明过拟合仍然是我们需要解决的问题。
- 损失曲线：训练损失稳步下降至接近零，而验证损失保持较高并逐渐上升。这与过拟合一致，尽管值得记住的是，上升的 softmax 损失并不总是反映分类质量的下降（实际上，在这里验证指标保持稳定，而验证损失上升）。

总结来说，这是我们迄今为止表现最好的模型——证实了在使用预训练骨干时完全微调的好处。在检查完全微调模型的混淆矩阵之前，让我们首先应用一些有针对性的优化，以进一步提高关键指标的性能。

## 优化 `FinetunedResNet` 模型

我们将应用四个关键修改：

- 一个预热加余弦衰减的学习率计划
- 数据增强（如“增强数据集”中介绍）
- 高比率 dropout 以减少过拟合（设置为`0.7`）
- 使用`adamw`优化器进行权重衰减（设置为`1e-4`）

我们将依次介绍这些内容。

### 学习率计划

微调预训练模型需要小心，以保留从像 ImageNet 这样的大型数据集中学习到的有价值的基本特征。一种常见策略是使用*预热学习率计划*，它以较小的学习率开始训练，并逐渐增加——允许模型温和地适应——然后随着时间的推移平滑衰减。这有助于：

- 在早期阶段稳定训练
- 防止对预训练权重进行突然更新
- 允许逐渐适应我们的新任务

我们将使用预热和余弦衰减计划，其中学习率将：

- 在前 20%的步骤中进行预热，逐渐增加学习率
- 在学习率为`0.001`（`1e-3`）时达到峰值
- 到训练结束时平滑衰减到`0.00001`（`1e-5`）

让我们可视化学习率在图 5-22 中的演变：

```py
import optax

learning_rate = 0.001
num_steps = 2000

warmup_cosine_decay_scheduler = optax.warmup_cosine_decay_schedule(
  init_value=0.0001,
  peak_value=learning_rate,
  end_value=0.00001,
  warmup_steps=int(num_steps * 0.2),
  decay_steps=num_steps,
)

lrs = [warmup_cosine_decay_scheduler(i) for i in range(num_steps)]

plt.scatter(range(num_steps), lrs)
plt.title("Learning Rate over Steps")
plt.ylabel("Learning Rate")
plt.xlabel("Step");
```

![](img/dlfb_0522.png)

###### 图 5-22. 训练过程中学习率的变化，遵循预热余弦衰减计划。学习率开始较小，在训练步骤的前 20%（预热）中逐渐增加，然后达到峰值并平滑衰减。

### 数据增强

我们之前观察到即使在我们的最佳模型中也存在过拟合。减轻这种过拟合的一种方法是通过应用数据增强，通过引入小的、标签保留的变换（例如，翻转、亮度抖动、裁剪）来增加训练集的有效多样性。正如本章前面所讨论的，增强有助于模型更稳健地泛化，因为它防止模型记住训练图像。

我们将通过使用本章前面定义的`rich_augmentor`来应用这些增强策略。

### 通过 dropout 和 adamw 进行正则化

除了数据增强外，我们还将应用两种方法的显式正则化：

**Dropout**

在训练过程中随机禁用神经元，鼓励网络依赖分布式表示并减少协同适应。我们设置了高 dropout 率 0.7 以强烈对抗过拟合。

**权重衰减**

通过在损失函数中对过大的权重进行惩罚来阻止过大的权重。我们使用`optax.adamw`来实现这一点，它被设计为与 L2 正则化一起正确工作。

###### 警告

您不能安全地使用标准 Adam 优化器与 L2 正则化一起使用——它不会以预期的方式应用权重衰减。这正是引入`adamw`的原因，并且在添加权重衰减时应使用它。

标准做法是*不*对以下内容应用权重衰减：

- 偏差参数：这些通常很小且数量很少，并不显著增加模型复杂性。对它们进行正则化通常不会帮助泛化，有时甚至可能损害性能。
- 批量归一化参数：这些包括学习到的尺度（`gamma`）和偏移（`beta`）项，以及运行统计量。对这些应用权重衰减可能会使训练不稳定，尤其是在微调场景中，因为它们控制激活的分布而不是模型容量。

相反，权重衰减通常只应用于卷积层或线性层的主要权重，这有助于通过阻止过于复杂的解决方案来防止过拟合。为了强制执行这一点，我们在构建优化器时将使用参数掩码——仅对相关参数应用衰减。

## 训练优化的 `FinetunedResNet` 模型

现在我们已经引入了学习率调度、数据增强、dropout 和权重衰减——是时候把所有东西放在一起了。我们可以训练优化的`FinetunedResNet`模型，看看这些变化是否有助于减少过拟合并提高验证性能。
