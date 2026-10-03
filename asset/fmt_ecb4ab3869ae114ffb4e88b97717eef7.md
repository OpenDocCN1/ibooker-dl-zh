
# 使用八个时，大约为 7.3×

这假设你使用足够大的全局批次大小，以保持每个 GPU 的满负荷运行。如果你的批次大小太小，本地批次大小将不足以保持 GPU 忙碌。

## 使用 JAX 进行模型并行

Keras 还提供了强大的工具，可以完全自定义你想要如何进行分布式训练，包括模型并行训练以及你可以想象的数据并行和模型并行训练的任何混合。让我们深入了解。

### DeviceMesh API

首先，你需要理解**设备网格**的概念。设备网格简单地说就是设备网格。考虑以下示例，有八个 GPU：

```py
gpu:0   |   gpu:4
--------|---------
gpu:1   |   gpu:5
--------|---------
gpu:2   |   gpu:6
--------|---------
gpu:3   |   gpu:7 
```

主要思想是将设备分为组，沿着轴组织。通常，一个轴将负责数据并行，另一个轴将负责模型并行（如图 18.2 所示，你的设备形成一个网格，其中水平轴处理数据并行，垂直轴处理模型并行）。

设备网格不必是二维的——它可以是你想要的任何形状。然而，在实践中，你只会看到一维和二维网格。

让我们在 Keras 中创建一个 2×4 的设备网格：

```py
device_mesh = keras.distribution.DeviceMesh(
    # We assume eight devices, organized as a 2 × 4 grid.
    shape=(2, 4),
    # It's convenient to give your axes meaningful names!
    axis_names=["data", "model"],
) 
```

请注意，你也可以明确指定你想要使用的设备：

```py
devices = [f"gpu:{i}" for i in range(8)]
device_mesh = keras.distribution.DeviceMesh(
    shape=(2, 4),
    axis_names=["data", "model"],
    devices=devices,
) 
```

如您从`axis_names`参数中猜到的，我们打算使用轴 0 上的设备进行数据并行，轴 1 上的设备进行模型并行。由于轴 0 上有两个设备，轴 1 上有四个，我们将把模型计算分配到四个 GPU 上，并将我们的分割模型复制两次，并行地在不同子批次的数据上运行。

现在我们有了网格，我们需要告诉 Keras 如何将不同的计算部分分配到我们的设备上。为此，我们将使用`LayoutMap` API。

### LayoutMap API

为了指定不同的计算部分应该在何处进行，我们使用**变量**作为我们的参考框架。我们将分割或复制变量到我们的设备上，并将所有与该变量部分相关的计算移动到相应的设备上。

考虑一个变量。它的形状是，比如说，`(32, 64)`。你可以对这个变量做两件事：

+   你可以在网格的轴上**复制**（复制）它，这样沿着该轴的每个设备都能看到相同的值。

+   你可以在网格的轴上**分割**（拆分）它——例如，你可以将其分割成四个形状为`(32, 16)`的块——这样沿着该轴的每个设备都能看到不同的块。

现在，请注意我们的变量有两个维度。重要的是，“分片”或“复制”是你可以独立为变量的每个维度做出的决定。

你将用来告诉 Keras 这些决定的 API 是`LayoutMap`类。`LayoutMap`类似于字典。它将模型变量（例如，你模型中第一个密集层的核变量）映射到关于该变量如何在设备网格上复制或分片的一些信息。具体来说，它将一个*变量路径*映射到一个元组，该元组有与变量维度一样多的条目，其中每个条目指定了对该变量维度的操作。它看起来像这样：

```py
{
    # None means "replicate the variable along this dimension."
    "sequential/dense_1/kernel": (None, "model"),
    # "model" means "shard the variable along this dimension across the
    # devices of the model axis of the device mesh."
    "sequential/dense_1/bias": ("model",),
    ...
} 
```

这是你第一次遇到*变量路径*的概念——它只是一个看起来像`"sequential/dense_1/kernel"`的字符串标识符。这是一个在不保留实际变量实例的情况下引用变量的有用方式。

这是你可以打印模型中所有变量路径的方法：

```py
for v in model.variables:
    print(v.path) 
```

在列表 18.4 的示例模型中，我们得到的结果如下：

```py
sequential/dense/kernel
sequential/dense/bias
sequential/dense_1/kernel
sequential/dense_1/bias 
```

现在，让我们分片和复制这些变量。对于这样一个简单的模型，你的变量分片的基本规则应该是这样的：

+   沿着`"model"`网格轴分片变量的最后一个维度。

+   将所有其他维度保持为复制状态。

简单吧？就像这样：

```py
layout_map = keras.distribution.LayoutMap(device_mesh)
layout_map["sequential/dense/kernel"] = (None, "model")
layout_map["sequential/dense/bias"] = ("model",)
layout_map["sequential/dense_1/kernel"] = (None, "model")
layout_map["sequential/dense_1/bias"] = ("model",) 
```

最后，我们通过设置分布配置来告诉 Keras 在实例化变量时参考这个分片布局，如下所示：

```py
model_parallel = keras.distribution.ModelParallel(
    layout_map=layout_map,
    # This argument tells Keras to use the mesh axis named "data" for
    # data parallelism.
    batch_dim_name="data",
)
keras.distribution.set_distribution(model_parallel) 
```

一旦设置了分布配置，你就可以创建你的模型并使用`fit()`函数来拟合它。你的代码的其他部分不会改变——你的模型定义代码和训练代码都是相同的。这适用于你使用内置的 API 如`fit()`和`evaluate()`，或者使用你自己的训练逻辑。假设你为你的变量有了正确的`LayoutMap`，你刚才看到的简短代码片段就足以分配任何大型语言模型训练运行的计算——它可以扩展到你拥有的任何数量的设备，以及任意大小的模型。

要检查你的变量是如何分片的，你可以检查`variable.value.sharding`属性，如下所示：

```py
>>> model.layers[0].kernel.value.sharding
NamedSharding(
    mesh=Mesh("data": 2, "model": 4),
    spec=PartitionSpec(None, "model")
)
```

你甚至可以通过 JAX 实用工具`jax.debug.visualize_sharding`来可视化它：

```py
import jax

value = model.layers[0].kernel.value
jax.debug.visualize_sharding(value.shape, value.sharding) 
```

## TPU 训练

除了 GPU 之外，在深度学习领域，通常有一个趋势是将工作流程转移到为深度学习工作流程专门设计的越来越专业的硬件上；这样的专用芯片被称为 ASIC（应用特定集成电路）。大小各种公司都在开发新的芯片，但今天在这一领域最突出的努力是谷歌的 Tensor Processing Unit (TPU)，它可在谷歌云和谷歌 Colab 上使用。

在 TPU 上训练确实需要跳过一些障碍。但这是值得的：TPU 真的非常快。在 Colab 上可用的 TPU v2（NVIDIA P100 GPU）上训练通常比训练快 15 倍。对于大多数模型，TPU 训练的平均成本效益比 GPU 训练高 3 倍。

你实际上可以在 Colab 中免费使用 TPU v2。在 Colab 菜单中，在“Runtime”选项卡下，在“Change Runtime Type”选项中，你会注意到除了 GPU 运行时之外，你还可以访问 TPU 运行时。对于更严肃的训练运行，Google Cloud 还提供了从 v3 到 v5 的 TPU，它们甚至更快。

当在启用 TPU 的笔记本上使用 JAX 后端运行 Keras 代码时，你不需要做任何更多的事情，只需调用`keras.distribution.set_distribution(distribution)`并传入一个`DataParallel`或`ModelParallel`分布实例即可开始使用你的 TPU 核心。确保在创建模型之前调用它！

### 使用步骤融合来提高 TPU 利用率

由于 TPU 拥有大量的计算能力，你需要使用非常大的批次来保持 TPU 核心忙碌。对于小型模型，所需的批次大小可能会非常大——每个批次超过 10,000 个样本。当处理巨大的批次时，你应该确保相应地增加你的优化器学习率：你将进行的权重更新会更少，但每次更新将更准确（因为梯度是使用更多的数据点计算的）；因此，你应该在每次更新中通过更大的幅度移动权重。

然而，有一个简单的技巧可以在保持合理大小的批次的同时，维持 TPU 的完全利用率：*步骤融合*。其思路是在每个 TPU 执行步骤中运行多个训练步骤。基本上，在虚拟机内存与 TPU 之间往返两次之间做更多的工作。要做到这一点，只需在`compile()`函数中指定`steps_per_execution`参数——例如，`steps_per_execution=8`表示在每个 TPU 执行中运行八个训练步骤。对于未充分利用 TPU 的小型模型，这可以带来显著的加速：

```py
model.compile(..., steps_per_execution=8) 
```

## 使用低精度计算加速训练和推理

如果我告诉你有一个简单的技巧，你可以用它来加速几乎任何模型的训练和推理，速度提高高达 2 倍，基本上是免费的？这听起来太好了，但确实存在这样的技巧。要了解它是如何工作的，首先，我们需要看看计算机科学中“精度”的概念。

### 理解浮点精度

精度对数字的重要性就像分辨率对图像的重要性。因为计算机只能处理 1 和 0，所以任何被计算机看到的数字都必须被编码为二进制字符串。例如，你可能熟悉`uint8`整数，这些整数是八位编码的整数：`00000000`在`uint8`中表示`0`，而`11111111`表示 255。要表示超过 255 的整数，你需要添加更多的位——八位是不够的。大多数整数都是 32 位存储的，我们可以用它们来表示从-2147483648 到 2147483647 的带符号整数。

浮点数也是一样。在数学中，实数构成一个连续的轴：在任意两个数字之间有无限多个点。你总是可以放大实数轴。在计算机科学中，这并不成立：例如，在 3 和 4 之间只有有限数量的中间点。有多少个？这取决于你工作的**精度**：存储数字所使用的位数。你只能放大到一定的分辨率。

你通常会使用三个级别的精度：

+   半精度，或`float16`，其中数字存储在 16 位上

+   单精度，或`float32`，其中数字存储在 32 位上

+   双精度，或`float64`，其中数字存储在 64 位上

你甚至可以提高到`float8`，稍后你将看到。

考虑浮点数的分辨率的方法是考虑两个任意数字之间你可以安全处理的最小距离。在单精度中，这大约是 1e-7。在双精度中，这大约是 1e-16。而在半精度中，只有 1e-3。

### Float16 推理

在这本书中你迄今为止看到的每一个模型都使用了单精度数字：它将状态存储为`float32`权重变量，并在`float32`输入上运行其计算。这种精度足以运行模型的正向和反向传播而不会丢失任何信息——特别是在涉及小的梯度更新时（回想一下，通常的学习率是 1e-3，权重更新的量级通常为 1e-6）。

现代 GPU 和 TPU 具有专门的硬件，可以比等效的 32 位操作更快地运行 16 位操作，并且使用更少的内存。通过尽可能使用这些低精度操作，你可以显著加快这些设备上的训练速度。你可以在 Keras 中将默认的浮点精度设置为`float16`：

```py
import keras

keras.config.set_dtype_policy("float16") 
```

注意，这应该在定义你的模型之前完成。这样做将为模型推理（例如，通过`model.predict()`）带来很好的加速效果。你应该期望在 GPU 和 TPU 上获得近 2 倍的速度提升。

对于某些设备，特别是 TPU，还有`float16`的替代方案：`bfloat16`。`bfloat16`也是一种 16 位精度的浮点数类型，但它与`float16`在结构上有所不同：它使用 8 位指数位而不是 5 位，7 位尾数位而不是 10 位（见表 18.1）。这意味着它可以覆盖更广泛的值范围，但在这个范围内的“分辨率”较低。与`float16`相比，一些设备对`bfloat16`进行了更好的优化，所以尝试两者之前选择最快的选项可能是个好主意。

| 数据类型 | `float16` | `bfloat16` |
| --- | --- | --- |
| 指数位 | 5 | 8 |
| 尾数位 | 10 | 7 |
| 符号位 | 1 | 1 |

表 18.1：`float16`和`bfloat16`之间的差异

### 混合精度训练

将默认浮点精度设置为 16 位是加快推理速度的好方法。现在，当涉及到训练时，有一个显著的复杂性。梯度下降过程在`float16`或`bfloa16`中不会顺利运行，因为我们无法表示大约 1e-5 或 1e-6 的小梯度更新，这在实践中相当常见。

然而，你可以使用一种混合方法：这就是*混合精度训练*的含义。其思路是在精度不是问题的地方使用 16 位计算，而在其他地方使用 32 位值以保持数值稳定性——特别是在处理梯度变量更新时。通过在完全精度下保持模型中精度敏感的部分，你可以在不显著影响模型质量的情况下，获得 16 位计算的大部分速度优势。

你可以这样开启混合精度：

```py
import keras

keras.config.set_dtype_policy("mixed_float16") 
```

通常，模型的正向传播的大部分操作将在`float16`（除了像 softmax 这样的数值不稳定操作）中完成，而模型的权重将存储和更新在`float32`中。你的`float16`梯度在更新`float32`变量之前将被转换为`float32`。

Keras 层有`variable_dtype`和`compute_dtype`属性。默认情况下，这两个都设置为`float32`。当你开启混合精度时，大多数层的`compute_dtype`将切换到`float16`。因此，这些层将把它们的输入转换为`float16`并在`float16`（使用半精度权重副本）中进行计算。然而，由于它们的`variable_dtype`仍然是`float32`，它们的权重将能够从优化器接收准确的`float32`更新，而不是半精度更新。

一些操作在`float16`中可能数值不稳定（特别是 softmax 和交叉熵）。如果你需要为特定层退出混合精度，只需将`dtype="float32"`参数传递给该层的构造函数即可。

### 使用混合精度进行损失缩放

在训练过程中，梯度可能会变得非常小。当使用混合精度时，你的梯度保持在`float16`（与前向传递相同）。因此，可表示数字的有限范围可能导致小梯度被舍入为零。这会阻止模型有效地学习。

梯度值与损失值成比例，因此为了鼓励梯度更大，一个简单的技巧是将损失乘以一个大的标量因子。你的梯度将不太可能被舍入为零。

Keras 使这变得简单。如果你想使用一个固定的损失缩放因子，你只需像这样将`loss_scale_factor`参数传递给你的优化器即可：

```py
optimizer = keras.optimizers.Adam(learning_rate=1e-3, loss_scale_factor=10) 
```

如果你希望优化器自动确定正确的缩放因子，你也可以使用`LossScaleOptimizer`包装器：

```py
optimizer = keras.optimizers.LossScaleOptimizer(
    keras.optimizers.Adam(learning_rate=1e-3)
) 
```

使用`LossScaleOptimizer`通常是你的最佳选择：正确的缩放值可以在训练过程中改变！

### 超越混合精度：`float8`训练

如果你以 16 位精度运行前向传递会产生如此整洁的性能优势，你可能想知道：我们能更低吗？8 位精度怎么样？四位，也许？两位？答案是，这很复杂。

使用`float16`进行前向传递的混合精度训练是“刚好工作”的最后一个精度级别——`float16`精度有足够的位来表示所有中间张量（除了梯度更新，这就是为什么我们为那些使用`float32`）。如果你降低到`float8`精度，这就不再成立了：你只是丢失了太多的信息。在某些计算中使用`float8`仍然是可能的，但这需要你对前向传递进行相当大的修改。你将**不能**简单地设置你的`compute_dtype`为`float8`并运行。
