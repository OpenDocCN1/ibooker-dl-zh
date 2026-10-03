
当你刚开始时，这并不是必要的，但随着时间的推移，熟悉多个深度学习框架可能会有所帮助。每个框架在不同的生态系统中都有其优势——例如，我们在第二章中使用 `PyTorch` 从 Hugging Face 模型中提取预训练嵌入，因为 Hugging Face 上的许多模型主要是以 `PyTorch` 的形式发布和维护的。

## 警告

深度学习领域发展迅速。虽然我们在这本书中始终使用 `Flax` 的 `linen` API，但最近出现了一个名为 `nnx` 的新 API，已成为构建模型的推荐方式。`linen` 仍然完全受支持，但请注意，你可能会遇到使用 `nnx` 的其他教程或示例，其语法略有不同。

我们将在本书中根据需要介绍关键的 `JAX` 概念，但不会详细涵盖整个库。为了更详细的动手学习，请查看官方 [JAX 教程](https://oreil.ly/Jcqtu)。如果你遇到意外行为，`JAX` 的 [“sharp bits”笔记本](https://oreil.ly/TcOzQ) 是了解常见问题及其解决方法的绝佳参考。

### 关于性能的说明

由于这是一本教育书籍，我们的重点是清晰度而不是峰值性能。这意味着我们不会涵盖诸如精确调整、高级硬件策略或分布式训练等内容。但在现实世界的设置中，这些事情可能非常重要。

如果你熟悉基础知识并想进一步探索，以下是一些值得研究的地方：

**数值精度和调整**

许多机器学习操作，尤其是矩阵乘法（matmuls），可以从降低精度格式（如 `bfloat16`）中受益，这可以在最小影响模型精度的前提下显著提高速度和内存使用。`JAX` 允许你通过 `jax.default_matmul_precision` 来控制用于 matmuls 的精度，帮助你利用专门的硬件，如 NVIDIA GPU 上的 Tensor Cores 或 TPU 上的矩阵单元。在大型设置中，低精度训练被广泛使用，因为它可以更高效、更经济地训练更大的模型。

**剖析工具，如 `jax.profiler` 或 TensorBoard**

性能分析可以帮助你识别代码在时间和内存上的消耗，从而在训练中找到瓶颈并优化最昂贵的操作。

**内存高效的训练技术**

梯度检查点（`JAX` 中的 `remat`）等方法让你可以在计算和内存之间进行权衡，允许你在不耗尽 RAM 的情况下训练更深的模型。

**多主机/多设备训练**

在多个 GPU、TPU 或甚至机器上训练可以使你扩展模型和数据集，这些模型和数据集无法适应单个设备。

在阅读这本书的过程中，你可能不需要这些内容，但了解它们是有益的，随着你对 `JAX` 生态系统的熟悉，它们值得探索。

## Python 小贴士

虽然这本书没有深入介绍 Python 的基础知识，但本节突出了几个在处理机器学习代码时可能会遇到的有用 Python 概念——特别是与 `JAX` 和 `Flax` 相关的。

### 类型注解和文档字符串

Python 是一种动态类型语言，这意味着你不需要显式声明变量类型（如字符串或整数）。相反，类型是在运行时确定的，这使得语言灵活——但这种灵活性也可能使错误更难捕捉，尤其是在大型代码库中。添加 *类型注解* 通过提高可读性、使用 `mypy` 等工具进行静态类型检查以及使调试更容易来减轻这一点。

这里有一个简单的函数，用于计算两个 NumPy 数组之间的均方误差（MSE）：

```py
import numpy as np

def mean_squared_error(y_true, y_pred):
  squared_errors = (y_true - y_pred) ** 2
  return np.mean(squared_errors)
```

这里是它使用的一个例子：

```py
y_true = np.array([1.1, 0.1, 1.0])
y_pred = np.array([0.9, 0.2, 1.2])
mean_squared_error(y_true, y_pred)
```

输出：

```py
np.float64(0.030000000000000002)
```

我们可以通过添加类型提示来改进这个函数，指定输入是 `np.ndarray` 对象，返回类型是 `float`，以及一个文档字符串来解释函数的功能：

```py
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """
  Calculate the Mean Squared Error (MSE) between two NumPy arrays.

  Args:
    y_true (np.ndarray): Ground-truth values.
    y_pred (np.ndarray): Predicted values.
  """
  squared_errors = (y_true - y_pred) ** 2
  return np.mean(squared_errors)
```

这些更改不会影响函数的行为，但它们提供了几个好处：

**明确输入和输出类型**

很明显，`y_true` 和 `y_pred` 应该是 NumPy 数组，返回值是一个 `float`。请注意，一些机器学习代码会进一步指定数组中的数据类型（例如，`arr: NDArray[np.float64]`），但在这本书中我们不会这样做。

**增强文档**

集成开发环境（IDE）和文档工具可以提供更好的内联帮助和自动完成功能。这可以真正提高生产力。

**提高可读性**

函数对其他人（或你未来的自己）来说更容易理解。

**启用静态检查**

`mypy` 等工具可以捕获与类型相关的错误。

MSE 示例非常简单，所以添加类型提示和完整的文档字符串可能是过度杀鸡用牛刀——但原则很重要。

由于篇幅限制，我们在这本书中不一定总是使用类型注解和文档字符串，但养成良好的习惯在自己的项目中是有益的。当我们确实在正文文本中包含文档字符串时，我们通常会将其限制在一行，这样可以节省一些打印纸张。

### 装饰器

*装饰器*是修改其他函数或方法行为的函数。在机器学习和数据科学中，装饰器通常用于提高性能、缓存结果或记录函数行为。

#### 使用 JAX 进行即时（JIT）编译

在 `JAX` 中工作时最常用的装饰器之一是 `jax.jit`，它执行 JIT 编译以加速代码执行。JIT 编译函数的第一次运行较慢，因为它被编译为 XLA（加速线性代数）机器代码。然而，后续调用运行得要快得多。

假设我们有一个函数，它接受一个 `JAX` 数组，将所有值提高到 10 次幂，并将它们相加：

```py
import jax
import jax.numpy as jnp

def compute_ten_power_sum(arr: jax.Array) -> float:
  """Raise values to the power of 10 and then sum."""
  return jnp.sum(arr**10)

arr = jnp.array([1, 2, 3, 4, 5])
compute_ten_power_sum(arr)
```

输出：

```py
Array(10874275, dtype=int32)
```

我们可以通过两种方式加快这个函数。要么直接：

```py
jitted_compute_ten_power_sum = jax.jit(compute_ten_power_sum)
jitted_compute_ten_power_sum(arr)
```

输出：

```py
Array(10874275, dtype=int32)
```

或者，通过 Python 语法糖，我们可以在使用 `@jax.jit` 装饰器定义函数时应用它：

```py
@jax.jit
def compute_ten_power_sum(arr: jax.Array) -> float:
  """Raise values to the power of 10 and then sum."""
  return jnp.sum(arr**10)

compute_ten_power_sum(arr)
```

输出：

```py
Array(10874275, dtype=int32)
```

对于所有人来说，输出都是相同的，但 JIT 函数运行得要快得多。如果你在一个 Jupyter 笔记本中工作，你可以使用 `%timeit` 来测量一行代码的执行时间（或 `%%timeit` 用于整个单元格）。尝试对函数进行计时，带有和不带有 `@jax.jit`。在 GPU 上，你可能看到大约 20 倍的速度提升。

## 注意

`@jax.jit` 是如何工作的？简而言之，当你应用 `@jax.jit` 时，`JAX` 不会像常规 Python 那样运行你的函数。相反，它首先跟踪函数——它使用特殊的跟踪对象（不是真实数据）运行一次，以构建计算图。这个图是所有数值操作静态表示，控制流展开，变量形状和类型固定。

一旦构建了图，`JAX` 会使用 XLA（加速线性代数）来编译它，这是一个生成高度优化的机器代码的后端。这个编译版本会被缓存并重复使用，每当函数再次以相同的输入形状和类型被调用时——从而带来显著的加速。

JIT 编译非常强大，但同时也伴随着权衡，尤其是在调试方面。这是因为：

+ Python 调试工具如 `print()` 语句或 `pdb` 在 JIT 函数内部的行为不符合预期。
+ 副作用（例如，`print()`、日志记录或修改列表）在跟踪过程中实际上并没有被执行，因为 `JAX` 会跳过任何不影响计算图的内容。
+ 错误信息可以引用 `JAX` 或 XLA 的内部代码而不是你的原始函数，并且可能非常难以理解。

虽然你可以通过取消注释 `@jax.jit` 来暂时禁用 JIT，但如果许多函数都依赖于 JIT，这就会变得不切实际。幸运的是，你可以通过设置环境变量 `JAX_DISABLE_JIT=True` 来全局禁用 JIT，这将强制所有 JIT 函数正常运行。这是一种在不重写代码的情况下进行调试的便捷方式。有关更多详细信息，请参阅 [JAX 调试文档](https://oreil.ly/oXI97)。

#### 预配置 JAX jit 使用部分

在机器学习代码中，关于 `partial` 的使用存在一个常见的混淆源，尤其是在 `JAX` 代码中。`functools.partial` 的使用是为了预先填充（或“绑定”）函数的一些参数，返回一个具有这些值固定的新函数。这是一个通用的 Python 工具，并不特定于 `JAX` 或 ML。

在这里，我们调整了 `scale` 函数并创建了一个新函数，`scale_by_10`：

```py
from functools import partial

def scale(x, scaling_factor):
  return x * scaling_factor

# Create a new function 'scale_by_10' where 'scaling_factor' is fixed to 10.
scale_by_10 = partial(scale, scaling_factor=10)
scale_by_10(3)
```

输出：

```py
30
```

在这里，`scale_by_10` 是一个新函数，其行为类似于 `scale(x, 10)`。

在 `JAX` 的上下文中，`partial` 通常用于在应用装饰器之前对其进行自定义，例如：`@partial(jax.jit, static_argnums=...)`。这是配置 `jax.jit` 装饰器本身的一种方式。如前所述，`jax.jit` 编译 Python 函数以提高速度。然而，`JAX` 需要知道某些参数是否是静态的。静态参数通常是控制计算结构（例如，在 if/else 语句中）的非 `JAX` 数组类型（如整数、字符串或布尔值）。如果静态参数发生变化，`JAX` 可能需要重新编译函数。

假设我们想要计算一个数组上的汇总统计量，根据字符串参数 `average_method` 选择均值或中位数。由于这个选择会影响控制流，`JAX` 需要在编译时知道 `average_method` 的值：

```py
from functools import partial

import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnums=(0,))
def summarize(average_method: str, x: jax.Array) -> float:
  if average_method == "mean":
    return jnp.mean(x)
  elif average_method == "median":
    return jnp.median(x)
  else:
    raise ValueError(f"Unsupported average type: {average_method}")

data_array = jnp.array([1.0, 2.0, 100.0])

# JAX compiles one version of 'summarize' for average_method="mean".
print(f"Mean: {summarize('mean', data_array)}")

# JAX compiles another version for average_method="median".
print(f"Median: {summarize('median', data_array)}")

# Calling with "mean" again uses the cached compiled version.
print(f"Mean again: {summarize('mean', data_array)}")
```

输出：

```py
Mean: 34.333335876464844
Median: 2.0
Mean again: 34.333335876464844
```

如果我们没有用 `static_argnums=(0,)` 将 `average` 标记为静态，`JAX` 会抛出错误，因为它无法跟踪依赖于字符串的控制流，除非它事先知道它们的值。将参数标记为静态告诉 `JAX` 为每个遇到的静态参数的唯一值编译一个单独的、专门的函数版本。

## 小贴士

关于“静态”与“动态”含义的一点点澄清：`JAX` 将大多数数值输入（如 `jax.Array`、`float` 或 `int`）视为动态的，这意味着只要它们的形状和类型保持不变，它们可以在调用之间变化，而无需重新编译。

其他输入，如字符串、Python 对象或函数，是静态的：它们影响控制流或不能作为计算图的一部分进行跟踪。如果您将它们传递给一个 jitted 函数，您必须使用 `static_argnums` 将其标记为静态，或者使用闭包来封装它们（参见下一节）。

### 闭包

闭包是一个“记住”其创建环境的函数。这意味着它可以访问其封装（外部）函数的作用域中的变量，即使在该作用域执行完成后也是如此：

```py
def outer_function(x):
  def inner_function(y):
    return x + y  # inner_function "closes over" x.

  return inner_function

add_five = outer_function(5)  # x is 5.
result = add_five(10)  # y is 10.
print(f"Closure result: {result}")
```

输出：

```py
Closure result: 15
```

在这个例子中，`add_five` 是一个闭包。它“记住”，当 `outer_function` 被调用时，`x` 的值是 5。

在基于 `JAX` 的机器学习代码中，闭包被广泛使用。许多组件——如损失函数、正则化器和增强管道——由配置值参数化。而不是将这些值作为参数传递（如果它们在控制流中使用可能需要 `static_argnums`），它们通常被封装起来。稍后我们将在定义 `JAX` 训练循环时看到这一点。

### 生成器

*生成器* 是允许你懒加载地遍历数据的函数，一次产生一个项目。它们在处理大型数据集时特别有用，因为一次性将所有内容加载到内存（RAM）中既不实际也不可能。

这里有一个简单的生成器函数，用于模拟流式传输数据批次：

```py
from typing import Iterator

def data_generator() -> Iterator[dict]:
  """Yield data samples with features and labels."""
  for i in range(5):
    yield {"feature": i, "label": i % 2}

# Example usage.
generator = data_generator()
next(generator)
```

输出：

```py
{'feature': 0, 'label': 0}
```

在某些章节中，我们将使用 TensorFlow 数据集（TFDS）。由于 `JAX` 不包含原生数据加载库，因此常见的是使用 TFDS 的混合设置。如果你有 NumPy 数组格式的数据，你可以使用 `tf.data.Dataset.from_tensor_slices` 容易地创建 TensorFlow 数据集。这允许你将 NumPy 数据集成到 TensorFlow 管道中，以实现高效的训练和预处理。它提供了一个干净的 API 用于分批、洗牌和预取（在需要之前预加载数据以增加训练速度），这对于入门很有帮助。
