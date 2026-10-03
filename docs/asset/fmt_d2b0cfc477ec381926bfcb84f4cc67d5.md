
如图 C.1 所示，`Qwen3` 架构与 `GPT-2` 非常相似，后者在我的《从零开始构建大语言模型》一书中有详细讲解。虽然本书不要求读者熟悉 `GPT-2`，但本附录为熟悉 `GPT-2` 的读者提供了与 `GPT-2` 的对比。事实上，在编写 `Qwen3` 实现时，我把我另一本书中的 `GPT-2` 模型逐段移植到 `Qwen3` 架构中，使其遵循相似的风格约定，也更易于阅读。

![](img/APPC_F01_Raschka2.png)

**图 C.1：`Qwen3` 与 `GPT-2` 的架构对比。两种模型都通过嵌入层和堆叠的 Transformer 块处理文本，但在某些设计选择上有所不同。**

如图 C.1 所示，`Qwen3`（2025 年发布）和 `GPT-2`（2019 年发布）大体相似，因为它们都基于原始 Transformer 架构的解码器子模块。但自 2019 年以来，一些设计选择已经发生了变化。请注意，`Qwen3` 中的大多数设计选择并非 `Qwen3` 独有，在其他许多当代 LLM 中也很常见，正如我在“大型 LLM 架构对比”([`magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison`](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)) 一文中所讨论的那样。

如果你是 LLM 新手，想了解这些架构是如何实现的，我建议从 `GPT-2` 开始。它的设计实现起来更简单，因此在探索更多现代变体之前，它是一个更容易的切入点。

由于本书并不聚焦于架构实现，本附录的其余部分仅对 `Qwen3` 的代码进行简要概述。

## C.1 `RMSNorm`

与使用标准 `LayerNorm` 的 `GPT-2` 不同，较新的 `Qwen3` 架构使用了*均方根层归一化*（`RMSNorm`）。这一趋势在最近的模型架构中变得越来越普遍。

`RMSNorm` 与 `LayerNorm` 的核心功能相同：对层激活进行归一化，以稳定并改进训练。但如图 C.2 所示，它通过移除均值中心化步骤来简化计算。这意味着激活仍然会被归一化，但不会以 0 为中心。

![](img/APPC_F02_Raschka2.png)

**图 C.2：`LayerNorm`（用于 `GPT-2`）和 `RMSNorm`（用于 `Qwen3`）的对比。`LayerNorm`（右）对激活进行归一化，使其平均值（均值）恰好为零，离散程度（方差）恰好为一。`RMSNorm`（左）则根据激活的均方根进行缩放，不强制均值为零或方差为一，但仍将均值和方差保持在合理范围内，以实现稳定的训练。**

如图 C.2 所示，`LayerNorm` 和 `RMSNorm` 都会将层输出缩放到一个合理的范围。

`LayerNorm` 会减去均值并除以标准差，从而使层输出的均值为零且方差为一（方差为一，标准差为一），这为稳定训练带来了有利的梯度性质。

`RMSNorm` 将输入除以均方根。这样可以将激活缩放到一个可比的数量级，而不强制均值为零或方差为一。在图 C.2 所示的示例中，均值为 0.77，方差为 0.41。

`LayerNorm` 和 `RMSNorm` 都能稳定激活尺度并改善优化，但在大规模 LLM 中，`RMSNorm` 通常更受青睐，因为它的计算成本更低。与 `LayerNorm` 不同，`RMSNorm` 默认不使用偏置（偏移）项，从而减少了可训练参数的数量。此外，`RMSNorm` 将昂贵的均值和方差计算简化为一次均方根运算。这将跨特征归约的次数从两次减少到一次，从而降低了 GPU 上的通信开销，略微提升了训练效率。

下面的清单展示了 `RMSNorm` 在代码中的实现。

**清单 C.1：`RMSNorm`**

```python
import torch.nn as nn

class RMSNorm(nn.Module):

    def __init__(
        self,
        emb_dim,
        eps=1e-6,
        bias=False,
        qwen3_compatible=True,
    ):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype
        if self.qwen3_compatible:
            x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale
        if self.shift is not None:
            norm_x = norm_x + self.shift
        return norm_x.to(input_dtype)
```

请注意，为简洁起见，本附录不会对每个 LLM 组件进行详细的代码讲解。相反，我们将在 C.6 节将所有组件集成到 `Qwen3Model` 类中，将预训练权重加载到其中，然后在 C.9 节使用该模型生成文本。

## C.2 前馈模块

*前馈模块*（一个小的多层感知器）被 2020 年论文《GLU 变体改进 Transformer》（[`arxiv.org/abs/2002.05202`](https://arxiv.org/abs/2002.05202)）中引入的*门控线性单元*（GLU）变体所取代。在这种设计中，标准的两个全连接层被替换成三个，如图 C.3 所示。

![](img/APPC_F03_Raschka2.png)

**图 C.3：在 `GPT-2`（上）中，前馈模块由两个全连接（线性）层组成，中间以一个非线性激活函数分隔。在 `Qwen3`（下）中，该模块是门控线性单元（GLU）变体，它增加了第三个线性层（线性层 3），并将该层的输出与线性层 1 的激活输出逐元素相乘。**

如图 C.3 所示的 `Qwen3` 前馈模块，可以通过下一个清单来实现。

**清单 C.2：`Qwen3` 的前馈模块**

```python
class FeedForward(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(
            cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"],
            bias=False
        )
        self.fc2 = nn.Linear(
            cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"],
            bias=False
        )
        self.fc3 = nn.Linear(
            cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"],
            bias=False
        )

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2  #1
        return self.fc3(x)
```

\#1 这里的非线性激活函数是 SiLU 函数，稍后将讨论。

乍一看，`Qwen3` 使用的 GLU 前馈变体似乎应该优于 `GPT-2` 中的标准前馈变体，因为它增加了一个额外的线性层（三个而不是两个），因此似乎拥有更多参数。

但这种直觉是有误导性的。GLU 变体中的 `fc1` 和 `fc2` 层的宽度各为标准前馈模块中 `fc1` 层宽度的一半，因此 GLU 变体反而拥有更少的参数。

为了用具体示例说明这一点，假设图 C.3 中线性层 1 的输入维度为 1,024。这对应于清单 C.2 中的 `cfg["emb_dim"]`。`fc1` 的输出维度为 3,072（`cfg["hidden_dim"]`）。这些是 `Qwen3` 0.6B 变体中使用的实际数值。在这种情况下，清单 C.2 中 GLU 变体的参数数量如下：

- `fc1`: 1024 × 3,072 = 3,145,728
- `fc2`: 1024 × 3,072 = 3,145,728
- `fc3`: 1024 × 3,072 = 3,145,728
- 总计：3 × 3,145,728 = 9,437,184 个参数

如果我们假设这个 GLU 变体中的 `fc1` 宽度仅为标准前馈模块中 `fc1` 通常选择宽度的一半，那么标准前馈模块的参数数量将如下所示：

- `fc1`: 1024 × 2 × 3,072 = 6,291,456
- `fc2`: 1024 × 2 × 3,072 = 6,291,456
- 总计：2 × 6,291,456 = 12,582,912 个参数

虽然 GLU 变体通常比常规前馈模块参数更少，但它们的表现更好。这种改进来自门控机制引入的额外乘法交互，`activation(x_fc1) * x_fc2`，它增强了模型的表达能力。这与在适当训练下，更深、更窄的网络可以优于更浅、更宽的网络类似。

在进入下一节之前，还有一件事需要说明。请注意，图 C.3 所示的前馈模块包含一个标记为“激活函数”的元素，但我们在清单 C.2 中使用了 `nn.functional.silu` 激活作为具体示例。

从历史上看，激活函数一直是一个争论激烈的话题，直到十多年前深度学习社区基本统一采用*修正线性单元*（ReLU）。ReLU 简单且计算成本低，但在零点处有一个尖锐的折点。这促使研究人员探索更平滑的函数，例如*高斯误差线性单元*（GELU）和*Sigmoid 线性单元*（SiLU），如图 C.4 所示。

![](img/APPC_F04_Raschka2.png)

**图 C.4** 可在前馈模块（神经网络）中使用的不同激活函数。GELU 和 SiLU（Swish）为 ReLU 提供了平滑的替代方案，ReLU 在输入为零时有一个尖锐的折点。

GELU 涉及高斯累积分布函数（CDF）。计算该 CDF 较慢，因为它使用了分段逻辑和指数运算，这使得很难编写融合且优化的 GPU 内核；不过，`tanh` 近似可以使用成本更低的操作，运行速度更快，结果几乎相同。简而言之，虽然 GELU 能产生平滑的激活曲线，但总体而言，它的计算成本比更简单的函数更高。

较新的模型大多已用 SiLU（也称为 *Swish*）函数取代 GELU。SiLU 能将较大的负输入平滑地抑制到接近 ~0，并且对较大的正输入近似线性，如图 C.4 所示。

SiLU 具有相似的平滑性，但计算成本略低于 GELU，并且能提供相当的建模性能。在实践中，SiLU 现在被大多数架构所使用，而 GELU 仅在一些模型中继续使用，例如 Google 的 Gemma 开放权重 LLM。在清单 C.2 的前馈模块实现中，SiLU 函数通过 `nn.functional.silu` 调用。清单 C.2 中的前馈模块也常被称为 *SwiGLU*，该缩写源自 Swish 和 GLU 两个术语。

## C.3 旋转位置嵌入

在基于 Transformer 的大型语言模型中，由于注意力机制的存在，位置编码是必要的。默认情况下，注意力机制将输入词元视为没有顺序。在最初的 GPT 架构中，绝对位置嵌入通过为序列中的每个位置添加一个可学习的嵌入向量，并将其加到词元嵌入上，从而解决了这个问题。

*RoPE*（*旋转位置嵌入*的缩写）采用了一种不同的方法：它不是将位置信息作为单独的嵌入添加，而是通过一种依赖于每个词元位置的方式，在注意力机制中旋转查询向量和键向量来编码该信息（注意力机制将在下一节讨论）。RoPE 是一个优雅的想法，但它本身也是一个很大的主题。如果你感兴趣，可以在原始 RoPE 论文“RoFormer: Enhanced Transformer with Rotary Position Embedding”中找到更多信息，参见 [`arxiv.org/abs/2104.09864`](https://arxiv.org/abs/2104.09864)。（RoPE 于 2021 年首次提出，并在 2023 年随着原始 Llama 模型的发布而被广泛采用。此后，它已成为现代大型语言模型的主流配置，因此并非 `Qwen3` 所独有。）

RoPE 可以通过两种数学上等价的方式实现：交错形式将相邻维度配对进行旋转；两半形式则为了便于处理将维度拆分为余弦和正弦两半。下面的清单实现的是两半变体，这种变体可能更容易阅读。

**清单 C.3** RoPE 函数

```python
import torch

def compute_rope_params(head_dim, theta_base=10_000, context_length=4096,
                        dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"
    inv_freq = 1.0 / (theta_base ** (
        torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float()
        / head_dim
    ))
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=1)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin

def apply_rope(x, cos, sin, offset=0):
    batch_size, num_heads, seq_len, head_dim = x.shape    #1
    assert head_dim % 2 == 0, "Head dimension must be even"
    x1 = x[..., : head_dim // 2]  # First half    #2
    x2 = x[..., head_dim // 2:]  # Second half     #2
    cos = cos[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    # Shape after:  (1, 1, seq_len, head_dim)
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated.to(dtype=x.dtype)    #3
```

1. 形状为 `(batch_size, num_heads, seq_len, head_dim)`。
2. 将 `x` 拆分为前半部分和后半部分。
3. 应用 `cos` 和 `sin` 旋转后，可以使用较低精度。

清单 C.3 中的 RoPE 代码将用于 C.4 节讨论的分组查询注意力机制。

### RoPE 实现变体

如果你熟悉原始 RoPE 论文（[`arxiv.org/abs/2104.09864`](https://arxiv.org/abs/2104.09864)），你可能会对我选择的特定实现感到疑惑。实现 RoPE 有两种常见风格，它们在数学上是等价的。这些实现的主要区别在于旋转矩阵配对维度的方式。我选择了拆分两半的风格，因为它更容易阅读和实现。

```
1) Split-halves style (this book, Hugging Face Transformers):

      [ x0   x1   x2   x3   x4   x5   x6   x7 ]

        │    │    │    │    │    │    │    │

        ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼

       cos  cos  cos  cos  sin  sin  sin  sin
```

这是旋转矩阵：

```
      [ cosθ   -sinθ    0      0   ... ]

      [ sinθ    cosθ    0      0   ... ]

      [  0       0    cosθ   -sinθ ... ]

      [  0       0    sinθ    cosθ ... ]
```

这里，嵌入维度被分成两半，每一半按块进行旋转。

原始论文（以及官方 Llama 代码仓库）中的交错（偶/奇）RoPE 风格如下所示：
