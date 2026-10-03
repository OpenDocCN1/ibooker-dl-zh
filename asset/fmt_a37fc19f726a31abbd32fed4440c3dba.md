
## 旋转矩阵与交错排列

```
      [ x0   x1   x2   x3   x4   x5   x6   x7 ]

        │    │    │    │    │    │    │    │

        ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼

       cos  sin  cos  sin  cos  sin  cos  sin
```

这是旋转矩阵：

```
      [ cosθ  -sinθ    0      0   ... ]

      [ sinθ   cosθ    0      0   ... ]

      [  0      0    cosθ   -sinθ ... ]

      [  0      0    sinθ    cosθ ... ]
```

此处，嵌入维度按偶/奇余弦/正弦对交错排列。

两种布局编码相同的相对位置。唯一的区别在于维度的配对方式。

## C.4 分组查询注意力

*分组查询注意力*（*GQA*）已成为原始多头注意力（MHA）机制的标准替代方案，在计算和参数方面效率更高。与每个头都有自己的键和值集合的 MHA 不同，GQA 通过将多个头分组共享相同的键和值投影来减少内存使用，如图 C.5 所示。

![](img/APPC_F05_Raschka2.png)

**图 C.5** MHA 与 GQA 的对比。此处，组大小为 2，也就是说一个键-值对由 2 个查询共享。

GQA 的核心思想是让多个查询头共享键和值头，从而减少键和值头的数量。这既降低了模型的参数量，也减少了推理期间键和值张量的内存带宽占用，因为需要从 KV 缓存中存储和取回的键和值更少（详见第 C.7 节）。

虽然 GQA 主要是一种针对 MHA 的计算效率变通方案，但原始 GQA 论文《[GQA：从多头检查点训练广义多查询 Transformer 模型](https://arxiv.org/abs/2305.13245)》中给出的消融研究表明，它在 LLM 建模性能上与标准 MHA 相当。

接下来的代码清单实现了带有 KV 缓存支持的 GQA 机制。

**代码清单 C.4** 分组查询注意力

```
class GroupedQueryAttention(nn.Module):

    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None,

                 qk_norm=False, dtype=None):

        super().__init__()

        assert num_heads % num_kv_groups == 0

        self.num_heads = num_heads

        self.num_kv_groups = num_kv_groups

        self.group_size = num_heads // num_kv_groups

        if head_dim is None:

            assert d_in % num_heads == 0

            head_dim = d_in // num_heads

        self.head_dim = head_dim

        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(

            d_in, self.d_out, bias=False, dtype=dtype

        )

        self.W_key = nn.Linear(

            d_in, num_kv_groups * head_dim, bias=False,dtype=dtype

        )

        self.W_value = nn.Linear(

            d_in, num_kv_groups * head_dim, bias=False, dtype=dtype

        )

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:

            self.q_norm = RMSNorm(head_dim, eps=1e-6)

            self.k_norm = RMSNorm(head_dim, eps=1e-6)

        else:

            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):

        b, num_tokens, _ = x.shape

        queries = self.W_query(x)            #1

        keys = self.W_key(x)                 #2

        values = self.W_value(x)              #2

        queries = queries.view(b, num_tokens, self.num_heads,

                               self.head_dim).transpose(1, 2)

        keys_new = keys.view(b, num_tokens, self.num_kv_groups,

                             self.head_dim).transpose(1, 2)

        values_new = values.view(b, num_tokens, self.num_kv_groups,

                                 self.head_dim).transpose(1, 2)

        if self.q_norm:

            queries = self.q_norm(queries)

        if self.k_norm:

            keys_new = self.k_norm(keys_new)

        queries = apply_rope(queries, cos, sin, offset=start_pos)

        keys_new = apply_rope(keys_new, cos, sin, offset=start_pos)

        if cache is not None:

            prev_k, prev_v = cache

            keys = torch.cat([prev_k, keys_new], dim=2)

            values = torch.cat([prev_v, values_new], dim=2)

        else:

            start_pos = 0                    #3

            keys, values = keys_new, values_new            

        next_cache = (keys, values)

        keys = keys.repeat_interleave(       #4

            self.group_size, dim=1            #4

        )                                     #4

        values = values.repeat_interleave(    #4

            self.group_size, dim=1            #4

        )                                     #4

 #4

        attn_scores = queries @ keys.transpose(2, 3)

        attn_scores = attn_scores.masked_fill(mask, -torch.inf)

        attn_weights = torch.softmax(

            attn_scores / self.head_dim**0.5, dim=-1

        )

        context = (attn_weights @ values).transpose(1, 2)

        context = context.reshape(b, num_tokens, self.d_out)

        return self.out_proj(context), next_cache
```

- #1 形状为 `(b, num_tokens, num_heads * head_dim)`。
- #2 形状为 `(b, num_tokens, num_kv_groups * head_dim)`。
- #3 重置 `RoPE`。
- #4 扩展 K 和 V，使其与头的数量匹配。

你可能已经注意到，代码清单 C.4 中的 GQA 机制还包含一个 `qk_norm` 参数，该参数并非标准 GQA 设计的一部分。当 `qk_norm=True` 时，模型在计算注意力分数之前会分别对查询和键向量应用 `RMSNorm`。这种变体通常被称为 `QKNorm`，因为它对查询（Q）和键（K）表示进行归一化。换句话说，`QKNorm` 使用第 C.1 节中介绍的相同 `RMSNorm` 操作，但将其应用于注意力机制内部，而不是作为 Transformer 块周围的常规归一化层。在 Qwen3 中，除了常规的 `RMSNorm` 层之外，还使用了这种查询/键归一化，以帮助稳定训练期间的注意力分数。

## C.5 Transformer 块

*Transformer 块*是 LLM 的核心组件，它将本附录目前介绍的所有独立元素组合在一起。如图 C.6 所示，它会被重复多次；在 Qwen3 的 0.6B 参数版本中，它出现了 28 次。

![](img/APPC_F06_Raschka2.png)

**图 C.6** Qwen3 中 Transformer 块的结构。每个块包含 `RMSNorm`、`RoPE`、带掩码的分组查询注意力和一个前馈模块；在 0.6B 参数模型中，它被重复 28 次。

代码清单 C.5 实现了图 C.6 所示的 Transformer 块。

**代码清单 C.5** Transformer 块

```
class TransformerBlock(nn.Module):

    def __init__(self, cfg):

        super().__init__()

        self.att = GroupedQueryAttention(

            d_in=cfg["emb_dim"],

            num_heads=cfg["n_heads"],

            head_dim=cfg["head_dim"],

            num_kv_groups=cfg["n_kv_groups"],

            qk_norm=cfg["qk_norm"],

            dtype=cfg["dtype"]

        )

        self.ff = FeedForward(cfg)

        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)

        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):

        shortcut = x

        x = self.norm1(x)

        x, next_cache = self.att(

            x, mask, cos, sin, start_pos=start_pos,cache=cache

        )  #1

        x = x + shortcut

        shortcut = x

        x = self.norm2(x)

        x = self.ff(x)

        x = x + shortcut

        return x, next_cache
```

- #1 形状为 `(batch_size, num_tokens, emb_size)`。

正如你在代码清单中所见，Transformer 块只是将我们在前面各节中实现的各种元素连接起来。

## C.6 主模型代码

在本节中，我们将定义第 2 章中导入并使用过的 `Qwen3Model` 类。

为了实现 `Qwen3Model` 类，下面的代码清单遵循图 C.6 所示的架构，并以 Transformer 块作为 LLM 的核心。

**代码清单 C.6** 主 `Qwen3Model` 代码

## C.6 `Qwen3Model` 类

```
class Qwen3Model(nn.Module):

    def __init__(self, cfg):

        super().__init__()

        # Main model parameters

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"],

                                    dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(

            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]

        )

        self.final_norm = RMSNorm(cfg["emb_dim"])

        self.out_head = nn.Linear(

            cfg["emb_dim"], cfg["vocab_size"],

            bias=False, dtype=cfg["dtype"]

        )

        # Reusable utilities

        if cfg["head_dim"] is None:

            head_dim = cfg["emb_dim"] // cfg["n_heads"]

        else:

            head_dim = cfg["head_dim"]

        cos, sin = compute_rope_params(

            head_dim=head_dim,

            theta_base=cfg["rope_base"],

            context_length=cfg["context_length"]

        )

        self.register_buffer("cos", cos, persistent=False)

        self.register_buffer("sin", sin, persistent=False)

        self.cfg = cfg

        self.current_pos = 0  # Track current position in KV cache

    def forward(self, in_idx, cache=None):

        tok_embeds = self.tok_emb(in_idx)

        x = tok_embeds

        num_tokens = x.shape[1]

        if cache is not None:

            pos_start = self.current_pos

            pos_end = pos_start + num_tokens

            self.current_pos = pos_end

            mask = torch.triu(

                torch.ones(

                    pos_end, pos_end, device=x.device, dtype=torch.bool

                ),

                diagonal=1

            )[pos_start:pos_end, :pos_end]

        else:

            pos_start = 0  # Not strictly necessary but helps torch.compile

            mask = torch.triu(

                torch.ones(num_tokens, num_tokens, device=x.device,

                           dtype=torch.bool),

                diagonal=1
```
