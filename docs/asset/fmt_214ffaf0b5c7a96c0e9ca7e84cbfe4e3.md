
```py
import numpy as np

def load_weights_into_gpt(gpt, params):           #1
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):     #2
        q_w, k_w, v_w = np.split(                            #3
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])    #4
```

#1 将模型的定位和标记嵌入权重设置为`params`中指定的那些。

#2 遍历模型中的每个 transformer 块

#3 使用`np.split`函数将注意力和偏置权重分为查询、键和值组件的三个相等部分。

#4 OpenAI 的原始 GPT-2 模型在输出层中重用了标记嵌入权重以减少参数总数，这是一个称为权重绑定的概念。

在`load_weights_into_gpt`函数中，我们仔细地将来自 OpenAI 实现的权重与我们的`GPTModel`实现进行匹配。以一个具体的例子来说，OpenAI 将第一个 transformer 块的输出投影层的权重张量存储为`params["blocks"][0]["attn"]["c_proj"]["w"]`。在我们的实现中，这个权重张量对应于`gpt.trf_blocks[b].att.out_proj.weight`，其中`gpt`是一个`GPTModel`实例。

开发`load_weights_into_gpt`函数花费了很多猜测工作，因为 OpenAI 使用的命名约定与我们略有不同。然而，如果尝试匹配两个维度不同的张量，`assign`函数会提醒我们。此外，如果我们在这个函数中犯了错误，我们会注意到这一点，因为生成的 GPT 模型将无法生成连贯的文本。

现在我们来实际尝试使用`load_weights_into_gpt`并将 OpenAI 模型权重加载到我们的`GPTModel`实例`gpt`中：

```py
load_weights_into_gpt(gpt, params)
gpt.to(device)
```

如果模型加载正确，我们现在可以使用它通过之前的`generate`函数生成新的文本：

```py
torch.manual_seed(123)
token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
```

产生的文本如下：

```py
Output text:
 Every effort moves you toward finding an ideal new way to practice something!
What makes us want to be on top of that?
```

我们可以确信我们已经正确加载了模型权重，因为模型可以生成连贯的文本。在这个过程中出现微小的错误会导致模型失败。在接下来的章节中，我们将进一步使用这个预训练模型，并对其进行微调以分类文本和遵循指令。

### 练习 5.5

在“The Verdict”数据集上，使用 OpenAI 提供的预训练权重计算`GPTModel`的训练和验证集损失。

### 练习 5.6

尝试不同大小的 GPT-2 模型——例如，最大的 1,558 百万参数模型——并将生成的文本与 1.24 亿参数模型进行比较。

## 摘要

+   当 LLM 生成文本时，它们一次输出一个标记。

+   默认情况下，下一个标记是通过将模型输出转换为概率分数并选择与最高概率分数对应的词汇表中的标记来生成的，这被称为“贪婪解码”。

+   通过概率采样和温度缩放，我们可以影响生成文本的多样性和连贯性。

+   在训练过程中，LLM 生成的文本的质量可以通过训练和验证集的损失来衡量。

+   预训练一个大型语言模型（LLM）涉及调整其权重以最小化训练损失。

+   LLM 的训练循环本身是深度学习中的一个标准程序，使用传统的交叉熵损失和 AdamW 优化器。

+   在大型文本语料库上预训练一个 LLM 既耗时又耗资源，因此我们可以加载公开可用的权重，作为我们自己在大数据集上预训练模型的替代方案。
