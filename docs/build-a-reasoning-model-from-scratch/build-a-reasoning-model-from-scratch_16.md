# 附录 G 构建聊天界面

本书的配套代码仓库包含一个基于浏览器的聊天界面，它是用开源的 Chainlit Python 包构建的（[`github.com/Chainlit/chainlit`](https://github.com/Chainlit/chainlit)）。这个类似 ChatGPT 的界面让我们能够与本书中的各种 LLM 和推理模型交互，如图 G.1 中的截图所示。当我们想要一种更便捷的方式与模型交互，而不必在 notebook 单元格或终端中输入提示时，这个界面会很有用。

![](img/APPG_F01_Raschka2.png)

**图 G.1** 面向 Qwen3 模型的 Chainlit 界面，它模仿了 ChatGPT 界面。

基于 Chainlit 库，这个界面的实现相对简单。我们不将 token 打印到终端，而是将前面章节中的模型加载和生成函数封装在一个小型 Web 应用中，该应用位于计算机本地，并在浏览器中运行。在实践中，这意味着我们可以重用从头实现的`Qwen3Model`和第 2 章的流式辅助函数，同时让 Chainlit 处理浏览器界面和消息事件。

本书的配套材料包含该界面的两个版本，位于`chG/01_main-chapter-code`中（[`github.com/rasbt/reasoning-from-scratch/tree/main/chG/01_main-chapter-code`](https://github.com/rasbt/reasoning-from-scratch/tree/main/chG/01_main-chapter-code)）：

* `qwen3_chat_interface.py` 是一个单轮界面。
* `qwen3_chat_interface_multiturn.py` 存储对话历史，并支持多轮交互。

在本附录中，我们将对这两个版本逐一讲解。我们还会讨论如何安装 Chainlit，以及如何通过第 7 章的`download_from_github`辅助函数下载脚本。

## G.1 安装 Chainlit

在运行聊天界面之前，我们首先需要安装 Chainlit。这是最简单的选择：

```
pip install chainlit
```

如果你使用的是`uv`，等价命令是：

```
uv add chainlit
```



如果你直接在`reasoning-from-scratch`仓库中工作，`Chainlit`也被列为项目`pyproject.toml`中的可选依赖。在这种情况下，以下是一个方便的替代方案：

```
uv sync --extra extra
```

或

```
pip install -e ".[extra]"
```

## G.2 将代码作为脚本运行

`Chainlit`运行普通的`Python`脚本。这一点很重要，因为`Chainlit`会将脚本作为模块导入，并查找带有`@chainlit.on_chat_start`和`@chainlit.on_message`装饰器的特殊函数，稍后你会看到。

这意味着，如果你想重用本附录中展示的代码，你应该将其放入一个`.py`文件中，例如

```
qwen3_chat_interface.py
```

或

```
app.py
```

然后按如下方式从终端启动它：

```
chainlit run app.py
```

或者，如果你使用的是`uv`，命令为：

```
uv run chainlit run app.py
```

换句话说，虽然你可以在`notebook`中查看代码，但实际的`Chainlit`应用程序应该放在一个独立的`Python`脚本中。

## G.3 下载脚本

如果你克隆了仓库，这些文件已经存在于`chG/01_main-chapter-code`中，你可以跳过本节。否则，如第 7 章所述，你可以使用`download_from_github(...)`获取这两个脚本。

**代码清单 G.1** 下载 Chainlit 脚本

```
from reasoning_from_scratch.ch07 import download_from_github

download_from_github(
    "chG/01_main-chapter-code/qwen3_chat_interface.py"
)

download_from_github(
    "chG/01_main-chapter-code/qwen3_chat_interface_multiturn.py"
)
```

## G.4 常规单轮脚本

这个界面的常规单轮版本位于`qwen3_chat_interface.py`，有意保持简洁。它几乎直接重用了第 2 章的`LLM`和推理代码，并用最少的`Chainlit`特有逻辑对其进行封装。

这里的“单轮”是指我们可以提交多个查询，但每个查询彼此独立，且不了解之前的查询。

这段代码相对较短，这里完整展示如下。

**代码清单 G.2** 用于单轮查询的 Chainlit 脚本

```
import os
from pathlib import Path
import torch
import chainlit
from reasoning_from_scratch.ch02 import (
    get_device,
)
from reasoning_from_scratch.ch02 import generate_text_basic_stream_cache
from reasoning_from_scratch.ch03 import (
    load_model_and_tokenizer,
    load_tokenizer_only
)
from reasoning_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B#1

WHICH_MODEL = "reasoning"  #2
MAX_NEW_TOKENS = 38912
LOCAL_DIR = "qwen3"
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH")
COMPILE = False
DEVICE = get_device()

def load_app_model_and_tokenizer():
    if CHECKPOINT_PATH is None:
        return load_model_and_tokenizer(
            which_model=WHICH_MODEL,
            device=DEVICE,
            use_compile=COMPILE,
            local_dir=LOCAL_DIR,
        )
    checkpoint_path = Path(CHECKPOINT_PATH)  #3
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}"
        )
    tokenizer = load_tokenizer_only(
        which_model=WHICH_MODEL, local_dir=LOCAL_DIR
    )
    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu")
    )
    model.to(DEVICE)
    if COMPILE:
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        model = torch.compile(model)
    return model, tokenizer

MODEL, TOKENIZER = load_app_model_and_tokenizer()
EOS_TOKEN_IDS = (
    TOKENIZER.encode("<|im_end|>")[0],
    TOKENIZER.encode("<|endoftext|>")[0]
)

@chainlit.on_chat_start  #4
async def on_start(): #4
    chainlit.user_session.set("history", []) #4
    chainlit.user_session.get("history").append( #4
        {"role": "system", "content": "You are a helpful assistant."} #4
    ) #4

@chainlit.on_message#5
async def main(message: chainlit.Message): #5
    input_ids = TOKENIZER.encode(message.content)#6
    input_ids_tensor = torch.tensor(input_ids, device=DEVICE).unsqueeze(0)
    out_msg = chainlit.Message(content="")#7
    await out_msg.send() #7
    for tok in generate_text_basic_stream_cache(#8
        model=MODEL,
        token_ids=input_ids_tensor,
        max_new_tokens=MAX_NEW_TOKENS,
    ):
        token_id = tok.squeeze(0).item()
        if token_id in EOS_TOKEN_IDS:
            break
        piece = TOKENIZER.decode([token_id])
        await out_msg.stream_token(piece)
    await out_msg.update()#9
```

1. 配置部分
2. 对于基础模型，改为`base`。
3. （可选）从第 7 章或第 8 章加载训练好的检查点。
4. 初始化每次会话的状态，并用默认系统提示填充。
5. 处理一条用户消息，并将模型响应流式传输到`Chainlit`界面。
6. 对输入进行编码。
7. 启动一条可流式写入的出站消息。
8. 流式生成。
9. 完成流式消息。

清单 G.2 中的`CHECKPOINT_PATH`环境变量是可选的。如果设置了该变量，脚本会加载对应的自定义`.pth`检查点，而不是默认的官方权重。`load_app_model_and_tokenizer()`辅助函数会处理这两种情况：如果没有提供自定义检查点，它会委托给第 3 章中的`load_model_and_tokenizer(...)`；如果提供了自定义检查点路径，它只会通过`load_tokenizer_only(...)`加载分词器，构建一个全新的`Qwen3Model(QWEN_CONFIG_06_B)`，然后从自定义状态字典中加载权重。

这样做的好处在于，只要分词器设置与检查点保持一致，单轮`Chainlit`界面也可以用来交互式地检查第 6、7、8 章的检查点。下一节将详细解释如何从命令行将自定义检查点作为输入提供。

`@chainlit.on_chat_start`函数会初始化一个历史对象，并插入一条默认的系统消息，这是一种常见的约定：

```
{"role": "system", "content": "You are a helpful assistant."}
```

但单轮脚本在生成响应时实际上并不会使用存储的历史记录。在`main(...)`内部，提示词仅根据当前的`message.content`构建：

```
input_ids = TOKENIZER.encode(message.content)
```

因此，尽管浏览器界面看起来像聊天应用，单轮版本的行为更像一个提示框，它会在视觉上显示之前的消息，但模型实际上并不会记住之前的对话。换句话说，模型本身只看到当前用户消息，并将每一轮视为独立的。

其余的生成逻辑与你在第 2 章看到的内容非常相似。

## G.5 运行单轮脚本

如果你在配套代码仓库中，启动用户界面最方便的命令通常是：

```
uv run chainlit run chG/01_main-chapter-code/qwen3_chat_interface.py
```

如果你将代码保存为其他文件名，比如`app.py`，只需相应地替换路径：

```
uv run chainlit run app.py
```

启动服务器后，`Chainlit`通常会自动打开一个浏览器标签页。如果没有自动打开，你可以将终端中显示的本地地址手动复制到浏览器中，通常是`http://localhost:8000`。

图 G.2 展示了`Chainlit`界面中的一些输出示例。

![](img/APPG_F02_Raschka2.png)

### 图 G.2 Chainlit 界面中的示例响应

默认情况下，脚本会初始化我在第 3 章简要介绍的`Qwen3`推理变体，图 G.2 中显示的响应就是由它生成的。要使用自定义检查点运行同一脚本，可以通过`CHECKPOINT_PATH`环境变量传递路径。示例如下：

```
CHECKPOINT_PATH=path/to/qwen3-0.6B-distill-step06682-epoch1.pth \
uv run chainlit run chG/01_main-chapter-code/qwen3_chat_interface.py
```

在这种情况下，`WHICH_MODEL`必须与检查点所预期的分词器保持一致。第 8 章的蒸馏检查点使用推理分词器，因此在这种情况下应保持`WHICH_MODEL = "reasoning"`。例如，对于第 6 章的检查点，你需要将其改为`WHICH_MODEL = "base"`。

直接编辑脚本中的配置可能看起来有些繁琐。之所以这样做，而不是使用命令行参数，是因为`Chainlit`本身通过`chainlit run ...`启动并管理应用程序入口。因此，该脚本的行为不像一个普通的独立`Python`程序，我们无法自由地使用`Python`的`argparse`库来定义和解析自定义命令行参数。对于模型选择、生成长度或检查点路径等简单选项，将设置保留在脚本中或通过环境变量传递会更加方便。

### 下载模型检查点

如果你想探索第 6、7 或 8 章中的某些模型检查点，但没有足够的计算能力自行训练，可以参阅补充资料中的下载说明，网址为[`github.com/rasbt/reasoning-from-scratch/tree/main/ch07/04_download_trainining_checkpoints`](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch07/04_download_trainining_checkpoints)和[`github.com/rasbt/reasoning-from-scratch/tree/main/ch08/05_download_training_checkpoints`](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch08/05_download_training_checkpoints)。

## G.6 多轮界面

第二个脚本`qwen3_chat_interface_multiturn.py`扩展了基本的`Chainlit`界面，使模型能够根据同一对话中的早期轮次进行条件生成。以下小节将说明多轮在实际中意味着什么、脚本如何实现多轮，以及何时适合使用多轮。

### G.6.1 多轮的含义

在单轮设置中，模型只能看到当前输入。为了说明，假设交互过程如下：

```
User: What is 1 + 1?
Assistant: 2
User: What did I just ask?
```

在普通的单轮脚本中，当模型接收到第二条消息“What did I just ask?”时，它可能会回答“我不知道你在问什么”。之前的问答虽然在浏览器中可见，但并未包含在模型输入中，因此模型并不真正拥有对话记忆。

在多轮设置（`qwen3_chat_interface_multiturn.py`）中，我们会显式地包含，更具体地说是*前置*之前的消息到提示词中，就像`ChatGPT`和其他聊天助手所做的那样。在这种情况下，模型可能会回答“What did I just ask?”为“你刚才问我 1+1 等于多少”。

### G.6.2 多轮变体

多轮脚本的起始方式与普通版本相同，但它增加了两个辅助函数，并在生成响应时使用存储的会话历史。

第一个辅助函数（如下清单所示）将存储的消息历史转换为所需的`Qwen`聊天格式。

#### 清单 G.3 从对话历史构建提示词

```
def build_prompt_from_history(history, add_assistant_header=True):
    parts = []
    for m in history:
        role = m["role"]
        content = m["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if add_assistant_header:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)
```

上述清单中的`build_prompt_from_history`函数将消息列表序列化为`Qwen`推理模型期望的聊天模板样式。从概念上讲，生成的提示词类似于：

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 1 + 1?<|im_end|>
<|im_start|>assistant
2<|im_end|>
<|im_start|>user
What did I just ask?<|im_end|>
<|im_start|>assistant
```

然后模型从这个最终的助手头部开始继续生成。

接下来展示的第二个辅助函数会在消息超过上下文长度时对其进行修剪，因为多轮对话可能会变得很长。

#### 清单 G.4 修剪提示词以适配上下文窗口

```
def trim_input_tensor(input_ids_tensor, context_len, max_new_tokens):
    assert max_new_tokens < context_len
    keep_len = max(1, context_len - max_new_tokens)
    if input_ids_tensor.shape[1] > keep_len: #1
        input_ids_tensor = input_ids_tensor[:, -keep_len:]
    return input_ids_tensor
```



# 1 如果提示词过长，则左侧截断至`keep_len`。

这个`trim_input_tensor`辅助函数确保上下文窗口为新 token 保留足够的空间。如果对话变得过长，最早生成的 token 会通过左侧截断被优先丢弃。（还有其他更复杂的选择，例如对之前的上下文进行总结，但这里为了简单起见并未实现。）

## G.6.3 多轮脚本如何使用历史记录

多轮行为出现在消息处理程序中。在每轮开始时，脚本会获取当前的会话历史，并将新的用户消息附加到其中：

```
history = chainlit.user_session.get("history")

history.append({"role": "user", "content": message.content})
```

然后，它根据该历史记录构建完整的提示，对其进行分词，按需截断，并像之前一样流式输出答案。在生成结束时，它将模型的回复存储回同一个历史记录列表中：

```
history.append({"role": "assistant", "content": out_msg.content})

chainlit.user_session.set("history", history)
```

这正是多轮脚本与单轮脚本的核心区别。多轮版本不仅仅是在浏览器中显示聊天记录；它还会将之前的轮次反馈给模型，以便未来的回复可以基于它们进行条件生成。

启动命令与单轮情况类似：

```
uv run chainlit run chG/01_main-chapter-code/qwen3_chat_interface_multiturn.py
```



在操作上，该脚本的行为与常规 Chainlit 界面完全相同，如图 G.3 所示。唯一的区别在于提示在内部是如何构建的。

![](img/APPG_F03_Raschka2.png)

#### 图 G.3 Chainlit 界面中多个提示和回复的示例响应

图 G.3 中的第二个回复表明，模型通过提供的上下文记住了第一个问题。

### G.6.4 建议

多轮 Chainlit 脚本是为官方 Qwen3 推理变体设计的。例如，它不太适合第 8 章的蒸馏检查点，因为这些检查点是在提示-回复风格的监督信号上训练的，而不是在持续的多轮对话历史上训练的。它们并没有像官方推理检查点那样经过训练，无法在多个轮次中延续对话。

因此，不应期望多轮脚本能与蒸馏检查点可靠地配合使用。我建议你按如下方式使用它们：

- 使用单轮 Chainlit 脚本进行快速的交互式测试，包括第 6、7、8 章的自定义检查点。
- 当你需要对话记忆时，使用多轮 Chainlit 脚本配合官方推理检查点。
