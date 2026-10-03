
如我们所见，模型正在被展示如何使用特定格式的表情符号来响应，例如`(sadkoala)`、`(tired)`和`(like)`。

```py
{"messages": [{"role": "system", "content": "You're a chatbot that only responds with emojis!"}, {"role": "user", "content": "I heard the koala habitat was destroyed by the fire."}, {"role": "assistant", "content": "(sadkoala)" }]}

{"messages": [{"role": "system", "content": "You're a chatbot that only responds with emojis!"}, {"role": "user", "content": "I've been working nonstop and need a break."}, {"role": "assistant", "content": "(tired)" }]}

{"messages": [{"role": "system", "content": "You're a chatbot that only responds with emojis!"}, {"role": "user", "content": "I just finished reading an amazing book!"}, {"role": "assistant", "content": "(like)" }]}
```

## 基本检查

在微调之前，对训练数据进行基本检查非常重要，以避免浪费时间和资源。这些检查可以包括数据可读性、格式验证、缺失对的分析以及令牌长度。

我们通过使用`basic_checks()`函数加载和读取数据文件来验证数据文件。该函数接受一个文件名作为输入，并返回找到的消息数量。消息必须采用聊天完成格式，以便微调 GPT-3.5 Turbo。

### 列表 9.2 数据集验证：基本检查

```py
# Basic checks to ensure the data file is valid
def basic_checks(data_file):
    try:
        with open(data_file, 'r', encoding='utf-8') as f:    #1
            dataset = [json.loads(line) for line in f]       #2

        print(f"Basic checks for file {data_file}:")
        print("Count of examples in training dataset:", len(dataset))
        print("First example:")                              #3
        for message in dataset[0]["messages"]:             #4
            print(message)
        return True
    except Exception as e:
        print(f"An error occurred in file {data_file}: {e}")
        return False
```

#1 以读取模式打开文件

#2 将文件的每一行加载为 JSON 对象，并将其存储在列表中

#3 打印数据集中的第一个示例，并帮助视觉检查事物是否直观上看起来正常

#4 遍历第一个示例中的消息并打印每个消息

## 格式检查

在完成基本检查后，下一步是检查文件格式，并确保在进一步处理之前它被正确地结构化。这是一个重要的步骤，主要是因为即使格式不正确，我们在开始训练作业时也不会得到错误，但生成的模型将非常糟糕，我们只有在部署后才会意识到这一点。为了避免许多这样的麻烦，强烈建议我们检查格式。

列表 9.3 展示了`format_checks()`，它检查聊天完成格式和配对，其两个参数是数据集和文件名。它捕获了大多数错误，但不是所有错误。该函数遍历数据集中的每个示例，并检查数据类型检查、消息列表的存在和消息键。它验证是否有相关的角色和内容验证。此函数还有助于调试数据相关的问题。

### 列表 9.3 数据集验证：检查格式

```py
def format_checks(dataset, filename):
    # Initialize a dictionary used to track format errors
    format_errors = defaultdict(int)

    # Iterate over each example in the dataset
    for ex in dataset:
        # Check if the example is a dictionary, if not 
        # increment the corresponding error count
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        # Check if the example has a "messages" key, 
        # if not increment the corresponding error count
        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        # Iterate over each message
        for message in messages:
            # Check if the message has "role" and "content" keys,
            # if not increment the corresponding error count
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            # Check if the message has any unrecognized keys,
            # if so increment the corresponding error count
            if any(k not in ("role", "content", "name", 
                   ↪"function_call") for k in message):
                format_errors["message_unrecognized_key"] += 1

            # Check if the role of the message is one of the recognized
            # roles, if not increment the corresponding error count
            if message.get("role", None) not in (
                "system",
                "user",
                "assistant",
                "function",
            ):
                format_errors["unrecognized_role"] += 1

            # Check if the message has either content or a function call, 
            # and if the content is a string, if not increment the 
            # corresponding error count
            content = message.get("content", None)
            function_call = message.get("function_call", None)
            if (not content and not function_call) or not 
            ↪isinstance(content, str):
                format_errors["missing_content"] += 1

        # Check if there is at least one message with the role "assistant",
        # if not increment the corresponding error count
        if not any(message.get("role", None) == "assistant" 
        ↪for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    # If there are any format errors, print them and return False
    if format_errors:
        print(f"Formatting errors found in file {filename}:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
        return False

    print(f"No formatting errors found in file {filename}")
    return True
```

最后，我们还应该了解数据集在简单数据分布、令牌计数和成本方面的表现。

注意：令牌计数很重要，不仅仅是因为成本。如果它超过了模型可以处理的令牌最大数量，它将被截断而不会发出警告。提前了解这一点非常有帮助。

下面的列表显示了我们可以如何完成对数据集的检查。

### 列表 9.4 数据集验证：成本估算和基本分析

```py
# Pricing and default n_epochs estimate
MAX_TOKENS = 4096

TARGET_EPOCHS = 3
MIN_TARGET_EXAMPLES = 100
MAX_TARGET_EXAMPLES = 25000
MIN_DEFAULT_EPOCHS = 1
MAX_DEFAULT_EPOCHS = 25

def estimate_tokens(dataset, assistant_tokens):
    # Set the initial number of epochs to the target epochs
    n_epochs = TARGET_EPOCHS

    # Get the number of examples in the dataset
    n_train_examples = len(dataset)

    # If the examples total is less than the minimum target
    # adjust the epochs to ensure we have enough examples for
    # training
    if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
        n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES 
           ↪// n_train_examples)
    # If the  number of examples is more than the maximum target
    # adjust the  epochs to ensure we don't exceed the maximum 
    # for training
    elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
        n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES 
          ↪// n_train_examples)

    # Calculate the total number of tokens in the dataset
    n_billing_tokens_in_dataset = sum(
        min(MAX_TOKENS, length) for length in assistant_tokens
    )

    # Print the total token count that will be charged during training
    print(
        f"Dataset has ~{n_billing_tokens_in_dataset} tokens that 
            ↪will be charged for during training"
    )

    # Print the default number of epochs for training
    print(f"You will train for {n_epochs} epochs on this dataset")

    # Print the total number of tokens that will be charged during training
    print(f"You will be charged for ~{n_epochs * 
                        ↪n_billing_tokens_in_dataset} tokens")

    # If the total token count exceeds the maximum tokens, print a warning 
    if n_billing_tokens_in_dataset > MAX_TOKENS:
        print(
            f"WARNING: Your dataset contains examples longer than 
                       ↪4K tokens by {n_billing_tokens_in_dataset – 
                       ↪MAX_TOKENS} tokens."
        )
        print(
            "You will be charged for the full length of these 
             ↪examples during training, but only the first
             ↪4K tokens will be used for training."
```

## LLM 评估

评估 LLM 对于确保其质量、可靠性和公平性非常重要。然而，评估 LLM 是复杂的，因为它涉及多个维度和挑战。维护多样化的自动指标可以帮助在适应周期中有效地跟踪模型改进，同时减少昂贵的手动审查。指标应根据每个适应模型的使用案例和业务需求进行定制。从生产系统持续记录可以启用对随时间推移的实时性能的评估。

将微调后的 GPT 模型与基线进行基准测试是评估微调模型的一个重要步骤。这涉及到将微调模型的性能与预先建立的标准或基线模型进行比较。这个基线可能是微调前的模型性能或在不同任务中因其熟练度而知名的不同模型。这种比较的目的是量化微调带来的改进。例如，在语言翻译任务中，微调后的模型可能被与标准翻译模型进行基准测试，以评估翻译准确度或流畅性的改进。这个过程有助于了解微调的有效性，并确定模型改进或仍需增强的领域。

### 评估标准

在准备微调数据集时，我们还应该定义评估标准。在微调过程中，评估过程首先通过建立明确的评估标准来开始，这些标准对于评估模型在其预期应用中的性能和有效性至关重要。这些标准通常包括相关性、连贯性、准确性和语言流畅性（表 9.1）。

#### 表 9.1 微调评估标准

| 评估标准 | 描述 |
| --- | --- |
| 相关性 | 衡量模型响应或输出与输入上下文和意图的匹配程度。这在聊天机器人等应用中尤为重要，在这些应用中，提供上下文适当的响应对于用户满意度至关重要。相关性通常通过检查模型是否能够保持主题并提供与查询或任务直接相关的信息或响应来评估。 |
| 连贯性 | 指的是模型输出的逻辑一致性。一个微调后的模型应该生成上下文相关、逻辑合理且连贯的文本。这意味着响应应该遵循逻辑结构和叙事流程，在对话或文本上下文中有意义。连贯性对于保持用户参与度和确保模型输出可理解且有意义至关重要。 |
