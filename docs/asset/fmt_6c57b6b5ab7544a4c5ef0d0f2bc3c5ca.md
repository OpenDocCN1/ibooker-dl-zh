
```py
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

completion = client.chat.completions.create(
  model="local-model",                           #1
  messages=[
    {"role": "system", "content": "Always answer in rhymes."},
    {"role": "user", "content": "Introduce yourself."}       #2
  ],
  temperature=0.7,
)

print(completion.choices[0].message)      #3
```

#1 目前未使用；可以是任何内容

#2 随意更改消息。

#3 默认代码输出整个消息。

如果你遇到连接到服务器的问题或遇到其他任何问题，请确保你的服务器模型设置配置与模型类型匹配。例如，在前面显示的图 2.6 中，加载的模型与服务器设置不同。修正后的设置如图 2.7 所示。

![图](img/2-7.png)

##### 图 2.7 选择加载模型的正确服务器模型设置

现在，你可以使用本地托管的 LLM 或商业模型来构建、测试，甚至可能运行你的代理。下一节将探讨如何更有效地使用提示工程构建提示。

## 2.3 使用提示工程提示语言模型

为 LLM 定义的提示是用于请求以获得更好的响应输出的消息内容。“提示工程”是一个新兴的领域，试图为构建提示构建一种方法论。不幸的是，提示构建不是一个成熟的科学，而且定义了越来越多的方法，这些方法被称为提示工程。

幸运的是，像 OpenAI 这样的组织已经开始记录一套通用的策略，如图 2.8 所示。这些策略涵盖了各种战术，其中一些需要额外的基础设施和考虑。因此，与更高级概念相关的提示工程策略将在指定的章节中介绍。

![图片](img/2-8.png)

##### 图 2.8 本书中按章节位置审查的 OpenAI 提示工程策略

图 2.8 中的每个策略都展开为可以进一步细化特定提示工程方法的战术。本章将检查基本的写清晰指令策略。图 2.9 更详细地展示了该策略的战术，以及每个战术的示例。我们将在下一节中查看如何使用代码演示运行这些示例。

![图片](img/2-9.png)

##### 图 2.9 写清晰指令策略的战术

写清晰指令策略是关于在要求时保持谨慎和具体。要求一个语言模型（LLM）执行一项任务与要求一个人完成同样的任务并无不同。一般来说，你能在请求中指定与任务相关的更多信息和上下文，那么响应就会越好。

这种策略已经被分解为可以应用于提示的具体战术。为了理解如何使用这些战术，一个包含各种提示示例的代码演示（`prompt_engineering.py`）位于“第二章”源代码文件夹中。

在 VS Code 中打开`prompt_engineering.py`文件，如图 2.8 所示。此代码首先加载`prompts`文件夹中的所有 JSON Lines 文件。然后，它显示文件列表作为选择，并允许用户选择提示选项。选择选项后，提示被提交给 LLM，并打印出响应。

##### 列表 2.8 `prompt_engineering.py` `(main())`

```py
def main():
    directory = "prompts"
    text_files = list_text_files_in_directory(directory)    #1

    if not text_files:
        print("No text files found in the directory.")
        return

    def print_available():                                     #2
        print("Available prompt tactics:")
        for i, filename in enumerate(text_files, start=1):
            print(f"{i}. {filename}")

    while True:
        try:
            print_available()                   #2              
            choice = int(input("Enter … 0 to exit): "))           #3
            if choice == 0:
                break
            elif 1 <= choice <= len(text_files):
                selected_file = text_files[choice - 1]
                file_path = os.path.join(directory,
      selected_file)
                prompts = 
↪ load_and_parse_json_file(file_path)                          #4
                print(f"Running prompts for {selected_file}")
                for i, prompt in enumerate(prompts):
                    print(f"PROMPT {i+1} --------------------")
                    print(prompt)
                    print(f"REPLY ---------------------------")
                    print(prompt_llm(prompt))                       #5
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")
```

#1 收集给定文件夹中的所有文件

#2 打印文件列表作为选择

#3 输入用户的选择

#4 加载提示并将其解析为消息

#5 将提示提交给 OpenAI 语言模型

列表中的注释部分演示了如何连接到本地 LLM。这将允许你探索应用于本地运行的开源 LLM 的相同提示工程策略。默认情况下，此示例使用我们在 2.1.1 节中配置的 OpenAI 模型。如果你没有完成之前的操作，请返回并完成它，然后再运行此示例。

图 2.10 显示了运行提示工程策略测试器的输出，即 VS Code 中的 `prompt_engineering.py` 文件。当你运行测试器时，你可以为要测试的策略输入一个值，并观察其运行。

![图](img/2-10.png)

##### 图 2.10 提示工程策略测试器的输出

在以下章节中，我们将更详细地探讨每个提示策略。我们还将检查各种示例。

### 2.3.1 创建详细查询

此策略的基本前提是尽可能提供详细的信息，但也要小心不要提供无关紧要的细节。以下列表显示了用于探索此策略的 JSON Lines 文件示例。

##### 列表 2.9 `detailed_queries.jsonl`

```py
[                        #1
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "What is an agent?"      #2
    }
]
[
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": """
What is a GPT Agent? 
Please give me 3 examples of a GPT agent
"""                                        #3
    }
]
```

#1 第一个示例没有使用详细的查询。

#2 首先向 LLM 提出一个非常一般的问题。

#3 提出一个更具体的问题，并请求示例。

此示例演示了使用详细查询和不使用查询之间的差异。它还进一步通过请求示例。记住，你能在提示中提供越多相关性和上下文，整体响应就越好。请求示例是加强问题与预期输出之间关系的一种方式。

### 2.3.2 采用角色

采用角色赋予 LLM 定义一个总体上下文或一组规则的能力。LLM 可以然后使用该上下文和/或规则来构建所有后续的输出响应。这是一个有吸引力的策略，我们将在整本书中大量使用它。

列表 2.10 显示了使用两个角色回答相同问题的示例。这可以是一种探索广泛新颖应用的愉快技术，从获取人口统计反馈到专门从事特定任务，甚至橡皮鸭技术。

##### GPT 橡皮鸭

*橡皮鸭* 是一种问题解决技术，其中一个人向一个无生命物体（如橡皮鸭）解释一个问题，以理解或找到解决方案。这种方法在编程和调试中很常见，因为大声阐述问题往往有助于澄清问题，并可能导致新的见解或解决方案。

GPT 橡皮鸭技术使用相同的技巧，但使用的是 LLM 而不是无生命物体。通过给 LLM 赋予特定于所需解决方案领域的角色，这种策略可以进一步扩展。

##### 列表 2.10 `adopting_personas.jsonl`

```py
[
    {
        "role": "system",
        "content": """
You are a 20 year old female who attends college 
in computer science. Answer all your replies as 
a junior programmer.
"""                         #1
    },
    {
        "role": "user",
        "content": "What is the best subject to study."
    }
]
[
    {
        "role": "system",
        "content": """
You are a 38 year old male registered nurse. 
Answer all replies as a medical professional.
"""                                             #2
    },
    {
        "role": "user",
        "content": "What is the best subject to study."
    }
]
```

#1 第一个角色

#2 第二个角色

代理配置文件的核心元素是角色。我们将使用各种角色来帮助代理完成他们的任务。当你运行此策略时，请特别注意 LLM 输出响应的方式。

### 2.3.3 使用分隔符

分隔符是一种有用的方法，可以隔离并使 LLM 专注于消息的某些部分。这个策略通常与其他策略结合使用，但也可以独立工作。以下列表展示了两个示例，但还有其他几种描述分隔符的方法，从 XML 标签到使用 Markdown。

##### 列表 2.11 `using_delimiters.jsonl`

```py
[
    {
        "role": "system",
        "content": """
Summarize the text delimited by triple quotes 
with a haiku.
"""               #1
    },
    {
        "role": "user",
        "content": "A gold chain is cool '''but a silver chain is better'''"
    }
]
[
    {
        "role": "system",
        "content": """
You will be provided with a pair of statements 
(delimited with XML tags) about the same topic. 
First summarize the arguments of each statement. 
Then indicate which of them makes a better statement
 and explain why.
"""                        #2
    },
    {
        "role": "user",
        "content": """
<statement>gold chains are cool</statement>
<statement>silver chains are better</statement>
"""
    }
]
```

#1 分隔符由字符类型和重复定义。

#2 分隔符由 XML 标准定义。

当你运行这个策略时，请注意 LLM 在输出响应时关注的文本部分。这个策略对于描述层次结构或其他关系模式的信息非常有用。

### 2.3.4 指定步骤

指定步骤是另一种强大的策略，可以有很多用途，包括在代理中，如列表 2.12 所示。在开发复杂多步骤任务的提示或代理配置文件时，它尤其强大。你可以指定步骤将复杂的提示分解为 LLM 可以遵循的逐步过程。反过来，这些步骤可以引导 LLM 在更长时间的对话和多次迭代中进行多次交互。

##### 列表 2.12 `specifying_steps.jsonl`

```py
[
    {
        "role": "system",
        "content": """
Use the following step-by-step instructions to respond to user inputs.
Step 1 - The user will provide you with text in triple single quotes. 
Summarize this text in one sentence with a prefix that says 'Summary: '.
Step 2 - Translate the summary from Step 1 into Spanish, 
with a prefix that says 'Translation: '.
"""                                          #1
    },
    {
        "role": "user",
        "content": "'''I am hungry and would like to order an appetizer.'''"
    }
]
[
    {
        "role": "system",
        "content": """
Use the following step-by-step instructions to respond to user inputs.
Step 1 - The user will provide you with text. Answer any questions in 
the text in one sentence with a prefix that says 'Answer: '.

Step 2 - Translate the Answer from Step 1 into a dad joke,
 with a prefix that says 'Dad Joke: '."""                      #2
    },
    {
        "role": "user",
        "content": "What is the tallest structure in Paris?"
    }
]
```

#1 注意使用分隔符的策略。

#2 步骤可以是完全不同的操作。

### 2.3.5 提供示例

提供示例是引导 LLM 期望输出的绝佳方式。有无数种方法可以向 LLM 展示示例。系统消息/提示可以是一种强调一般输出的有用方式。在下面的列表中，示例被添加为最后一个 LLM 助手回复，提示为“教我关于 Python。”

##### 列表 2.13 `providing_examples.jsonl`
