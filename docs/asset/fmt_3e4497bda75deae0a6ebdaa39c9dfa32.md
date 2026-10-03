
创建 PlayGround 本地版本的原因是为了演示代码结构，同时也提供以下列出的额外功能：

+ **行为（自定义行为）**——创建您自己的行为允许您向助手添加任何您想要的功能。正如我们将看到的，PlayGround 使创建您自己的行为变得非常简单。
+ **代码运行器**——API 确实附带了一个代码解释器，但它相对昂贵（每次运行 0.03 美元），不允许你安装你的模块，不能以交互方式运行代码，并且运行速度较慢。游戏场将使你能够在隔离的虚拟环境中本地运行 Python 代码。虽然不如将代码推送到 Docker 镜像安全，但它比其他平台更好地执行了代码窗口化和进程外执行。
+ **透明度和日志记录**——游戏场提供了全面的日志捕获功能，甚至可以显示助手如何使用内部和外部工具/操作。这可以是一个查看助手幕后所做事情的优秀方式。

这些功能将在接下来的几节中更详细地介绍。我们将在下一节中开始查看使用和消费操作。

## 6.2.2 使用和构建自定义操作

操作和工具是赋予代理和助手能力的基石。没有工具的访问权限，代理就变成了无功能的聊天机器人。OpenAI 平台在建立许多工具模式方面处于领先地位，正如我们在第三章中看到的。

游戏场提供了几个可以通过界面附加到助手的自定义操作。在接下来的练习中，我们将构建一个简单的助手并附加几个自定义操作，以查看可以做到什么程度。

图 6.4 显示了扩展的操作手风琴，它显示了许多可用的自定义操作。从终端或调试器运行游戏场，创建一个新的助手。然后，选择图中的操作。在你完成选择操作后，滚动到页面底部，点击添加助手以添加助手。助手在使用之前需要被创建。

![figure](img/6-4.png)

**图 6.4 在界面中选择和使用自定义操作**

在你创建助手之后，你可以要求它列出所有可用的助手。列出助手也会给你调用助手所需的 ID。你还可以调用其他助手并要求它们在其专业领域内完成任务。

添加自定义操作就像向文件中添加代码并将其放入正确的文件夹一样简单。从主项目文件夹中打开`playground/assistant_actions`文件夹，你会看到定义各种操作的几个文件。在 VS Code 中打开`file_actions.py`文件，如图 6.4 所示。

**列表 6.4 `playground/assistant_actions/file_actions.py`**

```py
import os

from playground.actions_manager import agent_action

OUTPUT_FOLDER = "assistant_outputs"

@agent_action     #1
def save_file(filename, content):      #2
    """
    Save content to a file.      #3

    :param filename: The name of the file including extension.
    :param content: The content to save in the file.
    """
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)
    print(f"File '{filename}' saved successfully.")      #4
```

#1 这个装饰器会自动将函数添加为操作。

#2 给你的函数起一个清晰且与其目的相符的名字。

#3 描述是助手用来确定函数的内容，所以请好好记录。

#4 通常返回一个表示成功或失败的消息

您可以通过将文件放置在`assistant_actions`文件夹中，并用`agent_action`装饰器装饰它来添加任何您想要的自定义操作。只需确保给函数起一个好名字，并输入关于如何使用函数的良好文档。当 Playground 启动时，它会加载文件夹中所有正确装饰并带有描述/文档的操作。

这很简单。您可以根据需要添加多个自定义操作。在下一节中，我们将探讨一个特殊的自定义操作，允许助手在本地运行代码。

## 6.2.3 安装助手数据库

要运行本章中的多个示例，您需要安装助手数据库。幸运的是，这可以通过界面轻松完成，只需询问代理即可。即将到来的说明详细介绍了安装助手的过程，并直接来自 GPT 助手 Playground 的 README。您可以从位于`assistants.db` SQLite 数据库中的`assistants.db`安装几个演示助手：

1. 创建一个新的助手，或使用现有的助手。
2. 给助手分配`create_manager_assistant`操作（在操作部分下找到）。
3. 请助手创建管理助手（即，“请创建管理助手”），并确保将助手的名称命名为“管理助手”。
4. 刷新您的浏览器以重新加载助手选择器。
5. 选择新的管理助手。这个助手拥有安装助手的指令和操作，这些操作将允许它从`assistants.db`数据库中安装助手。
6. 与管理助手交谈，以获取要安装的助手列表，或者直接请管理助手安装所有可用的助手。

## 6.2.4 获取本地运行代码的助手

使代理和助手生成和运行可执行代码具有很大的能力。与代码解释器不同，本地运行代码提供了快速迭代和调整的众多机会。我们之前在 AutoGen 中看到了这一点，代理可以持续运行代码，直到它按预期工作。

在 Playground 中，选择自定义操作`run_code`非常简单，如图 6.5 所示。您还希望选择`run_shell_command`操作，因为它允许助手`pip install`任何所需的模块。

![figure](img/6-5.png)

**图 6.5 选择助手运行 Python 代码的自定义操作**

您现在可以请助手生成和运行代码，以确保它代表您工作。通过添加自定义操作并请助手生成和运行代码来尝试这一点，如图 6.6 所示。如果代码没有按预期工作，告诉助手您遇到的问题。

![figure](img/6-6.png)

**图 6.6 使助手生成和运行 Python 代码**

再次强调，沙盒中运行的 Python 代码会在项目子文件夹中创建一个新的虚拟环境。如果你没有运行任何操作系统级别的代码或底层代码，这个系统工作得很好。如果你需要更健壮的解决方案，一个好的选择是 AutoGen，它使用 Docker 容器来运行隔离的代码。

添加运行代码或其他任务的动作可以使助手看起来像一个黑盒。幸运的是，OpenAI 助手 API 允许你消费事件并查看助手在幕后做了什么。在下一节中，我们将看到这看起来是什么样子。

## 6.2.5 通过日志调查助手进程

OpenAI 将一个功能添加到了助手 API 中，允许你监听通过工具/动作使用链式的事件和动作。这个功能已经集成到沙盒中，当助手调用另一个助手时，它会捕获动作和工具的使用。

我们可以通过要求助手使用一个工具并打开日志来尝试这一点。一个很好的例子是给助手提供代码解释器工具，然后要求它绘制一个方程式。图 6.7 展示了这个练习的一个例子。

![figure](img/6-7.png)

**图 6.7 内部助手日志被捕获**

通常情况下，当辅助代码解释器工具被启用时，你不会看到任何代码生成或执行。这个功能允许你看到助手在执行过程中使用的所有工具和动作。这不仅是一个出色的诊断工具，而且还能提供对 LLM（大型语言模型）功能的额外见解。

我们没有审查执行所有这些操作所需的代码，因为代码量很大，并且可能经历几次变化。话虽如此，如果你计划使用助手 API，这个项目是一个很好的起点。随着沙盒的引入，我们可以在下一节继续我们的 ABT 之旅。

## 6.3 介绍代理行为树

代理行为树（ABTs）在助手和代理系统中实现行为树。与常规行为树相比，ABTs 的关键区别在于它们使用提示来指导动作和条件。由于提示可能会返回高频率的随机结果，我们也可以将这些树称为*随机*行为树，这种树确实存在。为了简单起见，我们将用于控制代理的行为树区分开来，称其为代理行为树。

接下来，我们将进行一个练习来创建一个 ABT（代理行为树）。完成的树将以 Python 编写，但需要设置和配置各种助手。我们将介绍如何使用助手本身来管理助手。

### 6.3.1 使用助手管理助手

幸运的是，沙盒可以帮助我们快速管理和创建助手。我们首先安装管理助手，然后安装预定义的助手。让我们按照以下步骤开始安装管理助手：

1. 在你的浏览器中打开游乐场，创建一个新的简单助理或使用现有的助理。如果你需要一个新助理，创建它然后选择它。
2. 在选择助理后，打开操作面板，并选择`create_manager_assistant`操作。你不需要保存；界面将自动更新助理。
3. 现在，在聊天界面中，提示助理以下内容：“请创建经理助理。”
4. 几秒钟后，助理会说它已经完成。刷新你的浏览器，并确认经理助理现在可用。如果由于某种原因，新助理没有显示，尝试重新启动 Gradio 应用程序本身。

经理助理就像一个可以访问所有内容的管理员。在启用经理助理时，请确保你的请求具体明确。当经理助理处于激活状态时，你可以按照以下步骤安装书中使用的新助理：

1. 选择经理助理。如果你修改了经理助理，你可以随时删除并重新安装它。尽管可以拥有多个经理助理，但并不推荐这样做。
2. 在聊天界面中输入以下内容以询问经理助理可以安装哪些助理：

```py
Please list all the installable assistants.
```

3. 当你要求经理助理安装助理时，确定你想要安装哪个助理：

```py
Please install the Python Coding Assistant.
```

你可以使用游乐场管理并安装任何可用的助理。你还可以要求经理助理将所有助理的定义保存为 JSON：

```py
Please save all the assistants as JSON to a file called assistants.json.
```

经理助理可以访问所有操作，这些操作应被视为独特且应谨慎使用。在制作助理时，最好保持它们的目标特定，并将操作限制在它们需要的范围内。这不仅避免了给 AI 过多的决策，也避免了由幻觉引起的事故或错误。

在我们完成本章剩余的练习时，你可能需要安装所需的助理。或者，你可以要求经理助理安装所有可用的助理。无论哪种方式，我们将在下一节中查看使用助理创建 ABT。

### 6.3.2 构建编码挑战 ABT

编码挑战为测试和评估代理和助理系统提供了一个良好的基准。挑战和基准可以量化代理或代理系统的工作效果。我们已经在第四章中应用了编码挑战到多平台代理 AutoGen 和 CrewAI。

对于这个编码挑战，我们将更进一步，查看来自 Edabit 网站（[`edabit.com`](https://edabit.com)）的 Python 编码挑战，这些挑战的复杂度从入门级到专家级不等。我们将坚持使用专家级代码挑战，因为 GPT-4o 和其他模型都是优秀的编码者。查看下一列表中的挑战，并思考你会如何解决它。

**列表 6.5 Edabit 挑战：种植草地**

```py
Plant the Grass by AniXDownLoe

    You will be given a matrix representing a field g 
and two numbers x, y coordinate.

    There are three types of possible characters in the matrix:

        x representing a rock.
        o representing a dirt space.
        + representing a grassed space.

    You have to simulate grass growing from the position (x, y). 
    Grass can grow in all four directions (up, left, right, down). 
    Grass can only grow on dirt spaces and can't go past rocks.

    Return the simulated matrix.
    Examples

    simulate_grass([
    "xxxxxxx",
    "xooooox",
    "xxxxoox"
    "xoooxxx"
    "xxxxxxx"
    ], 1, 1) → [
    "xxxxxxx",
    "x+++++x",
    "xxxx++x"
    "xoooxxx"
    "xxxxxxx"
    ]

    Notes

    There will always be rocks on the perimeter
```

你可以使用任何你想要的挑战或编码练习，但以下是一些需要考虑的事项：

+ 挑战应该可以通过可量化的断言（通过/失败）进行测试。
+ 避免在请求游戏、构建网站或使用其他界面时打开窗口。在某个时候，测试完整界面将是可能的，但到目前为止，它只是文本输出。
+ 避免进行长时间运行的挑战，至少最初是这样。开始时，保持挑战简短且生命周期短暂。

任何挑战之外，你还会想要一组测试或断言来确认解决方案的有效性。在 Edabit 上，一个挑战通常提供一套全面的测试。以下列表显示了与挑战一起提供的附加测试。

**列表 6.6 种植草地测试**

```py
Test.assert_equals(simulate_grass(
["xxxxxxx","xooooox","xxxxoox","xoooxxx","xxxxxxx"],
 1, 1), 
["xxxxxxx","x+++++x","xxxx++x","xoooxxx","xxxxxxx"])
    Test.assert_equals(simulate_grass(
["xxxxxxx","xoxooox","xxoooox","xooxxxx",
"xoxooox","xoxooox","xxxxxxx"],
 2, 3), ["xxxxxxx","xox+++x","xx++++x","x++xxxx",
"x+xooox","x+xooox","xxxxxxx"])
    Test.assert_equals(simulate_grass(
["xxxxxx","xoxoox","xxooox","xoooox","xoooox","xxxxxx"], 
1, 1), 
["xxxxxx","x+xoox","xxooox","xoooox","xoooox","xxxxxx"])
    Test.assert_equals(simulate_grass(
["xxxxx","xooox","xooox","xooox","xxxxx"], 
1, 1),
["xxxxx","x+++x","x+++x","x+++x","xxxxx"])
    Test.assert_equals(simulate_grass(
["xxxxxx","xxxxox","xxooox","xoooxx","xooxxx",
"xooxxx","xxooox","xxxoxx","xxxxxx"], 
4, 1),
["xxxxxx","xxxx+x","xx+++x","x+++xx","x++xxx",
"x++xxx","xx+++x","xxx+xx","xxxxxx"])
    Test.assert_equals(simulate_grass(
["xxxxxxxxxxx", "xoxooooooox", "xoxoxxxxxox", 
"xoxoxoooxox", "xoxoxoxoxox", "xoxoxoxoxox", 
"xoxoxxxoxox", "xoxoooooxox", "xoxxxxxxxox", 
"xooooooooox", "xxxxxxxxxxx"], 1, 1), 
["xxxxxxxxxxx", "x+x+++++++x", "x+x+xxxxx+x", 
"x+x+x+++x+x", "x+x+x+x+x+x", "x+x+x+x+x+x", 
"x+x+xxx+x+x", "x+x+++++x+x", "x+xxxxxxx+x", 
"x+++++++++x", "xxxxxxxxxxx"])
```

测试将作为两步验证的一部分运行，以确认解决方案的有效性。我们还将使用所写的测试和挑战，这将进一步测试 AI。

图 6.8 显示了将用于解决各种编程挑战的简单行为树的组成。你会注意到这个 ABT 为动作和条件使用了不同的助手。对于第一步，Python 编码助手（称为黑客）生成一个解决方案，然后由编码挑战裁判（称为裁判）进行审查，产生一个经过改进的解决方案，然后由另一个 Python 编码助手（称为验证器）进行验证。

![figure](img/6-8.png)

**图 6.8 编码挑战的 ABT**

图 6.8 还显示了每个代理在哪个线程上进行交流。助手使用消息线程，类似于 Slack 或 Discord 频道，所有在某个线程上交流的助手将看到所有消息。对于这个 ABT，我们保留一个主要对话线程供黑客和裁判共享消息，而验证器则在单独的消息线程上工作。将验证器保持在单独的线程上可以将其从解决方案的努力中的噪音中隔离出来。

现在，在代码中构建 ABT 是一个将`py_trees`包和 Playground API 函数结合起来的问题。列表 6.7 显示了创建每个动作/条件节点并给助手下达指令的代码摘录。

**列表 6.7 `agentic_btree_coding_challenge.py`**
