

# 第十二章：未来方向

## 本章涵盖

- 使用提示模式编写和解释代码
- 生成式 AI 工具的当前局限性和未来方向

在这一章的最后，我们希望让你一窥人们目前正在使用生成式 AI 工具（如 GitHub Copilot 和 ChatGPT）的创造性方式。例如，可以使得 Copilot Chat 向你提问，而不是反过来。还可以让 Copilot 扮演不同的角色，以更好地帮助你完成当前的编程任务。我们将简要介绍，目前还不清楚这将成为多少标准做法，但我们想借此机会展示这些新工具的创造力。我们还将讨论一些生成式 AI 工具的当前局限性（你已经在本书中看到了一些！），并分享我们对未来可能性的看法。

## 12.1 提示模式

在整本书中，我们阐述了为什么直接在 Python 中编程与使用 Copilot 编程相比有着非常不同的体验。我们关注的焦点不是编写代码，而是编写提示并与生成的代码交互，以确定其是否正确，并在需要时进行修复。但伴随着这些差异，没有 Copilot 的编码与有 Copilot 的编码之间也存在令人惊讶的相似之处。

当程序员编写代码时，他们不会每次编写新程序时都从头开始。研究人员和程序员已经创建了设计**模式**的目录，即组织程序以使其更容易编写、调试和扩展的通用方法。最著名的此类目录是一本名为《设计模式：可重用面向对象软件元素》的书[1]，也被称为“四人帮”书，因为它是由四位作者共同撰写的。它需要熟悉面向对象编程，而我们在这本书中没有教授，所以我们不建议你马上拿起那本书。只需知道，成千上万的程序员已经通过使用这本书中的模式节省了数千小时，而没有重新发明轮子。

以设计模式的一个例子来说明，假设你正在编写一个电脑游戏，其中人类玩家与电脑对战。你希望实现几个不同难度（例如，初学者、中级、高级和专家）的 AI 对手，为人类提供不同技能水平的对手。每个 AI 对手都将有自己的代码来决定其行为。使用`if`条件确定 AI 对手应该做什么是可能的：

```py
if ai_opponent == 'beginner':
    # make decision consistent with beginner AI opponent
elif ai_opponent == 'intermediate':
    # make decision consistent with intermediate AI opponent
…
```

但那样做的缺点是组织上的。我们会把所有 AI 代码放在同一个地方。我们从这本书中学到的知识告诉我们，像那样的大型函数不容易设计和测试。

组织这类程序的干净方式是使用一种称为*策略模式*的模式。我们不会在这里详细介绍该模式，但重要的是，人们已经为他人使用记录了这种模式[1]。模式文档会告诉你其意图（目的）、我们为什么要使用它的动机、实现它所需的代码结构以及示例代码。

研究人员开始整理可用于 GitHub Copilot 和 ChatGPT 等生成 AI 工具的模式。它们被称为*提示模式*，类似于设计模式，它们告诉你如何构建一个提示以实现一个期望的目标。每个这些模式的文档与设计模式的文档类似，但与示例代码不同，我们得到的是我们可以使用的示例提示。

在这本书中，我们大部分时间都在关注两种类型的 AI 响应：生成代码（通过在 VS Code 中按 Tab 键）和解释代码（通过使用 Copilot Chat）。新兴的提示模式可以帮助生成 AI 工具超越这些，帮助处理其他与代码相关的任务。

为了让你了解提示模式的前沿状态，我们将在这里采样几个。对于更多，我们鼓励你查看我们使用的提示模式目录[2]。但请注意：我们正在引用一篇学术论文，而不是入门指南，因为我们目前不知道有任何入门指南（这还是新事物）。

### Copilot Chat 可能不会参与这些提示模式

在撰写本文时，Copilot Chat 拒绝参与一些已记录的提示模式，声称它们与编程辅助无关。虽然其中一些可能不太具体地与编程相关，但它们都是为了帮助程序员设计的。与 ChatGPT 不同，Copilot 被设计成始终试图专注于编程主题。如果你决定进一步探索各种模式，你可能希望在无法与 Copilot 有效互动时使用 ChatGPT。

### 12.1.1 交互模式反转

到目前为止，我们一直是请求代码和提问的人，而 Copilot 一直是回答的人。但我们可以反转脚本，让 Copilot 提问，我们回答。这被称为*交互模式反转*。

此模式的目的是在我们不知道 AI 需要什么信息来完成任务时，帮助我们与之互动。动机是，AI 可能知道如何更快地通过询问我们达到其目标，而不是如果我们提供不完整或不帮助的信息。我们的提示结构将看起来像这样：

```
我希望你问我问题以达到 X。你应该继续提问，直到我们达到这个目标，其中 X 是我们想要达到的目标。
```

这种结构并不是为了精确地规定我们应该使用的提示。相反，它提示我们需要在提示中传达的关键思想。例如，这里的关键思想是我们希望 Copilot 提问，并且我们希望它在我们达到目标之前不要停止。

在第三章中，我们编写了一个函数，不断询问用户输入密码，直到他们输入了一个强大的密码。假设我们不确定如何设计这个函数。我们可以通过提供代码提示让 Copilot 为我们设计它，但这样它可能会做出我们不认同的决定。相反，我们可以让 Copilot 提问。让我们使用 Copilot Chat 来做这件事。

**LPHuman** 我想要一个 Python 函数，它会不断询问用户输入密码，直到他们提供了一个强大的密码。在你能写出那个 Python 函数之前，一次问我一个问题。

![chatGpt](img/chatGpt.png)**GitHub Copilot** 当然，我很乐意帮助你编写一个不断询问用户输入强大密码的 Python 函数！以下是一些开始的问题：

1. 你认为什么是一把强大的密码？
2. 函数应该有任何输入参数吗？
3. 函数应该返回什么？
4. 函数应该有任何副作用，比如打印消息到控制台吗？
5. 函数应该有任何错误处理，比如处理用户无效的输入吗？

Copilot 未能遵循我们“一次问一个问题”的指示。让我们提醒它。

**LPHuman** 一次问我一个问题。

![chatGpt](img/chatGpt.png)**GitHub Copilot** 当然，让我们从第一个问题开始：

1. 你认为什么是一把强大的密码？

现在，我们可以开始回答这些问题。Copilot 将使用这些答案为我们编写代码。

**LPHuman** 密码必须至少包含一个大小写字母，至少一个数字和一个标点符号。

![chatGpt](img/chatGpt.png)**GitHub Copilot** 很好，感谢你告诉我！接下来是下一个问题：

2. 函数应该有任何输入参数吗？

**LPHuman** 我不知道。它们会被用来做什么？

![chatGpt](img/chatGpt.png)**GitHub Copilot** 输入参数用于将数据传递给函数，以便函数可以对数据进行某些操作。例如，如果我们想编写一个将两个数字相加的函数，我们需要将这两个数字作为输入参数传递给函数。

在我们的密码函数的情况下，我们可能想要传递一些参数来定义强大密码的要求。例如，我们可能想要传递密码的最小长度、所需的大小写字母数量、所需的数字数量和所需的标点符号数量。

或者，我们可以将这些要求硬编码到函数本身中，但使用输入参数会使函数更加灵活和可重用。

**LPHuman** 请将它们硬编码。

![chatGpt](img/chatGpt.png)**GitHub Copilot** 当然，我们可以将密码要求硬编码到函数本身中。这是下一个问题：

3. 函数应该返回什么？

**LPHuman** 该函数应返回用户的强密码。

![chatGpt](img/chatGpt.png)**GitHub Copilot** 很好，感谢你告诉我！这是下一个问题：

4. 函数是否应该有任何副作用，例如向控制台打印消息？

我们在这里停止，因为我们认为阅读完整的交互并不有趣。我们继续与 Copilot 交谈并回答了问题 4 和 5，Copilot 拥有了编写我们想要的代码所需的信息。在回答了所有五个问题后，我们最终收到的代码如下所示。

#### 列表 12.1 通过翻转交互模式设计的函数

```py
import string

def get_strong_password():
    '''
    Keep asking the user to enter a password until they provide 
    a strong password.
    A strong password has at least one uppercase character, one 
    number, and one punctuation character.
    '''
    # Print the password requirements
    print("Your password must contain at least:")
    print("- One uppercase letter")
    print("- One number")
    print("- One punctuation character")

    # Keep asking for a password until a strong one is provided
    while True:                                          #1
        password = input("Enter a password: ")

        # Check if the password meets the requirements
        if not any(c.isupper() for c in password):        #2
            print("Password does not contain an uppercase letter.")
        elif not any(c.isdigit() for c in password):        #3
            print("Password does not contain a number.")
        elif not any(c in string.punctuation      #4
                     for c in password):          #4
            print("Password does not contain a punctuation character.") #4
        else: #4
            print("Password is strong!") #4
            return password                     #5
```

- #1 没有办法使这个 `True` 条件变为 `False`！
- #2 如果密码中没有大写字母
- #3 如果密码中没有数字字符
- #4 如果密码中没有标点符号
- #5 这就是我们退出“`while True`”循环的方法。

到目前为止，在我们的 `while` 循环中，我们始终使用一个我们希望最终变为 `False` 的循环条件（否则，我们会遇到无限循环）。例如，我们可以想象有一个循环，实际上说的是，“当密码无效时继续循环。”但在这里，代码使用的是 `while True`，而 `True` 是一个永远不会变为 `False` 的条件。必须有一种其他方式退出循环——确实有！当密码有效时，我们通过 `return` 语句退出循环。也就是说，这里的逻辑是“永远循环；如果密码有效，停止循环。”`while True` 形式的 `while` 循环是本书中看到的一种常见替代结构。

这段代码很好地实现了我们的任务。它开始时通过一些 `print` 调用来告诉用户强密码的规则。如果用户的密码实际上不强，它还会告诉用户密码有什么问题。它持续循环，直到用户输入一个强密码。当我们最终从用户那里得到一个强密码时，我们返回它。

现在，我们已经知道了如何编写这个函数，因为我们已经在第三章的工作中做到了这一点。而且，到这一点，你的 Python 知识已经足够强大，你可以直接跳到编写函数签名，让 Copilot 编写代码，就像我们在整本书中做的那样。但对于你不太熟悉且可能不知道从何开始的任务，这种翻转交互模式可能会有所帮助。

### 12.1.2 个性模式

这种模式的目的是让 AI 扮演一个特定的角色或观点。我们为什么会想要这样做呢？好吧，这里有一个例子：假设你是一名入门级 Python 编程课程的学生。你的讲师正在课程中使用 Copilot，但他们禁止了一些他们认为会妨碍入门级 Python 学习的 Python 特性。你可以要求 Copilot 扮演那个讲师的角色，然后帮助你检查代码，以找到并（如果需要）移除任何不允许的 Python 特性。我们的提示结构将如下所示：

```
从现在起，扮演角色 X。提供角色 X 会创建的输出。
```

我们将用特定的角色来填充 `X`。让我们尝试使用我们的入门级编程讲师角色。我们将要求 Copilot 扮演这个角色，禁止一系列更高级的 Python 概念。（如果你从未听说过这些，请不要担心！）

**LPHuman** 从现在起，扮演一个入门级 Python 编程讲师的角色。在你教授的课程中，以下 Python 特性是不允许的：列表推导式、生成器、`lambda` 以及 `any` 函数。我会提供代码，你需要检查代码中是否有不允许的 Python 特性，并提出如何重写代码以避免使用这些特性。

![chatGpt](img/chatGpt.png)**GitHub Copilot** 当然，我可以做到！请提供你希望我审查的代码。

让我们给它提供上一节中的 `get_strong_password` 函数。它使用了 `any` 函数，根据我们的角色规则，这个函数是不允许的。以下是我们的原始代码以及 Copilot 的响应。

**LPHuman**

```py
def get_strong_password():
    '''
    Keep asking the user to enter a password until they provide 
    a strong password.
    A strong password has at least one uppercase character, one 
    number, and one punctuation character.
    '''
    # Print the password requirements
    print("Your password must contain at least:")
    print("- One uppercase letter")
    print("- One number")
    print("- One punctuation character")

    # Keep asking for a password until a strong one is provided
    while True:
        password = input("Enter a password: ")

        # Check if the password meets the requirements
        if not any(c.isupper() for c in password):
            print("Password does not contain an uppercase letter.")
        elif not any(c.isdigit() for c in password):
            print("Password does not contain a number.")
        elif not any(c in string.punctuation for c in password):
            print("Password does not contain a punctuation character.")
        else:
            print("Password is strong!")
            return password
```

![chatGpt](img/chatGpt.png)**GitHub Copilot** `get_strong_password` 函数使用了 `any` 函数，这在你的课程中是不允许的。下面是修改后的函数版本，它没有使用 `any` 函数：

