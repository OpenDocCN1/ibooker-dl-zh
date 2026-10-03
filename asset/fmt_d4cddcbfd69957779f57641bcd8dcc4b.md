
# 4.3 使用索引和切片

许多 Python 类型，包括字符串和列表，允许你使用索引来处理特定的值。当你想要处理字符串或列表的一部分而不是整个字符串或列表时，你需要使用索引。索引只是一个标识元素的数字。索引从 0 开始，对于第一个元素，然后一直增加到但不包括值的数量。第一个值有索引 0（不是索引 1！），第二个有索引 1，第三个有索引 2，依此类推。列表中的最后一个值在索引处，该索引是列表长度减 1。列表的长度可以通过使用`len`函数来确定。例如，如果我们做`len(books)`，我们会得到一个值为`4`（因此有效的索引是从 0 到包括 3）。人们也经常使用负索引，这为每个值提供了另一种索引方式：最右侧的值有索引-1，其左侧的值有索引-2，依此类推。图 4.7 展示了正索引和负索引的示例。

![图](img/4-7.png)

##### 图 4.7 列表元素可以通过正索引或负索引访问。

让我们在当前的书籍列表上练习索引：

```py
>>> books
['The Predator', 'The Message', 'The Encounter', 'The Invasion']
>>> books[0]        #1
'The Predator'
>>> books[1]
'The Message'
>>> books[2]
'The Encounter'
>>> books[3]
'The Invasion'
>>> books[4]                        #2
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: list index out of range
>>> books[-1]                   #3
'The Invasion'
>>> books[-2]
'The Encounter'
```

#1 `books[0]`对应于第一个元素。

#2 错误是因为索引 3 是最后一本书！

#3 `books[-1]`指向列表中的最后一个元素。

此外，还有一种方法可以从字符串或列表中提取多个值，而不仅仅是单个值。这被称为*切片*。我们指定第一个值的索引，一个冒号，以及值右侧的索引，如下所示：

```py
>>> books[1:3]             #1
['The Message', 'The Encounter']
```

#1 从索引 1 开始，到索引 2 结束（不是 3！）

我们指定了`1:3`，所以你可能期望得到包括索引 3 在内的值。但冒号后面的第二个索引（即值之后的索引）不包括在内。这听起来很反直觉，但却是真的！

如果我们省略起始或结束索引，Python 会根据需要使用起始或结束索引：

```py
>>> books[:3]                                      #1
['The Predator', 'The Message', 'The Encounter']
>>> books[1:]                                      #2
['The Message', 'The Encounter', 'The Invasion']
```

#1 与使用`books[0:3]`相同

#2 与使用`books[1:4]`相同

我们还可以使用索引来更改列表中的特定值，例如：

```py
>>> books
['The Predator', 'The Message', 'The Encounter', 'The Invasion']
>>> books[0] = 'The Android'      #1
>>> books[0]
'The Android'
>>> books[1] = books[1].upper()      #2
>>> books[1]
'THE MESSAGE'
>>> books
['The Android', 'THE MESSAGE', 'The Encounter', 'The Invasion']
```

#1 将`books[0]`更改为指向字符串值“Android”

#2 将`books[1]`更改为全部大写

如果我们在字符串上尝试这样做，我们会得到一个错误：

```py
>>> title = 'The Invasion'
>>> title[0]              #1
'T'
>>> title[1]
'h'
>>> title[-1]
'n'
>>> title[0] = 't'                   #2
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'str' object does not support item assignment
```

#1 查找字符没有问题。

#2 但赋值不会！

字符串被称为*不可变*值，这意味着您不能更改其字符。您只能创建一个全新的字符串。相比之下，列表被称为*可变*值，这意味着您可以更改它。如果您收到有关类型不支持项赋值错误的错误，您很可能是尝试更改无法更改的值。

在本章中，我们向您介绍了 Python 中最常见的五个代码特性。我们将在下一章继续介绍另外五个。我们还向您展示了如何使用 Copilot 解释工具来帮助您理解代码，并提供了验证这些解释真实性的指导。表 4.2 提供了本章所涵盖特性的总结。

##### 表 4.2 本章 Python 代码特性总结

| **代码元素** | 示例 | 简要描述 |
| --- | --- | --- |
| 函数 | `def larger(num1, num2)` | 允许我们管理代码复杂性的代码特性。函数接受输入，处理这些输入，并可能返回一个输出。 |
| 变量 | `age = 25` | 一个可读性高的人名，它指向一个存储的值。可以使用`=`赋值语句进行赋值。 |
| 条件语句 | `if age < 18: print("Can't vote") else: print("Can vote")` | 条件语句允许代码做出决策。在 Python 中，我们有与条件相关的三个关键字：`if`、`elif`和`else`。 |
| 字符串 | `name = 'Dan'` | 字符串存储一系列字符（文本）。有许多强大的方法可用于修改字符串。 |
| 列表 | `list = ['Leo', 'Dan']` | 一个包含任何类型值的序列。有许多强大的方法可用于修改列表。 |

## 4.4 练习

1. 回想一下我们在列表 4.2（第 4.3.3 节）中查看的条件代码。这个程序的行为与原始程序有何不同？

```py
def larger(num1, num2):
    if num1 < num2:
        return num1
    else:
        return num2
```

2. 在以下代码中，如果年龄是 32，会打印什么？如果您将除了第一个之外的所有`if`改为`elif`，会打印什么？

```py
if age >= 30 and age <= 39:
    print("Binging Friends")
if age >= 25 and age <= 34:      
    print("Binging The Office")
if age >= 15 and age <= 24:
    print("Binging Pretty Little Liars")
if age >= 0 and age <= 14:
    print("Binging Chi's Sweet Home")
else:
    print("What's binging?")
```

3. 你会使用什么 Python 数据类型来存储传感器每分钟记录的一系列温度读数？这种数据类型会存储什么？

4. 将以下 Python 代码块排列成程序，以计算提供的列表中所有偶数的总和。

```py
for number in numbers:
total += number
if number % 2 == 0: 
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
total = 0 
print(f"The sum of all even numbers is: {total}")
```

5. （回忆第 4.3.1 节。）考虑以下 Python 函数调用`min(input("Enter a number: "), input("Enter another number: "))`。在运行此代码之前，思考这一行代码的预期功能，然后回答以下问题：

    a. 你期望运行此代码时会发生什么？描述执行过程的每个步骤。

    b. 在这个场景中，Python 是如何处理输入和`min`函数的？

    c. 如果输入的不是数字，而是可以表示数字的字符串（例如，`5`或`2`），Python 的行为会怎样改变，如果有的话？

    d. 如果一个或两个输入不是数字，结果会怎样？通过尝试不同的输入来测试您的假设。

6. 在 4.3.2 节中，我们讨论了变量在 Python 中的持久性。变量的这种属性如何影响你编写代码的方式？考虑一个示例场景，其中在程序的不同部分保持变量状态将是有益的。

7. 我们要求 Copilot 解释以下代码的工作原理：

```py
age = int(input("Enter your age: "))

if age < 13:
    print("You are a child.")
elif age < 20:
    print("You are a teenager.")
elif age < 60:
    print("You are an adult.")
else:
    print("You are a senior.")
```

Copilot 提供了以下解释：

这段 Python 代码是一个简单的程序，它会询问用户的年龄，然后根据用户所属的年龄范围打印出一条消息。

代码的第一行使用了`input()`函数来提示用户输入他们的年龄。`input()`函数从输入中读取一行（通常来自用户的键盘），将其转换为字符串，并返回该字符串。在这种情况下，提示是“Enter your age: ”。然后使用`int()`函数将该字符串转换为整数并存储在变量`age`中。

代码的其余部分是一系列条件语句，用于检查年龄的值并打印出相应的消息。`if`语句检查年龄是否小于 13 岁，如果是，则打印“你是一个孩子。”如果年龄不小于 13 岁，程序将移动到下一个条件，即`elif`（代表`else if`）语句，它检查年龄是否小于 20 岁。如果是，则打印“你是一个青少年。”如果年龄不小于 20 岁，它将移动到下一个`elif`语句，该语句检查年龄是否小于 60 岁。如果是，则打印“你是一个成年人。”

结果表明，Copilot 的解释不完整，因为它没有描述代码对*所有输入*的处理。确定解释中缺少的内容，并将你的答案添加到解释中。

8. 审查提供的 Python 代码和初始提示给 Copilot。返回的解释过于模糊，没有帮助你理解代码中使用的字符串方法。

```py
text = "Hello, World! Programming is fun."
index = text.find("World")
new_text = text[index:].replace("World", "Everyone")

print(new_text)
```

给 Copilot 的提示：

你能解释一下这段代码的作用吗？

Copilot 给出的解释：

代码涉及通过查找特定单词并将其替换为另一个单词来操作存储在文本中的字符串。它使用方法来定位单词的开始位置，然后构建一个新的字符串，其中单词已被替换。最后，它打印出修改后的字符串。

你的任务是执行以下操作：

a. 确定为什么初始提示可能导致模糊或不实用的解释。

b. 改进提示以确保 Copilot 提供更详细和具体的解释。考虑你具体不理解字符串方法以及代码如何使用它们。

c. 将修订后的提示写给 Copilot，并讨论你的更改如何导致更好的解释。

## 摘要

+ 我们需要能够阅读代码以确定其正确性，有效地测试它，并在需要时编写自己的代码。

+ Copilot 可以逐行解释代码，以解释代码正在做什么，你可以使用自己的提示来影响你获得的解释类型。

+ Python 有内置函数，如`max`、`input`和`print`，我们可以像调用自己的函数一样调用它们。

+ 变量是一个指向值的名称。

+ 一个赋值语句使一个变量指向一个特定的值。

+ 一个`if`语句用于让我们的程序做出决策，并沿着多条路径中的其中一条继续执行。

+ 字符串用于存储和操作文本。

+ 方法是与特定类型相关联的函数。

+ 列表用于存储和操作一系列通用的值（例如，一系列数字或一系列字符串）。

+ 字符串或列表中的每个值都有一个索引；索引从 0 开始，而不是 1。

+ 字符串是不可变的（不可更改的）；列表是可变的（可更改的）*。
