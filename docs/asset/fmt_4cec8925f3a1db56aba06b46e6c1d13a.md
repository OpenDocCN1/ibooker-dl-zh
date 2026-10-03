
LPHuman 使用 Python，我该如何确定 `.png` 文件是否相同？

![chatGpt](img/chatGpt.png)GitHub Copilot 你可以使用 Python 中的 `filecmp` 模块来比较两个文件并检查它们是否相同。这里有一个比较两个 `.png` 文件的代码片段示例：

```py
import filecmp

# Replace these file paths with the actual file paths of the .png files
# you want to compare
file1 = '/path/to/file1.png'
file2 = '/path/to/file2.png'

if filecmp.cmp(file1, file2):
    print('The files are the same.')
else:
    print('The files are different.')
```

这段代码将比较两个文件的内容，如果它们相同则返回 `True`，如果不同则返回 `False`。

作为一个小插曲，Copilot 的最后一句是错误的。那段代码在任何地方都没有返回 `True` 或 `False`。就像 Copilot 可以在我们的 Python 文件中为我们生成错误的代码一样，Copilot Chat 也可以产生错误的解释。

我们的提示并没有精确说明我们需要比较的文件数量。我们是想比较两个文件吗？还是几百个文件？我们的提示没有说。Copilot 告诉我们如果想要比较两个文件应该怎么做，这是合理的。但这并不是我们需要的。我们需要能够在几百个文件中确定哪些文件是其他文件的副本。这样，我们就可以检测到重复文件，并且不会将其复制到新目录中。让我们再试一次。

LPHuman 如果我有一百多个 `.png` 文件，并且我想找到重复的文件，你给我的那种方法仍然有效吗？

![chatGpt](img/chatGpt.png)GitHub Copilot 如果您有数百个 `.png` 文件，并且想要查找重复项，您可以使用 `os` 模块遍历目录中的所有文件，并将每个文件与目录中的每个其他文件进行比较。以下是一个示例代码片段，展示了如何做到这一点：

```py
import os
import filecmp

# 将此替换为实际的目录路径，其中包含您的.png
# 文件位置
directory = '/path/to/directory'

# 遍历目录中的所有文件
for filename1 in os.listdir(directory):
    if filename1.endswith('.png'):
        for filename2 in os.listdir(directory):
            if filename2.endswith('.png') and filename1 != filename2:
                file1 = os.path.join(directory, filename1)
                file2 = os.path.join(directory, filename2)
                if filecmp.cmp(file1, file2):
                    print(f'{filename1} and {filename2} are duplicates.')
```

此代码将遍历目录中的所有文件，并将目录中的每个 `.png` 文件与每个其他 `.png` 文件进行比较。如果两个文件相同，它将打印一条消息，指出它们是重复的。请注意，对于包含大量文件的目录，这种方法可能很慢。

外层 `for` 循环遍历目录中的每个文件。我们如何知道当前文件是否有重复项呢？嗯，我们可以遍历目录中的其他文件，并将它们与当前文件进行比较。这就是内层 `for` 循环的作用：它使用 `filecmp.cmp` 将当前文件（`file1`）与每个其他文件（`file2`）进行比较。

在我们的提示中，我们没有提到我们关心跨多个目录查找重复项，所以 Copilot 在这里关注的是单个目录。如果这种差异成为障碍，我们可以使我们的提示更加精确。

Copilot 在这里使用了两个其他模块，`os` 和 `filecmp`。我们可以询问 Copilot 这些是否是内置的 Python 模块，但我们将节省一点时间，在这里直接告诉您它们是内置的。

现在，我们希望您关注 Copilot 的最后一句：“请注意，这种方法对于包含大量文件的目录来说可能很慢。”“慢”有多慢？“多”有多少？我们不知道。

您可能会想要求 Copilot 提供一个更好的解决方案，一个对于“对于包含大量文件的目录来说可能很慢”的解决方案。但许多程序员不会这样做。在我们尝试我们的（未优化的、显然慢的）方法之前，优化我们的解决方案通常是一个错误，原因有两个。首先，也许我们的“慢”程序最终足够快！我们不妨试试。其次，更优化的程序通常是更复杂的程序，它们可能更难正确实现。这并不总是这种情况，但它可能是。而且，如果我们的未优化程序完成了工作，我们甚至不必担心更优化的版本。

现在，如果我们的程序真的太慢，或者你发现自己反复使用这个程序，那么继续与 Copilot 合作以获得更快的解决方案可能值得额外的投资。不过，目前来说，我们做得很好。

### 9.5.2 自顶向下设计

这个任务比我们之前两个任务要复杂一些。一方面，我们需要小心不要覆盖我们新目录中已经存在的文件。另一方面，我们需要确定首先需要复制哪些文件（记住我们只想复制那些在新目录中不匹配的文件）。这与我们刚刚完成的 `.pdf` 合并任务形成对比，在那个任务中我们没有这些额外的担忧。

为了达到这个目的，我们将在这里使用自顶向下设计和问题分解。不用担心，这不会是一个像我们在第七章中做的完全自顶向下的设计示例。我们这里的任务比第七章中的拼写建议任务要小得多。我们只需进行一点自顶向下的设计，这将帮助 Copilot 为我们提供我们想要的结果。

我们的顶级函数将负责解决我们的整体任务：将 `pictures1` 和 `pictures2` 目录中的所有独特图片放入目标目录。在第三章，我们学习了我们应该尽可能使函数通用，以便使它们更有用或更易于推广到其他任务。在这里，我们一直在考虑合并两个图片目录。但为什么不是 3、5 或 50 个目录呢？谁在乎我们有多少个目录；我们应该能够合并我们想要的任意多个目录。

因此，我们不会设计我们的顶级函数以接受两个字符串（目录名）作为参数，我们将让函数接受一个字符串列表。这样，我们就可以用它来处理我们想要的任意多个图片目录。而且，我们仍然可以轻松地用它来处理两个图片目录——我们只需传递一个包含两个目录名称的列表。

我们将命名我们的顶级函数为 `make_copies`。我们需要两个参数：我们刚才讨论的目录名称列表，以及我们想要所有文件都放入的目标目录的名称。

这个函数将要做什么？它将遍历目录列表中的每个目录，然后，对于每个目录，它将遍历每个文件。对于每个文件，我们需要确定是否复制它，如果需要复制，则执行实际的复制操作。

确定是否复制文件，然后可能复制它，这是一个可以从 `make_copies` 中分离出来的子任务。我们将为这个子任务命名函数为 `make_copy`。我们的 `make_copy` 函数将接受两个参数：文件的名称和目标目录。如果文件与目标目录中的任何文件都不相同，那么该函数将把文件复制到目标目录中。

假设我们想要将名为 `9595.png` 的文件从一个图片目录复制到我们的目标目录，但该文件已经在目标目录中存在。我们不希望覆盖已经存在的文件，因此我们需要想出一个新的文件名。我们可能会尝试在文件名中的 `.png` 部分之前添加一个 `_`（下划线）字符。这将给我们 `9595_.png`。这个文件可能不在目标目录中，但如果它确实存在，我们可以尝试 `9595__.png`，`9595___.png`，等等，直到我们找到一个不存在的文件名。

生成一个唯一的文件名是我们可以从 `make_copy` 函数中分离出来的任务。我们将称它为 `get_good_filename`。它将接受一个文件名作为参数，并返回一个不存在的文件名版本。

有了这些，我们的自顶向下设计就完成了。图 9.4 描绘了我们的工作作为一个树（至少是树的树干），显示了哪个函数被哪个其他函数调用。

![图](img/9-4.png)

##### 图 9.4 图像合并的自顶向下设计。最顶层（最左侧）的函数是 `make_copies`，它的子函数是 `make_copy`，而 `make_copy` 的子函数是 `get_good_filename`。

### 9.5.3 编写工具

这次我们没有需要安装的模块。我们知道从我们的 Copilot 对话中，我们将使用内置的 `shutil` 模块来复制文件。我们还将使用内置的 `filecmp` 模块来比较文件，以及内置的 `os` 模块来获取目录中的文件列表。因此，我们将在 Python 程序的顶部导入这三个模块。

正如第七章所述，我们将从函数树的底部开始解决问题，逐步向上工作。我们这样做是为了当 Copilot 为父函数编写代码时，可以调用我们已编写的函数。对于每个函数，我们提供 `def` 行和文档字符串，然后 Copilot 编写代码。我们还提供了一些注释来解释代码的工作原理。

再次查看图 9.4，我们看到我们需要实现的第一项功能是 `get_good_filename`。现在让我们在下面的列表中完成这个功能。

##### 列表 9.6 为我们的图片合并任务编写的 `get_good_filename` 函数

```py
import shutil
import filecmp
import os

def get_good_filename(fname):
    '''
    fname is the name of a png file.

    While the file fname exists, add an _ character
    right before the .png part of the filename;
    e.g. 9595.png becomes 9595_.png.

    Return the resulting filename.
    '''
    while os.path.exists(fname):           #1
        fname = fname.replace('.png', '_.png')      #2
    return fname          #3
```

#1 当文件名存在时...

#2 ...在 `.png` 之前插入一个 `_`，通过将 `.png` 替换为 `_.png`。

#3 返回我们现在知道不存在的文件名

下一个我们需要编写的函数是 `make_copy`。这个函数将文件复制到目标目录，但前提是文件与我们之前复制的文件不相同。我们希望 Copilot 在其代码中完成以下几件事情：

+   使用 `os.listdir` 获取目标目录中的文件列表。

+   使用 `filecmp.cmp` 确定两个文件是否相同。

+   使用 `shutil.copy` 复制文件，如果没有找到相同的文件。

+   调用我们刚刚编写的 `get_good_filename` 函数。

下面的列表显示了我们的提示和 Copilot 提供的代码。请注意，代码正在做我们希望它做的事情。

##### 列表 9.7 `make_copy` 函数，用于我们的图片合并任务

```py
def make_copy(fname, target_dir):
    '''
    fname is a filename like pictures1/1262.png.
    target_dir is the name of a directory.

    Compare the file fname to all files in target_dir.
    If fname is not identical to any file in
    target_dir, copy it to target_dir
    '''
    for target_fname in os.listdir(target_dir):      #1
        if filecmp.cmp(fname, os.path.join(  \    #2
                   target_dir, target_fname)):   
            return                                    #3
    shutil.copy(fname, get_good_filename(      #4
            os.path.join(target_dir,  \        #4
                 os.path.basename(fname))))   #4
```

#1 遍历目标目录中的文件

#2 如果文件与目标目录中的某个文件相同，……

#3 ……从函数中返回而没有复制文件。

#4 否则，复制文件并使用一个尚未存在的良好文件名。

只剩下最后一个函数，那就是我们的顶级 `make_copies` 函数。对于我们的每个图片目录中的每个文件，我们期望代码调用 `make_copy` 来复制文件（如果需要的话），如下面的列表所示。

##### 列表 9.8 `make_copies` 函数，用于我们的图片合并任务

```py
def make_copies(dirs, target_dir):
    '''
    dirs is a list of directory names.
    target_dir is the name of a directory.

    Check each file in the directories and compare it to all files 
    in target_dir. If a file is not identical to any file in 
    target_dir, copy it to target_dir
    '''
    for dir in dirs:                         #1
        for fname in os.listdir(dir):              #2
            make_copy(os.path.join(dir, fname),  \   #3
                      target_dir)                    #3

make_copies(['pictures1', 'pictures2'],  #4
             'pictures_combined')        #3
```

#1 遍历我们的图片目录

#2 遍历当前图片目录中的文件

#3 如果需要，将当前文件复制到目标目录

#4 在我们的两个图片目录和给定的目标目录上运行我们的程序

Copilot 在 `make_copies` 函数下面的最后一行代码假设我们的目标目录将被命名为 `pictures_combined`。现在创建该目录，以便它位于你的 `pictures1` 和 `pictures2` 目录旁边。

正如我们在本章前面处理 `.pdf` 文件时讨论的那样，你首先在你不关心的样本目录上测试程序是很重要的。你的样本目录中应该只有几个文件，这样你可以手动确定程序是否按预期工作。你还应该包括重要的边缘情况，例如每个目录中都有相同的文件名。

一旦你有了样本目录，你应该创建一个“无害”的程序版本，该程序简单地输出消息而不是实际复制文件。对于我们的程序，你将更改 `make_copy` 中的行，使用 `print` 而不是 `shutil.copy`。

在仔细检查结果后，如果输出看起来不错，那么你应该只在你的真实目录上运行真正的程序。记住，我们的程序是在复制（而不是移动）文件，所以即使在我们的真实目录中，如果出现问题，有很大可能性问题出在我们的新目录中，而不是我们真正关心的原始目录中。

我们假设你现在已经准备好在 `pictures1` 和 `pictures2` 目录上运行程序。一旦运行，你可以检查你的 `pictures_combined` 目录以查看结果。你应该看到该目录有 200 个文件，这正好是我们两个图片目录中独特的图片数量。我们是否正确处理了在两个图片目录中都存在相同文件名但图片不同的情况？是的，你可以看到我们有名为 `9595.png` 和 `9595_.png` 的文件，因此我们没有覆盖彼此。

哦，你的程序在你的电脑上运行了多久？最多几秒钟，对吧？结果是，Copilot 的警告“这种方法对于包含许多文件的大型目录可能会很慢”对我们来说并不重要。

现在，我们都知道人们通常在手机上有成千上万的照片，而不是几百张。如果您在两个真实的手机图片库上运行这个程序，您还需要确定它是否在可接受的时间内完成。您可以运行程序，让它运行一分钟左右，或者您愿意等待多久。为了好玩，我们还测试了我们的程序在总共 10,000 个小图像文件上的运行情况（比我们在本章中使用的 `pictures1` 和 `pictures2` 目录中的 210 张图片更现实的场景），我们发现它只用了 1 分钟就完成了。在某个时候，我们的程序可能会变得太慢而无法实用，那时您就需要使用 Copilot Chat 进行进一步的研究，以得到一个更高效的程序。

在本章中，我们成功地自动化了三个繁琐的任务：清理电子邮件、为数百个 `.pdf` 文件添加封面，以及将多个图片库合并为一个。在每种情况下，方法都是相同的：使用 Copilot Chat 确定要使用哪个模块，然后遵循我们在整本书中磨练的方法，让 Copilot 编写所需的代码。

记住，无论何时您发现自己重复执行相同的任务，尝试使用 Copilot 和 Python 来自动化它都是值得的。除了本章中展示的之外，还有很多有用的 Python 模块可以做到这一点。例如，有用于操作图像、处理 Microsoft Excel 或 Microsoft Word 文件、发送电子邮件、从网站抓取数据等模块。如果是一项繁琐的任务，那么很可能有人已经编写了一个 Python 模块来帮助完成这项任务，并且 Copilot 可以帮助您有效地使用该模块。

## 9.6 练习
