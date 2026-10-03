
该代码定义了一个值列表（在这种情况下，一个名为`time_series`的时间序列），然后使用`len()`函数循环遍历其长度以应用条件。请注意，在每次循环中，当前时间步被称为`i`，因此上一个时间步为`i - 1`。

## 库和函数

在 Python 中，*库*是一组预先编写的代码，提供功能以使应用程序的创建更容易。*模块*是单独的 Python 文件，具有可重复使用的代码和数据，可以被导入和在其他 Python 代码中使用，通常在库中找到。因此，模块是一个包含函数和其他类型代码的单个 Python 文件，可以被其他 Python 程序使用和导入。通过使用模块将相似的代码分成不同的文件，通常更容易管理和维护大型代码库。

编码是简化任务并使其更清晰的过程。当你有一个重复的任务，比如计算时间序列的移动平均值时，你可以使用一个函数，这样你就不必每次想使用它时都重新编写移动平均代码。相反，你定义一个带有原始代码的函数，然后在需要计算移动平均值时调用它。但是什么是*函数*？它是一个可重复使用的代码块，当被调用时执行特定任务。它需要被定义一次。

多个函数形成一个模块，多个模块形成一个库。一个库通常是以主题为导向的。例如，在这本书中，将使用`sklearn`库与机器学习模型。类似地，数据操作和导入是使用`numpy`和`pandas`完成的，这两个库在本章的后面部分讨论。绘图和图表是使用`matplotlib`库完成的。

在使用之前，必须先将库导入到 Python 解释器中。这样做的语法如下：

```py
# The import statement must be followed by the name of the library
import numpy

# Optionally, you can give the library a shortcut for easier references
import numpy as np
```

有时，你需要从库中导入一个函数或模块。为此，你不需要导入整个库：

```py
# Importing one function from a library
from math import sqrt
```

因此，已经确定`math`是一个 Python 库，其中包含许多数学函数，即`sqrt`函数，用于找到给定数字的平方根。让我们看看如何定义一个函数。函数使用`def`后跟函数的名称和任何可选参数来定义。考虑以下示例，创建一个对任意两个给定变量求和的函数：

```py
# Defining the function sum_operation and giving it two arguments
def sum_operation(first_variable, second_variable):

    # Outputing the sum of the two variables
    print(first_variable + second_variable)

# Calling the function with 1 and 3 as arguments
sum_operation(1, 3) # The output of this line is 4
```

###### 注意

调用一个函数意味着执行它应该做的事情。换句话说，调用一个函数就是使用它。函数的时间线是被定义然后被调用。

让我们看看如何从库中导入函数并使用它的函数：

```py
# Importing the library   
import math

# Using the natural logarithm function
math.log(10)

# Using the exponential function (e)
math.exp(3)

# Using the factorial function
math.factorial(50)
```

作为一则旁注，*阶乘*操作是一种数学操作，用于计算从 1 到某个数字（在`math.factorial()`中请求的参数）的所有正整数的乘积。

库可能不像 1 加 1 那么简单。有时，外部库在能够导入 Python 解释器之前需要先安装。安装可以通过提示符使用以下语法完成：

```py
pip install library_name
```

让我们回到第三章讨论过的 MIC。在 MIC 的以下已经看到的代码之前：

```py
# Importing the library
from minepy import MINE

# Calculating the MIC
mine = MINE(alpha = 0.6, c = 15)
mine.compute_score(sine,cosine)
MIC = mine.mic()
print('Correlation | MIC: ', round(MIC, 3))
```

直接导入库可能会导致错误，因为它尚未通过 pip 安装。因此，您必须首先使用提示符上的以下语法进行安装（而不是 Python 解释器）：

```py
pip install minepy
```

重要的是阅读随库提供的文档，以便正确使用它们。*文档*有助于了解函数的目的以及应该放入哪些参数。此外，它告诉您函数可以接受什么类型的参数（例如，字符串或数字）。现在让我们回到函数（注意它们是如何交织在一起的，讨论一个可能导致讨论另一个）。

函数可以有一个`return`语句，允许将结果存储在一个变量中，以便在代码的其他部分中使用。让我们举两个简单的例子，然后逐步讨论它们：

```py
# Defining a function to sum two variables and return the result
def sum_operation(first_variable, second_variable):

  # The summing operation is stored in a variable called final_sum
    final_sum = first_variable + second_variable

    # The result is returned
    return final_sum

# Create a new variable that holds the result of the function     
summed_value = sum_operation(1, 2)

# Use the new variable in a new mathematical operation and store the result 
double_summed_value = summed_value * 2
```

前面的代码定义了`sum_operation`函数，带有两个参数，然后将操作存储在一个名为`final_sum`的变量中，然后返回它，以便可以将其存储在外部。然后，定义一个名为`summed_value`的新变量，作为函数的输出。最后，另一个名为`double_summed_value`的变量被创建，其结果是`summed_value`乘以 2。这是一个关于如何在外部操作中使用函数结果作为变量的示例。现在，让我们考虑另一个示例（同时记住先前定义的`sum_operation`函数）：

```py
# Defining a function to square the result gotten from the sum_operation function
def square_summed_value(first_variable, second_variable):

  # Calling the nested sum_operation function and storing its result
    final_sum = sum_operation(first_variable, second_variable)

  # Creating a variable that stores the square of final_sum
    squared_sum = final_sum ** 2

    # The result is returned     
    return squared_sum

# Create a new variable that holds the result of the function  
squared_summed_value = square_summed_value(1, 2)
```

最新的代码片段定义了一个名为`square_summed_value`的函数，它接受两个参数。此外，它在内部使用了一个嵌套函数，这种情况下是`sum_operation`。嵌套函数的结果再次存储在一个名为`final_sum`的变量中，该变量用作查找`squared_sum variable`的输入。该变量被找到为`final_sum`的平方。这是一个关于如何在函数内部创建其他函数的示例（换句话说，嵌套函数）。

让我们以 Python 和机器学习中常见的库结束本节（除了`numpy`和`pandas`）：

```py
matplotlib # For plotting and visualizing data
sklearn    # For machine learning models
scipy      # For scientific computing and optimization
keras      # For neural networks
math       # For using mathematical tools such as square roots
random     # For generating random variables
requests   # For making HTTP requests used in web scraping
```

## 异常处理和错误

经常会因为代码在执行过程中出现问题而遇到错误。换句话说，当代码执行时，解释器发现阻止其继续进行的障碍时，就会发生错误。最基本的错误是*SyntaxError*，当有拼写错误或缺少元素使代码无法理解时会发生：

```py
# Will not output a SyntaxError if executed
my_range = range(1, 10)

# Will output a SyntaxError is executed
my_range = range(1, 10
```

从前面的代码中可以看到，第二行代码末尾缺少一个括号，这个错误解释器无法理解。这种类型的错误可能是最常见的。另一个常见的错误是*NameError*，当在执行包含它的代码之前未定义变量时会发生。考虑以下示例：

```py
x + y
```

前面的代码会给出一个*NameError*，因为解释器不知道`x`和`y`的值，因为它们没有被定义。

*ModuleNotFoundError*发生在解释器无法找到您尝试导入的库或模块时。这通常发生在它安装在错误的目录或未正确安装时。解决此问题的常见方法包括：

+   验证模块的名称是否已正确书写。

+   验证模块是否已正确通过 pip 安装。

+   验证模块是否安装在正确的位置。

另一种常见错误类型是*TypeError*，当你在不兼容的元素上应用某种操作时会发生，比如将整数与字符串相加。以下操作会引发*TypeError*：

```py
# Defining variable x
x = 1

# Defining variable y
y = 'Hello

# Summing the two variables which will raise a TypeError
x + y
```

在时间序列分析中，你可能会遇到这四种错误：

+   *IndexError*: 当涉及到当前数组或数据框范围之外的索引时，会引发此错误。想象一下有一个包含 300 个值（行）的数组。如果你想循环遍历它们，并在每次循环时将数字 1 输入到下一个单元格（时间步长+1），你很可能会遇到*IndexError*，因为在最后一个循环中，没有下一个单元格，解释器会引发此错误。

+   *ValueError*: 当你尝试使用无效参数调用函数时会引发此错误。例如，尝试在调用函数时将整数元素传递为字符串。

+   *KeyError*: 当尝试访问数据框中不存在的元素时会发生此错误。例如，如果数据框中有三列，而你引用了一个不存在的列（可能是由于语法问题），你很可能会遇到*KeyError*。

+   *ZeroDivisionError*: 这个错误很直观，当尝试将一个数字除以零时会发生。

你可能会遇到其他类型的错误。重要的是要理解它们指的是什么，这样你才能够修复它们。

另一方面，*Exceptions*可能对代码并不致命，因为它们只是显示一个警告，但不一定终止代码。因此，异常发生在执行过程中（与错误不同，错误是因为解释器无法执行）。

###### 注意

理解错误将有助于你修复它并使代码重新运行。

为了忽略某些异常，使用`try`和`except`关键字。当你确信处理异常不会改变代码的输出时，这是很有用的。让我们举一个例子，创建一个函数，将时间序列的第一列除以第二列的下一个值。第一步是将时间序列定义为数据框或数组（或任何其他数据集合结构）：

```py
# Importing the required library to create an array
import numpy as np

# Creating a two-column list with 8 rows
my_time_series = [(1, 3), 
                  (1, 4), 
                  (1, 4), 
                  (1, 6), 
                  (1, 4), 
                  (0, 2), 
                  (1, 1), 
                  (0, 6)]

# Transforming the list into an array
my_time_series = np.array(my_time_series)
```

现在，让我们编写一个除以第二列中的下一个值的除法函数，该函数将接受第一列中的任何值：

```py
# Defining the function
def division(first_column, second_column):

 # Looping through the length of the created array
    for i in range(len(my_time_series)):

        # Division operation and storing it in the variable x
        x = my_time_series[i, first_column] / my_time_series[i + 1, second_column]

        # Outputting the result
        print(x)

# Calling the function
division(0, 1)
```

运行前两个代码块将会产生*IndexError*，因为在最后一个循环中，函数找不到第二列的下一个值，因为它根本不存在：

```py
IndexError: index 8 is out of bounds for axis 0 with size 8
```

通过`try`和`except`来修复这个问题将忽略导致问题的最后一个计算，并返回预期的结果：

```py
# Defining the function
def division(first_column, second_column):

    # Looping through the length of the created array   
    for i in range(len(my_time_series)):

        # First part of the exception handling
        try:

            # Division operation and storing it in the variable x
            x = my_time_series[i, first_column] / my_time_series[i + 1, second_column]

            # Outputting the result         
            print(x)

        # Exception handling of a specific error     
        except IndexError:

            # Ignoring (passing) the error
            pass

# Calling the function
division(0, 1)
```

输出如下：

```py
0.25
0.25
0.16666666666666666
0.25
0.5
0.0
0.16666666666666666
```

本节讨论了错误和异常以及如何处理这些问题。在接下来的章节中，由于其简单性和有效性，这些问题将使用`try`和`except`块来处理。

## Numpy 和 Pandas 中的数据结构

`numpy` 和 `pandas` 这两个词可能对你来说很熟悉，因为我在前几章的大部分代码片段中都用到了它们。此外，你现在了解了什么是库，因此你知道这两个库是用来在 Python 中操作、处理和导入数据的首选库。本节展示了这两者之间的区别以及它们的关键功能，这绝对是你数据分析的重要补充。但首先，让我们定义这两个库：

*numpy*

NumPy（Numerical Python 的缩写）是一个允许处理多维数组和矩阵的 Python 库。NumPy 提供了一个强大的接口，用于对数组和矩阵执行各种操作。

*pandas*
