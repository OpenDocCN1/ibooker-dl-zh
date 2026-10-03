

# 第五章：TinyML 的“Hello World”：构建一个应用程序

模型只是机器学习应用程序的一部分。单独来看，它只是一块信息；它几乎什么都做不了。要使用我们的模型，我们需要将其包装在代码中，为其设置必要的运行环境，提供输入，并使用其输出生成行为。图 5-1 显示了模型在右侧如何适配到基本 TinyML 应用程序中。

在本章中，我们将构建一个嵌入式应用程序，使用我们的正弦模型创建一个微小的灯光秀。我们将设置一个连续循环，将一个`x`值输入模型，运行推断，并使用结果来开关 LED 灯，或者控制动画，如果我们的设备有 LCD 显示器的话。

这个应用程序已经编写好了。这是一个 C++ 11 程序，其代码旨在展示一个完整的 TinyML 应用程序的最小可能实现，避免任何复杂的逻辑。这种简单性使它成为学习如何使用 TensorFlow Lite for Microcontrollers 的有用工具，因为您可以清楚地看到需要哪些代码，以及很少的其他内容。它也是一个有用的模板。阅读完本章后，您将了解 TensorFlow Lite for Microcontrollers 程序的一般结构，并可以在自己的项目中重用相同的结构。

本章将逐步介绍应用程序代码并解释其工作原理。下一章将提供详细的构建和部署说明，适用于多种设备。如果您不熟悉 C++，不要惊慌。代码相对简单，我们会详细解释一切。到最后，您应该对运行模型所需的所有代码感到满意，甚至可能在学习过程中学到一些 C++知识。

![显示基本 TinyML 应用程序架构的图表](img/timl_0501.png)

###### 图 5-1。基本 TinyML 应用程序架构

###### 提示

请记住，由于 TensorFlow 是一个积极开发的开源项目，这里打印的代码与在线代码之间可能存在一些细微差异。不用担心，即使有一些代码行发生变化，基本原则仍然保持不变。

## 测试步骤

在处理应用程序代码之前，编写一些测试通常是一个好主意。测试是演示特定逻辑片段的短代码片段。由于它们由可工作的代码组成，我们可以运行它们来证明代码是否按预期运行。编写完测试后，通常会自动运行测试，以持续验证项目是否仍然按照我们的期望运行，尽管我们可能对其代码进行了任何更改。它们也非常有用，作为如何执行操作的工作示例。

`hello_world`示例有一个测试，在[*hello_world_test.cc*](https://oreil.ly/QW0SS)中定义，加载我们的模型并使用它运行推断，检查其预测是否符合我们的期望。它包含了执行此操作所需的确切代码，没有其他内容，因此这将是学习 TensorFlow Lite for Microcontrollers 的绝佳起点。在本节中，我们将逐步介绍测试，并解释其中的每个部分的作用。阅读完代码后，我们可以运行测试以证明其正确性。

现在让我们逐节来看一下。如果您在电脑上，打开*hello_world_test.cc*并跟着走可能会有帮助。

### 包含依赖项

第一部分，在许可证标题下方（指定任何人都可以在[Apache 2.0](https://oreil.ly/Xa5_x)开源许可下使用或共享此代码），如下所示：

```cpp
#include "tensorflow/lite/micro/examples/hello_world/sine_model_data.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
```

`#include`指令是 C++代码指定其依赖的其他代码的一种方式。当使用`#include`引用代码文件时，它定义的任何逻辑或变量将可供我们使用。在本节中，我们使用`#include`导入以下项目：

- `tensorflow/lite/micro/examples/hello_world/sine_model_data.h`：我们训练的正弦模型，使用`xxd`转换并转换为 C++
- `tensorflow/lite/micro/kernels/all_ops_resolver.h`：一个允许解释器加载我们模型使用的操作的类
- `tensorflow/lite/micro/micro_error_reporter.h`：一个可以记录错误并输出以帮助调试的类
- `tensorflow/lite/micro/micro_interpreter.h`：将运行我们模型的 TensorFlow Lite for Microcontrollers 解释器
- `tensorflow/lite/micro/testing/micro_test.h`：一个用于编写测试的轻量级框架，允许我们将此文件作为测试运行
- `tensorflow/lite/schema/schema_generated.h`：定义了 TensorFlow Lite FlatBuffer 数据结构的模式，用于理解`sine_model_data.h`中的模型数据
- `tensorflow/lite/version.h`：模式的当前版本号，以便我们可以检查模型是否使用兼容版本定义

当我们深入代码时，我们将更多地讨论其中一些依赖关系。

###### 注意

按照惯例，设计用于与`#include`指令一起使用的 C++代码通常编写为两个文件：一个*.cc*文件，称为*源文件*，以及一个*.h*文件，称为*头文件*。头文件定义了允许代码连接到程序其他部分的接口。它们包含变量和类声明等内容，但几乎没有逻辑。源文件实现了执行计算和使事情发生的实际逻辑。当我们`#include`一个依赖项时，我们指定其头文件。例如，我们正在讨论的测试包括[*micro_interpreter.h*](https://oreil.ly/60uYt)。如果我们查看该文件，我们可以看到它定义了一个类，但没有包含太多逻辑。相反，它的逻辑包含在[*micro_interpreter.cc*](https://oreil.ly/twN7J)中。

### 设置测试

代码的下一部分用于 TensorFlow Lite for Microcontrollers 测试框架。看起来像这样：

```cpp
TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(LoadModelAndPerformInference) {
```

在 C++中，您可以定义特殊命名的代码块，可以通过在其他地方包含它们的名称来重用。这些代码块称为*宏*。这里的两个语句，`TF_LITE_MICRO_TESTS_BEGIN`和`TF_LITE_MICRO_TEST`，是宏的名称。它们在文件[*micro_test.h*](https://oreil.ly/NoGm4)中定义。

这些宏将我们代码的其余部分包装在必要的装置中，以便通过 TensorFlow Lite for Microcontrollers 测试框架执行。我们不需要担心这是如何工作的；我们只需要知道我们可以使用这些宏作为设置测试的快捷方式。

第二个名为`TF_LITE_MICRO_TEST`的宏接受一个参数。在这种情况下，传入的参数是`LoadModelAndPerformInference`。这个参数是测试名称，当运行测试时，它将与测试结果一起输出，以便我们可以看到测试是通过还是失败。

### 准备记录数据

文件中剩余的代码是我们测试的实际逻辑。让我们看一下第一部分：

```cpp
// Set up logging
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
```

在第一行中，我们定义了一个`MicroErrorReporter`实例。`MicroErrorReporter`类在[*micro_error_reporter.h*](https://oreil.ly/AkZrm)中定义。它提供了在推理期间记录调试信息的机制。我们将调用它来打印调试信息，而 TensorFlow Lite for Microcontrollers 解释器将使用它来打印遇到的任何错误。

###### 注意

您可能已经注意到每个类型名称之前的`tflite::`前缀，例如`tflite::MicroErrorReporter`。这是一个*命名空间*，只是帮助组织 C++代码的一种方式。TensorFlow Lite 在命名空间`tflite`下定义了所有有用的内容，这意味着如果另一个库恰好实现了具有相同名称的类，它们不会与 TensorFlow Lite 提供的类发生冲突。

第一个声明看起来很简单，但是第二行看起来有点奇怪，带有`*`和`&`字符？为什么我们要声明一个`ErrorReporter`当我们已经有一个`MicroErrorReporter`了？

```cpp
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
```

解释这里发生的事情，我们需要了解一些背景信息。

`MicroErrorReporter`是`ErrorReporter`类的一个子类，为 TensorFlow Lite 中这种调试日志机制应该如何工作提供了一个模板。`MicroErrorReporter`覆盖了`ErrorReporter`的一个方法，用专门为在微控制器上使用而编写的逻辑替换它。

在前面的代码行中，我们创建了一个名为`error_reporter`的变量，它的类型是`ErrorReporter`。它也是一个指针，其声明中使用了`*`表示。

指针是一种特殊类型的变量，它不是保存一个值，而是保存一个引用，指向内存中的一个值。在 C++中，一个类的指针（比如`ErrorReporter`）可以指向它的一个子类（比如`MicroErrorReporter`）的值。

正如我们之前提到的，`MicroErrorReporter`覆盖了`ErrorReporter`的一个方法。不详细讨论，覆盖这个方法的过程会导致一些其他方法被隐藏。

为了仍然可以访问`ErrorReporter`的未覆盖方法，我们需要将我们的`MicroErrorReporter`实例视为实际上是`ErrorReporter`。我们通过创建一个`ErrorReporter`指针并将其指向`micro_error_reporter`变量来实现这一点。赋值语句中`&`在`micro_error_reporter`前面的意思是我们正在分配它的指针，而不是它的值。

哎呀！听起来很复杂。如果你觉得难以理解，不要惊慌；C++可能有点难以掌握。对于我们的目的，我们只需要知道我们应该使用`error_reporter`来打印调试信息，并且它是一个指针。

### 映射我们的模型

我们立即建立打印调试信息的机制的原因是为了记录代码中发生的任何问题。我们在下一段代码中依赖于这一点：

```cpp
// Map the model into a usable data structure. This doesn't involve any
// copying or parsing, it's a very lightweight operation.
const tflite::Model* model = ::tflite::GetModel(g_sine_model_data);
if (model->version() != TFLITE_SCHEMA_VERSION) {
error_reporter->Report(
    "Model provided is schema version %d not equal "
    "to supported version %d.\n",
    model->version(), TFLITE_SCHEMA_VERSION);
    return 1;
}
```

在第一行中，我们将我们的模型数据数组（在文件[*sine_model_data.h*](https://oreil.ly/m68Wj)中定义）传递给一个名为`GetModel()`的方法。这个方法返回一个`Model`指针，被赋值给一个名为`model`的变量。正如你可能预料的那样，这个变量代表我们的模型。

类型`Model`是一个*struct*，在 C++中与类非常相似。它在[*schema_generated.h*](https://oreil.ly/SGNtU)中定义，它保存我们模型的数据并允许我们查询有关它的信息。

一旦`model`准备好，我们调用一个检索模型版本号的方法：

```cpp
if (model->version() != TFLITE_SCHEMA_VERSION) {
```

然后我们将模型的版本号与`TFLITE_SCHEMA_VERSION`进行比较，这表示我们当前使用的 TensorFlow Lite 库的版本。如果数字匹配，我们的模型是使用兼容版本的 TensorFlow Lite Converter 转换的。检查模型版本是一个好习惯，因为版本不匹配可能导致难以调试的奇怪行为。

###### 注意

在前一行代码中，`version()`是属于`model`的一个方法。注意箭头（`->`）从`model`指向`version()`。这是 C++的*箭头运算符*，当我们想要访问一个对象的成员时使用。如果我们有对象本身（而不仅仅是一个指针），我们将使用点（`.`）来访问它的成员。

如果版本号不匹配，我们仍然会继续，但我们会使用我们的`error_reporter`记录一个警告：

```cpp
error_reporter->Report(
    "Model provided is schema version %d not equal "
    "to supported version %d.\n",
    model->version(), TFLITE_SCHEMA_VERSION);
```

我们调用`error_reporter`的`Report()`方法来记录这个警告。由于`error_reporter`也是一个指针，我们使用`->`运算符来访问`Report()`。

`Report()`方法的设计类似于一个常用的 C++方法`printf()`，用于记录文本。作为它的第一个参数，我们传递一个我们想要记录的字符串。这个字符串包含两个`%d`格式说明符，它们充当变量在消息记录时插入的占位符。我们传递的下两个参数是模型版本和 TensorFlow Lite 模式版本。这些将按顺序插入到字符串中，以替换`%d`字符。

###### 注意

`Report()`方法支持不同的格式说明符，用作不同类型变量的占位符。`%d`应该用作整数的占位符，`%f`应该用作浮点数的占位符，`%s`应该用作字符串的占位符。

### 创建一个 AllOpsResolver

到目前为止一切顺利！我们的代码可以记录错误，我们已经将模型加载到一个方便的结构中，并检查它是否是兼容的版本。鉴于我们一路上在回顾一些 C++概念，我们进展有点慢，但事情开始变得清晰起来了。接下来，我们创建一个`AllOpsResolver`的实例：

```cpp
// This pulls in all the operation implementations we need
tflite::ops::micro::AllOpsResolver resolver;
```

这个类在[*all_ops_resolver.h*](https://oreil.ly/O0qgy)中定义，它允许 TensorFlow Lite for Microcontrollers 解释器访问*操作*。

在第三章中，您了解到机器学习模型由各种数学运算组成，这些运算按顺序运行，将输入转换为输出。`AllOpsResolver`类知道 TensorFlow Lite for Microcontrollers 可用的所有操作，并能够将它们提供给解释器。

### 定义张量区域

我们几乎已经准备好创建一个解释器所需的所有要素。我们需要做的最后一件事是分配一个工作内存区域，我们的模型在运行时将需要这个内存区域：

```cpp
// Create an area of memory to use for input, output, and intermediate arrays.
// Finding the minimum value for your model may require some trial and error.
const int tensor_arena_size = 2 × 1024;
uint8_t tensor_arena[tensor_arena_size];
```

正如注释所说，这个内存区域将用于存储模型的输入、输出和中间张量。我们称之为我们的*张量区域*。在我们的情况下，我们分配了一个大小为 2,048 字节的数组。我们用表达式`2 × 1024`来指定这一点。

那么，我们的张量区域应该有多大呢？这是一个很好的问题。不幸的是，没有一个简单的答案。不同的模型架构具有不同大小和数量的输入、输出和中间张量，因此很难知道我们需要多少内存。数字不需要精确——我们可以保留比我们需要的更多的内存——但由于微控制器的 RAM 有限，我们应该尽可能保持它小，以便为程序的其余部分留出空间。

我们可以通过试错来完成这个过程。这就是为什么我们将数组大小表示为`n × 1024`：这样可以很容易地通过改变`n`来扩大或缩小数字（保持为 8 的倍数）。要找到正确的数组大小，从一个相对较高的数字开始，以确保它有效。本书示例中使用的最大数字是`70 × 1024`。然后，减少数字直到您的模型不再运行。最后一个有效的数字就是正确的数字！

### 创建解释器

现在我们已经声明了`tensor_arena`，我们准备设置解释器。下面是具体步骤：

