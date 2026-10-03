
您可以通过打开 Colab 的左侧面板来查看检查点文件，该面板具有一个文件浏览器。要这样做，请点击图 8-15 中显示的按钮。

![打开 Colab 侧边栏的按钮截图](img/timl_0815.png)

###### 图 8-15\. 打开 Colab 侧边栏的按钮

在此面板中，点击“文件”选项卡以查看运行时的文件系统。如果您打开`speech_commands_train/`目录，您将看到检查点文件，如图 8-16 所示。每个文件名中的数字表示保存检查点的步骤。

![Colab 的文件浏览器显示检查点文件列表的截图](img/timl_0816.png)

###### 图 8-16\. Colab 的文件浏览器显示检查点文件列表

一个 TensorFlow 模型由两个主要部分组成：

+   训练产生的权重和偏差
+   将模型的输入与这些权重和偏差结合起来产生模型的输出的操作图

此时，我们的模型操作在 Python 脚本中定义，并且其训练的权重和偏差在最新的检查点文件中。我们需要将这两者合并为一个具有特定格式的单个模型文件，以便我们可以用来运行推断。创建此模型文件的过程称为*冻结*——我们正在创建一个具有*冻结*权重的图的静态表示。

为了冻结我们的模型，我们运行一个脚本。您将在下一个单元格中找到它，在“冻结图”部分。脚本的调用如下：

```cpp
!python tensorflow/tensorflow/examples/speech_commands/freeze.py \
  --model_architecture=tiny_conv --window_stride=20 --preprocess=micro \
  --wanted_words=${WANTED_WORDS} --quantize=1 \
  --output_file=/content/tiny_conv.pb \
  --start_checkpoint=/content/speech_commands_train/tiny_conv. \
  ckpt-${TOTAL_STEPS}
```

为了指向正确的操作图以冻结的脚本，我们传递了一些与训练中使用的相同参数。我们还传递了最终检查点文件的路径，该文件的文件名以训练步骤的总数结尾。

运行此单元格以冻结图。冻结的图将输出到名为`tiny_conv.pb`的文件中。

这个文件是完全训练过的 TensorFlow 模型。它可以被 TensorFlow 加载并用于运行推断。这很棒，但它仍然是常规 TensorFlow 使用的格式，而不是 TensorFlow Lite。我们的下一步是将模型转换为 TensorFlow Lite 格式。

### 转换为 TensorFlow Lite

转换是另一个简单的步骤：我们只需要运行一个命令。现在我们有一个冻结的图文件可以使用，我们将使用`toco`，TensorFlow Lite 转换器的命令行界面。

在“转换模型”部分，运行第一个单元格：

```cpp
!toco
  --graph_def_file=/content/tiny_conv.pb --output_file= \
  /content/tiny_conv.tflite \
  --input_shapes=1,49,40,1 --input_arrays=Reshape_2
  --output_arrays='labels_softmax' \
  --inference_type=QUANTIZED_UINT8 --mean_values=0 --std_dev_values=9.8077
```

在参数中，我们指定要转换的模型，TensorFlow Lite 模型文件的输出位置，以及一些取决于模型架构的其他值。因为模型在训练期间被量化，我们还提供了一些参数（`inference_type`，`mean_values`和`std_dev_values`），指导转换器如何将其低精度值映射到实数。

您可能想知道为什么`input_shape`参数在宽度、高度和通道参数之前有一个前导`1`。这是批处理大小；为了在训练期间提高效率，我们一次发送很多输入，但当我们在实时应用中运行时，我们每次只处理一个样本，这就是为什么批处理大小固定为`1`。

转换后的模型将被写入`tiny_conv.tflite`。恭喜！这是一个完全成型的 TensorFlow Lite 模型！

查看这个模型有多小，在下一个单元格中运行以下代码：

```cpp
import os
model_size = os.path.getsize("/content/tiny_conv.tflite")
print("Model is %d bytes" % model_size)
```

输出显示模型非常小：`模型大小为 18208 字节`。

我们的下一步是将这个模型转换为可以部署到微控制器的形式。

### 创建一个 C 数组

回到“转换为 C 文件”中，我们使用`xxd`命令将 TensorFlow Lite 模型转换为 C 数组。我们将在下一个单元格中做同样的事情：

```cpp
# Install xxd if it is not available
!apt-get -qq install xxd
# Save the file as a C source file
!xxd -i /content/tiny_conv.tflite > /content/tiny_conv.cc
# Print the source file
!cat /content/tiny_conv.cc
```

输出的最后部分将是文件的内容，其中包括一个 C 数组和一个保存其长度的整数，如下所示（您看到的确切值可能略有不同）：

```cpp
unsigned char _content_tiny_conv_tflite[] = {
  0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x0e, 0x00, 0x18, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0c, 0x00,
  // ...
  0x00, 0x09, 0x06, 0x00, 0x08, 0x00, 0x07, 0x00, 0x06, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x04
};
unsigned int _content_tiny_conv_tflite_len = 18208;
```

这段代码也被写入一个文件`tiny_conv.cc`，您可以使用 Colab 的文件浏览器下载。因为您的 Colab 运行时将在 12 小时后到期，现在将此文件下载到您的计算机是一个好主意。

接下来，我们将把这个新训练过的模型与`micro_speech`项目集成起来，以便我们可以将其部署到一些硬件上。

# 在我们的项目中使用模型

要使用我们的新模型，我们需要做三件事：

1. 在[*micro_features/tiny_conv_micro_features_model_data.cc*](https://oreil.ly/EAR0U)中，用我们的新模型替换原始模型数据。
2. 在[*micro_features/micro_model_settings.cc*](https://oreil.ly/bqw67)中用我们的新“on”和“off”标签更新标签名称。
3. 更新特定设备的`command_responder.cc`以执行我们对新标签的操作。

## 替换模型

要替换模型，请在文本编辑器中打开`micro_features/tiny_conv_micro_features_model_data.cc`。

###### 注意

如果你正在使用 Arduino 示例，该文件将显示为 Arduino IDE 中的一个选项卡。它的名称将是`micro_features_tiny_conv_micro_features_model_data.cpp`。如果你正在使用 SparkFun Edge，你可以直接在本地的 TensorFlow 存储库副本中编辑文件。如果你正在使用 STM32F746G，你应该在 Mbed 项目目录中编辑文件。

`tiny_conv_micro_features_model_data.cc`文件包含一个看起来像这样的数组声明：

```cpp
const unsigned char
    g_tiny_conv_micro_features_model_data[] DATA_ALIGN_ATTRIBUTE = {
        0x18, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x00, 0x00, 0x0e, 0x00,
        0x18, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x10, 0x00, 0x14, 0x00,
        //...
        0x00, 0x09, 0x06, 0x00, 0x08, 0x00, 0x07, 0x00, 0x06, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x04};
const int g_tiny_conv_micro_features_model_data_len = 18208;
```

需要替换数组的内容以及常量`g_tiny_conv_micro_features_model_data_len`的值，如果已经更改。

为此，打开你在上一节末尾下载的`tiny_conv.cc`文件。复制并粘贴数组的内容，但不包括定义，到`tiny_conv_micro_features_model_data.cc`中定义的数组中。确保你正在覆盖数组的内容，但不是它的声明。

在`tiny_conv.cc`的底部，你会找到`_content_tiny_conv_tflite_len`，一个变量，其值表示数组的长度。回到`tiny_conv_micro_features_model_data.cc`，用这个变量的值替换`g_tiny_conv_micro_features_model_data_len`的值。然后保存文件；你已经完成了更新。

## 更新标签

接下来，打开`micro_features/micro_model_settings.cc`。这个文件包含一个类标签的数组：

```cpp
const char* kCategoryLabels[kCategoryCount] = {
    "silence",
    "unknown",
    "yes",
    "no",
};
```

为了调整我们的新模型，我们可以简单地将“yes”和“no”交换为“on”和“off”。我们按顺序将标签与模型的输出张量元素匹配，因此重要的是按照它们提供给训练脚本的顺序列出这些标签。

以下是预期的代码：

```cpp
const char* kCategoryLabels[kCategoryCount] = {
    "silence",
    "unknown",
    "on",
    "off",
};
```

如果你训练了一个具有两个以上标签的模型，只需将它们全部添加到列表中。

我们现在已经完成了切换模型的工作。唯一剩下的步骤是更新使用标签的任何输出代码。

## 更新 command_responder.cc

该项目包含针对 Arduino、SparkFun Edge 和 STM32F746G 的不同设备特定实现的`command_responder.cc`。我们将在以下部分展示如何更新每个设备。

### Arduino

位于`arduino/command_responder.cc`中的 Arduino 命令响应器在听到“yes”时会点亮 LED 3 秒钟。让我们将其更新为在听到“on”或“off”时点亮 LED。在文件中，找到以下`if`语句：

```cpp
// If we heard a "yes", switch on an LED and store the time.
if (found_command[0] == 'y') {
  last_yes_time = current_time;
  digitalWrite(LED_BUILTIN, HIGH);
}
```

`if`语句测试命令的第一个字母是否为“y”，表示“yes”。如果我们将这个“y”改为“o”，LED 将点亮“on”或“off”，因为它们都以“o”开头：

```cpp
if (found_command[0] == 'o') {
  last_yes_time = current_time;
  digitalWrite(LED_BUILTIN, HIGH);
}
```

完成这些代码更改后，部署到你的设备并尝试一下。

### SparkFun Edge

位于`sparkfun_edge/command_responder.cc`中的 SparkFun Edge 命令响应器会根据听到的“yes”或“no”点亮不同的 LED。在文件中，找到以下`if`语句：

```cpp
if (found_command[0] == 'y') {
  am_hal_gpio_output_set(AM_BSP_GPIO_LED_YELLOW);
}
if (found_command[0] == 'n') {
  am_hal_gpio_output_set(AM_BSP_GPIO_LED_RED);
}
if (found_command[0] == 'u') {
  am_hal_gpio_output_set(AM_BSP_GPIO_LED_GREEN);
}
```

很容易更新这些，使得“on”和“off”分别点亮不同的 LED：

```cpp
if (found_command[0] == 'o' && found_command[1] == 'n') {
  am_hal_gpio_output_set(AM_BSP_GPIO_LED_YELLOW);
}
if (found_command[0] == 'o' && found_command[1] == 'f') {
  am_hal_gpio_output_set(AM_BSP_GPIO_LED_RED);
}
if (found_command[0] == 'u') {
  am_hal_gpio_output_set(AM_BSP_GPIO_LED_GREEN);
}
```

因为这两个命令都以相同的字母开头，我们需要查看它们的第二个字母来消除歧义。现在，当说“on”时，黄色 LED 将点亮，当说“off”时，红色 LED 将点亮。

完成更改后，部署并运行代码，使用与“运行示例”中遵循的相同过程。

### STM32F746G

位于`disco_f746ng/command_responder.cc`中的 STM32F746G 命令响应器会根据听到的命令显示不同的单词。在文件中，找到以下`if`语句：

```cpp
if (*found_command == 'y') {
  lcd.Clear(0xFF0F9D58);
  lcd.DisplayStringAt(0, LINE(5), (uint8_t *)"Heard yes!", CENTER_MODE);
} else if (*found_command == 'n') {
  lcd.Clear(0xFFDB4437);
  lcd.DisplayStringAt(0, LINE(5), (uint8_t *)"Heard no :(", CENTER_MODE);
} else if (*found_command == 'u') {
  lcd.Clear(0xFFF4B400);
  lcd.DisplayStringAt(0, LINE(5), (uint8_t *)"Heard unknown", CENTER_MODE);
} else {
  lcd.Clear(0xFF4285F4);
  lcd.DisplayStringAt(0, LINE(5), (uint8_t *)"Heard silence", CENTER_MODE);
}
```

很容易更新以便响应“on”和“off”：

```cpp
if (found_command[0] == 'o' && found_command[1] == 'n') {
  lcd.Clear(0xFF0F9D58);
  lcd.DisplayStringAt(0, LINE(5), (uint8_t *)"Heard on!", CENTER_MODE);
} else if (found_command[0] == 'o' && found_command[1] == 'f') {
  lcd.Clear(0xFFDB4437);
  lcd.DisplayStringAt(0, LINE(5), (uint8_t *)"Heard off", CENTER_MODE);
} else if (*found_command == 'u') {
  lcd.Clear(0xFFF4B400);
  lcd.DisplayStringAt(0, LINE(5), (uint8_t *)"Heard unknown", CENTER_MODE);
} else {
  lcd.Clear(0xFF4285F4);
  lcd.DisplayStringAt(0, LINE(5), (uint8_t *)"Heard silence", CENTER_MODE);
}
```

同样，因为这两个命令都以相同的字母开头，我们需要查看它们的第二个字母来消除歧义。现在我们为每个命令显示适当的文本。

## 运行脚本的其他方法

如果你无法使用 Colab，有两种其他推荐的训练模型的方法：

+   在一个带有 GPU 的云虚拟机（VM）中
+   在你的本地工作站上

进行基于 GPU 的训练所需的驱动程序仅在 Linux 上可用。没有 Linux，训练将需要大约四个小时。因此，建议使用带有 GPU 的云虚拟机或类似配置的 Linux 工作站。
