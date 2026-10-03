
接下来，您将重置板子到引导加载程序状态并刷写板子。在板子上，找到标记为`RST`和`14`的按钮，如图 6-14 所示。

![显示 SparkFun Edge 按钮的照片](img/timl_0614.png)

###### 图 6-14。SparkFun Edge 的按钮

执行以下步骤：

1. 确保您的板子连接到编程器，并且整个设备通过 USB 连接到计算机。
2. 在板子上，按住标记为`14`的按钮。*继续按住它*。
3. 在继续按住标记为`14`的按钮的同时，按下标记为`RST`的按钮重置板子。
4. 在计算机上按 Enter 键运行脚本。*继续按住按钮`14`*。

现在您应该在屏幕上看到类似以下内容的东西：

```cpp
Connecting with Corvette over serial port /dev/cu.usbserial-1440...
Sending Hello.
Received response for Hello
Received Status
length =  0x58
version =  0x3
Max Storage =  0x4ffa0
Status =  0x2
State =  0x7
AMInfo =
0x1
0xff2da3ff
0x55fff
0x1
0x49f40003
0xffffffff
[...lots more 0xffffffff...]
Sending OTA Descriptor =  0xfe000
Sending Update Command.
number of updates needed =  1
Sending block of size  0x158b0  from  0x0  to  0x158b0
Sending Data Packet of length  8180
Sending Data Packet of length  8180
[...lots more Sending Data Packet of length  8180...]
```

继续按住按钮`14`，直到看到`发送数据包长度为 8180`。在看到这个之后可以释放按钮（但如果您继续按住也没关系）。

程序将继续在终端上打印行。最终您会看到类似以下内容的东西：

```cpp
[...lots more Sending Data Packet of length  8180...]
Sending Data Packet of length  8180
Sending Data Packet of length  6440
Sending Reset Command.
Done.
```

这表明刷写成功。

###### 提示

如果程序输出以错误结束，请检查是否打印了`发送复位命令`。如果是，则刷写可能成功，尽管有错误。否则，刷写可能失败。尝试再次运行这些步骤（您可以跳过设置环境变量）。

## 测试程序

现在应该已经将二进制文件部署到设备上。按下标记为`RST`的按钮重新启动板子。您应该看到设备的四个 LED 按顺序闪烁。干得好！

## 查看调试数据

在程序运行时，板子会记录调试信息。要查看它，我们可以使用波特率 115200 监视板子的串行端口输出。在 macOS 和 Linux 上，以下命令应该有效：

```cpp
screen ${DEVICENAME} 115200
```

您会看到大量的输出飞过！要停止滚动，请按下 Ctrl-A，紧接着按 Esc。然后您可以使用箭头键浏览输出，其中包含对各种`x`值运行推断的结果：

```cpp
x_value: 1.1843798*2², y_value: -1.9542645*2^-1
```

要停止使用`screen`查看调试输出，请按下 Ctrl-A，紧接着按下 K 键，然后按下 Y 键。

###### 注意

程序`screen`是一个有用的连接到其他计算机的实用程序程序。在这种情况下，我们使用它来监听 SparkFun Edge 板通过其串行端口记录的数据。如果您使用 Windows，可以尝试使用程序[`CoolTerm`](https://oreil.ly/sPWQP)来做同样的事情。

## 进行您自己的更改

现在您已经部署了基本应用程序，请尝试玩耍并进行一些更改。您可以在*tensorflow/lite/micro/examples/hello_world*文件夹中找到应用程序的代码。只需编辑并保存，然后重复之前的说明以将修改后的代码部署到设备上。

以下是您可以尝试的一些事情：

+ 通过调整每个周期的推断次数使 LED 的闪烁速度变慢或变快。
+ 修改*output_handler.cc*以将基于文本的动画记录到串行端口。
+ 使用正弦波来控制其他组件，如额外的 LED 或声音发生器。

# ST Microelectronics STM32F746G Discovery Kit

[STM32F746G](https://oreil.ly/cvm4J)是一个带有相对强大的 Arm Cortex-M7 处理器核心的微控制器开发板。

这块板运行 Arm 的[Mbed OS](https://os.mbed.com)，这是一个专为构建和部署嵌入式应用程序而设计的嵌入式操作系统。这意味着我们可以使用本节中的许多指令来为其他 Mbed 设备构建。

STM32F746G 配备了一个附加的 LCD 屏幕，这将使我们能够构建一个更加复杂的视觉显示。

## 在 STM32F746G 上处理输出

现在我们有一个整个 LCD 可以玩耍，我们可以绘制一个漂亮的动画。让我们使用屏幕的*x*轴表示推断的数量，*y*轴表示我们预测的当前值。

我们将在这个值应该的地方绘制一个点，当我们循环遍历 0 到 2π的输入范围时，它将在屏幕上移动。图 6-15 展示了这个的线框图。

每个推断需要一定的时间，因此调整`kInferencesPerCycle`，在*constants.cc*中定义，将调整点的运动速度和平滑度。

图 6-16 显示了程序运行的[动画*.gif*](https://oreil.ly/1EM7C)的静止画面。

![我们将在 LCD 显示屏上绘制的动画的插图](img/timl_0615.png)

###### 图 6-15。我们将在 LCD 显示屏上绘制的动画

图 6-16 显示了程序运行的[动画*.gif*](https://oreil.ly/1EM7C)的静止画面。

![STM32F746G Discovery kit，带有 LCD 显示屏](img/timl_0616.png)

###### 图 6-16。在 STM32F746G Discovery kit 上运行的代码，带有 LCD 显示屏

实现 STM32F746G 的输出处理的代码位于[*hello_world/disco_f746ng/output_handler.cc*](https://oreil.ly/bj4iL)中，该文件用于替代原始文件*hello_world/output_handler.cc*。

让我们来看一下：

```cpp
#include "tensorflow/lite/micro/examples/hello_world/output_handler.h"
#include "LCD_DISCO_F746NG.h"
#include "tensorflow/lite/micro/examples/hello_world/constants.h"
```

首先，我们有一些头文件。我们的*output_handler.h*指定了这个文件的接口。由开发板制造商提供的*LCD_DISCO_F74NG.h*声明了我们将用来控制其 LCD 屏幕的接口。我们还包括*constants.h*，因为我们需要访问`kInferencesPerCycle`和`kXrange`。

接下来，我们设置了大量变量。首先是`LCD_DISCO_F746NG`的一个实例，它在*LCD_DISCO_F74NG.h*中定义，并提供了我们可以用来控制 LCD 的方法：

```cpp
// The LCD driver
LCD_DISCO_F746NG lcd;
```

有关`LCD_DISCO_F746NG`类的详细信息可在[Mbed 网站](https://oreil.ly/yiPHS)上找到。

接下来，我们定义一些控制视觉外观和感觉的常量：

```cpp
// The colors we'll draw
const uint32_t background_color = 0xFFF4B400;  // Yellow
const uint32_t foreground_color = 0xFFDB4437;  // Red
// The size of the dot we'll draw
const int dot_radius = 10;
```

颜色以十六进制值提供，如`0xFFF4B400`。它们的格式为`AARRGGBB`，其中`AA`表示 alpha 值（或不透明度，`FF`表示完全不透明），`RR`、`GG`和`BB`表示红色、绿色和蓝色的量。

###### 提示

通过练习，您可以学会从十六进制值中读取颜色。`0xFFF4B400`是完全不透明的，有很多红色和一定量的绿色，这使得它成为一个漂亮的橙黄色。

您也可以通过快速的谷歌搜索查找这些值。

然后我们声明了一些变量，定义了动画的形状和大小：

```cpp
// Size of the drawable area
int width;
int height;
// Midpoint of the y axis
int midpoint;
// Pixels per unit of x_value
int x_increment;
```

在变量之后，我们定义了`HandleOutput()`函数，并告诉它在第一次运行时要做什么：

```cpp
// Animates a dot across the screen to represent the current x and y values
void HandleOutput(tflite::ErrorReporter* error_reporter, float x_value,
                  float y_value) {
  // Track whether the function has run at least once
  static bool is_initialized = false;

  // Do this only once
  if (!is_initialized) {
    // Set the background and foreground colors
    lcd.Clear(background_color);
    lcd.SetTextColor(foreground_color);
    // Calculate the drawable area to avoid drawing off the edges
    width = lcd.GetXSize() - (dot_radius * 2);
    height = lcd.GetYSize() - (dot_radius * 2);
    // Calculate the y axis midpoint
    midpoint = height / 2;
    // Calculate fractional pixels per unit of x_value
    x_increment = static_cast<float>(width) / kXrange;
    is_initialized = true;
  }
```

里面有很多内容！首先，我们使用属于`lcd`的方法来设置背景和前景颜色。奇怪命名的`lcd.SetTextColor()`设置我们绘制的任何东西的颜色，不仅仅是文本：

```cpp
// Set the background and foreground colors
lcd.Clear(background_color);
lcd.SetTextColor(foreground_color);
```

接下来，我们计算实际可以绘制到屏幕的部分，以便知道在哪里绘制我们的圆。如果我们搞错了，我们可能会尝试绘制超出屏幕边缘，导致意想不到的结果：

```cpp
width = lcd.GetXSize() - (dot_radius * 2);
height = lcd.GetYSize() - (dot_radius * 2);
```

之后，我们确定屏幕中间的位置，我们将在其下方绘制负`y`值。我们还计算屏幕宽度中表示一个单位`x`值的像素数。请注意我们如何使用`static_cast`确保获得浮点结果：

```cpp
// Calculate the y axis midpoint
midpoint = height / 2;
// Calculate fractional pixels per unit of x_value
x_increment = static_cast<float>(width) / kXrange;
```

与之前一样，使用名为`is_initialized`的`static`变量确保此块中的代码仅运行一次。

初始化完成后，我们可以开始输出。首先，清除任何先前的绘图：

```cpp
// Clear the previous drawing
lcd.Clear(background_color);
```

接下来，我们使用`x_value`来计算我们应该在显示器的*x*轴上绘制点的位置：

```cpp
// Calculate x position, ensuring the dot is not partially offscreen,
// which causes artifacts and crashes
int x_pos = dot_radius + static_cast<int>(x_value * x_increment);
```

然后我们对`y`值执行相同的操作。这有点复杂，因为我们希望在`midpoint`上方绘制正值，在下方绘制负值：

```cpp
// Calculate y position, ensuring the dot is not partially offscreen
int y_pos;
if (y_value >= 0) {
  // Since the display's y runs from the top down, invert y_value
  y_pos = dot_radius + static_cast<int>(midpoint * (1.f - y_value));
} else {
  // For any negative y_value, start drawing from the midpoint
  y_pos =
      dot_radius + midpoint + static_cast<int>(midpoint * (0.f - y_value));
}
```

一旦确定了它的位置，我们就可以继续绘制点：

```cpp
// Draw the dot
lcd.FillCircle(x_pos, y_pos, dot_radius);
```

最后，我们使用我们的`ErrorReporter`将`x`和`y`值记录到串行端口：

```cpp
// Log the current X and Y values
error_reporter->Report("x_value: %f, y_value: %f\n", x_value, y_value);
```

###### 注意

由于自定义实现，`ErrorReporter`可以通过 STM32F746G 的串行接口输出数据，[*micro/disco_f746ng/debug_log.cc*](https://oreil.ly/eL1ft)，取代了[*micro/debug_log.cc*](https://oreil.ly/HpJ-t)中的原始实现。

## 运行示例

接下来，让我们构建项目！STM32F746G 运行 Arm 的 Mbed OS，因此我们将使用 Mbed 工具链将我们的应用程序部署到设备上。

###### 提示

建议检查[*README.md*](https://oreil.ly/WuhIz)以获取最新说明，因为构建过程可能会有所变化。

在开始之前，我们需要以下内容：

+ 一个 STM32F746G Discovery kit 板
+ 一个迷你 USB 电缆
+ Arm Mbed CLI（请参阅[Mbed 设置指南](https://oreil.ly/TkRwd)）
+ Python 3 和`pip`

与 Arduino IDE 类似，Mbed 要求源文件以特定方式结构化。 TensorFlow Lite for Microcontrollers Makefile 知道如何为我们做到这一点，并且可以生成适用于 Mbed 的目录。

为了这样做，请运行以下命令：

```cpp
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=mbed TAGS="CMSIS disco_f746ng" generate_hello_world_mbed_project
```

这将导致创建一个新目录：

```cpp
tensorflow/lite/micro/tools/make/gen/mbed_cortex-m4/prj/ \
  hello_world/mbed
```

该目录包含所有示例的依赖项，以适合 Mbed 构建。

首先，切换到目录，以便您可以在其中运行一些命令：

```cpp
cd tensorflow/lite/micro/tools/make/gen/mbed_cortex-m4/prj/ \
  hello_world/mbed
```

现在，您将使用 Mbed 下载依赖项并构建项目。

要开始，请使用以下命令指定 Mbed 当前目录是 Mbed 项目的根目录：

```cpp
mbed config root .
```

接下来，指示 Mbed 下载依赖项并准备构建：

```cpp
mbed deploy
```
