
```cpp
int main(int argc, char* argv[]) {
  setup();
  while (true) {
    loop();
  }
}
```

这就是整个程序！这个例子很棒，因为它表明与复杂的机器学习模型一起工作可以出奇地简单。模型包含了所有的复杂性，我们只需要提供数据给它。

在我们继续之前，您可以在本地运行程序进行尝试。图像提供程序的参考实现只返回虚拟数据，因此您不会得到有意义的识别结果，但至少可以看到代码在运行。

首先，使用以下命令构建程序：

```cpp
make -f tensorflow/lite/micro/tools/make/Makefile person_detection
```

构建完成后，您可以使用以下命令运行示例：

```cpp
tensorflow/lite/micro/tools/make/gen/osx_x86_64/bin/ \
person_detection
```

您会看到程序的输出在屏幕上滚动，直到按下 `Ctrl-C` 终止它：

```cpp
person score:129 no person score 202
person score:129 no person score 202
person score:129 no person score 202
person score:129 no person score 202
person score:129 no person score 202
person score:129 no person score 202
```

在接下来的部分中，我们将详细介绍特定设备的代码，该代码将捕获摄像头图像并在每个平台上输出结果。我们还展示了如何部署和运行此代码。

# 部署到微控制器

在这一部分中，我们将代码部署到两个熟悉的设备上：

+   [Arduino Nano 33 BLE Sense](https://oreil.ly/6qlMD)

+   [SparkFun Edge](https://oreil.ly/-hoL-)

这次有一个很大的不同：因为这两个设备都没有内置摄像头，我们建议您为您使用的任何设备购买摄像头模块。每个设备都有自己的`image_provider.cc`实现，它与摄像头模块进行接口，以捕获图像。`detection_responder.cc`中还有特定于设备的输出代码。

这很简单，所以它将是一个很好的模板，用来创建你自己的基于视觉的 ML 应用程序。

让我们开始探索 Arduino 的实现。

## Arduino

作为 Arduino 板，Arduino Nano 33 BLE Sense 可以访问大量兼容的第三方硬件和库的生态系统。我们使用了一个专为与 Arduino 配合使用而设计的第三方摄像头模块，以及一些 Arduino 库，这些库将与我们的摄像头模块进行接口，并理解其输出的数据。

### 要购买哪种摄像头模块

这个例子使用 [Arducam Mini 2MP Plus](https://oreil.ly/LAwhb) 摄像头模块。它很容易连接到 Arduino Nano 33 BLE Sense，并且可以由 Arduino 板的电源供应提供电力。它有一个大镜头，能够捕获高质量的 200 万像素图像 - 尽管我们将使用其内置的图像重缩放功能来获得较小的分辨率。它并不特别节能，但其高质量的图像使其非常适合构建图像捕获应用程序，比如用于记录野生动物。

### 在 Arduino 上捕获图像

我们通过一些引脚将 Arducam 模块连接到 Arduino 板。为了获取图像数据，我们从 Arduino 板向 Arducam 发送一个命令，指示它捕获图像。Arducam 将执行此操作，将图像存储在其内部数据缓冲区中。然后，我们发送进一步的命令，允许我们从 Arducam 的内部缓冲区中读取图像数据并将其存储在 Arduino 的内存中。为了执行所有这些操作，我们使用官方的 Arducam 库。

Arducam 相机模块具有一颗 200 万像素的图像传感器，分辨率为 1920×1080。我们的人体检测模型的输入尺寸仅为 96×96，因此我们不需要所有这些数据。事实上，Arduino 本身没有足够的内存来容纳一张 200 万像素的图像，其大小将达到几兆字节。

幸运的是，Arducam 硬件具有将输出调整为更小分辨率的能力，即 160×120 像素。我们可以通过在代码中仅保留中心的 96×96 像素来轻松将其裁剪为 96×96。然而，为了复杂化问题，Arducam 的调整大小输出使用了 JPEG，这是一种常见的图像压缩格式。我们的模型需要一个像素数组，而不是一个 JPEG 编码的图像，这意味着我们需要在使用之前解码 Arducam 的输出。我们可以使用一个开源库来实现这一点。

我们的最后任务是将 Arducam 的彩色图像输出转换为灰度，这是我们的人体检测模型所期望的。我们将灰度数据写入我们模型的输入张量。

图像提供程序实现在 [*arduino/image_provider.cc*](https://oreil.ly/kGx0-) 中。我们不会解释其每个细节，因为代码是特定于 Arducam 相机模块的。相反，让我们以高层次的方式来看一下发生了什么。

`GetImage()` 函数是图像提供程序与外部世界的接口。在我们的应用程序主循环中调用它以获取一帧图像数据。第一次调用时，我们需要初始化相机。这通过调用 `InitCamera()` 函数来实现，如下所示：

```cpp
  static bool g_is_camera_initialized = false;
  if (!g_is_camera_initialized) {
    TfLiteStatus init_status = InitCamera(error_reporter);
    if (init_status != kTfLiteOk) {
      error_reporter->Report("InitCamera failed");
      return init_status;
    }
    g_is_camera_initialized = true;
  }
```

`InitCamera()` 函数在 `image_provider.cc` 中进一步定义。我们不会在这里详细介绍它，因为它非常特定于设备，如果您想在自己的代码中使用它，只需复制粘贴即可。它配置 Arduino 的硬件以与 Arducam 通信，然后确认通信正常工作。最后，它指示 Arducam 输出 160×120 像素的 JPEG 图像。

`GetImage()` 函数调用的下一个函数是 `PerformCapture()`：

```cpp
TfLiteStatus capture_status = PerformCapture(error_reporter);
```

我们也不会详细介绍这个函数。它只是向相机模块发送一个命令，指示其捕获图像并将图像数据存储在其内部缓冲区中。然后，它等待确认图像已被捕获。此时，Arducam 的内部缓冲区中有图像数据，但 Arduino 本身还没有任何图像数据。

接下来我们调用的函数是 `ReadData()`：

```cpp
  TfLiteStatus read_data_status = ReadData(error_reporter);
```

`ReadData()` 函数使用更多的命令从 Arducam 获取图像数据。函数运行后，全局变量 `jpeg_buffer` 将填充从相机检索到的 JPEG 编码图像数据。

当我们有 JPEG 编码的图像时，我们的下一步是将其解码为原始图像数据。这发生在 `DecodeAndProcessImage()` 函数中：

```cpp
  TfLiteStatus decode_status = DecodeAndProcessImage(
      error_reporter, image_width, image_height, image_data);
```

该函数使用一个名为 JPEGDecoder 的库来解码 JPEG 数据，并直接将其写入模型的输入张量。在此过程中，它裁剪图像，丢弃一些 160×120 的数据，使剩下的只有 96×96 像素，大致位于图像中心。它还将图像的 16 位颜色表示减少到 8 位灰度。

在图像被捕获并存储在输入张量中后，我们准备运行推理。接下来，我们展示模型的输出是如何显示的。

### 在 Arduino 上响应检测

Arduino Nano 33 BLE Sense 内置了 RGB LED，这是一个包含独立红色、绿色和蓝色 LED 的单一组件，您可以分别控制它们。检测响应器的实现在每次推理运行时闪烁蓝色 LED。当检测到人时，点亮绿色 LED；当未检测到人时，点亮红色 LED。

实现在 [*arduino/detection_responder.cc*](https://oreil.ly/-WsSN) 中。让我们快速浏览一下。

`RespondToDetection()` 函数接受两个分数，一个用于“人”类别，另一个用于“非人”。第一次调用时，它设置蓝色、绿色和黄色 LED 为输出：

```cpp
void RespondToDetection(tflite::ErrorReporter* error_reporter,
                        uint8_t person_score, uint8_t no_person_score) {
  static bool is_initialized = false;
  if (!is_initialized) {
    pinMode(led_green, OUTPUT);
    pinMode(led_blue, OUTPUT);
    is_initialized = true;
  }
```

接下来，为了指示推理刚刚完成，我们关闭所有 LED，然后非常简要地闪烁蓝色 LED：

```cpp
  // Note: The RGB LEDs on the Arduino Nano 33 BLE
  // Sense are on when the pin is LOW, off when HIGH.

  // Switch the person/not person LEDs off
  digitalWrite(led_green, HIGH);
  digitalWrite(led_red, HIGH);

  // Flash the blue LED after every inference.
  digitalWrite(led_blue, LOW);
  delay(100);
  digitalWrite(led_blue, HIGH);
```

您会注意到，与 Arduino 内置 LED 不同，这些 LED 使用 `LOW` 打开，使用 `HIGH` 关闭。这只是 LED 连接到板上的方式的一个因素。

接下来，我们根据哪个类别的分数更高来打开和关闭适当的 LED：

```cpp
  // Switch on the green LED when a person is detected,
  // the red when no person is detected
  if (person_score > no_person_score) {
    digitalWrite(led_green, LOW);
    digitalWrite(led_red, HIGH);
  } else {
    digitalWrite(led_green, HIGH);
    digitalWrite(led_red, LOW);
  }
```

最后，我们使用 `error_reporter` 实例将分数输出到串行端口：

```cpp
  error_reporter->Report("Person score: %d No person score: %d", person_score,
                         no_person_score);
}
```

就是这样！函数的核心是一个基本的 `if` 语句，您可以轻松使用类似的逻辑来控制其他类型的输出。将如此复杂的视觉输入转换为一个布尔输出“人”或“非人”是非常令人兴奋的事情。

### 运行示例

运行此示例比我们其他 Arduino 示例更复杂，因为我们需要将 Arducam 连接到 Arduino 板。我们还需要安装和配置与 Arducam 接口并解码其 JPEG 输出的库。但不用担心，这仍然非常简单！

要部署此示例，我们需要以下内容：

+   一个 Arduino Nano 33 BLE Sense 板

+   一个 Arducam Mini 2MP Plus

+   跳线（和可选的面包板）

+   一根 Micro-USB 电缆

+   Arduino IDE

我们的第一个任务是使用跳线连接 Arducam 到 Arduino。这不是一本电子书，所以我们不会详细介绍使用电缆的细节。相反，表 9-1 显示了引脚应该如何连接。每个设备上都标有引脚标签。

表 9-1。Arducam Mini 2MP Plus 到 Arduino Nano 33 BLE Sense 的连接

| Arducam 引脚 | Arduino 引脚 |

| --- | --- |

| CS | D7（未标记，紧挨 D6 右侧） |

| MOSI | D11 |

| MISO | D12 |

| SCK | D13 |

| GND | GND（任何一个标记为 GND 的引脚都可以） |

| VCC | 3.3 V |

| SDA | A4 |

| SCL | A5 |

设置硬件后，您可以继续安装软件。

###### 提示

建立过程可能会有所变化，所以请查看 [*README.md*](https://oreil.ly/CR5Pb) 获取最新说明。

本书中的项目作为 TensorFlow Lite Arduino 库中的示例代码可用。如果您尚未安装该库，请打开 Arduino IDE 并从“工具”菜单中选择“管理库”。在弹出的窗口中，搜索并安装名为 `Arduino_TensorFlowLite` 的库。您应该能够使用最新版本，但如果遇到问题，本书测试过的版本是 1.14-ALPHA。

###### 注意

您还可以从 `.zip` 文件安装库，您可以从 TensorFlow Lite 团队 [下载](https://oreil.ly/blgB8) 或使用 TensorFlow Lite for Microcontrollers Makefile 自动生成。如果您更喜欢后者，请参阅附录 A。

安装完库后，`person_detection` 示例将显示在“文件”菜单下的“示例→Arduino_TensorFlowLite”中，如图 9-2 所示。

![“示例”菜单的屏幕截图](img/timl_0604.png)

###### 图 9-2。示例菜单

点击“person_detection”加载示例。它将显示为一个新窗口，每个源文件都有一个选项卡。第一个选项卡中的文件 `person_detection` 相当于我们之前介绍的 `main_functions.cc`。

###### 注意

“运行示例”已经解释了 Arduino 示例的结构，所以我们这里不再重复覆盖。

除了 TensorFlow 库，我们还需要安装另外两个库：

+   Arducam 库，以便我们的代码可以与硬件进行交互

+   JPEGDecoder 库，以便我们可以解码 JPEG 编码的图像

Arducam Arduino 库可从 [GitHub](https://oreil.ly/93OKK) 获取。要安装它，请下载或克隆存储库。接下来，将其 `ArduCAM` 子目录复制到 `Arduino/libraries` 目录中。要找到您机器上的 `libraries` 目录，请在 Arduino IDE 的首选项窗口中检查 Sketchbook 位置。

下载库后，您需要编辑其中一个文件，以确保为 Arducam Mini 2MP Plus 进行配置。为此，请打开 `Arduino/libraries/ArduCAM/memorysaver.h`。

您会看到一堆 `#define` 语句。确保它们都被注释掉，除了 `#define OV2640_MINI_2MP_PLUS`，如此处所示：

```cpp
//Step 1: select the hardware platform, only one at a time
//#define OV2640_MINI_2MP
//#define OV3640_MINI_3MP
//#define OV5642_MINI_5MP
//#define OV5642_MINI_5MP_BIT_ROTATION_FIXED
#define OV2640_MINI_2MP_PLUS
//#define OV5642_MINI_5MP_PLUS
//#define OV5640_MINI_5MP_PLUS
```

保存文件后，您已经完成了 Arducam 库的配置。

###### 提示

示例是使用 Arducam 库的提交 #e216049 开发的。如果您在使用库时遇到问题，可以尝试下载这个特定的提交，以确保您使用的是完全相同的代码。

下一步是安装 JPEGDecoder 库。您可以在 Arduino IDE 中完成这个操作。在工具菜单中，选择管理库选项并搜索 JPEGDecoder。您应该安装库的 1.8.0 版本。

安装完库之后，您需要配置它以禁用一些与 Arduino Nano 33 BLE Sense 不兼容的可选组件。打开 `Arduino/libraries/JPEGDecoder/src/User_Config.h`，确保 `#define LOAD_SD_LIBRARY` 和 `#define LOAD_SDFAT_LIBRARY` 都被注释掉，如文件中的摘录所示：
