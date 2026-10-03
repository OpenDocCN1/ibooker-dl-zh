
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

继续按住按钮`14`，直到看到`Sending Data Packet of length 8180`。在看到这个后可以释放按钮（但如果您继续按住也没关系）。

程序将继续在终端上打印行。最终，您会看到类似以下内容：

```cpp
[...lots more Sending Data Packet of length  8180...]
Sending Data Packet of length  8180
Sending Data Packet of length  6440
Sending Reset Command.
Done.
```

这表示刷写成功。

## 提示

如果程序输出以错误结束，请检查是否打印了`Sending Reset Command.`。如果是，则尽管有错误，刷写很可能成功。否则，刷写可能失败。尝试再次运行这些步骤（您可以跳过设置环境变量）。

## 测试程序

首先按下`RST`按钮确保程序正在运行。当程序运行时，黄色 LED 灯会每次推断时闪烁一次。

接下来，使用以下命令开始打印设备的串行输出：

```cpp
screen ${DEVICENAME} 115200
```

最初您应该会看到以下输出：

```cpp
Magic starts!
```

现在您可以尝试做一些手势。用一只手举起板子，组件朝上，USB 适配器朝向左侧，如图 11-13 所示。

![一只手拿着 SparkFun Edge 板子的照片](img/timl_1113.png)

**图 11-13\. 手势执行时如何握住板子**

图 11-14 展示了如何执行每个手势的图表。由于模型是在板子连接到魔杖时收集的数据上进行训练的，您可能需要尝试几次才能使它们正常工作。

![三个魔杖手势的图表](img/timl_1103.png)

**图 11-14\. 三个魔杖手势**

最容易开始的是“翅膀”。您应该将手移动得足够快，以便大约一秒钟完成手势。如果成功，红色 LED 灯应该会亮起，并且您应该会看到以下输出：

```cpp
WING:
*         *         *
 *       * *       *
  *     *   *     *
   *   *     *   *
    * *       * *
     *         *
```

恭喜，您已经使用 SparkFun Edge 施展了您的第一个魔法咒语！

## 注意

此时，您可以选择发挥创造力，将板子连接到魔杖的尖端，距离手最远的地方。任何棍子、尺子或其他长度约为一英尺（30 厘米）的家用物品都应该很好用。

确保设备牢固连接，并且朝向相同，组件朝上，USB 适配器朝向左侧。选择一个坚固的魔杖，而不是一个灵活的魔杖，因为任何摇晃都会影响加速度计读数。

接下来尝试“环”手势，用手（或魔杖的尖端）顺时针画一个圆圈。再次，尽量花大约一秒钟来执行手势。您应该会看到以下内容，仿佛是魔法般出现的：

```cpp
RING:
          *
       *     *
     *         *
    *           *
     *         *
       *     *
          *
```

对于最后一个手势，在空中画一个三角形的角。最好通过 ASCII 艺术演示来描述，如下所示：

```cpp
SLOPE:
        *
       *
      *
     *
    *
   *
  *
 * * * * * * * *
```

就像任何好的魔法咒语一样，您可能需要练习一下，才能每次都完美地执行它们。您可以在[*README.md*](https://oreil.ly/ilGJY)中看到手势的视频演示。

## 进行您自己的更改

现在您已经部署了基本应用程序，请尝试玩耍并进行一些更改。您可以在`tensorflow/lite/micro/examples/magic_wand`文件夹中找到应用程序的代码。只需编辑并保存，然后重复之前的说明以将修改后的代码部署到设备上。

以下是您可以尝试的一些事项：

+   尝试在[*constants.cc*](https://oreil.ly/s5bdg)中调整阈值，使手势更容易或更难执行（以更多的误报或漏报为代价）。

+   在您的计算机上编写一个程序，让您可以使用物理手势执行任务。

+   扩展程序以通过蓝牙传输检测结果。在[Ambiq SDK](https://oreil.ly/Bci3a)中有一个如何做到这一点的示例，在`AmbiqSuite-Rel2.0.0/boards/apollo3_evb/examples/uart_ble_bridge`中。构建魔法棒应用程序时，SDK 将下载到`tensorflow/tensorflow/lite/micro/tools/make/downloads/AmbiqSuite-Rel2.0.0`中。

# 总结

在本章中，您看到了一个有趣的例子，即嵌入式机器学习应用程序如何将模糊的传感器数据解释为更有用的形式。通过观察噪声中的模式，嵌入式机器学习模型使设备能够理解周围的世界并警示我们事件，即使原始数据对人类来说可能很难消化。

在第十二章中，我们探讨了我们的魔法棒模型是如何工作的，并学习如何收集数据并训练我们自己的魔法咒语。

¹ Mihaela Porumb 等人，“一种卷积神经网络方法来检测充血性心力衰竭。”*生物医学信号处理和控制*（2020 年 1 月）。[*https://oreil.ly/4HBFt*](https://oreil.ly/4HBFt)
