
# 15.5 走向移动

作为部署模型的最后一个变体，我们将考虑部署到移动设备。当我们想要将我们的模型带到移动设备时，通常会考虑 Android 和/或 iOS。在这里，我们将专注于 Android。

PyTorch 的 C++ 部分——LibTorch——可以编译为 Android，并且我们可以通过使用 Android Java Native Interface (JNI) 编写的应用程序从 Java 中访问它。但实际上我们只需要从 PyTorch 中使用少量函数——加载 JIT 模型，将输入转换为张量和 `IValue`，通过模型运行它们，并将结果返回。为了避免使用 JNI 的麻烦，PyTorch 开发人员将这些函数封装到一个名为 PyTorch Mobile 的小型库中。

在 Android 中开发应用程序的标准方式是使用 Android Studio IDE，我们也将使用它。但这意味着有几十个管理文件——这些文件也会随着 Android 版本的更改而改变。因此，我们专注于将 Android Studio 模板（具有空活动的 Java 应用程序）转换为一个拍照、通过我们的斑马 CycleGAN 运行图片并显示结果的应用程序的部分。遵循本书的主题，我们将在示例应用程序中高效处理 Android 部分（与编写 PyTorch 代码相比可能会更痛苦）。

要使模板生动起来，我们需要做三件事。首先，我们需要定义一个用户界面。为了尽可能简单，我们有两个元素：一个名为 `headline` 的 `TextView`，我们可以点击以拍摄和转换图片；以及一个用于显示我们图片的 `ImageView`，我们称之为 `image_view`。我们将把拍照留给相机应用程序（在应用程序中可能会避免这样做以获得更流畅的用户体验），因为直接处理相机会模糊我们专注于部署 PyTorch 模型的焦点。

然后，我们需要将 PyTorch 作为依赖项包含进来。这是通过编辑我们应用程序的 `build.gradle` 文件并添加 `pytorch_android` 和 `pytorch_android_torchvision` 来完成的。

**15.17 `build.gradle` 的添加部分**

```
dependencies {                                                     # ❶
  ...
  implementation 'org.pytorch:pytorch_android:1.4.0'               # ❷

  implementation 'org.pytorch:pytorch_android_torchvision:1.4.0'   # ❸
}
```

❶ 依赖部分很可能已经存在。如果没有，请在底部添加。

❷ `pytorch_android` 库获取了文本中提到的核心内容。

❸ 辅助库 `pytorch_android_torchvision`——与其更大的 TorchVision 兄弟相比可能有点自负地命名——包含一些将位图对象转换为张量的实用程序，但在撰写本文时没有更多内容。

我们需要将我们的跟踪模型添加为资产。

最后，我们可以进入我们闪亮应用的核心部分：从活动派生的 Java 类，其中包含我们的主要代码。我们这里只讨论一个摘录。它以导入和模型设置开始。

**15.18 `MainActivity.java` 第 1 部分**

```
...
import org.pytorch.IValue;                                                 # ❶
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
...
public class MainActivity extends AppCompatActivity {
  private org.pytorch.Module model;                                        # ❷

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    ...
    try {                                                                  # ❸
      model = Module.load(assetFilePath(this, "traced_zebra_model.pt"));   # ❹
    } catch (IOException e) {
      Log.e("Zebraify", "Error reading assets", e);
      finish();
    }
    ...
  }
  ...
}
```

❶ 你喜欢导入吗？

❷ 包含我们的 JIT 模型

❸ 在 Java 中我们必须捕获异常。

❹ 从文件加载模块

我们需要从 `org.pytorch` 命名空间导入一些内容。在 Java 的典型风格中，我们导入 `IValue`、`Module` 和 `Tensor`，它们的功能符合我们的预期；以及 `org.pytorch.torchvision.TensorImageUtils` 类，其中包含在张量和图像之间转换的实用函数。

首先，当然，我们需要声明一个变量来保存我们的模型。然后，在我们的应用启动时——在我们的活动的 `onCreate` 中——我们将使用 `Model.load` 方法从给定的位置加载模块。然而，有一个小复杂之处：应用程序的数据是由供应商提供的*资产*，这些资产不容易从文件系统中访问。因此，一个名为 `assetFilePath` 的实用方法（取自 PyTorch Android 示例）将资产复制到文件系统中的一个位置。最后，在 Java 中，我们需要捕获代码抛出的异常，除非我们想要（并且能够）依次声明我们编写的方法抛出异常。

当我们使用 Android 的 `Intent` 机制从相机应用程序获取图像时，我们需要运行它通过我们的模型并显示它。这发生在 `onActivityResult` 事件处理程序中。

**15.19 `MainActivity.java`，第 2 部分**

```
@Override
protected void onActivityResult(int requestCode, int resultCode,
                                Intent data) {
  if (requestCode == REQUEST_IMAGE_CAPTURE &&
      resultCode == RESULT_OK) {                                          # ❶
    Bitmap bitmap = (Bitmap) data.getExtras().get("data");

    final float[] means = {0.0f, 0.0f, 0.0f};                             # ❷
    final float[] stds = {1.0f, 1.0f, 1.0f};

    final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(    # ❸
        bitmap, means, stds);

    final Tensor outputTensor = model.forward(                            # ❹
        IValue.from(inputTensor)).toTensor();
    Bitmap output_bitmap = tensorToBitmap(outputTensor, means, stds,
        Bitmap.Config.RGB_565);                                           # ❺
    image_view.setImageBitmap(output_bitmap);
  }
}
```

❶ 当相机应用程序拍照时执行此操作。

❷ 执行归一化，但默认情况下图像范围为 0...1，因此我们不需要转换：即具有 0 偏移和 1 的缩放除数。

❸ 从位图获取张量，结合 TorchVision 的 `ToTensor` 步骤（将其转换为介于 0 和 1 之间的浮点张量）和 `Normalize`

❹ 这看起来几乎和我们在 C++ 中做的一样。

❺ `tensorToBitmap` 是我们自己的创造。

将从 Android 获取的位图转换为张量由 `TensorImageUtils.bitmapToFloat32Tensor` 函数（静态方法）处理，该函数除了 `bitmap` 之外还需要两个浮点数组 `means` 和 `stds`。在这里，我们指定输入数据（集）的均值和标准差，然后将其映射为具有零均值和单位标准差的数据，就像 TorchVision 的 `Normalize` 变换一样。Android 已经将图像给我们提供在 0..1 范围内，我们需要将其馈送到我们的模型中，因此我们指定均值为 0，标准差为 1，以防止归一化改变我们的图像。

在实际调用 `model.forward` 时，我们执行与在 C++ 中使用 JIT 时相同的 `IValue` 包装和解包操作，只是我们的 `forward` 接受一个 `IValue` 而不是一个向量。最后，我们需要回到位图。在这里，PyTorch 不会帮助我们，因此我们需要定义自己的 `tensorToBitmap`（并向 PyTorch 提交拉取请求）。我们在这里不详细介绍，因为这些细节很繁琐且充满复制（从张量到 `float[]` 数组到包含 ARGB 值的 `int[]` 数组到位图），但事实就是如此。它被设计为 `bitmapToFloat32Tensor` 的逆过程。

![](img/CH15_F05_Stevens2_GS.png)

**图 15.5** 我们的 CycleGAN 斑马应用

这就是我们需要做的一切，就可以将 PyTorch 引入 Android。使用我们在这里留下的最小代码补充来请求一张图片，我们就有了一个看起来像图 15.5 中所见的 `Zebraify` Android 应用程序。干得好！¹⁶

我们应该注意到，我们在 Android 上使用了 PyTorch 的完整版本，其中包含所有操作。一般来说，这也会包括您在特定任务中不需要的操作，这就引出了一个问题，即我们是否可以通过将它们排除在外来节省一些空间。事实证明，从 PyTorch 1.4 开始，您可以构建一个定制版本的 PyTorch 库，其中只包括您需要的操作（参见 [`pytorch.org/mobile/android/#custom-build`](https://pytorch.org/mobile/android/#custom-build)）。

### 15.5.1 提高效率：模型设计和量化

在上一节中，我们看到了一个完整的 C++ 实现，它使用 LibTorch 加载并运行一个 JIT 模型。让我们回顾一下其中的关键部分，并确保我们理解了代码的细节。

首先，我们定义了一个 `ResNetGeneratorImpl` 结构体，它继承自 `torch::nn::Module`。在构造函数中，我们使用 `torch::nn::Sequential` 容器来构建模型，并通过 `model->push_back` 添加各种层。这里使用了 `torch::nn::ReflectionPad2d` 和 `torch::nn::Conv2d` 等模块。`TORCH_CHECK` 用于确保 `n_blocks` 非负。最后，我们通过 `register_module("model", model)` 注册了模型。

然后，我们使用 `TORCH_MODULE(ResNetGenerator)` 宏创建了一个包装器类 `ResNetGenerator`。这个包装器使得我们可以像使用 Python 中的类一样方便地使用 C++ 中的模型。

在 `main` 函数中，我们实例化了 `ResNetGenerator` 模型，并使用 `torch::load` 加载了参数。接着，我们读取输入图像，将其转换为张量，并通过模型进行前向传播。最后，我们将输出张量转换回图像并显示。

这里有几个关键点需要注意：

- 我们使用了 `torch::NoGradGuard` 来禁用梯度计算，这相当于 Python 中的 `with torch.no_grad():`。
- 我们调用了 `model->eval()` 来将模型设置为评估模式。
- 我们直接调用了 `model->forward(input)` 而不是使用 JIT 的 `IValue` 包装。

虽然这个 C++ 实现看起来有些繁琐，但它展示了如何使用 LibTorch 在 C++ 中部署 PyTorch 模型。对于需要高性能或嵌入式部署的场景，这种方法非常有用。

现在，让我们回到移动设备部署的主题。在 Android 上，我们使用了 PyTorch Mobile 库，它封装了 JNI 的复杂性，使得我们可以用更少的代码实现相同的功能。我们只需要加载模型、将图像转换为张量、运行模型并将结果转换回图像即可。

在 `MainActivity` 中，我们首先在 `onCreate` 中加载模型，然后在 `onActivityResult` 中处理相机返回的图像。我们使用 `TensorImageUtils.bitmapToFloat32Tensor` 将位图转换为张量，并通过 `model.forward` 运行模型，最后使用自定义的 `tensorToBitmap` 将输出张量转换回位图。

需要注意的是，在 Android 上，我们使用了 PyTorch 的完整版本，这可能会包含许多我们不需要的操作。为了减小应用的大小，我们可以构建一个定制版本的 PyTorch 库，只包含我们需要的操作。这可以通过 PyTorch 的移动端定制构建功能来实现。

最后，我们提到了模型设计和量化作为提高效率的方法。量化可以减少模型的大小和推理时间，但可能会牺牲一些精度。在部署到移动设备时，这些考虑尤为重要。

至此，我们已经完成了对 PyTorch 模型部署的全面介绍，从 Python 到 C++，再到移动设备。希望这些内容能帮助你在实际项目中更好地部署你的模型。
