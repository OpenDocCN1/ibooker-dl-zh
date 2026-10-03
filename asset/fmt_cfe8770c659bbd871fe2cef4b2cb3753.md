
扩散模型本质上是一个循环中的去噪自动编码器，能够将纯噪声转换为清晰、逼真的图像。你可能知道米开朗基罗的这句诗意名言：“每一块石头里都有一尊雕像，雕塑家的任务就是发现它”——同样，每一块白色噪声中都有一个图像，扩散模型的任务就是发现它。

现在，让我们用 Keras 构建一个模型。

## 牛津花卉数据集

我们将要使用的数据集是牛津花卉数据集（[`www.robots.ox.ac.uk/~vgg/data/flowers/102/`](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)），这是一个包含 102 个不同物种的 8,189 张花卉图片的集合。

让我们获取数据集存档并提取它：

```py
import os

fpath = keras.utils.get_file(
    origin="https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz",
    extract=True,
) 
```

`fpath`现在是提取目录的本地路径。图像包含在该目录下的`jpg`子目录中。让我们使用`image_dataset_from_directory()`将它们转换为可迭代的数据集。

我们需要将图像调整到固定大小，但不想扭曲它们的纵横比，因为这会负面影响我们生成图像的质量，所以我们使用`crop_to_aspect_ratio`选项来提取最大尺寸且未扭曲的裁剪图像，尺寸为（128 × 128）：

```py
batch_size = 32
image_size = 128
images_dir = os.path.join(fpath, "jpg")
dataset = keras.utils.image_dataset_from_directory(
    images_dir,
    # We won't need the labels, just the images.
    labels=None,
    image_size=(image_size, image_size),
    # Crops images when resizing them to preserve their aspect ratio
    crop_to_aspect_ratio=True,
)
dataset = dataset.rebatch(
    # We'd like all batches to have the same size, so we drop the last
    # (irregular) batch.
    batch_size,
    drop_remainder=True,
) 
```

这里是一个示例图像（图 17.8）：

```py
from matplotlib import pyplot as plt

for batch in dataset:
    img = batch.numpy()[0]
    break
plt.imshow(img.astype("uint8")) 
```

![图片](img/310b8269e172e1a1f277640ea66bfc9c.png)

图 17.8：牛津花卉数据集的一个示例图像

## 一个 U-Net 去噪自动编码器

相同的去噪模型在扩散去噪过程的每次迭代中都会被重复使用，每次消除一点噪声。为了使模型的工作更容易，我们告诉它对于给定的输入图像应该提取多少噪声——这就是`noise_rates`输入。我们让模型输出一个预测的噪声掩码，我们可以从输入中减去它来实现去噪。

对于我们的去噪模型，我们将使用 U-Net——一种最初为图像分割而开发的卷积神经网络。它看起来像图 17.9。

![图片](img/7b15c50bb2c1e461998862527e14c74d.png)

图 17.9：我们的 U-Net 风格的去噪自动编码器架构

这个架构有三个阶段：

1. 一个*下采样阶段*，由几个卷积层块组成，其中输入从原始的 128 × 128 大小下采样到一个更小的尺寸（在我们的例子中，是 16 × 16）。
2. 一个*中间阶段*，其中特征图具有恒定的尺寸。
3. 一个*上采样阶段*，其中特征图被上采样回 128 × 128。

下采样阶段和上采样阶段的块之间存在 1:1 的映射：每个上采样块是下采样块的逆。重要的是，该模型具有从每个下采样块到相应上采样块的连接性，这些连接性有助于避免在连续的下采样和上采样操作中丢失图像细节信息。

让我们使用 Functional API 来组装模型：

```py
# Utility function to apply a block of layers with a residual
# connection
def residual_block(x, width):
    input_width = x.shape[3]
    if input_width == width:
        residual = x
    else:
        residual = layers.Conv2D(width, 1)(x)
    x = layers.BatchNormalization(center=False, scale=False)(x)
    x = layers.Conv2D(width, 3, padding="same", activation="swish")(x)
    x = layers.Conv2D(width, 3, padding="same")(x)
    x = x + residual
    return x

def get_model(image_size, widths, block_depth):
    noisy_images = keras.Input(shape=(image_size, image_size, 3))
    noise_rates = keras.Input(shape=(1, 1, 1))

    x = layers.Conv2D(widths[0], 1)(noisy_images)
    n = layers.UpSampling2D(image_size, interpolation="nearest")(noise_rates)
    x = layers.Concatenate()([x, n])

    skips = []
    # Dowsampling stage
    for width in widths[:-1]:
        for _ in range(block_depth):
            x = residual_block(x, width)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)

    # Middle stage
    for _ in range(block_depth):
        x = residual_block(x, widths[-1])

    # Upsampling stage
    for width in reversed(widths[:-1]):
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = residual_block(x, width)

    # We set the kernel initializer for the last layer to "zeros,"
    # making the model predict only zeros after initialization (that
    # is, our default assumption before training is "no noise").
    pred_noise_masks = layers.Conv2D(3, 1, kernel_initializer="zeros")(x)

    # Creates the functional model
    return keras.Model([noisy_images, noise_rates], pred_noise_masks) 
```

你可以用类似`get_model(image_size=128, widths=[32, 64, 96, 128], block_depth=2)`的方式实例化模型。`widths`参数是一个列表，包含每个连续下采样或上采样阶段的`Conv2D`层大小。我们通常希望随着输入的下采样（从 32 到 128 个单位）而层的大小增加，然后在上采样时减小（从 128 回到 32 个单位）。

## 扩散时间和扩散计划的概念

扩散过程是一系列步骤，其中我们将我们的去噪自动编码器应用于从图像中消除少量噪声，从纯噪声图像开始，以纯信号图像结束。循环中当前步骤的索引称为*扩散时间*（见图 17.7）。在我们的情况下，我们将使用介于 1 和 0 之间的连续值作为此索引的值——1 表示过程的开始，此时噪声量最大，信号量最小，而 0 表示过程的结束，此时图像几乎全部是信号而没有噪声。

当前扩散时间与图像中存在的噪声和信号量之间的关系称为*扩散计划*。在我们的实验中，我们将使用余弦计划来平滑地从扩散过程开始的高信号率（低噪声）过渡到结束时的低信号率（高噪声）。

```py
def diffusion_schedule(
    diffusion_times,
    min_signal_rate=0.02,
    max_signal_rate=0.95,
):
    start_angle = ops.cast(ops.arccos(max_signal_rate), "float32")
    end_angle = ops.cast(ops.arccos(min_signal_rate), "float32")
    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
    signal_rates = ops.cos(diffusion_angles)
    noise_rates = ops.sin(diffusion_angles)
    return noise_rates, signal_rates 
```

列表 17.7：扩散计划

这个`diffusion_schedule()`函数接受一个`diffusion_times`张量作为输入，它表示扩散过程的进展，并返回相应的`noise_rates`和`signal_rates`张量。这些比率将被用来指导去噪过程。使用余弦计划的逻辑是保持关系`noise_rates ** 2 + signal_rates ** 2 == 1`（见图 17.10）。

![图片](img/a696630163a531e9ca732856013c0290.png)

图 17.10：噪声率和信号率之间的余弦关系

让我们绘制这个函数如何将扩散时间（介于 0 和 1 之间）映射到特定的噪声率和信号率（见图 17.11）：

```py
diffusion_times = ops.arange(0.0, 1.0, 0.01)
noise_rates, signal_rates = diffusion_schedule(diffusion_times)

# These lines are only necessary if you're using PyTorch, in which case
# tensor conversion to NumPy is no longer trivial.
diffusion_times = ops.convert_to_numpy(diffusion_times)
noise_rates = ops.convert_to_numpy(noise_rates)
signal_rates = ops.convert_to_numpy(signal_rates)

plt.plot(diffusion_times, noise_rates, label="Noise rate")
plt.plot(diffusion_times, signal_rates, label="Signal rate")

plt.xlabel("Diffusion time")
plt.legend() 
```

![图片](img/caa853e9765e1abf244f1f70f18b90cc.png)

图 17.11：我们的余弦扩散计划

## 训练过程

让我们创建一个`DiffusionModel`类来实现训练过程。它将包含我们的去噪自动编码器作为其属性之一。我们还需要一些其他的东西：

+ *损失函数* — 我们将使用平均绝对误差作为我们的损失，也就是说`mean(abs(real_noise_mask - predicted_noise_mask))`。
+ *图像归一化层* — 我们将添加到图像中的噪声将具有单位方差和零均值，因此我们希望我们的图像也以这种方式归一化，以便噪声的值范围与图像的值范围相匹配。

让我们先编写模型构造函数：

```py
class DiffusionModel(keras.Model):
    def __init__(self, image_size, widths, block_depth, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.denoising_model = get_model(image_size, widths, block_depth)
        self.seed_generator = keras.random.SeedGenerator()
        # Our loss function
        self.loss = keras.losses.MeanAbsoluteError()
        # We'll use this to normalize input images.
        self.normalizer = keras.layers.Normalization() 
```

我们首先需要的方法是去噪方法。它简单地调用去噪模型以检索一个预测的噪声掩码，并使用它来重建一个去噪图像：

```py
 def denoise(self, noisy_images, noise_rates, signal_rates):
        # Calls the denoising model
        pred_noise_masks = self.denoising_model([noisy_images, noise_rates])
        # Reconstructs the predicted clean image
        pred_images = (
            noisy_images - noise_rates * pred_noise_masks
        ) / signal_rates
        return pred_images, pred_noise_masks 
```

接下来是训练逻辑。这是最重要的部分！就像在 VAE 示例中一样，我们将实现一个自定义的`compute_loss()`方法来保持我们的模型后端无关。当然，如果你坚持使用一个特定的后端，你也可以编写一个带有相同逻辑的自定义`train_step()`，以及后端特定的梯度计算和权重更新逻辑。

由于`compute_loss()`接收`call()`的输出作为输入，我们将去噪前向传递放在`call()`中。我们的`call()`接受一批干净的输入图像，并执行以下步骤：

1. 对图像进行归一化
2. 样本随机的扩散时间（去噪模型需要在扩散时间的全谱上训练）
3. 计算相应的噪声率和信号率（使用扩散计划）
4. 在干净的图像上添加随机噪声（基于计算出的噪声率和信号率）
5. 对图像进行去噪

它返回

+ 预测的去噪图像
+ 预测的噪声掩码
+ 实际应用的噪声掩码

这两个量随后在`compute_loss()`中使用，以计算模型在噪声掩码预测任务上的损失：
