
### 扩散模型实现

```py
def call(self, images):
    images = self.normalizer(images)
    # Samples random noise masks
    noise_masks = keras.random.normal(
        (batch_size, self.image_size, self.image_size, 3),
        seed=self.seed_generator,
    )
    # Samples random diffusion times
    diffusion_times = keras.random.uniform(
        (batch_size, 1, 1, 1),
        minval=0.0,
        maxval=1.0,
        seed=self.seed_generator,
    )
    noise_rates, signal_rates = diffusion_schedule(diffusion_times)
    # Adds noise to the images
    noisy_images = signal_rates * images + noise_rates * noise_masks
    # Denoises them
    pred_images, pred_noise_masks = self.denoise(
        noisy_images, noise_rates, signal_rates
    )
    return pred_images, pred_noise_masks, noise_masks

def compute_loss(self, x, y, y_pred, sample_weight=None, training=True):
    _, pred_noise_masks, noise_masks = y_pred
    return self.loss(noise_masks, pred_noise_masks)
```

### 生成过程

最后，让我们实现图像生成过程。我们从纯随机噪声开始，并反复应用`denoise()`方法，直到我们得到高信号、低噪声的图像。

```py
def generate(self, num_images, diffusion_steps):
    noisy_images = keras.random.normal(
        # Starts from pure noise
        (num_images, self.image_size, self.image_size, 3),
        seed=self.seed_generator,
    )
    step_size = 1.0 / diffusion_steps
    for step in range(diffusion_steps):
        # Computes appropriate noise rates and signal rates
        diffusion_times = ops.ones((num_images, 1, 1, 1)) - step * step_size
        noise_rates, signal_rates = diffusion_schedule(diffusion_times)
        # Calls denoising model
        pred_images, pred_noises = self.denoise(
            noisy_images, noise_rates, signal_rates
        )
        # Prepares noisy images for the next iteration
        next_diffusion_times = diffusion_times - step_size
        next_noise_rates, next_signal_rates = diffusion_schedule(
            next_diffusion_times
        )
        noisy_images = (
            next_signal_rates * pred_images + next_noise_rates * pred_noises
        )
    # Denormalizes images so their values fit between 0 and 255
    images = (
        self.normalizer.mean + pred_images * self.normalizer.variance**0.5
    )
    return ops.clip(images, 0.0, 255.0)
```

### 使用自定义回调可视化结果

我们没有合适的度量标准来评判我们生成的图像质量，所以你将需要在训练过程中自己可视化生成的图像来判断你的模型是否有所进展。一个简单的方法是使用自定义回调。以下回调在每一个 epoch 结束时使用`generate()`方法来显示一个 3 × 6 的生成图像网格：

```py
class VisualizationCallback(keras.callbacks.Callback):
    def __init__(self, diffusion_steps=20, num_rows=3, num_cols=6):
        self.diffusion_steps = diffusion_steps
        self.num_rows = num_rows
        self.num_cols = num_cols

    def on_epoch_end(self, epoch=None, logs=None):
        generated_images = self.model.generate(
            num_images=self.num_rows * self.num_cols,
            diffusion_steps=self.diffusion_steps,
        )

        plt.figure(figsize=(self.num_cols * 2.0, self.num_rows * 2.0))
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                i = row * self.num_cols + col
                plt.subplot(self.num_rows, self.num_cols, i + 1)
                img = ops.convert_to_numpy(generated_images[i]).astype("uint8")
                plt.imshow(img)
                plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()
```

### 是时候开始了！

终于到了在牛津花卉数据集上训练我们的扩散模型的时候了。让我们实例化模型：

```py
model = DiffusionModel(image_size, widths=[32, 64, 96, 128], block_depth=2)
# Computes the mean and variance necessary to perform normalization —
# don't forget it!
model.normalizer.adapt(dataset)
```

我们将使用`AdamW`作为我们的优化器，并启用一些实用的选项来帮助稳定训练并提高生成图像的质量：

+   *学习率衰减* — 我们在训练过程中通过`InverseTimeDecay`计划逐渐降低学习率。

+   *模型权重的指数移动平均*——也称为 Polyak 平均。这种技术在整个训练过程中维护模型权重的运行平均值。每 100 个批次，我们用这个平均权重量覆模型的权重。这有助于在损失景观嘈杂的情况下稳定模型的表示。

代码如下：

```py
model.compile(
    optimizer=keras.optimizers.AdamW(
        # Configures the learning rate decay schedule
        learning_rate=keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=1e-3,
            decay_steps=1000,
            decay_rate=0.1,
        ),
        # Turns on Polyak averaging
        use_ema=True,
        # Configures how often to overwrite the model's weights with
        # their exponential moving average
        ema_overwrite_frequency=100,
    ),
)
```

让我们拟合模型。我们将使用`VisualizationCallback`回调在每个 epoch 后绘制生成图像的示例，并使用`ModelCheckpoint`回调保存模型的权重：

```py
model.fit(
    dataset,
    epochs=100,
    callbacks=[
        VisualizationCallback(),
        keras.callbacks.ModelCheckpoint(
            filepath="diffusion_model.weights.h5",
            save_weights_only=True,
            save_best_only=True,
        ),
    ],
)
```

如果你正在 Colab 上运行，你可能会遇到错误：“缓冲数据在达到输出大小限制后截断。” 这是因为`fit()`的日志包括图像，它们占用了大量空间，而单个笔记本单元允许的输出是有限的。为了解决这个问题，你可以在五个连续的单元中简单地链式调用五个`model.fit(..., epochs=20)`，这相当于一个单独的`fit(..., epochs=100)`调用。

经过 100 个 epoch（在 T4 上大约需要 90 分钟，免费的 Colab GPU），我们得到了像这样的相当生成性的花朵（见图 17.12）。

![](img/c291aa0a0eb3da4b07c4f02d9212c102.png)

图 17.12：生成的花朵示例

你可以继续训练更长的时间，并得到越来越逼真的结果。

所以这就是扩散图像生成的工作原理！现在，下一步是解锁它们的潜力，添加*文本条件*，这将导致一个文本到图像模型，能够生成与给定文本标题匹配的图像。

## 文本到图像模型

我们可以使用相同的基本扩散过程来创建一个将文本输入映射到图像输出的模型。为此，我们需要一个预训练的文本编码器（例如第十五章中的 RoBERTa 这样的 transformer 编码器），它可以映射文本到连续嵌入空间中的向量。然后我们可以在`(prompt, image)`对上训练扩散模型，其中每个提示是输入图像的简短文本描述。

我们可以以前同样的方式处理图像输入，将噪声输入映射到去噪输出，逐渐接近我们的输入图像。关键的是，我们可以通过也将嵌入的文本提示传递给去噪模型来扩展这个设置。因此，我们的去噪模型不仅接受`noisy_images`输入，还将接受两个输入：`noisy_images`和`text_embeddings`。这使我们训练的先前花朵去噪器有了优势。模型不是在没有额外信息的情况下学习从图像中去除噪声，而是可以使用最终图像的文本表示来帮助指导去噪过程。

训练完成后，事情会变得更有趣。因为我们已经训练了一个可以将纯噪声映射到某些文本的向量表示的图像的条件上的模型，现在我们可以输入纯噪声和从未见过的提示，并将其去噪成图像。

让我们试试这个。在这本书中，我们实际上不会从头开始训练这样的模型——你已经有了所有需要的成分，但训练一个效果良好的文本到图像扩散模型既昂贵又耗时。相反，我们将使用 KerasHub 中一个流行的预训练模型 Stable Diffusion（图 17.13）。Stable Diffusion 是由一家名为 Stability AI 的公司制作的，该公司专门制作图像和视频生成的开源模型。我们只需几行代码就可以在 KerasHub 中使用他们图像生成模型的第三个版本：

```py
import keras_hub

height, width = 512, 512
task = keras_hub.models.TextToImage.from_preset(
    "stable_diffusion_3_medium",
    image_shape=(height, width, 3),
    # A trick to keep memory usage down. More details in chapter 18.
    dtype="float16",
)
prompt = "A NASA astraunaut riding an origami elephant in New York City"
task.generate(prompt)
```

列表 17.8：创建一个稳定扩散文本到图像模型

![](img/f8ec280f2a2008a60812cf66aa7a89c8.png)

图 17.13：我们的稳定扩散模型的一个示例输出

与我们在上一章中介绍的`CausalLM`任务类似，`TextToImage`任务是一个高级类，用于根据文本输入进行图像生成。它将标记化和扩散过程封装成一个高级的生成调用。

稳定扩散模型实际上为其模型添加了一个第二个“负提示”，可以用来将扩散过程引导远离某些文本输入。这里没有什么魔法。要添加一个负提示，你可以简单地训练一个三元组模型：`(image, positive_prompt, negative_prompt)`，其中正提示是对图像的描述，而负提示是一系列不描述图像的词语。通过将正负文本嵌入输入到去噪器中，去噪器将学习引导噪声向与正提示匹配的图像，并远离与负提示匹配的图像（图 17.14）。让我们尝试从我们的输入中移除颜色蓝色：

```py
task.generate(
    {
        "prompts": prompt,
        "negative_prompts": "blue color",
    }
)
```

![](img/fbd61fbad1abb2a0e1862fae44563a25.png)

图 17.14：使用负提示将模型引导远离蓝色

与我们在上一章中使用的文本模型的`generate()`方法类似，我们有一些额外的参数可以传递给模型以控制生成过程。让我们尝试向我们的模型传递一个可变的扩散步骤数，以查看去噪过程（图 17.15）的实际操作：

```py
import numpy as np
from PIL import Image

def display(images):
    return Image.fromarray(np.concatenate(images, axis=1))

display([task.generate(prompt, num_steps=x) for x in [5, 10, 15, 20, 25]])
```

![](img/aba6739b002e9fa379c845107a1d25a8.png)

图 17.15：控制扩散步骤的数量

### 探索文本到图像模型的潜在空间

没有比文本扩散模型更好地看到深度神经网络插值性质的方法了。我们模型使用的文本编码器将学习一个平滑、低维流形来表示我们的输入提示。它是连续的，这意味着我们学习了一个空间，我们可以从一个提示的文本表示走到另一个提示的文本表示，每个中间点都将具有语义意义。我们可以将这一点与我们的扩散过程相结合，通过简单地用文本提示描述每个端状态来在两个图像之间进行形态变化。

在我们能够做到这一点之前，我们需要将我们的高级`generate()`函数分解成其组成部分。让我们试试看。
