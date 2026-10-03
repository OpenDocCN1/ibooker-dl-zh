
```py
class DCGAN(models.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(DCGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer):
        super(DCGAN, self).compile()
        self.loss_fn = losses.BinaryCrossentropy() # ①
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_metric = metrics.Mean(name="d_loss")
        self.g_loss_metric = metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim)
        ) # ②

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(
                random_latent_vectors, training = True
            ) # ③
            real_predictions = self.discriminator(real_images, training = True) # ④
            fake_predictions = self.discriminator(
                generated_images, training = True
            ) # ⑤

            real_labels = tf.ones_like(real_predictions)
            real_noisy_labels = real_labels + 0.1 * tf.random.uniform(
                tf.shape(real_predictions)
            )
            fake_labels = tf.zeros_like(fake_predictions)
            fake_noisy_labels = fake_labels - 0.1 * tf.random.uniform(
                tf.shape(fake_predictions)
            )

            d_real_loss = self.loss_fn(real_noisy_labels, real_predictions)
            d_fake_loss = self.loss_fn(fake_noisy_labels, fake_predictions)
            d_loss = (d_real_loss + d_fake_loss) / 2.0 # ⑥

            g_loss = self.loss_fn(real_labels, fake_predictions) # ⑦

        gradients_of_discriminator = disc_tape.gradient(
            d_loss, self.discriminator.trainable_variables
        )
        gradients_of_generator = gen_tape.gradient(
            g_loss, self.generator.trainable_variables
        )

        self.d_optimizer.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables)
        ) # ⑧
        self.g_optimizer.apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables)
        )

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {m.name: m.result() for m in self.metrics}

dcgan = DCGAN(
    discriminator=discriminator, generator=generator, latent_dim=100
)

dcgan.compile(
    d_optimizer=optimizers.Adam(
        learning_rate=0.0002, beta_1 = 0.5, beta_2 = 0.999
    ),
    g_optimizer=optimizers.Adam(
        learning_rate=0.0002, beta_1 = 0.5, beta_2 = 0.999
    ),
)

dcgan.fit(train, epochs=300)
```

① 生成器和鉴别器的损失函数是`BinaryCrossentropy`。

② 为了训练网络，首先从多元标准正态分布中抽取一批向量。

③ 接下来，通过生成器生成一批生成的图像。

④ 现在让鉴别器预测一批真实图像的真实性...

⑤ ...和一批生成的图像。

⑥ 鉴别器损失是真实图像（标签为 1）和假图像（标签为 0）之间的平均二元交叉熵。

⑦ 生成器损失是鉴别器对生成图像的预测与标签 1 之间的二元交叉熵。

⑧ 分别更新鉴别器和生成器的权重。

鉴别器和生成器不断争夺主导地位，这可能使 DCGAN 训练过程不稳定。理想情况下，训练过程将找到一个平衡点，使生成器能够从鉴别器那里学习有意义的信息，图像的质量将开始提高。经过足够的 epochs，鉴别器往往最终占据主导地位，如图 4-6 所示，但这可能不是问题，因为生成器可能已经学会在这一点上生成足够高质量的图像。

![](img/gdl2_0406.png)

###### 图 4-6。训练过程中鉴别器和生成器的损失和准确率

# 向标签添加噪声

在训练 GAN 时的一个有用技巧是向训练标签添加少量随机噪声。这有助于改善训练过程的稳定性并增强生成的图像。这种*标签平滑*作为一种驯服鉴别器的方式，使其面临更具挑战性的任务，不会压倒生成器。

## DCGAN 的分析

通过观察训练过程中特定时期生成器生成的图像（图 4-7），可以清楚地看到生成器越来越擅长生成可能来自训练集的图像。

![](img/gdl2_0407.png)

###### 图 4-7。训练过程中特定时期生成器的输出

神奇的是神经网络能够将随机噪声转换为有意义的东西。值得记住的是，我们除了原始像素之外没有提供模型任何额外的特征，因此它必须自行解决如何绘制阴影、立方体和圆等高级概念。

成功生成模型的另一个要求是它不仅仅是复制训练集中的图像。为了测试这一点，我们可以找到训练集中与特定生成示例最接近的图像。一个好的距离度量是*L1 距离*，定义为：

```py
def compare_images(img1, img2):
    return np.mean(np.abs(img1 - img2))
```

图 4-8 显示了一些生成图像在训练集中最接近的观察结果。我们可以看到，虽然生成图像与训练集之间存在一定程度的相似性，但它们并不完全相同。这表明生成器已经理解了这些高级特征，并且能够生成与已经看到的图像不同的示例。

![](img/gdl2_0408.png)

###### 图 4-8\. 从训练集中生成图像的最接近匹配

## GAN 训练：技巧和窍门

虽然 GAN 是生成建模的重大突破，但训练起来也非常困难。在本节中，我们将探讨训练 GAN 时遇到的一些最常见问题和挑战，以及潜在的解决方案。在下一节中，我们将看一些更基本的调整 GAN 框架的方法，以解决许多这些问题。

### 鉴别器压倒生成器

如果鉴别器变得过于强大，损失函数的信号变得太弱，无法驱动生成器中的任何有意义的改进。在最坏的情况下，鉴别器完全学会区分真实图像和假图像，梯度完全消失，导致没有任何训练，如图 4-9 所示。

![](img/gdl2_0409.png)

###### 图 4-9\. 当鉴别器压倒生成器时的示例输出

如果发现鉴别器损失函数坍缩，需要找到削弱鉴别器的方法。尝试以下建议：

+   增加鉴别器中`Dropout`层的`rate`参数，以减少通过网络的信息量。

+   降低鉴别器的学习率。

+   减少鉴别器中的卷积滤波器数量。

+   在训练鉴别器时向标签添加噪音。

+   在训练鉴别器时，随机翻转一些图像的标签。

### 生成器压倒鉴别器

如果鉴别器不够强大，生成器将找到一种方法轻松欺骗鉴别器，只需少量几乎相同的图像样本。这被称为*模式坍塌*。

例如，假设我们在不更新鉴别器的情况下训练生成器多个批次。生成器会倾向于找到一个始终欺骗鉴别器的单个观察（也称为*模式*），并开始将潜在输入空间中的每个点映射到这个图像。此外，损失函数的梯度会坍缩到接近 0，因此无法从这种状态中恢复。

即使我们尝试重新训练鉴别器以阻止它被这一点欺骗，生成器也会简单地找到另一个欺骗鉴别器的模式，因为它已经对其输入麻木，因此没有多样化其输出的动机。

模式坍塌的效果可以在图 4-10 中看到。

![](img/gdl2_0410.png)

###### 图 4-10\. 当生成器压倒鉴别器时模式坍塌的示例

如果发现生成器遭受模式坍塌，可以尝试使用与前一节中列出的相反建议来加强鉴别器。此外，可以尝试降低两个网络的学习率并增加批量大小。

### 无信息损失

由于深度学习模型被编译为最小化损失函数，自然会认为生成器的损失函数越小，生成的图像质量就越好。然而，由于生成器只针对当前鉴别器进行评分，而鉴别器不断改进，我们无法比较在训练过程中不同点评估的损失函数。实际上，在图 4-6 中，生成器的损失函数随时间增加，尽管图像质量明显提高。生成器损失与图像质量之间的缺乏相关性有时使得 GAN 训练难以监控。

### 超参数

正如我们所看到的，即使是简单的 GAN，也有大量的超参数需要调整。除了鉴别器和生成器的整体架构外，还有控制批量归一化、丢弃、学习率、激活层、卷积滤波器、内核大小、步幅、批量大小和潜在空间大小的参数需要考虑。GAN 对所有这些参数的微小变化非常敏感，找到一组有效的参数通常是经过有教养的试错过程，而不是遵循一套已建立的指导方针。

这就是为什么重要理解 GAN 的内部工作原理并知道如何解释损失函数——这样你就可以识别出可能改善模型稳定性的超参数的合理调整。

### 解决 GAN 的挑战

近年来，一些关键进展大大提高了 GAN 模型的整体稳定性，并减少了一些早期列出的问题的可能性，比如模式崩溃。

在本章的其余部分，我们将研究带有梯度惩罚的 Wasserstein GAN（WGAN-GP），该模型对我们迄今为止探索的 GAN 框架进行了几个关键调整，以改善图像生成过程的稳定性和质量。

# 带有梯度惩罚的 Wasserstein GAN（WGAN-GP）

在本节中，我们将构建一个 WGAN-GP 来从我们在第三章中使用的 CelebA 数据集中生成人脸。

# 运行此示例的代码

这个示例的代码可以在书库中的*notebooks/04_gan/02_wgan_gp/wgan_gp.ipynb*中找到。

这段代码是从由 Aakash Kumar Nain 创建的优秀的[WGAN-GP 教程](https://oreil.ly/dHYbC)中改编而来，该教程可在 Keras 网站上找到。

Wasserstein GAN（WGAN）是由 Arjovsky 等人在 2017 年的一篇论文中引入的，是稳定 GAN 训练的第一步。通过一些改变，作者们能够展示如何训练具有以下两个特性的 GAN（引用自论文）：

+   一个与生成器的收敛和样本质量相关的有意义的损失度量

+   优化过程的稳定性提高

具体来说，该论文为鉴别器和生成器引入了*Wasserstein 损失函数*。使用这个损失函数而不是二元交叉熵会导致 GAN 更稳定地收敛。

在本节中，我们将定义 Wasserstein 损失函数，然后看看我们需要对模型架构和训练过程做哪些其他更改以整合我们的新损失函数。

您可以在书库中的*chapter05/wgan-gp/faces/train.ipynb*中找到完整的模型类。

## Wasserstein 损失

让我们首先回顾一下二元交叉熵损失的定义——我们目前用来训练 GAN 的函数（方程 4-1）。

##### 方程 4-1. 二元交叉熵损失

<math alttext="minus StartFraction 1 Over n EndFraction sigma-summation Underscript i equals 1 Overscript n Endscripts left-parenthesis y Subscript i Baseline log left-parenthesis p Subscript i Baseline right-parenthesis plus left-parenthesis 1 minus y Subscript i Baseline right-parenthesis log left-parenthesis 1 minus p Subscript i Baseline right-parenthesis right-parenthesis" display="block"><mrow><mo>-</mo> <mfrac><mn>1</mn> <mi>n</mi></mfrac> <munderover><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>n</mi></munderover> <mrow><mo>(</mo> <msub><mi>y</mi> <mi>i</mi></msub> <mo form="prefix">log</mo> <mrow><mo>(</mo> <msub><mi>p</mi> <mi>i</mi></msub> <mo>)</mo></mrow> <mo>+</mo> <mrow><mo>(</mo> <mn>1</mn> <mo>-</mo> <msub><mi>y</mi> <mi>i</mi></msub> <mo>)</mo></mrow> <mo form="prefix">log</mo> <mrow><mo>(</mo> <mn>1</mn> <mo>-</mo> <msub><mi>p</mi> <mi>i</mi></msub> <mo>)</mo></mrow> <mo>)</mo></mrow></mrow></math>

为了训练 GAN 鉴别器 D，我们计算了对比真实图像 x_i 的预测 p_i 与响应 y_i=1 以及对比生成图像 G(z_i)的预测 p_i 与响应 y_i=0 的损失。因此，对于 GAN 鉴别器，最小化损失函数可以写成方程 4-2 所示。

##### 方程 4-2. GAN 鉴别器损失最小化
