
# 构建 DCGAN 模型

以下代码构建了判别器模型。该模型是一个典型的分类神经网络，旨在将生成器生成的图像分类为真实或伪造。

```
def build_discriminator():
    discriminator = Sequential()                                             ❶

    discriminator.add(Conv2D(32, kernel_size=3, strides=2, 
 input_shape=(28,28,1), padding="same"))                ❷

    discriminator.add(LeakyReLU(alpha=0.2))                                  ❸

    discriminator.add(Dropout(0.25))                                         ❹

    discriminator.add(Conv2D(64, kernel_size=3, strides=2, 
 padding="same"))                                       ❺

    discriminator.add(ZeroPadding2D(padding=((0,1),(0,1))))                  ❻

    discriminator.add(BatchNormalization(momentum=0.8))                      ❼

    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.25))

    discriminator.add(Conv2D(128, kernel_size=3, strides=2, padding="same")) ❽
    discriminator.add(BatchNormalization(momentum=0.8))                      ❽
    discriminator.add(LeakyReLU(alpha=0.2))                                  ❽
    discriminator.add(Dropout(0.25))                                         ❽

    discriminator.add(Conv2D(256, kernel_size=3, strides=1, padding="same")) ❾
    discriminator.add(BatchNormalization(momentum=0.8))                      ❾
    discriminator.add(LeakyReLU(alpha=0.2))                                  ❾
    discriminator.add(Dropout(0.25))                                         ❾

    discriminator.add(Flatten())                                             ❿
    discriminator.add(Dense(1, activation='sigmoid'))                        ❿

    img = Input(shape=(28,28,1))                                             ⓫
    probability = discriminator(img)                                         ⓬

    return Model(inputs=img, outputs=probability)                            ⓭
```

❶ 实例化一个序列模型并将其命名为判别器

❷ 将卷积层添加到判别器模型中

❸ 添加一个带有泄漏 ReLU 激活函数的泄漏 ReLU 激活函数

❹ 添加一个具有 25% dropout 概率的 dropout 层

❺ 添加一个具有零填充的第二卷积层

❻ 添加一个零填充层以将维度从 7 × 7 更改为 8 × 8

❼ 添加一个批量归一化层以加快学习速度并提高准确性

❽ 添加具有批量归一化、泄漏 ReLU 和 dropout 的第三卷积层

❾ 添加具有批量归一化、泄漏 ReLU 和 dropout 的第四卷积层

❿ 将网络展平并添加具有 sigmoid 激活函数的输出密集层

⓫ 设置输入图像形状

⓬ 运行判别器模型以获取输出概率

⓭ 返回一个以图像作为输入并产生概率输出的模型

## 第 5 步：构建联合模型

如第 8.1.3 节所述，为了训练生成器，我们需要构建一个包含生成器和判别器的联合网络（图 8.21）。联合模型将噪声信号（`z`）作为输入并输出判别器的预测输出，作为伪造或真实。

![](img/8-21.png)

图 8.21 联合模型架构

记住，我们希望禁用联合模型中的判别器训练，如第 8.1.3 节中详细解释的那样。在训练生成器时，我们不希望判别器更新权重，但仍然希望将判别器模型包含在生成器训练中。因此，我们创建了一个包含两个模型的联合网络，但在联合网络中冻结了判别器模型的权重：

```
optimizer = Adam(learning_rate=0.0002, beta_1=0.5)                  ❶

discriminator = build_discriminator()                               ❷
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

discriminator.trainable = False                                     ❸

# Build the generator
generator = build_generator()                                       ❹

z = Input(shape=(100,))                                             ❺
img = generator(z)                                                  ❺

valid = discriminator(img)                                          ❻

combined = Model(inputs=z, outputs=valid)                           ❼
combined.compile(loss='binary_crossentropy', optimizer=optimizer)   ❼
```

❶ 定义优化器

❷ 构建并编译判别器

❸ 冻结判别器权重，因为我们不想在生成器训练期间训练它

❹ 构建生成器

❺ 生成器以 `latent_dim = 100` 的噪声作为输入并生成图像。

❻ 判别器将生成的图像作为输入并确定其有效性。

❼ 联合模型（堆叠的生成器和判别器）训练生成器以欺骗判别器。

## 第 6 步：构建训练函数

在训练 GAN 模型时，我们训练两个网络：判别器和我们在上一节中创建的联合网络。让我们构建一个 `train` 函数，它接受以下参数：

+   训练的轮数

+   批量大小

+   `save_interval` 来指定我们希望保存结果的频率

```
def train(epochs, batch_size=128, save_interval=50):

    valid = np.ones((batch_size, 1))                                ❶
    fake = np.zeros((batch_size, 1))                                ❶

    for epoch in range(epochs):

        ## Train Discriminator network

        idx = np.random.randint(0, X_train.shape[0], batch_size)    ❷
        imgs = X_train[idx]                                         ❷

        noise = np.random.normal(0, 1, (batch_size, 100))           ❸
        gen_imgs = generator.predict(noise)                         ❸

        d_loss_real = discriminator.train_on_batch(imgs, valid)     ❹
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)  ❹
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)             ❹

        ## Train the combined network (Generator)

        g_loss = combined.train_on_batch(noise, valid)              ❺

print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % 
      (epoch, d_loss[0], 100*d_loss[1], g_loss))                    ❻

        if epoch % save_interval == 0:                              ❼
            plot_generated_images(epoch, generator)                 ❼
```

❶ 对抗性真实值

❷ 选择随机的一半图像

❸ 样本噪声，并生成一批新图像

❹ 训练判别器（将真实分类为 1s，将生成分类为 0s）

❺ 训练生成器（希望判别器将图像误认为是真实的）

❻ 打印进度

❼ 如果在 `save_interval` 时保存生成的图像样本

在运行 `train()` 函数之前，您需要定义以下 `plot_generated_images()` 函数：

```
def plot_generated_images(epoch, generator, examples=100, dim=(10, 10), 
                          figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, latent_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image_epoch_%d.png' % epoch)
```

## 第 7 步：训练并观察结果

现在代码实现已经完成，我们准备开始 DCGAN 的训练。要训练模型，运行以下代码片段：

```
train(epochs=1000, batch_size=32, save_interval=50)
```

这将运行 1,000 个时代的训练，并且每 50 个时代保存一次图像。当您运行`train()`函数时，训练进度会像图 8.22 所示那样打印出来。

![](img/8-22.png)

图 8.22 前 16 个时代的训练进度

我自己运行了 10,000 个时代的训练。图 8.23 显示了 0、50、1,000 和 10,000 个时代后的结果。

![](img/8-23.png)

图 8.23 GAN 生成器在 0、50、1,000 和 10,000 个时代后的输出

如您在图 8.23 中看到的，在 0 个时代，图像只是随机噪声——没有模式或有意义的数据。在 50 个时代，模式开始形成。一个非常明显的模式是图像中心开始形成的亮像素，以及周围的较暗像素。这是因为训练数据中，所有形状都位于图像的中心。在训练过程的后期，在 1,000 个时代，您可以清楚地看到形状，并且可以猜测出提供给 GAN 模型的训练数据类型。快进到 10,000 个时代，您可以看到生成器已经非常擅长重新创建训练数据集中不存在的新的图像。例如，选择在这个时代创建的任何对象：比如说左上角的图像（连衣裙）。这是一个完全新的连衣裙设计，在训练数据集中不存在。GAN 模型在从训练集中学习连衣裙模式后，创建了一个全新的连衣裙设计。您可以运行更长时间的训练，或者使生成器网络更深，以获得更精细的结果。

## 在结束之前

对于这个项目，我使用了 Fashion-MNIST 数据集，因为图像非常小，是灰度图（单通道），这使得您在本地计算机上没有 GPU 的情况下进行训练时计算成本很低。Fashion-MNIST 数据集也非常干净：所有图像都居中，噪声较少，因此在启动 GAN 训练之前不需要太多预处理。这使得它成为一个很好的玩具数据集，可以用来启动您的第一个 GAN 项目。

如果您想尝试更高级的数据集，您可以将 CIFAR 作为您的下一步（[`www.cs.toronto.edu/~kriz/cifar.html`](https://www.cs.toronto.edu/~kriz/cifar.html)）或 Google 的 Quick, Draw!数据集（[`quickdraw.withgoogle.com`](https://quickdraw.withgoogle.com)），这是在撰写本文时被认为是世界上最大的涂鸦数据集。另一个更严肃的数据集是斯坦福的 Cars 数据集（[`ai.stanford.edu/~jkrause/cars/car_dataset.html`](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)），它包含超过 16,000 张 196 类汽车的图像。您可以尝试训练您的 GAN 模型为您的梦想汽车设计一个全新的设计！

## 摘要

+   GAN 从训练数据集中学习模式，并创建具有与训练集相似分布的新图像。

+   GAN 架构由两个相互竞争的深度神经网络组成。

+   生成器试图将随机噪声转换为看起来像是从原始数据集中采样的观察值。

+   判别器试图预测一个观察值是否来自原始数据集或生成器伪造的之一。

+   判别器的模型是一个典型的分类神经网络，旨在将生成器生成的图像分类为真实或伪造。

+   生成器的架构看起来像是一个倒置的 CNN，它以狭窄的输入开始，经过几次上采样，直到达到所需的大小。

+   上采样层通过重复其输入像素的每一行和每一列来调整图像尺寸。

+   为了训练 GAN，我们通过两个并行网络批量训练网络：判别器和冻结判别器权重并仅更新生成器权重的组合网络。

+   为了评估 GAN，我们主要依赖于我们对生成器创建的图像质量的观察。其他评估指标包括 Inception 得分和 Fréchet Inception 距离（FID）。

+   除了生成新图像外，GAN 还可以用于文本到照片合成、图像到图像翻译、图像超分辨率和其他许多应用。

***

1. 伊恩·J·古德费尔、让·普吉特-阿巴迪、梅赫迪·米尔扎、丁旭、大卫·沃德-法雷利、谢吉尔·奥齐尔、阿隆·库维尔和约书亚·本吉奥，“生成对抗网络，”2014 年，[`arxiv.org/abs/1406.2661`](http://arxiv.org/abs/1406.2661).

2. 亚历克·拉德福德、卢克·梅茨和索乌米特·钦塔拉，“使用深度卷积生成对抗网络的无监督表示学习，”2016 年，[`arxiv.org/abs/1511.06434`](http://arxiv.org/abs/1511.06434).

3. 蒂姆·萨利曼斯、伊恩·古德费尔、沃伊切赫·扎伦巴、维基·张、亚历克·拉德福德和`x[i]`陈。“改进 GAN 的训练技术，”2016 年，[`arxiv.org/abs/1606.03498`](http://arxiv.org/abs/1606.03498).

4. 马丁·海塞尔、胡伯特·拉姆萨乌尔、托马斯·乌特纳、伯纳德·内斯勒和塞普·霍克赖特，“通过双时间尺度更新规则训练的 GAN 收敛到局部纳什均衡，”2017 年，[`arxiv.org/abs/1706.08500`](http://arxiv.org/abs/1706.08500).

5. 康斯坦丁·施梅尔科夫、科德利亚·施密德和卡尔特克·阿拉哈里，“我的 GAN 有多好？”2018 年，[`arxiv.org/abs/1807.09499`](http://arxiv.org/abs/1807.09499).
